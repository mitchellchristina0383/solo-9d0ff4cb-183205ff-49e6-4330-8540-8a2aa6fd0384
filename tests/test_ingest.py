"""单实例录入协调测试：全局同号唯一、并发、取消后重试、写锁隔离、旧库兼容。"""
from __future__ import annotations


import asyncio
import sqlite3
import threading

import pytest
from fastapi import HTTPException

from app.main import ShardRegistry
from app.models import AppendRequest


def _req(alert_id: str, ts: float = 1.0) -> AppendRequest:
    return AppendRequest(id=alert_id, timestamp=ts, level="info", source="t", message="m")


def _row_count(reg: ShardRegistry, alert_id: str) -> int:
    total = 0
    for shard in reg.shards:
        conn = sqlite3.connect(shard.path)
        try:
            total += conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE id = ?", (alert_id,)
            ).fetchone()[0]
        finally:
            conn.close()
    return total


def _status(coro) -> int | None:
    """跑一个 append，返回成功为 None、HTTP 状态码为码值、取消为 -1。"""
    async def run():
        try:
            await coro
            return None
        except HTTPException as exc:
            return exc.status_code
        except asyncio.CancelledError:
            return -1

    return asyncio.run(run())


# ---------- 跨分片同号 ----------

def test_same_id_different_shards_sequential_second_409(tmp_path):
    reg = ShardRegistry(str(tmp_path))
    try:
        assert _status(reg.append(0, _req("dup-x"))) is None
        assert _status(reg.append(1, _req("dup-x"))) == 409
        assert _status(reg.append(2, _req("dup-x"))) == 409
        assert _row_count(reg, "dup-x") == 1
    finally:
        asyncio.run(reg.close())


def test_same_id_concurrent_across_shards_only_one_lands(tmp_path):
    async def scenario():
        reg = ShardRegistry(str(tmp_path))
        try:
            results = await asyncio.gather(
                reg.append(0, _req("race")),
                reg.append(1, _req("race")),
                reg.append(2, _req("race")),
                return_exceptions=True,
            )
        finally:
            await reg.close()

        codes = [
            r.status_code if isinstance(r, HTTPException) else None for r in results
        ]
        assert codes.count(None) == 1, codes
        assert codes.count(409) == 2, codes
        return codes

    asyncio.run(scenario())
    # 只有一个分片留下该编号
    total = 0
    for i in range(3):
        conn = sqlite3.connect(tmp_path / f"shard_{i}.db")
        try:
            total += conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE id='race'"
            ).fetchone()[0]
        finally:
            conn.close()
    assert total == 1


def test_different_ids_different_shards_do_not_serialize(tmp_path):
    """不同编号、不同分片应能并行写入。"""
    async def scenario():
        reg = ShardRegistry(str(tmp_path))
        try:
            await asyncio.gather(
                *(reg.append(i % 3, _req(f"par-{i}", ts=float(i) + 1)) for i in range(30))
            )
        finally:
            await reg.close()

    asyncio.run(scenario())
    reg = ShardRegistry(str(tmp_path))
    try:
        for i in range(30):
            assert _row_count(reg, f"par-{i}") == 1
    finally:
        asyncio.run(reg.close())


# ---------- 取消后重试 ----------

def test_cancel_request_mid_write_retry_waits_then_409(tmp_path):
    """客户端协程取消后，原写入仍落盘；同号重试等它结束并返回 409。"""

    async def scenario():
        reg = ShardRegistry(str(tmp_path))
        started = threading.Event()
        release = threading.Event()
        orig = reg.shards[0].append

        def slow_append(*args, **kwargs):
            started.set()
            assert release.wait(10), "测试未放行写入"
            return orig(*args, **kwargs)

        reg.shards[0].append = slow_append  # 实例属性覆盖，便于控制时序
        try:
            req = _req("cancel-commit", ts=2.0)
            first = asyncio.create_task(reg.append(0, req))
            # 等执行线程真正开始，再取消请求协程（模拟客户端断开/任务取消）
            await asyncio.get_running_loop().run_in_executor(None, started.wait, 10)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

            # 写入线程还没放行：重试不能抢跑，必须等原任务结果
            retry = asyncio.create_task(reg.append(1, req))
            await asyncio.sleep(0.1)
            assert not retry.done(), "原写入未结束前，同号重试不应有结果"

            release.set()
            with pytest.raises(HTTPException) as ei:
                await retry
            assert ei.value.status_code == 409
            assert _row_count(reg, "cancel-commit") == 1
        finally:
            release.set()
            await reg.close()

    asyncio.run(scenario())


def test_cancel_while_waiting_for_shard_lock_then_retry_200(tmp_path):
    """写入尚未开始就被取消（还在等分片写锁）：确认未提交，重试可成功录入。"""
    async def scenario():
        reg = ShardRegistry(str(tmp_path))
        await reg._write_locks[0].acquire()  # 占住分片 0 的写锁
        try:
            first = asyncio.create_task(reg.append(0, _req("lock-cancel", ts=5.0)))
            await asyncio.sleep(0.05)
            assert "lock-cancel" in reg._inflight, "应已注册在途任务"
            reg._inflight["lock-cancel"].cancel()   # 服务端录入任务被取消
            await asyncio.sleep(0.05)
        finally:
            reg._write_locks[0].release()

        # owner 的请求拿到“确认未提交”结果（503），而不是挂起
        with pytest.raises(HTTPException) as ei:
            await first
        assert ei.value.status_code == 503
        assert _row_count(reg, "lock-cancel") == 0

        result = await reg.append(1, _req("lock-cancel", ts=5.0))
        assert result["stored"] is True
        assert _row_count(reg, "lock-cancel") == 1
        await reg.close()

    asyncio.run(scenario())



def test_failed_write_confirmed_uncommitted_allows_retry(tmp_path):
    """原写入明确失败（未提交），同号重试可以继续并成功录入。"""
    async def scenario():
        reg = ShardRegistry(str(tmp_path))
        orig = reg.shards[0].append

        def boom(*args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        reg.shards[0].append = boom
        try:
            with pytest.raises(HTTPException) as ei:
                await reg.append(0, _req("retry-ok", ts=3.0))
            assert ei.value.status_code == 503

            reg.shards[0].append = orig  # 恢复后重试
            result = await reg.append(0, _req("retry-ok", ts=3.0))
            assert result["stored"] is True
            assert _row_count(reg, "retry-ok") == 1
        finally:
            await reg.close()

    asyncio.run(scenario())


def test_cancel_ingest_task_mid_write_resolves_result(tmp_path):
    """录入任务本身被取消：写入已启动时必须等它落定，结果明确。"""
    async def scenario():
        reg = ShardRegistry(str(tmp_path))
        started = threading.Event()
        release = threading.Event()
        orig = reg.shards[0].append

        def slow_append(*args, **kwargs):
            started.set()
            assert release.wait(10)
            return orig(*args, **kwargs)

        reg.shards[0].append = slow_append
        try:
            req = _req("inner-cancel", ts=4.0)
            task = asyncio.create_task(reg.append(0, req))
            await asyncio.get_running_loop().run_in_executor(None, started.wait, 10)
            reg._inflight["inner-cancel"].cancel()

            # 取消后同号重试：原写入线程还没放行 -> 必须等，不得抢跑
            retry = asyncio.create_task(reg.append(1, req))
            await asyncio.sleep(0.1)
            assert not retry.done(), "原写入未结束时，同号重试不应有结果"

            release.set()
            try:
                await task
                status = None
            except HTTPException as exc:
                status = exc.status_code
            # 写入最终提交；任务明确报成功（而不是误报失败/取消）
            assert status is None
            with pytest.raises(HTTPException) as ei:
                await retry
            assert ei.value.status_code == 409
            assert _row_count(reg, "inner-cancel") == 1
        finally:
            release.set()
            await reg.close()

    asyncio.run(scenario())


# ---------- 一个分片等写锁，其它分片仍可录入 ----------

def test_busy_shard_lock_does_not_block_other_shards(tmp_path):
    async def scenario():
        reg = ShardRegistry(str(tmp_path))
        # 用外部连接在分片 0 上持有写锁（模拟 SQLite 写锁被占用）
        blocker = sqlite3.connect(reg.shards[0].path, timeout=1)
        blocker.execute("BEGIN IMMEDIATE")
        blocker.execute(
            "INSERT INTO alerts (id, timestamp, level, source, message) "
            "VALUES ('lock-holder', 0.5, 'info', 'x', 'x')"
        )
        try:
            # 分片 0 的写入会等 SQLite 写锁（异步锁 + busy_timeout）
            busy = asyncio.create_task(reg.append(0, _req("busy-0", ts=0.6)))
            await asyncio.sleep(0.2)
            assert not busy.done()

            # 其它分片录入不同编号必须很快完成
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            await asyncio.gather(
                reg.append(1, _req("free-1", ts=1.5)),
                reg.append(2, _req("free-2", ts=2.5)),
            )
            elapsed = loop.time() - t0
            assert elapsed < 2.0, f"其它分片被分片 0 的写锁拖慢: {elapsed:.2f}s"

            blocker.rollback()  # 放行分片 0
            result = await busy
            assert result["stored"] is True
        finally:
            try:
                blocker.rollback()
            except sqlite3.OperationalError:
                pass
            blocker.close()
            await reg.close()

    asyncio.run(scenario())


# ---------- 旧库 / 已存在重号 ----------

def test_legacy_db_with_duplicate_id_still_queryable_and_rejects_new(tmp_path):
    """旧库跨分片已有同号：仍可正常查询；再次录入返回 409。"""
    # 手工造旧库：同一 id 同时存在于分片 0 和 1
    for shard_id in (0, 1):
        conn = sqlite3.connect(tmp_path / f"shard_{shard_id}.db")
        try:
            conn.executescript(
                "CREATE TABLE IF NOT EXISTS alerts ("
                "id TEXT PRIMARY KEY, timestamp REAL NOT NULL, level TEXT NOT NULL,"
                "source TEXT NOT NULL, message TEXT NOT NULL);"
            )
            conn.execute(
                "INSERT OR IGNORE INTO alerts VALUES ('legacy-dup', ?, 'info', 'old', 'old')",
                (10.0 + shard_id,),
            )
            conn.commit()
        finally:
            conn.close()

    reg = ShardRegistry(str(tmp_path))
    try:
        # 录入被全局查重挡住
        assert _status(reg.append(2, _req("legacy-dup", ts=99.0))) == 409
        # 旧库两条重号照常可查（查询不做去重）
        assert _row_count(reg, "legacy-dup") == 2
        snapshots = reg.open_snapshots()
        try:
            from app.merge import merge_latest

            result = asyncio.run(merge_latest(snapshots, n=10, timeout=5))
        finally:
            for snap in snapshots:
                snap.close()
        ids = [r["id"] for r in result.alerts]
        assert ids.count("legacy-dup") == 2
    finally:
        asyncio.run(reg.close())
