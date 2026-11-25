"""FastAPI 入口：告警录入、分片查询、补报测试。"""
from __future__ import annotations

import asyncio
import dataclasses
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
import sqlite3

from .config import settings, shard_delay
from .merge import merge_latest
from .models import AlertOut, AppendRequest, Blocker, LatestResponse, ShardStat
from .shard import Shard, ShardSnapshot


@dataclasses.dataclass
class IngestOutcome:
    """一次录入任务的最终结果。

    status:
      - stored    : 本任务成功 INSERT 并已提交；
      - duplicate : 编号已在任一分片入库（本任务未写入）；
      - failed    : 明确没有提交（写入失败），可安全由后续录入继续尝试。
    """

    status: str
    error: BaseException | None = None


class ShardRegistry:
    """单实例内的告警录入协调器。

    告警编号是全局唯一的，但分片各自只有本地的 PRIMARY KEY，因此录入前
    必须跨分片查重；并发同号还必须串行。做法：

    - 每个编号同一时刻最多一个在途录入任务（``_inflight``）。后到的同号
      请求不另起写入，而是等待该任务给出确定结果：
        * 已提交 -> 409；
        * 确认未提交 -> 由等待者继续自己的录入。
    - 每个分片一把 asyncio 写锁；不同分片、不同编号互不阻塞。
    - 录入任务用独立 Task 承载，请求协程被取消（客户端断开、超时等）不会
      取消写入；即使录入任务本身被取消，也会等写入线程给出确定结果再返回，
      同号重试据此判定 409 还是可以继续录入。
    """

    def __init__(self, db_dir: str | None = None) -> None:
        self.shards = [Shard(i, db_dir) for i in range(settings.shard_count)]
        # 串行化同分片写入；快照读走独立连接，不受影响
        self._write_locks = [asyncio.Lock() for _ in self.shards]
        # 编号 -> 在途录入 Task；任务结束（含失败）后由回调移除
        self._inflight: dict[str, asyncio.Task[IngestOutcome]] = {}

    # ---- 查重与单个录入任务 ----

    def _find_shard_with_id(self, alert_id: str) -> int | None:
        for shard in self.shards:
            if shard.contains(alert_id):
                return shard.shard_id
        return None

    async def _settle(self, fut: asyncio.Future) -> None:
        """一直等到 executor 任务结束；期间的取消只记录、不放弃等待。

        返回后 fut 必已完成；结果留给调用方用 fut.result()/exception() 取。
        """
        while not fut.done():
            try:
                await asyncio.shield(fut)
            except asyncio.CancelledError:
                continue
            except Exception:
                break

    async def _resolve_after_error(self, alert_id: str, exc: BaseException) -> IngestOutcome:
        """写入出错后判定编号是否其实已提交（期间不因取消而放弃判定）。"""
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, self._find_shard_with_id, alert_id)
        await self._settle(fut)
        try:
            found = fut.result()
        except Exception:
            found = None
        if found is not None:
            return IngestOutcome("duplicate")
        return IngestOutcome("failed", exc)

    async def _write(self, shard_id: int, req: AppendRequest) -> IngestOutcome:
        """只做目标分片的写入；无论怎么被取消，都返回确定结果。

        写入一旦启动，线程必须跑完；期间持有该分片的写锁，避免与其它写入
        共用 SQLite 连接产生竞态。
        """
        loop = asyncio.get_running_loop()
        lock = self._write_locks[shard_id]
        try:
            await lock.acquire()
        except asyncio.CancelledError:
            return IngestOutcome("failed", asyncio.CancelledError())  # 未开始写
        fut: asyncio.Future | None = None
        try:
            fut = loop.run_in_executor(
                None,
                lambda: self.shards[shard_id].append(
                    req.id, req.timestamp, req.level, req.source, req.message
                ),
            )
            await asyncio.shield(fut)
            return IngestOutcome("stored")
        except asyncio.CancelledError:
            # 写入线程已启动；必须等它落定才能判断是否已提交
            await self._settle(fut)
            if fut.cancelled():
                return IngestOutcome("failed", asyncio.CancelledError())
            exc = fut.exception()
            if exc is None:
                return IngestOutcome("stored")
            if isinstance(exc, sqlite3.IntegrityError):
                return IngestOutcome("duplicate")
            return await self._resolve_after_error(req.id, exc)
        except sqlite3.IntegrityError:
            # 目标分片已有该编号（例如旧库或其它写入）
            return IngestOutcome("duplicate")
        except Exception as exc:  # noqa: BLE001
            return await self._resolve_after_error(req.id, exc)
        finally:
            lock.release()

    @staticmethod
    def _caller_cancelled() -> bool:
        """当前协程是否收到取消请求（Python 3.11+ 的 Task.cancelling）。"""
        me = asyncio.current_task()
        return me is not None and me.cancelling() > 0

    async def _ingest(self, shard_id: int, req: AppendRequest) -> IngestOutcome:
        """一次录入的完整任务：先跨分片查重，再写入目标分片。"""
        loop = asyncio.get_running_loop()
        check = loop.run_in_executor(None, self._find_shard_with_id, req.id)
        await self._settle(check)  # 查重也必须给出确定结果，不被取消打断
        try:
            found = check.result()
        except Exception as exc:  # noqa: BLE001
            return IngestOutcome("failed", exc)
        if found is not None:
            return IngestOutcome("duplicate")
        if self._caller_cancelled():
            # 写入尚未开始就被取消：确认未提交，交由同号重试继续
            return IngestOutcome("failed", asyncio.CancelledError())
        return await self._write(shard_id, req)

    def _forget(self, alert_id: str, task: asyncio.Task[IngestOutcome]) -> None:
        # 只移除自己，避免旧任务回调误删后来者注册的新任务
        if self._inflight.get(alert_id) is task:
            self._inflight.pop(alert_id, None)

    async def _await_outcome(self, task: asyncio.Task[IngestOutcome]) -> IngestOutcome:
        """等待录入任务的确定结果；区分“自己被取消”和“任务被取消”。

        - 本请求被取消：录入任务仍在后台跑，重试会 join 它，直接把取消抛出；
        - 录入任务被取消：_ingest 在取消路径上仍会等写入落定再给出结果，
          所以任务结束时要么已有 result，要么真的没有确定结果按“未提交”处理。
        """
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if self._caller_cancelled():
                raise
            if task.done() and not task.cancelled():
                return task.result()
            # 任务被取消且没有结果：写入是否提交未知，按未提交交由重试查重
            return IngestOutcome("failed", asyncio.CancelledError())
        except Exception as exc:  # 理论上 _ingest 不抛业务异常
            return IngestOutcome("failed", exc)

    async def append(self, shard_id: int, req: AppendRequest) -> dict:
        if not 0 <= shard_id < len(self.shards):
            raise HTTPException(status_code=404, detail=f"分片 {shard_id} 不存在")

        while True:
            task = self._inflight.get(req.id)
            if task is not None and task.done():
                # 结束的任务尚未被回调移除：就地清理后按“无在途”处理
                self._forget(req.id, task)
                task = None

            owner = task is None
            if owner:
                task = asyncio.create_task(self._ingest(shard_id, req))
                self._inflight[req.id] = task
                task.add_done_callback(
                    lambda t, alert_id=req.id: self._forget(alert_id, t)
                )

            outcome = await self._await_outcome(task)

            if owner:
                if outcome.status == "stored":
                    return {"stored": True, "shard": shard_id, "id": req.id}
                if outcome.status == "duplicate":
                    raise HTTPException(
                        status_code=409, detail=f"告警编号已存在: {req.id}"
                    )
                return self._raise_failed(req.id, outcome.error)

            # 不是自己的任务：等到了别人的确定结果
            if outcome.status in ("stored", "duplicate"):
                # 同号并发只有一个能落盘；落盘者不是本请求 -> 409
                raise HTTPException(
                    status_code=409, detail=f"告警编号已存在: {req.id}"
                )
            # 原任务确认未提交：继续由本请求录入（循环会重新注册任务）

    @staticmethod
    def _raise_failed(alert_id: str, exc: BaseException | None) -> dict:
        detail = f"告警 {alert_id} 写入未完成（已确认未提交），可重试"
        raise HTTPException(status_code=503, detail=detail) from exc

    def open_snapshots(self) -> list[ShardSnapshot]:
        return [s.open_snapshot() for s in self.shards]

    async def close(self) -> None:
        # 先等在途写入有确定结果，再关连接
        pending = [t for t in self._inflight.values()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for shard in self.shards:
            await asyncio.get_running_loop().run_in_executor(None, shard.close)


registry: ShardRegistry | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry
    registry = ShardRegistry()
    try:
        yield
    finally:
        assert registry is not None
        await registry.close()


app = FastAPI(
    title="分片告警查询服务",
    description=(
        "三个分片各持一份只追加 SQLite 库。查询只看开始时已提交的告警，"
        "按 (timestamp, id) 倒序合并；分片先给最新排序键再给明细，每批最多 20 条；"
        "确认前 N 条立即返回，超时返回最长有序前缀并指出阻塞分片。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "shards": settings.shard_count}


@app.post("/alerts/{shard_id}", summary="向指定分片录入（追加）一条告警")
async def append_alert(shard_id: int, req: AppendRequest) -> dict:
    assert registry is not None
    return await registry.append(shard_id, req)


@app.post("/alerts/{shard_id}/auto", summary="自动生成编号并追加一条告警（便于 /docs 补报测试）")
async def append_auto(
    shard_id: int,
    timestamp: float | None = Query(None, description="缺省取服务器当前时间"),
    level: str = Query("info"),
    source: str = Query("manual"),
    message: str = Query(""),
) -> dict:
    assert registry is not None
    req = AppendRequest(
        id=str(uuid.uuid4()),
        timestamp=time.time() if timestamp is None else timestamp,
        level=level,
        source=source,
        message=message,
    )
    return await registry.append(shard_id, req)


@app.get(
    "/alerts/latest",
    response_model=LatestResponse,
    summary="查询三个分片合并后的最近 N 条告警",
)
async def latest_alerts(
    n: int = Query(..., ge=1, le=10_000, description="需要的条数"),
    timeout: float = Query(
        ..., gt=0, le=120, description="查询时限（秒），到时未补齐则返回有序前缀"
    ),
) -> LatestResponse:
    assert registry is not None
    snapshots = registry.open_snapshots()
    try:
        result = await merge_latest(snapshots, n=n, timeout=timeout)
    finally:
        for snap in snapshots:
            snap.close()

    alerts = [
        AlertOut(
            id=row["id"],
            timestamp=row["timestamp"],
            shard=row["shard"],
            level=row["level"],
            source=row["source"],
            message=row["message"],
        )
        for row in result.alerts
    ]
    stats = [
        ShardStat(
            shard=rt.shard_id,
            visible_rows=rt.visible_rows,
            head_key=list(rt.head_key) if rt.head_key else None,
            batches_read=rt.snapshot.batches_read,
            delay_seconds=shard_delay(rt.shard_id),
        )
        for rt in result.runtimes
    ]
    return LatestResponse(
        alerts=alerts,
        count=len(alerts),
        requested_n=n,
        complete=result.complete,
        status=result.status,
        timed_out=result.status == "timeout",
        elapsed_ms=result.elapsed * 1000,
        snapshot_at=result.snapshot_at,
        shards=stats,
        blockers=[Blocker(**b) for b in result.blockers],
        batch_size=settings.batch_size,
    )
