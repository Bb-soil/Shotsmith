"""Minimal localhost HTTP API for P0."""

from __future__ import annotations

import logging
import math
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .artifacts import CompositeArtifactStore, HttpArtifactStore, LocalArtifactStore
from .comfy import MockComfyTransport
from .comfy_http import ComfyHttpTransport
from .db import Database
from .errors import (
    IdempotencyConflict,
    InstanceAlreadyRunning,
    JobAlreadyTerminal,
    JobNotFound,
    P1ConfigError,
    ResultNotReady,
    StudioError,
)
from .p1_config import check_readiness
from .lock import InstanceLock
from .models import JobRequest
from .orchestrator import Orchestrator
from .settings import Settings
from .states import TERMINAL_STATUSES
from .worker import Worker


logger = logging.getLogger("anime_studio.api")


IdempotencyKeyHeader = Annotated[
    str,
    Header(
        alias="Idempotency-Key",
        min_length=8,
        max_length=128,
        description="Stable client key. Reusing it with a different payload returns 409.",
    ),
]


def _error_detail(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **extra}


def _json_safe(value: Any) -> Any:
    """Make validation details JSON-safe without retaining raw input."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _safe_validation_errors(exc: RequestValidationError) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    for error in exc.errors():
        item: dict[str, Any] = {
            "type": _json_safe(error.get("type", "validation_error")),
            "loc": _json_safe(list(error.get("loc", ()))),
            "msg": _json_safe(error.get("msg", "Validation error")),
        }
        if error.get("ctx") is not None:
            item["ctx"] = _json_safe(error["ctx"])
        # Deliberately omit error["input"]: it can contain inf/NaN or large raw
        # payloads, and JSONResponse must remain strict JSON.
        safe.append(item)
    return safe


def create_app(settings: Settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.ensure_dirs()
        lock = InstanceLock(settings.lock_path)
        try:
            lock.acquire()
        except InstanceAlreadyRunning:
            logger.error("refusing to start a second instance: %s", settings.lock_path)
            raise
        worker: Worker | None = None
        try:
            db = Database(settings.db_path)
            db.init()
            if settings.adapter_mode == "comfy-http":
                if settings.comfy_config_path is None:
                    raise P1ConfigError("comfy-http requires a P1 config path")
                if not settings.allow_live_transport:
                    raise P1ConfigError(
                        "comfy-http requires an explicit --allow-live-transport opt-in"
                    )
                readiness = check_readiness(settings.comfy_config_path)
                if not readiness.live_ready:
                    raise P1ConfigError(
                        "P1 readiness gate blocked live ComfyUI: "
                        + "; ".join(readiness.issues)
                    )
                token = (
                    os.environ.get(readiness.config.token_env)
                    if readiness.config.token_env
                    else None
                )
                if not readiness.config.loopback and not token:
                    raise P1ConfigError(
                        "Authenticated closed ComfyUI channel requires the token "
                        f"environment variable {readiness.config.token_env!r}"
                    )
                adapter = ComfyHttpTransport(
                    readiness.config,
                    readiness.workflow,
                    token=token,
                )
                artifacts = CompositeArtifactStore(
                    LocalArtifactStore(settings.artifacts_dir),
                    HttpArtifactStore(
                        settings.artifacts_dir,
                        base_url=readiness.config.base_url,
                        token=token,
                        timeout_seconds=readiness.config.artifact_timeout_seconds,
                        max_bytes=readiness.config.max_artifact_bytes,
                    ),
                )
            else:
                adapter = MockComfyTransport(db)
                artifacts = LocalArtifactStore(settings.artifacts_dir)
            orchestrator = Orchestrator(
                db,
                adapter,
                artifacts,
                lease_seconds=settings.lease_seconds,
            )
            worker = Worker(
                orchestrator,
                poll_interval=settings.poll_interval,
                lease_seconds=settings.lease_seconds,
            )
            app.state.settings = settings
            app.state.db = db
            app.state.adapter = adapter
            app.state.artifacts = artifacts
            app.state.orchestrator = orchestrator
            app.state.worker = worker
            app.state.instance_lock = lock
            if settings.enable_worker:
                worker.start()
            yield
        finally:
            worker_stopped = True
            if worker is not None:
                worker_stopped = worker.stop()
            if worker_stopped:
                artifacts_obj = getattr(app.state, "artifacts", None)
                remote_store = getattr(artifacts_obj, "remote", None)
                if remote_store is not None:
                    remote_store.close()
                lock.release()
            else:
                # Keep the OS-level lock: the worker may still be in a
                # blocking request and another process must not claim it.
                logger.error(
                    "worker did not stop within the shutdown timeout; "
                    "instance lock retained"
                )

    mode_label = "MOCK" if settings.adapter_mode == "mock" else "COMFY_HTTP"
    app = FastAPI(
        title="Anime Video Studio",
        version="0.2.0-p1-offline",
        description=(
            f"Transport mode: {mode_label}. "
            + (
                "Default MOCK: no GPU, RunPod, model or LoRA calls are made; "
                "results are not real video."
                if mode_label == "MOCK"
                else "Real ComfyUI HTTP transport is explicit opt-in and gated "
                "by a validated readiness manifest; live GPU work is not run by "
                "the offline test suite."
            )
        ),
        lifespan=lifespan,
    )

    # ------------------------------------------------------------ exceptions
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed.",
                    "errors": _safe_validation_errors(exc),
                }
            },
        )

    @app.exception_handler(JobNotFound)
    async def not_found_handler(request: Request, exc: JobNotFound) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"detail": _error_detail(exc.code, exc.message)},
        )

    @app.exception_handler(IdempotencyConflict)
    async def idempotency_handler(
        request: Request, exc: IdempotencyConflict
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "detail": _error_detail(
                    exc.code,
                    exc.message,
                    existing_job_id=exc.existing_job_id,
                )
            },
        )

    @app.exception_handler(JobAlreadyTerminal)
    async def terminal_handler(
        request: Request, exc: JobAlreadyTerminal
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "detail": _error_detail(exc.code, exc.message, status=exc.status)
            },
        )

    @app.exception_handler(ResultNotReady)
    async def result_not_ready_handler(
        request: Request, exc: ResultNotReady
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "detail": _error_detail(exc.code, exc.message, status=exc.status)
            },
        )

    @app.exception_handler(StudioError)
    async def studio_error_handler(
        request: Request, exc: StudioError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"detail": _error_detail(exc.code, exc.message)},
        )

    # ---------------------------------------------------------------- routes
    def _adapter_mode() -> str:
        return str(getattr(app.state.adapter, "mode", "UNKNOWN"))

    @app.get("/")
    def root() -> dict[str, Any]:
        mode = _adapter_mode()
        is_mock = mode == "MOCK"
        return {
            "name": "Anime Video Studio P0",
            "mode": mode,
            "mock": is_mock,
            "note": (
                "OFFLINE ONLY. Results are mock JSON, not real video."
                if is_mock
                else "REAL COMFY HTTP transport. Live GPU work is explicitly "
                "opted in and remains gated by P1 readiness."
            ),
            "docs": "/docs",
            "health": "/health",
            "endpoints": [
                "POST /jobs",
                "GET /jobs/{job_id}",
                "GET /jobs/{job_id}/events",
                "GET /jobs/{job_id}/result",
                "POST /jobs/{job_id}/cancel",
            ],
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        worker: Worker = app.state.worker
        healthy = (not app.state.settings.enable_worker) or worker.alive
        mode = _adapter_mode()
        is_mock = mode == "MOCK"
        return {
            "status": "ok" if healthy else "degraded",
            "mode": mode,
            "mock": is_mock,
            "worker_enabled": app.state.settings.enable_worker,
            "worker_alive": worker.alive,
            "worker_last_error": worker.last_error,
            "db_path": str(app.state.settings.db_path),
            "artifacts_dir": str(app.state.settings.artifacts_dir),
            "note": (
                "OFFLINE. No GPU or external calls."
                if is_mock
                else "REAL COMFY HTTP. No token or secret is exposed in health."
            ),
        }

    @app.post("/jobs", status_code=201)
    def create_job(
        payload: JobRequest,
        response: Response,
        idempotency_key: IdempotencyKeyHeader,
    ) -> dict[str, Any]:
        if not idempotency_key.strip():
            raise HTTPException(
                status_code=422,
                detail=_error_detail(
                    "VALIDATION_ERROR", "Idempotency-Key must not be blank."
                ),
            )
        orchestrator: Orchestrator = app.state.orchestrator
        job, created = orchestrator.create_job(payload, idempotency_key)
        response.status_code = 201 if created else 200
        return {"job": job, "idempotent_replay": not created}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        orchestrator: Orchestrator = app.state.orchestrator
        job = orchestrator.get_job(job_id)
        if job is None:
            raise JobNotFound(f"Unknown job: {job_id}")
        return {"job": job}

    @app.get("/jobs/{job_id}/events")
    def get_events(
        job_id: str,
        after: Annotated[int, Query(ge=0, le=9_223_372_036_854_775_807)] = 0,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> dict[str, Any]:
        orchestrator: Orchestrator = app.state.orchestrator
        return orchestrator.get_events(job_id, after=after, limit=limit)

    @app.get("/jobs/{job_id}/result")
    def get_result(job_id: str) -> dict[str, Any]:
        orchestrator: Orchestrator = app.state.orchestrator
        result = orchestrator.get_result(job_id)
        return {
            "job_id": job_id,
            "mock": bool(result.get("mock", False)),
            "mode": result.get("mode", _adapter_mode()),
            "result": result,
        }

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        orchestrator: Orchestrator = app.state.orchestrator
        job = orchestrator.request_cancel(job_id)
        if job.get("status") in TERMINAL_STATUSES and job.get("status") != "cancelled":
            # Cancellation raced with completion. Return the honest terminal
            # state instead of pretending the job was cancelled.
            return {"job": job, "cancel_result": "too_late"}
        return {"job": job, "cancel_result": "cancelled" if job.get("status") == "cancelled" else "requested"}

    return app
