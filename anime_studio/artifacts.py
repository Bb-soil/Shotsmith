"""Artifact storage boundary, separate from the orchestrator.

The adapter returns an opaque remote reference. This module is the only place
that materializes files. P0 writes a tiny JSON artifact clearly marked MOCK; it
never creates an MP4 and never downloads external content.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

import httpx

from .comfy import MOCK_NOTE
from .errors import ArtifactError
from .workflow import safe_relative_subfolder, safe_remote_filename


class ArtifactStore(Protocol):
    def fetch_result(self, ref: dict[str, Any], job_id: str) -> dict[str, Any]:
        ...


class LocalArtifactStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._job_locks: dict[str, threading.Lock] = {}
        self._job_locks_guard = threading.Lock()

    def _job_lock(self, job_id: str) -> threading.Lock:
        with self._job_locks_guard:
            return self._job_locks.setdefault(job_id, threading.Lock())

    def _atomic_write(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Every writer uses a unique temp file. Two finalizers racing for the
        # same job can never share result.mock.json.tmp.
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def fetch_result(self, ref: dict[str, Any], job_id: str) -> dict[str, Any]:
        if (
            not isinstance(ref, dict)
            or ref.get("scheme") != "mock"
            or not isinstance(ref.get("content"), dict)
        ):
            raise ArtifactError(
                "P0 artifact store accepts only scheme='mock' with inline mock "
                "content; real downloads are out of scope and must not be attempted.",
                code="UNSUPPORTED_ARTIFACT_REF",
            )
        content = {
            "mock": True,
            "note": MOCK_NOTE,
            "job_id": job_id,
            "remote_ref": ref,
        }
        data = json.dumps(content, sort_keys=True, ensure_ascii=False, indent=2).encode(
            "utf-8"
        )
        relative = f"{job_id}/result.mock.json"
        path = self.root / relative
        # Windows can reject os.replace if two writers target the same final
        # path concurrently. Serialize per job in the store as well as in the
        # orchestrator finalization path.
        with self._job_lock(job_id):
            try:
                self._atomic_write(path, data)
            except OSError as exc:
                raise ArtifactError(
                    f"Failed to write mock artifact for job {job_id}: {exc}",
                    code="ARTIFACT_IO_ERROR",
                ) from exc
        digest = hashlib.sha256(data).hexdigest()
        return {
            "kind": "mock_json",
            "mock": True,
            "media_type": "application/json",
            "relative_path": relative,
            "path": str(path),
            "sha256": digest,
            "size_bytes": len(data),
            "note": MOCK_NOTE,
        }


def validate_mp4_structure(path: Path) -> None:
    """Validate bounded top-level MP4 boxes without decoding media.

    This is intentionally structural only: ftyp + moov + mdat with valid sizes.
    It is not an ffprobe/decode claim. Truncated, overflowing or missing boxes
    are rejected.
    """

    try:
        total = path.stat().st_size
        if total < 24:
            raise ArtifactError(
                "MP4 fixture is too short for top-level boxes",
                code="ARTIFACT_INVALID_MEDIA",
            )
        boxes: list[str] = []
        offset = 0
        with open(path, "rb") as handle:
            while offset < total:
                handle.seek(offset)
                header = handle.read(16)
                if len(header) < 8:
                    raise ArtifactError(
                        "MP4 top-level box header is truncated",
                        code="ARTIFACT_INVALID_MEDIA",
                    )
                size = int.from_bytes(header[0:4], "big")
                box_type = header[4:8].decode("ascii", errors="replace")
                header_size = 8
                if size == 1:
                    if len(header) < 16:
                        raise ArtifactError(
                            "MP4 extended box header is truncated",
                            code="ARTIFACT_INVALID_MEDIA",
                        )
                    size = int.from_bytes(header[8:16], "big")
                    header_size = 16
                    if size < header_size:
                        raise ArtifactError(
                            "MP4 extended box size is invalid",
                            code="ARTIFACT_INVALID_MEDIA",
                        )
                elif size == 0:
                    size = total - offset
                elif size < 8:
                    raise ArtifactError(
                        "MP4 top-level box size is invalid",
                        code="ARTIFACT_INVALID_MEDIA",
                    )
                end = offset + size
                if end > total or end <= offset:
                    raise ArtifactError(
                        "MP4 top-level box overflows/truncates the file",
                        code="ARTIFACT_INVALID_MEDIA",
                    )
                boxes.append(box_type)
                offset = end
                if size == 0:
                    break
        if not boxes or boxes[0] != "ftyp":
            raise ArtifactError(
                "MP4 must start with an ftyp box",
                code="ARTIFACT_INVALID_MEDIA",
            )
        if "moov" not in boxes or "mdat" not in boxes:
            raise ArtifactError(
                "MP4 must contain moov and mdat top-level boxes",
                code="ARTIFACT_INVALID_MEDIA",
            )
    except ArtifactError:
        raise
    except OSError as exc:
        raise ArtifactError(
            f"Cannot validate MP4 structure: {exc}", code="ARTIFACT_IO_ERROR"
        ) from exc


_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _safe_job_dir(job_id: str) -> str:
    if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
        raise ArtifactError("Unsafe job id for artifact path", code="UNSAFE_ARTIFACT_PATH")
    return job_id


class HttpArtifactStore:
    """Download a configured ComfyUI output through a trusted endpoint.

    The reference is never an arbitrary URL. Only filename/subfolder/type are
    accepted, the request is built against the configured base URL, redirects
    are rejected, and the stream is bounded by both bytes and a monotonic total
    deadline. The async client is created per operation and cancelled by
    ``wait_for``; no background request is left running after a timeout.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        base_url: str,
        token: str | None = None,
        timeout_seconds: float = 120.0,
        max_bytes: int = 100 * 1024 * 1024,
        async_transport: httpx.AsyncBaseTransport | None = None,
        allowed_types: tuple[str, ...] = ("output",),
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.allowed_types = frozenset(allowed_types)
        self._token = token
        self._async_transport = async_transport
        self._job_locks: dict[str, threading.Lock] = {}
        self._job_locks_guard = threading.Lock()

    def close(self) -> None:
        # No persistent AsyncClient is owned by this store.
        return None

    def _job_lock(self, job_id: str) -> threading.Lock:
        with self._job_locks_guard:
            return self._job_locks.setdefault(job_id, threading.Lock())

    def _cleanup(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def fetch_result(self, ref: dict[str, Any], job_id: str) -> dict[str, Any]:
        if not isinstance(ref, dict) or ref.get("scheme") not in {"comfy-http", "http"}:
            raise ArtifactError(
                "HTTP artifact store accepts only scheme='comfy-http' references.",
                code="UNSUPPORTED_ARTIFACT_REF",
            )
        try:
            filename = safe_remote_filename(ref.get("filename", ""))
            subfolder = safe_relative_subfolder(ref.get("subfolder"))
        except Exception as exc:
            raise ArtifactError(
                f"Unsafe remote artifact reference: {exc}",
                code="UNSAFE_ARTIFACT_REF",
            ) from exc
        artifact_type = ref.get("type", "output")
        if artifact_type not in self.allowed_types:
            raise ArtifactError(
                f"Remote artifact type {artifact_type!r} is not allowed",
                code="UNSAFE_ARTIFACT_REF",
            )
        if ref.get("media_type", "video/mp4") != "video/mp4":
            raise ArtifactError(
                "P1 HTTP artifact store accepts only structural video/mp4 fixtures",
                code="UNSUPPORTED_ARTIFACT_MEDIA_TYPE",
            )

        job_dir = _safe_job_dir(job_id)
        relative = f"{job_dir}/result.mp4"
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            size, digest = asyncio.run(
                self._download_async(filename, subfolder, artifact_type, tmp)
            )
            if size == 0:
                raise ArtifactError(
                    "Artifact download was empty", code="ARTIFACT_EMPTY"
                )
            validate_mp4_structure(tmp)
            with self._job_lock(job_id):
                os.replace(tmp, target)
        except ArtifactError:
            self._cleanup(tmp)
            raise
        except asyncio.TimeoutError as exc:
            self._cleanup(tmp)
            raise ArtifactError(
                "Artifact download exceeded the total time limit",
                code="ARTIFACT_TIMEOUT",
            ) from exc
        except httpx.HTTPError as exc:
            self._cleanup(tmp)
            raise ArtifactError(
                f"Artifact download failed: {type(exc).__name__}",
                code="ARTIFACT_TRANSFER_ERROR",
            ) from exc
        except OSError as exc:
            self._cleanup(tmp)
            raise ArtifactError(
                f"Artifact write failed: {exc}", code="ARTIFACT_IO_ERROR"
            ) from exc
        finally:
            self._cleanup(tmp)

        return {
            "kind": "video",
            "mock": False,
            "media_type": "video/mp4",
            "relative_path": relative,
            "path": str(target),
            "sha256": digest.hexdigest(),
            "size_bytes": size,
            "source": {
                "scheme": "comfy-http",
                "filename": filename,
                "subfolder": subfolder,
                "type": artifact_type,
            },
            "media_validation": (
                "structural_mp4_top_level_boxes_only; no decode; ffprobe=NOT_RUN"
            ),
        }

    async def _download_async(
        self,
        filename: str,
        subfolder: str,
        artifact_type: str,
        tmp: Path,
    ) -> tuple[int, "hashlib._Hash"]:
        deadline = time.monotonic() + self.timeout_seconds
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        timeout = httpx.Timeout(
            connect=min(10.0, self.timeout_seconds),
            read=self.timeout_seconds,
            write=self.timeout_seconds,
            pool=self.timeout_seconds,
        )
        digest = hashlib.sha256()
        size = 0
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            follow_redirects=False,
            transport=self._async_transport,
        ) as client:
            request = client.build_request(
                "GET",
                "/view",
                params={
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": artifact_type,
                },
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError("artifact deadline expired")
            response = await asyncio.wait_for(
                client.send(request, stream=True), timeout=remaining
            )
            try:
                if response.is_redirect or 300 <= response.status_code < 400:
                    raise ArtifactError(
                        "Artifact download redirect is not allowed",
                        code="ARTIFACT_REDIRECT_REJECTED",
                    )
                if response.status_code != 200:
                    raise ArtifactError(
                        f"Artifact download failed with HTTP {response.status_code}",
                        code="ARTIFACT_HTTP_ERROR",
                    )
                declared = response.headers.get("content-length")
                if declared is not None:
                    try:
                        declared_size = int(declared)
                    except ValueError as exc:
                        raise ArtifactError(
                            "Artifact Content-Length is malformed",
                            code="ARTIFACT_BAD_CONTENT_LENGTH",
                        ) from exc
                    if declared_size < 0 or declared_size > self.max_bytes:
                        raise ArtifactError(
                            "Artifact Content-Length exceeds the configured limit",
                            code="ARTIFACT_TOO_LARGE",
                        )

                iterator = response.aiter_bytes().__aiter__()
                with open(tmp, "wb") as handle:
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise ArtifactError(
                                "Artifact download exceeded the total time limit",
                                code="ARTIFACT_TIMEOUT",
                            )
                        try:
                            chunk = await asyncio.wait_for(
                                iterator.__anext__(), timeout=remaining
                            )
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError as exc:
                            raise ArtifactError(
                                "Artifact download exceeded the total time limit",
                                code="ARTIFACT_TIMEOUT",
                            ) from exc
                        if not chunk:
                            continue
                        size += len(chunk)
                        if size > self.max_bytes:
                            raise ArtifactError(
                                "Artifact download exceeded max_artifact_bytes",
                                code="ARTIFACT_TOO_LARGE",
                            )
                        digest.update(chunk)
                        handle.write(chunk)
                    if time.monotonic() > deadline:
                        raise ArtifactError(
                            "Artifact download exceeded the total time limit",
                            code="ARTIFACT_TIMEOUT",
                        )
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                await response.aclose()
        return size, digest


class CompositeArtifactStore:
    """Route mock and real references to their separate stores."""

    def __init__(self, mock: LocalArtifactStore, remote: HttpArtifactStore) -> None:
        self.mock = mock
        self.remote = remote

    def fetch_result(self, ref: dict[str, Any], job_id: str) -> dict[str, Any]:
        scheme = ref.get("scheme") if isinstance(ref, dict) else None
        if scheme == "mock":
            return self.mock.fetch_result(ref, job_id)
        if scheme in {"comfy-http", "http"}:
            return self.remote.fetch_result(ref, job_id)
        raise ArtifactError(
            f"Unsupported artifact reference scheme: {scheme!r}",
            code="UNSUPPORTED_ARTIFACT_REF",
        )


__all__ = [
    "ArtifactStore",
    "CompositeArtifactStore",
    "HttpArtifactStore",
    "LocalArtifactStore",
    "validate_mp4_structure",
]
