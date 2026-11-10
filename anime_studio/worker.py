"""Single local executor.

One worker thread polls the durable queue. It is intentionally not a distributed
worker: the API process holds an OS-level single-instance lock. A crash leaves
the DB in a recoverable state; a restart reconciles stale dispatching jobs before
doing anything else.

The ANIME_STUDIO_FAULT environment variable is test-only fault injection used by
the subprocess crash/restart test. It is never set by the application.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid

from .errors import (
    RemoteAmbiguousSubmit,
    RemoteAuthError,
    RemoteDeadlineExpired,
    RemotePromptLost,
    RemoteProtocolError,
    RemoteTransientError,
    RemoteValidationRejected,
)
from .orchestrator import Orchestrator
from .states import JobStatus, TERMINAL_STATUSES, is_terminal
from .timeutils import now_iso, now_ms


logger = logging.getLogger("anime_studio.worker")


class Worker:
    def __init__(
        self,
        orchestrator: Orchestrator,
        *,
        poll_interval: float = 0.2,
        lease_seconds: float = 5.0,
    ) -> None:
        self.orchestrator = orchestrator
        self.poll_interval = poll_interval
        self.lease_seconds = lease_seconds
        self.worker_id = f"worker-{uuid.uuid4().hex}"
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._tick_lock = threading.Lock()
        self.last_error: str | None = None

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="anime-studio-worker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> bool:
        """Request stop and report whether the worker thread truly stopped.

        The API must not release the instance lock while this returns False.
        """

        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        if thread.is_alive():
            return False
        self._thread = None
        return True

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception as exc:  # keep the API alive and expose degraded state
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("worker tick failed")
            self._stop_event.wait(self.poll_interval)

    def tick(self) -> None:
        if not self._tick_lock.acquire(blocking=False):
            return
        try:
            # Deadline enforcement must run before recovery. An expired
            # dispatching job is reconciled for identity only and cancelled
            # best-effort; it is never submitted again.
            self.enforce_deadlines()
            self.recover_stale_dispatches()
            self.dispatch_one()
            self.poll_active()
        finally:
            self._tick_lock.release()

    # ------------------------------------------------------------- recovery
    def recover_stale_dispatches(self) -> None:
        current_ms = now_ms()
        with self.orchestrator.db.read() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                 WHERE status='dispatching'
                   AND (
                        lease_owner IS NULL
                        OR lease_owner != ?
                        OR lease_expires_at_ms IS NULL
                        OR lease_expires_at_ms <= ?
                   )
                 ORDER BY created_ms ASC
                 LIMIT 10
                """,
                (self.worker_id, current_ms),
            ).fetchall()

        for row in rows:
            job_id = row["job_id"]
            claimed = False
            with self.orchestrator.db.transaction() as conn:
                current = conn.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (job_id,)
                ).fetchone()
                if current is None or current["status"] != JobStatus.DISPATCHING.value:
                    continue
                if (
                    current["lease_owner"] == self.worker_id
                    and current["lease_expires_at_ms"] is not None
                    and int(current["lease_expires_at_ms"]) > current_ms
                ):
                    continue
                conn.execute(
                    """
                    UPDATE jobs
                       SET lease_owner=?, lease_expires_at_ms=?, updated_at=?, updated_ms=?
                     WHERE job_id=?
                    """,
                    (
                        self.worker_id,
                        current_ms + int(self.lease_seconds * 1000),
                        now_iso(),
                        current_ms,
                        job_id,
                    ),
                )
                claimed = True
            if not claimed:
                continue

            try:
                adapter = self.orchestrator.adapter
                remote = adapter.reconcile(row["correlation_key"])

                if int(row["deadline_ts"]) <= now_ms():
                    # Expired during downtime. Reconcile only identity for a
                    # best-effort cancel. Never submit a new remote job.
                    remote_state = remote.state if remote is not None else None
                    had_prompt_id = row["prompt_id"] is not None
                    cancel_attempted = False
                    if remote is not None:
                        linked = self.orchestrator.link_remote(
                            job_id,
                            remote,
                            event_type="dispatch_reconciled_at_deadline",
                            apply_result=False,
                        )
                        if linked is not None:
                            had_prompt_id = linked["prompt_id"] is not None
                            if linked["prompt_id"]:
                                cancel_attempted = True
                                try:
                                    cancelled = adapter.cancel(linked["prompt_id"])
                                    if cancelled is not None:
                                        remote_state = cancelled.state
                                except Exception:
                                    pass
                    if row["prompt_id"] and not cancel_attempted:
                        cancel_attempted = True
                        try:
                            cancelled = adapter.cancel(row["prompt_id"])
                            if cancelled is not None:
                                remote_state = cancelled.state
                        except Exception:
                            pass
                    # A reconcile may itself cross the deadline after the tick's
                    # initial enforce_deadlines(). Never submit or declare
                    # "confirmed not sent": preserve whatever identity/cancel
                    # evidence was actually observed.
                    self.orchestrator.mark_timeout(
                        job_id,
                        deadline_at=row["deadline_at"],
                        had_prompt_id=had_prompt_id,
                        cancel_attempted=cancel_attempted,
                        remote_state=remote_state,
                    )
                    continue

                if remote is not None:
                    linked = self.orchestrator.link_remote(
                        job_id, remote, event_type="dispatch_reconciled"
                    )
                    if (
                        linked is not None
                        and not is_terminal(linked["status"])
                        and linked["cancel_requested"]
                    ):
                        self.orchestrator.cancel_remote(job_id)
                    continue

                current_job = self.orchestrator.get_job(job_id)
                if current_job is None or is_terminal(current_job["status"]):
                    continue
                if current_job["cancel_requested"]:
                    if adapter.supports_reconcile_by_correlation_key:
                        # A reconciling adapter proved that no remote job exists
                        # for this correlation key, so cancellation is safe.
                        self.orchestrator.request_cancel_after_dispatch(job_id)
                    else:
                        # Crash-window cancel without a prompt_id and without
                        # reconciliation support does not prove remote absence.
                        self.orchestrator.mark_ambiguous(
                            job_id,
                            "cancellation requested in crash-window without a "
                            "prompt_id; adapter cannot reconcile by correlation "
                            "key, so remote absence/cancellation is NOT confirmed",
                        )
                    continue

                if (
                    adapter.supports_reconcile_by_correlation_key
                    and adapter.supports_idempotent_submit
                ):
                    # Re-check immediately before submit. A job that expired
                    # while this worker was down must never get a new remote job.
                    if int(row["deadline_ts"]) <= now_ms():
                        self.orchestrator.mark_timeout(
                            job_id,
                            deadline_at=row["deadline_at"],
                            had_prompt_id=False,
                            cancel_attempted=False,
                            remote_state=None,
                        )
                        continue
                    payload = json.loads(row["payload_canonical"])
                    try:
                        remote = adapter.submit(
                            row["correlation_key"],
                            payload,
                            deadline_ms=int(row["deadline_ts"]),
                        )
                    except RemoteDeadlineExpired as exc:
                        logger.warning(
                            "recovery submit deadline expired for %s", job_id
                        )
                        self.last_error = f"{exc.code}: {exc.message}"
                        self.orchestrator.mark_timeout(
                            job_id,
                            deadline_at=row["deadline_at"],
                            had_prompt_id=False,
                            cancel_attempted=False,
                            remote_state=None,
                        )
                        continue
                    if int(row["deadline_ts"]) <= now_ms():
                        # Deadline passed while submit was in flight. Do not
                        # apply a result; keep identity for best-effort cancel.
                        linked = self.orchestrator.link_remote(
                            job_id,
                            remote,
                            event_type="dispatch_reconciled_at_deadline",
                            apply_result=False,
                        )
                        cancel_attempted = False
                        if linked is not None and linked["prompt_id"]:
                            cancel_attempted = True
                            try:
                                adapter.cancel(linked["prompt_id"])
                            except Exception:
                                pass
                        self.orchestrator.mark_timeout(
                            job_id,
                            deadline_at=row["deadline_at"],
                            had_prompt_id=linked is not None
                            and linked["prompt_id"] is not None,
                            cancel_attempted=cancel_attempted,
                            remote_state=remote.state,
                        )
                        continue
                    linked = self.orchestrator.link_remote(
                        job_id, remote, event_type="dispatch_retried_idempotent"
                    )
                    if (
                        linked is not None
                        and not is_terminal(linked["status"])
                        and linked["cancel_requested"]
                    ):
                        self.orchestrator.cancel_remote(job_id)
                else:
                    self.orchestrator.mark_ambiguous(
                        job_id,
                        "stale dispatching and adapter cannot reconcile by correlation key",
                    )
            except Exception as exc:
                logger.exception("stale dispatch recovery failed for %s", job_id)
                self.last_error = f"{type(exc).__name__}: {exc}"
                current = self.orchestrator.get_job(job_id)
                if current is not None and current["status"] == JobStatus.DISPATCHING.value:
                    self.orchestrator.mark_ambiguous(
                        job_id,
                        f"stale dispatch recovery could not establish remote "
                        f"state: {type(exc).__name__}; no resubmit",
                    )

    # -------------------------------------------------------------- deadlines
    def enforce_deadlines(self) -> None:
        current_ms = now_ms()
        with self.orchestrator.db.read() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                 WHERE status NOT IN (?, ?, ?, ?)
                   AND deadline_ts <= ?
                 ORDER BY deadline_ts ASC
                 LIMIT 20
                """,
                (
                    JobStatus.SUCCEEDED.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                    JobStatus.TIMED_OUT.value,
                    current_ms,
                ),
            ).fetchall()

        for row in rows:
            job_id = row["job_id"]
            current = self.orchestrator.get_job(job_id)
            if current is None or is_terminal(current["status"]):
                continue
            cancel_attempted = False
            remote_state: str | None = None
            had_prompt_id = row["prompt_id"] is not None

            # Reconcile a crash-window dispatching job first. A remote job may
            # exist even though the local prompt_id was never saved.
            if (
                row["status"] == JobStatus.DISPATCHING.value
                and row["prompt_id"] is None
            ):
                try:
                    remote = self.orchestrator.adapter.reconcile(
                        row["correlation_key"]
                    )
                except Exception:
                    remote = None
                if remote is not None:
                    # Identity only. A remote success observed here must not
                    # become job_succeeded; the deadline terminal state wins.
                    linked = self.orchestrator.link_remote(
                        job_id,
                        remote,
                        event_type="dispatch_reconciled_at_deadline",
                        apply_result=False,
                    )
                    if linked is not None:
                        had_prompt_id = linked["prompt_id"] is not None
                        remote_state = remote.state
                        if linked["prompt_id"]:
                            cancel_attempted = True
                            try:
                                cancelled = self.orchestrator.adapter.cancel(
                                    linked["prompt_id"]
                                )
                                if cancelled is not None:
                                    remote_state = cancelled.state
                            except Exception:
                                pass

            if row["prompt_id"] and not cancel_attempted:
                cancel_attempted = True
                try:
                    remote = self.orchestrator.adapter.cancel(row["prompt_id"])
                    if remote is not None:
                        remote_state = remote.state
                except Exception:
                    pass

            self.orchestrator.mark_timeout(
                job_id,
                deadline_at=row["deadline_at"],
                had_prompt_id=had_prompt_id,
                cancel_attempted=cancel_attempted,
                remote_state=remote_state,
            )

    # --------------------------------------------------------------- dispatch
    def dispatch_one(self) -> None:
        current_ms = now_ms()
        with self.orchestrator.db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_ms ASC LIMIT 1"
            ).fetchone()
            if row is None:
                return
            job_id = row["job_id"]
            if (
                str(row["execution_mode"]) != self.orchestrator.execution_mode
                or str(row["execution_fingerprint"])
                != self.orchestrator.execution_fingerprint
            ):
                mismatch = (
                    str(row["execution_mode"]),
                    str(row["execution_fingerprint"]),
                )
            else:
                mismatch = None
                if int(row["deadline_ts"]) <= current_ms:
                    # enforce_deadlines() normally handles this before dispatch.
                    return
                if row["cancel_requested_at"] is not None:
                    self.orchestrator._transition(
                        conn,
                        job_id,
                        JobStatus.CANCELLED.value,
                        "job_cancelled",
                        {"reason": "cancel_before_dispatch"},
                    )
                    return
                updated = self.orchestrator._transition(
                    conn,
                    job_id,
                    JobStatus.DISPATCHING.value,
                    "dispatch_started",
                    {
                        "correlation_key": row["correlation_key"],
                        "worker_id": self.worker_id,
                        "mock": str(row["execution_mode"]) == "MOCK",
                        "mode": str(row["execution_mode"]),
                        "execution_fingerprint": str(
                            row["execution_fingerprint"]
                        ),
                    },
                    updates={
                        "dispatch_started_at": now_iso(),
                        "lease_owner": self.worker_id,
                        "lease_expires_at_ms": current_ms
                        + int(self.lease_seconds * 1000),
                    },
                )
                if updated is None:
                    return
                row = updated

        if mismatch is not None:
            stored_mode, stored_fingerprint = mismatch
            self.orchestrator.fail_job(
                job_id,
                "EXECUTION_BINDING_MISMATCH",
                "Queued job belongs to a different execution identity; refusing "
                "to submit or cancel it against the current adapter.",
                event_type="job_failed",
                data={
                    "stage": "dispatch",
                    "stored_mode": stored_mode,
                    "stored_fingerprint": stored_fingerprint,
                    "current_mode": self.orchestrator.execution_mode,
                    "current_fingerprint": self.orchestrator.execution_fingerprint,
                },
            )
            return

        if self._fault("crash_after_claim"):
            os._exit(17)

        fresh = self.orchestrator.get_job(job_id)
        if fresh is None or is_terminal(fresh["status"]):
            return
        if int(fresh["deadline_ts"]) <= now_ms():
            self.orchestrator.mark_timeout(
                job_id,
                deadline_at=fresh["deadline_at"],
                had_prompt_id=False,
                cancel_attempted=False,
                remote_state=None,
            )
            return
        if fresh["cancel_requested"]:
            self.orchestrator.request_cancel_after_dispatch(job_id)
            return

        # Final pre-submit deadline check. The job was already claimed; if the
        # deadline passed between claim and submit, do not create a remote job.
        if int(fresh["deadline_ts"]) <= now_ms():
            self.orchestrator.mark_timeout(
                job_id,
                deadline_at=fresh["deadline_at"],
                had_prompt_id=False,
                cancel_attempted=False,
                remote_state=None,
            )
            return

        try:
            payload = json.loads(row["payload_canonical"])
            remote = self.orchestrator.adapter.submit(
                row["correlation_key"],
                payload,
                deadline_ms=int(fresh["deadline_ts"]),
            )
        except RemoteDeadlineExpired as exc:
            # Confirmed not sent: the transport checked the persisted absolute
            # deadline before /prompt (including after upload/binding). This is
            # a local timeout, not an ambiguous dispatch and not a retry.
            logger.warning("dispatch deadline expired before send for %s", job_id)
            self.last_error = f"{exc.code}: {exc.message}"
            self.orchestrator.mark_timeout(
                job_id,
                deadline_at=fresh["deadline_at"],
                had_prompt_id=False,
                cancel_attempted=False,
                remote_state=None,
                extra_data={
                    "stage": "submit",
                    "classification": "confirmed_not_sent",
                    "accepted": False,
                    "prompt_requests": 0,
                },
            )
            return
        except RemoteValidationRejected as exc:
            logger.warning("dispatch validation rejected for %s", job_id)
            self.last_error = f"{exc.code}: {exc.message}"
            self.orchestrator.fail_job(
                job_id,
                exc.code,
                exc.message,
                event_type="job_failed",
                data={
                    "stage": "submit",
                    "classification": "validation_rejected",
                    "accepted": False,
                    "correlation_key": row["correlation_key"],
                },
            )
            return
        except RemoteTransientError as exc:
            logger.warning("dispatch transient failure for %s", job_id)
            self.last_error = f"{exc.code}: {exc.message}"
            self.orchestrator.fail_job(
                job_id,
                exc.code,
                exc.message
                + " No automatic resubmit was attempted; a new idempotency key "
                "requires an explicit decision.",
                event_type="job_failed",
                data={
                    "stage": "submit",
                    "classification": "transient_connect",
                    "accepted": False,
                    "correlation_key": row["correlation_key"],
                    "auto_retry": False,
                },
            )
            return
        except RemoteAuthError as exc:
            logger.error("dispatch authentication failure for %s", job_id)
            self.last_error = f"{exc.code}: {exc.message}"
            self.orchestrator.fail_job(
                job_id,
                exc.code,
                exc.message,
                event_type="job_failed",
                data={
                    "stage": "submit",
                    "classification": "authentication",
                    "accepted": False,
                    "correlation_key": row["correlation_key"],
                },
            )
            return
        except RemoteProtocolError as exc:
            logger.error("dispatch protocol failure for %s", job_id)
            self.last_error = f"{exc.code}: {exc.message}"
            self.orchestrator.fail_job(
                job_id,
                exc.code,
                exc.message,
                event_type="job_failed",
                data={
                    "stage": "submit",
                    "classification": "protocol_error",
                    "accepted": False,
                    "correlation_key": row["correlation_key"],
                },
            )
            return
        except RemoteAmbiguousSubmit as exc:
            logger.error("dispatch ambiguous for %s", job_id)
            self.last_error = f"{exc.code}: {exc.message}"
            self.orchestrator.mark_ambiguous(
                job_id,
                f"ambiguous submit: {exc.message}; correlation_key="
                f"{row['correlation_key']}",
            )
            return
        except Exception as exc:
            logger.exception("dispatch failed for %s", job_id)
            self.last_error = f"{type(exc).__name__}: {exc}"
            try:
                remote = self.orchestrator.adapter.reconcile(
                    row["correlation_key"]
                )
            except Exception:
                remote = None
            if remote is not None:
                self.orchestrator.link_remote(
                    job_id, remote, event_type="dispatch_reconciled"
                )
            else:
                self.orchestrator.mark_ambiguous(
                    job_id, f"submit raised {type(exc).__name__}: {exc}"
                )
            return

        if self._fault("crash_after_remote_submit"):
            os._exit(17)

        linked = self.orchestrator.link_remote(
            job_id, remote, event_type="remote_submitted"
        )
        if (
            linked is not None
            and not is_terminal(linked["status"])
            and linked["cancel_requested"]
        ):
            self.orchestrator.cancel_remote(job_id)

    # ------------------------------------------------------------------ poll
    def poll_active(self) -> None:
        with self.orchestrator.db.read() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                 WHERE status IN (?, ?)
                 ORDER BY updated_ms ASC
                 LIMIT 50
                """,
                (JobStatus.SUBMITTED.value, JobStatus.RUNNING.value),
            ).fetchall()

        for row in rows:
            job_id = row["job_id"]
            fresh = self.orchestrator.get_job(job_id)
            if fresh is None or fresh["status"] in TERMINAL_STATUSES:
                continue
            if fresh["cancel_requested"]:
                cancelled = self.orchestrator.cancel_remote(job_id)
                if cancelled is not None and is_terminal(cancelled["status"]):
                    continue
                fresh = self.orchestrator.get_job(job_id)
                if fresh is None or fresh["status"] in TERMINAL_STATUSES:
                    continue
            prompt_id = fresh["prompt_id"]
            if not prompt_id:
                continue
            try:
                remote = self.orchestrator.adapter.get(prompt_id)
            except RemotePromptLost as exc:
                logger.error("remote prompt lost for %s", job_id)
                self.last_error = f"{exc.code}: {exc.message}"
                self.orchestrator.fail_job(
                    job_id,
                    exc.code,
                    exc.message,
                    event_type="remote_prompt_lost",
                    data={
                        "prompt_id": prompt_id,
                        "classification": "lost_prompt",
                        "auto_retry": False,
                        "remote_absent_confirmed": False,
                    },
                )
                continue
            except RemoteProtocolError as exc:
                logger.error("poll protocol failure for %s", job_id)
                self.last_error = f"{exc.code}: {exc.message}"
                self.orchestrator.fail_job(
                    job_id,
                    exc.code,
                    exc.message,
                    event_type="job_failed",
                    data={
                        "prompt_id": prompt_id,
                        "classification": "protocol_error",
                        "auto_retry": False,
                    },
                )
                continue
            except RemoteAuthError as exc:
                logger.error("poll authentication failure for %s", job_id)
                self.last_error = f"{exc.code}: {exc.message}"
                self.orchestrator.fail_job(
                    job_id,
                    exc.code,
                    exc.message,
                    event_type="job_failed",
                    data={"prompt_id": prompt_id, "classification": "authentication"},
                )
                continue
            except RemoteTransientError as exc:
                logger.warning("poll transient failure for %s", job_id)
                self.last_error = f"{exc.code}: {exc.message}"
                continue
            except Exception as exc:
                logger.exception("poll failed for %s", job_id)
                self.last_error = f"{type(exc).__name__}: {exc}"
                # Unexpected adapter failures must not leave a job running
                # forever without a visible error. Deadlines remain the final
                # backstop, but an explicit terminal error is clearer.
                self.orchestrator.fail_job(
                    job_id,
                    "REMOTE_PROTOCOL_ERROR",
                    f"Unexpected remote poll failure: {type(exc).__name__}",
                    event_type="job_failed",
                    data={"prompt_id": prompt_id, "classification": "protocol_error"},
                )
                continue
            if remote is None:
                continue
            self.orchestrator.apply_remote_state(job_id, remote)

    @staticmethod
    def _fault(name: str) -> bool:
        return os.environ.get("ANIME_STUDIO_FAULT") == name
