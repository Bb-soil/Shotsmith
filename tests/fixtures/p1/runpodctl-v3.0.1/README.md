# runpodctl v3.0.1 stop/readback fixtures

Status: **SOURCE_RECONSTRUCTION_NOT_LIVE**

These files are source-shaped offline fixtures for the pinned `runpodctl`
contract. They are **not** live captures, were not produced by an account or a
Pod, and must not be used as live stop/readiness evidence.

## Pin

- Repository: `runpod/runpodctl`
- Release: `v3.0.1`
- Tag commit: `551d6a4310767ad9671961bea9f64cb2c365dc1e`
- Release published: 2026-10-15
- Public source read over HTTPS on 2026-10-21; no binary was downloaded or
  installed and no provider endpoint was called.

Pinned source blobs:

| File | Blob SHA | Public URL |
|---|---|---|
| `cmd/pod/get.go` | `d3861cad47cc4fd73302e83a162d3654d16b4630` | <https://github.com/runpod/runpodctl/blob/551d6a4310767ad9671961bea9f64cb2c365dc1e/cmd/pod/get.go> |
| `cmd/pod/stop.go` | `a83efb489b6cce8cfbfb930b093a94d8d58fb87c` | <https://github.com/runpod/runpodctl/blob/551d6a4310767ad9671961bea9f64cb2c365dc1e/cmd/pod/stop.go> |
| `internal/api/pods.go` | `e09331593df8c146329d88d09e81912aaeb47274` | <https://github.com/runpod/runpodctl/blob/551d6a4310767ad9671961bea9f64cb2c365dc1e/internal/api/pods.go> |
| `internal/api/pods_v2.go` | `91b49c733549a209ff679c1bd2538ef22d55d6a5` | <https://github.com/runpod/runpodctl/blob/551d6a4310767ad9671961bea9f64cb2c365dc1e/internal/api/pods_v2.go> |
| `internal/podstate/podstate.go` | `bb669f0c1fcf413539f2c5182c56954038417bcb` | <https://github.com/runpod/runpodctl/blob/551d6a4310767ad9671961bea9f64cb2c365dc1e/internal/podstate/podstate.go> |
| `internal/sshconnect/sshconnect.go` | `cdb4b867cf8564f80014ec145fc91182cd004f5c` | <https://github.com/runpod/runpodctl/blob/551d6a4310767ad9671961bea9f64cb2c365dc1e/internal/sshconnect/sshconnect.go> |

Public documentation read: <https://docs.runpod.io/runpodctl/reference/runpodctl-pod>.

## Contract used by the fixtures

`runpodctl pod get <POD_ID> --output json` prints one JSON object. The fields
used by the backend are:

- `id`: exact Pod ID;
- `desiredStatus`: `RUNNING`, `EXITED`, `TERMINATED`, `ERROR`;
- `runtimeStatus`: `running`, `initializing`, `stopped`, `terminated`, `error`,
  `unknown`;
- `runtimeStatusReason`: optional stable token;
- `createdAt`: optional, source `pod get` uses Go `time.String()` (for example
  `2026-10-21 12:00:00.123456789 +0000 UTC`); RFC3339 is also represented.

`STOPPED` requires `desiredStatus=EXITED` plus `runtimeStatus=stopped`.
`terminated` is not stop, `error` may still bill, and `unknown` +
`runtime_unavailable` is not proof of stop.

## Files

- `pod-get-stopped.json` — EXITED/stopped, legacy `createdAt` with nanoseconds.
- `pod-get-running.json` — RUNNING/running, RFC3339 `createdAt`.
- `pod-get-initializing.json` — RUNNING/initializing with reason.
- `pod-get-unknown.json` — RUNNING/unknown + runtime_unavailable, offset timezone.
- `pod-get-terminated.json`, `pod-get-error.json` — not STOPPED.
- `pod-get-wrong-id.json`, `pod-get-missing-id.json`,
  `pod-get-missing-fields.json`, `pod-get-contradictory.json`,
  `pod-get-null-runtime.json`, `pod-get-missing-runtime.json`,
  `pod-get-createdat-bad.json` — fail-closed shapes.
- `pod-get-malformed.json`, `pod-get-wrong-type.json` — malformed JSON types.
- `pod-get-secret-fields.json` — synthetic `env`/`ssh`/network-volume values;
  tests assert they never appear in exceptions or state.
- `*.stderr.json` — flat CLI error shapes; only `code` and numeric `status` are
  parsed. The human-readable `error` text is synthetic and must not leak.

None of these fixtures asserts a live provider state. They are offline contract
fixtures only.
