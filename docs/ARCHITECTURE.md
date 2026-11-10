# Architecture and naming

Shotsmith separates the interactive client from project data and execution.

The Terminal.Gui application sends commands to the Python bridge, which uses
the same handlers as the PowerShell entry points. Project authoring produces
a validated keyframe plan; keyframe execution retains versions and approvals;
video execution snapshots approved inputs and uses a pinned ComfyUI graph.
FFmpeg validates and assembles accepted video clips locally.

## Internal names

- `i1`: SSH installer and connection subsystem.
- `kf1`: reference binding, keyframe execution, version history and approvals.
- `kf2`: project authoring, compiling plans for `kf1`.
- `p1`: earlier orchestration and supporting components reused by the installer.
- `anime_studio`: shared Python package under its original import name.
- `avs.ps1`: compatible low-level entry point; `shotsmith.ps1` is the application launcher.

These are module names, not additional products or separate installations.
They remain stable to preserve imports, existing plans, recovery state and tests.
Renaming them requires a migration rather than cosmetic file changes.

## Local and remote responsibilities

The client stores projects, copied references, hashes, approvals and job records
locally under the ignored `data/` directory. A remote Linux worker runs ComfyUI
and holds model files. Access goes through SSH and a localhost tunnel.
Connection loss is distinct from a confirmed worker failure; recovery does not
submit a second generation merely because an observation failed.

The author supplies scene descriptions and A/B states. There is no hidden paid
language-model planner. Reviewing images and accepting video clips are explicit
author actions. Leaving the client does not terminate a rented server.
