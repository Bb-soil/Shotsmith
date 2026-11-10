# Third-party components and media

Apache-2.0 applies to Shotsmith original application code and documentation. It does not replace third-party licenses, grant rights to model weights or relicense upstream-derived workflow material.

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) is an external runtime, installed separately. Its source tree is not bundled here.
- [ComfyUI workflow templates](https://github.com/Comfy-Org/workflow_templates) are an upstream source for reference graphs. Provenance and pinned revisions are recorded in `workflows/K1/PROVENANCE.json`, `workflows/P1/reference/` and `manifests/`. Retain the applicable upstream notices and license terms when reusing derived material.
- [Wan 2.2](https://github.com/Wan-Video/Wan2.2), [Wan 2.1](https://github.com/Wan-Video/Wan2.1) and [Qwen Image](https://github.com/QwenLM/Qwen-Image) provide model families. No model weights are included.
- [Terminal.Gui](https://github.com/gui-cs/Terminal.Gui) supplies the terminal interface through NuGet.
- [FFmpeg](https://ffmpeg.org/) is an external video tool. No FFmpeg binary is included; install an appropriate build separately under its applicable license.
- Python dependencies are listed in `requirements.txt`; .NET dependencies are declared in `ui/Shotsmith/`.

Model profiles record source URLs, revisions, hashes and license-source links. Check the terms for the exact version and source you use.

Demo frames and videos showcase AI-generated project experiments. They are not distributed under the application-code license. The terminal help image is rendered from an actual offline application screen dump; `docs/media/terminal-help.txt` contains that dump. It is not evidence of a live GPU run.
