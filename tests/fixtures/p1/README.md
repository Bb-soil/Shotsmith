# P1 preflight fixtures

## `savevideo-object_info-7a283320.json`

Status: **SOURCE_RECONSTRUCTION_NOT_LIVE**

This file is an offline reconstruction of the `SaveVideo` entry in ComfyUI
`/object_info` at pinned commit
`7a28332047b6af638dc6c7f72ef3481a718769fd`. It is **not** a recording of a live
`/object_info` response and must not be used as live readiness evidence.

Pinned sources read read-only through HTTPS:

- `server.py` (node info path: `node_info` -> `GET_NODE_INFO_V1` ->
  `Schema.get_v1_info` -> `create_input_dict_v1`):
  <https://github.com/Comfy-Org/ComfyUI/blob/7a28332047b6af638dc6c7f72ef3481a718769fd/server.py>
- `comfy_api/latest/_io.py` (`DynamicCombo.Option.as_dict` keeps child inputs
  inside `options[].inputs`):
  <https://github.com/Comfy-Org/ComfyUI/blob/7a28332047b6af638dc6c7f72ef3481a718769fd/comfy_api/latest/_io.py>
- `comfy_extras/nodes_video.py` (`SaveVideo.define_schema`: `format`, nested
  `codec`, nested optional `encoding`; `mp4`/`h264`/`auto` are valid choices):
  <https://github.com/Comfy-Org/ComfyUI/blob/7a28332047b6af638dc6c7f72ef3481a718769fd/comfy_extras/nodes_video.py>

The expected prompt-facing path is `format=mp4` -> `format.codec=h264` ->
`format.codec.encoding=auto`. The API prompt keeps those flat keys; only
`object_info` nests them under DynamicCombo options.

Only the `SaveVideo` entry is source-derived. Every other class in the JSON is a
minimal **STUB** placeholder used to satisfy unrelated required-class checks; do
not read those entries as native schema claims.
