# Сторонние компоненты

Shotsmith использует или интегрируется со следующими проектами. Их код и веса не становятся собственностью Shotsmith и не получают новую лицензию из-за этой публикации.

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) — внешний runtime. Устанавливается отдельно; его исходное дерево не включено в репозиторий.
- [ComfyUI workflow templates](https://github.com/Comfy-Org/workflow_templates) — источник опорных графов. Сведения о происхождении и pinned revisions сохранены в `workflows/K1/PROVENANCE.json` и `manifests/`.
- [Wan 2.2](https://github.com/Wan-Video/Wan2.2) и [Wan 2.1](https://github.com/Wan-Video/Wan2.1) — семейство видеомоделей; файлы весов не включены.
- [Qwen Image](https://github.com/QwenLM/Qwen-Image) — генерация и редактирование опорных кадров; файлы весов не включены.
- [Terminal.Gui](https://github.com/gui-cs/Terminal.Gui) — терминальный интерфейс, NuGet-зависимость.
- [FFmpeg](https://ffmpeg.org/) — внешняя программа для проверки и сборки видео. Бинарный файл не включён; пользователь устанавливает подходящую сборку отдельно.
- Python-зависимости перечислены в `requirements.txt`; .NET-зависимости и lock-файл — в `ui/Shotsmith/`.

Для конкретных весов профили содержат source URL, revision, SHA256 и ссылки на лицензионные сведения. Перед использованием проверяйте условия выбранной версии и соответствующего источника. Демонстрационные кадры и ролик являются AI-generated материалами проекта, а не частью библиотек выше.
