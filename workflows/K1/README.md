# K1: опорные кадры по референсам

DRAFT / GPU_NOT_RUN. Три native API-варианта для 1/2/3 изображений; отсутствующий optional ref не подменять dummy-файлом. {{EDIT_INSTRUCTION}} необходимо связать до submit; черновик с placeholder не отправлять.

Image1 — scene anchor, image2 — постоянный character reference, image3 — optional environment reference. После подготовки binding проверять actual object_info, links, literal types, model names. Полной source-schema проверки ещё нет.

704×1280 center-crop выбран как стартовый candidate, не безусловная политика для всех исходников: показать/зафиксировать crop, не растягивать. Полученные PNG затем нормализуются для Wan отдельно. Не использовать character sheet как image1, если уже предоставлен первый кадр.

Источники и изменения: PROVENANCE.json; веса/лицензии/пины: manifests/K1.qwen-edit-2511.sources.draft.json. Никаких моделей не скачано; FP8 качество и 24GB VRAM ещё не проверены.
