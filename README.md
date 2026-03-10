# Plant AI – Best Model Weights

Файл `best.pt` содержит обученные веса модели сегментации для классов:
- root
- stem
- leaves

Источник: обучение в проекте `agro_ai_system` (локальный пайплайн).
## Быстрый запуск

1. Установить зависимости:

```bash
pip install -r requirements.txt
```

2. Запустить инференс на примере:

```bash
python inference.py --source sample.jpg --model best.pt --out results
```

Результаты сохраняются в `results/`:
- `*_overlay.jpg` — картинка с масками
- `*_detections.json` — список детекций

Параметры:
- `--conf` порог уверенности
- `--imgsz` размер входа
- `--device` `cpu` или `cuda:0`

Классы модели:
- `root`
- `stem`
- `leaves`
