# ECUP Counterfeit Classifier

Мультимодальная (текст + изображение + табличные признаки) офлайновая система для классификации товаров на контрафакт / оригинал. Проект запускается локально и/или в Docker, поддерживает офлайн-зависимости и опциональный OCR через Tesseract.

---

## 🚀 Возможности

* Три модальности: **текст** (локальная HF‑модель), **изображение** (CLIP + линейная голова), **таблица** (sklearn‑pipeline).
* Гибкая агрегация: `weighted | mean | max` + настраиваемые веса.
* Выход: `results/submission.csv` и отладочный `results/debug.csv`.
* Опциональный **OCR** (Tesseract `rus+eng`) — можно подмешать распознанный текст к текстовой модальности флагом `--use_ocr`.
* Полностью офлайновый режим (по желанию) — CLIP можно ставить из локального `third_party/CLIP`.

---

## 📁 Структура репозитория

```
├─ data/
│  ├─ imgs/
│  ├─ data.csv
│  └─ data_full.csv
├─ image_model/
│  ├─ classifier_checkpoint.pth
│  ├─ classifier_checkpoint3.pth
│  └─ ViT-B-32.pt
├─ results/
├─ src/
│  ├─ main.py
│  ├─ text_analyze.py
│  ├─ image_analyze.py
│  └─ tabular_analyze.py
├─ table_model/
│  └─ f1_pipeline_final.pkl
├─ text_model/  # локальная HF-модель
│  ├─ config.json
│  ├─ model.safetensors
│  ├─ tokenizer.json / tokenizer_config.json / vocab.txt
│  └─ special_tokens_map.json
├─ run_example.sh
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

---

## 🔧 Зависимости

* Python 3.10+ (для Docker используется `python:3.10-slim`).
* PyTorch, Transformers, scikit‑learn, pandas, Pillow и т.д. — см. `requirements.txt`.
* **OCR (опционально):** системный пакет `tesseract-ocr` + язык `tesseract-ocr-rus`.

### Онлайн vs офлайн CLIP

* Онлайн: `requirements.txt` содержит `git+https://github.com/openai/CLIP.git`.
* Офлайн: положите код CLIP в `third_party/CLIP` и замените в `requirements.txt` на `-e ./third_party/CLIP`.

---

## ⚙️ Подготовка данных

* Входной CSV по умолчанию: `./data/data.csv`.
* Обязательные столбцы: `id`, `name_rus`, `description`. (Если нет `id` — будет создан.)
* Опционально: `resolution` (0/1) — для подсчёта F1 и автоподбора порога (`--auto_threshold`).
* Изображения: по умолчанию ищутся как `data/imgs/<id>.jpg|.png|.jpeg|.webp` или по явному имени файла из столбца `--img_col`.

---

## 🖥️ Быстрый старт (локально)

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.main \
  --csv_path ./data/data.csv \
  --image_folder ./data/imgs \
  --text_model_dir ./text_model \
  --clip_model ./image_model/ViT-B-32.pt \
  --classifier_path ./image_model/classifier_checkpoint3.pth \
  --tabular_model_path ./table_model/f1_pipeline_final.pkl \
  --out_csv ./results/submission.csv
```

### С OCR

Добавьте флаг `--use_ocr` и убедитесь, что установлен Tesseract (`rus+eng`).

---

## 🐳 Docker

### Сборка

```bash
docker build -t ecup2025 .
```

### Запуск (Linux/macOS)

```bash
docker run --rm -v "$(pwd)":/project -w /project ecup2025 \
  bash -lc "python -m src.main --csv_path ./data/data.csv --image_folder ./data/imgs \
  --text_model_dir ./text_model --clip_model ./image_model/ViT-B-32.pt \
  --classifier_path ./image_model/classifier_checkpoint3.pth \
  --tabular_model_path ./table_model/f1_pipeline_final.pkl \
  --out_csv ./results/submission.csv"
```

### Запуск (Windows PowerShell, одна строка)

```powershell
docker build -t ecup2025 .; docker run --rm -v "${PWD}:/project" -w /project ecup2025 bash -lc "python -m src.main --csv_path ./data/data.csv --image_folder ./data/imgs --text_model_dir ./text_model --clip_model ./image_model/ViT-B-32.pt --classifier_path ./image_model/classifier_checkpoint3.pth --tabular_model_path ./table_model/f1_pipeline_final.pkl --out_csv ./results/submission.csv"
```

> Для OCR добавьте `--use_ocr` внутри кавычек. Для GPU при наличии CUDA добавьте `--gpus all` к `docker run`.

**Важно (Windows):** Ошибка вида `//./pipe/dockerDesktopLinuxEngine` означает, что Docker Desktop не запущен/не выбран Linux‑контекст. Откройте Docker Desktop → *Switch to Linux containers* или `docker context use desktop-linux`.

---

## 🧪 Использование CLI (ключевые флаги)

```text
--csv_path, --image_folder, --text_model_dir
--clip_model, --classifier_path, --tabular_model_path
--id_col, --name_col, --desc_col, --img_col, --img_ext
--agg {weighted,mean,max} --w_text --w_img --w_tab
--threshold 0.5 | --auto_threshold
--device cpu|cuda
--use_ocr  # подмешать распознанный с изображения текст
--disable_text --disable_image --disable_tabular
```

Пример с весами и автопорогом:

```bash
python -m src.main \
  --csv_path ./data/data_full.csv \
  --agg weighted --w_text 0.4 --w_img 0.4 --w_tab 0.2 \
  --auto_threshold \
  --out_csv ./results/submission.csv
```

---

## 📦 Выходные файлы

* `results/submission.csv` — столбцы: `id, prediction` (0/1).
* `results/debug.csv` — диагностика: `text_prob, image_prob, tabular_prob, final_prob, pred`.
* В логах печатается `F1` (если в данных есть `resolution`).

---

## 🔍 OCR (Tesseract)

* В Docker образе должен быть установлен `tesseract-ocr` и язык `tesseract-ocr-rus`.
* Быстрая проверка внутри контейнера:

```bash
tesseract --version && tesseract --list-langs | head -n 5
```

* Если Tesseract недоступен, `--use_ocr` просто не добавит текст (пустая строка), пайплайн продолжит работу.

---

## 🛠️ Скрипт запуска

См. `run_example.sh` — создаёт `results/` и прокидывает все параметры из командной строки (`"$@"`).
