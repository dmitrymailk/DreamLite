#!/usr/bin/env bash
# Запуск редактирования изображения через готовый infer_mobile.py (режим edit).
#
# Параметры можно переопределить переменными окружения или позиционными аргументами:
#   PYTHON=...  MODEL_ID=...  PROMPT=...  IMAGE_PATH=...  bash run_edit.sh
#   bash run_edit.sh "<prompt>" "<image_path>"
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-/code/.venv_code/bin/python}"
MODEL_ID="${MODEL_ID:-/code/models/DreamLite-mobile}"
PROMPT="${1:-${PROMPT:-turn it into a realistic photograph}}"
IMAGE_PATH="${2:-${IMAGE_PATH:-/code/inference_optimization/image (3).jpg}}"

echo "PYTHON=${PYTHON}"
echo "MODEL_ID=${MODEL_ID}"
echo "PROMPT=${PROMPT}"
echo "IMAGE_PATH=${IMAGE_PATH}"

"${PYTHON}" infer_mobile.py \
    --model_id "${MODEL_ID}" \
    --prompt "${PROMPT}" \
    --image_path "${IMAGE_PATH}"
