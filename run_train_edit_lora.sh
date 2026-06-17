#!/usr/bin/env bash
# Базовый smoke-запуск обучения Image Editing LoRA для DreamLite.
# Лосс пишется только в консоль (wandb/внешнего логирования в трейнере нет).
# Датасет берётся из локального кэша (/code/dataset/...), без скачивания.
#
# Переопределяемые параметры (env):
#   PYTHON, MODEL_ID, MAX_TRAIN_STEPS, RESOLUTION, OUTPUT_DIR, DEFAULT_PROMPT,
#   DATASET_NAME, DATASET_SPLIT, CACHE_DIR, IMAGE_COLUMN, COND_IMAGE_COLUMN,
#   VALIDATION_STEPS, NUM_VALIDATION_SAMPLES, NUM_VALIDATION_STEPS, VALIDATION_RESOLUTION
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-/code/.venv_code/bin/python}"
MODEL_ID="${MODEL_ID:-/code/models/DreamLite-mobile}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
RESOLUTION="${RESOLUTION:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/output_lora/edit_nfs_smoke}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-remaster into a photorealistic image}"

DATASET_NAME="${DATASET_NAME:-dim/nfs_pix2pix_1920_1080_v6_upscale_2x_raw_filtered}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
CACHE_DIR="${CACHE_DIR:-/code/dataset/nfs_pix2pix_1920_1080_v6_upscale_2x_raw_filtered}"
IMAGE_COLUMN="${IMAGE_COLUMN:-edited_image}"
COND_IMAGE_COLUMN="${COND_IMAGE_COLUMN:-input_image}"

VALIDATION_STEPS="${VALIDATION_STEPS:-25}"
NUM_VALIDATION_SAMPLES="${NUM_VALIDATION_SAMPLES:-20}"
NUM_VALIDATION_STEPS="${NUM_VALIDATION_STEPS:-4}"
VALIDATION_RESOLUTION="${VALIDATION_RESOLUTION:-${RESOLUTION}}"

# Берём датасет только из локального кэша, ничего не качаем.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# dreamlite-пакет лежит в корне репозитория (этот каталог), а скрипт — в lora/,
# поэтому добавляем корень в PYTHONPATH, чтобы импорт `dreamlite` сработал.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "PYTHON=${PYTHON}"
echo "MODEL_ID=${MODEL_ID}"
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "RESOLUTION=${RESOLUTION}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DEFAULT_PROMPT=${DEFAULT_PROMPT}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "DATASET_SPLIT=${DATASET_SPLIT}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "IMAGE_COLUMN=${IMAGE_COLUMN}"
echo "COND_IMAGE_COLUMN=${COND_IMAGE_COLUMN}"
echo "VALIDATION_STEPS=${VALIDATION_STEPS}"
echo "NUM_VALIDATION_SAMPLES=${NUM_VALIDATION_SAMPLES}"
echo "NUM_VALIDATION_STEPS=${NUM_VALIDATION_STEPS}"
echo "VALIDATION_RESOLUTION=${VALIDATION_RESOLUTION}"

"${PYTHON}" lora/train_edit_lora.py \
    --model_id "${MODEL_ID}" \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --resolution "${RESOLUTION}" \
    --output_dir "${OUTPUT_DIR}" \
    --default_prompt "${DEFAULT_PROMPT}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_split "${DATASET_SPLIT}" \
    --cache_dir "${CACHE_DIR}" \
    --image_column "${IMAGE_COLUMN}" \
    --cond_image_column "${COND_IMAGE_COLUMN}" \
    --validation_steps "${VALIDATION_STEPS}" \
    --num_validation_samples "${NUM_VALIDATION_SAMPLES}" \
    --num_validation_steps "${NUM_VALIDATION_STEPS}" \
    --validation_resolution "${VALIDATION_RESOLUTION}"
