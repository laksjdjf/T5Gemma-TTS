#!/usr/bin/env bash

set -euo pipefail

# This script trains a Masked Diffusion model from scratch using example data.
# Unlike autoregressive training, this uses a non-autoregressive masked diffusion approach.
# このスクリプトはMasked Diffusionモデルをスクラッチから学習します。
# 自己回帰とは異なり、非自己回帰的なマスク拡散アプローチを使用します。

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

T5GEMMA_MODEL_NAME=google/t5gemma-2b-2b-ul2
XCODEC2_MODEL_NAME=NandemoGHS/Anime-XCodec2-44.1kHz-v2

# Data for test training
EMILIA_YODAS_ROOT="${PROJECT_ROOT}/datasets/emilia-yodas-en_0-9"

EXP_ROOT="${PROJECT_ROOT}/runs/masked_diffusion_2b-2b"

NUM_GPUS="${NUM_GPUS:-1}"
# Set your wandb entity (username or team name) via environment variable or edit this line
WANDB_ENTITY="${WANDB_ENTITY:-your_wandb_entity}"
BATCH_SIZE=4
NUM_STEPS=143391
LR=0.035
WARMUP_FRAC=0.02
VAL_EVERY=2000
PRINT_EVERY=10

# Masked Diffusion specific parameters
DIFFUSION_STEPS=10
MASK_SCHEDULE=cosine

NEIGHBOR_PROB=0.5
NEIGHBOR_FOLDER="${NEIGHBOR_FOLDER_NAME:-neighbors}"
# Reserved tokens for T5Gemma
X_SEP_TOKEN=255999
N_SPECIAL=5
AUDIO_VOCAB_SIZE=65536
# EMPTY_TOKEN, EOG, AUDIO_PAD_TOKEN, EOS, Y_SEP_TOKEN are set internally

# If OOM, try reducing these values or increasing gradient_accumulation_steps
MAX_NUM_TOKENS=30000
VAL_MAX_NUM_TOKENS=5000

mkdir -p "${EXP_ROOT}"

DATASET_DIRS="['${EMILIA_YODAS_ROOT}']"
MANIFEST_NAMES="['manifest_final']"

echo "[Info] launching torchrun with ${NUM_GPUS} GPU(s) for Masked Diffusion training"
torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_ROOT}/main.py" \
  --model_arch masked_diffusion \
  --t5gemma_model_name "${T5GEMMA_MODEL_NAME}" \
  --text_input_type text \
  --text_tokenizer_name "${T5GEMMA_MODEL_NAME}" \
  --audio_tokenizer xcodec2 \
  --xcodec2_model_name "${XCODEC2_MODEL_NAME}" \
  --audio_vocab_size "${AUDIO_VOCAB_SIZE}" \
  --diffusion_steps "${DIFFUSION_STEPS}" \
  --mask_schedule "${MASK_SCHEDULE}" \
  --progress_scale 2000 \
  --neighbor_prompt_prob "${NEIGHBOR_PROB}" \
  --neighbor_folder_name "${NEIGHBOR_FOLDER}" \
  --n_special "${N_SPECIAL}" \
  --x_sep_token "${X_SEP_TOKEN}" \
  --no_loss_on_prefix 1 \
  --min_prompt_len 0.5 \
  --audio_max_length 40 \
  --audio_min_length 0.2 \
  --text_max_length 500 \
  --encodec_sr 50 \
  --dataset_dir "${DATASET_DIRS}" \
  --manifest_name "${MANIFEST_NAMES}" \
  --encodec_folder_name xcodec2_1cb \
  --audio_folder_name audio \
  --target_time_stretch_prob 0 \
  --time_stretch_prob 0 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers 2 \
  --max_num_tokens "${MAX_NUM_TOKENS}" \
  --val_max_num_tokens "${VAL_MAX_NUM_TOKENS}" \
  --num_steps "${NUM_STEPS}" \
  --lr "${LR}" \
  --warmup_fraction "${WARMUP_FRAC}" \
  --precision bfloat16 \
  --print_every_n_steps "${PRINT_EVERY}" \
  --val_every_n_steps "${VAL_EVERY}" \
  --inference_every_n_steps 100000000 \
  --save_every_n_steps 1000 \
  --tb_write_every_n_steps 1 \
  --seed 1 \
  --exp_dir "${EXP_ROOT}" \
  --drop_long 1 \
  --pad_x 0 \
  --text_pad_token 0 \
  --num_buckets 20 \
  --gradient_accumulation_steps 4 \
  --optimizer_name "ScaledAdam" \
  --pseudo_epoch_size 5000 \
  --reduce_lr_start_step 5000 \
  --reduce_lr_start_epoch 6 \
  --clipping_update_period 1000 \
  --validation_sample_cap 30000 \
  --t5_gradient_checkpointing 1 \
  --prune_text_modules 2 \
  --compile 0 \
  --attn_implementation sdpa \
  --ddp_find_unused_parameters 0 \
  --wandb_entity "${WANDB_ENTITY}" \
  --resume
