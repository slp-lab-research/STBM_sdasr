#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

HYDRA_FULL_ERROR=1 torchrun --nproc_per_node=1 train.py \
  --config-path configs \
  --config-name train_qwen25_omni_7b_speaker_kernel_alimeeting \
  "$@"
