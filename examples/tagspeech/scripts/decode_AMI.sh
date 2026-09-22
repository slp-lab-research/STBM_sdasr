#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

AMI_EXP_DIR="${AMI_EXP_DIR:-experiments/diarization_AMI_speaker_kernel_crossattn_qwen25_omni_7b}"
echo "AMI experiment directory: ${AMI_EXP_DIR}"

python decode.py \
  exp_dir="${AMI_EXP_DIR}" \
  data.test_data_config=configs/AMI/data_configs/test_data_config.yaml \
  decode.max_duration=80 \
  "$@"
