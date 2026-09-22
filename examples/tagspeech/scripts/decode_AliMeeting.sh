#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

ALI_EXP_DIR="${ALI_EXP_DIR:-experiments/diarization_AliMeeting_speaker_kernel_crossattn_qwen25_omni_7b}"
ALI_DECODE_ITER="${ALI_DECODE_ITER:-12000}"
ALI_DECODE_AVG="${ALI_DECODE_AVG:-5}"

echo "AliMeeting decode experiment: ${ALI_EXP_DIR}"
echo "AliMeeting checkpoint: iter=${ALI_DECODE_ITER}, avg=${ALI_DECODE_AVG}"

python decode.py \
  exp_dir="${ALI_EXP_DIR}" \
  data.test_data_config=configs/AliMeeting/data_configs/test_data_config.yaml \
  checkpoint.iter="${ALI_DECODE_ITER}" \
  checkpoint.avg="${ALI_DECODE_AVG}" \
  "$@"
