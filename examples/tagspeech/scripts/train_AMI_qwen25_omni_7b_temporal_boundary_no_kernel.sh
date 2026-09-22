#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Let torchrun choose an available local rendezvous port.  Its default (29500)
# can already be occupied when multiple single-node jobs share a compute node.
HYDRA_FULL_ERROR=1 torchrun --standalone --nproc_per_node=1 train.py \
  --config-path configs \
  --config-name train_qwen25_omni_7b_temporal_boundary_no_kernel_ami \
  "$@"
