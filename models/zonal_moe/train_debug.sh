#!/bin/bash
# Debug training script for Zonal MoE
# Enables CUDA debugging flags

export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TORCH_USE_CUDA_DSA=1

python -m models.zonal_moe.train "$@"