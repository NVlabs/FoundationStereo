#!/usr/bin/env bash
# Rectify + FoundationStereo depth over the whole SCJ mapping sequence, using the frames
# sub-sampled along the SLAM trajectory (2213 frames with the default 2 m / 15 deg steps).
#
# Usage:  scripts/run_slamcore_full_dataset.sh [extra args for run_demo_batch_slamcore.py]
#   e.g.  scripts/run_slamcore_full_dataset.sh --save_cam0 --save_vis
#
# Every setting below can be overridden from the environment, e.g.
#   OUT_DIR=/some/where SUBSAMPLE_DISTANCE_M=1 scripts/run_slamcore_full_dataset.sh
#
# Re-running with the same OUT_DIR resumes: frames whose depth map already exists are skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET="${DATASET:-/mnt/network/data/Datasets/customers/scj/09_2026/scj_mapping/202609301813_sc-aware-JR5NB141_run_ebf0991c_sequence/sequence}"
TRAJECTORY_DIR="${TRAJECTORY_DIR:-/data/perception_alexander/scj/sc_mapping_ht_lc_relax}"
OUT_DIR="${OUT_DIR:-/mnt/data_pipeline/manual_datasets/scj/scj_mapping/fs_ht_lc_relax_full}"
CKPT="${CKPT:-${REPO_DIR}/23-51-11/model_best_bp2.pth}"
SUBSAMPLE_DISTANCE_M="${SUBSAMPLE_DISTANCE_M:-2.0}"
SUBSAMPLE_ROTATION_DEG="${SUBSAMPLE_ROTATION_DEG:-15.0}"
PYTHON="${PYTHON:-/mnt/data_pipeline/anaconda3/envs/foundation_stereo/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

COMMON_ARGS=(
    --dataset "${DATASET}"
    --trajectory_dir "${TRAJECTORY_DIR}"
    --out_dir "${OUT_DIR}"
    --ckpt_dir "${CKPT}"
    --subsample_distance_m "${SUBSAMPLE_DISTANCE_M}"
    --subsample_rotation_deg "${SUBSAMPLE_ROTATION_DEG}"
)

mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

cd "${REPO_DIR}"
# Report the frame count first (cheap), then run.
"${PYTHON}" scripts/run_demo_batch_slamcore.py "${COMMON_ARGS[@]}" --dry_run 2>&1 | grep "To process"
echo "Output: ${OUT_DIR}"
echo "Log:    ${LOG}"

PYTHONUNBUFFERED=1 "${PYTHON}" scripts/run_demo_batch_slamcore.py "${COMMON_ARGS[@]}" "$@" \
    2>&1 | grep --line-buffered -v "FutureWarning\|autocast(" | tee "${LOG}"
