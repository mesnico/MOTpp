#!/bin/bash

set -e

# read CUDA_DEVICE and DATASET from command line
while getopts d:s: flag
do
    case "${flag}" in
        d) CUDA_DEVICE=${OPTARG};;
        s) DATASET=${OPTARG};;
    esac
done

MODEL=t2m
PYTHON_EXEC="conda run --no-capture-output -n t2m_tmr python"

TEXT_MODELS=(
    # bert-lstm
    # actor-style-encoder
    clip-tpt-5
    # clip
    # clip-hf
)

MOTION_MODELS=(
    actor-style-encoder
    # movit-v2-factenc-bodyparts-timemask-uniformsample-200frames-ff1024-4heads
    # # movit-factenc-bodyparts-timemask-uniformsample-200frames
    # # movit-factenc-bodyparts-timemask-uniformsample-200frames-ff1024-4heads
    # movit-v2-factenc-bodyparts-timemask-uniformsample-200frames-4layers
    # movit-v2-factenc-bodyparts-timemask-uniformsample-200frames-ff1024-4heads-4layers
)

LOSSES=(
    info-nce-with-filtering
    # info-nce-cross-consistent-listnet-80-140.yaml
    # info-nce-cross-consistent-listnet-500-600.yaml
    # info-nce
    # info-nce-cross-consistent-kldiv-80-140
    # info-nce-cross-consistent-kldiv-500-600
)

DATASETS=(
    # $DATASET
    # kitml
    humanml3d
)

DATA_REPS=(
    # cont_6d_plus_rifke
    cont_6d_plus_rifke_vels
)

# transform into a string of comma-separated values
IFS=,
TEXT_MODELS="${TEXT_MODELS[*]}"
MOTION_MODELS="${MOTION_MODELS[*]}"
DATASETS="${DATASETS[*]}"
LOSSES="${LOSSES[*]}"
DATA_REPS="${DATA_REPS[*]}"

# Train & Evaluate
# Perform multiple repetitions of the same experiment, varying the split seed (for cross-validation)

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE HYDRA_FULL_ERROR=1 conda run --no-capture-output -n t2m_tmr python train.py -m model="$MODEL" data="$DATASETS" data_rep="$DATA_REPS" model/motion_encoder="$MOTION_MODELS" model/text_encoder="$TEXT_MODELS" model/contrastive_loss="$LOSSES"
# HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/run-$REP --debug 

