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
    actor-style-encoder
    # clip
    # clip-tpt-4
    # clip-tpt-2
    # clip-tpt-8
    # clip-hf
)

MOTION_MODELS=(
    # actor-style-encoder
    movit-v2-factenc-bodyparts-timemask-uniformsample-200frames-ff1024-4heads-4layers
    # movit-v2-bodyparts-timemask-uniformsample-200frames-ff1024-4heads-4layers
    # movit-v2-joint-bodyparts-timemask-uniformsample-200frames-ff1024-4heads-4layers
    # movit-bodyparts-timemask-uniformsample-200frames
)

LOSSES=(
    # info-nce-with-filtering
    # info-nce-cross-consistent-kldiv-0-1
    info-nce-cross-consistent-kldiv-40-100
    # info-nce-cross-consistent-kldiv-80-140
    # info-nce-cross-consistent-kldiv-140-200
    # info-nce-cross-consistent-kldiv-500-600
    # info-nce-cross-consistent-kldiv-40-100-no-teacher-vs-t2t
    # info-nce-cross-consistent-kldiv-80-140-no-teacher-vs-t2t
)

DATASETS=(
    $DATASET
    # kitml
    # humanml3d
    # humanml3dwokit_plus_kitml
    # humanml3d_plus_kitml
)

DATA_REPS=(
    # cont_6d_plus_rifke
    cont_6d_plus_rifke_vels
)

REPS=(
    0
    1
    2
)

# Train & Evaluate
# Perform multiple repetitions of the same experiment, varying the split seed (for cross-validation)

for REP in ${REPS[@]}; do
    for TEXT_MODEL in ${TEXT_MODELS[@]}; do
        for MOTION_MODEL in ${MOTION_MODELS[@]}; do
            for LOSS in ${LOSSES[@]}; do
                for DATASET in ${DATASETS[@]}; do
                    for DATA_REP in ${DATA_REPS[@]}; do
                        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE HYDRA_FULL_ERROR=1 conda run --no-capture-output -n t2m_tmr python train.py -m model="$MODEL" data="$DATASET" data_rep="$DATA_REP" model/motion_encoder="$MOTION_MODEL" model/text_encoder="$TEXT_MODEL" model/contrastive_loss="$LOSS" ++rep=$REP
                    done
                done
            done
        done
    done
done
# HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/run-$REP --debug 

