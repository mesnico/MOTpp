#!/bin/bash

# read CUDA_DEVICE and DATASET from command line
while getopts d:s: flag
do
    case "${flag}" in
        d) CUDA_DEVICE=${OPTARG};;
        s) DATASET=${OPTARG};;
    esac
done

EXP_ROOT="./runs"
LR=0.0002  # "5e-05"

CKPT=latest # latest # best-rsum # best-rsum-t2m # best-rsum # latest
SKIP_DONE=true

MODELS=(
    rehamot_cosine
    rehamot_cps
)

TEXT_MODELS=(
    distilbert-linear
    # bert-lstm
    # actor-style-encoder
    # clip
    # clip-tpt-4
    # clip-tpt-2
    # clip-tpt-8
    # clip-hf
)

MOTION_MODELS=(
    transformer-rehamot
    # actor-style-encoder
    # movit-v2-factenc-bodyparts-timemask-uniformsample-200frames-ff1024-4heads-4layers
)

LOSSES=(
    soft-contrastive
    # info-nce-with-filtering
    # info-nce-cross-consistent-kldiv-0-1
    # info-nce-cross-consistent-kldiv-40-100
    # info-nce-cross-consistent-kldiv-80-140
    # info-nce-cross-consistent-kldiv-140-200
    # info-nce-cross-consistent-kldiv-500-600
    # info-nce-cross-consistent-kldiv-40-100-no-teacher-vs-t2t
    # info-nce-cross-consistent-kldiv-80-140-no-teacher-vs-t2t
)

DATASETS=(
    # $DATASET
    # kitml
    humanml3d
    humanml3d_plus_kitml
    # humanml3dwokit_plus_kitml
    # humanml3dwokit
)

DATA_REPS=(
    # cont_6d_plus_rifke
    cont_6d_plus_rifke_vels
)

LENGTH_THRESHOLDS=(
    null
    # 0.5
    # 1.0
    # 1.5
    # 2.0
)

REPS=(
    0
    1
    2
)

# transform into a string of comma-separated values
# IFS=,
# TEXT_MODELS="${TEXT_MODELS[*]}"
# MOTION_MODELS="${MOTION_MODELS[*]}"
# DATASETS="${DATASETS[*]}"
# LOSSES="${LOSSES[*]}"
# DATA_REPS="${DATA_REPS[*]}"

SPACE_DIM=256
TOTAL_EVALUATIONS=0
OK_EVALUATIONS=0

for REP in ${REPS[@]}; do
    for MODEL in ${MODELS[@]}; do
        for TEXT_MODEL in ${TEXT_MODELS[@]}; do
            for MOTION_MODEL in ${MOTION_MODELS[@]}; do
                for LOSS in ${LOSSES[@]}; do
                    for DATASET in ${DATASETS[@]}; do
                        for DATA_REP in ${DATA_REPS[@]}; do
                            EXP_PATH=${EXP_ROOT}/lr=${LR}/model=${MODEL}/data=${DATASET}/length_regressor=None/loss=${LOSS}/motion=${MOTION_MODEL}/text=${TEXT_MODEL}/data_rep=${DATA_REP}/space-dim=${SPACE_DIM}/rep=${REP}
                            if [ ! -d "$EXP_PATH" ]; then
                                echo "Experiment path $EXP_PATH does not exist... skipping"
                                continue
                            fi

                            echo "Evaluating $EXP_PATH"
                            if [ $? -eq 0 ]; then
                                OK_EVALUATIONS=$((OK_EVALUATIONS+1))
                            fi

                            # evaluate m2m retrieval
                            CUDA_VISIBLE_DEVICES=$CUDA_DEVICE HYDRA_FULL_ERROR=1 conda run --no-capture-output -n t2m_tmr python retrieval_m2m.py data=kitml run_dir="\"$EXP_PATH\"" ckpt=${CKPT} ++skip_already_done=False #${SKIP_DONE}

                            for THRESHOLD in ${LENGTH_THRESHOLDS[@]}; do
                                
                                # export checkpoints
                                # HYDRA_FULL_ERROR=1 conda run --no-capture-output -n t2m_tmr python extract.py run_dir="\"${EXP_PATH}\"" ckpt=${CKPT}
                                # # if there is an error, continue
                                # if [ $? -ne 0 ]; then
                                #     echo "Error in exporting checkpoints (maybe not existing?)... skipping"
                                #     continue
                                # fi
                                # evaluate on kit
                                # export checkpoints
                                # HYDRA_FULL_ERROR=1 conda run --no-capture-output -n t2m_tmr python extract.py run_dir="\"${EXP_PATH}\"" ckpt=${CKPT}
                                # # if there is an error, continue
                                # if [ $? -ne 0 ]; then
                                #     echo "Error in exporting checkpoints (maybe not existing?)... skipping"
                                #     continue
                                # fi
                                # evaluate on kit
                                CUDA_VISIBLE_DEVICES=$CUDA_DEVICE HYDRA_FULL_ERROR=1 conda run --no-capture-output -n t2m_tmr python retrieval.py data=kitml run_dir="\"$EXP_PATH\"" ckpt=${CKPT} ++skip_already_done=${SKIP_DONE} ++lengths_threshold=${THRESHOLD}
                                # if evaluation successfull, increment OK_EVALUATIONS
                                if [ $? -eq 0 ]; then
                                    OK_EVALUATIONS=$((OK_EVALUATIONS+1))
                                fi

                                # evaluate on humanml
                                CUDA_VISIBLE_DEVICES=$CUDA_DEVICE HYDRA_FULL_ERROR=1 conda run --no-capture-output -n t2m_tmr python retrieval.py data=humanml3d run_dir="\"$EXP_PATH\"" ckpt=${CKPT} ++skip_already_done=${SKIP_DONE} ++lengths_threshold=${THRESHOLD}
                                # if evaluation successfull, increment OK_EVALUATIONS
                                if [ $? -eq 0 ]; then
                                    OK_EVALUATIONS=$((OK_EVALUATIONS+1))
                                fi

                                # evaluate on humanml without kitml (wrong)
                                # CUDA_VISIBLE_DEVICES=$CUDA_DEVICE HYDRA_FULL_ERROR=1 conda run --no-capture-output -n t2m_tmr python retrieval.py data=humanml3dwokit run_dir="\"$EXP_PATH\"" ckpt=${CKPT} ++skip_already_done=${SKIP_DONE} ++lengths_threshold=${THRESHOLD}
                                # # if evaluation successfull, increment OK_EVALUATIONS
                                # if [ $? -eq 0 ]; then
                                #     OK_EVALUATIONS=$((OK_EVALUATIONS+1))
                                # fi

                                # evaluate on humanml without kitml (correct)
                                CUDA_VISIBLE_DEVICES=$CUDA_DEVICE HYDRA_FULL_ERROR=1 conda run --no-capture-output -n t2m_tmr python retrieval.py data=humanml3d_test_wokit run_dir="\"$EXP_PATH\"" ckpt=${CKPT} ++skip_already_done=${SKIP_DONE} ++lengths_threshold=${THRESHOLD}
                                # if evaluation successfull, increment OK_EVALUATIONS
                                if [ $? -eq 0 ]; then
                                    OK_EVALUATIONS=$((OK_EVALUATIONS+1))
                                fi

                                # evaluate on kitml without humanml (correct)
                                CUDA_VISIBLE_DEVICES=$CUDA_DEVICE HYDRA_FULL_ERROR=1 conda run --no-capture-output -n t2m_tmr python retrieval.py data=kitml_test_wohumanml3d run_dir="\"$EXP_PATH\"" ckpt=${CKPT} ++skip_already_done=${SKIP_DONE} ++lengths_threshold=${THRESHOLD}
                                # if evaluation successfull, increment OK_EVALUATIONS
                                if [ $? -eq 0 ]; then
                                    OK_EVALUATIONS=$((OK_EVALUATIONS+1))
                                fi

                                # remove processed checkpoints to free up space
                                # rm -rf ${EXP_PATH}/${CKPT}_weights

                                TOTAL_EVALUATIONS=$((TOTAL_EVALUATIONS+5))
                                echo "************* $OK_EVALUATIONS / $TOTAL_EVALUATIONS successful *************"
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

