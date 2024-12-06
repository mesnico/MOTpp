CUDA_DEVICE=1

RUNS_DIR=(
    runs/lr=5e-05/model=t2m/data=humanml3d_plus_kitml/length_regressor=None/loss=info-nce-cross-consistent-kldiv-40-100/motion=movit-v2-factenc-bodyparts-timemask-uniformsample-200frames-ff1024-4heads-4layers/text=actor-style-encoder/data_rep=cont_6d_plus_rifke_vels/space-dim=256/rep=0
    runs/lr=5e-05/model=t2m/data=humanml3d_plus_kitml/length_regressor=None/loss=info-nce-with-filtering/motion=actor-style-encoder/text=actor-style-encoder/data_rep=cont_6d_plus_rifke_vels/space-dim=256/rep=0
    runs/lr=5e-05/model=t2m/data=humanml3d_plus_kitml/length_regressor=None/loss=info-nce/motion=movit-bodyparts-timemask-uniformsample-200frames/text=clip/data_rep=cont_6d_plus_rifke_vels/space-dim=256/rep=0
)

for RUN in "${RUNS_DIR[@]}"
do
    echo $RUN
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE HYDRA_FULL_ERROR=1 conda run --no-capture-output -n t2m_tmr python retrieval_ranks.py ckpt=latest data=humanml3d ++skip_already_done=False ++lengths_threshold=null run_dir="\"$RUN\""
done