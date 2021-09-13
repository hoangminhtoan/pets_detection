declare -a P_ArrayImgs=(
    "/media/toanhoang/ToanMH_SE/Workspace/Datasets/HAR/Human_Detection/test_classify_human_final/test_public_pos"
)

declare -a N_ArrayImgs=(
    "/media/toanhoang/ToanMH_SE/Workspace/Datasets/HAR/Human_Detection/test_classify_human_final/test_public_neg"
)

for output_video in ${N_ArrayImgs[@]}; do
    CUDA_VISIBLE_DEVICES=0 python detect_human_0608.py \
    --img 608 --conf 0.5 --iou 0.45 \
    --classes 0 \
    --weights weights/20210608/best.pt \
    --source $output_video
done