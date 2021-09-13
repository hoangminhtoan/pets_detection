#210906
declare -a FP_ArrayImgs=(
"/content/drive/MyDrive/Personal_Docs/ToanHM/VINBRAINS/Projects/SmartCity/Pets/Datasets/FalseAlarm/pets_210912/"
)

declare -a FP_ArrayVideos=(
"/content/drive/MyDrive/Personal_Docs/ToanHM/VINBRAINS/Projects/SmartCity/Pets/Datasets/FalseAlarm/pets_210912/all_members_2.mp4"
)

echo "Number of testing video is : ${#FP_ArrayImgs[@]}"
declare -i counter=0
for output_video in ${FP_ArrayImgs[@]}; do
    ((++counter))
    echo "Testing image ${counter} / ${#FP_ArrayImgs[@]}"
    CUDA_VISIBLE_DEVICES=0 python ../detect_online_0610.py \
    --img 608 --conf 0.6 --iou 0.45 --half\
    --weights ../weights/20210825_5s/best.pt \
    --input image --source $output_video
done

# weight 
echo "Number of testing video is : ${#FP_ArrayVideos[@]}"
declare -i counter=0
for output_video in ${FP_ArrayVideos[@]}; do
    ((++counter))
    echo "Testing video ${counter} / ${#FP_ArrayVideos[@]}"
    CUDA_VISIBLE_DEVICES=0 python ../detect_online_0610.py \
    --img 608 --conf 0.6 --iou 0.45 --half\
    --weights ../weights/20210825_5s/best.pt \
    --input video --source $output_video
done