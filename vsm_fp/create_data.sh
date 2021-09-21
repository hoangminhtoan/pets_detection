#2109
declare -a FP_ArrayImgs=(
"/content/drive/MyDrive/Personal_Docs/ToanHM/VINBRAINS/Projects/SmartCity/Pets/Datasets/catsdogs/train/dogs"
"/content/drive/MyDrive/Personal_Docs/ToanHM/VINBRAINS/Projects/SmartCity/Pets/Datasets/catsdogs/train/cats"
)

declare -a FP_ArrayVideos=(
)

echo "Number of testing image folder is : ${#FP_ArrayImgs[@]}"
declare -i counter=0
for output_video in ${FP_ArrayImgs[@]}; do
    ((++counter))
    echo "Testing image ${counter} / ${#FP_ArrayImgs[@]}"
    CUDA_VISIBLE_DEVICES=0 python ../create_false_alarm_data.py \
    --img 608 --conf 0.5 --iou 0.45 --half\
    --weights ../weights/20210920_5s/best.pt \
    --input image --source $output_video
done

# weight 
#echo "Number of testing video is : ${#FP_ArrayVideos[@]}"
#declare -i counter=0
#for output_video in ${FP_ArrayVideos[@]}; do
#    ((++counter))
#    echo "Testing video ${counter} / ${#FP_ArrayVideos[@]}"
#    CUDA_VISIBLE_DEVICES=0 python ../create_false_alarm_data.py \
#    --img 608 --conf 0.6 --iou 0.45 --half\
#    --weights ../weights/20210825_5s/best.pt \
#    --input video --source $output_video
#done