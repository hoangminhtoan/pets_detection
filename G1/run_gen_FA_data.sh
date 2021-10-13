# Add full directory path to video
declare -a P_ArrayVideos=(
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P001.mp4"
)

path="20211005_yolov4_tiny" #change folder name
echo "Number of testing video is : ${#fP_ArrayVideos[@]}"
declare -i counter=0

for output_video in ${FP_ArrayVideos[@]}; do
    ((++counter))
    echo "Testing video ${counter} / ${#FP_ArrayVideos[@]}"
    CUDA_VISIBLE_DEVICES=0 python run_gen_FA_data.py \
    --source_url $output_video --input_type video \
    --conf_thres 0.5 --iou_thres 0.45 \
    --weight_file backup/$path/yolov4-tiny-pets_human_best.weights \
    --config_file backup/$path/yolov4-tiny-pets_human.cfg \
    --data_file backup/$path/pets.data \
    --names_file backup/$path/pets.name
done