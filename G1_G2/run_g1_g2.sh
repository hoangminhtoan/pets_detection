declare -a FP_ArrayVideos=(
    ""
)
echo "Number of testing video is : ${#FP_ArrayVideos[@]}"
declare -i counter=0

CUDA_VISIBLE_DEVICES=0 python run_gen_FA_data.py \
    
for output_video in ${P_ArrayVideos[@]}; do
    ((++counter))
    echo "Testing video ${counter} / ${#FP_ArrayVideos[@]}"
    CUDA_VISIBLE_DEVICES=0 python run_g1_g2.py \
    --source_url $output_video --input_type video \
    --conf_thres 0.5 --iou_thres 0.45 --half \
    --weight_g1 weights/20211005_yolov4_tiny/yolov4-tiny-pets_human_best.weights \
    --config_file weights/20211005_yolov4_tiny/yolov4-tiny-pets_human.cfg \
    --data_file weights/20211005_yolov4_tiny/pets.data \
    --names_file weights/20211005_yolov4_tiny/pets.names \
    --class 0 1 \
    --weight_cloud weights/20211007_5s/best.pt \
done