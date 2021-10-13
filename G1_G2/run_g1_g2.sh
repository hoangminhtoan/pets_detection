declare -a FP_ArrayVideos=(
    "/media/tuan-ng/39d94ede-b88e-49af-961d-6c1b096a14ea/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P022.mp4"
)

echo "Number of testing video is : ${#FP_ArrayVideos[@]}"
declare -i counter=0
    
for output_video in ${FP_ArrayVideos[@]}; do
    ((++counter))
    echo "Testing video ${counter} / ${#FP_ArrayVideos[@]}"
    CUDA_VISIBLE_DEVICES=0 python run_g1_g2.py \
    --source_url $output_video \
    --conf_thres 0.5 --iou_thres 0.45 --half \
    --weight_g1 weights/20211005_yolov4_tiny/yolov4-tiny-pets_human_best.weights \
    --config_file weights/20211005_yolov4_tiny/yolov4-tiny-pets_human.cfg \
    --data_file weights/20211005_yolov4_tiny/pets.data \
    --names_file weights/20211005_yolov4_tiny/pets.names \
    --weight_cloud weights/20211007_yolov5s/best.pt
done