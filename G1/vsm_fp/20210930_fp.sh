source_dir="/home/vinbrain/Workspace/Datasets/False_Alarm/pets_210930/"
path="20211001_yolov4_tiny"

echo "Run Testing on Image"
CUDA_VISIBLE_DEVICES=0 python ../run_demo_darknet_fp.py \
--source_url $source_dir --input image \
--thresh 0.4 \
--weights ../backup/$path/yolov4-tiny-pets_final.weights \
--config_file ../backup/$path/yolov4-tiny-pets.cfg \
--data_file ../backup/$path/pets.data

#echo "Run Testing on Video"
#CUDA_VISIBLE_DEVICES=0 python ../run_demo_darknet_fp.py \
#--source_url $source_dir --input video \
#--thresh 0.25 \
#--weights ../backup/$path/yolov4-tiny-pets_final.weights \
#--config_file ../backup/$path/yolov4-tiny-pets.cfg \
#--data_file ../backup/$path/pets.data