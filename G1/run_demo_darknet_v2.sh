path="20210901_yolov4_tiny"

CUDA_VISIBLE_DEVICES=0 python run_demo_darknet_v2.py \
    --path /home/vinbrain/Workspace/Datasets/PulicTest_Pets/ \
    --weights backup/$path/yolov4-tiny.weights \
    --config_file backup/$path/yolov4-tiny-pets.cfg \
    --data_file backup/$path/pets.data