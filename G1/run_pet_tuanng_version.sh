path="20210901_yolov4_tiny"

CUDA_VISIBLE_DEVICES=0 python pet_tuanng_version.py \
    --path /home/vinbrain/Workspace/Datasets/PulicTest_Pets/ \
    --weights backup/$path/yolov4-tiny-pets_best.weights \
    --config_file backup/$path/yolov4-tiny-pets.cfg \
    --data_file backup/$path/pets.data