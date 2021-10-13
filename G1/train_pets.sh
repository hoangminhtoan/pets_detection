#datetime training: 20211013 retrain yolov4 with 500k iters
./darknet detector train \
backup/20211013_yolov4_3_classes/pets.data \
backup/20211013_yolov4_3_classes/yolov4-custom-3-classes.cfg \
backup/20211010_yolov4_3_classes/yolov4-custom-3-classes_final.weights -clear