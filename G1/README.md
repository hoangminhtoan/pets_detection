# YOLOVv4 for PET DETECTION

## Table of Content
* [Compile opencv lib](#compile-opencv-lib)
* [Compile darknet](#compile-darknet)
* [Training](#training)
* [Testing](#testing)


### Compile opencv lib
1. check opencv 4.5.2
2. [tutorial](#https://cv-tricks.com/installation/opencv-4-1-ubuntu18-04/)

### Compile darknet
1. [tutorial](#https://github.com/AlexeyAB/darknet)


### Training
1. Put all images and annotation file in the same folder
2. Modify config file (e.g. yolov4-tiny-custom.cfg), data file (e.g. *.data)
```
check out example in backup/20210901_yolov4_tiny/
```
3. Modify ```train_pets.sh``` then run ```bash train_pets.sh```

### Testing
1. Modify the config file for testing
2. Modify the ```run_darknet_pets_<negative/positive>.sh```   
  2.1. Change data path to video
  2.2. Change params meters for input network
3. Run ```bash run_darknet_pets<negative/positive>.sh```




