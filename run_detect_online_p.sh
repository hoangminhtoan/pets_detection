declare -a P_ArrayVideos=(
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P001.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P002.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P003.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P004.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P005.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P006.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P007.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P008.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P009.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P010.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P011.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P012.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P013.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P014.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P015.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P016.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P017.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P018.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P019.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P020.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P021.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P022.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P023.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P024.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P025.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P026.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P027.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P028.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P029.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P030.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P031.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P032.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P033.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P034.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P035.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P036.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P037.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P038.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P039.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P040.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P041.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P042.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P043.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P044.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P045.mp4"
"/media/toanhoang/Data/Workspace/Source_Code/pets_detection_darknet/data/Positive/Pet_Pub_P046.mp4"
)

echo "Number of testing video is : ${#P_ArrayVideos[@]}"
declare -i counter=0

# No classififer
for output_video in ${P_ArrayVideos[@]}; do
    ((++counter))
    echo "Testing video ${counter} / ${#P_ArrayVideos[@]}"
    CUDA_VISIBLE_DEVICES=0 python detect_online_0610.py \
    --img 608 --conf 0.6 --iou 0.45 --half\
    --class 0 1 \
    --weights weights/20210920_5s/best.pt \
    --input video --source $output_video
done

# With classifier
#for output_video in ${P_ArrayVideos[@]}; do
#    ((++counter))
#    echo "Testing video ${counter} / ${#P_ArrayVideos[@]}"
#    CUDA_VISIBLE_DEVICES=0 python detect_classify_online_0610.py \
#    --img 608 --conf 0.6 --iou 0.45 --half\
#    --class 0 1 \
#    --weights weights/20210920_5s/best.pt \
#    --input video --source $output_video
#done