declare -a P_ArrayVideos=(
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P001.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P002.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P003.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P004.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P005.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P006.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P007.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P008.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P009.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P010.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P011.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P012.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P013.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P015.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P016.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P017.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P018.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P019.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P020.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P021.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P022.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P023.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P024.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P025.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P026.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P030.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P031.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P032.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P033.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P034.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P035.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P036.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P037.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P038.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P039.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P040.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P042.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P043.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P044.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P045.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Positive/Pet_Pub_P046.mp4"
)

# run v4
echo "Number of testing video is : ${#P_ArrayVideos[@]}"
declare -i counter=0
# Note
# You have to make directory "results/20210727_608/frames/" first
#for output_video in ${P_ArrayVideos[@]}; do
#    ((++counter))
#    echo "Testing video ${counter} / ${#P_ArrayVideos[@]}"
#    IFS='/' read -r -a tmps <<< "$output_video"
#    video_name=${tmps[-1]}
#    ./darknet detector demo \
#    /content/drive/MyDrive/Personal_Docs/ToanHM/VINBRAINS/Projects/SmartCity/Pets/Source_Code/pets_darknet/backup/20210901_yolov4_tiny/pets.data \
#    /content/drive/MyDrive/Personal_Docs/ToanHM/VINBRAINS/Projects/SmartCity/Pets/Source_Code/pets_darknet/backup/20210901_yolov4_tiny/yolov4-tiny-pets.cfg \
#    /content/drive/MyDrive/Personal_Docs/ToanHM/VINBRAINS/Projects/SmartCity/Pets/Source_Code/pets_darknet/backup/20210901_yolov4_tiny/yolov4-tiny-pets_best.weights \
#    -dont_show -thresh 0.6 \
#    -ext_output $output_video \
#    -prefix /content/drive/MyDrive/Personal_Docs/ToanHM/VINBRAINS/Projects/SmartCity/Pets/Source_Code/pets_darknet/results/20210906/frames/$video_name \
#    -out_filename /content/drive/MyDrive/Personal_Docs/ToanHM/VINBRAINS/Projects/SmartCity/Pets/Source_Code/pets_darknet/results/20210906/$video_name
#done

path="20211005_yolov4_tiny"

for output_video in ${P_ArrayVideos[@]}; do
    ((++counter))
    echo "Testing video ${counter} / ${#P_ArrayVideos[@]}"
    CUDA_VISIBLE_DEVICES=0 python run_demo_darknet.py \
    --source_url $output_video --input video \
    --thresh 0.5 \
    --weights backup/$path/yolov4-tiny-pets_human_best.weights \
    --config_file backup/$path/yolov4-tiny-pets_human.cfg \
    --data_file backup/$path/pets.data
done