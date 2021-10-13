declare -a N_ArrayVideos=(
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N001.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N002.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N003.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N004.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N005.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N006.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N007.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N008.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N009.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N011.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N014.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N015.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N016.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N017.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N018.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N019.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N020.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N021.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N022.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N023.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N024.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N025.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N026.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N027.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N029.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N032.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N033.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N035.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N036.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N037.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N038.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N039.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N040.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N072.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N073.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N074.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N075.mp4"
)


# run v4
echo "Number of testing video is : ${#N_ArrayVideos[@]}"
declare -i counter=0
#for output_video in ${N_ArrayVideos[@]}; do
#    ((++counter))
#    echo "Testing video ${counter} / ${#N_ArrayVideos[@]}"
#    IFS='/' read -r -a tmps <<< "$output_video"
#    video_name=${tmps[-1]}
#    ./darknet detector demo \
#    backup/20210901_yolov4_tiny/pets.data \
#    backup/20210901_yolov4_tiny/yolov4-tiny-pets.cfg \
#    backup/20210901_yolov4_tiny/yolov4-tiny-pets_best.weights \
#    -dont_show -thresh 0.6 \
#    -ext_output $output_video \
#    -prefix results/20210906/frames/$video_name \
#    -out_filename results/20210906/$video_name
#done


path="20211005_yolov4_tiny"

for output_video in ${N_ArrayVideos[@]}; do
    ((++counter))
    echo "Testing video ${counter} / ${#N_ArrayVideos[@]}"
    CUDA_VISIBLE_DEVICES=0 python run_demo_darknet.py \
    --source_url $output_video --input video \
    --thresh 0.5 \
    --weights backup/$path/yolov4-tiny-pets_human_best.weights \
    --config_file backup/$path/yolov4-tiny-pets_human.cfg \
    --data_file backup/$path/pets.data
done