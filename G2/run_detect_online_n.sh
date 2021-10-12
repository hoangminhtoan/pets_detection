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
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N010.mp4"
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
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N030.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N032.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N033.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N034.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N035.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N036.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N037.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N038.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N039.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N040.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N058.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N059.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N060.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N061.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N062.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N063.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N064.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N065.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N066.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N067.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N068.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N069.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N070.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N071.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N072.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N073.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N074.mp4"
"/home/vinbrain/Workspace/Datasets/PulicTest_Pets/Negative/Pet_Pub_N075.mp4"
)

echo "Number of testing video is : ${#N_ArrayVideos[@]}"
declare -i counter=0

# No classififer
for output_video in ${N_ArrayVideos[@]}; do
    ((++counter))
    echo "Testing video ${counter} / ${#N_ArrayVideos[@]}"
    CUDA_VISIBLE_DEVICES=0 python detect_online_0610.py \
    --img 640 --conf 0.6 --iou 0.45 --half\
    --class 0 1 \
    --weights weights/20210930_5s/best.pt \
    --input video --source $output_video
done

# With classifier
#for output_video in ${N_ArrayVideos[@]}; do
#    ((++counter))
#    echo "Testing video ${counter} / ${#N_ArrayVideos[@]}"
#    CUDA_VISIBLE_DEVICES=0 python detect_classify_online_0610.py \
#    --img 608 --conf 0.6 --iou 0.45 --half\
#    --class 0 1 \
#    --weights weights/20210920_5s/best.pt \
#    --input video --source $output_video
#done