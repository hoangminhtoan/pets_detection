CUDA_VISIBLE_DEVICES=0 python train.py \
--img 608 608 --batch 40 --epochs 200 \
--data data/pets.yaml \
--cfg models/yolov5s_pets.yaml \
--hyp data/hyp.finetune_pets.yaml \
--project runs/train/pets \
--weights weights/yolov5s.pt \
--recipe ../recipes/yolov5s.pruned.md