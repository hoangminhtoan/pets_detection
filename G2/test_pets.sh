CUDA_VISIBLE_DEVICES=0 python test.py --task test \
--img 608 --batch 32 \
--conf 0.45 --iou 0.45 \
--data data/pets.yaml \
--weights runs/train/pets/exp_20210825_5s/weights/best.pt