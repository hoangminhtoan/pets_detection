CUDA_VISIBLE_DEVICES=0 python test.py --task test \
--img 800 --batch 32 \
--conf 0.25 --iou 0.45 \
--data data/pets.yaml \
--weights runs/train/exp/weights/best.pt