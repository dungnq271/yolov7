#!/bin/zsh

python train_aux.py \
        --workers 1 \
        --device 0 \
        --batch-size 1 \
        --epochs 50 \
        --data data/face.yaml \
        --img 640 640 \
        --cfg cfg/training/yolov7-custom.yaml \
        --name yolov7-face-training \
        --hyp data/hyp.scratch.p6.yaml \
        --weights ./weights/yolov7-e6e.pt \
        --nosave
