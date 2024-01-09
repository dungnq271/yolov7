#!/bin/zsh
python detect_crop_image_save_bbox.py \
       --weights checkpoints/best.pt \
       --conf 0.25 \
       --iou-thres 0.5 \
       --img-size 640 \
       --source "public_test/images" \
       --project "yolov7_face_test_20" \
       --name  "annotated_imgs" \
       --cropped-dir "cropped_face_imgs" \
       --no-trace \
       --exist-ok \
       --nosave
