#!/bin/zsh
python align.py \
       --weight weights/pfld.onnx \
       --csv yolov7_face_test_20/answer.csv \
       --img-dir yolov7_face_test_20/cropped_face_imgs \
       --output-dir aligned_faces
