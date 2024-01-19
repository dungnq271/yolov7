import os
import os.path as osp
import math
import argparse
import ast

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import onnxruntime as ort


def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]

    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


def crop_face(img, rect):
    xmin, ymin, w, h = rect
    xmax, ymax = xmin + w, ymin + h
    org_face_roi = img[int(ymin):int(ymax), int(xmin):int(xmax)]

    if h > w:  # more upright face
        margin = h / 5
        ymin = 0 if ymin < margin else ymin - margin
    else:  # more horizontal face
        margin = w / 10
        xmin = 0 if xmin < margin else xmin - margin
        xmax += margin
    
    h, w = ymax - ymin, xmax - xmin
    if h > w:
        pad_w = (h - w) / 2
        xmin -= pad_w
        if xmin < 0:
            xmax = xmax + pad_w - xmin
            xmin = 0
        else:
            xmax += pad_w
    else:
        pad_h = (w - h) / 2
        ymin -= pad_h
        if ymin < 0:
            ymax = ymax + pad_h - ymin
            ymin = 0
        else:
            ymax += pad_h
        
    ymin, ymax, xmin, xmax = int(ymin), int(ymax), int(xmin), int(xmax)
    h, w = ymax - ymin, xmax - xmin
    assert abs(h-w) <= 1, print(w, h)

    if h > w:
        ymax -= 1
    elif h < w:
        xmax -= 1
    h, w = ymax - ymin, xmax - xmin
    
    face_roi = img[ymin:ymax, xmin:xmax]
    
    return org_face_roi, face_roi


def rotate_bound(image, angle):
    """
    From https://pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def align_face(img_raw, eye_1, eye_2):
    left_eye_x, left_eye_y = eye_1[0], eye_1[1]
    right_eye_x, right_eye_y = eye_2[0], eye_2[1]

    left_eye_center = (left_eye_x, left_eye_y)
    right_eye_center = (right_eye_x, right_eye_y)

    #----------------------
    #find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
        # print("rotate to clock direction")
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock
        # print("rotate to inverse clock direction")

    #----------------------
    a = euclidean_distance(left_eye_center, point_3rd)
    b = euclidean_distance(right_eye_center, point_3rd)
    c = euclidean_distance(right_eye_center, left_eye_center)

    cos_a = (b*b + c*c - a*a)/(2*b*c)
    angle = np.arccos(cos_a)

    angle = (angle * 180) / math.pi
    if direction == -1:
        angle = 90 - angle

    #--------------------
    #rotate image
    new_img = rotate_bound(img_raw, angle*direction)
    return new_img


def get_transforms():
    transform = transforms.Compose([
        transforms.Resize([112, 112]),
        transforms.ToTensor()        
    ])
    return transform


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def visualize_landmark(img, ldmk):
    annt_face_img = img.copy()
    print(annt_face_img.shape)
    h, w, _ = annt_face_img.shape
    
    # Radius of circle 
    radius = 1

    # Blue color in BGR 
    color = (255, 0, 0) 

    # Line thickness of 2 px 
    thickness = 3

    for ldm in ldmk:
        x = ldm[0] * w
        y = ldm[1] * h
        # Using cv2.circle() method 
        # Draw a circle with blue line borders of thickness of 2 px 
        annt_face_img = cv2.circle(annt_face_img, (int(x), int(y)), radius, color, thickness) 

    plt.imshow(annt_face_img)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str)
    parser.add_argument("--csv", type=str)
    parser.add_argument("--img-dir", type=str)
    parser.add_argument("--output-dir", type=str)            
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    labels = pd.read_csv(args.csv)
    if "face_file_name" not in labels.columns.tolist():
        labels["face_file_name"] = [str(i+1) + ".jpg" for i in range(len(labels))]
    
    img_dir = args.img_dir
    aligned_img_dir = args.output_dir
    os.makedirs(aligned_img_dir, exist_ok=True)
    
    ort_session_landmark = ort.InferenceSession(args.weight)
    transform = get_transforms()

    face_landmarks = {}

    for i, row in tqdm(labels.iterrows()):
        face_fn = row.face_file_name

        if face_fn in face_landmarks:
            continue
        img_fp = osp.join(img_dir, row.file_name)
        img_fn = osp.basename(img_fp)
        rect = ast.literal_eval(row.bbox)

        # load and crop
        img = cv2.imread(img_fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        org_face_img, face_img = crop_face(img, rect)

        # prepare befor extracting landmarks
        test_face = Image.fromarray(face_img)
        test_face = transform(test_face)
        #test_face = normalize(test_face)
        test_face.unsqueeze_(0)
    #     print(test_face.shape)

        # extracting landmarks
        ort_inputs = {ort_session_landmark.get_inputs()[0].name: to_numpy(test_face)}
        ort_outs = ort_session_landmark.run(None, ort_inputs)
        landmark = ort_outs[0]
        landmark = landmark.reshape(-1,2)

        if i == 0:
            visualize_landmark(face_img, landmark)

        # get the eyes
        eye_1 = landmark[36:42].mean(axis=0) 
        eye_2 = landmark[42:48].mean(axis=0)

        # align the face
        aligned_face_img = face_img.copy()
        aligned_face_img = align_face(aligned_face_img, eye_1, eye_2)

        # save the results
        face_fp = osp.join(aligned_img_dir, face_fn)
        face_landmarks[face_fn] = landmark.tolist()
        Image.fromarray(aligned_face_img).save(face_fp)
