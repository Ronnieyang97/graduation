import cv2
import os
import numpy as np
from models import *


def find_person(photo):
    img = cv2.imread('photos/find_who/' + photo)
    dets = detector(img, 1)
    target = []
    for index, face in enumerate(dets):
        shape = shape_predictor(img, face)
        face_descriptor = face_rec_model.compute_face_descriptor(img, shape)  # 计算人脸的128维的向量
        target.append([i for i in face_descriptor])
    global data
    result = []
    if target:  # 识别出的人脸
        for face in target:  # 对每个人脸进行比对
            temp = []
            for name, vector in data:
                if np.linalg.norm(np.array(face) - np.array(vector)) < 0.6:
                    temp.append([name, np.linalg.norm(np.array(face) - np.array(vector))])
                else:
                    continue
            if temp:  # 在阈值0.6的条件下，找出相似度最高的选项
                similar = ['', 0.6]
                for i in temp:
                    if i[1] < similar[1]:
                        similar = i
                result.append(similar[0])
            else:
                result.append('unpaired')
        print(result)
        cv2.imwrite('photos/find_who/result/' + '-'.join(result) + photo, img)
    else:
        cv2.imwrite('photos/find_who/result/' + 'unrecognized' + photo, img)


files = [file for file in os.listdir('photos/find_who') if file.endswith('.jpg')]


for file in files:
    find_person(file)
    print(file + '  Done!')

