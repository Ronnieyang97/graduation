import dlib
import cv2
import os
import sqlite3
import numpy as np


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


# 模型及初始化
predictor_path = "models/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "models/dlib_face_recognition_resnet_model_v1.dat"
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
files = [file for file in os.listdir('photos/find_who') if file.endswith('.jpg')]
db = sqlite3.connect('face.db')
cursor = db.cursor()
database = cursor.execute('''select * from person''')
data = []
for info in database:
    name = info[0]
    vector = [float(num) for num in info[1].split('/') if num]
    data.append([name, vector])

for file in files:
    find_person(file)
    print(file + '  Done!')

