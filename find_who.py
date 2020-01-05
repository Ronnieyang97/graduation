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
        for i in face_descriptor:
            target.append(i)
    global data
    temp = []
    if target:
        for name, vector in data:
            if np.linalg.norm(np.array(target) - np.array(vector)) < 0.6:
                temp.append([name, np.linalg.norm(np.array(target) - np.array(vector))])
            else:
                continue
        if temp:
            similar = ['', 0.6]
            for i in temp:
                if i[1] < similar[1]:
                    similar = i
            cv2.imwrite('photos/find_who/result/' + similar[0] + photo, img)
        else:
            cv2.imwrite('photos/find_who/result/' + 'unpaired' + photo, img)
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

