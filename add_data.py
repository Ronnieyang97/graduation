import os
import cv2
from models import *


def get_data(photo):
    img = cv2.imread('add/' + photo)
    name = photo.split('.jpg')[0]
    dets = detector(img, 1)
    temp = ''
    for index, face in enumerate(dets):
        shape = shape_predictor(img, face)
        face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
        for i in face_descriptor:  # 将128维向量转为字符串保存
            temp += (str(i) + '/')
    cursor.execute('''insert into person (name, face_data) values (?, ?);''', [name, temp])
    db.commit()
    db.close()


photos = [file for file in os.listdir('add') if file.endswith('.jpg')]
for photo in photos:
    get_data(photo)
    print(photo, ' Done!')
