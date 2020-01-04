import os
import sqlite3
import dlib
import cv2


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
    db = sqlite3.connect('face.db')
    c = db.cursor()
    c.execute('''insert into person (name, face_data)\
    values(?, ?);''', [name, temp])
    db.commit()
    db.close()


# 模型及初始化
predictor_path = "models/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "models/dlib_face_recognition_resnet_model_v1.dat"
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

photos = [file for file in os.listdir('add') if file.endswith('.jpg')]
for photo in photos:
    get_data(photo)
    print(photo, ' Done!')
