import dlib
import cv2
import numpy as np


# 模型及初始化
predictor_path = "models/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "models/dlib_face_recognition_resnet_model_v1.dat"
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

test1 = cv2.imread('4.jpg')
test2 = cv2.imread('5.jpg')
test3 = cv2.imread('6.jpg')
dets1 = detector(test1, 1)
dets2 = detector(test2, 1)
dets3 = detector(test3, 1)

x = []
for index, face in enumerate(dets1):
    shape = shape_predictor(test1, face)
    face_descriptor = face_rec_model.compute_face_descriptor(test1, shape)  # 计算人脸的128维的向量
    x.append(face_descriptor)

for index, face in enumerate(dets2):
    shape = shape_predictor(test2, face)
    for i, pt in enumerate(shape.parts()):
        pt_pos = (pt.x, pt.y)
    face_descriptor = face_rec_model.compute_face_descriptor(test2, shape)  # 计算人脸的128维的向量
    x.append(face_descriptor)

for index, face in enumerate(dets3):
    shape = shape_predictor(test3, face)
    for i, pt in enumerate(shape.parts()):
        pt_pos = (pt.x, pt.y)
    face_descriptor = face_rec_model.compute_face_descriptor(test3, shape)  # 计算人脸的128维的向量
    x.append(face_descriptor)

distance = 0
for i in range(len(x[0])):
    distance += (x[1][i] - x[2][i])**2
print(np.sqrt(distance))

