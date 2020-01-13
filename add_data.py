import os
import cv2
from models import *
import numpy


def get_data(photo):
    img = cv2.imread('add/' + photo)
    if 'jpg' in photo:
        name = photo.split('/')[-1].split('.jpg')[0]
    elif 'png' in photo:
        name = photo.split('/')[-1].split('.png')[0]
    else:
        print('wrong format name')

    dets = detector(img, 1)
    for index, face in enumerate(dets):
        shape = shape_predictor(img, face)
        face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
        vector = list(face_descriptor)
        length = numpy.linalg.norm(numpy.array(face_descriptor))
        info = {'name': name, 'vector': vector, 'length': length}
        main_sheet.insert_one(info)


photos = [file for file in os.listdir('add')]
for photo in photos:
    get_data(photo)
    print(photo, ' Done!')
