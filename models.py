import dlib
import sqlite3
import pymongo
# 模型及初始化
predictor_path = "models/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "models/dlib_face_recognition_resnet_model_v1.dat"
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

detector = dlib.get_frontal_face_detector()

connection = pymongo.MongoClient('localhost', 27017)
db = connection['graduation']
main_sheet = db['main']

