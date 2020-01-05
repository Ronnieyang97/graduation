import dlib
import sqlite3

# 模型及初始化
predictor_path = "models/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "models/dlib_face_recognition_resnet_model_v1.dat"
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

detector = dlib.get_frontal_face_detector()

db = sqlite3.connect('face.db')
cursor = db.cursor()
database = cursor.execute('''select * from person''')
data = []
for info in database:
    name = info[0]
    vector = [float(num) for num in info[1].split('/') if num]
    data.append([name, vector])
