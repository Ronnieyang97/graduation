import numpy
import cv2
from models import *


def find(frame):
    dets = detector(frame, 1)
    if not dets:
        return 'no face'
    else:
        target = []
        for index, face in enumerate(dets):
            shape = shape_predictor(frame, face)
            face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)  # 计算人脸的128维的向量
            target.append(list(face_descriptor))
        result = []
        if target:  # 识别出的人脸
            for face in target:  # 对每个人脸进行比对
                for face in target:  # 对每个人脸进行比对
                    length = numpy.linalg.norm(face)
                    temp = [[x['name'], x['vector']] for x in main_sheet.find({'length': {'$lt': length + 0.6,
                                                                                          '$gt': length - 0.6}})]
                    temp1 = []
                    for name, vector in temp:
                        if numpy.linalg.norm(face - numpy.array(vector)) < 0.6:
                            temp1.append([name, numpy.linalg.norm(face - numpy.array(vector))])
                    if temp1:  # 在阈值0.6的条件下，找出相似度最高的选项
                        similar = ['', 0.6]
                        for i in temp1:
                            if i[1] < similar[1]:
                                similar = i
                        result.append(similar[0])
                    else:
                        result.append('unpaired')
            return '-'.join(result)
        else:
            return 'unrecognized'


cap = cv2.VideoCapture(0)  # 获取摄像头
# 如果要实现视频的读取，就将cap = cv2.VideoCapture('test.avi');循环条件改为cap.isOpened()

while 1:
    ret, frame = cap.read()  # 得到摄像头的参数
    dets = detector(frame, 1)
    for index, face in enumerate(dets):  # 人脸框图
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)
    cv2.putText(frame, find(frame), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)  # 文本
    cv2.imshow('frame', cv2.cvtColor(frame, cv2.IMREAD_COLOR))

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 输入q结束程序
        break

cap.release()
cv2.destroyAllWindows()
