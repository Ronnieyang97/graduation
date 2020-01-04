import os
import dlib
import cv2
import time


path_glasses = 'photos/glasses'
path_normal = 'photos/normal'
path_shelter = 'photos/shelter'
path_side = 'photos/side'


def detect_face(path):
    photos = [file for file in os.listdir(path) if file.endswith('.jpg')]
    detector = dlib.get_frontal_face_detector()  # 初始化人脸检测器
    fault = []
    # window = dlib.image_window()
    for photo in photos:
        target = cv2.imread(path + '/' + photo)
        dets = detector(target, 1)  # 获取目标的特征点,第二个参数设置为1，否则当出现多张人脸时会仅显示三处
        if not dets:
            fault.append(photo)
            continue
        '''window.clear_overlay()  # 清除显示窗
        window.set_image(target)  # 加载图片
        window.add_overlay(dets)  # 框选人脸'''
        for k, d in enumerate(dets):
            cv2.rectangle(target, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 2)
            # 参数设置： 目标文件，（左，上），（右，下），颜色，边框粗细
        dlib.save_image(target, path + '/result/' + photo)
    print('over')
    print(fault)


detect_face(path_side)


'''
测试normal类型的人脸时成功率基本为100%；
当人脸是倒置时无法正确识别；
侧脸，眼镜或部分遮挡时，68特征点被干扰，无法保证100%识别出人脸的位置
失败集中于：眼镜处于非完全带正的状态，干扰到其他特征点的识别，其他失败原因为外部强光干扰和肤色干扰

'''


