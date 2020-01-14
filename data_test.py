from selenium import webdriver
import os
import cv2
import xpinyin
from models import *

import numpy
import shutil


def find_photo(target):
    if ' ' in target or target.encode('UTF-8').isalpha():
        name = target
    else:
        pinyin = xpinyin.Pinyin()
        name = ' '.join(pinyin.get_pinyin(target).split('-'))
    path = 'test_data/' + name + '/'

    if not os.path.exists(path):  # 检查保存路径
        os.makedirs(path)
    web = webdriver.Chrome()
    web.get('http://iamge.baidu.com')
    web.find_element_by_id('kw').send_keys(target)
    web.find_element_by_class_name('s_search').click()

    for i in range(1, 20):
        web.find_element_by_xpath('//*[@id="imgid"]/div/ul/li[' + str(i) + ']/div/a/img').click()
        web.switch_to.window(web.window_handles[-1])
        web.save_screenshot(str(i) + '.png')
        img = cv2.imread(str(i) + '.png', cv2.IMREAD_COLOR)
        cv2.imwrite(str(i) + '.png', img[64:400, 0:700])  # 裁剪截图
        shutil.move(str(i) + '.png', path + str(i) + '.png')
        web.close()
        web.switch_to.window(web.window_handles[-1])
    web.quit()


def get_data(photo, name):
    img = cv2.imread(photo)

    dets = detector(img, 1)
    for index, face in enumerate(dets):
        shape = shape_predictor(img, face)
        face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
        vector = list(face_descriptor)
        length = numpy.linalg.norm(numpy.array(face_descriptor))
        info = {'name': name, 'vector': vector, 'length': length}
        main_sheet.insert_one(info)


def suit(target):
    if ' ' in target or target.encode('UTF-8').isalpha():
        name = target
    else:
        pinyin = xpinyin.Pinyin()
        name = ' '.join(pinyin.get_pinyin(target).split('-'))

    if len([x for x in main_sheet.find({'name': name})]):
        standard = [numpy.array(x['vector']) for x in main_sheet.find({'name': name})]
        result = []
        for file in os.listdir('test_data/' + name):
            img = cv2.imread('test_data/' + name + '/' + file, cv2.IMREAD_COLOR)
            dets = detector(img, 1)
            if len(dets) != 1:
                os.remove('test_data/' + name + '/' + file)
                continue
            else:
                for index, face in enumerate(dets):
                    vector = list(face_rec_model.compute_face_descriptor(img, shape_predictor(img, face)))
                    result.append(numpy.linalg.norm(numpy.array(vector) - standard))

        return len([x for x in result if x < 0.6]) / len(result)  # 返回成功率

    else:  # 数据库中没有该数据时则直接提示
        print('no data！！！')
        return 0


targets = ['彭于晏']
result = []
for i in targets:
    find_photo(i)
    result.append({i: suit(i)})
print(result)
