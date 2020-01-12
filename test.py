from selenium import webdriver
from selenium.webdriver.support.expected_conditions import visibility_of_element_located
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
import shutil
import os
import cv2

'''web = webdriver.Chrome()
web.get('http://iamge.baidu.com')
target = '姜文'
path = 'C://Users/Ronnie/PycharmProjects/graduation/test_data/'
web.find_element_by_id('kw').send_keys(target)
web.find_element_by_class_name('s_search').click()

if not os.path.exists(path + target):
    os.makedirs(path + target)

for i in range(1, 3):
    web.find_element_by_xpath('//*[@id="imgid"]/div/ul/li[' + str(i) + ']/div/a/img').click()
    web.switch_to.window(web.window_handles[-1])
    web.save_screenshot(str(i) + '.png')
    #WebDriverWait(web, 10, 0.5).until(visibility_of_element_located((By.CLASS_NAME, 'down')))  # 显性等待
    #web.find_element_by_class_name('down').click()
    web.close()
    web.switch_to.window(web.window_handles[-1])

files = os.listdir()
print(files)
for file in files:
    shutil.move('C://users/ronnie/downloads/' + file, path+target)'''



img = cv2.imread('1.png')
cv2.imwrite('new1.png', img[64:700, 0:900])
