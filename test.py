import dlib
import cv2

test = dlib.load_rgb_image("test2.JPG")  # 读取图片将其转换为RGB数组形式
test2 = cv2.imread("test2.JPG")
detector = dlib.get_frontal_face_detector()  # 初始化人脸检测器
window = dlib.image_window()
dets = detector(test, 1)
print("Number of faces detected: {}".format(len(dets)))
for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))

window.clear_overlay()  # Remove all overlays from the image_window
window.set_image(test)  # Make the image_window display the given HOG detector’s filters
window.add_overlay(dets)  # Add a list of rectangles to the image_window


# dlib.hit_enter_to_continue()  # 使图像停留

for k, d in enumerate(dets):
    cv2.rectangle(test2, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 255), 2)
cv2.imwrite('result', test2)
