this is readme file

本项目为毕业设计-基于python的人脸识别技术

detect_face_interference用于测试各种各种遮挡物以及角度对人脸识别的干扰情况；
相关测试用的图片存放在photos下，glasses、normal、shelter、side文件夹中

add_data用于向face.db中添加新的人脸数据，人脸图片须存放到add文件夹中

models中为预先加载、初始化模型和检测器，以及连接数据库

find_who用于识别图片文件中的若干人脸，待识别人脸存放在photos/find_who中，识别结果存放在其子文件夹result中，同时控制台也会输出识别的结果

detect_in_video用于识别摄像头实时和视频中的人像，视频文件须放到videos中