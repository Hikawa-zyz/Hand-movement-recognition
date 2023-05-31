import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 图片文件路径
img_folder_path = 'VT1\\image6\\zuo'
# 输出图片文件夹路径
output_folder_path = 'VT1\\output_images'

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

image_files = os.listdir(img_folder_path)

for image_file in image_files:
    image_path = os.path.join(img_folder_path, image_file)
    img = cv2.imread(image_path)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(image)

        if result.multi_hand_landmarks:
            # 创建一个3D坐标系
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for hand_landmarks in result.multi_hand_landmarks:
                x = [landmark.x for landmark in hand_landmarks.landmark]
                y = [landmark.y for landmark in hand_landmarks.landmark]
                z = [landmark.z for landmark in hand_landmarks.landmark]

                # 在3D坐标系中绘制关键点
                ax.scatter(x, y, z)

                # 在3D坐标系中绘制关键点的连接线
                for connection in mp_hands.HAND_CONNECTIONS:
                    start_point = hand_landmarks.landmark[connection[0]]
                    end_point = hand_landmarks.landmark[connection[1]]
                    ax.plot([start_point.x, end_point.x], [start_point.y, end_point.y], [start_point.z, end_point.z], 'r')

                # 设置坐标轴标签
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')

            # 保存图片
            output_image_path = os.path.join(output_folder_path, f'3D_{image_file}')
            plt.savefig(output_image_path)
            plt.close(fig)  # 关闭图形以释放内存