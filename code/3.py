import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np




mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def process_image_folder(folder, label):
    keypoints_data = []
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)

        # 使用mediapipe检测手部关键点
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:  # 如果检测到手
            for hand_landmarks in results.multi_hand_landmarks:
                keypoints = []
                for point in hand_landmarks.landmark:
                    keypoints.extend([point.x, point.y])
                
                # 将关键点数据转换为numpy数组以便进行数学运算
                keypoints_np = np.array(keypoints).reshape(-1, 2)
                
                # 中心化处理
                keypoints_np_centered = keypoints_np - np.mean(keypoints_np, axis=0)
                
                # 将处理后的关键点数据重新塑形为列表形式并添加到keypoints_data中
                keypoints_data.append([label] + keypoints_np_centered.flatten().tolist())

    return keypoints_data

folders = [r'VT1\image6\hou', r'VT1\image6\qian', r'VT1\image6\you', r'VT1\image6\zuo',r'VT1\image6\stop',r'VT1\image6\yeah',r'VT1\image6\good',r'VT1\image6\hello',r'VT1\image6\ok',r'VT1\image6\bad']

labels = ['0', '1', '2', '3','4','5','6','7','8','9']

all_keypoints = []
for folder, label in zip(folders, labels):
    keypoints = process_image_folder(folder, label)
    all_keypoints.extend(keypoints)

columns = ['label']
for i in range(1, 22):
    columns += [f'x{i}', f'y{i}']

keypoints_df = pd.DataFrame(all_keypoints, columns=columns)
keypoints_df.to_csv('VT1\data2\keypoints8.csv', index=False)
