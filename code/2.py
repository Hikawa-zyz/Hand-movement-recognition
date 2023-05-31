import cv2
import mediapipe as mp
import pandas as pd
import os


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
                keypoints_data.append([label] + keypoints)

    return keypoints_data

#folders = ['VT1\image\hou (2)', 'VT1\image\qian (3)', 'VT1\image\you', 'VT1\image\zuo','VT1\image\stop (3)','VT1\image\anquan','VT1\image\good','VT1\image\hello','VT1\image\ok','VT1\image\you and i']
#folders = [r'VT1\image2\hou', r'VT1\image2\qian', r'VT1\image2\you', r'VT1\image2\zuo',r'VT1\image2\stop',r'VT1\image2\bixin',r'VT1\image2\good',r'VT1\image2\hello',r'VT1\image2\ok',r'VT1\image2\bad']
folders = [r'VT1\image3\hou', r'VT1\image3\qian', r'VT1\image3\you', r'VT1\image3\zuo',r'VT1\image3\stop',r'VT1\image3\yeah',r'VT1\image3\good',r'VT1\image3\hello',r'VT1\image3\ok',r'VT1\image3\bad']

labels = ['0', '1', '2', '3','4','5','6','7','8','9']

all_keypoints = []
for folder, label in zip(folders, labels):
    keypoints = process_image_folder(folder, label)
    all_keypoints.extend(keypoints)

columns = ['label']
for i in range(1, 22):
    columns += [f'x{i}', f'y{i}']

keypoints_df = pd.DataFrame(all_keypoints, columns=columns)
keypoints_df.to_csv('VT1\data2\keypoints3.csv', index=False)
