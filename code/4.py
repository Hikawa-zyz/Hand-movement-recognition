import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 对视频进行处理
video_path = 'VT1\\video4\\you.mp4'
cap = cv2.VideoCapture(video_path)
output_path = 'VT1\\video4\\out_video1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编解码器
fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 帧尺寸
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 对图像进行处理
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result = hands.process(image)

        # 创建一个白色背景的图像
        white_image = np.ones_like(frame) * 255

        # 添加坐标轴
        start_point_x = (0, frame_size[1]//2)
        end_point_x = (frame_size[0], frame_size[1]//2)
        start_point_y = (frame_size[0]//2, 0)
        end_point_y = (frame_size[0]//2, frame_size[1])
        color = (0, 0, 0)
        thickness = 2
        white_image = cv2.line(white_image, start_point_x, end_point_x, color, thickness)
        white_image = cv2.line(white_image, start_point_y, end_point_y, color, thickness)

        # 在白色背景的图像上绘制关键点和连接线
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(white_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=5),
                                          mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2))

        out.write(white_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
