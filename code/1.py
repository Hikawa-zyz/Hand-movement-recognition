import os
import cv2

# 视频文件夹路径
video_folder = 'VT1\\video3'

# 输出图片根文件夹路径
output_root_folder = 'VT1\\image4'

# 获取视频文件夹中所有的视频文件路径
video_paths = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]

for video_path in video_paths:
    # 根据视频文件名创建输出文件夹，如果不存在
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(output_root_folder, video_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取视频文件
    cap = cv2.VideoCapture(video_path)

    frame_number = 0

    while cap.isOpened():
        # 读取视频帧
        ret, frame = cap.read()

        # 如果视频结束，跳出循环
        if not ret:
            break

        # 保存图片
        output_filename = os.path.join(output_folder, f'{video_name}_frame{frame_number:04d}.png')
        cv2.imwrite(output_filename, frame)
        frame_number += 1

    # 释放资源
    cap.release()

    print(f'Saved {frame_number} frames from {video_path} to {output_folder}')
