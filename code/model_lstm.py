import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from collections import deque

# 1. 加载和预处理数据
data = pd.read_csv('VT1\data2\keypoints8.csv')
X = data.drop('label', axis=1)
y = data['label']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建 TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# 创建 DataLoader
batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. 构建模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = X_train.shape[1]
hidden_size = 256
num_layers = 2
output_size = 10
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

# 4. 训练模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 初始化数据列表
train_losses = []
validation_losses = []
accuracies = []

epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_losses.append(train_loss / len(train_loader))

    # 5. 验证和评估模型性能
    model.eval()
    validation_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output = model(batch_X)
            loss = criterion(output, batch_y)
            validation_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == batch_y).sum().item()
            total_predictions += batch_y.size(0)

    validation_losses.append(validation_loss / len(test_loader))
    accuracy = correct_predictions / total_predictions
    accuracies.append(accuracy)

    print(f"Epoch: {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]}, Validation Loss: {validation_losses[-1]}, Accuracy: {accuracy * 100}%")

torch.save(model.state_dict(), 'VT1\lstm_model2\model3.pth')

# 绘制曲线图
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.plot(accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss & Accuracy')
plt.title('Training and Validation Loss & Accuracy')
plt.legend()
plt.show()

loaded_model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
loaded_model.load_state_dict(torch.load('VT1\lstm_model2\model3.pth'))
loaded_model.eval()
'''
# 初始化 MediaPipe Hands API
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# 使用摄像头捕获实时视频
gesture_names = ['back', 'front', 'left', 'right','stop','yeah','good','hello','ok','bad']



cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hand_model:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand_model.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            x, y = [], []
            for lm in landmarks.landmark:
                x.append(lm.x)
                y.append(lm.y)

            points = np.asarray([x, y])
            
            # 中心化处理
            points_centered = points - np.mean(points, axis=1, keepdims=True)

            min = points_centered.min(axis=1, keepdims=True)
            max = points_centered.max(axis=1, keepdims=True)
            normalized = np.stack((points_centered - min) / (max - min), axis=1).flatten()

            keypoints = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
            with torch.no_grad():
                output = loaded_model(keypoints)
                probabilities = torch.softmax(output, dim=1)
                gesture = torch.argmax(probabilities, dim=1).item()

            confidence = probabilities.max().item()
            gesture_name = gesture_names[gesture]
            cv2.putText(frame, f'Gesture: {gesture_name}, Confidence: {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


#视频文件

# 初始化 MediaPipe Hands API
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# 手势列表
gesture_names = ['back', 'front', 'left', 'right','stop','yeah','good','hello','ok','bad']

# 视频文件路径
video_path = 'VT1\\test_video\\test.mp4'
# 输出视频文件路径
output_path = 'VT1\\test_video\\test_val_lstm.mp4'

cap = cv2.VideoCapture(video_path)

# 获取视频的编解码器信息，并创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编解码器
fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 帧尺寸
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hand_model:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand_model.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            x, y = [], []
            for lm in landmarks.landmark:
                x.append(lm.x)
                y.append(lm.y)

            points = np.asarray([x, y])
            min = points.min(axis=1, keepdims=True)
            max = points.max(axis=1, keepdims=True)
            normalized = np.stack((points - min) / (max - min), axis=1).flatten()

            keypoints = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
            with torch.no_grad():
                output = loaded_model(keypoints)
                probabilities = torch.softmax(output, dim=1)
                gesture = torch.argmax(probabilities, dim=1).item()

            confidence = probabilities.max().item()
            gesture_name = gesture_names[gesture]
            cv2.putText(frame, f'Gesture: {gesture_name}, Confidence: {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 将处理后的帧写入新的视频文件
        out.write(frame)

# 释放资源
cap.release()
out.release()
'''