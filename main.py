# 1. Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# 2. Configuration
VIDEO_DIR = 'kaggle/input/video-emotion/VideoFlash'
PARQUET_FILE_PATH = 'kaggle/input/optimized-video-facial-landmarks/emotion_landmark_dataset.parquet'
NUM_FRAMES_TO_PLOT = 16
LANDMARK_COUNT = 478
POINT_RADIUS = 2
POINT_COLOR = (0, 0, 255) # Red (BGR)
RANDOM_STATE = 26

# 3. Data Loading
df = pd.read_parquet(PARQUET_FILE_PATH)
print(df.head())

# 4. Visualization: Random frames landmark overlay
if len(df) < NUM_FRAMES_TO_PLOT:
    random_points_df = df
else:
    random_points_df = df.sample(n=NUM_FRAMES_TO_PLOT, replace=False, random_state=RANDOM_STATE)
random_points_list = random_points_df.to_dict('records')

processed_frames = []
for i, point_data in enumerate(random_points_list):
    video_filename = point_data['video_filename']
    frame_num = int(point_data['frame_num'])
    emotion = point_data['emotion']
    video_path = f"{VIDEO_DIR}/{video_filename}"
    norm_x_coords = [point_data[f'x_{j}'] for j in range(LANDMARK_COUNT)]
    norm_y_coords = [point_data[f'y_{j}'] for j in range(LANDMARK_COUNT)]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        continue

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        x_coords = [int(x * frame_width) for x in norm_x_coords]
        y_coords = [int(y * frame_height) for y in norm_y_coords]
        for x, y in zip(x_coords, y_coords):
            center = (x, y)
            cv2.circle(frame, center, radius=POINT_RADIUS, color=POINT_COLOR, thickness=-1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        title = f"V: {video_filename[-10:]}\nF: {frame_num} | Emotion: {emotion}"
        processed_frames.append({ 'frame': frame_rgb, 'title': title})
    cap.release()

plot_count = len(processed_frames)
rows = int(np.ceil(np.sqrt(plot_count)))
cols = int(np.ceil(plot_count / rows))
fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
axes = axes.ravel()

for i in range(rows * cols):
    axes[i].axis('off')
    if i < plot_count:
        frame_data = processed_frames[i]
        axes[i].imshow(frame_data['frame'])
        title = frame_data['title']
        axes[i].text(
            0.02, 0.97, title,
            color='white',
            fontsize=10,
            ha='left',
            va='top',
            transform=axes[i].transAxes,
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2')
        )
plt.tight_layout()
plt.show()

# 5. Prepare data for ML model
X = []
y = []
for _, row in df.iterrows():
    coords = []
    for j in range(LANDMARK_COUNT):
        coords.append(row[f'x_{j}'])
        coords.append(row[f'y_{j}'])
        coords.append(row[f'z_{j}'])
    X.append(coords)
    y.append(row['emotion'])

X = np.array(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 6. ML train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
input_dim = LANDMARK_COUNT * 3
output_dim = len(le.classes_)

# 7. ML Model Building
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

model = EmotionClassifier(input_dim, output_dim)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8. Training Loop
epochs = 20
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = loss_fn(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 9. Evaluation
model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test)
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    print(classification_report(y_test, predicted.numpy(), target_names=le.classes_))
