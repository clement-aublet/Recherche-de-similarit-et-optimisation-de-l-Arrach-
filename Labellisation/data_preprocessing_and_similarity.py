import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def get_video_data(video_path, labels_mp):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    frames = []
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            frame_data = [[results.pose_landmarks.landmark[label].x,
                           results.pose_landmarks.landmark[label].y,
                           results.pose_landmarks.landmark[label].z] for label in labels_mp]
            frames.append(frame_data)

    cap.release()
    pose.close()
    return np.array(frames, dtype=np.float32)


def create_lstm_model(input_shape, units=64, output_units=32):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        LSTM(units),
        Dense(output_units, activation='relu')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model



labels_mp = range(33)  # On peut prendre les 33 points de Mediapipe

# chemin vers les vidéos
video_path1 = r'C:\Users\Clément Aublet\OneDrive - ESIGELEC\Bureau\projetSeminaire\Recherche-de-similarit-et-optimisation-de-l-Arrach-\Labellisation\TestVideosOutput\output_snatch_video.mp4.mp4'

video_path2 = r'C:\Users\Clément Aublet\OneDrive - ESIGELEC\Bureau\projetSeminaire\Recherche-de-similarit-et-optimisation-de-l-Arrach-\Labellisation\TestVideosOutput\output_snatch_video3.mp4.mp4'


# Load data
data1 = get_video_data(video_path1, labels_mp)
data2 = get_video_data(video_path2, labels_mp)

# Les séquences doivent être de même distance pour LSTM (vidéos de même temps)
min_length = min(len(data1), len(data2))
data1 = data1[:min_length]
data2 = data2[:min_length]

# Prepare le modele LSTM
input_shape = (None, len(labels_mp), 3)  # (time_steps, features_per_step)
model = create_lstm_model(input_shape)

# Yon peut créer un daset des données des deux vidéos pour fit le modèle
combined_data = np.concatenate([data1, data2], axis=0)
model.fit(combined_data, combined_data, epochs=10, batch_size=1)

# Encodage des données avec le modèle
encoded_data1 = model.predict(np.expand_dims(data1, axis=0))
encoded_data2 = model.predict(np.expand_dims(data2, axis=0))

# Exemple Mesure la distance entre 2 sorties encodées de vidéos
distance = np.linalg.norm(encoded_data1 - encoded_data2)

print(f"Encoded distance between videos: {distance}")