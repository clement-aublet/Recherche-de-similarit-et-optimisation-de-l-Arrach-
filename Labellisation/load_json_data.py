import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['datas_fil']

def calculate_pose_distance(pose1, pose2):
    return np.linalg.norm(np.array(pose1) - np.array(pose2))

def compare_all_videos(poses):
    num_videos = len(poses)
    distances_matrix = np.zeros((num_videos, num_videos))

    for i in range(num_videos):
        for j in range(i + 1, num_videos):
            distances = [calculate_pose_distance(pose1, pose2) for pose1, pose2 in zip(poses[i], poses[j])]
            distances_matrix[i][j] = np.mean(distances)
            distances_matrix[j][i] = distances_matrix[i][j]

    return distances_matrix

def find_most_similar_video(distances_matrix, video_index):
    distances = distances_matrix[video_index].copy()
    distances[video_index] = np.inf
    similar_index = np.argmin(distances)
    return similar_index

def display_videos(video_path, index1, index2):
    video_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.mp4')])
    cap1 = cv2.VideoCapture(video_files[index1])
    cap2 = cv2.VideoCapture(video_files[index2])

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        combined_frame = np.concatenate((frame1, frame2), axis=1)
        cv2.imshow('Video Comparison', combined_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

# Chemin vers le fichier JSON et le dossier vidéo
json_path = 'datasForTraining.json'
video_path = 'C:\\Users\\Clément Aublet\\OneDrive - ESIGELEC\\Bureau\\projetSeminaire\\Recherche-de-similarit-et-optimisation-de-l-Arrach-\\videos'

# Charger les données
poses = load_data(json_path)

# Comparer les vidéos
distances_matrix = compare_all_videos(poses)

# Trouver la vidéo la plus similaire à la vidéo 42
video_index = 54
most_similar = find_most_similar_video(distances_matrix, video_index)
print(f"La vidéo la plus similaire à la vidéo {video_index} est la vidéo {most_similar}.")

# Visualisation des distances
sns.heatmap(distances_matrix, annot=True, fmt=".2f")
plt.title('Heatmap of Pose Distances Between Videos')
plt.xlabel('Video Index')
plt.ylabel('Video Index')
plt.show()

# Afficher les deux vidéos
display_videos(video_path, video_index, most_similar)
