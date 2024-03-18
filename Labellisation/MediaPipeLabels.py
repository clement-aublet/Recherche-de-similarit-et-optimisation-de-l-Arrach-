"""
    Title : MediaPipeLabels
    Description : Create labels for human pose on videos

    Author : Béatrice GARCIA CEGARRA

    Create date : 10/03/2024
    Last update : 17/03/2024
"""

##### Includes #####

# pip install -q mediapipe
# wget -O pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task


import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import scipy.signal
import scipy.io.wavfile

from matplotlib.pyplot import *

##### Initialize MediaPipe Pose and Drawing utilities #####

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()


##### Open the video file #####

video_path = 'TestVideosInput/snatch_video2.mp4'
cap = cv2.VideoCapture(video_path)


##### Get characteristics of video #####

ret, img = cap.read()

nb_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
width = img.shape[0]
height = img.shape[1]


##### Create video output #####

out = cv2.VideoWriter('TestVideosOutput/test_output.mp4', cv2.VideoWriter.fourcc(*'MP4V'), fps, (height, width))


##### Video Labellisation #####

labels_mediapipe = ['nose', 'left eye (inner)', 'left eye', 'left eye (outer)', 'right eye (inner)', 'right eye', 'right eye (outer)', 'left ear',
          'right ear', 'mouth (left)', 'mouth (right)', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist',
          'right wrist', 'left pinky', 'right pinky', 'left index', 'right index', 'left thumb', 'right thumb', 'left hip', 'right hip',
          'left knee', 'right knee', 'left ankle', 'right ankle', 'left heel', 'right heel', 'left foot index', 'right foot index']

datas = []
for i in range(0, len(labels_mediapipe)):
    coords = [[], [], [], []]
    datas.append(coords)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    result = pose.process(frame_rgb)

    # Add marked frame to output video
    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    out.write(frame)

    # Add coordinates to datas
    for i in range(0, len(labels_mediapipe)):
        datas[i][0].append(result.pose_landmarks.landmark[i].x)
        datas[i][1].append(result.pose_landmarks.landmark[i].y)
        datas[i][2].append(result.pose_landmarks.landmark[i].z)
        datas[i][3].append(result.pose_landmarks.landmark[i].visibility)

out.release()
cap.release()


##### Plot 3D curves of the video labels #####

def plot_3d_curve(array_x, array_y, array_z):

    # Parametric equations in cartesian coordinates
    fig = plt.figure('Left wrist curve')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(array_x, array_y, array_z, '-b', linewidth=3)

    # Setting the labels
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)

    plt.title('Parametric Curve 3D', fontsize=14)
    plt.show()


def plot_1d_curve(x, y, z, nb_frame, fps):
    times = np.linspace(0, int(nb_frame//fps), int(nb_frame-1))
    plt.title('Parametric curve 1D', fontsize=14)
    plt.plot(times, x, 'r')
    plt.plot(times, y, 'b')
    plt.plot(times, z, 'g')
    plt.show()


##### Numerical signal filtering (outliers) #####

def custom_outliers(array):
    for k in range(1, len(array)-1):
        delta = array[k-1] - array[k+1]

        if delta < 0:
            print("hello")


def clean_outliers(array_1d):
    q1 = np.quantile(array_1d, .25)
    q3 = np.quantile(array_1d, .75)

    ecart = q3 - q1
    b_inf = q1 - 1.5 * ecart
    b_sup = q3 + 1.5 * ecart

    new_array = np.where(array_1d < b_inf, b_inf, array_1d)
    new_array = np.where(new_array > b_sup, b_sup, new_array)

    return new_array


##### Numerical signal filtering (low passes) #####

def lowpass_savgol(array):
    filtered = savgol_filter(array, 99, 3)
    return filtered


def lowpass_butterworth(data, cutoff, sample_rate, poles=5):
    sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


def lowpass_butterworth2(array, fps, fc):
    f_nyq = fps / 2.  # Fréquence de nyquist (en Hz)

    # Préparation du filtre de Butterworth en passe-bas
    b, a = scipy.signal.butter(4, fc / f_nyq, 'low', analog=False)

    # Application du filtre
    s_but = scipy.signal.filtfilt(b, a, array)

    return s_but


##### Tests #####

# Selection of the label to plot
index = 25

x = datas[index][0]
y = datas[index][1]
z = datas[index][2]

filtered1_x = lowpass_savgol(x)
filtered1_y = lowpass_savgol(y)
filtered1_z = lowpass_savgol(z)

filtered2_x = lowpass_butterworth(x, 10, nb_frame)
filtered2_y = lowpass_butterworth(y, 10, nb_frame)
filtered2_z = lowpass_butterworth(z, 5, nb_frame)

filtered3_x = lowpass_butterworth2(x, fps, 1)
filtered3_y = lowpass_butterworth2(y, fps, 1)
filtered3_z = lowpass_butterworth2(z, fps, 0.5)

# plot_3d_curve(datas[index][0], datas[index][1], datas[index][2])
plot_3d_curve(x, y, z)
plot_3d_curve(filtered1_x, filtered1_y, filtered1_z)
plot_3d_curve(filtered2_x, filtered2_y, filtered2_z)
plot_3d_curve(filtered3_x, filtered3_y, filtered3_z)

plot_1d_curve(x, y, z, nb_frame, fps)
plot_1d_curve(filtered1_x, filtered1_y, filtered1_z, nb_frame, fps)
plot_1d_curve(filtered2_x, filtered2_y, filtered2_z, nb_frame, fps)
plot_1d_curve(filtered3_x, filtered3_y, filtered3_z, nb_frame, fps)
