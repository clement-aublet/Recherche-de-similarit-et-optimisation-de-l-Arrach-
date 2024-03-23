"""
    Title : MediaPipeLabels
    Description : Create labels for human pose on videos

    Author : Béatrice GARCIA CEGARRA

    Create date : 10/03/2024
    Last update : 17/03/2024
"""

##### Includes #####

# pip install -q mediapipe

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import scipy.signal
import scipy.io.wavfile
from matplotlib.pyplot import *


##### Get the video labels datas #####

def get_labels_video(path, labels_mp):

    # Initialize MediaPipe Pose and Drawing utilities
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    # Open the video file
    cap = cv2.VideoCapture(path)

    # Get characteristics of video
    ret, img = cap.read()

    nb_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = img.shape[0]
    height = img.shape[1]

    # Create video output
    out = cv2.VideoWriter('TestVideosOutput/test_output.mp4', cv2.VideoWriter.fourcc(*'MP4V'), fps, (height, width))

    # Video Labellisation
    datas = []
    for i in range(0, len(labels_mp)):
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
        for i in range(0, len(labels_mp)):
            datas[i][0].append(result.pose_landmarks.landmark[i].x)
            datas[i][1].append(result.pose_landmarks.landmark[i].y)
            datas[i][2].append(result.pose_landmarks.landmark[i].z)
            datas[i][3].append(result.pose_landmarks.landmark[i].visibility)

    out.release()
    cap.release()

    return datas, nb_frame, fps


##### Plot 3D curves of the video labels #####

def plot_3d_curve(array_x, array_y, array_z, name):

    # Parametric equations in cartesian coordinates
    fig = plt.figure('Left wrist curve')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(array_x, array_y, array_z, '-b', linewidth=3)

    # Setting the labels
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)

    plt.title(name, fontsize=14)
    plt.show()


def plot_1d_curve(x, y, z, nb_frame, fps, name):
    times = np.linspace(0, int(nb_frame//fps), int(nb_frame-1))
    plt.title('Parametric curve 1D', fontsize=14)
    plt.plot(times, x, 'r', label='x')
    plt.plot(times, y, 'b', label='y')
    plt.plot(times, z, 'g', label='z')

    plt.title(name, fontsize=14)
    plt.legend()
    plt.show()


def plot_1d_2curves(x1, y1, z1, x2, y2, z2, nb_frame, fps, name):
    times = np.linspace(0, int(nb_frame//fps), int(nb_frame-1))
    plt.title('Parametric curve 1D', fontsize=14)

    plt.plot(times, x1, 'r', label='x1')
    plt.plot(times, y1, 'b', label='y1')
    plt.plot(times, z1, 'g', label='z1')

    plt.plot(times, x2, 'tab:orange', label='x2')
    plt.plot(times, y2, 'tab:cyan', label='y2')
    plt.plot(times, z2, 'tab:olive', label='z2')

    plt.title(name, fontsize=14)
    plt.legend()
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


def lowpass_butter(array, fps, fc):
    f_nyq = fps / 2.  # Fréquence de nyquist (en Hz)

    # Préparation du filtre de Butterworth en passe-bas
    b, a = scipy.signal.butter(4, fc / f_nyq, 'low', analog=False)

    # Application du filtre
    s_but = scipy.signal.filtfilt(b, a, array)

    return s_but


##### Projection #####

def get_proj_vector(index1, index2, datas, nb_frame, fps):
    fil_x1 = lowpass_butter(datas[index1][0], fps, 1)
    fil_y1 = lowpass_butter(datas[index1][1], fps, 1)
    fil_z1 = lowpass_butter(datas[index1][2], fps, 1)

    fil_x2 = lowpass_butter(datas[index2][0], fps, 1)
    fil_y2 = lowpass_butter(datas[index2][1], fps, 1)
    fil_z2 = lowpass_butter(datas[index2][2], fps, 1)

    plot_1d_2curves(fil_x1, fil_y1, fil_z1, fil_x2, fil_y2, fil_z2, nb_frame, fps, "left and right soulder curves")

    proj_x_moy = np.sum(fil_x2-fil_x1) / len(fil_x1)
    proj_y_moy = np.sum(fil_y2-fil_y1) / len(fil_y1)
    proj_z_moy = np.sum(fil_z2-fil_z1) / len(fil_z1)

    plot_1d_curve(fil_x1-fil_x2, fil_y1-fil_y2, fil_z1-fil_z2, nb_frame, fps, "Difference vector between shoulders")

    return [proj_x_moy, proj_y_moy, proj_z_moy]


def angle_2vectors(v1, axis):
    match axis:
        case "x":
            v2 = [1, 0, 0]
        case "y":
            v2 = [0, 1, 0]
        case "z":
            v2 = [0, 0, 1]
        case _:
            v2 = [0, 0, 0]

    return np.arccos(np.dot(v2, v1) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def get_rot_matrix(labels_mp, proj_axis, rot_axis):
    index1 = labels_mp.index('left shoulder')
    index2 = labels_mp.index('right shoulder')

    vect_proj = get_proj_vector(index1, index2, datas, nb_frame, fps)
    angle = angle_2vectors(vect_proj, proj_axis)

    print("Angle : ", angle, "°")

    match rot_axis:
        case "x":
            matrix_rot = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        case "y":
            matrix_rot = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
            print("babas")
        case "z":
            matrix_rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), -np.cos(angle), 0], [0, 0, 1]])
        case _:
            matrix_rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    return matrix_rot


def rotation_space(datas, nb_frame, fps, labels_mp):
    matrix_rot = get_rot_matrix(labels_mp, "z", "y")

    rot_datas = []
    for ind in range(0, len(datas)): # for all indexes
        rot_datas.append([[], [], []])
        for tim in range(0, len(datas[0][0])): # for all times
            vect_tim = np.array([datas[ind][0][tim], datas[ind][1][tim], datas[ind][2][tim]])
            rot_vect_tim = matrix_rot.dot(vect_tim)

            rot_datas[ind][0].append(rot_vect_tim[0])
            rot_datas[ind][1].append(rot_vect_tim[1])
            rot_datas[ind][2].append(rot_vect_tim[2])

    return rot_datas


##### Tests #####

def tests_plot_results(index, datas, nb_frame, fps):
    x = datas[index][0]
    y = datas[index][1]
    z = datas[index][2]

    # Filter datas of selected index
    filtered1_x = lowpass_savgol(x)
    filtered1_y = lowpass_savgol(y)
    filtered1_z = lowpass_savgol(z)

    filtered2_x = lowpass_butter(x, fps, 1)
    filtered2_y = lowpass_butter(y, fps, 1)
    filtered2_z = lowpass_butter(z, fps, 0.5)

    # plot 3D curves
    plot_3d_curve(x, y, z, "Evolution of left shoulder in 3D (Raw datas)")
    plot_3d_curve(filtered1_x, filtered1_y, filtered1_z, "Evolution of left shoulder in 3D (Savgol filter)")
    plot_3d_curve(filtered2_x, filtered2_y, filtered2_z, "Evolution of left shoulder in 3D (Butterworth filter)")

    # plot 1D curves (decompose axis)
    plot_1d_curve(x, y, z, nb_frame, fps, "Evolution of left shoulder in 1D (Raw datas)")
    plot_1d_curve(filtered1_x, filtered1_y, filtered1_z, nb_frame, fps, "Evolution of left shoulder in 1D (Savgol filter)")
    plot_1d_curve(filtered2_x, filtered2_y, filtered2_z, nb_frame, fps, "Evolution of left shoulder in 1D (Butterworth filter)")


# Get labels of a selected videos
video_path = 'TestVideosInput/snatch_video2.mp4'

labels_mediapipe = ['nose', 'left eye (inner)', 'left eye', 'left eye (outer)', 'right eye (inner)', 'right eye', 'right eye (outer)', 'left ear',
          'right ear', 'mouth (left)', 'mouth (right)', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist',
          'right wrist', 'left pinky', 'right pinky', 'left index', 'right index', 'left thumb', 'right thumb', 'left hip', 'right hip',
          'left knee', 'right knee', 'left ankle', 'right ankle', 'left heel', 'right heel', 'left foot index', 'right foot index']


datas, nb_frame, fps = get_labels_video(video_path, labels_mediapipe)
datas_rotated = rotation_space(datas, nb_frame, fps, labels_mediapipe)

print("shape datas : ", np.array(datas).shape)

# Selection of the label to plot
index = labels_mediapipe.index('left shoulder')

tests_plot_results(index, datas, nb_frame, fps)
tests_plot_results(index, datas_rotated, nb_frame, fps)
