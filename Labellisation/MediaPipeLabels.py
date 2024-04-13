"""
    Title : MediaPipeLabels
    Description : Create labels for human pose on videos

    Author : Béatrice GARCIA CEGARRA

    Create date : 10/03/2024
    Last update : 17/03/2024
"""

##### Includes #####

# pip install -q mediapipe

import os
import cv2
import time

import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import scipy.signal
import scipy.io.wavfile
from matplotlib.pyplot import *


##### Get the video labels datas #####

def get_labels_video(path, labels_mp, video_name):
    start = time.time()

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
    out = cv2.VideoWriter('TestVideosOutput/output_'+video_name+'.mp4', cv2.VideoWriter.fourcc(*'MP4V'), fps, (height, width))

    # Video Labellisation
    datas = []
    for i in range(0, len(labels_mp)):
        coords = [[], [], [], []]
        datas.append(coords)

    cpt = 0
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

        cpt = cpt + 1

        # if cpt%30 == 0:
        #    mp_drawing.plot_landmarks(result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    out.release()
    cap.release()

    delta_time = time.time() - start

    return datas, nb_frame, fps, delta_time


def crop_datas_maxmin(datass, len_min):
    new_datass = []
    for datas in datass:
        new_datas = []

        for coords in datas:
            new_coords = []

            for coord in coords:
                new_coord = coord[:(int(len_min)-1)]
                new_coords.append(new_coord)

            new_datas.append(new_coords)

        new_datass.append(new_datas)

    return new_datass


def get_all_datas(video_repo_path, labels_mediapipe):
    tot_datas = []
    tot_filtered_datas = []

    tot_rotated_datas = []
    tot_rotated_filtered_datas = []

    tot_nb_frames = []
    tot_fps = []
    tot_deltas = []

    # Get all video datas by name from source directory
    video_names = os.listdir(video_repo_path)
    for video_name in video_names:
        print("Processing video : " + video_name + "...")

        video_path = video_repo_path + "/" + video_name

        # Get datas & rotate space
        datas, nb_frame, fps, delta_time = get_labels_video(video_path, labels_mediapipe, video_name)
        rotated_datas = rotation_space(datas, nb_frame, fps, labels_mediapipe)

        # Get filtered datas & rotate space
        filtered_datas = []
        for coord in datas:
            filtered_datas.append(lowpass_butter(coord, fps, 1))
        rotated_filtered_datas = rotation_space(filtered_datas, nb_frame, fps, labels_mediapipe)

        # Store datas for all videos
        tot_datas.append(datas)
        tot_filtered_datas.append(filtered_datas)

        tot_rotated_datas.append(rotated_datas)
        tot_rotated_filtered_datas.append(rotated_filtered_datas)

        tot_nb_frames.append(nb_frame)
        tot_fps.append(fps)
        tot_deltas.append(delta_time)

    # Crop all datas to maximum minimum number of frames
    len_min = min(tot_nb_frames)
    fps_min = min(tot_fps)

    new_tot_datas = crop_datas_maxmin(tot_datas, len_min)
    new_tot_filtered_datas = crop_datas_maxmin(tot_filtered_datas, len_min)
    new_tot_rotated_datas = crop_datas_maxmin(tot_rotated_datas, len_min)
    new_tot_rotated_filtered_datas = crop_datas_maxmin(tot_rotated_filtered_datas, len_min)

    # get median delta times of calcul

    return (new_tot_datas, new_tot_filtered_datas,
            new_tot_rotated_datas, new_tot_rotated_filtered_datas,
            video_names, len_min, fps_min, tot_deltas)


##### Center curves #####

def center_allvideos(tot_datas, tot_fil_datas, tot_rot_datas, tot_rot_fil_datas, labels_mediapipe):
    index = labels_mediapipe.index('right shoulder')

    middle_idexes = []
    nb_values_after = []
    nb_values_before = []
    for k in range(0, len(tot_rot_fil_datas)):
        y_array = tot_rot_fil_datas[k][index][1]
        middle_index = get_middle_point(y_array)

        if middle_index == -1:
            print("impossible to center video ", k)
            middle_idexes.append(len(y_array)//2)
        else:
            middle_idexes.append(middle_index)

        nb_values_after.append(len(y_array)-middle_index)
        nb_values_before.append(middle_idexes[k])

    max_after = min(nb_values_after)
    max_before = min(nb_values_before)

    new_tot_datas = []
    new_tot_fil_datas = []
    new_tot_rot_datas = []
    new_tot_rot_fil_datas = []
    for k in range(0, len(tot_rot_fil_datas)):
        new_tot_datas.append([])
        new_tot_fil_datas.append([])
        new_tot_rot_datas.append([])
        new_tot_rot_fil_datas.append([])
        for i in range(0, len(labels_mediapipe)):
            new_tot_datas[k].append([])
            new_tot_fil_datas[k].append([])
            new_tot_rot_datas[k].append([])
            new_tot_rot_fil_datas[k].append([])
            for j in range(0, 3):
                arr_tot = tot_datas[k][i][j][middle_idexes[k]-max_before:max_after+middle_idexes[k]]
                arr_tot_fil_datas = tot_fil_datas[k][i][j][middle_idexes[k]-max_before:max_after+middle_idexes[k]]
                arr_tot_rot_datas = tot_rot_datas[k][i][j][middle_idexes[k]-max_before : max_after+middle_idexes[k]]
                arr_tot_rot_fil_datas = tot_rot_fil_datas[k][i][j][middle_idexes[k]-max_before : max_after+middle_idexes[k]]

                new_tot_datas[k][i].append(arr_tot)
                new_tot_fil_datas[k][i].append(arr_tot_fil_datas)
                new_tot_rot_datas[k][i].append(arr_tot_rot_datas)
                new_tot_rot_fil_datas[k][i].append(arr_tot_rot_fil_datas)

    nb_frame = max_before + max_after + 1

    return new_tot_datas, new_tot_fil_datas, new_tot_rot_datas, new_tot_rot_fil_datas, nb_frame



def find_local_maxmin(arr):
    mx = []
    mn = []

    if arr[0] > arr[1]:
        mx.append(0)
    elif arr[0] < arr[1]:
        mn.append(0)

    for i in range(1, len(arr) - 1):
        if arr[i - 1] > arr[i] < arr[i + 1]:
            mn.append(i)

        elif arr[i - 1] < arr[i] > arr[i + 1]:
            mx.append(i)

    if arr[-1] > arr[-2]:
        mx.append(len(arr) - 1)
    elif arr[-1] < arr[-2]:
        mn.append(len(arr) - 1)

    return mx, mn


import bisect
def get_middle_point(arr):
    mx_indexes, mn_indexes = find_local_maxmin(arr)
    arr_max = max(arr)

    for i in range(0, len(mx_indexes)):
        if abs(arr[mx_indexes[i]]-arr_max) < 0.05:
            mn_indexes_temp = mn_indexes
            bisect.insort(mn_indexes_temp, mx_indexes[i])

            ind_max = mn_indexes_temp.index(mx_indexes[i])

            if ind_max != 0 & ind_max != len(mn_indexes_temp) - 1 :
                if abs(arr[mn_indexes_temp[ind_max-1]] - arr[mn_indexes_temp[ind_max+1]])<0.1:
                    return mx_indexes[i]

    return -1



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


def plot_1d_ncurves(datass, nb_frame, fps, label_name, index, legend):
    dims = ["x", "y", "z"]
    colors = ['r', 'b', 'g', 'c', 'm', 'y']

    for i in range(0, 3):
        plt.title('Comparison videos datas on ' + dims[i] + ' for ' + label_name + " (" + legend + ")", fontsize=14)
        times = np.linspace(0, int(nb_frame // fps), int(nb_frame - 1))

        for k in range(0, len(datass)):
            plt.plot(times, datass[k][index][i], colors[k % len(colors)], label="Input video " + dims[i] + str(k))

        plt.legend()
        plt.show()



##### Numerical signal filtering (outliers) #####

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

    # plot_1d_2curves(fil_x1, fil_y1, fil_z1, fil_x2, fil_y2, fil_z2, nb_frame, fps, "left and right soulder curves")

    proj_x_moy = np.sum(fil_x2-fil_x1) / len(fil_x1)
    proj_y_moy = np.sum(fil_y2-fil_y1) / len(fil_y1)
    proj_z_moy = np.sum(fil_z2-fil_z1) / len(fil_z1)

    # plot_1d_curve(fil_x1-fil_x2, fil_y1-fil_y2, fil_z1-fil_z2, nb_frame, fps, "Difference vector between shoulders")

    return [proj_x_moy, proj_y_moy, proj_z_moy]


def angle_2vectors(v1, axis):
    match axis:
        case "x":
            v2 = [1, 0]
            v1 = [v1[0], v1[1]]
        case "y":
            v2 = [1, 0]
            v1 = [v1[1], v1[2]]
        case "z":
            v2 = [0, 1]
            v1 = [v1[0], v1[2]]
        case _:
            v2 = [0, 0, 0]

    result_radians = np.arccos(np.dot(v2, v1) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    if result_radians > np.pi/2:
        result_radians = result_radians - np.pi

    return result_radians


def get_rot_matrix(datas, nb_frame, fps, labels_mp, proj_axis, rot_axis):
    index1 = labels_mp.index('left shoulder')
    index2 = labels_mp.index('right shoulder')

    vect_proj = get_proj_vector(index1, index2, datas, nb_frame, fps)
    angle = angle_2vectors(vect_proj, proj_axis)

    match rot_axis:
        case "x":
            matrix_rot = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        case "y":
            matrix_rot = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        case "z":
            matrix_rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), -np.cos(angle), 0], [0, 0, 1]])
        case _:
            matrix_rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    return matrix_rot


def rotation_space(datas, nb_frame, fps, labels_mp):
    matrix_rot = get_rot_matrix(datas, nb_frame, fps, labels_mp, "z", "y")

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

labels_mediapipe = ['nose', 'left eye (inner)', 'left eye', 'left eye (outer)', 'right eye (inner)', 'right eye', 'right eye (outer)',
                    'left ear', 'right ear', 'mouth (left)', 'mouth (right)', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow',
                    'left wrist', 'right wrist', 'left pinky', 'right pinky', 'left index', 'right index', 'left thumb', 'right thumb', 'left hip',
                    'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle', 'left heel', 'right heel', 'left foot index', 'right foot index']

video_repo_path = 'TestVideosInput'
label_test = 'mouth (right)'

tot_datas, tot_fil_datas, tot_rot_datas, tot_rot_fil_datas,  video_names, nb_frame, fps, deltas = get_all_datas(video_repo_path, labels_mediapipe)

# Display results
print("\nnumber of frames for all videos : ", nb_frame)
print("number of frames per second for all videos : ", fps)

sum_deltas = '%.2f' % (np.sum(np.array(deltas)))
moy_deltas = '%.2f' % (np.mean(np.array(deltas)))

print("Total processing time : ", sum_deltas)
print("Average processing time : ", moy_deltas)

index = labels_mediapipe.index(label_test)
"""
print("\nDisplaying 1D plots for mediapipe point on " + label_test + "...")

plot_1d_ncurves(tot_datas, nb_frame, fps, label_test, index, "raw datas")
plot_1d_ncurves(tot_fil_datas, nb_frame, fps, label_test, index, "filtered datas")
plot_1d_ncurves(tot_rot_datas, nb_frame, fps, label_test, index, "rotated datas")
plot_1d_ncurves(tot_rot_fil_datas, nb_frame, fps, label_test, index, "filtered & rotated datas")
"""

# Center all datas together

tot_datas_mid, tot_fil_datas_mid, tot_rot_datas_mid, tot_rot_fil_datas_mid, nb_frame_mid = center_allvideos(tot_datas, tot_fil_datas, tot_rot_datas, tot_rot_fil_datas, labels_mediapipe)

print("\nDisplaying 1D plots CENTERED for mediapipe point on " + label_test + "...")

plot_1d_ncurves(tot_datas_mid, nb_frame_mid, fps, label_test, index, "raw datas centered")
plot_1d_ncurves(tot_fil_datas_mid, nb_frame_mid, fps, label_test, index, "filtered datas centered")
plot_1d_ncurves(tot_rot_datas_mid, nb_frame_mid, fps, label_test, index, "rotated datas centered")
plot_1d_ncurves(tot_rot_fil_datas_mid, nb_frame_mid, fps, label_test, index, "filtered & rotated datas centered")


"""
video_path = 'TestVideosInput/snatch_video2.mp4'
datas, nb_frame, fps = get_labels_video(video_path, labels_mediapipe)
datas_rotated = rotation_space(datas, nb_frame, fps, labels_mediapipe)

print("shape datas : ", np.array(datas).shape)

# Selection of the label to plot
index = labels_mediapipe.index('left shoulder')

tests_plot_results(index, datas, nb_frame, fps)
tests_plot_results(index, datas_rotated, nb_frame, fps)
"""
