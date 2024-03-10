"""
    Title : MediaPipeLabels
    Description : Create labels for human pose on videos

    Author : Béatrice GARCIA CEGARRA

    Create date : 10/03/2024
    Last update : 10/03/2024
"""

##### Includes #####

# pip install -q mediapipe
# wget -O pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task


import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import matplotlib.pyplot as plt


##### Initialize MediaPipe Pose and Drawing utilities #####

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()


##### Open the video file #####

video_path = 'TestVideosInput/snatch_video.mp4'
cap = cv2.VideoCapture(video_path)


##### Get characteristics of video #####

ret, img = cap.read()

fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)
width = img.shape[0]
height = img.shape[1]


##### Create video output #####

out = cv2.VideoWriter('TestVideosOutput/test_output.mp4', cv2.VideoWriter.fourcc(*'MP4V'), fps, (height, width))


##### Video Labellisation #####

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    result = pose.process(frame_rgb)

    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    out.write(frame)

out.release()
cap.release()


