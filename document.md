# Code Documentation Report

## Introduction
This document provides a detailed explanation of the code that processes video files to extract pose landmarks and calculate angles between specified joints using MediaPipe's Pose solution. The code is designed to list directories, list all files in a root directory, and process videos to calculate and store joint angles at specified intervals.

## Function Definitions

### `list_folders(directory)`
This function lists all subdirectories within a given directory.

**Parameters:**
- `directory` (str): The path to the directory whose subdirectories are to be listed.

**Returns:**
- A list of subdirectories if the directory exists and is accessible.
- An error message if the directory does not exist, or if access is denied.

**Example Usage:**
```python
exercise = list_folders(root_directory_path)
print("Exercise in the directory:", exercise)
```

### `list_all_files(root_directory)`
This function lists all files within a root directory and its subdirectories.

**Parameters:**
- `root_directory` (str): The path to the root directory.

**Returns:**
- A list containing the full paths of all files in the root directory and its subdirectories.

**Example Usage:**
```python
all_files = list_all_files(root_directory_path)

# Print all file paths
for file in all_files:
    print(file)
```

### `calculate_angle(a, b, c)`
This function calculates the angle between three points.

**Parameters:**
- `a` (list): The [x, y] coordinates of the first point.
- `b` (list): The [x, y] coordinates of the second point.
- `c` (list): The [x, y] coordinates of the third point.

**Returns:**
- The calculated angle in degrees.

**Example Usage:**
```python
angle = calculate_angle(a, b, c)
```

## Main Code Block

### Importing Libraries
The code imports necessary libraries including `os` for directory operations, `cv2` for video processing, `mediapipe` for pose estimation, and `pandas` for data manipulation.

```python
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
```

### Initializing MediaPipe Pose
The MediaPipe Pose solution is initialized with specified confidence values for detection and tracking.

```python
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
```

### Processing Video Files
The code iterates over all files in the specified directory, processes each video to extract pose landmarks, and calculates angles between specified joints at set intervals.

#### Reading Video Files
```python
for video_path in all_files:
    cap = cv2.VideoCapture(video_path)
```

#### Initializing Variables
```python
angles_data = []
frame_count = 0
interval = 5  # Interval to process frames
```

#### Defining Landmark Pairs
The pairs of landmarks for which angles will be calculated are defined.
```python
angle_pairs = [
    ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
    ('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
    ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
    ('LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'),
    ('RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'),
    ('LEFT_ELBOW', 'LEFT_SHOULDER', 'LEFT_HIP'),
    ('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'RIGHT_HIP')
]
```

#### Processing Frames
The code reads frames from the video, converts them to RGB, processes them to extract pose landmarks, and calculates angles for the defined landmark pairs.
```python
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % interval == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            # Collect angles for each specified pair
            frame_angles = dict()
            for pair in angle_pairs:
                if pair not in exercise:
                    a = [landmarks[getattr(mp_pose.PoseLandmark, pair[0]).value].x,
                        landmarks[getattr(mp_pose.PoseLandmark, pair[0]).value].y]
                    b = [landmarks[getattr(mp_pose.PoseLandmark, pair[1]).value].x,
                        landmarks[getattr(mp_pose.PoseLandmark, pair[1]).value].y]
                    c = [landmarks[getattr(mp_pose.PoseLandmark, pair[2]).value].x,
                        landmarks[getattr(mp_pose.PoseLandmark, pair[2]).value].y]

                    angle = calculate_angle(a, b, c)
                    frame_angles[f'{pair[0]}-{pair[1]}-{pair[2]}'] = angle
                else:
                    frame_angles["Exercise"]=pair
            # Append the frame's angles to the list
            angles_data.append(frame_angles)

cap.release()
```

### Storing Angles Data
The collected angles data is stored in a CSV file using pandas.

```python
df = pd.DataFrame(angles_data)
df.to_csv("output.csv", index=False)
print(f'Angles saved')
```

## Conclusion
This code effectively lists directories and files, processes videos to extract pose landmarks using MediaPipe, calculates joint angles, and stores the results in a CSV file. It demonstrates a comprehensive workflow for pose estimation and angle calculation, suitable for applications such as exercise form analysis.