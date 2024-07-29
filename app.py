import streamlit as st
import tempfile
import cv2
import mediapipe as mp
import numpy as np
from joblib import load

# Load the pre-trained model
model = load('randomforest2.model')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define angle pairs
angle_pairs = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value),
    (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value),
    (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value),
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
    (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
    (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
]

# Calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

# Process video
def process_video(video_path, segment_length=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'AVC1'), 30, (frame_width, frame_height))

    segment_angles = []
    frame_num = 0
    prediction = ''

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_angles = []

            for (a, b, c) in angle_pairs:
                angle = calculate_angle(
                    [landmarks[a].x, landmarks[a].y],
                    [landmarks[b].x, landmarks[b].y],
                    [landmarks[c].x, landmarks[c].y]
                )
                frame_angles.append(angle)

            segment_angles.append(frame_angles)

        if len(segment_angles) == segment_length:
            segment_angles_np = np.array(segment_angles).mean(axis=0).reshape(1, -1)
            prediction = model.predict(segment_angles_np)[0]
            segment_angles = []
        cv2.putText(frame, f'Predicted Exercise: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        out.write(frame)

        frame_num += 1

    cap.release()
    out.release()

    with open(output_path, 'rb') as video_file:
        video_bytes = video_file.read()

    return output_path, video_bytes

st.title('Analyze the Exercise')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    if st.button('Process Video'):
        with st.spinner(text="In progress..."):
            processed_video_path, video_bytes = process_video(video_path)
            if processed_video_path:
                # Ensure the video file is properly closed before playing
                st.video(processed_video_path)
                st.download_button(
                    label="Download Processed Video",
                    data=video_bytes,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
