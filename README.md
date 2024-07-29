# Fitness-Tracker-with-Pose-Estimation


## Introduction:
This Project is a part of Infosys Internship 4.0 Project. This project focuses on exercise detection using Mediapipe, a Python module, to classify different exercises based on video data. The system leverages body point detection to calculate angles, which are then used to train a model for real-time exercise recognition. This project includes detection of 5 exercise(Deadlift, Hammer curl, Push up, Pull up, Squat). Then integrate the prediction part with streamlit for UI.

## Modules Used:
- **Boto3**: Used to interact with amazon services. 
- **Mediapipe**: For detecting body points and extracting features.
- **OpenCV**: For video processing and frame extraction.
- **Sklearn**: For training the machine learning model.
- **Pandas**: For data manipulation and storage.
- **os**: For file handling and directory operations.
- **joblib**: To save and load the model that is trained.
- **Streamlit**: Used for creating UI.
- **numpy**: For handling numberical calculation. 

## Steps Involved
### Data Collection
1. **Dataset Acquisition**: Obtain exercise videos from Kaggle [dataset link](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video) 
2. **Amazon S3**: Push the dataset to Amazon S3 bucket through boto3 then download it to the local device where do the further processing.
3. **File Handling**: Use the `os` module to navigate through files and directories.
4. **Feature Extraction**: Utilize `OpenCV` to open each video and Mediapipe to extract required body features.
5. **Angle Calculation**: Compute angles between different body points from the extracted features.
6. **Data Storage**: Save the calculated angles and corresponding exercise labels in a CSV file using `Pandas`.
### Model Training
1. **Data Import**: Load the CSV file containing angle data and exercise labels.
2. **Model Selection**: Use Sklearn to implement a Random forest classifier.
3. **Training**: Train the model with features `(X=all_angles)` and labels `(Y=exercise)`.
4. **Save the model**: Save the model with joblib module with the extension `.model` .

### UI developement:
1. **Upload**: The user uploads a video file using `st.file_uploader`.
2. **Temporary Storage**: The uploaded file is written to a temporary file.
3. **Processing Button**: When "Process Video" is clicked, a spinner indicates progress.
4. **Video Processing**: The video is processed to predict exercises.
5. **Playback and Download**: The processed video is displayed with `st.video` and a download button is provided with `st.download_button`.


