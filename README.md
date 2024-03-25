**Introduction:**
The project aims to develop a real-time gesture recognition system using computer vision techniques and machine learning. The system utilizes the Mediapipe library for pose estimation and TensorFlow for training a deep learning model to recognize gestures.

**2. Overview:**
The system consists of two main components:

- Data Collection: Captures video feed from a webcam, extracts keypoint data using Mediapipe, and saves the data for training.
- Gesture Recognition: Utilizes a pre-trained deep learning model to classify gestures in real-time, with visual feedback and text-to-speech output.

**3. Data Collection:**

- Utilizes OpenCV to capture video frames from the webcam.
- Mediapipe is employed to detect and extract keypoints representing various body parts and hand gestures from each frame.
- Keypoint data is saved in NumPy arrays for each gesture and sequence, creating a dataset for training.

**4. Gesture Recognition:**

- Trains a deep learning model using TensorFlow on the collected dataset.
- The model is a convolutional neural network (CNN) designed to classify gestures based on the extracted keypoints.
- In real-time, the system captures video frames, processes them through Mediapipe for keypoint extraction, and feeds the sequences of keypoints into the trained model.
- The model predicts the gesture being performed, and if the confidence exceeds a threshold, the recognized gesture is spoken aloud using text-to-speech.
- Visual feedback is provided on the video feed, with probabilities of recognized gestures displayed in a colored bar.

**5. Results:**

- The system demonstrates real-time gesture recognition capabilities, accurately identifying gestures from the webcam feed.
- Text-to-speech output provides auditory feedback, enhancing user interaction.
- Visual feedback enhances user experience by displaying recognized gestures and associated probabilities.
![Screenshot 2024-03-25 192003](https://github.com/yash9373/sign_to_audio_translator/assets/101787484/4ac4fda7-8f12-45a4-8841-c637180c8bde)
![Screenshot 2024-03-25 192131](https://github.com/yash9373/sign_to_audio_translator/assets/101787484/f85537e0-0158-495f-bf33-200d28d3ccf0)

**Additional Note:**
Press 'q' to quit the camera when running the project. Execute the `app.py` file using the command `python app.py`.
