# Face Mesh Detection using MediaPipe and OpenCV

## Overview

This project uses MediaPipe Face Mesh and OpenCV to detect facial landmarks in real-time from a webcam feed. The application captures video frames, processes them using MediaPipe's Face Mesh model, and draws facial mesh connections on the detected face.

## Features

* Real-time webcam face detection
* Face mesh landmark detection
* Draws facial mesh connections on the face
* Supports multiple face detection
* Lightweight and fast processing

## Technologies Used

* Python
* OpenCV
* MediaPipe

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd FaceMesh
```

2. Create and activate a virtual environment (optional):

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install opencv-python mediapipe
```

## Usage

Run the application:

```bash
python facemesh.py
```

### Controls

* Press `q` to exit the application.

## Project Structure

```text
FaceMesh/
│
├── facemesh.py
├── README.md
```

## How It Works

1. Captures video from the webcam using OpenCV.
2. Converts frames from BGR to RGB format.
3. Processes frames using MediaPipe Face Mesh.
4. Detects facial landmarks.
5. Draws mesh connections on the detected face.
6. Displays the processed video stream.

## Future Enhancements

* Eye blink detection
* Face orientation tracking
* Emotion recognition
* Head pose estimation
* AR filters and effects

## Output

The application displays a live webcam feed with facial mesh landmarks and connections drawn on the detected face.

## License

This project is open source and available for learning and educational purposes.
