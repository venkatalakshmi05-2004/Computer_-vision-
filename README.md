# Computer Vision Gesture & Face Detection with OpenCV

A real-time computer vision application built with Python, OpenCV, and NumPy. It uses a webcam to detect faces and estimate simple hand gestures.

## Features

- Real-time face detection using OpenCV Haar Cascade.
- Hand-region segmentation using the YCrCb color space.
- Basic finger-count estimation using contour convexity defects.
- Live webcam display with detection labels.
- Separate window showing the hand-segmentation mask.

## Technologies

- Python 3
- OpenCV
- NumPy
- Computer Vision

## Project Structure

```text
Computer-Vision-Gesture-Face-Detection/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. Install Python 3.
2. Clone this repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/Computer-Vision-Gesture-Face-Detection.git
   cd Computer-Vision-Gesture-Face-Detection
   ```

3. (Recommended) Create and activate a virtual environment:

   **Windows**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   **macOS / Linux**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the Project

```bash
python main.py
```

Allow webcam access if your operating system asks. Press **Q** to close the application.

If the webcam does not open, change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `main.py`.

## How It Works

### Face Detection
The webcam frame is converted to grayscale, equalized, and passed to OpenCV's pre-trained Haar Cascade face detector. Detected faces are shown with green rectangles.

### Hand Gesture Detection
The frame is converted to YCrCb color space and thresholded to create a rough skin-color mask. Morphological operations reduce noise. The largest suitable contour is treated as the hand candidate. Convexity defects are then used to estimate the number of visible raised fingers.

## Limitations

- Skin-color segmentation can mistake skin-colored objects for hands.
- Lighting, camera quality, background, hand orientation, and skin tone affect detection.
- Finger counting is an approximate demonstration and may not recognize every gesture correctly.
- Haar Cascade face detection may miss faces at unusual angles or in poor lighting.
- The application detects faces; it does not identify who a person is.

## Future Improvements

- Use MediaPipe Hands for more robust hand landmark detection.
- Add more gesture classes and custom gesture commands.
- Add face detection confidence and performance measurements.
- Build a graphical user interface.

## Author

**Your Name**

Artificial Intelligence Internship Project
