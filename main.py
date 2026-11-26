"""
Computer Vision Gesture & Face Detection with OpenCV
----------------------------------------------------
- Detects faces using OpenCV's built-in Haar Cascade.
- Detects a hand using YCrCb skin-color segmentation.
- Estimates the number of raised fingers using convexity defects.

Run:
    python main.py

Press 'q' to quit.

Note:
Skin-color segmentation is sensitive to lighting, background, and skin tone.
For best results, use a plain background and good, even lighting. This is a
basic OpenCV demonstration, not a biometric identification system.
"""

import cv2
import numpy as np


def create_face_detector():
    """Load the Haar Cascade included with the OpenCV Python package."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError(
            "Could not load the face-detection Haar Cascade. "
            "Check your OpenCV installation."
        )
    return detector


def detect_faces(frame, face_detector):
    """Detect faces and draw bounding boxes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, "Face", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

    return len(faces)


def build_skin_mask(frame):
    """
    Create a rough skin-color mask in YCrCb color space.
    Thresholds may need adjustment for the camera and lighting.
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

    mask = cv2.inRange(ycrcb, lower, upper)

    # Remove small noise and fill small gaps.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def count_fingers(contour, frame_shape):
    """
    Estimate raised fingers using convexity defects.
    Returns (finger_count, hull_points, contour_area).
    """
    area = cv2.contourArea(contour)
    if area < 2500:
        return 0, None, area

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 4:
        return 0, None, area

    defects = cv2.convexityDefects(contour, hull_indices)
    if defects is None:
        return 0, None, area

    finger_gaps = 0
    for i in range(defects.shape[0]):
        start_idx, end_idx, far_idx, depth = defects[i, 0]

        start = contour[start_idx][0].astype(float)
        end = contour[end_idx][0].astype(float)
        far = contour[far_idx][0].astype(float)

        a = np.linalg.norm(end - start)
        b = np.linalg.norm(far - start)
        c = np.linalg.norm(end - far)

        if b == 0 or c == 0:
            continue

        # Angle at the deepest point of the convexity defect.
        cosine = (b * b + c * c - a * a) / (2 * b * c)
        cosine = np.clip(cosine, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))

        # Ignore shallow gaps and wide angles.
        if angle < 90 and depth / 256.0 > 12:
            finger_gaps += 1

    # A palm with N gaps usually has about N+1 raised fingers.
    fingers = min(finger_gaps + 1, 5)

    hull_points = cv2.convexHull(contour, returnPoints=True)
    return fingers, hull_points, area


def detect_hand_gesture(frame):
    """
    Find the largest skin-colored contour and estimate a simple gesture.
    Returns a display label and the skin mask.
    """
    mask = build_skin_mask(frame)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return "Hand: Not detected", mask

    hand_contour = max(contours, key=cv2.contourArea)
    fingers, hull, area = count_fingers(hand_contour, frame.shape)

    if area < 2500:
        return "Hand: Not detected", mask

    # Draw the detected hand contour and its convex hull.
    cv2.drawContours(frame, [hand_contour], -1, (255, 0, 0), 2)
    if hull is not None:
        cv2.drawContours(frame, [hull], -1, (0, 255, 255), 2)

    gesture_names = {
        0: "Fist / Closed hand",
        1: "One finger",
        2: "Two fingers",
        3: "Three fingers",
        4: "Four fingers",
        5: "Open palm"
    }

    return f"Gesture: {gesture_names.get(fingers, 'Unknown')}", mask


def main():
    face_detector = create_face_detector()

    # 0 is usually the default built-in webcam. Change to 1 if needed.
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check camera permissions or try camera index 1."
        )

    print("Webcam started. Press 'q' to quit.")

    while True:
        success, frame = camera.read()
        if not success:
            print("Could not read a frame from the webcam.")
            break

        # Mirror the frame for a natural webcam experience.
        frame = cv2.flip(frame, 1)

        face_count = detect_faces(frame, face_detector)
        gesture_label, hand_mask = detect_hand_gesture(frame)

        cv2.putText(
            frame, f"Faces detected: {face_count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            frame, gesture_label, (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
        )
        cv2.putText(
            frame, "Press Q to quit", (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
        )

        cv2.imshow("Face & Hand Gesture Detection", frame)
        cv2.imshow("Hand Segmentation Mask", hand_mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
