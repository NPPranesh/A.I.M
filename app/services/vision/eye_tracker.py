import os
import sys
import cv2
import urllib.request
import numpy as np

# Suppress Python-level warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_L_CORNER = 33
LEFT_EYE_R_CORNER = 133

class SuppressStderr:
    """Redirects C++ low-level stderr stream to os.devnull to silence TFLite/Abseil warnings."""
    def __enter__(self):
        sys.stderr.flush()
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save_fd = os.dup(2)
        os.dup2(self.null_fd, 2)

    def __exit__(self, *args):
        os.dup2(self.save_fd, 2)
        os.close(self.null_fd)
        os.close(self.save_fd)

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def enhance_frame_for_dark_lighting(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def calculate_eye_contact_score(iris_center, left_corner, right_corner):
    min_x = min(left_corner[0], right_corner[0])
    max_x = max(left_corner[0], right_corner[0])
    total_width = max_x - min_x
    
    if total_width <= 2:
        return 50

    h_ratio = (iris_center[0] - min_x) / float(total_width)
    deviation = abs(h_ratio - 0.50)
    max_allowed_deviation = 0.20  
    
    score = int((1.0 - min(1.0, (deviation / max_allowed_deviation))) * 100)
    return max(0, min(100, score))

def run_eye_tracker():
    download_model_if_needed()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )

    # Mute C++ internal logging during MediaPipe detector creation
    with SuppressStderr():
        detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Camera 0 failed to open.")
        return

    print("Eye telemetry running. Click video window and press 'q' or 'ESC' to quit.")

    while True:
        success, raw_frame = cap.read()
        if not success or raw_frame is None:
            continue

        raw_frame = cv2.flip(raw_frame, 1)
        enhanced_frame = enhance_frame_for_dark_lighting(raw_frame)

        h, w, _ = raw_frame.shape
        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Mute C++ logging during inference calls
        with SuppressStderr():
            detection_result = detector.detect(mp_image)

        status_text = "No Face Detected"
        color = (0, 0, 255)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            def get_point(idx):
                return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

            l_iris = get_point(LEFT_IRIS[0])
            r_iris = get_point(RIGHT_IRIS[0])
            l_corner_l = get_point(LEFT_EYE_L_CORNER)
            l_corner_r = get_point(LEFT_EYE_R_CORNER)

            score = calculate_eye_contact_score(l_iris, l_corner_l, l_corner_r)

            if score >= 70:
                status_text = f"Eye Contact: EXCELLENT ({score}%)"
                color = (0, 255, 0)
            elif score >= 40:
                status_text = f"Eye Contact: FAIR ({score}%)"
                color = (0, 255, 255)
            else:
                status_text = f"Looking Away ({score}%)"
                color = (0, 0, 255)

            cv2.circle(raw_frame, l_iris, 3, (255, 0, 255), -1)
            cv2.circle(raw_frame, r_iris, 3, (255, 0, 255), -1)

        cv2.putText(raw_frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow('Project A.I.M. - Eye Telemetry Test', raw_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_eye_tracker()