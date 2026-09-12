import cv2
import os
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# MediaPipe Iris and Eye Corner Landmark Indices
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_L_CORNER = 33
LEFT_EYE_R_CORNER = 133
RIGHT_EYE_L_CORNER = 362
RIGHT_EYE_R_CORNER = 263

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("Downloading Face Landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def calculate_iris_ratio(iris_center, left_corner, right_corner):
    """Calculates horizontal position of iris relative to eye width (0.0 to 1.0)."""
    total_width = np.linalg.norm(np.array(right_corner) - np.array(left_corner))
    if total_width == 0:
        return 0.5
    iris_dist = np.linalg.norm(np.array(iris_center) - np.array(left_corner))
    return iris_dist / total_width

def run_eye_tracker():
    download_model_if_needed()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Reverted back to live hardware camera index
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Could not open live webcam on camera index 0.")
        return

    print("Running live vision telemetry... Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
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

            l_ratio = calculate_iris_ratio(l_iris, l_corner_l, l_corner_r)

            if 0.35 <= l_ratio <= 0.65:
                status_text = f"Eye Contact: OK ({int(l_ratio * 100)}%)"
                color = (0, 255, 0)
            else:
                status_text = "Looking Away"
                color = (0, 0, 255)

            cv2.circle(frame, l_iris, 3, (255, 0, 255), -1)
            cv2.circle(frame, r_iris, 3, (255, 0, 255), -1)

        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Project A.I.M. - Eye Telemetry Test', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_eye_tracker()