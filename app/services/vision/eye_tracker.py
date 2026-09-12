import os
import sys
import cv2
import json
import urllib.request
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

NOSE_TIP = 1
LEFT_IRIS = 468
RIGHT_IRIS = 473
LEFT_EYE_L, LEFT_EYE_R = 33, 133
RIGHT_EYE_L, RIGHT_EYE_R = 362, 263

LEFT_EYE_TOP, LEFT_EYE_BOT = 159, 145
RIGHT_EYE_TOP, RIGHT_EYE_BOT = 386, 374

class SuppressStderr:
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
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    limg = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

class EyeTrackerService:
    def __init__(self):
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
        with SuppressStderr():
            self.detector = vision.FaceLandmarker.create_from_options(options)
        
        self.smoothed_score = 100.0
        self.alpha = 0.20

    def _is_blinking(self, landmarks, w, h):
        def p(idx): return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
        l_dist = np.linalg.norm(p(LEFT_EYE_TOP) - p(LEFT_EYE_BOT))
        l_width = np.linalg.norm(p(LEFT_EYE_L) - p(LEFT_EYE_R))
        r_dist = np.linalg.norm(p(RIGHT_EYE_TOP) - p(RIGHT_EYE_BOT))
        r_width = np.linalg.norm(p(RIGHT_EYE_L) - p(RIGHT_EYE_R))
        ear = ((l_dist / max(1, l_width)) + (r_dist / max(1, r_width))) / 2.0
        return ear < 0.14

    def _calc_single_eye_score(self, iris, corner_l, corner_r):
        min_x, max_x = min(corner_l[0], corner_r[0]), max(corner_l[0], corner_r[0])
        width = max_x - min_x
        if width <= 2: return 50
        h_ratio = (iris[0] - min_x) / float(width)
        dev = abs(h_ratio - 0.50)
        score = int((1.0 - min(1.0, dev / 0.20)) * 100)
        return max(0, min(100, score))

    def process_frame(self, raw_frame):
        flipped = cv2.flip(raw_frame, 1)
        enhanced = enhance_frame_for_dark_lighting(flipped)
        h, w, _ = flipped.shape
        rgb_frame = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        with SuppressStderr():
            detection_result = self.detector.detect(mp_image)

        is_centered = False
        raw_score = 0
        landmarks_data = None

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            landmarks_data = landmarks
            
            nose = landmarks[NOSE_TIP]
            is_centered = (0.30 <= nose.x <= 0.70) and (0.25 <= nose.y <= 0.75)

            if self._is_blinking(landmarks, w, h):
                raw_score = int(self.smoothed_score)
            else:
                def get_pt(idx): return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
                l_score = self._calc_single_eye_score(get_pt(LEFT_IRIS), get_pt(LEFT_EYE_L), get_pt(LEFT_EYE_R))
                r_score = self._calc_single_eye_score(get_pt(RIGHT_IRIS), get_pt(RIGHT_EYE_L), get_pt(RIGHT_EYE_R))
                raw_score = int((l_score + r_score) / 2.0)

        self.smoothed_score = (self.alpha * raw_score) + ((1.0 - self.alpha) * self.smoothed_score)
        final_score = int(np.clip(self.smoothed_score, 0, 100))

        return {
            "type": "VISION_TELEMETRY",
            "payload": {
                "eye_contact_score": final_score,
                "is_face_centered": is_centered
            },
            "_raw_landmarks": landmarks_data
        }

def draw_futuristic_hud(frame, telemetry):
    h, w, _ = frame.shape
    payload = telemetry["payload"]
    score = payload["eye_contact_score"]
    is_centered = payload["is_face_centered"]
    landmarks = telemetry["_raw_landmarks"]

    if score >= 75: color, status = (0, 255, 127), "OPTIMAL FOCUS"
    elif score >= 45: color, status = (0, 215, 255), "FAIR ALIGNMENT"
    else: color, status = (0, 69, 255), "OFF-TARGET"

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, 80), (w, 80), color, 2)

    cv2.putText(frame, "PROJECT A.I.M. // VISION TELEMETRY Engine", (20, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, f"GAZE SCORE: {score}%", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"STATUS: {status}", (260, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)

    bar_width = int((score / 100.0) * (w - 40))
    cv2.rectangle(frame, (20, 72), (w - 20, 76), (40, 40, 50), -1)
    cv2.rectangle(frame, (20, 72), (20 + bar_width, 76), color, -1)

    box_color = (0, 255, 127) if is_centered else (0, 0, 255)
    cv2.rectangle(frame, (int(w * 0.30), int(h * 0.25)), (int(w * 0.70), int(h * 0.75)), box_color, 1)

    if landmarks:
        def pt(idx): return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
        
        for iris_idx in (LEFT_IRIS, RIGHT_IRIS):
            p = pt(iris_idx)
            cv2.circle(frame, p, 6, color, 1)
            cv2.circle(frame, p, 2, (255, 255, 255), -1)
            cv2.line(frame, (p[0] - 10, p[1]), (p[0] + 10, p[1]), color, 1)
            cv2.line(frame, (p[0], p[1] - 10), (p[0], p[1] + 10), color, 1)

if __name__ == "__main__":
    service = EyeTrackerService()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    print("AI Telemetry Engine active. Press 'q' or 'ESC' to terminate.")

    while True:
        success, raw_frame = cap.read()
        if not success or raw_frame is None: continue

        telemetry = service.process_frame(raw_frame)
        print(json.dumps(telemetry["payload"]))

        display_frame = cv2.flip(raw_frame, 1)
        draw_futuristic_hud(display_frame, telemetry)
        cv2.imshow("Project A.I.M. - Telemetry Stream", display_frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27): break

    cap.release()
    cv2.destroyAllWindows()