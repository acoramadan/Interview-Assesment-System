# src/comvis/eye_movement.py
import cv2
import mediapipe as mp
import numpy as np

# Iris landmarks
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Eye blink landmarks (EAR)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# Lower eyelid (untuk DOWN detection)
LOWER_LID_L = [145, 153, 154]
LOWER_LID_R = [374, 380, 386]


class EyeGazeEstimator:
    def __init__(self, refine_landmarks=True):
        mp_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_mesh.FaceMesh(refine_landmarks=refine_landmarks)

        # Threshold arah — DISARANKAN untuk webcam rata-rata
        self.TH_LEFT = -0.12
        self.TH_RIGHT = 0.12
        self.TH_UP = -0.12
        self.TH_DOWN = 0.12

        # Blink threshold
        self.BLINK_EAR = 0.20

        # Threshold DOWN melalui eyelid distance (semakin kecil → iris mendekati kelopak bawah)
        self.DOWN_EYELID_TH = 3.5


    # =============================
    # UTIL
    # =============================
    def _get_iris_center(self, lm, idxs, w, h):
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in idxs])
        cx, cy = pts.mean(axis=0)
        return float(cx), float(cy)

    def _normalize_gaze(self, iris_x, iris_y, eye_left, eye_right):
        mid_x = (eye_left[0] + eye_right[0]) / 2
        mid_y = (eye_left[1] + eye_right[1]) / 2

        eye_w = max(abs(eye_right[0] - eye_left[0]), 1.0)

        gx = (iris_x - mid_x) / (eye_w / 2)
        gy = (iris_y - mid_y) / (eye_w / 2)

        # scaling untuk akurasi lebih baik
        gx *= 1.8
        gy *= 2.8

        return gx, gy

    def _EAR(self, lm, idxs, w, h):
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in idxs])
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def _iris_to_lowerlid_dist(self, lm, iris_y, eyelid_idxs, w, h):
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in eyelid_idxs])
        lid_y = pts[:, 1].mean()
        return lid_y - iris_y


    # =============================
    # MAIN PREDICT
    # =============================
    def predict_from_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        out = {
            "gaze_label": None,
            "gaze_probs": None,
            "gx": None,
            "gy": None,
            "left_iris": None,
            "right_iris": None,
            "status": "NO_FACE"
        }

        if not res.multi_face_landmarks:
            return out

        lm = res.multi_face_landmarks[0].landmark

        # ===== IRIS CENTER =====
        lx, ly = self._get_iris_center(lm, LEFT_IRIS, w, h)
        rx, ry = self._get_iris_center(lm, RIGHT_IRIS, w, h)

        out["left_iris"] = (lx, ly)
        out["right_iris"] = (rx, ry)

        # ===== EYE CORNERS =====
        eye_left_corner = (lm[33].x * w, lm[33].y * h)
        eye_right_corner = (lm[263].x * w, lm[263].y * h)

        # ===== GAZE NORMALIZATION =====
        gx_left, gy_left = self._normalize_gaze(lx, ly, eye_left_corner, eye_right_corner)
        gx_right, gy_right = self._normalize_gaze(rx, ry, eye_left_corner, eye_right_corner)

        gx = (gx_left + gx_right) / 2
        gy = (gy_left + gy_right) / 2

        out["gx"] = float(gx)
        out["gy"] = float(gy)
        out["status"] = "OK"

        # ===== BLINK (EAR) =====
        ear_L = self._EAR(lm, LEFT_EYE, w, h)
        ear_R = self._EAR(lm, RIGHT_EYE, w, h)
        ear_mean = (ear_L + ear_R) / 2

        # ===== FIXED BLINK DETECTION (strong anti-false-down) =====
        is_blink_ear = ear_mean < self.BLINK_EAR
        is_vertical_move = abs(gy) > 0.08      # mata bergerak atas/bawah
        is_horizontal_move = abs(gx) > 0.12    # mata bergerak kiri/kanan

        # BLINK hanya kalau:
        # - EAR turun
        # - TIDAK sedang menggerakkan mata ke atas/bawah
        # - TIDAK sedang menggerakkan mata kiri/kanan
        if is_blink_ear and not is_vertical_move and not is_horizontal_move:
            out["gaze_label"] = "BLINK"
            return out



        # ===== EYELID DISTANCE FOR DOWN =====
        lower_L = self._iris_to_lowerlid_dist(lm, ly, LOWER_LID_L, w, h)
        lower_R = self._iris_to_lowerlid_dist(lm, ry, LOWER_LID_R, w, h)
        lower_dist = (lower_L + lower_R) / 2

        # ===== CLASSIFICATION =====
        label = "CENTER"

        if gx < self.TH_LEFT:
            label = "LEFT"
        elif gx > self.TH_RIGHT:
            label = "RIGHT"
        elif gy < self.TH_UP:
            label = "UP"
        elif gy > self.TH_DOWN:
            label = "DOWN"

        # priority override for DOWN using eyelid distance
        # STABLE DOWN DETECTION (no false triggers)
        is_vertical_down = gy > 0.10          # iris turun cukup jauh
        is_eyelid_close  = lower_dist < 2.0    # eyelid benar-benar dekat
        is_not_side      = abs(gx) < 0.25      # bukan lihat kiri/kanan
        is_not_up        = gy > -0.05          # bukan lihat ke atas
        is_not_blink     = ear_mean > self.BLINK_EAR

        if is_vertical_down and is_eyelid_close and is_not_side and is_not_up and is_not_blink:
            label = "DOWN"


        out["gaze_label"] = label
        return out
