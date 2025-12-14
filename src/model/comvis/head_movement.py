import cv2
import numpy as np
import mediapipe as mp


class HeadPoseEstimator:
    def __init__(self):
        # Mediapipe landmark tracker
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1
        )

        # Reference 3D face model (OpenCV standard)
        self.MODEL_POINTS = np.array([
            (0.0, 0.0, 0.0),          # nose tip
            (0.0, -330.0, -65.0),     # chin
            (-225.0, 170.0, -135.0),  # left eye left corner
            (225.0, 170.0, -135.0),   # right eye right corner
            (-150.0, -150.0, -125.0), # left mouth corner
            (150.0, -150.0, -125.0)   # right mouth corner
        ])

        # Matching Mediapipe landmarks for MODEL_POINTS
        self.LANDMARK_IDX = [1, 199, 33, 263, 61, 291]

        # ===== HYBRID CALIBRATION =====
        self.base_yaw = 0.0
        self.base_pitch = 0.0
        self.cal_samples = 0
        self.CALIBRATION_LIMIT = 40   # collect first ~40 frames (~1 second)
        self.hybrid_ready = False

        # ===== SMOOTHING =====
        self.prev_yaw = None
        self.prev_pitch = None
        self.ALPHA = 0.7  # smoothing factor


    # ---------------------------------------------------------
    def smooth(self, prev, new):
        if prev is None:
            return new
        return prev * self.ALPHA + new * (1 - self.ALPHA)


    # ---------------------------------------------------------
    def rotationMatrixToEulerAngles(self, R):
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2,1], R[2,2])  # pitch
            y = np.arctan2(-R[2,0], sy)     # yaw
            z = np.arctan2(R[1,0], R[0,0])  # roll
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0

        return np.degrees(y), np.degrees(x), np.degrees(z)


    # ---------------------------------------------------------
    def update_baseline(self, yaw, pitch):
        """Hybrid calibration: Only correct offsets if user remains stable."""
        if self.hybrid_ready:
            return

        # accumulate frames
        self.base_yaw += yaw
        self.base_pitch += pitch
        self.cal_samples += 1

        # when enough samples collected → freeze baseline
        if self.cal_samples >= self.CALIBRATION_LIMIT:
            self.base_yaw /= self.cal_samples
            self.base_pitch /= self.cal_samples
            self.hybrid_ready = True

            print("=== Hybrid Calibration Done ===")
            print(f"Yaw baseline   = {self.base_yaw:.2f}")
            print(f"Pitch baseline = {self.base_pitch:.2f}")


    # ---------------------------------------------------------
    def interpret_head(self, yaw_rel, pitch_rel):
        """
        Interpret movement based on RELATIVE differences from baseline.
        """

        # tolerance window for CENTER (user may be naturally tilted)
        if abs(yaw_rel) < 15 and abs(pitch_rel) < 12:
            return "CENTER"

        if yaw_rel > 25:
            return "LOOKING RIGHT"
        elif yaw_rel < -25:
            return "LOOKING LEFT"

        if pitch_rel > 20:
            return "LOOKING UP"
        elif pitch_rel < -20:
            return "LOOKING DOWN"

        return "CENTER"


    # ---------------------------------------------------------
    def estimate_from_frame(self, frame):
        """
        Returns:
            yaw_rel, pitch_rel, roll_raw, label
        """

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            return None, None, None, "NO_FACE"

        face = res.multi_face_landmarks[0]

        # 2D points from mediapipe
        image_points = np.array(
            [(face.landmark[i].x * w, face.landmark[i].y * h) for i in self.LANDMARK_IDX],
            dtype="double"
        )

        # Camera intrinsics
        focal_length = w
        center = (w/2, h/2)
        cam_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4,1))

        success, rot_vec, _ = cv2.solvePnP(
            self.MODEL_POINTS, image_points, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None, None, None, "NO_FACE"

        rmat, _ = cv2.Rodrigues(rot_vec)
        yaw_raw, pitch_raw, roll = self.rotationMatrixToEulerAngles(rmat)

        # ========= HYBRID BASELINE UPDATE =========
        self.update_baseline(yaw_raw, pitch_raw)

        # ========= RELATIVE MOTION =========
        yaw_rel = yaw_raw - self.base_yaw
        pitch_rel = pitch_raw - self.base_pitch

        # ========= SMOOTHING =========
        yaw_s = self.smooth(self.prev_yaw, yaw_rel)
        pitch_s = self.smooth(self.prev_pitch, pitch_rel)
        self.prev_yaw = yaw_s
        self.prev_pitch = pitch_s

        # ========= CLASSIFICATION =========
        label = self.interpret_head(yaw_s, pitch_s)

        return yaw_s, pitch_s, roll, label
