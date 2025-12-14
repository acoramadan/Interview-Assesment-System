import os
import logging
import cv2
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# ---- Matikan semua log YOLO ----
os.environ["YOLO_VERBOSE"] = "False"
LOGGER.setLevel(logging.ERROR)


class PeopleDetector:
    def __init__(self, model_path="yolov8n-pose.pt"):
        # YOLO pose model → jauh lebih akurat untuk deteksi manusia
        self.model = YOLO(model_path)

        # anti noise & stabilizer
        self.min_conf = 0.35          # confidence minimal
        self.min_area_ratio = 0.04    # area minimal bounding box
        self.last_valid_count = 0     # smoothing
        self.smooth_strength = 0.6    # semakin besar → semakin stabil

    def _is_valid_person(self, box, frame_w, frame_h):
        """Filter false positives based on confidence + bounding box area."""
        conf = float(box.conf[0])
        if conf < self.min_conf:
            return False

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        if area < self.min_area_ratio * frame_w * frame_h:
            return False

        return True

    def _has_skeleton(self, result):
        """YOLO pose: valid manusia harus punya keypoints."""
        return result.keypoints is not None and len(result.keypoints.xy) > 0

    def count_persons(self, frame):
        h, w = frame.shape[:2]

        # inference tanpa verbose
        results = self.model(frame, verbose=False)[0]
        count = 0

        # loop through detections
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])

            # YOLO pose hanya deteksi person saja → aman
            if cls_id != 0:
                continue

            # cek skeleton YOLO pose
            if not self._has_skeleton(results):
                continue

            # cek bounding box validity
            if not self._is_valid_person(box, w, h):
                continue

            count += 1

        # smoothing
        smoothed = int(self.smooth_strength * self.last_valid_count +
                       (1 - self.smooth_strength) * count)
        self.last_valid_count = smoothed

        return smoothed

    def detect_in_video(self, video_path):
        """Optional: debug helper to inspect person detection per video."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_id = 0
        summary = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            count = self.count_persons(frame)
            timestamp = frame_id / fps
            summary.append({
                "frame": frame_id,
                "timestamp_sec": timestamp,
                "person_count": count
            })

        cap.release()
        return summary
