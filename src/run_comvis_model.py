import os
import time
import json
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Callable
from huggingface_hub import hf_hub_download

import cv2
import numpy as np

# imports modul inference
from model.comvis.eye_movement import EyeGazeEstimator
from model.comvis.head_movement import HeadPoseEstimator
from model.comvis.people_detection import PeopleDetector


# -----------------------
# CONFIG
# -----------------------
VIDEO_PATH = "../data/interview_question_2.webm"

EYE_MODEL_PATH = hf_hub_download(
    repo_id="NazeeraAlthea/comvis-model",
    filename="unityeyes_eye_model.pkl"
)

YOLO_MODEL_PATH = hf_hub_download(
    repo_id="NazeeraAlthea/comvis-model",
    filename="yolov8n.pt"
)

# thresholds
EYE_CHEAT_MIN_DURATION = 1.0
HEAD_CHEAT_MIN_DURATION = 1.0
PEOPLE_MIN_DURATION = 0.3

HEAD_YAW_THRESHOLD = 25.0
HEAD_PITCH_THRESHOLD = 20.0

RESULT_DIR = "../result/comvis_output"
os.makedirs(RESULT_DIR, exist_ok=True)


# -----------------------
# Helper Functions
# -----------------------
def format_hhmmss(seconds: float) -> str:
    td = timedelta(seconds=float(seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = (total_seconds % 60)
    ms = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def now_ts() -> float:
    return time.time()


def csv_dump(path: str, fieldnames: List[str], rows: List[Dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------
# Event utilities
# -----------------------
def extract_intervals_from_boolean_series(series: List[Dict],
                                          cond_fn: Callable[[Dict], bool],
                                          min_duration: float) -> List[Dict]:
    intervals = []
    start = None
    end = None

    for row in series:
        if cond_fn(row):
            if start is None:
                start = row["timestamp"]
            end = row["timestamp"]
        else:
            if start is not None and (end - start) >= min_duration:
                intervals.append({"start": start, "end": end})
            start = None
            end = None

    if start is not None and end is not None and (end - start) >= min_duration:
        intervals.append({"start": start, "end": end})

    return intervals


def build_segments_from_events(events: Dict[str, List[Dict]], session_start_unix: float):
    segments = []
    track_id = 1

    for reason, intervals in events.items():
        for iv in intervals:
            start_offset = float(iv["start"])
            end_offset = float(iv["end"])
            duration = end_offset - start_offset

            segments.append({
                "track_id": track_id,
                "reason": reason,
                "start_ts": session_start_unix + start_offset,
                "end_ts": session_start_unix + end_offset,
                "duration_sec": duration,
                "start_hhmmss": format_hhmmss(start_offset),
                "end_hhmmss": format_hhmmss(end_offset)
            })

    return sorted(segments, key=lambda x: x["start_ts"])


# -----------------------
# Main Pipeline
# -----------------------
def run_full_pipeline(video_path: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print("Loading models...")
    eye_model = EyeGazeEstimator(model_path=EYE_MODEL_PATH)
    head_model = HeadPoseEstimator()
    people_model = PeopleDetector(model_path=YOLO_MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_id = 0

    eye_frames = []
    head_frames = []
    people_frames = []

    print("Starting inference loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        timestamp = frame_id / fps

        # Eye movement
        eye_out = eye_model.predict_from_frame(frame)
        eye_frames.append({
            "frame": frame_id,
            "timestamp": timestamp,
            "gaze": eye_out.get("gaze_label", "UNKNOWN"),
            "gx": eye_out.get("gx"),
            "gy": eye_out.get("gy"),
            "status": eye_out.get("status", "N/A"),
            "iris_quality": eye_out.get("iris_quality")
        })

        # Head pose
        yaw, pitch, roll, label = head_model.estimate_from_frame(frame)
        head_frames.append({
            "frame": frame_id,
            "timestamp": timestamp,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "label": label
        })

        # People detection
        count = people_model.count_persons(frame)
        people_frames.append({
            "frame": frame_id,
            "timestamp": timestamp,
            "person_count": int(count)
        })

        if frame_id % int(fps * 5) == 0:
            print(f"Processed {frame_id} frames (~{timestamp:.1f}s)")

    cap.release()
    print("Inference finished.")

    return eye_frames, head_frames, people_frames, fps, frame_id


# -----------------------
# Postprocessing
# -----------------------
def postprocess_and_export(eye_frames, head_frames, people_frames, fps, total_frames):

    session_start_unix = now_ts()
    session_start_readable = datetime.fromtimestamp(session_start_unix).strftime("%Y-%m-%d %H:%M:%S")

    print("Building events...")

    events = {
        "HEAD_POSE_OFF": extract_intervals_from_boolean_series(
            head_frames,
            lambda r: abs(r["yaw"]) > HEAD_YAW_THRESHOLD or abs(r["pitch"]) > HEAD_PITCH_THRESHOLD,
            HEAD_CHEAT_MIN_DURATION
        ),
        "EYES_MOVING": extract_intervals_from_boolean_series(
            eye_frames,
            lambda r: (r["gaze"].lower() != "center" and r.get("status") in ("OK", "ok", None)),
            EYE_CHEAT_MIN_DURATION
        ),
        "MULTI_PERSON": extract_intervals_from_boolean_series(
            people_frames,
            lambda r: r["person_count"] > 1,
            PEOPLE_MIN_DURATION
        ),
        "EYES_OFF": extract_intervals_from_boolean_series(
            eye_frames,
            lambda r: r.get("status") not in ("OK", "ok", None),
            0.1
        ),
    }

    segments = build_segments_from_events(events, session_start_unix)

    output = {
        "session_start_unix": session_start_unix,
        "session_start_readable": session_start_readable,
        "video_path": VIDEO_PATH,
        "fps": fps,
        "total_frames": total_frames,
        "segments": segments
    }

    ts_str = datetime.fromtimestamp(session_start_unix).strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(RESULT_DIR, f"{ts_str}_segments.json")
    csv_path = os.path.join(RESULT_DIR, f"{ts_str}_segments.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    csv_dump(csv_path,
             ["track_id", "reason", "start_ts", "end_ts",
              "duration_sec", "start_hhmmss", "end_hhmmss"],
             segments)

    print("Export done.")
    return output, json_path, csv_path


# -----------------------
# CLI
# -----------------------
def main():
    print("Video:", VIDEO_PATH)

    start = time.time()
    eye_frames, head_frames, people_frames, fps, total_frames = run_full_pipeline(VIDEO_PATH)
    output, jpath, cpath = postprocess_and_export(
        eye_frames, head_frames, people_frames, fps, total_frames
    )

    end = time.time()
    print(f"All done in {end - start:.1f}s")
    print("JSON saved at:", jpath)
    print("CSV saved at:", cpath)


if __name__ == "__main__":
    main()
