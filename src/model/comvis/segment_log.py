import os
import csv
import json
import time
from .utils import now_ts, hhmmss
from .track import Track
from .config import SEGMENTS_TO_CSV, SEGMENTS_TO_JSON

class SegmentLogger:
    def __init__(self, out_dir, prefix, session_start):
        self.out_dir = out_dir; os.makedirs(out_dir, exist_ok=True)
        self.prefix = prefix; self.session_start = session_start
        self.open_segments = {}
        self.closed_segments = []
        self.csv_path = os.path.join(out_dir, f"{prefix}_segments.csv")
        self.json_path = os.path.join(out_dir, f"{prefix}_segments.json")

        if SEGMENTS_TO_CSV and not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["track_id","reason","start_ts","end_ts","duration_sec","start_hhmmss","end_hhmmss"])

    def _append_closed(self, seg):
        self.closed_segments.append(seg)
        if SEGMENTS_TO_CSV:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([seg["track_id"], seg["reason"],
                            f'{seg["start_ts"]:.3f}', f'{seg["end_ts"]:.3f}',
                            f'{seg["duration_sec"]:.3f}',
                            seg["start_hhmmss"], seg["end_hhmmss"]])
                
    def update_track(self, track: Track, current_reason: str | None):
        tid = track.id; tnow = now_ts()

        if tid in self.open_segments:
            seg = self.open_segments[tid]

            if current_reason != seg["reason"]:
                seg["end_ts"] = tnow
                seg["duration_sec"] = seg["end_ts"] - seg["start_ts"]
                seg["start_hhmmss"] = hhmmss(seg["start_ts"] - self.session_start)
                seg["end_hhmmss"] = hhmmss(seg["end_ts"] - self.session_start)
                self._append_closed(seg)
                del self.open_segments[tid]

        if current_reason is not None and tid not in self.open_segments:
            self.open_segments[tid] = {"track_id": tid, "reason": current_reason, "start_ts": tnow}

    def close_all(self):
        tnow = now_ts()

        for tid, seg in list(self.open_segments.items()):
            seg["end_ts"] = tnow
            seg["duration_sec"] = seg["end_ts"] - seg["start_ts"]
            seg["start_hhmmss"] = hhmmss(seg["start_ts"] - self.session_start)
            seg["end_hhmmss"] = hhmmss(seg["end_ts"] - self.session_start)
            self._append_closed(seg)
            del self.open_segments[tid]

        if SEGMENTS_TO_JSON:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump({"session_start_unix": self.session_start,
                           "session_start_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.session_start)),
                           "segments": self.closed_segments}, f, ensure_ascii=False, indent=2)