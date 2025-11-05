import sys
from .config import * 
import numpy as np

from pathlib import Path
ROOT = Path.cwd().parent          
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lib.GazeTracking.gaze_tracking import GazeTracking
from .utils import now_ts


class GazeAdapter: 
    def __init__(self):
        self.gaze = GazeTracking()
        self.center_x = 0.0
        self.center_y = 0.0
        self.active_calib = True
        self.t0 = None
        self.gx_list, self.gy_list = [], []

    def start_calib(self): 
        self.active_calib = True
        self.t0 = now_ts()
        self.gx_list.clear(); self.gy_list.clear()

    def _roi_from_bbox(self, frame, bbox, margin=0.2): 
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2-x1, y2-y1
        mx, my = int(bw*margin), int(bh*margin)
        X1 = max(0, x1 - mx); Y1 = max(0, y1 - my)
        X2 = min(w, x2 + mx); Y2 = min(h, y2 + my)
        return frame[Y1:Y2, X1:X2].copy(), (X1, Y1)

    def refresh_on_bbox(self, frame, bbox):  
        roi, (ox, oy) = self._roi_from_bbox(frame, bbox)

        if roi.size == 0: return None

        self.gaze.refresh(roi)
        hx = self.gaze.horizontal_ratio()
        hy = self.gaze.vertical_ratio()

        if hx is None or hy is None:
            return {"gx": None, "gy": None, "blink": self.gaze.is_blinking(),
                    "left": self.gaze.is_left(), "right": self.gaze.is_right(), "center": self.gaze.is_center()}
        
        gx = float((hx - 0.5) * 2.0)  
        gy = float((hy - 0.5) * 2.0)  

        if self.active_calib:
            self.gx_list.append(gx); self.gy_list.append(gy)

        return {"gx": gx, "gy": gy, "blink": self.gaze.is_blinking(),
                "left": self.gaze.is_left(), "right": self.gaze.is_right(), "center": self.gaze.is_center()}

    def finish_calib_if_ready(self, seconds=CALIB_SECONDS): 
        if not self.active_calib: return False

        if self.t0 is None or (now_ts() - self.t0) < seconds: return False

        if self.gx_list:
            self.center_x = float(np.median(self.gx_list))

        if self.gy_list:
            self.center_y = float(np.median(self.gy_list))

        self.active_calib = False
        return True