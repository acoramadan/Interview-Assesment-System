import time
from .config import EMA_ALPHA, LMK_IDX, MODEL_3D
import numpy as np
import cv2

def now_ts(): return time.time()

def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-6)

def ema(prev, new, alpha= EMA_ALPHA):
    return new if prev is None else (alpha*new + (1-alpha)*prev)

def hhmmss(sec):
    sec = max(0.0, float(sec))
    h = int(sec//3600); m = int((sec%3600)//60); s = sec%60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def estimate_head_pose_flexible(w, h, lm2d, lm3d_or_none):
    pts2d = np.array([[lm2d[i].x*w, lm2d[i].y*h] for i in LMK_IDX], dtype=np.float64)
    if lm3d_or_none is not None:
        pts3d = np.array([[lm3d_or_none[i].x, lm3d_or_none[i].y, lm3d_or_none[i].z] for i in LMK_IDX], dtype=np.float64)

    else:
        pts3d = MODEL_3D.copy()

    K = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
    dist = np.zeros((4,1))
    ok, rvec, _ = cv2.solvePnP(pts3d, pts2d, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)

    if not ok: return None, None, None

    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)

    if sy < 1e-6:
        pitch = np.degrees(np.arctan2(-R[1,2], R[1,1])); yaw = np.degrees(np.arctan2(-R[2,0], sy)); roll = 0.0

    else:
        pitch = np.degrees(np.arctan2(R[2,1], R[2,2]))
        yaw   = np.degrees(np.arctan2(-R[2,0], sy))
        roll  = np.degrees(np.arctan2(R[1,0], R[0,0]))

    return yaw, pitch, roll