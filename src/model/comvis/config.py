import numpy as np
import time

TARGET_FPS = 8
PROCESS_INTERVAL = 1.0 / TARGET_FPS

FACING_YAW_DEG = 30.0
FACING_PITCH_DEG = 20.0
MIN_FACING_FOR_GAZE = 16.0

HYSTERESIS_FRAMES = 8
EMA_ALPHA = 0.45

IOU_THRESH = 0.3
MAX_MISSED_SEC = 2.0

CHEAT_MIN_OFFPOSE_SEC = 1.8
CHEAT_MIN_MULTIFACE_SEC = 0.8
CHEAT_MIN_OUTOFFRAME_SEC = 1.0
CHEAT_MIN_EYESOFF_SEC = 0.8
CHEAT_MIN_EYESMOV_SEC = 0.6

CALIB_SECONDS = 2.0

GT_X_DELTA = 0.22              
GT_Y_DELTA = 0.28                 
GT_SPEED_THR = 0.25              
GT_MIN_EOPEN = 0.20             
GT_SPEED_SMOOTH = 0.5             

HEAD_VEL_SMOOTH = 0.4
HEAD_STATIC_VEL_THR = 35.0

SEGMENTS_TO_CSV = True
SEGMENTS_TO_JSON = True
OUT_DIR = "../result/comvis_output"
SESSION_PREFIX = time.strftime("%Y%m%d_%H%M%S")

SHOW_DEBUG = False

LMK_IDX = [33, 263, 1, 61, 291, 199]
MODEL_3D = np.array([
    [-43.3,  32.7, -26.0],
    [ 43.3,  32.7, -26.0],
    [  0.0,   0.0,   0.0],
    [-28.9, -28.9, -24.1],
    [ 28.9, -28.9, -24.1],
    [  0.0, -63.6, -12.5],
], dtype=np.float64)