import numpy as np
from .utils import now_ts
from .config import CALIB_SECONDS

class HeadCalib:
    def __init__(self):
        self.active = True
        self.t0 = None
        self.yaw_list = []; self.pitch_list = []
        self.bias_yaw = 0.0; self.bias_pitch = 0.0

    def start(self):
        self.active = True; self.t0 = now_ts()
        self.yaw_list.clear(); self.pitch_list.clear()

    def feed(self, yaw, pitch):
        if yaw is not None and pitch is not None:
            self.yaw_list.append(yaw); self.pitch_list.append(pitch)

    def done_if_ready(self):
        if not self.active or (now_ts() - self.t0) < CALIB_SECONDS: return False
        if self.yaw_list:   self.bias_yaw = float(np.median(self.yaw_list))
        if self.pitch_list: self.bias_pitch = float(np.median(self.pitch_list))
        self.active = False
        return True