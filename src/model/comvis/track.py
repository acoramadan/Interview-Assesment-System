import numpy as np
from .utils import now_ts, ema
from .config import HEAD_VEL_SMOOTH, HEAD_STATIC_VEL_THR, GT_X_DELTA, GT_Y_DELTA, GT_SPEED_THR, GT_SPEED_SMOOTH, HYSTERESIS_FRAMES
class Track:
    _next_id = 1

    def __init__(self, bbox):
        self.id = Track._next_id; Track._next_id += 1
        self.bbox = bbox
        self.ts_last = now_ts()
        self.yaw_s = None; self.pitch_s = None; self.roll_s = None
        self.prev_yaw = None; self.prev_pitch = None; self.prev_ang_t = None
        self.head_vel = None
        self.facing_prob = None
        self.state = "UNKNOWN"
        self._up = 0; self._dn = 0
        self.ts_not_focus_start = None
        self.gx_s = None; self.gy_s = None
        self.prev_gx = None; self.prev_gy = None; self.prev_t = None
        self.gaze_speed = None
        self.ts_eyes_off_start = None
        self.ts_eyes_moving_start = None
        self.flags = {"HEAD_POSE_OFF":False, "EYES_OFF":False, "EYES_MOVING":False, "OUT_OF_FRAME":False}
        self.cheat_active = False
        self.cheat_reason = None
        self.ts_cheat_last = None

    def update_pose(self, yaw, pitch, roll):
        self.yaw_s = ema(self.yaw_s, yaw)
        self.pitch_s = ema(self.pitch_s, pitch)
        self.roll_s = ema(self.roll_s, roll)
        t = now_ts()

        if self.prev_yaw is not None and self.prev_pitch is not None and self.prev_ang_t is not None:
            dt = max(1e-3, t - self.prev_ang_t)
            dy = (self.yaw_s - self.prev_yaw) if self.yaw_s is not None else 0.0
            dp = (self.pitch_s - self.prev_pitch) if self.pitch_s is not None else 0.0
            vel = np.hypot(dy, dp) / dt
            self.head_vel = ema(self.head_vel, vel, alpha=HEAD_VEL_SMOOTH)
        self.prev_yaw, self.prev_pitch, self.prev_ang_t = self.yaw_s, self.pitch_s, t

    def update_bbox(self, bbox):
        self.bbox = bbox; self.ts_last = now_ts()

    def update_facing(self, facing_now):
        p = 1.0 if facing_now else 0.0
        self.facing_prob = ema(self.facing_prob, p)

        if facing_now: self._up += 1; self._dn = 0

        else: self._dn += 1; self._up = 0

        if self.state in ("UNKNOWN","NOT_FOCUS"):
            if self._up >= HYSTERESIS_FRAMES and (self.facing_prob or 0) >= 0.6:
                self.state = "FOCUS"

        if self.state in ("UNKNOWN","FOCUS"):
            if self._dn >= HYSTERESIS_FRAMES and (self.facing_prob or 1) <= 0.4:
                self.state = "NOT_FOCUS"

        t = now_ts()

        if self.state == "NOT_FOCUS":
            if self.ts_not_focus_start is None: self.ts_not_focus_start = t

        else:
            self.ts_not_focus_start = None

    def update_gaze(self, gx, gy, allow_eval):
        if gx is None or gy is None:
            self.ts_eyes_off_start = None
            self.ts_eyes_moving_start = None
            self.prev_gx = gx; self.prev_gy = gy; self.prev_t = now_ts()
            return
        
        self.gx_s = ema(self.gx_s, gx)
        self.gy_s = ema(self.gy_s, gy)
        t = now_ts()

        if self.prev_gx is not None and self.prev_gy is not None and self.prev_t is not None:
            dt = max(1e-3, t - self.prev_t)
            dx = self.gx_s - self.prev_gx
            dy = self.gy_s - self.prev_gy
            spd = (dx**2 + dy**2)**0.5 / dt
            self.gaze_speed = ema(self.gaze_speed, spd, alpha=GT_SPEED_SMOOTH)
        self.prev_gx = self.gx_s; self.prev_gy = self.gy_s; self.prev_t = t

        if not allow_eval:
            self.ts_eyes_off_start = None
            self.ts_eyes_moving_start = None
            return
        
        devx = abs(self.gx_s)
        devy = abs(self.gy_s)
        off = (devx > GT_X_DELTA) or (devy > GT_Y_DELTA)

        if off:
            if self.ts_eyes_off_start is None: self.ts_eyes_off_start = t
        else:
            self.ts_eyes_off_start = None

        head_static = (self.head_vel or 0.0) < HEAD_STATIC_VEL_THR
        fast = (self.gaze_speed or 0.0) > GT_SPEED_THR
        moving = head_static and fast

        if moving:
            if self.ts_eyes_moving_start is None: self.ts_eyes_moving_start = t
        else:
            self.ts_eyes_moving_start = None

    def mark_flag(self, key, val): self.flags[key] = val

    def mark_cheat(self, reason):
        self.cheat_active = True; self.cheat_reason = reason; self.ts_cheat_last = now_ts()

    def clear_cheat(self):
        self.cheat_active = False; self.cheat_reason = None; self.ts_cheat_last = now_ts()
