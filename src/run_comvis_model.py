from model.comvis.config import *
from model.comvis.utils import now_ts, estimate_head_pose_flexible
from model.comvis.tracker import Tracker
from model.comvis.head_calib import HeadCalib
from model.comvis.gaze_adapter import GazeAdapter
from model.comvis.segment_log import SegmentLogger
import cv2
import mediapipe as mp


def main():
    session_start = now_ts()
    seglog = SegmentLogger(OUT_DIR, SESSION_PREFIX, session_start)

    cap = cv2.VideoCapture(0)
    last_proc = 0.0
    tracker = Tracker()

    head_cal = HeadCalib(); head_cal.start()
    gaze_adapt = GazeAdapter(); gaze_adapt.start_calib() 

    mp_fd = mp.solutions.face_detection
    mp_fm = mp.solutions.face_mesh

    with mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd, \
         mp_fm.FaceMesh(max_num_faces=5, refine_landmarks=False, 
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as fm:

        while True:
            ok, frame = cap.read()
            if not ok: break
            h, w = frame.shape[:2]
            draw = frame.copy()
            tnow = now_ts()
            do_process = (tnow - last_proc) >= PROCESS_INTERVAL

            if do_process:
                last_proc = tnow
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                bboxes = []
                det = fd.process(rgb)
                if det and det.detections:
                    for d in det.detections:
                        rel = d.location_data.relative_bounding_box
                        x1 = int(rel.xmin*w); y1 = int(rel.ymin*h)
                        x2 = int((rel.xmin+rel.width)*w); y2 = int((rel.ymin+rel.height)*h)
                        x1 = max(0,x1); y1 = max(0,y1); x2 = min(w-1,x2); y2 = min(h-1,y2)
                        if x2>x1 and y2>y1: bboxes.append([x1,y1,x2,y2])

                tracks = tracker.update(bboxes)
                mesh_res = fm.process(rgb)

                lmk2d_list = []
                lmk3d_list = None
                if mesh_res and mesh_res.multi_face_landmarks:
                    lmk2d_list = mesh_res.multi_face_landmarks
                    has_world = hasattr(mesh_res,"multi_face_world_landmarks") and (mesh_res.multi_face_world_landmarks is not None)
                    lmk3d_list = mesh_res.multi_face_world_landmarks if has_world else [None]*len(lmk2d_list)

                    centers = []
                    for lm2d in lmk2d_list:
                        nose = lm2d.landmark[1]
                        centers.append((nose.x*w, nose.y*h))

                    for t in tracks:
                        cx = (t.bbox[0]+t.bbox[2])/2.0
                        cy = (t.bbox[1]+t.bbox[3])/2.0
                        if not centers: continue
                        idx = int(np.argmin([(cx-x)**2 + (cy-y)**2 for (x,y) in centers]))
                        lm2d = lmk2d_list[idx].landmark
                        lm3d = None if (lmk3d_list is None or lmk3d_list[idx] is None) else lmk3d_list[idx].landmark

                        yaw, pitch, roll = estimate_head_pose_flexible(w, h, lm2d, lm3d)
                        yaw_p = None if yaw is None else (yaw - head_cal.bias_yaw)
                        pit_p = None if pitch is None else (pitch - head_cal.bias_pitch)
                        t.update_pose(yaw_p, pit_p, roll)

                        head_cal.feed(yaw, pitch)
                        head_cal.done_if_ready()

                        facing_now = (t.yaw_s is not None and t.pitch_s is not None and
                                      abs(t.yaw_s) <= FACING_YAW_DEG and abs(t.pitch_s) <= FACING_PITCH_DEG)
                        t.update_facing(facing_now)

                        allow_eval_gaze = (not gaze_adapt.active_calib) and facing_now and \
                                          (abs(t.yaw_s) <= MIN_FACING_FOR_GAZE and abs(t.pitch_s) <= MIN_FACING_FOR_GAZE)

                        g = gaze_adapt.refresh_on_bbox(frame, t.bbox)
                        if g is not None:
                            gx = None if g["gx"] is None else (g["gx"] - gaze_adapt.center_x)  
                            gy = None if g["gy"] is None else (g["gy"] - gaze_adapt.center_y) 
                            t.update_gaze(gx, gy, allow_eval_gaze)  

                        gaze_adapt.finish_calib_if_ready()  

                        if not gaze_adapt.active_calib:
                            nowt = now_ts()
                            head_offset = (abs(t.yaw_s or 0) > FACING_YAW_DEG) or (abs(t.pitch_s or 0) > FACING_PITCH_DEG)
                            head_quiet  = (t.head_vel or 0.0) < HEAD_STATIC_VEL_THR
                            head_off_ready = head_offset and head_quiet and (t.ts_not_focus_start is not None and (nowt - t.ts_not_focus_start) >= CHEAT_MIN_OFFPOSE_SEC)
                            t.mark_flag("HEAD_POSE_OFF", head_off_ready)
                            t.mark_flag("EYES_OFF",      t.ts_eyes_off_start is not None and (nowt - t.ts_eyes_off_start) >= CHEAT_MIN_EYESOFF_SEC)
                            t.mark_flag("EYES_MOVING",   t.ts_eyes_moving_start is not None and (nowt - t.ts_eyes_moving_start) >= CHEAT_MIN_EYESMOV_SEC)
                            t.mark_flag("OUT_OF_FRAME",  (nowt - t.ts_last) >= CHEAT_MIN_OUTOFFRAME_SEC)

                multi_faces_flag = (len(tracker.tracks) > 1) and (not gaze_adapt.active_calib)
                if multi_faces_flag:
                    oldest_seen = min(t.ts_last for t in tracker.tracks) if tracker.tracks else now_ts()
                    multi_faces_flag = (tnow - oldest_seen) >= CHEAT_MIN_MULTIFACE_SEC

                for t in tracker.tracks:
                    current_reason = None
                    if multi_faces_flag: current_reason = "MULTIPLE_FACES"
                    elif t.flags["OUT_OF_FRAME"]: current_reason = "OUT_OF_FRAME"
                    elif t.flags["EYES_OFF"]: current_reason = "EYES_OFF"
                    elif t.flags["EYES_MOVING"]: current_reason = "EYES_MOVING"
                    elif t.flags["HEAD_POSE_OFF"]: current_reason = "HEAD_POSE_OFF"
                    if gaze_adapt.active_calib: current_reason = None  
                    prev_reason = t.cheat_reason if t.cheat_active else None
                    if current_reason != prev_reason:
                        if current_reason is None and t.cheat_active:
                            t.clear_cheat()
                        elif current_reason is not None:
                            t.mark_cheat(current_reason)
                    seglog.update_track(t, current_reason)

            for t in tracker.tracks:
                x1,y1,x2,y2 = map(int, t.bbox)
                base = (0,200,0) if t.state=="FOCUS" else (0,0,200)
                color = (0,0,255) if t.cheat_active else base
                cv2.rectangle(draw, (x1,y1), (x2,y2), color, 2)
                label = f"ID {t.id} | {t.state}"

                if SHOW_DEBUG and t.gaze_speed is not None:
                    label += f" | v {t.gaze_speed:.2f}"

                if t.cheat_active and t.cheat_reason:
                    label += f" | CHEATING:{t.cheat_reason}"

                cv2.putText(draw, label, (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

            if len(tracker.tracks) > 1:
                cv2.putText(draw, f"MULTIPLE FACES: {len(tracker.tracks)}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            if SHOW_DEBUG:
                st = "CALIB" if gaze_adapt.active_calib else "RUN"
                cv2.putText(draw, f"{st} | thr yaw<=±{int(FACING_YAW_DEG)}, pitch<=±{int(FACING_PITCH_DEG)}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                
                cv2.putText(draw, f"Proc ~{TARGET_FPS} FPS | Tracks:{len(tracker.tracks)}", (10, h-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("Cheating Detection with GazeTracking", draw)
            if cv2.waitKey(1) & 0xFF == 27: break

    seglog.close_all()
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()