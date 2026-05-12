import cv2
import time
import numpy as np
import threading
import hashlib
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
import logging
import traceback
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("TruthLens")

# static_folder='.' يعني Flask هيسرف app.html من نفس الفولدر
app = Flask(__name__, static_folder='.')
CORS(app)

# ====================== DATA ======================
EXAMINERS = {
    "admin":  {"id": 1, "name": "Admin",      "password": hashlib.sha256("admin123".encode()).hexdigest(), "role": "admin"},
    "dr.m":   {"id": 2, "name": "Dr. Mohamed","password": hashlib.sha256("pass1234".encode()).hexdigest(), "role": "examiner"},
    "dr.sara":{"id": 3, "name": "Dr. Sara",   "password": hashlib.sha256("pass1234".encode()).hexdigest(), "role": "examiner"},
}

ROOMS = {
    1: {"id": 1, "code": "A201", "name": "Lecture Hall A - 2nd Floor",    "type": "lecture", "capacity": 200},
    2: {"id": 2, "code": "A202", "name": "Lecture Hall B - 2nd Floor",    "type": "lecture", "capacity": 150},
    3: {"id": 3, "code": "A203", "name": "Lecture Hall C - 2nd Floor",    "type": "lecture", "capacity": 120},
    4: {"id": 4, "code": "B101", "name": "Section Room 1 - 1st Floor",    "type": "section", "capacity": 35},
    5: {"id": 5, "code": "B102", "name": "Section Room 2 - 1st Floor",    "type": "section", "capacity": 35},
    6: {"id": 6, "code": "B103", "name": "Section Room 3 - 1st Floor",    "type": "section", "capacity": 40},
    7: {"id": 7, "code": "B104", "name": "Section Room 4 - 1st Floor",    "type": "section", "capacity": 30},
    8: {"id": 8, "code": "C301", "name": "Computer Lab 1 - 3rd Floor",    "type": "lab",     "capacity": 50},
    9: {"id": 9, "code": "C302", "name": "Computer Lab 2 - 3rd Floor",    "type": "lab",     "capacity": 50},
}

state = {
    "persons":          {},
    "rt_events":        deque(maxlen=500),
    "frame_jpeg":       None,
    "fps":              0.0,
    "running":          False,
    "any_hazard":       False,
    "num_detected":     0,
    "total_violations": 0,
    "session_start":    time.time(),
    "stats": {
        "bad_posture": 0, "copying": 0, "phone_use": 0,
        "phone_found": 0, "paper_switch": 0, "restless": 0
    },
}
lock = threading.Lock()

CFG = {
    "alpha":              0.80,
    "max_persons":        8,
    "track_dist":         160,
    "max_missing_time":   4.0,
    "move_threshold":     50,
    "move_window":        2.8,
    "posture_threshold":  165,
    "head_offset_thr":    75,
    "inf_w":              640,
    "inf_h":              480,
    "pose_conf":          0.38,
    "obj_conf":           0.32,
}

_next_track_num = 1
_next_event_id  = 1  # ← مهم: لازم يكون global هنا


# ====================== TRACKING ======================
def _new_track(pos):
    global _next_track_num
    tid = f"T{_next_track_num:03d}"
    _next_track_num += 1
    return tid, {
        "smooth_nose":     pos.copy(),
        "prev_pos":        pos.copy(),
        "move_times":      deque(maxlen=60),
        "last_seen":       time.time(),
        "violation_count": 0,
        "last_logged":     {},
    }


def match_track(pos, persons):
    best_id, best_d = None, CFG["track_dist"]
    for pid, p in persons.items():
        sp = p.get("smooth_nose")
        if sp is None:
            continue
        d = np.linalg.norm(pos - sp)
        if d < best_d:
            best_d, best_id = d, pid
    return best_id


def log_event(pid, violation):
    global _next_event_id
    ts = datetime.now().strftime("%H:%M:%S")
    ev = {
        "id":        _next_event_id,
        "timestamp": ts,
        "track_id":  pid,
        "violation": violation,
    }
    _next_event_id += 1
    with lock:
        state["rt_events"].appendleft(ev)
        state["total_violations"] += 1
        state["any_hazard"] = True
        key = violation.lower().replace(" ", "_")
        if key in state["stats"]:
            state["stats"][key] += 1


# ====================== VIDEO STREAM ======================
def generate_frames():
    """MJPEG stream generator — يشتغل طول ما الكاميرا شغالة."""
    while True:
        with lock:
            frame = state.get("frame_jpeg")
        if frame:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )
        time.sleep(0.033)  # ~30 fps ceiling للـ stream


# ====================== DETECTION THREAD ======================
def run_detection():
    """
    يشتغل في thread منفصل.
    بيستخدم picamera2 (Camera Module 2 على Pi 5).
    لو الكاميرا مش موجودة هيعمل fallback على cv2.VideoCapture.
    """
    global _next_track_num
    persons  = {}
    picam2   = None
    cap      = None

    try:
        log.info("Loading YOLO models …")
        pose_model = YOLO("yolov8n-pose.pt")
        obj_model  = YOLO("yolov8n.pt")
        log.info("Models loaded ✅")

        # ── محاولة picamera2 أولاً ──────────────────────────────────
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            video_cfg = picam2.create_video_configuration(
                main={"size": (1280, 720), "format": "RGB888"},
                controls={"FrameRate": 25}
            )
            picam2.configure(video_cfg)
            picam2.start()
            log.info("picamera2 started ✅")
            use_picam = True
        except Exception as cam_err:
            log.warning("picamera2 failed (%s) — falling back to cv2", cam_err)
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                log.error("No camera available. Aborting detection.")
                return
            use_picam = False

        with lock:
            state["running"]       = True
            state["session_start"] = time.time()

        prev_t = time.time()

        while state["running"]:

            # ── اقرأ الفريم ──────────────────────────────────────────
            if use_picam:
                array = picam2.capture_array()
                frame = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

            cur_t  = time.time()
            fps    = 1.0 / max(cur_t - prev_t, 1e-6)
            prev_t = cur_t

            h, w = frame.shape[:2]
            pf   = cv2.resize(frame, (CFG["inf_w"], CFG["inf_h"]))
            sx   = w / CFG["inf_w"]
            sy   = h / CFG["inf_h"]

            # ── YOLO inference ───────────────────────────────────────
            pose_res = pose_model(pf, verbose=False, conf=CFG["pose_conf"])[0]
            obj_res  = obj_model(pf,  verbose=False, conf=CFG["obj_conf"],
                                 classes=[67, 73, 84])[0]

            now         = time.time()
            matched_ids = set()

            # ── Pose Processing ──────────────────────────────────────
            if pose_res.keypoints is not None:
                for k in pose_res.keypoints.data.cpu().numpy():
                    nose      = k[0]
                    left_eye  = k[1]
                    left_ear  = k[3]
                    right_ear = k[4]
                    left_sh   = k[5]
                    right_sh  = k[6]
                    left_wr   = k[9]
                    right_wr  = k[10]

                    if nose[2] < 0.32:
                        continue

                    pos = np.array([nose[0] * sx, nose[1] * sy])
                    pid = match_track(pos, persons)

                    if pid is None:
                        if len(persons) < CFG["max_persons"]:
                            pid, data = _new_track(pos)
                            persons[pid] = data
                        else:
                            continue

                    s = persons[pid]
                    matched_ids.add(pid)
                    s["last_seen"] = now

                    # Smoothing
                    s["smooth_nose"] = (
                        CFG["alpha"] * pos
                        + (1 - CFG["alpha"]) * s["smooth_nose"]
                    )
                    delta          = np.linalg.norm(s["smooth_nose"] - s["prev_pos"])
                    s["prev_pos"]  = s["smooth_nose"].copy()

                    if delta > CFG["move_threshold"]:
                        s["move_times"].append(now)

                    hazards = []

                    # — Bad Posture —
                    sh_cy = s["smooth_nose"][1] + 90
                    if left_sh[2] > 0.3 and right_sh[2] > 0.3:
                        sh_cy = ((left_sh[1] + right_sh[1]) / 2) * sy
                    if abs(s["smooth_nose"][1] - sh_cy) > CFG["posture_threshold"]:
                        hazards.append("Bad Posture")

                    # — Copying —
                    head_side = (left_ear[2] < 0.3 or right_ear[2] < 0.3)
                    sh_cx     = (
                        ((left_sh[0] + right_sh[0]) / 2) * sx
                        if left_sh[2] > 0.3 and right_sh[2] > 0.3
                        else s["smooth_nose"][0]
                    )
                    if head_side and abs(s["smooth_nose"][0] - sh_cx) > CFG["head_offset_thr"]:
                        hazards.append("Copying")

                    # — Phone Use (head-tilt heuristic) —
                    if left_eye[2] > 0.32 and nose[1] * sy > left_eye[1] * sy + 65:
                        hazards.append("Phone Use")

                    # — Phone Found (object detection) —
                    if any(int(b.cls) == 67 for b in obj_res.boxes):
                        hazards.append("Phone Found")

                    # — Paper Switch —
                    papers = [b for b in obj_res.boxes if int(b.cls) in [73, 84]]
                    if len(papers) > 1 and (left_wr[2] > 0.3 or right_wr[2] > 0.3):
                        hazards.append("Paper Switch")

                    # — Restless —
                    recent = sum(1 for t in s["move_times"] if now - t < CFG["move_window"])
                    if recent > 3:
                        hazards.append("Restless")

                    # ── Draw on frame ────────────────────────────────
                    x = int(s["smooth_nose"][0])
                    y = int(s["smooth_nose"][1])

                    if hazards:
                        color = (0, 0, 255)       # Red
                    elif delta > CFG["move_threshold"]:
                        color = (0, 165, 255)     # Orange
                    else:
                        color = (0, 255, 0)       # Green

                    cv2.rectangle(frame, (x - 60, y - 90), (x + 60, y + 90), color, 2)
                    cv2.putText(frame, pid, (x - 35, y - 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    if hazards:
                        cv2.putText(frame, hazards[0], (x - 55, y + 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                    # ── Log violations (cooldown 3s per type) ────────
                    for h in hazards:
                        if now - s["last_logged"].get(h, 0) > 3:
                            log_event(pid, h)
                            s["last_logged"][h] = now
                            s["violation_count"] += 1

            # ── Cleanup lost tracks ──────────────────────────────────
            for pid in list(persons.keys()):
                if (pid not in matched_ids
                        and now - persons[pid]["last_seen"] > CFG["max_missing_time"] * 2):
                    del persons[pid]

            # ── Encode & share frame ─────────────────────────────────
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            with lock:
                state["frame_jpeg"]   = jpeg.tobytes()
                state["persons"]      = persons
                state["num_detected"] = len(persons)
                state["fps"]          = round(fps, 1)

    except Exception:
        log.error("Detection thread crashed:\n%s", traceback.format_exc())
    finally:
        log.info("Detection thread stopping …")
        if picam2:
            picam2.stop()
            picam2.close()
        if cap:
            cap.release()
        with lock:
            state["running"] = False


# ====================== FLASK ROUTES ======================

@app.route('/')
def index():
    """سرف الـ frontend."""
    return send_from_directory('.', 'app.html')


@app.route('/api/rooms')
def get_rooms():
    return jsonify(list(ROOMS.values()))


@app.route('/api/start', methods=['POST'])
def start_detection():
    if not state["running"]:
        t = threading.Thread(target=run_detection, daemon=True)
        t.start()
    return jsonify({"status": "started"})


@app.route('/api/stop', methods=['POST'])
def stop_detection():
    with lock:
        state["running"] = False
    return jsonify({"status": "stopped"})


@app.route('/api/status')
def get_status():
    with lock:
        # deque مش JSON-serializable → نحوّله لـ list
        data = dict(state)
        data["rt_events"] = list(state["rt_events"])
        data["persons"]   = {}   # لا نبعت objects كبيرة للـ frontend
    return jsonify(data)


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ====================== ENTRY POINT ======================
if __name__ == '__main__':
    print("=" * 50)
    print("  TruthLens Pro — Raspberry Pi 5")
    print("  http://0.0.0.0:5050")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)
