import cv2
import time
import numpy as np
import threading
import hashlib
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("TruthLens")

app = Flask(__name__)
CORS(app)

# ====================== DATA ======================
EXAMINERS = {
    "admin": {"id": 1, "name": "Admin", "password": hashlib.sha256("admin123".encode()).hexdigest(), "role": "admin"},
    "dr.m": {"id": 2, "name": "Dr. Mohamed", "password": hashlib.sha256("pass1234".encode()).hexdigest(), "role": "examiner"},
    "dr.sara": {"id": 3, "name": "Dr. Sara", "password": hashlib.sha256("pass1234".encode()).hexdigest(), "role": "examiner"},
}

ROOMS = {
    1: {"id": 1, "code": "A201", "name": "Lecture Hall A - 2nd Floor", "type": "lecture", "capacity": 200},
    2: {"id": 2, "code": "A202", "name": "Lecture Hall B - 2nd Floor", "type": "lecture", "capacity": 150},
    3: {"id": 3, "code": "A203", "name": "Lecture Hall C - 2nd Floor", "type": "lecture", "capacity": 120},
    4: {"id": 4, "code": "B101", "name": "Section Room 1 - 1st Floor", "type": "section", "capacity": 35},
    5: {"id": 5, "code": "B102", "name": "Section Room 2 - 1st Floor", "type": "section", "capacity": 35},
    6: {"id": 6, "code": "B103", "name": "Section Room 3 - 1st Floor", "type": "section", "capacity": 40},
    7: {"id": 7, "code": "B104", "name": "Section Room 4 - 1st Floor", "type": "section", "capacity": 30},
    8: {"id": 8, "code": "C301", "name": "Computer Lab 1 - 3rd Floor", "type": "lab", "capacity": 50},
    9: {"id": 9, "code": "C302", "name": "Computer Lab 2 - 3rd Floor", "type": "lab", "capacity": 50},
}

SESSIONS_HISTORY = []
EVENTS_LOG = []
_next_session_id = 1
_next_event_id = 1

state = {
    "persons": {},
    "rt_events": deque(maxlen=500),
    "frame_jpeg": None,
    "fps": 0.0,
    "running": False,
    "any_hazard": False,
    "num_detected": 0,
    "total_violations": 0,
    "session_start": time.time(),
    "session_id": None,
    "stats": {"bad_posture": 0, "copying": 0, "phone_use": 0, "phone_found": 0, "paper_switch": 0, "restless": 0},
}
lock = threading.Lock()

CFG = {
    "alpha": 0.80, "max_persons": 8, "track_dist": 160, "max_missing_time": 4.0,
    "move_threshold": 50, "move_window": 2.8, "posture_threshold": 165,
    "posture_time": 3.2, "copying_time": 4.2, "phone_time": 3.8,
    "head_offset_thr": 75, "camera_id": 0, "inf_w": 640, "inf_h": 480,
    "pose_conf": 0.38, "obj_conf": 0.32,
}

_next_track_num = 1


# ====================== TRACKING ======================
def _new_track(pos):
    global _next_track_num
    tid = f"T{_next_track_num:03d}"
    _next_track_num += 1
    return tid, {
        "smooth_nose": pos.copy(), "prev_pos": pos.copy(),
        "move_times": deque(maxlen=60), "timers": {"POSTURE":0, "SIDE":0, "PHONE":0},
        "last_seen": time.time(), "violation_count": 0, "last_logged": {}, "active": True
    }


def match_track(pos, persons):
    best_id, best_d = None, CFG["track_dist"]
    for pid, p in persons.items():
        sp = p.get("smooth_nose")
        if sp is None: continue
        d = np.linalg.norm(pos - sp)
        if d < best_d:
            best_d, best_id = d, pid
    return best_id


def log_event(pid, violation):
    global _next_event_id
    ts = datetime.now().strftime("%H:%M:%S")
    ev = {"id": _next_event_id, "timestamp": ts, "track_id": pid, "violation": violation, "severity": "warning"}
    _next_event_id += 1

    with lock:
        state["any_hazard"] = True
        state["rt_events"].appendleft(ev)
        state["total_violations"] += 1
        key = violation.lower().replace(" ", "_")
        if key in state["stats"]:
            state["stats"][key] += 1


# ====================== VIDEO STREAM ======================
def generate_frames():
    while True:
        with lock:
            if state.get("frame_jpeg"):
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + state["frame_jpeg"] + b'\r\n')
        time.sleep(0.04)


# ====================== MAIN DETECTION WITH DRAWING ======================
def run_detection():
    global _next_track_num
    cap = None
    persons = {}

    try:
        log.info("Loading models...")
        pose_model = YOLO("yolov8n-pose.pt")
        obj_model = YOLO("yolov8n.pt")

        cap = cv2.VideoCapture(CFG["camera_id"])
        if not cap.isOpened():
            log.error("Cannot open camera")
            return

        with lock:
            state["running"] = True
            state["session_start"] = time.time()

        prev_t = time.time()

        while state["running"]:
            ret, frame = cap.read()
            if not ret: continue

            cur_t = time.time()
            fps = 1.0 / max(cur_t - prev_t, 1e-6)
            prev_t = cur_t

            h, w = frame.shape[:2]
            pf = cv2.resize(frame, (CFG["inf_w"], CFG["inf_h"]))
            sx, sy = w / CFG["inf_w"], h / CFG["inf_h"]

            pose_res = pose_model(pf, verbose=False, conf=CFG["pose_conf"])[0]
            obj_res = obj_model(pf, verbose=False, conf=CFG["obj_conf"], classes=[67, 73, 84])[0]

            now = time.time()
            any_hazard = False
            matched_ids = set()

            if pose_res.keypoints is not None:
                for k in pose_res.keypoints.data.cpu().numpy():
                    nose = k[0]
                    left_eye = k[1]
                    left_ear, right_ear = k[3], k[4]
                    left_sh, right_sh = k[5], k[6]
                    left_wr, right_wr = k[9], k[10]

                    if nose[2] < 0.32: continue

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

                    s["smooth_nose"] = CFG["alpha"] * pos + (1 - CFG["alpha"]) * s["smooth_nose"]
                    delta = np.linalg.norm(s["smooth_nose"] - s["prev_pos"])
                    s["prev_pos"] = s["smooth_nose"].copy()

                    if delta > CFG["move_threshold"]:
                        s["move_times"].append(now)

                    hazards = []

                    # Violation Detection
                    sh_cy = s["smooth_nose"][1] + 90
                    if left_sh[2] > 0.3 and right_sh[2] > 0.3:
                        sh_cy = ((left_sh[1] + right_sh[1]) / 2) * sy
                    if abs(s["smooth_nose"][1] - sh_cy) > CFG["posture_threshold"]:
                        hazards.append("Bad Posture")

                    head_side = (left_ear[2] < 0.3 or right_ear[2] < 0.3)
                    sh_cx = (((left_sh[0] + right_sh[0]) / 2) * sx if left_sh[2] > 0.3 and right_sh[2] > 0.3 else s["smooth_nose"][0])
                    if head_side and abs(s["smooth_nose"][0] - sh_cx) > CFG["head_offset_thr"]:
                        hazards.append("Copying")

                    if left_eye[2] > 0.32 and nose[1] * sy > left_eye[1] * sy + 65:
                        hazards.append("Phone Use")

                    if any(int(b.cls) == 67 for b in obj_res.boxes):
                        hazards.append("Phone Found")

                    papers = [b for b in obj_res.boxes if int(b.cls) in [73, 84]]
                    if len(papers) > 1 and (left_wr[2] > 0.3 or right_wr[2] > 0.3):
                        hazards.append("Paper Switch")

                    recent = sum(1 for t in s["move_times"] if now - t < CFG["move_window"])
                    if recent > 3:
                        hazards.append("Restless")

                    # ==================== DRAW ON FRAME ====================
                    x = int(s["smooth_nose"][0])
                    y = int(s["smooth_nose"][1])

                    if hazards:
                        color = (0, 0, 255)      # Red
                    elif delta > CFG["move_threshold"]:
                        color = (0, 165, 255)    # Orange
                    else:
                        color = (0, 255, 0)      # Green

                    # Draw Box
                    cv2.rectangle(frame, (x-60, y-90), (x+60, y+90), color, 2)

                    # Draw ID
                    cv2.putText(frame, pid, (x-35, y-100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Draw Violation
                    if hazards:
                        cv2.putText(frame, hazards[0], (x-55, y+120), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                    # Log violations
                    for h in hazards:
                        if now - s["last_logged"].get(h, 0) > 3:
                            log_event(pid, h)
                            s["last_logged"][h] = now
                            s["violation_count"] += 1

            # Cleanup
            for pid in list(persons.keys()):
                if pid not in matched_ids and now - persons[pid]["last_seen"] > CFG["max_missing_time"] * 2:
                    del persons[pid]

            # Encode frame
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])

            with lock:
                state["frame_jpeg"] = jpeg.tobytes()
                state["persons"] = persons
                state["num_detected"] = len(persons)
                state["any_hazard"] = any_hazard
                state["fps"] = round(fps, 1)

    except Exception as e:
        log.error("Error: %s", traceback.format_exc())
    finally:
        if cap: cap.release()


# ====================== ROUTES ======================
@app.route('/api/start', methods=['POST'])
def start_detection():
    if not state["running"]:
        threading.Thread(target=run_detection, daemon=True).start()
    return jsonify({"ok": True})

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    with lock: 
        state["running"] = False
    return jsonify({"ok": True})

@app.route('/api/status')
def get_status():
    with lock:
        return jsonify(state)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/rooms')
def get_rooms():
    return jsonify(list(ROOMS.values()))


if __name__ == '__main__':
    print("🚀 TruthLens Pro v3 Running → http://localhost:5050")
    app.run(host='0.0.0.0', port=5050, debug=False)

# """
# TruthLens Pro v3 — detector.py
# In-memory data structures, NO DATABASE
# Real-time exam surveillance with live alerts
# """

# import cv2
# import time
# import numpy as np
# import threading
# import hashlib
# from collections import deque
# from datetime import datetime
# from ultralytics import YOLO
# from flask import Flask, Response, jsonify, request
# from flask_cors import CORS
# import logging
# import traceback

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# log = logging.getLogger("TruthLens")

# app = Flask(__name__)
# CORS(app)

# # ====================== DATA ======================
# EXAMINERS = {
#     "admin": {"id": 1, "name": "Admin", "password": hashlib.sha256("admin123".encode()).hexdigest(), "role": "admin"},
#     "dr.m": {"id": 2, "name": "Dr. Mohamed", "password": hashlib.sha256("pass1234".encode()).hexdigest(), "role": "examiner"},
#     "dr.sara": {"id": 3, "name": "Dr. Sara", "password": hashlib.sha256("pass1234".encode()).hexdigest(), "role": "examiner"},
# }

# ROOMS = {
#     1: {"id": 1, "code": "A201", "name": "Lecture Hall A - 2nd Floor", "type": "lecture", "capacity": 200},
#     2: {"id": 2, "code": "A202", "name": "Lecture Hall B - 2nd Floor", "type": "lecture", "capacity": 150},
#     3: {"id": 3, "code": "A203", "name": "Lecture Hall C - 2nd Floor", "type": "lecture", "capacity": 120},
#     4: {"id": 4, "code": "B101", "name": "Section Room 1 - 1st Floor", "type": "section", "capacity": 35},
#     5: {"id": 5, "code": "B102", "name": "Section Room 2 - 1st Floor", "type": "section", "capacity": 35},
#     6: {"id": 6, "code": "B103", "name": "Section Room 3 - 1st Floor", "type": "section", "capacity": 40},
#     7: {"id": 7, "code": "B104", "name": "Section Room 4 - 1st Floor", "type": "section", "capacity": 30},
#     8: {"id": 8, "code": "C301", "name": "Computer Lab 1 - 3rd Floor", "type": "lab", "capacity": 50},
#     9: {"id": 9, "code": "C302", "name": "Computer Lab 2 - 3rd Floor", "type": "lab", "capacity": 50},
# }

# SESSIONS_HISTORY = []
# EVENTS_LOG = []
# _next_session_id = 1
# _next_event_id = 1

# state = {
#     "persons": {},
#     "rt_events": deque(maxlen=500),
#     "frame_jpeg": None,
#     "fps": 0.0,
#     "running": False,
#     "any_hazard": False,
#     "num_detected": 0,
#     "total_violations": 0,
#     "session_start": time.time(),
#     "session_id": None,
#     "stats": {"bad_posture": 0, "copying": 0, "phone_use": 0, "phone_found": 0, "paper_switch": 0, "restless": 0},
# }
# lock = threading.Lock()

# CFG = {
#     "alpha": 0.80, "max_persons": 8, "track_dist": 160, "max_missing_time": 4.0,
#     "move_threshold": 50, "move_window": 2.8, "posture_threshold": 165,
#     "posture_time": 3.2, "copying_time": 4.2, "phone_time": 3.8,
#     "head_offset_thr": 75, "camera_id": 0, "inf_w": 640, "inf_h": 480,
#     "pose_conf": 0.38, "obj_conf": 0.32,
# }

# _next_track_num = 1


# # ====================== TRACKING ======================
# def _new_track(pos):
#     global _next_track_num
#     tid = f"T{_next_track_num:03d}"
#     _next_track_num += 1
#     return tid, {
#         "smooth_nose": pos.copy(), "prev_pos": pos.copy(),
#         "move_times": deque(maxlen=60), "timers": {"POSTURE":0, "SIDE":0, "PHONE":0},
#         "last_seen": time.time(), "violation_count": 0, "last_logged": {}, "active": True
#     }


# def match_track(pos, persons):
#     best_id, best_d = None, CFG["track_dist"]
#     for pid, p in persons.items():
#         sp = p.get("smooth_nose")
#         if sp is None: continue
#         d = np.linalg.norm(pos - sp)
#         if d < best_d:
#             best_d, best_id = d, pid
#     return best_id


# def log_event(pid, violation):
#     global _next_event_id
#     ts = datetime.now().strftime("%H:%M:%S")
#     ev = {"id": _next_event_id, "timestamp": ts, "track_id": pid, "violation": violation, "severity": "warning"}
#     _next_event_id += 1

#     with lock:
#         state["any_hazard"] = True
#         state["rt_events"].appendleft(ev)
#         state["total_violations"] += 1
#         key = violation.lower().replace(" ", "_")
#         if key in state["stats"]:
#             state["stats"][key] += 1


# # ====================== VIDEO STREAM ======================
# def generate_frames():
#     while True:
#         with lock:
#             if state.get("frame_jpeg"):
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + state["frame_jpeg"] + b'\r\n')
#         time.sleep(0.04)


# # ====================== DETECTION ======================
# def run_detection():
#     global _next_track_num
#     cap = None
#     persons = {}

#     try:
#         log.info("Loading models...")
#         pose_model = YOLO("yolov8n-pose.pt")
#         obj_model = YOLO("yolov8n.pt")

#         cap = cv2.VideoCapture(CFG["camera_id"])
#         if not cap.isOpened():
#             log.error("Cannot open camera")
#             return

#         with lock:
#             state["running"] = True
#             state["session_start"] = time.time()

#         prev_t = time.time()

#         while state["running"]:
#             ret, frame = cap.read()
#             if not ret: continue

#             cur_t = time.time()
#             fps = 1.0 / max(cur_t - prev_t, 1e-6)
#             prev_t = cur_t

#             h, w = frame.shape[:2]
#             pf = cv2.resize(frame, (CFG["inf_w"], CFG["inf_h"]))
#             sx, sy = w / CFG["inf_w"], h / CFG["inf_h"]

#             pose_res = pose_model(pf, verbose=False, conf=CFG["pose_conf"])[0]
#             obj_res = obj_model(pf, verbose=False, conf=CFG["obj_conf"], classes=[67, 73, 84])[0]

#             now = time.time()
#             any_hazard = False
#             matched_ids = set()

#             if pose_res.keypoints is not None:
#                 for k in pose_res.keypoints.data.cpu().numpy():
#                     nose = k[0]; left_eye = k[1]
#                     left_ear, right_ear = k[3], k[4]
#                     left_sh, right_sh = k[5], k[6]
#                     left_wr, right_wr = k[9], k[10]

#                     if nose[2] < 0.32: continue

#                     pos = np.array([nose[0]*sx, nose[1]*sy])
#                     pid = match_track(pos, persons)

#                     if pid is None:
#                         if len(persons) < CFG["max_persons"]:
#                             pid, data = _new_track(pos)
#                             persons[pid] = data
#                         else:
#                             continue

#                     s = persons[pid]
#                     matched_ids.add(pid)
#                     s["last_seen"] = now

#                     s["smooth_nose"] = CFG["alpha"] * pos + (1 - CFG["alpha"]) * s["smooth_nose"]
#                     delta = np.linalg.norm(s["smooth_nose"] - s["prev_pos"])
#                     s["prev_pos"] = s["smooth_nose"].copy()

#                     if delta > CFG["move_threshold"]:
#                         s["move_times"].append(now)

#                     hazards = []

#                     # Bad Posture
#                     sh_cy = s["smooth_nose"][1] + 90
#                     if left_sh[2] > 0.3 and right_sh[2] > 0.3:
#                         sh_cy = ((left_sh[1] + right_sh[1])/2) * sy
#                     if abs(s["smooth_nose"][1] - sh_cy) > CFG["posture_threshold"]:
#                         hazards.append("Bad Posture")

#                     # Copying
#                     head_side = (left_ear[2] < 0.3 or right_ear[2] < 0.3)
#                     sh_cx = (((left_sh[0]+right_sh[0])/2)*sx if left_sh[2]>0.3 and right_sh[2]>0.3 else s["smooth_nose"][0])
#                     if head_side and abs(s["smooth_nose"][0] - sh_cx) > CFG["head_offset_thr"]:
#                         hazards.append("Copying")

#                     # Phone Use
#                     if left_eye[2] > 0.32 and nose[1]*sy > left_eye[1]*sy + 65:
#                         hazards.append("Phone Use")

#                     # Objects
#                     if any(int(b.cls)==67 for b in obj_res.boxes):
#                         hazards.append("Phone Found")
#                     papers = [b for b in obj_res.boxes if int(b.cls) in [73,84]]
#                     if len(papers)>1 and (left_wr[2]>0.3 or right_wr[2]>0.3):
#                         hazards.append("Paper Switch")

#                     # Restless
#                     recent = sum(1 for t in s["move_times"] if now-t < CFG["move_window"])
#                     if recent > 3:
#                         hazards.append("Restless")

#                     # Log
#                     for h in hazards:
#                         if now - s["last_logged"].get(h, 0) > 3:
#                             log_event(pid, h)
#                             s["last_logged"][h] = now
#                             s["violation_count"] += 1
#                             any_hazard = True

#             # Cleanup + Update frame
#             for pid in list(persons):
#                 if pid not in matched_ids and now - persons[pid]["last_seen"] > CFG["max_missing_time"]*2:
#                     del persons[pid]

#             _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

#             with lock:
#                 state["frame_jpeg"] = jpeg.tobytes()
#                 state["persons"] = persons
#                 state["num_detected"] = len(persons)
#                 state["any_hazard"] = any_hazard
#                 state["fps"] = round(fps, 1)

#     except Exception as e:
#         log.error("Error: %s", traceback.format_exc())
#     finally:
#         if cap: cap.release()


# # ====================== ROUTES ======================
# @app.route('/api/start', methods=['POST'])
# def start_detection():
#     if not state["running"]:
#         threading.Thread(target=run_detection, daemon=True).start()
#     return jsonify({"ok": True})

# @app.route('/api/stop', methods=['POST'])
# def stop_detection():
#     with lock: state["running"] = False
#     return jsonify({"ok": True})

# @app.route('/api/status')
# def get_status():
#     with lock:
#         return jsonify(state)

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/api/rooms')
# def get_rooms():
#     return jsonify(list(ROOMS.values()))


# if __name__ == '__main__':
#     print("🚀 TruthLens Pro v3 running at http://localhost:5050")
#     app.run(host='0.0.0.0', port=5050, debug=False)