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

app = Flask(__name__, static_folder='.')
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/video_feed": {"origins": "*"}})

# ====================== DATA ======================
EXAMINERS = {
    "admin":  {"id": 1, "name": "Admin",      "password": hashlib.sha256("admin123".encode()).hexdigest(), "role": "admin"},
    "dr.m":   {"id": 2, "name": "Dr. Mohamed","password": hashlib.sha256("pass1234".encode()).hexdigest(), "role": "examiner"},
    "dr.sara":{"id": 3, "name": "Dr. Sara",   "password": hashlib.sha256("pass1234".encode()).hexdigest(), "role": "examiner"},
}

ROOMS = {
    1: {"id": 1, "code": "A201", "name": "Lecture Hall A - 3rd Floor",    "type": "lecture", "capacity": 200},
    2: {"id": 2, "code": "A202", "name": "Lecture Hall B - 3rd Floor",    "type": "lecture", "capacity": 150},
    3: {"id": 3, "code": "A203", "name": "Lecture Hall C - 3rd Floor",    "type": "lecture", "capacity": 120},
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
    "alpha":             0.30,   # smoothing أقل = استجابة أسرع
    "max_persons":       8,
    "track_dist":        120,
    "iou_weight":        0.5,    # وزن الـ IoU في الـ matching
    "max_missing_time":  4.0,
    "move_threshold":    28,
    "move_window":       2.8,
    "posture_threshold": 145,
    "head_offset_thr":   70,
    "inf_w":             640,
    "inf_h":             480,
    "pose_conf":         0.38,
    "obj_conf":          0.32,
}

_next_track_num = 1
_next_event_id  = 1


# ====================== HELPERS ======================

def compute_iou(boxA, boxB):
    """IoU بين اتنين مستطيلات (x1,y1,x2,y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def keypoints_to_box(k, sx, sy):
    """
    Adaptive bounding box — حجم الـ box بيتغير حسب المسافة.
    بيستخدم المسافة الحقيقية بين الكتفين كمقياس للبُعد.
    """
    target_kps = [0, 1, 2, 3, 4, 5, 6]
    pts = []
    for i in target_kps:
        if k[i][2] > 0.25:
            pts.append((k[i][0] * sx, k[i][1] * sy))

    if len(pts) < 2:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    w_span = max(xs) - min(xs)

    # المسافة بين الكتفين كمقياس للبُعد — fallback لـ w_span
    shoulder_dist = w_span
    if k[5][2] > 0.3 and k[6][2] > 0.3:
        shoulder_dist = abs(k[5][0] - k[6][0]) * sx

    scale = max(shoulder_dist, 20)

    w_pad        = scale * 0.55
    h_pad_top    = scale * 1.1
    h_pad_bottom = scale * 1.8

    x1 = int(min(xs) - w_pad)
    y1 = int(min(ys) - h_pad_top)
    x2 = int(max(xs) + w_pad)
    y2 = int(max(ys) + h_pad_bottom)

    # clamp — منع الـ box من الاتوسع أو الاتقزم بشكل غير منطقي
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < 40 or box_h < 40:
        return None
    if box_w > 600:
        cx = (x1 + x2) // 2
        x1 = cx - 300
        x2 = cx + 300
    if box_h > 700:
        cy = (y1 + y2) // 2
        y1 = cy - 350
        y2 = cy + 350

    return (x1, y1, x2, y2)


# ====================== TRACKING ======================

def _new_track(pos, box):
    global _next_track_num
    tid = f"T{_next_track_num:03d}"
    _next_track_num += 1
    return tid, {
        "smooth_nose":     pos.copy(),
        "prev_pos":        pos.copy(),
        "last_box":        box,
        "move_times":      deque(maxlen=60),
        "last_seen":       time.time(),
        "violation_count": 0,
        "last_logged":     {},
    }


def match_track(pos, box, persons):
    """
    Matching مدمج:
    - distance من smooth_nose (normalized)
    - IoU مع last_box (للتأكد إن نفس الشخص)
    الـ combined score الأقل = أفضل match.
    """
    best_id    = None
    best_score = float('inf')

    for pid, p in persons.items():
        sp = p.get("smooth_nose")
        if sp is None:
            continue

        dist = np.linalg.norm(pos - sp)
        if dist > CFG["track_dist"] * 2.5:
            continue

        dist_score = dist / CFG["track_dist"]

        iou_score = 1.0  # worst case لو مفيش box
        prev_box  = p.get("last_box")
        if prev_box is not None and box is not None:
            iou       = compute_iou(prev_box, box)
            iou_score = 1.0 - iou

        score = (1 - CFG["iou_weight"]) * dist_score + CFG["iou_weight"] * iou_score

        if score < best_score:
            best_score = score
            best_id    = pid

    if best_score > 1.8:  # threshold — لو بعيد جداً = شخص جديد
        return None
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
    while True:
        with lock:
            frame = state.get("frame_jpeg")
        if frame:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )
        time.sleep(0.033)


# ====================== DETECTION THREAD ======================

def run_detection():
    global _next_track_num
    persons = {}
    picam2  = None
    cap     = None

    try:
        log.info("Loading YOLO models …")
        pose_model = YOLO("yolov8n-pose.pt")
        obj_model  = YOLO("yolov8n.pt")
        log.info("Models loaded ✅")

        # ── Camera ───────────────────────────────────────────────────
        use_picam = False
        try:
            from picamera2 import Picamera2
            picam2    = Picamera2()
            video_cfg = picam2.create_video_configuration(
                main={"size": (1280, 720), "format": "RGB888"},
                controls={"FrameRate": 25}
            )
            picam2.configure(video_cfg)
            picam2.start()
            log.info("picamera2 started ✅")
            use_picam = True
            try:
                _test = picam2.capture_array()
                log.info("Camera test capture OK — shape: %s", _test.shape)
            except Exception as test_err:
                log.error("Camera test capture FAILED: %s", test_err)
                log.warning("Falling back to cv2.VideoCapture")
                picam2.stop()
                picam2.close()
                picam2    = None
                use_picam = False
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    log.error("No camera available at all. Aborting.")
                    return
        except Exception as cam_err:
            log.warning("picamera2 failed (%s) — falling back to cv2", cam_err)
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                log.error("No camera available. Aborting.")
                return

        with lock:
            state["running"]       = True
            state["session_start"] = time.time()

        prev_t = time.time()

        while state["running"]:

            # ── Read frame ───────────────────────────────────────────
            if use_picam:
                array = picam2.capture_array()
                channels = array.shape[2] if array.ndim == 3 else 0

                if channels == 4:
                    # Pi 5 picamera2 default: XBGR — drop X channel, keep BGR
                    frame = array[:, :, :3].copy()          # slice off alpha/X → BGR directly
                elif channels == 3:
                    # Some configs output RGB — convert to BGR for OpenCV
                    b = array[:, :, 2].copy()
                    g = array[:, :, 1].copy()
                    r = array[:, :, 0].copy()
                    frame = cv2.merge([b, g, r])
                else:
                    frame = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

            cur_t  = time.time()
            fps    = 1.0 / max(cur_t - prev_t, 1e-6)
            prev_t = cur_t
            with lock:
                state["fps"] = round(fps, 1)

            h, w = frame.shape[:2]
            pf   = cv2.resize(frame, (CFG["inf_w"], CFG["inf_h"]))
            sx   = w / CFG["inf_w"]
            sy   = h / CFG["inf_h"]

            try:
                # ── YOLO ─────────────────────────────────────────────────
                pose_res = pose_model(pf, verbose=False, conf=CFG["pose_conf"])[0]
                obj_res  = obj_model(pf,  verbose=False, conf=CFG["obj_conf"],
                                     classes=[67, 73, 84])[0]

                now              = time.time()
                matched_ids      = set()
                frame_has_hazard = False

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
                        box = keypoints_to_box(k, sx, sy)   # ← real box

                        pid = match_track(pos, box, persons)

                        if pid is None:
                            if len(persons) < CFG["max_persons"]:
                                pid, data = _new_track(pos, box)
                                persons[pid] = data
                            else:
                                continue

                        s = persons[pid]
                        matched_ids.add(pid)
                        s["last_seen"] = now

                        # Smooth nose position
                        s["smooth_nose"] = (
                            CFG["alpha"] * pos
                            + (1 - CFG["alpha"]) * s["smooth_nose"]
                        )

                        # Update box
                        if box is not None:
                            s["last_box"] = box

                        delta         = np.linalg.norm(s["smooth_nose"] - s["prev_pos"])
                        s["prev_pos"] = s["smooth_nose"].copy()

                        if delta > CFG["move_threshold"]:
                            s["move_times"].append(now)

                        hazards = []

                        # — Bad Posture —
                        sh_cy = s["smooth_nose"][1] + 90
                        if left_sh[2] > 0.3 and right_sh[2] > 0.3:
                            sh_cy = ((left_sh[1] + right_sh[1]) / 2) * sy
                        if abs(s["smooth_nose"][1] - sh_cy) > CFG["posture_threshold"]:
                            hazards.append("Bad Posture")

                        # Forward lean — الرأس أمام الكتفين بشكل واضح
                        if left_sh[2] > 0.3 and right_sh[2] > 0.3:
                            sh_mid_y   = ((left_sh[1] + right_sh[1]) / 2) * sy
                            nose_y_real = nose[1] * sy
                            if nose_y_real > sh_mid_y + 30:
                                if "Bad Posture" not in hazards:
                                    hazards.append("Bad Posture")

                        # — Copying —
                        head_side = (left_ear[2] < 0.3 or right_ear[2] < 0.3)
                        sh_cx = (
                            ((left_sh[0] + right_sh[0]) / 2) * sx
                            if left_sh[2] > 0.3 and right_sh[2] > 0.3
                            else s["smooth_nose"][0]
                        )
                        if head_side and abs(s["smooth_nose"][0] - sh_cx) > CFG["head_offset_thr"]:
                            hazards.append("Copying")

                        # Phone Use — يتكتشف بس لو:
                        # 1. الإيدين قريبة من الوجه (الرسغ فوق مستوى الذقن)
                        # 2. الرأس بينزل على الشاشة (nose أعمق من الكتفين بكتير)
                        wrist_near_face = False
                        chin_y = nose[1] * sy + 60   # تقريب لمستوى الذقن

                        if left_wr[2] > 0.3 and left_wr[1] * sy < chin_y:
                            wrist_near_face = True
                        if right_wr[2] > 0.3 and right_wr[1] * sy < chin_y:
                            wrist_near_face = True

                        head_down = False
                        if left_sh[2] > 0.3 and right_sh[2] > 0.3:
                            sh_mid_y = ((left_sh[1] + right_sh[1]) / 2) * sy
                            if nose[1] * sy > sh_mid_y + 50:
                                head_down = True

                        if wrist_near_face and head_down:
                            hazards.append("Phone Use")

                        # — Phone Found —
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

                        if hazards:
                            frame_has_hazard = True

                        # ── Draw ─────────────────────────────────────────
                        if box is not None:
                            x1, y1, x2, y2 = box
                            x1 = max(0, int(x1))
                            y1 = max(0, int(y1))
                            x2 = min(w - 1, int(x2))
                            y2 = min(h - 1, int(y2))
                            if x2 - x1 < 10 or y2 - y1 < 10:
                                continue   # skip drawing if box collapsed after clamping

                            if hazards:
                                color = (0, 0, 255)
                            elif delta > CFG["move_threshold"]:
                                color = (0, 165, 255)
                            else:
                                color = (0, 200, 0)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                            # ID label فوق المستطيل
                            label_y = max(y1 - 14, 20)
                            (lw, lh), _ = cv2.getTextSize(pid, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                            cv2.rectangle(frame,
                                          (x1, label_y - lh - 4),
                                          (x1 + lw + 6, label_y + 4),
                                          (0, 0, 0), -1)
                            cv2.putText(frame, pid, (x1 + 3, label_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
                                        cv2.LINE_AA)

                            # Violation label تحت المستطيل مع خلفية سوداء
                            if hazards:
                                txt    = hazards[0]
                                by     = min(y2 + 20, h - 4)
                                (tw, th), _ = cv2.getTextSize(
                                    txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                                cv2.rectangle(frame,
                                              (x1, by - th - 4),
                                              (x1 + tw + 6, by + 4),
                                              (0, 0, 0), -1)
                                cv2.putText(frame, txt, (x1 + 3, by),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                            (0, 0, 255), 2, cv2.LINE_AA)

                        # ── Log violations ────────────────────────────────
                        for haz in hazards:
                            if now - s["last_logged"].get(haz, 0) > 3:
                                log_event(pid, haz)
                                s["last_logged"][haz] = now
                                s["violation_count"]  += 1

                # ── any_hazard: per-frame reset (مش sticky) ──────────────
                with lock:
                    state["any_hazard"] = frame_has_hazard

                # ── Cleanup lost tracks ──────────────────────────────────
                for pid in list(persons.keys()):
                    if (pid not in matched_ids
                            and now - persons[pid]["last_seen"] > CFG["max_missing_time"] * 2):
                        del persons[pid]

                # ── Encode & share ───────────────────────────────────────
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                with lock:
                    state["frame_jpeg"]   = jpeg.tobytes()
                    state["persons"]      = persons
                    state["num_detected"] = len(persons)
            except Exception as frame_err:
                log.warning("Frame processing error (skipping): %s", frame_err)
                continue

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
    return send_from_directory('.', 'app.html')

@app.route('/api/rooms')
def get_rooms():
    return jsonify(list(ROOMS.values()))

@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "running": state["running"]})

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
        data = dict(state)
        data["rt_events"] = list(state["rt_events"])
        data["persons"]   = {}
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
