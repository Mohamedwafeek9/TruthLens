#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import logging
import os
import signal
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file, session
from ultralytics import YOLO
from werkzeug.security import check_password_hash

try:
    import torch
except ImportError:
    torch = None

try:
    from picamera2 import Picamera2
except ImportError as exc:
    Picamera2 = None
    PICAMERA_IMPORT_ERROR = exc
else:
    PICAMERA_IMPORT_ERROR = None

try:
    from libcamera import Transform, controls
except ImportError:
    Transform = None
    controls = None


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = BASE_DIR / "templates" / os.getenv("TRUTHLENS_TEMPLATE", "index.html")
LOG_LEVEL = os.getenv("TRUTHLENS_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("TruthLens")


EXAMINERS: dict[str, dict[str, Any]] = {
    "admin": {
        "id": 1,
        "name": "Admin",
        "role": "admin",
        "password_hash_env": "TRUTHLENS_ADMIN_PASSWORD_HASH",
    },
    "dr.m": {
        "id": 2,
        "name": "Dr. Mohamed",
        "role": "examiner",
        "password_hash_env": "TRUTHLENS_DRM_PASSWORD_HASH",
    },
    "dr.sara": {
        "id": 3,
        "name": "Dr. Sara",
        "role": "examiner",
        "password_hash_env": "TRUTHLENS_DRSARA_PASSWORD_HASH",
    },
}

ROOMS = {
    1: {"id": 1, "code": "A201", "name": "Lecture Hall A - 3rd Floor", "type": "lecture", "capacity": 200},
    2: {"id": 2, "code": "A202", "name": "Lecture Hall B - 3rd Floor", "type": "lecture", "capacity": 150},
    3: {"id": 3, "code": "A203", "name": "Lecture Hall C - 3rd Floor", "type": "lecture", "capacity": 120},
    4: {"id": 4, "code": "B101", "name": "Section Room 1 - 1st Floor", "type": "section", "capacity": 35},
    5: {"id": 5, "code": "B102", "name": "Section Room 2 - 1st Floor", "type": "section", "capacity": 35},
    6: {"id": 6, "code": "B103", "name": "Section Room 3 - 1st Floor", "type": "section", "capacity": 40},
    7: {"id": 7, "code": "B104", "name": "Section Room 4 - 1st Floor", "type": "section", "capacity": 30},
    8: {"id": 8, "code": "C301", "name": "Computer Lab 1 - 3rd Floor", "type": "lab", "capacity": 50},
    9: {"id": 9, "code": "C302", "name": "Computer Lab 2 - 3rd Floor", "type": "lab", "capacity": 50},
}


@dataclass(frozen=True)
class CameraConfig:
    width: int = int(os.getenv("TRUTHLENS_CAMERA_WIDTH", "1280"))
    height: int = int(os.getenv("TRUTHLENS_CAMERA_HEIGHT", "720"))
    fps: int = int(os.getenv("TRUTHLENS_CAMERA_FPS", "25"))
    pixel_format: str = os.getenv("TRUTHLENS_CAMERA_FORMAT", "BGR888").strip().upper()
    buffer_count: int = max(2, int(os.getenv("TRUTHLENS_CAMERA_BUFFERS", "4")))
    queue_frames: bool = os.getenv("TRUTHLENS_QUEUE_FRAMES", "false").lower() == "true"
    warmup_seconds: float = float(os.getenv("TRUTHLENS_CAMERA_WARMUP_SEC", "1.5"))
    warmup_frames: int = int(os.getenv("TRUTHLENS_CAMERA_WARMUP_FRAMES", "8"))
    reconnect_delay: float = float(os.getenv("TRUTHLENS_CAMERA_RECONNECT_SEC", "2.0"))
    flicker_hz: int = int(os.getenv("TRUTHLENS_FLICKER_HZ", "50"))
    awb_mode: str = os.getenv("TRUTHLENS_AWB_MODE", "auto").strip().lower()
    colour_gains: str = os.getenv("TRUTHLENS_COLOUR_GAINS", "").strip()
    hflip: bool = os.getenv("TRUTHLENS_HFLIP", "false").lower() == "true"
    vflip: bool = os.getenv("TRUTHLENS_VFLIP", "false").lower() == "true"


@dataclass(frozen=True)
class DetectorConfig:
    inf_width: int = int(os.getenv("TRUTHLENS_INF_WIDTH", "640"))
    inf_height: int = int(os.getenv("TRUTHLENS_INF_HEIGHT", "480"))
    pose_conf: float = float(os.getenv("TRUTHLENS_POSE_CONF", "0.38"))
    object_conf: float = float(os.getenv("TRUTHLENS_OBJECT_CONF", "0.32"))
    object_stride: int = max(1, int(os.getenv("TRUTHLENS_OBJECT_STRIDE", "2")))
    max_persons: int = int(os.getenv("TRUTHLENS_MAX_PERSONS", "8"))
    max_missing_seconds: float = float(os.getenv("TRUTHLENS_MAX_MISSING_SEC", "6.0"))
    track_distance: float = float(os.getenv("TRUTHLENS_TRACK_DISTANCE", "130"))
    iou_weight: float = float(os.getenv("TRUTHLENS_IOU_WEIGHT", "0.55"))
    smooth_alpha: float = float(os.getenv("TRUTHLENS_TRACK_ALPHA", "0.25"))
    event_cooldown: float = float(os.getenv("TRUTHLENS_EVENT_COOLDOWN_SEC", "3.0"))
    move_threshold: float = float(os.getenv("TRUTHLENS_MOVE_THRESHOLD", "28"))
    move_window: float = float(os.getenv("TRUTHLENS_MOVE_WINDOW_SEC", "2.8"))
    posture_threshold: float = float(os.getenv("TRUTHLENS_POSTURE_THRESHOLD", "145"))
    head_offset_threshold: float = float(os.getenv("TRUTHLENS_HEAD_OFFSET_THRESHOLD", "70"))
    processing_error_limit: int = int(os.getenv("TRUTHLENS_PROCESSING_ERROR_LIMIT", "5"))
    jpeg_quality: int = int(os.getenv("TRUTHLENS_JPEG_QUALITY", "82"))
    target_loop_fps: float = float(os.getenv("TRUTHLENS_TARGET_LOOP_FPS", "0"))


def make_stats() -> dict[str, int]:
    return {
        "bad_posture": 0,
        "copying": 0,
        "phone_use": 0,
        "phone_found": 0,
        "paper_switch": 0,
        "restless": 0,
    }


def build_initial_state() -> dict[str, Any]:
    return {
        "running": False,
        "camera_online": False,
        "camera_error": None,
        "detector_error": None,
        "fps": 0.0,
        "num_detected": 0,
        "any_hazard": False,
        "total_violations": 0,
        "session_start": None,
        "persons": {},
        "rt_events": deque(maxlen=500),
        "stats": make_stats(),
        "frame_jpeg": None,
        "last_frame_at": None,
    }


class CameraPipelineError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_colour_gains(raw_value: str) -> tuple[float, float] | None:
    if not raw_value:
        return None
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 2:
        raise ValueError("TRUTHLENS_COLOUR_GAINS must look like 1.6,1.3")
    return float(parts[0]), float(parts[1])


def resolve_awb_mode(mode: str) -> Any | None:
    if controls is None:
        return None
    awb_enum = getattr(controls, "AwbModeEnum", None)
    if awb_enum is None:
        return None
    names = {
        "auto": "Auto",
        "tungsten": "Tungsten",
        "fluorescent": "Fluorescent",
        "indoor": "Indoor",
        "daylight": "Daylight",
        "cloudy": "Cloudy",
    }
    return getattr(awb_enum, names.get(mode, "Auto"), None)


def compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter = max(0, x_b - x_a) * max(0, y_b - y_a)
    if inter <= 0:
        return 0.0
    area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return inter / float(area_a + area_b - inter)


def keypoints_to_box(kps: np.ndarray, sx: float, sy: float) -> tuple[int, int, int, int] | None:
    points = []
    for index in [0, 1, 2, 3, 4, 5, 6]:
        if kps[index][2] > 0.25:
            points.append((float(kps[index][0] * sx), float(kps[index][1] * sy)))
    if len(points) < 2:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    shoulder_dist = max(xs) - min(xs)
    if kps[5][2] > 0.3 and kps[6][2] > 0.3:
        shoulder_dist = abs(float(kps[5][0] - kps[6][0])) * sx

    scale = max(shoulder_dist, 20)
    x1 = int(min(xs) - scale * 0.60)
    y1 = int(min(ys) - scale * 1.10)
    x2 = int(max(xs) + scale * 0.60)
    y2 = int(max(ys) + scale * 1.80)

    if x2 - x1 < 40 or y2 - y1 < 40:
        return None
    if x2 - x1 > 650:
        center_x = (x1 + x2) // 2
        x1, x2 = center_x - 325, center_x + 325
    if y2 - y1 > 760:
        center_y = (y1 + y2) // 2
        y1, y2 = center_y - 380, center_y + 380
    return x1, y1, x2, y2


class Picamera2Pipeline:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._camera: Any = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        with self._lock:
            return self._camera is not None

    def start(self) -> None:
        if Picamera2 is None:
            raise CameraPipelineError(f"Picamera2 import failed: {PICAMERA_IMPORT_ERROR}")
        if Transform is None:
            raise CameraPipelineError("libcamera Python bindings are unavailable")

        with self._lock:
            if self._camera is not None:
                return

        camera = Picamera2()
        try:
            frame_us = max(1, int(1_000_000 / max(1, self.config.fps)))
            control_values = self._build_controls(frame_us)
            transform = Transform(hflip=self.config.hflip, vflip=self.config.vflip)
            video_config = camera.create_video_configuration(
                main={"size": (self.config.width, self.config.height), "format": self.config.pixel_format},
                controls=control_values,
                buffer_count=self.config.buffer_count,
                queue=self.config.queue_frames,
                transform=transform,
                display=None,
                encode=None,
            )
            camera.configure(video_config)
            camera.start()
            if self.config.warmup_seconds > 0:
                time.sleep(self.config.warmup_seconds)
            for _ in range(max(0, self.config.warmup_frames)):
                camera.capture_array("main")
        except Exception as exc:
            self._close_camera(camera)
            raise CameraPipelineError(f"camera start failed: {exc}") from exc

        with self._lock:
            if self._camera is None:
                self._camera = camera
            else:
                self._close_camera(camera)

        log.info(
            "Picamera2 started: %sx%s %s fps=%s queue=%s buffers=%s",
            self.config.width,
            self.config.height,
            self.config.pixel_format,
            self.config.fps,
            self.config.queue_frames,
            self.config.buffer_count,
        )

    def stop(self) -> None:
        with self._lock:
            camera = self._camera
            self._camera = None
        self._close_camera(camera)

    def reconnect(self) -> None:
        self.stop()
        time.sleep(self.config.reconnect_delay)
        self.start()

    def read_bgr(self) -> np.ndarray:
        with self._lock:
            if self._camera is None:
                raise CameraPipelineError("camera is not running")
            try:
                frame = self._camera.capture_array("main")
            except Exception as exc:
                raise CameraPipelineError(f"capture_array failed: {exc}") from exc
        if frame is None or getattr(frame, "size", 0) == 0:
            raise CameraPipelineError("camera returned an empty frame")
        return self._to_bgr(frame)

    def _build_controls(self, frame_us: int) -> dict[str, Any]:
        values: dict[str, Any] = {
            "FrameDurationLimits": (frame_us, frame_us),
            "AeEnable": True,
            "AwbEnable": True,
            "Sharpness": 1.0,
            "Contrast": 1.0,
            "Saturation": 1.0,
        }
        colour_gains = parse_colour_gains(self.config.colour_gains)
        if colour_gains is not None:
            values["AwbEnable"] = False
            values["ColourGains"] = colour_gains

        awb_mode = resolve_awb_mode(self.config.awb_mode)
        if awb_mode is not None and colour_gains is None:
            values["AwbMode"] = awb_mode

        if controls is not None and self.config.flicker_hz in (50, 60):
            flicker_enum = getattr(controls, "AeFlickerModeEnum", None)
            if flicker_enum is not None and hasattr(flicker_enum, "FlickerManual"):
                values["AeFlickerMode"] = flicker_enum.FlickerManual
                values["AeFlickerPeriod"] = int(1_000_000 / (self.config.flicker_hz * 2))

        if controls is not None:
            noise_enum = getattr(getattr(controls, "draft", None), "NoiseReductionModeEnum", None)
            if noise_enum is not None and hasattr(noise_enum, "Fast"):
                values["NoiseReductionMode"] = noise_enum.Fast
        return values

    def _to_bgr(self, frame: np.ndarray) -> np.ndarray:
        fmt = self.config.pixel_format
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim != 3:
            raise CameraPipelineError(f"unexpected frame shape {frame.shape}")
        if frame.shape[2] == 3:
            if fmt == "RGB888":
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return np.ascontiguousarray(frame)
        if frame.shape[2] == 4:
            if fmt.startswith("XBGR"):
                return np.ascontiguousarray(frame[:, :, :3])
            if fmt.startswith("XRGB"):
                return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        raise CameraPipelineError(f"unexpected channel count in frame shape {frame.shape}")

    @staticmethod
    def _close_camera(camera: Any) -> None:
        if camera is None:
            return
        try:
            camera.stop()
        except Exception:
            log.debug("ignored camera stop error", exc_info=True)
        try:
            camera.close()
        except Exception:
            log.debug("ignored camera close error", exc_info=True)


class ModelBundle:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.pose_model: YOLO | None = None
        self.object_model: YOLO | None = None
        self._lock = threading.Lock()

    def load(self) -> tuple[YOLO, YOLO]:
        with self._lock:
            if self.pose_model is not None and self.object_model is not None:
                return self.pose_model, self.object_model

            if torch is not None:
                torch.set_num_threads(max(1, int(os.getenv("TRUTHLENS_TORCH_THREADS", "3"))))

            pose_path = self.base_dir / os.getenv("TRUTHLENS_POSE_MODEL", "yolov8n-pose.pt")
            object_path = self.base_dir / os.getenv("TRUTHLENS_OBJECT_MODEL", "yolov8n.pt")
            log.info("loading models: pose=%s object=%s", pose_path, object_path)
            self.pose_model = YOLO(str(pose_path))
            self.object_model = YOLO(str(object_path))
            try:
                self.pose_model.fuse()
                self.object_model.fuse()
            except Exception:
                log.debug("model fuse skipped", exc_info=True)
            return self.pose_model, self.object_model

    def reload(self) -> tuple[YOLO, YOLO]:
        with self._lock:
            self.pose_model = None
            self.object_model = None
        gc.collect()
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return self.load()


class TrackManager:
    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.tracks: dict[str, dict[str, Any]] = {}
        self.next_track_num = 1

    def match_or_create(self, pos: np.ndarray, box: tuple[int, int, int, int] | None) -> str | None:
        best_id = None
        best_score = float("inf")

        for track_id, track in self.tracks.items():
            smooth_pos = track.get("smooth_nose")
            if smooth_pos is None:
                continue
            distance = float(np.linalg.norm(pos - smooth_pos))
            if distance > self.config.track_distance * 2.75:
                continue

            dist_score = distance / self.config.track_distance
            iou_score = 1.0
            prev_box = track.get("last_box")
            if prev_box is not None and box is not None:
                iou_score = 1.0 - compute_iou(prev_box, box)

            age_bonus = min(0.15, track.get("seen_count", 0) * 0.01)
            score = (
                (1.0 - self.config.iou_weight) * dist_score
                + self.config.iou_weight * iou_score
                - age_bonus
            )
            if score < best_score:
                best_id = track_id
                best_score = score

        if best_id is not None and best_score <= 1.85:
            return best_id
        if len(self.tracks) >= self.config.max_persons:
            return None

        track_id = f"T{self.next_track_num:03d}"
        self.next_track_num += 1
        self.tracks[track_id] = {
            "smooth_nose": pos.copy(),
            "prev_pos": pos.copy(),
            "last_box": box,
            "move_times": deque(maxlen=80),
            "last_seen": time.time(),
            "seen_count": 0,
            "violation_count": 0,
            "last_logged": {},
        }
        return track_id

    def update(self, track_id: str, pos: np.ndarray, box: tuple[int, int, int, int] | None, now: float) -> float:
        track = self.tracks[track_id]
        track["last_seen"] = now
        track["seen_count"] += 1
        track["smooth_nose"] = (
            self.config.smooth_alpha * pos
            + (1.0 - self.config.smooth_alpha) * track["smooth_nose"]
        )
        if box is not None:
            track["last_box"] = box
        movement = float(np.linalg.norm(track["smooth_nose"] - track["prev_pos"]))
        track["prev_pos"] = track["smooth_nose"].copy()
        if movement > self.config.move_threshold:
            track["move_times"].append(now)
        return movement

    def cleanup(self, matched_ids: set[str], now: float) -> None:
        for track_id in list(self.tracks.keys()):
            if track_id in matched_ids:
                continue
            if now - self.tracks[track_id]["last_seen"] > self.config.max_missing_seconds:
                del self.tracks[track_id]


class TruthLensRuntime:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.camera_config = CameraConfig()
        self.detector_config = DetectorConfig()
        self.camera = Picamera2Pipeline(self.camera_config)
        self.models = ModelBundle(base_dir)
        self.state = build_initial_state()
        self.state_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.next_event_id = 1
        self.status_frame = self._make_status_frame(["TruthLens Pro", "Ready", "Waiting for camera"])

    def start(self) -> bool:
        with self.state_lock:
            if self.worker and self.worker.is_alive():
                return False
            self.stop_event.clear()
            self.state = build_initial_state()
            self.state["running"] = True
            self.state["session_start"] = now_iso()
            self.state["frame_jpeg"] = self.status_frame

        self.worker = threading.Thread(target=self._worker_loop, name="truthlens-detector", daemon=True)
        self.worker.start()
        return True

    def stop(self) -> None:
        self.stop_event.set()
        worker = self.worker
        if worker and worker.is_alive():
            worker.join(timeout=8)
        self.worker = None
        self.camera.stop()
        with self.state_lock:
            self.state["running"] = False
            self.state["camera_online"] = False
            self.state["any_hazard"] = False
            self.state["frame_jpeg"] = self._make_status_frame(["TruthLens Pro", "Stopped", "Detection is idle"])

    def snapshot(self) -> dict[str, Any]:
        with self.state_lock:
            return {
                "running": self.state["running"],
                "camera_online": self.state["camera_online"],
                "camera_error": self.state["camera_error"],
                "detector_error": self.state["detector_error"],
                "fps": self.state["fps"],
                "num_detected": self.state["num_detected"],
                "any_hazard": self.state["any_hazard"],
                "total_violations": self.state["total_violations"],
                "session_start": self.state["session_start"],
                "last_frame_at": self.state["last_frame_at"],
                "stats": dict(self.state["stats"]),
                "rt_events": list(self.state["rt_events"]),
            }

    def current_frame(self) -> bytes | None:
        with self.state_lock:
            return self.state.get("frame_jpeg")

    def _worker_loop(self) -> None:
        tracks = TrackManager(self.detector_config)
        frame_index = 0
        cached_object_classes: list[int] = []
        processing_errors = 0
        prev_time = time.perf_counter()

        while not self.stop_event.is_set():
            try:
                if not self.camera.running:
                    self._set_status(camera_online=False, camera_error="camera starting")
                    self._set_frame(self._make_status_frame(["TruthLens Pro", "Starting camera", "Warming up AE/AWB"]))
                    self.camera.start()
                    self._set_status(camera_online=True, camera_error=None)

                frame = self.camera.read_bgr()
                pose_model, object_model = self.models.load()
                result_frame, cached_object_classes = self._process_frame(
                    frame,
                    tracks,
                    pose_model,
                    object_model,
                    cached_object_classes,
                    frame_index,
                )

                ok, jpeg = cv2.imencode(
                    ".jpg",
                    result_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.detector_config.jpeg_quality],
                )
                if not ok:
                    raise RuntimeError("jpeg encoding failed")

                now_perf = time.perf_counter()
                raw_fps = 1.0 / max(now_perf - prev_time, 1e-6)
                prev_time = now_perf
                with self.state_lock:
                    prev_fps = self.state.get("fps", raw_fps)
                    self.state["fps"] = round(0.15 * raw_fps + 0.85 * prev_fps, 1)
                    self.state["frame_jpeg"] = jpeg.tobytes()
                    self.state["last_frame_at"] = now_iso()
                    self.state["num_detected"] = len(tracks.tracks)
                    self.state["camera_online"] = True
                    self.state["camera_error"] = None
                    self.state["detector_error"] = None

                frame_index += 1
                processing_errors = 0
                self._rate_limit(prev_time)

            except CameraPipelineError as exc:
                log.warning("camera failure: %s", exc)
                self._set_status(camera_online=False, camera_error=str(exc))
                self._set_frame(self._make_status_frame(["TruthLens Pro", "Camera recovering", str(exc)]))
                try:
                    self.camera.reconnect()
                except Exception as reconnect_exc:
                    log.warning("camera reconnect failed: %s", reconnect_exc)
                    time.sleep(self.camera_config.reconnect_delay)

            except Exception as exc:
                processing_errors += 1
                message = f"{type(exc).__name__}: {exc}"
                log.warning("detector failure: %s", message)
                log.debug("detector traceback:\n%s", traceback.format_exc())
                self._set_status(detector_error=message)
                if processing_errors >= self.detector_config.processing_error_limit:
                    log.warning("reloading YOLO models after repeated processing failures")
                    self._set_frame(self._make_status_frame(["TruthLens Pro", "Reloading AI models", message]))
                    try:
                        self.models.reload()
                    except Exception as reload_exc:
                        log.error("model reload failed: %s", reload_exc)
                    processing_errors = 0
                time.sleep(0.2)

        self.camera.stop()
        with self.state_lock:
            self.state["running"] = False
            self.state["camera_online"] = False

    def _process_frame(
        self,
        frame: np.ndarray,
        tracks: TrackManager,
        pose_model: YOLO,
        object_model: YOLO,
        cached_object_classes: list[int],
        frame_index: int,
    ) -> tuple[np.ndarray, list[int]]:
        cfg = self.detector_config
        h, w = frame.shape[:2]
        inference_frame = cv2.resize(frame, (cfg.inf_width, cfg.inf_height), interpolation=cv2.INTER_LINEAR)
        sx = w / cfg.inf_width
        sy = h / cfg.inf_height

        with torch.no_grad() if torch is not None else nullcontext():
            pose_res = pose_model.predict(
                inference_frame,
                verbose=False,
                conf=cfg.pose_conf,
                imgsz=max(cfg.inf_width, cfg.inf_height),
            )[0]

            if frame_index % cfg.object_stride == 0:
                object_res = object_model.predict(
                    inference_frame,
                    verbose=False,
                    conf=cfg.object_conf,
                    imgsz=max(cfg.inf_width, cfg.inf_height),
                    classes=[67, 73, 84],
                )[0]
                cached_object_classes = self._object_classes(object_res.boxes)

        now = time.time()
        matched_ids: set[str] = set()
        frame_has_hazard = False

        if pose_res.keypoints is not None:
            for kps in pose_res.keypoints.data.cpu().numpy():
                nose = kps[0]
                left_ear = kps[3]
                right_ear = kps[4]
                left_sh = kps[5]
                right_sh = kps[6]
                left_wr = kps[9]
                right_wr = kps[10]
                if nose[2] < 0.32:
                    continue

                pos = np.array([nose[0] * sx, nose[1] * sy], dtype=np.float32)
                box = keypoints_to_box(kps, sx, sy)
                track_id = tracks.match_or_create(pos, box)
                if track_id is None:
                    continue

                movement = tracks.update(track_id, pos, box, now)
                matched_ids.add(track_id)
                track = tracks.tracks[track_id]
                hazards = self._classify_hazards(
                    track,
                    nose,
                    left_ear,
                    right_ear,
                    left_sh,
                    right_sh,
                    left_wr,
                    right_wr,
                    sy,
                    sx,
                    cached_object_classes,
                    now,
                )
                if hazards:
                    frame_has_hazard = True
                self._draw_track(frame, track_id, box, hazards, movement > cfg.move_threshold)
                for hazard in hazards:
                    if now - track["last_logged"].get(hazard, 0) > cfg.event_cooldown:
                        track["last_logged"][hazard] = now
                        track["violation_count"] += 1
                        self._log_event(track_id, hazard)

        tracks.cleanup(matched_ids, now)
        with self.state_lock:
            self.state["any_hazard"] = frame_has_hazard
        return frame, cached_object_classes

    def _classify_hazards(
        self,
        track: dict[str, Any],
        nose: np.ndarray,
        left_ear: np.ndarray,
        right_ear: np.ndarray,
        left_sh: np.ndarray,
        right_sh: np.ndarray,
        left_wr: np.ndarray,
        right_wr: np.ndarray,
        sy: float,
        sx: float,
        object_classes: list[int],
        now: float,
    ) -> list[str]:
        cfg = self.detector_config
        hazards: list[str] = []

        if left_sh[2] > 0.3 and right_sh[2] > 0.3:
            shoulder_y = ((left_sh[1] + right_sh[1]) / 2.0) * sy
            if abs(float(track["smooth_nose"][1] - shoulder_y)) > cfg.posture_threshold:
                hazards.append("Bad Posture")

        if left_sh[2] > 0.3 and right_sh[2] > 0.3:
            shoulder_mid_y = ((left_sh[1] + right_sh[1]) / 2.0) * sy
            if nose[1] * sy > shoulder_mid_y + 30 and "Bad Posture" not in hazards:
                hazards.append("Bad Posture")

        shoulder_x = float(track["smooth_nose"][0])
        if left_sh[2] > 0.3 and right_sh[2] > 0.3:
            shoulder_x = ((left_sh[0] + right_sh[0]) / 2.0) * sx
        head_side = left_ear[2] < 0.3 or right_ear[2] < 0.3
        if head_side and abs(float(track["smooth_nose"][0] - shoulder_x)) > cfg.head_offset_threshold:
            hazards.append("Copying")

        chin_y = nose[1] * sy + 60
        wrist_near_face = (
            (left_wr[2] > 0.3 and left_wr[1] * sy < chin_y)
            or (right_wr[2] > 0.3 and right_wr[1] * sy < chin_y)
        )
        head_down = False
        if left_sh[2] > 0.3 and right_sh[2] > 0.3:
            shoulder_mid_y = ((left_sh[1] + right_sh[1]) / 2.0) * sy
            head_down = nose[1] * sy > shoulder_mid_y + 50
        if wrist_near_face and head_down:
            hazards.append("Phone Use")

        if 67 in object_classes:
            hazards.append("Phone Found")
        paper_count = object_classes.count(73) + object_classes.count(84)
        if paper_count > 1 and (left_wr[2] > 0.3 or right_wr[2] > 0.3):
            hazards.append("Paper Switch")

        recent_moves = sum(1 for ts in track["move_times"] if now - ts < cfg.move_window)
        if recent_moves > 3:
            hazards.append("Restless")
        return hazards

    def _draw_track(
        self,
        frame: np.ndarray,
        track_id: str,
        box: tuple[int, int, int, int] | None,
        hazards: list[str],
        moving: bool,
    ) -> None:
        if box is None:
            return
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return

        color = (0, 0, 255) if hazards else (0, 165, 255) if moving else (0, 210, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        self._label(frame, track_id, (x1, max(22, y1 - 12)), color)
        if hazards:
            self._label(frame, hazards[0], (x1, min(h - 8, y2 + 24)), (0, 0, 255), scale=0.55)

    @staticmethod
    def _label(
        frame: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        scale: float = 0.65,
    ) -> None:
        x, y = origin
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.rectangle(frame, (x, y - th - 5), (x + tw + 7, y + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    def _log_event(self, track_id: str, violation: str) -> None:
        event = {
            "id": self.next_event_id,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "track_id": track_id,
            "violation": violation,
        }
        self.next_event_id += 1
        key = violation.lower().replace(" ", "_")
        with self.state_lock:
            self.state["rt_events"].appendleft(event)
            self.state["total_violations"] += 1
            if key in self.state["stats"]:
                self.state["stats"][key] += 1

    def _set_status(self, **updates: Any) -> None:
        with self.state_lock:
            self.state.update(updates)

    def _set_frame(self, jpeg: bytes) -> None:
        with self.state_lock:
            self.state["frame_jpeg"] = jpeg

    def _make_status_frame(self, lines: list[str]) -> bytes:
        width, height = self.camera_config.width, self.camera_config.height
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (19, 21, 27)
        cv2.rectangle(frame, (44, 44), (width - 44, height - 44), (70, 76, 88), 2)
        cv2.putText(frame, lines[0], (84, 154), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 210, 245), 3, cv2.LINE_AA)
        y = 236
        for line in lines[1:4]:
            cv2.putText(frame, line[:96], (84, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (224, 230, 238), 2, cv2.LINE_AA)
            y += 48
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.detector_config.jpeg_quality])
        if not ok:
            raise RuntimeError("status frame encode failed")
        return jpeg.tobytes()

    def _rate_limit(self, loop_started_at: float) -> None:
        target = self.detector_config.target_loop_fps
        if target <= 0:
            return
        minimum = 1.0 / target
        elapsed = time.perf_counter() - loop_started_at
        if elapsed < minimum:
            time.sleep(minimum - elapsed)

    @staticmethod
    def _object_classes(boxes: Any) -> list[int]:
        if boxes is None or getattr(boxes, "cls", None) is None:
            return []
        return [int(value) for value in boxes.cls.tolist()]


class nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: Any) -> bool:
        return False


def load_examiners() -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    json_path = os.getenv("TRUTHLENS_EXAMINERS_FILE", "").strip()
    if json_path:
        with open(json_path, "r", encoding="utf-8") as handle:
            raw_users = json.load(handle)
        for username, data in raw_users.items():
            if data.get("password_hash"):
                loaded[username] = data

    for username, data in EXAMINERS.items():
        password_hash = os.getenv(data["password_hash_env"], "").strip()
        if not password_hash:
            continue
        loaded[username] = {
            "id": data["id"],
            "name": data["name"],
            "role": data["role"],
            "password_hash": password_hash,
        }

    if not loaded:
        log.error("no enabled examiners found; configure password hashes in /etc/default/truthlens-pro")
    return loaded


runtime = TruthLensRuntime(BASE_DIR)
USERS = load_examiners()

app = Flask(__name__)
app.secret_key = os.getenv("TRUTHLENS_SECRET_KEY", "")
if not app.secret_key:
    if os.getenv("TRUTHLENS_ALLOW_INSECURE_SECRET", "false").lower() != "true":
        raise RuntimeError("TRUTHLENS_SECRET_KEY must be set in /etc/default/truthlens-pro")
    log.warning("using insecure development session secret because TRUTHLENS_ALLOW_INSECURE_SECRET=true")
    app.secret_key = "truthlens-development-secret"
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("TRUTHLENS_SECURE_COOKIE", "false").lower() == "true",
    PERMANENT_SESSION_LIFETIME=int(os.getenv("TRUTHLENS_SESSION_SECONDS", "28800")),
)


def require_login(view: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(view)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not session.get("username"):
            return jsonify({"error": "unauthorized"}), 401
        return view(*args, **kwargs)

    return wrapped


def current_user() -> dict[str, Any] | None:
    username = session.get("username")
    if not username:
        return None
    user = USERS.get(username)
    if not user:
        session.clear()
        return None
    return {"username": username, "name": user["name"], "role": user["role"], "room": session.get("room")}


@app.after_request
def add_security_headers(response: Response) -> Response:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/")
def index() -> Response:
    return send_file(TEMPLATE_PATH)


@app.route("/api/rooms")
def rooms() -> Response:
    return jsonify(list(ROOMS.values()))


@app.route("/api/login", methods=["POST"])
def login() -> Response:
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    room_id = payload.get("room_id")
    user = USERS.get(username)

    if not user or not check_password_hash(user.get("password_hash", ""), password):
        log.warning("failed login for username=%s from %s", username, request.remote_addr)
        return jsonify({"error": "invalid credentials"}), 401
    try:
        room_key = int(room_id)
    except (TypeError, ValueError):
        return jsonify({"error": "invalid room"}), 400
    if room_key not in ROOMS:
        return jsonify({"error": "unknown room"}), 400

    session.clear()
    session.permanent = True
    session["username"] = username
    session["room"] = ROOMS[room_key]
    log.info("login success username=%s room=%s", username, ROOMS[room_key]["code"])
    return jsonify({"user": current_user(), "status": runtime.snapshot()})


@app.route("/api/logout", methods=["POST"])
def logout() -> Response:
    session.clear()
    return jsonify({"status": "ok"})


@app.route("/api/session")
def session_status() -> Response:
    user = current_user()
    if not user:
        return jsonify({"authenticated": False}), 401
    return jsonify({"authenticated": True, "user": user, "status": runtime.snapshot()})


@app.route("/api/health")
def health() -> Response:
    status = runtime.snapshot()
    return jsonify(
        {
            "status": "ok",
            "running": status["running"],
            "camera_online": status["camera_online"],
            "camera_error": status["camera_error"],
            "detector_error": status["detector_error"],
        }
    )


@app.route("/api/status")
@require_login
def status() -> Response:
    return jsonify(runtime.snapshot())


@app.route("/api/start", methods=["POST"])
@require_login
def start_detection() -> Response:
    started = runtime.start()
    return jsonify({"status": "started" if started else "already_running", "snapshot": runtime.snapshot()})


@app.route("/api/stop", methods=["POST"])
@require_login
def stop_detection() -> Response:
    user = current_user()
    if user and user["role"] != "admin":
        return jsonify({"error": "admin required"}), 403
    runtime.stop()
    return jsonify({"status": "stopped", "snapshot": runtime.snapshot()})


def mjpeg_frames():
    interval = max(0.02, float(os.getenv("TRUTHLENS_STREAM_INTERVAL_SEC", "0.04")))
    while True:
        frame = runtime.current_frame()
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(interval)


@app.route("/video_feed")
@require_login
def video_feed() -> Response:
    response = Response(mjpeg_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


def shutdown_handler(*_: Any) -> None:
    log.info("shutdown requested")
    runtime.stop()
    raise SystemExit(0)


def main() -> None:
    host = os.getenv("TRUTHLENS_HOST", "0.0.0.0")
    port = int(os.getenv("TRUTHLENS_PORT", "5050"))
    autostart = os.getenv("TRUTHLENS_AUTOSTART_DETECTION", "true").lower() == "true"

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    if autostart:
        runtime.start()

    try:
        from waitress import serve
    except ImportError:
        log.warning("waitress not installed; using Flask development server")
        app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
    else:
        serve(app, host=host, port=port, threads=int(os.getenv("TRUTHLENS_WAITRESS_THREADS", "8")))


if __name__ == "__main__":
    main()
