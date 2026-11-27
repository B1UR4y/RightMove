"""Local browser interface for the RightMove pose detector, rep counter and technique assessment."""
import argparse
import json
import math
import os
import secrets
import threading
import time
import webbrowser
from dataclasses import asdict, replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "true")

import cv2
import numpy as np
from camera_demo import YOLO, detect, overall_outcome

from rep_counter import MultiPersonRepCounter, SQUAT

from squat_assess import CAT_NA, ERR, NA, WARN, SquatAssessor

API_VERSION = "assessment-2"
ROOT = Path(__file__).resolve().parent
WARMUP_FRAMES = 8
RULE_NAMES = {
    "E1": "Недостаточная глубина",
    "E2": "Чрезмерный наклон корпуса",
    "E3": "Таз поднимается раньше плеч",
    "E4": "Неполное выпрямление",
}


class Workout:
    def __init__(self, now):
        self.last_seen = now
        self.warmup = []
        self.counter = None
        self.assessor = None
        self.rep_numbers = {}
        self.last_assessment = None
        self.wrong_view = False

    def update(self, boxes, people, now):
        self.last_seen = now
        if self.counter is None:
            self.warmup.append(now)
            if len(self.warmup) < WARMUP_FRAMES:
                return None
            elapsed = self.warmup[-1] - self.warmup[0]
            fps = max(1.0, min(60.0, (WARMUP_FRAMES - 1) / max(elapsed, 0.001)))
            self.counter = MultiPersonRepCounter(replace(SQUAT, fps=fps))
            self.assessor = SquatAssessor(fps)

        frame = self.counter.update(boxes, people)
        was_ready = self.assessor.calib is not None
        if was_ready and frame.event is not None and frame.event.valid:
            self.rep_numbers[frame.event.end_frame] = self.count + 1
        completed = self.assessor.update(
            self.counter.frame_idx, frame.kpts, frame.event, self.counter.state
        )
        if not was_ready and self.assessor.calib is not None:
            if self.assessor.calib.view_ok:
                # После стойки ждём новое движение, сохраняя общую нумерацию кадров.
                for tracked_counter in self.counter.counters.values():
                    tracked_counter.fsm.reset()
                self.wrong_view = False
            else:
                self.wrong_view = True
                self.assessor.recalibrate()
        if completed:
            self.last_assessment = self.describe(completed[-1])
        return frame

    @property
    def count(self):
        return len(self.rep_numbers)

    def recalibrate(self):
        if self.assessor is not None:
            self._flush()
            self.assessor.recalibrate()
            self.wrong_view = False

    def _flush(self):
        completed = self.assessor.flush(self.counter.frame_idx)
        if completed:
            self.last_assessment = self.describe(completed[-1])

    def describe(self, result):
        return {
            "attempt": result.attempt_id,
            "rep": self.rep_numbers.get(result.end_frame),
            "category": result.category,
            "outcome": overall_outcome(result),
            "rules": {code: result.rules.get(code, NA) for code in RULE_NAMES},
            "score": result.score,
            "note": result.note if result.category == CAT_NA else "",
            "unchecked": [
                RULE_NAMES.get(code, code) for code, value in result.rules.items() if value == NA
            ],
            "issues": [
                {"name": RULE_NAMES.get(code, code), "status": value}
                for code, value in result.rules.items() if value in (ERR, WARN)
            ],
        }

    def finish(self):
        if self.assessor is not None:
            self._flush()
        assessor = self.assessor or SquatAssessor()
        rows = {result.end_frame: self.describe(result) for result in assessor.results}
        missing = self.rep_numbers.keys() - rows.keys()
        for frame_idx in missing:
            rows[frame_idx] = {
                "rep": self.rep_numbers[frame_idx], "category": CAT_NA, "outcome": NA,
                "score": None, "note": "нет результата оценивания", "issues": [],
                "rules": {}, "unchecked": [],
            }
        attempts = [
            {**rows[frame_idx], "attempt": number}
            for number, frame_idx in enumerate(sorted(rows), start=1)
        ]
        summary = asdict(assessor.summary())
        summary["not_evaluated"] += len(missing)
        summary["preliminary"] = bool(attempts) and summary["not_evaluated"] * 2 >= len(attempts)
        return {**summary, "reps": self.count, "attempts": attempts}


class PoseService:
    def __init__(self, weights):
        self.model = YOLO(str(weights))
        detect(self.model, np.zeros((480, 640, 3), dtype=np.uint8))
        self.sessions = {}
        self.lock = threading.Lock()

    def open(self):
        now = time.monotonic()
        with self.lock:
            self.sessions = {key: item for key, item in self.sessions.items()
                             if now - item.last_seen < 300}
            key = secrets.token_urlsafe(24)
            self.sessions[key] = Workout(now)
        return key

    def close(self, key):
        with self.lock:
            workout = self.sessions.pop(key, None)
            return workout.finish() if workout is not None else None

    def recalibrate(self, key):
        with self.lock:
            workout = self.sessions[key]
            workout.recalibrate()
            workout.last_seen = time.monotonic()

    def frame(self, key, image):
        with self.lock:
            workout = self.sessions[key]
            boxes, people = detect(self.model, image)
            result = workout.update(boxes, people, time.monotonic())
            ready = workout.assessor is not None and workout.assessor.calib is not None
            sample = result.sample if result is not None else None
            points = result.kpts if result is not None else None
            return {
                "points": [] if points is None else np.where(np.isfinite(points), points, 0).tolist(),
                "width": image.shape[1], "height": image.shape[0],
                "ready": ready,
                "calibration": (100 if ready else round(100 * workout.assessor.calibrator.progress))
                if workout.assessor is not None else 0,
                "wrong_view": workout.wrong_view,
                "phase": workout.counter.state if ready else "UP",
                "count": workout.count,
                "summary": asdict(workout.assessor.summary()) if workout.assessor is not None else None,
                "angle": sample.smoothed if ready and sample and math.isfinite(sample.smoothed) else None,
                "sides": sample.sides_used if ready and sample else 0,
                "assessment": workout.last_assessment,
            }


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def respond(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-RightMove-API", API_VERSION)
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if host != self.server.local_host or (origin and origin != "http://" + host):
            return self.respond(403, {"error": "Откройте сайт по адресу из окна запуска."})
        if self.headers.get("X-RightMove-API") != API_VERSION:
            return self.respond(409, {"error": "Обновите страницу RightMove. Если ошибка осталась, перезапустите сервер из SquatCounter."})
        service = self.server.pose
        key = self.headers.get("X-Session", "")
        if self.path == "/api/session":
            return self.respond(201, {"session": service.open()})
        if self.path == "/api/finish":
            summary = service.close(key)
            if summary is None:
                return self.respond(410, {"error": "Сессия завершена. Начните тренировку ещё раз."})
            return self.respond(200, summary)
        if self.path == "/api/recalibrate":
            try:
                service.recalibrate(key)
            except KeyError:
                return self.respond(410, {"error": "Сессия завершена. Начните тренировку ещё раз."})
            return self.respond(200, {})
        if self.path != "/api/frame":
            return self.respond(404, {})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if self.headers.get_content_type() != "image/jpeg" or not 0 < size <= 2_000_000:
                return self.respond(400, {"error": "Некорректный кадр камеры."})
            image = cv2.imdecode(np.frombuffer(self.rfile.read(size), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return self.respond(400, {"error": "Не удалось прочитать кадр."})
            data = service.frame(key, image)
        except KeyError:
            return self.respond(410, {"error": "Сеанс завершён. Начните тренировку ещё раз."})
        except (ValueError, cv2.error, RuntimeError) as exc:
            print(exc, flush=True)
            return self.respond(500, {"error": "Не удалось обработать кадр камеры."})
        self.respond(200, data)

    def log_request(self, code="-", size="-"):
        if self.path != "/api/frame" or code != 200:
            super().log_request(code, size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--model", type=Path, default=ROOT / "yolov8n-pose.pt")
    args = parser.parse_args()
    print("Загрузка модели оценки позы…", flush=True)
    service = PoseService(args.model)
    handler = partial(Handler, directory=str(ROOT / "web"))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        server.pose = service
        server.local_host = f"127.0.0.1:{server.server_port}"
        url = f"http://{server.local_host}/"
        print(f"Верное движение — {url}\nДля остановки нажмите Ctrl+C.", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
