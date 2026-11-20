import sys
import time
import dataclasses
import logging
import os
from datetime import datetime

import cv2
import numpy as np
import torch

from ultralytics import YOLO

from rep_counter import SQUAT, COCO, MultiPersonRepCounter
from squat_assess import (SquatAssessor, NORM, WARN, ERR, NA,
                          CAT_REP, CAT_PARTIAL, CAT_NA)

# python camera_demo.py --window

# ============================================================

# ДЕМО: ПОДСЧЁТ ПРИСЕДАНИЙ С ВЕБ-КАМЕРЫ
#
#   python camera_demo.py          — камера по умолчанию
#   python camera_demo.py 1        — другая камера
#   python camera_demo.py file.mp4 — видеофайл вместо камеры
#   python camera_demo.py --window — сразу в окне, а не во весь экран
#                                    (можно вместе с камерой: camera_demo.py 1 --window)
#
# По умолчанию окно открывается во весь экран.
# Клавиши: Esc или q — выход, r — сбросить счёт, f — полный экран / окно,
# v — начать / остановить запись видео (recordings/live/): весь экран с панелями и скелетом;
# --rec-raw — писать исходный кадр камеры без оверлея.
# --hd — запросить у камеры 1280x720 (по умолчанию режим камеры не меняется).
# c — повторить калибровку (стойка боком 2-3 с).
# Сначала идёт калибровка: встаньте боком и стойте спокойно, пока полоса не заполнится.
# После каждой попытки показывается категория: REP / PARTIAL / NOT RATED
# и исход по правилам (OK / WARN / ERROR). Журнал пишется в logs/.
# Работают в любой раскладке (q = й, r = к, f = а).
# Встаньте боком к камере, чтобы были видны бедро, колено и лодыжка.
# ============================================================
WINDOW = "Squat counter"
# Окно всегда 16:9. Размер по умолчанию 1280x720 (HD): помещается на экран 1366x768
# и выше, текст и панели читаются, а на Full HD окно можно растянуть мышью.
CANVAS_W, CANVAS_H = 1280, 720
MODEL_FILE = "yolov8n-pose.pt"
DETECTION_CONF = 0.25
# Видеокарта NVIDIA, если есть, иначе процессор (медленнее, но работает на любом компьютере)
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WARMUP_FRAMES = 20  # по ним меряем реальный FPS: от него зависят пороги длительности повторения

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
SIGNAL_JOINTS = {COCO[name] for t in SQUAT.angle_triplets for name in t}
STATE_COLORS = {"UP": (80, 175, 76), "DESCENT": (0, 165, 255),
                "BOTTOM": (60, 60, 230), "ASCENT": (230, 160, 30)}

# Коды клавиш в латинской и русской раскладке. В русской раскладке waitKey
# возвращает код буквы (й, к, а) — в Unicode или в cp1251, проверяем все варианты.
def _keys(*chars):
    codes = set()
    for ch in chars:
        codes.add(ord(ch))
        codes.add(ch.encode("cp1251")[0])
    return codes


KEYS_QUIT = {27} | _keys("q", "Q", "й", "Й")
KEYS_RESET = _keys("r", "R", "к", "К")
KEYS_FULLSCREEN = _keys("f", "F", "а", "А")
KEYS_CALIB = _keys("c", "C", "с", "С")
KEYS_REC = _keys("v", "V", "м", "М")
REC_DIR = os.path.join("recordings", "live")


class Recorder:
    """Запись видео в recordings/live/rec_<время>.mp4 (кодек mp4v, есть в OpenCV без доп. библиотек)."""

    def __init__(self, fps):
        self.fps = max(5.0, min(60.0, fps))
        self.writer = None
        self.path = None
        self.frames = 0
        self.t0 = 0.0

    @property
    def active(self):
        return self.writer is not None

    def start(self, frame):
        os.makedirs(REC_DIR, exist_ok=True)
        self.path = os.path.join(REC_DIR, datetime.now().strftime("rec_%Y%m%d_%H%M%S.mp4"))
        h, w = frame.shape[:2]
        self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        if not self.writer.isOpened():
            self.writer = None
            return False
        self.frames = 0
        self.t0 = time.perf_counter()
        return True

    def write(self, frame):
        if self.writer is not None:
            self.writer.write(frame)
            self.frames += 1

    def stop(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        return self.path, self.frames

    def elapsed(self):
        return time.perf_counter() - self.t0

OUTCOME_COLORS = {NORM: (80, 175, 76), WARN: (0, 200, 255), ERR: (60, 60, 230), NA: (170, 170, 170)}
CATEGORY_LABELS = {CAT_REP: "REP", CAT_PARTIAL: "PARTIAL", CAT_NA: "NOT RATED"}
RULE_NAMES = {"E1": "depth", "E2": "lean", "E3": "hips", "E4": "top"}


def setup_log():
    """Журнал в файл logs/session_<время>.log (UTF-8) и в консоль."""
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", datetime.now().strftime("session_%Y%m%d_%H%M%S.log"))
    log = logging.getLogger("squat")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    log.addHandler(fh)
    return log, path


def overall_outcome(result):
    """Худший исход среди правил попытки."""
    if result.category == CAT_NA:
        return NA
    outs = list(result.rules.values())
    return ERR if ERR in outs else WARN if WARN in outs else NORM


def log_result(log, r):
    feats = ", ".join(f"{k}={v:.1f}" for k, v in r.features.items() if v is not None)
    rules = ", ".join(f"{k}:{v}" for k, v in r.rules.items())
    msg = (f"Попытка {r.attempt_id}: {r.category}; правила: {rules or '-'}; "
           f"балл: {r.score}; {r.note}; признаки: {feats}")
    log.info(msg)
    print(msg)


def to_canvas(frame):
    """Кадр камеры вписывается в холст 16:9 с чёрными полями, пропорции кадра не искажаются.
    Панели рисуются уже на холсте, поэтому их размер не зависит от разрешения камеры."""
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    scale = min(CANVAS_W / w, CANVAS_H / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x0, y0 = (CANVAS_W - nw) // 2, (CANVAS_H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def set_fullscreen(on, frame_size):
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)
    if not on:
        # На Windows после выхода из полного экрана окно остаётся размером с экран
        # и без рамки. Задаём обычный размер и ставим окно в угол экрана.
        w, h = frame_size
        cv2.resizeWindow(WINDOW, w, h)
        cv2.moveWindow(WINDOW, 60, 60)


def window_closed():
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def draw_skeleton(img, kpts, conf_th=0.3):
    if kpts is None:
        return
    pts = [(int(x), int(y), s) for x, y, s in kpts]
    for a, b in COCO_SKELETON:
        if pts[a][2] < conf_th or pts[b][2] < conf_th:
            continue
        color = (0, 220, 255) if a in SIGNAL_JOINTS and b in SIGNAL_JOINTS else (255, 128, 0)
        cv2.line(img, pts[a][:2], pts[b][:2], color, 3)
    for i, (x, y, s) in enumerate(pts):
        if s >= conf_th:
            cv2.circle(img, (x, y), 6 if i in SIGNAL_JOINTS else 4, (0, 255, 0), -1)


def draw_panel(img, count, state, angle, fps, flash):
    cv2.rectangle(img, (10, 10), (370, 130), (0, 0, 0), -1)
    cv2.putText(img, f"REPS: {count}", (22, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    if flash:
        cv2.putText(img, "+1", (225, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (80, 220, 80), 3)
    cv2.putText(img, state, (22, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, STATE_COLORS[state], 2)
    angle_txt = "--" if angle is None or np.isnan(angle) else f"{angle:.0f} deg"
    cv2.putText(img, f"knee {angle_txt}", (155, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    cv2.putText(img, f"{fps:.0f} FPS  Esc quit  r reset  c calib  v rec  f screen", (22, 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


def draw_assess(img, asr, last):
    """Нижняя панель: калибровка, категория последней попытки, итоги подхода."""
    h, w = img.shape[:2]
    if asr.calib is None:
        cv2.rectangle(img, (10, 140), (460, 215), (0, 0, 0), -1)
        cv2.putText(img, "CALIBRATION: stand sideways, stay still", (20, 168),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.rectangle(img, (20, 185), (440, 203), (90, 90, 90), 2)
        cv2.rectangle(img, (20, 185), (20 + int(420 * asr.calibrator.progress), 203), (0, 200, 255), -1)
        return
    c = asr.calib
    if not c.view_ok:
        cv2.rectangle(img, (10, 140), (520, 180), (0, 0, 0), -1)
        cv2.putText(img, f"TURN SIDEWAYS (view {c.view_ratio:.2f})  c: recalibrate", (20, 168),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 230), 2)
    cv2.rectangle(img, (10, 140), (370, 260), (0, 0, 0), -1)
    if last is None:
        cv2.putText(img, "Waiting for attempt...", (22, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    else:
        out = overall_outcome(last)
        color = OUTCOME_COLORS[out]
        label = {NORM: "OK", WARN: "WARN", ERR: "ERROR", NA: "-"}[out]
        cv2.putText(img, f"{CATEGORY_LABELS[last.category]}  {label}", (22, 178),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
        x = 22
        for rule, name in RULE_NAMES.items():
            o = last.rules.get(rule, NA)
            cv2.circle(img, (x + 8, 200), 8, OUTCOME_COLORS[o], -1)
            cv2.putText(img, name, (x + 20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            x += 85
        if last.score is not None:
            cv2.putText(img, f"score {last.score}", (22, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    s = asr.summary()
    rating = "-" if s.rating is None else f"{s.rating}/5" + ("*" if s.preliminary else "")
    cv2.putText(img, f"partial {s.partial}  not rated {s.not_evaluated}  grade {rating}", (22, 252),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)


def detect(model, frame):
    r = model.predict(frame, device=DEVICE, imgsz=640, verbose=False, conf=DETECTION_CONF)[0]
    if r.keypoints is None or len(r.keypoints) == 0:
        return [], []
    return list(r.boxes.xyxy.cpu().numpy()), list(r.keypoints.data.cpu().numpy())


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    windowed = "--window" in sys.argv[1:]
    arg = args[0] if args else "0"
    source = int(arg) if arg.isdigit() else arg
    is_camera = isinstance(source, int)

    # CAP_DSHOW — быстрый запуск камеры на Windows; на других системах — по умолчанию
    if is_camera and sys.platform == "win32":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if is_camera and "--hd" in sys.argv[1:]:
        # Необязательно: просим у камеры 1280x720 в формате MJPG. По умолчанию режим камеры не
        # меняем: некоторые камеры на Windows после смены режима отдают чёрный кадр или ~1 FPS.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"Не удалось открыть источник: {source}")
        return

    print(f"Загрузка модели... (вычисления: {'видеокарта' if DEVICE == 0 else 'процессор'})")
    model = YOLO(MODEL_FILE)
    model.predict(np.zeros((480, 640, 3), dtype=np.uint8), device=DEVICE, verbose=False)

    # Замер реального FPS (камера + модель): пороги в секундах пересчитываются в кадры по нему
    print("Замер скорости...")
    t0 = time.perf_counter()
    for _ in range(WARMUP_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        detect(model, frame)
    fps = WARMUP_FRAMES / (time.perf_counter() - t0)
    if not is_camera:
        fps = cap.get(cv2.CAP_PROP_FPS) or fps  # у файла свой темп
    print(f"FPS: {fps:.1f}, кадр камеры {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    if is_camera and fps < 10:
        print("Низкий FPS: камера или модель работают медленно. Не используйте --hd, закройте другие программы с камерой.")
    config = dataclasses.replace(SQUAT, fps=fps)

    counter = MultiPersonRepCounter(config)
    asr = SquatAssessor(fps)
    last_result = None
    rec = Recorder(fps)
    rec_overlay = "--rec-raw" not in sys.argv[1:]  # по умолчанию с оверлеем
    log, log_path = setup_log()
    log.info(f"Старт, источник {source}, FPS {fps:.1f}")
    print(f"Журнал: {log_path}")
    flash_until = 0.0
    shown_fps = fps
    prev = time.perf_counter()

    # WINDOW_KEEPRATIO: на весь экран без растягивания картинки
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # Размер кадра камеры — размер окна в оконном режиме
    frame_size = (CANVAS_W, CANVAS_H)
    fullscreen = not windowed
    set_fullscreen(fullscreen, frame_size)
    print("Готово. Приседайте. Esc или q — выход, r — сброс счёта, f — полный экран / окно.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if is_camera:
            frame = cv2.flip(frame, 1)  # как в зеркале

        boxes, kpts = detect(model, frame)
        res = counter.update(boxes, kpts)
        if res.event is not None and res.event.valid:
            flash_until = time.perf_counter() + 0.7
            print(f"Повторение #{res.count}: угол {res.event.primary_signal_max:.0f}° -> "
                  f"{res.event.primary_signal_min:.0f}°, {res.event.duration_ms / 1000:.1f} с")

        had_calib = asr.calib is not None
        for r in asr.update(counter.frame_idx, res.kpts, res.event, counter.state):
            last_result = r
            log_result(log, r)
        if asr.calib is not None and not had_calib:
            c = asr.calib
            msg = (f"Калибровка: сторона {c.side}, верх колена {c.knee_top:.0f}, "
                   f"ракурс {c.view_ratio:.2f} ({'ок' if c.view_ok else 'не боком'}), бедро/торс {c.ratio:.2f}, "
                   f"наклон камеры {c.roll:.1f}°")
            log.info(msg)
            print(msg)

        now = time.perf_counter()
        shown_fps = 0.9 * shown_fps + 0.1 / max(now - prev, 1e-6)
        prev = now

        raw = frame.copy()  # исходный кадр камеры для записи
        draw_skeleton(frame, res.kpts)
        frame = to_canvas(frame)
        draw_panel(frame, res.count, counter.state,
                   res.sample.smoothed if res.sample is not None else None,
                   shown_fps, now < flash_until)
        draw_assess(frame, asr, last_result)
        if rec.active:
            rec.write(frame if rec_overlay else raw)
            if int(rec.elapsed() * 2) % 2 == 0:
                cv2.circle(frame, (CANVAS_W - 150, 36), 11, (0, 0, 255), -1)
            cv2.putText(frame, f"REC {int(rec.elapsed()) // 60:02d}:{int(rec.elapsed()) % 60:02d}",
                        (CANVAS_W - 130, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1)
        if key in KEYS_QUIT or window_closed():
            break
        if key in KEYS_RESET:
            counter = MultiPersonRepCounter(config)
            calib = asr.calib
            asr = SquatAssessor(fps)
            asr.set_calibration(calib)
            last_result = None
            log.info("Счёт сброшен")
            print("Счёт сброшен")
        if key in KEYS_REC:
            if rec.active:
                path, n = rec.stop()
                msg = f"Запись остановлена: {path} ({n} кадров)"
            else:
                # по умолчанию пишем весь экран с оверлеем; с --rec-raw - исходный кадр камеры
                ok_rec = rec.start(frame if rec_overlay else raw)
                msg = f"Запись начата: {rec.path}" if ok_rec else "Не удалось начать запись"
            log.info(msg)
            print(msg)
        if key in KEYS_CALIB:
            asr.recalibrate()
            last_result = None
            log.info("Повторная калибровка")
            print("Калибровка: встаньте боком и стойте спокойно")
        if key in KEYS_FULLSCREEN:
            fullscreen = not fullscreen
            set_fullscreen(fullscreen, frame_size)

    cap.release()
    cv2.destroyAllWindows()
    if rec.active:
        path, n = rec.stop()
        log.info(f"Запись сохранена: {path} ({n} кадров)")
        print(f"Запись сохранена: {path}")
    for r in asr.flush(counter.frame_idx):
        log_result(log, r)
    sm = asr.summary()
    msg = (f"Итого: повторений {counter.count}, неполных {sm.partial}, не оценено {sm.not_evaluated}, "
           f"балл {sm.mean_score}, оценка {sm.rating}{' (предварительная)' if sm.preliminary else ''}")
    log.info(msg)
    print(msg)


if __name__ == "__main__":
    main()
