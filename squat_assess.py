"""Оценка техники приседания: подходы A (правила на признаках) и D (персональная нормировка).

Работает поверх счётчика rep_counter.py и получает те же точки COCO-17.
Порядок работы:
    1. Calibrator - человек стоит боком 2-3 с, получаем Calibration.
    2. SquatAssessor.update(...) на каждый кадр вместе с событием счётчика.
       Когда попытка оценена, возвращается AttemptResult.
Все пороги собраны в Thresholds. Это стартовые значения, их уточняют по записям.
Описание: Metodika_otsenki_pravilnosti_prisedaniy.docx.
"""
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# Исходы правила
NORM, WARN, ERR, NA = "норма", "предупреждение", "ошибка", "не оценено"
# Категории попытки
CAT_REP, CAT_PARTIAL, CAT_NA = "Повторение", "Неполный присед", "Не оценено"

SH, HIP, KNEE, ANK = 0, 1, 2, 3
SIDES = {"left": (5, 11, 13, 15), "right": (6, 12, 14, 16)}


@dataclass
class Thresholds:
    min_conf: float = 0.5
    window_half: int = 3          # окно нижней точки: плюс-минус столько кадров
    min_valid_frac: float = 0.5   # доля кадров окна с хорошей уверенностью
    # E1
    f2_warn: float = 15.0
    f2_err: float = 25.0
    f1_warn: float = 70.0
    f1_err: float = 85.0
    # E2
    f3_warn: float = 35.0
    f3_err: float = 45.0
    ratio_ref: float = 0.9        # типичное отношение бедро/торс
    ratio_k: float = 30.0         # поправка: град. на единицу отношения
    # E3
    f7_warn: float = 1.25
    f7_err: float = 1.5
    f7_frames: int = 3
    f9_warn: float = 2.0          # рост наклона корпуса к середине подъёма, град.
    f9_err: float = 6.0
    f7_min_rise: float = 0.1      # подъём плеча меньше доли бедра - признак не считается
    # E4
    f5_warn: float = 6.0
    f5_err: float = 15.0
    f5_window_s: float = 1.0
    # баллы
    pen_err: int = 25
    pen_warn: int = 10
    # калибровка и ракурс
    calib_seconds: float = 2.0
    view_max: float = 0.2         # ширина плеч (бёдер) / длина торса
    roll_max: float = 10.0        # наклон камеры больше этого не корректируем
    still_max: float = 0.10       # колебание таза в калибровке, доли торса


# ============================================================
# ГЕОМЕТРИЯ
# ============================================================
def angle3(a, b, c):
    """Угол abc в градусах, 0..180. Работает с массивами (..., 2)."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    cs = (v1 * v2).sum(-1) / np.where(n == 0, np.nan, n)
    return np.degrees(np.arccos(np.clip(cs, -1, 1)))


def thigh_tilt(hip, knee):
    """F2: угол линии колено-бедро к горизонтали. 90 - вертикально,
    0 - параллельно полу, меньше 0 - таз ниже колена (ось y кадра направлена вниз)."""
    dx = np.abs(hip[..., 0] - knee[..., 0])
    return np.degrees(np.arctan2(knee[..., 1] - hip[..., 1], dx))


def trunk_lean(hip, sh):
    """F3: угол линии бедро-плечо к вертикали, 0..180."""
    d = sh - hip
    return np.degrees(np.arctan2(np.abs(d[..., 0]), -d[..., 1]))


def zone(value, warn, err):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return NA
    if value > err:
        return ERR
    if value > warn:
        return WARN
    return NORM


# ============================================================
# КАЛИБРОВКА (подход D)
# ============================================================
@dataclass
class Calibration:
    side: str
    thigh: float          # длина бедра, пикселей
    shin: float
    torso: float
    knee_top: float       # личный "верх": угол колена стоя
    lean_top: float       # наклон корпуса стоя
    view_ratio: float     # ширина плеч (бёдер) / торс
    view_ok: bool
    ratio: float          # бедро / торс
    roll: float = 0.0     # наклон кадра (камеры), град.: линия лодыжка-плечо стоя считается вертикалью

    @property
    def idx(self):
        return SIDES[self.side]


class Calibrator:
    """Копит кадры стоя. update() возвращает Calibration, когда накоплено calib_seconds
    спокойного стояния. Если человек двигается или точки не видны - накопление
    начинается заново."""

    def __init__(self, fps, th=None):
        self.th = th or Thresholds()
        self.need = int(round(self.th.calib_seconds * fps))
        self.frames = []
        self.result = None

    @property
    def progress(self):
        return min(1.0, len(self.frames) / self.need)

    def reset(self):
        self.frames = []
        self.result = None

    def _good(self, k):
        if k is None or np.isnan(k).any():
            return False
        return any(np.all(k[list(ix), 2] >= self.th.min_conf) for ix in SIDES.values())

    def update(self, kpts):
        if self.result is not None:
            return self.result
        if not self._good(kpts):
            self.frames = []
            return None
        self.frames.append(np.asarray(kpts, dtype=float))
        if len(self.frames) >= 3:
            arr = np.array(self.frames)
            side = self._side(arr)
            ix = SIDES[side]
            hip_y = running_median(arr[:, ix[HIP], 1], 9)  # сглаживаем шум точек
            torso = np.linalg.norm(np.median(arr[:, ix[SH], :2], 0) - np.median(arr[:, ix[HIP], :2], 0))
            moved = (hip_y.max() - hip_y.min()) > self.th.still_max * max(torso, 1.0)
            if moved:
                self.frames = [self.frames[-1]]
                return None
        if len(self.frames) < self.need:
            return None
        self.result = self._build(np.array(self.frames))
        return self.result

    @staticmethod
    def _side(arr):
        score = {s: arr[:, list(ix), 2].mean() for s, ix in SIDES.items()}
        return max(score, key=score.get)

    def _build(self, arr):
        side = self._side(arr)
        ix = SIDES[side]
        med = {n: np.median(arr[:, ix[n], :2], 0) for n in (SH, HIP, KNEE, ANK)}
        torso = float(np.linalg.norm(med[SH] - med[HIP]))
        thigh = float(np.linalg.norm(med[HIP] - med[KNEE]))
        shin = float(np.linalg.norm(med[KNEE] - med[ANK]))
        knee_top = float(angle3(med[HIP], med[KNEE], med[ANK]))
        lean = float(trunk_lean(med[HIP], med[SH]))
        shw = float(np.median(np.abs(arr[:, 5, 0] - arr[:, 6, 0])))
        hw = float(np.median(np.abs(arr[:, 11, 0] - arr[:, 12, 0])))
        view = max(shw, hw) / torso if torso > 0 else 9.9
        # Наклон камеры: стоя линия лодыжка-плечо вертикальна, отклонение - наклон кадра
        v = med[SH] - med[ANK]
        roll = float(np.degrees(np.arctan2(v[0], -v[1])))
        if abs(roll) > self.th.roll_max:
            roll = 0.0
        return Calibration(side, thigh, shin, torso, knee_top, lean, view,
                           view <= self.th.view_max, thigh / torso if torso > 0 else 0.0, roll)


# ============================================================
# РЕЗУЛЬТАТЫ
# ============================================================
@dataclass
class AttemptResult:
    attempt_id: int
    category: str
    start_frame: int
    bottom_frame: int
    end_frame: int
    rules: dict = field(default_factory=dict)      # "E1".."E4" -> исход
    features: dict = field(default_factory=dict)   # "F1".. -> число или None
    score: object = None                           # 0..100 или None, если не оценено
    note: str = ""

    @property
    def counted(self):
        return self.category == CAT_REP

    @property
    def evaluated(self):
        return self.category != CAT_NA

    def summary_line(self):
        parts = [f"{r}:{o}" for r, o in self.rules.items() if o != NORM]
        s = "" if self.score is None else f" балл {self.score}"
        return f"попытка {self.attempt_id}: {self.category}{s}" + (" [" + ", ".join(parts) + "]" if parts else "") + \
            (f" ({self.note})" if self.note else "")


def score_from_rules(rules, th):
    s = 100
    for o in rules.values():
        s -= th.pen_err if o == ERR else th.pen_warn if o == WARN else 0
    return max(0, s)


def rating_from_mean(mean):
    return 5 if mean >= 90 else 4 if mean >= 75 else 3 if mean >= 60 else 2 if mean >= 40 else 1


@dataclass
class SetSummary:
    reps: int            # засчитанные повторения
    partial: int         # неполные приседы
    not_evaluated: int
    mean_score: object
    rating: object       # 1-5 или None
    preliminary: bool    # не оценена половина попыток и более


def summarize(results):
    ev = [r for r in results if r.evaluated and r.score is not None]
    n_na = sum(1 for r in results if not r.evaluated)
    mean = sum(r.score for r in ev) / len(ev) if ev else None
    total = len(results)
    return SetSummary(
        reps=sum(1 for r in results if r.counted),
        partial=sum(1 for r in results if r.category == CAT_PARTIAL),
        not_evaluated=n_na, mean_score=mean,
        rating=rating_from_mean(mean) if mean is not None else None,
        preliminary=total > 0 and n_na * 2 >= total)


# ============================================================
# ПРИЗНАКИ
# ============================================================
class FrameBuffer:
    """Хранит точки последних кадров по номеру кадра."""

    def __init__(self, maxlen=600):
        self.d = {}
        self.order = deque()
        self.maxlen = maxlen

    def add(self, idx, kpts):
        self.d[idx] = None if kpts is None else np.asarray(kpts, dtype=float)
        self.order.append(idx)
        while len(self.order) > self.maxlen:
            self.d.pop(self.order.popleft(), None)

    def get(self, idx):
        return self.d.get(idx)

    def has(self, idx):
        return idx in self.d


def side_points(buf, frames, calib, min_conf):
    """Массив (n, 4, 2) точек ближней ноги и маска кадров, где все 4 точки уверены."""
    pts = np.full((len(frames), 4, 2), np.nan)
    ok = np.zeros(len(frames), dtype=bool)
    ix = list(calib.idx)
    for i, f in enumerate(frames):
        k = buf.get(f)
        if k is None or np.isnan(k[ix]).any():
            continue
        if np.all(k[ix, 2] >= min_conf):
            pts[i] = k[ix, :2]
            ok[i] = True
    if calib.roll:
        a = np.radians(calib.roll)  # поворачиваем точки так, чтобы стойка стала вертикальной
        c, sn = np.cos(a), np.sin(a)
        x, y = pts[..., 0].copy(), pts[..., 1].copy()
        pts[..., 0] = c * x + sn * y
        pts[..., 1] = -sn * x + c * y
    return pts, ok


def window_feature(buf, center, calib, th, fn):
    """Медиана признака fn(pts) по окну center +- window_half. None, если
    уверенных кадров меньше min_valid_frac."""
    frames = list(range(center - th.window_half, center + th.window_half + 1))
    pts, ok = side_points(buf, frames, calib, th.min_conf)
    if ok.sum() < th.min_valid_frac * len(frames) or ok.sum() == 0:
        return None
    return float(np.median(fn(pts[ok])))


def f1_fn(p): return angle3(p[:, HIP], p[:, KNEE], p[:, ANK])
def f2_fn(p): return thigh_tilt(p[:, HIP], p[:, KNEE])
def f3_fn(p): return trunk_lean(p[:, HIP], p[:, SH])


def running_median(x, w=5):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    h = w // 2
    for i in range(len(x)):
        seg = x[max(0, i - h): i + h + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = np.median(seg)
    return out


def feature_f5(buf, end_frame, last_frame, calib, th, fps):
    """Максимум угла колена после кадра окончания. Возвращает (разница с личным верхом, максимум)."""
    frames = list(range(end_frame + 1, last_frame + 1))
    if len(frames) < 3:
        return None, None
    pts, ok = side_points(buf, frames, calib, th.min_conf)
    if ok.sum() < th.min_valid_frac * len(frames):
        return None, None
    ang = np.full(len(frames), np.nan)
    ang[ok] = f1_fn(pts[ok])
    ang = running_median(ang, 5)
    if np.all(np.isnan(ang)):
        return None, None
    mx = float(np.nanmax(ang))
    return calib.knee_top - mx, mx


def feature_f7(buf, bottom, end, top_angle, bottom_angle, calib, th):
    """Отношение подъёма таза к подъёму плеча в первой половине подъёма.
    Возвращает (F7 медиана или None, исход по числу кадров подряд)."""
    frames = list(range(bottom, end + 1))
    pts, ok = side_points(buf, frames, calib, th.min_conf)
    if ok.sum() < 3:
        return None, NA
    ang = np.full(len(frames), np.nan)
    ang[ok] = f1_fn(pts[ok])
    ang = running_median(ang, 3)
    half_level = bottom_angle + 0.5 * (top_angle - bottom_angle)
    stop = len(frames)
    for i, a in enumerate(ang):
        if not np.isnan(a) and a >= half_level:
            stop = i + 1
            break
    hy = running_median(pts[:, HIP, 1], 3)
    sy = running_median(pts[:, SH, 1], 3)
    ratios = []
    for i in range(1, stop):
        if np.isnan(hy[i]) or np.isnan(sy[i]) or np.isnan(hy[0]) or np.isnan(sy[0]):
            ratios.append(np.nan)
            continue
        hip_rise, sh_rise = hy[0] - hy[i], sy[0] - sy[i]
        if sh_rise < th.f7_min_rise * calib.thigh:
            ratios.append(np.nan)
        else:
            ratios.append(hip_rise / sh_rise)
    arr = np.array(ratios, dtype=float)
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return None, NA
    f7 = float(np.nanmedian(arr))

    def run_over(level):
        run = best = 0
        for v in arr:
            run = run + 1 if (not np.isnan(v) and v > level) else 0
            best = max(best, run)
        return best >= th.f7_frames

    outcome = ERR if run_over(th.f7_err) else WARN if f7 > th.f7_warn else NORM
    return f7, outcome


def feature_f9(buf, bottom, end, top_angle, bottom_angle, calib, th):
    """Рост наклона корпуса от нижней точки до середины подъёма (град.).
    При нормальном подъёме корпус выпрямляется (значение <= 0); если таз
    опережает, наклон растёт. Надёжнее F7: не зависит от малого подъёма плеча."""
    frames = list(range(bottom, end + 1))
    pts, ok = side_points(buf, frames, calib, th.min_conf)
    if ok.sum() < 3:
        return None
    ang = np.full(len(frames), np.nan)
    ang[ok] = f1_fn(pts[ok])
    ang = running_median(ang, 3)
    lean = np.full(len(frames), np.nan)
    lean[ok] = f3_fn(pts[ok])
    lean = running_median(lean, 5)
    half_level = bottom_angle + 0.5 * (top_angle - bottom_angle)
    idx = [i for i, a in enumerate(ang) if not np.isnan(a) and a >= half_level]
    i = idx[0] if idx else len(frames) - 1
    if np.isnan(lean[0]) or np.isnan(lean[i]):
        return None
    return float(lean[i] - lean[0])


# ============================================================
# ОЦЕНКА ПОПЫТКИ
# ============================================================
def rule_e1(f1, f2, th):
    if f1 is None or f2 is None:
        return NA
    z1, z2 = zone(f1, th.f1_warn, th.f1_err), zone(f2, th.f2_warn, th.f2_err)
    if z1 == ERR and z2 == ERR:
        return ERR
    if z1 == NORM and z2 == NORM:
        return NORM
    return WARN


def assess_attempt(buf, ev, last_frame, calib, th, fps, attempt_id):
    """Оценивает одну попытку по событию счётчика ev (RepEvent) и накопленным точкам."""
    res = AttemptResult(attempt_id, CAT_NA, ev.start_frame, ev.bottom_frame, ev.end_frame)
    if calib is None:
        res.note = "нет калибровки"
        return res
    if not calib.view_ok:
        res.note = "ракурс: встаньте боком"
        return res
    if ev.reason in ("слишком быстро", "сброс по таймеру"):
        res.note = ev.reason
        return res

    # Точки должны быть видны хотя бы в половине кадров самой попытки: иначе
    # счётчик мог найти нижнюю точку уже после пропажи точек
    _, ok_all = side_points(buf, range(ev.start_frame, ev.end_frame + 1), calib, th.min_conf)
    if ok_all.mean() < th.min_valid_frac:
        res.note = "низкая уверенность точек"
        return res

    f1 = window_feature(buf, ev.bottom_frame, calib, th, f1_fn)
    f2 = window_feature(buf, ev.bottom_frame, calib, th, f2_fn)
    res.features.update(F1=f1, F2=f2)
    if f1 is None or f2 is None:
        res.note = "низкая уверенность точек"
        return res

    if not ev.valid:  # размах мал, счётчик не засчитал: ошибка E1 по определению
        if rule_e1(f1, f2, th) == NORM:
            # Глубина достаточна, а размах мал: счётчик взял неверный "верх"
            # (например, после сброса по таймеру). Это не неполный присед.
            res.note = "размах счётчика мал при достаточной глубине"
            return res
        res.category = CAT_PARTIAL
        res.rules = {"E1": ERR}
        res.score = score_from_rules(res.rules, th)
        res.note = "размах меньше порога счётчика"
        return res

    res.category = CAT_REP
    rules = {"E1": rule_e1(f1, f2, th)}

    f3 = window_feature(buf, ev.bottom_frame, calib, th, f3_fn)
    f3c = None if f3 is None else f3 - th.ratio_k * (calib.ratio - th.ratio_ref)
    frames = list(range(ev.start_frame, ev.end_frame + 1))
    pts, ok = side_points(buf, frames, calib, th.min_conf)
    f3max = float(np.nanmax(f3_fn(pts[ok]))) if ok.any() else None
    f3_start = window_feature(buf, ev.start_frame, calib, th, f3_fn)
    res.features.update(F3=f3, F3c=f3c, F3max=f3max,
                        F4=None if (f3 is None) else f3 - (calib.lean_top if f3_start is None else f3_start))
    rules["E2"] = zone(f3c, th.f3_warn, th.f3_err)

    f7, o7 = feature_f7(buf, ev.bottom_frame, ev.end_frame, calib.knee_top,
                        ev.primary_signal_min, calib, th)
    f9 = feature_f9(buf, ev.bottom_frame, ev.end_frame, calib.knee_top, ev.primary_signal_min, calib, th)
    res.features["F7"] = f7
    res.features["F9"] = f9
    o9 = zone(f9, th.f9_warn, th.f9_err)
    known = [o for o in (o7, o9) if o != NA]
    rules["E3"] = ERR if ERR in known else WARN if WARN in known else NORM if known else NA

    f5, f5max = feature_f5(buf, ev.end_frame, last_frame, calib, th, fps)
    res.features["F5"] = f5
    rules["E4"] = zone(f5, th.f5_warn, th.f5_err)

    res.features["F6_down_s"] = (ev.bottom_frame - ev.start_frame) / fps
    res.features["F6_up_s"] = (ev.end_frame - ev.bottom_frame) / fps
    res.rules = rules
    res.score = score_from_rules(rules, th)
    return res


class SquatAssessor:
    """Потоковый оценщик. На каждый кадр: update(frame_idx, kpts, event, fsm_state)
    -> список готовых AttemptResult (обычно пустой).

    Попытка, засчитанная счётчиком, оценивается после окна F5 (до 1 с после кадра
    окончания или до начала следующего спуска). Остальные - сразу."""

    def __init__(self, fps=30.0, th=None):
        self.fps = fps
        self.th = th or Thresholds()
        self.buf = FrameBuffer(int(15 * fps) + 60)
        self.calibrator = Calibrator(fps, self.th)
        self.calib = None
        self.pending = []
        self.results = []
        self.next_id = 1

    def set_calibration(self, calib):
        """Взять готовую калибровку (например, сохранённую раньше)."""
        self.calib = calib

    def recalibrate(self):
        self.calibrator = Calibrator(self.fps, self.th)
        self.calib = None
        self.pending = []

    def update(self, frame_idx, kpts, event=None, fsm_state="UP"):
        self.buf.add(frame_idx, kpts)
        done = []
        if self.calib is None:
            self.calib = self.calibrator.update(kpts)
            return done  # во время калибровки попытки не оцениваем
        if event is not None:
            if event.valid:
                self.pending.append(event)
            else:
                done.append(self._finish(event, frame_idx))
        keep = []
        for ev in self.pending:
            window_end = ev.end_frame + int(self.th.f5_window_s * self.fps)
            if frame_idx >= window_end or (frame_idx > ev.end_frame and fsm_state in ("DESCENT", "BOTTOM")):
                done.append(self._finish(ev, frame_idx))
            else:
                keep.append(ev)
        self.pending = keep
        return done

    def flush(self, last_frame):
        """Конец записи: оценить ожидающие попытки."""
        done = [self._finish(ev, last_frame) for ev in self.pending]
        self.pending = []
        return done

    def _finish(self, ev, frame_idx):
        r = assess_attempt(self.buf, ev, frame_idx, self.calib, self.th, self.fps, self.next_id)
        self.next_id += 1
        self.results.append(r)
        return r

    def summary(self):
        return summarize(self.results)
