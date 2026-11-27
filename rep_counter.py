import math
from collections import deque
from dataclasses import dataclass

import numpy as np

# ============================================================
# СХЕМА ТОЧЕК
# Счётчик работает с COCO-17 (YOLO, RTMPose). Разметку Penn Action
# (13 точек) переводим в COCO через penn_to_coco().
# ============================================================
COCO = {
    "nose": 0,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

# Индексы COCO для точек Penn Action в их порядке:
# head, l_sh, r_sh, l_el, r_el, l_wr, r_wr, l_hip, r_hip, l_knee, r_knee, l_ank, r_ank
COCO_TO_PENN = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def penn_to_coco(x, y, vis):
    """Точки Penn Action одного кадра -> массив (17, 3) в формате COCO.
    Отсутствующие в Penn точки (глаза, уши) получают confidence 0."""
    kpts = np.zeros((17, 3), dtype=np.float64)
    for penn_idx, coco_idx in enumerate(COCO_TO_PENN):
        # float: в .mat координаты бывают uint16
        kpts[coco_idx] = (float(x[penn_idx]), float(y[penn_idx]), float(vis[penn_idx]))
    return kpts


# ============================================================
# КОНФИГ УПРАЖНЕНИЯ
# ============================================================
@dataclass
class ExerciseConfig:
    name: str
    # Тройки точек (a, b, c): угол считается в вершине b. Обычно левая и правая сторона.
    angle_triplets: tuple
    min_conf: float = 0.5   # порог confidence точки (README, раздел 13)
    median_window: int = 5  # медиана по последним N углам до EMA: убирает выбросы в 1–2 кадра
                            # (нога закрыта блином и т.п.). 1 — выключить. Задержка (N-1)/2 кадра
    ema_alpha: float = 0.5  # вес нового значения в EMA: больше -> быстрее реакция, меньше -> глаже
    max_gap: int = 10       # сколько кадров подряд держать последнее значение, если точек нет

    # Пороги автомата задаются относительно верхнего положения человека,
    # а не в абсолютных градусах: глубина приседа и ракурс у всех разные.
    start_drop: float = 12.0     # угол упал на столько ниже верха -> началось опускание
    min_amplitude: float = 30.0  # верх минус низ: меньше -> повторение не засчитывается
    min_attempt_amplitude: float = 15.0  # меньше этого размаха - помеха, события нет;
                                         # от этого размаха и до min_amplitude - попытка без зачёта
    rise_hyst: float = 8.0       # угол поднялся на столько выше низа -> начался подъём
    return_frac: float = 0.3     # повторение завершено, когда угол вернулся выше
                                 # верх - return_frac * амплитуда (т.е. на 70% пути наверх)
    min_rep_s: float = 0.5       # быстрее — скорее всего шум
    max_rep_s: float = 10.0      # дольше — человек застрял внизу или ушёл из кадра
    fps: float = 30.0            # Penn Action — 30 кадров/с
    check_hip_drop: bool = False


SQUAT = ExerciseConfig(
    name="squat",
    check_hip_drop=True,
    angle_triplets=(
        ("left_hip", "left_knee", "left_ankle"),
        ("right_hip", "right_knee", "right_ankle"),
    ),
)

EXERCISES = {"squat": SQUAT}


# ============================================================
# ШАГ 1. СИГНАЛ
# ============================================================
def joint_angle(a, b, c):
    """Угол ABC в градусах (в вершине b), 0..180."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return None
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


@dataclass
class SignalSample:
    frame_idx: int
    raw: float       # угол в этом кадре; nan, если точек не видно
    smoothed: float  # сглаженный угол; nan, пока данных не было или пропуск длиннее max_gap
    sides_used: int  # сколько сторон дали угол: 0, 1 или 2


class SignalExtractor:
    """Покадрово превращает точки в сглаженный угол сустава.

    Работает только с прошлыми кадрами, поэтому подходит для камеры
    в реальном времени. Пропуски не интерполируются (для этого нужны
    будущие кадры): последнее значение держится до max_gap кадров.
    """

    def __init__(self, config):
        self.config = config
        self.triplets = [tuple(COCO[name] for name in t) for t in config.angle_triplets]
        self.reset()

    def reset(self):
        self.frame_idx = -1
        self.smoothed = None
        self.gap = 0
        self.recent = deque(maxlen=self.config.median_window)

    def raw_angle(self, kpts):
        """Угол по видимым сторонам. Две стороны -> среднее, чтобы
        одно повторение не учитывалось дважды. (угол или None, число сторон)."""
        if kpts is None:
            return None, 0
        angles = []
        for ia, ib, ic in self.triplets:
            pts = kpts[[ia, ib, ic]]
            if np.any(pts[:, 2] < self.config.min_conf):
                continue
            ang = joint_angle(pts[0], pts[1], pts[2])
            if ang is not None:
                angles.append(ang)
        if not angles:
            return None, 0
        return float(np.mean(angles)), len(angles)

    def update(self, kpts):
        """kpts: массив (17, 3) [x, y, conf] в COCO или None, если человека нет."""
        self.frame_idx += 1
        angle, sides = self.raw_angle(kpts)

        if angle is None:
            self.gap += 1
            if self.gap > self.config.max_gap:
                self.smoothed = None
                self.recent.clear()
        else:
            self.gap = 0
            self.recent.append(angle)
            filtered = float(np.median(self.recent))
            if self.smoothed is None:
                self.smoothed = filtered
            else:
                a = self.config.ema_alpha
                self.smoothed = a * filtered + (1 - a) * self.smoothed

        return SignalSample(
            frame_idx=self.frame_idx,
            raw=math.nan if angle is None else angle,
            smoothed=math.nan if self.smoothed is None else self.smoothed,
            sides_used=sides,
        )


# ============================================================
# ШАГ 2. КОНЕЧНЫЙ АВТОМАТ
# ============================================================
UP = "UP"                # стоит, ждём начала движения
DESCENT = "DESCENT"      # опускается, но амплитуды ещё мало
BOTTOM = "BOTTOM"        # опустился достаточно глубоко, ждём подъёма
ASCENT = "ASCENT"        # поднимается


@dataclass
class RepEvent:
    rep_id: int                # номер засчитанного повторения; 0 для незасчитанных
    start_frame: int           # кадр, где был верх перед опусканием
    bottom_frame: int
    end_frame: int
    primary_signal_min: float  # угол в нижней точке
    primary_signal_max: float  # угол в верхней точке
    duration_ms: float
    valid: bool
    reason: str = ""           # почему не засчитано


class RepFSM:
    """Автомат ВЕРХ -> ОПУСКАНИЕ -> НИЗ -> ПОДЪЁМ -> ВЕРХ по сглаженному углу.

    Гистерезис: каждый переход требует сдвига угла на заметную величину,
    поэтому дрожание около одной точки не даёт ложных повторений.
    """

    def __init__(self, config):
        self.config = config
        self.pose_window = max(2, round(config.fps * 0.12))
        self.poses = deque(maxlen=math.ceil(config.max_rep_s * config.fps) + 2 * self.pose_window + 2)
        self.reset()
        self.count = 0

    def reset(self):
        self.poses.clear()
        self.state = UP
        self.top = None
        self.top_frame = None
        self.start_frame = None
        self.bottom = None
        self.bottom_frame = None

    def _back_to_up(self, angle, frame_idx):
        self.state = UP
        self.top = angle
        self.top_frame = frame_idx
        self.start_frame = frame_idx
        self.bottom = None
        self.bottom_frame = None

    def _has_hip_drop(self, frame_idx):
        window = self.pose_window
        upper = (self.start_frame - window + 1, self.start_frame)
        lower = (self.bottom_frame - window, min(frame_idx, self.bottom_frame + window))
        descent = []
        for side in (0, 1):
            top = [pose[side] for idx, pose in self.poses
                   if upper[0] <= idx <= upper[1] and pose[side] is not None]
            bottom = [pose[side] for idx, pose in self.poses
                      if lower[0] <= idx <= lower[1] and pose[side] is not None]
            if len(top) < 2 or len(bottom) < 2:
                continue
            top_y, top_gap, torso = np.median(top, axis=0)
            bottom_y, bottom_gap, _ = np.median(bottom, axis=0)
            descent.append(((bottom_y - top_y) / torso, (bottom_gap - top_gap) / torso))
        if not descent:
            return True
        # Ход таза измеряем в долях длины торса. Нижняя граница отсекает
        # покачивания, верхняя сохраняет возможность оценить неглубокий присед.
        amplitude = self.top - self.bottom
        minimum = max(0.09, min(0.20, (amplitude / 100.0) ** 2))
        return min(np.mean(descent, axis=0)) >= minimum

    def _event(self, frame_idx, valid, reason=""):
        if self.config.check_hip_drop and not self._has_hip_drop(frame_idx):
            return None
        if valid:
            self.count += 1
        return RepEvent(
            rep_id=self.count if valid else 0,
            start_frame=self.start_frame,
            bottom_frame=self.bottom_frame,
            end_frame=frame_idx,
            primary_signal_min=self.bottom,
            primary_signal_max=self.top,
            duration_ms=1000.0 * (frame_idx - self.start_frame) / self.config.fps,
            valid=valid,
            reason=reason,
        )

    def update(self, angle, frame_idx, pose=None):
        """angle: сглаженный угол или nan. Возвращает RepEvent, когда повторение завершилось."""
        cfg = self.config
        if pose is not None:
            self.poses.append((frame_idx, pose))
        if math.isnan(angle):
            # Точек не видно: автомат замирает в текущем состоянии и продолжает,
            # когда точки вернутся. Застрявшее повторение сбросит проверка max_rep_s.
            return None

        if self.top is None:
            self._back_to_up(angle, frame_idx)
            return None

        # Слишком долгое повторение: сбрасываемся, верхом считаем текущее положение
        if self.state != UP and frame_idx - self.start_frame > cfg.max_rep_s * cfg.fps:
            ev = None
            if self.bottom is not None and self.top - self.bottom >= cfg.min_attempt_amplitude:
                ev = self._event(frame_idx, valid=False, reason="сброс по таймеру")
            self._back_to_up(angle, frame_idx)
            return ev

        if self.state == UP:
            if angle > self.top:
                self.top = angle
                self.top_frame = frame_idx
            if angle >= self.top - cfg.start_drop / 2:
                # Последний кадр, когда человек ещё стоял: отсюда считаем длительность,
                # чтобы пауза перед повторением в неё не входила
                self.start_frame = frame_idx
            elif angle < self.top - cfg.start_drop:
                self.state = DESCENT
                self.bottom = angle
                self.bottom_frame = frame_idx

        elif self.state == DESCENT:
            if angle < self.bottom:
                self.bottom = angle
                self.bottom_frame = frame_idx
            if self.top - self.bottom >= cfg.min_amplitude:
                self.state = BOTTOM
            elif angle > self.top - cfg.start_drop / 2:
                # Вернулся наверх, не присев достаточно глубоко. Сообщаем только
                # о заметных попытках, мелкие покачивания и шум молча пропускаем.
                ev = None
                if self.top - self.bottom >= cfg.min_attempt_amplitude:
                    ev = self._event(frame_idx, valid=False, reason="мала амплитуда")
                self._back_to_up(angle, frame_idx)
                return ev

        elif self.state == BOTTOM:
            if angle < self.bottom:
                self.bottom = angle
                self.bottom_frame = frame_idx
            elif angle > self.bottom + cfg.rise_hyst:
                self.state = ASCENT

        elif self.state == ASCENT:
            if angle < self.bottom:
                # Снова пошёл вниз глубже прежнего низа
                self.state = BOTTOM
                self.bottom = angle
                self.bottom_frame = frame_idx
            elif angle >= self.top - cfg.return_frac * (self.top - self.bottom):
                duration_s = (frame_idx - self.start_frame) / cfg.fps
                if duration_s < cfg.min_rep_s:
                    ev = self._event(frame_idx, valid=False, reason="слишком быстро")
                else:
                    ev = self._event(frame_idx, valid=True)
                self._back_to_up(angle, frame_idx)
                return ev

        return None


class RepCounter:
    """Сигнал + автомат. На каждый кадр: counter.update(kpts) -> (SignalSample, RepEvent | None)."""

    def __init__(self, config):
        self.config = config
        self.signal = SignalExtractor(config)
        self.fsm = RepFSM(config)

    @property
    def count(self):
        return self.fsm.count

    @property
    def state(self):
        return self.fsm.state

    def update(self, kpts):
        sample = self.signal.update(kpts)
        pose = None
        if self.config.check_hip_drop and kpts is not None:
            pose = []
            for indices in ((5, 11, 15), (6, 12, 16)):
                points = kpts[list(indices)]
                if not np.isfinite(points).all() or min(points[:, 2]) < self.config.min_conf:
                    pose.append(None)
                    continue
                shoulder, hip, ankle = points[:, :2]
                torso = math.hypot(*(shoulder - hip))
                pose.append((hip[1], hip[1] - ankle[1], torso) if torso > 1 else None)
        event = self.fsm.update(sample.smoothed, sample.frame_idx, pose)
        return sample, event


# ============================================================
# НЕСКОЛЬКО ЛЮДЕЙ В КАДРЕ
# Раньше в каждом кадре брался самый крупный человек, и при нескольких
# людях скелет перескакивал между ними (видео 1842, 1738).
# Теперь каждого человека ведём от кадра к кадру и считаем ему повторения
# отдельно. Главный — тот, у кого сильнее всего менялся угол в колене:
# тренер или прохожий рядом не перебьют того, кто занимается.
# ============================================================
def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_intersection(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def box_iou(a, b):
    inter = box_intersection(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


class PersonTracker:
    """Привязывает рамки людей к постоянным id по пересечению с прошлым кадром."""

    def __init__(self, iou_thresh=0.3, max_missed=15):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed  # столько кадров подряд человека может не быть
        self.tracks = {}              # id -> {"box": [x1, y1, x2, y2], "missed": int}
        self.next_id = 1

    def update(self, boxes):
        """boxes: список рамок [x1, y1, x2, y2]. Возвращает id для каждой рамки."""
        pairs = sorted(
            ((box_iou(t["box"], b), tid, i)
             for tid, t in self.tracks.items() for i, b in enumerate(boxes)),
            reverse=True,
        )
        ids = [None] * len(boxes)
        used_tracks = set()
        for iou, tid, i in pairs:
            if iou < self.iou_thresh:
                break
            if tid in used_tracks or ids[i] is not None:
                continue
            ids[i] = tid
            used_tracks.add(tid)

        for i, b in enumerate(boxes):
            if ids[i] is None:
                ids[i] = self.next_id
                self.next_id += 1
                self.tracks[ids[i]] = {"box": b, "missed": 0}
            else:
                self.tracks[ids[i]]["box"] = b
                self.tracks[ids[i]]["missed"] = 0

        for tid in list(self.tracks):
            if tid not in ids:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]
        return ids


@dataclass
class FrameResult:
    track_id: object  # id показываемого человека или None
    kpts: object      # его точки в этом кадре или None
    sample: object    # SignalSample или None
    event: object     # RepEvent этого человека в этом кадре или None
    count: int


class MultiPersonRepCounter:
    """На каждый кадр: update(boxes, kpts_list) -> FrameResult главного человека."""

    def __init__(self, config, tracker=None, switch_margin=20.0,
                 duplicate_overlap=0.7, min_height_ratio=0.5):
        self.config = config
        self.tracker = tracker or PersonTracker()
        self.switch_margin = switch_margin        # градусов размаха угла, чтобы сменить главного
        self.duplicate_overlap = duplicate_overlap  # рамка лежит внутри другой на эту долю -> дубль
        self.min_height_ratio = min_height_ratio  # главным может быть только человек не ниже этой
                                                  # доли самого высокого: люди на заднем плане не в счёт
        self.heights = {}   # id -> [сумма высот рамок, число кадров]
        self.counters = {}  # id -> RepCounter
        self.ranges = {}    # id -> [мин, макс] сглаженного угла
        self.main_id = None
        self.frame_idx = -1
        # Общий счёт: повторения, завершённые тем, кто в этот момент был главным.
        # Если трек человека оборвался (вышел из кадра и вернулся), счёт не теряется.
        self.total = 0

    @property
    def count(self):
        return self.total

    @property
    def state(self):
        return self.counters[self.main_id].state if self.main_id is not None else UP

    def _score(self, tid):
        """Размах угла: главный — тот, кто сильнее всех сгибает колени.
        По числу повторений не выбираем: иначе человек рядом, закончивший
        присед на пару кадров раньше, перехватывал бы счёт у занимающегося."""
        lo, hi = self.ranges[tid]
        return hi - lo if lo is not None else 0.0

    def drop_duplicates(self, boxes, kpts_list):
        """YOLO иногда даёт второй, частичный скелет того же человека (IoU 0.6–0.7
        проходит её NMS). Дубль получал новый id и посреди повторения забирал
        человека себе — повторение терялось. Оставляем большую рамку, если
        меньшая почти целиком в ней."""
        order = sorted(range(len(boxes)), key=lambda i: -box_area(boxes[i]))
        kept = []
        for i in order:
            b = boxes[i]
            if all(box_intersection(b, boxes[j]) < self.duplicate_overlap * box_area(b) for j in kept):
                kept.append(i)
        kept.sort()
        return [boxes[i] for i in kept], [kpts_list[i] for i in kept]

    def update(self, boxes, kpts_list):
        self.frame_idx += 1
        if self.duplicate_overlap is not None:
            boxes, kpts_list = self.drop_duplicates(boxes, kpts_list)
        ids = self.tracker.update(boxes)
        seen = dict(zip(ids, kpts_list))
        for tid, b in zip(ids, boxes):
            h = self.heights.setdefault(tid, [0.0, 0])
            h[0] += b[3] - b[1]
            h[1] += 1

        results = {}
        for tid in set(self.counters) | set(ids):
            if tid not in self.counters:
                self.counters[tid] = RepCounter(self.config)
                self.ranges[tid] = [None, None]
                # Новый человек догоняет общий номер кадра пропусками,
                # чтобы номера кадров в его RepEvent совпадали с видео
                for _ in range(self.frame_idx):
                    self.counters[tid].update(None)
            kpts = seen.get(tid)
            sample, event = self.counters[tid].update(kpts)
            if not math.isnan(sample.smoothed):
                r = self.ranges[tid]
                r[0] = sample.smoothed if r[0] is None else min(r[0], sample.smoothed)
                r[1] = sample.smoothed if r[1] is None else max(r[1], sample.smoothed)
            results[tid] = (kpts, sample, event)

        # Повторение засчитывается, если его завершил тот, кто был главным
        # до этого кадра. Иначе человек рядом, закончивший присед раньше,
        # добавил бы лишнее повторение в момент переключения.
        prev_main = self.main_id
        main_event = results[prev_main][2] if prev_main in results else None
        if main_event is not None and main_event.valid:
            self.total += 1

        # Главным может быть только тот, кто сейчас в кадре (трек жив)
        alive = [tid for tid in self.counters if tid in self.tracker.tracks]
        if alive:
            mean_h = {tid: self.heights[tid][0] / self.heights[tid][1]
                      for tid in alive if self.heights.get(tid, [0, 0])[1]}
            tallest = max(mean_h.values(), default=0.0)
            eligible = [tid for tid in alive
                        if mean_h.get(tid, 0.0) >= self.min_height_ratio * tallest]
            best = max(eligible or alive, key=self._score)
            if self.main_id is None:
                self.main_id = best
            elif best != self.main_id:
                # Переключаемся, только если у нового размах угла заметно больше,
                # иначе картинка дёргается между людьми, которые двигаются похоже
                main_ok = self.main_id in eligible
                if not main_ok or self._score(best) > self._score(self.main_id) + self.switch_margin:
                    self.main_id = best

        # Ушедших людей забываем, чтобы при долгой работе с камерой не копились счётчики
        for tid in list(self.counters):
            if tid not in self.tracker.tracks and tid != self.main_id:
                del self.counters[tid], self.ranges[tid]
                self.heights.pop(tid, None)

        if self.main_id is None:
            return FrameResult(None, None, None, None, 0)
        kpts, sample, _ = results[self.main_id]
        # Событие — от того, кто был главным до кадра (именно оно попало в счёт)
        return FrameResult(self.main_id, kpts, sample, main_event, self.count)
