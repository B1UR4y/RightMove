"""Синтетические приседания (вид сбоку, человек смотрит вправо) для тестов."""
import numpy as np

THIGH, SHIN, TORSO = 200.0, 200.0, 230.0
ANKLE = np.array([400.0, 900.0])


def pose(knee_angle, lean, view=0.0, shin_tilt_k=0.3, conf=1.0):
    """Один кадр (17, 3). knee_angle - угол бедро-колено-лодыжка, lean - наклон корпуса от вертикали.
    view - расстояние между левым и правым плечом (и бедром) в долях торса."""
    s = np.radians(shin_tilt_k * (180 - knee_angle))
    knee = ANKLE + SHIN * np.array([np.sin(s), -np.cos(s)])
    u = (ANKLE - knee) / SHIN                       # колено -> лодыжка
    a = np.radians(knee_angle)
    # колено -> бедро: u, повёрнутый на knee_angle в сторону "назад и вверх" (-x)
    rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    hip = knee + THIGH * (rot @ u)
    if hip[0] > knee[0]:  # выбрать поворот, при котором таз позади колена
        rot = rot.T
        hip = knee + THIGH * (rot @ u)
    l = np.radians(lean)
    sh = hip + TORSO * np.array([np.sin(l), -np.cos(l)])
    k = np.zeros((17, 3))
    k[:, 2] = conf
    off = np.array([view * TORSO, 0.0])
    k[5, :2], k[6, :2] = sh, sh + off
    k[11, :2], k[12, :2] = hip, hip + off
    k[13, :2], k[14, :2] = knee, knee + off
    k[15, :2], k[16, :2] = ANKLE, ANKLE + off
    k[0, :2] = sh + np.array([40.0, -80.0])
    for i in (1, 2, 3, 4):
        k[i, :2] = k[0, :2]
    for i in (7, 8, 9, 10):
        k[i, :2] = sh
    return k


def smooth(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


def squat_frames(fps=30, top=170.0, bottom=55.0, lean_top=2.0, lean_bottom=22.0,
                 down_s=1.1, up_s=0.7, hold_s=0.2, hip_ahead=0.0, top_after=None,
                 stand_before_s=2.5, stand_after_s=2.5, view=0.0, n_reps=1, pause_s=1.0,
                 low_conf_bottom=False):
    """Серия приседаний. hip_ahead 0..1: насколько корпус отстаёт по выпрямлению на подъёме.
    top_after - угол колена, до которого человек выпрямляется (по умолчанию top).
    Возвращает список кадров (17, 3)."""
    top_after = top if top_after is None else top_after
    frames = [pose(top, lean_top, view) for _ in range(int(stand_before_s * fps))]
    for r in range(n_reps):
        nd, nu, nh = int(down_s * fps), max(2, int(up_s * fps)), int(hold_s * fps)
        for i in range(nd):
            f = smooth((i + 1) / nd)
            frames.append(pose(top + (bottom - top) * f, lean_top + (lean_bottom - lean_top) * f, view,
                               conf=0.2 if (low_conf_bottom and i >= nd - 4) else 1.0))
        for i in range(nh):
            frames.append(pose(bottom, lean_bottom, view, conf=0.2 if low_conf_bottom else 1.0))
        for i in range(nu):
            f = smooth((i + 1) / nu)
            knee = bottom + (top_after - bottom) * f
            # при опережении таза наклон на подъёме растёт, а не убывает
            lean = lean_top + (lean_bottom - lean_top) * (1 - f) + hip_ahead * 25.0 * np.sin(np.pi * f)
            frames.append(pose(knee, lean, view,
                               conf=0.2 if (low_conf_bottom and i < 3) else 1.0))
        frames += [pose(top_after, lean_top, view) for _ in range(int(pause_s * fps))]
    frames += [pose(top_after, lean_top, view) for _ in range(int(stand_after_s * fps))]
    return frames
