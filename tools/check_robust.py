"""Проверка устойчивости оценки: шум в точках и ракурс. Запуск: python tools/check_robust.py"""
import dataclasses, glob, os, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rep_counter import SQUAT, RepCounter
from squat_assess import SquatAssessor, Thresholds, ERR, WARN, NORM


def attempts(kpts, fps, th, calib=None):
    c = RepCounter(dataclasses.replace(SQUAT, fps=fps))
    a = SquatAssessor(fps, th)
    if calib:
        a.set_calibration(calib)
    out = []
    for k in kpts:
        s, ev = c.update(k)
        out += a.update(s.frame_idx, k, ev, c.state)
    return out + a.flush(len(kpts) - 1), a


def worst(r):
    o = list(r.rules.values())
    return ERR if ERR in o else WARN if WARN in o else NORM


def main():
    kp = {os.path.basename(p)[:-4]: np.load(p) for p in sorted(glob.glob(os.path.join(ROOT, "data", "keypoints", "*.npz")))}
    th = Thresholds()
    fb = attempts(kp["01_stoyka"]["kpts"], 30.0, th)[1].calib
    print("Шум в координатах точек (пиксели, СКО), доля попыток с ошибкой в правильных приседаниях 02-04:")
    for sigma in (0, 2, 4, 8):
        errs = tot = 0
        rng = np.random.default_rng(1)
        for n in ("02_pravilno_1", "03_pravilno_2", "04_pravilno_3"):
            k = kp[n]["kpts"].copy()
            k[:, :, :2] += rng.normal(0, sigma, k[:, :, :2].shape)
            out, _ = attempts(k, 30.0, th)
            tot += len(out)
            errs += sum(worst(r) == ERR for r in out)
        print(f"  sigma {sigma}: ложных ошибок {errs} из {tot}")
    print("Ракурс 30 и 60 град. с отключённой проверкой ракурса (только для справки):")
    th2 = dataclasses.replace(th, view_max=9.0)
    for n in ("09_ugol30", "10_ugol60"):
        out, _ = attempts(kp[n]["kpts"], 30.0, th2)
        print(f"  {n}:", [(r.category, worst(r)) for r in out])


if __name__ == "__main__":
    main()
