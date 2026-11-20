"""Прогон счётчика и оценщика по сохранённым точкам (.npz).
Запуск: python tools/run_assess.py [файл.npz ...]   (по умолчанию data/keypoints/*.npz)"""
import dataclasses, glob, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rep_counter import SQUAT, RepCounter
from squat_assess import SquatAssessor, Thresholds


def run(path, th=None, verbose=True, fallback=None):
    d = np.load(path)
    fps = float(d["fps"])
    counter = RepCounter(dataclasses.replace(SQUAT, fps=fps))
    asr = SquatAssessor(fps, th or Thresholds())
    out = []
    if fallback is not None:
        asr.set_calibration(fallback)
    for i, k in enumerate(d["kpts"]):
        kk = None if np.isnan(k).any() else k
        sample, ev = counter.update(kk)
        out += asr.update(sample.frame_idx, kk, ev, counter.state)
    out += asr.flush(len(d["kpts"]) - 1)
    if verbose:
        c = asr.calib
        print(f"== {os.path.basename(path)}  счётчик {counter.count}", end="")
        print("  калибровка: нет" if c is None else
              f"  сторона {c.side}, верх {c.knee_top:.0f}, ракурс {c.view_ratio:.2f}, бедро/торс {c.ratio:.2f}")
        for r in out:
            f = {k: (None if v is None else round(v, 1)) for k, v in r.features.items()}
            print("  ", r.summary_line(), f)
        s = asr.summary()
        print(f"   итог: повторений {s.reps}, неполных {s.partial}, не оценено {s.not_evaluated}, "
              f"балл {s.mean_score}, оценка {s.rating}{' (предв.)' if s.preliminary else ''}")
    return counter, asr, out


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "data", "keypoints", "*.npz")))
    # Запись без своей стойки (07) берёт калибровку из 01_stoyka
    fb = run(os.path.join(ROOT, "data", "keypoints", "01_stoyka.npz"), verbose=False)[1].calib
    for p in files:
        _, a, _ = run(p, verbose=False)
        run(p, fallback=None if a.calib is not None else fb)
