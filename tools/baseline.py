"""Прогон счётчика по сохранённым точкам. Печатает события и пишет JSON.
Запуск: python tools/baseline.py [файл_результата.json]"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rep_counter import SQUAT, RepCounter
import dataclasses


def run(path):
    d = np.load(path)
    cfg = dataclasses.replace(SQUAT, fps=float(d["fps"]))
    c = RepCounter(cfg)
    evs = []
    for k in d["kpts"]:
        kk = None if np.isnan(k).any() else k
        _, e = c.update(kk)
        if e is not None:
            evs.append(dict(valid=e.valid, reason=e.reason, start=e.start_frame,
                            bottom=e.bottom_frame, end=e.end_frame,
                            mn=round(e.primary_signal_min, 1), mx=round(e.primary_signal_max, 1)))
    return c.count, evs


if __name__ == "__main__":
    out = {}
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    for p in sorted(glob.glob(os.path.join(root, "keypoints", "*.npz")) + glob.glob(os.path.join(root, "penn", "*.npz"))):
        n, evs = run(p)
        name = os.path.basename(p)[:-4]
        out[name] = dict(count=n, events=evs)
        bad = [e["reason"] for e in evs if not e["valid"]]
        print(f"{name}: засчитано {n}, незасчитанных {len(bad)} {bad}")
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
