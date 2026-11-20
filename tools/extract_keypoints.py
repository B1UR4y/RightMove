"""Извлечение точек COCO-17 (YOLOv8n-pose) из видео в файлы .npz.

Запуск: python tools/extract_keypoints.py [папка_видео] [папка_результата]
По умолчанию: recordings/cut -> data/keypoints
В кадре выбирается человек с наибольшей рамкой. Если людей нет - NaN.
Файл .npz: kpts (N,17,3: x, y, уверенность), boxes (N,4), fps, width, height.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def extract(model, video, imgsz=640):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    kpts, boxes = [], []
    w = h = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        r = model(frame, imgsz=imgsz, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            kpts.append(np.full((17, 3), np.nan, np.float32))
            boxes.append(np.full(4, np.nan, np.float32))
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        i = int(np.argmax(area))
        xy = r.keypoints.xy.cpu().numpy()[i]
        cf = r.keypoints.conf.cpu().numpy()[i]
        kpts.append(np.concatenate([xy, cf[:, None]], axis=1).astype(np.float32))
        boxes.append(xyxy[i].astype(np.float32))
    cap.release()
    return (np.array(kpts), np.array(boxes), float(fps), w, h)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("recordings/cut")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/keypoints")
    dst.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n-pose.pt")
    for v in sorted(src.glob("*.mp4")):
        k, b, fps, w, h = extract(model, v)
        np.savez_compressed(dst / (v.stem + ".npz"), kpts=k, boxes=b,
                            fps=fps, width=w, height=h)
        found = int(np.sum(~np.isnan(b[:, 0])))
        print(f"{v.name}: кадров {len(k)}, человек найден в {found}")


if __name__ == "__main__":
    main()
