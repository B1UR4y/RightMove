"""Run from the SquatCounter directory using its CUDA-enabled Python."""
import argparse
from collections import Counter, defaultdict
import ctypes
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,
        17, 18, 19, 21, 20, 23, 22, 25, 24]


def label(person, width, height, counts):
    points = person['keypoints'][:78]
    if len(points) != 78:
        raise ValueError(f"Invalid keypoints for annotation {person.get('id')}")
    x, y, w, h = person['bbox']
    left, top = max(0, x), max(0, y)
    right, bottom = min(width, x+w), min(height, y+h)
    if left >= right or top >= bottom:
        counts['invalid_box'] += 1
        return None
    values = [0, (left+right)/(2*width), (top+bottom)/(2*height),
              (right-left)/width, (bottom-top)/height]
    marked = 0
    for i in range(26):
        px, py, v = points[3*i:3*i+3]
        if v > 0 and math.isfinite(px) and math.isfinite(py) and 0 <= px <= width and 0 <= py <= height:
            values.extend([px/width, py/height, min(2, v)])
            marked += 1
            if i >= 20:
                counts['foot_points'] += 1
        else:
            values.extend([0, 0, 0])
    if not marked:
        counts['no_body_points'] += 1
        return None
    return ' '.join(format(n, '.8g') for n in values)


def convert(raw, stage, split):
    from PIL import Image

    doc = json.loads((raw / f'halpe_{split}_v1.json').read_bytes())
    base = raw / ('hico_20160224_det' if split == 'train' else 'val2017')
    sources = {}
    for path in base.rglob('*.jpg'):
        if path.name in sources:
            raise ValueError(f'Duplicate image name: {path.name}')
        sources[path.name] = path
    people = defaultdict(list)
    for ann in doc['annotations']:
        if not ann.get('iscrowd') and ann.get('keypoints'):
            people[ann['image_id']].append(ann)
    images = stage / 'images' / split
    labels = stage / 'labels' / split
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    counts = Counter()
    photos = defaultdict(list)
    for img in doc['images']:
        photos[Path(img['file_name']).name].append(img)
    for name, entries in photos.items():
        persons = [p for img in entries for p in people.pop(img['id'], [])]
        if not persons:
            continue
        if name not in sources:
            raise FileNotFoundError(f'{name} missing under {base}')
        src = sources[name]
        with Image.open(src) as image:
            width, height = image.size
        if any((img.get('width', width), img.get('height', height)) != (width, height) for img in entries):
            counts['size_mismatch_using_jpeg'] += 1
        rows = list(dict.fromkeys(row for p in persons if (row := label(p, width, height, counts))))
        if rows:
            dest = images / name
            try:
                os.link(src, dest)
            except FileExistsError:
                raise
            except OSError:
                shutil.copyfile(src, dest)
            (labels / (dest.stem + '.txt')).write_text('\n'.join(rows)+'\n', encoding='ascii')
            counts['images'] += 1
            counts['people'] += len(rows)
    print(split, dict(counts), flush=True)
    if people or not counts['images'] or not counts['foot_points']:
        raise ValueError(f'Incomplete {split} conversion')


def prepare(root):
    data = root / 'data'
    data.mkdir(exist_ok=True)
    dest = data / 'halpe26_pose_v1'
    config = dest / 'halpe26.yaml'
    if config.is_file():
        return config
    stage = Path(tempfile.mkdtemp(dir=data, prefix='halpe26_build_'))
    try:
        for split in ('train', 'val'):
            convert(root / 'halpe-26', stage, split)
        (stage / config.name).write_text(
            'path: ' + json.dumps(dest.as_posix(), ensure_ascii=False) +
            '\ntrain: images/train\nval: images/val\nnames: [person]\nkpt_shape: [26, 3]\n'
            + 'flip_idx: ' + str(FLIP) + '\n', encoding='utf-8')
        stage.rename(dest)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=60)
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Use the Python environment with CUDA-enabled PyTorch.')
    root = Path.cwd()
    weights = root / 'yolov8n-pose.pt'
    project = root / 'data' / 'training_runs'
    name = 'halpe26_v1'
    run = project / name
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    if args.resume:
        checkpoint = run / 'weights' / 'last.pt'
        if not checkpoint.is_file():
            raise SystemExit(f'Missing checkpoint: {checkpoint}')
        YOLO(str(checkpoint)).train(resume=True)
        return
    if run.exists():
        raise SystemExit(f'Run exists: {run}. Use --resume if last.pt exists, or rename the run directory.')
    if not weights.is_file():
        raise SystemExit(f'Missing initial weights: {weights}. Run from the project root.')
    config = prepare(root)
    model = YOLO(str(weights))
    model.train(data=str(config), project=str(project), name=name, epochs=args.epochs,
                patience=15, imgsz=640, batch=args.batch, device=0, workers=2,
                cache=False, save=True, pretrained=True, seed=42)


if __name__ == '__main__':
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        main()
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
