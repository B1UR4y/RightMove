"""Render the RightMove pose model's predictions or export them to CVAT."""

import argparse
from collections import Counter
import json
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from camera_demo import DETECTION_CONF, draw_skeleton
from train_halpe26 import label as halpe_yolo_label

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POINT_CONFIDENCE = 0.3
KEYPOINT_COUNT = 26


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def render_frame(source, destination, image, people):
    destination.parent.mkdir(parents=True, exist_ok=True)
    has_points = False
    for person in people:
        if np.any(person[:, 2] >= POINT_CONFIDENCE):
            draw_skeleton(image, person, conf_th=POINT_CONFIDENCE)
            has_points = True
    if has_points:
        ok, encoded = cv2.imencode(source.suffix.lower(), image)
        if not ok:
            raise ValueError("cannot encode image")
        destination.write_bytes(encoded.tobytes())
    else:
        shutil.copyfile(source, destination)
    return has_points


def pose_txt(result, people):
    if len(people) == 0:
        return ""
    height, width = result.orig_shape
    lines = []
    counts = Counter()
    for box, points in zip(result.boxes.xywh.cpu().numpy(), people, strict=True):
        # The training converter expects an upper-left box and COCO visibility codes.
        cx, cy, box_width, box_height = map(float, box)
        points = points.copy()
        points[:, 2] = 2 * (points[:, 2] >= POINT_CONFIDENCE)
        line = halpe_yolo_label(
            {"bbox": (cx - box_width / 2, cy - box_height / 2, box_width, box_height),
             "keypoints": points.ravel().tolist()},
            width, height, counts,
        )
        if line:
            lines.append(line + "\n")
    return "".join(lines)


def process_frames(paths, model, device, batch_size, save_frame):
    saved = annotated = failed = 0
    last_report = 0
    for start in range(0, len(paths), batch_size):
        batch_paths, images = [], []
        for path in paths[start:start + batch_size]:
            try:
                image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8),
                                     cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("cannot decode image")
            except (OSError, ValueError, cv2.error) as exc:
                print(f"{path}: {exc}", file=sys.stderr)
                failed += 1
                continue
            batch_paths.append(path)
            images.append(image)

        results = model.predict(images, device=device, imgsz=640, conf=DETECTION_CONF,
                                verbose=False) if images else []
        for path, image, result in zip(batch_paths, images, results, strict=True):
            people = result.keypoints.data.cpu().numpy() if result.keypoints is not None else ()
            try:
                annotated += save_frame(path, image, result, people)
                saved += 1
            except (OSError, ValueError, cv2.error) as exc:
                print(f"{path}: {exc}", file=sys.stderr)
                failed += 1

        processed = saved + failed
        if processed - last_report >= 256 or processed == len(paths):
            print(f"Processed {processed}/{len(paths)}", flush=True)
            last_report = processed
    return saved, annotated, failed


def export_cvat(paths, source, output, model, device, batch_size, label_name):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as staging:
        archive_path = Path(staging) / output.name
        with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
            manifest = []

            def save_frame(path, _image, result, people):
                relative = path.relative_to(source)
                text = pose_txt(result, people)
                archive.writestr(f"labels/train/{relative.with_suffix('.txt').as_posix()}", text)
                manifest.append(f"images/train/{relative.as_posix()}\n")
                return bool(text)

            summary = process_frames(paths, model, device, batch_size, save_frame)
            archive.writestr("train.txt", "".join(manifest))
            archive.writestr(
                "data.yaml",
                f"path: ./\ntrain: train.txt\nkpt_shape: [{KEYPOINT_COUNT}, 3]\n"
                f"names:\n  0: {json.dumps(label_name, ensure_ascii=False)}\n",
            )
        if summary[2]:
            print("CVAT archive was not written because some frames failed.", file=sys.stderr)
        else:
            archive_path.replace(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/frames")
    parser.add_argument("--output", type=Path, help="image directory or CVAT .zip file")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "data/training_runs/halpe26_v1/weights/best.pt")
    parser.add_argument("--device", help="YOLO device, e.g. 0 or cpu (default: CUDA if available)")
    parser.add_argument("--batch", type=positive_int, default=16, help="frames per inference batch")
    parser.add_argument("--limit", type=positive_int, help="process only the first N frames")
    parser.add_argument("--cvat", action="store_true",
                        help="export an Ultralytics YOLO Pose 1.0 annotation archive")
    parser.add_argument("--cvat-label", default="person", help="skeleton label name in CVAT")
    args = parser.parse_args()

    started = perf_counter()
    source = args.input.resolve()
    default_output = "data/frames_cvat.zip" if args.cvat else "data/frames_annotated"
    output = (args.output or ROOT / default_output).resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        parser.error("input and output must be separate")
    if args.cvat and output.suffix.lower() != ".zip":
        parser.error("CVAT output must have the .zip extension")
    if not source.is_dir():
        parser.error(f"input directory not found: {source}")
    if not args.model.is_file():
        parser.error(f"model weights not found: {args.model}")

    paths = sorted(path for path in source.rglob("*")
                   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    paths = paths[:args.limit]
    if not paths:
        parser.error(f"no supported images found in {source}")
    if args.cvat and len({path.with_suffix(".txt") for path in paths}) != len(paths):
        parser.error("images with the same directory and basename would share one label file")

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"Frames: {len(paths)} | Device: {device} | Output: {output}", flush=True)
    model = YOLO(str(args.model), task="pose")
    if args.cvat:
        if tuple(model.model.yaml.get("kpt_shape") or ()) != (KEYPOINT_COUNT, 3):
            parser.error("CVAT export requires a Halpe-26 pose model with 26 x 3 keypoints")
        saved, annotated, failed = export_cvat(
            paths, source, output, model, device, args.batch, args.cvat_label,
        )
    else:
        def save_frame(path, image, _result, people):
            return render_frame(path, output / path.relative_to(source), image, people)

        saved, annotated, failed = process_frames(paths, model, device, args.batch, save_frame)

    elapsed = perf_counter() - started
    print(f"Processed successfully: {saved} | With pose: {annotated} | Failed: {failed} | "
          f"Time: {elapsed:.1f} s ({len(paths) / elapsed:.1f} frames/s)")
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
