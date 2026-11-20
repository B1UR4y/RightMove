"""Обрезка видео по таблице recordings/Таблица_записей.csv.

Оставляет только время от начала до конца стойки (колонки "Начало, с" и "Конец, с").
Исходные файлы не меняются, результат пишется в recordings/cut/.
Нужен ffmpeg в PATH.
"""
import csv
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent / "recordings"
OUT = ROOT / "cut"


def main():
    OUT.mkdir(exist_ok=True)
    with open(ROOT / "Таблица_записей.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    for r in rows:
        name = r["Файл (без расширения)"]
        if not r["Начало, с"] or not r["Конец, с"]:
            print(f"{name}: нет отметок времени, пропуск")
            continue
        start, end = float(r["Начало, с"]), float(r["Конец, с"])
        src, dst = ROOT / f"{name}.mp4", OUT / f"{name}.mp4"
        if not src.exists():
            print(f"{name}: нет файла")
            continue
        # -ss после -i: точная обрезка по кадрам (ролики короткие, перекодирование быстрое)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ss", str(start), "-to", str(end),
             "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)],
            check=True)
        print(f"{name}: {start:g}-{end:g} с -> {dst.name}")


if __name__ == "__main__":
    main()
