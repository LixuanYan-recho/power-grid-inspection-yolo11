"""Train YOLO11s on the inspection dataset.

The repository labels contain polygons (YOLO segmentation format), while
``yolo11s.pt`` is a detection model.  This script converts every polygon to
its enclosing bounding box and then starts object-detection training.

Run from this directory:
    python train.py

For pixel-accurate segmentation, download ``yolo11s-seg.pt`` and use a
segmentation-specific script instead.
"""

from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "yolo11s.pt"
LABEL_SOURCE = ROOT / "YOLO (labels)"
DATASET = ROOT / "yolo_dataset"
DATA_YAML = ROOT / "power_grid.yaml"


def image_index() -> dict[str, Path]:
    """Index the JPG files by stem (the labels use the same stem)."""
    result: dict[str, Path] = {}
    for folder in (ROOT / "Defective", ROOT / "Normal"):
        for image in folder.glob("*"):
            if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                result[image.stem] = image
    return result


def polygon_to_box(parts: list[str]) -> str | None:
    """Convert one YOLO polygon row to ``class xc yc width height``."""
    if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
        return None
    try:
        class_id = int(float(parts[0]))
        coordinates = [float(value) for value in parts[1:]]
    except ValueError:
        return None

    xs = coordinates[0::2]
    ys = coordinates[1::2]
    x_min, x_max = max(0.0, min(xs)), min(1.0, max(xs))
    y_min, y_max = max(0.0, min(ys)), min(1.0, max(ys))
    width, height = x_max - x_min, y_max - y_min
    if width <= 0 or height <= 0:
        return None
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def prepare_dataset() -> tuple[int, int]:
    """Create a clean YOLO detection dataset from train.txt/val.txt."""
    if not LABEL_SOURCE.exists():
        raise FileNotFoundError(f"找不到标注目录: {LABEL_SOURCE}")

    # Only remove the generated dataset, never the original images/labels.
    if DATASET.exists():
        shutil.rmtree(DATASET)

    images = image_index()
    counts = {"train": 0, "val": 0}
    missing = []

    for split in ("train", "val"):
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        split_file = LABEL_SOURCE / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"找不到划分文件: {split_file}")

        # The original txt files contain absolute paths from another computer;
        # only the filename stem is needed here.
        for line in split_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            stem = Path(line.strip()).stem
            image = images.get(stem)
            source_label = LABEL_SOURCE / f"{stem}.txt"
            if image is None or not source_label.exists():
                missing.append(stem)
                continue

            converted = []
            for row in source_label.read_text(encoding="utf-8").splitlines():
                row = row.strip()
                if not row:
                    continue
                box = polygon_to_box(row.split())
                if box is not None:
                    converted.append(box)
            if not converted:
                missing.append(f"{stem} (无有效标注)")
                continue

            shutil.copy2(image, image_dir / image.name)
            (label_dir / f"{stem}.txt").write_text("\n".join(converted) + "\n", encoding="utf-8")
            counts[split] += 1

    DATA_YAML.write_text(
        "path: " + DATASET.as_posix() + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: defective\n"
        "  1: normal\n",
        encoding="utf-8",
    )

    print(f"已准备数据集: train={counts['train']}, val={counts['val']}")
    if missing:
        print("跳过的文件:", ", ".join(missing))
    return counts["train"], counts["val"]


def main() -> None:
    if not MODEL.exists():
        raise FileNotFoundError(f"找不到模型权重: {MODEL}")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("请先安装依赖: python -m pip install ultralytics", file=sys.stderr)
        raise

    train_count, val_count = prepare_dataset()
    if train_count == 0 or val_count == 0:
        raise RuntimeError("训练集或验证集为空，请检查图片和标注文件。")

    model = YOLO(str(MODEL))
    try:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"使用设备: {device}")
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=4,
        workers=0,       # Windows 下更稳定
        device=device,
        project=str(ROOT / "runs"),
        name="power_grid_detect",
        patience=20,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
