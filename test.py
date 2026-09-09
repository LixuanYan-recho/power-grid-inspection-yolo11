"""Validate and run inference with the trained YOLO11 detection model.

Examples (run in the project directory):

    python test.py                         # validate + predict on val images
    python test.py --mode val              # only calculate validation metrics
    python test.py --mode predict --source "Defective/A_F (1).JPG"
    python test.py --mode predict --source "D:/some/images" --conf 0.40

There is no separate test split in this repository.  ``--mode val`` evaluates
the 48-image validation split described by ``power_grid.yaml``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "runs" / "power_grid_detect" / "weights" / "best.pt"
DEFAULT_DATA = ROOT / "power_grid.yaml"
DEFAULT_SOURCE = ROOT / "yolo_dataset" / "images" / "val"


def choose_device(value: str) -> int | str:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if value.lower() != "auto":
        # Keep explicit values such as ``cpu`` or ``0`` unchanged.
        try:
            return int(value)
        except ValueError:
            return value
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def validate(model, data: Path, imgsz: int, batch: int, device: int | str) -> None:
    print(f"\n开始验证: {data}")
    metrics = model.val(
        data=str(data),
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,
        plots=True,
        project=str(ROOT / "runs"),
        name="power_grid_val",
        exist_ok=True,
    )

    # Ultralytics exposes these values for detection models.  Use getattr so
    # the script still prints a useful result if the library changes slightly.
    box = getattr(metrics, "box", None)
    if box is not None:
        print("\n验证指标:")
        print(f"  Precision : {box.mp:.4f}")
        print(f"  Recall    : {box.mr:.4f}")
        print(f"  mAP50     : {box.map50:.4f}")
        print(f"  mAP50-95  : {box.map:.4f}")
    print(f"验证结果和曲线已保存到: {ROOT / 'runs' / 'power_grid_val'}")


def predict(
    model,
    source: Path,
    imgsz: int,
    conf: float,
    device: int | str,
    output_name: str,
    save_txt: bool,
    save_conf: bool,
    show: bool,
) -> None:
    if not source.exists():
        raise FileNotFoundError(f"找不到预测输入: {source}")

    print(f"\n开始预测: {source}")
    results = model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=conf,
        device=device,
        save=True,
        save_txt=save_txt,
        save_conf=save_conf,
        show=show,
        project=str(ROOT / "runs"),
        name=output_name,
        exist_ok=True,
        verbose=True,
    )

    # Print a compact per-image summary in addition to the saved annotated
    # images.  ``results`` is a list for normal (non-streaming) prediction.
    total = 0
    for result in results:
        count = 0 if result.boxes is None else len(result.boxes)
        total += count
        print(f"  {Path(result.path).name}: {count} 个目标")
    print(f"共检测到 {total} 个目标")
    print(f"预测图片已保存到: {ROOT / 'runs' / output_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证和测试 YOLO11 电力巡检模型")
    parser.add_argument("--mode", choices=("val", "predict", "both"), default="both")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="模型 .pt 路径")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="YOLO 数据集 yaml")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="图片或图片文件夹")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25, help="预测置信度阈值")
    parser.add_argument("--device", default="auto", help="auto、cpu 或 GPU 编号（如 0）")
    parser.add_argument("--output-name", default="power_grid_predictions")
    parser.add_argument("--save-txt", action="store_true", help="同时保存 YOLO 检测 txt")
    parser.add_argument("--save-conf", action="store_true", help="在 txt 中保存置信度")
    parser.add_argument("--show", action="store_true", help="显示预测窗口")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    data_path = args.data if args.data.is_absolute() else ROOT / args.data
    source_path = args.source if args.source.is_absolute() else ROOT / args.source

    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到模型: {model_path}\n请确认训练已完成，或用 --model 指定 best.pt。"
        )
    if args.mode in ("val", "both") and not data_path.exists():
        raise FileNotFoundError(f"找不到数据配置: {data_path}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("请先安装依赖: python -m pip install ultralytics", file=sys.stderr)
        raise

    device = choose_device(args.device)
    print(f"使用模型: {model_path}")
    print(f"使用设备: {device}")
    model = YOLO(str(model_path))

    if args.mode in ("val", "both"):
        validate(model, data_path, args.imgsz, args.batch, device)
    if args.mode in ("predict", "both"):
        predict(
            model,
            source_path,
            args.imgsz,
            args.conf,
            device,
            args.output_name,
            args.save_txt,
            args.save_conf,
            args.show,
        )


if __name__ == "__main__":
    main()
