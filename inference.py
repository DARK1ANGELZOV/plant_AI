from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


CLASSES = ["root", "stem", "leaves"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Minimal inference for plant segmentation (root/stem/leaves)."
    )
    p.add_argument(
        "--model", type=str, default="best.pt", help="Path to YOLO-seg weights (best.pt)."
    )
    p.add_argument("--source", type=str, default="sample.jpg", help="Image path to analyze.")
    p.add_argument("--out", type=str, default="results", help="Output directory for overlay + json.")
    p.add_argument("--conf", type=float, default=0.08, help="Confidence threshold.")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    p.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Weights not found: {model_path}")

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"Image not found: {src}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(src),
        conf=float(args.conf),
        imgsz=int(args.imgsz),
        retina_masks=True,
        verbose=False,
        device=args.device,
    )

    if not results:
        raise SystemExit("No results returned from model.")

    r = results[0]
    overlay = r.plot()
    overlay_path = out_dir / f"{src.stem}_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay)

    summary = []
    boxes = r.boxes
    masks = r.masks
    if boxes is not None and masks is not None:
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        for i, cls_id in enumerate(cls_ids):
            class_name = r.names.get(int(cls_id), str(cls_id))
            summary.append(
                {
                    "class_id": int(cls_id),
                    "class_name": class_name,
                    "confidence": float(confs[i]) if i < len(confs) else 0.0,
                }
            )

    json_path = out_dir / f"{src.stem}_detections.json"
    json_path.write_text(
        __import__("json").dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Overlay:", overlay_path)
    print("Detections:", json_path)


if __name__ == "__main__":
    main()
