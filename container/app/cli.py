import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/app")
from app.detector import CatDetector


def cmd_info():
    with open("/app/STUDENT.json", "r") as f:
        print(f.read())


def cmd_predict():
    input_dir  = Path("/data/input")
    output_dir = Path("/data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "predictions.csv"

    detector = CatDetector(
        onnx_path="/app/models/best.onnx",
        imgsz=640,
        conf=0.25
    )

    img_exts = {".jpg", ".jpeg", ".png"}
    image_paths = sorted([
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in img_exts
    ])

    print(f"Found {len(image_paths)} images", flush=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "xmin", "ymin", "xmax", "ymax", "confidence", "class"])

        for img_path in image_paths:
            rel_path = img_path.relative_to(input_dir).as_posix()
            detections = detector.predict(str(img_path))

            if not detections:
                writer.writerow([rel_path, "", "", "", "", "", ""])
            else:
                for det in detections:
                    writer.writerow([
                        rel_path,
                        det["xmin"],
                        det["ymin"],
                        det["xmax"],
                        det["ymax"],
                        det["confidence"],
                        det["class"]
                    ])

    print(f"Predictions saved to {output_csv}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Cat detector CLI")
    parser.add_argument("command", choices=["info", "predict"])
    args = parser.parse_args()

    if args.command == "info":
        cmd_info()
    elif args.command == "predict":
        cmd_predict()


if __name__ == "__main__":
    main()
