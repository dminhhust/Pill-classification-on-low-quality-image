import os
from pathlib import Path
import cv2
from ultralytics import YOLO

def detect_folder(
    model_path,
    input_dir,
    output_dir,
    imgsz=640,
    conf=0.25,
    iou=0.45,
):
    # Load model
    model = YOLO(model_path)

    # Create output directories
    output_dir = Path(output_dir)
    img_out_dir = output_dir / "images"
    label_out_dir = output_dir / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    for img_path in Path(input_dir).iterdir():
        if img_path.suffix.lower() not in img_exts:
            continue

        # Read image (to get width/height)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Cannot read {img_path.name}")
            continue
        h, w = img.shape[:2]

        # Run detection
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            save=False,
            verbose=False,
        )

        r = results[0]

        # =========================
        # Save annotated image
        # =========================
        annotated_img = r.plot()
        out_img_path = img_out_dir / img_path.name
        cv2.imwrite(str(out_img_path), annotated_img)

        # =========================
        # Save bounding boxes
        # YOLO format: class x_center y_center width height (normalized)
        # =========================
        label_path = label_out_dir / f"{img_path.stem}.txt"

        with open(label_path, "w") as f:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()

                for box, cls, score in zip(boxes, classes, confs):
                    x1, y1, x2, y2 = box

                    # Convert to YOLO normalized format
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                    f.write(
                        f"{int(cls)} {x_center:.6f} {y_center:.6f} "
                        f"{bw:.6f} {bh:.6f} {score:.4f}\n"
                    )

        print(f"✔ Saved image: {out_img_path}")
        print(f"✔ Saved boxes: {label_path}")

    print("✅ Detection + bounding box export completed.")


if __name__ == "__main__":
    detect_folder(
        model_path="runs/detect/train/weights/best.pt",
        input_dir="train",
        output_dir="output_results_7",
        imgsz=640,
        conf=0.2,
        iou=0.25,
    )
