from pathlib import Path
import cv2

def draw_bbox_from_folder(
    image_dir,
    label_dir,
    output_dir,
    color=(0, 255, 0),
    thickness=2,
    show_conf=True,
):
    """
    image_dir : folder chứa ảnh
    label_dir : folder chứa file txt YOLO
    output_dir: folder lưu ảnh đã vẽ bbox
    """

    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    for img_path in image_dir.iterdir():
        if img_path.suffix.lower() not in img_exts:
            continue

        label_path = label_dir / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Cannot read {img_path.name}")
            continue

        h, w = img.shape[:2]

        if not label_path.exists():
            print(f"⚠ No label for {img_path.name}, skip bbox")
            cv2.imwrite(str(output_dir / img_path.name), img)
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            cls, xc, yc, bw, bh, conf = parts
            xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
            conf = float(conf)

            # YOLO → pixel
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            # Clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            if show_conf:
                label = f"{cls}:{conf:.2f}"
                cv2.putText(
                    img,
                    label,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        # Save image
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        print(f"✔ Saved: {out_path}")

    print("✅ All images processed.")
draw_bbox_from_folder(
    image_dir="dataset/train",
    label_dir="output_results_7/labels",
    output_dir="visualized_results",
)
