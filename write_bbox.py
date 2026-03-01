from pathlib import Path

def write_best_bbox_for_all(label_dir):
    """
    Ghi 1 bbox tốt nhất vào file KHÔNG có _msr hoặc _ssr
    """
    label_dir = Path(label_dir)

    def read_best_bbox(txt_path):
        if not txt_path.exists():
            return None

        best_line = None
        best_conf = -1

        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                conf = float(parts[5])
                if conf > best_conf:
                    best_conf = conf
                    best_line = line.strip()

        return best_line

    # Duyệt các file KHÔNG có _msr
    for txt_file in label_dir.glob("*.txt"):
        if "_msr" in txt_file.stem:
            continue
        if "_ssr" in txt_file.stem:
            continue
        base_name = txt_file.stem
        print(f"Processing: {base_name}")

        # 1️⃣ đọc file gốc
        best_bbox = read_best_bbox(txt_file)

        # 2️⃣ nếu không có → đọc file _msr_bilateral
        if not best_bbox:
            msr_file = label_dir / f"{base_name}_msr.txt"
            best_bbox = read_best_bbox(msr_file)
            if not best_bbox:
                # đọc file _ssr
                ssr_file = label_dir / f"{base_name}_ssr.txt"
                best_bbox = read_best_bbox(ssr_file)

        # 3️⃣ nếu vẫn không có → bbox mặc định
        if not best_bbox:
            best_bbox = "0 0.5 0.5 1 1 0.1"

        # 4️⃣ ghi lại vào file gốc
        with open(txt_file, "w") as f:
            f.write(best_bbox + "\n")

        print(f"  ✔ Written: {best_bbox}")

    print("✅ All labels processed.")
write_best_bbox_for_all("output_results_7/labels")
