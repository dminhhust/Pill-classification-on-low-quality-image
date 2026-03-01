import albumentations as A

from ultralytics import YOLO

def main():
    # Load model
    model = YOLO("yolo11n.pt")

    # Define custom transforms with various augmentation techniques
    custom_transforms = [
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=1.0,
            rotate=(-180, 180),
            shear=0,
            fit_output=False,
            p=0.5,
        ),
        # Blur variations
        A.OneOf(
            [
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
            ],
            p=0.4,
        ),
        # Noise variations
        A.OneOf(
            [
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ],
            p=0.2,
        ),
        # Color and contrast adjustments
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        # Simulate occlusions
        A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.2
        ),
    ]

    # Train with custom transforms
    results = model.train(
        data="coco.yaml",
        epochs=200,
        warmup_epochs=5,
        patience=20,
        lr0=0.005,
        weight_decay=0.0005,
        batch= 16,
        imgsz=640,
        augmentations=custom_transforms,
        mixup=0.1,
    )
if __name__ == "__main__":
    main()