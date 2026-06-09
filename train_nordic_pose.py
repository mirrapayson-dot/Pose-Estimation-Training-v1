from ultralytics import YOLO

model = YOLO("yolov8s-pose.pt")

results = model.train(
    data    = "nordic_skier.yaml",
    epochs  = 100,
    imgsz   = 640,
    batch   = 8,          # lower to 4 if you get memory errors
    device  = "mps",      # Apple Silicon; switch to "cpu" if errors occur
    project = "nordic_training",
    name    = "skier_pose_v1",

    # Augmentation
    flipud  = 0.0,        # never flip upside down
    fliplr  = 0.5,        # horizontal flip is fine
    mosaic  = 0.5,
    degrees = 5,

    # Keypoint loss weights
    pose    = 12.0,
    kobj    = 2.0,
)

print(f"Training complete. Best model: {results.save_dir}/weights/best.pt")