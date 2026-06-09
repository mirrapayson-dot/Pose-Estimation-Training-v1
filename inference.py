from ultralytics import YOLO

# Load the pretrained YOLOv8 pose model
# 'n' = nano (fastest, good for testing)
model = YOLO("yolov8s-pose.pt")

# Run pose estimation on your video
results = model(
    source="/Users/mpayson/Ski Vid/leftvidcropped.mov",  # replace with your actual video path
    show=False,                # set to True if you want a live preview window
    save=True,                 # saves annotated output video
    device="mps",        # Apple Silicon Mac — change to "cpu" if Intel
    stream=True,
)

# Print keypoint data for each frame
for result in results:
    if result.keypoints is not None:
        print("Keypoints:", result.keypoints.xy)   # x,y coords for each joint
        print("Confidence:", result.keypoints.conf)