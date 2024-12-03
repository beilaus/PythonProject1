import cv2  # OpenCV for video processing
import torch
import pathlib
from pathlib import Path

# Fixing Path compatibility issue (for Windows environments)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model (custom model)
model = torch.hub.load("ultralytics/yolov5", 'custom', "bestCone.pt", force_reload=True)
model.conf = 0.6  # confidence threshold (0-1)
# Video input (file path)
video_path = "fsd1.mp4"

# Open video using OpenCV
cap = cv2.VideoCapture(video_path)

# Get video frame rate (FPS) and frame size
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video file (to save the result)
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (YOLOv5 expects RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Inference (process the frame with YOLOv5)
    results = model(frame_rgb)

    # Render the results on the frame (e.g., bounding boxes)
    annotated_frame = results.render()[0]  # This adds the boxes on the image

    # Convert back to BGR for OpenCV to show or save the frame
    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Write the frame with results to the output video
    out.write(annotated_frame_bgr)

    # Show the frame (optional)
    cv2.imshow('Frame', annotated_frame_bgr)
    # If you want to exit early press 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()

# Close OpenCV window
cv2.destroyAllWindows()

print(f"Processed video saved to {output_path}")
