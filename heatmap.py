import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque

window_size = 300  # Number of frames for the sliding window
heatmap_buffer = deque(maxlen=window_size)

def generate_heatmap(frame, results):
    frame_heatmap = np.zeros((video_height, video_width), dtype=np.float32)
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box[:4]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(frame_heatmap, (cx, cy), radius=20, color=1, thickness=-1)
    heatmap_buffer.append(frame_heatmap)

    # Sum the heatmaps in the buffer
    sliding_heatmap = np.sum(heatmap_buffer, axis=0)
    sliding_heatmap = cv2.GaussianBlur(sliding_heatmap, (0, 0), sigmaX=25, sigmaY=25)
    sliding_heatmap = np.clip(sliding_heatmap / sliding_heatmap.max(), 0, 1)
    heatmap_color = cv2.applyColorMap((sliding_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)

    return overlay

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "videos/prueba4_640x360s.mp4"
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.mp4', fourcc, video_fps, (video_width, video_height))


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.predict(frame, verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        heatmap = generate_heatmap(frame, results)
        # Display the annotated frame
        cv2.imshow("Heatmap", heatmap)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()