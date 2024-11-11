import time
from collections import deque
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
import cv2
from picamera2 import Picamera2, Preview

# Initialize YOLO model (using the smaller 'yolov8n.pt' model for better performance)
model = YOLO("yolov8n.pt")  # Use the 'n' model (nano model) for faster inference
names = model.model.names

# Known object width in real-world units (meters) and focal length of camera
KNOWN_WIDTH = 0.5  # Width of the object in meters (adjust according to the object you're tracking)
FOCAL_LENGTH = 600  # Adjust based on your camera's calibration

# Initialize the Picamera2
picam2 = Picamera2()

# Configure the camera for maximum FPS: Lower resolution and raw formats
picam2.preview_configuration.main.size=(1920,1080)
picam2.preview_configuration.main.format="RGB888"
picam2.video_configuration.controls.FrameRate=25.0
# Start the camera
picam2.start()

# Variables for tracking
prev_distance = None
prev_time = None
distance_buffer = deque(maxlen=5)  # Rolling buffer for smoothing distance values
speed_buffer = deque(maxlen=5)  # Rolling buffer for smoothing speed values

# Frame center coordinates
frame_center_x = 1920 // 2
frame_center_y = 1080// 2

# Variables for determining object direction
approaching_from_left = False
approaching_from_right = False

while True:
    # Capture frame from Picamera2
    frame = picam2.capture_array()

    # Ensure the frame is in RGB (not RGBA or other formats)
    if frame.shape[-1] == 4:  # If the image has 4 channels (RGBA)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)  # Convert to RGB

    # Perform detection on the current frame
    results = model.predict(frame, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()

    # Annotator for drawing bounding boxes and labels
    annotator = Annotator(frame, line_width=2, example=names)

    if boxes:
        # Assume we're tracking the first detected object
        x_min, y_min, x_max, y_max = boxes[0]
        pixel_width = x_max - x_min
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2

        # Ensure the bounding box width is valid
        if pixel_width > 0:
            # Estimate distance from the camera
            current_distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width
            distance_buffer.append(current_distance)

            # Calculate smoothed distance using the rolling average
            smoothed_distance = sum(distance_buffer) / len(distance_buffer)
            distance_text = f"{smoothed_distance:.2f} meters"

            # Calculate speed if previous distance is available
            if prev_distance is not None and prev_time is not None:
                time_diff = time.time() - prev_time
                if time_diff > 0:
                    # Calculate the change in distance and estimate speed
                    distance_diff = smoothed_distance - prev_distance
                    current_speed = distance_diff / time_diff  
                    speed_buffer.append(abs(current_speed))
                    smoothed_speed = sum(speed_buffer) / len(speed_buffer)
                    speed_text = f"Speed: {smoothed_speed:.2f} m/s"
                else:
                    speed_text = "Calculating speed..."
            else:
                speed_text = "Calculating speed..."

            # Update previous distance and time for the next frame
            prev_distance = smoothed_distance
            prev_time = time.time()

            # Determine whether the object is approaching from the left or right
            if center_x < frame_center_x:
                # Object is on the left side of the frame
                approaching_from_left = False
                approaching_from_right = True
            else:
                # Object is on the right side of the frame
                approaching_from_left = True
                approaching_from_right = False

            # Add side information to the label
            side_label = "Left" if approaching_from_left else "Right"
            annotator.box_label(boxes[0], color=colors(int(clss[0]), True),
                                label=f"{names[int(clss[0])]} {distance_text}, {speed_text}, {side_label}")

    # Display the annotated frame
    annotated_frame = annotator.result()

    # Convert frame back to BGR for OpenCV compatibility (Picamera2 uses RGB)
    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Show frame using OpenCV
    cv2.imshow("YOLOv8 Left/Right Detection", annotated_frame_bgr)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
picam2.stop()
cv2.destroyAllWindows()
