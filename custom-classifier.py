import cv2
from ultralytics import YOLO
import numpy as np

# Open the default camera (usually webcam)
cap = cv2.VideoCapture(0)
model = YOLO("yolov8m.pt")

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Couldn't open camera")
    exit()

# Loop to capture frames from the camera
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    results = model(frame, device="mps")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
    
    
    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
