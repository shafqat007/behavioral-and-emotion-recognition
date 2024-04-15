import cv2
from ultralytics import YOLO
import numpy as np
import pygame


cap = cv2.VideoCapture(0)
model = YOLO("yolov8m.pt")

pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav")

text_classes = []
with open('classes.txt', "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        text_classes.append(class_name)
        
# print(classes)
# print(type(classes))

if not cap.isOpened():
    print("Error: Couldn't open camera")
    exit()

while True:
    
    ret, frame = cap.read()

    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    results = model(frame, device="mps")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    for cls, bbox in zip(classes, bboxes):
        if cls == 67:
            (x, y, x2, y2) = bbox
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
            cv2.putText(frame, text_classes[cls], (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
            alert_sound.play()
    
    
    cv2.imshow('frame', cv2.resize(frame, (640, 480)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
