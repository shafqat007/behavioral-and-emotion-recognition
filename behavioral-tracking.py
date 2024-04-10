import cv2
import dlib
import numpy as np
from imutils import face_utils

print("Imported Successfully!")

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
    

face_detector = dlib.get_frontal_face_detector()
faceLandmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

sleepiness = 0
drowsiness = 0
awakeness = 0
activity = ""
color = (0,0,0)

def distance(Px, Py):
    displacement = np.linalg.norm(Px - Py)
    return displacement

def blinking_detection(a, b, c, d, e, f):
    short_distance = distance(b, d) + distance(c, e)
    long_distance = distance(a, f)
    ratio = short_distance / (2.0 * long_distance)

    if ratio > 0.25:
        return 2
    elif 0.21 < ratio <= 0.25:
        return 1
    else:
        return 0

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    mouth_aspect_distance = abs(top_mean[1] - low_mean[1])

    if mouth_aspect_distance > 20:
        return 3
    else:
        return 4

def detect_face_features(face):
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    landmarks = faceLandmarks(gray, face)
    landmarks = face_utils.shape_to_np(landmarks)
    return landmarks

def check_activity(left_eye, right_eye, mar):
    global sleepiness, drowsiness, awakeness, activity, color

    if left_eye == 0 or right_eye == 0 or mar == 0:
        sleepiness += 1
        drowsiness = 0
        awakeness = 0
        if sleepiness > 6:
            activity = "Alert! Are you sleeping?"
            color = (0, 0, 255)
    elif left_eye == 1 or right_eye == 1 or mar == 3:
        sleepiness = 0
        drowsiness += 1
        awakeness = 0
        if drowsiness > 6:
            activity = "Hushh! You look sleepy!"
            color = (0, 0, 0)
    elif left_eye == 2 or right_eye == 2 and mar == 4:
        drowsiness = 0
        sleepiness = 0
        awakeness += 1
        if awakeness > 6:
            activity = "Having a safe driving!"
            color = (0, 255, 0)
    elif left_eye == 2 or right_eye == 2 and mar == 3:
        drowsiness = 0
        sleepiness += 1
        awakeness = 0
        if sleepiness > 1:
            activity = "Hushh! You look sleepy!"
            color = (0, 255, 0)

def draw_activity_text(frame, activity, color):
    cv2.rectangle(frame, (10, 5), (445, 40), (255, 255, 255), -1)
    cv2.putText(frame, activity, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray)

    if len(faces) > 0:
        for face in faces:
            landmarks = detect_face_features(face)
            left_eye = blinking_detection(landmarks[36], landmarks[37], landmarks[38],
                                           landmarks[41], landmarks[40], landmarks[39])
            right_eye = blinking_detection(landmarks[42], landmarks[43], landmarks[44],
                                            landmarks[47], landmarks[46], landmarks[45])
            mar = lip_distance(landmarks)

            check_activity(left_eye, right_eye, mar)

    else:
        activity = "No Driver!"
        color = (0, 0, 0)

    draw_activity_text(frame, activity, color)

    cv2.imshow("Hola!", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
