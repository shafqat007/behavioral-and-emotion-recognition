import cv2
import pygame
from deepface import DeepFace

def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier('haar_face.xml')
    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayimg, 1.1, 5)
    return faces

def draw_emotion_text(frame, face, emotion):
    a, b, c, d = face
    font_scale = 2.0  
    text_size = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
    text_position = (a + c // 2 - text_size[0] // 2, b + d + 50)  
    cv2.rectangle(frame, (text_position[0], text_position[1] - text_size[1]), 
                  (text_position[0] + text_size[0], text_position[1]), (0, 0, 0), -1)
    cv2.putText(frame, emotion, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

def play_alert_sound(emotion):
    if emotion in ["sad", "angry"]:
        pygame.mixer.init()
        alert_sound = pygame.mixer.Sound("alert.wav")
        alert_sound.play()

def main():
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise IOError("Cannot open webcam")
    
    while True:
        ret, frame = cam.read()
        res = DeepFace.analyze(frame, actions=("emotion"), enforce_detection=False)
        faces = detect_faces(frame)
        
        for face in faces:
            cv2.rectangle(frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255,0,0), 3)
            emotion = res[0]['dominant_emotion']
            draw_emotion_text(frame, face, emotion)
            play_alert_sound(emotion)

        cv2.imshow("yayy", cv2.resize(frame, (640, 480)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()