import multiprocessing
import subprocess

def run_behavioral_tracking():
    subprocess.run(["python", "behavioral-tracking.py"])

def run_emotion_detection():
    subprocess.run(["python", "realtime-emotion.py"])
    
def run_custom_classification():
    subprocess.run(["python", "custom-classifier.py"])

if __name__ == "__main__":
    behavioral_tracking_process = multiprocessing.Process(target=run_behavioral_tracking)
    emotion_detection_process = multiprocessing.Process(target=run_emotion_detection)
    custom_classifier_process = multiprocessing.Process(target=run_custom_classification)

    behavioral_tracking_process.start()
    emotion_detection_process.start()
    custom_classifier_process.start()

    behavioral_tracking_process.join()
    emotion_detection_process.join()
    custom_classifier_process.join()
