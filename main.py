import multiprocessing
import subprocess

def run_behavioral_tracking():
    subprocess.run(["python", "behavioral-tracking.py"])

def run_emotion_detection():
    subprocess.run(["python", "realtime-emotion.py"])

if __name__ == "__main__":
    behavioral_tracking_process = multiprocessing.Process(target=run_behavioral_tracking)
    emotion_detection_process = multiprocessing.Process(target=run_emotion_detection)

    behavioral_tracking_process.start()
    emotion_detection_process.start()

    behavioral_tracking_process.join()
    emotion_detection_process.join()
