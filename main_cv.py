import cv2
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from datetime import datetime

from utils.eye_metrics import calculate_avg_ear
from utils.distance_estimator import estimate_distance_cm

# Settings
EAR_THRESHOLD = 0.22
SAFE_DISTANCE_CM = 50.0
BREAK_INTERVAL = 30  # Set to 30s for live demonstration

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
blink_frames = 0
total_blinks = 0
blinks_current_min = 0
minute_start = time.time()
timer_start = time.time()
session_start = time.time()
current_bpm = 15

# Logging lists
timestamps, bpms, distances, fatigues = [], [], [], []

csv_file = open("visioncare_session_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Elapsed_Sec", "BPM", "Distance_CM", "Fatigue_Score"])

print("VisionCare AI starting. Press 'q' to stop and generate the analytics report.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_ear = 0.0
    dist_cm = 60.0
    alerts = []
    now = time.time()

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        current_ear = calculate_avg_ear(landmarks, w, h)
        if current_ear < EAR_THRESHOLD:
            blink_frames += 1
        else:
            if blink_frames >= 2:
                total_blinks += 1
                blinks_current_min += 1
            blink_frames = 0

        dist_cm = estimate_distance_cm(landmarks, w, h)
        if dist_cm < SAFE_DISTANCE_CM:
            alerts.append("WARNING: TOO CLOSE TO SCREEN (<50cm)")

    elapsed_m = now - minute_start
    if elapsed_m >= 60.0:
        current_bpm = blinks_current_min
        blinks_current_min = 0
        minute_start = now
    else:
        current_bpm = int((blinks_current_min / max(1, elapsed_m)) * 60)

    if current_bpm < 10 and (now - session_start) > 10:
        alerts.append("LOW BLINK RATE: DRY EYE RISK")

    if now - timer_start >= BREAK_INTERVAL:
        alerts.append("20-20-20 BREAK: LOOK 20 FT AWAY!")

    fatigue = 0
    if current_bpm < 10:
        fatigue += min(40, (10 - current_bpm) * 4)
    if dist_cm < SAFE_DISTANCE_CM:
        fatigue += min(40, int((SAFE_DISTANCE_CM - dist_cm) * 2))
    fatigue += min(20, int(((now - session_start) / 60) * 1.5))
    fatigue = min(100, max(0, fatigue))

    # Log every second
    elapsed_sec = int(now - session_start)
    if len(timestamps) == 0 or timestamps[-1] != elapsed_sec:
        timestamps.append(elapsed_sec)
        bpms.append(current_bpm)
        distances.append(round(dist_cm, 1))
        fatigues.append(fatigue)
        csv_writer.writerow([datetime.now().strftime("%H:%M:%S"), elapsed_sec, current_bpm, round(dist_cm, 1), fatigue])

    # Semi-transparent HUD card
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (380, 210), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "VISIONCARE AI MONITOR", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"EAR: {current_ear:.2f}", (35, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Blink Count: {total_blinks} ({current_bpm} BPM)", (35, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if current_bpm >= 10 else (0, 165, 255), 1)
    cv2.putText(frame, f"Distance: {dist_cm:.1f} cm", (35, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if dist_cm >= SAFE_DISTANCE_CM else (0, 0, 255), 1)
    cv2.putText(frame, f"Fatigue Index: {fatigue}%", (35, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255) if fatigue > 50 else (0, 255, 0), 2)

    y_alert = 250
    for alert in alerts:
        cv2.putText(frame, alert, (35, y_alert), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        y_alert += 30

    cv2.imshow("VisionCare AI - OpenCV HUD", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()

# Generate visual report
if len(timestamps) > 2:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax1.plot(timestamps, bpms, label="Blink Rate (BPM)", color="blue")
    ax1.axhline(y=10, color="red", linestyle="--", label="Low Blink Risk Cutoff")
    ax1.set_ylabel("BPM")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(timestamps, fatigues, label="Fatigue Score (0-100)", color="crimson")
    ax2.plot(timestamps, distances, label="Distance (cm)", color="green", linestyle=":")
    ax2.set_xlabel("Elapsed Seconds")
    ax2.set_ylabel("Score / cm")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("visioncare_session_report.png")
    plt.show()