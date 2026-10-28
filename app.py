import streamlit as st
import cv2
import time
import base64
import numpy as np
import pandas as pd
import mediapipe as mp

from utils.eye_metrics import calculate_avg_ear, LEFT_EYE_INDICES, RIGHT_EYE_INDICES
from utils.distance_estimator import estimate_distance_cm

st.set_page_config(
    page_title="VisionCare AI",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ VisionCare AI: Real-Time Ergonomics & Eye-Strain Monitor")
st.caption("Active Computer Vision Syndrome (CVS) detection using MediaPipe FaceMesh, EAR Blink Analysis, and Distance Projection.")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Detection Controls")
ear_threshold = st.sidebar.slider("EAR Blink Cutoff", 0.15, 0.32, 0.22, 0.01)
safe_distance_cm = st.sidebar.slider("Min Safe Screen Distance (cm)", 35, 75, 50, 5)
break_interval_sec = st.sidebar.slider("20-20-20 Break Demo Timer (Seconds)", 10, 1200, 30, 5)

run_monitor = st.sidebar.toggle("Start Camera Monitor", value=True)

# --- Layout Grid ---
col_cam, col_metrics = st.columns([1.7, 1.3])

with col_cam:
    video_placeholder = st.empty()
    status_alert_box = st.empty()

with col_metrics:
    kpi_col1, kpi_col2 = st.columns(2)
    metric_ear = kpi_col1.empty()
    metric_bpm = kpi_col2.empty()

    kpi_col3, kpi_col4 = st.columns(2)
    metric_dist = kpi_col3.empty()
    metric_fatigue = kpi_col4.empty()

    st.subheader("📊 Live Session Telemetry")
    graph_placeholder = st.empty()

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Runtime trackers
cap = cv2.VideoCapture(0)
blink_consec_frames = 0
total_blinks = 0
blinks_current_minute = 0
minute_window_start = time.time()
timer_20_start = time.time()
session_start = time.time()
current_bpm = 16

telemetry_history = []

while run_monitor and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Webcam not detected. Ensure no other application (Zoom, Teams, etc.) is accessing the camera.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_ear = 0.0
    dist_cm = 60.0
    active_warnings = []
    now = time.time()

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Draw green dots on eyes to showcase CV landmark tracking
        for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
            px = int(landmarks[idx].x * w)
            py = int(landmarks[idx].y * h)
            cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

        # 1. EAR Calculation
        current_ear = calculate_avg_ear(landmarks, w, h)

        if current_ear < ear_threshold:
            blink_consec_frames += 1
        else:
            if blink_consec_frames >= 2:
                total_blinks += 1
                blinks_current_minute += 1
            blink_consec_frames = 0

        # 2. Metric Distance
        dist_cm = estimate_distance_cm(landmarks, w, h)
        if dist_cm < safe_distance_cm:
            active_warnings.append(f"⚠️ **Too Close:** You are {dist_cm:.1f} cm away (Recommended ≥ {safe_distance_cm} cm).")

    # 3. Dynamic Blinks/Min (BPM)
    elapsed_min = now - minute_window_start
    if elapsed_min >= 60.0:
        current_bpm = blinks_current_minute
        blinks_current_minute = 0
        minute_window_start = now
    else:
        current_bpm = int((blinks_current_minute / max(1, elapsed_min)) * 60)

    if current_bpm < 10 and (now - session_start) > 10:
        active_warnings.append("⚠️ **Low Blink Frequency:** Blink rate dropped under 10 BPM. High dry eye risk.")

    # 4. 20-20-20 Timer
    elapsed_break_timer = now - timer_20_start
    if elapsed_break_timer >= break_interval_sec:
        active_warnings.append("🔔 **20-20-20 Rule Alert:** Rest your eyes! Look 20 feet away for 20 seconds.")

    # 5. Composite Fatigue Score
    fatigue_index = 0
    if current_bpm < 10:
        fatigue_index += min(40, (10 - current_bpm) * 4)
    if dist_cm < safe_distance_cm:
        fatigue_index += min(40, int((safe_distance_cm - dist_cm) * 2))
    session_minutes = (now - session_start) / 60
    fatigue_index += min(20, int(session_minutes * 1.5))
    fatigue_index = min(100, max(0, fatigue_index))

    # Telemetry logging (sampled once per second)
    elapsed_total_sec = int(now - session_start)
    if len(telemetry_history) == 0 or telemetry_history[-1]["Seconds"] != elapsed_total_sec:
        telemetry_history.append({
            "Seconds": elapsed_total_sec,
            "Blink Rate (BPM)": current_bpm,
            "Fatigue Index (%)": fatigue_index,
            "Distance (cm)": round(dist_cm, 1)
        })

    # Encode frame to JPEG Base64 to bypass Streamlit MediaFileManager bug
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    video_placeholder.markdown(
        f'<img src="data:image/jpeg;base64,{img_b64}" style="width:100%; border-radius:10px; border:2px solid #333;">',
        unsafe_allow_html=True
    )

    # Render Metrics
    metric_ear.metric("Average EAR", f"{current_ear:.2f}")
    metric_bpm.metric("Blink Rate", f"{current_bpm} BPM", delta=f"{total_blinks} total")
    metric_dist.metric("Screen Distance", f"{dist_cm:.1f} cm")
    metric_fatigue.metric(
        "Fatigue Index",
        f"{fatigue_index}%",
        delta="-Normal" if fatigue_index < 45 else "+High Strain",
        delta_color="inverse"
    )

    # Render Status Alerts
    if active_warnings:
        status_alert_box.warning("\n\n".join(active_warnings))
    else:
        remaining_break = max(0, int(break_interval_sec - elapsed_break_timer))
        status_alert_box.success(f"Ocular state healthy. Next 20-20-20 reminder in **{remaining_break}s**.")

    # Render Live Telemetry Chart
    if len(telemetry_history) > 1:
        df = pd.DataFrame(telemetry_history[-30:])
        graph_placeholder.line_chart(df.set_index("Seconds")[["Blink Rate (BPM)", "Fatigue Index (%)"]])

    time.sleep(0.02)

cap.release()