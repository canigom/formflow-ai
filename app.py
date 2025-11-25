import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import google.generativeai as genai
from PIL import Image
import os

# --- SEITENEINSTELLUNGEN (SAYFA AYARLARI) ---
st.set_page_config(
    page_title="FormFlow AI - Der smarte Trainer",
    page_icon="🏋️",
    layout="wide"
)

# --- TITEL UND BESCHREIBUNG (BAŞLIK VE AÇIKLAMA) ---
st.title("🏋️ FormFlow AI")
st.markdown("""
**KI-gestützte biomechanische Bewegungsanalyse** Laden Sie Ihr Video hoch, lassen Sie Ihre Form von der Künstlichen Intelligenz analysieren und erhalten Sie individuelles Feedback zur Verletzungsprävention.
""")
    
# --- FUNKTIONEN (FONKSİYONLAR) ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def process_video(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    
    angle_history = []
    frame_indices = []
    frame_count = 0
    squat_count = 0
    stage = None
    
    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0: total_frames = 1
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        try:
            landmarks = results.pose_landmarks.landmark
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            angle = calculate_angle(hip, knee, ankle)
            angle_history.append(angle)
            frame_indices.append(frame_count)
            
            # Squat Logic
            if angle > 160:
                stage = "UP"
            if angle < 90 and stage == 'UP':
                stage = "DOWN"
                squat_count += 1
                
        except:
            pass
            
        frame_count += 1
        if frame_count % 10 == 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            status_text.text(f"Video wird verarbeitet... Frame: {frame_count}")

    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    return angle_history, frame_indices, squat_count

# --- HAUPTABLAUF (ANA AKIŞ) ---
uploaded_file = st.file_uploader("Wählen Sie ein Video zur Analyse (MP4/MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    st.video(video_path)
    
    if st.button("🚀 Analyse starten"):
        with st.spinner('KI berechnet Biomechanik...'):
            angles, frames, count = process_video(video_path)
            
            st.success("Analyse abgeschlossen!")
            
            # 1. Metriken (Metrikler)
            col1, col2 = st.columns(2)
            col1.metric("Wiederholungen gesamt", f"{count}", "Squats")
            if angles:
                col1.metric("Minimale Tiefe (Winkel)", f"{int(min(angles))}°", "Grad")
            
            # 2. Grafik (Grafik Çizimi)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(frames, angles, label='Kniewinkel', color='#007acc')
            ax.axhline(y=90, color='green', linestyle='--', label='Ziel (90°)')
            ax.axhline(y=160, color='red', linestyle='--', label='Start (160°)')
            ax.set_title("Bewegungsanalyse-Diagramm")
            ax.set_xlabel("Zeit (Frames)")
            ax.set_ylabel("Winkel (Grad)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            plt.savefig("temp_graph.png")
            
            # 3. Gemini Feedback
# 3. Gemini Feedback
            final_api_key = api_key_input
            
            if final_api_key:
                st.subheader("🤖 KI-Coach Empfehlung")
                with st.spinner('Gemini analysiert das Diagramm...'):
                    try:
                        genai.configure(api_key=final_api_key)
                        model_name = 'gemini-2.0-flash' 

                        # Sırayla modelleri dener, hangisi çalışırsa onu kullanır.
                        model = None
                        model = genai.GenerativeModel(model_name)
                        # Eğer hiçbir model çalışmazsa hata ver
                        if model is None:
                            st.error("Kein passendes KI-Modell gefunden.")
                        else:
                            img = Image.open("temp_graph.png")
                            
                            prompt = f"""
                            Du bist ein professioneller Sporttrainer.
                            Diese Grafik zeigt die Kniebeugen (Squats) Leistung. Total: {count} Wiederholungen.
                            
                            Analysiere auf DEUTSCH:
                            1. Wurde die 90-Grad-Tiefe erreicht? (Blaue vs Grüne Linie).
                            2. Gibt es Ermüdungserscheinungen?
                            3. Kurzer, motivierender Rat.
                            """
                            response = model.generate_content([prompt, img])
                            st.markdown(response.text)
                            
                    except Exception as e:
                        st.error(f"KI-Verbindungsfehler: {e}")
            else:
                st.warning("⚠️ Bitte API-Key eingeben.")












