import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import google.generativeai as genai
from PIL import Image
import os

# --- SEITENEINSTELLUNGEN ---
st.set_page_config(
    page_title="FormFlow AI - Pro Analyst",
    page_icon="🔬",
    layout="wide"
)

# --- BAŞLIK ---
st.title("🔬 FormFlow AI: Professionelle Analyse")
st.markdown("""
**Deep-Dive Biomechanik:** Das System kombiniert Computer-Vision-Daten mit Generativer KI für ein Feedback auf Profi-Niveau.
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Einstellungen")
    
    if "GOOGLE_API_KEY" in st.secrets:
        api_key_input = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ API-Key geladen.")
    else:
        api_key_input = st.text_input("Google Gemini API-Schlüssel", type="password")
    
    st.divider()
    st.info("Das System analysiert nun auch Konsistenz, Tempo und Symmetrie.")
    st.write("Dev: FormFlow Team")

# --- FONKSİYONLAR ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def detect_exercise_type(landmarks):
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    return "Squat" if height > width else "Push-Up"

def process_video(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    
    # İstatistikleri daha detaylı tutuyoruz
    stats = {
        "Squat": {"count": 0, "angles": [], "frames": [], "stage": None, "min_angles": []},
        "Push-Up": {"count": 0, "angles": [], "frames": [], "stage": None, "min_angles": []}
    }
    
    frame_count = 0
    
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
            current_exercise = detect_exercise_type(landmarks)
            angle = 0
            
            # --- AÇI HESAPLAMA ---
            if current_exercise == "Squat":
                p1 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                p2 = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                p3 = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                angle = calculate_angle(p1, p2, p3)
                
                # Mantık ve Min Açı Kaydı
                if angle > 160:
                    stats["Squat"]["stage"] = "UP"
                if angle < 90 and stats["Squat"]["stage"] == 'UP':
                    stats["Squat"]["stage"] = "DOWN"
                    stats["Squat"]["count"] += 1
                    stats["Squat"]["min_angles"].append(int(angle)) # Her tekrarın en derin noktasını kaydet
                
                stats["Squat"]["angles"].append(angle)
                stats["Squat"]["frames"].append(frame_count)
            
            elif current_exercise == "Push-Up":
                p1 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                p2 = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                p3 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle = calculate_angle(p1, p2, p3)
                
                if angle > 160:
                    stats["Push-Up"]["stage"] = "UP"
                if angle < 90 and stats["Push-Up"]["stage"] == 'UP':
                    stats["Push-Up"]["stage"] = "DOWN"
                    stats["Push-Up"]["count"] += 1
                    stats["Push-Up"]["min_angles"].append(int(angle))
                
                stats["Push-Up"]["angles"].append(angle)
                stats["Push-Up"]["frames"].append(frame_count)

        except:
            pass
            
        frame_count += 1
        if frame_count % 10 == 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            status_text.text(f"Analysiere Frame {frame_count}...")

    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    return stats

# --- HAUPTABLAUF ---
uploaded_file = st.file_uploader("Video hochladen (MP4)", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    st.video(video_path)
    
    if st.button("🚀 Detaillierte Analyse starten"):
        with st.spinner('KI berechnet Biomechanik...'):
            stats = process_video(video_path)
            st.success("Daten extrahiert. KI-Bericht wird erstellt...")
            
            # Grafik Hazırlama
            fig, ax = plt.subplots(figsize=(12, 5))
            
            has_squat = bool(stats["Squat"]["frames"])
            has_pushup = bool(stats["Push-Up"]["frames"])
            
            if has_squat:
                ax.plot(stats["Squat"]["frames"], stats["Squat"]["angles"], label='Squat', color='#007acc')
            if has_pushup:
                ax.plot(stats["Push-Up"]["frames"], stats["Push-Up"]["angles"], label='Push-Up', color='#ff7f0e')

            ax.axhline(y=90, color='green', linestyle='--', label='Ideal (90°)')
            ax.set_title("Bewegungsamplitude (Range of Motion)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.savefig("graph_detailed.png")

            # --- GELİŞMİŞ GEMINI PROMPT ---
            final_api_key = api_key_input
            if final_api_key:
                st.subheader("🔬 Wissenschaftliche Analyse")
                with st.spinner('Der KI-Sportwissenschaftler schreibt den Bericht...'):
                    try:
                        genai.configure(api_key=final_api_key)
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        img = Image.open("graph_detailed.png")
                        
                        # İstatistikleri Metne Dökme
                        data_summary = ""
                        if has_squat:
                            avg_depth = int(sum(stats["Squat"]["min_angles"])/len(stats["Squat"]["min_angles"])) if stats["Squat"]["min_angles"] else 0
                            data_summary += f"\n- SQUATS: {stats['Squat']['count']} Wdh. Durchschnittliche Tiefe: {avg_depth} Grad."
                        if has_pushup:
                            avg_depth = int(sum(stats["Push-Up"]["min_angles"])/len(stats["Push-Up"]["min_angles"])) if stats["Push-Up"]["min_angles"] else 0
                            data_summary += f"\n- PUSH-UPS: {stats['Push-Up']['count']} Wdh. Durchschnittliche Tiefe: {avg_depth} Grad."

                        # PROMPT MÜHENDİSLİĞİ BURADA
                        prompt = f"""
                        Du bist ein leitender Sportwissenschaftler für olympische Athleten.
                        Hier sind die gemessenen Daten aus dem Computer-Vision-System:
                        {data_summary}
                        
                        Aufgabe: Analysiere die beigefügte Grafik und die Daten extrem präzise.
                        Antworte strukturiert auf DEUTSCH in diesem Format:
                        
                        ### 1. 📏 Qualität & Range of Motion (ROM)
                        - Bewerte die Tiefe basierend auf den Daten (Ziel ist <90 Grad).
                        - Vergleiche die erste und die letzte Wiederholung in der Grafik (Ermüdung?).
                        
                        ### 2. ⏱️ Rhythmus & Konsistenz
                        - Ist die Kurve gleichmäßig oder zittrig? (Zittern deutet auf Instabilität hin).
                        - War das Tempo konstant?
                        
                        ### 3. ⚠️ Verletzungsrisiko
                        - Gibt es plötzliche Einbrüche in der Kurve?
                        - Bewertung der Sicherheit (Hoch/Mittel/Niedrig).
                        
                        ### 4. 🚀 Profi-Tipp zur Optimierung
                        - Ein konkreter biomechanischer Rat zur Verbesserung.
                        
                        Tonfall: Sachlich, wissenschaftlich, motivierend.
                        """
                        
                        response = model.generate_content([prompt, img])
                        st.markdown(response.text)
                        
                    except Exception as e:
                        st.error(f"KI-Fehler: {e}")
