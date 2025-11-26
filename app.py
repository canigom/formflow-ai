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
    page_title="FormFlow AI - Auto Trainer",
    page_icon="🧠",
    layout="wide"
)

# --- TITEL UND BESCHREIBUNG ---
st.title("🧠 FormFlow AI: Auto-Modus")
st.markdown("""
**Vollautomatische biomechanische Analyse:** Laden Sie ein Video hoch – die KI erkennt automatisch Ihre Übung und analysiert Ihre Technik.
""")

# --- SIDEBAR (EINSTELLUNGEN) ---
with st.sidebar:
    st.header("⚙️ Einstellungen")
    
    # API-KEY VERWALTUNG
    if "GOOGLE_API_KEY" in st.secrets:
        api_key_input = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ API-Key vom System geladen.")
    else:
        api_key_input = st.text_input("Google Gemini API-Schlüssel", type="password")
        st.info("Manuelle Eingabe aktiv.")
    
    st.divider()
    st.info("ℹ️ Das System erkennt die Übung automatisch anhand Ihrer Körperhaltung (Stehend/Liegend).")
    st.write("Entwickler: FormFlow Team")

# --- FUNKTIONEN ---
def calculate_angle(a, b, c):
    """Berechnet den Winkel zwischen 3 Punkten"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def detect_exercise_type(landmarks):
    """
    Schätzt die Übung basierend auf dem Seitenverhältnis des Körpers.
    """
    # Alle x- und y-Koordinaten abrufen
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # WENN HÖHE > BREITE --> SQUAT (Stehend)
    # WENN BREITE > HÖHE --> PUSH-UP (Liegend)
    if height > width:
        return "Squat"
    else:
        return "Push-Up"

def process_video(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    
    angle_history = []
    frame_indices = []
    frame_count = 0
    count = 0
    stage = None
    
    # Übung ist noch unbekannt
    detected_exercise = "Unbekannt"
    
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
            
            # --- 1. AUTOMATISCHE ERKENNUNG (Entscheidung im 10. Frame) ---
            if frame_count == 10: 
                detected_exercise = detect_exercise_type(landmarks)
                st.toast(f"Übung erkannt: {detected_exercise} 🏃", icon="✅")

            # --- 2. WINKELAUSWAHL JE NACH ÜBUNG ---
            angle = 0
            
            if detected_exercise == "Squat":
                # SQUAT: Hüfte - Knie - Knöchel
                p1 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                p2 = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                p3 = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                angle = calculate_angle(p1, p2, p3)
                
            elif detected_exercise == "Push-Up":
                # PUSH-UP: Schulter - Ellenbogen - Handgelenk
                p1 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                p2 = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                p3 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle = calculate_angle(p1, p2, p3)
            
            # --- 3. SPEICHERN UND ZÄHLEN ---
            if detected_exercise != "Unbekannt":
                angle_history.append(angle)
                frame_indices.append(frame_count)
                
                # Zähl-Logik
                if angle > 160:
                    stage = "UP"
                if angle < 90 and stage == 'UP':
                    stage = "DOWN"
                    count += 1
                
        except:
            pass
            
        frame_count += 1
        if frame_count % 10 == 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            status_text.text(f"Video wird verarbeitet... Frame: {frame_count}")

    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    return angle_history, frame_indices, count, detected_exercise

# --- HAUPTABLAUF ---
uploaded_file = st.file_uploader("Video zur Analyse hochladen (MP4)", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    st.video(video_path)
    
    if st.button("🚀 Automatische Analyse starten"):
        with st.spinner('KI erkennt und analysiert die Bewegung...'):
            
            # Funktion aufrufen
            angles, frames, count, detected_type = process_video(video_path)
            
            if detected_type == "Unbekannt":
                st.error("Keine Person erkannt oder Übung konnte nicht identifiziert werden.")
            else:
                st.success(f"Analyse abgeschlossen! Erkannte Übung: **{detected_type}**")
                
                # 1. Metriken
                col1, col2 = st.columns(2)
                col1.metric("Wiederholungen", f"{count}", detected_type)
                
                if angles:
                    label = "Min. Kniewinkel" if detected_type == "Squat" else "Min. Ellenbogenwinkel"
                    col1.metric(label, f"{int(min(angles))}°", "Grad")
                
                # 2. Grafik
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(frames, angles, label='Winkelverlauf', color='#007acc')
                ax.axhline(y=90, color='green', linestyle='--', label='Ziel (90°)')
                ax.axhline(y=160, color='red', linestyle='--', label='Start (160°)')
                ax.set_title(f"Biomechanische Analyse: {detected_type}")
                ax.set_xlabel("Zeit (Frames)")
                ax.set_ylabel("Winkel (Grad)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.savefig("temp_graph.png")
                
                # 3. Gemini Feedback
                final_api_key = api_key_input
                
                if final_api_key:
                    st.subheader("🤖 KI-Coach Empfehlung")
                    with st.spinner('Gemini analysiert...'):
                        try:
                            genai.configure(api_key=final_api_key)

                            model = genai.GenerativeModel('gemini-2.0-flash')
                            img = Image.open("temp_graph.png")
                            
                            # Python'dan gelen matematiksel verileri hesapla
                            min_angle_val = int(min(angles)) if angles else 0
                            avg_angle_val = int(sum(angles)/len(angles)) if angles else 0
                            
                            # --- PROFESYONEL PROMPT (KOMUT) ---
                            prompt = f"""
                            Du bist ein erfahrener Sportwissenschaftler und Biomechanik-Experte für olympische Athleten.
                            
                            Hintergrunddaten zur Übung:
                            - Erkannte Übung: {detected_type}
                            - Anzahl der Wiederholungen: {count}
                            - Tiefster gemessener Winkel: {min_angle_val} Grad
                            - Durchschnittlicher Gelenkwinkel: {avg_angle_val} Grad
                            
                            Aufgabe:
                            Analysiere die beigefügte Grafik (Zeit vs. Winkel) und die Daten extrem detailliert.
                            Antworte strukturiert auf DEUTSCH in folgendem Format:
                            
                            ### 1. 📏 Bewegungsqualität & Tiefe (Range of Motion)
                            - Bewerte die Tiefe basierend auf dem tiefsten Winkel ({min_angle_val}°). 
                            - Ist das für einen {detected_type} biomechanisch optimal (Ziel: <90° für Squat)?
                            - Vergleiche die erste und die letzte Wiederholung in der Grafik.
                            
                            ### 2. 📉 Ermüdungsanalyse & Konsistenz
                            - Betrachte die Spitzen (Peaks) und Täler (Valleys) der blauen Linie.
                            - Sind alle Wiederholungen gleichmäßig (Konsistenz)?
                            - Gibt es "Zittern" (kleine Wellen in der Linie) oder wird die Bewegung langsamer (breitere Wellen)? Das deutet auf Muskelversagen hin.
                            
                            ### 3. ⚠️ Verletzungsrisiko & Fehler
                            - Gibt es plötzliche Einbrüche oder Pausen an der falschen Stelle?
                            - Bewerte das Risiko basierend auf der Stabilität der Kurve.
                            
                            ### 4. 🚀 Profi-Tipp zur Optimierung
                            - Gib EINEN konkreten, biomechanischen Tipp, um die Technik sofort zu verbessern.
                            - Empfiehl eine Hilfsübung (z.B. "Mehr Mobilitätstraining für die Hüfte").
                            
                            Tonfall: Professionell, motivierend, datenbasiert.
                            """
                            
                            response = model.generate_content([prompt, img])
                            st.markdown(response.text)
                            
                        except Exception as e:
                            st.error(f"KI-Fehler: {e}")
                else:
                    st.warning("⚠️ Bitte API-Schlüssel eingeben.")


