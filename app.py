import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import google.generativeai as genai
from PIL import Image
import os
import io  # NEU: Für die Verarbeitung von Bildern im Arbeitsspeicher

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="FormFlow AI - Global Trainer",
    page_icon="🌍",
    layout="wide"
)

# --- DİL SÖZLÜĞÜ (SADECE BAYRAKLAR) ---
TRANSLATIONS = {
    "Deutsch": {
        "title": "🏋️ FormFlow AI: Auto-Modus",
        "desc": "**KI-gestützte biomechanische Analyse:** Laden Sie ein Video hoch – die KI erkennt automatisch Ihre Übung und analysiert Ihre Technik.",
        "sidebar_header": "⚙️ Einstellungen",
        "api_success": "✅ API-Key geladen.",
        "api_manual": "Manuelle Eingabe aktiv.",
        "info_text": "ℹ️ Das System erkennt die Übung automatisch anhand Ihrer Körperhaltung.",
        "upload_label": "Video zur Analyse hochladen (MP4)",
        "btn_start": "🚀 Analyse starten",
        "spinner_calc": "KI berechnet Biomechanik...",
        "success_complete": "Analyse abgeschlossen!",
        "detect_unknown": "Keine Person erkannt oder Übung unbekannt.",
        "metric_reps": "Wiederholungen",
        "metric_angle": "Min. Winkel",
        "chart_title": "Biomechanische Analyse",
        "chart_x": "Zeit (Frames)",
        "chart_y": "Winkel (Grad)",
        "ai_header": "🤖 KI-Coach Empfehlung",
        "spinner_ai": "Der KI-Coach schreibt den Bericht...",
        "warning_api": "⚠️ Bitte API-Schlüssel eingeben.",
        "exercise_names": {"Squat": "Kniebeugen (Squat)", "Push-Up": "Liegestütze (Push-Up)"},
        "prompt_template": """
        Du bist ein leitender Sportwissenschaftler für olympische Athleten.
        Hier sind die gemessenen Daten aus dem Computer-Vision-System:
        {data_summary}
        
        Aufgabe: Analysiere die beigefügte Grafik und die Daten extrem präzise.
        Antworte strukturiert auf DEUTSCH in diesem Format:
        
        ### 1. 📏 Qualität & Range of Motion (ROM)
        - Bewerte die Tiefe basierend auf den Daten (Ziel ist <90 Grad).
        - Vergleiche die erste und die letzte Wiederholung.
        
        ### 2. ⏱️ Rhythmus & Konsistenz
        - Ist die Kurve gleichmäßig oder zittrig?
        - War das Tempo konstant?
        
        ### 3. ⚠️ Verletzungsrisiko
        - Gibt es plötzliche Einbrüche in der Kurve?
        - Bewertung der Sicherheit (Hoch/Mittel/Niedrig).
        
        ### 4. 🚀 Profi-Tipp zur Optimierung
        - Ein konkreter biomechanischer Rat zur Verbesserung.
        
        Tonfall: Sachlich, wissenschaftlich, motivierend.
        """
    },
    "English": {
        "title": "🏋️ FormFlow AI: Auto-Mode",
        "desc": "**AI-Powered Biomechanical Analysis:** Upload a video – the AI automatically detects your exercise and analyzes your technique.",
        "sidebar_header": "⚙️ Settings",
        "api_success": "✅ API Key loaded.",
        "api_manual": "Manual entry active.",
        "info_text": "ℹ️ The system automatically detects the exercise based on your body posture.",
        "upload_label": "Upload video for analysis (MP4)",
        "btn_start": "🚀 Start Analysis",
        "spinner_calc": "AI is calculating biomechanics...",
        "success_complete": "Analysis complete!",
        "detect_unknown": "No person detected or exercise unknown.",
        "metric_reps": "Repetitions",
        "metric_angle": "Min. Angle",
        "chart_title": "Biomechanical Analysis",
        "chart_x": "Time (Frames)",
        "chart_y": "Angle (Degree)",
        "ai_header": "🤖 AI Coach Feedback",
        "spinner_ai": "AI Coach is writing the report...",
        "warning_api": "⚠️ Please enter API Key.",
        "exercise_names": {"Squat": "Squat", "Push-Up": "Push-Up"},
        "prompt_template": """
        You are a lead sports scientist for Olympic athletes.
        Here is the measured data from the Computer Vision system:
        {data_summary}
        
        Task: Analyze the attached graph and data with extreme precision.
        Answer structured in ENGLISH in this format:
        
        ### 1. 📏 Quality & Range of Motion (ROM)
        - Evaluate depth based on data (Target is <90 degrees).
        - Compare the first and last repetition.
        
        ### 2. ⏱️ Rhythm & Consistency
        - Is the curve smooth or shaky?
        - Was the tempo constant?
        
        ### 3. ⚠️ Injury Risk
        - Are there sudden drops in the curve?
        - Safety Rating (High/Medium/Low).
        
        ### 4. 🚀 Pro Tip for Optimization
        - One concrete biomechanical advice to improve.
        
        Tone: Professional, scientific, motivating.
        """
    },
    "Türkçe": {
        "title": "🏋️ FormFlow AI: Otomatik Mod",
        "desc": "**Yapay Zeka Destekli Biyomekanik Analiz:** Videonuzu yükleyin, yapay zeka hareketinizi otomatik tanısın ve tekniğinizi analiz etsin.",
        "sidebar_header": "⚙️ Ayarlar",
        "api_success": "✅ API Anahtarı yüklendi.",
        "api_manual": "Manuel giriş yapılıyor.",
        "info_text": "ℹ️ Sistem, vücut duruşunuza göre hareketi otomatik algılar.",
        "upload_label": "Analiz için video yükle (MP4)",
        "btn_start": "🚀 Analizi Başlat",
        "spinner_calc": "YZ Biyomekaniği hesaplıyor...",
        "success_complete": "Analiz tamamlandı!",
        "detect_unknown": "Kişi bulunamadı veya hareket tanınamadı.",
        "metric_reps": "Tekrar Sayısı",
        "metric_angle": "Min. Açı",
        "chart_title": "Biyomekanik Analiz",
        "chart_x": "Zaman (Kare)",
        "chart_y": "Açı (Derece)",
        "ai_header": "🤖 Yapay Zeka Koç Tavsiyesi",
        "spinner_ai": "YZ Koç raporu yazıyor...",
        "warning_api": "⚠️ Lütfen API Anahtarı giriniz.",
        "exercise_names": {"Squat": "Squat (Çömelme)", "Push-Up": "Şınav (Push-Up)"},
        "prompt_template": """
        Sen olimpik sporcular için çalışan baş spor bilimcisisin.
        İşte Bilgisayarlı Görü sisteminden gelen ölçüm verileri:
        {data_summary}
        
        Görev: Ekli grafiği ve verileri son derece hassas bir şekilde analiz et.
        Cevabını TÜRKÇE olarak şu formatta ver:
        
        ### 1. 📏 Kalite & Hareket Açıklığı (ROM)
        - Verilere dayanarak derinliği değerlendir (Hedef <90 derece).
        - İlk ve son tekrarı karşılaştır (Yorgunluk var mı?).
        
        ### 2. ⏱️ Ritim & Tutarlılık
        - Eğri pürüzsüz mü yoksa titrek mi? (Titreme kas yetmezliğini gösterir).
        - Tempo sabit miydi?
        
        ### 3. ⚠️ Sakatlanma Riski
        - Grafikte ani düşüşler var mı?
        - Güvenlik Derecelendirmesi (Yüksek/Orta/Düşük).
        
        ### 4. 🚀 Gelişim İçin Profesyonel İpucu
        - Tekniği geliştirmek için tek bir somut biyomekanik tavsiye ver.
        
        Ton: Profesyonel, bilimsel, motive edici.
        """
    }
}

# --- DİL SEÇİMİ ---
language_options = list(TRANSLATIONS.keys())
language = st.sidebar.selectbox("Language", language_options, index=0)
t = TRANSLATIONS[language]

# --- BAŞLIK VE AÇIKLAMA ---
st.title(t["title"])
st.markdown(t["desc"])

# --- SIDEBAR (AYARLAR) ---
with st.sidebar:
    st.header(t["sidebar_header"])
    
    if "GOOGLE_API_KEY" in st.secrets:
        api_key_input = st.secrets["GOOGLE_API_KEY"]
        st.success(t["api_success"])
    else:
        api_key_input = st.text_input("Google Gemini API Key", type="password")
        st.info(t["api_manual"])
    
    st.divider()
    st.info(t["info_text"])
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
    # Einfache Heuristik: Ist die Bounding Box höher als breit? -> Squat, sonst PushUp
    return "Squat" if height > width else "Push-Up"

def process_video(video_path):
    mp_pose = mp.solutions.pose
    stats = {
        "Squat": {"count": 0, "angles": [], "frames": [], "stage": None, "min_angles": []},
        "Push-Up": {"count": 0, "angles": [], "frames": [], "stage": None, "min_angles": []}
    }
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return stats

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0: total_frames = 1
    
    frame_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # OPTIMIERUNG: Context Manager für MediaPipe Pose (besseres Speichermanagement)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False # Performance-Boost
            results = pose.process(image)
            image.flags.writeable = True
            
            # WICHTIG: Prüfen ob Landmarks existieren, statt generisches try/except
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                current_exercise = detect_exercise_type(landmarks)
                angle = 0
                
                if current_exercise == "Squat":
                    p1 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    p2 = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    p3 = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    angle = calculate_angle(p1, p2, p3)
                    
                    if angle > 160:
                        stats["Squat"]["stage"] = "UP"
                    if angle < 90 and stats["Squat"]["stage"] == 'UP':
                        stats["Squat"]["stage"] = "DOWN"
                        stats["Squat"]["count"] += 1
                        stats["Squat"]["min_angles"].append(int(angle))
                    
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

            frame_count += 1
            if frame_count % 10 == 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))
                status_text.text(f"Processing... Frame: {frame_count} / {total_frames}")

    cap.release()
    progress_bar.empty()
    status_text.empty()
    return stats

# --- HAUPTABLAUF ---
uploaded_file = st.file_uploader(t["upload_label"], type=["mp4", "mov"])

if uploaded_file is not None:
    # Temporäre Datei erstellen
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    tfile.close() # Datei schließen, damit andere Prozesse darauf zugreifen können
    
    try:
        st.video(video_path)
        
        if st.button(t["btn_start"]):
            with st.spinner(t["spinner_calc"]):
                stats = process_video(video_path)
                st.success(t["success_complete"])
                
                # Grafik Hazırlama
                fig, ax = plt.subplots(figsize=(12, 5))
                
                has_squat = bool(stats["Squat"]["frames"])
                has_pushup = bool(stats["Push-Up"]["frames"])
                
                if not has_squat and not has_pushup:
                    st.error(t["detect_unknown"])
                else:
                    if has_squat:
                        ax.plot(stats["Squat"]["frames"], stats["Squat"]["angles"], label=t["exercise_names"]["Squat"], color='#007acc')
                    if has_pushup:
                        ax.plot(stats["Push-Up"]["frames"], stats["Push-Up"]["angles"], label=t["exercise_names"]["Push-Up"], color='#ff7f0e')

                    ax.axhline(y=90, color='green', linestyle='--', label='Target (90°)')
                    ax.set_title(t["chart_title"])
                    ax.set_xlabel(t["chart_x"])
                    ax.set_ylabel(t["chart_y"])
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # OPTIMIERUNG: Grafik in BytesIO speichern statt auf Disk
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    img = Image.open(buf)
                    
                    # --- GEMINI ANALİZİ ---
                    final_api_key = api_key_input
                    if final_api_key:
                        st.subheader(t["ai_header"])
                        with st.spinner(t["spinner_ai"]):
                            try:
                                genai.configure(api_key=final_api_key)
                                model = genai.GenerativeModel('gemini-2.0-flash')
                                
                                # Verileri Özetle
                                data_summary = ""
                                if has_squat:
                                    # Sicherstellen, dass nicht durch 0 geteilt wird
                                    count_sq = len(stats["Squat"]["min_angles"])
                                    avg_depth = int(sum(stats["Squat"]["min_angles"])/count_sq) if count_sq > 0 else 0
                                    data_summary += f"\n- {t['exercise_names']['Squat']}: {stats['Squat']['count']} {t['metric_reps']}. Avg Depth: {avg_depth}°."
                                if has_pushup:
                                    count_pu = len(stats["Push-Up"]["min_angles"])
                                    avg_depth = int(sum(stats["Push-Up"]["min_angles"])/count_pu) if count_pu > 0 else 0
                                    data_summary += f"\n- {t['exercise_names']['Push-Up']}: {stats['Push-Up']['count']} {t['metric_reps']}. Avg Depth: {avg_depth}°."

                                prompt = t["prompt_template"].format(data_summary=data_summary)
                                
                                response = model.generate_content([prompt, img])
                                st.markdown(response.text)
                                
                            except Exception as e:
                                st.error(f"AI Error: {e}")
                    else:
                        st.warning(t["warning_api"])
                        
    finally:
        # AUFRÄUMEN: Temporäre Videodatei löschen
        if os.path.exists(video_path):
            os.unlink(video_path)
