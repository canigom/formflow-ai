import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import google.generativeai as genai
from PIL import Image
import os

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="FormFlow AI - Auto Trainer",
    page_icon="🧠",
    layout="wide"
)

# --- BAŞLIK ---
st.title("🧠 FormFlow AI: Auto-Mode")
st.markdown("""
**Tam Otomatik Biyomekanik Analiz:** Video yükleyin, yapay zeka hangi hareketi yaptığınızı **kendisi anlasın** ve analiz etsin.
""")

# --- SIDEBAR (YAN MENÜ) ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    # API KEY YÖNETİMİ
    if "GOOGLE_API_KEY" in st.secrets:
        api_key_input = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ API-Key sistemden yüklendi.")
    else:
        api_key_input = st.text_input("Google Gemini API-Key", type="password")
        st.info("Manuel giriş yapılıyor.")
    
    st.divider()
    st.info("ℹ️ Sistem, vücudunuzun duruşuna göre (Dikey/Yatay) hareketi otomatik algılar.")
    st.write("Dev: FormFlow Team")

# --- FONKSİYONLAR ---
def calculate_angle(a, b, c):
    """3 nokta arasındaki açıyı hesaplar"""
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
    Vücudun en boy oranına bakarak hareketi tahmin eder.
    """
    # Tüm noktaların x ve y koordinatlarını al
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # EĞER YÜKSEKLİK > GENİŞLİK --> SQUAT (Ayakta)
    # EĞER GENİŞLİK > YÜKSEKLİK --> PUSH-UP (Yerde)
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
    
    # Hareketi henüz bilmiyoruz
    detected_exercise = "Bilinmiyor"
    
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
            
            # --- 1. OTOMATİK TESPİT (İlk 10 karede karar verilir) ---
            # Videonun başında hareketi anlamaya çalışır
            if frame_count == 10: 
                detected_exercise = detect_exercise_type(landmarks)
                st.toast(f"Hareket Algılandı: {detected_exercise} 🏃", icon="✅")

            # --- 2. HAREKETE GÖRE AÇI SEÇİMİ ---
            angle = 0
            
            if detected_exercise == "Squat":
                # SQUAT: Kalça - Diz - Bilek
                p1 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                p2 = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                p3 = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                angle = calculate_angle(p1, p2, p3)
                
            elif detected_exercise == "Push-Up":
                # PUSH-UP: Omuz - Dirsek - Bilek
                p1 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                p2 = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                p3 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle = calculate_angle(p1, p2, p3)
            
            # --- 3. KAYIT VE SAYMA ---
            # Sadece hareket tespit edildiyse kaydet
            if detected_exercise != "Bilinmiyor":
                angle_history.append(angle)
                frame_indices.append(frame_count)
                
                # Ortak Sayma Mantığı (Squat ve Şınav benzer çalışır)
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
            status_text.text(f"Video işleniyor... Kare: {frame_count}")

    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    return angle_history, frame_indices, count, detected_exercise

# --- ANA AKIŞ ---
uploaded_file = st.file_uploader("Analiz için Video Yükle (MP4)", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    st.video(video_path)
    
    if st.button("🚀 Otomatik Analizi Başlat"):
        with st.spinner('Yapay Zeka hareketi algılıyor ve analiz ediyor...'):
            
            # Fonksiyonu çağır (Artık hareket tipi göndermiyoruz, o bize söylüyor)
            angles, frames, count, detected_type = process_video(video_path)
            
            if detected_type == "Bilinmiyor":
                st.error("Videoda insan tespit edilemedi veya hareket anlaşılamadı.")
            else:
                st.success(f"Analiz Tamamlandı! Tespit Edilen Hareket: **{detected_type}**")
                
                # 1. Metrikler
                col1, col2 = st.columns(2)
                col1.metric("Tekrar Sayısı", f"{count}", detected_type)
                
                if angles:
                    label = "Min. Diz Açısı" if detected_type == "Squat" else "Min. Dirsek Açısı"
                    col1.metric(label, f"{int(min(angles))}°", "Derece")
                
                # 2. Grafik
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(frames, angles, label='Açı Değişimi', color='#007acc')
                ax.axhline(y=90, color='green', linestyle='--', label='Hedef (90°)')
                ax.axhline(y=160, color='red', linestyle='--', label='Başlangıç (160°)')
                ax.set_title(f"Biyomekanik Analiz: {detected_type}")
                ax.set_xlabel("Zaman (Kare)")
                ax.set_ylabel("Açı (Derece)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.savefig("temp_graph.png")
                
                # 3. Gemini Feedback (Otomatik Prompt)
                final_api_key = api_key_input
                
                if final_api_key:
                    st.subheader("🤖 Yapay Zeka Koç Tavsiyesi")
                    with st.spinner('Gemini yorumluyor...'):
                        try:
                            genai.configure(api_key=final_api_key)
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            img = Image.open("temp_graph.png")
                            
                            prompt = f"""
                            Sen profesyonel bir spor antrenörüsün.
                            Kullanıcı şu hareketi yaptı: {detected_type}.
                            Toplam Tekrar: {count}.
                            Grafik verilerine bakarak:
                            1. Derinlik yeterli mi? (90 derece çizgisine inilmiş mi?)
                            2. Performans düşüklüğü (yorgunluk) var mı?
                            3. {detected_type} için teknik bir tavsiye ver.
                            Cevabın Almanca olsun.
                            """
                            response = model.generate_content([prompt, img])
                            st.markdown(response.text)
                            
                        except Exception as e:
                            st.error(f"Yapay Zeka Hatası: {e}")
                else:
                    st.warning("⚠️ Lütfen API Key giriniz.")
