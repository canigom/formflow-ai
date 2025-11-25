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
    page_title="FormFlow AI - Akıllı Antrenör",
    page_icon="🏋️",
    layout="wide"
)

# --- BAŞLIK VE AÇIKLAMA ---
st.title("🏋️ FormFlow AI")
st.markdown("""
**Yapay Zeka Destekli Biyomekanik Hareket Analizi** Videonuzu yükleyin, yapay zeka formunuzu analiz etsin ve size özel tavsiyeler versin.
""")

# --- SIDEBAR (YAN MENÜ) ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    api_key = "AIzaSyDucpNYIaL-LR57PjZWrLNDE4KtqAsS9fQ"
    st.divider()
    st.write("Geliştirici: FormFlow Team")

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

def process_video(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    
    angle_history = []
    frame_indices = []
    frame_count = 0
    squat_count = 0
    stage = None
    
    # Progress Bar (İlerleme Çubuğu)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Görüntü işleme
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
            
            # Squat Sayacı
            if angle > 160:
                stage = "UP"
            if angle < 90 and stage == 'UP':
                stage = "DOWN"
                squat_count += 1
                
        except:
            pass
            
        frame_count += 1
        # İlerlemeyi güncelle
        if frame_count % 10 == 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            status_text.text(f"Video İşleniyor... Kare: {frame_count}")

    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    return angle_history, frame_indices, squat_count

# --- ANA AKIŞ ---
uploaded_file = st.file_uploader("Analiz edilecek videoyu seçin (MP4)", type=["mp4", "mov"])

if uploaded_file is not None:
    # Videoyu geçici dosyaya kaydet
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    st.video(video_path) # Videoyu göster
    
    if st.button("🚀 Analizi Başlat"):
        with st.spinner('Yapay Zeka Biyomekaniği Hesaplanıyor...'):
            angles, frames, count = process_video(video_path)
            
            st.success("Analiz Tamamlandı!")
            
            # 1. Metrikler
            col1, col2 = st.columns(2)
            col1.metric("Toplam Tekrar", f"{count}", "Squat")
            col1.metric("Minimum Açı", f"{int(min(angles))}°", "Derinlik")
            
            # 2. Grafik Çizme
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(frames, angles, label='Diz Açısı', color='#007acc')
            ax.axhline(y=90, color='green', linestyle='--', label='Hedef (90°)')
            ax.axhline(y=160, color='red', linestyle='--', label='Başlangıç (160°)')
            ax.set_title("Hareket Analiz Grafiği")
            ax.set_xlabel("Zaman")
            ax.set_ylabel("Derece")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig) # Grafiği ekrana bas
            
            # Grafiği kaydet (Gemini için)
            plt.savefig("temp_graph.png")
            
            # 3. Gemini Yorumu
            if api_key:
                st.subheader("🤖 Yapay Zeka Koç Tavsiyesi")
                with st.spinner('Gemini Grafiği Yorumluyor...'):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        img = Image.open("temp_graph.png")
                        prompt = f"""
                        Sen profesyonel bir spor antrenörüsün. Bu grafik bir sporcunun Squat performansını gösteriyor.
                        Toplam {count} tekrar yapmış. Grafiğe bakarak:
                        1. Derinlik analizi yap (90 dereceye inebilmiş mi?).
                        2. Yorulma belirtisi var mı?
                        3. Kısa ve motive edici bir tavsiye ver.
                        Türkçe cevapla.
                        """
                        response = model.generate_content([prompt, img])
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Yapay Zeka Hatası: {e}")
            else:
                st.warning("Detaylı yapay zeka yorumu için lütfen sol menüden API Key giriniz.")