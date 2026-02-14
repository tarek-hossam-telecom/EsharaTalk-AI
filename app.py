import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from gtts import gTTS
import os
import tempfile
import math

# ─── إعداد MediaPipe ───────────────────────────────────────────────────────────
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

hand_landmarker = HandLandmarker.create_from_options(options)

# ─── واجهة Streamlit محترفة ────────────────────────────────────────────────────
st.set_page_config(page_title="EsharaTalk EG", layout="wide", page_icon="🖐️")

# CSS للشكل الجميل + اللوجو فوق يمين + حقوق تحت يمين
st.markdown("""
    <style>
    .logo-right {position: absolute; top: 15px; right: 30px; z-index: 999;}
    .copyright  {position: absolute; bottom: 15px; right: 30px; color:#00ff88; font-size:22px; font-weight:bold; z-index:999;}
    .title-ar   {font-size:48px; font-weight:bold; color:#00ff88; direction:rtl; text-align:center; margin:0;}
    .title-en   {font-size:32px; color:#ffffff; text-align:center; margin:5px 0;}
    .trans-box  {background:rgba(0,0,0,0.7); padding:25px; border-radius:15px; margin:20px auto; max-width:900px; text-align:center;}
    </style>
""", unsafe_allow_html=True)

# اللوجو (غير الرابط ده برابط لوجوك الحقيقي على imgur أو github)
st.markdown("""
    <div class="logo-right">
        <img src="https://i.imgur.com/YourLogoHere.png" width="240" alt="EsharaTalk EG">
    </div>
    <div class="copyright">© TҜ</div>
""", unsafe_allow_html=True)

st.markdown('<p class="title-ar">مترجم لغة الإشارة المصرية</p>', unsafe_allow_html=True)
st.markdown('<p class="title-en">Egyptian Sign Language Translator</p>', unsafe_allow_html=True)

run = st.checkbox("شغل الكاميرا / Start Camera", value=True)
frame_placeholder = st.empty()
trans_box = st.empty()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("الكاميرا مش شغالة! قفل أي برنامج تاني بيستخدمها.")
else:
    timestamp = 0
    last_speech = 0

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("مش قادر أقرا من الكاميرا")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = hand_landmarker.detect_for_video(mp_image, timestamp)
        timestamp += 33

        text_ar = "مش شايف إيد واضحة"
        text_en = "No clear hand"

        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]

            # رسم النقاط الخضراء + خطوط زرقاء
            for lm in hand_landmarks:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

            connections = [
                (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                (5,9),(9,13),(13,17),(9,10),(10,11),(11,12),
                (13,14),(14,15),(15,16),(17,18),(18,19),(19,20)
            ]
            for s, e in connections:
                start = hand_landmarks[s]
                end = hand_landmarks[e]
                cv2.line(frame,
                         (int(start.x * frame.shape[1]), int(start.y * frame.shape[0])),
                         (int(end.x * frame.shape[1]), int(end.y * frame.shape[0])),
                         (255, 0, 0), 5)

            # شروط محسّنة جدًا (أكثر دقة)
            thumb = hand_landmarks[4]
            index = hand_landmarks[8]
            middle = hand_landmarks[12]
            wrist = hand_landmarks[0]
            d_ti = math.hypot(thumb.x - index.x, thumb.y - index.y)
            d_tm = math.hypot(thumb.x - middle.x, thumb.y - middle.y)

            if thumb.y < 0.18 and d_ti > 0.12:
                text_ar = "شكرا!"
                text_en = "Thank you!"
            elif d_ti < 0.03 and middle.y > 0.58:
                text_ar = "لا!"
                text_en = "No!"
            elif d_tm < 0.045 and wrist.y > 0.62:
                text_ar = "نعم!"
                text_en = "Yes!"
            elif all(lm.y > 0.68 for lm in [thumb, index, middle]):
                text_ar = "كويس!"
                text_en = "Good!"
            elif d_ti < 0.07 and thumb.y < 0.38:
                text_ar = "حلو!"
                text_en = "Nice!"
            else:
                text_ar = "مرحبا!"
                text_en = "Hello!"

        # عرض الترجمة
        trans_box.markdown(
            f'<div class="trans-box"><p class="big-ar">{text_ar}</p><p class="big-en">{text_en}</p></div>',
            unsafe_allow_html=True
        )

        # نطق صوتي
        now = time.time()
        if now - last_speech > 2.8:
            tts = gTTS(text_ar, lang='ar')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                tts.save(fp.name)
                os.system(f"start {fp.name}")
            last_speech = now

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB")

    cap.release()
    st.success("تم الإيقاف بنجاح! / Stopped successfully!")