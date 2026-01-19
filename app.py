import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import tensorflow as tf
import json
from fpdf import FPDF
from datetime import datetime

# ================ PAGE CONFIG ===================
st.set_page_config(
    page_title="InstruNet - Instrument Recognition",
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================ ADVANCED CSS ===================
st.markdown("""
<style>
/* Global Styling */
body, [class*="st-"] {
    font-family: 'Segoe UI', sans-serif;
}

/* Card Styling */
.card {
    padding: 15px 20px;
    border-radius: 12px;
    background-color: #0f1116;
    border: 1px solid #272b33;
    box-shadow: 0 4px 12px rgba(0,255,255,0.08);
    margin-bottom: 20px;
}

/* Title Styling */
h1 {
    font-weight: 800;
    color: cyan !important;
}

/* Subtitle */
.subtitle {
    color: #9aa0a6;
    font-size: 18px;
    margin-bottom: 25px;
    text-align: center;
}

/* Progress Bar */
.stProgress > div > div > div > div {
    background-color: cyan;
}

/* Footer */
.footer {
    margin-top: 35px;
    padding: 10px;
    text-align: center;
    color: #787878;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ================ HEADER ===================
st.markdown("<h1 style='text-align:center;'>🎶 InstruNet Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload • Analyze • Discover Musical Patterns</p>", unsafe_allow_html=True)

# ================ LOAD MODEL ===================
interpreter = tf.lite.Interpreter(model_path="instruNet_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
label_map = ["flute", "trumpet", "violin"]

# ================ PREDICTION FUNCTION ===================
def predict_instrument(audio_file):
    audio, sr = sf.read(audio_file)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
    audio = audio[:22050*3] if len(audio) > 22050*3 else np.pad(audio, (0, max(0, 22050*3 - len(audio))))

    mel = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.resize(mel_db, (128, 128)).astype(np.float32)
    mel_db = mel_db.reshape(1, 128, 128, 1)

    interpreter.set_tensor(input_details[0]['index'], mel_db)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    return output, audio, 22050

# ================ PDF EXPORT ===================
def generate_pdf(filename, pred_label, conf):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)

    pdf.cell(200, 10, txt="Instrument Recognition Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Predicted Instrument: {pred_label}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt="Confidence Scores:", ln=True)

    for i, inst in enumerate(label_map):
        pdf.cell(200, 8, txt=f"{inst}: {conf[i]*100:.2f}%", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Export Timestamp: {datetime.now()}", ln=True)
    pdf.output(filename)

# ================ UPLOAD UI ===================
st.sidebar.header("📁 Upload Audio")
uploaded_file = st.sidebar.file_uploader("Choose .wav file", type=["wav"])

if uploaded_file:
    st.sidebar.audio(uploaded_file, format="audio/wav")
    analyze_btn = st.sidebar.button("🔍 Analyze Track")

    if analyze_btn:
        conf, audio, sr = predict_instrument(uploaded_file)
        pred_idx = np.argmax(conf)
        pred_label = label_map[pred_idx]

        # ======= RESULT CARD =======
        st.markdown(f"<div class='card'><h3>🎯 Detected Instrument: <span style='color:cyan;'>{pred_label.upper()}</span></h3></div>", unsafe_allow_html=True)

        # ======= CONFIDENCE CARDS =======
        st.subheader("📊 Confidence Levels")
        for i, lbl in enumerate(label_map):
            st.write(f"**{lbl.upper()}**")
            st.progress(float(conf[i]))

        # ======= TWO COLUMN VISUALS =======
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Audio Waveform")
            fig, ax = plt.subplots(figsize=(6,3))
            librosa.display.waveshow(audio, sr=sr, ax=ax, color='cyan')
            st.pyplot(fig)

        with col2:
            st.subheader("🎛 Mel-Spectrogram")
            fig2, ax2 = plt.subplots(figsize=(6,3))
            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            librosa.display.specshow(mel_db, sr=sr, ax=ax2, cmap="magma")
            st.pyplot(fig2)

        # ======= TIMELINE VISUAL =======
        st.subheader("⏱ Activation Timeline (Mock Visualization)")
        fig3, ax3 = plt.subplots(figsize=(10,2))
        mock_line = np.sin(np.linspace(0, 3*np.pi, 150)) * conf[pred_idx] + conf[pred_idx]
        ax3.plot(mock_line, color='cyan', linewidth=2)
        ax3.set_yticks([])
        ax3.set_title(f"Timeline Activation: {pred_label.upper()}")
        st.pyplot(fig3)

        # ======= EXPORT SECTION =======
        st.subheader("📤 Export Options")

        # JSON Export
        report_data = {
            "audio_file": uploaded_file.name,
            "prediction": pred_label,
            "confidence_scores": {label_map[i]: float(conf[i]) for i in range(len(label_map))},
            "timestamp": str(datetime.now())
        }
        json_filename = uploaded_file.name.replace(".wav", "_report.json")

        st.download_button("📁 Download JSON", json.dumps(report_data, indent=4), file_name=json_filename)

        # PDF Export
        pdf_filename = uploaded_file.name.replace(".wav", "_report.pdf")
        generate_pdf(pdf_filename, pred_label, conf)

        with open(pdf_filename, "rb") as f:
            st.download_button("📄 Download PDF", data=f, file_name=pdf_filename, mime="application/pdf")

# ================ FOOTER ===================
st.markdown("<div class='footer'>InstruNet © 2026 • AI-based Music Instrument Recognition Dashboard</div>", unsafe_allow_html=True)
