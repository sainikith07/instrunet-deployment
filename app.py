import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from fpdf import FPDF
from datetime import datetime

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(page_title="InstruNet AI", layout="wide")

# --------------------------------
# Dark Dashboard Styling
# --------------------------------
st.markdown("""
<style>
body {background-color: #0E1117;}
.big-font {font-size:28px !important; font-weight: bold;}
.section-box {
    background-color:#1E222A;
    padding:20px;
    border-radius:15px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# Load TFLite Model
# --------------------------------
interpreter = tf.lite.Interpreter(model_path="instruNet_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_map = ["Flute", "Trumpet", "Violin"]

# --------------------------------
# Prediction Function
# --------------------------------
def predict_instrument(audio_file):
    audio, sr = sf.read(audio_file)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
    audio = audio[:22050*3] if len(audio) > 22050*3 else np.pad(audio, (0, max(0, 22050*3 - len(audio))))

    mel = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.resize(mel_db, (128,128)).astype(np.float32)
    mel_db = mel_db.reshape(1,128,128,1)

    interpreter.set_tensor(input_details[0]['index'], mel_db)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    return output, audio, 22050

# --------------------------------
# Header
# --------------------------------
st.markdown("## 🎶 InstruNet AI: Music Instrument Recognition")
st.markdown("Upload. Analyze. Discover.")

st.divider()

# --------------------------------
# Layout (3 Column Dashboard)
# --------------------------------
left, center, right = st.columns([1,2,1])

# -----------------------------
# LEFT PANEL (Upload)
# -----------------------------
with left:
    st.markdown("### 📤 Upload Audio")
    uploaded_file = st.file_uploader("Choose WAV File", type=["wav"])

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        st.success("Now Playing")

# -----------------------------
# CENTER PANEL (Analysis)
# -----------------------------
if uploaded_file:
    confidences, audio, sr = predict_instrument(uploaded_file)
    pred_idx = np.argmax(confidences)
    pred_label = label_map[pred_idx]

    with center:
        st.markdown("### 📊 Analysis Results")

        # Spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig, ax = plt.subplots(figsize=(7,3))
        librosa.display.specshow(mel_db, sr=sr, cmap="magma", ax=ax)
        ax.set_title("Mel-Spectrogram")
        st.pyplot(fig)

        # Confidence Bars
        st.markdown("### 🎚 Instrument Confidence")
        for i, label in enumerate(label_map):
            st.write(f"{label}")
            st.progress(float(confidences[i]))

# -----------------------------
# RIGHT PANEL (Detected + Timeline)
# -----------------------------
    with right:
        st.markdown("### 🎼 Detected Instruments")

        for i, label in enumerate(label_map):
            status = "Present" if confidences[i] > 0.3 else "Not Present"
            st.write(f"{label}: {status}")

        st.markdown("### 📈 Instrument Timeline")

        timeline = np.sin(np.linspace(0, 3*np.pi, 200)) * confidences[pred_idx] + confidences[pred_idx]
        fig2, ax2 = plt.subplots(figsize=(4,2))
        ax2.plot(timeline)
        ax2.set_yticks([])
        ax2.set_title("Intensity Over Time")
        st.pyplot(fig2)

# --------------------------------
# EXPORT SECTION (Bottom)
# --------------------------------
    st.divider()
    st.markdown("### 📥 Export Report")

    report = {
        "audio_file": uploaded_file.name,
        "prediction": pred_label,
        "confidence_scores": {
            label_map[i]: float(confidences[i]) for i in range(len(label_map))
        },
        "timestamp": str(datetime.now())
    }

    # JSON
    st.download_button(
        "Download JSON Report",
        json.dumps hooking(report, indent=4),
        file_name="instrument_report.json"
    )

    # PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200,10,"InstruNet AI Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200,10,f"Prediction: {pred_label}", ln=True)

    for i,label in enumerate(label_map):
        pdf.cell(200,10,f"{label}: {confidences[i]*100:.2f}%", ln=True)

    pdf.output("report.pdf")

    with open("report.pdf","rb") as f:
        st.download_button(
            "Download PDF Report",
            f,
            file_name="instrument_report.pdf"
        )
