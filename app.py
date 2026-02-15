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

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="InstruNet AI", layout="wide")

# ----------------------------
# Load TFLite Model
# ----------------------------
interpreter = tf.lite.Interpreter(model_path="instruNet_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_map = ["flute", "trumpet", "violin"]

# ----------------------------
# Prediction Function
# ----------------------------
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

# ----------------------------
# Header
# ----------------------------
st.markdown("# 🎶 InstruNet AI - Music Instrument Recognition")
st.markdown("Analyze. Discover. Classify instruments from audio.")

st.divider()

# ----------------------------
# Layout Columns
# ----------------------------
col1, col2, col3 = st.columns([1,2,1])

# ----------------------------
# Upload Section (Left)
# ----------------------------
with col1:
    st.subheader("Upload Audio")
    uploaded_file = st.file_uploader("Choose WAV File", type=["wav"])

# ----------------------------
# Main Analysis Section (Center)
# ----------------------------
if uploaded_file:
    confidences, audio, sr = predict_instrument(uploaded_file)
    pred_idx = np.argmax(confidences)
    pred_label = label_map[pred_idx]

    with col2:
        st.subheader("Analysis Results")

        # Waveform
        fig, ax = plt.subplots(figsize=(6,3))
        librosa.display.waveshow(audio, sr=sr, ax=ax)
        st.pyplot(fig)

        # Spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig2, ax2 = plt.subplots(figsize=(6,3))
        librosa.display.specshow(mel_db, sr=sr, cmap="magma", ax=ax2)
        st.pyplot(fig2)

# ----------------------------
# Confidence + Timeline (Right)
# ----------------------------
    with col3:
        st.subheader("Detected Instruments")

        for i, label in enumerate(label_map):
            st.write(f"{label.capitalize()}")
            st.progress(float(confidences[i]))

        st.subheader("Instrument Timeline")

        timeline = np.sin(np.linspace(0, 3*np.pi, 200)) * confidences[pred_idx] + confidences[pred_idx]
        fig3, ax3 = plt.subplots(figsize=(4,2))
        ax3.plot(timeline)
        ax3.set_yticks([])
        st.pyplot(fig3)

# ----------------------------
# Export Buttons
# ----------------------------
    st.divider()
    st.subheader("Export Report")

    report_data = {
        "file": uploaded_file.name,
        "prediction": pred_label,
        "confidence_scores": {
            label_map[i]: float(confidences[i]) for i in range(len(label_map))
        },
        "timestamp": str(datetime.now())
    }

    # JSON Export
    st.download_button(
        "Download JSON",
        json.dumps(report_data, indent=4),
        file_name="instrument_report.json"
    )

    # PDF Export
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="InstruNet AI - Instrument Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Prediction: {pred_label}", ln=True)

    for i, label in enumerate(label_map):
        pdf.cell(200, 10, txt=f"{label}: {confidences[i]*100:.2f}%", ln=True)

    pdf.output("report.pdf")

    with open("report.pdf", "rb") as f:
        st.download_button(
            "Download PDF",
            f,
            file_name="instrument_report.pdf"
        )
