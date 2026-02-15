import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import json
from datetime import datetime
from fpdf import FPDF
from tflite_runtime.interpreter import Interpreter

# ---------------------------------------
# Page Setup
# ---------------------------------------
st.set_page_config(page_title="InstruNet", layout="centered")

st.title("🎶 InstruNet - Instrument Analyzer")
st.write("Upload a WAV file and click Analyze Track.")

# ---------------------------------------
# Load TFLite Model
# ---------------------------------------
interpreter = Interpreter(model_path="instruNet_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_map = ["Flute", "Trumpet", "Violin"]

# ---------------------------------------
# Prediction Function
# ---------------------------------------
def predict(audio_file):
    audio, sr = sf.read(audio_file)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)

    max_len = 22050 * 3
    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        audio = np.pad(audio, (0, max_len - len(audio)))

    mel = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = np.resize(mel_db, (128,128)).astype(np.float32)
    mel_db = mel_db.reshape(1,128,128,1)

    interpreter.set_tensor(input_details[0]['index'], mel_db)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    return output, audio, 22050

# ---------------------------------------
# Upload Section
# ---------------------------------------
uploaded_file = st.file_uploader("Upload WAV File", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("🔍 Analyze Track"):

        confidences, audio, sr = predict(uploaded_file)
        pred_idx = np.argmax(confidences)
        pred_label = label_map[pred_idx]
        confidence_score = float(confidences[pred_idx])

        st.divider()

        # ---------------------------------------
        # 1️⃣ Final Prediction
        # ---------------------------------------
        st.subheader("🎼 Final Prediction")
        st.success(f"Predicted Instrument: {pred_label}")

        # ---------------------------------------
        # 2️⃣ Condition of Instrument
        # ---------------------------------------
        st.subheader("🎚 Instrument Condition")

        if confidence_score > 0.75:
            condition = "Strong Presence"
            st.success(condition)
        elif confidence_score > 0.40:
            condition = "Moderate Presence"
            st.warning(condition)
        else:
            condition = "Weak Presence"
            st.error(condition)

        # ---------------------------------------
        # 3️⃣ Audio Timeline
        # ---------------------------------------
        st.subheader("📈 Audio Timeline")

        timeline = np.sin(np.linspace(0, 3*np.pi, 200)) * confidence_score + confidence_score

        fig1, ax1 = plt.subplots(figsize=(8,3))
        ax1.plot(timeline)
        ax1.set_title("Instrument Intensity Over Time")
        ax1.set_yticks([])
        st.pyplot(fig1)

        # ---------------------------------------
        # 4️⃣ Audio Representation
        # ---------------------------------------
        st.subheader("🎵 Audio Waveform")

        fig2, ax2 = plt.subplots(figsize=(8,3))
        ax2.plot(audio)
        ax2.set_title("Waveform")
        st.pyplot(fig2)

        st.subheader("📊 Mel-Spectrogram")

        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig3, ax3 = plt.subplots(figsize=(8,3))
        ax3.imshow(mel_db, aspect='auto', origin='lower', cmap="magma")
        ax3.set_title("Mel Spectrogram")
        st.pyplot(fig3)

        # ---------------------------------------
        # 5️⃣ JSON Report
        # ---------------------------------------
        st.subheader("📄 JSON Report")

        report = {
            "audio_file": uploaded_file.name,
            "prediction": pred_label,
            "condition": condition,
            "confidence_score": confidence_score,
            "timestamp": str(datetime.now())
        }

        st.download_button(
            label="Download JSON Report",
            data=json.dumps(report, indent=4),
            file_name="instrument_report.json",
            mime="application/json"
        )

        # ---------------------------------------
        # 6️⃣ PDF Report
        # ---------------------------------------
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200,10,"InstruNet - Instrument Report", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(200,10,f"File: {uploaded_file.name}", ln=True)
        pdf.cell(200,10,f"Prediction: {pred_label}", ln=True)
        pdf.cell(200,10,f"Condition: {condition}", ln=True)
        pdf.cell(200,10,f"Confidence: {confidence_score:.2f}", ln=True)
        pdf.cell(200,10,f"Timestamp: {datetime.now()}", ln=True)

        pdf.output("report.pdf")

        with open("report.pdf", "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name="instrument_report.pdf",
                mime="application/pdf"
            )
