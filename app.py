import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import json
from datetime import datetime
from fpdf import FPDF
from tflite_runtime.interpreter import Interpreter

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="InstruNet AI", layout="wide")

# --------------------------------------------------
# LOAD TFLITE MODEL
# --------------------------------------------------
interpreter = Interpreter(model_path="instruNet_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_map = ["Flute", "Trumpet", "Violin"]

# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------
def predict_instrument(audio_file):
    audio, sr = sf.read(audio_file)

    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Resample
    audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)

    # Fix length to 3 seconds
    max_len = 22050 * 3
    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        audio = np.pad(audio, (0, max_len - len(audio)))

    # Create Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = np.resize(mel_db, (128, 128)).astype(np.float32)
    mel_db = mel_db.reshape(1, 128, 128, 1)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], mel_db)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    return output, audio, 22050

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🎶 InstruNet - Instrument Recognition System")
st.write("Upload a WAV file to analyze and classify musical instruments.")
st.divider()

# --------------------------------------------------
# LAYOUT (3 COLUMN DASHBOARD)
# --------------------------------------------------
left, center, right = st.columns([1,2,1])

# --------------------------------------------------
# LEFT PANEL (Upload)
# --------------------------------------------------
with left:
    st.subheader("📤 Upload Audio")
    uploaded_file = st.file_uploader("Choose WAV file", type=["wav"])

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        st.success("Audio Loaded")

# --------------------------------------------------
# PROCESS AFTER UPLOAD
# --------------------------------------------------
if uploaded_file:

    confidences, audio, sr = predict_instrument(uploaded_file)
    pred_idx = np.argmax(confidences)
    pred_label = label_map[pred_idx]

    # --------------------------------------------------
    # CENTER PANEL (Spectrogram + Confidence)
    # --------------------------------------------------
    with center:
        st.subheader("📊 Spectrogram")

        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig, ax = plt.subplots(figsize=(7,3))
        ax.imshow(mel_db, aspect='auto', origin='lower', cmap="magma")
        ax.set_title("Mel Spectrogram")
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel Frequency")
        st.pyplot(fig)

        st.subheader("🎚 Confidence Scores")

        for i, label in enumerate(label_map):
            st.write(f"{label}")
            st.progress(float(confidences[i]))

    # --------------------------------------------------
    # RIGHT PANEL (Detected + Timeline)
    # --------------------------------------------------
    with right:
        st.subheader("🎼 Detected Instrument")
        st.success(f"{pred_label}")

        st.subheader("📈 Instrument Timeline")

        timeline = np.sin(np.linspace(0, 3*np.pi, 200)) * confidences[pred_idx] + confidences[pred_idx]

        fig2, ax2 = plt.subplots(figsize=(4,2))
        ax2.plot(timeline)
        ax2.set_title("Intensity Over Time")
        ax2.set_yticks([])
        st.pyplot(fig2)

    # --------------------------------------------------
    # EXPORT SECTION
    # --------------------------------------------------
    st.divider()
    st.subheader("📥 Export Report")

    report = {
        "audio_file": uploaded_file.name,
        "prediction": pred_label,
        "confidence_scores": {
            label_map[i]: float(confidences[i]) for i in range(len(label_map))
        },
        "timestamp": str(datetime.now())
    }

    # JSON EXPORT
    st.download_button(
        label="Download JSON Report",
        data=json.dumps(report, indent=4),
        file_name="instrument_report.json",
        mime="application/json"
    )

    # PDF EXPORT
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200,10,"InstruNet AI Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)

    pdf.cell(200,10,f"Prediction: {pred_label}", ln=True)

    for i, label in enumerate(label_map):
        pdf.cell(200,10,f"{label}: {confidences[i]*100:.2f}%", ln=True)

    pdf.output("report.pdf")

    with open("report.pdf", "rb") as f:
        st.download_button(
            label="Download PDF Report",
            data=f,
            file_name="instrument_report.pdf",
            mime="application/pdf"
        )
