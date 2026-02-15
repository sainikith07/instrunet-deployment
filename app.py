import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import json
from datetime import datetime
from fpdf import FPDF
from tflite_runtime.interpreter import Interpreter

st.set_page_config(page_title="InstruNet", layout="centered")

st.title("🎶 InstruNet - Instrument Analyzer")
st.write("Upload a WAV file and click Analyze Track.")

# Load TFLite model
interpreter = Interpreter(model_path="instruNet_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_map = ["Flute", "Trumpet", "Violin"]

def predict(audio_file):
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

    # Generate Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize to model input shape
    mel_db = np.resize(mel_db, (128, 128)).astype(np.float32)
    mel_db = mel_db.reshape(1, 128, 128, 1)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], mel_db)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    return output, mel_db

uploaded_file = st.file_uploader("Upload WAV File", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("🔍 Analyze Track"):

        confidences, mel_input = predict(uploaded_file)
        pred_idx = np.argmax(confidences)
        pred_label = label_map[pred_idx]
        confidence_score = float(confidences[pred_idx])

        st.divider()

        # ----------------------------
        # Prediction
        # ----------------------------
        st.subheader("🎼 Final Prediction")
        st.success(f"{pred_label} ({confidence_score:.2f})")

        # ----------------------------
        # Condition
        # ----------------------------
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

        # ----------------------------
        # Mel Spectrogram Display
        # ----------------------------
        st.subheader("📊 Mel-Spectrogram")

        st.image(mel_input[0, :, :, 0], caption="Mel Spectrogram", use_column_width=True)

        # ----------------------------
        # JSON Report
        # ----------------------------
        report = {
            "file": uploaded_file.name,
            "prediction": pred_label,
            "confidence": confidence_score,
            "condition": condition,
            "timestamp": str(datetime.now())
        }

        st.download_button(
            "Download JSON Report",
            data=json.dumps(report, indent=4),
            file_name="instrument_report.json",
            mime="application/json"
        )

        # ----------------------------
        # PDF Report
        # ----------------------------
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200,10,"InstruNet - Instrument Report", ln=True)
        pdf.cell(200,10,f"Prediction: {pred_label}", ln=True)
        pdf.cell(200,10,f"Confidence: {confidence_score:.2f}", ln=True)
        pdf.cell(200,10,f"Condition: {condition}", ln=True)
        pdf.cell(200,10,f"Timestamp: {datetime.now()}", ln=True)

        pdf.output("report.pdf")

        with open("report.pdf", "rb") as f:
            st.download_button(
                "Download PDF Report",
                data=f,
                file_name="instrument_report.pdf",
                mime="application/pdf"
            )
