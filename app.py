import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import json
from datetime import datetime
from fpdf import FPDF

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="instruNet_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_map = ["flute", "trumpet", "violin"]

def predict_instrument(audio_file):
    audio, sr = sf.read(audio_file)

    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
    audio = audio[:22050*3] if len(audio) > 22050*3 else np.pad(audio, (0, max(0, 22050*3 - len(audio))))

    mel = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.resize(mel_db, (128, 128))
    mel_db = mel_db.astype(np.float32)
    mel_db = mel_db.reshape(1, 128, 128, 1)

    interpreter.set_tensor(input_details[0]['index'], mel_db)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_idx = np.argmax(output)
    confidence = float(output[pred_idx])

    return label_map[pred_idx], confidence, mel_db[0, :, :, 0]

# ---------------------------------------
# UI
# ---------------------------------------

st.title("🎶 InstruNet - Instrument Recognition System")
st.write("Upload a `.wav` file and click Analyze Track.")

uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("🔍 Analyze Track"):

        prediction, confidence, mel_image = predict_instrument(uploaded_file)

        st.divider()

        # ---------------------------
        # Prediction
        # ---------------------------
        st.subheader("🎼 Final Prediction")
        st.success(f"Predicted Instrument: {prediction}")
        st.write(f"Confidence Score: {confidence:.4f}")

        # ---------------------------
        # Mel Spectrogram Display
        # ---------------------------
        st.subheader("📊 Mel-Spectrogram")
        st.image(mel_image, caption="Mel-Spectrogram", use_column_width=True)

        # ---------------------------
        # JSON Report
        # ---------------------------
        report = {
            "file_name": uploaded_file.name,
            "prediction": prediction,
            "confidence": confidence,
            "timestamp": str(datetime.now())
        }

        st.subheader("📄 Download JSON Report")
        st.download_button(
            label="Download JSON",
            data=json.dumps(report, indent=4),
            file_name="instrument_report.json",
            mime="application/json"
        )

        # ---------------------------
        # PDF Report
        # ---------------------------
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, "InstruNet Instrument Report", ln=True)
        pdf.ln(5)
        pdf.cell(200, 10, f"File: {uploaded_file.name}", ln=True)
        pdf.cell(200, 10, f"Prediction: {prediction}", ln=True)
        pdf.cell(200, 10, f"Confidence: {confidence:.4f}", ln=True)
        pdf.cell(200, 10, f"Timestamp: {datetime.now()}", ln=True)

        pdf.output("report.pdf")

        with open("report.pdf", "rb") as f:
            st.subheader("📄 Download PDF Report")
            st.download_button(
                label="Download PDF",
                data=f,
                file_name="instrument_report.pdf",
                mime="application/pdf"
            )
