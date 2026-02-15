import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import json
from datetime import datetime
from fpdf import FPDF

# -----------------------------------
# Load TFLite Model
# -----------------------------------
interpreter = tf.lite.Interpreter(model_path="instruNet_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_map = ["flute", "trumpet", "violin"]

# -----------------------------------
# Prediction Function
# -----------------------------------
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

    # Generate Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize for model input
    mel_resized = np.resize(mel_db, (128, 128)).astype(np.float32)
    mel_input = mel_resized.reshape(1, 128, 128, 1)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], mel_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_idx = np.argmax(output)
    confidence = float(output[pred_idx])

    return label_map[pred_idx], confidence, mel_resized

# -----------------------------------
# UI
# -----------------------------------
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

        # Normalize image for Streamlit
        mel_image_norm = (mel_image - mel_image.min()) / (mel_image.max() - mel_image.min())

        st.image(
            mel_image_norm,
            caption="Mel-Spectrogram",
            use_column_width=True,
            clamp=True
        )

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
