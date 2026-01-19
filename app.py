import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from tflite_runtime.interpreter import Interpreter
import json
from fpdf import FPDF
from datetime import datetime

st.set_page_config(page_title="InstruNet", page_icon="🎶", layout="wide")

# Load TFLite Model
interpreter = Interpreter(model_path="instruNet_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_map = ["flute", "trumpet", "violin"]

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

uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded:
    st.audio(uploaded, format="audio/wav")
    
    if st.button("Analyze Track"):
        conf, audio, sr = predict_instrument(uploaded)
        idx = np.argmax(conf)
        pred_label = label_map[idx]

        st.success(f"Predicted Instrument: **{pred_label.upper()}**")

        # Confidence bars
        st.subheader("Confidence Scores")
        for i, lbl in enumerate(label_map):
            st.write(f"{lbl}: {conf[i]*100:.2f}%")
            st.progress(float(conf[i]))

        # Waveform
        st.subheader("Waveform")
        fig, ax = plt.subplots(figsize=(6,3))
        librosa.display.waveshow(audio, sr=sr, ax=ax)
        st.pyplot(fig)

        # Spectrogram
        st.subheader("Mel-Spectrogram")
        fig2, ax2 = plt.subplots(figsize=(6,3))
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, ax=ax2, cmap="magma")
        st.pyplot(fig2)

        # Fake timeline (for visualization)
        st.subheader("Timeline Visualization")
        fig3, ax3 = plt.subplots(figsize=(10,2))
        line = np.sin(np.linspace(0,3*np.pi,150)) * conf[idx] + conf[idx]
        ax3.plot(line, color='cyan')
        ax3.set_yticks([])
        st.pyplot(fig3)

        # JSON export
        report = {
            "file": uploaded.name,
            "prediction": pred_label,
            "confidence_scores": {label_map[i]: float(conf[i]) for i in range(len(label_map))}
        }

        st.download_button("📁 Download JSON", json.dumps(report, indent=4), file_name="report.json")

        # PDF export
        def export_pdf(filename):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200,10,txt="Instrument Recognition Report",ln=True,align='C')
            pdf.set_font("Arial", size=12)
            pdf.ln(5)
            pdf.cell(200,10,txt=f"Prediction: {pred_label}",ln=True)
            pdf.cell(200,10,txt="Confidence Scores:",ln=True)
            for i, lbl in enumerate(label_map):
                pdf.cell(200,10,txt=f"{lbl}: {conf[i]*100:.2f}%",ln=True)
            pdf.output(filename)

        export_pdf("report.pdf")
        with open("report.pdf","rb") as f:
            st.download_button("📄 Download PDF", data=f, file_name="report.pdf", mime="application/pdf")
