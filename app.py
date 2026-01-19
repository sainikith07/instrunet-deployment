import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from tflite_runtime.interpreter import Interpreter
import matplotlib.pyplot as plt
import json
from fpdf import FPDF
from datetime import datetime

st.set_page_config(page_title="InstruNet", page_icon="🎶", layout="wide")

interpreter = Interpreter(model_path="instruNet_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_map = ["flute", "trumpet", "violin"]

def predict_instrument(file):
    audio, sr = sf.read(file)
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

uploaded = st.file_uploader("Upload WAV file", type=["wav"])
if uploaded:
    st.audio(uploaded, format="audio/wav")

    if st.button("Analyze"):
        conf, audio, sr = predict_instrument(uploaded)
        idx = np.argmax(conf)
        pred = label_map[idx]

        st.success(f"Detected Instrument: **{pred.upper()}**")

        st.subheader("Confidence Levels")
        for i, lbl in enumerate(label_map):
            st.write(f"{lbl}: {conf[i]*100:.1f}%")
            st.progress(float(conf[i]))

        st.subheader("Waveform")
        fig, ax = plt.subplots(figsize=(6,3))
        librosa.display.waveshow(audio, sr=sr, ax=ax)
        st.pyplot(fig)

        st.subheader("Spectrogram")
        fig2, ax2 = plt.subplots(figsize=(6,3))
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, cmap="magma", ax=ax2)
        st.pyplot(fig2)

        line = np.sin(np.linspace(0,3*np.pi,200)) * conf[idx] + conf[idx]
        st.subheader("Timeline (Mock Visualization)")
        fig3, ax3 = plt.subplots(figsize=(8,2))
        ax3.plot(line, color="cyan")
        ax3.set_yticks([])
        st.pyplot(fig3)

        report = {
            "file": uploaded.name,
            "prediction": pred,
            "confidence_scores": {label_map[i]: float(conf[i]) for i in range(len(label_map))},
            "timestamp": str(datetime.now())
        }

        st.download_button("📁 Download JSON", json.dumps(report, indent=4), file_name="report.json")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200,10,txt="Instrument Recognition Report",ln=True,align='C')
        pdf.set_font("Arial", size=12)
        pdf.ln(5)
        pdf.cell(200,10,txt=f"Prediction: {pred}",ln=True)
        for i,l in enumerate(label_map):
            pdf.cell(200,10,txt=f"{l}: {conf[i]*100:.1f}%",ln=True)
        pdf.output("report.pdf")

        with open("report.pdf","rb") as f:
            st.download_button("📄 Download PDF", data=f, file_name="report.pdf", mime="application/pdf")
