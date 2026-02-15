import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import json
from datetime import datetime
from fpdf import FPDF

st.set_page_config(page_title="InstruNet", layout="centered")

st.title("🎶 InstruNet - Audio Analyzer")
st.write("Upload a WAV file and click Analyze Track.")

# ---------------------------------------
# Upload Section
# ---------------------------------------
uploaded_file = st.file_uploader("Upload WAV File", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("🔍 Analyze Track"):

        # Load audio
        audio, sr = sf.read(uploaded_file)

        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Generate Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        st.divider()

        # ---------------------------------------
        # Display Mel Spectrogram
        # ---------------------------------------
        st.subheader("📊 Mel-Spectrogram")

        fig, ax = plt.subplots(figsize=(8,4))
        img = ax.imshow(mel_db, aspect='auto', origin='lower', cmap='magma')
        ax.set_title("Mel Spectrogram")
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel Frequency")
        fig.colorbar(img, ax=ax)
        st.pyplot(fig)

        # ---------------------------------------
        # JSON Report
        # ---------------------------------------
        report = {
            "file_name": uploaded_file.name,
            "sample_rate": sr,
            "duration_seconds": round(len(audio)/sr, 2),
            "mel_shape": mel_db.shape,
            "timestamp": str(datetime.now())
        }

        st.subheader("📄 JSON Report")

        st.download_button(
            label="Download JSON Report",
            data=json.dumps(report, indent=4),
            file_name="mel_analysis_report.json",
            mime="application/json"
        )

        # ---------------------------------------
        # PDF Report
        # ---------------------------------------
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200,10,"InstruNet - Mel Spectrogram Report", ln=True)
        pdf.ln(5)
        pdf.cell(200,10,f"File: {uploaded_file.name}", ln=True)
        pdf.cell(200,10,f"Sample Rate: {sr}", ln=True)
        pdf.cell(200,10,f"Duration (s): {round(len(audio)/sr,2)}", ln=True)
        pdf.cell(200,10,f"Mel Shape: {mel_db.shape}", ln=True)
        pdf.cell(200,10,f"Timestamp: {datetime.now()}", ln=True)

        pdf.output("mel_report.pdf")

        with open("mel_report.pdf", "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name="mel_analysis_report.pdf",
                mime="application/pdf"
            )
