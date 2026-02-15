import streamlit as st
import numpy as np
import json
from datetime import datetime
from fpdf import FPDF
import random

st.set_page_config(page_title="InstruNet", layout="centered")

st.title("🎶 InstruNet - Instrument Analyzer")
st.write("Upload a WAV file and click Analyze Track.")

uploaded_file = st.file_uploader("Upload WAV File", type=["wav"])

label_map = ["Flute", "Trumpet", "Violin"]

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("🔍 Analyze Track"):

        # Simulated prediction (deployment safe)
        pred_label = random.choice(label_map)
        confidence_score = round(random.uniform(0.5, 0.95), 2)

        st.divider()

        # Final Prediction
        st.subheader("🎼 Final Prediction")
        st.success(pred_label)

        # Instrument Condition
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

        # Audio Timeline (Simulated)
        st.subheader("📈 Audio Timeline")

        timeline = np.sin(np.linspace(0, 3*np.pi, 200)) * confidence_score + confidence_score
        st.line_chart(timeline)

        # Audio Representation (Simulated waveform)
        st.subheader("🎵 Audio Representation")

        waveform = np.sin(np.linspace(0, 20*np.pi, 500))
        st.line_chart(waveform)

        # JSON Report
        report = {
            "file": uploaded_file.name,
            "prediction": pred_label,
            "condition": condition,
            "confidence_score": confidence_score,
            "timestamp": str(datetime.now())
        }

        st.download_button(
            "Download JSON Report",
            data=json.dumps(report, indent=4),
            file_name="instrument_report.json",
            mime="application/json"
        )

        # PDF Report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200,10,"InstruNet Instrument Report", ln=True)
        pdf.cell(200,10,f"Prediction: {pred_label}", ln=True)
        pdf.cell(200,10,f"Condition: {condition}", ln=True)
        pdf.cell(200,10,f"Confidence: {confidence_score}", ln=True)
        pdf.cell(200,10,f"Timestamp: {datetime.now()}", ln=True)

        pdf.output("report.pdf")

        with open("report.pdf", "rb") as f:
            st.download_button(
                "Download PDF Report",
                data=f,
                file_name="instrument_report.pdf",
                mime="application/pdf"
            )
