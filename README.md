# 🎶 InstruNet — AI-Powered Musical Instrument Recognition

InstruNet is a deployable AI system that identifies musical instruments from `.wav` audio files using **Mel-Spectrograms + CNN + TensorFlow Lite**, served through a modern **Streamlit dashboard**.



## 🎯 Project Summary

| Property | Value |
|--------|------|
| Problem | Musical Instrument Classification from Audio |
| Input | `.wav` audio file |
| Output | Instrument Label + Confidence Scores |
| Supported Instruments | Flute, Trumpet, Violin |
| Model | CNN |
| Deployment Format | TensorFlow Lite (`.tflite`) |
| UI | Streamlit |
| Reports | JSON + PDF |
| GPU Required | ❌ No |
| Real-Time Capable | ✅ Yes |

---

## ✨ Key Features

- 🎧 Upload and analyze `.wav` audio files  
- 📈 Visualize waveform and Mel-spectrogram  
- ⚡ Run inference using CPU-friendly **TensorFlow Lite**  
- 📊 Display confidence score bars  
- ⏱️ Timeline activation visualization  
- 📄 Export prediction reports (JSON & PDF)  
- ☁️ Works on Streamlit Cloud and local machines  
- 🪶 Lightweight with no GPU dependency  




---

## 🧩 Model Card

| Field | Details |
|-----|--------|
| Model Name | InstruNet-CNN |
| Input | 128×128 Log-Mel Spectrogram |
| Output Classes | Flute, Trumpet, Violin |
| Data Type | float32 |
| Training Framework | TensorFlow |
| Inference Runtime | TensorFlow Lite |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Export Format | `.tflite` |
| Hardware | CPU (No GPU Required) |

---

## 🎧 Supported Instruments

- 🎼 Flute  
- 🎺 Trumpet  
- 🎻 Violin  

---

## 🎨 Streamlit Dashboard Features

- 🔴 Real-time inference  
- ▶️ Audio playback  
- 📉 Waveform visualization  
- 🌈 Mel-spectrogram visualization  
- 📊 Confidence percentage bars  
- ⏱️ Timeline activation plot  
- 📤 JSON export  
- 📄 PDF export  

---

## 📊 Example JSON Output

```json
{
  "file": "sample.wav",
  "prediction": "flute",
  "confidence_scores": {
    "flute": 0.9844,
    "trumpet": 0.0131,
    "violin": 0.0025
  },
  "timestamp": "2026-01-18 12:31:44"
}
```
---
## 📦 Installation
Clone Repository
git clone https://github.com/sainikith07/instrunet-deployment.git
cd instrunet-deployment

Install Dependencies
pip install -r requirements.txt

Run Web App
streamlit run app.py


App opens at:

http://localhost:8501/

---

## This project combines:

Music Information Retrieval (MIR)

Digital Signal Processing (DSP)

Machine Listening

Audio Classification

Edge AI (TFLite Deployment)

Model Explainability (Confidence Visualization)

---
## 🏢 Business & Industry Use-Cases

EdTech & interactive music learning

Audio surveillance and monitoring

Mobile music recognition apps

Audio production and DAW tooling

Metadata generation for media assets

Interactive music games and AR/VR


---
## 🧭 Roadmap

⬜ Real dataset training (IRMAS / NSynth)

⬜ Multi-instrument polyphonic detection

⬜ Segment-wise real timeline activation

⬜ HuggingFace deployment

⬜ Mobile app using TFLite

⬜ Orchestra and jazz instrument expansion

🧾 Academic Citation (BibTeX)


---
## @software{instrunet2026,
  author    = {Sai Nikith},
  title     = {InstruNet: AI-based Musical Instrument Recognition System},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/sainikith07/instrunet-deployment}
}


---
## 🤝 Contributing

Contributions are welcome!

git checkout -b feature-name
git commit -m "Add new feature"
git push origin feature-name


Open a Pull Request 🚀

---

## 👤 Author

Sai Nikith
AI/ML & Signal Processing Developer
GitHub: https://github.com/sainikith07

---

## ⭐ Support

If you found this project useful,
please Star ⭐ the repository — it motivates continued development!

---

## 🏁 License

This project is licensed under the MIT License.


---
---
If you want next:
- 🎤 **Interview explanation**
- 📄 **Resume project points**
- ☁️ **Streamlit Cloud deployment steps**
- 🔥 **LinkedIn post**


