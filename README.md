🎶 InstruNet — AI-Powered Musical Instrument Recognition

InstruNet is a deployable AI system that identifies musical instruments from .wav audio using Mel-Spectrograms + CNN + TensorFlow Lite, served through a modern Streamlit dashboard.

📛 Project Badges












🏷 Banner Placeholder

Add this image later:

assets/banner.png

🎯 Summary
Property	Value
Problem	Instrument Classification from Audio
Input	.wav file
Output	Instrument Label + Confidence Scores
Instruments	Flute, Trumpet, Violin
Model	CNN
Deployment Format	.tflite
UI	Streamlit
Reports	JSON + PDF
GPU Required	No
Real-Time Capable	Yes
✨ Key Features

Upload and analyze .wav audio

Visualize waveform and Mel-spectrogram

Run inference using TFLite (CPU-friendly)

Display confidence score bars

Show timeline activation visualization

Export JSON and PDF reports

Works on Streamlit Cloud / Local

Lightweight with no GPU dependency

🧠 System Architecture
.wav File
    ↓
Audio Preprocessing (Resample, Trim, Pad)
    ↓
Mel-Spectrogram (128×128 Log-Mel)
    ↓
CNN Classifier (Softmax via TFLite)
    ↓
Streamlit Dashboard (Waveform, Spectrogram, Confidence, Timeline, Export)

🧩 Model Card
Field	Details
Model Name	InstruNet-CNN
Input	128×128 Log-Mel Spectrogram
Output Classes	Flute, Trumpet, Violin
Datatype	float32
Training Framework	TensorFlow
Inference Runtime	TensorFlow Lite
Optimizer	Adam
Loss	Sparse Categorical Crossentropy
Export Format	.tflite
Hardware	CPU (No GPU Required)
🎧 Supported Instruments

🎼 Flute

🎺 Trumpet

🎻 Violin

🎨 Dashboard Features

Real-time inference

Audio playback

Waveform visualization

Mel-Spectrogram visualization

Confidence percentage bars

Mock timeline activation plot

JSON export

PDF export

📊 Example JSON Output
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

📁 Repository Structure
instrunet-deployment/
│
├── app.py                 # Streamlit Web App
├── instruNet_model.tflite # TFLite Model for Inference
├── requirements.txt       # Python Dependencies
├── .runtime.txt           # Python Version Pinning (3.10)
├── README.md              # Documentation (this file)
└── assets/                # Screenshots/GIFs (optional)

📦 Installation
1. Clone Repo
git clone https://github.com/sainikith07/instrunet-deployment.git
cd instrunet-deployment

2. Install Dependencies
pip install -r requirements.txt

3. Run Web App
streamlit run app.py


App will open at:

http://localhost:8501/

🌐 Deployment Options
Platform	Status
Streamlit Cloud	Supported
Local PC	Supported
Google Colab (via Ngrok)	Supported
HuggingFace Spaces	Planned
Android (TFLite)	Planned
iOS (TFLite)	Planned
🔬 Research Motivation

This project sits at the intersection of:

Music Information Retrieval (MIR)

Digital Signal Processing (DSP)

Machine Listening

Audio Classification

Edge AI (TFLite Deployment)

ML Explainability (Confidence Visualization)

🏢 Business / Industry Use-Cases

EdTech: Interactive music learning

Audio Surveillance & Monitoring

Mobile Music Recognition Apps

Audio Production Tooling

DAW-aware audio tagging

Meta-data generation for media assets

Interactive music games and AR/VR

🧭 Roadmap

 Real dataset training (IRMAS / NSynth)

 Multi-instrument polyphonic detection

 Segment-wise real timeline activation

 HuggingFace deployment

 Mobile app using TFLite

 Jazz/Orchestra instruments expansion

🧾 Academic Citation (BibTeX)

If you use this system in research:

@software{instrunet2026,
  author       = {Sai Nikith},
  title        = {InstruNet: AI-based Musical Instrument Recognition System},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/sainikith07/instrunet-deployment}
}

🤝 Contributing

Contributions are welcome!

Steps:

git checkout -b feature-name
git commit -m "Add new feature"
git push origin feature-name


Open a Pull Request 🚀

👤 Author

Name: Sai Nikith
Role: AI/ML & Signal Processing Developer
GitHub: https://github.com/sainikith07

⭐ Support

If you found this project useful:

Please Star ⭐ this repository

It motivates continued development!

🏁 License

This project is licensed under the MIT License.
