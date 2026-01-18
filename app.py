import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="instruNet_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_map = ["flute", "trumpet", "violin"]

def predict_instrument(audio_file):
    audio, sr = sf.read(audio_file)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
    audio = audio[:22050*3] if len(audio) > 22050*3 else np.pad(audio, (0, max(0, 22050*3 - len(audio))))

    mel = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.resize(mel_db, (128, 128))
    mel_db = mel_db.astype(np.float32)

    mel_db = mel_db.reshape(1, 128, 128, 1)

    interpreter.set_tensor(input_details[0]['index'], mel_db)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    pred_idx = np.argmax(output)
    return label_map[pred_idx]

st.title("🎶 InstruNet - Instrument Recognition System")
st.write("Upload a `.wav` file to classify the instrument.")

uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    if st.button("Predict"):
        pred = predict_instrument(uploaded_file)
        st.success(f"Predicted Instrument: **{pred}**")
