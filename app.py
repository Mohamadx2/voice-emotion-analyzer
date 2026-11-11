"""Streamlit app for local speech-emotion analysis and spectrogram visualization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "r-f/wav2vec-english-speech-emotion-recognition"
SAMPLE_RATE = 16_000
MAX_SECONDS = 30

# The model was trained on seven labels. Tired and Quiet are approximate display
# categories because they are not native classes in this checkpoint.
DISPLAY_CATEGORIES = [
    "Angry tone",
    "Sad",
    "Scared",
    "Happy",
    "Tired",
    "Quiet",
    "Emotionless",
]
MODEL_TO_DISPLAY = {
    "angry": "Angry tone",
    "sad": "Sad",
    "fear": "Scared",
    "happy": "Happy",
    "disgust": "Tired",       # approximate proxy; not a native tired class
    "surprise": "Quiet",     # approximate proxy; not a native quiet class
    "neutral": "Emotionless",
}


@st.cache_resource(show_spinner=False)
def load_model():
    """Download and cache the local speech-emotion model on first use."""
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    model.eval()
    return extractor, model


def load_audio(uploaded_file) -> tuple[np.ndarray, int, Path]:
    """Save the upload temporarily and decode it with librosa."""
    suffix = Path(uploaded_file.name).suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        path = Path(handle.name)
    audio, sample_rate = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio, sample_rate, path


def predict(audio: np.ndarray, extractor, model) -> list[dict]:
    """Return model probabilities with the requested display labels."""
    inputs = extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    results = []
    for index, probability in enumerate(probabilities):
        raw_label = model.config.id2label[index].lower()
        results.append(
            {
                "raw_label": raw_label,
                "label": MODEL_TO_DISPLAY.get(raw_label, raw_label.title()),
                "score": float(probability),
                "proxy": raw_label in {"disgust", "surprise"},
            }
        )
    return sorted(results, key=lambda item: item["score"], reverse=True)


def plot_spectrogram(audio: np.ndarray, sample_rate: int):
    """Create a log-power mel spectrogram figure."""
    fig, axis = plt.subplots(figsize=(11, 4))
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=96)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    image = librosa.display.specshow(
        mel_db, sr=sample_rate, x_axis="time", y_axis="mel", ax=axis, cmap="magma"
    )
    axis.set_title("Mel spectrogram (log power)")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Frequency (mel)")
    fig.colorbar(image, ax=axis, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Voice Emotion Analyzer", page_icon="🎙️", layout="wide")
    st.title("Voice Emotion Analyzer")
    st.write("Upload a short WAV or MP3 recording to inspect its spectrogram and estimate vocal emotion locally.")

    with st.sidebar:
        st.header("Supported categories")
        for category in DISPLAY_CATEGORIES:
            st.write(f"- {category}")
        st.caption("Tired and Quiet are approximate proxy labels because the selected model does not natively classify them.")
        st.caption(f"Model: `{MODEL_ID}`")

    uploaded = st.file_uploader("Upload a short voice recording", type=["wav", "mp3"])
    true_label = st.selectbox(
        "Optional: known label for this recording (used to calculate one-file correctness)",
        ["Not provided"] + DISPLAY_CATEGORIES,
    )

    if uploaded is None:
        st.info("Choose a WAV or MP3 file to begin. Keep recordings short; the demo analyzes at most 30 seconds.")
        return

    try:
        audio, sample_rate, temp_path = load_audio(uploaded)
    except Exception as error:
        st.error(f"Could not decode this audio file: {error}")
        return

    if len(audio) == 0:
        st.error("The uploaded recording contains no audio samples.")
        return
    if len(audio) > MAX_SECONDS * sample_rate:
        st.warning(f"The recording is longer than {MAX_SECONDS} seconds; only the first {MAX_SECONDS} seconds are analyzed.")
        audio = audio[: MAX_SECONDS * sample_rate]

    st.audio(uploaded)
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Spectrogram")
        st.pyplot(plot_spectrogram(audio, sample_rate), clear_figure=True)
    with right:
        st.subheader("Prediction")
        with st.spinner("Loading the local model and analyzing the voice..."):
            extractor, model = load_model()
            results = predict(audio, extractor, model)
        top = results[0]
        st.metric("Predicted category", top["label"])
        st.metric("Model confidence", f"{top['score']:.1%}")
        if top["proxy"]:
            st.warning("This category is a proxy mapping, not a native class in the model.")
        if true_label != "Not provided":
            is_correct = top["label"] == true_label
            st.metric("One-file correctness", "Correct" if is_correct else "Incorrect")
            st.caption("This is correctness for this labeled file, not a statistically valid model accuracy estimate.")

    st.subheader("All model scores")
    st.bar_chart({item["label"]: item["score"] for item in results})
    st.caption("Scores are model probabilities. They are not guaranteed real-world accuracy, especially for accents, languages, microphones, background noise, or labels outside the training data.")


if __name__ == "__main__":
    main()
