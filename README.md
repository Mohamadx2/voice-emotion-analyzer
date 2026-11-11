# Voice Emotion Analyzer

A local Streamlit app that analyzes a short WAV or MP3 voice recording, draws a mel spectrogram, and estimates vocal emotion with a pretrained Hugging Face model.

## Features

- Upload `.wav` or `.mp3` audio.
- Analyze up to the first 30 seconds of a recording.
- Display a log-power mel spectrogram.
- Run inference locally after the model downloads.
- Show the predicted category, confidence, and all model scores.
- Optionally select a known label to report correctness for that one recording.

## Requested categories

The app displays these categories: **Angry tone, Sad, Scared, Happy, Tired, Quiet, and Emotionless**.

The selected model natively predicts `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`. Therefore, `Tired` is shown as an approximate proxy for `disgust`, and `Quiet` is shown as an approximate proxy for `surprise`. These are not genuine native model classes and should not be treated as reliable tiredness or quietness detection.

## Install and run

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. The model is downloaded from Hugging Face the first time it is used and cached locally afterward. The checkpoint is considerably larger than the tiny text model in the earlier projects, but it runs locally and does not require an API key.

## Accuracy and responsible use

The model card reports an evaluation accuracy of about 97.5% on its own evaluation setup, but that number does not transfer automatically to your recordings. Real-world performance can change with language, accent, speaker, microphone, noise, acting style, and category definitions.

The app therefore labels the top score as **model confidence**, not guaranteed accuracy. If you select a known label, it reports **one-file correctness** only. A meaningful accuracy estimate requires a representative labeled test set and an evaluation script.

This is an experimental signal, not a medical, employment, safety, or psychological assessment. Vocal emotion is not the same as a person's actual mental state.

## Model attribution

This project uses [`r-f/wav2vec-english-speech-emotion-recognition`](https://huggingface.co/r-f/wav2vec-english-speech-emotion-recognition), a Wav2Vec2 model fine-tuned on SAVEE, RAVDESS, and TESS emotion datasets. Review the model's license and terms before redistribution.

## License

The project code is released under the MIT License.
