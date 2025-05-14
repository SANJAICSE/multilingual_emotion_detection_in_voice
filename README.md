
# Multilingual Emotion Detection in Voice

This project uses the OpenAI Whisper model along with machine learning techniques to **transcribe audio**, **detect the spoken language**, and **classify the emotion** conveyed in speech. It supports **multilingual** inputs and extracts **audio features** for emotion classification.

## 🧠 Features

* 🎙️ **Speech Transcription** using Whisper
* 🌍 **Language Detection** of the audio
* 😠 **Emotion Detection** via MFCC features + SVM
* 📁 Processes audio files directly

---

## 📦 Requirements

Install dependencies using:

```bash
pip install openai-whisper librosa scikit-learn moviepy
```

---

## 🛠️ How It Works

1. **Load Audio & Transcribe Speech**

   * Uses OpenAI Whisper (`base` model) to transcribe audio and detect language.

2. **Extract Audio Features**

   * Extracts MFCC (Mel Frequency Cepstral Coefficients) using `librosa`.

3. **Train Emotion Classifier**

   * Trains a Support Vector Machine (SVM) model on labeled MFCC features.
   * Emotions like *happy, sad, angry, neutral*, etc., can be classified.

4. **Predict Emotion**

   * For any new audio input, it transcribes, extracts features, and classifies the emotion.

---

## 📂 File Structure

* `transcribe_audio(audio_path)`: Transcribes and detects language.
* `extract_features(audio_path)`: Extracts MFCC features.
* `train_model(X, y)`: Trains the emotion classifier.
* `predict_emotion(audio_path)`: Predicts the emotion of the speaker.

---

## 🧪 Example Usage

```python
transcription, lang = transcribe_audio("example.wav")
features = extract_features("example.wav")
emotion = predict_emotion("example.wav")
print(f"Detected language: {lang}, Emotion: {emotion}")
```

---

## 📌 Notes

* Ensure your input audio is clean and has minimal background noise for best results.
* You can expand the emotion classes by providing more labeled training data.

