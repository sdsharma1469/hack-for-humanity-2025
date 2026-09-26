# Emergency Siren Detection

A **Hack for Humanity 2025** project that combines digital signal processing, deep learning, and mobile software to detect siren-like sounds from recorded urban audio.

The system transforms raw audio into time-frequency features using a **Short-Time Fourier Transform (STFT)**, generates **Mel spectrograms**, and classifies those features with a convolutional neural network (CNN). A Flutter client records audio and communicates with a Python/Flask inference backend.

## Overview

Emergency sirens can be difficult to recognize reliably in noisy urban environments. This project explores whether frequency-domain audio features and a learned classifier can distinguish siren-like sounds from other common city noises.

### Signal-processing pipeline

```text
Recorded Audio
      |
      v
STFT / FFT-based spectral analysis
(n_fft = 2048, hop_length = 512)
      |
      v
Power Spectrum
      |
      v
24-band Mel Spectrogram
      |
      v
Decibel Conversion
      |
      v
Pad / Truncate + Normalize
      |
      v
CNN Classifier
      |
      v
Siren / Not Siren
```

## Digital Signal Processing

For an audio signal `x[n]`, the Short-Time Fourier Transform analyzes how its frequency content changes over time:

```text
X(m,k) = sum_n x[n] w[n - mH] exp(-j 2 pi k n / N)
```

where `N` is the FFT size, `H` is the hop length, and `w` is the analysis window.

The project uses Librosa with:

- FFT size: **2048 samples**
- Hop length: **512 samples**
- Mel bands: **24**
- Frequency representation: **power converted to decibels**

The STFT magnitude is squared to obtain spectral power, projected onto the Mel scale, and converted to dB. The resulting spectrogram provides a compact time-frequency representation for the CNN.

## Machine Learning Pipeline

Training data is based on the **UrbanSound8K** dataset. Audio clips are mapped into two output classes:

- **Siren**
- **Not siren**

The training pipeline includes:

- pitch-shift data augmentation
- STFT-based spectral feature extraction
- Mel spectrogram generation
- fixed-length padding and truncation
- feature normalization
- train/test splitting
- CNN training
- L2 regularization
- dropout
- confusion-matrix evaluation
- model and label-encoder serialization

### CNN architecture

```text
Mel Spectrogram
      |
Conv2D (32) + ReLU
      |
Max Pool
      |
Conv2D (64) + ReLU
      |
Max Pool
      |
Conv2D (128) + ReLU
      |
Max Pool
      |
Flatten
      |
Dense (256) + ReLU
      |
Dropout (0.5)
      |
Softmax
      |
Siren / Not Siren
```

## System Architecture

```text
+-------------------+
|   Flutter Client  |
|  Record / Upload  |
+---------+---------+
          |
          | encrypted audio
          v
+-------------------+
|   Flask Backend   |
| Upload / Decrypt  |
+---------+---------+
          |
          v
+-------------------+
| Signal Processing |
| STFT -> Mel -> dB |
+---------+---------+
          |
          v
+-------------------+
| TensorFlow CNN    |
| Audio Classifier  |
+---------+---------+
          |
          v
+-------------------+
| Prediction +      |
| Confidence        |
+-------------------+
```

The Flask backend accepts uploaded audio, decrypts encrypted recordings, applies the same spectral preprocessing used during training, and runs the saved TensorFlow model.

## Tech Stack

**Signal Processing & Machine Learning**

- Python
- NumPy
- Librosa
- TensorFlow / Keras
- scikit-learn
- Pandas
- Matplotlib

**Backend**

- Flask
- Flask-CORS
- AES-256/CBC audio decryption

**Client**

- Flutter
- Dart

**Dataset**

- UrbanSound8K

## Repository Structure

```text
hack-for-humanity-2025/
├── Backend/
│   ├── process.py       # preprocessing and CNN training
│   ├── Inference.py     # STFT/Mel preprocessing and inference
│   ├── Server.py        # Flask API and encrypted uploads
│   ├── Kaggle.py        # dataset utilities
│   ├── models/          # trained model and label encoder
│   └── uploads/         # uploaded recordings
├── app/                 # Flutter application
└── build/
```

## Running the Backend

From the backend directory:

```bash
cd Backend
pip install -r requirements.txt
python Server.py
```

The development server runs on port `5050`.

> **Note:** Some training paths were configured for the original hackathon development machine. Update the dataset and model paths as necessary before retraining.

## Training

The training pipeline is implemented in `Backend/process.py`.

It loads UrbanSound8K audio, extracts spectral features, trains the CNN, evaluates predictions, and saves the trained model and label encoder.

Update `csv_path` and `audio_path` for your local UrbanSound8K installation, then run:

```bash
cd Backend
python process.py
```

## Inference

`Backend/Inference.py` performs the inference preprocessing pipeline:

1. Load the audio waveform.
2. Compute the STFT.
3. Calculate spectral power.
4. Convert the spectrum to a Mel spectrogram.
5. Convert power to decibels.
6. Pad or truncate the feature matrix.
7. Normalize the input.
8. Run CNN inference.
9. Return the predicted class and confidence.

## Hack for Humanity 2025

Built during **Hack for Humanity 2025** as an exploration of real-time acoustic awareness using signal processing, machine learning, and mobile software.

## Future Improvements

- Evaluate on a larger held-out siren dataset.
- Add precision, recall, F1, ROC-AUC, and false-positive-rate evaluation.
- Support streaming audio instead of file-based inference.
- Optimize the CNN for on-device TensorFlow Lite inference.
- Replace development encryption keys with secure key management.
- Add automated tests to guarantee preprocessing consistency between training and inference.
- Move dataset/model paths into portable configuration files.
- Benchmark inference latency for real-time deployment.
