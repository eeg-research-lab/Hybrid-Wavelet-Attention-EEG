# Hybrid Wavelet-Attention EEG Framework

Multi-channel EEG artifact removal and seizure detection framework using:

- Butterworth filtering
- Notch filtering
- Wavelet denoising
- Residual attention reconstruction
- CNN-BiLSTM with attention
- Focal loss optimization

## Dataset

CHB-MIT EEG Dataset

## Requirements

- Python 3.10+
- TensorFlow
- MNE
- SciPy
- PyWavelets
- Streamlit

## Models

- Wavelet preprocessing model
- Hybrid wavelet-attention model

## Usage

Run training:

python train.py

Run Streamlit:

streamlit run app.py

## Features

- EEG artifact removal
- Seizure detection
- EEG visualization
- Attention-based classification
