# ============================================================
# streamlit_app.py
# FINAL EEG SEIZURE DETECTION STREAMLIT APPLICATION
# WAVELET VS HYBRID COMPARISON SYSTEM
# ============================================================

import re
import gc
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import mne
import pywt
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import iirnotch
from scipy.signal import welch
from scipy.signal import spectrogram

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from labels import get_seizure_ranges

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EEG Seizure Detection",
    layout="wide"
)

st.title(
    "EEG Seizure Detection System"
)

st.markdown(
    """
    Multi-Channel EEG Artifact Removal via
    Hybrid Wavelet-Attention Mechanisms
    """
)

# ============================================================
# MODEL PATHS
# ============================================================

WAVELET_MODEL_PATH = (
    r"C:\Users\jagad\Desktop"
    r"\EEG_FINAL_OUTPUT"
    r"\wavelet_results"
    r"\wavelet_model.keras"
)

HYBRID_MODEL_PATH = (
    r"C:\Users\jagad\Desktop"
    r"\EEG_FINAL_OUTPUT"
    r"\hybrid_results"
    r"\hybrid_model.keras"
)

# ============================================================
# SETTINGS
# ============================================================

FS = 256

WINDOW_SIZE = 1024
STEP_SIZE = 512

LOWCUT = 0.5
HIGHCUT = 40

NOTCH_FREQ = 50

SEIZURE_THRESHOLD = 0.25

WAVELET = "db4"
LEVEL = 5

# ============================================================
# FOCAL LOSS
# ============================================================

def focal_loss(alpha=0.85,
               gamma=2.5):

    def loss(y_true,
             y_pred):

        y_true = tf.cast(
            y_true,
            tf.float32
        )

        epsilon = 1e-7

        y_pred = tf.clip_by_value(
            y_pred,
            epsilon,
            1.0 - epsilon
        )

        cross_entropy = (
            -y_true *
            tf.math.log(y_pred)
        )

        weight = alpha * tf.pow(
            1 - y_pred,
            gamma
        )

        focal = (
            weight *
            cross_entropy
        )

        return tf.reduce_mean(
            tf.reduce_sum(
                focal,
                axis=1
            )
        )

    return loss

# ============================================================
# FILTERS
# ============================================================

def bandpass_filter(signal,
                    lowcut,
                    highcut,
                    fs,
                    order=5):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype='band'
    )

    return filtfilt(
        b,
        a,
        signal
    )

def notch_filter(signal,
                 freq,
                 fs,
                 quality=30):

    b, a = iirnotch(
        freq,
        quality,
        fs
    )

    return filtfilt(
        b,
        a,
        signal
    )

# ============================================================
# WAVELET
# ============================================================

def adaptive_wavelet_denoise(signal):

    coeffs = pywt.wavedec(
        signal,
        WAVELET,
        level=LEVEL
    )

    sigma = (
        np.median(
            np.abs(coeffs[-1])
        ) / 0.6745
    )

    threshold = (
        sigma *
        np.sqrt(
            2 * np.log(len(signal))
        )
    )

    denoised_coeffs = [coeffs[0]]

    for i in range(1, len(coeffs)):

        scale_factor = 1 + (i * 0.2)

        adaptive_threshold = (
            threshold * scale_factor
        )

        denoised_detail = pywt.threshold(
            coeffs[i],
            adaptive_threshold,
            mode='soft'
        )

        denoised_coeffs.append(
            denoised_detail
        )

    reconstructed = pywt.waverec(
        denoised_coeffs,
        WAVELET
    )

    return reconstructed[:len(signal)]

# ============================================================
# HYBRID ATTENTION
# ============================================================

def attention_refinement(raw_signal,
                         denoised_signal):

    residual = (
        raw_signal -
        denoised_signal
    )

    energy = np.abs(
        residual
    )

    attention = (
        energy /
        (np.max(energy) + 1e-8)
    )

    attention = np.power(
        attention,
        0.7
    )

    attention = np.clip(
        attention,
        0,
        1
    )

    refined = (
        denoised_signal +
        (
            0.6 *
            attention *
            residual
        )
    )

    return (
        refined,
        residual,
        attention
    )

# ============================================================
# WINDOWING
# ============================================================

def create_windows(data,
                   labels):

    X = []
    y = []

    total_samples = data.shape[1]

    for start in range(
        0,
        total_samples - WINDOW_SIZE,
        STEP_SIZE
    ):

        end = start + WINDOW_SIZE

        segment = data[:, start:end]

        label_segment = labels[start:end]

        seizure = int(
            np.any(label_segment == 1)
        )

        X.append(segment)
        y.append(seizure)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.uint8)
    )

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource

def load_models():

    wavelet_model = tf.keras.models.load_model(
        WAVELET_MODEL_PATH,
        custom_objects={
            'loss': focal_loss()
        }
    )

    hybrid_model = tf.keras.models.load_model(
        HYBRID_MODEL_PATH,
        custom_objects={
            'loss': focal_loss()
        }
    )

    return (
        wavelet_model,
        hybrid_model
    )

wavelet_model, hybrid_model = load_models()

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload EDF File",
    type=["edf"]
)

# ============================================================
# PROCESS
# ============================================================

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".edf"
    ) as tmp_file:

        tmp_file.write(
            uploaded_file.read()
        )

        EDF_FILE = tmp_file.name

    # ========================================================
    # PATIENT
    # ========================================================

    original_name = uploaded_file.name

    patient = re.findall(
        r"(chb\d+)",
        original_name
    )[0]

    seizure_ranges = get_seizure_ranges(
        patient,
        original_name
    )

    # ========================================================
    # LOAD EDF
    # ========================================================

    raw = mne.io.read_raw_edf(
        EDF_FILE,
        preload=True,
        verbose=False
    )

    raw.rename_channels(
        {
            name: f"{name}_{i}"
            for i, name in enumerate(raw.ch_names)
        }
    )

    data = raw.get_data()

    channels = raw.ch_names

    duration_sec = (
        data.shape[1] / FS
    )

    # ========================================================
    # EDF TABLE
    # ========================================================

    st.header("EDF Information")

    info_df = pd.DataFrame({

        "Property": [

            "Patient",
            "EDF File",
            "Channels",
            "Sampling Rate",
            "Duration (sec)",
            "Seizures Found"

        ],

        "Value": [

            patient,
            original_name,
            len(channels),
            FS,
            round(duration_sec, 2),
            len(seizure_ranges)

        ]

    })

    st.table(info_df)

    # ========================================================
    # LABEL ARRAY
    # ========================================================

    labels = np.zeros(
        data.shape[1],
        dtype=np.uint8
    )

    for start_sec, end_sec in seizure_ranges:

        start_sample = int(start_sec * FS)
        end_sample = int(end_sec * FS)

        labels[start_sample:end_sample] = 1

    # ========================================================
    # PROCESSING
    # ========================================================

    st.header("Processing EEG Signals")

    wavelet_processed = []
    hybrid_processed = []

    progress = st.progress(0)

    for ch in range(data.shape[0]):

        raw_signal = data[ch]

        filtered = bandpass_filter(
            raw_signal,
            LOWCUT,
            HIGHCUT,
            FS
        )

        filtered = notch_filter(
            filtered,
            NOTCH_FREQ,
            FS
        )

        # ----------------------------------------------------
        # WAVELET
        # ----------------------------------------------------

        wavelet_signal = adaptive_wavelet_denoise(
            filtered
        )

        wavelet_processed.append(
            wavelet_signal
        )

        # ----------------------------------------------------
        # HYBRID
        # ----------------------------------------------------

        hybrid_signal, residual, attention = (
            attention_refinement(
                filtered,
                wavelet_signal
            )
        )

        scaler = StandardScaler()

        hybrid_normalized = scaler.fit_transform(
            hybrid_signal.reshape(-1,1)
        ).flatten()

        hybrid_processed.append(
            hybrid_normalized
        )

        progress.progress(
            (ch + 1) / data.shape[0]
        )

    # ========================================================
    # ARRAYS
    # ========================================================

    wavelet_processed = np.array(
        wavelet_processed,
        dtype=np.float32
    )

    hybrid_processed = np.array(
        hybrid_processed,
        dtype=np.float32
    )

    # ========================================================
    # NORMALIZE WAVELET
    # ========================================================

    mean = np.mean(
        wavelet_processed,
        axis=1,
        keepdims=True
    )

    std = np.std(
        wavelet_processed,
        axis=1,
        keepdims=True
    )

    wavelet_processed = (
        wavelet_processed - mean
    ) / (std + 1e-8)

    # ========================================================
    # WINDOWS
    # ========================================================

    X_wavelet, y_true = create_windows(
        wavelet_processed,
        labels
    )

    X_hybrid, _ = create_windows(
        hybrid_processed,
        labels
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    st.header("Running Predictions")

    wavelet_prob = wavelet_model.predict(
        X_wavelet
    )

    hybrid_prob = hybrid_model.predict(
        X_hybrid
    )

    wavelet_pred = (
        wavelet_prob[:,1] >
        SEIZURE_THRESHOLD
    ).astype(int)

    hybrid_pred = (
        hybrid_prob[:,1] >
        SEIZURE_THRESHOLD
    ).astype(int)

    # ========================================================
    # METRICS
    # ========================================================

    def evaluate(y_true,
                 y_pred,
                 y_prob):

        return {

            "Accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

            "Precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

            "Recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

            "F1 Score":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

            "ROC-AUC":
            roc_auc_score(
                y_true,
                y_prob[:,1]
            )

        }

    wavelet_metrics = evaluate(
        y_true,
        wavelet_pred,
        wavelet_prob
    )

    hybrid_metrics = evaluate(
        y_true,
        hybrid_pred,
        hybrid_prob
    )

    # ========================================================
    # METRICS TABLE
    # ========================================================

    st.header("Performance Comparison")

    metrics_df = pd.DataFrame({

        "Metric": list(
            wavelet_metrics.keys()
        ),

        "Wavelet": [

            round(v,4)
            for v in wavelet_metrics.values()
        ],

        "Hybrid": [

            round(v,4)
            for v in hybrid_metrics.values()
        ]

    })

    st.table(metrics_df)

    # ========================================================
    # FINAL PREDICTIONS
    # ========================================================

    st.header("Final Prediction")

    wavelet_detected = (
        np.sum(wavelet_pred) > 0
    )

    hybrid_detected = (
        np.sum(hybrid_pred) > 0
    )

    st.subheader("Wavelet Model")

    st.write(
        "SEIZURE DETECTED"
        if wavelet_detected
        else "NO SEIZURE DETECTED"
    )

    st.write(
        f"Confidence: "
        f"{np.max(wavelet_prob[:,1])*100:.2f}%"
    )

    st.subheader("Hybrid Model")

    st.write(
        "SEIZURE DETECTED"
        if hybrid_detected
        else "NO SEIZURE DETECTED"
    )

    st.write(
        f"Confidence: "
        f"{np.max(hybrid_prob[:,1])*100:.2f}%"
    )

    # ========================================================
    # COMPARISON PLOTS
    # ========================================================

    st.header("Comparison Plots")

    channel_index = st.slider(
        "Select Channel",
        0,
        data.shape[0]-1,
        0
    )

    raw_signal = data[channel_index]

    filtered = bandpass_filter(
        raw_signal,
        LOWCUT,
        HIGHCUT,
        FS
    )

    filtered = notch_filter(
        filtered,
        NOTCH_FREQ,
        FS
    )

    wavelet_signal = adaptive_wavelet_denoise(
        filtered
    )

    hybrid_signal, residual, attention = (
        attention_refinement(
            filtered,
            wavelet_signal
        )
    )

    # ========================================================
    # RAW / WAVELET / HYBRID
    # ========================================================

    fig1, ax = plt.subplots(
        3,
        1,
        figsize=(15,10)
    )

    ax[0].plot(raw_signal[:3000])
    ax[0].set_title("Raw EEG")

    ax[1].plot(wavelet_signal[:3000])
    ax[1].set_title("Wavelet EEG")

    ax[2].plot(hybrid_signal[:3000])
    ax[2].set_title("Hybrid EEG")

    st.pyplot(fig1)

    # ========================================================
    # PSD
    # ========================================================

    f1, p1 = welch(
        filtered,
        FS
    )

    f2, p2 = welch(
        wavelet_signal,
        FS
    )

    f3, p3 = welch(
        hybrid_signal,
        FS
    )

    fig2, ax2 = plt.subplots(
        figsize=(12,5)
    )

    ax2.semilogy(
        f1,
        p1,
        label='Raw'
    )

    ax2.semilogy(
        f2,
        p2,
        label='Wavelet'
    )

    ax2.semilogy(
        f3,
        p3,
        label='Hybrid'
    )

    ax2.legend()

    ax2.set_title(
        "PSD Comparison"
    )

    st.pyplot(fig2)

    # ========================================================
    # ATTENTION MAP
    # ========================================================

    fig3, ax3 = plt.subplots(
        figsize=(12,4)
    )

    ax3.plot(attention[:3000])

    ax3.set_title(
        "Hybrid Attention Weights"
    )

    st.pyplot(fig3)

    # ========================================================
    # SPECTROGRAM
    # ========================================================

    f, t, Sxx = spectrogram(
        hybrid_signal,
        FS
    )

    fig4, ax4 = plt.subplots(
        figsize=(12,5)
    )

    mesh = ax4.pcolormesh(
        t,
        f,
        10*np.log10(Sxx),
        shading='gouraud'
    )

    ax4.set_title(
        "Hybrid Spectrogram"
    )

    fig4.colorbar(mesh)

    st.pyplot(fig4)

    # ========================================================
    # CONFUSION MATRICES
    # ========================================================

    st.header("Confusion Matrices")

    wavelet_cm = confusion_matrix(
        y_true,
        wavelet_pred
    )

    hybrid_cm = confusion_matrix(
        y_true,
        hybrid_pred
    )

    st.subheader("Wavelet")

    st.write(wavelet_cm)

    st.subheader("Hybrid")

    st.write(hybrid_cm)

    # ========================================================
    # CLEANUP
    # ========================================================

    del data
    del wavelet_processed
    del hybrid_processed
    del X_wavelet
    del X_hybrid

    gc.collect()