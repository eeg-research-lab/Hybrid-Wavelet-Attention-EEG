# ============================================================
# utils.py
# ============================================================

import gc
import numpy as np
import pywt
import mne

from datetime import datetime

from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import iirnotch

# ============================================================
# GLOBAL SETTINGS
# ============================================================

FS = 256

LOWCUT = 0.5
HIGHCUT = 40

NOTCH_FREQ = 50

WAVELET = "db4"
LEVEL = 4

WINDOW_SIZE = 1024

NORMAL_STEP = 512
SEIZURE_STEP = 64

TARGET_CHANNELS = 23

# ============================================================
# TERMINAL BAR
# ============================================================

def print_bar():

    print("=" * 60)

# ============================================================
# LOGGER
# ============================================================

def log(message):

    current_time = datetime.now().strftime(
        "%H:%M:%S"
    )

    print(f"[{current_time}] {message}")

# ============================================================
# FIX DUPLICATE CHANNELS
# ============================================================

def fix_duplicate_channels(raw):

    unique_names = {}
    new_names = {}

    for ch in raw.ch_names:

        if ch not in unique_names:

            unique_names[ch] = 0
            new_names[ch] = ch

        else:

            unique_names[ch] += 1

            new_names[ch] = (
                f"{ch}_{unique_names[ch]}"
            )

    raw.rename_channels(new_names)

    return raw

# ============================================================
# BANDPASS FILTER
# ============================================================

def bandpass_filter(signal,
                    lowcut=LOWCUT,
                    highcut=HIGHCUT,
                    fs=FS,
                    order=5):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype='band'
    )

    filtered = filtfilt(
        b,
        a,
        signal
    )

    return filtered

# ============================================================
# NOTCH FILTER
# ============================================================

def notch_filter(signal,
                 freq=NOTCH_FREQ,
                 fs=FS,
                 quality=30):

    w0 = freq / (0.5 * fs)

    b, a = iirnotch(
        w0,
        quality
    )

    filtered = filtfilt(
        b,
        a,
        signal
    )

    return filtered

# ============================================================
# WAVELET DENOISING
# ============================================================

def wavelet_denoise(signal):

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

    for c in coeffs[1:]:

        denoised = pywt.threshold(
            c,
            threshold,
            mode='soft'
        )

        denoised_coeffs.append(
            denoised
        )

    reconstructed = pywt.waverec(
        denoised_coeffs,
        WAVELET
    )

    reconstructed = reconstructed[
        :len(signal)
    ]

    return reconstructed

# ============================================================
# HYBRID ATTENTION REFINEMENT
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
# Z-SCORE NORMALIZATION
# ============================================================

def normalize_signal(signal):

    mean = np.mean(signal)
    std = np.std(signal)

    normalized = (
        signal - mean
    ) / (std + 1e-8)

    return normalized.astype(np.float32)

# ============================================================
# LOAD EDF
# ============================================================

def load_edf(edf_path):

    log(f"Loading EDF: {edf_path}")

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        verbose=False
    )

    raw = fix_duplicate_channels(raw)

    data = raw.get_data()

    channels = raw.ch_names

    if data.shape[0] >= TARGET_CHANNELS:

        data = data[:TARGET_CHANNELS]

        channels = channels[:TARGET_CHANNELS]

    log(f"EEG Shape: {data.shape}")

    return (
        raw,
        data,
        channels
    )

# ============================================================
# CREATE LABEL ARRAY
# ============================================================

def create_label_array(total_samples,
                       seizure_ranges):

    labels = np.zeros(
        total_samples,
        dtype=np.uint8
    )

    for start_sec, end_sec in seizure_ranges:

        start_sample = int(start_sec * FS)

        end_sample = int(end_sec * FS)

        labels[start_sample:end_sample] = 1

    return labels

# ============================================================
# ADAPTIVE WINDOWING
# ============================================================

def create_windows(data,
                   labels):

    X = []
    y = []

    idx = 0

    total_samples = data.shape[1]

    seizure_windows = 0
    normal_windows = 0

    while idx + WINDOW_SIZE <= total_samples:

        eeg_window = data[
            :,
            idx:idx + WINDOW_SIZE
        ]

        label_window = labels[
            idx:idx + WINDOW_SIZE
        ]

        seizure = int(
            np.any(label_window == 1)
        )

        X.append(
            eeg_window.T
        )

        y.append(seizure)

        if seizure:

            idx += SEIZURE_STEP
            seizure_windows += 1

        else:

            idx += NORMAL_STEP
            normal_windows += 1

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.uint8
    )

    return (
        X,
        y,
        seizure_windows,
        normal_windows
    )

# ============================================================
# METRICS
# ============================================================

def calculate_snr(original,
                  processed):

    signal_power = np.mean(
        original ** 2
    )

    noise_power = np.mean(
        (
            original -
            processed
        ) ** 2
    )

    if noise_power == 0:
        return 0

    return 10 * np.log10(
        signal_power /
        noise_power
    )

def calculate_rmse(original,
                   processed):

    return np.sqrt(
        np.mean(
            (
                original -
                processed
            ) ** 2
        )
    )

def calculate_correlation(original,
                          processed):

    return np.corrcoef(
        original,
        processed
    )[0,1]

# ============================================================
# MEMORY CLEANUP
# ============================================================

def clear_memory(*variables):

    for var in variables:

        del var

    gc.collect()

# ============================================================
# DEBUG TEST
# ============================================================

if __name__ == "__main__":

    print_bar()
    print(" EEG UTILITY SYSTEM ")
    print_bar()

    log("Utility module loaded successfully.")

    print_bar()
    log("READY.")
    print_bar()