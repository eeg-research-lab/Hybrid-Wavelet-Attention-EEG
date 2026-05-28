# ============================================================
# preprocess_wavelet.py
# ============================================================

import os
import gc
import time
import numpy as np
import mne
import pywt

# ============================================================
# MATPLOTLIB RAM FIX
# ============================================================

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import iirnotch
from scipy.signal import welch
from scipy.signal import spectrogram

from labels import (
    TRAIN_PATIENTS,
    get_patient_edf_files,
    get_edf_path,
    get_seizure_ranges
)

# ============================================================
# OUTPUT ROOT
# ============================================================

OUTPUT_ROOT = r"C:\Users\jagad\Desktop\EEG_FINAL_OUTPUT"

# ============================================================
# OUTPUT FOLDERS
# ============================================================

NPZ_ROOT = os.path.join(
    OUTPUT_ROOT,
    "wavelet_npz"
)

PLOT_ROOT = os.path.join(
    OUTPUT_ROOT,
    "wavelet_plots"
)

os.makedirs(NPZ_ROOT, exist_ok=True)
os.makedirs(PLOT_ROOT, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

FS = 256

WINDOW_SIZE = 1024

# ============================================================
# RESTORED ORIGINAL VALIDATED LOGIC
# ============================================================

NORMAL_STEP = 512
SEIZURE_STEP = 64

LOWCUT = 0.5
HIGHCUT = 40

NOTCH_FREQ = 50

WAVELET = "db4"
LEVEL = 5

# ============================================================
# TERMINAL BAR
# ============================================================

def print_bar():

    print("=" * 60)

# ============================================================
# BANDPASS FILTER
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
                 freq,
                 fs,
                 quality=30):

    b, a = iirnotch(
        freq,
        quality,
        fs
    )

    filtered = filtfilt(
        b,
        a,
        signal
    )

    return filtered

# ============================================================
# ADAPTIVE WAVELET DENOISING
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

    reconstructed = reconstructed[
        :len(signal)
    ]

    return reconstructed

# ============================================================
# SNR
# ============================================================

def calculate_snr(raw_signal,
                  processed_signal):

    noise = (
        raw_signal -
        processed_signal
    )

    signal_power = np.mean(
        processed_signal ** 2
    )

    noise_power = np.mean(
        noise ** 2
    )

    if noise_power == 0:
        return 0

    snr = 10 * np.log10(
        signal_power / noise_power
    )

    return snr

# ============================================================
# RMSE
# ============================================================

def calculate_rmse(raw_signal,
                   processed_signal):

    rmse = np.sqrt(
        np.mean(
            (raw_signal - processed_signal) ** 2
        )
    )

    return rmse

# ============================================================
# CORRELATION
# ============================================================

def calculate_correlation(raw_signal,
                          processed_signal):

    correlation = np.corrcoef(
        raw_signal,
        processed_signal
    )[0,1]

    return correlation

# ============================================================
# RESTORED ORIGINAL ADAPTIVE WINDOWING
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

        X.append(eeg_window)
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
# START
# ============================================================

print_bar()
print(" FINAL SEQUENTIAL WAVELET PIPELINE ")
print_bar()

pipeline_start = time.time()

# ============================================================
# PATIENT LOOP
# ============================================================

for patient in TRAIN_PATIENTS:

    print_bar()
    print(f"PATIENT: {patient}")
    print_bar()

    patient_npz_folder = os.path.join(
        NPZ_ROOT,
        patient
    )

    patient_plot_folder = os.path.join(
        PLOT_ROOT,
        patient
    )

    os.makedirs(
        patient_npz_folder,
        exist_ok=True
    )

    os.makedirs(
        patient_plot_folder,
        exist_ok=True
    )

    edf_files = get_patient_edf_files(
        patient
    )

    print(f"\nEDF Files Found: {len(edf_files)}")

    # ========================================================
    # EDF LOOP
    # ========================================================

    for idx, edf_file in enumerate(edf_files):

        edf_start = time.time()

        print_bar()

        print(
            f"[{idx+1:03d}/{len(edf_files)}] "
            f"{edf_file}"
        )

        print_bar()

        seizure_ranges = get_seizure_ranges(
            patient,
            edf_file
        )

        print(
            f"Seizures Found: "
            f"{len(seizure_ranges)}"
        )

        EDF_FILE = get_edf_path(
            patient,
            edf_file
        )

        # ====================================================
        # LOAD EDF
        # ====================================================

        print("\nLoading EDF...")

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

        channel_names = raw.ch_names

        print("\nChannels:", len(channel_names))
        print("Raw Shape:", data.shape)

        raw_data = data.copy()

        # ====================================================
        # LABEL ARRAY
        # ====================================================

        labels = np.zeros(
            data.shape[1],
            dtype=np.uint8
        )

        for start_sec, end_sec in seizure_ranges:

            start_sample = int(start_sec * FS)
            end_sample = int(end_sec * FS)

            labels[start_sample:end_sample] = 1

        # ====================================================
        # EDF PLOT FOLDER
        # ====================================================

        edf_name = os.path.basename(
            EDF_FILE
        ).replace(".edf", "")

        EDF_PLOT_FOLDER = os.path.join(
            patient_plot_folder,
            edf_name
        )

        os.makedirs(
            EDF_PLOT_FOLDER,
            exist_ok=True
        )

        # ====================================================
        # PREPROCESSING
        # ====================================================

        processed = []

        snr_list = []
        rmse_list = []
        corr_list = []

        print("\nProcessing Channels...\n")

        for ch in range(data.shape[0]):

            print(f"Processing Channel {ch+1}")

            signal = data[ch]

            signal = bandpass_filter(
                signal,
                LOWCUT,
                HIGHCUT,
                FS
            )

            signal = notch_filter(
                signal,
                NOTCH_FREQ,
                FS
            )

            wavelet_signal = adaptive_wavelet_denoise(
                signal
            )

            processed.append(
                wavelet_signal
            )

            snr = calculate_snr(
                raw_data[ch],
                wavelet_signal
            )

            rmse = calculate_rmse(
                raw_data[ch],
                wavelet_signal
            )

            corr = calculate_correlation(
                raw_data[ch],
                wavelet_signal
            )

            snr_list.append(snr)
            rmse_list.append(rmse)
            corr_list.append(corr)

            # =================================================
            # PLOT 1
            # =================================================

            plt.figure(figsize=(15,6))

            plt.subplot(2,1,1)

            plt.plot(
                raw_data[ch][:3000]
            )

            plt.title(
                f"Raw EEG - Channel {ch+1}"
            )

            plt.subplot(2,1,2)

            plt.plot(
                wavelet_signal[:3000]
            )

            plt.title(
                f"Wavelet EEG - Channel {ch+1}"
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    EDF_PLOT_FOLDER,
                    f"channel_{ch+1}_raw_vs_wavelet.png"
                )
            )

            plt.close()

            # =================================================
            # PLOT 2 PSD
            # =================================================

            f1, pxx1 = welch(
                raw_data[ch],
                FS
            )

            f2, pxx2 = welch(
                wavelet_signal,
                FS
            )

            plt.figure(figsize=(12,5))

            plt.semilogy(
                f1,
                pxx1,
                label='Raw'
            )

            plt.semilogy(
                f2,
                pxx2,
                label='Wavelet'
            )

            plt.legend()

            plt.title(
                f"PSD Comparison - Channel {ch+1}"
            )

            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power")

            plt.savefig(
                os.path.join(
                    EDF_PLOT_FOLDER,
                    f"channel_{ch+1}_psd.png"
                )
            )

            plt.close()

            # =================================================
            # PLOT 3 SPECTROGRAM
            # =================================================

            f, t, Sxx = spectrogram(
                wavelet_signal,
                FS
            )

            plt.figure(figsize=(12,5))

            plt.pcolormesh(
                t,
                f,
                10 * np.log10(Sxx),
                shading='gouraud'
            )

            plt.title(
                f"Spectrogram - Channel {ch+1}"
            )

            plt.ylabel("Frequency (Hz)")
            plt.xlabel("Time (sec)")

            plt.colorbar(
                label='Power dB'
            )

            plt.ylim([0, 60])

            plt.savefig(
                os.path.join(
                    EDF_PLOT_FOLDER,
                    f"channel_{ch+1}_spectrogram.png"
                )
            )

            plt.close()

        # ====================================================
        # ARRAY
        # ====================================================

        processed = np.array(
            processed,
            dtype=np.float32
        )

        # ====================================================
        # NORMALIZATION
        # ====================================================

        print("\nApplying Z-score Normalization...")

        mean = np.mean(
            processed,
            axis=1,
            keepdims=True
        )

        std = np.std(
            processed,
            axis=1,
            keepdims=True
        )

        processed_normalized = (
            processed - mean
        ) / (std + 1e-8)

        processed_normalized = (
            processed_normalized.astype(
                np.float32
            )
        )

        # ====================================================
        # WINDOWING
        # ====================================================

        print("\nCreating Adaptive Windows...")

        (
            X,
            y,
            seizure_windows,
            normal_windows
        ) = create_windows(
            processed_normalized,
            labels
        )

        print("\nFinal Shape:", X.shape)

        print(
            f"Seizure Windows: "
            f"{seizure_windows}"
        )

        print(
            f"Normal Windows: "
            f"{normal_windows}"
        )

        # ====================================================
        # SAVE NPZ
        # ====================================================

        npz_path = os.path.join(
            patient_npz_folder,
            f"{edf_name}_wavelet.npz"
        )

        np.savez_compressed(

            npz_path,

            X=X,
            y=y,

            channels=np.array(channel_names),

            patient=patient,

            edf_file=edf_file,

            seizure_ranges=np.array(
                seizure_ranges,
                dtype=object
            ),

            fs=FS

        )

        # ====================================================
        # FINAL METRICS
        # ====================================================

        print("\n========== FINAL METRICS ==========")

        print(
            f"\nAverage SNR: "
            f"{np.mean(snr_list):.2f} dB"
        )

        print(
            f"\nAverage RMSE: "
            f"{np.mean(rmse_list):.8f}"
        )

        print(
            f"\nAverage Correlation: "
            f"{np.mean(corr_list):.4f}"
        )

        print("\nSaved NPZ:")
        print(npz_path)

        print("\nPlots Saved:")
        print(EDF_PLOT_FOLDER)

        edf_time = (
            time.time() - edf_start
        ) / 60

        print(
            f"\nEDF Time: "
            f"{edf_time:.2f} min"
        )

        # ====================================================
        # MEMORY CLEANUP
        # ====================================================

        plt.close('all')

        del raw_data
        del processed
        del processed_normalized
        del X
        del y
        del raw
        del data
        del labels

        gc.collect()

        print("\nMemory Cleared.")

        print_bar()

# ============================================================
# FINAL PIPELINE SUMMARY
# ============================================================

pipeline_time = (
    time.time() - pipeline_start
) / 3600

print_bar()
print(" FINAL PIPELINE COMPLETE ")
print_bar()

print(
    f"\nTotal Pipeline Time: "
    f"{pipeline_time:.2f} hours"
)

print("\nWAVELET PIPELINE COMPLETE.")
print_bar()