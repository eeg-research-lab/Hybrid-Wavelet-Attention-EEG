# ============================================================
# preprocess_hybrid.py
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

from sklearn.preprocessing import StandardScaler

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
# CREATE OUTPUT FOLDERS
# ============================================================

NPZ_ROOT = os.path.join(
    OUTPUT_ROOT,
    "hybrid_npz"
)

PLOT_ROOT = os.path.join(
    OUTPUT_ROOT,
    "hybrid_plots"
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
LEVEL = 4

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

    return filtfilt(
        b,
        a,
        signal
    )

# ============================================================
# NOTCH FILTER
# ============================================================

def notch_filter(signal,
                 freq,
                 fs,
                 quality=30):

    nyquist = 0.5 * fs

    w0 = freq / nyquist

    b, a = iirnotch(
        w0,
        quality
    )

    return filtfilt(
        b,
        a,
        signal
    )

# ============================================================
# WAVELET DENOISING
# ============================================================

def wavelet_denoise(signal):

    coeffs = pywt.wavedec(
        signal,
        WAVELET,
        level=LEVEL
    )

    sigma = np.median(
        np.abs(coeffs[-1])
    ) / 0.6745

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
print(" FINAL SEQUENTIAL HYBRID PIPELINE ")
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

        print("\nLoading EDF...\n")

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

        fs = int(
            raw.info['sfreq']
        )

        data = raw.get_data()

        channels = raw.ch_names

        print(f"Channels: {len(channels)}")
        print(f"Sampling Rate: {fs}")
        print(f"Shape: {data.shape}")

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
        # PROCESS CHANNELS
        # ====================================================

        processed_channels = []

        all_snr = []
        all_rmse = []
        all_corr = []

        print("\nProcessing Channels...\n")

        for ch in range(data.shape[0]):

            print(f"Processing Channel {ch+1}")

            raw_signal = data[ch]

            filtered = bandpass_filter(
                raw_signal,
                LOWCUT,
                HIGHCUT,
                fs
            )

            filtered = notch_filter(
                filtered,
                NOTCH_FREQ,
                fs
            )

            wavelet_signal = wavelet_denoise(
                filtered
            )

            hybrid_signal, residual, attention = (
                attention_refinement(
                    filtered,
                    wavelet_signal
                )
            )

            snr = calculate_snr(
                filtered,
                hybrid_signal
            )

            rmse = calculate_rmse(
                filtered,
                hybrid_signal
            )

            corr = calculate_correlation(
                filtered,
                hybrid_signal
            )

            all_snr.append(snr)
            all_rmse.append(rmse)
            all_corr.append(corr)

            print(f"SNR: {snr:.2f} dB")
            print(f"RMSE: {rmse:.8f}")
            print(f"Correlation: {corr:.4f}")

            scaler = StandardScaler()

            hybrid_normalized = scaler.fit_transform(
                hybrid_signal.reshape(-1, 1)
            ).flatten()

            processed_channels.append(
                hybrid_normalized
            )

            # =================================================
            # PLOT 1
            # =================================================

            plt.figure(figsize=(15,6))

            plt.subplot(2,1,1)

            plt.plot(filtered[:3000])

            plt.title(
                f"Raw EEG - Channel {ch+1}"
            )

            plt.subplot(2,1,2)

            plt.plot(hybrid_signal[:3000])

            plt.title(
                f"Hybrid EEG - Channel {ch+1}"
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    EDF_PLOT_FOLDER,
                    f"channel_{ch+1}_raw_vs_hybrid.png"
                )
            )

            plt.close()

            # =================================================
            # PLOT 2 ATTENTION
            # =================================================

            plt.figure(figsize=(15,5))

            plt.plot(attention[:3000])

            plt.title(
                f"Attention Weights - Channel {ch+1}"
            )

            plt.savefig(
                os.path.join(
                    EDF_PLOT_FOLDER,
                    f"channel_{ch+1}_attention.png"
                )
            )

            plt.close()

            # =================================================
            # PLOT 3 PSD
            # =================================================

            f1, p1 = welch(
                filtered,
                fs
            )

            f2, p2 = welch(
                hybrid_signal,
                fs
            )

            plt.figure(figsize=(12,5))

            plt.semilogy(
                f1,
                p1,
                label='Raw'
            )

            plt.semilogy(
                f2,
                p2,
                label='Hybrid'
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

        # ====================================================
        # FINAL ARRAY
        # ====================================================

        processed_data = np.array(
            processed_channels,
            dtype=np.float32
        )

        print("\nProcessed Shape:")
        print(processed_data.shape)

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
            processed_data,
            labels
        )

        print("\n3D Dataset Shape:")
        print(X.shape)

        print(
            f"\nSeizure Windows: "
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
            f"{edf_name}_hybrid.npz"
        )

        np.savez_compressed(
            npz_path,

            X=X,
            y=y,

            channels=np.array(channels),

            patient=patient,

            edf_file=edf_file,

            seizure_ranges=np.array(
                seizure_ranges,
                dtype=object
            ),

            fs=fs
        )

        # ====================================================
        # FINAL METRICS
        # ====================================================

        print("\n========== FINAL METRICS ==========\n")

        print(
            f"Average SNR: "
            f"{np.mean(all_snr):.2f} dB"
        )

        print(
            f"Average RMSE: "
            f"{np.mean(all_rmse):.8f}"
        )

        print(
            f"Average Correlation: "
            f"{np.mean(all_corr):.4f}"
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

        del data
        del processed_data
        del X
        del y
        del raw
        del labels

        gc.collect()

        print("\nMemory Cleared.")

        print_bar()

# ============================================================
# FINAL SUMMARY
# ============================================================

pipeline_time = (
    time.time() - pipeline_start
) / 3600

print_bar()
print(" FINAL HYBRID PIPELINE COMPLETE ")
print_bar()

print(
    f"\nTotal Pipeline Time: "
    f"{pipeline_time:.2f} hours"
)

print("\nHYBRID PIPELINE COMPLETE.")
print_bar()