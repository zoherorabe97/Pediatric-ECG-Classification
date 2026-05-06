import os
import random
import pandas as pd
import numpy as np
import scipy.io as sio
import wfdb
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks

# =====================================================
# PATHS
# =====================================================

adult_base_dir = r"C:\Users\zoorab\Downloads\g12ecg_testing_samples\G12ECG_testing_samples"
pediatric_csv_path = r"C:\Users\zoorab\Desktop\zoher\University\Projects\ECGFounder\extracted_few_pediatic_ecg_with_labels.csv"
pediatric_base_dir = r"C:\Users\zoorab\Desktop\zoher\University\Projects\Zhengzhou_ECG\Child_ecg"
output_dir = r"C:\Users\zoorab\Desktop\zoher\University\Projects\ECGFounder\comparisons_final"

os.makedirs(output_dir, exist_ok=True)

# =====================================================
# LABEL MAPPING
# =====================================================

LABEL_MAPPING = {
    '1st_degree_av_block': 'First-degree AV block',
    'atrial_fibrillation': 'Atrial Fibrillation',
    'atrial_flutter': 'Atrial Flutter',
    'incomplete_right_bundle_branch_block': 'Incomplete Right Bundle Branch Block',
    'left_anterior_fascicular_block': 'Left Anterior Fascicular Block',
    'left_axis_deviation': 'Left Axis Deviation',
    'left_bundle_branch_block': 'Left Bundle Branch Block',
    'low_qrs_voltages': 'Low QRS Voltages',
    'premature_atrial_contraction': 'Premature Atrial Contraction',
    'right_axis_deviation': 'Right Axis Deviation',
    'right_bundle_branch_block': 'Right Bundle Branch Block',
    'sinus_arrhythmia': 'Sinus Arrhythmia',
    'sinus_bradycardia': 'Sinus Bradycardia',
    'sinus_rhythm': 'Sinus Rhythm',
    'sinus_tachycardia': 'Sinus Tachycardia',
    't_wave_abnormal': 'T-wave abnormality',
    't_wave_inversion': 'T-wave inversion'
}

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
         'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# =====================================================
# HELPERS
# =====================================================

def load_pediatric_csv(csv_path):
    return pd.read_csv(csv_path)


def bandpass_filter(signal, fs=500):
    """
    signal shape = (12, N)
    """
    low = 0.5 / (fs / 2)
    high = 40 / (fs / 2)

    b, a = butter(3, [low, high], btype='band')
    filtered = filtfilt(b, a, signal, axis=1)

    return filtered


def normalize_per_lead(signal):
    """
    Normalize each lead independently
    """
    mx = np.max(np.abs(signal), axis=1, keepdims=True)
    mx[mx == 0] = 1
    return signal / mx


def detect_r_peak(signal_lead, fs):
    """
    Detect one strong R peak from Lead II
    """
    peaks, _ = find_peaks(signal_lead,
                          distance=int(0.4 * fs),
                          prominence=np.std(signal_lead))

    if len(peaks) == 0:
        return len(signal_lead) // 2

    return peaks[len(peaks) // 2]


def extract_centered_beat(signal, fs, beat_window=1.2):
    """
    Extract one beat centered around R peak
    signal shape = (12, N)
    """
    lead2 = signal[1, :]   # Lead II

    r_peak = detect_r_peak(lead2, fs)

    half = int((beat_window / 2) * fs)

    start = max(0, r_peak - half)
    end = min(signal.shape[1], r_peak + half)

    beat = signal[:, start:end]

    return beat


def resample_to_same_length(sig1, sig2):
    """
    Make both same width
    """
    target = min(sig1.shape[1], sig2.shape[1])

    sig1 = sig1[:, :target]
    sig2 = sig2[:, :target]

    return sig1, sig2


# =====================================================
# MAIN PLOT
# =====================================================

def plot_fair_comparison(adult_signal,
                         pediatric_signal,
                         adult_name,
                         ped_name,
                         disease_name,
                         adult_fs=500,
                         ped_fs=500):

    # Pediatric may come as (N,12)
    if pediatric_signal.shape[1] == 12:
        pediatric_signal = pediatric_signal.T

    # ----------------------------
    # FILTER
    # ----------------------------
    adult_signal = bandpass_filter(adult_signal, adult_fs)
    pediatric_signal = bandpass_filter(pediatric_signal, ped_fs)

    # ----------------------------
    # NORMALIZE
    # ----------------------------
    adult_signal = normalize_per_lead(adult_signal)
    pediatric_signal = normalize_per_lead(pediatric_signal)

    # ----------------------------
    # EXTRACT REPRESENTATIVE BEAT
    # ----------------------------
    adult_beat = extract_centered_beat(adult_signal, adult_fs)
    ped_beat = extract_centered_beat(pediatric_signal, ped_fs)

    # ----------------------------
    # SAME LENGTH
    # ----------------------------
    adult_beat, ped_beat = resample_to_same_length(adult_beat, ped_beat)

    N = adult_beat.shape[1]
    t = np.arange(N) / adult_fs

    # ----------------------------
    # PLOT
    # ----------------------------
    fig, axes = plt.subplots(12, 1, figsize=(14, 24), sharex=True)

    fig.suptitle(
        f"{disease_name}\nFair Adult vs Pediatric ECG Comparison",
        fontsize=20,
        fontweight='bold'
    )

    for i in range(12):

        axes[i].plot(t, ped_beat[i], color='blue',
                     linewidth=1.4, label='Pediatric')

        axes[i].plot(t, adult_beat[i], color='red',
                     linewidth=1.2, alpha=0.85, label='Adult')

        axes[i].set_ylabel(LEADS[i], rotation=0, labelpad=20, fontsize=11)

        axes[i].grid(True, linestyle='--', alpha=0.4)

        axes[i].set_ylim(-1.3, 1.3)

        if i == 0:
            axes[i].legend(loc='upper right', fontsize=10)

    axes[-1].set_xlabel("Time (seconds)", fontsize=12)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])

    safe = disease_name.replace(" ", "_").replace("-", "_")

    out = os.path.join(output_dir, f"{safe}_fair_comparison.png")

    plt.savefig(out, dpi=200)
    plt.close()

    print("Saved:", out)


# =====================================================
# MAIN LOOP
# =====================================================

def main():

    ped_df = load_pediatric_csv(pediatric_csv_path)

    adult_folders = [
        f for f in os.listdir(adult_base_dir)
        if os.path.isdir(os.path.join(adult_base_dir, f))
    ]

    for folder in adult_folders:

        disease_name = LABEL_MAPPING.get(folder)

        if not disease_name:
            print("Skipping:", folder)
            continue

        print(f"\nProcessing: {disease_name}")

        # --------------------------------------
        # Match pediatric
        # --------------------------------------
        matches = ped_df[
            ped_df['Target_Category'].str.lower()
            == disease_name.lower()
        ]

        if matches.empty:
            print("No pediatric match.")
            continue

        # Reproducible first sample
        ped_sample = matches.iloc[0]

        ped_filename = ped_sample['Filename']
        ped_full = os.path.join(pediatric_base_dir, ped_filename)

        if not os.path.exists(ped_full + ".dat"):
            print("Missing pediatric file.")
            continue

        try:
            rec = wfdb.rdrecord(ped_full)
            pediatric_signal = rec.p_signal
            ped_fs = rec.fs
        except Exception as e:
            print("Pediatric load error:", e)
            continue

        # --------------------------------------
        # Adult sample
        # --------------------------------------
        adult_folder = os.path.join(adult_base_dir, folder)

        adult_files = sorted([
            f for f in os.listdir(adult_folder)
            if f.endswith(".mat")
        ])

        if not adult_files:
            continue

        adult_file = adult_files[0]
        adult_full = os.path.join(adult_folder, adult_file)

        try:
            mat = sio.loadmat(adult_full)
            adult_signal = mat['val']
        except Exception as e:
            print("Adult load error:", e)
            continue

        print("Adult:", adult_file)
        print("Pediatric:", os.path.basename(ped_filename))

        plot_fair_comparison(
            adult_signal,
            pediatric_signal,
            adult_file,
            os.path.basename(ped_filename),
            disease_name,
            adult_fs=500,
            ped_fs=int(ped_fs)
        )


if __name__ == "__main__":
    main()