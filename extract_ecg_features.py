import os
import random
import pandas as pd
import numpy as np
import scipy.io as sio
import wfdb
from scipy.signal import find_peaks

# Paths
adult_base_dir = r"C:\Users\zoorab\Downloads\g12ecg_testing_samples\G12ECG_testing_samples"
pediatric_csv_path = r"C:\Users\zoorab\Desktop\zoher\University\Projects\ECGFounder\extracted_pediatic_ecg_with_labels.csv"
pediatric_base_dir = r"C:\Users\zoorab\Desktop\zoher\University\Projects\Zhengzhou_ECG\Child_ecg"

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

def extract_features(signal, fs):
    # Normalize orientation if necessary to (12, N)
    if signal.shape[1] == 12:
        signal = signal.T
        
    # Standardize scale: subtract median baseline to center signal
    signal = signal - np.median(signal, axis=1, keepdims=True)
    
    # 1. Peak-to-Peak Amplitudes (across all 12 leads)
    ptp_amps = np.ptp(signal, axis=1)
    mean_amp = np.mean(ptp_amps)
    max_amp = np.max(ptp_amps)
    
    # 2. Heart Rate Estimation using Lead II (Index 1)
    lead_ii = signal[1]
    
    # Find peaks. A reasonable distance for RR is ~0.3 seconds at max 200 BPM
    # Prominence tries to grab major R-peaks
    prominence_val = np.ptp(lead_ii) * 0.4 
    peaks, _ = find_peaks(lead_ii, distance=int(fs*0.3), prominence=prominence_val)
    
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        hr = 60.0 / np.mean(rr_intervals)
        rr_std = np.std(rr_intervals)
    else:
        hr = np.nan
        rr_std = np.nan
        
    return {
        'Heart Rate (BPM)': hr,
        'RR Variability (s)': rr_std,
        'Mean Peak-to-Peak Amp': mean_amp,
        'Max Peak-to-Peak Amp': max_amp
    }

def main():
    ped_df = pd.read_csv(pediatric_csv_path)
    adult_folders = [f for f in os.listdir(adult_base_dir) if os.path.isdir(os.path.join(adult_base_dir, f))]
    
    results = []
    
    for folder in adult_folders:
        disease_name = LABEL_MAPPING.get(folder)
        if not disease_name:
            continue
            
        matches = ped_df[ped_df['Target_Category'].str.lower() == disease_name.lower()]
        if matches.empty:
            continue
            
        # Select Random Pediatric file
        ped_sample = matches.sample(1).iloc[0]
        ped_filename = ped_sample['Filename']
        ped_full_path = os.path.join(pediatric_base_dir, ped_filename)
        
        if not os.path.exists(ped_full_path + ".dat"):
            continue
            
        try:
            rec = wfdb.rdrecord(ped_full_path)
            ped_signal = rec.p_signal
            ped_fs = rec.fs
        except Exception:
            continue
            
        # Select Random Adult file
        adult_folder_path = os.path.join(adult_base_dir, folder)
        adult_files = [f for f in os.listdir(adult_folder_path) if f.endswith(".mat")]
        if not adult_files:
            continue
            
        adult_filename = random.choice(adult_files)
        adult_full_path = os.path.join(adult_folder_path, adult_filename)
        
        try:
            mat_data = sio.loadmat(adult_full_path)
            adult_signal = mat_data['val']
            adult_fs = 500  # Default frequency
        except Exception:
            continue
            
        # Extract features
        adult_features = extract_features(adult_signal, adult_fs)
        ped_features = extract_features(ped_signal, ped_fs)
        
        # Append to results
        results.append({
            'Disease': disease_name,
            'Adult HR (BPM)': adult_features['Heart Rate (BPM)'],
            'Peds HR (BPM)': ped_features['Heart Rate (BPM)'],
            'Adult Max_Amp': adult_features['Max Peak-to-Peak Amp'],
            'Peds Max_Amp': ped_features['Max Peak-to-Peak Amp'],
            'Adult HRV (s)': adult_features['RR Variability (s)'],
            'Peds HRV (s)': ped_features['RR Variability (s)']
        })
        
    df_results = pd.DataFrame(results).round(2)
    print("\n--- Extracted Feature Comparison (Adult vs Pediatric) ---")
    print(df_results.to_markdown(index=False))

if __name__ == "__main__":
    main()
