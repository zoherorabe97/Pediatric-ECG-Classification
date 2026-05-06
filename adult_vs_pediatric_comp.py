import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import os

base_dir = 'C:/Users/zoorab/Desktop/zoher/University/Projects/ECGFounder'

def load_and_clean(path):
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
        # Clean column names (strip spaces/hidden chars)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def find_column_safely(df, keywords):
    """Finds a column matching all keywords. Returns None if not found."""
    for col in df.columns:
        if all(k.lower() in col.lower() for k in keywords):
            return col
    return None

def parse_ci(ci_str):
    if pd.isna(ci_str): return np.nan, np.nan
    try:
        # Handle formats like "(0.8, 0.9)" or "(np.float64(0.8), ...)"
        clean_str = re.sub(r'np\.float64\(|\)', '', str(ci_str)).replace('(', '').replace(')', '')
        parts = clean_str.split(',')
        return float(parts[0].strip()), float(parts[1].strip())
    except:
        return np.nan, np.nan

def generate_comparison_plot(datasets, title, save_path):
    dfs = {}
    for name, p in datasets.items():
        df = load_and_clean(p)
        if df is not None:
            label_col = find_column_safely(df, ['label'])
            auc_col = [c for c in df.columns if 'auroc' in c.lower() and 'ci' not in c.lower()]
            ci_col = find_column_safely(df, ['auroc', 'ci'])
            
            if label_col and auc_col and ci_col:
                auc_col = auc_col[0] # Take the first matching AUROC column
                df['Label_std'] = df[label_col].astype(str).str.upper().str.strip()
                df['low'], df['high'] = zip(*df[ci_col].apply(parse_ci))
                # Handle potential duplicates by taking the first occurrence
                df = df.drop_duplicates(subset=['Label_std']).set_index('Label_std')
                dfs[name] = {'df': df, 'auc_col': auc_col}
            else:
                print(f"Missing required columns in {name} dataset. Available: {df.columns.tolist()}")
                sys.exit(1)
        else:
            sys.exit(1)

    # Find shared labels between pediatric and at least one adult dataset
    p_df = dfs['Pediatric']['df']
    shared_labels = sorted(p_df.index.tolist())

    adult_labels = set()
    for name in ['G12EC', 'Chapman', 'CPSC', 'PTB', 'SPH']:
        adult_labels.update(dfs[name]['df'].index.tolist())

    shared = sorted([lbl for lbl in p_df.index if lbl in adult_labels])
    print(f"[{title}] Found {len(shared)} shared labels for comparison: {shared}")

    # Plotting
    x = np.arange(len(shared))
    width = 0.12
    fig, ax = plt.subplots(figsize=(18, 8))

    colors = ['#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#e67e22', '#e74c3c']
    labels = ['G12EC', 'Chapman', 'CPSC', 'PTB', 'SPH', 'Pediatric']

    for i, (name, color) in enumerate(zip(labels, colors)):
        dataset_info = dfs[name]
        df_sub = dataset_info['df']
        auc_col = dataset_info['auc_col']
        
        aucs = []
        err_low = []
        err_high = []
        for lbl in shared:
            if lbl in df_sub.index:
                row = df_sub.loc[lbl]
                aucs.append(row[auc_col])
                # If CI values couldn't be parsed, set errors to 0
                low = row['low'] if not pd.isna(row['low']) else row[auc_col]
                high = row['high'] if not pd.isna(row['high']) else row[auc_col]
                err_low.append(row[auc_col] - low)
                err_high.append(high - row[auc_col])
            else:
                aucs.append(0)
                err_low.append(0)
                err_high.append(0)
        
        offset = (i - 2.5) * width
        ax.bar(x + offset, aucs, width, label=name, yerr=[err_low, err_high], 
               capsize=3, color=color, edgecolor='black', alpha=0.8)

    ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(shared, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(title='Dataset', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved successfully as '{save_path}'")
    plt.close(fig)

if __name__ == "__main__":
    # Zero-shot Paths
    zs_datasets = {
        'G12EC': os.path.join(base_dir, 'cinc_res', 'G12EC_zero_shot_results.csv'),
        'Chapman': os.path.join(base_dir, 'cinc_res', 'Chapman_zero_shot_results.csv'),
        'CPSC': os.path.join(base_dir, 'cinc_res', 'CPSC_zero_shot_results.csv'),
        'PTB': os.path.join(base_dir, 'cinc_res', 'PTB_zero_shot_results.csv'),
        'SPH': os.path.join(base_dir, 'cinc_res', 'SPH_zero_shot_results.csv'),
        'Pediatric': os.path.join(base_dir, 'res', 'ablation_ft', 'zeroshot', 'zeroshot_test_results.csv')
    }

    # FT Paths
    ft_datasets = {
        'G12EC': os.path.join(base_dir, 'cinc_res', 'G12EC_ft_results.csv'),
        'Chapman': os.path.join(base_dir, 'cinc_res', 'Chapman_ft_results.csv'),
        'CPSC': os.path.join(base_dir, 'cinc_res', 'CPSC_ft_results.csv'),
        'PTB': os.path.join(base_dir, 'cinc_res', 'PTB_ft_results.csv'),
        'SPH': os.path.join(base_dir, 'cinc_res', 'SPH_ft_results.csv'),
        'Pediatric': os.path.join(base_dir, 'res', 'ablation_ft', 'ft_full', 'pediatric_full_test_results.csv')
    }

    generate_comparison_plot(
        datasets=zs_datasets, 
        title='Zero-shot Performance Comparison (Shared Labels)', 
        save_path=os.path.join(base_dir, 'res', 'comparison_all_hospitals_zeroshot.png')
    )

    generate_comparison_plot(
        datasets=ft_datasets, 
        title='Fine-Tuning Performance Comparison (Shared Labels)', 
        save_path=os.path.join(base_dir, 'res', 'comparison_all_hospitals_ft.png')
    )