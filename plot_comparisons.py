import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def extract_trainable_params(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path, 'r') as f:
        content = f.read()
    # Search for "Trainable params: 123,456"
    matches = re.findall(r"Trainable params:\s*([0-9,]+)", content, re.IGNORECASE)
    if matches:
        # Take the most recent one (last match) showing up in the log
        val_str = matches[-1].replace(",", "")
        return int(val_str)
    return None

def main():
    base_dir = r"c:\Users\zoorab\Desktop\zoher\University\Projects\ECGFounder"
    res_dir = os.path.join(base_dir, "res")
    log_dir = os.path.join(base_dir, "logging")
    output_dir = os.path.join(res_dir, "FTcomparison")
    
    os.makedirs(output_dir, exist_ok=True)
    
    methods = {
        "Full Finetune": {"res_dir": os.path.join(res_dir, "full_finetune"), "log_file": os.path.join(log_dir, "child_ecg_ft.log")},
        "PEFT AdaLoRA": {"res_dir": os.path.join(res_dir, "peft_adalora"), "log_file": os.path.join(log_dir, "child_ecg_peft_adalora.log")},
        "Linear Probe": {"res_dir": os.path.join(res_dir, "linear_probe"), "log_file": os.path.join(log_dir, "child_ecg_linear_probe.log")},
        "PEFT IA3": {"res_dir": os.path.join(res_dir, "peft_ia3"), "log_file": os.path.join(log_dir, "child_ecg_peft_ia3.log")},
        "PEFT LoRA": {"res_dir": os.path.join(res_dir, "peft_lora"), "log_file": os.path.join(log_dir, "child_ecg_peft_lora.log")}
    }

    results = []

    for method_name, paths in methods.items():
        csv_file = os.path.join(paths["res_dir"], "test_results.csv")
        macro_auc = None
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                if 'AUROC' in df.columns:
                    # Macro AUC is the unweighted mean of class AUROCs
                    macro_auc = df['AUROC'].mean()
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        
        trainable_params = extract_trainable_params(paths["log_file"])
        
        results.append({
            "Method": method_name,
            "Macro AUC": macro_auc,
            "Trainable Params": trainable_params
        })

    df_plot = pd.DataFrame(results)
    print("Extracted Data:")
    print(df_plot)

    sns.set_theme(style="whitegrid")

    # Plot 1: Macro AUC Comparison
    plt.figure(figsize=(10, 6))
    df_auc = df_plot.dropna(subset=['Macro AUC']).sort_values(by="Macro AUC", ascending=False)
    sns.barplot(data=df_auc, x="Method", y="Macro AUC", hue="Method", palette="viridis", legend=False)
    plt.title("Macro AUC Comparison", fontsize=16)
    plt.ylabel("Macro AUC", fontsize=14)
    plt.xlabel("Adaptation Method", fontsize=14)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    for i, val in enumerate(df_auc['Macro AUC']):
        plt.text(i, val + 0.01, f"{val:.4f}", ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    auc_plot_path = os.path.join(output_dir, "macro_auc_comparison.png")
    plt.savefig(auc_plot_path, dpi=300)
    plt.close()
    
    import numpy as np
    
    # Plot 2: Trainable Parameters Comparison
    plt.figure(figsize=(10, 6))
    df_params = df_plot.dropna(subset=['Trainable Params']).sort_values(by="Trainable Params", ascending=False)
    df_params['Log10 Params'] = np.log10(df_params['Trainable Params'])
    
    # Ploting log10 values on linear axes to ensure visible bars for all magnitudes
    ax = sns.barplot(data=df_params, x="Method", y="Log10 Params", hue="Method", palette="magma", legend=False)
    
    # Adjust axes limits to give room for text annotations
    plt.ylim(0, df_params['Log10 Params'].max() * 1.2)
    plt.title("Trainable Parameters Comparison", fontsize=16)
    plt.ylabel("Log10(Number of Trainable Parameters)", fontsize=14)
    plt.xlabel("Adaptation Method", fontsize=14)
    plt.xticks(rotation=45)
    
    # Annotate with the original parameter counts
    for i, (log_val, orig_val) in enumerate(zip(df_params['Log10 Params'], df_params['Trainable Params'])):
        plt.text(i, log_val + 0.1, f"{int(orig_val):,}", ha='center', va='bottom', fontsize=11)
        
    plt.tight_layout()
    params_plot_path = os.path.join(output_dir, "trainable_params_comparison.png")
    plt.savefig(params_plot_path, dpi=300)
    plt.close()

    print(f"Plots saved successfully to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()
