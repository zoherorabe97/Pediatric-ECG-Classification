import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.cm as cm

def plot_publication_roc(gt_path, pred_path, tasks_path, out_path):
    """
    Plots a publication-ready ROC curve from saved numpy arrays.
    """
    # 1. Load Data
    all_gt = np.load(gt_path)
    all_pred = np.load(pred_path)
    
    with open(tasks_path, "r") as f:
        labels = [line.strip() for line in f if line.strip()]
        
    if all_gt.shape[1] != len(labels):
        print(f"Warning: GT shape {all_gt.shape[1]} != number of labels {len(labels)}")
    
    # 2. Calculate ROC curves
    valid_fprs, valid_tprs, valid_aucs, valid_labels, valid_idx = [], [], [], [], []
    for i, label in enumerate(labels):
        if i >= all_gt.shape[1]:
            break
        true = all_gt[:, i]
        pred = all_pred[:, i]
        if len(np.unique(true)) < 2:
            continue
        
        fpr, tpr, _ = roc_curve(true, pred)
        roc_auc = auc(fpr, tpr)
        
        valid_fprs.append(fpr)
        valid_tprs.append(tpr)
        valid_aucs.append(roc_auc)
        valid_labels.append(label)
        valid_idx.append(i)

    # 3. Setup Plot Configuration for a Paper
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 8,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
    
    # Generate colors using a combination of cmaps to get enough distinct colors
    cmap_a = cm.get_cmap("tab20", 20)
    cmap_b = cm.get_cmap("tab20b", 20)
    cmap_c = cm.get_cmap("tab20c", 20)
    def get_color(k):
        if k < 20: return cmap_a(k)
        elif k < 40: return cmap_b(k - 20)
        else: return cmap_c(k - 40)
    
    # Plot all individual class lines
    for i in range(len(valid_labels)):
        color = get_color(i)
        ax.plot(valid_fprs[i], valid_tprs[i], color=color, lw=1.2, alpha=0.6,
                label=f"{valid_labels[i]} (AUC = {valid_aucs[i]:.2f})")

    # 4. Calculate Macro and Micro Averages
    # Macro
    all_fpr = np.linspace(0, 1, 500)
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(valid_labels)):
        mean_tpr += np.interp(all_fpr, valid_fprs[i], valid_tprs[i])
    mean_tpr /= len(valid_labels)
    mean_tpr[0] = 0.0
    macro_auc = auc(all_fpr, mean_tpr)
    
    ax.plot(all_fpr, mean_tpr, color="black", lw=3.0, ls="-", zorder=10, 
            label=f"Macro-average (AUC = {macro_auc:.3f})")
    
    # Micro
    gt_micro = all_gt[:, valid_idx].ravel()
    pred_micro = all_pred[:, valid_idx].ravel()
    fpr_micro, tpr_micro, _ = roc_curve(gt_micro, pred_micro)
    micro_auc = auc(fpr_micro, tpr_micro)
    
    ax.plot(fpr_micro, tpr_micro, color="#d62728", lw=3.0, ls="--", zorder=10, 
            label=f"Micro-average (AUC = {micro_auc:.3f})")

    # 5. Chance line
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, ls=":", zorder=5, label="Chance (AUC = 0.500)")

    # 6. Formatting
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Zero-Shot ROC Curves\n({len(valid_labels)} Valid Classes)")
    
    # Clean grid
    ax.grid(color="lightgray", lw=0.5, linestyle="--")
    
    # Spines
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)
        
    # Legend
    # Separate the class legends from the summary legends
    handles, labels = ax.get_legend_handles_labels()
    class_handles, class_labels = handles[:-3], labels[:-3]
    summary_handles, summary_labels = handles[-3:], labels[-3:]
    
    # Create the summary legend inside the plot (lower right) FIRST
    summary_leg = ax.legend(summary_handles, summary_labels, loc="lower right", frameon=True,
                            fontsize=9, edgecolor="gray", facecolor="white", framealpha=0.9)
    ax.add_artist(summary_leg)
    
    # Create the class legend outside the plot SECOND (so it's tracked for tight bbox)
    ncol = 2 if len(class_labels) > 25 else 1
    class_leg = ax.legend(class_handles, class_labels, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                          ncol=ncol, frameon=True, fontsize=7, borderpad=0.6, labelspacing=0.4,
                          edgecolor="gray", title="Classes")
    class_leg.get_title().set_fontsize(8)
    class_leg.get_title().set_fontweight('bold')
    
    # Adjust layout to make room for the right-side legend
    right_margin = 0.55 if ncol == 2 else 0.75
    plt.tight_layout(rect=[0, 0, right_margin, 1])
    
    # Save, explicitly telling tight_layout about BOTH legends just to be perfectly safe
    fig.savefig(out_path, bbox_inches="tight", bbox_extra_artists=(summary_leg, class_leg))
    plt.close(fig)
    print(f"Publication-ready ROC curve saved to: {out_path}")

if __name__ == "__main__":
    base_dir = r"c:\Users\zoorab\Desktop\zoher\University\Projects\ECGFounder"
    gt_path = os.path.join(base_dir, "res", "ablation_ft", "zeroshot_test_gt.npy")
    pred_path = os.path.join(base_dir, "res", "ablation_ft", "zeroshot_test_pred.npy")
    tasks_path = os.path.join(base_dir, "tasks.txt")
    out_path = os.path.join(base_dir, "res", "ablation_ft", "zeroshot", "zeroshot_paper_roc_curve.png")
    
    if os.path.exists(gt_path) and os.path.exists(pred_path) and os.path.exists(tasks_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plot_publication_roc(gt_path, pred_path, tasks_path, out_path)
    else:
        print("Required files not found. Please check paths:")
        print(f"GT: {gt_path}")
        print(f"Pred: {pred_path}")
        print(f"Tasks: {tasks_path}")

