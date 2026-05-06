# =============================================================================
# Differentially-Private Fine-tuning of ECGFounder on Child ECG Dataset
# Same design as ft.py — adds Opacus DP engine, BN→GN conversion,
# gradient-norm logging, and privacy accounting on top.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import logging
import json
from tqdm import tqdm
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, auc
)

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

import wfdb
from net1d import Net1D

# =============================================================================
# CONFIG
# =============================================================================
LINEAR_PROBE  = False
gpu_id        = 0
batch_size    = 64          # larger batch → better DP utility
lr            = 1e-4
lp_lr         = 1e-3
weight_decay  = 1e-5
Epochs        = 30
early_stop_lr = 1e-5
target_fs     = 5000

# DP parameters
PRIVACY_BUDGET    = 8.0     # target epsilon
DELTA             = 1e-5    # target delta (should be < 1/N)
MAX_GRAD_NORM     = 1.0     # per-sample gradient clipping threshold
ACCOUNTANT        = "rdp"   # "rdp" | "gdp" | "prv"
POISSON_SAMPLING  = False   # Poisson sampling (True = stricter DP guarantee)
FREEZE_NORM       = True    # freeze GroupNorm layers (recommended for DP)
NUM_GROUPS        = 32      # groups for BN → GN conversion

# Paths
ecg_path   = "C:/Users/zoorab/Desktop/zoher/University/Projects/Zhengzhou_ECG/Child_ecg/"
csv_path   = "./ecg_with_exact_match.csv"
saved_dir  = "./res/dp_finetune/"
pth        = "./12_lead_ECGFounder.pth"
tasks_path = "./tasks.txt"

# Zero-shot baseline cache — generated automatically if missing
ZEROSHOT_GT_PATH   = "./res/dp_finetune/zeroshot_gt.npy"
ZEROSHOT_PRED_PATH = "./res/dp_finetune/zeroshot_pred.npy"

os.makedirs(saved_dir, exist_ok=True)
os.makedirs("logging", exist_ok=True)

# =============================================================================
# LOGGING
# =============================================================================
log_file = "logging/child_ecg_dp_ft.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

fh = logging.FileHandler(log_file, encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass
logger.addHandler(ch)

# =============================================================================
# DEVICE
# =============================================================================
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# =============================================================================
# LOAD TASKS
# =============================================================================
with open(tasks_path, "r") as f:
    labels = [line.strip() for line in f if line.strip()]
n_classes = len(labels)
logger.info(f"Number of classes: {n_classes}")

# =============================================================================
# DATASET
# =============================================================================
class ECG_Dataset(Dataset):
    def __init__(self, ecg_path, df):
        self.ecg_path  = ecg_path
        self.data      = df.copy().reset_index(drop=True)
        self.target_fs = 5000

    def z_score_normalization(self, signal):
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    def resample_unequal(self, ts, fs_in, fs_out):
        if fs_in == 0 or len(ts) == 0:
            return ts
        t = ts.shape[1] / fs_in
        fs_in, fs_out = int(fs_in), int(fs_out)
        if fs_out == fs_in:
            return ts
        if 2 * fs_out == fs_in:
            return ts[:, ::2]
        resampled_ts = np.zeros((ts.shape[0], fs_out))
        x_old = np.linspace(0, t, num=ts.shape[1], endpoint=True)
        x_new = np.linspace(0, t, num=int(fs_out), endpoint=True)
        for i in range(ts.shape[0]):
            f = interp1d(x_old, ts[i, :], kind="linear")
            resampled_ts[i, :] = f(x_new)
        return resampled_ts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row         = self.data.iloc[idx]
        label       = torch.tensor(row["label"], dtype=torch.float)
        file_path   = self.ecg_path + row["Filename"]
        sample_rate = row["Sampling_point"]
        try:
            data, _ = wfdb.rdsamp(file_path)
            data    = np.transpose(data, (1, 0))
            data    = self.z_score_normalization(data)
            data    = self.resample_unequal(data, sample_rate, self.target_fs)
            signal  = torch.FloatTensor(data)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e} -- returning zeros")
            signal = torch.zeros((12, self.target_fs))
        return signal, label


# =============================================================================
# DP UTILITIES — BatchNorm → GroupNorm conversion + norm-layer freezing
# =============================================================================
def convert_batchnorm_to_groupnorm(module, num_groups=32):
    """
    Recursively replace BatchNorm1d with GroupNorm, preserving weights.
    Opacus does not support BatchNorm — GN is the standard replacement.
    """
    module_output = module
    if isinstance(module, nn.BatchNorm1d):
        num_channels = module.num_features
        groups = min(num_groups, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        module_output = nn.GroupNorm(
            num_groups=groups,
            num_channels=num_channels,
            eps=module.eps,
            affine=module.affine
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight.copy_(module.weight)
                module_output.bias.copy_(module.bias)

    for name, child in module.named_children():
        module_output.add_module(name, convert_batchnorm_to_groupnorm(child, num_groups))

    del module
    return module_output


def freeze_normalization_layers(model):
    """
    Freeze all GroupNorm / LayerNorm layers so they are excluded from DP
    gradient computation. Recommended when FREEZE_NORM=True.
    """
    frozen_count = 0
    for module in model.modules():
        if isinstance(module, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
            frozen_count += 1
    return model, frozen_count


# =============================================================================
# MODEL
# =============================================================================
class ft_ChildECG(nn.Module):
    def __init__(self, device, pth, n_classes, linear_probe=False):
        super(ft_ChildECG, self).__init__()
        self.backbone = Net1D(
            in_channels=12, base_filters=64, ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16, stride=2, groups_width=16,
            verbose=False, use_bn=False, use_do=False, n_classes=n_classes
        )
        checkpoint = torch.load(pth, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"]
        log        = self.backbone.load_state_dict(state_dict, strict=False)
        logger.info(
            f"Pretrained weights loaded | "
            f"missing: {len(log.missing_keys)} | unexpected: {len(log.unexpected_keys)}"
        )
        if linear_probe:
            logger.info("Mode: Linear Probe -- freezing backbone")
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.dense.parameters():
                param.requires_grad = True
        else:
            logger.info("Mode: Full Fine-tuning -- all params trainable")
            for param in self.backbone.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


# =============================================================================
# METRICS
# =============================================================================
def calculate_performance_metrics(true, pred, threshold):
    true        = np.array(true)
    pred        = np.array(pred)
    pred_binary = (pred >= threshold).astype(int)
    tp = np.sum((true == 1) & (pred_binary == 1))
    fp = np.sum((true == 0) & (pred_binary == 1))
    tn = np.sum((true == 0) & (pred_binary == 0))
    fn = np.sum((true == 1) & (pred_binary == 0))
    sens      = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec      = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv       = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1        = 2 * (precision * sens) / (precision + sens) if (precision + sens) > 0 else 0
    auroc     = roc_auc_score(true, pred)           if len(np.unique(true)) > 1 else np.nan
    auprc     = average_precision_score(true, pred) if len(np.unique(true)) > 1 else np.nan
    return sens, spec, precision, f1, precision, npv, auroc, auprc


def bootstrap_ci(metric_func, true, pred, threshold, n_resamples=100):
    true, pred = np.array(true), np.array(pred)
    dist = []
    for _ in range(n_resamples):
        idx = np.random.choice(len(true), len(true), replace=True)
        try:
            val = metric_func(true[idx], pred[idx], threshold)
            if not np.isnan(val):
                dist.append(val)
        except Exception:
            continue
    if len(dist) == 0:
        return (np.nan, np.nan)
    return (round(np.percentile(dist, 2.5), 3), round(np.percentile(dist, 97.5), 3))


def compute_optimal_threshold(true, pred):
    thresholds = np.linspace(0, 1, 100)
    best_f1, best_thresh = 0, 0.5
    for t in thresholds:
        pred_bin = (pred >= t).astype(int)
        tp = np.sum((true == 1) & (pred_bin == 1))
        fp = np.sum((true == 0) & (pred_bin == 1))
        fn = np.sum((true == 1) & (pred_bin == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh


def compute_macro_auroc(all_gt, all_pred, n_classes):
    scores = []
    for i in range(n_classes):
        if len(np.unique(all_gt[:, i])) > 1:
            scores.append(roc_auc_score(all_gt[:, i], all_pred[:, i]))
    return (np.mean(scores) if scores else 0.0), len(scores)


# =============================================================================
# ZERO-SHOT BASELINE  (original weights, no fine-tuning, no DP)
# =============================================================================
def run_zeroshot_baseline(df, saved_dir):
    logger.info("\nGenerating zero-shot baseline predictions...")

    dataset = ECG_Dataset(ecg_path=ecg_path, df=df)
    loader  = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )

    # Load clean pretrained model — no BN→GN needed for inference
    zs_model = ft_ChildECG(
        device=device, pth=pth, n_classes=n_classes, linear_probe=False
    ).to(device)
    zs_model.eval()

    all_gt, all_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Zero-shot inference"):
            x, y = x.to(device), y.to(device)
            all_pred.append(torch.sigmoid(zs_model(x)).cpu().numpy())
            all_gt.append(y.cpu().numpy())

    zs_gt   = np.concatenate(all_gt)
    zs_pred = np.concatenate(all_pred)

    np.save(os.path.join(saved_dir, "zeroshot_gt.npy"),   zs_gt)
    np.save(os.path.join(saved_dir, "zeroshot_pred.npy"), zs_pred)
    logger.info(f"Zero-shot baseline saved — gt: {zs_gt.shape}  pred: {zs_pred.shape}")
    return zs_gt, zs_pred


# =============================================================================
# PLOT 1 — per-class ROC curves (+ macro + micro average)
# =============================================================================
def plot_roc_curves(all_gt, all_pred, labels, n_classes, split_name, saved_dir):
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    valid_fprs, valid_tprs, valid_aucs, valid_labels, valid_indices = [], [], [], [], []
    for i, label in enumerate(labels):
        true = all_gt[:, i]
        pred = all_pred[:, i]
        if len(np.unique(true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(true, pred)
        valid_fprs.append(fpr)
        valid_tprs.append(tpr)
        valid_aucs.append(auc(fpr, tpr))
        valid_labels.append(label)
        valid_indices.append(i)

    n_valid = len(valid_labels)
    logger.info(f"[{split_name}] Plotting {n_valid} valid ROC curves")

    cmap_a = cm.get_cmap("tab20",  20)
    cmap_b = cm.get_cmap("tab20b", 20)
    def get_color(k):
        return cmap_a(k % 20) if k < 20 else cmap_b((k - 20) % 20)

    for k, (fpr, tpr, roc_auc_val, label) in enumerate(
        zip(valid_fprs, valid_tprs, valid_aucs, valid_labels)
    ):
        ax.plot(fpr, tpr, color=get_color(k), lw=0.7, alpha=0.40,
                label=f"{label} (AUC={roc_auc_val:.2f})")

    # macro-average
    mean_fpr    = np.linspace(0, 1, 500)
    mean_tpr    = np.mean(
        [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(valid_fprs, valid_tprs)],
        axis=0
    )
    mean_tpr[0] = 0.0
    macro_auc_val = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, color="#f0c040", lw=2.8, ls="--", zorder=10,
            label=f"Macro-average  (AUC = {macro_auc_val:.3f})")

    # micro-average
    gt_micro   = all_gt[:,   valid_indices].ravel()
    pred_micro = all_pred[:, valid_indices].ravel()
    fpr_m, tpr_m, _ = roc_curve(gt_micro, pred_micro)
    micro_auc_val   = auc(fpr_m, tpr_m)
    ax.plot(fpr_m, tpr_m, color="#40e0d0", lw=2.8, ls="-.", zorder=10,
            label=f"Micro-average  (AUC = {micro_auc_val:.3f})")

    ax.plot([0, 1], [0, 1], color="#666666", lw=1.0, ls=":", zorder=5,
            label="Chance  (AUC = 0.500)")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", color="#cccccc", fontsize=13, labelpad=8)
    ax.set_ylabel("True Positive Rate",  color="#cccccc", fontsize=13, labelpad=8)
    ax.set_title(
        f"ROC Curves per Class — {split_name}\n"
        f"{n_valid} valid / {n_classes} total labels  |  "
        f"Macro AUC={macro_auc_val:.3f}   Micro AUC={micro_auc_val:.3f}",
        color="#ffffff", fontsize=13, pad=14
    )
    ax.tick_params(colors="#888888", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a2a")
    ax.grid(color="#1e1e1e", lw=0.5, linestyle="--")

    handles, leg_labels = ax.get_legend_handles_labels()
    summary_h,  summary_l  = handles[-3:], leg_labels[-3:]
    perclass_h, perclass_l = handles[:-3], leg_labels[:-3]

    legend_summary = ax.legend(
        summary_h, summary_l, loc="lower right", fontsize=11,
        framealpha=0.75, facecolor="#141920", edgecolor="#444444", labelcolor="#eeeeee"
    )
    ax.add_artist(legend_summary)

    ncol = 2 if n_valid > 30 else 1
    ax.legend(
        perclass_h, perclass_l,
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=5.5, framealpha=0.55,
        facecolor="#141920", edgecolor="#333333", labelcolor="#bbbbbb",
        ncol=ncol, handlelength=1.2, handleheight=0.8,
        borderpad=0.5, labelspacing=0.25
    )

    right_margin = 0.62 if ncol == 2 else 0.73
    plt.tight_layout(rect=[0, 0, right_margin, 1])

    out_path = os.path.join(saved_dir, f"{split_name}_roc_curves.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"[{split_name}] ROC curve plot saved: {out_path}")

    return macro_auc_val, micro_auc_val


# =============================================================================
# PLOT 2 — zero-shot vs DP fine-tuned comparison
# =============================================================================
def plot_comparison(ft_gt, ft_pred, zs_gt, zs_pred, labels, n_classes, saved_dir,
                    plot_name="comparison_zeroshot_vs_dp_finetuned"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0d1117")

    valid_indices = [
        i for i in range(n_classes)
        if len(np.unique(ft_gt[:, i])) > 1 and len(np.unique(zs_gt[:, i])) > 1
    ]
    logger.info(f"[comparison] Shared valid labels: {len(valid_indices)}/{n_classes}")

    mean_fpr = np.linspace(0, 1, 500)
    panel_configs = [
        ("Macro-average", "#f0c040", "#e05858"),
        ("Micro-average", "#40e0d0", "#e05858"),
    ]

    for ax, (title_suffix, ft_color, zs_color) in zip(axes, panel_configs):
        ax.set_facecolor("#0d1117")

        if title_suffix == "Macro-average":
            ft_tprs = []
            for i in valid_indices:
                fpr, tpr, _ = roc_curve(ft_gt[:, i], ft_pred[:, i])
                ft_tprs.append(np.interp(mean_fpr, fpr, tpr))
            ft_mean_tpr    = np.mean(ft_tprs, axis=0); ft_mean_tpr[0] = 0.0
            ft_auc_val     = auc(mean_fpr, ft_mean_tpr)

            zs_tprs = []
            for i in valid_indices:
                fpr, tpr, _ = roc_curve(zs_gt[:, i], zs_pred[:, i])
                zs_tprs.append(np.interp(mean_fpr, fpr, tpr))
            zs_mean_tpr    = np.mean(zs_tprs, axis=0); zs_mean_tpr[0] = 0.0
            zs_auc_val     = auc(mean_fpr, zs_mean_tpr)

            ax.plot(mean_fpr, ft_mean_tpr, color=ft_color, lw=2.5,
                    label=f"DP fine-tuned  (AUC={ft_auc_val:.3f})")
            ax.plot(mean_fpr, zs_mean_tpr, color=zs_color, lw=2.5, ls="--",
                    label=f"Zero-shot      (AUC={zs_auc_val:.3f})")

        else:
            fpr_ft, tpr_ft, _ = roc_curve(
                ft_gt[:, valid_indices].ravel(), ft_pred[:, valid_indices].ravel()
            )
            ft_auc_val = auc(fpr_ft, tpr_ft)

            fpr_zs, tpr_zs, _ = roc_curve(
                zs_gt[:, valid_indices].ravel(), zs_pred[:, valid_indices].ravel()
            )
            zs_auc_val = auc(fpr_zs, tpr_zs)

            ax.plot(fpr_ft, tpr_ft, color=ft_color, lw=2.5,
                    label=f"DP fine-tuned  (AUC={ft_auc_val:.3f})")
            ax.plot(fpr_zs, tpr_zs, color=zs_color, lw=2.5, ls="--",
                    label=f"Zero-shot      (AUC={zs_auc_val:.3f})")

        ax.plot([0, 1], [0, 1], color="#555555", lw=1.0, ls=":",
                label="Chance  (AUC=0.500)")

        delta_auc = ft_auc_val - zs_auc_val
        ax.set_title(
            f"{title_suffix} ROC\nΔ AUC = {delta_auc:+.3f}  (DP fine-tuned − zero-shot)",
            color="#ffffff", fontsize=12, pad=10
        )
        ax.set_xlabel("False Positive Rate", color="#cccccc", fontsize=11)
        ax.set_ylabel("True Positive Rate",  color="#cccccc", fontsize=11)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.05])
        ax.tick_params(colors="#888888", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a2a")
        ax.grid(color="#1e1e1e", lw=0.5, linestyle="--")
        ax.legend(
            loc="lower right", fontsize=10,
            framealpha=0.7, facecolor="#141920",
            edgecolor="#444444", labelcolor="#eeeeee"
        )

    scope = "Full dataset (train+val+test)" if "full" in plot_name else "Test set"
    fig.suptitle(
        f"Zero-shot Baseline vs DP Fine-tuned Model — {scope}\n"
        f"ε={PRIVACY_BUDGET}, δ={DELTA}, C={MAX_GRAD_NORM}  |  "
        f"{len(valid_indices)} shared valid labels",
        color="#ffffff", fontsize=13, y=1.01
    )
    plt.tight_layout()

    out_path = os.path.join(saved_dir, f"{plot_name}.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"[comparison] Plot saved: {out_path}")


# =============================================================================
# FULL EVALUATION  (per-class metrics + ROC plot)
# =============================================================================
def full_evaluation(all_gt, all_pred, labels, n_classes, split_name, saved_dir, n_resamples=100):
    logger.info(f"\nComputing full evaluation for: {split_name}")
    results = []
    skipped = []

    for i, label in enumerate(tqdm(labels, desc=f"Metrics [{split_name}]")):
        true  = all_gt[:, i]
        pred  = all_pred[:, i]
        n_pos = int(true.sum())
        n_neg = int((1 - true).sum())

        if len(np.unique(true)) < 2:
            skipped.append({"Label": label, "n_pos": n_pos, "reason": "all zeros — AUROC undefined"})
            continue

        threshold = compute_optimal_threshold(true, pred)
        sens, spec, prec, f1, ppv, npv, auroc, auprc = calculate_performance_metrics(
            true, pred, threshold
        )

        sens_ci  = bootstrap_ci(lambda t, p, th: calculate_performance_metrics(t, p, th)[0], true, pred, threshold, n_resamples)
        spec_ci  = bootstrap_ci(lambda t, p, th: calculate_performance_metrics(t, p, th)[1], true, pred, threshold, n_resamples)
        f1_ci    = bootstrap_ci(lambda t, p, th: calculate_performance_metrics(t, p, th)[3], true, pred, threshold, n_resamples)
        ppv_ci   = bootstrap_ci(lambda t, p, th: calculate_performance_metrics(t, p, th)[4], true, pred, threshold, n_resamples)
        npv_ci   = bootstrap_ci(lambda t, p, th: calculate_performance_metrics(t, p, th)[5], true, pred, threshold, n_resamples)
        auroc_ci = bootstrap_ci(lambda t, p, th: calculate_performance_metrics(t, p, th)[6], true, pred, threshold, n_resamples)
        auprc_ci = bootstrap_ci(lambda t, p, th: calculate_performance_metrics(t, p, th)[7], true, pred, threshold, n_resamples)

        results.append({
            "Label"          : label,
            "n_pos"          : n_pos,
            "n_neg"          : n_neg,
            "Threshold"      : round(threshold, 3),
            "Sensitivity"    : round(sens,  3), "Sensitivity_CI" : sens_ci,
            "Specificity"    : round(spec,  3), "Specificity_CI" : spec_ci,
            "F1"             : round(f1,    3), "F1_CI"          : f1_ci,
            "PPV"            : round(ppv,   3), "PPV_CI"         : ppv_ci,
            "NPV"            : round(npv,   3), "NPV_CI"         : npv_ci,
            "AUROC"          : round(auroc, 3)  if not np.isnan(auroc) else np.nan,
            "AUROC_CI"       : auroc_ci,
            "AUPRC"          : round(auprc, 3)  if not np.isnan(auprc) else np.nan,
            "AUPRC_CI"       : auprc_ci,
        })

    results_df = pd.DataFrame(results).sort_values("AUROC", ascending=False)
    skipped_df = pd.DataFrame(skipped)

    out_path     = os.path.join(saved_dir, f"{split_name}_results.csv")
    skipped_path = os.path.join(saved_dir, f"{split_name}_skipped_labels.csv")
    results_df.to_csv(out_path,     index=False)
    skipped_df.to_csv(skipped_path, index=False)

    macro_auroc = results_df["AUROC"].dropna().mean()

    logger.info(f"\n{'='*70}")
    logger.info(f"[{split_name}] PER-CLASS AUROC  ({len(results_df)} valid / {n_classes} total)")
    logger.info(f"{'='*70}")
    logger.info(f"  {'Label':<52} {'n_pos':>6}  {'AUROC':>6}  95% CI")
    logger.info(f"  {'-'*68}")
    for _, row in results_df.iterrows():
        ci = row["AUROC_CI"]
        logger.info(
            f"  {row['Label']:<52} {row['n_pos']:>6}  {row['AUROC']:>6.3f}"
            f"  ({ci[0]:.3f}, {ci[1]:.3f})"
        )

    if len(skipped_df) > 0:
        logger.info(f"\n{'='*70}")
        logger.info(f"[{split_name}] SKIPPED ({len(skipped_df)}) — all-zero labels, AUROC undefined")
        logger.info(f"{'='*70}")
        for _, row in skipped_df.iterrows():
            logger.info(f"  {row['Label']:<52}  {row['reason']}")

    logger.info(f"\n[{split_name}] Valid labels : {len(results_df)} / {n_classes}")
    logger.info(f"[{split_name}] Skipped      : {len(skipped_df)} / {n_classes}")
    logger.info(f"[{split_name}] Macro AUROC  : {macro_auroc:.4f}  ({len(results_df)} valid labels)")
    logger.info(f"[{split_name}] Results saved: {out_path}")

    plot_macro, plot_micro = plot_roc_curves(
        all_gt, all_pred, labels, n_classes, split_name, saved_dir
    )
    logger.info(f"[{split_name}] Plot Macro AUC={plot_macro:.4f}  Micro AUC={plot_micro:.4f}")

    return results_df


# =============================================================================
# LABEL STATISTICS LOGGER
# =============================================================================
def log_label_stats(df, split_name, labels, n_classes):
    label_matrix  = np.array(df["label"].tolist())
    counts        = label_matrix.sum(axis=0).astype(int)
    active_labels = [(labels[i], counts[i]) for i in range(n_classes) if counts[i] > 0]
    active_labels.sort(key=lambda x: -x[1])

    logger.info(f"\n{'='*60}")
    logger.info(f"Label stats [{split_name}]  --  {len(df)} samples total")
    logger.info(f"  Active labels (>= 1 sample) : {len(active_labels)} / {n_classes}")
    logger.info(f"  Avg labels per sample        : {counts.sum() / max(len(df), 1):.2f}")
    logger.info(f"  Max labels per sample        : {label_matrix.sum(axis=1).max():.0f}")
    logger.info(f"  {'Label':<50} Count   Prevalence")
    logger.info(f"  {'-'*70}")
    for lbl, cnt in active_labels:
        logger.info(f"  {lbl:<50} {cnt:<8} {cnt / len(df) * 100:.1f}%")
    logger.info(f"{'='*60}\n")


# =============================================================================
# DATA SPLITS
# =============================================================================
def prepare_splits(ecg_df):
    unique_patients               = ecg_df["Patient_ID"].unique()
    train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)
    train_patients, val_patients  = train_test_split(train_patients,  test_size=0.1, random_state=42)

    train_df = ecg_df[ecg_df["Patient_ID"].isin(train_patients)].reset_index(drop=True)
    val_df   = ecg_df[ecg_df["Patient_ID"].isin(val_patients)].reset_index(drop=True)
    test_df  = ecg_df[ecg_df["Patient_ID"].isin(test_patients)].reset_index(drop=True)

    train_df = train_df[train_df["label"].apply(lambda x: np.array(x).sum() > 0)].reset_index(drop=True)
    val_df   = val_df[val_df["label"].apply(lambda x: np.array(x).sum() > 0)].reset_index(drop=True)
    test_df  = test_df[test_df["label"].apply(lambda x: np.array(x).sum() > 0)].reset_index(drop=True)

    assert len(set(train_df["Patient_ID"]) & set(val_df["Patient_ID"]))  == 0, "Train/Val leakage!"
    assert len(set(train_df["Patient_ID"]) & set(test_df["Patient_ID"])) == 0, "Train/Test leakage!"

    logger.info(f"Train : {len(train_df)} samples | {len(train_patients)} patients")
    logger.info(f"Val   : {len(val_df)}   samples | {len(val_patients)} patients")
    logger.info(f"Test  : {len(test_df)}  samples | {len(test_patients)} patients")

    return train_df, val_df, test_df


# =============================================================================
# MAIN
# =============================================================================
def main():

    # ── load data ─────────────────────────────────────────────────────────────
    ecg_df = pd.read_csv(csv_path)
    if isinstance(ecg_df["label"].iloc[0], str):
        ecg_df["label"] = ecg_df["label"].apply(json.loads)

    train_df, val_df, test_df = prepare_splits(ecg_df)

    log_label_stats(train_df, "TRAIN", labels, n_classes)
    log_label_stats(val_df,   "VAL",   labels, n_classes)
    log_label_stats(test_df,  "TEST",  labels, n_classes)

    train_dataset = ECG_Dataset(ecg_path=ecg_path, df=train_df)
    val_dataset   = ECG_Dataset(ecg_path=ecg_path, df=val_df)
    test_dataset  = ECG_Dataset(ecg_path=ecg_path, df=test_df)

    # ── note: trainloader will be replaced by Opacus below ────────────────────
    trainloader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    valloader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )
    testloader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )

    logger.info(
        f"Train batches: {len(trainloader)} | "
        f"Val batches:   {len(valloader)}   | "
        f"Test batches:  {len(testloader)}"
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = ft_ChildECG(
        device=device, pth=pth, n_classes=n_classes, linear_probe=LINEAR_PROBE
    ).to(device)

    # ── BN → GN conversion (required by Opacus) ───────────────────────────────
    logger.info(f"Converting BatchNorm → GroupNorm (num_groups={NUM_GROUPS}) for DP compatibility...")
    model = convert_batchnorm_to_groupnorm(model, num_groups=NUM_GROUPS).to(device)

    # validate Opacus compatibility
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        logger.warning(f"Remaining compatibility issues after GN conversion: {errors}")
        logger.info("Applying ModuleValidator.fix() as fallback...")
        model = ModuleValidator.fix(model).to(device)
    else:
        logger.info("Model is fully DP-compatible after BN→GN conversion.")

    # ── freeze norm layers (recommended) ──────────────────────────────────────
    if FREEZE_NORM:
        model, frozen_count = freeze_normalization_layers(model)
        logger.info(f"Frozen {frozen_count} normalization layers (excluded from DP noise)")

    # ── optimizer (must be created AFTER model modifications) ─────────────────
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lp_lr if LINEAR_PROBE else lr,
        weight_decay=weight_decay
    )

    # ── Opacus privacy engine ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DIFFERENTIAL PRIVACY CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"  Target ε (privacy budget) : {PRIVACY_BUDGET}")
    logger.info(f"  Target δ                  : {DELTA}")
    logger.info(f"  Max grad norm (C)         : {MAX_GRAD_NORM}")
    logger.info(f"  Accountant                : {ACCOUNTANT}")
    logger.info(f"  Poisson sampling          : {POISSON_SAMPLING}")
    logger.info(f"  Freeze norm layers        : {FREEZE_NORM}")
    logger.info("=" * 60)

    privacy_engine = PrivacyEngine(accountant=ACCOUNTANT)

    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module        = model,
        optimizer     = optimizer,
        data_loader   = trainloader,
        target_epsilon   = PRIVACY_BUDGET,
        target_delta     = DELTA,
        epochs           = Epochs,
        max_grad_norm    = MAX_GRAD_NORM,
        poisson_sampling = POISSON_SAMPLING,
    )

    logger.info(f"  Noise multiplier (σ)      : {optimizer.noise_multiplier:.6f}")
    logger.info(f"  Expected noise scale      : {optimizer.noise_multiplier * optimizer.max_grad_norm:.6f}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1, mode="max")
    criterion = nn.BCEWithLogitsLoss()

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Mode            : {'Linear Probe' if LINEAR_PROBE else 'Full Fine-tuning (DP)'}")
    logger.info(f"Total params    : {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_auroc = 0.0
    global_step    = 0
    last_ckpt_path = None

    for epoch in range(Epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{Epochs}")

        # keep norm layers in eval mode if frozen
        model.train()
        if FREEZE_NORM:
            for module in model.modules():
                if isinstance(module, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                    module.eval()

        train_loss           = 0.0
        all_per_sample_norms = []

        for batch_idx, (x, y) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1} Train")):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()

            # collect per-sample gradient norms before optimizer step
            try:
                norms_sq = []
                for p in model.parameters():
                    if hasattr(p, "grad_sample") and p.grad_sample is not None and p.requires_grad:
                        bs   = p.grad_sample.size(0)
                        flat = p.grad_sample.reshape(bs, -1)
                        norms_sq.append((flat ** 2).sum(dim=1))
                if norms_sq:
                    total = torch.stack(norms_sq).sum(dim=0).sqrt()
                    all_per_sample_norms.extend(total.cpu().numpy().tolist())
            except (AttributeError, RuntimeError):
                pass

            optimizer.step()
            train_loss  += loss.item()
            global_step += 1

            # log gradient stats every 100 batches
            if batch_idx > 0 and batch_idx % 100 == 0 and all_per_sample_norms:
                recent = all_per_sample_norms[-min(100 * batch_size, len(all_per_sample_norms)):]
                clip_rate = 100 * sum(1 for n in recent if n > optimizer.max_grad_norm) / len(recent)
                logger.info(
                    f"  Batch {batch_idx}/{len(trainloader)} | "
                    f"Grad norm — mean: {np.mean(recent):.3f}  std: {np.std(recent):.3f}  "
                    f"max: {np.max(recent):.3f} | clip rate: {clip_rate:.1f}%"
                )

        avg_train_loss = train_loss / len(trainloader)

        # epoch-level gradient stats
        if all_per_sample_norms:
            clip_rate = 100 * sum(1 for n in all_per_sample_norms if n > optimizer.max_grad_norm) / len(all_per_sample_norms)
            logger.info(f"\n  Epoch gradient stats (pre-clip/noise):")
            logger.info(f"    mean={np.mean(all_per_sample_norms):.4f}  "
                        f"std={np.std(all_per_sample_norms):.4f}  "
                        f"median={np.median(all_per_sample_norms):.4f}  "
                        f"max={np.max(all_per_sample_norms):.4f}")
            logger.info(f"    clip rate: {clip_rate:.2f}%  (C={optimizer.max_grad_norm})")

        epsilon = privacy_engine.get_epsilon(delta=DELTA)
        logger.info(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
            f"Privacy spent: (ε={epsilon:.3f}, δ={DELTA})"
        )

        # check budget — stop if exceeded
        if epsilon >= PRIVACY_BUDGET:
            logger.warning(
                f"Privacy budget exhausted: ε={epsilon:.3f} ≥ {PRIVACY_BUDGET}. Stopping training."
            )
            break

        # validate
        model.eval()
        all_gt, all_pred, val_loss = [], [], 0.0
        with torch.no_grad():
            for x, y in tqdm(valloader, desc=f"Epoch {epoch+1} Val", leave=False):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss  += criterion(logits, y).item()
                all_pred.append(torch.sigmoid(logits).cpu().numpy())
                all_gt.append(y.cpu().numpy())

        all_gt   = np.concatenate(all_gt)
        all_pred = np.concatenate(all_pred)

        val_macro_auroc, valid_labels_count = compute_macro_auroc(all_gt, all_pred, n_classes)
        logger.info(
            f"Val Loss: {val_loss/len(valloader):.4f} | "
            f"Val Macro AUROC: {val_macro_auroc:.4f} | "
            f"Valid labels: {valid_labels_count}/{n_classes}"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_macro_auroc)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != current_lr:
            logger.info(f"LR reduced: {current_lr:.6f} -> {new_lr:.6f}")

        # save best checkpoint — delete previous
        if val_macro_auroc > best_val_auroc:
            best_val_auroc = val_macro_auroc
            mode_suffix    = "linear_probe" if LINEAR_PROBE else "full_ft"
            norm_suffix    = "frozen_norm"  if FREEZE_NORM  else "trainable_norm"
            new_ckpt_path  = os.path.join(
                saved_dir,
                f"child_ecg_dp_{mode_suffix}_{norm_suffix}"
                f"_eps{PRIVACY_BUDGET}_clip{MAX_GRAD_NORM}"
                f"_epoch{epoch+1}_auroc{val_macro_auroc:.4f}.pth"
            )

            if last_ckpt_path is not None and os.path.exists(last_ckpt_path):
                os.remove(last_ckpt_path)
                logger.info(f"[DELETED] Old checkpoint: {os.path.basename(last_ckpt_path)}")

            torch.save({
                "epoch"           : epoch + 1,
                "step"            : global_step,
                "state_dict"      : model.state_dict(),
                "optimizer"       : optimizer.state_dict(),
                "scheduler"       : scheduler.state_dict(),
                "val_auroc"       : val_macro_auroc,
                "epsilon"         : epsilon,
                "delta"           : DELTA,
                "noise_multiplier": optimizer.noise_multiplier,
                "max_grad_norm"   : optimizer.max_grad_norm,
                "freeze_norm"     : FREEZE_NORM,
                "config"          : {
                    "linear_probe"  : LINEAR_PROBE,
                    "n_classes"     : n_classes,
                    "batch_size"    : batch_size,
                    "lr"            : lp_lr if LINEAR_PROBE else lr,
                    "privacy_budget": PRIVACY_BUDGET,
                    "delta"         : DELTA,
                    "max_grad_norm" : MAX_GRAD_NORM,
                    "accountant"    : ACCOUNTANT,
                    "freeze_norm"   : FREEZE_NORM,
                    "num_groups"    : NUM_GROUPS,
                }
            }, new_ckpt_path)
            last_ckpt_path = new_ckpt_path
            logger.info(f"[SAVED] Best model: {os.path.basename(new_ckpt_path)}")

        if optimizer.param_groups[0]["lr"] < early_stop_lr:
            logger.info(f"Early stopping: LR < {early_stop_lr:.2e}")
            break

    # final privacy accounting
    final_epsilon = privacy_engine.get_epsilon(delta=DELTA)
    logger.info(f"\nTraining complete | Best Val Macro AUROC: {best_val_auroc:.4f}")
    logger.info("=" * 60)
    logger.info("FINAL PRIVACY ACCOUNTING")
    logger.info("=" * 60)
    logger.info(f"  Final ε spent : {final_epsilon:.4f}  (budget: {PRIVACY_BUDGET})")
    logger.info(f"  δ             : {DELTA}")
    logger.info(f"  σ             : {optimizer.noise_multiplier:.6f}")
    logger.info(f"  C             : {optimizer.max_grad_norm}")
    logger.info(f"  Norm layers   : {'FROZEN' if FREEZE_NORM else 'TRAINABLE'}")
    logger.info("=" * 60)

    # ── final test evaluation ─────────────────────────────────────────────────
    logger.info("\nRunning final test evaluation...")
    model.eval()
    all_gt, all_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(testloader, desc="Final Test"):
            x, y = x.to(device), y.to(device)
            all_pred.append(torch.sigmoid(model(x)).cpu().numpy())
            all_gt.append(y.cpu().numpy())

    all_gt   = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    np.save(os.path.join(saved_dir, "test_gt.npy"),   all_gt)
    np.save(os.path.join(saved_dir, "test_pred.npy"), all_pred)

    full_evaluation(
        all_gt, all_pred, labels, n_classes,
        split_name="test", saved_dir=saved_dir, n_resamples=100
    )

    # ── test-set comparison: zero-shot vs DP fine-tuned ───────────────────────
    if os.path.exists(ZEROSHOT_GT_PATH) and os.path.exists(ZEROSHOT_PRED_PATH):
        logger.info("\nLoading cached zero-shot baseline predictions...")
        zs_gt   = np.load(ZEROSHOT_GT_PATH)
        zs_pred = np.load(ZEROSHOT_PRED_PATH)
    else:
        logger.info("\nZero-shot cache not found — generating baseline on test split...")
        zs_gt, zs_pred = run_zeroshot_baseline(test_df, saved_dir)

    if zs_gt.shape[0] != all_gt.shape[0]:
        logger.info(
            f"Zero-shot has {zs_gt.shape[0]} samples but test has {all_gt.shape[0]}. "
            "Re-running zero-shot on test split for aligned comparison..."
        )
        zs_gt, zs_pred = run_zeroshot_baseline(test_df, saved_dir)

    plot_comparison(
        ft_gt=all_gt,   ft_pred=all_pred,
        zs_gt=zs_gt,    zs_pred=zs_pred,
        labels=labels,  n_classes=n_classes,
        saved_dir=saved_dir,
        plot_name="comparison_zeroshot_vs_dp_finetuned_test"
    )

    # ── full-dataset comparison (train + val + test) ───────────────────────────
    logger.info("\nRunning full-dataset inference for combined comparison...")
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    full_dataset_ft = ECG_Dataset(ecg_path=ecg_path, df=full_df)
    full_loader_ft  = DataLoader(
        full_dataset_ft, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )
    model.eval()
    ft_full_gt, ft_full_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(full_loader_ft, desc="DP fine-tuned inference (full dataset)"):
            x, y = x.to(device), y.to(device)
            ft_full_pred.append(torch.sigmoid(model(x)).cpu().numpy())
            ft_full_gt.append(y.cpu().numpy())
    ft_full_gt   = np.concatenate(ft_full_gt)
    ft_full_pred = np.concatenate(ft_full_pred)

    logger.info("Running zero-shot inference on full dataset...")
    zs_full_gt, zs_full_pred = run_zeroshot_baseline(full_df, saved_dir)

    np.save(os.path.join(saved_dir, "full_dp_ft_gt.npy"),   ft_full_gt)
    np.save(os.path.join(saved_dir, "full_dp_ft_pred.npy"), ft_full_pred)

    full_evaluation(
        ft_full_gt, ft_full_pred, labels, n_classes,
        split_name="full_dataset", saved_dir=saved_dir, n_resamples=100
    )

    plot_comparison(
        ft_gt=ft_full_gt,   ft_pred=ft_full_pred,
        zs_gt=zs_full_gt,   zs_pred=zs_full_pred,
        labels=labels,      n_classes=n_classes,
        saved_dir=saved_dir,
        plot_name="comparison_zeroshot_vs_dp_finetuned_full"
    )


if __name__ == "__main__":
    main()