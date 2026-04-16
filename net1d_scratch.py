# =============================================================================
# Net1D Ablation Study — Model Size × Training Sample Size
# Architecture: Net1D in three widths: Large / Medium / Small
# 
# Experiments (per model size):
#   • Full  : all available training samples (~6690)
#   • 3 000 : random subset of 3 000 training samples
#   • 1 000 : random subset of 1 000 training samples
#   •   500 : random subset of   500 training samples
#
# Patient-level data leakage is verified HARD at split time.
# Every experiment saves its best checkpoint and a summary CSV.
# A final comparison table is written to  saved_dir/ablation_summary.csv
# =============================================================================

import os
import sys
import copy
import numpy as np
import pandas as pd
import logging
import json
import random
from tqdm import tqdm
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, auc
)

import wfdb
from net1d import Net1D

# =============================================================================
# CONFIG
# =============================================================================
SEED          = 42
gpu_id        = 0
batch_size    = 32
lr            = 1e-3
weight_decay  = 1e-5
Epochs        = 50
early_stop_lr = 1e-6
target_fs     = 5000

EARLY_STOP_PATIENCE  = 10
EARLY_STOP_MIN_DELTA = 1e-4

ecg_path   = "C:/Users/zoorab/Desktop/zoher/University/Projects/Zhengzhou_ECG/Child_ecg/"
csv_path   = "./ecg_with_exact_match.csv"
saved_dir  = "./res/ablation_net1d/"
tasks_path = "./tasks.txt"

# ── Ablation axes ─────────────────────────────────────────────────────────────
MODEL_SIZES    = ["large", "medium", "small"]   # network widths
SAMPLE_BUDGETS = [None, 3000, 1000, 500]        # None = use all training data

os.makedirs(saved_dir, exist_ok=True)
os.makedirs("logging", exist_ok=True)

# =============================================================================
# SEEDS
# =============================================================================
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seeds(SEED)

# =============================================================================
# LOGGING
# =============================================================================
log_file = "logging/child_ecg_ablation_net1d.log"

logger = logging.getLogger("ablation_net1d")
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
    labels_list = [line.strip() for line in f if line.strip()]
n_classes = len(labels_list)
logger.info(f"Number of classes: {n_classes}")

# =============================================================================
# EARLY STOPPING
# =============================================================================
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0
        self.triggered  = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        improvement = score - self.best_score
        if improvement >= self.min_delta:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            logger.info(
                f"EarlyStopping: no improvement for {self.counter}/{self.patience} epoch(s) "
                f"(best={self.best_score:.4f}, current={score:.4f})"
            )
            if self.counter >= self.patience:
                logger.info(f"EarlyStopping triggered. Best val AUROC: {self.best_score:.4f}")
                self.triggered = True
                return True
        return False

    def reset(self):
        self.best_score = None
        self.counter    = 0
        self.triggered  = False

# =============================================================================
# DATASET
# =============================================================================
class ECG_Dataset(Dataset):
    def __init__(self, ecg_path: str, df: pd.DataFrame):
        self.ecg_path  = ecg_path
        self.data      = df.copy().reset_index(drop=True)
        self.target_fs = target_fs

    def z_score_normalization(self, signal: np.ndarray) -> np.ndarray:
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    def resample_unequal(self, ts: np.ndarray, fs_in: float, fs_out: int) -> np.ndarray:
        if fs_in == 0 or len(ts) == 0:
            return ts
        t      = ts.shape[1] / fs_in
        fs_in  = int(fs_in)
        fs_out = int(fs_out)
        if fs_out == fs_in:
            return ts
        if 2 * fs_out == fs_in:
            return ts[:, ::2]
        resampled = np.zeros((ts.shape[0], fs_out), dtype=np.float32)
        x_old = np.linspace(0, t, num=ts.shape[1], endpoint=True)
        x_new = np.linspace(0, t, num=fs_out,       endpoint=True)
        for i in range(ts.shape[0]):
            f = interp1d(x_old, ts[i, :], kind="linear")
            resampled[i, :] = f(x_new)
        return resampled

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
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
            logger.warning(f"Failed to load {file_path}: {e} — returning zeros")
            signal = torch.zeros((12, self.target_fs))
        return signal, label

# =============================================================================
# MODEL ARCHITECTURE VARIANTS
# =============================================================================
# Three Net1D variants with different base_filters (width multiplier)
# and M_blocks_list (depth/complexity scaling)
#
# Large  : base_filters=64,  m_blocks=standard    (most params)
# Medium : base_filters=48,  m_blocks=slightly reduced
# Small  : base_filters=32,  m_blocks=significantly reduced (fewest params)
# =============================================================================

MODEL_CONFIGS = {
    "large": {
        "base_filters"  : 64,
        "filter_list"   : [64, 160, 160, 400, 400, 1024, 1024],
        "m_blocks_list" : [2, 2, 2, 3, 3, 4, 4],
        "kernel_size"   : 16,
        "stride"        : 2,
        "groups_width"  : 16,
    },
    "medium": {
        "base_filters"  : 48,
        "filter_list"   : [48, 120, 120, 300, 300, 768, 768],
        "m_blocks_list" : [2, 2, 2, 3, 3, 4, 4],
        "kernel_size"   : 16,
        "stride"        : 2,
        "groups_width"  : 12,  # reduced to be divisible by 48, 120, 300, 768
    },
    "small": {
        "base_filters"  : 32,
        "filter_list"   : [32, 80, 80, 200, 200, 512, 512],
        "m_blocks_list" : [2, 2, 2, 2, 2, 3, 3],
        "kernel_size"   : 16,
        "stride"        : 2,
        "groups_width"  : 8,  # reduced to be divisible by 32, 80, 200, 512
    },
}


class Net1D_Variant(nn.Module):
    """
    Net1D with parameterised size (large/medium/small).
    
    Args:
        n_classes : number of output classes
        size      : "large" | "medium" | "small"
    """
    def __init__(self, n_classes: int, size: str = "large"):
        super().__init__()
        cfg = MODEL_CONFIGS[size]
        self.backbone = Net1D(
            in_channels   = 12,
            base_filters  = cfg["base_filters"],
            ratio         = 1,
            filter_list   = cfg["filter_list"],
            m_blocks_list = cfg["m_blocks_list"],
            kernel_size   = cfg["kernel_size"],
            stride        = cfg["stride"],
            groups_width  = cfg["groups_width"],
            verbose       = False,
            use_bn        = False,
            use_do        = False,
            n_classes     = n_classes
        )

    def forward(self, x):
        return self.backbone(x)


def count_params(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

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
    f1        = 2*(precision*sens)/(precision+sens) if (precision+sens) > 0 else 0
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
    if not dist:
        return (np.nan, np.nan)
    return (round(np.percentile(dist, 2.5), 3), round(np.percentile(dist, 97.5), 3))


def compute_optimal_threshold(true, pred) -> float:
    thresholds = np.linspace(0, 1, 100)
    best_f1, best_thresh = 0.0, 0.5
    for t in thresholds:
        pred_bin = (pred >= t).astype(int)
        tp = np.sum((true == 1) & (pred_bin == 1))
        fp = np.sum((true == 0) & (pred_bin == 1))
        fn = np.sum((true == 1) & (pred_bin == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh


def compute_macro_auroc(all_gt, all_pred, n_cls):
    scores = []
    for i in range(n_cls):
        if len(np.unique(all_gt[:, i])) > 1:
            scores.append(roc_auc_score(all_gt[:, i], all_pred[:, i]))
    return (np.mean(scores) if scores else 0.0), len(scores)

# =============================================================================
# ROC PLOT
# =============================================================================
def plot_roc_curves(all_gt, all_pred, labels, n_cls,
                    split_name: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_facecolor("#0d1117");  fig.patch.set_facecolor("#0d1117")

    valid_fprs, valid_tprs, valid_aucs, valid_labels, valid_idx = [], [], [], [], []
    for i, label in enumerate(labels):
        t, p = all_gt[:, i], all_pred[:, i]
        if len(np.unique(t)) < 2:
            continue
        fpr, tpr, _ = roc_curve(t, p)
        valid_fprs.append(fpr);  valid_tprs.append(tpr)
        valid_aucs.append(auc(fpr, tpr))
        valid_labels.append(label);  valid_idx.append(i)

    cmap_a = cm.get_cmap("tab20", 20)
    cmap_b = cm.get_cmap("tab20b", 20)
    get_c  = lambda k: cmap_a(k % 20) if k < 20 else cmap_b((k - 20) % 20)

    for k, (fpr, tpr, a_val, lbl) in enumerate(
            zip(valid_fprs, valid_tprs, valid_aucs, valid_labels)):
        ax.plot(fpr, tpr, color=get_c(k), lw=0.7, alpha=0.4,
                label=f"{lbl} (AUC={a_val:.2f})")

    mean_fpr = np.linspace(0, 1, 500)
    interp_t = [np.interp(mean_fpr, fp, tp)
                for fp, tp in zip(valid_fprs, valid_tprs)]
    mean_tpr = np.mean(interp_t, axis=0);  mean_tpr[0] = 0.0
    std_tpr  = np.std(interp_t, axis=0)
    macro_auc_val = auc(mean_fpr, mean_tpr)

    macro_line, = ax.plot(mean_fpr, mean_tpr, color="#f0c040", lw=2.8, ls="--",
                          zorder=10, label=f"Macro (AUC={macro_auc_val:.3f})")
    ax.fill_between(mean_fpr,
                    np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    color="#f0c040", alpha=0.10, zorder=9)

    gt_m = all_gt[:, valid_idx].ravel();  pd_m = all_pred[:, valid_idx].ravel()
    fpr_m, tpr_m, _ = roc_curve(gt_m, pd_m)
    micro_auc_val   = auc(fpr_m, tpr_m)
    micro_line, = ax.plot(fpr_m, tpr_m, color="#40e0d0", lw=2.8, ls="-.",
                          zorder=10, label=f"Micro (AUC={micro_auc_val:.3f})")
    chance, = ax.plot([0, 1], [0, 1], color="#666666", lw=1.0, ls=":",
                      zorder=5, label="Chance (AUC=0.500)")

    ax.set_xlim([-0.01, 1.01]);  ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", color="#cccccc", fontsize=13)
    ax.set_ylabel("True Positive Rate",  color="#cccccc", fontsize=13)
    ax.set_title(f"ROC — {split_name}  |  Macro AUC={macro_auc_val:.3f}  Micro AUC={micro_auc_val:.3f}",
                 color="#ffffff", fontsize=12)
    ax.tick_params(colors="#888888", labelsize=9)
    for s in ax.spines.values(): s.set_edgecolor("#2a2a2a")
    ax.grid(color="#1e1e1e", lw=0.5, ls="--")

    summary_h = [macro_line, Patch(facecolor="#f0c040", alpha=0.2, label="±1 std"),
                 micro_line, chance]
    summary_l = [f"Macro (AUC={macro_auc_val:.3f})", "Macro ±1 std",
                 f"Micro (AUC={micro_auc_val:.3f})", "Chance (AUC=0.500)"]
    legend_s  = ax.legend(summary_h, summary_l, loc="lower right", fontsize=9,
                           framealpha=0.75, facecolor="#141920",
                           edgecolor="#444444", labelcolor="#eeeeee")
    ax.add_artist(legend_s)

    handles, leg_labels = ax.get_legend_handles_labels()
    ncol = 2 if len(valid_labels) > 30 else 1
    ax.legend(handles[:-3], leg_labels[:-3],
              loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=5.5, framealpha=0.55, facecolor="#141920",
              edgecolor="#333333", labelcolor="#bbbbbb",
              ncol=ncol, handlelength=1.2, borderpad=0.5, labelspacing=0.25)

    right_margin = 0.62 if ncol == 2 else 0.73
    plt.tight_layout(rect=[0, 0, right_margin, 1])
    out_path = os.path.join(out_dir, f"{split_name}_roc_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"[{split_name}] ROC plot saved: {out_path}")
    return macro_auc_val, micro_auc_val

# =============================================================================
# FULL EVALUATION
# =============================================================================
def full_evaluation(all_gt, all_pred, labels, n_cls,
                    split_name, out_dir, n_resamples=100):
    results, skipped = [], []
    for i, label in enumerate(tqdm(labels, desc=f"Metrics [{split_name}]")):
        true, pred = all_gt[:, i], all_pred[:, i]
        n_pos = int(true.sum())
        if len(np.unique(true)) < 2:
            skipped.append({"Label": label, "n_pos": n_pos,
                            "reason": "all zeros — AUROC undefined"})
            continue
        thresh = compute_optimal_threshold(true, pred)
        sens, spec, prec, f1, ppv, npv, auroc, auprc = \
            calculate_performance_metrics(true, pred, thresh)

        auroc_ci = bootstrap_ci(
            lambda t, p, th: calculate_performance_metrics(t, p, th)[6],
            true, pred, thresh, n_resamples)
        auprc_ci = bootstrap_ci(
            lambda t, p, th: calculate_performance_metrics(t, p, th)[7],
            true, pred, thresh, n_resamples)

        results.append({
            "Label"    : label,
            "n_pos"    : n_pos,
            "n_neg"    : int((1 - true).sum()),
            "Threshold": round(thresh, 3),
            "Sens"     : round(sens,  3),
            "Spec"     : round(spec,  3),
            "F1"       : round(f1,    3),
            "PPV"      : round(ppv,   3),
            "NPV"      : round(npv,   3),
            "AUROC"    : round(auroc, 3) if not np.isnan(auroc) else np.nan,
            "AUROC_CI" : auroc_ci,
            "AUPRC"    : round(auprc, 3) if not np.isnan(auprc) else np.nan,
            "AUPRC_CI" : auprc_ci,
        })

    results_df = pd.DataFrame(results).sort_values("AUROC", ascending=False)
    results_df.to_csv(os.path.join(out_dir, f"{split_name}_results.csv"), index=False)

    macro_auroc = results_df["AUROC"].dropna().mean()
    logger.info(f"[{split_name}] Valid labels: {len(results_df)} / {n_cls} | "
                f"Macro AUROC: {macro_auroc:.4f}")

    plot_roc_curves(all_gt, all_pred, labels, n_cls, split_name, out_dir)
    return macro_auroc, results_df

# =============================================================================
# PATIENT-LEVEL SPLIT WITH HARD LEAKAGE CHECK
# =============================================================================
def prepare_splits(ecg_df: pd.DataFrame):
    """
    Split at PATIENT level so no patient appears in more than one partition.
    Raises AssertionError on any detected leakage.
    """
    unique_patients               = ecg_df["Patient_ID"].unique()
    train_pts, test_pts           = train_test_split(
        unique_patients, test_size=0.2, random_state=SEED)
    train_pts, val_pts            = train_test_split(
        train_pts,       test_size=0.1, random_state=SEED)

    train_df = ecg_df[ecg_df["Patient_ID"].isin(train_pts)].reset_index(drop=True)
    val_df   = ecg_df[ecg_df["Patient_ID"].isin(val_pts)].reset_index(drop=True)
    test_df  = ecg_df[ecg_df["Patient_ID"].isin(test_pts)].reset_index(drop=True)

    # Remove samples with no positive label
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        mask = df["label"].apply(lambda x: np.array(x).sum() > 0)
        if name == "train":
            train_df = df[mask].reset_index(drop=True)
        elif name == "val":
            val_df   = df[mask].reset_index(drop=True)
        else:
            test_df  = df[mask].reset_index(drop=True)

    # ── HARD leakage assertions ───────────────────────────────────────────────
    train_ids = set(train_df["Patient_ID"])
    val_ids   = set(val_df["Patient_ID"])
    test_ids  = set(test_df["Patient_ID"])

    train_val_overlap  = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap   = val_ids   & test_ids

    if train_val_overlap:
        raise AssertionError(
            f"DATA LEAKAGE: {len(train_val_overlap)} patient(s) appear in both "
            f"TRAIN and VAL: {list(train_val_overlap)[:5]} ...")

    if train_test_overlap:
        raise AssertionError(
            f"DATA LEAKAGE: {len(train_test_overlap)} patient(s) appear in both "
            f"TRAIN and TEST: {list(train_test_overlap)[:5]} ...")

    if val_test_overlap:
        raise AssertionError(
            f"DATA LEAKAGE: {len(val_test_overlap)} patient(s) appear in both "
            f"VAL and TEST: {list(val_test_overlap)[:5]} ...")

    # ── Double-check: no sample-level duplicates across splits ────────────────
    all_filenames = (list(train_df["Filename"]) + list(val_df["Filename"]) +
                     list(test_df["Filename"]))
    if len(all_filenames) != len(set(all_filenames)):
        duplicates = [f for f in set(all_filenames)
                      if all_filenames.count(f) > 1]
        raise AssertionError(
            f"DATA LEAKAGE: {len(duplicates)} file(s) appear in multiple splits: "
            f"{duplicates[:5]} ...")

    logger.info("✓ Patient-level split verified — no leakage detected.")
    logger.info(f"  TRAIN : {len(train_df):5d} samples | {len(train_ids):5d} patients")
    logger.info(f"  VAL   : {len(val_df):5d} samples | {len(val_ids):5d} patients")
    logger.info(f"  TEST  : {len(test_df):5d} samples | {len(test_ids):5d} patients")

    return train_df, val_df, test_df


def subsample_train(train_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Randomly subsample `n` training records, keeping the same random seed
    so results are reproducible across model-size runs with the same budget.
    """
    if n >= len(train_df):
        logger.info(f"  Subsample budget {n} >= available {len(train_df)} — using all.")
        return train_df
    rng   = np.random.default_rng(seed)
    idx   = rng.choice(len(train_df), size=n, replace=False)
    sub   = train_df.iloc[sorted(idx)].reset_index(drop=True)
    logger.info(f"  Subsampled training set: {len(sub)} / {len(train_df)} records")
    return sub

# =============================================================================
# SINGLE EXPERIMENT  (one model size × one sample budget)
# =============================================================================
def run_experiment(
    size: str,
    sample_budget,        # int or None
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    experiment_id: str,   # e.g. "large_3000"
) -> dict:
    """
    Train one Net1D variant, evaluate on the test split, return summary dict.
    """
    exp_dir = os.path.join(saved_dir, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)

    banner = f"{'='*70}\nEXPERIMENT: {experiment_id}\n{'='*70}"
    logger.info(f"\n{banner}")

    # ── training data for this budget ─────────────────────────────────────────
    budget_label = str(sample_budget) if sample_budget is not None else "full"
    if sample_budget is not None:
        cur_train_df = subsample_train(train_df, sample_budget, seed=SEED)
    else:
        cur_train_df = train_df
        logger.info(f"  Using full training set: {len(cur_train_df)} records")

    train_dataset = ECG_Dataset(ecg_path, cur_train_df)
    val_dataset   = ECG_Dataset(ecg_path, val_df)
    test_dataset  = ECG_Dataset(ecg_path, test_df)

    trainloader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True,  num_workers=0, pin_memory=True)
    valloader   = DataLoader(val_dataset,   batch_size=batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)
    testloader  = DataLoader(test_dataset,  batch_size=batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    # ── model ─────────────────────────────────────────────────────────────────
    set_seeds(SEED)
    model = Net1D_Variant(n_classes=n_classes, size=size).to(device)
    total_p, train_p = count_params(model)
    cfg = MODEL_CONFIGS[size]
    logger.info(f"  Model size      : {size}")
    logger.info(f"  Base filters    : {cfg['base_filters']}")
    logger.info(f"  Filter list     : {cfg['filter_list']}")
    logger.info(f"  M blocks        : {cfg['m_blocks_list']}")
    logger.info(f"  Total params    : {total_p:,}")
    logger.info(f"  Trainable params: {train_p:,}")
    logger.info(f"  Train samples   : {len(cur_train_df)}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()
    es        = EarlyStopping(patience=EARLY_STOP_PATIENCE,
                               min_delta=EARLY_STOP_MIN_DELTA)

    best_val_auroc = 0.0
    last_ckpt_path = None

    for epoch in range(Epochs):
        # train
        model.train()
        train_loss = 0.0
        for x, y in tqdm(trainloader,
                          desc=f"[{experiment_id}] E{epoch+1} Train",
                          leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(trainloader)

        # validate
        model.eval()
        all_gt_v, all_pred_v, val_loss = [], [], 0.0
        with torch.no_grad():
            for x, y in valloader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item()
                all_pred_v.append(torch.sigmoid(logits).cpu().numpy())
                all_gt_v.append(y.cpu().numpy())

        all_gt_v   = np.concatenate(all_gt_v)
        all_pred_v = np.concatenate(all_pred_v)
        val_macro, valid_cnt = compute_macro_auroc(all_gt_v, all_pred_v, n_classes)

        logger.info(
            f"[{experiment_id}] E{epoch+1:03d}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={val_loss/len(valloader):.4f}  "
            f"val_AUROC={val_macro:.4f}  "
            f"valid_labels={valid_cnt}/{n_classes}"
        )

        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_macro)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != cur_lr:
            logger.info(f"  LR: {cur_lr:.2e} → {new_lr:.2e}")

        # checkpoint
        if val_macro > best_val_auroc:
            best_val_auroc = val_macro
            new_ckpt = os.path.join(
                exp_dir,
                f"{experiment_id}_epoch{epoch+1}_auroc{val_macro:.4f}.pth"
            )
            if last_ckpt_path and os.path.exists(last_ckpt_path):
                os.remove(last_ckpt_path)
            torch.save({
                "epoch"       : epoch + 1,
                "state_dict"  : model.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "scheduler"   : scheduler.state_dict(),
                "val_auroc"   : val_macro,
                "experiment"  : experiment_id,
                "model_size"  : size,
                "n_train"     : len(cur_train_df),
                "config": {
                    "n_classes" : n_classes,
                    "batch_size": batch_size,
                    "lr"        : lr,
                    "seed"      : SEED,
                },
            }, new_ckpt)
            last_ckpt_path = new_ckpt
            logger.info(f"  [SAVED] {os.path.basename(new_ckpt)}")

        # early stop
        if optimizer.param_groups[0]["lr"] < early_stop_lr:
            logger.info(f"  LR floor reached — stopping.")
            break
        if es(val_macro):
            logger.info(f"  Early stopping triggered at epoch {epoch+1}.")
            break

    logger.info(f"[{experiment_id}] Training done | best val AUROC: {best_val_auroc:.4f}")

    # reload best
    if last_ckpt_path and os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        logger.info(f"[{experiment_id}] Reloaded best checkpoint (epoch {ckpt['epoch']})")
    else:
        logger.warning(f"[{experiment_id}] No checkpoint — using last-epoch weights.")

    # test evaluation
    model.eval()
    test_gt, test_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(testloader, desc=f"[{experiment_id}] Test", leave=False):
            x, y = x.to(device), y.to(device)
            test_pred.append(torch.sigmoid(model(x)).cpu().numpy())
            test_gt.append(y.cpu().numpy())

    test_gt   = np.concatenate(test_gt)
    test_pred = np.concatenate(test_pred)

    np.save(os.path.join(exp_dir, "test_gt.npy"),   test_gt)
    np.save(os.path.join(exp_dir, "test_pred.npy"), test_pred)

    test_macro, _ = full_evaluation(
        test_gt, test_pred, labels_list, n_classes,
        split_name=f"{experiment_id}_test",
        out_dir=exp_dir,
        n_resamples=100
    )

    summary = {
        "experiment"         : experiment_id,
        "model_size"         : size,
        "n_train_requested"  : budget_label,
        "n_train_actual"     : len(cur_train_df),
        "n_val"              : len(val_df),
        "n_test"             : len(test_df),
        "total_params"       : total_p,
        "trainable_params"   : train_p,
        "best_val_AUROC"     : round(best_val_auroc, 4),
        "test_macro_AUROC"   : round(test_macro, 4),
    }
    logger.info(f"[{experiment_id}] SUMMARY: {summary}")
    return summary

# =============================================================================
# COMPARISON PLOT
# =============================================================================
def plot_comparison(summary_df: pd.DataFrame, out_dir: str):
    """
    Two subplots:
      Left  : test AUROC vs n_train_actual, one line per model size
      Right  : test AUROC grouped by model size × training budget (bar chart)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#111720")

    colors = {"large": "#f0c040", "medium": "#40e0d0", "small": "#e06060"}

    # ── left: line plot ────────────────────────────────────────────────────────
    ax = axes[0]
    for size in MODEL_SIZES:
        sub = summary_df[summary_df["model_size"] == size].sort_values("n_train_actual")
        ax.plot(sub["n_train_actual"], sub["test_macro_AUROC"],
                marker="o", color=colors[size], label=size.capitalize(), lw=2)
        for _, row in sub.iterrows():
            ax.annotate(f"{row['test_macro_AUROC']:.3f}",
                        (row["n_train_actual"], row["test_macro_AUROC"]),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=colors[size])

    ax.set_xlabel("Training Samples", color="#cccccc", fontsize=12)
    ax.set_ylabel("Test Macro AUROC", color="#cccccc", fontsize=12)
    ax.set_title("AUROC vs Training Size", color="#ffffff", fontsize=13)
    ax.tick_params(colors="#888888")
    for s in ax.spines.values(): s.set_edgecolor("#2a2a2a")
    ax.grid(color="#1e1e1e", lw=0.5, ls="--")
    ax.legend(fontsize=10, facecolor="#141920", edgecolor="#444", labelcolor="#eee")

    # ── right: grouped bar ─────────────────────────────────────────────────────
    ax  = axes[1]
    budgets = summary_df["n_train_requested"].unique().tolist()
    # sort: full first, then descending numeric
    def _sort_key(b):
        return (0, 0) if b == "full" else (1, -int(b))
    budgets = sorted(budgets, key=_sort_key)

    x     = np.arange(len(budgets))
    width = 0.25
    offsets = {"large": -width, "medium": 0, "small": width}

    for size in MODEL_SIZES:
        vals = []
        for b in budgets:
            row = summary_df[(summary_df["model_size"] == size) &
                             (summary_df["n_train_requested"] == b)]
            vals.append(row["test_macro_AUROC"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + offsets[size], vals, width=width - 0.02,
                      color=colors[size], label=size.capitalize(), alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=colors[size])

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={b}" for b in budgets], color="#cccccc", fontsize=10)
    ax.set_ylabel("Test Macro AUROC", color="#cccccc", fontsize=12)
    ax.set_title("AUROC by Model Size & Budget", color="#ffffff", fontsize=13)
    ax.tick_params(colors="#888888")
    for s in ax.spines.values(): s.set_edgecolor("#2a2a2a")
    ax.grid(axis="y", color="#1e1e1e", lw=0.5, ls="--")
    ax.legend(fontsize=10, facecolor="#141920", edgecolor="#444", labelcolor="#eee")

    plt.tight_layout(pad=2)
    out_path = os.path.join(out_dir, "ablation_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Comparison plot saved: {out_path}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    # ── load & split data ─────────────────────────────────────────────────────
    ecg_df = pd.read_csv(csv_path)
    if isinstance(ecg_df["label"].iloc[0], str):
        ecg_df["label"] = ecg_df["label"].apply(json.loads)

    train_df, val_df, test_df = prepare_splits(ecg_df)

    # ── log model size info ───────────────────────────────────────────────────
    logger.info("\nModel size overview:")
    logger.info(f"  {'Size':<8} {'Base Filters':>12} {'Filter List':<35} {'Total Params':>14} {'Trainable Params':>18}")
    logger.info(f"  {'-'*90}")
    for size in MODEL_SIZES:
        cfg = MODEL_CONFIGS[size]
        dummy = Net1D_Variant(n_classes, size)
        tp, trp = count_params(dummy)
        logger.info(f"  {size:<8} {cfg['base_filters']:>12} {str(cfg['filter_list']):<35} {tp:>14,} {trp:>18,}")
        del dummy

    # ── run all experiments ───────────────────────────────────────────────────
    all_summaries = []

    for size in MODEL_SIZES:
        for budget in SAMPLE_BUDGETS:
            budget_label = str(budget) if budget is not None else "full"
            exp_id       = f"{size}_{budget_label}"
            try:
                summary = run_experiment(
                    size=size,
                    sample_budget=budget,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    experiment_id=exp_id,
                )
                all_summaries.append(summary)
            except Exception as exc:
                logger.error(f"Experiment {exp_id} FAILED: {exc}", exc_info=True)
                all_summaries.append({
                    "experiment"       : exp_id,
                    "model_size"       : size,
                    "n_train_requested": budget_label,
                    "error"            : str(exc),
                })

    # ── save summary table ────────────────────────────────────────────────────
    summary_df   = pd.DataFrame(all_summaries)
    summary_path = os.path.join(saved_dir, "ablation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nAblation summary saved: {summary_path}")

    # ── print comparison table ────────────────────────────────────────────────
    cols = ["experiment", "model_size", "n_train_requested",
            "n_train_actual", "total_params",
            "best_val_AUROC", "test_macro_AUROC"]
    display_df = summary_df[[c for c in cols if c in summary_df.columns]]
    logger.info(f"\n{'='*80}\nABLATION RESULTS\n{'='*80}")
    logger.info(display_df.to_string(index=False))

    # ── comparison plot ───────────────────────────────────────────────────────
    try:
        plot_df = summary_df.dropna(subset=["test_macro_AUROC"])
        if not plot_df.empty:
            plot_comparison(plot_df, saved_dir)
    except Exception as exc:
        logger.warning(f"Comparison plot failed: {exc}")

    logger.info("\nAll experiments complete.")


if __name__ == "__main__":
    main()