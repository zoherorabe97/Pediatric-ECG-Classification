# =============================================================================
# Fine-tuning Ablation Study — ECGFounder × Training Sample Size
#
# Experiments (fixed full fine-tuning, varying sample budget):
#   • Full  : all available training samples (~6 690)
#   • 3 000 : random subset of 3 000 training samples
#   • 1 000 : random subset of 1 000 training samples
#   •   500 : random subset of   500 training samples
#
# Features:
#   • Skip-retrain: if a checkpoint already exists for an experiment, it is
#     loaded directly and only the test evaluation is re-run.
#   • Zero-shot baseline: ECGFounder with original pretrained weights (no FT).
#     Plotted at x=0 on comparison figures.
#   • Two comparison plots:
#       1. ECGFounder FT vs ResNet-scratch  (from resnet_ablation.py summary)
#       2. ECGFounder FT vs Net1D-scratch   (from net1d_ablation.py summary)
#     Both plots include the zero-shot anchor at x=0.
#     ResNet/Net1D scratch lines start at AUROC=0.5 at x=0 (random init).
#
# IMPORTANT — split compatibility
# -------------------------------------------------------
# Both scripts import prepare_splits() and subsample_train() from
# split_utils.py, which uses an identical SEED and split logic.
# This guarantees that:
#   • The test set is the same in both studies  →  AUROCs are comparable.
#   • Each budget (500 / 1000 / 3000) draws the exact same training records
#     in both studies  →  the only variable is the model, not the data.
#
# Usage:
#   python ft_ablation.py
# =============================================================================

import os
import sys
import glob
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
from matplotlib.patches import Patch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc

import wfdb
from net1d import Net1D

# Shared split logic — MUST match resnet_ablation.py and net1d_ablation.py
from split_utils import prepare_splits, subsample_train, SEED

# =============================================================================
# CONFIG
# =============================================================================
gpu_id        = 0
batch_size    = 32
weight_decay  = 1e-5
Epochs        = 50
LR            = 1e-4          # full fine-tuning LR (same as ft.py FULL_FT_LR)
early_stop_lr = 1e-5
target_fs     = 5000

EARLY_STOP_PATIENCE  = 10
EARLY_STOP_MIN_DELTA = 1e-4

ecg_path   = "C:/Users/zoorab/Desktop/zoher/University/Projects/Zhengzhou_ECG/Child_ecg/"
csv_path   = "./ecg_with_exact_match.csv"
pth        = "./12_lead_ECGFounder.pth"
tasks_path = "./tasks.txt"
saved_dir  = "./res/ablation_ft/"

# Paths to scratch-trained baseline summaries (produced by other ablation scripts)
RESNET_SUMMARY_PATH = "./res/ablation_resnet/ablation_summary.csv"
NET1D_SUMMARY_PATH  = "./res/ablation_net1d/ablation_summary.csv"

# Zero-shot cache files (saved once, reused on subsequent runs)
ZEROSHOT_GT_PATH   = os.path.join(saved_dir, "zeroshot_test_gt.npy")
ZEROSHOT_PRED_PATH = os.path.join(saved_dir, "zeroshot_test_pred.npy")

# ── Ablation axis ─────────────────────────────────────────────────────────────
# None  → use all available training samples
SAMPLE_BUDGETS = [None, 3000, 1000, 500, 100]

os.makedirs(saved_dir, exist_ok=True)
os.makedirs("logging", exist_ok=True)

# =============================================================================
# LOGGING
# =============================================================================
log_file = "logging/child_ecg_ablation_ft.log"

logger = logging.getLogger("ft_ablation")
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
                logger.info(
                    f"EarlyStopping triggered. Best val AUROC: {self.best_score:.4f}"
                )
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
        sample_rate =  500
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
# MODEL  (full fine-tuning only — no linear probe)
# =============================================================================
class FT_Model(nn.Module):
    """
    ECGFounder backbone with all weights unfrozen for full fine-tuning.
    A fresh copy is instantiated per experiment so weights never carry over.
    """
    def __init__(self):
        super().__init__()
        self.backbone = Net1D(
            in_channels=12, base_filters=64, ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16, stride=2, groups_width=16,
            verbose=False, use_bn=False, use_do=False, n_classes=n_classes
        )
        checkpoint = torch.load(pth, map_location=device, weights_only=False)
        log = self.backbone.load_state_dict(checkpoint["state_dict"], strict=False)
        logger.info(
            f"  Pretrained weights loaded | "
            f"missing: {len(log.missing_keys)} | "
            f"unexpected: {len(log.unexpected_keys)}"
        )
        # Full fine-tuning: all parameters trainable
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# =============================================================================
# ZERO-SHOT MODEL  (frozen pretrained weights, no training)
# =============================================================================
class ZeroShotModel(nn.Module):
    """
    ECGFounder with original pretrained weights and NO gradient updates.
    Used to establish the zero-shot baseline (x=0 on the ablation plot).
    """
    def __init__(self):
        super().__init__()
        self.backbone = Net1D(
            in_channels=12, base_filters=64, ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16, stride=2, groups_width=16,
            verbose=False, use_bn=False, use_do=False, n_classes=n_classes
        )
        checkpoint = torch.load(pth, map_location=device, weights_only=False)
        log = self.backbone.load_state_dict(checkpoint["state_dict"], strict=False)
        logger.info(
            f"  [Zero-shot] Pretrained weights loaded | "
            f"missing: {len(log.missing_keys)} | "
            f"unexpected: {len(log.unexpected_keys)}"
        )
        # Freeze everything — zero-shot means no fine-tuning
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    sens  = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv   = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1    = 2*(prec*sens)/(prec+sens) if (prec+sens) > 0 else 0
    auroc = roc_auc_score(true, pred)           if len(np.unique(true)) > 1 else np.nan
    auprc = average_precision_score(true, pred) if len(np.unique(true)) > 1 else np.nan
    return sens, spec, prec, f1, prec, npv, auroc, auprc


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
    return (round(np.percentile(dist, 2.5), 3),
            round(np.percentile(dist, 97.5), 3))


def compute_optimal_threshold(true, pred) -> float:
    thresholds = np.linspace(0, 1, 100)
    best_f1, best_thresh = 0.0, 0.5
    for t in thresholds:
        pb = (pred >= t).astype(int)
        tp = np.sum((true == 1) & (pb == 1))
        fp = np.sum((true == 0) & (pb == 1))
        fn = np.sum((true == 1) & (pb == 0))
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2*p*r/(p+r)   if (p+r)     > 0 else 0
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
    mean_tpr    = np.mean(interp_t, axis=0);  mean_tpr[0] = 0.0
    std_tpr     = np.std(interp_t, axis=0)
    macro_auc_v = auc(mean_fpr, mean_tpr)

    macro_line, = ax.plot(mean_fpr, mean_tpr, color="#f0c040", lw=2.8, ls="--",
                          zorder=10, label=f"Macro (AUC={macro_auc_v:.3f})")
    ax.fill_between(mean_fpr,
                    np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    color="#f0c040", alpha=0.10, zorder=9)

    gt_m = all_gt[:, valid_idx].ravel();  pd_m = all_pred[:, valid_idx].ravel()
    fpr_m, tpr_m, _ = roc_curve(gt_m, pd_m)
    micro_auc_v     = auc(fpr_m, tpr_m)
    micro_line, = ax.plot(fpr_m, tpr_m, color="#40e0d0", lw=2.8, ls="-.",
                          zorder=10, label=f"Micro (AUC={micro_auc_v:.3f})")
    chance, = ax.plot([0, 1], [0, 1], color="#666666", lw=1.0, ls=":",
                      zorder=5, label="Chance (AUC=0.500)")

    ax.set_xlim([-0.01, 1.01]);  ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", color="#cccccc", fontsize=13)
    ax.set_ylabel("True Positive Rate",  color="#cccccc", fontsize=13)
    ax.set_title(
        f"ROC — {split_name}  |  "
        f"Macro AUC={macro_auc_v:.3f}  Micro AUC={micro_auc_v:.3f}",
        color="#ffffff", fontsize=12)
    ax.tick_params(colors="#888888", labelsize=9)
    for s in ax.spines.values(): s.set_edgecolor("#2a2a2a")
    ax.grid(color="#1e1e1e", lw=0.5, ls="--")

    summary_h = [macro_line,
                 Patch(facecolor="#f0c040", alpha=0.2, label="±1 std"),
                 micro_line, chance]
    summary_l = [f"Macro (AUC={macro_auc_v:.3f})", "Macro ±1 std",
                 f"Micro (AUC={micro_auc_v:.3f})", "Chance (AUC=0.500)"]
    leg_s = ax.legend(summary_h, summary_l, loc="lower right", fontsize=9,
                      framealpha=0.75, facecolor="#141920",
                      edgecolor="#444444", labelcolor="#eeeeee")
    ax.add_artist(leg_s)

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
    return macro_auc_v, micro_auc_v

# =============================================================================
# FULL EVALUATION
# =============================================================================
def full_evaluation(all_gt, all_pred, labels, n_cls,
                    split_name: str, out_dir: str,
                    n_resamples: int = 100) -> float:
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
    results_df.to_csv(
        os.path.join(out_dir, f"{split_name}_results.csv"), index=False)
    pd.DataFrame(skipped).to_csv(
        os.path.join(out_dir, f"{split_name}_skipped_labels.csv"), index=False)

    macro_auroc = results_df["AUROC"].dropna().mean()
    logger.info(
        f"[{split_name}] Valid: {len(results_df)}/{n_cls} | "
        f"Macro AUROC: {macro_auroc:.4f}"
    )
    plot_roc_curves(all_gt, all_pred, labels, n_cls, split_name, out_dir)
    return macro_auroc

# =============================================================================
# ZERO-SHOT BASELINE
# =============================================================================
def run_zeroshot_baseline(test_df: pd.DataFrame) -> float:
    """
    Run zero-shot inference with frozen pretrained ECGFounder weights on the
    test split. Results are cached to disk; subsequent calls load from cache.

    Returns the macro AUROC of the zero-shot model.
    """
    # ── check cache ───────────────────────────────────────────────────────────
    if os.path.exists(ZEROSHOT_GT_PATH) and os.path.exists(ZEROSHOT_PRED_PATH):
        logger.info("  [Zero-shot] Loading cached predictions …")
        zs_gt   = np.load(ZEROSHOT_GT_PATH)
        zs_pred = np.load(ZEROSHOT_PRED_PATH)
        # Verify the cached arrays match the current test split size
        if zs_gt.shape[0] == len(test_df):
            macro, valid_cnt = compute_macro_auroc(zs_gt, zs_pred, n_classes)
            logger.info(
                f"  [Zero-shot] Cache loaded — macro AUROC: {macro:.4f} "
                f"({valid_cnt}/{n_classes} valid labels)"
            )
            return macro
        else:
            logger.warning(
                f"  [Zero-shot] Cache size mismatch "
                f"({zs_gt.shape[0]} vs {len(test_df)}) — re-running inference."
            )

    # ── run inference ─────────────────────────────────────────────────────────
    logger.info("  [Zero-shot] Running inference with frozen pretrained weights …")
    zs_model = ZeroShotModel().to(device)
    zs_model.eval()

    testloader = DataLoader(
        ECG_Dataset(ecg_path, test_df),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    all_gt, all_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(testloader, desc="[Zero-shot] Inference", leave=False):
            x, y = x.to(device), y.to(device)
            all_pred.append(torch.sigmoid(zs_model(x)).cpu().numpy())
            all_gt.append(y.cpu().numpy())

    zs_gt   = np.concatenate(all_gt)
    zs_pred = np.concatenate(all_pred)

    # Cache to disk
    np.save(ZEROSHOT_GT_PATH,   zs_gt)
    np.save(ZEROSHOT_PRED_PATH, zs_pred)
    logger.info(
        f"  [Zero-shot] Predictions cached — gt: {zs_gt.shape}  pred: {zs_pred.shape}"
    )

    # Full per-class evaluation
    zs_dir = os.path.join(saved_dir, "zeroshot")
    os.makedirs(zs_dir, exist_ok=True)
    macro = full_evaluation(
        zs_gt, zs_pred, labels_list, n_classes,
        split_name="zeroshot_test",
        out_dir=zs_dir,
        n_resamples=100,
    )
    logger.info(f"  [Zero-shot] Macro AUROC: {macro:.4f}")

    del zs_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return macro

# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================
def find_best_checkpoint(exp_dir: str, exp_id: str):
    """
    Search exp_dir for any .pth file matching the experiment naming pattern.
    Returns the path of the checkpoint with the highest AUROC in its filename,
    or None if no checkpoint is found.
    """
    pattern = os.path.join(exp_dir, f"{exp_id}_epoch*.pth")
    matches = glob.glob(pattern)
    if not matches:
        return None
    # Sort by the AUROC value embedded in the filename (highest first)
    def _auroc_from_name(p):
        try:
            return float(p.split("auroc")[-1].replace(".pth", ""))
        except Exception:
            return 0.0
    matches.sort(key=_auroc_from_name, reverse=True)
    return matches[0]


def experiment_already_done(exp_dir: str, exp_id: str) -> bool:
    """
    An experiment is considered 'done' if:
      1. A checkpoint exists in exp_dir, AND
      2. The test predictions numpy arrays exist (test_gt.npy / test_pred.npy).
    If only the checkpoint exists but predictions are missing, the test
    evaluation will be re-run from the saved checkpoint.
    """
    ckpt = find_best_checkpoint(exp_dir, exp_id)
    return ckpt is not None

# =============================================================================
# COMPARISON PLOTS  (FT vs scratch baseline, with zero-shot anchor)
# =============================================================================
def _plot_one_comparison(
    ft_sub:         pd.DataFrame,   # FT results (has n_train_actual + test_macro_AUROC)
    scratch_sub:    pd.DataFrame,   # scratch baseline results (same columns)
    scratch_sizes:  list,           # e.g. ["large", "medium", "small"]
    scratch_colors: dict,           # size → hex colour
    scratch_label:  str,            # e.g. "ResNet (scratch)" or "Net1D (scratch)"
    zeroshot_auroc: float,          # scalar — zero-shot AUROC
    ft_color:       str,
    title:          str,
    out_path:       str,
):
    """
    Single-panel line plot: Macro AUROC vs training samples.

    x=0  →  ECGFounder zero-shot (for FT line) / random init ~0.5 (for scratch)
    x>0  →  actual n_train values
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#111720")

    # ── ECGFounder FT line (with zero-shot anchor at x=0) ────────────────────
    ft_pts = ft_sub.sort_values("n_train_actual")
    xs_ft  = [0] + list(ft_pts["n_train_actual"])
    ys_ft  = [zeroshot_auroc] + list(ft_pts["test_macro_AUROC"])

    ax.plot(xs_ft, ys_ft, marker="o", color=ft_color, lw=2.5,
            label="ECGFounder FT")
    for x, y in zip(xs_ft, ys_ft):
        label_txt = f"ZS={y:.3f}" if x == 0 else f"{y:.3f}"
        ax.annotate(label_txt, (x, y),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.5, color=ft_color)

    # Mark zero-shot anchor specially
    ax.scatter([0], [zeroshot_auroc], color=ft_color, s=120, zorder=6,
               marker="*", label=f"ECGFounder Zero-shot ({zeroshot_auroc:.3f})")

    # ── Scratch baseline lines (random-init anchor ~0.5 at x=0) ──────────────
    for size in scratch_sizes:
        color = scratch_colors.get(size, "#aaaaaa")
        sub   = scratch_sub[scratch_sub["model_size"] == size].sort_values("n_train_actual")
        if sub.empty:
            continue
        xs = [0] + list(sub["n_train_actual"])
        ys = [0.5] + list(sub["test_macro_AUROC"])
        ax.plot(xs, ys, marker="s", color=color, lw=1.8, ls="--",
                alpha=0.85, label=f"{scratch_label} {size.capitalize()}")
        for x, y in zip(xs[1:], ys[1:]):   # skip x=0 label for scratch (0.5 is implied)
            ax.annotate(f"{y:.3f}", (x, y),
                        textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=8, color=color, alpha=0.85)

    # Reference line at 0.5
    ax.axhline(0.5, color="#555555", lw=1.0, ls=":", zorder=3,
               label="Random (AUROC=0.500)")

    ax.set_xlabel("Training Samples (0 = no fine-tuning / random init)",
                  color="#cccccc", fontsize=12)
    ax.set_ylabel("Test Macro AUROC", color="#cccccc", fontsize=12)
    ax.set_title(title, color="#ffffff", fontsize=13)
    ax.tick_params(colors="#888888")
    for s in ax.spines.values():
        s.set_edgecolor("#2a2a2a")
    ax.grid(color="#1e1e1e", lw=0.5, ls="--")
    ax.legend(fontsize=9.5, facecolor="#141920",
              edgecolor="#444", labelcolor="#eee",
              loc="lower right")

    plt.tight_layout(pad=2)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Comparison plot saved: {out_path}")


def plot_ablation_comparisons(
    summary_df:     pd.DataFrame,
    zeroshot_auroc: float,
    out_dir:        str,
):
    """
    Produce two comparison plots:
      1. ECGFounder FT vs ResNet-scratch
      2. ECGFounder FT vs Net1D-scratch
    Both include the zero-shot anchor at x=0.
    """
    ft_color    = "#40e0d0"
    ft_sub      = summary_df.dropna(subset=["test_macro_AUROC"]) \
                             .sort_values("n_train_actual")

    # ── Plot 1: vs ResNet scratch ─────────────────────────────────────────────
    if os.path.exists(RESNET_SUMMARY_PATH):
        rn_df = pd.read_csv(RESNET_SUMMARY_PATH)
        rn_df = rn_df.dropna(subset=["test_macro_AUROC"])
        rn_sizes  = ["large", "medium", "small"]
        rn_colors = {"large": "#f0c040", "medium": "#e8a030", "small": "#e06060"}
        _plot_one_comparison(
            ft_sub         = ft_sub,
            scratch_sub    = rn_df,
            scratch_sizes  = rn_sizes,
            scratch_colors = rn_colors,
            scratch_label  = "ResNet (scratch)",
            zeroshot_auroc = zeroshot_auroc,
            ft_color       = ft_color,
            title          = "ECGFounder FT vs ResNet (scratch) — AUROC vs Training Budget\n"
                             "(x=0: ECGFounder zero-shot  |  ResNet random-init ≈ 0.50)",
            out_path       = os.path.join(out_dir, "ft_vs_resnet_comparison.png"),
        )
        logger.info("FT vs ResNet comparison plot generated.")
    else:
        logger.warning(
            f"ResNet ablation summary not found at {RESNET_SUMMARY_PATH} — "
            "skipping FT vs ResNet plot."
        )

    # ── Plot 2: vs Net1D scratch ──────────────────────────────────────────────
    if os.path.exists(NET1D_SUMMARY_PATH):
        n1d_df = pd.read_csv(NET1D_SUMMARY_PATH)
        n1d_df = n1d_df.dropna(subset=["test_macro_AUROC"])
        n1d_sizes  = ["large", "medium", "small"]
        n1d_colors = {"large": "#c77dff", "medium": "#9d4edd", "small": "#7b2d8b"}
        _plot_one_comparison(
            ft_sub         = ft_sub,
            scratch_sub    = n1d_df,
            scratch_sizes  = n1d_sizes,
            scratch_colors = n1d_colors,
            scratch_label  = "Net1D (scratch)",
            zeroshot_auroc = zeroshot_auroc,
            ft_color       = ft_color,
            title          = "ECGFounder FT vs Net1D (scratch) — AUROC vs Training Budget\n"
                             "(x=0: ECGFounder zero-shot  |  Net1D random-init ≈ 0.50)",
            out_path       = os.path.join(out_dir, "ft_vs_net1d_comparison.png"),
        )
        logger.info("FT vs Net1D comparison plot generated.")
    else:
        logger.warning(
            f"Net1D ablation summary not found at {NET1D_SUMMARY_PATH} — "
            "skipping FT vs Net1D plot."
        )

    # ── Legacy single-panel FT-only plot (kept for backward compatibility) ────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#111720")

    xs = [0] + list(ft_sub.sort_values("n_train_actual")["n_train_actual"])
    ys = [zeroshot_auroc] + list(ft_sub.sort_values("n_train_actual")["test_macro_AUROC"])
    ax.plot(xs, ys, marker="o", color=ft_color, lw=2.5, label="ECGFounder FT")
    ax.scatter([0], [zeroshot_auroc], color=ft_color, s=120, zorder=6,
               marker="*", label=f"Zero-shot ({zeroshot_auroc:.3f})")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=ft_color)
    ax.axhline(0.5, color="#555555", lw=1.0, ls=":", label="Random (AUROC=0.500)")
    ax.set_xlabel("Training Samples (0 = zero-shot)", color="#cccccc", fontsize=12)
    ax.set_ylabel("Test Macro AUROC", color="#cccccc", fontsize=12)
    ax.set_title("ECGFounder FT — AUROC vs Training Budget",
                 color="#ffffff", fontsize=13)
    ax.tick_params(colors="#888888")
    for s in ax.spines.values():
        s.set_edgecolor("#2a2a2a")
    ax.grid(color="#1e1e1e", lw=0.5, ls="--")
    ax.legend(fontsize=10, facecolor="#141920", edgecolor="#444", labelcolor="#eee")
    plt.tight_layout(pad=2)
    out_path = os.path.join(out_dir, "ft_ablation_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"FT-only ablation comparison plot saved: {out_path}")

# =============================================================================
# SINGLE EXPERIMENT  (one sample budget, full FT)
# skip-retrain: if checkpoint + test arrays exist, load and evaluate only
# =============================================================================
def run_experiment(
    budget,               # int or None
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> dict:
    budget_label = str(budget) if budget is not None else "full"
    exp_id       = f"ft_{budget_label}"
    exp_dir      = os.path.join(saved_dir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT: {exp_id}")
    logger.info(f"{'='*70}")

    # ── training subset ───────────────────────────────────────────────────────
    if budget is not None:
        cur_train_df = subsample_train(train_df, budget, seed=SEED, logger=logger)
    else:
        cur_train_df = train_df
        logger.info(f"  Using full training set: {len(cur_train_df)} records")

    # ── check if already trained ──────────────────────────────────────────────
    existing_ckpt = find_best_checkpoint(exp_dir, exp_id)
    test_gt_path   = os.path.join(exp_dir, "test_gt.npy")
    test_pred_path = os.path.join(exp_dir, "test_pred.npy")

    if existing_ckpt is not None:
        logger.info(
            f"  [SKIP TRAINING] Checkpoint found: {os.path.basename(existing_ckpt)}"
        )
        # Load model from checkpoint
        model = FT_Model().to(device)
        ckpt  = torch.load(existing_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        best_val_auroc = ckpt.get("val_auroc", float("nan"))
        total_p     = sum(p.numel() for p in model.parameters())
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"  Loaded epoch {ckpt.get('epoch', '?')} | "
            f"val AUROC: {best_val_auroc:.4f}"
        )

        # Re-use cached test predictions if they exist and match test set size
        if os.path.exists(test_gt_path) and os.path.exists(test_pred_path):
            test_gt   = np.load(test_gt_path)
            test_pred = np.load(test_pred_path)
            if test_gt.shape[0] == len(test_df):
                logger.info(
                    f"  [SKIP TEST EVAL] Cached test predictions found "
                    f"({test_gt.shape[0]} samples). Re-computing metrics …"
                )
                test_macro = full_evaluation(
                    test_gt, test_pred, labels_list, n_classes,
                    split_name=f"{exp_id}_test",
                    out_dir=exp_dir,
                    n_resamples=100,
                )
                return {
                    "experiment"        : exp_id,
                    "model"             : "ECGFounder_FT",
                    "n_train_requested" : budget_label,
                    "n_train_actual"    : len(cur_train_df),
                    "n_val"             : len(val_df),
                    "n_test"            : len(test_df),
                    "total_params"      : total_p,
                    "trainable_params"  : trainable_p,
                    "best_val_AUROC"    : round(best_val_auroc, 4),
                    "test_macro_AUROC"  : round(test_macro, 4),
                }
            else:
                logger.warning(
                    f"  Cached test predictions size mismatch — re-running test eval."
                )

        # No cached predictions: run test evaluation from loaded checkpoint
        logger.info("  Running test evaluation from loaded checkpoint …")
        testloader = DataLoader(
            ECG_Dataset(ecg_path, test_df),
            batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )
        model.eval()
        test_gt_list, test_pred_list = [], []
        with torch.no_grad():
            for x, y in tqdm(testloader, desc=f"[{exp_id}] Test", leave=False):
                x, y = x.to(device), y.to(device)
                test_pred_list.append(torch.sigmoid(model(x)).cpu().numpy())
                test_gt_list.append(y.cpu().numpy())
        test_gt   = np.concatenate(test_gt_list)
        test_pred = np.concatenate(test_pred_list)
        np.save(test_gt_path,   test_gt)
        np.save(test_pred_path, test_pred)

        test_macro = full_evaluation(
            test_gt, test_pred, labels_list, n_classes,
            split_name=f"{exp_id}_test",
            out_dir=exp_dir,
            n_resamples=100,
        )
        return {
            "experiment"        : exp_id,
            "model"             : "ECGFounder_FT",
            "n_train_requested" : budget_label,
            "n_train_actual"    : len(cur_train_df),
            "n_val"             : len(val_df),
            "n_test"            : len(test_df),
            "total_params"      : total_p,
            "trainable_params"  : trainable_p,
            "best_val_AUROC"    : round(best_val_auroc, 4),
            "test_macro_AUROC"  : round(test_macro, 4),
        }

    # =========================================================================
    # FULL TRAINING PATH (no checkpoint found)
    # =========================================================================
    trainloader = DataLoader(
        ECG_Dataset(ecg_path, cur_train_df),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valloader = DataLoader(
        ECG_Dataset(ecg_path, val_df),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    testloader = DataLoader(
        ECG_Dataset(ecg_path, test_df),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    logger.info(
        f"  Train batches: {len(trainloader)} | "
        f"Val batches: {len(valloader)} | "
        f"Test batches: {len(testloader)}"
    )

    # ── model — fresh pretrained weights every experiment ─────────────────────
    model = FT_Model().to(device)
    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total params    : {total_p:,}")
    logger.info(f"  Trainable params: {trainable_p:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.1)
    criterion = nn.BCEWithLogitsLoss()
    es        = EarlyStopping(patience=EARLY_STOP_PATIENCE,
                               min_delta=EARLY_STOP_MIN_DELTA)

    best_val_auroc = 0.0
    last_ckpt_path = None

    for epoch in range(Epochs):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for x, y in tqdm(trainloader,
                          desc=f"[{exp_id}] E{epoch+1} Train", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(trainloader)

        # ── validate ──────────────────────────────────────────────────────────
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
            f"[{exp_id}] E{epoch+1:03d}  "
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

        # ── checkpoint ────────────────────────────────────────────────────────
        if val_macro > best_val_auroc:
            best_val_auroc = val_macro
            new_ckpt = os.path.join(
                exp_dir,
                f"{exp_id}_epoch{epoch+1}_auroc{val_macro:.4f}.pth"
            )
            if last_ckpt_path and os.path.exists(last_ckpt_path):
                os.remove(last_ckpt_path)
            torch.save({
                "epoch"      : epoch + 1,
                "state_dict" : model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "scheduler"  : scheduler.state_dict(),
                "val_auroc"  : val_macro,
                "experiment" : exp_id,
                "n_train"    : len(cur_train_df),
                "config": {
                    "n_classes" : n_classes,
                    "batch_size": batch_size,
                    "lr"        : LR,
                    "seed"      : SEED,
                },
            }, new_ckpt)
            last_ckpt_path = new_ckpt
            logger.info(f"  [SAVED] {os.path.basename(new_ckpt)}")

        # ── stops ─────────────────────────────────────────────────────────────
        if optimizer.param_groups[0]["lr"] < early_stop_lr:
            logger.info("  LR floor reached — stopping.")
            break
        if es(val_macro):
            logger.info(f"  Early stopping triggered at epoch {epoch+1}.")
            break

    logger.info(
        f"[{exp_id}] Training done | best val AUROC: {best_val_auroc:.4f}"
    )

    # ── reload best weights ───────────────────────────────────────────────────
    if last_ckpt_path and os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        logger.info(
            f"[{exp_id}] Reloaded best checkpoint (epoch {ckpt['epoch']})"
        )
    else:
        logger.warning(f"[{exp_id}] No checkpoint — using last-epoch weights.")

    # ── test evaluation ───────────────────────────────────────────────────────
    model.eval()
    test_gt_list, test_pred_list = [], []
    with torch.no_grad():
        for x, y in tqdm(testloader, desc=f"[{exp_id}] Test", leave=False):
            x, y = x.to(device), y.to(device)
            test_pred_list.append(torch.sigmoid(model(x)).cpu().numpy())
            test_gt_list.append(y.cpu().numpy())

    test_gt   = np.concatenate(test_gt_list)
    test_pred = np.concatenate(test_pred_list)

    np.save(test_gt_path,   test_gt)
    np.save(test_pred_path, test_pred)

    test_macro = full_evaluation(
        test_gt, test_pred, labels_list, n_classes,
        split_name=f"{exp_id}_test",
        out_dir=exp_dir,
        n_resamples=100,
    )

    return {
        "experiment"        : exp_id,
        "model"             : "ECGFounder_FT",
        "n_train_requested" : budget_label,
        "n_train_actual"    : len(cur_train_df),
        "n_val"             : len(val_df),
        "n_test"            : len(test_df),
        "total_params"      : total_p,
        "trainable_params"  : trainable_p,
        "best_val_AUROC"    : round(best_val_auroc, 4),
        "test_macro_AUROC"  : round(test_macro, 4),
    }

# =============================================================================
# MAIN
# =============================================================================
def main():
    # ── load data ─────────────────────────────────────────────────────────────
    ecg_df = pd.read_csv(csv_path)
    if isinstance(ecg_df["label"].iloc[0], str):
        ecg_df["label"] = ecg_df["label"].apply(json.loads)

    # ── identical split to resnet_ablation.py and net1d_ablation.py ───────────
    train_df, val_df, test_df = prepare_splits(ecg_df, logger=logger)

    # ── zero-shot baseline (run once, cached thereafter) ──────────────────────
    logger.info("\n" + "="*70)
    logger.info("ZERO-SHOT BASELINE (ECGFounder, no fine-tuning)")
    logger.info("="*70)
    zeroshot_auroc = run_zeroshot_baseline(test_df)
    logger.info(f"Zero-shot macro AUROC: {zeroshot_auroc:.4f}")

    # ── run all FT experiments ────────────────────────────────────────────────
    all_summaries = []

    for budget in SAMPLE_BUDGETS:
        budget_label = str(budget) if budget is not None else "full"
        try:
            summary = run_experiment(
                budget   = budget,
                train_df = train_df,
                val_df   = val_df,
                test_df  = test_df,
            )
            all_summaries.append(summary)
        except Exception as exc:
            logger.error(
                f"Experiment ft_{budget_label} FAILED: {exc}", exc_info=True
            )
            all_summaries.append({
                "experiment"        : f"ft_{budget_label}",
                "model"             : "ECGFounder_FT",
                "n_train_requested" : budget_label,
                "error"             : str(exc),
            })

    # ── save summary table ────────────────────────────────────────────────────
    summary_df   = pd.DataFrame(all_summaries)
    summary_path = os.path.join(saved_dir, "ft_ablation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nFT ablation summary saved: {summary_path}")

    # Also append zero-shot row for reference
    zs_row = pd.DataFrame([{
        "experiment"        : "ft_zeroshot",
        "model"             : "ECGFounder_ZeroShot",
        "n_train_requested" : "0",
        "n_train_actual"    : 0,
        "n_val"             : len(val_df),
        "n_test"            : len(test_df),
        "best_val_AUROC"    : float("nan"),
        "test_macro_AUROC"  : round(zeroshot_auroc, 4),
    }])
    summary_with_zs = pd.concat([zs_row, summary_df], ignore_index=True)
    summary_with_zs.to_csv(
        os.path.join(saved_dir, "ft_ablation_summary_with_zeroshot.csv"),
        index=False
    )

    # ── print table ───────────────────────────────────────────────────────────
    cols = ["experiment", "n_train_requested", "n_train_actual",
            "total_params", "best_val_AUROC", "test_macro_AUROC"]
    display_df = summary_df[[c for c in cols if c in summary_df.columns]]
    logger.info(f"\n{'='*70}\nFT ABLATION RESULTS\n{'='*70}")
    logger.info(display_df.to_string(index=False))
    logger.info(f"\nZero-shot baseline macro AUROC: {zeroshot_auroc:.4f}")

    # ── comparison plots (FT vs ResNet-scratch and FT vs Net1D-scratch) ───────
    try:
        plot_ablation_comparisons(
            summary_df     = summary_df.dropna(subset=["test_macro_AUROC"]),
            zeroshot_auroc = zeroshot_auroc,
            out_dir        = saved_dir,
        )
    except Exception as exc:
        logger.warning(f"Comparison plots failed: {exc}", exc_info=True)

    logger.info("\nAll FT ablation experiments complete.")


if __name__ == "__main__":
    main()