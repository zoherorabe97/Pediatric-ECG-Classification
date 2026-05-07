import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import pandas as pd
import json
import logging
from collections import Counter
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
gpu_id     = 0
batch_size = 1   # batch over patches (see evaluate_split)

ecg_path   = "C:/Users/zoorab/Desktop/zoher/University/Projects/Zhengzhou_ECG/Child_ecg/"
csv_path   = "./ecg_with_exact_match.csv"
pth        = "./12_lead_ECGFounder.pth"
tasks_path = "./tasks.txt"

saved_dir  = "./res/zeroshot_pediatric_only/"
os.makedirs(saved_dir, exist_ok=True)
os.makedirs("logging", exist_ok=True)

# =============================================================================
# PATCH / SEGMENTATION PARAMETERS
# =============================================================================
FS           = 500          # Hz – sampling frequency the model was trained on
SEGMENT_LEN  = 10 * FS     # 5 000 samples  (10 s at 500 Hz)
OVERLAP      = 256          # adjustable overlap between consecutive patches (samples)
AGG_METHOD   = "mean"       # "mean" or "max"  –  how to aggregate patch predictions

# =============================================================================
# LOGGING
# =============================================================================
log_file = "logging/zeroshot_pediatric_only.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")

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
# PATCH EXTRACTION
# =============================================================================
def extract_patches(signal: np.ndarray, seg_len: int = SEGMENT_LEN, overlap: int = OVERLAP) -> np.ndarray:
    """
    Slice a (12, T) signal into overlapping fixed-length patches.

    Strategy
    --------
    - If T >= seg_len : extract consecutive patches with step = seg_len - overlap.
      The last patch is anchored at the very end of the signal (adaptive overlap)
      so no signal is discarded.
    - If T <  seg_len : zero-pad on the right to reach seg_len; return 1 patch.

    Returns
    -------
    patches : np.ndarray of shape (P, 12, seg_len)
    """
    n_leads, T = signal.shape

    # ── short recording: zero-pad ─────────────────────────────────────────────
    if T < seg_len:
        pad = np.zeros((n_leads, seg_len - T), dtype=signal.dtype)
        patch = np.concatenate([signal, pad], axis=-1)          # (12, seg_len)
        return patch[np.newaxis]                                 # (1, 12, seg_len)

    # ── long recording: sliding window ────────────────────────────────────────
    step = seg_len - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be < seg_len ({seg_len})")

    starts = list(range(0, T - seg_len + 1, step))

    # Adaptive last patch: make sure the very end of the signal is covered.
    last_start = T - seg_len
    if starts[-1] != last_start:
        starts.append(last_start)

    patches = np.stack([signal[:, s: s + seg_len] for s in starts], axis=0)  # (P, 12, seg_len)
    return patches


# =============================================================================
# DATASET  –  returns all patches for a single recording
# =============================================================================
class ECG_Dataset(Dataset):
    """
    Each __getitem__ call returns:
        patches : FloatTensor  (P, 12, seg_len)   – zero-normed patches
        label   : FloatTensor  (n_classes,)
    """

    def __init__(self, ecg_path, df,
                 seg_len: int = SEGMENT_LEN, overlap: int = OVERLAP):
        self.ecg_path = ecg_path
        self.data     = df.copy().reset_index(drop=True)
        self.seg_len  = seg_len
        self.overlap  = overlap

    # ------------------------------------------------------------------
    @staticmethod
    def z_score_normalization(signal: np.ndarray) -> np.ndarray:
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row       = self.data.iloc[idx]
        label     = torch.tensor(row["label"], dtype=torch.float)
        file_path = self.ecg_path + row["Filename"]

        try:
            data, hdr = wfdb.rdsamp(file_path)          # (T, 12)
            data = np.transpose(data, (1, 0))            # (12, T)
            data = self.z_score_normalization(data)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}  – returning zero patch")
            data = np.zeros((12, self.seg_len), dtype=np.float32)

        patches = extract_patches(data, seg_len=self.seg_len, overlap=self.overlap)
        # patches : (P, 12, seg_len)  as float32
        patches = patches.astype(np.float32)

        return torch.from_numpy(patches), label


# =============================================================================
# COLLATE  –  each sample already has a fixed patch size (seg_len)
# but different samples may have a different number of patches P.
# Since we run with batch_size=1 this is fine; keep a trivial collate.
# =============================================================================
def identity_collate(batch):
    # batch is a list of length 1: [(patches_tensor, label_tensor)]
    patches, label = batch[0]
    return patches.unsqueeze(0), label.unsqueeze(0)
    # patches : (1, P, 12, seg_len)
    # label   : (1, n_classes)


# =============================================================================
# DATA SPLITS
# =============================================================================
def prepare_splits(ecg_df):
    unique_patients               = ecg_df["Patient_ID"].unique()
    train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)
    train_patients, val_patients  = train_test_split(train_patients,  test_size=0.1, random_state=42)

    test_df = ecg_df[ecg_df["Patient_ID"].isin(test_patients)].reset_index(drop=True)
    test_df = test_df[test_df["label"].apply(lambda x: np.array(x).sum() > 0)].reset_index(drop=True)

    full_df = ecg_df[ecg_df["label"].apply(lambda x: np.array(x).sum() > 0)].reset_index(drop=True)

    logger.info(f"Full Valid Dataset : {len(full_df)} samples")
    logger.info(f"Test Dataset       : {len(test_df)} samples")

    return full_df, test_df


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
    cmap_a = cm.get_cmap("tab20",  20)
    cmap_b = cm.get_cmap("tab20b", 20)
    def get_color(k):
        return cmap_a(k % 20) if k < 20 else cmap_b((k - 20) % 20)

    for k, (fpr, tpr, roc_auc_val, label) in enumerate(zip(valid_fprs, valid_tprs, valid_aucs, valid_labels)):
        ax.plot(fpr, tpr, color=get_color(k), lw=0.7, alpha=0.40, label=f"{label} (AUC={roc_auc_val:.2f})")

    mean_fpr    = np.linspace(0, 1, 500)
    mean_tpr    = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(valid_fprs, valid_tprs)], axis=0)
    mean_tpr[0] = 0.0
    macro_auc_val = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, color="#f0c040", lw=2.8, ls="--", zorder=10, label=f"Macro-average  (AUC = {macro_auc_val:.3f})")

    gt_micro   = all_gt[:,   valid_indices].ravel()
    pred_micro = all_pred[:, valid_indices].ravel()
    fpr_m, tpr_m, _ = roc_curve(gt_micro, pred_micro)
    micro_auc_val   = auc(fpr_m, tpr_m)
    ax.plot(fpr_m, tpr_m, color="#40e0d0", lw=2.8, ls="-.", zorder=10, label=f"Micro-average  (AUC = {micro_auc_val:.3f})")

    ax.plot([0, 1], [0, 1], color="#666666", lw=1.0, ls=":", zorder=5, label="Chance  (AUC = 0.500)")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", color="#cccccc", fontsize=13, labelpad=8)
    ax.set_ylabel("True Positive Rate",  color="#cccccc", fontsize=13, labelpad=8)
    ax.set_title(
        f"ZeroShot ROC Curves — {split_name}\n"
        f"{n_valid} valid / {n_classes} total labels  |  Macro AUC={macro_auc_val:.3f}   Micro AUC={micro_auc_val:.3f}",
        color="#ffffff", fontsize=13, pad=14
    )
    ax.tick_params(colors="#888888", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a2a")
    ax.grid(color="#1e1e1e", lw=0.5, linestyle="--")

    handles, leg_labels = ax.get_legend_handles_labels()
    ax.legend(handles[-3:], leg_labels[-3:], loc="lower right", fontsize=11,
              framealpha=0.75, facecolor="#141920", edgecolor="#444444", labelcolor="#eeeeee")
    ax.legend(handles[:-3], leg_labels[:-3], loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=5.5, framealpha=0.55, facecolor="#141920", edgecolor="#333333",
              labelcolor="#bbbbbb", ncol=2 if n_valid > 30 else 1,
              handlelength=1.2, handleheight=0.8, borderpad=0.5, labelspacing=0.25)
    plt.tight_layout(rect=[0, 0, 0.62 if (2 if n_valid > 30 else 1) == 2 else 0.73, 1])

    out_path = os.path.join(saved_dir, f"{split_name}_roc_curves.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return macro_auc_val, micro_auc_val


def full_evaluation(all_gt, all_pred, split_name, n_resamples=100):
    logger.info(f"\nComputing evaluation for: {split_name}")
    results = []

    for i, label in enumerate(tqdm(labels, desc=f"Metrics [{split_name}]")):
        true  = all_gt[:, i]
        pred  = all_pred[:, i]
        if len(np.unique(true)) < 2:
            continue

        threshold = compute_optimal_threshold(true, pred)
        sens, spec, prec, f1, ppv, npv, auroc, auprc = calculate_performance_metrics(true, pred, threshold)

        auroc_ci = bootstrap_ci(
            lambda t, p, th: calculate_performance_metrics(t, p, th)[6],
            true, pred, threshold, n_resamples
        )

        results.append({
            "Label"    : label,
            "n_pos"    : int(true.sum()),
            "AUROC"    : round(auroc, 3) if not np.isnan(auroc) else np.nan,
            "AUROC_CI" : auroc_ci,
        })

    results_df = pd.DataFrame(results).sort_values("AUROC", ascending=False)
    results_df.to_csv(os.path.join(saved_dir, f"{split_name}_results.csv"), index=False)

    macro_auroc = results_df["AUROC"].dropna().mean()
    logger.info(f"[{split_name}] Macro AUROC: {macro_auroc:.4f}  ({len(results_df)} valid labels)")

    plot_macro, plot_micro = plot_roc_curves(all_gt, all_pred, labels, n_classes, split_name, saved_dir)
    logger.info(f"[{split_name}] Plot Macro AUC={plot_macro:.4f}  Micro AUC={plot_micro:.4f}")


# =============================================================================
# PATCH COUNT DISTRIBUTION LOGGER
# =============================================================================
def log_patch_distribution(split_name: str, patch_counts: list[int]) -> None:
    arr = np.array(patch_counts)
    buckets = Counter(arr.tolist())
    logger.info(
        f"\n[{split_name}] Patch-count distribution over {len(arr)} recordings:\n"
        f"  min={arr.min()}  max={arr.max()}  mean={arr.mean():.1f}  median={np.median(arr):.0f}"
    )
    logger.info(f"  Patch counts per recording:")
    for n_patches, count in sorted(buckets.items()):
        bar = "█" * min(count, 60)
        logger.info(f"    {n_patches:>4} patch(es) : {count:>5}  {bar}")

    csv_out = os.path.join(saved_dir, f"{split_name}_patch_distribution.csv")
    pd.DataFrame({"n_patches": arr}).to_csv(csv_out, index=False)
    logger.info(f"  Patch distribution saved → {csv_out}")


# =============================================================================
# INFERENCE ROUTINE  –  patch-based
# =============================================================================
def aggregate_patch_predictions(patch_probs: np.ndarray, method: str = AGG_METHOD) -> np.ndarray:
    """
    Aggregate probabilities across P patches.

    Parameters
    ----------
    patch_probs : (P, n_classes)
    method      : "mean" or "max"

    Returns
    -------
    (n_classes,) aggregated probabilities
    """
    if method == "mean":
        return patch_probs.mean(axis=0)
    elif method == "max":
        return patch_probs.max(axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method!r}. Use 'mean' or 'max'.")


def evaluate_split(model, df, split_name,
                   seg_len: int  = SEGMENT_LEN,
                   overlap: int  = OVERLAP,
                   agg_method: str = AGG_METHOD):
    """
    Run zero-shot patch-based inference on `df`.

    Each recording is sliced into fixed-length patches (seg_len samples).
    Recordings shorter than seg_len are zero-padded to exactly seg_len.
    The model predicts on every patch independently; predictions are then
    aggregated (mean or max) into a single recording-level prediction.

    Parameters
    ----------
    seg_len    : patch length in samples (default 5000 = 10 s × 500 Hz)
    overlap    : number of overlapping samples between consecutive patches
    agg_method : "mean" or "max" aggregation over patches
    """
    logger.info(
        f"\n[{split_name}] Patch config:  seg_len={seg_len} ({seg_len/FS:.1f}s)  "
        f"overlap={overlap}  step={seg_len - overlap}  agg={agg_method!r}"
    )

    dataset = ECG_Dataset(ecg_path=ecg_path, df=df, seg_len=seg_len, overlap=overlap)
    # batch_size=1 at recording level; identity_collate keeps (1, P, 12, seg_len)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=identity_collate)

    all_gt, all_pred = [], []
    patch_counts     = []

    with torch.no_grad():
        for patches_batch, labels_batch in tqdm(loader, desc=f"Patch inference ({split_name})"):
            # patches_batch : (1, P, 12, seg_len)
            # labels_batch  : (1, n_classes)
            patches = patches_batch[0]       # (P, 12, seg_len)
            label   = labels_batch[0]        # (n_classes,)

            P = patches.shape[0]
            patch_counts.append(P)

            # Run model on all P patches in one forward pass (fits in GPU memory
            # for typical P; if OOM, fall back to a per-patch loop).
            patches = patches.to(device)     # (P, 12, seg_len)
            try:
                patch_logits = model(patches)    # (P, n_classes)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM on {P} patches – falling back to per-patch loop")
                    torch.cuda.empty_cache()
                    patch_logits = torch.cat(
                        [model(patches[i:i+1]) for i in range(P)], dim=0
                    )
                else:
                    raise

            patch_probs = torch.sigmoid(patch_logits).cpu().numpy()  # (P, n_classes)

            # Aggregate across patches → recording-level prediction
            rec_pred = aggregate_patch_predictions(patch_probs, method=agg_method)  # (n_classes,)

            all_pred.append(rec_pred)
            all_gt.append(label.numpy())

    # ── patch-count stats ─────────────────────────────────────────────────────
    log_patch_distribution(split_name, patch_counts)

    zs_gt   = np.stack(all_gt)    # (N, n_classes)
    zs_pred = np.stack(all_pred)  # (N, n_classes)

    full_evaluation(zs_gt, zs_pred, split_name)


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Load dataset
    ecg_df = pd.read_csv(csv_path)
    if isinstance(ecg_df["label"].iloc[0], str):
        ecg_df["label"] = ecg_df["label"].apply(json.loads)

    # Get splits
    full_df, test_df = prepare_splits(ecg_df)

    # Load Foundation Model
    model = Net1D(
        in_channels=12, base_filters=64, ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16, stride=2, groups_width=16,
        verbose=False, use_bn=False, use_do=False, n_classes=n_classes
    )
    checkpoint = torch.load(pth, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]
    log = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Model loaded. Missing: {len(log.missing_keys)}, Unexpected: {len(log.unexpected_keys)}")

    for param in model.parameters():
        param.requires_grad = False

    model.to(device)
    model.eval()

    # ── patch-based inference ─────────────────────────────────────────────────
    # Tweak SEGMENT_LEN, OVERLAP, and AGG_METHOD at the top of this file,
    # or pass them directly to evaluate_split() for per-run overrides.

    logger.info("\n--- EVALUATING ON ENTIRE PEDIATRIC DATASET (patch-based) ---")
    evaluate_split(model, full_df, "Full_Data",
                   seg_len=SEGMENT_LEN, overlap=OVERLAP, agg_method=AGG_METHOD)

    logger.info("\n--- EVALUATING ON TEST SPLIT ONLY (patch-based) ---")
    evaluate_split(model, test_df, "Test_Set",
                   seg_len=SEGMENT_LEN, overlap=OVERLAP, agg_method=AGG_METHOD)


if __name__ == "__main__":
    main()