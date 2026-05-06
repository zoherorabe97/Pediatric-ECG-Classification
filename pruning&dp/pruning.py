# =============================================================================
# Pruning of Fine-tuned ECGFounder on Child ECG Dataset
# Prunes the fine-tuned model (not the original pretrained weights),
# evaluates each pruning ratio on the full ecg_df, and produces
# per-class ROC plots + a pruning summary table.
#
# Key fix vs previous version:
#   load_fresh_model() now strips the 'backbone.' prefix that ft_ChildECG
#   adds when saving, so fine-tuned weights load with strict=True correctly.
#   A sanity-check AUROC gate (> 0.55) aborts early if loading fails.
# =============================================================================

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc

import torch_pruning as tp
import wfdb

from net1d import Net1D

# =============================================================================
# CONFIG
# =============================================================================
gpu_id     = 0
batch_size = 32

# Pruning settings
PRUNING_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]
PRUNING_METHOD = "magnitude"    # "magnitude" | "random"
GLOBAL_PRUNING = False
ISOMORPHIC     = True
ROUND_TO       = 8

# Paths
ecg_path   = "C:/Users/zoorab/Desktop/zoher/University/Projects/Zhengzhou_ECG/Child_ecg/"
csv_path   = "./ecg_with_exact_match.csv"
tasks_path = "./tasks.txt"
saved_dir  = "./res/pruning_ft/"
log_file   = "logging/pruning_ft.log"

# Path to the fine-tuned checkpoint saved by ft.py
pth = "C:/Users/zoorab/Desktop/zoher/University/Projects/ECGFounder/res/finetune/child_ecg_full_ft_epoch1_auroc0.8408.pth"

target_fs = 5000

os.makedirs(saved_dir,                 exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# =============================================================================
# LOGGING
# =============================================================================
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
    def __init__(self, ecg_path, df, target_fs=5000):
        self.ecg_path  = ecg_path
        self.data      = df.copy().reset_index(drop=True)
        self.target_fs = target_fs

    def z_score_normalization(self, signal):
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    def resample_unequal(self, ts, fs_in, fs_out):
        if fs_in == 0 or len(ts) == 0:
            return ts
        t      = ts.shape[1] / fs_in
        fs_in  = int(fs_in)
        fs_out = int(fs_out)
        if fs_out == fs_in:
            return ts
        if 2 * fs_out == fs_in:
            return ts[:, ::2]
        resampled = np.zeros((ts.shape[0], fs_out))
        x_old     = np.linspace(0, t, num=ts.shape[1], endpoint=True)
        x_new     = np.linspace(0, t, num=int(fs_out), endpoint=True)
        for i in range(ts.shape[0]):
            f = interp1d(x_old, ts[i, :], kind="linear")
            resampled[i, :] = f(x_new)
        return resampled

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
            logger.warning(f"Failed to load {file_path}: {e} — returning zeros")
            signal = torch.zeros((12, self.target_fs))
        return signal, label


# =============================================================================
# MODEL LOADER  — KEY FIX
# =============================================================================
def load_fresh_model(pth, n_classes, device):
    """
    Load a bare Net1D from a fine-tuned ft_ChildECG checkpoint.

    ft_ChildECG wraps Net1D under self.backbone, so every saved key is
    prefixed 'backbone.XXX'.  A bare Net1D expects 'XXX'.
    We detect and strip that prefix so strict=True loading works correctly.

    If the checkpoint was saved directly from Net1D (no wrapper, e.g. the
    original pretrained model), keys have no prefix and load as-is.
    """
    model = Net1D(
        in_channels   = 12,
        base_filters  = 64,
        ratio         = 1,
        filter_list   = [64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list = [2, 2, 2, 3, 3, 4, 4],
        kernel_size   = 16,
        stride        = 2,
        groups_width  = 16,
        verbose       = False,
        use_bn        = False,
        use_do        = False,
        n_classes     = n_classes
    )

    checkpoint = torch.load(pth, map_location=device, weights_only=False)
    raw_sd     = checkpoint["state_dict"]

    # detect wrapper prefix
    has_backbone_prefix = any(k.startswith("backbone.") for k in raw_sd)

    if has_backbone_prefix:
        state_dict = {
            k[len("backbone."):]: v
            for k, v in raw_sd.items()
            if k.startswith("backbone.")
        }
        logger.info("  Detected 'backbone.' prefix — stripping for bare Net1D")
    else:
        state_dict = raw_sd
        logger.info("  No wrapper prefix detected — loading keys as-is")

    log = model.load_state_dict(state_dict, strict=True)
    logger.info(
        f"  Weights loaded (strict=True) | "
        f"missing: {len(log.missing_keys)} | unexpected: {len(log.unexpected_keys)}"
    )

    # warn if anything is off — strict=True would have raised, but log anyway
    if log.missing_keys:
        logger.warning(f"  Missing  : {log.missing_keys[:5]}{'...' if len(log.missing_keys) > 5 else ''}")
    if log.unexpected_keys:
        logger.warning(f"  Unexpected: {log.unexpected_keys[:5]}{'...' if len(log.unexpected_keys) > 5 else ''}")

    return model.to(device)


# =============================================================================
# METRICS
# =============================================================================
def calculate_performance_metrics(true, pred, threshold):
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
    auroc     = roc_auc_score(true, pred)            if len(np.unique(true)) > 1 else np.nan
    auprc     = average_precision_score(true, pred)  if len(np.unique(true)) > 1 else np.nan
    return sens, spec, precision, f1, precision, npv, auroc, auprc


def bootstrap_ci(metric_func, true, pred, threshold, n_resamples=100):
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


def compute_optimal_threshold(true, pred):
    thresholds        = np.linspace(0, 1, 100)
    best_f1, best_thr = 0, 0.5
    for t in thresholds:
        pb   = (pred >= t).astype(int)
        tp   = np.sum((true == 1) & (pb == 1))
        fp   = np.sum((true == 0) & (pb == 1))
        fn   = np.sum((true == 1) & (pb == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return best_thr


def compute_macro_auroc(all_gt, all_pred, n_classes):
    scores = []
    for i in range(n_classes):
        if len(np.unique(all_gt[:, i])) > 1:
            scores.append(roc_auc_score(all_gt[:, i], all_pred[:, i]))
    return (np.mean(scores) if scores else 0.0), len(scores)


# =============================================================================
# ROC CURVE PLOT
# =============================================================================
def plot_roc_curves(all_gt, all_pred, labels, n_classes, split_name, saved_dir):
    valid_indices = [i for i in range(n_classes) if len(np.unique(all_gt[:, i])) > 1]
    n_valid       = len(valid_indices)

    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    cmap_a = cm.get_cmap("tab20",  20)
    cmap_b = cm.get_cmap("tab20b", 20)
    def get_color(k):
        return cmap_a(k % 20) if k < 20 else cmap_b((k - 20) % 20)

    mean_fpr    = np.linspace(0, 1, 500)
    tprs_interp = []
    class_lines = []

    for colour_idx, i in enumerate(valid_indices):
        true = all_gt[:, i]
        pred = all_pred[:, i]
        fpr, tpr, _ = roc_curve(true, pred)
        roc_auc_val = auc(fpr, tpr)

        tpr_i    = np.interp(mean_fpr, fpr, tpr); tpr_i[0] = 0.0
        tprs_interp.append(tpr_i)

        line, = ax.plot(fpr, tpr, color=get_color(colour_idx), lw=0.7, alpha=0.40)
        class_lines.append((line, f"{labels[i]}  (AUC={roc_auc_val:.2f})", roc_auc_val))

    # macro-average + std band
    mean_tpr      = np.mean(tprs_interp, axis=0); mean_tpr[0] = 0.0
    std_tpr       = np.std(tprs_interp,  axis=0)
    macro_auc_val = auc(mean_fpr, mean_tpr)

    macro_line, = ax.plot(
        mean_fpr, mean_tpr,
        color="#f0c040", lw=2.8, ls="--", zorder=10,
        label=f"Macro-average  (AUC={macro_auc_val:.3f})"
    )
    ax.fill_between(
        mean_fpr,
        np.clip(mean_tpr - std_tpr, 0, 1),
        np.clip(mean_tpr + std_tpr, 0, 1),
        color="#f0c040", alpha=0.10, zorder=9
    )

    # micro-average
    gt_flat   = all_gt[:,   valid_indices].ravel()
    pred_flat = all_pred[:, valid_indices].ravel()
    fpr_m, tpr_m, _ = roc_curve(gt_flat, pred_flat)
    micro_auc_val   = auc(fpr_m, tpr_m)

    micro_line, = ax.plot(
        fpr_m, tpr_m,
        color="#40e0d0", lw=2.8, ls="-.", zorder=10,
        label=f"Micro-average  (AUC={micro_auc_val:.3f})"
    )

    chance_line, = ax.plot(
        [0, 1], [0, 1],
        color="#666666", lw=1.0, ls=":", zorder=5,
        label="Chance  (AUC=0.500)"
    )

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

    std_patch = Patch(facecolor="#f0c040", alpha=0.20, label="Macro ± 1 std")
    summary_h = [macro_line, std_patch, micro_line, chance_line]
    summary_l = [
        f"Macro-average  (AUC={macro_auc_val:.3f})",
        "Macro ± 1 std",
        f"Micro-average  (AUC={micro_auc_val:.3f})",
        "Chance  (AUC=0.500)"
    ]
    legend_summary = ax.legend(
        summary_h, summary_l, loc="lower right", fontsize=10,
        framealpha=0.75, facecolor="#141920", edgecolor="#444444", labelcolor="#eeeeee"
    )
    ax.add_artist(legend_summary)

    class_sorted = sorted(class_lines, key=lambda x: x[2], reverse=True)
    ncol         = 2 if n_valid > 30 else 1
    ax.legend(
        [x[0] for x in class_sorted], [x[1] for x in class_sorted],
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
# FULL EVALUATION
# =============================================================================
def full_evaluation(all_gt, all_pred, labels, n_classes, split_name, saved_dir, n_resamples=100):
    logger.info(f"\nComputing full evaluation: {split_name}")
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

        threshold                                     = compute_optimal_threshold(true, pred)
        sens, spec, prec, f1, ppv, npv, auroc, auprc = calculate_performance_metrics(true, pred, threshold)

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
    skipped_df = pd.DataFrame(skipped) if skipped else pd.DataFrame(columns=["Label", "n_pos", "reason"])

    out_path     = os.path.join(saved_dir, f"{split_name}_results.csv")
    skipped_path = os.path.join(saved_dir, f"{split_name}_skipped_labels.csv")
    results_df.to_csv(out_path,     index=False)
    skipped_df.to_csv(skipped_path, index=False)

    macro_auroc = results_df["AUROC"].dropna().mean()

    logger.info(f"\n{'='*72}")
    logger.info(f"[{split_name}] PER-CLASS AUROC  ({len(results_df)} valid / {n_classes} total)")
    logger.info(f"{'='*72}")
    logger.info(f"  {'Label':<52} {'n_pos':>6}  {'AUROC':>6}  95% CI")
    logger.info(f"  {'-'*70}")
    for _, row in results_df.iterrows():
        ci = row["AUROC_CI"]
        logger.info(
            f"  {row['Label']:<52} {int(row['n_pos']):>6}  {row['AUROC']:>6.3f}"
            f"  ({ci[0]:.3f}, {ci[1]:.3f})"
        )

    if len(skipped_df) > 0:
        logger.info(f"\n{'='*72}")
        logger.info(f"[{split_name}] SKIPPED ({len(skipped_df)}) — all-zero labels, AUROC undefined")
        logger.info(f"{'='*72}")
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

    return results_df, macro_auroc


# =============================================================================
# INFERENCE
# =============================================================================
def run_inference(model, dataloader, device):
    model.eval()
    all_gt, all_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Inference", leave=False):
            x, y = x.to(device), y.to(device)
            all_gt.append(y.cpu().numpy())
            all_pred.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(all_gt), np.concatenate(all_pred)


# =============================================================================
# BUILD PRUNER
# =============================================================================
def build_pruner(model, example_inputs, pruning_ratio, n_classes,
                 method, global_pruning, isomorphic, round_to):

    ignored_layers = [
        m for m in model.modules()
        if isinstance(m, nn.Linear) and m.out_features == n_classes
    ]
    logger.info(f"  Ignoring {len(ignored_layers)} final classifier layer(s)")

    if method == "magnitude":
        importance   = tp.importance.GroupMagnitudeImportance(p=2)
        pruner_class = tp.pruner.MagnitudePruner
    elif method == "random":
        importance   = tp.importance.RandomImportance()
        pruner_class = tp.pruner.MagnitudePruner
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'magnitude' or 'random'.")

    kwargs = dict(
        model           = model,
        example_inputs  = example_inputs,
        importance      = importance,
        iterative_steps = 1,
        pruning_ratio   = pruning_ratio,
        global_pruning  = global_pruning,
        ignored_layers  = ignored_layers,
        round_to        = round_to,
    )
    if isomorphic:
        kwargs["isomorphic"] = True

    return pruner_class(**kwargs)


# =============================================================================
# MAIN
# =============================================================================
def main():

    # ── load data ─────────────────────────────────────────────────────────────
    logger.info("Loading dataset...")
    ecg_df = pd.read_csv(csv_path)

    def parse_label(val):
        if isinstance(val, list):
            return np.array(val, dtype=np.float32)
        val = str(val).strip()
        if "," not in val:
            val = val.replace("[", "").replace("]", "")
            return np.array([int(x) for x in val.split()], dtype=np.float32)
        return np.array(json.loads(val), dtype=np.float32)

    ecg_df["label"] = ecg_df["label"].apply(parse_label)

    valid_mask = ecg_df["label"].apply(lambda x: x.sum() > 0)
    ecg_df     = ecg_df[valid_mask].reset_index(drop=True)
    logger.info(f"Samples with at least one active label: {len(ecg_df)}")

    # ── dataloader ────────────────────────────────────────────────────────────
    full_dataset = ECG_Dataset(ecg_path=ecg_path, df=ecg_df, target_fs=target_fs)
    full_loader  = DataLoader(
        full_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )
    logger.info(f"Total samples: {len(full_dataset)} | Batches: {len(full_loader)}")

    sample_x, _    = full_dataset[0]
    example_inputs = torch.randn(1, *sample_x.shape).to(device)

    # ── baseline: fine-tuned model BEFORE any pruning ─────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE — fine-tuned model (no pruning)")
    logger.info("=" * 60)

    baseline_model = load_fresh_model(pth, n_classes, device)

    # sanity-check: macro AUC must be well above chance
    logger.info("Running sanity-check inference to verify weights loaded correctly...")
    base_gt, base_pred = run_inference(baseline_model, full_loader, device)
    sanity_macro, sanity_n = compute_macro_auroc(base_gt, base_pred, n_classes)
    logger.info(f"Sanity-check Macro AUROC: {sanity_macro:.4f} over {sanity_n} valid labels")

    if sanity_macro < 0.55:
        raise RuntimeError(
            f"\nSanity check FAILED: macro AUROC = {sanity_macro:.4f} (expected >> 0.5).\n"
            "The fine-tuned weights did not load correctly.\n"
            "Verify that 'pth' points to an ft.py checkpoint, not the original pretrained model."
        )
    logger.info("Sanity check PASSED — fine-tuned weights loaded correctly.")

    base_macs, base_params = tp.utils.count_ops_and_params(baseline_model, example_inputs)
    logger.info(f"Baseline MACs  : {base_macs  / 1e9:.4f} G")
    logger.info(f"Baseline Params: {base_params / 1e6:.4f} M")

    # reuse already-computed predictions as baseline evaluation
    _, baseline_macro_auroc = full_evaluation(
        base_gt, base_pred, labels, n_classes,
        split_name  = "baseline",
        saved_dir   = saved_dir,
        n_resamples = 100
    )

    # save baseline arrays for downstream comparison plots
    np.save(os.path.join(saved_dir, "baseline_gt.npy"),   base_gt)
    np.save(os.path.join(saved_dir, "baseline_pred.npy"), base_pred)
    logger.info(f"Baseline gt/pred saved to {saved_dir}")

    summary_rows = [{
        "pruning_ratio"   : 0.0,
        "method"          : "none",
        "macs_G"          : round(base_macs   / 1e9, 4),
        "params_M"        : round(base_params / 1e6, 4),
        "macs_reduction"  : 0.0,
        "params_reduction": 0.0,
        "macro_auroc"     : round(baseline_macro_auroc, 4),
        "auroc_drop"      : 0.0,
    }]

    # ── pruning loop ──────────────────────────────────────────────────────────
    for ratio in PRUNING_RATIOS:
        logger.info("\n" + "=" * 60)
        logger.info(f"PRUNING RATIO {ratio}  |  method: {PRUNING_METHOD}")
        logger.info("=" * 60)

        # independent fresh copy of the fine-tuned model for each ratio
        model = load_fresh_model(pth, n_classes, device)

        pruner = build_pruner(
            model          = model,
            example_inputs = example_inputs,
            pruning_ratio  = ratio,
            n_classes      = n_classes,
            method         = PRUNING_METHOD,
            global_pruning = GLOBAL_PRUNING,
            isomorphic     = ISOMORPHIC,
            round_to       = ROUND_TO,
        )

        logger.info("Applying single-shot pruning step...")
        pruner.step()

        pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
        macs_red   = (1 - pruned_macs   / base_macs)   * 100
        params_red = (1 - pruned_params / base_params)  * 100

        logger.info(f"  MACs  : {base_macs/1e9:.4f} G → {pruned_macs/1e9:.4f} G  ({macs_red:.2f}% reduced)")
        logger.info(f"  Params: {base_params/1e6:.4f} M → {pruned_params/1e6:.4f} M  ({params_red:.2f}% reduced)")

        gt, pred   = run_inference(model, full_loader, device)
        split_name = f"pruned_{PRUNING_METHOD}_ratio{int(ratio*100):02d}"

        results_df, macro_auroc = full_evaluation(
            gt, pred, labels, n_classes,
            split_name  = split_name,
            saved_dir   = saved_dir,
            n_resamples = 100
        )

        auroc_drop = baseline_macro_auroc - macro_auroc
        logger.info(
            f"  Macro AUROC: {baseline_macro_auroc:.4f} → {macro_auroc:.4f}  "
            f"(Δ {-auroc_drop:+.4f} vs fine-tuned baseline)"
        )

        ckpt_path = os.path.join(
            saved_dir,
            f"pruned_{PRUNING_METHOD}_ratio{ratio}_auroc{macro_auroc:.4f}.pth"
        )
        model.zero_grad()
        torch.save({
            "model"           : model,
            "pruning_ratio"   : ratio,
            "pruning_method"  : PRUNING_METHOD,
            "macro_auroc"     : macro_auroc,
            "baseline_auroc"  : baseline_macro_auroc,
            "macs"            : pruned_macs,
            "params"          : pruned_params,
            "baseline_macs"   : base_macs,
            "baseline_params" : base_params,
            "source_pth"      : pth,
        }, ckpt_path)
        logger.info(f"  Saved: {ckpt_path}")

        summary_rows.append({
            "pruning_ratio"   : ratio,
            "method"          : PRUNING_METHOD,
            "macs_G"          : round(pruned_macs   / 1e9, 4),
            "params_M"        : round(pruned_params / 1e6, 4),
            "macs_reduction"  : round(macs_red,   2),
            "params_reduction": round(params_red, 2),
            "macro_auroc"     : round(macro_auroc, 4),
            "auroc_drop"      : round(auroc_drop,  4),
        })

    # ── final summary ─────────────────────────────────────────────────────────
    summary_df   = pd.DataFrame(summary_rows)
    summary_path = os.path.join(saved_dir, "pruning_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    logger.info("\n" + "=" * 60)
    logger.info("PRUNING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"\n{summary_df.to_string(index=False)}")
    logger.info(f"\nSummary saved: {summary_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()