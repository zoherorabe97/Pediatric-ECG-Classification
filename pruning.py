# =============================================================================
# Zero-Shot Pruning + Evaluation on Child ECG Dataset
# No training data required — prune with random/magnitude importance,
# then evaluate the pruned model on the full ecg_df.
# =============================================================================

import os
import sys
import json
import copy
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score

import torch_pruning as tp
import wfdb

from net1d import Net1D

# =============================================================================
# CONFIG  —  edit these paths / settings
# =============================================================================
gpu_id        = 0
batch_size    = 32

# Pruning settings
PRUNING_RATIOS   = [0.1, 0.2, 0.3, 0.4, 0.5]   # evaluate each ratio independently
PRUNING_METHOD   = "magnitude"                    # "magnitude" | "random"
#   ↑ Taylor needs gradients (i.e. training data), so magnitude/random only here
GLOBAL_PRUNING   = False
ISOMORPHIC       = True                           # recommended for transformer-style blocks
ROUND_TO         = 8                              # channel multiple for GPU efficiency

# Paths
ecg_path   = "C:/Users/zoorab/Desktop/zoher/University/Projects/Zhengzhou_ECG/Child_ecg/"
csv_path   = "./ecg_df.csv"
pth        = "./12_lead_ECGFounder.pth"
tasks_path = "./tasks.txt"
saved_dir  = "./res/zeroshot_pruning/"
log_file   = "logging/zeroshot_pruning.log"

target_fs  = 5000

os.makedirs(saved_dir,              exist_ok=True)
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
        x_old     = np.linspace(0, t, num=ts.shape[1],   endpoint=True)
        x_new     = np.linspace(0, t, num=int(fs_out),   endpoint=True)
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
# MODEL LOADER  (fresh copy each pruning ratio so ratios are independent)
# =============================================================================
def load_fresh_model(pth, n_classes, device):
    model = Net1D(
        in_channels      = 12,
        base_filters     = 64,
        ratio            = 1,
        filter_list      = [64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list    = [2, 2, 2, 3, 3, 4, 4],
        kernel_size      = 16,
        stride           = 2,
        groups_width     = 16,
        verbose          = False,
        use_bn           = False,
        use_do           = False,
        n_classes        = n_classes
    )
    checkpoint = torch.load(pth, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]
    log        = model.load_state_dict(state_dict, strict=False)
    logger.info(
        f"  Pretrained weights loaded | "
        f"missing: {len(log.missing_keys)} | "
        f"unexpected: {len(log.unexpected_keys)}"
    )
    return model.to(device)


# =============================================================================
# METRICS  (identical to ft.py)
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


def full_evaluation(all_gt, all_pred, labels, n_classes, split_name, saved_dir, n_resamples=100):
    logger.info(f"\nComputing full evaluation: {split_name}")
    results = []

    for i, label in enumerate(tqdm(labels, desc=f"Metrics [{split_name}]")):
        true  = all_gt[:, i]
        pred  = all_pred[:, i]
        n_pos = int(true.sum())
        n_neg = int((1 - true).sum())

        if len(np.unique(true)) < 2:
            continue

        threshold                                          = compute_optimal_threshold(true, pred)
        sens, spec, prec, f1, ppv, npv, auroc, auprc      = calculate_performance_metrics(true, pred, threshold)

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
    out_path   = os.path.join(saved_dir, f"{split_name}_results.csv")
    results_df.to_csv(out_path, index=False)

    macro_auroc = results_df["AUROC"].dropna().mean()
    logger.info(f"[{split_name}] Valid labels : {len(results_df)}/{n_classes}")
    logger.info(f"[{split_name}] Macro AUROC  : {macro_auroc:.4f}")
    logger.info(f"[{split_name}] Results saved: {out_path}")
    logger.info(
        f"\n{results_df[['Label','n_pos','AUROC','F1','Sensitivity','Specificity','PPV','NPV']].to_string(index=False)}"
    )

    return results_df, macro_auroc


# =============================================================================
# INFERENCE  (no training, no grad updates)
# =============================================================================
def run_inference(model, dataloader, device):
    model.eval()
    all_gt, all_pred = [], []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Inference", leave=False):
            x, y = x.to(device), y.to(device)
            pred = torch.sigmoid(model(x))
            all_gt.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    return np.concatenate(all_gt), np.concatenate(all_pred)


# =============================================================================
# BUILD PRUNER  (magnitude or random — both data-free)
# =============================================================================
def build_pruner(model, example_inputs, pruning_ratio, n_classes,
                 method, global_pruning, isomorphic, round_to):

    # Never prune the final classification head
    ignored_layers = [
        m for m in model.modules()
        if isinstance(m, nn.Linear) and m.out_features == n_classes
    ]
    logger.info(f"  Ignoring {len(ignored_layers)} final classifier layer(s)")

    if method == "magnitude":
        importance    = tp.importance.GroupMagnitudeImportance(p=2)
        pruner_class  = tp.pruner.MagnitudePruner
    elif method == "random":
        importance    = tp.importance.RandomImportance()
        pruner_class  = tp.pruner.MagnitudePruner
    else:
        raise ValueError(f"Unsupported zero-shot pruning method: {method}. Use 'magnitude' or 'random'.")

    kwargs = dict(
        model           = model,
        example_inputs  = example_inputs,
        importance      = importance,
        iterative_steps = 1,          # single-shot for zero-shot evaluation
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
    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("Loading ecg_df ...")
    ecg_df = pd.read_csv(csv_path)

    # parse label column (JSON string → list → numpy array)
    def parse_label(val):
        if isinstance(val, list):
            return np.array(val, dtype=np.float32)
        val = str(val).strip()
        if ',' not in val:                        # numpy repr without commas
            val = val.replace('[', '').replace(']', '')
            return np.array([int(x) for x in val.split()], dtype=np.float32)
        return np.array(json.loads(val), dtype=np.float32)

    ecg_df["label"] = ecg_df["label"].apply(parse_label)

    # drop rows with all-zero labels (unencodeable / fully unmapped)
    valid_mask = ecg_df["label"].apply(lambda x: x.sum() > 0)
    ecg_df     = ecg_df[valid_mask].reset_index(drop=True)
    logger.info(f"Samples with at least one active label: {len(ecg_df)}")

    # ── Build dataloader (entire dataset — zero-shot, no splits needed) ────────
    full_dataset = ECG_Dataset(ecg_path=ecg_path, df=ecg_df, target_fs=target_fs)
    full_loader  = DataLoader(
        full_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True
    )
    logger.info(f"Total samples: {len(full_dataset)} | Batches: {len(full_loader)}")

    # ── Example input shape for dependency graph ───────────────────────────────
    sample_x, _ = full_dataset[0]
    example_inputs = torch.randn(1, *sample_x.shape).to(device)

    # ── Baseline evaluation (unpruned model) ──────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE — unpruned model")
    logger.info("=" * 60)

    baseline_model           = load_fresh_model(pth, n_classes, device)
    base_macs, base_params   = tp.utils.count_ops_and_params(baseline_model, example_inputs)
    logger.info(f"Baseline MACs  : {base_macs  / 1e9:.4f} G")
    logger.info(f"Baseline Params: {base_params / 1e6:.4f} M")

    base_gt, base_pred = run_inference(baseline_model, full_loader, device)
    baseline_results, baseline_macro_auroc = full_evaluation(
        base_gt, base_pred, labels, n_classes,
        split_name = "baseline",
        saved_dir  = saved_dir,
        n_resamples = 100
    )

    # ── Summary table (filled as we go) ───────────────────────────────────────
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

    # ── Iterate over pruning ratios ────────────────────────────────────────────
    for ratio in PRUNING_RATIOS:
        logger.info("\n" + "=" * 60)
        logger.info(f"PRUNING RATIO {ratio}  |  method: {PRUNING_METHOD}")
        logger.info("=" * 60)

        # fresh model for every ratio (independent experiments)
        model = load_fresh_model(pth, n_classes, device)

        # build pruner and apply single-shot pruning (no training)
        pruner = build_pruner(
            model          = model,
            example_inputs = example_inputs,
            pruning_ratio  = ratio,
            n_classes      = n_classes,
            method         = PRUNING_METHOD,
            global_pruning  = GLOBAL_PRUNING,
            isomorphic     = ISOMORPHIC,
            round_to       = ROUND_TO,
        )

        logger.info("Applying pruning step (zero-shot — no training data) ...")
        pruner.step()

        # model size after pruning
        pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
        macs_red   = (1 - pruned_macs   / base_macs)   * 100
        params_red = (1 - pruned_params / base_params)  * 100

        logger.info(f"  MACs  : {base_macs/1e9:.4f} G → {pruned_macs/1e9:.4f} G  ({macs_red:.2f}% reduced)")
        logger.info(f"  Params: {base_params/1e6:.4f} M → {pruned_params/1e6:.4f} M  ({params_red:.2f}% reduced)")

        # inference on full dataset
        gt, pred = run_inference(model, full_loader, device)

        split_name = f"pruned_{PRUNING_METHOD}_ratio{int(ratio*100):02d}"
        results_df, macro_auroc = full_evaluation(
            gt, pred, labels, n_classes,
            split_name  = split_name,
            saved_dir   = saved_dir,
            n_resamples = 100
        )

        auroc_drop = baseline_macro_auroc - macro_auroc
        logger.info(f"  Macro AUROC: {baseline_macro_auroc:.4f} → {macro_auroc:.4f}  (Δ {-auroc_drop:+.4f})")

        # save pruned model
        ckpt_path = os.path.join(
            saved_dir,
            f"pruned_{PRUNING_METHOD}_ratio{ratio}_auroc{macro_auroc:.4f}.pth"
        )
        model.zero_grad()
        torch.save({
            "model"            : model,
            "pruning_ratio"    : ratio,
            "pruning_method"   : PRUNING_METHOD,
            "macro_auroc"      : macro_auroc,
            "baseline_auroc"   : baseline_macro_auroc,
            "macs"             : pruned_macs,
            "params"           : pruned_params,
            "baseline_macs"    : base_macs,
            "baseline_params"  : base_params,
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

    # ── Save and print summary ─────────────────────────────────────────────────
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