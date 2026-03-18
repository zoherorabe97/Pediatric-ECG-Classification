# =============================================================================
# Fine-tuning ECGFounder on Child ECG Dataset
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import logging
import json
from tqdm import tqdm
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

import wfdb
from net1d import Net1D

# =============================================================================
# CONFIG
# =============================================================================
LINEAR_PROBE  = False         # True = freeze backbone | False = full fine-tuning
gpu_id        = 0
batch_size    = 32
lr            = 1e-4
lp_lr         = 1e-3
weight_decay  = 1e-5
Epochs        = 10
early_stop_lr = 1e-5
target_fs     = 5000

ecg_path   = "C:/Users/zoorab/Desktop/zoher/University/Projects/Zhengzhou_ECG/Child_ecg/"
csv_path   = "./ecg_df.csv"
saved_dir  = "./res/finetune/"
pth        = "./12_lead_ECGFounder.pth"
tasks_path = "./tasks.txt"

os.makedirs(saved_dir, exist_ok=True)
os.makedirs("logging", exist_ok=True)

# =============================================================================
# LOGGING  (utf-8 on both file and console to avoid Windows cp1252 crash)
# =============================================================================
log_file = "logging/child_ecg_ft.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# file handler - utf-8
fh = logging.FileHandler(log_file, encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)

# console handler - utf-8
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
    labels = [line.strip() for line in f]
n_classes = len(labels)
logger.info(f"Number of classes: {n_classes}")

# =============================================================================
# DATASET CLASS
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
            data, meta = wfdb.rdsamp(file_path)
            data       = np.transpose(data, (1, 0))
            data       = self.z_score_normalization(data)
            data       = self.resample_unequal(data, sample_rate, self.target_fs)
            signal     = torch.FloatTensor(data)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e} -- returning zeros")
            signal = torch.zeros((12, self.target_fs))

        return signal, label


# =============================================================================
# MODEL
# =============================================================================
class ft_ChildECG(nn.Module):
    def __init__(self, device, pth, n_classes, linear_probe=False):
        super(ft_ChildECG, self).__init__()

        self.backbone = Net1D(
            in_channels=12,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            use_bn=False,
            use_do=False,
            n_classes=n_classes
        )

        checkpoint = torch.load(pth, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"]
        log        = self.backbone.load_state_dict(state_dict, strict=False)
        logger.info(f"Pretrained weights loaded | missing: {len(log.missing_keys)} | unexpected: {len(log.unexpected_keys)}")

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
    """Find threshold that maximises F1."""
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


def full_evaluation(all_gt, all_pred, labels, n_classes, split_name, saved_dir, n_resamples=100):
    """Compute per-class metrics with bootstrap CI and save to CSV."""
    logger.info(f"\nComputing full evaluation for: {split_name}")
    results = []

    for i, label in enumerate(tqdm(labels, desc=f"Metrics [{split_name}]")):
        true  = all_gt[:, i]
        pred  = all_pred[:, i]
        n_pos = int(true.sum())
        n_neg = int((1 - true).sum())

        if len(np.unique(true)) < 2:
            continue

        threshold = compute_optimal_threshold(true, pred)
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
            "AUROC"          : round(auroc, 3) if not np.isnan(auroc) else np.nan,
            "AUROC_CI"       : auroc_ci,
            "AUPRC"          : round(auprc, 3) if not np.isnan(auprc) else np.nan,
            "AUPRC_CI"       : auprc_ci,
        })

    results_df = pd.DataFrame(results).sort_values("AUROC", ascending=False)
    out_path   = os.path.join(saved_dir, f"{split_name}_results.csv")
    results_df.to_csv(out_path, index=False)

    macro_auroc = results_df["AUROC"].dropna().mean()
    logger.info(f"[{split_name}] Valid labels : {len(results_df)}/{n_classes}")
    logger.info(f"[{split_name}] Macro AUROC  : {macro_auroc:.4f}")
    logger.info(f"[{split_name}] Results saved: {out_path}")
    logger.info(f"\n{results_df[['Label','n_pos','AUROC','F1','Sensitivity','Specificity','PPV','NPV']].to_string(index=False)}")

    return results_df


# =============================================================================
# LABEL STATISTICS LOGGER
# =============================================================================
def log_label_stats(df, split_name, labels, n_classes):
    label_matrix  = np.array(df["label"].tolist())       # (N, 150)
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
        prevalence = cnt / len(df) * 100
        logger.info(f"  {lbl:<50} {cnt:<8} {prevalence:.1f}%")
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

    # ---------- DATA ----------
    ecg_df = pd.read_csv(csv_path)
    if isinstance(ecg_df["label"].iloc[0], str):
        ecg_df["label"] = ecg_df["label"].apply(json.loads)

    train_df, val_df, test_df = prepare_splits(ecg_df)

    # log label statistics
    log_label_stats(train_df, "TRAIN", labels, n_classes)
    log_label_stats(val_df,   "VAL",   labels, n_classes)
    log_label_stats(test_df,  "TEST",  labels, n_classes)

    train_dataset = ECG_Dataset(ecg_path=ecg_path, df=train_df)
    val_dataset   = ECG_Dataset(ecg_path=ecg_path, df=val_df)
    test_dataset  = ECG_Dataset(ecg_path=ecg_path, df=test_df)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    valloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    testloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    logger.info(f"Train batches: {len(trainloader)} | Val batches: {len(valloader)} | Test batches: {len(testloader)}")

    # ---------- MODEL ----------
    model = ft_ChildECG(
        device=device,
        pth=pth,
        n_classes=n_classes,
        linear_probe=LINEAR_PROBE
    ).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Mode            : {'Linear Probe' if LINEAR_PROBE else 'Full Fine-tuning'}")
    logger.info(f"Total params    : {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")

    # ---------- OPTIMIZER ----------
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lp_lr if LINEAR_PROBE else lr,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1, mode="max")
    criterion = nn.BCEWithLogitsLoss()

    # ---------- TRAINING LOOP ----------
    best_val_auroc = 0.0
    global_step    = 0

    for epoch in range(Epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{Epochs}")

        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        for x, y in tqdm(trainloader, desc=f"Epoch {epoch+1} Train"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss  += loss.item()
            global_step += 1

        avg_train_loss = train_loss / len(trainloader)
        logger.info(f"Train Loss: {avg_train_loss:.4f}")

        # ---- VALIDATE ----
        model.eval()
        all_gt, all_pred, val_loss = [], [], 0.0

        with torch.no_grad():
            for x, y in tqdm(valloader, desc=f"Epoch {epoch+1} Val", leave=False):
                x, y   = x.to(device), y.to(device)
                logits = model(x)
                loss   = criterion(logits, y)
                val_loss  += loss.item()
                all_pred.append(torch.sigmoid(logits).cpu().numpy())
                all_gt.append(y.cpu().numpy())

        all_gt   = np.concatenate(all_gt)
        all_pred = np.concatenate(all_pred)

        val_macro_auroc, valid_labels = compute_macro_auroc(all_gt, all_pred, n_classes)
        avg_val_loss = val_loss / len(valloader)

        logger.info(f"Val Loss: {avg_val_loss:.4f} | Val Macro AUROC: {val_macro_auroc:.4f} | Valid labels: {valid_labels}/{n_classes}")

        # lr scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_macro_auroc)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != current_lr:
            logger.info(f"LR reduced: {current_lr:.6f} -> {new_lr:.6f}")

        # ---- SAVE BEST ----
        if val_macro_auroc > best_val_auroc:
            best_val_auroc = val_macro_auroc
            mode_suffix    = "linear_probe" if LINEAR_PROBE else "full_ft"
            ckpt_name      = f"child_ecg_{mode_suffix}_epoch{epoch+1}_auroc{val_macro_auroc:.4f}.pth"
            ckpt_path      = os.path.join(saved_dir, ckpt_name)
            torch.save({
                "epoch"      : epoch + 1,
                "step"       : global_step,
                "state_dict" : model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "scheduler"  : scheduler.state_dict(),
                "val_auroc"  : val_macro_auroc,
                "config": {
                    "linear_probe": LINEAR_PROBE,
                    "n_classes"   : n_classes,
                    "batch_size"  : batch_size,
                    "lr"          : lp_lr if LINEAR_PROBE else lr,
                }
            }, ckpt_path)
            logger.info(f"[SAVED] Best model: {ckpt_name}")

            # full per-class metrics at best val epoch
            full_evaluation(all_gt, all_pred, labels, n_classes,
                            split_name=f"val_epoch{epoch+1}",
                            saved_dir=saved_dir, n_resamples=100)

        # ---- EARLY STOPPING ----
        if optimizer.param_groups[0]["lr"] < early_stop_lr:
            logger.info(f"Early stopping: LR {optimizer.param_groups[0]['lr']:.2e} < {early_stop_lr:.2e}")
            break

    logger.info(f"\nTraining complete | Best Val Macro AUROC: {best_val_auroc:.4f}")

    # ---------- FINAL TEST EVALUATION ----------
    logger.info("\nRunning final test evaluation...")
    model.eval()
    all_gt, all_pred = [], []

    with torch.no_grad():
        for x, y in tqdm(testloader, desc="Final Test"):
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            all_pred.append(torch.sigmoid(logits).cpu().numpy())
            all_gt.append(y.cpu().numpy())

    all_gt   = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    full_evaluation(
        all_gt, all_pred, labels, n_classes,
        split_name="test", saved_dir=saved_dir, n_resamples=100
    )


if __name__ == "__main__":
    main()