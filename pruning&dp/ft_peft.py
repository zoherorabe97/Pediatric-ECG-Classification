# =============================================================================
# Manual PEFT Fine-tuning of ECGFounder on Child ECG Dataset
# Supports three PEFT modes selectable at launch via --mode:
#   lora      — Low-Rank Adaptation on Linear layers only  (default)
#   ia3       — IA³ element-wise scale adapters (fewest params)
#   adalora   — AdaLoRA: adaptive-rank LoRA (auto-prunes via SVD)
#
# FIX vs ft_peft_v1.py:
# ----------------------
# The PEFT library's get_peft_model() wraps the backbone in a
# PeftModelForFeatureExtraction which, during forward(), injects NLP-style
# keyword arguments (input_ids, attention_mask, …) that Net1D.forward() does
# not accept → TypeError crash at first batch.
#
# Solution: inject adapters MANUALLY by replacing target nn.Linear modules
# with thin wrapper classes (LoRALinear, IA3Linear, AdaLoRALinear) that keep
# Net1D.forward() completely unmodified. No PEFT library dependency at all.
#
# Usage:
#   python ft_peft.py --mode lora
#   python ft_peft.py --mode ia3
#   python ft_peft.py --mode adalora
#   python ft_peft.py --mode lora --lora_r 16 --lora_alpha 32
# =============================================================================

import os
import sys
import math
import copy
import argparse
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, auc
)

import wfdb
from net1d import Net1D

# =============================================================================
# CLI
# =============================================================================
_parser = argparse.ArgumentParser(
    description="Manual PEFT fine-tuning of ECGFounder on Child ECG dataset"
)
_parser.add_argument(
    "--mode",
    choices=["lora", "ia3", "adalora"],
    default="lora",
    help=(
        "PEFT method:\n"
        "  lora    — Low-Rank Adaptation (default)\n"
        "  ia3     — IA³ element-wise scale vectors\n"
        "  adalora — AdaLoRA: SVD-based adaptive-rank LoRA\n"
    ),
)
_parser.add_argument("--lora_r",        type=int,   default=8,    help="LoRA rank r (default: 8)")
_parser.add_argument("--lora_alpha",    type=float, default=16.0, help="LoRA scaling alpha (default: 16)")
_parser.add_argument("--lora_dropout",  type=float, default=0.05, help="LoRA dropout (default: 0.05)")
_parser.add_argument("--adalora_r",     type=int,   default=8,    help="AdaLoRA initial rank (default: 8)")
_parser.add_argument("--adalora_r_min", type=int,   default=4,    help="AdaLoRA minimum rank (default: 4)")
_args = _parser.parse_args()

MODE = _args.mode

# =============================================================================
# CONFIG
# =============================================================================
gpu_id        = 0
batch_size    = 32
weight_decay  = 1e-5
Epochs        = 50
early_stop_lr = 1e-5

LR_MAP = {
    "lora"   : 5e-4,
    "ia3"    : 5e-4,
    "adalora": 5e-4,
}
ACTIVE_LR = LR_MAP[MODE]

# ── Early Stopping ────────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE  = 10
EARLY_STOP_MIN_DELTA = 1e-4

# ── PEFT hyper-params (from CLI) ──────────────────────────────────────────────
LORA_R        = _args.lora_r
LORA_ALPHA    = _args.lora_alpha
LORA_DROPOUT  = _args.lora_dropout
ADALORA_R     = _args.adalora_r
ADALORA_R_MIN = _args.adalora_r_min

# AdaLoRA pruning schedule: prune every N steps, total T steps
ADALORA_PRUNE_EVERY = 100   # steps between pruning passes
ADALORA_WARMUP_STEPS = 200  # don't prune during warmup

# ── Paths ─────────────────────────────────────────────────────────────────────
target_fs  = 5000
ecg_path   = "C:/Users/zoorab/Desktop/zoher/University/Projects/Zhengzhou_ECG/Child_ecg/"
csv_path   = "./ecg_with_exact_match.csv"
pth        = "./12_lead_ECGFounder.pth"
tasks_path = "./tasks.txt"

saved_dir = f"./res/peft_{MODE}/"

ZEROSHOT_GT_PATH   = "./res/zeroshot_gt.npy"
ZEROSHOT_PRED_PATH = "./res/zeroshot_pred.npy"

os.makedirs(saved_dir, exist_ok=True)
os.makedirs("./res/",  exist_ok=True)
os.makedirs("logging", exist_ok=True)

# =============================================================================
# LOGGING
# =============================================================================
log_file = f"logging/child_ecg_peft_{MODE}.log"
logger   = logging.getLogger(__name__)
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
logger.info(f"Using device : {device}")
logger.info(f"PEFT mode    : {MODE}")

# =============================================================================
# LOAD TASKS
# =============================================================================
with open(tasks_path, "r") as f:
    labels = [line.strip() for line in f if line.strip()]
n_classes = len(labels)
logger.info(f"Number of classes: {n_classes}")

# =============================================================================
# MANUAL ADAPTER LAYERS
# =============================================================================

class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with two low-rank matrices A (r × d_in) and
    B (d_out × r).

    Forward:  W_frozen(x) + (alpha / r) * B( A( dropout(x) ) )

    Only lora_A and lora_B are trained; the original weight is never updated.
    Initialisation follows the original LoRA paper:
        A ~ Kaiming-uniform,  B = 0  (so adapter output starts at zero).
    """
    def __init__(self, linear: nn.Linear, r: int = 8,
                 alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.r       = r
        self.scaling = alpha / r
        d_out, d_in  = linear.weight.shape

        # Keep frozen weight & bias as non-parameter attributes so that
        # optimizer.parameters() never picks them up.
        self.register_buffer("weight", linear.weight.data.clone())
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None

        # Trainable low-rank matrices
        self.lora_A  = nn.Parameter(torch.empty(r, d_in))
        self.lora_B  = nn.Parameter(torch.zeros(d_out, r))
        self.dropout = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base + self.scaling * lora

    def extra_repr(self):
        d_out, d_in = self.weight.shape
        return (f"in={d_in}, out={d_out}, r={self.r}, "
                f"scaling={self.scaling:.3f}")


class IA3Linear(nn.Module):
    """
    IA³ adapter for nn.Linear.

    Learns a single vector l (shape: d_out) that element-wise scales the
    output of the frozen linear layer:
        output = l  ⊙  W_frozen(x)

    Extremely lightweight: only d_out trainable scalars per layer.
    Initialised to ones so the adapter is identity at the start.
    """
    def __init__(self, linear: nn.Linear):
        super().__init__()
        d_out, _ = linear.weight.shape

        self.register_buffer("weight", linear.weight.data.clone())
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None

        # Trainable scale vector — init to 1 (identity)
        self.ia3_l = nn.Parameter(torch.ones(d_out))

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        return out * self.ia3_l          # broadcast over batch / sequence dims

    def extra_repr(self):
        return f"out={self.ia3_l.shape[0]}"


class AdaLoRALinear(nn.Module):
    """
    Simplified AdaLoRA adapter for nn.Linear.

    Parameterises the adapter as  P · Λ · Q  where:
        P  : (d_out × r)  — left singular vectors
        Λ  : (r,)         — singular values (the importance scores)
        Q  : (r × d_in)   — right singular vectors

    During training a budget-aware pruning step (called externally) zeroes out
    the smallest singular values to push the effective rank toward target_r.

    Forward:  W_frozen(x) + (alpha / r) * P · diag(Λ) · Q · dropout(x)
    """
    def __init__(self, linear: nn.Linear, r: int = 8,
                 alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.r       = r
        self.scaling = alpha / r
        d_out, d_in  = linear.weight.shape

        self.register_buffer("weight", linear.weight.data.clone())
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None

        # SVD-style decomposition: P (d_out × r), Λ (r,), Q (r × d_in)
        self.lora_P   = nn.Parameter(torch.empty(d_out, r))
        self.lora_diag = nn.Parameter(torch.ones(r))   # singular values
        self.lora_Q   = nn.Parameter(torch.empty(r, d_in))
        self.dropout  = nn.Dropout(p=dropout)

        nn.init.orthogonal_(self.lora_P)
        nn.init.orthogonal_(self.lora_Q)

        # Mask for pruned singular values (1 = active, 0 = pruned)
        self.register_buffer("sv_mask", torch.ones(r))

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        # Apply singular-value mask (stops gradient through pruned directions)
        diag = self.lora_diag * self.sv_mask
        # Compute low-rank adapter: P · diag(Λ) · Q
        adapter_weight = self.lora_P * diag.unsqueeze(0)  # (d_out, r)
        adapter_weight = adapter_weight @ self.lora_Q      # (d_out, d_in)
        lora = F.linear(self.dropout(x), adapter_weight)
        return base + self.scaling * lora

    def prune_to_rank(self, target_r: int):
        """Zero-out smallest |Λ_i| values to reach target_r active directions."""
        with torch.no_grad():
            abs_sv = self.lora_diag.abs()
            if target_r >= self.r:
                self.sv_mask.fill_(1.0)
                return
            # Keep the top-target_r singular values active
            threshold = abs_sv.topk(target_r).values.min()
            self.sv_mask.copy_((abs_sv >= threshold).float())

    def extra_repr(self):
        d_out = self.lora_P.shape[0]
        d_in  = self.lora_Q.shape[1]
        active = int(self.sv_mask.sum().item())
        return (f"in={d_in}, out={d_out}, r={self.r}, "
                f"active_rank={active}, scaling={self.scaling:.3f}")


# =============================================================================
# ADAPTER INJECTION HELPERS
# =============================================================================

# The two SE-attention linears inside every BasicBlock.
# We deliberately exclude `dense` (the head) because we unfreeze it fully —
# no need to add adapter overhead on a 1024→150 projection that is already
# re-trained from scratch.
TARGET_NAMES = {"se_fc1", "se_fc2"}


def _inject(model: nn.Module, make_adapter_fn, target_names: set) -> nn.Module:
    """
    Recursively walk the module tree. For every nn.Linear whose *attribute
    name* (last segment) is in target_names, replace it with the adapter
    returned by make_adapter_fn(original_linear).

    Named children are iterated over a copy of the list so that setattr
    during iteration is safe.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and name in target_names:
            setattr(model, name, make_adapter_fn(child))
        else:
            _inject(child, make_adapter_fn, target_names)
    return model


def inject_lora(model, r, alpha, dropout):
    return _inject(model, lambda lin: LoRALinear(lin, r=r, alpha=alpha, dropout=dropout), TARGET_NAMES)


def inject_ia3(model):
    return _inject(model, lambda lin: IA3Linear(lin), TARGET_NAMES)


def inject_adalora(model, r, alpha, dropout):
    return _inject(model, lambda lin: AdaLoRALinear(lin, r=r, alpha=alpha, dropout=dropout), TARGET_NAMES)


def collect_adalora_layers(model):
    """Return a list of all AdaLoRALinear modules in the model."""
    return [m for m in model.modules() if isinstance(m, AdaLoRALinear)]


def adalora_prune_step(model, target_r):
    """Call prune_to_rank on every AdaLoRALinear in the model."""
    for layer in collect_adalora_layers(model):
        layer.prune_to_rank(target_r)


def log_param_budget(model, mode):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    logger.info(f"\n{'='*60}")
    logger.info(f"PEFT mode       : {mode}")
    logger.info(f"Total params    : {total:,}")
    logger.info(f"Trainable params: {trainable:,}  ({100*trainable/total:.2f}%)")
    logger.info(f"Frozen params   : {frozen:,}  ({100*frozen/total:.2f}%)")
    logger.info(f"{'='*60}\n")


# =============================================================================
# MODEL BUILDER  (replaces build_peft_model from v1)
# =============================================================================

def build_model(device, pth, n_classes, mode):
    """
    1. Instantiate Net1D and load pretrained weights.
    2. Freeze ALL parameters.
    3. Inject lightweight adapter wrappers into SE-attention linears only.
       This keeps Net1D.forward() completely untouched — no PEFT wrapper,
       no keyword-argument clash.
    4. Unfreeze the classification head (dense) fully.
    """
    backbone = Net1D(
        in_channels=12, base_filters=64, ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16, stride=2, groups_width=16,
        verbose=False, use_bn=False, use_do=False, n_classes=n_classes
    )
    checkpoint = torch.load(pth, map_location=device, weights_only=False)
    log = backbone.load_state_dict(checkpoint["state_dict"], strict=False)
    logger.info(
        f"Pretrained weights loaded | "
        f"missing: {len(log.missing_keys)} | unexpected: {len(log.unexpected_keys)}"
    )

    # ── Step 1: freeze everything ──────────────────────────────────────────
    for p in backbone.parameters():
        p.requires_grad = False

    # ── Step 2: inject adapters into SE linears ────────────────────────────
    if mode == "lora":
        inject_lora(backbone, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
        logger.info(
            f"LoRA injected | r={LORA_R}, alpha={LORA_ALPHA}, "
            f"dropout={LORA_DROPOUT} | targets: {TARGET_NAMES}"
        )
    elif mode == "ia3":
        inject_ia3(backbone)
        logger.info(f"IA³ injected | targets: {TARGET_NAMES}")
    elif mode == "adalora":
        inject_adalora(backbone, r=ADALORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
        logger.info(
            f"AdaLoRA injected | r={ADALORA_R}, target_r={ADALORA_R_MIN}, "
            f"alpha={LORA_ALPHA}, dropout={LORA_DROPOUT} | targets: {TARGET_NAMES}"
        )

    # Adapter parameters are nn.Parameter inside the wrapper → already trainable.
    # The frozen weight/bias were stored as buffers → not in parameters() at all.

    # ── Step 3: unfreeze head fully ───────────────────────────────────────
    for p in backbone.dense.parameters():
        p.requires_grad = True
    logger.info("Head unfrozen: dense")

    log_param_budget(backbone, mode)
    return backbone.to(device)


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
                f"(best={self.best_score:.4f}, current={score:.4f}, delta={improvement:+.4f})"
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
# ZERO-SHOT BASELINE
# =============================================================================
def run_zeroshot_baseline(df, out_dir):
    logger.info("\nGenerating zero-shot baseline predictions...")

    dataset = ECG_Dataset(ecg_path=ecg_path, df=df)
    loader  = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )

    zs_backbone = Net1D(
        in_channels=12, base_filters=64, ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16, stride=2, groups_width=16,
        verbose=False, use_bn=False, use_do=False, n_classes=n_classes
    )
    checkpoint = torch.load(pth, map_location=device, weights_only=False)
    zs_backbone.load_state_dict(checkpoint["state_dict"], strict=False)
    zs_backbone = zs_backbone.to(device)
    zs_backbone.eval()

    all_gt, all_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Zero-shot inference"):
            x, y = x.to(device), y.to(device)
            all_pred.append(torch.sigmoid(zs_backbone(x)).cpu().numpy())
            all_gt.append(y.cpu().numpy())

    zs_gt   = np.concatenate(all_gt)
    zs_pred = np.concatenate(all_pred)

    np.save(os.path.join(out_dir, "zeroshot_gt.npy"),   zs_gt)
    np.save(os.path.join(out_dir, "zeroshot_pred.npy"), zs_pred)
    logger.info(f"Zero-shot predictions saved — gt: {zs_gt.shape}  pred: {zs_pred.shape}")
    return zs_gt, zs_pred


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
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
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
# PLOT 1 — per-class ROC curves (+ macro + micro average)
# =============================================================================
def plot_roc_curves(all_gt, all_pred, labels, n_classes, split_name, saved_dir):
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    valid_fprs, valid_tprs, valid_aucs, valid_labels, valid_indices = [], [], [], [], []
    for i, label in enumerate(labels):
        true = all_gt[:, i];  pred = all_pred[:, i]
        if len(np.unique(true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(true, pred)
        valid_fprs.append(fpr);  valid_tprs.append(tpr)
        valid_aucs.append(auc(fpr, tpr))
        valid_labels.append(label);  valid_indices.append(i)

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

    mean_fpr    = np.linspace(0, 1, 500)
    mean_tpr    = np.mean(
        [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(valid_fprs, valid_tprs)], axis=0
    )
    mean_tpr[0] = 0.0
    macro_auc_val = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, color="#f0c040", lw=2.8, ls="--", zorder=10,
            label=f"Macro-average  (AUC = {macro_auc_val:.3f})")

    gt_micro   = all_gt[:,   valid_indices].ravel()
    pred_micro = all_pred[:, valid_indices].ravel()
    fpr_m, tpr_m, _ = roc_curve(gt_micro, pred_micro)
    micro_auc_val   = auc(fpr_m, tpr_m)
    ax.plot(fpr_m, tpr_m, color="#40e0d0", lw=2.8, ls="-.", zorder=10,
            label=f"Micro-average  (AUC = {micro_auc_val:.3f})")

    ax.plot([0, 1], [0, 1], color="#666666", lw=1.0, ls=":", zorder=5,
            label="Chance  (AUC = 0.500)")

    ax.set_xlim([-0.01, 1.01]);  ax.set_ylim([-0.01, 1.05])
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
# PLOT 2 — zero-shot vs PEFT comparison
# =============================================================================
def plot_comparison(ft_gt, ft_pred, zs_gt, zs_pred, labels, n_classes, saved_dir,
                    plot_name="comparison_zeroshot_vs_peft"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0d1117")

    valid_indices = [
        i for i in range(n_classes)
        if len(np.unique(ft_gt[:, i])) > 1 and len(np.unique(zs_gt[:, i])) > 1
    ]
    logger.info(f"[comparison] Shared valid labels: {len(valid_indices)}/{n_classes}")

    mean_fpr     = np.linspace(0, 1, 500)
    panel_labels = ["Macro-average", "Micro-average"]

    for ax, title_suffix in zip(axes, panel_labels):
        ax.set_facecolor("#0d1117")

        if title_suffix == "Macro-average":
            ft_tprs = [np.interp(mean_fpr, *roc_curve(ft_gt[:, i], ft_pred[:, i])[:2])
                       for i in valid_indices]
            ft_mean_tpr    = np.mean(ft_tprs, axis=0);  ft_mean_tpr[0] = 0.0
            ft_auc_val     = auc(mean_fpr, ft_mean_tpr)

            zs_tprs = [np.interp(mean_fpr, *roc_curve(zs_gt[:, i], zs_pred[:, i])[:2])
                       for i in valid_indices]
            zs_mean_tpr    = np.mean(zs_tprs, axis=0);  zs_mean_tpr[0] = 0.0
            zs_auc_val     = auc(mean_fpr, zs_mean_tpr)

            ax.plot(mean_fpr, ft_mean_tpr, color="#f0c040", lw=2.5,
                    label=f"PEFT ({MODE})  (AUC={ft_auc_val:.3f})")
            ax.plot(mean_fpr, zs_mean_tpr, color="#e05858", lw=2.5, ls="--",
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

            ax.plot(fpr_ft, tpr_ft, color="#40e0d0", lw=2.5,
                    label=f"PEFT ({MODE})  (AUC={ft_auc_val:.3f})")
            ax.plot(fpr_zs, tpr_zs, color="#e05858", lw=2.5, ls="--",
                    label=f"Zero-shot      (AUC={zs_auc_val:.3f})")

        ax.plot([0, 1], [0, 1], color="#555555", lw=1.0, ls=":", label="Chance (AUC=0.500)")
        delta = ft_auc_val - zs_auc_val
        ax.set_title(
            f"{title_suffix} ROC\nΔ AUC = {delta:+.3f}  (PEFT − zero-shot)",
            color="#ffffff", fontsize=12, pad=10
        )
        ax.set_xlabel("False Positive Rate", color="#cccccc", fontsize=11)
        ax.set_ylabel("True Positive Rate",  color="#cccccc", fontsize=11)
        ax.set_xlim([-0.01, 1.01]);  ax.set_ylim([-0.01, 1.05])
        ax.tick_params(colors="#888888", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a2a")
        ax.grid(color="#1e1e1e", lw=0.5, linestyle="--")
        ax.legend(loc="lower right", fontsize=10, framealpha=0.7,
                  facecolor="#141920", edgecolor="#444444", labelcolor="#eeeeee")

    scope = "Test set" if "test" in plot_name else "Full dataset"
    fig.suptitle(
        f"Zero-shot Baseline vs PEFT ({MODE}) — {scope}\n"
        f"{len(valid_indices)} shared valid labels",
        color="#ffffff", fontsize=14, y=1.01
    )
    plt.tight_layout()
    out_path = os.path.join(saved_dir, f"{plot_name}.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"[comparison] Plot saved: {out_path}")


# =============================================================================
# FULL EVALUATION
# =============================================================================
def full_evaluation(all_gt, all_pred, labels, n_classes, split_name, saved_dir,
                    n_resamples=100):
    logger.info(f"\nComputing full evaluation for: {split_name}")
    results, skipped = [], []

    for i, label in enumerate(tqdm(labels, desc=f"Metrics [{split_name}]")):
        true  = all_gt[:, i];  pred = all_pred[:, i]
        n_pos = int(true.sum()); n_neg = int((1 - true).sum())

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
            "Label"         : label,
            "n_pos"         : n_pos,
            "n_neg"         : n_neg,
            "Threshold"     : round(threshold, 3),
            "Sensitivity"   : round(sens,  3), "Sensitivity_CI": sens_ci,
            "Specificity"   : round(spec,  3), "Specificity_CI": spec_ci,
            "F1"            : round(f1,    3), "F1_CI"         : f1_ci,
            "PPV"           : round(ppv,   3), "PPV_CI"        : ppv_ci,
            "NPV"           : round(npv,   3), "NPV_CI"        : npv_ci,
            "AUROC"         : round(auroc, 3) if not np.isnan(auroc) else np.nan,
            "AUROC_CI"      : auroc_ci,
            "AUPRC"         : round(auprc, 3) if not np.isnan(auprc) else np.nan,
            "AUPRC_CI"      : auprc_ci,
        })

    results_df = pd.DataFrame(results).sort_values("AUROC", ascending=False)
    skipped_df = pd.DataFrame(skipped)

    results_df.to_csv(os.path.join(saved_dir, f"{split_name}_results.csv"),        index=False)
    skipped_df.to_csv(os.path.join(saved_dir, f"{split_name}_skipped_labels.csv"), index=False)

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
        logger.info(f"\n[{split_name}] SKIPPED ({len(skipped_df)}) — all-zero labels")
        for _, row in skipped_df.iterrows():
            logger.info(f"  {row['Label']:<52}  {row['reason']}")

    logger.info(f"\n[{split_name}] Valid labels : {len(results_df)} / {n_classes}")
    logger.info(f"[{split_name}] Macro AUROC  : {macro_auroc:.4f}")

    plot_macro, plot_micro = plot_roc_curves(
        all_gt, all_pred, labels, n_classes, split_name, saved_dir
    )
    logger.info(f"[{split_name}] Plot Macro AUC={plot_macro:.4f}  Micro AUC={plot_micro:.4f}")
    return results_df


# =============================================================================
# LABEL STATS
# =============================================================================
def log_label_stats(df, split_name, labels, n_classes):
    label_matrix  = np.array(df["label"].tolist())
    counts        = label_matrix.sum(axis=0).astype(int)
    active_labels = [(labels[i], counts[i]) for i in range(n_classes) if counts[i] > 0]
    active_labels.sort(key=lambda x: -x[1])
    logger.info(f"\n{'='*60}")
    logger.info(f"Label stats [{split_name}]  --  {len(df)} samples total")
    logger.info(f"  Active labels : {len(active_labels)} / {n_classes}")
    logger.info(f"  Avg labels/sample: {counts.sum() / max(len(df), 1):.2f}")
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

    # ── Load data ──────────────────────────────────────────────────────────────
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

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    valloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    testloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    logger.info(
        f"Train batches: {len(trainloader)} | "
        f"Val batches:   {len(valloader)}   | "
        f"Test batches:  {len(testloader)}"
    )

    # ── Build model with manual PEFT adapters ──────────────────────────────────
    model = build_model(device=device, pth=pth, n_classes=n_classes, mode=MODE)

    # ── Optimizer / scheduler / loss ───────────────────────────────────────────
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ACTIVE_LR,
        weight_decay=weight_decay,
    )
    logger.info(f"Learning rate   : {ACTIVE_LR}")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.1, mode="max"
    )
    criterion = nn.BCEWithLogitsLoss()

    # ── Early stopping ─────────────────────────────────────────────────────────
    early_stopping = EarlyStopping(
        patience  = EARLY_STOP_PATIENCE,
        min_delta = EARLY_STOP_MIN_DELTA,
    )
    logger.info(
        f"Early stopping  : patience={EARLY_STOP_PATIENCE}, "
        f"min_delta={EARLY_STOP_MIN_DELTA}"
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_auroc = 0.0
    global_step    = 0
    last_ckpt_path = None

    for epoch in range(Epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{Epochs}")

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for x, y in tqdm(trainloader, desc=f"Epoch {epoch+1} Train"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss  += loss.item()
            global_step += 1

            # AdaLoRA: prune singular values on schedule after warmup
            if MODE == "adalora" and global_step > ADALORA_WARMUP_STEPS:
                if global_step % ADALORA_PRUNE_EVERY == 0:
                    adalora_prune_step(model, target_r=ADALORA_R_MIN)

        logger.info(f"Train Loss: {train_loss / len(trainloader):.4f}")

        # ── Validate ───────────────────────────────────────────────────────────
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

        val_macro_auroc, valid_count = compute_macro_auroc(all_gt, all_pred, n_classes)
        logger.info(
            f"Val Loss: {val_loss/len(valloader):.4f} | "
            f"Val Macro AUROC: {val_macro_auroc:.4f} | "
            f"Valid labels: {valid_count}/{n_classes}"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_macro_auroc)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != current_lr:
            logger.info(f"LR reduced: {current_lr:.6f} -> {new_lr:.6f}")

        # ── Save best checkpoint ───────────────────────────────────────────────
        if val_macro_auroc > best_val_auroc:
            best_val_auroc = val_macro_auroc
            new_ckpt_path  = os.path.join(
                saved_dir,
                f"child_ecg_peft_{MODE}_epoch{epoch+1}_auroc{val_macro_auroc:.4f}.pth"
            )
            if last_ckpt_path is not None and os.path.exists(last_ckpt_path):
                os.remove(last_ckpt_path)
                logger.info(f"[DELETED] Old checkpoint: {os.path.basename(last_ckpt_path)}")

            torch.save({
                "epoch"        : epoch + 1,
                "step"         : global_step,
                "state_dict"   : model.state_dict(),
                "optimizer"    : optimizer.state_dict(),
                "scheduler"    : scheduler.state_dict(),
                "val_auroc"    : val_macro_auroc,
                "peft_mode"    : MODE,
                "lora_r"       : LORA_R,
                "lora_alpha"   : LORA_ALPHA,
                "adalora_r"    : ADALORA_R,
                "adalora_r_min": ADALORA_R_MIN,
                "n_classes"    : n_classes,
            }, new_ckpt_path)
            last_ckpt_path = new_ckpt_path
            logger.info(f"[SAVED] Best checkpoint: {os.path.basename(new_ckpt_path)}")

        # ── LR-floor stop ──────────────────────────────────────────────────────
        if optimizer.param_groups[0]["lr"] < early_stop_lr:
            logger.info(f"LR floor reached (< {early_stop_lr:.2e}) — stopping.")
            break

        # ── Patience-based early stopping ──────────────────────────────────────
        if early_stopping(val_macro_auroc):
            logger.info(f"Early stopping at epoch {epoch+1}.")
            break

    logger.info(
        f"\nTraining complete | Best Val Macro AUROC: {best_val_auroc:.4f} | "
        f"Stopped at epoch: {epoch+1}/{Epochs}"
    )

    # ── Reload best checkpoint ─────────────────────────────────────────────────
    if last_ckpt_path is not None and os.path.exists(last_ckpt_path):
        logger.info(f"\nReloading best checkpoint: {os.path.basename(last_ckpt_path)}")
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        logger.info(f"Loaded epoch {ckpt['epoch']} (val AUROC={ckpt['val_auroc']:.4f})")
    else:
        logger.warning("No checkpoint found — using last-epoch weights.")

    # ── Final test evaluation ──────────────────────────────────────────────────
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

    # ── Zero-shot comparison on test split ────────────────────────────────────
    if os.path.exists(ZEROSHOT_GT_PATH) and os.path.exists(ZEROSHOT_PRED_PATH):
        logger.info("\nLoading cached zero-shot baseline predictions...")
        zs_gt   = np.load(ZEROSHOT_GT_PATH)
        zs_pred = np.load(ZEROSHOT_PRED_PATH)
        if zs_gt.shape[0] != all_gt.shape[0]:
            logger.info(
                f"Cache shape mismatch ({zs_gt.shape[0]} vs {all_gt.shape[0]}) "
                "— re-running zero-shot on current test split..."
            )
            zs_gt, zs_pred = run_zeroshot_baseline(test_df, saved_dir)
    else:
        logger.info("\nZero-shot cache not found — generating baseline on test split...")
        zs_gt, zs_pred = run_zeroshot_baseline(test_df, saved_dir)

    plot_comparison(
        ft_gt=all_gt, ft_pred=all_pred,
        zs_gt=zs_gt,  zs_pred=zs_pred,
        labels=labels, n_classes=n_classes,
        saved_dir=saved_dir,
        plot_name="comparison_zeroshot_vs_peft_test"
    )

    # ── Full dataset inference + comparison ───────────────────────────────────
    logger.info("\nRunning inference on full dataset (train + val + test)...")
    full_df      = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_dataset = ECG_Dataset(ecg_path=ecg_path, df=full_df)
    full_loader  = DataLoader(
        full_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )

    model.eval()
    ft_full_gt, ft_full_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(full_loader, desc="PEFT inference (full dataset)"):
            x, y = x.to(device), y.to(device)
            ft_full_pred.append(torch.sigmoid(model(x)).cpu().numpy())
            ft_full_gt.append(y.cpu().numpy())
    ft_full_gt   = np.concatenate(ft_full_gt)
    ft_full_pred = np.concatenate(ft_full_pred)

    logger.info("Running zero-shot inference on full dataset...")
    zs_full_gt, zs_full_pred = run_zeroshot_baseline(full_df, saved_dir)

    np.save(os.path.join(saved_dir, "full_ft_gt.npy"),   ft_full_gt)
    np.save(os.path.join(saved_dir, "full_ft_pred.npy"), ft_full_pred)

    full_evaluation(
        ft_full_gt, ft_full_pred, labels, n_classes,
        split_name="full_dataset", saved_dir=saved_dir, n_resamples=100
    )

    plot_comparison(
        ft_gt=ft_full_gt,   ft_pred=ft_full_pred,
        zs_gt=zs_full_gt,   zs_pred=zs_full_pred,
        labels=labels,      n_classes=n_classes,
        saved_dir=saved_dir,
        plot_name="comparison_zeroshot_vs_peft_full"
    )

    logger.info("\nAll done.")


if __name__ == "__main__":
    main()