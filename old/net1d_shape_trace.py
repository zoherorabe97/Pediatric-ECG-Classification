"""
net1d_shape_trace.py
====================
Traces the tensor shape through every layer of Net1D for four synthetic ECG
samples with different recording durations:
    ─────────────────────────────────────────────
    Label   Duration   fs (Hz)   Samples (T)
    5 s       5 s        500       2 500
   10 s      10 s        500       5 000
   50 s      50 s        500      25 000
  120 s     120 s        500      60 000
    ─────────────────────────────────────────────

Outputs
-------
  1. Printed table to stdout  (shape at each named checkpoint)
  2. net1d_shape_trace.png    (architecture diagram with per-sample shapes)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from net1d import Net1D

# =============================================================================
# CONFIG  (mirror paths from zeroshot_pediatric.py)
# =============================================================================
DEVICE     = torch.device("cpu")
PTH        = "./12_lead_ECGFounder.pth"
TASKS_PATH = "./tasks.txt"
OUT_PNG    = "./net1d_shape_trace.png"
ECG_PATH   = "C:/Users/zoorab/Desktop/zoher/University/Projects/Zhengzhou_ECG/Child_ecg/"
CSV_PATH   = "./ecg_with_exact_match.csv"
N_PICK     = 4    # how many real samples to trace
RAND_SEED  = 0

# Named checkpoints we want to capture  (order matters – same order as forward)
CHECKPOINTS = [
    "Input",
    "first_conv",
    "first_bn+act",
    "Stage 0 - block 0 (downsample)",
    "Stage 0 - block 1",
    "Stage 1 - block 0 (downsample)",
    "Stage 1 - block 1",
    "Stage 2 - block 0 (downsample)",
    "Stage 2 - block 1",
    "Stage 3 - block 0 (downsample)",
    "Stage 3 - block 1",
    "Stage 3 - block 2",
    "Stage 4 - block 0 (downsample)",
    "Stage 4 - block 1",
    "Stage 4 - block 2",
    "Stage 5 - block 0 (downsample)",
    "Stage 5 - block 1",
    "Stage 5 - block 2",
    "Stage 5 - block 3",
    "Stage 6 - block 0 (downsample)",
    "Stage 6 - block 1",
    "Stage 6 - block 2",
    "Stage 6 - block 3",
    "GlobalAvgPool",
    "Dense (logits)",
    "Sigmoid (probs)",
]

# =============================================================================
# MODEL LOADING
# =============================================================================
def load_model(n_classes: int) -> Net1D:
    model = Net1D(
        in_channels=12, base_filters=64, ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16, stride=2, groups_width=16,
        verbose=False, use_bn=False, use_do=False, n_classes=n_classes,
    )
    if os.path.isfile(PTH):
        ckpt = torch.load(PTH, map_location=DEVICE, weights_only=False)
        sd   = ckpt.get("state_dict", ckpt)
        log  = model.load_state_dict(sd, strict=False)
        print(f"[model] Loaded {PTH}  missing={len(log.missing_keys)}  unexpected={len(log.unexpected_keys)}")
    else:
        print(f"[model] {PTH} not found - using random weights (shapes are identical)")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(DEVICE)

# =============================================================================
# INSTRUMENTED FORWARD  (no hooks needed – we step through manually)
# =============================================================================
def traced_forward(model: Net1D, x: torch.Tensor):
    """
    Replicate Net1D.forward() step-by-step and record the tensor shape after
    every logical checkpoint.  Returns ordered list of (checkpoint_name, shape).
    """
    trace: list[tuple[str, tuple]] = []

    def record(name: str, t: torch.Tensor):
        trace.append((name, tuple(t.shape)))

    out = x
    record("Input", out)

    # first conv
    out = model.first_conv(out)
    # use_bn=False in the ECGFounder checkpoint, but handle both cases
    if model.use_bn:
        out = model.first_bn(out)
    out = model.first_activation(out)
    record("first_conv", out)
    record("first_bn+act", out)   # same tensor, same shape

    # stages
    stage_cfg = [
        (0, [2, 2]),
        (1, [2, 2]),
        (2, [2, 2]),
        (3, [3, 3, 3]),
        (4, [3, 3, 3]),
        (5, [4, 4, 4, 4]),
        (6, [4, 4, 4, 4]),
    ]
    for i_stage, block_ids in stage_cfg:
        stage = model.stage_list[i_stage]
        for i_block in range(stage.m_blocks):
            block = stage.block_list[i_block]
            out = block(out)
            ds_tag = " (downsample)" if block.downsample else ""
            record(f"Stage {i_stage} - block {i_block}{ds_tag}", out)

    # global average pool
    deep_features = out.mean(-1)
    record("GlobalAvgPool", deep_features)

    # dense
    logits = model.dense(deep_features)
    record("Dense (logits)", logits)

    # sigmoid
    probs = torch.sigmoid(logits)
    record("Sigmoid (probs)", probs)

    return trace

# =============================================================================
# PRETTY PRINT TABLE
# =============================================================================
def fmt_shape(shape: tuple) -> str:
    return "(" + ", ".join(str(d) for d in shape) + ")"

def print_table(all_traces: dict[str, list[tuple[str, tuple]]]):
    sample_labels = list(all_traces.keys())
    rows          = list(all_traces[sample_labels[0]])   # (name, shape) pairs

    col_w_name = max(len(name) for name, _ in rows) + 2
    col_w_shp  = 30

    header = f"{'Layer / Checkpoint':<{col_w_name}}" + "".join(
        f"{lbl:^{col_w_shp}}" for lbl in sample_labels
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print("  Net1D  - tensor shape trace  (batch dimension = 1)")
    print(sep)
    print(header)
    print(sep)

    n_rows = len(rows)
    for i in range(n_rows):
        name  = rows[i][0]
        shapes_this_row = [all_traces[lbl][i][1] for lbl in sample_labels]
        # highlight rows where all shapes agree in channels but differ in T
        row_str = f"{name:<{col_w_name}}"
        for shp in shapes_this_row:
            row_str += f"{fmt_shape(shp):^{col_w_shp}}"
        print(row_str)

    print(sep + "\n")  # noqa

# =============================================================================
# ARCHITECTURE DIAGRAM
# =============================================================================
BLOCK_COLOR = {
    "Input":           "#2a4a7a",
    "first_conv":      "#1a6a5a",
    "first_bn+act":    "#1a6a5a",
    "Stage":           "#5a2a6a",
    "GlobalAvgPool":   "#7a4a1a",
    "Dense (logits)":  "#7a1a1a",
    "Sigmoid (probs)": "#1a7a2a",
}

SAMPLE_COLORS = ["#61d4f5", "#f5c261", "#f57861", "#a8f561"]
SAMPLE_LS     = ["-",       "--",      "-.",      ":"]

def block_color(name: str) -> str:
    for key, color in BLOCK_COLOR.items():
        if name.startswith(key):
            return color
    return "#3a3a3a"

# =============================================================================
# REAL DATA LOADER
# =============================================================================
def z_score(signal: np.ndarray) -> np.ndarray:
    return (signal - signal.mean()) / (signal.std() + 1e-8)


def load_real_samples(n_pick: int = 4) -> dict:
    """
    Load `n_pick` real ECGs from the dataset, chosen to maximise temporal
    length diversity.  Returns {label: tensor(1,12,T)}.
    """
    df = pd.read_csv(CSV_PATH)
    if isinstance(df["label"].iloc[0], str):
        df["label"] = df["label"].apply(json.loads)

    # Probe a random subset to find their actual lengths
    probe = df.sample(n=min(200, len(df)), random_state=RAND_SEED).reset_index(drop=True)
    lengths, valid_idx = [], []
    for i, row in probe.iterrows():
        try:
            data, _ = wfdb.rdsamp(ECG_PATH + row["Filename"])
            lengths.append(data.shape[0])   # T is axis-0 from wfdb
            valid_idx.append(i)
        except Exception:
            pass

    probe = probe.loc[valid_idx].copy()
    probe["_T"] = lengths
    probe = probe.sort_values("_T").reset_index(drop=True)

    # Pick n_pick evenly spaced across the sorted length distribution
    indices = np.linspace(0, len(probe) - 1, n_pick, dtype=int)
    chosen  = probe.iloc[indices]

    inputs = {}
    print("\n[inputs] Real ECG tensors loaded from dataset (shape = 1 x 12 x T)")
    for k, (_, row) in enumerate(chosen.iterrows()):
        data, _ = wfdb.rdsamp(ECG_PATH + row["Filename"])
        data = np.transpose(data, (1, 0))   # (12, T)
        data = z_score(data)
        x    = torch.FloatTensor(data).unsqueeze(0)   # (1, 12, T)
        rec_id = row["Filename"].split("/")[-1]        # e.g. P05629_E02
        label  = f"#{k+1} {rec_id} T={x.shape[-1]:,}"
        inputs[label] = x
        print(f"  {row['Filename']:<40s}  ->  {tuple(x.shape)}")
    return inputs


# =============================================================================
# LAYER PARAMS MAP  (weight shapes for every checkpoint)
# =============================================================================
def get_layer_params_map(model: Net1D) -> dict:
    """
    Returns a dict  {checkpoint_name: human-readable param string}
    showing Conv/Linear kernel shapes for every recorded checkpoint.
    """
    pm = {}
    pm["Input"] = "(batch, 12 leads, T samples)"

    # first_conv: MyConv1dPadSame wraps a plain Conv1d
    c = model.first_conv.conv
    pm["first_conv"]   = f"Conv1d  {c.in_channels}->{c.out_channels}  k={c.kernel_size[0]}  s=2"
    pm["first_bn+act"] = "BatchNorm1d(64) + Swish" if model.use_bn else "Swish  [BN off]"

    for i_stage in range(model.n_stages):
        stage = model.stage_list[i_stage]
        for i_block in range(stage.m_blocks):
            blk    = stage.block_list[i_block]
            ds_tag = " (downsample)" if blk.downsample else ""
            key    = f"Stage {i_stage} - block {i_block}{ds_tag}"
            c1, c2, c3 = blk.conv1.conv, blk.conv2.conv, blk.conv3.conv
            s = c2.stride[0]
            pm[key] = (
                f"conv1x1 {c1.in_channels}->{c1.out_channels}  "
                f"| convKxK {c2.in_channels}->{c2.out_channels} k={c2.kernel_size[0]} s={s} g={c2.groups}  "
                f"| conv1x1 {c3.in_channels}->{c3.out_channels}  "
                f"| SE {blk.se_fc1.in_features}->{blk.se_fc2.out_features}"
            )

    pm["GlobalAvgPool"]   = "mean(dim=-1)  (1,1024,T) -> (1,1024)"
    pm["Dense (logits)"]  = f"Linear  {model.dense.in_features} -> {model.dense.out_features}"
    pm["Sigmoid (probs)"] = f"sigmoid  -> (1, {model.dense.out_features})"
    return pm


def make_diagram(all_traces: dict[str, list[tuple[str, tuple]]], params_map: dict = None):
    sample_labels = list(all_traces.keys())
    rows          = list(all_traces[sample_labels[0]])

    n_rows   = len(rows)
    fig_h    = max(18, n_rows * 0.55 + 3)
    fig, ax  = plt.subplots(figsize=(20, fig_h), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n_rows + 0.5)
    ax.axis("off")

    # ── title ────────────────────────────────────────────────────────────────
    ax.text(0.5, n_rows + 0.3,
            "Net1D — Tensor Shape Trace (batch=1, 12-lead ECG)",
            ha="center", va="bottom", fontsize=14, fontweight="bold",
            color="#ffffff", fontfamily="monospace")

    # ── legend (built from actual sample labels) ──────────────────────────────
    handles = [
        mpatches.Patch(color=c, label=lbl)
        for lbl, c in zip(sample_labels, SAMPLE_COLORS)
    ]
    ax.legend(handles=handles, loc="upper right",
              bbox_to_anchor=(0.99, 1.0),
              fontsize=8, framealpha=0.3, facecolor="#1a1a2a",
              edgecolor="#555555", labelcolor="#cccccc", handlelength=1.8)

    # ── layout: [layer box] [params] [sample shapes x4] ────────────────────
    BOX_X    = 0.01
    BOX_W    = 0.22
    PARAM_X  = 0.24   # centre of the params column
    PARAM_W  = 0.20
    SHAPE_X  = [0.45 + i * 0.135 for i in range(len(sample_labels))]

    # header row
    y_hdr = n_rows - 0.05
    ax.text(BOX_X + BOX_W / 2, y_hdr, "Layer / Checkpoint",
            ha="center", va="center", fontsize=7.5, color="#aaaaaa",
            fontfamily="monospace")
    ax.text(PARAM_X, y_hdr, "Params / Filters",
            ha="center", va="center", fontsize=7.5, color="#ffcc66",
            fontfamily="monospace")
    for xi, lbl, col in zip(SHAPE_X, sample_labels, SAMPLE_COLORS):
        ax.text(xi, y_hdr, lbl, ha="center", va="center",
                fontsize=8.0, color=col, fontweight="bold",
                fontfamily="monospace")

    # ── draw rows ────────────────────────────────────────────────────────────
    for i, (name, _) in enumerate(rows):
        y     = n_rows - 1 - i
        color = block_color(name)

        # layer name box
        rect = mpatches.FancyBboxPatch(
            (BOX_X, y - 0.33), BOX_W, 0.66,
            boxstyle="round,pad=0.01",
            linewidth=0.6, edgecolor="#555555", facecolor=color, alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(BOX_X + BOX_W / 2, y, name,
                ha="center", va="center", fontsize=6.8, color="#eeeeee",
                fontfamily="monospace", clip_on=True)

        # downward connector arrow (skip last row)
        if i < n_rows - 1:
            ax.annotate("", xy=(BOX_X + BOX_W / 2, y - 0.34),
                        xytext=(BOX_X + BOX_W / 2, y - 0.33),
                        arrowprops=dict(arrowstyle="-|>", color="#555555",
                                        lw=0.7, mutation_scale=8))

        # params / filter text
        if params_map and name in params_map:
            # For stage blocks, split into 2 lines for readability
            pstr = params_map[name]
            parts = pstr.split(" | ")
            if len(parts) > 1:
                line1 = " | ".join(parts[:2])
                line2 = " | ".join(parts[2:])
                ax.text(PARAM_X, y + 0.12, line1, ha="center", va="center",
                        fontsize=4.8, color="#ffcc66", fontfamily="monospace")
                ax.text(PARAM_X, y - 0.12, line2, ha="center", va="center",
                        fontsize=4.8, color="#ffcc66", fontfamily="monospace")
            else:
                ax.text(PARAM_X, y, pstr, ha="center", va="center",
                        fontsize=5.0, color="#ffcc66", fontfamily="monospace")

        # tensor shape for every sample
        for xi, lbl, col in zip(SHAPE_X, sample_labels, SAMPLE_COLORS):
            shape   = all_traces[lbl][i][1]
            shp_str = fmt_shape(shape)
            ax.text(xi, y, shp_str, ha="center", va="center",
                    fontsize=6.5, color=col, fontfamily="monospace")

    # ── vertical guides ───────────────────────────────────────────────────────
    ax.axvline(PARAM_X, color="#ffcc66", lw=0.3, ls="--", alpha=0.18)
    for xi, col, ls in zip(SHAPE_X, SAMPLE_COLORS, SAMPLE_LS):
        ax.axvline(xi, color=col, lw=0.3, ls=ls, alpha=0.18)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT_PNG, dpi=170, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n[diagram] Saved -> {OUT_PNG}")

# =============================================================================
# ECG WAVEFORM PLOTTING
# =============================================================================
def plot_ecg_samples(inputs: dict):
    """
    Plots the 12-lead ECG waveforms for each loaded sample and saves them.
    """
    print("\n[plots] Generating ECG waveform plots...")
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    for label, x in inputs.items():
        data = x[0].numpy() # shape (12, T)
        T = data.shape[1]
        
        # Determine a reasonable time window to plot (e.g., first 5 seconds or 2500 samples)
        # Plotting 25000 samples might be too dense, but we can plot the whole thing or a segment
        # Let's plot the entire signal but use a large figure size.
        fig, axes = plt.subplots(12, 1, figsize=(15, 20), sharex=True, facecolor="#0d1117")
        fig.patch.set_facecolor("#0d1117")
        fig.suptitle(f"ECG Waveform - {label}", color="white", fontsize=16)
        
        time_axis = np.arange(T) / 500.0 # fs = 500
        
        for i, ax in enumerate(axes):
            ax.set_facecolor("#0d1117")
            ax.plot(time_axis, data[i], color="#61d4f5", lw=0.8)
            ax.set_ylabel(lead_names[i], color="white", rotation=0, labelpad=20, va="center")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#555555")
            ax.grid(True, color="#333333", linestyle="--", alpha=0.5)
            
        axes[-1].set_xlabel("Time (seconds)", color="white")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Clean up filename
        safe_label = label.replace(" ", "_").replace("=", "").replace(",", "")
        out_name = f"./ecg_plot_{safe_label}.png"
        plt.savefig(out_name, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved -> {out_name}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    # ── n_classes from tasks.txt ──────────────────────────────────────────────
    if os.path.isfile(TASKS_PATH):
        with open(TASKS_PATH) as f:
            n_classes = sum(1 for ln in f if ln.strip())
    else:
        n_classes = 150
        print(f"[warn] {TASKS_PATH} not found - using n_classes={n_classes}")
    print(f"[config] n_classes = {n_classes}")

    model = load_model(n_classes)

    # ── load real ECG samples ─────────────────────────────────────────────────
    inputs = load_real_samples(n_pick=N_PICK)

    # ── forward pass with tracing ─────────────────────────────────────────────
    all_traces: dict[str, list[tuple[str, tuple]]] = {}
    print("\n[trace] Running instrumented forward passes...")
    for label, x in inputs.items():
        with torch.no_grad():
            trace = traced_forward(model, x)
        all_traces[label] = trace
        print(f"  {label}  ->  {len(trace)} checkpoints recorded")

    # ── extract weight/filter shapes from model ───────────────────────────────
    params_map = get_layer_params_map(model)

    # ── console table ─────────────────────────────────────────────────────────
    print_table(all_traces)

    # ── diagram ───────────────────────────────────────────────────────────────
    make_diagram(all_traces, params_map=params_map)
    
    # ── waveform plots ────────────────────────────────────────────────────────
    plot_ecg_samples(inputs)
    
    print("\n[done]")


if __name__ == "__main__":
    main()
