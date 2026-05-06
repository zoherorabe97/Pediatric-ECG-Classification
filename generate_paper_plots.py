# -*- coding: utf-8 -*-
"""
generate_paper_plots.py  -  8 publication-quality comparison plots
Plots saved to: <project>/paper_plots/

Fixes applied (v3):
  - Plot 02: zero-shot point included at x=0 so lines start from y=zero_shot_auroc
  - Plot 03: starts at 500 (drops 100), pediatric ResNet bars fixed
  - Plot 06: ResNet/Net1D bars fixed for all sample sizes; y-axis starts at 0.65
  - Plot 08: replaced efficiency with 2x3 per-dataset subplot grid (FT vs Scratch)
  - All print() calls use ASCII only (Windows cp1252 safe)
  - "mode" column accessed via brackets to avoid pandas .mode() method clash
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

BASE  = r"c:\Users\zoorab\Desktop\zoher\University\Projects\ECGFounder"
CINC  = os.path.join(BASE, "cinc_res")
RES   = os.path.join(BASE, "res")
OUT   = os.path.join(BASE, "paper_plots")
os.makedirs(OUT, exist_ok=True)

PAL = {
    "Chapman": "#4C72B0",
    "CPSC":    "#DD8452",
    "G12EC":   "#55A868",
    "PTB":     "#C44E52",
    "SPH":     "#8172B2",
}
PC = "#E84393"   # pediatric colour

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_cinc():
    frames = []
    for name in PAL:
        d = pd.read_csv(os.path.join(CINC, f"{name}_aggregate_summary.csv"))
        d["dataset"] = name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)

cinc  = load_cinc()
pft   = pd.read_csv(os.path.join(RES, "ablation_ft",    "ft_ablation_summary_with_zeroshot.csv"))
resn  = pd.read_csv(os.path.join(RES, "ablation_resnet", "ablation_summary.csv"))
net1d = pd.read_csv(os.path.join(RES, "ablation_net1d",  "ablation_summary.csv"))

# Use bracket notation — "mode" clashes with pandas DataFrame.mode() method
zs = cinc[cinc["mode"] == "zero_shot"].copy()
ft = cinc[cinc["mode"] == "fine_tune"].copy()
sc = cinc[cinc["mode"] == "scratch"].copy()

# Pediatric scalars
def ped(exp, col):
    return float(pft.loc[pft["experiment"] == exp, col].values[0])

pz   = ped("ft_zeroshot", "test_macro_AUROC")
pfu  = ped("ft_full",     "test_macro_AUROC")
p3k  = ped("ft_3000",     "test_macro_AUROC")
p1k  = ped("ft_1000",     "test_macro_AUROC")
p500 = ped("ft_500",      "test_macro_AUROC")
p100 = ped("ft_100",      "test_macro_AUROC")
n_fu = int(ped("ft_full", "n_train_actual"))   # 7674

# Best ResNet/Net1D per training size (pediatric scratch baselines)
def best_scratch(df, n_req):
    """Return best test_macro_AUROC for a given n_train_requested value."""
    sub = df[df["n_train_requested"].astype(str) == str(n_req)]
    return sub["test_macro_AUROC"].max() if len(sub) else np.nan

# Pre-compute scratch bests — note: "full" key is a string in the CSVs
resn_bests  = {n: best_scratch(resn,  n) for n in [100, 500, 1000, 3000]}
resn_bests["full"]  = best_scratch(resn,  "full")
net1d_bests = {n: best_scratch(net1d, n) for n in [100, 500, 1000, 3000]}
net1d_bests["full"] = best_scratch(net1d, "full")


# ===========================================================================
# Plot 1 - Zero-shot bar: adult CinC vs pediatric
# ===========================================================================
def plot_1():
    ds    = list(PAL)
    av    = [zs.loc[zs["dataset"] == d, "macro_auroc"].values[0] for d in ds]
    all_d = ds + ["Pediatric\n(ZU pECG)"]
    all_v = av + [pz]
    all_c = [PAL[d] for d in ds] + [PC]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(all_d, all_v, color=all_c, width=0.55, edgecolor="white", linewidth=0.8)
    for b, v in zip(bars, all_v):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    avg = np.mean(av)
    ax.axhline(avg, color="grey", linestyle="--", linewidth=1.2,
               label=f"Adult avg ({avg:.3f})")
    gap = avg - pz
    ax.annotate(f"Gap = {gap:.3f}",
                xy=(len(ds), pz), xytext=(len(ds) - 1.4, pz + 0.055),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
    ax.set_ylim(0.50, 1.03)
    ax.set_ylabel("Macro-AUROC", fontsize=12)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_title("Zero-Shot Macro-AUROC: Adult CinC vs. Pediatric ECG",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_zeroshot_bar.png"))
    plt.close(fig)
    print("[OK] Plot 1 - Zero-shot bar")


# ===========================================================================
# Plot 2 - Fine-tune scaling: lines start at x=0 (zero-shot point)
# ===========================================================================
def plot_2():
    fig, ax = plt.subplots(figsize=(10, 6))

    # We will map x values to their index for categorical spacing
    used_xs = sorted(set([0, 100, 500, 1000, 3000, 6000, 8000, 10000, n_fu]))
    x_map = {val: i for i, val in enumerate(used_xs)}

    for ds, c in PAL.items():
        sub = ft[ft["dataset"] == ds].sort_values("n_samples")
        zs_val = zs.loc[zs["dataset"] == ds, "macro_auroc"].values[0]

        # Prepend the zero-shot point at n=0
        xs = [0] + list(sub["n_samples"])
        ys = [zs_val] + list(sub["macro_auroc"])
        mapped_xs = [x_map[x] for x in xs]
        ax.plot(mapped_xs, ys, marker="o", color=c, lw=2, ms=5, label=ds)

    # Pediatric: zero-shot at 0, then FT points
    ped_xs = [0,   100,  500,  1000, 3000, n_fu]
    ped_ys = [pz, p100, p500,  p1k,  p3k,  pfu]
    mapped_ped_xs = [x_map[x] for x in ped_xs]
    ax.plot(mapped_ped_xs, ped_ys, marker="s", color=PC, lw=2.5, ms=8,
            label="Pediatric ZU pECGs", zorder=5)

    # Categorical x-ticks for clarity
    tick_map = {0: "0\n(ZS)", 100: "100", 500: "500", 1000: "1k",
                3000: "3k", 6000: "6k", 8000: "8k", 10000: "10k", n_fu: f"7.5k"}
    tick_positions = [x_map[x] for x in used_xs]
    tick_labels    = [tick_map.get(x, str(x)) for x in used_xs]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_xlim(-0.5, len(used_xs) - 0.5)

    ax.set_xlabel("Training Samples", fontsize=12)
    ax.set_ylabel("Macro-AUROC", fontsize=12)
    ax.set_title("Fine-Tuning Scaling: Adult CinC vs. Pediatric",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, ncol=2, loc="lower right")
    ax.set_ylim(0.5, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "02_ft_scaling.png"))
    plt.close(fig)
    print("[OK] Plot 2 - Fine-tune scaling (zero-shot at x=0)")


# ===========================================================================
# Plot 3 - Scratch vs FT gap (grouped bar, starting from 100 samples)
# ===========================================================================
def plot_3():
    # Only use sample sizes >= 100 and common to ft, sc, AND available in pediatric
    sizes_adult = sorted([n for n in set(ft["n_samples"].unique()) & set(sc["n_samples"].unique())
                          if n >= 100])
    # Pediatric data points: 100, 500, 1000, 3000 (no 6000/8000/10000 for pediatric FT)
    ped_map   = {100: p100, 500: p500, 1000: p1k, 3000: p3k}
    resn_map  = {100: resn_bests[100], 500: resn_bests[500], 1000: resn_bests[1000], 3000: resn_bests[3000]}

    # Limit to sizes where ALL four series have data
    sizes = [n for n in sizes_adult if n in ped_map]

    aft = [ft[ft["n_samples"] == n]["macro_auroc"].mean() for n in sizes]
    asc = [sc[sc["n_samples"] == n]["macro_auroc"].mean() for n in sizes]
    pv  = [ped_map[n]  for n in sizes]
    rv  = [resn_map[n] for n in sizes]

    x = np.arange(len(sizes))
    w = 0.18
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - 1.5*w, aft, w, label="Adult FT avg",            color="#4C72B0")
    ax.bar(x - 0.5*w, asc, w, label="Adult Scratch avg",       color="#4C72B0", alpha=0.45, hatch="//")
    ax.bar(x + 0.5*w, pv,  w, label="Pediatric ECGFounder FT", color=PC)
    ax.bar(x + 1.5*w, rv,  w, label="Pediatric ResNet (best)", color="#2ca02c", alpha=0.85)

    # Value labels
    for bar_x, vals in [(x-1.5*w, aft), (x-0.5*w, asc), (x+0.5*w, pv), (x+1.5*w, rv)]:
        for bxi, v in zip(bar_x, vals):
            if not np.isnan(v):
                ax.text(bxi, v + 0.003, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=6.5, fontweight="bold", rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("Training Samples", fontsize=12)
    ax.set_ylabel("Macro-AUROC", fontsize=12)
    ax.set_title("FT vs Scratch: Adult Average vs Pediatric (>=100 samples)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0.5, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "03_scratch_vs_ft_gap.png"))
    plt.close(fig)
    print("[OK] Plot 3 - Scratch vs FT gap (>=100)")


# ===========================================================================
# Plot 4 - ResNet ablation (model size x training size, pediatric)
# ===========================================================================
def plot_4():
    sizes = ["small", "medium", "large"]
    nreqs = sorted(resn["n_train_requested"].unique(),
                   key=lambda x: (x == "full", int(x) if x != "full" else 99999))
    pal2  = {"small": "#88CCEE", "medium": "#44AA99", "large": "#117733"}
    mks   = {"small": "o",       "medium": "s",       "large": "^"}
    
    actual_xs = [0, 100, 500, 1000, 3000, n_fu]
    mapped_xs = list(range(len(actual_xs)))
    xtick_labels = ["0\n(ZS)", "100", "500", "1k", "3k", f"Full\n({n_fu})"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    # Plot FT line
    ft_ys = [pz, p100, p500, p1k, p3k, pfu]
    ax.plot(mapped_xs, ft_ys, marker="D", color=PC, lw=2.5, ms=7,
            label="ECGFounder (Fine-Tuned)", zorder=5)
            
    # Annotate FT line
    for x, y in zip(mapped_xs, ft_ys):
        label_txt = f"ZS={y:.3f}" if x == 0 else f"{y:.3f}"
        ax.annotate(label_txt, (x, y),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.5, color=PC, fontweight="bold")

    # Mark zero-shot anchor
    ax.scatter([0], [pz], color=PC, s=100, zorder=6,
               marker="*", label=f"ECGFounder Zero-shot ({pz:.3f})")

    for sz in sizes:
        ys = [0.5]
        for n in nreqs:
            sub = resn[(resn["model_size"] == sz) & (resn["n_train_requested"].astype(str) == str(n))]
            ys.append(sub["test_macro_AUROC"].values[0] if len(sub) else np.nan)
        params = resn[resn["model_size"] == sz]["total_params"].values[0]
        ax.plot(mapped_xs, ys, marker=mks[sz], color=pal2[sz], lw=2, ms=6, ls="--", alpha=0.85,
                label=f"ResNet-{sz} Scratch ({params/1e6:.1f}M)")
                
        # Annotate Scratch line
        for x, y in zip(mapped_xs[1:], ys[1:]):
            if not np.isnan(y):
                ax.annotate(f"{y:.3f}", (x, y),
                            textcoords="offset points", xytext=(0, -13),
                            ha="center", fontsize=8, color=pal2[sz], alpha=0.85)

    ax.axhline(0.5, color="#555555", lw=1.0, ls=":", zorder=3,
               label="Random (AUROC=0.500)")

    ax.set_xticks(mapped_xs)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(-0.5, len(mapped_xs) - 0.5)
    
    ax.set_xlabel("Training Samples", fontsize=12)
    ax.set_ylabel("Test Macro-AUROC", fontsize=12)
    ax.set_title("ECGFounder FT vs ResNet Scratch Ablation - Pediatric ECG",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.set_ylim(0.48, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "04_resnet_ablation.png"))
    plt.close(fig)
    print("[OK] Plot 4 - ResNet ablation")


# ===========================================================================
# Plot 5 - Net1D ablation (model size x training size, pediatric)
# ===========================================================================
def plot_5():
    sizes = ["small", "medium", "large"]
    nreqs = sorted(net1d["n_train_requested"].unique(),
                   key=lambda x: (x == "full", int(x) if x != "full" else 99999))
    pal2  = {"small": "#DDCC77", "medium": "#CC6677", "large": "#882255"}
    mks   = {"small": "o",       "medium": "s",       "large": "^"}
    
    actual_xs = [0, 100, 500, 1000, 3000, n_fu]
    mapped_xs = list(range(len(actual_xs)))
    xtick_labels = ["0\n(ZS)", "100", "500", "1k", "3k", f"Full\n({n_fu})"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    # Plot FT line
    ft_ys = [pz, p100, p500, p1k, p3k, pfu]
    ax.plot(mapped_xs, ft_ys, marker="D", color=PC, lw=2.5, ms=7,
            label="ECGFounder (Fine-Tuned)", zorder=5)
            
    # Annotate FT line
    for x, y in zip(mapped_xs, ft_ys):
        label_txt = f"ZS={y:.3f}" if x == 0 else f"{y:.3f}"
        ax.annotate(label_txt, (x, y),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.5, color=PC, fontweight="bold")

    # Mark zero-shot anchor
    ax.scatter([0], [pz], color=PC, s=100, zorder=6,
               marker="*", label=f"ECGFounder Zero-shot ({pz:.3f})")

    for sz in sizes:
        ys = [0.5]
        for n in nreqs:
            sub = net1d[(net1d["model_size"] == sz) & (net1d["n_train_requested"].astype(str) == str(n))]
            ys.append(sub["test_macro_AUROC"].values[0] if len(sub) else np.nan)
        params = net1d[net1d["model_size"] == sz]["total_params"].values[0]
        ax.plot(mapped_xs, ys, marker=mks[sz], color=pal2[sz], lw=2, ms=6, ls="--", alpha=0.85,
                label=f"Net1D-{sz} Scratch ({params/1e6:.1f}M)")
                
        # Annotate Scratch line
        for x, y in zip(mapped_xs[1:], ys[1:]):
            if not np.isnan(y):
                ax.annotate(f"{y:.3f}", (x, y),
                            textcoords="offset points", xytext=(0, -13),
                            ha="center", fontsize=8, color=pal2[sz], alpha=0.85)

    ax.axhline(0.5, color="#555555", lw=1.0, ls=":", zorder=3,
               label="Random (AUROC=0.500)")

    ax.set_xticks(mapped_xs)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(-0.5, len(mapped_xs) - 0.5)
    
    ax.set_xlabel("Training Samples", fontsize=12)
    ax.set_ylabel("Test Macro-AUROC", fontsize=12)
    ax.set_title("ECGFounder FT vs Net1D Scratch Ablation - Pediatric ECG",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.set_ylim(0.48, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "05_net1d_ablation.png"))
    plt.close(fig)
    print("[OK] Plot 5 - Net1D ablation")


# ===========================================================================
# Plot 6 - ECGFounder FT vs best baselines (pediatric) — all 4 sizes
# ===========================================================================
def plot_6():
    sizes  = [100, 500, 1000, 3000, n_fu]
    xlbls  = ["100", "500", "1000", "3000", f"Full\n({n_fu})"]
    pv     = [p100, p500,  p1k,   p3k,   pfu]
    rv     = [resn_bests[100], resn_bests[500],  resn_bests[1000],  resn_bests[3000],  resn_bests["full"]]
    nv     = [net1d_bests[100], net1d_bests[500], net1d_bests[1000], net1d_bests[3000], net1d_bests["full"]]

    x = np.arange(len(sizes))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))

    b1 = ax.bar(x - w, pv, w, label="ECGFounder FT",        color=PC,       zorder=3)
    b2 = ax.bar(x,     rv, w, label="ResNet (best scratch)", color="#2ca02c", alpha=0.85, zorder=3)
    b3 = ax.bar(x + w, nv, w, label="Net1D (best scratch)",  color="#ff7f0e", alpha=0.85, zorder=3)

    ax.axhline(pz, color=PC, linestyle="--", lw=1.5, label=f"ECGFounder zero-shot ({pz:.3f})")

    for bar_x, vals in [(x - w, pv), (x, rv), (x + w, nv)]:
        for bxi, v in zip(bar_x, vals):
            if not np.isnan(v):
                ax.text(bxi, v + 0.002, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(xlbls)
    ax.set_xlabel("Training Samples", fontsize=12)
    ax.set_ylabel("Test Macro-AUROC", fontsize=12)
    ax.set_title("ECGFounder FT vs Scratch Baselines - Pediatric ECG (ZU pECG)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    # Start y-axis just below the lowest visible value
    all_vals = [v for v in pv + rv + nv if not np.isnan(v)]
    ax.set_ylim(0.5, 1.00)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "06_founder_vs_baselines.png"))
    plt.close(fig)
    print("[OK] Plot 6 - ECGFounder vs baselines")


# ===========================================================================
# Plot 7 - Heatmap: macro-AUROC across all modes x datasets
# ===========================================================================
def plot_7():
    rows = []
    for _, r in cinc.iterrows():
        key = f"{r['mode']} ({int(r['n_samples'])})" if r["n_samples"] > 0 else "zero_shot"
        rows.append({"Key": key, "Dataset": r["dataset"], "AUROC": r["macro_auroc"]})
    df_h  = pd.DataFrame(rows)
    pivot = df_h.pivot_table(index="Key", columns="Dataset", values="AUROC")

    order = (["zero_shot"] +
             [f"fine_tune ({n})" for n in sorted(ft["n_samples"].unique())] +
             [f"scratch ({n})"   for n in sorted(sc["n_samples"].unique())])
    pivot = pivot.reindex([r for r in order if r in pivot.index])

    ped_c = {
        "zero_shot":      pz,
        "fine_tune (100)": p100,
        "fine_tune (500)": p500,
        "fine_tune (1000)": p1k,
        "fine_tune (3000)": p3k,
    }
    pivot["Pediatric"] = pd.Series(ped_c)

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.55, vmax=1.0, linewidths=0.4, ax=ax,
                cbar_kws={"label": "Macro-AUROC"})
    ax.set_title("Macro-AUROC Heatmap: All Modes x All Datasets",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Mode (n_samples)", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "07_heatmap.png"))
    plt.close(fig)
    print("[OK] Plot 7 - Heatmap")


# ===========================================================================
# Plot 8 - 2x3 Per-dataset subplot: ECGFounder FT vs ResNet Scratch
#          (inspired by reference image; zero-shot shown as red dot at x=0)
# ===========================================================================
def plot_8():
    # Build ResNet scratch data per dataset (adult only; use the CINC scratch rows)
    # For pediatric we use the ablation_resnet file
    datasets_order = ["Chapman", "CPSC", "G12EC", "PTB", "SPH", "Pediatric"]

    # x-tick setup: 0, 100, 500, 1000, 3000, full
    xtick_vals   = [0, 100, 500, 1000, 3000, 10000]
    xtick_labels = ["0", "100", "500", "1k", "3k", "Full"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
    axes = axes.flatten()

    ft_color  = "#1f77b4"   # blue
    sc_color  = "#ff7f0e"   # orange
    zs_color  = "#d62728"   # red dot

    for idx, ds in enumerate(datasets_order):
        ax = axes[idx]

        if ds == "Pediatric":
            # ECGFounder FT line (zero-shot at x=0)
            ft_xs = [0,   100,  500,  1000, 3000, n_fu]
            ft_ys = [pz, p100, p500,  p1k,  p3k,  pfu]
            # ResNet best scratch (from ablation file) — starts at 0.5 for 0 samples
            sc_xs = [0, 100, 500,                1000,                 3000,                  n_fu]
            sc_ys = [0.5, resn_bests[100], resn_bests[500],    resn_bests[1000],     resn_bests[3000],      resn_bests["full"]]
            zs_val = pz
        else:
            sub_ft = ft[ft["dataset"] == ds].sort_values("n_samples")
            sub_sc = sc[sc["dataset"] == ds].sort_values("n_samples")
            zs_val = zs.loc[zs["dataset"] == ds, "macro_auroc"].values[0]

            ft_xs = [0] + list(sub_ft["n_samples"])
            ft_ys = [zs_val] + list(sub_ft["macro_auroc"])
            sc_xs = [0] + list(sub_sc["n_samples"])
            sc_ys = [0.5] + list(sub_sc["macro_auroc"]) # Scratch starts at 0.5 instead of zs_val

        ax.plot(ft_xs, ft_ys, color=ft_color, marker="o", lw=2, ms=5,
                label="ECGFounder (Fine-Tuned)")
        ax.plot(sc_xs, sc_ys, color=sc_color, marker="s", lw=2, ms=5,
                linestyle="--", label="ResNet (Scratch)")
        ax.scatter([0], [zs_val], color=zs_color, s=70, zorder=5,
                   label="Zero-Shot")

        ax.set_title(f"{ds} Performance", fontsize=11, fontweight="bold")
        ax.set_ylim(0.45, 1.05)

        # Use categorical x-axis to avoid log-scale blank space
        used_xs = sorted(set(ft_xs) | set(sc_xs))
        ax.set_xticks(used_xs)
        tick_map = {0: "0", 100: "100", 500: "500", 1000: "1k",
                    3000: "3k", 6000: "6k", 8000: "8k",
                    10000: "10k", n_fu: "Full"}
        ax.set_xticklabels([tick_map.get(v, str(v)) for v in used_xs],
                           fontsize=8, rotation=30, ha="right")
        ax.set_xlim(-300, max(used_xs) + 500)

        if idx in [0, 3]:
            ax.set_ylabel("Macro AUROC", fontsize=10)
        if idx in [3, 4, 5]:
            ax.set_xlabel("Training Samples", fontsize=10)

        if idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("ECGFounder Fine-Tuned vs ResNet Scratch: Per-Dataset Scaling",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "08_per_dataset_scaling.png"), bbox_inches="tight")
    plt.close(fig)
    print("[OK] Plot 8 - Per-dataset 2x3 subplot grid")


# ===========================================================================
if __name__ == "__main__":
    print("Output ->", OUT, "\n")
    plot_1()
    plot_2()
    plot_3()
    plot_4()
    plot_5()
    plot_6()
    plot_7()
    plot_8()
    print("\n[DONE] All 8 plots generated successfully.")
