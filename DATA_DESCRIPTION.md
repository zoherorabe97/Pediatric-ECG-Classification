# ECGFounder Pediatric Evaluation: Experimental Data & Plot Descriptions

This document outlines the evaluation datasets, experiment configurations, and resulting visualizations comparing the ECGFounder foundation model performance on adult datasets against a pediatric dataset (ZU pECG).

## 1. Experimental Data Description

The experiments generated several CSV summary files under `cinc_res/` (for the adult CinC challenge datasets) and `res/` (for the pediatric dataset).

### 1.1 Adult Benchmark Data (`cinc_res/`)
Contains 5 CSV files for adult datasets:
- `Chapman_aggregate_summary.csv`
- `CPSC_aggregate_summary.csv`
- `G12EC_aggregate_summary.csv`
- `PTB_aggregate_summary.csv`
- `SPH_aggregate_summary.csv`

**Data Structure & Rows:**
Each file aggregates the results for one specific adult dataset across different training modes and sample sizes. The rows include:
- `zero_shot` mode (always 0 samples).
- `fine_tune` mode evaluated at `100`, `500`, `1000`, and `3000` training samples.
- `scratch` mode evaluated at `100`, `500`, `1000`, and `3000` training samples.

**Key Columns & Evaluation Methods:**
- `n_labels_reported`, `n_labels_macro_auc`: Total diagnostic classes, and the subset of classes with both positive and negative examples (required for valid AUROC calculation).
- `mean_sensitivity`, `mean_specificity`, `mean_f1`, `mean_ppv`, `mean_npv`: Macro-averaged metrics calculated by finding the optimal threshold for each class independently (via F1 optimization) and averaging the metrics across all valid classes.
- `mean_auroc`, `mean_auprc`: The arithmetic mean of the AUROC and AUPRC computed independently for each valid label.
- `macro_auroc`, `macro_auprc`: The primary evaluation metrics. `macro_auroc` is calculated strictly by evaluating `roc_auc_score` for each individual valid class and taking the unweighted average across all classes (as implemented in `compute_macro_auroc`).
- `micro_auroc`, `micro_auprc`: Micro-averaged metrics computed globally by flattening predictions and ground truths across all valid classes.
- `mode`: Training paradigm (`zero_shot`, `fine_tune`, or `scratch`).
- `n_samples`: Number of training samples used.

### 1.2 Pediatric Data (`res/`)
Contains ablation and baseline files for the ZU pECG pediatric dataset:

**A. ECGFounder Fine-Tuning (`res/ablation_ft/ft_ablation_summary_with_zeroshot.csv`)**
- `experiment`: e.g., `ft_zeroshot`, `ft_500`, `ft_full`.
- `n_train_requested`: Requested sample size (e.g., 500, full).
- `n_train_actual`: Actual number of samples used (full=7674).
- `test_macro_AUROC`: Primary performance metric on the test set.

**B. ResNet Scratch Baselines (`res/ablation_resnet/resnet_ablation_summary.csv`)**
- Baseline model (ResNet1d) trained from scratch.
- `model_size`: Network size (`small`, `medium`, `large`).
- `total_params`: Parameter count.
- `n_train_requested`: Training subset size (500, 1000, 3000, full).
- `test_macro_AUROC`: Performance metric.

**C. Net1D Scratch Baselines (`res/ablation_net1d/net1d_scratch_ablation_summary.csv`)**
- Baseline model (Net1D) trained from scratch.
- Same column structure as the ResNet file.

---

## 2. Plot Descriptions and Captions

The plotting script `generate_paper_plots.py` outputs 8 academic-quality plots to `paper_plots/` comparing these datasets. Below are the descriptions and appropriate manuscript captions for each figure.

### Plot 1: Zero-Shot Performance Gap (`01_zeroshot_bar.png`)
**Description:** A bar chart comparing the zero-shot macro-AUROC performance of ECGFounder on the 5 adult datasets versus the pediatric dataset.
**Caption:** **Figure 1: Zero-Shot Performance Gap.** Macro-AUROC of the ECGFounder model under zero-shot inference across five adult datasets (Chapman, CPSC, G12EC, PTB, SPH) and the pediatric dataset (ZU pECG). The dashed line represents the adult average AUROC. A noticeable domain gap is present, with the pediatric dataset underperforming the adult average.

### Plot 2: Fine-Tuning Scaling (`02_ft_scaling.png`)
**Description:** A line chart showing macro-AUROC as a function of the number of fine-tuning samples for all datasets. The y-intercept (at x=0) represents the zero-shot baseline.
**Caption:** **Figure 2: Fine-Tuning Data Scaling.** The effect of training sample size on macro-AUROC during fine-tuning. The zero-shot performance is plotted at 0 samples. The pediatric dataset (ZU pECG, square markers) exhibits rapid improvement with minimal fine-tuning, demonstrating the data efficiency of the foundation model when adapting to a new demographic domain.

### Plot 3: Scratch vs Fine-Tuning Gap (`03_scratch_vs_ft_gap.png`)
**Description:** A grouped bar chart comparing adult and pediatric performance at specific data milestones (500, 1k, 3k samples). It contrasts fine-tuning against training from scratch.
**Caption:** **Figure 3: Efficiency of Fine-Tuning vs. Training from Scratch.** Comparison of macro-AUROC between fine-tuned ECGFounder and models trained from scratch, aggregated across adult datasets and compared against the pediatric domain. At severely limited data regimes (e.g., 500 samples), fine-tuning vastly outperforms training from scratch for both adult and pediatric cohorts.

### Plot 4: ResNet Ablation (`04_resnet_ablation.png`)
**Description:** A line chart evaluating the ResNet scratch baseline on the pediatric dataset, varying both the number of training samples and model capacity (small, medium, large).
**Caption:** **Figure 4: ResNet Capacity Ablation on Pediatric ECGs.** Performance of ResNet models trained from scratch on the ZU pECG pediatric dataset. Increasing model capacity (Small to Large) provides diminishing returns, whereas increasing the training data volume yields steady improvements up to the full dataset (7,674 samples).

### Plot 5: Net1D Ablation (`05_net1d_ablation.png`)
**Description:** Similar to Plot 4, but for the Net1D architecture.
**Caption:** **Figure 5: Net1D Capacity Ablation on Pediatric ECGs.** Performance scaling of Net1D baseline models trained from scratch on the pediatric dataset across varying capacities and training sample sizes.

### Plot 6: ECGFounder vs. Baselines (`06_founder_vs_baselines.png`)
**Description:** A grouped bar chart directly comparing ECGFounder fine-tuning against the best-performing ResNet and Net1D models trained from scratch on the pediatric dataset across different sample sizes.
**Caption:** **Figure 6: ECGFounder Fine-Tuning vs. Best Scratch Baselines.** Direct comparison of ECGFounder's fine-tuned macro-AUROC against the best-performing configurations of ResNet and Net1D trained from scratch on the pediatric dataset. The horizontal dashed line denotes ECGFounder's zero-shot performance. Fine-tuning on merely 500 samples matches or exceeds the performance of baseline models trained on over 3,000 samples.

### Plot 7: Mode x Dataset Heatmap (`07_heatmap.png`)
**Description:** A comprehensive heatmap visualizing macro-AUROC scores across all combinations of dataset, training mode (zero-shot, scratch, fine-tune), and sample sizes.
**Caption:** **Figure 7: Macro-AUROC Performance Heatmap.** Comprehensive matrix of evaluation results across all datasets, inference paradigms, and training sample sizes. Darker green indicates superior performance.

### Plot 8: Per-Dataset FT vs Scratch Scaling (`08_per_dataset_scaling.png`)
**Description:** A 2x3 grid of subplots showing, for each dataset independently, the learning curves of ECGFounder Fine-Tuning versus ResNet trained from scratch. The zero-shot performance is marked with a red dot.
**Caption:** **Figure 8: Per-Dataset Fine-Tuning vs. Scratch Convergence.** Individual learning curves for all six datasets comparing ECGFounder fine-tuning (solid line) against a ResNet model trained from scratch (dashed line). The zero-shot performance of ECGFounder is denoted by the red marker at 0 samples. The scratch baseline is initialized at 0.5 AUROC. Across all datasets, fine-tuning exhibits significantly faster convergence and higher asymptotic performance.

### Plot 9: Shared Labels Zero-Shot Comparison (`comparison_all_hospitals.png`)
**Description:** A grouped bar chart generated by `adult_vs_pediatric_comp.py`. It compares the zero-shot AUROC performance for specific, shared diagnostic labels across all 5 adult datasets and the pediatric dataset. 
**Methodology & Confidence Intervals:**
- **Comparison Logic:** The script scans the zero-shot output CSV files (e.g., `G12EC_zero_shot_results.csv`) and extracts the per-class AUROC values for any diagnostic labels that are shared between the pediatric dataset and at least one adult dataset.
- **Bar Values:** Each bar represents the exact zero-shot AUROC score for a specific dataset on a specific diagnostic label. 
- **Error Bars (95% CI):** The error bars depict the 95% Confidence Interval (CI) for the AUROC metric.
- **CI Calculation Method:** The CIs are computed during the evaluation phase inside `ft.py` using a non-parametric bootstrapping method. The script resamples the predicted and true labels 100 times with replacement (`n_resamples=100`). For each resample, the AUROC is recalculated. The final CI bounds are determined by taking the 2.5th and 97.5th percentiles of the resulting AUROC distribution. These bounds are saved as tuples (e.g., `(np.float64(0.81), np.float64(0.92))`) in the result CSVs, which `adult_vs_pediatric_comp.py` parses via regular expressions to render the upper and lower error bars.
**Caption:** **Figure 9: Zero-Shot Performance Comparison Across Shared Labels.** Label-by-label zero-shot AUROC comparison between the pediatric dataset and the five adult benchmark datasets. Error bars represent 95% confidence intervals derived via 100-iteration non-parametric bootstrapping. This highlights label-specific domain gaps, indicating which specific cardiac abnormalities are most impacted by the demographic shift to pediatric ECGs.
