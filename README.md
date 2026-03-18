# Pediatric ECG Classification with ECGFounder

This repository contains code to evaluate, fine-tune, and prune the **ECGFounder** foundation model on pediatric (child) 12-lead ECG datasets. It explores both fully supervised learning (full fine-tuning and linear probing) and zero-shot efficiency pipelines (model pruning and pure inference).

## 🚀 Use Case & Project Overview

The primary use case of this project is to build an automated diagnostic assistant for pediatric-specific electrocardiogram (ECG) data. Child ECGs often differ morphologically from adult ECGs, making adult-trained models potentially less accurate.

To address this, the project utilizes the **ECGFounder** model—a powerful, pre-trained 1D Convolutional Neural Network (Net1D architecture) designed for representation learning on ECG data—and adapts it accurately to detect multiple cardiac conditions from 12-lead children ECGs.

Key features of this repository:
1. **Fine-tuning & Linear Probing**: Fine-tune the pre-trained foundation model on labeled child ECG data. Readily tracks AUROC, AUPRC, Sensitivity, Specificity, and F1 across multiple diagnostic classes (`ft.py`).
2. **Zero-Shot Pruning**: Apply data-free structural pruning (magnitude and random selection methods) on the convolution filters and measure efficiency vs. performance trade-offs (Giga MACs and millions of parameters vs. AUROC drop) (`pruning.py`).
3. **Zero-Shot Mapping & Evaluation**: Detailed notebooks providing mapping of clinical raw AHA codes into the 150-class multi-label space needed by the pre-trained backbone (`zero_shot.ipynb`).

---

## 🧠 Foundation Model & Data

### 1. Foundation Model: ECGFounder
The underlying backbone is **ECGFounder** (`Net1D`), which extracts powerful spatiotemporal representations across the 12 leads. 

**How to download:**
- You need the pre-trained model checkpoint: `12_lead_ECGFounder.pth`.
- Download this file from the official ECGFounder repository or your model provider.
- Place it directly into the root folder of this project (`./12_lead_ECGFounder.pth`).

### 2. Dataset: Child ECG / Zhengzhou ECG
The project is built around the **Child ECG Dataset** (structured as `.dat` and `.hea` files typical for WFDB, alongside patient CSVs).

**How to download & prepare:**
1. Download the Child ECG dataset (e.g., Zhengzhou pediatric dataset).
2. Extract the WFDB format records into a specific directory.
3. Update the `ecg_path` variable at the top of the python scripts (e.g. `ft.py`, `pruning.py`) to point to the local absolute path of the data folders. For example: `C:/Users/.../Zhengzhou_ECG/Child_ecg/`
4. The repository already includes various mapping dictionaries:
   - `ecg_df.csv`: Main dataframe with Patient IDs, file paths, and JSON string arrays representing the labels.
   - `AttributesDictionary.csv`, `ECGCode.csv`, `ecg_label_mapping.csv`: Cross-reference indices used to map raw hospital diagnoses/AHA codes into exactly 150 standardized diagnostic classes.

---

## 📂 Project Structure

- **`ft.py`**: The training script. 
  - **Function**: Handles the full fine-tuning or linear probing (by freezing the backbone using `LINEAR_PROBE = True`) of ECGFounder. Employs `BCEWithLogitsLoss` for multi-label classification.
  - **Output**: Logs metrics, computes bootstrap confidence intervals (CI), optimal F1 thresholds, and saves `.pth` checkpoints to `./res/finetune/`.

- **`pruning.py`**: Zero-shot pruning and efficiency tool.
  - **Function**: Evaluates the model with zero training. Uses `torch_pruning` to cut down the network's filters incrementally (e.g., 10%, 20%, 30%). Great for embedded deployment checks. 
  - **Output**: Spits out `pruning_summary.csv` and outputs reduced model checkpoints to `./res/zeroshot_pruning/`.

- **`zero_shot.ipynb`**: Data processing exploration loop. 
  - **Function**: A comprehensive Jupyter Notebook exploring how to map raw dataset disease descriptors and AHA codes (`C21`, `D37`, etc.) to the 150 discrete classes using specific priority heuristics.

- **`net1d.py`** *(imported)*: Source code for the 1D ResNet-style architecture (the backbone of ECGFounder).

- **`tasks.txt`**: A plaintext dictionary listing out the labels mapped sequentially for the 150 target classification indices.

- **`logging/` & `res/`**: Automatically generated directories where run logs (`.log`) and model output matrices/checkpoints (`.pth`, `.csv` results) are dynamically archived.

---

## 🛠️ Usage

### Setup Environment
Make sure you have standard ML/ECG libraries installed, importantly:
```bash
pip install torch torchvision torchaudio numpy pandas scikit-learn scipy tqdm wfdb torch-pruning
```

### Running Fine-Tuning
Check the variables at the top of `ft.py` to ensure `csv_path`, `ecg_path`, and `pth` are aligned with your machine setup, then execute:
```bash
python ft.py
```

### Running Zero-Shot Pruning
To assess how much the model can be pruned without sacrificing its representation strength on the pediatric dataset:
```bash
python pruning.py
```
Check `./logging/zeroshot_pruning.log` for layer-by-layer parameter drops.