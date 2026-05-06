# ZU pECG (Pediatric) Dataset Summary & Label Mapping

This document details the main characteristics of the pediatric ECG dataset (ZU pECG) used for evaluating the ECGFounder foundation model, and the methodology used to map its native clinical labels to the model's standardized pretraining vocabulary.

## 1. Main Characteristics of the Pediatric Data

The ZU pECG dataset represents a distinct physiological domain compared to the adult datasets (like Chapman, CPSC, PTB, etc.) that the ECGFounder model was originally trained on. The main characteristics include:

- **Signal Dimensions & Sampling:** All recordings are standard 12-lead ECGs uniformly sampled at **500 Hz**.
- **Varying Recording Lengths:** Unlike standardized adult datasets (often exactly 10 seconds), the pediatric dataset contains continuous recordings ranging from **5 seconds to 120 seconds** in length. The majority of the samples are approximately **30 seconds** long.
- **Physiological Differences:** Pediatric ECGs inherently differ from adult ECGs due to differences in chest wall thickness, heart size, and autonomic tone. This manifests as:
  - Significantly higher resting heart rates.
  - Different expected signal amplitudes and QRS axis orientations.
  - Narrower baseline QRS and PR intervals.
- **Preprocessing Adaptation:** To evaluate these varying-length signals on the ECGFounder model (which expects 10-second inputs), the signals are truncated (cropped) at 5,000 samples (10 seconds) rather than resampled, preventing disastrous time-domain distortions.

## 2. Label Mapping Methodology

The ECGFounder model is capable of zero-shot classification across 148 specific diagnostic classes (defined in `tasks.txt`). However, the native annotations in the ZU pECG dataset used a different clinical nomenclature. 

To evaluate the model, a rigorous label alignment was performed (documented in `ecg_label_mapping.csv`). The mapping process was divided into three categories:

### A. Exact Keyword Matching
Whenever possible, the pediatric labels were mapped to the ECGFounder tasks using direct keyword matching. This was fully automated and accounted for minor stylistic differences (like hyphenation or capitalization).
*Examples of Exact Matches:*
- `Sinus tachycardia` $\rightarrow$ `SINUS TACHYCARDIA`
- `First-degree AV block` $\rightarrow$ `WITH 1ST DEGREE AV BLOCK`
- `Left bundle-branch block` $\rightarrow$ `LEFT BUNDLE BRANCH BLOCK`

### B. Expert Semantic Matching (Close/Partial Match)
For pediatric labels that had no direct structural or exact keyword equivalent in the adult vocabulary, an **expert-driven semantic mapping** was performed. A clinician or domain expert matched the underlying pathophysiology of the pediatric label to the closest/broadest equivalent label in the `tasks.txt` dictionary. 
*Examples of Semantic Matches:*
- `Prolonged PR interval` $\rightarrow$ `WITH PROLONGED AV CONDUCTION` *(Clinically equivalent)*
- `Junctional premature complex(es)` $\rightarrow$ `PREMATURE SUPRAVENTRICULAR COMPLEXES` *(Best available broad category)*
- `Accelerated atrial autonomous rhythm` $\rightarrow$ `ECTOPIC ATRIAL RHYTHM` *(Functionally equivalent)*
- `Preexcitation syndrome` $\rightarrow$ `WOLFF-PARKINSON-WHITE` *(WPW is the main preexcitation syndrome in the task list)*

### C. Unmapped Labels
Any pediatric labels representing pathologies that were strictly absent from the ECGFounder pretraining taxonomy were intentionally left unmapped to ensure a fair evaluation of the model's capabilities. These samples/labels were excluded from the macro-AUROC calculations.
*Examples of Unmapped Labels:*
- `Osborn wave`
- `Hyperkalemia` / `Hypocalcemia` (Electrolyte imbalances)
- `Ostium primum ASD` (Congenital structural defects)

## Conclusion
This mapping strategy ensures that the zero-shot and fine-tuning evaluations accurately reflect the model's ability to diagnose equivalent cardiac rhythms and conduction abnormalities, while acknowledging the inherent taxonomic mismatch between adult clinical databases and pediatric cardiology.
