# =============================================================================
# split_utils.py
# Shared patient-level split logic used by BOTH ablation scripts:
#   - resnet_ablation.py
#   - ft_ablation.py
#
# Importing this module from both scripts guarantees that the train / val /
# test partitions are IDENTICAL, making ResNet vs FT comparisons valid.
#
# Usage:
#   from split_utils import prepare_splits, subsample_train, SEED
# =============================================================================

import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

SEED = 42

logger = logging.getLogger("split_utils")


def prepare_splits(ecg_df: pd.DataFrame, logger=logger):
    """
    Split at PATIENT level so no patient appears in more than one partition.

    Split ratios (applied to unique patients):
        Test  : 20 % of all patients
        Val   : 10 % of remaining (i.e. ~8 % of total)
        Train : remainder          (~72 % of total)

    After splitting, samples with no positive label are removed from every
    partition.

    Hard leakage checks are performed:
        1. Patient-set intersection across all three partition pairs.
        2. Filename-level deduplication across all partitions.

    Raises AssertionError immediately on any detected leakage.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        Patient-disjoint, label-filtered DataFrames ready for Dataset wrappers.
    """
    unique_patients           = ecg_df["Patient_ID"].unique()
    train_pts, test_pts       = train_test_split(
        unique_patients, test_size=0.2, random_state=SEED)
    train_pts, val_pts        = train_test_split(
        train_pts,       test_size=0.1, random_state=SEED)

    train_df = ecg_df[ecg_df["Patient_ID"].isin(train_pts)].reset_index(drop=True)
    val_df   = ecg_df[ecg_df["Patient_ID"].isin(val_pts)].reset_index(drop=True)
    test_df  = ecg_df[ecg_df["Patient_ID"].isin(test_pts)].reset_index(drop=True)

    # Remove samples with no positive label
    train_df = train_df[train_df["label"].apply(
        lambda x: np.array(x).sum() > 0)].reset_index(drop=True)
    val_df   = val_df[val_df["label"].apply(
        lambda x: np.array(x).sum() > 0)].reset_index(drop=True)
    test_df  = test_df[test_df["label"].apply(
        lambda x: np.array(x).sum() > 0)].reset_index(drop=True)

    # ── Hard leakage assertions ───────────────────────────────────────────────
    train_ids = set(train_df["Patient_ID"])
    val_ids   = set(val_df["Patient_ID"])
    test_ids  = set(test_df["Patient_ID"])

    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids   & test_ids

    if tv:
        raise AssertionError(
            f"DATA LEAKAGE: {len(tv)} patient(s) in both TRAIN and VAL: "
            f"{list(tv)[:5]} ...")
    if tt:
        raise AssertionError(
            f"DATA LEAKAGE: {len(tt)} patient(s) in both TRAIN and TEST: "
            f"{list(tt)[:5]} ...")
    if vt:
        raise AssertionError(
            f"DATA LEAKAGE: {len(vt)} patient(s) in both VAL and TEST: "
            f"{list(vt)[:5]} ...")

    # Filename-level check (catches multi-ECG-per-patient edge cases)
    all_files = (list(train_df["Filename"]) +
                 list(val_df["Filename"]) +
                 list(test_df["Filename"]))
    if len(all_files) != len(set(all_files)):
        dupes = [f for f in set(all_files) if all_files.count(f) > 1]
        raise AssertionError(
            f"DATA LEAKAGE: {len(dupes)} file(s) appear in multiple splits: "
            f"{dupes[:5]} ...")

    logger.info("✓ Patient-level split verified — no leakage detected.")
    logger.info(f"  TRAIN : {len(train_df):5d} samples | {len(train_ids):5d} patients")
    logger.info(f"  VAL   : {len(val_df):5d} samples | {len(val_ids):5d} patients")
    logger.info(f"  TEST  : {len(test_df):5d} samples | {len(test_ids):5d} patients")

    return train_df, val_df, test_df


def subsample_train(train_df: pd.DataFrame, n: int, seed: int = SEED,
                    logger=logger) -> pd.DataFrame:
    """
    Randomly subsample `n` records from train_df using a fixed seed.

    The same seed is used regardless of model type, so ResNet and FT
    experiments with the same budget train on exactly the same records.

    If n >= len(train_df), the full DataFrame is returned unchanged.
    """
    if n >= len(train_df):
        logger.info(
            f"  Subsample budget {n} >= available {len(train_df)} — using all.")
        return train_df
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_df), size=n, replace=False)
    sub = train_df.iloc[sorted(idx)].reset_index(drop=True)
    logger.info(
        f"  Subsampled training set: {len(sub)} / {len(train_df)} records")
    return sub