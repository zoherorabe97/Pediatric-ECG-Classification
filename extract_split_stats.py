import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# We use the exact same seed from split_utils.py to ensure identical splits
SEED = 42

def extract_split_stats(csv_path="ecg_with_exact_match.csv", tasks_path="tasks.txt"):
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Convert string representation of list to actual list
    print("Parsing labels...")
    if isinstance(df["label"].iloc[0], str):
        df["label"] = df["label"].apply(json.loads)
        
    # Load tasks (class names)
    print(f"Loading task names from {tasks_path}...")
    with open(tasks_path, "r") as f:
        labels_list = [line.strip() for line in f if line.strip()]
    
    # Split logic (replicated exactly from split_utils.py)
    unique_patients = df["Patient_ID"].unique()
    
    # 20% test
    train_pts, test_pts = train_test_split(unique_patients, test_size=0.2, random_state=SEED)
    # 10% of remaining (i.e. ~8% of total) for val
    train_pts, val_pts = train_test_split(train_pts, test_size=0.1, random_state=SEED)

    # Assign samples to their respective splits based on Patient_ID
    splits = {
        "Train": df[df["Patient_ID"].isin(train_pts)].copy(),
        "Val": df[df["Patient_ID"].isin(val_pts)].copy(),
        "Test": df[df["Patient_ID"].isin(test_pts)].copy()
    }

    print("\n" + "="*50)
    print("SPLIT STATISTICS")
    print("="*50)

    total_valid_samples = 0
    stats = {}

    # First pass to compute total valid samples across all splits (for percentage calculation)
    for name, split_df in splits.items():
        valid_mask = split_df["label"].apply(lambda x: np.sum(x) > 0)
        total_valid_samples += valid_mask.sum()

    # Compute statistics for each split
    for name, split_df in splits.items():
        # Total before filtering missing labels
        total_initial = len(split_df)
        unique_pts = split_df["Patient_ID"].nunique()
        
        # Missing labels (all zeros)
        valid_mask = split_df["label"].apply(lambda x: np.sum(x) > 0)
        missing_labels_count = (~valid_mask).sum()
        
        # Final valid dataset
        valid_df = split_df[valid_mask]
        total_final = len(valid_df)
        percentage = (total_final / total_valid_samples) * 100 if total_valid_samples > 0 else 0
        
        # Label distribution (sum of positives across all samples)
        labels_array = np.stack(valid_df["label"].values)
        label_counts = labels_array.sum(axis=0)
        
        stats[name] = {
            "Total Initially Assigned": total_initial,
            "Unique Patients": unique_pts,
            "Missing Labels (Zero Positives)": missing_labels_count,
            "Final Valid Samples": total_final,
            "Percentage of Total Valid": percentage,
            "Label Distribution": label_counts
        }
        
        print(f"\n[{name} Split]")
        print(f"  - Unique Patients           : {unique_pts}")
        print(f"  - Initially Assigned Samples: {total_initial}")
        print(f"  - Samples w/ Missing Labels : {missing_labels_count} (Dropped in `prepare_splits`)")
        print(f"  - Final Valid Samples       : {total_final}")
        print(f"  - Percentage of Dataset     : {percentage:.2f}%")
        
    print("\n" + "="*50)
    print("LABEL DISTRIBUTION (Top 10 classes per split)")
    print("="*50)
    
    # Display the top 10 most common classes in each split
    for name in splits.keys():
        print(f"\n--- {name} Split ---")
        counts = stats[name]["Label Distribution"]
        dist_df = pd.DataFrame({"Class": labels_list, "Count": counts})
        dist_df = dist_df.sort_values(by="Count", ascending=False).head(10)
        for _, row in dist_df.iterrows():
            print(f"  {row['Class']:<50}: {row['Count']}")

    # Save comprehensive label distribution to a CSV file
    all_dist = {"Class": labels_list}
    for name in splits.keys():
        all_dist[f"{name}_Count"] = stats[name]["Label Distribution"]
        all_dist[f"{name}_Percentage (%)"] = (stats[name]["Label Distribution"] / stats[name]["Final Valid Samples"] * 100).round(2)
        
    dist_df_all = pd.DataFrame(all_dist)
    dist_df_all["Total_Count"] = dist_df_all[[f"{name}_Count" for name in splits.keys()]].sum(axis=1)
    dist_df_all["Total_Percentage (%)"] = (dist_df_all["Total_Count"] / total_valid_samples * 100).round(2)
    dist_df_all = dist_df_all.sort_values(by="Total_Count", ascending=False)
    
    out_csv = "split_label_distribution.csv"
    dist_df_all.to_csv(out_csv, index=False)
    print(f"\nDetailed label distribution across splits saved to '{out_csv}'")

if __name__ == "__main__":
    extract_split_stats()
