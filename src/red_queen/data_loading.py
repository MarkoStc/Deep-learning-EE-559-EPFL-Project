"""
Data loading: download datasets from HuggingFace, binarize labels, split.
Supports: civil_comments (primary), hatexplain (secondary).
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_civil_comments(cfg: dict) -> pd.DataFrame:
    """
    Load Civil Comments from HuggingFace datasets.
    Binarize toxicity using the configured threshold.
    Subsample if max_samples is set.
    """
    from datasets import load_dataset

    data_dir = Path(cfg["data_dir"])
    cache_path = data_dir / "civil_comments_processed.parquet"

    # Validate cache: must exist AND have required columns
    cache_valid = False
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if "text" in df.columns and "label" in df.columns:
                cache_valid = True
                print(f"[INFO] Loading cached Civil Comments from {cache_path}")
                print(f"[INFO] Cached columns: {list(df.columns)}")
            else:
                print(
                    f"[WARN] Cache missing required columns (found: {list(df.columns)}). Re-downloading..."
                )
                cache_path.unlink()
        except Exception as e:
            print(f"[WARN] Cache corrupted ({e}). Re-downloading...")
            cache_path.unlink()

    if not cache_valid:
        print("[INFO] Downloading Civil Comments from HuggingFace...")
        ds = load_dataset("google/civil_comments", split="train")
        df = ds.to_pandas()

        print(f"[DEBUG] Raw dataset columns: {list(df.columns)}")
        print(f"[DEBUG] Raw dataset shape: {df.shape}")

        threshold = cfg["dataset"].get("toxicity_threshold", 0.5)

        # The toxicity column may vary by dataset version
        toxicity_col = None
        for candidate in ["toxicity", "severe_toxicity", "score"]:
            if candidate in df.columns:
                toxicity_col = candidate
                break

        if toxicity_col is None:
            # Fallback: look for any float column that could be toxicity
            print(f"[WARN] No 'toxicity' column found. Available: {list(df.columns)}")
            # Try to use the dataset's built-in label if it exists
            if "label" in df.columns:
                print("[INFO] Using existing 'label' column from dataset")
            else:
                raise ValueError(
                    f"Cannot find toxicity or label column. Available columns: {list(df.columns)}"
                )
        else:
            print(f"[INFO] Using '{toxicity_col}' column with threshold {threshold}")
            df["label"] = (df[toxicity_col] >= threshold).astype(int)

        # Keep useful columns
        keep_cols = ["text", "label"]
        if toxicity_col and toxicity_col in df.columns:
            keep_cols.append(toxicity_col)
            if toxicity_col != "toxicity":
                df = df.rename(columns={toxicity_col: "toxicity"})
                keep_cols = ["text", "label", "toxicity"]

        # Add identity columns if they exist (for fairness analysis)
        identity_cols = [
            "male",
            "female",
            "transgender",
            "other_gender",
            "heterosexual",
            "homosexual_gay_or_lesbian",
            "bisexual",
            "other_sexual_orientation",
            "christian",
            "jewish",
            "muslim",
            "hindu",
            "buddhist",
            "atheist",
            "other_religion",
            "black",
            "white",
            "asian",
            "latino",
            "other_race_or_ethnicity",
            "physical_disability",
            "intellectual_or_learning_disability",
            "psychiatric_or_mental_illness",
            "other_disability",
        ]
        for col in identity_cols:
            if col in df.columns:
                keep_cols.append(col)

        df = df[[c for c in keep_cols if c in df.columns]].copy()
        df = df.dropna(subset=["text"]).reset_index(drop=True)

        # Verify label column exists before caching
        assert "label" in df.columns, f"BUG: label column missing. Columns: {list(df.columns)}"
        print(f"[INFO] Label distribution: {df['label'].value_counts().to_dict()}")

        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"[INFO] Cached to {cache_path} (columns: {list(df.columns)})")

    return df


def load_hatexplain(cfg: dict) -> pd.DataFrame:
    """
    Load HateXplain from HuggingFace.
    Map labels: 0=normal, 1=offensive, 2=hate → binarize to 0=safe, 1=hate.
    """
    from datasets import load_dataset

    data_dir = Path(cfg["data_dir"])
    cache_path = data_dir / "hatexplain_processed.parquet"

    if cache_path.exists():
        print(f"[INFO] Loading cached HateXplain from {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        print("[INFO] Downloading HateXplain from HuggingFace...")
        ds = load_dataset("hatexplain")

        records = []
        for split in ["train", "validation", "test"]:
            if split in ds:
                for row in ds[split]:
                    # Majority vote for label
                    from collections import Counter

                    label_counts = Counter(row["annotators"]["label"])
                    majority_label = label_counts.most_common(1)[0][0]
                    text = " ".join(row["post_tokens"])
                    # Binarize: 0=normal→0, 1=offensive→1, 2=hatespeech→1
                    binary_label = 0 if majority_label == 0 else 1
                    records.append(
                        {
                            "text": text,
                            "label": binary_label,
                            "original_label": majority_label,
                            "split": split,
                        }
                    )

        df = pd.DataFrame(records)
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"[INFO] Cached to {cache_path}")

    return df


def prepare_splits(df: pd.DataFrame, cfg: dict):
    """
    Split dataframe into train/val/test.
    Returns dict with keys 'train', 'val', 'test', each a DataFrame.
    """
    max_samples = cfg["dataset"].get("max_samples")
    if max_samples and len(df) > max_samples:
        print(f"[INFO] Subsampling to {max_samples} rows (stratified)")
        # Avoid groupby.apply — pandas 3.x drops the grouped column
        frac = max_samples / len(df)
        sampled_parts = []
        for lbl in df["label"].unique():
            group = df[df["label"] == lbl]
            n = max(1, int(len(group) * frac))
            sampled_parts.append(group.sample(n=n, random_state=cfg["seed"]))
        df = pd.concat(sampled_parts, ignore_index=True)
        print(f"[INFO] Subsampled to {len(df)} rows. Columns: {list(df.columns)}")

    test_size = cfg["dataset"].get("test_size", 0.15)
    val_size = cfg["dataset"].get("val_size", 0.15)

    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=cfg["seed"]
    )
    # val_size is relative to the original total, adjust for remaining
    val_relative = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_relative, stratify=train_df["label"], random_state=cfg["seed"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"[INFO] Splits — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    print(f"[INFO] Label distribution (train): {train_df['label'].value_counts().to_dict()}")

    return {"train": train_df, "val": val_df, "test": test_df}


def load_and_split(cfg: dict) -> dict:
    """Main entry point: load dataset by name, return splits."""
    name = cfg["dataset"]["name"]
    if name == "civil_comments":
        df = load_civil_comments(cfg)
    elif name == "hatexplain":
        df = load_hatexplain(cfg)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return prepare_splits(df, cfg)
