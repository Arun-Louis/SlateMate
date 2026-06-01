# train_tfidf_logreg_binary_from_excel_folder.py
# Baseline binary classifier for lecture transcript segments using TF-IDF + Logistic Regression.
# Designed for the same 24 labeled Excel lecture files, with leakage-safe group-wise splitting.

import os
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# =========================
# Configuration
# =========================
INPUT_FOLDER = r"C:\Users\nithi\Downloads\SlateMate\excel_labeled_dataset"
OUTPUT_DIR = "./tfidf_logreg_binary_outputs"

MAX_FEATURES = 20000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95

RANDOM_SEED = 42
SPLIT_MODE = "group"  # Default is leakage-safe group split; use "random" only for comparison experiments

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DROP_DUPLICATES = True
LOWERCASE = True
USE_CLASS_WEIGHT_BALANCED = True

TEXT_COLUMN_CANDIDATES = [
    "text",
    "segment_text",
    "transcript",
    "chunk_text",
    "utterance",
    "content",
    "sentence",
    "value",
]
LABEL_COLUMN_CANDIDATES = [
    "label",
    "class",
    "target",
    "annotation",
    "labels",
    "category",
]
LECTURE_GROUP_CANDIDATES = [
    "lecture_id",
    "lecture",
    "video_id",
    "source_lecture",
    "file_id",
]

LABEL_MAP = {
    "NO_VISUAL": 0,
    "VISUAL": 1,
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}


# =========================
# Reproducibility
# =========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# =========================
# File / JSON Helpers
# =========================
def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_text(text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


# =========================
# Column Detection Helpers
# =========================
def normalize_column_name(name: str) -> str:
    return str(name).strip().lower()


def detect_text_column(columns) -> str:
    normalized = {normalize_column_name(c): c for c in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]

    for c in columns:
        col_name = normalize_column_name(c)
        if "text" in col_name or "transcript" in col_name or "chunk" in col_name or "segment" in col_name:
            return c
    return None


def detect_label_column(columns) -> str:
    normalized = {normalize_column_name(c): c for c in columns}
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]

    for c in columns:
        col_name = normalize_column_name(c)
        if "label" in col_name or "target" in col_name or "class" in col_name or "annot" in col_name:
            return c
    return None


def detect_group_column(columns) -> str:
    normalized = {normalize_column_name(c): c for c in columns}
    for candidate in LECTURE_GROUP_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


# =========================
# Data Cleaning
# =========================
def normalize_label(value):
    if pd.isna(value):
        return None
    value = str(value).strip().upper()
    value = value.replace("-", "_").replace(" ", "_")
    return value


def clean_text_value(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if LOWERCASE:
        text = text.lower()
    text = " ".join(text.split())
    return text


def clean_dataframe(df: pd.DataFrame, text_col: str, label_col: str, source_file: str, group_col: str = None):
    original_rows = len(df)

    working = df.copy()
    working["text"] = working[text_col].apply(clean_text_value)
    working["label_raw"] = working[label_col].apply(normalize_label)

    if group_col is not None and group_col in working.columns:
        working["lecture_id"] = working[group_col].astype(str).fillna("").apply(lambda x: str(x).strip())
        working["lecture_id"] = working["lecture_id"].replace({"": source_file})
    else:
        working["lecture_id"] = source_file

    working["source_file"] = source_file

    invalid_text_mask = (
        working["text"].isna()
        | (working["text"].astype(str).str.strip() == "")
        | (working["text"].astype(str).str.lower() == "nan")
    )
    invalid_label_mask = working["label_raw"].isna() | (~working["label_raw"].isin(LABEL_MAP.keys()))

    dropped_missing_text = int(invalid_text_mask.sum())
    dropped_invalid_label = int(invalid_label_mask.sum())

    working = working[~invalid_text_mask].copy()
    working = working[~invalid_label_mask].copy()

    working["label"] = working["label_raw"].map(LABEL_MAP).astype(int)
    working["label_name"] = working["label"].map(ID2LABEL)

    dropped_duplicates = 0
    if DROP_DUPLICATES:
        before_dup = len(working)
        working = working.drop_duplicates(subset=["text", "label", "lecture_id"])
        dropped_duplicates = before_dup - len(working)

    usable_after = len(working)
    dropped_rows = original_rows - usable_after

    file_stats = {
        "source_file": source_file,
        "rows_loaded": int(original_rows),
        "usable_rows": int(usable_after),
        "dropped_rows": int(dropped_rows),
        "dropped_missing_text": dropped_missing_text,
        "dropped_invalid_label": dropped_invalid_label,
        "dropped_duplicates": int(dropped_duplicates),
    }

    return working, file_stats


# =========================
# File Loading
# =========================
def load_excel_files(input_folder: str):
    input_path = Path(input_folder)
    if not input_path.exists():
        raise FileNotFoundError(f"INPUT_FOLDER does not exist: {input_folder}")

    excel_files = sorted(list(input_path.glob("*.xlsx")) + list(input_path.glob("*.xls")))
    if not excel_files:
        raise FileNotFoundError(f"No .xlsx or .xls files found in: {input_folder}")

    cleaned_dfs = []
    file_reports = []
    skipped_files = []

    for file_path in excel_files:
        source_file = file_path.name
        try:
            if file_path.suffix.lower() == ".xlsx":
                df = pd.read_excel(file_path, engine="openpyxl")
            else:
                df = pd.read_excel(file_path)
        except Exception as e:
            warnings.warn(f"[SKIP] Could not read file {source_file}: {e}")
            skipped_files.append({"source_file": source_file, "reason": f"read_error: {str(e)}"})
            continue

        text_col = detect_text_column(df.columns)
        label_col = detect_label_column(df.columns)
        group_col = detect_group_column(df.columns)

        if text_col is None or label_col is None:
            reason = (
                f"missing required columns. "
                f"text_col={text_col}, label_col={label_col}, available_columns={list(df.columns)}"
            )
            warnings.warn(f"[SKIP] {source_file}: {reason}")
            skipped_files.append({"source_file": source_file, "reason": reason})
            continue

        cleaned_df, file_stats = clean_dataframe(
            df=df,
            text_col=text_col,
            label_col=label_col,
            source_file=source_file,
            group_col=group_col,
        )

        file_stats["detected_text_column"] = text_col
        file_stats["detected_label_column"] = label_col
        file_stats["detected_group_column"] = group_col if group_col is not None else "source_file_fallback"

        print(
            f"[FILE] {source_file} | rows_loaded={file_stats['rows_loaded']} "
            f"| usable_rows={file_stats['usable_rows']} | dropped_rows={file_stats['dropped_rows']}"
        )

        if len(cleaned_df) > 0:
            cleaned_dfs.append(cleaned_df)
            file_reports.append(file_stats)
        else:
            warnings.warn(f"[SKIP] {source_file}: no usable rows after cleaning.")
            skipped_files.append({"source_file": source_file, "reason": "no usable rows after cleaning"})

    if not cleaned_dfs:
        raise ValueError("No usable data found after loading and cleaning Excel files.")

    combined_df = pd.concat(cleaned_dfs, ignore_index=True)
    return combined_df, pd.DataFrame(file_reports), pd.DataFrame(skipped_files)


def combine_datasets(df: pd.DataFrame):
    total_rows_before = len(df)
    duplicate_count = 0

    if DROP_DUPLICATES:
        before = len(df)
        df = df.drop_duplicates(subset=["text", "label", "lecture_id"])
        duplicate_count = before - len(df)

    total_rows_after = len(df)
    class_distribution = df["label_name"].value_counts(dropna=False).to_dict()

    print("\n[COMBINED DATASET SUMMARY]")
    print(f"Files loaded: {df['source_file'].nunique()}")
    print(f"Unique lecture groups: {df['lecture_id'].nunique()}")
    print(f"Total rows before final cleaning: {total_rows_before}")
    print(f"Total rows after final cleaning: {total_rows_after}")
    print(f"Duplicate rows removed in final combine step: {duplicate_count}")
    print(f"Final class distribution: {class_distribution}")

    return df, {
        "files_loaded": int(df["source_file"].nunique()),
        "unique_lecture_groups": int(df["lecture_id"].nunique()),
        "total_rows_before_final_clean": int(total_rows_before),
        "total_rows_after_final_clean": int(total_rows_after),
        "duplicate_rows_removed_final": int(duplicate_count),
        "class_distribution": class_distribution,
    }


# =========================
# Group Split Utilities
# =========================
def get_group_label_counts(df: pd.DataFrame, group_col: str = "lecture_id"):
    counts = (
        df.groupby([group_col, "label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "NO_VISUAL", 1: "VISUAL"})
        .reset_index()
    )
    if "NO_VISUAL" not in counts.columns:
        counts["NO_VISUAL"] = 0
    if "VISUAL" not in counts.columns:
        counts["VISUAL"] = 0

    counts["total"] = counts["NO_VISUAL"] + counts["VISUAL"]
    return counts


def summarize_split(df: pd.DataFrame, split_name: str, group_col: str = "lecture_id"):
    class_counts = df["label_name"].value_counts().to_dict()
    return {
        "split_name": split_name,
        "rows": int(len(df)),
        "unique_groups": int(df[group_col].nunique()),
        "groups": sorted(df[group_col].astype(str).unique().tolist()),
        "source_files": sorted(df["source_file"].astype(str).unique().tolist()),
        "class_distribution": class_counts,
    }


def assert_group_disjoint(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, group_col: str = "lecture_id"):
    train_groups = set(train_df[group_col].astype(str).unique())
    val_groups = set(val_df[group_col].astype(str).unique())
    test_groups = set(test_df[group_col].astype(str).unique())

    inter_train_val = train_groups.intersection(val_groups)
    inter_train_test = train_groups.intersection(test_groups)
    inter_val_test = val_groups.intersection(test_groups)

    assert len(inter_train_val) == 0, f"Leakage detected between train and val groups: {inter_train_val}"
    assert len(inter_train_test) == 0, f"Leakage detected between train and test groups: {inter_train_test}"
    assert len(inter_val_test) == 0, f"Leakage detected between val and test groups: {inter_val_test}"


def greedy_group_split(df: pd.DataFrame, seed: int = 42, group_col: str = "lecture_id"):
    rng = random.Random(seed)

    group_stats = get_group_label_counts(df, group_col=group_col)
    group_records = group_stats.to_dict("records")
    rng.shuffle(group_records)
    group_records = sorted(group_records, key=lambda x: x["total"], reverse=True)

    total_visual = int(group_stats["VISUAL"].sum())
    total_no_visual = int(group_stats["NO_VISUAL"].sum())
    total_rows = int(group_stats["total"].sum())

    targets = {
        "train": {
            "rows": total_rows * TRAIN_RATIO,
            "VISUAL": total_visual * TRAIN_RATIO,
            "NO_VISUAL": total_no_visual * TRAIN_RATIO,
        },
        "val": {
            "rows": total_rows * VAL_RATIO,
            "VISUAL": total_visual * VAL_RATIO,
            "NO_VISUAL": total_no_visual * VAL_RATIO,
        },
        "test": {
            "rows": total_rows * TEST_RATIO,
            "VISUAL": total_visual * TEST_RATIO,
            "NO_VISUAL": total_no_visual * TEST_RATIO,
        },
    }

    assignments = {"train": [], "val": [], "test": []}
    current = {
        "train": {"rows": 0, "VISUAL": 0, "NO_VISUAL": 0},
        "val": {"rows": 0, "VISUAL": 0, "NO_VISUAL": 0},
        "test": {"rows": 0, "VISUAL": 0, "NO_VISUAL": 0},
    }

    def placement_score(split_name, record):
        future_rows = current[split_name]["rows"] + record["total"]
        future_visual = current[split_name]["VISUAL"] + record["VISUAL"]
        future_no_visual = current[split_name]["NO_VISUAL"] + record["NO_VISUAL"]

        row_score = abs(future_rows - targets[split_name]["rows"]) / max(targets[split_name]["rows"], 1.0)
        vis_score = abs(future_visual - targets[split_name]["VISUAL"]) / max(targets[split_name]["VISUAL"], 1.0)
        novis_score = abs(future_no_visual - targets[split_name]["NO_VISUAL"]) / max(targets[split_name]["NO_VISUAL"], 1.0)

        split_size_penalty = 0.0
        if len(assignments[split_name]) == 0:
            split_size_penalty -= 0.05

        return row_score + vis_score + novis_score + split_size_penalty

    for record in group_records:
        candidate_scores = {split_name: placement_score(split_name, record) for split_name in ["train", "val", "test"]}
        best_split = min(candidate_scores, key=candidate_scores.get)
        assignments[best_split].append(record[group_col])
        current[best_split]["rows"] += record["total"]
        current[best_split]["VISUAL"] += record["VISUAL"]
        current[best_split]["NO_VISUAL"] += record["NO_VISUAL"]

    for split_name in ["val", "test"]:
        if len(assignments[split_name]) == 0:
            donor = max(["train", "val", "test"], key=lambda k: len(assignments[k]))
            if donor != split_name and len(assignments[donor]) > 1:
                moved_group = assignments[donor].pop()
                assignments[split_name].append(moved_group)

    train_df = df[df[group_col].isin(assignments["train"])].copy()
    val_df = df[df[group_col].isin(assignments["val"])].copy()
    test_df = df[df[group_col].isin(assignments["test"])].copy()

    assert len(train_df) > 0, "Train split is empty."
    assert len(val_df) > 0, "Validation split is empty."
    assert len(test_df) > 0, "Test split is empty."
    assert_group_disjoint(train_df, val_df, test_df, group_col=group_col)

    return train_df, val_df, test_df


def prepare_splits(df: pd.DataFrame, split_mode: str = "group", seed: int = 42):
    """
    Prepare train/validation/test splits.

    Important:
    - The 24 Excel files represent separate lectures.
    - To reduce lecture-level data leakage, the recommended and default mode is "group".
    - In group mode, entire lectures stay together in exactly one split.
    - Grouping uses lecture_id when available, otherwise source_file is used as a fallback.
    - This ensures segments from the same lecture do not appear in both train and test.
    """
    if split_mode not in {"group", "random"}:
        raise ValueError("SPLIT_MODE must be either 'group' or 'random'.")

    if split_mode == "random":
        train_df, temp_df = train_test_split(
            df,
            test_size=(1.0 - TRAIN_RATIO),
            stratify=df["label"],
            random_state=seed,
        )

        relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_ratio,
            stratify=temp_df["label"],
            random_state=seed,
        )
        grouping_column_used = "row_level_random_split"
    else:
        train_df, val_df, test_df = greedy_group_split(df, seed=seed, group_col="lecture_id")
        grouping_column_used = "lecture_id (with source_file fallback)"

    print("\n[SPLIT SUMMARY]")
    for split_name, split_df in [("TRAIN", train_df), ("VALIDATION", val_df), ("TEST", test_df)]:
        print(
            f"{split_name}: rows={len(split_df)} | "
            f"unique_groups={split_df['lecture_id'].nunique()} | "
            f"class_distribution={split_df['label_name'].value_counts().to_dict()}"
        )

    if split_mode == "group":
        print("\n[GROUP ASSIGNMENTS]")
        print(f"TRAIN groups: {sorted(train_df['lecture_id'].astype(str).unique().tolist())}")
        print(f"VALIDATION groups: {sorted(val_df['lecture_id'].astype(str).unique().tolist())}")
        print(f"TEST groups: {sorted(test_df['lecture_id'].astype(str).unique().tolist())}")

    split_summary = {
        "split_mode": split_mode,
        "grouping_column_used": grouping_column_used,
        "train": summarize_split(train_df, "train"),
        "validation": summarize_split(val_df, "validation"),
        "test": summarize_split(test_df, "test"),
    }

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True), split_summary


# =========================
# Metrics / Evaluation
# =========================
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_predictions(y_true, y_pred):
    metrics = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        target_names=[ID2LABEL[0], ID2LABEL[1]],
        digits=4,
        zero_division=0,
    )
    return metrics, cm, report


def make_prediction_dataframe(df_split, probs):
    result_df = df_split.copy().reset_index(drop=True)
    result_df["prob_no_visual"] = probs[:, 0]
    result_df["prob_visual"] = probs[:, 1]
    result_df["confidence"] = probs.max(axis=1)
    result_df["predicted_label_id"] = np.argmax(probs, axis=1)
    result_df["predicted_label"] = result_df["predicted_label_id"].map(ID2LABEL)
    result_df["true_label"] = result_df["label"].map(ID2LABEL)
    result_df["is_correct"] = (result_df["predicted_label_id"] == result_df["label"]).astype(int)

    columns_to_save = [
        "text",
        "true_label",
        "predicted_label",
        "confidence",
        "prob_visual",
        "prob_no_visual",
        "source_file",
        "lecture_id",
        "is_correct",
    ]
    extra_cols = [c for c in result_df.columns if c not in columns_to_save]
    return result_df[columns_to_save + extra_cols]


# =========================
# Per-File Analysis
# =========================
def generate_file_summary(full_df, test_predictions_df=None):
    summary = (
        full_df.groupby("source_file")
        .agg(
            total_segments=("text", "count"),
            visual_count=("label", lambda x: int((x == 1).sum())),
            no_visual_count=("label", lambda x: int((x == 0).sum())),
        )
        .reset_index()
    )
    summary["percent_visual"] = (summary["visual_count"] / summary["total_segments"]) * 100.0

    if test_predictions_df is not None and len(test_predictions_df) > 0:
        test_file_metrics = (
            test_predictions_df.groupby("source_file")
            .agg(
                test_segments_in_file=("text", "count"),
                correct_predictions=("is_correct", "sum"),
            )
            .reset_index()
        )
        test_file_metrics["file_accuracy"] = (
            test_file_metrics["correct_predictions"] / test_file_metrics["test_segments_in_file"]
        )
        summary = summary.merge(test_file_metrics, on="source_file", how="left")
    else:
        summary["test_segments_in_file"] = np.nan
        summary["correct_predictions"] = np.nan
        summary["file_accuracy"] = np.nan

    return summary.sort_values("source_file").reset_index(drop=True)


# =========================
# Training / Inference
# =========================
def train_tfidf_logreg(train_df: pd.DataFrame):
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        lowercase=LOWERCASE,
        strip_accents="unicode",
    )

    X_train = vectorizer.fit_transform(train_df["text"].tolist())
    y_train = train_df["label"].values

    model = LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=2000,
        class_weight="balanced" if USE_CLASS_WEIGHT_BALANCED else None,
        solver="liblinear",
    )
    model.fit(X_train, y_train)

    return vectorizer, model


def predict_with_model(df_split: pd.DataFrame, vectorizer, model):
    X = vectorizer.transform(df_split["text"].tolist())
    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=1)
    return preds, probs


def predict_label(text, vectorizer, model):
    clean_text = clean_text_value(text)
    X = vectorizer.transform([clean_text])
    probs = model.predict_proba(X)[0]
    pred_id = int(np.argmax(probs))
    pred_label = ID2LABEL[pred_id]
    confidence = float(np.max(probs))

    return {
        "predicted_label": pred_label,
        "confidence": confidence,
        "prob_no_visual": float(probs[0]),
        "prob_visual": float(probs[1]),
    }


def save_top_features(vectorizer, model, output_path, top_k=100):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    top_visual_idx = np.argsort(coefs)[-top_k:][::-1]
    top_no_visual_idx = np.argsort(coefs)[:top_k]

    top_visual_df = pd.DataFrame({
        "feature": feature_names[top_visual_idx],
        "coefficient": coefs[top_visual_idx],
        "direction": "VISUAL",
    })

    top_no_visual_df = pd.DataFrame({
        "feature": feature_names[top_no_visual_idx],
        "coefficient": coefs[top_no_visual_idx],
        "direction": "NO_VISUAL",
    })

    feature_df = pd.concat([top_visual_df, top_no_visual_df], ignore_index=True)
    feature_df.to_csv(output_path, index=False)
    print(f"[INFO] Saved top feature weights to: {output_path}")


# =========================
# Main
# =========================
def main():
    set_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[INFO] Starting TF-IDF + Logistic Regression baseline pipeline...")
    print(f"[INFO] INPUT_FOLDER={INPUT_FOLDER}")
    print(f"[INFO] OUTPUT_DIR={OUTPUT_DIR}")
    print(f"[INFO] SPLIT_MODE={SPLIT_MODE}")

    config = {
        "INPUT_FOLDER": INPUT_FOLDER,
        "OUTPUT_DIR": OUTPUT_DIR,
        "MAX_FEATURES": MAX_FEATURES,
        "NGRAM_RANGE": list(NGRAM_RANGE),
        "MIN_DF": MIN_DF,
        "MAX_DF": MAX_DF,
        "RANDOM_SEED": RANDOM_SEED,
        "SPLIT_MODE": SPLIT_MODE,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VAL_RATIO": VAL_RATIO,
        "TEST_RATIO": TEST_RATIO,
        "DROP_DUPLICATES": DROP_DUPLICATES,
        "LOWERCASE": LOWERCASE,
        "USE_CLASS_WEIGHT_BALANCED": USE_CLASS_WEIGHT_BALANCED,
        "LABEL_MAP": LABEL_MAP,
    }
    save_json(config, os.path.join(OUTPUT_DIR, "config.json"))

    combined_df, file_report_df, skipped_files_df = load_excel_files(INPUT_FOLDER)
    combined_df, combine_summary = combine_datasets(combined_df)

    file_report_df.to_csv(os.path.join(OUTPUT_DIR, "file_loading_report.csv"), index=False)
    if len(skipped_files_df) > 0:
        skipped_files_df.to_csv(os.path.join(OUTPUT_DIR, "skipped_files_report.csv"), index=False)

    train_df, val_df, test_df, split_summary = prepare_splits(
        combined_df,
        split_mode=SPLIT_MODE,
        seed=RANDOM_SEED,
    )
    save_json(split_summary, os.path.join(OUTPUT_DIR, "split_summary.json"))

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_split.csv"), index=False)

    vectorizer, model = train_tfidf_logreg(train_df)

    # Validation
    val_preds, val_probs = predict_with_model(val_df, vectorizer, model)
    val_metrics, val_cm, val_report = evaluate_predictions(val_df["label"].values, val_preds)

    print("\n[VALIDATION RESULTS]")
    print(json.dumps(val_metrics, indent=2))
    print("Validation Confusion Matrix:")
    print(val_cm)
    print("Validation Classification Report:")
    print(val_report)

    save_json(val_metrics, os.path.join(OUTPUT_DIR, "val_metrics.json"))
    pd.DataFrame(
        val_cm,
        index=[ID2LABEL[0], ID2LABEL[1]],
        columns=[ID2LABEL[0], ID2LABEL[1]],
    ).to_csv(os.path.join(OUTPUT_DIR, "val_confusion_matrix.csv"))
    save_text(val_report, os.path.join(OUTPUT_DIR, "val_classification_report.txt"))

    val_predictions_df = make_prediction_dataframe(val_df, val_probs)
    val_predictions_df.to_csv(os.path.join(OUTPUT_DIR, "val_predictions.csv"), index=False)

    # Test
    test_preds, test_probs = predict_with_model(test_df, vectorizer, model)
    test_metrics, test_cm, test_report = evaluate_predictions(test_df["label"].values, test_preds)

    print("\n[TEST RESULTS]")
    print(json.dumps(test_metrics, indent=2))
    print("Test Confusion Matrix:")
    print(test_cm)
    print("Test Classification Report:")
    print(test_report)

    save_json(test_metrics, os.path.join(OUTPUT_DIR, "test_metrics.json"))
    pd.DataFrame(
        test_cm,
        index=[ID2LABEL[0], ID2LABEL[1]],
        columns=[ID2LABEL[0], ID2LABEL[1]],
    ).to_csv(os.path.join(OUTPUT_DIR, "test_confusion_matrix.csv"))
    save_text(test_report, os.path.join(OUTPUT_DIR, "test_classification_report.txt"))

    test_predictions_df = make_prediction_dataframe(test_df, test_probs)
    test_predictions_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

    # Per-file summary
    dataset_summary = generate_file_summary(combined_df, test_predictions_df=test_predictions_df)
    dataset_summary.to_csv(os.path.join(OUTPUT_DIR, "per_file_summary.csv"), index=False)

    overall_summary = {
        "combine_summary": combine_summary,
        "split_summary": split_summary,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(overall_summary, os.path.join(OUTPUT_DIR, "overall_summary.json"))

    # Save model artifacts in simple reusable form
    try:
        import joblib
        joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib"))
        joblib.dump(model, os.path.join(OUTPUT_DIR, "logistic_regression_model.joblib"))
        print("[INFO] Saved model artifacts with joblib.")
    except Exception as e:
        warnings.warn(f"Could not save joblib artifacts: {e}")

    save_top_features(
        vectorizer=vectorizer,
        model=model,
        output_path=os.path.join(OUTPUT_DIR, "top_feature_weights.csv"),
        top_k=100,
    )

    # Example predictions
    print("\n[EXAMPLE PREDICTIONS]")
    sample_texts = [
        "Let us derive the quadratic formula step by step.",
        "Now we move to the next topic.",
        "This graph shows the relationship between pressure and volume.",
        "Please submit the assignment by Friday.",
        "Observe this triangle carefully.",
    ]

    example_predictions = []
    for text in sample_texts:
        pred = predict_label(text=text, vectorizer=vectorizer, model=model)
        row = {"text": text, **pred}
        example_predictions.append(row)
        print(
            f"Text: {text}\n"
            f"Predicted: {pred['predicted_label']} | "
            f"Confidence: {pred['confidence']:.4f} | "
            f"P(NO_VISUAL): {pred['prob_no_visual']:.4f} | "
            f"P(VISUAL): {pred['prob_visual']:.4f}\n"
        )

    pd.DataFrame(example_predictions).to_csv(
        os.path.join(OUTPUT_DIR, "example_predictions.csv"),
        index=False,
    )

    print("[INFO] TF-IDF + Logistic Regression baseline finished successfully.")
    print(f"[INFO] All outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
