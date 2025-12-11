#!/usr/bin/env python3
"""
combine_nlp_and_audio.py

- Reads translated lyric text files (Turkish & Balkan)
- Runs BERT-based emotion classifier on each lyric
- Loads audio metadata CSV and creates merge keys
- Joins NLP results with audio features
- Computes PCA-derived emotionality (culture-specific, thresholded weights)
- Normalizes emotionality within each culture to [0,1]
- Outputs FINAL_PROJECT_DATASET.csv with columns:
    id, artists, track name, emotionality, emotion_type, emotion_score, culture
"""

import os
import re
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler

# ---------------- CONFIG ----------------
# Folders containing translated lyrics (each file: "Track - Artist.txt" or "Track.txt")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRANSLATED_FOLDERS = {
    "Turkish": os.path.join(BASE_DIR, "../../lyrics_files_turkish_translated"),
    "Balkan":  os.path.join(BASE_DIR, "../../lyrics_files_balkan_translated")
}

# Path to audio CSV (must contain at least: track_name, artists, and audio feature columns)
AUDIO_CSV_PATH = os.path.join(BASE_DIR, "songs_with_language.csv")

# Output
OUTPUT_CSV = os.path.join(BASE_DIR, "FINAL_PROJECT_DATASET.csv")

# HuggingFace emotion model
MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion"

# PCA thresholded weights (absolute < 0.10 removed) — uses your provided values
BALKAN_WEIGHTS = {
    "danceability": 0.212,
    "energy": 0.480,
    "loudness": 0.487,
    "speechiness": 0.0,      # removed
    "acousticness": -0.440,
    "instrumentalness": -0.399,
    "liveness": 0.0,         # removed
    "valence": 0.311,
    "tempo": 0.0             # removed
}

TURKISH_WEIGHTS = {
    "danceability": 0.321,
    "energy": 0.565,
    "loudness": 0.419,
    "speechiness": 0.268,
    "acousticness": -0.465,
    "instrumentalness": 0.0,  # removed
    "liveness": 0.0,          # removed
    "valence": 0.314,
    "tempo": 0.0              # removed
}

# Audio features expected to be present in audio CSV (order not critical)
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

# Verbose prints
VERBOSE = True

# ---------------- Helpers ----------------

def clean_track_metadata(text: str) -> str:
    """Clean track/artist names safely for matching."""
    if not text or pd.isna(text):
        return ""

    s = str(text).lower().strip()

    # Remove bracket contents
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"\[.*?\]", "", s)

    # Remove noise words
    noise_words = ["remix", "live", "akustik", "acoustic", "version", "feat", "ft.", "edit", "remastered"]
    for w in noise_words:
        s = s.replace(w, "")

    # Normalize Turkish characters
    s = s.translate(str.maketrans("çğıöşü", "cgiosu"))

    # Remove punctuation but KEEP spaces
    s = re.sub(r"[^a-z0-9 ]+", " ", s)

    # Collapse repeated spaces
    s = " ".join(s.split())

    return s


def build_merge_key(track: str, artist: str) -> str:
    t = clean_track_metadata(track)
    a = clean_track_metadata(artist)
    key = f"{t} {a}".strip()
    key = re.sub(r"\s+", " ", key)  # collapse spaces
    return key


def find_translated_files():
    """Collect tuples (culture, folder, filename, filepath) for all translated txt files."""
    tasks = []
    for culture, folder in TRANSLATED_FOLDERS.items():
        if not os.path.exists(folder):
            if VERBOSE:
                print(f"Warning: translated folder not found -> {folder} (skipping {culture})")
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith(".txt")]
        for fn in files:
            tasks.append((culture, folder, fn, os.path.join(folder, fn)))
    return tasks

# ---------------- NLP STEP ----------------

def run_nlp_on_translated_files(model_name=MODEL_NAME, device=None):
    """Run HuggingFace emotion model on each translated lyric file, return df_nlp."""
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print(f"Loading model '{model_name}' on {device_name}...")
    classifier = pipeline(
        "text-classification",
        model=model_name,
        top_k=None,
        truncation=True,
        device=device
    )

    tasks = find_translated_files()
    if not tasks:
        print("No translated lyric files found in folders.")
        return pd.DataFrame([])

    records = []
    for culture, folder, filename, filepath in tqdm(tasks, desc="NLP files", unit="file"):
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                text = fh.read().strip()
            if not text:
                # skip empty
                continue

            base = filename[:-4]  # remove .txt
            if "-" in base:
                parts = base.rsplit("-", 1)
                track_name = parts[0].strip()
                artist_name = parts[1].strip()
            else:
                track_name = base.strip()
                artist_name = ""

            merge_key = build_merge_key(track_name, artist_name)

            preds = classifier(text)

            # ---- FIX FOR ALL HF PIPELINE OUTPUT SHAPES ----
            def extract_top(preds):
                # Case 1: direct dict
                if isinstance(preds, dict):
                    return preds

                # Case 2: list output
                if isinstance(preds, list):
                    # Scenario: [[{label, score}, {label, score}, ...]]
                    if len(preds) > 0 and isinstance(preds[0], list):
                        preds = preds[0]

                    # Now preds should be list of dicts
                    preds = [p for p in preds if isinstance(p, dict)]

                    if len(preds) == 0:
                        return {"label": "UNKNOWN", "score": 0.0}

                    return max(preds, key=lambda x: x.get("score", 0.0))

                # Unknown format → safe fallback
                return {"label": "UNKNOWN", "score": 0.0}

            top = extract_top(preds)


            records.append({
                "artists": artist_name,
                "track name": track_name,
                "merge_key": merge_key,
                "emotion_type": str(top.get("label", "")).upper(),
                "emotion_score": float(top.get("score", 0.0)),
                "culture": culture
            })
        except Exception as e:
            tqdm.write(f"Error processing {filename}: {e}")
            continue

    df_nlp = pd.DataFrame(records)
    if VERBOSE:
        print(f"NLP completed: {len(df_nlp)} songs processed.")
    return df_nlp

# ---------------- AUDIO & MERGE STEP ----------------

def load_and_prepare_audio(audio_csv_path=AUDIO_CSV_PATH):
    """Load audio CSV and prepare merge_key and feature subset."""
    if not os.path.exists(audio_csv_path):
        raise FileNotFoundError(f"Audio CSV not found: {audio_csv_path}")

    df_audio = pd.read_csv(audio_csv_path)
    # Ensure required columns exist
    if "track_name" not in df_audio.columns or "artists" not in df_audio.columns:
        raise KeyError("Audio CSV must contain 'track_name' and 'artists' columns")

    # Create primary_artist and merge_key
    df_audio['primary_artist'] = (
        df_audio['artists']
        .astype(str)
        .apply(lambda x: x.split(';')[0].split(',')[0].strip())
    )

    df_audio['merge_key'] = df_audio.apply(
        lambda r: build_merge_key(r['track_name'], r['primary_artist']),
        axis=1
    )

    # ensure no double spaces + strip
    df_audio['merge_key'] = df_audio['merge_key'].str.replace(r"\s+", " ", regex=True).str.strip()


    # Drop duplicate merge_keys keeping first
    df_audio = df_audio.drop_duplicates(subset="merge_key", keep="first")

    if VERBOSE:
        print(f"Audio rows after dedup by merge_key: {len(df_audio)}")

    return df_audio

# ---------------- EMOTIONALITY COMPUTATION ----------------

def compute_pca_emotionality(merged_df):
    """Compute PCA-weighted emotionality per culture, normalize to [0,1] within culture."""
    # Identify features available
    features_present = [f for f in AUDIO_FEATURES if f in merged_df.columns]
    if VERBOSE:
        print("Audio features present for computation:", features_present)

    # Scale features to 0-1
    scaler = MinMaxScaler()
    merged_df[features_present] = scaler.fit_transform(merged_df[features_present])

    # compute raw pca score using culture-specific weights
    def raw_score(row):
        cult = str(row.get("culture", "")).strip().lower()
        weights = BALKAN_WEIGHTS if cult.startswith("balkan") else TURKISH_WEIGHTS
        s = 0.0
        for feat, w in weights.items():
            if feat in merged_df.columns:
                s += float(row[feat]) * float(w)
        return s

    merged_df["pca_raw"] = merged_df.apply(raw_score, axis=1)

    # Normalize pca_raw to 0-1 within each culture
    merged_df["emotionality"] = merged_df.groupby("culture")["pca_raw"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
    )

    # Drop helper column
    merged_df.drop(columns=["pca_raw"], inplace=True)
    return merged_df

# ---------------- MAIN ----------------

def combine_and_save():
    # Run NLP
    try:
        df_nlp = run_nlp_on_translated_files()
    except Exception as e:
        print("Error running NLP:", e)
        return

    if df_nlp.empty:
        print("No NLP output. Exiting.")
        return

    # Load audio metadata
    try:
        df_audio = load_and_prepare_audio()
    except Exception as e:
        print("Error loading audio CSV:", e)
        return

    # Prepare audio columns to merge
    audio_feature_cols = [c for c in AUDIO_FEATURES if c in df_audio.columns]
    extra_cols = []
    if "track_id" in df_audio.columns:
        extra_cols.append("track_id")
    # keep merge_key + optional track_id + audio features
    audio_cols = ["merge_key"] + extra_cols + audio_feature_cols

    # Merge (left: NLP, right: audio)
    merged = pd.merge(df_nlp, df_audio[audio_cols], on="merge_key", how="left", validate="m:1")
    if VERBOSE:
        print(f"Merged rows (NLP left): {len(merged)}")

    # Drop rows missing required audio features (we need at least the features for emotionality)
    required = ["valence", "energy", "loudness", "acousticness", "danceability", "tempo"]
    required_present = [r for r in required if r in merged.columns]
    if not required_present:
        print("Required audio features missing from audio CSV. Found none of:", required)
        # still continue but emotionality cannot be computed
    else:
        merged = merged.dropna(subset=required_present)
        if VERBOSE:
            print(f"Rows after dropping missing required audio features: {len(merged)}")

    if merged.empty:
        print("No rows to process after merging. Exiting.")
        return

    # Compute PCA-based emotionality (normalized per culture)
    merged = compute_pca_emotionality(merged)

    # Final ID: use track_id if available (from audio), otherwise create incremental id
    if "track_id" in merged.columns:
        merged["id"] = merged["track_id"]
    else:
        merged = merged.reset_index(drop=True)
        merged["id"] = merged.index + 1

    # Finalize columns order and write CSV
    final_cols = ["id", "artists", "track name", "emotionality", "emotion_type", "emotion_score", "culture"]
    final_present = [c for c in final_cols if c in merged.columns]
    final_df = merged[final_present].copy()

    # Ensure correct column types / formatting
    final_df["emotionality"] = final_df["emotionality"].astype(float)
    final_df["emotion_score"] = final_df["emotion_score"].astype(float)

    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved final dataset to: {OUTPUT_CSV} (rows: {len(final_df)})")
    if VERBOSE:
        print(final_df.head(10))

if __name__ == "__main__":
    combine_and_save()
