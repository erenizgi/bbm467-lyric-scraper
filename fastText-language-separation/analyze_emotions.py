import os
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

# --- CONFIGURATION ---
TRANSLATED_FOLDERS = {
    "Turkish": "../lyrics_files_turkish_translated",
    "Balkan": "../lyrics_files_balkan_translated"
}

TURKISH_CSV = "fastText-language-separation/songs_turkish_only.csv"
BALKAN_CSV  = "fastText-language-separation/songs_balkan_only.csv"

FINAL_CSV_NAME = "final_music_analysis_dataset.csv"
MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion"


def normalize_text(text):
    """Lowercase, strip spaces, remove problematic characters for matching."""
    return (
        text.lower()
        .replace("(", "")
        .replace(")", "")
        .replace("-", " ")
        .replace("_", " ")
        .replace("&", "and")
        .strip()
    )


def load_metadata():
    """Load Turkish and Balkan metadata and build fast lookup tables."""
    metadata = {
        "Turkish": pd.read_csv(TURKISH_CSV),
        "Balkan": pd.read_csv(BALKAN_CSV)
    }
    
    # Normalize for lookup
    for culture, df in metadata.items():
        df["track_name_clean"] = df["track_name"].apply(normalize_text)
        df["artists_clean"] = df["artists"].apply(normalize_text)
    
    return metadata


def find_track_id(filename_base, culture, metadata):
    """Find the corresponding track_id using fuzzy/clean matching."""
    df = metadata[culture]
    
    # Extract track/artist from filename
    if "-" in filename_base:
        parts = filename_base.rsplit("-", 1)
        track = normalize_text(parts[0])
        artist = normalize_text(parts[1])
    else:
        track = normalize_text(filename_base)
        artist = None
    
    # 1) Exact match on track name
    match = df[df["track_name_clean"] == track]
    if len(match) == 1:
        return match.iloc[0]["track_id"]
    
    # 2) Match track + artist if possible
    if artist:
        match = df[(df["track_name_clean"] == track) &
                   (df["artists_clean"].str.contains(artist))]
        if len(match) == 1:
            return match.iloc[0]["track_id"]
    
    # 3) Loose contains match
    match = df[df["track_name_clean"].str.contains(track)]
    if len(match) >= 1:
        return match.iloc[0]["track_id"]

    return None  # Not found


def run_analysis():
    metadata = load_metadata()

    # GPU Check
    device = 0 if torch.cuda.is_available() else -1
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print(f"⏳ Loading Model... (Device: {device_name})")

    emotion_classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,
        truncation=True,
        device=device
    )

    all_data = []

    # Gather all tasks
    all_tasks = []
    for culture, folder_path in TRANSLATED_FOLDERS.items():
        if not os.path.exists(folder_path):
            print(f"⚠️ {folder_path} not found. Skipping.")
            continue
        
        files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            all_tasks.append((file_path, filename, culture))

    total_files = len(all_tasks)
    if total_files == 0:
        print("❌ No files found.")
        return

    print(f"\n✅ Found {total_files} files. Starting NLP...")

    for file_path, filename, culture in tqdm(all_tasks, desc="Total NLP Progress", unit="song"):
        filename_base = filename.replace(".txt", "")
        
        # Get the original track ID
        track_id = find_track_id(filename_base, culture, metadata)

        if track_id is None:
            tqdm.write(f"⚠️ ID not found for: {filename}")
            continue

        # Parse artist/track for output (not for matching)
        if "-" in filename_base:
            parts = filename_base.rsplit("-", 1)
            track_name = parts[0].strip()
            artist_name = parts[1].strip()
        else:
            track_name = filename_base
            artist_name = "Unknown"

        # NLP Emotion Extraction
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_english = f.read()

            if not text_english:
                continue

            predictions = emotion_classifier(text_english)
            scores = predictions[0]
            top_emotion = max(scores, key=lambda x: x["score"])

            # Append result
            all_data.append({
                "id": track_id,      # <-- ORIGINAL ID
                "artists": artist_name,
                "track name": track_name,
                "emotionality": "",
                "emotion_type": top_emotion["label"].upper(),
                "emotion_score": top_emotion["score"],
                "culture": culture
            })

        except Exception as e:
            tqdm.write(f"Error in {filename}: {e}")
            continue

    # Save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv(FINAL_CSV_NAME, index=False)

    print(f"\n🎉 ALL DONE! Saved → {FINAL_CSV_NAME}")
    print(f"Total processed songs: {len(df)}")
    print(df.head())


if __name__ == "__main__":
    run_analysis()
