import pandas as pd
import fasttext
import re
from tqdm import tqdm

# --- CONFIGURATION ---
CSV_PATH = "songs.csv"
MODEL_PATH = "lid.176.bin"
OUTPUT_PATH = "songs_with_language.csv"

# Load model and data
model = fasttext.load_model(MODEL_PATH)
df = pd.read_csv(CSV_PATH)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove punctuation and newlines, keep basic text
    return text.strip().replace("-", " ").replace("_", " ").replace("\n", " ")

def contains_turkish_chars(text):
    """
    Checks for specific Turkish characters.
    If a song contains these, it's highly likely to be Turkish even if the model is unsure.
    """
    turkish_chars = set("çğıöşüÇĞİÖŞÜ")
    return any(char in turkish_chars for char in text)

# --- DETECTION LOGIC ---
langs = []
confidences = []
final_decisions = [] # To keep track of 'Why' we kept it (High Conf or Char Match)


# Combine columns for better context: Artist + Album + Track
# Example: "Ezginin Günlüğü İlk Aşk Zerdaliler"
combined_inputs = (
    df["artists"].fillna("") + " " + 
    df["album_name"].fillna("") + " " + 
    df["track_name"].fillna("")
).astype(str)

for idx, text in tqdm(enumerate(combined_inputs), total=len(df), desc="Detecting languages"):
    cleaned = clean_text(text)
    
    # 1. FastText Prediction
    pred = model.predict(cleaned, k=1)
    lang_code = pred[0][0].replace("__label__", "")
    confidence = float(pred[1][0])
    
    # 2. Heuristic Check (The Safety Net)
    # Get the raw track name for character checking
    raw_track_name = str(df.iloc[idx]["track_name"])
    raw_artist_name = str(df.iloc[idx]["artists"])
    full_check_text = raw_track_name + " " + raw_artist_name
    
    has_tr_char = contains_turkish_chars(full_check_text)
    
    # --- LOGIC TO BOOST CONFIDENCE ---
    # If model says 'tr' BUT confidence is low...
    if lang_code == 'tr' and confidence < 0.50:
        if has_tr_char:
            # If it has distinctive chars (e.g. 'Ezginin GÜNLÜĞÜ'), trust it!
            # Artificially boost confidence or mark as valid
            confidence = 0.99 
        else:
            # If no turkish chars and low confidence, it might be junk (like the Japanese song)
            pass 

    langs.append(lang_code)
    confidences.append(confidence)

# Add results to DataFrame
df["lang_fasttext"] = langs
df["lang_confidence"] = confidences

# --- FILTERING STRATEGY ---
# Now you can safely filter with a high threshold because we boosted the valid ones
# or simply filter > 0.5
df_filtered = df[df["lang_confidence"] > 0.5] 

print(f"Total Rows: {len(df)}")
print(f"Rows after robust filtering: {len(df_filtered)}")

# Save
df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")