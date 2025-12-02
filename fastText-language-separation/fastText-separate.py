import pandas as pd
import fasttext
from tqdm import tqdm

CSV_PATH = "songs.csv"       # senin dosya
TRACK_COL = "track_name"     # şarkı adı kolonun ismi
MODEL_PATH = "lid.176.bin"   # fastText dili tanıma modeli

model = fasttext.load_model(MODEL_PATH)

df = pd.read_csv(CSV_PATH)

def clean_title(text):
    if not isinstance(text, str):
        return ""
    return text.strip().replace("-", " ").replace("_", " ")

# --- LANGUAGE PREDICTION ---
langs = []
confidences = []

for title in tqdm(df[TRACK_COL].fillna("").astype(str), desc="Detecting languages"):
    cleaned = clean_title(title)
    pred = model.predict(cleaned, k=1)
    lang_code = pred[0][0].replace("__label__", "")
    confidence = float(pred[1][0])
    
    langs.append(lang_code)
    confidences.append(confidence)

df["lang_fasttext"] = langs
df["lang_confidence"] = confidences

print(langs)

df.to_csv("songs_with_language.csv", index=False)

print("Done! -> songs_with_language.csv")
