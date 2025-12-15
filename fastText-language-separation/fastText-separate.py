import pandas as pd
import fasttext
import re
from tqdm import tqdm

# --- AYARLAR ---
CSV_PATH = "songs.csv"
MODEL_PATH = "lid.176.bin"
OUTPUT_PATH = "songs_with_language.csv"

# Modeli ve Veriyi Yükle
model = fasttext.load_model(MODEL_PATH)
df = pd.read_csv(CSV_PATH)

# 🔥 KRİTİK: 'Unnamed: 0' sütununu 'original_id' yapıyoruz
if "Unnamed: 0" in df.columns:
    df = df.rename(columns={"Unnamed: 0": "original_id"})
else:
    # Eğer sütun ismi yoksa indexi id yap
    df["original_id"] = df.index

def clean_text(text):
    if not isinstance(text, str): return ""
    return text.strip().replace("-", " ").replace("_", " ").replace("\n", " ")

def contains_turkish_chars(text):
    turkish_chars = set("çğıöşüÇĞİÖŞÜ")
    return any(char in turkish_chars for char in text)

langs = []
confidences = []

combined_inputs = (
    df["artists"].fillna("") + " " + 
    df["album_name"].fillna("") + " " + 
    df["track_name"].fillna("")
).astype(str)

for idx, text in tqdm(enumerate(combined_inputs), total=len(df), desc="Dil Tespiti"):
    cleaned = clean_text(text)
    
    # Tahmin
    pred = model.predict(cleaned, k=1)
    lang_code = pred[0][0].replace("__label__", "")
    confidence = float(pred[1][0])
    
    # Türkçe Karakter Kontrolü (Güven Artırıcı)
    raw_track_name = str(df.iloc[idx]["track_name"])
    raw_artist_name = str(df.iloc[idx]["artists"])
    if lang_code == 'tr' and confidence < 0.50:
        if contains_turkish_chars(raw_track_name + " " + raw_artist_name):
            confidence = 0.99 

    langs.append(lang_code)
    confidences.append(confidence)

df["lang_fasttext"] = langs
df["lang_confidence"] = confidences

# Kaydet (original_id sütunuyla birlikte)
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ İşlem Tamam! '{OUTPUT_PATH}' dosyasına 'original_id' ile kaydedildi.")