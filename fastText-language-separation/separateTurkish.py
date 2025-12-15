import pandas as pd

INPUT_FILE = "songs_with_language.csv" 

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Hata: {INPUT_FILE} bulunamadı.")
    exit()

# Filtreleme: Türkçe (tr) VE Güven > 0.5
filtered = df[
    (df["lang_fasttext"] == "tr") & 
    (df["lang_confidence"] > 0.5)
]

# original_id sütunu zaten var, direkt kaydediyoruz.
filtered.to_csv("songs_turkish_only.csv", index=False)
print(f"✅ Türkçe şarkılar ayrıldı: {len(filtered)} adet -> songs_turkish_only.csv")