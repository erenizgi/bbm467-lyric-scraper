import pandas as pd
import numpy as np

df = pd.read_csv("songs_with_language.csv")

# 'language' veya 'tr' kolonu hangisiyse ona göre filtrele
# Eğer kolon adı 'language' ise:


filtered = df[df["lang_fasttext"] == "tr"]

filtered.to_csv("songs_turkish_only.csv", index=False)

print(f"Ayıklandı! {len(filtered)} Türkçe kayıt bulundu.")
