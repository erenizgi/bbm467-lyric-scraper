import pandas as pd
import numpy as np

df = pd.read_csv("songs_with_language.csv")

# 'language' veya 'tr' kolonu hangisiyse ona göre filtrele
# Eğer kolon adı 'language' ise:


filtered = df[df["lang_fasttext"] == "bg" | df["lang_fasttext"] == "el" | df["lang_fasttext"] == "hr" | df["lang_fasttext"] == "hu" | df["lang_fasttext"] == "ro" | df["lang_fasttext"] == "bs" | df["lang_fasttext"] == "mk" | df["lang_fasttext"] == "cn" | df["lang_fasttext"] == "sq" | df["lang_fasttext"] == "sr"]

filtered.to_csv("song_balkans_only.csv", index=False)

print(f"Ayıklandı! {len(filtered)} Türkçe kayıt bulundu.")
