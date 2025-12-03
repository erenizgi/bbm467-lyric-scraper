import pandas as pd

df = pd.read_csv("songs_with_language.csv")

balkan_map = {
    "Bulgarian": ["bg"],
    "Greek": ["el"],
    "Croatian": ["hr"],
    "Hungarian": ["hu"],
    "Romanian": ["ro"],
    "Bosnian": ["bs"],
    "Macedonian": ["mk"],
    "Montenegrin": ["sr"],  # sr / sh genelde Montenegrin’i temsil ediyor
    "Albanian": ["sq"],
    "Serbian": ["sr"]
}
balkan_codes = [code for codes in balkan_map.values() for code in codes]



balkan_songs = df[
    (df["lang_fasttext"].isin(balkan_codes)) 
]

balkan_songs.to_csv("songs_balkan_only.csv", index=False)

print(f"Ayıklandı! {len(balkan_songs)} Balkan kayıt bulundu.")
