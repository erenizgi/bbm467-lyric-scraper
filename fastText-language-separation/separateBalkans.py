import pandas as pd

INPUT_FILE = "songs_with_language.csv"
df = pd.read_csv(INPUT_FILE)

balkan_map = {
    "Bulgarian": ["bg"], 
    "Greek": ["el"], 
    "Croatian": ["hr"],
    "Hungarian": ["hu"], 
    "Romanian": ["ro"], 
    "Bosnian": ["bs"],
    "Macedonian": ["mk"], 
    "Montenegrin": ["sr"], 
    "Albanian": ["sq"],
    "Serbian": ["sr"]
}
balkan_codes = [code for codes in balkan_map.values() for code in codes]

filtered = df[df["lang_fasttext"].isin(balkan_codes)]

filtered.to_csv("songs_balkan_only.csv", index=False)
print(f"✅ Balkan şarkıları ayrıldı: {len(filtered)} adet -> songs_balkan_only.csv")