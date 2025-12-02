import pandas as pd

# DİKKAT: Yeni oluşturduğumuz 'robust' dosyayı okuyoruz
INPUT_FILE = "songs_with_language.csv" 

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Hata: {INPUT_FILE} bulunamadı. Önce fastText kodunu çalıştırdın mı?")
    exit()

# FİLTRELEME: Hem 'tr' olsun HEM DE güven oranı 0.5'ten yüksek olsun
# (Zerdaliler gibi şarkıları zaten boost etmiştik, o yüzden yüksek confidence ile gelirler)
filtered = df[
    (df["lang_fasttext"] == "tr") & 
    (df["lang_confidence"] > 0.5)
]

filtered.to_csv("songs_turkish_only.csv", index=False)
print(f"Ayıklandı! {len(filtered)} Temiz Türkçe kayıt bulundu -> songs_turkish_final.csv")