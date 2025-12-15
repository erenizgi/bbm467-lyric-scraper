import requests
import os

# --- AYARLAR ---
url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
target_folder = "intensive/bbm467-lyric-scraper/fastText-language-separation"  # İndirilecek klasör adı
filename = "lid.176.bin"

# 1. Hedef klasör var mı kontrol et, yoksa oluştur
if not os.path.exists(target_folder):
    os.makedirs(target_folder)
    print(f"📁 '{target_folder}' klasörü bulunamadı, oluşturuldu.")
else:
    print(f"📁 '{target_folder}' klasörü bulundu.")

# 2. Tam dosya yolunu birleştir (fastText-language-separation/lid.176.bin)
save_path = os.path.join(target_folder, filename)

print(f"⬇️ {filename} indiriliyor... Hedef: {save_path}")

# 3. Dosyayı indir ve belirtilen yola kaydet
response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open(save_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)

print(f"✅ İndirme Başarılı! Dosya şurada hazır: {save_path}")