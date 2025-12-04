import os
import time
from deep_translator import GoogleTranslator
from tqdm import tqdm

# --- AYARLAR ---
FOLDERS_TO_TRANSLATE = {
    "Turkish": {
        "input": "../lyrics_files_turkish",           # Kaynak (Orijinal)
        "output": "../lyrics_files_turkish_translated" # Hedef (İngilizce)
    },
    "Balkan": {
        "input": "../lyrics_files_balkan",
        "output": "../lyrics_files_balkan_translated"
    }
}

def run_translation():
    translator = GoogleTranslator(source='auto', target='en')

    for culture, paths in FOLDERS_TO_TRANSLATE.items():
        input_dir = paths["input"]
        output_dir = paths["output"]

        # Hedef klasörü oluştur
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Klasör yoksa atla
        if not os.path.exists(input_dir):
            print(f"⚠️ {input_dir} bulunamadı, geçiliyor.")
            continue

        files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
        print(f"\n🌍 {culture} Şarkıları Çevriliyor ({len(files)} dosya)...")

        for filename in tqdm(files, desc=f"{culture} Çeviri"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # 1. Zaten çevrildiyse tekrar yapma (Cache mantığı)
            if os.path.exists(output_path):
                continue

            try:
                # 2. Dosyayı Oku
                with open(input_path, "r", encoding="utf-8") as f:
                    original_text = f.read()

                if len(original_text) < 10: continue

                # 3. Çevir (İlk 600 karakter yeterli)
                chunk = original_text[:600]
                translated_text = translator.translate(chunk)

                # 4. Kaydet
                if translated_text:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(translated_text)
                    
                    # API Ban yememek için minik bekleme
                    time.sleep(0.5)

            except Exception as e:
                print(f"Hata ({filename}): {e}")
                continue
    
    print("\n✅ ÇEVİRİ İŞLEMİ TAMAMLANDI!")

if __name__ == "__main__":
    run_translation()