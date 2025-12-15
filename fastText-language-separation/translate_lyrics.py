import os
import time
from deep_translator import GoogleTranslator
from tqdm import tqdm

# --- AYARLAR ---
# Python dosyanın alt klasörde olduğunu varsayarak bir üst dizine (../) çıkıyoruz.
# Eğer kodun ana dizindeyse "../" kısımlarını silip "./" yapmalısın.
FOLDERS_TO_TRANSLATE = {
    "Turkish": {
        "input": "../lyrics_files_turkish",
        "output": "../lyrics_files_turkish_translated"
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

        if not os.path.exists(output_dir): os.makedirs(output_dir)
        if not os.path.exists(input_dir):
            print(f"⚠️ {input_dir} bulunamadı, geçiliyor.")
            continue

        files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
        print(f"\n🌍 {culture} Çevriliyor ({len(files)} dosya)...")

        for filename in tqdm(files, desc=f"{culture}"):
            output_path = os.path.join(output_dir, filename)

            # Zaten çevrildiyse atla (Zaman kazancı)
            if os.path.exists(output_path): continue

            try:
                with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
                    text = f.read()

                # Boş veya çok kısa dosyaları atla
                if len(text) < 10: continue

                # BERT sınırı ve Hız için ilk 600 karakteri çeviriyoruz
                translated = translator.translate(text[:600])

                if translated:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(translated)
                    
                    # Google API engelini aşmak için minik bekleme
                    time.sleep(0.5)

            except Exception as e:
                # Hata olsa bile durma, sonrakine geç
                continue
    
    print("\n✅ ÇEVİRİ TAMAMLANDI!")

if __name__ == "__main__":
    run_translation()