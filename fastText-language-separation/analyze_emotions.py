import os
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

# --- AYARLAR ---
TRANSLATED_FOLDERS = {
    "Turkish": "../lyrics_files_turkish_translated",
    "Balkan": "../lyrics_files_balkan_translated"
}
FINAL_CSV_NAME = "final_music_analysis_dataset.csv"
MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion"

def run_analysis():
    # --- DEĞİŞİKLİK BURADA: Zorla CPU (-1) kullanıyoruz ---
    # RTX 5060 uyumsuzluğu yüzünden GPU'yu kapatıyoruz.
    device = -1 
    print(f"⏳ Model Yükleniyor... (CPU Modu Aktif)")

    classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True, device=device)
    all_data = []

    print("\n🔍 Dosyalar taranıyor...")
    tasks = []
    for culture, path in TRANSLATED_FOLDERS.items():
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith(".txt")]
            for f in files: tasks.append((os.path.join(path, f), f, culture))

    if not tasks:
        print("❌ Dosya bulunamadı."); return

    for path, filename, culture in tqdm(tasks, desc="NLP Analizi"):
        try:
            # --- ID PARSE ETME ---
            if "_" in filename:
                parts = filename.split("_", 1)
                
                # ID al
                try:
                    s_id = int(parts[0]) 
                except ValueError:
                    # Eğer ID sayı değilse (örn: manuel dosya), atla veya 0 ver
                    continue

                # İsimleri ayıkla
                rest = parts[1].replace(".txt", "")
                if "-" in rest:
                    p = rest.rsplit("-", 1) 
                    track, artist = p[0].strip(), p[1].strip()
                else:
                    track, artist = rest, "Unknown"
            else:
                continue 

            # Dosyayı oku
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Boş dosya kontrolü
            if not text or len(text.strip()) == 0: 
                continue

            # Analiz et
            pred = classifier(text)
            top = max(pred[0], key=lambda x: x['score'])

            all_data.append({
                "original_id": s_id,
                "artists": artist,
                "track name": track,
                "emotion_type": top['label'].upper(),
                "emotion_score": top['score'],
                "culture": culture
            })
        except Exception as e:
            # Hata olursa ekrana basalım ki görelim
            # tqdm.write ilerleme çubuğunu bozmadan yazdırır
            tqdm.write(f"Hata ({filename}): {e}")
            continue

    if all_data:
        # CSV Kaydet
        df = pd.DataFrame(all_data)
        df.to_csv(FINAL_CSV_NAME, index=False)
        print(f"\n✅ Analiz bitti! Sonuçlar '{FINAL_CSV_NAME}' dosyasına kaydedildi.")
        print(f"Toplam İşlenen Şarkı: {len(df)}")
    else:
        print("❌ Veri oluşturulamadı. Hiçbir dosya analiz edilemedi.")

if __name__ == "__main__":
    run_analysis()