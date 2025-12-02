import os
import random
import time
from transformers import pipeline
from deep_translator import GoogleTranslator

# --- AYARLAR ---
# Şarkı dosyalarının olduğu klasör yolu (Kendi yoluna göre düzenle)
LYRICS_DIR = "../lyrics_files" 
MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion"

def analyze_local_files():
    # 1. Klasördeki tüm .txt dosyalarını bul
    if not os.path.exists(LYRICS_DIR):
        print(f"HATA: '{LYRICS_DIR}' klasörü bulunamadı!")
        return

    all_files = [f for f in os.listdir(LYRICS_DIR) if f.endswith(".txt")]
    
    if not all_files:
        print("Klasörde hiç .txt dosyası yok!")
        return

    # Rastgele 5 şarkı seç (Test amaçlı)
    selected_files = random.sample(all_files, min(5, len(all_files)))
    
    print(f"📂 Toplam {len(all_files)} dosya bulundu. Rastgele {len(selected_files)} tanesi analiz edilecek.\n")
    
    # 2. Modeli ve Çevirmeni Hazırla
    print("⏳ Model yükleniyor...")
    emotion_classifier = pipeline(
        "text-classification", 
        model=MODEL_NAME, 
        return_all_scores=True,
        truncation=True  # Çok uzun metinleri otomatik kırpar (Hata almamak için şart)
    )
    translator = GoogleTranslator(source='tr', target='en')

    print("\n🚀 DOSYA ANALİZİ BAŞLIYOR...\n" + "="*60)

    for filename in selected_files:
        filepath = os.path.join(LYRICS_DIR, filename)
        
        # Dosya isminden şarkı ve sanatçıyı ayıkla (Görsellik için)
        display_name = filename.replace(".txt", "")
        
        try:
            # Dosyayı Oku
            with open(filepath, "r", encoding="utf-8") as f:
                original_lyrics = f.read()
            
            # Boş dosya kontrolü
            if not original_lyrics.strip():
                print(f"⚠️ {display_name} -> Dosya boş, geçiliyor.")
                continue

            # Veri temizliği (Çok uzun satırları birleştir vs.)
            # BERT modeli en fazla 512 token alır. Çeviri API'sini yormamak için ilk 1000 karakteri alalım.
            text_to_process = original_lyrics[:1000] 

            print(f"🎵 {display_name}")
            
            # Çeviri (TR -> EN)
            translated_text = translator.translate(text_to_process)
            
            # Çeviri bazen boş dönebilir kontrolü
            if not translated_text:
                print("❌ Çeviri başarısız oldu.")
                continue

            # Duygu Analizi
            predictions = emotion_classifier(translated_text)
            
            # Skorlama
            scores = predictions[0]
            scores.sort(key=lambda x: x['score'], reverse=True)
            
            top_emotion = scores[0]
            second_emotion = scores[1]
            
            print(f"   🏆 {top_emotion['label'].upper()} (%{top_emotion['score']*100:.1f}) | 🥈 {second_emotion['label']} (%{second_emotion['score']*100:.1f})")
            print(f"   🌍 (Çeviri Özeti): \"{translated_text[:60]}...\"")
            print("-" * 60)

        except Exception as e:
            print(f"❌ HATA ({display_name}): {e}")
        
        # Google Translate API ban yememek için bekleme
        time.sleep(1.5)

if __name__ == "__main__":
    analyze_local_files()