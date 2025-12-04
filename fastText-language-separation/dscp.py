import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import re

# --- DOSYA İSİMLERİ ---
NLP_DATASET_PATH = "final_music_analysis_dataset.csv"
AUDIO_DATASET_PATH = "songs_with_language.csv" # Veya senin en son temiz audio dosyan
OUTPUT_FILENAME = "FINAL_PROJECT_DATASET.csv"

def clean_track_metadata(text):
    """
    Şarkı ismindeki gürültüleri (Remix, Live, Parantezler) temizler.
    """
    if pd.isna(text): return ""
    text = str(text).lower()
    
    # 1. Parantez içlerini sil (...) ve [...]
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    
    # 2. Gereksiz kelimeleri sil
    # Buraya eşleşmeyi bozan kelimeleri ekleyebilirsin
    noise_words = ["remix", "live", "akustik", "acoustic", "version", "feat", "ft.", "edit"]
    for word in noise_words:
        text = text.replace(word, "")
        
    return text

def create_final_dataset():
    print("📂 Dosyalar yükleniyor...")
    
    try:
        df_nlp = pd.read_csv(NLP_DATASET_PATH)
        df_audio = pd.read_csv(AUDIO_DATASET_PATH)
    except FileNotFoundError:
        print("❌ Dosyalar bulunamadı.")
        return

    # --- 1. Audio Verisindeki Sanatçıları Temizle ---
    print("🧹 Sanatçı ve Şarkı isimleri temizleniyor...")
    # Noktalı virgül (;) varsa böl ve ilkini al
    df_audio['primary_artist'] = df_audio['artists'].astype(str).apply(lambda x: x.split(';')[0].split(',')[0].strip())

    # --- 2. Anahtar Oluşturma Fonksiyonu (GÜNCELLENDİ) ---
    def make_merge_key(row, source_type):
        """
        Hem NLP hem Audio için ortak bir anahtar üretir.
        """
        if source_type == 'audio':
            track = row['track_name']
            artist = row['primary_artist']
        else: # nlp
            track = row['track name']
            artist = row['artists']
            
        # A. Önce Gürültüleri Sil (Live, Remix, Parantez)
        track_clean = clean_track_metadata(track)
        artist_clean = clean_track_metadata(artist) # Sanatçıda genelde gerekmez ama garanti olsun
        
        # B. Türkçe Karakter ve Sembol Temizliği
        combined = track_clean + artist_clean
        
        # Türkçe karakterleri İngilizceye çevir
        replacements = str.maketrans("çğıöşü", "cgiosu")
        combined = combined.translate(replacements)
        
        # Alfanümerik olmayan her şeyi sil
        final_key = re.sub(r'[^a-z0-9]', '', combined)
        
        return final_key

    print("🔗 Akıllı Anahtarlar Oluşturuluyor...")
    
    # NLP Key
    df_nlp['merge_key'] = df_nlp.apply(lambda row: make_merge_key(row, 'nlp'), axis=1)
    
    # Audio Key
    df_audio['merge_key'] = df_audio.apply(lambda row: make_merge_key(row, 'audio'), axis=1)

    # Audio verisindeki kopyaları temizle (Aynı key'den birden fazla varsa ilki kalsın)
    df_audio = df_audio.drop_duplicates(subset="merge_key", keep="first")

    # Birleştirme
    audio_cols = [
        "merge_key", "danceability", "energy", "loudness", 
        "speechiness", "acousticness", "instrumentalness", 
        "liveness", "valence", "tempo"
    ]
    # Sadece var olan sütunları seç (Hata almamak için)
    available_cols = [c for c in audio_cols if c in df_audio.columns]
    
    # LEFT MERGE: NLP verisi ana tablomuz, Audio verilerini yanına çekiyoruz
    merged_df = pd.merge(df_nlp, df_audio[available_cols], on="merge_key", how="left")

    # Rapor
    missing_mask = merged_df['valence'].isna()
    missing_count = missing_mask.sum()
    
    print("-" * 30)
    print(f"📊 Toplam NLP Şarkısı: {len(df_nlp)}")
    print(f"✅ Eşleşen Şarkı: {len(df_nlp) - missing_count}")
    print(f"⚠️ Bulunamayan: {missing_count}")
    print("-" * 30)
    
    # --- DEBUG: Bulunamayanları Göster ---
    if missing_count > 0:
        print("\n🔍 Eşleşmeyen İlk 10 Örnek (Hata Ayıklama İçin):")
        missing_rows = merged_df[missing_mask].head(10)
        for idx, row in missing_rows.iterrows():
            print(f"   -> {row['track name']} | Sanatçı: {row['artists']}")
            print(f"      Oluşan Key: {row['merge_key']}")
        print("\n(Not: Bu şarkılar Audio CSV dosyasında olmayabilir veya isimleri çok farklı olabilir.)")

    # Bulunamayanları çıkar (Analiz için boş veri işe yaramaz)
    merged_df = merged_df.dropna(subset=['valence'])

    if merged_df.empty:
        print("❌ HATA: Hiçbir veri eşleşmedi! Dosya isimlerini veya sütunları kontrol et.")
        return

    # --- Normalizasyon ve Hesaplama ---
    print("🧮 Hesaplamalar yapılıyor...")
    features_to_scale = [
        "danceability", "energy", "loudness", "speechiness", 
        "acousticness", "instrumentalness", "liveness", "valence", "tempo"
    ]
    
    scaler = MinMaxScaler()
    merged_df[features_to_scale] = scaler.fit_transform(merged_df[features_to_scale])

    merged_df["emotionality"] = (
        (1 - merged_df["valence"]) * 0.40 +
        merged_df["acousticness"] * 0.20 +
        (1 - merged_df["energy"]) * 0.10 +
        merged_df["instrumentalness"] * 0.10 +
        (1 - merged_df["tempo"]) * 0.10 +
        (1 - merged_df["loudness"]) * 0.10
    )

    # --- Final Format ---
    merged_df = merged_df.reset_index(drop=True)
    merged_df["id"] = merged_df.index + 1
    
    # Final sütunları seç
    final_cols = ["id", "artists", "track name", "emotionality", "emotion_type", "emotion_score", "culture"]
    
    final_output = merged_df[final_cols]
    final_output.to_csv(OUTPUT_FILENAME, index=False)
    
    print(f"\n✅ DOSYA HAZIR: {OUTPUT_FILENAME}")
    print(final_output.head())

if __name__ == "__main__":
    create_final_dataset()