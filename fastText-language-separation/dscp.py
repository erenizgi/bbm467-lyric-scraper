import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# --- AYARLAR ---
NLP_CSV = "final_music_analysis_dataset.csv"  # analyze_emotions.py çıktısı
AUDIO_CSV = "songs_with_language.csv"         # fastText çıktısı
OUTPUT_CSV = "FINAL_PROJECT_DATASET.csv"      # SONUÇ

def create_final_dataset():
    print("📂 Dosyalar yükleniyor...")
    
    if not os.path.exists(NLP_CSV) or not os.path.exists(AUDIO_CSV):
        print("❌ HATA: Dosyalar eksik.")
        return

    df_nlp = pd.read_csv(NLP_CSV)
    df_audio = pd.read_csv(AUDIO_CSV)

    # ID Sütunu Hazırlığı
    if "original_id" not in df_audio.columns:
        if "Unnamed: 0" in df_audio.columns:
            df_audio = df_audio.rename(columns={"Unnamed: 0": "original_id"})
        else:
            df_audio["original_id"] = df_audio.index

    # Tür dönüşümü
    df_nlp["original_id"] = df_nlp["original_id"].astype(int)
    df_audio["original_id"] = df_audio["original_id"].astype(int)

    # --- BİRLEŞTİRME ---
    print("🔗 Veriler birleştiriliyor...")
    
    # PCA'da kullanacağımız sütunlar
    audio_cols = [
        "danceability", "energy", "loudness", "speechiness", 
        "acousticness", "instrumentalness", "liveness", 
        "valence", "tempo"
    ]
    
    # Sadece gerekli sütunları alarak birleştir
    cols_to_merge = ["original_id"] + audio_cols
    merged_df = pd.merge(df_nlp, df_audio[cols_to_merge], on="original_id", how="left")

    # Temizlik
    merged_df = merged_df.dropna(subset=['valence'])
    
    if merged_df.empty:
        print("❌ HATA: Veri eşleşmedi."); return

    print("-" * 30)
    print(f"✅ Analiz İçin Hazır Şarkı Sayısı: {len(merged_df)}")
    print("-" * 30)

    # --- PCA HESAPLAMA (Emotionality Index) ---
    print("🧮 PCA ile Emotionality İndeksi Hesaplanıyor...")

    # 1. Standartlaştırma (PCA için zorunlu)
    X = merged_df[audio_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. PCA Uygula (Tek bileşen: Emotionality Axis)
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(X_scaled)
    
    # 3. Yükleri (Weights) İncele ve Yönü Belirle
    loadings = pca.components_[0]
    loading_dict = dict(zip(audio_cols, loadings))
    
    print("\n🔍 PCA Ağırlıkları (Data-Driven Formula):")
    for k, v in loading_dict.items():
        print(f"   {k}: {v:.3f}")

    # --- KRİTİK KONTROL: Yön Belirleme ---
    # Biz "Emotionality" derken genelde "Hüzünlü/Sakin" kastediyoruz.
    # Bu yüzden 'Valence' (Mutluluk) ve 'Energy' PCA sonucunda NEGATİF olmalı.
    # Eğer PCA bunları Pozitif bulduysa, sonuçları ters çevirmeliyiz (-1 ile çarp).
    
    # Valence'ın yüküne bakıyoruz:
    if loading_dict['valence'] > 0:
        print("\n🔄 Yön Düzeltme: PCA 'Mutluluk' yönünü pozitif buldu. 'Hüzün' için ters çevriliyor...")
        principal_components = principal_components * -1
    else:
        print("\n✅ Yön Doğru: PCA zaten 'Hüzün/Sakinlik' yönünü pozitif buldu.")

    # 4. Sonuçları 0-1 arasına sıkıştır (Normalize et)
    min_max_scaler = MinMaxScaler()
    emotionality_scores = min_max_scaler.fit_transform(principal_components)

    # DataFrame'e ekle
    merged_df["emotionality"] = emotionality_scores

    # --- FINAL FORMAT ---
    final_cols = [
        "original_id", "artists", "track name", 
        "emotionality", "emotion_type", "emotion_score", "culture"
    ]
    
    final_output = merged_df[final_cols]
    final_output.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ PROJE TAMAMLANDI! Dosya hazır: {OUTPUT_CSV}")
    print(final_output.head())

if __name__ == "__main__":
    create_final_dataset()