import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# --- SETTINGS ---
NLP_CSV = "final_music_analysis_dataset.csv"  # Output from analyze_emotions.py
AUDIO_CSV = "songs_with_language.csv"         # Output from fastText
OUTPUT_CSV = "FINAL_PROJECT_DATASET.csv"      # RESULT

def create_final_dataset():
    print("📂 Loading files...")
    
    if not os.path.exists(NLP_CSV) or not os.path.exists(AUDIO_CSV):
        print("❌ ERROR: Missing files.")
        return

    df_nlp = pd.read_csv(NLP_CSV)
    df_audio = pd.read_csv(AUDIO_CSV)

    # ID Column Preparation
    if "original_id" not in df_audio.columns:
        if "Unnamed: 0" in df_audio.columns:
            df_audio = df_audio.rename(columns={"Unnamed: 0": "original_id"})
        else:
            df_audio["original_id"] = df_audio.index

    # Type conversion
    df_nlp["original_id"] = df_nlp["original_id"].astype(int)
    df_audio["original_id"] = df_audio["original_id"].astype(int)

    # --- MERGING ---
    print("🔗 Merging data...")
    
    # Columns to be used in PCA
    audio_cols = [
        "danceability", "energy", "loudness", "speechiness", 
        "acousticness", "instrumentalness", "liveness", 
        "valence", "tempo"
    ]
    
    # Merge only necessary columns
    cols_to_merge = ["original_id"] + audio_cols
    merged_df = pd.merge(df_nlp, df_audio[cols_to_merge], on="original_id", how="left")

    # Cleanup
    merged_df = merged_df.dropna(subset=['valence'])
    
    if merged_df.empty:
        print("❌ ERROR: Data did not match."); return

    print("-" * 30)
    print(f"✅ Songs Ready for Analysis: {len(merged_df)}")
    print("-" * 30)

    # --- PCA CALCULATION (Emotionality Index) ---
    print("🧮 Calculating Emotionality Index with PCA...")

    # 1. Standardization (Mandatory for PCA)
    X = merged_df[audio_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Apply PCA (Single component: Emotionality Axis)
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(X_scaled)
    
    # 3. Examine Weights (Loadings) and Determine Direction
    loadings = pca.components_[0]
    loading_dict = dict(zip(audio_cols, loadings))
    
    print("\n🔍 PCA Weights (Data-Driven Formula):")
    for k, v in loading_dict.items():
        print(f"   {k}: {v:.3f}")

    # --- CRITICAL CHECK: Direction Determination ---
    # When we say "Emotionality", we generally mean "Sad/Calm".
    # Therefore, 'Valence' (Happiness) and 'Energy' should be NEGATIVE in the PCA result.
    # If PCA found these as Positive, we must invert the results (multiply by -1).
    
    # Checking the weight of Valence:
    if loading_dict['valence'] > 0:
        print("\n🔄 Direction Correction: PCA found 'Happiness' direction as positive. Inverting for 'Sadness'...")
        principal_components = principal_components * -1
    else:
        print("\n✅ Direction Correct: PCA already found 'Sadness/Calmness' direction as positive.")

    # 4. Squeeze results between 0-1 (Normalize)
    min_max_scaler = MinMaxScaler()
    emotionality_scores = min_max_scaler.fit_transform(principal_components)

    # Add to DataFrame
    merged_df["emotionality"] = emotionality_scores

    # --- FINAL FORMAT ---
    final_cols = [
        "original_id", "artists", "track name", 
        "emotionality", "emotion_type", "emotion_score", "culture"
    ]
    
    final_output = merged_df[final_cols]
    final_output.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ PROJECT COMPLETED! File ready: {OUTPUT_CSV}")
    print(final_output.head())

if __name__ == "__main__":
    create_final_dataset()