import os
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

# --- CONFIGURATION ---
TRANSLATED_FOLDERS = {
    "Turkish": "../lyrics_files_turkish_translated",
    "Balkan": "../lyrics_files_balkan_translated"
}

FINAL_CSV_NAME = "final_music_analysis_dataset.csv"
MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion"

def run_analysis():
    # 1. GPU Check
    device = 0 if torch.cuda.is_available() else -1
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print(f"⏳ Loading Model... (Device: {device_name})")

    emotion_classifier = pipeline(
        "text-classification", 
        model=MODEL_NAME, 
        top_k=None, 
        truncation=True,
        device=device
    )

    all_data = []
    global_id_counter = 1
    
    # --- PREPARATION STEP: GATHER ALL FILES FIRST ---
    # We collect all files into a single list to have one global progress bar
    all_tasks = []
    
    print("\n🔍 Scanning folders...")
    for culture, folder_path in TRANSLATED_FOLDERS.items():
        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: {folder_path} not found. Skipping.")
            continue
            
        files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        for filename in files:
            full_path = os.path.join(folder_path, filename)
            # Add to the task list: (path, filename, culture)
            all_tasks.append((full_path, filename, culture))

    total_files = len(all_tasks)
    if total_files == 0:
        print("❌ No files found to analyze.")
        return

    print(f"✅ Found {total_files} total songs. Starting analysis...\n")

    # --- EXECUTION STEP: SINGLE PROGRESS BAR ---
    # tqdm will now show: [Progress Bar] 25% | 150/600 [00:30<01:30, 5.00it/s]
    for file_path, filename, culture in tqdm(all_tasks, desc="Total NLP Progress", unit="song"):
        
        # --- A. Parse Artist/Track ---
        try:
            base_name = filename.replace(".txt", "")
            if "-" in base_name:
                parts = base_name.rsplit("-", 1)
                track_name = parts[0].strip()
                artist_name = parts[1].strip()
            else:
                track_name = base_name
                artist_name = "Unknown"
        except:
            track_name = filename
            artist_name = "Unknown"

        # --- B. Analysis ---
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_english = f.read()
            
            if not text_english: continue

            # Run the model
            predictions = emotion_classifier(text_english)
            scores = predictions[0]
            
            # Find dominant emotion
            top_emotion = max(scores, key=lambda x: x['score'])
            
            # --- C. Append to List ---
            all_data.append({
                "id": global_id_counter,
                "artists": artist_name,
                "track name": track_name,
                "emotionality": "", # Placeholder for your friend
                "emotion_type": top_emotion['label'].upper(),
                "emotion_score": top_emotion['score'],
                "culture": culture
            })

            global_id_counter += 1

        except Exception as e:
            # tqdm.write allows printing without breaking the progress bar layout
            tqdm.write(f"Error processing {filename}: {e}")
            continue

    # 3. Save CSV
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Reorder columns
        df = df[[
            "id", 
            "artists", 
            "track name", 
            "emotionality", 
            "emotion_type", 
            "emotion_score", 
            "culture"
        ]]
        
        df.to_csv(FINAL_CSV_NAME, index=False)
        print(f"\n✅ PROCESS COMPLETE! Saved to: {FINAL_CSV_NAME}")
        print(f"Total Processed: {len(df)}")
        print("-" * 30)
        print(df.head())
    else:
        print("❌ No data processed.")

if __name__ == "__main__":
    run_analysis()