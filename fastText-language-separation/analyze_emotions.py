import os
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

# --- SETTINGS ---
TRANSLATED_FOLDERS = {
    "Turkish": "../lyrics_files_turkish_translated",
    "Balkan": "../lyrics_files_balkan_translated"
}
FINAL_CSV_NAME = "final_music_analysis_dataset.csv"
MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion"

def run_analysis():
    # --- CHECK: GPU Check and Assignment ---
    # If CUDA is available, use device=0 (GPU), otherwise -1 (CPU)
    if torch.cuda.is_available():
        device = 0
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Found: {device_name}")
        print(f"🚀 Operations will be performed on {device_name}.")
    else:
        device = -1
        print("⚠️ GPU not found, running in CPU mode (might be slow).")

    print(f"⏳ Loading Model...")

    # Dynamically passing the device parameter to the pipeline
    classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True, device=device)
    all_data = []

    print("\n🔍 Scanning files...")
    tasks = []
    for culture, path in TRANSLATED_FOLDERS.items():
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith(".txt")]
            for f in files: tasks.append((os.path.join(path, f), f, culture))

    if not tasks:
        print("❌ No files found."); return

    # Loop with TQDM progress bar
    for path, filename, culture in tqdm(tasks, desc="NLP Analysis"):
        try:
            # --- ID PARSING ---
            if "_" in filename:
                parts = filename.split("_", 1)
                
                # Get ID
                try:
                    s_id = int(parts[0]) 
                except ValueError:
                    # Skip if ID is not a number
                    continue

                # Extract names
                rest = parts[1].replace(".txt", "")
                if "-" in rest:
                    p = rest.rsplit("-", 1) 
                    track, artist = p[0].strip(), p[1].strip()
                else:
                    track, artist = rest, "Unknown"
            else:
                continue 

            # Read file
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Empty file check
            if not text or len(text.strip()) == 0: 
                continue

            # Analyze (on GPU)
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
            tqdm.write(f"Error ({filename}): {e}")
            continue

    if all_data:
        # Save CSV
        df = pd.DataFrame(all_data)
        df.to_csv(FINAL_CSV_NAME, index=False)
        print(f"\n✅ Analysis complete! Results saved to '{FINAL_CSV_NAME}'.")
        print(f"Total Processed Songs: {len(df)}")
    else:
        print("❌ Data could not be generated. No files were analyzed.")

if __name__ == "__main__":
    run_analysis()