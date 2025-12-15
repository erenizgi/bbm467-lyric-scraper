import os
import time
from deep_translator import GoogleTranslator
from tqdm import tqdm

# --- SETTINGS ---
# We assume the Python file is in a subfolder, so we go up one level (../).
# If the code is in the root directory, change "../" to "./".
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
            print(f"⚠️ {input_dir} not found, skipping.")
            continue

        files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
        print(f"\n🌍 Translating {culture} ({len(files)} files)...")

        for filename in tqdm(files, desc=f"{culture}"):
            output_path = os.path.join(output_dir, filename)

            # Skip if already translated (Time saving)
            if os.path.exists(output_path): continue

            try:
                with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
                    text = f.read()

                # Skip empty or very short files
                if len(text) < 10: continue

                # Translating first 600 chars for BERT limit and speed
                translated = translator.translate(text[:600])

                if translated:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(translated)
                    
                    # Short wait to avoid Google API blocking
                    time.sleep(0.5)

            except Exception as e:
                # Continue even if there is an error
                continue
    
    print("\n✅ TRANSLATION COMPLETED!")

if __name__ == "__main__":
    run_translation()