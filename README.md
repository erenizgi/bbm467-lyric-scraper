# 🎵 Cultural Music Analysis: Turkish vs. Balkan Pop
**BBM467 - Data Intensive Applications Project**

This project performs a statistical comparison of cultural differences between Turkish and Balkan popular music through **Lyrical** and **Audio** emotional analysis. The pipeline involves web scraping, language detection (fastText), machine translation, NLP (BERT), and PCA-based music analysis.

***

## 🚀 Installation

Please install the required dependencies before running the project.

### 1. Python Requirements
The project is tested with Python 3.11. Run the following inside the `fastText-language-separation` directory:

```bash
cd fastText-language-separation
pip install pandas numpy seaborn matplotlib scikit-learn scipy statsmodels deep-translator torch transformers tqdm requests fasttext
```

> **Note:** If you encounter a NumPy version error, run `pip install "numpy<2.0"`.

### 2. Frontend (Scraper) Requirements
Node.js must be installed. Run the following in the project root directory:

```bash
npm install
```

### 3. Genius API Configuration
Create a `.env.local` file in the project root directory and add your Genius API key:

```bash
CLIENT_TOKEN=YOUR_GENIUS_ACCESS_TOKEN_HERE
CLIENT_SECRET=YOUR_SECRET_HERE
CLIENT_ID=YOUR_CLIENT_ID_HERE
```

***

## 🛠️ Execution Pipeline
Follow these steps in order to perform the analysis from scratch.

### Step 1: Language Detection & Data Separation
First, we detect the languages of the songs and separate them into Turkish and Balkan datasets.

Download the FastText model:
```bash
python download_model.py
```

Label songs by language:
```bash
python fastText-separate.py
```

Separate CSV files:
```bash
python separateTurkish.py
python separateBalkans.py
```
**Output:** `songs_turkish_only.csv` and `songs_balkan_only.csv`

### Step 2: Scraping Lyrics ⚠️ IMPORTANT
In this step, we use the Frontend interface to fetch lyrics from Genius.

1. Start the frontend server (from the root directory):
   ```bash
   npm run dev
   ```
2. Open your browser and go to `http://localhost:3000`.

**For Turkish Songs:**
* Enter `./songs_turkish_only.csv` into the input field and click "Process".
* Once finished, a `lyrics_files` folder will appear in the root directory.
* 🛑 **STOP AND RENAME:** Rename this folder to `lyrics_files_turkish`.

**For Balkan Songs:**
* Enter `./songs_balkan_only.csv` into the input field and click "Process".
* Rename the newly created `lyrics_files` folder to `lyrics_files_balkan`.

> At the end of this step, you must have `lyrics_files_turkish` and `lyrics_files_balkan` folders in the root directory.

### Step 3: Translation & NLP Analysis
We translate the downloaded lyrics into English and perform emotion analysis using a BERT model.

Start the translation process (uses Google Translate API):
```bash
cd fastText-language-separation
python translate_lyrics.py
```
**Output:** `lyrics_files_turkish_translated` and `lyrics_files_balkan_translated` folders are created.

Run Emotion Analysis (BERT):
```bash
python analyze_emotions.py
```
**Output:** `final_music_analysis_dataset.csv` (Lyrical analysis results).

### Step 4: Final Dataset & PCA Analysis
We merge the NLP results with Spotify Audio data and calculate the Emotionality Index using PCA (Principal Component Analysis).

```bash
python dscp.py
```
**Output:** `FINAL_PROJECT_DATASET.csv` (Final file ready for analysis).

### Step 5: Statistical Analysis & Visualization
Finally, we perform hypothesis testing (T-Test, ANOVA, Chi-Square) and generate visualization plots.

```bash
python dscp_analysis.py
```
📊 **Results:** All plots and reports are saved as high-resolution PNG files in the `fastText-language-separation/analysis/` folder.

* `5_interaction_plot_hypothesis.png`: The key chart proving the main hypothesis ("Sad Joy").
* `1_emotionality_distribution_kde.png`: Distribution of cultural musical sadness.
* `3_emotion_type_percentage.png`: Usage rates of emotion types by culture.

***

## 📂 Project Structure

```plaintext
.
├── src/                        # Next.js Frontend codes
├── lyrics_files_turkish/       # Downloaded Turkish lyrics (Txt)
├── lyrics_files_balkan/        # Downloaded Balkan lyrics (Txt)
├── lyrics_files_turkish_translated/
├── lyrics_files_balkan_translated/
├── .env                  # API Token
├── fastText-language-separation/
│   ├── analysis/               # Output plots (PNG)
│   ├── download_model.py       # Downloads FastText model
│   ├── fastText-separate.py    # Language detection logic
│   ├── separateTurkish.py      # Filters Turkish songs
│   ├── separateBalkans.py      # Filters Balkan songs
│   ├── translate_lyrics.py     # Google Translate wrapper
│   ├── analyze_emotions.py     # BERT NLP Analysis
│   ├── dscp.py                 # PCA & Data Merging
│   ├── dscp_analysis.py        # Statistical Tests & Plotting
│   └── FINAL_PROJECT_DATASET.csv
└── README.md
```

***

## 👥 Contributors
[Yusuf Emir Cömert] - Hacettepe University, AI Engineering\
[Eren İzgi] - Hacettepe University, Computer Engineering\
[Sudenaz Yazıcı] - Hacettepe University, Computer Engineering
