# ============================================================
#   TURKISH vs BALKAN SONG EMOTIONALITY — FINAL ANALYSIS
#   (Outputs saved to 'analysis/' folder)
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os  # <-- Added for folder creation

from scipy.stats import (
    ttest_ind, 
    shapiro, 
    mannwhitneyu, 
    levene,
    f_oneway,
    chi2_contingency
)

from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf

# Graph settings
sns.set(style="whitegrid", font_scale=1.2)

# Create output directory
OUTPUT_DIR = "analysis"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"📁 '{OUTPUT_DIR}' folder created.")

# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------

try:
    df = pd.read_csv("FINAL_PROJECT_DATASET.csv")
    df["culture"] = df["culture"].astype(str).str.strip()
    print("✅ Data Loaded. Row Count:", len(df))
except FileNotFoundError:
    print("❌ ERROR: 'FINAL_PROJECT_DATASET.csv' not found.")
    exit()

# -------------------------------------------------------------
# 2. STATISTICAL ANALYSIS (Results printed to Terminal)
# -------------------------------------------------------------

print("\n--- STATISTICAL REPORT STARTING ---")

# A. Chi-Square (Culture vs Emotion Type)
print("\n[1] CHI-SQUARE TEST (Culture vs Emotion Type)")
emotion_table = pd.crosstab(df["emotion_type"], df["culture"])
chi2, p, dof, expected = chi2_contingency(emotion_table)
print(f"Chi2: {chi2:.4f}, p-value: {p:.4f}")
if p < 0.05: print("-> RESULT: Significant relationship between Culture and Emotion Type FOUND.")
else: print("-> RESULT: NO relationship.")

# B. T-Test (Difference in Emotionality)
print("\n[2] T-TEST (Musical Sadness Difference)")
t_vals = df[df["culture"]=="Turkish"]["emotionality"]
b_vals = df[df["culture"]=="Balkan"]["emotionality"]
t_stat, p_val = ttest_ind(t_vals, b_vals, equal_var=False)
d = (t_vals.mean() - b_vals.mean()) / np.sqrt(((t_vals.var() + b_vals.var()) / 2))
print(f"p-value: {p_val:.5f}, Cohen's d: {d:.4f}")

# C. Regression
print("\n[3] REGRESSION ANALYSIS")
model = smf.ols("emotionality ~ C(culture) + C(emotion_type)", data=df).fit()
print(model.summary())

# -------------------------------------------------------------
# 3. VISUALIZATION AND SAVING
# -------------------------------------------------------------
print("\n📊 Saving plots to 'analysis/' folder...")

# Plot 1: Distribution (KDE Plot)
plt.figure(figsize=(10,6))
sns.kdeplot(df[df["culture"]=="Turkish"]["emotionality"], label="Turkish", fill=True, alpha=0.3)
sns.kdeplot(df[df["culture"]=="Balkan"]["emotionality"], label="Balkan", fill=True, alpha=0.3)
plt.title("Emotionality Distribution (Density)")
plt.xlabel("Emotionality Score (Higher = Sadder)")
plt.legend()
save_path = os.path.join(OUTPUT_DIR, "1_emotionality_distribution_kde.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   -> Saved: {save_path}")

# Plot 2: Boxplot
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="culture", y="emotionality", palette="Set2")
plt.title("Emotionality Boxplot by Culture")
save_path = os.path.join(OUTPUT_DIR, "2_emotionality_boxplot.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   -> Saved: {save_path}")

# Plot 3: Normalized Emotion Type (%)
emotion_counts = df.groupby(["culture", "emotion_type"]).size().reset_index(name="count")
emotion_counts["percentage"] = (
    emotion_counts["count"] / emotion_counts.groupby("culture")["count"].transform("sum") * 100
)
plt.figure(figsize=(10,6))
sns.barplot(data=emotion_counts, x="emotion_type", y="percentage", hue="culture", palette="viridis")
plt.title("Emotion Usage Rates by Culture (%)")
plt.ylabel("Percentage (%)")
save_path = os.path.join(OUTPUT_DIR, "3_emotion_type_percentage.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   -> Saved: {save_path}")

# Plot 4: NLP Confidence Distribution (NORMALIZED VERSION)
plt.figure(figsize=(10,6))
sns.histplot(
    data=df, 
    x="emotion_score", 
    hue="culture", 
    kde=True, 
    bins=20,
    stat="percent",      # Y-axis will now show Percentage (%)
    common_norm=False,   # Normalize each culture within itself (equal weight)
    element="step",      # Hollow step style to reduce visual clutter
    fill=True,
    alpha=0.3
)
plt.title("NLP Model Confidence Distribution (Normalized %)")
plt.ylabel("Percentage of Songs within Culture") 
plt.xlabel("Confidence Score (1.0 = Very Sure)")

save_path = os.path.join(OUTPUT_DIR, "4_nlp_confidence_distribution.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   -> Saved: {save_path}")


# Plot 5: INTERACTION PLOT 
plt.figure(figsize=(12, 6))
sns.pointplot(
    data=df, 
    x="emotion_type", 
    y="emotionality", 
    hue="culture", 
    capsize=.1, 
    palette="Set1",
    dodge=True,
    markers=["o", "s"],
    linestyles=["-", "--"]
)
plt.title("INTERACTION PLOT: Lyrical Emotion vs. Audio Emotionality")
plt.ylabel("Audio Emotionality Score (Higher = Sadder)")
plt.xlabel("Lyrical Emotion Label (NLP)")
plt.grid(True, axis='y', alpha=0.3)
save_path = os.path.join(OUTPUT_DIR, "5_interaction_plot_hypothesis.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   -> Saved: {save_path}")

print("\n✅ ALL OPERATIONS COMPLETED! Check 'analysis' folder.")