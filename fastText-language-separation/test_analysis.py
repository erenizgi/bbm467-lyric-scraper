import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# basit bir test. silmek istemedim kalsın bir süre.

# 1. Temiz CSV'leri Yükle
df_tr = pd.read_csv("songs_turkish_only.csv")
df_balkan = pd.read_csv("songs_balkan_only.csv")

# Etiketle ve Birleştir
df_tr['culture'] = 'Turkish'
df_balkan['culture'] = 'Balkan'
df_all = pd.concat([df_tr, df_balkan])

# 2. İstatistiksel Test (T-Test)
# Valence (Mutluluk) karşılaştırması
t_stat, p_val = stats.ttest_ind(df_tr['valence'], df_balkan['valence'], equal_var=False)

print(f"Turkish Mean Valence: {df_tr['valence'].mean():.4f}")
print(f"Balkan Mean Valence:  {df_balkan['valence'].mean():.4f}")
print(f"P-Value: {p_val:.5f}")

if p_val < 0.05:
    print("SONUÇ: İki kültür arasında anlamlı bir fark VAR!")
else:
    print("SONUÇ: Anlamlı bir fark YOK.")

# 3. Görselleştirme
sns.boxplot(x='culture', y='valence', data=df_all, hue='culture', palette="Set2", legend=False)
plt.title('Comparison of Musical Positivity (Valence)')
plt.show()
