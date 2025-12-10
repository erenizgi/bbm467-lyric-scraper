import pandas as pd

df_balkan = pd.read_csv('songs_balkan_only.csv')
df_turkish = pd.read_csv('songs_turkish_only.csv')
from sklearn.preprocessing import StandardScaler

audio_cols = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

X_balkan = df_balkan[audio_cols]
X_turkish = df_turkish[audio_cols]

scaler = StandardScaler()
X_balkan_scaled = scaler.fit_transform(X_balkan)
X_turkish_scaled = scaler.fit_transform(X_turkish)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

balkan_pca = pca.fit_transform(X_balkan_scaled)
print("Balkan bileşen yükleri:")
print(pca.components_)

pca2 = PCA(n_components=2)
turkish_pca = pca2.fit_transform(X_turkish_scaled)
print("Türk bileşen yükleri:")
print(pca2.components_)

df_balkan["emotionality_pca"] = balkan_pca[:, 0]
df_turkish["emotionality_pca"] = turkish_pca[:, 0]

from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()

df_balkan["emotionality_pca_norm"] = minmax.fit_transform(df_balkan[["emotionality_pca"]])
df_turkish["emotionality_pca_norm"] = minmax.fit_transform(df_turkish[["emotionality_pca"]])

print("Balkan PCA - Açıklanan varyans oranları:", pca.explained_variance_ratio_)
print("Türk PCA - Açıklanan varyans oranları:", pca2.explained_variance_ratio_)

print("Balkan - 1. bileşenin audio feature'a yükleri:")
for col, value in zip(audio_cols, pca.components_[0]):
    print(f"{col}: {value:.3f}")

print("Türk - 1. bileşenin audio feature'a yükleri:")
for col, value in zip(audio_cols, pca2.components_[0]):
    print(f"{col}: {value:.3f}")