import pandas as pd

# =========================
# 1. CSV'leri oku
# =========================
human_df = pd.read_csv("Veriler/Ham/arxiv_human_3000.csv")
ai_df = pd.read_csv("Veriler/Ham/ai_abstracts.csv")

# =========================
# 2. Gerekli kolonları seç
# =========================
human_df = human_df[["abstract"]]
ai_df = ai_df[["ai_abstract"]]

# =========================
# 3. Kolon isimlerini normalize et
# =========================
human_df.rename(columns={"abstract": "text"}, inplace=True)
ai_df.rename(columns={"ai_abstract": "text"}, inplace=True)

# =========================
# 4. Label ekle (MANUEL)
# =========================
human_df["label"] = 0   # human
ai_df["label"] = 1      # ai

# =========================
# 5. Birleştir
# =========================
df = pd.concat([human_df, ai_df], ignore_index=True)

# =========================
# 6. NaN temizliği
# =========================
df.dropna(subset=["text"], inplace=True)

# =========================
# 7. Karıştır
# =========================
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# =========================
# 8. Kontroller
# =========================
print(df.head())
print("\nLabel dağılımı:")
print(df["label"].value_counts())
print("\nToplam satır:", len(df))

# =========================
# 9. Kaydet
# =========================
df.to_csv("dataset_combined.csv", index=False)
