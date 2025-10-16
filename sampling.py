import pandas as pd
import numpy as np

# ===============================
# CONFIG
# ===============================
INPUT_PATH = "/kaggle/input/hindi-translit/hin_train.json"
OUTPUT_PATH = "/kaggle/working/hin_train_sampled.jsonl"
SAMPLE_SIZE = 100_000
RANDOM_STATE = 42

# ===============================
# 1. LOAD DATA
# ===============================
print("Loading data...")
df = pd.read_json(INPUT_PATH, lines=True)

print(f"Total records loaded: {len(df):,}")

# ===============================
# 2. BASIC CLEANING
# ===============================

# Keep only alphabetic English words (avoid punctuation/digits)
df = df[df["english word"].str.isalpha()]

# Drop duplicates (same english + native)
df = df.drop_duplicates(subset=["english word", "native word"])

# Remove extremely low-probability pairs (bottom 5%)
low_cutoff = df["score"].quantile(0.05)
df = df[df["score"] > low_cutoff]

# Optional: remove very short or very long words
df = df[df["english word"].str.len().between(3, 20)]

print(f"After cleaning: {len(df):,} records remain")

# ===============================
# 3. STRATIFICATION
# ===============================

# Word length bins
df["len_bin"] = pd.cut(
    df["english word"].str.len(),
    bins=[0, 4, 7, 10, 20],
    labels=["short", "medium", "long", "xlong"]
)

# Score bins (quantile-based)
df["score_bin"] = pd.qcut(
    df["score"],
    q=5,
    labels=["very_low", "low", "mid", "high", "very_high"]
)

# Optional: consonant-vowel pattern for phonetic diversity
def to_cv_pattern(word):
    word = word.lower()
    word = ''.join(['V' if ch in 'aeiou' else 'C' for ch in word])
    return word[:10]  # truncate for simplicity

df["cv_pattern"] = df["english word"].apply(to_cv_pattern)

# ===============================
# 4. WEIGHTED SAMPLING
# ===============================

# Weight = combination of score rank (70%) + length rank (30%)
score_rank = df["score"].rank(pct=True)
len_rank = df["english word"].str.len().rank(pct=True)
weights = (0.7 * score_rank + 0.3 * len_rank)
weights = weights / weights.sum()

# Sample 100k with these weights
np.random.seed(RANDOM_STATE)
df_sampled = df.sample(
    n=SAMPLE_SIZE,
    weights=weights,
    random_state=RANDOM_STATE
)

print(f"Sampled {len(df_sampled):,} records.")

# ===============================
# 5. SAVE OUTPUT
# ===============================
df_sampled.to_json(
    OUTPUT_PATH,
    orient="records",
    lines=True,
    force_ascii=False
)

print(f"Saved sampled dataset to: {OUTPUT_PATH}")

# ===============================
# 6. OPTIONAL: DISTRIBUTION CHECK
# ===============================
print("\nDistribution check:")
print(df_sampled["len_bin"].value_counts(normalize=True))
print(df_sampled["score_bin"].value_counts(normalize=True))
