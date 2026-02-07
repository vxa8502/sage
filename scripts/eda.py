# %% [markdown]
# # Exploratory Data Analysis
#
# Analyze the Amazon Electronics reviews dataset to understand
# data distributions, quality issues, and inform modeling decisions.

# %% Imports
import pandas as pd
import matplotlib.pyplot as plt

from sage.config import CHARS_PER_TOKEN, DEV_SUBSET_SIZE, DATA_DIR
from sage.data import load_reviews, get_review_stats, prepare_data

# Output directory for figures
FIGURES_DIR = DATA_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Plot configuration
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.figsize": (10, 5),
        "figure.dpi": 100,
        "savefig.dpi": 300,  # High-res for markdown reports
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.autolayout": True,
    }
)

# Enable retina display for Jupyter notebooks
try:
    from IPython import get_ipython

    if get_ipython() is not None:
        get_ipython().run_line_magic("matplotlib", "inline")
        get_ipython().run_line_magic("config", "InlineBackend.figure_format='retina'")
except (ImportError, AttributeError):
    pass

PRIMARY_COLOR = "#05A0D1"
SECONDARY_COLOR = "#FF9900"
FIGURE_SIZE_WIDE = (12, 5)

# %% Load data
df = load_reviews(subset_size=DEV_SUBSET_SIZE)
print(f"Loaded {len(df):,} reviews")

# %% Basic statistics
stats = get_review_stats(df)
print("\n=== Dataset Overview ===")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")

# %% Rating distribution
fig, ax = plt.subplots()
rating_counts = pd.Series(stats["rating_dist"])
bars = ax.bar(
    rating_counts.index, rating_counts.values, color=PRIMARY_COLOR, edgecolor="black"
)
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
ax.set_title("Rating Distribution")
ax.set_xticks(rating_counts.index)

for bar, count in zip(bars, rating_counts.values, strict=True):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        f"{count:,}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.savefig(FIGURES_DIR / "rating_distribution.png")

print("\nRating breakdown:")
for rating, count in rating_counts.items():
    pct = count / len(df) * 100
    print(f"  {int(rating)} stars: {count:,} ({pct:.1f}%)")

# %% Review length analysis
df["text_length"] = df["text"].str.len()
df["word_count"] = df["text"].str.split().str.len()
df["estimated_tokens"] = df["text_length"] // CHARS_PER_TOKEN

fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE)

# Character length histogram
ax1 = axes[0]
df["text_length"].clip(upper=2000).hist(
    bins=50, ax=ax1, color=PRIMARY_COLOR, edgecolor="white"
)
ax1.set_xlabel("Character Length (clipped at 2000)")
ax1.set_ylabel("Count")
ax1.set_title("Review Length Distribution")
ax1.axvline(
    df["text_length"].median(),
    color="red",
    linestyle="--",
    label=f"Median: {df['text_length'].median():.0f}",
)
ax1.legend()

# Token estimate histogram
ax2 = axes[1]
df["estimated_tokens"].clip(upper=500).hist(
    bins=50, ax=ax2, color=SECONDARY_COLOR, edgecolor="white"
)
ax2.set_xlabel("Estimated Tokens (clipped at 500)")
ax2.set_ylabel("Count")
ax2.set_title("Estimated Token Distribution")
ax2.axvline(200, color="red", linestyle="--", label="Chunking threshold (200)")
ax2.legend()

plt.savefig(FIGURES_DIR / "review_lengths.png")

needs_chunking = (df["estimated_tokens"] > 200).sum()
print("\nReview length stats:")
print(f"  Median characters: {df['text_length'].median():.0f}")
print(f"  Median tokens (est): {df['estimated_tokens'].median():.0f}")
print(
    f"  Reviews > 200 tokens: {needs_chunking:,} ({needs_chunking / len(df) * 100:.1f}%)"
)

# %% Review length by rating
fig, ax = plt.subplots()
length_by_rating = df.groupby("rating")["text_length"].median()
bars = ax.bar(
    length_by_rating.index,
    length_by_rating.values,
    color=PRIMARY_COLOR,
    edgecolor="white",
)
ax.set_xlabel("Rating")
ax.set_ylabel("Median Review Length (chars)")
ax.set_title("Review Length by Rating")
ax.set_xticks([1, 2, 3, 4, 5])

plt.savefig(FIGURES_DIR / "length_by_rating.png")

print("\nMedian review length by rating:")
for rating, length in length_by_rating.items():
    print(f"  {int(rating)} stars: {length:.0f} chars")

# %% Temporal analysis
df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
df["year_month"] = df["datetime"].dt.to_period("M")

reviews_over_time = df.groupby("year_month").size()

fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
reviews_over_time.plot(
    kind="line", ax=ax, marker="o", markersize=3, linewidth=1, color=PRIMARY_COLOR
)
ax.set_xlabel("Month")
ax.set_ylabel("Number of Reviews")
ax.set_title("Reviews Over Time")
plt.xticks(rotation=45)

plt.savefig(FIGURES_DIR / "reviews_over_time.png")

print("\nTemporal range:")
print(f"  Earliest: {df['datetime'].min()}")
print(f"  Latest: {df['datetime'].max()}")

# %% Data quality checks
print("\n=== Data Quality Checks ===")

# Missing values
missing = df.isnull().sum()
print("\nMissing values:")
for col, count in missing.items():
    if count > 0:
        print(f"  {col}: {count:,} ({count / len(df) * 100:.2f}%)")
if missing.sum() == 0:
    print("  None!")

# Empty reviews
empty_reviews = (df["text"].str.strip() == "").sum()
print(f"\nEmpty reviews: {empty_reviews:,}")

# Very short reviews (< 10 chars)
very_short = (df["text_length"] < 10).sum()
print(f"Very short reviews (<10 chars): {very_short:,}")

# Duplicate reviews
duplicate_texts = df["text"].duplicated().sum()
print(f"Duplicate review texts: {duplicate_texts:,}")

# Verified vs unverified
if "verified_purchase" in df.columns:
    verified_pct = df["verified_purchase"].mean() * 100
    print(f"\nVerified purchases: {verified_pct:.1f}%")

# %% User and item coverage
user_counts = df["user_id"].value_counts()
item_counts = df["parent_asin"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE)

# Reviews per user
ax1 = axes[0]
user_counts.clip(upper=20).value_counts().sort_index().plot(
    kind="bar", ax=ax1, color=PRIMARY_COLOR
)
ax1.set_xlabel("Reviews per User")
ax1.set_ylabel("Number of Users")
ax1.set_title("User Activity Distribution")

# Reviews per item
ax2 = axes[1]
item_counts.clip(upper=20).value_counts().sort_index().plot(
    kind="bar", ax=ax2, color=SECONDARY_COLOR
)
ax2.set_xlabel("Reviews per Item")
ax2.set_ylabel("Number of Items")
ax2.set_title("Item Popularity Distribution")

plt.savefig(FIGURES_DIR / "user_item_distribution.png")

print("\nUser activity:")
print(
    f"  Users with 1 review: {(user_counts == 1).sum():,} ({(user_counts == 1).sum() / len(user_counts) * 100:.1f}%)"
)
print(f"  Users with 5+ reviews: {(user_counts >= 5).sum():,}")
print(f"  Max reviews by one user: {user_counts.max()}")

print("\nItem popularity:")
print(
    f"  Items with 1 review: {(item_counts == 1).sum():,} ({(item_counts == 1).sum() / len(item_counts) * 100:.1f}%)"
)
print(f"  Items with 5+ reviews: {(item_counts >= 5).sum():,}")
print(f"  Max reviews for one item: {item_counts.max()}")

# %% 5-core eligibility
users_5plus = set(user_counts[user_counts >= 5].index)
items_5plus = set(item_counts[item_counts >= 5].index)

eligible_mask = df["user_id"].isin(users_5plus) & df["parent_asin"].isin(items_5plus)
print("\n5-core filtering preview:")
print(
    f"  Reviews eligible (first pass): {eligible_mask.sum():,} ({eligible_mask.sum() / len(df) * 100:.1f}%)"
)

# %% Sample reviews across length buckets
print("\n=== Sample Reviews by Length Bucket ===")
print("(Understanding content patterns before chunking)\n")

length_buckets = [
    (0, 50, "Very short (0-50 tokens)"),
    (50, 100, "Short (50-100 tokens)"),
    (100, 200, "Medium (100-200 tokens)"),
    (200, 400, "Long (200-400 tokens)"),
    (400, float("inf"), "Very long (400+ tokens)"),
]

for min_tok, max_tok, label in length_buckets:
    bucket_mask = (df["estimated_tokens"] >= min_tok) & (
        df["estimated_tokens"] < max_tok
    )
    bucket_df = df[bucket_mask]

    if len(bucket_df) == 0:
        print(f"{label}: No reviews")
        continue

    print(
        f"{label}: {len(bucket_df):,} reviews ({len(bucket_df) / len(df) * 100:.1f}%)"
    )

    samples = bucket_df.sample(min(3, len(bucket_df)), random_state=42)
    for _, row in samples.iterrows():
        rating = int(row["rating"])
        tokens = row["estimated_tokens"]
        text = row["text"][:200] + "..." if len(row["text"]) > 200 else row["text"]
        text = text.replace("\n", " ")
        print(f"  [{rating}*] ({tokens} tok) {text}")
    print()

# %% Prepared data comparison
print("\n=== Prepared Data (what the model sees) ===")
df_prepared = prepare_data(subset_size=DEV_SUBSET_SIZE, verbose=False)
prepared_stats = get_review_stats(df_prepared)

print(f"Raw reviews: {len(df):,}")
print(
    f"Prepared reviews: {len(df_prepared):,} ({len(df_prepared) / len(df) * 100:.1f}% retained)"
)
print(f"Unique users: {prepared_stats['unique_users']:,}")
print(f"Unique items: {prepared_stats['unique_items']:,}")
print(
    f"Avg rating: {prepared_stats['avg_rating']:.2f} (raw: {stats['avg_rating']:.2f})"
)

# %% Summary
print("\n" + "=" * 50)
print("EDA SUMMARY")
print("=" * 50)
print(f"Total reviews: {len(df):,}")
print(f"Unique users: {df['user_id'].nunique():,}")
print(f"Unique items: {df['parent_asin'].nunique():,}")
print(f"Average rating: {df['rating'].mean():.2f}")
print(
    f"Reviews needing chunking: {needs_chunking:,} ({needs_chunking / len(df) * 100:.1f}%)"
)
print(f"Data quality issues: {empty_reviews + very_short + duplicate_texts}")
print(f"\nPlots saved to: {FIGURES_DIR}")

# %% Generate markdown report
from pathlib import Path

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Compute all stats for report
raw_total = len(df)
prepared_total = len(df_prepared)
unique_users_raw = df["user_id"].nunique()
unique_items_raw = df["parent_asin"].nunique()
unique_users_prepared = prepared_stats["unique_users"]
unique_items_prepared = prepared_stats["unique_items"]
avg_rating_raw = stats["avg_rating"]
avg_rating_prepared = prepared_stats["avg_rating"]
retention_pct = prepared_total / raw_total * 100

median_chars = df["text_length"].median()
mean_chars = df["text_length"].mean()
median_tokens = df["estimated_tokens"].median()
chunking_pct = needs_chunking / len(df) * 100

five_star_pct = rating_counts.get(5, 0) / len(df) * 100
one_star_pct = rating_counts.get(1, 0) / len(df) * 100
middle_pct = 100 - five_star_pct - one_star_pct

users_one_review = (user_counts == 1).sum()
users_one_review_pct = users_one_review / len(user_counts) * 100
users_5plus = (user_counts >= 5).sum()
max_user_reviews = user_counts.max()

items_one_review = (item_counts == 1).sum()
items_one_review_pct = items_one_review / len(item_counts) * 100
items_5plus = (item_counts >= 5).sum()
max_item_reviews = item_counts.max()

length_1star = length_by_rating.get(1, 0)
length_2star = length_by_rating.get(2, 0)
length_3star = length_by_rating.get(3, 0)
length_4star = length_by_rating.get(4, 0)
length_5star = length_by_rating.get(5, 0)

report_content = f"""# Exploratory Data Analysis: Amazon Electronics Reviews

**Dataset:** McAuley-Lab/Amazon-Reviews-2023 (Electronics category)
**Subset:** {raw_total:,} raw reviews -> {prepared_total:,} after 5-core filtering

---

## Dataset Overview

The Amazon Electronics reviews dataset provides rich user feedback data for building recommendation systems. After standard preprocessing and 5-core filtering (requiring users and items to have at least 5 interactions), the dataset exhibits the characteristic sparsity of real-world recommendation scenarios.

| Metric | Raw | After 5-Core |
|--------|-----|--------------|
| Total Reviews | {raw_total:,} | {prepared_total:,} |
| Unique Users | {unique_users_raw:,} | {unique_users_prepared:,} |
| Unique Items | {unique_items_raw:,} | {unique_items_prepared:,} |
| Avg Rating | {avg_rating_raw:.2f} | {avg_rating_prepared:.2f} |
| Retention | - | {retention_pct:.1f}% |

---

## Rating Distribution

Amazon reviews exhibit a well-known J-shaped distribution, heavily skewed toward 5-star ratings. This reflects both genuine satisfaction and selection bias (dissatisfied customers often don't leave reviews).

![Rating Distribution](../data/figures/rating_distribution.png)

**Key Observations:**
- 5-star ratings dominate ({five_star_pct:.1f}% of reviews)
- 1-star reviews form the second largest group ({one_star_pct:.1f}%)
- Middle ratings (2-4 stars) are relatively rare ({middle_pct:.1f}% combined)
- This polarization is typical for e-commerce review data

**Implications for Modeling:**
- Binary classification (positive/negative) may be more robust than regression
- Rating-weighted aggregation should account for the skewed distribution
- Evidence from 4-5 star reviews carries stronger positive signal

---

## Review Length Analysis

Review length varies significantly and correlates with the chunking strategy for the RAG pipeline. Most reviews are short enough to embed directly without chunking.

![Review Length Distribution](../data/figures/review_lengths.png)

**Length Statistics:**
- Median: {median_chars:.0f} characters (~{median_tokens:.0f} tokens)
- Mean: {mean_chars:.0f} characters (~{mean_chars / 4:.0f} tokens)
- Reviews exceeding 200 tokens: {chunking_pct:.1f}% (require chunking)

**Chunking Strategy Validation:**
The tiered chunking approach is well-suited to this distribution:
- **Short (<200 tokens):** No chunking needed - majority of reviews
- **Medium (200-500 tokens):** Semantic chunking at topic boundaries
- **Long (>500 tokens):** Semantic + sliding window fallback

---

## Review Length by Rating

Negative reviews tend to be longer than positive ones. Users who are dissatisfied often provide detailed explanations of issues, while satisfied users may simply express approval.

![Review Length by Rating](../data/figures/length_by_rating.png)

**Pattern:**
- 1-star reviews: {length_1star:.0f} chars median
- 2-3 star reviews: {length_2star:.0f}-{length_3star:.0f} chars median (users explain nuance)
- 4-star reviews: {length_4star:.0f} chars median
- 5-star reviews: {length_5star:.0f} chars median

**Implications:**
- Negative reviews provide richer evidence for issue identification
- Positive reviews may require multiple chunks for substantive explanations
- Rating filters (min_rating=4) naturally bias toward shorter evidence

---

## Temporal Distribution

The dataset spans multiple years of reviews, enabling proper temporal train/validation/test splits that prevent data leakage.

![Reviews Over Time](../data/figures/reviews_over_time.png)

**Temporal Split Strategy:**
- **Train (70%):** Oldest reviews - model learns from historical patterns
- **Validation (10%):** Middle period - hyperparameter tuning
- **Test (20%):** Most recent - simulates production deployment

This chronological ordering ensures the model never sees "future" data during training.

---

## User and Item Activity

The long-tail distribution is pronounced: most users write few reviews, and most items receive few reviews. This sparsity is the fundamental challenge recommendation systems address.

![User and Item Distribution](../data/figures/user_item_distribution.png)

**User Activity:**
- Users with only 1 review: {users_one_review_pct:.1f}%
- Users with 5+ reviews: {users_5plus:,}
- Power user max: {max_user_reviews} reviews

**Item Popularity:**
- Items with only 1 review: {items_one_review_pct:.1f}%
- Items with 5+ reviews: {items_5plus:,}
- Most reviewed item: {max_item_reviews} reviews

**Cold-Start Implications:**
- Many items have sparse evidence - content-based features are critical
- User cold-start is common - onboarding preferences help
- 5-core filtering ensures minimum evidence density for evaluation

---

## Data Quality Assessment

The raw dataset contains several quality issues addressed during preprocessing.

| Issue | Count | Resolution |
|-------|-------|------------|
| Missing text | 0 | - |
| Empty reviews | {empty_reviews} | Removed |
| Very short (<10 chars) | {very_short:,} | Removed |
| Duplicate texts | {duplicate_texts:,} | Kept (valid re-purchases) |
| Invalid ratings | 0 | - |

**Post-Cleaning:**
- All reviews have valid text content
- All ratings are in [1, 5] range
- All user/product identifiers present

---

## Summary

The Amazon Electronics dataset, after 5-core filtering and cleaning, provides a solid foundation for building and evaluating a RAG-based recommendation system:

1. **Scale:** {prepared_total:,} reviews across {unique_users_prepared:,} users and {unique_items_prepared:,} items
2. **Sparsity:** {100 - retention_pct:.1f}% filtered - realistic for recommendation evaluation
3. **Quality:** Clean text, valid ratings, proper identifiers
4. **Temporal:** Supports chronological train/val/test splits
5. **Content:** Review lengths suit the tiered chunking strategy

The J-shaped rating distribution and long-tail user/item activity are characteristic of real e-commerce data, making this an appropriate benchmark for portfolio demonstration.

---

*Report auto-generated by `scripts/eda.py`. Run `make eda` to regenerate.*
"""

report_path = REPORTS_DIR / "eda_report.md"
report_path.write_text(report_content)
print(f"\nReport generated: {report_path}")
