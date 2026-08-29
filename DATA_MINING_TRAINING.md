# Data Mining in Movie Recommendation Systems
## Lumina — A Hybrid Recommender Case Study

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [The Dataset — MovieLens 100K](#2-the-dataset--movielens-100k)
3. [Data Preprocessing & Feature Engineering](#3-data-preprocessing--feature-engineering)
4. [Data Mining Techniques Applied](#4-data-mining-techniques-applied)
5. [The Training Process](#5-the-training-process)
6. [Model Evaluation](#6-model-evaluation)
7. [Hybrid Recommendation Architecture](#7-hybrid-recommendation-architecture)
8. [Deployment & Inference](#8-deployment--inference)

---

## 1. Introduction

Data mining is the process of discovering patterns, correlations, and anomalies within large datasets to predict outcomes and extract actionable knowledge. In the context of recommender systems, data mining transforms raw user–item interaction data into personalized predictions.

**Lumina** is a hybrid movie recommendation system built on four data mining signals:

| Signal | Technique | Purpose |
|--------|-----------|---------|
| Collaborative Filtering | Biased Matrix Factorization (SVD) | "Users who liked X also liked Y" |
| Content-Based Filtering | Cosine Similarity on Genre Vectors | Match movies to user favorite genres |
| Search & History | TF-IDF + Cosine Similarity | Surface titles related to past queries |
| Popularity Prior | Bayesian Average | Robust ranking for cold-start users |

This document explains the complete data mining pipeline — from raw data ingestion to model deployment.

---

## 2. The Dataset — MovieLens 100K

### 2.1 Overview

MovieLens 100K is a benchmark dataset maintained by the GroupLens Research Lab at the University of Minnesota. It contains 100,000 ratings from 943 users on 1,682 movies.

| Property | Value |
|----------|-------|
| Total ratings | 100,000 |
| Unique users | 943 |
| Unique movies | 1,682 |
| Rating scale | 1–5 (integer stars) |
| Collection period | September 1997 – April 1998 |
| Sparsity | ~93.7% (each user rated ~20 of 1,682 movies) |

### 2.2 Data Files

```
ml-100k/
├── u.data           # 100K ratings (user_id, item_id, rating, timestamp)
├── u.item           # 1,682 movies (title, release_date, 19 binary genre flags)
├── u.user           # 943 users (age, gender, occupation, zip)
├── u.info           # Dataset metadata
├── u.genre          # Genre list (19 genres)
├── ua.base / ua.test # Official 80/20 train/test split
└── ub.base / ub.test # Alternative train/test split
```

**u.data** format (tab-separated):
```
user_id \t item_id \t rating \t timestamp
```

**u.item** format (pipe-separated, 24 columns):
```
item_id | title | release_date | video_release_date | imdb_url |
unknown | Action | Adventure | ... | Western
```

The last 19 columns are binary genre indicators (0 or 1).

### 2.3 Data Characteristics

- **Cold-start problem**: New users have no rating history. The system must rely on content signals (genres, popularity) until enough interactions are collected.
- **Sparse matrix**: The user–item rating matrix is 943 × 1,682 with only 5.9% density. Matrix factorization is essential to uncover latent structure.
- **Skewed distributions**: Some movies have hundreds of ratings; others have as few as 1. Bayesian averaging prevents rare 5-star reviews from dominating rankings.

---

## 3. Data Preprocessing & Feature Engineering

### 3.1 Data Loading

The `load_movielens()` function in `recommender.py` reads the three primary data files using `pandas.read_csv()`:

```python
ratings = pd.read_csv("u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
movies  = pd.read_csv("u.item", sep="|", header=None, encoding="latin-1")
users   = pd.read_csv("u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip"])
```

### 3.2 Title Parsing

Raw movie titles include release years in parentheses, e.g., `"Toy Story (1995)"`. A regular expression extracts both components:

```python
YEAR_RE = re.compile(r"\((\d{4})\)\s*$")

def _parse_title(raw):
    match = YEAR_RE.search(raw)
    if not match:
        return raw, None
    return raw[:match.start()].strip(), int(match.group(1))
```

This creates two separate features: `title` (for text search) and `year` (for temporal filtering).

### 3.3 Genre Encoding

The 19 genre columns are already binary indicators. They are extracted as a `genre_matrix` of shape `(1682, 19)` and normalized to unit length using L2 normalization:

```python
genre_matrix = movies[GENRE_COLUMNS].to_numpy(dtype=np.float64)
norms = np.linalg.norm(genre_matrix, axis=1, keepdims=True)
genre_norm = genre_matrix / (norms + 1e-9)
```

This normalization enables cosine similarity between genre vectors.

### 3.4 TF-IDF Document Construction

For search functionality, each movie is represented as a text document combining title, year, and genre tags:

```python
search_docs = []
for _, row in movies.iterrows():
    genres = " ".join(g for g in DISPLAY_GENRES if row[g] == 1)
    year = row["year"] if pd.notna(row["year"]) else ""
    search_docs.append(f"{row['title']} {year} {genres}")
```

These documents are vectorized using `TfidfVectorizer` with 1-2 gram ranges, producing a sparse matrix where rare but discriminative terms (e.g., "space opera", "romantic comedy") receive higher weights.

### 3.5 Bias Extraction

Before matrix factorization, global and per-user/per-item biases are computed:

```python
mu = ratings["rating"].mean()              # Global mean
bu = (user_means - mu).reindex(...)         # User bias
bi = (item_means - mu).reindex(...)         # Item bias
```

The residual matrix `R` captures deviations from these biases:
```
R = ratings - mu - bu - bi
```

This biased approach is standard in recommender systems and significantly improves prediction accuracy by accounting for systematic rater tendencies (e.g., generous vs. strict users, universally popular vs. niche items).

---

## 4. Data Mining Techniques Applied

### 4.1 Matrix Factorization (SVD)

Singular Value Decomposition reduces the high-dimensional, sparse user–item matrix into low-dimensional latent factor vectors.

**Mathematical Formulation:**

The rating prediction for user `u` on item `i` is:

```
r̂(u,i) = μ + b_u + b_i + p_u · q_i
```

Where:
- `μ` = global mean rating
- `b_u` = user bias (user u's tendency to rate high/low)
- `b_i` = item bias (item i's inherent quality)
- `p_u` = user latent factor vector (size k)
- `q_i` = item latent factor vector (size k)

**Implementation:**

`TruncatedSVD` from scikit-learn is applied to the sparse residual matrix `R`:

```python
svd = TruncatedSVD(n_components=k, random_state=42)
P = svd.fit_transform(R)      # User factors (943 × k)
Q = svd.components_.T         # Item factors (1682 × k)
```

The `n_components` hyperparameter controls the dimensionality of latent space. In Lumina, `k=64` balances expressiveness and computational efficiency.

### 4.2 Cosine Similarity

Cosine similarity measures the angle between two vectors, producing a score between -1 and 1:

```
sim(A, B) = (A · B) / (||A|| × ||B||)
```

Used in three contexts:

1. **Item-Item Collaborative Filtering**: `cf_sim = Q_norm @ Q_norm.T`
2. **Genre Matching**: `genre_sim = genre_norm @ genre_norm.T`
3. **Search Ranking**: `cosine_similarity(query_tfidf, title_matrix)`

### 4.3 Hybrid Similarity

CF and content similarities are linearly combined with learned weights:

```python
hybrid_sim = 0.65 * cf_sim + 0.35 * genre_sim
```

The 0.65 / 0.35 split reflects the empirical finding that collaborative signals are stronger predictors of preference, but genre signals provide crucial coverage for cold-start items.

### 4.4 Bayesian Averaging

Raw average ratings are misleading for items with few ratings. Bayesian averaging shrinks estimates toward the global mean based on confidence:

```
bayesian(i) = (C × μ + Σ r_i) / (C + Σ)
```

Where:
- `C` = prior strength (median rating count among rated items, default 20)
- `μ` = global mean rating
- `Σ r_i` = sum of all ratings for item i

Items with few ratings are pulled toward the mean; items with many ratings are trusted more. This is critical for the "Critically Acclaimed" and "Hidden Gems" rails in the UI.

### 4.5 TF-IDF Vectorization

Term Frequency-Inverse Document Frequency converts text documents into numerical vectors:

- **TF**: How often a term appears in a document
- **IDF**: Logarithmic discount for terms that appear in many documents

The `TfidfVectorizer(ngram_range=(1, 2))` captures both single words ("action") and phrases ("sci-fi"), enabling fuzzy title matching and genre-aware search.

---

## 5. The Training Process

### 5.1 Pipeline Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Load Data          → ratings, movies, users              │
│  2. Train/Test Split   → ua.base (train), ua.test (eval)    │
│  3. Preprocess         → parse titles, extract genres        │
│  4. Compute Biases     → mu, bu, bi                          │
│  5. Build Residuals    → R = ratings - mu - bu - bi          │
│  6. Matrix Factorize   → TruncatedSVD → P, Q factors         │
│  7. Content Features   → genre_norm, TF-IDF matrix           │
│  8. Popularity         → Bayesian averages                   │
│  9. Hybrid Similarity  → 0.65×CF + 0.35×genre               │
│ 10. Evaluate           → RMSE, MAE on train & test           │
│ 11. Retrain Full       → fit on all 100K ratings             │
│ 12. Serialize          → models/lumina.joblib                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 Step-by-Step Training

#### Step 1: Load Data

```python
ratings, movies, users = load_movielens("ml-100k")
```

All three dataframes are loaded. Movies are sorted by `item_id` to ensure consistent indexing.

#### Step 2: Train/Test Evaluation Split

The official MovieLens split `ua.base` (80%) / `ua.test` (20%) is used for offline evaluation. Each user has exactly 10 ratings in the test set.

```python
train_ratings = load_split("ua.base")    # 80,000 ratings
test_ratings  = load_split("ua.test")     # 20,000 ratings
```

#### Step 3: Fit the Model on Training Data

```python
model = HybridRecommender(n_components=64)
model.fit(train_ratings, movies, users, test_ratings=test_ratings)
```

Inside `fit()`, the following computations occur in order:

##### 3a. Compute Global Statistics

```python
mu = ratings["rating"].mean()                  # Global mean (~3.53)
```

##### 3b. Compute Biases

```python
user_means = ratings.groupby("user_id")["rating"].mean()
item_means = ratings.groupby("item_id")["rating"].mean()
bu = (user_means - mu)                         # Per-user bias
bi = (item_means - mu)                         # Per-item bias
```

##### 3c. Build Residual Matrix

```python
residuals = ratings - mu - bu[u] - bi[i]
R = csr_matrix((residuals, (rows, cols)), shape=(n_users, n_items))
```

##### 3d. Matrix Factorization

```python
svd = TruncatedSVD(n_components=64, random_state=42)
P = svd.fit_transform(R)    # (943, 64) user factors
Q = svd.components_.T       # (1682, 64) item factors
```

The SVD decomposes `R ≈ P × Q^T`, where `P` captures user preferences and `Q` captures item characteristics in a shared 64-dimensional latent space.

##### 3e. Build Content Features

```python
genre_norm = normalize(genre_matrix)           # (1682, 19)
tfidf = TfidfVectorizer(ngram_range=(1,2))
title_matrix = tfidf.fit_transform(search_docs)  # (1682, ~5000)
```

##### 3f. Compute Popularity Prior

```python
prior = median(counts[counts > 0])             # Typically ~20
bayesian = (prior * mu + counts * avgs) / (prior + counts)
```

##### 3g. Build Hybrid Similarity Matrix

```python
cf_sim = Q_norm @ Q_norm.T                     # Item-item CF similarity
genre_sim = genre_norm @ genre_norm.T          # Genre similarity
hybrid_sim = 0.65 * cf_sim + 0.35 * genre_sim
```

This 1682 × 1682 matrix is precomputed at training time for fast "similar movies" lookups during inference.

#### Step 4: Evaluate on Test Set

```python
test_pred = model._predict_pairs(test_user_ids, test_item_ids)
test_rmse = sqrt(mean((test_true - test_pred)^2))
test_mae  = mean(abs(test_true - test_pred))
```

#### Step 5: Retrain on Full Dataset

After confirming acceptable test metrics, the model is retrained on all 100,000 ratings:

```python
full_model = HybridRecommender(n_components=64)
full_model.fit(ratings, movies, users)
```

The test metrics from the evaluation phase are preserved as metadata:

```python
full_model.metrics["test_rmse"] = test_rmse
full_model.metrics["test_mae"]  = test_mae
```

#### Step 6: Serialize

```python
full_model.save("models/lumina.joblib")
```

The entire `HybridRecommender` object is serialized using `joblib.dump()`, including:
- Latent factors (`P`, `Q`)
- Precomputed matrices (`hybrid_sim`, `genre_norm`, `title_matrix`)
- Fitted vectorizers (`tfidf`)
- Movie catalog and lookup tables

---

## 6. Model Evaluation

### 6.1 Metrics

Two standard regression metrics are used to assess rating prediction accuracy:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | √(Σ(y_true − y_pred)² / N) | Penalizes large errors; primary metric |
| **MAE** | Σ\|y_true − y_pred\| / N | Average absolute error; interpretable |

Both are computed on training and test sets.

### 6.2 Expected Performance

On the MovieLens 100K `ua.base`/`ua.test` split with `n_components=64`:

| Metric | Training | Test |
|--------|----------|------|
| RMSE   | ~0.85    | ~0.95 |
| MAE    | ~0.68    | ~0.75 |

A test RMSE of ~0.95 means predictions are off by less than 1 star on average — reasonable for a lightweight model with no deep learning.

### 6.3 Explained Variance

SVD's `explained_variance_ratio_` indicates how much of the residual matrix's variance is captured by the latent factors. With 64 components:

```
explained_variance ≈ 0.35–0.45
```

This means ~35–45% of the variance in user preferences (after removing biases) is captured by the collaborative filtering model. The remainder reflects noise, randomness, and unmodeled signals (context, mood, etc.).

---

## 7. Hybrid Recommendation Architecture

### 7.1 Signal Composition

During inference, `session_scores()` combines multiple signals with weighted linear interpolation:

```python
def session_scores(favorite_genres, seed_ids, queries, user_id):
    parts = []

    # 1. Popularity Prior (weight: 0.12)
    pop = normalize(bayesian)
    parts.append((0.12, pop))

    # 2. Genre Match (weight: 0.38)
    if favorite_genres:
        genre_score = genre_norm @ genre_vector(favorite_genres)
        parts.append((0.38, genre_score))

    # 3. Seed-Based CF + Content (weight: 0.34)
    if seed_ids:
        cf = Q_norm @ Q_norm[seeds].mean(axis=0)
        content = genre_norm @ genre_norm[seeds].mean(axis=0)
        parts.append((0.34, 0.7 * cf + 0.3 * content))

    # 4. Search TF-IDF (weight: 0.22)
    if queries:
        tfidf_score = cosine_similarity(query_tfidf, title_matrix).max(axis=0)
        parts.append((0.22, tfidf_score))

    # 5. User CF (weight: 0.42) — if MovieLens user_id provided
    if user_id:
        cf_pred = mu + bu[u] + bi + P[u] @ Q.T
        parts.append((0.42, normalize(cf_pred)))

    # Weighted sum with normalization
    scores = sum((w / weight_sum) * vec for w, vec in parts)
    return scores
```

### 7.2 Cold-Start Handling

New users (no rating history) receive recommendations from:
- **Genre preferences** (0.38 weight)
- **Search history** (0.22 weight)
- **Bayesian popularity** (0.12 weight)

No user ID is required. The system personalizes dynamically based on session interactions stored in the frontend's `localStorage`.

### 7.3 Row Generation for the UI

The `home_rows()` method generates Netflix-style categorized rails:

| Row | Signal Source | Example |
|-----|---------------|---------|
| "Top picks for you" | Weighted hybrid scores | Personalized blend |
| "Because you like [Genre]" | Genre filter + Bayesian sort | "Because you like Sci-Fi" |
| "Because you watched [Title]" | `hybrid_sim` neighbors | "Because you watched Toy Story" |
| "Because you searched '[Query]'" | TF-IDF + boost | "Because you searched 'space'" |
| "Most watched" | `rating_count` descending | Popularity |
| "Critically acclaimed" | Bayesian + count ≥ 50 | Quality |
| "Hidden gems" | Bayesian + 20 ≤ count ≤ 90 | Underrated |

---

## 8. Deployment & Inference

### 8.1 Model Serialization

The trained `HybridRecommender` is saved as a single `joblib` file:

```bash
python train.py
# Output: models/lumina.joblib (~several MB)
```

### 8.2 Server Startup

The FastAPI application loads the model at startup using a lifespan context manager:

```python
@asynccontextmanager
async def lifespan(app):
    model = joblib.load("models/lumina.joblib")
    app.state.model = model
    yield
```

### 8.3 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Model status + metrics |
| `/api/stats` | GET | Training metrics (RMSE, MAE, etc.) |
| `/api/movies` | GET | Paginated movie catalog |
| `/api/movies/{id}` | GET | Movie detail + similar titles |
| `/api/search?q=` | GET | TF-IDF search with boosts |
| `/api/recommend` | POST | Hybrid recommendations |
| `/api/recommend/rows` | POST | Netflix-style home rows |
| `/api/users/{id}/taste` | GET | CF taste profile for demo users |

### 8.4 Client-Side State

The frontend persists user signals in `localStorage` under the key `"lumina"`:

```json
{
  "onboarded": true,
  "favoriteGenres": ["Sci-Fi", "Drama"],
  "searchHistory": [{"query": "space", "ts": 1725000000}],
  "likedIds": [1, 50, 200],
  "demoUser": null
}
```

These are sent with every recommendation request, enabling session-level personalization without server-side authentication.

---

## Summary

Lumina demonstrates a complete data mining pipeline for recommender systems:

1. **Data ingestion** from a structured benchmark dataset
2. **Feature engineering** including text parsing, genre encoding, and TF-IDF
3. **Matrix factorization** via TruncatedSVD to learn latent user/item factors
4. **Hybrid scoring** combining collaborative, content, search, and popularity signals
5. **Offline evaluation** using train/test splits with RMSE and MAE
6. **Serialization** for production serving via joblib + FastAPI

The system achieves ~0.95 RMSE on MovieLens 100K while supporting real-time personalization through weighted signal fusion — a practical example of how classical data mining techniques power modern recommendation engines.
