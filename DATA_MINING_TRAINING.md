# Hybrid Movie Recommendation System in WEKA

**Case Study:** Converting Lumina Recommender Architecture to WEKA Explorer

---

## 1. Pipeline Overview

This project converts the hybrid recommendation architecture from python/scikit-learn into WEKA. By merging MovieLens 100K ratings with item metadata, the system implements a content-aware collaborative filtering model using instance-based learning ($k$-NN).

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE ARCHITECTURE                      │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Ingestion       → u.data (Ratings) + u.item (Movie Metadata)        │
│ 2. Feature Fusion  → Python script joins ratings and 19 genre flags    │
│ 3. Export          → Formatted dataset to native hybrid_ratings.arff   │
│ 4. Training        → WEKA IBk (k=10, Cosine Distance)                  │
│ 5. Validation      → 10-Fold Cross-Validation for RMSE & MAE           │
└────────────────────────────────────────────────────────────────────────┘

```

---

## 2. Dataset Specifications

| Property | Value |
| --- | --- |
| **Relation Name** | `hybrid_ratings` |
| **Total Instances** | 100,000 rating interactions |
| **Total Attributes** | 22 attributes |
| **Target Attribute** | `rating` (Numeric, scale 1.0–5.0) |
| **Feature Split** | User interaction (`user_id`, `item_id`) + 19 Genre Flags |

**Attribute List:**

* `user_id` (Numeric)
* `item_id` (Numeric)
* **19 Genre Indicators:** `unknown`, `Action`, `Adventure`, `Animation`, `Childrens`, `Comedy`, `Crime`, `Documentary`, `Drama`, `Fantasy`, `FilmNoir`, `Horror`, `Musical`, `Mystery`, `Romance`, `SciFi`, `Thriller`, `War`, `Western`
* `rating` (Numeric — Target Class)

---

## 3. Data Preprocessing & Fusion

The raw tab-separated ratings (`u.data`) and pipe-separated metadata (`u.item`) were joined on `item_id` using a Python script. Missing entries were handled, and the target vector was positioned as the final attribute in the native WEKA header:

```text
@relation hybrid_ratings

@attribute user_id numeric
@attribute item_id numeric
@attribute unknown {0, 1}
...
@attribute Western {0, 1}
@attribute rating numeric

@data
1,242,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,3

```

---

## 4. Model Configuration & Training

Training was configured inside the **WEKA Explorer GUI** under the **Classify** panel using instance-based distance calculation.

* **Algorithm:** `weka.classifiers.lazy.IBk` ($k$-Nearest Neighbors)
* **Parameters:**
* **Number of Neighbors ($k$):** `10` (`-K 10`)
* **Distance Weighting:** Equal weighting (`-W 0`)
* **Search Algorithm:** `LinearNNSearch` with `EuclideanDistance` across normalized vector space


* **Target Class:** `(Num) rating`
* **Evaluation Scheme:** `10-fold Cross-Validation`

---

## 5. Model Execution Log

```text
=== Run Information ===

Scheme:       weka.classifiers.lazy.IBk -K 10 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""
Relation:     hybrid_ratings
Instances:    100000
Attributes:   22
              user_id
              item_id
              unknown
              Action
              ...
              Western
              rating
Test mode:    10-fold cross-validation

=== Classifier model (full training set) ===

IB1 instance-based classifier
using 10 nearest neighbour(s) for classification

Time taken to build model: 0.01 seconds
Status: Evaluating model for 10-fold cross-validation...

```

---

## 6. Model Evaluation Strategy

Performance is evaluated across 10 folds using standard regression metrics:

* **Mean Absolute Error (MAE):** $\frac{1}{N} \sum \vert{}y_i - \hat{y}_i\vert{}$
* **Root Mean Squared Error (RMSE):** $\sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2}$

Once processing finishes, the model artifact can be saved as `hybrid_recommender.model` via the WEKA Result list for deployment in Java application runtimes.