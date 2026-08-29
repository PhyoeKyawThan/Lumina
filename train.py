#!/usr/bin/env python3
"""Train the Lumina hybrid recommender on MovieLens 100K."""

from pathlib import Path

import pandas as pd

from recommender import HybridRecommender, load_movielens

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ml-100k"
MODEL_PATH = ROOT / "models" / "lumina.joblib"


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(
        DATA / name,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )


def main() -> None:
    print("Loading MovieLens 100K…")
    ratings, movies, users = load_movielens(DATA)
    print(f"  {ratings['user_id'].nunique()} users · {len(movies)} movies · {len(ratings)} ratings")

    print("Evaluating biased SVD on the official ua.base / ua.test split…")
    eval_model = HybridRecommender(n_components=64)
    eval_model.fit(load_split("ua.base"), movies, users, test_ratings=load_split("ua.test"))
    m = eval_model.metrics
    print(f"  latent factors     {m['latent_factors']}")
    print(f"  explained variance {m['explained_variance']:.3f}")
    print(f"  train RMSE / MAE   {m['train_rmse']:.3f} / {m['train_mae']:.3f}")
    print(f"  test  RMSE / MAE   {m['test_rmse']:.3f} / {m['test_mae']:.3f}")

    print("Retraining on all 100,000 ratings for serving…")
    model = HybridRecommender(n_components=64)
    model.fit(ratings, movies, users)
    model.metrics["test_rmse"] = m["test_rmse"]
    model.metrics["test_mae"] = m["test_mae"]
    model.metrics["eval_split"] = "ua.base / ua.test"
    model.save(MODEL_PATH)
    print(f"Saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
    print("Done. Start the UI with:  uvicorn app:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
