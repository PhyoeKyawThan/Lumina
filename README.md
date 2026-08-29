# Lumina

A hybrid movie recommender trained on [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) (100,000 ratings, 943 users, 1,682 movies) with a cinematic UI.

Recommendations blend four signals:

1. **Collaborative filtering** — biased SVD (`r = μ + b_u + b_i + p_u · q_i`) so “viewers who liked X also liked Y” still works
2. **Favorite genres** — cosine match against each title’s genre vector
3. **Search history & saves** — queries hit a title/genre TF-IDF index; clicked or saved movies seed item-factor neighbors
4. **Popularity prior** — Bayesian average rating, so a single 5-star doesn’t outrank Titanic

New visitors (cold start) are ranked from genres + search + popularity. Optional “watch as MovieLens user #N” uses that user’s latent factor for a true CF demo.

## Run

```bash
source venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Pick a few genres, search something like `star wars` or `titanic`, save a title, then go home — new rows appear for those signals.

## Training

`train.py` reports RMSE/MAE on the official `ua.base` / `ua.test` split, then retrains on all 100k ratings and writes `models/lumina.joblib`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Model status + metrics |
| GET | `/api/search?q=` | Title / genre search |
| POST | `/api/recommend/rows` | Netflix-style home (hero + rows) |
| POST | `/api/recommend` | Flat ranked list |
| GET | `/api/movies/{id}` | Detail + similar titles |
| GET | `/api/users/{id}/taste` | CF preview for a MovieLens user |
| GET | `/api/poster/{id}` | Generated typographic poster |

POST body:

```json
{
  "favorite_genres": ["Sci-Fi", "Thriller"],
  "search_history": [{"query": "star wars", "movie_id": 50}],
  "liked_ids": [181],
  "user_id": null
}
```

Taste (genres, history, saves) lives in the browser’s `localStorage` so the UI stays personal without accounts.
