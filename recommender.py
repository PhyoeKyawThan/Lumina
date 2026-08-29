"""Hybrid movie recommender trained on MovieLens 100K.

Signals
-------
* Biased SVD collaborative filtering (users who liked X also liked Y)
* Favorite-genre content match
* Search-history / liked-title similarity (item factors + TF-IDF)
* Bayesian-average popularity (cold-start prior)
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GENRE_COLUMNS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

DISPLAY_GENRES = [g for g in GENRE_COLUMNS if g != "unknown"]
YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def _parse_title(raw: str) -> tuple[str, int | None]:
    raw = str(raw).strip()
    match = YEAR_RE.search(raw)
    if not match:
        return raw, None
    return raw[: match.start()].strip(), int(match.group(1))


def load_movielens(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    ratings = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    movies = pd.read_csv(
        data_dir / "u.item",
        sep="|",
        header=None,
        encoding="latin-1",
        engine="python",
    )
    movies.columns = ["item_id", "raw_title", "release_date", "video_release_date", "imdb_url"] + GENRE_COLUMNS
    users = pd.read_csv(
        data_dir / "u.user",
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"],
        engine="python",
    )
    parsed = movies["raw_title"].map(_parse_title)
    movies["title"] = parsed.map(lambda x: x[0])
    movies["year"] = parsed.map(lambda x: x[1])
    return ratings, movies, users


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


class HybridRecommender:
    def __init__(self, n_components: int = 64, poster_dir: str | Path | None = None):
        self.n_components = n_components
        self.poster_dir = Path(poster_dir) if poster_dir else None
        self.movies: list[dict] = []
        self.by_id: dict[int, dict] = {}
        self.titles: list[str] = []
        self.item_ids: np.ndarray | None = None
        self.id_to_idx: dict[int, int] = {}
        self.n_users = 0
        self.n_items = 0
        self.mu = 0.0
        self.bu: np.ndarray | None = None
        self.bi: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self.Q_norm: np.ndarray | None = None
        self.genre_matrix: np.ndarray | None = None
        self.genre_norm: np.ndarray | None = None
        self.tfidf: TfidfVectorizer | None = None
        self.title_matrix = None
        self.bayesian: np.ndarray | None = None
        self.avg_rating: np.ndarray | None = None
        self.rating_count: np.ndarray | None = None
        self.hybrid_sim: np.ndarray | None = None
        self.metrics: dict = {}
        self.users: list[dict] = []

    def fit(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        users: pd.DataFrame | None = None,
        test_ratings: pd.DataFrame | None = None,
    ) -> "HybridRecommender":
        self.n_users = int(ratings["user_id"].max())
        if test_ratings is not None:
            self.n_users = max(self.n_users, int(test_ratings["user_id"].max()))
        self.n_items = int(movies["item_id"].max())

        movies = movies.sort_values("item_id").reset_index(drop=True)
        self.item_ids = movies["item_id"].to_numpy()
        self.id_to_idx = {int(i): idx for idx, i in enumerate(self.item_ids)}

        stats = ratings.groupby("item_id")["rating"].agg(["mean", "count"])
        self.mu = float(ratings["rating"].mean())
        counts = stats["count"].reindex(self.item_ids).fillna(0).to_numpy()
        avgs = stats["mean"].reindex(self.item_ids).fillna(self.mu).to_numpy()
        prior = float(np.median(counts[counts > 0])) if np.any(counts > 0) else 20.0
        self.rating_count = counts
        self.avg_rating = avgs
        self.bayesian = (prior * self.mu + counts * avgs) / (prior + counts)

        self.genre_matrix = movies[GENRE_COLUMNS].to_numpy(dtype=np.float64)
        norms = np.linalg.norm(self.genre_matrix, axis=1, keepdims=True)
        self.genre_norm = self.genre_matrix / (norms + 1e-9)

        self.titles = movies["title"].fillna("").tolist()
        search_docs = []
        for _, row in movies.iterrows():
            genres = " ".join(g for g in DISPLAY_GENRES if row[g] == 1)
            year = row["year"] if pd.notna(row["year"]) else ""
            search_docs.append(f"{row['title']} {year} {genres}")
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)
        self.title_matrix = self.tfidf.fit_transform(search_docs)

        user_means = ratings.groupby("user_id")["rating"].mean()
        item_means = ratings.groupby("item_id")["rating"].mean()
        self.bu = (user_means - self.mu).reindex(range(1, self.n_users + 1), fill_value=0.0).to_numpy()
        self.bi = (item_means - self.mu).reindex(self.item_ids, fill_value=0.0).to_numpy()

        residuals = (
            ratings["rating"].to_numpy()
            - self.mu
            - self.bu[ratings["user_id"].to_numpy() - 1]
            - np.array([self.bi[self.id_to_idx[int(i)]] for i in ratings["item_id"]])
        )
        rows = ratings["user_id"].to_numpy() - 1
        cols = np.array([self.id_to_idx[int(i)] for i in ratings["item_id"]])
        R = csr_matrix((residuals, (rows, cols)), shape=(self.n_users, len(self.item_ids)))

        k = min(self.n_components, min(R.shape) - 1)
        svd = TruncatedSVD(n_components=k, random_state=42)
        self.P = svd.fit_transform(R)
        self.Q = svd.components_.T
        qn = np.linalg.norm(self.Q, axis=1, keepdims=True)
        self.Q_norm = self.Q / (qn + 1e-9)

        cf_sim = self.Q_norm @ self.Q_norm.T
        genre_sim = self.genre_norm @ self.genre_norm.T
        self.hybrid_sim = 0.65 * cf_sim + 0.35 * genre_sim
        np.fill_diagonal(self.hybrid_sim, 0.0)

        self.movies = []
        for idx, row in movies.iterrows():
            genres = [g for g in DISPLAY_GENRES if row[g] == 1]
            year = int(row["year"]) if pd.notna(row["year"]) else None
            movie = {
                "id": int(row["item_id"]),
                "title": row["title"],
                "year": year,
                "genres": genres,
                "avg_rating": round(float(self.avg_rating[idx]), 2),
                "rating_count": int(self.rating_count[idx]),
                "bayesian": round(float(self.bayesian[idx]), 3),
                "imdb_url": row["imdb_url"] if isinstance(row["imdb_url"], str) else None,
            }
            self.movies.append(movie)
        self.by_id = {m["id"]: m for m in self.movies}

        self.users = []
        if users is not None:
            for _, row in users.iterrows():
                self.users.append(
                    {
                        "id": int(row["user_id"]),
                        "age": int(row["age"]),
                        "gender": row["gender"],
                        "occupation": row["occupation"],
                    }
                )

        train_pred = self._predict_pairs(ratings["user_id"].to_numpy(), ratings["item_id"].to_numpy())
        self.metrics = {
            "n_users": self.n_users,
            "n_movies": len(self.movies),
            "n_ratings": int(len(ratings)),
            "latent_factors": int(k),
            "explained_variance": float(svd.explained_variance_ratio_.sum()),
            "train_rmse": _rmse(ratings["rating"].to_numpy(), train_pred),
            "train_mae": _mae(ratings["rating"].to_numpy(), train_pred),
        }
        if test_ratings is not None and len(test_ratings):
            test_pred = self._predict_pairs(
                test_ratings["user_id"].to_numpy(), test_ratings["item_id"].to_numpy()
            )
            self.metrics["test_rmse"] = _rmse(test_ratings["rating"].to_numpy(), test_pred)
            self.metrics["test_mae"] = _mae(test_ratings["rating"].to_numpy(), test_pred)
        return self

    def _predict_pairs(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        u = user_ids.astype(int) - 1
        idx = np.array([self.id_to_idx.get(int(i), 0) for i in item_ids])
        valid = np.array([int(i) in self.id_to_idx for i in item_ids])
        u = np.clip(u, 0, self.n_users - 1)
        preds = self.mu + self.bu[u] + self.bi[idx] + np.sum(self.P[u] * self.Q[idx], axis=1)
        preds = np.clip(preds, 1.0, 5.0)
        preds = np.where(valid, preds, self.mu)
        return preds

    def movie_payload(self, movie_id: int, extra: dict | None = None) -> dict | None:
        movie = self.by_id.get(int(movie_id))
        if not movie:
            return None
        payload = dict(movie)
        if self.poster_dir and (self.poster_dir / f"{movie_id}.jpg").exists():
            payload["poster"] = f"/static/posters/{movie_id}.jpg"
        else:
            payload["poster"] = f"/api/poster/{movie_id}"
        if extra:
            payload.update(extra)
        return payload

    def _genre_vector(self, genres: list[str]) -> np.ndarray:
        vec = np.zeros(len(GENRE_COLUMNS), dtype=np.float64)
        lookup = {g.lower(): i for i, g in enumerate(GENRE_COLUMNS)}
        for name in genres:
            i = lookup.get(str(name).lower())
            if i is not None:
                vec[i] = 1.0
        n = np.linalg.norm(vec)
        return vec / n if n else vec

    def _seed_indices(self, movie_ids: list[int]) -> list[int]:
        out = []
        seen = set()
        for mid in movie_ids:
            idx = self.id_to_idx.get(int(mid))
            if idx is not None and idx not in seen:
                seen.add(idx)
                out.append(idx)
        return out

    def session_scores(
        self,
        favorite_genres: list[str] | None = None,
        seed_ids: list[int] | None = None,
        queries: list[str] | None = None,
        user_id: int | None = None,
    ) -> np.ndarray:
        n = len(self.movies)
        parts: list[tuple[float, np.ndarray]] = []

        pop = self.bayesian.copy()
        pop = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)
        parts.append((0.12, pop))

        if favorite_genres:
            gvec = self._genre_vector(favorite_genres)
            if gvec.sum() > 0:
                parts.append((0.38, self.genre_norm @ gvec))

        seeds = self._seed_indices(seed_ids or [])
        if seeds:
            cf = self.Q_norm @ self.Q_norm[seeds].mean(axis=0)
            content = self.genre_norm @ self.genre_norm[seeds].mean(axis=0)
            parts.append((0.34, 0.7 * cf + 0.3 * content))

        clean_queries = [q.strip() for q in (queries or []) if q and q.strip()]
        if clean_queries and self.tfidf is not None:
            qmat = self.tfidf.transform(clean_queries)
            tscore = cosine_similarity(qmat, self.title_matrix).max(axis=0)
            parts.append((0.22, np.asarray(tscore).ravel()))

        if user_id and 1 <= int(user_id) <= self.n_users:
            u = int(user_id) - 1
            cf_pred = self.mu + self.bu[u] + self.bi + self.P[u] @ self.Q.T
            cf_pred = (cf_pred - cf_pred.min()) / (cf_pred.max() - cf_pred.min() + 1e-9)
            parts.append((0.42, cf_pred))

        weight_sum = sum(w for w, _ in parts) or 1.0
        scores = np.zeros(n)
        for w, vec in parts:
            scores += (w / weight_sum) * vec

        for idx in seeds:
            scores[idx] = -1e9
        return scores

    def _top_movies(self, scores: np.ndarray, limit: int, exclude: set[int] | None = None) -> list[dict]:
        order = np.argsort(-scores)
        out = []
        exclude = exclude or set()
        for idx in order:
            movie = self.movies[int(idx)]
            if movie["id"] in exclude:
                continue
            if scores[idx] < -1e8:
                continue
            out.append(self.movie_payload(movie["id"]))
            if len(out) >= limit:
                break
        return out

    def recommend(
        self,
        favorite_genres: list[str] | None = None,
        search_history: list[dict] | None = None,
        liked_ids: list[int] | None = None,
        user_id: int | None = None,
        limit: int = 24,
        exclude: list[int] | None = None,
    ) -> list[dict]:
        seeds, queries = self._history_signals(search_history, liked_ids)
        scores = self.session_scores(favorite_genres, seeds, queries, user_id)
        return self._top_movies(scores, limit, set(exclude or []))

    def _history_signals(
        self, search_history: list[dict] | None, liked_ids: list[int] | None
    ) -> tuple[list[int], list[str]]:
        seeds = list(liked_ids or [])
        queries: list[str] = []
        for item in search_history or []:
            if not isinstance(item, dict):
                continue
            mid = item.get("movie_id")
            if mid:
                seeds.append(int(mid))
            q = (item.get("query") or "").strip()
            if q:
                queries.append(q)
        return seeds, queries

    def similar(self, movie_id: int, limit: int = 12) -> list[dict]:
        idx = self.id_to_idx.get(int(movie_id))
        if idx is None:
            return []
        sims = self.hybrid_sim[idx]
        return self._top_movies(sims, limit, {int(movie_id)})

    def search(self, query: str, limit: int = 24) -> list[dict]:
        q = (query or "").strip()
        if not q:
            return []
        tfidf_s = cosine_similarity(self.tfidf.transform([q]), self.title_matrix).ravel()
        ql = q.lower()
        boost = np.zeros(len(self.movies))
        for i, movie in enumerate(self.movies):
            title = movie["title"].lower()
            if title == ql:
                boost[i] += 1.2
            elif title.startswith(ql) or ql in title:
                boost[i] += 0.55
            if any(ql == g.lower() or ql in g.lower() for g in movie["genres"]):
                boost[i] += 0.25
        scores = tfidf_s + boost
        return self._top_movies(scores, limit)

    def by_genre(self, genre: str, limit: int = 24, sort: str = "bayesian") -> list[dict]:
        g = genre.lower()
        scored = []
        for i, movie in enumerate(self.movies):
            if not any(x.lower() == g for x in movie["genres"]):
                continue
            if sort == "ratings":
                key = self.rating_count[i]
            elif sort == "year":
                key = movie["year"] or 0
            else:
                key = self.bayesian[i]
            scored.append((key, movie["id"]))
        scored.sort(reverse=True)
        return [self.movie_payload(mid) for _, mid in scored[:limit]]

    def trending(self, limit: int = 24) -> list[dict]:
        order = np.argsort(-self.rating_count)
        return [self.movie_payload(self.movies[int(i)]["id"]) for i in order[:limit]]

    def acclaimed(self, limit: int = 24) -> list[dict]:
        mask = self.rating_count >= 50
        scores = np.where(mask, self.bayesian, -1)
        return self._top_movies(scores, limit)

    def hidden_gems(self, limit: int = 24) -> list[dict]:
        mask = (self.rating_count >= 20) & (self.rating_count <= 90)
        scores = np.where(mask, self.bayesian, -1)
        return self._top_movies(scores, limit)

    def user_taste(self, user_id: int, limit: int = 12) -> dict | None:
        if not (1 <= int(user_id) <= self.n_users):
            return None
        u = int(user_id) - 1
        pred = self.mu + self.bu[u] + self.bi + self.P[u] @ self.Q.T
        top = self._top_movies(pred, limit)
        genre_counts: dict[str, float] = {}
        for movie in top:
            for g in movie["genres"]:
                genre_counts[g] = genre_counts.get(g, 0) + 1
        inferred = sorted(genre_counts, key=genre_counts.get, reverse=True)[:4]
        profile = next((x for x in self.users if x["id"] == int(user_id)), None)
        return {"user": profile, "inferred_genres": inferred, "top_predictions": top}

    def home_rows(
        self,
        favorite_genres: list[str] | None = None,
        search_history: list[dict] | None = None,
        liked_ids: list[int] | None = None,
        user_id: int | None = None,
        per_row: int = 16,
    ) -> dict:
        favorite_genres = favorite_genres or []
        search_history = search_history or []
        liked_ids = liked_ids or []
        seeds, queries = self._history_signals(search_history, liked_ids)
        used: set[int] = set(seeds)

        for_you = self.recommend(
            favorite_genres, search_history, liked_ids, user_id, limit=per_row, exclude=list(used)
        )
        used.update(m["id"] for m in for_you)
        hero = for_you[0] if for_you else (self.acclaimed(1)[0] if self.movies else None)
        if hero:
            hero = dict(hero)
            hero["reason"] = self._hero_reason(favorite_genres, search_history, liked_ids, user_id)

        rows = []
        if for_you:
            rows.append(
                {
                    "id": "for-you",
                    "title": "Top picks for you",
                    "reason": "Matched to your genres, searches, and similar viewers",
                    "movies": for_you,
                }
            )

        for genre in favorite_genres[:3]:
            movies = self.by_genre(genre, limit=per_row)
            if movies:
                rows.append(
                    {
                        "id": f"genre-{genre.lower()}",
                        "title": f"Because you like {genre}",
                        "reason": f"Highly rated {genre} from MovieLens 100K",
                        "movies": movies,
                    }
                )

        last_seed = next((s for s in reversed(seeds) if s in self.by_id), None)
        if last_seed:
            src = self.by_id[last_seed]
            similar = self.similar(last_seed, limit=per_row)
            if similar:
                rows.append(
                    {
                        "id": f"similar-{last_seed}",
                        "title": f"Because you watched {src['title']}",
                        "reason": "Collaborative + genre neighbors of a title you engaged with",
                        "movies": similar,
                    }
                )

        last_query = next((q for q in reversed(queries) if q.strip()), None)
        if last_query:
            related = self.search(last_query, limit=per_row)
            related = [m for m in related if m["id"] not in (seeds or [])][:per_row]
            if related:
                rows.append(
                    {
                        "id": "search-echo",
                        "title": f'Because you searched “{last_query}”',
                        "reason": "Title and genre neighbors of your recent search",
                        "movies": related,
                    }
                )

        rows.append(
            {
                "id": "trending",
                "title": "Most watched",
                "reason": "Movies with the most ratings in MovieLens 100K",
                "movies": self.trending(per_row),
            }
        )
        rows.append(
            {
                "id": "acclaimed",
                "title": "Critically acclaimed",
                "reason": "Bayesian average of 50+ ratings, so one-off 5-stars don't dominate",
                "movies": self.acclaimed(per_row),
            }
        )
        rows.append(
            {
                "id": "gems",
                "title": "Hidden gems",
                "reason": "Loved by viewers, but not the usual blockbusters",
                "movies": self.hidden_gems(per_row),
            }
        )
        return {"hero": hero, "rows": rows, "metrics": self.metrics}

    def _hero_reason(self, genres, history, liked, user_id) -> str:
        if user_id:
            return f"Predicted for MovieLens user #{int(user_id)}"
        if liked:
            return "Picked from titles you saved"
        if history:
            return "Tuned to your recent searches"
        if genres:
            return f"Because you like {', '.join(genres[:2])}"
        return "A standout from the MovieLens 100K catalog"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "HybridRecommender":
        return joblib.load(path)
