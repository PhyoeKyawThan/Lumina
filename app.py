"""Lumina — movie recommendation API + static UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from posters import render_poster
from recommender import DISPLAY_GENRES, HybridRecommender

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "lumina.joblib"
WEB = ROOT / "web"
STATIC = ROOT / "static"
POSTERS = STATIC / "posters"
POSTERS.mkdir(parents=True, exist_ok=True)

model: HybridRecommender | None = None


class HistoryItem(BaseModel):
    query: str = ""
    movie_id: Optional[int] = None


class RecommendRequest(BaseModel):
    favorite_genres: list[str] = Field(default_factory=list)
    search_history: list[HistoryItem] = Field(default_factory=list)
    liked_ids: list[int] = Field(default_factory=list)
    user_id: Optional[int] = None
    limit: int = 24


def get_model() -> HybridRecommender:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run `python train.py` first.",
        )
    return model


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    if MODEL_PATH.exists():
        model = HybridRecommender.load(MODEL_PATH)
        model.poster_dir = POSTERS
    yield


app = FastAPI(title="Lumina", version="1.0.0", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {
        "ok": model is not None,
        "model_path": str(MODEL_PATH),
        "metrics": None if model is None else model.metrics,
    }


@app.get("/api/genres")
def genres():
    return {"genres": DISPLAY_GENRES}


@app.get("/api/stats")
def stats():
    return get_model().metrics


@app.get("/api/movies")
def list_movies(
    q: str = "",
    genre: str = "",
    sort: str = "bayesian",
    limit: int = Query(24, ge=1, le=60),
    offset: int = Query(0, ge=0),
):
    rec = get_model()
    if q.strip():
        movies = rec.search(q, limit=offset + limit)
        return {"movies": movies[offset : offset + limit], "total": len(movies)}
    if genre.strip():
        movies = rec.by_genre(genre, limit=offset + limit, sort=sort)
        return {"movies": movies[offset : offset + limit], "total": len(movies)}
    if sort == "ratings":
        movies = rec.trending(offset + limit)
    else:
        movies = rec.acclaimed(offset + limit)
    return {"movies": movies[offset : offset + limit], "total": len(rec.movies)}


@app.get("/api/movies/{movie_id}")
def movie_detail(movie_id: int):
    rec = get_model()
    movie = rec.movie_payload(movie_id)
    if not movie:
        raise HTTPException(404, "Movie not found")
    movie["similar"] = rec.similar(movie_id, limit=12)
    return movie


@app.get("/api/movies/{movie_id}/similar")
def movie_similar(movie_id: int, limit: int = 12):
    rec = get_model()
    if movie_id not in rec.by_id:
        raise HTTPException(404, "Movie not found")
    return {"movies": rec.similar(movie_id, limit=limit)}


@app.get("/api/search")
def search(q: str = "", limit: int = Query(20, ge=1, le=40)):
    return {"movies": get_model().search(q, limit=limit), "query": q}


@app.post("/api/recommend")
def recommend(body: RecommendRequest):
    rec = get_model()
    history = [h.model_dump() for h in body.search_history]
    movies = rec.recommend(
        favorite_genres=body.favorite_genres,
        search_history=history,
        liked_ids=body.liked_ids,
        user_id=body.user_id,
        limit=body.limit,
    )
    return {"movies": movies}


@app.post("/api/recommend/rows")
def recommend_rows(body: RecommendRequest):
    rec = get_model()
    return rec.home_rows(
        favorite_genres=body.favorite_genres,
        search_history=[h.model_dump() for h in body.search_history],
        liked_ids=body.liked_ids,
        user_id=body.user_id,
    )


@app.get("/api/users/{user_id}/taste")
def user_taste(user_id: int):
    taste = get_model().user_taste(user_id)
    if not taste:
        raise HTTPException(404, "User not found")
    return taste


@app.get("/api/poster/{movie_id}")
def poster(movie_id: int):
    img_path = POSTERS / f"{movie_id}.jpg"
    if img_path.exists():
        return FileResponse(img_path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})
    rec = get_model()
    movie = rec.by_id.get(movie_id)
    if not movie:
        raise HTTPException(404, "Movie not found")
    svg = render_poster(movie)
    return Response(content=svg, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
