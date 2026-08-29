#!/usr/bin/env python3
"""Download movie poster images for the Lumina recommender system.

Supports bulk downloading from TMDB (recommended), OMDB, or Wikipedia.
Automatically skips existing files and resumes interrupted runs.

Usage:
    # Get a free API key from https://www.themoviedb.org/settings/api
    python download_posters.py --tmdb-key YOUR_KEY --all

    # Download with OMDB (limited to 1,000/day on free tier)
    python download_posters.py --omdb-key YOUR_KEY --all

    # Try Wikipedia for specific movies (no API key, limited success)
    python download_posters.py --wikipedia --limit 50

    # Download sample posters from public domain sources
    python download_posters.py --sample
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parent
POSTERS_DIR = ROOT / "static" / "posters"
POSTERS_DIR.mkdir(parents=True, exist_ok=True)

MOVIES_CSV = ROOT / "ml-100k" / "u.item"
HEADERS = {"User-Agent": "LuminaRecommender/1.0 (educational project)"}


def load_movies() -> dict[int, dict]:
    movies: dict[int, dict] = {}
    with open(MOVIES_CSV, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 5:
                continue
            item_id = int(parts[0])
            title = parts[1]
            year_match = re.search(r"\((\d{4})\)\s*$", title)
            year = int(year_match.group(1)) if year_match else None
            clean_title = title[: year_match.start()].strip() if year_match else title
            genres = []
            genre_names = [
                "unknown", "Action", "Adventure", "Animation", "Children's",
                "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
                "Sci-Fi", "Thriller", "War", "Western"
            ]
            for i, g in enumerate(parts[5:24]):
                if i < len(genre_names) and g == "1":
                    genres.append(genre_names[i])
            movies[item_id] = {
                "id": item_id,
                "title": clean_title,
                "year": year,
                "genres": genres,
            }
    return movies


def download_file(url: str, dest: Path, timeout: int = 30) -> bool:
    if dest.exists() and dest.stat().st_size > 1024:
        return True
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", ""):
            dest.write_bytes(resp.content)
            return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
    return False


def fetch_from_tmdb(movie: dict, api_key: str, session: requests.Session) -> str | None:
    title = movie["title"]
    year = movie.get("year")
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={quote(title)}"
    if year:
        url += f"&year={year}"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 401:
            print(f"  TMDB auth failed for {title}: invalid API key")
            return None
        if resp.status_code == 429:
            print(f"  TMDB rate limited for {title}: waiting 10s...")
            time.sleep(10)
            return fetch_from_tmdb(movie, api_key, session)
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        poster_path = results[0].get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print(f"  TMDB error for {title}: {e}")
    return None


def fetch_from_omdb(movie: dict, api_key: str, session: requests.Session) -> str | None:
    title = movie["title"]
    year = movie.get("year")
    url = f"https://www.omdbapi.com/?t={quote(title)}&apikey={api_key}"
    if year:
        url += f"&y={year}"
    try:
        resp = session.get(url, timeout=15)
        data = resp.json()
        poster = data.get("Poster")
        if poster and poster != "N/A":
            return poster
    except Exception as e:
        print(f"  OMDB error for {title}: {e}")
    return None


def fetch_from_wikipedia(movie: dict, session: requests.Session) -> str | None:
    title = movie["title"]
    year = movie.get("year")
    search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(title + ' film poster')}&format=json&srlimit=3"
    try:
        resp = session.get(search_url, timeout=15)
        data = resp.json()
        search_results = data.get("query", {}).get("search", [])
        for result in search_results:
            page_title = result.get("title", "")
            if "poster" in page_title.lower() or title.lower() in page_title.lower():
                img_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={quote(page_title)}&prop=images&format=json&imlimit=5"
                img_resp = session.get(img_url, timeout=15)
                img_data = img_resp.json()
                pages = img_data.get("query", {}).get("pages", {})
                for page_id, page in pages.items():
                    for img in page.get("images", []):
                        img_name = img.get("title", "")
                        if "poster" in img_name.lower() or "cover" in img_name.lower():
                            if img_name.startswith("File:"):
                                img_name = img_name[5:]
                            file_url = f"https://en.wikipedia.org/w/api.php?action=query&titles=File:{quote(img_name)}&prop=imageinfo&iiprop=url&format=json"
                            file_resp = session.get(file_url, timeout=15)
                            file_data = file_resp.json()
                            pages = file_data.get("query", {}).get("pages", {})
                            for pid, p in pages.items():
                                for info in p.get("imageinfo", []):
                                    url = info.get("url", "")
                                    if url and ("upload" in url or "commons" in url):
                                        return url
    except Exception as e:
        print(f"  Wikipedia error for {title}: {e}")
    return None


def download_sample_posters():
    """Download a few sample posters from public sources for demonstration."""
    print("Downloading sample posters from public sources...")

    sample_movies = [
        (1, "Toy Story", 1995),
        (50, "Star Wars", 1977),
        (100, "Fargo", 1996),
        (258, "Contact", 1997),
    ]

    public_domain_posters = {
        1: "https://upload.wikimedia.org/wikipedia/en/1/13/Toy_Story.jpg",
        50: "https://upload.wikimedia.org/wikipedia/en/8/87/StarWarsMoviePoster1977.jpg",
        100: "https://upload.wikimedia.org/wikipedia/en/9/97/Fargo_%28film%29.jpg",
    }

    movies = load_movies()
    downloaded = 0

    with requests.Session() as session:
        for movie_id, title, year in sample_movies:
            dest = POSTERS_DIR / f"{movie_id}.jpg"
            if dest.exists() and dest.stat().st_size > 1024:
                print(f"  {title}: already exists")
                downloaded += 1
                continue

            if movie_id in public_domain_posters:
                if download_file(public_domain_posters[movie_id], dest):
                    print(f"  {title}: downloaded from Wikimedia Commons")
                    downloaded += 1
                    continue

            movie = movies.get(movie_id, {"title": title, "year": year})
            url = fetch_from_wikipedia(movie, session)
            if url and download_file(url, dest):
                print(f"  {title}: downloaded from Wikipedia")
                downloaded += 1
                time.sleep(0.5)
                continue

            print(f"  {title}: no poster found")

    print(f"\nDownloaded {downloaded} sample posters.")


def download_all(
    api_key: str | None = None,
    source: str = "tmdb",
    delay: float = 0.35,
    limit: int | None = None,
    verbose: bool = False,
):
    """Download posters for all movies in the dataset.

    Args:
        api_key: TMDB or OMDB API key
        source: 'tmdb', 'omdb', or 'wikipedia'
        delay: Seconds to wait between requests (TMDB allows ~40 req/10s)
        limit: Maximum number of movies to process (None = all)
    """
    if source == "tmdb" and not api_key:
        print("Error: --tmdb-key is required for TMDB source")
        print("Get a free key at: https://www.themoviedb.org/settings/api")
        return
    if source == "omdb" and not api_key:
        print("Error: --omdb-key is required for OMDB source")
        print("Get a free key at: https://www.omdbapi.com/apikey.aspx")
        return

    movies = load_movies()
    total = len(movies)
    if limit:
        total = min(total, limit)

    downloaded = 0
    skipped = 0
    failed = 0
    errors = []

    print(f"Downloading {total} movie posters from {source.upper()}...")
    print(f"Saving to: {POSTERS_DIR}")
    print("-" * 60)

    with requests.Session() as session:
        for idx, (movie_id, movie) in enumerate(movies.items()):
            if limit and idx >= limit:
                break

            dest = POSTERS_DIR / f"{movie_id}.jpg"

            # Skip if already downloaded
            if dest.exists() and dest.stat().st_size > 1024:
                skipped += 1
                if skipped % 100 == 0:
                    print(f"  Progress: {idx+1}/{total} | Downloaded: {downloaded} | Skipped: {skipped} | Failed: {failed}")
                continue

            url = None
            if source == "tmdb":
                url = fetch_from_tmdb(movie, api_key, session)
            elif source == "omdb":
                url = fetch_from_omdb(movie, api_key, session)
            elif source == "wikipedia":
                url = fetch_from_wikipedia(movie, session)

            if url and download_file(url, dest):
                downloaded += 1
                if downloaded % 50 == 0:
                    print(f"  Progress: {idx+1}/{total} | Downloaded: {downloaded} | Skipped: {skipped} | Failed: {failed}")
            else:
                failed += 1
                if verbose:
                    print(f"  FAILED: {movie['title']}")
                errors.append(f"{movie_id}: {movie['title']}")

            # Rate limiting
            time.sleep(delay)

    print("-" * 60)
    print(f"\nDownload complete!")
    print(f"  Total processed: {total}")
    print(f"  Downloaded:      {downloaded}")
    print(f"  Skipped:         {skipped}")
    print(f"  Failed:          {failed}")

    if errors:
        print(f"\nFirst {min(10, len(errors))} failures:")
        for err in errors[:10]:
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Download movie posters for Lumina recommender system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download ALL 1,682 posters from TMDB (requires free API key)
  python download_posters.py --tmdb-key YOUR_KEY --all

  # Download first 50 posters from OMDB
  python download_posters.py --omdb-key YOUR_KEY --limit 50

  # Try Wikipedia for all movies (no API key, lower success rate)
  python download_posters.py --wikipedia --all

  # Download sample posters from public domain
  python download_posters.py --sample

Get API keys:
  TMDB:  https://www.themoviedb.org/settings/api
  OMDB:  https://www.omdbapi.com/apikey.aspx
        """,
    )
    parser.add_argument("--tmdb-key", help="TMDB API key (free from themoviedb.org)")
    parser.add_argument("--omdb-key", help="OMDB API key (free from omdbapi.com)")
    parser.add_argument("--wikipedia", action="store_true", help="Use Wikipedia as source (no API key)")
    parser.add_argument("--sample", action="store_true", help="Download sample posters from public sources")
    parser.add_argument("--all", action="store_true", help="Download posters for ALL movies in dataset")
    parser.add_argument("--limit", type=int, default=50, help="Max posters to download (default: 50)")
    parser.add_argument("--delay", type=float, default=0.35, help="Delay between requests in seconds (default: 0.35)")
    parser.add_argument("--resume", action="store_true", help="Resume from last run (skip existing files)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all failures, not just first 5")

    args = parser.parse_args()

    if args.sample:
        download_sample_posters()
        return

    if not args.all and not args.limit:
        args.limit = 50

    # Determine source
    if args.tmdb_key:
        source = "tmdb"
    elif args.omdb_key:
        source = "omdb"
    elif args.wikipedia:
        source = "wikipedia"
    else:
        print("Error: Please provide one of:")
        print("  --tmdb-key YOUR_KEY    (recommended, free from themoviedb.org)")
        print("  --omdb-key YOUR_KEY    (free from omdbapi.com, 1,000/day limit)")
        print("  --wikipedia            (no key, lower success rate)")
        print("  --sample               (download 4 sample posters)")
        return

    # Validate API key
    if source in ("tmdb", "omdb"):
        key = args.tmdb_key or args.omdb_key
        if not key or key.upper() in ("YOUR_KEY", "YOUR_API_KEY", "PLACEHOLDER", ""):
            print(f"Error: Please provide a real {source.upper()} API key.")
            if source == "tmdb":
                print("Get one free at: https://www.themoviedb.org/settings/api")
            else:
                print("Get one at: https://www.omdbapi.com/apikey.aspx")
            return

    # Resume mode
    if args.resume:
        args.limit = None
        print("Resume mode: skipping already downloaded posters")

    # --all overrides limit
    if args.all:
        args.limit = None

    download_all(
        api_key=args.tmdb_key or args.omdb_key,
        source=source,
        delay=args.delay,
        limit=args.limit,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
