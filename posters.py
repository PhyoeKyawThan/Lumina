"""Deterministic typographic posters — one look per movie, no external art."""

from __future__ import annotations

import hashlib

GENRE_PALETTES = {
    "Action": ("#1b0b0b", "#c23b22", "#f2a65a"),
    "Adventure": ("#1c1408", "#c47b17", "#f0d6a6"),
    "Animation": ("#14201c", "#3d8b7a", "#f3c98b"),
    "Children's": ("#1a1810", "#e0b25a", "#8ecae6"),
    "Comedy": ("#1a160c", "#e9c46a", "#f4a261"),
    "Crime": ("#0c1018", "#1d3557", "#e63946"),
    "Documentary": ("#141416", "#6d6875", "#c9ada7"),
    "Drama": ("#140c12", "#6d2e46", "#d4a373"),
    "Fantasy": ("#120818", "#7b2cbf", "#e0aaff"),
    "Film-Noir": ("#090909", "#2f2f2f", "#d6d3cd"),
    "Horror": ("#100000", "#6a040f", "#dc2f02"),
    "Musical": ("#180814", "#c9184a", "#ffc2d4"),
    "Mystery": ("#0c1018", "#2b2d42", "#8d99ae"),
    "Romance": ("#16080c", "#c9184a", "#ffb3c1"),
    "Sci-Fi": ("#061018", "#0077b6", "#90e0ef"),
    "Thriller": ("#0c0c14", "#4a4e69", "#9a8c98"),
    "War": ("#10140c", "#606c38", "#dda15e"),
    "Western": ("#14100a", "#bc6c25", "#e6ccb2"),
    "unknown": ("#101014", "#6c757d", "#dee2e6"),
}


def _hash_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _wrap(title: str, width: int = 16) -> list[str]:
    words = title.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if len(trial) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
        if len(lines) == 3:
            break
    if current and len(lines) < 4:
        lines.append(current)
    if len(lines) == 4 and words:
        lines[-1] = lines[-1][: width - 1] + "…"
    return lines or [title[:width]]


def render_poster(movie: dict) -> str:
    title = movie.get("title") or "Untitled"
    year = movie.get("year") or ""
    genres = movie.get("genres") or []
    primary = genres[0] if genres else "Drama"
    bg, mid, accent = GENRE_PALETTES.get(primary, GENRE_PALETTES["Drama"])
    if len(genres) > 1:
        accent = GENRE_PALETTES.get(genres[1], GENRE_PALETTES["Drama"])[2]

    seed = _hash_int(f"{movie.get('id')}-{title}")
    layout = seed % 4
    lines = _wrap(title, 14)
    font_size = 42 if max(len(x) for x in lines) < 12 else 32
    genre_line = " · ".join(genres[:3]) if genres else "Film"
    initial = next((ch.upper() for ch in title if ch.isalnum()), "L")

    shapes = ""
    if layout == 0:
        shapes = f"""
        <circle cx="320" cy="90" r="110" fill="{accent}" fill-opacity="0.22"/>
        <circle cx="40" cy="520" r="90" fill="{mid}" fill-opacity="0.35"/>
        <rect x="28" y="28" width="8" height="544" fill="{accent}"/>
        """
    elif layout == 1:
        shapes = f"""
        <polygon points="0,0 400,0 400,220 0,140" fill="{mid}" fill-opacity="0.55"/>
        <circle cx="200" cy="430" r="120" fill="none" stroke="{accent}" stroke-width="10"/>
        <circle cx="200" cy="430" r="48" fill="{accent}" fill-opacity="0.8"/>
        """
    elif layout == 2:
        shapes = f"""
        <rect x="0" y="0" width="400" height="600" fill="{mid}" fill-opacity="0.25"/>
        <rect x="46" y="70" width="308" height="460" fill="none" stroke="{accent}" stroke-width="3"/>
        <rect x="0" y="250" width="400" height="18" fill="{accent}" fill-opacity="0.7"/>
        """
    else:
        shapes = f"""
        <rect x="0" y="0" width="400" height="210" fill="{mid}"/>
        <text x="200" y="168" text-anchor="middle" font-family="Georgia, serif"
              font-size="150" fill="{accent}" fill-opacity="0.35">{_esc(initial)}</text>
        <rect x="30" y="470" width="340" height="2" fill="{accent}" fill-opacity="0.6"/>
        """

    title_block = []
    start_y = 250 if layout != 3 else 300
    for i, line in enumerate(lines):
        title_block.append(
            f'<text x="48" y="{start_y + i * (font_size + 8)}" font-family="Georgia, Times New Roman, serif" '
            f'font-size="{font_size}" font-weight="600" fill="#f7f1e5">{_esc(line)}</text>'
        )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 600" width="400" height="600">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{bg}"/>
      <stop offset="100%" stop-color="{mid}"/>
    </linearGradient>
  </defs>
  <rect width="400" height="600" fill="url(#g)"/>
  {shapes}
  <text x="48" y="64" font-family="Helvetica, Arial, sans-serif" font-size="13"
        letter-spacing="4" fill="{accent}">{_esc(str(year) if year else "CLASSIC")}</text>
  {''.join(title_block)}
  <text x="48" y="560" font-family="Helvetica, Arial, sans-serif" font-size="13"
        fill="#f7f1e5" fill-opacity="0.72">{_esc(genre_line)}</text>
</svg>
"""
