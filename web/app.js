const GENRE_COLORS = {
  Action: "#c23b22",
  Adventure: "#c47b17",
  Animation: "#3d8b7a",
  "Children's": "#e0b25a",
  Comedy: "#e9c46a",
  Crime: "#1d3557",
  Documentary: "#6d6875",
  Drama: "#6d2e46",
  Fantasy: "#7b2cbf",
  "Film-Noir": "#8d99ae",
  Horror: "#6a040f",
  Musical: "#c9184a",
  Mystery: "#2b2d42",
  Romance: "#c9184a",
  "Sci-Fi": "#0077b6",
  Thriller: "#4a4e69",
  War: "#606c38",
  Western: "#bc6c25",
};

const store = {
  read() {
    try {
      return JSON.parse(localStorage.getItem("lumina") || "{}");
    } catch {
      return {};
    }
  },
  write(patch) {
    const next = { ...this.read(), ...patch };
    localStorage.setItem("lumina", JSON.stringify(next));
    return next;
  },
};

function state() {
  const s = store.read();
  return {
    onboarded: Boolean(s.onboarded),
    favoriteGenres: s.favoriteGenres || [],
    searchHistory: s.searchHistory || [],
    likedIds: s.likedIds || [],
    demoUser: s.demoUser || null,
  };
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

function payload() {
  const s = state();
  return {
    favorite_genres: s.favoriteGenres,
    search_history: s.searchHistory.slice(-12),
    liked_ids: s.likedIds,
    user_id: s.demoUser,
  };
}

function stars(n) {
  if (n == null) return "";
  return `${Number(n).toFixed(1)} ★`;
}

function metaLine(m) {
  return [m.year, (m.genres || []).slice(0, 3).join(" · "), stars(m.avg_rating)]
    .filter(Boolean)
    .join("  ·  ");
}

function toast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add("hidden"), 2200);
}

function card(movie) {
  return `
    <button class="card" data-open="${movie.id}" type="button">
      <div class="poster"><img alt="" src="${movie.poster}" /></div>
      <p class="card-title">${escapeHtml(movie.title)}</p>
      <p class="card-sub">${movie.year || ""} · ${stars(movie.avg_rating)}</p>
    </button>`;
}

function rail(row) {
  const id = `rail-${row.id}`;
  return `
    <section class="rail-section">
      <div class="rail-head">
        <div>
          <h2>${escapeHtml(row.title)}</h2>
          <p>${escapeHtml(row.reason || "")}</p>
        </div>
      </div>
      <div class="rail-wrap">
        <button class="rail-btn left" type="button" data-scroll="${id}" data-dir="-1" aria-label="Scroll left">‹</button>
        <div class="rail" id="${id}">${row.movies.map(card).join("")}</div>
        <button class="rail-btn right" type="button" data-scroll="${id}" data-dir="1" aria-label="Scroll right">›</button>
      </div>
    </section>`;
}

function escapeHtml(s) {
  return String(s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function hash() {
  const raw = location.hash.replace(/^#/, "") || "/";
  const [path, query] = raw.split("?");
  const params = new URLSearchParams(query || "");
  return { path, params };
}

const app = document.getElementById("app");

async function renderHome() {
  app.innerHTML = `<div class="hero"><div class="skeleton" style="height:72vh;width:100%"></div></div>`;
  try {
    const data = await api("/api/recommend/rows", {
      method: "POST",
      body: JSON.stringify(payload()),
    });
    const hero = data.hero;
    const heroHtml = hero
      ? `<section class="hero">
          <div class="hero-bg" style="background-image:url('${hero.poster}')"></div>
          <div class="hero-copy">
            <p class="eyebrow">Tonight on Lumina</p>
            <h1>${escapeHtml(hero.title)}</h1>
            <div class="meta"><span>${escapeHtml(metaLine(hero))}</span>
              <span>${hero.rating_count || 0} ratings</span></div>
            <span class="reason">${escapeHtml(hero.reason || "Recommended for you")}</span>
            <div class="hero-actions">
              <button class="btn gold" data-open="${hero.id}" type="button">More info</button>
              <button class="btn ghost" data-like="${hero.id}" type="button">${likeLabel(hero.id)}</button>
            </div>
          </div>
        </section>`
      : "";
    app.innerHTML = heroHtml + data.rows.map(rail).join("");
  } catch (err) {
    app.innerHTML = `<section class="browse"><h1 class="page-title">Lumina isn’t ready</h1>
      <p class="empty">${escapeHtml(err.message)}. Train the model with <code>python train.py</code> then restart the server.</p></section>`;
  }
}

async function renderBrowse(genre) {
  const genres = (await api("/api/genres")).genres;
  const active = genre || "";
  const qs = active ? `?genre=${encodeURIComponent(active)}&limit=48` : "?sort=bayesian&limit=48";
  const { movies } = await api(`/api/movies${qs}`);
  app.innerHTML = `
    <section class="browse">
      <p class="eyebrow">Catalog</p>
      <h1 class="page-title">${active ? escapeHtml(active) : "Browse the archive"}</h1>
      <p class="muted">1,682 titles from MovieLens 100K. Filter by the genres you love.</p>
      <div class="filters">
        <button class="chip ${active ? "" : "on"}" data-genre="">All</button>
        ${genres
          .map(
            (g) =>
              `<button class="chip ${g === active ? "on" : ""}" data-genre="${escapeHtml(g)}">${escapeHtml(g)}</button>`
          )
          .join("")}
      </div>
      <div class="grid">${movies.map(card).join("")}</div>
    </section>`;
}

async function renderSearch(q) {
  const query = (q || "").trim();
  rememberSearch(query);
  const { movies } = await api(`/api/search?q=${encodeURIComponent(query)}&limit=30`);
  let extra = "";
  if (movies[0]) {
    const sim = await api(`/api/movies/${movies[0].id}/similar`);
    if (sim.movies.length) {
      extra = rail({
        id: "more-like",
        title: `More like ${movies[0].title}`,
        reason: "Neighbors from collaborative filtering and shared genres",
        movies: sim.movies,
      });
    }
  }
  app.innerHTML = `
    <section class="search-page">
      <p class="eyebrow">Search</p>
      <h1 class="page-title">${escapeHtml(query)}</h1>
      <p class="muted">${movies.length} titles · this query is now part of your recommendation history</p>
      <div class="grid">${movies.map(card).join("") || `<p class="empty">No titles matched. Try a 90s classic — Star Wars, Titanic, Toy Story.</p>`}</div>
    </section>${extra}`;
}

function renderSaved() {
  const s = state();
  const ids = s.likedIds;
  app.innerHTML = `<section class="saved"><p class="eyebrow">Library</p>
    <h1 class="page-title">Saved for later</h1>
    <p class="muted">Saves are strong signals — they weigh more than a search.</p>
    <div id="saved-grid" class="grid"></div></section>`;
  if (!ids.length) {
    document.getElementById("saved-grid").innerHTML =
      `<p class="empty">Nothing saved yet. Open a title and tap Save — Lumina will treat it as a favorite.</p>`;
    return;
  }
  Promise.all(ids.map((id) => api(`/api/movies/${id}`)))
    .then((movies) => {
      document.getElementById("saved-grid").innerHTML = movies.map(card).join("");
    })
    .catch(() => {
      document.getElementById("saved-grid").innerHTML = `<p class="empty">Could not load saved titles.</p>`;
    });
}

function likeLabel(id) {
  return state().likedIds.includes(Number(id)) ? "Saved" : "Save";
}

function toggleLike(id) {
  id = Number(id);
  const liked = new Set(state().likedIds);
  if (liked.has(id)) liked.delete(id);
  else liked.add(id);
  store.write({ likedIds: [...liked] });
  toast(liked.has(id) ? "Saved to your library" : "Removed from saved");
  const btn = document.querySelector(`[data-like="${id}"]`);
  if (btn) btn.textContent = likeLabel(id);
}

function rememberSearch(query, movieId) {
  if (!query && !movieId) return;
  const history = state().searchHistory.filter(
    (h) => !(h.query === query && h.movie_id === (movieId || null))
  );
  history.push({ query: query || "", movie_id: movieId || null, at: Date.now() });
  store.write({ searchHistory: history.slice(-40) });
}

async function openMovie(id, fromQuery) {
  const movie = await api(`/api/movies/${id}`);
  if (fromQuery) rememberSearch(fromQuery, Number(id));
  const modal = document.getElementById("modal");
  modal.classList.remove("hidden");
  modal.innerHTML = `
    <div class="modal-card">
      <div class="modal-poster"><img alt="" src="${movie.poster}" /></div>
      <div class="modal-body">
        <button class="icon-btn" id="close-modal" type="button" aria-label="Close">×</button>
        <p class="eyebrow">${(movie.genres || []).join(" · ")}</p>
        <h2>${escapeHtml(movie.title)}</h2>
        <div class="meta">
          <span>${movie.year || ""}</span>
          <span class="star">${stars(movie.avg_rating)}</span>
          <span>${movie.rating_count} ratings</span>
        </div>
        <p class="muted" style="margin:16px 0 20px">
          Neighbors are a blend of viewers who rated this highly and films that share its genre DNA.
        </p>
        <div class="hero-actions">
          <button class="btn gold" data-like="${movie.id}" type="button">${likeLabel(movie.id)}</button>
          ${movie.imdb_url ? `<a class="btn ghost" href="${movie.imdb_url}" target="_blank" rel="noopener">IMDb</a>` : ""}
        </div>
        <h3 style="margin:28px 0 10px;font-size:14px;letter-spacing:.08em;text-transform:uppercase;color:var(--gold)">More like this</h3>
        <div class="similar-mini">
          ${(movie.similar || [])
            .map(
              (m) =>
                `<img data-open="${m.id}" alt="${escapeHtml(m.title)}" title="${escapeHtml(m.title)}" src="${m.poster}" />`
            )
            .join("")}
        </div>
      </div>
    </div>`;
}

function closeModal() {
  const modal = document.getElementById("modal");
  modal.classList.add("hidden");
  modal.innerHTML = "";
}

async function route() {
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.remove("active"));
  const { path, params } = hash();
  if (path.startsWith("/search")) {
    await renderSearch(params.get("q") || "");
  } else if (path.startsWith("/browse")) {
    document.querySelector('[data-nav="browse"]')?.classList.add("active");
    await renderBrowse(params.get("genre") || "");
  } else if (path.startsWith("/saved")) {
    document.querySelector('[data-nav="saved"]')?.classList.add("active");
    renderSaved();
  } else {
    document.querySelector('[data-nav="home"]')?.classList.add("active");
    await renderHome();
  }
}

function showOnboarding(genres) {
  const root = document.getElementById("onboarding");
  const grid = document.getElementById("genre-grid");
  const selected = new Set(state().favoriteGenres);
  grid.innerHTML = genres
    .map((g) => {
      const color = GENRE_COLORS[g] || "#d4a853";
      return `<button type="button" class="genre-card ${selected.has(g) ? "on" : ""}" data-pick="${escapeHtml(g)}" style="--accent:${color}">${escapeHtml(g)}</button>`;
    })
    .join("");
  document.getElementById("onboard-go").disabled = selected.size === 0;
  root.classList.remove("hidden");
}

function syncOnboardButton() {
  document.getElementById("onboard-go").disabled = state().favoriteGenres.length === 0;
}

function renderProfile() {
  const s = state();
  const box = document.getElementById("profile-genres");
  api("/api/genres").then(({ genres }) => {
    box.innerHTML = genres
      .map(
        (g) =>
          `<button type="button" class="chip ${s.favoriteGenres.includes(g) ? "on" : ""}" data-fav="${escapeHtml(g)}">${escapeHtml(g)}</button>`
      )
      .join("");
  });
  const list = document.getElementById("history-list");
  const hist = [...s.searchHistory].reverse().slice(0, 12);
  list.innerHTML = hist.length
    ? hist
        .map(
          (h) =>
            `<li><span>${escapeHtml(h.query || "Opened a title")}${h.movie_id ? ` · #${h.movie_id}` : ""}</span></li>`
        )
        .join("")
    : "<li class='muted'>No searches yet</li>";
  const demoMeta = document.getElementById("demo-meta");
  const clear = document.getElementById("demo-clear");
  if (s.demoUser) {
    demoMeta.textContent = `Watching as MovieLens user #${s.demoUser}`;
    clear.classList.remove("hidden");
    document.getElementById("demo-user").value = s.demoUser;
  } else {
    demoMeta.textContent = "";
    clear.classList.add("hidden");
  }
}

function openDrawer(open) {
  const d = document.getElementById("drawer");
  d.classList.toggle("hidden", !open);
  d.setAttribute("aria-hidden", open ? "false" : "true");
  if (open) renderProfile();
}

let suggestTimer;
async function onSearchInput(e) {
  const q = e.target.value.trim();
  const box = document.getElementById("suggest");
  if (!q) {
    box.classList.add("hidden");
    return;
  }
  clearTimeout(suggestTimer);
  suggestTimer = setTimeout(async () => {
    const { movies } = await api(`/api/search?q=${encodeURIComponent(q)}&limit=6`);
    if (!movies.length) {
      box.classList.add("hidden");
      return;
    }
    box.innerHTML = movies
      .map(
        (m) =>
          `<button type="button" data-open="${m.id}" data-from-q="${escapeHtml(q)}">
            <img alt="" src="${m.poster}" />
            <span>${escapeHtml(m.title)}<small>${escapeHtml(metaLine(m))}</small></span>
          </button>`
      )
      .join("");
    box.classList.remove("hidden");
  }, 160);
}

document.addEventListener("click", (e) => {
  const pick = e.target.closest("[data-pick]");
  if (pick) {
    const g = pick.dataset.pick;
    const set = new Set(state().favoriteGenres);
    if (set.has(g)) set.delete(g);
    else set.add(g);
    store.write({ favoriteGenres: [...set] });
    pick.classList.toggle("on");
    syncOnboardButton();
    return;
  }
  const fav = e.target.closest("[data-fav]");
  if (fav) {
    const g = fav.dataset.fav;
    const set = new Set(state().favoriteGenres);
    if (set.has(g)) set.delete(g);
    else set.add(g);
    store.write({ favoriteGenres: [...set] });
    fav.classList.toggle("on");
    return;
  }
  const genre = e.target.closest("[data-genre]");
  if (genre) {
    const g = genre.dataset.genre;
    location.hash = g ? `#/browse?genre=${encodeURIComponent(g)}` : "#/browse";
    return;
  }
  const like = e.target.closest("[data-like]");
  if (like) {
    toggleLike(like.dataset.like);
    return;
  }
  const open = e.target.closest("[data-open]");
  if (open) {
    document.getElementById("suggest").classList.add("hidden");
    openMovie(open.dataset.open, open.dataset.fromQ);
    return;
  }
  const sc = e.target.closest("[data-scroll]");
  if (sc) {
    const el = document.getElementById(sc.dataset.scroll);
    el.scrollBy({ left: Number(sc.dataset.dir) * el.clientWidth * 0.85, behavior: "smooth" });
  }
});

document.getElementById("onboard-go").addEventListener("click", () => {
  store.write({ onboarded: true });
  document.getElementById("onboarding").classList.add("hidden");
  route();
});
document.getElementById("onboard-skip").addEventListener("click", () => {
  store.write({ onboarded: true });
  document.getElementById("onboarding").classList.add("hidden");
  route();
});
document.getElementById("profile-btn").addEventListener("click", () => openDrawer(true));
document.querySelector(".close-drawer").addEventListener("click", () => openDrawer(false));
document.getElementById("drawer").addEventListener("click", (e) => {
  if (e.target.id === "drawer") openDrawer(false);
});
document.getElementById("clear-history").addEventListener("click", () => {
  store.write({ searchHistory: [] });
  renderProfile();
  toast("Search history cleared");
});
document.getElementById("demo-go").addEventListener("click", async () => {
  const id = Number(document.getElementById("demo-user").value);
  if (!id || id < 1 || id > 943) {
    toast("Enter a user id from 1 to 943");
    return;
  }
  try {
    const taste = await api(`/api/users/${id}/taste`);
    store.write({
      demoUser: id,
      favoriteGenres: taste.inferred_genres || state().favoriteGenres,
    });
    renderProfile();
    toast(`Now recommending as user #${id}`);
    location.hash = "#/";
    route();
  } catch {
    toast("That user was not found");
  }
});
document.getElementById("demo-clear").addEventListener("click", () => {
  store.write({ demoUser: null });
  renderProfile();
  route();
});
document.getElementById("modal").addEventListener("click", (e) => {
  if (e.target.id === "modal" || e.target.id === "close-modal") closeModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeModal();
    openDrawer(false);
    document.getElementById("suggest").classList.add("hidden");
  }
  if (e.key === "/" && document.activeElement.tagName !== "INPUT") {
    e.preventDefault();
    document.getElementById("search-input").focus();
  }
});

const searchInput = document.getElementById("search-input");
searchInput.addEventListener("input", onSearchInput);
searchInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    const q = searchInput.value.trim();
    document.getElementById("suggest").classList.add("hidden");
    if (q) location.hash = `#/search?q=${encodeURIComponent(q)}`;
  }
});
document.addEventListener("click", (e) => {
  if (!e.target.closest(".search-wrap")) {
    document.getElementById("suggest").classList.add("hidden");
  }
});

window.addEventListener("hashchange", route);

(async function boot() {
  let genres = [];
  try {
    genres = (await api("/api/genres")).genres;
  } catch {
    app.innerHTML = `<section class="browse"><h1 class="page-title">Train the model first</h1>
      <p class="empty">Run <code>python train.py</code> then restart <code>uvicorn app:app --port 8000</code>.</p></section>`;
    return;
  }
  if (!state().onboarded) showOnboarding(genres);
  else await route();
})();
