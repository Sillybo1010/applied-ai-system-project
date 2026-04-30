"""
CineMatch — Interactive Streamlit Web App
Run: streamlit run src/app.py
"""

import os
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from recommender import (
    UserProfile, Recommender, load_movies,
    confidence_score, _DEFAULT_WEIGHTS,
)

_BASE         = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH     = os.path.join(_BASE, "data", "movies.csv")

GENRES = ["action", "sci-fi", "drama", "animation", "horror",
          "comedy", "thriller", "romance", "war", "western"]
MOODS  = ["thrilling", "emotional", "heartwarming", "epic", "scary",
          "quirky", "dark", "romantic", "intense", "fun", "disturbing"]
MODES  = ["balanced", "genre-first", "mood-first", "vibe-first"]

RATE_LIMIT    = 10      # max requests
WINDOW_SECS   = 3600    # per hour


@st.cache_resource
def load_data():
    movies = load_movies(DATA_PATH)
    return Recommender(movies)


# ── Rate limiter ──────────────────────────────────────────────────────────────

def requests_this_hour() -> list:
    """Return timestamps from the last hour, pruning stale ones."""
    now = time.time()
    history = st.session_state.get("request_times", [])
    fresh = [t for t in history if now - t < WINDOW_SECS]
    st.session_state["request_times"] = fresh
    return fresh


def record_request():
    history = st.session_state.get("request_times", [])
    history.append(time.time())
    st.session_state["request_times"] = history


def rate_limit_status():
    history = requests_this_hour()
    used      = len(history)
    remaining = RATE_LIMIT - used
    if history:
        oldest      = min(history)
        resets_in   = int(WINDOW_SECS - (time.time() - oldest))
        reset_mins  = resets_in // 60
        reset_secs  = resets_in % 60
        reset_str   = f"{reset_mins}m {reset_secs}s"
    else:
        reset_str = "—"
    return used, remaining, reset_str


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

st.title("🎬 CineMatch")
st.caption("Tell us what you're in the mood for and we'll find the right film.")

rec = load_data()

# ── Sidebar — Profile Builder ─────────────────────────────────────────────────

with st.sidebar:
    st.header("What are you in the mood for?")

    name = st.text_input("Your name", value="Movie Fan")

    col1, col2 = st.columns(2)
    with col1:
        genre = st.selectbox("Favourite genre", GENRES)
    with col2:
        mood = st.selectbox("Preferred mood", MOODS, index=8)

    intensity = st.slider("Intensity",            0.0, 1.0, 0.80, 0.05,
                          help="0 = calm, 1 = high-octane")
    runtime   = st.slider("Target runtime (min)", 60, 210, 120, 5)
    tone      = st.slider("Tone",                 0.0, 1.0, 0.50, 0.05,
                          help="0 = dark, 1 = uplifting")
    pacing    = st.slider("Pacing",               0.0, 1.0, 0.75, 0.05,
                          help="0 = slow, 1 = fast-paced")
    dialogue  = st.slider("Dialogue-driven",      0.0, 1.0, 0.30, 0.05,
                          help="0 = visual/action, 1 = lots of dialogue")

    st.divider()
    top_k     = st.slider("Number of results", 1, 10, 5)
    mode      = st.selectbox("Ranking mode", MODES)
    diversity = st.checkbox("Avoid repeat directors", value=True)

    st.divider()

    # Rate limit display
    used, remaining, reset_str = rate_limit_status()
    st.caption("Usage this hour")
    st.progress(used / RATE_LIMIT, text=f"{used} / {RATE_LIMIT} searches used")
    if remaining > 0:
        st.caption(f"Resets in {reset_str}" if used > 0 else "")
    else:
        st.error(f"Limit reached — resets in {reset_str}")

    search = st.button("Find Films", type="primary", use_container_width=True,
                       disabled=(remaining == 0))

# ── Build profile ─────────────────────────────────────────────────────────────

profile = UserProfile(
    name=name,
    favorite_genre=genre,
    preferred_mood=mood,
    target_intensity=intensity,
    target_runtime=runtime,
    tone_preference=tone,
    pacing_preference=pacing,
    dialogue_preference=dialogue,
)

penalty = 0.20 if diversity else 0.0

# ── On search click: record request and cache results ─────────────────────────

if search:
    record_request()
    results = rec.recommend(profile, k=top_k, mode=mode, diversity_penalty=penalty)
    st.session_state["last_results"] = results
    st.session_state["last_profile"] = profile
    st.session_state["last_mode"]    = mode
    st.session_state["last_genre"]   = genre
    _, remaining, reset_str = rate_limit_status()

# ── Display results ───────────────────────────────────────────────────────────

if "last_results" not in st.session_state:
    st.info("Set your preferences in the sidebar and click **Find Films** to get recommendations.")
    st.stop()

results  = st.session_state["last_results"]
weights  = _DEFAULT_WEIGHTS[st.session_state["last_mode"]]
last_genre = st.session_state["last_genre"]

genre_hits = sum(1 for m, _ in results if m.genre.lower() == last_genre.lower())
top_conf   = confidence_score(results[0][1], weights) if results else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Best match confidence", f"{top_conf:.0%}")
c2.metric("Genre matches", f"{genre_hits} / {len(results)}")
c3.metric("Ranking mode", st.session_state["last_mode"])

if genre_hits == 0:
    st.warning(f"No '{last_genre}' films in the catalog — showing closest vibe matches.")

st.divider()

last_profile = st.session_state["last_profile"]

for rank, (movie, score) in enumerate(results, 1):
    conf = confidence_score(score, weights)
    conf_color = "green" if conf >= 0.80 else ("orange" if conf >= 0.55 else "red")

    with st.expander(
        f"#{rank}  **{movie.title}**  ·  {movie.director}  —  :{conf_color}[{conf:.0%} match]",
        expanded=(rank == 1),
    ):
        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.markdown(f"**Genre:** {movie.genre}")
            st.markdown(f"**Mood:** {movie.mood}")
            st.markdown(f"**Runtime:** {movie.runtime_min} min")
            st.markdown(f"**Decade:** {movie.release_decade}")
            st.markdown(f"**Language:** {movie.language}")
            if movie.awards:
                st.success("Award winner")

        with col_b:
            st.markdown("**Why this film?**")
            explanation = rec.explain_recommendation(last_profile, movie)
            for line in explanation.split("\n")[1:]:
                st.markdown(f"- {line.strip()}")

            st.markdown("**Feature breakdown**")
            fa, fb, fc, fd = st.columns(4)
            fa.metric("Intensity", f"{movie.intensity:.2f}",
                      delta=f"{movie.intensity - last_profile.target_intensity:+.2f}")
            fb.metric("Tone",      f"{movie.tone:.2f}",
                      delta=f"{movie.tone - last_profile.tone_preference:+.2f}")
            fc.metric("Pacing",    f"{movie.pacing:.2f}",
                      delta=f"{movie.pacing - last_profile.pacing_preference:+.2f}")
            fd.metric("Dialogue",  f"{movie.dialogue_heavy:.2f}",
                      delta=f"{movie.dialogue_heavy - last_profile.dialogue_preference:+.2f}")
