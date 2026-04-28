"""
CineMatch — Interactive Streamlit Web App
Run: streamlit run src/app.py
"""

import contextlib
import io
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from recommender import (
    UserProfile, Recommender, load_movies,
    confidence_score, _DEFAULT_WEIGHTS,
)
from rag_retriever import RAGRetriever
from specializer import specialize, measure_modes, ExplainerMode
from agent import CineAgent

# ── Paths ─────────────────────────────────────────────────────────────────────

_BASE = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH     = os.path.join(_BASE, "data", "movies.csv")
PROFILES_PATH = os.path.join(_BASE, "data", "director_profiles.json")

# ── Catalog constants ──────────────────────────────────────────────────────────

GENRES = ["action", "sci-fi", "drama", "animation", "horror",
          "comedy", "thriller", "romance", "war", "western"]
MOODS  = ["thrilling", "emotional", "heartwarming", "epic", "scary",
          "quirky", "dark", "romantic", "intense", "fun", "disturbing"]
MODES  = ["balanced", "genre-first", "mood-first", "vibe-first"]


# ── Cached data loading ────────────────────────────────────────────────────────

@st.cache_resource
def load_data():
    movies = load_movies(DATA_PATH)
    rag    = RAGRetriever(PROFILES_PATH)
    rec    = Recommender(movies)
    agent  = CineAgent(DATA_PATH, PROFILES_PATH)
    return movies, rag, rec, agent


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎬 CineMatch — Movie Recommender")
st.caption("Content-based recommendation with RAG augmentation, agentic reasoning, and multi-tone explanations.")

movies, rag, rec, agent = load_data()

# ── Sidebar — Profile Builder ─────────────────────────────────────────────────

with st.sidebar:
    st.header("Your Taste Profile")

    name = st.text_input("Your name", value="Movie Fan")

    col1, col2 = st.columns(2)
    with col1:
        genre = st.selectbox("Favorite genre", GENRES, index=0)
    with col2:
        mood = st.selectbox("Preferred mood", MOODS, index=8)

    intensity  = st.slider("Intensity",          0.0, 1.0, 0.80, 0.05,
                           help="0 = calm / contemplative, 1 = high-octane")
    runtime    = st.slider("Target runtime (min)", 60, 210, 120, 5)
    tone       = st.slider("Tone preference",     0.0, 1.0, 0.50, 0.05,
                           help="0 = dark / bleak, 1 = uplifting / hopeful")
    pacing     = st.slider("Pacing preference",   0.0, 1.0, 0.75, 0.05,
                           help="0 = slow / contemplative, 1 = fast-paced")
    dialogue   = st.slider("Dialogue preference", 0.0, 1.0, 0.30, 0.05,
                           help="0 = visual / action-driven, 1 = dialogue-heavy")

    st.divider()
    mode      = st.selectbox("Ranking mode", MODES)
    top_k     = st.slider("Results to show", 1, 10, 5)
    diversity = st.checkbox("Director diversity penalty (−20%)", value=True)

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

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_rec, tab_agent, tab_rag, tab_spec, tab_eval = st.tabs([
    "Recommendations",
    "Agent Trace",
    "RAG Demo",
    "Specializer",
    "Eval Harness",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Recommendations
# ─────────────────────────────────────────────────────────────────────────────

with tab_rec:
    st.subheader(f"Top {top_k} picks for {name or 'you'}")

    results = rec.recommend(profile, k=top_k, mode=mode, diversity_penalty=penalty)
    weights = _DEFAULT_WEIGHTS[mode]

    genre_hits = sum(1 for m, _ in results if m.genre.lower() == genre.lower())
    top_conf   = confidence_score(results[0][1], weights) if results else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Top result confidence", f"{top_conf:.0%}")
    m2.metric("Genre matches in results", f"{genre_hits} / {len(results)}")
    m3.metric("Ranking mode", mode)

    if genre_hits == 0:
        st.warning(f"No '{genre}' films in the catalog — showing best vibe matches instead.")

    st.divider()

    for rank, (movie, score) in enumerate(results, 1):
        conf = confidence_score(score, weights)
        conf_color = "green" if conf >= 0.80 else ("orange" if conf >= 0.55 else "red")

        with st.expander(
            f"#{rank}  **{movie.title}** ({movie.director})  —  "
            f":{conf_color}[{conf:.0%} confidence]",
            expanded=(rank == 1),
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Raw score", f"{score:.4f}")
            c2.metric("Confidence", f"{conf:.0%}")
            c3.metric("Genre / Mood", f"{movie.genre} / {movie.mood}")

            c4, c5, c6, c7 = st.columns(4)
            c4.metric("Intensity",  f"{movie.intensity:.2f}")
            c5.metric("Tone",       f"{movie.tone:.2f}")
            c6.metric("Pacing",     f"{movie.pacing:.2f}")
            c7.metric("Runtime",    f"{movie.runtime_min} min")

            st.markdown("**Why this film?**")
            explanation = rec.explain_recommendation(profile, movie)
            st.code(explanation, language=None)

            if movie.awards:
                st.success("Award-winning film")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Agent Trace
# ─────────────────────────────────────────────────────────────────────────────

with tab_agent:
    st.subheader("5-Step Agentic Reasoning Chain")
    st.caption("CineAgent.run() makes every intermediate decision observable.")

    if st.button("Run Agent", type="primary"):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            agent_results = agent.run(profile, k=top_k, verbose=True)
        trace = buf.getvalue()

        st.code(trace, language=None)

        st.divider()
        st.markdown("**Agent results with RAG context**")
        for rank, (movie, score, rag_ctx) in enumerate(agent_results, 1):
            conf = confidence_score(score, _DEFAULT_WEIGHTS["balanced"])
            with st.expander(f"#{rank}  {movie.title}  ({conf:.0%} conf)", expanded=(rank == 1)):
                st.write(f"**Director:** {movie.director}  |  **Genre/Mood:** {movie.genre} / {movie.mood}")
                if rag_ctx:
                    st.info(f"[RAG] {rag_ctx}")
                else:
                    st.caption("No RAG profile available for this director.")
    else:
        st.info("Click **Run Agent** to see the 5-step reasoning chain for your current profile.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — RAG Demo
# ─────────────────────────────────────────────────────────────────────────────

with tab_rag:
    st.subheader("RAG Enhancement — Before vs After")
    st.caption(
        "The base recommender explains matches using feature similarity only. "
        "RAG augments each explanation with director style context from a second data source."
    )

    demo_results = rec.recommend(profile, k=1, mode=mode)
    if demo_results:
        demo_movie, demo_score = demo_results[0]
        base_exp = rec.explain_recommendation(profile, demo_movie)
        aug_exp  = rag.augment_explanation(base_exp, demo_movie, profile)
        uplift   = RAGRetriever.measure_improvement(base_exp, aug_exp)

        st.markdown(f"**Showing for:** {demo_movie.title} (director: {demo_movie.director})")
        st.divider()

        left, right = st.columns(2)
        with left:
            st.markdown("#### Base explanation")
            st.caption("No RAG context")
            st.code(base_exp, language=None)
            st.metric("Word count", uplift["base_words"])

        with right:
            st.markdown("#### RAG-augmented explanation")
            st.caption("Director profile retrieved from director_profiles.json")
            st.code(aug_exp, language=None)
            st.metric("Word count", uplift["aug_words"],
                      delta=f"+{uplift['added_words']} words (+{uplift['pct_increase']}%)")

        st.divider()
        st.markdown("**All directors with RAG profiles in this result set**")
        rag_results = rec.recommend(profile, k=top_k, mode=mode)
        rows = []
        for movie, score in rag_results:
            ctx = rag.retrieve_director_context(movie.director)
            rows.append({
                "Film":    movie.title,
                "Director": movie.director,
                "RAG context": ctx if ctx else "—",
            })
        st.table(rows)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Specializer
# ─────────────────────────────────────────────────────────────────────────────

with tab_spec:
    st.subheader("Few-Shot Explanation Specializer")
    st.caption(
        "The same recommendation is explained in three constrained tones. "
        "`measure_modes()` quantifies the difference in word count and lexical diversity."
    )

    spec_results = rec.recommend(profile, k=1, mode=mode)
    if spec_results:
        spec_movie, _ = spec_results[0]
        measurements  = measure_modes(spec_movie, profile)

        st.markdown(f"**Film:** {spec_movie.title} &nbsp;|&nbsp; **Director:** {spec_movie.director}")
        st.divider()

        c_std, c_cas, c_cin = st.columns(3)

        with c_std:
            st.markdown("#### Standard")
            st.caption("Feature-by-feature match report")
            d = measurements["standard"]
            st.code(d["text"], language=None)
            st.metric("Words",            d["word_count"])
            st.metric("Lexical diversity", d["lexical_diversity"])

        with c_cas:
            st.markdown("#### Casual")
            st.caption("Short, friendly, plain language")
            d = measurements["casual"]
            st.code(d["text"], language=None)
            st.metric("Words",            d["word_count"])
            st.metric("Lexical diversity", d["lexical_diversity"])

        with c_cin:
            st.markdown("#### Cinephile")
            st.caption("Film-theory vocabulary, directorial analysis")
            d = measurements["cinephile"]
            st.code(d["text"], language=None)
            st.metric("Words",            d["word_count"])
            st.metric("Lexical diversity", d["lexical_diversity"])

        st.divider()
        st.markdown("**Try any film directly**")
        all_titles = [m.title for m in movies]
        chosen = st.selectbox("Select a film", all_titles)
        chosen_movie = next(m for m in movies if m.title == chosen)
        chosen_mode  = st.radio("Explanation mode", ["standard", "casual", "cinephile"],
                                horizontal=True)
        mode_map = {
            "standard":  ExplainerMode.STANDARD,
            "casual":    ExplainerMode.CASUAL,
            "cinephile": ExplainerMode.CINEPHILE,
        }
        st.code(specialize(chosen_movie, profile, mode_map[chosen_mode]), language=None)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Eval Harness
# ─────────────────────────────────────────────────────────────────────────────

with tab_eval:
    st.subheader("Evaluation Harness")
    st.caption("8 predefined test cases covering core recommender, RAG, and specializer behaviour.")

    if st.button("Run All Tests", type="primary"):

        def make_profile(**kw):
            base = dict(
                name="Test", favorite_genre="action", preferred_mood="intense",
                target_intensity=0.80, target_runtime=120,
                tone_preference=0.50, pacing_preference=0.80, dialogue_preference=0.20,
            )
            base.update(kw)
            return UserProfile(**base)

        weights_bal = _DEFAULT_WEIGHTS["balanced"]

        tests = [
            {
                "name": "Action fan: genre match in top result",
                "run":  lambda: rec.recommend(make_profile(favorite_genre="action"), k=5),
                "check": lambda r: r[0][0].genre.lower() == "action",
            },
            {
                "name": "Drama seeker: genre match in top result",
                "run":  lambda: rec.recommend(
                    make_profile(favorite_genre="drama", preferred_mood="emotional",
                                 target_intensity=0.25, pacing_preference=0.25,
                                 dialogue_preference=0.95, target_runtime=140), k=5),
                "check": lambda r: r[0][0].genre.lower() == "drama",
            },
            {
                "name": "Animation fan: animation in top 3",
                "run":  lambda: rec.recommend(
                    make_profile(favorite_genre="animation", preferred_mood="heartwarming",
                                 target_intensity=0.45, tone_preference=0.90,
                                 target_runtime=95), k=5),
                "check": lambda r: any(m.genre.lower() == "animation" for m, _ in r[:3]),
            },
            {
                "name": "Results sorted highest score first",
                "run":  lambda: rec.recommend(make_profile(), k=5),
                "check": lambda r: all(r[i][1] >= r[i+1][1] for i in range(len(r)-1)),
            },
            {
                "name": "Confidence always 0.0–1.0",
                "run":  lambda: rec.recommend(make_profile(), k=5),
                "check": lambda r: all(0.0 <= confidence_score(s, weights_bal) <= 1.0 for _, s in r),
            },
            {
                "name": "Ghost genre confidence < real genre match",
                "run":  lambda: (
                    rec.recommend(make_profile(favorite_genre="western"), k=5),
                    rec.recommend(make_profile(favorite_genre="action"),  k=5),
                ),
                "check": lambda pair: (
                    confidence_score(pair[0][0][1], weights_bal)
                    < confidence_score(pair[1][0][1], weights_bal)
                ),
            },
            {
                "name": "RAG: augmented explanation longer than base",
                "run":  lambda: rec.recommend(make_profile(), k=1)[0],
                "check": lambda t: (
                    RAGRetriever.measure_improvement(
                        rec.explain_recommendation(make_profile(), t[0]),
                        rag.augment_explanation(
                            rec.explain_recommendation(make_profile(), t[0]),
                            t[0], make_profile()
                        )
                    )["added_words"] > 0
                ),
            },
            {
                "name": "Specializer: cinephile longer than casual",
                "run":  lambda: rec.recommend(make_profile(), k=1)[0][0],
                "check": lambda m: (
                    measure_modes(m, make_profile())["cinephile"]["word_count"]
                    > measure_modes(m, make_profile())["casual"]["word_count"]
                ),
            },
        ]

        passed = failed = 0
        conf_scores = []
        rows = []

        for tc in tests:
            try:
                result = tc["run"]()
                ok = tc["check"](result)
                if isinstance(result, list) and result and isinstance(result[0], tuple):
                    conf_scores.append(confidence_score(result[0][1], weights_bal))
            except Exception as exc:
                ok = False
                rows.append({"Result": "ERROR", "Test": tc["name"], "Note": str(exc)})
                failed += 1
                continue

            status = "PASS" if ok else "FAIL"
            passed += ok
            failed += (not ok)
            rows.append({"Result": status, "Test": tc["name"], "Note": ""})

        avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0

        # Summary metrics
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Passed", f"{passed} / {len(tests)}")
        mc2.metric("Failed", str(failed))
        mc3.metric("Avg top confidence", f"{avg_conf:.0%}")

        # Results table with colour
        for row in rows:
            icon = "✅" if row["Result"] == "PASS" else ("❌" if row["Result"] == "FAIL" else "⚠️")
            if row["Result"] == "PASS":
                st.success(f"{icon}  {row['Test']}")
            elif row["Result"] == "FAIL":
                st.error(f"{icon}  {row['Test']}")
            else:
                st.warning(f"{icon}  {row['Test']}  —  {row['Note']}")

        if failed == 0:
            st.balloons()
    else:
        st.info("Click **Run All Tests** to execute the 8-case evaluation harness.")
