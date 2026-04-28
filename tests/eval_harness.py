"""
Test Harness / Evaluation Script
=================================
Runs CineMatch against 8 predefined inputs and prints a structured
PASS/FAIL report with confidence ratings and a one-line summary.

Usage:
    python tests/eval_harness.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recommender import (
    UserProfile, Recommender, load_movies,
    confidence_score, _DEFAULT_WEIGHTS,
)
from rag_retriever import RAGRetriever
from specializer import specialize, measure_modes, ExplainerMode

DATA_PATH     = os.path.join(os.path.dirname(__file__), "..", "data", "movies.csv")
PROFILES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "director_profiles.json")

WEIGHTS = _DEFAULT_WEIGHTS["balanced"]


# ── Helper ────────────────────────────────────────────────────────────────────

def make_profile(**kw) -> UserProfile:
    base = dict(
        name="Test", favorite_genre="action", preferred_mood="intense",
        target_intensity=0.80, target_runtime=120,
        tone_preference=0.50, pacing_preference=0.80, dialogue_preference=0.20,
    )
    base.update(kw)
    return UserProfile(**base)


# ── Test cases ────────────────────────────────────────────────────────────────

def run_harness() -> bool:
    movies = load_movies(DATA_PATH)
    rec    = Recommender(movies)
    rag    = RAGRetriever(PROFILES_PATH)

    tests = [
        # ── Core recommender tests ────────────────────────────────────────────
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
            "name": "Animation fan: animation appears in top 3",
            "run":  lambda: rec.recommend(
                make_profile(favorite_genre="animation", preferred_mood="heartwarming",
                             target_intensity=0.45, tone_preference=0.90,
                             target_runtime=95), k=5),
            "check": lambda r: any(m.genre.lower() == "animation" for m, _ in r[:3]),
        },
        {
            "name": "Results are sorted highest score first",
            "run":  lambda: rec.recommend(make_profile(), k=5),
            "check": lambda r: all(r[i][1] >= r[i + 1][1] for i in range(len(r) - 1)),
        },
        {
            "name": "Confidence score is always 0.0 – 1.0",
            "run":  lambda: rec.recommend(make_profile(), k=5),
            "check": lambda r: all(0.0 <= confidence_score(s, WEIGHTS) <= 1.0
                                   for _, s in r),
        },
        {
            "name": "Ghost genre scores lower confidence than real genre match",
            "run":  lambda: (
                rec.recommend(make_profile(favorite_genre="western"), k=5),
                rec.recommend(make_profile(favorite_genre="action"),  k=5),
            ),
            "check": lambda pair: (
                confidence_score(pair[0][0][1], WEIGHTS)
                < confidence_score(pair[1][0][1], WEIGHTS)
            ),
        },
        # ── RAG retrieval test ────────────────────────────────────────────────
        {
            "name": "RAG: augmented explanation is longer than base explanation",
            "run":  lambda: (
                rec.recommend(make_profile(), k=1)[0],
            ),
            "check": lambda pair: _check_rag_uplift(pair[0], rag,
                                                    make_profile()),
        },
        # ── Specialiser test ──────────────────────────────────────────────────
        {
            "name": "Specializer: cinephile output longer than casual output",
            "run":  lambda: rec.recommend(make_profile(), k=1)[0][0],
            "check": lambda movie: (
                measure_modes(movie, make_profile())["cinephile"]["word_count"]
                > measure_modes(movie, make_profile())["casual"]["word_count"]
            ),
        },
    ]

    passed = 0
    failed = 0
    conf_scores = []

    print("=" * 60)
    print("CINEMATIC EVAL HARNESS")
    print("=" * 60)

    for tc in tests:
        try:
            result = tc["run"]()
            ok = tc["check"](result)
            # Collect confidence where applicable
            if isinstance(result, list) and result and isinstance(result[0], tuple):
                conf_scores.append(confidence_score(result[0][1], WEIGHTS))
        except Exception as exc:
            ok = False
            print(f"  [ERROR] {tc['name']}: {exc}")

        status = "PASS" if ok else "FAIL"
        passed += ok
        failed += (not ok)
        print(f"  [{status}]  {tc['name']}")

    avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0

    print("=" * 60)
    print(f"Result  : {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"Avg top-result confidence : {avg_conf:.0%}")
    if failed == 0:
        print("All tests passed.")
    else:
        print(f"{failed} test(s) failed — review output above.")
    print("=" * 60)
    return failed == 0


def _check_rag_uplift(result_tuple, rag: RAGRetriever, profile: UserProfile) -> bool:
    movie, _ = result_tuple
    from recommender import Recommender
    base = Recommender([movie]).explain_recommendation(profile, movie)
    augmented = rag.augment_explanation(base, movie, profile)
    m = RAGRetriever.measure_improvement(base, augmented)
    return m["added_words"] > 0


if __name__ == "__main__":
    success = run_harness()
    sys.exit(0 if success else 1)
