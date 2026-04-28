import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from recommender import (
    Movie, UserProfile, Recommender,
    score_movie, confidence_score, _DEFAULT_WEIGHTS,
)


def make_small_recommender():
    movies = [
        Movie(
            id=1, title="Explosive Action", director="Dir A",
            genre="action", mood="intense",
            intensity=0.95, runtime_min=120, tone=0.50,
            pacing=0.90, dialogue_heavy=0.20,
            popularity=0.90, release_decade="2010s",
            language="english", awards=False,
        ),
        Movie(
            id=2, title="Quiet Character Study", director="Dir B",
            genre="drama", mood="emotional",
            intensity=0.20, runtime_min=110, tone=0.70,
            pacing=0.20, dialogue_heavy=0.95,
            popularity=0.70, release_decade="2010s",
            language="english", awards=True,
        ),
    ]
    return Recommender(movies)


# ── Test 1 ────────────────────────────────────────────────────────────────────

def test_recommend_returns_movies_sorted_by_score():
    rec = make_small_recommender()
    user = UserProfile(
        name="Action Fan",
        favorite_genre="action", preferred_mood="intense",
        target_intensity=0.90, target_runtime=120,
        tone_preference=0.50, pacing_preference=0.90,
        dialogue_preference=0.20,
    )
    results = rec.recommend(user, k=2)
    assert len(results) == 2
    scores = [s for _, s in results]
    assert scores[0] >= scores[1], "Results must be sorted highest score first"
    assert results[0][0].title == "Explosive Action"


# ── Test 2 ────────────────────────────────────────────────────────────────────

def test_explain_recommendation_returns_non_empty_string():
    rec = make_small_recommender()
    user = UserProfile(
        name="Action Fan",
        favorite_genre="action", preferred_mood="intense",
        target_intensity=0.90, target_runtime=120,
        tone_preference=0.50, pacing_preference=0.90,
        dialogue_preference=0.20,
    )
    explanation = rec.explain_recommendation(user, rec.movies[0])
    assert isinstance(explanation, str)
    assert len(explanation) > 0


# ── Test 3 ────────────────────────────────────────────────────────────────────

def test_genre_match_adds_exactly_genre_weight():
    weights = _DEFAULT_WEIGHTS["balanced"]
    movie = Movie(
        id=1, title="X", director="D", genre="action", mood="fun",
        intensity=0.5, runtime_min=100, tone=0.5,
        pacing=0.5, dialogue_heavy=0.5,
        popularity=0.5, release_decade="2010s",
        language="english", awards=False,
    )
    user_match = UserProfile("U", "action", "fun", 0.5, 100, 0.5, 0.5, 0.5)
    user_miss  = UserProfile("U", "horror", "fun", 0.5, 100, 0.5, 0.5, 0.5)
    diff = score_movie(movie, user_match, weights) - score_movie(movie, user_miss, weights)
    assert abs(diff - weights["genre"]) < 0.0001, "Genre match should add exactly the genre weight"


# ── Test 4 ────────────────────────────────────────────────────────────────────

def test_director_penalty_reduces_score_for_repeat_director():
    movies = [
        Movie(id=1, title="Film A", director="Same", genre="action", mood="fun",
              intensity=0.8, runtime_min=100, tone=0.5, pacing=0.8, dialogue_heavy=0.2,
              popularity=0.8, release_decade="2010s", language="english", awards=False),
        Movie(id=2, title="Film B", director="Same", genre="action", mood="fun",
              intensity=0.8, runtime_min=100, tone=0.5, pacing=0.8, dialogue_heavy=0.2,
              popularity=0.8, release_decade="2010s", language="english", awards=False),
    ]
    rec = Recommender(movies)
    user = UserProfile("U", "action", "fun", 0.8, 100, 0.5, 0.8, 0.2)
    results = rec.recommend(user, k=2, diversity_penalty=0.20)
    assert results[1][1] < results[0][1], "Second film from same director must score lower"


# ── Test 5 ────────────────────────────────────────────────────────────────────

def test_confidence_score_is_between_zero_and_one():
    weights = _DEFAULT_WEIGHTS["balanced"]
    movie = Movie(
        id=1, title="X", director="D", genre="action", mood="intense",
        intensity=0.9, runtime_min=120, tone=0.5,
        pacing=0.9, dialogue_heavy=0.2,
        popularity=0.9, release_decade="2010s",
        language="english", awards=False,
    )
    user = UserProfile("U", "action", "intense", 0.9, 120, 0.5, 0.9, 0.2)
    raw = score_movie(movie, user, weights)
    conf = confidence_score(raw, weights)
    assert 0.0 <= conf <= 1.0, f"Confidence must be 0–1, got {conf}"


# ── Test 6 ────────────────────────────────────────────────────────────────────

def test_invalid_mode_raises_value_error():
    rec = make_small_recommender()
    user = UserProfile("U", "action", "intense", 0.9, 120, 0.5, 0.9, 0.2)
    with pytest.raises(ValueError, match="Unknown mode"):
        rec.recommend(user, mode="not-a-real-mode")
