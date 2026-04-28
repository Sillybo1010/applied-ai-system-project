import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from recommender import Movie, UserProfile, Recommender


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


def test_recommend_returns_movies_sorted_by_score():
    rec = make_small_recommender()
    user = UserProfile(
        name="Test Action Fan",
        favorite_genre="action",
        preferred_mood="intense",
        target_intensity=0.90,
        target_runtime=120,
        tone_preference=0.50,
        pacing_preference=0.90,
        dialogue_preference=0.20,
    )
    results = rec.recommend(user, k=2)
    assert len(results) == 2
    scores = [score for _, score in results]
    assert scores[0] >= scores[1]
    assert results[0][0].title == "Explosive Action"


def test_explain_recommendation_returns_non_empty_string():
    rec = make_small_recommender()
    user = UserProfile(
        name="Test Action Fan",
        favorite_genre="action",
        preferred_mood="intense",
        target_intensity=0.90,
        target_runtime=120,
        tone_preference=0.50,
        pacing_preference=0.90,
        dialogue_preference=0.20,
    )
    explanation = rec.explain_recommendation(user, rec.movies[0])
    assert isinstance(explanation, str)
    assert len(explanation) > 0
