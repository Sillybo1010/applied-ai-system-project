from dataclasses import dataclass
from typing import List, Dict
import csv
import os


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Movie:
    id: int
    title: str
    director: str
    genre: str
    mood: str
    intensity: float       # 0.0 (calm) → 1.0 (high-octane)
    runtime_min: int       # length in minutes
    tone: float            # 0.0 (dark/bleak) → 1.0 (uplifting/hopeful)
    pacing: float          # 0.0 (slow/contemplative) → 1.0 (fast-paced)
    dialogue_heavy: float  # 0.0 (visual/action-driven) → 1.0 (dialogue/character-driven)
    popularity: float      # 0.0 → 1.0
    release_decade: str    # e.g. "2010s"
    language: str          # "english" or "foreign"
    awards: bool


@dataclass
class UserProfile:
    name: str
    favorite_genre: str
    preferred_mood: str
    target_intensity: float
    target_runtime: int
    tone_preference: float
    pacing_preference: float
    dialogue_preference: float


# ── Default scoring weights per ranking mode ──────────────────────────────────

_DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "balanced": {
        "genre":    3.00,
        "mood":     2.00,
        "intensity": 1.00,
        "tone":     0.75,
        "pacing":   0.50,
        "dialogue": 0.50,
        "runtime":  0.50,
        "popularity": 0.25,
    },
    "genre-first": {
        "genre":    5.00,
        "mood":     1.50,
        "intensity": 0.75,
        "tone":     0.50,
        "pacing":   0.25,
        "dialogue": 0.25,
        "runtime":  0.25,
        "popularity": 0.10,
    },
    "mood-first": {
        "genre":    1.50,
        "mood":     4.00,
        "intensity": 0.75,
        "tone":     0.50,
        "pacing":   0.25,
        "dialogue": 0.25,
        "runtime":  0.25,
        "popularity": 0.10,
    },
    "vibe-first": {
        "genre":    1.50,
        "mood":     1.50,
        "intensity": 2.00,
        "tone":     1.50,
        "pacing":   1.00,
        "dialogue": 0.50,
        "runtime":  0.25,
        "popularity": 0.10,
    },
}


# ── Core scoring functions ────────────────────────────────────────────────────

def score_movie(movie: Movie, profile: UserProfile, weights: Dict[str, float]) -> float:
    w = weights

    # Categorical matching: genre and mood
    genre_score = w["genre"] if movie.genre.lower() == profile.favorite_genre.lower() else 0.0
    mood_score  = w["mood"]  if movie.mood.lower()  == profile.preferred_mood.lower()  else 0.0

    # Continuous similarity on 0.0–1.0 scale (1.0 = perfect match)
    intensity_sim = 1.0 - abs(movie.intensity      - profile.target_intensity)
    tone_sim      = 1.0 - abs(movie.tone           - profile.tone_preference)
    pacing_sim    = 1.0 - abs(movie.pacing         - profile.pacing_preference)
    dialogue_sim  = 1.0 - abs(movie.dialogue_heavy - profile.dialogue_preference)

    # Runtime similarity: ±60-minute tolerance window (mirrors ±100 BPM in music)
    runtime_sim = max(0.0, 1.0 - abs(movie.runtime_min - profile.target_runtime) / 60.0)

    total = (
        genre_score
        + mood_score
        + w["intensity"] * intensity_sim
        + w["tone"]      * tone_sim
        + w["pacing"]    * pacing_sim
        + w["dialogue"]  * dialogue_sim
        + w["runtime"]   * runtime_sim
        + w["popularity"] * movie.popularity
    )
    return round(total, 4)


def apply_director_penalty(results: List[tuple], penalty: float = 0.20) -> List[tuple]:
    """Discount repeat directors to prevent filter bubbles."""
    seen: Dict[str, int] = {}
    adjusted = []
    for movie, score in results:
        count = seen.get(movie.director, 0)
        adj = score * ((1 - penalty) ** count)
        seen[movie.director] = count + 1
        adjusted.append((movie, round(adj, 4)))
    return adjusted


def recommend_movies(
    movies: List[Movie],
    profile: UserProfile,
    k: int = 5,
    mode: str = "balanced",
    diversity_penalty: float = 0.20,
) -> List[tuple]:
    weights = _DEFAULT_WEIGHTS[mode]
    scored = [(m, score_movie(m, profile, weights)) for m in movies]
    scored.sort(key=lambda x: x[1], reverse=True)
    scored = apply_director_penalty(scored, diversity_penalty)
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# ── OOP wrapper ───────────────────────────────────────────────────────────────

class Recommender:
    def __init__(self, movies: List[Movie]):
        self.movies = movies

    def recommend(
        self,
        profile: UserProfile,
        k: int = 5,
        mode: str = "balanced",
        diversity_penalty: float = 0.20,
    ) -> List[tuple]:
        return recommend_movies(self.movies, profile, k, mode, diversity_penalty)

    def explain_recommendation(self, profile: UserProfile, movie: Movie) -> str:
        lines = [f"Why '{movie.title}'?"]

        if movie.genre.lower() == profile.favorite_genre.lower():
            lines.append(f"  Genre match: {movie.genre}")
        if movie.mood.lower() == profile.preferred_mood.lower():
            lines.append(f"  Mood match: {movie.mood}")

        intensity_sim = 1.0 - abs(movie.intensity - profile.target_intensity)
        if intensity_sim >= 0.80:
            lines.append(f"  Intensity close ({movie.intensity:.2f} vs target {profile.target_intensity:.2f})")

        tone_sim = 1.0 - abs(movie.tone - profile.tone_preference)
        if tone_sim >= 0.80:
            lines.append(f"  Tone close ({movie.tone:.2f} vs pref {profile.tone_preference:.2f})")

        runtime_sim = max(0.0, 1.0 - abs(movie.runtime_min - profile.target_runtime) / 60.0)
        if runtime_sim >= 0.80:
            lines.append(f"  Runtime close ({movie.runtime_min} min vs target {profile.target_runtime} min)")

        if len(lines) == 1:
            lines.append("  Matched on overall weighted similarity")

        return "\n".join(lines)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_movies(filepath: str) -> List[Movie]:
    movies = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies.append(Movie(
                id=int(row["id"]),
                title=row["title"],
                director=row["director"],
                genre=row["genre"],
                mood=row["mood"],
                intensity=float(row["intensity"]),
                runtime_min=int(row["runtime_min"]),
                tone=float(row["tone"]),
                pacing=float(row["pacing"]),
                dialogue_heavy=float(row["dialogue_heavy"]),
                popularity=float(row["popularity"]),
                release_decade=row["release_decade"],
                language=row["language"],
                awards=row["awards"].strip().lower() == "true",
            ))
    return movies
