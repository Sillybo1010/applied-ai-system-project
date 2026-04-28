import csv
import logging
import os
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)


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
    release_decade: str
    language: str
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


def _max_possible_score(weights: Dict[str, float]) -> float:
    """Theoretical ceiling: every component scores its full weight."""
    return sum(weights.values())


# ── Core scoring functions ────────────────────────────────────────────────────

def score_movie(movie: Movie, profile: UserProfile, weights: Dict[str, float]) -> float:
    w = weights

    genre_score = w["genre"] if movie.genre.lower() == profile.favorite_genre.lower() else 0.0
    mood_score  = w["mood"]  if movie.mood.lower()  == profile.preferred_mood.lower()  else 0.0

    intensity_sim = 1.0 - abs(movie.intensity      - profile.target_intensity)
    tone_sim      = 1.0 - abs(movie.tone           - profile.tone_preference)
    pacing_sim    = 1.0 - abs(movie.pacing         - profile.pacing_preference)
    dialogue_sim  = 1.0 - abs(movie.dialogue_heavy - profile.dialogue_preference)

    # ±60-minute runtime tolerance window
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


def confidence_score(raw_score: float, weights: Dict[str, float]) -> float:
    """Normalize raw score to 0.0–1.0 against the theoretical maximum."""
    max_score = _max_possible_score(weights)
    if max_score == 0:
        return 0.0
    return round(raw_score / max_score, 4)


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
    if mode not in _DEFAULT_WEIGHTS:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(_DEFAULT_WEIGHTS)}")

    weights = _DEFAULT_WEIGHTS[mode]
    scored = [(m, score_movie(m, profile, weights)) for m in movies]
    scored.sort(key=lambda x: x[1], reverse=True)
    scored = apply_director_penalty(scored, diversity_penalty)
    scored.sort(key=lambda x: x[1], reverse=True)

    top = scored[:k]

    # Log warnings for low-confidence or zero-genre-match situations
    genre_matches = [m for m, _ in top if m.genre.lower() == profile.favorite_genre.lower()]
    if not genre_matches:
        logger.warning(
            "No genre match for '%s' (wanted '%s') — falling back to vibe similarity.",
            profile.name, profile.favorite_genre
        )

    top_conf = confidence_score(top[0][1], weights) if top else 0.0
    if top_conf < 0.50:
        logger.warning(
            "Low confidence for '%s': best match is only %.0f%% — catalog may not suit this profile.",
            profile.name, top_conf * 100
        )
    else:
        logger.info(
            "Recommendations for '%s' (mode=%s): top result '%s' confidence=%.2f",
            profile.name, mode, top[0][0].title if top else "none", top_conf
        )

    return top


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

    def confidence_for(self, raw_score: float, mode: str = "balanced") -> float:
        return confidence_score(raw_score, _DEFAULT_WEIGHTS[mode])

    def explain_recommendation(self, profile: UserProfile, movie: Movie) -> str:
        lines = [f"Why '{movie.title}'?"]

        if movie.genre.lower() == profile.favorite_genre.lower():
            lines.append(f"  Genre match: {movie.genre}")
        if movie.mood.lower() == profile.preferred_mood.lower():
            lines.append(f"  Mood match: {movie.mood}")

        if 1.0 - abs(movie.intensity - profile.target_intensity) >= 0.80:
            lines.append(f"  Intensity close ({movie.intensity:.2f} vs target {profile.target_intensity:.2f})")
        if 1.0 - abs(movie.tone - profile.tone_preference) >= 0.80:
            lines.append(f"  Tone close ({movie.tone:.2f} vs pref {profile.tone_preference:.2f})")
        runtime_sim = max(0.0, 1.0 - abs(movie.runtime_min - profile.target_runtime) / 60.0)
        if runtime_sim >= 0.80:
            lines.append(f"  Runtime close ({movie.runtime_min} min vs target {profile.target_runtime} min)")

        if len(lines) == 1:
            lines.append("  Matched on overall weighted similarity")

        return "\n".join(lines)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_movies(filepath: str) -> List[Movie]:
    if not os.path.exists(filepath):
        logger.error("Data file not found: %s", filepath)
        raise FileNotFoundError(f"Movie data not found at: {filepath}")

    movies = []
    try:
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
    except KeyError as e:
        logger.error("Missing expected column in CSV: %s", e)
        raise

    logger.info("Loaded %d movies from %s", len(movies), filepath)
    return movies
