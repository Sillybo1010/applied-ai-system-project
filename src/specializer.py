"""
Fine-Tuning / Specialization — three few-shot explanation modes.

Modes
------
STANDARD   Baseline: feature-by-feature match report (existing behaviour).
CASUAL     Conversational tone: short, friendly, uses plain language.
CINEPHILE  Technical tone: film-theory vocabulary, references directorial style.

Measurability
-------------
Call measure_modes(movie, profile) to get word counts and lexical diversity
for all three modes side-by-side, proving output differs quantifiably.
"""

from enum import Enum
from typing import Dict

from recommender import Movie, UserProfile


class ExplainerMode(Enum):
    STANDARD  = "standard"
    CASUAL    = "casual"
    CINEPHILE = "cinephile"


# ── Few-shot intensity labels per mode ────────────────────────────────────────

_INTENSITY_LABELS: Dict[str, Dict[str, str]] = {
    "casual": {
        "low":    "pretty chill",
        "medium": "moderately exciting",
        "high":   "super intense",
    },
    "cinephile": {
        "low":    "contemplative and restrained",
        "medium": "dynamically calibrated",
        "high":   "viscerally high-octane",
    },
}

_TONE_LABELS: Dict[str, Dict[str, str]] = {
    "casual": {
        "dark":    "pretty dark",
        "neutral": "balanced",
        "uplifting": "feel-good",
    },
    "cinephile": {
        "dark":    "tonally bleak",
        "neutral": "affectively neutral",
        "uplifting": "emotionally cathartic",
    },
}


def _intensity_tier(value: float) -> str:
    if value < 0.35:
        return "low"
    if value < 0.70:
        return "medium"
    return "high"


def _tone_tier(value: float) -> str:
    if value < 0.35:
        return "dark"
    if value < 0.65:
        return "neutral"
    return "uplifting"


def _label(table: Dict[str, Dict[str, str]], mode_key: str, tier: str) -> str:
    return table.get(mode_key, {}).get(tier, tier)


# ── Core specialisation function ──────────────────────────────────────────────

def specialize(movie: Movie, profile: UserProfile, mode: ExplainerMode) -> str:
    """
    Generate a recommendation explanation in the requested style.
    STANDARD delegates to the standard feature-match report format;
    CASUAL and CINEPHILE use few-shot templates to produce measurably
    different output in tone, length, and vocabulary.
    """
    if mode == ExplainerMode.STANDARD:
        return _standard(movie, profile)
    if mode == ExplainerMode.CASUAL:
        return _casual(movie, profile)
    if mode == ExplainerMode.CINEPHILE:
        return _cinephile(movie, profile)
    raise ValueError(f"Unknown mode: {mode}")


# ── Mode implementations ──────────────────────────────────────────────────────

def _standard(movie: Movie, profile: UserProfile) -> str:
    lines = [f"'{movie.title}'"]
    if movie.genre.lower() == profile.favorite_genre.lower():
        lines.append(f"Genre match: {movie.genre}")
    if movie.mood.lower() == profile.preferred_mood.lower():
        lines.append(f"Mood match: {movie.mood}")
    if 1.0 - abs(movie.intensity - profile.target_intensity) >= 0.80:
        lines.append(f"Intensity close ({movie.intensity:.2f} vs {profile.target_intensity:.2f})")
    if len(lines) == 1:
        lines.append("Matched on weighted similarity")
    return " | ".join(lines)


def _casual(movie: Movie, profile: UserProfile) -> str:
    # Few-shot pattern: short, direct, encouraging
    genre_hit = movie.genre.lower() == profile.favorite_genre.lower()
    mood_hit  = movie.mood.lower()  == profile.preferred_mood.lower()
    itier = _intensity_tier(movie.intensity)
    ttier = _tone_tier(movie.tone)
    i_lbl = _label(_INTENSITY_LABELS, "casual", itier)
    t_lbl = _label(_TONE_LABELS,      "casual", ttier)

    if genre_hit and mood_hit:
        lead = f"This one's perfect for you — it's {movie.genre}, {movie.mood}, and {i_lbl}."
    elif genre_hit:
        lead = f"Good {movie.genre} pick! It's {i_lbl} and {t_lbl}, close to what you're after."
    else:
        lead = f"Not your usual genre, but it's {i_lbl} and {t_lbl} — might surprise you."

    extras = []
    if abs(movie.runtime_min - profile.target_runtime) <= 20:
        extras.append(f"Runtime is spot-on ({movie.runtime_min} min).")
    if movie.awards:
        extras.append("Awards winner — people love it.")

    return " ".join([lead] + extras)


def _cinephile(movie: Movie, profile: UserProfile) -> str:
    # Few-shot pattern: technical, uses film-theory vocabulary
    genre_hit = movie.genre.lower() == profile.favorite_genre.lower()
    mood_hit  = movie.mood.lower()  == profile.preferred_mood.lower()
    itier = _intensity_tier(movie.intensity)
    ttier = _tone_tier(movie.tone)
    i_lbl = _label(_INTENSITY_LABELS, "cinephile", itier)
    t_lbl = _label(_TONE_LABELS,      "cinephile", ttier)

    if genre_hit and mood_hit:
        lead = (f"'{movie.title}' represents a canonical instance of {movie.genre} filmmaking "
                f"with a {movie.mood} register — a precise vector match to your aesthetic profile.")
    elif genre_hit:
        lead = (f"Genre-aligned with your {movie.genre} preference; its {movie.mood} affect "
                f"diverges from your target {profile.preferred_mood} but compensates "
                f"through {i_lbl} execution.")
    else:
        lead = (f"'{movie.title}' falls outside your primary genre vector ({profile.favorite_genre}) "
                f"but its {i_lbl} formal qualities satisfy your intensity signature ({movie.intensity:.2f}).")

    craft = (f"{movie.director}'s direction is characteristically {i_lbl} and {t_lbl}, "
             f"with pacing at {movie.pacing:.2f} and dialogue density at {movie.dialogue_heavy:.2f}.")
    award_note = " Critically canonised." if movie.awards else ""

    return f"{lead} {craft}{award_note}"


# ── Measurement helper ────────────────────────────────────────────────────────

def measure_modes(movie: Movie, profile: UserProfile) -> Dict[str, dict]:
    """
    Run all three modes and return word-count + lexical diversity metrics
    so the measurable difference in specialisation can be quantified.
    """
    results = {}
    for mode in ExplainerMode:
        text = specialize(movie, profile, mode)
        words = text.split()
        unique = set(w.lower().strip(".,!?'\"") for w in words)
        results[mode.value] = {
            "text":             text,
            "word_count":       len(words),
            "unique_words":     len(unique),
            "lexical_diversity": round(len(unique) / max(len(words), 1), 2),
        }
    return results
