"""
RAG Enhancement — retrieves context from a secondary document store
(director_profiles.json) to augment movie recommendations.

Two data sources:
  1. data/movies.csv        — structured feature data (primary)
  2. data/director_profiles.json — unstructured style/theme text (secondary, RAG)
"""

import json
import os
from typing import Optional

from recommender import Movie, UserProfile


class RAGRetriever:
    def __init__(self, profiles_path: str):
        if not os.path.exists(profiles_path):
            raise FileNotFoundError(f"Director profiles not found: {profiles_path}")
        with open(profiles_path, encoding="utf-8") as f:
            self._profiles: dict = json.load(f)

    # ── Retrieval methods ─────────────────────────────────────────────────────

    def retrieve_director_context(self, director: str) -> Optional[str]:
        """Return a one-line style summary for a director, or None if unknown."""
        p = self._profiles.get(director)
        if not p:
            return None
        return f"{p['known_for']}. Style: {p['style']}."

    def retrieve_theme_alignment(self, director: str, profile: UserProfile) -> Optional[str]:
        """Return themes that overlap with the user's preferred mood keywords."""
        p = self._profiles.get(director, {})
        themes = p.get("themes", [])
        mood_words = set(profile.preferred_mood.lower().split())
        # Simple keyword overlap between mood words and theme text
        hits = [t for t in themes if any(w in t.lower() for w in mood_words)]
        if hits:
            return "Matching themes: " + ", ".join(hits)
        return None

    def augment_explanation(
        self, base: str, movie: Movie, profile: UserProfile
    ) -> str:
        """Append RAG-sourced director context to a base explanation."""
        lines = [base]
        ctx = self.retrieve_director_context(movie.director)
        if ctx:
            lines.append(f"  [RAG] Director — {ctx}")
        theme = self.retrieve_theme_alignment(movie.director, profile)
        if theme:
            lines.append(f"  [RAG] {theme}")
        return "\n".join(lines)

    # ── Measurement helper ────────────────────────────────────────────────────

    @staticmethod
    def measure_improvement(base: str, augmented: str) -> dict:
        """Return word-count delta so callers can quantify the RAG uplift."""
        base_words = len(base.split())
        aug_words  = len(augmented.split())
        return {
            "base_words":  base_words,
            "aug_words":   aug_words,
            "added_words": aug_words - base_words,
            "pct_increase": round((aug_words - base_words) / max(base_words, 1) * 100, 1),
        }
