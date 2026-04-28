"""
Agentic Workflow Enhancement — CineAgent executes recommendation as a
5-step chain with printed intermediate reasoning at each step.

Observable steps:
  1. PLAN    — analyse profile, choose ranking strategy
  2. RETRIEVE — load both data sources (CSV + RAG director profiles)
  3. FILTER  — count genre / mood pre-matches, flag ghost-genre edge cases
  4. SCORE   — run full scoring pipeline, log top result + confidence
  5. EXPLAIN — attach RAG-augmented explanation to each result
"""

import os
import sys
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))
from recommender import (
    Movie, UserProfile, Recommender,
    load_movies, _DEFAULT_WEIGHTS, confidence_score,
)
from rag_retriever import RAGRetriever

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
_MOVIES_PATH = os.path.join(_BASE_DIR, "data", "movies.csv")
_PROFILES_PATH = os.path.join(_BASE_DIR, "data", "director_profiles.json")

# Return type: (Movie, raw_score, rag_context_or_None)
AgentResult = Tuple[Movie, float, Optional[str]]


class CineAgent:
    """
    Recommendation agent that makes its reasoning chain fully observable.
    Call run() to see every decision printed step-by-step.
    """

    def __init__(
        self,
        movies_path: str = _MOVIES_PATH,
        profiles_path: str = _PROFILES_PATH,
    ):
        self._movies_path = movies_path
        self._profiles_path = profiles_path
        self._movies: Optional[List[Movie]] = None
        self._rag: Optional[RAGRetriever] = None

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        profile: UserProfile,
        k: int = 5,
        verbose: bool = True,
    ) -> List[AgentResult]:
        TOTAL = 5

        def step(n: int, label: str, msg: str) -> None:
            if verbose:
                print(f"  [STEP {n}/{TOTAL}] {label}: {msg}")

        if verbose:
            print(f"\n{'-'*60}")
            print(f"[AGENT] Recommend for '{profile.name}'")

        # ── Step 1: Plan ──────────────────────────────────────────────────────
        strategy = self._choose_strategy(profile)
        step(1, "PLAN",
             f"genre={profile.favorite_genre!r}, mood={profile.preferred_mood!r}, "
             f"intensity={profile.target_intensity:.2f} "
             f"=> strategy: {strategy!r}")

        # ── Step 2: Retrieve ──────────────────────────────────────────────────
        if self._movies is None:
            self._movies = load_movies(self._movies_path)
        if self._rag is None:
            self._rag = RAGRetriever(self._profiles_path)
        step(2, "RETRIEVE",
             f"{len(self._movies)} movies from CSV  +  "
             f"{len(self._rag._profiles)} director profiles from RAG store")

        # ── Step 3: Filter ────────────────────────────────────────────────────
        genre_hits = [m for m in self._movies
                      if m.genre.lower() == profile.favorite_genre.lower()]
        mood_hits  = [m for m in self._movies
                      if m.mood.lower()  == profile.preferred_mood.lower()]
        genre_note = (f"{len(genre_hits)} genre match(es)" if genre_hits
                      else "0 genre matches — GHOST GENRE, falling back to vibe similarity")
        step(3, "FILTER",
             f"{genre_note}, {len(mood_hits)} mood match(es) "
             f"out of {len(self._movies)} total")

        # ── Step 4: Score ─────────────────────────────────────────────────────
        rec = Recommender(self._movies)
        results = rec.recommend(profile, k=k, mode=strategy, diversity_penalty=0.20)
        top_movie, top_score = results[0]
        top_conf = confidence_score(top_score, _DEFAULT_WEIGHTS[strategy])
        step(4, "SCORE",
             f"top result: '{top_movie.title}'  "
             f"raw={top_score:.4f}  conf={top_conf:.0%}")

        # ── Step 5: Explain (RAG-augmented) ───────────────────────────────────
        augmented: List[AgentResult] = []
        rag_hits = 0
        for movie, score in results:
            ctx = self._rag.retrieve_director_context(movie.director)
            if ctx:
                rag_hits += 1
            augmented.append((movie, score, ctx))

        avg_conf = (sum(confidence_score(s, _DEFAULT_WEIGHTS[strategy])
                        for _, s, _ in augmented)
                    / len(augmented))
        step(5, "EXPLAIN",
             f"RAG context attached for {rag_hits}/{len(augmented)} results")

        if verbose:
            print(f"[AGENT] Done - {k} results, avg confidence: {avg_conf:.0%}")

        return augmented

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _choose_strategy(profile: UserProfile) -> str:
        """Pick ranking mode based on profile signal strength."""
        if profile.target_intensity >= 0.75:
            return "vibe-first"
        if profile.target_intensity <= 0.30 and profile.dialogue_preference >= 0.80:
            return "mood-first"
        return "balanced"
