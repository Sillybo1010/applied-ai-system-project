import logging
import os
import sys

try:
    from recommender import UserProfile, Recommender, load_movies
    from rag_retriever import RAGRetriever
except ImportError:
    from src.recommender import UserProfile, Recommender, load_movies
    from src.rag_retriever import RAGRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)

try:
    from tabulate import tabulate
except ImportError:
    print("Install tabulate:  pip install tabulate")
    sys.exit(1)

DATA_PATH     = os.path.join(os.path.dirname(__file__), "..", "data", "movies.csv")
PROFILES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "director_profiles.json")

# ── User profiles: 3 standard + 3 edge-case ──────────────────────────────────

profiles = [
    # Standard
    UserProfile(
        name="High-Intensity Action Fan",
        favorite_genre="action",
        preferred_mood="intense",
        target_intensity=0.90,
        target_runtime=120,
        tone_preference=0.50,
        pacing_preference=0.90,
        dialogue_preference=0.20,
    ),
    UserProfile(
        name="Slow-Burn Drama Seeker",
        favorite_genre="drama",
        preferred_mood="emotional",
        target_intensity=0.25,
        target_runtime=140,
        tone_preference=0.65,
        pacing_preference=0.25,
        dialogue_preference=0.95,
    ),
    UserProfile(
        name="Feel-Good Family Night",
        favorite_genre="animation",
        preferred_mood="heartwarming",
        target_intensity=0.45,
        target_runtime=95,
        tone_preference=0.90,
        pacing_preference=0.70,
        dialogue_preference=0.65,
    ),
    # Edge-case: contradicting preferences
    UserProfile(
        name="Conflicting: Horror + Uplifting Tone",
        favorite_genre="horror",
        preferred_mood="scary",
        target_intensity=0.80,
        target_runtime=110,
        tone_preference=0.90,   # horror is typically dark, not uplifting
        pacing_preference=0.65,
        dialogue_preference=0.50,
    ),
    # Edge-case: genre not in catalog
    UserProfile(
        name="Ghost Genre: Western",
        favorite_genre="western",
        preferred_mood="intense",
        target_intensity=0.75,
        target_runtime=130,
        tone_preference=0.50,
        pacing_preference=0.70,
        dialogue_preference=0.40,
    ),
    # Edge-case: max mismatch (wants opposite of catalog)
    UserProfile(
        name="Max Mismatch: Quiet Arthouse",
        favorite_genre="romance",
        preferred_mood="romantic",
        target_intensity=0.05,
        target_runtime=200,
        tone_preference=0.50,
        pacing_preference=0.05,
        dialogue_preference=1.00,
    ),
]


# ── Display helpers ───────────────────────────────────────────────────────────

def results_to_table(results, rec: Recommender, profile: UserProfile) -> str:
    rows = []
    for rank, (movie, score) in enumerate(results, 1):
        conf = rec.confidence_for(score)
        explanation = rec.explain_recommendation(profile, movie).replace("\n", " | ")
        rows.append([
            rank,
            f"{movie.title} / {movie.director}",
            f"{score:.4f}",
            f"{conf:.0%}",
            f"{movie.genre} / {movie.mood}",
            f"{movie.intensity:.2f}",
            f"{movie.tone:.2f}",
            f"{movie.pacing:.2f}",
            f"{movie.dialogue_heavy:.2f}",
            explanation,
        ])
    headers = ["#", "Title / Director", "Score", "Conf.", "Genre / Mood",
               "Intens.", "Tone", "Pacing", "Dialogue", "Why"]
    return tabulate(rows, headers=headers, tablefmt="grid")


def print_profile(profile: UserProfile) -> None:
    print(f"\n{'='*70}")
    print(f"Profile : {profile.name}")
    print(f"  Genre / Mood  : {profile.favorite_genre} / {profile.preferred_mood}")
    print(f"  Intensity     : {profile.target_intensity:.2f}   Runtime target: {profile.target_runtime} min")
    print(f"  Tone / Pacing : {profile.tone_preference:.2f} / {profile.pacing_preference:.2f}")
    print(f"  Dialogue pref : {profile.dialogue_preference:.2f}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    movies = load_movies(DATA_PATH)
    rec = Recommender(movies)

    print("CineMatch — Movie Recommender Simulation")
    print("Scoring max: genre(3.0) + mood(2.0) + intensity(1.0) + tone(0.75)")
    print("           + pacing(0.50) + dialogue(0.50) + runtime(0.50) + popularity(0.25)")
    print("Director diversity penalty: 20%")

    # Run all profiles with balanced mode
    for profile in profiles:
        print_profile(profile)
        results = rec.recommend(profile, k=5, mode="balanced", diversity_penalty=0.20)
        print(results_to_table(results, rec, profile))

    # Ranking mode comparison for "High-Intensity Action Fan"
    demo = profiles[0]
    print(f"\n{'='*70}")
    print(f"Ranking mode comparison — {demo.name}")
    for mode in ["balanced", "genre-first", "mood-first", "vibe-first"]:
        results = rec.recommend(demo, k=3, mode=mode, diversity_penalty=0.0)
        titles = ", ".join(f"{m.title} ({s:.3f})" for m, s in results)
        print(f"  [{mode:12s}]  {titles}")

    # Diversity penalty ON vs OFF for "Slow-Burn Drama Seeker"
    drama = profiles[1]
    print(f"\n{'='*70}")
    print(f"Diversity penalty effect — {drama.name}")
    for label, penalty in [("Penalty OFF (0%)", 0.0), ("Penalty ON (20%)", 0.20)]:
        results = rec.recommend(drama, k=5, mode="balanced", diversity_penalty=penalty)
        titles = ", ".join(m.title for m, _ in results)
        print(f"  {label}: {titles}")

    # ── RAG Enhancement: before / after demonstration ─────────────────────────
    rag = RAGRetriever(PROFILES_PATH)
    demo_profile = profiles[0]  # High-Intensity Action Fan
    demo_result  = rec.recommend(demo_profile, k=1, mode="balanced")[0][0]

    base_explanation = rec.explain_recommendation(demo_profile, demo_result)
    augmented_explanation = rag.augment_explanation(base_explanation, demo_result, demo_profile)
    uplift = RAGRetriever.measure_improvement(base_explanation, augmented_explanation)

    print(f"\n{'='*70}")
    print("RAG ENHANCEMENT — Before / After")
    print(f"{'='*70}")
    print(f"Movie  : {demo_result.title}  (director: {demo_result.director})")
    print(f"\n  [BASE — no RAG]\n  {base_explanation}")
    print(f"\n  [AUGMENTED — with RAG director profile]\n  {augmented_explanation}")
    print(f"\n  Uplift : +{uplift['added_words']} words "
          f"({uplift['base_words']} -> {uplift['aug_words']}, "
          f"+{uplift['pct_increase']}%)")

    # ── Evaluation summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")

    conf_scores = []
    genre_hit_count = 0
    low_conf_profiles = []

    for profile in profiles:
        results = rec.recommend(profile, k=5, mode="balanced", diversity_penalty=0.20)
        top_conf = rec.confidence_for(results[0][1]) if results else 0.0
        conf_scores.append(top_conf)

        has_genre_hit = any(
            m.genre.lower() == profile.favorite_genre.lower() for m, _ in results
        )
        if has_genre_hit:
            genre_hit_count += 1
        if top_conf < 0.50:
            low_conf_profiles.append(profile.name)

        flag = "! LOW CONF" if top_conf < 0.50 else "OK"
        print(f"  {flag}  {profile.name:<40}  top confidence: {top_conf:.0%}")

    avg_conf = sum(conf_scores) / len(conf_scores)
    print(f"\n  Profiles with genre match in top 5 : {genre_hit_count}/{len(profiles)}")
    print(f"  Average top-result confidence       : {avg_conf:.0%}")
    print(f"  Low-confidence profiles (<50%)      : {len(low_conf_profiles)}")
    if low_conf_profiles:
        for name in low_conf_profiles:
            print(f"    - {name}")
