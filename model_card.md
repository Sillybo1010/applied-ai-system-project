# Model Card — CineMatch

## System Description

**CineMatch** is a content-based movie recommendation engine. It scores films against a user taste profile using weighted similarity across 8 features (genre, mood, intensity, tone, pacing, dialogue preference, runtime, popularity), applies a director diversity penalty, and produces ranked recommendations with natural-language explanations.

- **Base project:** VibeFinder 1.0 (music recommender, Modules 1–3)
- **Language:** Python 3.x
- **No external ML model** — all scoring is deterministic, rule-based similarity math
- **Stretch features:** RAG retrieval, 5-step agentic chain, few-shot specialization, evaluation harness

---

## Intended Use

| Use case | Supported |
|---|---|
| Demonstrating explainable AI recommendation | Yes |
| Showing how confidence scoring flags low-quality matches | Yes |
| Illustrating RAG augmentation of natural-language output | Yes |
| Production movie recommendation at scale | No — catalog is 20 films, manually curated |
| Personalized recommendations from watch history | No — no user history or collaborative filtering |

---

## AI Collaboration

### How AI tools were used in building this system

Claude (Anthropic) was used throughout the project as a pair-programming assistant:

- **Architecture design:** Discussed how to translate the VibeFinder music scoring model into cinematic features; Claude helped identify which audio features mapped naturally to film attributes (e.g., energy → intensity, valence → tone).
- **Code generation:** Initial scaffolding for `score_movie()`, `confidence_score()`, and `explain_recommendation()` was drafted collaboratively, then hand-edited to match the project's specific weighting logic.
- **Stretch feature implementation:** The RAGRetriever, CineAgent 5-step chain, and Specializer few-shot templates were designed in dialogue with Claude, with the final implementations reviewed and adjusted manually.
- **Debugging:** Claude helped diagnose edge cases (ghost genre fallback, confidence paradox on conflicting profiles) and suggested the `WARNING` log approach for zero-genre-match situations.
- **Documentation:** README structure and model card prompts were shaped with Claude's assistance; all written content was reviewed and rewritten in the author's own words.

### What AI did NOT do

- AI did not define the grading rubric, project goals, or success criteria — those came from the course assignment.
- AI did not choose which 20 films to include or manually assign feature values to them — that was the author's editorial judgment.
- AI did not run or verify the code — all test runs and output verification were done by the author in a local environment.

---

## Bias and Limitations

### Data biases

| Bias | Description |
|---|---|
| **Catalog size** | 20 films is a non-representative sample of cinema. Recommendations reflect the author's selection choices more than any meaningful distribution of film taste. |
| **Western/English-language dominance** | Most catalog films are English-language, Hollywood or indie US/UK productions. Non-English cinema is underrepresented (only *Parasite*, *Moonlight*). |
| **Director coverage** | RAG profiles cover 17 directors — all from the catalog. A director not in `director_profiles.json` gets no RAG-augmented explanation. |
| **Manual feature assignment** | Intensity, tone, pacing, and dialogue values were assigned by the author. These reflect subjective interpretation, not measured audio/visual analysis. |
| **Genre taxonomy** | Genres are flat single-labels. A film like *Parasite* (thriller/dark comedy/drama) is labeled `thriller` — a user requesting `drama` will not find it via genre match. |

### Algorithmic limitations

- **Categorical-weight dominance:** Genre carries weight 3.0 — more than mood (2.0) and all continuous features combined. A genre match can override strong mismatches everywhere else, producing high-confidence but potentially wrong recommendations (demonstrated by the "Conflicting: Horror + Uplifting Tone" profile scoring 93%).
- **No personalization over time:** Every session starts fresh. There is no memory of previous recommendations, dislikes, or evolving taste.
- **Ghost genre confidence:** When no genre match exists, the system still computes a score from other features and can return 60%+ confidence. A future improvement would cap confidence at 50% when the genre weight earns zero.

### Ethical considerations

- CineMatch does not collect or store user data.
- All recommendations are fully explainable — the scoring formula, weights, and rationale for each result are printed explicitly.
- The system has no feedback loop, so it cannot amplify past biases through reinforcement.

---

## Testing Results

### Automated test harness (`tests/eval_harness.py`)

```
Result  : 8 passed, 0 failed out of 8 tests
Avg top-result confidence : 96%
All tests passed.
```

| Test | Result |
|---|---|
| Action fan: genre match in top result | PASS |
| Drama seeker: genre match in top result | PASS |
| Animation fan: animation appears in top 3 | PASS |
| Results are sorted highest score first | PASS |
| Confidence score is always 0.0–1.0 | PASS |
| Ghost genre scores lower confidence than real genre match | PASS |
| RAG: augmented explanation is longer than base explanation | PASS |
| Specializer: cinephile output longer than casual output | PASS |

### Human evaluation (`src/main.py` — 6 personas)

```
OK   High-Intensity Action Fan          top confidence: 98%
OK   Slow-Burn Drama Seeker             top confidence: 97%
OK   Feel-Good Family Night             top confidence: 99%
OK   Conflicting: Horror + Uplifting    top confidence: 93%  <-- known limitation
OK   Ghost Genre: Western               top confidence: 60%  <-- WARNING logged
OK   Max Mismatch: Quiet Arthouse       top confidence: 84%

Profiles with genre match in top 5 : 5/6
Average top-result confidence       : 88%
Low-confidence profiles (<50%)      : 0
```

### RAG uplift measurement

For the top result returned to the "High-Intensity Action Fan" profile (*Mad Max Fury Road*, director George Miller):

```
Base explanation    : 31 words
RAG-augmented       : 54 words  (+74.2%)
Added context       : Director style, known-for summary, thematic alignment
```

### Specializer mode comparison (*Mad Max Fury Road*)

| Mode | Words | Lexical diversity |
|---|---|---|
| standard | 18 | 0.83 |
| casual | 23 | 0.91 |
| cinephile | 45 | 0.89 |

---

## Reflection on AI Collaboration

**What worked well:** Using Claude as a sounding board for design decisions (e.g., "should runtime similarity use a hard cutoff or a continuous window?") produced better-reasoned choices than working alone. The ±60-minute runtime window came directly from a conversation comparing it to the ±100 BPM tempo window in VibeFinder.

**Where I had to push back:** Claude's first draft of `confidence_score()` normalized against a fixed theoretical maximum that did not account for mode-specific weight sums. I caught the bug during manual testing and corrected the formula. This reinforced that AI-generated code requires the same scrutiny as any other code.

**What I learned about AI-assisted development:** AI is most useful at the *design* and *explanation* stages, and least reliable at quietly handling edge cases in numeric logic. The ghost-genre confidence bug — where a system can appear highly confident despite having no genre match — was something I discovered through adversarial profiling, not through AI suggestion. Human evaluation remains irreplaceable for catching these classes of failure.
