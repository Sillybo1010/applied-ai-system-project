# CineMatch — System Architecture Diagram

```mermaid
flowchart TD
    %% ── Inputs ──────────────────────────────────────────────
    CSV["data/movies.csv\n20 movies · 14 features"]
    DIRJSON["data/director_profiles.json\n17 director style profiles (RAG)"]
    HUMAN["Human Input\nUserProfile\ngenre · mood · intensity\ntone · pacing · runtime"]

    %% ── Data Loader ─────────────────────────────────────────
    LOADER["Data Loader\nload_movies()\nParses CSV → Movie dataclasses"]
    RAG["RAGRetriever\nretrieve_director_context()\naugment_explanation()"]

    %% ── Scoring Engine ──────────────────────────────────────
    WEIGHTS["_DEFAULT_WEIGHTS\nRanking Mode\nbalanced · genre-first\nmood-first · vibe-first"]
    SCORER["Scorer\nscore_movie()\nCategorical: genre + mood\nContinuous: intensity · tone\npacing · dialogue · runtime"]

    %% ── Diversity Filter ────────────────────────────────────
    DIVERSITY["Diversity Filter\napply_director_penalty()\n-20% per repeat director"]

    %% ── Agentic Chain ───────────────────────────────────────
    CINEAGENT["CineAgent (agent.py)\n5-step observable chain\nPLAN → RETRIEVE → FILTER\nSCORE → EXPLAIN"]

    %% ── Specializer ─────────────────────────────────────────
    SPEC["Specializer (specializer.py)\nFew-shot tone modes\nSTANDARD · CASUAL · CINEPHILE"]

    %% ── Recommender Agent ───────────────────────────────────
    AGENT["Recommender Agent\nRecommender.recommend()\nOrchestrates score → filter → rank"]

    %% ── Explainer ───────────────────────────────────────────
    EXPLAIN["Explainer\nexplain_recommendation()\nWhy each film was picked"]

    %% ── Outputs ─────────────────────────────────────────────
    OUTPUT["Output\nRanked Top-K Movies\nTitle · Score · Confidence · Why"]

    %% ── Human Review ────────────────────────────────────────
    REVIEW["Human Review\nmain.py profiles\n6 personas · mode comparison\ndiversity ON vs OFF · RAG demo"]

    %% ── Testing Layer ───────────────────────────────────────
    TESTS["Eval Harness\ntests/eval_harness.py\n8 predefined inputs\nPASS/FAIL + confidence"]

    %% ── Data flow ───────────────────────────────────────────
    CSV        --> LOADER
    DIRJSON    --> RAG
    LOADER     --> AGENT
    HUMAN      --> AGENT
    HUMAN      --> WEIGHTS
    WEIGHTS    --> SCORER
    AGENT      --> SCORER
    SCORER     --> DIVERSITY
    DIVERSITY  --> AGENT
    AGENT      --> EXPLAIN
    RAG        --> EXPLAIN
    EXPLAIN    --> SPEC
    SPEC       --> OUTPUT
    OUTPUT     --> REVIEW

    CINEAGENT  --> AGENT
    CINEAGENT  --> RAG

    %% ── Testing hooks ───────────────────────────────────────
    AGENT    -.->|"unit test"| TESTS
    EXPLAIN  -.->|"unit test"| TESTS
    RAG      -.->|"unit test"| TESTS
    SPEC     -.->|"unit test"| TESTS
    TESTS    -.->|"pass / fail"| REVIEW

    %% ── Styles ──────────────────────────────────────────────
    style HUMAN      fill:#d4edda,stroke:#28a745,color:#000
    style REVIEW     fill:#d4edda,stroke:#28a745,color:#000
    style TESTS      fill:#fff3cd,stroke:#ffc107,color:#000
    style AGENT      fill:#cce5ff,stroke:#004085,color:#000
    style CINEAGENT  fill:#cce5ff,stroke:#004085,color:#000
    style SCORER     fill:#cce5ff,stroke:#004085,color:#000
    style DIVERSITY  fill:#cce5ff,stroke:#004085,color:#000
    style EXPLAIN    fill:#cce5ff,stroke:#004085,color:#000
    style SPEC       fill:#cce5ff,stroke:#004085,color:#000
    style RAG        fill:#fde8d8,stroke:#e07020,color:#000
    style DIRJSON    fill:#f8d7da,stroke:#721c24,color:#000
    style CSV        fill:#f8d7da,stroke:#721c24,color:#000
    style LOADER     fill:#f8d7da,stroke:#721c24,color:#000
    style WEIGHTS    fill:#e2d9f3,stroke:#6f42c1,color:#000
    style OUTPUT     fill:#f0f0f0,stroke:#666,color:#000
```

**Color key**
- Green — human involvement (input profiles, review)
- Blue — AI/processing pipeline (recommender, agent, explainer, specializer)
- Orange — RAG layer (director profile retrieval)
- Red — data layer (CSV + loader)
- Purple — configuration (weights / ranking modes)
- Yellow — evaluation / testing layer
- Dotted arrows — test hooks validating AI outputs
