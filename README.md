# Redrob AI Candidate Ranking System — Team *Quiet Disasters*

**Live Sandbox (HuggingFace Spaces):** https://huggingface.co/spaces/Abhii2005/Quiet-Disasters-Ranker

This repository contains the full source code, dependencies, precomputation scripts, and exact reproduction commands for our submission to the Redrob AI Candidate Ranking Hackathon.

Our solution ranks the top 100 candidates from a 100,000-candidate pool for the released Senior AI Engineer job description. It is a **multi-stage retrieval-and-ranking pipeline** built to satisfy the competition's hard constraints: **CPU-only, no network, ≤ 16 GB RAM, ≤ 5 minutes** for the ranking step.

We deliberately avoid the "one LLM call per candidate" approach. That cannot scale to a production pool inside the compute budget. Instead we combine dense vector retrieval, a local cross-encoder, and a transparent feature-scoring layer — each stage chosen for a specific latency-vs-quality tradeoff.

---

## 1. Architecture at a glance

```
candidates.jsonl ──(precompute.py, offline, one-time)──▶ artifacts/  (~850 MB)
                                                              │
                                                              ▼
                                          rank.py  (CPU-only, offline, ~90 - 110s)
                                                              │
                                                              ▼
                                                       submission.csv  (top 100)
```

The system is **two-phase by design**, which the spec explicitly permits (Section 10.3: *"pre-computation may exceed the 5-minute window, but the ranking step that produces the CSV must complete within it"*):

- **Phase A — Precompute (`precompute.py`):** runs once, offline, may use a GPU and downloads models from HuggingFace. Encodes all 100K candidates, builds the FAISS index, precomputes skill matches and honeypot flags, and saves the cross-encoder locally. Output: the `artifacts/` directory.
- **Phase B — Rank (`rank.py`):** the scored submission step. Loads only the precomputed `artifacts/`, runs entirely on CPU with **no network**, and writes the final CSV in well under the 5-minute budget.

Because the candidate embeddings and JD vectors are precomputed, the ranking step never re-encodes raw candidate text — it scores vectors. This is what keeps Phase B fast and CPU-only.

---

## 2. Reproduction (Stage 3)

### Step 0 — Reconstruct the dataset

GitHub caps single files at 100 MB, so the ~465 MB `candidates.jsonl` is committed as 10 line-aligned shards under `data_chunks/`. Reconstruct it first:

```bash
# Merge ALL chunks -> full 100K dataset (candidates.jsonl)
python merge_chunks.py

# Optional: merge only specific parts for quick partial testing (e.g. first 30K)
# python merge_chunks.py --parts 1 2 3 --out candidates_30k.jsonl
```

> **Do not skip this.** `candidates.jsonl` is git-ignored due to size. If you run `precompute.py` before merging, it will fail with a *file not found* error.

### Step 1 — Precompute artifacts (one-time, offline)

```bash
pip install -r requirements.txt

python precompute.py --candidates ./candidates.jsonl --out-dir ./artifacts/
```

This reads `candidates.jsonl`, downloads the bi-encoder (`BAAI/bge-small-en-v1.5`) and cross-encoder (`cross-encoder/ms-marco-MiniLM-L-12-v2`), computes embeddings for all 100K candidates, builds the FAISS index, and **saves the cross-encoder into `artifacts/models/ce`** so the ranking step needs no network. Output: `./artifacts/` (~850 MB). Network is used here (model download) and this phase may exceed 5 minutes — both are allowed for precompute.

### Step 2 — Rank (the single submission command)

```bash
python rank.py --candidates ./candidates.jsonl --artifacts ./artifacts --out ./submission.csv
```

This is the **single command that produces the submission CSV** (Section 10.3). It runs **entirely on CPU**, makes **no network calls** (the cross-encoder loads from the local `artifacts/models/ce` path), and completes in **~110 - 205 seconds** on a 10-core CPU — well within the 5-minute budget. It outputs exactly 100 candidates with monotonically non-increasing scores, deterministic tie-breaks (equal scores ordered by `candidate_id` ascending), and per-candidate reasoning strings. The script self-validates header, row count, monotonicity, and tie-breaks before writing.

> Note: `rank.py` reads the precomputed `artifacts/` (representing all 100K candidates) rather than re-parsing the raw file; the `--candidates` flag is accepted for spec-command compatibility. This is what allows the ranking step to stay within the CPU/time budget.

### Step 3 — Validate

```bash
python validate_submission.py submission.csv
# expect: "Submission is valid."
```

---

## 3. Compute compliance (ranking step)

| Constraint | Limit | Our ranking step |
| --- | --- | --- |
| Runtime | ≤ 5 min | ~110 - 205 s |
| Memory | ≤ 16 GB | within budget |
| Compute | CPU only | CPU only (no GPU) |
| Network | Off | Off — cross-encoder loaded from local `artifacts/models/ce` |
| Disk | ≤ 5 GB intermediate | within budget |

---

## 4. Pipeline detail

### Stage 1 — Retrieval (100K → 500)
Builds a high-recall candidate pool by taking the **union** of three signals, then trims to the top 500:
- **Dense semantic search** via `BAAI/bge-small-en-v1.5` embeddings (384-dim) + FAISS. JD queries and role archetypes are encoded with BGE's query-side instruction; candidate profiles are encoded as passages.
- **Skill match pool** — candidates with the most JD must-have skill matches (threshold auto-calibrated at precompute).
- **Behavioral pool** — top candidates by availability signals.

Non-technical titles and hard honeypots are excluded before scoring.

### Stage 2 — Cross-encoder re-ranking (500)
Runs `cross-encoder/ms-marco-MiniLM-L-12-v2` locally on CPU over the 500-candidate pool, scoring deep JD-relevance that the bi-encoder misses. Outputs are sigmoid-normalized.

### Stage 3 — Seven-feature weighted scoring
Each candidate's final score is a weighted blend of:

| Feature | Weight | What it measures |
| --- | --- | --- |
| Career domain evidence | 0.32 | Cross-encoder + semantic relevance to the JD domain (+ product-company lift) |
| Retrieval / search expertise | 0.26 | Depth & coverage of retrieval / vector-DB skills |
| Production deployment | 0.15 | Evidence of shipping/scaling ML in production |
| Vector-DB infrastructure | 0.10 | Direct tool detection (Pinecone, Milvus, Weaviate, Qdrant, FAISS) |
| Behavioral availability | 0.10 (capped at 0.80) | Notice period, recruiter response rate, recency |
| LLM-adjacent experience | 0.04 | Nice-to-have skill coverage |
| Career progression | 0.03 | Title trajectory (junior → senior) |

Skill credibility is itself a weighted blend: `0.35·proficiency + 0.25·endorsements + 0.25·duration + 0.15·assessment`.
**JD-fit signals** (reading what the JD means, not just keywords)

### Multiplicative penalties
- **Product-company evidence (lift):** real tenure at a product company raises domain evidence — a candidate who built a recommender at a product company isn't buried for lacking buzzwords.
- **Keyword-stuffer (×0.20):** many JD-matching skills but near-zero duration/endorsements.
- **Services-firm (×0.40 / ×0.80):** candidates from services firms lacking product-company domain evidence (×0.40 when domain is weak, ×0.80 when partial).
- **Title-chaser / job-hopping (×0.90):** short average tenure with rising titles — the pattern the JD calls out.
- **CV/speech/robotics without NLP/IR (×0.85):** the JD says these are not a fit without retrieval/NLP depth.
- **Research-only without production (×0.70):** the JD explicitly rejects pure-research profiles.
- **Recent-framework-only (×0.92):** LangChain/LLM-API only, no pre-LLM ML depth.
- **Location (×0.85):** outside India and not open to relocation (the JD doesn't sponsor visas).
- **Experience band (gentle nudge, ×0.80 minimum):** soft down-weight outside the 5–9yr range — deliberately gentle because the JD treats 5–9 as "a range, not a requirement" and asks that strong candidates outside it still be considered.
- **Honeypot soft penalties (×0.75 / ×0.50):** profiles with suspicious-but-not-conclusive impossibility signals.
- **Availability cap (0.80):** prevents behavioral signals from overwhelming domain expertise.

### Reasoning generation
Reasoning strings are **built from each candidate's parsed attributes** current title, company, years of experience, named skills with endorsement counts and durations, and behavioral signals — in plain , varied language (no templated skeleton, no internal model scores).  The narrative is anchored on the highest weighted-contribution feature, connects to specific **JD** requirements, and **acknowledges** concerns honestly (long notice period, off-band experience, location, services-only career) where present. Every claim corresponds to a real profile field (no hallucination) and strings vary structurally across candidates.

---

## 5. Honeypot handling

The dataset seeds ~80 honeypot profiles (subtly impossible: e.g. 8 years at a 3-year-old company; "expert" in 10 skills with 0 months used) that are forced to ground-truth tier 0. Submissions with a honeypot rate > 10% in the top 100 are disqualified at Stage 3.

We use an explicit two-layer detector:
- **Layer 1 (precompute):** flags profiles claiming high proficiency with zero duration, timeline mismatches, and experience inflation.
- **Layer 2 (runtime):**  catches ghost skills (0 endorsements + 0 duration yet high JD match) and implausibly broad coverage.

Hard-flagged honeypots are excluded from the pool; borderline cases receive soft score penalties. Measured honeypot rate in our final top-100 submission: **0%**.

---

## 6. Sandbox / demo (Section 10.5)

**Live:** https://huggingface.co/spaces/Abhii2005/Quiet-Disasters-Ranker

A Streamlit app (`app.py`) wraps the pipeline for the small-sample reproducibility check. It offers two run modes:

- **Run on Pre-loaded 100K Dataset** → executes `rank.py` against the precomputed artifacts and shows the ranked top-100 table, timing, and pipeline logs.
- **Run on Uploaded Sample** → executes `rank_small.py`. This parses the candidate IDs from an uploaded sample (≤100 candidates) and filters the precomputed artifacts down to **only those candidates**, then runs the identical Stage 1–3 scoring on that subset and emits a correspondingly sized CSV (e.g. 50 rows for a 50-candidate upload). Note: this re-ranks candidates that exist in the precomputed pool; it is a reproducibility demo of the scoring logic, not live re-embedding of novel profiles.

Per Section 10.5, the sandbox only needs to handle a small sample and complete within the CPU budget — full 100K reproduction is done from this repo at Stage 3.

---

## 7. Repository structure

```text
├── README.md                 # This file
├── requirements.txt          # All dependencies with versions (see note below)
├── submission_metadata.yaml  # Metadata mirroring the portal submission
├── merge_chunks.py           # Reconstructs candidates.jsonl from data_chunks/
├── data_chunks/              # 10 line-aligned shards of candidates.jsonl
├── precompute.py             # Phase A: builds embeddings, FAISS index, artifacts
├── rank.py                   # Phase B: CPU-only ranking step (produces submission.csv)
├── rank_small.py             # Sandbox: ranks an uploaded sub-sample of the pool
├── app.py                    # Streamlit UI for the HuggingFace Spaces sandbox
├── validate_submission.py    # CSV format validator
└── sample_candidates.json    # Small sample for local sandbox testing
```

> `candidates.jsonl` and `artifacts/` are **not** committed (size). Reconstruct the dataset with `merge_chunks.py` and generate artifacts with `precompute.py`.

### Core dependency stack
`sentence-transformers`, `transformers`, `torch`, `faiss-cpu`, `numpy`, `pandas`, `pyarrow`, `streamlit` (sandbox only).

> Exact, pinned versions are in `requirements.txt`, generated from the working environment with `pip freeze` so the Stage-3 reproduction resolves the same versions used to build the artifacts.

---

## 8. Development & AI tooling

This submission was built by Team Quiet Disasters with AI tools used as assisted-engineering aids. Full per-tool declaration is in `submission_metadata.yaml`.

No candidate data is sent to any hosted LLM during ranking; the ranking step runs fully offline on CPU.
