# Soft-Gate benchmarks

Benchmark code and result tables for the paper *The Legible Gate: Toward an Architecture of Interpretable Specialization in Multi-Domain Retrieval and Generation*.

Soft-Gate routes documents and queries to per-domain HNSW indexes ("experts")
using a softmax over negative L2 distance to each expert's profile embedding.
This repository reproduces every retrieval number in the paper (systems
B0–B19), the gate and calibration analyses, and the LLM endpoint-selection
experiment (B20–B21).

## Setup

Python 3.10+ is required.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Datasets and models are downloaded on first run into `benchmarks/data/`
(about 150 MB, git-ignored).

## Reproducing the paper

Run from the repository root. Each script writes into `tables/`.

| Script | Paper section | Output |
|---|---|---|
| `python benchmarks/run_benchmark.py` | §VI-A–D: retrieval quality, latency, ablations, significance (B0–B19) | `main-results.csv`, `latency.csv`, `significance.csv`, `ablations.csv`, `results.json`, `appendix/experimental-config.md` |
| `python benchmarks/gate_analysis.py` | §VI-A, §VI-C: gate distribution, temperature sweep, per-domain breakdown, misroutes | `gate_analysis.json` |
| `python benchmarks/calibration_sweep.py` | §VI-C: learned-router temperature scaling | `calibration_sweep.json` |
| `python benchmarks/run_llm_benchmark.py` | §VI-E: LLM endpoint selection (B20–B21) | `llm-results.csv`, `llm-results.json` |

`calibration_sweep.py` reuses the corpus and embedding cache written by
`gate_analysis.py`, so run `gate_analysis.py` first.

### Qdrant baseline (B19)

B19 compares B0's hnswlib index against Qdrant. qdrant-client's local mode
does not build an HNSW graph, it searches exhaustively, so for an HNSW-to-HNSW
comparison point B19 at a Qdrant server:

```bash
docker run -p 6333:6333 qdrant/qdrant
QDRANT_URL=http://localhost:6333 python benchmarks/run_benchmark.py
```

Without `QDRANT_URL`, B19 runs in local mode and is reported as an
exact-search reference.

### LLM experiment

B21 needs a local [Ollama](https://ollama.com) server; B20 needs AWS
credentials with Bedrock access.

```bash
ollama serve
ollama pull qwen2.5-coder:32b              # override with OLLAMA_MODEL=<tag>

export BEDROCK_MODEL_ID=<model-id-or-inference-profile-arn>
export AWS_REGION=eu-west-1                # default
python benchmarks/run_llm_benchmark.py

python benchmarks/run_llm_benchmark.py --skip-bedrock   # Ollama only
```

## Configuration

All settings are constants at the top of each script: seed 42, `intfloat/e5-base-v2`
embeddings, HNSW `M=16, efConstruction=200, efSearch=128`, top-k 10, at most 4,500
documents and 450 queries per domain. The full configuration for the published
run is in [`appendix/experimental-config.md`](appendix/experimental-config.md).

The published run used Apple Silicon. Retrieval quality should reproduce exactly
for a fixed seed and library versions; latency numbers depend on hardware.

## Data

Nothing is redistributed here. Each dataset is fetched at runtime and used
under its own license:

| Domain | Dataset |
|---|---|
| Finance | FiQA-2018 (BEIR) |
| Legal | CaseHOLD (LexGLUE), converted to IR: holdings as documents, citing contexts as queries |
| Technical | NFCorpus (BEIR) |
| Code (LLM experiment) | CodeSearchNet (Python, Java); `b-mc2/sql-create-context` (SQL) |

## Citation

```bibtex
@article{fikri2026legiblegate,
  title  = {The Legible Gate: Toward an Architecture of Interpretable Specialization in Multi-Domain Retrieval and Generation},
  author = {Fikri, Noussair and Abghour, Nourddine and Moussaid, Khalid},
  year   = {2026},
  note   = {Under review}
}
```

## License

MIT. See [LICENSE](LICENSE).
