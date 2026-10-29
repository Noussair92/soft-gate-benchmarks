#!/usr/bin/env python3
"""
LLM Soft-Gate benchmark: generalist API model vs. local code model on a fixed retrieved pool.

B20: flat retrieval → Claude Sonnet (via AWS Bedrock,
     generalist, API cost ~$3/1M in + $15/1M out)
B21: flat retrieval → local code model via Ollama
     (OLLAMA_MODEL, default qwen2.5-coder:32b; zero marginal cost)

Both systems receive the same retrieved pool, so the comparison isolates the
generation step an expert's LLM endpoint would control. It does not exercise
gate-driven endpoint selection.

Scope: Java / Python / SQL code retrieval.
  - Given a natural-language description (docstring / question), select the matching
    code snippet from a retrieved pool of candidates.
  - Metric: accuracy (LLM selects the gold snippet from the retrieved pool).

Data sources:
  Python / Java: code_search_net  (docstring → function body)
  SQL:           b-mc2/sql-create-context  (question → SQL answer)

Prerequisites
  Ollama:   ollama serve                      # start daemon
            ollama pull qwen2.5-coder:32b     # or set OLLAMA_MODEL=<tag>
  Bedrock:  AWS credentials with bedrock:InvokeModel in AWS_REGION (default eu-west-1)
            Required:  BEDROCK_MODEL_ID=<model-id-or-inference-profile-arn>

Usage:
  python benchmarks/run_llm_benchmark.py [--max-queries-per-lang N]
  python benchmarks/run_llm_benchmark.py --skip-bedrock   # Ollama only (no AWS needed)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "benchmarks" / "data"
OUT_DIR = ROOT / "tables"
SEED = 42

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128
RETRIEVE_POOL = 5   # candidates shown to the LLM; must be >= 1

EMBED_MODEL = "intfloat/e5-base-v2"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:32b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "")
BEDROCK_REGION = os.environ.get("AWS_REGION", "eu-west-1")

# Claude 3.5 Sonnet pricing (USD per 1M tokens, as of 2025)
BEDROCK_INPUT_COST_PER_1M = 3.00
BEDROCK_OUTPUT_COST_PER_1M = 15.00

MAX_LLM_TOKENS = 512  # reasoning models (R1) emit <think> blocks before the answer

CODE_SPECIALIST_SYSTEM_PROMPT = (
    "You are a software engineer with deep expertise in Java, Python, and SQL. "
    "You understand code semantics, algorithms, data structures, and query languages."
)
GENERALIST_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant with broad expertise across programming languages "
    "including Java, Python, and SQL."
)

MAX_CONTEXT_CHARS = 1200   # truncate long contexts before sending to LLM
MAX_HOLDING_CHARS = 600    # truncate long code snippets


# ---------------------------------------------------------------------------
# Retrieval helpers (mirror of run_benchmark.py — kept here for standalone use)
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)



class HNSWIndex:
    def __init__(self, dim: int) -> None:
        import hnswlib
        self.ids: List[int] = []
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(max_elements=1, ef_construction=HNSW_EF_CONSTRUCTION, M=HNSW_M)
        self.index.set_ef(HNSW_EF_SEARCH)
        self._capacity = 1

    def add(self, ids: List[int], vectors: np.ndarray) -> None:
        needed = len(self.ids) + len(ids)
        if needed > self._capacity:
            self._capacity = max(needed, self._capacity * 2)
            self.index.resize_index(self._capacity)
        start = len(self.ids)
        self.index.add_items(vectors, list(range(start, start + len(ids))))
        self.ids.extend(ids)

    def search(self, vector: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        if not self.ids:
            return [], []
        labels, distances = self.index.knn_query(vector, min(k, len(self.ids)))
        return [self.ids[i] for i in labels[0]], [1.0 - d for d in distances[0]]


def embed_texts(model, texts: List[str], *, prefix: str = "") -> np.ndarray:
    prefixed = [prefix + t for t in texts] if prefix else texts
    return np.asarray(
        model.encode(prefixed, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    )



def load_code_dataset(max_docs_per_lang: int = 1500, max_queries_per_lang: int = 150) -> Tuple[List[dict], List[dict], Dict]:
    """Load Java/Python (CodeSearchNet) and SQL (sql-create-context) for code retrieval."""
    from datasets import load_dataset

    docs: List[dict] = []
    raw_queries: List[dict] = []
    raw_qrels: Dict[str, int] = {}

    # Python and Java from CodeSearchNet
    for lang in ["python", "java"]:
        ds = load_dataset("code_search_net", lang, split="test", trust_remote_code=True)
        indices = list(range(len(ds)))
        random.shuffle(indices)
        pool = indices[:max_docs_per_lang]

        base_id = len(docs)
        for local_i, ex_i in enumerate(pool):
            docs.append({
                "id": base_id + local_i,
                "str_id": f"{lang}_{ex_i}",
                "text": ds[ex_i]["whole_func_string"][:600].strip(),
                "domain": lang,
            })

        for local_i, ex_i in enumerate(pool[:max_queries_per_lang]):
            doc_str = (ds[ex_i].get("func_documentation_string") or "").strip()
            if not doc_str:
                continue
            qid = str(len(raw_queries))
            raw_queries.append({"id": qid, "text": doc_str[:MAX_CONTEXT_CHARS], "domain": lang})
            raw_qrels[qid] = base_id + local_i

    # SQL from b-mc2/sql-create-context
    sql_ds = load_dataset("b-mc2/sql-create-context", split="train")
    sql_indices = list(range(len(sql_ds)))
    random.shuffle(sql_indices)
    sql_pool = sql_indices[:max_docs_per_lang]

    base_id = len(docs)
    for local_i, ex_i in enumerate(sql_pool):
        docs.append({
            "id": base_id + local_i,
            "str_id": f"sql_{ex_i}",
            "text": sql_ds[ex_i]["answer"].strip(),
            "domain": "sql",
        })

    for local_i, ex_i in enumerate(sql_pool[:max_queries_per_lang]):
        question = (sql_ds[ex_i].get("question") or "").strip()
        if not question:
            continue
        qid = str(len(raw_queries))
        raw_queries.append({"id": qid, "text": question[:MAX_CONTEXT_CHARS], "domain": "sql"})
        raw_qrels[qid] = base_id + local_i

    random.shuffle(raw_queries)
    queries: List[dict] = []
    qrels: Dict[str, int] = {}
    for new_i, q in enumerate(raw_queries):
        new_qid = str(new_i)
        queries.append({"id": new_qid, "text": q["text"], "domain": q["domain"]})
        qrels[new_qid] = raw_qrels[q["id"]]

    return docs, queries, qrels


# ---------------------------------------------------------------------------
# LLM call functions
# ---------------------------------------------------------------------------

def call_ollama(system: str, user: str) -> Tuple[str, float, int, int]:
    """Call Ollama chat API via HTTP. Returns (response_text, latency_ms, input_tokens, output_tokens)."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"num_predict": MAX_LLM_TOKENS, "temperature": 0.0},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama not reachable at {OLLAMA_URL}.\n"
            "  Start it with: ollama serve\n"
            f"  Then pull the model: ollama pull {OLLAMA_MODEL}\n"
            f"  Original error: {exc}"
        ) from exc
    latency_ms = (time.perf_counter() - t0) * 1000
    input_tokens = result.get("prompt_eval_count", 0)
    output_tokens = result.get("eval_count", 0)
    return result["message"]["content"].strip(), latency_ms, input_tokens, output_tokens


def call_bedrock(system: str, user: str) -> Tuple[str, float, int, int]:
    """Call Claude via AWS Bedrock Converse API. Returns (response_text, latency_ms, input_tokens, output_tokens)."""
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 not installed: pip install boto3") from exc
    client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    t0 = time.perf_counter()
    try:
        response = client.converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": MAX_LLM_TOKENS, "temperature": 0.0},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Bedrock call failed (model={BEDROCK_MODEL_ID}, region={BEDROCK_REGION}).\n"
            "  Check AWS credentials and that the model is enabled in your region.\n"
            f"  Original error: {exc}"
        ) from exc
    latency_ms = (time.perf_counter() - t0) * 1000
    text = response["output"]["message"]["content"][0]["text"].strip()
    usage = response.get("usage", {})
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)
    return text, latency_ms, input_tokens, output_tokens


_ORDINAL_WORDS = {
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    '1st': 1, '2nd': 2, '3rd': 3, '4th': 4, '5th': 5,
}


def parse_selection(response: str, n_candidates: int) -> Optional[int]:
    """Extract 1-based candidate index from LLM response. None if unparseable.

    Strips <think>...</think> blocks (deepseek-r1 reasoning output) before
    parsing. Handles bare digits ("3"), ordinal abbreviations ("3rd"), and
    ordinal words ("third"). Returns whichever valid reference appears earliest
    in the post-think text.
    """
    # Strip reasoning chain emitted by R1-family models
    text = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    if not text:
        text = response  # fall back to full response if think block consumed everything

    best_pos: int = len(text) + 1
    best_k: Optional[int] = None

    lower = text.lower()
    for word, k in _ORDINAL_WORDS.items():
        if 1 <= k <= n_candidates:
            pos = lower.find(word)
            if pos != -1 and pos < best_pos:
                best_pos = pos
                best_k = k

    for m in re.finditer(r'\b([1-9]\d*)\b', text):
        k = int(m.group(1))
        if 1 <= k <= n_candidates and m.start() < best_pos:
            best_pos = m.start()
            best_k = k

    return best_k


def build_rag_prompt(context: str, candidates: List[str]) -> str:
    lines = [
        "Description:",
        context[:MAX_CONTEXT_CHARS],
        "",
        "Candidate code snippets:",
    ]
    for i, c in enumerate(candidates, 1):
        lines.append(f"{i}. {c[:MAX_HOLDING_CHARS]}")
    valid = ", ".join(str(i) for i in range(1, len(candidates) + 1))
    lines += [
        "",
        "Which numbered snippet best matches the description? Always pick the closest match.",
        f"Output ONLY the digit ({valid}). No words, no explanation.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

@dataclass
class LLMResult:
    system: str
    n_queries: int
    llm_accuracy: float               # LLM picked gold / all queries
    recall_in_pool: float             # gold was in retrieved pool
    llm_accuracy_given_recall: float  # LLM accuracy conditioned on gold in pool
    parse_failure_rate: float         # fraction of responses that couldn't be parsed
    p50_retrieve_ms: float
    p50_llm_ms: float
    p50_total_ms: float
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0   # non-zero only for Bedrock (API cost)


def run_experiment(
    system: str,
    docs: List[dict],
    queries: List[dict],
    qrels: Dict[str, int],
    query_embeddings: np.ndarray,
    retrieve_fn: Callable[[int, np.ndarray], List[Tuple[int, str]]],
    llm_fn: Callable[[str, str], Tuple[str, float, int, int]],
    system_prompt: str,
    bedrock_pricing: bool = False,
) -> LLMResult:
    id_to_text = {d["id"]: d["text"] for d in docs}

    correct, in_pool, conditioned_correct = 0, 0, 0
    parse_failures = 0
    retrieve_lats, llm_lats, total_lats = [], [], []
    total_input_tokens, total_output_tokens = 0, 0

    for qi, query in enumerate(queries):
        qid = query["id"]
        gold_id = qrels.get(qid)

        t_ret0 = time.perf_counter()
        pool = retrieve_fn(qi, query_embeddings[qi])  # [(doc_id, text), ...]
        t_ret1 = time.perf_counter()
        retrieve_lats.append((t_ret1 - t_ret0) * 1000)

        candidate_texts = [text for _, text in pool]
        candidate_ids = [did for did, _ in pool]

        gold_in_pool = gold_id in candidate_ids
        if gold_in_pool:
            in_pool += 1

        prompt = build_rag_prompt(query["text"], candidate_texts)
        response, llm_ms, in_tok, out_tok = llm_fn(system_prompt, prompt)
        llm_lats.append(llm_ms)
        total_lats.append((t_ret1 - t_ret0) * 1000 + llm_ms)
        total_input_tokens += in_tok
        total_output_tokens += out_tok

        selection = parse_selection(response, len(candidate_texts))
        if selection is None:
            parse_failures += 1
        else:
            picked_id = candidate_ids[selection - 1]
            if picked_id == gold_id:
                correct += 1
                if gold_in_pool:
                    conditioned_correct += 1
            elif gold_in_pool:
                pass  # gold was available but LLM didn't pick it

        if (qi + 1) % 25 == 0 or qi + 1 == len(queries):
            pct = 100.0 * (qi + 1) / len(queries)
            running_acc = correct / (qi + 1)
            print(f"   {qi + 1}/{len(queries)} ({pct:.0f}%)  acc={running_acc:.3f}  llm_p50={sorted(llm_lats)[len(llm_lats)//2]:.0f}ms")

    n = len(queries)

    def p50(vals: List[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        return s[len(s) // 2]

    estimated_cost = 0.0
    if bedrock_pricing:
        estimated_cost = (
            total_input_tokens * BEDROCK_INPUT_COST_PER_1M / 1_000_000
            + total_output_tokens * BEDROCK_OUTPUT_COST_PER_1M / 1_000_000
        )

    return LLMResult(
        system=system,
        n_queries=n,
        llm_accuracy=correct / n if n else 0.0,
        recall_in_pool=in_pool / n if n else 0.0,
        llm_accuracy_given_recall=conditioned_correct / in_pool if in_pool else 0.0,
        parse_failure_rate=parse_failures / n if n else 0.0,
        p50_retrieve_ms=p50(retrieve_lats),
        p50_llm_ms=p50(llm_lats),
        p50_total_ms=p50(total_lats),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        estimated_cost_usd=estimated_cost,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Soft-Gate benchmark (B20/B21)")
    parser.add_argument("--max-queries-per-lang", type=int, default=150,
                        help="Queries per language for a quick test (default: 150 → 450 total)")
    parser.add_argument("--skip-bedrock", action="store_true",
                        help="Skip B20 (Bedrock) and only run B21 (Ollama)")
    parser.add_argument("--skip-ollama", action="store_true",
                        help="Skip B21 (Ollama) and only run B20 (Bedrock)")
    args = parser.parse_args()
    if not args.skip_bedrock and not BEDROCK_MODEL_ID:
        parser.error("set BEDROCK_MODEL_ID to run B20, or pass --skip-bedrock")

    set_seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Java/Python/SQL datasets (max_queries_per_lang={args.max_queries_per_lang})...")
    docs, queries, qrels = load_code_dataset(
        max_docs_per_lang=1500,
        max_queries_per_lang=args.max_queries_per_lang,
    )
    print(f"  {len(docs)} docs, {len(queries)} queries")

    print(f"Embedding with {EMBED_MODEL}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    doc_vecs = embed_texts(model, [d["text"] for d in docs], prefix=PASSAGE_PREFIX)
    query_vecs = embed_texts(model, [q["text"] for q in queries], prefix=QUERY_PREFIX)
    id_to_text = {d["id"]: d["text"] for d in docs}

    # Build flat index
    print("Building flat HNSW index...")
    flat_idx = HNSWIndex(doc_vecs.shape[1])
    flat_idx.add([d["id"] for d in docs], doc_vecs)

    # B20 and B21 share this retriever on purpose: holding the candidate pool
    # fixed isolates the LLM choice from retrieval effects.
    def retrieve_flat(qi: int, qvec: np.ndarray) -> List[Tuple[int, str]]:
        ids, _ = flat_idx.search(qvec, RETRIEVE_POOL)
        return [(did, id_to_text[did]) for did in ids]

    results: List[LLMResult] = []

    if not args.skip_bedrock:
        print(f"\nRunning B20: flat retrieval + Claude Bedrock ({BEDROCK_MODEL_ID})...")
        print(f"  Region: {BEDROCK_REGION}")
        res_b20 = run_experiment(
            "B20 Flat + Claude Bedrock (generalist) [Java/Python/SQL]",
            docs, queries, qrels, query_vecs,
            retrieve_flat,
            call_bedrock,
            GENERALIST_SYSTEM_PROMPT,
            bedrock_pricing=True,
        )
        results.append(res_b20)
        print(f"  accuracy={res_b20.llm_accuracy:.3f}  recall_in_pool={res_b20.recall_in_pool:.3f}"
              f"  p50_llm={res_b20.p50_llm_ms:.0f}ms  parse_fail={res_b20.parse_failure_rate:.3f}"
              f"  cost=${res_b20.estimated_cost_usd:.4f}"
              f"  (${res_b20.estimated_cost_usd / res_b20.n_queries * 1000:.4f}/1k queries)")

    if not args.skip_ollama:
        print(f"\nRunning B21: flat retrieval + Ollama ({OLLAMA_MODEL}) local code model...")
        print("  (Ensure `ollama serve` is running and model is pulled)")
        res_b21 = run_experiment(
            f"B21 Flat + {OLLAMA_MODEL} (code fine-tune, local) [Java/Python/SQL]",
            docs, queries, qrels, query_vecs,
            retrieve_flat,
            call_ollama,
            CODE_SPECIALIST_SYSTEM_PROMPT,
            bedrock_pricing=False,
        )
        results.append(res_b21)
        print(f"  accuracy={res_b21.llm_accuracy:.3f}  recall_in_pool={res_b21.recall_in_pool:.3f}"
              f"  p50_llm={res_b21.p50_llm_ms:.0f}ms  parse_fail={res_b21.parse_failure_rate:.3f}"
              f"  cost=$0.00 (local inference)")

    if not results:
        print("No experiments ran. Use --skip-bedrock or --skip-ollama to run one side.")
        sys.exit(1)

    # Write results
    out_path = OUT_DIR / "llm-results.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"\nWrote {out_path}")

    # Also dump JSON for convenience
    json_path = OUT_DIR / "llm-results.json"
    json_path.write_text(
        json.dumps({"results": [asdict(r) for r in results], "config": {
            "seed": SEED,
            "embed_model": EMBED_MODEL,
            "ollama_model": OLLAMA_MODEL,
            "bedrock_model": BEDROCK_MODEL_ID,
            "retrieve_pool": RETRIEVE_POOL,
            "n_queries": len(queries),
        }}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {json_path}")

    print("\n--- Summary ---")
    for r in results:
        print(f"  {r.system}")
        print(f"    accuracy={r.llm_accuracy:.3f}  recall_in_pool={r.recall_in_pool:.3f}"
              f"  acc|recall={r.llm_accuracy_given_recall:.3f}")
        print(f"    p50_total={r.p50_total_ms:.0f}ms  (retrieve={r.p50_retrieve_ms:.1f}ms"
              f"  llm={r.p50_llm_ms:.0f}ms)  parse_fail={r.parse_failure_rate:.3f}")
        if r.estimated_cost_usd > 0:
            cost_per_1k = r.estimated_cost_usd / r.n_queries * 1000
            print(f"    tokens={r.total_input_tokens}in/{r.total_output_tokens}out"
                  f"  total_cost=${r.estimated_cost_usd:.4f}  (${cost_per_1k:.4f}/1k queries)")
        else:
            print(f"    tokens={r.total_input_tokens}in/{r.total_output_tokens}out"
                  f"  cost=$0.00 (local)")

    # Cost efficiency comparison
    bedrock_res = next((r for r in results if r.estimated_cost_usd > 0), None)
    ollama_res = next((r for r in results if r.estimated_cost_usd == 0.0
                       and r.total_input_tokens > 0), None)
    if bedrock_res and ollama_res:
        acc_delta = ollama_res.llm_accuracy - bedrock_res.llm_accuracy
        cost_per_1k_bedrock = bedrock_res.estimated_cost_usd / bedrock_res.n_queries * 1000
        print(f"\n--- Efficiency claim ---")
        print(f"  Accuracy delta (B21 - B20): {acc_delta:+.3f}  "
              f"({'specialist within ±1pp' if abs(acc_delta) < 0.01 else 'specialist wins' if acc_delta > 0 else 'generalist wins'})")
        print(f"  B20 cost per 1k queries:  ${cost_per_1k_bedrock:.4f}")
        print(f"  B21 cost per 1k queries:  $0.00 (local {OLLAMA_MODEL})")
        print(f"  Input tokens B20/B21:     {bedrock_res.total_input_tokens}/{ollama_res.total_input_tokens}")


if __name__ == "__main__":
    main()
