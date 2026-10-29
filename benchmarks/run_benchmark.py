#!/usr/bin/env python3
"""
Soft-Gate benchmark harness (paper protocol, Epic 6).

Compares:
  B0 Qdrant-style flat HNSW
  B1 Soft-Gate flat (single expert)
  B2 Soft-Gate routed (ingest + query routing)
  B3 Query-only routing
  B4 Random expert assignment
  B5 Soft-Gate centroid routed (data-driven profiles + one reassignment pass)
  B6 Soft-Gate centroid + RRF fusion (gates for diagnostics only)
  B7 Soft-Gate centroid + gate-weighted RRF fusion
  B8 Soft-Gate centroid, top-1 early-exit (search only the gate's top expert)
  B9 Soft-Gate centroid, top-2 early-exit (search the gate's top-2 experts)
  B10 Soft-Gate mixed-membership ingest (dual-write docs with g_i > tau)
  B11 BM25 lexical baseline
  B12 Hybrid BM25 + dense fusion on the flat (unrouted) index
  B13 Unweighted multi-index merge (centroid experts, no gate weighting)
  B14 Learned router: oracle ingest + logistic-regression query gate
  B15 Learned router, top-1 early-exit
  B16 Learned router, top-2 early-exit
  B17 Flat retrieval + cross-encoder rerank
  B18 Centroid top-2 routed retrieval + cross-encoder rerank
  B19 Real Qdrant flat baseline (HNSW on a Qdrant server when QDRANT_URL is set,
      otherwise qdrant-client local mode, which is exact brute-force search)
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import shutil
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import hnswlib
import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CollectionStatus,
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    SearchParams,
    VectorParams,
)
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "benchmarks" / "data"
OUT_DIR = ROOT / "tables"
SEED = 42

DOMAINS = {
    "finance": {
        "source": "beir",
        "name": "fiqa",
        "profile": (
            "Financial markets, banking, investments, corporate finance, "
            "and economic analysis."
        ),
    },
    "legal": {
        # CaseHOLD (LexGLUE): US case-law holding identification, converted to IR.
        "source": "casehold",
        "name": "coastalcph/lex_glue:case_hold",
        "profile": (
            "United States case law, judicial holdings, legal precedents, "
            "statutory interpretation, and court opinions."
        ),
    },
    "tech": {
        "source": "beir",
        "name": "nfcorpus",
        "profile": (
            "Technical documentation, software troubleshooting, IT support, "
            "and product manuals."
        ),
    },
}

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128
TOP_K = 10
RRF_K = 60
RRF_POOL = 50
MIXED_MEMBERSHIP_DUAL_FRACTION = 0.20
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_POOL = 30
RERANK_MAX_CHARS = 800
RERANK_BATCH_SIZE = 256
MAX_DOCS_PER_DOMAIN = 4500
MAX_QUERIES_PER_DOMAIN = 450
EMBED_MODEL = "intfloat/e5-base-v2"
# e5 requires asymmetric prefixes; profiles are treated as passages (domain descriptions).
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


@dataclass
class RunResult:
    system: str
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    routing_top1_acc: float
    gate_entropy: float
    p50_e2e_ms: float
    p99_e2e_ms: float
    p50_routing_ms: float
    p50_retrieval_ms: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def softmax_neg_l2(query: np.ndarray, profiles: np.ndarray) -> np.ndarray:
    dists = np.linalg.norm(profiles - query, axis=1)
    logits = -dists
    logits -= logits.max()
    exps = np.exp(logits)
    return exps / exps.sum()


def entropy(gates: np.ndarray) -> float:
    g = gates[gates > 0]
    return float(-(g * np.log(g)).sum())


def log_progress(i: int, total: int, every: int = 100, prefix: str = "   ") -> None:
    """Periodic progress line for long per-query loops, so a live run shows
    movement instead of going silent for minutes at a time."""
    if total and ((i + 1) % every == 0 or i + 1 == total):
        pct = 100.0 * (i + 1) / total
        print(f"{prefix}{i + 1}/{total} ({pct:.0f}%)")


def assign_by_profiles(doc_embeddings: np.ndarray, profiles: np.ndarray) -> List[str]:
    domain_keys = list(DOMAINS.keys())
    assignment: List[str] = []
    for i in range(doc_embeddings.shape[0]):
        g = softmax_neg_l2(doc_embeddings[i], profiles)
        assignment.append(domain_keys[int(np.argmax(g))])
    return assignment


def centroid_profiles_from_assignment(doc_embeddings: np.ndarray, assignment: List[str]) -> np.ndarray:
    """p_i = mean(embeddings currently in expert i), re-normalized to unit length
    so it stays on the same scale as the (normalized) doc/query embeddings for
    the L2-distance gate."""
    domain_keys = list(DOMAINS.keys())
    profiles = np.zeros((len(domain_keys), doc_embeddings.shape[1]))
    for di, domain in enumerate(domain_keys):
        idxs = [i for i, a in enumerate(assignment) if a == domain]
        if not idxs:
            raise ValueError(f"No docs assigned to expert '{domain}' during centroid computation")
        mean = doc_embeddings[idxs].mean(axis=0)
        norm = np.linalg.norm(mean)
        profiles[di] = mean / norm if norm > 0 else mean
    return profiles


def all_gates(doc_embeddings: np.ndarray, profiles: np.ndarray) -> np.ndarray:
    """Gate vector per doc, shape (n_docs, n_experts)."""
    return np.stack([softmax_neg_l2(doc_embeddings[i], profiles) for i in range(doc_embeddings.shape[0])])


def mixed_membership_tau_for_fraction(gates: np.ndarray, dual_write_fraction: float) -> float:
    """Pick tau so that roughly `dual_write_fraction` of docs (the most ambiguous
    ones, by runner-up gate) qualify for a second expert. A fixed absolute tau
    doesn't work here: this corpus's gate values cluster tightly near 1/3 for
    almost every doc (checked empirically — 0.15/0.25/0.30 all caught ~100% of
    docs, 0.35 caught 0%), so the threshold has to be derived from the actual
    per-run distribution instead of hand-picked."""
    runner_up = np.sort(gates, axis=1)[:, -2]
    return float(np.percentile(runner_up, 100 * (1 - dual_write_fraction)))


def mixed_membership_domains(gates: np.ndarray, tau: float) -> List[List[str]]:
    """Dual-write assignment: each doc goes into every expert whose gate exceeds
    tau, not just its argmax. Falls back to the single argmax expert when no
    gate clears tau, so every doc is always retrievable from at least one index."""
    domain_keys = list(DOMAINS.keys())
    assignment: List[List[str]] = []
    for row in gates:
        members = [domain_keys[di] for di in range(len(domain_keys)) if row[di] > tau]
        if not members:
            members = [domain_keys[int(np.argmax(row))]]
        assignment.append(members)
    return assignment


def train_learned_router(doc_embeddings: np.ndarray, doc_domains: List[str]) -> LogisticRegression:
    """Lightweight learned router: multinomial logistic regression on doc
    embeddings -> true domain label. Supervised on the domain each doc actually
    came from (realistic at ingest time — you typically know which source
    system/corpus a document came from), unlike the unsupervised centroid gate."""
    clf = LogisticRegression(max_iter=2000)
    clf.fit(doc_embeddings, doc_domains)
    return clf


def make_learned_gate_fn(clf: LogisticRegression) -> Callable[[np.ndarray], np.ndarray]:
    """Wraps a fitted classifier as a gate_fn(query_vec) -> probabilities, in the
    same domain_keys order used everywhere else (sklearn's clf.classes_ order is
    alphabetical, not necessarily DOMAINS' insertion order)."""
    domain_keys = list(DOMAINS.keys())
    class_to_col = {c: i for i, c in enumerate(clf.classes_)}
    col_order = [class_to_col[d] for d in domain_keys]

    def gate_fn(qvec: np.ndarray) -> np.ndarray:
        proba = clf.predict_proba(qvec.reshape(1, -1))[0]
        return proba[col_order]

    return gate_fn


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int = 10) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(ranked[:k]):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 2)
    if not relevant:
        return 0.0
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg else 0.0


def mrr_at_k(ranked: Sequence[int], relevant: set[int], k: int = 10) -> float:
    for i, doc_id in enumerate(ranked[:k]):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int = 10) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / len(relevant)


class HNSWIndex:
    def __init__(self, dim: int):
        self.dim = dim
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
        out_ids = [self.ids[i] for i in labels[0]]
        # hnswlib cosine distance -> similarity
        scores = [1.0 - d for d in distances[0]]
        return out_ids, scores


def download_beir(beir_name: str) -> Tuple[Dict[str, dict], Dict[str, str], Dict[str, Dict[str, int]]]:
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{beir_name}.zip"
    path = util.download_and_unzip(url, str(DATA_DIR))
    corpus, queries, qrels = GenericDataLoader(data_folder=path).load(split="test")
    return corpus, queries, qrels


def load_casehold_ir(
    max_docs: int = MAX_DOCS_PER_DOMAIN,
    max_queries: int = MAX_QUERIES_PER_DOMAIN,
) -> Tuple[Dict[str, dict], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Convert LexGLUE CaseHOLD (multiple-choice holdings) into a BEIR-style IR set.

    Each example has a citing-case context and five candidate holdings. We treat
    holdings as documents and contexts as queries; the labeled holding is relevant.
    The document pool is built from a larger CaseHOLD subset so retrieval is not
    limited to the five candidates of a single query.
    """
    from datasets import load_dataset

    ds = load_dataset("coastalcph/lex_glue", "case_hold", split="test")
    indices = list(range(len(ds)))
    random.shuffle(indices)

    # 5 holdings per example → need max_docs/5 examples in the pool.
    n_pool = min(len(indices), max(max_docs // 5, max_queries))
    pool = indices[:n_pool]
    query_idxs = pool[: min(max_queries, n_pool)]

    corpus: Dict[str, dict] = {}
    for ex_i in pool:
        for end_i, ending in enumerate(ds[ex_i]["endings"]):
            if len(corpus) >= max_docs:
                break
            doc_id = f"ch_{ex_i}_{end_i}"
            corpus[doc_id] = {"title": "", "text": str(ending).strip()}
        if len(corpus) >= max_docs:
            break

    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}
    for ex_i in query_idxs:
        qid = f"cq_{ex_i}"
        label = int(ds[ex_i]["label"])
        gold_id = f"ch_{ex_i}_{label}"
        if gold_id not in corpus:
            continue
        queries[qid] = str(ds[ex_i]["context"]).strip()
        qrels[qid] = {gold_id: 1}

    return corpus, queries, qrels


def load_domain(cfg: dict) -> Tuple[Dict[str, dict], Dict[str, str], Dict[str, Dict[str, int]]]:
    source = cfg.get("source", "beir")
    if source == "beir":
        return download_beir(cfg["name"])
    if source == "casehold":
        return load_casehold_ir()
    raise ValueError(f"Unknown domain source: {source}")


def load_corpus() -> Tuple[List[dict], List[dict], Dict[str, Dict[str, int]]]:
    docs: List[dict] = []
    queries: List[dict] = []
    qrels: Dict[str, Dict[str, int]] = {}

    for domain, cfg in DOMAINS.items():
        corpus, domain_queries, domain_qrels = load_domain(cfg)
        judged_ids = {doc_id for rels in domain_qrels.values() for doc_id in rels}
        judged_items = [(did, meta) for did, meta in corpus.items() if did in judged_ids]
        other_items = [(did, meta) for did, meta in corpus.items() if did not in judged_ids]
        random.shuffle(other_items)
        # Keep all judged docs first so capped samples retain qrels coverage.
        doc_items = judged_items + other_items
        doc_items = doc_items[:MAX_DOCS_PER_DOMAIN]
        base = len(docs)
        for offset, (doc_id, meta) in enumerate(doc_items):
            docs.append(
                {
                    "id": base + offset,
                    "text": (meta.get("title", "") + " " + meta.get("text", "")).strip(),
                    "domain": domain,
                }
            )
        id_map = {old: base + i for i, (old, _) in enumerate(doc_items)}

        q_items = list(domain_queries.items())
        random.shuffle(q_items)
        q_items = q_items[:MAX_QUERIES_PER_DOMAIN]
        q_base = len(queries)
        for offset, (qid, qtext) in enumerate(q_items):
            text = qtext if isinstance(qtext, str) else str(qtext.get("text", qtext))
            queries.append({"id": q_base + offset, "text": text, "domain": domain})
            mapped = {id_map[d]: r for d, r in domain_qrels.get(qid, {}).items() if d in id_map}
            if mapped:
                qrels[str(q_base + offset)] = mapped

    return docs, queries, qrels


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    *,
    prefix: str = "",
    batch_size: int = 64,
) -> np.ndarray:
    prefixed = [prefix + t for t in texts] if prefix else texts
    return np.asarray(
        model.encode(prefixed, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    )


def build_indices(
    doc_embeddings: np.ndarray,
    docs: List[dict],
    placement: str,
    gates_at_ingest: np.ndarray | None,
    profiles: np.ndarray,
    fixed_assignment: List[str] | None = None,
    mixed_assignment: List[List[str]] | None = None,
) -> Dict[str, HNSWIndex] | HNSWIndex:
    dim = doc_embeddings.shape[1]
    if placement == "flat":
        idx = HNSWIndex(dim)
        idx.add([d["id"] for d in docs], doc_embeddings)
        return idx

    per_domain: Dict[str, HNSWIndex] = {d: HNSWIndex(dim) for d in DOMAINS}

    if placement == "mixed":
        # Dual-write: a doc can land in more than one expert's index.
        for i, doc in enumerate(docs):
            for domain in mixed_assignment[i]:
                per_domain[domain].add([doc["id"]], doc_embeddings[i : i + 1])
        return per_domain

    for i, doc in enumerate(docs):
        if placement == "oracle":
            domain = doc["domain"]
        elif placement == "profile_ingest":
            g = softmax_neg_l2(doc_embeddings[i], profiles)
            domain = list(DOMAINS.keys())[int(np.argmax(g))]
        elif placement == "random":
            domain = random.choice(list(DOMAINS.keys()))
        elif placement == "fixed":
            domain = fixed_assignment[i]
        else:
            raise ValueError(placement)
        per_domain[domain].add([doc["id"]], doc_embeddings[i : i + 1])
    return per_domain


def evaluate(
    system: str,
    docs: List[dict],
    queries: List[dict],
    qrels: Dict[str, Dict[str, int]],
    query_embeddings: np.ndarray,
    profiles: np.ndarray,
    placement: str,
    route_at_query: bool,
    use_gates_in_merge: bool,
    random_route: bool = False,
    fixed_assignment: List[str] | None = None,
    mixed_assignment: List[List[str]] | None = None,
    merge_mode: str = "gate",
    top_experts: int | None = None,
    gate_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Tuple[RunResult, np.ndarray, np.ndarray]:
    doc_embeddings = embed_texts_cached["docs"]
    indices = build_indices(doc_embeddings, docs, placement, None, profiles, fixed_assignment, mixed_assignment)

    recalls, mrrs, ndcgs = [], [], []
    route_acc, entropies = [], []
    e2e, routing_lat, retrieval_lat = [], [], []

    domain_keys = list(DOMAINS.keys())

    for qi, query in enumerate(queries):
        qvec = query_embeddings[qi]
        qid = str(query["id"])
        relevant = {int(d) for d in qrels.get(qid, {})}

        t0 = time.perf_counter()
        if random_route:
            gates = np.zeros(len(domain_keys))
            gates[random.randrange(len(domain_keys))] = 1.0
        elif route_at_query:
            gates = gate_fn(qvec) if gate_fn is not None else softmax_neg_l2(qvec, profiles)
        else:
            gates = np.ones(len(domain_keys)) / len(domain_keys)
        t1 = time.perf_counter()

        # Early exit: only search the experts the gate ranks highest, instead of
        # every expert. Cuts retrieval latency roughly proportionally and drops
        # any expert whose gate mass the router considers negligible.
        if top_experts is not None:
            order = list(np.argsort(gates)[::-1][:top_experts])
        else:
            order = list(range(len(domain_keys)))

        candidates: List[Tuple[int, float, str]] = []
        if isinstance(indices, HNSWIndex):
            ids, scores = indices.search(qvec, TOP_K)
            for doc_id, score in zip(ids, scores):
                candidates.append((doc_id, score, "flat"))
        elif merge_mode == "rrf":
            # Rank-based fusion (RRF): scale-invariant across experts, so gates
            # are used only for routing diagnostics above, not to weight the merge.
            for di in order:
                domain = domain_keys[di]
                ids, _ = indices[domain].search(qvec, RRF_POOL)
                for rank, doc_id in enumerate(ids):
                    rrf_score = 1.0 / (RRF_K + rank + 1)
                    candidates.append((doc_id, rrf_score, domain))
        elif merge_mode == "weighted_rrf":
            # RRF rank score scaled by the routing gate, so a confidently-routed
            # expert's candidates outrank equally-ranked candidates from experts
            # the gate considers unlikely.
            for di in order:
                domain = domain_keys[di]
                ids, _ = indices[domain].search(qvec, RRF_POOL)
                for rank, doc_id in enumerate(ids):
                    rrf_score = gates[di] / (RRF_K + rank + 1)
                    candidates.append((doc_id, rrf_score, domain))
        else:
            for di in order:
                domain = domain_keys[di]
                ids, scores = indices[domain].search(qvec, TOP_K)
                for doc_id, score in zip(ids, scores):
                    merged = score * gates[di] if use_gates_in_merge else score
                    candidates.append((doc_id, merged, domain))

        t2 = time.perf_counter()
        candidates.sort(key=lambda x: x[1], reverse=True)
        ranked = []
        seen = set()
        for doc_id, _, _ in candidates:
            if doc_id not in seen:
                ranked.append(doc_id)
                seen.add(doc_id)
            if len(ranked) >= TOP_K:
                break

        recalls.append(recall_at_k(ranked, relevant, TOP_K))
        mrrs.append(mrr_at_k(ranked, relevant, TOP_K))
        ndcgs.append(ndcg_at_k(ranked, relevant, TOP_K))

        if route_at_query and not random_route:
            pred = domain_keys[int(np.argmax(gates))]
            route_acc.append(1.0 if pred == query["domain"] else 0.0)
            entropies.append(entropy(gates))

        e2e.append((t2 - t0) * 1000)
        routing_lat.append((t1 - t0) * 1000)
        retrieval_lat.append((t2 - t1) * 1000)
        log_progress(qi, len(queries))

    def pct(vals: List[float], p: float) -> float:
        if not vals:
            return 0.0
        vals = sorted(vals)
        idx = min(int(math.ceil(p / 100 * len(vals))) - 1, len(vals) - 1)
        return vals[idx]

    result = RunResult(
        system=system,
        recall_at_10=statistics.mean(recalls) if recalls else 0.0,
        mrr_at_10=statistics.mean(mrrs) if mrrs else 0.0,
        ndcg_at_10=statistics.mean(ndcgs) if ndcgs else 0.0,
        routing_top1_acc=statistics.mean(route_acc) if route_acc else 0.0,
        gate_entropy=statistics.mean(entropies) if entropies else 0.0,
        p50_e2e_ms=pct(e2e, 50),
        p99_e2e_ms=pct(e2e, 99),
        p50_routing_ms=pct(routing_lat, 50),
        p50_retrieval_ms=pct(retrieval_lat, 50),
    )
    # Per-query arrays, kept alongside the aggregate result so callers can run
    # bootstrap CIs / paired significance tests without re-evaluating.
    return result, np.asarray(ndcgs), np.asarray(recalls)


N_BOOT = 5000


def bootstrap_mean_ci(
    values: np.ndarray, idx: np.ndarray
) -> Tuple[float, float, float]:
    """Percentile-bootstrap 95% CI on the mean of `values`, reusing a shared
    resample index matrix `idx` of shape (n_boot, n)."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, idx: np.ndarray
) -> Tuple[float, float, float, float]:
    """Paired percentile-bootstrap test of mean(a) - mean(b) != 0, reusing a
    shared resample index matrix `idx`. Returns (diff, ci_lo, ci_hi, two_sided_p)."""
    diff = a - b
    boot_diff = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot_diff, [2.5, 97.5])
    p = float(min(2 * min((boot_diff <= 0).mean(), (boot_diff >= 0).mean()), 1.0))
    return float(diff.mean()), float(lo), float(hi), p


def bm25_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def evaluate_bm25(
    system: str,
    docs: List[dict],
    queries: List[dict],
    qrels: Dict[str, Dict[str, int]],
    bm25: BM25Okapi,
    dense_index: HNSWIndex | None = None,
    dense_query_embeddings: np.ndarray | None = None,
    hybrid: bool = False,
) -> Tuple[RunResult, np.ndarray, np.ndarray]:
    """BM25-only, or BM25 fused with a dense flat index via RRF. Not part of the
    routed/expert machinery at all — a lexical (and lexical+dense) baseline on
    the plain unrouted corpus, to check whether the dense-only baselines above
    are even a strong reference point."""
    doc_ids = [d["id"] for d in docs]
    recalls, mrrs, ndcgs = [], [], []
    e2e, routing_lat, retrieval_lat = [], [], []

    for qi, query in enumerate(queries):
        qid = str(query["id"])
        relevant = {int(d) for d in qrels.get(qid, {})}

        t0 = time.perf_counter()
        bm25_scores = bm25.get_scores(bm25_tokenize(query["text"]))
        t1 = time.perf_counter()

        if not hybrid:
            top_idx = np.argsort(bm25_scores)[::-1][:TOP_K]
            ranked = [doc_ids[i] for i in top_idx]
        else:
            fused: Dict[int, float] = {}
            bm25_order = np.argsort(bm25_scores)[::-1][:RRF_POOL]
            for rank, i in enumerate(bm25_order):
                did = doc_ids[i]
                fused[did] = fused.get(did, 0.0) + 1.0 / (RRF_K + rank + 1)
            dense_ids, _ = dense_index.search(dense_query_embeddings[qi], RRF_POOL)
            for rank, did in enumerate(dense_ids):
                fused[did] = fused.get(did, 0.0) + 1.0 / (RRF_K + rank + 1)
            ranked = [did for did, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]]
        t2 = time.perf_counter()

        recalls.append(recall_at_k(ranked, relevant, TOP_K))
        mrrs.append(mrr_at_k(ranked, relevant, TOP_K))
        ndcgs.append(ndcg_at_k(ranked, relevant, TOP_K))

        e2e.append((t2 - t0) * 1000)
        routing_lat.append((t1 - t0) * 1000)
        retrieval_lat.append((t2 - t1) * 1000)
        log_progress(qi, len(queries))

    def pct(vals: List[float], p: float) -> float:
        if not vals:
            return 0.0
        vals = sorted(vals)
        idx = min(int(math.ceil(p / 100 * len(vals))) - 1, len(vals) - 1)
        return vals[idx]

    result = RunResult(
        system=system,
        recall_at_10=statistics.mean(recalls) if recalls else 0.0,
        mrr_at_10=statistics.mean(mrrs) if mrrs else 0.0,
        ndcg_at_10=statistics.mean(ndcgs) if ndcgs else 0.0,
        routing_top1_acc=0.0,
        gate_entropy=0.0,
        p50_e2e_ms=pct(e2e, 50),
        p99_e2e_ms=pct(e2e, 99),
        p50_routing_ms=pct(routing_lat, 50),
        p50_retrieval_ms=pct(retrieval_lat, 50),
    )
    return result, np.asarray(ndcgs), np.asarray(recalls)


def evaluate_rerank(
    system: str,
    docs: List[dict],
    queries: List[dict],
    qrels: Dict[str, Dict[str, int]],
    cross_encoder: CrossEncoder,
    retrieve_fn: Callable[[int, np.ndarray], List[int]],
    query_embeddings: np.ndarray,
) -> Tuple[RunResult, np.ndarray, np.ndarray]:
    """Two-stage: `retrieve_fn` returns an already-ranked pool of up to
    RERANK_POOL candidate doc ids for a query; the cross-encoder rescoores
    (query, doc) pairs and the final top-TOP_K comes from that rescoring, not
    the first-stage order.

    Cross-encoder inference is batched across ALL queries in one predict()
    call instead of one call per query — with 1000+ queries, per-call overhead
    (not the actual model compute) was the dominant cost. Rerank latency is
    therefore measured once in aggregate and divided evenly across queries
    (amortized), not measured per query; only the first-stage retrieval time
    is genuinely per-query. Document text is capped at RERANK_MAX_CHARS before
    pairing, since a cross-encoder only needs enough text to judge relevance,
    not the full passage."""
    id_to_text = {d["id"]: d["text"][:RERANK_MAX_CHARS] for d in docs}

    retrieval_lat: List[float] = []
    per_query_candidates: List[List[int]] = []
    flat_pairs: List[Tuple[str, str]] = []
    pair_owner: List[int] = []

    for qi, query in enumerate(queries):
        qvec = query_embeddings[qi]
        t0 = time.perf_counter()
        candidate_ids = retrieve_fn(qi, qvec)
        t1 = time.perf_counter()
        retrieval_lat.append((t1 - t0) * 1000)
        per_query_candidates.append(candidate_ids)
        for did in candidate_ids:
            flat_pairs.append((query["text"], id_to_text[did]))
            pair_owner.append(qi)
        log_progress(qi, len(queries), prefix="    retrieval pool ")

    print(f"    reranking {len(flat_pairs)} (query, doc) pairs in batches of {RERANK_BATCH_SIZE}...")
    t2 = time.perf_counter()
    all_scores = (
        cross_encoder.predict(flat_pairs, batch_size=RERANK_BATCH_SIZE, show_progress_bar=True)
        if flat_pairs
        else np.array([])
    )
    t3 = time.perf_counter()
    rerank_ms_per_query = ((t3 - t2) * 1000 / len(queries)) if queries else 0.0
    print(f"    reranking took {t3 - t2:.1f}s total ({rerank_ms_per_query:.1f}ms/query amortized)")

    scores_by_query: List[List[float]] = [[] for _ in queries]
    for owner, score in zip(pair_owner, all_scores):
        scores_by_query[owner].append(score)

    recalls, mrrs, ndcgs, e2e = [], [], [], []
    for qi, query in enumerate(queries):
        qid = str(query["id"])
        relevant = {int(d) for d in qrels.get(qid, {})}
        candidate_ids = per_query_candidates[qi]
        scores = scores_by_query[qi]
        if candidate_ids:
            order = np.argsort(scores)[::-1]
            ranked = [candidate_ids[i] for i in order][:TOP_K]
        else:
            ranked = []

        recalls.append(recall_at_k(ranked, relevant, TOP_K))
        mrrs.append(mrr_at_k(ranked, relevant, TOP_K))
        ndcgs.append(ndcg_at_k(ranked, relevant, TOP_K))
        e2e.append(retrieval_lat[qi] + rerank_ms_per_query)

    def pct(vals: List[float], p: float) -> float:
        if not vals:
            return 0.0
        vals = sorted(vals)
        idx = min(int(math.ceil(p / 100 * len(vals))) - 1, len(vals) - 1)
        return vals[idx]

    result = RunResult(
        system=system,
        recall_at_10=statistics.mean(recalls) if recalls else 0.0,
        mrr_at_10=statistics.mean(mrrs) if mrrs else 0.0,
        ndcg_at_10=statistics.mean(ndcgs) if ndcgs else 0.0,
        routing_top1_acc=0.0,
        gate_entropy=0.0,
        p50_e2e_ms=pct(e2e, 50),
        p99_e2e_ms=pct(e2e, 99),
        p50_routing_ms=pct(retrieval_lat, 50),
        p50_retrieval_ms=rerank_ms_per_query,
    )
    return result, np.asarray(ndcgs), np.asarray(recalls)


QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_LOCAL_PATH = ROOT / "benchmarks" / "qdrant_local"
QDRANT_COLLECTION = "softgate_flat"
QDRANT_UPSERT_BATCH = 1000
QDRANT_INDEX_TIMEOUT_S = 600


def qdrant_system_name() -> str:
    if QDRANT_URL:
        return "B19 Real Qdrant (server, HNSW) flat baseline"
    return "B19 Real Qdrant (local mode, exact search) flat baseline"


def wait_for_qdrant_index(client: QdrantClient, n_points: int) -> None:
    """Block until the server has built the HNSW graph over every point, so
    queries hit the index rather than falling back to a full scan."""
    deadline = time.monotonic() + QDRANT_INDEX_TIMEOUT_S
    while True:
        info = client.get_collection(QDRANT_COLLECTION)
        indexed = info.indexed_vectors_count or 0
        if info.status == CollectionStatus.GREEN and indexed >= n_points:
            print(f"    HNSW index ready ({indexed}/{n_points} vectors indexed)")
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Qdrant indexed only {indexed}/{n_points} vectors after {QDRANT_INDEX_TIMEOUT_S}s"
            )
        time.sleep(1.0)


def evaluate_qdrant(
    system: str,
    docs: List[dict],
    queries: List[dict],
    qrels: Dict[str, Dict[str, int]],
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
) -> Tuple[RunResult, np.ndarray, np.ndarray]:
    """B0's flat baseline, but against a real Qdrant collection instead of the
    hnswlib index B0 uses.

    With QDRANT_URL set (e.g. a `docker run -p 6333:6333 qdrant/qdrant` server),
    this is Qdrant's own HNSW engine with M/ef_construct/ef_search matched to
    B0. Indexing and full-scan thresholds are forced down so even this small
    collection is served from the HNSW graph, and we wait for indexing to finish
    before querying.

    Without QDRANT_URL, qdrant-client's local mode is used. Local mode does not
    build HNSW at all — it is exact brute-force search in Python — so in that
    case B19 is an exact-search reference, not an HNSW comparison."""
    if QDRANT_URL:
        client = QdrantClient(url=QDRANT_URL)
        if client.collection_exists(QDRANT_COLLECTION):
            client.delete_collection(QDRANT_COLLECTION)
    else:
        if QDRANT_LOCAL_PATH.exists():
            shutil.rmtree(QDRANT_LOCAL_PATH)
        QDRANT_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(QDRANT_LOCAL_PATH))
    try:
        dim = doc_embeddings.shape[1]
        client.create_collection(
            QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCTION, full_scan_threshold=1),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=1),
        )
        n_batches = math.ceil(len(docs) / QDRANT_UPSERT_BATCH)
        for bi, start in enumerate(range(0, len(docs), QDRANT_UPSERT_BATCH)):
            chunk = docs[start : start + QDRANT_UPSERT_BATCH]
            client.upsert(
                QDRANT_COLLECTION,
                points=[
                    PointStruct(id=d["id"], vector=doc_embeddings[start + i].tolist())
                    for i, d in enumerate(chunk)
                ],
            )
            print(f"    upserted batch {bi + 1}/{n_batches} ({start + len(chunk)}/{len(docs)} docs)")
        if QDRANT_URL:
            wait_for_qdrant_index(client, len(docs))

        search_params = SearchParams(hnsw_ef=HNSW_EF_SEARCH)
        recalls, mrrs, ndcgs = [], [], []
        e2e, routing_lat, retrieval_lat = [], [], []

        for qi, query in enumerate(queries):
            qid = str(query["id"])
            relevant = {int(d) for d in qrels.get(qid, {})}

            t0 = time.perf_counter()
            res = client.query_points(
                QDRANT_COLLECTION,
                query=query_embeddings[qi].tolist(),
                limit=TOP_K,
                search_params=search_params,
            )
            t1 = time.perf_counter()
            ranked = [p.id for p in res.points]

            recalls.append(recall_at_k(ranked, relevant, TOP_K))
            mrrs.append(mrr_at_k(ranked, relevant, TOP_K))
            ndcgs.append(ndcg_at_k(ranked, relevant, TOP_K))

            e2e.append((t1 - t0) * 1000)
            routing_lat.append(0.0)
            retrieval_lat.append((t1 - t0) * 1000)
            log_progress(qi, len(queries))
    finally:
        if QDRANT_URL:
            client.delete_collection(QDRANT_COLLECTION)
        client.close()
        if not QDRANT_URL:
            shutil.rmtree(QDRANT_LOCAL_PATH, ignore_errors=True)

    def pct(vals: List[float], p: float) -> float:
        if not vals:
            return 0.0
        vals = sorted(vals)
        idx = min(int(math.ceil(p / 100 * len(vals))) - 1, len(vals) - 1)
        return vals[idx]

    result = RunResult(
        system=system,
        recall_at_10=statistics.mean(recalls) if recalls else 0.0,
        mrr_at_10=statistics.mean(mrrs) if mrrs else 0.0,
        ndcg_at_10=statistics.mean(ndcgs) if ndcgs else 0.0,
        routing_top1_acc=0.0,
        gate_entropy=0.0,
        p50_e2e_ms=pct(e2e, 50),
        p99_e2e_ms=pct(e2e, 99),
        p50_routing_ms=pct(routing_lat, 50),
        p50_retrieval_ms=pct(retrieval_lat, 50),
    )
    return result, np.asarray(ndcgs), np.asarray(recalls)


embed_texts_cached: Dict[str, np.ndarray] = {}


def main() -> None:
    set_seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading BEIR subsets...")
    docs, queries, qrels = load_corpus()
    print(f"Corpus: {len(docs)} docs, {len(queries)} queries")

    model = SentenceTransformer(EMBED_MODEL)
    doc_texts = [d["text"] for d in docs]
    query_texts = [q["text"] for q in queries]
    profile_texts = [cfg["profile"] for cfg in DOMAINS.values()]

    print(f"Embedding with {EMBED_MODEL}...")
    embed_texts_cached["docs"] = embed_texts(model, doc_texts, prefix=PASSAGE_PREFIX)
    query_embeddings = embed_texts(model, query_texts, prefix=QUERY_PREFIX)
    profiles = embed_texts(model, profile_texts, prefix=PASSAGE_PREFIX)

    configs = [
        ("B0 Qdrant flat HNSW", "flat", False, False, False),
        ("B1 Soft-Gate flat", "flat", False, False, False),
        ("B2 Soft-Gate routed (ingest+query)", "profile_ingest", True, True, False),
        ("B3 Query-only routing", "oracle", True, True, False),
        ("B4 Random routing", "random", False, True, True),
    ]

    results: List[RunResult] = []
    raw_metrics: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, placement, route_q, merge_g, rand in configs:
        print(f"Running {name}...")
        res, ndcg_arr, recall_arr = evaluate(
            name, docs, queries, qrels, query_embeddings, profiles, placement, route_q, merge_g, rand
        )
        results.append(res)
        raw_metrics[name] = (ndcg_arr, recall_arr)
        print(f"  R@10={res.recall_at_10:.3f} nDCG@10={res.ndcg_at_10:.3f} p50={res.p50_e2e_ms:.2f}ms")

    print("Computing centroid profiles (ingest + one reassignment pass)...")
    initial_assignment = assign_by_profiles(embed_texts_cached["docs"], profiles)
    centroid_profiles = centroid_profiles_from_assignment(embed_texts_cached["docs"], initial_assignment)
    final_assignment = assign_by_profiles(embed_texts_cached["docs"], centroid_profiles)
    moved = sum(1 for a, b in zip(initial_assignment, final_assignment) if a != b)
    print(f"  Reassignment pass moved {moved}/{len(final_assignment)} docs")

    print("Running B5 Soft-Gate centroid routed (ingest+query)...")
    res_b5, ndcg_b5, recall_b5 = evaluate(
        "B5 Soft-Gate centroid routed (ingest+query)",
        docs,
        queries,
        qrels,
        query_embeddings,
        centroid_profiles,
        "fixed",
        True,
        True,
        False,
        fixed_assignment=final_assignment,
    )
    results.append(res_b5)
    raw_metrics[res_b5.system] = (ndcg_b5, recall_b5)
    print(f"  R@10={res_b5.recall_at_10:.3f} nDCG@10={res_b5.ndcg_at_10:.3f} p50={res_b5.p50_e2e_ms:.2f}ms")

    print("Running B6 Soft-Gate centroid + RRF (ingest+query)...")
    res_b6, ndcg_b6, recall_b6 = evaluate(
        "B6 Soft-Gate centroid + RRF (ingest+query)",
        docs,
        queries,
        qrels,
        query_embeddings,
        centroid_profiles,
        "fixed",
        True,
        False,
        False,
        fixed_assignment=final_assignment,
        merge_mode="rrf",
    )
    results.append(res_b6)
    raw_metrics[res_b6.system] = (ndcg_b6, recall_b6)
    print(f"  R@10={res_b6.recall_at_10:.3f} nDCG@10={res_b6.ndcg_at_10:.3f} p50={res_b6.p50_e2e_ms:.2f}ms")

    print("Running B7 Soft-Gate centroid + weighted RRF (ingest+query)...")
    res_b7, ndcg_b7, recall_b7 = evaluate(
        "B7 Soft-Gate centroid + weighted RRF (ingest+query)",
        docs,
        queries,
        qrels,
        query_embeddings,
        centroid_profiles,
        "fixed",
        True,
        False,
        False,
        fixed_assignment=final_assignment,
        merge_mode="weighted_rrf",
    )
    results.append(res_b7)
    raw_metrics[res_b7.system] = (ndcg_b7, recall_b7)
    print(f"  R@10={res_b7.recall_at_10:.3f} nDCG@10={res_b7.ndcg_at_10:.3f} p50={res_b7.p50_e2e_ms:.2f}ms")

    print("Running B8 Soft-Gate centroid top-1 early-exit (ingest+query)...")
    res_b8, ndcg_b8, recall_b8 = evaluate(
        "B8 Soft-Gate centroid top-1 early-exit (ingest+query)",
        docs,
        queries,
        qrels,
        query_embeddings,
        centroid_profiles,
        "fixed",
        True,
        True,
        False,
        fixed_assignment=final_assignment,
        top_experts=1,
    )
    results.append(res_b8)
    raw_metrics[res_b8.system] = (ndcg_b8, recall_b8)
    print(f"  R@10={res_b8.recall_at_10:.3f} nDCG@10={res_b8.ndcg_at_10:.3f} p50={res_b8.p50_e2e_ms:.2f}ms")

    print("Running B9 Soft-Gate centroid top-2 early-exit (ingest+query)...")
    res_b9, ndcg_b9, recall_b9 = evaluate(
        "B9 Soft-Gate centroid top-2 early-exit (ingest+query)",
        docs,
        queries,
        qrels,
        query_embeddings,
        centroid_profiles,
        "fixed",
        True,
        True,
        False,
        fixed_assignment=final_assignment,
        top_experts=2,
    )
    results.append(res_b9)
    raw_metrics[res_b9.system] = (ndcg_b9, recall_b9)
    print(f"  R@10={res_b9.recall_at_10:.3f} nDCG@10={res_b9.ndcg_at_10:.3f} p50={res_b9.p50_e2e_ms:.2f}ms")

    print(f"Computing mixed-membership ingest (target dual-write fraction={MIXED_MEMBERSHIP_DUAL_FRACTION:.0%})...")
    doc_gates = all_gates(embed_texts_cached["docs"], centroid_profiles)
    runner_up_pct = np.percentile(np.sort(doc_gates, axis=1)[:, -2], [10, 25, 50, 75, 90])
    mixed_tau = mixed_membership_tau_for_fraction(doc_gates, MIXED_MEMBERSHIP_DUAL_FRACTION)
    print(f"  Runner-up gate percentiles (10/25/50/75/90): {runner_up_pct.round(4).tolist()}")
    print(f"  Derived tau: {mixed_tau:.4f}")
    mixed_assignment = mixed_membership_domains(doc_gates, mixed_tau)
    membership_sizes = [len(m) for m in mixed_assignment]
    dual_or_more = sum(1 for s in membership_sizes if s > 1)
    total_writes = sum(membership_sizes)
    print(
        f"  Avg experts/doc: {statistics.mean(membership_sizes):.2f}; "
        f"{dual_or_more}/{len(docs)} docs written into 2+ experts; "
        f"{total_writes} total index entries ({total_writes / len(docs):.2f}x vs {len(docs)} docs)"
    )

    print("Running B10 Soft-Gate mixed-membership ingest (ingest+query)...")
    res_b10, ndcg_b10, recall_b10 = evaluate(
        "B10 Soft-Gate mixed-membership ingest (ingest+query)",
        docs,
        queries,
        qrels,
        query_embeddings,
        centroid_profiles,
        "mixed",
        True,
        True,
        False,
        mixed_assignment=mixed_assignment,
    )
    results.append(res_b10)
    raw_metrics[res_b10.system] = (ndcg_b10, recall_b10)
    print(f"  R@10={res_b10.recall_at_10:.3f} nDCG@10={res_b10.ndcg_at_10:.3f} p50={res_b10.p50_e2e_ms:.2f}ms")

    print("Building BM25 index...")
    bm25 = BM25Okapi([bm25_tokenize(d["text"]) for d in docs])
    flat_index = build_indices(embed_texts_cached["docs"], docs, "flat", None, None)

    print("Running B11 BM25 lexical baseline...")
    res_b11, ndcg_b11, recall_b11 = evaluate_bm25("B11 BM25 lexical baseline", docs, queries, qrels, bm25)
    results.append(res_b11)
    raw_metrics[res_b11.system] = (ndcg_b11, recall_b11)
    print(f"  R@10={res_b11.recall_at_10:.3f} nDCG@10={res_b11.ndcg_at_10:.3f} p50={res_b11.p50_e2e_ms:.2f}ms")

    print("Running B12 Hybrid BM25 + dense fusion (flat)...")
    res_b12, ndcg_b12, recall_b12 = evaluate_bm25(
        "B12 Hybrid BM25 + dense fusion (flat)",
        docs,
        queries,
        qrels,
        bm25,
        dense_index=flat_index,
        dense_query_embeddings=query_embeddings,
        hybrid=True,
    )
    results.append(res_b12)
    raw_metrics[res_b12.system] = (ndcg_b12, recall_b12)
    print(f"  R@10={res_b12.recall_at_10:.3f} nDCG@10={res_b12.ndcg_at_10:.3f} p50={res_b12.p50_e2e_ms:.2f}ms")

    print("Running B13 Unweighted multi-index merge (centroid experts)...")
    res_b13, ndcg_b13, recall_b13 = evaluate(
        "B13 Unweighted multi-index merge (centroid experts)",
        docs,
        queries,
        qrels,
        query_embeddings,
        centroid_profiles,
        "fixed",
        True,
        False,
        False,
        fixed_assignment=final_assignment,
    )
    results.append(res_b13)
    raw_metrics[res_b13.system] = (ndcg_b13, recall_b13)
    print(f"  R@10={res_b13.recall_at_10:.3f} nDCG@10={res_b13.ndcg_at_10:.3f} p50={res_b13.p50_e2e_ms:.2f}ms")

    print("Training learned router (logistic regression, doc embeddings -> true domain)...")
    doc_domains = [d["domain"] for d in docs]
    router_clf = train_learned_router(embed_texts_cached["docs"], doc_domains)
    router_train_acc = router_clf.score(embed_texts_cached["docs"], doc_domains)
    print(f"  Router training accuracy on docs: {router_train_acc:.3f}")
    learned_gate_fn = make_learned_gate_fn(router_clf)

    print("Running B14 Learned router (oracle ingest + LR query gate)...")
    res_b14, ndcg_b14, recall_b14 = evaluate(
        "B14 Learned router (oracle ingest + LR query gate)",
        docs,
        queries,
        qrels,
        query_embeddings,
        centroid_profiles,
        "oracle",
        True,
        True,
        False,
        gate_fn=learned_gate_fn,
    )
    results.append(res_b14)
    raw_metrics[res_b14.system] = (ndcg_b14, recall_b14)
    print(f"  R@10={res_b14.recall_at_10:.3f} nDCG@10={res_b14.ndcg_at_10:.3f} p50={res_b14.p50_e2e_ms:.2f}ms")

    print("Running B15 Learned router, top-1 early-exit (oracle ingest)...")
    res_b15, ndcg_b15, recall_b15 = evaluate(
        "B15 Learned router, top-1 early-exit (oracle ingest)",
        docs,
        queries,
        qrels,
        query_embeddings,
        centroid_profiles,
        "oracle",
        True,
        True,
        False,
        gate_fn=learned_gate_fn,
        top_experts=1,
    )
    results.append(res_b15)
    raw_metrics[res_b15.system] = (ndcg_b15, recall_b15)
    print(f"  R@10={res_b15.recall_at_10:.3f} nDCG@10={res_b15.ndcg_at_10:.3f} p50={res_b15.p50_e2e_ms:.2f}ms")

    print("Running B16 Learned router, top-2 early-exit (oracle ingest)...")
    res_b16, ndcg_b16, recall_b16 = evaluate(
        "B16 Learned router, top-2 early-exit (oracle ingest)",
        docs,
        queries,
        qrels,
        query_embeddings,
        centroid_profiles,
        "oracle",
        True,
        True,
        False,
        gate_fn=learned_gate_fn,
        top_experts=2,
    )
    results.append(res_b16)
    raw_metrics[res_b16.system] = (ndcg_b16, recall_b16)
    print(f"  R@10={res_b16.recall_at_10:.3f} nDCG@10={res_b16.ndcg_at_10:.3f} p50={res_b16.p50_e2e_ms:.2f}ms")

    print(f"Loading cross-encoder ({RERANK_MODEL})...")
    cross_encoder = CrossEncoder(RERANK_MODEL)
    centroid_indices = build_indices(
        embed_texts_cached["docs"], docs, "fixed", None, centroid_profiles, fixed_assignment=final_assignment
    )
    domain_keys_list = list(DOMAINS.keys())

    def retrieve_flat(_qi: int, qvec: np.ndarray) -> List[int]:
        ids, _ = flat_index.search(qvec, RERANK_POOL)
        return ids

    def retrieve_centroid_top2(_qi: int, qvec: np.ndarray) -> List[int]:
        gates = softmax_neg_l2(qvec, centroid_profiles)
        order = np.argsort(gates)[::-1][:2]
        scored: List[Tuple[int, float]] = []
        for di in order:
            domain = domain_keys_list[di]
            ids, scores = centroid_indices[domain].search(qvec, RERANK_POOL)
            for did, score in zip(ids, scores):
                scored.append((did, score * gates[di]))
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked, seen = [], set()
        for did, _ in scored:
            if did not in seen:
                ranked.append(did)
                seen.add(did)
            if len(ranked) >= RERANK_POOL:
                break
        return ranked

    print(f"Running B17 Flat + cross-encoder rerank (pool={RERANK_POOL})...")
    res_b17, ndcg_b17, recall_b17 = evaluate_rerank(
        "B17 Flat + cross-encoder rerank", docs, queries, qrels, cross_encoder, retrieve_flat, query_embeddings
    )
    results.append(res_b17)
    raw_metrics[res_b17.system] = (ndcg_b17, recall_b17)
    print(f"  R@10={res_b17.recall_at_10:.3f} nDCG@10={res_b17.ndcg_at_10:.3f} p50={res_b17.p50_e2e_ms:.2f}ms")

    print(f"Running B18 Centroid top-2 routed + cross-encoder rerank (pool={RERANK_POOL})...")
    res_b18, ndcg_b18, recall_b18 = evaluate_rerank(
        "B18 Centroid top-2 routed + cross-encoder rerank",
        docs,
        queries,
        qrels,
        cross_encoder,
        retrieve_centroid_top2,
        query_embeddings,
    )
    results.append(res_b18)
    raw_metrics[res_b18.system] = (ndcg_b18, recall_b18)
    print(f"  R@10={res_b18.recall_at_10:.3f} nDCG@10={res_b18.ndcg_at_10:.3f} p50={res_b18.p50_e2e_ms:.2f}ms")

    if QDRANT_URL:
        print(f"Running B19 Qdrant server at {QDRANT_URL} (M={HNSW_M}, ef_construct={HNSW_EF_CONSTRUCTION}, ef_search={HNSW_EF_SEARCH})...")
    else:
        print("Running B19 Qdrant local mode (exact search; set QDRANT_URL for HNSW on a Qdrant server)...")
    res_b19, ndcg_b19, recall_b19 = evaluate_qdrant(
        qdrant_system_name(),
        docs,
        queries,
        qrels,
        embed_texts_cached["docs"],
        query_embeddings,
    )
    results.append(res_b19)
    raw_metrics[res_b19.system] = (ndcg_b19, recall_b19)
    print(f"  R@10={res_b19.recall_at_10:.3f} nDCG@10={res_b19.ndcg_at_10:.3f} p50={res_b19.p50_e2e_ms:.2f}ms")

    print(f"Computing bootstrap CIs and paired significance vs {results[0].system} (n_boot={N_BOOT})...")
    rng = np.random.default_rng(SEED)
    n_queries = len(queries)
    boot_idx = rng.integers(0, n_queries, size=(N_BOOT, n_queries))
    base_ndcg, base_recall = raw_metrics[results[0].system]

    significance_rows = []
    for r in results:
        ndcg_arr, recall_arr = raw_metrics[r.system]
        ndcg_mean, ndcg_lo, ndcg_hi = bootstrap_mean_ci(ndcg_arr, boot_idx)
        recall_mean, recall_lo, recall_hi = bootstrap_mean_ci(recall_arr, boot_idx)
        row = {
            "system": r.system,
            "ndcg_mean": ndcg_mean,
            "ndcg_ci_lo": ndcg_lo,
            "ndcg_ci_hi": ndcg_hi,
            "recall_mean": recall_mean,
            "recall_ci_lo": recall_lo,
            "recall_ci_hi": recall_hi,
        }
        if r.system != results[0].system:
            ndcg_diff, ndcg_diff_lo, ndcg_diff_hi, ndcg_p = paired_bootstrap_diff(ndcg_arr, base_ndcg, boot_idx)
            recall_diff, recall_diff_lo, recall_diff_hi, recall_p = paired_bootstrap_diff(
                recall_arr, base_recall, boot_idx
            )
            row.update(
                {
                    "ndcg_diff_vs_baseline": ndcg_diff,
                    "ndcg_diff_ci_lo": ndcg_diff_lo,
                    "ndcg_diff_ci_hi": ndcg_diff_hi,
                    "ndcg_p_vs_baseline": ndcg_p,
                    "recall_diff_vs_baseline": recall_diff,
                    "recall_diff_ci_lo": recall_diff_lo,
                    "recall_diff_ci_hi": recall_diff_hi,
                    "recall_p_vs_baseline": recall_p,
                }
            )
            sig = "significant" if ndcg_p < 0.05 else "not significant"
            print(
                f"  {r.system}: nDCG diff={ndcg_diff:+.4f} [{ndcg_diff_lo:+.4f}, {ndcg_diff_hi:+.4f}] "
                f"p={ndcg_p:.3f} ({sig} at alpha=0.05)"
            )
        else:
            row.update(
                {
                    "ndcg_diff_vs_baseline": 0.0,
                    "ndcg_diff_ci_lo": 0.0,
                    "ndcg_diff_ci_hi": 0.0,
                    "ndcg_p_vs_baseline": 1.0,
                    "recall_diff_vs_baseline": 0.0,
                    "recall_diff_ci_lo": 0.0,
                    "recall_diff_ci_hi": 0.0,
                    "recall_p_vs_baseline": 1.0,
                }
            )
        significance_rows.append(row)

    # Ablation: expert count / min_gate proxy via top-1 only merge
    print("Running ablations...")
    ablations = []
    for n_experts in [2, 3]:
        subset_profiles = profiles[:n_experts]
        # reuse full evaluate with truncated profiles by temporarily overriding DOMAINS keys in routing only
        res, _, _ = evaluate(
            f"routed_{n_experts}_experts",
            docs,
            queries,
            qrels,
            query_embeddings,
            subset_profiles,
            "profile_ingest",
            True,
            True,
        )
        ablations.append({"experts": n_experts, **asdict(res)})

    main_path = OUT_DIR / "main-results.csv"
    latency_path = OUT_DIR / "latency.csv"
    ablation_path = OUT_DIR / "ablations.csv"
    significance_path = OUT_DIR / "significance.csv"
    config_path = ROOT / "appendix" / "experimental-config.md"

    with main_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))

    with significance_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(significance_rows[0].keys()))
        w.writeheader()
        for row in significance_rows:
            w.writerow(row)

    with latency_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "system",
                "p50_e2e_ms",
                "p99_e2e_ms",
                "p50_routing_ms",
                "p50_retrieval_ms",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "system": r.system,
                    "p50_e2e_ms": round(r.p50_e2e_ms, 3),
                    "p99_e2e_ms": round(r.p99_e2e_ms, 3),
                    "p50_routing_ms": round(r.p50_routing_ms, 3),
                    "p50_retrieval_ms": round(r.p50_retrieval_ms, 3),
                }
            )

    with ablation_path.open("w", newline="") as f:
        if ablations:
            w = csv.DictWriter(f, fieldnames=list(ablations[0].keys()))
            w.writeheader()
            for row in ablations:
                w.writerow(row)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "# Experimental Configuration",
                "",
                f"- Seed: {SEED}",
                f"- Embedding model: {EMBED_MODEL}",
                f"- HNSW: M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}, efSearch={HNSW_EF_SEARCH}",
                f"- Top-k: {TOP_K}",
                f"- Docs per domain (cap): {MAX_DOCS_PER_DOMAIN}",
                f"- Queries per domain (cap): {MAX_QUERIES_PER_DOMAIN}",
                "- Domains: fiqa (finance), CaseHOLD/LexGLUE (legal), nfcorpus (tech)",
                "- B5 centroid profiles: initial ingest via caption-embedding gates, then "
                "p_i = mean(embeddings in expert i) followed by one reassignment pass "
                f"({moved}/{len(final_assignment)} docs moved on this run)",
                f"- B6 fusion: RRF (k={RRF_K}, pool={RRF_POOL} per expert) over B5's "
                "centroid-routed experts; gates kept for routing diagnostics only, not merge weight",
                f"- B7 fusion: RRF (k={RRF_K}, pool={RRF_POOL} per expert) with each candidate's "
                "RRF score scaled by its expert's routing gate",
                "- B8/B9: same centroid ingest as B5/B6/B7, but query time searches only the "
                "gate's top-1 (B8) or top-2 (B9) experts instead of all 3, gate-weighted merge",
                f"- B10: mixed-membership ingest (target dual-write fraction={MIXED_MEMBERSHIP_DUAL_FRACTION:.0%}, "
                f"derived tau={mixed_tau:.4f} this run) — a doc is dual-written into every expert whose "
                "centroid gate exceeds tau, falling back to its single argmax expert when none does; "
                "query time searches all experts, gate-weighted merge "
                "(note: fixed tau in {0.15, 0.25, 0.30} all degenerated to ~100% dual-write and tau=0.35 "
                "degenerated to 0% — this corpus's gate values cluster tightly near 1/3 for nearly every "
                "doc, so tau is now derived per-run from the runner-up-gate percentile instead of a "
                "hand-picked constant)",
                "- B11: BM25 (rank_bm25 Okapi, lowercase alnum tokenizer) over the full corpus, no "
                "embeddings at all",
                f"- B12: RRF (k={RRF_K}, pool={RRF_POOL}) fusion of B11's BM25 ranking with B0's flat "
                "dense ranking — lexical+dense on the plain unrouted index, no expert routing",
                "- B13: same centroid hard-partition ingest as B5, but query time merges all 3 experts' "
                "raw cosine scores unweighted by the gate (isolates whether gate-weighting the merge "
                "matters at all, vs. just fanning out to every expert)",
                f"- B14: multinomial logistic regression trained on doc embeddings -> true domain label "
                f"(training accuracy {router_train_acc:.3f}), replacing the L2-centroid heuristic as the "
                "query-time gate; ingest placement is oracle (true domain), same as B3, so this isolates "
                "the query-side router itself against B3's heuristic-gate result",
                "- B15/B16: same learned router as B14, but early-exit to only its top-1 (B15) or "
                "top-2 (B16) ranked experts instead of gate-weighting a full 3-expert search — tests "
                "whether early-exit sidesteps the confident-wrong-suppression failure mode found in B14",
                f"- B17/B18: two-stage retrieval with {RERANK_MODEL} reranking a top-{RERANK_POOL} pool "
                "down to the final top-10; B17's pool comes from flat dense search, B18's from the same "
                "gate-weighted centroid top-2 pool B9 uses — isolates whether routing still matters once "
                "a reranker is cleaning up the final order. Cross-encoder inference is batched across all "
                f"queries in one predict() call (batch_size={RERANK_BATCH_SIZE}) instead of one call per "
                f"query, and doc text is capped at {RERANK_MAX_CHARS} chars before pairing; rerank cost is "
                "therefore reported as a per-query average (amortized), not a true per-query percentile — "
                "only the first-stage retrieval latency is genuinely per-query",
                (
                    f"- B19: Qdrant server HNSW (M={HNSW_M}, ef_construct={HNSW_EF_CONSTRUCTION}, "
                    f"ef_search={HNSW_EF_SEARCH}, same as B0; indexing forced and awaited before querying) — "
                    "checks whether B0's hnswlib flat index is a faithful proxy for a real vector DB"
                    if QDRANT_URL
                    else "- B19: qdrant-client local mode, which performs exact brute-force search (no HNSW) — "
                    "an exact-search reference for B0's approximate hnswlib index, not an HNSW-vs-HNSW comparison"
                ),
                f"- Significance: paired percentile bootstrap ({N_BOOT} resamples over queries) "
                f"of each system vs {results[0].system} on nDCG@10 and Recall@10; "
                "95% CIs on both raw means and paired diffs in tables/significance.csv",
                "- Hardware: Apple Silicon, Python 3 benchmark harness",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = {
        "results": [asdict(r) for r in results],
        "ablations": ablations,
        "significance": significance_rows,
        "config": {
            "seed": SEED,
            "model": EMBED_MODEL,
            "docs": len(docs),
            "queries": len(queries),
            "n_boot": N_BOOT,
        },
    }
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {main_path}")


if __name__ == "__main__":
    main()
