#!/usr/bin/env python3
"""
Learned-router (B14) temperature-scaling calibration analysis, Section 6.3.

B14 replaces the centroid gate with a multinomial logistic regression trained
on true domain labels: more accurate at routing (91.7% top-1 vs. the centroid
gate's 85.4%) but, used as-is, far more confident and worse on retrieval
quality (nDCG@10 = 0.414 vs. flat's ~0.432). This script asks two questions:

  1. Is that a calibration artifact rather than a fundamental accuracy-versus-
     safety trade-off? We rescale the router's decision-function output by a
     temperature T before the softmax (no retraining) and sweep T from 0.1 to
     128 under full three-expert search, reporting entropy, routing accuracy,
     nDCG@10, and Recall@10 at each point.
  2. Does the router's higher routing accuracy pay off once early-exit search
     is in play, where only the top-1 (argmax) or top-2 ranked experts are
     searched at all? Under top-1 early exit the merge draws from a single
     expert, so the result is temperature-invariant and depends purely on
     routing accuracy; under top-2 it depends on both accuracy and T. We
     compare both against B8 (centroid gate, top-1 early-exit, nDCG@10=0.376)
     and, for top-2, against B9 (centroid gate, top-2 early-exit, nDCG@10=0.430).

Reuses the cached corpus/embeddings written by gate_analysis.py
(benchmarks/data/.reviewer_cache.pkl / .npz); run gate_analysis.py first if
that cache does not exist yet.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_benchmark as rb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "tables" / "calibration_sweep.json"
CACHE_PKL = Path(__file__).resolve().parent / "data" / ".reviewer_cache.pkl"
CACHE_NPZ = Path(__file__).resolve().parent / "data" / ".reviewer_cache.npz"

TEMPS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
DOMAIN_KEYS = list(rb.DOMAINS.keys())

B8_CENTROID_TOP1_NDCG_REF = 0.376
B9_CENTROID_TOP2_NDCG_REF = 0.430


def log(msg: str) -> None:
    print(f"[calibration_sweep] {msg}", flush=True)


def load_cache():
    if not (CACHE_PKL.exists() and CACHE_NPZ.exists()):
        raise FileNotFoundError(
            "No cached corpus/embeddings found. Run gate_analysis.py first; "
            "it downloads the corpus, embeds it, and writes the cache this script reuses."
        )
    with open(CACHE_PKL, "rb") as f:
        docs, queries, qrels = pickle.load(f)
    npz = np.load(CACHE_NPZ)
    return docs, queries, qrels, npz["doc_emb"], npz["query_emb"], npz["profiles"]


def entropy(gates: np.ndarray) -> float:
    return rb.entropy(gates)


def main() -> None:
    docs, queries, qrels, doc_emb, query_emb, profiles = load_cache()
    rb.embed_texts_cached["docs"] = doc_emb

    initial_assignment = rb.assign_by_profiles(doc_emb, profiles)
    centroid_profiles = rb.centroid_profiles_from_assignment(doc_emb, initial_assignment)

    log("Training learned router (B14: multinomial logistic regression, oracle ingest)...")
    doc_domains = [d["domain"] for d in docs]
    router_clf = rb.train_learned_router(doc_emb, doc_domains)
    class_to_col = {c: i for i, c in enumerate(router_clf.classes_)}
    col_order = [class_to_col[d] for d in DOMAIN_KEYS]

    def lr_gate_fn_T(qvec: np.ndarray, T: float) -> np.ndarray:
        logits = router_clf.decision_function(qvec.reshape(1, -1))[0]
        logits = logits[col_order]
        logits = logits / T
        logits = logits - logits.max()
        exps = np.exp(logits)
        return exps / exps.sum()

    out: dict = {}

    # ── 1. Full three-expert search: temperature-scaling sweep ──
    log("Temperature-scaling sweep, full three-expert search...")
    full_sweep = []
    for T in TEMPS:
        gates_T = np.stack([lr_gate_fn_T(query_emb[i], T) for i in range(len(queries))])
        ent = float(np.mean([entropy(g) for g in gates_T]))
        pred_T = [DOMAIN_KEYS[int(np.argmax(g))] for g in gates_T]
        top1_acc_T = float(np.mean([1.0 if pred_T[i] == queries[i]["domain"] else 0.0 for i in range(len(queries))]))

        gate_fn_T = (lambda T=T: (lambda qvec: lr_gate_fn_T(qvec, T)))()
        res, ndcg_arr, recall_arr = rb.evaluate(
            f"learned_full_T{T}", docs, queries, qrels, query_emb, centroid_profiles, "oracle", True, True, False,
            gate_fn=gate_fn_T,
        )
        row = {"T": T, "entropy_mean": ent, "routing_top1_acc": top1_acc_T,
               "ndcg_at_10": res.ndcg_at_10, "recall_at_10": res.recall_at_10}
        full_sweep.append(row)
        log(f"  T={T}: entropy={ent:.4f} top1acc={top1_acc_T:.3f} "
            f"nDCG={res.ndcg_at_10:.4f} R@10={res.recall_at_10:.4f}")
    out["full_search_temperature_sweep"] = full_sweep

    # ── 2a. Top-1 early exit (temperature-invariant: single expert, any T) ──
    log("Top-1 early exit (single expert searched; result is temperature-invariant)...")
    top1_rows = []
    for T in [0.1, 1.0, 128.0]:  # sanity-check invariance across widely different T
        gate_fn_T = (lambda T=T: (lambda qvec: lr_gate_fn_T(qvec, T)))()
        res, _, _ = rb.evaluate(
            f"learned_top1_T{T}", docs, queries, qrels, query_emb, centroid_profiles, "oracle", True, True, False,
            gate_fn=gate_fn_T, top_experts=1,
        )
        row = {"T": T, "ndcg_at_10": res.ndcg_at_10, "recall_at_10": res.recall_at_10, "p50_e2e_ms": res.p50_e2e_ms}
        top1_rows.append(row)
        log(f"  T={T}: nDCG={res.ndcg_at_10:.4f} R@10={res.recall_at_10:.4f} p50={res.p50_e2e_ms:.3f}ms")
    out["top1_early_exit"] = top1_rows
    out["top1_vs_B8"] = {
        "learned_top1_ndcg": top1_rows[-1]["ndcg_at_10"],
        "B8_centroid_top1_ndcg_ref": B8_CENTROID_TOP1_NDCG_REF,
        "learned_beats_B8": top1_rows[-1]["ndcg_at_10"] > B8_CENTROID_TOP1_NDCG_REF,
    }

    # ── 2b. Top-2 early exit: temperature matters, sweep it ──
    log("Top-2 early exit, temperature sweep...")
    top2_rows = []
    for T in TEMPS:
        gate_fn_T = (lambda T=T: (lambda qvec: lr_gate_fn_T(qvec, T)))()
        res, _, _ = rb.evaluate(
            f"learned_top2_T{T}", docs, queries, qrels, query_emb, centroid_profiles, "oracle", True, True, False,
            gate_fn=gate_fn_T, top_experts=2,
        )
        ent = float(np.mean([entropy(lr_gate_fn_T(query_emb[i], T)) for i in range(len(queries))]))
        row = {"T": T, "entropy_mean": ent, "ndcg_at_10": res.ndcg_at_10,
               "recall_at_10": res.recall_at_10, "p50_e2e_ms": res.p50_e2e_ms}
        top2_rows.append(row)
        log(f"  T={T}: entropy={ent:.4f} nDCG={res.ndcg_at_10:.4f} "
            f"R@10={res.recall_at_10:.4f} p50={res.p50_e2e_ms:.3f}ms")
    out["top2_early_exit_sweep"] = top2_rows
    best = max(top2_rows, key=lambda r: r["ndcg_at_10"])
    out["top2_best"] = best
    out["top2_vs_B8_and_B9"] = {
        "learned_top2_best_ndcg": best["ndcg_at_10"], "at_T": best["T"],
        "B8_centroid_top1_ndcg_ref": B8_CENTROID_TOP1_NDCG_REF,
        "B9_centroid_top2_ndcg_ref": B9_CENTROID_TOP2_NDCG_REF,
        "learned_beats_B8": best["ndcg_at_10"] > B8_CENTROID_TOP1_NDCG_REF,
        "learned_beats_B9": best["ndcg_at_10"] > B9_CENTROID_TOP2_NDCG_REF,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
