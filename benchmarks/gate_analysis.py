#!/usr/bin/env python3
"""
Gate-distribution, temperature-sweep, per-domain, and misroute analyses
reported in Sections 6.1 and 6.3 of the paper. Reuses the data loading,
embedding, and evaluation machinery in run_benchmark.py so every number here
is produced by the same pipeline as the main sweep (B0-B19).

Produces, into tables/gate_analysis.json:
  1. A sanity re-run of B1 (flat) and B5 (centroid gate, T=1), checked against
     the published main-results.csv numbers.
  2. Gate entropy/confidence distribution for the centroid gate (not just the
     mean): percentiles of entropy and top-1 gate probability.
  3. A softmax-temperature sweep over the centroid gate: entropy, routing
     accuracy, and full retrieval quality (nDCG@10, Recall@10) at each T.
  4. A per-domain breakdown (finance / legal / tech) of flat vs. centroid-gate
     retrieval quality and routing accuracy, including where misroutes land.
  5. Worked misroute examples (query text, true vs. predicted domain, gate
     vector) for a debugging-with-the-gate illustration.

The learned-router calibration analysis (Section 6.3, B14) lives in the
companion script calibration_sweep.py.

Caches corpus+embeddings to benchmarks/data/.reviewer_cache.pkl / .npz so a
re-run (e.g. after a crash) does not re-download or re-embed from scratch;
calibration_sweep.py reuses the same cache.
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
OUT_PATH = ROOT / "tables" / "gate_analysis.json"
CACHE_PKL = Path(__file__).resolve().parent / "data" / ".reviewer_cache.pkl"
CACHE_NPZ = Path(__file__).resolve().parent / "data" / ".reviewer_cache.npz"

TEMPS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
DOMAIN_KEYS = list(rb.DOMAINS.keys())


def log(msg: str) -> None:
    print(f"[gate_analysis] {msg}", flush=True)


def load_or_build_corpus_and_embeddings():
    if CACHE_PKL.exists() and CACHE_NPZ.exists():
        log("Loading cached corpus + embeddings...")
        with open(CACHE_PKL, "rb") as f:
            docs, queries, qrels = pickle.load(f)
        npz = np.load(CACHE_NPZ)
        doc_emb, query_emb, profiles = npz["doc_emb"], npz["query_emb"], npz["profiles"]
        return docs, queries, qrels, doc_emb, query_emb, profiles

    rb.set_seed(rb.SEED)
    rb.DATA_DIR.mkdir(parents=True, exist_ok=True)
    log("Loading BEIR/CaseHOLD corpus (network download if not cached)...")
    docs, queries, qrels = rb.load_corpus()
    log(f"Corpus: {len(docs)} docs, {len(queries)} queries")

    log(f"Loading embedding model {rb.EMBED_MODEL}...")
    model = rb.SentenceTransformer(rb.EMBED_MODEL)
    doc_texts = [d["text"] for d in docs]
    query_texts = [q["text"] for q in queries]
    profile_texts = [cfg["profile"] for cfg in rb.DOMAINS.values()]

    log("Embedding docs...")
    doc_emb = rb.embed_texts(model, doc_texts, prefix=rb.PASSAGE_PREFIX)
    log("Embedding queries...")
    query_emb = rb.embed_texts(model, query_texts, prefix=rb.QUERY_PREFIX)
    log("Embedding profiles...")
    profiles = rb.embed_texts(model, profile_texts, prefix=rb.PASSAGE_PREFIX)

    CACHE_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PKL, "wb") as f:
        pickle.dump((docs, queries, qrels), f)
    np.savez(CACHE_NPZ, doc_emb=doc_emb, query_emb=query_emb, profiles=profiles)
    log("Cached corpus + embeddings for future re-runs.")
    return docs, queries, qrels, doc_emb, query_emb, profiles


def softmax_neg_l2_temp(query: np.ndarray, profiles: np.ndarray, T: float) -> np.ndarray:
    dists = np.linalg.norm(profiles - query, axis=1)
    logits = -dists / T
    logits -= logits.max()
    exps = np.exp(logits)
    return exps / exps.sum()


def gate_distribution(gates_all: np.ndarray) -> dict:
    entropies = np.array([rb.entropy(g) for g in gates_all])
    top1 = gates_all.max(axis=1)
    pct = lambda a, p: float(np.percentile(a, p))
    return {
        "entropy_mean": float(entropies.mean()),
        "entropy_p10": pct(entropies, 10),
        "entropy_p25": pct(entropies, 25),
        "entropy_p50": pct(entropies, 50),
        "entropy_p75": pct(entropies, 75),
        "entropy_p90": pct(entropies, 90),
        "entropy_max_possible": float(np.log(len(DOMAIN_KEYS))),
        "top1_prob_mean": float(top1.mean()),
        "top1_prob_p10": pct(top1, 10),
        "top1_prob_p50": pct(top1, 50),
        "top1_prob_p90": pct(top1, 90),
        "frac_entropy_above_90pct_of_max": float((entropies > 0.9 * np.log(len(DOMAIN_KEYS))).mean()),
        "frac_top1_prob_above_0.5": float((top1 > 0.5).mean()),
        "frac_top1_prob_above_0.8": float((top1 > 0.8).mean()),
    }


def per_domain_breakdown(queries, ndcg_arr, recall_arr, route_pred=None):
    out = {}
    for dom in DOMAIN_KEYS:
        idxs = [i for i, q in enumerate(queries) if q["domain"] == dom]
        if not idxs:
            continue
        entry = {
            "n_queries": len(idxs),
            "ndcg_at_10": float(np.mean([ndcg_arr[i] for i in idxs])),
            "recall_at_10": float(np.mean([recall_arr[i] for i in idxs])),
        }
        if route_pred is not None:
            entry["routing_top1_acc"] = float(
                np.mean([1.0 if route_pred[i] == dom else 0.0 for i in idxs])
            )
            # where misroutes land, by predicted domain
            wrong = [route_pred[i] for i in idxs if route_pred[i] != dom]
            if wrong:
                counts = {d: wrong.count(d) for d in set(wrong)}
                entry["misroute_targets"] = counts
        out[dom] = entry
    return out


def main() -> None:
    docs, queries, qrels, doc_emb, query_emb, profiles = load_or_build_corpus_and_embeddings()
    rb.embed_texts_cached["docs"] = doc_emb

    log("Building centroid profiles (authored-profile ingest + one reassignment pass)...")
    initial_assignment = rb.assign_by_profiles(doc_emb, profiles)
    centroid_profiles = rb.centroid_profiles_from_assignment(doc_emb, initial_assignment)
    final_assignment = rb.assign_by_profiles(doc_emb, centroid_profiles)
    moved = sum(1 for a, b in zip(initial_assignment, final_assignment) if a != b)
    log(f"Reassignment moved {moved}/{len(final_assignment)} docs")

    out = {"config": {"n_docs": len(docs), "n_queries": len(queries), "seed": rb.SEED}}

    # ── Sanity check against the published CSV: rerun flat + centroid(T=1) ──
    log("Sanity re-run: B1-equivalent (flat)...")
    res_flat, ndcg_flat, recall_flat = rb.evaluate(
        "flat_sanity", docs, queries, qrels, query_emb, profiles, "flat", False, False, False
    )
    log(f"  flat: nDCG={res_flat.ndcg_at_10:.4f} R@10={res_flat.recall_at_10:.4f}")

    log("Sanity re-run: B5-equivalent (centroid gate, T=1, full search)...")
    res_c1, ndcg_c1, recall_c1 = rb.evaluate(
        "centroid_T1", docs, queries, qrels, query_emb, centroid_profiles, "fixed", True, True, False,
        fixed_assignment=final_assignment,
    )
    log(f"  centroid T=1: nDCG={res_c1.ndcg_at_10:.4f} R@10={res_c1.recall_at_10:.4f} "
        f"top1={res_c1.routing_top1_acc:.4f} entropy={res_c1.gate_entropy:.4f}")

    out["sanity_check_vs_published_csv"] = {
        "flat_ndcg": res_flat.ndcg_at_10, "flat_recall": res_flat.recall_at_10,
        "centroid_T1_ndcg": res_c1.ndcg_at_10, "centroid_T1_recall": res_c1.recall_at_10,
        "centroid_T1_top1_acc": res_c1.routing_top1_acc, "centroid_T1_entropy": res_c1.gate_entropy,
        "published_flat_ndcg_ref": 0.4316, "published_centroid_ndcg_ref": 0.4301,
    }

    # ── Gate distribution (centroid gate, T=1) ──
    log("Computing full gate-vector distribution for every query (centroid gate)...")
    gates_all = np.stack([rb.softmax_neg_l2(query_emb[i], centroid_profiles) for i in range(len(queries))])
    route_pred_centroid = [DOMAIN_KEYS[int(np.argmax(g))] for g in gates_all]
    out["gate_distribution_centroid_T1"] = gate_distribution(gates_all)

    # ── Temperature sweep on the centroid gate ──
    log("Temperature sweep on centroid gate...")
    temp_sweep = []
    for T in TEMPS:
        gates_T = np.stack([softmax_neg_l2_temp(query_emb[i], centroid_profiles, T) for i in range(len(queries))])
        dist = gate_distribution(gates_T)
        pred_T = [DOMAIN_KEYS[int(np.argmax(g))] for g in gates_T]
        top1_acc_T = float(np.mean([1.0 if pred_T[i] == queries[i]["domain"] else 0.0 for i in range(len(queries))]))

        gate_fn_T = (lambda T=T: (lambda qvec: softmax_neg_l2_temp(qvec, centroid_profiles, T)))()
        res_T, ndcg_T, recall_T = rb.evaluate(
            f"centroid_T{T}", docs, queries, qrels, query_emb, centroid_profiles, "fixed", True, True, False,
            fixed_assignment=final_assignment, gate_fn=gate_fn_T,
        )
        row = {
            "T": T,
            "entropy_mean": dist["entropy_mean"],
            "top1_prob_mean": dist["top1_prob_mean"],
            "routing_top1_acc": top1_acc_T,
            "ndcg_at_10": res_T.ndcg_at_10,
            "recall_at_10": res_T.recall_at_10,
        }
        temp_sweep.append(row)
        log(f"  T={T}: entropy={row['entropy_mean']:.3f} top1acc={top1_acc_T:.3f} "
            f"nDCG={res_T.ndcg_at_10:.4f} R@10={res_T.recall_at_10:.4f}")
    out["centroid_temperature_sweep"] = temp_sweep

    # ── Per-domain breakdown ──
    log("Per-domain breakdown (flat vs centroid gate)...")
    out["per_domain_flat"] = per_domain_breakdown(queries, ndcg_flat, recall_flat)
    out["per_domain_centroid_T1"] = per_domain_breakdown(
        queries, ndcg_c1, recall_c1, route_pred=route_pred_centroid
    )

    # ── Worked misroute examples ──
    log("Extracting worked misroute examples...")
    misroute_idxs = [i for i in range(len(queries)) if route_pred_centroid[i] != queries[i]["domain"]]
    examples = []
    for i in misroute_idxs[:5]:
        examples.append({
            "query_text": queries[i]["text"][:220],
            "true_domain": queries[i]["domain"],
            "predicted_domain": route_pred_centroid[i],
            "gate_vector": {DOMAIN_KEYS[j]: float(gates_all[i][j]) for j in range(len(DOMAIN_KEYS))},
            "ndcg_at_10_this_query": float(ndcg_c1[i]),
            "recall_at_10_this_query": float(recall_c1[i]),
        })
    out["worked_misroute_examples"] = examples

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
