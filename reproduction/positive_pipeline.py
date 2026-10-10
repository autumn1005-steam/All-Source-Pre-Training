"""Declared completion of the manuscript's missing positive generator.

This is a new, explicitly specified implementation, NOT a recovered original
generator. It never opens labels or train/validation/test split files. All
strategies use the same complete non-self meta-path-union neighbor candidates.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp


VERSION = "declared_simrank_union_v1"
ROOT = Path(__file__).resolve().parent
CONFIG = {
    "acm": {"mps": ["pap", "psp"], "feat": "p_feat"},
    "dblp": {"mps": ["apa", "apcpa", "aptpa"], "feat": "a_feat"},
    "aminer": {"mps": ["pap", "prp"], "feat": None},
    "freebase": {"mps": ["mdm", "mam", "mwm"], "feat": None},
}
MODES = ("cosine", "simrank", "combined", "random")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def csr_content_hash(matrix):
    """Stable value hash, independent of compressed NPZ ZIP timestamps."""
    x = matrix.tocsr(copy=True)
    x.sum_duplicates()
    x.sort_indices()
    h = hashlib.sha256()
    for array in (np.asarray(x.shape, dtype="<i8"),
                  x.indptr.astype("<i8"), x.indices.astype("<i8"),
                  x.data.astype("<f8")):
        h.update(array.tobytes())
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def current_rss_bytes():
    """Current RSS for sampling; process lifetime peak is reported separately."""
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        if sys.platform.startswith("linux"):
            fields = Path("/proc/self/statm").read_text().split()
            return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
        return None


class MemorySampler:
    def __enter__(self):
        self.before = current_rss_bytes()
        self.peak = self.before
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(.02):
                value = current_rss_bytes()
                if value is not None:
                    self.peak = max(self.peak or 0, value)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *unused):
        value = current_rss_bytes()
        if value is not None:
            self.peak = max(self.peak or 0, value)
        self.stop.set()
        self.thread.join()

    def report(self):
        scale = 2 ** 20
        return {
            "host_rss_baseline_mib": None if self.before is None else self.before / scale,
            "host_rss_sampled_peak_mib": None if self.peak is None else self.peak / scale,
            "host_rss_sampled_increment_mib": None if self.before is None or self.peak is None else (self.peak - self.before) / scale,
            "rss_sampling_interval_seconds": .02,
            "rss_caveat": "Sampled current process RSS; 20 ms sampling can miss transient peaks.",
        }


def binary_incoming_union(raw):
    """Rows contain incoming union neighbors; remove loops and multiplicities."""
    if not raw:
        raise ValueError("At least one target meta-path matrix is required")
    n = raw[0].shape[0]
    union = sp.csr_matrix((n, n), dtype=bool)
    for a in raw:
        if a.shape != (n, n) or not np.isfinite(a.data).all():
            raise ValueError("Invalid target meta-path matrix")
        if (a.data < 0).any():
            raise ValueError("Negative graph edge values are unsupported")
        union = union.maximum(a.astype(bool))
    union.setdiag(False)
    union.eliminate_zeros()
    incoming = union.T.tocsr()
    incoming.sort_indices()
    return incoming


def transition(incoming, dtype=np.float32):
    degree = np.diff(incoming.indptr)
    inverse = np.zeros(len(degree), dtype=dtype)
    inverse[degree > 0] = 1 / degree[degree > 0]
    return (sp.diags(inverse) @ incoming.astype(dtype)).tocsr()


def simrank_cpu(incoming, c=.9, iterations=3, dtype=np.float32):
    """Exact finite-depth recurrence: O(L*m*n) products with quadratic state."""
    if not 0 <= c < 1 or iterations < 0:
        raise ValueError("Require 0 <= C < 1 and a nonnegative integer depth")
    w = transition(incoming, dtype)
    score = np.eye(incoming.shape[0], dtype=dtype)
    for _ in range(iterations):
        left = w @ score
        score = np.asarray((w @ left.T).T, dtype=dtype)
        score *= c
        np.fill_diagonal(score, 1)
        del left
    return score


def simrank_cuda(incoming, c=.9, iterations=3, block=256):
    """Same recurrence, with bounded temporary right-product row blocks.

    S and W@S remain n*n. Blocking avoids a third full n*n GPU result. Time
    remains proportional to finite-depth sparse-dense multiplication; this
    implementation does not claim subquadratic storage or arithmetic.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not 0 <= c < 1 or iterations < 0 or block < 1:
        raise ValueError("Invalid SimRank parameters")
    w = transition(incoming).tocoo()
    indices = torch.as_tensor(np.vstack([w.row, w.col]), dtype=torch.long, device="cuda")
    values = torch.as_tensor(w.data, dtype=torch.float32, device="cuda")
    wt = torch.sparse_coo_tensor(indices, values, w.shape, device="cuda").coalesce()
    n = incoming.shape[0]
    score = torch.eye(n, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        for _ in range(iterations):
            left = torch.sparse.mm(wt, score)
            # W @ left.T = (left @ W.T).T. Overwrite S by row blocks after
            # left is complete; no old S entries are read in this phase.
            for lo in range(0, n, block):
                hi = min(n, lo + block)
                score[lo:hi] = torch.sparse.mm(wt, left[lo:hi].T).T * c
            score.diagonal().fill_(1)
            del left
    torch.cuda.synchronize()
    result = score.cpu().numpy()
    del score, wt, indices, values
    return result


def normalized_raw_features(features):
    if features is None:
        return None
    x = features.tocsr().astype(np.float32)
    if not np.isfinite(x.data).all():
        raise ValueError("Nonfinite feature value")
    norm = np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
    inverse = np.zeros_like(norm)
    inverse[norm > 0] = 1 / norm[norm > 0]
    return (sp.diags(inverse) @ x).tocsr()


def cosine_block(x, lo, hi, n):
    if x is None:
        # Identity features have exactly zero cosine for non-self candidates.
        out = np.zeros((hi - lo, n), dtype=np.float32)
        out[np.arange(hi - lo), np.arange(lo, hi)] = 1
        return out
    return (x[lo:hi] @ x.T).toarray().astype(np.float32, copy=False)


def select_all(incoming, features, structural, modes=MODES, k=8,
               alpha=.5, random_seed=20261010, block=128):
    if k < 0 or not 0 <= alpha <= 1 or block < 1:
        raise ValueError("Invalid ranking parameters")
    modes = tuple(modes)
    if not set(modes).issubset(MODES):
        raise ValueError("Unknown positive ranking strategy")
    n = incoming.shape[0]
    if structural is None and set(modes) & {"simrank", "combined"}:
        raise ValueError("Structural similarities required")
    if structural is not None and structural.shape != (n, n):
        raise ValueError("Wrong SimRank shape")
    x = normalized_raw_features(features)
    if x is not None and x.shape[0] != n:
        raise ValueError("Wrong target feature row count")
    rows = {mode: [] for mode in modes}
    cols = {mode: [] for mode in modes}
    for lo in range(0, n, block):
        hi = min(n, lo + block)
        cosine = cosine_block(x, lo, hi, n) if set(modes) & {"cosine", "combined"} else None
        for i in range(lo, hi):
            candidates = incoming.indices[incoming.indptr[i]:incoming.indptr[i + 1]]
            budget = min(k, len(candidates))
            for mode in modes:
                if mode == "cosine":
                    score = cosine[i - lo, candidates]
                elif mode == "simrank":
                    score = structural[i, candidates]
                elif mode == "combined":
                    score = alpha * cosine[i - lo, candidates] + (1 - alpha) * structural[i, candidates]
                else:
                    rng = np.random.default_rng(np.random.SeedSequence([random_seed, i]))
                    score = rng.random(len(candidates))
                # Full candidate set, no cap. Primary: descending score;
                # secondary: ascending node id, including exact zero-score ties.
                order = np.lexsort((candidates, -score))[:budget]
                chosen = np.concatenate(([i], candidates[order]))
                rows[mode].extend([i] * len(chosen))
                cols[mode].extend(chosen.tolist())
    result = {}
    budget = 1 + np.minimum(k, np.diff(incoming.indptr))
    for mode in modes:
        matrix = sp.csr_matrix((np.ones(len(rows[mode]), dtype=np.float32),
                                (rows[mode], cols[mode])), shape=(n, n))
        matrix.sort_indices()
        if not np.array_equal(np.diff(matrix.indptr), budget):
            raise AssertionError("Positive budgets differ from the declared candidates")
        if not np.array_equal(matrix.diagonal(), np.ones(n)):
            raise AssertionError("Each row must include self once")
        off = matrix.copy()
        off.setdiag(0)
        off.eliminate_zeros()
        if (off - off.multiply(incoming)).nnz:
            raise AssertionError("A positive is outside the candidate set")
        result[mode] = matrix
    return result


def load_inputs(source_root, dataset):
    """Exactly adjacency matrices and target raw features; no label API."""
    cfg = CONFIG[dataset]
    data = Path(source_root) / "data" / dataset
    paths = [data / (name + ".npz") for name in cfg["mps"]]
    raw = [sp.load_npz(path).tocsr() for path in paths]
    feat = None
    if cfg["feat"]:
        path = data / (cfg["feat"] + ".npz")
        feat = sp.load_npz(path).tocsr()
        paths.append(path)
    hashes = {path.name: sha256(path) for path in paths}
    return raw, feat, hashes


def generate_dataset(source_root, output_root, dataset, modes=MODES,
                     device="cpu", k=8, c=.9, iterations=3, alpha=.5,
                     random_seed=20261010, block=256):
    destination = Path(output_root) / dataset
    destination.mkdir(parents=True, exist_ok=True)
    gpu = {}
    if device == "cuda":
        import torch
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        gpu = {"cuda_baseline_allocated_mib": torch.cuda.memory_allocated() / 2 ** 20,
               "cuda_baseline_reserved_mib": torch.cuda.memory_reserved() / 2 ** 20,
               "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
               "cuda": torch.version.cuda}
    phase = {}
    start = time.perf_counter()
    with MemorySampler() as memory:
        t = time.perf_counter()
        raw, features, input_hashes = load_inputs(source_root, dataset)
        phase["input_loading_seconds"] = time.perf_counter() - t
        t = time.perf_counter()
        incoming = binary_incoming_union(raw)
        phase["binary_union_seconds"] = time.perf_counter() - t
        t = time.perf_counter()
        structural = None
        if set(modes) & {"simrank", "combined"}:
            structural = (simrank_cuda(incoming, c, iterations, block)
                          if device == "cuda" else simrank_cpu(incoming, c, iterations))
        phase["simrank_seconds"] = time.perf_counter() - t
        t = time.perf_counter()
        positives = select_all(incoming, features, structural, modes, k, alpha, random_seed)
        phase["all_requested_ranking_seconds"] = time.perf_counter() - t
        t = time.perf_counter()
        files = {}
        for mode, matrix in positives.items():
            path = destination / (mode + ".npz")
            sp.save_npz(path, matrix, compressed=True)
            counts = np.diff(matrix.indptr)
            files[mode] = {"file": path.name, "sha256": sha256(path),
                           "csr_content_sha256": csr_content_hash(matrix),
                           "nnz": int(matrix.nnz), "self_pairs": len(counts),
                           "total_per_row_min": int(counts.min()),
                           "total_per_row_max": int(counts.max()),
                           "total_per_row_mean": float(counts.mean())}
        phase["serialization_seconds"] = time.perf_counter() - t
        phase["total_generation_seconds"] = time.perf_counter() - start
    if device == "cuda":
        torch.cuda.synchronize()
        gpu.update({"cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                    "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20})
    degree = np.diff(incoming.indptr)
    summary = {
        "version": VERSION, "dataset": dataset, "n_target": incoming.shape[0],
        "origin": "New declared completion of missing generator using original C=.9, L=3, alpha=.5; not recovered original code or claimed exact original reproduction.",
        "candidate_rule": "All non-self incoming neighbors of the binary union of released target meta-path matrices; no cap or non-neighbor fill; all strategies share exactly this set.",
        "graph_direction": "Incoming adjacency is the transpose of the binary outgoing union; released symmetric projections yield the same neighbor sets.",
        "union_is_symmetric": (incoming != incoming.T).nnz == 0,
        "union_nonself_directed_entries": int(incoming.nnz),
        "union_csr_content_sha256": csr_content_hash(incoming),
        "budget": "One self pair plus min(K, non-self candidate degree) per node, equal for every strategy.",
        "k_nonself": k, "c": c, "iterations": iterations, "alpha_cosine": alpha,
        "simrank_recurrence": "S0=I; W=row-normalized incoming binary union; St+1=C W St W^T; overwrite diagonal with 1 each round; zero-degree off-diagonal similarities stay 0.",
        "combined_score": "alpha * raw-feature cosine + (1-alpha) * finite-depth SimRank, without additional score normalization.",
        "cosine_features": CONFIG[dataset]["feat"] or "identity; all non-self cosine values are 0, so cosine and mixed semantic interpretation are degenerate",
        "raw_feature_use": "Fixed released target attributes, L2-normalized per row; zero-feature rows give 0 cosine; no learned embedding.",
        "tie_rule": "Descending score then ascending target node id; ties never selected by labels.",
        "random_rule": "One frozen PCG64 uniform ranking per node seeded by SeedSequence([random_seed, node_id]); candidate set and per-row budget matched.",
        "random_seed": random_seed, "label_use": "No labels or split files opened by generator.",
        "requested_modes": list(modes), "isolated_target_nodes": int((degree == 0).sum()),
        "nodes_with_fewer_than_k_candidates": int((degree < k).sum()),
        "candidate_degree_min": int(degree.min()), "candidate_degree_max": int(degree.max()),
        "candidate_degree_mean": float(degree.mean()), "input_sha256": input_hashes,
        "files": files, "phase_seconds": phase, "memory": {**memory.report(), **gpu},
        "measurement_boundary": "Loading supplied projections/features, binary union, all three SimRank rounds, requested rankings, and compressed positive serialization. Original heterogeneous-to-meta-path projection construction is outside this boundary and measured separately in the prior study.",
        "measurement_execution": "Run on idle device; do not overlap generation profiling with encoder jobs.",
        "device": device, "cuda_product_block_rows": block if device == "cuda" else None,
        "environment": {"numpy": np.__version__, "scipy": scipy.__version__,
                        "python": sys.version, "platform": platform.platform()},
    }
    atomic_json(destination / "manifest.json", summary)
    del raw, features, structural, incoming, positives
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print("GENERATED", dataset, json.dumps({"seconds": phase["total_generation_seconds"],
          "nodes": summary["n_target"], "degree_below_k": summary["nodes_with_fewer_than_k_candidates"],
          "cuda_peak_mib": gpu.get("cuda_peak_allocated_mib")}), flush=True)
    return summary


def reference_simrank(incoming, c, iterations):
    n = incoming.shape[0]
    score = np.eye(n, dtype=np.float64)
    neighbors = [incoming.indices[incoming.indptr[i]:incoming.indptr[i + 1]] for i in range(n)]
    for _ in range(iterations):
        new = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            new[i, i] = 1
            for j in range(n):
                if i != j and len(neighbors[i]) and len(neighbors[j]):
                    new[i, j] = c * score[np.ix_(neighbors[i], neighbors[j])].mean()
        score = new
    return score


def self_test(check_cuda=False):
    """Mathematical/protocol tests independent of paper numerical outcomes."""
    tests = []
    graphs = [
        sp.csr_matrix([[0, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0],
                       [0, 1, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float64),
        sp.csr_matrix([[0, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64),
        sp.csr_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
    ]
    for graph_id, a in enumerate(graphs):
        incoming = binary_incoming_union([a])
        for depth in (0, 1, 2, 3):
            expected = reference_simrank(incoming, .9, depth)
            actual = simrank_cpu(incoming, .9, depth, np.float64)
            difference = float(np.max(np.abs(expected - actual)))
            assert difference < 1e-12
            assert np.array_equal(np.diag(actual), np.ones(a.shape[0]))
            isolated = np.diff(incoming.indptr) == 0
            isolated_off = actual[isolated].copy()
            isolated_off[np.arange(isolated.sum()), np.where(isolated)[0]] = 0
            assert np.all(isolated_off == 0)
            if check_cuda:
                gpu = simrank_cuda(incoming, .9, depth, block=2)
                assert np.max(np.abs(expected - gpu)) < 2e-6
            tests.append({"test": "finite-depth recurrence including isolated and directed graph",
                          "graph": graph_id, "depth": depth, "max_abs_difference": difference})
    a = sp.csr_matrix(np.ones((14, 14)) - np.eye(14))
    incoming = binary_incoming_union([a])
    features = sp.eye(14, format="csr")
    structural = simrank_cpu(incoming)
    first = select_all(incoming, features, structural)
    second = select_all(incoming, features, structural)
    for mode in MODES:
        assert (first[mode] != second[mode]).nnz == 0
        assert np.array_equal(np.diff(first[mode].indptr), np.full(14, 9))
    assert np.array_equal(first["cosine"].indices[first["cosine"].indptr[13]:], np.array([0, 1, 2, 3, 4, 5, 6, 7, 13]))
    # Tie ranking is deterministic and includes all candidates before selection.
    tests.append({"test": "fixed K=8, one self, candidate closure, deterministic ranking and node-id ties", "passed": True})
    # Malformed labels/splits cannot be read by the generation API. Changing
    # them does not affect any positive matrix or any declared input hash.
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source"
        data = source / "data" / "acm"
        data.mkdir(parents=True)
        for name in CONFIG["acm"]["mps"]:
            sp.save_npz(data / (name + ".npz"), a)
        sp.save_npz(data / "p_feat.npz", features)
        (data / "labels.npy").write_bytes(b"NOT A NUMPY FILE")
        (data / "test_20.npy").write_bytes(b"NOT A NUMPY FILE")
        raw1, f1, h1 = load_inputs(source, "acm")
        result1 = select_all(binary_incoming_union(raw1), f1, structural)
        (data / "labels.npy").write_bytes(b"DIFFERENT MALFORMED LABELS")
        (data / "test_20.npy").write_bytes(b"DIFFERENT MALFORMED SPLIT")
        raw2, f2, h2 = load_inputs(source, "acm")
        result2 = select_all(binary_incoming_union(raw2), f2, structural)
        assert h1 == h2 and all((result1[m] != result2[m]).nnz == 0 for m in MODES)
        assert all("label" not in name and "test" not in name for name in h1)
    tests.append({"test": "label/split isolation using unreadable files and changed bytes", "passed": True})
    # Zero-degree rows get only self; degree<8 budgets remain identical.
    incoming = binary_incoming_union([graphs[0]])
    structural = simrank_cpu(incoming)
    small = select_all(incoming, None, structural)
    expected_budget = 1 + np.minimum(8, np.diff(incoming.indptr))
    assert all(np.array_equal(np.diff(v.indptr), expected_budget) for v in small.values())
    assert small["cosine"][4].nnz == 1
    tests.append({"test": "isolate, low-degree equal budgets and identity cosine degeneration", "passed": True})
    summary = {"version": VERSION, "status": "passed", "cuda_checked": check_cuda, "tests": tests}
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "original/TriView-HCL-main")
    parser.add_argument("--output-root", type=Path, default=ROOT / "positive_extension/positives")
    parser.add_argument("--datasets", nargs="+", choices=CONFIG, default=list(CONFIG))
    parser.add_argument("--modes", nargs="+", choices=MODES)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--c", type=float, default=.9)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--random-seed", type=int, default=20261010)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    if args.self_test or args.test_only:
        tested = self_test(args.device == "cuda")
        args.output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_root / "mathematical_tests.json", tested)
    if args.test_only:
        return
    records = []
    for dataset in args.datasets:
        modes = args.modes or (list(MODES) if CONFIG[dataset]["feat"] else ["simrank", "random"])
        records.append(generate_dataset(args.source_root, args.output_root, dataset, modes,
            args.device, args.k, args.c, args.iterations, args.alpha, args.random_seed, args.block))
    atomic_json(args.output_root / "generation_manifest.json", {"version": VERSION, "datasets": records})


if __name__ == "__main__":
    main()
