"""Pre-registered, held-out-graph transfer study for TriView-HCL.

The two graphs are reconstructed from raw typed relations, not slices of the
released metapath matrices.  CPU graph preparation never opens labels or public
label splits.  Encoder optimization accepts only a label-free graph dictionary.
Target labels are loaded only after optimization and frozen inference finish.

Run --audit first; then one --dataset/--variant/--mode/--seed job at a time.
This is within-dataset held-out-graph transfer, not cross-dataset transfer.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
SPLIT_SEED = 20261010
PROTOCOL = "typed_owner_disjoint_local_features_v3"
POSITIVE_K = 8
DATA = {
    "acm": {
        "types": {"p": 4019, "a": 7167, "s": 60},
        "target": "p", "neighbor_types": ["a", "s"],
        "relations": {"pa": ("p", "a"), "ps": ("p", "s")},
        "paths": ["pap", "psp"], "feature": "p_feat",
    },
    "dblp": {
        "types": {"a": 4057, "p": 14328, "c": 20, "t": 7723},
        "target": "a", "neighbor_types": ["p"],
        "relations": {"pa": ("p", "a"), "pc": ("p", "c"), "pt": ("p", "t")},
        "paths": ["apa", "apcpa", "aptpa"], "feature": "a_feat",
    },
}


def source_root():
    for p in (ROOT / "original/TriView-HCL-main", ROOT / "source/TriView-HCL-main"):
        if (p / "data/acm/pa.txt").is_file():
            return p
    raise FileNotFoundError("Cannot locate the supplied TriView-HCL raw relations")


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def array_hash(x):
    x = np.ascontiguousarray(x)
    return sha_bytes(str(x.dtype).encode() + repr(x.shape).encode() + x.tobytes())


def sparse_hash(x):
    x = x.copy().tocsr(); x.sum_duplicates(); x.sort_indices()
    return sha_bytes(repr(x.shape).encode() + x.indptr.tobytes() + x.indices.tobytes() + x.data.tobytes())


def json_hash(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def binary(x):
    x = x.tocsr().astype(bool).astype("float32")
    x.eliminate_zeros(); x.sort_indices()
    return x


def load_relations(dataset):
    spec = DATA[dataset]; folder = source_root() / "data" / dataset
    edges = {}
    for name, (left, right) in spec["relations"].items():
        e = np.loadtxt(folder / (name + ".txt"), dtype="int64", ndmin=2)
        if e.ndim != 2 or e.shape[1] != 2 or np.any(e < 0):
            raise ValueError("Invalid raw relation " + name)
        if e[:, 0].max() >= spec["types"][left] or e[:, 1].max() >= spec["types"][right]:
            raise ValueError("Out-of-range original node identifier")
        edges[name] = np.unique(e, axis=0)
    return edges


def owners_for(dataset):
    # Exactly reproduce the initial feasibility audit's independent Bernoulli
    # owners, before any encoder/test performance was observed.  This order is
    # a protocol constant, not Python's process-dependent hash ordering.
    rng = np.random.default_rng(SPLIT_SEED)
    result = {}
    for kind, size in DATA[dataset]["types"].items():
        result[kind] = rng.integers(0, 2, size)
    return result


def target_neighbor_ids(spec, edges, kind, target_ids):
    target = spec["target"]
    for relation, (left, right) in spec["relations"].items():
        if (left, right) == (target, kind):
            e = edges[relation]
            return e[np.isin(e[:, 0], target_ids)], 0, 1
        if (left, right) == (kind, target):
            e = edges[relation]
            return e[np.isin(e[:, 1], target_ids)], 1, 0
    raise ValueError("Missing required direct neighbor relation")


def rebuild_graph(dataset, owner_id, owners, original):
    spec = DATA[dataset]; target = spec["target"]
    assigned = {kind: np.flatnonzero(v == owner_id) for kind, v in owners.items()}
    retained = {}
    for name, (left, right) in spec["relations"].items():
        e = original[name]
        retained[name] = e[(owners[left][e[:, 0]] == owner_id) & (owners[right][e[:, 1]] == owner_id)]
    # Models sample required direct NG neighbors; delete targets with an empty
    # required neighbor set, using structure only.  Do not retry the split.
    keep = assigned[target].copy()
    for kind in spec["neighbor_types"]:
        e, target_col, _ = target_neighbor_ids(spec, retained, kind, keep)
        keep = np.intersect1d(keep, np.unique(e[:, target_col]))
    node_ids = {kind: ids.copy() for kind, ids in assigned.items()}
    node_ids[target] = keep
    # Prune removed target incidence; then unused auxiliary nodes.  DBLP venue
    # and term edges are also restricted to surviving owner-specific papers.
    for name, (left, right) in spec["relations"].items():
        e = retained[name]
        if left == target: e = e[np.isin(e[:, 0], keep)]
        if right == target: e = e[np.isin(e[:, 1], keep)]
        retained[name] = e
    for kind in spec["neighbor_types"]:
        e, _, neighbor_col = target_neighbor_ids(spec, retained, kind, keep)
        node_ids[kind] = np.unique(e[:, neighbor_col])
    # This also filters pc and pt by surviving papers before collecting c/t IDs.
    for name, (left, right) in spec["relations"].items():
        e = retained[name]
        e = e[np.isin(e[:, 0], node_ids[left]) & np.isin(e[:, 1], node_ids[right])]
        retained[name] = e
    for kind in node_ids:
        if kind == target or kind in spec["neighbor_types"]: continue
        adjacent = []
        for name, (left, right) in spec["relations"].items():
            if left == kind: adjacent.append(retained[name][:, 0])
            if right == kind: adjacent.append(retained[name][:, 1])
        node_ids[kind] = np.unique(np.concatenate(adjacent)) if adjacent else np.array([], dtype="int64")
    matrices = {}
    for name, (left, right) in spec["relations"].items():
        e = retained[name]
        rows = np.searchsorted(node_ids[left], e[:, 0]); cols = np.searchsorted(node_ids[right], e[:, 1])
        matrices[name] = sp.csr_matrix((np.ones(len(e), dtype="float32"), (rows, cols)),
                                     shape=(len(node_ids[left]), len(node_ids[right])))
        assert np.array_equal(node_ids[left][rows], e[:, 0])
        assert np.array_equal(node_ids[right][cols], e[:, 1])
    if dataset == "acm":
        pa, ps = matrices["pa"], matrices["ps"]
        raw = [binary(pa @ pa.T), binary(ps @ ps.T)]
        direct = [pa, ps]
    else:
        pa, pc, pt = matrices["pa"], matrices["pc"], matrices["pt"]
        apc = pa.T @ pc; apt = pa.T @ pt
        raw = [binary(pa.T @ pa), binary(apc @ apc.T), binary(apt @ apt.T)]
        direct = [pa.T.tocsr()]
    neis = [[a.indices[a.indptr[i]:a.indptr[i + 1]].copy() for i in range(a.shape[0])] for a in direct]
    assert all(len(v) > 0 for group in neis for v in group)
    if dataset == "dblp":
        # DBLP-L: a_feat has undocumented upstream aggregation.  Never load it.
        # Rebuild authors from the support union of only retained local papers,
        # using a fixed shared 4231-column paper vocabulary.  No source author
        # receives any feature contribution from a target or removed paper.
        paper_x = sp.load_npz(source_root() / "data" / dataset / "p_feat.npz").tocsr().astype("float32")
        paper_x = binary(paper_x[node_ids["p"]])
        x = binary(matrices["pa"].T @ paper_x)
        feature_provenance = "DBLP-L: author attributes rebuilt as the binary support union of supplied paper-word features over retained graph-local PA edges, then row-normalized. Fixed shared 4231-column paper vocabulary; released a_feat is never loaded. Paper-word vocabulary provenance is fixed supplied input, and no claim of learning the vocabulary using source data only is made."
        feature_formula = "row_normalize(1[PA_local.T @ 1[P_feat_local != 0] > 0])"
        contributor_hash = array_hash(node_ids["p"])
    else:
        x = sp.load_npz(source_root() / "data" / dataset / (spec["feature"] + ".npz")).tocsr().astype("float32")[keep]
        feature_provenance = "ACM: fixed supplied paper-word attributes, row-normalized independently per retained node. Shared feature-column semantics are fixed input; upstream construction of the supplied paper vocabulary is not newly inferred from source-only data."
        feature_formula = "row_normalize(P_feat[retained_graph_papers])"
        contributor_hash = array_hash(keep)
    degree = np.asarray(x.sum(1)).ravel()
    inverse = np.divide(1., degree, out=np.zeros_like(degree), where=degree != 0)
    x = (sp.diags(inverse) @ x).tocsr().astype("float32")
    manifest = {
        "side": "source" if owner_id == 0 else "target", "owner_id": owner_id,
        "assigned_node_counts": {k: len(v) for k, v in assigned.items()},
        "retained_node_counts": {k: len(v) for k, v in node_ids.items()},
        "target_retention_from_assigned": len(keep) / len(assigned[target]),
        "original_node_ids": {k: v.tolist() for k, v in node_ids.items()},
        "node_id_hashes": {k: array_hash(v) for k, v in node_ids.items()},
        "raw_relation_edge_counts": {k: len(v) for k, v in retained.items()},
        "raw_relation_edge_hashes": {k: array_hash(v) for k, v in retained.items()},
        "metapath_counts": {p: int(a.nnz) for p, a in zip(spec["paths"], raw)},
        "metapath_hashes": {p: sparse_hash(a) for p, a in zip(spec["paths"], raw)},
        "feature_hash": sparse_hash(x), "feature_dimension": x.shape[1],
        "feature_provenance": feature_provenance,
        "feature_formula": feature_formula,
        "feature_contributor_paper_id_hash": contributor_hash,
        "feature_contributors_within_graph_owner": True,
        "positive_policy": "Sum symmetrically normalized graph-local binary metapath adjacency; rank off-diagonal neighbors, at most 8; include self; all variants use the identical fixed positives.",
    }
    graph = dict(dataset=dataset, spec=spec, node_ids=node_ids, edges=retained,
                 matrices=matrices, raw=raw, features=x, neis=neis, manifest=manifest)
    validate_projection(graph)
    manifest["projection_validation"] = "Every sampled row verified by independent raw-relation path traversal, without released metapath/positive matrices."
    return graph


def validate_projection(graph):
    # Independently enumerate a small deterministic row sample through the typed
    # relations.  This detects direction/remapping errors in rebuilt projections.
    spec = graph["spec"]; node_ids = graph["node_ids"]
    adjacent = {}
    for name, (left, right) in spec["relations"].items():
        e = graph["edges"][name]
        for a, b, left_col, right_col in [(left, right, 0, 1), (right, left, 1, 0)]:
            index = {}
            for u, v in e[:, [left_col, right_col]]:
                index.setdefault(int(u), set()).add(int(v))
            adjacent[(a, b)] = index
    n = len(node_ids[spec["target"]])
    rows = np.unique(np.linspace(0, n - 1, min(12, n), dtype=int))
    for path, raw in zip(spec["paths"], graph["raw"]):
        for row in rows:
            current = {int(node_ids[spec["target"]][row])}
            for a, b in zip(path, path[1:]):
                current = set().union(*(adjacent[(a, b)].get(u, set()) for u in current)) if current else set()
            actual = set(node_ids[spec["target"]][raw.indices[raw.indptr[row]:raw.indptr[row + 1]]].tolist())
            assert current == actual, (path, int(row), len(current), len(actual))


def graph_pair(dataset):
    original = load_relations(dataset); owners = owners_for(dataset)
    source = rebuild_graph(dataset, 0, owners, original)
    target = rebuild_graph(dataset, 1, owners, original)
    node_overlap = {k: int(len(np.intersect1d(source["node_ids"][k], target["node_ids"][k]))) for k in owners}
    edge_overlap = {}
    for name, (left, right) in DATA[dataset]["relations"].items():
        a, b = source["edges"][name], target["edges"][name]
        edge_overlap[name] = len(set(map(tuple, a.tolist())) & set(map(tuple, b.tolist())))
        for side, graph in enumerate((source, target)):
            e = graph["edges"][name]
            assert np.all(owners[left][e[:, 0]] == side) and np.all(owners[right][e[:, 1]] == side)
    assert not any(node_overlap.values()) and not any(edge_overlap.values())
    for graph in (source, target):
        assert graph["features"].shape[1] == source["features"].shape[1]
    manifest = dict(protocol=PROTOCOL, dataset=dataset, split_seed=SPLIT_SEED,
                    owner_algorithm="Single default_rng(20261010), rng.integers(0,2,size) per type in DATA insertion order; independent Bernoulli .5 owners; source=0,target=1; no split retry",
                    owner_hashes={k: array_hash(v) for k, v in owners.items()},
                    original_relation_hashes={k: array_hash(v) for k, v in original.items()},
                    original_relation_counts={k: len(v) for k, v in original.items()},
                    source_target_node_overlap=node_overlap, source_target_edge_overlap=edge_overlap,
                    source=source["manifest"], target=target["manifest"],
                    limitation="Within one dataset and fixed attribute schema; all typed nodes and raw edges are disjoint. Source/target class distributions can differ. DBLP-L uses graph-local rebuilt 4231-column author attributes and is not numerically comparable to the original 334-column DBLP main-table protocol. The fixed supplied paper-word vocabulary is shared. This is not cross-dataset transfer or a graph foundation model test.")
    manifest["split_hash"] = json_hash(manifest)
    return source, target, manifest


def downstream_split(labels):
    # Only called AFTER encoder optimization and frozen target inference.
    rng = np.random.default_rng(SPLIT_SEED + 9000)
    parts = {k: [] for k in ("train", "val", "test")}
    counts = {}
    for c in np.unique(labels):
        ids = rng.permutation(np.flatnonzero(labels == c))
        counts[str(int(c))] = len(ids)
        if len(ids) < 31:
            raise ValueError("A target class has fewer than 31 nodes; fixed split is not evaluable, and must not be retried")
        parts["train"].extend(ids[:20]); parts["val"].extend(ids[20:30]); parts["test"].extend(ids[30:])
    parts = {k: np.asarray(v, dtype="int64") for k, v in parts.items()}
    assert sum(len(v) for v in parts.values()) == len(labels)
    assert len(np.unique(np.concatenate(list(parts.values())))) == len(labels)
    return parts, counts


@contextlib.contextmanager
def forbid_label_loading():
    original = np.load
    def guarded(file, *args, **kwargs):
        name = Path(str(file)).name.lower()
        if name == "labels.npy" or name.startswith(("train_", "val_", "test_")):
            raise AssertionError("Encoder code attempted to open evaluation labels/splits")
        return original(file, *args, **kwargs)
    np.load = guarded
    try: yield
    finally: np.load = original


def experiment_module():
    # The local archive is called source/, while the remote experiment package
    # uses original/.  Existing experiment.py is left unchanged.
    sys.path.insert(0, str(source_root() / "code"))
    exp = importlib.import_module("experiment")
    exp.SRC = source_root()
    return exp


def device_graph(graph, exp):
    cfg = copy.deepcopy(exp.CONFIG[graph["dataset"]])
    spec = graph["spec"]
    cfg["types"] = [len(graph["node_ids"][spec["target"]])] + [len(graph["node_ids"][t]) for t in spec["neighbor_types"]]
    result = dict(name=graph["dataset"], cfg=cfg,
                  feats=[exp.sparse_tensor(graph["features"])] + [None] * len(spec["neighbor_types"]),
                  dims=[graph["features"].shape[1]] + cfg["types"][1:],
                  mps=[exp.sparse_tensor(exp.normalized_adj(a)) for a in graph["raw"]],
                  raw=graph["raw"], neis=graph["neis"])
    assert not ({"y", "labels", "splits", "pos"} & result.keys())
    return result


def model_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def frozen_retained_embeddings(model, target_data):
    # Guard against accidentally invoking source-size SM/NG or any source-only
    # identity projection on new graph IDs.  Only fc[0] and MP are deployed.
    def forbidden(module, args):
        raise AssertionError("Frozen target inference called a discarded auxiliary module")
    handles = [m.register_forward_pre_hook(forbidden) for m in [model.ng, model.sm, *list(model.fc)[1:]]]
    old_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters(): p.requires_grad_(False)
    before = model_hash(model); model.eval()
    try:
        embedding = model.embeds({"feats": [target_data["feats"][0]], "mps": target_data["mps"]})
        assert not embedding.requires_grad and before == model_hash(model)
    finally:
        for h in handles: h.remove()
        for p, flag in zip(model.parameters(), old_flags): p.requires_grad_(flag)
    return embedding, before


def fit_encoder(data, args, exp):
    import torch
    assert not ({"y", "labels", "splits", "pos"} & data.keys())
    model = exp.Model(data, args.variant, block=args.block, lam=.5).cuda()
    idx, mask = exp.static_positive(data, "structure", POSITIVE_K)
    optimizer = torch.optim.Adam(model.parameters(), lr=data["cfg"]["lr"], weight_decay=0.)
    best = float("inf"); state = None; wait = 0; history = []; best_epoch = 0
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad(); loss = model(data, idx, mask)
        if not torch.isfinite(loss): raise ValueError("Nonfinite SSL objective")
        value = float(loss.item()); history.append(value)
        if value < best:
            best = value; best_epoch = epoch; wait = 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: wait += 1
        loss.backward(); optimizer.step()
        if epoch % 50 == 0: print("TRAIN", args.dataset, args.variant, args.mode, args.seed, epoch, value, flush=True)
        if epoch + 1 >= args.min_epochs and wait >= args.patience: break
    torch.cuda.synchronize(); elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    model.load_state_dict(state); model.eval()
    positive = dict(indices=idx.cpu().numpy(), mask=mask.cpu().numpy())
    return model, dict(epochs=len(history), best_epoch=best_epoch, best_loss=best,
                       train_seconds=elapsed, peak_training_mib=peak, loss_history=history,
                       learning_rate=data["cfg"]["lr"], learning_rate_selection="Original fixed base rate; no source/target label tuning and no full-graph selected rates",
                       checkpoint_selection="Minimum pre-update training SSL loss, never target labels"), positive


def run(args):
    import torch
    exp = experiment_module(); exp.seed_all(args.seed)
    out = ROOT / "transfer_results"; out.mkdir(exist_ok=True)
    key = f"{PROTOCOL}__{args.dataset}__{args.variant}__{args.mode}__seed{args.seed}"
    path = out / (key + ".json")
    if path.exists() and not args.force: print("EXISTS", key, flush=True); return
    start = time.perf_counter()
    with forbid_label_loading():
        source, target, manifest = graph_pair(args.dataset)
        manifest_path = out / (args.dataset + "_graph_manifest.json")
        if manifest_path.exists():
            assert json.loads(manifest_path.read_text(encoding="utf-8"))["split_hash"] == manifest["split_hash"]
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        pretraining = source if args.mode in ("source_transfer", "random_encoder") else target
        train_data = device_graph(pretraining, exp); target_data = device_graph(target, exp)
        if args.mode == "raw_features":
            z = exp.sparse_tensor(target["features"]).to_dense()
            training = {"epochs": 0, "train_seconds": 0, "description": "Row-normalized fixed supplied node attributes; no encoder"}
            positive = {}; frozen_hash = None
        elif args.mode == "random_encoder":
            model = exp.Model(train_data, "full", block=args.block, lam=.5).cuda()
            z, frozen_hash = frozen_retained_embeddings(model, target_data)
            training = {"epochs": 0, "train_seconds": 0, "description": "Untrained target projection and max-summary MP encoder; identical seed and source-sized auxiliary initialization"}
            positive = {}
        else:
            model, training, positive = fit_encoder(train_data, args, exp)
            torch.cuda.synchronize(); inference_start = time.perf_counter()
            z, frozen_hash = frozen_retained_embeddings(model, target_data)
            torch.cuda.synchronize(); training["frozen_target_inference_seconds"] = time.perf_counter() - inference_start
    # Labels are first loaded here, after training and frozen inference.  They
    # cannot influence owners, filtering, graph projections, positives, or SSL.
    all_labels = np.load(source_root() / "data" / args.dataset / "labels.npy", allow_pickle=False).astype("int64").reshape(-1)
    target_labels = all_labels[target["node_ids"][DATA[args.dataset]["target"]]]
    split, class_counts = downstream_split(target_labels)
    evaluation_start = time.perf_counter()
    evaluation = exp.probe(z, target_labels, split, seed=args.seed, repeats=args.probes, epochs=args.probe_epochs)
    # Standard exp.probe reports all metrics at one validation-CE-selected epoch.
    result = dict(protocol=PROTOCOL, key=key, args=vars(args), dataset=args.dataset,
                  mode=args.mode, variant=args.variant, encoder_seed=args.seed,
                  split_hash=manifest["split_hash"], graph_manifest_file=args.dataset + "_graph_manifest.json",
                  pretraining_side=pretraining["manifest"]["side"] if args.mode not in ("raw_features", "random_encoder") else "none",
                  pretraining_graph=pretraining["manifest"], target_graph=target["manifest"],
                  source_target_node_overlap=manifest["source_target_node_overlap"], source_target_edge_overlap=manifest["source_target_edge_overlap"],
                  target_evaluation_class_counts=class_counts,
                  source_evaluation_class_counts={str(int(c)): int(n) for c, n in zip(*np.unique(all_labels[source["node_ids"][DATA[args.dataset]["target"]]], return_counts=True))},
                  downstream_split={k: v.tolist() for k, v in split.items()},
                  downstream_split_hash=json_hash({k: v.tolist() for k, v in split.items()}),
                  downstream_split_sizes={k: len(v) for k, v in split.items()},
                  classification={"20": evaluation}, training=training, frozen_encoder_hash=frozen_hash,
                  no_encoder_labels=True, frozen_inference_auxiliary_hooks_passed=args.mode != "raw_features",
                  label_use="Only downstream 20-per-class train / 10-per-class validation / remaining-node test; no encoder training or checkpoint selection",
                  feature_boundary=target["manifest"]["feature_provenance"],
                  evaluation_seconds=time.perf_counter() - evaluation_start,
                  total_seconds=time.perf_counter() - start,
                  environment=dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()))
    if positive:
        result["positive_hash"] = json_hash({k: array_hash(v) for k, v in positive.items()})
        result["positive_counts_min_mean_max"] = [int(positive["mask"].sum(1).min()), float(positive["mask"].sum(1).mean()), int(positive["mask"].sum(1).max())]
    np.savez_compressed(out / (key + ".npz"), embedding=z.detach().cpu().numpy(), **positive)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("DONE", key, json.dumps(evaluation["mean"]), flush=True)


def audit(dataset):
    with forbid_label_loading():
        _, _, manifest = graph_pair(dataset)
    folder = ROOT / "transfer_results"; folder.mkdir(exist_ok=True)
    (folder / (dataset + "_graph_manifest.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(dict(dataset=dataset, split_hash=manifest["split_hash"],
                         source_counts=manifest["source"]["retained_node_counts"],
                         target_counts=manifest["target"]["retained_node_counts"],
                         node_overlap=manifest["source_target_node_overlap"], edge_overlap=manifest["source_target_edge_overlap"],
                         feature_dimension=manifest["source"]["feature_dimension"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATA, default="acm")
    parser.add_argument("--variant", choices=["full", "heco", "mp_aug"], default="full")
    parser.add_argument("--mode", choices=["source_transfer", "target_only_reference", "random_encoder", "raw_features"], default="source_transfer")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--probes", type=int, default=5)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    audit(args.dataset) if args.audit else run(args)
