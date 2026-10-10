"""Frozen-budget positive-generator study using the existing experiment API.

No old result files or the existing experiment.py are edited. Generation must
finish on an idle GPU before this runner starts. Four strategies on ACM/DBLP
and two structural strategies on identity-feature AMiner/Freebase: 36 encoders.
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from positive_pipeline import VERSION, CONFIG, MODES, atomic_json, csr_content_hash, sha256


ROOT = Path(__file__).resolve().parent
EXPECTED_FULL_LR = {"acm": 1., "dblp": 2., "aminer": 1., "freebase": 2.}
ENCODER_SEEDS = [1, 2, 3]


def read_final_lr(plan_path):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    lr = {}
    for ds in CONFIG:
        record = plan["primary"][ds]["full"]
        # Do not accept the stale local early snapshot with empty primary keys.
        if len(record.get("keys", [])) != 3:
            raise ValueError(f"Incomplete primary study plan for {ds}: use the final plan, not an early snapshot")
        lr[ds] = float(record["lrscale"])
    if lr != EXPECTED_FULL_LR:
        raise ValueError(f"Primary full-model learning rates differ from registered study: {lr}")
    return lr


def key(dataset, mode, seed, lr):
    return f"{dataset}__full__generated_{mode}__k8__lam0.5__lr{float(lr)}__seed{seed}"


def verify_manifest(positive_root, dataset, mode):
    directory = Path(positive_root) / dataset
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    fixed = {"version": VERSION, "k_nonself": 8, "c": .9,
             "iterations": 3, "alpha_cosine": .5, "random_seed": 20261010}
    for name, value in fixed.items():
        if manifest.get(name) != value:
            raise ValueError(f"Generator protocol differs: {name}={manifest.get(name)}")
    record = manifest["files"][mode]
    path = directory / record["file"]
    if sha256(path) != record["sha256"]:
        raise ValueError("Positive NPZ differs from its manifest")
    positive = sp.load_npz(path).tocsr()
    if csr_content_hash(positive) != record["csr_content_sha256"]:
        raise ValueError("Positive CSR values differ from their manifest")
    return manifest, positive, path


def execute_job(args):
    """Intercept only the existing static-positive hook; reuse encoder/probes."""
    import experiment as exp
    import torch

    lr = read_final_lr(args.study_plan)[args.dataset]
    manifest, positive, path = verify_manifest(args.positive_root, args.dataset, args.mode)
    root = Path(args.output_root)
    (root / "results").mkdir(parents=True, exist_ok=True)
    name = key(args.dataset, args.mode, args.seed, lr)
    result_path = root / "results" / (name + ".json")
    expected_npz_sha = manifest["files"][args.mode]["sha256"]
    if result_path.exists() and not args.force:
        record = json.loads(result_path.read_text(encoding="utf-8"))
        if (record.get("positive_generation", {}).get("positive_file_sha256") == expected_npz_sha
                and record.get("extension_protocol") == VERSION
                and record["args"]["epochs"] == 600):
            print("REUSE", name, flush=True)
            return name
        raise ValueError(f"Existing extension output has different/incomplete provenance: {name}; inspect before --force")
    # Imports resolve original modules/SRC before changing only the result root.
    # load_dataset itself is retained, and labels only reach the existing probes
    # and explicit post-training diagnostics. The generator never sees them.
    exp.ROOT = root
    original_static = exp.static_positive
    original_load = exp.load_dataset

    def checked_load(dataset):
        data = original_load(dataset)
        if dataset != args.dataset or positive.shape != data["pos"].shape:
            raise ValueError("Generated matrix does not match encoder dataset")
        # Verify generation inputs against the exact encoder graph/feature files.
        for filename, digest in manifest["input_sha256"].items():
            if sha256(exp.SRC / "data" / dataset / filename) != digest:
                raise ValueError("Generator/encoder source data mismatch")
        return data

    def fixed_positive(data, mode="structure", k=16):
        if mode != "generated_" + args.mode or k != 8:
            raise ValueError("Unexpected positive-generator hook invocation")
        return exp.pos_indices(positive)

    exp.load_dataset = checked_load
    exp.static_positive = fixed_positive
    runargs = SimpleNamespace(dataset=args.dataset, variant="full", positive="generated_" + args.mode,
        seed=args.seed, epochs=600, min_epochs=60, patience=40, probes=5,
        probe_epochs=200, block=args.block, k=8, warmup=60, refresh=40,
        lam=.5, lrscale=lr, force=args.force)
    try:
        exp.run(runargs)
    finally:
        exp.static_positive = original_static
        exp.load_dataset = original_load
    record = json.loads(result_path.read_text(encoding="utf-8"))
    record["extension_protocol"] = VERSION
    record["positive_generation"] = {
        "mode": args.mode, "manifest_file": str((Path(args.positive_root) / args.dataset / "manifest.json").resolve()),
        "manifest_sha256": sha256(Path(args.positive_root) / args.dataset / "manifest.json"),
        "positive_file": str(path.resolve()), "positive_file_sha256": expected_npz_sha,
        "positive_csr_content_sha256": manifest["files"][args.mode]["csr_content_sha256"],
        "candidate_rule": manifest["candidate_rule"], "budget": manifest["budget"],
        "k_nonself": 8, "c": .9, "iterations": 3, "alpha_cosine": .5,
        "fixed_for_all_epochs_and_encoder_seeds": True,
        "identity_feature_degeneracy": CONFIG[args.dataset]["feat"] is None,
    }
    record["timing_caveat"] = "Training/evaluation times may overlap independent jobs; use isolated generation profile for preprocessing cost. No inference-speed claim from this study."
    atomic_json(result_path, record)
    del record, positive
    gc.collect()
    torch.cuda.empty_cache()
    print("EXTENSION_DONE", name, flush=True)
    return name


def run_batch(args):
    lr = read_final_lr(args.study_plan)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "job_logs").mkdir(exist_ok=True)
    # Verify all matrices before dispatching any GPU job.
    specs = []
    for ds in args.datasets:
        modes = args.modes or (list(MODES) if CONFIG[ds]["feat"] else ["simrank", "random"])
        if CONFIG[ds]["feat"] is None and set(modes) & {"cosine", "combined"}:
            raise ValueError("Identity-feature graphs are restricted to SR/random in this study")
        for mode in modes:
            verify_manifest(args.positive_root, ds, mode)
            for seed in ENCODER_SEEDS:
                specs.append({"dataset": ds, "mode": mode, "seed": seed,
                              "lrscale": lr[ds], "key": key(ds, mode, seed, lr[ds])})
    plan = {
        "version": VERSION, "status": "running", "seeds": ENCODER_SEEDS,
        "primary_learning_rate_source": str(Path(args.study_plan).resolve()),
        "primary_learning_rate_source_sha256": sha256(args.study_plan),
        "primary_full_lrscale": lr, "jobs": specs, "workers": args.workers,
        "epochs_max": 600, "minimum_epochs": 60, "unsupervised_loss_patience": 40,
        "linear_probes_per_encoder": 5, "linear_probe_epochs": 200,
        "kmeans_repeats_per_encoder": 3, "kmeans_n_init": 10,
        "ranking_selection": "All rules fixed before results; no validation/test selection of positive strategies or generator hyperparameters.",
        "registration": "ACM/DBLP four modes; identity-feature AMiner/Freebase SR/random. This extends the study without replacing prior primary results.",
        "generation_profiling": "Already completed before batch; do not overlap generation and training.",
        "paired_seeds": "Encoder seeds and frozen positive files shared across compared modes; seed-dependent stochastic training streams may differ after divergent optimization histories.",
    }
    atomic_json(root / "study_plan.json", plan)

    def dispatch(spec):
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--job",
            "--dataset", spec["dataset"], "--mode", spec["mode"], "--seed", str(spec["seed"]),
            "--study-plan", str(Path(args.study_plan).resolve()), "--positive-root", str(Path(args.positive_root).resolve()),
            "--output-root", str(root.resolve()), "--block", str(args.block)]
        if args.force:
            command.append("--force")
        log = root / "job_logs" / (spec["key"] + ".log")
        print("EXTENSION_START", spec["key"], flush=True)
        with log.open("w", encoding="utf-8") as f:
            subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
        print("EXTENSION_FINISH", spec["key"], flush=True)
        return spec["key"]

    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        names = list(executor.map(dispatch, specs))
    # Manifest covers all raw per-seed JSON evidence and generation profiles;
    # results from the previous 153-run study remain in their original location.
    plan["status"] = "complete"
    plan["completed_keys"] = names
    plan["wall_seconds"] = time.perf_counter() - t
    atomic_json(root / "study_plan.json", plan)
    files = {}
    for path in sorted((root / "results").glob("*.json")):
        files[str(path.relative_to(root))] = sha256(path)
    for path in sorted(Path(args.positive_root).rglob("*.json")):
        files["positives/" + str(path.relative_to(args.positive_root)).replace("\\", "/")] = sha256(path)
    for path in sorted(Path(args.positive_root).rglob("*.npz")):
        files["positives/" + str(path.relative_to(args.positive_root)).replace("\\", "/")] = sha256(path)
    files["study_plan.json"] = sha256(root / "study_plan.json")
    atomic_json(root / "evidence_sha256.json", {"version": VERSION, "files": files})
    (root / "STUDY_COMPLETE").write_text("complete\n", encoding="utf-8")
    print("POSITIVE_STUDY_COMPLETE", len(names), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-plan", type=Path, default=ROOT / "study_plan.json")
    p.add_argument("--positive-root", type=Path, default=ROOT / "positive_extension/positives")
    p.add_argument("--output-root", type=Path, default=ROOT / "positive_extension")
    p.add_argument("--datasets", nargs="+", choices=CONFIG, default=list(CONFIG))
    p.add_argument("--modes", nargs="+", choices=MODES)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--force", action="store_true")
    p.add_argument("--job", action="store_true")
    p.add_argument("--dataset", choices=CONFIG, default="acm")
    p.add_argument("--mode", choices=MODES, default="simrank")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if args.workers < 1:
        p.error("workers must be positive")
    if args.job:
        # Cross-process lock permits a second dispatcher to assist with fixed
        # jobs while ensuring a queued job never trains the same encoder twice.
        import fcntl
        lockroot = args.output_root / 'locks'
        lockroot.mkdir(parents=True, exist_ok=True)
        lockpath = lockroot / f'{args.dataset}__{args.mode}__seed{args.seed}.lock'
        with lockpath.open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            execute_job(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
