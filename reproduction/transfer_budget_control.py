"""Three fixed-step DBLP-L target-repeat replay jobs; previous scores untouched.

The first phase is REUSED from a trusted saved retained checkpoint, not rerun.
Exactly one new 600-update target SSL phase is run per encoder seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import transfer_study as v3
import transfer_adaptation as adaptation

ROOT = Path(__file__).resolve().parent
PROTOCOL = "dblp_target_repeat_fixed600_followup_v1"
SEEDS = (1, 2, 3)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    import torch
    out = ROOT / "budget_control_results"; out.mkdir(exist_ok=True)
    key = f"{PROTOCOL}__dblp__full__seed{args.seed}"
    result_path = out / (key + ".json")
    if result_path.exists() and not args.force and not args.interface_only:
        print("EXISTS", key, flush=True); return
    plan = read(ROOT / "adaptation_results/study_plan.json")
    assert plan["status"] == "complete" and plan["protocol"] == adaptation.PROTOCOL
    base_key = f"{adaptation.PROTOCOL}__dblp__full__init_target__seed{args.seed}"
    base_path = ROOT / "adaptation_results" / (base_key + ".json")
    previous = read(base_path)
    assert previous["dataset"] == "dblp" and previous["variant"] == "full"
    assert previous["initial_graph"] == "target" and previous["encoder_seed"] == args.seed
    assert previous["no_encoder_labels"] is True
    checkpoint_relative = "adaptation_results/" + base_key + "__initial_retained.pt"
    checkpoint_path = ROOT / checkpoint_relative
    assert checkpoint_path.is_file()
    exp = v3.experiment_module(); exp.seed_all(args.seed); start = time.perf_counter()
    with v3.forbid_label_loading():
        source, target, manifest = v3.graph_pair("dblp")
        assert manifest == read(ROOT / "transfer_results/dblp_graph_manifest.json")
        assert previous["graph_split_hash"] == manifest["split_hash"]
        data = v3.device_graph(target, exp)
        retained = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert isinstance(retained, dict) and retained
        assert all(k.startswith(("fc.0.", "mp.")) for k in retained)
        assert adaptation.state_hash(retained) == previous["copy_verification"]["initial_retained_hash"]
        model, copying = adaptation.warmstart_target(retained, data, args, exp)
        copying["initial_graph"] = "target"
        assert copying["initial_retained_hash"] == previous["copy_verification"]["initial_retained_hash"]
        assert copying["target_aux_initial_hash"] == previous["copy_verification"]["target_aux_initial_hash"]
        assert copying["target_random_retained_hash"] == previous["copy_verification"]["target_random_retained_hash"]
        model, training, positive = adaptation.fit_target(model, data, args, exp)
        assert training["epochs"] == args.epochs
        torch.cuda.synchronize(); t = time.perf_counter()
        embedding, frozen_hash = v3.frozen_retained_embeddings(model, data)
        torch.cuda.synchronize(); inference_seconds = time.perf_counter() - t
        assert adaptation.positive_hash(positive) == previous["target_positive_hash"]
        if args.interface_only:
            print(json.dumps(dict(protocol=PROTOCOL, interface_only=True, labels_loaded=False,
                                  reused_first_phase=True, new_training_phases=1,
                                  actual_updates=training["epochs"], target_shape=list(embedding.shape),
                                  initial_checkpoint_hash_passed=True, copy_verification=copying,
                                  frozen_inference_auxiliary_hooks_passed=True), indent=2))
            return
    # Evaluation labels are first opened only after the new 600-update phase
    # and frozen inference. Old split metadata never influences optimization.
    labels = np.load(v3.source_root() / "data/dblp/labels.npy", allow_pickle=False).astype("int64").reshape(-1)
    labels = labels[target["node_ids"]["a"]]
    split, class_counts = v3.downstream_split(labels)
    split_hash = v3.json_hash({k: a.tolist() for k, a in split.items()})
    assert split_hash == previous["downstream_split_hash"]
    t = time.perf_counter()
    evaluation = exp.probe(embedding, labels, split, seed=args.seed,
                           repeats=args.probes, epochs=args.probe_epochs)
    result = dict(protocol=PROTOCOL, key=key, args=vars(args), dataset="dblp", variant="full",
                  encoder_seed=args.seed, initial_graph="target", base_protocol=adaptation.PROTOCOL,
                  hypothesis_timing="Budget fairness follow-up fixed after observing original adaptation results and unequal phase-2 early stopping; all three seeds reported",
                  old_target_repeat_key=base_key, old_target_repeat_json_sha256=sha(base_path),
                  initial_retained_checkpoint_file=checkpoint_relative,
                  initial_retained_checkpoint_sha256=sha(checkpoint_path),
                  initial_retained_checkpoint_hash_verified=True,
                  first_phase_reused=True, new_training_phases=1,
                  initial_training=previous["initial_training"], target_training=training,
                  graph_split_hash=manifest["split_hash"], source_graph=manifest["source"],
                  target_graph=manifest["target"], feature_boundary=manifest["target"]["feature_provenance"],
                  source_target_node_overlap=manifest["source_target_node_overlap"],
                  source_target_edge_overlap=manifest["source_target_edge_overlap"],
                  initial_positive_hash=previous["initial_positive_hash"],
                  target_positive_hash=adaptation.positive_hash(positive),
                  copy_verification=copying, no_encoder_labels=True,
                  frozen_inference_auxiliary_hooks_passed=True, frozen_encoder_hash=frozen_hash,
                  downstream_split_hash=split_hash, downstream_split={k: a.tolist() for k, a in split.items()},
                  downstream_split_sizes={k: len(a) for k, a in split.items()},
                  target_evaluation_class_counts=class_counts, classification={"20": evaluation},
                  label_use="Downstream probes only, after fixed600 target SSL and frozen inference",
                  budget_note="600 target optimizer updates matched to existing source-initialized target phases; first-phase epochs and graph costs still differ; minimum SSL checkpoint can precede update600",
                  original_target_repeat_phase2_epochs=previous["target_training"]["epochs"],
                  inference_seconds=inference_seconds, evaluation_seconds=time.perf_counter() - t,
                  total_seconds=time.perf_counter() - start,
                  environment=dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()))
    np.savez_compressed(out / (key + ".npz"), embedding=embedding.detach().cpu().numpy(),
                        positive_indices=positive["indices"], positive_mask=positive["mask"])
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("DONE", key, json.dumps(evaluation["mean"]), flush=True)


def batch(args):
    out = ROOT / "budget_control_results"; out.mkdir(exist_ok=True)
    logs = out / "job_logs"; logs.mkdir(exist_ok=True)
    prior = read(ROOT / "adaptation_results/study_plan.json")
    assert prior["status"] == "complete" and prior["protocol"] == adaptation.PROTOCOL
    jobs = [dict(dataset="dblp", variant="full", seed=s,
                 key=f"{PROTOCOL}__dblp__full__seed{s}") for s in SEEDS]
    plan = dict(protocol=PROTOCOL, base_protocol=adaptation.PROTOCOL, status="running",
                jobs=jobs, new_groups=3, new_training_phases=3, reused_initial_phases=3,
                seeds=list(SEEDS), probe_initializations_per_encoder=5, workers=args.workers,
                hypothesis_timing="New budget fairness follow-up after observed adaptation outcomes; no adaptation result is overwritten",
                target_epochs=600, target_min_epochs=600, patience=40, learning_rate=.0008,
                checkpoint="Minimum target SSL loss among the full600-update trajectory",
                old_adaptation_study_sha256=sha(ROOT / "adaptation_results/study_plan.json"),
                script_sha256={p: sha(ROOT / p) for p in ("transfer_budget_control.py", "budget_control_protocol.txt", "transfer_adaptation.py", "transfer_study.py")},
                execution="Two independent accuracy workers; not an isolated latency benchmark")
    def save():
        path = out / "study_plan.json"; tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(plan, indent=2), encoding="utf-8"); tmp.replace(path)
    save()
    def execute(job):
        result_path = out / (job["key"] + ".json")
        if result_path.exists():
            value = read(result_path)
            assert value["protocol"] == PROTOCOL and value["encoder_seed"] == job["seed"]
            assert value["target_training"]["epochs"] == 600 and value["first_phase_reused"] is True
            print("BUDGET_REUSE", job["key"], flush=True); return job["key"]
        command = [sys.executable, "-u", str(ROOT / "transfer_budget_control.py"), "--seed", str(job["seed"])]
        print("BUDGET_START", job["key"], flush=True)
        with (logs / (job["key"] + ".log")).open("w", encoding="utf-8") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print("BUDGET_FINISH", job["key"], flush=True)
        return job["key"]
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        completed = list(executor.map(execute, jobs))
    plan.update(status="complete", completed_keys=completed, wall_seconds=time.perf_counter() - t)
    save()
    (out / "evidence_sha256.json").write_text(json.dumps(dict(protocol=PROTOCOL, files={p.name: sha(p) for p in sorted(out.glob("*.json"))}), indent=2), encoding="utf-8")
    (out / "STUDY_COMPLETE").write_text("complete\n", encoding="utf-8")
    print("BUDGET_STUDY_COMPLETE", len(completed), "groups and three NEW target phases", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("dblp",), default="dblp")
    parser.add_argument("--variant", choices=("full",), default="full")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=1)
    parser.add_argument("--initial-graph", choices=("target",), default="target")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--min-epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--probes", type=int, default=5)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--interface-only", action="store_true")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    assert args.epochs == args.min_epochs
    if not args.interface_only:
        assert args.epochs == 600 and args.probes == 5
    batch(args) if args.batch else run(args)
