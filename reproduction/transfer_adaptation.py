"""Secondary follow-up: transfer retained MP/attribute weights, then target SSL.

This hypothesis follows the completed frozen-transfer study. Its eighteen jobs
are fixed before their own test outcomes are available; original v3 scores are
preserved. No old v3 script or result is modified.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import transfer_study as v3

ROOT = Path(__file__).resolve().parent
PROTOCOL = "retained_warmstart_followup_v2"
VARIANTS = ("full", "heco")
DATASETS = ("acm", "dblp")
SEEDS = (1, 2, 3)


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_state(model, retained=True):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if k.startswith(("fc.0.", "mp.")) == retained}


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        a = value.detach().cpu().numpy()
        h.update(key.encode()); h.update(str(a.dtype).encode())
        h.update(repr(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


@contextlib.contextmanager
def preserved_rng(torch):
    py = random.getstate(); npstate = np.random.get_state()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        try: yield
        finally:
            random.setstate(py); np.random.set_state(npstate)


def warmstart_target(retained, data, args, exp):
    """Create v3 target-only initialization, verify auxiliary equality, copy.

    This function returns with RNG streams exactly where the target-only job
    would be immediately after its model initialization. Weight copying and
    independent initialization verification consume no subsequent random draws.
    """
    import torch
    exp.seed_all(args.seed)
    target_model = exp.Model(data, args.variant, block=args.block, lam=.5).cuda()
    initial_retained_hash = state_hash(selected_state(target_model, True))
    initial_aux_hash = state_hash(selected_state(target_model, False))
    rng_before = torch.get_rng_state().clone()
    cuda_before = [s.clone() for s in torch.cuda.get_rng_state_all()]
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    with preserved_rng(torch):
        exp.seed_all(args.seed)
        reference = exp.Model(data, args.variant, block=args.block, lam=.5).cuda()
        reference_aux_hash = state_hash(selected_state(reference, False))
        reference_retained_hash = state_hash(selected_state(reference, True))
        del reference
    assert initial_aux_hash == reference_aux_hash
    assert initial_retained_hash == reference_retained_hash
    assert torch.equal(rng_before, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
    after_numpy = np.random.get_state()
    assert numpy_before[0] == after_numpy[0] and np.array_equal(numpy_before[1], after_numpy[1])
    assert numpy_before[2:] == after_numpy[2:]
    assert python_before == random.getstate()
    # strict=False permits only the intentionally fresh auxiliary modules.
    result = target_model.load_state_dict(retained, strict=False)
    assert not result.unexpected_keys
    assert all(not k.startswith(("fc.0.", "mp.")) for k in result.missing_keys)
    copied_hash = state_hash(selected_state(target_model, True))
    source_hash = state_hash(retained)
    assert copied_hash == source_hash
    assert state_hash(selected_state(target_model, False)) == initial_aux_hash
    assert torch.equal(rng_before, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
    assert python_before == random.getstate()
    after_numpy = np.random.get_state()
    assert np.array_equal(numpy_before[1], after_numpy[1]) and numpy_before[2:] == after_numpy[2:]
    evidence = dict(initial_retained_hash=source_hash, copied_target_retained_hash=copied_hash,
                    target_random_retained_hash=initial_retained_hash,
                    target_aux_initial_hash=initial_aux_hash, target_only_reference_aux_initial_hash=reference_aux_hash,
                    target_only_reference_retained_initial_hash=reference_retained_hash,
                    copied_parameter_keys=sorted(retained), auxiliary_fresh=True,
                    copied_retained_state_matches_initial=True, auxiliary_init_matches_target_only=True,
                    rng_after_copy_matches_target_only=True,
                    transferred="Only fc[0] and MP encoder; NG, SM, ContraNorm/LayerNorm, auxiliary identity projections, contrastive projector, and Adam are freshly initialized for target graph")
    return target_model, evidence


def fit_target(model, data, args, exp):
    """Same fixed target SSL budget and optimizer as v3, with warm-start weights."""
    import torch
    assert not ({"y", "labels", "splits", "pos"} & data.keys())
    idx, mask = exp.static_positive(data, "structure", v3.POSITIVE_K)
    optimizer = torch.optim.Adam(model.parameters(), lr=data["cfg"]["lr"], weight_decay=0.)
    best = float("inf"); state = None; wait = 0; history = []; best_epoch = 0
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad(); loss = model(data, idx, mask)
        if not torch.isfinite(loss): raise ValueError("Nonfinite target SSL objective")
        value = float(loss.item()); history.append(value)
        if value < best:
            best = value; best_epoch = epoch; wait = 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: wait += 1
        loss.backward(); optimizer.step()
        if epoch % 50 == 0:
            print("TARGET_ADAPTATION", args.dataset, args.variant, args.seed, epoch, value, flush=True)
        if epoch + 1 >= args.min_epochs and wait >= args.patience: break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start; peak = torch.cuda.max_memory_allocated() / 2 ** 20
    model.load_state_dict(state); model.eval()
    positive = dict(indices=idx.cpu().numpy(), mask=mask.cpu().numpy())
    training = dict(epochs=len(history), best_epoch=best_epoch, best_loss=best,
                    train_seconds=elapsed, peak_training_mib=peak, loss_history=history,
                    learning_rate=data["cfg"]["lr"],
                    learning_rate_selection="Original fixed base rate .0008; no additional tuning",
                    checkpoint_selection="Minimum target-only phase SSL training loss, never labels",
                    optimizer="Fresh Adam; no source optimizer state",
                    extra_initial_phase_budget=True)
    return model, training, positive


def positive_hash(positive):
    return v3.json_hash({k: v3.array_hash(a) for k, a in positive.items()})


def validate_previous(dataset, variant, seed, manifest):
    folder = ROOT / "transfer_results"
    plan = json_read(folder / "study_plan.json")
    assert plan["status"] == "complete" and plan["protocol"] == v3.PROTOCOL
    reference_key = f"{v3.PROTOCOL}__{dataset}__{variant}__target_only_reference__seed{seed}"
    source_key = f"{v3.PROTOCOL}__{dataset}__{variant}__source_transfer__seed{seed}"
    reference = json_read(folder / (reference_key + ".json"))
    source = json_read(folder / (source_key + ".json"))
    for result in (reference, source):
        assert result["split_hash"] == manifest["split_hash"]
        assert result["args"]["epochs"] == 600 and result["args"]["probes"] == 5
        assert result["no_encoder_labels"] is True
        assert result["training"]["learning_rate"] == .0008
    assert reference["downstream_split_hash"] == source["downstream_split_hash"]
    return reference, source


def run(args):
    import torch
    out = ROOT / "adaptation_results"; out.mkdir(exist_ok=True)
    key = f"{PROTOCOL}__{args.dataset}__{args.variant}__init_{args.initial_graph}__seed{args.seed}"
    result_path = out / (key + ".json")
    if result_path.exists() and not args.force and not args.interface_only:
        print("EXISTS", key, flush=True); return
    exp = v3.experiment_module(); exp.seed_all(args.seed); start = time.perf_counter()
    with v3.forbid_label_loading():
        source, target, manifest = v3.graph_pair(args.dataset)
        previous_manifest = json_read(ROOT / "transfer_results" / (args.dataset + "_graph_manifest.json"))
        assert manifest == previous_manifest
        reference, old_source = validate_previous(args.dataset, args.variant, args.seed, manifest)
        source_data = v3.device_graph(source, exp); target_data = v3.device_graph(target, exp)
        # Exactly repeat the fixed source phase; old experiment saved embeddings,
        # not model weights. No downstream probe is run for the repeated phase.
        initial_data = source_data if args.initial_graph == "source" else target_data
        old_initial = old_source if args.initial_graph == "source" else reference
        first_phase_args = copy.copy(args)
        first_phase_args.mode = "source_transfer" if args.initial_graph == "source" else "target_only_reference"
        source_model, source_training, source_positive = v3.fit_encoder(initial_data, first_phase_args, exp)
        source_model_hash = v3.model_hash(source_model)
        retained = selected_state(source_model, True)
        retained_source_hash = state_hash(retained)
        del source_model
        target_model, copying = warmstart_target(retained, target_data, args, exp)
        assert copying["initial_retained_hash"] == retained_source_hash
        copying["initial_graph"] = args.initial_graph
        if not args.interface_only:
            torch.save(retained, out / (key + "__initial_retained.pt"))
        target_model, target_training, target_positive = fit_target(target_model, target_data, args, exp)
        torch.cuda.synchronize(); t = time.perf_counter()
        embedding, frozen_hash = v3.frozen_retained_embeddings(target_model, target_data)
        torch.cuda.synchronize(); inference_seconds = time.perf_counter() - t
        assert embedding.shape[0] == len(target["node_ids"][v3.DATA[args.dataset]["target"]])
        assert positive_hash(source_positive) == old_initial["positive_hash"]
        assert positive_hash(target_positive) == reference["positive_hash"]
        if args.interface_only:
            print(json.dumps(dict(protocol=PROTOCOL, interface_only=True, labels_loaded=False,
                                  dataset=args.dataset, variant=args.variant, initial_graph=args.initial_graph,
                                  source_epochs=source_training["epochs"], target_epochs=target_training["epochs"],
                                  copy_checks=copying, frozen_inference_auxiliary_hooks_passed=True,
                                  frozen_encoder_hash=frozen_hash, target_embedding_shape=list(embedding.shape)), indent=2))
            return
    # First load actual labels only after BOTH SSL phases and frozen inference.
    all_labels = np.load(v3.source_root() / "data" / args.dataset / "labels.npy", allow_pickle=False).astype("int64").reshape(-1)
    target_kind = v3.DATA[args.dataset]["target"]
    labels = all_labels[target["node_ids"][target_kind]]
    split, class_counts = v3.downstream_split(labels)
    split_hash = v3.json_hash({k: a.tolist() for k, a in split.items()})
    assert split_hash == reference["downstream_split_hash"]
    t = time.perf_counter()
    evaluation = exp.probe(embedding, labels, split, seed=args.seed, repeats=args.probes, epochs=args.probe_epochs)
    result = dict(protocol=PROTOCOL, key=key, args=vars(args), dataset=args.dataset, variant=args.variant,
                  encoder_seed=args.seed, initial_graph=args.initial_graph,
                  hypothesis_timing="Secondary follow-up fixed after examining completed frozen-transfer results, before adaptation test results",
                  base_protocol=v3.PROTOCOL, base_target_reference_key=reference["key"], base_source_key=old_source["key"],
                  graph_split_hash=manifest["split_hash"], downstream_split_hash=split_hash,
                  downstream_split={k: a.tolist() for k, a in split.items()},
                  downstream_split_sizes={k: len(a) for k, a in split.items()},
                  target_evaluation_class_counts=class_counts,
                  source_target_node_overlap=manifest["source_target_node_overlap"],
                  source_target_edge_overlap=manifest["source_target_edge_overlap"],
                  source_graph=manifest["source"], target_graph=manifest["target"],
                  feature_boundary=manifest["target"]["feature_provenance"],
                  initial_training=source_training, target_training=target_training,
                  repeat_initial_full_model_hash=source_model_hash,
                  old_v3_initial_full_model_hash=old_initial["frozen_encoder_hash"],
                  repeated_initial_bitwise_matches_old=source_model_hash == old_initial["frozen_encoder_hash"],
                  copy_verification=copying,
                  initial_positive_hash=positive_hash(source_positive),
                  target_positive_hash=positive_hash(target_positive), no_encoder_labels=True,
                  frozen_inference_auxiliary_hooks_passed=True, frozen_encoder_hash=frozen_hash,
                  inference_seconds=inference_seconds, evaluation_seconds=time.perf_counter() - t,
                  total_seconds=time.perf_counter() - start, classification={"20": evaluation},
                  label_use="Only downstream probes after both SSL phases and frozen inference; target SSL does use unlabeled target graph/features",
                  training_budget="Two SSL phases each up to 600 epochs; first graph is source or target according to initial_graph. No equal-total-cost claim versus one-phase target-only reference",
                  environment=dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()))
    np.savez_compressed(out / (key + ".npz"), embedding=embedding.detach().cpu().numpy(),
                        source_positive_indices=source_positive["indices"], source_positive_mask=source_positive["mask"],
                        target_positive_indices=target_positive["indices"], target_positive_mask=target_positive["mask"])
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("DONE", key, json.dumps(evaluation["mean"]), flush=True)


def batch(args):
    out = ROOT / "adaptation_results"; out.mkdir(exist_ok=True)
    logs = out / "job_logs"; logs.mkdir(exist_ok=True)
    old_plan = json_read(ROOT / "transfer_results" / "study_plan.json")
    assert old_plan["status"] == "complete" and old_plan["protocol"] == v3.PROTOCOL
    jobs = [dict(dataset=d, variant=v, seed=s, initial_graph="source",
                 key=f"{PROTOCOL}__{d}__{v}__init_source__seed{s}") for d in DATASETS for v in VARIANTS for s in SEEDS]
    jobs += [dict(dataset=d, variant="full", seed=s, initial_graph="target",
                  key=f"{PROTOCOL}__{d}__full__init_target__seed{s}") for d in DATASETS for s in SEEDS]
    plan = dict(protocol=PROTOCOL, base_protocol=v3.PROTOCOL, status="running", jobs=jobs,
                adaptation_groups=18, source_initialized_groups=12, target_repeat_control_groups=6,
                initial_phase_encoder_runs=18, target_phase_encoder_runs=18,
                encoder_seeds=list(SEEDS), probe_initializations_per_encoder=5, workers=args.workers,
                hypothesis_timing="Secondary follow-up after observed frozen-transfer results; not part of original frozen-transfer registration",
                transfer_parameters="Only fc[0] and MP; target auxiliary branches and contrastive projector are fresh",
                target_phase_budget="Same as completed target-only references: base LR .0008, epochs600, minimum60, patience40; additional source phase disclosed",
                hyperparameter_selection="None; frozen v3 graph split, positive policies, features and probe split",
                old_study_sha256=file_sha(ROOT / "transfer_results" / "study_plan.json"),
                graph_manifest_sha256={d: file_sha(ROOT / "transfer_results" / (d + "_graph_manifest.json")) for d in DATASETS},
                script_sha256={p: file_sha(ROOT / p) for p in ("transfer_adaptation.py", "adaptation_protocol.txt", "transfer_study.py")},
                execution="Two independent accuracy workers; elapsed timings are not isolated latency benchmarks")
    def save():
        path = out / "study_plan.json"; tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(plan, indent=2), encoding="utf-8"); tmp.replace(path)
    save()
    def execute(job):
        path = out / (job["key"] + ".json")
        if path.exists():
            result = json_read(path)
            assert result["protocol"] == PROTOCOL and result["encoder_seed"] == job["seed"]
            assert result["args"]["epochs"] == 600 and result["args"]["probes"] == 5
            print("ADAPTATION_REUSE", job["key"], flush=True); return job["key"]
        command = [sys.executable, "-u", str(ROOT / "transfer_adaptation.py")]
        for field in ("dataset", "variant", "seed", "initial_graph"):
            command.extend(["--" + field.replace("_", "-"), str(job[field])])
        print("ADAPTATION_START", job["key"], flush=True)
        with (logs / (job["key"] + ".log")).open("w", encoding="utf-8") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print("ADAPTATION_FINISH", job["key"], flush=True)
        return job["key"]
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        completed = list(executor.map(execute, jobs))
    plan.update(status="complete", completed_keys=completed, wall_seconds=time.perf_counter() - t)
    save()
    (out / "evidence_sha256.json").write_text(json.dumps(dict(protocol=PROTOCOL, files={p.name: file_sha(p) for p in sorted(out.glob("*.json"))}), indent=2), encoding="utf-8")
    (out / "STUDY_COMPLETE").write_text("complete\n", encoding="utf-8")
    print("ADAPTATION_STUDY_COMPLETE", len(completed), "groups; 36 SSL phases", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="acm")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--initial-graph", choices=("source", "target"), default="source")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--probes", type=int, default=5)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--interface-only", action="store_true")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    assert args.initial_graph == "source" or args.variant == "full"
    if args.batch:
        assert not args.interface_only and args.epochs == 600 and args.probes == 5
        batch(args)
    else:
        run(args)
