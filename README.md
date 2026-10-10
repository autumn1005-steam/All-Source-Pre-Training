# TriView-HCL

Research code and recorded evidence for **TriView-HCL** by Qiurui Song and Xuchu Jiang.
Corresponding author: Xuchu Jiang, <xuchujiang@zuel.edu.cn>.

This branch contains the revised experimental implementation, including an explicit
positive-pair generator, disjoint-graph transfer, unlabeled target adaptation, and the
DBLP-L fixed-target-update follow-up. The scientific implementation and recorded JSON
results are preserved byte for byte from the completed supplementary source package.
The small packaging, verification, data-import, and result-summary utilities are new.

## What is included

- `reproduction/`: original TriView-HCL modules, the revised harness, baseline modules,
  fixed study plans, protocol notes, raw per-encoder JSON metrics and input manifests.
- `summaries/study_summary.json`: readable aggregates recomputed from the raw JSON files.
- `SOURCE_PROVENANCE.json`: exact source-file hashes and baseline origins.
- `data_manifest.json`: exact paths, sizes, and SHA-256 hashes of the 96 preserved input
  files in the original supplementary package.
- `RELEASE_MANIFEST.json`: hashes of the published source tree.

Graph arrays, raw edge tables, generated positive matrices, embeddings, model weights,
execution logs and the manuscript are excluded from Git. This is a source-and-evidence
release; running experiments additionally requires the data described below. The
recorded evidence remains available for inspection without graph data or a GPU.

## Read and verify the recorded results

These commands use only the Python standard library and run no training:

```bash
python verify_release.py
python summarize_results.py --output runs/recorded_summary.json
```

Metrics in the summary are fractions. Sample standard deviations are across
encoder-level means, with five probe initializations nested inside each encoder.
The archive contains 153 original trained encoder records (including rate-search
and attention-control runs), 36 generated-positive encoder records, 36 held-out-graph
trained encoder records, six untrained-encoder controls, six raw-feature probe groups,
18 two-stage adaptation trials, and three additional fixed600 target phases.
Thus there are **246 final trained encoder trial records and 264 recorded SSL training
phases**. These are not independent graph samples.

## Environment

Use Python 3.12 on Linux, or Linux inside WSL2, and a CUDA-capable GPU. The positive
dispatcher uses Linux file locking (`fcntl`). The recorded runs used PyTorch
2.8.0 with CUDA 12.8 on an NVIDIA RTX 5090. Exact observed library versions are in
`reproduction/environment.json` and the per-study evidence. GPU memory needs vary:
the full loss and ContraNorm have quadratic pairwise storage, and the exact SimRank
generator also retains quadratic similarities. AdaMEOW accuracy jobs run alone in
the original dispatcher. Using a smaller GPU can require exact block evaluation.

One compatible installation entry point is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

`requirements.txt` specifies compatible dependency ranges rather than an exact lock
of the historical environment. Different versions, hardware and CUDA sparse kernels
can change floating-point trajectories; bitwise equality of newly trained weights
is not claimed. The low-cost CPU mathematical checks are:

```bash
python reproduction/positive_pipeline.py --test-only --device cpu --output-root runs/math_tests
```

## Data: exact inputs and public origins

The benchmark families are ACM, DBLP, AMiner and Freebase. Public source references
are the [HeCo data snapshot](https://github.com/liun-online/HeCo/tree/20e8ca2042838877b2c815279920218b6bbea1db/data)
and the [MEOW repository snapshot](https://github.com/jianxiangyu/MEOW/tree/6a072d79648ab83f7ec7b7482af1696ed00938a1).
The latter README gives the authors' external data-bundle link. These fixed commits
are lookup references, not a claim that every exact input used here is downloadable
at those locations. In particular, DBLP-L additionally needs the supplied
`p_feat.npz` (14,328 papers by 4,231 feature columns), raw `pa.txt`, `pc.txt` and
`pt.txt`. Graph indices and public train/validation/test splits must match the
recorded manifests. Rebuilding arbitrary substitutes does not reproduce this study.

The exact input bundle is the previously supplied **`TriView_HCL_source_package.zip`**
supplementary archive. It is not included in this Git branch and this release does
not claim a fully automatic public download of all exact inputs. If the paper's
supplementary archive is available to you, import and verify its data as follows:

```bash
python prepare_data.py --source-package /path/to/TriView_HCL_source_package.zip
python prepare_data.py --verify
```

The importer extracts only the 96 manifest-listed data files, checks every SHA-256
before writing, and never extracts model checkpoints or connection helpers. A package
with the same data but a changed manuscript is accepted because each data file is
checked independently. Alternatively, obtain the upstream inputs, place them at the
following paths, and run `prepare_data.py --verify`. A failed check identifies missing
or different files and must be resolved before comparing reproduction scores.

```text
reproduction/original/TriView-HCL-main/data/{acm,dblp,aminer,freebase}/
reproduction/baselines/MEOW-main/data/{acm,dblp,aminer}/
```

`data_manifest.json` enumerates every required filename. Original inputs include
meta-path matrices, neighbor arrays, labels, the released `pos.npz` files, fixed
20/40/60-per-class splits, graph edge tables, and ACM/DBLP attributes. AdaMEOW-CF
additionally uses MEOW's count-preserving meta-path matrices and sparse neighbor
incidence matrices. Generated-positive matrices and adaptation checkpoints are
produced in a new run; they are not required downloads for a complete fresh run.

## Reproduce the complete study in a fresh workspace

Do not run training inside the archived `reproduction/` tree: existing JSON files
can be reused by dispatchers, and some plans/manifests are rewritten. Create a fresh
workspace to preserve the published evidence and ensure each required phase runs.
Some older dispatchers (`study_parallel.py`, `efficiency.py`, `positive_tail_batch.py`)
execute at module scope. Do not import them or invoke `--help` expecting a dry run;
use only the documented commands when ready to execute the study.
From the repository root:

```bash
python make_run_workspace.py --output runs/full
python prepare_data.py --workspace runs/full --source-package /path/to/TriView_HCL_source_package.zip
python prepare_data.py --workspace runs/full --verify
cd runs/full
```

Run the original comparison, then the numerical and preprocessing checks. The
efficiency script first completes six mean-attention controls and then profiles
isolated latency. Keep the GPU otherwise idle while profiling:

```bash
python study_parallel.py
python numerical_check.py
python efficiency.py
python preprocess_final.py
```

The original comparison performs a validation-only rate-scale search for TriView-HCL
and HeCo, followed by the registered comparisons, ablations, random-positive
controls and loss-weight sensitivity. Encoder checkpoints use self-supervised
training loss; each probe uses one minimum-validation-cross-entropy checkpoint for
all reported classification metrics. `NS4GC-P` uses the target-node union projection;
`AdaMEOW-CF` uses common node inputs and native count-preserving PathSim contexts.
Their suffixes explicitly identify these adapters.

Then profile and generate positive matrices on the idle GPU before training the
36 generated-positive encoders. This extension uses the **archived frozen primary
rate choices**, not new target-test tuning:

```bash
python positive_pipeline.py --device cuda --self-test
python positive_study.py --study-plan ../../reproduction/study_plan.json --workers 2
```

The generator declares the binary incoming meta-path union, three SimRank iterations
with damping 0.9, raw-feature cosine, a 0.5 mixture, deterministic score ties,
at most eight non-self pairs plus self, and a same-candidate random control.
ACM/DBLP test all four selectors; identity-feature AMiner/Freebase test SimRank and
random. No labels or learned embeddings enter this static generator. Supplied
positive files in the original comparison are kept separate from these newly
generated matrices. Compressed NPZ timestamps can change file hashes on regeneration;
regenerate the entire matched study in this fresh workspace, rather than mix a new
matrix with an old raw result manifest.

Execute frozen transfer, then adaptation, then the fixed600 replay in that order:

```bash
python transfer_study.py --dataset acm --audit
python transfer_study.py --dataset dblp --audit
python transfer_batch.py --workers 2
python transfer_adaptation.py --batch --workers 2
python transfer_budget_control.py --batch --workers 2
cd ../..
python summarize_results.py --root runs/full --output runs/full_summary.json
```

Frozen transfer splits typed nodes and raw edges into disjoint source/target graphs
using one fixed owner partition per dataset. ACM shares the supplied feature-column
semantics. DBLP-L rebuilds each author's 4,231-column features from retained
graph-local papers, rather than load the main benchmark's 334-column author features.
Only `fc[0]` and the MP encoder are retained for target inference; auxiliary branches
are excluded. Labels are first opened after SSL and frozen inference, for the
small target probe. This is within-dataset graph transfer with a labeled downstream
probe, not zero-shot or cross-dataset transfer.

Adaptation uses unlabeled target SSL with fresh target auxiliary parameters and
optimizer state. Its target-repeat controls also use two stages. Actual early-stop
epochs and graph costs can differ. The three DBLP-L fixed600 replays reuse the trusted
initial target checkpoints generated by the preceding adaptation stage and execute
exactly 600 additional target updates. Newly run adaptation must finish first:
the public archived JSON files alone cannot substitute for the excluded checkpoints.
Matching target updates does not establish equal total two-stage compute. Minimum
SSL checkpoints can precede update600. The three original target-repeat outcomes
and the later fixed600 results are both retained.

Historical protocol files contain original host-path examples; the portable commands
above are the reproduction entry points. Study plans and result manifests keep those
historical paths as provenance, not runtime configuration. Concurrent accuracy-job
wall times are not isolated latency measurements. The negative and mixed outcomes
are preserved; this code release does not assert that generated positives or frozen
transfer improve every dataset.

## Sources and licensing

Original TriView-HCL code is preserved under `reproduction/original/TriView-HCL-main`.
Baseline origins and snapshot boundaries are described in [THIRD_PARTY.md](THIRD_PARTY.md).
Upstream MEOW and NS4GC licenses are retained beside their code. The supplied
TriView-HCL and HeCo snapshots did not contain a license; no new blanket license
has been assigned to those files. Code availability and copyright licensing are
separate. Preserve upstream notices when redistributing, and consult the respective
authors for any permission not specified by an included license.

The manuscript's code-availability link for this release is
[the `triview-hcl` branch](https://github.com/autumn1005-steam/All-Source-Pre-Training/tree/triview-hcl).
