# Verification

Frozen-validation package generated from OOS-HRCSR v3 without changing the model/gate algorithm.

- Full unit/integration suite: **178 passed, 1 skipped** (`PYTHONPATH=src:. pytest -q`; the skipped shell-syntax check requires bash)
- Added configurable seed lists to `run_calibrated_safety_check.py`.
- Added 5-seed × 3-target frozen validation launcher.
- Added OOS safety decision-system analyzer.
- Recomputed the five-target gate-confidence sweep (`c=0.90,0.95,0.99`) from
  the same trained checkpoints; all 25 c=.95 decisions match the main gate.
- Existing historical collision-free negative-sampling and chronological OOS calibration protocol retained.
- Rebuilt `paper/TGFM_paper.tex`, root `main.tex`, and the final TeX/PDF
  successfully (13 pages each); every page was rendered for visual QC.
- Rebuilt `supplement.tex` successfully (1 page); the reduced-width checkpoint
  boundary diagnostic is included with an explicit scope caveat.
- Audited five direct SocialEvo seeds and regenerated the separate sparse-domain
  table and dense/sparse gate panels.
- Removed the method belonging to a separate concurrent submission from the
  manuscript, supplement, unified table, and paper-facing evidence ledgers.
