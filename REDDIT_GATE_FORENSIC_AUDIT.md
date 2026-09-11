# Reddit Gate Forensic Audit

## Verdict

`legacy_sensitivity_is_not_a_matched_operating_point_sweep`

The legacy confidence experiment is admissible as a matched threshold sweep only if the fitted-model invariants below are identical. Missing fields remain `NA` and are not inferred.

## Seed-level inventory

| Run | c | Seed | Config digest | Sources | Params | LCB | Gate | Identity MRR | Route MRR | Hard MRR |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| main | 0.95 | 0 | b9eb21c05716e5e7f2a841b1cbec5a9463decc7a079a7874a60ba541871e86cc | uci | 269112 | -0.001502581398 | 0 | 0.4204879403 | 0.4205915034 | 0.4204879403 |
| main | 0.95 | 1 | b9eb21c05716e5e7f2a841b1cbec5a9463decc7a079a7874a60ba541871e86cc | wikipedia | 278328 | -0.001272903484 | 0 | 0.4130363762 | 0.4131545126 | 0.4130363762 |
| main | 0.95 | 2 | b9eb21c05716e5e7f2a841b1cbec5a9463decc7a079a7874a60ba541871e86cc | wikipedia | 278328 | -2.080794838e-05 | 0 | 0.4524988234 | 0.4521450102 | 0.4524988234 |
| main | 0.95 | 3 | b9eb21c05716e5e7f2a841b1cbec5a9463decc7a079a7874a60ba541871e86cc | uci | 269112 | -0.0003966062393 | 0 | 0.4407099783 | 0.4413630068 | 0.4407099783 |
| main | 0.95 | 42 | b9eb21c05716e5e7f2a841b1cbec5a9463decc7a079a7874a60ba541871e86cc | mooc,uci | 295354 | -0.0002865282694 | 0 | 0.4316948354 | 0.432112366 | 0.4316948354 |
| legacy sensitivity | 0.9 | 0 | 83951efbd1420f36f105d45dd6b042085f44b9da7000e4360524b6f0ccb1a645 | wikipedia | 76200 | -0.0003958264814 | 0 | 0.4138762951 | 0.4134673774 | 0.4138762951 |
| legacy sensitivity | 0.9 | 1 | 83951efbd1420f36f105d45dd6b042085f44b9da7000e4360524b6f0ccb1a645 | wikipedia | 76200 | 7.500777642e-06 | 1 | 0.398956567 | 0.3989670873 | 0.3989670873 |
| legacy sensitivity | 0.9 | 42 | 83951efbd1420f36f105d45dd6b042085f44b9da7000e4360524b6f0ccb1a645 | mooc,wikipedia | 83178 | 0.000142823719 | 1 | 0.3991762698 | 0.3986564279 | 0.3986564279 |
| legacy sensitivity | 0.95 | 0 | 52a45c819c86d3a9223b460a2caf47a64f746230aad18b86304f51b5c0f41508 | wikipedia | 76200 | -0.0005703588107 | 0 | 0.4138762951 | 0.4134673774 | 0.4138762951 |
| legacy sensitivity | 0.95 | 1 | 52a45c819c86d3a9223b460a2caf47a64f746230aad18b86304f51b5c0f41508 | wikipedia | 76200 | 5.809730954e-06 | 1 | 0.398956567 | 0.3989670873 | 0.3989670873 |
| legacy sensitivity | 0.95 | 42 | 52a45c819c86d3a9223b460a2caf47a64f746230aad18b86304f51b5c0f41508 | mooc,wikipedia | 83178 | 0.0001020856996 | 1 | 0.3991762698 | 0.3986564279 | 0.3986564279 |
| legacy sensitivity | 0.99 | 0 | c20ae1815233dd1ee90509ada3ea0e2bbb555b576271c2fb714c1ef5ee14ce59 | wikipedia | 76200 | -0.0005999125612 | 0 | 0.4138762951 | 0.4134673774 | 0.4138762951 |
| legacy sensitivity | 0.99 | 1 | c20ae1815233dd1ee90509ada3ea0e2bbb555b576271c2fb714c1ef5ee14ce59 | wikipedia | 76200 | 0 | 0 | 0.398956567 | 0.3989670873 | 0.398956567 |
| legacy sensitivity | 0.99 | 42 | c20ae1815233dd1ee90509ada3ea0e2bbb555b576271c2fb714c1ef5ee14ce59 | mooc,wikipedia | 83178 | 4.42994833e-05 | 1 | 0.3991762698 | 0.3986564279 | 0.3986564279 |

## Proven cause

The archived sensitivity branch changed fitted-model invariants (configuration digest, selected source portfolio, parameter count and predictions). It therefore reran training under a reduced configuration instead of changing only the confidence quantile on the authoritative fitted model. Its c=0.95 gate count is not comparable to the main five-seed c=0.95 count.

## Completed repair

`scripts/recompute_matched_gate_sensitivity.py` recomputed
`c in {0.90,0.95,0.99}` from each authoritative seed's stored calibration
window means. It reused the same bootstrap seed policy, source portfolio,
checkpoint, calibration suffix, and route/identity test predictions; no model
was retrained. The repaired c=0.95 decision equals the main decision for all 25
dense target-seed cells, as recorded in `GATE_CONSISTENCY_CHECK.txt`.

For Reddit, the authoritative gate count is 0/5 at all three confidence
levels. This resolves the former 0/5 versus 2/3 contradiction. It also exposes
the real limitation: because future forced routing is positive in four of five
Reddit seeds, the conservative gate creates four missed opportunities. The
paper may describe the gate as empirically conservative, but not as an accurate
or guaranteed predictor of future transfer utility.
