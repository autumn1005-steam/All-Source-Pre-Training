# Preserved sources

This repository retains scientific source bytes from the completed TriView-HCL
supplementary package. `SOURCE_PROVENANCE.json` records file-level SHA-256 hashes.
The historical package did not record upstream checkout commit IDs. The following
fixed references are lookup snapshots, not assertions about the historical code
checkout; the included source hashes identify the precise copies used here.

| Component | Origin | Included path | License in supplied snapshot |
| --- | --- | --- | --- |
| TriView-HCL | Author-supplied original code archive | `reproduction/original/TriView-HCL-main` | No license file supplied |
| HeCo | <https://github.com/liun-online/HeCo/tree/20e8ca2042838877b2c815279920218b6bbea1db> | `reproduction/baselines/HeCo-main`, `reproduction/heco_impl` | No license file supplied |
| MEOW / AdaMEOW | <https://github.com/jianxiangyu/MEOW/tree/6a072d79648ab83f7ec7b7482af1696ed00938a1> | `reproduction/baselines/MEOW-main` | MIT; retained `LICENSE` |
| NS4GC | <https://github.com/Cloudy1225/NS4GC/tree/6d261bca09eabd68edfcbc0705b4085464263e2a> | `reproduction/baselines/NS4GC-main` | Apache-2.0; retained `LICENSE` |

`heco_impl` uses a separate package namespace to avoid module-name collisions with
TriView-HCL. `adameow_runner.py` prepares its execution namespace at runtime and
performs declared sparse/block adapters. These changes are described in the main
README and protocol records. The bundled baseline modules and original README files
are preserved for attribution; baseline-native instructions and historical dependency
files do not replace this revision's common-input experiment protocol.

The unmodified baseline README image links may point to figures not included in this
code-only release. Consult the upstream snapshots to view those illustrations.

No root copyright license is added. An included component license applies to that
component as its text specifies; it does not imply a license for unrelated original
code. Data are not redistributed in Git; their public origins and exact-file
requirements are documented separately in the main README and data manifest.
