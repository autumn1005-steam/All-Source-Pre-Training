"""Export the authoritative inline IEEE bibliography as a BibTeX ledger."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ITEM_RE = re.compile(
    r"\\bibitem\{(?P<key>[^}]+)\}\s*(?P<body>.*?)(?=\\bibitem\{|\\end\{thebibliography\})",
    re.DOTALL,
)


def export_inline_bibliography(tex_path: Path, output_path: Path) -> int:
    text = tex_path.read_text(encoding="utf-8")
    items = list(ITEM_RE.finditer(text))
    if not items:
        raise ValueError(f"no inline bibliography items found in {tex_path}")
    blocks = [
        "% Machine-generated ledger of the authoritative inline bibliography.\n",
        "% The manuscript compiles from its inline thebibliography environment.\n\n",
    ]
    for item in items:
        body = " ".join(item.group("body").split())
        blocks.append(f"@misc{{{item.group('key')},\n  note = {{{body}}}\n}}\n\n")
    output_path.write_text("".join(blocks), encoding="utf-8")
    return len(items)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", type=Path, default=Path("main.tex"))
    parser.add_argument("--output", type=Path, default=Path("references.bib"))
    args = parser.parse_args()
    count = export_inline_bibliography(args.tex, args.output)
    print(f"exported {count} entries to {args.output}")


if __name__ == "__main__":
    main()
