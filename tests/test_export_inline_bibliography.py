from pathlib import Path

from scripts.export_inline_bibliography import export_inline_bibliography


def test_export_inline_bibliography(tmp_path: Path) -> None:
    source = tmp_path / "paper.tex"
    output = tmp_path / "references.bib"
    source.write_text(
        r"\begin{thebibliography}{2}"
        r"\bibitem{alpha} A. Author, ``Alpha.''"
        r"\bibitem{beta} B. Author, ``Beta.''"
        r"\end{thebibliography}",
        encoding="utf-8",
    )

    assert export_inline_bibliography(source, output) == 2
    text = output.read_text(encoding="utf-8")
    assert "@misc{alpha," in text
    assert "@misc{beta," in text
    assert "A. Author" in text
