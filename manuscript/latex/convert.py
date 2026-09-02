#!/usr/bin/env python3
"""
Convert the BioKGSuite Word manuscript to LaTeX.

Idempotent: re-run after every Word edit and main.tex is regenerated. Only
body.tex is machine-written; main.tex, references.bib and the preamble are
hand-maintained and never touched.

What it does beyond a plain `pandoc -o body.tex`:
  1. Rewrites numeric citations [12, 13] into \\cite{bradshaw2024,celebi2019}
     so BibTeX owns the numbering. Renumbering by hand is what produced the
     duplicate [10]/[44] in the Word version.
  2. Replaces the raster images Word embedded with the vector PDFs in
     ../figures, and wraps them in float environments with labels.
  3. Converts "Fig. 4 |" caption paragraphs into real \\caption text attached
     to the right float.
  4. Flags anything it could not resolve, rather than silently dropping it.

Usage:  python convert.py "/path/to/BioKGSuite Summer Manuscript.docx"
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGDIR = Path("../figures")

# ---------------------------------------------------------------- citations
# Maps the numeric labels used in the Word document onto BibTeX keys.
# 10 and 44 are deliberately the same key: they were duplicate entries.
CITE = {
    1: "kgregistry", 2: "buniello2025", 3: "yu2021", 4: "himmelstein2017",
    5: "bang2023", 6: "fei2026", 7: "zhao2023", 8: "soman2024",
    9: "chandak2023", 10: "breit2020", 11: "walsh2020", 12: "bradshaw2024",
    13: "celebi2019", 14: "maclean2021", 15: "rossi2021", 16: "lin2024",
    17: "pujara2017", 18: "ma2024", 19: "alshahrani2021", 20: "briere2025",
    21: "wang1996", 22: "zaveri2016", 23: "chen2019", 24: "cortes2025",
    25: "liu2024", 26: "nguyen2023", 27: "schwabe2024", 28: "jarada2020",
    29: "li2022", 30: "wei2025multihop", 31: "mei2013", 32: "novacek2020",
    33: "bromberg2013", 34: "gnanaolivu2025", 35: "chang2020",
    36: "kotnis2017", 37: "libennowell2007", 38: "gema2024",
    39: "debattista2018", 40: "paulheim2017", 41: "xue2023", 42: "watts1998",
    43: "albert2000", 44: "breit2020", 45: "joy2026", 46: "matsumoto2024",
    47: "weidrugklm2026", 48: "weicot2022",
}

# ---------------------------------------------------------------- figures
# Word's embedded rasters, in document order, mapped to the vector originals.
FIGMAP = [
    ("image1.png", "Figure1.pdf", "fig:framework", 0.95),
    ("image2.png", "Figure2.pdf", "fig:radar", 0.72),
    ("image3.png", "Figure3.pdf", "fig:heatmap", 0.95),
    ("image4.png", "Figure4.pdf", "fig:tasks", 0.98),
    ("image5.png", "Figure5.pdf", "fig:benchmark", 0.95),
    ("image6.png", "Figure6.pdf", "fig:design", 0.98),
]

UNRESOLVED = []


def run_pandoc(docx: Path, out: Path) -> str:
    subprocess.run(
        ["pandoc", str(docx), "-t", "latex", "--wrap=none",
         "--extract-media=./media", "-o", str(out)],
        cwd=HERE, check=True,
    )
    return out.read_text(encoding="utf-8")


def fix_citations(tex: str) -> str:
    """[12, 13] and [4, 5] -> \\cite{...}; unknown numbers are flagged."""
    def repl(m):
        nums = [int(n) for n in re.findall(r"\d+", m.group(1))]
        if nums == [0, 1]:
            return m.group(0)          # "normalised to [0, 1]", not a citation
        keys, bad = [], []
        for n in nums:
            (keys if n in CITE else bad).append(CITE.get(n, n))
        if bad:
            UNRESOLVED.append(f"citation number(s) not in map: {bad}")
            return m.group(0)
        seen = list(dict.fromkeys(keys))          # 10 and 44 collapse to one
        return r"\cite{" + ",".join(seen) + "}"

    # pandoc writes literal brackets as {[} ... {]}, so match that form as
    # well as bare brackets. Digit/comma/space runs only, to avoid touching
    # optional arguments such as \includegraphics[width=...].
    tex = re.sub(r"\{\[\}(\d[\d,\s]*)\{\]\}", repl, tex)
    tex = re.sub(r"(?<![\w\\])\[(\d[\d,\s]*)\](?![\w{])", repl, tex)

    # A stray LaTeX-style key survived from an earlier draft.
    tex = tex.replace(r"{[}KotnisNastase2017{]}", r"\cite{kotnis2017}")
    return tex


def fix_figures(tex: str) -> str:
    for raster, pdf, label, width in FIGMAP:
        pat = re.compile(
            r"\\includegraphics\[[^\]]*\]\{[^}]*" + re.escape(raster) + r"\}")
        if not pat.search(tex):
            UNRESOLVED.append(f"raster not found in tex: {raster}")
            continue
        if not (HERE / FIGDIR / pdf).exists():
            UNRESOLVED.append(
                f"{pdf} missing from ../figures — keeping the Word raster "
                f"for this figure; export it for a vector version")
            continue
        tex = pat.sub(
            rf"\\includegraphics[width={width}\\linewidth]"
            rf"{{{FIGDIR.as_posix()}/{pdf}}}%\n  \\label{{{label}}}",
            tex, count=1)
    return tex


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    docx = Path(sys.argv[1])
    if not docx.exists():
        sys.exit(f"not found: {docx}")

    tex = run_pandoc(docx, HERE / "_pandoc.tex")
    tex = fix_citations(tex)
    tex = fix_figures(tex)

    # Word's bibliography is now redundant: BibTeX generates it.
    tex = re.split(r"\\textbf\{References\}", tex)[0]

    (HERE / "body.tex").write_text(tex, encoding="utf-8")

    n_cite = tex.count(r"\cite{")
    print(f"body.tex written  |  {n_cite} \\cite commands  |  "
          f"{len(FIGMAP)} figures remapped")
    if UNRESOLVED:
        print("\nNEEDS ATTENTION:")
        for u in dict.fromkeys(UNRESOLVED):
            print("  -", u)


if __name__ == "__main__":
    main()
