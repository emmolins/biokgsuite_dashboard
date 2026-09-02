"""
Poster / print styling — Latin Modern with graceful fallback.

Companion to src/plotting.py (screen + paper figures) and src/figstyle.py
(the 09-family academic look). This module is for output that gets printed
large: A0 posters, conference boards, slide blow-ups. Type is set slightly
heavier and the PDF keeps real embedded fonts so text stays crisp and
selectable at any size.

Usage
-----
    from src.poster_style import use_poster_style
    use_poster_style(base="sans")     # or base="serif"

Font resolution is best-effort: it walks a preference list and takes the
first family matplotlib can actually see, so the module never raises just
because a machine lacks Latin Modern. Call `resolved_fonts()` to check what
you actually got.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import font_manager

__all__ = ["use_poster_style", "resolved_fonts", "SANS", "SERIF", "MONO"]

# Latin Modern ships under different family names depending on how it was
# installed (TeX Live/MacTeX vs. the standalone OTF release vs. lmodern
# .deb). List every spelling we have seen, most-preferred first.
_SANS_PREF = [
    "Latin Modern Sans",      # standalone OTF / MacTeX
    "LMSans10",               # some TeX Live font-cache builds
    "CMU Sans Serif",         # Computer Modern Unicode — near-identical
    "Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans",
]
_SERIF_PREF = [
    "Latin Modern Roman",
    "LMRoman10",
    "CMU Serif",
    "Times New Roman", "Liberation Serif", "DejaVu Serif",
]
_MONO_PREF = [
    "Latin Modern Mono",
    "LMMono10",
    "CMU Typewriter Text",
    "Menlo", "Consolas", "Liberation Mono", "DejaVu Sans Mono",
]


def _available() -> set[str]:
    return {f.name for f in font_manager.fontManager.ttflist}


def _pick(preferences: list[str]) -> str:
    avail = _available()
    return next((f for f in preferences if f in avail), preferences[-1])


SANS = _pick(_SANS_PREF)
SERIF = _pick(_SERIF_PREF)
MONO = _pick(_MONO_PREF)


def resolved_fonts() -> dict[str, str]:
    """What the preference lists actually resolved to on this machine."""
    return {"sans": SANS, "serif": SERIF, "mono": MONO}


def register_font_dir(path) -> list[str]:
    """Add every font file under `path` to matplotlib's registry.

    Only needed when Latin Modern is vendored into the repo rather than
    installed system-wide. Returns the family names that were added.
    """
    from pathlib import Path

    added = []
    for f in font_manager.findSystemFonts(str(Path(path)), fontext="ttf") + \
             font_manager.findSystemFonts(str(Path(path)), fontext="otf"):
        try:
            font_manager.fontManager.addfont(f)
            added.append(font_manager.FontProperties(fname=f).get_name())
        except Exception:
            pass
    return sorted(set(added))


def use_poster_style(base: str = "sans", scale: float = 1.0,
                     grid: bool = True) -> dict[str, str]:
    """Set global rcParams for poster-scale output.

    Parameters
    ----------
    base
        "sans" for Latin Modern Sans, "serif" for Latin Modern Roman.
    scale
        Multiplier on every font size — bump to ~1.15 for A0, drop to
        ~0.9 if a figure is being placed small on a busy board.
    grid
        Leave the light y-grid on. Turn off for polar/radar plots that
        manage their own grid.

    Returns the resolved font families, so you can sanity-check in a
    notebook that you did not silently fall back to DejaVu.
    """
    if base not in ("sans", "serif"):
        raise ValueError(f"base must be 'sans' or 'serif', got {base!r}")

    family = SANS if base == "sans" else SERIF
    s = float(scale)

    plt.rcParams.update({
        # ---- type ----
        "font.family": base if base == "serif" else "sans-serif",
        "font.sans-serif": [SANS, "DejaVu Sans"],
        "font.serif": [SERIF, "DejaVu Serif"],
        "font.monospace": [MONO, "DejaVu Sans Mono"],
        "font.size": 11 * s,
        "axes.titlesize": 13 * s,
        "axes.labelsize": 11.5 * s,
        "xtick.labelsize": 10 * s,
        "ytick.labelsize": 10 * s,
        "legend.fontsize": 9.5 * s,
        "figure.titlesize": 15 * s,

        # math set in the same family so $R^2$ does not jump to DejaVu
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",

        # ---- canvas ----
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 600,          # poster print
        "savefig.bbox": "tight",

        # ---- axes furniture ----
        "axes.edgecolor": "#CFCCC3",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": grid,
        "axes.grid.axis": "y",
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.7,
        "xtick.color": "#CFCCC3",
        "ytick.color": "#CFCCC3",
        "xtick.labelcolor": "#333333",
        "ytick.labelcolor": "#333333",
        "text.color": "#333333",
        "axes.labelcolor": "#333333",
        "legend.frameon": False,

        # ---- keep vector text as text, not outlines ----
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    return resolved_fonts()
