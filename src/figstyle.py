"""
Shared house style for the 09 family (09, 09a, 09b, 09c).

Elevated/academic look — white panel, light horizontal gridlines, teal/coral
semantic palette, left-aligned descriptive titles. The ONLY thing tied to the
early notebooks is the per-KG colours: PrimeKG / DRKG / BioKG use the canonical
KG_PALETTE from src/plotting.py so KG comparisons match notebooks 00–08.
"""
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
try:
    from .plotting import KG_PALETTE
except ImportError:
    from plotting import KG_PALETTE

# ---- semantic palette (figstyle's own academic theme) ----
PALETTE = {
    'covered': '#2A9D8F',   # teal   — covered arm / kept / primary accent
    'pooled':  '#9A988F',   # warm grey — pooled / neutral
    'nokg':    '#C2705A',   # coral  — no-KG baseline / dropped / contrast
    'ink':     '#2C2C2A',   # titles / strong text
    'muted':   '#6B6A64',   # subtitles / secondary
    'axis':    '#CFCCC3',   # spines / ticks
    'grid':    '#ECE9E2',   # gridlines
}
# KG-comparison colours = canonical KG_PALETTE (match notebooks 00–08)
KG = {'no-KG': '#C9C6BD', 'PrimeKG': KG_PALETTE['primekg'], 'DRKG': KG_PALETTE['drkg'], 'BioKG': KG_PALETTE['biokg']}
# model-identity colours (figstyle's own trio)
MODEL = {'GPT': '#3C6E9E', 'Gemini': '#8C6BA8', 'Llama': '#2A9D8F'}
GEN_MARK = {'3.1': 'o', '3.2': 's', '3.3': 'D'}
COV_RAMP = {1: '#BFD8D2', 2: '#6FBBAA', 3: '#2A9D8F'}
RANK_RAMP = ['#1F6F5E', '#5DA98E', '#BFD8D2', '#E2A98F', '#C2705A']
SEQ_CMAP = LinearSegmentedColormap.from_list('biokg_seq', ['#E6E3DA', '#7FC3B2', '#1F6F5E'])
FIGSIZE = (7.6, 4.8)

_PREF = ['Helvetica', 'Helvetica Neue', 'Arial', 'Liberation Sans', 'DejaVu Sans']
_AVAIL = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in _PREF if f in _AVAIL), 'DejaVu Sans')


def apply():
    """Set global rcParams (figstyle academic look)."""
    plt.rcParams.update({
        'font.family': FONT,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'figure.dpi': 150, 'savefig.dpi': 300,
        'axes.edgecolor': PALETTE['axis'], 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.axisbelow': True,
        'axes.grid': True, 'axes.grid.axis': 'y',
        'grid.color': PALETTE['grid'], 'grid.linewidth': 0.7,
        'xtick.color': PALETTE['axis'], 'ytick.color': PALETTE['axis'],
        'xtick.labelcolor': PALETTE['muted'], 'ytick.labelcolor': PALETTE['muted'],
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'text.color': PALETTE['ink'], 'axes.labelcolor': '#3A3A37', 'axes.labelsize': 11,
        'axes.titlelocation': 'left',
        'legend.frameon': False, 'legend.fontsize': 9.5,
    })


def title(ax, title, subtitle=None, pad=26):
    """Left-aligned bold title with an optional grey subtitle above the axes."""
    ax.set_title(title, fontsize=14, fontweight='bold', loc='left', pad=pad, color=PALETTE['ink'])
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10,
                color=PALETTE['muted'], va='bottom', ha='left')


def suptitle(fig, title, subtitle=None, x=0.012, y=1.0):
    """Left-aligned bold figure title (for multi-panel figures), matching `title`."""
    fig.suptitle(title, fontsize=14, fontweight='bold', x=x, y=y, ha='left', color=PALETTE['ink'])
    if subtitle:
        fig.text(x, y - 0.045, subtitle, fontsize=10, color=PALETTE['muted'], ha='left')
