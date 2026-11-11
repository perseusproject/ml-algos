"""
tp_utils.py — utilitaires communs aux notebooks du TP « Algorithms for Data Analysis ».

- style graphique homogène (palette catégorielle validée, axes discrets)
- petites fonctions de test (`check`, `check_close`) utilisées dans les cellules « Test »
- chronométrage
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# --------------------------------------------------------------------------------------
# Palette (ordre fixe : une série garde toujours la même couleur)
# --------------------------------------------------------------------------------------
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = PALETTE
GRAY = "#8a8985"
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"

# séquentiel (une teinte, clair -> foncé) et divergent (bleu <-> gris <-> rouge)
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b52f2f"])
CAT = ListedColormap(PALETTE, name="cat8")


def set_style():
    """Style matplotlib sobre : traits fins, grille discrète, palette fixe."""
    mpl.rcParams.update({
        "figure.figsize": (8, 4.5),
        "figure.dpi": 100,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "scatter.edgecolors": SURFACE,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "image.cmap": "seq_blue",
        "savefig.facecolor": SURFACE,
    })
    try:
        mpl.colormaps.register(SEQ)
        mpl.colormaps.register(DIV)
    except ValueError:  # déjà enregistrées
        pass


set_style()


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------
class TPTestError(AssertionError):
    pass


def check(cond, msg: str = "test"):
    """Lève une erreur explicite si `cond` est faux, sinon affiche ✅."""
    if not bool(cond):
        raise TPTestError(f"❌ {msg}")
    print(f"✅ {msg}")


def observe(cond, msg: str = "constat"):
    """Comme check, mais ne bloque pas : pour les constats empiriques qui dépendent des données
    (réelles ou simulées). Affiche ✅ si le constat attendu est vérifié, ⚠️ sinon."""
    if bool(cond):
        print(f"✅ {msg}")
    else:
        print(f"⚠️  constat attendu non observé sur ces données : {msg} — à discuter !")


def check_close(a, b, msg: str = "valeurs proches", rtol: float = 1e-6, atol: float = 1e-8):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise TPTestError(f"❌ {msg} : formes différentes {a.shape} vs {b.shape}")
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        err = np.max(np.abs(a - b))
        raise TPTestError(f"❌ {msg} : écart max = {err:.3g}")
    print(f"✅ {msg}")


@contextmanager
def timer(label: str = "durée"):
    t0 = time.perf_counter()
    yield
    print(f"⏱  {label} : {time.perf_counter() - t0:.3f} s")


def timeit(f, *args, repeat: int = 3, **kwargs) -> float:
    """Meilleur temps (s) sur `repeat` exécutions."""
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        f(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best


def summary_table(rows, columns):
    """Petit DataFrame pour les tableaux comparatifs."""
    import pandas as pd
    return pd.DataFrame(rows, columns=columns)


def plot_decision_boundary(ax, predict, X, y, title="", h=0.03, alpha=0.18, pad=0.5,
                           levels=None):
    """Trace la frontière de décision d'un classifieur 2D (`predict(X) -> labels`)."""
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.arange(x0, x1, h), np.arange(y0, y1, h))
    Z = np.asarray(predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    classes = np.unique(y)
    cmap = ListedColormap([PALETTE[i] for i in range(len(classes))])
    Zi = np.searchsorted(classes, Z)
    ax.contourf(xx, yy, Zi, levels=np.arange(len(classes) + 1) - 0.5, cmap=cmap, alpha=alpha)
    ax.contour(xx, yy, Zi, levels=np.arange(len(classes)) + 0.5, colors=INK2, linewidths=1)
    for i, c in enumerate(classes):
        m = y == c
        ax.scatter(X[m, 0], X[m, 1], s=16, color=PALETTE[i], label=f"classe {c}",
                   edgecolors=SURFACE, linewidths=0.5)
    ax.set_title(title)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    return ax
