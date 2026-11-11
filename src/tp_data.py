"""
tp_data.py — accès aux données du TP (réelles si possible, simulées sinon).

Sources réelles (gratuites, sans clé) :
  * Yahoo Finance via `yfinance`            : actions US et ETF sectoriels (prix ajustés)
  * Kenneth French Data Library             : facteurs Fama-French (quotidiens)
  * FRED (Federal Reserve Bank of St. Louis): courbe des taux du Trésor US

Chaque loader suit la même logique :
  1. si un cache CSV existe dans `data/`, on le relit ;
  2. sinon, si USE_REAL_DATA est vrai, on télécharge puis on met en cache ;
  3. en cas d'échec (pas d'internet, package manquant, site indisponible),
     on bascule sur un générateur simulé réaliste et on le signale.

Les générateurs simulés reproduisent les faits stylisés utiles au TP :
structure sectorielle des corrélations, mode « marché » dominant, queues épaisses,
clustering de volatilité (GARCH), courbe des taux à 3 facteurs (Nelson-Siegel).
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

USE_REAL_DATA = True
DATA_DIR = Path(__file__).resolve().parent / "data"

# --------------------------------------------------------------------------------------
# Univers d'actions : ~6 grandes capitalisations par secteur GICS
# --------------------------------------------------------------------------------------
UNIVERSE = {
    # Technology
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "ORCL": "Tech", "CSCO": "Tech",
    "INTC": "Tech", "ADBE": "Tech",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "MS": "Financials", "C": "Financials", "AXP": "Financials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy", "EOG": "Energy",
    "OXY": "Energy",
    # Health Care
    "JNJ": "Health", "PFE": "Health", "MRK": "Health", "ABT": "Health", "LLY": "Health",
    "UNH": "Health", "AMGN": "Health",
    # Consumer Staples
    "PG": "Staples", "KO": "Staples", "PEP": "Staples", "WMT": "Staples", "COST": "Staples",
    "CL": "Staples", "MO": "Staples",
    # Consumer Discretionary
    "AMZN": "Discretionary", "HD": "Discretionary", "MCD": "Discretionary",
    "NKE": "Discretionary", "SBUX": "Discretionary", "LOW": "Discretionary",
    # Industrials
    "CAT": "Industrials", "HON": "Industrials", "UPS": "Industrials", "LMT": "Industrials",
    "DE": "Industrials", "UNP": "Industrials",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "D": "Utilities",
    "AEP": "Utilities", "EXC": "Utilities",
    # Real Estate
    "AMT": "RealEstate", "SPG": "RealEstate", "PLD": "RealEstate", "PSA": "RealEstate",
    "O": "RealEstate", "EQR": "RealEstate",
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials", "ECL": "Materials",
    "NEM": "Materials", "FCX": "Materials",
}

SECTOR_ETFS = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLP", "XLY", "XLI", "XLU", "XLB"]
ETF_SECTOR = {"SPY": "Market", "XLK": "Tech", "XLF": "Financials", "XLE": "Energy",
              "XLV": "Health", "XLP": "Staples", "XLY": "Discretionary",
              "XLI": "Industrials", "XLU": "Utilities", "XLB": "Materials"}


def _say(msg):
    print(f"[tp_data] {msg}")


def _cache_path(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{name}.csv"


def _http_get(url: str, timeout: int = 30) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (TP data analysis)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# --------------------------------------------------------------------------------------
# Générateurs simulés
# --------------------------------------------------------------------------------------
def _garch_path(T, rng, ann_vol=0.16, alpha=0.09, beta=0.89, nu=5.0):
    """Rendements quotidiens GARCH(1,1) à innovations Student-t (variance unitaire)."""
    target = (ann_vol ** 2) / 252
    omega = target * (1 - alpha - beta)
    z = rng.standard_t(nu, size=T) * np.sqrt((nu - 2) / nu)
    r = np.empty(T)
    s2 = target
    for t in range(T):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return r


def simulate_market(tickers, sectors, start="2015-01-01", end="2024-12-31", seed=0,
                    drift=0.07, idio=(0.15, 0.30), sector_vol=0.12):
    """Modèle à facteurs : marché (GARCH) + facteur sectoriel + idiosyncratique (t).

    Retourne un DataFrame de prix (base 100) indexé par jours ouvrés.
    `sectors` : dict ticker -> secteur (le facteur sectoriel est partagé).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    mkt = _garch_path(T, rng) + drift / 252
    sec_names = sorted(set(sectors[t] for t in tickers))
    sec_f = {s: _garch_path(T, rng, ann_vol=sector_vol, alpha=0.06, beta=0.90) for s in sec_names}
    sec_f["Market"] = np.zeros(T)
    out = {}
    for tk in tickers:
        s = sectors[tk]
        b_m = rng.uniform(0.7, 1.3) if s != "Market" else 1.0
        b_s = rng.uniform(0.7, 1.3)
        idio_vol = rng.uniform(*idio) / np.sqrt(252) if s != "Market" else 0.002
        eps = rng.standard_t(4, size=T) * np.sqrt(2 / 4) * idio_vol
        r = b_m * mkt + b_s * sec_f[s] + eps
        out[tk] = 100 * np.cumprod(1 + r)
    return pd.DataFrame(out, index=dates)


def simulate_ff_factors(start="2000-01-01", end="2024-12-31", seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    mkt = _garch_path(T, rng) + 0.06 / 252
    smb = rng.standard_t(5, T) * np.sqrt(3 / 5) * 0.06 / np.sqrt(252) - 0.15 * mkt
    hml = rng.standard_t(5, T) * np.sqrt(3 / 5) * 0.07 / np.sqrt(252) + 0.02 / 252
    return pd.DataFrame({"Mkt-RF": mkt, "SMB": smb, "HML": hml,
                         "RF": np.full(T, 0.02 / 252)}, index=dates)


def simulate_yields(start="2000-01-01", end="2024-12-31", seed=2):
    """Courbe des taux Nelson-Siegel avec facteurs (niveau, pente, courbure) AR(1)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    lam = 0.6
    load_s = (1 - np.exp(-lam * mats)) / (lam * mats)
    load_c = load_s - np.exp(-lam * mats)
    f = np.zeros((T, 3))
    mu = np.array([4.0, -1.5, 0.5])
    phi = np.array([0.999, 0.997, 0.995])
    sig = np.array([0.06, 0.08, 0.12])
    f[0] = mu
    for t in range(1, T):
        f[t] = mu + phi * (f[t - 1] - mu) + sig * rng.standard_normal(3)
    Y = f[:, [0]] + f[:, [1]] * load_s + f[:, [2]] * load_c
    Y += 0.02 * rng.standard_normal(Y.shape)
    cols = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    return pd.DataFrame(np.maximum(Y, 0.0), index=dates, columns=cols)


# --------------------------------------------------------------------------------------
# Loaders (cache -> téléchargement -> simulation)
# --------------------------------------------------------------------------------------
def get_prices(tickers, start, end, name, sectors=None, seed=0, idio=(0.15, 0.30)):
    """Prix de clôture ajustés. Retourne (DataFrame, source)."""
    tickers = list(tickers)
    path = _cache_path(name)
    if path.exists():
        px = pd.read_csv(path, index_col=0, parse_dates=True)
        _say(f"{name} : relu depuis le cache {path.name} ({px.shape[1]} séries)")
        return px, "cache"
    if USE_REAL_DATA:
        try:
            import yfinance as yf
            df = yf.download(tickers, start=start, end=end, auto_adjust=True,
                             progress=False, threads=True)
            px = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
            if not isinstance(df.columns, pd.MultiIndex):
                px.columns = tickers[:1]
            px = px.dropna(how="all")
            if px.shape[0] > 100 and px.shape[1] >= max(1, len(tickers) // 2):
                px.to_csv(path)
                _say(f"{name} : téléchargé via yfinance ({px.shape[1]} séries), mis en cache")
                return px, "yfinance"
            _say(f"{name} : téléchargement incomplet, bascule sur la simulation")
        except Exception as e:  # pas d'internet, yfinance absent, etc.
            hint = " — `pip install yfinance` pour les données réelles" if isinstance(e, ImportError) else ""
            _say(f"{name} : yfinance indisponible ({type(e).__name__}){hint} ; bascule sur la simulation")
    sectors = sectors or {t: "Market" for t in tickers}
    px = simulate_market(tickers, sectors, start, end, seed=seed, idio=idio)
    _say(f"{name} : données SIMULÉES (modèle à facteurs GARCH)")
    return px, "simulé"


def _clean_returns(px, max_nan_frac=0.05):
    keep = px.columns[px.isna().mean() <= max_nan_frac]
    rets = px[keep].pct_change(fill_method=None).iloc[1:]
    return rets.dropna()


def load_stock_returns(start="2015-01-01", end="2024-12-31"):
    """Rendements quotidiens simples de l'univers UNIVERSE + secteurs.

    Retourne (returns: DataFrame [dates x tickers], sectors: Series ticker -> secteur, source)
    """
    px, src = get_prices(UNIVERSE.keys(), start, end, name=f"stocks_{start[:4]}_{end[:4]}",
                         sectors=UNIVERSE, seed=0)
    rets = _clean_returns(px)
    sectors = pd.Series({t: UNIVERSE[t] for t in rets.columns}, name="sector")
    return rets, sectors, src


def load_sector_etf_returns(start="2005-01-01", end="2024-12-31"):
    px, src = get_prices(SECTOR_ETFS, start, end, name=f"etfs_{start[:4]}_{end[:4]}",
                         sectors=ETF_SECTOR, seed=3, idio=(0.02, 0.05))
    return _clean_returns(px), src


def load_index_returns(ticker="SPY", start="2000-01-01", end="2024-12-31"):
    """Rendements quotidiens simples d'un indice / ETF (Series)."""
    px, src = get_prices([ticker], start, end, name=f"{ticker}_{start[:4]}_{end[:4]}",
                         sectors={ticker: "Market"}, seed=4)
    r = px.iloc[:, 0].pct_change(fill_method=None).dropna()
    r.name = ticker
    return r, src


FF_URLS = {
    "3": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
         "F-F_Research_Data_Factors_daily_CSV.zip",
    "5": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
         "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
}


def _parse_french_csv(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    h = next(i for i, l in enumerate(lines) if "Mkt-RF" in l)
    cols = [c.strip() for c in lines[h].split(",")[1:]]
    rows = []
    for l in lines[h + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) < 2 or not parts[0].isdigit() or len(parts[0]) != 8:
            if rows:
                break
            continue
        rows.append([parts[0]] + [float(x) for x in parts[1:len(cols) + 1]])
    df = pd.DataFrame(rows, columns=["date"] + cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date") / 100.0


def load_fama_french(model="3", start="2000-01-01", end="2024-12-31"):
    """Facteurs Fama-French quotidiens (en décimal). Retourne (DataFrame, source)."""
    path = _cache_path(f"ff{model}_daily")
    df, src = None, None
    if path.exists():
        df, src = pd.read_csv(path, index_col=0, parse_dates=True), "cache"
        _say(f"Fama-French {model} : relu depuis le cache")
    elif USE_REAL_DATA:
        try:
            raw = _http_get(FF_URLS[model])
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                text = z.read(z.namelist()[0]).decode("latin-1")
            df = _parse_french_csv(text)
            df.to_csv(path)
            src = "Kenneth French Data Library"
            _say(f"Fama-French {model} : téléchargé et mis en cache")
        except Exception as e:
            _say(f"Fama-French : téléchargement impossible ({type(e).__name__}), simulation")
    if df is None:
        df, src = simulate_ff_factors(start, end), "simulé"
        _say("Fama-French : données SIMULÉES")
    return df.loc[start:end], src


FRED_SERIES = {"3M": "DGS3MO", "6M": "DGS6MO", "1Y": "DGS1", "2Y": "DGS2", "3Y": "DGS3",
               "5Y": "DGS5", "7Y": "DGS7", "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30"}


def load_treasury_yields(start="2000-01-01", end="2024-12-31"):
    """Taux du Trésor US (en %) pour 10 maturités. Retourne (DataFrame, source)."""
    path = _cache_path("treasury_yields")
    if path.exists():
        _say("Taux US : relus depuis le cache")
        return pd.read_csv(path, index_col=0, parse_dates=True).loc[start:end], "cache"
    if USE_REAL_DATA:
        try:
            cols = {}
            for name, sid in FRED_SERIES.items():
                raw = _http_get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")
                s = pd.read_csv(io.BytesIO(raw), na_values=["."])
                s = s.set_index(pd.to_datetime(s.iloc[:, 0])).iloc[:, 1]
                cols[name] = pd.to_numeric(s, errors="coerce")
            df = pd.DataFrame(cols).dropna()
            df.to_csv(path)
            _say("Taux US : téléchargés depuis FRED et mis en cache")
            return df.loc[start:end], "FRED"
        except Exception as e:
            _say(f"FRED : téléchargement impossible ({type(e).__name__}), simulation")
    _say("Taux US : données SIMULÉES (Nelson-Siegel)")
    return simulate_yields(start, end), "simulé"


# --------------------------------------------------------------------------------------
# Jeu jouet de référence : mélange de gaussiennes (ESL, Hastie-Tibshirani-Friedman §2.3)
# C'est le jeu « n = 100 + 100, mixture of 2 Gaussians, Bayes error ≈ 21 % » des slides.
# --------------------------------------------------------------------------------------
class ESLMixture:
    """Deux classes, chacune mélange de 10 gaussiennes N(m_k, I/5).

    Les centres m_k sont tirés une fois pour toutes (graine `seed`) :
    classe 0 autour de (1, 0), classe 1 autour de (0, 1).
    """

    def __init__(self, seed=12, n_centers=10, spread=1 / 5):
        rng = np.random.default_rng(seed)
        self.means = [rng.normal([1, 0], 1, size=(n_centers, 2)),
                      rng.normal([0, 1], 1, size=(n_centers, 2))]
        self.var = spread

    def sample(self, n_per_class=100, seed=None):
        rng = np.random.default_rng(seed)
        X, y = [], []
        for c in (0, 1):
            k = rng.integers(0, len(self.means[c]), size=n_per_class)
            X.append(self.means[c][k] + rng.normal(0, np.sqrt(self.var), (n_per_class, 2)))
            y.append(np.full(n_per_class, c))
        return np.vstack(X), np.concatenate(y)

    def class_density(self, X, c):
        d2 = ((X[:, None, :] - self.means[c][None]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * self.var)).mean(1) / (2 * np.pi * self.var)

    def posterior(self, X):
        """P(Y = 1 | X = x) (classes équiprobables)."""
        p0, p1 = self.class_density(X, 0), self.class_density(X, 1)
        return p1 / (p0 + p1)

    def bayes_predict(self, X):
        return (self.posterior(X) > 0.5).astype(int)

    def bayes_error(self, n_mc=200_000, seed=123):
        X, y = self.sample(n_mc // 2, seed=seed)
        return np.mean(self.bayes_predict(X) != y)
