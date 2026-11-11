#!/usr/bin/env python3
"""
CFD Technical Scanner (non-equity CFD universe)
============================================================

What it does, for every instrument:
  1. Downloads daily (2y) and hourly (180d) prices from Yahoo Finance.
  2. Builds 1D / 4H / 1H candles and computes EMA 20/50/200, RSI 14, ATR 14.
  3. Scores the trend on each timeframe and combines them into a bias.
  4. Finds swing highs/lows on the 4H chart -> support / resistance levels.
  5. Proposes a trend-following setup: entry (pullback to 4H EMA20 /
     resistance or support), stop (1 x 4H ATR beyond, or beyond the swing),
     target (next opposite level), reward:risk, and position size in CHF.
  6. Ranks everything and writes a Markdown + CSV report.

Usage examples
--------------
  python cfd_scanner.py                              # full universe
  python cfd_scanner.py --groups energy,metals       # only some groups
  python cfd_scanner.py --only WTI,GOLD,EURUSD       # only some symbols
  python cfd_scanner.py --account 1000 --risk 2      # sizing: 2% of 1000 CHF
  python cfd_scanner.py --only WTI --broker WTI=80.00
        -> shifts all WTI levels to your broker's CFD price
  python cfd_scanner.py --list                       # show the universe

Install once:  pip install yfinance pandas numpy

Notes
-----
* Yahoo data is futures / index / spot data, NOT your broker's CFD feed.
  Use --broker NAME=price to align levels with the price in your ticket.
* "Units" in sizing = units of the underlying (barrels, ounces, index points,
  currency units, coins). Check your broker's contract size per lot.
* Sizing includes the spread in the risk, rounds down to the size step, and
  marks a setup "not tradable" when the minimum size alone exceeds your risk.
  Reward is net of spread and financing (--hold-days). Margin and notional
  are reported. Contract specs (GROUP_SPECS / NAME_SPECS) are ESTIMATES:
  verify them and override with --spec NAME=spread:0.06,min:1,margin:10.
* RSI and entry-distance penalties are graded, not hard cutoffs.
* This is a decision-support tool, not advice. Always check levels on a chart.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Universe: name -> (yahoo ticker, group, quote currency, unit label)
# --------------------------------------------------------------------------
UNIVERSE: dict[str, tuple[str, str, str, str]] = {
    # Energies (futures, front month)
    "WTI":       ("CL=F", "energy", "USD", "bbl"),
    "BRENT":     ("BZ=F", "energy", "USD", "bbl"),
    "NATGAS":    ("NG=F", "energy", "USD", "MMBtu"),
    "GASOLINE":  ("RB=F", "energy", "USD", "gal"),
    "HEATOIL":   ("HO=F", "energy", "USD", "gal"),   # proxy for gasoil
    # Metals
    "GOLD":      ("GC=F", "metals", "USD", "oz"),
    "SILVER":    ("SI=F", "metals", "USD", "oz"),
    "PLATINUM":  ("PL=F", "metals", "USD", "oz"),
    "PALLADIUM": ("PA=F", "metals", "USD", "oz"),
    "COPPER":    ("HG=F", "metals", "USD", "lb"),
    # Indices (futures where available: trade almost 24h like CFDs)
    "US500":     ("ES=F", "indices", "USD", "pt"),
    "NAS100":    ("NQ=F", "indices", "USD", "pt"),
    "US30":      ("YM=F", "indices", "USD", "pt"),
    "US2000":    ("RTY=F", "indices", "USD", "pt"),
    "DE40":      ("^GDAXI", "indices", "EUR", "pt"),
    "EU50":      ("^STOXX50E", "indices", "EUR", "pt"),
    "FR40":      ("^FCHI", "indices", "EUR", "pt"),
    "UK100":     ("^FTSE", "indices", "GBP", "pt"),
    "CH20":      ("^SSMI", "indices", "CHF", "pt"),
    "ES35":      ("^IBEX", "indices", "EUR", "pt"),
    "IT40":      ("FTSEMIB.MI", "indices", "EUR", "pt"),
    "JP225":     ("^N225", "indices", "JPY", "pt"),
    "HK50":      ("^HSI", "indices", "HKD", "pt"),
    "AUS200":    ("^AXJO", "indices", "AUD", "pt"),
    # Bonds (US futures only; Bund/Gilt not on Yahoo)
    "US10Y_NOTE": ("ZN=F", "bonds", "USD", "pt"),
    "US30Y_BOND": ("ZB=F", "bonds", "USD", "pt"),
    # Crypto
    "BTCUSD":    ("BTC-USD", "crypto", "USD", "coin"),
    "ETHUSD":    ("ETH-USD", "crypto", "USD", "coin"),
    "SOLUSD":    ("SOL-USD", "crypto", "USD", "coin"),
    "XRPUSD":    ("XRP-USD", "crypto", "USD", "coin"),
    "LTCUSD":    ("LTC-USD", "crypto", "USD", "coin"),
}

FX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "EURCHF", "EURGBP", "EURJPY", "GBPJPY", "GBPCHF", "AUDJPY", "CHFJPY",
    "EURAUD", "EURCAD", "AUDCAD", "AUDNZD", "CADJPY", "NZDJPY", "CADCHF",
    "AUDCHF", "USDSEK", "USDNOK", "USDMXN", "USDZAR", "USDSGD", "USDPLN",
]
for _p in FX_PAIRS:
    UNIVERSE[_p] = (f"{_p}=X", "fx", _p[3:], _p[:3])

# --------------------------------------------------------------------------
# Parameters (tweak here)
# --------------------------------------------------------------------------
PIVOT_K = 3              # bars on each side for a 4H swing high/low
PIVOT_LOOKBACK = 180     # 4H bars scanned for swings (~30 trading days)
CLUSTER_ATR = 0.5        # merge levels closer than 0.5 x ATR(4H)
STOP_ATR = 1.0           # stop distance beyond entry in ATR(4H)
ENTRY_DIST_OK_ATR = 1.5  # entry within this many ATR(4H): no penalty
ENTRY_DIST_MAX_ATR = 3.0 # penalty grows linearly up to here (score x0.3)
TF_WEIGHTS = {"1D": 2.0, "4H": 1.5, "1H": 1.0}

# --------------------------------------------------------------------------
# Contract specs (ESTIMATES: verify each in your broker's instrument info and
# override with --spec NAME=spread:0.06,min:1,step:1,margin:10,fin:6)
#   spread_pct : typical spread as % of price (used if no absolute spread)
#   margin     : margin requirement, % of notional
#   min / step : minimum size and size increment, in units of the underlying
#   fin        : overnight financing cost, % per year of notional
# --------------------------------------------------------------------------
FX_MAJORS = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD"}
FX_EXOTIC_CCY = {"SEK", "NOK", "MXN", "ZAR", "SGD", "PLN"}

GROUP_SPECS: dict[str, dict] = {
    "fx_major": dict(spread_pct=0.015, margin=3.33, min=1000, step=1000, fin=4),
    "fx_cross": dict(spread_pct=0.03,  margin=5,    min=1000, step=1000, fin=4),
    "fx_exotic": dict(spread_pct=0.12, margin=10,   min=1000, step=1000, fin=8),
    "energy":   dict(spread_pct=0.07,  margin=10,   min=1,    step=1,    fin=6),
    "metals":   dict(spread_pct=0.03,  margin=5,    min=1,    step=1,    fin=6),
    "indices":  dict(spread_pct=0.01,  margin=5,    min=1,    step=1,    fin=6),
    "bonds":    dict(spread_pct=0.03,  margin=5,    min=1,    step=1,    fin=6),
    "crypto":   dict(spread_pct=0.5,   margin=50,   min=0.01, step=0.01, fin=20),
}
NAME_SPECS: dict[str, dict] = {
    "WTI":      dict(spread=0.06),                 # typical oil CFD spread
    "NATGAS":   dict(spread_pct=0.3, min=10, step=10),
    "GASOLINE": dict(spread_pct=0.15, min=100, step=100),
    "HEATOIL":  dict(spread_pct=0.15, min=100, step=100),
    "SILVER":   dict(spread_pct=0.05, min=10, step=1, margin=10),
    "COPPER":   dict(spread_pct=0.1, min=100, step=100),
    "XRPUSD":   dict(min=1, step=1),
    "SOLUSD":   dict(min=0.1, step=0.1),
    "LTCUSD":   dict(min=0.1, step=0.1),
    "BTCUSD":   dict(min=0.001, step=0.001),
}
SPEC_CLI: dict[str, dict] = {}   # filled from --spec

HOLD_DAYS = 3            # expected holding period for financing cost
MAX_FEE_PCT = 10.0       # flag when spread + financing exceed this % of reward
MAX_MARGIN_PCT = 50.0    # flag when one position's margin exceeds this % of account
PORTFOLIO_TOP_N = 5      # combined risk / margin check over the top N setups


def instrument_spec(name: str, price: float) -> dict:
    group = UNIVERSE[name][1]
    if group == "fx":
        if name in FX_MAJORS:
            key = "fx_major"
        elif name[:3] in FX_EXOTIC_CCY or name[3:] in FX_EXOTIC_CCY:
            key = "fx_exotic"
        else:
            key = "fx_cross"
    else:
        key = group
    sp = dict(GROUP_SPECS[key])
    sp.update(NAME_SPECS.get(name, {}))
    sp.update(SPEC_CLI.get(name, {}))
    if "spread" not in sp:
        sp["spread"] = price * sp["spread_pct"] / 100
    return sp


def rsi_penalty(bias: str, r1d: float, r4h: float) -> float:
    """0 = no issue, 1 = extreme. Graded instead of a hard cutoff."""
    if bias == "SHORT":    # oversold: 1D 35 -> 25, 4H 30 -> 20
        p = max((35 - r1d) / 10, (30 - r4h) / 10)
    else:                  # overbought: 1D 65 -> 75, 4H 70 -> 80
        p = max((r1d - 65) / 10, (r4h - 70) / 10)
    return min(max(p, 0.0), 1.0)


def entry_dist_factor(d: float) -> float:
    if math.isnan(d) or d <= ENTRY_DIST_OK_ATR:
        return 1.0
    x = (d - ENTRY_DIST_OK_ATR) / (ENTRY_DIST_MAX_ATR - ENTRY_DIST_OK_ATR)
    return max(0.3, 1 - 0.7 * x)


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"] - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for n in (20, 50, 200):
        df[f"EMA{n}"] = ema(df["Close"], n)
    df["RSI"] = rsi(df["Close"])
    df["ATR"] = atr(df)
    return df


def trend_score(df: pd.DataFrame, use_200: bool) -> int:
    """+1 bullish stack, -1 bearish stack, 0 mixed."""
    r = df.iloc[-1]
    c, e20, e50, e200 = r["Close"], r["EMA20"], r["EMA50"], r["EMA200"]
    if c > e20 > e50 and (not use_200 or e50 > e200 or c > e200):
        return 1
    if c < e20 < e50 and (not use_200 or e50 < e200 or c < e200):
        return -1
    return 0


# --------------------------------------------------------------------------
# Swing levels
# --------------------------------------------------------------------------
def swing_levels(df: pd.DataFrame, k: int, lookback: int) -> tuple[list, list]:
    d = df.tail(lookback)
    highs, lows = d["High"].values, d["Low"].values
    sh, sl = [], []
    for i in range(k, len(d) - k):
        if highs[i] == highs[i - k:i + k + 1].max():
            sh.append(float(highs[i]))
        if lows[i] == lows[i - k:i + k + 1].min():
            sl.append(float(lows[i]))
    # also include the absolute extremes of the window
    sh.append(float(highs.max()))
    sl.append(float(lows.min()))
    return sh, sl


def cluster(levels: list[float], tol: float) -> list[float]:
    if not levels:
        return []
    levels = sorted(levels)
    groups, cur = [], [levels[0]]
    for x in levels[1:]:
        if x - cur[-1] <= tol:
            cur.append(x)
        else:
            groups.append(cur)
            cur = [x]
    groups.append(cur)
    return [float(np.mean(g)) for g in groups]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def fetch(ticker: str, tries: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download daily + hourly bars, retrying (Yahoo rate-limits cloud IPs)."""
    import time
    import yfinance as yf
    err: Exception | None = None
    for i in range(tries):
        try:
            t = yf.Ticker(ticker)
            daily = clean(t.history(period="2y", interval="1d", auto_adjust=False))
            hourly = clean(t.history(period="180d", interval="1h", auto_adjust=False))
            if not daily.empty and not hourly.empty:
                return daily, hourly
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(2 * (i + 1))
    if err:
        raise err
    return pd.DataFrame(), pd.DataFrame()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df = df[(df["High"] >= df["Low"]) & (df["Close"] > 0)]
    return df


def resample_4h(h: pd.DataFrame) -> pd.DataFrame:
    if h.empty:
        return h
    return h.resample("4h").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()


_fx_cache: dict[str, float] = {}


def to_chf(ccy: str, fetcher=None) -> float:
    """Value of 1 unit of `ccy` in CHF."""
    if ccy == "CHF":
        return 1.0
    if ccy in _fx_cache:
        return _fx_cache[ccy]
    rate = float("nan")
    try:
        if fetcher is None:
            import yfinance as yf
            h = yf.Ticker(f"{ccy}CHF=X").history(period="5d", interval="1d")
            rate = float(h["Close"].dropna().iloc[-1])
        else:
            rate = fetcher(ccy)
    except Exception:
        pass
    _fx_cache[ccy] = rate
    return rate


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
@dataclass
class Result:
    name: str
    group: str
    ticker: str
    price: float = float("nan")
    bias: str = "n/a"
    strength: float = 0.0
    trend_1d: int = 0
    trend_4h: int = 0
    trend_1h: int = 0
    rsi_1d: float = float("nan")
    rsi_4h: float = float("nan")
    rsi_1h: float = float("nan")
    atr_1d_pct: float = float("nan")
    atr_4h: float = float("nan")
    support: float = float("nan")
    resistance: float = float("nan")
    setup: str = "none"
    order: str = ""
    entry: float = float("nan")
    stop: float = float("nan")
    target: float = float("nan")
    rr: float = float("nan")
    entry_dist_atr: float = float("nan")
    rr_net: float = float("nan")
    units: float = float("nan")
    min_units: float = float("nan")
    tradable: bool = True
    spread: float = float("nan")
    risk_chf_per_unit: float = float("nan")
    risk_chf: float = float("nan")
    reward_chf: float = float("nan")
    fees_chf: float = float("nan")
    fee_pct: float = float("nan")
    notional_chf: float = float("nan")
    margin_chf: float = float("nan")
    # inputs the dashboard needs to re-size for any account / risk %
    fx_chf: float = float("nan")
    step_units: float = float("nan")
    margin_pct: float = float("nan")
    fin_pct: float = float("nan")
    entry_factor: float = 1.0
    rsi_pen: float = 0.0
    score: float = 0.0
    notes: list = field(default_factory=list)       # technical notes
    size_notes: list = field(default_factory=list)  # depend on account size
    error: str = ""


def analyse(name: str, daily: pd.DataFrame, hourly: pd.DataFrame,
            quote_ccy: str, unit: str, account: float, risk_pct: float,
            broker_price: float | None, fx_fetcher=None) -> Result:
    ticker, group = UNIVERSE[name][0], UNIVERSE[name][1]
    res = Result(name=name, group=group, ticker=ticker)
    if daily.empty or len(daily) < 210 or hourly.empty or len(hourly) < 300:
        res.error = "not enough data"
        return res

    d = add_indicators(daily)
    h1 = add_indicators(hourly)
    h4 = add_indicators(resample_4h(hourly))

    # Broker alignment: shift all prices by a constant offset
    offset = 0.0
    if broker_price is not None:
        offset = broker_price - float(h1["Close"].iloc[-1])
        res.notes.append(f"levels shifted by {offset:+.4g} to broker price")

    price = float(h1["Close"].iloc[-1]) + offset
    res.price = price
    res.trend_1d = trend_score(d, use_200=True)
    res.trend_4h = trend_score(h4, use_200=True)
    res.trend_1h = trend_score(h1, use_200=False)
    res.rsi_1d = float(d["RSI"].iloc[-1])
    res.rsi_4h = float(h4["RSI"].iloc[-1])
    res.rsi_1h = float(h1["RSI"].iloc[-1])
    a4 = float(h4["ATR"].iloc[-1])
    res.atr_4h = a4
    res.atr_1d_pct = float(d["ATR"].iloc[-1] / d["Close"].iloc[-1] * 100)

    w = (TF_WEIGHTS["1D"] * res.trend_1d + TF_WEIGHTS["4H"] * res.trend_4h
         + TF_WEIGHTS["1H"] * res.trend_1h)
    res.strength = round(w / sum(TF_WEIGHTS.values()), 2)  # -1..+1
    if res.strength >= 0.5:
        res.bias = "LONG"
    elif res.strength <= -0.5:
        res.bias = "SHORT"
    else:
        res.bias = "NEUTRAL"

    sh, sl = swing_levels(h4, PIVOT_K, PIVOT_LOOKBACK)
    levels = cluster([x + offset for x in sh + sl], CLUSTER_ATR * a4)
    above = sorted(x for x in levels if x > price + 0.1 * a4)
    below = sorted((x for x in levels if x < price - 0.1 * a4), reverse=True)
    res.resistance = above[0] if above else float("nan")
    res.support = below[0] if below else float("nan")

    e20_4h = float(h4["EMA20"].iloc[-1]) + offset

    if res.bias == "SHORT":
        if e20_4h > price:
            entry, res.order = e20_4h, "SELL LIMIT"
            # confluence: resistance near the EMA
            if above and abs(above[0] - e20_4h) <= 0.5 * a4:
                entry = (above[0] + e20_4h) / 2
                res.notes.append("EMA20(4H) + resistance confluence")
        else:
            entry, res.order = price, "SELL MARKET"
        stop = entry + STOP_ATR * a4
        near = [x for x in above if entry < x <= entry + 1.5 * a4]
        if near:
            stop = max(stop, near[0] + 0.25 * a4)
        tgt_lv = [x for x in below if x < entry - 0.5 * a4]
        target = tgt_lv[0] + 0.1 * a4 if tgt_lv else entry - 2 * (stop - entry)
        if not tgt_lv:
            res.notes.append("no support below: target = 2R")
        risk, reward = stop - entry, entry - target
        res.setup = "trend short"
    elif res.bias == "LONG":
        if e20_4h < price:
            entry, res.order = e20_4h, "BUY LIMIT"
            if below and abs(below[0] - e20_4h) <= 0.5 * a4:
                entry = (below[0] + e20_4h) / 2
                res.notes.append("EMA20(4H) + support confluence")
        else:
            entry, res.order = price, "BUY MARKET"
        stop = entry - STOP_ATR * a4
        near = [x for x in below if entry - 1.5 * a4 <= x < entry]
        if near:
            stop = min(stop, near[0] - 0.25 * a4)
        tgt_lv = [x for x in above if x > entry + 0.5 * a4]
        target = tgt_lv[0] - 0.1 * a4 if tgt_lv else entry + 2 * (entry - stop)
        if not tgt_lv:
            res.notes.append("no resistance above: target = 2R")
        risk, reward = entry - stop, target - entry
        res.setup = "trend long"
    else:
        res.setup = "range / no trend"
        res.notes.append("mixed timeframes: no trade, or trade the range edges")
        return finish(res)

    res.entry, res.stop, res.target = entry, stop, target
    res.rr = reward / risk if risk > 0 else float("nan")
    res.entry_dist_atr = abs(entry - price) / a4 if a4 > 0 else float("nan")

    # Sizing in CHF, after spread, financing, min size and step
    fx = to_chf(quote_ccy, fx_fetcher)
    if math.isnan(fx) or risk <= 0:
        res.notes.append(f"no {quote_ccy}CHF rate: sizing skipped")
        res.tradable = False
        res.score = 0.0
        return finish(res)

    sp = instrument_spec(name, price)
    spread, step, min_u = sp["spread"], sp["step"], sp["min"]
    res.spread, res.min_units, res.step_units = spread, min_u, step
    res.fx_chf, res.margin_pct, res.fin_pct = fx, sp["margin"], sp["fin"]
    budget = account * risk_pct / 100
    fin_unit = entry * fx * sp["fin"] / 100 / 365 * HOLD_DAYS
    per_unit = (risk + spread) * fx          # stop loss + spread paid on entry
    res.risk_chf_per_unit = per_unit + fin_unit
    units = math.floor(budget / res.risk_chf_per_unit / step + 1e-9) * step
    if units < min_u:
        units = min_u
        res.tradable = False
        res.size_notes.append(f"min size {min_u:g} risks "
                         f"{min_u * res.risk_chf_per_unit:.2f} CHF "
                         f"> budget {budget:.2f}: not tradable at your risk")
    units = round(units, 8)
    res.units = units
    res.notional_chf = units * entry * fx
    res.margin_chf = res.notional_chf * sp["margin"] / 100
    financing = res.notional_chf * sp["fin"] / 100 / 365 * HOLD_DAYS
    spread_cost = units * spread * fx
    gross_reward = units * reward * fx
    res.fees_chf = spread_cost + financing
    res.risk_chf = units * per_unit + financing
    res.reward_chf = gross_reward - res.fees_chf
    res.rr_net = res.reward_chf / res.risk_chf if res.risk_chf > 0 else float("nan")
    res.fee_pct = res.fees_chf / gross_reward * 100 if gross_reward > 0 else float("nan")

    # Score: trend strength x capped net R:R, graded penalties
    s = abs(res.strength) * min(max(res.rr_net, 0), 4)
    f = entry_dist_factor(res.entry_dist_atr)
    res.entry_factor = f
    if f < 1:
        s *= f
        res.notes.append(f"entry {res.entry_dist_atr:.1f} ATR away: may not fill")
    p = rsi_penalty(res.bias, res.rsi_1d, res.rsi_4h)
    res.rsi_pen = p
    if p > 0:
        s *= 1 - 0.6 * p
        word = ("oversold" if res.bias == "SHORT" else "overbought")
        res.notes.append(f"{'near ' if p < 0.5 else ''}{word} "
                         f"(RSI {res.rsi_1d:.0f}/{res.rsi_4h:.0f}): "
                         f"wait, avoid market orders")
    if res.rr_net < 1.5:
        s *= 0.5
    if res.fee_pct > MAX_FEE_PCT:
        s *= 0.5
        res.size_notes.append(f"fees {res.fee_pct:.0f}% of reward (> {MAX_FEE_PCT:g}%)")
    if res.margin_chf > account * MAX_MARGIN_PCT / 100:
        s *= 0.5
        res.size_notes.append(f"margin {res.margin_chf:.0f} CHF > {MAX_MARGIN_PCT:g}% "
                         f"of account")
    if not res.tradable:
        s = 0.0
    res.score = round(s, 2)
    return finish(res)


def finish(res: Result) -> Result:
    return res


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def fmt(x: float, ref: float | None = None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    ref = abs(ref) if ref else abs(x)
    if ref >= 1000:
        return f"{x:,.1f}"
    if ref >= 100:
        return f"{x:.2f}"
    if ref >= 10:
        return f"{x:.3f}"
    if ref >= 1:
        return f"{x:.4f}"
    return f"{x:.5f}"


def arrow(t: int) -> str:
    return {1: "UP", -1: "DN", 0: "--"}[t]


def fmt_units(u: float) -> str:
    if u is None or math.isnan(u):
        return "-"
    return f"{u:,.0f}" if u >= 1 and float(u).is_integer() else f"{u:g}"


def setup_row(i, r: Result) -> str:
    p = r.price
    return (
        f"| {i} | {r.name} | {r.bias} | "
        f"{arrow(r.trend_1d)}/{arrow(r.trend_4h)}/{arrow(r.trend_1h)} | "
        f"{r.rsi_1d:.0f}/{r.rsi_4h:.0f} | {r.order} | {fmt(r.entry, p)} | "
        f"{fmt(r.stop, p)} | {fmt(r.target, p)} | {r.rr_net:.1f} | "
        f"{fmt_units(r.units)} {UNIVERSE[r.name][3]} | {r.risk_chf:.2f} | "
        f"{r.reward_chf:.2f} | {r.fee_pct:.0f}% | {r.margin_chf:,.0f} | "
        f"{r.score:.2f} |")


SETUP_HEADER = [
    "| # | Market | Bias | 1D/4H/1H | RSI 1D/4H | Order | Entry | Stop | "
    "Target | R:R net | Units | Risk CHF | Reward CHF net | Fees % | "
    "Margin CHF | Score |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]


def fx_exposure(rs: list[Result]) -> dict[str, list[str]]:
    """Currency -> list of 'NAME(+/-)' for FX setups (long base / short quote)."""
    exp: dict[str, list[str]] = {}
    for r in rs:
        if r.group != "fx":
            continue
        sign = 1 if r.bias == "LONG" else -1
        for ccy, s in ((r.name[:3], sign), (r.name[3:], -sign)):
            exp.setdefault(f"{ccy}{'+' if s > 0 else '-'}", []).append(r.name)
    return {k: v for k, v in exp.items() if len(v) > 1}


def build_report(results: list[Result], account: float, risk_pct: float) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    ok = [r for r in results if not r.error]
    setups = [r for r in ok if r.setup.startswith("trend")
              and not math.isnan(r.units)]
    trades = sorted([r for r in setups if r.tradable],
                    key=lambda r: r.score, reverse=True)
    blocked = sorted([r for r in setups if not r.tradable],
                     key=lambda r: r.name)
    ranging = [r for r in ok if not r.setup.startswith("trend")]
    errs = [r for r in results if r.error
            or (r.setup.startswith("trend") and math.isnan(r.units))]

    L = [f"# CFD scan {now}",
         f"Account {account:g} CHF, risk {risk_pct:g}% = "
         f"{account * risk_pct / 100:.2f} CHF per trade. "
         f"Instruments scanned: {len(results)} (ok {len(ok)}). "
         f"Fees = spread + {HOLD_DAYS}d financing. Specs are estimates: "
         f"verify with your broker and override with --spec.", ""]

    L += ["## Ranked setups (tradable at your size)", ""] + SETUP_HEADER
    for i, r in enumerate(trades, 1):
        L.append(setup_row(i, r))
    L.append("")

    top = trades[:PORTFOLIO_TOP_N]
    if top:
        L += [f"## If the top {len(top)} all fill", "",
              f"- Total risk {sum(r.risk_chf for r in top):.2f} CHF "
              f"({sum(r.risk_chf for r in top) / account * 100:.1f}% of account), "
              f"total margin {sum(r.margin_chf for r in top):,.0f} CHF, "
              f"total notional {sum(r.notional_chf for r in top):,.0f} CHF."]
        for k, v in fx_exposure(top).items():
            L.append(f"- Correlated: {k[:-1]} {'long' if k[-1] == '+' else 'short'}"
                     f" in {', '.join(v)}: treat as one trade.")
        L.append("")

    L += ["## Details (top 10)", ""]
    for r in trades[:10]:
        p = r.price
        L.append(
            f"- **{r.name}** ({r.ticker}) price {fmt(p)} | ATR4H {fmt(r.atr_4h, p)} "
            f"| daily ATR {r.atr_1d_pct:.1f}% | S {fmt(r.support, p)} / "
            f"R {fmt(r.resistance, p)} | entry {r.entry_dist_atr:.1f} ATR away "
            f"| spread {fmt(r.spread, p)} | notional {r.notional_chf:,.0f} CHF"
            + (f" | notes: {'; '.join(r.notes + r.size_notes)}"
               if r.notes or r.size_notes else ""))
    L.append("")

    if blocked:
        L += ["## Not tradable at your risk (minimum size too large)", "",
              "| Market | Bias | Entry | Stop | Min size | Risk at min CHF |",
              "|---|---|---|---|---|---|"]
        for r in blocked:
            p = r.price
            L.append(f"| {r.name} | {r.bias} | {fmt(r.entry, p)} | "
                     f"{fmt(r.stop, p)} | {fmt_units(r.min_units)} "
                     f"{UNIVERSE[r.name][3]} | {r.risk_chf:.2f} |")
        L.append("")

    if ranging:
        L += ["## No trend (range)", "",
              "| Market | 1D/4H/1H | RSI 1D | Support | Resistance | Price |",
              "|---|---|---|---|---|---|"]
        for r in sorted(ranging, key=lambda r: r.name):
            p = r.price
            L.append(f"| {r.name} | {arrow(r.trend_1d)}/{arrow(r.trend_4h)}/"
                     f"{arrow(r.trend_1h)} | {r.rsi_1d:.0f} | {fmt(r.support, p)} "
                     f"| {fmt(r.resistance, p)} | {fmt(p)} |")
        L.append("")

    if errs:
        L.append("Skipped: " + ", ".join(
            f"{r.name} ({r.error or '; '.join(r.notes)})" for r in errs))
        L.append("")

    L += ["_Units = units of the underlying (bbl, oz, index points, currency, "
          "coins). Convert to broker lots with the contract size in the "
          "instrument info. Levels come from Yahoo data; use --broker to align "
          "with your CFD price. Not investment advice._"]
    return "\n".join(L)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Multi-asset CFD technical scanner")
    ap.add_argument("--groups", help="comma list: energy,metals,indices,bonds,crypto,fx")
    ap.add_argument("--only", help="comma list of names, e.g. WTI,GOLD,EURUSD")
    ap.add_argument("--account", type=float, default=1000.0, help="account size in CHF")
    ap.add_argument("--risk", type=float, default=2.0, help="risk per trade in %%")
    ap.add_argument("--broker", action="append", default=[],
                    help="NAME=price to align levels, e.g. WTI=80.00 (repeatable)")
    ap.add_argument("--spec", action="append", default=[],
                    help="NAME=key:val,... override contract spec; keys: spread "
                         "(price units), spread_pct, min, step, margin (%%), "
                         "fin (%%/yr). e.g. WTI=spread:0.06,min:1,step:1")
    ap.add_argument("--hold-days", type=float, default=HOLD_DAYS,
                    help="expected holding days for financing cost")
    ap.add_argument("--max-fee", type=float, default=MAX_FEE_PCT,
                    help="flag setups whose fees exceed this %% of reward")
    ap.add_argument("--max-margin", type=float, default=MAX_MARGIN_PCT,
                    help="flag positions whose margin exceeds this %% of account")
    ap.add_argument("--out", default="scans", help="output folder")
    ap.add_argument("--json", help="also write a JSON file for the dashboard")
    ap.add_argument("--min-ok", type=float, default=0.0,
                    help="exit with error (and write nothing) if fewer than this "
                         "share of instruments downloaded, e.g. 0.8")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--list", action="store_true", help="print the universe and exit")
    return ap.parse_args(argv)


def select(args) -> list[str]:
    names = list(UNIVERSE)
    if args.groups:
        g = {x.strip().lower() for x in args.groups.split(",")}
        names = [n for n in names if UNIVERSE[n][1] in g]
    if args.only:
        o = [x.strip().upper() for x in args.only.split(",")]
        bad = [x for x in o if x not in UNIVERSE]
        if bad:
            sys.exit(f"Unknown names: {bad}. Use --list to see the universe.")
        names = o
    return names


def run(args, fetcher=fetch, fx_fetcher=None) -> tuple[str, list[Result]]:
    names = select(args)
    broker = {}
    for b in args.broker:
        k, v = b.split("=")
        broker[k.strip().upper()] = float(v)

    global HOLD_DAYS, MAX_FEE_PCT, MAX_MARGIN_PCT
    HOLD_DAYS = getattr(args, "hold_days", HOLD_DAYS)
    MAX_FEE_PCT = getattr(args, "max_fee", MAX_FEE_PCT)
    MAX_MARGIN_PCT = getattr(args, "max_margin", MAX_MARGIN_PCT)
    SPEC_CLI.clear()
    for s in getattr(args, "spec", []):
        k, v = s.split("=", 1)
        k = k.strip().upper()
        if k not in UNIVERSE:
            sys.exit(f"--spec: unknown name {k}")
        d = SPEC_CLI.setdefault(k, {})
        for kv in v.split(","):
            kk, vv = kv.split(":")
            d[kk.strip()] = float(vv)

    results: list[Result] = []

    def job(n):
        tkr, grp, ccy, unit = UNIVERSE[n]
        try:
            daily, hourly = fetcher(tkr)
            return analyse(n, daily, hourly, ccy, unit, args.account, args.risk,
                           broker.get(n), fx_fetcher)
        except Exception as e:  # keep scanning other markets
            return Result(name=n, group=grp, ticker=tkr, error=str(e)[:60])

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(job, n): n for n in names}
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            print(f"\r  scanned {i}/{len(names)}", end="", file=sys.stderr)
    print("", file=sys.stderr)

    n_ok = sum(1 for r in results if not r.error)
    if n_ok < getattr(args, "min_ok", 0.0) * len(results):
        sys.exit(f"Only {n_ok}/{len(results)} instruments downloaded: "
                 f"aborting (data source likely blocked).")

    if getattr(args, "json", None):
        write_json(results, args, args.json)

    report = build_report(results, args.account, args.risk)
    os.makedirs(args.out, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    md = os.path.join(args.out, f"scan_{stamp}.md")
    csv = os.path.join(args.out, f"scan_{stamp}.csv")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write(report)
    rows = []
    for r in results:
        d = asdict(r)
        d["notes"] = "; ".join(r.notes + r.size_notes)
        d.pop("size_notes")
        rows.append(d)
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(csv, index=False)
    print(f"Saved {md} and {csv}", file=sys.stderr)
    return report, results


def _clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, (np.floating, np.integer)):
        return _clean(x.item())
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def write_json(results: list[Result], args, path: str) -> None:
    """Machine-readable scan: everything needed to re-size in the browser."""
    import json
    meta = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "instruments": len(results),
        "ok": sum(1 for r in results if not r.error),
        "hold_days": HOLD_DAYS, "max_fee_pct": MAX_FEE_PCT,
        "max_margin_pct": MAX_MARGIN_PCT, "top_n": PORTFOLIO_TOP_N,
        "rr_min": 1.5, "stop_atr": STOP_ATR,
        "entry_dist_ok_atr": ENTRY_DIST_OK_ATR,
        "entry_dist_max_atr": ENTRY_DIST_MAX_ATR,
        "broker_aligned": sorted(b.split("=")[0].upper() for b in args.broker),
    }
    rows = []
    for r in results:
        d = asdict(r)
        d["unit"] = UNIVERSE[r.name][3]
        d["quote"] = UNIVERSE[r.name][2]
        rows.append(d)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_clean({"meta": meta, "results": rows}), fh, indent=1)
    print(f"Saved {path}", file=sys.stderr)


def main():
    args = parse_args()
    if args.list:
        for n, (t, g, c, u) in UNIVERSE.items():
            print(f"{n:12s} {g:8s} {t:12s} quote {c}  unit {u}")
        return
    report, _ = run(args)
    print(report)


if __name__ == "__main__":
    main()
