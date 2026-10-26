"""Data loading and cleaning.

The monthly return panel is a matrix of simple returns indexed by month-end
date (rows) and instrument ID (columns). Missing values mean the instrument
did not trade (or has no data) in that month. Cleaning, gap reconstruction
from daily prices and instrument merging happen upstream in the data
notebook; this module only loads its output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .config import BacktestConfig


@dataclass
class MarketData:
    returns: pd.DataFrame          # month-end x instrument, simple returns
    asset_class: pd.Series         # instrument -> asset class
    names: pd.Series               # instrument -> descriptive name


def load_returns(path: str) -> pd.DataFrame:
    """Load the monthly return panel and normalise the index to month-end."""
    df = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    df.index = pd.to_datetime(df.index, format="%m/%d/%Y")
    df = df.sort_index().astype(float)
    if df.index.has_duplicates:
        raise ValueError("Duplicate dates in return panel")
    # Rows are month-end business days; collapse to calendar month periods so
    # that "next month" is unambiguous.
    periods = df.index.to_period("M")
    if periods.has_duplicates:
        raise ValueError("More than one row per calendar month")
    return df


def load_asset_map(path: str) -> pd.DataFrame:
    am = pd.read_csv(path)
    am["ID"] = am["ID"].str.strip()
    return am.set_index("ID")


def load_market_data(cfg: BacktestConfig) -> MarketData:
    """Load returns and asset map, apply the configured cleaning rules."""
    rets = load_returns(cfg.returns_path)
    am = load_asset_map(cfg.asset_map_path)

    asset_class = am["AssetClass"].copy()
    for k, v in cfg.asset_class_overrides.items():
        asset_class.loc[k] = v

    # Keep only instruments present in both the panel and the asset map.
    keep = [c for c in rets.columns if c in asset_class.index and c not in cfg.drop_instruments]
    rets = rets[keep]
    return MarketData(
        returns=rets,
        asset_class=asset_class.loc[keep],
        names=am["Name"].reindex(keep),
    )


def coverage_table(md: MarketData) -> pd.DataFrame:
    """One row per instrument: class, first/last date, count and gaps."""
    rows: Dict[str, dict] = {}
    for c in md.returns.columns:
        s = md.returns[c]
        fv, lv = s.first_valid_index(), s.last_valid_index()
        rows[c] = {
            "name": md.names.get(c),
            "asset_class": md.asset_class.get(c),
            "first": fv.date() if fv is not None else None,
            "last": lv.date() if lv is not None else None,
            "n_obs": int(s.notna().sum()),
            "gaps": int(s.loc[fv:lv].isna().sum()) if fv is not None else 0,
            "ann_vol": float(s.std() * 12 ** 0.5),
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_values(["asset_class", "first"])
