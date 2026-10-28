"""Strict comparison of per-cell metric tables (used by the tests and the validation scripts).

    from lidarmetricpy.validate import compare_tables, occupied_cells
    occ = occupied_cells(x, y, x0, y1, ncol, nrow, cell)          # the cells that have points
    res = compare_tables(reference_df, result_df, ["zq20", "rumple"], occupied=occ)
    assert res.equal.all(), res

Rules
- Cell keys (default x, y) must be unique in both tables; a duplicate raises ValueError.
- Which cells must be present is decided apart from the metric values: every cell of `occupied` (a table of
  keys), or every reference cell when `occupied` is None. A required cell without a result row is a mismatch
  whatever its values (a metric can be NA in a cell with points). A result row for a cell that is not in the
  reference, or not occupied, is a mismatch.
- Values must be numbers or NA; anything else (e.g. a string) raises ValueError.
- Every value is classed as finite, NaN, +Inf or -Inf first; the classes must be equal.
- Finite values must agree within rtol * max(1, |reference|).
- status: "equal", "different", or "empty" when no cell was compared (then equal is False).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _classes(v):
    c = np.zeros(len(v), np.int8)                      # 0 finite, 1 NaN, 2 +Inf, 3 -Inf
    c[np.isnan(v)] = 1
    c[np.isposinf(v)] = 2
    c[np.isneginf(v)] = 3
    return c


def _numbers(col, what):
    try:
        return pd.to_numeric(col, errors="raise").to_numpy(float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{what}: non-numeric value ({e})") from None


def occupied_cells(x, y, x0, y1, ncol, nrow, cell):
    """Centres (x, y) of the grid cells that hold at least one point, with closed outer edges (as lidR);
    deliberately written apart from lidarmetricpy's own grid code."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x1, y0 = x0 + ncol * cell, y1 - nrow * cell
    k = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    c = np.minimum(np.floor((x[k] - x0) / cell).astype(np.int64), ncol - 1)
    r = np.minimum(np.floor((y1 - y[k]) / cell).astype(np.int64), nrow - 1)
    ids = np.unique(r * ncol + c)
    return pd.DataFrame({"x": x0 + (ids % ncol + 0.5) * cell, "y": y1 - (ids // ncol + 0.5) * cell})


def compare_tables(ref, got, metrics, keys=("x", "y"), rtol=1e-6, ref_cols=None, got_cols=None, occupied=None):
    """One row per metric: n_ref, n_got, n_cells (cells compared), missing (required cells absent from the
    result or the reference), extra (result rows for cells that are not required), class_mismatch, value_mismatch, n_finite,
    max_abs_diff, max_rel_diff, equal, status. ref_cols / got_cols: optional dicts metric -> column name."""
    keys = list(keys)
    for name, t in (("reference", ref), ("result", got)) + ((("occupied", occupied),) if occupied is not None else ()):
        if t.duplicated(keys).any():
            raise ValueError(f"{name} table has duplicated cells ({int(t.duplicated(keys).sum())})")
    req = (ref[keys] if occupied is None else occupied[keys]).assign(_req=True)
    ref_cols, got_cols = ref_cols or {}, got_cols or {}
    rsel = {ref_cols.get(m, m): f"{m}__ref" for m in metrics}
    gsel = {got_cols.get(m, m): f"{m}__got" for m in metrics}
    a = ref[keys + [c for c in rsel if c in ref.columns]].rename(columns=rsel).assign(_inref=True)
    b = got[keys + [c for c in gsel if c in got.columns]].rename(columns=gsel).assign(_ingot=True)
    j = a.merge(b, on=keys, how="outer", validate="one_to_one").merge(req, on=keys, how="outer")
    inref, ingot, isreq = (j[c].fillna(False).astype(bool).to_numpy() for c in ("_inref", "_ingot", "_req"))
    missing = int((isreq & ~(ingot & inref)).sum())         # a required cell absent from either table
    extra = int((ingot & ~(inref & isreq)).sum())
    both = inref & ingot & isreq
    rows = []
    for m in metrics:
        rc, gc = f"{m}__ref", f"{m}__got"
        if rc not in j or gc not in j:
            rows.append({"metric": m, "n_ref": len(ref), "n_got": len(got), "n_cells": 0, "missing": missing,
                         "extra": extra, "class_mismatch": -1, "value_mismatch": -1, "n_finite": 0,
                         "max_abs_diff": np.nan, "max_rel_diff": np.nan, "equal": False, "status": "column absent"})
            continue
        r = _numbers(j[rc], f"reference {m}")
        g = _numbers(j[gc], f"result {m}")
        cr, cg = _classes(r[both]), _classes(g[both])
        cls = int((cr != cg).sum())
        fin = (cr == 0) & (cg == 0)
        d = np.abs(r[both][fin] - g[both][fin])
        scale = np.maximum(1.0, np.abs(r[both][fin]))
        val = int((d > rtol * scale).sum())
        ncells = int(both.sum())
        ok = missing == 0 and extra == 0 and cls == 0 and val == 0
        status = "empty" if ncells == 0 else ("equal" if ok else "different")
        rows.append({"metric": m, "n_ref": len(ref), "n_got": len(got), "n_cells": ncells, "missing": missing,
                     "extra": extra, "class_mismatch": cls, "value_mismatch": val, "n_finite": int(fin.sum()),
                     "max_abs_diff": float(d.max()) if len(d) else np.nan,
                     "max_rel_diff": float((d / scale).max()) if len(d) else np.nan,
                     "equal": status == "equal", "status": status})
    return pd.DataFrame(rows)
