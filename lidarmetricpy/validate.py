"""Strict comparison of per-cell metric tables (used by the tests and the validation scripts).

    from lidarmetricpy.validate import compare_tables
    res = compare_tables(reference_df, result_df, ["zq20", "rumple"])
    assert res.equal.all(), res

Rules
- Cell keys (default x, y) must be unique in both tables; a duplicate raises ValueError.
- Cells are matched with an outer join. A reference cell without a result row is accepted only if the
  reference value is NA (an empty cell: lidR lists every cell of the raster, the result only cells with
  points). A result cell without a reference row is always a mismatch.
- Every value is classed as finite, NaN, +Inf or -Inf first; the classes must be equal.
- Finite values must agree within rtol * max(1, |reference|).
- A metric with no finite value compared is reported (n_finite = 0) so that an empty comparison is visible.
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


def compare_tables(ref, got, metrics, keys=("x", "y"), rtol=1e-6, ref_cols=None, got_cols=None):
    """One row per metric: n_ref, n_got, missing (reference value, no result row), extra (result row, no
    reference row), class_mismatch, value_mismatch, n_finite, max_abs_diff, max_rel_diff, equal.
    ref_cols / got_cols: optional dicts metric -> column name in that table."""
    keys = list(keys)
    for name, t in (("reference", ref), ("result", got)):
        if t.duplicated(keys).any():
            raise ValueError(f"{name} table has duplicated cells ({int(t.duplicated(keys).sum())})")
    ref_cols, got_cols = ref_cols or {}, got_cols or {}
    rsel = {ref_cols.get(m, m): f"{m}__ref" for m in metrics}
    gsel = {got_cols.get(m, m): f"{m}__got" for m in metrics}
    a = ref[keys + [c for c in rsel if c in ref.columns]].rename(columns=rsel)
    b = got[keys + [c for c in gsel if c in got.columns]].rename(columns=gsel)
    j = a.merge(b, on=keys, how="outer", indicator=True, validate="one_to_one")
    only_ref, only_got = (j._merge == "left_only").to_numpy(), (j._merge == "right_only").to_numpy()
    rows = []
    for m in metrics:
        rc, gc = f"{m}__ref", f"{m}__got"
        if rc not in j or gc not in j:
            rows.append({"metric": m, "n_ref": len(ref), "n_got": len(got), "missing": -1, "extra": -1,
                         "class_mismatch": -1, "value_mismatch": -1, "n_finite": 0, "max_abs_diff": np.nan,
                         "max_rel_diff": np.nan, "equal": False, "note": "column absent"})
            continue
        r = pd.to_numeric(j[rc], errors="coerce").to_numpy(float)
        g = pd.to_numeric(j[gc], errors="coerce").to_numpy(float)
        missing = int((only_ref & ~np.isnan(r)).sum())
        extra = int(only_got.sum())
        both = ~only_ref & ~only_got
        cr, cg = _classes(r[both]), _classes(g[both])
        cls = int((cr != cg).sum())
        fin = (cr == 0) & (cg == 0)
        d = np.abs(r[both][fin] - g[both][fin])
        scale = np.maximum(1.0, np.abs(r[both][fin]))
        val = int((d > rtol * scale).sum())
        rows.append({"metric": m, "n_ref": len(ref), "n_got": len(got), "missing": missing, "extra": extra,
                     "class_mismatch": cls, "value_mismatch": val, "n_finite": int(fin.sum()),
                     "max_abs_diff": float(d.max()) if len(d) else np.nan,
                     "max_rel_diff": float((d / scale).max()) if len(d) else np.nan,
                     "equal": missing == 0 and extra == 0 and cls == 0 and val == 0, "note": ""})
    return pd.DataFrame(rows)
