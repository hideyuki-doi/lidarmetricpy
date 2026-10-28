"""Memory budgets and processing paths must not change any value."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lidarmetricpy import Options, compute_tile, compute_tiles, plan_blocks
from test_reference import BACKENDS, NC, NR, X0, Y1, _available

LAZ = str(Path(__file__).resolve().parent / "data" / "synthetic.laz")


def run(backend, **kw):
    df = compute_tile(LAZ, X0, Y1, NC, NR, backend=backend, **kw)
    return df.sort_values(["y", "x"]).reset_index(drop=True)


def assert_identical(a, b):
    assert list(a.columns) == list(b.columns) and len(a) == len(b)
    for k in a.columns:
        x, y = a[k].to_numpy(float), b[k].to_numpy(float)
        assert np.array_equal(np.isnan(x), np.isnan(y)), k
        assert np.array_equal(x[~np.isnan(x)], y[~np.isnan(y)]), k


@pytest.mark.parametrize("backend", BACKENDS)
def test_ram_budget_multi_pass(backend):
    base = run(backend)
    small = run(backend, cpu_memory_gb=0.008)                    # raw points do not fit: several reading passes
    assert small.attrs["stats"]["passes"] > 1 or small.attrs["stats"]["path"] == "device"
    assert_identical(base, small)


@pytest.mark.skipif(not _available("cupy"), reason="cupy not available")
@pytest.mark.parametrize("gb", [0.05, 0.2, 1.0])
def test_gpu_budget_paths(gb):
    base = run("cupy")
    assert base.attrs["stats"]["path"] == "device"
    assert_identical(base, run("cupy", gpu_memory_gb=gb))


@pytest.mark.parametrize("spike", [True, False])
@pytest.mark.skipif(not _available("cupy"), reason="cupy not available")
def test_device_and_host_paths_agree(spike):
    opt = Options(spike_filter=spike)
    dev = run("cupy", options=opt)
    host = run("cupy", options=Options(spike_filter=spike, device_prepare=False))
    assert dev.attrs["stats"]["path"] == "device" and host.attrs["stats"]["path"] == "host"
    assert_identical(dev, host)


def test_numpy_and_gpu_agree():
    others = [b for b in ("cupy", "torch-cuda") if _available(b)]
    if not others:
        pytest.skip("no GPU backend")
    base = run("numpy", options=Options())
    for b in others:
        got = run(b, options=Options())
        for k in base.columns:
            x, y = base[k].to_numpy(float), got[k].to_numpy(float)
            assert np.array_equal(np.isnan(x), np.isnan(y)), (b, k)
            assert np.allclose(x, y, rtol=1e-9, atol=1e-9, equal_nan=True), (b, k)


def test_old_option_names():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a = run("numpy", host_memory_gb=0.008)
    assert any("cpu_memory_gb" in str(x.message) for x in w)
    assert a.attrs["stats"]["passes"] > 1
    with pytest.warns(UserWarning, match="no effect"):
        run("numpy", gpu_memory_gb=1)


def test_plan_blocks():
    assert plan_blocks([5, 5, 5], 4, 10) == [(0, 2, 0, 4), (2, 3, 0, 4)]
    assert plan_blocks([25], 4, 10) == [(0, 1, 0, 2), (0, 1, 2, 4)]
    assert plan_blocks([0, 0], 4, 10) == [(0, 2, 0, 4)]


def test_empty_grid():
    df = compute_tile(LAZ, X0 + 10_000, Y1, NC, NR, backend="numpy")
    assert len(df) == 0 and {"x", "y"} <= set(df.columns)


def test_compute_tiles_order():
    jobs = [dict(name=f"t{i}", source=LAZ, x0=X0, y1=Y1, ncol=NC, nrow=NR) for i in range(2)]
    out = list(compute_tiles(jobs, workers=2, backend="numpy"))
    assert [n for n, _ in out] == ["t0", "t1"]
    pd.testing.assert_frame_equal(out[0][1], out[1][1])
