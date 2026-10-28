"""Array backends.

The metric code is written against the small numpy-like interface of ``Backend``.
Three implementations:

- ``numpy``  : CPU, any platform (including Apple Silicon CPUs)
- ``cupy``   : NVIDIA CUDA (and AMD ROCm builds of CuPy)
- ``torch``  : PyTorch on ``cuda``, ``mps`` (Apple Silicon GPU) or ``cpu``

Apple's MPS device has no float64, so the torch/mps backend runs in float32.
Results then differ from lidR in the last digits; use a float64 backend when
exact agreement with lidR/lidRmetrics matters.
"""
from __future__ import annotations

import numpy as np


class NumpyLike:
    """numpy and cupy share one implementation."""

    def __init__(self, xp, name, float_dtype=np.float64):
        self.xp = xp
        self.name = name
        self.f = float_dtype
        self.i64 = xp.int64
        self.i32 = xp.int32
        self.bool = xp.bool_
        self.device = "cuda" if name == "cupy" else "cpu"
        self.pi = np.pi
        self.inf = np.inf
        self.nan = np.nan

    # -- creation / transfer
    def asarray(self, a, dtype=None):
        if self.name == "cupy" and isinstance(a, np.ndarray) and a.nbytes > (64 << 20):
            # large host→device copies through pinned (page-locked) memory: several times faster
            import cupyx
            buf = cupyx.empty_pinned(a.shape, a.dtype)
            np.copyto(buf, a)
            return self.xp.asarray(buf, dtype=dtype)
        return self.xp.asarray(a, dtype=dtype)

    def to_numpy(self, a):
        if self.name == "cupy":
            return self.xp.asnumpy(a)
        return np.asarray(a)

    def zeros(self, n, dtype=None):
        return self.xp.zeros(n, dtype=dtype or self.f)

    def full(self, n, v, dtype=None):
        return self.xp.full(n, v, dtype=dtype or self.f)

    def arange(self, *a, dtype=None):
        return self.xp.arange(*a, dtype=dtype)

    def concatenate(self, arrs):
        return self.xp.concatenate(arrs)

    def stack(self, arrs, axis=0):
        return self.xp.stack(arrs, axis=axis)

    # -- elementwise
    def where(self, c, a, b):
        return self.xp.where(c, a, b)

    def __getattr__(self, k):            # sqrt, log, exp, floor, ceil, rint, abs, sign, isfinite, isnan, isinf,
        return getattr(self.xp, k)       # maximum, minimum, clip, cumsum, diff, sort, searchsorted, real, conj

    def astype(self, a, dtype):
        return a.astype(dtype)

    def nan_to_num(self, a, nan=0.0):
        return self.xp.nan_to_num(a, nan=nan)

    def sdiv(self, a, s):
        """a / s with IEEE division (no reciprocal multiplication)."""
        return a / s

    def round_away(self, a):
        """C round(): halves away from zero (lidR:::round_any uses it)."""
        t = self.xp.trunc(a)
        return self.xp.where(self.xp.abs(a - t) == 0.5, t + self.xp.sign(a), self.xp.rint(a))

    def sum(self, a, axis=None, keepdims=False):
        return a.sum(axis=axis, keepdims=keepdims)

    def amax(self, a, axis=None):
        return a.max(axis=axis)

    def amin(self, a, axis=None):
        return a.min(axis=axis)

    # -- grouping
    def bincount(self, idx, weights=None, minlength=0):
        return self.xp.bincount(idx, weights=weights, minlength=minlength)

    def lexsort(self, keys):
        """numpy convention: the LAST key is the primary key. Stable."""
        if self.name == "cupy":
            # cupy.lexsort is slow; chained stable radix sorts (CUB) give the same order
            xp = self.xp
            order = xp.argsort(keys[0], kind="stable")
            for k in keys[1:]:
                order = order[xp.argsort(k[order], kind="stable")]
            return order
        return self.xp.lexsort(self.xp.stack(keys))

    def segsum(self, vals, start, n):
        """Sums over contiguous segments [start, start + n) (points sorted by cell).
        Much faster than bincount on a few thousand cells (no atomic contention)."""
        xp = self.xp
        ncell = int(n.shape[0])
        if self.name == "cupy":
            return _cupy_segsum(vals, start, n)
        out = xp.zeros(ncell, dtype=vals.dtype if vals.dtype.kind == "f" else np.float64)
        nz = xp.flatnonzero(n > 0)
        if len(nz):
            out[nz] = xp.add.reduceat(vals.astype(out.dtype), start[nz])
        return out

    def argsort_stable(self, a):
        return self.xp.argsort(a, kind="stable")

    def unique_inverse(self, a):
        return self.xp.unique(a, return_inverse=True)

    def _scatter_float_atomic(self, n, idx, vals, init, fn):
        """Exact float64 max/min with integer atomics: IEEE doubles map to int64 monotonically
        (flip the magnitude bits of negatives); CuPy's float scatter_max/min use float32 atomics."""
        xp = self.xp
        m = np.int64(0x7FFFFFFFFFFFFFFF)
        v = vals.astype(xp.float64).view(xp.int64)
        v = v ^ ((v >> 63) & m)
        ini = np.array([init], np.float64).view(np.int64)[0]
        ini = ini ^ ((ini >> 63) & m)
        out = xp.full(n, ini, dtype=xp.int64)
        fn(out, idx, v)
        out = out ^ ((out >> 63) & m)
        return out.view(xp.float64).astype(vals.dtype)

    def _int_max(self, out, idx, vals):
        """In-place integer max with 64-bit atomics (cupy.maximum.at; cupyx.scatter_max before CuPy 13)."""
        if hasattr(self.xp.maximum, "at"):
            self.xp.maximum.at(out, idx, vals)
        else:
            import cupyx
            cupyx.scatter_max(out, idx, vals)

    def _int_min(self, out, idx, vals):
        if hasattr(self.xp.minimum, "at"):
            self.xp.minimum.at(out, idx, vals)
        else:
            import cupyx
            cupyx.scatter_min(out, idx, vals)

    def scatter_max(self, n, idx, vals, init):
        if self.name == "cupy":
            if vals.dtype.kind == "f":
                return self._scatter_float_atomic(n, idx, vals, init, self._int_max)
            out = self.xp.full(n, init, dtype=vals.dtype)
            self._int_max(out, idx, vals)
            return out
        out = self.xp.full(n, init, dtype=vals.dtype)
        np.maximum.at(out, idx, vals)
        return out

    def scatter_min(self, n, idx, vals, init):
        if self.name == "cupy":
            if vals.dtype.kind == "f":
                return self._scatter_float_atomic(n, idx, vals, init, self._int_min)
            out = self.xp.full(n, init, dtype=vals.dtype)
            self._int_min(out, idx, vals)
            return out
        out = self.xp.full(n, init, dtype=vals.dtype)
        np.minimum.at(out, idx, vals)
        return out

    def fft(self, a, axis=-1):
        return self.xp.fft.fft(a, axis=axis)

    def ifft(self, a, axis=-1):
        return self.xp.fft.ifft(a, axis=axis)

    def item(self, a):
        return a.item()

    def all(self, a):
        return self.xp.all(a)

    def nonzero(self, a):
        return self.xp.nonzero(a)[0]

    def free(self):
        if self.name == "cupy":
            self.xp.get_default_memory_pool().free_all_blocks()

    def mem_free_bytes(self):
        if self.name == "cupy":
            free, _ = self.xp.cuda.runtime.memGetInfo()
            return free
        return _host_free_bytes()


class TorchB:
    """PyTorch backend (cuda / mps / cpu). Mirrors the NumpyLike interface."""

    def __init__(self, device, float32=None):
        import torch
        self.t = torch
        self.name = "torch"
        self.device = device
        if float32 is None:
            float32 = device == "mps"                     # MPS has no float64
        self.f = torch.float32 if float32 else torch.float64
        self.i64 = torch.int64
        self.i32 = torch.int32
        self.bool = torch.bool
        self.pi = np.pi
        self.inf = float("inf")
        self.nan = float("nan")

    def _t(self, a):
        return a if isinstance(a, self.t.Tensor) else self.t.as_tensor(a, device=self.device)

    def asarray(self, a, dtype=None):
        if isinstance(a, np.ndarray):
            if a.dtype == np.float64 and self.f == self.t.float32:
                a = a.astype(np.float32)
            a = self.t.from_numpy(np.ascontiguousarray(a)).to(self.device)
        else:
            a = self._t(a)
        return a.to(dtype) if dtype is not None else a

    def to_numpy(self, a):
        return a.detach().cpu().numpy() if isinstance(a, self.t.Tensor) else np.asarray(a)

    def zeros(self, n, dtype=None):
        return self.t.zeros(n, dtype=dtype or self.f, device=self.device)

    def full(self, n, v, dtype=None):
        return self.t.full((n,) if isinstance(n, int) else n, v, dtype=dtype or self.f, device=self.device)

    def arange(self, *a, dtype=None):
        return self.t.arange(*a, dtype=dtype, device=self.device)

    def concatenate(self, arrs):
        return self.t.cat([self._t(a) for a in arrs])

    def stack(self, arrs, axis=0):
        return self.t.stack(arrs, dim=axis)

    def where(self, c, a, b):
        t = self.t
        if not isinstance(a, t.Tensor):
            a = t.as_tensor(a, dtype=b.dtype if isinstance(b, t.Tensor) else self.f, device=self.device)
        if not isinstance(b, t.Tensor):
            b = t.as_tensor(b, dtype=a.dtype, device=self.device)
        if a.dtype != b.dtype:
            dt = t.promote_types(a.dtype, b.dtype)
            a, b = a.to(dt), b.to(dt)
        return t.where(c, a, b)

    def sqrt(self, a): return self.t.sqrt(a)
    def log(self, a): return self.t.log(a)
    def exp(self, a): return self.t.exp(a)
    def floor(self, a): return self.t.floor(a)
    def ceil(self, a): return self.t.ceil(a)
    def rint(self, a): return self.t.round(a)          # half to even, like numpy.rint and R's round
    def abs(self, a): return self.t.abs(a)
    def sign(self, a): return self.t.sign(a)
    def isfinite(self, a): return self.t.isfinite(a)
    def isnan(self, a): return self.t.isnan(a)
    def isinf(self, a): return self.t.isinf(a)
    def real(self, a): return a.real
    def conj(self, a): return self.t.conj(a).resolve_conj()

    def maximum(self, a, b):
        if not isinstance(b, self.t.Tensor):
            b = self.t.as_tensor(b, dtype=a.dtype, device=self.device)
        if not isinstance(a, self.t.Tensor):
            a = self.t.as_tensor(a, dtype=b.dtype, device=self.device)
        return self.t.maximum(a, b.to(a.dtype) if a.dtype != b.dtype else b)

    def minimum(self, a, b):
        if not isinstance(b, self.t.Tensor):
            b = self.t.as_tensor(b, dtype=a.dtype, device=self.device)
        if not isinstance(a, self.t.Tensor):
            a = self.t.as_tensor(a, dtype=b.dtype, device=self.device)
        return self.t.minimum(a, b.to(a.dtype) if a.dtype != b.dtype else b)

    def clip(self, a, lo, hi):
        return self.t.clamp(a, lo, hi)

    def cumsum(self, a, axis=0):
        return self.t.cumsum(a, dim=axis)

    def diff(self, a, axis=-1):
        return self.t.diff(a, dim=axis)

    def sort(self, a):
        return self.t.sort(a).values

    def searchsorted(self, a, v, side="left"):
        return self.t.searchsorted(a, v, right=(side == "right"))

    def astype(self, a, dtype):
        return a.to(dtype)

    def nan_to_num(self, a, nan=0.0):
        return self.t.nan_to_num(a, nan=nan)

    def round_away(self, a):
        t = self.t.trunc(a)
        return self.t.where(self.t.abs(a - t) == 0.5, t + self.t.sign(a), self.t.round(a))

    def sdiv(self, a, s):
        """a / s with IEEE division: PyTorch CUDA turns division by a Python scalar into a
        multiplication by its reciprocal, which moves values that sit exactly on a bin edge."""
        return a / self.t.full_like(a, s)

    def sum(self, a, axis=None, keepdims=False):
        return a.sum() if axis is None else a.sum(dim=axis, keepdim=keepdims)

    def amax(self, a, axis=None):
        return a.max() if axis is None else a.max(dim=axis).values

    def amin(self, a, axis=None):
        return a.min() if axis is None else a.min(dim=axis).values

    def bincount(self, idx, weights=None, minlength=0):
        if weights is not None and weights.dtype == self.t.bool:
            weights = weights.to(self.f)
        return self.t.bincount(idx, weights=weights, minlength=minlength)

    def lexsort(self, keys):
        order = self.t.argsort(keys[0], stable=True)
        for k in keys[1:]:
            order = order[self.t.argsort(k[order], stable=True)]
        return order

    def argsort_stable(self, a):
        return self.t.argsort(a, stable=True)

    def segsum(self, vals, start, n):
        if vals.dtype == self.t.bool or not vals.dtype.is_floating_point:
            vals = vals.to(self.f)
        try:
            return self.t.segment_reduce(vals, "sum", lengths=n, unsafe=True)
        except Exception:  # noqa: BLE001  (devices without segment_reduce)
            cell = self.t.repeat_interleave(self.t.arange(len(n), device=self.device), n)
            return self.t.bincount(cell, weights=vals, minlength=len(n))

    def unique_inverse(self, a):
        return self.t.unique(a, sorted=True, return_inverse=True)

    def scatter_max(self, n, idx, vals, init):
        out = self.t.full((n,), init, dtype=vals.dtype, device=self.device)
        return out.scatter_reduce_(0, idx, vals, reduce="amax", include_self=True)

    def scatter_min(self, n, idx, vals, init):
        out = self.t.full((n,), init, dtype=vals.dtype, device=self.device)
        return out.scatter_reduce_(0, idx, vals, reduce="amin", include_self=True)

    def fft(self, a, axis=-1):
        return self.t.fft.fft(a, dim=axis)

    def ifft(self, a, axis=-1):
        return self.t.fft.ifft(a, dim=axis)

    def item(self, a):
        return a.item()

    def all(self, a):
        return a.all()

    def nonzero(self, a):
        return a.nonzero(as_tuple=True)[0]

    def free(self):
        if self.device == "cuda":
            self.t.cuda.empty_cache()
        elif self.device == "mps":
            self.t.mps.empty_cache()

    def mem_free_bytes(self):
        if self.device == "cuda":
            free, _ = self.t.cuda.mem_get_info()
            return free
        # MPS shares the unified memory of the Mac; CPU uses host memory
        return _host_free_bytes()


def _host_free_bytes():
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
    return 8 * 1024 ** 3


def get_backend(name="auto"):
    """name: 'auto' | 'cupy' | 'cuda' | 'torch' | 'torch-cuda' | 'mps' | 'torch-cpu' | 'numpy' | 'cpu'.

    auto: CuPy with a CUDA device → PyTorch CUDA → PyTorch MPS → NumPy.
    """
    name = (name or "auto").lower()
    if name in ("numpy", "cpu"):
        return NumpyLike(np, "numpy")
    if name in ("cupy", "cuda"):
        import cupy
        return NumpyLike(cupy, "cupy")
    if name in ("torch-cuda",):
        return TorchB("cuda")
    if name in ("mps", "torch-mps"):
        return TorchB("mps")
    if name in ("torch-cpu",):
        return TorchB("cpu")
    if name in ("torch-cpu32", "torch-cuda32"):            # float32 like Apple MPS, for testing
        return TorchB(name[6:-2], float32=True)
    if name == "torch":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        return TorchB(dev)
    if name == "auto":
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() > 0:
                return NumpyLike(cupy, "cupy")
        except Exception:  # noqa: BLE001
            pass
        try:
            import torch
            if torch.cuda.is_available():
                return TorchB("cuda")
            if torch.backends.mps.is_available():
                return TorchB("mps")
        except Exception:  # noqa: BLE001
            pass
        return NumpyLike(np, "numpy")
    raise ValueError(f"unknown backend {name!r}")


_SEGSUM = {}


def _cupy_segsum(vals, start, n):
    """One CUDA block per segment, fixed-order tree reduction (deterministic)."""
    import cupy as cp
    if vals.dtype.kind != "f":
        vals = vals.astype(cp.float64)
    t = "double" if vals.dtype == cp.float64 else "float"
    if t not in _SEGSUM:
        _SEGSUM[t] = cp.RawKernel(r"""
extern "C" __global__ void segsum(const T* v, const long long* start, const long long* cnt, T* out, long long ncell) {
  long long c = blockIdx.x;
  if (c >= ncell) return;
  __shared__ T sh[256];
  T s = 0;
  long long a = start[c], m = cnt[c];
  for (long long i = threadIdx.x; i < m; i += blockDim.x) s += v[a + i];
  sh[threadIdx.x] = s;
  __syncthreads();
  for (int st = 128; st > 0; st >>= 1) {
    if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[c] = sh[0];
}""".replace("T", t), "segsum")
    ncell = int(n.shape[0])
    out = cp.zeros(ncell, dtype=vals.dtype)
    if ncell:
        _SEGSUM[t]((ncell,), (256,), (cp.ascontiguousarray(vals), cp.ascontiguousarray(start.astype(cp.int64)),
                                     cp.ascontiguousarray(n.astype(cp.int64)), out, np.int64(ncell)))
    return out
