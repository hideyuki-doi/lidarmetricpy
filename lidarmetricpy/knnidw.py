"""lidR knnidw() for filling empty canopy cells, reproduced exactly.

On a regular grid the k nearest neighbours are often tied in distance (e.g. the 9th–12th
neighbours all at 1 cell), so the result depends on which tied neighbours lidR keeps.
lidR collects candidates with its GridPartition spatial index (cells visited column by
column, points in insertion order) and orders them with std::sort (libstdc++ introsort,
not stable). Both are ported here so that ties resolve the same way.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)

EPS = 1e-8


class GridPartition:
    """lidR GridPartition (2D) built on points x, y (insertion order = array order)."""

    def __init__(self, x, y):
        n = len(x)
        self.xmin, self.xmax = x.min() - 1, x.max() + 1
        self.ymin, self.ymax = y.min() - 1, y.max() + 1
        depth = math.floor(math.log(n) / math.log(4)) if n > 0 else 0
        ncells = (1 << depth) * (1 << depth)
        xrange, yrange = self.xmax - self.xmin, self.ymax - self.ymin
        xyratio = xrange / yrange
        self.ncols = max(1, _cround(math.sqrt(ncells * xyratio)))
        self.nrows = max(1, _cround(self.ncols / xyratio))
        self.xres = xrange / self.ncols
        self.yres = yrange / self.nrows
        self.area = xrange * yrange
        self.npoints = n
        col = np.floor((x - self.xmin) / self.xres).astype(np.int64)
        row = np.floor((self.ymax - y) / self.yres).astype(np.int64)
        cell = row * self.ncols + col
        order = np.argsort(cell, kind="stable")
        self.idx = order.astype(np.int64)
        self.start = np.searchsorted(cell[order], np.arange(self.ncols * self.nrows + 1)).astype(np.int64)


def _cround(v):
    """C std::round (halves away from zero)."""
    return int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(-v + 0.5))


def knnidw(gx, gy, gz, qx, qy, k=10, p=2.0, rmax=50.0):
    """Values at (qx, qy) from grid points (gx, gy, gz), as lidR::knnidw(k, p, rmax)."""
    G = GridPartition(gx, gy)
    density = G.npoints / G.area
    return _knnidw(gx, gy, gz, qx, qy, k, p, rmax, density, G.xmin, G.ymax, G.xres, G.yres,
                   G.ncols, G.nrows, G.idx, G.start)


@njit(cache=True)
def _lookup(cx, cy, r, gx, gy, xmin, ymax, xres, yres, ncols, nrows, idx, start, buf):
    colmin = int(math.floor((cx - r - xmin) / xres))
    colmax = int(math.ceil((cx + r - xmin) / xres))
    rowmin = int(math.floor((ymax - (cy + r)) / yres))
    rowmax = int(math.ceil((ymax - (cy - r)) / yres))
    m = 0
    for col in range(max(colmin, 0), min(colmax, ncols - 1) + 1):
        for row in range(max(rowmin, 0), min(rowmax, nrows - 1) + 1):
            cell = row * ncols + col
            for t in range(start[cell], start[cell + 1]):
                i = idx[t]
                A = cx - gx[i]
                B = cy - gy[i]
                if A * A + B * B <= r * r + 1e-8:
                    buf[m] = i
                    m += 1
    return m


@njit(cache=True)
def _less(a, b, px, py, gx, gy):
    # DSort2D: lhs closer than rhs
    dx1 = px - gx[b]
    dy1 = py - gy[b]
    d1 = dx1 * dx1 + dy1 * dy1
    dx2 = px - gx[a]
    dy2 = py - gy[a]
    d2 = dx2 * dx2 + dy2 * dy2
    return d2 < d1


@njit(cache=True)
def _lg(n):
    k = 0
    while n > 1:
        n >>= 1
        k += 1
    return k


@njit(cache=True)
def _unguarded_linear_insert(a, last, px, py, gx, gy):
    val = a[last]
    nxt = last - 1
    while _less(val, a[nxt], px, py, gx, gy):
        a[last] = a[nxt]
        last = nxt
        nxt -= 1
    a[last] = val


@njit(cache=True)
def _insertion_sort(a, first, last, px, py, gx, gy):
    if first == last:
        return
    for i in range(first + 1, last):
        if _less(a[i], a[first], px, py, gx, gy):
            val = a[i]
            for j in range(i, first, -1):
                a[j] = a[j - 1]
            a[first] = val
        else:
            _unguarded_linear_insert(a, i, px, py, gx, gy)


@njit(cache=True)
def _adjust_heap(a, first, hole, length, value, px, py, gx, gy):
    """libstdc++ std::__adjust_heap followed by std::__push_heap."""
    top = hole
    child = hole
    while child < (length - 1) // 2:
        child = 2 * (child + 1)
        if _less(a[first + child], a[first + child - 1], px, py, gx, gy):
            child -= 1
        a[first + hole] = a[first + child]
        hole = child
    if (length & 1) == 0 and child == (length - 2) // 2:
        child = 2 * (child + 1)
        a[first + hole] = a[first + child - 1]
        hole = child - 1
    parent = (hole - 1) // 2
    while hole > top and _less(a[first + parent], value, px, py, gx, gy):
        a[first + hole] = a[first + parent]
        hole = parent
        parent = (hole - 1) // 2
    a[first + hole] = value


@njit(cache=True)
def _heap_sort_range(a, first, last, px, py, gx, gy):
    """std::partial_sort(first, last, last): __make_heap then __sort_heap."""
    length = last - first
    if length < 2:
        return
    parent = (length - 2) // 2
    while True:
        _adjust_heap(a, first, parent, length, a[first + parent], px, py, gx, gy)
        if parent == 0:
            break
        parent -= 1
    end = last
    while end - first > 1:
        end -= 1
        value = a[end]
        a[end] = a[first]
        _adjust_heap(a, first, 0, end - first, value, px, py, gx, gy)


@njit(cache=True)
def _introsort_loop(a, first0, last0, depth0, px, py, gx, gy):
    """libstdc++ std::__introsort_loop (threshold 16). The recursion on the right part is
    replaced by an explicit stack; the parts are disjoint, so the result is the same."""
    stack = np.empty((128, 3), np.int64)
    sp = 0
    stack[0, 0], stack[0, 1], stack[0, 2] = first0, last0, depth0
    sp = 1
    while sp > 0:
        sp -= 1
        first, last, depth = stack[sp, 0], stack[sp, 1], stack[sp, 2]
        while last - first > 16:
            if depth == 0:
                _heap_sort_range(a, first, last, px, py, gx, gy)
                break
            depth -= 1
            mid = first + (last - first) // 2
            x1, x2, x3 = first + 1, mid, last - 1
            if _less(a[x1], a[x2], px, py, gx, gy):
                if _less(a[x2], a[x3], px, py, gx, gy):
                    s = x2
                elif _less(a[x1], a[x3], px, py, gx, gy):
                    s = x3
                else:
                    s = x1
            elif _less(a[x1], a[x3], px, py, gx, gy):
                s = x1
            elif _less(a[x2], a[x3], px, py, gx, gy):
                s = x3
            else:
                s = x2
            v = a[first]
            a[first] = a[s]
            a[s] = v
            lo, hi = first + 1, last
            while True:
                while _less(a[lo], a[first], px, py, gx, gy):
                    lo += 1
                hi -= 1
                while _less(a[first], a[hi], px, py, gx, gy):
                    hi -= 1
                if not lo < hi:
                    break
                v = a[lo]
                a[lo] = a[hi]
                a[hi] = v
                lo += 1
            stack[sp, 0], stack[sp, 1], stack[sp, 2] = lo, last, depth
            sp += 1
            last = lo


@njit(cache=True)
def std_sort(a, n, px, py, gx, gy):
    if n == 0:
        return
    _introsort_loop(a, 0, n, _lg(n) * 2, px, py, gx, gy)
    if n > 16:
        _insertion_sort(a, 0, 16, px, py, gx, gy)
        for i in range(16, n):
            _unguarded_linear_insert(a, i, px, py, gx, gy)
    else:
        _insertion_sort(a, 0, n, px, py, gx, gy)


@njit(cache=True)
def _knnidw(gx, gy, gz, qx, qy, k, p, rmax, density, xmin, ymax, xres, yres, ncols, nrows, idx, start):
    out = np.empty(len(qx))
    buf = np.empty(len(gx), np.int64)
    npts = len(gx)
    for q in range(len(qx)):
        cx, cy = qx[q], qy[q]
        radius = math.sqrt(k / (density * 3.14))
        m = 0
        if radius < rmax:
            while m < k and m < npts and radius <= rmax:
                m = _lookup(cx, cy, radius, gx, gy, xmin, ymax, xres, yres, ncols, nrows, idx, start, buf)
                radius *= 1.5
        if radius >= rmax:
            m = _lookup(cx, cy, rmax, gx, gy, xmin, ymax, xres, yres, ncols, nrows, idx, start, buf)
        std_sort(buf, m, cx, cy, gx, gy)
        szw = 0.0
        sw = 0.0
        for j in range(min(k, m)):
            i = buf[j]
            dx = gx[i] - cx
            dy = gy[i] - cy
            d = math.sqrt(dx * dx + dy * dy)
            if d > 0:
                w = 1.0 / d ** p
                szw += gz[i] * w
                sw += w
            else:
                szw = gz[i]
                sw = 1.0
                break
        out[q] = szw / sw if sw > 0 else np.nan
    return out
