"""Bit-exact port of R's stats::density(x, bw) (gaussian kernel, n = 512, cut = 3, ext = 4) and of
lidRmetrics' peak count, for kde_peaks_count.

A peak count is discrete, so tiny rounding differences matter where the density has a flat top: a
symmetric density can show two equal values at its top (R counts two peaks) or one slightly larger value
(one peak). Matching R therefore needs the same floating-point operations in the same order, not just
the same formula. Ported from the R 4.5.3 sources (GPL-2 | GPL-3):

  src/library/stats/R/density.R        density.default
  src/library/stats/src/massdist.c     BinDist (linear binning, points in their original order)
  src/main/seq.c                       seq.int(from, to, length.out =)
  src/nmath/dnorm.c                    dnorm4
  src/library/stats/src/fft.c          fft_factor, fftmx (Singleton's mixed-radix FFT; only the radix-2/4
                                       code paths and the square-factor permutation, which cover 2n = 1024)
  src/library/stats/src/approx.c       approx1 (linear)
  lidRmetrics 0.1.5 get_peaks          which(diff(sign(diff(d$y))) < 0) + 1

Requires an IEEE double without fused multiply-add (numba does not contract a*b + c by default) and the
platform's libm exp / sin / cos, which R uses too.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import numba

    def _jit(f):
        return numba.njit(cache=True)(f)

    def _pjit(f):
        return numba.njit(cache=True, parallel=True)(f)

    _prange = numba.prange
except ImportError:  # pragma: no cover
    def _jit(f):
        return f

    _pjit = _jit
    _prange = range

M_PI_4 = 0.785398163397448309615660845820
M_SQRT_3 = 1.732050807568877293527446341506
M_1_SQRT_2PI = 0.398942280401432677939946059934
M_LN2 = 0.693147180559945309417232121458
# sqrt(-2 * M_LN2 * (DBL_MIN_EXP + 1 - DBL_MANT_DIG)), as in dnorm.c (C evaluates -2*M_LN2 first)
DNORM_ZERO = math.sqrt(-2 * M_LN2 * (-1021 + 1 - 53))
NG = 512                       # density(n = 512): grid of the binned data; the FFT has 2 * NG points


@_jit
def _fft_factor(n):
    """fft_factor(): factors in nfac[0..m-1], m, kt (number of square factors)."""
    nfac = np.zeros(20, np.int64)
    m = 0
    k = n
    kt = 0
    while k % 16 == 0:
        nfac[m] = 4
        m += 1
        k //= 16
    sqrtk = int(math.sqrt(k))
    j = 3
    while j <= sqrtk:
        jj = j * j
        changed = False
        while k % jj == 0:
            nfac[m] = j
            m += 1
            k //= jj
            changed = True
        if changed:
            sqrtk = int(math.sqrt(k))
        j += 2
    if k <= 4:
        kt = m
        nfac[m] = k
        if k != 1:
            m += 1
    else:
        if k % 4 == 0:
            nfac[m] = 2
            m += 1
            k //= 4
        kt = m
        j = 2
        while True:
            if k % j == 0:
                nfac[m] = j
                m += 1
                k //= j
            j = ((j + 1) // 2) * 2 + 1
            if j > k:
                break
    if kt != 0:
        j = kt
        while j != 0:
            j -= 1
            nfac[m] = nfac[j]
            m += 1
    return nfac, m, kt


@_jit
def _fftmx(Z, n, isn, nfac, m, kt):
    """In-place complex FFT of Z (interleaved re, im; length 2n), R's fftmx with nseg = nspn = 1.
    Only factors 2 and 4 (and square-factor permutations) are ported; other factors raise."""
    # a[j] = Z[j - 1], b[j] = Z[j] for the 1-based odd indices j of the Fortran code (inc = 2)
    inc = abs(isn)
    ntot = n
    nt = inc * ntot
    ks = inc * n
    rad = M_PI_4
    if isn <= 0:
        rad = -rad
    kspan = ks
    nn = nt - inc
    jc = ks // n
    lim = 32
    klim = lim * jc
    i = 0
    npv = np.zeros(32, np.int64)
    c1 = 0.0
    s1 = 0.0
    c2 = 0.0
    s2 = 0.0
    c3 = 0.0
    s3 = 0.0
    cd = 0.0
    sd = 0.0
    dr = 0.0
    kk = 0
    k1 = 0
    k2 = 0
    k3 = 0
    mm = 0
    kspnn = 0
    akp = akm = ajp = ajm = bkp = bkm = bjp = bjm = 0.0
    j = 0
    k = 0
    L = 0                          # label
    while True:
        if L == 0:                 # L_start
            dr = (8.0 * jc) / kspan
            cd = math.sin(0.5 * dr * rad)
            cd = 2.0 * cd * cd
            sd = math.sin(dr * rad)
            kk = 1
            i += 1
            if nfac[i - 1] != 2:
                L = 110
                continue
            # factor of 2
            kspan //= 2
            k1 = kspan + 2
            while True:
                while True:
                    k2 = kk + kspan
                    ak = Z[k2 - 1]
                    bk = Z[k2]
                    Z[k2 - 1] = Z[kk - 1] - ak
                    Z[k2] = Z[kk] - bk
                    Z[kk - 1] += ak
                    Z[kk] += bk
                    kk = k2 + kspan
                    if not (kk <= nn):
                        break
                kk -= nn
                if not (kk <= jc):
                    break
            if kk > kspan:
                L = 999
                continue
            L = 60
        elif L == 60:
            c1 = 1.0 - cd
            s1 = sd
            mm = min(k1 // 2, klim)
            L = 80
        elif L == 70:
            ak = c1 - (cd * c1 + sd * s1)
            s1 = (sd * c1 - cd * s1) + s1
            c1 = ak
            L = 80
        elif L == 80:
            while True:
                k2 = kk + kspan
                ak = Z[kk - 1] - Z[k2 - 1]
                bk = Z[kk] - Z[k2]
                Z[kk - 1] += Z[k2 - 1]
                Z[kk] += Z[k2]
                Z[k2 - 1] = c1 * ak - s1 * bk
                Z[k2] = s1 * ak + c1 * bk
                kk = k2 + kspan
                if not (kk < nt):
                    break
            k2 = kk - nt
            c1 = -c1
            kk = k1 - k2
            if kk > k2:
                L = 80
                continue
            kk += jc
            if kk <= mm:
                L = 70
                continue
            if kk >= k2:
                k1 = k1 + inc + inc
                kk = (k1 - kspan) // 2 + jc
                if kk <= jc + jc:
                    L = 60
                    continue
                L = 0
                continue
            s1 = ((kk - 1) // jc) * dr * rad
            c1 = math.cos(s1)
            s1 = math.sin(s1)
            mm = min(k1 // 2, mm + klim)
            L = 80
        elif L == 110:
            if nfac[i - 1] != 4:
                raise ValueError("fft factor not ported")
            kspnn = kspan
            kspan //= 4
            L = 120
        elif L == 120:
            c1 = 1.0
            s1 = 0.0
            mm = min(kspan, klim)
            L = 150
        elif L == 130:
            c2 = c1 - (cd * c1 + sd * s1)
            s1 = (sd * c1 - cd * s1) + s1
            c1 = c2
            L = 140
        elif L == 140:
            c2 = c1 * c1 - s1 * s1
            s2 = c1 * s1 * 2.0
            c3 = c2 * c1 - s2 * s1
            s3 = c2 * s1 + s2 * c1
            L = 150
        elif L == 150:
            k1 = kk + kspan
            k2 = k1 + kspan
            k3 = k2 + kspan
            akp = Z[kk - 1] + Z[k2 - 1]
            akm = Z[kk - 1] - Z[k2 - 1]
            ajp = Z[k1 - 1] + Z[k3 - 1]
            ajm = Z[k1 - 1] - Z[k3 - 1]
            Z[kk - 1] = akp + ajp
            ajp = akp - ajp
            bkp = Z[kk] + Z[k2]
            bkm = Z[kk] - Z[k2]
            bjp = Z[k1] + Z[k3]
            bjm = Z[k1] - Z[k3]
            Z[kk] = bkp + bjp
            bjp = bkp - bjp
            if isn < 0:
                L = 180
                continue
            akp = akm - bjm
            akm = akm + bjm
            bkp = bkm + ajm
            bkm = bkm - ajm
            if s1 == 0.0:
                L = 190
                continue
            L = 160
        elif L == 160:
            Z[k1 - 1] = akp * c1 - bkp * s1
            Z[k1] = akp * s1 + bkp * c1
            Z[k2 - 1] = ajp * c2 - bjp * s2
            Z[k2] = ajp * s2 + bjp * c2
            Z[k3 - 1] = akm * c3 - bkm * s3
            Z[k3] = akm * s3 + bkm * c3
            kk = k3 + kspan
            if kk <= nt:
                L = 150
                continue
            L = 170
        elif L == 170:
            kk = kk - nt + jc
            if kk <= mm:
                L = 130
                continue
            if kk < kspan:
                L = 200
                continue
            kk = kk - kspan + inc
            if kk <= jc:
                L = 120
                continue
            if kspan == jc:
                L = 999
                continue
            L = 0
        elif L == 180:
            akp = akm + bjm
            akm = akm - bjm
            bkp = bkm - ajm
            bkm = bkm + ajm
            if s1 != 0.0:
                L = 160
                continue
            L = 190
        elif L == 190:
            Z[k1 - 1] = akp
            Z[k1] = bkp
            Z[k2 - 1] = ajp
            Z[k2] = bjp
            Z[k3 - 1] = akm
            Z[k3] = bkm
            kk = k3 + kspan
            if kk <= nt:
                L = 150
                continue
            L = 170
        elif L == 200:
            s1 = ((kk - 1) // jc) * dr * rad
            c1 = math.cos(s1)
            s1 = math.sin(s1)
            mm = min(kspan, mm + klim)
            L = 140
        elif L == 999:             # L_fin: permutation for square factors
            npv[1] = ks
            if kt == 0:
                L = 440
                continue
            k = kt + kt + 1
            if m < k:
                k -= 1
            npv[k + 1] = jc
            j = 1
            while j < k:
                npv[j + 1] = npv[j] // nfac[j - 1]
                npv[k] = npv[k + 1] * nfac[j - 1]
                j += 1
                k -= 1
            k3 = npv[k + 1]
            kspan = npv[2]
            kk = jc + 1
            k2 = kspan + 1
            j = 1
            if n != ntot:
                raise ValueError("multivariate permutation not ported")
            L = 370
        elif L == 370:
            while True:
                ak = Z[kk - 1]
                Z[kk - 1] = Z[k2 - 1]
                Z[k2 - 1] = ak
                bk = Z[kk]
                Z[kk] = Z[k2]
                Z[k2] = bk
                kk += inc
                k2 += kspan
                if not (k2 < ks):
                    break
            L = 380
        elif L == 380:
            while True:
                k2 -= npv[j]
                j += 1
                k2 += npv[j + 1]
                if not (k2 > npv[j]):
                    break
            j = 1
            jump = False
            while True:
                if kk < k2:
                    jump = True
                    break
                kk += inc
                k2 += kspan
                if not (k2 < ks):
                    break
            if jump:
                L = 370
                continue
            if kk < ks:
                L = 380
                continue
            jc = k3
            L = 440
        elif L == 440:
            if 2 * kt + 1 >= m:
                return
            raise ValueError("square-free permutation not ported")


@_jit
def _dnorm(x, sigma):
    """dnorm4(x, mean = 0, sd = sigma, log = FALSE) for finite x, sigma > 0."""
    x = x / sigma
    x = abs(x)
    if x < 5:
        return M_1_SQRT_2PI * math.exp(-0.5 * x * x) / sigma
    if x > DNORM_ZERO:
        return 0.0
    x1 = math.ldexp(np.rint(math.ldexp(x, 16)), -16)
    x2 = x - x1
    return M_1_SQRT_2PI / sigma * (math.exp(-0.5 * x1 * x1) * math.exp((-0.5 * x2 - x1) * x2))


@_jit
def _seq_len(frm, to, lout):
    """seq.int(from, to, length.out = lout), lout > 2, finite (to - from)."""
    out = np.empty(lout)
    by = (to - frm) / (lout - 1)
    out[0] = frm
    out[lout - 1] = to
    for i in range(1, lout - 1):
        out[i] = frm + i * by
    return out


@_jit
def _density_y(x, bw, nfac, m, kt):
    """density(x, bw = bw)$y (n = 512, gaussian, cut = 3, ext = 4); x in the order R receives it."""
    nx = x.shape[0]
    w = 1.0 / nx
    xmn = x[0]
    xmx = x[0]
    for v in x:
        if v < xmn:
            xmn = v
        if v > xmx:
            xmx = v
    frm = xmn - 3.0 * bw
    to = xmx + 3.0 * bw
    lo = frm - 4.0 * bw
    up = to + 4.0 * bw
    n = NG
    # BinDist
    Y = np.zeros(4 * n)                              # interleaved complex, length 2n
    xdelta = (up - lo) / (n - 1)
    ixmax = n - 2
    for v in x:
        xpos = (v - lo) / xdelta
        ix = int(math.floor(xpos))
        fx = xpos - ix
        if 0 <= ix <= ixmax:
            Y[2 * ix] += (1 - fx) * w
            Y[2 * (ix + 1)] += fx * w
        elif ix == -1:
            Y[0] += fx * w
        elif ix == ixmax + 1:
            Y[2 * ix] += (1 - fx) * w
    # kords <- seq.int(0, (2n - 1) / (n - 1) * (up - lo), length.out = 2n); mirrored; dnorm(kords, sd = bw)
    kords = _seq_len(0.0, ((2.0 * n - 1) / (n - 1)) * (up - lo), 2 * n)
    for t in range(n - 1):
        kords[n + 1 + t] = -kords[n - 1 - t]
    K = np.zeros(4 * n)
    for t in range(2 * n):
        K[2 * t] = _dnorm(kords[t], bw)
    _fftmx(Y, 2 * n, -2, nfac.copy(), m, kt)
    _fftmx(K, 2 * n, -2, nfac.copy(), m, kt)
    P = np.empty(4 * n)
    for t in range(2 * n):                           # fft(y) * Conj(fft(kords)), C99 complex product
        a = Y[2 * t]
        b = Y[2 * t + 1]
        c = K[2 * t]
        d = -K[2 * t + 1]
        P[2 * t] = a * c - b * d
        P[2 * t + 1] = a * d + b * c
    _fftmx(P, 2 * n, 2, nfac.copy(), m, kt)          # inverse, unnormalized
    kd = np.empty(n)
    for t in range(n):
        v = P[2 * t] / (2.0 * n)
        kd[t] = v if v > 0.0 else 0.0                # pmax.int(0, .)
    xords = _seq_len(lo, up, n)
    xo = _seq_len(frm, to, 512)
    yo = np.empty(512)
    for q in range(512):                             # approx(xords, kords, xo)$y, approx1 linear
        v = xo[q]
        ii = 0
        jj = n - 1
        if v < xords[ii] or v > xords[jj]:
            yo[q] = np.nan
            continue
        while ii < jj - 1:
            ij = (ii + jj) // 2
            if v < xords[ij]:
                jj = ij
            else:
                ii = ij
        if v == xords[jj]:
            yo[q] = kd[jj]
        elif v == xords[ii]:
            yo[q] = kd[ii]
        else:
            yo[q] = kd[ii] + (kd[jj] - kd[ii]) * ((v - xords[ii]) / (xords[jj] - xords[ii]))
    return yo


@_jit
def _peaks(y):
    """lidRmetrics get_peaks: number of i with diff(sign(diff(y)))[i] < 0."""
    cnt = 0
    prev = 0.0
    for t in range(1, y.shape[0]):
        d = y[t] - y[t - 1]
        s = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
        if t >= 2 and s - prev < 0:
            cnt += 1
        prev = s
    return cnt


@_pjit
def _peaks_cells(z, start, count, bw, nfac, m, kt):
    out = np.full(start.shape[0], np.nan)
    for c in _prange(start.shape[0]):
        if count[c] > 2:
            out[c] = _peaks(_density_y(z[start[c]:start[c] + count[c]], bw, nfac, m, kt))
    return out


_FAC = None


def density_y(x, bw=2.0):
    """R's density(x, bw = bw)$y (512 values); x as a float64 array in R's order."""
    nfac, m, kt = _factors()
    return _density_y(np.ascontiguousarray(x, np.float64), float(bw), nfac, m, kt)


def _factors():
    global _FAC
    if _FAC is None:
        _FAC = _fft_factor(2 * NG)
    return _FAC


def kde_peaks_count(z, start, count, bw=2.0):
    """kde_peaks_count per cell: z holds each cell's heights (> zmin) in file order, cell c at
    z[start[c]:start[c] + count[c]]. NA where count <= 2 (lidRmetrics needs length(z) > 2)."""
    nfac, m, kt = _factors()
    return _peaks_cells(np.ascontiguousarray(z, np.float64), np.asarray(start, np.int64),
                        np.asarray(count, np.int64), float(bw), nfac, m, kt)
