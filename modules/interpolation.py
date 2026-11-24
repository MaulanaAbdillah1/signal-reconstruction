# -*- coding: utf-8 -*-
"""Numerical Computation: interpolasi linier dan cubic spline."""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def linear(x, y, xq):
    return np.interp(xq, x, y)


def cubic(x, y, bc="not-a-knot"):
    return CubicSpline(x, y, bc_type=bc)


def segment_index(x, xq):
    i = np.searchsorted(x, xq, side="right") - 1
    return int(np.clip(i, 0, len(x) - 2))


def linear_steps(x, y, xq):
    """Langkah hitung manual interpolasi linier di satu titik."""
    i = segment_index(x, xq)
    x0, x1, y0, y1 = x[i], x[i + 1], y[i], y[i + 1]
    slope = (y1 - y0) / (x1 - x0)
    yq = y0 + slope * (xq - x0)
    return {"i": i, "x0": x0, "x1": x1, "y0": y0, "y1": y1, "slope": slope, "yq": yq}


def spline_table(cs, x):
    """Koefisien tiap segmen: S(x) = a + b*dx + c*dx^2 + d*dx^3, dx = x - x_i."""
    c = cs.c
    return pd.DataFrame({
        "segmen": np.arange(len(x) - 1),
        "x_i (ms)": x[:-1],
        "x_i+1 (ms)": x[1:],
        "a": c[3],
        "b": c[2],
        "c": c[1],
        "d": c[0],
    })


def spline_steps(cs, x, xq):
    i = segment_index(x, xq)
    dx = xq - x[i]
    a, b, c, d = cs.c[3, i], cs.c[2, i], cs.c[1, i], cs.c[0, i]
    yq = a + b * dx + c * dx ** 2 + d * dx ** 3
    return {"i": i, "xi": x[i], "dx": dx, "a": a, "b": b, "c": c, "d": d, "yq": yq}
