# -*- coding: utf-8 -*-
"""Data Management: pembuatan, pemuatan, validasi, pemotongan, dan pemilihan knot."""
import io

import numpy as np
import pandas as pd

DEFAULT_PARAMS = {"A1": 1.20, "f1": 50.0, "A2": 0.70, "f2": 120.0}


def true_signal(t_ms, A1, f1, A2, f2):
    """Sinyal analitik tanpa noise (referensi 'kebenaran')."""
    t = np.asarray(t_ms, dtype=float) / 1000.0
    return A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t)


def generate_dataset(A1, f1, A2, f2, fs=1000, n_points=101, noise_std=0.0, seed=42):
    """Dataset sintetis: dua sinyal sinus dijumlahkan (default 101 titik, 0-100 ms)."""
    t = np.arange(n_points) * (1000.0 / fs)
    s1 = A1 * np.sin(2 * np.pi * f1 * t / 1000.0)
    s2 = A2 * np.sin(2 * np.pi * f2 * t / 1000.0)
    amp = s1 + s2
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        amp = amp + rng.normal(0.0, noise_std, size=n_points)
    return pd.DataFrame({"time_ms": t, "signal1": s1, "signal2": s2, "amplitude": amp})


def load_csv(file_obj):
    """Baca CSV upload. Kolom dikenali: time_ms & amplitude (atau 2 kolom pertama)."""
    notes = []
    df = pd.read_csv(file_obj)
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = cols.get("time_ms") or cols.get("time") or cols.get("t")
    acol = cols.get("amplitude") or cols.get("amp") or cols.get("y")
    if tcol is None or acol is None:
        if df.shape[1] < 2:
            raise ValueError("CSV minimal harus punya 2 kolom (waktu dan amplitudo).")
        tcol, acol = df.columns[0], df.columns[1]
        notes.append(f"Kolom tidak dikenali, memakai '{tcol}' sebagai waktu dan '{acol}' sebagai amplitudo.")
    out = pd.DataFrame({
        "time_ms": pd.to_numeric(df[tcol], errors="coerce"),
        "amplitude": pd.to_numeric(df[acol], errors="coerce"),
    })
    return out, notes


def clean(df):
    """Validasi: buang NaN, urutkan, hapus waktu duplikat."""
    notes = []
    n0 = len(df)
    df = df.dropna(subset=["time_ms", "amplitude"])
    if len(df) < n0:
        notes.append(f"{n0 - len(df)} baris NaN dibuang.")
    n1 = len(df)
    df = df.drop_duplicates(subset="time_ms").sort_values("time_ms")
    if len(df) < n1:
        notes.append(f"{n1 - len(df)} baris waktu duplikat dibuang.")
    return df.reset_index(drop=True), notes


def crop_range(df, t_min, t_max):
    """Ambil data pada rentang t_min..t_max (ms), inklusif."""
    m = (df["time_ms"] >= t_min - 1e-9) & (df["time_ms"] <= t_max + 1e-9)
    return df.loc[m].reset_index(drop=True)


def split_knots(df_roi, k):
    """Knot = setiap titik ke-k (titik awal & akhir selalu ikut).
    Sisanya menjadi data uji (held-out) untuk evaluasi."""
    n = len(df_roi)
    idx = list(range(0, n, k))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    knots = df_roi.iloc[idx].reset_index(drop=True)
    test = df_roi.drop(df_roi.index[idx]).reset_index(drop=True)
    return knots, test


def to_csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
