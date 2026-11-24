# -*- coding: utf-8 -*-
"""Evaluation & Engineering Interpretation."""
import numpy as np
import pandas as pd


def metrics(y_ref, y_hat):
    y_ref = np.asarray(y_ref, dtype=float)
    e = np.asarray(y_hat, dtype=float) - y_ref
    mse = float(np.mean(e ** 2))
    ss_tot = float(np.sum((y_ref - y_ref.mean()) ** 2))
    r2 = 1.0 - float(np.sum(e ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(e))),
        "Error maks": float(np.max(np.abs(e))),
        "R²": r2,
    }


def metrics_table(y_ref, y_lin, y_cs):
    ml, mc = metrics(y_ref, y_lin), metrics(y_ref, y_cs)
    df = pd.DataFrame({"Linier": ml, "Cubic Spline": mc})
    df["Perbaikan spline (x)"] = [
        (ml[k] / mc[k]) if k != "R²" and mc[k] > 0 else np.nan for k in df.index
    ]
    return df


def linear_error_bound(h_ms, A1, f1, A2, f2):
    """Batas error interpolasi linier: h^2/8 * max|f''|."""
    h = h_ms / 1000.0
    m2 = A1 * (2 * np.pi * f1) ** 2 + A2 * (2 * np.pi * f2) ** 2
    return h ** 2 / 8.0 * m2


def interpret(ctx):
    """Interpretasi teknik (bahasa sederhana). ctx berisi h_ms, f_max, rmse_lin, rmse_cs, bound, dll."""
    h_ms, f_max = ctx["h_ms"], ctx["f_max"]
    fs_eff = 1000.0 / h_ms
    nyq = 2 * f_max
    spp = fs_eff / f_max
    out = []

    out.append(
        f"**Laju sampling efektif.** Jarak antar knot {h_ms:g} ms, jadi laju sampling efektif "
        f"≈ {fs_eff:.0f} Hz. Frekuensi tertinggi sinyal {f_max:g} Hz, sehingga batas Nyquist "
        f"adalah {nyq:g} Hz. Ada sekitar {spp:.1f} sampel per periode pada frekuensi tertinggi."
    )
    if fs_eff > nyq:
        out.append("✅ **Nyquist terpenuhi.** Laju sampling efektif di atas 2×f_max, sehingga sinyal masih bisa direkonstruksi.")
    else:
        out.append(
            "⚠️ **Nyquist tidak terpenuhi (aliasing).** Laju sampling efektif di bawah 2×f_max. "
            "Informasi frekuensi tinggi hilang dan tidak bisa dipulihkan oleh metode interpolasi apa pun. "
            "Kecilkan nilai k."
        )

    rl, rc = ctx["rmse_lin"], ctx["rmse_cs"]
    if rc > 0 and rl > 0:
        if rc < rl:
            out.append(
                f"**Akurasi.** RMSE cubic spline ({rc:.3g}) lebih kecil dari linier ({rl:.3g}), "
                f"sekitar {rl / rc:.1f}× lebih akurat. Spline mengikuti lengkung gelombang, "
                "sedangkan linier hanya menarik garis lurus antar titik."
            )
        else:
            out.append(
                f"**Akurasi.** RMSE linier ({rl:.3g}) tidak lebih besar dari cubic spline ({rc:.3g}). "
                "Biasanya terjadi karena data sangat rapat atau mengandung noise."
            )

    if ctx.get("bound") is not None:
        out.append(
            f"**Batas teori linier.** Error maksimum interpolasi linier ≤ h²/8·max|f''| ≈ {ctx['bound']:.3g}. "
            f"Error terukur ({ctx['max_lin']:.3g}) berada "
            f"{'di bawah' if ctx['max_lin'] <= ctx['bound'] else 'di atas'} batas ini."
        )

    out.append(
        "**Orde error.** Error linier turun sebanding h² (orde 2), sedangkan cubic spline sebanding h⁴ "
        "(orde 4). Jika jarak knot dibagi dua, error linier turun ±4×, error spline turun ±16×."
    )
    out.append(
        "**Catatan praktis.** Interpolasi linier sederhana dan cepat, cocok untuk pratinjau. "
        "Cubic spline lebih halus (turunan pertama dan kedua kontinu), cocok untuk rekonstruksi "
        "sinyal yang akan diturunkan atau dianalisis lebih lanjut. Jika data berisik, spline bisa "
        "ikut meniru noise (overshoot)."
    )
    return out
