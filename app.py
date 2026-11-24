# -*- coding: utf-8 -*-
"""Aplikasi web Rekonstruksi Sinyal: Interpolasi Linier vs Cubic Spline.
Jalankan:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

from modules import data_manager as dm
from modules import evaluation as ev
from modules import interpolation as ip
from modules import report as rp
from modules import visualization as vz

st.set_page_config(page_title="Rekonstruksi Sinyal", page_icon="📈", layout="wide")
st.title("📈 Rekonstruksi Sinyal: Interpolasi Linier vs Cubic Spline")

# ───────────────────────── Sidebar ─────────────────────────
sb = st.sidebar
sb.header("Pengaturan")
source = sb.radio("Sumber data", ["Generate (101 titik)", "Upload CSV"])

params = None
notes = []
if source == "Generate (101 titik)":
    sb.caption("Sinyal = A1·sin(2πf1t) + A2·sin(2πf2t), fs = 1000 Hz, 0–100 ms")
    A1 = sb.number_input("A1", value=dm.DEFAULT_PARAMS["A1"], step=0.1)
    f1 = sb.number_input("f1 (Hz)", value=dm.DEFAULT_PARAMS["f1"], step=5.0)
    A2 = sb.number_input("A2", value=dm.DEFAULT_PARAMS["A2"], step=0.1)
    f2 = sb.number_input("f2 (Hz)", value=dm.DEFAULT_PARAMS["f2"], step=5.0)
    noise = sb.slider("Noise (std)", 0.0, 0.5, 0.0, 0.01)
    params = {"A1": A1, "f1": f1, "A2": A2, "f2": f2}
    df_full = dm.generate_dataset(A1, f1, A2, f2, noise_std=noise)
    f_max = max(f1, f2)
    src_label = "Dataset sintetis (101 titik)"
else:
    up = sb.file_uploader("CSV (kolom: time_ms, amplitude)", type=["csv"])
    if up is None:
        st.info("Upload CSV di sidebar untuk memakai data sendiri.")
        st.stop()
    try:
        raw, n1 = dm.load_csv(up)
        df_full, n2 = dm.clean(raw)
        notes = n1 + n2
    except Exception as e:  # noqa: BLE001
        st.error(f"Gagal membaca CSV: {e}")
        st.stop()
    f_max = sb.number_input("Estimasi frekuensi maksimum sinyal (Hz)", value=120.0, step=5.0)
    src_label = f"CSV upload ({up.name}, {len(df_full)} titik)"

if len(df_full) < 5:
    st.error("Data terlalu sedikit (minimal 5 titik).")
    st.stop()

tmin_all, tmax_all = float(df_full["time_ms"].min()), float(df_full["time_ms"].max())
d_t0, d_t1 = max(tmin_all, 40.0), min(tmax_all, 60.0)
if d_t1 <= d_t0:
    d_t0, d_t1 = tmin_all, tmax_all
t_min, t_max = sb.slider("Rentang analisis (ms)", tmin_all, tmax_all, (d_t0, d_t1), step=1.0)
k = sb.select_slider("Langkah knot k (pakai setiap titik ke-k)", options=[1, 2, 3, 4, 5, 6, 8, 10], value=2)
bc = sb.selectbox("Syarat batas spline", ["not-a-knot", "natural", "clamped"])
step = sb.select_slider("Resolusi grid halus (ms)", options=[0.05, 0.1, 0.2, 0.5], value=0.1)

# ───────────────────────── Pipeline ─────────────────────────
df_roi = dm.crop_range(df_full, t_min, t_max)
if len(df_roi) < 5:
    st.error("Rentang terlalu sempit: minimal 5 titik data di dalam rentang.")
    st.stop()
knots, test = dm.split_knots(df_roi, k)
if len(knots) < 4:
    st.error(f"Knot hanya {len(knots)} titik. Kecilkan k atau perlebar rentang (minimal 4 knot).")
    st.stop()

x, y = knots["time_ms"].to_numpy(), knots["amplitude"].to_numpy()
tf = np.linspace(x[0], x[-1], int(round((x[-1] - x[0]) / step)) + 1)
cs = ip.cubic(x, y, bc)
y_lin, y_cs = ip.linear(x, y, tf), cs(tf)
y_true = dm.true_signal(tf, **params) if params else None

m_test = None
if len(test):
    xt, yt = test["time_ms"].to_numpy(), test["amplitude"].to_numpy()
    m_test = ev.metrics_table(yt, ip.linear(x, y, xt), cs(xt))
m_fine = ev.metrics_table(y_true, y_lin, y_cs) if y_true is not None else None

h_ms = float(np.mean(np.diff(x)))
ref = m_fine if m_fine is not None else m_test
ctx_i = {
    "h_ms": h_ms, "f_max": f_max,
    "rmse_lin": ref.loc["RMSE", "Linier"] if ref is not None else float("nan"),
    "rmse_cs": ref.loc["RMSE", "Cubic Spline"] if ref is not None else float("nan"),
    "bound": ev.linear_error_bound(h_ms, **params) if params else None,
    "max_lin": ref.loc["Error maks", "Linier"] if ref is not None else float("nan"),
}

# ───────────────────────── Tabs ─────────────────────────
tabs = st.tabs(["Overview", "Data Management", "Numerical Computation", "Visualization",
                "Evaluation", "Engineering Interpretation", "Report"])

# 1. Overview
with tabs[0]:
    st.subheader("Overview")
    st.markdown(
        """
Aplikasi ini merekonstruksi sinyal dari titik sampel memakai dua metode interpolasi dan
membandingkan akurasinya.

**Alur kerja**
1. **Data Management**: buat/unggah data (default 101 titik, 0–100 ms), lalu potong ke rentang 40–60 ms.
2. **Numerical Computation**: hitung interpolasi linier dan cubic spline (dengan langkah hitung manual).
3. **Visualization**: bandingkan kurva, knot, dan error.
4. **Evaluation**: MSE, RMSE, MAE, error maksimum, R².
5. **Engineering Interpretation**: Nyquist, orde error, dan rekomendasi.
6. **Report**: unduh laporan (Markdown / Word).

**Metode**
- *Linier*: garis lurus antar dua knot. `y = y0 + (y1−y0)/(x1−x0)·(x−x0)`
- *Cubic spline*: polinomial orde 3 per segmen, turunan pertama & kedua kontinu. `S(x) = a + b·dx + c·dx² + d·dx³`

**Skema evaluasi**: hanya setiap titik ke-*k* dipakai sebagai knot. Titik sisanya menjadi data uji
(*held-out*), dan untuk data sintetis hasil juga dibandingkan dengan sinyal analitik pada grid halus.
"""
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data awal", f"{len(df_full)} titik")
    c2.metric("Rentang analisis", f"{t_min:g}–{t_max:g} ms")
    c3.metric("Titik dalam rentang", f"{len(df_roi)}")
    c4.metric("Knot / uji", f"{len(knots)} / {len(test)}")

# 2. Data Management
with tabs[1]:
    st.subheader("Data Management")
    for n in notes:
        st.warning(n)
    st.markdown(f"**Sumber:** {src_label}")
    a, b = st.columns(2)
    with a:
        st.markdown(f"**Data awal ({len(df_full)} titik)**")
        st.dataframe(df_full, height=320, width="stretch")
        st.download_button("⬇️ Unduh data awal (CSV)", dm.to_csv_bytes(df_full), "signal_full.csv", "text/csv")
    with b:
        st.markdown(f"**Data rentang {t_min:g}–{t_max:g} ms ({len(df_roi)} titik)**")
        st.dataframe(df_roi, height=320, width="stretch")
        st.download_button("⬇️ Unduh data rentang (CSV)", dm.to_csv_bytes(df_roi), "signal_roi.csv", "text/csv")
    st.markdown("**Statistik ringkas (rentang analisis)**")
    st.dataframe(df_roi["amplitude"].describe().to_frame().T, width="stretch")
    a, b = st.columns(2)
    a.markdown(f"**Knot ({len(knots)} titik)**")
    a.dataframe(knots, width="stretch")
    b.markdown(f"**Data uji ({len(test)} titik)**")
    b.dataframe(test, width="stretch")

# 3. Numerical Computation
with tabs[2]:
    st.subheader("Numerical Computation")
    default_q = float(min(x[-1], (x[0] + x[-1]) / 2 + 0.5))
    xq = st.number_input("Titik waktu yang dihitung (ms)", min_value=float(x[0]), max_value=float(x[-1]),
                         value=default_q, step=0.1)
    L = ip.linear_steps(x, y, xq)
    S = ip.spline_steps(cs, x, xq)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Interpolasi Linier")
        st.markdown(f"Segmen [{L['x0']:g}, {L['x1']:g}] ms")
        st.latex(r"y=y_0+\frac{y_1-y_0}{x_1-x_0}(x-x_0)")
        st.latex(
            rf"y={L['y0']:.5f}+\frac{{{L['y1']:.5f}-({L['y0']:.5f})}}{{{L['x1']:g}-{L['x0']:g}}}"
            rf"({xq:g}-{L['x0']:g})={L['yq']:.5f}"
        )
        st.metric("y linier", f"{L['yq']:.6f}")
    with c2:
        st.markdown(f"#### Cubic Spline ({bc})")
        st.markdown(f"Segmen [{x[S['i']]:g}, {x[S['i'] + 1]:g}] ms, dx = {S['dx']:g}")
        st.latex(r"S(x)=a+b\,dx+c\,dx^2+d\,dx^3")
        st.latex(
            rf"S={S['a']:.5f}+({S['b']:.5f})dx+({S['c']:.5f})dx^2+({S['d']:.5f})dx^3={S['yq']:.5f}"
        )
        st.metric("y spline", f"{S['yq']:.6f}")
    if params:
        yt_q = float(dm.true_signal(xq, **params))
        st.markdown(
            f"Nilai sebenarnya: **{yt_q:.6f}** → error linier **{L['yq'] - yt_q:+.2e}**, "
            f"error spline **{S['yq'] - yt_q:+.2e}**"
        )
    st.markdown("#### Koefisien cubic spline per segmen")
    st.dataframe(ip.spline_table(cs, x), width="stretch")
    st.markdown("#### Hasil interpolasi pada grid halus")
    res = pd.DataFrame({"time_ms": tf, "linier": y_lin, "cubic_spline": y_cs})
    if y_true is not None:
        res["sebenarnya"] = y_true
        res["err_linier"] = y_lin - y_true
        res["err_spline"] = y_cs - y_true
    st.dataframe(res, height=300, width="stretch")
    st.download_button("⬇️ Unduh hasil interpolasi (CSV)", dm.to_csv_bytes(res), "hasil_interpolasi.csv", "text/csv")

# 4. Visualization
with tabs[3]:
    st.subheader("Visualization")
    show_full = st.checkbox("Tampilkan seluruh 101 titik (konteks)", value=False)
    fig = vz.main_fig(df_full, knots, test, tf, y_lin, y_cs, y_true, t_min, t_max, show_full)
    st.plotly_chart(fig, width="stretch")
    if y_true is not None:
        st.markdown("**Error terhadap sinyal sebenarnya**")
        st.plotly_chart(vz.error_fig(tf, y_lin - y_true, y_cs - y_true), width="stretch")
    else:
        st.caption("Plot error terhadap sinyal sebenarnya hanya tersedia untuk data sintetis.")

# 5. Evaluation
with tabs[4]:
    st.subheader("Evaluation")
    if m_test is not None:
        st.markdown(f"**A. Terhadap data uji / held-out ({len(test)} titik)**")
        st.dataframe(m_test.style.format("{:.4g}"), width="stretch")
    else:
        st.info("k = 1: semua titik jadi knot, tidak ada data uji. Naikkan k untuk evaluasi held-out.")
    if m_fine is not None:
        st.markdown(f"**B. Terhadap sinyal sebenarnya (grid {step:g} ms, {len(tf)} titik)**")
        st.dataframe(m_fine.style.format("{:.4g}"), width="stretch")
    ref_tbl = m_fine if m_fine is not None else m_test
    if ref_tbl is not None:
        st.plotly_chart(vz.metric_bar(ref_tbl), width="stretch")
        winner = "Cubic spline" if ref_tbl.loc["RMSE", "Cubic Spline"] < ref_tbl.loc["RMSE", "Linier"] else "Linier"
        st.success(f"Metode dengan RMSE terkecil: **{winner}**")

# 6. Engineering Interpretation
with tabs[5]:
    st.subheader("Engineering Interpretation")
    if ref_tbl is None:
        st.info("Naikkan k (≥2) atau gunakan data sintetis agar ada data pembanding.")
    else:
        for line in ev.interpret(ctx_i):
            st.markdown(f"- {line}")

# 7. Report
with tabs[6]:
    st.subheader("Report")
    if ref_tbl is None:
        st.info("Naikkan k (≥2) atau gunakan data sintetis untuk membuat laporan.")
    else:
        rctx = {
            "params": params, "source": src_label, "n_full": len(df_full), "n_roi": len(df_roi),
            "n_knots": len(knots), "n_test": len(test), "k": k, "bc": bc,
            "t_min": t_min, "t_max": t_max, "m_test": m_test, "m_fine": m_fine,
            "interp": ev.interpret(ctx_i), "rmse_lin": ctx_i["rmse_lin"], "rmse_cs": ctx_i["rmse_cs"],
            "nyquist_ok": (1000.0 / h_ms) > 2 * f_max,
        }
        rctx["conclusion"] = rp.conclusion_text(rctx)
        md = rp.build_markdown(rctx)
        st.markdown(md)
        png = rp.make_plot_png(knots, test, tf, y_lin, y_cs, y_true, t_min, t_max)
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Unduh laporan (Markdown)", md.encode("utf-8"), "laporan.md", "text/markdown")
        try:
            docx_bytes = rp.build_docx(rctx, png)
            c2.download_button("⬇️ Unduh laporan (Word)", docx_bytes, "laporan.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        except ImportError:
            c2.warning("Install python-docx untuk unduhan Word.")
