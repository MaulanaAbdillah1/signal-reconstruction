# -*- coding: utf-8 -*-
"""Report: ringkasan Markdown dan file Word (.docx)."""
import io
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _plain(text):
    """Hapus penanda markdown (**) untuk dokumen Word."""
    return re.sub(r"\*\*(.+?)\*\*", r"\1", text)


def make_plot_png(knots, test, tf, y_lin, y_cs, y_true, t_min, t_max):
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    if y_true is not None:
        ax.plot(tf, y_true, ":", color="#2a9d4b", label="Sinyal sebenarnya")
    ax.plot(tf, y_lin, color="#e07b00", label="Linier")
    ax.plot(tf, y_cs, color="#1f77b4", label="Cubic spline")
    if len(test):
        ax.plot(test["time_ms"], test["amplitude"], "x", color="gray", label="Data uji")
    ax.plot(knots["time_ms"], knots["amplitude"], "o", color="#d62728", label="Knot")
    ax.set_xlim(t_min, t_max)
    ax.set_xlabel("Waktu (ms)")
    ax.set_ylabel("Amplitudo")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _metrics_md(df):
    lines = ["| Metrik | Linier | Cubic Spline |", "|---|---|---|"]
    for k in df.index:
        lines.append(f"| {k} | {df.loc[k, 'Linier']:.4g} | {df.loc[k, 'Cubic Spline']:.4g} |")
    return "\n".join(lines)


def build_markdown(ctx):
    p = ctx["params"]
    md = [
        "# Laporan Rekonstruksi Sinyal: Interpolasi Linier vs Cubic Spline",
        f"_Dibuat: {datetime.now():%d-%m-%Y %H:%M}_",
        "",
        "## 1. Tujuan",
        "Merekonstruksi sinyal dari titik-titik sampel dengan interpolasi linier dan cubic spline, "
        "lalu membandingkan akurasinya.",
        "",
        "## 2. Data",
        f"- Sumber: {ctx['source']}",
        f"- Data awal: {ctx['n_full']} titik",
        f"- Rentang analisis: {ctx['t_min']:g}–{ctx['t_max']:g} ms ({ctx['n_roi']} titik)",
        f"- Knot dipakai: {ctx['n_knots']} titik (setiap k = {ctx['k']}), data uji: {ctx['n_test']} titik",
    ]
    if p:
        md.append(f"- Parameter sinyal: A1={p['A1']}, f1={p['f1']} Hz, A2={p['A2']}, f2={p['f2']} Hz")
    md += [
        "",
        "## 3. Metode",
        "- Linier: y = y0 + (y1 − y0)/(x1 − x0) · (x − x0)",
        f"- Cubic spline (syarat batas: {ctx['bc']}): S(x) = a + b·dx + c·dx² + d·dx³, dx = x − xi",
        "",
        "## 4. Hasil Evaluasi",
        "**Terhadap data uji (held-out):**" if ctx["m_test"] is not None else "",
        _metrics_md(ctx["m_test"]) if ctx["m_test"] is not None else "",
        "",
        "**Terhadap sinyal sebenarnya (grid halus):**" if ctx["m_fine"] is not None else "",
        _metrics_md(ctx["m_fine"]) if ctx["m_fine"] is not None else "",
        "",
        "## 5. Analisa",
    ]
    md += [f"- {t}" for t in ctx["interp"]]
    md += ["", "## 6. Kesimpulan", ctx["conclusion"]]
    return "\n".join(x for x in md if x is not None)


def conclusion_text(ctx):
    rl, rc = ctx["rmse_lin"], ctx["rmse_cs"]
    better = "cubic spline" if rc < rl else "interpolasi linier"
    txt = (
        f"Pada rentang {ctx['t_min']:g}–{ctx['t_max']:g} ms dengan {ctx['n_knots']} knot, "
        f"{better} memberi hasil lebih akurat (RMSE linier {rl:.3g}, cubic spline {rc:.3g}). "
    )
    txt += ("Syarat Nyquist terpenuhi, sehingga sinyal dapat direkonstruksi dengan baik."
            if ctx["nyquist_ok"] else
            "Syarat Nyquist tidak terpenuhi, sehingga hasil rekonstruksi tidak dapat diandalkan.")
    return txt


def build_docx(ctx, png_bytes):
    from docx import Document
    from docx.shared import Inches, Pt

    doc = Document()
    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(12)

    doc.add_heading("Laporan Rekonstruksi Sinyal", level=0)
    doc.add_paragraph("Interpolasi Linier dan Cubic Spline")
    doc.add_paragraph(f"Dibuat: {datetime.now():%d-%m-%Y %H:%M}")

    doc.add_heading("1. Tujuan", level=1)
    doc.add_paragraph("Merekonstruksi sinyal dari titik sampel menggunakan interpolasi linier dan "
                      "cubic spline, lalu membandingkan akurasinya.")

    doc.add_heading("2. Data dan Parameter", level=1)
    rows = [("Sumber data", ctx["source"]), ("Jumlah data awal", f"{ctx['n_full']} titik"),
            ("Rentang analisis", f"{ctx['t_min']:g}–{ctx['t_max']:g} ms ({ctx['n_roi']} titik)"),
            ("Knot (k)", f"{ctx['n_knots']} titik (k = {ctx['k']})"),
            ("Data uji", f"{ctx['n_test']} titik"), ("Syarat batas spline", ctx["bc"])]
    p = ctx["params"]
    if p:
        rows += [("A1 / f1", f"{p['A1']} / {p['f1']} Hz"), ("A2 / f2", f"{p['A2']} / {p['f2']} Hz")]
    t = doc.add_table(rows=1, cols=2)
    t.style = "Table Grid"
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Parameter", "Nilai"
    for a, b in rows:
        c = t.add_row().cells
        c[0].text, c[1].text = a, str(b)

    doc.add_heading("3. Metode", level=1)
    doc.add_paragraph("Linier: y = y0 + (y1 − y0)/(x1 − x0) · (x − x0)")
    doc.add_paragraph("Cubic spline: S(x) = a + b·dx + c·dx² + d·dx³, dengan dx = x − xi")

    doc.add_heading("4. Hasil", level=1)
    doc.add_picture(io.BytesIO(png_bytes), width=Inches(6))
    doc.add_paragraph("Gambar 1. Hasil interpolasi pada rentang analisis")

    for title, dfm in (("Evaluasi terhadap data uji (held-out)", ctx["m_test"]),
                       ("Evaluasi terhadap sinyal sebenarnya", ctx["m_fine"])):
        if dfm is None:
            continue
        doc.add_paragraph(title)
        tb = doc.add_table(rows=1, cols=3)
        tb.style = "Table Grid"
        for i, h in enumerate(("Metrik", "Linier", "Cubic Spline")):
            tb.rows[0].cells[i].text = h
        for k in dfm.index:
            c = tb.add_row().cells
            c[0].text = k
            c[1].text = f"{dfm.loc[k, 'Linier']:.4g}"
            c[2].text = f"{dfm.loc[k, 'Cubic Spline']:.4g}"

    doc.add_heading("5. Analisa", level=1)
    for line in ctx["interp"]:
        doc.add_paragraph(_plain(line), style="List Bullet")

    doc.add_heading("6. Kesimpulan", level=1)
    doc.add_paragraph(ctx["conclusion"])

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
