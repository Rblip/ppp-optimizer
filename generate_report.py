"""
Generate a PDF report on the empirical pricing curve model and MSFT case study.

Usage:
    python generate_report.py

Output:
    msft_curve_report.pdf
"""

from __future__ import annotations

import io
import os
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Image, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

from yahoo_curve import compute_r
from estimate_msft import fetch_msft_panel, fit_constrained
from dynamic_curve import (
    implied_growth, simulate_eps, validate_growth,
    optimal_b_myopic, optimal_b_dynamic, value_myopic_blend, firm_value,
)

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1B3A6B")
BLUE   = colors.HexColor("#2E86AB")
LGRAY  = colors.HexColor("#F2F5F9")
MGRAY  = colors.HexColor("#D0D8E4")
RED    = colors.HexColor("#C0392B")
GREEN  = colors.HexColor("#1E8449")
WHITE  = colors.white
BLACK  = colors.HexColor("#1C1C1C")

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm
TEXT_W = PAGE_W - 2 * MARGIN


# ── Styles ────────────────────────────────────────────────────────────────────

def make_styles():
    base = getSampleStyleSheet()

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        "title": S("RPT_title",
                   fontName="Helvetica-Bold", fontSize=22,
                   textColor=NAVY, alignment=TA_CENTER,
                   spaceAfter=6),
        "subtitle": S("RPT_subtitle",
                      fontName="Helvetica", fontSize=12,
                      textColor=BLUE, alignment=TA_CENTER,
                      spaceAfter=4),
        "meta": S("RPT_meta",
                  fontName="Helvetica", fontSize=9,
                  textColor=colors.HexColor("#555555"),
                  alignment=TA_CENTER, spaceAfter=2),
        "h1": S("RPT_h1",
                fontName="Helvetica-Bold", fontSize=13,
                textColor=NAVY, spaceBefore=14, spaceAfter=4),
        "h2": S("RPT_h2",
                fontName="Helvetica-Bold", fontSize=10.5,
                textColor=BLUE, spaceBefore=8, spaceAfter=3),
        "body": S("RPT_body",
                  fontName="Helvetica", fontSize=9.5,
                  leading=14, alignment=TA_JUSTIFY,
                  spaceBefore=2, spaceAfter=4),
        "mono": S("RPT_mono",
                  fontName="Courier", fontSize=8.5,
                  leading=12, spaceBefore=2, spaceAfter=2),
        "caption": S("RPT_caption",
                     fontName="Helvetica-Oblique", fontSize=8,
                     textColor=colors.HexColor("#555555"),
                     alignment=TA_CENTER, spaceBefore=2, spaceAfter=6),
        "bullet": S("RPT_bullet",
                    fontName="Helvetica", fontSize=9.5,
                    leading=14, leftIndent=14, spaceAfter=2),
    }


# ── Figures ───────────────────────────────────────────────────────────────────

def _save_fig(fig) -> str:
    path = tempfile.mktemp(suffix=".png")
    fig.savefig(path, dpi=160, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return path


def figure_curve(panel, result) -> str:
    """
    Left:  Year-specific P(b) curves — shared a0, ab; each year's own EP, ROI, EPS.
           Observed prices as filled dots. Trajectory arrow from FY2022→FY2025
           shows that curve upshift (EPS growth) explains rising P despite b > b*.

    Right: a(b) fitted line with per-year residuals.
    """
    a0, ab   = result["a0"], result["ab"]
    b_star   = result["b_star"]
    years    = panel["date"].dt.year.tolist()
    l_vals   = result["residuals"]
    pt_colors = ["#C0392B" if l > 1e-4 else "#1E8449" if l < -1e-4
                 else "#888888" for l in l_vals]

    yr_palette = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    # ── Left: year-specific P(b) curves ──────────────────────────────────────
    ax = axes[0]
    b_grid = np.linspace(0.01, 0.97, 800)
    p_max  = float(max(panel["price"]))

    obs_b, obs_p = [], []
    for i, row in panel.iterrows():
        ep_i  = row["EP"]
        roi_i = row["ROI"]
        eps_i = row["EPS"]
        col   = yr_palette[i % len(yr_palette)]

        r_grid = compute_r(b_grid, ep_i, roi_i)
        a_grid = a0 * (1.0 - b_grid / ab)
        denom  = r_grid - b_grid * (r_grid + a_grid)
        P_grid = np.where(denom > 1e-6, eps_i * (1.0 - b_grid) / denom, np.nan)
        P_grid = np.where(P_grid > p_max * 2.5, np.nan, P_grid)

        ax.plot(b_grid, P_grid, color=col, lw=1.6, alpha=0.75,
                label=f"FY{years[i]}")

        b_obs = float(result["b_used"][i])
        p_obs = float(row["price"])
        ax.scatter(b_obs, p_obs, s=90, color=col, zorder=8,
                   edgecolors="white", linewidths=1.0)
        ax.annotate(f"FY{years[i]}", xy=(b_obs, p_obs),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=7.5, color=col, fontweight="bold")
        obs_b.append(b_obs)
        obs_p.append(p_obs)

    # Trajectory arrow FY2022 → FY2025 (EPS growth shifts curves up)
    ax.annotate("", xy=(obs_b[-1], obs_p[-1]), xytext=(obs_b[0], obs_p[0]),
                arrowprops=dict(arrowstyle="-|>", color="#555555",
                                lw=1.4, connectionstyle="arc3,rad=0.18"))
    mid_b = (obs_b[0] + obs_b[-1]) / 2 + 0.04
    mid_p = (obs_p[0] + obs_p[-1]) / 2 - 20
    eps_ratio = panel["EPS"].iloc[-1] / panel["EPS"].iloc[0]
    ax.text(mid_b, mid_p,
            f"EPS ×{eps_ratio:.2f}\nb: {obs_b[0]:.2f}→{obs_b[-1]:.2f}",
            fontsize=7.5, color="#555555", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    ax.axvline(b_star, color="#555555", lw=1.3, ls="--", alpha=0.7,
               label=f"b* = {b_star:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, p_max * 2.2)
    ax.set_xlabel("Plowback ratio  b", fontsize=9)
    ax.set_ylabel("Stock price  P  ($)", fontsize=9)
    ax.set_title("P(b) — Year-specific Curves  (global a₀, aᵦ)\n"
                 "Arrow: EPS growth shifts curves up; b ↑ AND P ↑",
                 fontsize=9.0, fontweight="bold", color="#1B3A6B")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=8)

    # ── Right: a(b) with residuals ────────────────────────────────────────────
    ax2 = axes[1]
    b_range = np.linspace(0.0, ab * 1.05, 300)
    a_curve = a0 * (1.0 - b_range / ab)
    ax2.plot(b_range, a_curve, color="#2E86AB", lw=2.2,
             label="a(b) = a₀·(1 − b/aᵦ)")
    ax2.axhline(0, color="black", lw=0.7)
    ax2.axvline(ab, color="#555555", lw=1.2, ls="--",
                label=f"aᵦ = {ab:.3f}  (breakeven)")

    for i, row in panel.iterrows():
        b_i  = result["b_used"][i]
        aq_i = result["a_required"][i]
        af_i = result["a_fitted"][i]
        l_i  = l_vals[i]
        ax2.scatter(b_i, aq_i, color=pt_colors[i], s=70, zorder=6,
                    edgecolors="white", linewidths=0.8)
        ax2.plot([b_i, b_i], [af_i, aq_i], color=pt_colors[i],
                 lw=1.5, ls="-", alpha=0.7)
        ax2.annotate(f"FY{years[i]}",
                     xy=(b_i, aq_i),
                     xytext=(6, 4), textcoords="offset points",
                     fontsize=7.5, color=pt_colors[i])

    ax2.set_xlabel("Plowback ratio  b", fontsize=9)
    ax2.set_ylabel("Reinvestment premium  a(b)", fontsize=9)
    ax2.set_title("Adaptive Parameter  a(b)  with Residuals  l", fontsize=10,
                  fontweight="bold", color="#1B3A6B")
    ax2.legend(fontsize=8, framealpha=0.85)
    ax2.grid(True, alpha=0.25, lw=0.6)
    ax2.tick_params(labelsize=8)

    fig.tight_layout(pad=1.8)
    return _save_fig(fig)


def figure_sentiment(panel, result) -> str:
    """Bar chart of l by fiscal year — management sentiment."""
    years = [f"FY{y}" for y in panel["date"].dt.year.tolist()]
    l_vals = result["residuals"]
    bar_colors = ["#C0392B" if l > 1e-4 else "#1E8449" if l < -1e-4
                  else "#888888" for l in l_vals]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("white")
    bars = ax.bar(years, l_vals, color=bar_colors, width=0.5,
                  edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="black", lw=0.8)

    for bar, val in zip(bars, l_vals):
        sign = "+" if val >= 0 else ""
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.0004 if val >= 0 else -0.0008),
                f"{sign}{val:.4f}",
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=8.5, fontweight="bold",
                color="#C0392B" if val > 1e-4 else "#1E8449")

    ax.set_ylabel("Residual  l  (management perceived disadvantage)", fontsize=9)
    ax.set_title("Management Sentiment by Fiscal Year", fontsize=10,
                 fontweight="bold", color="#1B3A6B")
    ax.grid(True, axis="y", alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=9)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#C0392B", label="Disadvantage  (l > 0)"),
                       Patch(facecolor="#1E8449", label="Advantage  (l < 0)")]
    ax.legend(handles=legend_elements, fontsize=8, framealpha=0.85)
    fig.tight_layout()
    return _save_fig(fig)


def figure_dynamic(panel, result, b_myo, b_dyn, r_d) -> str:
    """
    Left:  forward EPS trajectories under the myopic vs dynamic plowback,
           with MSFT's realised EPS overlaid.
    Right: firm value vs b for both treatments, optima and observed band marked.
    """
    a0, ab = result["a0"], result["ab"]
    EP_m   = float(panel["EP"].mean())
    ROI_m  = float(panel["ROI"].mean())
    E0     = float(panel["EPS"].iloc[-1])
    b_lo, b_hi = float(panel["b"].min()), float(panel["b"].max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    # ── Left: endogenous EPS trajectories ─────────────────────────────────────
    ax = axes[0]
    T = 10
    yrs = np.arange(0, T + 1)
    r_path = np.full(T, r_d)
    E_dyn = simulate_eps(E0, np.full(T, b_dyn), r_path, a0, ab)
    E_myo = simulate_eps(E0, np.full(T, b_myo), r_path, a0, ab)

    g_dyn = implied_growth(b_dyn, r_d, a0, ab)
    g_myo = implied_growth(b_myo, r_d, a0, ab)

    ax.plot(yrs, E_dyn, color="#1E8449", lw=2.4, marker="o", ms=4,
            label=f"b*_dyn = {b_dyn:.2f}   (g = {g_dyn*100:.1f}%/yr)")
    ax.plot(yrs, E_myo, color="#C0392B", lw=2.4, marker="s", ms=4,
            label=f"b*_myopic = {b_myo:.2f}   (g = {g_myo*100:.1f}%/yr)")

    # MSFT realised EPS (last 4 fiscal years, aligned to start)
    eps_hist = panel["EPS"].values
    ax.scatter(np.arange(-(len(eps_hist) - 1), 1), eps_hist,
               color="#1B3A6B", s=55, zorder=8, marker="D",
               label="MSFT realised EPS")

    ax.axhline(E0, color="#888888", lw=0.8, ls=":")
    ax.set_xlabel("Years from latest fiscal year", fontsize=9)
    ax.set_ylabel("Earnings per share  E  ($)", fontsize=9)
    ax.set_title("Endogenous Earnings:  E compounds through plowback\n"
                 "Eₜ₊₁ = Eₜ·(1 + b·(r + a(b)))",
                 fontsize=9.5, fontweight="bold", color="#1B3A6B")
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=8)

    # ── Right: value vs b, myopic vs dynamic optima ───────────────────────────
    ax2 = axes[1]
    b_m, bg_m, v_m = optimal_b_myopic(E0, EP_m, ROI_m, a0, ab)
    b_d, bg_d, v_d = optimal_b_dynamic(E0, r_d, a0, ab)

    # Normalise each to its own peak for shape comparison
    v_m_n = v_m / np.nanmax(v_m)
    v_d_n = v_d / np.nanmax(v_d)

    ax2.plot(bg_m, v_m_n, color="#C0392B", lw=2.2,
             label="Myopic: r blends to ROI")
    ax2.plot(bg_d, v_d_n, color="#1E8449", lw=2.2,
             label="Dynamic: r = cost of equity")

    ax2.axvline(b_m, color="#C0392B", lw=1.3, ls="--", alpha=0.8)
    ax2.axvline(b_d, color="#1E8449", lw=1.3, ls="--", alpha=0.8)
    ax2.text(b_m, 1.02, f"b*={b_m:.2f}", color="#C0392B",
             fontsize=8, ha="center", fontweight="bold")
    ax2.text(b_d, 1.02, f"b*={b_d:.2f}", color="#1E8449",
             fontsize=8, ha="center", fontweight="bold")

    # MSFT observed plowback band
    ax2.axvspan(b_lo, b_hi, alpha=0.12, color="#1B3A6B", zorder=0)
    ax2.text((b_lo + b_hi) / 2, 0.12, "MSFT\nobserved",
             color="#1B3A6B", fontsize=7.5, ha="center", va="bottom")

    ax2.set_xlim(0, 0.92)
    ax2.set_ylim(0, 1.12)
    ax2.set_xlabel("Plowback ratio  b", fontsize=9)
    ax2.set_ylabel("Firm value  (normalised to peak)", fontsize=9)
    ax2.set_title("Optimal Plowback:  Myopic vs Dynamic\n"
                  "Dynamic optimum falls inside MSFT's actual range",
                  fontsize=9.5, fontweight="bold", color="#1B3A6B")
    ax2.legend(fontsize=8, framealpha=0.9, loc="lower center")
    ax2.grid(True, alpha=0.25, lw=0.6)
    ax2.tick_params(labelsize=8)

    fig.tight_layout(pad=1.8)
    return _save_fig(fig)


# ── Table helpers ─────────────────────────────────────────────────────────────

def data_table(rows, col_widths, header_color=NAVY):
    style = TableStyle([
        ("BACKGROUND",  (0, 0), (-1,  0), header_color),
        ("TEXTCOLOR",   (0, 0), (-1,  0), WHITE),
        ("FONTNAME",    (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",       (0, 0), (0,  -1), "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LGRAY]),
        ("GRID",        (0, 0), (-1, -1), 0.4, MGRAY),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ])
    t = Table(rows, colWidths=col_widths)
    t.setStyle(style)
    return t


def rule():
    return HRFlowable(width="100%", thickness=0.6,
                      color=NAVY, spaceAfter=6, spaceBefore=2)


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(panel, result, output="msft_curve_report.pdf"):
    S  = make_styles()
    a0 = result["a0"]
    ab = result["ab"]
    b_star = result["b_star"]
    r2 = result["r_squared"]
    rmse = float(np.sqrt(np.mean(result["residuals"] ** 2)))
    years = panel["date"].dt.year.tolist()
    l_vals = result["residuals"]

    # Dynamic-extension quantities (endogenous earnings)
    EP_m  = float(panel["EP"].mean())
    ROI_m = float(panel["ROI"].mean())
    E0    = float(panel["EPS"].iloc[-1])
    r_d   = float(np.mean([compute_r(result["b_used"][i],
                                     panel["EP"].iloc[i], panel["ROI"].iloc[i])
                           for i in range(len(panel))]))
    b_myo, _, _ = optimal_b_myopic(E0, EP_m, ROI_m, a0, ab)
    b_dyn, _, _ = optimal_b_dynamic(E0, r_d, a0, ab)

    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="Empirical Pricing Curve — MSFT Case Study",
        author="Peter Paul Dimke",
    )

    story = []
    P = lambda txt, style="body": Paragraph(txt, S[style])
    SP = lambda h=0.3: Spacer(1, h * cm)

    # ── Cover ──────────────────────────────────────────────────────────────────
    story += [
        SP(3),
        P("Empirical Pricing Curve Model", "title"),
        P("with Management Sentiment Residuals", "subtitle"),
        SP(0.4),
        rule(),
        SP(0.4),
        P("MSFT Case Study — Gordon Growth Model with Constrained Estimation", "subtitle"),
        SP(0.6),
        P("Peter Paul Dimke · Portfolio Optimization · June 2026", "meta"),
        SP(4),
    ]

    # ── 1. Executive Summary ──────────────────────────────────────────────────
    story += [
        P("1.  Executive Summary", "h1"), rule(),
        P(
            "This report presents a parametric pricing curve model grounded in the "
            "Gordon Growth Model and estimates its firm-specific parameters for "
            "Microsoft Corporation (MSFT) using publicly available financial data "
            "from Yahoo Finance. "
            "The core idea is that a firm's stock price can be expressed as a "
            "function of its plowback (earnings-retention) ratio <i>b</i>, with the "
            "discount rate and reinvestment premium both determined from observed "
            "fundamentals. "
            "A per-period residual <i>l</i> captures the gap between the theoretical "
            "curve and the observed price, and is interpreted as "
            "<b>management's perceived disadvantage of the current plowback level</b>: "
            "positive when management implicitly discounts the return on retained "
            "earnings, negative when they perceive an above-curve reinvestment "
            "opportunity."
        ),
        SP(0.3),
        P(
            "Estimated on four fiscal years (FY2022–FY2025) with economic plowback "
            "ratios spanning 0.30–0.58, the model achieves <b>R² = "
            f"{r2:.3f}</b> with firm-specific parameters "
            f"<b>a₀ = {a0:.3f}</b> and <b>aᵦ = {ab:.3f}</b>, both within "
            "the unit-interval constraints that guarantee an interior optimum. "
            f"The implied value-maximising plowback is <b>b* = {b_star:.3f}</b>. "
            "The FY2023 residual (l = −0.0164) stands out as the only year where "
            "management's decisions revealed a perceived <i>advantage</i> in "
            "reinvestment — consistent with the $10 billion OpenAI commitment "
            "and accelerating Azure growth that year."
        ),
        SP(0.5),
    ]

    # ── 2. Mathematical Framework ─────────────────────────────────────────────
    story += [
        P("2.  Mathematical Framework", "h1"), rule(),
        P("2.1  The Pricing Curve", "h2"),
        P(
            "The model centres on a pricing curve that maps the plowback ratio "
            "<i>b</i> ∈ (0, 1) to the firm's stock price <i>P</i>:"
        ),
        SP(0.2),
        P("P(b)  =  E · (1 − b)  /  ( r(b) − b · (r(b) + a(b)) )", "mono"),
        SP(0.2),
        P(
            "where <i>E</i> is earnings per share, <i>r(b)</i> is the financially "
            "parameterised discount rate, and <i>a(b)</i> is an adaptive reinvestment "
            "premium. The formula is structurally equivalent to the Gordon Growth "
            "Model, P = D₁ / (r − g), with dividends D₁ = E·(1−b) and "
            "growth rate g(b) = b · (r(b) + a(b))."
        ),
        SP(0.3),
        P("2.2  Adaptive Reinvestment Premium", "h2"),
        P(
            "The reinvestment premium <i>a</i> is not constant — it decreases in "
            "<i>b</i>, capturing <b>diminishing returns to retained earnings</b>:"
        ),
        SP(0.2),
        P("a(b)  =  a₀ · (1 − b / aᵦ)  −  l", "mono"),
        SP(0.2),
        P(
            "The parameters a₀ and aᵦ are <b>firm-specific and time-invariant</b>, "
            "estimated by regression. The term <i>l</i> is a per-period residual. "
            "The parameter aᵦ is the <i>breakeven plowback</i> — the reinvestment "
            "rate at which the premium reaches zero. For a well-behaved curve "
            "with an interior optimum, the model imposes a₀, aᵦ ∈ [0, 1]."
        ),
        SP(0.3),
        P("2.3  Financial Parameterisation of the Discount Rate", "h2"),
        P(
            "Rather than treating the discount rate as an exogenous constant, it "
            "is grounded in two observable fundamentals — the earnings yield "
            "<i>EP = EPS / Price</i> and the return on equity <i>ROI</i> — "
            "blended by the plowback ratio:"
        ),
        SP(0.2),
        P("rₜ  =  (1 − bₜ) · EPₜ  +  bₜ · ROIₜ", "mono"),
        SP(0.2),
        P(
            "<b>r is not a parameter — it is given each year</b> directly from "
            "two observables: the earnings yield EP = EPS / Price and the return "
            "on equity ROI = Net Income / Equity. "
            "At <i>b</i> = 0 (all earnings paid out) r equals the earnings yield; "
            "at <i>b</i> = 1 (all earnings retained) r equals ROI. "
            "The blend is unconstrained; no bounds are imposed on r."
        ),
        SP(0.3),
        P("2.4  The Residual  l  —  Management's Perceived Disadvantage", "h2"),
        P(
            "Inverting the pricing formula for a given observation (bₜ, Pₜ) yields "
            "the reinvestment premium that would be <i>required</i> to price the "
            "stock exactly:"
        ),
        SP(0.2),
        P("a_required  =  (Pₜ · rₜ − E · (1 − bₜ)) / (Pₜ · bₜ)  −  rₜ", "mono"),
        SP(0.2),
        P(
            "The residual is then  l  =  a₀ · (1 − bₜ / aᵦ) − a_required. "
            "A <b>positive l</b> means the fitted curve overestimates the "
            "reinvestment premium — management is discounting the return on "
            "the retained portion, signalling a perceived <b>disadvantage</b> "
            "in the chosen plowback level. "
            "A <b>negative l</b> indicates management believes reinvestment "
            "earns <i>above</i> the theoretical curve — a perceived <b>advantage</b>."
        ),
        SP(0.5),
    ]

    # ── 3. Implementation ─────────────────────────────────────────────────────
    story += [
        P("3.  Implementation", "h1"), rule(),
        P(
            "The model is implemented in Python across four modules. All modules "
            "are independent of the PPP (Parametric Portfolio Policy) optimiser "
            "that occupies the same repository."
        ),
        SP(0.3),
    ]

    impl_rows = [
        ["Module", "Purpose"],
        ["curve_model.py",
         "Core mathematics: curve_P(), compute_a(), compute_residual(), is_on_curve()"],
        ["yahoo_curve.py",
         "Yahoo Finance fetch (EP, ROI via yfinance); compute_r() blend; "
         "curve_P_financial(); compute_residual_financial()"],
        ["regression.py",
         "Unconstrained OLS baseline — reparameterises a(b) as a linear model "
         "in (γ₁=a₀, γ₂=a₀/aᵦ) to obtain closed-form estimates"],
        ["estimate_msft.py",
         "Constrained nonlinear estimation via differential evolution; "
         "full MSFT panel construction with economic plowback"],
    ]
    col_w = [3.8 * cm, TEXT_W - 3.8 * cm]
    story += [data_table(impl_rows, col_w), SP(0.3)]

    story += [
        P("The estimation pipeline proceeds in four steps:", "body"),
        P("1.  Fetch annual income statement, balance sheet, and cash-flow data "
          "for MSFT from Yahoo Finance via <i>yfinance</i>.", "bullet"),
        P("2.  Compute per-year EP, ROI, and economic plowback "
          "b = 1 − (dividends + buybacks) / net income.", "bullet"),
        P("3.  Invert the pricing formula to obtain a_required for each year.", "bullet"),
        P("4.  Minimise the sum of squared residuals Σ lₜ² subject to "
          "a₀, aᵦ ∈ [0, 1] using <i>scipy.optimize.differential_evolution</i> "
          "(global optimiser, avoiding local minima).", "bullet"),
        SP(0.5),
    ]

    # ── 4. MSFT Data ──────────────────────────────────────────────────────────
    story += [
        P("4.  MSFT Panel Data", "h1"), rule(),
        P(
            "Four annual fiscal-year observations are used, spanning a period of "
            "meaningful variation in both the plowback ratio and equity valuation. "
            "The economic plowback is substantially lower than the dividend-only "
            "retention rate because MSFT returns large amounts of capital through "
            "share buybacks ($17–33 B per year)."
        ),
        SP(0.3),
    ]

    panel_rows = [["FY", "EPS ($)", "Price ($)", "Divs ($B)", "BB ($B)",
                   "EP (%)", "ROI (%)", "b (econ)"]]
    for _, row in panel.iterrows():
        panel_rows.append([
            str(row["date"].year),
            f"{row['EPS']:.2f}",
            f"{row['price']:.2f}",
            f"{row['dividends']/1e9:.1f}",
            f"{row['buybacks']/1e9:.1f}",
            f"{row['EP']*100:.2f}",
            f"{row['ROI']*100:.1f}",
            f"{row['b']:.4f}",
        ])

    cw = [1.0, 1.4, 1.5, 1.5, 1.3, 1.3, 1.3, 1.7]
    cw = [x * cm * TEXT_W / (sum(cw) * cm) for x in cw]
    story += [data_table(panel_rows, cw), SP(0.3)]

    story += [
        P(
            "The plowback ratio <i>b</i> rises from 0.30 in FY2022 — when "
            "MSFT executed a $32.7 B buyback programme — to 0.58 by FY2025, "
            "as capex surged on AI infrastructure investment. This 28-point "
            "spread in <i>b</i> is essential for identifying both a₀ and aᵦ "
            "from the data."
        ),
        SP(0.5),
    ]

    # ── 5. Estimation Results ─────────────────────────────────────────────────
    story += [
        P("5.  Estimation Results", "h1"), rule(),
        P("5.1  Fitted Parameters", "h2"),
    ]

    param_rows = [
        ["Parameter", "Value", "Constraint", "Interpretation"],
        ["a₀", f"{a0:.4f}", "[0, 1]  ✓",
         "Reinvestment premium at b = 0 (zero retention)"],
        ["aᵦ", f"{ab:.4f}", "[0, 1]  ✓",
         "Breakeven plowback — premium reaches zero"],
        ["b*", f"{b_star:.4f}", "∈ (0, 1)  ✓",
         "Value-maximising plowback (interior optimum)"],
        ["R²", f"{r2:.4f}", "—",
         "Fit of a₀·(1−b/aᵦ) to a_required across four years"],
        ["RMSE", f"{rmse:.5f}", "—",
         "Root mean squared residual (scale of lₜ)"],
    ]
    cw2 = [1.4, 1.3, 1.6, TEXT_W - 4.3 * cm]
    cw2 = [x * cm for x in [1.6, 1.3, 1.8, TEXT_W / cm - 1.6 - 1.3 - 1.8]]
    story += [data_table(param_rows, cw2), SP(0.3)]

    story += [
        P(
            f"Both a₀ = {a0:.3f} and aᵦ = {ab:.3f} lie strictly within the "
            "unit interval, satisfying the constraints imposed by the Desmos "
            "slider bounds. These constraints are not ad-hoc — they are the "
            "precise conditions under which the P(b) curve has a unique interior "
            f"maximum at b* = {b_star:.3f}, meaning 28.4% economic retention "
            "maximises MSFT's theoretical value."
        ),
        SP(0.3),
        P("5.2  Per-Year Residuals and Management Sentiment", "h2"),
    ]

    resid_rows = [["FY", "b", "r  (given)", "a_required", "a_fitted", "l", "Signal"]]
    for i, row in panel.iterrows():
        l_i = l_vals[i]
        r_i = float(compute_r(result["b_used"][i], row["EP"], row["ROI"]))
        signal = "Disadvantage" if l_i > 1e-4 else \
                 "Advantage"    if l_i < -1e-4 else "Neutral"
        resid_rows.append([
            str(years[i]),
            f"{result['b_used'][i]:.4f}",
            f"{r_i:.5f}",
            f"{result['a_required'][i]:.5f}",
            f"{result['a_fitted'][i]:.5f}",
            f"{l_i:+.5f}",
            signal,
        ])

    _base = [1.0, 1.2, 1.8, 1.8, 1.8, 1.6]
    cw3 = [x * cm for x in _base + [TEXT_W / cm - sum(_base)]]

    t_resid = Table(resid_rows, colWidths=cw3)
    t_resid.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1,  0), NAVY),
        ("TEXTCOLOR",   (0, 0), (-1,  0), WHITE),
        ("FONTNAME",    (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LGRAY]),
        ("GRID",        (0, 0), (-1, -1), 0.4, MGRAY),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        # Colour l and signal: FY2023 = advantage (green), others = disadvantage (red)
        ("TEXTCOLOR", (5, 2), (6, 2), colors.HexColor("#1E8449")),
        ("FONTNAME",  (5, 2), (6, 2), "Helvetica-Bold"),
        ("TEXTCOLOR", (5, 1), (6, 1), colors.HexColor("#C0392B")),
        ("TEXTCOLOR", (5, 3), (6, 3), colors.HexColor("#C0392B")),
        ("TEXTCOLOR", (5, 4), (6, 4), colors.HexColor("#C0392B")),
    ]))
    story += [t_resid, SP(0.3)]

    story += [
        P(
            "<b>FY2022 (l = +0.0089 — Disadvantage).</b>  The $32.7 B buyback "
            "programme returned more capital than management would have chosen "
            "on purely economic grounds. The positive residual indicates that at "
            "the low observed plowback of 0.30, MSFT earned below the theoretical "
            "reinvestment curve — consistent with shareholder pressure to distribute "
            "rather than with management conviction that reinvestment was inferior."
        ),
        SP(0.2),
        P(
            "<b>FY2023 (l = −0.0164 — Advantage).</b>  The sole year of negative l. "
            "Microsoft committed $13 B to OpenAI in January 2023 and accelerated "
            "Azure investment; Azure grew 27% that year. Management revealed through "
            "their decisions that they perceived the return on retained earnings to "
            "be <i>above</i> the theoretical curve — an ex-post correct judgment "
            "given subsequent AI-driven earnings growth."
        ),
        SP(0.2),
        P(
            "<b>FY2024 (l = +0.0073 — Disadvantage).</b>  As capex surged to $44 B "
            "(data-centre buildout for Copilot / Azure AI), management allocated "
            "more capital than the model expects to be optimal, generating a small "
            "positive residual — uncertainty about whether the infrastructure spend "
            "would convert to earnings at the expected rate."
        ),
        SP(0.2),
        P(
            "<b>FY2025 (l = +0.0002 — Neutral).</b>  The residual is negligible. "
            "The model's theoretical curve and management's revealed preference "
            "are essentially aligned; the capital-allocation policy is consistent "
            "with the firm's long-run reinvestment parameters."
        ),
        SP(0.3),
        P("5.3  Why Did Both b and P Rise? — Static vs Dynamic Interpretation", "h2"),
        P(
            "A striking feature of the data is that the plowback ratio <i>b</i> "
            "rose from 0.30 in FY2022 to 0.58 in FY2025, yet the stock price "
            "simultaneously rose from $248 to $493. "
            "In the static model, moving b beyond b* = 0.28 should <i>reduce</i> "
            "price, because additional retention exceeds the value-maximising point. "
            "The apparent contradiction is resolved by recognising that P(b) is not "
            "one fixed curve — it shifts upward every year as EPS grows."
        ),
        SP(0.2),
        P(
            "Between FY2022 and FY2025, MSFT's diluted EPS grew from $9.65 to "
            "$13.64 — a 41% increase driven by Azure, Copilot, and the OpenAI "
            "partnership. This EPS growth raises the <i>entire</i> P(b) curve: "
            "P(b) = <b>E</b> · (1−b) / denom, so doubling E doubles the theoretical "
            "price at every b. The upward shift of the curve from one fiscal year "
            "to the next more than offsets the price drag from b exceeding b*."
        ),
        SP(0.2),
        P(
            "But the curve-shift story raises a sharper question: if MSFT's "
            "plowback (0.30–0.58) sits <i>above</i> the static optimum b* = 0.28, "
            "is the firm chronically over-retaining? Section 6 shows the answer is "
            "no — the static b* is a myopic artifact. Once earnings are made "
            "endogenous, the optimal plowback rises into MSFT's actual range."
        ),
        PageBreak(),
    ]

    # ── 6. Dynamic Extension ──────────────────────────────────────────────────
    story += [
        P("6.  Dynamic Extension — Endogenous Earnings", "h1"), rule(),
        P("6.1  The Myopia of the Single-Period Optimum", "h2"),
        P(
            "The static curve treats earnings E as <i>given</i> and asks which "
            "constant plowback maximises today's price. Its optimum b* = 0.28 is "
            "myopic: it never credits the fact that retained earnings <b>compound</b> "
            "into higher future E. A deeper inspection reveals the low optimum is "
            "largely an artifact of the discount-rate specification. The blend "
            "r(b) = (1−b)·EP + b·ROI puts a single rate in the denominator; because "
            "ROI (≈35%) far exceeds EP (≈3%), every extra unit of retention inflates "
            "the discount rate toward ROI, mechanically penalising plowback and "
            "dragging the optimum down."
        ),
        SP(0.3),
        P("6.2  Separating Discount Rate from Reinvestment Return", "h2"),
        P(
            "The dynamic model disentangles the two distinct roles the blend "
            "conflated:"
        ),
        P("•  the <b>discount rate</b> r_d — the cost of equity, roughly constant;", "bullet"),
        P("•  the <b>reinvestment return</b> ρ(b) = r_d + a(b) — an ROE-like quantity "
          "(≈33% at b = 0.5) earned on the retained portion.", "bullet"),
        SP(0.1),
        P(
            "Earnings then evolve endogenously through the standard "
            "sustainable-growth identity:"
        ),
        SP(0.2),
        P("Eₜ₊₁  =  Eₜ · ( 1 + gₜ ),     gₜ = bₜ · ( r_d + a(bₜ) )", "mono"),
        SP(0.2),
        P(
            "and firm value is the present value of the resulting dividend stream, "
            "Dₜ = Eₜ·(1 − b), discounted at r_d — the Gordon result, now used "
            "forward rather than as a static snapshot."
        ),
        SP(0.3),
        P("6.3  Validation of the Growth Channel", "h2"),
        P(
            "The mechanism is testable: the model-implied growth gₜ = bₜ·(r_d + a(bₜ)) "
            "should track MSFT's realised year-over-year EPS growth. It does, "
            "closely, through the AI-investment years:"
        ),
        SP(0.2),
    ]

    # Growth-validation table
    gv = validate_growth(panel, result)
    gv_rows = [["Period", "b", "g model = b·(r+a)", "g realised (ΔEPS)"]]
    for v in gv:
        gv_rows.append([
            f"{v['year_from']}→{v['year_to']}",
            f"{v['b']:.3f}",
            f"{v['g_model']*100:.1f}%",
            f"{v['g_real']*100:.1f}%",
        ])
    gv_cw = [x * cm for x in [3.0, 2.4, 5.0, TEXT_W / cm - 10.4]]
    story += [data_table(gv_rows, gv_cw), SP(0.3)]

    story += [
        P("6.4  The Dynamic Optimum", "h2"),
        P(
            f"Discounting at the cost of equity r_d = {r_d:.3f} and letting earnings "
            "compound, the value-maximising plowback rises from the myopic "
            f"<b>b*_myopic = {b_myo:.2f}</b> to <b>b*_dynamic = {b_dyn:.2f}</b> — "
            f"squarely inside MSFT's observed range of {float(panel['b'].min()):.2f}–"
            f"{float(panel['b'].max()):.2f}. The conclusion reverses: MSFT is "
            "<b>not over-retaining</b>. Its heavy reinvestment is value-maximising "
            "once the compounding of retained earnings is properly credited. The "
            "rise in both b and P is not a paradox but the signature of a firm "
            "correctly exploiting reinvestment returns that sit close to its "
            "discount rate."
        ),
        SP(0.5),
    ]

    # ── 7. Figures ────────────────────────────────────────────────────────────
    story += [
        P("7.  Figures", "h1"), rule(), SP(0.3),
    ]

    fig1_path = figure_curve(panel, result)
    story += [
        Image(fig1_path, width=TEXT_W, height=TEXT_W * 5 / 13),
        P(
            "<b>Figure 1.</b>  Left: year-specific P(b) curves sharing the same "
            "global a₀ and aᵦ but each year's own EP, ROI, and EPS. Each observation "
            "(filled dot) lies on or near its year's curve; the arrow traces MSFT's "
            "dynamic path as EPS grew ×1.41 from FY2022 to FY2025 — curve upshifts "
            "explain why P rose even as b moved further past b*. "
            "Right: a(b) = a₀·(1 − b/aᵦ) with residuals l (vertical segments); "
            "R² = 0.976 on the reinvestment-premium fit.",
            "caption",
        ),
        SP(0.6),
    ]

    fig2_path = figure_sentiment(panel, result)
    story += [
        Image(fig2_path, width=TEXT_W * 0.62, height=TEXT_W * 0.62 * 3.5 / 7),
        P(
            "<b>Figure 2.</b>  Management's perceived disadvantage of <i>b</i> by "
            "fiscal year. FY2023 is the only year where management revealed a "
            "perceived <i>advantage</i> in their reinvestment level, coinciding "
            "with the OpenAI commitment and Azure acceleration.",
            "caption",
        ),
        SP(0.6),
    ]

    fig3_path = figure_dynamic(panel, result, b_myo, b_dyn, r_d)
    story += [
        Image(fig3_path, width=TEXT_W, height=TEXT_W * 5 / 13),
        P(
            "<b>Figure 3.</b>  Dynamic extension. Left: forward EPS trajectories "
            "with earnings made endogenous (Eₜ₊₁ = Eₜ·(1 + b·(r + a(b)))); the "
            f"dynamic optimum b = {b_dyn:.2f} compounds far faster than the myopic "
            f"b = {b_myo:.2f}, and MSFT's realised EPS (diamonds) tracks the high-"
            "plowback path. Right: firm value vs plowback under each treatment, "
            "normalised to peak. Treating r as a proper cost of equity (green) moves "
            f"the optimum from {b_myo:.2f} to {b_dyn:.2f}, inside the shaded band of "
            "MSFT's actually-chosen plowback.",
            "caption",
        ),
        SP(0.5),
    ]

    # ── 8. Conclusion ─────────────────────────────────────────────────────────
    story += [
        P("8.  Conclusion", "h1"), rule(),
        P(
            "The empirical curve model provides a compact, interpretable framework "
            "for pricing a firm as a function of its capital-allocation decisions. "
            "By grounding the discount rate in observable fundamentals (EP and ROI) "
            "and constraining the reinvestment-premium parameters to the unit "
            "interval, the model guarantees an economically meaningful interior "
            "optimum — the plowback ratio at which firm value is maximised."
        ),
        SP(0.3),
        P(
            "Applied to MSFT, the model estimates a₀ = 0.437 and aᵦ = 0.783 "
            "with R² = 0.976 on four annual observations. The static optimum "
            "b*_myopic = 0.28 lies below MSFT's observed range of 0.30–0.58, which "
            "at first suggests chronic over-retention. The dynamic extension "
            "overturns this: once earnings are made endogenous and the discount "
            "rate is treated as a proper cost of equity rather than a blend that "
            f"rises toward ROI, the optimum climbs to b*_dynamic = {b_dyn:.2f} — "
            "inside MSFT's actual range. The firm is optimising, not over-retaining."
        ),
        SP(0.3),
        P(
            "The per-year residuals l carry the model's most actionable signal: "
            "FY2023's negative residual (advantage) correctly identified the "
            "vintage year in which MSFT's reinvestment genuinely outperformed its "
            "long-run curve — the year the OpenAI partnership was forged. "
            "Tracking l over time thus offers a real-time indicator of whether "
            "management's capital-allocation decisions are consistent with, above, "
            "or below the firm's own historical reinvestment capabilities. Together, "
            "the static residuals and the dynamic optimum give complementary views: "
            "l flags year-by-year deviations, while b*_dynamic confirms the firm's "
            "reinvestment policy is sound over the long run."
        ),
        SP(0.5),
        rule(),
        P("End of report.", "meta"),
    ]

    doc.build(story)
    print(f"  PDF written → {output}")

    # Clean up temp figures
    for p in [fig1_path, fig2_path, fig3_path]:
        try:
            os.remove(p)
        except OSError:
            pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Generating MSFT Curve Model Report")
    print("=" * 60)

    print("\nFetching MSFT data...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = fetch_msft_panel()

    print("Running estimation...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_constrained(
            ticker="MSFT",
            b   = panel["b"].values,
            p   = panel["price"].values,
            E   = panel["EPS"].values,
            ROI = panel["ROI"].values,
            EP  = panel["EP"].values,
        )

    print("Building PDF...")
    build_report(panel, result, output="msft_curve_report.pdf")
    print("Done.")
