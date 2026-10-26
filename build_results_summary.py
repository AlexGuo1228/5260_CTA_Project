"""Build annotated figures and a three-page English results summary.

Run after run_execution_study.py. Use --charts-only or --pdf-only when plotting
and document dependencies live in separate environments. All figures use the
2000-2014 historical holdout; no policy is selected using those observations.
"""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/execution_study"
FIG = OUT / "figures"
PDF = ROOT / "output/pdf/LECTURE8_RESULTS_SUMMARY.pdf"


def charts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.ticker import PercentFormatter

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold",
                         "axes.labelcolor": "#334155", "text.color": "#17324d"})
    colors = ["#94a3b8", "#167d9a", "#d27036", "#5e609d"]
    fig, ax = plt.subplots(figsize=(8.3, 3.1), constrained_layout=True)
    scenarios = [("zero", "Before costs"), ("base", "Trading + rolling"),
                 ("margin20_rate5", "20% margin + 5% funding"),
                 ("margin20_no_borrow", "20% margin, no borrowing")]
    for (key, label), color in zip(scenarios, colors):
        d = pd.read_csv(OUT / "runs" / f"CTA_v2_{key}.csv", index_col=0, parse_dates=True).loc["2000":"2014"]
        # Include initial wealth so all paths share the same starting level.
        nav = pd.concat([pd.Series([1.], index=[pd.Timestamp("1999-12-31")]),
                         (1 + d.net_return).cumprod()])
        ax.plot(nav.index, nav, label=label, color=color, linewidth=1.8)
    ax.set(ylabel="Growth of 1 unit (excess-return wealth)", title="CTA_v2 | costs and funding change the path")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(alpha=.15)
    fig.savefig(FIG / "net_wealth.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.3, 2.9), constrained_layout=True)
    for key, label, color in [("margin10_rate2", "10% margin requirement", colors[1]),
                              ("margin20_rate5", "20% margin requirement", colors[2])]:
        d = pd.read_csv(OUT / "runs" / f"CTA_v2_{key}.csv", index_col=0, parse_dates=True).loc["2000":"2014"]
        ax.plot(d.index, d.margin_required, label=label, color=color, linewidth=1.4)
    ax.axhline(.8, color="#17324d", linestyle="--", label="Own-cash capacity: 80% NAV")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(ylabel="Initial margin / NAV", title="Borrow only when required margin exceeds available cash")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(alpha=.15)
    fig.savefig(FIG / "margin_capacity.png", dpi=190)
    plt.close(fig)

    d = pd.read_csv(OUT / "cost_sensitivity.csv")
    d = d[(d.period == "holdout") & (d.trade_bps <= 50)]
    fig, ax = plt.subplots(figsize=(8.3, 2.8), constrained_layout=True)
    for name, color in [("TSMOM_12", colors[3]), ("CTA_v2", colors[1])]:
        sub = d[d.strategy == name]
        ax.plot(sub.trade_bps, sub.sharpe, "o-", label=name, color=color, linewidth=1.8)
    ax.set(xlabel="One-way trading cost (bps of traded notional)", ylabel="Net Sharpe",
           title="Cost sensitivity | rolling held at 2 bps; financing excluded")
    ax.axhline(0, color="#94a3b8", linewidth=.7)
    ax.legend(frameon=False)
    ax.grid(alpha=.15)
    fig.savefig(FIG / "cost_sensitivity.png", dpi=190)
    plt.close(fig)


def document():
    import pandas as pd
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                   TableStyle, Image, PageBreak)

    PDF.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyCustom", fontName="Helvetica", fontSize=9.2,
                              leading=12.7, textColor=colors.HexColor("#334155"), spaceAfter=7))
    styles.add(ParagraphStyle(name="NoteCustom", parent=styles["BodyCustom"], fontSize=8,
                              leading=10.5, textColor=colors.HexColor("#64748b")))
    styles.add(ParagraphStyle(name="TitleCustom", fontName="Helvetica-Bold", fontSize=23,
                              leading=27, textColor=colors.HexColor("#17324d"), spaceAfter=9))
    styles.add(ParagraphStyle(name="SectionCustom", parent=styles["TitleCustom"], fontSize=13,
                              leading=17, spaceBefore=8, spaceAfter=7))
    story = []
    def p(text, kind="BodyCustom"):
        story.append(Paragraph(text, styles[kind]))
    def tab(headers, rows, widths):
        cells = [[Paragraph(str(x), styles["NoteCustom"]) for x in headers]]
        cells += [[Paragraph(str(x), styles["BodyCustom"]) for x in row] for row in rows]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e6eef4")),
                               ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f8fa")]),
                               ("VALIGN", (0,0), (-1,-1), "TOP"),
                               ("LEFTPADDING", (0,0), (-1,-1), 7),
                               ("RIGHTPADDING", (0,0), (-1,-1), 7),
                               ("TOPPADDING", (0,0), (-1,-1), 5),
                               ("BOTTOMPADDING", (0,0), (-1,-1), 3)]))
        story.append(t)
        story.append(Spacer(1, 7))
    def chart(name, height):
        story.append(Image(str(FIG / name), width=510, height=height))
        story.append(Spacer(1, 6))
    summary = pd.read_csv(OUT / "baseline_comparison.csv")
    hold = summary[summary.period == "holdout"]
    base = hold[hold.scenario == "base"].set_index("strategy")
    zero = hold[hold.scenario == "zero"].set_index("strategy")
    funding = pd.read_csv(OUT / "funding_comparison.csv")
    funding = funding[funding.period == "holdout"]
    fcta = funding[funding.strategy == "CTA_v2"].set_index("scenario")
    selected = json.loads((OUT / "manifest.json").read_text())["selected_policy"]
    b, f = base.loc["CTA_v2"], fcta.loc["margin20_rate5"]

    p("LECTURE 8 / RESULTS SUMMARY", "NoteCustom")
    p("What survives implementation costs?", "TitleCustom")
    p("ORIE 5260 | Managed futures | 2000-2014 historical holdout (180 months)")
    p(f"<b>CTA_v2:</b> Sharpe {zero.loc['CTA_v2','sharpe']:.3f} before costs, "
      f"{b.sharpe:.3f} after trading and rolling, and {f.sharpe:.3f} under the "
      "20% margin / 5% incremental funding scenario. Funding is charged on the cash shortfall, not gross notional.")
    p("1 / Baseline comparison", "SectionCustom")
    rows = []
    fund20 = funding[funding.scenario == "margin20_rate5"].set_index("strategy")
    for name, row in base.iterrows():
        rows.append([name, f"{zero.loc[name,'sharpe']:.3f}", f"{row.sharpe:.3f}",
                     f"{fund20.loc[name,'sharpe']:.3f}", f"{row.ann_return:.2%}"])
    tab(["Strategy", "Gross SR", "Net SR: T+R", "Net SR: +funding", "T+R net mean / yr"], rows, [140,75,85,105,105])
    p("T+R: 2 bps per unit traded plus 2 bps per combined roll, four rolls/year. "
      "Funding column adds 20% uniform margin, 20% cash reserve and 5% incremental financing. "
      "Annual means are arithmetic annualizations; returns exclude cash income.", "NoteCustom")
    chart("net_wealth.png", 190)
    p("The no-borrow path changes positions to fit cash capacity; its performance is not a pure fee deduction. "
      "All paths start at 1 on 31 December 1999. Lines summarize a previously inspected historical period.", "NoteCustom")
    p("Design: six original strategies reproduced; no signal or risk-model parameter retuning. "
      "Training ends in 1999. Zero-cost full-execution returns and turnover match the legacy engine.")
    story.append(PageBreak())

    p("2 / Margin and financing", "TitleCustom")
    p("Notional exposure is not the borrowed amount. Margin is posted collateral, not a return expense.")
    p("<b>Monthly model</b><br/>Margin / NAV = margin rate x gross notional / NAV<br/>"
      "Borrowing / NAV = max(margin / NAV - 80%, 0)<br/>"
      "Financing cost / NAV = borrowing / NAV x annual incremental rate / 12")
    labels = {"margin10_rate2": "10% margin / 2% rate", "margin20_rate5": "20% margin / 5% rate",
              "margin30_rate8": "30% margin / 8% rate", "margin20_no_borrow": "20% / no borrowing"}
    rows = [[labels[k], f"{r.avg_margin:.1%}", f"{r.avg_borrowing:.1%}",
             f"{int(r.borrowing_months)}/180", f"{r.ann_financing_cost:.2%}", f"{r.sharpe:.3f}"]
            for k,r in fcta.iterrows()]
    tab(["CTA_v2 scenario", "Avg margin / NAV", "Avg debt / NAV", "Months with debt", "Funding / yr", "Net SR"],
        rows, [145,76,76,76,77,60])
    p("All scenarios use a 20% NAV cash reserve and base trading/rolling costs. Rates are hypothetical "
      "incremental borrowing spreads net of collateral/reference cash income, not observed all-in loan quotes.", "NoteCustom")
    chart("margin_capacity.png", 178)
    low, high = fcta.loc["margin10_rate2"], fcta.loc["margin20_rate5"]
    p(f"<b>Reading the chart:</b> at 10% margin, borrowing is needed in {int(low.borrowing_months)} "
      f"of 180 months. At 20%, that rises to {int(high.borrowing_months)} months; peak debt is "
      f"{high.peak_borrowing:.1%} of NAV. Averages alone conceal funding pressure.")
    scaled = fcta.loc["margin20_no_borrow"]
    p(f"<b>No-credit alternative:</b> scale exposure whenever margin would exceed 80% NAV. "
      f"This binds in {int(scaled.scaled_months)} months; realized volatility becomes {scaled.ann_vol:.2%} "
      f"and net annual mean return {scaled.ann_return:.2%}. Recomputed turnover includes the scaling trades.")
    p("Scope: monthly initial-margin proxy; credit assumed available in borrow cases. No intramonth "
      "variation margin, offsets, haircuts, lender limits or forced intraday liquidation. Zero modeled "
      "borrowing does not mean no margin-call risk.", "NoteCustom")
    story.append(PageBreak())

    p("3 / Execution choice and next steps", "TitleCustom")
    policies = pd.read_csv(OUT / "policy_comparison.csv")
    train = policies[policies.period == "train"].set_index("policy")
    test = policies[policies.period == "holdout"].set_index("policy")
    rows = [[name, f"{train.loc[name,'sharpe']:.3f}", f"{r.sharpe:.3f}",
             f"{r.ann_turnover:.2f}x", f"{r.max_drawdown:.2%}"] for name,r in test.iterrows()]
    tab(["Policy", "Train net SR", "Holdout net SR", "Turnover / yr", "Max drawdown"],
        rows, [135,90,95,95,95])
    p(f"Training selected <b>{selected}</b>. Policy results include trading/rolling costs but exclude "
      "financing, isolating the original execution experiment. Holdout improvements are diagnostic, "
      "not a basis for changing the selected rule. Entries, exits and reversals execute immediately.", "NoteCustom")
    chart("cost_sensitivity.png", 172)
    p("Grid points are scenarios, not a calibrated impact model. CTA_v2's Sharpe advantage over TSMOM_12 "
      "disappears between 20 and 50 bps one-way trading cost. Funding is excluded in this comparison.", "NoteCustom")
    p("Priorities for the next iteration", "SectionCustom")
    p("<b>1.</b> Calibrate per-contract margin, liquidity, roll calendars and funding terms.<br/>"
      "<b>2.</b> Add daily variation-margin stress, credit limits and final-position concentration checks.<br/>"
      "<b>3.</b> Validate execution rules with walk-forward selection and new data; then optimize net returns subject to capital constraints.")
    p("Sources: project execution-study CSVs and manifest; Lecture 8, pp. 24, 25 and 32. "
      "The 2000-2014 period was already examined in the midterm. Estimates are not proof of significance "
      "or production readiness. Reproduce: run_execution_study.py, then build_results_summary.py.", "NoteCustom")

    def page_number(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#d7e2eb"))
        canvas.line(42, 36, A4[0]-42, 36)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(42, 23, "ORIE 5260 | Lecture 8 | Research scenarios")
        canvas.drawRightString(A4[0]-42, 23, f"{doc.page} / 3")
    doc = SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=42, leftMargin=42,
                            topMargin=37, bottomMargin=48, title="Lecture 8 Results Summary",
                            author="ORIE 5260 Project Team")
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(PDF)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--charts-only", action="store_true")
    mode.add_argument("--pdf-only", action="store_true")
    args = parser.parse_args()
    if not args.pdf_only:
        charts()
    if not args.charts_only:
        document()
