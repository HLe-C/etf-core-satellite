"""
Generate final V2.9 reports for the B2 homework track.

Outputs:
- yearly attribution for the final candidate
- final execution rules
- a homework-ready Markdown research report
- a homework-ready HTML research report
- core charts used by the report
"""
from __future__ import annotations

import contextlib
import html
import io
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_v2 import BacktestEngineV2
from family_strategy_research_v2 import (
    CASH_YIELD,
    FAMILY_TARGETS,
    gold_nav,
    metrics_from_nav,
    pct,
    quality_no_gold_params,
    ratio,
)


ROOT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR = ROOT_OUTPUT_DIR / "final"
CHART_DIR = OUTPUT_DIR / "charts"
INTERMEDIATE_OUTPUT_DIR = ROOT_OUTPUT_DIR / "intermediate"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEE_RATE = 0.0001

FINAL_WEIGHTS = {
    "risk": 0.60,
    "gold": 0.20,
    "defensive": 0.20,
}

DEFENSIVE_PRIORITY = [
    ("511360", "短融ETF"),
    ("511880", "银华日利"),
    ("511010", "国债ETF"),
    ("511260", "十年国债ETF"),
]


def run_risk_strategy() -> pd.DataFrame:
    params = replace(quality_no_gold_params(), start_date="2019-10-01", end_date="2025-12-31")
    engine = BacktestEngineV2(params)
    with contextlib.redirect_stdout(io.StringIO()):
        engine.run()
    nav = engine.get_nav_df().copy()
    nav["risk_nav"] = nav["nav"] / nav["nav"].iloc[0]
    nav["benchmark_nav"] = nav["bench_nav"]
    return nav


def cash_proxy_nav(dates: pd.DatetimeIndex, annual_yield: float = CASH_YIELD) -> pd.Series:
    daily = (1 + annual_yield) ** (1 / 252) - 1
    return pd.Series((1 + daily) ** np.arange(len(dates)), index=dates, name="cash_proxy")


def load_defensive_nav(dates: pd.DatetimeIndex) -> tuple[pd.Series, str, str]:
    for symbol, name in DEFENSIVE_PRIORITY:
        files = list(DATA_DIR.glob(f"{symbol}_*.csv"))
        if not files:
            continue
        df = pd.read_csv(files[0], index_col=0, parse_dates=True).sort_index()
        if "close" not in df.columns or df.empty:
            continue
        close = df["close"].reindex(dates).ffill()
        if close.dropna().empty:
            continue
        nav = close / close.dropna().iloc[0]
        nav = nav.ffill().fillna(1.0)
        return nav.rename(symbol), symbol, name
    return cash_proxy_nav(dates), "CASH_PROXY", "2%年化现金代理"


def monthly_first_dates(dates: pd.DatetimeIndex) -> set[pd.Timestamp]:
    return set(pd.Series(dates, index=dates).groupby([dates.year, dates.month]).first().values)


def combine_final_nav(
    risk_nav: pd.Series,
    gold: pd.Series,
    defensive: pd.Series,
) -> tuple[pd.Series, pd.DataFrame]:
    dates = risk_nav.index
    rebal_dates = monthly_first_dates(dates)
    risk_ret = risk_nav.pct_change().fillna(0.0)
    gold_ret = gold.reindex(dates).ffill().pct_change().fillna(0.0)
    defensive_ret = defensive.reindex(dates).ffill().pct_change().fillna(0.0)
    nav = pd.Series(index=dates, dtype=float, name="v2.9_final_nav")
    nav.iloc[0] = 1.0
    rows = []
    target_weights = np.array([FINAL_WEIGHTS["risk"], FINAL_WEIGHTS["gold"], FINAL_WEIGHTS["defensive"]], dtype=float)
    current_weights = target_weights.copy()

    for i, date in enumerate(dates):
        if i == 0:
            rows.append({
                "date": date,
                "risk_contribution": 0.0,
                "gold_contribution": 0.0,
                "defensive_contribution": 0.0,
                "fee_drag": 0.0,
                "portfolio_return": 0.0,
            })
            continue

        turnover = float(np.abs(target_weights - current_weights).sum()) if date in rebal_dates else 0.0
        fee_drag = -turnover * FEE_RATE
        nav_after_fee = nav.iloc[i - 1] * (1 + fee_drag)
        if date in rebal_dates:
            current_weights = target_weights.copy()

        asset_returns = np.array([risk_ret.loc[date], gold_ret.loc[date], defensive_ret.loc[date]], dtype=float)
        contributions = current_weights * asset_returns
        risk_ctr, gold_ctr, defensive_ctr = contributions
        gross_return = float(contributions.sum())
        nav.iloc[i] = nav_after_fee * (1 + gross_return)
        if 1 + gross_return != 0:
            current_weights = current_weights * (1 + asset_returns) / (1 + gross_return)
        rows.append({
            "date": date,
            "risk_contribution": risk_ctr,
            "gold_contribution": gold_ctr,
            "defensive_contribution": defensive_ctr,
            "fee_drag": fee_drag,
            "portfolio_return": nav.iloc[i] / nav.iloc[i - 1] - 1,
        })
    return nav, pd.DataFrame(rows).set_index("date")


def period_metrics(prefix: str, nav: pd.Series, benchmark: pd.Series, start: str, end: str) -> dict:
    s = nav.loc[start:end]
    b = benchmark.loc[s.index]
    metrics = metrics_from_nav(s / s.iloc[0], b / b.iloc[0])
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def yearly_attribution(
    nav: pd.Series,
    benchmark: pd.Series,
    daily_attr: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for year, group in daily_attr.groupby(daily_attr.index.year):
        nav_year = nav.loc[group.index]
        bench_year = benchmark.loc[group.index]
        rows.append({
            "year": int(year),
            "portfolio_return": nav_year.iloc[-1] / nav_year.iloc[0] - 1,
            "benchmark_return": bench_year.iloc[-1] / bench_year.iloc[0] - 1,
            "risk_contribution": group["risk_contribution"].sum(),
            "gold_contribution": group["gold_contribution"].sum(),
            "defensive_contribution": group["defensive_contribution"].sum(),
            "fee_drag": group["fee_drag"].sum(),
            "max_drawdown": (nav_year / nav_year.cummax() - 1).min(),
        })
    return pd.DataFrame(rows)


def fmt_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if col == "year":
                vals.append(str(int(val)))
            elif isinstance(val, (float, np.floating)):
                vals.append(pct(float(val)))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1


def save_core_charts(nav_export: pd.DataFrame, attribution: pd.DataFrame) -> list[Path]:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    chart_paths: list[Path] = []
    colors = {
        "final": "#2563eb",
        "risk": "#10b981",
        "benchmark": "#f97316",
        "gold": "#d4a017",
        "defensive": "#64748b",
        "fee": "#dc2626",
    }

    def pct_label(value: float) -> str:
        return f"{value:.0%}"

    def write_svg(path: Path, content: str):
        path.write_text(content, encoding="utf-8")
        chart_paths.append(path)

    def line_chart(path: Path, title: str, series: dict[str, pd.Series], y_label: str, pct_axis: bool = False):
        width, height = 920, 470
        left, right, top, bottom = 70, 24, 92, 56
        plot_w, plot_h = width - left - right, height - top - bottom
        all_values = pd.concat(series.values()).dropna()
        y_min = float(all_values.min())
        y_max = float(all_values.max())
        pad = (y_max - y_min) * 0.08 or 0.1
        y_min -= pad
        y_max += pad
        dates = next(iter(series.values())).dropna().index
        x0, x1 = dates.min().value, dates.max().value

        def x_pos(ts: pd.Timestamp) -> float:
            return left + (ts.value - x0) / (x1 - x0) * plot_w

        def y_pos(value: float) -> float:
            return top + (y_max - value) / (y_max - y_min) * plot_h

        year_ticks = pd.date_range(dates.min(), dates.max(), freq="YS")
        if len(year_ticks) == 0 or year_ticks[0] != dates.min():
            year_ticks = year_ticks.insert(0, dates.min())
        grid = []
        for tick in np.linspace(y_min, y_max, 5):
            y = y_pos(float(tick))
            label = pct_label(tick) if pct_axis else f"{tick:.2f}"
            grid.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/>')
            grid.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="axis">{label}</text>')
        for tick in year_ticks:
            x = x_pos(pd.Timestamp(tick))
            grid.append(f'<text x="{x:.1f}" y="{height-22}" text-anchor="middle" class="axis">{pd.Timestamp(tick).year}</text>')

        lines = []
        legend = []
        for idx, (label, ser) in enumerate(series.items()):
            clean = ser.dropna()
            points = " ".join(f"{x_pos(ts):.1f},{y_pos(float(val)):.1f}" for ts, val in clean.items())
            color = [colors["final"], colors["risk"], colors["benchmark"]][idx]
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4"/>')
            legend.append(f'<rect x="{left + idx*170}" y="54" width="12" height="12" fill="{color}"/><text x="{left + idx*170 + 18}" y="65" class="legend">{html.escape(label)}</text>')

        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>.title{{font:700 22px -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;fill:#172033}}.axis{{font:12px -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;fill:#64748b}}.legend{{font:13px -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;fill:#334155}}</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="30" class="title">{html.escape(title)}</text>
<text x="18" y="{top + plot_h/2}" transform="rotate(-90 18 {top + plot_h/2})" class="axis">{html.escape(y_label)}</text>
{''.join(legend)}
{''.join(grid)}
<line x1="{left}" x2="{width-right}" y1="{top+plot_h}" y2="{top+plot_h}" stroke="#94a3b8"/>
<line x1="{left}" x2="{left}" y1="{top}" y2="{top+plot_h}" stroke="#94a3b8"/>
{''.join(lines)}
</svg>'''
        write_svg(path, svg)

    def bar_chart(path: Path, title: str, years: list[str], bars: list[tuple[str, np.ndarray, str]], stacked: bool = False):
        width, height = 920, 470
        left, right, top, bottom = 70, 24, 92, 60
        plot_w, plot_h = width - left - right, height - top - bottom
        if stacked:
            totals = np.zeros(len(years))
            for _, vals, _ in bars:
                totals = totals + vals
            y_values = np.concatenate([totals, np.concatenate([vals for _, vals, _ in bars])])
        else:
            y_values = np.concatenate([vals for _, vals, _ in bars])
        y_min = min(0.0, float(y_values.min()))
        y_max = max(0.0, float(y_values.max()))
        pad = (y_max - y_min) * 0.12 or 0.05
        y_min -= pad
        y_max += pad

        def y_pos(value: float) -> float:
            return top + (y_max - value) / (y_max - y_min) * plot_h

        zero = y_pos(0.0)
        group_w = plot_w / len(years)
        grid = []
        for tick in np.linspace(y_min, y_max, 5):
            y = y_pos(float(tick))
            grid.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/>')
            grid.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="axis">{pct_label(tick)}</text>')
        shapes = []
        if stacked:
            bar_w = group_w * 0.50
            for i, year in enumerate(years):
                pos_base = 0.0
                neg_base = 0.0
                x = left + i * group_w + (group_w - bar_w) / 2
                for _, vals, color in bars:
                    val = float(vals[i])
                    if val >= 0:
                        y_top = y_pos(pos_base + val)
                        y_bottom = y_pos(pos_base)
                        pos_base += val
                    else:
                        y_top = y_pos(neg_base)
                        y_bottom = y_pos(neg_base + val)
                        neg_base += val
                    shapes.append(f'<rect x="{x:.1f}" y="{min(y_top,y_bottom):.1f}" width="{bar_w:.1f}" height="{abs(y_bottom-y_top):.1f}" fill="{color}"/>')
                shapes.append(f'<text x="{x + bar_w/2:.1f}" y="{height-24}" text-anchor="middle" class="axis">{year}</text>')
        else:
            bar_w = group_w * 0.30
            for i, year in enumerate(years):
                for j, (_, vals, color) in enumerate(bars):
                    val = float(vals[i])
                    x = left + i * group_w + group_w * 0.20 + j * bar_w
                    y = y_pos(max(val, 0.0))
                    h = abs(y_pos(val) - zero)
                    if val < 0:
                        y = zero
                    shapes.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
                shapes.append(f'<text x="{left + i * group_w + group_w/2:.1f}" y="{height-24}" text-anchor="middle" class="axis">{year}</text>')
        legend = []
        for idx, (label, _, color) in enumerate(bars):
            legend.append(f'<rect x="{left + idx*150}" y="54" width="12" height="12" fill="{color}"/><text x="{left + idx*150 + 18}" y="65" class="legend">{html.escape(label)}</text>')
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>.title{{font:700 22px -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;fill:#172033}}.axis{{font:12px -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;fill:#64748b}}.legend{{font:13px -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;fill:#334155}}</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="30" class="title">{html.escape(title)}</text>
{''.join(legend)}
{''.join(grid)}
<line x1="{left}" x2="{width-right}" y1="{zero:.1f}" y2="{zero:.1f}" stroke="#334155"/>
<line x1="{left}" x2="{left}" y1="{top}" y2="{top+plot_h}" stroke="#94a3b8"/>
{''.join(shapes)}
</svg>'''
        write_svg(path, svg)

    nav = nav_export[["v2.9_final", "risk_strategy", "benchmark"]].dropna()
    line_chart(
        CHART_DIR / "v2.9_nav_comparison.svg",
        "净值曲线对比",
        {"V2.9 最终组合": nav["v2.9_final"], "风险策略": nav["risk_strategy"], "沪深300基准": nav["benchmark"]},
        "归一化净值",
    )
    line_chart(
        CHART_DIR / "v2.9_drawdown.svg",
        "回撤曲线",
        {"V2.9 最终组合": drawdown(nav["v2.9_final"]), "沪深300基准": drawdown(nav["benchmark"])},
        "回撤",
        pct_axis=True,
    )
    years = attribution["year"].astype(str).tolist()
    bar_chart(
        CHART_DIR / "v2.9_yearly_return.svg",
        "年度收益对比",
        years,
        [
            ("V2.9 最终组合", attribution["portfolio_return"].to_numpy(dtype=float), colors["final"]),
            ("沪深300基准", attribution["benchmark_return"].to_numpy(dtype=float), colors["benchmark"]),
        ],
    )
    bar_chart(
        CHART_DIR / "v2.9_yearly_attribution.svg",
        "年度收益归因",
        years,
        [
            ("风险策略贡献", attribution["risk_contribution"].to_numpy(dtype=float), colors["risk"]),
            ("黄金贡献", attribution["gold_contribution"].to_numpy(dtype=float), colors["gold"]),
            ("短债贡献", attribution["defensive_contribution"].to_numpy(dtype=float), colors["defensive"]),
            ("调仓成本", attribution["fee_drag"].to_numpy(dtype=float), colors["fee"]),
        ],
        stacked=True,
    )
    return chart_paths


def markdown_inline_to_html(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def markdown_to_html(markdown: str, title: str) -> str:
    lines = markdown.splitlines()
    body: list[str] = []
    in_ul = False
    in_ol = False
    in_table = False
    table_rows: list[str] = []

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            body.append("</ul>")
            in_ul = False
        if in_ol:
            body.append("</ol>")
            in_ol = False

    def flush_table():
        nonlocal in_table, table_rows
        if not in_table:
            return
        body.append("<table>")
        for idx, row in enumerate(table_rows):
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            if idx == 1 and all(set(cell.replace(":", "")) <= {"-"} for cell in cells):
                continue
            tag = "th" if idx == 0 else "td"
            body.append("<tr>" + "".join(f"<{tag}>{markdown_inline_to_html(cell)}</{tag}>" for cell in cells) + "</tr>")
        body.append("</table>")
        in_table = False
        table_rows = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("|") and line.endswith("|"):
            close_lists()
            in_table = True
            table_rows.append(line)
            continue
        flush_table()
        if not line:
            close_lists()
            continue
        if line.startswith("![](") or line.startswith("!["):
            close_lists()
            alt = re.search(r"!\[(.*?)\]", line)
            src = re.search(r"\((.*?)\)", line)
            if src:
                body.append(f'<figure><img src="{html.escape(src.group(1))}" alt="{html.escape(alt.group(1) if alt else "")}"></figure>')
            continue
        if line.startswith("# "):
            close_lists()
            body.append(f"<h1>{markdown_inline_to_html(line[2:])}</h1>")
        elif line.startswith("## "):
            close_lists()
            body.append(f"<h2>{markdown_inline_to_html(line[3:])}</h2>")
        elif line.startswith("### "):
            close_lists()
            body.append(f"<h3>{markdown_inline_to_html(line[4:])}</h3>")
        elif line.startswith("- "):
            if not in_ul:
                close_lists()
                body.append("<ul>")
                in_ul = True
            body.append(f"<li>{markdown_inline_to_html(line[2:])}</li>")
        elif re.match(r"^\d+\. ", line):
            if not in_ol:
                close_lists()
                body.append("<ol>")
                in_ol = True
            item_text = re.sub(r"^\d+\. ", "", line)
            body.append(f"<li>{markdown_inline_to_html(item_text)}</li>")
        else:
            close_lists()
            body.append(f"<p>{markdown_inline_to_html(line)}</p>")
    flush_table()
    close_lists()

    css = """
    :root { color-scheme: light; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans CJK SC", "PingFang SC", sans-serif;
      color: #172033;
      background: #f7f8fb;
      line-height: 1.65;
    }
    main {
      max-width: 1080px;
      margin: 0 auto;
      padding: 44px 28px 72px;
      background: #ffffff;
      min-height: 100vh;
      box-shadow: 0 0 36px rgba(18, 30, 52, 0.08);
    }
    h1 { font-size: 34px; line-height: 1.25; margin: 0 0 24px; }
    h2 { font-size: 24px; margin: 38px 0 14px; border-top: 1px solid #e6e9ef; padding-top: 26px; }
    h3 { font-size: 18px; margin: 24px 0 10px; }
    p, li { font-size: 15px; }
    code { background: #eef2f7; padding: 1px 5px; border-radius: 4px; }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0 24px;
      font-size: 14px;
    }
    th, td { border: 1px solid #d8dde8; padding: 8px 10px; text-align: right; }
    th:first-child, td:first-child, td:nth-child(2) { text-align: left; }
    th { background: #eef2f7; font-weight: 700; }
    figure { margin: 22px 0 30px; }
    img { max-width: 100%; display: block; border: 1px solid #d8dde8; }
    """
    return (
        "<!doctype html>\n"
        '<html lang="zh-CN">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{css}</style>\n"
        "</head>\n<body><main>\n"
        + "\n".join(body)
        + "\n</main></body>\n</html>\n"
    )


def maybe_load_v27_practical() -> dict | None:
    path = INTERMEDIATE_OUTPUT_DIR / "v2.7_family_strategy_research.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    rows = df[df["variant"] == "no_gold_sweep_static_r060_g030_c010"]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def maybe_load_stress_summary() -> pd.DataFrame | None:
    path = INTERMEDIATE_OUTPUT_DIR / "v2.8_family_strategy_stress_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    keep = [
        "v2.7_practical_60_30_10",
        "v2.9_gold_cap20_60_20_20",
        "v2.6_original_70_15_15",
        "v2.7_gold_heavy_55_45_0",
    ]
    df = df[df["variant"].isin(keep)].copy()
    order = {variant: i for i, variant in enumerate(keep)}
    df["order"] = df["variant"].map(order)
    return df.sort_values("order")


def stress_table_lines(stress: pd.DataFrame | None) -> list[str]:
    if stress is None or stress.empty:
        return ["暂无 V2.8 压力测试汇总文件；可运行 `python scripts/family_strategy_stress_v2.py` 生成。"]
    names = {
        "v2.7_practical_60_30_10": "V2.7 实用候选 60/30/10",
        "v2.9_gold_cap20_60_20_20": "V2.9 黄金上限 60/20/20",
        "v2.6_original_70_15_15": "V2.6 原候选 70/15/15",
        "v2.7_gold_heavy_55_45_0": "V2.7 高黄金 55/45/0",
    }
    lines = [
        "| 候选 | 压力网格最低达标数 | 4项全达标比例 | 最差年化 | 最差回撤 | 最差夏普 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in stress.iterrows():
        lines.append(
            "| "
            + " | ".join([
                names.get(row["variant"], row["variant"]),
                str(int(row["min_family_score"])),
                f"{float(row['pass_4of4_rate']):.2%}",
                pct(float(row["worst_ann_return"])),
                pct(float(row["worst_max_drawdown"])),
                ratio(float(row["worst_sharpe"])),
            ])
            + " |"
        )
    return lines


def write_execution_rules(defensive_symbol: str, defensive_name: str):
    if defensive_symbol == "CASH_PROXY":
        defensive_note = "- 当前防守仓仍为 2% 年化现金代理；正式执行前必须替换为真实可交易货币/短债 ETF。"
    else:
        defensive_note = f"- 当前防守仓已使用真实 ETF 数据：`{defensive_symbol}` {defensive_name}。"
    lines = [
        "# V2.9 最终候选执行规则",
        "",
        "## 组合框架",
        "",
        "- 每月第一个交易日检查并调仓。",
        "- 60% 配置到风险策略。",
        "- 20% 配置到黄金 ETF：`518880`。",
        f"- 20% 配置到现金/短债仓：`{defensive_symbol}` {defensive_name}。",
        defensive_note,
        "",
        "## 风险策略内部规则",
        "",
        "- 核心池：`510300` 沪深300、`510500` 中证500、`512890` 红利低波、`513100` 纳指。",
        "- 不把黄金放入风险策略内部，避免与外层黄金防守仓重复。",
        "- 卫星池：原行业/主题 ETF 池中剔除核心标的和黄金。",
        "- 月度选择 2 只卫星：收盘价高于 MA200，20 日动量为正，按 60 日动量排名取前 2。",
        "- 市场状态仍用沪深300 MA50/MA200 判断，风险策略内部使用原有仓位和止损/熔断规则。",
        "",
        "## 风控约束",
        "",
        "- 黄金仓上限固定为 20%。",
        "- 现金/短债仓不低于 20%。",
        "- 所有信号只使用调仓日及之前的数据，避免未来函数。",
        "- 交易成本按单边万分之一计入。",
    ]
    (OUTPUT_DIR / "v2.9_final_execution_rules.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    metrics: dict,
    attribution: pd.DataFrame,
    defensive_symbol: str,
    defensive_name: str,
    chart_paths: list[Path],
):
    if defensive_symbol == "CASH_PROXY":
        defensive_reflection = "- 现金/短债仓当前仍为代理口径，正式执行前必须替换为真实货币/短债 ETF。"
    else:
        defensive_reflection = f"- 现金/短债仓已替换为真实 ETF `{defensive_symbol}` {defensive_name}；后续若用于实盘，应继续检查流动性、费率、折溢价和申赎约束。"
    full = {k.replace("full_", ""): v for k, v in metrics.items() if k.startswith("full_")}
    oos = {k.replace("oos_2025_", ""): v for k, v in metrics.items() if k.startswith("oos_2025_")}
    v27 = maybe_load_v27_practical()
    stress = maybe_load_stress_summary()
    chart_captions = {
        "v2.9_nav_comparison.svg": "图 1：V2.9 最终组合、风险策略与沪深300基准净值对比",
        "v2.9_drawdown.svg": "图 2：V2.9 最终组合与沪深300基准回撤对比",
        "v2.9_yearly_return.svg": "图 3：V2.9 最终组合与沪深300基准年度收益对比",
        "v2.9_yearly_attribution.svg": "图 4：V2.9 年度收益归因",
    }
    chart_lines = []
    for path in chart_paths:
        rel = path.relative_to(OUTPUT_DIR)
        chart_lines.extend([f"![{chart_captions.get(path.name, path.stem)}]({rel.as_posix()})", ""])
    lines = [
        "# 面向普通家庭的 ETF 核心-卫星组合构建与回测",
        "",
        "## 摘要",
        "",
        "本文构建一个面向普通家庭的 ETF 核心-卫星资产配置策略。研究目标不是寻找单一年份收益最高的组合，而是在普通家庭可承受的回撤、低频调仓和可解释资产配置约束下，提高组合的风险调整后收益。最终候选 V2.9 采用 60% 风险策略、20% 黄金 ETF、20% 短债 ETF 的结构，其中风险策略负责收益来源，黄金和短债负责降低组合深回撤。",
        "",
        "在 2019-2024 年开发样本中，V2.9 年化收益为 "
        f"{pct(full['ann_return'])}，最大回撤为 {pct(full['max_drawdown'])}，夏普比率为 {ratio(full['sharpe'])}；"
        "在未参与参数选择的 2025 年样本外区间中，策略年化收益为 "
        f"{pct(oos['ann_return'])}，最大回撤为 {pct(oos['max_drawdown'])}，夏普比率为 {ratio(oos['sharpe'])}。"
        "样本外结果说明策略在 2025 年仍保持正收益和正超额，但由于 OOS 只有一年，且 2025 年同时包含权益与黄金偏强的市场环境，本文将其作为初步外推证据，而不是长期有效性的最终证明。正式解读时，2025 OOS 只能回答“开发期结束后是否立即失效”，不能回答“未来多轮牛熊是否持续有效”。",
        "",
        "## 1. 研究问题与设计",
        "",
        "本文对应 B2 题目，核心问题是：能否构建一个普通家庭可以理解、可以执行、并且在风险收益上优于单一宽基持有的 ETF 组合策略。与单纯追求最高收益的择时模型不同，本文把最大回撤、调仓频率、标的可交易性和资产配置解释性作为同等重要的约束。",
        "",
        "研究设计分为三步：第一，构建核心-卫星风险策略，用趋势和动量信号选择风险资产；第二，在风险策略外层加入黄金和短债防守仓，降低纯权益轮动的路径波动；第三，用样本外检验和压力测试检查最终方案是否过度依赖某一类资产或某一年市场环境。",
        "",
        "### 1.1 文献与案例启发",
        "",
        "本文没有直接复现某一篇复杂模型论文，而是采用投资实践中常见、可解释性较强的三类思想：第一，核心-卫星配置框架，用宽基和低相关资产承担组合底仓，再用主题 ETF 捕捉阶段性弹性；第二，时间序列动量和趋势过滤思想，用 MA200、20 日动量和 60 日动量减少弱势资产暴露；第三，风险预算思想，把普通家庭难以承受的权益轮动波动，通过黄金和短债防守仓进行外层缓冲。课程要求强调完整投资研究流程，因此本文重点放在数据、规则、回测、解释、样本外和风险反思的闭环，而不是追求黑箱预测模型复杂度。",
        "",
        "## 2. 数据与样本划分",
        "",
        "- 数据频率：ETF 日线收盘价。",
        "- 风险资产：A 股宽基、行业主题 ETF、纳指 ETF 等。",
        f"- 防守资产：黄金 ETF `518880` 与 `{defensive_symbol}` {defensive_name}。",
        "- 基准：沪深300指数/沪深300 ETF 口径。",
        "- 开发样本：2019-2024，用于策略迭代、参数筛选和版本选择。",
        "- 样本外区间：2025，仅用于最终候选的外推检验，不参与参数选择；本次数据口径覆盖至 2025-12-31。",
        "",
        "为避免未来函数，所有信号只使用调仓日及之前可获得的数据。策略按月度检查和调仓，交易成本按单边万分之一计入。由于部分 ETF 成立时间较晚，实证起点由可用数据共同区间决定。",
        "",
        "## 3. 策略方法",
        "",
        "### 3.1 外层家庭组合",
        "",
        "- 风险策略：60%",
        "- 黄金 ETF：20%，标的 `518880`",
        f"- 现金/短债仓：20%，当前数据口径 `{defensive_symbol}` {defensive_name}",
        "",
        "这个外层结构的作用是把风险预算分清楚：风险策略承担主要收益波动，黄金提供危机和通胀情景下的防守弹性，短债仓提供低波动缓冲。最终版本将黄金上限固定为 20%，是为了避免模型过度吃到 2019-2025 年黄金强势样本的红利。",
        "",
        "### 3.2 风险策略内部规则",
        "",
        "- 核心池：`510300` 沪深300、`510500` 中证500、`512890` 红利低波、`513100` 纳指。",
        "- 卫星池：行业/主题 ETF 池中剔除核心标的和黄金。",
        "- 市场状态：用沪深300 MA50/MA200 判断 bull、range、bear。",
        "- 卫星选择：收盘价高于 MA200，20 日动量为正，再按 60 日动量排名取前 2。",
        "- 风控规则：MA20 止损、沪深300单日大跌熔断、ATR 止盈和仓位偏离再平衡。",
        "",
        "## 4. 回测设定与评价指标",
        "",
        "本文主要评价指标包括年化收益、最大回撤、夏普比率、Calmar 比率、相对沪深300的超额年化收益，以及年度收益归因。对于普通家庭策略，最大回撤和 Calmar 比率尤其重要，因为策略即使长期收益较高，如果中途回撤过深，也很难被真实家庭账户长期坚持。",
        "",
        "本文将 2019-2024 作为开发样本，2025 作为样本外区间。需要强调的是，2025 OOS 并不是重新调参后的结果，而是把开发样本中确定的 V2.9 权重和规则直接外推到 2025 年；本次报告的数据截止日为 2025-12-31。",
        "",
        "### 4.1 交易成本与真实执行约束",
        "",
        f"回测按单边交易成本 {FEE_RATE:.2%} 计入，风险策略内部交易由回测引擎处理，外层 60/20/20 家庭组合按月初再平衡后的权重漂移计算换手成本。由于 ETF 实盘还会受到冲击成本、买卖价差、流动性、折溢价、申赎限制、停牌/临停、调仓日成交可得性和个人账户费率影响，本文把交易成本结果视为保守但不完整的近似。若未来用于真实资金，应额外做不同滑点、买卖价差和资金规模下的容量测试，并优先检查短融 ETF 与行业主题 ETF 的成交额是否足以承载月度调仓。",
        "",
        "## 5. 核心图表",
        "",
        "下列图表对应最终报告中的核心结论：净值曲线用于观察长期累计收益，回撤曲线用于检查普通家庭是否可能坚持持有，年度收益和归因图用于判断收益是否集中在单一年份或单一资产。",
        "",
    ]
    lines.extend(chart_lines)
    lines.extend([
        "## 6. 样本内结果",
        "",
        "2019-2024 年，V2.9 在收益和回撤之间取得了较好的平衡：",
        "",
        "| 区间 | 年化 | 最大回撤 | 夏普 | Calmar | 基准年化 | 超额年化 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 2019-2024 | {pct(full['ann_return'])} | {pct(full['max_drawdown'])} | {ratio(full['sharpe'])} | {ratio(full['calmar'])} | {pct(full['ann_bench'])} | {pct(full['excess'])} |",
        "",
        "相对 V2.3a，V2.9 的主要提升不是简单提高风险仓位，而是通过外层黄金和短债仓降低路径波动。V2.3a 的最大回撤为 -15.21%，V2.9 降至 "
        f"{pct(full['max_drawdown'])}；V2.3a 的夏普比率为 0.158，V2.9 提升至 {ratio(full['sharpe'])}。",
        "",
        "## 7. 样本外检验：2025 OOS",
        "",
        "2025 年作为样本外区间，没有参与 V2.9 的参数选择。该检验的目的不是证明策略必然长期有效，而是检查策略在开发样本之后是否立即失效。",
        "",
        "| 区间 | 年化 | 最大回撤 | 夏普 | Calmar | 基准年化 | 超额年化 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 2025 OOS | {pct(oos['ann_return'])} | {pct(oos['max_drawdown'])} | {ratio(oos['sharpe'])} | {ratio(oos['calmar'])} | {pct(oos['ann_bench'])} | {pct(oos['excess'])} |",
        "",
        "OOS 结果显示，V2.9 在 2025 年实现正收益，并略微跑赢沪深300基准。更重要的是，最大回撤仍控制在 "
        f"{pct(oos['max_drawdown'])}，说明外层防守仓没有在样本外阶段失去控回撤作用。但 2025 年只有一年，且黄金在该阶段仍有较强贡献，因此本文将 OOS 结果解释为“通过初步外推检验”，而不是把它当作长期稳健性的充分证据。更严格地说，2025 OOS 只能说明策略没有在开发样本后的下一年立刻失效；若要证明长期稳健性，还需要未来继续滚动记录，或在更长历史、更丰富市场状态和更多防守资产替代口径下复核。",
        "",
        "## 8. 对照组与版本选择",
        "",
        "为了避免只展示最终版本，本文保留了从 V2.3a 到 V2.9 的迭代证据。关键对照如下：",
        "",
        "| 版本 | 配置/变化 | 年化收益 | 最大回撤 | 夏普 | 选择判断 |",
        "|---|---|---:|---:|---:|---|",
        "| V2.3a | 原核心-卫星风险策略 | +5.05% | -15.21% | 0.158 | 作为风险策略基准，但回撤偏高 |",
        "| V2.4-rc1 | 熊市恢复与熔断冷却微调 | +5.53% | -14.33% | 0.216 | 有改善，但幅度不足以解决家庭持有问题 |",
        "| V2.7 60/30/10 | 60%风险 + 30%黄金 + 10%现金代理 | "
        + (f"{pct(float(v27['full_ann_return']))} | {pct(float(v27['full_max_drawdown']))} | {ratio(float(v27['full_sharpe']))}" if v27 else "+8.25% | -8.26% | 0.613")
        + " | 指标更高，但黄金仓位偏重 |",
        f"| V2.9 60/20/20 | 60%风险 + 20%黄金 + 20%短债ETF | {pct(full['ann_return'])} | {pct(full['max_drawdown'])} | {ratio(full['sharpe'])} | 最终主线，收益、回撤和解释性更均衡 |",
        "",
        "V2.7 的 30% 黄金候选在回测中表现更强，但它更依赖黄金资产在样本期的强势表现。考虑到普通家庭策略不应把胜负过度押在单一资产上，最终选择 V2.9：黄金只保留 20% 上限，同时把 20% 配置到真实短融 ETF。",
        "",
        "## 9. 稳健性分析",
        "",
        "本文针对黄金依赖和防守仓收益假设做了压力测试。压力测试将黄金日收益按 100%、75%、50%、25%、0% 缩放，并把现金/短债收益设为 0%、1%、2%，观察不同候选在压力网格中的表现。",
        "",
    ])
    lines.extend(stress_table_lines(stress))
    lines.extend([
        "",
        "压力测试的达标口径沿用家庭策略硬标准：年化收益不低于 6.5%、最大回撤不深于 -12%、夏普比率不低于 0.4、Calmar 不低于 0.6。",
        "",
        "压力测试的主要结论是：高黄金仓位组合在原始样本中指标更好，但当黄金收益被打折后，收益目标会明显依赖黄金贡献。V2.9 的 20% 黄金上限降低了这种单一资产依赖，虽然牺牲了一部分高黄金版本的年化收益，但更适合作为普通家庭长期配置框架。",
        "",
        "## 10. 年度归因",
        "",
        "年度归因用于观察收益来源是否集中在少数年份或单一资产。表中风险策略、黄金和短债贡献相加后，再扣除调仓成本，形成组合年度收益。注意：本节表格中的 `portfolio_return` 和 `benchmark_return` 是对应自然年的实际总收益，不是前文 2025 OOS 表中的年化收益，因此数值会略有差异。",
        "",
        "",
    ])
    table = attribution.copy()
    lines.extend(fmt_table(table, [
        "year",
        "portfolio_return",
        "benchmark_return",
        "risk_contribution",
        "gold_contribution",
        "defensive_contribution",
        "fee_drag",
        "max_drawdown",
    ]))
    lines.extend([
        "",
        "从年度结果看，V2.9 在 2022 年权益市场下跌时仍保持小幅正收益，说明黄金和短债防守仓在弱市中有实际贡献；2024 年和 2025 年黄金贡献较高，也提示本文不能忽视黄金强势样本带来的正向影响。",
        "",
        "## 11. 题目要求问题回应",
        "",
        "1. 投资对象：A 股 ETF、跨境 ETF、黄金 ETF 和短债 ETF，主要面向普通家庭账户。",
        "2. 组合目标：在低频可执行的前提下，提高相对沪深300的风险调整后收益，并控制最大回撤。",
        "3. 基准：沪深300指数/沪深300 ETF 口径，同时在研究过程中保留风险策略和早期版本作为内部对照。",
        "4. 未来函数：所有趋势、动量和调仓信号只使用调仓日及之前的数据；2025 年仅用于最终候选的样本外检验。",
        f"5. 交易成本：回测计入单边 {FEE_RATE:.2%} 交易成本；真实投资还需额外考虑滑点、申赎限制、折溢价、流动性和个人费率。",
        f"6. 样本内外一致性：2019-2024 样本内年化 {pct(full['ann_return'])}、最大回撤 {pct(full['max_drawdown'])}；2025 OOS 年化 {pct(oos['ann_return'])}、最大回撤 {pct(oos['max_drawdown'])}。方向上保持正收益和控回撤，但 OOS 时间较短。",
        "7. 最赚钱和最差阶段：2025 年贡献最高，2021 年出现组合最大回撤；2022 年权益市场下跌时组合仍保持小幅正收益，是防守仓发挥作用的阶段。",
        "8. 回测好看时最该怀疑：黄金在 2019-2025 样本中偏强、2025 OOS 只有一年、ETF 历史长度有限、部分 ETF 成立时间较晚，以及真实交易成本和成交约束可能高于回测假设。",
        "9. 真实投入最大风险：黄金和短债的防守效果可能阶段性失效，权益反弹时策略可能跟不上，短融/行业 ETF 的成交额和买卖价差可能影响实际成交，且家庭投资者可能在回撤或相对落后阶段提前放弃。",
        "10. 下一步改进：扩大样本外区间，引入更多防守资产和成本情景，比较多个基准，进一步做容量、滑点和真实账户可执行性测试。",
        "",
        "## 12. 局限性",
        "",
        "- 黄金在 2019-2025 样本中表现较强，因此最终版本设置 20% 黄金上限，避免策略过度依赖黄金。",
        defensive_reflection,
        "- 2025 OOS 只有一年，能够提供初步外推证据，但不能替代更长周期、多市场状态的样本外检验；报告中的 OOS 结论应限定为“开发期后一年未立即失效”。",
        "- 回测使用日线收盘价和固定交易成本，真实执行还会受到流动性、冲击成本、买卖价差、申赎限制、折溢价、停牌/临停、调仓日成交可得性和个人税费影响。",
        "- 策略以月度调仓为核心，适合低频家庭账户；若用于更高频或更大资金规模，需要重新评估交易容量。",
        "",
        "## 13. 结论",
        "",
        "本文最终选择 V2.9 作为大作业主线策略。该策略在 2019-2024 年开发样本中实现 "
        f"{pct(full['ann_return'])} 年化收益、{pct(full['max_drawdown'])} 最大回撤和 {ratio(full['sharpe'])} 夏普比率；"
        "在 2025 年样本外区间中继续保持正收益和正超额。相较早期纯风险策略，V2.9 的优势在于把收益来源和防守资产分层，让策略更接近普通家庭可以理解、可以执行、也更容易长期持有的配置方案。",
        "",
        "本文结论不是“V2.9 已经证明未来一定有效”，而是：在当前数据和约束下，V2.9 是收益、回撤、解释性和可执行性之间最均衡的候选版本。后续研究应继续扩大样本外区间，并用真实交易约束检验该策略的可执行性。",
        "",
        "## 14. 参考文献与资料说明",
        "",
        "- 课程大作业题目：《人工智能与投资研究》大作业选题说明，B2 ETF 核心-卫星组合方向。",
        "- 数据获取：AKShare 开源数据接口；本项目使用 ETF 日线价格、沪深300基准数据和货币/短债/国债 ETF 数据。",
        "- 策略思想：核心-卫星资产配置、时间序列动量、移动均线趋势过滤、最大回撤控制和风险预算等投资研究常见方法。",
        "- 开源工具：Python、pandas、numpy；最终 HTML 与 SVG 图表由本项目脚本生成，代码和结果表均保存在仓库中，便于复现。",
        "",
        "## 15. AI 使用说明",
        "",
        "本项目使用 OpenClaw 总控，并结合 Codex 与 Claude Code 辅助完成代码调试、回测脚本编写、策略报告整理、HTML 生成、SVG 图表生成、风险压力测试设计和文字结构化表达。策略判断、题目适配和最终取舍由研究者审阅确认。",
    ])
    markdown = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "v2.9_final_homework_report.md").write_text(markdown, encoding="utf-8")
    html_report = markdown_to_html(markdown, "面向普通家庭的 ETF 核心-卫星组合构建与回测")
    (OUTPUT_DIR / "v2.9_final_homework_report.html").write_text(html_report, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    base = run_risk_strategy()
    dates = pd.DatetimeIndex(base.index)
    risk_nav = base["risk_nav"]
    benchmark = base["benchmark_nav"]
    gold = gold_nav(dates)
    defensive, defensive_symbol, defensive_name = load_defensive_nav(dates)
    final_nav, daily_attr = combine_final_nav(risk_nav, gold, defensive)

    metrics = {
        **period_metrics("full", final_nav, benchmark, "2019-10-01", "2024-12-31"),
        **period_metrics("oos_2025", final_nav, benchmark, "2025-01-01", "2025-12-31"),
    }
    attribution = yearly_attribution(final_nav, benchmark, daily_attr)

    nav_export = pd.DataFrame({
        "v2.9_final": final_nav,
        "risk_strategy": risk_nav,
        "gold": gold.reindex(dates).ffill(),
        "defensive": defensive.reindex(dates).ffill(),
        "benchmark": benchmark / benchmark.iloc[0],
    })
    chart_paths = save_core_charts(nav_export, attribution)
    nav_export.to_csv(OUTPUT_DIR / "v2.9_final_nav.csv")
    daily_attr.to_csv(OUTPUT_DIR / "v2.9_final_daily_attribution.csv")
    attribution.to_csv(OUTPUT_DIR / "v2.9_final_yearly_attribution.csv", index=False)
    pd.DataFrame([metrics | {"defensive_symbol": defensive_symbol, "defensive_name": defensive_name}]).to_csv(
        OUTPUT_DIR / "v2.9_final_metrics.csv",
        index=False,
    )
    write_execution_rules(defensive_symbol, defensive_name)
    write_report(metrics, attribution, defensive_symbol, defensive_name, chart_paths)

    print("V2.9 final reports generated.")
    print(f"HTML report: {OUTPUT_DIR / 'v2.9_final_homework_report.html'}")
    print(f"Defensive sleeve: {defensive_symbol} {defensive_name}")
    print(
        f"2019-2024 ann={metrics['full_ann_return']:.2%}, "
        f"mdd={metrics['full_max_drawdown']:.2%}, sharpe={metrics['full_sharpe']:.3f}"
    )


if __name__ == "__main__":
    main()
