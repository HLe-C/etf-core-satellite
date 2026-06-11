"""
Generate final V2.9 reports for the B2 homework track.

Outputs:
- yearly attribution for the final candidate
- final execution rules
- a homework-ready Markdown research report skeleton
"""
from __future__ import annotations

import contextlib
import io
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


OUTPUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent / "data"
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
    old_weights = (1.0, 0.0, 0.0)
    new_weights = (FINAL_WEIGHTS["risk"], FINAL_WEIGHTS["gold"], FINAL_WEIGHTS["defensive"])

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
        turnover = sum(abs(new_weights[j] - old_weights[j]) for j in range(3)) if date in rebal_dates else 0.0
        old_weights = new_weights
        risk_ctr = FINAL_WEIGHTS["risk"] * risk_ret.loc[date]
        gold_ctr = FINAL_WEIGHTS["gold"] * gold_ret.loc[date]
        defensive_ctr = FINAL_WEIGHTS["defensive"] * defensive_ret.loc[date]
        fee_drag = -turnover * FEE_RATE
        daily_return = risk_ctr + gold_ctr + defensive_ctr + fee_drag
        nav.iloc[i] = nav.iloc[i - 1] * (1 + daily_return)
        rows.append({
            "date": date,
            "risk_contribution": risk_ctr,
            "gold_contribution": gold_ctr,
            "defensive_contribution": defensive_ctr,
            "fee_drag": fee_drag,
            "portfolio_return": daily_return,
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
):
    if defensive_symbol == "CASH_PROXY":
        defensive_reflection = "- 现金/短债仓当前仍为代理口径，正式执行前必须替换为真实货币/短债 ETF。"
    else:
        defensive_reflection = f"- 现金/短债仓已替换为真实 ETF `{defensive_symbol}` {defensive_name}；后续若用于实盘，应继续检查流动性、费率、折溢价和申赎约束。"
    full = {k.replace("full_", ""): v for k, v in metrics.items() if k.startswith("full_")}
    oos = {k.replace("oos_2025_", ""): v for k, v in metrics.items() if k.startswith("oos_2025_")}
    lines = [
        "# 面向普通家庭的 ETF 核心-卫星组合构建与回测",
        "",
        "## 摘要",
        "",
        "本文构建一个面向普通家庭的 ETF 核心-卫星资产配置策略。最终候选 V2.9 采用 60% 风险策略、20% 黄金 ETF、20% 现金/短债仓，目标是在保持中等收益的同时显著降低最大回撤。",
        "",
        "## 最终候选 V2.9",
        "",
        "- 风险策略：60%",
        "- 黄金 ETF：20%，标的 `518880`",
        f"- 现金/短债仓：20%，当前数据口径 `{defensive_symbol}` {defensive_name}",
        "",
        "## 核心结果",
        "",
        "| 区间 | 年化 | 最大回撤 | 夏普 | Calmar | 基准年化 | 超额年化 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 2019-2024 | {pct(full['ann_return'])} | {pct(full['max_drawdown'])} | {ratio(full['sharpe'])} | {ratio(full['calmar'])} | {pct(full['ann_bench'])} | {pct(full['excess'])} |",
        f"| 2025 OOS | {pct(oos['ann_return'])} | {pct(oos['max_drawdown'])} | {ratio(oos['sharpe'])} | {ratio(oos['calmar'])} | {pct(oos['ann_bench'])} | {pct(oos['excess'])} |",
        "",
        "## 年度归因",
        "",
    ]
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
        "## 风险反思",
        "",
        "- 黄金在 2019-2025 样本中表现较强，因此最终版本设置 20% 黄金上限，避免策略过度依赖黄金。",
        defensive_reflection,
        "- 该策略适合课程研究和普通家庭配置框架展示，不构成投资建议。",
        "",
        "## AI 使用说明",
        "",
        "本项目使用 Codex 辅助完成代码调试、回测脚本编写、策略报告整理、风险压力测试设计和文字结构化表达。策略判断、题目适配和最终取舍由研究者审阅确认。",
    ])
    (OUTPUT_DIR / "v2.9_final_homework_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        "benchmark": benchmark,
    })
    nav_export.to_csv(OUTPUT_DIR / "v2.9_final_nav.csv")
    daily_attr.to_csv(OUTPUT_DIR / "v2.9_final_daily_attribution.csv")
    attribution.to_csv(OUTPUT_DIR / "v2.9_final_yearly_attribution.csv", index=False)
    pd.DataFrame([metrics | {"defensive_symbol": defensive_symbol, "defensive_name": defensive_name}]).to_csv(
        OUTPUT_DIR / "v2.9_final_metrics.csv",
        index=False,
    )
    write_execution_rules(defensive_symbol, defensive_name)
    write_report(metrics, attribution, defensive_symbol, defensive_name)

    print("V2.9 final reports generated.")
    print(f"Defensive sleeve: {defensive_symbol} {defensive_name}")
    print(
        f"2019-2024 ann={metrics['full_ann_return']:.2%}, "
        f"mdd={metrics['full_max_drawdown']:.2%}, sharpe={metrics['full_sharpe']:.3f}"
    )


if __name__ == "__main__":
    main()
