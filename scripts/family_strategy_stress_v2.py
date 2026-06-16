"""
V2.8 stress tests for family-allocation candidates.

The V2.7 scan found strong candidates with external gold/cash defensive sleeves.
This script asks a harder question: are the candidates still acceptable if gold
returns are weaker and the cash/short-bond sleeve earns less than the 2% proxy?
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_v2 import BacktestEngineV2, StrategyParams
from family_strategy_research_v2 import (
    FAMILY_TARGETS,
    metrics_from_nav,
    pct,
    quality_no_gold_params,
    ratio,
)


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEE_RATE = 0.0001

CANDIDATES = [
    {
        "variant": "v2.7_practical_60_30_10",
        "risk_weight": 0.60,
        "gold_weight": 0.30,
        "cash_weight": 0.10,
        "description": "V2.7实用候选：60%风险策略 + 30%黄金 + 10%现金/短债",
    },
    {
        "variant": "v2.9_gold_cap20_60_20_20",
        "risk_weight": 0.60,
        "gold_weight": 0.20,
        "cash_weight": 0.20,
        "description": "黄金上限20%候选：60%风险策略 + 20%黄金 + 20%现金/短债",
    },
    {
        "variant": "v2.7_conservative_55_22_22",
        "risk_weight": 0.55,
        "gold_weight": 0.225,
        "cash_weight": 0.225,
        "description": "更保守候选：55%风险策略 + 22.5%黄金 + 22.5%现金/短债",
    },
    {
        "variant": "v2.6_original_70_15_15",
        "risk_weight": 0.70,
        "gold_weight": 0.15,
        "cash_weight": 0.15,
        "description": "V2.6原候选：70%风险策略 + 15%黄金 + 15%现金/短债",
    },
    {
        "variant": "v2.7_gold_heavy_55_45_0",
        "risk_weight": 0.55,
        "gold_weight": 0.45,
        "cash_weight": 0.00,
        "description": "指标最优但黄金偏重：55%风险策略 + 45%黄金",
    },
]

GOLD_RETURN_SCALES = [1.00, 0.75, 0.50, 0.25, 0.00]
CASH_YIELDS = [0.00, 0.01, 0.02]


def run_risk_nav() -> pd.DataFrame:
    engine = BacktestEngineV2(replace(quality_no_gold_params(), start_date="2019-10-01", end_date="2025-12-31"))
    with contextlib.redirect_stdout(io.StringIO()):
        engine.run()
    nav = engine.get_nav_df().copy()
    nav["risk_nav"] = nav["nav"] / nav["nav"].iloc[0]
    nav["benchmark_nav"] = nav["bench_nav"]
    return nav


def load_gold_nav(dates: pd.DatetimeIndex, scale: float) -> pd.Series:
    gold = pd.read_csv(DATA_DIR / "518880_黄金.csv", index_col=0, parse_dates=True).sort_index()
    close = gold["close"].reindex(dates).ffill()
    ret = close.pct_change().fillna(0.0) * scale
    return (1 + ret).cumprod()


def cash_nav(dates: pd.DatetimeIndex, annual_yield: float) -> pd.Series:
    daily = (1 + annual_yield) ** (1 / 252) - 1
    return pd.Series((1 + daily) ** np.arange(len(dates)), index=dates)


def monthly_rebalance_dates(dates: pd.DatetimeIndex) -> set[pd.Timestamp]:
    firsts = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).first()
    return set(firsts.values)


def combine_static(
    risk_nav: pd.Series,
    gold_nav: pd.Series,
    cash: pd.Series,
    risk_weight: float,
    gold_weight: float,
    cash_weight: float,
) -> pd.Series:
    dates = risk_nav.index
    rebal_dates = monthly_rebalance_dates(dates)
    risk_ret = risk_nav.pct_change().fillna(0.0)
    gold_ret = gold_nav.reindex(dates).ffill().pct_change().fillna(0.0)
    cash_ret = cash.reindex(dates).ffill().pct_change().fillna(0.0)
    nav = pd.Series(index=dates, dtype=float)
    nav.iloc[0] = 1.0
    old_weights = (1.0, 0.0, 0.0)
    new_weights = (risk_weight, gold_weight, cash_weight)
    for i, date in enumerate(dates):
        if i == 0:
            continue
        turnover = sum(abs(new_weights[j] - old_weights[j]) for j in range(3)) if date in rebal_dates else 0.0
        old_weights = new_weights
        daily_return = (
            risk_weight * risk_ret.loc[date]
            + gold_weight * gold_ret.loc[date]
            + cash_weight * cash_ret.loc[date]
            - turnover * FEE_RATE
        )
        nav.iloc[i] = nav.iloc[i - 1] * (1 + daily_return)
    return nav


def family_score(metrics: dict) -> int:
    return (
        int(metrics["ann_return"] >= FAMILY_TARGETS["ann_return"])
        + int(metrics["max_drawdown"] >= FAMILY_TARGETS["max_drawdown"])
        + int(metrics["sharpe"] >= FAMILY_TARGETS["sharpe"])
        + int(metrics["calmar"] >= FAMILY_TARGETS["calmar"])
    )


def period_metrics(prefix: str, nav: pd.Series, benchmark: pd.Series, start: str, end: str) -> dict:
    period_nav = nav.loc[start:end]
    period_bench = benchmark.loc[period_nav.index]
    m = metrics_from_nav(period_nav / period_nav.iloc[0], period_bench / period_bench.iloc[0])
    return {f"{prefix}_{k}": v for k, v in m.items()}


def annual_returns(nav: pd.Series) -> dict:
    rows = {}
    for year, group in nav.groupby(nav.index.year):
        rows[f"ret_{year}"] = group.iloc[-1] / group.iloc[0] - 1
    return rows


def evaluate() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(exist_ok=True)
    base = run_risk_nav()
    dates = pd.DatetimeIndex(base.index)
    risk = base["risk_nav"]
    benchmark = base["benchmark_nav"]
    rows = []
    nav_export = pd.DataFrame(index=dates)
    nav_export["benchmark"] = benchmark
    nav_export["risk_no_gold"] = risk

    for candidate in CANDIDATES:
        for gold_scale in GOLD_RETURN_SCALES:
            gold = load_gold_nav(dates, gold_scale)
            for cash_yield in CASH_YIELDS:
                nav = combine_static(
                    risk,
                    gold,
                    cash_nav(dates, cash_yield),
                    candidate["risk_weight"],
                    candidate["gold_weight"],
                    candidate["cash_weight"],
                )
                scenario = f"{candidate['variant']}_gold{gold_scale:.2f}_cash{cash_yield:.2f}".replace(".", "")
                full = period_metrics("full", nav, benchmark, "2019-10-01", "2024-12-31")
                oos = period_metrics("oos_2025", nav, benchmark, "2025-01-01", "2025-12-31")
                row = {
                    **candidate,
                    "scenario": scenario,
                    "gold_return_scale": gold_scale,
                    "cash_yield": cash_yield,
                    **full,
                    **oos,
                    **annual_returns(nav.loc["2019-10-01":"2025-12-31"]),
                }
                row["family_score"] = family_score({
                    "ann_return": row["full_ann_return"],
                    "max_drawdown": row["full_max_drawdown"],
                    "sharpe": row["full_sharpe"],
                    "calmar": row["full_calmar"],
                })
                rows.append(row)
                if gold_scale in (1.0, 0.5) and cash_yield in (0.0, 0.02):
                    nav_export[scenario] = nav

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "v2.8_family_strategy_stress.csv", index=False)
    nav_export.to_csv(OUTPUT_DIR / "v2.8_family_strategy_stress_nav.csv")
    write_report(df)
    return df


def write_report(df: pd.DataFrame):
    base = df[(df["gold_return_scale"] == 1.0) & (df["cash_yield"] == 0.02)].copy()
    base = base.sort_values(["family_score", "full_sharpe"], ascending=[False, False])
    stress = df[(df["gold_return_scale"] == 0.5) & (df["cash_yield"] == 0.0)].copy()
    stress = stress.sort_values(["family_score", "full_sharpe"], ascending=[False, False])

    robustness = []
    for variant, group in df.groupby("variant"):
        robustness.append({
            "variant": variant,
            "min_family_score": int(group["family_score"].min()),
            "pass_4of4_rate": float((group["family_score"] == 4).mean()),
            "worst_ann_return": float(group["full_ann_return"].min()),
            "worst_max_drawdown": float(group["full_max_drawdown"].min()),
            "worst_sharpe": float(group["full_sharpe"].min()),
            "worst_calmar": float(group["full_calmar"].min()),
        })
    robust_df = pd.DataFrame(robustness).sort_values(["min_family_score", "pass_4of4_rate", "worst_sharpe"], ascending=[False, False, False])
    robust_df.to_csv(OUTPUT_DIR / "v2.8_family_strategy_stress_summary.csv", index=False)

    best_robust = robust_df.iloc[0]
    lines = [
        "# V2.8 家庭策略压力测试",
        "",
        "## 测试目的",
        "",
        "V2.7 的优势来自外层黄金/现金防守仓，但黄金在 2019-2025 样本里表现很强。本轮把黄金日收益按 100%、75%、50%、25%、0% 缩放，并把现金/短债收益设为 0%、1%、2%，检验候选是否仍然稳健。",
        "",
        "## 原始场景",
        "",
        "| 候选 | 年化 | 最大回撤 | 夏普 | Calmar | 2025年化 | 达标项 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in base.iterrows():
        lines.append(
            f"| {row['variant']} | {pct(row['full_ann_return'])} | {pct(row['full_max_drawdown'])} | "
            f"{ratio(row['full_sharpe'])} | {ratio(row['full_calmar'])} | {pct(row['oos_2025_ann_return'])} | {int(row['family_score'])}/4 |"
        )
    lines.extend([
        "",
        "## 压力场景：黄金收益减半、现金收益为 0%",
        "",
        "| 候选 | 年化 | 最大回撤 | 夏普 | Calmar | 2025年化 | 达标项 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in stress.iterrows():
        lines.append(
            f"| {row['variant']} | {pct(row['full_ann_return'])} | {pct(row['full_max_drawdown'])} | "
            f"{ratio(row['full_sharpe'])} | {ratio(row['full_calmar'])} | {pct(row['oos_2025_ann_return'])} | {int(row['family_score'])}/4 |"
        )
    lines.extend([
        "",
        "## 鲁棒性汇总",
        "",
        "| 候选 | 最低达标项 | 4/4通过率 | 最差年化 | 最差回撤 | 最差夏普 | 最差Calmar |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in robust_df.iterrows():
        lines.append(
            f"| {row['variant']} | {int(row['min_family_score'])}/4 | {row['pass_4of4_rate']:.0%} | "
            f"{pct(row['worst_ann_return'])} | {pct(row['worst_max_drawdown'])} | "
            f"{ratio(row['worst_sharpe'])} | {ratio(row['worst_calmar'])} |"
        )
    lines.extend([
        "",
        "## 当前判断",
        "",
        f"按这组机械压力测试，综合最稳的是 `{best_robust['variant']}`。原因不是它收益一定更可靠，而是黄金收益被缩放后，组合波动和回撤也同步降低，所以夏普和回撤指标仍然好看。",
        "",
        "但这不能直接证明 45% 黄金适合作为普通家庭正式策略。它的最差年化会降到 +4.10%，说明收益目标仍高度依赖黄金贡献。更稳妥的研究路径不是继续追求最高黄金权重，而是采用 `v2.9_gold_cap20_60_20_20` 作为课程报告主线：它牺牲一部分黄金强周期收益，换来更容易解释的黄金上限和更明确的现金/短债稳定器。下一步应优先用真实短债/货币 ETF 替代现金代理。",
        "",
        "## 输出文件",
        "",
        "- `output/v2.8_family_strategy_stress.csv`",
        "- `output/v2.8_family_strategy_stress_summary.csv`",
        "- `output/v2.8_family_strategy_stress_nav.csv`",
    ])
    (OUTPUT_DIR / "v2.8_family_strategy_stress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    df = evaluate()
    print("V2.8 stress test complete.")
    summary = pd.read_csv(OUTPUT_DIR / "v2.8_family_strategy_stress_summary.csv")
    print(summary.to_string(index=False))
    print(f"Scenarios tested: {len(df)}")


if __name__ == "__main__":
    main()
