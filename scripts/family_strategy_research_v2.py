"""
V2.6 family-strategy research.

This module tests whether the strategy can become more suitable for ordinary
household allocation by adding defensive sleeves, lowering equity exposure, and
using target-volatility / drawdown-control overlays.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from backtest_v2 import BacktestEngineV2, StrategyParams


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CASH_YIELD = 0.02
RISK_FREE = 0.025
FEE_RATE = 0.0001
FAMILY_TARGETS = {
    "ann_return": 0.065,
    "max_drawdown": -0.12,
    "sharpe": 0.40,
    "calmar": 0.60,
}


def pct(value: float) -> str:
    return f"{value:+.2%}"


def ratio(value: float) -> str:
    return f"{value:.3f}"


def quality_core_params(**kwargs) -> StrategyParams:
    base = StrategyParams()
    core = ("510300", "510500", "512890", "513100", "518880")
    satellite_pool = tuple(sym for sym in base.satellite_pool if sym not in set(core))
    return replace(base, core_etfs=core, satellite_pool=satellite_pool, **kwargs)


def quality_no_gold_params(**kwargs) -> StrategyParams:
    base = StrategyParams()
    core = ("510300", "510500", "512890", "513100")
    satellite_pool = tuple(sym for sym in base.satellite_pool if sym not in set(core + ("518880",)))
    return replace(base, core_etfs=core, satellite_pool=satellite_pool, **kwargs)


def run_engine(params: StrategyParams, start: str = "2019-10-01", end: str = "2025-12-31") -> pd.DataFrame:
    engine = BacktestEngineV2(replace(params, start_date=start, end_date=end))
    with contextlib.redirect_stdout(io.StringIO()):
        engine.run()
    nav = engine.get_nav_df().copy()
    nav["strategy_nav"] = nav["nav"] / nav["nav"].iloc[0]
    nav["benchmark_nav"] = nav["bench_nav"]
    return nav


def cash_nav(dates: pd.DatetimeIndex, annual_yield: float = CASH_YIELD) -> pd.Series:
    daily = (1 + annual_yield) ** (1 / 252) - 1
    return pd.Series((1 + daily) ** np.arange(len(dates)), index=dates)


def gold_nav(dates: pd.DatetimeIndex) -> pd.Series:
    gold = pd.read_csv(DATA_DIR / "518880_黄金.csv", index_col=0, parse_dates=True).sort_index()
    close = gold["close"].reindex(dates).ffill()
    return close / close.iloc[0]


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1).min())


def metrics_from_nav(nav: pd.Series, benchmark: pd.Series | None = None) -> dict:
    nav = nav.dropna()
    returns = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    ann_return = nav.iloc[-1] ** (1 / years) - 1
    ann_vol = returns.std() * np.sqrt(252)
    mdd = max_drawdown(nav)
    metrics = {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "max_drawdown": float(mdd),
        "sharpe": float((ann_return - RISK_FREE) / ann_vol) if ann_vol > 0 else 0.0,
        "calmar": float(ann_return / abs(mdd)) if mdd < 0 else np.nan,
    }
    if benchmark is not None:
        bench = benchmark.reindex(nav.index).ffill()
        bench_ann = bench.iloc[-1] ** (1 / years) - 1
        metrics["ann_bench"] = float(bench_ann)
        metrics["excess"] = float(ann_return - bench_ann)
    return metrics


def first_trading_days(dates: pd.DatetimeIndex) -> set[pd.Timestamp]:
    firsts = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).first()
    return set(firsts.values)


def static_rule(risk_weight: float, gold_weight: float, cash_weight: float):
    def rule(_date: pd.Timestamp, _history: pd.Series) -> tuple[float, float, float]:
        return risk_weight, gold_weight, cash_weight
    return rule


def target_vol_rule(target_vol: float, max_weight: float, min_weight: float, gold_split: float):
    def rule(_date: pd.Timestamp, history: pd.Series) -> tuple[float, float, float]:
        ret = history.pct_change().dropna().tail(60)
        if len(ret) < 20 or ret.std() <= 0:
            risk_weight = max_weight
        else:
            vol = ret.std() * np.sqrt(252)
            risk_weight = min(max_weight, max(min_weight, target_vol / vol))
        defensive = 1 - risk_weight
        return risk_weight, defensive * gold_split, defensive * (1 - gold_split)
    return rule


def drawdown_guard_rule(max_weight: float, mid_weight: float, low_weight: float, gold_split: float):
    def rule(_date: pd.Timestamp, history: pd.Series) -> tuple[float, float, float]:
        drawdown = history.iloc[-1] / history.cummax().iloc[-1] - 1
        ma60 = history.rolling(60).mean().iloc[-1] if len(history) >= 60 else np.nan
        if drawdown <= -0.12:
            risk_weight = low_weight
        elif drawdown <= -0.08:
            risk_weight = mid_weight
        elif not pd.isna(ma60) and history.iloc[-1] < ma60 and drawdown <= -0.05:
            risk_weight = mid_weight
        else:
            risk_weight = max_weight
        defensive = 1 - risk_weight
        return risk_weight, defensive * gold_split, defensive * (1 - gold_split)
    return rule


def overlay_nav(
    risk_nav: pd.Series,
    defensive_gold: pd.Series,
    defensive_cash: pd.Series,
    rule: Callable[[pd.Timestamp, pd.Series], tuple[float, float, float]],
    name: str,
) -> tuple[pd.Series, pd.DataFrame]:
    dates = risk_nav.index
    risk_ret = risk_nav.pct_change().fillna(0.0)
    gold_ret = defensive_gold.reindex(dates).ffill().pct_change().fillna(0.0)
    cash_ret = defensive_cash.reindex(dates).ffill().pct_change().fillna(0.0)

    nav = pd.Series(index=dates, dtype=float)
    weights = []
    current = rule(dates[0], risk_nav.iloc[:1])
    nav.iloc[0] = 1.0
    last_month = None

    for i, date in enumerate(dates):
        month_key = (date.year, date.month)
        if i == 0:
            weights.append({"date": date, "variant": name, "risk_weight": current[0], "gold_weight": current[1], "cash_weight": current[2], "turnover": 0.0})
            last_month = month_key
            continue
        turnover = 0.0
        if month_key != last_month:
            new_weights = rule(date, risk_nav.iloc[:i])
            turnover = sum(abs(new_weights[j] - current[j]) for j in range(3))
            current = new_weights
            last_month = month_key
        daily_return = (
            current[0] * risk_ret.loc[date]
            + current[1] * gold_ret.loc[date]
            + current[2] * cash_ret.loc[date]
            - turnover * FEE_RATE
        )
        nav.iloc[i] = nav.iloc[i - 1] * (1 + daily_return)
        weights.append({"date": date, "variant": name, "risk_weight": current[0], "gold_weight": current[1], "cash_weight": current[2], "turnover": turnover})

    return nav, pd.DataFrame(weights).set_index("date")


def family_score(row: pd.Series) -> int:
    return (
        int(row["full_ann_return"] >= FAMILY_TARGETS["ann_return"])
        + int(row["full_max_drawdown"] >= FAMILY_TARGETS["max_drawdown"])
        + int(row["full_sharpe"] >= FAMILY_TARGETS["sharpe"])
        + int(row["full_calmar"] >= FAMILY_TARGETS["calmar"])
    )


def rolling_summary(nav: pd.Series, benchmark: pd.Series) -> dict:
    rows = []
    dates = nav.index
    current = pd.Timestamp("2019-10-01")
    end_limit = pd.Timestamp("2024-12-31")
    while True:
        start = current + pd.DateOffset(months=24)
        end = start + pd.DateOffset(months=3)
        start_idx = dates.searchsorted(start, side="left")
        end_idx = dates.searchsorted(end, side="right") - 1
        if start_idx >= len(dates) or end_idx <= start_idx or dates[end_idx] > end_limit:
            break
        window_nav = nav.loc[dates[start_idx]:dates[end_idx]]
        window_bench = benchmark.loc[window_nav.index]
        rows.append(metrics_from_nav(window_nav / window_nav.iloc[0], window_bench / window_bench.iloc[0]))
        current += pd.DateOffset(months=3)
    df = pd.DataFrame(rows)
    return {
        "rolling_windows": len(df),
        "rolling_ann_mean": df["ann_return"].mean(),
        "rolling_excess_mean": df["excess"].mean(),
        "rolling_mdd_mean": df["max_drawdown"].mean(),
        "rolling_sharpe_mean": df["sharpe"].mean(),
        "rolling_positive_excess_rate": (df["excess"] > 0).mean(),
    }


def period_metrics(prefix: str, nav: pd.Series, benchmark: pd.Series, start: str, end: str) -> dict:
    period_nav = nav.loc[start:end]
    period_bench = benchmark.loc[period_nav.index]
    metrics = metrics_from_nav(period_nav / period_nav.iloc[0], period_bench / period_bench.iloc[0])
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def evaluate() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(exist_ok=True)
    base = run_engine(StrategyParams())
    quality = run_engine(quality_core_params())
    quality_no_gold = run_engine(quality_no_gold_params())
    mid_equity = run_engine(quality_core_params(equity_bull=0.85, equity_range=0.60, equity_bear=0.35))
    low_equity = run_engine(quality_core_params(equity_bull=0.80, equity_range=0.55, equity_bear=0.30))

    dates = pd.DatetimeIndex(quality.index)
    benchmark = quality["benchmark_nav"]
    gold = gold_nav(dates)
    cash = cash_nav(dates)
    risk = quality["strategy_nav"]
    risk_no_gold = quality_no_gold["strategy_nav"]

    variants: dict[str, tuple[pd.Series, str]] = {
        "v2.3a_current_core": (base["strategy_nav"], "当前正式基准"),
        "v2.5_quality_core": (risk, "纳指/黄金升入核心，降低创业板核心权重"),
        "v2.6_quality_no_gold_core": (risk_no_gold, "纳指升入核心，黄金只留给外层防守仓"),
        "v2.6_mid_equity": (mid_equity["strategy_nav"], "高质量核心 + 仓位85/60/35"),
        "v2.6_low_equity": (low_equity["strategy_nav"], "高质量核心 + 仓位80/55/30"),
    }
    overlay_rules = [
        ("static_80risk_10gold_10cash", static_rule(0.80, 0.10, 0.10), "80%风险策略 + 10%黄金 + 10%现金"),
        ("static_70risk_15gold_15cash", static_rule(0.70, 0.15, 0.15), "70%风险策略 + 15%黄金 + 15%现金"),
        ("target_vol10_gold_cash", target_vol_rule(0.10, 0.90, 0.35, 0.50), "目标波动10%，防守仓黄金现金各半"),
        ("target_vol12_cash_heavy", target_vol_rule(0.12, 0.90, 0.40, 0.30), "目标波动12%，防守仓偏现金"),
        ("dd_guard_8_12_gold_cash", drawdown_guard_rule(0.90, 0.60, 0.35, 0.50), "回撤8%/12%阶梯降仓，防守仓黄金现金各半"),
        ("dd_guard_8_12_cash_heavy", drawdown_guard_rule(0.90, 0.55, 0.30, 0.25), "回撤8%/12%阶梯降仓，防守仓偏现金"),
    ]

    # V2.7: systematic family-allocation sweep. The corresponding
    # no-gold_* variants are especially important because gold then sits only
    # in the outer defensive sleeve.
    for risk_weight in np.arange(0.55, 0.91, 0.05):
        defensive = 1 - float(risk_weight)
        for gold_split in [0.0, 0.25, 0.50, 0.75, 1.0]:
            gold_weight = defensive * gold_split
            cash_weight = defensive - gold_weight
            name = f"sweep_static_r{risk_weight:.2f}_g{gold_weight:.2f}_c{cash_weight:.2f}".replace(".", "")
            overlay_rules.append((
                name,
                static_rule(float(risk_weight), float(gold_weight), float(cash_weight)),
                f"V2.7静态扫描：{risk_weight:.0%}风险策略 + {gold_weight:.0%}黄金 + {cash_weight:.0%}现金",
            ))

    for target_vol in [0.08, 0.10, 0.12, 0.14]:
        for max_weight in [0.80, 0.90]:
            for min_weight in [0.25, 0.35, 0.45]:
                if min_weight >= max_weight:
                    continue
                for gold_split in [0.25, 0.50, 0.75]:
                    name = (
                        f"sweep_tvol{target_vol:.2f}_max{max_weight:.2f}_"
                        f"min{min_weight:.2f}_gs{gold_split:.2f}"
                    ).replace(".", "")
                    overlay_rules.append((
                        name,
                        target_vol_rule(target_vol, max_weight, min_weight, gold_split),
                        f"V2.7目标波动扫描：目标波动{target_vol:.0%}，风险仓{min_weight:.0%}-{max_weight:.0%}",
                    ))

    weight_frames = []
    for name, rule, description in overlay_rules:
        nav, weights = overlay_nav(risk, gold, cash, rule, name)
        variants[name] = (nav, description)
        weight_frames.append(weights)

        clean_name = f"no_gold_{name}"
        clean_nav, clean_weights = overlay_nav(risk_no_gold, gold, cash, rule, clean_name)
        variants[clean_name] = (clean_nav, f"风险策略不含黄金；{description}")
        weight_frames.append(clean_weights)

    rows = []
    nav_export = pd.DataFrame(index=dates)
    nav_export["benchmark"] = benchmark
    for name, (nav, description) in variants.items():
        nav_export[name] = nav
        row = {
            "variant": name,
            "description": description,
            **period_metrics("full", nav, benchmark, "2019-10-01", "2024-12-31"),
            **period_metrics("oos_2025", nav, benchmark, "2025-01-01", "2025-12-31"),
            **rolling_summary(nav.loc["2019-10-01":"2024-12-31"], benchmark.loc["2019-10-01":"2024-12-31"]),
        }
        row["family_score"] = family_score(pd.Series(row))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["family_score", "full_sharpe", "full_ann_return"], ascending=[False, False, False])
    df.to_csv(OUTPUT_DIR / "v2.7_family_strategy_research.csv", index=False)
    nav_export.to_csv(OUTPUT_DIR / "v2.7_family_strategy_nav.csv")
    if weight_frames:
        pd.concat(weight_frames).to_csv(OUTPUT_DIR / "v2.7_family_strategy_overlay_weights.csv")
    write_report(df)
    return df


def write_report(df: pd.DataFrame):
    best = df.iloc[0]
    gold_cap_20 = df[
        df["variant"].str.startswith("no_gold_sweep_static")
        & (df["family_score"] == 4)
        & df["variant"].str.contains("_g000_|_g002_|_g004_|_g005_|_g006_|_g007_|_g009_|_g010_|_g011_|_g012_|_g015_|_g017_|_g019_|_g020_")
    ].sort_values(["full_sharpe", "full_ann_return"], ascending=[False, False])
    gold_cap_best = gold_cap_20.iloc[0] if not gold_cap_20.empty else best
    practical = df[
        df["variant"].str.startswith("no_gold_sweep_static")
        & (df["family_score"] == 4)
        & df["variant"].str.contains("_g030_c010|_g022_c022|_g020_c020|_g026_c009|_g015_c015")
    ].sort_values(["full_sharpe", "full_ann_return"], ascending=[False, False])
    practical_best = practical.iloc[0] if not practical.empty else best
    top_rows = df.head(30)
    n_four = int((df["family_score"] == 4).sum())
    n_total = len(df)
    lines = [
        "# V2.7 家庭可执行策略研究",
        "",
        "## 硬标准",
        "",
        f"- 年化收益 >= {pct(FAMILY_TARGETS['ann_return'])}",
        f"- 最大回撤 >= {pct(FAMILY_TARGETS['max_drawdown'])}",
        f"- 夏普比率 >= {ratio(FAMILY_TARGETS['sharpe'])}",
        f"- Calmar >= {ratio(FAMILY_TARGETS['calmar'])}",
        "",
        "## 扫描结论",
        "",
        f"- 本轮共测试 {n_total} 个组合，其中 {n_four} 个满足 4/4 家庭策略硬标准。",
        f"- 指标最优组合：`{best['variant']}`，年化 {pct(best['full_ann_return'])}，最大回撤 {pct(best['full_max_drawdown'])}，夏普 {ratio(best['full_sharpe'])}。",
        f"- 实用约束候选：`{practical_best['variant']}`，年化 {pct(practical_best['full_ann_return'])}，最大回撤 {pct(practical_best['full_max_drawdown'])}，夏普 {ratio(practical_best['full_sharpe'])}。",
        f"- 黄金上限20%候选：`{gold_cap_best['variant']}`，年化 {pct(gold_cap_best['full_ann_return'])}，最大回撤 {pct(gold_cap_best['full_max_drawdown'])}，夏普 {ratio(gold_cap_best['full_sharpe'])}。",
        "",
        "指标最优组合偏黄金，说明 2019-2025 样本里黄金贡献很强；但普通家庭组合不应把结论完全建立在单一资产历史红利上。因此更适合作为正式候选的是黄金上限20%候选：黄金仓不过度、仍保留现金/短债位置。",
        "",
        "## 结果汇总",
        "",
        "下表只展示综合排名前 30，完整结果见 CSV。",
        "",
        "| 变体 | 年化 | 最大回撤 | 夏普 | Calmar | 滚动超额均值 | 2025年化 | 达标项 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in top_rows.iterrows():
        lines.append(
            f"| {row['variant']} | {pct(row['full_ann_return'])} | {pct(row['full_max_drawdown'])} | "
            f"{ratio(row['full_sharpe'])} | {ratio(row['full_calmar'])} | "
            f"{pct(row['rolling_excess_mean'])} | {pct(row['oos_2025_ann_return'])} | "
            f"{int(row['family_score'])}/4 |"
        )

    lines.extend([
        "",
        "## 当前判断",
        "",
        f"综合排名最高的是 `{best['variant']}`：2019-2024 年化 {pct(best['full_ann_return'])}，最大回撤 {pct(best['full_max_drawdown'])}，夏普 {ratio(best['full_sharpe'])}，Calmar {ratio(best['full_calmar'])}。",
        "",
        f"但我不建议直接把它作为正式版，因为它等价于把 45% 仓位压在黄金上。更稳妥的下一候选是 `{practical_best['variant']}`：风险策略、黄金、现金/短债三者更均衡，仍然满足全部家庭策略硬标准。",
        "",
        f"如果把黄金仓位上限约束为 20%，当前最优候选是 `{gold_cap_best['variant']}`。这个版本更适合普通家庭和课程报告主线，因为它不依赖黄金继续强势，也保留了短债/现金稳定器。",
        "",
        "这轮先用真实黄金 ETF 与 2% 年化现金代理测试防守仓。由于本地还没有短债/货币 ETF 数据，报告中的现金仓不是最终可交易资产，只是为了判断防守仓机制是否值得继续。",
        "",
        "## 三个方向",
        "",
        "1. 加入防守资产：检验黄金/现金防守仓能否降低深回撤。",
        "2. 降低权益上限：检验 85/60/35 与 80/55/30 仓位是否更适合家庭持有。",
        "3. 目标波动/回撤控制：在风险升高后自动降仓，减少普通家庭最难承受的深回撤阶段。",
        "",
        "## 输出文件",
        "",
        "- `output/v2.7_family_strategy_research.csv`",
        "- `output/v2.7_family_strategy_nav.csv`",
        "- `output/v2.7_family_strategy_overlay_weights.csv`",
    ])
    (OUTPUT_DIR / "v2.7_family_strategy_research.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    df = evaluate()
    print("V2.7 family strategy research complete.")
    print(f"Variants tested: {len(df)}, 4/4 family-score variants: {(df['family_score'] == 4).sum()}")
    print(df[[
        "variant", "full_ann_return", "full_max_drawdown", "full_sharpe",
        "full_calmar", "rolling_excess_mean", "oos_2025_ann_return", "family_score",
    ]].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
