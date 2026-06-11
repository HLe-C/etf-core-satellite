"""
Variant research for V2.3a.

Focus: diagnose the 2024 underperformance and test conservative recovery
mechanisms without changing the production strategy.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pandas as pd

from backtest_v2 import BacktestEngineV2, StrategyParams
from rolling_window_v2 import generate_windows, available_benchmark_dates


OUTPUT_DIR = Path(__file__).parent / "output"


class CoreMA50RecoveryEngine(BacktestEngineV2):
    """In bear state, allow core ETFs above MA50 back into the core sleeve."""

    def _filter_active_core(self, date: pd.Timestamp, state: str) -> list[str]:
        if state != "bear":
            return list(self.p.core_etfs)
        active = []
        for sym in self.p.core_etfs:
            if sym == "512890":
                active.append(sym)
                continue
            ind = self._get_indicator(sym, date)
            close = self._get_close(sym, date)
            if ind is None or close is None or pd.isna(ind["MA50"]):
                active.append(sym)
            elif close > ind["MA50"]:
                active.append(sym)
        return active or ["512890"]


class CoreMom20RecoveryEngine(BacktestEngineV2):
    """In bear state, allow core ETFs with positive 20-day momentum."""

    def _filter_active_core(self, date: pd.Timestamp, state: str) -> list[str]:
        if state != "bear":
            return list(self.p.core_etfs)
        active = []
        for sym in self.p.core_etfs:
            if sym == "512890":
                active.append(sym)
                continue
            ind = self._get_indicator(sym, date)
            close = self._get_close(sym, date)
            if ind is None or close is None or pd.isna(ind["mom20"]):
                active.append(sym)
            elif ind["mom20"] > 0 and close > ind["MA20"]:
                active.append(sym)
        return active or ["512890"]


class ConditionalBear60Mom20RecoveryEngine(CoreMom20RecoveryEngine):
    """Use bear 60% only when HS300 has positive 20-day momentum."""

    def _get_equity_target(self, state: str, date: pd.Timestamp = None) -> float:
        if state != "bear" or date is None:
            return super()._get_equity_target(state, date)
        row = self.bench_ind.loc[date]
        if row.get("mom20", 0) > 0 and row["close"] > row["MA20"]:
            return 0.60
        return self.p.equity_bear


def run_variant(
    name: str,
    engine_cls: type[BacktestEngineV2],
    params: StrategyParams,
    start: str,
    end: str,
) -> tuple[dict, pd.DataFrame]:
    engine = engine_cls(replace(params, start_date=start, end_date=end))
    with contextlib.redirect_stdout(io.StringIO()):
        engine.run()
    metrics = engine.get_metrics()
    metrics["variant"] = name
    metrics["start"] = start
    metrics["end"] = end
    return metrics, engine.get_nav_df()


def window_return(nav: pd.DataFrame, start: str, end: str) -> dict:
    g = nav.loc[start:end]
    if len(g) < 2:
        return {"strategy": None, "benchmark": None, "excess": None}
    strategy = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1
    benchmark = g["bench_nav"].iloc[-1] / g["bench_nav"].iloc[0] - 1
    return {"strategy": strategy, "benchmark": benchmark, "excess": strategy - benchmark}


def evaluate_variants():
    base = StrategyParams()
    variants: list[tuple[str, type[BacktestEngineV2], StrategyParams]] = [
        ("v2.3a_base", BacktestEngineV2, base),
        ("bear_equity_60", BacktestEngineV2, replace(base, equity_bear=0.60)),
        ("bear_equity_70", BacktestEngineV2, replace(base, equity_bear=0.70)),
        ("core_ma50_recovery", CoreMA50RecoveryEngine, base),
        ("core_mom20_recovery", CoreMom20RecoveryEngine, base),
        ("mom20_recovery_bear60", CoreMom20RecoveryEngine, replace(base, equity_bear=0.60)),
        ("conditional_bear60_mom20", ConditionalBear60Mom20RecoveryEngine, base),
    ]

    rows = []
    for name, cls, params in variants:
        full, nav = run_variant(name, cls, params, "2019-10-01", "2024-12-31")
        y2024 = window_return(nav, "2024-01-02", "2024-12-31")
        rally = window_return(nav, "2024-09-24", "2024-10-08")
        selloff = window_return(nav, "2024-10-08", "2024-10-31")
        rolling = rolling_summary(name, cls, params)
        oos, _ = run_variant(name, cls, params, "2025-01-01", "2025-12-31")
        rows.append({
            "variant": name,
            "ann_return": full["ann_return"],
            "excess": full["excess"],
            "max_drawdown": full["max_drawdown"],
            "sharpe": full["sharpe"],
            "calmar": full["calmar"],
            "n_trades": full["n_trades"],
            "y2024_strategy": y2024["strategy"],
            "y2024_benchmark": y2024["benchmark"],
            "y2024_excess": y2024["excess"],
            "rally_2024_strategy": rally["strategy"],
            "rally_2024_benchmark": rally["benchmark"],
            "selloff_2024_strategy": selloff["strategy"],
            "selloff_2024_benchmark": selloff["benchmark"],
            "rolling_test_ann_mean": rolling["ann_return_mean"],
            "rolling_test_excess_mean": rolling["excess_mean"],
            "rolling_positive_excess_rate": rolling["positive_excess_rate"],
            "rolling_positive_return_rate": rolling["positive_return_rate"],
            "oos_2025_ann_return": oos["ann_return"],
            "oos_2025_excess": oos["excess"],
            "oos_2025_max_drawdown": oos["max_drawdown"],
        })

    df = pd.DataFrame(rows)
    window_df = compare_candidate_windows()
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / "v2.3a_variant_research.csv", index=False)
    window_df.to_csv(OUTPUT_DIR / "v2.3b_candidate_window_comparison.csv", index=False)
    write_report(df, window_df)
    return df


def compare_candidate_windows() -> pd.DataFrame:
    base_params = StrategyParams()
    candidate_params = replace(base_params, equity_bear=0.60)
    dates = available_benchmark_dates(base_params)
    windows = generate_windows(dates)
    rows = []
    for w in windows:
        start = str(w["test_start"].date())
        end = str(w["test_end"].date())
        base, _ = run_variant("v2.3a_base", BacktestEngineV2, base_params, start, end)
        cand, _ = run_variant("mom20_recovery_bear60", CoreMom20RecoveryEngine, candidate_params, start, end)
        cond, _ = run_variant("conditional_bear60_mom20", ConditionalBear60Mom20RecoveryEngine, base_params, start, end)
        rows.append({
            "window": w["window"],
            "test_start": start,
            "test_end": end,
            "base_ann_return": base["ann_return"],
            "candidate_ann_return": cand["ann_return"],
            "delta_ann_return": cand["ann_return"] - base["ann_return"],
            "base_excess": base["excess"],
            "candidate_excess": cand["excess"],
            "delta_excess": cand["excess"] - base["excess"],
            "base_max_drawdown": base["max_drawdown"],
            "candidate_max_drawdown": cand["max_drawdown"],
            "delta_max_drawdown": cand["max_drawdown"] - base["max_drawdown"],
            "base_trades": base["n_trades"],
            "candidate_trades": cand["n_trades"],
            "delta_trades": cand["n_trades"] - base["n_trades"],
            "conditional_excess": cond["excess"],
            "conditional_delta_excess": cond["excess"] - base["excess"],
            "conditional_max_drawdown": cond["max_drawdown"],
            "conditional_delta_max_drawdown": cond["max_drawdown"] - base["max_drawdown"],
            "conditional_trades": cond["n_trades"],
            "conditional_delta_trades": cond["n_trades"] - base["n_trades"],
        })
    return pd.DataFrame(rows)


def rolling_summary(name: str, engine_cls: type[BacktestEngineV2], params: StrategyParams) -> dict:
    dates = available_benchmark_dates(params)
    windows = generate_windows(dates)
    rows = []
    for w in windows:
        metrics, _ = run_variant(
            name,
            engine_cls,
            params,
            str(w["test_start"].date()),
            str(w["test_end"].date()),
        )
        rows.append(metrics)
    df = pd.DataFrame(rows)
    return {
        "ann_return_mean": df["ann_return"].mean(),
        "excess_mean": df["excess"].mean(),
        "positive_excess_rate": (df["excess"] > 0).mean(),
        "positive_return_rate": (df["ann_return"] > 0).mean(),
    }


def pct(value: float) -> str:
    return f"{value:+.2%}"


def write_report(df: pd.DataFrame, window_df: pd.DataFrame):
    base = df[df["variant"] == "v2.3a_base"].iloc[0]
    ranked = df.sort_values(["excess", "max_drawdown"], ascending=[False, False])
    best_return = ranked.iloc[0]
    risk_floor = float(base["max_drawdown"]) - 0.01
    constrained = df[df["max_drawdown"] >= risk_floor].sort_values(
        ["rolling_test_excess_mean", "excess"], ascending=[False, False]
    )
    best_constrained = constrained.iloc[0]

    lines = [
        "# V2.3a 变体研究：2024 修复方向",
        "",
        "## 为什么研究这个方向",
        "",
        "V2.3a 的最大相对短板是 2024：策略 -4.11%，基准 +16.20%，相对落后 -20.31%。诊断显示问题不是大熊市防守，而是反弹市重新上车太慢：2024-09-24 至 2024-10-08，基准上涨约 +26.98%，策略只上涨约 +3.63%。",
        "",
        "因此本轮只测试低复杂度恢复机制：提高熊市权益仓位，或允许核心 ETF 在熊市中用 MA50/20日动量更早回到核心仓。没有新增宏观预测，也没有扩大卫星池。",
        "",
        "## 结果汇总",
        "",
        "| 变体 | 年化 | 超额年化 | 最大回撤 | 2024超额 | 滚动超额均值 | 滚动正超额 | 2025超额 | 交易流水 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['variant']} | {pct(row['ann_return'])} | {pct(row['excess'])} | "
            f"{pct(row['max_drawdown'])} | {pct(row['y2024_excess'])} | "
            f"{pct(row['rolling_test_excess_mean'])} | {row['rolling_positive_excess_rate']:.0%} | "
            f"{pct(row['oos_2025_excess'])} | {int(row['n_trades'])} |"
        )

    lines.extend([
        "",
        "## 初步判断",
        "",
        f"- 当前基准 V2.3a：年化 {pct(base['ann_return'])}，超额 {pct(base['excess'])}，最大回撤 {pct(base['max_drawdown'])}，滚动超额均值 {pct(base['rolling_test_excess_mean'])}。",
        f"- 收益最高变体是 `{best_return['variant']}`：年化 {pct(best_return['ann_return'])}，超额 {pct(best_return['excess'])}，但最大回撤扩大到 {pct(best_return['max_drawdown'])}，不符合家庭组合的防守约束。",
        f"- 风险约束下最值得跟踪的是 `{best_constrained['variant']}`：年化 {pct(best_constrained['ann_return'])}，超额 {pct(best_constrained['excess'])}，最大回撤 {pct(best_constrained['max_drawdown'])}，滚动超额均值 {pct(best_constrained['rolling_test_excess_mean'])}。",
        f"- 逐窗口看，`mom20_recovery_bear60` 在 {int((window_df['delta_excess'] > 0).sum())}/{len(window_df)} 个外推窗口改善超额，平均多贡献 {pct(window_df['delta_excess'].mean())}；但平均交易流水增加 {window_df['delta_trades'].mean():.1f} 条。",
        "- 条件版 `conditional_bear60_mom20` 没有通过：它试图只在 HS300 20日动量转强时把熊市仓位提到 60%，但年化只到 +5.28%，最大回撤扩大到 -17.50%，滚动正超额比例降到 46%。这说明短期指数动量作为恢复开关噪声偏高。",
        "",
        "但我不建议立刻把最强变体升为正式版。原因是这些机制明显针对 2024 的反弹错失问题，存在样本后视风险。更稳妥的下一步是把最有希望的 1-2 个变体放进滚动窗口/OOS 框架，检查它们是否只是修复 2024，还是在多数窗口里都改善风险收益。",
        "",
        "## V2.3b候选逐窗口对比",
        "",
        "| 窗口 | 区间 | 超额改善 | 回撤变化 | 交易变化 |",
        "|---:|---|---:|---:|---:|",
    ])
    for _, row in window_df.iterrows():
        lines.append(
            f"| {int(row['window'])} | {row['test_start']} 至 {row['test_end']} | "
            f"{pct(row['delta_excess'])} | {pct(row['delta_max_drawdown'])} | "
            f"{int(row['delta_trades']):+d} |"
        )

    lines.extend([
        "",
        "## 输出",
        "",
        "- `output/v2.3a_variant_research.csv`",
        "- `output/v2.3b_candidate_window_comparison.csv`",
    ])
    (OUTPUT_DIR / "v2.3a_variant_research.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = evaluate_variants()
    print(result.to_string(index=False))
