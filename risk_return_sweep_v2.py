"""
Multi-objective risk/return sweep for V2.

Goal: find candidates that improve Sharpe and drawdown without giving up
annualized return relative to V2.3a.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import replace
from pathlib import Path

import pandas as pd

from backtest_v2 import BacktestEngineV2, StrategyParams
from variant_research_v2 import CoreMom20RecoveryEngine


OUTPUT_DIR = Path(__file__).parent / "output"
START = "2019-10-01"
END = "2024-12-31"


def run(engine_cls: type[BacktestEngineV2], params: StrategyParams) -> dict:
    engine = engine_cls(params)
    with contextlib.redirect_stdout(io.StringIO()):
        engine.run()
    metrics = engine.get_metrics()
    metrics["n_trades"] = len(engine.trade_log)
    return metrics


def build_grid() -> list[tuple[str, type[BacktestEngineV2], StrategyParams]]:
    base = StrategyParams(start_date=START, end_date=END)
    grid = [("v2.3a_base", BacktestEngineV2, base)]

    engine_options = [
        ("base", BacktestEngineV2),
        ("mom20_core", CoreMom20RecoveryEngine),
    ]
    for engine_name, engine_cls in engine_options:
        for n_sat in [1, 2]:
            for sat_w in [0.08, 0.10]:
                for bear in [0.45, 0.50, 0.55, 0.60]:
                    for cb_drop in [-0.025, -0.030, -0.035]:
                        for cb_cooldown in [10, 15]:
                            params = replace(
                                base,
                                n_satellites=n_sat,
                                satellite_weight_each=sat_w,
                                equity_bear=bear,
                                circuit_breaker_drop=cb_drop,
                                circuit_breaker_cooldown=cb_cooldown,
                            )
                            name = (
                                f"{engine_name}_n{n_sat}_sat{int(sat_w*100)}"
                                f"_bear{int(bear*100)}_cb{abs(cb_drop):.1%}_cd{cb_cooldown}"
                            )
                            grid.append((name, engine_cls, params))
    return grid


def evaluate():
    rows = []
    for name, engine_cls, params in build_grid():
        metrics = run(engine_cls, params)
        rows.append({
            "variant": name,
            "engine": engine_cls.__name__,
            "n_satellites": params.n_satellites,
            "satellite_weight_each": params.satellite_weight_each,
            "equity_bear": params.equity_bear,
            "circuit_breaker_drop": params.circuit_breaker_drop,
            "circuit_breaker_cooldown": params.circuit_breaker_cooldown,
            **metrics,
        })
    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / "v2.4_risk_return_sweep.csv", index=False)
    write_report(df)
    return df


def pct(x: float) -> str:
    return f"{x:+.2%}"


def write_report(df: pd.DataFrame):
    base = df[df["variant"] == "v2.3a_base"].iloc[0]
    viable = df[
        (df["ann_return"] >= base["ann_return"])
        & (df["sharpe"] > base["sharpe"])
        & (df["max_drawdown"] >= base["max_drawdown"])
    ].copy()
    if viable.empty:
        viable = df[
            (df["ann_return"] >= base["ann_return"])
            & (df["max_drawdown"] >= base["max_drawdown"] - 0.01)
        ].copy()
    viable = viable.sort_values(["sharpe", "ann_return", "max_drawdown"], ascending=[False, False, False])
    top = viable.head(10)

    pareto = df.sort_values(["sharpe", "max_drawdown", "ann_return"], ascending=[False, False, False]).head(15)
    best = top.iloc[0]

    lines = [
        "# V2.4 风险收益多目标搜索",
        "",
        "## 搜索目标",
        "",
        "这轮不是追求单一最高年化，而是寻找同时满足三件事的候选：年化不低于 V2.3a、夏普更高、最大回撤不更差。若严格条件过窄，则放宽为最大回撤最多恶化 1 个百分点。",
        "",
        "## V2.3a 基准",
        "",
        f"- 年化收益：{pct(base['ann_return'])}",
        f"- 超额年化：{pct(base['excess'])}",
        f"- 最大回撤：{pct(base['max_drawdown'])}",
        f"- 夏普比率：{base['sharpe']:.3f}",
        f"- 卡玛比率：{base['calmar']:.3f}",
        "",
        "## 最值得关注的候选",
        "",
        "| 变体 | 年化 | 超额 | 最大回撤 | 夏普 | 卡玛 | 交易流水 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {row['variant']} | {pct(row['ann_return'])} | {pct(row['excess'])} | "
            f"{pct(row['max_drawdown'])} | {row['sharpe']:.3f} | {row['calmar']:.3f} | {int(row['n_trades'])} |"
        )

    lines.extend([
        "",
        "## 初步判断",
        "",
        f"- 当前最优候选是 `{best['variant']}`：年化 {pct(best['ann_return'])}，最大回撤 {pct(best['max_drawdown'])}，夏普 {best['sharpe']:.3f}。",
        "- 如果候选只是靠提高仓位获得年化，但回撤没有改善，就不应该升为正式版。",
        "- 如果候选靠减少卫星或提高熔断阈值提高夏普，要重点检查它有没有牺牲 2025 OOS 和 2024 反弹捕捉。",
        "",
        "## Sharpe/回撤前列样本",
        "",
        "| 变体 | 年化 | 最大回撤 | 夏普 | 卡玛 |",
        "|---|---:|---:|---:|---:|",
    ])
    for _, row in pareto.iterrows():
        lines.append(
            f"| {row['variant']} | {pct(row['ann_return'])} | {pct(row['max_drawdown'])} | "
            f"{row['sharpe']:.3f} | {row['calmar']:.3f} |"
        )

    lines.extend([
        "",
        "## 输出",
        "",
        "- `output/v2.4_risk_return_sweep.csv`",
    ])
    (OUTPUT_DIR / "v2.4_risk_return_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = evaluate()
    print(result.sort_values(["sharpe", "ann_return"], ascending=[False, False]).head(20).to_string(index=False))
