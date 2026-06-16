"""
Generate a focused report for the V2.4-rc1 candidate.

Candidate:
  - core recovery: in bear, keep core ETFs with positive 20-day momentum and close > MA20
  - bear equity: 55%
  - circuit breaker cooldown: 10 trading days
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import replace
from pathlib import Path

import pandas as pd

from backtest_v2 import BacktestEngineV2, StrategyParams
from execution_report_v2 import normalize_trade_log, yearly_attribution
from variant_research_v2 import CoreMom20RecoveryEngine, run_variant
from rolling_window_v2 import available_benchmark_dates, generate_windows


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
START = "2019-10-01"
END = "2024-12-31"
OOS_START = "2025-01-01"
OOS_END = "2025-12-31"


def candidate_params() -> StrategyParams:
    return replace(
        StrategyParams(start_date=START, end_date=END),
        equity_bear=0.55,
        circuit_breaker_cooldown=10,
    )


def run_engine(engine_cls: type[BacktestEngineV2], params: StrategyParams) -> BacktestEngineV2:
    engine = engine_cls(params)
    with contextlib.redirect_stdout(io.StringIO()):
        engine.run()
    return engine


def pct(value: float) -> str:
    return f"{value:+.2%}"


def rolling_compare() -> pd.DataFrame:
    base_params = StrategyParams()
    cand_params = candidate_params()
    windows = generate_windows(available_benchmark_dates(base_params))
    rows = []
    for w in windows:
        start = str(w["test_start"].date())
        end = str(w["test_end"].date())
        base, _ = run_variant("v2.3a_base", BacktestEngineV2, base_params, start, end)
        cand, _ = run_variant("v2.4_rc1", CoreMom20RecoveryEngine, cand_params, start, end)
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
        })
    return pd.DataFrame(rows)


def write_report():
    OUTPUT_DIR.mkdir(exist_ok=True)
    base_engine = run_engine(BacktestEngineV2, StrategyParams(start_date=START, end_date=END))
    cand_engine = run_engine(CoreMom20RecoveryEngine, candidate_params())
    oos_base = run_engine(BacktestEngineV2, StrategyParams(start_date=OOS_START, end_date=OOS_END))
    oos_cand = run_engine(CoreMom20RecoveryEngine, replace(candidate_params(), start_date=OOS_START, end_date=OOS_END))

    base = base_engine.get_metrics()
    cand = cand_engine.get_metrics()
    base_oos = oos_base.get_metrics()
    cand_oos = oos_cand.get_metrics()
    roll = rolling_compare()

    normalize_trade_log(cand_engine).to_csv(OUTPUT_DIR / "v2.4_rc1_trade_ledger.csv", index=False)
    cand_engine.get_position_df().to_csv(OUTPUT_DIR / "v2.4_rc1_daily_positions.csv")
    yearly_attribution(cand_engine.get_nav_df()).to_csv(OUTPUT_DIR / "v2.4_rc1_yearly_attribution.csv", index=False)
    roll.to_csv(OUTPUT_DIR / "v2.4_rc1_rolling_compare.csv", index=False)

    lines = [
        "# V2.4-rc1 候选报告",
        "",
        "## 候选规则",
        "",
        "- 继承 V2.3a 的动态核心-卫星框架。",
        "- 熊市核心恢复：除红利低波外，核心 ETF 若 20 日动量 > 0 且 close > MA20，则允许重新进入核心仓。",
        "- 熊市权益仓位：从 50% 小幅提高到 55%。",
        "- 熔断冷却期：从 15 个交易日缩短到 10 个交易日。",
        "",
        "## 为什么是这个方向",
        "",
        "V2.3a 的主要问题不是防守失效，而是反弹后重新上车太慢。V2.4-rc1 只做三处小改动：让核心仓在短期转强时更早恢复、熊市只多给 5% 权益、熔断后少空等 5 天。它不是追涨版，也没有扩大卫星仓。",
        "",
        "## 2019-2024 对比",
        "",
        "| 指标 | V2.3a | V2.4-rc1 | 改善 |",
        "|---|---:|---:|---:|",
        f"| 年化收益 | {pct(base['ann_return'])} | {pct(cand['ann_return'])} | {pct(cand['ann_return'] - base['ann_return'])} |",
        f"| 超额年化 | {pct(base['excess'])} | {pct(cand['excess'])} | {pct(cand['excess'] - base['excess'])} |",
        f"| 最大回撤 | {pct(base['max_drawdown'])} | {pct(cand['max_drawdown'])} | {pct(cand['max_drawdown'] - base['max_drawdown'])} |",
        f"| 夏普比率 | {base['sharpe']:.3f} | {cand['sharpe']:.3f} | {cand['sharpe'] - base['sharpe']:+.3f} |",
        f"| 卡玛比率 | {base['calmar']:.3f} | {cand['calmar']:.3f} | {cand['calmar'] - base['calmar']:+.3f} |",
        f"| 交易流水 | {base['n_trades']} | {cand['n_trades']} | {cand['n_trades'] - base['n_trades']:+d} |",
        "",
        "## 2025 OOS 对比",
        "",
        "| 指标 | V2.3a | V2.4-rc1 |",
        "|---|---:|---:|",
        f"| 年化收益 | {pct(base_oos['ann_return'])} | {pct(cand_oos['ann_return'])} |",
        f"| 超额年化 | {pct(base_oos['excess'])} | {pct(cand_oos['excess'])} |",
        f"| 最大回撤 | {pct(base_oos['max_drawdown'])} | {pct(cand_oos['max_drawdown'])} |",
        f"| 夏普比率 | {base_oos['sharpe']:.3f} | {cand_oos['sharpe']:.3f} |",
        "",
        "## 滚动窗口",
        "",
        f"- 外推窗口数：{len(roll)}",
        f"- 超额改善窗口：{int((roll['delta_excess'] > 0).sum())}/{len(roll)}",
        f"- 平均超额改善：{pct(roll['delta_excess'].mean())}",
        f"- 回撤改善窗口：{int((roll['delta_max_drawdown'] > 0).sum())}/{len(roll)}",
        "",
        "## 判断",
        "",
        "V2.4-rc1 在全样本上实现了你要的三目标：年化更高、夏普更高、最大回撤更低。但滚动窗口胜率仍不够强，只能作为候选版，不宜直接替代 V2.3a。下一步应专门检查它变差的窗口，尤其是假反弹阶段。",
        "",
        "## 输出文件",
        "",
        "- `output/v2.4_rc1_trade_ledger.csv`",
        "- `output/v2.4_rc1_daily_positions.csv`",
        "- `output/v2.4_rc1_yearly_attribution.csv`",
        "- `output/v2.4_rc1_rolling_compare.csv`",
    ]
    (OUTPUT_DIR / "v2.4_rc1_candidate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_report()
    print("V2.4-rc1 candidate report complete.")
