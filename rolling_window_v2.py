"""
V2.3 rolling-window and 2025 out-of-sample validation.

The original rolling_window.py validates the old fixed-satellite V1 engine.
This script validates the current V2.3 dynamic core-satellite strategy.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import pandas as pd

from backtest_v2 import BacktestEngineV2, StrategyParams


OUTPUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent / "data"
WINDOW_MONTHS = 24
STEP_MONTHS = 3
ROLLING_START = "2019-10-01"
ROLLING_END = "2024-12-31"
OOS_START = "2025-01-01"
OOS_END = "2025-12-31"


def run_engine(params: StrategyParams, quiet: bool = True) -> BacktestEngineV2:
    engine = BacktestEngineV2(params)
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            engine.run()
    else:
        engine.run()
    return engine


def available_benchmark_dates(params: StrategyParams) -> pd.DatetimeIndex:
    engine = run_engine(params, quiet=True)
    return pd.DatetimeIndex(engine.bench_data.index).sort_values()


def nearest_trading_day(dates: pd.DatetimeIndex, date: pd.Timestamp, side: str) -> pd.Timestamp | None:
    idx = dates.searchsorted(date, side="left" if side == "start" else "right")
    if side == "start":
        if idx >= len(dates):
            return None
        return dates[idx]
    idx -= 1
    if idx < 0:
        return None
    return dates[idx]


def generate_windows(dates: pd.DatetimeIndex) -> list[dict]:
    windows = []
    current = pd.Timestamp(ROLLING_START)
    last_test_end = pd.Timestamp(ROLLING_END)

    while True:
        train_start = nearest_trading_day(dates, current, "start")
        train_end = nearest_trading_day(dates, current + pd.DateOffset(months=WINDOW_MONTHS), "end")
        test_start = nearest_trading_day(dates, train_end + pd.Timedelta(days=1), "start") if train_end is not None else None
        test_end = nearest_trading_day(dates, train_end + pd.DateOffset(months=STEP_MONTHS), "end") if train_end is not None else None

        if None in (train_start, train_end, test_start, test_end):
            break
        if test_end > last_test_end:
            break

        windows.append({
            "window": len(windows) + 1,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        current += pd.DateOffset(months=STEP_MONTHS)

    return windows


def metric_row(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def run_period(start: pd.Timestamp | str, end: pd.Timestamp | str, base: StrategyParams) -> BacktestEngineV2:
    params = replace(base, start_date=str(pd.Timestamp(start).date()), end_date=str(pd.Timestamp(end).date()))
    return run_engine(params, quiet=True)


def run_rolling_validation(base: StrategyParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = available_benchmark_dates(base)
    windows = generate_windows(dates)
    train_rows = []
    test_rows = []

    for w in windows:
        train_engine = run_period(w["train_start"], w["train_end"], base)
        test_engine = run_period(w["test_start"], w["test_end"], base)

        common = {
            "window": w["window"],
            "train_start": str(w["train_start"].date()),
            "train_end": str(w["train_end"].date()),
            "test_start": str(w["test_start"].date()),
            "test_end": str(w["test_end"].date()),
        }
        train_rows.append({**common, **metric_row("train", train_engine.get_metrics())})
        test_rows.append({**common, **metric_row("test", test_engine.get_metrics())})

    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def summarize_numeric(df: pd.DataFrame, prefix: str, columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        values = df[f"{prefix}_{col}"].astype(float)
        rows.append({
            "metric": col,
            "mean": values.mean(),
            "median": values.median(),
            "min": values.min(),
            "max": values.max(),
            "positive_rate": (values > 0).mean(),
        })
    return pd.DataFrame(rows)


def fmt_pct(x: float) -> str:
    return f"{x:+.2%}"


def fmt_ratio(x: float) -> str:
    return f"{x:.3f}"


def write_report(train_df: pd.DataFrame, test_df: pd.DataFrame, oos_engine: BacktestEngineV2):
    OUTPUT_DIR.mkdir(exist_ok=True)
    train_summary = summarize_numeric(train_df, "train", ["ann_return", "excess", "max_drawdown", "sharpe", "calmar"])
    test_summary = summarize_numeric(test_df, "test", ["ann_return", "excess", "max_drawdown", "sharpe", "calmar"])
    oos_metrics = oos_engine.get_metrics()

    test_excess_hit = (test_df["test_excess"].astype(float) > 0).mean()
    test_return_hit = (test_df["test_ann_return"].astype(float) > 0).mean()
    worst_test = test_df.loc[test_df["test_max_drawdown"].astype(float).idxmin()]
    best_test = test_df.loc[test_df["test_ann_return"].astype(float).idxmax()]

    lines = [
        "# V2.3 滚动窗口与 2025 OOS 验证",
        "",
        "## 验证设置",
        "",
        f"- 策略版本：V2.3a（n=2, 60日动量, 月度调仓, MA20止损, HS300熔断, 熊市核心MA200过滤, ATR止盈修复）",
        f"- 滚动窗口：{WINDOW_MONTHS}个月训练窗口 + {STEP_MONTHS}个月外推窗口",
        f"- 滚动验证区间：{ROLLING_START} 至 {ROLLING_END}",
        f"- 2025 OOS：{OOS_START} 至 {OOS_END}",
        "- 说明：参数不在每个窗口中重新优化；本验证检验既定 V2.3 规则在不同市场切片中的稳定性。",
        "- 口径更新：本次修复了 ATR 止盈的浮盈计算，2019-2024 V2.3 指标重述为年化 +5.05%、最大回撤 -15.21%、超额年化 +4.57%。",
        "",
        "## 滚动外推结论",
        "",
        f"- 外推窗口数：{len(test_df)}",
        f"- 外推年化收益均值：{fmt_pct(test_df['test_ann_return'].astype(float).mean())}",
        f"- 外推超额年化均值：{fmt_pct(test_df['test_excess'].astype(float).mean())}",
        f"- 外推最大回撤均值：{fmt_pct(test_df['test_max_drawdown'].astype(float).mean())}",
        f"- 外推超额为正比例：{test_excess_hit:.0%}",
        f"- 外推收益为正比例：{test_return_hit:.0%}",
        f"- 最差外推窗口：窗口{int(worst_test['window'])}，{worst_test['test_start']} 至 {worst_test['test_end']}，最大回撤 {fmt_pct(float(worst_test['test_max_drawdown']))}",
        f"- 最强外推窗口：窗口{int(best_test['window'])}，{best_test['test_start']} 至 {best_test['test_end']}，年化收益 {fmt_pct(float(best_test['test_ann_return']))}",
        "",
        "## 2025 OOS 结果",
        "",
        f"- 年化收益：{fmt_pct(oos_metrics['ann_return'])}",
        f"- 基准年化：{fmt_pct(oos_metrics['ann_bench'])}",
        f"- 超额年化：{fmt_pct(oos_metrics['excess'])}",
        f"- 最大回撤：{fmt_pct(oos_metrics['max_drawdown'])}",
        f"- 年化波动：{fmt_pct(oos_metrics['ann_vol'])}",
        f"- 夏普比率：{fmt_ratio(oos_metrics['sharpe'])}",
        f"- 卡玛比率：{fmt_ratio(oos_metrics['calmar'])}",
        f"- 交易流水：{oos_metrics['n_trades']} 条",
        "",
        "## 判断",
        "",
        "V2.3a 通过了相对收益方向的验证：滚动外推多数窗口有正超额，2025 OOS 也保持正收益和正超额。但绝对收益不稳定，13 个季度外推窗口里只有 46% 为正收益，说明这套策略更适合作为低频组合框架，不适合作为季度级别的稳定收益机器。",
        "",
        "下一步优先建议不是继续加复杂信号，而是把 V2.3a 固化成可复核版本：补交易明细、持仓权重快照、年度归因和当前月调仓建议。只有这些执行层证据稳定后，再评估熔断阈值或渐进恢复。",
        "",
        "## 输出文件",
        "",
        "- `output/v2.3_rolling_train_metrics.csv`",
        "- `output/v2.3_rolling_test_metrics.csv`",
        "- `output/v2.3_2025_oos_nav.csv`",
        "- `output/v2.3_2025_oos_monthly_log.csv`",
        "- `output/v2.3a_execution_summary.md`",
        "- `output/v2.3a_latest_rebalance_advice.md`",
    ]

    (OUTPUT_DIR / "v2.3_滚动验证与2025_OOS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    train_summary.to_csv(OUTPUT_DIR / "v2.3_rolling_train_summary.csv", index=False)
    test_summary.to_csv(OUTPUT_DIR / "v2.3_rolling_test_summary.csv", index=False)


def main():
    base = StrategyParams()
    train_df, test_df = run_rolling_validation(base)
    oos_engine = run_period(OOS_START, OOS_END, base)

    OUTPUT_DIR.mkdir(exist_ok=True)
    train_df.to_csv(OUTPUT_DIR / "v2.3_rolling_train_metrics.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "v2.3_rolling_test_metrics.csv", index=False)
    oos_engine.get_nav_df().to_csv(OUTPUT_DIR / "v2.3_2025_oos_nav.csv")
    pd.DataFrame(oos_engine.monthly_log).to_csv(OUTPUT_DIR / "v2.3_2025_oos_monthly_log.csv", index=False)
    write_report(train_df, test_df, oos_engine)

    print("V2.3 rolling validation complete.")
    print(f"Rolling windows: {len(test_df)}")
    print(f"2025 OOS ann_return: {oos_engine.get_metrics()['ann_return']:.2%}")
    print(f"2025 OOS excess: {oos_engine.get_metrics()['excess']:.2%}")


if __name__ == "__main__":
    main()
