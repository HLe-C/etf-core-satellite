"""
Generate execution evidence for the V2.3a strategy.

Outputs:
  - complete trade ledger
  - daily position weights
  - calendar-year attribution
  - latest rebalance advice based on the newest local data
"""
from __future__ import annotations

import ast
import contextlib
import io
from pathlib import Path

import pandas as pd

from backtest_v2 import BacktestEngineV2, StrategyParams


OUTPUT_DIR = Path(__file__).parent / "output"
BACKTEST_START = "2019-10-01"
BACKTEST_END = "2024-12-31"
FULL_END = "2025-12-31"

ETF_NAMES = {
    "159915": "创业板",
    "159928": "消费",
    "510300": "沪深300",
    "510500": "中证500",
    "512010": "医药",
    "512200": "房地产",
    "512580": "环保",
    "512660": "军工",
    "512690": "酒",
    "512720": "计算机",
    "512760": "半导体",
    "512800": "银行",
    "512880": "证券",
    "512890": "红利低波",
    "512980": "传媒",
    "513100": "纳指",
    "515050": "5G通信",
    "518880": "黄金",
}


def run_engine(start: str, end: str) -> BacktestEngineV2:
    params = StrategyParams(start_date=start, end_date=end)
    engine = BacktestEngineV2(params)
    with contextlib.redirect_stdout(io.StringIO()):
        engine.run()
    return engine


def fmt_pct(value: float) -> str:
    return f"{value:+.2%}"


def normalize_trade_log(engine: BacktestEngineV2) -> pd.DataFrame:
    trades = pd.DataFrame(engine.trade_log)
    if trades.empty:
        return trades
    trades["date"] = pd.to_datetime(trades["date"])
    trades["name"] = trades["symbol"].map(ETF_NAMES).fillna(trades["symbol"])
    for col in ["price", "shares", "amount", "fee", "target_weight"]:
        if col not in trades.columns:
            trades[col] = pd.NA
    return trades[[
        "date", "symbol", "name", "action", "price", "shares",
        "amount", "fee", "target_weight", "reason",
    ]].sort_values(["date", "symbol", "action"])


def yearly_attribution(nav_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in nav_df.groupby(nav_df.index.year):
        if len(g) < 2:
            continue
        strategy_ret = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1
        bench_ret = g["bench_nav"].iloc[-1] / g["bench_nav"].iloc[0] - 1
        max_dd = (g["nav"] / g["nav"].cummax() - 1).min()
        rows.append({
            "year": year,
            "strategy_return": strategy_ret,
            "benchmark_return": bench_ret,
            "excess_return": strategy_ret - bench_ret,
            "max_drawdown": max_dd,
            "year_end_nav": g["nav"].iloc[-1],
        })
    return pd.DataFrame(rows)


def latest_target_weights(engine: BacktestEngineV2) -> dict[str, float]:
    monthly = pd.DataFrame(engine.monthly_log)
    if monthly.empty:
        return {}
    last = monthly.iloc[-1]
    active_core = ast.literal_eval(last["active_core"]) if isinstance(last.get("active_core"), str) else last.get("active_core", [])
    selected_satellites = (
        ast.literal_eval(last["selected_satellites"])
        if isinstance(last.get("selected_satellites"), str)
        else last.get("selected_satellites", [])
    )
    weights = {}
    if active_core:
        core_each = float(last["core_weight"]) / len(active_core)
        for sym in active_core:
            weights[sym] = core_each
    for sym in selected_satellites:
        weights[sym] = float(last["sat_weight"])
    return weights


def current_weights(position_df: pd.DataFrame, symbols: list[str]) -> dict[str, float]:
    last = position_df.iloc[-1]
    weights = {"cash": float(last.get("cash_weight", 0.0))}
    for sym in symbols:
        weights[sym] = float(last.get(f"{sym}_weight", 0.0))
    return weights


def write_rebalance_advice(engine: BacktestEngineV2, position_df: pd.DataFrame):
    latest_date = position_df.index[-1]
    monthly = pd.DataFrame(engine.monthly_log)
    latest_log = monthly.iloc[-1]
    symbols = list(engine.p.core_etfs) + list(engine.p.satellite_pool)
    target = latest_target_weights(engine)
    current = current_weights(position_df, symbols)

    rows = []
    for sym in symbols:
        target_w = target.get(sym, 0.0)
        current_w = current.get(sym, 0.0)
        diff = target_w - current_w
        if abs(diff) >= engine.p.deviation_threshold or target_w > 0 or current_w > 0:
            rows.append((sym, ETF_NAMES.get(sym, sym), current_w, target_w, diff))
    cash_target = 1 - sum(target.values())
    rows.append(("cash", "现金", current["cash"], cash_target, cash_target - current["cash"]))

    lines = [
        "# V2.3a 最新调仓建议",
        "",
        f"- 数据日期：{latest_date.date()}",
        f"- 最近一次策略调仓/风控日期：{pd.Timestamp(latest_log['date']).date()}",
        f"- 市场状态：{latest_log['market_state']}",
        f"- 目标权益仓位：{float(latest_log['equity_target']):.0%}",
        f"- 当前总资产口径：{position_df.iloc[-1]['total_value']:,.2f}",
        "",
        "## 目标权重",
        "",
        "| 代码 | 名称 | 当前权重 | 目标权重 | 差异 | 操作判断 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for sym, name, cur, tgt, diff in rows:
        action = "保持"
        if diff > engine.p.deviation_threshold:
            action = "买入/增配"
        elif diff < -engine.p.deviation_threshold:
            action = "卖出/减配"
        lines.append(f"| {sym} | {name} | {cur:.2%} | {tgt:.2%} | {diff:+.2%} | {action} |")

    lines.extend([
        "",
        "## 解释",
        "",
        "这份建议使用本地最新数据和 V2.3a 规则生成。若真实执行，需要用实际账户市值替换回测资产，并在交易日前重新确认数据已经更新。",
        "",
        "最近一次月度筛选摘要：",
        "",
    ])
    for item in latest_log.get("sat_reasons", []):
        lines.append(f"- {item}")

    (OUTPUT_DIR / "v2.3a_latest_rebalance_advice.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_execution_summary(engine: BacktestEngineV2, trades: pd.DataFrame, yearly: pd.DataFrame):
    metrics = engine.get_metrics()
    action_counts = trades["action"].value_counts().to_dict() if not trades.empty else {}
    best_year = yearly.loc[yearly["excess_return"].idxmax()]
    worst_year = yearly.loc[yearly["excess_return"].idxmin()]

    lines = [
        "# V2.3a 执行层证据包",
        "",
        "## 核心指标",
        "",
        f"- 区间：{BACKTEST_START} 至 {BACKTEST_END}",
        f"- 年化收益：{fmt_pct(metrics['ann_return'])}",
        f"- 基准年化：{fmt_pct(metrics['ann_bench'])}",
        f"- 超额年化：{fmt_pct(metrics['excess'])}",
        f"- 最大回撤：{fmt_pct(metrics['max_drawdown'])}",
        f"- 交易流水条数：{len(trades)}",
        f"- 交易类型：{action_counts}",
        "",
        "## 年度归因",
        "",
        f"- 最强相对年份：{int(best_year['year'])}，超额 {fmt_pct(best_year['excess_return'])}",
        f"- 最弱相对年份：{int(worst_year['year'])}，超额 {fmt_pct(worst_year['excess_return'])}",
        "",
        "| 年份 | 策略收益 | 基准收益 | 超额 | 最大回撤 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in yearly.iterrows():
        lines.append(
            f"| {int(row['year'])} | {fmt_pct(row['strategy_return'])} | "
            f"{fmt_pct(row['benchmark_return'])} | {fmt_pct(row['excess_return'])} | "
            f"{fmt_pct(row['max_drawdown'])} |"
        )

    lines.extend([
        "",
        "## 输出文件",
        "",
        "- `output/v2.3a_trade_ledger.csv`",
        "- `output/v2.3a_daily_positions.csv`",
        "- `output/v2.3a_yearly_attribution.csv`",
        "- `output/v2.3a_latest_rebalance_advice.md`",
    ])
    (OUTPUT_DIR / "v2.3a_execution_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    engine = run_engine(BACKTEST_START, BACKTEST_END)
    full_engine = run_engine(BACKTEST_START, FULL_END)

    trades = normalize_trade_log(engine)
    positions = engine.get_position_df()
    yearly = yearly_attribution(engine.get_nav_df())

    trades.to_csv(OUTPUT_DIR / "v2.3a_trade_ledger.csv", index=False)
    positions.to_csv(OUTPUT_DIR / "v2.3a_daily_positions.csv")
    yearly.to_csv(OUTPUT_DIR / "v2.3a_yearly_attribution.csv", index=False)

    write_rebalance_advice(full_engine, full_engine.get_position_df())
    write_execution_summary(engine, trades, yearly)

    print("V2.3a execution package complete.")
    print(f"Trade ledger rows: {len(trades)}")
    print(f"Daily position rows: {len(positions)}")
    print(f"Yearly attribution rows: {len(yearly)}")


if __name__ == "__main__":
    main()
