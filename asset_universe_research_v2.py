"""
Asset-universe research for V2 core-satellite strategy.

Question: is the low annualized return mainly caused by the chosen core ETFs?

This script keeps the strategy mechanics stable and tests several alternative
core pools. It deliberately removes any core ETF from the satellite pool to
avoid duplicate target-weight overwrites during rebalancing.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import replace
from pathlib import Path

import pandas as pd

from backtest_v2 import BacktestEngineV2, StrategyParams
from rolling_window_v2 import available_benchmark_dates, generate_windows
from variant_research_v2 import CoreMom20RecoveryEngine


OUTPUT_DIR = Path(__file__).parent / "output"

ETF_NAMES = {
    "510300": "沪深300",
    "510500": "中证500",
    "159915": "创业板",
    "512890": "红利低波",
    "513100": "纳指",
    "518880": "黄金",
    "512800": "银行",
    "512760": "半导体",
    "512690": "酒",
}


def pct(value: float) -> str:
    return f"{value:+.2%}"


def ratio(value: float) -> str:
    return f"{value:.3f}"


def core_label(core: tuple[str, ...]) -> str:
    return " / ".join(f"{sym}{ETF_NAMES.get(sym, '')}" for sym in core)


def params_with_core(base: StrategyParams, core: tuple[str, ...]) -> StrategyParams:
    sat_pool = tuple(sym for sym in base.satellite_pool if sym not in set(core))
    return replace(base, core_etfs=core, satellite_pool=sat_pool)


def run_engine(
    name: str,
    params: StrategyParams,
    start: str,
    end: str,
    engine_cls: type[BacktestEngineV2] = BacktestEngineV2,
) -> tuple[dict, pd.DataFrame]:
    engine = engine_cls(replace(params, start_date=start, end_date=end))
    with contextlib.redirect_stdout(io.StringIO()):
        engine.run()
    metrics = engine.get_metrics()
    metrics["variant"] = name
    metrics["start"] = start
    metrics["end"] = end
    return metrics, engine.get_nav_df()


def rolling_summary(
    name: str,
    params: StrategyParams,
    engine_cls: type[BacktestEngineV2] = BacktestEngineV2,
) -> dict:
    dates = available_benchmark_dates(params)
    rows = []
    for window in generate_windows(dates):
        metrics, _ = run_engine(
            name,
            params,
            str(window["test_start"].date()),
            str(window["test_end"].date()),
            engine_cls,
        )
        rows.append(metrics)
    df = pd.DataFrame(rows)
    return {
        "rolling_windows": len(df),
        "rolling_ann_mean": df["ann_return"].mean(),
        "rolling_excess_mean": df["excess"].mean(),
        "rolling_drawdown_mean": df["max_drawdown"].mean(),
        "rolling_sharpe_mean": df["sharpe"].mean(),
        "rolling_positive_excess_rate": (df["excess"] > 0).mean(),
        "rolling_positive_return_rate": (df["ann_return"] > 0).mean(),
    }


def build_variants() -> list[dict]:
    base = StrategyParams()
    variants = [
        {
            "variant": "v2.3a_current_core",
            "engine": BacktestEngineV2,
            "core": ("510300", "510500", "159915", "512890"),
            "reason": "当前核心池：A股宽基 + 红利低波。",
        },
        {
            "variant": "drop_cyb_add_nasdaq_gold",
            "engine": BacktestEngineV2,
            "core": ("510300", "510500", "512890", "513100", "518880"),
            "reason": "降低创业板核心权重，把纳指和黄金从卫星升为结构性核心。",
        },
        {
            "variant": "barbell_core",
            "engine": BacktestEngineV2,
            "core": ("510300", "512890", "513100", "518880"),
            "reason": "杠铃核心：沪深300代表A股 beta，红利低波/纳指/黄金提供质量、海外成长和防守。",
        },
        {
            "variant": "defensive_global_core",
            "engine": BacktestEngineV2,
            "core": ("512890", "512800", "513100", "518880"),
            "reason": "偏防守全球核心：红利低波、银行、纳指、黄金。",
        },
        {
            "variant": "growth_global_core",
            "engine": BacktestEngineV2,
            "core": ("159915", "512760", "513100", "518880"),
            "reason": "偏进攻核心：创业板、半导体、纳指、黄金，检验年化上限与回撤代价。",
        },
        {
            "variant": "v2.4rc1_current_core",
            "engine": CoreMom20RecoveryEngine,
            "params_override": {"equity_bear": 0.55, "circuit_breaker_cooldown": 10},
            "core": ("510300", "510500", "159915", "512890"),
            "reason": "当前 V2.4-rc1 规则候选，作为规则改进基线。",
        },
        {
            "variant": "v2.4rc1_barbell_core",
            "engine": CoreMom20RecoveryEngine,
            "params_override": {"equity_bear": 0.55, "circuit_breaker_cooldown": 10},
            "core": ("510300", "512890", "513100", "518880"),
            "reason": "把 V2.4-rc1 的恢复规则与杠铃核心组合起来，检验是否叠加有效。",
        },
    ]
    result = []
    for item in variants:
        params = params_with_core(base, item["core"])
        if item.get("params_override"):
            params = replace(params, **item["params_override"])
        result.append({**item, "params": params})
    return result


def evaluate() -> pd.DataFrame:
    rows = []
    for item in build_variants():
        full, _ = run_engine(
            item["variant"],
            item["params"],
            "2019-10-01",
            "2024-12-31",
            item["engine"],
        )
        oos, _ = run_engine(
            item["variant"],
            item["params"],
            "2025-01-01",
            "2025-12-31",
            item["engine"],
        )
        rolling = rolling_summary(item["variant"], item["params"], item["engine"])
        rows.append({
            "variant": item["variant"],
            "core": ",".join(item["core"]),
            "core_desc": core_label(item["core"]),
            "reason": item["reason"],
            "engine": item["engine"].__name__,
            "ann_return": full["ann_return"],
            "excess": full["excess"],
            "max_drawdown": full["max_drawdown"],
            "ann_vol": full["ann_vol"],
            "sharpe": full["sharpe"],
            "calmar": full["calmar"],
            "n_trades": full["n_trades"],
            "rolling_ann_mean": rolling["rolling_ann_mean"],
            "rolling_excess_mean": rolling["rolling_excess_mean"],
            "rolling_drawdown_mean": rolling["rolling_drawdown_mean"],
            "rolling_sharpe_mean": rolling["rolling_sharpe_mean"],
            "rolling_positive_excess_rate": rolling["rolling_positive_excess_rate"],
            "rolling_positive_return_rate": rolling["rolling_positive_return_rate"],
            "oos_2025_ann_return": oos["ann_return"],
            "oos_2025_excess": oos["excess"],
            "oos_2025_max_drawdown": oos["max_drawdown"],
            "oos_2025_sharpe": oos["sharpe"],
            "oos_2025_trades": oos["n_trades"],
        })
    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame):
    base = df[df["variant"] == "v2.3a_current_core"].iloc[0]
    v24 = df[df["variant"] == "v2.4rc1_current_core"].iloc[0]
    best_full = df.sort_values(["sharpe", "ann_return"], ascending=[False, False]).iloc[0]
    best_rolling = df.sort_values(["rolling_excess_mean", "rolling_sharpe_mean"], ascending=[False, False]).iloc[0]

    lines = [
        "# V2 标的池诊断：年化是否被核心标的拖累",
        "",
        "## 结论先行",
        "",
        "年化收益偏低，确实有一部分来自标的结构，而不只是择时规则问题。当前核心池里，沪深300、中证500、创业板在 2019-2024 的长期收益/回撤比并不强；而历史表现更好的纳指、黄金、酒、半导体大多被放在卫星池，实际平均仓位很低。",
        "",
        "但这不等于应该把酒、半导体这类高波动行业直接升成大核心。它们能抬高年化，也会显著抬高回撤和路径风险。更稳的方向，是把纳指和黄金这类驱动更分散的资产从卫星升为核心，再保留红利低波作为A股防守锚。",
        "",
        "## 核心池实验结果",
        "",
        "| 变体 | 核心池 | 年化 | 超额 | 最大回撤 | 夏普 | 卡玛 | 滚动超额均值 | 2025年化 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.sort_values(["ann_return"], ascending=False).iterrows():
        lines.append(
            f"| {row['variant']} | {row['core_desc']} | {pct(row['ann_return'])} | "
            f"{pct(row['excess'])} | {pct(row['max_drawdown'])} | {ratio(row['sharpe'])} | "
            f"{ratio(row['calmar'])} | {pct(row['rolling_excess_mean'])} | "
            f"{pct(row['oos_2025_ann_return'])} |"
        )

    lines.extend([
        "",
        "## 和当前版本对比",
        "",
        f"- 当前 V2.3a：年化 {pct(base['ann_return'])}，最大回撤 {pct(base['max_drawdown'])}，夏普 {ratio(base['sharpe'])}。",
        f"- 当前 V2.4-rc1 规则候选：年化 {pct(v24['ann_return'])}，最大回撤 {pct(v24['max_drawdown'])}，夏普 {ratio(v24['sharpe'])}。",
        f"- 全样本夏普最高：`{best_full['variant']}`，年化 {pct(best_full['ann_return'])}，回撤 {pct(best_full['max_drawdown'])}，夏普 {ratio(best_full['sharpe'])}。",
        f"- 滚动外推超额最好：`{best_rolling['variant']}`，滚动超额均值 {pct(best_rolling['rolling_excess_mean'])}，滚动夏普均值 {ratio(best_rolling['rolling_sharpe_mean'])}。",
        "",
        "## 我的判断",
        "",
        "低年化不是单纯因为参数保守；资产摆放本身也压低了上限。当前组合把较多长期核心仓交给 A 股宽基和创业板，而它们在样本期里承担了不小回撤，却没有贡献足够年化。与此同时，纳指和黄金明明在池子里，却主要作为卫星出现，被 MA200、动量排序、止损和熔断反复限制，长期仓位太小。",
        "",
        "下一步我更倾向先把 `drop_cyb_add_nasdaq_gold` 作为保守候选，再把 `barbell_core` 作为进取候选继续打磨。前者年化提升不如杠铃核心激进，但回撤和夏普最漂亮；后者年化更高，但需要接受更深回撤。两者共同指向同一个机制：把纳指和黄金从短期卫星变成结构性核心，比继续加创业板/半导体/酒更有机会同时改善年化、夏普和回撤。",
        "",
        "## 风险",
        "",
        "- 纳指和黄金的历史表现不能保证未来延续，尤其纳指可能带来汇率、估值和海外市场集中风险。",
        "- 黄金升为核心会降低某些A股强反弹阶段的进攻性。",
        "- 如果核心池包含原卫星资产，必须像本实验一样从卫星池移除，避免调仓时同一标的被核心和卫星双重赋权。",
        "",
        "## 输出文件",
        "",
        "- `output/v2.5_asset_universe_research.csv`",
        "- `output/v2.5_asset_universe_research.md`",
    ])

    (OUTPUT_DIR / "v2.5_asset_universe_research.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = evaluate()
    df.to_csv(OUTPUT_DIR / "v2.5_asset_universe_research.csv", index=False)
    write_report(df)
    print("Asset-universe research complete.")
    print(df.sort_values(["ann_return"], ascending=False)[[
        "variant", "ann_return", "max_drawdown", "sharpe", "rolling_excess_mean", "oos_2025_ann_return"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
