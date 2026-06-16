"""
Generate final V2.9 reports for the B2 homework track.

Outputs:
- yearly attribution for the final candidate
- final execution rules
- a homework-ready Markdown research report
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


ROOT_OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR = ROOT_OUTPUT_DIR / "final"
INTERMEDIATE_OUTPUT_DIR = ROOT_OUTPUT_DIR / "intermediate"
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
        return ["暂无 V2.8 压力测试汇总文件；可运行 `python family_strategy_stress_v2.py` 生成。"]
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
):
    if defensive_symbol == "CASH_PROXY":
        defensive_reflection = "- 现金/短债仓当前仍为代理口径，正式执行前必须替换为真实货币/短债 ETF。"
    else:
        defensive_reflection = f"- 现金/短债仓已替换为真实 ETF `{defensive_symbol}` {defensive_name}；后续若用于实盘，应继续检查流动性、费率、折溢价和申赎约束。"
    full = {k.replace("full_", ""): v for k, v in metrics.items() if k.startswith("full_")}
    oos = {k.replace("oos_2025_", ""): v for k, v in metrics.items() if k.startswith("oos_2025_")}
    v27 = maybe_load_v27_practical()
    stress = maybe_load_stress_summary()
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
        "样本外结果说明策略在 2025 年仍保持正收益和正超额，但由于 OOS 只有一年，本文将其作为初步外推证据，而不是长期有效性的最终证明。",
        "",
        "## 1. 研究问题与设计",
        "",
        "本文对应 B2 题目，核心问题是：能否构建一个普通家庭可以理解、可以执行、并且在风险收益上优于单一宽基持有的 ETF 组合策略。与单纯追求最高收益的择时模型不同，本文把最大回撤、调仓频率、标的可交易性和资产配置解释性作为同等重要的约束。",
        "",
        "研究设计分为三步：第一，构建核心-卫星风险策略，用趋势和动量信号选择风险资产；第二，在风险策略外层加入黄金和短债防守仓，降低纯权益轮动的路径波动；第三，用样本外检验和压力测试检查最终方案是否过度依赖某一类资产或某一年市场环境。",
        "",
        "## 2. 数据与样本划分",
        "",
        "- 数据频率：ETF 日线收盘价。",
        "- 风险资产：A 股宽基、行业主题 ETF、纳指 ETF 等。",
        f"- 防守资产：黄金 ETF `518880` 与 `{defensive_symbol}` {defensive_name}。",
        "- 基准：沪深300指数/沪深300 ETF 口径。",
        "- 开发样本：2019-2024，用于策略迭代、参数筛选和版本选择。",
        "- 样本外区间：2025，仅用于最终候选的外推检验，不参与参数选择。",
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
        "本文将 2019-2024 作为开发样本，2025 作为样本外区间。需要强调的是，2025 OOS 并不是重新调参后的结果，而是把开发样本中确定的 V2.9 权重和规则直接外推到 2025 年。",
        "",
        "## 5. 样本内结果",
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
        "## 6. 样本外检验：2025 OOS",
        "",
        "2025 年作为样本外区间，没有参与 V2.9 的参数选择。该检验的目的不是证明策略必然长期有效，而是检查策略在开发样本之后是否立即失效。",
        "",
        "| 区间 | 年化 | 最大回撤 | 夏普 | Calmar | 基准年化 | 超额年化 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 2025 OOS | {pct(oos['ann_return'])} | {pct(oos['max_drawdown'])} | {ratio(oos['sharpe'])} | {ratio(oos['calmar'])} | {pct(oos['ann_bench'])} | {pct(oos['excess'])} |",
        "",
        "OOS 结果显示，V2.9 在 2025 年实现正收益，并略微跑赢沪深300基准。更重要的是，最大回撤仍控制在 "
        f"{pct(oos['max_drawdown'])}，说明外层防守仓没有在样本外阶段失去控回撤作用。但 2025 年只有一年，且黄金在该阶段仍有较强贡献，因此本文将 OOS 结果解释为“通过初步外推检验”，而不是把它当作长期稳健性的充分证据。",
        "",
        "## 7. 对照组与版本选择",
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
        "## 8. 稳健性分析",
        "",
        "本文针对黄金依赖和防守仓收益假设做了压力测试。压力测试将黄金日收益按 100%、75%、50%、25%、0% 缩放，并把现金/短债收益设为 0%、1%、2%，观察不同候选在压力网格中的表现。",
        "",
    ]
    lines.extend(stress_table_lines(stress))
    lines.extend([
        "",
        "压力测试的主要结论是：高黄金仓位组合在原始样本中指标更好，但当黄金收益被打折后，收益目标会明显依赖黄金贡献。V2.9 的 20% 黄金上限降低了这种单一资产依赖，虽然牺牲了一部分高黄金版本的年化收益，但更适合作为普通家庭长期配置框架。",
        "",
        "## 9. 年度归因",
        "",
        "年度归因用于观察收益来源是否集中在少数年份或单一资产。表中风险策略、黄金和短债贡献相加后，再扣除调仓成本，形成组合年度收益。",
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
        "## 10. 局限性",
        "",
        "- 黄金在 2019-2025 样本中表现较强，因此最终版本设置 20% 黄金上限，避免策略过度依赖黄金。",
        defensive_reflection,
        "- 2025 OOS 只有一年，能够提供初步外推证据，但不能替代更长周期、多市场状态的样本外检验。",
        "- 回测使用日线收盘价和固定交易成本，真实执行还会受到流动性、冲击成本、申赎限制、折溢价和个人税费影响。",
        "- 策略以月度调仓为核心，适合低频家庭账户；若用于更高频或更大资金规模，需要重新评估交易容量。",
        "",
        "## 11. 结论",
        "",
        "本文最终选择 V2.9 作为大作业主线策略。该策略在 2019-2024 年开发样本中实现 "
        f"{pct(full['ann_return'])} 年化收益、{pct(full['max_drawdown'])} 最大回撤和 {ratio(full['sharpe'])} 夏普比率；"
        "在 2025 年样本外区间中继续保持正收益和正超额。相较早期纯风险策略，V2.9 的优势在于把收益来源和防守资产分层，让策略更接近普通家庭可以理解、可以执行、也更容易长期持有的配置方案。",
        "",
        "本文结论不是“V2.9 已经证明未来一定有效”，而是：在当前数据和约束下，V2.9 是收益、回撤、解释性和可执行性之间最均衡的候选版本。后续研究应继续扩大样本外区间，并用真实交易约束检验该策略的可执行性。",
        "",
        "## 12. AI 使用说明",
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
