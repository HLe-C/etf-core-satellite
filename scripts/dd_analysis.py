"""
V2.1 回撤归因分析 — 定位最大回撤的根源
"""
import pandas as pd
import numpy as np
from backtest_v2 import StrategyParams, BacktestEngineV2
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


class DetailedBacktest(BacktestEngineV2):
    """扩展回测引擎，记录每只持仓的每日明细"""

    def __init__(self, params):
        super().__init__(params)
        self.daily_positions: list = []  # 每日每只ETF的市值

    def run(self):
        self.prepare()
        last_period = None

        for i, date in enumerate(self.trade_dates):
            # ---- 快速估算当前总资产 ----
            est_total = self.cash
            for sym, pos in self.positions.items():
                close = self._get_close(sym, date)
                if close is not None and pos.shares > 0:
                    est_total += pos.shares * close

            # ---- 每日风控检查（MA20止损 + HS300熔断）----
            self._check_daily_risk(date, est_total, i)

            # 风控后重新计算净值
            total_value = self.cash
            equity_value = 0.0
            daily_pos = {"date": date}
            for sym, pos in self.positions.items():
                close = self._get_close(sym, date)
                if close is not None and pos.shares > 0:
                    pos_value = pos.shares * close
                    total_value += pos_value
                    equity_value += pos_value
                    daily_pos[sym] = pos_value
                else:
                    daily_pos[sym] = 0.0

            bench_val = self.bench_data.loc[date, "benchmark"] if date in self.bench_data.index else None

            self.nav_history.append({
                "date": date,
                "nav": total_value,
                "equity": equity_value,
                "cash": self.cash,
                "bench": bench_val,
            })

            daily_pos["total_nav"] = total_value
            self.daily_positions.append(daily_pos)

            # 月度调仓
            current_month, current_year = date.month, date.year
            month_key = (current_year, current_month)

            do_rebalance = (month_key != last_period)
            if do_rebalance and i > 0:
                last_period = month_key
                self._monthly_rebalance(date, total_value)

            if (i + 1) % 500 == 0:
                print(f"  进度: {i+1}/{len(self.trade_dates)}")


def analyze_drawdown():
    print("=" * 60)
    print("V2.1 回撤归因分析  [n=2, mom=120, monthly]")
    print("=" * 60)

    params = StrategyParams(n_satellites=2, mom_rank=120, rebalance_freq="monthly")
    engine = DetailedBacktest(params)
    engine.run()

    # 获取净值序列
    nav_df = pd.DataFrame(engine.nav_history)
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    nav_df = nav_df.set_index("date").sort_index()

    # 计算回撤
    nav_df["cummax"] = nav_df["nav"].cummax()
    nav_df["drawdown"] = nav_df["nav"] / nav_df["cummax"] - 1

    # 找最大回撤
    dd_max = nav_df["drawdown"].min()
    dd_end_idx = nav_df["drawdown"].idxmin()
    dd_valley = nav_df.loc[dd_end_idx, "nav"]

    # 回撤起点：从valley往回找cummax
    peak_val = nav_df.loc[:dd_end_idx, "nav"].max()
    dd_start_idx = nav_df.loc[:dd_end_idx][nav_df["nav"] == peak_val].index[0]

    print(f"\n最大回撤: {dd_max:.2%}")
    print(f"回撤起点: {dd_start_idx.date()}  (NAV={peak_val:,.0f})")
    print(f"回撤终点: {dd_end_idx.date()}  (NAV={dd_valley:,.0f})")
    print(f"回撤区间: {(dd_end_idx - dd_start_idx).days} 天")

    # 分析回撤区间内的持仓构成
    pos_df = pd.DataFrame(engine.daily_positions)
    pos_df["date"] = pd.to_datetime(pos_df["date"])
    pos_df = pos_df.set_index("date").sort_index()

    dd_pos = pos_df.loc[dd_start_idx:dd_end_idx]

    # 区间起止的各持仓权重
    print(f"\n{'='*60}")
    print("回撤区间起点持仓:")
    print("-" * 60)
    start_row = pos_df.loc[dd_start_idx]
    total = start_row["total_nav"]
    for sym in list(engine.p.core_etfs) + list(engine.p.satellite_pool):
        v = start_row.get(sym, 0)
        if v > 0:
            print(f"  {sym}: {v:>10,.0f}  ({v/total*100:5.1f}%)")

    print(f"\n回撤区间终点持仓:")
    print("-" * 60)
    end_row = pos_df.loc[dd_end_idx]
    total_e = end_row["total_nav"]
    for sym in list(engine.p.core_etfs) + list(engine.p.satellite_pool):
        v = end_row.get(sym, 0)
        if v > 0:
            print(f"  {sym}: {v:>10,.0f}  ({v/total_e*100:5.1f}%)")

    # 区间内各ETF的涨跌幅贡献拆解
    print(f"\n{'='*60}")
    print("回撤区间内各ETF涨跌幅:")
    print("-" * 60)
    total_loss = peak_val - dd_valley
    print(f"总亏损: {total_loss:,.0f}")

    for sym in list(engine.p.core_etfs) + list(engine.p.satellite_pool):
        start_v = start_row.get(sym, 0)
        if start_v <= 0:
            continue
        end_v = end_row.get(sym, 0)
        change = end_v - start_v
        if abs(change) > 100:
            print(f"  {sym}: {start_v:>10,.0f} → {end_v:>9,.0f}  ({change:+,.0f}, {change/start_v*100:+.1f}%)")

    # 现金变化
    cash_start = start_row["total_nav"] - sum(start_row.get(s, 0) for s in list(engine.p.core_etfs) + list(engine.p.satellite_pool))
    cash_end = end_row["total_nav"] - sum(end_row.get(s, 0) for s in list(engine.p.core_etfs) + list(engine.p.satellite_pool))
    print(f"  CASH: {cash_start:>10,.0f} → {cash_end:>9,.0f}  ({cash_end-cash_start:+,.0f})")

    # 回撤区间内的月度调仓记录
    print(f"\n{'='*60}")
    print("回撤区间内的月度调仓:")
    print("-" * 60)
    monthly_df = pd.DataFrame(engine.monthly_log)
    monthly_df["date"] = pd.to_datetime(monthly_df["date"])
    dd_monthly = monthly_df[(monthly_df["date"] >= dd_start_idx) & (monthly_df["date"] <= dd_end_idx)]
    for _, row in dd_monthly.iterrows():
        print(f"  [{row['date'].date()}] state={row['market_state']}, "
              f"equity={row['equity_target']:.0%}, "
              f"sats={row['selected_satellites']}")

    # 额外：回撤期间市场状态
    print(f"\n{'='*60}")
    print("回撤期间市场状态统计:")
    print("-" * 60)
    # 逐日检查市场状态
    dd_states = []
    for d in nav_df.loc[dd_start_idx:dd_end_idx].index:
        state = engine._get_market_state(d)
        dd_states.append({"date": d, "state": state})

    state_df = pd.DataFrame(dd_states)
    print(state_df["state"].value_counts().to_string())

    # 找出所有较大回撤（超过15%）
    print(f"\n{'='*60}")
    print("所有较大回撤事件 (>15%):")
    print("-" * 60)
    nav_df["dd_phase"] = 0
    phase = 0
    in_dd = False
    for i in range(len(nav_df)):
        if nav_df["nav"].iloc[i] < nav_df["cummax"].iloc[i]:
            if not in_dd:
                phase += 1
                in_dd = True
            nav_df.iloc[i, nav_df.columns.get_loc("dd_phase")] = phase
        else:
            in_dd = False

    for ph in nav_df["dd_phase"].unique():
        if ph == 0:
            continue
        ph_data = nav_df[nav_df["dd_phase"] == ph]
        ph_dd = ph_data["drawdown"].min()
        if ph_dd < -0.15:
            ph_start = ph_data.index[0]
            ph_end = ph_data["drawdown"].idxmin()
            print(f"  阶段{ph}: {ph_start.date()} → {ph_end.date()}, 最大回撤 {ph_dd:.1%}")

    # 保存
    OUTPUT_DIR.mkdir(exist_ok=True)
    nav_df.to_csv(OUTPUT_DIR / "v2_dd_analysis.csv")
    pos_df.to_csv(OUTPUT_DIR / "v2_daily_positions.csv")
    monthly_df.to_csv(OUTPUT_DIR / "v2_monthly_log_detailed.csv", index=False)

    print(f"\n详细数据已保存到 output/")


if __name__ == "__main__":
    analyze_drawdown()
