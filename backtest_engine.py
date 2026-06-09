"""
ETF 核心-卫星策略回测引擎

策略参数：
  核心(70%): 沪深300 ETF(50%) + 红利低波 ETF(20%)
  卫星(30%): 半导体 ETF(10%) + 5G通信 ETF(10%) + 计算机 ETF(10%)

风控机制：
  1. 偏离再平衡：任一仓位偏离目标 ±5% → 再平衡（冷却期20个交易日）
  2. 回撤止损：卫星组合回撤 > 20% → 卫星仓位减半至15%，转入红利ETF
  3. 回撤恢复：卫星组合反弹至前高 90% → 卫星仓位恢复至30%
  4. 波动率目标：月已实现波动率(年化) > 18% → 整体仓位打8折（现金20%）

交易成本：单边万分之一 (0.01%)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).parent / "data"

# ============================================================
# 策略常量
# ============================================================

TARGET_WEIGHTS = {
    "510300": 0.50,  # 核心-沪深300
    "512890": 0.20,  # 核心-红利低波
    "512760": 0.10,  # 卫星-半导体
    "515050": 0.10,  # 卫星-5G通信
    "512720": 0.10,  # 卫星-计算机
}

ETF_CODES = list(TARGET_WEIGHTS.keys())
CORE_ETFS = {"510300", "512890"}
SATELLITE_ETFS = {"512760", "515050", "512720"}

REBALANCE_THRESHOLD = 0.05       # 偏离阈值
REBALANCE_COOLDOWN = 20          # 再平衡冷却期（交易日）
DD_STOP_THRESHOLD = -0.20        # 卫星回撤止损线
DD_RECOVER_THRESHOLD = 0.90      # 恢复至前高90%
DD_COOLDOWN = 20                 # 止损后冷却期
VOL_TARGET = 0.18                # 年化波动率上限
VOL_CASH_RATIO = 0.20            # 超波动时减仓比例
VOL_COOLDOWN = 20                # 波动率控制冷却期
TRADING_COST = 0.0001            # 万分之一单边

BUILD_UP_MONTHS = 6
BUILD_UP_INTERVAL = 21           # 每21个交易日建仓一步

# ETF名称映射
ETF_NAMES = {
    "510300": "沪深300ETF",
    "512890": "红利低波ETF",
    "512760": "半导体ETF",
    "515050": "5G通信ETF",
    "512720": "计算机ETF",
}


# ============================================================
# 工具函数
# ============================================================

def _portfolio_value(holdings: dict, prices: dict, cash: float) -> float:
    return cash + sum(holdings[c] * prices[c] for c in ETF_CODES)


def _total_stock_value(holdings: dict, prices: dict) -> float:
    return sum(holdings[c] * prices[c] for c in ETF_CODES)


def _calc_weights(holdings: dict, prices: dict, cash: float) -> dict:
    tv = _portfolio_value(holdings, prices, cash)
    if tv <= 0:
        return {c: 0.0 for c in ETF_CODES}
    return {c: holdings[c] * prices[c] / tv for c in ETF_CODES}


def _get_target_weights(in_stop: bool) -> dict:
    """获取当前状态下的目标权重"""
    if not in_stop:
        return dict(TARGET_WEIGHTS)
    # 止损状态: 卫星各5%, 沪深300保持50%, 红利低波吸收转入=35%
    return {
        "510300": 0.50,
        "512890": 0.35,
        "512760": 0.05,
        "515050": 0.05,
        "512720": 0.05,
    }


def _check_deviation(current_weights: dict, target_weights: dict) -> bool:
    """检查是否有任一仓位偏离目标超过阈值"""
    for code in ETF_CODES:
        if abs(current_weights[code] - target_weights[code]) > REBALANCE_THRESHOLD:
            return True
    return False


# ============================================================
# 回测引擎
# ============================================================

@dataclass
class BacktestResult:
    nav: pd.Series
    daily_returns: pd.Series
    trades: pd.DataFrame
    weights_history: pd.DataFrame
    satellite_dd_history: pd.Series
    metrics: dict
    benchmark_nav: Optional[pd.Series] = None


def load_data() -> Dict[str, pd.Series]:
    etf_files = {
        "510300": "510300_沪深300.csv",
        "512890": "512890_红利低波.csv",
        "512760": "512760_半导体.csv",
        "515050": "515050_5G通信.csv",
        "512720": "512720_计算机.csv",
    }
    data = {}
    for code, fname in etf_files.items():
        fp = DATA_DIR / fname
        if fp.exists():
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            data[code] = df["close"]
    bp = DATA_DIR / "benchmark_000300.csv"
    if bp.exists():
        data["benchmark"] = pd.read_csv(bp, index_col=0, parse_dates=True)["close"]
    return data


def align_prices(data: Dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.DataFrame({c: data[c] for c in ETF_CODES})
    return df.dropna(how="any")


def calculate_metrics(nav: pd.Series, benchmark_nav: Optional[pd.Series] = None,
                      rf_annual: float = 0.02) -> dict:
    returns = nav.pct_change().dropna()
    td = 252
    years = len(returns) / td
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    ann_vol = returns.std() * np.sqrt(td)
    excess = returns - rf_annual / td
    sharpe = (excess.mean() / returns.std()) * np.sqrt(td) if returns.std() > 0 else 0
    cum = (1 + returns).cumprod()
    rolling_max = cum.expanding().max()
    dd = cum / rolling_max - 1
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (returns > 0).sum() / len(returns)
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0
    plr = avg_win / avg_loss if avg_loss > 0 else float("inf")

    metrics = {
        "累计收益": f"{total_ret:.2%}",
        "年化收益": f"{ann_ret:.2%}",
        "年化波动": f"{ann_vol:.2%}",
        "夏普比率": f"{sharpe:.3f}",
        "最大回撤": f"{max_dd:.2%}",
        "卡玛比率": f"{calmar:.3f}",
        "胜率": f"{win_rate:.2%}",
        "盈亏比": f"{plr:.2f}",
        "回测天数": len(returns),
    }

    if benchmark_nav is not None and len(benchmark_nav) > 1:
        br = benchmark_nav.pct_change().dropna()
        common = returns.index.intersection(br.index)
        if len(common) > 1:
            r = returns.loc[common]
            brc = br.loc[common]
            ex = r - brc
            te = ex.std() * np.sqrt(td)
            ir = (ex.mean() / ex.std()) * np.sqrt(td) if ex.std() > 0 else 0
            bt = benchmark_nav.loc[common[-1]] / benchmark_nav.loc[common[0]] - 1
            ba = (1 + bt) ** (1 / years) - 1 if years > 0 else 0
            metrics["超额年化"] = f"{ann_ret - ba:.2%}"
            metrics["跟踪误差"] = f"{te:.2%}"
            metrics["信息比率"] = f"{ir:.3f}"
            metrics["基准年化"] = f"{ba:.2%}"
    return metrics


def run_backtest(
    prices: pd.DataFrame,
    benchmark: Optional[pd.Series] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    verbose: bool = False,
) -> BacktestResult:
    if start_date:
        prices = prices.loc[prices.index >= start_date]
    if end_date:
        prices = prices.loc[prices.index <= end_date]
    prices = prices.dropna(how="any")

    n = len(prices)
    dates = prices.index

    # ---- 输出容器 ----
    nav = pd.Series(np.nan, index=dates, dtype=float)
    w_cols = ["cash"] + ETF_CODES
    weights_df = pd.DataFrame(0.0, index=dates, columns=w_cols)
    trades: list = []

    # ---- 初始状态 ----
    cash = 1.0
    holdings = {c: 0.0 for c in ETF_CODES}

    # ---- 风控状态 ----
    sat_hwm = 1.0            # 卫星参考指数历史最高
    in_stop = False
    sat_ref_base_prices = None
    sat_weights_init = np.array([TARGET_WEIGHTS[c] for c in SATELLITE_ETFS])
    sat_weights_init = sat_weights_init / sat_weights_init.sum()

    in_vol_control = False

    # ---- 冷却计时器 ----
    last_rebalance_day = -999
    last_dd_action_day = -999
    last_vol_action_day = -999

    def _sat_ref_nav(px_row) -> float:
        """计算卫星参考指数净值"""
        sp = np.array([px_row[c] for c in SATELLITE_ETFS])
        if sat_ref_base_prices is None:
            return 1.0
        return float((sp / sat_ref_base_prices * sat_weights_init).sum())

    def _do_trade(action, code, units, price, reason):
        """原子交易：返回(cash变化, holdings变化)"""
        nonlocal cash
        if action == "BUY":
            cost = units * price
            fee = cost * TRADING_COST
            if cost + fee <= cash:
                holdings[code] += units
                cash -= cost + fee
                trades.append({"date": dates[i], "code": code, "action": "BUY",
                               "price": price, "units": units, "reason": reason})
                return True
        else:  # SELL
            sellable = min(units, holdings[code])
            if sellable > 0:
                proceeds = sellable * price
                fee = proceeds * TRADING_COST
                holdings[code] -= sellable
                cash += proceeds - fee
                trades.append({"date": dates[i], "code": code, "action": "SELL",
                               "price": price, "units": sellable, "reason": reason})
                return True
        return False

    def _rebalance_to_target(target_w: dict, reason: str):
        """按目标权重再平衡"""
        tv = _portfolio_value(holdings, px, cash)
        if tv <= 0:
            return
        for code in ETF_CODES:
            current_w = holdings[code] * px[code] / tv
            target_val = tv * target_w[code]
            cur_val = holdings[code] * px[code]
            diff = target_val - cur_val
            if abs(diff) < tv * 0.005:  # 0.5%以下不交易
                continue
            if diff > 0:
                units = diff / px[code]
                _do_trade("BUY", code, units, px[code], reason)
            else:
                units = abs(diff) / px[code]
                _do_trade("SELL", code, units, px[code], reason)

    # ============================================================
    # 主循环
    # ============================================================
    for i in range(n):
        date = dates[i]
        px = {c: float(prices.iloc[i][c]) for c in ETF_CODES}

        # 初始化卫星参考基准价（第一个交易日）
        if sat_ref_base_prices is None:
            sat_ref_base_prices = np.array([px[c] for c in SATELLITE_ETFS])

        # ---- 建仓阶段 ----
        build_step_current = i // BUILD_UP_INTERVAL
        if build_step_current < BUILD_UP_MONTHS:
            if i == 0 or i % BUILD_UP_INTERVAL == 0:
                step_idx = build_step_current + 1  # 1~6
                target_invested_ratio = step_idx / BUILD_UP_MONTHS
                tv = _portfolio_value(holdings, px, cash)
                for code in ETF_CODES:
                    target_val = tv * TARGET_WEIGHTS[code] * target_invested_ratio
                    cur_val = holdings[code] * px[code]
                    diff = target_val - cur_val
                    if diff > 0:
                        units = diff / px[code]
                        _do_trade("BUY", code, units, px[code],
                                  f"建仓{step_idx}/{BUILD_UP_MONTHS}")

        # ---- 计算当前净值 ----
        tv = _portfolio_value(holdings, px, cash)
        nav.iloc[i] = tv

        # ---- 更新卫星回撤 ----
        srn = _sat_ref_nav(px)
        if srn > sat_hwm:
            sat_hwm = srn
            if in_stop and srn >= sat_hwm * DD_RECOVER_THRESHOLD:
                if verbose:
                    print(f"  [{date.date()}] 卫星恢复至前高{DD_RECOVER_THRESHOLD:.0%}，恢复仓位")
                in_stop = False
        sat_dd = srn / sat_hwm - 1

        # ---- 回撤止损（建仓完成后） ----
        if i >= BUILD_UP_INTERVAL * BUILD_UP_MONTHS:
            days_since_dd = i - last_dd_action_day

            if not in_stop and sat_dd < DD_STOP_THRESHOLD and days_since_dd >= DD_COOLDOWN:
                if verbose:
                    print(f"  [{date.date()}] 卫星回撤{sat_dd:.1%}触发止损，卫星仓位减半")
                in_stop = True
                last_dd_action_day = i
                _rebalance_to_target(_get_target_weights(True), f"止损-卫星回撤{sat_dd:.1%}")

            elif in_stop and srn >= sat_hwm * DD_RECOVER_THRESHOLD and days_since_dd >= DD_COOLDOWN:
                if verbose:
                    print(f"  [{date.date()}] 卫星恢复至前高{DD_RECOVER_THRESHOLD:.0%}，恢复仓位")
                in_stop = False
                last_dd_action_day = i
                _rebalance_to_target(_get_target_weights(False), "回撤恢复")

        # ---- 再平衡（建仓完成后，有冷却期） ----
        if i >= BUILD_UP_INTERVAL * BUILD_UP_MONTHS:
            days_since_reb = i - last_rebalance_day
            if days_since_reb >= REBALANCE_COOLDOWN:
                cw = _calc_weights(holdings, px, cash)
                tw = _get_target_weights(in_stop)
                if _check_deviation(cw, tw):
                    if verbose:
                        print(f"  [{date.date()}] 偏离超{REBALANCE_THRESHOLD:.0%}，触发再平衡")
                    _rebalance_to_target(tw, "再平衡")
                    last_rebalance_day = i

        # ---- 波动率控制（建仓完成后，冷却期内不触发） ----
        vol_check_idx = i - BUILD_UP_INTERVAL * BUILD_UP_MONTHS
        if vol_check_idx >= 22 and i % 21 == 0:
            days_since_vol = i - last_vol_action_day
            if days_since_vol >= VOL_COOLDOWN:
                lookback = nav.iloc[max(0, i-22):i+1].pct_change().dropna()
                if len(lookback) >= 10:
                    rv = lookback.std() * np.sqrt(252)

                    if rv > VOL_TARGET and not in_vol_control:
                        if verbose:
                            print(f"  [{date.date()}] 波动率{rv:.1%}>{VOL_TARGET:.0%}，减仓{VOL_CASH_RATIO:.0%}")
                        in_vol_control = True
                        last_vol_action_day = i
                        for code in ETF_CODES:
                            units = holdings[code] * VOL_CASH_RATIO
                            _do_trade("SELL", code, units, px[code],
                                      f"波动率控制-年化{rv:.1%}")

                    elif rv <= VOL_TARGET and in_vol_control:
                        if verbose:
                            print(f"  [{date.date()}] 波动率回落{rv:.1%}，恢复仓位")
                        in_vol_control = False
                        last_vol_action_day = i
                        _rebalance_to_target(_get_target_weights(in_stop), "波动率恢复")

        # ---- 记录权重 ----
        tv_final = _portfolio_value(holdings, px, cash)
        weights_df.loc[date, "cash"] = cash / tv_final if tv_final > 0 else 0
        for code in ETF_CODES:
            weights_df.loc[date, code] = holdings[code] * px[code] / tv_final if tv_final > 0 else 0

    # ---- 后处理 ----
    daily_returns = nav.pct_change().fillna(0)

    bench_nav = None
    if benchmark is not None:
        bc = benchmark.reindex(dates).dropna()
        if len(bc) > 1:
            bench_nav = bc / bc.iloc[0]

    metrics = calculate_metrics(nav, bench_nav)

    # 卫星回撤序列
    sat_ref_vals = [_sat_ref_nav({c: float(prices.iloc[j][c]) for c in ETF_CODES})
                    for j in range(n)]
    # re-init base prices for series calc
    sat_ref_series = pd.Series(sat_ref_vals, index=dates)
    sat_hwm_series = sat_ref_series.expanding().max()
    sat_dd_series = sat_ref_series / sat_hwm_series - 1

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["date", "code", "action", "price", "units", "reason"])

    return BacktestResult(
        nav=nav, daily_returns=daily_returns, trades=trades_df,
        weights_history=weights_df, satellite_dd_history=sat_dd_series,
        metrics=metrics, benchmark_nav=bench_nav)


# ============================================================
if __name__ == "__main__":
    data = load_data()
    prices = align_prices(data)
    print(f"回测数据: {prices.index[0].date()} ~ {prices.index[-1].date()} ({len(prices)}天)")
    result = run_backtest(prices, data.get("benchmark"), verbose=True)
    print("\n" + "=" * 60)
    print("全区间回测结果 (2019-10 ~ 2025-12)")
    print("=" * 60)
    for k, v in result.metrics.items():
        print(f"  {k:10s}: {v}")
    print(f"\n总交易次数: {len(result.trades)}")
