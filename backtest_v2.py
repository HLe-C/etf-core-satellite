"""
V2 回测引擎 — 技术分析为主、月度动态选股
核心原则：每次调仓只用当日及之前的OHLCV数据，零未来信息
"""
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"

# ============ 策略参数（可调） ============

@dataclass
class StrategyParams:
    # 核心-卫星结构
    core_etfs: Tuple[str, ...] = ("510300", "510500", "159915", "512890")
    satellite_pool: Tuple[str, ...] = (
        "512880", "512800", "512660", "512010", "159928",
        "512580", "512980", "512200", "513100", "518880",
        "512690", "512760", "512720", "515050",
    )
    n_satellites: int = 2
    satellite_weight_each: float = 0.10

    # 技术信号
    ma_short: int = 50
    ma_long: int = 200
    mom_filter: int = 20
    mom_rank: int = 60

    # 仓位管理
    equity_bull: float = 0.90
    equity_range: float = 0.70
    equity_bear: float = 0.50

    # 波动率缩放（None = 不启用）
    target_vol: float = None  # 目标年化波动率，如 0.15

    # 交易执行
    rebalance_freq: str = "monthly"    # monthly / weekly
    deviation_threshold: float = 0.03  # 偏离阈值
    fee_rate: float = 0.0001           # 单边万一
    initial_cash: float = 100_000.0

    # 熔断
    circuit_breaker_drop: float = -0.025
    circuit_breaker_cooldown: int = 15

    # ATR止盈
    atr_period: int = 20
    atr_mult: float = 3.0              # 浮盈 > N倍ATR 触发止盈

    # 递进加入
    listing_warmup_days: int = 60      # 上市后多少天才可入选

    # 回测区间
    start_date: str = "2019-10-01"     # 留足MA200计算空间
    end_date: str = "2024-12-31"

    def label(self) -> str:
        return f"n{self.n_satellites}_mom{self.mom_rank}_f{self.rebalance_freq}"


# ============ 数据加载 ============

def load_all_data(params: StrategyParams) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """加载所有ETF日线和HS300基准"""
    etf_data = {}
    all_symbols = list(params.core_etfs) + list(params.satellite_pool)

    for sym in all_symbols:
        # 找对应的csv
        for f in DATA_DIR.glob(f"{sym}_*.csv"):
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            df = df.sort_index()
            # 补全OHLCV列（部分旧文件只有close）
            if "open" not in df.columns:
                df["open"] = df["close"]
            if "high" not in df.columns:
                df["high"] = df["close"]
            if "low" not in df.columns:
                df["low"] = df["close"]
            if "volume" not in df.columns:
                df["volume"] = 0.0
            df = df[["open", "close", "high", "low", "volume"]].astype(float)
            etf_data[sym] = df
            break

    bench = pd.read_csv(DATA_DIR / "benchmark_000300.csv", index_col=0, parse_dates=True)
    bench = bench[["close"]].astype(float).sort_index()
    bench.columns = ["benchmark"]

    print(f"已加载 {len(etf_data)} 只ETF + HS300基准")
    return etf_data, bench


# ============ 技术指标计算 ============

def compute_indicators(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """在日线数据上计算所有技术指标（无未来信息）"""
    if columns is None:
        columns = ["close", "high", "low", "volume"]
    result = df[columns].copy()

    # 均线
    result["MA20"] = result["close"].rolling(20).mean()
    result["MA50"] = result["close"].rolling(50).mean()
    result["MA200"] = result["close"].rolling(200).mean()

    # 动量（多个窗口，覆盖可调参数范围）
    for w in [10, 20, 40, 60, 120]:
        result[f"mom{w}"] = result["close"].pct_change(w)

    # ATR
    h, l, c = result["high"], result["low"], result["close"].shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - c).abs(),
        (l - c).abs(),
    ], axis=1).max(axis=1)
    result["ATR20"] = tr.rolling(20).mean()

    # 成交量均线
    result["vol_ma20"] = result["volume"].rolling(20).mean()

    return result


# ============ 回测引擎 ============

@dataclass
class Position:
    symbol: str
    shares: float = 0.0
    cost_basis: float = 0.0       # 加权平均成本
    entry_date: pd.Timestamp = None

    @property
    def value(self) -> float:
        return 0.0

    @property
    def pnl_pct(self) -> float:
        if self.cost_basis <= 0 or self.shares <= 0:
            return 0.0
        return self.value / (self.shares * self.cost_basis) - 1


class BacktestEngineV2:
    def __init__(self, params: StrategyParams):
        self.p = params
        self.etf_data: Dict[str, pd.DataFrame] = {}
        self.indicators: Dict[str, pd.DataFrame] = {}
        self.bench_data: pd.DataFrame = None
        self.bench_ind: pd.DataFrame = None

        # 状态
        self.cash: float = params.initial_cash
        self.positions: Dict[str, Position] = {}
        self.trade_dates: List[pd.Timestamp] = []

        # 记录
        self.nav_history: List[Dict] = []
        self.position_history: List[Dict] = []  # 每日持仓权重快照
        self.trade_log: List[Dict] = []       # 每笔交易理由
        self.monthly_log: List[Dict] = []     # 月度调仓理由

    # ---------- 准备阶段 ----------

    def prepare(self):
        """加载数据，预计算所有指标"""
        self.etf_data, self.bench_data = load_all_data(self.p)

        for sym, df in self.etf_data.items():
            self.indicators[sym] = compute_indicators(df)

        self.bench_ind = compute_indicators(
            self.bench_data.rename(columns={"benchmark": "close"}).assign(
                high=lambda x: x["close"], low=lambda x: x["close"], volume=0
            ),
            columns=["close", "high", "low", "volume"],
        )

        # 预计算 HS300 20日滚动波动率（年化）
        bench_ret = self.bench_data["benchmark"].pct_change().dropna()
        self.bench_vol = (bench_ret.rolling(20).std() * np.sqrt(252)).reindex(
            self.bench_data.index
        ).ffill()

        # 确定交易日期
        all_dates = self.bench_ind.index
        mask = (all_dates >= self.p.start_date) & (all_dates <= self.p.end_date)
        self.trade_dates = sorted(all_dates[mask])

        # 初始化每个ETF的头寸记录
        for sym in {**{s: s for s in self.p.core_etfs}, **{s: s for s in self.p.satellite_pool}}:
            if sym in self.etf_data:
                self.positions[sym] = Position(symbol=sym)

        print(f"回测区间: {self.trade_dates[0].date()} ~ {self.trade_dates[-1].date()}")
        print(f"交易日: {len(self.trade_dates)} 天")
        print(f"核心ETF: {list(self.p.core_etfs)}")
        print(f"卫星候选池: {len(self.p.satellite_pool)} 只")

    # ---------- 辅助函数 ----------

    def _get_close(self, sym: str, date: pd.Timestamp) -> Optional[float]:
        df = self.etf_data.get(sym)
        if df is None:
            return None
        try:
            return df.loc[date, "close"]
        except KeyError:
            return None

    def _get_indicator(self, sym: str, date: pd.Timestamp) -> Optional[pd.Series]:
        df = self.indicators.get(sym)
        if df is None:
            return None
        try:
            return df.loc[date]
        except KeyError:
            return None

    def _etf_listed(self, sym: str, date: pd.Timestamp) -> bool:
        """检查ETF是否已上市足够天数"""
        df = self.etf_data.get(sym)
        if df is None:
            return False
        first_date = df.index[0]
        return date >= first_date + pd.Timedelta(days=self.p.listing_warmup_days)

    def _get_market_state(self, date: pd.Timestamp) -> str:
        row = self.bench_ind.loc[date]
        close, ma50, ma200 = row["close"], row["MA50"], row["MA200"]
        if pd.isna(ma200) or pd.isna(ma50):
            return "range"
        if close > ma200 and ma50 > ma200:
            return "bull"
        elif close < ma200:
            return "bear"
        else:
            return "range"

    def _get_equity_target(self, state: str, date: pd.Timestamp = None) -> float:
        return {"bull": self.p.equity_bull, "range": self.p.equity_range, "bear": self.p.equity_bear}[state]

    def _get_core_weight(self, equity_target: float) -> float:
        return max(0, equity_target - self.p.n_satellites * self.p.satellite_weight_each)

    def _get_vol_scale(self, date: pd.Timestamp) -> float:
        """波动率缩放系数（target_vol=None时返回1.0，不生效）"""
        if self.p.target_vol is None:
            return 1.0
        try:
            cur_vol = float(self.bench_vol.loc[date])
        except (KeyError, TypeError):
            return 1.0
        if pd.isna(cur_vol) or cur_vol <= 0:
            return 1.0
        if cur_vol > self.p.target_vol:
            return self.p.target_vol / cur_vol
        return 1.0

    def _filter_active_core(self, date: pd.Timestamp, state: str) -> List[str]:
        """熊市核心向防守倾斜：仅保留MA200以上的核心，512890始终保留"""
        if state != "bear":
            return list(self.p.core_etfs)
        active = []
        for sym in self.p.core_etfs:
            if sym == "512890":
                active.append(sym)
                continue
            ind = self._get_indicator(sym, date)
            close = self._get_close(sym, date)
            if ind is None or close is None or pd.isna(ind["MA200"]):
                active.append(sym)
            elif close > ind["MA200"]:
                active.append(sym)
        if not active:
            active = ["512890"]
        return active

    def _get_price(self, sym: str, date: pd.Timestamp) -> Optional[float]:
        return self._get_close(sym, date)

    def _record_position_snapshot(self, date: pd.Timestamp):
        """记录当日收盘价下的持仓市值和权重。"""
        row = {"date": date, "cash": self.cash}
        total_value = self.cash
        for sym, pos in self.positions.items():
            close = self._get_close(sym, date)
            value = pos.shares * close if close is not None and pos.shares > 0 else 0.0
            row[f"{sym}_value"] = value
            total_value += value

        row["total_value"] = total_value
        row["cash_weight"] = self.cash / total_value if total_value > 0 else 0.0
        for sym in self.positions:
            value = row.get(f"{sym}_value", 0.0)
            row[f"{sym}_weight"] = value / total_value if total_value > 0 else 0.0
        self.position_history.append(row)

    # ---------- 核心逻辑 ----------

    def _check_daily_risk(self, date: pd.Timestamp, total_value: float, i: int):
        """
        每日风控检查（方案A改进）：
        1. 持仓卫星 close < MA20 → 立即清仓
        2. HS300 单日跌幅 > 2.5% → 熔断（清仓卫星，降仓至50%，冷却15日）
        """
        # 熔断冷却期检查
        if hasattr(self, 'cb_days_left') and self.cb_days_left > 0:
            self.cb_days_left -= 1
            return

        # 卫星 MA20 止损
        for sym in self.p.satellite_pool:
            pos = self.positions.get(sym)
            if pos is None or pos.shares <= 0:
                continue
            ind = self._get_indicator(sym, date)
            close = self._get_close(sym, date)
            if ind is None or close is None or pd.isna(ind["MA20"]):
                continue
            if close < ind["MA20"]:
                proceeds = pos.shares * close * (1 - self.p.fee_rate)
                self.cash += proceeds
                self.trade_log.append({
                    "date": date, "symbol": sym, "action": "MA20止损",
                    "price": close,
                    "shares": pos.shares,
                    "amount": proceeds,
                    "fee": pos.shares * close * self.p.fee_rate,
                    "reason": f"close({close:.3f}) < MA20({ind['MA20']:.3f})",
                })
                pos.shares = 0
                pos.cost_basis = 0.0

        # HS300 单日暴跌熔断
        if i >= 1:
            prev_bench = self.bench_data.loc[self.trade_dates[i-1], "benchmark"] \
                if self.trade_dates[i-1] in self.bench_data.index else None
            curr_bench = self.bench_data.loc[date, "benchmark"] \
                if date in self.bench_data.index else None
            if prev_bench and curr_bench and prev_bench > 0:
                daily_ret = curr_bench / prev_bench - 1
                if daily_ret < self.p.circuit_breaker_drop:
                    # 熔断：清仓所有卫星
                    for sym in self.p.satellite_pool:
                        pos = self.positions.get(sym)
                        if pos and pos.shares > 0:
                            close = self._get_close(sym, date)
                            if close:
                                proceeds = pos.shares * close * (1 - self.p.fee_rate)
                                self.cash += proceeds
                                self.trade_log.append({
                                    "date": date, "symbol": sym, "action": "熔断清仓",
                                    "price": close,
                                    "shares": pos.shares,
                                    "amount": proceeds,
                                    "fee": pos.shares * close * self.p.fee_rate,
                                    "reason": f"HS300单日跌{daily_ret:.2%}，触发熔断",
                                })
                                pos.shares = 0
                                pos.cost_basis = 0.0
                    self.cb_days_left = self.p.circuit_breaker_cooldown
                    self.monthly_log.append({
                        "date": date,
                        "market_state": "CIRCUIT_BREAKER",
                        "equity_target": 0.50,
                        "core_weight": 0.50,
                        "selected_satellites": [],
                        "n_selected": 0,
                        "sat_reasons": [f"熔断触发: HS300日跌{daily_ret:.2%}，冷却{self.p.circuit_breaker_cooldown}日"],
                        "total_value": total_value,
                    })

    def _select_satellites(self, date: pd.Timestamp) -> Tuple[List[str], List[Dict]]:
        """
        月度卫星选股
        返回: (选中symbol列表, 每只的判断理由)
        """
        candidates = []
        reasons = []

        for sym in self.p.satellite_pool:
            if not self._etf_listed(sym, date):
                reasons.append({"symbol": sym, "status": "排除", "reason": "上市不足60交易日"})
                continue

            ind = self._get_indicator(sym, date)
            close = self._get_close(sym, date)

            if ind is None or close is None or pd.isna(ind["MA200"]):
                reasons.append({"symbol": sym, "status": "排除", "reason": "数据不足(MA200未就绪)"})
                continue

            ma200 = ind["MA200"]
            mom_filter_key = f"mom{self.p.mom_filter}"
            mom_rank_key = f"mom{self.p.mom_rank}"
            mom_filter_val = ind.get(mom_filter_key)
            mom_rank_val = ind.get(mom_rank_key)

            # 条件1: close > MA200
            if close <= ma200:
                reasons.append({
                    "symbol": sym,
                    "status": "淘汰",
                    "reason": f"close({close:.3f}) <= MA200({ma200:.3f})",
                })
                continue

            # 条件2: 动量过滤 > 0
            if pd.isna(mom_filter_val) or mom_filter_val <= 0:
                reasons.append({
                    "symbol": sym,
                    "status": "淘汰",
                    "reason": f"{self.p.mom_filter}日动量({mom_filter_val:.1%}) <= 0",
                })
                continue

            candidates.append({
                "symbol": sym,
                "mom_rank": mom_rank_val if not pd.isna(mom_rank_val) else -999,
            })

        # 没有候选 → 全部额度给核心
        if not candidates:
            return [], reasons

        # 按N日动量排序，取前N
        candidates.sort(key=lambda x: x["mom_rank"], reverse=True)
        selected = [c["symbol"] for c in candidates[:self.p.n_satellites]]

        for c in candidates:
            if c["symbol"] in selected:
                reasons.append({
                    "symbol": c["symbol"],
                    "status": "入选",
                    "reason": f"close>MA200, {self.p.mom_filter}日动量>0, {self.p.mom_rank}日动量={c['mom_rank']:.1%}, 排名第{selected.index(c['symbol'])+1}",
                })
            else:
                reasons.append({
                    "symbol": c["symbol"],
                    "status": "淘汰",
                    "reason": f"满足条件但{self.p.mom_rank}日动量({c['mom_rank']:.1%})未排进前{self.p.n_satellites}",
                })

        return selected, reasons

    def _check_atr_take_profit(self, date: pd.Timestamp, sym: str,
                                 pos: Position, target_weight: float,
                                 total_value: float) -> Tuple[float, str]:
        """检查是否需要ATR止盈，返回(调整后权重, 理由)"""
        ind = self._get_indicator(sym, date)
        if ind is None or pd.isna(ind["ATR20"]):
            return target_weight, ""

        atr = ind["ATR20"]
        if pos.cost_basis <= 0 or pos.shares <= 0:
            return target_weight, ""

        current_price = self._get_close(sym, date)
        if current_price is None:
            return target_weight, ""

        pnl = current_price / pos.cost_basis - 1
        if pnl > self.p.atr_mult * atr / (pos.cost_basis or 1):
            # 止盈 1/3
            new_weight = target_weight * (2 / 3)
            return new_weight, f"ATR止盈(浮盈{pnl:.1%} > {self.p.atr_mult}×ATR({atr/pos.cost_basis:.1%}))"

        return target_weight, ""

    # ---------- 主循环 ----------

    def run(self):
        self.prepare()

        print("\n开始回测...")
        last_period = None
        rebalance_count = 0

        for i, date in enumerate(self.trade_dates):
            # ---- 快速估算当前总资产 ----
            est_total = self.cash
            for sym, pos in self.positions.items():
                close = self._get_close(sym, date)
                if close is not None and pos.shares > 0:
                    est_total += pos.shares * close

            # ---- 每日风控检查（MA20止损 + HS300熔断）----
            self._check_daily_risk(date, est_total, i)

            # ---- 每日记录净值（风控后） ----
            total_value = self.cash
            equity_value = 0.0
            for sym, pos in self.positions.items():
                close = self._get_close(sym, date)
                if close is not None and pos.shares > 0:
                    pos_value = pos.shares * close
                    total_value += pos_value
                    equity_value += pos_value

            bench_val = self.bench_data.loc[date, "benchmark"] if date in self.bench_data.index else None

            self.nav_history.append({
                "date": date,
                "nav": total_value,
                "equity": equity_value,
                "cash": self.cash,
                "bench": bench_val,
            })

            # ---- 月度调仓检查 ----
            current_month = date.month
            current_year = date.year
            month_key = (current_year, current_month)

            do_rebalance = False
            if self.p.rebalance_freq == "monthly":
                do_rebalance = (month_key != last_period)
            else:  # weekly
                week_num = date.isocalendar()[1]  # ISO week number
                week_key = (current_year, week_num)
                do_rebalance = (week_key != last_period)

            if do_rebalance and i > 0:
                last_period = month_key if self.p.rebalance_freq == "monthly" else week_key
                rebalance_count += 1
                self._monthly_rebalance(date, total_value)

            self._record_position_snapshot(date)

            # 进度
            if (i + 1) % 500 == 0:
                print(f"  进度: {i+1}/{len(self.trade_dates)} ({date.date()}), NAV={total_value:.2f}")

        print(f"回测完成! {len(self.trade_dates)}天, {rebalance_count}次调仓")

    def _monthly_rebalance(self, date: pd.Timestamp, total_value: float):
        """月度调仓 — MA200状态判断 + 熊市核心过滤"""
        in_cb = hasattr(self, 'cb_days_left') and self.cb_days_left > 0
        if in_cb:
            state = "bear"
            equity_target = self.p.equity_bear
            core_weight = equity_target
            selected_sats = []
            sat_reasons = [{"symbol": "-", "status": "熔断冷却", "reason": f"剩余{self.cb_days_left}日"}]
        else:
            state = self._get_market_state(date)
            equity_target = self._get_equity_target(state, date)
            core_weight = self._get_core_weight(equity_target)
            selected_sats, sat_reasons = self._select_satellites(date)

        # 波动率缩放：高波动时降仓位
        if self.p.target_vol is not None:
            vol_scale = self._get_vol_scale(date)
            if vol_scale < 1.0:
                equity_target *= vol_scale
                core_weight = max(0, equity_target - self.p.n_satellites * self.p.satellite_weight_each)

        active_core = self._filter_active_core(date, state)
        n_active_core = len(active_core)

        target_weights = {}
        core_each = core_weight / n_active_core
        for sym in active_core:
            target_weights[sym] = core_each
        for sym in selected_sats:
            pos = self.positions.get(sym)
            if pos and pos.shares > 0:
                adj_w, atr_reason = self._check_atr_take_profit(date, sym, pos, self.p.satellite_weight_each, total_value)
                target_weights[sym] = adj_w
                if atr_reason:
                    self.trade_log.append({"date": date, "symbol": sym, "action": "ATR止盈", "reason": atr_reason})
            else:
                target_weights[sym] = self.p.satellite_weight_each

        n_selected = len(selected_sats)
        reasons_text = []
        for r in sat_reasons[:8]:
            reasons_text.append(f"  {r['symbol']}: {r['status']}({r['reason']})")
        dropped_core = [s for s in self.p.core_etfs if s not in active_core]
        if dropped_core:
            reasons_text.append(f"  核心过滤: 剔除{dropped_core}")

        self.monthly_log.append({
            "date": date,
            "market_state": state,
            "equity_target": equity_target,
            "core_weight": core_weight,
            "active_core": active_core,
            "sat_weight": self.p.satellite_weight_each,
            "selected_satellites": selected_sats,
            "n_selected": n_selected,
            "sat_reasons": reasons_text,
            "total_value": total_value,
        })

        # ---- 执行调仓 ----
        for sym, target_w in target_weights.items():
            close = self._get_close(sym, date)
            if close is None or close <= 0:
                continue

            pos = self.positions.get(sym)
            if pos is None:
                continue

            target_value = total_value * target_w
            current_value = pos.shares * close
            diff_value = target_value - current_value
            diff_pct = abs(diff_value) / total_value if total_value > 0 else 0

            # 偏离 < 阈值 不交易
            if diff_pct < self.p.deviation_threshold and pos.shares > 0:
                continue

            if diff_value > 0:
                # 买入
                shares_to_buy = diff_value / close
                cost = shares_to_buy * close * (1 + self.p.fee_rate)
                self.cash -= cost
                new_shares = pos.shares + shares_to_buy
                pos.cost_basis = ((pos.shares * pos.cost_basis + shares_to_buy * close) /
                                  new_shares if new_shares > 0 else 0)
                pos.shares = new_shares
                pos.entry_date = date
                self.trade_log.append({
                    "date": date, "symbol": sym, "action": "买入",
                    "price": close, "shares": shares_to_buy,
                    "amount": cost, "fee": shares_to_buy * close * self.p.fee_rate,
                    "target_weight": target_w,
                    "reason": "月度调仓至目标权重",
                })

            elif diff_value < 0:
                # 卖出
                shares_to_sell = min(pos.shares, abs(diff_value) / close)
                proceeds = shares_to_sell * close * (1 - self.p.fee_rate)
                self.cash += proceeds
                pos.shares -= shares_to_sell
                self.trade_log.append({
                    "date": date, "symbol": sym, "action": "卖出",
                    "price": close, "shares": shares_to_sell,
                    "amount": proceeds, "fee": shares_to_sell * close * self.p.fee_rate,
                    "target_weight": target_w,
                    "reason": "月度调仓至目标权重",
                })
                if pos.shares <= 0:
                    pos.cost_basis = 0.0

        # 清退不在目标中的卫星
        for sym in self.p.satellite_pool:
            if sym not in target_weights:
                pos = self.positions.get(sym)
                if pos and pos.shares > 0:
                    close = self._get_close(sym, date)
                    if close and close > 0:
                        proceeds = pos.shares * close * (1 - self.p.fee_rate)
                        self.cash += proceeds
                        self.trade_log.append({
                            "date": date, "symbol": sym, "action": "清仓",
                            "price": close,
                            "shares": pos.shares,
                            "amount": proceeds,
                            "fee": pos.shares * close * self.p.fee_rate,
                            "reason": f"月度调仓，未入选卫星",
                        })
                        pos.shares = 0
                        pos.cost_basis = 0.0

        # 清退被MA200过滤掉的核心ETF
        for sym in self.p.core_etfs:
            if sym not in active_core:
                pos = self.positions.get(sym)
                if pos and pos.shares > 0:
                    close = self._get_close(sym, date)
                    if close and close > 0:
                        proceeds = pos.shares * close * (1 - self.p.fee_rate)
                        self.cash += proceeds
                        self.trade_log.append({
                            "date": date, "symbol": sym, "action": "核心清仓",
                            "price": close,
                            "shares": pos.shares,
                            "amount": proceeds,
                            "fee": pos.shares * close * self.p.fee_rate,
                            "reason": f"非牛市且close < MA200，剔除核心",
                        })
                        pos.shares = 0
                        pos.cost_basis = 0.0

    # ---------- 结果分析 ----------

    def get_nav_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.nav_history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # 计算基准归一化
        if "bench" in df.columns and df["bench"].notna().any():
            base_bench = df["bench"].iloc[0]
            df["bench_nav"] = df["bench"] / base_bench * self.p.initial_cash

        base_nav = df["nav"].iloc[0]
        df["nav_pct"] = df["nav"] / self.p.initial_cash - 1
        if "bench_nav" in df.columns:
            df["bench_pct"] = df["bench_nav"] / self.p.initial_cash - 1
        df["nav_return"] = df["nav"].pct_change()
        df["drawdown"] = (df["nav"] / df["nav"].cummax() - 1)

        return df

    def get_position_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.position_history)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def get_metrics(self) -> Dict:
        df = self.get_nav_df()
        if len(df) < 2:
            return {}

        total_days = (df.index[-1] - df.index[0]).days
        total_years = total_days / 365.25

        total_return = df["nav"].iloc[-1] / df["nav"].iloc[0] - 1
        ann_return = (1 + total_return) ** (1 / total_years) - 1 if total_years > 0 else 0
        ann_vol = df["nav_return"].std() * np.sqrt(252) if len(df["nav_return"].dropna()) > 1 else 0
        max_dd = df["drawdown"].min()
        sharpe = (ann_return - 0.025) / ann_vol if ann_vol > 0 else 0  # 2.5%无风险
        calmar = ann_return / abs(max_dd) if max_dd < 0 else 0

        # 超额 vs 基准
        if "bench_nav" in df.columns:
            bench_return = df["bench_nav"].iloc[-1] / df["bench_nav"].iloc[0] - 1
            ann_bench = (1 + bench_return) ** (1 / total_years) - 1
            excess = ann_return - ann_bench
        else:
            ann_bench = 0
            excess = 0

        return {
            "total_return": total_return,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "calmar": calmar,
            "ann_bench": ann_bench,
            "excess": excess,
            "total_days": total_days,
            "total_years": round(total_years, 2),
            "n_rebalances": len(self.monthly_log),
            "n_trades": len(self.trade_log),
        }

    def print_summary(self):
        m = self.get_metrics()
        print("\n" + "=" * 55)
        print(f"V2 策略回测总结  [{self.p.label()}]")
        print(f"区间: {self.p.start_date} ~ {self.p.end_date} ({m['total_years']}年)")
        print("=" * 55)
        print(f"  累计收益:     {m['total_return']:>+8.2%}")
        print(f"  年化收益:     {m['ann_return']:>+8.2%}")
        print(f"  基准年化:     {m['ann_bench']:>+8.2%}")
        print(f"  超额年化:     {m['excess']:>+8.2%}")
        print(f"  年化波动:     {m['ann_vol']:>8.2%}")
        print(f"  最大回撤:     {m['max_drawdown']:>8.2%}")
        print(f"  夏普比率:     {m['sharpe']:>8.3f}")
        print(f"  卡玛比率:     {m['calmar']:>8.3f}")
        print(f"  月度调仓:     {m['n_rebalances']:>8} 次")
        print(f"  交易笔数:     {m['n_trades']:>8} 次")


# ============ 主程序 ============

def run_base():
    """运行基准参数"""
    params = StrategyParams()
    engine = BacktestEngineV2(params)
    engine.run()
    engine.print_summary()

    # 保存结果
    OUTPUT_DIR.mkdir(exist_ok=True)
    nav_df = engine.get_nav_df()
    nav_df.to_csv(OUTPUT_DIR / "v2_nav.csv")

    # 保存月度日志
    monthly_df = pd.DataFrame(engine.monthly_log)
    monthly_df.to_csv(OUTPUT_DIR / "v2_monthly_log.csv", index=False)
    print(f"\n结果已保存到 output/")


if __name__ == "__main__":
    run_base()
