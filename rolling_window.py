"""
滚动窗口回测模块

参数：
  - 窗口长度: 24个月
  - 滚动步长: 3个月
  - In-sample: 2019.10 ~ 2024.12
  - Out-of-sample验证: 2025.01 ~ 2025.12
"""
import pandas as pd
import numpy as np
from pathlib import Path
from backtest_engine import (
    load_data, align_prices, run_backtest, calculate_metrics,
    TARGET_WEIGHTS, ETF_CODES
)

DATA_DIR = Path(__file__).parent / "data"
WINDOW_MONTHS = 24
STEP_MONTHS = 3


def generate_windows(prices: pd.DataFrame):
    """
    生成滚动窗口的起止日期列表
    返回: [(train_start, train_end, test_start, test_end), ...]
    """
    dates = pd.Series(prices.index).sort_values()
    start = dates.iloc[0]
    
    windows = []
    current = pd.Timestamp(start)
    
    while True:
        train_end = current + pd.DateOffset(months=WINDOW_MONTHS)
        test_end = train_end + pd.DateOffset(months=STEP_MONTHS)
        
        if test_end > pd.Timestamp("2025-01-01"):
            break
        
        # 找到最近的交易日
        train_start_idx = dates.searchsorted(current)
        train_end_idx = dates.searchsorted(train_end) - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = dates.searchsorted(test_end) - 1
        
        if train_start_idx >= len(dates) or train_end_idx >= len(dates):
            break
        if test_start_idx >= len(dates) or test_end_idx >= len(dates):
            break
        
        windows.append({
            "train_start": dates.iloc[train_start_idx],
            "train_end": dates.iloc[train_end_idx],
            "test_start": dates.iloc[test_start_idx] if test_start_idx < len(dates) else None,
            "test_end": dates.iloc[min(test_end_idx, len(dates)-1)],
        })
        
        current += pd.DateOffset(months=STEP_MONTHS)
    
    return windows


def run_rolling_backtest(prices: pd.DataFrame, benchmark=None, verbose: bool = True):
    """
    执行滚动窗口回测
    每个窗口: 用训练数据确定参数 → 在外推窗口验证
    """
    windows = generate_windows(prices)
    
    if verbose:
        print(f"滚动窗口回测：窗口={WINDOW_MONTHS}月, 步长={STEP_MONTHS}月")
        print(f"共 {len(windows)} 个窗口")
        print("-" * 60)
    
    all_results = []
    train_metrics_list = []
    test_metrics_list = []
    
    for idx, w in enumerate(windows):
        # 训练窗口回测
        train_result = run_backtest(
            prices,
            benchmark=benchmark,
            start_date=str(w["train_start"].date()),
            end_date=str(w["train_end"].date()),
            verbose=False,
        )
        
        train_m = {
            "window": idx + 1,
            "train_start": str(w["train_start"].date()),
            "train_end": str(w["train_end"].date()),
            "test_start": str(w["test_start"].date()),
            "test_end": str(w["test_end"].date()),
        }
        train_m.update({f"train_{k}": v for k, v in train_result.metrics.items()})
        
        # 外推窗口回测
        test_result = run_backtest(
            prices,
            benchmark=benchmark,
            start_date=str(w["test_start"].date()),
            end_date=str(w["test_end"].date()),
            verbose=False,
        )
        
        test_m = {
            "window": idx + 1,
        }
        test_m.update({f"test_{k}": v for k, v in test_result.metrics.items()})
        
        train_metrics_list.append(train_m)
        test_metrics_list.append(test_m)
        
        all_results.append({
            "window": w,
            "train": train_result,
            "test": test_result,
        })
        
        if verbose:
            train_ret = train_result.metrics.get("累计收益", "N/A")
            test_ret = test_result.metrics.get("累计收益", "N/A")
            print(f"  窗口{idx+1:2d}: "
                  f"训练[{w['train_start'].date()}~{w['train_end'].date()}] "
                  f"收益={train_ret} → "
                  f"外推[{w['test_start'].date()}~{w['test_end'].date()}] "
                  f"收益={test_ret}")
    
    train_df = pd.DataFrame(train_metrics_list)
    test_df = pd.DataFrame(test_metrics_list)
    
    return all_results, train_df, test_df


def run_validation_2025(prices: pd.DataFrame, benchmark=None, verbose: bool = True):
    """
    2025年 out-of-sample 验证
    策略参数冻结（使用2019-2024全区间训练后的配置）
    """
    if verbose:
        print("\n" + "=" * 60)
        print("2025年 Out-of-Sample 验证")
        print("=" * 60)
    
    result = run_backtest(
        prices,
        benchmark=benchmark,
        start_date="2025-01-01",
        end_date="2025-12-31",
        verbose=verbose,
    )
    
    if verbose:
        print("\n2025年验证结果:")
        for k, v in result.metrics.items():
            print(f"  {k:10s}: {v}")
        print(f"\n  交易次数: {len(result.trades)}")
    
    return result


def summarize_rolling(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """汇总滚动窗口统计"""
    print("\n" + "=" * 60)
    print("滚动窗口汇总统计")
    print("=" * 60)
    
    # 提取数值列
    def parse_pct(s):
        if isinstance(s, str) and "%" in s:
            return float(s.replace("%", "")) / 100
        return float(s) if s else 0
    
    # 训练窗口汇总
    train_summary = {}
    for col in train_df.columns:
        if col.startswith("train_") and col != "train_start" and col != "train_end" and col != "test_start" and col != "test_end":
            vals = train_df[col].apply(parse_pct)
            train_summary[col] = {
                "mean": vals.mean(),
                "std": vals.std(),
                "min": vals.min(),
                "max": vals.max(),
            }
    
    # 外推窗口汇总
    test_summary = {}
    for col in test_df.columns:
        if col.startswith("test_") and col != "window":
            vals = test_df[col].apply(parse_pct)
            test_summary[col] = {
                "mean": vals.mean(),
                "std": vals.std(),
                "min": vals.min(),
                "max": vals.max(),
            }
    
    print("\n--- 训练窗口 (In-Sample) ---")
    for k, v in train_summary.items():
        label = k.replace("train_", "")
        print(f"  {label}: 均值={v['mean']:.2%}, 标准差={v['std']:.2%}, "
              f"范围=[{v['min']:.2%}, {v['max']:.2%}]")
    
    print("\n--- 外推窗口 (滚动验证) ---")
    for k, v in test_summary.items():
        label = k.replace("test_", "")
        print(f"  {label}: 均值={v['mean']:.2%}, 标准差={v['std']:.2%}, "
              f"范围=[{v['min']:.2%}, {v['max']:.2%}]")
    
    return train_summary, test_summary


if __name__ == "__main__":
    data = load_data()
    prices = align_prices(data)
    benchmark = data.get("benchmark")
    
    # 滚动窗口
    results, train_df, test_df = run_rolling_backtest(prices, benchmark, verbose=True)
    summarize_rolling(train_df, test_df)
    
    # 2025验证
    val_result = run_validation_2025(prices, benchmark, verbose=True)
    
    # 保存结果
    train_df.to_csv(DATA_DIR / "rolling_train_metrics.csv", index=False)
    test_df.to_csv(DATA_DIR / "rolling_test_metrics.csv", index=False)
    print("\n滚动窗口结果已保存到 data/ 目录")
