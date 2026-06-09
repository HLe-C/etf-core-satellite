"""
数据获取模块 v2
拉取全部18只ETF（4核心 + 14卫星候选池） + 沪深300指数基准
"""
import akshare as ak
import pandas as pd
from pathlib import Path
import time

DATA_DIR = Path(__file__).parent / "data"

# ============ ETF 配置 ============

CORE_ETFS = {
    "510300": "沪深300",
    "510500": "中证500",
    "159915": "创业板",
    "512890": "红利低波",
}

SATELLITE_ETFS = {
    "512880": "证券",
    "512800": "银行",
    "512660": "军工",
    "512010": "医药",
    "159928": "消费",
    "512580": "环保",
    "512980": "传媒",
    "512200": "房地产",
    "513100": "纳指",
    "518880": "黄金",
    "512690": "酒",
    "512760": "半导体",
    "512720": "计算机",
    "515050": "5G通信",
}

ALL_ETFS = {**CORE_ETFS, **SATELLITE_ETFS}
BENCHMARK_IDX = "000300"


def fetch_etf(symbol: str, name: str) -> pd.DataFrame:
    """从 akshare 拉取单只ETF的日线数据"""
    print(f"  正在获取 {name}({symbol})...")
    try:
        df = ak.fund_etf_hist_em(
            symbol=symbol,
            period="daily",
            start_date="20180101",
            end_date="20251231",
            adjust="qfq",
        )
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df.set_index("date", inplace=True)
        print(f"    {name}: {len(df)} 条, {df.index[0].date()} ~ {df.index[-1].date()}")
        return df
    except Exception as e:
        print(f"    {name} 获取失败: {e}")
        return pd.DataFrame()


def fetch_benchmark() -> pd.DataFrame:
    """获取沪深300指数"""
    print("  正在获取沪深300指数(000300)...")
    try:
        df = ak.stock_zh_index_daily(symbol="sh000300")
        df = df.rename(columns={
            "date": "date", "open": "open", "close": "close",
            "high": "high", "low": "low", "volume": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df.set_index("date", inplace=True)
        print(f"    沪深300指数: {len(df)} 条, {df.index[0].date()} ~ {df.index[-1].date()}")
        return df
    except Exception as e:
        print(f"    沪深300指数获取失败: {e}")
        return pd.DataFrame()


def fetch_all():
    """拉取所有数据并保存为CSV"""
    DATA_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 55)
    print("V2 数据获取 — 18只ETF + 沪深300基准")
    print("=" * 55)

    # 1. 所有ETF
    for symbol, name in ALL_ETFS.items():
        df = fetch_etf(symbol, name)
        if not df.empty:
            df.to_csv(DATA_DIR / f"{symbol}_{name}.csv")
        time.sleep(0.5)  # 避免请求过快

    # 2. 基准
    bench_df = fetch_benchmark()
    if not bench_df.empty:
        bench_df.to_csv(DATA_DIR / "benchmark_000300.csv")

    # 3. 元信息
    meta = []
    for symbol, name in ALL_ETFS.items():
        fp = DATA_DIR / f"{symbol}_{name}.csv"
        if fp.exists():
            d = pd.read_csv(fp, index_col=0, parse_dates=True)
            meta.append({
                "symbol": symbol, "name": name,
                "start": d.index[0].strftime("%Y-%m-%d"),
                "end": d.index[-1].strftime("%Y-%m-%d"),
                "records": len(d),
            })
    meta_df = pd.DataFrame(meta)
    meta_df.to_csv(DATA_DIR / "etf_meta.csv", index=False)
    print("\n元信息已保存到 etf_meta.csv")
    print(meta_df.to_string(index=False))

    print("\n数据获取完成！")


if __name__ == "__main__":
    fetch_all()
