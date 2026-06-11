"""
Fetch defensive ETF data for the family-allocation sleeve.

These assets are candidates to replace the cash proxy in V2.9. The script is
kept separate from fetch_data.py so the original ETF research universe remains
reproducible.
"""
from __future__ import annotations

from pathlib import Path
import time

import akshare as ak
import pandas as pd


DATA_DIR = Path(__file__).parent / "data"

DEFENSIVE_ETFS = {
    "511880": "银华日利",
    "511360": "短融ETF",
    "511010": "国债ETF",
    "511260": "十年国债ETF",
}


def fetch_etf(symbol: str, name: str) -> pd.DataFrame:
    print(f"获取防守资产 {name}({symbol})...")
    df = ak.fund_etf_hist_em(
        symbol=symbol,
        period="daily",
        start_date="20180101",
        end_date="20251231",
        adjust="qfq",
    )
    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df


def main():
    DATA_DIR.mkdir(exist_ok=True)
    rows = []
    for symbol, name in DEFENSIVE_ETFS.items():
        try:
            df = fetch_etf(symbol, name)
        except Exception as exc:
            print(f"  失败: {symbol} {name}: {exc}")
            rows.append({"symbol": symbol, "name": name, "status": "failed", "error": str(exc)})
            continue
        if df.empty:
            print(f"  失败: {symbol} {name}: empty dataframe")
            rows.append({"symbol": symbol, "name": name, "status": "empty", "error": "empty dataframe"})
            continue
        path = DATA_DIR / f"{symbol}_{name}.csv"
        df.to_csv(path)
        print(f"  保存 {path.name}: {len(df)} 条, {df.index[0].date()} ~ {df.index[-1].date()}")
        rows.append({
            "symbol": symbol,
            "name": name,
            "status": "ok",
            "start": df.index[0].date().isoformat(),
            "end": df.index[-1].date().isoformat(),
            "records": len(df),
            "file": path.name,
        })
        time.sleep(0.5)

    pd.DataFrame(rows).to_csv(DATA_DIR / "defensive_etf_meta.csv", index=False)
    print("防守资产元信息已保存到 data/defensive_etf_meta.csv")


if __name__ == "__main__":
    main()
