"""
V2.3 + 波动率缩放 参数扫描
目标波动率: 10%, 12%, 15% + 不做缩放（baseline）
"""
import pandas as pd
from backtest_v2 import StrategyParams, BacktestEngineV2
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def run_vol_sweep():
    targets = [None, 0.10, 0.12, 0.15]  # None = 不做缩放
    results = []

    for tv in targets:
        label = f"vol{int(tv*100)}%" if tv else "no_vol_scale"
        params = StrategyParams(
            n_satellites=2,
            mom_rank=60,
            rebalance_freq="monthly",
            target_vol=tv,
        )
        print(f"\n{'='*55}")
        print(f"运行: {label}")
        print(f"{'='*55}")

        engine = BacktestEngineV2(params)
        engine.run()
        engine.print_summary()
        m = engine.get_metrics()

        results.append({"variant": label, "target_vol": f"{tv:.0%}" if tv else "无", **m})

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("sharpe", ascending=False)

    OUTPUT_DIR.mkdir(exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / "v2_vol_sweep.csv", index=False)

    print("\n\n" + "=" * 80)
    print("波动率缩放扫描汇总")
    print("=" * 80)
    cols = ["variant", "target_vol", "ann_return", "excess", "max_drawdown", "sharpe", "calmar", "ann_vol", "n_trades"]
    fmt = {
        "ann_return": "{:+.2%}", "excess": "{:+.2%}",
        "max_drawdown": "{:.2%}", "sharpe": "{:.3f}", "calmar": "{:.3f}",
        "ann_vol": "{:.2%}",
    }
    print(results_df[cols].to_string(
        index=False,
        formatters={c: fmt.get(c, "{}").format for c in cols},
    ))

    return results_df


if __name__ == "__main__":
    run_vol_sweep()
