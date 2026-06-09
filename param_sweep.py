"""
V2 参数扫描 — 对比不同参数组合在 2019-2024 上的表现
不涉及 OOS，纯调参
"""
import pandas as pd
from backtest_v2 import StrategyParams, BacktestEngineV2
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"


def run_sweep():
    param_grid = [
        # (n_satellites, mom_rank, freq)
        (2, 60, "monthly"),
        (3, 60, "monthly"),
        (5, 60, "monthly"),
        (3, 20, "monthly"),
        (3, 120, "monthly"),
        (3, 60, "weekly"),
        (2, 120, "monthly"),
        (5, 20, "monthly"),
    ]

    results = []
    for n_sat, mom_rank, freq in param_grid:
        params = StrategyParams(
            n_satellites=n_sat,
            mom_rank=mom_rank,
            rebalance_freq=freq,
        )
        label = params.label()
        print(f"\n{'='*50}")
        print(f"运行: {label}")
        print(f"{'='*50}")

        engine = BacktestEngineV2(params)
        engine.run()
        engine.print_summary()
        m = engine.get_metrics()

        results.append({
            "variant": label,
            "n_satellites": n_sat,
            "mom_rank": mom_rank,
            "freq": freq,
            **m,
        })

    # 汇总表
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("excess", ascending=False)

    OUTPUT_DIR.mkdir(exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / "v2_param_sweep.csv", index=False)

    print("\n\n" + "=" * 90)
    print("参数扫描汇总（按超额年化排序）")
    print("=" * 90)

    cols = ["variant", "ann_return", "ann_bench", "excess", "max_drawdown", "sharpe", "calmar", "n_rebalances"]
    fmt = {
        "ann_return": "{:+.2%}", "ann_bench": "{:+.2%}", "excess": "{:+.2%}",
        "max_drawdown": "{:.2%}", "sharpe": "{:.3f}", "calmar": "{:.3f}",
    }

    print(results_df[cols].to_string(
        index=False,
        formatters={c: fmt.get(c, "{}").format for c in cols},
    ))

    return results_df


if __name__ == "__main__":
    run_sweep()
