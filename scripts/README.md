# Python Scripts

The Python files are grouped here. The recommended run order is below.

## 0. Data

1. `fetch_data.py`  
   Fetch base ETF and benchmark daily data into `data/`.

2. `fetch_defensive_data.py`  
   Fetch defensive ETF data, including short bond and money-market-like ETFs.

## 1. Core Engines

- `backtest_v2.py`  
  Main V2 backtest engine used by the final research pipeline.

- `backtest_engine.py`  
  Earlier engine kept for historical reports and legacy scripts.

## 2. Baseline And Diagnostics

3. `param_sweep.py`  
   Early parameter sweep for the V2 strategy.

4. `vol_sweep.py`  
   Volatility scaling experiment.

5. `dd_analysis.py`  
   Drawdown attribution analysis.

6. `rolling_window.py`  
   Earlier rolling-window validation pipeline.

## 3. V2.3 To V2.5 Research

7. `rolling_window_v2.py`  
   V2.3 rolling-window and 2025 OOS validation.

8. `execution_report_v2.py`  
   Trade ledger, position snapshots, yearly attribution, and rebalance advice.

9. `variant_research_v2.py`  
   V2.3 recovery and re-entry variant research.

10. `risk_return_sweep_v2.py`  
    V2.4 risk-return search.

11. `v24_candidate_report.py`  
    Focused V2.4-rc1 candidate report.

12. `asset_universe_research_v2.py`  
    V2.5 asset universe and core-asset research.

## 4. Family Strategy Research

13. `family_strategy_research_v2.py`  
    V2.6/V2.7 family allocation scan.

14. `family_strategy_stress_v2.py`  
    V2.8 stress tests for gold dependence and defensive-sleeve assumptions.

## 5. Final Deliverables

15. `final_v29_report.py`  
    Generate final V2.9 homework HTML/Markdown report, SVG charts, execution rules, NAV, metrics, and attribution tables into `output/final/`.

## Legacy Report Builder

- `main.py`  
  Earlier report builder kept for historical completeness. The final project path should use `final_v29_report.py`.
