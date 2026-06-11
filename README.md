# ETF 核心-卫星策略

面向普通家庭的 A 股 ETF 核心-卫星组合构建与回测系统。

## 策略概览

- **框架**：核心-卫星（技术分析驱动）
- **调仓**：月度，每次只用当日及之前的 OHLCV 数据（零未来信息）
- **核心**：沪深300(510300) + 中证500(510500) + 创业板(159915) + 红利低波(512890)
- **卫星**：14 只行业 ETF 每月动态筛选，按 60 日动量取前 2 只
- **风控**：MA20 日止损 + HS300 熔断(-2.5%) + 熊市核心 MA200 过滤 + ATR 止盈

## 回测结果 (2019.10 — 2024.12)

| 指标 | 策略 | 基准(HS300) |
|------|:--:|:--:|
| 年化收益 | **+5.05%** | +0.48% |
| 超额年化 | **+4.57%** | — |
| 最大回撤 | **-15.21%** | — |
| 夏普比率 | **0.158** | — |
| 交易流水 | 277 条（月均约 4.4 条） | — |

## 快速开始

```bash
# 安装依赖
pip install akshare pandas numpy

# 获取数据
python fetch_data.py

# 获取真实防守资产数据（货币/短债/国债 ETF，用于替代现金代理）
python fetch_defensive_data.py

# 运行回测（完整参数扫描）
python param_sweep.py

# 波动率缩放测试
python vol_sweep.py

# V2.3a 滚动窗口 + 2025 OOS 验证
python rolling_window_v2.py

# 生成执行层证据包与最新调仓建议
python execution_report_v2.py

# 研究 V2.3a 的恢复期变体
python variant_research_v2.py

# V2.4-rc1 候选报告
python v24_candidate_report.py

# V2.9 家庭可执行策略研究（黄金上限20%）
python family_strategy_research_v2.py

# V2.8 家庭策略压力测试
python family_strategy_stress_v2.py

# V2.9 最终候选报告、年度归因和执行规则
python final_v29_report.py
```

## 项目结构

```
etf_backtest/
├── backtest_v2.py       # V2.3 回测引擎（最终版）
├── fetch_data.py        # 数据获取（18只ETF + HS300基准）
├── fetch_defensive_data.py # 防守资产数据获取（货币/短债/国债ETF）
├── param_sweep.py       # 参数扫描
├── vol_sweep.py         # 波动率缩放测试
├── rolling_window_v2.py # V2.3a滚动窗口与2025 OOS验证
├── execution_report_v2.py # 交易流水/持仓/年度归因/调仓建议
├── variant_research_v2.py # 2024短板诊断与恢复期变体研究
├── risk_return_sweep_v2.py # 多目标风险收益搜索
├── v24_candidate_report.py # V2.4-rc1候选报告
├── family_strategy_research_v2.py # 家庭策略：防守仓比例/低仓位/目标风险扫描
├── family_strategy_stress_v2.py # 家庭策略：黄金收益/现金收益压力测试
├── final_v29_report.py # V2.9最终候选报告/归因/执行规则
├── dd_analysis.py       # 回撤归因分析
├── strategy_log.md      # 策略版本日志
├── output/              # 输出报告和图表
│   ├── 策略完整总结_V2.3.md
│   ├── v2_参数扫描报告.md
│   ├── v2_回撤归因分析.md
│   └── v2_波动率缩放测试.md
└── data/                # ETF日线数据（需运行 fetch_data.py）
```

## 版本演进

| 版本 | 年化收益 | 最大回撤 | 夏普 | 关键改进 |
|------|:--:|:--:|:--:|------|
| V2.1 | +6.37% | -19.40% | 0.173 | 纯技术分析，全市场选股 |
| V2.2 | +5.24% | -16.44% | 0.175 | +MA20止损 + HS300熔断 |
| V2.3 | **+5.05%** | **-15.21%** | **0.158** | **+熊市核心MA200过滤 + ATR止盈修复** |

## 策略规则（7条）

1. **市场状态**：HS300 MA200/MA50 → bull(90%)/range(70%)/bear(50%)
2. **核心仓**：4只等权，熊市过滤跌破MA200的品种（512890始终保留）
3. **卫星仓**：14只候选中选2只（close>MA200 且 20日动量>0，按60日动量排名）
4. **MA20止损**：卫星 close<MA20 → 当日清仓
5. **HS300熔断**：单日跌 >2.5% → 清卫星 + 降仓至50% + 冷却15日
6. **ATR止盈**：浮盈 >3×ATR(20) → 卖1/3
7. **偏离再平衡**：实际权重偏离目标 >3% 才交易

## License

MIT
