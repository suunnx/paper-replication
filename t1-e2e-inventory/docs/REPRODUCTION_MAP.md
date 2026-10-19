# 论文图表—代码—输出对照

所有输出都由本项目代码计算。未将原文图片截取后称为复现，未把原文数值抄入“本次结果”。“对应实验”指方法/横轴/比较问题对应，数据规模和部分假设不同。

| 论文项 | 本项目入口/输出 | 当前状态与差异 |
|---|---|---|
| Figure 1 网络结构 | `src/network.py`; `figure_1_architecture.png` | 三输入分支/三个输出及 review、stock 后注入已实现。缺少原文132维特征；解码器细节独立实现 |
| Figure 2(a) 需求预测 | `figure_2a_forecasts.csv/png` | 真实销量前三个 SKU，均值及多分位数输出；不是原文示例 SKU |
| Figure 2(b) 策略比较 | `figure_2b_costs.csv/png` | 7 策略均运行，连续30天成本；五SKU/固定VLT/构造初始库存，不能对齐原文全样本数值 |
| Table B.1 超参数 | `configs/paper.json` | 明确默认值落实；未给出的细节列于 METHODS |
| Table B.2 融合层数 | `table_B2_layers.csv` | 1/2/3/4 层，其他参数固定，分别训练 |
| Table B.3 隐藏维度 | `table_B3_hidden.csv` | 30/40/50/60，分别训练 |
| Table B.4 学习率 | `table_B4_learning_rate.csv` | 0.0001/0.001/0.01，分别训练；统一60轮上限，未收敛风险不隐藏 |
| Table B.5 / Figure B.1 | `table_B5_training_fraction.csv`; `figure_B1_training_size.png` | 20/40/60/80/100%训练窗口；原文按SKU子集，本项目仅5个SKU，因此不能等同其样本量结论 |
| Figure B.2 b/h | `appendix_sensitivity.csv`; `figure_B2_cost_ratio.png` | 1/3/9/19，标签和E2E均重新训练；网格为本项目公开选择 |
| Figure B.3 / Table B.6 | `figure_B3_initialization.png`; `table_B6_seed.csv` | 种子7/17，比较VLT第一层权重与成本；固定VLT不能验证真实供应商泛化 |
| Figure 3、Tables 1–4 | 无 | 按用户要求跳过第5章，不运行倾向得分匹配或实地结果回归 |

## 为何不直接对3000行测试文件重画原文成本图

作者的 `README.docx` 将 `demand_hist` 定义为历史需求，`vlt` 定义为预测交货期，`OPT_pred`/`E2E_RNN_pred`/`gbm_pred` 是已计算的订货量。缺少可信的逐日未来需求、真实到货与库存事件链，以及完整预测分位数。因此仅能直接运行 Eq. (7)/(8) 并比较订货量，不能把历史需求硬当作未来需求计算“原文成本”。

这不是第五章造成的限制；第四章训练及全样本评价本身也需要未公开字段。代码可实现论文算法，运行必须明确当前数据允许的仿真假设。项目对这两件事分别报告。
