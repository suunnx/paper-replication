# T1 · End-to-End Inventory Management 论文复现

论文：**A Practical End-to-End Inventory Management Model with Deep Learning**，Qi et al., Management Science 69(2), 759–773 (2023)。[DOI 与作者补充材料](https://doi.org/10.1287/mnsc.2022.4564)。

本项目根据论文第 3、4 章及附录 B 独立实现算法，使用作者公开材料实际运行；按任务要求**不包含第五章实地试验**，也不收集额外数据。

**范围：方法与实验流程的部分复现。** 作者仅提供五条销量序列作为训练格式示例，缺少完整订单、库存、132 维特征及随机交货期数据。因此输出图表属于“公开小样本 + 已披露仿真假设”的对应实验结果，不是原论文全样本数值的精确重现。所有差异在 [逐图逐表对照](docs/REPRODUCTION_MAP.md) 中列明。

## 查看成果

- [实际运行报告](results/REPORT.md)
- [论文图表对应清单](docs/REPRODUCTION_MAP.md)
- [实现与假设](docs/METHODS.md)
- [Figure 2(b) 对应的库存成本比较](results/figures/figure_2b_costs.png)
- [附录敏感性结果](results/appendix_sensitivity.csv)

## 一键运行

建议 Python 3.12，在此目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --data-dir /path/to/replication_package_MS-FTBDA-20-01585 --suite all
```

Windows 使用 `.venv\Scripts\activate`。`--data-dir` 是已经解压的作者数据目录，应直接包含 `df_test.csv` 和 `E2E-model-data/`。本项目不会自动下载任何数据。首次安装需要网络获取软件依赖；实验运行只读取本地数据。

只运行主要离线比较：

```bash
python run.py --data-dir /path/to/replication_package_MS-FTBDA-20-01585 --suite main --output results_main
```

完整模式实际执行 19 个不同配置/目标的神经网络训练（相同配置缓存复用），以及一个 LightGBM 模型。日志、权重、输入指纹、配置、图表均由程序输出。耗时取决于机器。`--epochs` 可用于显式诊断运行，覆盖值会写入清单；提交结果使用配置的 60 轮上限和验证早停。

macOS 若缺少全局 OpenMP，入口会使用 scikit-learn 官方安装包内附带的库，不修改系统设置。先从 `run.py` 启动可避免直接导入 LightGBM 的库路径问题。

## 实现内容

| 部分 | 实现 |
|---|---|
| 定理 1 / Eq. (5) | 完美信息最优订货量标签，穷举验证 |
| Eq. (1)–(2) | 连续 30 天多次补货、先到货后需求、允许积压的库存模拟 |
| Figure 1 / Eq. (6) | LSTM 多分位数需求模块、VLT MLP、两层融合头和三输出损失 |
| Eq. (7)–(8) | Normal / Gamma 基库存策略 |
| Eq. (9), BM1 | 独立训练的总需求分位数预测，再作 newsvendor 决策 |
| Eq. (10), BM2 | 独立训练的每日需求预测，再作多周期决策 |
| E2E_GBM | LightGBM，使用销量统计特征学习相同最优标签 |
| Appendix B | 层数、隐藏维度、学习率、训练窗口量、b/h、初始化敏感性 |
| 3000 行测试文件 | 作者预测与 Normal/Gamma 订货量诊断；不伪造成本轨迹 |

BM1 不将每日分位数相加冒充总需求分位数；BM1/BM2 分位数由验证期成本选择，不用测试数据挑选。

## 文件组织

```text
run.py                     # 一键实验、校验和报告
configs/paper.json         # Table B.1 默认参数 + 明示的数据假设
src/inventory.py           # 公式、策略与库存更新
src/network.py             # MQRNN 风格全局/局部解码器与 E2E 网络
src/data.py                # 时间切分、特征与标签构造
src/experiment.py          # 独立模型训练、闭环库存评价
src/report.py              # 按论文编号绘图并生成报告
tests/test_methods.py      # 公式、时序、泄漏、梯度、模拟器测试
docs/                      # 方法说明和逐图逐表清单
results/                   # CSV、清单、报告和图片
```

`results/checkpoints/` 的模型与逐日轨迹在运行时生成，默认不提交到 GitHub。原论文 PDF、原始数据、虚拟环境、缓存也不提交。用 `results/data_manifest.json` 的 SHA-256 核对本地输入；公开并不自动等于可无限制再分发，数据取得方式仍以原论文入口为准。

## 验证

```bash
python -m unittest discover -s tests -v
```

包含 300 组最优标签与穷举对照、连续双订单问题与联合穷举对照、到货时间/成本规则、训练目标区间隔离、未来数据不改变当前特征、三个目标的梯度检查。每次完整运行还验证所有保存模型重载结果逐值一致，以及策略成本不低于相同条件下的完美信息 OPT。

训练数据太少，不能预期独立实现获得原文的优势幅度。本项目如实保留劣于基线的结果、分位数交叉和不单调的敏感性曲线。它展示当前材料能执行的工作，不声称完成缺失数据下的精确复现。
