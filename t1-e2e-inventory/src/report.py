"""Generate paper-numbered figures from measured local experiment outputs."""
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path('.mplconfig').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open() as handle:return list(csv.DictReader(handle))


def render(output):
    output=Path(output);figures=output/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):fig.savefig(figures/f'{name}.png',dpi=170,bbox_inches='tight');plt.close(fig)
    manifest=json.loads((output/'run_manifest.json').read_text());c=manifest['config']
    # Figure 1 is a code-native architecture diagram, not copied paper artwork.
    fig,ax=plt.subplots(figsize=(10,5));ax.axis('off')
    nodes={'history':(.12,.1,'180-day sales + SKU'), 'vlt_input':(.75,.1,'VLT fixture + SKU'),
       'demand':(.12,.43,'LSTM + global/local\nquantile decoder'), 'vlt':(.75,.43,'VLT MLP 50 / 20'),
       'fusion':(.45,.66,'Fusion 100 / 100\nreview then stock injected'),
       'order':(.45,.91,'Out 1: order'), 'forecast':(.1,.85,'Out 2: demand'), 'lead':(.85,.85,'Out 3: VLT')}
    artists={key:ax.text(x,y,text,ha='center',va='center',zorder=3,bbox=dict(boxstyle='round,pad=.5',fc='#dceff0',ec='#007f86')) for key,(x,y,text) in nodes.items()}
    for a,b in [('history','demand'),('vlt_input','vlt'),('demand','fusion'),('vlt','fusion'),('fusion','order'),('demand','forecast'),('vlt','lead')]:
        ax.annotate('',xy=nodes[b][:2],xytext=nodes[a][:2],arrowprops=dict(arrowstyle='->',color='#456',patchA=artists[a].get_bbox_patch(),patchB=artists[b].get_bbox_patch(),shrinkA=4,shrinkB=4),zorder=1)
    ax.set_title('Figure 1 — independent implementation of the paper topology',pad=20);save(fig,'figure_1_architecture')
    rows=read(output/'figure_2a_forecasts.csv');fig,axes=plt.subplots(3,1,figsize=(9,8),layout='constrained')
    crossing=0
    for sku,ax in enumerate(axes):
        values=[r for r in rows if int(r['sku_index'])==sku];x=[int(r['horizon_day']) for r in values]
        lo=np.array([float(r['q10']) for r in values]);hi=np.array([float(r['q90']) for r in values]);crossing+=int((lo>hi).sum())
        ax.fill_between(x,lo,hi,alpha=.2,color='#007f86',label='10–90% forecasts (not rearranged)')
        ax.plot(x,[float(r['observed']) for r in values],color='black',label='Observed sales')
        ax.plot(x,[float(r['q50']) for r in values],color='#d98b38',label='Median forecast')
        ax.set(title=f'SKU example {sku+1}',xlabel='Days ahead',ylabel='Units');ax.legend(fontsize=8,loc='upper right')
    fig.suptitle('Figure 2(a) — released-sales fixture; not original SKU examples');save(fig,'figure_2a_forecasts')
    main=read(output/'figure_2b_costs.csv');order=['OPT','E2E_RNN','E2E_GBM','BM1','BM2','Normal','Gamma']
    main=sorted(main,key=lambda r:order.index(r['policy']))
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    holding=[float(r['holding_cost']) for r in main];short=[float(r['stockout_cost']) for r in main]
    ax.bar(order,holding,label='Holding',color='#007f86');ax.bar(order,short,bottom=holding,label='Stockout',color='#d9a16c')
    ax.set(ylabel='30-day cumulative cost per SKU (mean over 5 SKUs)',title='Figure 2(b) — local fixture, continuous inventory simulation')
    ax.legend();save(fig,'figure_2b_costs')
    appendix=read(output/'appendix_sensitivity.csv') if (output/'appendix_sensitivity.csv').exists() else []
    for name in ['B2_layers','B3_hidden','B4_learning_rate','B5_training_fraction','B6_seed']:
        subset=[r for r in appendix if r['experiment']==name]
        if subset:
            with (output/f'table_{name}.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(subset[0]));w.writeheader();w.writerows(subset)
    subset=[r for r in appendix if r['experiment']=='B5_training_fraction' and r['policy']=='E2E_RNN']
    if subset:
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        percent=[100*float(r['value']) for r in subset]
        axes[0].plot(percent,[float(r['total_cost']) for r in subset],'o-',color='#007f86')
        axes[1].plot(percent,[float(r['training_seconds']) for r in subset],'o-',color='#d98b38')
        for ax in axes:ax.set_xlabel('Training-window fraction (%)')
        axes[0].set_ylabel('30-day cost / SKU');axes[1].set_ylabel('Training time (seconds)')
        fig.suptitle('Figure B.1 — training-data size sensitivity on five series');save(fig,'figure_B1_training_size')
    subset=[r for r in appendix if r['experiment']=='B2_cost_ratio']
    if subset:
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for policy in ['OPT','E2E_RNN','Normal','Gamma']:
            values=sorted([r for r in subset if r['policy']==policy],key=lambda r:float(r['value']))
            for ax,metric in zip(axes,['total_cost','holding_cost','stockout_cost']):
                ax.plot([float(r['value']) for r in values],[float(r[metric]) for r in values],'o-',label=policy)
                ax.set(xlabel='b/h',ylabel='30-day cost / SKU',title=metric.replace('_',' ').title())
        axes[-1].legend(fontsize=8);fig.suptitle('Figure B.2 — retrain labels and E2E for every cost ratio');save(fig,'figure_B2_cost_ratio')
    weights=json.loads((output/'vlt_weights.json').read_text())
    if len(weights)>1:
        limit=max(np.abs(r['weights']).max() for r in weights)
        fig,axes=plt.subplots(1,len(weights),figsize=(8,5),layout='constrained')
        for ax,item in zip(axes,weights):
            im=ax.imshow(item['weights'],aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit)
            ax.set(title=f'Seed {item["seed"]}',xlabel='VLT / SKU feature',ylabel='Layer-1 neuron')
        fig.colorbar(im,ax=axes,shrink=.7);fig.suptitle('Figure B.3 — VLT first-layer weights; fixed-VLT fixture');save(fig,'figure_B3_initialization')
    logs=read(output/'training_history.csv');fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    selected=[r for r in logs if r['run']==manifest['architecture_run']]
    for key in ['train_loss','validation_loss']:ax.plot([int(r['epoch']) for r in selected],[float(r[key]) for r in selected],label=key)
    ax.set(xlabel='Epoch',ylabel='Normalized joint loss',title='Main E2E training diagnostics');ax.legend();save(fig,'training_curve')
    table='\n'.join(f'| {r["policy"]} | {float(r["holding_cost"]):.2f} | {float(r["stockout_cost"]):.2f} | {float(r["total_cost"]):.2f} |' for r in main)
    report=f'''# 离线复现实验结果

本报告由实际运行输出生成。范围为论文第 3、4 章与附录 B；按用户要求不运行第 5 章。

**以下图表对应论文的实验问题和绘图形式，但数值来自公开五条销量序列与明确的库存仿真设定，不是论文全样本数值。** 不用作者提供的 3000 条测试预测来训练网络，不用虚构的未来需求替代真实销量。

## 运行范围

- 5 条真实销量序列，每条 457 个日编号；日期映射缺失，保留相对顺序。
- 历史窗口 180 天；固定 L={c['lead_time']}，取公开唯一 VLT 示例 4.9705 天向上取整。示例标识不能关联到销量 SKU，因此这是跨 SKU 统一的仿真假设，不是其真实交货期。
- 固定复核周期 R={c['review_period']}；训练库存覆盖水平为保护期历史平均需求的 0、0.5、1 倍。库存示例无法关联，不将构造状态说成观测库存。
- 按时间 60%/20%/20% 分区，去掉跨区目标窗口，7 天步长。训练/验证/测试样本数：{manifest['samples']}。同一时点三种库存状态并非三个独立 SKU。
- 只在最后 30 天进行连续库存仿真，库存与订单逐日传递，允许积压，先到货再扣需求。初始库存为过去 180 天均值乘以 L+R；无初始在途订单。
- E2E、BM1、BM2 分别训练，BM1 直接预测总需求分位数；BM2 预测每日需求。分位数按验证期库存成本选择，未使用测试结果挑选。固定 VLT 实验使用已知 L，而非网络 VLT 输出，避免伪装成随机 VLT 实验。

## Figure 2(b) 对应的策略比较

| 策略 | 持有成本 | 缺货成本 | 总成本 |
|---|---:|---:|---:|
{table}

单位是每 SKU 的 30 天累计成本，再对 5 个 SKU 等权平均。不同策略有各自连续库存轨迹；OPT 使用未来真实销量，是确定性下界。所有策略都在同一需求、初始库存、订货时点和到货时点下比较。

![Figure 2(b)](figures/figure_2b_costs.png)

这张表如出现 E2E 不优于其他模型，照实保留；小样本、构造库存、缺失特征、优化器和分布变化都可能影响结果，不通过挑选测试集或调整图轴来迎合原文。

## 其余产物

- `figures/figure_1_architecture.png`：按 Figure 1 组织的三输出结构。
- `figures/figure_2a_forecasts.png`：预先固定前三个 SKU 的多分位数预测，未挑选最优示例。
- `appendix_sensitivity.csv`、`table_B*.csv`：层数、隐藏维度、学习率、训练样本量、随机种子实验。
- `figures/figure_B1_training_size.png`：训练样本量—成本/耗时。
- `figures/figure_B2_cost_ratio.png`：b/h={1,3,9,19}；每种比值重新生成标签并训练。
- `figures/figure_B3_initialization.png`：两个随机种子的 VLT 第一层权重。
- `released_3000_diagnostics.csv`：作者预测及 Eq. (7)/(8) 在公开测试记录上的订货量误差诊断；没有声称得到其真实库存成本。
- `verification.json`：模型保存/重载逐值一致、完美信息 OPT 下界检查。
- `training_history.csv`：所有实际训练日志，包含未改善的实验。

Figure 2(a) 三条预测中出现 {crossing} 个 10%/90% 分位数交叉点。模型未加单调约束，绘图不偷偷排序；交叉反映拟合限制。

## 与原论文的差异

Table B.1 的隐藏维度、VLT 层、融合层、SGD 动量、学习率、batch size、初始化和 dropout 已落实。原文未充分说明的 LSTM/全局局部解码器细节、损失范数、辅助权重、学习率衰减形式和早停以代码与配置为准，属于独立实现选择。

原文写 Q=6，却同时列出均值和六个分位数；Figure 2(a) 又使用中位数。实现输出均值 + 10/50/60/70/80/90/95% 七个分位数，中位数用于可视化，BM 候选使用均值及原文列出的六分位数。没有把各日分位数之和当成总需求的分位数。

训练集不是原文的 24333 SKU / 前30天订单，敏感性样本比例也不是原文 SKU 子集比例。当前数据只能对同五个 SKU 的训练窗口做比例分析；没有统计显著性或跨数据集泛化主张。固定 VLT 的第一层权重图只验证不同初始化的模型行为，不能支持随机供应商交货期学习结论。

原始 `df_test.csv` 只提供预测、摘要和定义为历史需求的 `demand_hist`，缺少可靠的未来轨迹与真实到货。不能把其历史需求当未来需求来重建原文成本。原文 Figure 2(b)、Tables B.2–B.6 的全样本数值仍无法精确对齐，原因明确记录在 `docs/REPRODUCTION_MAP.md`。

本次没有收集额外数据；没有进行实地实验；没有将论文中的原始数值复制到“复现结果”列。
'''
    (output/'REPORT.md').write_text(report,encoding='utf-8')
