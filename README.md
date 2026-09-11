# Connectome Dynamics Feasibility Study

最小可行性验证：**神经网络模型能否从 LIF simulator 的时序数据中学到动力学？connectome 结构约束是否能提升长期 rollout 稳定性与对未见刺激的泛化能力？**

本阶段不做完整电子果蝇；N = 100（冒烟测试）/ 1000（正式实验），placeholder 稀疏有向 connectome（环形几何 + Dale 定律，可替换为 FlyWire 子图）。

## 环境

任选其一（`device.py` 自动选 XPU > CUDA > CPU，全代码无 `.cuda()` 硬编码）：

```bash
# NVIDIA GPU (CUDA 12.8, RTX 4090/5090)
pip install torch --index-url https://download.pytorch.org/whl/cu128
# Intel Arc A750 (XPU)
pip install torch --index-url https://download.pytorch.org/whl/xpu
# CPU-only
pip install torch

pip install -r requirements.txt
```

- float32（phase 1 不用混合精度）
- 云端 quickstart：见 [CLOUD.md](CLOUD.md)

## 快速开始

```bash
python test_lif.py --scale small          # Step 1: LIF simulator sanity
python generate_dataset.py --scale small --inspect 100   # Step 2: 数据合理性
python run_all.py --scale small           # 完整小规模 pipeline (Step 6)
python run_all.py --scale full --unroll 8 # 正式实验 (Step 7-10)，含多步微调
```

单步执行：

```bash
python train.py --model connectome --scale full            # phase 1: one-step
python train.py --model connectome --scale full --unroll 8 # phase 2: 多步展开微调(可选)
python evaluate.py --scale full                            # one-step + rollout + perturbation + 全部图
python plots.py --scale full                               # 仅重画图
```

## 结构

```
feasibility/
├── config.py          # 全部超参数（seed、N、T、K、LIF、connectome、训练、loss、rollout）
├── device.py          # XPU > CUDA > CPU 选择器
├── connectome.py      # edge_index/edge_weight 表示 + placeholder 生成（可换 FlyWire）
├── lif.py             # 离散时间 batched LIF simulator（刺激/沉默/edge lesion/状态分支）
├── dataset.py         # 种子化轨迹生成 + OOD 刺激 split + 时间窗切片
├── generate_dataset.py# 预生成并缓存 val/test_seen/test_ood
├── models/
│   ├── gru.py             # flatten baseline
│   ├── transformer.py     # vanilla spatial transformer（时序编码器+神经元 embedding）
│   ├── connectome_transformer.py  # attention 受 connectome 约束（mask + log|W| + 类型 bias）
│   └── gnn.py             # edge-weighted message passing baseline
├── metrics.py         # 统一 loss（V MSE + spike BCE(pos_weight) + R MSE）与指标
├── train.py           # phase 1 one-step 训练 + phase 2 多步 unroll 微调
├── rollout.py         # 自回归 rollout（真实未来刺激作为已知输入）
├── perturbation.py    # neuron silencing / edge lesion 的 GT-vs-model Δ 对比
├── evaluate.py        # 汇总评估 -> metrics_{scale}.json / .csv
├── plots.py           # 8 张图 -> results/figures/
└── run_all.py         # 一键 pipeline
```

## 实验设计要点

**任务**：给定 `X[t-K:t], U[t-K:t]`（K=32/16），预测 `X[t+1]`；状态 = `[V, spike, refractory]`，不只预测 spike。

**OOD split（关键）**：train 刺激神经元 `[0, 0.8N)`，val `[0.7N, 0.9N)`，test_OOD `[0.8N, N)`。测试集中的神经元在训练中**从未被直接刺激**，模型必须利用 connectome 推断传播。

**可复现**：每条轨迹由整数种子完全决定（`cfg.traj_seed(split, idx)`），训练集无需落盘；val/test 生成一次并缓存（`results/cache/`），所有模型看到完全相同的数据。connectome 种子固定，所有模型/数据共享同一张图。

**Connectome Transformer 约束**：`score(i,j) = Q_i·K_j/√d + α_h·log(1+|W_ji|) + type_emb[type_j]`（仅当 edge j→i 存在），无边则 `-1e4`，自环恒允许。当前为 dense additive bias（N≤1000 验证可行，SDPA fused kernel 不显式物化 N²）；**N≫2000 时必须改为稀疏/邻域 attention**（已知限制）。

**rollout**：真实 context + 真实未来刺激（外部输入已知），状态自回归反馈。状态组合与 simulator 对齐：spike→V=V_reset、R=1，R 高时 V 保持 reset（hard reset，所有模型一致）。spike 判定阈值分别在 val one-step（F1）与 val rollout（F1+pop_sim−rate_err）上调节。

**多步微调（phase 2，可选）**：one-step 训练的 rollout 存在 exposure bias（活动逐渐熄灭）。`--unroll U` 用 straight-through 状态组合做 U 步展开 loss，直接优化多步一致性；保存为独立 `_ms_` checkpoint，evaluate 优先加载。

**naive baseline**：`x[t+1] = x[t]` 在 one-step 与 rollout 中均作参照（Milestone 0 的判定标准）。

## 结果（full scale, N=1000, T=256）

> 待填入。跑完 `python run_all.py --scale full --unroll 8` 后更新。

### 四个核心问题

1. **Transformer 能否拟合 LIF one-step dynamics？** 待答。
2. **Connectome constraint 是否提高 OOD stimulus generalization？** 待答。
3. **Connectome constraint 是否提高长期 rollout stability？** 待答。
4. **模型能否预测 unseen lesion / silencing effect？** 待答。

（即使答案为否，也保留完整实验结果于 `results/`。）

## Milestone 判定

- M0：显著优于 naive baseline 且学会 one-step 动力学
- M1：Connectome TF 在 unseen stimulus 上明显优于 Vanilla TF / GRU
- M2：Connectome TF 的 rollout error 增长更慢
- M3：能预测 silencing / edge lesion 的 downstream effect

## 已知限制

- connectome 为 placeholder（环形几何随机图），非真实 FlyWire 子图；`connectome.py` 的表示（edge_index/edge_weight/neuron_type/i_bias）支持直接替换
- attention bias 为 dense [N,N]：N≫2000 需稀疏化
- simulator 用 dense W matmul：N≫2000 需 sparse mm
- GPU 上 matmul 存在微小非确定性，轨迹在 bit 级可能不完全一致（种子协议保证分布级可复现）
