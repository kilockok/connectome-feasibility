# latent_state_v11b — 结论：选择性修正是否可以拯救 latent residual 模型

核心问题：在基础物理算子（精确 hard LIF）已经高度准确的条件下，
Transformer/GNN 学到的历史隐藏状态信息，能否转化为低噪声、选择性的
机制修正，并最终改善闭环神经动力学预测？

**答案：在当前效应量级与观测通道下，不能。两轮独立证据（v11
free-running hybrid、v11b sparse/gated）一致为阴性。**

## 1. 核心问题与 v11 阴性结果如何导出 v11b 假设

v11 Stage 1（commit af828f9）：B5 hybrid 在 testB 全部家族/mask 上输给
精确 B0（G1 FAIL），NULL 对照显示修正通道固有噪声底 0.012-0.018 RMS 超过
free-step 机制残差（0.008-0.013）。v11b 假设：如果修正**稀疏化、门控化**
（只在有足够可观测证据时干预），噪声底可被压到残差以下，net benefit
可能出现。Part A 先独立诊断 rollout 误差来源，Part B 在冻结协议下训练并
评估门控模型。

## 2. Part A 独立 rollout 诊断的主要结论（rollout_diagnostic.md）

- B0 的少量 one-step 错误**不会**长期放大：H=100 时 V RMSE 仅
  0.025-0.038（gain/adapt/stp），NULL 上永远 bitwise；50-59% 轨迹 100 步内
  无 spike 分歧；A4 因果分支显示单个 spike 翻转在 32 步内衰减到 ~0
  （teacher 动力学强收缩，无混沌放大）。
- B2/B5 闭环 rollout 灾难性发散（H=100 V RMSE 0.19-0.25，NULL 上同样
  发散）——问题是**闭环分布偏移 + always-on 修正噪声**，不是 teacher 侧
  放大，也不是 residual 在关键点的精度。
- A5 oracle 上限测量：**对现有 delta 预测器施加完美门控，只能比 B0 好
  0.0003-0.0008（all-mask V RMSE）** —— 门控现有 delta 几乎没有 headroom；
  真正缺的是 delta 质量（info/event 步），不是门控时机。

## 3. sparse/gated correction 是否降低 NULL 噪声？

部分降低但不达标：L_null 惩罚使 NULL 上修正 RMS 从 0.071 降到 0.048
（-32%），门控确实稀疏化（NULL 上 gate mean 0.018、active 0.8%）；但
M1 的 V 通道 NULL 噪声底是 0.0131，G1b 的 50% 门槛（0.0066）未达到，且
M3b 的 NULL V-RMSE（0.0424）反而高于 M1（0.0241）——稀疏但偶发的大幅
开门仍会越过放电阈值。G1b FAIL。

## 4. latent 相对于 no-latent gate 是否提供额外价值？

没有可确认的价值：M3-nolatent ≈ M3b（val 0.0212 vs 0.0223；testD
all-mask 三个家族上 nolatent 略优）。可用的门控信号本质上是
"距阈值距离 + |I_syn|" 等当前可观测状态。z-shuffle 仍显示 latent 携带
真实信息（info-mask +0.008，5/5 seeds），但该信息**不能转化为门控收益**。
（结论 A 成立 ≠ 结论 B 成立。）

## 5. 新模型是否在精确 hard LIF 上取得 one-step 净收益？

没有。testD（最终集，全新内插参数+种子），all-mask V RMSE / spike F1，
5 paired seeds 均值：

| model | gain | adapt | stp | null |
|---|---|---|---|---|
| M0 exact hard LIF | 0.0224/0.973 | 0.0156/0.973 | 0.0101/0.993 | 0.0000/1.000 |
| M1 v11 hybrid | 0.0279/0.929 | 0.0203/0.902 | 0.0229/0.939 | 0.0241/0.934 |
| M2 shrinkage | 0.0279/0.763 | 0.0203/0.767 | 0.0229/0.774 | 0.0241/0.787 |
| M3a gated | 0.0453/0.943 | 0.0347/0.925 | 0.0378/0.954 | 0.0396/0.948 |
| M3b gated+null | 0.0483/0.939 | 0.0365/0.919 | 0.0401/0.949 | 0.0424/0.945 |
| M3-nolatent | 0.0443/0.952 | 0.0341/0.933 | 0.0373/0.962 | 0.0391/0.959 |

全部模型全部家族 5/5 seeds 一致劣于 M0；M3 的 F1 退化也超过 0.01 条款。
G1a FAIL。唯一真实的正面成分：M3b 在它**确实捕获**的 teacher-spike event
步上，硬复位 V 预测近乎完美（event V RMSE 0.000-0.059 vs M0 0.095-0.192，
gain/stp）——但它引入的 spike 错误（943/1000 量级 gain）远多于修复（220）。

## 6. 是否有证据支持进一步研究长期动力学？

没有以当前模型类继续的依据。Part A 显示：B0 的长期漂移是缓慢累积而非
放大；现有修正通道在闭环中是纯损害；one-step 净收益为负的模型不可能在
闭环中获益。G1d 按冻结规则未运行。

## 7. 所有 Gate 结果及失败原因

| Gate | 结果 | 失败原因 |
|---|---|---|
| Part A | COMPLETE（诊断） | B0 误差不放大；修正通道闭环有害；oracle 门控 headroom ≈ 0 |
| G1a 基础精度 | FAIL | 所有模型 V-RMSE/F1 双线劣于 M0，5/5 seeds、B/D 两测试集一致 |
| G1b NULL 保护 | FAIL | 修正 RMS 降 32% 但未达 50% 门槛；NULL V-RMSE 退化超限 |
| G1c 事件纠正 | FAIL | spike 归因净收益为负（引入 >> 修复）；event-V 局部收益太小 |
| G1d 长期动力学 | NOT RUN（按规则） | G1a/G1b 失败 |
| C2 z-shuffle | latent 信息真实但小 | — |
| C3 no-latent gate | latent 无可确认额外价值 | — |
| M3c / M4 | NOT RUN（提前停止规则） | M2/M3a/M3b 在 val 上全部未过 G1a |

## 8. 当前实验不能证明的内容

- 不能证明 latent 信息不存在（z-shuffle 显示它存在，只是太小）；
- 不能证明更大效应量级/更丰富观测通道下选择性修正无效；
- 不能证明有任何门控设计的上限（只证明了：对当前 delta 质量，
  oracle 门控也无 headroom）；
- 不能将任何结果解释为真实果蝇机制。

## 9. 最合理的下一步方向（技术依据）

按 §十二 决策规则映射：oracle（O2）无法改善当前任务 → 应重新检查
teacher/base 定义、机制残差的实际作用和评价目标，而不是增大模型：

1. 若继续这条线，唯一有依据的改动是**提高效应量级区间**（v9 的
   effect-matched 低效应带使 free-step 残差 0.008-0.013 低于任何可训练
   修正通道的噪声底 ~0.013）；在 residual >> 噪声底的 regime 重跑 G1a。
2. 评价目标转向**放电统计层面**（rate/ISI 分布、群体事件时序）而非
   逐神经元逐步 V——Part A 显示 V/spike 单步误差在 teacher 动力学下会
   自然衰减，逐步精确本来就不是必需的。
3. 不建议：继续增加模型容量、继续在同一效应带内调门控、或以 rollout
   指标重新包装本轮阴性结果。

## 产物清单

results/latent_state_v11b/: protocol.md（冻结）、rollout_diagnostic.md、
model_design.md、audit/artifact_control.md、configs/（testd_data.pt、
m2_shrink.json）、results/{rollout,onestep,null_control,zshuffle,ablation,
training,checkpoints}、gates.json、conclusion.md。
代码：rollout_v11b.py、branch_v11b.py、oracle_v11b.py、data_v11b.py、
models/latent_hybrid_v11b.py、train_v11b.py、m2_select_v11b.py、
eval_v11b.py、attrib_v11b.py、test_v11b.py（单元测试全过）。
v1-v11 全部历史结果未修改（git 验证）。
