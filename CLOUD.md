# 云服务器训练 Quickstart（NVIDIA GPU）

以 Ubuntu 22.04/24.04 + RTX 4090/5090 为例（AutoDL / Vast.ai / RunPod 通用）。

## 1. 克隆

```bash
git clone https://github.com/kilockok/connectome-feasibility.git
cd connectome-feasibility
```

## 2. 环境

镜像自带 PyTorch（CUDA 版）的话只需补依赖：

```bash
pip install -r requirements.txt
```

纯净镜像（无 torch）或要确认 sm_120 (5090) 支持：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

验证 GPU 可见：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. 跑实验

```bash
# 完整 full-scale pipeline：sanity -> 数据检查 -> 4 模型 phase-1 -> evaluate
nohup python run_all.py --scale full > logs_full.txt 2>&1 &

# 如需 phase-2 多步展开微调（推荐，修复 rollout 熄灭问题）
nohup python run_all.py --scale full --unroll 8 > logs_full.txt 2>&1 &

tail -f logs_full.txt   # 实时监控
```

`run_all.py` 依次训练 gru / transformer / connectome / gnn，最后统一评估
（one-step + rollout + perturbation），结果落在 `results/`：

- `history_<model>_full.csv` — 每 epoch 训练/验证指标
- `metrics_full.csv` / `.json` — 最终对比表
- `results/figures/` — 全部图

预计耗时（单卡 4090/5090）：phase-1 约 10–20 分钟，加 unroll 与评估共 30–40 分钟。

## 4. 常用单步命令

```bash
python train.py --model connectome --scale full              # 单模型 phase-1
python train.py --model connectome --scale full --unroll 8   # phase-2 微调
python evaluate.py --scale full --models connectome gnn      # 重评估指定模型
python plots.py --scale full                                 # 仅重画图
```

## 备注

- 轨迹由整数种子现场生成，无数据集文件需要下载。
- 断线不丢进度：用 `nohup`/`tmux` 包一层即可；checkpoint 在 `results/checkpoints/`。
- 多 seed 复现：`--seed <int>`。
