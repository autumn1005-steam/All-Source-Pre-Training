# RCSR-TKDE 风险控制源路由增强版

本版本在原 TempoFound-RASP 工程上新增 **Risk-Calibrated Causal Source Routing (RCSR)**。
它修复了 legacy `source_ranked` 在 unseen target 上新建 target expert、从而没有真正组合 source experts 的迁移路径；
RCSR 保留已训练 source expert bank，以 RASP source affinity 为全局先验，使用 source-response-aware 的事件级共享路由器进行组合，并通过 routing entropy、identity abstention 和 counterfactual regret regularization 抑制负迁移。

新增入口：`configs/rcsr_fair_3seed.yaml`、`experiments/run_rcsr_smoke.sh`、`experiments/run_rcsr_fair_3seed.sh`。
完整实验计划见 `docs/TKDE_NEXT_STAGE_RCSR_CN.md`。

---

# TempoFound-RASP 统一研究代码

本项目按照论文当前主线把两个原仓库整理为一个可运行工程：先用
Forward-Pair 在目标域训练期内部筛选源域，再用 TempoFound 对所选源域
进行跨数据集时序图预训练，最后在目标域完成冻结编码器迁移评测。新版 RASP
在保留原版基线的同时，引入风险校准组合搜索、目标亲和度预算、冲突梯度投影、
领域对抗学习和固定负例 MRR 协议。

## 关键保证

- 默认主流程只运行论文中的 Forward-Pair，不再误跑旧版 `proposed` 或
  leave-one 选择器。
- K=4 与 K=6 共用一份源码，只切换 YAML 配置。
- 代理阶段只读取目标训练期；训练期按时间顺序切成前 80% 校准和后 20%
  探测，目标验证集和测试集不参与选源。
- 每个 epoch 的源域总更新次数严格不超过 160，源域较小时会将剩余预算
  重新分配给其他源域。
- 新实验默认使用论文中的归一化 `paper_sincos` 时间编码；没有版本元数据
  的旧 TempoFound checkpoint 自动按 `legacy_cos` 加载。
- 旧版只有 4 个自适应任务权重时会确定性迁移：原值不变，新增 NPP 权重置零。
- v2 的 5 权重检查点会显式重排，旧领域分类权重不会被误当作 EDC；新增 TIP
  权重置零，新检查点记录目标名称并使用格式版本 3。
- Forward-Pair 必须得到 K 个单源和 K-1 个锚点配对，共 `2K-1` 个有效候选；
  缺失、重复、NaN、目标/种子不一致或配置摘要过期都会立即报错。
- RASP 使用多个连续时间窗口估计代理分数不确定性，并在固定候选预算内搜索
  非锚点组合；最终训练按目标亲和度分配同一全局预算并记录梯度冲突统计。
- 评测负例按查询固定、排除查询时刻前已观测边，主指标为 MRR。

## 安装与自检

```bash
cd TempoFound-TKDE
python -m pip install -e '.[test]'
python -m pytest -q
bash experiments/run_smoke.sh
bash experiments/run_rasp_smoke.sh
```

冒烟测试使用自动生成的微型数据，只验证代码链路，严禁写入论文结果表。

## 正式 RASP K=4 实验

```bash
nohup env \
  PROCESSED_DIR=/root/autodl-tmp/temp/processed \
  OUTPUT_ROOT=/root/autodl-tmp/temp/server_runs/tempofound_rasp_k4 \
  DEVICE=cuda:0 \
  bash experiments/run_rasp_k4.sh \
  > /root/autodl-tmp/temp/rasp_k4.log 2>&1 &
```

先完成服务器验收与数据预检，再启动正式任务。RASP、全源、原版
Forward-Pair 必须使用同一全局批次数上限、相同种子和同一固定负例文件。

## 正式 K=4 实验

```bash
nohup env \
  PROCESSED_DIR=/root/autodl-tmp/temp/processed \
  OUTPUT_ROOT=/root/autodl-tmp/temp/server_runs/tempofound_forward_pair_k4 \
  DEVICE=cuda:0 \
  bash experiments/run_k4_primary.sh \
  > /root/autodl-tmp/temp/k4.log 2>&1 &
```

默认目标为 MOOC、Wikipedia、Reddit，随机种子为 0、1、2、3、42。对每个
目标，配置自动从五数据集池中排除目标，得到四个候选源域。

## K=6 扩展实验

```bash
nohup env \
  PROCESSED_DIR=/root/autodl-tmp/temp/processed \
  OUTPUT_ROOT=/root/autodl-tmp/temp/server_runs/tempofound_forward_pair_k6 \
  DEVICE=cuda:0 \
  bash experiments/run_k6_scalability.sh \
  > /root/autodl-tmp/temp/k6.log 2>&1 &
```

七数据集池为 Wikipedia、Reddit、MOOC、UCI、Enron、LastFM、SocialEvo。
MOOC 使用 0、1、42 三个种子；Wikipedia 与 Reddit 使用 seed 42 诊断。

## 输出结构

每个 `target/seed*` 目录包含：

- `candidates/`：单源和锚点配对的 checkpoint 与训练期代理分数；
- `selection/selection.json`：锚点、最佳配对、增益、阈值和最终源集合；
- `final/forward_pair/` 与 `final/all_sources/`：同预算最终预训练模型；
- `transfer/`：冻结编码器迁移结果；
- `summary.json`：Forward-Pair 与全源 AP 对照。

RASP 运行额外记录 `selection/domain_weights.json`、逐候选风险效用、固定负例
排名结果，以及 checkpoint 中的源域预算与梯度冲突诊断。

服务器验收步骤见 `docs/SERVER_ACCEPTANCE_CN.md`，论文模块和文件的逐项
对应见 `docs/PAPER_CODE_MAPPING.md`。

## HRCSR v2（2026-09-05）

针对首轮 RCSR 在 Wikipedia 为正、Reddit 3/3 seeds 为负的结果，本包新增 Hierarchical Risk-Controlled Source Routing (HRCSR) 诊断实现：

- 修复 v1 regret 实际为 batch-level scalar 的问题，新增 `transfer_regret_mode: event_pairwise`；
- 新增 target-train-only bootstrap LCB gate，可在 target 整体迁移风险高时收缩 source routing；
- 新增 strict ablation variants，统一 historical collision-free negatives；
- 保留 v1 `rcsr` 和 legacy variants 用于复现旧结果；
- 详细实验顺序见 `docs/HRCSR_V2_NEXT_EXPERIMENT_CN.md`。

推荐先运行：

```bash
PROCESSED_DIR=/path/to/processed \
OUTPUT_ROOT=/path/to/results/hrcsr_strict_ablation \
./experiments/run_hrcsr_strict_ablation.sh
```

## v3：独立时序安全校准

新增 `docs/OOS_SAFETY_CALIBRATION_V3_CN.md`。本版本针对 HRCSR v2 在 Reddit 上稳定负迁移及 target gate 半开问题，引入 chronological target-train optimization/calibration split、MRR-aligned reciprocal-rank calibration、path-neutral target fitting，以及同一模型上的 identity/route/hard/soft gate sweep。推荐先运行 `experiments/run_calibrated_safety_ablation.sh`，不要直接扩展 MOOC。

## v3.1 冻结验证扩展（2026-09-06）

本版本**不修改 OOS-HRCSR v3 模型参数或算法**，只增加顶刊证据所需的验证工具：

- `experiments/run_oos_safety_5seed_3target.sh`：Wikipedia/Reddit/MOOC × 5 seeds 的严格冻结验证；
- `scripts/analyze_oos_safety.py`：自动计算 gate decision accuracy、safety violation、missed opportunity、selector regret、LCB-vs-future-gain Spearman，以及 hard/identity 相对 strict_ranked 的差值；
- `docs/OOS_HRCSR_V3_VALIDATION_STAGE_CN.md`：下一阶段判定标准。

推荐先对已经完成的 v3 结果运行分析器，再启动 5-seed × 3-target 扩展。
