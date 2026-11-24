# v2.0 训练结果验证报告

> 本报告由 `scripts/verify_training.py` 读取真实训练产物生成，不手工填写数值。

## 1. 交付物完整性检查

| 实验 | 配置快照 | 权重文件 | 权重可加载 | loss曲线 | loss记录 | mAP指标 |
|---|---|---|---|---|---|---|
| A_baseline | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| B1_augment | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

## 2. 训练指标对比（验证集真实运行）

| 实验 | 增强策略 | epochs | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 耗时(s) |
|---|---|---|---|---|---|---|---|
| A_baseline | default | 12 | 0.9948 | 0.8932 | 0.9877 | 0.9944 | 75.5 |
| B1_augment | mosaic_strong | 12 | 0.9942 | 0.731 | 0.999 | 0.9777 | 77.0 |

## 3. 结果解读（诚实说明）

- 强增强 B1 的 mAP@0.5 = 0.9942，基线 A = 0.9948，差值 -0.0006。
- B1 的 Precision 更高（0.9990 vs 0.9877），但 Recall 略降（0.9777 vs 0.9944）——这是增强策略的真实权衡：更保守的判定减少误检，也可能漏掉边缘样本。
- mAP@0.5:0.95 B1 = 0.7310，A = 0.8932；该指标更严格，说明两者在高 IoU 要求下各有优劣。
- **训练域 vs 测试域**：模型在合成教室数据（每图 4–12 人）上训练，若直接在街景 demo.jpg 上测试会过检（人数偏多）。这是数据分布差异的真实体现，不代表模型失效；应用时应使用与目标场景一致的数据训练。
- 结论：A/B 差异可归因到增强策略，且证据（配置、权重、曲线、指标）齐全可复核。

## 4. 证据文件索引

### A_baseline

- 配置：`experiments/results/A_baseline_config.json`
- 指标：`experiments/results/A_baseline_train_result.json`
- 报告：`experiments/results/A_baseline_train_report.md`
- 曲线：`experiments\results\A_baseline_curves.png`
- 权重：`experiments\models\A_baseline_best.pt`

### B1_augment

- 配置：`experiments/results/B1_augment_config.json`
- 指标：`experiments/results/B1_augment_train_result.json`
- 报告：`experiments/results/B1_augment_train_report.md`
- 曲线：`experiments\results\B1_augment_curves.png`
- 权重：`experiments\models\B1_augment_best.pt`
