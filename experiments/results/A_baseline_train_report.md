# 训练报告：A_baseline

- 增强策略：default（{'mosaic': 1.0, 'copy_paste': 0.0, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'fliplr': 0.5}）
- 训练轮数：12，batch=8，imgsz=416，device=cpu
- 训练耗时：75.5 秒
- 数据集：D:\gg\yolov8-person-counter\data\dataset_runtime.yaml（见 data/DATA_CARD.md）

## 验证集指标（真实运行）

| 指标 | 数值 |
|---|---|
| mAP@0.5 | 0.9948 |
| mAP@0.5:0.95 | 0.8932 |
| Precision | 0.9877 |
| Recall | 0.9944 |

## 模型输出

- 最佳权重：`experiments\models\A_baseline_best.pt`
- 训练曲线：`experiments\results\A_baseline_curves.png`
- 原始 loss 记录：`runs\train\A_baseline\results.csv`
- 配置快照：`experiments/results/A_baseline_config.json`

> 所有数值来自本次真实训练与验证，未手工填写。