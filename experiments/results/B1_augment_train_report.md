# 训练报告：B1_augment

- 增强策略：mosaic_strong（{'mosaic': 1.0, 'copy_paste': 0.3, 'hsv_h': 0.03, 'hsv_s': 0.8, 'hsv_v': 0.5, 'degrees': 5.0, 'translate': 0.15, 'scale': 0.7, 'fliplr': 0.5}）
- 训练轮数：12，batch=8，imgsz=416，device=cpu
- 训练耗时：77.0 秒
- 数据集：D:\gg\yolov8-person-counter\data\dataset_runtime.yaml（见 data/DATA_CARD.md）

## 验证集指标（真实运行）

| 指标 | 数值 |
|---|---|
| mAP@0.5 | 0.9942 |
| mAP@0.5:0.95 | 0.7310 |
| Precision | 0.9990 |
| Recall | 0.9777 |

## 模型输出

- 最佳权重：`experiments\models\B1_augment_best.pt`
- 训练曲线：`experiments\results\B1_augment_curves.png`
- 原始 loss 记录：`runs\train\B1_augment\results.csv`
- 配置快照：`experiments/results/B1_augment_config.json`

> 所有数值来自本次真实训练与验证，未手工填写。