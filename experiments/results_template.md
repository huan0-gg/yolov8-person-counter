# A/B 实验记录模板

> 复制本文件填写，所有数值必须来自真实运行，禁止手写编造。
> 运行 `python scripts/run_ablation.py` 可自动生成完整报告。

## 1. 实验基本信息

- 实验日期：
- 数据集来源：COCO person 抽取 / 手机自采
- 训练集图片数：
- 验证集图片数：
- 设备：CPU / GPU（型号：）
- 预训练权重：
- YOLOv8 版本：

## 2. 基线 A 配置

- 模型：YOLOv8n
- 输入分辨率：640
- 置信度阈值：0.25
- 座位判定规则：overlap，阈值 0.30
- 跟踪去重：关闭
- 数据增强：官方默认

## 3. 结果对比表

| 方案 | 改动点 | mAP@0.5 | mAP@0.5:0.95 | FPS | 检测人数 | 占用座位 | 备注 |
|---|---|---|---|---|---|---|---|
| A_baseline | 默认配置 | | | | | | |
| B1_augment | 强 Mosaic + Copy-Paste | | | | | | |
| ctrl_no_mosaic | 关闭 Mosaic（对照组） | | | | | | |
| B2_track | ByteTrack 去重 | | | | | | |
| B3_center_roi | 中心点判定 | | | | | | |
| B_all | 三创新点合并 | | | | | | |

## 4. 3 条标准化创新点表述

按课程模板填写：针对【xx问题】在【yy模块】作【zz改动】，在【条件】下得到【B结果】，相对基线 A，提升 xxx。

1. 针对【】在【】作【】，在【】下得到【】，相对基线 A，提升/变化 xxx。
2. 针对【】在【】作【】，在【】下得到【】，相对基线 A，提升/变化 xxx。
3. 针对【】在【】作【】，在【】下得到【】，相对基线 A，提升/变化 xxx。

## 5. 证据文件清单

- [ ] `runs/train/A_baseline/weights/best.pt`
- [ ] `runs/train/B1_augment/weights/best.pt`
- [ ] `experiments/results/ablation_results.json`
- [ ] `experiments/results/ablation_results.csv`
- [ ] `experiments/results/ablation_report.md`
- [ ] Web 界面演示截图 / 录屏
