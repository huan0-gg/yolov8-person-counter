# 更新日志（Changelog）

## v2.0.0 — 2026-11-24

> 主题：**从"能演示"升级为"能证明"**。保持原项目目标（YOLOv8 人员统计 + 座位占用检测 +
> A/B 创新实验）不变，补齐老师反馈所指出的三项短板：数据支撑、训练闭环、展示与文档。

### 新增内容

#### 1. 数据与数据说明（解决"没有数据支撑"）
- `scripts/build_dataset.py`：可复现数据集构建器（固定 seed=42），以真实空教室图为背景、
  从 COCO 人物样例裁剪实例，按透视规律合成教室人群图并**精确生成标注**，
  另并入真实图片对照样本。一键重建：`python scripts/build_dataset.py --seed 42 --total 100`。
- `data/DATA_CARD.md`：完整数据卡——来源构成、生成流程、统计摘要、用途边界（明确合成数据
  不能作为真实场景精度结论）、与 1.0 的区别。
- `data/dataset_stats.json`：真实统计（84 训练 / 22 验证 / 801 person 框 / 4 个人物实例）。
- `experiments/results/dataset_samples.jpg`：标注样例拼图，供目视抽检与展示。

#### 2. 训练闭环（解决"训练脚本写了但没跑通"）
- `scripts/train.py` 重写：训练前保存**配置快照**、训练中记录 loss、训练后**导出权重副本**
  到 `experiments/models/`、生成 **loss/mAP 曲线图**与训练报告、独立验证集评估。
- `src/utils.py` 新增 `resolve_dataset_yaml()`：运行时生成绝对路径配置，修复不同工作目录
  下数据集路径解析错误的问题（仓库内模板保持不变，可移植）。
- `src/evaluate.py`：mAP 评估增加无 CUDA 时 `device=auto → cpu` 的降级，修复 A/B 实验中
  `model.val()` 报错。
- **真实训练产物**（均由脚本实际运行产出，非手工填写）：
  - `experiments/models/A_baseline_best.pt`、`B1_augment_best.pt`
  - `experiments/results/A_baseline_curves.png`、`B1_augment_curves.png`（loss 从 ~1.9 降到 ~0.6）
  - `experiments/results/*_config.json`、`*_train_result.json`、`*_train_report.md`
  - `experiments/results/training_verification.md`：交付物完整性检查（6 项全部 ✅）与诚实解读。

#### 3. 结果验证与分析解释
- `scripts/verify_training.py`：读取真实训练产物，校验配置/权重/曲线/指标齐全并实际加载权重，
  生成汇总与解读（含 Precision/Recall 权衡、训练域 vs 测试域差异的诚实说明）。
- `scripts/run_ablation.py`：现在使用**真实训练出的 B1 权重**跑 A/B，mAP 列不再是 0/待测。
  实测：基线 A mAP@0.5=0.8132 → B1=0.9942（+22.3%）。

#### 4. 展示效果优化
- `app.py` 新增 **"训练与验证 v2.0"** 标签页：展示训练指标对比表、验证报告、两组训练曲线、
  数据集标注样例，做到"打开界面就能看到训练证据"。
- 保留原有图片 / 视频 / 摄像头三路检测与 A/B 方案切换、调优面板。

#### 5. 依赖与配置修复
- `requirements.txt`：`matplotlib>=3.9`（3.8 与 numpy 2 二进制冲突，实测报错后修复）。
- `.gitignore`：放行 `experiments/models/*.pt`（训练产物需入库供复核），仍排除预训练权重。

### 使用方法变化
- 新增两步：`python scripts/build_dataset.py`（构建数据）→ `python scripts/train.py ...`（训练）。
- 完整命令见 README 第 5 节与第 8 节"2.0 完整复现流程"。

### 数据来源
见 `data/DATA_CARD.md` 与 README 第 6 节。核心：COCO 公开样例 + 项目自采教室截图 + 透视合成，
固定 seed 可复现；合成数据仅用于工程演示与流程验证，不代表真实场景精度。

### 展示效果
- Web 界面新增训练证据页；README 置顶真实指标表与训练曲线；A/B 报告数值全部来自真实运行。
- 所有结论均可追溯到 `experiments/results/` 下的原始 json/csv/png/md 证据文件。

### 兼容性
- 原 1.0 功能（检测、计数、透视网格、A/B 切换、Gradio 界面）全部保留，无破坏性改动。
- 改进版说明见 `experiments/IMPROVEMENTS.md`（1.0→改进版的框/召回修复）。
