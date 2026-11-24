# 更新日志（Changelog）

## v2.2.0 — 2026-11-24

> 主题：**把"要管的区域"画出来，把"误检/漏检"量出来**。
> 新增指定区域标注能力；检测后处理重做（多尺度融合 + 嵌套框去重 + 放宽坐姿宽高比）；
> 补一套检测级评测与 31 个单元测试，所有改动用同一批数据做了前后对比。

### 新增

#### 1. 指定区域标注（`src/region.py` + `docs/REGION.md`）
- 配置里用**归一化多边形或矩形**指定监测区域，结果图自动：2px 描边 + 10% 填充 +
  四角标记 + 标签条（区域名 + 区域内人数）。
- 区域内人员框黄色、区域外灰色并带 `OUT`；右上角面板新增 `Inside ROI / Outside`。
- 可选 `filter_outside: true`（区域模式）直接丢弃区域外检测；可选 `dim_outside` 压暗区域外。
- 多区域支持：一个框只归属覆盖面积最大的区域，不重复计数；接口返回
  `result["regions"]` / `inside_persons` / `outside_persons`。
- `scripts/preview_region.py`：只渲染区域，秒级确认标定是否正确。

#### 2. 检测后处理重做（`src/detector.py`）
- **多尺度融合**：同图在 960 / 1152 各跑一次，WBF 按置信度加权融合，互补小目标。
- **嵌套框去重**：新增 IoS（交集占较小框面积比）判据，抑制"只框住头/腿"的碎片框。
- **放宽坐姿宽高比**：`roi.aspect_range` 由 `[0.5, 4.0]` → `[0.30, 5.00]`（实测漏检主因）。
- 新增可选 **切片推理**（2×2 + WBF）与 **Soft-NMS**，并附实测结论：在本项目数据上两者
  都会显著增加误检，故默认关闭，仅保留为可复现的实验开关。
- 小图保护：`tiling.min_image_side` 默认 900，最短边不足时自动放弃切片。

#### 3. 评测与测试
- `src/metrics.py`：TP/FP/FN 贪心匹配、P/R/F1、单类 AP@0.5、计数 MAE。
- `scripts/eval_detection.py`：7 个变体在同一批 22 张验证集上跑，输出
  `experiments/results/detection_eval.json`。
- `scripts/render_eval_report.py`：由 JSON 生成 `docs/EVAL_v2.2.md`，文档数字与代码不会脱节。
- `scripts/make_region_demo.py`：生成"更新前 / 更新后"区域标注对照图。
- `tests/`：31 个用例覆盖区域几何、框融合、切片窗口、指标口径。

### 实测（22 张验证集 / 179 个标注，同一权重，置信度 0.12，IoU 0.5）

| 版本 | Precision | Recall | F1 | AP@0.5 | 误检 FP | 漏检 FN |
|---|---|---|---|---|---|---|
| v2.1 基线 | 0.8692 | 0.6313 | 0.7314 | 0.6216 | 17 | 66 |
| **v2.2 默认** | 0.8788 | **0.8101** | **0.8430** | **0.7861** | 20 | **34** |
| **v2.2 区域模式** | **0.9329** | 0.7765 | **0.8476** | 0.7613 | **10** | 40 |

- 默认配置：漏检 −48.5%、F1 +15.3%、AP@0.5 +26.5%，误检 +3（每图 +0.14）。
- 区域模式：误检 −41.2%、精度 +7.3%、漏检 −39.4%；代价是区域边界上的人可能被判为区域外。

### 变更
- 示例图改为 `assets/ai/` 下的 AI 生成场景图；界面新增「切片高精度推理 / 标注指定区域 / 区域外压暗」三个开关，
  统计明细新增推理模式与区域人数。
- `scripts/evaluate.py` 未改动，仍走 ultralytics 官方 mAP 口径，与 v2.2 检测级口径分开报告。
- 密集教室示例在 v2.2 默认配置下为 **24 人 / 占用 10 座**（v2.1 为 23 / 11），差异已在 README 与本节注明。

## v2.1.0 — 2026-11-24

> 主题：**补上"项目传播"这一环**。按课程"用 rea-video 拆参考片 → 用 OpenMotion 生成可编辑视频"
> 的要求，交付一支 45 秒项目短片，并把视频变成"数据驱动、可重制"的工程产物；
> 同时修掉三个影响运行与展示一致性的真实缺陷。

### 新增内容

#### 1. 45 秒项目短片（问题 → 方案 → Demo → A/B → GitHub CTA）
- `docs/video/promo_45s.mp4`：1280×720 / 30fps / 45.0s / H.264，9 个镜头、7 次转场。
- `docs/video/cover.gif`：640×360 / 50 帧 / 约 0.1 MB，README 顶部封面动图。
- `docs/video/ui.png`：Gradio 界面真实截图（无头浏览器抓取，非设计稿）。
- `docs/video/VIDEO.md`：分镜表（时间码 / 画面 / 字幕 / 素材来源）、节奏与转场说明、
  **每个数字的溯源表**、参考片拆解（待补）、重制指引、发布清单。

#### 2. 视频不是"录一次就完"，而是可重制的工程
- `scripts/collect_promo_assets.py`：真实跑一遍流水线，导出标注帧、检测框坐标、
  人数/座位数到 `docs/video/evidence/summary.json`。
- `scripts/make_promo.py`：场景化时间轴渲染器，读 `summary.json` 合成视频。
  改 `dur` 改时长、改 `ITEMS` 改文案、改 `ACC/WARN/BG` 改品牌色，一条命令出新片。
- 因此**片中的检测框、人数、座位数都不是画上去的**，换图重跑即自动更新。

#### 3. 演示体验
- `app.py` 新增**「一键载入示例并统计」**：下拉选"密集教室 / 空教室"，点一下直接出结果，
  课堂演示不需要提前准备图片；失败时返回明确原因而不是空白。
- `app.py` 换用 `gr.themes.Soft(primary_hue="teal")`，与文档/视频配色统一。

### 修复

| 问题 | 影响 | 修复 |
|---|---|---|
| `bottleneck 1.3.7` 由 NumPy 1.x 编译 | `import gradio` 直接崩，**应用起不来** | 升到 1.6.1；`requirements.txt` 加 `bottleneck>=1.4.0` 约束并注释原因 |
| Gradio 传入 RGB、流水线按 BGR 推理 | 界面 21 人 / 命令行 23 人，**数字对不上** | `app.py` 统一 `RGB→BGR`，界面/CLI/文档三处一致（23 人 / 11 座） |
| 默认权重 `yolov8n.pt` 与文档 23 人不符 | 老师按文档跑会得到 27 人 | 仓库内置 `yolo26n.pt` 并设为默认；`src/utils.resolve_weights()` 把相对权重解析成仓库内绝对路径，换目录启动不再变成联网下载 |

### 已知边界（写进 VIDEO.md 与 README，不回避）
- 文档中的「13 → 23」里，**13 无法用当前权重复现**；同权重、同图、只改推理参数的可复现值是 **9 → 23**，
  视频采用后者。对外建议统一为 9 → 23，或注明 13 为旧版截图状态。
- 成片为**无音轨**版本；仓库不放版权不明的 BGM，需要配音请自录并按分镜表对齐。

---

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
