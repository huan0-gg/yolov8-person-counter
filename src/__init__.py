"""src: 人员检测与座位占用统计核心模块。

模块划分：
  detector    - YOLOv8 检测器封装（基线 A）
  augment     - 创新点 1：数据增强策略
  tracker     - 创新点 2：ByteTrack 跟踪去重
  roi_counter - 创新点 3：ROI 座位计数规则
  pipeline    - 组合以上模块的完整推理流水线
  visualize   - 结果可视化
  utils       - 通用工具
"""
