# cross_test.py 详细诊断输出设计文档

**日期**: 2025-01-12
**作者**: Claude + 用户
**状态**: 已批准

## 概述

为 `scripts/cross_test.py` 添加详细的诊断输出功能，帮助持续优化 `src/wespeaker` 模块。通过分级输出模式（默认/verbose/debug）和多格式报告（终端/Markdown/JSON），提供全面的性能分析和错误诊断。

## 目标

1. **全面诊断** - 覆盖注册阶段、识别阶段、结果分析的全流程诊断
2. **分级输出** - 通过 `--verbose` 和 `--debug` 控制输出详细程度
3. **性能优化** - 正常运行时低开销，详细信息按需计算
4. **多格式报告** - 终端摘要 + Markdown 可读报告 + JSON 原始数据

## 设计

### 1. 输出模式

| 模式 | 触发方式 | 终端输出 | 文件输出 | 性能开销 |
|------|----------|----------|----------|----------|
| 默认 | 无参数 | 摘要（进度 + 结果） | Markdown + JSON | 低 |
| Verbose | `--verbose` | 详细（每步状态） | Markdown + JSON | 中 |
| Debug | `--debug` | 最详细（向量值） | Markdown + JSON + DEBUG日志 | 高 |

### 2. 注册阶段诊断

#### 2.1 基础信息
- 说话人名称
- 注册片段数量和文件列表
- 每个片段的时长、采样率
- SNR 级别列表
- 总 embedding 数量、维度

#### 2.2 向量质量分析
- 每个 embedding 的 L2 范数
- 与均值的余弦距离（标准差、最大值、最小值）
- 类内紧密度（within-class compactness）
- 向量分布统计

#### 2.3 噪声注入效果
- 原始音频 RMS 能量
- 每个 SNR 级别混合后的实际 SNR 估计值
- 噪声 profile 长度
- 混合前后能量对比

### 3. 识别阶段诊断

#### 3.1 音频预处理分析
- 测试音频时长、采样率
- VAD 切分详情（启用时）
- 音频裁剪策略效果
- RMS 能量分布

#### 3.2 向量匹配分析
- 测试音频 embedding 提取时间
- 与每个参考声纹的余弦相似度
- Top-2 相似度差异（边界分析）

#### 3.3 错误案例分析
- **误接受分析**:
  - 得分 vs 阈值的距离
  - 测试音频变体类型
  - 涉及的说话人
- **误拒绝分析**:
  - 得分 vs 阈值的距离
  - 测试音频变体类型
  - 建议阈值调整

#### 3.4 性能计时
- 每个步骤的耗时（音频加载、预处理、embedding 提取、相似度计算）
- 总体执行时间

### 4. src/wespeaker 改造

为 `src/wespeaker/wespeaker.py` 和 `src/wespeaker/best.py` 添加 logging 支持：

```python
import logging

logger = logging.getLogger(__name__)

# 不同等级的使用：
# logger.debug() - 最详细的调试信息（向量值、中间计算）
# logger.info()  - 一般信息（文件加载、参数设置）
# logger.warning() - 警告（低质量音频、边界情况）
# logger.error()   - 错误（文件不存在、处理失败）
```

### 5. 输出文件结构

```
outputs/
├── cross_test_report_YYYYMMDD_HHMMSS.md    # Markdown 报告
├── cross_test_data_YYYYMMDD_HHMMSS.json    # JSON 原始数据
└── cross_test_debug_YYYYMMDD_HHMMSS.log    # Debug 日志（仅 --debug）
```

### 6. Markdown 报告结构

```markdown
# 声纹交叉测试诊断报告

## 测试配置
- 阈值: 0.55
- SNR 级别: [20, 15, 10, 5, 0]
- 测试时间: YYYY-MM-DD HH:MM:SS

## 注册阶段分析
### [说话人1]
- 片段数量: X
- 向量质量统计: ...
- 噪声注入效果: ...

## 识别阶段分析
### 性能统计
- 平均识别时间: ...
- 详细计时: ...

### 错误案例分析
- 误接受案例: ...
- 误拒绝案例: ...

### 音频变体性能分析
- 电话音效: ...
- 大厅回音: ...
- ...

## 结论与建议
- 当前配置评估
- 阈值调整建议
- 需要改进的说话人/音频变体
```

### 7. JSON 数据结构

```json
{
  "meta": {
    "timestamp": "2025-01-12T14:30:22",
    "threshold": 0.55,
    "snr_levels": [20, 15, 10, 5, 0],
    "speakers": ["Frank", "John", ...]
  },
  "registration": {
    "Frank": {
      "num_segments": 5,
      "embeddings": [...],
      "quality_metrics": {...}
    },
    ...
  },
  "recognition": {
    "test_cases": [...],
    "performance": {...},
    "errors": {
      "false_accepts": [...],
      "false_rejects": [...]
    }
  }
}
```

## 实现

### 新增类

1. **`RegistrationDiagnostics`** - 收集注册阶段诊断数据
2. **`RecognitionDiagnostics`** - 收集识别阶段诊断数据
3. **`PerformanceMetrics`** - 性能计时统计
4. **`TerminalReporter`** - 终端输出（分级）
5. **`MarkdownReportGenerator`** - Markdown 报告生成
6. **`JsonDataExporter`** - JSON 数据导出

### 修改文件

1. **`scripts/cross_test.py`** - 主流程改造，集成诊断收集
2. **`src/wespeaker/wespeaker.py`** - 添加 logging 支持
3. **`src/wespeaker/best.py`** - 添加 logging 支持

### CLI 参数

```bash
# 现有参数保持不变
--noise
--snrs
--threshold
--output-dir

# 新增参数
--verbose    # 详细输出模式
--debug      # 调试模式（最详细）
```

## 优化策略

1. **惰性计算** - 只有在 verbose/debug 模式下才计算额外的统计指标
2. **缓存复用** - 避免重复计算 embedding
3. **条件日志** - 使用 logging 等级控制输出

## 验收标准

1. 默认模式下性能无明显下降
2. Markdown 报告可读且包含所有关键信息
3. JSON 数据结构清晰，便于后续分析
4. --debug 模式提供足够的调试信息
