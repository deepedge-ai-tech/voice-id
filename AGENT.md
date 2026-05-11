# AGENT.md

## AI 交互规范

Claude Code 与本项目交互时必须遵循以下规范。

### 任务执行流程

1. **理解任务** → 阅读 CLAUDE.md 和 AGENT.md
2. **Plan 模式** → 必须使用 company-global-constraints 约束
3. **执行任务** → 遵循 CLAUDE.md 中的技术栈和规范
4. **验证完成** → 对照强制检查清单验证

### Plan 模式强制约束

- 必须应用 company-global-constraints 技能
- 每个 Plan 步骤必须包含验证标准
- 文件名使用 kebab-case
- 函数/变量使用 snake_case
- 类名使用 PascalCase
- 代码变更必须遵循 TDD 流程
- 必须包含类型注解和 docstring

### 代码实现强制约束

| 约束 | 要求 |
|------|------|
| 命名 | 文件 kebab-case，函数 snake_case，类 PascalCase |
| 类型注解 | 所有函数必须包含参数和返回值类型注解 |
| 文档字符串 | 所有公开函数必须包含 docstring（Args/Returns/Raises） |
| 日志 | 禁止 print()，使用 logging 模块 |
| 异常 | 禁止裸 except，必须指定异常类型并记录日志 |
| 格式化 | black + isort |

### 任务完成强制验证

- [ ] 已读取 CLAUDE.md 和 AGENT.md
- [ ] 已应用 company-global-constraints 技能
- [ ] Plan 步骤包含验证标准
- [ ] 代码变更遵循 TDD 流程
- [ ] 运行 pytest 并确保通过
- [ ] 检查代码覆盖率 ≥ 80%
- [ ] 运行 black . && isort . 格式化代码
- [ ] 更新项目图表（6 种 Mermaid 图）

### 禁止事项汇总

- ❌ 不创建测试或跳过 TDD 流程
- ❌ 使用 print() 而非 logging
- ❌ 使用裸 except
- ❌ 忽略类型注解
- ❌ 忽略 docstring
- ❌ 提交音频文件、模型文件、pkl 文件到 git
- ❌ 使用与 CLAUDE.md 冲突的技术栈
- ❌ 跳过代码格式化（black + isort）
