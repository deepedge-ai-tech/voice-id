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
- 代码变更必须遵循 TDD 流程

### 任务完成验证

- [ ] 代码可正常运行（`python wespeaker.py enroll clean.wav test.pkl`）
- [ ] 无语法错误
- [ ] 符合 snake_case 命名规范
- [ ] 包含类型注解

### 禁止事项

- 不要引入 FastAPI/Django 等 Web 框架
- 不要修改核心类 WespeakerClient 的公共 API 签名
- 不要删除已有的 CLI 入口
- 不要提交模型文件到 git
