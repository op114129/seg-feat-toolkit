感谢你决定为本项目贡献力量！为了保证代码质量与协作效率，请在提交前阅读并遵守以下约定。

1. Issues

提问前先搜索：避免重复问题。若确认是 bug，请提供：系统/环境信息、可复现步骤、错误日志或截图。

功能需求：尽量描述清楚使用场景、预期行为，以及你愿意参与实现的部分。

2. 分支策略

main：稳定版，仅存放已发布、可运行代码。

dev：日常集成分支。

功能/修复请从 dev 派生：git checkout -b feat/<name> 或 fix/<name>。

3. 代码规范

语言

工具链

规则

Python

black + isort

行宽 88、4 空格缩进

Markdown

markdownlint

标题层级正确，列表缩进一致

在根目录执行：

pip install pre-commit
pre-commit install

即可在每次 git commit 时自动格式化。

4. Commit 信息

使用 Conventional Commits：

feat: 新功能

fix: Bug 修复

docs: 文档变更

refactor / style / test / chore 等

限制在 72 字符以内；必要时正文附动机 & 影响范围。

5. Pull Request 流程

Fork → Push branch → 提 PR 到 dev。

PR 标题 = 「type(scope): summary」，如 feat(seg): add opacity flag。

确保 python semantic_segmentation_and_features.py --help 正常运行。

通过 CI（GitHub Actions）后，维护者会 Review 并合并。

6. 代码 / 文档同步

添加或修改功能时，请同步更新：

README.md 快速开始示例

CHANGELOG.md 下一个版本的 Added / Fixed 条目

7. 开发环境

Python ≥ 3.8，推荐使用 virtualenv 或 conda：

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

GPU 场景请安装与显卡驱动 / CUDA 版本匹配的 torch、mmcv-full。

8. 许可证

本项目采用 MIT 许可。提交 PR 即表示你同意以 MIT 许可发布你的贡献。

Happy coding! 🚀

