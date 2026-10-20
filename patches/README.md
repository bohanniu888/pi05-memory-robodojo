# 上游补丁

这些补丁由 2026-10-20 记录的工作树生成，只适用于根目录 README 中列出的确切提交，不适用于之后任意版本的
上游代码。

| 文件 | 目标 | 执行目录 |
| --- | --- | --- |
| `robodojo.patch` | RoboDojo 已跟踪文件（不含 XPolicyLab 子模块） | `robodojo/` |
| `xpolicylab.patch` | XPolicyLab 已跟踪文件 | `robodojo/XPolicyLab/` |

应用顺序：

1. `git -C robodojo apply ../patches/robodojo.patch`
2. `git -C robodojo/XPolicyLab apply ../../patches/xpolicylab.patch`
3. 将 `overlay/robodojo/.` 复制到 `robodojo/.`
4. 将 `overlay/xpolicylab/.` 复制到 `robodojo/XPolicyLab/.`

项目同时提供补丁和覆盖层，是因为 Git diff 不包含未跟踪文件。XPolicyLab 补丁覆盖训练配置、训练器、
数据加载器、权重加载器、策略适配器和部署配置。`patch_pi05_history.py` 只覆盖五个 openpi 文件，
不能替代这组补丁。

这些补丁属于研究性改动，并非上游正式发布。分发时必须保留上游许可证条款。
