# 声明

本仓库包含将 π0.5 适配到 RoboDojo Memory 维度所需的项目专属文档、脚本和补丁。它不是 RoboDojo 或
XPolicyLab 的官方发布。

## 上游组件

- **RoboDojo** — Copyright (c) 2025 Yue Chen。MIT License。上游地址：
  <https://github.com/RoboDojo-Benchmark/RoboDojo>。完整 MIT 许可证文本位于
  `third_party/RoboDojo-LICENSE.txt`。
- **XPolicyLab** — Apache License 2.0。上游地址：<https://github.com/XPolicyLab/XPolicyLab>。完整 Apache
  许可证文本位于 `third_party/XPolicyLab-LICENSE.txt` 和仓库根目录的 `LICENSE`。

## 本仓库许可证的适用范围

除文件另有说明外，为本项工作编写的项目专属文档和脚本采用 Apache-2.0 许可。`patches/` 和 `overlay/` 中的文件
修改或扩展了上游代码，仍受对应上游项目许可证约束：

- `patches/robodojo.patch` 和 `overlay/robodojo/**` 派生自 RoboDojo，继续适用 MIT 条款。
- `patches/xpolicylab.patch` 和 `overlay/xpolicylab/**` 派生自 XPolicyLab，继续适用 Apache-2.0 条款。

收录上游代码不会使其被重新许可。

## 不随仓库分发的产物

本仓库不包含 RoboDojo 数据集、模型权重、微调后的 checkpoint、仿真器镜像、评测日志或第三方模型文件。
使用者必须从相应官方来源获取这些产物，并遵守适用的许可证和基准提交规则。
