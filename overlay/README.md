# 覆盖层文件

上游仓库中存在已记录实验所需、但未被 Git 跟踪的新文件。应用已跟踪文件的补丁后，再复制两个覆盖层目录：

```bash
cp -a overlay/robodojo/. robodojo/
cp -a overlay/xpolicylab/. robodojo/XPolicyLab/
```

RoboDojo 覆盖层：

- `env/diagnostics.py` 和 `env/scene_manager` 挂接点：用于 reset/PhysX 排查的分阶段诊断。
- `docker/`：Isaac Sim 6 镜像、入口脚本、conda shim 和 `sitecustomize` 兼容层。
- `utils/__init__.py`：让 RoboDojo 自带的 `utils` 包优先于 OpenCV 的同名命名空间冲突。

XPolicyLab 覆盖层：

- `policy/Pi_05/openpi/src/openpi/policies/robodojo_history.py`：用于 20 个历史槽位的 sidecar 变换，
  支持置零/`repeat_oldest` 填充、可选关键帧权重以及内容滚动对比视图。

覆盖层不包含 checkpoint、备份文件、`__pycache__` 或生成的 sidecar `.npy` 文件。
