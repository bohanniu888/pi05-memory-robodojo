#!/usr/bin/env python3
"""GPU 保活 worker（单卡）：占住显存 + 周期性算力脉冲，避免被判定为长时间闲置。

由 scripts/pi05/gpu_keepalive.sh 启动，一般不需要直接调用。
CUDA_VISIBLE_DEVICES 由启动器设置；KEEPALIVE_GPU 仅用于日志标注物理卡号。
"""
from __future__ import annotations

import argparse
import os
import signal
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="单卡 GPU 保活（占显存 + 算力脉冲）")
    parser.add_argument("--mem-mib", type=int, default=2048, help="每卡占住的显存（MiB）")
    parser.add_argument("--interval", type=float, default=0.1, help="脉冲周期（秒）")
    parser.add_argument("--burst-ms", type=float, default=90.0, help="每周期实际计算的时长（毫秒）")
    parser.add_argument("--matmul", type=int, default=256, help="脉冲矩阵边长（越大越耗算力）")
    args = parser.parse_args()

    gpu = os.environ.get("KEEPALIVE_GPU", "?")
    import torch

    if not torch.cuda.is_available():
        print(f"[keepalive] gpu={gpu} ✗ CUDA 不可用", flush=True)
        return 2

    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)

    elems = max(args.mem_mib, 64) * 1024 * 1024 // 4
    hold = torch.empty(elems, dtype=torch.float32, device="cuda:0")
    hold.fill_(0.5)

    size = max(args.matmul, 256)
    a = torch.randn(size, size, device="cuda:0")
    b = torch.randn(size, size, device="cuda:0")
    out = torch.empty(size, size, device="cuda:0")
    torch.cuda.synchronize()

    stopped = {"flag": False}

    def _term(signum, frame):  # noqa: ARG001
        stopped["flag"] = True

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    duty = 100.0 * args.burst_ms / max(args.interval * 1000.0, 1e-6)
    print(
        f"[keepalive] gpu={gpu} {name} 占显存 {args.mem_mib} MiB；"
        f"周期 {args.interval}s / 脉冲 {args.burst_ms:.0f}ms（占空比 {duty:.0f}%）",
        flush=True,
    )

    burst = args.burst_ms / 1000.0
    rounds = 0
    last_report = 0.0
    while not stopped["flag"]:
        started = time.time()
        iters = 0
        while time.time() - started < burst and not stopped["flag"]:
            torch.mm(a, b, out=out)
            iters += 1
        torch.cuda.synchronize()
        rounds += 1
        if rounds == 1 or time.time() - last_report >= 60.0:
            last_report = time.time()
            used = torch.cuda.memory_allocated(0) / 2**20
            print(
                f"[keepalive] gpu={gpu} 第 {rounds} 轮：耗时 {time.time() - started:.2f}s / "
                f"{iters} 次 {size}³ matmul，显存 {used:.0f} MiB，校验值 {float(out[0, 0]):.1f}，"
                f"t={time.strftime('%H:%M:%S')}",
                flush=True,
            )
        rest = args.interval - (time.time() - started)
        if rest > 0:
            time.sleep(rest)

    print(f"[keepalive] gpu={gpu} 收到退出信号，释放显存退出", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
