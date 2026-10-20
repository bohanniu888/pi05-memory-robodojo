import os
import time


_START = time.monotonic()


def diag_stage(message: str) -> None:
    if os.environ.get("ROBODOJO_DIAG_STAGES") != "1":
        return
    print(f"[diag-stage] +{time.monotonic() - _START:.3f}s {message}", flush=True)
