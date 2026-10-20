#!/bin/bash
# Shell hook for the conda shim (see ./bin/conda for the rationale).
#
# Sourcing this file must make the `conda activate` / `conda deactivate` /
# `conda info --base` calls used by XPolicyLab succeed. Defining `conda` as a
# shell function keeps activation inside the caller's shell, which is what
# scripts that call `conda activate` and then rely on the environment expect.

conda() {
    case "${1:-}" in
        activate)
            echo "[conda-shim] activate '${2:-}' -> no-op: the RoboDojo venv (/opt/robodojo) is always active" >&2
            ;;
        deactivate)
            : # deliberately a no-op; /opt/robodojo stays on PATH
            ;;
        *)
            command conda "$@"
            ;;
    esac
}
