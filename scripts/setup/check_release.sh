#!/usr/bin/env bash
# Fail if the release tree contains large artifacts, upstream checkouts, or obvious leaks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: release check must run inside the release Git repository" >&2
    exit 1
fi

failed=0

if git ls-files --error-unmatch robodojo >/dev/null 2>&1 || [[ -n "$(git ls-files 'robodojo/*')" ]]; then
    echo "ERROR: upstream checkout robodojo/ is tracked by Git" >&2
    failed=1
fi

while IFS= read -r -d '' f; do
    if [[ -f "$f" ]] && (( $(stat -c %s "$f") > 5 * 1024 * 1024 )); then
        echo "ERROR: tracked file larger than 5 MiB: $f" >&2
        failed=1
    fi
done < <(git ls-files -z)

while IFS= read -r -d '' f; do
    case "$f" in
        *.npy|*.npz|*.pt|*.pth|*.safetensors|*.tar|*.pyc|*.bak|*.bak.*|*.orig.*)
            echo "ERROR: generated artifact tracked by Git: $f" >&2
            failed=1
            ;;
    esac
done < <(git ls-files -z)

if git grep --cached -nE '/share/workspace3/|/(home|Users)/[^/ ]+|hf_[A-Za-z0-9]{10,}|WANDB_API_KEY|wandb_[A-Za-z0-9]{10,}|password[[:space:]]*[:=]' \
    -- . ':!scripts/setup/check_release.sh' >/tmp/pi05-release-scan.txt 2>/dev/null; then
    echo "ERROR: possible absolute paths or secrets:" >&2
    cat /tmp/pi05-release-scan.txt >&2
    failed=1
fi

if (( failed )); then
    exit 1
fi
echo "release check: OK"
