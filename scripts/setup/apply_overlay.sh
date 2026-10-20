#!/usr/bin/env bash
# Apply tracked-file patches and untracked-file overlays to an upstream checkout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ROBODOJO_DIR="${ROBODOJO_DIR:-$ROOT/robodojo}"
XPOLICYLAB_DIR="$ROBODOJO_DIR/XPolicyLab"

if [[ ! -d "$ROBODOJO_DIR/.git" && ! -f "$ROBODOJO_DIR/.git" ]]; then
    echo "missing RoboDojo checkout: $ROBODOJO_DIR" >&2
    exit 1
fi
if [[ ! -d "$XPOLICYLAB_DIR/.git" && ! -f "$XPOLICYLAB_DIR/.git" ]]; then
    echo "missing XPolicyLab submodule: $XPOLICYLAB_DIR" >&2
    exit 1
fi

apply_once() {
    local repo="$1" patch="$2"
    if git -C "$repo" apply --reverse --check "$patch" >/dev/null 2>&1; then
        echo "[already applied] $patch"
        return
    fi
    git -C "$repo" apply --check "$patch"
    git -C "$repo" apply "$patch"
    echo "[applied] $patch"
}

apply_once "$ROBODOJO_DIR" "$ROOT/patches/robodojo.patch"
apply_once "$XPOLICYLAB_DIR" "$ROOT/patches/xpolicylab.patch"

cp -a "$ROOT/overlay/robodojo/." "$ROBODOJO_DIR/"
cp -a "$ROOT/overlay/xpolicylab/." "$XPOLICYLAB_DIR/"

echo "[overlay] copied robodojo and xpolicylab new files"
python3 "$ROOT/scripts/pi05/patch_pi05_history.py" --check
