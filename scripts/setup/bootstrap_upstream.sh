#!/usr/bin/env bash
# Clone the exact upstream bases used by this project.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PI05_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ROBODOJO_DIR="${ROBODOJO_DIR:-$ROOT/robodojo}"
ROBODOJO_URL="${ROBODOJO_URL:-https://github.com/RoboDojo-Benchmark/RoboDojo.git}"
ROBODOJO_COMMIT="726e9aabfaa642203722eb126f5eaf0f37f3e1ad"
XPOLICYLAB_COMMIT="bb9a0b5f5136a74503b679af830bfd0a3a837d5c"

if [[ -e "$ROBODOJO_DIR" ]]; then
    echo "refusing to overwrite existing $ROBODOJO_DIR" >&2
    exit 1
fi

git clone "$ROBODOJO_URL" "$ROBODOJO_DIR"
git -C "$ROBODOJO_DIR" checkout --detach "$ROBODOJO_COMMIT"
git -C "$ROBODOJO_DIR" submodule update --init XPolicyLab
git -C "$ROBODOJO_DIR/XPolicyLab" checkout --detach "$XPOLICYLAB_COMMIT"

cat <<EOF
RoboDojo:   $ROBODOJO_COMMIT
XPolicyLab: $XPOLICYLAB_COMMIT

Next:
  bash scripts/setup/apply_overlay.sh
  bash scripts/setup/check_release.sh
EOF
