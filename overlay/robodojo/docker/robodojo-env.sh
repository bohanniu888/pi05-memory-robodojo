#!/bin/bash
# Environment for the RoboDojo simulation stack on the Isaac Sim 6.0 base image.
#
# Sourced (never executed) by docker/entrypoint.isaac6.sh, by the Dockerfile's
# build steps, and by anything else that needs a working isaacsim/isaaclab
# interpreter. Kept as one file so build-time and run-time cannot drift apart --
# that matters here because pip builds C extensions (scipy, open3d, Pillow)
# against whichever numpy it can see, and the image ships two of them.
#
# This image replaces `python.sh` rather than calling it: python.sh hard-codes
# /isaac-sim/kit/python, but the RoboDojo stack lives in its own venv. The four
# things python.sh does that actually matter are reproduced below.

# 1. Kit needs to know where it lives before it can start in-process.
export ISAAC_PATH=/isaac-sim
export CARB_APP_PATH=/isaac-sim/kit
export EXP_PATH=/isaac-sim/apps

ROBODOJO_VENV=/opt/robodojo

# Where the RoboDojo tree lives inside the image. Read by
# /opt/robodojo-patch/sitecustomize.py, which has to claim the top-level names
# `utils` and `env` before OpenCV's sys.path.insert(1, ...) can hijack them.
export ROBODOJO_ROOT=/workspace/RoboDojo

# 2. Extension roots, LD_LIBRARY_PATH and the kit python lib dir. This is
#    NVIDIA's own helper, so /isaac-sim/exts*, extscache and the kit plugins stay
#    in sync with the installed Isaac Sim build instead of being hand-listed.
#    Side effect: it appends the container's pip_prebundle directories to
#    PYTHONPATH, which is where numpy 2.3.1, pillow 12.2.0 and packaging come
#    from. See step 5 for how that is handled.
# shellcheck disable=SC1091
source /isaac-sim/setup_python_env.sh

# 3. Put the simulator venv in front. `isaaclab.sh` picks the interpreter up
#    through VIRTUAL_ENV (see extract_python_exe in IsaacLab/isaaclab.sh).
export VIRTUAL_ENV="$ROBODOJO_VENV"
export PATH="$ROBODOJO_VENV/bin:/opt/conda-shim/bin:/isaac-sim/kit/python/bin:$PATH"

# 4. sitecustomize shim (omni.physics.tensors.impl alias) and the repo itself.
export PYTHONPATH="/opt/robodojo-patch:/workspace/RoboDojo${PYTHONPATH:+:$PYTHONPATH}"

# 5. numpy precedence.
#    RoboDojo pins numpy==1.26.0 (scripts/install.sh, three times) and Isaac Lab
#    2.3.2 requires numpy<2, but the Isaac Sim 6.0 container puts numpy 2.3.1 on
#    PYTHONPATH from its own pip_prebundle -- PYTHONPATH beats site-packages, so
#    without this the pin would be silently ignored and every C extension we
#    build against numpy 1.x would meet a numpy 2.x runtime.
#    Prepending the venv's site-packages makes the venv -- and therefore the pin
#    -- win. Set ROBODOJO_VENV_FIRST=0 to let the container's numpy 2.3.1 win
#    instead; that is the fallback if an isaacsim extension turns out to need
#    numpy 2.x APIs.
if [[ "${ROBODOJO_VENV_FIRST:-1}" == "1" ]]; then
    export PYTHONPATH="${ROBODOJO_VENV}/lib/python3.12/site-packages:${PYTHONPATH}"
fi

# 6. python.sh preloads libcarb.so ("WAR for missing libcarb.so") and any caller
#    provided LD_PRELOAD is preserved. Doing the same here keeps the in-process
#    SimulationApp start-up on the officially supported path.
if [[ "${ROBODOJO_LD_PRELOAD:-1}" == "1" && -f /isaac-sim/kit/libcarb.so ]]; then
    export LD_PRELOAD="/isaac-sim/kit/libcarb.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi
