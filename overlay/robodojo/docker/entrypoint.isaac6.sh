#!/bin/bash
# Entrypoint for the Isaac Sim 6.0 ("Path G") RoboDojo image.
#
# Mirrors docker/entrypoint.sh from the 5.1 image, minus conda: the simulator
# environment is a plain venv here and docker/robodojo-env.sh makes it current.
set -e

# shellcheck disable=SC1091
source /usr/local/bin/robodojo-env.sh

cd /workspace/RoboDojo
exec "$@"
