#!/usr/bin/env bash

# Load local enviromnet
source /img/conda.local/env.sh
source activate coral_reef

exec "$@"
