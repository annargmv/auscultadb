#!/usr/bin/env bash
# SPRSound (CC BY 4.0): clone the official repository into raw/sprsound (~1.2 GB of WAV + JSON).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p raw
if [ -d raw/sprsound/.git ]; then git -C raw/sprsound pull --ff-only; else git clone --depth 1 https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git raw/sprsound; fi
