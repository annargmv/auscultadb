#!/usr/bin/env bash
# CirCor DigiScope phonocardiogram dataset (PhysioNet) into raw/circor. Needs `pip install physionet`.
# Check the licence on https://physionet.org/content/circor-heart-sound/1.0.3/ before republishing any audio.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p raw/circor && cd raw/circor
physionet download circor-heart-sound --version 1.0.3
