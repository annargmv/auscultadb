"""Shared paths. Everything is relative to the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'raw'                 # downloaded datasets (git-ignored)
SPR_RAW = RAW / 'sprsound'         # git clone of SJTU-YONGFU-RESEARCH-GRP/SPRSound
KAUH_RAW = RAW / 'kauh'            # KAUH Mendeley download (Audio Files.zip, Data_annotation.xlsx)
CIRCOR_RAW = RAW / 'circor'        # PhysioNet CirCor DigiScope (ODC-By)
CIRCOR_TRAIN = CIRCOR_RAW / 'circor-heart-sound-1.0.3' / 'training_data'   # read directly; names are already safe
META = ROOT / 'meta'               # small derived tables kept in git
FEATURES = META / 'features'       # per-recording features for Discoveries
BUILD = ROOT / 'build'             # intermediate outputs (git-ignored)
WEB = ROOT / 'web'                 # the static site that is served / published
WEB_DATA = WEB / 'data'
WEB_AUDIO = WEB / 'audio'
SPR_FEATURES = sorted(FEATURES.glob('spr.*.jsonl'))
KAUH_FEATURES = FEATURES / 'kauh.jsonl'
