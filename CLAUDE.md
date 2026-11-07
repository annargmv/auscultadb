# AuscultaDB — notes for Claude Code

A respiratory **and cardiac** auscultation platform built on **open, real recordings**:
- a patient database with a profile per patient, and a doctors page;
- per patient, a baseline (normal) and a current recording;
- anomaly findings with type and chest location;
- a live player whose "ghost" lane shows the baseline under the current recording, aligned breath by breath;
- a chest map to hear each site, and a heart map to hear each valve (CirCor);
- a breathing-flow graph with apnea/hypopnea and AHI;
- a **Discoveries** page of cohort-level analyses.

It is decision support for research and teaching, not a diagnostic device.

Owner: Anna, a HealthTech software engineer (Python/React).

## Standing rules from the owner — follow these
1. **Real public data only.** No synthetic or demo patients in the database. Synthetic audio is allowed only in tests (`tests/make_test_audio.py`).
2. **Every graph must be drawn by a library from the actual sound**: wavesurfer.js, the Web Audio AnalyserNode, Meyda or Chart.js fed with computed values. No hand-drawn or static "illustrative" graphs; remove any graph that isn't driven by the audio.
3. **Licences:**
   - Check every dataset's licence before using it.
   - CC BY (SPRSound, KAUH) and ODC-By (CirCor) can be republished with attribution.
   - Research-only sets (e.g. ICBHI) may be analysed but their audio must not go into the published site.
   - Record licence and citation on the Datasets page and in `README.md`.
4. **Honest statistics.**
   - Discoveries text is data-driven: claim a trend only if it is significant.
   - Label weak or non-significant results as such, and state confounds (age, recording batch, device).
   - Don't tune detector thresholds on the set you report validation on.
5. Small, careful changes. After any change, run `npm test` and look at the screenshots in `tests/out/`.

## Layout
```
web/            the static site — this is what gets served/published
  index.html    the whole app (≈180 KB): UI + all signal analysis (see "Code map")
  data/index.json      patients, sites, labels, findings summary, validation; `version` = cache key
  data/d/<n>.json      per-patient precomputed analysis, 20 patients per shard
  data/discover.json   Discoveries results
  audio/<key>.json     {"<recording>.mp3": base64 MP3} — one bundle per patient
pipeline/       data pipeline (Python + Node/Playwright)
meta/           small derived tables kept in git (indexes, cohort selections, per-recording features)
  circor_index.json      all 941 CirCor patients — the validation set
  circor_cohort.json     the 120 browsable CirCor patients
  circor_validation.json the cardiac validation result (Discoveries 13 reads it)
raw/            downloaded datasets (git-ignored)
build/          intermediates (git-ignored): kauh_wav/, mp3/, p/, index_spr.json, index_kauh.json, index_circor.json
tests/          Playwright end-to-end tests
tools/serve.js  local static server
```

## Commands
```bash
npm install && npx playwright install chromium   # once (Playwright is pinned to 1.63.0; after upgrading it, run the install again)
pip install -r requirements.txt                  # once (+ ffmpeg on PATH for audio encoding)
npm run serve            # http://localhost:8000  (must be http, not file://)
npm test                 # e2e smoke test (offline: CDN libs served from node_modules)
npm run test:ahi         # upload flow + AASM apnea scoring on synthetic audio (expects 11/11)

# CirCor heart sounds
npm run build:circor                     # raw/circor -> meta/circor_index.json + meta/circor_cohort.json
npm run validate:circor                  # cardiac validation, all 941 patients (~45 min); writes meta/circor_validation.json
node pipeline/circor_validate.js 30      #   smoke run, first 30 patients
node pipeline/precompute.js circor       # -> build/index_circor.json + build/p/C<id>.json
python3 pipeline/merge.py --add circor   # append to web/data without rebuilding SPRSound/KAUH
python3 pipeline/circor_discover.py      # -> discover.json['circor'] (threshold/feature analysis)
python3 pipeline/merge.py --bump         # re-stamp the cache key — ALWAYS last, after any discover script

# rebuild data (needs raw/ — see README "Getting the data")
npm run build            # index → KAUH prep → MP3 → precompute → merge → discoveries
npm run build:features   # per-recording features for Discoveries (slow: all 6,567 SPRSound recordings)
npm run build:discover   # only re-run the statistics from meta/features
```

## How the data flows
1. `build_spr_index.py`: raw/sprsound → `meta/spr_full_index.json`. Covers all 6,567 recordings. The record label is derived from the event labels when a release has none (`labelSrc: 'events'`).
2. `meta/cohort_spr.json` holds the **191 browsable children** — a frozen selection made in the original session.
   - It covers every diagnosis, all severe pneumonia and the rare diagnoses, plus age-spread controls.
   - `ref` = the matched healthy control used as baseline when a child has no Normal site.
   - No script regenerates it. To change the cohort, write `pipeline/select_cohort.py`, keeping the same schema.
3. `kauh_prep.py`: raw/kauh → `build/kauh_wav` (safe names) + `meta/kauh_cohort.json` (112 subjects).
   - The three filter modes are stored as three "sites", `<zone>-<B|D|E>`.
   - The baseline is the same mode of a healthy KAUH subject: nearest age, same zone and sex preferred.
4. `encode_audio.py`: MP3, 8 kHz mono, 32 kbps via ffmpeg, for playback only. Analysis always uses the original WAVs.
5. `precompute.js`: runs **the page's own analysis functions** (`analyzeSignal`, `compareRecordings`, `expertCompare`) in headless Chromium on the original WAVs.
   - It starts its own server on :8766.
   - **If you change analysis code in `web/index.html`, rerun precompute + merge.** Otherwise the site keeps showing the old results.
6. `merge.py` writes `web/data/index.json` (new `version`, so browsers drop their old cache), the shards and the audio bundles.
7. `discover.py` → `person_probe.py` → `kauh_discover.py` write `web/data/discover.json`, in that order. Each later script adds a key.
8. `circor_prep.py`: raw/circor → `meta/circor_index.json` (all 941 patients with audio) + `meta/circor_cohort.json` (the 120 browsable).
   - Nothing is copied: CirCor file names are already safe, so the pipeline reads `raw/circor/.../training_data` in place.
   - The 4 valve sites AV/PV/TV/MV are the "sites". Per-site label comes from the CSV's `Murmur locations`, so each valve has its own ground truth.
   - Age is categorical in CirCor; `AGE_Y` maps the bands to a representative numeric age because the cardiac rules gate on a numeric heart-rate range.
   - The browsable cohort deliberately over-samples murmurs (70 of 120 vs 19 % real prevalence) to cover every grade, timing and the 5 diastolic cases. Report rates from the full set, never from the cohort.
   - `ref` is recomputed *inside* the cohort, because precompute.js resolves baselines against the cohort only.
9. `circor_validate.js` runs the page's own `analyzeSignal` + `compareRecordings` over all 941 patients and scores them against the expert labels, writing `meta/circor_validation.json` and `meta/features/circor.jsonl`. It tunes nothing.
10. **`merge.py --add <src>` appends one dataset** to the existing `web/data` instead of rebuilding everything. A full `merge.py` re-precomputes every patient, which needs all of `raw/` (SPRSound alone is a 1.2 GB clone); `--add` only touches its own patients and allocates new shard numbers.
11. `circor_discover.py` adds `discover.json['circor']` — the threshold-vs-feature analysis behind Discoveries 13.
12. **`merge.py --bump` must run last.** `index.json`'s `version` is the browser cache key, and the discover
    scripts write `discover.json` *after* merge has stamped it. Without a final re-stamp, a browser that loads
    the site in between caches the stale `discover.json` under the new key and never refetches it.
    `npm run build:discover` now ends with the bump; do the same for any new discover script.

## Code map (web/index.html)
- §0 constants: `SR = 8000`; §3 signal analysis
  - `analyzeSignal` (l.~481): breath cycles, transients (crackles), tonal (wheeze), heart rate;
  - `compareRecordings` (l.~507): findings against the baseline;
  - detector constants (l.~422):
    - `TR_THR = 6`, `HL_FINE = -6`, `HL_COARSE = -12`;
    - `TONAL_THR = 5.5`, `NARROW = 1.8`, `TONAL_LO = 150`. 150 Hz, because the 125–150 Hz band edge produced false wheezes in the SPRSound 2023–25 releases.
- §4 data layer:
  - `cachedFetch` (Cache Storage `auscultadb-<version>`), `getDetail` (shards), `fetchAudioBytes` (bundles);
  - `loadRealDataset` (l.~734), `ensureSite` (l.~757);
  - `isPre(p)` = precomputed dataset patient (sources `sprsound`, `kauh`); `SRC_META` = per-source credit and citation.
- §6 views: patient list (filters `LF`, `AGE_GROUPS`), patient page, `chestMapCard` / `kauhMapCard`, doctors, `renderDiscover` (`DISCOVER_MAP` l.~1434), `renderDatasets` (l.~1583).
- §7 live player: wavesurfer.js 7.12.12 (Regions, Spectrogram, Timeline, Minimap) plus AnalyserNode panels.
- §8 flow / apnea:
  - `computeFlow` / `scoreFlow` implement the AASM rules: apnea ≥90 % drop, hypopnea ≥30 % drop, each against the 2 preceding breaths;
  - minimum duration ≥10 s for adults, ≥2 breaths for children;
  - AHI only for ≥1 h of recording;
  - severity scales: adults 5/15/30, children 1/5/10.
- CDN libraries, pinned: wavesurfer.js 7.12.12 (unpkg), Meyda 5.6.3 and Chart.js 4.6.1 (jsDelivr). The same versions are in package.json, for offline tests.

## Datasets
| Dataset | Status | Licence | Notes |
|---|---|---|---|
| SPRSound (SJTU) | in use: 191 browsable children; all 6,567 recordings in Discoveries | CC BY 4.0 | 8 kHz; WAV header has an odd blockAlign, so the custom `parseWav` is used; 390 BioCAS2023 files have no patient number (key `'0'`) |
| KAUH (Fraiwan et al., Mendeley jwyy9np4gv v3) | in use: 112 subjects, 12–90 y | CC BY 4.0 | 4 kHz Littmann 3200, resampled to 8 kHz; one site per subject; whole-recording labels, no event timings |
| CirCor DigiScope 1.0.3 (PhysioNet) | in use: 120 browsable patients; all 941 validated | **ODC-By 1.0** (checked: `LICENSE.txt` in the release — attribution, republishing allowed) | 942 patients, 3,163 recordings (3,118 primary takes used; `_1/_2/_3` repeats and `Phc` skipped). 4 kHz, up to 4 valve sites. Per-valve murmur truth plus `.tsv` S1/systole/S2/diastole segmentation — the only set here that can score the cardiac detectors |
| PhysioNet overnight / sleep sets | wanted | varies | For real apnea/AHI validation against PSG |
| ICBHI 2017 | wanted | research only | Same devices from children to elderly, so a true lifespan curve; don't republish its audio |
| HF_Lung | optional | V1_IP CC BY-NC | |

## Known limitations and what is blocked

**Short clips.** Every recording in SPRSound and KAUH is 9–15 s (CirCor runs 5–65 s). Three things
cannot be answered on clips that short and all wait on overnight recordings from PhysioNet:
- apnea and AHI measured on real sleep, not a test file;
- breathing rate and heart rate over a whole night rather than one clip;
- the acoustic AHI checked against a scored sleep study (PSG).

**Apnea / AHI scoring is untested on real data.** `computeFlow` / `scoreFlow` implement the AASM
rules and pass `npm run test:ahi` (11/11), but that test runs on a *synthetic* file from
`tests/make_test_audio.py`. Nothing here has been scored against a real night. Two limits are
structural, not bugs:
- it cannot separate obstructive from central events — airflow sound alone does not carry that;
- there is no SpO₂ channel, so hypopneas are scored on the airflow drop only, without the
  desaturation criterion the AASM rules also allow.

**Google models (HeAR, MedGemma).** Both need a Hugging Face token issued after accepting Google's
Health AI Developer Foundations terms, and `huggingface.co` reachable from the workspace. MedGemma
additionally needs more memory than this machine has (4B ≈ 8 GB). HeAR would embed breath sounds;
MedGemma would draft the written report from the structured findings — it takes text and images
only, so it never listens to the audio.

**Which Discoveries entries this holds up** (numbers are the position in `DISCOVER_MAP`):

| # | Entry | Status | Blocker |
|---|---|---|---|
| 9 | HeAR embeddings vs hand-made features | blocked | HF token + network; compare against the existing probe (AUC 0.61 pooled, 0.52–0.56 within a release) |
| 12 | Overnight apnea / AHI vs PSG | blocked | PhysioNet overnight data |
| 10 | Breath sounds across the lifespan | *has a result*, device-confounded | needs ICBHI (children→elderly on one device) for a true curve |
| 11 | Adult diseases | *has a result* from KAUH | more adult data would strengthen it; not blocked |
| 14 | Generalisation across stethoscopes | *has a result* (SPRSound vs KAUH) | per-device recalibration is open work, not missing access |
| 13 | Murmur / S3 on real heart sounds | **done** — CirCor 1.0.3 ingested and validated | — |

Only 9 and 12 are genuinely waiting on access. 10, 11 and 14 are published with their limits stated;
13 was completed and its result is live.

## Open tasks (priority order)
1. **Cardiac detector recalibration — the CirCor validation says it is needed.**
   - Measured on all 941 patients / 3,118 recordings, thresholds unchanged (`meta/circor_validation.json`):
     per patient **sens 40 % (71/179), spec 99 % (687/694)**; per valve sens 30 %, spec 99.6 %.
   - It is **grade-limited**: I/VI **18 %**, II/VI **39 %**, III/VI **89 %** (Spearman ρ = 0.59). Loud murmurs
     are found, soft ones are not, and soft ones are what screening needs.
   - It is **both** the threshold and the feature, and the threshold is the bigger, cheaper win.
     `pipeline/circor_discover.py` scores `sysDia` threshold-free: the ROC AUC is moderate, and the
     ratio alone has **ROC AUC 0.742** (2,816 labelled valves) — real signal. The shipped `2.5×` cut
     sits far out on the specificity end: **30.4 % sens / 99.6 % spec**. The Youden point, `1.97×`,
     gives **58.5 % sens / 90.9 % spec** — roughly double the detection for 8.7 points of specificity.
   - But refitting alone does not make it good: even that best point misses **42 %** of murmur valves,
     so a shape- or variance-based feature is still needed to go further. Threshold first, feature second.
   - `pipeline/circor_discover.py` recomputes all of this; it is also what `discover.json['circor']` holds.
   - S1/S2 segmentation is the upstream limit: **71 %** recall for both (median timing error 49 / 45 ms,
     ±100 ms tolerance, 16 % extra S1 detections). The murmur rule can only work on beats it resolved.
   - Rule 4 applies: fit on one split of CirCor, report on a held-out split. Do **not** refit on the set already reported in Discoveries 13.
   - S3 cannot be validated here — CirCor does not label S3. It needs a different dataset.
2. **Detector recalibration per device.**
   - On KAUH (diaphragm mode): healthy kept normal 57 %, wheeze found 72 %, crackles found only 23 % (heart-failure crackles are missed).
   - Fit on one split and report on another — never on the reporting set.
3. **Google HeAR embeddings** (Hugging Face `google/hear-pytorch`; needs the HAI-DEF terms and an HF token).
   - Embed 2-s windows at 16 kHz, mean-pool per patient, then the same linear probe as `person_probe.py`.
   - Compare against the hand-made features: AUC 0.61 pooled, 0.52–0.56 within a release.
4. **MedGemma** (text and images only, no audio) could draft the written report from the structured findings. 4B needs about 8 GB of RAM.
5. Make all 860 SPRSound children browsable (currently 191). Uploaded patients are not saved across reloads.
6. Overnight data → real AHI validation.

## Publishing as a claude.ai artifact (if asked)
- The artifact hosts at most **512 files**.
- One publish call takes **≤256 paths** (files + removals) and **≤64 MB**.
- Only text and HTML/JSON extensions are served (no `.bin`, no `.mp3`), which is why audio is base64 JSON and the analysis is sharded.
- Existing artifact: claude.ai/artifact/JawqvH3vX7cXXqbo94SXDB (slug `8e68b339-2080-481b-822f-6581919abf92`).
- Publish `web/` only. Never publish `raw/`, `build/`, `meta/` or `tests/`.
