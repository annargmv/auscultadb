# AuscultaDB — notes for Claude Code

A respiratory **and cardiac** auscultation platform built on **open, real recordings**:
- a patient database with a profile per patient, and a doctors page;
- per patient, a baseline (normal) and a current recording;
- anomaly findings with type and chest location;
- a live player whose "ghost" lane shows the baseline under the current recording, aligned breath by breath;
- a chest map to hear each site, and a heart map to hear each valve (CirCor);
- a breathing-flow graph with apnea/hypopnea and AHI;
- a **Discoveries** page of cohort-level analyses, including #16, the biological-age / longevity-precondition test.

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
npm run test:ahi         # upload flow + AASM apnea scoring: 11/11 inserted pauses AND 0 events on clean breathing
npm run test:corsa       # CORSA crackle calibration: recover a known IDW from synthetic crackles

# chronic bronchitis / emphysema (Mendeley 9hdrddjpny) — Discoveries only
npm run build:cbemph                     # raw/cbemph -> meta/cbemph_index.json
node pipeline/precompute.js cbfeatures   # -> meta/features/cbemph.jsonl
python3 pipeline/cbemph_discover.py      # -> discover.json['cbemph']  (then merge.py --bump)

# CirCor heart sounds
npm run build:circor                     # raw/circor -> meta/circor_index.json + meta/circor_cohort.json
npm run validate:circor                  # cardiac validation, all 941 patients (~45 min); writes meta/circor_validation.json
node pipeline/circor_validate.js 30      #   smoke run, first 30 patients
node pipeline/precompute.js circor       # -> build/index_circor.json + build/p/C<id>.json
python3 pipeline/merge.py --add circor   # append to web/data without rebuilding SPRSound/KAUH
python3 pipeline/circor_discover.py      # -> discover.json['circor'] (threshold/feature analysis)
python3 pipeline/merge.py --bump         # re-stamp the cache key — ALWAYS last, after any discover script

# apnea / AHI on real nights (Discoveries 12) — PSG-Audio pilot, needs huggingface.co + us.aws.cdn.hf.co
npm run build:apnea                      # prep 10 patients -> validate -> discover -> --bump
python3 pipeline/psg_audio_prep.py 10    #   HF mirror -> raw/psg_audio/<pid>/h<k>_{trach,mic}.wav (8 kHz) + meta/psg_audio_index.json
node pipeline/psg_audio_validate.js      #   page's own computeFlow/scoreFlow per hour -> meta/features/psg_audio.jsonl (resumable)
python3 pipeline/psg_audio_discover.py   #   -> meta/psg_audio_validation.json + discover.json['apnea']

# clinician validation reports + ML models (Validation page) — Seismometer 0.5.0 (pip, BSD-3-Clause)
npm run build:validation                 # clinical_validation.py -> ml_validate.py -> --bump
python3 pipeline/clinical_validation.py  #   web/data/validation.json['reports']: one STARD-style report per detector
python3 pipeline/ml_validate.py          #   validation.json['ml']: sklearn models, patient split, Seismometer metrics + fairness

# models behind "Analyze a recording" (upload one recording -> pattern probabilities, never a diagnosis)
npm run build:models                     # train_models.py -> codec_check.js -> --bump
python3 pipeline/train_models.py         #   web/data/models.json: 4 models, patient split, gate, thresholds, parity vectors
node pipeline/codec_check.js             #   models.json['robustness']: site MP3 copies vs original WAV (~7 min)

# biological-age signal (Discoveries 16 + the CirCor "heart rate for age" card)
npm run build:age                        # pipeline/age_discover.py -> discover.json['age'], then --bump

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
12a. `age_discover.py` adds `discover.json['age']`: a ridge age clock on the detector features (fit on 60 % of
    patients, scored on the held-out 40 %, seed fixed, null = training mean, bootstrap CI + permutation p), CirCor heart
    rate by age band, and HR-for-age deviation vs clinical outcome. The healthy HR reference (`age.heart.ref`) also feeds
    the per-patient card. Result on the current data: **no breath-sound age signal in adults** (MAE 17.4 vs null 17.5 y),
    one SPRSound release beats the null by 0.13 y, heart rate tracks age band (ρ −0.43), and HR-for-age does **not**
    separate CirCor outcome (p 0.45). No dataset here has survival data — never describe anything as a longevity predictor.
12. **`merge.py --bump` must run last.** It also stamps every patient with `rel` (SPRSound release, CirCor campaign,
    KAUH release) from `meta/`, which the UI shows as "Release". `index.json`'s `version` is the browser cache key, and the discover
    scripts write `discover.json` *after* merge has stamped it. Without a final re-stamp, a browser that loads
    the site in between caches the stale `discover.json` under the new key and never refetches it.
    `npm run build:discover` now ends with the bump; do the same for any new discover script.

## Validation reports — how to write one (owner's rule: a physician must understand relevance and accuracy)
Every validated result gets a report in `pipeline/clinical_validation.py`, laid out like a diagnostic-accuracy study
(STARD 2015): clinical question, intended use, who was tested, reference standard, exact rule and threshold,
**independence** (`external` = thresholds fixed before the data; `development` = tuned on it, optimistic; `pilot` =
external but small; `research` = no clinical claim), then sensitivity / specificity / PPV / NPV with Agresti–Coull 95 %
CIs, prevalence, LR+ / LR− (log-method CI) read in words (Jaeschke 1994 bands), subgroups, bedside sentences
("of 100 patients with it, it flags N"), limits and a one-line verdict. Numbers only from existing validation outputs —
never typed in. ML models (`ml_validate.py`): patient-level split, model + threshold chosen on training patients only,
the shipped rule scored on the same test patients, Seismometer fairness rule (largest subgroup = reference, ×0.80–1.25).
Current ML results (seed 20261108): murmur AUC 0.82, 54 % sens at 90 % spec vs shipped rule 42 % / 98.6 %, infants
critically lower; breath-sound AUC 0.76, frozen on KAUH adults AUC 0.71. Research models — not used by the pages.

## Analyze a recording — models on new audio
- Feature code lives in the page: `recordingFeatures(A)` (lung) and `heartFeatures(A)` (heart). `precompute.js`
  calls `recordingFeatures`; `circor_validate.js` records the same heart fields. Change a feature → the stored
  `meta/features/*.jsonl` no longer match and the models must be retrained on recomputed features (raw/ needed).
- `train_models.py`: patient split 60/40 (seed 20261111), LR vs gradient boosting chosen by grouped CV on train,
  the deployed model is the train-only fit (the held-out numbers describe exactly what runs). Per-class flag
  threshold = 90 % training out-of-fold specificity. Exported as JSON; `modelProba` in the page reproduces
  scikit-learn to ~1e-9 (e2e parity check on exported vectors).
- **Deployment gate** (set after the first run, say so when reporting): held-out macro AUC ≥ 0.70, every class AUC CI
  above 0.5, and beating the majority-class (and recording-batch) baseline. Results (2026-11-11):
  heart_murmur AUC 0.80 — on; lung_sound AUC 0.72 (KAUH external 0.71) — on; **lung_pattern_child AUC 0.61, no better
  than always "pneumonia" or the release-only baseline — off; lung_pattern_adult AUC 0.60, heart failure 0.40 — off.**
  Disease from these hand-made features does not work. **Owner's decision 2026-11-11: both run anyway as
  "experimental"** (`OWNER_ON` in `train_models.py` → `override` in models.json). The gate result is never altered:
  they stay "failed validation", and the page shows each output inside a red box with its held-out accuracy against
  the always-the-commonest-class baseline and its per-group AUC. Do not lower the gate, and do not present their
  output like the validated models' output.
- **Age gate (owner, 2026-11-11):** models with `ageGate` (both pattern models) are not run outside their training
  ages — children's pattern under 18, adults' pattern 12–90 — and the page says why. Before it, the adults' model
  answered "heart failure 100 %" for a 2-month-old. The validated models only warn on age.
- `train_models.py` keeps `models.json['robustness']` when the retrained models are bit-identical; otherwise rerun
  `node pipeline/codec_check.js`.
- Compressed uploads (`codec_check.js`): median probability shift 0–1 pt, 95th pct 3–9 pts, a flag changes in
  1.5–4.6 % of decisions.
- The page always says "not a diagnosis" and shows warnings for age outside the training range, features outside the
  training 1st–99th percentile, and unmeasurable features.

## No invented content
There are no treating doctors in the open data, so none are shown. The former Doctors page is **Recording sites**
(route `#doctors` kept): the three real institutions/devices, with every count, age range, release and diagnosis
computed from the loaded index. Footer, sidebar data stamp ("data built … UTC · n min ago", from `index.json`
`version`) and the patient-list summary are generated from the data. Don't reintroduce typed-in counts.

## Code map (web/index.html)
- §0 constants: `SR = 8000`; §3 signal analysis
  - `analyzeSignal` (l.~481): breath cycles, transients (crackles), tonal (wheeze), heart rate;
  - `compareRecordings` (l.~507): findings against the baseline;
  - CORSA crackle waveform measures on every transient (`detectTransients`): `idw` (initial deflection
    width), `cd2i` (two-cycle duration), `tdw` (total deflection width), zero crossings linearly
    interpolated. **Reported only — `fine` is still `cd2 < 6 ms`**, so precomputed results stay valid.
    Reference ranges (Sovijärvi 2000; Palaniappan, PMC2990233): fine IDW 0.50–0.90 / 2CD 3.3–6.0 ms,
    coarse IDW 1.00–1.25 / 2CD 5.1–9.5 ms.
    *Measured bias:* `npm run test:corsa` recovers a known IDW to within ~10 % at SR = 8000 (0.50→0.55,
    1.00→1.11, 2.00→2.06), so the measure is sound. But on real 8 kHz recordings the medians land about
    2× the CORSA ranges (DAS: fine IDW 1.32, coarse 1.95 ms). CORSA assumes ≥10 kHz; at 8 kHz the sharp
    edges that define a narrow deflection are gone. **Do not compare these absolute values with published
    CORSA cut-offs** — use them as a relative measure within this dataset.
  - detector constants (l.~422):
    - `TR_THR = 6`, `HL_FINE = -6`, `HL_COARSE = -12`;
    - `TONAL_THR = 5.5`, `NARROW = 1.8`, `TONAL_LO = 150`. 150 Hz, because the 125–150 Hz band edge produced false wheezes in the SPRSound 2023–25 releases.
- §4 data layer:
  - `cachedFetch` (Cache Storage `auscultadb-<version>`), `getDetail` (shards), `fetchAudioBytes` (bundles);
  - `loadRealDataset` (l.~734), `ensureSite` (l.~757);
  - `isPre(p)` = precomputed dataset patient (sources `sprsound`, `kauh`); `SRC_META` = per-source credit and citation.
- §6 views: `renderAnalyze` / `runAnalyze` / `modelCard` / `modelProba` (Analyze page), `renderValidation` (Validation page, `reportCard` / `mlCard`, CI charts), `renderLongevity` (Longevity page; reuses `ageCard` / `ageCharts`); app shell = fixed dark sidebar (`header`, collapses to a top bar < 900 px; neutral ink chrome so data colours stand out), patient list (filters `LF`, `AGE_GROUPS`), patient page (`.ptgrid`), `chestMapCard` / `kauhMapCard`, `fillHrAge` (CirCor heart rate for age), recording sites (`renderDoctors`), `ageCard`/`ageCharts` (Discoveries 16), `renderDiscover` (`DISCOVER_MAP` l.~1434), `renderDatasets` (l.~1583).
- §7 live player: wavesurfer.js 7.12.12 (Regions, Spectrogram, Timeline, Minimap) plus AnalyserNode panels.
- §8 flow / apnea:
  - `computeFlow` / `scoreFlow` implement the AASM rules: apnea ≥90 % drop, hypopnea ≥30 % drop, each against the 2 preceding breaths;
  - minimum duration ≥10 s for adults, ≥2 breaths for children;
  - AHI only for ≥1 h of recording;
  - severity scales: adults 5/15/30, children 1/5/10;
  - each scored event also carries `shape` (nadir position, flat fraction, entry/exit slope, recovery
    overshoot), `quality` (pre-event baseline spread, breaths available, clipped) and `conf` — a
    confidence from flow evidence alone, since there is no SpO2 or arousal channel. All descriptive:
    the apnea/hypopnea decision and the AHI are unchanged. Obstructive vs central differ in exactly
    these shapes but **cannot be separated from breath sound**, so nothing is ever labelled as either.
- CDN libraries, pinned: wavesurfer.js 7.12.12 (unpkg), Meyda 5.6.3 and Chart.js 4.6.1 (jsDelivr). The same versions are in package.json, for offline tests.

## Datasets
| Dataset | Status | Licence | Notes |
|---|---|---|---|
| SPRSound (SJTU) | in use: 191 browsable children; all 6,567 recordings in Discoveries | CC BY 4.0 | 8 kHz; WAV header has an odd blockAlign, so the custom `parseWav` is used; 390 BioCAS2023 files have no patient number (key `'0'`) |
| KAUH (Fraiwan et al., Mendeley jwyy9np4gv v3) | in use: 112 subjects, 12–90 y | CC BY 4.0 | 4 kHz Littmann 3200, resampled to 8 kHz; one site per subject; whole-recording labels, no event timings |
| CB / emphysema (Fraiwan, Haider & Khassawneh, Mendeley 9hdrddjpny v1) | in use: Discoveries only, 234 recordings | CC BY 4.0 | 86 chronic bronchitis + 148 emphysema, **same Littmann 3200 and protocol as KAUH** — the only device-matched disease comparison here. No healthy recordings and no per-file age/sex/site; filter mode not stated. Not browsable: findings need a baseline and the set has no controls |
| CirCor DigiScope 1.0.3 (PhysioNet) | in use: 120 browsable patients; all 941 validated | **ODC-By 1.0** (checked: `LICENSE.txt` in the release — attribution, republishing allowed) | 942 patients, 3,163 recordings (3,118 primary takes used; `_1/_2/_3` repeats and `Phc` skipped). 4 kHz, up to 4 valve sites. Per-valve murmur truth plus `.tsv` S1/systole/S2/diastole segmentation — the only set here that can score the cardiac detectors |
| PSG-Audio (Korompili et al., Sci Data 2021; Science Data Bank 10.11922/sciencedb.00345) | in use: Discoveries 12 only, **pilot of 10 patients / 44 h** | CC BY 4.0 (HF mirror card + scidb page) | 212 adults, Athens sleep lab, PSG scored by technician + physician; tracheal contact mic + ambient mic at 48 kHz in hourly EDF (V3), `.rml` events + **UserStaging** (ignore MachineStaging). Read from the partial HF mirror `dust-systems/psg-audio` (the scidb FTP logins are time-limited). File names literally contain `%5B001%5D` — escape `%` in URLs. **Audio never goes to `web/`** (overnight audio can hold speech) |
| PhysioNet overnight / sleep sets | wanted | varies | Still useful for healthy sleepers and children — PSG-Audio is ~89 % severe OSA, adults only |
| ICBHI 2017 | wanted — **not obtainable**, see below | research only | Same devices from children to elderly, so a true lifespan curve; don't republish its audio |
| HF_Lung | optional | V1_IP CC BY-NC | |

## Known limitations and what is blocked

**Short clips.** Every recording in SPRSound and KAUH is 9–15 s (CirCor runs 5–65 s). Three things
cannot be answered on clips that short and all wait on overnight recordings from PhysioNet:
- apnea and AHI measured on real sleep, not a test file;
- breathing rate and heart rate over a whole night rather than one clip;
- the acoustic AHI checked against a scored sleep study (PSG).

**Apnea / AHI scoring on real nights — measured (PSG-Audio pilot, 10 adults, 44.2 h, shipped thresholds).**
Tracheal mic: finds **44.8 %** of 1,975 scored events (OA 52 %, H 34 %, CA 25 % of 20, MA 59 %), and **72.5 %** of
what it scores is real. Patient AHI **under-counts by 26 /h** on average (LoA −52 to −0.2; all 10 under even per PSG
sleep hour, so it is missed events, not missing TST); rank correlation ρ 0.58 (p 0.08, n.s.); severity class 4/10
(all misses are one class too low). Ambient mic: 38.4 % / 68.8 %, ρ 0.70 (p 0.02). **Why**: detection tracks how far
quiet breathing sits above the channel's noise floor (tracheal: 26 % found below the median headroom vs 58 % above,
ρ 0.53, p < 0.001) — the flow proxy has no noise-floor subtraction, and EDF storage cut the audio to 16 bit. Note
also that the breath period often falls back to the age norm overnight (autocorrelation finds no rhythm). The
synthetic `npm run test:ahi` (11/11 + 0 on clean breathing) still passes. Two limits are structural, not bugs:
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
| 12 | Overnight apnea / AHI vs PSG | **pilot result** (PSG-Audio, 10 patients) | more patients, healthy sleepers, children |
| 10 | Breath sounds across the lifespan | *has a result*, device-confounded | needs ICBHI (children→elderly on one device) for a true curve |
| 11 | Adult diseases | *has a result* from KAUH | more adult data would strengthen it; not blocked |
| 14 | Generalisation across stethoscopes | *has a result* (SPRSound vs KAUH) | per-device recalibration is open work, not missing access |
| 13 | Murmur / S3 on real heart sounds | **done** — CirCor 1.0.3 ingested and validated | — |

Only 9 is genuinely waiting on access; 12 has a pilot result. 10, 11 and 14 are published with their limits stated;
13 was completed and its result is live.

## Source notes (checked 2026-11-08)

- **ICBHI 2017 cannot be downloaded.** `bhichallenge.med.auth.gr` does not respond, it is not on
  PhysioNet (404) and not on Zenodo. It is normally distributed via Kaggle, which needs an account.
  Nothing is written for it yet. When the data arrives, note rule 3: ICBHI is research-only, so it may
  be **analysed into Discoveries but its audio must never reach `web/`** — that means no browsable
  patients, unlike SPRSound / KAUH / CirCor.
- **Pal & Barney 2024**, *Automatic breathing phase identification based on the second derivative of the
  recorded lung sounds* (Biomed Signal Process Control 94:106315). Savitzky-Golay second derivative,
  normalised absolute value; reported 92.8 % / 92.2 % sensitivity for inspiration / expiration. A real
  upgrade to the `estimated: true` alternation fallback in `segmentCycles`. **Not implemented**: no
  dataset here carries phase ground truth (SPRSound events are sound-type, KAUH is whole-recording
  "I E" with no timings), so it could be implemented but not validated. If added, keep it additive —
  changing `analyzeSignal` invalidates all 303 precomputed SPRSound/KAUH patients and `raw/` for both
  is gone.
- **Palaniappan et al., PMC2990233** — the CORSA review the IDW/2CD/TDW ranges above come from.
- *Magnetic resonance imaging with gradient sound respiration guide* (PLOS ONE) is **not applicable**:
  it paces a patient's breathing with MRI gradient noise and validates against a respiratory bellows.
  Nothing to port without belt data.

## Clinical audit (2026-11-08) — what was corrected and what still stands

Audited against AASM, the Levine murmur scale, CORSA and standard paediatric vital ranges.
**Correct as written:** `ageNormRR` bands; apnea ≥90 % / hypopnea ≥30 %; adult ≥10 s and paediatric
≥2 breaths; severity 5/15/30 and 1/5/10; refusing an AHI under an hour; stating that no SpO2 or
arousal channel exists; Levine I–VI used properly.

**Fixed**
1. **The cardiac HR gate rejected normal children.** `hrN` gates findings on an age-banded normal
   heart rate, and CirCor records age only as a band. Collapsing "Child" (1–11 y) to age 6 applied
   the 6–12 y gate of 70–120 bpm, so normal toddlers — whose range is 90–150 — were suppressed:
   13 % of "Child" recordings fell outside the gate. Worse, the 74 patients with **no** recorded age
   got the **adult** gate (50–110) inside a paediatric database — 24 % outside, the worst band.
   `circor_prep.py` now emits `AGE_SPAN` (ageLo/ageHi) and `hrN` takes the **union** over the band;
   unknown age gets the full 50–160 span, never the adult range.
   **Measured effect on the full 941-patient set: none, and a shade worse.** Sensitivity is
   39.7 % (71/179) before and after — not one finding flipped — while specificity slipped from
   99.0 % (687/694) to 98.6 % (684/694): the widened gate let three more false positives through
   and no true ones. The bug was real but it was never the binding constraint; the `sysDia`
   threshold is (AUC 0.742, live cut 2.5× at 30.4 % sens, Youden 1.97× at 58.5 %). Keep the fix —
   it is correct, and rejecting normal toddlers on a mis-derived age would bite on any dataset
   with a wider HR spread — but do not expect detection to improve until the threshold is refit.
2. **Respiratory panels on heart-sound recordings.** A CirCor patient page printed
   "Resp. rate (est.) 36 /min" and a full apnea/AHI card from a 6.6 s valve recording. `isCardiacOnly()`
   now suppresses both and says why. Guarded by two e2e checks.

**Open, documented, not fixed**
3. **The breathing-rate estimator is biased and it is not a disease effect.** KAUH healthy adults
   measure 25.6 /min (tachypnoeic; normal is 12–20) against 13.8 /min in CB/emphysema — backwards,
   since COPD raises rate. Not a recording-length artefact: rate does not track duration
   (ρ = −0.08, p = 0.19) and the gap survives inside the overlapping 9.8–27.9 s band (p = 0.0065).
   It most likely reflects the cycle detector over-counting on quiet healthy breathing and
   under-counting when continuous adventitious sound masks cycle boundaries. **`scoreFlow` rejects
   implausible periods using `ageNormRR`, so apnea scoring inherits this bias.** Fix with the
   detector work in task 1/2, not by adjusting the norms.
4. **Infant cardiac detection is close to useless** — 3/25 murmurs found (12 %) against 45 % in
   children. At a median 143 bpm diastole is too short for the S1/S2 separation the murmur rule
   needs. Report infants separately rather than inside a pooled figure.

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
5. **Apnea flow proxy: noise-floor correction.** The PSG-Audio pilot shows missed events are a headroom problem
   (see "Apnea / AHI scoring on real nights"). Try subtracting the channel's floor (e.g. a low percentile of band
   RMS) before the 30 % / 90 % ratios. Rule 4: fit on *new* PSG-Audio patients (not the 10 in
   `meta/psg_audio_index.json`) and report on another fresh set. Changing `computeFlow`/`scoreFlow` does not touch
   the precomputed SPRSound/KAUH analysis (flow is computed live), but rerun `npm run test:ahi`.
6. Make all 860 SPRSound children browsable (currently 191). Uploaded patients are not saved across reloads.
7. Healthy sleepers and children for apnea specificity — PSG-Audio has neither.

## Publishing as a claude.ai artifact (if asked)
- The artifact hosts at most **512 files**.
- One publish call takes **≤256 paths** (files + removals) and **≤64 MB**.
- Only text and HTML/JSON extensions are served (no `.bin`, no `.mp3`), which is why audio is base64 JSON and the analysis is sharded.
- Existing artifact: claude.ai/artifact/JawqvH3vX7cXXqbo94SXDB (slug `8e68b339-2080-481b-822f-6581919abf92`).
- Publish `web/` only. Never publish `raw/`, `build/`, `meta/` or `tests/`.

## Cloud environment
This repo's sessions run on the personal environment **auscultadb** (not Default): package registries plus
`huggingface.co`, `cdn-lfs.huggingface.co`, `cas-bridge.xethub.hf.co`, `us.aws.cdn.hf.co` (HF file downloads) and
`www.scidb.cn`. The HF hosts are also what task 3 (HeAR) needs, together with a token.
