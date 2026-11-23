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
  index.html    the whole app (≈370 KB): UI + all signal analysis (see "Code map")
  data/index.json      patients, sites, labels, findings summary, validation; `version` = cache key
  data/d/<n>.json      per-patient precomputed analysis, 20 patients per shard (CirCor 4: `merge.SHARD_BY`; `pipeline/reshard.py <src>` re-shards without recomputing)
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

# CirCor extended features + HRV validation (needs raw/circor: PhysioNet zip, physionet.org allowed)
npm run build:hrv                        # circor_ext.js (~30 min) -> learned beat finder (frames, train, held-out validate, cohort) -> hrv_discover.py -> clinical_validation.py -> --bump
node pipeline/circor_ext.js              #   meta/features/circor_ext.jsonl: heartFeatures + heartFeaturesExt + HRV detector vs expert .tsv
python3 pipeline/hrv_discover.py         #   discover.json['hrv']: agreement, repeatability, by band, per-recording values

# ICBHI 2017 (research licence — analysed only, audio NEVER in web/): sound-type + illness models for Analyze
npm run build:icbhi                      # prep -> features (~15 min) -> train -> clinical_validation -> --bump
python3 pipeline/icbhi_prep.py           #   raw/icbhi -> build/icbhi_wav (8 kHz, polyphase) + meta/icbhi_index.json
node pipeline/icbhi_features.js          #   page's analyzeSignal / cycleFeatures / cycleMel / recordingFeatures -> meta/features/icbhi.jsonl
python3 pipeline/train_icbhi.py          #   web/data/icbhi_models.json (sound model + 2 illness models, gates, baselines)

# murmur spectrogram network (PyTorch, CPU) — needs raw/circor; replaced the feature murmur model on Analyze/Batch
npm run build:cnn                        # circor_spec.js (~8 min) -> train_cnn.py (~8 min) -> cnn_eval.py -> clinical_validation -> --bump

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
12b. `icbhi_age.py` adds `discover.json['age']['icbhi']` (same `clock()`; needs meta/features/icbhi.jsonl): a second, independent
    adult test. **No age signal** — ICBHI adults MAE 8.07 vs null 8.03 y (perm p 0.61, 76 patients, 30 held out); COPD on one
    microphone (AKG C417L) 7.39 vs 7.32 y (p 0.81, 32 patients but only 13 held out — **too few to tell**, gain CI −0.60 to +0.52 y;
    the owner's correction 2026-11-17: cite it as uninformative, not as a confirmation). It also stores two PubMed-checked marker references read by the Longevity
    page: Baumert 2019 (PMID 31151958; nocturnal RR ≥ 16/min → all-cause HR 1.18 men / 1.50 women) and Carpenter 1989
    (PMID 2506967; *reported* wheeze → chronic-bronchitis death MR 10.5). No population study links auscultated crackles to
    mortality (searched 2026-11-17), so crackles are not listed.
12c. `hrv_discover.py` also writes `hrv.segmenter.refByBand` (RMSSD quartiles per band, finder vs experts, held-out) and
    `ageTrend`: RMSSD rises with age band, ρ 0.31 (p < 0.001, 290 patients, one value each; experts' beats ρ 0.33). Not
    heart-rate adjusted; a reference, not a risk marker. Longevity page shows both (`#lg-icbhiAge`, `#lg-hrvAge`).
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
- **Model update 2026-11-16 (frozen split, same held-out patients as before — `FROZEN` from the previous testPids):**
  candidates are now (feature set × model), chosen by grouped CV on train. Heart: `heartFeaturesExt` (per-beat
  systolic/diastolic power in 50–150/150–300/300–600/600–1000 Hz, spread, centroids, flatness) won (CV 0.852 vs 0.837);
  held-out AUC **0.804 → 0.825 (Δ +0.021, 95 % CI −0.001 to +0.042, patient bootstrap) — small, borderline**; stored in
  `vsPrevious`. Lung sound: age did not help, unchanged. Disease models now take sound + age and are compared with an
  **age-only baseline** (`ageOnly`, also in the gate): children 0.652 vs age 0.523, adults 0.679 vs age 0.635 — both
  still fail. Valve/age context features did not help the murmur model in CV (tried and dropped).
- `train_models.py` keeps `models.json['robustness']` when the retrained models are bit-identical; otherwise rerun
  `node pipeline/codec_check.js`.
- Compressed uploads (`codec_check.js`): median probability shift 0–1 pt, 95th pct 3–9 pts, a flag changes in
  1.5–4.6 % of decisions.
- The page always says "not a diagnosis" and shows warnings for age outside the training range, features outside the
  training 1st–99th percentile, and unmeasurable features.

## HRV (heart-rate variability) — shipped beat finder failed; learned beat finder passed (2026-11-16)
`hrvFromBeats` / `hrvFromTimes` (page): RMSSD from S1 intervals, intervals > 20 % from the median dropped, ≥ 8 successive
differences; SDNN only with ≥ 30 s of clean intervals. **Pre-set rule (written before any result): ICC ≥ 0.75 and
|bias| ≤ 10 ms against RMSSD from the CirCor experts' S1 onsets (.tsv) on the same annotated window.**
- **Shipped beat finder (`detectHeart`) — failed.** `circor_ext.js`, 1,666 recordings: ICC 0.516 (0.47–0.56), bias
  +8.8 ms, LoA −30 to +48. It constrains beats to ±12 % of the period and misplaces onsets (40.8 % of expert S1 within
  ±50 ms on held-out patients). Heart rate from it is fine (ICC 0.81). `detectHeart` is unchanged (feeds precomputed data).
- **Learned beat finder — passed; this is what the pages show.** Additive code in the page: `segBands` (log power in
  25–60/60–120/120–250/250–500/500–1000 Hz per 10-ms frame) → `segFeatures` (robust per-recording z, 9 context offsets
  + slope = 50 features) → gradient-boosted frame classifier S1/systole/S2/diastole (`modelProba`, gbtree) → `segViterbi`
  (cyclic 4-state, minimum durations = training 10th percentile, geometric tail to the mean) → `segmentHeartSounds(x).s1`.
  Model in `web/data/segmenter.json`, loaded by `ensureSeg()`.
  - Pipeline: `node pipeline/circor_frames.js` (page's `segBands` on all 3,118 recordings → `build/seg/`, ~7 min) →
    `python3 pipeline/train_segmenter.py` (seed 20261116, fixed settings, **trained only on patients outside
    models.json heart_murmur `testPids`**: 1,933 recordings, 1.13 M frames) → `node pipeline/seg_validate.js` (held-out
    patients only, ~12 min; also checks page–Python parity) → `node pipeline/seg_validate.js --cohort` (whole-recording
    RMSSD for the browsable patients) → `hrv_discover.py` (`discover.json['hrv']['segmenter']`) → `clinical_validation.py`.
  - Held-out result (349 patients, 1,185 recordings): S1 within ±50 ms **94.6 %** found, PPV 94.2 % (old: 40.8 / 46.7).
    RMSSD on 909 recordings: **ICC 0.849 (0.818–0.875), bias −1.7 ms, LoA −24 to +21 → passed.** Same 663 recordings
    for both: learned 0.797 vs shipped 0.485. Heart rate ICC 0.95.
  - **By band:** child 0.853 (n 727), adolescent 0.844 (n 73) — shown; **infant 0.297 (n 29) — not shown**; neonates
    and unknown age too few — not shown. `showBands` in discover.json drives this; adults never (no adult data).
  - Not measured: compressed uploads (Analyze page says so). Experts' own RMSSD repeats across valves at ICC 0.71, so
    short clips limit even perfect beats. Expert RMSSD by band (reference): infant ~23, child ~36, adolescent ~43 ms.
  - Changing `segBands`/`segFeatures` means rerunning circor_frames → train_segmenter → seg_validate.
Refs (PubMed-checked): Munoz 2015 PMID 26414314 (10-s RMSSD valid); Tsuji 1994 PMID 8044959 (reduced HRV → mortality in
older adults; SDNN/spectral, not RMSSD); Punjabi 2009 PMID 19688045 (severe SDB AHI ≥ 30: HR 1.46 for all-cause mortality).
No dataset here links HRV to outcome in children — never present it as a risk.

## ICBHI 2017 — sound-type and illness models with explanations (2026-11-16)
Source: `raw/icbhi` → clone of `github.com/aashnajoshi/Respiratory_Sound_Dataset` (copy of the Kaggle ICBHI release; Rocha et al.,
Physiol Meas 2019, PMID 30708353). **Research licence: analysed only; no ICBHI audio, patient page or example clip in `web/`.**
126 patients, 920 recordings, 6,897 annotated cycles (one malformed 5-value row dropped). **Confound:** every healthy patient
is a child on the Meditron, COPD are adults on four devices — so no all-ages illness model, and every model is compared with
age alone and stethoscope alone.
- Page code (additive): `cycleFeatures(A,t0,t1)` (27 per-cycle measurements: band levels vs the recording's median frame,
  spectral shape, tonal/transient evidence of the shipped detectors), `cycleMel` (20 mel bands, mean + spread),
  `scoringCycles(A)` (detected full cycles, else 2.5-s windows), `icbhiScore` / `icbhiFindings` / `icbhiCard` /
  `clinicalNote` / `modelExplain` (logistic contributions = coef × z) and `FEAT_TEXT` (feature → clinical phrase).
- **Sound model** (patient split 60/40, seed 20261116; set × model by grouped CV on train: cycle+mel GB 0.674
  vs cycle GB 0.672 — chosen, though held-out it was slightly worse than the cycle-only set, 0.627 vs 0.638 macro AUC):
  held-out 50 patients / 2715 cycles: crackles AUC 0.611, wheezes AUC 0.68,
  ICBHI score 52.5 % (Se 21.4 / Sp 83.6); device alone 50 %. Weak. Deployment (page's own cycles):
  recording-level crackles AUC 0.603, wheezes 0.639.
- **Shipped rules on ICBHI (external, `resp_icbhi` report):** crackle rule finds **9.1 %** of crackle cycles
  (spec 90.2 %); wheeze rule 49.0 % / 89.2 %; ICBHI score 48.9 %.
- **Illness models** (logistic, fixed beforehand; sound probabilities cross-fitted; patient-grouped 5-fold CV ×10 —
  leave-one-patient-out was dropped because it pushes base-rate baselines below chance):
  children Healthy/URTI/LRI macro AUC **0.634** vs age alone 0.575 → **fails** (LRI class alone 0.791);
  adults COPD/other **0.547** vs age 0.52, **stethoscope alone 0.932** → **fails**. Both are listed
  as "failed validation — not run" on Analyze. Do not switch them on without the owner's explicit decision (precedent:
  `OWNER_ON` in train_models.py), and do not pick a new task after seeing these numbers (forking paths).
- Analyze shows, for lung uploads: per-cycle crackle/wheeze probabilities (Chart.js), findings with each model's held-out AUC
  in words, clinical meaning of each sound (Bohadana, Izbicki & Kraman, NEJM 2014, PMID 24552321), illness-model status,
  and a copyable plain-text clinical summary. Fine vs coarse crackles are deliberately not reported (8 kHz CORSA bias).
- Not obtainable / not usable (2026-11-16): COVID-19 Sounds (NeurIPS 2021 D&B) needs a signed data-transfer agreement and is
  phone audio, not auscultation; the JASA-EL 2024 PKU auscultation set (doi 10.1121/10.0025851) — publisher blocks automated
  access, data URL unknown; BIGR COPD (bigr.nl) is chest-CT features, research/education only, no redistribution.

## Murmur spectrogram network — PyTorch (2026-11-17)
`pipeline/train_cnn.py` (torch from PyPI; download.pytorch.org is not on the allowlist). Input = page's `cnnSpec` (log power of
FFT bins 1–32 = 31–1000 Hz, 10-ms hop) normalised by `cnnNorm` (per-bin median removed, ÷ recording IQR; same formula in
Python). Network: 3 × [conv 3×3 → ReLU → max-pool 2] (12/24/32 ch) → global average → dropout 0.3 → linear; random 3-s crops
in training; a recording = mean of 3-s window probabilities (hop 1.5 s). Seed 20261117, 30 epochs, AdamW 1e-3, pos_weight.
- **Rules fixed before training:** held-out = heart_murmur `testPids` (same patients as the feature model); epoch and flag
  threshold (90 % spec) chosen on 15 % of training patients; deploy only if it passes the gate; replace the feature model only
  if the ΔAUC 95 % CI is above 0.
- **Result (held-out 349 patients, 1185 recordings): AUC 0.91 (0.882–0.939) vs feature model 0.825
  on the same recordings; Δ +0.086 (CI 0.048 to 0.129) → replaced.** Flag: sens 69.6 %, spec 94.2 %. Better in every
  band (`cnn_eval.py` → `byBand`): infants 0.84 vs 0.73 (but spec 78 %), children 0.92 vs 0.84, adolescents 0.92 vs 0.85.
- Page: `cnnSpec` / `cnnNorm` / `cnnConvPool` / `cnnWindow` / `cnnMurmur`, weights in `web/data/cnn_murmur.json` (277 KB),
  `ensureCnn()`. JS = PyTorch to ~1e-7 (e2e parity). ~50 ms per window. Analyze shows it first (`cnnBlock`) with the feature
  model below "for comparison"; Batch's murmur question uses it (`murmur-prev` = feature model). Report `cnn_murmur` on
  Validation (`development`: internal held-out). MP3 (`pipeline/cnn_codec.js`, 465 site MP3 copies vs WAV): median shift 0.1 pt,
  95th pct 0.3 pt, 1 flag changed. Not tested: adults, other stethoscopes.
- Changing `cnnSpec`/`cnnNorm` means rerunning `npm run build:cnn`.

## Round of fixes (2026-11-18)
- **Crackle/wheeze spectrogram network** (`pipeline/icbhi_spec.js` → `train_lung_cnn.py`, PyTorch; 64 bins 31–2000 Hz, one
  2.56-s window per expert cycle, two outputs). Same ICBHI held-out patients: crackles AUC 0.606 vs current 0.611,
  wheezes 0.718 vs 0.68 (Δ CI -0.016 to 0.099). **Did not meet the replacement rule (mean-Δ CI above 0) → not
  deployed**; weights in `meta/cnn_lung.json` (research), report `cnn_lung` on Validation.
- **Breathing rate vs ICBHI expert cycles** (`pipeline/rr_validate.py` → `meta/rr_validation.json`, report `rr_icbhi`, external):
  shipped estimator MAE 5.66/min (limits -17.9 to 20.0), ρ 0.491; candidate (envelope autocorrelation, tuned on
  training patients) MAE 4.94, reduction CI -0.56 to 2.29 → **not adopted**. Children: 8.35 vs 3.67
  (few recordings). Audit item 3 is now measured, not fixed.
- **Murmur rule cut-off refit** (`pipeline/murmur_refit.py` → `meta/murmur_refit.json`, report `murmur_refit`): ratio alone,
  held-out AUC 0.76; shipped 2.5× sens 52.1 / spec 94.8; refitted 1.886× sens 61.2 / spec 88.7.
  Precomputed findings keep 2.5×; the network handles new recordings (open task 1 closed this way).
- **Speed:** CirCor shards of 4 patients (`merge.SHARD_BY`, `pipeline/reshard.py`): one CirCor page load ~1 MB instead of ~5 MB.
- **Phone + accessibility:** `#navToggle` menu (< 900 px), `wrapTables()` (tables scroll in a focusable `.tblwrap`), forms
  stack < 700 px, `REC_TXT` (label-chip text ≥ 5.8:1), `.bad-t` #b42d2d, aria-labels on filters/zoom/speed/region, `for=`
  on Analyze labels, tooltip named, links underlined in text, `prefers-reduced-motion`. Remaining axe items: wavesurfer's
  own scroll regions (inside its shadow DOM), empty table headers (minor), tooltip outside a landmark (moving it into
  `<main>` breaks it — main is re-rendered).
- **Repo:** `.github/workflows/test.yml` runs `npm test`, `test:ahi`, `test:corsa` on PRs; `requirements.txt` pinned.

## Batch analysis and local validation (`#batch`, 2026-11-17)
`renderBatch` / `runBatch` / `batchQuestions` / `batchEval` / `batchWords` (page). Many files (multi-select or a folder) are
analysed in the browser with exactly the Analyze code and models; nothing is uploaded. Optional labels CSV (`parseLabels`):
`file` (required), `label`, `type`, `age`, `site`, `patient`; vocabulary in `LAB_HEART` / `LAB_LUNG` / `LAB_DX`.
- Questions (validated models only, each compared with its published held-out result): murmur (heart_murmur flag),
  abnormal breath sound (lung_sound: not flagged normal), crackles and wheeze (lung_sound per-class flags), crackles / wheeze in
  any cycle (ICBHI cycle model, ≥ 0.5; compared by AUC only). Experimental models: per-class AUC in a red box, framed as failed.
- Statistics in JS, same methods as the Validation page: Agresti–Coull CIs, LR± with log CIs (0.5 correction), Mann–Whitney AUC
  with Hanley–McNeil CI, two-proportion z-test against the published sensitivity/specificity. **Fewer than 10 positives or
  10 negatives → "too few cases", no percentage.** With a `patient` column everything is counted per patient (positive if any
  recording is; flagged if any recording is); without it the page says intervals are too narrow.
- Words (`batchWords`): natural frequencies, CI width in words, LR bands (Jaeschke 1994), post-test probability from the user's
  pre-test estimate (default: the batch's own rate), AUC in words, comparison with the published result. Copy summary /
  copy CSV (downloads are blocked inside the claude.ai artifact, so copy is the export). Thresholds are never refit (rule 4).
- e2e: 20 real CirCor valve recordings + labels → measured and worded; 4 recordings → "too few cases".

## Longevity page — systems view, personal brief, change over time (2026-11-17)
Pattern taken from clinical longevity platforms (longevity-ai.com, read 2026-11-17: system-by-system risk, "where the patient
stands", what changed since the last visit, next steps tied to a finding, a report at a chosen depth, clinician review) and
cut down to what sound measures and what was validated here. **No risk score and no biological-age number** — no dataset has
follow-up, and the acoustic age clock has no adult signal (KAUH and ICBHI).
- `lgSystems(D)`: Heart / Lungs / Sleep cards at the top of `#longevity`; each marker: status pill (validated / weak / not
  reliable / no signal / pilot), what was measured here, the published link. Every number read from discover.json /
  icbhi_models.json.
- `longevityBrief(ctx, level)` / `briefCard(ctx)` on Analyze (short / detailed, copy): heart rate against the healthy CirCor
  reference for the age band, HRV only where validated, wheeze share per cycle; next steps tied to a finding, plain language.
- Batch "Change over time" (`batchTrends`): with `patient` + `date` columns, per-patient line chart and first→last change,
  judged against measurement noise from `hrv.segmenter.repeat` (two valves of one child at one sitting, 271 children:
  **heart rate ICC 0.95, MDC95 10.3 bpm; RMSSD ICC 0.70, MDC95 35.7 ms** — about the size of the median RMSSD, so one-clip
  HRV changes rarely mean anything). Between-day noise is larger; wheeze/crackle shares have no measured repeatability.

## Presentation layer — Home, tooltips, glossary
- **Home (`#home`, the default route)**: hero, live KPIs, four clinical tracks (lungs, heart, sleep apnea, longevity)
  and a "What you can trust today" table. Every number and every status pill is read from `index.json`,
  `validation.json`, `models.json` and `discover.json` — never typed in.
- **Tooltips**: `GLOSS` holds one clinical definition per term (category, title, text, surface forms). `tipify()` tags
  the first mention of each term per card on every render (MutationObserver on `#view`, debounced); `#tipbox` shows it
  on hover and keyboard focus; elements with `data-tip` show their own text (sidebar, example buttons).
  Rules: definitions carry **clinical constants only, never dataset counts**; ambiguous words stay out of the surface
  lists (e.g. "baseline" also means "majority baseline", so only "ghost lane" phrasing is tagged); live readouts
  (`.livebox`), inputs, canvases, `.mono` and `[data-notip]` are never tagged.
- **Glossary (`#glossary`)**: every `GLOSS` entry by category, with a filter.
- **Held-out examples on Analyze**: `train_models.py` exports `testPids`; "Try a real example" loads a recording of a
  patient the model never saw and shows the expert label beside the model output.

## Writing style for site copy (owner, 2026-11-15)
Plain, factual clinical prose. No slogans, rhetorical questions, aphorisms or asides ("which is the point").
No novelty claims ("new", "first", "unique") — they cannot be verified; describe the analysis type instead
(exploratory, hypothesis, replication, validation). Avoid "honest", "simply", "powerful", "seamless", "robust",
"at a glance", "in 30 seconds". Safety statements stay explicit ("not a diagnosis", "failed validation").
Text colour `--soft` is kept at ≥ 4.5:1 contrast on every background (WCAG 2.2 AA).

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
| CirCor DigiScope 1.0.3 (PhysioNet) | in use: 120 browsable patients; all 941 validated; **raw re-downloaded 2026-11-16** (zip from physionet.org, SHA256 verified; `raw/circor/circor-heart-sound-1.0.3` is a symlink to the extracted folder) | **ODC-By 1.0** (checked: `LICENSE.txt` in the release — attribution, republishing allowed) | 942 patients, 3,163 recordings (3,118 primary takes used; `_1/_2/_3` repeats and `Phc` skipped). 4 kHz, up to 4 valve sites. Per-valve murmur truth plus `.tsv` S1/systole/S2/diastole segmentation — the only set here that can score the cardiac detectors |
| PSG-Audio (Korompili et al., Sci Data 2021; Science Data Bank 10.11922/sciencedb.00345) | in use: Discoveries 12 only, **pilot of 10 patients / 44 h** | CC BY 4.0 (HF mirror card + scidb page) | 212 adults, Athens sleep lab, PSG scored by technician + physician; tracheal contact mic + ambient mic at 48 kHz in hourly EDF (V3), `.rml` events + **UserStaging** (ignore MachineStaging). Read from the partial HF mirror `dust-systems/psg-audio` (the scidb FTP logins are time-limited). File names literally contain `%5B001%5D` — escape `%` in URLs. **Audio never goes to `web/`** (overnight audio can hold speech) |
| PhysioNet overnight / sleep sets | wanted | varies | Still useful for healthy sleepers and children — PSG-Audio is ~89 % severe OSA, adults only |
| ICBHI 2017 | **in use: analysed only** (Analyze models + `resp_icbhi` / `icbhi_models` reports); via a GitHub copy, see the ICBHI section | research only | 126 patients, 4 devices, per-cycle crackle/wheeze labels. Never republish its audio. Lifespan curve (Discoveries 10) not yet redone with it |
| PhysioNet/CinC 2016 | in use: adult external test (murmur network, HR vs ECG); 3 recordings in the lean brief | ODC-By 1.0 | 3,240 recordings, 6 databases, 2 kHz, abnormal/normal labels; set a has ECG. Zip sha256 in meta/cinc2016_validation.json |
| EPHNOGRAM 1.0.0 | in use: adult HR/HRV vs research-grade ECG, 14 resting records | ODC-By 1.0 | 8 kHz PCG + ECG, healthy men 23–29; checksums verified (SHA256SUMS.txt); analysis only |
| RespiratoryDatabase@TR | in use: COPD severity test; 1 recording in the lean brief | CC BY 4.0 | 42 COPD adults GOLD 0–4, 12 channels, 4 kHz, Littmann 3200 |
| HF_Lung | optional | V1_IP CC BY-NC | |
| NHANES 1999–2012 + NCHS Linked Mortality (2019) | in use: lite Longevity model | public domain (US government; cite NCHS) | pulse, self-reported wheeze, BMI, smoking; `nhanes_prep.py` |

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

**Google models (HeAR, MedGemma) — access granted and both tested 2026-11-18/19.** HeAR: see Discoveries 9 /
`pipeline/hear_probe.py` (research only, too large for the page). MedGemma 4B: replaced by the fixed-template written
summary (see "MedGemma report drafts").

**Which Discoveries entries this holds up** (numbers are the position in `DISCOVER_MAP`):

| # | Entry | Status | Blocker |
|---|---|---|---|
| 9 | HeAR embeddings vs this platform's models | **done 2026-11-19** (ICBHI + CirCor) | wheezes better (0.737 vs 0.68, CI above 0); crackles no difference; murmur worse than the network. Research only — not in the page |
| 12 | Overnight apnea / AHI vs PSG | **pilot result** (PSG-Audio, 10 patients) | more patients, healthy sleepers, children |
| 10 | Breath sounds across the lifespan | *has a result*, device-confounded | needs ICBHI (children→elderly on one device) for a true curve |
| 11 | Adult diseases | *has a result* from KAUH | more adult data would strengthen it; not blocked |
| 14 | Generalisation across stethoscopes | *has a result* (SPRSound vs KAUH) | per-device recalibration is open work, not missing access |
| 13 | Murmur / S3 on real heart sounds | **done** — CirCor 1.0.3 ingested and validated | — |

9 now has a result (HeAR on ICBHI and CirCor); 12 has a pilot result. 10, 11 and 14 are published with their limits stated;
13 was completed and its result is live.

## Source notes (checked 2026-11-08)

- **ICBHI 2017** — the official host still does not respond, but a public GitHub copy of the Kaggle release
  (`aashnajoshi/Respiratory_Sound_Dataset`) is now used (2026-11-16). Rule 3: research-only, analysed only, no audio in `web/`.
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
4. **Infant cardiac detection was close to useless** for the rule (the murmur network reaches AUC 0.84 in infants, specificity 78 %) — 3/25 murmurs found (12 %) against 45 % in
   children. At a median 143 bpm diastole is too short for the S1/S2 separation the murmur rule
   needs. Report infants separately rather than inside a pooled figure.

## ICBHI provenance and the age-clock caveat (2026-11-17)

- ICBHI came from **github.com/aashnajoshi/Respiratory_Sound_Dataset**, a third-party copy of the
  Kaggle release — not from the challenge organisers, who are unreachable. `icbhi_prep.py` now
  records a **manifest sha256** (name + size + content of every source file, in name order) in
  `meta/icbhi_index.json` and refuses to rebuild from a different source set unless
  `ICBHI_ACCEPT_NEW=1`. That establishes *reproducibility*, not authenticity: the mirror publishes
  no checksum to verify against, unlike CirCor from PhysioNet.
- Rule 3 is respected — no ICBHI audio in `web/`, verified. `web/audio` holds one bundle per
  browsable patient and nothing else.
- **The age-clock replication's controlled arm is underpowered.** ICBHI adults (30 held out) and
  the COPD/one-microphone arm (**13 held out**) both show no age signal, replicating KAUH. But the
  n = 13 interval runs −0.60 to +0.52 y, too wide to rule a real gain out. The Longevity card now
  shows the held-out count beside the group total, labels that arm *too few to tell* rather than
  *no age signal*, and states that the conclusion rests on the arms with enough patients. Absence
  of evidence, not evidence of absence — don't cite the n = 13 arm as independent confirmation.

## Open tasks (priority order)
**`OPEN_WORK.md` is the current list** (2026-11-18): every unfinished item with what is missing and the steps to finish it.
The notes below are the history behind tasks 1–8; tasks 1 (murmur) and 3 (HeAR token saved, access pending) moved on.
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
8. Redo Discoveries 10 (breath sounds across the lifespan) with ICBHI's AKG C417L recordings (one device, wide ages) —
   mind that diagnosis tracks age there too.

## Publishing as a claude.ai artifact (if asked)
- The artifact hosts at most **512 files**.
- One publish call takes **≤256 paths** (files + removals) and **≤64 MB**.
- Only text and HTML/JSON extensions are served (no `.bin`, no `.mp3`), which is why audio is base64 JSON and the analysis is sharded.
- Current artifact: claude.ai/artifact/AwfP2h99UAWWWchAJFhjUW (published 2026-11-16 from `web/`; now 477 files after the CirCor
  re-shard: page, 53 data files, 423 audio bundles; publish in batches of ≤ 58 MB). After a re-shard, remove the old shard
  paths that no longer exist in the same publish.
- Older artifact claude.ai/artifact/JawqvH3vX7cXXqbo94SXDB (slug `8e68b339-2080-481b-822f-6581919abf92`) holds a
  pre-CirCor build from before the first commit; it is stale and was left untouched.
- Publish `web/` only. Never publish `raw/`, `build/`, `meta/` or `tests/`.

## MedGemma report drafts (2026-11-18) — research, not on the pages
`python3 pipeline/medgemma_draft.py [n]` (google/medgemma-4b-it, HAI-DEF terms, CPU ~9 GB RAM, ~1 min per draft) drafts a
summary from the page's structured findings only (no audio, no expert label, no diagnosis; heart recordings get cardiac
findings only). Mechanical checks: numbers traceable, "not a diagnosis", no diagnosis named, no clinician-implying words,
findings attributed to the right site. **12 drafts: numbers always traceable, but 3 of 12 misattribute a finding**
(and one more is ambiguous on reading) → not fit to use unreviewed. Do not put drafts on the pages.
**Owner's decision 2026-11-19: fixed template instead** — `writtenSummary(p)` (patient page, "Written summary" card): every
site's precomputed findings in fixed sentences, stored values unrounded, cardiac findings only on heart recordings, "not a
diagnosis". Same checks, same 12 patients: template 12/12, MedGemma 9/12 (`pipeline/summary_export.js` → `--template`).

## Adult external tests and the lean clinical brief (2026-11-21)
Three new open adult sets, licences checked, raw data on the private remote (`raw/cinc2016`, `raw/respdbtr`, `raw/ephnogram`).
All tests run the page's FROZEN code; rules are in each script's header and were fixed before the run (`npm run validate:adults`).
- **PhysioNet/CinC 2016** (ODC-By; Liu et al., Physiol Meas 2016, doi 10.1088/0967-3334/37/12/2181; 3,240 recordings, 6 adult
  databases, label = abnormal vs normal) — `cinc_run.js` → `cinc_validate.py` → `meta/cinc2016_validation.json`.
  Murmur network within database: a (valve prolapse, benign murmurs) AUC 0.687 (0.634–0.741);
  c (echo AS/MR, ~75 y, n 31) 1.0; b 0.75; e 0.743; d 0.553; f 0.531.
  **Pre-set rule (CI lower bound > 0.70 in both a and c) not met.** The children's flag level does not transfer: set a sens
  20.5 %, set f spec 0.0 % — per-stethoscope calibration needed. Heart rate vs set a's ECG: ICC 0.328 (fails), mostly
  where gqrs gives implausible resting ECG rates (104–157 bpm with heart sounds at 52–80) — not proven, so it stays a fail.
- **EPHNOGRAM** (ODC-By; PhysioNet doi 10.13026/tjtq-5911; research-grade ECG + PCG, healthy men 23–29) — 14 resting records
  (ECG "Good", PCG not saturated), 30-s windows — `eph_validate.py prep` → `eph_run.js` → `eph_validate.py` → `meta/eph_validation.json`.
  Heart rate: bias -0.5 bpm, LoA -10.3 to 9.3, **ICC 0.798 → misses the pre-set 0.90**; per session (median of
  windows, 14 records) ICC 0.999, LoA -1.2 to 0.9 bpm (reported beside, not the pre-set unit). RMSSD ICC 0.734,
  bias 8.8 ms → **fails** (rule 0.75). Adult HRV is not shown anywhere.
- **RespiratoryDatabase@TR** (CC BY 4.0; Mendeley 10.17632/p9z4h98s6j.2; 42 COPD adults GOLD 0–4, 12 channels, Littmann 3200)
  — `respdbtr_run.js` → `respdbtr_validate.py` → `meta/respdbtr_validation.json`. Breath-sound model's abnormal probability
  vs GOLD: ρ 0.105 (CI -0.194 to 0.395, p 0.51) → **does not track severity**. The ICBHI cycle model
  flags wheeze in 60–80 % of cycles at every stage (not calibrated for this stethoscope).
- **Lean clinical brief `lean/`** (owner, 2026-11-21: a short version for a clinician audience, the full site kept):
  `python3 pipeline/lean_build.py` → `lean/data/brief.json` + 10 audio bundles; `npm run test:lean`. Only skills that at least
  mostly work, each output with its own validation line; batch of 10 drawn at random per group (seed 20261121) before
  looking at outputs; conclusions and the two charts (adult HR vs ECG, adult murmur AUC per database) computed from the
  result files. EPHNOGRAM audio is not in the brief (analysis only). Published as its own artifact, separate from web/.

## Disease atlas, longevity model, daily retraining (owner, 2026-11-23) — lite app pages "Which disease?" and "Longevity"
- **Lite player annotations**: `annotate()` in `lite/index.html` draws what is heard as wavesurfer regions, computed in the browser
  by the engine — S1/S2 (learned beat finder path), murmur-like windows (network ≥ its flag level), inspiration/expiration
  (`analyzeSignal` cycles), wheeze (tonal segments < 0.3 s apart joined, median pitch). Toggle per layer; labels only on regions
  ≥ 4 % of the clip.
- **`pipeline/disease_train.py` → `meta/disease_model.json`**: children (SPRSound, ICBHI children) and adults (KAUH mode D, ICBHI
  adults, CB/emphysema, RespiratoryDatabase@TR GOLD ≥ 1), 9 recordingFeatures averaged per patient, gradient boosting, patient-grouped
  5-fold CV, baselines database-only / age-only, leave-one-database-out. Rules in the header, fixed before the first run.
  **Result: fails in both** — children macro 0.63 (< 0.70, LODO at chance); adults 0.82 but database-only 0.87 and age-only 0.85
  (it learns the collection). Batch: out-of-fold top class matches 2 of 5. Gap analysis with PubMed refs: `docs/DISEASE_GAPS.md`.
  Never present its output as a diagnosis.
- **`pipeline/nhanes_prep.py` → `meta/nhanes_cohort.csv`** (NHANES 1999–2018 DEMO/BPX/BMX/SMQ/RDQ + NCHS linked mortality to 2019,
  public domain; raw in `raw/nhanes`, sha256 in `meta/nhanes_files.json`). **`pipeline/longevity_train.py` → `meta/longevity_model.json`**:
  ages 40–79, train 1999–2006 / test 2007–2012, Cox(age, sex, resting HR/10, wheeze, BMI, smoking) vs age+sex and vs gradient
  boosting. Test C 0.764 vs 0.735 (ΔC +0.029, CI 0.023–0.035); GB worse (−0.010) → Cox deployed. HR per 10 bpm 1.17, wheeze 1.46.
  Top risk decile over-predicted (24 % vs 18 %). The page's `coxRisk` reproduces lifelines (`cox.check`, e2e). Improvement
  scenarios = Reimers 2018 (PMID 30513777) exercise effects −3.3 / −5.8 bpm; association carried through the model, not a trial
  effect. No improvement is predicted for wheeze (no trial gives one). Survey weights not used.
- **Literature on these pages (owner, 2026-11-23): articles from the last year only** (Nov 2025 – Nov 2026, PubMed-checked,
  `REFS` in disease_train.py, `refs` in longevity_train.py, text in `docs/DISEASE_GAPS.md`). One exception, labelled on the page:
  Reimers 2018, because it is the only meta-analysis giving the exercise effect in beats/min. Chen 2026 (Open Heart) replicates
  HR 1.17 per 10 bpm; Yu 2026 (Heart Lung) finds the asthma–wheeze cluster not linked with mortality after full adjustment — the
  Longevity limits say so. Dataset papers stay in README as licence attribution.
- **`pipeline/daily_train.py`**: input sha256 → disease_train → longevity_train → `meta/daily/<date>.json` (summary, comparison with
  the previous day; exits 2 if metrics move while inputs did not) → lite_build. `--refresh-nhanes` re-downloads from CDC,
  `--remote` uploads `meta/daily` to the private store. It never adds a database (licence check is a person's job), never
  changes a pass rule, never publishes or pushes.

## AuscultaDB Lite `lite/` (owner, 2026-11-22) — the app, lighter, for a clinician audience
Same shell as the full app, five pages only: Home (validated measurements), Batch of 10 (the brief's pre-registered batch),
patient pages (player with beat ticks or detected wheeze segments; each finding with its validation line), Analyze (only
validated models: murmur network, heart rate, HRV for children/adolescents, wheeze), Conclusions (data-driven, 2 charts),
Data and methods. `lite/engine.js` is the full app's analysis code copied function by function by `pipeline/lite_engine.js`
(headless Chromium, Function.prototype.toString, transitive closure) — rerun it after any analysis change; `npm run test:lite`
checks engine = full app on an original WAV (murmur probability identical) and Analyze vs stored results on the MP3 copies.
Build: `npm run build:lite` (lean_build → lite_engine → lite_build → test). Published as its own artifact; web/ unchanged.

## Remote data store (2026-11-18) — `pipeline/remote.py`
The machine's disk is small and replaced between sessions, so `raw/` and the expensive `build/` folders live in a
**private** Hugging Face dataset repository (`$AUSCULTADB_REMOTE`, default `<HF user>/auscultadb-data`). Same paths as on
disk; folders with more than 1,000 files are stored as ~1 GB tar parts under `archives/<folder>/` (HF allows ≤ 10,000 files
per directory). `MANIFEST.json` holds size, sha256, licence and upload time for every file.
```bash
python3 pipeline/remote.py status                     # remote vs local, per folder
python3 pipeline/remote.py pull raw/circor build/cnn  # restore after a fresh machine (sha256-checked)
python3 pipeline/remote.py push raw/icbhi             # upload new or changed files
python3 pipeline/remote.py free raw/psg_audio         # delete local copies whose sha256 matches the remote; keeps anything else
```
- Stored (9.7 GB, 13,307 files): raw/circor, raw/icbhi, raw/psg_audio, build/cnn, build/icbhi_cnn, build/icbhi_wav, build/seg.
  Round trip checked (free → pull → every sha256 matches).
- The script **refuses to push to a repository that is not private** — ICBHI is research-only (rule 3). Never make it public.
- Auth: the HF token is a network secret of the `auscultadb` environment; code uses `HF_TOKEN=proxy-injected`.
- `raw/psg_audio` was freed locally on 2026-11-18 (disk); `pull raw/psg_audio` before rerunning psg_audio_validate.js.

## Cloud environment
This repo's sessions run on the personal environment **auscultadb** (not Default): package registries plus
`huggingface.co`, `cdn-lfs.huggingface.co`, `cdn-lfs-us-1.hf.co`, `cas-bridge.xethub.hf.co`, `cas-server.xethub.hf.co`,
`us.aws.cdn.hf.co` (HF downloads and uploads), `www.scidb.cn` and `physionet.org` (CirCor). A Hugging Face token is saved
as a network secret for `huggingface.co`; HeAR and MedGemma 4B-it access were granted 2026-11-18.
