# Open work — what is not done, what is missing, and how to do it

Last updated 2026-11-18. Every item: **status**, **what is missing**, **steps to apply it**. Items are ordered by how soon
they can start. Rules from `CLAUDE.md` apply to all of them: real public data only, licence checked first, fit on one set
and report on another, and nothing is shown on the pages unless it passes its gate.

---

## A. Ready to start as soon as one thing is provided

### 1. Google HeAR embeddings for breath sounds (CLAUDE.md task 3, Discoveries 9)
- **Status:** blocked on model access. A Hugging Face token is saved in the `auscultadb` environment (network secret for
  `huggingface.co`, works — `whoami` returns 200), but the model answers *"you are not in the authorized list"*.
- **Missing:** approval for `google/hear-pytorch` on the owner's Hugging Face account.
- **To apply:**
  1. Owner: open https://huggingface.co/google/hear-pytorch logged in, accept the Health AI Developer Foundations terms.
  2. `pip install transformers` (HeAR loads with `trust_remote_code`), download `pytorch_model.bin` (~ size on the card).
  3. New `pipeline/hear_embed.py`: resample each recording to 16 kHz, cut 2-s windows, embed, mean-pool per recording
     (CirCor from `raw/circor`, ICBHI from `build/icbhi_wav`; SPRSound audio is not on disk, see item 6).
  4. Linear probe with the same patient splits (`models.json` / `icbhi_models.json` testPids), compared with the current
     features and with the spectrogram networks on the same held-out patients; report on Validation and Discoveries 9.
  5. HeAR cannot run in the browser (size); results are research only unless a small distilled model passes the gate.

### 2. MedGemma to draft the written report (CLAUDE.md task 4) — **tested 2026-11-18; not fit to use unreviewed**
- **Done:** `pipeline/medgemma_draft.py` runs `google/medgemma-4b-it` offline (CPU, bf16, ~1 min per draft) on the page's own
  structured findings — never the audio, the expert label or the diagnosis. Checks per draft: every number appears in the
  input; it says "not a diagnosis"; no diagnosis named; no phrasing that implies a clinician listened; **every finding named
  in a sentence exists at every site that sentence names**. Results in `meta/medgemma_drafts.json`
  (`--recheck` re-scores stored drafts).
- **Result (12 drafts, 4 each SPRSound / KAUH / CirCor):** all numbers traceable, but **3 of 12 put a finding at the wrong
  site or contradict themselves**, and 1 more lists rates without saying which filter mode they come from (found by
  reading, not by the check). The first run, before heart recordings were limited to cardiac findings, had 5 of 12 wrong.
- **Missing:** an owner decision. The drafts read fluently but are wrong often enough that each would need a clinician to
  check it line by line, so they are **not shown on the pages**. Options: drop the idea; try a larger model (27B — too big
  for this machine); or generate the summary with a fixed template from the same structured findings (no model, no errors).

### 3. Remove or approve network hosts added during research — **done 2026-11-18**
- Removed `pubs.aip.org`, `bigr.nl`, `www.longevity-ai.com`, `longevity-ai.com`; added `cdn-lfs-us-1.hf.co` and
  `cas-server.xethub.hf.co` (Hugging Face uploads for the remote data store). `download.pytorch.org` still not approved —
  PyPI torch works.

### Remote data store — **done 2026-11-18**
- `pipeline/remote.py` keeps `raw/` and the expensive `build/` folders on a private Hugging Face dataset repository
  (9.7 GB, 13,307 files, sha256 manifest). Large downloads in items 4–7 no longer depend on this machine's disk: push what
  you fetch, `free` it locally, `pull` it when needed. See CLAUDE.md "Remote data store".

---

## B. Needs data that is not on this machine

### 4. Apnea flow proxy: noise-floor correction (CLAUDE.md task 5, Discoveries 12)
- **Status:** measured problem (misses 55 % of scored events, under-counts AHI by 26/h; detection tracks the noise
  headroom). Fix not written.
- **Missing:** new PSG-Audio patients — one set to fit the correction, a second to report on (the 10 pilot patients are
  already used for the reported numbers). Disk: the HF mirror's EDF files are ~0.7 GB per patient-hour before
  decimation; ~8 GB free now.
- **To apply:** make room with `python3 pipeline/remote.py free build/cnn build/icbhi_cnn` (stored on the private remote,
  restored with `pull`); download 10 + 10 new
  patients a few at a time with `pipeline/psg_audio_prep.py`, keeping only the 8-kHz microphones; add the floor
  subtraction as an *additive* option in `computeFlow` (low percentile of band RMS); fit on set A, report on set B with
  `psg_audio_validate.js` / `psg_audio_discover.py`; keep `npm run test:ahi` passing.
- **Also possible:** PSG-Audio's flow channel gives a true overnight breathing rate to test the candidate breathing-rate
  estimator (item 8). The pilot prep kept only the two 8-kHz microphones, so the flow (and the `.rml` breath timing) must be
  downloaded again from the HF mirror for the same 10 patients; `raw/psg_audio` itself is on the remote (`pull raw/psg_audio`).

### 5. Healthy sleepers and children for apnea specificity (CLAUDE.md task 7)
- **Status:** blocked — no open dataset found with breath sound recorded alongside a scored sleep study in healthy adults
  or children (PSG-Audio is ~89 % severe OSA, adults only).
- **Missing:** a dataset. Search PhysioNet / NSRR for sets with a tracheal or ambient microphone channel (NSRR sets need a
  data-use agreement — licence check first).

### 6. Make all 860 SPRSound children browsable (CLAUDE.md task 6)
- **Status:** 191 browsable. Blocked by two things.
- **Missing:** (a) the SPRSound audio (`raw/sprsound`, 1.2 GB GitHub clone, deleted earlier); (b) room on the claude.ai
  artifact — it holds at most 511 files and the site has 477; one audio file per patient would add ~670.
- **To apply:** change the audio layout from one bundle per patient to multi-patient bundles (e.g. 10 patients per file;
  `fetchAudioBytes` and `merge.py write_patients`); `npm run fetch:sprsound`; write `pipeline/select_cohort.py` (same schema
  as `meta/cohort_spr.json`); `node pipeline/precompute.js` for the new children (hours); `merge.py`; republish.

### 7. Better crackle detection (CLAUDE.md task 2)
- **Status:** weak everywhere — shipped rule finds 9 % of ICBHI crackle cycles; ICBHI cycle model AUC 0.61; the PyTorch
  lung network (2026-11-18) did not improve crackles (0.606).
- **Missing:** more and better-labelled crackle data. SPRSound has event-level crackle timings (CC BY) but its audio is not
  on disk; HF_Lung has event labels but is CC BY-NC (non-commercial — owner decision).
- **To apply:** re-clone SPRSound (item 6 step); train the spectrogram network on SPRSound events + ICBHI cycles with the
  existing frozen held-out patients; replace the cycle model only if the ΔAUC CI is above 0 (same rule as now).

### 8. Breathing-rate estimator (clinical audit item 3)
- **Status:** measured, not fixed. Shipped MAE 5.7/min vs ICBHI experts; a candidate (envelope autocorrelation) reached 4.9
  but the gain's CI included 0, so it was not adopted. It looks much better in children (3.7 vs 8.4/min, 32 recordings).
- **Missing:** a second labelled set to confirm the candidate.
- **To apply:** score the frozen candidate (`pipeline/rr_validate.py`) on PSG-Audio flow-derived breathing rate (item 4) or
  on more child recordings; adopt only if confirmed there. `scoreFlow` relies on the rate, so rerun `npm run test:ahi`.

### 9. Lung illness models (all failed their gate)
- **Status:** SPRSound/KAUH pattern models run as "experimental" by the owner's choice; ICBHI illness models are off.
  Diagnosis is confounded with age and stethoscope in every set here.
- **Missing:** diagnosed recordings with ages and devices balanced across diagnoses, ideally several hundred per class.
- **To apply:** add such a dataset (licence first), retrain with the same gate (beat age-only and stethoscope-only).
  Do not lower the gate.

### 10. Heart sounds in adults; S3
- **Status:** the murmur network is trained and tested on Brazilian children only; S3 cannot be validated (CirCor has no S3
  labels).
- **Missing:** an adult heart-sound set (e.g. the PhysioNet/CinC 2016 challenge database — check its licence) and a set with
  S3 labels.
- **To apply:** run `cnn_eval.py`-style scoring of the frozen network on the new set as an external test; report per age.

### 11. Longevity as an outcome
- **Status:** the platform reports markers and references only; no risk score or biological age (no follow-up in any
  open auscultation dataset; no adult acoustic age signal in KAUH or ICBHI).
- **Missing:** recordings linked to years of follow-up (deaths, cardiovascular events).
- **To apply:** only with such a cohort (likely a research collaboration and ethics approval); pre-register the analysis.

### 12. Breath sounds across the lifespan (Discoveries 10, CLAUDE.md task 8)
- **Status:** replaced by the ICBHI age-clock replication (no adult age signal). A true curve needs healthy people of all
  ages on one stethoscope — ICBHI's healthy patients are all children.
- **Missing:** healthy adults recorded on the same device as healthy children.

---

## C. Engineering work, no blocker — just time

### 13. Dark mode
- **Status:** not started. **Missing:** colour tokens — the page has ~310 hard-coded colours (`#fff` ×31, chart colours,
  chips, canvas drawing in JS).
- **To apply:** move every colour to CSS variables on `:root`, add `@media (prefers-color-scheme: dark)` values, re-check
  contrast (WCAG 4.5:1) and Chart.js / canvas colours in both themes, add an e2e screenshot in dark mode. A day of careful
  work; do it as its own change so a contrast regression is easy to find.

### 14. Remaining accessibility warnings (axe) — **done 2026-11-18**
- wavesurfer scroll areas get `tabindex=0`, `role=region` and a name after each waveform loads (`wsA11y`); empty `<th>`
  cells carry visually hidden labels (`.sr-only`); the tooltip box is mounted in an `<aside aria-label="Term definitions">`
  outside `#view`, so it sits in a landmark and survives re-renders.

### 15. First-load size
- `index.json` is ~690 KB because it lists every site of every patient. Split it into a small list summary and per-patient
  site details loaded with the shard; keep `version` as the cache key.

### 16. Breathing phase detection (Pal & Barney 2024)
- Could be implemented, but no dataset here has inspiration/expiration timings to validate it. Needs phase-annotated
  recordings first.

### 17. Continuous integration — **done**
- `.github/workflows/test.yml` ran green on the pull request and on the merge to `main` (2026-11-18, runs 1 and 2).
