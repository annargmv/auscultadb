# AuscultaDB for a new programmer

A short orientation. `README.md` lists every feature, `CLAUDE.md` holds the full working notes and `OPEN_WORK.md` lists what is
unfinished. Read this first, then those.

## What it is
A research and teaching platform for breath and heart sounds. Every patient is a real, openly licensed recording. You can
listen by chest site or heart valve, see what the detectors found, and read how accurate each detector and model is. It is
**not a diagnostic device**, and the pages say so wherever a result is shown.

## How it is built
- **The site is one static page**, `web/index.html` (~370 KB): the UI and all signal analysis in plain JavaScript. It has no
  build step and no server code. Charts come from libraries fed with values computed from the audio (wavesurfer.js, Web
  Audio, Meyda, Chart.js).
- **Data lives in JSON next to it**, under `web/data/`:
  - `index.json` lists the patients; its `version` is the browser cache key;
  - `d/<n>.json` holds the precomputed analysis, in shards;
  - `discover.json`, `validation.json` and `models.json` hold the results;
  - `web/audio/` holds the MP3s as base64 JSON.
- **The pipeline (`pipeline/`, Python + Node) makes that data.** The key idea is that `precompute.js` and the validation
  scripts run **the page's own JavaScript functions** in headless Chromium on the original recordings. So what the
  browser shows and what was validated come from the same code. If you change analysis code in the page, rerun the
  pipeline step that uses it.
- **Models are trained in Python** (scikit-learn, PyTorch), exported to JSON and run in the browser. End-to-end tests
  check that the browser reproduces the Python output to about 1e-7.
- **Large files are kept out of git:**
  - `raw/` (downloaded datasets) and `build/` (intermediates) are git-ignored;
  - they are stored in a **private** Hugging Face dataset repository through `pipeline/remote.py`
    (`status / push / pull / free`, sha256-checked);
  - `meta/` holds the small derived tables that are kept in git.

| Folder | What is in it |
|---|---|
| `web/` | the published site: page, data, audio |
| `pipeline/` | dataset preparation, precompute, training, validation, statistics |
| `meta/` | indexes, cohort selections, per-recording features, research results |
| `tests/` | Playwright end-to-end tests; screenshots in `tests/out/` |
| `raw/`, `build/` | local only; restore them with `python3 pipeline/remote.py pull <folder>` |

## What has been done
- **Data:**
  - SPRSound children, KAUH adults and CirCor heart sounds, all browsable;
  - chronic bronchitis / emphysema and PSG-Audio overnight recordings, used in analyses only;
  - ICBHI 2017, analysed only, because it is licensed for research and its audio may not be published.
- **Patient pages:**
  - a live player whose "ghost" lane shows a normal baseline aligned breath by breath;
  - chest and heart maps;
  - findings with type, phase and site;
  - a breathing-flow graph with apnea/AHI (AASM rules);
  - a **written summary** built from the findings by a fixed template.
- **Analyze and Batch:** upload one recording or many; the browser runs the same detectors and models. With a labels CSV,
  Batch measures accuracy on your own files and writes each number out in words.
- **Validation page:** one diagnostic-accuracy report per detector or model, laid out like a STARD study (sensitivity,
  specificity, likelihood ratios with 95 % CIs, subgroups, limits, verdict).
- **Discoveries page:** 16 cohort analyses. The text is driven by the numbers, and weak results are labelled weak.
- **Longevity page:** markers with a published mortality link, and how well this platform measures each one. It gives no
  risk score, because no dataset here has follow-up.

## What works and what does not (held-out patients)

| Result | Status |
|---|---|
| Murmur spectrogram network (PyTorch), CirCor | AUC 0.91; used by the pages |
| Learned heart-beat finder for HRV | ICC 0.85 against experts; shown for children and adolescents, not infants |
| Breath-sound type model | AUC 0.72; used by the pages |
| "Which disease" models (lungs) | **failed validation**; shown only as experimental, in a red box |
| Shipped crackle rule on ICBHI | finds 9 % of crackle cycles; weak |
| Breathing-rate estimate | error about 5.7 breaths/min; documented, not fixed |
| Apnea scoring on real nights (pilot, 10 adults) | finds 45 % of scored events; under-counts AHI |
| Google HeAR embeddings | better for wheezes, not for crackles or murmurs; research only, too large for the page |
| MedGemma report drafts | 3 of 12 put a finding at the wrong site; replaced by the template |

## Rules you must follow
1. **Real public data only.** No synthetic patients; synthetic audio is allowed only in tests.
2. **Every graph is drawn from the actual sound** by a library. Never use a static picture.
3. **Check the licence before using a dataset.** ICBHI and PSG-Audio audio never go into `web/`, and the remote store
   stays private.
4. **Fit on one set of patients and report on another.** Never tune on the patients you report, and decide the pass rule
   before you see the result.
5. **Small changes.** After each one, run `npm test` and look at the screenshots.
6. **After any script that writes `discover.json`, run `python3 pipeline/merge.py --bump` last**, so browsers pick up the
   new data.
7. **Ask the owner before any push, merge or publish.**

## First day
```bash
npm install && npx playwright install chromium
pip install -r requirements.txt
npm run serve        # http://localhost:8000 (http, not file://)
npm test             # about 3 minutes, 59 checks
```
Then read the "Code map" in `CLAUDE.md` to find your way around `web/index.html`, and pick an item from `OPEN_WORK.md`.
Each item says what is missing and lists the steps to finish it.
