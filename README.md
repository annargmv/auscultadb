# AuscultaDB

Respiratory and cardiac sound database and analysis platform built on open, real recordings:
- **SPRSound** — 191 browsable children, 6,567 recordings analysed;
- **KAUH** — 112 adolescents and adults, 12–90 y;
- **CirCor DigiScope** — 120 browsable paediatric heart-sound patients, all 941 used to validate the cardiac detectors;
- **Chronic bronchitis / emphysema** (Mendeley 9hdrddjpny) — 234 recordings on the same stethoscope as KAUH, for a device-matched disease comparison (Discoveries only).

What it does:
- patient and doctor pages;
- a chest map to listen by site (or by stethoscope filter mode), and a heart map to listen by valve;
- anomaly findings with type and location;
- a live player with a cycle-aligned "ghost" of the baseline;
- a breathing-flow graph with apnea/hypopnea and AHI (AASM rules);
- a Discoveries page of cohort analyses, including a held-out validation of the murmur detector against expert annotations.
- a biological-age test (Discoveries 16): can auscultation support an "age gap" longevity marker, as ECG can? Held-out
  result: breath sounds carry essentially no age signal; heart rate tracks age band; heart rate for age does not predict
  CirCor's clinical outcome. No open auscultation set has survival data, so longevity itself is not measured.
  Rebuild with `npm run build:age`. References: Lima et al., Nat Commun 2021 (PMID 34433816); Zhang et al., CMAJ 2016
  (PMID 26598376).
- a **Validation** page written for clinicians: one STARD-style report per detector (clinical question, intended use,
  who was tested, reference standard, exact rule, sensitivity / specificity / PPV / NPV / likelihood ratios with 95 % CIs,
  subgroups, bedside meaning, limits, and whether the test data were independent of tuning), plus two research
  machine-learning models trained on one patient split and scored on another, with a subgroup fairness audit.
  Statistics come from **Seismometer** (Epic Systems, open source, BSD-3-Clause, https://github.com/epic-open-source/seismometer).
  Rebuild with `npm run build:validation`.
- a **Longevity** page: the markers with a published mortality link that sound could carry, how reliably this platform
  measures each one, and the biological-age test. It states plainly that nothing here estimates longevity.
- an apnea / AHI validation on real nights (Discoveries 12): the page's own flow scoring run unchanged on a pilot of
  **PSG-Audio** (Korompili et al., *Scientific Data* 2021, doi 10.1038/s41597-021-00977-w; data doi 10.11922/sciencedb.00345,
  **CC BY 4.0**), overnight polysomnograms with synchronised tracheal and ambient microphones from Sismanoglio – Amalia Fleming
  General Hospital, Athens. Read from the public Hugging Face mirror `dust-systems/psg-audio`. Rebuild with `npm run build:apnea`
  (downloads ~0.7 GB per patient-hour, keeps only the two microphones at 8 kHz in `raw/psg_audio/`). **No PSG-Audio audio is
  published** — overnight bedroom audio can contain speech; only the statistics reach `web/`.

*Decision support for research and teaching — not a diagnosis.*

## Quick start
```bash
npm install
npx playwright install chromium   # for tests and the precompute step
npm run serve                     # open http://localhost:8000
npm test                          # end-to-end check; screenshots in tests/out/
npm run test:ahi                  # apnea/AHI scoring on synthetic audio (sensitivity + specificity)
npm run test:corsa                # CORSA crackle calibration (known IDW in, measured IDW out)
```
`web/` is prebuilt, so the site runs straight away without downloading any dataset.

## Getting the data (only needed to rebuild)
| Dataset | How to get it | Put it in |
|---|---|---|
| SPRSound | `npm run fetch:sprsound` (git clone, ~1.2 GB) | `raw/sprsound/` |
| KAUH | Download "Audio Files.zip" and "Data_annotation.xlsx" from https://data.mendeley.com/datasets/jwyy9np4gv/3 | `raw/kauh/` |
| CirCor | `pip install physionet && npm run fetch:circor` (ODC-By) | `raw/circor/` |
| CB / emphysema | Download both zips from https://data.mendeley.com/datasets/9hdrddjpny/1 (CC BY 4.0) and unzip | `raw/cbemph/` |

Then:
```bash
pip install -r requirements.txt   # ffmpeg on PATH, or the imageio-ffmpeg wheel it installs
npm run build                     # index → KAUH prep → MP3 → precompute → merge → discoveries
```
`npm run build:features` recomputes the per-recording features behind Discoveries. It is slow (every SPRSound recording); the results are already in `meta/features/`.

### Adding just CirCor
A full `npm run build` re-precomputes every dataset and needs all of `raw/`. To add the heart sounds
on their own, once `raw/circor/` is in place:
```bash
python3 pipeline/circor_prep.py          # cohort + validation index
node pipeline/circor_validate.js         # score the cardiac detectors on all 941 patients (~45 min)
python3 pipeline/encode_audio.py         # playback MP3s (needs ffmpeg)
node pipeline/precompute.js circor       # run the page's analysis on the original WAVs
python3 pipeline/merge.py --add circor   # append to web/data, leaving the other datasets alone
```

## Licences and citations
- **Code:** yours to licence.
- **SPRSound** — CC BY 4.0. Q. Zhang et al., "SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database," *IEEE TBioCAS* 16(5):867–881, 2022. https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound
- **KAUH** — CC BY 4.0. M. Fraiwan, L. Fraiwan, B. Khassawneh, A. Ibnian, "A dataset of lung sounds recorded from the chest wall using an electronic stethoscope," *Data in Brief* 35:106913, 2021. https://data.mendeley.com/datasets/jwyy9np4gv/3
- **Chronic bronchitis / emphysema** — CC BY 4.0. M. Fraiwan, N. S. Haider, B. Khassawneh, "Respiratory Sound Dataset of Chronic Bronchitis and Emphysema Patients," *Mendeley Data* v1, 2026. doi:10.17632/9hdrddjpny.1
- **CirCor DigiScope** — ODC-By 1.0 (Open Data Commons Attribution). J. Oliveira et al., "The CirCor DigiScope Dataset: From Murmur Detection to Murmur Classification," *IEEE J. Biomed. Health Inform.* 26(6):2524–2535, 2022. https://physionet.org/content/circor-heart-sound/1.0.3/
- **Libraries** (loaded from CDN, pinned): wavesurfer.js 7.12.12 (BSD-3-Clause), Meyda 5.6.3 (MIT), Chart.js 4.6.1 (MIT).

`web/audio` and `web/data` contain derived copies of the CC BY and ODC-By recordings and labels. Keep the attribution when you share the site.

See `CLAUDE.md` for the architecture, the code map and open tasks.
