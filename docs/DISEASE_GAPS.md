# Which disease does each patient in a batch have? What the open data can answer, and what is missing

The question: "I have a batch of 10 (or more) patients. I want to know which disease each one has. The model should be trained,
up to date, on all the open respiratory-sound databases, with adults and children kept apart."

This note records what was tried, the result, and what would have to exist for the answer to become "yes". Numbers are from
the run of 2026-11-23 (`meta/disease_model.json`, `pipeline/disease_train.py`). `pipeline/daily_train.py` reruns it every day
and writes `meta/daily/<date>.json`; the lite app's **Which disease?** page always shows the latest run.

## What was built

- **One model per population**: children (SPRSound, ICBHI children) and adults (KAUH, ICBHI adults, chronic bronchitis /
  emphysema, RespiratoryDatabase@TR). Every open database with a diagnosis that is on this machine and whose licence was
  checked; ICBHI is analysed only (research licence).
- **Diagnoses harmonised** into a few classes per population. Children: Healthy, Pneumonia, Bronchitis / bronchiolitis,
  Asthma, Other. Adults: Healthy, Asthma, COPD, Heart failure, Pneumonia, Other.
- **Inputs**: the app's own recording measurements (`recordingFeatures`: spectral shape, breathing rate, crackle rate, wheeze
  fraction, I:E), averaged per patient. Recording length is left out, because it is a protocol difference between databases.
- **Rules fixed before the first run**:
  - patient-grouped 5-fold cross-validation;
  - gradient boosting with fixed settings;
  - baselines on the same folds: database only, age only, commonest class;
  - leave-one-database-out testing;
  - pass = macro AUC ≥ 0.70, beats both baselines, and every leave-one-database-out AUC CI above 0.5.

## Result: failed in both populations

| | Patients | Macro AUC | Database only | Age only | Leave-one-database-out above chance |
|---|---|---|---|---|---|
| Children | 789 | 0.63 | 0.55 | 0.59 | 0 of 7 class tests |
| Adults | 458 | 0.82 | **0.87** | **0.85** | 2 of 7 |

- **Adults.** The pooled AUC of 0.82 looks good, but knowing only which database a recording came from scores higher (0.87). Each
  database holds a different mix of diagnoses:
  - healthy adults come from KAUH only;
  - two databases contain only COPD.

  On a database it was not trained on, the model is close to chance for most classes. It learned the collection, not the disease.
- **Children.** It beats both baselines (0.63 vs 0.55 and 0.59) but stays below 0.70. On ICBHI children, or on SPRSound when
  trained without it, every class is at chance.
- **The batch.** Each lung recording in the batch was scored by a fold that did not contain that patient. The top class
  matched the recorded diagnosis in 2 of 5. Both matches are COPD, which the model can get from the collection alone.

## Why — and what the literature says (published November 2025 – November 2026, checked on PubMed 2026-11-23)

- **The same failure is reported for deep networks.** Saky et al. reached a macro AUC of 0.99 inside one dataset. Tested on
  ICBHI and a COPD cohort, healthy recordings were poorly recognised, and source overlap had inflated the internal result.
  Life (Basel) 2026, [doi:10.3390/life16071108](https://doi.org/10.3390/life16071108), PMID 42514178.
- **High pooled accuracy, little external validation.** A meta-analysis of 56 deep-learning studies found high pooled
  accuracy for COPD yes/no (12 on breath sounds), but limited accuracy for GOLD stage. The authors warn of heterogeneity and
  little external validation. Yang et al., J Med Internet Res 2026, [doi:10.2196/83459](https://doi.org/10.2196/83459),
  PMID 41562629.
- **The device changes the sound.** 147 children were recorded on both a stethoscope and a smartphone; the spectra differed.
  AUROC was 0.91 on the primary set and 0.86 at an external site. Jeong et al., IEEE J Biomed Health Inform 2026,
  [doi:10.1109/JBHI.2026.3718492](https://doi.org/10.1109/JBHI.2026.3718492), PMID 42531109.
- **Sound plus context, with the device effect removed.** Over 10,000 multi-centre adults: cough sound combined with
  demographics and symptoms, trained to ignore the device. Yang et al., NPJ Digit Med 2026,
  [doi:10.1038/s41746-026-02445-4](https://doi.org/10.1038/s41746-026-02445-4), PMID 41748911.
- **Signs as evidence, not labels.** Wheeze, crackle and breath phases are used as evidence for the class, on KAUH,
  SPRSound and ICBHI. The authors state that independent-cohort and device-stratified tests are still needed. Yuan et al.,
  Front Med 2026, [doi:10.3389/fmed.2026.1987793](https://doi.org/10.3389/fmed.2026.1987793), PMID 42985215.
- **Heart sounds against echocardiography.** 8 studies, 4,374 patients: sensitivity 72 %, specificity 82 %. Prospective
  multi-centre validation is still needed. Saeed et al., Cardiol Rev 2026,
  [doi:10.1097/CRD.0000000000001559](https://doi.org/10.1097/CRD.0000000000001559), PMID 43059005.
- **AI as an aid beside the clinician.** The output works best as a reflective aid; the clinical meaning of sound features
  needs more research. Melbye et al., Scand J Prim Health Care 2026,
  [doi:10.1080/02813432.2026.2739502](https://doi.org/10.1080/02813432.2026.2739502), PMID 42799994.

Dataset papers (SPRSound, ICBHI) stay cited as data sources in `README.md`, as their licences require.

## What is missing to answer "which disease"

1. **The same diagnoses recorded in several hospitals on the same protocol.** Today most classes come mostly from one
   database (`classByDatabase` in the result file), so disease and collection cannot be separated.
2. **Healthy controls in every database.** RespiratoryDatabase@TR and the chronic-bronchitis / emphysema set have none.
3. **A confirmed reference standard per patient**:
   - spirometry for COPD and asthma;
   - imaging for pneumonia;
   - echocardiography or BNP for heart failure.

   Age and sex are also needed for every recording. The chronic-bronchitis / emphysema set has no patient number and no age.
4. **More than one stethoscope per disease**, so device and disease can be told apart and a per-device calibration measured.
   The heart-sound results already show that a flag level does not transfer between stethoscopes (CinC 2016 test).
5. **Clinical context with the sound**: symptoms, vital signs, oxygen saturation, temperature. These are the inputs
   clinicians use to tell pneumonia from bronchiolitis from asthma, and what the 2026 multi-centre model adds (Yang, NPJ Digit Med 2026).
6. **A prospective test set**: one site, one stethoscope, collected after the model is frozen, with the pass rule written down
   first. Most of the 2026 papers above name independent or prospective testing as the step still missing. This is what a hospital such as Sheba could contribute, and it is the only test that answers the question for its
   own patients.

## What can be answered today

- Which **sounds** are present: wheeze, with moderate accuracy, and murmur in children. Each finding carries its own
  validation line in the lite app.
- Resting heart rate.

These are findings a clinician combines with the rest of the examination. The disease is the clinician's conclusion.
