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

## Why — and what the literature says

- **Most published models are single-centre.** Park et al. reviewed 41 paediatric studies. Wheeze and abnormal-sound
  detection is accurate (pooled sensitivity about 0.90). But most studies use small, single-centre datasets with little
  external validation. Park JS et al., J Med Internet Res 2025;27:e66491, [doi:10.2196/66491](https://doi.org/10.2196/66491),
  PMID 40249944.
- **Disease from the whole record is hard even in-house.** SPRSound's own record-level classification scores were 38–62 %.
  Zhang Q et al., IEEE TBioCAS 2022, [doi:10.1109/TBCAS.2022.3204910](https://doi.org/10.1109/TBCAS.2022.3204910),
  PMID 36070274.
- **Diagnosis is tied to age and device in ICBHI.** It was recorded on four devices, and its best challenge system scored
  52.5 %. Rocha BM et al., Physiol Meas 2019, [doi:10.1088/1361-6579/ab03ea](https://doi.org/10.1088/1361-6579/ab03ea),
  PMID 30708353.
- **Crackles and wheeze are shared by several diseases.** They are signs, not diagnoses. Bohadana A, Izbicki G, Kraman SS,
  N Engl J Med 2014, [doi:10.1056/NEJMra1302901](https://doi.org/10.1056/NEJMra1302901), PMID 24552321.

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
   clinicians use to tell pneumonia from bronchiolitis from asthma.
6. **A prospective test set**: one site, one stethoscope, collected after the model is frozen, with the pass rule written down
   first. This is what a hospital such as Sheba could contribute, and it is the only test that answers the question for its
   own patients.

## What can be answered today

- Which **sounds** are present: wheeze, with moderate accuracy, and murmur in children. Each finding carries its own
  validation line in the lite app.
- Resting heart rate.

These are findings a clinician combines with the rest of the examination. The disease is the clinician's conclusion.
