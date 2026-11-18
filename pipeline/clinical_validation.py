"""Physician-readable validation reports → web/data/validation.json (the Validation page).

Every detector this platform ships gets one report laid out the way a diagnostic-accuracy study is read
(STARD 2015 items in brackets): the clinical question and intended use, who was tested [participants], what it
was compared with [reference standard], the exact rule [index test, threshold], the results with 95 % confidence
intervals [estimates of accuracy, imprecision], subgroups, what the numbers mean at the bedside, and the limits.

Nothing is re-measured here. The numbers come from the validation runs already in the repository:
  index.json  validation / validationKAUH / validationCircor   (precompute.js, circor_validate.js)
  meta/features/circor.jsonl                                    (per-valve detector output, CirCor)
  meta/psg_audio_validation.json                                (PSG-Audio apnea pilot)
  discover.json['age']                                          (biological-age analysis)
Statistics come from Seismometer (Epic's open-source healthcare-AI validation package, pip `seismometer`):
Agresti–Coull intervals for proportions and Hanley–McNeil intervals for the ROC AUC. Likelihood ratios use the
standard log-method interval (Simel et al. 1991).

How a physician should read "evidence level":
  external     thresholds were fixed before this dataset was seen; the estimate applies to similar patients
  development  thresholds were tuned on this same dataset — optimistic, expect worse on new patients
  pilot        external but small — direction is informative, the precise number is not

  python3 pipeline/clinical_validation.py      (then merge.py --bump; ml_validate.py adds the ML section)
"""
import collections, datetime, json, math
import numpy as np
from seismometer.data.confidence import agresti_coull_interval, hanley_mcneill_confidence
from seismometer.data.confidence.parameters import PerformanceMetricConfidenceParam, ROCConfidenceParam
from paths import META, FEATURES, WEB_DATA

CI = PerformanceMetricConfidenceParam(0.95)
Z = 1.959964


def prop(k, n, label, explain=''):
    """A proportion with its Agresti–Coull 95 % interval (Seismometer)."""
    if not n: return {'label': label, 'k': k, 'n': n, 'value': None, 'lo': None, 'hi': None, 'explain': explain}
    r = agresti_coull_interval(CI, k / n, n)
    return {'label': label, 'k': int(k), 'n': int(n), 'value': round(100 * k / n, 1),
            'lo': round(100 * max(0.0, float(r.lower)), 1), 'hi': round(100 * min(1.0, float(r.upper)), 1), 'explain': explain}


def lr(tp, fn, fp, tn):
    """Positive and negative likelihood ratios with log-method 95 % CIs (0.5 continuity correction if a cell is 0)."""
    if 0 in (tp, fn, fp, tn): tp, fn, fp, tn = tp + .5, fn + .5, fp + .5, tn + .5
    se, sp = tp / (tp + fn), tn / (tn + fp)
    out = {}
    for name, v, s in (('lrPos', se / (1 - sp), math.sqrt(1 / tp - 1 / (tp + fn) + 1 / fp - 1 / (fp + tn))),
                       ('lrNeg', (1 - se) / sp, math.sqrt(1 / fn - 1 / (tp + fn) + 1 / tn - 1 / (fp + tn)))):
        out[name] = {'value': round(v, 2), 'lo': round(v * math.exp(-Z * s), 2), 'hi': round(v * math.exp(Z * s), 2)}
    return out


def lr_words(L):
    """Standard bedside reading of likelihood ratios (Jaeschke, Guyatt & Sackett, JAMA 1994)."""
    p, n = L['lrPos']['value'], L['lrNeg']['value']
    rule_in = 'raises the odds a lot — often decisive' if p >= 10 else 'raises the odds moderately' if p >= 5 else 'raises the odds a little' if p >= 2 else 'barely changes the odds'
    rule_out = 'lowers the odds a lot — often decisive' if n <= 0.1 else 'lowers the odds moderately' if n <= 0.2 else 'lowers the odds a little' if n <= 0.5 else 'barely changes the odds'
    return rule_in, rule_out


def per100(sens, spec):
    return (f"Of 100 patients who have the finding it flags about {round(sens)}; "
            f"of 100 who do not, it wrongly flags about {round(100 - spec)}.")


def two_by_two(tp, fn, fp, tn, what, unit):
    s, c = prop(tp, tp + fn, 'Sensitivity', f'{what} correctly flagged'), prop(tn, tn + fp, 'Specificity', f'{unit} without it correctly left unflagged')
    ppv, npv = prop(tp, tp + fp, 'PPV', 'flags that were correct, at this sample’s prevalence'), prop(tn, tn + fn, 'NPV', 'negatives that were correct, at this sample’s prevalence')
    prev = prop(tp + fn, tp + fn + fp + tn, 'Prevalence in the sample', 'changes PPV and NPV; sensitivity and specificity do not depend on it')
    return [s, c, ppv, npv, prev], lr(tp, fn, fp, tn)


# --------------------------------------------------------------------------------------------- murmur (CirCor)
def murmur(I):
    V = I['validationCircor']['full']
    tp, P = V['murmurPatient']['sens']; tn, N = V['murmurPatient']['spec']; fn, fp = P - tp, N - tn
    metrics, L = two_by_two(tp, fn, fp, tn, 'children with an annotated murmur', 'children')
    ri, ro = lr_words(L)
    rows = [json.loads(l) for l in open(FEATURES / 'circor.jsonl')]
    # patient-level subgroups recomputed from the per-valve detector output (must reproduce the totals above)
    pts = collections.defaultdict(list)
    for r in rows: pts[r['pid']].append(r)
    pat = [{'murmur': v[0]['murmur'], 'det': any(x['det']['murmur'] for x in v), 'sex': 'female' if v[0]['g'] == '1' else 'male',
            'campaign': v[0]['campaign'], 'band': v[0]['ageBand']} for v in pts.values()]
    sub = []
    for key, name in (('band', 'Age band'), ('sex', 'Sex'), ('campaign', 'Screening campaign')):
        vals = sorted({p[key] for p in pat if p[key]})
        sub.append({'group': name, 'rows': [
            {**prop(sum(1 for p in pat if p[key] == v and p['murmur'] == 'Present' and p['det']), sum(1 for p in pat if p[key] == v and p['murmur'] == 'Present'), v), 'kind': 'sensitivity',
             'spec': prop(sum(1 for p in pat if p[key] == v and p['murmur'] == 'Absent' and not p['det']), sum(1 for p in pat if p[key] == v and p['murmur'] == 'Absent'), v)}
            for v in vals]})
    sub.insert(0, {'group': 'Murmur loudness (Levine grade)', 'rows': [{**prop(k, n, g), 'kind': 'sensitivity'} for g, (k, n) in V['byGrade'].items()]})
    # threshold-free ability of the underlying feature (valve level), Hanley–McNeil CI
    lab = [r for r in rows if r['siteTruth'] in ('Murmur', 'Normal') and r.get('sysDia') is not None]
    y = np.array([r['siteTruth'] == 'Murmur' for r in lab], float); x = np.array([r['sysDia'] for r in lab], float)
    a = hanley_mcneill_confidence(ROCConfidenceParam(0.95), y, x / x.max())
    s0 = metrics[0]; c0 = metrics[1]
    return {
        'id': 'murmur', 'title': 'Systolic murmur detector (heart sounds)', 'discover': 'd-heart', 'evidence': 'external',
        'question': 'Does this child have a heart murmur that an expert would annotate?',
        'intendedUse': 'Research and teaching: point a learner or reviewer to recordings likely to contain a systolic murmur. Not a screening or diagnostic test.',
        'population': {'n': P + N, 'text': f'{P + N} children and adolescents (with a recorded murmur status) from two mass screening campaigns in Paraíba, Brazil (CirCor DigiScope 1.0.3), up to four valve sites each; {V["murmurPatient"]["unknownFlagged"][1]} with murmur status “unknown” excluded. Consecutive screening, not a referred population.'},
        'reference': 'Murmur presence, location, timing and Levine grade as annotated by the CirCor dataset’s expert annotators, per valve. The detector never saw these labels.',
        'indexTest': 'Automatic: mean systolic-to-diastolic energy ratio over resolved beats ≥ 2.5 × that of a murmur-free child of the same age band at the same valve, with a heart rate plausible for age. Flagged at any valve = patient positive.',
        'independence': 'External: the threshold was set before CirCor was added and has not been changed on it.',
        'metrics': metrics, **L,
        'extra': [{'label': 'ROC AUC of the underlying ratio (per valve, threshold-free)', 'value': round(float(a.value), 3), 'lo': round(float(a.lower), 3), 'hi': round(float(min(1, a.upper)), 3), 'n': len(lab),
                   'explain': '0.5 = chance, 1 = perfect. Shows the feature carries real signal even where the fixed cut-off misses.'}],
        'subgroups': sub,
        'bedside': [per100(s0['value'], c0['value']),
                    f'A positive flag {ri} (LR+ {L["lrPos"]["value"]}): when it flags a murmur, it is usually right.',
                    f'A negative result {ro} (LR− {L["lrNeg"]["value"]}): <b>it cannot rule a murmur out</b>.',
                    f'It finds loud murmurs and misses soft ones: grade I/VI {round(100 * V["byGrade"]["I/VI"][0] / V["byGrade"]["I/VI"][1])} %, grade III/VI {round(100 * V["byGrade"]["III/VI"][0] / V["byGrade"]["III/VI"][1])} %. Soft murmurs are the ones screening most needs.'],
        'limits': ['Children only; adults were not tested.', 'Infants are poorly served (fast heart rate leaves too little diastole to separate S1 and S2).',
                   'Innocent and pathological murmurs are not distinguished — the reference is “murmur present”, not structural disease.',
                   'Heart-sound segmentation finds about 71 % of S1/S2; beats it cannot resolve cannot be scored.'],
        'verdict': 'Rule-in aid for research review; not usable to exclude a murmur.'}


# ------------------------------------------------------------------------------------ respiratory, SPRSound
def resp_spr(I):
    cm = I['validation']['cm']; cls = ['Normal', 'CAS', 'DAS', 'CAS & DAS']
    tn = cm['Normal']['Normal']; fp = sum(cm['Normal'][c] for c in cls[1:])
    tp = sum(cm[a][b] for a in cls[1:] for b in cls[1:]); fn = sum(cm[a]['Normal'] for a in cls[1:])
    metrics, L = two_by_two(tp, fn, fp, tn, 'expert-labelled adventitious breath events', 'events')
    ri, ro = lr_words(L)
    cas = prop(sum(cm[a][b] for a in ('CAS', 'CAS & DAS') for b in ('CAS', 'CAS & DAS')), sum(sum(cm[a].values()) for a in ('CAS', 'CAS & DAS')), 'Continuous sounds (wheeze, rhonchi, stridor) recognised as continuous')
    das = prop(sum(cm[a][b] for a in ('DAS', 'CAS & DAS') for b in ('DAS', 'CAS & DAS')), sum(sum(cm[a].values()) for a in ('DAS', 'CAS & DAS')), 'Crackles recognised as crackles')
    return {
        'id': 'resp_spr', 'title': 'Adventitious breath-sound detector — children (SPRSound)', 'discover': 'd-device', 'evidence': 'development',
        'question': 'Does this breath (event) contain an adventitious sound — wheeze, rhonchus, stridor or crackle?',
        'intendedUse': 'Research and teaching: highlight where on a recording adventitious sounds probably are. Not a diagnostic test.',
        'population': {'n': I['validation']['n'], 'text': f'{I["validation"]["n"]} expert-annotated breath events in {I["validation"]["nRec"]} recordings of the {len({r["key"] for r in I["validation"]["rows"]})} browsable SPRSound children (Shanghai Children’s Medical Center, Yunting stethoscope).'},
        'reference': 'Per-event labels (normal, wheeze, rhonchi, stridor, fine/coarse crackle) by SPRSound physicians.',
        'indexTest': 'Automatic tonality (wheeze/rhonchus) and transient (crackle) detectors against the patient’s own normal site, shipped thresholds.',
        'independence': '<b>Development data:</b> these thresholds were tuned on SPRSound. The figures are optimistic; the KAUH result below estimates performance on new patients.',
        'metrics': metrics + [cas, das], **L, 'subgroups': [],
        'bedside': [per100(metrics[0]['value'], metrics[1]['value']),
                    f'A flag {ri} (LR+ {L["lrPos"]["value"]}); a clean result {ro} (LR− {L["lrNeg"]["value"]}).',
                    'Continuous sounds are recognised more reliably than crackles.'],
        'limits': ['Tuned and tested on the same database (see independence).', 'Short clips (9–15 s) from one hospital and one stethoscope.', 'Event-level figures; events from one child are not independent, so the intervals are too narrow.'],
        'verdict': 'Teaching aid; its accuracy here is an upper bound.'}


# ------------------------------------------------------------------------------------ respiratory, KAUH
def resp_kauh(I):
    B = I['validationKAUH']['byMode']; out = []
    D = B['D']; tp, P = D['sens']; tn, N = D['spec']
    metrics, L = two_by_two(tp, P - tp, N - tn, tn, 'abnormal recordings', 'healthy adults')
    ri, ro = lr_words(L)
    names = {'D': 'diaphragm', 'B': 'bell', 'E': 'extended'}
    sub = [{'group': 'Stethoscope filter mode (same recordings)', 'rows': [{**prop(*B[m]['sens'], names[m]), 'kind': 'sensitivity', 'spec': prop(*B[m]['spec'], names[m])} for m in 'DBE']},
           {'group': 'Sound type (diaphragm)', 'rows': [{**prop(*D['cas'], 'continuous sounds (wheeze)'), 'kind': 'sensitivity'}, {**prop(*D['das'], 'crackles'), 'kind': 'sensitivity'}]}]
    return {
        'id': 'resp_kauh', 'title': 'Adventitious breath-sound detector — adults (KAUH)', 'discover': 'd-device', 'evidence': 'external',
        'question': 'Does this adult’s recording contain an adventitious sound?',
        'intendedUse': 'Research and teaching only.',
        'population': {'n': P + N, 'text': f'{P + N} adolescents and adults 12–90 y (King Abdullah University Hospital, Jordan; 3M Littmann 3200): {N} healthy, {P} with asthma, heart failure, COPD, pneumonia, fibrosis and others. One chest site each.'},
        'reference': 'Whole-recording sound label from the KAUH dataset’s clinical annotation (normal, wheeze, crackles, …); no event timings.',
        'indexTest': 'Same detectors and thresholds as SPRSound, unchanged; diaphragm mode is the primary result.',
        'independence': 'External: a different country, age range and stethoscope, thresholds untouched.',
        'metrics': metrics, **L, 'subgroups': sub,
        'bedside': [per100(metrics[0]['value'], metrics[1]['value']),
                    f'A flag {ri} (LR+ {L["lrPos"]["value"]}); a clean result {ro} (LR− {L["lrNeg"]["value"]}): on a different stethoscope it is weak in both directions.',
                    f'Crackles — the sign of heart failure and fibrosis — are found in only {round(100 * D["das"][0] / D["das"][1])} % of recordings that have them.'],
        'limits': ['Small: 35 healthy adults, so specificity is imprecise.', 'Filter mode changes the result substantially; per-device recalibration is open work.'],
        'verdict': 'Does not generalise to the Littmann 3200 without recalibration.'}


# ------------------------------------------------------------------------------------ apnea (PSG-Audio)
def apnea():
    f = META / 'psg_audio_validation.json'
    if not f.exists(): return None
    A = json.load(open(f)); T = A['channels']['trach']; e = T['events']
    sens = prop(e['found'], e['expert'], 'Event sensitivity', 'scored apneas/hypopneas overlapped by an acoustic event')
    ppv = prop(e['matched'], e['acoustic'], 'Event PPV', 'acoustic events that overlapped a scored event')
    by = [{**prop(v['found'], v['n'], {'OA': 'obstructive apnea', 'CA': 'central apnea', 'MA': 'mixed apnea', 'H': 'hypopnea'}[k]), 'kind': 'sensitivity'} for k, v in e['byType'].items() if v['n']]
    P = T['perPatient']; pos = [q for q in P if q['ahiPsg'] >= 15]
    mod = prop(sum(1 for q in pos if q['ahiAcoustic'] >= 15), len(pos), 'Patients with AHI ≥ 15 (moderate–severe) recognised as ≥ 15')
    a = T.get('ahi', {}); hd = T.get('headroom')
    return {
        'id': 'apnea', 'title': 'Apnea / hypopnea and AHI from breath sound (overnight)', 'discover': 'd-apnea', 'evidence': 'pilot',
        'question': 'How many apneas and hypopneas per hour does this sleeper have, and in which severity class?',
        'intendedUse': 'Research only: estimate an acoustic AHI from a long breath-sound recording. Not a substitute for polysomnography or home sleep apnea testing.',
        'population': {'n': T['patients'], 'text': f'{T["patients"]} adults referred to a sleep lab for suspected sleep apnea (Athens; PSG-Audio), {T["hours"]} h of diagnostic recording, {e["expert"]} scored events. ~89 % of this dataset has severe apnea.'},
        'reference': 'Full polysomnography scored by a sleep technician and a certified physician (AASM rules), using airflow, effort, SpO₂ and EEG. Sound was not used for scoring.',
        'indexTest': 'The page’s acoustic flow proxy on the tracheal contact microphone; AASM adult rules (≥ 30 % drop for hypopnea, ≥ 90 % for apnea, ≥ 10 s); shipped thresholds.',
        'independence': 'External, untuned: the scoring was written before these nights were seen.',
        'metrics': [sens, ppv, mod], 'subgroups': [{'group': 'Expert event type', 'rows': by}] + ([{'group': 'Signal headroom (breathing above noise floor)', 'rows': [
            {'label': 'below median', 'value': hd['sensLow'], 'k': None, 'n': None, 'lo': None, 'hi': None, 'kind': 'sensitivity'},
            {'label': 'above median', 'value': hd['sensHigh'], 'k': None, 'n': None, 'lo': None, 'hi': None, 'kind': 'sensitivity'}]}] if hd else []),
        'extra': [{'label': 'AHI bias (acoustic − PSG), events/h', 'value': a.get('bias'), 'lo': (a.get('loa') or [None])[0], 'hi': (a.get('loa') or [None, None])[1], 'n': T['patients'],
                   'explain': 'interval shown = 95 % limits of agreement (Bland–Altman): where one patient’s error will fall'},
                  {'label': 'AHI rank correlation with PSG (Spearman ρ)', 'value': a.get('rho'), 'lo': (a.get('rhoCI') or [None])[0], 'hi': (a.get('rhoCI') or [None, None])[1], 'n': T['patients'],
                   'explain': f'bootstrap 95 % CI; p = {a.get("p", float("nan")):.2f}'}],
        'bedside': [f'It finds about {round(e["sens"])} of every 100 scored events; about {round(e["ppv"])} of every 100 events it reports are real.',
                    f'Its AHI is too low by {abs(a.get("bias", 0))} events/h on average, and for one patient the error can be anywhere from {a.get("loa", ["?", "?"])[0]} to {a.get("loa", ["?", "?"])[1]} events/h.',
                    f'Severity class matched polysomnography in {T.get("severity", {}).get("agree")} of {T["patients"]} patients; every miss put the patient in a milder class.',
                    '<b>A low acoustic AHI does not exclude sleep apnea.</b>'],
        'limits': [f'Pilot of {T["patients"]} patients — the intervals are wide.', 'Almost no healthy sleepers: specificity cannot be estimated.',
                   'Adults only; the paediatric rule is untested.', 'No SpO₂ or effort channel: obstructive vs central and desaturation-based hypopneas are out of reach.',
                   'Events are clustered within patients; event-level intervals are too narrow.'],
        'verdict': 'Not ready for clinical use; under-counts systematically.'}


# ------------------------------------------------------------------------------------ age (no clinical claim)
def age(D):
    A = D.get('age')
    if not A: return None
    a = A['adults']; h = A['heart']; o = A['outcome']
    return {
        'id': 'age', 'title': 'Biological-age signal (longevity precondition)', 'discover': 'd-age', 'evidence': 'research',
        'question': 'Does the sound carry information about a person’s age, so that an “acoustic age gap” could one day be tested as a longevity marker?',
        'intendedUse': 'Research question only. No longevity, mortality or risk output exists or is implied.',
        'population': {'n': a['nTest'], 'text': f'Held-out patients: {a["nTest"]} KAUH adults, {sum(c["nTest"] for c in A["children"])} SPRSound children, {h["nTest"]} CirCor children.'},
        'reference': 'Recorded chronological age (KAUH, SPRSound: years; CirCor: age band). No survival or follow-up data exist in any dataset here.',
        'indexTest': 'Ridge regression on detector features, fit on 60 % of patients, scored on the held-out 40 %.',
        'independence': 'Held-out split; null model = guess the training mean.',
        'metrics': [], 'subgroups': [],
        'extra': [{'label': 'Adult age error, years (model vs guessing the mean)', 'value': a['mae'], 'lo': None, 'hi': None, 'n': a['nTest'], 'explain': f'guessing the mean: {a["maeNull"]} y — no improvement (permutation p = {a["pPerm"]:.2f})'},
                  {'label': 'Heart rate vs age band (Spearman ρ)', 'value': h['rho'], 'lo': None, 'hi': None, 'n': h['nTest'], 'explain': f'p = {h["pRho"]:.1g} — heart rate falls with age, as physiology predicts'},
                  {'label': 'Heart-rate-for-age deviation, normal vs abnormal outcome (effect r)', 'value': o['z']['r'], 'lo': None, 'hi': None, 'n': o['z']['nA'] + o['z']['nB'], 'explain': f'p = {o["z"]["p"]:.2f} — no association'}],
        'bedside': ['Breath sounds as measured here do not reveal an adult’s age; there is no “acoustic age gap” to interpret.',
                    'Heart rate from heart sounds behaves as expected with age in children — a sanity check, not a new marker.',
                    '<b>Nothing on this platform estimates longevity or mortality risk.</b>'],
        'limits': ['No outcome data: longevity cannot be validated with any open auscultation dataset available today.'],
        'verdict': 'No clinical meaning; research only.'}


PLURAL = {'Neonate': 'neonates', 'Infant': 'infants', 'Child': 'children', 'Adolescent': 'adolescents'}


def hrv(D):
    H = D.get('hrv')
    if not H or not H.get('agreement'): return None
    G = H.get('segmenter')
    if not G or not G.get('agreement'): return hrv_detector(H)
    g = G['agreement']; P = G.get('paired', {}); d, s = G['det'], G['seg']; rep = H.get('repeatability', {})
    return {
        'id': 'hrv', 'title': 'Heart-rate variability (RMSSD) from heart sounds', 'discover': 'd-age', 'evidence': 'external',
        'question': 'Can beat-to-beat variability be measured from a heart-sound recording as accurately as from hand-marked beats?',
        'intendedUse': 'Research: a candidate longevity-related marker. Shown on pages only if it agrees with expert-marked beats on patients the beat finder never saw (pre-set rule: ICC ≥ 0.75 and |bias| ≤ 10 ms).',
        'population': {'n': g['n'], 'text': f"{G['n']['recordings']:,} expert-annotated CirCor recordings from {G['n']['patients']} held-out children and adolescents (Brazil) — the same held-out patients the murmur model is scored on; {g['n']:,} had enough clean beats for RMSSD from both the beat finder and the experts."},
        'reference': 'RMSSD computed from the S1 onsets the CirCor annotators marked in the .tsv segmentation, on the same annotated stretch and with the same artefact rule (intervals within 20 % of the median).',
        'indexTest': (f"RMSSD from the S1 onsets of a learned beat finder: band energies every 10 ms, a gradient-boosted frame classifier (S1 / systole / S2 / diastole) and a cyclic duration model, "
                      f"trained on {G['trained']['recordings']:,} recordings of the other CirCor patients. At least 8 successive differences required (Munoz 2015)."),
        'independence': 'External: trained only on other patients, settings fixed before scoring, the pass rule written before the first HRV result. The previous beat finder had already failed the rule on all recordings.',
        'metrics': [], 'subgroups': [],
        'extra': [{'label': 'Agreement with expert-marked beats (ICC 2,1)', 'value': g['icc'], 'lo': g['iccCI'][0], 'hi': g['iccCI'][1], 'n': g['n'], 'explain': f"required ≥ {H['rule']['iccMin']}; 0.75 is “good” agreement"},
                  {'label': 'Mean bias, beat finder − experts (ms)', 'value': g['bias'], 'lo': g['loa'][0], 'hi': g['loa'][1], 'n': g['n'], 'explain': f"95 % limits of agreement; {g['within10ms']} % of recordings within ±10 ms"},
                  {'label': 'Previous beat finder on the same recordings (ICC)', 'value': P.get('det', {}).get('icc'), 'lo': (P.get('det', {}).get('iccCI') or [None])[0], 'hi': (P.get('det', {}).get('iccCI') or [None, None])[1], 'n': P.get('det', {}).get('n'),
                   'explain': f"learned finder on these {P.get('seg', {}).get('n')} recordings: ICC {P.get('seg', {}).get('icc')}"},
                  {'label': 'Expert S1 sounds found within ±50 ms (%)', 'value': s['s1Sens'], 'lo': None, 'hi': None, 'n': s['tp'] + s['fn'],
                   'explain': f"{s['s1Ppv']} % of detected S1 are real; median timing error {s['medAbsErrMs']} ms. Previous finder: {d['s1Sens']} % found, {d['s1Ppv']} % real"},
                  {'label': 'Heart rate agreement (ICC)', 'value': s.get('hr', {}).get('icc'), 'lo': None, 'hi': None, 'n': s.get('hr', {}).get('n'), 'explain': f"bias {s.get('hr', {}).get('bias')} bpm; {s.get('hr', {}).get('within5bpm')} % within ±5 bpm"}]
                 + [{'label': f"Age band {b['band']}: agreement (ICC)", 'value': b['icc'], 'lo': None, 'hi': None, 'n': b['n'],
                     'explain': f"bias {b['bias']} ms, limits {b['loa'][0]} to {b['loa'][1]} — {'shown' if b['pass'] else 'below the rule, not shown'}"} for b in G.get('byBand', [])],
        'bedside': [f"On children it never saw, the learned beat finder’s RMSSD agrees with RMSSD from hand-marked beats at ICC {g['icc']}; one recording can still differ from it by {g['loa'][0]:.0f} to +{g['loa'][1]:.0f} ms.",
                    f"It finds {s['s1Sens']:.0f} of 100 expert-marked first heart sounds within 50 ms (previous finder: {d['s1Sens']:.0f}).",
                    f"Shown for {' and '.join(PLURAL.get(b, b) for b in G['showBands'])}; " + ', '.join(f"{PLURAL.get(b['band'], b['band'])} (ICC {b['icc']}, n {b['n']})" for b in G.get('byBand', []) if not b['pass']) + ' did not pass and get no number.' if G['pass'] else '<b>HRV is not shown.</b>'],
        'limits': ['Children and adolescents only; the mortality evidence for HRV is from older adults, and no dataset here links HRV to outcome.',
                   'Clips of 5–65 s: RMSSD only; SDNN needs 30 s or more of clean beats.',
                   f"The experts’ own RMSSD repeats across two valves of the same child at ICC {rep.get('expert', {}).get('icc')}: short clips limit even perfect beat marking.",
                   'Validated on the original WAV files; uploads that are compressed were not tested for HRV.',
                   'In the Framingham cohort the predictive HRV measures were SDNN and spectral power, not RMSSD (Tsuji 1994).'],
        'verdict': ('Passed its pre-set agreement rule on held-out patients; measurement only, no clinical claim.' if G['pass'] else 'Failed its pre-set agreement rule — not shown.')}


def hrv_detector(H):
    g = H['agreement']; rep = H.get('repeatability', {}); hr = H.get('hrAgreement', {})
    return {
        'id': 'hrv', 'title': 'Heart-rate variability (RMSSD) from heart sounds', 'discover': 'd-age', 'evidence': 'external',
        'question': 'Can beat-to-beat variability be measured from a heart-sound recording as accurately as from hand-marked beats?',
        'intendedUse': 'Research: a candidate longevity-related marker. Shown on pages only if it agrees with expert-marked beats (pre-set rule: ICC ≥ 0.75 and |bias| ≤ 10 ms).',
        'population': {'n': H['n']['both'], 'text': f"{H['n']['recordings']:,} CirCor heart-sound recordings (children, Brazil); {H['n']['both']:,} had enough clean beats for RMSSD from both the detector and the experts on the same annotated stretch."},
        'reference': 'RMSSD computed from the S1 onsets the CirCor annotators marked in the .tsv segmentation, on the same stretch of recording and with the same artefact rule (intervals within 20 % of the median).',
        'indexTest': 'RMSSD from the S1 times found by the page’s heart-sound detector; at least 8 successive differences required (a single 10-s recording suffices for RMSSD — Munoz 2015).',
        'independence': 'External: the beat finder and the pass rule were fixed before the comparison was run.',
        'metrics': [], 'subgroups': [],
        'extra': [{'label': 'Agreement with expert-marked beats (ICC 2,1)', 'value': g['icc'], 'lo': g['iccCI'][0], 'hi': g['iccCI'][1], 'n': H['n']['both'], 'explain': f"required ≥ {H['rule']['iccMin']}; 0.75 is “good” agreement"},
                  {'label': 'Mean bias, detector − experts (ms)', 'value': g['bias'], 'lo': g['loa'][0], 'hi': g['loa'][1], 'n': H['n']['both'], 'explain': '95 % limits of agreement: where one recording’s error falls'},
                  {'label': 'Heart rate agreement (ICC)', 'value': hr.get('icc'), 'lo': None, 'hi': None, 'n': hr.get('n'), 'explain': f"bias {hr.get('bias')} bpm; {hr.get('within5bpm')} % within ±5 bpm — the beats are found, their exact timing is the problem"},
                  {'label': 'Same child, two valves (ICC): detector / experts', 'value': rep.get('detector', {}).get('icc'), 'lo': None, 'hi': None, 'n': rep.get('detector', {}).get('children'),
                   'explain': f"experts’ own beats repeat at ICC {rep.get('expert', {}).get('icc')} — short clips limit even perfect beat marking"}],
        'bedside': [f"The detector’s RMSSD agrees only moderately with RMSSD from hand-marked beats (ICC {g['icc']}); a single recording can be off by {abs(g['loa'][0]):.0f}–{abs(g['loa'][1]):.0f} ms.",
                    f"Heart rate from the same beats is reliable (ICC {hr.get('icc')}).",
                    '<b>HRV is therefore not shown as a number on patient pages or for uploads.</b>' if not H['pass'] else 'HRV meets the pre-set agreement rule and is shown on heart-sound pages.'],
        'limits': ['Children only; the mortality evidence for HRV is from older adults.', 'Clips of 5–65 s: RMSSD only; SDNN needs 30 s or more of clean beats.',
                   'In the Framingham cohort the predictive HRV measures were SDNN and spectral power, not RMSSD (Tsuji 1994).'],
        'verdict': 'Failed its pre-set agreement rule — not shown.' if not H['pass'] else 'Passed its pre-set agreement rule.'}


# ------------------------------------------------------------------------------------ ICBHI 2017 (research licence)
ICBHI_CITE = 'Rocha BM et al., Physiol Meas 2019;40:035001 (PMID 30708353)'


def icbhi_rules():
    """The SHIPPED crackle / wheeze rules scored on held-out ICBHI cycles: external (thresholds fixed before ICBHI)."""
    f = WEB_DATA / 'icbhi_models.json'
    if not f.exists(): return None
    M = json.load(open(f)); S = next(m for m in M['models'] if m['id'] == 'icbhi_sound'); R = S['rules']
    c, w = R['crackles'], R['wheezes']; tp, P = c['sens']['k'], c['sens']['n']; tn, N = c['spec']['k'], c['spec']['n']
    metrics, L = two_by_two(tp, P - tp, N - tn, tn, 'breath cycles with crackles', 'breath cycles')
    ri, ro = lr_words(L)
    sub = [{'group': 'Wheeze rule (same cycles)', 'rows': [{**prop(w['sens']['k'], w['sens']['n'], 'wheezes'), 'kind': 'sensitivity', 'spec': prop(w['spec']['k'], w['spec']['n'], 'wheezes')}]},
           {'group': 'ICBHI-trained sound model on the same cycles (threshold 0.5)', 'rows': [
               {**prop(S['rules']['model']['crackles']['sens']['k'], S['rules']['model']['crackles']['sens']['n'], 'crackles'), 'kind': 'sensitivity', 'spec': prop(S['rules']['model']['crackles']['spec']['k'], S['rules']['model']['crackles']['spec']['n'], 'crackles')},
               {**prop(S['rules']['model']['wheezes']['sens']['k'], S['rules']['model']['wheezes']['sens']['n'], 'wheezes'), 'kind': 'sensitivity', 'spec': prop(S['rules']['model']['wheezes']['spec']['k'], S['rules']['model']['wheezes']['spec']['n'], 'wheezes')}]}]
    return {
        'id': 'resp_icbhi', 'title': 'Crackle / wheeze detectors per breath cycle — ICBHI 2017', 'discover': None, 'evidence': 'external',
        'question': 'Does this breath cycle contain crackles (or wheezes)?',
        'intendedUse': 'Research and teaching only.',
        'population': {'n': S['nTestCycles'], 'text': f"{S['nTestCycles']:,} expert-annotated breath cycles from {S['nTestPatients']} held-out ICBHI patients (children and adults, Portugal and Greece; four stethoscopes/microphones). Diagnoses: COPD, bronchiectasis, pneumonia, bronchiolitis, URTI, healthy. {ICBHI_CITE}."},
        'reference': 'Per-cycle crackle and wheeze labels from the ICBHI respiratory experts (start and end of every cycle annotated).',
        'indexTest': 'The shipped rules, unchanged: any crackle candidate (transient above the half-level threshold) in the cycle; any tonal segment overlapping the cycle.',
        'independence': 'External: the rules and thresholds were set on SPRSound before ICBHI was available. The ICBHI-trained model in the subgroup rows is internal (same database, held-out patients).',
        'metrics': metrics, **L, 'subgroups': sub,
        'bedside': [per100(metrics[0]['value'], metrics[1]['value']),
                    f'A crackle flag {ri} (LR+ {L["lrPos"]["value"]}); no flag {ro} (LR− {L["lrNeg"]["value"]}).',
                    f"Wheeze rule: finds {w['sens']['value']} % of wheeze cycles, leaves {w['spec']['value']} % of others unflagged."],
        'limits': ['ICBHI recordings are noisy (stethoscope rubbing, speech, heart sounds) and labels are per cycle, not per event.',
                   'Diagnosis, age and device are entangled in ICBHI (all healthy patients are children on one stethoscope).',
                   'Research licence: no ICBHI audio is shown on this site.'],
        'verdict': f"ICBHI score of the shipped rules {R['icbhi']['score']} % vs {S['icbhi']['score']} % for the ICBHI-trained model on the same held-out cycles."}


def icbhi_models():
    """ICBHI-trained sound and illness models: internal validation (held-out patients / repeated patient-grouped CV)."""
    f = WEB_DATA / 'icbhi_models.json'
    if not f.exists(): return None
    M = json.load(open(f)); S = next(m for m in M['models'] if m['id'] == 'icbhi_sound'); b = S['binary']
    ex = [{'label': 'Crackles per cycle (AUC)', 'value': b['crackles']['value'], 'lo': b['crackles']['lo'], 'hi': b['crackles']['hi'], 'n': S['nTestCycles'], 'explain': 'held-out patients; 0.5 = coin toss'},
          {'label': 'Wheezes per cycle (AUC)', 'value': b['wheezes']['value'], 'lo': b['wheezes']['lo'], 'hi': b['wheezes']['hi'], 'n': S['nTestCycles'], 'explain': 'held-out patients'},
          {'label': 'ICBHI score (%)', 'value': S['icbhi']['score'], 'lo': None, 'hi': None, 'n': S['nTestCycles'], 'explain': f"mean of sensitivity {S['icbhi']['se']} % and specificity {S['icbhi']['sp']} %; stethoscope alone {S['deviceOnly']['icbhi']['score']} %"},
          {'label': 'Recording has crackles — page’s own cycles (AUC)', 'value': S['deploy']['crackles']['value'] if S['deploy']['crackles'] else None, 'lo': S['deploy']['crackles']['lo'] if S['deploy']['crackles'] else None,
           'hi': S['deploy']['crackles']['hi'] if S['deploy']['crackles'] else None, 'n': S['deploy']['recordings'], 'explain': 'no expert cycle boundaries: what an upload gets'},
          {'label': 'Recording has wheezes — page’s own cycles (AUC)', 'value': S['deploy']['wheezes']['value'] if S['deploy']['wheezes'] else None, 'lo': S['deploy']['wheezes']['lo'] if S['deploy']['wheezes'] else None,
           'hi': S['deploy']['wheezes']['hi'] if S['deploy']['wheezes'] else None, 'n': S['deploy']['recordings'], 'explain': 'no expert cycle boundaries'}]
    bed = [f"On patients it never saw, the sound model separates crackle cycles from others with AUC {b['crackles']['value']} and wheeze cycles with AUC {b['wheezes']['value']}."]
    for m in M['models']:
        if m['unit'] != 'recording': continue
        t = m['test']
        ex.append({'label': f"{m['title']} (macro AUC)", 'value': t['macroAuc'], 'lo': None, 'hi': None, 'n': m['nPatients'],
                   'explain': f"{m['scheme']}; age alone {m['ageOnly']['macroAuc']}, stethoscope alone {m['deviceOnly']['macroAuc'] if m['deviceOnly']['macroAuc'] is not None else '— (one stethoscope)'}, sound without age {m['soundOnly']['macroAuc']} — {'passed' if m['gate']['deploy'] else 'failed: ' + '; '.join(m['gate']['reasons'])}"})
        bed.append(f"{m['title']}: {'passes its gate and runs on the Analyze page' if m['gate']['deploy'] else 'fails its gate and is not run'} (macro AUC {t['macroAuc']} vs age alone {m['ageOnly']['macroAuc']}).")
    return {
        'id': 'icbhi_models', 'title': 'Sound-type and illness models trained on ICBHI 2017', 'discover': None, 'evidence': 'development',
        'question': 'Which adventitious sound does each breath cycle contain, and which illness pattern does the recording resemble?',
        'intendedUse': 'Research and teaching: the Analyze page shows the per-cycle sound probabilities and, only for models that pass the gate, the illness pattern, each with its explanation. Not a diagnosis.',
        'population': {'n': M['data']['patients'], 'text': f"ICBHI 2017: {M['data']['patients']} patients, {M['data']['recordings']} recordings, {M['data']['cycles']:,} annotated breath cycles. {ICBHI_CITE}."},
        'reference': 'Per-cycle crackle/wheeze labels from the ICBHI experts; per-patient clinical diagnosis from the database.',
        'indexTest': f"Sound model: {S['chosen']} on 27 per-cycle measurements of the page's detectors, chosen by grouped CV on {S['nTrainPatients']} training patients. Illness: logistic regression (fixed beforehand) on the per-cycle sound probabilities, recording features and age, one model per age group.",
        'independence': 'Internal: held-out patients of the same database (sound model) and patient-grouped 5-fold cross-validation repeated 10 times (illness models, too few patients for a split). Same centres and devices as training, so external accuracy will be lower.',
        'metrics': [], 'subgroups': [], 'extra': ex, 'bedside': bed,
        'limits': ['126 patients from two centres; diagnoses are entangled with age (all healthy patients are children) and device, hence separate age groups and age-only / stethoscope-only baselines.',
                   'Rare diagnoses (asthma 1, LRTI 2 patients) are grouped; nothing here can tell asthma from COPD.',
                   'Uploads are scored on the cycles the page finds itself; the deployment rows show what that costs.'],
        'verdict': 'Sound type per cycle: usable for research. Illness pattern: see each model’s gate.'}


def cnn_murmur():
    """Murmur spectrogram network (PyTorch, train_cnn.py): held-out CirCor patients, same as the feature model."""
    f = WEB_DATA / 'cnn_murmur.json'
    if not f.exists(): return None
    M = json.load(open(f)); t = M['test']; v = M['vsCurrent']
    tp, P = t['sens']['k'], t['sens']['n']; tn, N = t['spec']['k'], t['spec']['n']
    metrics, L = two_by_two(tp, P - tp, N - tn, tn, 'valve recordings with a murmur', 'valve recordings')
    ri, ro = lr_words(L)
    sub = [{'group': 'Age band (held-out)', 'rows': [{**prop(b['sens']['k'], b['sens']['n'], f"{b['band']} (AUC {b['auc']['value']}; feature model {b['aucCurrent']['value']})"), 'kind': 'sensitivity',
                                                    'spec': prop(b['spec']['k'], b['spec']['n'], b['band'])} for b in M.get('byBand', [])]}]
    return {
        'id': 'cnn_murmur', 'title': 'Murmur at a valve — spectrogram network (PyTorch)', 'discover': None, 'evidence': 'development',
        'question': 'Is a murmur audible at this valve, as an expert annotator would mark it?',
        'intendedUse': 'Research and teaching: the murmur result on Analyze and Batch. Not a diagnosis.',
        'population': {'n': t['recordings'], 'text': f"{t['recordings']:,} valve recordings from {t['patients']} held-out CirCor children (Brazil, DigiScope stethoscope) — the same patients the previous murmur model is scored on."},
        'reference': 'Per-valve murmur annotation by the CirCor experts (Murmur locations); valves marked Unknown excluded.',
        'indexTest': f"A convolutional network (3 conv blocks, {sum(len(c['w']) + len(c['b']) for c in M['conv']) + len(M['fc']['w']) + 1:,} weights) on the log-power spectrogram 31–1000 Hz, 3-s windows averaged; flag at probability ≥ {M['threshold']} (90 % specificity on training-side validation patients). Trained on {M['train']['patients']} other patients; epoch chosen on {M['train']['validationPatients']} training-side patients.",
        'independence': 'Internal: held-out patients of the same database, never used for training, epoch choice or threshold. Same campaigns and stethoscope, so accuracy on other devices and adults is unknown.',
        'metrics': metrics, **L, 'subgroups': sub,
        'extra': [{'label': 'ROC AUC', 'value': t['auc']['value'], 'lo': t['auc']['lo'], 'hi': t['auc']['hi'], 'n': t['recordings'], 'explain': 'held-out; 0.5 = coin toss'},
                  {'label': 'Gain over the feature model (ΔAUC)', 'value': v['delta'], 'lo': v['ci'][0], 'hi': v['ci'][1], 'n': t['patients'], 'explain': f"same recordings; feature model AUC {v['aucCurrent']}; patient bootstrap 95 % CI"}]
                 + ([{'label': 'Compressed audio: median probability shift (points)', 'value': M['robustness']['medianAbsShift'], 'lo': None, 'hi': None, 'n': M['robustness']['recordings'],
                      'explain': f"8 kHz 32 kbps MP3 vs original WAV; 95th percentile {M['robustness']['p95AbsShift']} points; flag changed in {M['robustness']['flagChanged']} of {M['robustness']['recordings']}"}] if M.get('robustness') else []),
        'bedside': [per100(metrics[0]['value'], metrics[1]['value']), f'A flag {ri} (LR+ {L["lrPos"]["value"]}); no flag {ro} (LR− {L["lrNeg"]["value"]}).',
                    f"It replaced the feature model: better on the same children by {v['delta']} AUC (95 % CI {v['ci'][0]} to {v['ci'][1]}), in every age band."],
        'limits': ['Children from one country and one stethoscope type; adults and other devices untested.', 'Infants: more false flags (lower specificity) than older children.',
                   'Compressed audio tested only as the site’s own 32 kbps MP3 copies of CirCor recordings.' if M.get('robustness') else 'Compressed (MP3) uploads not yet tested for this model.'],
        'verdict': 'Passed its gate and replaced the previous murmur model; internal validation only.'}


def murmur_refit():
    """sysDia cut-off refitted on CirCor training patients (murmur_refit.py), scored on held-out patients."""
    f = META / 'murmur_refit.json'
    if not f.exists(): return None
    R = json.load(open(f)); a, y = R['shipped'], R['youden']
    tp, P = y['sens']['k'], y['sens']['n']; tn, N = y['spec']['k'], y['spec']['n']
    metrics, L = two_by_two(tp, P - tp, N - tn, tn, 'valve recordings with a murmur', 'valve recordings')
    return {
        'id': 'murmur_refit', 'title': 'Murmur rule — cut-off refitted on a training split (systolic : diastolic ratio)', 'discover': 'd-heart', 'evidence': 'development',
        'question': 'Would a better cut-off on the systolic : diastolic energy ratio find more murmurs?',
        'intendedUse': 'Research: recalibration of the rule-based murmur finding (open task 1). The pages use the spectrogram network for murmur.',
        'population': {'n': R['testRecordings'], 'text': f"{R['testRecordings']} valve recordings from {R['testPatients']} held-out CirCor children; cut-offs fitted on {R['trainRecordings']} recordings of other patients."},
        'reference': 'Per-valve murmur annotation by the CirCor experts.',
        'indexTest': f"Ratio alone, flagged at ≥ {y['cut']}× (Youden point on training patients) instead of the shipped 2.5×. {R['note']}.",
        'independence': 'Cut-off chosen on training patients only; scored on held-out patients.',
        'metrics': metrics, **L, 'subgroups': [],
        'extra': [{'label': 'Ratio ROC AUC (held-out)', 'value': R['auc']['value'], 'lo': R['auc']['lo'], 'hi': R['auc']['hi'], 'n': R['testRecordings'], 'explain': 'threshold-free'},
                  {'label': 'Shipped 2.5× on the same recordings: sensitivity (%)', 'value': a['sens']['value'], 'lo': a['sens']['lo'], 'hi': a['sens']['hi'], 'n': a['sens']['n'], 'explain': f"specificity {a['spec']['value']} %"}],
        'bedside': [per100(metrics[0]['value'], metrics[1]['value']),
                    f"Against the shipped 2.5×: {y['sens']['value'] - a['sens']['value']:+.1f} points of sensitivity for {y['spec']['value'] - a['spec']['value']:+.1f} points of specificity.",
                    'The spectrogram network does better than either cut-off on the same children (see its report).'],
        'limits': ['The ratio alone ignores the beat-resolution and heart-rate conditions of the full rule.', 'Precomputed patient-page findings still use the shipped 2.5×.'],
        'verdict': 'A refitted cut-off trades specificity for sensitivity; the network supersedes it for new recordings.'}


def rr_icbhi():
    """Breathing rate vs ICBHI expert cycles (rr_validate.py): external for the shipped estimator."""
    f = META / 'rr_validation.json'
    if not f.exists(): return None
    R = json.load(open(f)); a, c = R['shipped'], R['candidate']
    ex = [{'label': 'Mean absolute error, shipped estimator (breaths/min)', 'value': a['mae'], 'lo': None, 'hi': None, 'n': R['recordings'], 'explain': f"bias {a['bias']}, 95 % limits {a['loa'][0]} to {a['loa'][1]}; {a['within3']} % within ±3/min; ρ {a['rho']}"},
          {'label': 'Mean absolute error, candidate estimator (breaths/min)', 'value': c['mae'], 'lo': None, 'hi': None, 'n': R['recordings'], 'explain': f"{c['within3']} % within ±3/min; ρ {c['rho']}; tuned on training patients only"},
          {'label': 'Error reduction by the candidate (breaths/min)', 'value': R['reduction']['mae'], 'lo': R['reduction']['ci'][0], 'hi': R['reduction']['ci'][1], 'n': R['patients'], 'explain': 'patient bootstrap 95 % CI; adopted only if the CI is above 0'}]
    ex += [{'label': f"{g['group'].capitalize()}: error shipped / candidate", 'value': g['maeShipped'], 'lo': None, 'hi': None, 'n': g['recordings'], 'explain': f"candidate {g['maeCandidate']} breaths/min"} for g in R['byGroup']]
    return {
        'id': 'rr_icbhi', 'title': 'Breathing rate from breath sound — against expert-marked breath cycles (ICBHI)', 'discover': None, 'evidence': 'external',
        'question': 'How many breaths per minute is this patient taking?',
        'intendedUse': 'Research and teaching: the breathing rate shown on patient pages and Analyze.',
        'population': {'n': R['recordings'], 'text': f"{R['recordings']} recordings (≥ 3 annotated cycles) from {R['patients']} held-out ICBHI patients, children and adults. {ICBHI_CITE}."},
        'reference': '60 ÷ the median length of the respiratory cycles marked by the ICBHI experts in that recording.',
        'indexTest': 'Shipped: cycle detector in analyzeSignal (unchanged since before ICBHI). Candidate: autocorrelation of the 125–1000 Hz sound envelope, lag 1–8 s.',
        'independence': 'External for the shipped estimator (set before ICBHI). The candidate was tuned on other ICBHI patients.',
        'metrics': [], 'subgroups': [], 'extra': ex,
        'bedside': [f"The shipped rate is off by {a['mae']} breaths/min on average, and one recording can be off by {abs(a['loa'][0]):.0f} to {a['loa'][1]:.0f}/min — too wide to call tachypnoea from one clip.",
                    f"The candidate is closer ({c['mae']}/min) but the gain ({R['reduction']['mae']}, CI {R['reduction']['ci'][0]} to {R['reduction']['ci'][1]}) is not certain, so it is {'adopted' if R['adopt'] else 'not adopted'}."],
        'limits': ['Expert cycles define the reference; they are hand-marked, not from a flow sensor.', 'Clips of ~20 s: a handful of breaths each.'],
        'verdict': 'Not reliable enough for clinical use; reported for transparency.'}


def cnn_lung():
    """Crackle / wheeze spectrogram network on ICBHI (train_lung_cnn.py): a comparison, not used by the pages."""
    f = META / 'cnn_lung.json'
    if not f.exists(): return None
    M = json.load(open(f)); t = M['test']; c, w = t['crackles'], t['wheezes']
    ex = [{'label': f"{n.capitalize()} per cycle — network (AUC)", 'value': r['auc']['value'], 'lo': r['auc']['lo'], 'hi': r['auc']['hi'], 'n': t['cycles'], 'explain': f"held-out; current cycle model on the same cycles {r['aucCurrentSame']}"} for n, r in (('crackles', c), ('wheezes', w))]
    ex += [{'label': f"Gain over the current cycle model — {n} (ΔAUC)", 'value': r['delta'], 'lo': r['ci'][0], 'hi': r['ci'][1], 'n': t['patients'], 'explain': 'patient bootstrap 95 % CI'} for n, r in (('crackles', c), ('wheezes', w))]
    return {
        'id': 'cnn_lung', 'title': 'Crackle / wheeze spectrogram network per breath cycle (PyTorch, ICBHI)', 'discover': None, 'evidence': 'research',
        'question': 'Does a network on the spectrogram find crackles and wheezes per breath cycle better than the current cycle model?',
        'intendedUse': 'Research comparison only. Not used by the pages: it did not meet the replacement rule set before training.',
        'population': {'n': t['cycles'], 'text': f"{t['cycles']:,} expert-annotated breath cycles from {t['patients']} held-out ICBHI patients (the same patients the current cycle model is scored on). {ICBHI_CITE}."},
        'reference': 'Per-cycle crackle and wheeze labels from the ICBHI experts.',
        'indexTest': f"Convolutional network on the log-power spectrogram 31–2000 Hz, one 2.56-s window centred on each cycle, two outputs; trained on {M['train']['patients']} other patients, epoch chosen on {M['train']['validationPatients']} training-side patients.",
        'independence': 'Internal: held-out patients of the same database; nothing tuned on them.',
        'metrics': [], 'subgroups': [], 'extra': ex,
        'bedside': [f"Wheezes: AUC {w['auc']['value']} against {w['aucCurrentSame']} for the current model — a gain of {w['delta']} whose 95 % CI ({w['ci'][0]} to {w['ci'][1]}) includes no gain.",
                    f"Crackles: AUC {c['auc']['value']} against {c['aucCurrentSame']} — no improvement.",
                    'Rule set before training: replace the current model only if the mean gain has a CI above 0. Not met, so the pages keep the current model.'],
        'limits': ['ICBHI is noisy and small (126 patients); crackle labels are per cycle, not per event.', 'Research licence: no ICBHI audio on this site.'],
        'verdict': 'No reliable gain over the current cycle model; not deployed.'}


def main():
    I = json.load(open(WEB_DATA / 'index.json')); D = json.load(open(WEB_DATA / 'discover.json'))
    old = json.load(open(WEB_DATA / 'validation.json')) if (WEB_DATA / 'validation.json').exists() else {}
    reports = [r for r in (murmur(I), cnn_murmur(), resp_spr(I), resp_kauh(I), icbhi_rules(), apnea(), hrv(D), age(D), icbhi_models(), cnn_lung(), rr_icbhi(), murmur_refit()) if r]
    out = {'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'engine': 'Seismometer 0.5.0 (Epic, open source): Agresti–Coull and Hanley–McNeil intervals; likelihood ratios by the log method',
           'reports': reports, 'ml': old.get('ml')}
    (WEB_DATA / 'validation.json').write_text(json.dumps(out, separators=(',', ':'), ensure_ascii=False))
    for r in reports:
        m = ' · '.join(f"{x['label']} {x['value']} % [{x['lo']}–{x['hi']}]" for x in r['metrics'][:2])
        print(f"{r['id']:10} {r['evidence']:12} {m}  {('LR+ ' + str(r['lrPos']['value']) + ' LR- ' + str(r['lrNeg']['value'])) if 'lrPos' in r else ''}")
    print('-> web/data/validation.json   (then merge.py --bump)')


if __name__ == '__main__':
    main()
