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
  external     thresholds were fixed before this dataset was seen — an honest estimate for similar patients
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
        'independence': '<b>Development data:</b> these thresholds were tuned on SPRSound. The figures are optimistic — the KAUH result below is the honest estimate on new patients.',
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


def main():
    I = json.load(open(WEB_DATA / 'index.json')); D = json.load(open(WEB_DATA / 'discover.json'))
    old = json.load(open(WEB_DATA / 'validation.json')) if (WEB_DATA / 'validation.json').exists() else {}
    reports = [r for r in (murmur(I), resp_spr(I), resp_kauh(I), apnea(), age(D)) if r]
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
