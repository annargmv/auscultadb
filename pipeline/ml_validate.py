"""Machine-learning models on the AuscultaDB features, validated with Seismometer → validation.json['ml'].

Seismometer (Epic's open-source healthcare-AI validation package, `pip install seismometer`) supplies the metrics:
`BinaryClassifierMetricGenerator` for sensitivity / specificity / PPV / NPV at the operating threshold, Agresti–Coull
intervals for every proportion, Hanley–McNeil intervals for the ROC AUC, and its fairness-audit rule (largest
subgroup = reference; a subgroup whose metric is outside ×0.80–1.25 of it is flagged, outside ×0.67–1.50 critical).

Everything below was fixed before looking at any test result (rule 4):
  * split by PATIENT, stratified on the label, 60 % train / 40 % test, seed SEED — no patient is in both;
  * two candidate models (standardised L2 logistic regression; shallow gradient boosting), chosen by 5-fold
    cross-validated ROC AUC on the TRAINING split only;
  * the operating threshold is the score that reaches TARGET_SPEC specificity on the training split's out-of-fold
    predictions — never tuned on the test split;
  * the shipped rule-based detector is scored on the same test patients, so the two can be compared directly.

Models
  A  murmur — CirCor patients: valve features (systolic:diastolic ratio vs baseline, beat consistency, heart rate,
     resolved beats …) aggregated per patient (max / mean over valves). Label: expert murmur Present vs Absent.
  B  adventitious sound — SPRSound recordings: spectral and detector features. Label: expert record label not
     Normal. Then applied UNCHANGED to KAUH adults (diaphragm mode) as an external test on another stethoscope.

These are research models. Nothing here changes what the patient pages show.

  python3 pipeline/ml_validate.py      (after clinical_validation.py; then merge.py --bump)
"""
import collections, datetime, json, math
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from seismometer.data.performance import BinaryClassifierMetricGenerator
from seismometer.data.confidence import agresti_coull_interval, hanley_mcneill_confidence
from seismometer.data.confidence.parameters import PerformanceMetricConfidenceParam, ROCConfidenceParam
from paths import FEATURES, WEB_DATA

SEED, TEST_FRAC, TARGET_SPEC, FAIR = 20261108, 0.4, 0.90, 0.25
GEN = BinaryClassifierMetricGenerator(rho=0.5)
CI = PerformanceMetricConfidenceParam(0.95)
CANDIDATES = {
    'logistic regression': lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
    'gradient boosting': lambda: make_pipeline(SimpleImputer(strategy='median'), HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED)),
}


def prop(k, n, label):
    if not n: return {'label': label, 'k': int(k), 'n': int(n), 'value': None, 'lo': None, 'hi': None}
    r = agresti_coull_interval(CI, k / n, n)
    return {'label': label, 'k': int(k), 'n': int(n), 'value': round(100 * k / n, 1), 'lo': round(100 * max(0, float(r.lower)), 1), 'hi': round(100 * min(1, float(r.upper)), 1)}


def split_patients(df, rng):
    """Stratified patient-level split: each patient's label is its (single) label."""
    pl = df.groupby('pid')['y'].max()
    test = set()
    for lab in (0, 1):
        ids = sorted(pl[pl == lab].index); rng.shuffle(ids)
        test |= set(ids[:int(round(len(ids) * TEST_FRAC))])
    return df[~df.pid.isin(test)], df[df.pid.isin(test)]


def at_threshold(y, s, thr):
    """Seismometer's binary metrics at one threshold (scores are probabilities)."""
    d = pd.DataFrame({'y': np.asarray(y).astype(int), 's': np.clip(np.asarray(s, float), 0, 1)})
    m = GEN.delegate_call(d, ['TP', 'FN', 'FP', 'TN'], target_col='y', score_col='s', score_threshold=thr)
    tp, fn, fp, tn = (int(m[k]) for k in ('TP', 'FN', 'FP', 'TN'))
    return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'metrics': [prop(tp, tp + fn, 'Sensitivity'), prop(tn, tn + fp, 'Specificity'), prop(tp, tp + fp, 'PPV'), prop(tn, tn + fn, 'NPV')]}


def counts_metrics(tp, fn, fp, tn):
    return [prop(tp, tp + fn, 'Sensitivity'), prop(tn, tn + fp, 'Specificity'), prop(tp, tp + fp, 'PPV'), prop(tn, tn + fn, 'NPV')]


def auc_ci(y, s):
    a = hanley_mcneill_confidence(ROCConfidenceParam(0.95), np.asarray(y, float), np.clip(np.asarray(s, float), 0, 1))
    return {'value': round(float(a.value), 3), 'lo': round(float(a.lower), 3), 'hi': round(float(min(1, a.upper)), 3)}


def fairness(test, score_col, thr, cohorts):
    """Seismometer's fairness-audit rule, on sensitivity and specificity, per cohort."""
    out = []
    for col, name in cohorts:
        groups = [(g, d) for g, d in test.groupby(col) if len(d) >= 10 and d.y.nunique() == 2]
        if len(groups) < 2: continue
        ref_g, ref_d = max(groups, key=lambda x: len(x[1]))
        ref = at_threshold(ref_d.y, ref_d[score_col], thr)
        rows = []
        for g, d in groups:
            m = at_threshold(d.y, d[score_col], thr); flags = []
            for i, k in ((0, 'sensitivity'), (1, 'specificity')):
                a, b = m['metrics'][i]['value'], ref['metrics'][i]['value']
                if a is None or b is None or b == 0 or a == 0: continue
                r = a / b; worst = max(r, 1 / r)
                if worst > 1 + 2 * FAIR: flags.append(f'{k} critically {"higher" if r > 1 else "lower"} (×{r:.2f})')
                elif worst > 1 + FAIR: flags.append(f'{k} {"higher" if r > 1 else "lower"} (×{r:.2f})')
            rows.append({'group': str(g), 'n': len(d), 'positives': int(d.y.sum()), 'reference': g == ref_g,
                         'sens': m['metrics'][0], 'spec': m['metrics'][1], 'flags': flags})
        out.append({'cohort': name, 'rows': rows})
    return out


def fit_and_validate(df, feats, rng):
    tr, te = split_patients(df, rng)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cvres = {}
    for name, mk in CANDIDATES.items():
        oof = cross_val_predict(mk(), tr[feats], tr.y, cv=cv, method='predict_proba')[:, 1]
        cvres[name] = (roc_auc_score(tr.y, oof), oof)
    best = max(cvres, key=lambda k: cvres[k][0]); oof = cvres[best][1]
    # operating point from TRAINING out-of-fold scores: lowest threshold (1 % grid, Seismometer's) reaching TARGET_SPEC
    neg = np.sort(oof[tr.y.values == 0]); thr = None
    for t in np.arange(0.01, 1.0, 0.01):
        if np.mean(neg < t) >= TARGET_SPEC: thr = round(float(t), 2); break
    model = CANDIDATES[best]().fit(tr[feats], tr.y)
    te = te.copy(); te['score'] = model.predict_proba(te[feats])[:, 1]
    res = at_threshold(te.y, te.score, thr)
    return model, tr, te, {'chosen': best, 'cvAuc': {k: round(v[0], 3) for k, v in cvres.items()}, 'threshold': thr,
                           'nTrain': int(tr.pid.nunique()), 'nTest': int(te.pid.nunique()), 'testAuc': auc_ci(te.y, te.score), **res}


def lr_pair(tp, fn, fp, tn):
    if 0 in (tp, fn, fp, tn): tp, fn, fp, tn = tp + .5, fn + .5, fp + .5, tn + .5
    se, sp = tp / (tp + fn), tn / (tn + fp)
    return {'lrPos': round(se / (1 - sp), 2), 'lrNeg': round((1 - se) / sp, 2)}


# ---------------------------------------------------------------------------------------------------- A: murmur
def murmur(rng):
    R = [json.loads(l) for l in open(FEATURES / 'circor.jsonl')]
    rows = []
    for r in R:
        rows.append({'pid': r['pid'], 'murmur': r['murmur'], 'det': r['det']['murmur'], 'band': r['ageBand'] or 'unknown',
                     'sex': 'female' if r['g'] == '1' else 'male', 'campaign': r['campaign'],
                     'ratio': r['sysDia'], 'rel': (r['sysDia'] / r['baseSysDia']) if r.get('sysDia') and r.get('baseSysDia') else np.nan,
                     'consistency': r['consistency'], 'hr': r['hr'], 'quiet': r['quietBeats'], 'beats': r['nBeats'], 's3': r['nS3'], 'dur': r['dur']})
    v = pd.DataFrame(rows)
    agg = {f: ['max', 'mean'] for f in ('ratio', 'rel', 'consistency')} | {f: ['mean'] for f in ('hr', 'quiet', 'beats', 's3', 'dur')}
    P = v.groupby('pid').agg(agg); P.columns = [f'{a}_{b}' for a, b in P.columns]
    meta = v.groupby('pid').agg(murmur=('murmur', 'first'), det=('det', 'max'), band=('band', 'first'), sex=('sex', 'first'), campaign=('campaign', 'first'))
    P = P.join(meta).reset_index(); P = P[P.murmur.isin(['Present', 'Absent'])].copy(); P['y'] = (P.murmur == 'Present').astype(int)
    feats = [c for c in P.columns if c.endswith(('_max', '_mean'))]
    model, tr, te, res = fit_and_validate(P, feats, rng)
    # the shipped rule on the SAME test patients
    sh = te.det.astype(bool); y = te.y.astype(bool)
    tp, fn, fp, tn = int((sh & y).sum()), int((~sh & y).sum()), int((sh & ~y).sum()), int((~sh & ~y).sum())
    return {'id': 'ml_murmur', 'title': 'Machine-learning murmur model (heart sounds)', 'task': 'Expert-annotated murmur present vs absent, per patient',
            'data': 'CirCor DigiScope 1.0.3 — children, two Brazilian screening campaigns', 'features': feats, **res, **lr_pair(res['tp'], res['fn'], res['fp'], res['tn']),
            'comparator': {'name': 'shipped rule (sysDia ≥ 2.5× baseline), same test patients', 'metrics': counts_metrics(tp, fn, fp, tn), **lr_pair(tp, fn, fp, tn)},
            'fairness': fairness(te, 'score', res['threshold'], [('band', 'Age band'), ('sex', 'Sex'), ('campaign', 'Screening campaign')]),
            'note': 'Fit on training patients only. The rule shown on the patient pages is unchanged; Discoveries 13 still reports the shipped rule on all 941 patients.'}


# ------------------------------------------------------------------------------------ B: adventitious sound
FEATS_R = ['f50', 'f75', 'cen', 'rr', 'hr', 'crkRate', 'fineRate', 'tonFrac', 'ie']


def resp(rng):
    S = [json.loads(l) for f in sorted(FEATURES.glob('spr.*.jsonl')) for l in open(f)]
    S = pd.DataFrame([{**{f: r.get(f) for f in FEATS_R}, 'pid': r['pid'], 'y': int(r['label'] != 'Normal'), 'release': r['rel'],
                       'sex': 'female' if r['g'] == '1' else 'male', 'ageg': '<1 y' if r['age'] < 1 else '1–5 y' if r['age'] < 6 else '6–17 y'}
                      for r in S if r.get('age') is not None and r['label'] in ('Normal', 'CAS', 'DAS', 'CAS & DAS')]).astype({f: float for f in FEATS_R})
    model, tr, te, res = fit_and_validate(S, FEATS_R, rng)
    # external: the frozen model and threshold on KAUH adults, diaphragm mode
    K = pd.DataFrame([{**{f: r.get(f) for f in FEATS_R}, 'pid': r['pid'], 'y': int(r['label'] != 'Normal')}
                      for r in (json.loads(l) for l in open(FEATURES / 'kauh.jsonl'))
                      if r['mode'] == 'D' and r['label'] in ('Normal', 'CAS', 'DAS', 'CAS & DAS')]).astype({f: float for f in FEATS_R})
    K['score'] = model.predict_proba(K[FEATS_R])[:, 1]
    ext = {'name': 'KAUH adults, Littmann 3200 diaphragm — model and threshold frozen', 'n': int(len(K)), 'auc': auc_ci(K.y, K.score), **at_threshold(K.y, K.score, res['threshold'])}
    return {'id': 'ml_resp', 'title': 'Machine-learning adventitious-sound model (breath sounds)', 'task': 'Expert recording label abnormal (CAS, DAS or both) vs normal, per recording',
            'data': 'SPRSound, all four releases — children, Shanghai, Yunting stethoscope', 'features': FEATS_R, **res, **lr_pair(res['tp'], res['fn'], res['fp'], res['tn']),
            'external': ext, 'fairness': fairness(te, 'score', res['threshold'], [('release', 'Release (recording batch)'), ('ageg', 'Age group'), ('sex', 'Sex')]),
            'note': 'Recording-level; several recordings per child, all in the same split. Release is a known confound (labels and devices drift between releases).'}


def bedside(m):
    s, c = m['metrics'][0], m['metrics'][1]
    t = [f"On {m['nTest']} held-out patients it flags {s['value']} % (95 % CI {s['lo']}–{s['hi']}) of those with the finding and wrongly flags {round(100 - c['value'], 1)} % of those without.",
         f"ROC AUC {m['testAuc']['value']} (95 % CI {m['testAuc']['lo']}–{m['testAuc']['hi']}); 0.5 is chance."]
    if 'comparator' in m:
        cs, cc = m['comparator']['metrics'][0], m['comparator']['metrics'][1]
        t.append(f"The rule the site uses today, on the same patients: sensitivity {cs['value']} %, specificity {cc['value']} %. "
                 + ('The model finds more murmurs' if s['value'] > cs['value'] else 'The model finds fewer murmurs')
                 + f" at {'lower' if c['value'] < cc['value'] else 'higher'} specificity — a different point on the same trade-off, not a free gain." )
    if 'external' in m:
        e = m['external']; es, ec = e['metrics'][0], e['metrics'][1]
        t.append(f"On adults recorded with a different stethoscope (KAUH) the frozen model reaches AUC {e['auc']['value']} — sensitivity {es['value']} %, specificity {ec['value']} %."
                 + (' It does not transfer.' if e['auc']['value'] < 0.65 else ''))
    flags = [f"{r['group']} ({f['cohort']}): {', '.join(r['flags'])}" for f in m['fairness'] for r in f['rows'] if r['flags']]
    t.append('Fairness audit (Seismometer rule): ' + ('; '.join(flags) if flags else 'no subgroup outside ×0.80–1.25 of the reference group') + '.')
    return t


def main():
    rng = np.random.default_rng(SEED)
    models = [murmur(rng), resp(rng)]
    for m in models: m['bedside'] = bedside(m)
    VJ = WEB_DATA / 'validation.json'; V = json.load(open(VJ)) if VJ.exists() else {'reports': []}
    V['ml'] = {'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'engine': 'scikit-learn models; Seismometer 0.5.0 metrics, intervals and fairness rule',
               'protocol': {'seed': SEED, 'testFrac': TEST_FRAC, 'targetSpec': TARGET_SPEC, 'fairnessRatio': FAIR, 'candidates': list(CANDIDATES)}, 'models': models}
    VJ.write_text(json.dumps(V, separators=(',', ':'), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    for m in models:
        s, c = m['metrics'][0], m['metrics'][1]
        print(f"{m['id']:10} {m['chosen']:20} cvAUC {m['cvAuc']} thr {m['threshold']} | test AUC {m['testAuc']} sens {s['value']} [{s['lo']}-{s['hi']}] spec {c['value']} [{c['lo']}-{c['hi']}]")
        if 'comparator' in m: print('   shipped rule:', [(x['label'], x['value']) for x in m['comparator']['metrics'][:2]])
        if 'external' in m: print('   external KAUH:', m['external']['auc'], [(x['label'], x['value']) for x in m['external']['metrics'][:2]])
        for f in m['fairness']:
            for r in f['rows']:
                if r['flags']: print('   fairness', f['cohort'], r['group'], r['flags'])
    print('-> web/data/validation.json [ml]   (then merge.py --bump)')


if __name__ == '__main__':
    main()
