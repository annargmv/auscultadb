"""Train the models behind the Analyze page and export them for the browser → web/data/models.json.

A new recording uploaded on the Analyze page goes through the page's own analysis (`analyzeSignal`) and the page's
own feature extractor (`recordingFeatures`, the same code that built meta/features/*.jsonl). These models turn those
features into probabilities. They answer "which labelled pattern in the open datasets does this recording resemble",
not "what disease does this person have" — the data cannot support a diagnosis, and the page says so.

Models (one recording in, probabilities out)
  heart_murmur        CirCor valve recordings → murmur audible at this valve (expert per-valve label)        children
  lung_sound          SPRSound recordings → expert sound label: normal / continuous (wheeze, rhonchi) /
                      discontinuous (crackles) / both                                                          children
  lung_pattern_child  SPRSound recordings → the child's diagnosis group: pneumonia / bronchitis / asthma /
                      healthy control                                                                         children
  lung_pattern_adult  KAUH recordings (all three filter modes) → healthy / asthma / heart failure            adults, pilot

Protocol, fixed before any test result was seen (rule 4):
  * split by PATIENT, stratified by label, 60 % train / 40 % test, seed SEED — no patient in both;
  * candidates: multinomial logistic regression (standardised, median-imputed) and shallow gradient boosting
    (native missing values), chosen by grouped 5-fold CV on the training patients (macro one-vs-rest AUC);
  * the DEPLOYED model is the one fit on the training patients only, so the held-out numbers below describe
    exactly the model the browser runs;
  * held-out: accuracy (Agresti–Coull CI), chance and confound baselines, macro AUC, per class sensitivity /
    specificity / PPV (CIs) and one-vs-rest AUC (Hanley–McNeil CI, Seismometer), confusion matrix, calibration;
  * external: lung_sound frozen on KAUH adults (another country, age range and stethoscope).
Also exported: training feature ranges (1st–99th percentile) so the page can warn when a recording is unlike the
training data, and parity vectors so the e2e test can prove the browser reproduces these probabilities.

  python3 pipeline/train_models.py      (then merge.py --bump)
"""
import collections, datetime, json, math
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from seismometer.data.confidence import agresti_coull_interval, hanley_mcneill_confidence
from seismometer.data.confidence.parameters import PerformanceMetricConfidenceParam, ROCConfidenceParam
from paths import FEATURES, WEB_DATA

SEED, TEST_FRAC = 20261111, 0.4
FLAG_SPEC = 0.90          # per-class flag threshold: the score that leaves 90 % of TRAINING non-members unflagged
GATE_AUC = 0.70           # deployment gate (see `gate`): set after the first run showed the diagnosis models fail
# The owner switched these on on 2026-11-11 although they FAIL the gate. They run as "experimental": the page shows
# their failed held-out numbers beside every output. The gate result itself is never altered.
OWNER_ON = {'lung_pattern_child', 'lung_pattern_adult'}
CI = PerformanceMetricConfidenceParam(0.95)
LUNG = ['f50', 'f75', 'cen', 'rr', 'hr', 'crkRate', 'fineRate', 'tonFrac', 'ie', 'dur']
HEART = ['sysDia', 'consistency', 'hr', 'quietBeats', 'nBeats', 'nS3', 'dur']


def prop(k, n):
    if not n: return {'k': int(k), 'n': int(n), 'value': None, 'lo': None, 'hi': None}
    r = agresti_coull_interval(CI, k / n, n)
    return {'k': int(k), 'n': int(n), 'value': round(100 * k / n, 1), 'lo': round(100 * max(0, float(r.lower)), 1), 'hi': round(100 * min(1, float(r.upper)), 1)}


def auc_ci(y, s):
    if len(set(y)) < 2: return None
    a = hanley_mcneill_confidence(ROCConfidenceParam(0.95), np.asarray(y, float), np.clip(np.asarray(s, float), 0, 1))
    return {'value': round(float(roc_auc_score(y, s)), 3), 'lo': round(float(a.lower), 3), 'hi': round(float(min(1, a.upper)), 3)}


# ------------------------------------------------------------------------------------------------ candidates
class LR:
    name = 'logistic regression'
    def fit(self, X, y):
        self.imp = SimpleImputer(strategy='median').fit(X); Z = self.imp.transform(X)
        self.sc = StandardScaler().fit(Z); self.m = LogisticRegression(C=1.0, max_iter=5000).fit(self.sc.transform(Z), y); return self
    def proba(self, X): return self.m.predict_proba(self.sc.transform(self.imp.transform(X)))
    def export(self):
        coef, icpt = self.m.coef_, self.m.intercept_
        return {'kind': 'logreg', 'median': self.imp.statistics_.tolist(), 'mean': self.sc.mean_.tolist(), 'scale': self.sc.scale_.tolist(),
                'coef': coef.tolist(), 'intercept': icpt.tolist(), 'binary': coef.shape[0] == 1}


class GB:
    name = 'gradient boosting'
    def fit(self, X, y):
        self.m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=SEED).fit(X, y); return self
    def proba(self, X): return self.m.predict_proba(X)
    def export(self):
        trees = []
        for it in self.m._predictors:
            per = []
            for tp in it:
                n = tp.nodes
                per.append([[int(r['feature_idx']), float(r['num_threshold']), int(r['missing_go_to_left']), int(r['left']), int(r['right']), int(r['is_leaf']), float(r['value'])] for r in n])
            trees.append(per)
        return {'kind': 'gbtree', 'baseline': np.asarray(self.m._baseline_prediction).ravel().tolist(), 'trees': trees,
                'binary': len(self.m.classes_) == 2}


def split(df, rng):
    """Patient-level, stratified by each patient's label."""
    pl = df.groupby('pid')['y'].agg(lambda s: s.mode().iloc[0]); test = set()
    for lab in sorted(pl.unique()):
        ids = sorted(pl[pl == lab].index); rng.shuffle(ids); test |= set(ids[:int(round(len(ids) * TEST_FRAC))])
    return df[~df.pid.isin(test)].reset_index(drop=True), df[df.pid.isin(test)].reset_index(drop=True)


def macro_auc(y, P, K):
    return float(np.mean([roc_auc_score(y == k, P[:, k]) for k in range(K) if 0 < (y == k).sum() < len(y)]))


def choose(tr, feats, K):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED); res = {}
    for C in (LR, GB):
        oof = np.zeros((len(tr), K))
        for a, b in cv.split(tr[feats], tr.y, tr.pid):
            oof[b] = C().fit(tr.loc[a, feats].values, tr.y[a].values).proba(tr.loc[b, feats].values)
        res[C.name] = (round(macro_auc(tr.y.values, oof, K), 3), C, oof)
    best = max(res, key=lambda k: res[k][0])
    return res[best][1], {k: v[0] for k, v in res.items()}, res[best][2]


def flag_metrics(isk, score, t):
    f = score >= t; tp = int((f & isk).sum()); fn = int((~f & isk).sum()); fp = int((f & ~isk).sum()); tn = int((~f & ~isk).sum())
    return {'threshold': t, 'sens': prop(tp, tp + fn), 'spec': prop(tn, tn + fp), 'ppv': prop(tp, tp + fp)}


def thresholds(y, oof, K):
    """Per class, the lowest score (0.01 grid) that leaves FLAG_SPEC of training non-members below it — out-of-fold."""
    out = []
    for k in range(K):
        neg = oof[y != k, k]; t = next((round(float(x), 2) for x in np.arange(0.01, 1.0, 0.01) if np.mean(neg < x) >= FLAG_SPEC), 0.99)
        out.append(t)
    return out


def gate(m):
    """A model runs on the Analyze page only if, on held-out patients, its macro AUC reaches GATE_AUC and every class's
    one-vs-rest AUC has a 95 % CI lower bound above 0.5. Otherwise it is listed as failed, with its numbers."""
    t = m['test']; why = []
    if t['macroAuc'] < GATE_AUC: why.append(f"macro AUC {t['macroAuc']} < {GATE_AUC}")
    bad = [c['class'] for c in t['perClass'] if not c['auc'] or c['auc']['lo'] <= 0.5]
    if bad: why.append('no better than chance for ' + ', '.join(bad))
    if m.get('confound') and m['confound']['value'] is not None and t['accuracy']['value'] <= m['confound']['value']:
        why.append(f"accuracy {t['accuracy']['value']} % does not beat the recording-batch baseline ({m['confound']['value']} %)")
    if t['accuracy']['value'] <= m['majority']['value']: why.append(f"accuracy {t['accuracy']['value']} % does not beat always answering the commonest class ({m['majority']['value']} %)")
    return {'deploy': not why, 'reasons': why}


def evaluate(y, P, classes, thr=None):
    K = len(classes); pred = P.argmax(1); cm = confusion_matrix(y, pred, labels=list(range(K)))
    per = []
    for k, c in enumerate(classes):
        tp = int(cm[k, k]); fn = int(cm[k].sum() - tp); fp = int(cm[:, k].sum() - tp); tn = int(cm.sum() - tp - fn - fp)
        per.append({'class': c, 'n': int((y == k).sum()), 'sens': prop(tp, tp + fn), 'spec': prop(tn, tn + fp), 'ppv': prop(tp, tp + fp),
                    'auc': auc_ci((y == k).astype(int), P[:, k])})
    # calibration of the top-class probability: is "70 % sure" right 70 % of the time?
    conf = P.max(1); right = (pred == y); bins = []
    for lo, hi in ((0, .4), (.4, .55), (.55, .7), (.7, .85), (.85, 1.01)):
        m = (conf >= lo) & (conf < hi)
        if m.sum() >= 5: bins.append({'range': [lo, min(hi, 1)], 'n': int(m.sum()), 'meanConf': round(float(conf[m].mean()), 3), 'accuracy': round(float(right[m].mean()), 3)})
    brier = float(np.mean(np.sum((P - np.eye(K)[y]) ** 2, 1)))
    return {'n': int(len(y)), 'accuracy': prop(int(right.sum()), len(y)), 'macroAuc': round(macro_auc(y, P, K), 3), 'perClass': per,
            'flags': [{'class': c, **flag_metrics(y == k, P[:, k], thr[k])} for k, c in enumerate(classes)] if thr else None,
            'confusion': cm.tolist(), 'calibration': bins, 'brier': round(brier, 3)}


def train(task, df, feats, classes, rng, meta):
    df = df.copy(); df['y'] = df.label.map({c: i for i, c in enumerate(classes)}); df = df[df.y.notna()].copy(); df['y'] = df.y.astype(int)
    tr, te = split(df, rng); K = len(classes)
    C, cvres, oof = choose(tr, feats, K)
    thr = thresholds(tr.y.values, oof, K)
    model = C().fit(tr[feats].values, tr.y.values)
    P = model.proba(te[feats].values)
    out = {'id': task, 'classes': classes, 'features': feats, 'model': model.export(), 'chosen': C.name, 'cv': cvres,
           'thresholds': thr, 'nTrainPatients': int(tr.pid.nunique()), 'nTestPatients': int(te.pid.nunique()), 'nTrainRec': len(tr), 'nTestRec': len(te),
           'test': evaluate(te.y.values, P, classes, thr),
           'majority': prop(int((te.y == tr.y.mode().iloc[0]).sum()), len(te)),
           'range': {f: [None if np.isnan(v) else round(float(v), 4) for v in np.nanpercentile(tr[f].astype(float), [1, 99])] for f in feats},
           'parity': [{'x': [None if (isinstance(v, float) and math.isnan(v)) else float(v) for v in te.loc[i, feats]], 'p': [round(float(q), 9) for q in P[i]]}
                      for i in range(0, len(te), max(1, len(te) // 25))][:25], **meta}
    # per patient (average the patient's recordings), where patients have several
    if te.groupby('pid').size().max() > 1:
        g = pd.DataFrame(P).groupby(te.pid.values).mean(); yp = te.groupby('pid').y.first().loc[g.index].values
        out['testPatient'] = evaluate(yp, g.values, classes, thr)
    return out, model, te, P


def main():
    rng = np.random.default_rng(SEED); models = []
    # ---- heart: valve-level murmur
    C = pd.DataFrame([{**{f: r.get(f) for f in HEART}, 'pid': r['pid'], 'label': {'Murmur': 'murmur', 'Normal': 'no murmur'}.get(r['siteTruth']), 'band': r['ageBand']}
                      for r in (json.loads(l) for l in open(FEATURES / 'circor.jsonl')) if 'err' not in r]).astype({f: float for f in HEART})
    m, *_ = train('heart_murmur', C, HEART, ['no murmur', 'murmur'], rng, {
        'title': 'Murmur at this valve', 'kind': 'heart', 'ageRange': [0, 21], 'data': 'CirCor DigiScope 1.0.3 — children, Brazil, DigiScope stethoscope, per-valve expert murmur labels',
        'question': 'Is a murmur audible at this valve, as an expert annotator would mark it?'})
    models.append(m)
    # ---- lung: sound type (children), external check on KAUH adults
    S = [json.loads(l) for f in sorted(FEATURES.glob('spr.*.jsonl')) for l in open(f)]
    snd = {'Normal': 'normal', 'CAS': 'continuous (wheeze / rhonchi)', 'DAS': 'crackles', 'CAS & DAS': 'both'}
    Sd = pd.DataFrame([{**{f: r.get(f) for f in LUNG}, 'pid': r['pid'], 'label': snd.get(r['label']), 'release': r['rel'], 'dx': r['disease']} for r in S if r.get('age') is not None]).astype({f: float for f in LUNG})
    cls = list(snd.values())
    m, model, te, P = train('lung_sound', Sd, LUNG, cls, rng, {
        'title': 'Breath-sound type', 'kind': 'lung', 'ageRange': [0, 18], 'data': 'SPRSound (all releases) — children, Shanghai, Yunting stethoscope, expert recording labels',
        'question': 'Which kind of sound would an expert label this recording with?'})
    K = pd.DataFrame([{**{f: r.get(f) for f in LUNG}, 'label': snd.get(r['label'])} for r in (json.loads(l) for l in open(FEATURES / 'kauh.jsonl')) if r['mode'] == 'D']).astype({f: float for f in LUNG})
    K = K[K.label.notna()]; yK = K.label.map({c: i for i, c in enumerate(cls)}).values
    m['external'] = {'data': 'KAUH adults, Littmann 3200 diaphragm mode — model frozen', **evaluate(yK, model.proba(K[LUNG].values), cls, m['thresholds'])}
    models.append(m)
    # ---- lung: child diagnosis group; plus the release-only confound baseline
    grp = {'Pneumonia (non-severe)': 'pneumonia', 'Pneumonia (severe)': 'pneumonia', 'Bronchitis': 'bronchitis', 'Asthma': 'asthma', 'Control Group': 'healthy'}
    D = Sd.assign(label=Sd.dx.map(grp))
    m, model, te, P = train('lung_pattern_child', D, LUNG, ['pneumonia', 'bronchitis', 'asthma', 'healthy'], rng, {
        'title': 'Lung pattern — children', 'kind': 'lung', 'ageRange': [0, 18], 'ageGate': True, 'data': 'SPRSound — children whose diagnosis is pneumonia, bronchitis, asthma or healthy control',
        'question': 'Which diagnosis group do children with recordings like this one belong to?'})
    trD = D[D.label.notna() & ~D.pid.isin(set(te.pid))]
    rel_major = trD.groupby('release').label.agg(lambda s: s.mode().iloc[0])
    cls = m['classes']; ytrue = te.y.values; yrel = te.release.map(rel_major).map({c: i for i, c in enumerate(cls)}).values
    m['confound'] = {'what': 'Release-only baseline: predict the most common diagnosis of the child’s recording batch, ignoring the sound', **prop(int((yrel == ytrue).sum()), len(ytrue))}
    models.append(m)
    # ---- lung: adults (pilot)
    kg = {'Healthy adult (control)': 'healthy', 'Asthma': 'asthma', 'Heart failure': 'heart failure', 'Heart failure + COPD': 'heart failure', 'Heart failure + lung fibrosis': 'heart failure'}
    A = pd.DataFrame([{**{f: r.get(f) for f in LUNG}, 'pid': r['pid'], 'label': kg.get(r['disease'])} for r in (json.loads(l) for l in open(FEATURES / 'kauh.jsonl'))]).astype({f: float for f in LUNG})
    m, *_ = train('lung_pattern_adult', A, LUNG, ['healthy', 'asthma', 'heart failure'], rng, {
        'title': 'Lung pattern — adults (pilot)', 'kind': 'lung', 'ageRange': [12, 90], 'ageGate': True, 'pilot': True, 'data': 'KAUH — adolescents and adults, Jordan, Littmann 3200 (all three filter modes)',
        'question': 'Which group — healthy, asthma or heart failure — do adults with recordings like this one belong to?'})
    models.append(m)
    for m in models:
        m['gate'] = gate(m)
        m['override'] = {'on': True, 'by': 'owner', 'date': '2026-11-11', 'note': 'switched on by the owner despite failing the gate; experimental'} if (m['id'] in OWNER_ON and not m['gate']['deploy']) else None
    out = {'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'seed': SEED, 'testFrac': TEST_FRAC,
           'engine': 'scikit-learn 1.9 models; held-out statistics with Seismometer 0.5.0 (Agresti–Coull, Hanley–McNeil)', 'models': models}
    # keep codec_check.js's robustness results across a retrain when every model is bit-identical (same seed + data)
    old = json.loads((WEB_DATA / 'models.json').read_text()) if (WEB_DATA / 'models.json').exists() else {}
    same = lambda: {m['id']: m['model'] for m in old.get('models', [])} == {m['id']: m['model'] for m in models}
    if old.get('robustness') and same(): out['robustness'] = old['robustness']
    elif old.get('robustness'): print('models changed: rerun node pipeline/codec_check.js')
    (WEB_DATA / 'models.json').write_text(json.dumps(out, separators=(',', ':'), ensure_ascii=False))
    for m in models:
        t = m['test']; a = t['accuracy']
        print(f"{m['id']:20} {m['chosen']:20} cv {m['cv']} | test {m['nTestPatients']} pts/{m['nTestRec']} rec: acc {a['value']} [{a['lo']}-{a['hi']}] (majority {m['majority']['value']}) macroAUC {t['macroAuc']}"
              + (f" | per patient acc {m['testPatient']['accuracy']['value']}" if 'testPatient' in m else '')
              + (f" | release-only {m['confound']['value']}" if 'confound' in m else '')
              + (f" | external acc {m['external']['accuracy']['value']} macroAUC {m['external']['macroAuc']}" if 'external' in m else ''))
        print('    gate:', m['gate'])
        if m['test'].get('flags'): print('    flags:', '; '.join(f"{f['class']} t={f['threshold']} sens {f['sens']['value']} spec {f['spec']['value']}" for f in m['test']['flags']))
        print('   ', '; '.join(f"{c['class']}: sens {c['sens']['value']} spec {c['spec']['value']} AUC {c['auc']['value'] if c['auc'] else '—'}" for c in t['perClass']))
    print(f"-> web/data/models.json ({(WEB_DATA / 'models.json').stat().st_size // 1024} KB)   (then merge.py --bump)")


if __name__ == '__main__':
    main()
