"""ICBHI 2017 → sound-type and illness models for "Analyze a recording" → web/data/icbhi_models.json.

Reads meta/icbhi_index.json + meta/features/icbhi.jsonl (pipeline/icbhi_prep.py, pipeline/icbhi_features.js: the page's
own analyzeSignal / cycleFeatures / recordingFeatures). ICBHI audio never reaches web/: this file holds only model
weights and validation numbers.

1. SOUND TYPE per breath cycle — normal / crackles / wheezes / both (the ICBHI challenge task).
   Patient-level split 60/40 (seed 20261116, stratified by diagnosis); logistic regression vs gradient boosting chosen
   by grouped 5-fold CV on the training patients only; flag thresholds = 90 % training out-of-fold specificity.
   Reported on the held-out patients: per-class AUC, the ICBHI score (mean of sensitivity over abnormal cycles and
   specificity over normal cycles), per device, a device-only baseline, the SHIPPED rule-based detectors on the same
   cycles (external: their thresholds predate ICBHI), and a deployment check — the page's own cycle finder on the
   held-out recordings (no expert cycle boundaries) against the experts' recording-level crackle/wheeze presence.

2. ILLNESS — one recording → which illness pattern it resembles. Rows are recordings, patients are the unit.
   The diagnosis in ICBHI is confounded by AGE (every healthy patient is a child, every COPD patient an adult) and by
   DEVICE (all children on one stethoscope). So there is no all-ages model. Two models, each inside one age group:
     children (< 18 y): Healthy / URTI / lower respiratory infection (bronchiolitis, LRTI, pneumonia)
     adults  (>= 18 y): COPD / other (bronchiectasis, pneumonia, asthma)
   Fixed beforehand: logistic regression (C = 1, standardised), sound features + age, each patient weighted equally.
   Too few patients for a held-out split, so every patient is scored by models fitted WITHOUT that patient
   (patient-grouped 5-fold CV, repeated 10 times, probabilities averaged); the sound-model inputs are cross-fitted. Baselines on the same folds: age
   only, device only, commonest class. Same deployment gate as the other Analyze models; failures are kept off.

  python3 pipeline/train_icbhi.py      (then python3 pipeline/clinical_validation.py && python3 pipeline/merge.py --bump)
"""
import json, math, datetime, collections, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import roc_auc_score
from paths import META, FEATURES, WEB_DATA
import train_models as TM
from train_models import LR, GB, prop, auc_ci, macro_auc, evaluate, thresholds

warnings.filterwarnings('ignore')
SEED, TEST_FRAC = 20261116, 0.4
CYC = ['dur', 'b0med', 'b0p90', 'b0rng', 'b1med', 'b1p90', 'b1rng', 'b2med', 'b2p90', 'b2rng', 'b3med', 'b3p90', 'b3rng', 'cen', 'f50', 'flat',
       'tonP90', 'tonMax', 'tonFrac', 'tonSeg', 'tonHz', 'poly', 'trRate', 'crkRate', 'fineRate', 'trHl', 'trDur']   # = CYC_FEATS in the page
MEL = [f'm{i}' for i in range(20)] + [f's{i}' for i in range(20)]   # = CYC_MEL in the page
SOUND = ['Normal', 'Crackles', 'Wheezes', 'Both']
REC = ['f50', 'cen', 'crkRate', 'fineRate', 'tonFrac', 'rr', 'ie']
AGG = ['pCrk', 'pWhz', 'fCrk', 'fWhz']
CHILD = {'Healthy': 'Healthy', 'URTI': 'URTI', 'Bronchiolitis': 'Lower respiratory infection', 'LRTI': 'Lower respiratory infection', 'Pneumonia': 'Lower respiratory infection'}
ADULT = {'COPD': 'COPD', 'Bronchiectasis': 'Other lung disease', 'Pneumonia': 'Other lung disease', 'Asthma': 'Other lung disease'}


def num(v): return np.nan if v is None else float(v)


def load():
    I = json.load(open(META / 'icbhi_index.json')); P = {p['pid']: p for p in I['patients']}
    R = [json.loads(l) for l in open(FEATURES / 'icbhi.jsonl')]; R = [r for r in R if 'err' not in r]
    cyc, sc, rec = [], [], []
    for r in R:
        p = P[r['pid']]
        for e in r['exp']:
            if e['f'] is None: continue
            cyc.append({'pid': r['pid'], 'name': r['name'], 'dev': r['dev'], 'dx': p['dx'], 'y': int(e['c']) + 2 * int(e['w']) if True else 0,
                        'ruleC': e['rule']['crk'], 'ruleW': e['rule']['whz'], **{k: num(e['f'].get(k)) for k in CYC + MEL}})
        for e in r['sc']:
            if e['f'] is None: continue
            sc.append({'pid': r['pid'], 'name': r['name'], 'src': e['src'], **{k: num(e['f'].get(k)) for k in CYC + MEL}})
        rec.append({'pid': r['pid'], 'name': r['name'], 'dev': r['dev'], 'dx': p['dx'], 'age': num(p['age']), 'group': p['group'],
                    'anyC': int(any(e['c'] for e in r['exp'])), 'anyW': int(any(e['w'] for e in r['exp'])), **{k: num(r['rf'].get(k)) for k in REC}})
    # y: 0 normal, 1 crackles, 2 wheezes, 3 both
    return I, pd.DataFrame(cyc), pd.DataFrame(sc), pd.DataFrame(rec)


def icbhi_score(y, pred):
    ab = y > 0; se = float(np.mean(pred[ab] == y[ab])) if ab.any() else None; sp = float(np.mean(pred[~ab] == 0)) if (~ab).any() else None
    return {'se': round(100 * se, 1), 'sp': round(100 * sp, 1), 'score': round(50 * (se + sp), 1), 'nAbnormal': int(ab.sum()), 'nNormal': int((~ab).sum())}


def patient_split(rec, rng):
    pl = rec.groupby('pid').dx.first(); test = set()
    for d in sorted(pl.unique()):
        ids = sorted(pl[pl == d].index); rng.shuffle(ids); test |= set(ids[:int(round(len(ids) * TEST_FRAC))])
    return test


def sound_model(cyc, sc, rec, test):
    tr = cyc[~cyc.pid.isin(test)].reset_index(drop=True); te = cyc[cyc.pid.isin(test)].reset_index(drop=True)
    tr['y'] = tr.y.astype(int); te['y'] = te.y.astype(int); K = 4
    fsets = {'cycle': CYC, 'cycle + mel spectrum': CYC + MEL}
    fs, C, cvres, oof = TM.choose(tr, fsets, K); F = fsets[fs]
    thr = thresholds(tr.y.values, oof, K)
    m = C().fit(tr[F].values, tr.y.values); P = m.proba(te[F].values); pred = P.argmax(1); y = te.y.values
    out = {'id': 'icbhi_sound', 'kind': 'lung', 'unit': 'cycle', 'title': 'Adventitious sound per breath cycle (ICBHI-trained)', 'classes': SOUND, 'features': F, 'featureSet': fs,
           'model': m.export(), 'chosen': C.name, 'cv': cvres, 'thresholds': thr, 'nTrainPatients': int(tr.pid.nunique()), 'nTestPatients': int(te.pid.nunique()),
           'nTrainCycles': len(tr), 'nTestCycles': len(te), 'test': evaluate(y, P, SOUND, thr), 'icbhi': icbhi_score(y, pred),
           'majority': prop(int((y == 0).sum()), len(y)),
           'range': {f: [None if np.isnan(v) else round(float(v), 4) for v in np.nanpercentile(tr[f].astype(float), [1, 99])] for f in F},
           'parity': [{'x': [None if math.isnan(v) else float(v) for v in te.loc[i, F]], 'p': [round(float(q), 9) for q in P[i]]} for i in range(0, len(te), len(te) // 25)][:25]}
    # crackle / wheeze as two yes/no questions (what a clinician asks): P(crackle) = P(C) + P(B), P(wheeze) = P(W) + P(B)
    pc, pw = P[:, 1] + P[:, 3], P[:, 2] + P[:, 3]; yc, yw = np.isin(y, [1, 3]), np.isin(y, [2, 3])
    out['binary'] = {'crackles': auc_ci(yc.astype(int), pc), 'wheezes': auc_ci(yw.astype(int), pw)}
    # shipped rule-based detectors on the same held-out cycles (external — their thresholds were fixed on other data)
    rp = (te.ruleC.values + 2 * te.ruleW.values).astype(int)
    def bin_(flag, truth):
        tp = int((flag & truth).sum()); fn = int((~flag & truth).sum()); fp = int((flag & ~truth).sum()); tn = int((~flag & ~truth).sum())
        return {'sens': prop(tp, tp + fn), 'spec': prop(tn, tn + fp)}
    out['rules'] = {'icbhi': icbhi_score(y, rp), 'crackles': bin_(te.ruleC.values.astype(bool), yc), 'wheezes': bin_(te.ruleW.values.astype(bool), yw),
                    'model': {'crackles': bin_(pc >= 0.5, yc), 'wheezes': bin_(pw >= 0.5, yw)}}
    # device-only baseline (same split): how much of the answer is the stethoscope?
    D = pd.get_dummies(cyc.dev).astype(float); Dtr, Dte = D[~cyc.pid.isin(test)].values, D[cyc.pid.isin(test)].values
    Pd = LogisticRegression(max_iter=2000).fit(Dtr, tr.y.values).predict_proba(Dte)
    out['deviceOnly'] = {'macroAuc': round(macro_auc(y, Pd, K), 3), 'icbhi': icbhi_score(y, Pd.argmax(1))}
    out['byDevice'] = []
    for d in sorted(te.dev.unique()):
        k = (te.dev == d).values
        if k.sum() >= 50 and len(set(y[k])) >= 2:
            out['byDevice'].append({'device': d, 'n': int(k.sum()), 'patients': int(te.pid[k].nunique()), 'icbhi': icbhi_score(y[k], pred[k]),
                                    'crackleAuc': auc_ci(yc[k].astype(int), pc[k]), 'wheezeAuc': auc_ci(yw[k].astype(int), pw[k])})
    # deployment check: the page's own cycles on held-out recordings → recording-level presence vs experts
    s = sc[sc.pid.isin(test)].reset_index(drop=True); Ps = m.proba(s[F].values); s['pc'] = Ps[:, 1] + Ps[:, 3]; s['pw'] = Ps[:, 2] + Ps[:, 3]
    g = s.groupby('name').agg(pc=('pc', 'max'), pw=('pw', 'max'), n=('pc', 'size'), win=('src', lambda v: float(np.mean(v == 'window'))))
    rr = rec.set_index('name').loc[g.index]
    out['deploy'] = {'recordings': int(len(g)), 'windowFallback': round(100 * float(g.win.mean()), 1),
                     'crackles': auc_ci(rr.anyC.values, g.pc.values), 'wheezes': auc_ci(rr.anyW.values, g.pw.values),
                     'what': 'Held-out recordings scored on the cycles the page finds itself (no expert boundaries); highest cycle probability vs the experts saying the recording has any crackle / wheeze cycle'}
    return out, C, F


def cross_fit_sound(cyc, sc, C, F):
    """Sound-model probabilities for every page-found cycle from a model that never saw that patient (5-fold, by patient)."""
    sc = sc.copy(); sc['pc'] = np.nan; sc['pw'] = np.nan
    for a, b in GroupKFold(5).split(cyc, groups=cyc.pid):
        held = set(cyc.pid.iloc[b]); m = C().fit(cyc.iloc[a][F].values, cyc.iloc[a].y.astype(int).values)
        k = sc.pid.isin(held).values
        if k.any(): P = m.proba(sc.loc[k, F].values); sc.loc[k, 'pc'] = P[:, 1] + P[:, 3]; sc.loc[k, 'pw'] = P[:, 2] + P[:, 3]
    return sc


def agg(sc, thr_c=0.5, thr_w=0.5):
    return sc.groupby('name').agg(pCrk=('pc', 'mean'), pWhz=('pw', 'mean'), fCrk=('pc', lambda v: float(np.mean(v >= thr_c))), fWhz=('pw', lambda v: float(np.mean(v >= thr_w))))


class WLR:
    """Standardised logistic regression, C = 1, with per-row weights (each patient counts once)."""
    def fit(self, X, y, w=None):
        self.imp = SimpleImputer(strategy='median').fit(X); Z = self.imp.transform(X); self.sc = StandardScaler().fit(Z)
        self.m = LogisticRegression(C=1.0, max_iter=5000).fit(self.sc.transform(Z), y, sample_weight=w); return self
    def proba(self, X): return self.m.predict_proba(self.sc.transform(self.imp.transform(X)))
    export = LR.export


def illness(task, title, df, classes, feats, devices):
    df = df.reset_index(drop=True).copy(); df['y'] = df.label.map({c: i for i, c in enumerate(classes)}).astype(int); K = len(classes)
    w = 1.0 / df.groupby('pid').pid.transform('size').values
    pids = df.pid.unique(); oof = np.zeros((len(df), K)); oa = np.zeros((len(df), K)); od = np.zeros((len(df), K)); ok = np.zeros((len(df), K))
    D = pd.get_dummies(df.dev).astype(float).values
    # Repeated patient-grouped 5-fold CV (10 repeats, probabilities averaged). Leave-one-patient-out was used first and
    # dropped: removing a patient shifts the class balance against that patient's class, which pushes base-rate
    # predictors (the age-only and stethoscope-only baselines) below chance and so flatters the model against them.
    snd = [f for f in feats if f != 'age']; cnt = np.zeros(len(df))
    for rep in range(10):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED + rep)
        for tr_i, te_i in cv.split(df, df.y, df.pid):
            tr = np.zeros(len(df), bool); tr[tr_i] = True; te = ~tr
            if len(set(df.y[tr])) < K: continue
            oof[te] += WLR().fit(df.loc[tr, feats].values, df.y[tr].values, w[tr]).proba(df.loc[te, feats].values)
            oa[te] += WLR().fit(df.loc[tr, ['age']].values, df.y[tr].values, w[tr]).proba(df.loc[te, ['age']].values)
            od[te] += WLR().fit(D[tr], df.y[tr].values, w[tr]).proba(D[te])
            ok[te] += WLR().fit(df.loc[tr, snd].values, df.y[tr].values, w[tr]).proba(df.loc[te, snd].values)
            cnt[te] += 1
    for P in (oof, oa, od, ok): P /= np.maximum(cnt, 1)[:, None]
    # patient level: average a patient's recordings (what the page would do with several sites)
    def per_patient(P):
        g = pd.DataFrame(P).groupby(df.pid.values).mean(); yp = df.groupby('pid').y.first().loc[g.index].values; return yp, g.values
    yp, Pp = per_patient(oof); ya, Pa = per_patient(oa); yd, Pd = per_patient(od); ys, Ps = per_patient(ok)
    test = evaluate(df.y.values, oof, classes); thr = thresholds(df.y.values, oof, K)
    test = evaluate(df.y.values, oof, classes, thr)
    major = df.groupby('pid').y.first().value_counts().idxmax()
    out = {'id': task, 'kind': 'lung', 'unit': 'recording', 'title': title, 'classes': classes, 'features': feats, 'chosen': 'logistic regression (fixed beforehand)',
           'model': WLR().fit(df[feats].values, df.y.values, w).export(), 'thresholds': thr, 'scheme': 'patient-grouped 5-fold CV, 10 repeats',
           'nPatients': int(len(pids)), 'nRec': int(len(df)), 'perClassPatients': {c: int((df.groupby('pid').y.first() == i).sum()) for i, c in enumerate(classes)},
           'devices': devices, 'test': test, 'testPatient': evaluate(yp, Pp, classes, thr),
           'majority': prop(int((df.y == major).sum()), len(df)),
           'ageOnly': {'what': 'Age alone, same folds', 'macroAuc': round(macro_auc(df.y.values, oa, K), 3), 'accuracy': prop(int((oa.argmax(1) == df.y.values).sum()), len(df))},
           'deviceOnly': {'what': 'Stethoscope alone, same folds', 'macroAuc': round(macro_auc(df.y.values, od, K), 3), 'accuracy': prop(int((od.argmax(1) == df.y.values).sum()), len(df))} if df.dev.nunique() > 1
                         else {'what': 'One stethoscope only in this group: no stethoscope baseline', 'macroAuc': None, 'accuracy': None},
           'soundOnly': {'what': 'Sound features without age, same folds', 'macroAuc': round(macro_auc(df.y.values, ok, K), 3)},
           'range': {f: [None if np.isnan(v) else round(float(v), 4) for v in np.nanpercentile(df[f].astype(float), [1, 99])] for f in feats},
           'ageRange': [float(df.age.min()), float(df.age.max())]}
    why = []
    t = out['test']
    if t['macroAuc'] < TM.GATE_AUC: why.append(f"macro AUC {t['macroAuc']} < {TM.GATE_AUC}")
    bad = [c['class'] for c in t['perClass'] if not c['auc'] or c['auc']['lo'] <= 0.5]
    if bad: why.append('no better than chance for ' + ', '.join(bad))
    if t['accuracy']['value'] <= out['majority']['value']: why.append(f"accuracy {t['accuracy']['value']} % does not beat always answering the commonest class ({out['majority']['value']} %)")
    if t['macroAuc'] <= out['ageOnly']['macroAuc']: why.append(f"macro AUC {t['macroAuc']} does not beat age alone ({out['ageOnly']['macroAuc']})")
    if out['deviceOnly']['macroAuc'] is not None and t['macroAuc'] <= out['deviceOnly']['macroAuc']: why.append(f"macro AUC {t['macroAuc']} does not beat the stethoscope alone ({out['deviceOnly']['macroAuc']})")
    out['gate'] = {'deploy': not why, 'reasons': why}
    out['parity'] = [{'x': [None if (isinstance(v, float) and math.isnan(v)) else float(v) for v in df.loc[i, feats]],
                      'p': [round(float(q), 9) for q in WLR().fit(df[feats].values, df.y.values, w).proba(df.loc[[i], feats].values)[0]]} for i in range(0, len(df), max(1, len(df) // 10))][:10]
    return out


def main():
    rng = np.random.default_rng(SEED)
    I, cyc, sc, rec = load()
    cyc['y'] = cyc.y.astype(int)
    test = patient_split(rec, rng)
    snd, C, F = sound_model(cyc, sc, rec, test)
    t = snd['test']; print(f"SOUND  {snd['chosen']}  cycles train/test {snd['nTrainCycles']}/{snd['nTestCycles']}  macro AUC {t['macroAuc']}  ICBHI {snd['icbhi']}  device-only {snd['deviceOnly']}")
    print('  feature set', snd['featureSet'], snd['cv'])
    print('  per class AUC', {c['class']: c['auc'] for c in t['perClass']}, '\n  crackle/wheeze', snd['binary'], '\n  rules', snd['rules']['icbhi'], snd['rules']['crackles'], snd['rules']['wheezes'])
    print('  deploy', {k: v for k, v in snd['deploy'].items() if k != 'what'}, '\n  by device', [(d['device'], d['icbhi']['score']) for d in snd['byDevice']])
    scx = cross_fit_sound(cyc, sc, C, F); A = agg(scx)
    df = rec.set_index('name').join(A, how='inner').reset_index()
    models = [snd]
    ch = df[(df.group == 'child') & df.dx.isin(CHILD)].copy(); ch['label'] = ch.dx.map(CHILD)
    ad = df[(df.group == 'adult') & df.dx.isin(ADULT)].copy(); ad['label'] = ad.dx.map(ADULT)
    for task, title, d, cls in (('icbhi_child', 'Children: healthy / upper / lower respiratory infection (ICBHI)', ch, ['Healthy', 'URTI', 'Lower respiratory infection']),
                                ('icbhi_adult', 'Adults: COPD / other lung disease (ICBHI)', ad, ['COPD', 'Other lung disease'])):
        m = illness(task, title, d, cls, AGG + REC + ['age'], dict(collections.Counter(d.dev)))
        m['ageGate'] = [0, 18] if task == 'icbhi_child' else [18, 120]
        models.append(m)
        print(f"{task}: patients {m['nPatients']} {m['perClassPatients']}  macro AUC {m['test']['macroAuc']}  age-only {m['ageOnly']['macroAuc']}  device-only {m['deviceOnly']['macroAuc']}  sound-only {m['soundOnly']['macroAuc']}  gate {m['gate']}")
        print('   per class', {c['class']: c['auc'] for c in m['test']['perClass']}, ' patient-level', m['testPatient']['macroAuc'])
    conf = {'byDiagnosis': [{'dx': dx, 'patients': int(g.pid.nunique()), 'ageMedian': float(g.groupby('pid').age.first().median()) if g.age.notna().any() else None,
                             'devices': dict(collections.Counter(g.groupby('pid').dev.first()))} for dx, g in rec.groupby('dx')]}
    out = {'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'seed': SEED,
           'source': 'ICBHI 2017 Respiratory Sound Database — Rocha BM et al., An open access database for the evaluation of respiratory sound classification algorithms. Physiol Meas 2019;40:035001. doi:10.1088/1361-6579/ab03ea',
           'licence': 'Research use (ICBHI challenge). Analysed only — no ICBHI audio is published on this site.',
           'data': {'patients': len(I['patients']), 'recordings': int(len(rec)), 'cycles': int(len(cyc)), 'droppedRows': I.get('droppedRows', 0)},
           'testPids': sorted(test), 'confounds': conf, 'models': models}
    (WEB_DATA / 'icbhi_models.json').write_text(json.dumps(out, separators=(',', ':')))
    print('->', WEB_DATA / 'icbhi_models.json', (WEB_DATA / 'icbhi_models.json').stat().st_size // 1024, 'KB')


if __name__ == '__main__':
    main()
