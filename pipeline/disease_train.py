"""Which disease? Training on every public respiratory-sound database available, separately for children and adults
→ meta/disease_model.json (read by the lite app's Disease page). Rerun daily by pipeline/daily_train.py.

Databases (licences checked; features = the full app's recordingFeatures, one row per recording, averaged per patient):
  children  SPRSound (CC BY 4.0, 860 children, 4 releases) · ICBHI 2017 children (research licence, statistics only)
  adults    KAUH (CC BY 4.0) · ICBHI 2017 adults · chronic bronchitis / emphysema, Mendeley (CC BY 4.0) · RespiratoryDatabase@TR (CC BY 4.0)
Diagnoses are harmonised into a few classes per population (CLASSES below). Recording length is NOT a feature (it is a
protocol difference between databases, not a sign of disease).

Rules fixed before the first run (rule 4 of the project):
  * unit: the patient (recordings averaged); folds are patient-grouped; gradient boosting with fixed settings, nothing tuned;
  * pooled 5-fold CV within each population → per-class AUC and macro AUC;
  * baselines with the same folds: database only (which collection the recording came from), age only, commonest class;
  * leave-one-database-out: train on the other databases, test on one, for each class present in both with ≥ 5 patients
    with and without it — the test of whether the model learned the disease or the collection;
  * a population's model "passes" only if macro AUC ≥ 0.70, it beats the database-only and the age-only baselines, and
    every leave-one-database-out AUC it is scored on has a 95 % CI above 0.5. Nothing is shown as a diagnosis either way.
  * the batch patients get their out-of-fold probabilities (they were held out in their fold).

  python3 pipeline/disease_train.py
"""
import json, datetime, collections
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from paths import META, FEATURES
from train_models import auc_ci

SEED = 20261123
F = ['f50', 'f75', 'cen', 'rr', 'hr', 'crkRate', 'fineRate', 'tonFrac', 'ie']
CLASSES = {'children': ['Healthy', 'Pneumonia', 'Bronchitis / bronchiolitis', 'Asthma', 'Other'],
           'adults': ['Healthy', 'Asthma', 'COPD', 'Heart failure', 'Pneumonia', 'Other']}
SPR = {'Control Group': 'Healthy', 'Pneumonia (non-severe)': 'Pneumonia', 'Pneumonia (severe)': 'Pneumonia', 'Bronchitis': 'Bronchitis / bronchiolitis',
       'Bronchiolitis': 'Bronchitis / bronchiolitis', 'Protracted bacterial bronchitis': 'Bronchitis / bronchiolitis', 'Asthma': 'Asthma'}
ICB = {'Healthy': 'Healthy', 'Pneumonia': 'Pneumonia', 'Bronchiolitis': 'Bronchitis / bronchiolitis', 'LRTI': 'Other', 'URTI': 'Other', 'COPD': 'COPD', 'Asthma': 'Asthma', 'Bronchiectasis': 'Other'}
KAU = {'Healthy adult (control)': 'Healthy', 'Asthma': 'Asthma', 'COPD': 'COPD', 'Heart failure': 'Heart failure', 'Pneumonia (adult)': 'Pneumonia'}
DEVICE = {'SPRSound': 'Yunting model II', 'KAUH': 'Littmann 3200', 'CB/emphysema': 'Littmann 3200', 'RespiratoryDatabase@TR': 'Littmann 3200', 'ICBHI': 'four devices'}


def rows():
    R = []
    for f in ('spr.0.jsonl', 'spr.1.jsonl'):
        for l in open(FEATURES / f):
            r = json.loads(l)
            if r.get('disease') is None or r.get('pid') in (None, '0'): continue
            R.append({'db': 'SPRSound', 'pid': 'S' + r['pid'], 'pop': 'children', 'age': r.get('age'), 'cls': SPR.get(r['disease'], 'Other'), **{k: r.get(k) for k in F}})
    I = {p['pid']: p for p in json.load(open(META / 'icbhi_index.json'))['patients']}
    for l in open(FEATURES / 'icbhi.jsonl'):
        r = json.loads(l); p = I.get(r['pid'])
        if not p or p['age'] is None: continue
        R.append({'db': 'ICBHI', 'pid': 'I' + r['pid'], 'pop': 'children' if p['age'] < 18 else 'adults', 'age': p['age'], 'cls': ICB.get(p['dx'], 'Other'), **{k: r['rf'].get(k) for k in F}})
    for l in open(FEATURES / 'kauh.jsonl'):
        r = json.loads(l)
        if r['mode'] != 'D': continue          # one filter mode per subject (diaphragm), as in the KAUH external test
        R.append({'db': 'KAUH', 'pid': 'K' + r['pid'], 'pop': 'adults', 'age': r['age'], 'cls': KAU.get(r['disease'], 'Other'), 'key': r['key'], **{k: r.get(k) for k in F}})
    for l in open(FEATURES / 'cbemph.jsonl'):
        r = json.loads(l)                       # no patient id in the release: each recording counted as its own patient (stated as a limit)
        R.append({'db': 'CB/emphysema', 'pid': 'C' + r['key'], 'pop': 'adults', 'age': None, 'cls': 'COPD', **{k: r.get(k) for k in F}})
    gold = {p['pid']: p['gold'] for p in json.load(open(META / 'respdbtr_validation.json'))['perPatient']}
    for l in open(FEATURES / 'respdbtr.jsonl'):
        r = json.loads(l)
        if gold.get(r['pid'], 0) == 0 or 'rf' not in r: continue      # GOLD 0 = at risk, not COPD; excluded
        R.append({'db': 'RespiratoryDatabase@TR', 'pid': 'T' + r['pid'], 'pop': 'adults', 'age': None, 'cls': 'COPD', 'key': r['pid'], **{k: r['rf'].get(k) for k in F}})
    D = pd.DataFrame(R)
    for k in F: D[k] = pd.to_numeric(D[k], errors='coerce')
    P = D.groupby('pid').agg({**{k: 'mean' for k in F}, 'db': 'first', 'pop': 'first', 'age': 'first', 'cls': 'first', 'key': 'first'}).reset_index()
    P['n_rec'] = D.groupby('pid').size().values
    return P


REFS = [{'pmid': '40249944', 'doi': '10.2196/66491', 'cite': 'Park JS et al. Artificial intelligence models for pediatric lung sound analysis: systematic review and meta-analysis. J Med Internet Res 2025',
         'why': '41 studies; high accuracy for wheeze and abnormal-sound detection, but most use small single-centre datasets with little external validation'},
        {'pmid': '36070274', 'doi': '10.1109/TBCAS.2022.3204910', 'cite': 'Zhang Q et al. SPRSound: open-source SJTU paediatric respiratory sound database. IEEE Trans Biomed Circuits Syst 2022',
         'why': 'the children\'s database here; its own record-level classification scores were 38–62 %'},
        {'pmid': '30708353', 'doi': '10.1088/1361-6579/ab03ea', 'cite': 'Rocha BM et al. An open access database for the evaluation of respiratory sound classification algorithms. Physiol Meas 2019',
         'why': 'ICBHI 2017: four devices, and diagnosis tracks age and device'},
        {'pmid': '24552321', 'doi': '10.1056/NEJMra1302901', 'cite': 'Bohadana A, Izbicki G, Kraman SS. Fundamentals of lung auscultation. N Engl J Med 2014',
         'why': 'crackles and wheeze are signs shared by several diseases, not specific to one'}]


def gb(): return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=SEED)


def oof(X, y, groups, make, K):
    P = np.zeros((len(y), K)); sk = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for a, b in sk.split(X, y, groups):
        m = make().fit(X[a], y[a]); pr = np.zeros((len(b), K)); pr[:, m.classes_] = m.predict_proba(X[b]); P[b] = pr
    return P


def aucs(y, P, classes):
    per = []
    for k, c in enumerate(classes):
        yk = (y == k).astype(int)
        per.append({'class': c, 'n': int(yk.sum()), 'auc': auc_ci(yk, P[:, k]) if 0 < yk.sum() < len(yk) else None})
    ok = [p['auc']['value'] for p in per if p['auc']]
    return per, round(float(np.mean(ok)), 3) if ok else None


def population(P, pop):
    S = P[P['pop'] == pop].reset_index(drop=True); cls = [c for c in CLASSES[pop] if (S['cls'] == c).sum() >= 5]
    S = S[S['cls'].isin(cls)].reset_index(drop=True); y = S['cls'].map({c: i for i, c in enumerate(cls)}).values; K = len(cls); g = S['pid'].values
    X = S[F].values.astype(float); Pm = oof(X, y, g, gb, K)
    dbX = pd.get_dummies(S['db']).values.astype(float); Pd = oof(dbX, y, g, lambda: LogisticRegression(max_iter=2000), K)
    ageX = S[['age']].fillna(S['age'].median() if S['age'].notna().any() else 0).values.astype(float); Pa = oof(ageX, y, g, gb, K)
    per, macro = aucs(y, Pm, cls); _, macroDb = aucs(y, Pd, cls); _, macroAge = aucs(y, Pa, cls)
    major = round(float(np.max(np.bincount(y)) / len(y)), 3)
    # leave one database out
    lodo = []
    for db in sorted(S['db'].unique()):
        te = (S['db'] == db).values; tr = ~te
        if len(set(y[tr])) < 2: continue
        m = gb().fit(X[tr], y[tr]); pr = np.zeros((te.sum(), K)); pr[:, m.classes_] = m.predict_proba(X[te])
        for k, c in enumerate(cls):
            yk = (y[te] == k).astype(int)
            if k in m.classes_ and yk.sum() >= 5 and (1 - yk).sum() >= 5:
                lodo.append({'heldOut': db, 'class': c, 'n': int(te.sum()), 'pos': int(yk.sum()), 'auc': auc_ci(yk, pr[:, k])})
    passed = bool(macro and macro >= 0.70 and macro > (macroDb or 0) and macro > (macroAge or 0) and lodo and all(l['auc']['lo'] > 0.5 for l in lodo))
    inv = [{'db': db, 'patients': int((S['db'] == db).sum()), 'recordings': int(S.loc[S['db'] == db, 'n_rec'].sum()), 'device': DEVICE.get(db, ''),
            'classes': dict(collections.Counter(S.loc[S['db'] == db, 'cls']))} for db in sorted(S['db'].unique())]
    conc = [{'class': c, 'patients': int((S['cls'] == c).sum()), 'databases': int(S.loc[S['cls'] == c, 'db'].nunique()),
             'largestShare': round(float(S.loc[S['cls'] == c, 'db'].value_counts(normalize=True).iloc[0]), 3), 'largest': S.loc[S['cls'] == c, 'db'].value_counts().index[0]} for c in cls]
    batch = {S.loc[i, 'key']: {c: round(float(Pm[i, k]), 3) for k, c in enumerate(cls)} for i in range(len(S)) if isinstance(S.loc[i, 'key'], str)}
    return {'classes': cls, 'patients': int(len(S)), 'perClass': per, 'macroAuc': macro, 'baselines': {'databaseOnly': macroDb, 'ageOnly': macroAge, 'commonestClassShare': major},
            'leaveOneDatabaseOut': lodo, 'passed': passed, 'inventory': inv, 'classByDatabase': conc, 'batchOof': batch}


def main():
    P = rows(); out = {'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'seed': SEED, 'features': F, 'refs': REFS, 'populations': {}}
    for pop in ('children', 'adults'):
        r = population(P, pop); out['populations'][pop] = r
        print(f"\n{pop}: {r['patients']} patients, classes {r['classes']}"); print('  macro AUC', r['macroAuc'], '| database-only', r['baselines']['databaseOnly'], '| age-only', r['baselines']['ageOnly'], '| passed', r['passed'])
        for p in r['perClass']: print('   ', p['class'], p['n'], p['auc'])
        for l in r['leaveOneDatabaseOut']: print('   LODO', l['heldOut'], l['class'], l['pos'], '/', l['n'], l['auc'])
    (META / 'disease_model.json').write_text(json.dumps(out, indent=1, default=str)); print('->', META / 'disease_model.json')


if __name__ == '__main__':
    main()
