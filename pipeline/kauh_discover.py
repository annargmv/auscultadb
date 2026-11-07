# Adult / elderly analyses on KAUH (Littmann 3200) → merged into data/discover.json as D['kauh']
import json, collections, math, sys
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from paths import KAUH_FEATURES, SPR_FEATURES, WEB_DATA
R = [json.loads(l) for l in open(KAUH_FEATURES)]
def mwu(a, b):
    a, b = [x for x in a if x is not None and not math.isnan(x)], [x for x in b if x is not None and not math.isnan(x)]
    if len(a) < 4 or len(b) < 4: return None
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    return {'nA': len(a), 'nB': len(b), 'medA': float(np.median(a)), 'medB': float(np.median(b)), 'meanA': float(np.mean(a)), 'meanB': float(np.mean(b)), 'p': float(p), 'r': float(1 - 2 * u / (len(a) * len(b)))}
D = [r for r in R if r['mode'] == 'D']
healthy = [r for r in D if r['label'] == 'Normal']
out = {'n': {'subjects': len(D), 'healthy': len(healthy), 'recordings': len(R), 'hours': round(sum(r['dur'] for r in R) / 3600, 2)}}
# ---- 1 lifespan: centroid vs age, healthy only, per device
BINS = [(12, 30, '12–29 y'), (30, 45, '30–44 y'), (45, 60, '45–59 y'), (60, 100, '60+ y')]
def curve(rows, key='cen'):
    c = []
    for lo, hi, lab in BINS:
        v = [r[key] for r in rows if lo <= r['age'] < hi and r[key] is not None]
        if v: c.append({'bin': lab, 'n': len(v), 'p25': float(np.percentile(v, 25)), 'p50': float(np.median(v)), 'p75': float(np.percentile(v, 75))})
    return c
rho, p = stats.spearmanr([r['age'] for r in healthy], [r['cen'] for r in healthy])
out['life'] = {'curve': curve(healthy), 'scatter': [[r['age'], r['cen']] for r in healthy], 'rho': float(rho), 'p': float(p), 'n': len(healthy),
               'byMode': {m: curve([r for r in R if r['mode'] == m and r['label'] == 'Normal']) for m in 'BDE'}}
# ---- 2 adult disease signatures (diaphragm mode) vs healthy adults
GROUPS = [('Asthma', lambda d: 'Asthma' in d), ('Heart failure (any)', lambda d: 'Heart failure' in d), ('COPD (any)', lambda d: 'COPD' in d),
          ('Pneumonia', lambda d: 'Pneumonia' in d), ('Lung fibrosis (any)', lambda d: 'fibrosis' in d)]
FEATS = [('cen', 'Spectral centroid (Hz)'), ('crkRate', 'Crackle-like transients /s'), ('fineRate', 'Fine-crackle-like /s'), ('tonFrac', 'Tonal (wheeze-like) time fraction')]
dis = []
for g, f in GROUPS:
    G = [r for r in D if f(r['disease'])]
    lab = collections.Counter(r['label'] for r in G)
    row = {'dx': g, 'n': len(G), 'age': float(np.median([r['age'] for r in G])), 'wheeze': sum(v for k, v in lab.items() if 'CAS' in k) / len(G), 'crackles': sum(v for k, v in lab.items() if 'DAS' in k) / len(G), 'f': {}}
    for k, _ in FEATS: row['f'][k] = mwu([r[k] for r in healthy], [r[k] for r in G])
    dis.append(row)
out['disease'] = {'rows': dis, 'feats': FEATS, 'healthyAge': float(np.median([r['age'] for r in healthy]))}
# ---- 3 heart failure vs asthma ("cardiac asthma"): can sound alone tell them apart?
def probe(rows, y, cols, reps=30):
    X = np.array([[r[c] if r[c] is not None else 0 for c in cols] for r in rows]); a = []
    for s in range(reps):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000))
        pr = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=s), method='predict_proba')[:, 1]; a.append(roc_auc_score(y, pr))
    return float(np.mean(a)), float(np.std(a))
HA = [r for r in D if r['disease'] in ('Asthma', 'Heart failure')]; y = np.array([1 if r['disease'] == 'Heart failure' else 0 for r in HA])
S = ['cen', 'f50', 'f75', 'crkRate', 'fineRate', 'tonFrac']
# use all three filter modes as extra features of the same recording
byKey = collections.defaultdict(dict)
for r in R: byKey[r['key']][r['mode']] = r
def multi(r): return {f'{m}_{c}': byKey[r['key']][m][c] for m in 'BDE' for c in S} | {'age': r['age']}
HA3 = [multi(r) for r in HA]
out['hfAsthma'] = {'nHF': int(y.sum()), 'nAsthma': int(len(y) - y.sum()), 'ageHF': float(np.median([r['age'] for r in HA if r['disease'] == 'Heart failure'])), 'ageAsthma': float(np.median([r['age'] for r in HA if r['disease'] == 'Asthma'])),
                   'age': probe(HA, y, ['age']), 'sound': probe(HA, y, S), 'sound3': probe(HA3, y, [f'{m}_{c}' for m in 'BDE' for c in S]), 'sound3age': probe(HA3, y, [f'{m}_{c}' for m in 'BDE' for c in S] + ['age']),
                   'labels': {'HF': dict(collections.Counter(r['label'] for r in HA if r['disease'] == 'Heart failure')), 'Asthma': dict(collections.Counter(r['label'] for r in HA if r['disease'] == 'Asthma'))}}
# ---- 4 filter modes: same recording, three filters
modes = {}
for m in 'BDE':
    Rm = [r for r in R if r['mode'] == m]
    modes[m] = {'cen': float(np.median([r['cen'] for r in Rm])), 'crk': float(np.mean([r['crkRate'] for r in Rm])), 'ton': float(np.mean([r['tonFrac'] for r in Rm])),
                'rrOk': sum(1 for r in Rm if r['rr'] and not r['est']) / len(Rm)}
def icc1(groups):
    groups = [g for g in groups if len(g) >= 2]; k = np.mean([len(g) for g in groups]); allv = np.concatenate(groups); gm = allv.mean()
    msb = sum(len(g) * (np.mean(g) - gm) ** 2 for g in groups) / (len(groups) - 1); msw = sum(((np.array(g) - np.mean(g)) ** 2).sum() for g in groups) / (len(allv) - len(groups))
    return float((msb - msw) / (msb + (k - 1) * msw))
rrG = [[byKey[k][m]['rr'] for m in 'BDE' if byKey[k][m]['rr'] and not byKey[k][m]['est']] for k in byKey]
hrG = [[byKey[k][m]['hr'] for m in 'BDE' if byKey[k][m]['hr']] for k in byKey]
out['modes'] = {'byMode': modes, 'iccRR': icc1(rrG), 'nRR': sum(1 for g in rrG if len(g) >= 2), 'iccHR': icc1(hrG), 'nHR': sum(1 for g in hrG if len(g) >= 2)}
# rates vs clip length (both datasets)
spr = [json.loads(l) for f in SPR_FEATURES for l in open(f)]
def okfrac(rows, lo, hi):
    v = [r for r in rows if lo <= r['dur'] < hi and 'err' not in r]
    return [sum(1 for r in v if r.get('rr') and not r.get('est')), len(v)]
out['rateByLen'] = [{'len': lab, 'spr': okfrac(spr, lo, hi), 'kauh': okfrac(D, lo, hi)} for lo, hi, lab in [(0, 10, '< 10 s'), (10, 16, '10–15 s'), (16, 21, '16–20 s'), (21, 31, '21–30 s')]]
Dj = json.load(open(WEB_DATA / 'discover.json')); Dj['kauh'] = out; json.dump(Dj, open(WEB_DATA / 'discover.json', 'w'), separators=(',', ':'))
print(json.dumps({k: out[k] for k in ('n', 'hfAsthma', 'modes', 'rateByLen')}, indent=0)[:2500])
print('life rho', out['life']['rho'], out['life']['p'], out['life']['curve'])
for r in dis: print(r['dx'], r['n'], round(r['age']), {k: (round(v['r'], 2), round(v['p'], 3)) if v else None for k, v in r['f'].items()})
