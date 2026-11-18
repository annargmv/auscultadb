"""Refit the murmur rule's cut-off on CirCor TRAINING patients, report on the held-out ones → meta/murmur_refit.json.

The shipped rule flags a valve when the systolic : diastolic energy ratio (sysDia) is ≥ 2.5×. The cut-off is refitted on the
patients outside models.json heart_murmur testPids (Youden point and the 90 %-specificity point) and scored on the held-out
patients, against the shipped 2.5× on the same recordings (open task 1, rule 4: fit on one split, report on another).
The page keeps the shipped rule for its precomputed findings; the spectrogram network replaced it for Analyze and Batch.

  python3 pipeline/murmur_refit.py
"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from paths import META, FEATURES, WEB_DATA
from train_models import prop, auc_ci

R = [json.loads(l) for l in open(FEATURES / 'circor_ext.jsonl')]
R = [r for r in R if 'err' not in r and r.get('siteTruth') in ('Murmur', 'Normal') and r.get('sysDia') is not None]
test = set(next(m for m in json.loads((WEB_DATA / 'models.json').read_text())['models'] if m['id'] == 'heart_murmur')['testPids'])
tr = [r for r in R if r['pid'] not in test]; te = [r for r in R if r['pid'] in test]
x = lambda S: np.array([r['sysDia'] for r in S]); y = lambda S: np.array([r['siteTruth'] == 'Murmur' for r in S])
xt, yt = x(tr), y(tr); cands = np.unique(np.round(xt, 3))
J = [(np.mean(xt[yt] >= c) + np.mean(xt[~yt] < c) - 1, c) for c in cands]; youden = max(J)[1]
spec90 = float(np.quantile(xt[~yt], 0.90))


def score(c):
    X, Y = x(te), y(te); f = X >= c
    return {'cut': round(float(c), 3), 'sens': prop(int((f & Y).sum()), int(Y.sum())), 'spec': prop(int((~f & ~Y).sum()), int((~Y).sum()))}


out = {'note': 'sysDia ratio alone — the shipped finding also requires resolved beats and a plausible heart rate, so its sensitivity is lower',
       'what': 'sysDia cut-off refitted on training patients, scored on held-out valve recordings', 'trainRecordings': len(tr), 'testRecordings': len(te), 'testPatients': len(test),
       'auc': auc_ci(y(te).astype(int), (np.argsort(np.argsort(x(te))) + 0.5) / len(te)),   # ranks: auc_ci clips scores to 0–1
       'shipped': score(2.5), 'youden': score(youden), 'spec90': score(spec90)}
json.dump(out, open(META / 'murmur_refit.json', 'w'), indent=1)
for k in ('shipped', 'youden', 'spec90'): print(k, out[k]['cut'], 'sens', out[k]['sens']['value'], 'spec', out[k]['spec']['value'])
print('held-out sysDia AUC', out['auc'])
