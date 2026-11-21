"""External test of the breath-sound model on ADULT COPD patients — RespiratoryDatabase@TR → meta/respdbtr_validation.json.

Input: meta/features/respdbtr.jsonl (pipeline/respdbtr_run.js: the page's frozen lung_sound model and ICBHI cycle model on
all 12 channels of 42 patients). Labels: GOLD stage 0–4 per patient (Labels.xlsx; pulmonologists using spirometry, chest
X-ray and auscultation — Altan et al., Mendeley 10.17632/p9z4h98s6j.2, CC BY 4.0). Same stethoscope type as KAUH (Littmann 3200).

The labels are COPD severity, not sound types, so this asks one question, fixed BEFORE the first result:
  PRIMARY: does the model's abnormal-breath-sound probability (1 − P(normal)), averaged over a patient's 12 channels, rise
  with GOLD stage? Spearman ρ across the 42 patients, patient-bootstrap 95 % CI. "Tracks COPD severity" only if ρ > 0 and
  p < 0.05. Nothing else is claimed.
  DESCRIPTIVE (no claim, listed for completeness): the same for P(continuous: wheeze / rhonchi) and for the ICBHI cycle
  model's wheeze share; GOLD 0–1 vs GOLD 3–4 AUC of the primary score.

  python3 pipeline/respdbtr_validate.py
"""
import json, datetime, collections
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from paths import ROOT, META, FEATURES
from train_models import auc_ci

SEED = 20261121


def rho_ci(x, y, B=2000):
    rng = np.random.default_rng(SEED); v = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        if len(set(y[i])) > 1: v.append(stats.spearmanr(x[i], y[i])[0])
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def main():
    L = pd.read_excel(next((ROOT / 'raw' / 'respdbtr').rglob('Labels.xlsx')))
    gold = {r['Patient ID']: int(str(r['Diagnosis']).replace('COPD', '')) for _, r in L.iterrows()}
    R = [json.loads(l) for l in open(FEATURES / 'respdbtr.jsonl')]
    by = collections.defaultdict(list)
    for r in R: by[r['pid']].append(r)
    P = []
    for pid, rs in sorted(by.items()):
        if pid not in gold: continue
        P.append({'pid': pid, 'gold': gold[pid], 'channels': len(rs), 'pAbn': float(np.mean([1 - r['p'][0] for r in rs])),
                  'pCont': float(np.mean([r['p'][1] for r in rs])), 'wheezeShare': float(np.mean([r['wheezeShare'] for r in rs if r['wheezeShare'] is not None] or [np.nan])),
                  'flaggedAbn': int(sum(not r['flags'][0] for r in rs))})
    g = np.array([p['gold'] for p in P]); out = {'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'patients': len(P), 'channels': len(R),
           'byGold': {str(k): v for k, v in sorted(collections.Counter(g.tolist()).items())},
           'source': 'RespiratoryDatabase@TR (Altan, Kutlu et al.; Mendeley Data 10.17632/p9z4h98s6j.2), CC BY 4.0', 'zipSha256': 'a3d5dc16bbad9a020538ed85551bf773c206222aec2132aed75d02a4ee75ae55'}
    for key, lab in (('pAbn', 'abnormal-breath-sound probability (primary)'), ('pCont', 'wheeze / rhonchi probability'), ('wheezeShare', 'share of cycles with wheeze (ICBHI cycle model)')):
        x = np.array([p[key] for p in P]); ok = np.isfinite(x); rho, pv = stats.spearmanr(x[ok], g[ok])
        out[key] = {'what': lab, 'rho': round(float(rho), 3), 'p': float(pv), 'ci': rho_ci(x[ok], g[ok]), 'n': int(ok.sum()),
                    'medianByGold': {str(k): round(float(np.median(x[ok][g[ok] == k])), 3) for k in sorted(set(g[ok].tolist()))}}
    s = np.array([p['pAbn'] for p in P]); m = (g <= 1) | (g >= 3)
    out['gold01_vs_34'] = {'auc': auc_ci((g[m] >= 3).astype(int), s[m]), 'n01': int((g <= 1).sum()), 'n34': int((g >= 3).sum())}
    out['tracksSeverity'] = bool(out['pAbn']['rho'] > 0 and out['pAbn']['p'] < 0.05)
    out['perPatient'] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()} for p in P]
    (META / 'respdbtr_validation.json').write_text(json.dumps(out, indent=1))
    for k in ('pAbn', 'pCont', 'wheezeShare'): print(k, {q: out[k][q] for q in ('rho', 'p', 'ci', 'n', 'medianByGold')})
    print('GOLD 0–1 vs 3–4', out['gold01_vs_34'], '| tracks severity:', out['tracksSeverity'])


if __name__ == '__main__':
    main()
