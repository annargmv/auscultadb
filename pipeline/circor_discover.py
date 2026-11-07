"""Cardiac analyses on CirCor (meta/features/circor.jsonl) → merged into data/discover.json as D['circor'].

Run after `node pipeline/circor_validate.js`, which writes the per-recording rows this reads.
Like kauh_discover.py, this only adds a key to the existing discover.json.

The question it is built to answer, for the recalibration task: is the systolic-murmur rule limited
by its *threshold* or by its *feature*? If the systolic:diastolic energy ratio separates murmur from
clean valves well (high AUC) then the fixed cut is simply in the wrong place and re-fitting helps.
If the AUC is poor, no threshold can rescue it and the feature itself has to change. Nothing here
tunes the live detector — it only measures what the existing feature can and cannot do.
"""
import collections, json, math
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from paths import FEATURES, WEB_DATA

GRADES = ['I/VI', 'II/VI', 'III/VI']
TIMINGS = ['Early-systolic', 'Mid-systolic', 'Late-systolic', 'Holosystolic']
BANDS = ['Neonate', 'Infant', 'Child', 'Adolescent']
ok = lambda v: v is not None and not (isinstance(v, float) and math.isnan(v))


def mwu(a, b):
    a, b = [x for x in a if ok(x)], [x for x in b if ok(x)]
    if len(a) < 4 or len(b) < 4: return None
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    return {'nA': len(a), 'nB': len(b), 'medA': float(np.median(a)), 'medB': float(np.median(b)),
            'p': float(p), 'r': float(2 * u / (len(a) * len(b)) - 1)}


def rate(rows, hit):
    n = len(rows)
    return {'k': sum(1 for r in rows if hit(r)), 'n': n, 'pct': round(100 * sum(1 for r in rows if hit(r)) / n, 1) if n else None}


def main():
    R = [json.loads(l) for l in open(FEATURES / 'circor.jsonl')]
    R = [r for r in R if 'err' not in r]
    out = {'n': {'patients': len({r['key'] for r in R}), 'recordings': len(R),
                 'hours': round(sum(r['dur'] for r in R) / 3600, 2)}}

    # ---- 1 can the feature separate at all?  (threshold-free)
    lab = [r for r in R if r['siteTruth'] in ('Murmur', 'Normal') and ok(r['sysDia'])]
    y = np.array([1 if r['siteTruth'] == 'Murmur' else 0 for r in lab])
    x = np.array([r['sysDia'] for r in lab])
    auc = float(roc_auc_score(y, x)) if len(set(y)) == 2 else None
    best = None
    if auc is not None:
        fpr, tpr, thr = roc_curve(y, x)
        j = int(np.argmax(tpr - fpr))
        best = {'thr': float(thr[j]), 'sens': round(100 * float(tpr[j]), 1), 'spec': round(100 * (1 - float(fpr[j])), 1)}
    # what the live rule actually does, for comparison with the best the feature allows
    live = {'sens': rate([r for r in lab if r['siteTruth'] == 'Murmur'], lambda r: r['det']['murmur']),
            'spec': rate([r for r in lab if r['siteTruth'] == 'Normal'], lambda r: not r['det']['murmur'])}
    out['feature'] = {'auc': auc, 'n': len(lab), 'bestThreshold': best, 'liveRule': live, 'liveThreshold': 2.5,
                      'dist': mwu([r['sysDia'] for r in lab if r['siteTruth'] == 'Normal'],
                                  [r['sysDia'] for r in lab if r['siteTruth'] == 'Murmur'])}

    # ---- 2 grade dose-response (per patient: flagged at any valve)
    P = {}
    for r in R:
        p = P.setdefault(r['key'], {'murmur': r['murmur'], 'grading': r['grading'], 'timing': r['timing'],
                                    'ageBand': r['ageBand'], 'campaign': r['campaign'], 'det': False, 'sysDia': []})
        p['det'] = p['det'] or r['det']['murmur']
        if ok(r['sysDia']): p['sysDia'].append(r['sysDia'])
    PL = list(P.values())
    byGrade = {g: rate([p for p in PL if p['grading'] == g], lambda p: p['det']) for g in GRADES}
    gp = [(GRADES.index(p['grading']), 1 if p['det'] else 0) for p in PL if p['grading'] in GRADES]
    rho, pv = stats.spearmanr([a for a, _ in gp], [b for _, b in gp]) if len(gp) > 3 else (None, None)
    out['grade'] = {'rows': byGrade, 'trendRho': None if rho is None else float(rho),
                    'trendP': None if pv is None else float(pv), 'n': len(gp)}
    out['timing'] = {t: rate([p for p in PL if p['timing'] == t], lambda p: p['det']) for t in TIMINGS}
    out['age'] = {b: rate([p for p in PL if p['ageBand'] == b and p['murmur'] == 'Present'], lambda p: p['det']) for b in BANDS}

    # ---- 3 is segmentation the upstream limit?
    seg = [r for r in R if r.get('gt') and r['gt']['nS1'] > 0]
    for r in seg: r['_rec'] = r['gt']['s1hit'] / r['gt']['nS1']
    mur = [r for r in seg if r['siteTruth'] == 'Murmur']
    out['segmentation'] = {
        'n': len(seg),
        's1Recall': round(100 * sum(r['gt']['s1hit'] for r in seg) / max(1, sum(r['gt']['nS1'] for r in seg)), 1),
        's2Recall': round(100 * sum(r['gt']['s2hit'] for r in seg) / max(1, sum(r['gt']['nS2'] for r in seg)), 1),
        # among true murmur valves, is recall higher where the detector fired?
        'recallWhenFound': mwu([r['_rec'] for r in mur if not r['det']['murmur']],
                               [r['_rec'] for r in mur if r['det']['murmur']]),
        'byHr': [{'bin': lab, 'n': len(v), 'recall': round(100 * float(np.mean(v)), 1)}
                 for lab, v in [(l, [r['_rec'] for r in seg if ok(r['hr']) and lo <= r['hr'] < hi])
                                for l, lo, hi in [('<80', 0, 80), ('80–99', 80, 100), ('100–119', 100, 120), ('120+', 120, 999)]] if v]}

    # ---- 4 confound: the two screening campaigns are different batches
    camps = sorted({p['campaign'] for p in PL if p['campaign']})
    out['campaign'] = {c: {'patients': sum(1 for p in PL if p['campaign'] == c),
                           'murmurPrev': rate([p for p in PL if p['campaign'] == c], lambda p: p['murmur'] == 'Present'),
                           'found': rate([p for p in PL if p['campaign'] == c and p['murmur'] == 'Present'], lambda p: p['det']),
                           'falsePos': rate([p for p in PL if p['campaign'] == c and p['murmur'] == 'Absent'], lambda p: p['det'])}
                       for c in camps}

    D = json.load(open(WEB_DATA / 'index.json'))  # touch nothing here; only discover.json is written
    DJ = WEB_DATA / 'discover.json'
    disc = json.load(open(DJ))
    disc['circor'] = out
    json.dump(disc, open(DJ, 'w'), separators=(',', ':'))
    print(f"circor: {out['n']['patients']} patients, {out['n']['recordings']} recordings, {out['n']['hours']} h")
    print(f"  sysDia AUC = {auc:.3f} (n={len(lab)} labelled valves)" if auc else '  sysDia AUC: n/a')
    if best: print(f"  best possible at any threshold: sens {best['sens']} % / spec {best['spec']} % at {best['thr']:.2f}x")
    print(f"  live rule (threshold 2.5):      sens {live['sens']['pct']} % / spec {live['spec']['pct']} %")
    print(f"  by grade: " + '  '.join(f"{g} {byGrade[g]['pct']}%" for g in GRADES) + f"   trend rho={rho:.2f}" if rho is not None else '')
    print(f"  S1 recall {out['segmentation']['s1Recall']} %, S2 {out['segmentation']['s2Recall']} %")
    print('-> web/data/discover.json [circor]')


if __name__ == '__main__':
    main()
