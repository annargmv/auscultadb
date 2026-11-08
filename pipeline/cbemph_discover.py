"""Chronic bronchitis / emphysema vs healthy adults, on ONE device → discover.json['cbemph'].

Run after `node pipeline/precompute.js cbfeatures`.

The point of this set: Fraiwan/Haider/Khassawneh recorded it with the **same 3M Littmann 3200 and the
same protocol as KAUH**, so these 234 recordings can be compared with KAUH's 35 healthy adults without
the stethoscope confound that limits every SPRSound-vs-KAUH comparison on the Discoveries page.

Caveat carried through every result: KAUH exported each recording in three Littmann filter modes
(bell / diaphragm / extended) and this set does not say which mode it used. Rather than assume
diaphragm, each comparison is run against all three KAUH modes and the spread is reported — if a
difference only survives in one mode it is a filter artefact, not a disease effect.
"""
import collections, json, math
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from paths import FEATURES, KAUH_FEATURES, WEB_DATA

FEATS = [('cen', 'Spectral centroid (Hz)'), ('crkRate', 'Crackle-like transients /s'),
         ('fineRate', 'Fine-crackle-like /s'), ('tonFrac', 'Tonal (wheeze-like) time fraction'),
         ('rr', 'Breathing rate (/min)'), ('f50', 'Median power frequency (Hz)')]
ok = lambda v: v is not None and not (isinstance(v, float) and math.isnan(v))


def mwu(a, b):
    a, b = [x for x in a if ok(x)], [x for x in b if ok(x)]
    if len(a) < 4 or len(b) < 4: return None
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    return {'nA': len(a), 'nB': len(b), 'medA': float(np.median(a)), 'medB': float(np.median(b)),
            'p': float(p), 'r': float(2 * u / (len(a) * len(b)) - 1)}


def probe(rows, y, cols, reps=30):
    X = np.array([[r[c] if ok(r.get(c)) else 0 for c in cols] for r in rows]); a = []
    for s in range(reps):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000))
        pr = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=s), method='predict_proba')[:, 1]
        a.append(roc_auc_score(y, pr))
    return [float(np.mean(a)), float(np.std(a))]


def main():
    R = [json.loads(l) for l in open(FEATURES / 'cbemph.jsonl')]
    R = [r for r in R if 'err' not in r]
    K = [json.loads(l) for l in open(KAUH_FEATURES)]
    healthy = {m: [r for r in K if r['mode'] == m and r['label'] == 'Normal'] for m in 'DBE'}
    CB = [r for r in R if r['group'] == 'CB']; EM = [r for r in R if r['group'] == 'EMPHYSEMA']

    out = {'n': {'recordings': len(R), 'cb': len(CB), 'emphysema': len(EM),
                 'minutes': round(sum(r['dur'] for r in R) / 60, 1),
                 'kauhHealthy': {m: len(v) for m, v in healthy.items()}},
           'device': '3M Littmann 3200 — same as KAUH, so these comparisons carry no stethoscope confound',
           'modeUnknown': True}

    # --- 1 each disease vs KAUH healthy, in every KAUH filter mode
    vs = {}
    for name, grp in (('Chronic bronchitis', CB), ('Emphysema', EM)):
        vs[name] = {}
        for m in 'DBE':
            vs[name][m] = {k: mwu([h[k] for h in healthy[m]], [r[k] for r in grp]) for k, _ in FEATS}
    out['vsHealthy'] = {'feats': FEATS, 'rows': vs}
    # which features survive in ALL three modes (robust to the unknown filter)
    robust = {}
    for name in vs:
        robust[name] = [k for k, _ in FEATS
                        if all(vs[name][m][k] and vs[name][m][k]['p'] < 0.05 for m in 'DBE')
                        and len({np.sign(vs[name][m][k]['r']) for m in 'DBE'}) == 1]
    out['robust'] = robust

    # --- 2 CB vs emphysema: can sound alone separate two COPD-spectrum diseases?
    rows = CB + EM; y = np.array([0] * len(CB) + [1] * len(EM))
    cols = [k for k, _ in FEATS]
    out['cbVsEmph'] = {'nCB': len(CB), 'nEM': len(EM), 'auc': probe(rows, y, cols),
                       'perFeature': {k: mwu([r[k] for r in CB], [r[k] for r in EM]) for k in cols}}

    # --- 3 what the detector actually reports on them
    out['detector'] = {g: {'crkRate': float(np.median([r['crkRate'] for r in v if ok(r['crkRate'])])),
                           'tonFrac': float(np.median([r['tonFrac'] for r in v if ok(r['tonFrac'])])),
                           'rrEstimated': sum(1 for r in v if r.get('est')), 'n': len(v)}
                       for g, v in (('Chronic bronchitis', CB), ('Emphysema', EM))}

    # --- 4 confound check: these recordings are ~25 s, KAUH's are ~16 s. Breathing rate from a short
    #     clip is estimated from very few breaths, so a "disease" difference in rr may just be duration.
    allr = [r for r in R if ok(r.get('rr')) and ok(r.get('dur'))] + \
           [h for h in healthy['D'] if ok(h.get('rr')) and ok(h.get('dur'))]
    rho, prho = stats.spearmanr([r['dur'] for r in allr], [r['rr'] for r in allr])
    kd = [h['dur'] for h in healthy['D']]; cd = [r['dur'] for r in R]
    # restrict both groups to the overlapping duration band and re-test rr there
    lo, hi = max(min(kd), min(cd)), min(max(kd), max(cd))
    kin = [h for h in healthy['D'] if lo <= h['dur'] <= hi and ok(h.get('rr'))]
    cin = [r for r in R if lo <= r['dur'] <= hi and ok(r.get('rr'))]
    out['durationConfound'] = {
        'kauhMedianDur': float(np.median(kd)), 'cbemphMedianDur': float(np.median(cd)),
        'rhoDurRr': float(rho), 'pDurRr': float(prho), 'n': len(allr),
        'overlapBand': [round(lo, 1), round(hi, 1)],
        'rrMatched': mwu([h['rr'] for h in kin], [r['rr'] for r in cin]),
        'note': 'Breathing rate is estimated from a handful of breaths in a short clip. If rr tracks '
                'duration, the healthy-vs-disease rr difference is a recording-length artefact, not physiology.'}

    DJ = WEB_DATA / 'discover.json'
    disc = json.load(open(DJ)); disc['cbemph'] = out
    json.dump(disc, open(DJ, 'w'), separators=(',', ':'))

    print(f"cbemph: {len(R)} recordings ({len(CB)} CB, {len(EM)} emphysema), {out['n']['minutes']} min")
    print(f"  KAUH healthy per mode: {out['n']['kauhHealthy']}")
    for name in vs:
        print(f"  {name} vs healthy — significant in ALL three filter modes: {robust[name] or 'none'}")
        d = vs[name]['D']
        for k, lab in FEATS:
            m = d[k]
            if m: print(f"      {lab:34} healthy {m['medA']:8.2f}  vs  {m['medB']:8.2f}   r={m['r']:+.2f}  p={m['p']:.1e}")
    print(f"  CB vs emphysema from sound alone: AUC {out['cbVsEmph']['auc'][0]:.3f} ± {out['cbVsEmph']['auc'][1]:.3f}")
    dc = out['durationConfound']
    print(f"  CONFOUND CHECK  duration: KAUH {dc['kauhMedianDur']:.1f} s vs cbemph {dc['cbemphMedianDur']:.1f} s")
    print(f"     rr vs duration across both sets: rho={dc['rhoDurRr']:+.2f} (p={dc['pDurRr']:.1e}, n={dc['n']})")
    m = dc['rrMatched']
    print(f"     rr re-tested inside the overlapping band {dc['overlapBand']} s: " +
          (f"healthy {m['medA']:.1f} vs disease {m['medB']:.1f}, r={m['r']:+.2f}, p={m['p']:.1e} (nA={m['nA']}, nB={m['nB']})" if m else 'too few recordings'))
    print('-> web/data/discover.json [cbemph]')


if __name__ == '__main__':
    main()
