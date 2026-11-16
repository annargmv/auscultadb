"""Train the learned heart-sound segmenter on the CirCor experts' segmentation → web/data/segmenter.json.

Input: build/seg/frames.f32 + index.jsonl (pipeline/circor_frames.js: the page's segBands on every recording) and the
experts' .tsv files (1 = S1, 2 = systole, 3 = S2, 4 = diastole, 0 = not annotated).

Protocol (rule 4): the held-out patients are the SAME ones the murmur model is scored on (models.json heart_murmur
testPids). Nothing from them is used here: not for the classifier, not for the priors, not for the duration rules.
Their recordings are only scored afterwards by pipeline/seg_validate.js.

Features (identical formulas in the page's segFeatures): per band, robust z = (x − median) / IQR over the recording
(quantile = sorted[floor(p·(n−1))]); the z values at frame offsets −20, −12, −6, −3, 0, +3, +6, +12, +20 (10-ms frames,
clamped at the edges); plus the local slope z[f+1] − z[f−1]. Classifier: gradient boosting, fixed settings.
Decoder (page, segViterbi): cyclic S1 → systole → S2 → diastole with minimum durations = 10th percentile of the
training segments and a geometric tail matched to the mean duration.

  python3 pipeline/train_segmenter.py
"""
import json, datetime
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from paths import ROOT, META, WEB_DATA

SEG = ROOT / 'build' / 'seg'
TD = ROOT / 'raw' / 'circor' / 'circor-heart-sound-1.0.3' / 'training_data'
OFFS = [-20, -12, -6, -3, 0, 3, 6, 12, 20]
SEED = 20261116


def q(sorted_v, p): return sorted_v[int(np.floor(p * (len(sorted_v) - 1)))]


def features(B):
    nF, nb = B.shape; Z = np.empty_like(B)
    for j in range(nb):
        s = np.sort(B[:, j]); md = q(s, 0.5); iq = max(1e-6, q(s, 0.75) - q(s, 0.25)); Z[:, j] = (B[:, j] - md) / iq
    cl = lambda i: np.clip(i, 0, nF - 1); f = np.arange(nF)
    cols = [Z[cl(f + o)] for o in OFFS] + [Z[cl(f + 1)] - Z[cl(f - 1)]]
    return np.hstack(cols).astype(np.float32)


def labels(name, nF, t0, hop):
    p = TD / f'{name}.tsv'; y = np.zeros(nF, np.int8)
    if not p.exists(): return y, []
    rows = [list(map(float, l.split())) for l in open(p) if l.strip()]
    t = t0 + np.arange(nF) * hop
    for a, b, lab in rows:
        if lab > 0: y[(t >= a) & (t < b)] = int(lab)
    return y, rows


def export(m):
    trees = [[[[int(r['feature_idx']), float(r['num_threshold']), int(r['missing_go_to_left']), int(r['left']), int(r['right']), int(r['is_leaf']), float(r['value'])]
                for r in tp.nodes] for tp in it] for it in m._predictors]
    return {'kind': 'gbtree', 'baseline': np.asarray(m._baseline_prediction).ravel().tolist(), 'trees': trees, 'binary': False}


def main():
    idx = [json.loads(l) for l in open(SEG / 'index.jsonl')]
    F = np.fromfile(SEG / 'frames.f32', dtype='<f4').reshape(-1, idx[0]['nb'])
    test = set(next(m for m in json.loads((WEB_DATA / 'models.json').read_text())['models'] if m['id'] == 'heart_murmur')['testPids'])
    X, Y, durs = [], [], {c: [] for c in (1, 2, 3, 4)}
    for r in idx:
        if r['pid'] in test: continue
        B = F[r['offset']:r['offset'] + r['nF']]; y, rows = labels(r['name'], r['nF'], r['t0'], r['hop'])
        keep = np.where(y > 0)[0]
        if not len(keep): continue
        keep = keep[::2]                                   # every 2nd annotated frame: neighbours are near-duplicates
        X.append(features(B)[keep]); Y.append(y[keep] - 1)
        for a, b, lab in rows:
            if lab > 0: durs[int(lab)].append((b - a) / r['hop'])
    X = np.vstack(X); Y = np.concatenate(Y)
    print(f'training frames {len(Y):,} from {sum(1 for r in idx if r["pid"] not in test)} recordings; class shares', np.bincount(Y) / len(Y))
    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, max_iter=200, l2_regularization=1.0, random_state=SEED).fit(X, Y)
    priors = (np.bincount(Y, minlength=4) / len(Y)).tolist()
    minDur, stay, leave, meanDur = [], [], [], []
    for c in (1, 2, 3, 4):
        d = np.array(durs[c]); mn = max(2, int(np.floor(np.percentile(d, 10)))); mu = float(d.mean())
        p_stay = max(0.0, 1 - 1 / max(1.0, mu - mn + 1)); minDur.append(mn); meanDur.append(round(mu, 1))
        stay.append(float(np.log(max(p_stay, 1e-6)))); leave.append(float(np.log(max(1 - p_stay, 1e-6))))
    # parity sample: one held-out recording's raw bands + Python probabilities for 40 frames
    pr = next(r for r in idx if r['pid'] in test and 300 < r['nF'] < 800)
    Bp = F[pr['offset']:pr['offset'] + pr['nF']]; Pp = m.predict_proba(features(Bp))
    out = {'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'seed': SEED, 'bands': [[25, 60], [60, 120], [120, 250], [250, 500], [500, 1000]], 'offsets': OFFS,
           'classes': ['S1', 'systole', 'S2', 'diastole'], 'model': export(m), 'priors': priors,
           'hmm': {'minDur': minDur, 'meanDur': meanDur, 'stayLog': stay, 'leaveLog': leave},
           'train': {'frames': int(len(Y)), 'recordings': sum(1 for r in idx if r['pid'] not in test), 'heldOutPatients': len(test)},
           'parity': {'name': pr['name'], 'bands': [round(float(v), 6) for v in Bp.ravel()], 'nb': int(Bp.shape[1]),
                      'frames': list(range(0, pr['nF'], max(1, pr['nF'] // 40)))[:40]}}
    out['parity']['p'] = [[round(float(v), 9) for v in Pp[f]] for f in out['parity']['frames']]
    # parity bands are rounded: recompute Python probabilities on the ROUNDED bands so JS and Python see identical input
    Bq = np.array(out['parity']['bands'], dtype=np.float32).reshape(-1, Bp.shape[1]); Pq = m.predict_proba(features(Bq))
    out['parity']['p'] = [[round(float(v), 9) for v in Pq[f]] for f in out['parity']['frames']]
    (WEB_DATA / 'segmenter.json').write_text(json.dumps(out, separators=(',', ':')))
    print('min durations (frames)', minDur, 'mean', meanDur, '| ->', WEB_DATA / 'segmenter.json', (WEB_DATA / 'segmenter.json').stat().st_size // 1024, 'KB')


if __name__ == '__main__':
    main()
