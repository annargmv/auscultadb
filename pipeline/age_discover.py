"""Does auscultation carry a biological-age signal? → discover.json['age'].

Motivation. In other modalities, the gap between a model's predicted age and a person's real age is
a longevity marker: the ECG-age gap predicts all-cause mortality (Lima et al., Nat Commun 2021,
PMID 34433816), and resting heart rate is itself associated with mortality in the general population
(Zhang et al., CMAJ 2016, PMID 26598376). The question here is whether breath or heart sounds carry
enough age information for such a gap to exist at all. **No dataset here has survival or follow-up
data, so nothing in this file can measure longevity** — it can only test the precondition.

All analyses are fixed in advance and fit on one split, reported on the other (rule 4):
  * split by patient, seeded (SEED), TEST_FRAC held out; the model never sees a test patient;
  * model: standardised ridge regression on the existing per-recording detector features, alpha chosen
    by inner cross-validation on the training split only; no other model or feature set is tried here;
  * skill is reported as test MAE against the null model (predict the training-split mean), with a
    bootstrap 95 % CI on the MAE difference and a label-permutation p-value.

  A  breath-sound age clock, adults — KAUH (one hospital, one stethoscope, ages 12–90).
  B  breath-sound age clock, children — SPRSound, *within each release*, because age and release
     (recording batch) are confounded across releases.
  C  heart-sound maturation — CirCor: does acoustic heart rate track the age band? Reference ranges are
     fit on training-split patients with no murmur and a normal clinical outcome.
  D  HR-for-age deviation vs CirCor's clinical outcome (normal / abnormal), on the test split only —
     the nearest thing here to an outcome a deviation could predict.

Run after the feature tables exist (meta/features/*.jsonl). Then `python3 pipeline/merge.py --bump`.
"""
import collections, json, math, datetime
import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from paths import FEATURES, WEB_DATA

SEED, TEST_FRAC, N_PERM, N_BOOT = 20261108, 0.4, 500, 2000
FEATS = ['f50', 'f75', 'cen', 'rr', 'hr', 'crkRate', 'fineRate', 'tonFrac', 'ie']
BANDS = ['Neonate', 'Infant', 'Child', 'Adolescent']
ALPHAS = np.logspace(-2, 3, 21)
ok = lambda v: v is not None and not (isinstance(v, float) and math.isnan(v))


def load(name):
    return [json.loads(l) for l in open(FEATURES / name)]


def split(ids, rng):
    ids = sorted(set(ids)); rng.shuffle(ids)
    k = int(round(len(ids) * TEST_FRAC))
    return set(ids[k:]), set(ids[:k])          # train, test


def matrix(rows):
    X = np.array([[float(r[f]) if ok(r.get(f)) else np.nan for f in FEATS] for r in rows])
    return np.log1p(np.clip(X, 0, None))       # rates and frequencies are right-skewed


def fit(X, y):
    med = np.nanmedian(X, 0); med = np.where(np.isnan(med), 0, med)
    Xf = np.where(np.isnan(X), med, X)
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(Xf, y)
    return m, med


def predict(m, med, X):
    return m.predict(np.where(np.isnan(X), med, X))


def per_patient(pids, y, p):
    """Average recordings of one patient: one prediction per person, as an age clock would report."""
    acc = collections.defaultdict(list)
    for i, pid in enumerate(pids): acc[pid].append(i)
    P = sorted(acc)
    return P, np.array([y[acc[k][0]] for k in P]), np.array([np.mean(p[acc[k]]) for k in P])


def clock(rows, key, rng, label):
    """Fit on train patients, report on test patients. Returns the summary dict."""
    tr, te = split([r[key] for r in rows], rng)
    R_tr = [r for r in rows if r[key] in tr]; R_te = [r for r in rows if r[key] in te]
    ytr = np.array([r['age'] for r in R_tr], float); yte = np.array([r['age'] for r in R_te], float)
    m, med = fit(matrix(R_tr), ytr)
    P, yt, pt = per_patient([r[key] for r in R_te], yte, predict(m, med, matrix(R_te)))
    null = float(np.mean(ytr))
    mae, mae0 = float(np.mean(np.abs(pt - yt))), float(np.mean(np.abs(null - yt)))
    rho, prho = stats.spearmanr(pt, yt) if len(P) > 3 else (float('nan'), float('nan'))
    # bootstrap over test patients: CI of (null MAE - model MAE); > 0 means the model helps
    b = rng.integers(0, len(P), (N_BOOT, len(P)))
    gain = np.mean(np.abs(null - yt)[b], 1) - np.mean(np.abs(pt - yt)[b], 1)
    # permutation: refit on shuffled training ages; how often is shuffled MAE as good as real MAE?
    Xtr, Xte = matrix(R_tr), matrix(R_te); hits = 0
    for _ in range(N_PERM):
        mp, mdp = fit(Xtr, rng.permutation(ytr))
        _, _, pp = per_patient([r[key] for r in R_te], yte, predict(mp, mdp, Xte))
        hits += np.mean(np.abs(pp - yt)) <= mae
    coef = dict(zip(FEATS, map(float, m[-1].coef_)))
    out = {'label': label, 'nTrain': len(tr), 'nTest': len(te), 'recTrain': len(R_tr), 'recTest': len(R_te),
           'ageRange': [float(min(yte.min(), ytr.min())), float(max(yte.max(), ytr.max()))],
           'mae': round(mae, 2), 'maeNull': round(mae0, 2), 'gainCI': [round(float(np.percentile(gain, 2.5)), 2), round(float(np.percentile(gain, 97.5)), 2)],
           'pPerm': round((hits + 1) / (N_PERM + 1), 4), 'rho': None if math.isnan(rho) else round(float(rho), 3),
           'pRho': None if math.isnan(prho) else float(prho), 'alpha': float(m[-1].alpha_), 'coef': coef,
           'test': [{'age': float(a), 'pred': round(float(q), 2)} for a, q in zip(yt, pt)]}
    out['significant'] = bool(out['pPerm'] < 0.05 and out['gainCI'][0] > 0)
    return out


def main():
    rng = np.random.default_rng(SEED)
    out = {'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'seed': SEED, 'testFrac': TEST_FRAC,
           'refs': [{'pmid': '34433816', 'cite': 'Lima EM et al. Deep neural network-estimated electrocardiographic age as a mortality predictor. Nat Commun 2021;12:5117'},
                    {'pmid': '26598376', 'cite': 'Zhang D et al. Resting heart rate and all-cause and cardiovascular mortality in the general population: a meta-analysis. CMAJ 2016;188(3):E53–E63'}]}

    # ---- A adults (KAUH): one row per recording (3 filter modes), averaged per subject
    K = [r for r in load('kauh.jsonl') if ok(r.get('age'))]
    out['adults'] = clock(K, 'pid', rng, 'KAUH adults')

    # ---- B children (SPRSound), within release
    S = [r for f in sorted(FEATURES.glob('spr.*.jsonl')) for r in load(f.name) if ok(r.get('age')) and r['pid'] != '0']
    out['children'] = []
    for rel in sorted({r['rel'] for r in S}):
        rs = [r for r in S if r['rel'] == rel]
        if len({r['pid'] for r in rs}) < 40: continue
        out['children'].append({'release': rel, **clock(rs, 'pid', rng, f'SPRSound {rel}')})

    # ---- C/D heart (CirCor): one median HR per patient
    C = [r for r in load('circor.jsonl') if ok(r.get('hr')) and r.get('ageBand') in BANDS]
    by = collections.defaultdict(list)
    for r in C: by[r['pid']].append(r)
    P = [{'pid': k, 'hr': float(np.median([r['hr'] for r in v])), 'band': v[0]['ageBand'], 'outcome': v[0]['outcome'],
          'murmur': v[0]['murmur'], 'campaign': v[0]['campaign']} for k, v in by.items()]
    tr, te = split([p['pid'] for p in P], rng)
    healthy = [p for p in P if p['pid'] in tr and p['murmur'] == 'Absent' and p['outcome'] == 'Normal']
    ref = {}
    for b in BANDS:
        h = np.array([p['hr'] for p in healthy if p['band'] == b])
        if len(h) >= 5:
            q1, q2, q3 = np.percentile(h, [25, 50, 75])
            ref[b] = {'n': int(len(h)), 'p10': float(np.percentile(h, 10)), 'p25': float(q1), 'p50': float(q2), 'p75': float(q3),
                      'p90': float(np.percentile(h, 90)), 'scale': float(max((q3 - q1) / 1.349, 1.0))}
    T = [p for p in P if p['pid'] in te and p['band'] in ref]
    rank = {b: i for i, b in enumerate(BANDS)}
    rho, prho = stats.spearmanr([rank[p['band']] for p in T], [p['hr'] for p in T])
    out['heart'] = {'ref': ref, 'refN': len(healthy), 'nTest': len(T), 'rho': round(float(rho), 3), 'pRho': float(prho),
                    'byBand': [{'band': b, 'n': sum(1 for p in T if p['band'] == b),
                                'p50': float(np.median([p['hr'] for p in T if p['band'] == b]))} for b in BANDS if any(p['band'] == b for p in T)],
                    'significant': bool(prho < 0.05)}
    for p in T: p['z'] = (p['hr'] - ref[p['band']]['p50']) / ref[p['band']]['scale']

    def mwu(a, b):
        u, pv = stats.mannwhitneyu(a, b, alternative='two-sided')
        return {'nA': len(a), 'nB': len(b), 'medA': round(float(np.median(a)), 3), 'medB': round(float(np.median(b)), 3),
                'p': float(pv), 'r': round(float(2 * u / (len(a) * len(b)) - 1), 3)}
    ab, nm = [p['z'] for p in T if p['outcome'] == 'Abnormal'], [p['z'] for p in T if p['outcome'] == 'Normal']
    out['outcome'] = {'z': mwu(nm, ab), 'absZ': mwu([abs(x) for x in nm], [abs(x) for x in ab]),
                      'byCampaign': {c: mwu([p['z'] for p in T if p['campaign'] == c and p['outcome'] == 'Normal'],
                                            [p['z'] for p in T if p['campaign'] == c and p['outcome'] == 'Abnormal'])
                                     for c in sorted({p['campaign'] for p in T})},
                      'murmur': mwu([p['z'] for p in T if p['murmur'] == 'Absent'], [p['z'] for p in T if p['murmur'] == 'Present'])}
    out['outcome']['significant'] = bool(out['outcome']['z']['p'] < 0.05 or out['outcome']['absZ']['p'] < 0.05)

    DJ = WEB_DATA / 'discover.json'
    disc = json.load(open(DJ)); disc['age'] = out
    txt = json.dumps(disc, separators=(',', ':'))   # serialise before opening: a failure must not truncate the file
    DJ.write_text(txt)
    a = out['adults']
    print(f"A adults   : test MAE {a['mae']} y vs null {a['maeNull']} y, gain CI {a['gainCI']}, perm p={a['pPerm']}, rho={a['rho']}  -> {'SIGNAL' if a['significant'] else 'no signal'}")
    for c in out['children']:
        print(f"B {c['release']:<10}: test MAE {c['mae']} vs null {c['maeNull']}, gain CI {c['gainCI']}, perm p={c['pPerm']}, rho={c['rho']} -> {'SIGNAL' if c['significant'] else 'no signal'}")
    h = out['heart']
    print(f"C heart    : HR by band {[(x['band'], round(x['p50'])) for x in h['byBand']]}, rho={h['rho']} p={h['pRho']:.2g} (n={h['nTest']} test patients)")
    o = out['outcome']
    print(f"D outcome  : HR-for-age z normal {o['z']['medA']} vs abnormal {o['z']['medB']} p={o['z']['p']:.3f}; |z| p={o['absZ']['p']:.3f}")
    print('-> web/data/discover.json [age]   (then: python3 pipeline/merge.py --bump)')


if __name__ == '__main__':
    main()
