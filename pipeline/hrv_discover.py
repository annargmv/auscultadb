"""Heart-rate variability from heart sounds: is it measured well enough to show? → discover.json['hrv'].

Reads meta/features/circor_ext.jsonl (pipeline/circor_ext.js): for each CirCor recording, RMSSD from the page's beat
finder and RMSSD from the CirCor experts' S1 annotation on the SAME annotated window (same artefact rule, same code).

Pass rule — written down BEFORE the first result was seen, and not changed afterwards:
  the HRV marker is shown on patient pages and on the Analyze page only if, on recordings where both RMSSDs exist,
  agreement between beat finder and experts has ICC(2,1) >= 0.75 ("good", Koo & Li 2016) AND |mean bias| <= 10 ms.
  Otherwise the pages say the marker failed validation and show no number.

Also reported: how often RMSSD can be measured at all; heart-rate agreement (sanity check); valve-to-valve
repeatability within a child (recordings from one sitting) for the beat finder and for the experts (the ceiling);
RMSSD by age band from the expert annotation (a reference, children only).

Literature (PubMed-verified): a single 10-s ECG gives a valid RMSSD (Munoz et al., PLoS One 2015, PMID 26414314);
reduced HRV predicted all-cause mortality in older adults — SDNN and spectral power from 2-h ECG, not RMSSD
(Tsuji et al., Circulation 1994, PMID 8044959). No dataset here links HRV to outcome.

  python3 pipeline/hrv_discover.py      (then merge.py --bump)
"""
import collections, json, math, datetime
import numpy as np
from scipy import stats
from paths import FEATURES, WEB_DATA

ICC_MIN, BIAS_MAX = 0.75, 10.0
BANDS = ['Neonate', 'Infant', 'Child', 'Adolescent']


def icc21(a, b):
    """ICC(2,1): two-way random effects, absolute agreement, single measure (Shrout & Fleiss)."""
    X = np.column_stack([a, b]).astype(float); n, k = X.shape
    gm = X.mean(); rm = X.mean(1); cm = X.mean(0)
    msr = k * ((rm - gm) ** 2).sum() / (n - 1); msc = n * ((cm - gm) ** 2).sum() / (k - 1)
    mse = ((X - rm[:, None] - cm[None, :] + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def icc11(pairs):
    """ICC(1,1): one-way random, single measure — repeatability of one method across two valves of the same child."""
    X = np.array(pairs, float); n, k = X.shape; gm = X.mean(); rm = X.mean(1)
    msb = k * ((rm - gm) ** 2).sum() / (n - 1); msw = ((X - rm[:, None]) ** 2).sum() / (n * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw))


def boot(f, n, B=2000, seed=20261116):
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        try: v.append(f(i))
        except Exception: pass
    v = [x for x in v if np.isfinite(x)]
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if v else None


def main():
    R = [json.loads(l) for l in open(FEATURES / 'circor_ext.jsonl')]; R = [r for r in R if 'err' not in r]
    ann = [r for r in R if r.get('hrvExp')]
    both = [r for r in ann if r['hrvDet']['rmssd'] is not None and r['hrvExp']['rmssd'] is not None]
    a = np.array([r['hrvDet']['rmssd'] for r in both]); b = np.array([r['hrvExp']['rmssd'] for r in both]); d = a - b
    out = {'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'rule': {'iccMin': ICC_MIN, 'biasMax': BIAS_MAX},
           'refs': [{'pmid': '26414314', 'cite': 'Munoz ML et al. Validity of (ultra-)short recordings for heart rate variability measurements. PLoS One 2015;10:e0138921'},
                    {'pmid': '8044959', 'cite': 'Tsuji H et al. Reduced heart rate variability and mortality risk in an elderly cohort. The Framingham Heart Study. Circulation 1994;90:878–83'}],
           'n': {'recordings': len(R), 'annotated': len(ann), 'both': len(both)},
           'coverage': {'detectorWhole': round(100 * sum(1 for r in R if r['hrvAll']['rmssd'] is not None) / len(R), 1),
                        'detectorAnnotated': round(100 * sum(1 for r in ann if r['hrvDet']['rmssd'] is not None) / max(1, len(ann)), 1),
                        'expertAnnotated': round(100 * sum(1 for r in ann if r['hrvExp']['rmssd'] is not None) / max(1, len(ann)), 1)}}
    if len(both) >= 20:
        icc = icc21(a, b)
        out['agreement'] = {'icc': round(icc, 3), 'iccCI': boot(lambda i: icc21(a[i], b[i]), len(a)), 'bias': round(float(d.mean()), 1),
                            'loa': [round(float(d.mean() - 1.96 * d.std(ddof=1)), 1), round(float(d.mean() + 1.96 * d.std(ddof=1)), 1)],
                            'rho': round(float(stats.spearmanr(a, b)[0]), 3), 'medianDet': float(np.median(a)), 'medianExp': float(np.median(b)),
                            'within10ms': round(100 * float(np.mean(np.abs(d) <= 10)), 1),
                            'points': [[round(float(x), 1), round(float(y), 1)] for x, y in zip(b, a)][:1500]}
        out['pass'] = bool(icc >= ICC_MIN and abs(d.mean()) <= BIAS_MAX)
    else:
        out['pass'] = False
    # heart-rate agreement on the same windows (sanity: the beats are found, the variability is what is in question)
    hb = [r for r in ann if r['hrvDet']['hr'] and r['hrvExp']['hr']]
    if len(hb) >= 20:
        x = np.array([r['hrvDet']['hr'] for r in hb]); y = np.array([r['hrvExp']['hr'] for r in hb])
        out['hrAgreement'] = {'n': len(hb), 'icc': round(icc21(x, y), 3), 'bias': round(float((x - y).mean()), 1), 'within5bpm': round(100 * float(np.mean(np.abs(x - y) <= 5)), 1)}
    # repeatability across two valves of the same child (one sitting)
    rep = {}
    for key, get in (('detector', lambda r: r['hrvAll']['rmssd']), ('expert', lambda r: r['hrvExp']['rmssd'] if r.get('hrvExp') else None)):
        by = collections.defaultdict(list)
        for r in R:
            v = get(r)
            if v is not None: by[r['pid']].append(v)
        pairs = [v[:2] for v in by.values() if len(v) >= 2]
        if len(pairs) >= 20:
            P = np.array(pairs); rep[key] = {'children': len(pairs), 'icc': round(icc11(P), 3), 'iccCI': boot(lambda i: icc11(P[i]), len(P))}
    out['repeatability'] = rep
    # expert RMSSD by age band: a reference for children, not a norm
    out['byBand'] = []
    for band in BANDS:
        v = [r['hrvExp']['rmssd'] for r in ann if r['ageBand'] == band and r['hrvExp']['rmssd'] is not None]
        if len(v) >= 5: out['byBand'].append({'band': band, 'n': len(v), 'p25': float(np.percentile(v, 25)), 'p50': float(np.median(v)), 'p75': float(np.percentile(v, 75))})
    # values for the browsable patients' recordings (from the ORIGINAL WAV), read by the patient pages
    from paths import META
    C = json.load(open(META / 'circor_cohort.json')); C = C['patients'] if isinstance(C, dict) else C
    names = {st['name'] for p in C for st in p['sites'].values()}
    out['perRecording'] = {r['name']: {'rmssd': r['hrvAll']['rmssd'], 'hr': r['hrvAll']['hr'], 'n': r['hrvAll']['n'], 'cleanFrac': r['hrvAll'].get('cleanFrac')}
                           for r in R if r['name'] in names}
    DJ = WEB_DATA / 'discover.json'; disc = json.load(open(DJ)); disc['hrv'] = out
    DJ.write_text(json.dumps(disc, separators=(',', ':')))
    g = out.get('agreement', {})
    print(f"recordings {out['n']}, coverage {out['coverage']}")
    print(f"RMSSD beat finder vs experts: ICC {g.get('icc')} {g.get('iccCI')}, bias {g.get('bias')} ms, LoA {g.get('loa')}, rho {g.get('rho')}, within ±10 ms {g.get('within10ms')} %")
    print(f"HR agreement {out.get('hrAgreement')}")
    print(f"repeatability across valves {rep}")
    print(f"PASS (ICC >= {ICC_MIN} and |bias| <= {BIAS_MAX} ms): {out['pass']}")
    print('-> discover.json[hrv]   (then merge.py --bump)')


if __name__ == '__main__':
    main()
