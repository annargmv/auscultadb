"""External test of the heart skills on ADULT recordings — PhysioNet/CinC 2016 → meta/cinc2016_validation.json.

Inputs: meta/features/cinc2016.jsonl (pipeline/cinc_run.js: the page's frozen murmur network and learned beat finder) and the
ECG simultaneously recorded in training set a (raw/cinc2016/cinc/training-a/*.dat, WFDB).
Data: Liu et al., Physiol Meas 2016;37:2181 (doi 10.1088/0967-3334/37/12/2181), ODC-By 1.0. Per database (paper, Table 2):
  a MITHSDB  normal / mitral valve prolapse / benign murmur / aortic disease / misc. pathology, Meditron, with ECG
  b AADHSDB  normal / coronary artery disease (CAD), Littmann E4000
  c AUTHHSDB normal (29±8 y) / aortic stenosis (75±7 y) / mitral regurgitation (76±10 y), echo-confirmed
  d UHAHSDB  normal (18–40 y) / pathological (44–90 y)
  e DLUTHSDB healthy volunteers (25±3 y) / CAD (60±12 y)
  f SUAHSDB  normal / pathological, adults 16–88 y
Label = the Challenge reference: abnormal (confirmed cardiac diagnosis) vs normal. Not a murmur label.

Rules fixed BEFORE the first result (nothing is trained, tuned or re-thresholded here):
  1. Murmur network: AUC (Hanley–McNeil 95 % CI) abnormal vs normal WITHIN each database (each is one stethoscope, so no
     device mix-up). Primary = databases a and c, whose abnormal recordings are valve disease or murmurs. The network
     "transfers to adult valve disease" only if the AUC CI lower bound is > 0.70 in BOTH a and c. Databases b, d, e, f
     (mainly CAD, often without a murmur) are reported without a claim. At the shipped flag threshold: sensitivity and
     specificity (Agresti–Coull). Sensitivity check: recordings the Challenge marked good quality only.
  2. Heart rate and HRV from the learned beat finder vs the simultaneous ECG (set a): R peaks by wfdb's gqrs detector
     (band-passed ECG, both polarities; records with < 8 ECG beats excluded and counted), then
     the page's own interval rule (drop intervals > 20 % from the median; RMSSD needs ≥ 8 successive differences) applied to
     both. Heart rate passes if ICC(2,1) ≥ 0.90 and |bias| ≤ 3 bpm; RMSSD passes on the rule used for children
     (ICC(2,1) ≥ 0.75 and |bias| ≤ 10 ms).

  python3 pipeline/cinc_validate.py
"""
import json, datetime
import numpy as np, wfdb
from scipy.signal import butter, filtfilt
from wfdb import processing
from paths import ROOT, META, FEATURES, WEB_DATA
from train_models import auc_ci, prop
from hrv_discover import agree, icc21, boot

SRC = ROOT / 'raw' / 'cinc2016' / 'cinc'
DB = {'a': 'MITHSDB — valve prolapse, benign murmurs, aortic disease', 'b': 'AADHSDB — coronary artery disease', 'c': 'AUTHHSDB — aortic stenosis, mitral regurgitation (echo-confirmed, ~75 y)',
      'd': 'UHAHSDB — mixed pathology, 44–90 y', 'e': 'DLUTHSDB — coronary artery disease, ~60 y', 'f': 'SUAHSDB — mixed pathology, 16–88 y'}


def times_rule(t):
    """= the page's hrvFromTimes: heart rate and RMSSD from beat times, same artefact rule."""
    rr = np.diff(np.asarray(t, float))
    if len(rr) < 4: return None, None
    md = np.median(rr); ok = np.abs(rr - md) <= 0.2 * md; clean = rr[ok]
    d = [rr[i] - rr[i - 1] for i in range(1, len(rr)) if ok[i] and ok[i - 1]]
    hr = 60 / clean.mean() if len(clean) else None
    rmssd = round(float(np.sqrt(np.mean(np.square(d)))) * 1000, 1) if len(d) >= 8 else None
    return hr, rmssd


def ecg_beats(name):
    """Reference beats. The stored ECG has a large DC offset and an arbitrary gain, so: band-pass 0.5–40 Hz, scale to the 99th
    percentile, wfdb gqrs on both polarities, keep the one with more beats. Records with < 8 ECG beats are excluded (counted)."""
    rec = wfdb.rdrecord(str(SRC / 'training-a' / name), channel_names=['ECG']); fs = rec.fs
    b, a = butter(2, [0.5 / (fs / 2), 40 / (fs / 2)], 'band'); f = filtfilt(b, a, rec.p_signal[:, 0]); f = f / max(1e-9, np.percentile(np.abs(f), 99))
    q = max((processing.gqrs_detect(sig=f, fs=fs), processing.gqrs_detect(sig=-f, fs=fs)), key=len)
    return np.asarray(q) / fs


def main():
    R = [json.loads(l) for l in open(FEATURES / 'cinc2016.jsonl')]
    thr = json.loads((WEB_DATA / 'cnn_murmur.json').read_text())['threshold']
    out = {'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source': 'PhysioNet/CinC Challenge 2016 training sets a–f (Liu et al., Physiol Meas 2016, doi 10.1088/0967-3334/37/12/2181), ODC-By 1.0',
           'zipSha256': 'ffb9135b4d5e943863c351f3477e9b3b721e4ae592b3202e9b098ce44edbfa88', 'threshold': thr, 'murmur': {}, 'n': len(R)}
    for good in (False, True):
        key = 'goodQuality' if good else 'all'; res = []
        for db in 'abcdef':
            S = [r for r in R if r['db'] == db and r['cnnP'] is not None and (r['sqi'] == 1 or not good)]
            y = np.array([r['y'] == 1 for r in S], int); p = np.array([r['cnnP'] for r in S])
            if len(S) < 10 or y.sum() < 5 or (1 - y).sum() < 5: continue
            fl = p >= thr
            res.append({'db': db, 'what': DB[db], 'recordings': len(S), 'abnormal': int(y.sum()), 'auc': auc_ci(y, p),
                        'sens': prop(int((fl & (y == 1)).sum()), int(y.sum())), 'spec': prop(int((~fl & (y == 0)).sum()), int((1 - y).sum()))})
        out['murmur'][key] = res
    prim = {r['db']: r for r in out['murmur']['all'] if r['db'] in 'ac'}
    out['murmur']['transfers'] = bool(len(prim) == 2 and all(r['auc']['lo'] > 0.70 for r in prim.values()))
    out['murmur']['skipped'] = sum(1 for r in R if r['cnnP'] is None)
    # heart rate and RMSSD against the simultaneous ECG (set a)
    pairs, noref = [], 0
    for r in R:
        if r['db'] != 'a' or not r.get('s1') or not (SRC / 'training-a' / f"{r['name']}.dat").exists(): continue
        try: eb = ecg_beats(r['name'])
        except Exception: eb = []
        if len(eb) < 8: noref += 1; continue
        hp, rp = times_rule(r['s1']); he, re_ = times_rule(eb)
        pairs.append({'name': r['name'], 'y': r['y'], 'hrPcg': hp, 'hrEcg': he, 'rmssdPcg': rp, 'rmssdEcg': re_})
    H = [(p['hrPcg'], p['hrEcg']) for p in pairs if p['hrPcg'] and p['hrEcg']]; V = [(p['rmssdPcg'], p['rmssdEcg']) for p in pairs if p['rmssdPcg'] is not None and p['rmssdEcg'] is not None]
    hr = agree([a for a, b in H], [b for a, b in H]); hr['pass'] = bool(hr['icc'] >= 0.90 and abs(hr['bias']) <= 3)
    hrv = agree([a for a, b in V], [b for a, b in V]) if len(V) >= 10 else None
    out['ecg'] = {'recordings': len(pairs), 'excludedNoEcgBeats': noref, 'heartRate': hr, 'rmssd': hrv, 'rmssdMeasurable': {'pcg': sum(p['rmssdPcg'] is not None for p in pairs), 'ecg': sum(p['rmssdEcg'] is not None for p in pairs)},
                  'pairs': pairs, 'rule': 'heart rate: ICC(2,1) ≥ 0.90 and |bias| ≤ 3 bpm; RMSSD: ICC(2,1) ≥ 0.75 and |bias| ≤ 10 ms'}
    (META / 'cinc2016_validation.json').write_text(json.dumps(out, indent=1))
    for r in out['murmur']['all']: print(f"{r['db']} {r['recordings']:5d} rec, {r['abnormal']:4d} abnormal  AUC {r['auc']}  sens {r['sens']['value']} spec {r['spec']['value']}")
    print('transfers to adult valve disease:', out['murmur']['transfers'])
    print('HR vs ECG', {k: hr[k] for k in ('n', 'icc', 'iccCI', 'bias', 'loa', 'pass')}); print('RMSSD vs ECG', hrv and {k: hrv[k] for k in ('n', 'icc', 'iccCI', 'bias', 'loa', 'pass')})


if __name__ == '__main__':
    main()
