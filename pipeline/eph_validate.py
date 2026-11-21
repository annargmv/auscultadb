"""Heart rate and HRV from heart sounds against a research-grade ECG in ADULTS — EPHNOGRAM → meta/eph_validation.json.

EPHNOGRAM 1.0.0 (Kazemnejad et al., PhysioNet, doi 10.13026/tjtq-5911; ODC-By 1.0): simultaneous ECG and PCG from one
device at 8 kHz, healthy men aged 23–29. Second adult test, run because the CinC 2016 ECG (set a) gave implausible resting
rates in part of its records; the rules are the SAME as there and were fixed before this run:
  * records: every RESTING record whose spreadsheet notes say ECG "Good" and PCG not saturated (14 records, 12 men);
  * unit: consecutive 30-s windows from the start of each record (a 30-s record is one window);
  * reference beats: wfdb gqrs on the band-passed ECG, both polarities, the one with more beats; windows with < 8 ECG beats
    are excluded and counted;
  * heart sounds: the page's learned beat finder (segmentHeartSounds, trained on CirCor children, frozen) on the PCG channel;
  * both beat series go through the page's interval rule (hrvFromTimes); heart rate passes if ICC(2,1) ≥ 0.90 and
    |bias| ≤ 3 bpm; RMSSD passes if ICC(2,1) ≥ 0.75 and |bias| ≤ 10 ms. Windows from one man are not independent: the
    record-level agreement (median of each record's windows) is reported beside the window-level one.

  python3 pipeline/eph_validate.py prep     # 30-s PCG windows -> build/eph/*.wav, ECG beats -> build/eph/ecg.json
  node pipeline/eph_run.js                  # page's beat finder on each window -> meta/features/eph.jsonl
  python3 pipeline/eph_validate.py          # agreement -> meta/eph_validation.json
"""
import json, sys, datetime
import numpy as np, pandas as pd, wfdb
from wfdb import processing
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from paths import ROOT, META, FEATURES
from hrv_discover import agree
from cinc_validate import times_rule

SRC = ROOT / 'raw' / 'ephnogram'; OUT = ROOT / 'build' / 'eph'; WIN = 30


def records():
    d = pd.read_csv(SRC / 'ECGPCGSpreadsheet.csv').iloc[:, :12].dropna(subset=['Record Name'])
    r = d[d['Recording Scenario'].str.startswith('Rest', na=False) & (d['ECG Notes'].str.strip() == 'Good') & ~d['PCG Notes'].str.contains('aturat', na=False)]
    return [(x['Record Name'], x['Subject ID'], x['Age (years)']) for _, x in r.iterrows()]


def gqrs(sig, fs):
    b, a = butter(2, [0.5 / (fs / 2), 40 / (fs / 2)], 'band'); f = filtfilt(b, a, sig); f = f / max(1e-9, np.percentile(np.abs(f), 99))
    return np.asarray(max((processing.gqrs_detect(sig=f, fs=fs), processing.gqrs_detect(sig=-f, fs=fs)), key=len)) / fs


def prep():
    OUT.mkdir(parents=True, exist_ok=True); E = {}
    for name, subj, age in records():
        rec = wfdb.rdrecord(str(SRC / 'WFDB' / name)); fs = rec.fs; ch = {n: i for i, n in enumerate(rec.sig_name)}
        ecg, pcg = rec.p_signal[:, ch['ECG']], rec.p_signal[:, ch['PCG']]; n = int(len(ecg) // (WIN * fs))
        for k in range(max(1, n)):
            a, b = k * WIN * fs, min(len(ecg), (k + 1) * WIN * fs); w = f'{name}_w{k:02d}'
            x = pcg[a:b] - np.median(pcg[a:b]); x = np.clip(x / max(1e-9, np.percentile(np.abs(x), 99.95)) * 0.8, -1, 1)
            wavfile.write(OUT / f'{w}.wav', fs, (x * 32767).astype(np.int16))
            E[w] = {'record': name, 'subject': subj, 'age': None if pd.isna(age) else float(age), 'ecg': gqrs(ecg[a:b], fs).round(4).tolist()}
        print(name, subj, n, 'windows')
    (OUT / 'ecg.json').write_text(json.dumps(E)); print(len(E), 'windows ->', OUT)


def main():
    E = json.loads((OUT / 'ecg.json').read_text()); S = {json.loads(l)['name']: json.loads(l) for l in open(FEATURES / 'eph.jsonl')}
    rows, noref = [], 0
    for w, e in E.items():
        if len(e['ecg']) < 8: noref += 1; continue
        s = S.get(w); hp, rp = times_rule(s['s1']) if s and s.get('s1') else (None, None); he, re_ = times_rule(e['ecg'])
        rows.append({'window': w, 'record': e['record'], 'subject': e['subject'], 'hrPcg': hp, 'hrEcg': he, 'rmssdPcg': rp, 'rmssdEcg': re_})
    H = [r for r in rows if r['hrPcg'] and r['hrEcg']]; V = [r for r in rows if r['rmssdPcg'] is not None and r['rmssdEcg'] is not None]
    hr = agree([r['hrPcg'] for r in H], [r['hrEcg'] for r in H]); hr['pass'] = bool(hr['icc'] >= 0.90 and abs(hr['bias']) <= 3)
    hrv = agree([r['rmssdPcg'] for r in V], [r['rmssdEcg'] for r in V]) if len(V) >= 10 else None
    per = []
    for rec in sorted({r['record'] for r in rows}):
        q = [r for r in H if r['record'] == rec]; v = [r for r in V if r['record'] == rec]
        if q: per.append({'record': rec, 'subject': q[0]['subject'], 'windows': len(q), 'hrPcg': round(float(np.median([r['hrPcg'] for r in q])), 1), 'hrEcg': round(float(np.median([r['hrEcg'] for r in q])), 1),
                          'rmssdPcg': round(float(np.median([r['rmssdPcg'] for r in v])), 1) if v else None, 'rmssdEcg': round(float(np.median([r['rmssdEcg'] for r in v])), 1) if v else None})
    recHr = agree([p['hrPcg'] for p in per], [p['hrEcg'] for p in per]) if len(per) >= 5 else None
    pv = [p for p in per if p['rmssdPcg'] is not None and p['rmssdEcg'] is not None]; recHrv = agree([p['rmssdPcg'] for p in pv], [p['rmssdEcg'] for p in pv]) if len(pv) >= 5 else None
    out = {'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source': 'EPHNOGRAM 1.0.0 (PhysioNet, doi 10.13026/tjtq-5911), ODC-By 1.0',
           'records': len(per), 'subjects': len({p['subject'] for p in per}), 'windows': len(rows), 'excludedNoEcgBeats': noref,
           'heartRate': hr, 'rmssd': hrv, 'perRecord': per, 'recordLevel': {'heartRate': recHr, 'rmssd': recHrv},
           'rmssdMeasurable': {'pcg': sum(r['rmssdPcg'] is not None for r in rows), 'ecg': sum(r['rmssdEcg'] is not None for r in rows)},
           'points': [[round(r['hrEcg'], 1), round(r['hrPcg'], 1)] for r in H], 'rule': 'heart rate: ICC(2,1) ≥ 0.90 and |bias| ≤ 3 bpm; RMSSD: ICC(2,1) ≥ 0.75 and |bias| ≤ 10 ms'}
    (META / 'eph_validation.json').write_text(json.dumps(out, indent=1))
    k = ('n', 'icc', 'iccCI', 'bias', 'loa', 'pass')
    print(out['records'], 'records,', out['subjects'], 'men,', out['windows'], 'windows, excluded', noref)
    print('HR', {q: hr[q] for q in k}); print('RMSSD', hrv and {q: hrv[q] for q in k}); print('record-level HR', recHr and {q: recHr[q] for q in k}, '\nrecord-level RMSSD', recHrv and {q: recHrv[q] for q in k})


if __name__ == '__main__':
    prep() if sys.argv[1:] == ['prep'] else main()
