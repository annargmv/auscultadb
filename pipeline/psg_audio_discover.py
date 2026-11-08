"""Score the page's acoustic apnea / AHI against the PSG on PSG-Audio → meta/psg_audio_validation.json and
discover.json['apnea'] (Discoveries 12). Reads meta/psg_audio_index.json (expert scoring, sleep time) and
meta/features/psg_audio.jsonl (what the page scored, from psg_audio_validate.js).

The metrics were fixed before any result was seen, and nothing is tuned (rule 4):
  event level, per microphone, pooled over all analysed hours
    sensitivity = expert respiratory events (OA, CA, MA, H) overlapped by any scored acoustic event
    PPV         = scored acoustic events overlapping any expert event
    also by expert type, and apnea-vs-hypopnea agreement on matched pairs
  patient level
    PSG AHI       = expert events in the analysed hours / scored sleep time (h) — the reference
    acoustic AHI  = scored acoustic events / recording time (h) — sound has no sleep time, so this is what a
                    sound-only device would report; also shown per PSG sleep hour to separate the TST effect
    agreement     = Spearman ρ, Bland–Altman bias and 95 % limits, severity class (adult 5 / 15 / 30) exact
                    agreement and confusion table
  'candidate' rows and events flagged `edge` (touching the end of the hour) are not counted — exactly as the page's
  own AHI excludes them.
Small pilot: report counts, CIs and the cohort's skew (mostly severe OSA) with every number.

  python3 pipeline/psg_audio_discover.py      (then merge.py --bump)
"""
import collections, json, datetime
import numpy as np
from scipy import stats
import wave
from scipy.signal import butter, sosfilt
from paths import META, FEATURES, WEB_DATA, RAW

SEV = [(30, 'severe'), (15, 'moderate'), (5, 'mild'), (0, 'normal')]
CLASSES = ['normal', 'mild', 'moderate', 'severe']
sev = lambda ahi: next(lab for thr, lab in SEV if ahi >= thr)
overlap = lambda a0, a1, b0, b1: a0 < b1 and b0 < a1
rng = np.random.default_rng(20261108)
SOS = butter(2, [150, 800], btype='band', fs=8000, output='sos')   # the band computeFlow listens to


def headroom(wav):
    """Breathing level over the channel's noise floor: median / 5th percentile of 1-s band RMS. Descriptive only —
    explains where the flow proxy can and cannot see a 30 % drop; it is not used to select or score anything."""
    f = RAW / 'psg_audio' / wav
    if not f.exists(): return None
    w = wave.open(str(f)); x = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float32)
    y = sosfilt(SOS, x); n = len(y) // 8000
    if n < 60: return None
    r = np.sqrt(np.mean(y[:n * 8000].reshape(n, 8000) ** 2, 1))
    return float(np.median(r) / max(np.percentile(r, 5), 1e-9))


def boot_ci(f, n, B=2000):
    vals = [f(rng.integers(0, n, n)) for _ in range(B)]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)] if vals else None


def main():
    idx = json.load(open(META / 'psg_audio_index.json'))
    rows = [json.loads(l) for l in open(FEATURES / 'psg_audio.jsonl')]
    acc = {(r['pid'], r['k'], r['ch']): r for r in rows}
    chans = sorted({r['ch'] for r in rows})
    out = {'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source': idx['source'], 'channels': {}}
    for ch in chans:
        ev_tot = collections.Counter(); ev_hit = collections.Counter(); det_tot = det_hit = 0
        typ = collections.Counter(); pats = []; hourRows = []
        for p in idx['patients']:
            hrs = [h for h in p['hours'] if (p['pid'], h['k'], ch) in acc]
            if not hrs: continue
            n_exp = n_det = 0; rec = sleep = 0.0
            for h in hrs:
                a = acc[(p['pid'], h['k'], ch)]
                det = [e for e in a['events'] if e['type'] in ('apnea', 'hypopnea') and not e['edge']]   # as the page counts them
                exp = [(e['start'], e['start'] + e['dur'], e['type']) for e in h['events']]
                for s, e, t in exp:
                    ev_tot[t] += 1
                    m = [d for d in det if overlap(s, e, d['start'], d['end'])]
                    if m:
                        ev_hit[t] += 1
                        typ[('apnea' if t != 'H' else 'hypopnea', m[0]['type'])] += 1
                det_tot += len(det); det_hit += sum(1 for d in det if any(overlap(s, e, d['start'], d['end']) for s, e, _ in exp))
                n_exp += len(exp); n_det += len(det); rec += a['dur']; sleep += h['sleep']
                if len(exp) >= 5:
                    hit = sum(1 for s, e, _ in exp if any(overlap(s, e, d['start'], d['end']) for d in det))
                    hourRows.append({'pid': p['pid'], 'k': h['k'], 'headroom': headroom(h['wav'][ch]), 'sens': hit / len(exp), 'n': len(exp)})
            if sleep < 3600: continue        # AASM: no AHI under an hour of sleep
            pats.append({'pid': p['pid'], 'hours': round(rec / 3600, 2), 'sleepH': round(sleep / 3600, 2), 'nExpert': n_exp, 'nAcoustic': n_det,
                         'ahiPsg': round(n_exp / (sleep / 3600), 1), 'ahiAcoustic': round(n_det / (rec / 3600), 1),
                         'ahiAcousticPerSleepH': round(n_det / (sleep / 3600), 1)})
        for q in pats: q['sevPsg'], q['sevAcoustic'] = sev(q['ahiPsg']), sev(q['ahiAcoustic'])
        E = sum(ev_tot.values()); H = sum(ev_hit.values())
        x = np.array([q['ahiPsg'] for q in pats]); y = np.array([q['ahiAcoustic'] for q in pats])
        res = {'patients': len(pats), 'hours': round(sum(q['hours'] for q in pats), 1),
               'events': {'expert': E, 'found': H, 'sens': round(100 * H / E, 1) if E else None,
                          'acoustic': det_tot, 'matched': det_hit, 'ppv': round(100 * det_hit / det_tot, 1) if det_tot else None,
                          'byType': {t: {'n': ev_tot[t], 'found': ev_hit[t], 'sens': round(100 * ev_hit[t] / ev_tot[t], 1) if ev_tot[t] else None} for t in ['OA', 'CA', 'MA', 'H']},
                          'typeAgreement': {f'{a}->{b}': n for (a, b), n in typ.items()}},
               'perPatient': pats}
        hr = [q for q in hourRows if q['headroom'] is not None]
        if len(hr) >= 6:
            rho_h, p_h = stats.spearmanr([q['headroom'] for q in hr], [q['sens'] for q in hr])
            med = float(np.median([q['headroom'] for q in hr]))
            lo, hi = [q for q in hr if q['headroom'] < med], [q for q in hr if q['headroom'] >= med]
            res['headroom'] = {'hours': len(hr), 'rho': round(float(rho_h), 3), 'p': float(p_h), 'median': round(med, 2),
                               'sensLow': round(100 * float(np.mean([q['sens'] for q in lo])), 1) if lo else None,
                               'sensHigh': round(100 * float(np.mean([q['sens'] for q in hi])), 1) if hi else None,
                               'rows': [{**q, 'headroom': round(q['headroom'], 2), 'sens': round(q['sens'], 3)} for q in hr]}
        if len(pats) >= 3:
            rho, p = stats.spearmanr(x, y); d = y - x
            res['ahi'] = {'rho': round(float(rho), 3), 'p': float(p), 'rhoCI': boot_ci(lambda i: stats.spearmanr(x[i], y[i])[0] if len(set(x[i])) > 1 else None, len(x)),
                          'bias': round(float(d.mean()), 1), 'loa': [round(float(d.mean() - 1.96 * d.std(ddof=1)), 1), round(float(d.mean() + 1.96 * d.std(ddof=1)), 1)],
                          'medianPsg': float(np.median(x)), 'medianAcoustic': float(np.median(y))}
            res['severity'] = {'agree': sum(q['sevPsg'] == q['sevAcoustic'] for q in pats), 'n': len(pats),
                               'confusion': {a: {b: sum(1 for q in pats if q['sevPsg'] == a and q['sevAcoustic'] == b) for b in CLASSES} for a in CLASSES}}
        out['channels'][ch] = res
    json.dump(out, open(META / 'psg_audio_validation.json', 'w'), indent=1)
    DJ = WEB_DATA / 'discover.json'; disc = json.load(open(DJ)); disc['apnea'] = out
    DJ.write_text(json.dumps(disc, separators=(',', ':')))
    for ch, r in out['channels'].items():
        e = r['events']; a = r.get('ahi', {})
        print(f"{ch:6} {r['patients']} patients {r['hours']} h | events sens {e['sens']} % ({e['found']}/{e['expert']}), PPV {e['ppv']} % ({e['matched']}/{e['acoustic']})"
              f" | AHI rho {a.get('rho')} bias {a.get('bias')} LoA {a.get('loa')} | severity {r.get('severity', {}).get('agree')}/{r['patients']}")
    print('-> meta/psg_audio_validation.json + discover.json[apnea]   (then merge.py --bump)')


if __name__ == '__main__':
    main()
