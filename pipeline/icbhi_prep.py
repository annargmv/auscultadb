"""ICBHI 2017 Respiratory Sound Database → meta/icbhi_index.json + build/icbhi_wav/*.wav (8 kHz, 16-bit mono).

Source: raw/icbhi — a copy of the ICBHI 2017 challenge database (Rocha et al., Physiol Meas 2019;40:035001,
doi 10.1088/1361-6579/ab03ea), here from github.com/aashnajoshi/Respiratory_Sound_Dataset (itself the Kaggle copy).
Licence: the challenge released it for research use. Owner's rule 3: ICBHI may be ANALYSED, but its audio must never
reach web/ — no browsable patients, no audio bundles, no example clips. Only derived statistics and model weights.

126 patients, 920 recordings (4 stethoscopes/microphones: Meditron, Littmann C2SE, Littmann 3200, AKG C417L),
7 chest locations, every respiratory cycle annotated by experts: start, end, crackles 0/1, wheezes 0/1.
Recordings are resampled with a polyphase anti-alias filter to the platform's 8 kHz (the page analyses at SR = 8000).

Provenance: this came from a third-party re-upload, not from the challenge organisers, so the
content is fingerprinted here rather than trusted. The run records a manifest hash over every
source file (name + size + sha256) in meta/icbhi_index.json, and fails if the set of files
changes against a previously recorded hash unless ICBHI_ACCEPT_NEW=1 is set. Compare with
pipeline/fetch_circor.sh, where PhysioNet publishes checksums we can verify directly; no such
reference exists for this mirror, so the hash establishes reproducibility, not authenticity.

  python3 pipeline/icbhi_prep.py
"""
import csv, hashlib, json, os, collections
from math import gcd
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from paths import RAW, META, BUILD

SRC = RAW / 'icbhi'


def manifest(paths):
    """Stable fingerprint of the source files: name, size and sha256 of each, hashed in name order."""
    h, files = hashlib.sha256(), []
    for p in sorted(paths, key=lambda q: q.name):
        d = hashlib.sha256(p.read_bytes()).hexdigest()
        files.append({'name': p.name, 'bytes': p.stat().st_size, 'sha256': d})
        h.update(f'{p.name}:{p.stat().st_size}:{d}\n'.encode())
    return h.hexdigest(), files


def check_manifest(digest, n):
    """Refuse to rebuild from a source set that differs from the one already recorded."""
    prev = META / 'icbhi_index.json'
    if not prev.exists(): return
    try:
        old = (json.load(open(prev)).get('provenance') or {}).get('manifestSha256')
    except Exception:
        return
    if old and old != digest and os.environ.get('ICBHI_ACCEPT_NEW') != '1':
        raise SystemExit(
            f'ICBHI source files changed.\n  recorded: {old}\n  now:      {digest} ({n} files)\n'
            'The index on disk was built from different audio. Re-run with ICBHI_ACCEPT_NEW=1 to accept '
            'this copy and overwrite the recorded hash.')
AUD = SRC / 'audio_and_txt_files'
OUT = BUILD / 'icbhi_wav'
SR = 8000
CHILD_AGE = 18


def num(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not SRC.exists() or not AUD.exists():
        raise SystemExit(
            f'no ICBHI data at {SRC}.\n'
            '  The challenge host (bhichallenge.med.auth.gr) is unreachable and it is not on PhysioNet;\n'
            '  this index was built from github.com/aashnajoshi/Respiratory_Sound_Dataset (the Kaggle copy).\n'
            '  Put patient_diagnosis.csv and audio_and_txt_files/ under raw/icbhi and re-run.\n'
            '  Rule 3: ICBHI may be analysed, but its audio must never reach web/.')
    D = {r['Patient_ID']: r for r in csv.DictReader(open(SRC / 'patient_diagnosis.csv'))}
    pats = collections.OrderedDict()
    srs = collections.Counter(); bad = [0]
    for w in sorted(AUD.glob('*.wav')):
        name = w.stem; pid, rec, loc, mode, dev = name.split('_')
        rows = [l.split() for l in open(w.with_suffix('.txt')) if l.strip()]
        bad[0] += sum(1 for r in rows if len(r) != 4)          # one row in the release has 5 values: ambiguous, dropped
        cyc = [[round(float(a), 3), round(float(b), 3), int(c), int(d)] for a, b, c, d in (r for r in rows if len(r) == 4)]
        sr, x = wavfile.read(w); srs[sr] += 1
        x = x.astype(np.float64)
        if x.ndim > 1: x = x.mean(1)
        x /= max(1e-9, np.abs(x).max())
        if sr != SR: g = gcd(sr, SR); x = resample_poly(x, SR // g, sr // g)
        y = np.clip(np.round(x / max(1e-9, np.abs(x).max()) * 0.9 * 32767), -32768, 32767).astype(np.int16)
        out = OUT / f'{name}.wav'
        if not out.exists(): wavfile.write(out, SR, y)
        r = D[pid]
        p = pats.setdefault(pid, {'pid': pid, 'dx': r['DIAGNOSIS'], 'age': num(r['AGE']), 'sex': r['GENDER'] if r['GENDER'] in ('M', 'F') else None,
                                  'bmi': num(r['BMI (Kg/m2)']), 'recs': []})
        p['recs'].append({'name': name, 'rec': rec, 'loc': loc, 'mode': mode, 'dev': dev, 'srOrig': sr, 'dur': round(len(y) / SR, 2), 'cycles': cyc})
    P = list(pats.values())
    for p in P:
        p['group'] = None if p['age'] is None else ('child' if p['age'] < CHILD_AGE else 'adult')
        p['devs'] = sorted({r['dev'] for r in p['recs']})
    srcFiles = sorted(AUD.glob('*.wav')) + sorted(AUD.glob('*.txt'))
    digest, files = manifest(srcFiles)
    check_manifest(digest, len(srcFiles))
    out = {'source': 'ICBHI 2017 Respiratory Sound Database (Rocha et al., Physiol Meas 2019;40:035001)', 'licence': 'research use (ICBHI challenge); audio never published',
           'via': 'github.com/aashnajoshi/Respiratory_Sound_Dataset (copy of the Kaggle release)', 'sr': SR, 'droppedRows': bad[0],
           'provenance': {'manifestSha256': digest, 'nSourceFiles': len(srcFiles),
                          'note': 'sha256 over name+size+content of every source file, in name order. '
                                  'The mirror publishes no official checksum, so this fingerprints the copy '
                                  'that produced these numbers; it does not authenticate it against the organisers.',
                          'files': files},
           'patients': P}
    (META / 'icbhi_index.json').write_text(json.dumps(out, separators=(',', ':')))
    nc = sum(len(r['cycles']) for p in P for r in p['recs'])
    print(f"  source manifest sha256 {digest[:16]}… over {len(srcFiles)} files")
    print(f"{len(P)} patients, {sum(len(p['recs']) for p in P)} recordings, {nc} annotated cycles ({bad[0]} malformed row dropped); original rates {dict(srs)} -> {OUT}")
    print(collections.Counter((p['dx'], p['group']) for p in P))


if __name__ == '__main__':
    main()
