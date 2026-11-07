"""Prepare the KAUH lung-sound dataset (Fraiwan et al., Mendeley Data jwyy9np4gv v3, CC BY 4.0).

Input : raw/kauh/  containing "Audio Files.zip" (or Audio_Files.zip, or an already-extracted folder of WAVs)
Output: build/kauh_wav/KAUH_<mode><nnn>.wav   (safe file names; mode B=bell, D=diaphragm, E=extended)
        meta/kauh_cohort.json                (one patient per subject, the 3 filter modes as 3 'sites')

File names encode the labels:  BP12_Asthma,E W,P L M,12,F.wav
  = <mode>P<subject>_<diagnosis>,<sound type>,<location>,<age>,<sex>
Sound type: N normal, W wheeze, C / Crep crackles, B bronchial, I/E inspiratory/expiratory.
Location:   P/A posterior/anterior, L/R left/right, U/M/L upper/middle/lower (order not always consistent).
Baseline:   for every subject, the nearest-age healthy subject (same chest zone and sex preferred).
"""
import collections, json, re, shutil, sys, zipfile
from paths import KAUH_RAW, BUILD, META

DX = {'n': 'Healthy adult (control)', 'asthma': 'Asthma', 'heart failure': 'Heart failure', 'copd': 'COPD', 'pneumonia': 'Pneumonia (adult)',
      'lung fibrosis': 'Lung fibrosis', 'bron': 'Bronchitis (adult)', 'heart failure + copd': 'Heart failure + COPD', 'plueral effusion': 'Pleural effusion',
      'asthma and lung fibrosis': 'Asthma + lung fibrosis', 'heart failure + lung fibrosis': 'Heart failure + lung fibrosis'}
MODE = {'B': 'bell', 'D': 'diaphragm', 'E': 'extended'}
PAT = re.compile(r'^([BDE])P(\d+)_(.*)\.wav$', re.I)


def label(st):
    t = st.upper().replace('CREP', 'C').split()
    w, c, b = 'W' in t, 'C' in t, ('B' in t or 'BRONCHIAL' in t)
    if t == ['N']: return 'Normal'
    if w and c: return 'CAS & DAS'
    if w: return 'CAS'
    if c: return 'DAS'
    return 'Bronchial' if b else 'Other'


def where(loc):
    t = loc.upper().split(); plane = {'P': 'posterior', 'A': 'anterior'}.get(t[0], '?'); r = t[1:]; side = zone = None
    if len(r) == 2 and r[0] in 'LR' and r[1] in 'UML': side, zone = r
    elif len(r) == 2 and r[0] in 'UM' and r[1] in 'LR': zone, side = r
    elif r == ['L', 'R']: zone, side = 'L', 'R'
    elif r == ['L', 'L', 'R']: zone, side = 'L', 'B'          # ambiguous code: read as lower zone, both sides
    code = t[0] + (side or 'X') + (zone or 'X')
    txt = f"{plane.capitalize()} {({'L': 'left', 'R': 'right', 'B': 'left and right'}.get(side, 'side not given'))} {({'U': 'upper', 'M': 'middle', 'L': 'lower'}.get(zone, ''))}"
    return code, ' '.join(txt.split()) + ' chest', plane


def source_wavs():
    wavs = [p for p in KAUH_RAW.rglob('*.wav') if PAT.match(p.name)]
    if wavs: return wavs
    zips = [z for z in KAUH_RAW.glob('*.zip') if 'audio' in z.name.lower()]
    if not zips: sys.exit(f'no KAUH WAVs or "Audio Files.zip" in {KAUH_RAW}')
    out = BUILD / 'kauh_src'; out.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(zips[0]).extractall(out)
    return [p for p in out.rglob('*.wav') if PAT.match(p.name)]


def main():
    dst = BUILD / 'kauh_wav'; dst.mkdir(parents=True, exist_ok=True)
    P = {}
    for f in sorted(source_wavs(), key=lambda p: p.name):
        m = PAT.match(f.name); mode, n = m.group(1).upper(), int(m.group(2))
        dx, st, loc, age, sex = [x.strip() for x in m.group(3).split(',')]
        p = P.setdefault(n, {'n': n, 'disease': DX[dx.lower()], 'sound': st, 'label': label(st), 'locRaw': loc, 'age': int(age), 'sex': sex.upper(), 'files': {}})
        name = f'KAUH_{mode}{n:03d}'; shutil.copy(f, dst / (name + '.wav')); p['files'][mode] = name
    for p in P.values():
        p['zone'], p['siteTxt'], p['plane'] = where(p['locRaw'])
    ctrl = [p for p in P.values() if p['label'] == 'Normal']
    for p in P.values():
        pool = [c for c in ctrl if c['n'] != p['n']]
        score = lambda c: abs(c['age'] - p['age']) + (0 if c['zone'] == p['zone'] else 12 if c['plane'] == p['plane'] else 25) + (0 if c['sex'] == p['sex'] else 4)
        p['ref'] = 'K%03d' % min(pool, key=score)['n']
    cohort = [{'key': 'K%03d' % p['n'], 'pid': 'BP%d' % p['n'], 'src': 'kauh', 'age': p['age'], 'g': '0' if p['sex'] == 'M' else '1', 'disease': p['disease'],
               'ref': p['ref'], 'sound': p['sound'], 'locRaw': p['locRaw'],
               'sites': {f"{p['zone']}-{m}": {'label': p['label'], 'labelSrc': 'record', 'name': p['files'][m], 'dir': 'KAUH', 'ev': [],
                                             'site': f"{p['siteTxt']} · {MODE[m]} mode", 'zone': p['zone'], 'mode': m} for m in 'DBE'}}
              for p in sorted(P.values(), key=lambda x: x['n'])]
    json.dump(cohort, open(META / 'kauh_cohort.json', 'w'))
    print(len(cohort), 'subjects;', dict(collections.Counter(c['disease'] for c in cohort)))


if __name__ == '__main__':
    main()
