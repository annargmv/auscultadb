"""Prepare the CirCor DigiScope phonocardiogram dataset (PhysioNet 1.0.3, ODC-By).

Input : raw/circor/circor-heart-sound-1.0.3/  (training_data/, training_data.csv)
Output: meta/circor_index.json   all 942 patients / 3,163 recordings — the validation set
        meta/circor_cohort.json  the browsable cohort (~120 patients, the 4 valve sites as 'sites')

The WAV file names are already safe (`<patient>_<VALVE>.wav`), so nothing is copied:
the pipeline reads raw/ directly. Files are 4 kHz; parseWav resamples to SR = 8000 on load.

Labels, per patient, from training_data.csv:
  Murmur                  Present / Absent / Unknown
  Murmur locations        which valves carry it  -> per-site ground truth
  Most audible location   single valve
  Systolic / Diastolic    timing, shape, grading, pitch, quality
  Outcome                 Normal / Abnormal (the clinical outcome, not the murmur)
Age is categorical; AGE_Y maps it to a representative age in years because the
cardiac rules in web/index.html gate on a numeric age (heart-rate range per age band).

Baseline: every patient needs a murmur-free reference for compareRecordings. We use the
nearest control (Murmur Absent AND Outcome Normal), preferring the same age band and sex.
"""
import collections, csv, json, re, sys
from paths import CIRCOR_RAW, META

VALVES = ['AV', 'PV', 'TV', 'MV']                       # aortic, pulmonary, tricuspid, mitral
VALVE_TXT = {'AV': 'Aortic valve · 2nd right intercostal space', 'PV': 'Pulmonary valve · 2nd left intercostal space',
             'TV': 'Tricuspid valve · 4th left intercostal space', 'MV': 'Mitral valve · 5th left intercostal space, apex'}
AGE_Y = {'Neonate': 0.04, 'Infant': 0.5, 'Child': 6, 'Adolescent': 14}   # representative age per band, for display
# The SPAN of each band. The cardiac rules gate on an age-banded normal heart rate, and collapsing a
# band to one number mis-gates its edges: "Child" covers 1–11 y, whose normal HR runs 90–150 at the
# young end but only 70–120 at the old end. Passing the span lets the gate take the union.
AGE_SPAN = {'Neonate': (0.0, 0.08), 'Infant': (0.08, 1.0), 'Child': (1.0, 11.0), 'Adolescent': (12.0, 18.0)}
NA = ('nan', '', 'NA')


def nn(v):
    """CSV blanks arrive as the string 'nan'."""
    return None if v is None or v.strip() in NA else v.strip()


def root():
    d = CIRCOR_RAW / 'circor-heart-sound-1.0.3'
    if not d.exists():                                   # tolerate an extra nesting level from the downloader
        c = [p for p in CIRCOR_RAW.glob('**/training_data.csv')]
        if not c: sys.exit(f'no CirCor data under {CIRCOR_RAW} — see pipeline/fetch_circor.sh')
        d = c[0].parent
    return d


def read_patients(d):
    td = d / 'training_data'
    out = []
    for r in csv.DictReader(open(d / 'training_data.csv')):
        pid = r['Patient ID'].strip()
        murLoc = set((nn(r['Murmur locations']) or '').split('+')) - {''}
        p = {'pid': pid, 'key': 'C' + pid, 'src': 'circor',
             'ageBand': nn(r['Age']), 'age': AGE_Y.get(nn(r['Age'])),
             'ageLo': (AGE_SPAN.get(nn(r['Age'])) or (None, None))[0], 'ageHi': (AGE_SPAN.get(nn(r['Age'])) or (None, None))[1],
             'g': '0' if nn(r['Sex']) == 'Male' else '1',
             'height': nn(r['Height']), 'weight': nn(r['Weight']),
             'murmur': nn(r['Murmur']) or 'Unknown', 'murLoc': sorted(murLoc),
             'audible': nn(r['Most audible location']), 'outcome': nn(r['Outcome']), 'campaign': nn(r['Campaign']),
             'sys': {k: nn(r['Systolic murmur ' + k]) for k in ('timing', 'shape', 'grading', 'pitch', 'quality')},
             'dia': {k: nn(r['Diastolic murmur ' + k]) for k in ('timing', 'shape', 'grading', 'pitch', 'quality')},
             'files': {}}
        # the primary recording per valve; ignore repeat takes (_1/_2/_3) and the extra 'Phc' site
        for v in VALVES:
            f = td / f'{pid}_{v}.wav'
            if f.exists(): p['files'][v] = f'{pid}_{v}'
        if p['files']: out.append(p)
    return out


def disease(p):
    """A short label for the patient list, in the same spirit as the KAUH 'disease' field."""
    if p['murmur'] == 'Present':
        g, t = p['sys']['grading'], p['sys']['timing']
        if p['dia']['timing']: return 'Systolic + diastolic murmur' if t else 'Diastolic murmur'
        return f"Murmur {g}" if g else 'Murmur'
    if p['murmur'] == 'Unknown': return 'Murmur status unknown'
    return 'No murmur (control)' if p['outcome'] == 'Normal' else 'No murmur, abnormal outcome'


def site_label(p, v):
    """Per-site ground truth: does this valve carry the murmur?"""
    if p['murmur'] == 'Unknown': return 'Unknown'
    return 'Murmur' if v in p['murLoc'] else 'Normal'


def attach_refs(P, pool_from=None):
    """Nearest murmur-free control per patient: same age band and sex preferred.

    `pool_from` restricts the controls to a subset (used so that every cohort patient's
    baseline is itself in the cohort — precompute.js resolves `ref` against the cohort only).
    """
    src = pool_from if pool_from is not None else P
    ctrl = [c for c in src if c['murmur'] == 'Absent' and c['outcome'] == 'Normal' and len(c['files']) >= 2]
    if not ctrl: sys.exit('no usable controls')
    for p in P:
        pool = [c for c in ctrl if c['pid'] != p['pid']]
        pa = p['age'] if p['age'] is not None else 6
        def score(c):
            ca = c['age'] if c['age'] is not None else 6
            return (abs(ca - pa), 0 if c['g'] == p['g'] else 1, -len(c['files']), c['pid'])
        p['ref'] = min(pool, key=score)['key']


def sites(p):
    return {v: {'label': site_label(p, v), 'labelSrc': 'record', 'name': p['files'][v], 'dir': 'CIRCOR',
                'ev': [], 'site': VALVE_TXT[v], 'zone': v, 'mode': None,
                'audible': p['audible'] == v}
            for v in VALVES if v in p['files']}


def select(P, target=120, n_ctrl=45):
    """Deterministic, stratified browsable cohort. Rare cells first, then fill.

    Everything rare is taken whole: the 5 diastolic murmurs, every grade and systolic
    timing, the neonates. Then controls and common murmurs are added to `target`,
    preferring patients with all 4 valve sites recorded.
    """
    rank = lambda p: (-len(p['files']), p['pid'])     # 4 sites first, then stable by id
    chosen, seen = [], set()

    def take(pool, n=None):
        for p in sorted(pool, key=rank):
            if n is not None and n <= 0: break
            if p['key'] in seen: continue
            seen.add(p['key']); chosen.append(p)
            if n is not None: n -= 1

    take([p for p in P if p['dia']['timing']])                                   # all 5 diastolic
    for g in sorted({p['sys']['grading'] for p in P if p['sys']['grading']}):    # every grade
        take([p for p in P if p['sys']['grading'] == g], 5)
    for t in sorted({p['sys']['timing'] for p in P if p['sys']['timing']}):      # every systolic timing
        take([p for p in P if p['sys']['timing'] == t], 5)
    for b in ('Neonate', 'Infant', 'Adolescent', 'Child'):                       # every age band, murmur present
        take([p for p in P if p['ageBand'] == b and p['murmur'] == 'Present'], 4)
    take([p for p in P if p['murmur'] == 'Unknown'], 5)
    murmurs = len(chosen)
    for b in ('Neonate', 'Infant', 'Adolescent', 'Child'):                       # age-spread controls
        take([p for p in P if p['ageBand'] == b and p['murmur'] == 'Absent' and p['outcome'] == 'Normal'], 6)
    take([p for p in P if p['murmur'] == 'Absent' and p['outcome'] == 'Abnormal'], 6)   # abnormal outcome, no murmur
    # top up with controls until the cohort holds enough murmur-free baselines
    take([p for p in P if p['murmur'] == 'Absent' and p['outcome'] == 'Normal'],
         max(0, min(target - len(chosen), n_ctrl - (len(chosen) - murmurs))))
    take([p for p in P if p['murmur'] == 'Present'], max(0, target - len(chosen)))
    return sorted(chosen, key=lambda p: p['pid'])[:target]


def row(p):
    return {'key': p['key'], 'pid': p['pid'], 'src': 'circor', 'age': p['age'], 'ageBand': p['ageBand'],
            'ageLo': p['ageLo'], 'ageHi': p['ageHi'], 'g': p['g'],
            'disease': disease(p), 'ref': p['ref'], 'murmur': p['murmur'], 'murLoc': p['murLoc'], 'audible': p['audible'],
            'outcome': p['outcome'], 'sys': p['sys'], 'dia': p['dia'], 'campaign': p['campaign'],
            'sound': p['sys']['timing'] or ('none' if p['murmur'] == 'Absent' else p['murmur']),
            'locRaw': '+'.join(sorted(p['files'])), 'sites': sites(p)}


def main():
    d = root()
    P = read_patients(d)
    attach_refs(P)
    index = {'dataset': 'circor-heart-sound-1.0.3', 'licence': 'ODC-By',
             'dir': str((d / 'training_data').relative_to(d.parents[2])),
             'patients': [row(p) for p in P]}
    json.dump(index, open(META / 'circor_index.json', 'w'))
    sel = select(P)
    attach_refs(sel, pool_from=sel)          # baselines must live inside the cohort
    keys = {p['key'] for p in sel}
    missing = [p['key'] for p in sel if p['ref'] not in keys]
    if missing: sys.exit(f'refs outside the cohort: {missing[:5]}')
    cohort = [row(p) for p in sel]
    json.dump(cohort, open(META / 'circor_cohort.json', 'w'))
    nrec = sum(len(p['sites']) for p in index['patients'])
    ncsv = sum(1 for _ in csv.DictReader(open(d / 'training_data.csv')))
    if ncsv != len(index['patients']): print(f'  note: {ncsv - len(index["patients"])} of {ncsv} CSV patients have no primary valve recording')
    print(f"index : {len(index['patients'])} patients, {nrec} recordings -> meta/circor_index.json")
    print(f"cohort: {len(cohort)} patients, {sum(len(c['sites']) for c in cohort)} recordings -> meta/circor_cohort.json")
    print('  murmur :', dict(collections.Counter(c['murmur'] for c in cohort)))
    print('  age    :', dict(collections.Counter(c['ageBand'] for c in cohort)))
    print('  grading:', dict(collections.Counter(c['sys']['grading'] for c in cohort if c['sys']['grading'])))
    print('  timing :', dict(collections.Counter(c['sys']['timing'] for c in cohort if c['sys']['timing'])))
    print('  diast. :', dict(collections.Counter(c['dia']['timing'] for c in cohort if c['dia']['timing'])))
    print('  sites/p:', dict(collections.Counter(len(c['sites']) for c in cohort)))


if __name__ == '__main__':
    main()
