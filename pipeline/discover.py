"""Discoveries: exploratory statistics over the full SPRSound database (all releases).
Input : meta/features/spr.*.jsonl  (per-recording features from pipeline/precompute.js features)
Output: data/discover.json  (consumed by the Discoveries page)"""
import json, glob, collections, math
from paths import SPR_FEATURES, WEB_DATA, META
import numpy as np
from scipy import stats

rows = []
for f in SPR_FEATURES:
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('err') or r.get('f50') is None:
            continue
        rows.append(r)
print(len(rows), 'recordings with features')

AGE_BINS = [(0, 1, '<1 y'), (1, 3, '1–3 y'), (3, 6, '3–6 y'), (6, 9, '6–9 y'), (9, 12, '9–12 y'), (12, 18, '12–18 y')]
def abin(a):
    for lo, hi, lab in AGE_BINS:
        if lo <= a < hi: return lab
    return '12–18 y'

def mwu(a, b):
    a, b = [x for x in a if x is not None and not math.isnan(x)], [x for x in b if x is not None and not math.isnan(x)]
    if len(a) < 5 or len(b) < 5: return None
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    rbc = 1 - 2 * u / (len(a) * len(b))          # rank-biserial correlation: positive when B tends to be larger than A
    return {'nA': len(a), 'nB': len(b), 'medA': float(np.median(a)), 'medB': float(np.median(b)), 'meanA': float(np.mean(a)), 'meanB': float(np.mean(b)), 'U': float(u), 'p': float(p), 'r': float(rbc)}

def pct_curve(xs, ys, bins=AGE_BINS):
    out = []
    for lo, hi, lab in bins:
        v = [y for x, y in zip(xs, ys) if lo <= x < hi and y is not None]
        if len(v) >= 8:
            q = np.percentile(v, [10, 25, 50, 75, 90])
            out.append({'bin': lab, 'n': len(v), 'p10': float(q[0]), 'p25': float(q[1]), 'p50': float(q[2]), 'p75': float(q[3]), 'p90': float(q[4])})
    return out

core = [r for r in rows if r['loc'] in ('p1', 'p2', 'p3', 'p4')]
ctrl = [r for r in core if r['disease'] == 'Control Group' and r['label'] == 'Normal']
sick = [r for r in core if r['disease'] not in (None, 'Control Group')]
normal_any = [r for r in core if r['label'] == 'Normal']
D = {'n': {'recordings': len(rows), 'children': len({r['key'] for r in rows}), 'controls': len({r['key'] for r in ctrl}), 'hours': round(sum(r['dur'] for r in rows) / 3600, 1)}}

# ---------- D1 acoustic growth chart: median frequency of normal breath sounds vs age ----------
xs = [r['age'] for r in normal_any]; ys = [r['cen'] for r in normal_any]
rho, p = stats.spearmanr(xs, ys)
rhoC, pC = stats.spearmanr([r['age'] for r in ctrl], [r['cen'] for r in ctrl]) if len(ctrl) > 10 else (None, None)
D['growth'] = {'curve': pct_curve(xs, ys), 'curveCen': pct_curve(xs, [r['cen'] for r in normal_any]), 'rho': float(rho), 'p': float(p), 'n': len(xs),
               'rhoControls': None if rhoC is None else float(rhoC), 'pControls': None if pC is None else float(pC), 'nControls': len(ctrl),
               'curveF50': pct_curve(xs, [r['f50'] for r in normal_any]), 'scatter': [[round(r['age'], 2), r['cen']] for r in normal_any[::3]], 'ctrlCurve': pct_curve([r['age'] for r in ctrl], [r['cen'] for r in ctrl])}

# ---------- D2 hidden abnormality: expert-Normal sites in sick children vs healthy controls, age-stratified ----------
def strat_compare(feat):
    res = []
    for lo, hi, lab in AGE_BINS:
        a = [r[feat] for r in ctrl if lo <= r['age'] < hi]
        b = [r[feat] for r in sick if r['label'] == 'Normal' and lo <= r['age'] < hi]
        m = mwu(a, b)
        if m: res.append({'bin': lab, **m})
    allm = mwu([r[feat] for r in ctrl], [r[feat] for r in sick if r['label'] == 'Normal'])
    return {'bins': res, 'all': allm}
D['hidden'] = {f: strat_compare(f) for f in ('cen', 'crkRate', 'fineRate', 'tonFrac')}
# by diagnosis (expert-Normal sites only)
hd = []
for dx in sorted({r['disease'] for r in sick}):
    b = [r for r in sick if r['disease'] == dx and r['label'] == 'Normal']
    if len({r['key'] for r in b}) < 5: continue
    hd.append({'dx': dx, 'children': len({r['key'] for r in b}), 'cen': mwu([r['cen'] for r in ctrl], [r['cen'] for r in b]), 'crk': mwu([r['crkRate'] for r in ctrl], [r['crkRate'] for r in b])})
D['hiddenByDx'] = hd
# confounder check: controls vs sick children's Normal sites WITHIN the same release (recording batch)
rel = []
for R in sorted({r['rel'] for r in core}):
    a = [r['cen'] for r in ctrl if r['rel'] == R]; b = [r['cen'] for r in sick if r['label'] == 'Normal' and r['rel'] == R]
    m = mwu(a, b)
    rel.append({'release': R, 'ctrlRec': len(a), 'sickNormalRec': len(b), 'cen': m, 'crk': mwu([r['crkRate'] for r in ctrl if r['rel'] == R], [r['crkRate'] for r in sick if r['label'] == 'Normal' and r['rel'] == R])})
D['hiddenByRelease'] = rel

# ---------- D3 spatial pattern of expert findings by diagnosis ----------
byChild = collections.defaultdict(dict)
for r in core:
    k = r['key']; byChild[k]['disease'] = r['disease']; byChild[k]['age'] = r['age']
    s = byChild[k].setdefault('sites', {})
    lab = r['label']; prev = s.get(r['loc'])
    rank = {'Normal': 0, 'CAS': 1, 'DAS': 1, 'CAS & DAS': 2}
    if prev is None or rank.get(lab, 0) > rank.get(prev, 0): s[r['loc']] = lab
spatial = []
for dx in sorted({c['disease'] for c in byChild.values() if c.get('disease')}):
    kids = [c for c in byChild.values() if c.get('disease') == dx and len(c['sites']) >= 3]
    if len(kids) < 4: continue
    nab = [sum(1 for v in c['sites'].values() if v != 'Normal') / len(c['sites']) for c in kids]
    def side(c):
        L = any(c['sites'].get(l, 'Normal') != 'Normal' for l in ('p1', 'p2')); R = any(c['sites'].get(l, 'Normal') != 'Normal' for l in ('p3', 'p4'))
        return 'bilateral' if L and R else 'left only' if L else 'right only' if R else 'none'
    sides = collections.Counter(side(c) for c in kids)
    cas = sum(1 for c in kids if any('CAS' in v for v in c['sites'].values())) / len(kids)
    das = sum(1 for c in kids if any('DAS' in v for v in c['sites'].values())) / len(kids)
    spatial.append({'dx': dx, 'children': len(kids), 'fracAbnSites': float(np.mean(nab)), 'anyAbn': sum(1 for x in nab if x > 0) / len(kids),
                    'sides': dict(sides), 'cas': cas, 'das': das})
D['spatial'] = spatial
# focal vs diffuse: among children with ≥1 abnormal site, share of abnormal sites (pneumonia vs asthma)
def share(dx):
    return [sum(1 for v in c['sites'].values() if v != 'Normal') / len(c['sites']) for c in byChild.values() if c.get('disease') == dx and len(c['sites']) >= 3 and any(v != 'Normal' for v in c['sites'].values())]
D['focal'] = {'asthma_vs_pneumonia': mwu(share('Asthma'), share('Pneumonia (non-severe)')), 'severe_vs_nonsevere': mwu(share('Pneumonia (non-severe)'), share('Pneumonia (severe)'))}

# ---------- D4 severity: severe vs non-severe pneumonia ----------
sev = {}
for name, f in [('fracAbnSites', None), ('crkRate', 'crkRate'), ('fineRate', 'fineRate'), ('tonFrac', 'tonFrac'), ('cen', 'cen'), ('rr', 'rr'), ('hr', 'hr')]:
    if f is None:
        a = [x for x in share('Pneumonia (non-severe)')] + [0.0] * sum(1 for c in byChild.values() if c.get('disease') == 'Pneumonia (non-severe)' and len(c['sites']) >= 3 and all(v == 'Normal' for v in c['sites'].values()))
        b = [x for x in share('Pneumonia (severe)')] + [0.0] * sum(1 for c in byChild.values() if c.get('disease') == 'Pneumonia (severe)' and len(c['sites']) >= 3 and all(v == 'Normal' for v in c['sites'].values()))
    else:
        # child-level median to avoid pseudo-replication
        def child_med(dx):
            g = collections.defaultdict(list)
            for r in core:
                if r['disease'] == dx and r.get(f) is not None and (f != 'rr' or not r['est']): g[r['key']].append(r[f])
            return [float(np.median(v)) for v in g.values()]
        a, b = child_med('Pneumonia (non-severe)'), child_med('Pneumonia (severe)')
    sev[name] = mwu(a, b)
D['severity'] = sev
# age of severe vs non-severe (confounder check)
ages = lambda dx: [c['age'] for c in byChild.values() if c.get('disease') == dx]
D['severityAge'] = mwu(ages('Pneumonia (non-severe)'), ages('Pneumonia (severe)'))

# ---------- D5 wheeze pitch vs age and diagnosis ----------
pitch = []
for r in core:
    for pz in r.get('pitches', []):
        if pz['type'] == 'Wheeze': pitch.append((r['age'], pz['hz'], r['disease'], r['key']))
# one value per child (median) to avoid pseudo-replication
pc = collections.defaultdict(list)
for a, hz, dx, k in pitch: pc[(k, dx, a)].append(hz)
pchild = [(a, float(np.median(v)), dx) for (k, dx, a), v in pc.items()]
if len(pchild) >= 10:
    rho, p = stats.spearmanr([x[0] for x in pchild], [x[1] for x in pchild])
else:
    rho, p = float('nan'), float('nan')
byDx = []
for dx in sorted({x[2] for x in pchild if x[2]}):
    v = [x[1] for x in pchild if x[2] == dx]
    if len(v) >= 5: byDx.append({'dx': dx, 'n': len(v), 'median': float(np.median(v)), 'q1': float(np.percentile(v, 25)), 'q3': float(np.percentile(v, 75))})
D['wheezePitch'] = {'nEvents': len(pitch), 'nChildren': len(pchild), 'rho': float(rho), 'p': float(p), 'byDx': byDx, 'curve': pct_curve([x[0] for x in pchild], [x[1] for x in pchild]), 'scatter': [[round(a, 2), hz] for a, hz, dx in pchild]}

# ---------- D6 breathing / heart rate from auscultation + test–retest reliability ----------
rrRows = [r for r in core if r.get('rr') and not r['est']]
def icc1(groups):
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 10: return None
    k = np.mean([len(g) for g in groups]); allv = np.concatenate(groups); gm = allv.mean()
    msb = sum(len(g) * (np.mean(g) - gm) ** 2 for g in groups) / (len(groups) - 1)
    msw = sum(((np.array(g) - np.mean(g)) ** 2).sum() for g in groups) / (len(allv) - len(groups))
    return float((msb - msw) / (msb + (k - 1) * msw))
gRR = collections.defaultdict(list); gHR = collections.defaultdict(list)
for r in rrRows: gRR[r['key']].append(r['rr'])
for r in core:
    if r.get('hr'): gHR[r['key']].append(r['hr'])
WHO = lambda a: 60 if a < 2 / 12 else 50 if a < 1 else 40 if a < 5 else None
fast = collections.defaultdict(lambda: [0, 0])
for k, v in gRR.items():
    c = byChild.get(k, {}); a = c.get('age')
    if a is None or WHO(a) is None: continue
    grp = 'Pneumonia (severe)' if c.get('disease') == 'Pneumonia (severe)' else 'Pneumonia (non-severe)' if c.get('disease') == 'Pneumonia (non-severe)' else 'Control Group' if c.get('disease') == 'Control Group' else None
    if not grp: continue
    fast[grp][1] += 1; fast[grp][0] += 1 if np.median(v) >= WHO(a) else 0
D['rates'] = {'rrCurve': pct_curve([r['age'] for r in rrRows], [r['rr'] for r in rrRows]), 'hrCurve': pct_curve([r['age'] for r in core if r.get('hr')], [r['hr'] for r in core if r.get('hr')]),
              'iccRR': icc1(list(gRR.values())), 'iccHR': icc1(list(gHR.values())), 'nRR': len(rrRows), 'fracRRreliable': len(rrRows) / max(1, len(core)),
              'whoFast': {k: {'fast': v[0], 'n': v[1]} for k, v in fast.items()}}

# ---------- D7 label audit: expert-Normal recordings with strong adventitious-like evidence ----------
cands = []
for r in core:
    if r['label'] != 'Normal': continue
    score = 0; why = []
    if r['tonFrac'] >= 0.10: score += r['tonFrac'] * 10; why.append(f"tonal {r['tonFrac']*100:.0f}% of time at {r['tonHz']} Hz")
    if r['fineRate'] >= 3: score += r['fineRate'] / 2; why.append(f"{r['fineRate']:.1f} fine-crackle-like transients/s")
    if score: cands.append({'key': r['key'], 'name': r['name'], 'loc': r['loc'], 'age': r['age'], 'disease': r['disease'], 'score': round(score, 2), 'why': '; '.join(why), 'labelSrc': r['labelSrc']})
cands.sort(key=lambda c: -c['score'])
D['audit'] = {'nNormal': sum(1 for r in core if r['label'] == 'Normal'), 'nFlagged': len(cands), 'top': cands[:40]}

json.dump(D, open(WEB_DATA / 'discover.json', 'w'), default=lambda o: None)
print(json.dumps({k: (v if k in ('n',) else '…') for k, v in D.items()}))
print('growth rho', D['growth']['rho'], 'p', D['growth']['p'])
print('hidden cen all', D['hidden']['cen']['all']); print('hidden crk all', D['hidden']['crkRate']['all'])
print('severity', {k: (v and (round(v['medA'], 3), round(v['medB'], 3), '%.3g' % v['p'])) for k, v in D['severity'].items()})
print('wheeze pitch', D['wheezePitch']['nChildren'], D['wheezePitch']['rho'], D['wheezePitch']['p'], D['wheezePitch']['byDx'])
print('rates icc', D['rates']['iccRR'], D['rates']['iccHR'], D['rates']['fracRRreliable'], D['rates']['whoFast'])
print('audit', D['audit']['nFlagged'], '/', D['audit']['nNormal'])
print('by release', [(x['release'], x['ctrlRec'], x['sickNormalRec'], x['cen'] and (x['cen']['medA'], x['cen']['medB'], '%.2g' % x['cen']['p'])) for x in D['hiddenByRelease']])
print('spatial', [(s['dx'], s['children'], round(s['fracAbnSites'], 2), s['sides']) for s in D['spatial']])
print('focal', D['focal'])
