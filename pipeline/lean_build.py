"""Lean presentation version → lean/ (a separate static site; web/ is untouched).

A short brief for a clinician: only the skills that at least mostly work, a batch of 10 real recordings with each
output shown next to its own validation, and conclusions written from the numbers. Every number is read from an existing
result file — nothing is typed in:
  web/data/cnn_murmur.json, models.json, discover.json (hrv)   validated on held-out children / KAUH adults
  meta/cinc2016_validation.json                                adult heart sounds + simultaneous ECG (PhysioNet/CinC 2016)
  meta/respdbtr_validation.json                                adult COPD patients, GOLD 0–4 (RespiratoryDatabase@TR)

Batch selection (fixed before looking at any output, seed 20261121): within each group one recording is drawn at random;
every output is shown whatever it is.
  3 adult heart  CinC set a normal (with ECG), set a abnormal (valve / murmur, with ECG), set c abnormal (AS or MR, ~75 y)
  2 child heart  CirCor, patients the murmur network never saw: one with a murmur (a murmur valve), one without
  4 adult lung   KAUH diaphragm mode: healthy, asthma, COPD, heart failure
  1 adult lung   RespiratoryDatabase@TR, GOLD 4, one random channel of 12
Audio: base64 MP3 bundles (8 kHz, 32 kbps), like web/audio. Licences: CirCor and CinC ODC-By 1.0; KAUH and TR CC BY 4.0.

  python3 pipeline/lean_build.py
"""
import base64, json, math, shutil, subprocess, tempfile, datetime
from pathlib import Path
import numpy as np, pandas as pd
from paths import ROOT, META, FEATURES, WEB_DATA
from encode_audio import FFMPEG

SEED = 20261121; OUT = ROOT / 'lean'; rng = np.random.default_rng(SEED)
WEB = ROOT / 'web'; CINC = ROOT / 'raw' / 'cinc2016' / 'cinc'; TR = ROOT / 'raw' / 'respdbtr' / 'RespiratoryDatabase@TR'
pct = lambda v: f"{v:.0f} %" if v >= 10 else f"{v:.1f} %"


def ci(a): return f"{a['value']} ({a['lo']}–{a['hi']})"


def auc_words(v):
    return 'excellent' if v >= 0.9 else 'good' if v >= 0.8 else 'moderate' if v >= 0.7 else 'weak' if v >= 0.6 else 'close to chance'


def logreg(m, x):
    mo = m['model']; z = []
    v = [(mo['median'][i] if (x[i] is None or (isinstance(x[i], float) and math.isnan(x[i]))) else x[i]) for i in range(len(x))]
    v = [(v[i] - mo['mean'][i]) / mo['scale'][i] for i in range(len(v))]
    for k in range(len(mo['coef'])): z.append(sum(c * q for c, q in zip(mo['coef'][k], v)) + mo['intercept'][k])
    mx = max(z); e = [math.exp(q - mx) for q in z]; s = sum(e); return [q / s for q in e]


def mp3_from_wav(src):
    with tempfile.TemporaryDirectory() as d:
        o = Path(d) / 'a.mp3'
        subprocess.run([FFMPEG, '-loglevel', 'error', '-y', '-i', str(src), '-ar', '8000', '-ac', '1', '-b:a', '32k', str(o)], check=True)
        return base64.b64encode(o.read_bytes()).decode()


def pick(seq): seq = sorted(seq, key=str); return seq[int(rng.integers(0, len(seq)))]


def main():
    shutil.rmtree(OUT / 'data', ignore_errors=True); shutil.rmtree(OUT / 'audio', ignore_errors=True)
    (OUT / 'data').mkdir(parents=True); (OUT / 'audio').mkdir(parents=True)
    I = json.loads((WEB_DATA / 'index.json').read_text()); P = {p['key']: p for p in I['patients']}
    MJ = json.loads((WEB_DATA / 'models.json').read_text()); HM = next(m for m in MJ['models'] if m['id'] == 'heart_murmur'); LS = next(m for m in MJ['models'] if m['id'] == 'lung_sound')
    CN = json.loads((WEB_DATA / 'cnn_murmur.json').read_text()); thr = CN['threshold']; T = CN['test']
    SEG = json.loads((WEB_DATA / 'discover.json').read_text())['hrv']['segmenter']
    CV = json.loads((META / 'cinc2016_validation.json').read_text()); TV = json.loads((META / 'respdbtr_validation.json').read_text())
    ext = LS['external']; kc = next(c for c in ext['perClass'] if c['class'].startswith('continuous')); kf = next(f for f in ext['flags'] if f['class'].startswith('continuous'))
    cdb = {r['db']: r for r in CV['murmur']['all']}; ecg = CV['ecg']; hrA, hrvA = ecg['heartRate'], ecg['rmssd']
    band = {b['band']: b for b in SEG['byBand']}

    # validation sentences, one per skill, used beside every output of that skill
    V = {
        'murmur_child': f"Tested on {T['patients']} children it never saw ({T['recordings']:,} valve recordings): AUC {ci(T['auc'])}, {auc_words(T['auc']['value'])}. At this flag level it finds {pct(T['sens']['value'])} of murmur valves and leaves {pct(T['spec']['value'])} of normal valves unflagged.",
        'hr_child': f"Agrees with the CirCor experts' hand-marked beats on {SEG['seg']['hr']['n']:,} held-out recordings: ICC {SEG['seg']['hr']['icc']}, within 5 bpm in {pct(SEG['seg']['hr']['within5bpm'])}.",
        'hrv_child': f"RMSSD agrees with the experts' beats in children: ICC {band['Child']['icc']} (n {band['Child']['n']}), bias {band['Child']['bias']} ms. Not valid in infants (ICC {band['Infant']['icc']}).",
        'hr_adult': f"Against the simultaneous ECG in {hrA['n']} adult recordings: ICC {hrA['icc']} ({hrA['iccCI'][0]}–{hrA['iccCI'][1]}), bias {hrA['bias']} bpm, 95 % limits {hrA['loa'][0]} to {hrA['loa'][1]} bpm — {'passes' if hrA['pass'] else 'fails'} the rule set beforehand (ICC ≥ 0.90, |bias| ≤ 3 bpm).",
        'hrv_adult': (f"RMSSD against the ECG in {hrvA['n']} adult recordings: ICC {hrvA['icc']}, bias {hrvA['bias']} ms — {'passes' if hrvA['pass'] else 'fails'} the rule (ICC ≥ 0.75, |bias| ≤ 10 ms)." if hrvA else 'RMSSD could not be measured on enough adult recordings.'),
        'wheeze_adult': f"On {ext['n']} KAUH adults (model frozen before it saw them): AUC {ci(kc['auc'])}, {auc_words(kc['auc']['value'])}. At this flag level it finds {pct(kf['sens']['value'])} of recordings with wheeze or rhonchi and leaves {pct(kf['spec']['value'])} of the others unflagged.",
    }
    for db in 'ac':
        r = cdb.get(db)
        if r: V[f'murmur_adult_{db}'] = f"On {r['recordings']} adult recordings from this database (external, never trained on): AUC {ci(r['auc'])}, {auc_words(r['auc']['value'])}, for any confirmed heart diagnosis against normal. At this flag level it finds {pct(r['sens']['value'])} of abnormal recordings and leaves {pct(r['spec']['value'])} of normal ones unflagged."

    cases = []
    # ---- adult heart, CinC
    R = [json.loads(l) for l in open(FEATURES / 'cinc2016.jsonl')]; pairs = {p['name']: p for p in ecg['pairs']}
    for db, y, title in (('a', -1, 'Adult heart — normal'), ('a', 1, 'Adult heart — valve disease or murmur'), ('c', 1, 'Adult heart — aortic stenosis or mitral regurgitation')):
        pool = [r for r in R if r['db'] == db and r['y'] == y and r['cnnP'] is not None and (db != 'a' or r['name'] in pairs and pairs[r['name']]['hrPcg'] and pairs[r['name']]['hrEcg'])]
        r = pick([q['name'] for q in pool]); r = next(q for q in pool if q['name'] == r)
        cid = f"cinc-{r['name']}"; (OUT / 'audio' / f'{cid}.json').write_text(json.dumps({f"{r['name']}.mp3": mp3_from_wav(CINC / f'training-{db}' / f"{r['name']}.wav")}))
        outs = [{'skill': 'Murmur network', 'value': f"{round(100 * r['cnnP'])} %", 'flag': r['cnnP'] >= thr, 'flagText': 'flagged' if r['cnnP'] >= thr else 'not flagged', 'validation': V[f'murmur_adult_{db}']}]
        if db == 'a':
            pp = pairs[r['name']]; outs.append({'skill': 'Heart rate', 'value': f"{round(pp['hrPcg'])} bpm", 'ref': f"ECG {round(pp['hrEcg'])} bpm", 'validation': V['hr_adult']})
            if hrvA and hrvA['pass'] and pp['rmssdPcg'] is not None: outs.append({'skill': 'HRV (RMSSD)', 'value': f"{pp['rmssdPcg']} ms", 'ref': f"ECG {pp['rmssdEcg']} ms" if pp['rmssdEcg'] is not None else None, 'validation': V['hrv_adult']})
        cases.append({'id': cid, 'group': 'heart', 'adult': True, 'title': title, 'who': CV_DB_WHO[db], 'reference': 'Confirmed cardiac diagnosis (Challenge label: abnormal)' if y == 1 else 'Normal (Challenge label)',
                      'source': 'PhysioNet/CinC 2016, set ' + db, 'licence': 'ODC-By 1.0', 'file': f"{r['name']}.mp3", 'dur': r['dur'], 'outputs': outs})
    # ---- child heart, CirCor (held-out patients only)
    test = set(HM['testPids']); per = SEG['perRecording']
    import train_cnn as TC, cnn_eval as CE
    net = CE.load_net(CN); idx = {json.loads(l)['name']: json.loads(l) for l in open(TC.CNN / 'index.jsonl')}; F = np.fromfile(TC.CNN / 'spec.f32', dtype='<f4').reshape(-1, TC.BINS)
    for mur, title in (('Present', 'Child heart — murmur'), ('Absent', 'Child heart — no murmur')):
        pool = [p for p in I['patients'] if p['src'] == 'circor' and p['pid'] in test and p.get('ageBand') == 'Child' and p['murmur'] == mur]
        p = P[pick([q['key'] for q in pool])]
        sites = [s for s in p['sites'] if (s['record'] == 'Murmur') == (mur == 'Present') and s['name'] in idx and idx[s['name']]['nF'] >= TC.WIN and s['name'] in per]
        s = sorted(sites, key=lambda q: 'AV PV TV MV'.split().index(q['loc']))[0]
        rr = idx[s['name']]; prob = float(TC.score(net, [{**rr, 'Z': TC.norm(F[rr['offset']:rr['offset'] + rr['nF']].astype(np.float64))}])[0])
        h = per[s['name']]; cid = f"circor-{s['name']}"
        b = json.loads((WEB / 'audio' / f"{p['key']}.json").read_text()); fn = f"{s['name']}.mp3"; (OUT / 'audio' / f'{cid}.json').write_text(json.dumps({fn: b[fn]}))
        outs = [{'skill': 'Murmur network', 'value': f"{round(100 * prob)} %", 'flag': prob >= thr, 'flagText': 'flagged' if prob >= thr else 'not flagged', 'validation': V['murmur_child']},
                {'skill': 'Heart rate', 'value': f"{round(h['hr'])} bpm", 'validation': V['hr_child']}]
        if h.get('rmssd') is not None: outs.append({'skill': 'HRV (RMSSD)', 'value': f"{h['rmssd']} ms", 'validation': V['hrv_child']})
        sys_ = p.get('sys') or {}
        ref = f"Murmur at this valve (expert): {sys_.get('grading') or ''} {sys_.get('timing') or ''}".strip() if mur == 'Present' else 'No murmur (expert)'
        cases.append({'id': cid, 'group': 'heart', 'adult': False, 'title': title, 'who': f"Child, {p.get('age')} y band, {'male' if p.get('sex') == 'M' else 'female'} · {s['site']}",
                      'reference': ref, 'source': 'CirCor DigiScope (PhysioNet)', 'licence': 'ODC-By 1.0', 'file': fn, 'dur': None, 'outputs': outs})
    # ---- adult lung, KAUH (diaphragm mode; the breath-sound model was frozen before seeing KAUH)
    K = [json.loads(l) for l in open(FEATURES / 'kauh.jsonl')]; K = [k for k in K if k['mode'] == 'D']
    for dx, title in (('Healthy adult (control)', 'Adult lungs — healthy'), ('Asthma', 'Adult lungs — asthma'), ('COPD', 'Adult lungs — COPD'), ('Heart failure', 'Adult lungs — heart failure')):
        k = pick([q['name'] for q in K if q['disease'] == dx]); k = next(q for q in K if q['name'] == k)
        pr = logreg(LS, [k.get(f) for f in LS['features']]); pc = pr[1] + pr[3]   # continuous sound: "continuous" or "both"
        cid = f"kauh-{k['name']}"; b = json.loads((WEB / 'audio' / f"{k['key']}.json").read_text()); fn = f"{k['name']}.mp3"; (OUT / 'audio' / f'{cid}.json').write_text(json.dumps({fn: b[fn]}))
        lab = {'CAS': 'continuous sound (wheeze / rhonchi)', 'DAS': 'crackles', 'Normal': 'normal', 'CAS & DAS': 'wheeze and crackles'}.get(k['label'], k['label'])
        cases.append({'id': cid, 'group': 'lung', 'adult': True, 'title': title, 'who': f"Adult, {k['age']} y, {'male' if P[k['key']].get('sex') == 'M' else 'female'} · {P[k['key']]['sites'][0]['site'].split(' · ')[0]}",
                      'reference': f"Expert label: {lab} · diagnosis {dx.lower()}", 'source': 'KAUH (Fraiwan et al., Mendeley)', 'licence': 'CC BY 4.0', 'file': fn, 'dur': k.get('dur'),
                      'outputs': [{'skill': 'Wheeze / rhonchi', 'value': f"{round(100 * pc)} %", 'flag': pr[1] >= kf['threshold'], 'flagText': 'flagged' if pr[1] >= kf['threshold'] else 'not flagged', 'validation': V['wheeze_adult']}]})
    # ---- adult lung, COPD GOLD 4 (RespiratoryDatabase@TR)
    TRf = [json.loads(l) for l in open(FEATURES / 'respdbtr.jsonl')]; g4 = [p['pid'] for p in TV['perPatient'] if p['gold'] == 4]
    pid = pick(g4)
    rows = sorted([r for r in TRf if r['pid'] == pid], key=lambda r: r['name']); r = rows[int(rng.integers(0, len(rows)))]
    cid = f"tr-{r['name']}"; (OUT / 'audio' / f'{cid}.json').write_text(json.dumps({f"{r['name']}.mp3": mp3_from_wav(TR / f"{r['name']}.wav")}))
    pa = TV['pAbn']
    cases.append({'id': cid, 'group': 'lung', 'adult': True, 'title': 'Adult lungs — COPD, GOLD 4', 'who': f"Adult with COPD · channel {r['ch']} of 12", 'reference': 'GOLD stage 4 (pulmonologists: spirometry, chest X-ray, auscultation)',
                  'source': 'RespiratoryDatabase@TR (Altan et al., Mendeley)', 'licence': 'CC BY 4.0', 'file': f"{r['name']}.mp3", 'dur': r['dur'],
                  'outputs': [{'skill': 'Wheeze / rhonchi', 'value': f"{round(100 * (r['p'][1] + r['p'][3]))} %", 'flag': bool(r['flags'][1]), 'flagText': 'flagged' if r['flags'][1] else 'not flagged', 'validation': V['wheeze_adult']},
                              {'skill': 'COPD severity', 'value': 'not measured', 'validation': f"Tested on {TV['patients']} COPD patients (GOLD 0–4): the abnormal-sound score does not rise with stage (ρ {pa['rho']}, 95 % CI {pa['ci'][0]} to {pa['ci'][1]}, p {pa['p']:.2f}). Severity is therefore not reported."}]})

    # ---- skills table (only what at least mostly works) and what is not ready
    skills = [
        {'skill': 'Murmur at a heart valve (children)', 'evidence': f"{T['patients']} held-out children", 'result': f"AUC {T['auc']['value']}", 'status': 'works'},
        {'skill': 'Murmur in adults with valve disease', 'evidence': f"{cdb['a']['recordings']} + {cdb['c']['recordings']} adult recordings (external)" if 'a' in cdb and 'c' in cdb else 'adult recordings',
         'result': ' · '.join(f"set {d} AUC {cdb[d]['auc']['value']}" for d in 'ac' if d in cdb), 'status': 'works' if CV['murmur']['transfers'] else 'partly'},
        {'skill': 'Heart rate from heart sounds (children)', 'evidence': f"{SEG['seg']['hr']['n']:,} held-out recordings vs experts", 'result': f"ICC {SEG['seg']['hr']['icc']}", 'status': 'works'},
        {'skill': 'Heart rate from heart sounds (adults)', 'evidence': f"{hrA['n']} recordings vs simultaneous ECG", 'result': f"ICC {hrA['icc']}, bias {hrA['bias']} bpm", 'status': 'works' if hrA['pass'] else 'partly'},
        {'skill': 'Heart-rate variability, RMSSD (children, adolescents)', 'evidence': f"{band['Child']['n'] + band['Adolescent']['n']} held-out recordings vs experts", 'result': f"ICC {band['Child']['icc']} / {band['Adolescent']['icc']}", 'status': 'works'},
        {'skill': 'Heart-rate variability, RMSSD (adults)', 'evidence': f"{hrvA['n']} recordings vs ECG" if hrvA else '—', 'result': f"ICC {hrvA['icc']}, bias {hrvA['bias']} ms" if hrvA else '—', 'status': 'works' if hrvA and hrvA['pass'] else 'fails'},
        {'skill': 'Wheeze / rhonchi (adults)', 'evidence': f"{ext['n']} KAUH adults (external)", 'result': f"AUC {kc['auc']['value']}", 'status': 'partly'},
    ]
    skills = [s for s in skills if s['status'] != 'fails']
    lm = next(m for m in MJ['models'] if m['id'] == 'lung_pattern_adult')
    notReady = [f"Crackles: the rule finds few crackles in adults, and the breath-sound model's crackle AUC on KAUH adults is {next(c for c in ext['perClass'] if c['class'] == 'crackles')['auc']['value']}.",
                f"COPD severity from breath sounds: no relation to GOLD stage in {TV['patients']} patients (ρ {pa['rho']}, p {pa['p']:.2f}).",
                f"Naming a disease from breath sounds: the adults' model reached AUC {lm['test']['macroAuc']} on held-out patients and failed its gate.",
                "Sleep apnea severity (AHI) from breath sounds: in a 10-patient pilot it found under half of the scored events."]
    if not (hrvA and hrvA['pass']): notReady.insert(0, V['hrv_adult'])
    out = {'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'seed': SEED, 'threshold': thr, 'skills': skills, 'notReady': notReady, 'cases': cases,
           'ecgPoints': [[round(p['hrEcg'], 1), round(p['hrPcg'], 1)] for p in ecg['pairs'] if p['hrPcg'] and p['hrEcg']],
           'adultMurmur': [{'db': r['db'], 'what': r['what'], 'n': r['recordings'], 'abnormal': r['abnormal'], 'auc': r['auc']} for r in CV['murmur']['all']],
           'copd': {'byGold': TV['pAbn']['medianByGold'], 'rho': pa['rho'], 'p': pa['p'], 'n': TV['patients']},
           'sources': [
               'CirCor DigiScope 1.0.3 — Oliveira et al., IEEE JBHI 2022; PhysioNet; ODC-By 1.0',
               'PhysioNet/CinC Challenge 2016 — Liu et al., Physiol Meas 2016;37:2181, doi 10.1088/0967-3334/37/12/2181; ODC-By 1.0',
               'KAUH lung sounds — Fraiwan et al., Mendeley Data jwyy9np4gv v3; CC BY 4.0',
               'RespiratoryDatabase@TR — Altan, Kutlu et al., Mendeley Data 10.17632/p9z4h98s6j.2; CC BY 4.0']}
    (OUT / 'data' / 'brief.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(len(cases), 'cases ->', OUT); [print(' ', c['id'], c['title'], [(o['skill'], o['value'], o.get('flagText')) for o in c['outputs']]) for c in cases]


CV_DB_WHO = {'a': 'Adult (MIT heart sounds database) · recorded with simultaneous ECG', 'c': 'Older adult, ~75 y (Thessaloniki; diagnosis by echocardiography)'}

if __name__ == '__main__':
    main()
