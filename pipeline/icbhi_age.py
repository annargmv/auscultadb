"""Does adult breath sound carry age? ICBHI 2017 → discover.json['age']['icbhi'].

The same age clock as pipeline/age_discover.py (ridge on the page's recordingFeatures, fit on 60 % of patients, scored
on the held-out 40 %, null = training-mean age, bootstrap CI of the MAE gain, permutation p), run on ICBHI adults:
  1. all adults (every diagnosis and device) — confounded: diagnosis and stethoscope both track age;
  2. COPD patients on one microphone (AKG C417L) — diagnosis and device fixed, so only age varies.
The KAUH adult clock found no age signal (MAE 17.4 vs null 17.5 y). ICBHI is a second, independent adult test.
Research licence: only these summary numbers reach web/. No longevity claim: ICBHI has no follow-up.

  python3 pipeline/icbhi_age.py      (then python3 pipeline/merge.py --bump)
"""
import json, datetime
import numpy as np
from paths import META, FEATURES, WEB_DATA
import age_discover as AD

SEED = 20261117


def main():
    I = json.load(open(META / 'icbhi_index.json')); P = {p['pid']: p for p in I['patients']}
    rows = []
    for l in open(FEATURES / 'icbhi.jsonl'):
        r = json.loads(l)
        if 'err' in r: continue
        p = P[r['pid']]
        if p['age'] is None or p['age'] < 18: continue
        rows.append({'pid': r['pid'], 'age': p['age'], 'dx': p['dx'], 'dev': r['dev'], **{k: r['rf'].get(k) for k in AD.FEATS}})
    rng = np.random.default_rng(SEED); out = {'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'seed': SEED,
                                              'source': 'ICBHI 2017 (Rocha et al., Physiol Meas 2019, PMID 30708353)', 'clocks': []}
    # PubMed-checked evidence for two more sound-related markers, shown on the Longevity page (text quoted from the abstracts)
    out['refs'] = {
        'rr': {'pmid': '31151958', 'doi': '10.1183/13993003.02175-2018', 'cite': 'Baumert M et al. Mean nocturnal respiratory rate predicts cardiovascular and all-cause mortality in community-dwelling older men and women. Eur Respir J 2019;54:1802175',
               'finding': 'Mean nocturnal breathing rate ≥ 16/min (from overnight polysomnography) independently predicted all-cause mortality in community-dwelling older adults: HR 1.18 (95 % CI 1.04–1.32) in 2,686 men, 1.50 (1.02–2.20) in 406 women.'},
        'wheeze': {'pmid': '2506967', 'doi': '10.1136/bmj.299.6695.357', 'cite': 'Carpenter L, Beral V, Strachan D et al. Respiratory symptoms as predictors of 27 year mortality in a representative sample of British adults. BMJ 1989;299:357–61',
                   'finding': 'Wheeze most days or nights (questionnaire, not auscultation) predicted death from chronic bronchitis in men: mortality ratio 10.5 (95 % CI 4.4–24.6); most respiratory symptoms carried 20–50 % higher all-cause death rates, adjusted for age, sex and smoking.'}}
    for label, sel in (('ICBHI adults — all diagnoses and devices (age, diagnosis and device entangled)', lambda r: True),
                       ('ICBHI COPD patients, one microphone (AKG C417L) — diagnosis and device fixed', lambda r: r['dx'] == 'COPD' and r['dev'] == 'AKGC417L')):
        R = [r for r in rows if sel(r)]
        if len({r['pid'] for r in R}) < 15: continue
        c = AD.clock(R, 'pid', rng, label); c['patients'] = len({r['pid'] for r in R}); c.pop('test', None)
        out['clocks'].append(c)
        print(f"{label}: {c['patients']} patients, MAE {c['mae']} vs null {c['maeNull']} y, gain CI {c['gainCI']}, perm p {c['pPerm']}, rho {c['rho']} -> {'SIGNAL' if c['significant'] else 'no signal'}")
    DJ = WEB_DATA / 'discover.json'; D = json.load(open(DJ)); D.setdefault('age', {})['icbhi'] = out
    DJ.write_text(json.dumps(D, separators=(',', ':')))
    print('-> discover.json[age][icbhi]   (then merge.py --bump)')


if __name__ == '__main__':
    main()
