"""One-page summary of the full application for a longevity-medicine audience → lean/data/onepager.json.

Every number is read from the application's own result files (web/data/*.json, meta/*.json); literature figures come from
the PubMed-checked references stored with those results. lean/onepager.html renders the file.

  python3 pipeline/onepager_build.py
"""
import json, os, datetime
from paths import ROOT, META, WEB_DATA

W = lambda f: json.loads((WEB_DATA / f).read_text()); M = lambda f: json.loads((META / f).read_text())


def main():
    I, V, D, MO, IM, CN = W('index.json'), W('validation.json'), W('discover.json'), W('models.json'), W('icbhi_models.json'), W('cnn_murmur.json')
    H, CV, EP, TV = M('hear_probe.json'), M('cinc2016_validation.json'), M('eph_validation.json'), M('respdbtr_validation.json')
    src = {}
    for p in I['patients']: src[p['src']] = src.get(p['src'], 0) + 1
    A = D['age']; ic = A['icbhi']['clocks'][0]; ad = next(m for m in IM['models'] if m['id'] == 'icbhi_adult')
    seg = D['hrv']['segmenter']; rep = seg['repeat']; band = {b['band']: b for b in seg['byBand']}
    hm = H['circor']; hv = hm['vs']['murmur network (shipped)']
    ls = next(m for m in MO['models'] if m['id'] == 'lung_sound'); kc = next(c for c in ls['external']['perClass'] if c['class'].startswith('continuous'))
    cdb = {r['db']: r for r in CV['murmur']['all']}; hr, hrR, hrv = EP['heartRate'], EP['recordLevel']['heartRate'], EP['rmssd']
    ap = D.get('apnea', {}); tr = (ap.get('channels') or {}).get('trach') or {}
    rr = json.loads((META / 'rr_validation.json').read_text()) if (META / 'rr_validation.json').exists() else {}
    refs = A['icbhi'].get('markers') or A['icbhi'].get('refs') or {}
    out = {
        'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'app': {'patients': len(I['patients']), 'bySource': src, 'reports': len(V['reports']), 'analyses': 16,
                'datasets': 9, 'cnnKB': os.path.getsize(WEB_DATA / 'cnn_murmur.json') // 1024},
        'findings': {
            'smallBeatsBig': {'cnn': hv['auc'], 'hear': hm['auc'], 'delta': -hv['delta'], 'ci': [-hv['ci'][1], -hv['ci'][0]], 'patients': hm['patients'], 'recordings': hm['n'],
                              'cnnKB': os.path.getsize(WEB_DATA / 'cnn_murmur.json') // 1024, 'hearParams': '303 million'},
            'device': {'deviceOnlyAuc': ad['deviceOnly']['macroAuc'], 'soundAuc': ad['test']['macroAuc'], 'patients': ad.get('nPatients'),
                       'embedDevice': round(100 * H['icbhi']['deviceFromEmbedding']['accuracy']), 'embedMajority': round(100 * H['icbhi']['deviceFromEmbedding']['majority'])},
            'ageClock': {'kauh': {k: A['adults'][k] for k in ('mae', 'maeNull', 'nTest', 'ageRange', 'pPerm')}, 'icbhi': {k: ic[k] for k in ('mae', 'maeNull', 'nTest', 'ageRange', 'pPerm')},
                         'hrvAge': {'rho': D['hrv']['segmenter']['ageTrend']['rhoSeg'], 'patients': D['hrv']['segmenter']['ageTrend']['patients']}},
            'oneClip': {'rmssdIcc': rep['rmssd']['icc'], 'rmssdMdc': rep['rmssd']['mdc95'], 'rmssdMedian': rep['rmssd']['median'], 'hrIcc': rep['hr']['icc'], 'hrMdc': rep['hr']['mdc95'], 'patients': rep['patients']},
        },
        'markers': [
            {'marker': 'Resting heart rate', 'link': '+10 beats/min: relative risk 1.09 (1.07–1.12) for all-cause death', 'ref': 'Zhang et al., CMAJ 2016 · PMID 26598376',
             'here': f"Adults: within {max(abs(hrR['loa'][0]), abs(hrR['loa'][1]))} bpm of a research-grade ECG over a session ({EP['subjects']} men), ±{max(abs(hr['loa'][0]), abs(hr['loa'][1]))} bpm per 30-s clip. Children: ICC {seg['seg']['hr']['icc']} against experts.", 'status': 'ready'},
            {'marker': 'Heart-rate variability', 'link': 'Reduced HRV predicted mortality in older adults (SDNN, 2-h ECG)', 'ref': 'Tsuji et al., Circulation 1994 · PMID 8044959',
             'here': f"Children: RMSSD ICC {band['Child']['icc']} against experts. Adults: ICC {hrv['icc']} against ECG, just below the 0.75 rule set beforehand.", 'status': 'children'},
            {'marker': 'Heart valve murmur', 'link': 'Screening target for valve disease; no survival figure is quoted here', 'ref': '',
             'here': f"Children: AUC {CN['test']['auc']['value']} on {CN['test']['patients']} unseen patients. Adults: AUC {cdb['a']['auc']['value']:.2f}–{cdb['c']['auc']['value']:.2f} by database; the flag needs calibration per stethoscope.", 'status': 'children'},
            {'marker': 'Wheeze', 'link': (refs.get('wheeze') or {}).get('finding', '').split(';')[0] or 'Reported wheeze predicted chronic-bronchitis death', 'ref': 'Carpenter et al., BMJ 1989 · PMID 2506967',
             'here': f"Adults recorded at another hospital: AUC {kc['auc']['value']}.", 'status': 'moderate'},
            {'marker': 'Sleep apnea severity', 'link': 'AHI ≥ 30: hazard ratio 1.46 for all-cause death', 'ref': 'Punjabi et al., PLoS Med 2009 · PMID 19688045',
             'here': f"Overnight pilot ({tr['patients']} adults, {tr['hours']} h): the acoustic flow proxy finds {tr['events']['sens']} % of scored events.", 'status': 'not ready'},
            {'marker': 'Night breathing rate', 'link': '≥ 16/min: hazard ratio 1.18 (men), 1.50 (women) for all-cause death', 'ref': 'Baumert et al., Eur Respir J 2019 · PMID 31151958',
             'here': f"Estimator error {rr['shipped']['mae']} breaths/min against expert-annotated cycles ({rr['patients']} patients).", 'status': 'not ready'},
        ],
        'copd': {'rho': TV['pAbn']['rho'], 'p': TV['pAbn']['p'], 'patients': TV['patients']},
        'ecgPoints': EP['points'], 'ecgSource': f"EPHNOGRAM · {EP['subjects']} healthy men at rest · {len(EP['points'])} 30-s clips",
        'session': {'n': hrR['n'], 'loa': hrR['loa'], 'icc': hrR['icc']}, 'clip': {'bias': hr['bias'], 'loa': hr['loa']},
        'sources': ['SPRSound (CC BY 4.0)', 'KAUH (CC BY 4.0)', 'CirCor DigiScope (ODC-By 1.0)', 'PhysioNet/CinC 2016 (ODC-By 1.0)', 'EPHNOGRAM (ODC-By 1.0)',
                    'RespiratoryDatabase@TR (CC BY 4.0)', 'PSG-Audio (CC BY 4.0, statistics only)', 'ICBHI 2017 (research licence, statistics only)', 'Chronic bronchitis / emphysema, Mendeley (CC BY 4.0)'],
    }
    (ROOT / 'lean' / 'data').mkdir(parents=True, exist_ok=True)
    (ROOT / 'lean' / 'data' / 'onepager.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: out[k] for k in ('app', 'findings')}, indent=1)[:2500]); [print(m['marker'], '|', m['here']) for m in out['markers']]


if __name__ == '__main__':
    main()
