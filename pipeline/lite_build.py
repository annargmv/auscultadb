"""AuscultaDB Lite → lite/ (a lighter, focused version of the full app; web/ is untouched).

Same batch of 10 recordings and the same validation lines as the clinical brief (pipeline/lean_build.py → lean/data/brief.json),
plus what the app needs to analyse a recording itself: the murmur network, the learned beat finder and the breath-sound model,
run by lite/engine.js (the full app's own analysis code, copied by pipeline/lite_engine.js).
For KAUH lung recordings the detected continuous-sound segments (full app's precomputed findings) are added for the player.

  python3 pipeline/lean_build.py && node pipeline/lite_engine.js && python3 pipeline/lite_build.py && npm run test:lite
"""
import json, shutil
from paths import ROOT, FEATURES, WEB_DATA
from disease_train import KAU

LITE = ROOT / 'lite'; LEAN = ROOT / 'lean'


def main():
    (LITE / 'data').mkdir(parents=True, exist_ok=True); shutil.rmtree(LITE / 'audio', ignore_errors=True); shutil.copytree(LEAN / 'audio', LITE / 'audio')
    B = json.loads((LEAN / 'data' / 'brief.json').read_text())
    I = json.loads((WEB_DATA / 'index.json').read_text()); P = {p['key']: p for p in I['patients']}
    K = {json.loads(l)['name']: json.loads(l) for l in open(FEATURES / 'kauh.jsonl')}
    for c in B['cases']:
        c['kind'] = c['group']
        if c['id'].startswith('kauh-'):
            k = K[c['id'][5:]]; p = P[k['key']]; c['ageYears'] = k['age']
            sh = json.loads((WEB_DATA / 'd' / f"{p['shard']}.json").read_text())[k['key']]
            F = (sh['pairs'].get(k['loc']) or {}).get('findings', [])
            c['regions'] = [{'t0': a, 't1': b, 'label': f['type']} for f in F if f['type'] in ('wheeze', 'rhonchus') for a, b in f.get('events', [])]
        elif c['id'].startswith('circor-'):
            c['ageBand'] = 'Child'
    MO = json.loads((WEB_DATA / 'models.json').read_text())
    ls = next(m for m in MO['models'] if m['id'] == 'lung_sound')
    (LITE / 'data' / 'lung_sound.json').write_text(json.dumps(ls, separators=(',', ':')))
    for f in ('cnn_murmur.json', 'segmenter.json'): shutil.copy(WEB_DATA / f, LITE / 'data' / f)
    # disease atlas (pipeline/disease_train.py) and longevity model (pipeline/longevity_train.py): copied as computed
    DM = json.loads((ROOT / 'meta' / 'disease_model.json').read_text()); oof = DM['populations']['adults']['batchOof']
    for c in B['cases']:
        key = K[c['id'][5:]]['key'] if c['id'].startswith('kauh-') else c['id'][3:].split('_')[0] if c['id'].startswith('tr-') else None
        if key in oof: c['diseaseOof'] = oof[key]
        if c['id'].startswith('kauh-'): c['diagnosisClass'] = KAU.get(K[c['id'][5:]]['disease'], 'Other')
        elif c['id'].startswith('tr-'): c['diagnosisClass'] = 'COPD'
    for pop in DM['populations'].values(): pop.pop('batchOof', None)
    B['disease'] = DM; B['longevity'] = json.loads((ROOT / 'meta' / 'longevity_model.json').read_text())
    (LITE / 'data' / 'lite.json').write_text(json.dumps(B, ensure_ascii=False, separators=(',', ':')))
    print(len(B['cases']), 'cases ->', LITE, '| regions:', {c['id']: len(c.get('regions', [])) for c in B['cases'] if 'regions' in c})


if __name__ == '__main__':
    main()
