"""Assemble the static site data from the precompute outputs.

Input : build/index_spr.json, build/index_kauh.json / build/index_circor.json (optional),
        build/p/<key>.json, build/mp3/*.mp3, meta/circor_validation.json (optional)
Output: web/data/index.json   — all patients + validation summaries + data version (cache key)
        web/data/d/<n>.json   — per-patient analysis, 20 patients per shard (keeps the hosted file count low)
        web/audio/<key>.json  — {recording.mp3: base64 MP3} per patient (one request + one cache entry per patient)

The shard/bundle layout exists because a claude.ai artifact can hold at most 512 files.
For a normal web server you could serve the MP3s directly, but the page expects this layout.

  python3 pipeline/merge.py                   full rebuild — needs every dataset in raw/
  python3 pipeline/merge.py --add circor      append one dataset to the existing web/data
  python3 pipeline/merge.py --no-audio        skip the audio bundles
  python3 pipeline/merge.py --bump            re-stamp `version` only, touching no data

--bump exists because `version` is the browser cache key and the discover scripts write
web/data/discover.json *after* merge has stamped it (that is the order in `npm run build`).
Without a re-stamp at the end, a browser that loads the site in between caches the stale
discover.json under the new key and never refetches it. Run it last, after discover.

--add exists because a full rebuild re-precomputes every patient, which needs all of raw/
(SPRSound alone is a 1.2 GB clone). Adding a dataset only touches its own patients: the
existing shards and audio bundles are left exactly as they are, and only new shard numbers
are allocated. The data `version` is always bumped, so browsers drop their old cache.
"""
import collections, base64, datetime, json, sys
from paths import BUILD, META, WEB, WEB_DATA, WEB_AUDIO

SHARD = 20
CAS_T, DAS_T = ('wheeze', 'rhonchus', 'stridor'), ('fine_crackle', 'coarse_crackle', 'rub')


def kauh_validation(rows):
    out = {}
    isC = lambda r: any(t in r['types'] for t in CAS_T)
    isD = lambda r: any(t in r['types'] for t in DAS_T)
    cnt = lambda L, g: [sum(1 for r in L if g(r)), len(L)]
    for m in 'DBE':
        R = [r for r in rows if r['mode'] == m]
        out[m] = {'spec': cnt([r for r in R if r['record'] == 'Normal'], lambda r: not isC(r) and not isD(r)),
                  'sens': cnt([r for r in R if r['record'] != 'Normal'], lambda r: isC(r) or isD(r)),
                  'cas': cnt([r for r in R if 'CAS' in r['record']], isC), 'das': cnt([r for r in R if 'DAS' in r['record']], isD),
                  'recHit': cnt(R, lambda r: r['hit'])}
    return out


def circor_validation(cf):
    """The CirCor cohort rows plus, when it has been run, the full 941-patient validation."""
    C = json.load(open(cf)); out = {'cohort': C['validation']}
    vf = META / 'circor_validation.json'
    if vf.exists():
        V = json.load(open(vf))
        out['full'] = {k: V[k] for k in ('nPatients', 'nRecordings', 'murmurPatient', 'murmurSite', 'byGrade', 'byTiming',
                                         'byAge', 'diastolic', 's3', 'segmentation', 'gates', 'sysDiaByTruth',
                                         'confounds', 'generated', 'licence') if k in V}
    else:
        print('  note: meta/circor_validation.json missing — run `node pipeline/circor_validate.js`')
    return out


def write_patients(patients, audio, first_shard=0):
    """Shard the per-patient analysis and bundle the playback audio. Returns the shards written."""
    shards = {}
    for i, p in enumerate(patients):
        p['shard'] = first_shard + i // SHARD
        shards.setdefault(p['shard'], {})[p['key']] = json.load(open(BUILD / 'p' / f"{p['key']}.json"))
        if audio:
            names = sorted({s['file'].split('#')[1] for s in p['sites']})
            bundle = {n: base64.b64encode(open(BUILD / 'mp3' / n, 'rb').read()).decode() for n in names}
            json.dump(bundle, open(WEB_AUDIO / f"{p['key']}.json", 'w'), separators=(',', ':'))
    for sh, d in shards.items(): json.dump(d, open(WEB_DATA / 'd' / f'{sh}.json', 'w'), separators=(',', ':'))
    return shards


def provenance(I):
    """Stamp each patient with where its recordings come from, read from the meta tables (never typed in):
    SPRSound release(s), CirCor screening campaign, KAUH release. Idempotent, touches no analysis."""
    spr = collections.defaultdict(set)
    for r in json.load(open(META / 'spr_full_index.json'))['recs']: spr[r['key']].add(r['rel'])
    cc = {p['key']: p.get('campaign') for p in json.load(open(META / 'circor_index.json'))['patients']}
    for p in I['patients']:
        if p['src'] == 'sprsound': p['rel'] = sorted(spr.get(p['key'], ())) or None
        elif p['src'] == 'circor': p['rel'] = [cc[p['key']]] if cc.get(p['key']) else None
        elif p['src'] == 'kauh': p['rel'] = ['Mendeley jwyy9np4gv v3']


def bump(I):
    provenance(I)
    I['version'] = datetime.datetime.now(datetime.timezone.utc).isoformat()   # new version => browsers drop the old cache
    json.dump(I, open(WEB_DATA / 'index.json', 'w'), separators=(',', ':'))


def add(src, audio=True):
    """Append one dataset to the existing web/data, leaving the other patients untouched."""
    cf = BUILD / f'index_{src}.json'
    if not cf.exists(): sys.exit(f'{cf} missing — run `node pipeline/precompute.js {src}` first')
    I = json.load(open(WEB_DATA / 'index.json'))
    C = json.load(open(cf))
    have = {p['key'] for p in I['patients']}
    new = [p for p in C['patients'] if p['key'] not in have]
    kept = [p for p in I['patients'] if p['src'] != src]          # replace this source, keep the rest
    dropped = len(I['patients']) - len(kept)
    if dropped:
        for p in I['patients']:
            if p['src'] == src: (WEB_AUDIO / f"{p['key']}.json").unlink(missing_ok=True)
        new = C['patients']
    first = max((p['shard'] for p in kept), default=-1) + 1
    for sh in range(first, 1000):                                  # clear shards this source will reuse
        q = WEB_DATA / 'd' / f'{sh}.json'
        if not q.exists(): break
        q.unlink()
    write_patients(new, audio, first_shard=first)
    I['patients'] = kept + new
    if src == 'circor': I['validationCircor'] = circor_validation(cf)
    elif src == 'kauh': I['validationKAUH'] = {'byMode': kauh_validation(C['validation']['rows']), 'rows': C['validation']['rows']}
    bump(I)
    print(f"+{len(new)} {src} patients ({dropped} replaced), shards from {first}, {len(I['patients'])} total")
    print(f"  web/ now holds {sum(1 for _ in WEB.rglob('*') if _.is_file())} files of the 512-file artifact limit")


def main(audio=True):
    I = json.load(open(BUILD / 'index_spr.json'))
    kf = BUILD / 'index_kauh.json'
    if kf.exists():
        K = json.load(open(kf)); I['patients'] += K['patients']
        I['validationKAUH'] = {'byMode': kauh_validation(K['validation']['rows']), 'rows': K['validation']['rows']}
    cf = BUILD / 'index_circor.json'
    if cf.exists():
        C = json.load(open(cf)); I['patients'] += C['patients']
        I['validationCircor'] = circor_validation(cf)
    (WEB_DATA / 'd').mkdir(parents=True, exist_ok=True); WEB_AUDIO.mkdir(parents=True, exist_ok=True)
    for f in (WEB_DATA / 'd').glob('*.json'): f.unlink()
    shards = write_patients(I['patients'], audio)
    bump(I)
    print(len(I['patients']), 'patients,', len(shards), 'shards', '' if audio else '(audio bundles not rebuilt)')


def bump_only():
    """Re-stamp the data version so browsers drop caches written before the last data change."""
    I = json.load(open(WEB_DATA / 'index.json'))
    old = I['version']; bump(I)
    print(f"version {old} -> {I['version']} (no data changed)")


if __name__ == '__main__':
    audio = '--no-audio' not in sys.argv
    if '--bump' in sys.argv: bump_only()
    elif '--add' in sys.argv: add(sys.argv[sys.argv.index('--add') + 1], audio)
    else: main(audio)
