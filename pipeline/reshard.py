"""Re-shard one source's per-patient analysis in web/data/d without recomputing anything.

Reads the existing shard files of that source, writes them again at merge.SHARD_BY[src] patients per shard (new shard
numbers after the highest shard of the other sources), deletes the old files and updates index.json. Audio bundles are
untouched. Run merge.py --bump afterwards (new cache key).

  python3 pipeline/reshard.py circor
"""
import json, sys
from paths import WEB_DATA
from merge import SHARD_BY, SHARD


def main(src):
    I = json.load(open(WEB_DATA / 'index.json')); mine = [p for p in I['patients'] if p['src'] == src]
    old = sorted({p['shard'] for p in mine}); data = {}
    for sh in old: data.update(json.load(open(WEB_DATA / 'd' / f'{sh}.json')))
    first = max((p['shard'] for p in I['patients'] if p['src'] != src), default=-1) + 1; size = SHARD_BY.get(src, SHARD)
    for f in WEB_DATA.joinpath('d').glob('*.json'):
        if int(f.stem) in old: f.unlink()
    new = {}
    for i, p in enumerate(mine):
        p['shard'] = first + i // size; new.setdefault(p['shard'], {})[p['key']] = data[p['key']]
    for sh, d in new.items(): json.dump(d, open(WEB_DATA / 'd' / f'{sh}.json', 'w'), separators=(',', ':'))
    json.dump(I, open(WEB_DATA / 'index.json', 'w'), separators=(',', ':'))
    sizes = sorted((WEB_DATA / 'd' / f'{sh}.json').stat().st_size // 1024 for sh in new)
    print(f'{src}: {len(old)} shards -> {len(new)} shards of {size} (numbers {first}–{first + len(new) - 1}); size {sizes[0]}–{sizes[-1]} KB')


if __name__ == '__main__':
    main(sys.argv[1])
