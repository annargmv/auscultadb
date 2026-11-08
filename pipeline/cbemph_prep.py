"""Prepare the chronic-bronchitis / emphysema respiratory sound set
(Fraiwan, Haider & Khassawneh, Mendeley Data 9hdrddjpny v1, CC BY 4.0).

Input : raw/cbemph/{CB,EMPHYSEMA}/NE_<n>.wav   (the two zips from Mendeley, extracted)
Output: meta/cbemph_index.json                  234 recordings with their diagnosis

Why this set matters here: it is the **same team, same device and same protocol as KAUH**
(3M Littmann 3200, 4 kHz, one upper-left posterior site, subject seated and rested). So its
features are directly comparable with `meta/features/kauh.jsonl` — including KAUH's 35 healthy
adults — with no stethoscope confound. That is the one thing the SPRSound/KAUH lifespan and
device comparisons could never control for.

Not browsable: the set contains no healthy recordings, and a baseline WAV is required to produce
findings (`compareRecordings`). KAUH's healthy WAVs are not in raw/ on this machine, only their
precomputed features. So this is a Discoveries-only dataset until raw/kauh is restored.

The file names carry no metadata — no age, sex or per-file site. Everything known about the
recording conditions comes from the dataset description and is recorded here once.
"""
import collections, json, sys, wave
from paths import RAW, META

CBEMPH_RAW = RAW / 'cbemph'
GROUPS = {'CB': 'Chronic bronchitis', 'EMPHYSEMA': 'Emphysema'}
# constant for the whole set, from the Mendeley description
SITE = {'zone': 'PLU', 'site': 'Posterior left upper chest', 'device': '3M Littmann 3200', 'mode': 'unspecified'}


def root():
    if (CBEMPH_RAW / 'CB').exists(): return CBEMPH_RAW
    c = [p.parent for p in CBEMPH_RAW.glob('**/CB') if p.is_dir()]
    if not c: sys.exit(f'no CB/ and EMPHYSEMA/ under {CBEMPH_RAW} — download Mendeley 9hdrddjpny v1 and unzip there')
    return c[0]


def main():
    d = root(); recs = []
    for g, dx in GROUPS.items():
        for f in sorted((d / g).glob('*.wav')):
            try:
                with wave.open(str(f)) as w:
                    sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
            except Exception as e:
                print(f'  skip {f.name}: {e}'); continue
            recs.append({'key': f'CE_{g[:2]}_{f.stem}', 'name': f.stem, 'dir': g, 'src': 'cbemph',
                         'disease': dx, 'group': g, 'dur': round(n / sr, 2), 'sr': sr, 'ch': ch, 'width': sw,
                         'age': None, 'g': None, **SITE})
    if not recs: sys.exit('no WAVs found')
    out = {'dataset': 'cbemph-9hdrddjpny-v1', 'licence': 'CC BY 4.0',
           'cite': 'M. Fraiwan, N. S. Haider, B. Khassawneh, "Respiratory Sound Dataset of Chronic Bronchitis and Emphysema Patients", Mendeley Data 9hdrddjpny v1 (2026), CC BY 4.0',
           'doi': '10.17632/9hdrddjpny.1', 'device': SITE['device'], 'browsable': False,
           'note': 'Discoveries only: no healthy recordings in the set and no baseline WAV available, so no findings are produced.',
           'dir': str((d).relative_to(d.parents[1])), 'recs': recs}
    json.dump(out, open(META / 'cbemph_index.json', 'w'))
    c = collections.Counter(r['disease'] for r in recs)
    tot = sum(r['dur'] for r in recs)
    print(f"{len(recs)} recordings, {tot/60:.1f} min -> meta/cbemph_index.json")
    for k, v in c.items(): print(f"  {k:22} {v:3}  median {sorted(r['dur'] for r in recs if r['disease']==k)[v//2]:.1f} s")
    print(f"  sample rates: {sorted({r['sr'] for r in recs})}  (resampled to 8 kHz on load, like KAUH)")


if __name__ == '__main__':
    main()
