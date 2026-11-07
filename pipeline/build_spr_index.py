"""Index every SPRSound recording (all releases) into meta/spr_full_index.json.

Input : raw/sprsound  (git clone of github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound)
Output: meta/spr_full_index.json = {recs: [...], summ: {patient_key: [disease, age, sex]}}

Record label: the release's record_annotation when present, otherwise derived from the
event annotations (labelSrc = 'events'): any wheeze/rhonchi/stridor -> CAS, any crackle -> DAS.
"""
import csv, json, sys
from paths import SPR_RAW, META

# json folder -> wav folder, release name
SETS = [('BioCAS2022/train2022_json', 'BioCAS2022/train2022_wav', 'BioCAS2022'),
        ('BioCAS2022/test2022_json', 'BioCAS2022/test2022_wav', 'BioCAS2022'),
        ('BioCAS2023/test2023_json', 'BioCAS2023/test2023_wav', 'BioCAS2023'),
        ('BioCAS2024/test2024_json', 'BioCAS2024/test2024_wav', 'BioCAS2024'),
        ('BioCAS2025/test2025_json', 'BioCAS2025/test2025_wav', 'BioCAS2025')]
CAS = {'Wheeze', 'Rhonchi', 'Stridor', 'Wheeze+Crackle'}
DAS = {'Fine Crackle', 'Coarse Crackle', 'Wheeze+Crackle'}


def derive(ev):
    t = {e['type'] for e in ev}
    c, d = bool(t & CAS), bool(t & DAS)
    return 'CAS & DAS' if c and d else 'CAS' if c else 'DAS' if d else 'Normal'


def main():
    if not SPR_RAW.exists():
        sys.exit(f'missing {SPR_RAW} — run pipeline/fetch_sprsound.sh first')
    recs, seen = [], set()
    for jdir, wdir, rel in SETS:
        for f in sorted((SPR_RAW / jdir).rglob('*.json')):
            name = f.stem
            if (wdir, name) in seen or not (SPR_RAW / wdir / (name + '.wav')).exists():
                continue
            seen.add((wdir, name))
            a = json.load(open(f))
            pid, age, g, loc = name.split('_')[:4]
            ev = [{'start': int(e['start']) / 1000, 'end': int(e['end']) / 1000, 'type': e['type']} for e in a.get('event_annotation', [])]
            rec = a.get('record_annotation')
            recs.append({'rel': rel, 'dir': wdir, 'name': name, 'pid': pid, 'key': str(int(pid or 0)), 'age': float(age), 'g': g, 'loc': loc,
                         'rec': rec, 'ev': ev, 'label': rec if rec else derive(ev), 'labelSrc': 'record' if rec else 'events'})
    summ = {}
    for fn, skip_index in [('SPRSound_patient_summary.csv', True), ("Grand_Challenge'23_patient_summary.csv", False), ("Grand_Challenge'24_patient_summary.csv", False)]:
        for row in csv.DictReader(open(SPR_RAW / 'Patient Summary' / fn)):
            pn = row['patient_num'].strip() or '0'   # blank number -> '0', matching the 390 unnumbered 2023 recordings
            if not pn.isdigit():
                continue                              # one malformed row ('-') in the 2023 summary
            dx = row['disease'].strip().replace('Control group', 'Control Group')
            summ[str(int(pn))] = [dx, str(float(row['age'])), row['gender'].strip()]
    out = META / 'spr_full_index.json'
    json.dump({'recs': recs, 'summ': summ}, open(out, 'w'))
    print(f'{len(recs)} recordings, {len(summ)} patients -> {out}')


if __name__ == '__main__':
    main()
