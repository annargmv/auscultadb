"""NHANES 1999–2018 + NCHS Linked Mortality Files (follow-up through 31 Dec 2019) → meta/nhanes_cohort.csv.

Public-use data of the US National Center for Health Statistics (public domain; cite NCHS). Downloaded from wwwn.cdc.gov
(examination / questionnaire XPT files) and ftp.cdc.gov (linked mortality .dat files). Per participant:
  age (RIDAGEYR), sex (RIAGENDR), 60-s resting pulse (BPXPLS), BMI (BMXBMI), ever / current smoking (SMQ020, SMQ040),
  wheezing or whistling in the chest in the past 12 months (RDQ070), cycle; outcome: death (MORTSTAT) and months from the
  examination to death or censoring (PERMTH_EXM). Raw files are kept in raw/nhanes (and on the private remote store).

  python3 pipeline/nhanes_prep.py
"""
import io, os, urllib.request, urllib.error, hashlib, json
import pandas as pd
from paths import ROOT, META

RAW = ROOT / 'raw' / 'nhanes'; RAW.mkdir(parents=True, exist_ok=True)
CYCLES = {1999: '', 2001: '_B', 2003: '_C', 2005: '_D', 2007: '_E', 2009: '_F', 2011: '_G', 2013: '_H', 2015: '_I', 2017: '_J'}
TABLES = {'DEMO': ['RIDAGEYR', 'RIAGENDR'], 'BPX': ['BPXPLS'], 'BMX': ['BMXBMI'], 'SMQ': ['SMQ020', 'SMQ040'], 'RDQ': ['RDQ070']}


def get(url, dest):
    if not dest.exists():
        with urllib.request.urlopen(url, timeout=120) as r: dest.write_bytes(r.read())
    return dest


def mort(y):
    p = get(f'https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_{y}_{y + 1}_MORT_2019_PUBLIC.dat', RAW / f'MORT_{y}.dat')
    # layout from the NCHS read-in program: SEQN 1-6 (publicid 1-14), ELIGSTAT 15, MORTSTAT 16, UCOD_LEADING 17-19,
    # DIABETES 20, HYPERTEN 21, PERMTH_INT 43-45, PERMTH_EXM 46-48
    df = pd.read_fwf(p, colspecs=[(0, 14), (14, 15), (15, 16), (16, 19), (42, 45), (45, 48)], names=['SEQN', 'ELIGSTAT', 'MORTSTAT', 'UCOD', 'PERMTH_INT', 'PERMTH_EXM'],
                     dtype=str)
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def main():
    rows, files = [], {}
    for y, s in CYCLES.items():
        d = None
        for t, cols in TABLES.items():
            name = f'{t}{s}'
            try: p = get(f'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{y}/DataFiles/{name}.xpt', RAW / f'{name}.xpt')
            except urllib.error.HTTPError as e:
                if e.code != 404: raise
                print(f'  {name}: not collected in this cycle'); continue
            files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
            x = pd.read_sas(p, format='xport'); keep = ['SEQN'] + [c for c in cols if c in x.columns]
            x = x[keep]; d = x if d is None else d.merge(x, on='SEQN', how='left')
        m = mort(y); files[f'MORT_{y}.dat'] = hashlib.sha256((RAW / f'MORT_{y}.dat').read_bytes()).hexdigest()
        d = d.merge(m, on='SEQN', how='left'); d['cycle'] = y; rows.append(d)
        print(y, len(d), 'participants;', int((d['ELIGSTAT'] == 1).sum()), 'eligible for mortality linkage')
    D = pd.concat(rows, ignore_index=True)
    D.to_csv(META / 'nhanes_cohort.csv', index=False)
    (META / 'nhanes_files.json').write_text(json.dumps({'source': 'NHANES 1999–2018 (NCHS, public domain) + NCHS Linked Mortality Files 2019', 'sha256': files}, indent=1))
    print(len(D), 'rows ->', META / 'nhanes_cohort.csv')


if __name__ == '__main__':
    main()
