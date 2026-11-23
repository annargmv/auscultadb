"""Daily retraining of the two models behind the lite app's "Which disease?" and "Longevity" pages.

  python3 pipeline/daily_train.py                    # retrain from the inputs in git, write meta/daily/<date>.json, refresh lite/data
  python3 pipeline/daily_train.py --refresh-nhanes   # first re-download NHANES + the NCHS linked mortality files (CDC)
  python3 pipeline/daily_train.py --remote           # also upload meta/daily/ to the private data store (pipeline/remote.py)

What it does, in order:
  1. records the sha256 of every input (recording features in meta/features, ICBHI index, RespiratoryDatabase@TR GOLD stages,
     NHANES cohort) so each daily result can be traced to its exact data;
  2. reruns pipeline/disease_train.py and pipeline/longevity_train.py — same fixed rules, seeds and settings every day;
     nothing is tuned on the results;
  3. compares the result with the previous daily report: if the inputs did not change, the metrics must not change either
     (a reproducibility check); if they changed, the report lists which inputs and how the metrics moved;
  4. rebuilds lite/data/lite.json so the pages show this run.
What it does not do: add a new database (each one needs a licence check by a person — rule 3 of the project), change a model's
pass rule, or publish anything. Publishing the lite artifact and pushing to git stay manual.
"""
import json, sys, hashlib, datetime, subprocess
from paths import ROOT, META, FEATURES

INPUTS = [FEATURES / f for f in ('spr.0.jsonl', 'spr.1.jsonl', 'icbhi.jsonl', 'kauh.jsonl', 'cbemph.jsonl', 'respdbtr.jsonl')] + \
         [META / 'icbhi_index.json', META / 'respdbtr_validation.json', META / 'nhanes_cohort.csv']
DAILY = META / 'daily'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def run(*cmd):
    print('$', ' '.join(cmd), flush=True); subprocess.run([sys.executable, *cmd], cwd=ROOT / 'pipeline', check=True)


def summary():
    D = json.loads((META / 'disease_model.json').read_text()); L = json.loads((META / 'longevity_model.json').read_text())
    pop = {k: {'patients': p['patients'], 'macroAuc': p['macroAuc'], 'databaseOnly': p['baselines']['databaseOnly'], 'ageOnly': p['baselines']['ageOnly'],
               'lodoAboveChance': sum(l['auc']['lo'] > 0.5 for l in p['leaveOneDatabaseOut']), 'lodoTests': len(p['leaveOneDatabaseOut']), 'passed': p['passed']}
           for k, p in D['populations'].items()}
    T = L['test']
    lon = {'train': L['cohort']['train'], 'test': L['cohort']['test'], 'cIndex': T['cIndex'], 'deltaC_vs_ageSex': T['deltaC_vs_ageSex'], 'deployed': T['deployed'],
           'hrPer10bpm': L['hazardRatios']['hr10'], 'wheeze': L['hazardRatios']['wheeze']}
    return {'disease': pop, 'longevity': lon}


def flat(d, pre=''):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict): o.update(flat(v, pre + k + '.'))
        else: o[pre + k] = v
    return o


def main():
    t0 = datetime.datetime.now(datetime.timezone.utc); DAILY.mkdir(exist_ok=True)
    if '--refresh-nhanes' in sys.argv: run('nhanes_prep.py')
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in INPUTS}
    run('disease_train.py'); run('longevity_train.py')
    S = summary()
    prev = sorted(DAILY.glob('20*.json'))
    prev = [p for p in prev if p.stem != t0.date().isoformat()]
    cmp = None
    if prev:
        P = json.loads(prev[-1].read_text()); changedIn = sorted(k for k in set(inputs) | set(P['inputs']) if inputs.get(k) != P['inputs'].get(k))
        a, b = flat(P['summary']), flat(S); moved = {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
        cmp = {'previous': prev[-1].stem, 'inputsChanged': changedIn, 'metricsChanged': moved,
               'reproducible': (not changedIn and not moved) or bool(changedIn)}
    rep = {'date': t0.date().isoformat(), 'started': t0.isoformat(), 'finished': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'inputs': inputs, 'summary': S, 'comparison': cmp,
           'notDone': 'No new database added automatically (licence check by a person); pass rules unchanged; nothing published or pushed.'}
    out = DAILY / f"{rep['date']}.json"; out.write_text(json.dumps(rep, indent=1)); print('->', out)
    run('lite_build.py')
    if cmp and not cmp['reproducible']:
        print('WARNING: inputs unchanged but metrics moved:', cmp['metricsChanged']); sys.exit(2)
    if '--remote' in sys.argv: run('remote.py', 'push', 'meta/daily')
    print(json.dumps(S['disease'], indent=0)); print(json.dumps(S['longevity'], indent=0))


if __name__ == '__main__':
    main()
