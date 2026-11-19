"""MedGemma report drafts from the platform's own structured findings → meta/medgemma_drafts.json (research; not shown on the pages).

MedGemma (google/medgemma-4b-it, Health AI Developer Foundations terms) takes text and images, never audio: it only rewords
what the detectors already measured. Each draft is checked mechanically before anyone reads it:
  * every number in the draft must appear in the structured input (a number it computed or invented = fail);
  * the draft must say it is not a diagnosis;
  * none of the dataset's diagnosis labels may appear (the model is not given them, so naming one is a guess);
  * no banned phrasing from the site's writing rules.
The input never contains the expert labels or the diagnosis — only what the page's own analysis found, per site.

Owner's decision pending: AI-written text does not go on the pages until the owner says so. Runs on CPU (bf16, ~9 GB RAM).

  python3 pipeline/medgemma_draft.py [n_per_source]      # default 4 patients each from SPRSound, KAUH, CirCor
"""
import json, re, sys, time, datetime, os
import numpy as np, torch
from paths import ROOT, META, WEB_DATA

MODEL = 'google/medgemma-4b-it'; SEED = 20261118
CARDIAC = {'murmur', 'dmurmur', 's3'}
TERMS = {'crackle': ('fine_crackle', 'coarse_crackle', 'crackle'), 'rhonch': ('rhonchus',), 'wheez': ('wheeze',), 'rub': ('rub',),
         'murmur': ('murmur', 'dmurmur'), 'prolonged exp': ('prolonged_exp',), 'third heart sound': ('s3',), 's3': ('s3',)}
HUMAN = ['heard', 'revealed', 'auscultation showed', 'on examination']   # implies a clinician listened
BANNED = ['honest', 'simply', 'powerful', 'seamless', 'robust', 'at a glance', 'unique', 'first ', 'novel']
SYSTEM = ('You draft a short auscultation summary for a clinician from structured machine findings. Rules: use only facts and numbers '
          'that appear in the input, copied exactly; do not calculate new numbers; do not name a disease or diagnosis; say once that this '
          'is research output and not a diagnosis; the findings come from automatic sound analysis, so write "the analysis detected" and '
          'never "heard", "revealed" or "auscultation showed"; plain clinical prose, at most 120 words, no headings, no lists.')


def structured(p, det):
    """What the page itself measured — never the expert label or the diagnosis."""
    S = {'age_years': p.get('age'), 'sex': p.get('sex'), 'recording': {'sprsound': 'paediatric lung sounds', 'kauh': 'adult lung sounds', 'circor': 'paediatric heart sounds'}[p['src']], 'sites': []}
    for s in p['sites']:
        r = (det.get('rec') or {}).get(s['loc']) or {}; pr = (det.get('pairs') or {}).get(s['loc']) or {}
        e = {'site': s['site']}
        if r.get('dur') is not None: e['duration_s'] = r['dur']
        if p['src'] != 'circor' and r.get('rr') is not None: e['breathing_rate_per_min_estimated'] = r['rr']
        if (r.get('heart') or {}).get('hr') is not None: e['heart_rate_bpm'] = r['heart']['hr']
        F = pr.get('findings', [])
        if p['src'] == 'circor': F = [f for f in F if f.get('type') in CARDIAC]   # lung detectors on a valve recording are artefacts
        e['findings'] = [{k: f[k] for k in ('type', 'phase', 'pos', 'count', 'detail') if f.get(k) is not None} for f in F]
        S['sites'].append(e)
    return S


def numbers(t):
    return [m.replace(',', '') for m in re.findall(r'(?<![\w.])\d+(?:[.,]\d+)?', t)]


def site_names(site):
    base = site.split(' · ')[0].lower(); al = {base}
    if 'valve' in base: al.add(base.split()[0])                        # "aortic valve" -> also "aortic"
    mode = site.split(' · ')[1].split()[0].lower() if ' · ' in site and site.split(' · ')[1].endswith('mode') else None
    return al, mode


def attribution(draft, src):
    """Every finding named in a sentence must exist at a site named in that sentence; 'no findings' at a site must be true."""
    S = [(site_names(e['site']), {f['type'] for f in e['findings']}) for e in src['sites']]; allT = set().union(*[t for _, t in S]) if S else set(); bad = []
    for sent in re.split(r'(?<=[.;])\s+', draft):
        low = sent.lower(); terms = {k for k in TERMS if k in low}
        hit = [(al, mode, T) for (al, mode), T in S if any(re.search(r'\b' + re.escape(a) + r'\b', low) for a in al) and (mode is None or mode in low or not re.search(r'\b(bell|diaphragm|extended)\b', low))]
        if not re.search(r'\b(bell|diaphragm|extended)\b', low):         # no filter mode named: one chest site = union of its modes
            g = {}
            for al, mode, T in hit: g.setdefault(frozenset(al), set()).update(T)
            hit = [(set(a), None, T) for a, T in g.items()]
        if re.search(r'\bno (findings|abnormal)', low):
            if any(T for _, _, T in hit) and not terms: bad.append(f'"no findings" at a site that has findings: {sent.strip()[:90]}')
            if terms and any(not T for _, _, T in hit): bad.append(f'finding and "no findings" for the same site: {sent.strip()[:90]}')
            continue
        for k in terms:
            agg = all if len(terms) == 1 else any                     # one finding named for several sites: each must have it
            ok = agg(bool(set(TERMS[k]) & T) for _, _, T in hit) if hit else bool(set(TERMS[k]) & allT)
            if not ok: bad.append(f'{k} not found {"at the site named" if hit else "anywhere"}: {sent.strip()[:90]}')
    return bad


def check(draft, src, dx_words):
    txt = json.dumps(src, ensure_ascii=False); have = set(numbers(txt)) | {str(len(src['sites']))}   # true derived facts: number of sites, age in months
    if src.get('age_years') is not None and src['age_years'] < 2: have.add(str(round(src['age_years'] * 12)))
    have |= {str(int(float(x))) for x in have if float(x).is_integer()} | {x.rstrip('0').rstrip('.') for x in have if '.' in x}
    num = numbers(draft); bad = [x for x in num if x not in have and x.rstrip('0').rstrip('.') not in have]
    low = draft.lower(); dx = [w for w in dx_words if re.search(r'\b' + re.escape(w.lower()) + r'\b', low)]
    return {'numbers': len(num), 'unsupported': bad, 'saysNotDiagnosis': bool(re.search(r'not (a|intended as a|for) diagnos', low)),
            'namesDiagnosis': dx, 'banned': [w for w in BANNED if w in low], 'impliesClinician': [w for w in HUMAN if w in low], 'words': len(draft.split()),
            'misattributed': attribution(draft, src),
            'pass': not attribution(draft, src) and not bad and not dx and bool(re.search(r'not (a|intended as a|for) diagnos', low)) and not [w for w in BANNED if w in low] and not [w for w in HUMAN if w in low]}


def dx_list(idx):
    w = sorted({p['disease'] for p in idx['patients'] if p.get('disease')} - {'Healthy', 'Normal', 'None'}, key=len)
    return [x for x in w if not re.search(r'\d|/|murmur', x, re.I)] + ['pneumonia', 'asthma', 'COPD', 'bronchitis', 'heart failure', 'bronchiolitis', 'murmur grade']


def summary(out):
    return {'drafts': len(out), 'passed': sum(o['check']['pass'] for o in out), 'withUnsupportedNumbers': sum(bool(o['check']['unsupported']) for o in out),
            'namingDiagnosis': sum(bool(o['check']['namesDiagnosis']) for o in out), 'missingNotDiagnosis': sum(not o['check']['saysNotDiagnosis'] for o in out),
            'impliesClinician': sum(bool(o['check']['impliesClinician']) for o in out), 'misattributed': sum(bool(o['check']['misattributed']) for o in out)}


def recheck():
    f = META / 'medgemma_drafts.json'; R = json.loads(f.read_text()); dx = dx_list(json.loads((WEB_DATA / 'index.json').read_text()))
    for o in R['drafts']: o['check'] = check(o['draft'], o['input'], dx)
    R['summary'] = summary(R['drafts']); f.write_text(json.dumps(R, indent=1, ensure_ascii=False)); print(R['summary'])


def template():
    """Score the page's fixed-template summaries (pipeline/summary_export.js -> build/summaries.json) with the same checks."""
    f = META / 'medgemma_drafts.json'; R = json.loads(f.read_text()); dx = dx_list(json.loads((WEB_DATA / 'index.json').read_text()))
    T = json.loads((ROOT / 'build' / 'summaries.json').read_text()); out = []
    for o in R['drafts']:
        if o['key'] in T: out.append({'key': o['key'], 'src': o['src'], 'text': T[o['key']], 'check': check(T[o['key']], o['input'], dx)})
    R['template'] = {'what': "the page's writtenSummary(): fixed template over the same structured findings, no language model", 'summary': summary(out), 'items': out}
    f.write_text(json.dumps(R, indent=1, ensure_ascii=False)); print('template', R['template']['summary'])
    for o in out:
        if not o['check']['pass']: print(' ', o['key'], {k: v for k, v in o['check'].items() if v and k in ('unsupported', 'misattributed', 'namesDiagnosis', 'impliesClinician', 'banned')})


def main(n):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    torch.set_num_threads(int(os.environ.get('THREADS', '4'))); tok = os.environ.get('HF_TOKEN', 'proxy-injected')
    idx = json.loads((WEB_DATA / 'index.json').read_text()); rng = np.random.default_rng(SEED)
    shards = {}
    for f in (WEB_DATA / 'd').glob('*.json'): shards.update(json.loads(f.read_text()))
    dx_words = dx_list(idx)
    pick = []
    for src in ('sprsound', 'kauh', 'circor'):
        P = [p for p in idx['patients'] if p['src'] == src and p['key'] in shards]; pick += [P[i] for i in rng.choice(len(P), min(n, len(P)), replace=False)]
    proc = AutoProcessor.from_pretrained(MODEL, token=tok); model = AutoModelForImageTextToText.from_pretrained(MODEL, token=tok, dtype=torch.bfloat16).eval()
    out = []
    for p in pick:
        src = structured(p, shards[p['key']]); t0 = time.time()
        msgs = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]}, {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(src, ensure_ascii=False)}]}]
        x = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt')
        with torch.inference_mode(): g = model.generate(**x, max_new_tokens=200, do_sample=False)
        draft = proc.decode(g[0][x['input_ids'].shape[-1]:], skip_special_tokens=True).strip(); c = check(draft, src, dx_words)
        out.append({'key': p['key'], 'src': p['src'], 'input': src, 'draft': draft, 'check': c, 'seconds': round(time.time() - t0, 1)})
        print(f"{p['key']} ({p['src']}) {'PASS' if c['pass'] else 'FAIL'} {c} {time.time() - t0:.0f}s\n  {draft}\n", flush=True)
    res = {'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'model': MODEL, 'licence': 'Health AI Developer Foundations terms',
           'decoding': 'greedy, max 200 new tokens, bf16 on CPU', 'system': SYSTEM, 'shownOnPages': False,
           'summary': {'drafts': len(out), 'passed': sum(o['check']['pass'] for o in out), 'withUnsupportedNumbers': sum(bool(o['check']['unsupported']) for o in out),
                       'namingDiagnosis': sum(bool(o['check']['namesDiagnosis']) for o in out), 'missingNotDiagnosis': sum(not o['check']['saysNotDiagnosis'] for o in out), 'impliesClinician': sum(bool(o['check']['impliesClinician']) for o in out), 'misattributed': sum(bool(o['check']['misattributed']) for o in out)},
           'reviewed': False, 'manualRead': None, 'note': 'Mechanical checks only. Whether each number is attached to the right finding still needs a clinician to read the draft.',
           'drafts': out}
    (META / 'medgemma_drafts.json').write_text(json.dumps(res, indent=1, ensure_ascii=False)); print(res['summary'], '->', META / 'medgemma_drafts.json')


if __name__ == '__main__':
    a = sys.argv[1:]
    recheck() if a == ['--recheck'] else template() if a == ['--template'] else main(int(a[0]) if a else 4)
