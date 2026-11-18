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
from paths import META, WEB_DATA

MODEL = 'google/medgemma-4b-it'; SEED = 20261118
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
        e['findings'] = [{k: f[k] for k in ('type', 'phase', 'pos', 'count', 'detail') if f.get(k) is not None} for f in pr.get('findings', [])]
        S['sites'].append(e)
    return S


def numbers(t):
    return [m.replace(',', '') for m in re.findall(r'(?<![\w.])\d+(?:[.,]\d+)?', t)]


def check(draft, src, dx_words):
    txt = json.dumps(src, ensure_ascii=False); have = set(numbers(txt))
    have |= {str(int(float(x))) for x in have if float(x).is_integer()} | {x.rstrip('0').rstrip('.') for x in have if '.' in x}
    num = numbers(draft); bad = [x for x in num if x not in have and x.rstrip('0').rstrip('.') not in have]
    low = draft.lower(); dx = [w for w in dx_words if re.search(r'\b' + re.escape(w.lower()) + r'\b', low)]
    return {'numbers': len(num), 'unsupported': bad, 'saysNotDiagnosis': bool(re.search(r'not (a|intended as a|for) diagnos', low)),
            'namesDiagnosis': dx, 'banned': [w for w in BANNED if w in low], 'impliesClinician': [w for w in HUMAN if w in low], 'words': len(draft.split()),
            'pass': not bad and not dx and bool(re.search(r'not (a|intended as a|for) diagnos', low)) and not [w for w in BANNED if w in low] and not [w for w in HUMAN if w in low]}


def main(n):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    torch.set_num_threads(int(os.environ.get('THREADS', '4'))); tok = os.environ.get('HF_TOKEN', 'proxy-injected')
    idx = json.loads((WEB_DATA / 'index.json').read_text()); rng = np.random.default_rng(SEED)
    shards = {}
    for f in (WEB_DATA / 'd').glob('*.json'): shards.update(json.loads(f.read_text()))
    dx_words = sorted({p['disease'] for p in idx['patients'] if p.get('disease')} - {'Healthy', 'Normal', 'None'}, key=len)
    dx_words = [w for w in dx_words if not re.search(r'\d|/', w)] + ['pneumonia', 'asthma', 'COPD', 'bronchitis', 'heart failure', 'bronchiolitis', 'murmur grade']
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
                       'namingDiagnosis': sum(bool(o['check']['namesDiagnosis']) for o in out), 'missingNotDiagnosis': sum(not o['check']['saysNotDiagnosis'] for o in out), 'impliesClinician': sum(bool(o['check']['impliesClinician']) for o in out)},
           'reviewed': False, 'note': 'Mechanical checks only. Whether each number is attached to the right finding still needs a clinician to read the draft.',
           'drafts': out}
    (META / 'medgemma_drafts.json').write_text(json.dumps(res, indent=1, ensure_ascii=False)); print(res['summary'], '->', META / 'medgemma_drafts.json')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
