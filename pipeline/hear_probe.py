"""Discoveries 9 — Google HeAR embeddings vs this platform's own models, on the same held-out patients → meta/hear_probe.json
and discover.json['hear'].

Inputs: build/hear/icbhi.npz and build/hear/circor.npz (pipeline/hear_embed.py; 512-dim embedding of 2-s 16-kHz windows).

Rules fixed before any result was seen (rule 4):
  * probe = logistic regression on standardised embeddings; C from {0.01, 0.1, 1} chosen by patient-grouped 5-fold CV on
    training patients only; nothing else is tuned;
  * ICBHI: one probe per question (crackles, wheezes) per breath cycle; held-out = icbhi_models.json testPids; compared with
    the current cycle model (gradient boosting on cycle features) on the same cycles;
  * CirCor: murmur vs normal per valve recording; held-out = models.json heart_murmur testPids; compared with the murmur
    network (shipped) and the feature model on the same recordings;
  * ΔAUC with a patient bootstrap (2,000 resamples, seed fixed). "Better" only if the 95 % CI is above 0.
  * HeAR is a 303-M-parameter ViT and cannot run in the page; whatever the result, it stays a research comparison.

  python3 pipeline/hear_probe.py && python3 pipeline/merge.py --bump
"""
import json, datetime
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from paths import ROOT, META, FEATURES, WEB_DATA
from train_models import auc_ci
from train_lung_cnn import gb_multi

SEED = 20261118; H = ROOT / 'build' / 'hear'; CS = (0.01, 0.1, 1.0)


def probe(X, y, g):
    """C by grouped CV on the training rows only; returns the fitted pipeline and the CV table."""
    cv = []
    for C in CS:
        s = []
        for a, b in GroupKFold(5).split(X, y, g):
            if len(set(y[b])) < 2: continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X[a], y[a]); s.append(roc_auc_score(y[b], m.predict_proba(X[b])[:, 1]))
        cv.append({'C': C, 'cvAuc': round(float(np.mean(s)), 4)})
    C = max(cv, key=lambda r: r['cvAuc'])['C']
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X, y), cv, C


def boot(y, pids, a, b, n=2000):
    up = np.unique(pids); ix = {u: np.where(pids == u)[0] for u in up}; rng = np.random.default_rng(SEED); d = []
    for _ in range(n):
        i = np.concatenate([ix[u] for u in rng.choice(up, len(up))])
        if 0 < y[i].sum() < len(i): d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]


def compare(name, y, pids, p, others):
    r = {'auc': auc_ci(y, p), 'n': int(len(y)), 'positives': int(y.sum()), 'patients': int(len(set(pids))), 'vs': {}}
    for k, q in others.items():
        dlt = round(float(roc_auc_score(y, p) - roc_auc_score(y, q)), 3); ci = boot(y, pids, p, q)
        r['vs'][k] = {'auc': auc_ci(y, q), 'delta': dlt, 'ci': ci, 'verdict': 'better' if ci[0] > 0 else ('worse' if ci[1] < 0 else 'no clear difference')}
    print(f"{name}: HeAR {r['auc']}  " + '  '.join(f"vs {k} {v['auc']['value']} Δ {v['delta']:+.3f} {v['ci']} {v['verdict']}" for k, v in r['vs'].items()))
    return r


def icbhi():
    E = np.load(H / 'icbhi.npz'); X = E['emb']; pid = E['pid'].astype(str); name = E['name'].astype(str); cyc = E['cycle']
    IM = json.loads((WEB_DATA / 'icbhi_models.json').read_text()); S = next(m for m in IM['models'] if m['id'] == 'icbhi_sound'); test = set(IM['testPids'])
    feats = {json.loads(l)['name']: json.loads(l) for l in open(FEATURES / 'icbhi.jsonl')}
    dev = np.array([n.split('_')[-1] for n in name]); te = np.isin(pid, list(test)); out = {}
    fe = [((feats.get(n) or {}).get('exp') or [None] * (k + 1))[k] if k < len((feats.get(n) or {}).get('exp') or []) else None for n, k in zip(name, cyc)]
    keep = te & np.array([f is not None for f in fe])
    P0 = gb_multi(S, [[(lambda v: float('nan') if v is None else float(v))(fe[i]['f'].get(k)) for k in S['features']] for i in np.where(keep)[0]])
    base = {'crackles': P0[:, 1] + P0[:, 3], 'wheezes': P0[:, 2] + P0[:, 3]}
    for q in ('crackles', 'wheezes'):
        y = E['crackle' if q == 'crackles' else 'wheeze'].astype(int); m, cv, C = probe(X[~te], y[~te], pid[~te]); p = m.predict_proba(X)[:, 1]
        r = compare(f'ICBHI {q}', y[keep], pid[keep], p[keep], {'current cycle model': base[q]})
        r['allHeldOut'] = auc_ci(y[te], p[te]); r['cv'] = cv; r['C'] = C
        r['byDevice'] = [{'device': d, 'cycles': int(((dev == d) & te).sum()), 'auc': auc_ci(y[(dev == d) & te], p[(dev == d) & te])}
                         for d in sorted(set(dev[te])) if ((dev == d) & te).sum() >= 50 and 0 < y[(dev == d) & te].sum() < ((dev == d) & te).sum()]
        out[q] = r
    # how much of the recording device the embedding carries (the ICBHI confound): device from embedding, held-out patients
    dv = sorted(set(dev)); yd = np.array([dv.index(d) for d in dev]); md = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[~te], yd[~te])
    out['deviceFromEmbedding'] = {'accuracy': round(float((md.predict(X[te]) == yd[te]).mean()), 3), 'majority': round(float(np.bincount(yd[te]).max() / te.sum()), 3), 'devices': dv}
    out['trainPatients'] = int(len(set(pid[~te]))); out['testPatients'] = int(len(set(pid[te])))
    return out


def circor():
    import train_cnn as T, cnn_eval as CE
    E = np.load(H / 'circor.npz'); X = E['emb']; pid = E['pid'].astype(str); name = E['name'].astype(str); lab = E['label'].astype(str)
    MJ = json.loads((WEB_DATA / 'models.json').read_text()); HM = next(m for m in MJ['models'] if m['id'] == 'heart_murmur'); test = set(HM['testPids'])
    ok = np.isin(lab, ['Murmur', 'Normal']); y = (lab == 'Murmur').astype(int); te = np.isin(pid, list(test)) & ok; tr = ~np.isin(pid, list(test)) & ok
    m, cv, C = probe(X[tr], y[tr], pid[tr]); p = m.predict_proba(X)[:, 1]
    # the shipped network and the feature model, scored on the same held-out recordings
    J = json.loads((WEB_DATA / 'cnn_murmur.json').read_text()); net = CE.load_net(J)
    idx = {json.loads(l)['name']: json.loads(l) for l in open(T.CNN / 'index.jsonl')}; F = np.fromfile(T.CNN / 'spec.f32', dtype='<f4').reshape(-1, T.BINS)
    ext = {json.loads(l)['name']: json.loads(l) for l in open(FEATURES / 'circor_ext.jsonl')}
    sel = [i for i in np.where(te)[0] if name[i] in idx and idx[name[i]]['nF'] >= T.WIN and name[i] in ext]
    R = [{**idx[name[i]], 'Z': T.norm(F[idx[name[i]]['offset']:idx[name[i]]['offset'] + idx[name[i]]['nF']].astype(np.float64))} for i in sel]
    pc = np.asarray(T.score(net, R)); feat = lambda n: [(lambda v: float('nan') if v is None else float(v))({**ext[n], **(ext[n].get('ext') or {})}.get(f)) for f in HM['features']]
    pf = np.asarray(T.gb_predict(HM, [feat(name[i]) for i in sel])); sel = np.array(sel)
    r = compare('CirCor murmur', y[sel], pid[sel], p[sel], {'murmur network (shipped)': pc, 'feature model': pf})
    r['allHeldOut'] = auc_ci(y[te], p[te]); r['cv'] = cv; r['C'] = C; r['trainPatients'] = int(len(set(pid[tr]))); r['testPatients'] = int(len(set(pid[te])))
    band = {q['pid']: q.get('ageBand') for q in json.load(open(META / 'circor_index.json'))['patients']}; bb = np.array([band.get(x) for x in pid[sel]])
    r['byBand'] = [{'band': b, 'recordings': int((bb == b).sum()), 'hear': auc_ci(y[sel][bb == b], p[sel][bb == b]), 'network': auc_ci(y[sel][bb == b], pc[bb == b])}
                   for b in ('Infant', 'Child', 'Adolescent') if (bb == b).sum() >= 30 and 0 < y[sel][bb == b].sum() < (bb == b).sum()]
    return r


def main():
    torch.set_num_threads(4)
    res = {'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'seed': SEED, 'model': 'google/hear-pytorch (HeAR, ViT-L MAE, 512-dim; Health AI Developer Foundations terms)',
           'ref': 'Baur et al., HeAR — Health Acoustic Representations, arXiv 2403.02522 (2024)',
           'input': {'icbhi': 'one 2-s window centred on each expert breath cycle, 8 kHz → 16 kHz', 'circor': 'mean of up to four consecutive 2-s windows from the start of each valve recording, 4 kHz → 16 kHz'},
           'probe': 'logistic regression on standardised embeddings, C by patient-grouped 5-fold CV on training patients', 'browser': False}
    if (H / 'icbhi.npz').exists(): res['icbhi'] = icbhi()
    if (H / 'circor.npz').exists(): res['circor'] = circor()
    (META / 'hear_probe.json').write_text(json.dumps(res, indent=1))
    D = json.loads((WEB_DATA / 'discover.json').read_text()); D['hear'] = res; (WEB_DATA / 'discover.json').write_text(json.dumps(D, separators=(',', ':')))
    print('->', META / 'hear_probe.json', "and discover.json['hear'] — now run merge.py --bump")


if __name__ == '__main__':
    main()
