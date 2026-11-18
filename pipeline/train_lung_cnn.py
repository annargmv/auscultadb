"""Crackle / wheeze spectrogram network per breath cycle (PyTorch) → meta/cnn_lung.json (research result; not used by the pages unless it meets the replacement rule).

Input: build/icbhi_cnn (pipeline/icbhi_spec.js: the page's cnnSpec(x, 64) = log power 31–2000 Hz, 10-ms hop, on every ICBHI
recording), normalised per recording with the page's cnnNorm. One 2.56-s window (256 frames) centred on each expert-annotated
breath cycle; two outputs (crackles yes/no, wheezes yes/no), as the clinician's two questions.

Rules fixed before training (rule 4):
  * held-out = icbhi_models.json testPids (the patients the current ICBHI cycle model is scored on); epoch chosen on 15 % of
    the training patients; nothing else is tuned;
  * compared with the current cycle model on the same held-out cycles (patient bootstrap of ΔAUC for crackles and wheezes);
  * the page uses it only if both AUC CIs lie above 0.5, and it replaces the current cycle model only if the mean of the two
    ΔAUCs has a 95 % CI above 0 and neither point estimate is negative.
Research licence: only weights and numbers reach web/.

  python3 pipeline/train_lung_cnn.py
"""
import json, math, datetime, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from paths import ROOT, FEATURES, WEB_DATA, META
from train_models import auc_ci
import train_cnn as T

SEED, BINS, WIN, EPOCHS, LR, WD = 20261118, 64, 256, 25, 1e-3, 1e-3
D = ROOT / 'build' / 'icbhi_cnn'; T0, HOPS = 256 / 2 / 8000, 0.01
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(4)
CYC = ['dur', 'b0med', 'b0p90', 'b0rng', 'b1med', 'b1p90', 'b1rng', 'b2med', 'b2p90', 'b2rng', 'b3med', 'b3p90', 'b3rng', 'cen', 'f50', 'flat',
       'tonP90', 'tonMax', 'tonFrac', 'tonSeg', 'tonHz', 'poly', 'trRate', 'crkRate', 'fineRate', 'trHl', 'trDur']
MEL = [f'm{i}' for i in range(20)] + [f's{i}' for i in range(20)]


class Net(nn.Module):
    def __init__(self):
        super().__init__(); ch = [1, 12, 24, 32]
        self.conv = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 3, padding=1) for i in range(3)]); self.drop = nn.Dropout(0.3); self.fc = nn.Linear(32, 2)
    def forward(self, x):
        for c in self.conv: x = torch.max_pool2d(torch.relu(c(x)), 2)
        return self.fc(self.drop(x.mean((2, 3))))


def win_start(t0, t1, nF):                  # = cnnCycleStart in the page: window centred on the cycle, kept inside the recording
    c = ((t0 + t1) / 2 - T0) / HOPS; f = int(round(c - WIN / 2)); return max(0, min(nF - WIN, f))


def gb_multi(m, X):                         # = modelProba for the exported multiclass gbtree (current ICBHI cycle model)
    mo = m['model']; K = len(mo['baseline']); P = []
    for x in X:
        raw = list(mo['baseline'])
        for it in mo['trees']:
            for k in range(K):
                nodes = it[k]; i = 0
                while not nodes[i][5]:
                    f, t, ml, l, r = nodes[i][:5]; v = x[f]
                    i = (l if ml else r) if (v is None or (isinstance(v, float) and math.isnan(v))) else (l if v <= t else r)
                raw[k] += nodes[i][6]
        mx = max(raw); e = [math.exp(v - mx) for v in raw]; s = sum(e); P.append([v / s for v in e])
    return np.array(P)


def batches(X, Y, bs=64):
    for i in range(0, len(X), bs): yield torch.from_numpy(X[i:i + bs])[:, None], torch.from_numpy(Y[i:i + bs])


def predict(net, X):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 256): out.append(torch.sigmoid(net(torch.from_numpy(X[i:i + 256])[:, None])).numpy())
    return np.concatenate(out)


def main():
    idx = [json.loads(l) for l in open(D / 'index.jsonl')]; F = np.fromfile(D / 'spec.f32', dtype='<f4').reshape(-1, BINS)
    IM = json.loads((WEB_DATA / 'icbhi_models.json').read_text()); S = next(m for m in IM['models'] if m['id'] == 'icbhi_sound'); test = set(IM['testPids'])
    feats = {json.loads(l)['name']: json.loads(l) for l in open(FEATURES / 'icbhi.jsonl')}
    rows = []
    for r in idx:
        if r['nF'] < WIN: continue
        Z = T.norm(F[r['offset']:r['offset'] + r['nF']].astype(np.float64)); ex = feats[r['name']]['exp']
        for k, (t0, t1, c, w) in enumerate(r['cycles']):
            f0 = win_start(t0, t1, r['nF']); fe = ex[k]['f'] if k < len(ex) else None
            rows.append({'pid': r['pid'], 'dev': r['dev'], 'x': Z[f0:f0 + WIN].T.copy(), 'y': [c, w], 'f': fe})
    tr_p = sorted({r['pid'] for r in rows if r['pid'] not in test}); rng = np.random.default_rng(SEED); rng.shuffle(tr_p); val = set(tr_p[:int(round(0.15 * len(tr_p)))])
    TR = [r for r in rows if r['pid'] not in test and r['pid'] not in val]; VA = [r for r in rows if r['pid'] in val]; TE = [r for r in rows if r['pid'] in test]
    X = lambda R: np.stack([r['x'] for r in R]).astype(np.float32); Yf = lambda R: np.array([r['y'] for r in R], np.float32)
    Xtr, Ytr, Xva, Yva, Xte, Yte = X(TR), Yf(TR), X(VA), Yf(VA), X(TE), Yf(TE)
    print(f'cycles train {len(TR)} · validation {len(VA)} · held-out {len(TE)} ({len(test)} patients); crackle/wheeze share in train {Ytr.mean(0).round(3)}')
    pw = torch.tensor((1 - Ytr.mean(0)) / Ytr.mean(0)); net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD); lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    best, hist = -1, []
    for ep in range(EPOCHS):
        net.train(); t0 = time.time(); o = rng.permutation(len(Xtr)); tot = 0
        Xs = Xtr[o].copy(); sh = rng.integers(-20, 21, len(Xs))          # time-shift augmentation (±0.2 s, circular)
        for i in range(len(Xs)): Xs[i] = np.roll(Xs[i], sh[i], axis=1)
        for xb, yb in batches(Xs, Ytr[o]):
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step(); tot += float(loss.detach()) * len(yb)
        pv = predict(net, Xva); va = (roc_auc_score(Yva[:, 0], pv[:, 0]) + roc_auc_score(Yva[:, 1], pv[:, 1])) / 2
        hist.append({'epoch': ep + 1, 'loss': round(tot / len(Xs), 4), 'valAuc': round(va, 4)}); print(f'epoch {ep + 1:2d} loss {tot / len(Xs):.4f} validation mean AUC {va:.4f} {time.time() - t0:.0f}s', flush=True)
        if va > best: best, state, best_ep = va, {k: v.clone() for k, v in net.state_dict().items()}, ep + 1
    net.load_state_dict(state); P = predict(net, Xte)
    # current cycle model on the same held-out cycles (cycles without features are scored by the network only and dropped here)
    keep = np.array([r['f'] is not None for r in TE]); P0 = gb_multi(S, [[(lambda v: float('nan') if v is None else float(v))(r['f'].get(k)) for k in S['features']] for r in np.array(TE)[keep]])
    p0c, p0w = P0[:, 1] + P0[:, 3], P0[:, 2] + P0[:, 3]
    res = {}; pids = np.array([r['pid'] for r in TE])[keep]; up = np.unique(pids); ix = {u: np.where(pids == u)[0] for u in up}; brng = np.random.default_rng(SEED)
    yk, Pk = Yte[keep], P[keep]; d = {'crackles': [], 'wheezes': [], 'mean': []}
    for _ in range(2000):
        i = np.concatenate([ix[u] for u in brng.choice(up, len(up))]); g = []
        for j, (name, base) in enumerate((('crackles', p0c), ('wheezes', p0w))):
            if 0 < yk[i, j].sum() < len(i): g.append(roc_auc_score(yk[i, j], Pk[i, j]) - roc_auc_score(yk[i, j], base[i])); d[name].append(g[-1])
        if len(g) == 2: d['mean'].append(sum(g) / 2)
    ci = {k: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] for k, v in d.items()}
    for j, (name, base) in enumerate((('crackles', p0c), ('wheezes', p0w))):
        res[name] = {'auc': auc_ci(Yte[:, j], P[:, j]), 'aucCurrentSame': round(float(roc_auc_score(yk[:, j], base)), 3), 'aucSame': round(float(roc_auc_score(yk[:, j], Pk[:, j])), 3)}
        res[name]['delta'] = round(res[name]['aucSame'] - res[name]['aucCurrentSame'], 3); res[name]['ci'] = ci[name]
    # thresholds: 90 % specificity on validation (training-side) cycles
    pv = predict(net, Xva); thr = [float(np.quantile(pv[Yva[:, j] == 0, j], 0.90)) for j in range(2)]
    gate = {'deploy': all(res[n]['auc']['lo'] > 0.5 for n in res), 'reasons': [f"{n}: AUC CI not above 0.5" for n in res if res[n]['auc']['lo'] <= 0.5]}
    replace = bool(ci['mean'][0] > 0 and all(res[n]['delta'] >= 0 for n in res))
    by_dev = []
    for dev in sorted({r['dev'] for r in TE}):
        k = np.array([r['dev'] == dev for r in TE])
        if k.sum() >= 50: by_dev.append({'device': dev, 'cycles': int(k.sum()), **{n: (auc_ci(Yte[k, j], P[k, j]) if 0 < Yte[k, j].sum() < k.sum() else None) for j, n in enumerate(('crackles', 'wheezes'))}})
    for n in res: print(f"{n}: network AUC {res[n]['auc']} | same cycles network {res[n]['aucSame']} vs current {res[n]['aucCurrentSame']}  Δ {res[n]['delta']:+.3f} CI {res[n]['ci']}")
    print(f"mean Δ CI {ci['mean']}  gate {gate}  replace current cycle model: {replace}")
    sd = {k: v.numpy() for k, v in net.state_dict().items()}
    out = {'id': 'cnn_lung', 'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'seed': SEED, 'framework': f'PyTorch {torch.__version__}',
           'input': {'bins': BINS, 'hzFrom': 31.25, 'hzTo': 2000, 'hopMs': 10, 'win': WIN}, 'outputs': ['crackles', 'wheezes'],
           'conv': [{'out': int(sd[f'conv.{i}.weight'].shape[0]), 'w': [round(float(v), 7) for v in sd[f'conv.{i}.weight'].ravel()], 'b': [round(float(v), 7) for v in sd[f'conv.{i}.bias']]} for i in range(3)],
           'fc': {'w': [[round(float(v), 7) for v in row] for row in sd['fc.weight']], 'b': [round(float(v), 7) for v in sd['fc.bias']]},
           'thresholds': [round(t, 4) for t in thr], 'bestEpoch': best_ep, 'history': hist,
           'train': {'cycles': len(TR), 'patients': len({r['pid'] for r in TR}), 'validationPatients': len(val)},
           'test': {'cycles': len(TE), 'patients': len(test), **res, 'meanDeltaCi': ci['mean'], 'byDevice': by_dev},
           'gate': gate, 'replace': replace, 'source': 'ICBHI 2017 (Rocha et al., Physiol Meas 2019, PMID 30708353); research licence — weights only'}
    net.eval()
    with torch.no_grad():
        out['parity'] = [{'z': [round(float(v), 6) for v in TE[i]['x'].T.ravel()], 'p': [float(v) for v in torch.sigmoid(net(torch.from_numpy(np.round(TE[i]['x'], 6).astype(np.float32))[None, None]))[0]]} for i in (0, 1)]
    (META / 'cnn_lung.json').write_text(json.dumps(out, separators=(',', ':')))
    print('->', META / 'cnn_lung.json', (META / 'cnn_lung.json').stat().st_size // 1024, 'KB')


if __name__ == '__main__':
    main()
