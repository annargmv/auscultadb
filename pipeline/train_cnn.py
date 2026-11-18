"""Spectrogram network for murmur (PyTorch) → web/data/cnn_murmur.json.

Input: build/cnn/spec.f32 + index.jsonl (pipeline/circor_spec.js: the page's cnnSpec on every CirCor recording), normalised
with the page's cnnNorm formula. Labels: CirCor per-valve murmur (Murmur / Normal; Unknown skipped).

Protocol (rules fixed before training, rule 4):
  * held-out patients = models.json heart_murmur testPids — the SAME patients the current murmur model is scored on;
  * architecture, training length, learning rate and seed fixed in this file; the epoch is chosen on a validation group of
    TRAINING patients (15 %), never on the held-out ones;
  * scored on held-out recordings and compared with the current model on the same recordings (patient bootstrap of ΔAUC);
  * the page may run it only if it passes the same gate as every Analyze model; it replaces the current murmur model only if
    the ΔAUC 95 % CI lies above 0.

Network: [conv 3×3 → ReLU → max-pool 2×2] × 3 (12, 24, 32 channels), global average pool, dropout 0.3, linear → logit.
Trained on random 3-s crops (300 frames × 32 bins); a recording is scored as the mean window probability (hop 1.5 s).

  python3 pipeline/train_cnn.py
"""
import json, math, datetime, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from paths import ROOT, FEATURES, WEB_DATA
from train_models import auc_ci, prop

SEED, BINS, WIN, HOP, EPOCHS, CROPS, LR, WD = 20261117, 32, 300, 150, 30, 6, 1e-3, 1e-3
CNN = ROOT / 'build' / 'cnn'
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(4)


def q(s, p): return s[int(math.floor(p * (len(s) - 1)))]


def norm(B):                                # = cnnNorm in the page
    med = np.array([q(np.sort(B[:, j]), 0.5) for j in range(B.shape[1])])
    d = B - med; s = np.sort(d.ravel()); sc = max(1e-6, q(s, 0.75) - q(s, 0.25))
    return (d / sc).astype(np.float32)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [1, 12, 24, 32]
        self.conv = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 3, padding=1) for i in range(3)])
        self.drop = nn.Dropout(0.3); self.fc = nn.Linear(32, 1)
    def forward(self, x):                   # x: [N, 1, 32 bins, 300 frames]
        for c in self.conv: x = torch.max_pool2d(torch.relu(c(x)), 2)
        return self.fc(self.drop(x.mean((2, 3)))).squeeze(1)


def windows(Z):
    return [Z[f:f + WIN].T for f in range(0, Z.shape[0] - WIN + 1, HOP)]


def score(net, recs):
    net.eval(); out = []
    with torch.no_grad():
        for r in recs:
            W = windows(r['Z'])
            out.append(float(torch.sigmoid(net(torch.from_numpy(np.stack(W))[:, None])).mean()) if W else float('nan'))
    return np.array(out)


def gb_predict(M, X):                       # = modelProba for the exported current murmur model (binary gbtree)
    m = M['model']; P = []
    for x in X:
        raw = m['baseline'][0]
        for it in m['trees']:
            nodes = it[0]; i = 0
            while not nodes[i][5]:
                f, t, ml, l, r = nodes[i][:5]; v = x[f]
                i = (l if ml else r) if (v is None or (isinstance(v, float) and math.isnan(v))) else (l if v <= t else r)
            raw += nodes[i][6]
        P.append(1 / (1 + math.exp(-raw)))
    return np.array(P)


def main():
    idx = [json.loads(l) for l in open(CNN / 'index.jsonl')]
    F = np.fromfile(CNN / 'spec.f32', dtype='<f4').reshape(-1, BINS)
    MJ = json.loads((WEB_DATA / 'models.json').read_text()); HM = next(m for m in MJ['models'] if m['id'] == 'heart_murmur')
    test = set(HM['testPids'])
    recs = []
    for r in idx:
        y = {'Murmur': 1, 'Normal': 0}.get(r['label'])
        if y is None or r['nF'] < WIN: continue
        recs.append({**r, 'y': y, 'Z': norm(F[r['offset']:r['offset'] + r['nF']].astype(np.float64))})
    tr_p = sorted({r['pid'] for r in recs if r['pid'] not in test}); rng = np.random.default_rng(SEED); rng.shuffle(tr_p)
    val = set(tr_p[:int(round(0.15 * len(tr_p)))])
    TR = [r for r in recs if r['pid'] not in test and r['pid'] not in val]; VA = [r for r in recs if r['pid'] in val]; TE = [r for r in recs if r['pid'] in test]
    pos = sum(r['y'] for r in TR); print(f'recordings train {len(TR)} (murmur {pos}) · validation {len(VA)} · held-out {len(TE)} ({len(test)} patients)')
    net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(TR) - pos) / pos))
    best, best_state, hist = -1, None, []
    for ep in range(EPOCHS):
        net.train(); t0 = time.time(); X, Y = [], []
        for r in TR:
            for _ in range(CROPS):
                f0 = rng.integers(0, r['Z'].shape[0] - WIN + 1); X.append(r['Z'][f0:f0 + WIN].T); Y.append(r['y'])
        order = rng.permutation(len(X)); X = np.stack(X)[order]; Y = np.array(Y, np.float32)[order]; tot = 0
        for i in range(0, len(X), 64):
            xb = torch.from_numpy(X[i:i + 64])[:, None]; yb = torch.from_numpy(Y[i:i + 64])
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step(); tot += float(loss) * len(yb)
        va = roc_auc_score([r['y'] for r in VA], score(net, VA)); hist.append({'epoch': ep + 1, 'loss': round(tot / len(X), 4), 'valAuc': round(va, 4)})
        print(f'epoch {ep + 1:2d}  loss {tot / len(X):.4f}  validation AUC {va:.4f}  {time.time() - t0:.0f}s', flush=True)
        if va > best: best, best_state, best_ep = va, {k: v.clone() for k, v in net.state_dict().items()}, ep + 1
    net.load_state_dict(best_state)
    p = score(net, TE); y = np.array([r['y'] for r in TE])
    # the current murmur model on the very same held-out recordings
    ext = {json.loads(l)['name']: json.loads(l) for l in open(FEATURES / 'circor_ext.jsonl')}
    feat = lambda r: [(lambda v: float('nan') if v is None else float(v))({**ext[r['name']], **(ext[r['name']].get('ext') or {})}.get(f)) for f in HM['features']]
    p0 = gb_predict(HM, [feat(r) for r in TE])
    a1, a0 = roc_auc_score(y, p), roc_auc_score(y, p0)
    pids = np.array([r['pid'] for r in TE]); up = np.unique(pids); ix = {u: np.where(pids == u)[0] for u in up}; brng = np.random.default_rng(SEED); d = []
    for _ in range(2000):
        i = np.concatenate([ix[u] for u in brng.choice(up, len(up))])
        if 0 < y[i].sum() < len(i): d.append(roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], p0[i]))
    ci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    # flag threshold: 90 % specificity on the validation (training-side) patients, as for the other models
    pv = score(net, VA); yv = np.array([r['y'] for r in VA]); thr = float(np.quantile(pv[yv == 0], 0.90))
    fl = p >= thr; tp, fn, fp, tn = int((fl & (y == 1)).sum()), int((~fl & (y == 1)).sum()), int((fl & (y == 0)).sum()), int((~fl & (y == 0)).sum())
    auc = auc_ci(y, p)
    gate_why = []
    if auc['value'] < 0.70: gate_why.append(f"AUC {auc['value']} < 0.70")
    if auc['lo'] <= 0.5: gate_why.append('no better than chance')
    acc = float(np.mean(fl == (y == 1))); maj = float(np.mean(y == 0))
    if acc <= maj: gate_why.append(f'accuracy {100 * acc:.1f} % does not beat always answering “no murmur” ({100 * maj:.1f} %)')
    replace = ci[0] > 0
    print(f'HELD-OUT  network AUC {a1:.3f} {auc}  current model AUC {a0:.3f} (published {HM["test"]["perClass"][1]["auc"]["value"]})  ΔAUC {a1 - a0:+.3f} CI {ci[0]:+.3f} to {ci[1]:+.3f}')
    print(f'  at the 90 %-specificity threshold (set on validation patients {thr:.3f}): sens {tp}/{tp + fn}, spec {tn}/{tn + fp}; gate {"PASS" if not gate_why else gate_why}; replace current: {replace}')
    sd = {k: v.numpy() for k, v in net.state_dict().items()}
    out = {'id': 'cnn_murmur', 'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'seed': SEED, 'framework': f'PyTorch {torch.__version__}',
           'input': {'bins': BINS, 'hzFrom': 31.25, 'hzTo': 1000, 'hopMs': 10, 'win': WIN, 'hop': HOP},
           'conv': [{'out': int(sd[f'conv.{i}.weight'].shape[0]), 'w': [round(float(v), 7) for v in sd[f'conv.{i}.weight'].ravel()], 'b': [round(float(v), 7) for v in sd[f'conv.{i}.bias']]} for i in range(3)],
           'fc': {'w': [round(float(v), 7) for v in sd['fc.weight'].ravel()], 'b': round(float(sd['fc.bias'][0]), 7)},
           'threshold': round(thr, 4), 'bestEpoch': best_ep, 'history': hist,
           'train': {'recordings': len(TR), 'patients': len({r['pid'] for r in TR}), 'validationPatients': len(val)},
           'test': {'recordings': len(TE), 'patients': len(test), 'auc': auc, 'sens': prop(tp, tp + fn), 'spec': prop(tn, tn + fp)},
           'vsCurrent': {'aucCurrent': round(a0, 3), 'auc': round(a1, 3), 'delta': round(a1 - a0, 3), 'ci': [round(ci[0], 3), round(ci[1], 3)], 'replace': replace},
           'gate': {'deploy': not gate_why, 'reasons': gate_why}}
    # parity: two held-out windows and their PyTorch probabilities, for the page's JS forward pass
    net.eval(); par = []
    with torch.no_grad():
        for r in TE[:2]:
            w = r['Z'][:WIN]; par.append({'z': [round(float(v), 6) for v in w.ravel()], 'p': float(torch.sigmoid(net(torch.from_numpy(np.round(w, 6).T.copy())[None, None])))})
    out['parity'] = par
    (WEB_DATA / 'cnn_murmur.json').write_text(json.dumps(out, separators=(',', ':')))
    print('->', WEB_DATA / 'cnn_murmur.json', (WEB_DATA / 'cnn_murmur.json').stat().st_size // 1024, 'KB')


if __name__ == '__main__':
    main()
