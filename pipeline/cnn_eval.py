"""Subgroups for the murmur spectrogram network → web/data/cnn_murmur.json['byBand'].

Rebuilds the trained network from cnn_murmur.json (no retraining), scores the SAME held-out recordings as train_cnn.py and
reports AUC per age band for the network and the current murmur model side by side, plus the flag sensitivity/specificity
per band at the network's fixed threshold. Infants were the weak band for the rule and the feature model.

  python3 pipeline/cnn_eval.py
"""
import json, collections
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from paths import META, FEATURES, WEB_DATA
import train_cnn as T
from train_models import auc_ci, prop


def load_net(J):
    net = T.Net(); sd = net.state_dict()
    for i, L in enumerate(J['conv']):
        sd[f'conv.{i}.weight'] = torch.tensor(L['w'], dtype=torch.float32).reshape(sd[f'conv.{i}.weight'].shape); sd[f'conv.{i}.bias'] = torch.tensor(L['b'], dtype=torch.float32)
    sd['fc.weight'] = torch.tensor(J['fc']['w'], dtype=torch.float32).reshape(1, -1); sd['fc.bias'] = torch.tensor([J['fc']['b']], dtype=torch.float32)
    net.load_state_dict(sd); net.eval(); return net


def main():
    J = json.loads((WEB_DATA / 'cnn_murmur.json').read_text()); net = load_net(J)
    MJ = json.loads((WEB_DATA / 'models.json').read_text()); HM = next(m for m in MJ['models'] if m['id'] == 'heart_murmur'); test = set(HM['testPids'])
    band = {p['pid']: p.get('ageBand') for p in json.load(open(META / 'circor_index.json'))['patients']}
    idx = [json.loads(l) for l in open(T.CNN / 'index.jsonl')]; F = np.fromfile(T.CNN / 'spec.f32', dtype='<f4').reshape(-1, T.BINS)
    TE = []
    for r in idx:
        y = {'Murmur': 1, 'Normal': 0}.get(r['label'])
        if y is None or r['nF'] < T.WIN or r['pid'] not in test: continue
        TE.append({**r, 'y': y, 'Z': T.norm(F[r['offset']:r['offset'] + r['nF']].astype(np.float64))})
    p = T.score(net, TE); y = np.array([r['y'] for r in TE])
    ext = {json.loads(l)['name']: json.loads(l) for l in open(FEATURES / 'circor_ext.jsonl')}
    feat = lambda r: [(lambda v: float('nan') if v is None else float(v))({**ext[r['name']], **(ext[r['name']].get('ext') or {})}.get(f)) for f in HM['features']]
    p0 = T.gb_predict(HM, [feat(r) for r in TE])
    print('check: held-out AUC', round(roc_auc_score(y, p), 3), '(stored', J['test']['auc']['value'], ')')
    thr = J['threshold']; out = []
    for b in ['Neonate', 'Infant', 'Child', 'Adolescent', None]:
        k = np.array([band.get(r['pid']) == b for r in TE])
        if k.sum() < 10 or len(set(y[k])) < 2: continue
        fl = p[k] >= thr; yy = y[k] == 1
        out.append({'band': b or 'unknown', 'recordings': int(k.sum()), 'murmur': int(yy.sum()), 'auc': auc_ci(y[k], p[k]), 'aucCurrent': auc_ci(y[k], p0[k]),
                    'sens': prop(int((fl & yy).sum()), int(yy.sum())), 'spec': prop(int((~fl & ~yy).sum()), int((~yy).sum()))})
        o = out[-1]; print(f"{o['band']:10s} n {o['recordings']:4d} murmur {o['murmur']:3d}  network AUC {o['auc']['value']} ({o['auc']['lo']}–{o['auc']['hi']})  current {o['aucCurrent']['value']}  sens {o['sens']['value']} spec {o['spec']['value']}")
    J['byBand'] = out
    (WEB_DATA / 'cnn_murmur.json').write_text(json.dumps(J, separators=(',', ':')))


if __name__ == '__main__':
    main()
