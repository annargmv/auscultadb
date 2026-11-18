"""Google HeAR embeddings (google/hear-pytorch, Health AI Developer Foundations terms) → build/hear/<set>.npz.

  icbhi : one 2-s window centred on every expert-annotated breath cycle (build/icbhi_wav, 8 kHz → 16 kHz)
  circor: up to four consecutive 2-s windows per valve recording (raw/circor, 4 kHz → 16 kHz), averaged per recording

HeAR needs: pip install transformers; the preprocessing code from github.com/Google-Health/hear (cloned to
build/hear/hear-src); model access accepted on Hugging Face (token = environment network secret). Embeddings stay in build/
(ICBHI: research licence) and can be stored with pipeline/remote.py. pipeline/hear_probe.py scores them.

  python3 pipeline/hear_embed.py icbhi
  python3 pipeline/hear_embed.py circor
"""
import json, os, sys, subprocess, time
from math import gcd
import numpy as np, torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from paths import ROOT, META

OUT = ROOT / 'build' / 'hear'; OUT.mkdir(parents=True, exist_ok=True)
SRC = OUT / 'hear-src'
if not SRC.exists(): subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/Google-Health/hear.git', str(SRC)], check=True)
sys.path.insert(0, str(SRC))
from python.data_processing import audio_utils   # noqa: E402
from transformers import AutoModel                # noqa: E402

SR, W = 16000, 32000
torch.set_num_threads(4)


def load(path):
    sr, x = wavfile.read(path); x = x.astype(np.float64)
    if x.ndim > 1: x = x.mean(1)
    x /= max(1e-9, np.abs(x).max())
    g = gcd(sr, SR); return resample_poly(x, SR // g, sr // g).astype(np.float32)


def window(x, centre):
    a = int(round(centre * SR - W / 2)); a = max(0, min(len(x) - W, a))
    w = x[a:a + W] if len(x) >= W else np.pad(x, (0, W - len(x)))
    return w


def embed(model, clips, bs=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(clips), bs):
            out.append(model(audio_utils.preprocess_audio(torch.from_numpy(np.stack(clips[i:i + bs]))), return_dict=True).pooler_output.numpy())
    return np.concatenate(out).astype(np.float32)


def main(which):
    model = AutoModel.from_pretrained('google/hear-pytorch', token=os.environ.get('HF_TOKEN', 'proxy-injected')).eval(); t0 = time.time()
    if which == 'icbhi':
        I = json.load(open(META / 'icbhi_index.json')); keys, E = [], []
        for n, p in enumerate(I['patients']):
            for r in p['recs']:
                x = load(ROOT / 'build' / 'icbhi_wav' / f"{r['name']}.wav")
                clips = [window(x, (a + b) / 2) for a, b, c, w in r['cycles']]
                if clips: E.append(embed(model, clips)); keys += [(p['pid'], r['name'], k, c, w) for k, (a, b, c, w) in enumerate(r['cycles'])]
            if n % 10 == 0: print(f'{n + 1}/{len(I["patients"])} patients, {len(keys)} cycles, {(time.time() - t0) / 60:.1f} min', flush=True)
        np.savez(OUT / 'icbhi.npz', emb=np.concatenate(E), pid=[k[0] for k in keys], name=[k[1] for k in keys], cycle=[k[2] for k in keys],
                 crackle=[k[3] for k in keys], wheeze=[k[4] for k in keys])
    else:
        I = json.load(open(META / 'circor_index.json')); TD = ROOT / 'raw' / 'circor' / 'circor-heart-sound-1.0.3' / 'training_data'; rows, E = [], []
        for n, p in enumerate(I['patients']):
            for loc, s in p['sites'].items():
                x = load(TD / f"{s['name']}.wav"); k = max(1, min(4, len(x) // W))
                E.append(embed(model, [window(x, (i + 0.5) * 2.0) for i in range(k)]).mean(0, keepdims=True)); rows.append((p['pid'], s['name'], s['label']))
            if n % 50 == 0: print(f'{n + 1}/{len(I["patients"])} patients, {len(rows)} recordings, {(time.time() - t0) / 60:.1f} min', flush=True)
        np.savez(OUT / 'circor.npz', emb=np.concatenate(E), pid=[r[0] for r in rows], name=[r[1] for r in rows], label=[r[2] for r in rows])
    print(f'done {which} in {(time.time() - t0) / 60:.1f} min -> {OUT}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'icbhi')
