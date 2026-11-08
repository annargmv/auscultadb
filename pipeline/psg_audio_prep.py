"""PSG-Audio (Korompili et al., Sci Data 2021, doi 10.1038/s41597-021-00977-w) → overnight validation set
for the apnea / AHI feature (Discoveries 12).

Source: the public Hugging Face mirror `dust-systems/psg-audio` (CC BY 4.0; a partial copy of Science Data Bank
doi 10.11922/sciencedb.00345, whose original FTP access is time-limited). Version V3: hourly EDF parts with 20
channels, including `Tracheal` (contact microphone on the trachea) and `Mic` (ambient, ~1 m above the bed), both
48 kHz, plus the scored `.rml` (Philips Sleepware G3) with sleep stages and respiratory events.

Per patient this script:
  * reads the .rml: respiratory events (Obstructive / Central / Mixed apnea, Hypopnea), sleep-stage change
    points, and the list of hourly segments;
  * downloads each hourly EDF in turn, keeps ONLY the two microphone channels, decimates 48 kHz → 8 kHz
    (polyphase anti-alias FIR — the platform's SR), writes them as 16-bit WAV under raw/psg_audio/<pid>/,
    then deletes the 700 MB EDF;
  * writes meta/psg_audio_index.json: per patient, per hour — events, sleep time, WAV names.

Selection is fixed before any scoring: V3 patients that have an .rml AND every hourly EDF part in the mirror,
sorted by id, first N (default 10). Nothing is tuned here; this only prepares data.

The audio stays in raw/ (git-ignored) and is NEVER copied to web/: overnight bedroom audio can hold speech.

  python3 pipeline/psg_audio_prep.py [N]
"""
import json, re, sys, os, ssl, shutil, time, urllib.request, urllib.parse, wave
import numpy as np
from scipy.signal import resample_poly
from paths import RAW, META

REPO = 'https://huggingface.co/datasets/dust-systems/psg-audio'
API = 'https://huggingface.co/api/datasets/dust-systems/psg-audio/tree/main?recursive=true'
OUT = RAW / 'psg_audio'
SR_OUT, SR_IN = 8000, 48000
CHANNELS = {'Tracheal': 'trach', 'Mic': 'mic'}
RESP = {'ObstructiveApnea': 'OA', 'CentralApnea': 'CA', 'MixedApnea': 'MA', 'Hypopnea': 'H'}
CTX = ssl.create_default_context(cafile=os.environ.get('SSL_CERT_FILE') or '/root/.ccr/ca-bundle.crt') \
    if os.path.exists('/root/.ccr/ca-bundle.crt') else ssl.create_default_context()


def url_of(path):
    # file names literally contain "%5B001%5D", so every '%' must itself be escaped
    return f"{REPO}/resolve/main/" + urllib.parse.quote(path, safe='/')


def listing():
    files, url = [], API
    while url:
        r = urllib.request.urlopen(url, context=CTX, timeout=120)
        files += json.load(r)
        link = r.headers.get('Link') or ''
        url = link.split(';')[0].strip('<> ') if 'rel="next"' in link else None
    return [f for f in files if f['type'] == 'file']


def fetch(path, dest, size=None, tries=5):
    """Download with retries (the CDN occasionally resets long transfers); a short file counts as a failure."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(url_of(path), context=CTX, timeout=600) as r, open(dest, 'wb') as f:
                shutil.copyfileobj(r, f, 1 << 22)
            if size is None or dest.stat().st_size == size: return
            raise IOError(f'short file {dest.stat().st_size} of {size}')
        except (OSError, IOError) as e:
            if i == tries - 1: raise
            wait = 2 ** (i + 1); print(f'    retry {i + 1} in {wait}s after: {e}'); time.sleep(wait)


def parse_rml(txt):
    """Respiratory events, sleep-stage change points and the hourly segments of one study."""
    ev = [{'type': RESP[t], 'start': float(s), 'dur': float(d)}
          for f, t, s, d in re.findall(r'<Event Family="(\w+)" Type="(\w+)" Start="([\d.]+)" Duration="([\d.]+)"', txt)
          if f == 'Respiratory' and t in RESP]
    # the .rml holds the human scorer's staging (UserStaging) AND the software's (MachineStaging): use the human one
    user = re.search(r'<UserStaging>(.*?)</UserStaging>', txt, re.S)
    stages = [(t, float(s)) for t, s in re.findall(r'<Stage Type="(\w+)" Start="([\d.]+)"', user.group(1) if user else '')]
    segs = [float(d) for d in re.findall(r'<Segment>\s*<StartTime>[^<]*</StartTime>\s*<Duration>([\d.]+)</Duration>', txt)]
    total = float(re.search(r'<Session>.*?<Duration>([\d.]+)</Duration>', txt, re.S).group(1))
    gender = (re.search(r'<Gender>(\w+)</Gender>', txt) or [None, None])[1]
    return ev, stages, segs, total, gender


def sleep_seconds(stages, t0, t1, end):
    """Seconds of scored sleep (any stage except Wake / NotScored) inside [t0, t1)."""
    s = 0.0
    for i, (st, a) in enumerate(stages):
        b = stages[i + 1][1] if i + 1 < len(stages) else end
        if st in ('Wake', 'NotScored'): continue
        s += max(0.0, min(b, t1) - max(a, t0))
    return s


def edf_channels(path, names):
    """Read the named signals from an EDF file as int16 digital values (memory-mapped, no extra copy)."""
    with open(path, 'rb') as f: h = f.read(256)
    ns, nrec, hbytes = int(h[252:256]), int(h[236:244]), int(h[184:192])
    with open(path, 'rb') as f: H = f.read(hbytes)
    field = lambda off, w, i: H[256 + off * ns + i * w: 256 + off * ns + (i + 1) * w].decode('latin1').strip()
    # header layout: label16 trans80 dim8 pmin8 pmax8 dmin8 dmax8 pre80 nsamp8 reserved32
    labels = [field(0, 16, i) for i in range(ns)]
    nsamp = [int(field(16 + 80 + 8 * 5 + 80, 8, i)) for i in range(ns)]
    rec = sum(nsamp)
    raw = np.memmap(path, dtype='<i2', mode='r', offset=hbytes, shape=(nrec, rec))
    out = {}
    for n in names:
        i = labels.index(n); o = sum(nsamp[:i])
        out[n] = (np.asarray(raw[:, o:o + nsamp[i]]).reshape(-1), nsamp[i])
    return out, labels


def to_wav(x, path):
    y = resample_poly(x.astype(np.float32), 1, SR_IN // SR_OUT)
    y = np.clip(np.round(y), -32768, 32767).astype('<i2')
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR_OUT); w.writeframes(y.tobytes())
    return len(y) / SR_OUT


def main(n=10):
    files = listing()
    rml = {re.search(r'(\d+-\d+)\.rml$', f['path']).group(1): f['path'] for f in files if f['path'].startswith('V3/APNEA_RML/')}
    parts = {}
    for f in files:
        m = re.match(r'V3/APNEA_EDF/(\d+-\d+)/.*\[(\d+)\]\.edf$', urllib.parse.unquote(f['path']))
        if m: parts.setdefault(m.group(1), {})[int(m.group(2))] = f['path']
    sizes = {f['path']: f.get('size') for f in files}
    OUT.mkdir(parents=True, exist_ok=True)
    idx_path = META / 'psg_audio_index.json'
    idx = json.load(open(idx_path)) if idx_path.exists() else {'source': {}, 'patients': []}
    idx['source'] = {'name': 'PSG-Audio (V3)', 'mirror': REPO, 'doi': '10.11922/sciencedb.00345',
                     'paper': '10.1038/s41597-021-00977-w', 'licence': 'CC BY 4.0',
                     'hospital': 'Sismanoglio – Amalia Fleming General Hospital, Athens',
                     'channels': {'trach': 'Tracheal contact microphone (Clockaudio CTH100)', 'mic': 'Ambient microphone ~1 m above bed (Behringer ECM8000)'},
                     'selection': f'V3 patients with an .rml and every hourly EDF part in the mirror, sorted by id, first {n}'}
    done = {p['pid'] for p in idx['patients']}
    eligible = []
    for pid in sorted(rml):
        txt = urllib.request.urlopen(url_of(rml[pid]), context=CTX, timeout=120).read().decode('utf-8-sig')
        ev, stages, segs, total, gender = parse_rml(txt)
        have = parts.get(pid, {})
        if not segs or sorted(have) != list(range(1, len(segs) + 1)): continue
        eligible.append((pid, ev, stages, segs, total, gender))
        if len(eligible) >= n: break
    print(f'{len(eligible)} eligible patients (of {len(rml)} with annotations)')
    for pid, ev, stages, segs, total, gender in eligible:
        if pid in done: print(pid, 'already prepared'); continue
        d = OUT / pid; d.mkdir(exist_ok=True); hours = []; t0 = 0.0
        for k, seg in enumerate(segs, 1):
            if not all((d / f'h{k}_{tag}.wav').exists() for tag in CHANNELS.values()):   # resume: skip finished hours
                tmp = OUT / 'tmp.edf'
                fetch(parts[pid][k], tmp, sizes.get(parts[pid][k]))
                sig, labels = edf_channels(tmp, list(CHANNELS))
                for name, tag in CHANNELS.items():
                    x, ns = sig[name]
                    assert ns == SR_IN, f'{name} is {ns} Hz, expected {SR_IN}'
                    to_wav(x, d / f'h{k}_{tag}.wav')
                del sig; tmp.unlink()
            t1 = t0 + seg
            hev = [e for e in ev if t0 <= e['start'] < t1]
            hours.append({'k': k, 'start': t0, 'dur': seg, 'sleep': sleep_seconds(stages, t0, t1, total),
                          'wav': {tag: f'{pid}/h{k}_{tag}.wav' for tag in CHANNELS.values()},
                          'events': [{**e, 'start': round(e['start'] - t0, 2)} for e in hev]})
            print(f'  {pid} hour {k}/{len(segs)}: {len(hev)} scored events, sleep {hours[-1]["sleep"] / 60:.0f} min')
            t0 = t1
        idx['patients'].append({'pid': pid, 'sex': gender, 'hours': hours, 'recorded': total,
                                'events': len(ev), 'sleep': sum(h['sleep'] for h in hours)})
        json.dump(idx, open(idx_path, 'w'), indent=1)   # checkpoint after every patient
    print(f'-> {idx_path} ({len(idx["patients"])} patients); audio in {OUT}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
