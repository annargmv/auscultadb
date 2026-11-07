"""Encode playback audio for every browsable recording: MP3, 8 kHz mono, 32 kbps (ffmpeg).

Uses `ffmpeg` from PATH when it is there, otherwise the static binary shipped by the
`imageio-ffmpeg` wheel (`pip install imageio-ffmpeg`) — a plain pip install, no compiler,
which matters on machines where Homebrew would build ffmpeg and its whole dependency tree
from source. Both produce the same file; FFMPEG=/path/to/ffmpeg overrides the choice.

Input : meta/cohort_spr.json + raw/sprsound WAVs, meta/kauh_cohort.json + build/kauh_wav,
        meta/circor_cohort.json + raw/circor/.../training_data (read in place)
Output: build/mp3/<recording name>.mp3   (merge.py bundles them per patient into web/audio/<key>.json)
Analysis always runs on the original WAVs; MP3 is only for playback.
"""
import json, os, shutil, subprocess, sys
from paths import META, SPR_RAW, BUILD, CIRCOR_TRAIN


def ffmpeg_bin():
    if os.environ.get('FFMPEG'): return os.environ['FFMPEG']
    if shutil.which('ffmpeg'): return 'ffmpeg'
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit('no ffmpeg: install it (brew install ffmpeg), or `pip install imageio-ffmpeg`, or set FFMPEG=/path/to/ffmpeg')


FFMPEG = ffmpeg_bin()


MISSING = []


def enc(src, out):
    """Encode one file. Returns 1 if written, 0 if already there, -1 if the source is absent."""
    if out.exists(): return 0
    if not src.exists(): MISSING.append(src); return -1
    subprocess.run([FFMPEG, '-loglevel', 'error', '-y', '-i', str(src), '-ar', '8000', '-ac', '1', '-b:a', '32k', str(out)], check=True)
    return 1

# cohort file -> where that dataset's source WAVs live (dir of the site record -> path)
SETS = [('sprsound', 'cohort_spr.json', lambda r: SPR_RAW / r['dir'] / (r['name'] + '.wav')),
        ('kauh',     'kauh_cohort.json', lambda r: BUILD / 'kauh_wav' / (r['name'] + '.wav')),
        ('circor',   'circor_cohort.json', lambda r: CIRCOR_TRAIN / (r['name'] + '.wav'))]


def main(only=None):
    out = BUILD / 'mp3'; out.mkdir(parents=True, exist_ok=True); total = 0
    for name, cohort, src in SETS:
        if only and name != only: continue
        cf = META / cohort
        if not cf.exists(): continue
        MISSING.clear(); n = skipped = 0
        for p in json.load(open(cf)):
            for r in p['sites'].values():
                v = enc(src(r), out / (r['name'] + '.mp3'))
                if v > 0: n += 1
                elif v == 0: skipped += 1
        total += n
        if MISSING:
            # a dataset whose raw/ is not on this machine: its web/audio bundles are already built,
            # so skip it rather than fail the whole run (see merge.py --add)
            print(f'  {name}: skipped — {len(MISSING)} source WAVs not present (e.g. {MISSING[0]})')
        else:
            print(f'  {name}: {n} encoded, {skipped} already there')
    print(total, 'files encoded ->', out, f'(ffmpeg: {FFMPEG})')

if __name__ == '__main__':
    main(only=sys.argv[1] if len(sys.argv) > 1 else None)
