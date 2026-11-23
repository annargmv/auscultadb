"""Remote data store for AuscultaDB: a PRIVATE Hugging Face dataset repository that holds what does not fit in git.

The machine has a small, temporary disk and is replaced between sessions; raw datasets (raw/) and expensive intermediates
(build/) are kept here instead and pulled back on demand, one folder at a time.

  layout in the repo (same paths as on disk):
    raw/circor/...        CirCor DigiScope 1.0.3            ODC-By 1.0
    raw/icbhi/...         ICBHI 2017 (working files only)   research use — this is why the repo must stay PRIVATE
    raw/psg_audio/...     PSG-Audio pilot (8-kHz mics)      CC BY 4.0
    build/<name>/...      intermediates (spectrograms, frames, resampled WAV)
    MANIFEST.json         path → size, sha256, licence, uploaded at — written by `push`

  python3 pipeline/remote.py status                 # what is local, what is remote, what differs
  python3 pipeline/remote.py push raw/circor build/cnn
  python3 pipeline/remote.py pull raw/icbhi         # restore a folder after a fresh machine
  python3 pipeline/remote.py free build/cnn         # delete the local copy, only if the remote copy matches

Authentication: the Hugging Face token is a network secret of the cloud environment, injected by the proxy; the client
gets a placeholder (HF_TOKEN=proxy-injected). Elsewhere, set HF_TOKEN to a real token with write access. Repository id:
$AUSCULTADB_REMOTE, default <your user>/auscultadb-data. The script refuses to upload to a repository that is not private.
"""
import hashlib, json, os, shutil, sys, datetime, tarfile
from pathlib import Path
from paths import ROOT

os.environ.setdefault('HF_TOKEN', 'proxy-injected')
from huggingface_hub import HfApi, snapshot_download   # noqa: E402

LICENCE = {'raw/nhanes': 'public domain (US government work; NCHS/CDC NHANES and Linked Mortality Files — cite the data)', 'meta/daily': 'derived results of pipeline/daily_train.py', 'raw/circor': 'ODC-By 1.0', 'raw/icbhi': 'research use only (ICBHI 2017) — never public', 'raw/psg_audio': 'CC BY 4.0', 'raw/cinc2016': 'ODC-By 1.0 (PhysioNet/CinC Challenge 2016)', 'raw/respdbtr': 'CC BY 4.0 (RespiratoryDatabase@TR, Mendeley)', 'raw/ephnogram': 'ODC-By 1.0 (EPHNOGRAM, PhysioNet)',
           'build/icbhi_wav': 'derived from ICBHI — research use only', 'build/icbhi_cnn': 'derived from ICBHI — research use only',
           'build/cnn': 'derived from CirCor — ODC-By 1.0', 'build/hear': 'HeAR embeddings (HAI-DEF terms) of ICBHI (research use only) and CirCor (ODC-By 1.0)', 'build/seg': 'derived from CirCor — ODC-By 1.0'}
SKIP = {'.git', '__pycache__', '.DS_Store'}
PACK_OVER, PART_BYTES = 1000, 1_000_000_000   # folders with more files are stored as ~1 GB tar parts (HF: ≤ 10,000 files per directory)
api = HfApi(token=os.environ['HF_TOKEN'])


def repo_id():
    return os.environ.get('AUSCULTADB_REMOTE') or f"{api.whoami()['name']}/auscultadb-data"


def ensure_private(rid):
    api.create_repo(rid, repo_type='dataset', private=True, exist_ok=True)
    if not api.repo_info(rid, repo_type='dataset').private:
        sys.exit(f'{rid} is not private — refusing: ICBHI may not be republished (owner rule 3). Make it private first.')


def files(folder):
    base = (ROOT / folder).resolve()
    for p in sorted(base.rglob('*')):
        if p.is_file() and not (set(p.relative_to(base).parts) & SKIP): yield p, f"{folder}/{p.relative_to(base).as_posix()}"


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def manifest(rid):
    try:
        p = api.hf_hub_download(rid, 'MANIFEST.json', repo_type='dataset'); return json.load(open(p))
    except Exception:
        return {}


def push_packed(rid, folder, todo, M):
    """Many small files → tar parts under archives/<folder>/part-NNN.tar, written, uploaded and deleted one at a time."""
    tmp = ROOT / 'build' / '_remote_tmp'; tmp.mkdir(parents=True, exist_ok=True); allf = list(files(folder)); part, size, k, members = None, 0, 0, {}
    def close():
        nonlocal part
        if part is None: return
        part.close(); name = f'archives/{folder}/part-{k:03d}.tar'
        api.upload_file(path_or_fileobj=str(tmp / 'part.tar'), path_in_repo=name, repo_id=rid, repo_type='dataset', commit_message=f'push {name}')
        (tmp / 'part.tar').unlink(); print(f'  {name} uploaded'); part = None
    for p, rp in allf:
        if part is None or size >= PART_BYTES:
            close(); k += 0 if part is None and size == 0 else 1; part = tarfile.open(tmp / 'part.tar', 'w'); size = 0
        part.add(p, arcname=rp); size += p.stat().st_size; members[rp] = k
    close(); shutil.rmtree(tmp, ignore_errors=True)
    return members


def push(rid, folders):
    ensure_private(rid); M = manifest(rid)
    for folder in folders:
        src = (ROOT / folder).resolve()
        if not src.exists(): print(f'skip {folder}: not on disk'); continue
        todo = [(p, rp) for p, rp in files(folder) if M.get(rp, {}).get('size') != p.stat().st_size or M[rp].get('sha256') != sha(p)]
        if not todo: print(f'{folder}: remote copy already matches'); continue
        print(f'{folder}: uploading {len(todo)} files ({sum(p.stat().st_size for p, _ in todo) / 1e6:.0f} MB)')
        allf = list(files(folder)); packed = len(allf) > PACK_OVER
        if packed: parts = push_packed(rid, folder, todo, M)
        else: api.upload_folder(repo_id=rid, repo_type='dataset', folder_path=str(src), path_in_repo=folder, commit_message=f'push {folder}',
                          ignore_patterns=[f'{x}/**' for x in SKIP] + [f'**/{x}/**' for x in SKIP])
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for p, rp in (allf if packed else todo):
            M[rp] = {'size': p.stat().st_size, 'sha256': sha(p), 'licence': next((v for k, v in LICENCE.items() if rp.startswith(k)), 'see README'), 'uploaded': now}
            if packed: M[rp]['part'] = f'archives/{folder}/part-{parts[rp]:03d}.tar'
        api.upload_file(path_or_fileobj=json.dumps(M, indent=0).encode(), path_in_repo='MANIFEST.json', repo_id=rid, repo_type='dataset', commit_message=f'manifest: {folder}')
        print(f'{folder}: done')


def pull(rid, folders):
    M = manifest(rid)
    for folder in folders:
        parts = sorted({m['part'] for rp, m in M.items() if rp.startswith(folder + '/') and m.get('part')})
        if parts:
            for name in parts:
                p = api.hf_hub_download(rid, name, repo_type='dataset')
                with tarfile.open(p) as t: t.extractall(ROOT, filter='data')
                os.remove(os.path.realpath(p)); print(f'  {name} unpacked')
        else:
            snapshot_download(rid, repo_type='dataset', allow_patterns=[f'{folder}/**'], local_dir=str(ROOT))
        bad = [rp for rp, m in M.items() if rp.startswith(folder + '/') and (not (ROOT / rp).exists() or sha(ROOT / rp) != m['sha256'])]
        print(f'{folder}: restored under {ROOT / folder}; ' + ('all files match the manifest' if not bad else f'{len(bad)} files do not match the manifest'))


def status(rid):
    M = manifest(rid); by = {}
    for rp, m in M.items(): by.setdefault('/'.join(rp.split('/')[:2]), [0, 0]); by['/'.join(rp.split('/')[:2])][0] += 1; by['/'.join(rp.split('/')[:2])][1] += m['size']
    print(f'remote {rid} (private): {len(M)} files')
    for k, (n, s) in sorted(by.items()):
        local = ROOT / k; print(f'  {k:22s} remote {n:6d} files {s / 1e9:6.2f} GB   local: {"yes" if local.exists() else "no"}')


def free(rid, folders):
    M = manifest(rid)
    for folder in folders:
        F = list(files(folder)); bad = [rp for p, rp in F if M.get(rp, {}).get('sha256') != sha(p)]
        if bad: print(f'{folder}: {len(bad)} files not safely on the remote — not deleted'); continue
        # delete exactly the verified files, then empty directories; anything not on the remote (e.g. .git) stays
        for p, _ in F: p.unlink()
        real = (ROOT / folder).resolve()
        for d in sorted((x for x in real.rglob('*') if x.is_dir()), key=lambda x: -len(x.parts)):
            if not any(d.iterdir()): d.rmdir()
        left = any(real.iterdir())
        if not left:
            real.rmdir()
            if (ROOT / folder).is_symlink(): (ROOT / folder).unlink()
        print(f'{folder}: {len(F)} files deleted locally' + (' (files not on the remote were kept)' if left else '') + f'; `pull {folder}` restores them')


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('push', 'pull', 'status', 'free'): sys.exit(__doc__)
    rid = repo_id(); {'push': push, 'pull': pull, 'free': free}.get(sys.argv[1], lambda r, f: status(r))(rid, sys.argv[2:])
