// Raw per-frame band energies for every CirCor recording, computed by the page's own segBands()
//   → build/seg/frames.f32 (float32, frames × 5 bands, all recordings concatenated) + build/seg/index.jsonl
// train_segmenter.py builds features and labels from these and the experts' .tsv files.
//   node pipeline/circor_frames.js
const { open } = require('../tests/helpers'); const fs = require('fs'); const path = require('path');
const ROOT = path.resolve(__dirname, '..'), TD = path.join(ROOT, 'raw', 'circor', 'circor-heart-sound-1.0.3', 'training_data');
const OUT = path.join(ROOT, 'build', 'seg'); fs.mkdirSync(OUT, { recursive: true });
const IDX = JSON.parse(fs.readFileSync(path.join(ROOT, 'meta', 'circor_index.json')));
(async () => {
  const { page: pg, url, close, errors } = await open({ port: 8775 });
  await pg.route('**/__cc/**', r => { const f = path.join(TD, decodeURIComponent(r.request().url().split('/__cc/')[1]));
    if (!f.startsWith(TD) || !fs.existsSync(f)) return r.fulfill({ status: 404 }); r.fulfill({ status: 200, contentType: 'audio/wav', body: fs.readFileSync(f) }) });
  await pg.goto(url('#none')); await pg.waitForFunction(() => typeof segBands === 'function');
  const bin = fs.openSync(path.join(OUT, 'frames.f32'), 'w'), idx = fs.createWriteStream(path.join(OUT, 'index.jsonl'));
  let offset = 0, n = 0; const base = url('').replace(/index\.html.*$/, '');
  for (const p of IDX.patients) for (const [loc, s] of Object.entries(p.sites)) {
    const r = await pg.evaluate(async u => {
      const x = normalisePeak(parseWav(await (await fetch(u)).arrayBuffer())); const S = segBands(x);
      const u8 = new Uint8Array(S.B.buffer); let str = ''; for (let i = 0; i < u8.length; i += 8192) str += String.fromCharCode.apply(null, u8.subarray(i, i + 8192));
      return { nF: S.nF, nb: S.nb, t0: S.t0, hop: S.hop, b64: btoa(str) };
    }, `${base}__cc/${encodeURIComponent(s.name)}.wav`);
    const buf = Buffer.from(r.b64, 'base64'); fs.writeSync(bin, buf);
    idx.write(JSON.stringify({ name: s.name, pid: p.pid, loc, nF: r.nF, nb: r.nb, t0: r.t0, hop: r.hop, offset }) + '\n');
    offset += r.nF; if (++n % 500 === 0) console.log(n, 'recordings');
  }
  fs.closeSync(bin); idx.end(); if (errors.length) console.log('page errors:', errors.slice(0, 3));
  await close(); console.log(`done: ${n} recordings, ${offset} frames -> ${OUT}`);
})();
