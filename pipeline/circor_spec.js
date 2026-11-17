// Raw log-power spectrogram (FFT bins 1–32 = 31–1000 Hz, 10-ms hop) of every CirCor recording, computed by the page's own
// cnnSpec() → build/cnn/spec.f32 (float32, frames × 32, all recordings concatenated) + build/cnn/index.jsonl.
// train_cnn.py normalises with the same formula as the page's cnnNorm and trains the murmur network.
//   node pipeline/circor_spec.js
const { open } = require('../tests/helpers'); const fs = require('fs'); const path = require('path');
const ROOT = path.resolve(__dirname, '..'), TD = path.join(ROOT, 'raw', 'circor', 'circor-heart-sound-1.0.3', 'training_data');
const OUT = path.join(ROOT, 'build', 'cnn'); fs.mkdirSync(OUT, { recursive: true });
const IDX = JSON.parse(fs.readFileSync(path.join(ROOT, 'meta', 'circor_index.json')));
(async () => {
  const { page: pg, url, close, errors } = await open({ port: 8778 });
  await pg.route('**/__cc/**', r => { const f = path.join(TD, decodeURIComponent(r.request().url().split('/__cc/')[1]));
    if (!f.startsWith(TD) || !fs.existsSync(f)) return r.fulfill({ status: 404 }); r.fulfill({ status: 200, contentType: 'audio/wav', body: fs.readFileSync(f) }) });
  await pg.goto(url('#none')); await pg.waitForFunction(() => typeof cnnSpec === 'function');
  const bin = fs.openSync(path.join(OUT, 'spec.f32'), 'w'), idx = fs.createWriteStream(path.join(OUT, 'index.jsonl'));
  let offset = 0, n = 0; const base = url('').replace(/index\.html.*$/, '');
  for (const p of IDX.patients) for (const [loc, s] of Object.entries(p.sites)) {
    const r = await pg.evaluate(async u => {
      const x = normalisePeak(parseWav(await (await fetch(u)).arrayBuffer())); const S = cnnSpec(x);
      const u8 = new Uint8Array(S.B.buffer); let str = ''; for (let i = 0; i < u8.length; i += 8192) str += String.fromCharCode.apply(null, u8.subarray(i, i + 8192));
      return { nF: S.nF, b64: btoa(str) };
    }, `${base}__cc/${encodeURIComponent(s.name)}.wav`);
    fs.writeSync(bin, Buffer.from(r.b64, 'base64'));
    idx.write(JSON.stringify({ name: s.name, pid: p.pid, loc, label: s.label, nF: r.nF, offset }) + '\n');
    offset += r.nF; if (++n % 500 === 0) console.log(n, 'recordings');
  }
  fs.closeSync(bin); idx.end(); if (errors.length) console.log('page errors:', errors.slice(0, 3));
  await close(); console.log(`done: ${n} recordings, ${offset} frames -> ${OUT}`);
})();
