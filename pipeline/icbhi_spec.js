// Raw log-power spectrogram (FFT bins 1–64 = 31–2000 Hz, 10-ms hop) of every ICBHI recording with the page's cnnSpec(x, 64)
//   → build/icbhi_cnn/spec.f32 (float32, frames × 64) + index.jsonl (name, pid, nF, offset, expert cycles).
// Reads build/icbhi_wav (research licence: never leaves build/). train_lung_cnn.py trains the crackle/wheeze network.
//   node pipeline/icbhi_spec.js
const { open } = require('../tests/helpers'); const fs = require('fs'); const path = require('path');
const ROOT = path.resolve(__dirname, '..'), WAV = path.join(ROOT, 'build', 'icbhi_wav'), OUT = path.join(ROOT, 'build', 'icbhi_cnn'); fs.mkdirSync(OUT, { recursive: true });
const IDX = JSON.parse(fs.readFileSync(path.join(ROOT, 'meta', 'icbhi_index.json')));
(async () => {
  const { page: pg, url, close, errors } = await open({ port: 8779 });
  await pg.route('**/__ic/**', r => { const f = path.join(WAV, decodeURIComponent(r.request().url().split('/__ic/')[1]));
    if (!f.startsWith(WAV) || !fs.existsSync(f)) return r.fulfill({ status: 404 }); r.fulfill({ status: 200, contentType: 'audio/wav', body: fs.readFileSync(f) }) });
  await pg.goto(url('#none')); await pg.waitForFunction(() => typeof cnnSpec === 'function');
  const bin = fs.openSync(path.join(OUT, 'spec.f32'), 'w'), idx = fs.createWriteStream(path.join(OUT, 'index.jsonl'));
  let offset = 0, n = 0; const base = url('').replace(/index\.html.*$/, '');
  for (const p of IDX.patients) for (const rec of p.recs) {
    const r = await pg.evaluate(async u => {
      const x = normalisePeak(parseWav(await (await fetch(u)).arrayBuffer())); const S = cnnSpec(x, 64);
      const u8 = new Uint8Array(S.B.buffer); let str = ''; for (let i = 0; i < u8.length; i += 8192) str += String.fromCharCode.apply(null, u8.subarray(i, i + 8192));
      return { nF: S.nF, b64: btoa(str) };
    }, `${base}__ic/${encodeURIComponent(rec.name)}.wav`);
    fs.writeSync(bin, Buffer.from(r.b64, 'base64'));
    idx.write(JSON.stringify({ name: rec.name, pid: p.pid, dev: rec.dev, nF: r.nF, offset, cycles: rec.cycles }) + '\n');
    offset += r.nF; if (++n % 200 === 0) console.log(n, 'recordings');
  }
  fs.closeSync(bin); idx.end(); if (errors.length) console.log('page errors:', errors.slice(0, 3));
  await close(); console.log(`done: ${n} recordings, ${offset} frames -> ${OUT}`);
})();
