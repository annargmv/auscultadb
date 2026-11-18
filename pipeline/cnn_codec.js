// Murmur network on compressed audio: every browsable CirCor recording scored from the site's 8 kHz 32 kbps MP3 copy and
// from the original WAV (raw/circor), with the page's own code → web/data/cnn_murmur.json['robustness']. Fits nothing.
//   node pipeline/cnn_codec.js
const { open } = require('../tests/helpers'); const fs = require('fs'); const path = require('path');
const ROOT = path.resolve(__dirname, '..'), WEB = path.join(ROOT, 'web'), TD = path.join(ROOT, 'raw', 'circor', 'circor-heart-sound-1.0.3', 'training_data');
(async () => {
  const I = JSON.parse(fs.readFileSync(path.join(WEB, 'data', 'index.json'))); const jobs = [];
  for (const p of I.patients) if (p.src === 'circor') for (const s of p.sites) { const [file, mp3] = s.file.split('#'); jobs.push({ file, mp3, name: mp3.replace(/\.mp3$/, '') }) }
  const { page: pg, url, close } = await open({ port: 8780 });
  await pg.route('**/__cc/**', r => { const f = path.join(TD, decodeURIComponent(r.request().url().split('/__cc/')[1]));
    if (!f.startsWith(TD) || !fs.existsSync(f)) return r.fulfill({ status: 404 }); r.fulfill({ status: 200, contentType: 'audio/wav', body: fs.readFileSync(f) }) });
  await pg.goto(url('#none')); await pg.waitForFunction(() => typeof cnnMurmur === 'function'); await pg.evaluate(() => ensureCnn());
  const base = url('').replace(/index\.html.*$/, ''); const out = []; let last = null, bundle = null;
  for (const j of jobs) {
    if (j.file !== last) { bundle = JSON.parse(fs.readFileSync(path.join(WEB, j.file))); last = j.file }
    const r = await pg.evaluate(async ({ b64, u }) => {
      const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0)); const { x } = await decodeToSR(new File([bin], 'x.mp3', { type: 'audio/mpeg' }), MAX_ANALYSE_S);
      const w = normalisePeak(parseWav(await (await fetch(u)).arrayBuffer())); const a = cnnMurmur(w), b = cnnMurmur(x);
      return a && b ? { wav: a.p, mp3: b.p } : null;
    }, { b64: bundle[j.mp3], u: `${base}__cc/${encodeURIComponent(j.name)}.wav` });
    if (r) out.push({ name: j.name, ...r });
  }
  await close();
  const M = JSON.parse(fs.readFileSync(path.join(WEB, 'data', 'cnn_murmur.json'))), t = M.threshold, d = out.map(o => Math.abs(o.mp3 - o.wav)).sort((a, b) => a - b);
  const q = p => d[Math.floor(p * (d.length - 1))];
  M.robustness = { generated: new Date().toISOString(), what: 'Browsable CirCor recordings: the site’s 8 kHz 32 kbps MP3 copy vs the original WAV', recordings: out.length,
    medianAbsShift: +(100 * q(0.5)).toFixed(1), p95AbsShift: +(100 * q(0.95)).toFixed(1), flagChanged: out.filter(o => (o.wav >= t) !== (o.mp3 >= t)).length };
  fs.writeFileSync(path.join(WEB, 'data', 'cnn_murmur.json'), JSON.stringify(M));
  console.log(M.robustness);
})();
