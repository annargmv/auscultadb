// External test on ADULT heart sounds: PhysioNet/CinC Challenge 2016 training sets a–f (ODC-By 1.0; Liu et al., Physiol Meas
// 2016, doi 10.1088/0967-3334/37/12/2181). Runs the page's own frozen code on every recording — nothing is trained or tuned:
//   cnnMurmur(x)          murmur spectrogram network (trained on CirCor children) -> probability
//   segmentHeartSounds(x) learned beat finder (trained on CirCor children)        -> S1 times -> hrvFromTimes (HR, RMSSD)
// Output: meta/features/cinc2016.jsonl. Scored by pipeline/cinc_validate.py, whose rules are fixed in its header.
//   node pipeline/cinc_run.js
const { open } = require('../tests/helpers'); const fs = require('fs'); const path = require('path');
const ROOT = path.resolve(__dirname, '..'), SRC = path.join(ROOT, 'raw', 'cinc2016', 'cinc');
const OUT = path.join(ROOT, 'meta', 'features', 'cinc2016.jsonl');
(async () => {
  const rows = [];
  for (const db of fs.readdirSync(SRC).filter(d => d.startsWith('training-')).sort()) {
    const sqi = Object.fromEntries(fs.readFileSync(path.join(SRC, db, 'REFERENCE-SQI.csv'), 'utf8').trim().split(/\r?\n/).map(l => l.split(',')).map(([n, y, q]) => [n, { y: +y, q: +q }]));
    for (const n of Object.keys(sqi)) rows.push({ db: db.slice(-1), name: n, y: sqi[n].y, sqi: sqi[n].q, file: path.join(db, n + '.wav') });
  }
  const done = new Set(fs.existsSync(OUT) ? fs.readFileSync(OUT, 'utf8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l).name) : []);
  const { page: pg, url, close, errors } = await open({ port: 8779 });
  await pg.route('**/__cinc/**', r => { const f = path.join(SRC, decodeURIComponent(r.request().url().split('/__cinc/')[1]));
    if (!f.startsWith(SRC) || !fs.existsSync(f)) return r.fulfill({ status: 404 }); r.fulfill({ status: 200, contentType: 'audio/wav', body: fs.readFileSync(f) }) });
  await pg.goto(url('#none')); await pg.waitForFunction(() => typeof cnnMurmur === 'function');
  await pg.evaluate(async () => { await ensureCnn(); await ensureSeg() });
  const base = url('').replace(/index\.html.*$/, ''), out = fs.createWriteStream(OUT, { flags: 'a' }); let k = 0;
  for (const r of rows) {
    if (done.has(r.name)) continue;
    const res = await pg.evaluate(async ([u, seg]) => {
      const x = normalisePeak(parseWav(await (await fetch(u)).arrayBuffer())), c = cnnMurmur(x), s = seg ? segmentHeartSounds(x) : null;   // beats only where an ECG exists (set a)
      const h = s ? hrvFromTimes(s.s1) : null;
      return { dur: +(x.length / SR).toFixed(2), cnnP: c ? +c.p.toFixed(5) : null, nWin: c ? c.windows.length : 0, s1: s ? s.s1 : null, hr: h && h.hr, rmssd: h && h.rmssd, cleanFrac: h && h.cleanFrac };
    }, [`${base}__cinc/${encodeURIComponent(r.file).replace(/%2F/g, '/')}`, r.db === 'a']);
    out.write(JSON.stringify({ db: r.db, name: r.name, y: r.y, sqi: r.sqi, ...res, s1: r.db === 'a' ? res.s1 : undefined }) + '\n');
    if (++k % 200 === 0) console.log(k, 'recordings', new Date().toISOString().slice(11, 19));
  }
  out.end(); if (errors.length) console.log('page errors:', errors.slice(0, 3)); await close(); console.log('done', k, '->', OUT);
})();
