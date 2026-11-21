// EPHNOGRAM: the page's frozen learned beat finder on every 30-s PCG window (build/eph/*.wav from pipeline/eph_validate.py prep)
// -> meta/features/eph.jsonl (S1 times). Scored by pipeline/eph_validate.py against the simultaneous ECG.
//   node pipeline/eph_run.js
const { open } = require('../tests/helpers'); const fs = require('fs'); const path = require('path');
const ROOT = path.resolve(__dirname, '..'), SRC = path.join(ROOT, 'build', 'eph'), OUT = path.join(ROOT, 'meta', 'features', 'eph.jsonl');
(async () => {
  const files = fs.readdirSync(SRC).filter(f => f.endsWith('.wav')).sort();
  const { page: pg, url, close, errors } = await open({ port: 8782 });
  await pg.route('**/__eph/**', r => { const f = path.join(SRC, decodeURIComponent(r.request().url().split('/__eph/')[1]));
    if (!f.startsWith(SRC) || !fs.existsSync(f)) return r.fulfill({ status: 404 }); r.fulfill({ status: 200, contentType: 'audio/wav', body: fs.readFileSync(f) }) });
  await pg.goto(url('#none')); await pg.waitForFunction(() => typeof segmentHeartSounds === 'function'); await pg.evaluate(async () => { await ensureSeg() });
  const base = url('').replace(/index\.html.*$/, ''), out = fs.createWriteStream(OUT); let k = 0;
  for (const f of files) {
    const r = await pg.evaluate(async u => { const x = normalisePeak(parseWav(await (await fetch(u)).arrayBuffer())), s = segmentHeartSounds(x); return { dur: +(x.length / SR).toFixed(2), s1: s ? s.s1 : null } }, `${base}__eph/${encodeURIComponent(f)}`);
    out.write(JSON.stringify({ name: f.replace('.wav', ''), ...r }) + '\n'); if (++k % 50 === 0) console.log(k, 'windows');
  }
  out.end(); if (errors.length) console.log('page errors:', errors.slice(0, 3)); await close(); console.log('done', k, '->', OUT);
})();
