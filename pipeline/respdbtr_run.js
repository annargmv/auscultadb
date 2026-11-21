// External test on ADULT lung sounds: RespiratoryDatabase@TR (Altan & Kutlu, Mendeley 10.17632/p9z4h98s6j.2, CC BY 4.0) —
// 42 COPD patients, GOLD stage 0–4, 12 chest channels each, 4 kHz (Littmann 3200, the same stethoscope as KAUH).
// Runs the page's own frozen code on every channel; nothing is trained or tuned:
//   analyzeSignal -> recordingFeatures -> modelProba(lung_sound)   breath-sound type probabilities (normal / continuous / crackles / both)
//   icbhiScore                                                     per-cycle wheeze and crackle probabilities (ICBHI cycle model)
// Output: meta/features/respdbtr.jsonl. Scored by pipeline/respdbtr_validate.py (rules in its header).
//   node pipeline/respdbtr_run.js
const { open } = require('../tests/helpers'); const fs = require('fs'); const path = require('path');
const ROOT = path.resolve(__dirname, '..'), SRC = path.join(ROOT, 'raw', 'respdbtr', 'RespiratoryDatabase@TR');
const OUT = path.join(ROOT, 'meta', 'features', 'respdbtr.jsonl');
(async () => {
  const files = fs.readdirSync(SRC).filter(f => f.endsWith('.wav')).sort();
  const { page: pg, url, close, errors } = await open({ port: 8780 });
  await pg.route('**/__tr/**', r => { const f = path.join(SRC, decodeURIComponent(r.request().url().split('/__tr/')[1]));
    if (!f.startsWith(SRC) || !fs.existsSync(f)) return r.fulfill({ status: 404 }); r.fulfill({ status: 200, contentType: 'audio/wav', body: fs.readFileSync(f) }) });
  await pg.goto(url('#none')); await pg.waitForFunction(() => typeof analyzeSignal === 'function');
  await pg.evaluate(async () => { MODELS = await cachedFetch('data/models.json', 'json'); await ensureIcbhi() });
  const base = url('').replace(/index\.html.*$/, ''), out = fs.createWriteStream(OUT); let k = 0;
  for (const f of files) {
    const r = await pg.evaluate(async u => {
      const x = normalisePeak(parseWav(await (await fetch(u)).arrayBuffer())), A = analyzeSignal(x), lf = recordingFeatures(A);
      const m = MODELS.models.find(q => q.id === 'lung_sound'), p = modelProba(m, m.features.map(k => lf[k] == null ? null : +lf[k]));
      const R = icbhiScore(A, lf, 60), c = R.cyc;
      return { dur: +(x.length / SR).toFixed(2), p: p.map(v => +v.toFixed(5)), flags: p.map((q, i) => q >= m.thresholds[i]), rr: lf.rr, tonFrac: lf.tonFrac, crkRate: lf.crkRate,
        nCyc: c.length, wheezeShare: c.length ? +(c.filter(q => q.pw >= 0.5).length / c.length).toFixed(3) : null, crackleShare: c.length ? +(c.filter(q => q.pc >= 0.5).length / c.length).toFixed(3) : null };
    }, `${base}__tr/${encodeURIComponent(f)}`);
    const [pid, ch] = f.replace('.wav', '').split('_');
    out.write(JSON.stringify({ name: f.replace('.wav', ''), pid, ch, ...r }) + '\n'); if (++k % 100 === 0) console.log(k, 'channels');
  }
  out.end(); if (errors.length) console.log('page errors:', errors.slice(0, 3)); await close(); console.log('done', k, '->', OUT);
})();
