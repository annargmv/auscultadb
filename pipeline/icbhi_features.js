// ICBHI 2017, every recording, with the page's own code → meta/features/icbhi.jsonl
//   * recordingFeatures (the same per-recording features the lung models use)
//   * cycleFeatures for every EXPERT-annotated cycle (with its crackle/wheeze label) — training and benchmark scoring
//   * cycleFeatures for the cycles the PAGE itself would score on a new upload (scoringCycles) — deployment check
//   * the shipped rule-based detectors' verdict per expert cycle (any crackle candidate / any tonal segment)
// Reads build/icbhi_wav (pipeline/icbhi_prep.py). Audio never leaves build/. Tunes nothing. Resumable.
//   node pipeline/icbhi_features.js [limit]
const { open } = require('../tests/helpers'); const fs = require('fs'); const path = require('path');
const ROOT = path.resolve(__dirname, '..'), WAV = path.join(ROOT, 'build', 'icbhi_wav');
const OUT = path.join(ROOT, 'meta', 'features', 'icbhi.jsonl'), LIMIT = +process.argv[2] || 0;
const IDX = JSON.parse(fs.readFileSync(path.join(ROOT, 'meta', 'icbhi_index.json')));
(async () => {
  const done = new Set(fs.existsSync(OUT) ? fs.readFileSync(OUT, 'utf8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l).name) : []);
  const { page: pg, url, close, errors } = await open({ port: 8777 });
  await pg.route('**/__ic/**', r => { const f = path.join(WAV, decodeURIComponent(r.request().url().split('/__ic/')[1]));
    if (!f.startsWith(WAV) || !fs.existsSync(f)) return r.fulfill({ status: 404 }); r.fulfill({ status: 200, contentType: 'audio/wav', body: fs.readFileSync(f) }) });
  await pg.goto(url('#none')); await pg.waitForFunction(() => typeof cycleFeatures === 'function' && window.Meyda);
  const out = fs.createWriteStream(OUT, { flags: 'a' }); let n = 0; const T = Date.now(), base = url('').replace(/index\.html.*$/, '');
  const recs = IDX.patients.flatMap(p => p.recs.map(r => ({ ...r, pid: p.pid })));
  for (const r of (LIMIT ? recs.slice(0, LIMIT) : recs)) {
    if (done.has(r.name)) continue;
    let o;
    try {
      o = await pg.evaluate(async ({ u, cyc }) => {
        const x = normalisePeak(parseWav(await (await fetch(u)).arrayBuffer())); DUR = x.length / SR;
        const A = analyzeSignal(x), rf = recordingFeatures(A);
        const rule = (t0, t1) => ({ crk: A.transients.some(e => e.t0 >= t0 && e.t0 < t1 && e.hl > (e.fine ? HL_FINE : HL_COARSE)) ? 1 : 0,
                                    whz: A.tonal.some(g => g.t1 > t0 && g.t0 < t1) ? 1 : 0 });
        const exp = cyc.map(([t0, t1, c, w]) => ({ t0, t1, c, w, f: (() => { const a = cycleFeatures(A, t0, Math.min(t1, A.dur)); return a && { ...a, ...(cycleMel(A, t0, Math.min(t1, A.dur)) || {}) } })(), rule: rule(t0, t1) }));
        const sc = scoringCycles(A).map(([t0, t1, src]) => ({ t0: +t0.toFixed(3), t1: +t1.toFixed(3), src, f: (() => { const a = cycleFeatures(A, t0, t1); return a && { ...a, ...(cycleMel(A, t0, t1) || {}) } })() }));
        return { rf, exp, sc, nDetected: A.cycles.length };
      }, { u: `${base}__ic/${encodeURIComponent(r.name)}.wav`, cyc: r.cycles });
    } catch (e) { o = { err: String(e.message || e) } }
    out.write(JSON.stringify({ name: r.name, pid: r.pid, loc: r.loc, mode: r.mode, dev: r.dev, ...o }) + '\n');
    if (++n % 100 === 0) console.log(`${n} recordings, ${((Date.now() - T) / 60000).toFixed(1)} min`);
  }
  out.end(); if (errors.length) console.log('page errors:', errors.slice(0, 5).join(' | '));
  await close(); console.log(`done: ${n} recordings -> ${OUT}`);
})();
