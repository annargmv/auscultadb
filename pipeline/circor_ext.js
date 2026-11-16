// CirCor, every recording, with the page's own code → meta/features/circor_ext.jsonl
//   * heartFeatures + heartFeaturesExt (richer systole-vs-diastole measurements for the murmur model)
//   * HRV from the beat finder vs HRV from the CirCor experts' S1 annotation (.tsv), on the SAME annotated window —
//     the validation that decides whether the HRV longevity marker can be shown at all
// Reads raw/circor (PhysioNet 1.0.3, ODC-By). Tunes nothing. Resumable.
//   node pipeline/circor_ext.js [limit]
const { open } = require('../tests/helpers'); const fs = require('fs'); const path = require('path');
const ROOT = path.resolve(__dirname, '..'), META = path.join(ROOT, 'meta'), TD = path.join(ROOT, 'raw', 'circor', 'circor-heart-sound-1.0.3', 'training_data');
const OUT = path.join(META, 'features', 'circor_ext.jsonl'), LIMIT = +process.argv[2] || 0;
const IDX = JSON.parse(fs.readFileSync(path.join(META, 'circor_index.json')));
// expert S1 onsets (label 1) and the annotated window (first to last labelled segment)
function expert(name) {
  const f = path.join(TD, name + '.tsv'); if (!fs.existsSync(f)) return null;
  const rows = fs.readFileSync(f, 'utf8').trim().split('\n').map(l => l.split(/\s+/).map(Number)).filter(r => r.length === 3);
  const lab = rows.filter(r => r[2] > 0); if (!lab.length) return null;
  return { s1: rows.filter(r => r[2] === 1).map(r => r[0]), t0: lab[0][0], t1: lab[lab.length - 1][1] };
}
(async () => {
  const done = new Set(fs.existsSync(OUT) ? fs.readFileSync(OUT, 'utf8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l).name) : []);
  const { page: pg, url, close, errors } = await open({ port: 8774 });
  await pg.route('**/__cc/**', r => { const f = path.join(TD, decodeURIComponent(r.request().url().split('/__cc/')[1]));
    if (!f.startsWith(TD) || !fs.existsSync(f)) return r.fulfill({ status: 404 }); r.fulfill({ status: 200, contentType: 'audio/wav', body: fs.readFileSync(f) }) });
  await pg.goto(url('#none')); await pg.waitForFunction(() => typeof heartFeaturesExt === 'function' && typeof hrvFromTimes === 'function');
  const out = fs.createWriteStream(OUT, { flags: 'a' }); let n = 0; const t0 = Date.now();
  const pats = LIMIT ? IDX.patients.slice(0, LIMIT) : IDX.patients;
  for (const p of pats) for (const [loc, s] of Object.entries(p.sites)) {
    if (done.has(s.name)) continue;
    const ex = expert(s.name);
    let r;
    try {
      r = await pg.evaluate(async ({ u, ex }) => {
        const x = normalisePeak(parseWav(await (await fetch(u)).arrayBuffer())); DUR = x.length / SR;
        const A = analyzeSignal(x);
        const o = { ...heartFeatures(A), ext: heartFeaturesExt(A), hrvAll: hrvFromBeats(A) };
        if (ex) { o.hrvDet = hrvFromBeats(A, ex.t0, ex.t1); o.hrvExp = hrvFromTimes(ex.s1); o.annSec = +(ex.t1 - ex.t0).toFixed(2) }
        return o;
      }, { u: `${url('').replace(/index\.html.*$/, '')}__cc/${encodeURIComponent(s.name)}.wav`, ex });
    } catch (e) { r = { err: String(e.message || e) } }
    out.write(JSON.stringify({ name: s.name, pid: p.pid, loc, siteTruth: s.label, murmur: p.murmur, ageBand: p.ageBand, g: p.g, campaign: p.campaign, outcome: p.outcome, ...r }) + '\n');
    if (++n % 100 === 0) console.log(`${n} recordings, ${((Date.now() - t0) / 1000 / 60).toFixed(1)} min`);
  }
  out.end(); if (errors.length) console.log('page errors:', errors.slice(0, 5).join(' | '));
  await close(); console.log(`done: ${n} new recordings -> ${OUT}`);
})();
