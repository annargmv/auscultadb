// Held-out validation of the learned heart-sound segmenter → meta/features/seg_validate.jsonl
// For every recording of the held-out patients (models.json heart_murmur testPids — none of them was used by
// train_segmenter.py) that the CirCor experts annotated, with the page's own code:
//   * S1 onsets from the learned segmenter (segmentHeartSounds) and from the shipped beat finder (analyzeSignal → detectHeart)
//   * matched one-to-one to the experts' S1 onsets within ±50 ms, inside the annotated window
//   * heart rate and RMSSD from each, with the same artefact rule (hrvFromTimes)
// Also checks that the page reproduces the Python classifier (parity sample in segmenter.json). Tunes nothing.
//   node pipeline/seg_validate.js [limit]      then  python3 pipeline/hrv_discover.py
//   node pipeline/seg_validate.js --cohort     whole-recording RMSSD for the browsable patients → meta/features/seg_cohort.jsonl
const { open } = require('../tests/helpers'); const fs = require('fs'); const path = require('path');
const ROOT = path.resolve(__dirname, '..'), META = path.join(ROOT, 'meta'), TD = path.join(ROOT, 'raw', 'circor', 'circor-heart-sound-1.0.3', 'training_data');
const OUT = path.join(META, 'features', 'seg_validate.jsonl'), LIMIT = +process.argv[2] || 0, TOL = 0.05;
const IDX = JSON.parse(fs.readFileSync(path.join(META, 'circor_index.json')));
const MOD = JSON.parse(fs.readFileSync(path.join(ROOT, 'web', 'data', 'models.json')));
const TEST = new Set(MOD.models.find(m => m.id === 'heart_murmur').testPids);
function expert(name) {
  const f = path.join(TD, name + '.tsv'); if (!fs.existsSync(f)) return null;
  const rows = fs.readFileSync(f, 'utf8').trim().split('\n').map(l => l.split(/\s+/).map(Number)).filter(r => r.length === 3);
  const lab = rows.filter(r => r[2] > 0); if (!lab.length) return null;
  return { s1: rows.filter(r => r[2] === 1).map(r => r[0]), t0: lab[0][0], t1: lab[lab.length - 1][1] };
}
function match(ref, det) {   // greedy one-to-one, nearest first
  const pairs = []; for (let i = 0; i < ref.length; i++) for (let j = 0; j < det.length; j++) { const d = Math.abs(ref[i] - det[j]); if (d <= TOL) pairs.push([d, i, j]) }
  pairs.sort((a, b) => a[0] - b[0]); const ui = new Set(), uj = new Set(), err = [];
  for (const [d, i, j] of pairs) if (!ui.has(i) && !uj.has(j)) { ui.add(i); uj.add(j); err.push(+(1000 * (det[j] - ref[i])).toFixed(1)) }
  return { tp: ui.size, fn: ref.length - ui.size, fp: det.length - uj.size, err };
}
(async () => {
  const { page: pg, url, close, errors } = await open({ port: 8776 });
  await pg.route('**/__cc/**', r => { const f = path.join(TD, decodeURIComponent(r.request().url().split('/__cc/')[1]));
    if (!f.startsWith(TD) || !fs.existsSync(f)) return r.fulfill({ status: 404 }); r.fulfill({ status: 200, contentType: 'audio/wav', body: fs.readFileSync(f) }) });
  await pg.goto(url('#none')); await pg.waitForFunction(() => typeof segmentHeartSounds === 'function');
  const parity = await pg.evaluate(async () => {
    await ensureSeg(); const P = SEG.parity, nb = P.nb, B = Float32Array.from(P.bands), nF = B.length / nb;
    const rows = segFeatures({ nF, nb, B }); let mx = 0;
    P.frames.forEach((f, i) => modelProba(SEG, rows[f]).forEach((p, c) => { mx = Math.max(mx, Math.abs(p - P.p[i][c])) }));
    return mx;
  });
  console.log('parity: max |p_js − p_python| =', parity.toExponential(2)); if (!(parity < 1e-6)) throw new Error('segmenter parity failed');
  if (process.argv.includes('--cohort')) {   // whole-recording HRV for the browsable patients' recordings (patient pages)
    const C = JSON.parse(fs.readFileSync(path.join(META, 'circor_cohort.json'))), CP = Array.isArray(C) ? C : C.patients;
    const o = fs.createWriteStream(path.join(META, 'features', 'seg_cohort.jsonl')); let k = 0;
    for (const p of CP) for (const [loc, s] of Object.entries(p.sites)) {
      const h = await pg.evaluate(async u => { const x = normalisePeak(parseWav(await (await fetch(u)).arrayBuffer())); const G = segmentHeartSounds(x); return hrvFromTimes(G ? G.s1 : []) },
        `${url('').replace(/index\.html.*$/, '')}__cc/${encodeURIComponent(s.name)}.wav`);
      o.write(JSON.stringify({ name: s.name, pid: p.pid, loc, hrvSeg: h }) + '\n'); k++;
    }
    o.end(); await close(); console.log(`done: ${k} cohort recordings -> meta/features/seg_cohort.jsonl`); return;
  }
  const out = fs.createWriteStream(OUT); let n = 0; const T = Date.now();
  const pats = IDX.patients.filter(p => TEST.has(p.pid)); const use = LIMIT ? pats.slice(0, LIMIT) : pats;
  for (const p of use) for (const [loc, s] of Object.entries(p.sites)) {
    const ex = expert(s.name); if (!ex) continue;
    let r;
    try {
      r = await pg.evaluate(async ({ u, ex }) => {
        const x = normalisePeak(parseWav(await (await fetch(u)).arrayBuffer())); DUR = x.length / SR;
        const A = analyzeSignal(x), G = segmentHeartSounds(x);
        const inW = t => t >= ex.t0 && t <= ex.t1;
        const det = (A.heart && A.heart.beats || []).map(b => b.s1).filter(inW).sort((a, b) => a - b), seg = (G ? G.s1 : []).filter(inW);
        return { det, seg, hrvDet: hrvFromTimes(det), hrvSeg: hrvFromTimes(seg), hrvExp: hrvFromTimes(ex.s1) };
      }, { u: `${url('').replace(/index\.html.*$/, '')}__cc/${encodeURIComponent(s.name)}.wav`, ex });
      const ref = ex.s1.filter(t => t >= ex.t0 && t <= ex.t1);
      r = { annSec: +(ex.t1 - ex.t0).toFixed(2), nExp: ref.length, mDet: match(ref, r.det), mSeg: match(ref, r.seg), hrvDet: r.hrvDet, hrvSeg: r.hrvSeg, hrvExp: r.hrvExp };
    } catch (e) { r = { err: String(e.message || e) } }
    out.write(JSON.stringify({ name: s.name, pid: p.pid, loc, ageBand: p.ageBand, murmur: p.murmur, ...r }) + '\n');
    if (++n % 100 === 0) console.log(`${n} recordings, ${((Date.now() - T) / 60000).toFixed(1)} min`);
  }
  out.end(); if (errors.length) console.log('page errors:', errors.slice(0, 5).join(' | '));
  await close(); console.log(`done: ${n} held-out recordings from ${use.length} patients -> ${OUT}`);
})();
