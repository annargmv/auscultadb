// Validate the cardiac findings (murmur / diastolic murmur / S3) against CirCor DigiScope expert labels.
// Runs the page's OWN analysis code (web/index.html) in headless Chromium on the original 4 kHz WAVs,
// exactly as the site does: analyzeSignal on the recording and on the patient's murmur-free control,
// then compareRecordings -> findings. Nothing is tuned here; this only measures.
//
//   node pipeline/circor_validate.js              all 941 patients (~30-60 min)
//   node pipeline/circor_validate.js 40           first 40 patients (smoke run)
//
// Writes meta/features/circor.jsonl  (one row per recording)
//        meta/circor_validation.json (the summary that Discoveries reads)
//
// Ground truth
//   training_data.csv  Murmur Present/Absent/Unknown, "Murmur locations" -> per-valve truth,
//                      systolic timing / shape / grading / pitch / quality, Outcome
//   <rec>.tsv          segmentation: 1=S1 2=systole 3=S2 4=diastole, 0=unannotated
//                      -> S1/S2 timing truth, scored only inside annotated spans
// CirCor does not label S3, so the S3 rate is reported but cannot be scored.
const { chromium } = require('playwright'); const fs = require('fs'); const path = require('path'); const http = require('http');
const ROOT = path.resolve(__dirname, '..'); const META = path.join(ROOT, 'meta');
const PORT = +(process.env.PORT || 8767);
const LIMIT = process.argv[2] ? +process.argv[2] : null;
const MIME = { '.html': 'text/html', '.js': 'application/javascript', '.json': 'application/json', '.wav': 'audio/wav' };
function serve() { return http.createServer((q, r) => { const f = path.join(ROOT, decodeURIComponent(q.url.split('?')[0].split('#')[0]));
  if (!f.startsWith(ROOT) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); r.end(); return }
  r.writeHead(200, { 'Content-Type': MIME[path.extname(f)] || 'application/octet-stream' }); fs.createReadStream(f).pipe(r) }).listen(PORT) }

const IDX = JSON.parse(fs.readFileSync(path.join(META, 'circor_index.json')));
const TD = path.join(ROOT, IDX.dir);
const wavUrl = name => `http://localhost:${PORT}/${path.relative(ROOT, TD)}/${encodeURIComponent(name)}.wav`;
const byKey = Object.fromEntries(IDX.patients.map(p => [p.key, p]));

// --- TSV ground truth, scored only where the experts annotated
function truth(name) {
  const f = path.join(TD, name + '.tsv');
  if (!fs.existsSync(f)) return null;
  const rows = fs.readFileSync(f, 'utf8').trim().split('\n').map(l => l.split('\t').map(Number));
  const s1 = [], s2 = [], spans = [];
  for (const [t0, t1, k] of rows) {
    if (k === 1) s1.push(t0); else if (k === 3) s2.push(t0);
    if (k > 0) { const last = spans[spans.length - 1]; if (last && Math.abs(last[1] - t0) < 1e-6) last[1] = t1; else spans.push([t0, t1]) }
  }
  return { s1, s2, spans, inside: t => spans.some(([a, b]) => t >= a - 0.05 && t <= b + 0.05) };
}
const TOL = 0.1;   // 100 ms — S1/S2 are ~50-120 ms long, so this is one sound's width
function match(det, gt, inside) {
  const d = det.filter(inside);                       // only judge detections the experts labelled
  const used = new Set(); let hit = 0;
  for (const t of gt) { let bi = -1, bd = TOL;
    d.forEach((x, i) => { const e = Math.abs(x - t); if (e < bd && !used.has(i)) { bd = e; bi = i } });
    if (bi >= 0) { used.add(bi); hit++ } }
  const errs = [];
  for (const t of gt) { let bd = Infinity; for (const x of d) bd = Math.min(bd, Math.abs(x - t)); if (bd <= TOL) errs.push(bd * 1000) }
  return { nDet: d.length, nGt: gt.length, hit, fp: d.length - used.size, err: errs };
}
const med = a => { if (!a.length) return null; const s = a.slice().sort((x, y) => x - y); return s[s.length >> 1] };

(async () => {
  const srv = serve(); fs.mkdirSync(path.join(META, 'features'), { recursive: true });
  const b = await chromium.launch(); const ctx = await b.newContext();
  for (const h of ['https://unpkg.com/**', 'https://cdn.jsdelivr.net/**', 'https://fonts.googleapis.com/**']) await ctx.route(h, r => r.abort());
  const pg = await ctx.newPage(); pg.on('pageerror', e => console.log('PAGEERROR', e.message));
  await pg.goto(`http://localhost:${PORT}/web/index.html#none`); await pg.waitForTimeout(400);
  await pg.evaluate(() => {
    window.loadWav = async u => normalisePeak(parseWav(await (await fetch(u)).arrayBuffer()));
    window.__A = {};
    window.grab = async u => { if (!window.__A[u]) window.__A[u] = analyzeSignal(await loadWav(u)); const H = window.__A[u].heart;
      return { dur: window.__A[u].dur, hr: H.hr, quietBeats: H.quietBeats, sysDia: H.sysDia, consistency: H.consistency,
               s1: H.beats.map(z => z.s1), s2: H.beats.map(z => z.s2), s3: H.beats.filter(z => z.s3).map(z => z.s3) } };
    window.cmp = (u1, u2, site, age) => compareRecordings(window.__A[u1], window.__A[u2], { site, age })
      .filter(f => ['murmur', 'dmurmur', 's3'].includes(f.type)).map(f => ({ type: f.type, conf: f.conf, count: f.count, pos: f.pos }));
  });

  // process patients grouped by baseline, so each control is analysed once
  let pats = IDX.patients.slice().sort((x, y) => (x.ref || '').localeCompare(y.ref || '') || x.pid.localeCompare(y.pid));
  if (LIMIT) pats = pats.slice(0, LIMIT);
  const out = fs.createWriteStream(path.join(META, 'features', 'circor.jsonl'));
  const rows = []; let done = 0, t0 = Date.now(), lastRef = null;

  for (const p of pats) {
    const ref = byKey[p.ref]; if (!ref) continue;
    if (p.ref !== lastRef) { await pg.evaluate(() => { window.__A = {} }); lastRef = p.ref }   // drop cache when the control changes
    const refSites = Object.keys(ref.sites);
    for (const [v, r] of Object.entries(p.sites)) {
      const bName = ref.sites[v] ? ref.sites[v].name : ref.sites[refSites[0]].name;   // same valve if the control has it
      let A, B, finds;
      try {
        A = await pg.evaluate(u => grab(u), wavUrl(r.name));
        B = await pg.evaluate(u => grab(u), wavUrl(bName));
        finds = await pg.evaluate(a => cmp(a[0], a[1], a[2], a[3]), [wavUrl(r.name), wavUrl(bName), r.site, p.age]);
      } catch (e) { out.write(JSON.stringify({ key: p.key, loc: v, name: r.name, err: String(e.message || e) }) + '\n'); continue }
      const gt = truth(r.name);
      const types = finds.map(f => f.type);
      const row = { key: p.key, pid: p.pid, loc: v, name: r.name, ageBand: p.ageBand, age: p.age, g: p.g,
        murmur: p.murmur, siteTruth: r.label, audible: !!r.audible, outcome: p.outcome, campaign: p.campaign,
        grading: p.sys.grading, timing: p.sys.timing, shape: p.sys.shape, pitch: p.sys.pitch, quality: p.sys.quality,
        diaTiming: p.dia.timing, base: bName, baseKey: p.ref,
        dur: +A.dur.toFixed(2), hr: A.hr && +A.hr.toFixed(1), quietBeats: A.quietBeats,
        sysDia: A.sysDia == null ? null : +A.sysDia.toFixed(3), baseSysDia: B.sysDia == null ? null : +B.sysDia.toFixed(3),
        consistency: A.consistency == null ? null : +A.consistency.toFixed(3), nBeats: A.s1.length, nS3: A.s3.length,
        det: { murmur: types.includes('murmur'), dmurmur: types.includes('dmurmur'), s3: types.includes('s3') },
        conf: Object.fromEntries(finds.map(f => [f.type, +f.conf.toFixed(3)])) };
      if (gt) {
        const m1 = match(A.s1, gt.s1.filter(gt.inside), gt.inside), m2 = match(A.s2, gt.s2.filter(gt.inside), gt.inside);
        row.gt = { nS1: m1.nGt, nS2: m2.nGt, s1hit: m1.hit, s2hit: m2.hit, s1fp: m1.fp, s2fp: m2.fp,
                   s1err: med(m1.err) && +med(m1.err).toFixed(1), s2err: med(m2.err) && +med(m2.err).toFixed(1),
                   annotated: +gt.spans.reduce((a, [x, y]) => a + y - x, 0).toFixed(2) };
      }
      out.write(JSON.stringify(row) + '\n'); rows.push(row);
    }
    if (++done % 25 === 0) console.log(done, '/', pats.length, ((Date.now() - t0) / 1000).toFixed(0) + 's', `${rows.length} recordings`);
  }
  out.end(); await new Promise(r => out.on('finish', r));

  // ---------- scoring ----------
  const ok = rows.filter(r => !r.err);
  const rate = (f, g) => { const d = ok.filter(g); return [d.filter(f).length, d.length] };
  const pct = ([a, n]) => n ? (100 * a / n).toFixed(1) + '%' : '—';
  // per patient: the site flags a murmur anywhere
  const perPat = {}; for (const r of ok) { const z = perPat[r.key] = perPat[r.key] || { murmur: r.murmur, det: false, outcome: r.outcome, grading: r.grading, timing: r.timing, ageBand: r.ageBand, n: 0 }; z.det = z.det || r.det.murmur; z.n++ }
  const P = Object.values(perPat);
  const pRate = (f, g) => { const d = P.filter(g); return [d.filter(f).length, d.length] };
  const s1 = ok.filter(r => r.gt && r.gt.nS1 > 0);
  const sum = (a, f) => a.reduce((s, x) => s + f(x), 0);
  const V = {
    dataset: IDX.dataset, licence: IDX.licence, generated: new Date().toISOString(),
    nPatients: P.length, nRecordings: ok.length, nErrors: rows.length - ok.length,
    murmurPatient: { sens: pRate(z => z.det, z => z.murmur === 'Present'), spec: pRate(z => !z.det, z => z.murmur === 'Absent'),
                     unknownFlagged: pRate(z => z.det, z => z.murmur === 'Unknown') },
    murmurSite:    { sens: rate(r => r.det.murmur, r => r.siteTruth === 'Murmur'), spec: rate(r => !r.det.murmur, r => r.siteTruth === 'Normal'),
                     audibleSens: rate(r => r.det.murmur, r => r.audible) },
    byGrade:  Object.fromEntries(['I/VI', 'II/VI', 'III/VI'].map(g => [g, pRate(z => z.det, z => z.grading === g)])),
    byTiming: Object.fromEntries(['Holosystolic', 'Early-systolic', 'Mid-systolic', 'Late-systolic'].map(t => [t, pRate(z => z.det, z => z.timing === t)])),
    byAge:    Object.fromEntries(['Neonate', 'Infant', 'Child', 'Adolescent'].map(a => [a, pRate(z => z.det, z => z.ageBand === a && z.murmur === 'Present')])),
    diastolic: { truthN: ok.filter(r => r.diaTiming).length, detected: rate(r => r.det.dmurmur, r => !!r.diaTiming),
                 falsePos: rate(r => r.det.dmurmur, r => !r.diaTiming) },
    s3: { note: 'CirCor does not label S3; rate only, not validated', rate: rate(r => r.det.s3, () => true) },
    segmentation: { nRecordings: s1.length,
      s1Recall: [sum(s1, r => r.gt.s1hit), sum(s1, r => r.gt.nS1)], s2Recall: [sum(s1, r => r.gt.s2hit), sum(s1, r => r.gt.nS2)],
      s1FalsePos: [sum(s1, r => r.gt.s1fp), sum(s1, r => r.gt.nS1)],
      s1ErrMedianMs: med(s1.map(r => r.gt.s1err).filter(x => x != null)), s2ErrMedianMs: med(s1.map(r => r.gt.s2err).filter(x => x != null)),
      tolMs: TOL * 1000 },
    gates: { hrResolved: rate(r => r.hr != null, () => true), beatsGE3: rate(r => r.nBeats >= 3, () => true),
             quietGE3: rate(r => r.quietBeats >= 3, () => true), sysDiaResolved: rate(r => r.sysDia != null, () => true) },
    sysDiaByTruth: { murmur: med(ok.filter(r => r.siteTruth === 'Murmur').map(r => r.sysDia).filter(x => x != null)),
                     normal: med(ok.filter(r => r.siteTruth === 'Normal').map(r => r.sysDia).filter(x => x != null)) },
    confounds: ['Browsable cohort over-samples murmurs (70 of 120) vs 19% real prevalence; these figures are from the full set, not the cohort.',
                'Single auscultation campaign per patient (Campaign field); device and setting not randomised.',
                'Baseline is another patient\'s control recording, not the same patient at rest.',
                'CirCor is paediatric; the adult heart-rate gate is untested here.'],
    partial: LIMIT ? `smoke run: first ${LIMIT} patients only` : null,
  };
  fs.writeFileSync(path.join(META, 'circor_validation.json'), JSON.stringify(V, null, 1));
  console.log('\n===== CirCor validation' + (LIMIT ? ` (smoke: ${LIMIT} patients)` : '') + ' =====');
  console.log(`patients ${V.nPatients}  recordings ${V.nRecordings}  errors ${V.nErrors}`);
  console.log(`murmur, per patient : sens ${pct(V.murmurPatient.sens)} (${V.murmurPatient.sens})   spec ${pct(V.murmurPatient.spec)} (${V.murmurPatient.spec})`);
  console.log(`murmur, per valve   : sens ${pct(V.murmurSite.sens)}   spec ${pct(V.murmurSite.spec)}   at most-audible valve ${pct(V.murmurSite.audibleSens)}`);
  console.log(`by grade            : ` + Object.entries(V.byGrade).map(([k, v]) => `${k} ${pct(v)}`).join('  '));
  console.log(`by timing           : ` + Object.entries(V.byTiming).map(([k, v]) => `${k} ${pct(v)}`).join('  '));
  console.log(`diastolic murmur    : ${pct(V.diastolic.detected)} of ${V.diastolic.truthN} labelled, false pos ${pct(V.diastolic.falsePos)}`);
  console.log(`S1 recall ${pct(V.segmentation.s1Recall)}  S2 recall ${pct(V.segmentation.s2Recall)}  median |err| S1 ${V.segmentation.s1ErrMedianMs} ms / S2 ${V.segmentation.s2ErrMedianMs} ms`);
  console.log(`sysDia median: murmur valves ${V.sysDiaByTruth.murmur}  vs normal valves ${V.sysDiaByTruth.normal}`);
  console.log(`gates: HR resolved ${pct(V.gates.hrResolved)}  >=3 beats ${pct(V.gates.beatsGE3)}  >=3 quiet ${pct(V.gates.quietGE3)}  sysDia ${pct(V.gates.sysDiaResolved)}`);
  console.log('-> meta/circor_validation.json, meta/features/circor.jsonl');
  await b.close(); srv.close();
})();
