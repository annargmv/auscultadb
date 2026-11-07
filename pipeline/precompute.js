// Offline precompute: runs the platform's own analysis code (web/index.html) inside headless Chromium
// on the ORIGINAL WAV files, and writes compact JSON for fast loading. It starts its own static server.
//   node pipeline/precompute.js cohort            → build/index_spr.json  + build/p/<key>.json   (SPRSound cohort, meta/cohort_spr.json)
//   node pipeline/precompute.js kauh              → build/index_kauh.json + build/p/K###.json    (KAUH, meta/kauh_cohort.json)
//   node pipeline/precompute.js circor            → build/index_circor.json + build/p/C####.json (CirCor, meta/circor_cohort.json)
//   node pipeline/precompute.js features [i] [n]  → meta/features/spr.<i>.jsonl  (every SPRSound recording, shard i of n — for Discoveries)
//   node pipeline/precompute.js kfeatures         → meta/features/kauh.jsonl
// Then run pipeline/merge.py to build web/data and web/audio.
const { chromium } = require('playwright'); const fs = require('fs'); const path = require('path'); const http = require('http');
const ROOT = path.resolve(__dirname, '..'); const META = path.join(ROOT, 'meta'), BUILD = path.join(ROOT, 'build');
const PORT = +(process.env.PORT || 8766);
// raw WAVs: SPRSound clone under raw/sprsound, KAUH WAVs (renamed by kauh_prep.py) under build/kauh_wav,
// CirCor straight from raw/ (its file names are already safe, so circor_prep.py copies nothing)
const CIRCOR_DIR = 'raw/circor/circor-heart-sound-1.0.3/training_data';
const wavDir = dir => dir === 'KAUH' ? 'build/kauh_wav' : dir === 'CIRCOR' ? CIRCOR_DIR : 'raw/sprsound/' + dir;
const wavUrl = (dir, name) => `http://localhost:${PORT}/${wavDir(dir)}/${encodeURIComponent(name)}.wav`;
const MIME = { '.html': 'text/html', '.js': 'application/javascript', '.json': 'application/json', '.wav': 'audio/wav' };
function serve() { return http.createServer((q, r) => { const f = path.join(ROOT, decodeURIComponent(q.url.split('?')[0].split('#')[0])); if (!f.startsWith(ROOT) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); r.end(); return } r.writeHead(200, { 'Content-Type': MIME[path.extname(f)] || 'application/octet-stream' }); fs.createReadStream(f).pipe(r) }).listen(PORT) }
const MODE = process.argv[2] || 'cohort'; const KAUH = MODE === 'kauh' || MODE === 'kfeatures'; const CIRCOR = MODE === 'circor';
const COHORT_FILE = CIRCOR ? 'circor_cohort.json' : KAUH ? 'kauh_cohort.json' : 'cohort_spr.json';
const INDEX_FILE = CIRCOR ? 'index_circor.json' : KAUH ? 'index_kauh.json' : 'index_spr.json';
const LOC = { p1: 'Left posterior chest', p2: 'Left lateral chest', p3: 'Right posterior chest', p4: 'Right lateral chest', p5: 'Additional site 5 (location not specified)', p6: 'Additional site 6 (location not specified)', p7: 'Additional site 7 (location not specified)', p8: 'Additional site 8 (location not specified)' };
(async () => {
  const srv = serve(); fs.mkdirSync(path.join(BUILD, 'p'), { recursive: true }); fs.mkdirSync(path.join(META, 'features'), { recursive: true });
  const b = await chromium.launch(); const ctx = await b.newContext();
  for (const host of ['https://unpkg.com/**', 'https://cdn.jsdelivr.net/**', 'https://fonts.googleapis.com/**']) await ctx.route(host, r => r.abort());
  const pg = await ctx.newPage(); pg.on('pageerror', e => console.log('PAGEERROR', e.message));
  await pg.goto(`http://localhost:${PORT}/web/index.html#none`); await pg.waitForTimeout(500);
  await pg.evaluate(() => {
    window.b64 = u => { let s = ''; for (let i = 0; i < u.length; i += 8192) s += String.fromCharCode.apply(null, u.subarray(i, i + 8192)); return btoa(s) };
    const r3 = v => Math.round(v * 1000) / 1000;
    window.compactA = A => {
      const q = (n, f) => { const u = new Uint8Array(n); for (let i = 0; i < n; i++) u[i] = Math.max(0, Math.min(255, Math.round(f(i)))); return b64(u) };
      return { dur: r3(A.dur), rr: A.rr && +A.rr.toFixed(2), ie: A.ie && +A.ie.toFixed(3), period: A.period && +A.period.toFixed(3),
        cycles: A.cycles.map(c => [r3(c.start), r3(c.split), r3(c.end), r3(c.expEnd), c.partial ? 1 : 0, c.estimated ? 1 : 0, c.single ? 1 : 0]),
        heart: { hr: A.heart.hr && +A.heart.hr.toFixed(1), quietBeats: A.heart.quietBeats },
        tr: A.transients.map(e => [r3(e.t0), r3(e.t1), +(e.dur * 1000).toFixed(2), Math.round(e.freq), e.fine ? 1 : 0, +e.hl.toFixed(1)]),
        ton: A.tonal.map(s => [r3(s.t0), r3(s.t1), Math.round(s.freq), s.poly ? 1 : 0]),
        nF: A.F.nF, tR: q(A.F.nF, i => A.F.tonal[i].ratio * 10), tF: q(A.F.nF, i => A.F.tonal[i].freq / 8), env: q(A.envDb.length, i => (A.envDb[i] + 80) * 2) };
    };
    window.roundFinding = f => ({ ...f, t0: r3(f.t0), t1: r3(f.t1), conf: +f.conf.toFixed(3), events: f.events.map(([a, b]) => [r3(a), r3(b)]) });
    window.loadWav = async url => normalisePeak(parseWav(await (await fetch(url)).arrayBuffer()));
    // discovery features from a full analysis
    window.features = (A, events) => {
      const F = A.F, nb = F.specs[0].length, k0 = Math.round(150 / binHz), k1 = Math.round(1500 / binHz);
      const fl = Array.from(F.floor), srt = fl.slice().sort((a, b) => a - b), thr = srt[Math.floor(srt.length * 0.6)];
      const ms = new Float64Array(nb); let n = 0;
      for (let f = 0; f < F.nF; f++) if (F.floor[f] >= thr) { const s = F.specs[f]; for (let k = 0; k < nb; k++) ms[k] += s[k] * s[k]; n++ }
      let tot = 0; for (let k = k0; k <= k1; k++) tot += ms[k]; let acc = 0, f50 = null, f75 = null, cen = 0;
      for (let k = k0; k <= k1; k++) { acc += ms[k]; cen += ms[k] * k * binHz; if (f50 == null && acc >= tot * 0.5) f50 = k * binHz; if (f75 == null && acc >= tot * 0.75) f75 = k * binHz }
      cen /= tot || 1;
      const cand = A.transients.filter(e => e.hl > (e.fine ? HL_FINE : HL_COARSE));
      const tonDur = A.tonal.reduce((a, s) => a + s.t1 - s.t0, 0);
      // our pitch estimate inside expert continuous-sound events
      const pitches = [];
      for (const e of events) if (/Wheeze|Rhonchi|Stridor/.test(e.type)) {
        const fr = []; for (let f = 0; f < F.nF; f++) { const t = F.t[f]; if (t >= e.start && t <= e.end && F.tonal[f].ratio > TONAL_THR) fr.push(F.tonal[f].freq) }
        if (fr.length >= 5) { fr.sort((a, b) => a - b); pitches.push({ type: e.type, hz: Math.round(fr[fr.length >> 1]), n: fr.length }) }
      }
      return { dur: +A.dur.toFixed(2), f50, f75, cen: Math.round(cen), rr: A.rr && +A.rr.toFixed(1), est: A.cycles.some(c => c.estimated), period: A.period && +A.period.toFixed(2),
        hr: A.heart.hr && +A.heart.hr.toFixed(1), crkRate: +(cand.length / A.dur).toFixed(3), fineRate: +(cand.filter(e => e.fine).length / A.dur).toFixed(3),
        tonFrac: +(tonDur / A.dur).toFixed(4), tonHz: A.tonal.length ? Math.round(A.tonal.map(s => s.freq).sort((a, b) => a - b)[A.tonal.length >> 1]) : null,
        ie: A.ie && +A.ie.toFixed(3), pitches };
    };
  });

  if (MODE === 'kfeatures') {
    const C = JSON.parse(fs.readFileSync(path.join(META, 'kauh_cohort.json'))); const out = fs.createWriteStream(path.join(META, 'features', 'kauh.jsonl'));
    for (const p of C) for (const [loc, r] of Object.entries(p.sites)) {
      const res = await pg.evaluate(async url => { try { const A = analyzeSignal(await loadWav(url)); return features(A, []) } catch (e) { return { err: e.message } } }, wavUrl('KAUH', r.name));
      out.write(JSON.stringify({ key: p.key, pid: p.pid, age: p.age, g: p.g, loc, zone: r.zone, mode: r.mode, label: r.label, sound: p.sound, rel: 'KAUH', name: r.name, disease: p.disease, ...res }) + '\n');
    }
    out.end(); await new Promise(r => out.on('finish', r)); await b.close(); srv.close(); return;
  }
  if (MODE === 'features') {
    const SH = +(process.argv[3] || 0), NS = +(process.argv[4] || 1);
    const D = JSON.parse(fs.readFileSync(path.join(META, 'spr_full_index.json'))); const out = fs.createWriteStream(path.join(META, 'features', `spr.${SH}.jsonl`));
    const recs = D.recs.filter(r => r.label && r.label !== 'Poor Quality').filter((r, i) => i % NS === SH); let i = 0, t0 = Date.now();
    for (const r of recs) {
      const url = wavUrl(r.dir, r.name);
      if (!fs.existsSync(path.join(ROOT, 'raw/sprsound', r.dir, r.name + '.wav'))) continue;
      const res = await pg.evaluate(async ([url, ev]) => { try { const x = await loadWav(url); const A = analyzeSignal(x); return features(A, ev) } catch (e) { return { err: e.message } } }, [url, r.ev]);
      out.write(JSON.stringify({ key: r.key, pid: r.pid, age: r.age, g: r.g, loc: r.loc, label: r.label, labelSrc: r.labelSrc, rel: r.rel, name: r.name, disease: (D.summ[r.key] || [null])[0],
        nEv: r.ev.length, evTypes: r.ev.reduce((m, e) => (m[e.type] = (m[e.type] || 0) + 1, m), {}), ...res }) + '\n');
      if (++i % 250 === 0) console.log(i, '/', recs.length, ((Date.now() - t0) / 1000).toFixed(0) + 's');
    }
    out.end(); await new Promise(r => out.on('finish', r)); await b.close(); srv.close(); return;
  }

  // ---------- cohort precompute ----------
  const cohort = JSON.parse(fs.readFileSync(path.join(META, COHORT_FILE))); const byKey = Object.fromEntries(cohort.map(p => [p.key, p]));
  const index = { version: new Date().toISOString(), src: CIRCOR ? 'circor' : KAUH ? 'kauh' : 'sprsound', patients: [] }; const valRows = []; const cm = {}; const cls = ['Normal', 'CAS', 'DAS', 'CAS & DAS']; cls.forEach(a => { cm[a] = {}; cls.forEach(b => cm[a][b] = 0) });
  const PLANE = { p1: 'post', p3: 'post', p2: 'lat', p4: 'lat' };
  const siteList = p => Object.entries(p.sites).sort(([a], [b]) => a.localeCompare(b)).map(([loc, r]) => ({ loc, site: r.site || LOC[loc] || loc, record: r.label, labelSrc: r.labelSrc, name: r.name, dir: r.dir, file: `audio/${p.key}.json#${r.name}.mp3` /* one audio bundle per patient, built by merge.py */, zone: r.zone, mode: r.mode, audible: r.audible, events: r.ev.map(e => ({ start: +e.start.toFixed(3), end: +e.end.toFixed(3), type: e.type })) }));
  const compactCache = {};
  const analyse = async s => { const c = await pg.evaluate(async ([url, need]) => { window.__A = window.__A || {}; if (!window.__A[url]) { const x = await loadWav(url); window.__A[url] = analyzeSignal(x) } return need ? compactA(window.__A[url]) : null }, [wavUrl(s.dir, s.name), !compactCache[s.name]]); if (c) compactCache[s.name] = c; return compactCache[s.name] };
  let done = 0;
  for (const p of cohort) {
    const sites = siteList(p); const rec = {}; const pairs = {};
    for (const s of sites) rec[s.loc] = await analyse(s);
    // KAUH: the 3 'sites' are one recording through 3 filters. CirCor: the 4 valve sites are the same
    // heart at the same moment and a murmur radiates between them. Neither may be its own baseline.
    const ownNormals = (p.src === 'kauh' || p.src === 'circor') ? [] : sites.filter(s => s.record === 'Normal');
    const refP = p.ref ? byKey[p.ref] : null;
    if (p.ref && !refP) throw new Error(`${p.key}: baseline ${p.ref} is not in ${COHORT_FILE}`);
    const ref = refP ? siteList(refP).filter(s => s.record === 'Normal') : [];
    for (const s of sites) {
      let base = ownNormals.filter(x => x.loc !== s.loc).find(x => PLANE[x.loc] && PLANE[x.loc] === PLANE[s.loc]) || ownNormals.filter(x => x.loc !== s.loc)[0];
      let from = 'own';
      if (!base && s.record === 'Normal' && p.src !== 'kauh' && p.src !== 'circor') base = s;
      if (!base && ref.length) { base = ref.find(x => x.loc === s.loc) || (p.src === 'kauh' && ref.find(x => x.mode === s.mode)) || ref.find(x => PLANE[x.loc] === PLANE[s.loc]) || ref[0]; from = 'control' }
      if (!base) { base = s; from = 'self' }
      if (from === 'control') await analyse(base);
      const res = await pg.evaluate(([u1, u2, site, age, events]) => {
        const A = window.__A[u1], B = window.__A[u2]; const f = compareRecordings(A, B, { site, age }).map(roundFinding);
        const cmp = expertCompare(events, A, B).map(e => ({ start: e.start, end: e.end, type: e.type, expert: e.expert, ours: e.ours, tonal: +e.tonal.toFixed(3), crackles: e.crackles, agree: e.agree, abnAgree: e.abnAgree }));
        return { f, cmp };
      }, [wavUrl(s.dir, s.name), wavUrl(base.dir, base.name), s.site, p.age, s.events]);
      pairs[s.loc] = { base: { from, key: from === 'control' ? p.ref : p.key, loc: base.loc, site: base.site, record: base.record, name: base.name, file: base.file, events: base.events, rec: from === 'control' ? compactCache[base.name] : undefined }, findings: res.f, cmp: res.cmp };
      for (const e of res.cmp) cm[e.expert][e.ours]++;
      const types = [...new Set(res.f.map(z => z.type))];
      if (CIRCOR) {
        // the site label is per valve: does this valve carry the murmur? (pipeline/circor_validate.js
        // is the full validation, over all 941 patients; this row only mirrors the browsable cohort)
        const det = types.includes('murmur') || types.includes('dmurmur');
        const hit = s.record === 'Unknown' ? null : (s.record === 'Murmur' ? det : !det);
        valRows.push({ key: p.key, loc: s.loc, disease: p.disease, record: s.record, labelSrc: s.labelSrc, types, det,
          hit, audible: !!byKey[p.key].sites[s.loc].audible, grading: p.sys && p.sys.grading, timing: p.sys && p.sys.timing, base: from });
      } else {
        const autoCls = (types.some(t => ['wheeze', 'rhonchus', 'stridor'].includes(t)) ? 'CAS' : '') + (types.some(t => ['fine_crackle', 'coarse_crackle', 'rub'].includes(t)) ? ' DAS' : '');
        const hit = s.record === 'Normal' ? !autoCls : ((s.record.includes('CAS') ? autoCls.includes('CAS') : true) && (s.record.includes('DAS') ? autoCls.includes('DAS') : true));
        const partial = !hit && s.record !== 'Normal' && ((s.record.includes('CAS') && autoCls.includes('CAS')) || (s.record.includes('DAS') && autoCls.includes('DAS')));
        valRows.push({ key: p.key, loc: s.loc, mode: s.mode, disease: p.disease, record: s.record, labelSrc: s.labelSrc, types, hit, partial, agree: res.cmp.filter(e => e.agree).length, n: res.cmp.length, base: from });
      }
    }
    const all = Object.values(pairs).flatMap(x => x.findings);
    fs.writeFileSync(path.join(BUILD, `p/${p.key}.json`), JSON.stringify({ key: p.key, rec, pairs }));
    index.patients.push({ key: p.key, pid: p.pid, src: p.src || 'sprsound', sound: p.sound, locRaw: p.locRaw, age: p.age, sex: p.g === '0' ? 'M' : 'F', disease: p.disease, ref: p.ref,
      // CirCor: the expert annotation the heart map and the patient notes render
      ...(CIRCOR ? { murmur: p.murmur, murLoc: p.murLoc, audible: p.audible, outcome: p.outcome, ageBand: p.ageBand, sys: p.sys, dia: p.dia } : {}),
      sites: sites.map(s => ({ loc: s.loc, site: s.site, zone: s.zone, mode: s.mode, record: s.record, labelSrc: s.labelSrc, file: s.file, name: s.name, events: s.events, audible: s.audible, types: [...new Set(pairs[s.loc].findings.map(z => z.type))], nFind: pairs[s.loc].findings.length, base: pairs[s.loc].base.from })),
      summary: { n: all.length, types: [...new Set(all.map(z => z.type))] } });
    await pg.evaluate(() => { window.__A = {} });
    if (++done % 10 === 0) console.log(done, '/', cohort.length);
  }
  // validation summary
  if (CIRCOR) {
    const scored = valRows.filter(r => r.hit !== null);
    const cnt = (L, g) => [L.filter(g).length, L.length];
    index.validation = { kind: 'cardiac', nRec: valRows.length, rows: valRows,
      sens: cnt(scored.filter(r => r.record === 'Murmur'), r => r.det), spec: cnt(scored.filter(r => r.record === 'Normal'), r => !r.det),
      audibleSens: cnt(scored.filter(r => r.audible), r => r.det), recHit: cnt(scored, r => r.hit),
      note: 'Browsable cohort only, and it over-samples murmurs. meta/circor_validation.json holds the full-set figures.' };
  } else {
    const allEv = []; for (const a of cls) for (const b2 of cls) for (let k = 0; k < cm[a][b2]; k++) allEv.push([a, b2]);
    const frac = (f, g) => { const d = allEv.filter(g); return [d.filter(f).length, d.length] };
    index.validation = { cm, n: allEv.length, nRec: valRows.length, rows: valRows,
      spec: frac(e => e[1] === 'Normal', e => e[0] === 'Normal'), sens: frac(e => e[1] !== 'Normal', e => e[0] !== 'Normal'),
      cas: frac(e => e[1].includes('CAS'), e => e[0].includes('CAS')), das: frac(e => e[1].includes('DAS'), e => e[0].includes('DAS')),
      recHit: [valRows.filter(r => r.hit).length, valRows.length] };
  }
  fs.writeFileSync(path.join(BUILD, INDEX_FILE), JSON.stringify(index));
  console.log('cohort done', cohort.length, 'validation', JSON.stringify(CIRCOR ? { sens: index.validation.sens, spec: index.validation.spec, audibleSens: index.validation.audibleSens } : { spec: index.validation.spec, sens: index.validation.sens, cas: index.validation.cas, das: index.validation.das, recHit: index.validation.recHit }));
  await b.close(); srv.close();
})();
