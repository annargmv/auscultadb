// Apnea / AHI validation on real nights: run the page's OWN flow pipeline on PSG-Audio and record what it scores.
// For every prepared patient-hour (meta/psg_audio_index.json, made by psg_audio_prep.py) and both microphones
// (tracheal contact mic, ambient mic), in headless Chromium:
//     x = normalisePeak(parseWav(wav))   — the same steps as an upload (decodeToSR → normalisePeak)
//     F = await computeFlow(x)           — the page's acoustic flow proxy
//     S = scoreFlow(F, AGE)              — the page's AASM scoring, adult rules
// One hour per call, which is the page's own limit (MAX_FLOW_S = 3600). Nothing is tuned: thresholds are the
// shipped ones. Output: meta/features/psg_audio.jsonl, one row per patient × hour × channel.
// Age is not usable (the .rml birth date is encrypted); every PSG-Audio patient is an adult (23–85 y), and the
// adult rules do not depend on the exact age, so AGE = 40 selects them.
//   node pipeline/psg_audio_validate.js
const { open } = require('../tests/helpers'); const fs = require('fs'); const path = require('path');
const ROOT = path.resolve(__dirname, '..'); const RAW = path.join(ROOT, 'raw', 'psg_audio');
const IDX = path.join(ROOT, 'meta', 'psg_audio_index.json'); const OUT = path.join(ROOT, 'meta', 'features', 'psg_audio.jsonl');
const AGE = 40;
(async () => {
  const idx = JSON.parse(fs.readFileSync(IDX, 'utf8'));
  const { page: pg, url, close, errors } = await open({ port: 8771 });
  await pg.route('**/__psg/**', r => { const f = path.join(RAW, decodeURIComponent(r.request().url().split('/__psg/')[1]));
    if (!f.startsWith(RAW) || !fs.existsSync(f)) return r.fulfill({ status: 404 }); r.fulfill({ status: 200, contentType: 'audio/wav', body: fs.readFileSync(f) }) });
  await pg.goto(url('#none')); await pg.waitForFunction(() => window.Meyda && typeof computeFlow === 'function', null, { timeout: 60000 });
  const done = new Set(fs.existsSync(OUT) ? fs.readFileSync(OUT, 'utf8').trim().split('\n').filter(Boolean).map(l => { const o = JSON.parse(l); return `${o.pid}|${o.k}|${o.ch}` }) : []);
  const out = fs.createWriteStream(OUT, { flags: 'a' });
  for (const p of idx.patients) for (const h of p.hours) for (const [ch, wav] of Object.entries(h.wav)) {
    if (done.has(`${p.pid}|${h.k}|${ch}`)) continue;
    const t0 = Date.now();
    const r = await pg.evaluate(async ({ u, age }) => {
      const x = normalisePeak(parseWav(await (await fetch(u)).arrayBuffer()));
      const F = await computeFlow(x), S = scoreFlow(F, age);
      const r3 = v => Math.round(v * 1000) / 1000;
      return { dur: r3(F.dur), period: r3(S.period), periodSrc: S.periodSrc, B: S.B,
        events: S.events.map(e => ({ type: e.type, start: r3(e.start), end: r3(e.end), dur: r3(e.dur), drop: e.drop, conf: e.conf ?? null, edge: !!e.edge })) };
    }, { u: `http://localhost:8771/__psg/${wav}`, age: AGE });
    out.write(JSON.stringify({ pid: p.pid, k: h.k, ch, age: AGE, ...r }) + '\n');
    const n = r.events.filter(e => e.type !== 'candidate').length;
    console.log(`${p.pid} h${h.k} ${ch.padEnd(5)} ${n} events (expert ${h.events.length}) period ${r.period}s [${r.periodSrc}] ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  }
  out.end(); if (errors.length) console.log('page errors:', errors.join(' | '));
  await close();
})();
