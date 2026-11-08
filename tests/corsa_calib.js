// CORSA crackle-waveform calibration: does detectTransients recover a KNOWN initial deflection width?
//   npm run test:corsa     (generates tests/audio/crk_*.wav first via tests/make_test_audio.py)
//
// IDW (initial deflection width) is a CORSA measure — fine crackles 0.50–0.90 ms, coarse 1.00–1.25 ms
// (Sovijärvi et al. 2000; ranges in Palaniappan et al., PMC2990233). At SR = 8000 those widths are only
// 4–10 samples, so the detector interpolates its zero crossings. This test feeds in damped sinusoids whose
// first half-cycle is exactly <idw> ms and checks what comes back, separating algorithm error from the
// sample-rate limit. Expect a small positive bias: the exponential decay pushes each zero crossing late.
const { chromium } = require('playwright'); const fs = require('fs'); const path = require('path'); const http = require('http');
const ROOT = path.resolve(__dirname, '..'); const PORT = +(process.env.PORT || 8804);
const IDWS = [0.50, 0.70, 1.00, 1.25, 2.00];
const TOL = 0.20;        // 20 % — the measured bias is ~10 %, so this catches a real regression, not noise
const MIME = { '.html': 'text/html', '.js': 'application/javascript', '.wav': 'audio/wav' };
let bad = 0;
const check = (ok, msg) => { if (!ok) { console.error('FAIL', msg); bad = 1; process.exitCode = 1 } else console.log('ok  ', msg) };
const srv = http.createServer((q, r) => { const f = path.join(ROOT, decodeURIComponent(q.url.split('?')[0]));
  if (!f.startsWith(ROOT) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); r.end(); return }
  r.writeHead(200, { 'Content-Type': MIME[path.extname(f)] || 'application/octet-stream' }); fs.createReadStream(f).pipe(r) }).listen(PORT);
(async () => {
  const b = await chromium.launch(); const ctx = await b.newContext();
  for (const h of ['https://unpkg.com/**', 'https://cdn.jsdelivr.net/**', 'https://fonts.googleapis.com/**']) await ctx.route(h, r => r.abort());
  const pg = await ctx.newPage(); const errors = []; pg.on('pageerror', e => errors.push(e.message));
  await pg.goto(`http://localhost:${PORT}/web/index.html#none`); await pg.waitForTimeout(400);
  for (const idw of IDWS) {
    const f = `tests/audio/crk_${idw.toFixed(2)}.wav`;
    if (!fs.existsSync(path.join(ROOT, f))) { check(false, `${f} missing — run python3 tests/make_test_audio.py`); continue }
    const r = await pg.evaluate(async u => {
      const x = normalisePeak(parseWav(await (await fetch(u)).arrayBuffer()));
      const ev = detectTransients(x); const med = a => { const s = a.filter(v => v != null).sort((p, q) => p - q); return s.length ? s[s.length >> 1] : null };
      return { n: ev.length, idw: med(ev.map(e => e.idw)), cd2: med(ev.map(e => e.cd2i)), tdw: med(ev.map(e => e.tdw)) };
    }, `http://localhost:${PORT}/${f}`);
    if (r.idw == null) { check(false, `IDW ${idw} ms: nothing detected (${r.n} events)`); continue }
    const got = r.idw * 1000, err = (got - idw) / idw;
    check(r.n >= 35, `IDW ${idw.toFixed(2)} ms: ${r.n}/40 crackles detected`);
    check(Math.abs(err) <= TOL, `IDW ${idw.toFixed(2)} ms -> ${got.toFixed(2)} ms (${(100 * err).toFixed(0)} %, 2CD ${(r.cd2 * 1000).toFixed(2)} ms, TDW ${(r.tdw * 1000).toFixed(2)} ms)`);
  }
  check(errors.length === 0, 'no page errors' + (errors.length ? ': ' + errors.join(' | ') : ''));
  await b.close(); srv.close();
  console.log(bad ? '\nFAILED' : '\nall CORSA calibration checks passed');
})();
