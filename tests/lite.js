// End-to-end check of AuscultaDB Lite (lite/): pages render, every finding carries its validation, the player marks beats /
// wheeze segments, Analyze runs the extracted engine and agrees with the stored results, and the engine reproduces the full
// app's numbers on an original recording (when raw/cinc2016 is on disk). Screenshots: tests/out/lite-*.png
//   npm run test:lite
const { open, check, ROOT } = require('./helpers'); const fs = require('fs'); const path = require('path');
const OUT = path.join(ROOT, 'tests', 'out'); fs.mkdirSync(OUT, { recursive: true });
const WAV = path.join(ROOT, 'raw', 'cinc2016', 'cinc', 'training-a');
(async () => {
  const { page: pg, errors, url, close } = await open({ port: 8786, root: path.join(ROOT, 'lite') });
  await pg.route('**/__wav/**', r => { const f = path.join(WAV, path.basename(r.request().url())); fs.existsSync(f) ? r.fulfill({ status: 200, contentType: 'audio/wav', body: fs.readFileSync(f) }) : r.fulfill({ status: 404 }) });
  const go = async h => { await pg.goto(url(h)); await pg.waitForFunction(() => window.__lite, null, { timeout: 30000 }) };
  try {
    await go('#home'); await pg.waitForSelector('table tbody tr');
    const home = await pg.evaluate(() => ({ rows: document.querySelectorAll('table tbody tr').length, notDx: /not a diagnostic device/i.test(document.body.innerText) }));
    check(home.rows >= 5 && home.notDx, `home: ${home.rows} validated measurements listed, "not a diagnostic device" shown`);
    await pg.screenshot({ path: path.join(OUT, 'lite-home.png'), fullPage: true });
    await go('#patients'); const n0 = await pg.$$eval('tbody tr', r => r.length); await pg.click('.chip[data-f="heart"]'); const nh = await pg.$$eval('tbody tr', r => r.length);
    check(n0 === 10 && nh === 5, `batch list: ${n0} recordings, heart filter ${nh}`);
    // heart patient: beats marked, every output with its validation line
    await go('#patient-1'); await pg.waitForFunction(() => (window.__ready || 0) >= 1, null, { timeout: 60000 });
    const p1 = await pg.evaluate(() => ({ finds: document.querySelectorAll('.find').length, vals: document.querySelectorAll('.val-line').length, mark: document.querySelector('#pmark').textContent }));
    const an1 = await pg.evaluate(() => window.__annot);
    check(p1.finds >= 2 && p1.finds === p1.vals && /beats found/.test(p1.mark) && an1 && an1.layers.includes('s1') && an1.layers.includes('s2'), `heart patient: ${p1.finds} findings with validation; annotated ${an1 && an1.n} regions (${an1 && an1.layers.join(', ')})`);
    await pg.waitForTimeout(2500); await pg.screenshot({ path: path.join(OUT, 'lite-patient-heart.png'), fullPage: true });
    const ia = await pg.evaluate(() => window.__lite.cases.findIndex(c => c.regions && c.regions.length));
    await pg.evaluate(() => { window.__annot = null; window.__ready = 0 }); await pg.evaluate(i => location.hash = '#patient-' + i, ia);
    await pg.waitForFunction(() => window.__annot && (window.__ready || 0) >= 1, null, { timeout: 60000 });
    const pl = await pg.evaluate(() => document.querySelector('#pmark').textContent);
    const anl = await pg.evaluate(() => window.__annot);
    check(/breath cycles/.test(pl) && anl && anl.layers.includes('insp') && anl.layers.includes('wz'), `lung patient annotated: ${anl && anl.layers.join(', ')} — ${pl.slice(0, 60)}…`);
    await pg.waitForTimeout(2500); await pg.screenshot({ path: path.join(OUT, 'lite-patient-lung.png'), fullPage: true });
    // Analyze on batch recordings (playback MP3) agrees with the stored results from the original files
    for (const i of [1, ia]) {
      await go('#analyze'); await pg.selectOption('#aCase', String(i)); await pg.evaluate(() => window.__analyze = null); await pg.click('#aGo');
      await pg.waitForFunction(() => window.__analyze, null, { timeout: 120000 });
      const r = await pg.evaluate(i => { const c = window.__lite.cases[i], got = window.__analyze.outs[0], want = c.outputs[0]; return { skill: got.skill, got: parseFloat(got.value), want: parseFloat(want.value) } }, i);
      check(Math.abs(r.got - r.want) <= 5, `analyze ${r.skill}: ${r.got} on the MP3 copy vs ${r.want} stored from the original (≤ 5 points)`);
    }
    await pg.screenshot({ path: path.join(OUT, 'lite-analyze.png'), fullPage: true });
    // engine parity with the full app on an original recording
    const st = fs.existsSync(path.join(WAV, 'a0251.wav')) ? JSON.parse(fs.readFileSync(path.join(ROOT, 'meta', 'features', 'cinc2016.jsonl'), 'utf8').split('\n').find(l => l.includes('"a0251"'))) : null;
    if (st) {
      const p = await pg.evaluate(async () => { await ensureModels('heart'); const x = normalisePeak(parseWav(await (await fetch('/__wav/a0251.wav')).arrayBuffer())); return cnnMurmur(x).p });
      check(Math.abs(p - st.cnnP) < 1e-4, `engine = full app on the original WAV: murmur probability ${p.toFixed(5)} vs ${st.cnnP}`);
    } else check(true, 'engine parity skipped (raw/cinc2016 not on disk)');
    await go('#conclusions'); await pg.waitForSelector('ol.concl li');
    const c = await pg.evaluate(() => ({ li: document.querySelectorAll('ol.concl li').length, charts: [...document.querySelectorAll('canvas')].filter(x => x.width).length }));
    check(c.li >= 5 && c.charts === 2, `conclusions: ${c.li} statements, ${c.charts} charts`);
    await pg.screenshot({ path: path.join(OUT, 'lite-conclusions.png'), fullPage: true });
    // disease atlas: result stated as failed validation when it failed, batch answers out-of-fold, two charts from the training result
    await go('#disease'); await pg.waitForSelector('#dLodo');
    const d = await pg.evaluate(() => { const P = window.__lite.disease.populations; return { failedShown: /failed validation/.test(document.body.innerText), anyFail: !P.children.passed || !P.adults.passed,
      rows: document.querySelectorAll('.card table')[0].querySelectorAll('tbody tr').length, nOof: window.__lite.cases.filter(c => c.diseaseOof).length, charts: [...document.querySelectorAll('canvas')].filter(x => x.width).length, refs: document.querySelectorAll('.refs a[href^="https://doi.org/"]').length } });
    check((d.failedShown || !d.anyFail) && d.rows === d.nOof && d.nOof >= 4 && d.charts === 2 && d.refs >= 3, `disease page: ${d.rows} batch answers (out-of-fold), ${d.charts} charts, ${d.refs} DOI links, failed validation stated: ${d.failedShown}`);
    await pg.screenshot({ path: path.join(OUT, 'lite-disease.png'), fullPage: true });
    // longevity: the page's Cox formula reproduces lifelines' own prediction, and the exercise scenarios lower the risk
    await go('#longevity'); await pg.waitForSelector('#lgChart');
    const L = await pg.evaluate(() => { const M = window.__lite.longevity, k = M.cox.check; return { js: coxRisk(M, k.x), py: k.risk, out: document.querySelector('#lgOut').textContent, charts: [...document.querySelectorAll('canvas')].filter(x => x.width).length } });
    const nums = [...L.out.matchAll(/(?:Now|\)): ([\d.]+) %/g)].map(m => +m[1]);
    check(Math.abs(L.js - L.py) < 1e-6 && L.charts === 2 && nums.length === 3 && nums[1] < nums[0] && nums[2] < nums[1], `longevity: page risk ${L.js.toFixed(6)} = lifelines ${L.py.toFixed(6)}; ${L.out.slice(0, 90)}…`);
    await pg.screenshot({ path: path.join(OUT, 'lite-longevity.png'), fullPage: true });
    await pg.setViewportSize({ width: 400, height: 900 }); await go('#patient-1'); await pg.waitForTimeout(800);
    const wide = await pg.evaluate(() => document.documentElement.scrollWidth - innerWidth);
    check(wide <= 1, `phone width: no sideways scroll (${wide} px)`); await pg.screenshot({ path: path.join(OUT, 'lite-phone.png') });
    check(errors.length === 0, `no page errors${errors.length ? ': ' + errors.slice(0, 2).join(' | ') : ''}`);
  } catch (e) { check(false, 'lite: ' + e.message) } finally { await close() }
})();
