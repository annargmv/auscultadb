// End-to-end check of the lean clinical brief (lean/): 10 recordings load and draw, every output carries its validation,
// only working skills are listed, conclusions and the two charts render. Screenshots: tests/out/lean-*.png
//   npm run test:lean      (build first: python3 pipeline/lean_build.py)
const { open, check, ROOT } = require('./helpers'); const fs = require('fs'); const path = require('path');
const OUT = path.join(ROOT, 'tests', 'out'); fs.mkdirSync(OUT, { recursive: true });
(async () => {
  const { page: pg, errors, url, close } = await open({ port: 8781, root: path.join(ROOT, 'lean') });
  try {
    await pg.goto(url('')); await pg.waitForFunction(() => window.__brief && document.querySelectorAll('.case').length, null, { timeout: 30000 });
    await pg.waitForFunction(() => (window.__ready || 0) >= window.__brief.cases.length, null, { timeout: 120000 });
    const r = await pg.evaluate(() => { const B = window.__brief; return {
      cases: B.cases.length, adults: B.cases.filter(c => c.adult).length,
      outs: B.cases.flatMap(c => c.outputs).length, withVal: B.cases.flatMap(c => c.outputs).filter(o => o.validation && o.validation.length > 30).length,
      skills: document.querySelectorAll('#skills tbody tr').length, failing: document.querySelectorAll('#skills .pill.fails').length,
      concl: document.querySelectorAll('#concl li').length, notDx: /not a diagnos/i.test(document.body.innerText),
      charts: [...document.querySelectorAll('canvas')].filter(c => c.width > 0).length, spect: document.querySelectorAll('.wave').length } });
    check(r.cases === 10, `brief shows ${r.cases} recordings (${r.adults} adults)`);
    check(r.outs > 0 && r.withVal === r.outs, `every output has its validation beside it (${r.withVal}/${r.outs})`);
    check(r.skills >= 4 && r.failing === 0, `skills table lists ${r.skills} working skills and none that failed`);
    check(r.concl >= 4 && r.notDx, `conclusions written (${r.concl}), "not a diagnosis" stated`);
    check(r.charts >= 2, `two charts drawn from the validation results (${r.charts} canvases)`);
    await pg.screenshot({ path: path.join(OUT, 'lean-desktop.png'), fullPage: true });
    await pg.setViewportSize({ width: 400, height: 900 }); await pg.waitForTimeout(500);
    const wide = await pg.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    check(wide <= 1, `no sideways scroll at phone width (${wide} px over)`);
    await pg.screenshot({ path: path.join(OUT, 'lean-phone.png'), fullPage: false });
    check(errors.length === 0, `no page errors${errors.length ? ': ' + errors.slice(0, 2).join(' | ') : ''}`);
    // investor demo page (lean/demo.html): same data, hero recording, three tiles, batch player switches recordings
    await pg.setViewportSize({ width: 1360, height: 1000 }); await pg.goto(url('').replace('index.html', 'demo.html'));
    await pg.waitForFunction(() => window.__demo && (window.__ready || 0) >= 2, null, { timeout: 60000 });
    await pg.click('.pick[data-i="5"]'); await pg.waitForFunction(() => (window.__ready || 0) >= 3, null, { timeout: 60000 });
    const d = await pg.evaluate(() => ({ tiles: document.querySelectorAll('.tile').length, hero: document.querySelectorAll('#hRead .ro').length, picks: document.querySelectorAll('.pick').length,
      shown: document.querySelector('#detail h3').textContent, want: window.__demo.cases[5].title, notDx: /not a diagnos/i.test(document.body.innerText), charts: document.querySelectorAll('canvas').length }));
    check(d.tiles === 3 && d.hero >= 2 && d.picks === 10 && d.shown === d.want && d.notDx && d.charts >= 2, `investor demo: hero readouts ${d.hero}, ${d.tiles} tiles, player switches to "${d.shown}", ${d.charts} charts`);
    await pg.screenshot({ path: path.join(OUT, 'demo-desktop.png'), fullPage: true });
    check(errors.length === 0, `demo: no page errors${errors.length ? ': ' + errors.slice(0, 2).join(' | ') : ''}`);
  } catch (e) { check(false, 'lean brief: ' + e.message) } finally { await close() }
})();
