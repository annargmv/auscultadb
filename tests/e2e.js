// End-to-end smoke test of the built site (web/): patient list, a child, a KAUH adult, a CirCor heart-sound
// patient, site/mode/valve switch, cache, Datasets, Discoveries (incl. the cardiac validation section).
// Usage: npm test        (screenshots go to tests/out/)
const { open, liveReady, check, ROOT } = require('./helpers'); const fs = require('fs'); const path = require('path');
const OUT = path.join(ROOT, 'tests', 'out'); fs.mkdirSync(OUT, { recursive: true });
(async () => {
  const { page: pg, errors, url, close } = await open();
  try {
    let t0 = Date.now(); await pg.goto(url('#patients'));
    await pg.waitForFunction(() => window.__app.REAL.status === 'ready' && document.querySelector('#fDx'), null, { timeout: 60000 });
    const n = await pg.evaluate(() => [__app.patients.length, __app.patients.filter(p => p.source === 'sprsound').length, __app.patients.filter(p => p.source === 'kauh').length, __app.patients.filter(p => p.source === 'circor').length]);
    check(n[1] > 0 && n[2] > 0 && n[3] > 0, `list: ${n[0]} patients (${n[1]} SPRSound, ${n[2]} KAUH, ${n[3]} CirCor) in ${Date.now() - t0} ms`);
    await pg.screenshot({ path: path.join(OUT, 'list.png') });

    // a severe child that uses a matched healthy control as baseline
    const child = await pg.evaluate(() => (__app.patients.find(p => p.severe && p.ref) || __app.patients.find(p => p.source === 'sprsound')).id);
    t0 = Date.now(); await pg.goto(url('#patient-' + child)); await liveReady(pg);
    check(true, `child ${child} playable in ${Date.now() - t0} ms`);
    await pg.click('#playPath'); await pg.waitForTimeout(1000);
    await pg.screenshot({ path: path.join(OUT, 'child.png'), fullPage: true });
    const other = await pg.evaluate(id => { const p = __app.patients.find(x => x.id === id); return (p.sites.find(s => s.loc !== p.curLoc) || p.sites[0]).loc }, child);
    t0 = Date.now(); await pg.click(`g.site[data-loc="${other}"]`); await liveReady(pg); check(true, `site switch in ${Date.now() - t0} ms`);

    // revisit comes from Cache Storage
    await pg.goto(url('#patients')); t0 = Date.now(); await pg.goto(url('#patient-' + child)); await liveReady(pg);
    const cache = await pg.evaluate(async () => { const k = await caches.keys(); return k.length ? (await (await caches.open(k[0])).keys()).length : 0 });
    check(cache > 0, `revisit in ${Date.now() - t0} ms, ${cache} cache entries`);

    // a KAUH adult, switch filter mode
    const adult = await pg.evaluate(() => __app.patients.find(p => p.source === 'kauh' && !p.control).id);
    await pg.goto(url('#patient-' + adult)); await liveReady(pg);
    await pg.click('[data-listen$="-B"]');
    await pg.waitForFunction(id => __app.patients.find(p => p.id === id).curLoc.endsWith('-B') && window.__live() && window.__live().ok, adult, { timeout: 60000 });
    check(true, `KAUH ${adult}: switched to ${await pg.evaluate(id => __app.patients.find(p => p.id === id).site, adult)}`);
    await pg.screenshot({ path: path.join(OUT, 'adult.png'), fullPage: true });

    // a CirCor heart-sound patient: four valve sites, the heart map, and the expert annotation
    const heart = await pg.evaluate(() => { const p = __app.patients.find(x => x.source === 'circor' && x.murmur === 'Present' && x.sites.length === 4) || __app.patients.find(x => x.source === 'circor'); return p && { id: p.id, murmur: p.murmur, murLoc: p.murLoc, audible: p.audible } });
    check(heart && heart.murmur, `CirCor patient ${heart && heart.id}: murmur ${heart && heart.murmur} at ${heart && (heart.murLoc || []).join('+')}, audible ${heart && heart.audible}`);
    t0 = Date.now(); await pg.goto(url('#patient-' + heart.id)); await liveReady(pg);
    check(true, `CirCor ${heart.id} playable in ${Date.now() - t0} ms`);
    const valves = await pg.evaluate(() => [...document.querySelectorAll('svg[aria-label*="cardiac auscultation"] g.site')].map(g => g.dataset.loc));
    check(valves.length === 4, `heart map shows ${valves.length} valves (${valves.join(',')})`);
    const lic = await pg.evaluate(() => { const d = [...document.querySelectorAll('dt')].find(x => x.textContent === 'Source'); const t = d && d.nextElementSibling.textContent; return { t, ok: /ODC-By/.test(t) && !/CC BY/.test(t) } });
    check(lic.ok, `CirCor credited under its own licence: ${(lic.t || '').replace(/\s+/g, ' ').trim().slice(0, 64)}`);
    check(await pg.evaluate(() => !document.body.innerText.includes('SPRSound physicians labelled')), 'no SPRSound event-label card on a dataset without event timings');
    await pg.screenshot({ path: path.join(OUT, 'circor.png'), fullPage: true });
    const v2 = await pg.evaluate(id => { const p = __app.patients.find(x => x.id === id); return (p.sites.find(s => s.loc !== p.curLoc) || p.sites[0]).loc }, heart.id);
    await pg.click(`g.site[data-loc="${v2}"]`); await liveReady(pg);
    check(await pg.evaluate(id => __app.patients.find(p => p.id === id).curLoc, heart.id) === v2, `valve switch to ${v2}`);

    await pg.goto(url('#datasets')); await pg.waitForTimeout(1000); await pg.screenshot({ path: path.join(OUT, 'datasets.png'), fullPage: true });
    check(await pg.evaluate(() => /KAUH/.test(document.body.innerText) && /SPRSound/.test(document.body.innerText)), 'datasets page lists SPRSound and KAUH');
    check(await pg.evaluate(() => /CirCor DigiScope/.test(document.body.innerText) && /ODC-By/.test(document.body.innerText)), 'datasets page lists CirCor with its licence');
    await pg.goto(url('#discover')); await pg.waitForFunction(() => document.querySelector('#d-device'), null, { timeout: 30000 }); await pg.waitForTimeout(500);
    await pg.screenshot({ path: path.join(OUT, 'discover.png'), fullPage: true });
    check(await pg.evaluate(() => document.querySelectorAll('canvas').length) >= 4, 'discoveries charts rendered');
    const h = await pg.evaluate(() => { const c = document.querySelector('#d-heart'); return c && { chart: !!c.querySelector('#c-heart'), rows: c.querySelectorAll('table tr').length } });
    check(h && h.chart && h.rows > 5, `cardiac validation section rendered (${h ? h.rows : 0} rows, grade chart ${h && h.chart})`);
  } catch (e) { check(false, e.message) }
  check(errors.length === 0, 'no page errors' + (errors.length ? ': ' + errors.join(' | ') : ''));
  await close();
})();
