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
    const gh = () => pg.evaluate(() => document.querySelector('.livegrid').offsetHeight);
    const hs = new Set([await gh()]); await pg.click('#playPath');
    for (let i = 0; i < 12; i++) { await pg.waitForTimeout(80); hs.add(await gh()); }
    check(hs.size === 1, `live panels keep their height while playing (${[...hs].join(', ')} px)`);
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
    const tr = await pg.evaluate(() => { const f = document.querySelector('#findings'); const body = document.body.innerText;
      return { key: !!document.querySelector('.markkey'), det: !!document.querySelector('#findings .swatch.det') || !f || !f.textContent.trim(),
               record: /Track record:/.test(body) || !/confidence/.test(body) }; });
    check(tr.key, 'expert vs detected mark key shown');
    check(tr.record, 'findings carry a measured track record, not just a confidence score');
    const valves = await pg.evaluate(() => [...document.querySelectorAll('svg[aria-label*="cardiac auscultation"] g.site')].map(g => g.dataset.loc));
    check(valves.length === 4, `heart map shows ${valves.length} valves (${valves.join(',')})`);
    const lic = await pg.evaluate(() => { const d = [...document.querySelectorAll('dt')].find(x => x.textContent === 'Source'); const t = d && d.nextElementSibling.textContent; return { t, ok: /ODC-By/.test(t) && !/CC BY/.test(t) } });
    check(lic.ok, `CirCor credited under its own licence: ${(lic.t || '').replace(/\s+/g, ' ').trim().slice(0, 64)}`);
    // a heart-sound recording must not advertise respiratory measurements it cannot make
    const resp = await pg.evaluate(() => {
      const body = document.body.innerText;
      const rr = [...document.querySelectorAll('dt')].filter(d => /Resp\. rate/.test(d.textContent)).map(d => d.nextElementSibling.textContent.trim());
      return { chart: !!document.querySelector('#flowChart'), stats: !!document.querySelector('#flowStats'),
               rr, explained: /not applicable — heart-sound recording/.test(body) };
    });
    check(!resp.chart && !resp.stats, 'no apnea/AHI panel on a cardiac-only recording');
    check(resp.rr.every(t => /not applicable/.test(t)) && resp.explained, `respiratory rate withheld on heart sounds (${resp.rr[0] || '—'})`);
    check(await pg.evaluate(() => !document.body.innerText.includes('SPRSound physicians labelled')), 'no SPRSound event-label card on a dataset without event timings');
    await pg.waitForFunction(() => { const c = document.querySelector('#hrAgeCard'); return c && !/loading/.test(c.innerText) }, null, { timeout: 15000 });
    const ha = await pg.evaluate(() => { const c = document.querySelector('#hrAgeCard'); return { t: c.innerText.replace(/\s+/g, ' '), chart: !!c.querySelector('#c-hrAge') } });
    check(/bpm/.test(ha.t) && (ha.chart || /No heart rate|No healthy reference/.test(ha.t)) && /not a risk score|No heart rate|No healthy/.test(ha.t), `heart rate for age: ${ha.t.slice(18, 90)}`);
    const rel = await pg.evaluate(() => [...document.querySelectorAll('dt')].find(d => d.textContent === 'Release').nextElementSibling.textContent);
    check(/CC201[45]/.test(rel), `CirCor release from the data: ${rel}`);
    await pg.screenshot({ path: path.join(OUT, 'circor.png'), fullPage: true });
    const v2 = await pg.evaluate(id => { const p = __app.patients.find(x => x.id === id); return (p.sites.find(s => s.loc !== p.curLoc) || p.sites[0]).loc }, heart.id);
    await pg.click(`g.site[data-loc="${v2}"]`); await liveReady(pg);
    check(await pg.evaluate(id => __app.patients.find(p => p.id === id).curLoc, heart.id) === v2, `valve switch to ${v2}`);

    await pg.goto(url('#datasets')); await pg.waitForTimeout(1000); await pg.screenshot({ path: path.join(OUT, 'datasets.png'), fullPage: true });
    check(await pg.evaluate(() => /KAUH/.test(document.body.innerText) && /SPRSound/.test(document.body.innerText)), 'datasets page lists SPRSound and KAUH');
    check(await pg.evaluate(() => /CirCor DigiScope/.test(document.body.innerText) && /ODC-By/.test(document.body.innerText)), 'datasets page lists CirCor with its licence');
    check(await pg.evaluate(() => /Chronic bronchitis/.test(document.body.innerText) && /9hdrddjpny/.test(document.body.innerText)), 'datasets page lists the CB/emphysema set with its DOI');
    check(await pg.evaluate(() => /PSG-Audio/.test(document.body.innerText) && /10\.11922\/sciencedb\.00345/.test(document.body.innerText) && /no recordings from this set are published/.test(document.body.innerText)), 'datasets page lists PSG-Audio with licence, DOI and the no-audio rule');
    await pg.goto(url('#discover')); await pg.waitForFunction(() => document.querySelector('#d-device'), null, { timeout: 30000 }); await pg.waitForTimeout(500);
    await pg.screenshot({ path: path.join(OUT, 'discover.png'), fullPage: true });
    check(await pg.evaluate(() => document.querySelectorAll('canvas').length) >= 4, 'discoveries charts rendered');
    const h = await pg.evaluate(() => { const c = document.querySelector('#d-heart'); return c && { chart: !!c.querySelector('#c-heart'), rows: c.querySelectorAll('table tr').length } });
    check(h && h.chart && h.rows > 5, `cardiac validation section rendered (${h ? h.rows : 0} rows, grade chart ${h && h.chart})`);
    const toc = await pg.evaluate(() => { const t = document.querySelector('#dToc'); if (!t) return null;
      const a = [...t.querySelectorAll('a')];
      return { n: a.length, live: a.filter(x => !x.classList.contains('pending')).length, sticky: getComputedStyle(t).position,
               dead: a.filter(x => x.classList.contains('pending') && getComputedStyle(x).pointerEvents !== 'none').length };
    });
    check(toc && toc.n >= 15 && toc.live >= 12 && toc.sticky === 'sticky' && toc.dead === 0,
      `contents rail: ${toc ? toc.n : 0} links, ${toc ? toc.live : 0} live, blocked ones disabled`);
    const cb = await pg.evaluate(() => { const c = document.querySelector('#d-cbemph'); return c && { rows: c.querySelectorAll('table tr').length, device: /no stethoscope confound/.test(c.innerText), caveat: /does not say which mode/.test(c.innerText) } });
    check(cb && cb.rows > 8 && cb.device && cb.caveat, `CB/emphysema section rendered (${cb ? cb.rows : 0} rows, device + filter-mode caveats present)`);
    // 12 · apnea / AHI on real nights (PSG-Audio): scored by the page's own code, audio never published
    const ap = await pg.evaluate(() => { const c = document.querySelector('#d-apnea'); return c && { chart: !!c.querySelector('#c-apnea'), rows: c.querySelectorAll('table tr').length,
      same: /same code as an upload/.test(c.innerText), off: /Audio stays off this site/.test(c.innerText), live: !document.querySelector('#dToc a[data-jump="d-apnea"]').classList.contains('pending') } });
    check(ap && ap.chart && ap.rows > 8 && ap.same && ap.off && ap.live, `apnea validation section rendered (${ap ? ap.rows : 0} rows, PSG-Audio, page's own scoring)`);
    // 16 · age signal: data-driven, held-out, never claims longevity
    const ag = await pg.evaluate(() => { const c = document.querySelector('#d-age'); return c && { clock: !!c.querySelector('#c-ageclock'), hr: !!c.querySelector('#c-agehr'),
      rows: c.querySelectorAll('table tr').length, honest: /longevity is not measured/.test(c.innerText), pmid: /PMID 34433816/.test(c.innerText) } });
    check(ag && ag.clock && ag.hr && ag.rows > 5 && ag.honest && ag.pmid, `age-signal section rendered (${ag ? ag.rows : 0} rows, longevity caveat ${ag && ag.honest}, refs ${ag && ag.pmid})`);
    await (await pg.$('#d-age')).screenshot({ path: path.join(OUT, 'discover-age.png') });

    // no invented people: recording sites come from the datasets, with live counts
    await pg.goto(url('#doctors')); await pg.waitForTimeout(500);
    const rs = await pg.evaluate(() => ({ dr: /\bDr\.|555-01/.test(document.body.innerText), cards: document.querySelectorAll('#view .card').length,
      sum: [...document.querySelectorAll('#view .kpi b')].filter((_, i) => i % 3 === 0).reduce((a, b) => a + +b.textContent, 0), n: __app.patients.length,
      stamp: /data built \d{4}-\d\d-\d\d/.test(document.querySelector('#dataStamp').textContent), foot: /open datasets/.test(document.querySelector('#footer').textContent) }));
    check(!rs.dr && rs.cards >= 3 && rs.sum === rs.n, `recording sites: ${rs.cards} institutions, ${rs.sum}/${rs.n} patients, no invented doctors`);
    check(rs.stamp && rs.foot, 'live data stamp and generated footer shown');
    await pg.screenshot({ path: path.join(OUT, 'sites.png'), fullPage: true });
  } catch (e) { check(false, e.message) }
  check(errors.length === 0, 'no page errors' + (errors.length ? ': ' + errors.join(' | ') : ''));
  await close();
})();
