// Upload flow + apnea/AHI scoring, on two synthetic 12-minute recordings:
//   long_test.wav   11 inserted 15-s pauses -> the AASM scorer must find exactly 11 apneas (sensitivity)
//   quiet_test.wav  regular breathing, no pauses -> it must find 0 events (specificity)
// The specificity case matters: without it nothing in the suite catches a false-positive regression.
// Usage: npm run test:ahi   (generates tests/audio/*.wav first via tests/make_test_audio.py)
// Port note: 8771 collides with macOS `sharingd` on some machines — keep this off it.
const { open, check, ROOT } = require('./helpers'); const path = require('path');

async function score(pg, url, file, name, age) {
  await pg.goto(url()); await pg.waitForFunction(() => window.__app.REAL.status === 'ready', null, { timeout: 90000 });
  await pg.click('#btnAdd'); await pg.fill('input[name=name]', name); await pg.fill('input[name=age]', String(age));
  await pg.setInputFiles('input[name=fBase]', path.join(ROOT, 'tests/audio/base_test.wav'));
  await pg.setInputFiles('input[name=fCur]', path.join(ROOT, 'tests/audio/' + file));
  await pg.click('#pf button.primary'); await pg.waitForFunction(() => location.hash.startsWith('#patient-up-'), null, { timeout: 60000 });
  await pg.waitForFunction(() => document.querySelector('#flowStats .stat'), null, { timeout: 180000 });
  return pg.evaluate(() => {
    const ev = (window.__flow && window.__flow.events) || [];
    return { txt: document.querySelector('#flowStats').innerText.replace(/\n+/g, ' | '),
             apnea: ev.filter(e => e.type === 'apnea').length, hypo: ev.filter(e => e.type === 'hypopnea').length,
             cand: ev.filter(e => e.type === 'candidate').length,
             withShape: ev.filter(e => e.shape).length, withConf: ev.filter(e => typeof e.conf === 'number').length,
             conf: ev.filter(e => typeof e.conf === 'number').map(e => e.conf) };
  });
}

(async () => {
  const { page: pg, errors, url, close } = await open({ port: 8781 });
  try {
    // --- sensitivity: 11 real pauses
    const a = await score(pg, url, 'long_test.wav', 'Test adult', 45);
    console.log(a.txt);
    const ap = +((a.txt.match(/(\d+)\s*\|?\s*apneas?/i) || a.txt.match(/apneas?\D*(\d+)/i) || [])[1]);
    check(ap === 11, `apneas scored: ${ap} (expected 11)`);
    check(a.withShape === a.apnea + a.hypo, `shape descriptors on ${a.withShape}/${a.apnea + a.hypo} scored events`);
    check(a.withConf === a.apnea + a.hypo && a.conf.every(c => c >= 0 && c <= 1), `confidence in [0,1] on all ${a.withConf} events`);
    check(Math.min(...a.conf) >= 0.5, `real apneas score confidently (min ${Math.min(...a.conf).toFixed(2)})`);

    // --- specificity: the same breathing with no pauses must score nothing
    const q = await score(pg, url, 'quiet_test.wav', 'Test adult quiet', 45);
    console.log(q.txt);
    check(q.apnea === 0 && q.hypo === 0, `no events on uninterrupted breathing (apnea ${q.apnea}, hypopnea ${q.hypo}, candidate ${q.cand})`);
  } catch (e) { check(false, e.message) }
  check(errors.length === 0, 'no page errors' + (errors.length ? ': ' + errors.join(' | ') : ''));
  await close();
})();
