// Upload flow + apnea/AHI scoring on a synthetic 12-minute recording with 11 inserted 15-s pauses.
// Usage: npm run test:ahi   (generates tests/audio/*.wav first via tests/make_test_audio.py)
const { open, check, ROOT } = require('./helpers'); const path = require('path');
(async () => {
  const { page: pg, errors, url, close } = await open({ port: 8771 });
  try {
    await pg.goto(url()); await pg.waitForFunction(() => window.__app.REAL.status === 'ready', null, { timeout: 90000 });
    await pg.click('#btnAdd'); await pg.fill('input[name=name]', 'Test adult'); await pg.fill('input[name=age]', '45');
    await pg.setInputFiles('input[name=fBase]', path.join(ROOT, 'tests/audio/base_test.wav'));
    await pg.setInputFiles('input[name=fCur]', path.join(ROOT, 'tests/audio/long_test.wav'));
    await pg.click('#pf button.primary'); await pg.waitForFunction(() => location.hash.startsWith('#patient-up-'), null, { timeout: 60000 });
    await pg.waitForFunction(() => document.querySelector('#flowStats .stat'), null, { timeout: 180000 });
    const txt = await pg.evaluate(() => document.querySelector('#flowStats').innerText.replace(/\n+/g, ' | '));
    console.log(txt);
    const ap = +((txt.match(/(\d+)\s*\|?\s*apneas?/i) || txt.match(/apneas?\D*(\d+)/i) || [])[1]);
    check(ap === 11, `apneas scored: ${ap} (expected 11)`);
  } catch (e) { check(false, e.message) }
  check(errors.length === 0, 'no page errors' + (errors.length ? ': ' + errors.join(' | ') : ''));
  await close();
})();
