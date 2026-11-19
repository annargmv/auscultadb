// Template summaries (the page's writtenSummary) for the patients in meta/medgemma_drafts.json -> build/summaries.json,
// so pipeline/medgemma_draft.py --template can score them with the same checks as the MedGemma drafts.
//   node pipeline/summary_export.js && python3 pipeline/medgemma_draft.py --template
const fs = require('fs'); const path = require('path');
const { open, ROOT } = require('../tests/helpers');
(async () => {
  const keys = JSON.parse(fs.readFileSync(path.join(ROOT, 'meta', 'medgemma_drafts.json'))).drafts.map(d => d.key);
  const { page, url, close } = await open({ port: 8771 });
  await page.goto(url('#patients')); await page.waitForFunction(() => window.__app.REAL.status === 'ready', null, { timeout: 60000 });
  const out = await page.evaluate(async keys => { const A = window.__app, r = {};
    for (const k of keys) { const p = A.patients.find(x => x.key === k); if (!p) continue; p.detail = p.detail || await A.getDetail(p); r[k] = A.writtenSummary(p) }
    return r }, keys);
  fs.mkdirSync(path.join(ROOT, 'build'), { recursive: true }); fs.writeFileSync(path.join(ROOT, 'build', 'summaries.json'), JSON.stringify(out, null, 1));
  console.log(Object.keys(out).length, 'summaries -> build/summaries.json'); await close();
})();
