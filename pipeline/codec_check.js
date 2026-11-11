// Robustness of the Analyze-page models to compressed audio → models.json['robustness'].
// The models were trained on features from the ORIGINAL WAV files. Uploads are often compressed. The site itself
// holds every browsable recording as an 8 kHz, 32 kbps MP3 (web/audio/*.json), and meta/features/*.jsonl holds the
// same recordings' features from the original WAV. So: analyse each MP3 with the page's own code, run the deployed
// models on both feature sets, and report how much the probabilities move. Nothing is fitted here.
//   node pipeline/codec_check.js        (then merge.py --bump)
const { open } = require('../tests/helpers'); const fs = require('fs'); const path = require('path');
const ROOT = path.resolve(__dirname, '..'); const WEB = path.join(ROOT, 'web'), FEAT = path.join(ROOT, 'meta', 'features');
const jsonl = f => fs.readFileSync(path.join(FEAT, f), 'utf8').trim().split('\n').map(l => JSON.parse(l));
(async () => {
  const I = JSON.parse(fs.readFileSync(path.join(WEB, 'data', 'index.json')));
  const ref = {};   // recording name -> WAV features
  for (const r of jsonl('circor.jsonl')) ref[r.name] = { kind: 'heart', f: r };
  for (const f of fs.readdirSync(FEAT).filter(f => /^spr\.\d+\.jsonl$/.test(f))) for (const r of jsonl(f)) ref[r.name] = { kind: 'lung', f: r };
  const jobs = [];
  for (const p of I.patients) if (p.src === 'circor' || p.src === 'sprsound') for (const s of p.sites) {
    const [file, mp3] = s.file.split('#'), name = mp3.replace(/\.mp3$/, ''); if (ref[name]) jobs.push({ file, mp3, name, ...ref[name] });
  }
  const { page: pg, url, close } = await open({ port: 8773 });
  await pg.goto(url('#none')); await pg.waitForFunction(() => window.Meyda && typeof recordingFeatures === 'function');
  await pg.evaluate(async () => { window.M = await (await fetch('data/models.json')).json() });
  const rows = []; let last = null, bundle = null;
  for (const j of jobs) {
    if (j.file !== last) { bundle = JSON.parse(fs.readFileSync(path.join(WEB, j.file))); last = j.file }
    const r = await pg.evaluate(async ({ b64, kind, wav }) => {
      const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
      const { x } = await decodeToSR(new File([bin], 'x.mp3', { type: 'audio/mpeg' }), MAX_ANALYSE_S);
      const A = analyzeSignal(x), f = { ...recordingFeatures(A), ...heartFeatures(A) };
      return M.models.filter(m => m.gate.deploy && m.kind === kind).map(m => {
        const v = o => m.features.map(k => o[k] == null ? null : +o[k]);
        const a = modelProba(m, v(wav)), b = modelProba(m, v(f));
        return { id: m.id, d: a.map((q, k) => b[k] - q), flipWav: a.map((q, k) => q >= m.thresholds[k]), flipMp3: b.map((q, k) => q >= m.thresholds[k]),
                 feat: Object.fromEntries(m.features.map(k => [k, [wav[k], f[k]]])) };
      });
    }, { b64: bundle[j.mp3], kind: j.kind, wav: j.f });
    rows.push(...r.map(x => ({ name: j.name, ...x })));
  }
  await close();
  const out = {};
  for (const id of [...new Set(rows.map(r => r.id))]) {
    const R = rows.filter(r => r.id === id), K = R[0].d.length, M = JSON.parse(fs.readFileSync(path.join(WEB, 'data', 'models.json')));
    const cls = M.models.find(m => m.id === id).classes, med = a => { const s = a.slice().sort((x, y) => x - y); return s[s.length >> 1] }, p95 = a => { const s = a.slice().sort((x, y) => x - y); return s[Math.floor(0.95 * (s.length - 1))] };
    out[id] = { recordings: R.length, perClass: cls.map((c, k) => ({ class: c, medianAbsShift: +(100 * med(R.map(r => Math.abs(r.d[k])))).toFixed(1), p95AbsShift: +(100 * p95(R.map(r => Math.abs(r.d[k])))).toFixed(1),
      flagChanged: R.filter(r => r.flipWav[k] !== r.flipMp3[k]).length })),
      features: Object.fromEntries(Object.keys(R[0].feat).map(k => { const d = R.map(r => r.feat[k]).filter(([a, b]) => a != null && b != null).map(([a, b]) => Math.abs(b - a) / (Math.abs(a) || 1));
        return [k, { n: d.length, medianRelChange: +(100 * med(d)).toFixed(1) }] })) };
    console.log(id, R.length, 'recordings;', out[id].perClass.map(c => `${c.class}: median |Δp| ${c.medianAbsShift} pts, 95th ${c.p95AbsShift}, flag changed ${c.flagChanged}`).join(' | '));
  }
  const MF = path.join(WEB, 'data', 'models.json'), M = JSON.parse(fs.readFileSync(MF));
  M.robustness = { generated: new Date().toISOString(), what: 'Same recordings analysed from the site’s 8 kHz 32 kbps MP3 copy vs the original WAV the models were trained on', models: out };
  fs.writeFileSync(MF, JSON.stringify(M));
  console.log('-> web/data/models.json [robustness]   (then merge.py --bump)');
})();
