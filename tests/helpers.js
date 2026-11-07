// Shared test helpers: a static server for web/ and a Chromium page whose CDN libraries
// (wavesurfer, Meyda, Chart.js) are served from node_modules, so tests run offline and pinned.
const { chromium } = require('playwright'); const fs = require('fs'); const path = require('path'); const http = require('http');
const ROOT = path.resolve(__dirname, '..'); const WEB = path.join(ROOT, 'web'); const NM = path.join(ROOT, 'node_modules');
const MIME = { '.html': 'text/html', '.js': 'application/javascript', '.json': 'application/json', '.png': 'image/png', '.wav': 'audio/wav' };
function serve(port) {
  return new Promise(res => { const s = http.createServer((q, r) => { const f = path.join(WEB, decodeURIComponent(q.url.split('?')[0].split('#')[0]) || '/index.html');
    const file = fs.existsSync(f) && fs.statSync(f).isDirectory() ? path.join(f, 'index.html') : f;
    if (!file.startsWith(WEB) || !fs.existsSync(file)) { r.writeHead(404); r.end(); return }
    r.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' }); fs.createReadStream(file).pipe(r) }).listen(port, () => res(s)) });
}
async function open({ port = 8770, offline = true, viewport = { width: 1360, height: 1000 } } = {}) {
  const server = await serve(port);
  const browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] }); const ctx = await browser.newContext({ viewport });
  if (offline) {
    await ctx.route('https://unpkg.com/**', r => { const m = r.request().url().match(/wavesurfer\.js@[^/]+\/(.+)$/); r.fulfill({ status: 200, contentType: 'application/javascript', body: fs.readFileSync(path.join(NM, 'wavesurfer.js', m[1])) }) });
    await ctx.route('https://cdn.jsdelivr.net/**', r => { const m = r.request().url().match(/npm\/([^@]+)@[^/]+\/(.+)$/); r.fulfill({ status: 200, contentType: 'application/javascript', body: fs.readFileSync(path.join(NM, m[1], m[2])) }) });
    await ctx.route('https://fonts.googleapis.com/**', r => r.abort());
  }
  const page = await ctx.newPage(); const errors = []; page.on('pageerror', e => errors.push(e.message));
  const url = h => `http://localhost:${port}/index.html${h || ''}`;
  const close = async () => { await browser.close(); server.close() };
  return { page, errors, url, close };
}
const liveReady = pg => pg.waitForFunction(() => window.__live && window.__live() && window.__live().ok, null, { timeout: 60000 });
const check = (cond, msg) => { if (!cond) { console.error('FAIL', msg); process.exitCode = 1 } else console.log('ok  ', msg) };
module.exports = { open, liveReady, check, ROOT };
