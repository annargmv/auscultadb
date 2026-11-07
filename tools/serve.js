// Serve web/ at http://localhost:8000 (PORT env to change). The page needs http(s), not file://, for fetch and Cache Storage.
const http = require('http'), fs = require('fs'), path = require('path');
const WEB = path.resolve(__dirname, '..', 'web'), PORT = +(process.env.PORT || 8000);
const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'application/javascript', '.json': 'application/json', '.png': 'image/png' };
http.createServer((q, r) => {
  let f = path.join(WEB, decodeURIComponent(q.url.split('?')[0]));
  if (fs.existsSync(f) && fs.statSync(f).isDirectory()) f = path.join(f, 'index.html');
  if (!f.startsWith(WEB) || !fs.existsSync(f)) { r.writeHead(404); r.end('not found'); return }
  r.writeHead(200, { 'Content-Type': MIME[path.extname(f)] || 'application/octet-stream' }); fs.createReadStream(f).pipe(r);
}).listen(PORT, () => console.log(`AuscultaDB on http://localhost:${PORT}`));
