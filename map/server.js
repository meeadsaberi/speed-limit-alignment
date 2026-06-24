// Minimal static file server for the Speed Safety Atlas map (preview-friendly).
const http = require('http'), fs = require('fs'), path = require('path');
const ROOT = __dirname, PORT = process.env.PORT || 8731;
const MIME = { '.html':'text/html', '.js':'text/javascript', '.css':'text/css',
  '.json':'application/json', '.geojson':'application/geo+json', '.png':'image/png' };
http.createServer((req, res) => {
  let p = decodeURIComponent(req.url.split('?')[0]);
  if (p === '/') p = '/index.html';
  const fp = path.join(ROOT, path.normalize(p));
  if (!fp.startsWith(ROOT)) { res.writeHead(403); return res.end('forbidden'); }
  fs.readFile(fp, (err, data) => {
    if (err) { res.writeHead(404); return res.end('not found'); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(fp)] || 'application/octet-stream' });
    res.end(data);
  });
}).listen(PORT, () => console.log(`atlas-map on http://localhost:${PORT}`));
