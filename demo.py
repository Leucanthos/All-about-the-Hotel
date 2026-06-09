"""
宿说 Demo — 可视化交互展示多模态检索接口.

    python demo.py              # → http://localhost:8080
"""

import sys, os, json, webbrowser, argparse, threading, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

_project = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_project, "vibecoding", "src"))
from multimodal_api import MultimodalAPI

api = MultimodalAPI()
IMAGE_DIR = os.path.join(_project, "data", "images")

HTML = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>宿说 Demo</title>
<style>
:root{--brown:#8b4513;--light:#faf8f5;--bg:#f5f0eb;--card:#fff}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,"Microsoft YaHei",sans-serif;background:var(--bg);color:#333;min-height:100vh}
header{background:linear-gradient(135deg,#6b3410,#8b4513,#a0522d);color:#fff;padding:20px 28px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}
header h1{font-size:26px;letter-spacing:2px}header .sub{opacity:.7;font-size:13px}
main{max-width:1300px;margin:0 auto;padding:16px}
.cli-bar{background:#1e1e1e;color:#0f0;font-family:Consolas,monospace;padding:12px 18px;border-radius:10px 10px 0 0;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.cli-bar .prompt{color:#0f0;white-space:nowrap}
.cli-bar input{flex:1;min-width:200px;background:transparent;border:none;color:#0f0;font:inherit;outline:none;caret-color:#0f0}
.cli-bar input::placeholder{color:#0a0}
.cli-bar button{background:#333;color:#0f0;border:1px solid #0f0;padding:6px 14px;border-radius:6px;cursor:pointer;font:inherit;font-size:13px}
.cli-bar button:hover{background:#0f0;color:#000}
.methods{display:flex;gap:2px;background:#1e1e1e;padding:0 18px 8px}
.methods label{color:#888;font-size:13px;cursor:pointer;padding:5px 12px;border-radius:6px;background:#2a2a2a;white-space:nowrap}
.methods label.active{background:var(--brown);color:#fff}
.methods input{display:none}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:0;border-radius:0 0 10px 10px;overflow:hidden}
@media(max-width:800px){.panels{grid-template-columns:1fr}}
.panel{background:var(--card);padding:14px;max-height:500px;overflow-y:auto}
.panel:nth-child(odd){border-right:1px solid #eee}
.panel h3{font-size:14px;color:var(--brown);margin-bottom:10px;padding-bottom:6px;border-bottom:2px solid var(--brown)}
.img-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px}
.img-card{background:var(--light);border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.img-card img{width:100%;height:110px;object-fit:cover}
.img-card .meta{padding:6px 8px;font-size:12px}
.img-card .score{color:var(--brown);font-weight:700}
.img-card .rank{float:right;color:#aaa}
.text-item{padding:10px;background:var(--light);border-radius:8px;margin-bottom:6px;font-size:13px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.text-item .score{color:var(--brown);font-weight:700;margin-right:8px}
.empty{text-align:center;color:#ccc;padding:60px 20px;font-size:15px}
.bar-wrap{height:6px;background:#eee;border-radius:3px;margin-top:4px}
.bar-fill{height:100%;background:var(--brown);border-radius:3px;transition:width .3s}
.stats-row{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px}
.stat{flex:1;min-width:100px;background:var(--card);padding:16px;border-radius:10px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.04)}
.stat .n{font-size:28px;font-weight:700;color:var(--brown)}.stat .l{font-size:11px;color:#999;margin-top:3px}
footer{text-align:center;color:#ccc;font-size:12px;padding:20px}
.examples{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.examples span{cursor:pointer;font-size:12px;color:var(--brown);text-decoration:underline}
</style></head><body>

<header>
  <div><h1>宿说</h1><div class="sub">酒店评论多模态检索 — 4 种算法实时对比</div></div>
  <div class="stats-row" id="header-stats"></div>
</header>

<main>
<div class="cli-bar">
  <span class="prompt">$ 宿说 search</span>
  <input id="query" placeholder="输入查询文本..." value="游泳池干净吗" autofocus>
  <button onclick="search()">检索 (Enter)</button>
</div>
<div class="methods" id="method-tabs">
  <label class="active"><input type="radio" name="m" value="clip" checked onchange="search()">CLIP 基础</label>
  <label><input type="radio" name="m" value="granular" onchange="search()">多粒度</label>
  <label><input type="radio" name="m" value="rerank" onchange="search()">LTR 重排序</label>
  <label><input type="radio" name="m" value="filter" onchange="search()">分类器预过滤</label>
</div>
<div class="examples">
  试试: <span onclick="q('房间很好 装修很厚重奢华')">装修</span>
  <span onclick="q('早餐很丰盛 花园很漂亮')">早餐花园</span>
  <span onclick="q('服务态度非常好 前台热情')">服务</span>
  <span onclick="q('大床房 干净整洁')">大床房</span>
</div>

<div class="panels">
  <div class="panel" id="panel-text">
    <h3>📝 文本 → 图片 (prompt2image)</h3>
    <div class="img-grid" id="result-images"></div>
    <div class="empty" id="empty-images">输入查询，选择算法，查看结果</div>
  </div>
  <div class="panel" id="panel-meta">
    <h3>📊 详情</h3>
    <div id="result-meta"></div>
    <div class="empty" id="empty-meta">选中图片查看详情</div>
  </div>
</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px">
  <div style="background:var(--card);border-radius:10px;padding:14px">
    <h3 style="font-size:14px;color:var(--brown);margin-bottom:10px">🖼️ 图片 → 文本 (image2text)</h3>
    <div style="display:flex;gap:8px;margin-bottom:8px">
      <input id="rev-id" value="0" style="width:80px;padding:6px;border:1px solid #ddd;border-radius:6px;font-size:13px" placeholder="img id">
      <button onclick="reverse()" style="background:var(--brown);color:#fff;border:none;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:13px">检索</button>
    </div>
    <div id="result-reverse" style="max-height:300px;overflow-y:auto"></div>
  </div>
  <div style="background:var(--card);border-radius:10px;padding:14px">
    <h3 style="font-size:14px;color:var(--brown);margin-bottom:10px">🏷️ 房型分类 (classify_room_type)</h3>
    <div style="display:flex;gap:8px;margin-bottom:8px">
      <input id="cls-text" value="房间非常好 装修很厚重奢华 套房很舒适" style="flex:1;padding:6px;border:1px solid #ddd;border-radius:6px;font-size:13px">
      <button onclick="classify()" style="background:var(--brown);color:#fff;border:none;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:13px">分类</button>
    </div>
    <div id="result-classify"></div>
  </div>
</div>
</main>
<footer>宿说 · Chinese-CLIP + LTR + Classifier · Intel Arc B390 GPU</footer>

<script>
let curMethod='clip';
document.querySelectorAll('#method-tabs label').forEach(l=>l.addEventListener('click',function(){
  document.querySelectorAll('#method-tabs label').forEach(x=>x.classList.remove('active'));
  this.classList.add('active');curMethod=this.querySelector('input').value;search()}));
document.getElementById('query').addEventListener('keydown',e=>{if(e.key==='Enter')search()});
document.getElementById('rev-id').addEventListener('keydown',e=>{if(e.key==='Enter')reverse()});
document.getElementById('cls-text').addEventListener('keydown',e=>{if(e.key==='Enter')classify()});

function q(t){document.getElementById('query').value=t;search()}

async function search(){
  const q=document.getElementById('query').value;
  const r=await fetch('/api/search?q='+encodeURIComponent(q)+'&method='+curMethod+'&top=12');
  const d=await r.json();
  const div=document.getElementById('result-images');
  document.getElementById('empty-images').style.display='none';
  let h='';d.images.forEach((img,i)=>{h+=`<div class="img-card" onclick="showMeta(${i},'${img}')"><img src="/image?path=${encodeURIComponent(img)}"><div class="meta"><span class="score">${d.score[i].toFixed(4)}</span><span class="rank">#${i+1}</span></div></div>`});
  div.innerHTML=h||'<div class="empty">无结果</div>';
  document.getElementById('result-meta').innerHTML='<div class="empty">点击图片查看详情</div>';
  window._lastResult=d;
}

function showMeta(i,img){
  const d=window._lastResult;if(!d)return;
  document.getElementById('empty-meta').style.display='none';
  const div=document.getElementById('result-meta');
  div.innerHTML=`<div style="font-size:13px"><b>#${i+1}</b> | score: <b style="color:var(--brown)">${d.score[i].toFixed(4)}</b></div>
    <img src="/image?path=${encodeURIComponent(img)}" style="width:100%;max-height:300px;object-fit:cover;border-radius:8px;margin:8px 0">`;
}

async function reverse(){
  const id=document.getElementById('rev-id').value;
  const div=document.getElementById('result-reverse');
  div.innerHTML='<div class="empty">...</div>';
  const r=await fetch('/api/reverse?img='+id+'&top=5');const d=await r.json();
  let h=`<img src="/image?path=data/images/img_${id}.jpg" style="width:100%;max-height:180px;object-fit:cover;border-radius:8px;margin-bottom:10px">`;
  d.texts.forEach((t,i)=>{h+=`<div class="text-item"><span class="score">${d.score[i].toFixed(4)}</span>${t.substring(0,150)}...</div>`});
  div.innerHTML=h;
}

async function classify(){
  const t=document.getElementById('cls-text').value;
  const r=await fetch('/api/classify?text='+encodeURIComponent(t));const d=await r.json();
  let h=`<div style="font-size:18px;font-weight:700;color:var(--brown);margin-bottom:8px">${d.room_type||'?'} (${(d.confidence*100).toFixed(0)}%)</div>`;
  for(const[room,p]of Object.entries(d.all_probs||{})){
    h+=`<div style="margin:4px 0;font-size:12px;display:flex;align-items:center;gap:8px"><span style="width:50px">${room}</span><div class="bar-wrap" style="flex:1"><div class="bar-fill" style="width:${p*100}%"></div></div><span style="width:35px;text-align:right">${(p*100).toFixed(0)}%</span></div>`;
  }
  document.getElementById('result-classify').innerHTML=h;
}

async function loadStats(){
  const r=await fetch('/api/stats');const d=await r.json();
  document.getElementById('header-stats').innerHTML=`<div class="stat" style="background:rgba(255,255,255,.1);min-width:70px"><div class="n" style="color:#fff">${d.images}</div><div class="l" style="color:rgba(255,255,255,.6)">图片</div></div><div class="stat" style="background:rgba(255,255,255,.1);min-width:70px"><div class="n" style="color:#fff">${d.texts}</div><div class="l" style="color:rgba(255,255,255,.6)">文本</div></div>`;
}
loadStats();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        p = urlparse(self.path); qs = parse_qs(p.query)

        if p.path in ("/", "/index.html"):
            self._html(HTML)

        elif p.path == "/api/search":
            q, m, top = qs.get("q", [""])[0], qs.get("method", ["clip"])[0], int(qs.get("top", ["10"])[0])
            try:
                r = ({"rerank": api.prompt2image_rerank, "filter": api.prompt2image_filtered,
                      "granular": lambda q,t: api.prompt2image_granular(q,topK=t,granularity="room_type")}
                     .get(m, api.prompt2image)(q, topK=top))
                self._json(r)
            except Exception as e:
                self._json({"error": str(e)}, 500)

        elif p.path == "/api/reverse":
            iid, top = qs.get("img", ["0"])[0], int(qs.get("top", ["5"])[0])
            try:
                self._json(api.image2text(os.path.join(_project, f"data/images/img_{iid}.jpg"), topK=top))
            except Exception as e:
                self._json({"error": str(e)}, 500)

        elif p.path == "/api/classify":
            try:
                self._json(api.classify_room_type(qs.get("text", [""])[0]))
            except Exception as e:
                self._json({"error": str(e)}, 500)

        elif p.path == "/api/stats":
            self._json(api.stats)

        elif p.path == "/image":
            fp = os.path.join(_project, qs.get("path", [""])[0])
            if os.path.exists(fp):
                with open(fp, "rb") as f: d = f.read()
                self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", len(d)); self.end_headers(); self.wfile.write(d)
            else:
                self._json({"error": "not found"}, 404)
        else:
            self._json({"error": "not found"}, 404)

    def _html(self, c):
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
        self.wfile.write(c.encode("utf-8"))

    def _json(self, d, s=200):
        self.send_response(s); self.send_header("Content-Type", "application/json; charset=utf-8"); self.end_headers()
        self.wfile.write(json.dumps(d, ensure_ascii=False).encode("utf-8"))

    def log_message(self, f, *a): pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    url = f"http://localhost:{args.port}"

    def _open():
        time.sleep(0.5); webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()

    print(f"\n  宿说 Demo → {url}\n")
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
