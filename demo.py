"""宿说 Web Demo — 多算法实时对比 (v2: +dyx BM25/多路融合/意图检测).

    python demo.py              # → http://localhost:8080
"""

import sys, os, json, webbrowser, argparse, threading, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

_PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJ, "scripts"))
from _shared.retriever import retrieve
from _shared.cache import cache
from dyx.api import intent_classify

_api = MultimodalAPI()
IMAGE_DIR = os.path.join(_PROJ, "data", "images")

HTML = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>宿说 Demo v2</title>
<style>
:root{--brown:#8b4513;--light:#faf8f5;--bg:#f5f0eb;--card:#fff}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,"Microsoft YaHei",sans-serif;background:var(--bg);color:#333;min-height:100vh}
header{background:linear-gradient(135deg,#6b3410,#8b4513,#a0522d);color:#fff;padding:20px 28px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}
header h1{font-size:26px;letter-spacing:2px}header .sub{opacity:.7;font-size:13px}
main{max-width:1400px;margin:0 auto;padding:16px}
.cli-bar{background:#1e1e1e;color:#0f0;font-family:Consolas,monospace;padding:12px 18px;border-radius:10px 10px 0 0;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.cli-bar .prompt{color:#0f0;white-space:nowrap}
.cli-bar input{flex:1;min-width:200px;background:transparent;border:none;color:#0f0;font:inherit;outline:none;caret-color:#0f0}
.cli-bar input::placeholder{color:#0a0}
.cli-bar button{background:#333;color:#0f0;border:1px solid #0f0;padding:6px 14px;border-radius:6px;cursor:pointer;font:inherit;font-size:13px}
.cli-bar button:hover{background:#0f0;color:#000}
.methods{display:flex;gap:2px;background:#1e1e1e;padding:0 18px 8px;flex-wrap:wrap}
.methods label{color:#888;font-size:13px;cursor:pointer;padding:5px 12px;border-radius:6px;background:#2a2a2a;white-space:nowrap}
.methods label.active{background:var(--brown);color:#fff}
.methods input{display:none}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:0;border-radius:0 0 10px 10px;overflow:hidden}
@media(max-width:800px){.panels{grid-template-columns:1fr}}
.panel{background:var(--card);padding:14px;max-height:500px;overflow-y:auto}
.panel:nth-child(odd){border-right:1px solid #eee}
.panel h3{font-size:14px;color:var(--brown);margin-bottom:10px;padding-bottom:6px;border-bottom:2px solid var(--brown)}
.panel h3 .badge{float:right;font-size:11px;background:var(--brown);color:#fff;padding:1px 8px;border-radius:8px;font-weight:400}
.img-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px}
.img-card{background:var(--light);border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.06);cursor:pointer}
.img-card img{width:100%;height:100px;object-fit:cover}
.img-card .meta{padding:5px 8px;font-size:12px}
.img-card .score{color:var(--brown);font-weight:700}
.img-card .rank{float:right;color:#aaa}
.text-item{padding:8px;background:var(--light);border-radius:8px;margin-bottom:4px;font-size:12px;box-shadow:0 1px 3px rgba(0,0,0,.04);line-height:1.5}
.text-item .score{color:var(--brown);font-weight:700;margin-right:6px}
.text-item .method{float:right;font-size:10px;color:#999;background:#eee;padding:0 6px;border-radius:4px}
.empty{text-align:center;color:#ccc;padding:40px 20px;font-size:14px}
.bar-wrap{height:6px;background:#eee;border-radius:3px;margin-top:4px}
.bar-fill{height:100%;background:var(--brown);border-radius:3px;transition:width .3s}
.stats-row{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px}
.stat{flex:1;min-width:80px;background:var(--card);padding:14px;border-radius:10px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.04)}
.stat .n{font-size:24px;font-weight:700;color:var(--brown)}.stat .l{font-size:11px;color:#999;margin-top:3px}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
@media(max-width:800px){.two-col{grid-template-columns:1fr}}
.card-box{background:var(--card);border-radius:10px;padding:14px;box-shadow:0 1px 4px rgba(0,0,0,.04)}
footer{text-align:center;color:#ccc;font-size:12px;padding:20px}
.examples{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.examples span{cursor:pointer;font-size:12px;color:var(--brown);text-decoration:underline}
.intent-box{padding:10px;background:var(--light);border-radius:8px;margin-top:8px;font-size:13px}
.intent-box .tag{display:inline-block;background:var(--brown);color:#fff;padding:1px 8px;border-radius:8px;font-size:11px;margin-right:4px}
.latency{font-size:11px;color:#999;text-align:right;margin-top:4px}
</style></head><body>

<header>
  <div><h1>宿说 v2</h1><div class="sub">酒店评论智能检索 — 多模态 + BM25 + 多路融合</div></div>
  <div class="stats-row" id="header-stats"></div>
</header>

<main>
<div class="cli-bar">
  <span class="prompt">$ hotel search</span>
  <input id="query" placeholder="输入查询文本..." value="游泳池干净吗" autofocus>
  <button onclick="search()">检索 (Enter)</button>
</div>
<div class="methods" id="method-tabs">
  <label class="active"><input type="radio" name="m" value="clip" checked onchange="search()">CLIP 基础</label>
  <label><input type="radio" name="m" value="granular" onchange="search()">多粒度</label>
  <label><input type="radio" name="m" value="rerank" onchange="search()">LTR 重排序</label>
  <label><input type="radio" name="m" value="bm25" onchange="search()">BM25 关键词 <span style="color:#ff6">★</span></label>
  <label><input type="radio" name="m" value="fusion" onchange="search()">多路融合 <span style="color:#ff6">★</span></label>
</div>
<div class="examples">
  试试: <span onclick="q('房间很好 装修很厚重奢华')">装修</span>
  <span onclick="q('早餐很丰盛 花园很漂亮')">早餐花园</span>
  <span onclick="q('服务态度非常好 前台热情')">服务</span>
  <span onclick="q('大床房 干净整洁')">大床房</span>
  <span onclick="q('适合带孩子吗')">亲子</span>
  <span onclick="q('隔音好不好')">噪音</span>
</div>

<div class="panels" id="main-panels">
  <div class="panel" id="panel-text">
    <h3 id="panel-title">文本 → 图片 <span class="badge">CLIP</span></h3>
    <div class="img-grid" id="result-images"></div>
    <div class="latency" id="result-latency"></div>
    <div class="empty" id="empty-images">输入查询，选择算法，查看结果</div>
  </div>
  <div class="panel" id="panel-meta">
    <h3>意图分析 <span class="badge">dyx</span></h3>
    <div id="result-intent"><div class="empty">检索后将显示意图分类</div></div>
    <h3 style="margin-top:12px">详情</h3>
    <div id="result-meta"></div>
    <div class="empty" id="empty-meta">选中图片查看详情</div>
  </div>
</div>

<div class="two-col">
  <div class="card-box">
    <h3 style="font-size:14px;color:var(--brown);margin-bottom:8px">反向检索 (图片→文本)</h3>
    <div style="display:flex;gap:6px">
      <input id="rev-id" placeholder="图片 ID (0-1200)" value="0" style="flex:1;padding:6px;border:1px solid #ddd;border-radius:6px;font-size:13px">
      <button onclick="reverse()" style="padding:6px 14px;background:var(--brown);color:#fff;border:none;border-radius:6px;cursor:pointer">检索</button>
    </div>
    <div id="result-reverse" style="margin-top:8px"><div class="empty">输入 ID 检索</div></div>
  </div>
  <div class="card-box">
    <h3 style="font-size:14px;color:var(--brown);margin-bottom:8px">房型分类器</h3>
    <div style="display:flex;gap:6px">
      <input id="cls-text" placeholder="输入评论文本..." value="房间很大 床很舒服" onkeydown="if(event.key==='Enter')classify()" style="flex:1;padding:6px;border:1px solid #ddd;border-radius:6px;font-size:13px">
      <button onclick="classify()" style="padding:6px 14px;background:var(--brown);color:#fff;border:none;border-radius:6px;cursor:pointer">分类</button>
    </div>
    <div id="result-classify" style="margin-top:8px"><div class="empty">输入文本分类</div></div>
  </div>
</div>

<footer>宿说 — 大模型技术原理与商业应用 · Spring 2026 · Fudan</footer>

<script>
const q=(v)=>{document.getElementById("query").value=v;search()};
document.getElementById("query").addEventListener("keydown",e=>{if(e.key==="Enter")search()});

let curMethod="clip";
document.querySelectorAll(".methods input").forEach(r=>{
  r.addEventListener("change",()=>{curMethod=r.value});
});

async function search(){
  const q=document.getElementById("query").value;
  if(!q)return;
  
  if(curMethod==="bm25"){
    await searchBM25(q);
  }else if(curMethod==="fusion"){
    await searchFusion(q);
  }else{
    await searchCLIP(q);
  }
  await loadIntent(q);
}

async function searchCLIP(q){
  const r=await fetch("/api/search?q="+encodeURIComponent(q)+"&method="+curMethod+"&top=12");
  const d=await r.json();
  if(d.error){document.getElementById("result-images").innerHTML='<div class="empty">'+d.error+'</div>';return}
  
  const title=d.method?d.method:"CLIP";
  document.getElementById("panel-title").innerHTML="文本 → 图片 <span class='badge'>"+title+"</span>";
  document.getElementById("empty-images").style.display="none";
  
  let h="";
  (d.images||[]).forEach((img,i)=>{
    h+='<div class="img-card" onclick="showMeta('+i+',\''+img+'\')"><img src="/image?path='+encodeURIComponent(img)+'"><div class="meta"><span class="score">'+(d.score[i]||0).toFixed(4)+'</span><span class="rank">#'+(i+1)+'</span></div></div>'
  });
  document.getElementById("result-images").innerHTML=h||'<div class="empty">无结果</div>';
  document.getElementById("result-latency").textContent=d.latency_ms?"延迟: "+d.latency_ms+"ms":"";
  window._lastResult=d;
}

async function searchBM25(q){
  const r=await fetch("/api/bm25?q="+encodeURIComponent(q)+"&top=10");
  const d=await r.json();
  if(d.error){document.getElementById("result-images").innerHTML='<div class="empty">'+d.error+'</div>';return}
  
  document.getElementById("panel-title").innerHTML="BM25 关键词检索 <span class='badge'>jieba+BM25</span>";
  document.getElementById("empty-images").style.display="none";
  
  let h="";
  (d.results||[]).forEach((item,i)=>{
    h+='<div class="text-item"><span class="score">'+item.score.toFixed(4)+'</span>'
      +(item.comment||'')+'<span class="method">#'+(i+1)+'</span></div>';
  });
  document.getElementById("result-images").innerHTML=h||'<div class="empty">无结果</div>';
  document.getElementById("result-latency").textContent="延迟: "+(d.latency_ms||0)+"ms | 索引: "+d.total_docs+" 篇评论";
}

async function searchFusion(q){
  const r=await fetch("/api/fusion?q="+encodeURIComponent(q)+"&top=10");
  const d=await r.json();
  if(d.error){document.getElementById("result-images").innerHTML='<div class="empty">'+d.error+'</div>';return}
  
  document.getElementById("panel-title").innerHTML="多路融合 (RRF) <span class='badge'>BM25+向量</span>";
  document.getElementById("empty-images").style.display="none";
  
  // Show RRF fused results
  let h='<div style="font-size:12px;color:#666;margin-bottom:6px">融合 BM25 + CLIP 向量 共 '+(d.fused_results?.length||0)+' 条</div>';
  (d.fused_results||[]).forEach((item,i)=>{
    h+='<div class="text-item"><span class="score">'+(item.fused_score||0).toFixed(4)+'</span>'
      +(item.comment||'')+'<span class="method">#'+(i+1)+'</span></div>';
  });
  // Show source breakdown
  h+='<div style="margin-top:8px;font-size:11px;color:#999">BM25: '+(d.bm25_results?.length||0)+' 条 | 向量: '+(d.vector_results?.length||0)+' 条</div>';
  
  document.getElementById("result-images").innerHTML=h||'<div class="empty">无结果</div>';
  document.getElementById("result-latency").textContent="延迟: "+(d.latency_ms||0)+"ms";
}

async function loadIntent(q){
  const r=await fetch("/api/intent?q="+encodeURIComponent(q));
  const d=await r.json();
  let h='<div class="intent-box"><b>意图</b>: <span class="tag">'+d.primary+'</span>';
  if(d.categories&&d.categories.length){
    h+=' | <b>维度</b>: '+d.categories.map(c=>'<span class="tag">'+c+'</span>').join(" ");
  }
  h+=' | <b>置信度</b>: '+(d.confidence*100).toFixed(0)+'%';
  h+='</div>';
  document.getElementById("result-intent").innerHTML=h;
}

function showMeta(i,img){
  const d=window._lastResult;if(!d)return;
  document.getElementById("empty-meta").style.display="none";
  document.getElementById("result-meta").innerHTML='<div style="font-size:13px"><b>#'+(i+1)+'</b> | score: <b style="color:var(--brown)">'+(d.score[i]||0).toFixed(4)+'</b></div><img src="/image?path='+encodeURIComponent(img)+'" style="width:100%;max-height:300px;object-fit:cover;border-radius:8px;margin:8px 0">';
}

async function reverse(){
  const id=document.getElementById("rev-id").value;
  const div=document.getElementById("result-reverse");
  div.innerHTML='<div class="empty">...</div>';
  const r=await fetch("/api/reverse?img="+id+"&top=5");const d=await r.json();
  let h='<img src="/image?path=data/images/img_'+id+'.jpg" style="width:100%;max-height:160px;object-fit:cover;border-radius:8px;margin-bottom:8px">';
  (d.texts||[]).forEach((t,i)=>{h+='<div class="text-item"><span class="score">'+(d.score[i]||0).toFixed(4)+'</span>'+t.substring(0,150)+'...</div>'});
  div.innerHTML=h;
}

async function classify(){
  const t=document.getElementById("cls-text").value;
  const r=await fetch("/api/classify?text="+encodeURIComponent(t));const d=await r.json();
  let h='<div style="font-size:18px;font-weight:700;color:var(--brown);margin-bottom:8px">'+(d.room_type||"?")+" ("+((d.confidence||0)*100).toFixed(0)+"%)</div>";
  for(const[room,p]of Object.entries(d.all_probs||{})){
    h+='<div style="margin:3px 0;font-size:12px;display:flex;align-items:center;gap:8px"><span style="width:60px">'+room+'</span><div class="bar-wrap" style="flex:1"><div class="bar-fill" style="width:'+(p*100)+'%"></div></div><span style="width:35px;text-align:right">'+(p*100).toFixed(0)+"%</span></div>";
  }
  document.getElementById("result-classify").innerHTML=h;
}

async function loadStats(){
  const r=await fetch("/api/stats");const d=await r.json();
  document.getElementById("header-stats").innerHTML=
    '<div class="stat" style="background:rgba(255,255,255,.1);min-width:60px"><div class="n" style="color:#fff">'+(d.images||0)+'</div><div class="l" style="color:rgba(255,255,255,.6)">图片</div></div>'+
    '<div class="stat" style="background:rgba(255,255,255,.1);min-width:60px"><div class="n" style="color:#fff">'+(d.texts||0)+'</div><div class="l" style="color:rgba(255,255,255,.6)">文本</div></div>'+
    '<div class="stat" style="background:rgba(255,255,255,.1);min-width:60px"><div class="n" style="color:#fff">'+(d.documents||0)+'</div><div class="l" style="color:rgba(255,255,255,.6)">文档索引</div></div>';
}
loadStats();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        p = urlparse(self.path)
        qs = parse_qs(p.query)

        if p.path in ("/", "/index.html"):
            self._html(HTML)

        elif p.path == "/api/search":
            q = qs.get("q", [""])[0]
            m = qs.get("method", ["clip"])[0]
            top = int(qs.get("top", ["10"])[0])
            try:
                r = ({"rerank": get_api().prompt2image_rerank,
                      "filter": get_api().prompt2image_filtered,
                      "granular": lambda q, t: get_api().prompt2image_granular(q, topK=t, granularity="room_type")}
                     .get(m, get_api().prompt2image))(q, topK=top)
                r["latency_ms"] = 0
                self._json(r)
            except Exception as e:
                self._json({"error": str(e)}, 500)

        # [dyx] BM25 关键词检索
        elif p.path == "/api/bm25":
            q = qs.get("q", [""])[0]
            top = int(qs.get("top", ["10"])[0])
            try:
                r = bm25_search(q, top_k=top)
                self._json(r)
            except Exception as e:
                self._json({"error": str(e)}, 500)

        # [dyx] 多路融合检索
        elif p.path == "/api/fusion":
            q = qs.get("q", [""])[0]
            top = int(qs.get("top", ["10"])[0])
            try:
                r = multi_path_search(q, top_k=top, use_vector=True)
                # Shim for API compatibility
                r["results"] = r["fused_results"]
                self._json(r)
            except Exception as e:
                self._json({"error": str(e)}, 500)

        # [dyx] 意图分类
        elif p.path == "/api/intent":
            q = qs.get("q", [""])[0]
            try:
                r = intent_classify(q)
                self._json(r)
            except Exception as e:
                self._json({"error": str(e)}, 500)

        elif p.path == "/api/reverse":
            iid = qs.get("img", ["0"])[0]
            top = int(qs.get("top", ["5"])[0])
            try:
                self._json(get_api().image2text(
                    os.path.join(_PROJ, f"data/images/img_{iid}.jpg"), topK=top))
            except Exception as e:
                self._json({"error": str(e)}, 500)

        elif p.path == "/api/classify":
            try:
                self._json(get_api().classify_room_type(qs.get("text", [""])[0]))
            except Exception as e:
                self._json({"error": str(e)}, 500)

        elif p.path == "/api/stats":
            try:
                s = get_api().stats
                s["documents"] = "BM25"
                self._json(s)
            except Exception as e:
                self._json({"error": str(e)}, 500)

        elif p.path == "/image":
            fp = os.path.join(_PROJ, qs.get("path", [""])[0])
            if os.path.exists(fp):
                with open(fp, "rb") as f:
                    d = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", len(d))
                self.end_headers()
                self.wfile.write(d)
            else:
                self._json({"error": "not found"}, 404)
        else:
            self._json({"error": "not found"}, 404)

    def _html(self, c):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(c.encode("utf-8"))

    def _json(self, d, s=200):
        self.send_response(s)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(d, ensure_ascii=False).encode("utf-8"))

    def log_message(self, f, *a):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="宿说 Web Demo v2")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    url = f"http://localhost:{args.port}"

    def _open():
        time.sleep(0.5)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()
    print(f"\n  宿说 v2 Demo → {url}  (Ctrl+C 退出)\n")
    print(f"  新增 API:")
    print(f"    /api/bm25     BM25 关键词检索 (jieba 分词)")
    print(f"    /api/fusion   多路融合 (BM25 + CLIP 向量 + RRF)")
    print(f"    /api/intent   细粒度意图分类 (dyx 算法)")
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
