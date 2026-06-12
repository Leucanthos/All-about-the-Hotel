"""单元测试 — 意图分类 + BM25 + RAG + 融合 + 路由 (30条检查点).
"""
import sys, os, json, time, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

OUTPUT_DIR = os.path.join("tests", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
QUERIES_PATH = os.path.join("tests", "data", "queries_unit.json")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "test_results.json")

# ── 1. 构建测试查询集 ──

TEST_QUERIES = [
    {"id": "intent_facility",     "query": "游泳池干净吗",         "expected_intent": "facility"},
    {"id": "intent_food",         "query": "早餐好吃吗",           "expected_intent": "food"},
    {"id": "intent_price",        "query": "多少钱一晚",           "expected_intent": "price"},
    {"id": "intent_location",     "query": "交通方便吗",           "expected_intent": "location"},
    {"id": "intent_service",      "query": "前台服务态度怎么样",   "expected_intent": "service"},
    {"id": "intent_cleanliness",  "query": "房间卫生干净吗",       "expected_intent": "cleanliness"},
    {"id": "intent_quiet",        "query": "隔音效果好不好",       "expected_intent": "quiet"},
    {"id": "intent_child",        "query": "适合带孩子住吗",       "expected_intent": "child_friendly"},
    {"id": "intent_elder",        "query": "适合老人入住吗",       "expected_intent": "service"},
    {"id": "intent_overall",      "query": "酒店体验怎么样",       "expected_intent": "overall"},
    {"id": "intent_mixed_1",      "query": "早餐种类多吗 游泳池怎么样",   "expected_intent": "facility"},
    {"id": "intent_mixed_2",      "query": "房间装修豪华吗 服务态度好不", "expected_intent": "facility"},
    {"id": "bm25_breakfast",      "query": "早餐 品种 丰富",       "expected_bm25_hit": True},
    {"id": "bm25_pool",           "query": "游泳池 干净",          "expected_bm25_hit": True},
    {"id": "bm25_parking",        "query": "停车场 方便",          "expected_bm25_hit": True},
    {"id": "bm25_noise",          "query": "隔音 噪音 吵",         "expected_bm25_hit": True},
    {"id": "rag_service",         "query": "前台服务怎么样？有没有常见问题？",       "expected_rag_evidence": True},
    {"id": "rag_breakfast",       "query": "早餐有什么推荐？品质如何？",             "expected_rag_evidence": True},
    {"id": "rag_family",          "query": "带老人小孩入住合适吗？有什么需要注意的？", "expected_rag_evidence": True},
    {"id": "fusion_room",         "query": "套房 豪华装修",        "expected_fusion_multi": True},
    {"id": "fusion_garden",       "query": "花园 漂亮 环境",       "expected_fusion_multi": True},
    {"id": "edge_short",          "query": "早餐",                 "expected_intent": "food"},
    {"id": "edge_empty",          "query": "",                     "expected_intent": "mixed"},
    {"id": "edge_special_chars",  "query": "房间！！干净？？",      "expected_intent": "facility"},
    {"id": "edge_english",        "query": "breakfast quality",    "expected_intent": "mixed"},
]

json.dump(TEST_QUERIES, open(QUERIES_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"Test queries saved: {len(TEST_QUERIES)} cases -> {QUERIES_PATH}")

# ── 2. 测试函数 ──

passed = 0; failed = 0; errors = []
def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1; print(f"  PASS  {name}")
    else:
        failed += 1; msg = f"FAIL  {name}  {detail}"; print(f"  {msg}"); errors.append(msg)

print("\n" + "=" * 60)
print("TEST SUITE: 宿说 — 单元测试 (30条)")
print("=" * 60)

# 3.1 意图分类
print("\n--- Intent Classification (dyx) ---")
from _shared.router import classify_intent_detailed
for tc in TEST_QUERIES:
    if "expected_intent" not in tc: continue
    q = tc["query"]
    if not q:
        r = classify_intent_detailed(q)
        check(f"[{tc['id']}] empty query -> mixed", r["primary"] == "mixed"); continue
    r = classify_intent_detailed(q)
    check(f"[{tc['id']}] '{q[:20]}' -> {tc['expected_intent']}", r["primary"] == tc["expected_intent"], f"got {r['primary']}")

# 3.2 BM25
print("\n--- BM25 Search (dyx) ---")
from _shared.cache import cache
index = cache.get_bm25()
for tc in TEST_QUERIES:
    if "expected_bm25_hit" not in tc: continue
    q = tc["query"]; t0 = time.time()
    res = index.search(q, topk=10)
    check(f"[{tc['id']}] '{q}' -> {len(res)} results in {(time.time()-t0)*1000:.0f}ms", len(res) > 0)

# 3.3 RAG
print("\n--- RAG Pipeline (Yuhao) ---")
from yuhao.agentic.engine import HotelCorpus, AgenticRAG
corpus = HotelCorpus(max_docs=500); rag = AgenticRAG(corpus)
for tc in TEST_QUERIES:
    if "expected_rag_evidence" not in tc: continue
    q = tc["query"]; t0 = time.time()
    r = rag.run(q, top_k=5)
    has_ev = r["evidence_score"] > 0; has_cat = len(r.get("categories", [])) > 0
    check(f"[{tc['id']}] evidence={r['evidence_score']:.2f} cats={r.get('categories',[])}", has_ev and has_cat)

# 3.4 Fusion
print("\n--- Fusion (dyx) ---")
from _shared.fusion import rrf_fuse
for tc in TEST_QUERIES:
    if "expected_fusion_multi" not in tc: continue
    q = tc["query"]
    bm25_raw = index.search(q, topk=10)
    bm25_items = [{"rank": i+1, "method": "bm25", "doc_id": doc_id, "score": round(s,4)} for i,(doc_id,s) in enumerate(bm25_raw)]
    fused = rrf_fuse([bm25_items, bm25_items], topk=5)
    check(f"[{tc['id']}] '{q}' -> {len(fused)} fused", len(fused) > 0 and len(bm25_items) > 0)

# 3.5 路由
print("\n--- Router / Retriever Integration ---")
from _shared.retriever import retrieve
r = retrieve("早餐", strategies=["bm25"], top_k=3)
check("retriever(bm25) has results", "bm25" in r.get("results", {}))
r2 = retrieve("游泳池", top_k=3)
check("retriever(auto) detects intent", len(r2.get("strategies", [])) > 0)
from _shared.router import route
r3 = route("早餐好吃吗", strategies=["bm25"], top_k=3)
check("route() returns query", r3.get("query") == "早餐好吃吗")

# 3.6 性能
print("\n--- Performance Benchmarks ---")
import pandas as pd
df = cache.get_comments()
t0 = time.time()
from _shared.bm25 import InvertedIndex
bench_idx = InvertedIndex()
documents = {}
for i, row in df.iterrows():
    comment = str(row.get("comment", "")).strip()
    if comment: documents[str(row.get("_id", i))] = comment
    if len(documents) >= 2000: break
bench_idx.build(documents)
bt = time.time() - t0
qt = []
for _ in range(10):
    t0 = time.time(); bench_idx.search("早餐 服务 房间", topk=10); qt.append((time.time()-t0)*1000)
check(f"BM25 build 2000 docs < 2s", bt < 2.0, f"took {bt:.2f}s")
check(f"BM25 query avg < 5ms", sum(qt)/len(qt) < 5.0, f"avg={sum(qt)/len(qt):.2f}ms")

# ── 汇总 ──
print(f"\n{'='*60}")
print(f"RESULTS: {passed}/{passed+failed} passed ({passed/(passed+failed)*100:.0f}%)")
if errors:
    for e in errors: print(f"  {e}")
print(f"{'='*60}")
json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "total": passed+failed, "passed": passed, "failed": failed, "errors": errors},
          open(RESULTS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"Results saved: {RESULTS_PATH}")
sys.exit(0 if failed == 0 else 1)
