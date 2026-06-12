"""多酒店扩展测试 — 从 296MB comments.csv 抽样 12 家酒店.

用法: python tests/run_extended_tests.py
"""

import sys, os, json, time, csv, random, math
from collections import Counter, defaultdict

random.seed(42)

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "scripts"))
os.chdir(PROJ)

OUTPUT_DIR = os.path.join(PROJ, "tests")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(PROJ, "comments.csv")
SAMPLES_PER_HOTEL = 200

SELECTED_HOTELS = [
    "北京索菲特大酒店", "上海浦东香格里拉大酒店", "上海佘山茂御臻品之选酒店",
    "北京世纪金源大饭店", "北京大兴国际机场木棉花酒店", "北京三里屯通盈中心洲际酒店",
    "北京远航国际酒店（首都机场新国展店）", "朗丽兹太和府酒店（北京生命科学园地铁站店）",
    "汉庭酒店(北京天坛店)", "桔子水晶北京南锣鼓巷酒店",
    "北京王府井文华东方酒店", "北京万达文华酒店",
]


def load_sampled_data():
    print(f"[data] Loading {CSV_PATH}...")
    hotel_data = {h: [] for h in SELECTED_HOTELS}
    total_rows = 0
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = row["hotel_name"]
            if h in hotel_data and len(hotel_data[h]) < SAMPLES_PER_HOTEL:
                hotel_data[h].append({
                    "_id": row["_id"],
                    "comment": row["comment"],
                    "hotel_name": h,
                    "score": float(row.get("score", 0)),
                    "room_type": row.get("room_type", ""),
                })
            total_rows += 1
    print(f"[data] Total rows: ~{total_rows}")
    for h in SELECTED_HOTELS:
        print(f"  {h}: {len(hotel_data[h])} samples")
    return hotel_data


def build_test_queries(hotel_data):
    queries = []
    seen = set()

    # 2a. intent
    intent_cases = [
        ("游泳池干净吗", "facility"), ("健身房开放时间", "facility"), ("有停车场吗", "facility"),
        ("早餐品种多不多", "food"), ("餐厅有什么好吃的", "food"), ("早餐是自助的吗", "food"),
        ("价格贵不贵", "price"), ("性价比怎么样", "price"), ("一晚多少钱", "price"),
        ("交通方便吗", "location"), ("离地铁站多远", "location"), ("周边有商圈吗", "location"),
        ("前台服务态度好吗", "service"), ("入住退房效率高吗", "service"), ("工作人员热情吗", "service"),
        ("房间干净吗", "cleanliness"), ("卫生间卫生怎么样", "cleanliness"), ("床单干净吗", "cleanliness"),
        ("隔音效果好吗", "quiet"), ("晚上吵不吵", "quiet"), ("安静吗", "quiet"),
        ("适合带孩子去吗", "child_friendly"), ("儿童乐园怎么样", "child_friendly"), ("亲子设施完善吗", "child_friendly"),
        ("适合老人住吗", "elder_friendly"), ("有电梯吗老人方便吗", "elder_friendly"),
        ("整体体验怎么样", "overall"), ("值得推荐吗", "overall"), ("住的感受如何", "overall"),
    ]
    for q, intent in intent_cases:
        if q not in seen:
            n = sum(1 for x in queries if x["id"].startswith(f"intent_{intent}_"))
            queries.append({"id": f"intent_{intent}_{n}", "query": q,
                            "expected_intent": intent, "type": "intent"})
            seen.add(q)

    # 2b. hotel-specific
    hotel_qs = [
        ("北京索菲特大酒店", "北京索菲特大酒店的游泳池怎么样"),
        ("上海浦东香格里拉大酒店", "浦东香格里拉的江景房值得吗"),
        ("汉庭酒店(北京天坛店)", "汉庭天坛店离天坛多远"),
        ("北京王府井文华东方酒店", "文华东方王府井服务怎么样"),
    ]
    for h, q in hotel_qs:
        if q not in seen:
            n = sum(1 for x in queries if x["type"] == "hotel_specific")
            queries.append({"id": f"hotel_{n}", "query": q,
                            "expected_intent": "mixed", "type": "hotel_specific", "hotel": h})
            seen.add(q)

    # 2c. mixed
    mixed_cases = [
        ("早餐好吃吗 游泳池怎么样", "facility"),
        ("房间大吗 服务好吗", "facility"),
        ("交通方便吗 早餐丰富吗", "location"),
        ("价格贵不贵 隔音好不好", "price"),
        ("适合带孩子吗 房间干净吗", "child_friendly"),
    ]
    for q, intent in mixed_cases:
        if q not in seen:
            n = sum(1 for x in queries if x["type"] == "mixed")
            queries.append({"id": f"mixed_{intent}_{n}", "query": q,
                            "expected_intent": intent, "type": "mixed"})
            seen.add(q)

    # 2d. BM25
    bm25_cases = [
        ("早餐 品种 丰富", True), ("游泳池 干净 水质", True),
        ("前台 服务 态度", True), ("隔音 噪音 吵", True),
        ("价格 性价比 划算", True), ("亲子 儿童 乐园", True),
    ]
    for q, hit in bm25_cases:
        if q not in seen:
            n = sum(1 for x in queries if x["type"] == "bm25")
            queries.append({"id": f"bm25_{n}", "query": q,
                            "expected_bm25_hit": hit, "type": "bm25"})
            seen.add(q)

    # 2e. edge
    edge_cases = [("早餐", "food"), ("", "mixed"),
                  ("breakfast", "food"), ("hskjdhfksjdhf", "mixed")]
    for q, intent in edge_cases:
        if q not in seen:
            n = sum(1 for x in queries if x["type"] == "edge")
            queries.append({"id": f"edge_{n}", "query": q,
                            "expected_intent": intent, "type": "edge"})
            seen.add(q)

    print(f"[queries] Built {len(queries)} test queries")
    return queries


def run_tests(queries, hotel_data):
    passed = 0
    failed = 0
    errors = []

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            msg = f"FAIL  {name}  {detail}"
            print(f"  {msg}")
            errors.append(msg)

    print(f"\n{'='*60}")
    print("EXTENDED TEST SUITE: 多酒店跨场景测试")
    print(f"12 hotels, {sum(len(v) for v in hotel_data.values())} samples")
    print(f"{'='*60}")

    # ── 1. Intent Classification ──
    from _shared.router import classify_intent_detailed

    n_intent = sum(1 for q in queries if q["type"] in ("intent", "mixed", "edge"))
    print(f"\n--- 1. Intent Classification ({n_intent} cases) ---")
    for tc in queries:
        if tc["type"] not in ("intent", "mixed", "edge"):
            continue
        q = tc["query"]
        if not q:
            r = classify_intent_detailed(q)
            check(f"[{tc['id']}] empty->mixed", r["primary"] == "mixed")
            continue
        r = classify_intent_detailed(q)
        exp = tc["expected_intent"]
        check(f"[{tc['id']}] {q[:24]:<24s} -> {exp:<16s}",
              r["primary"] == exp, f"got {r['primary']}")

    # ── 2. BM25 ──
    from _shared.bm25 import InvertedIndex

    print(f"\n--- 2. BM25 Multi-Hotel Index ---")
    print("[bm25] Building...")
    multi_idx = InvertedIndex()
    multi_docs = {}
    for h, docs in hotel_data.items():
        for d in docs:
            if d["comment"].strip():
                multi_docs[d["_id"]] = d["comment"]
    multi_idx.build(multi_docs)
    print(f"[bm25] {multi_idx.num_docs} docs, {len(multi_idx.index)} terms")

    for tc in queries:
        if tc["type"] != "bm25":
            continue
        q = tc["query"]
        t0 = time.time()
        res = multi_idx.search(q, topk=10)
        ems = (time.time() - t0) * 1000
        check(f"[{tc['id']}] {q[:20]:<20s} -> {len(res):<2d} hits in {ems:.1f}ms",
              len(res) > 0)

    # ── 3. Cross-Hotel ──
    print(f"\n--- 3. Cross-Hotel BM25 ---")
    for h in SELECTED_HOTELS[:8]:
        short = h[:18]
        t0 = time.time()
        res = multi_idx.search(h, topk=5)
        ems = (time.time() - t0) * 1000
        own = sum(1 for d_id, _ in res
                  if any(d["_id"] == d_id for d in hotel_data[h]))
        check(f"[cross] {short:<18s} -> {len(res)} hits, {own} own-hotel in {ems:.1f}ms",
              len(res) > 0)

    # ── 4. Router ──
    print(f"\n--- 4. Router / Retriever Integration ---")
    from _shared.router import route

    for q in ["游泳池", "早餐", "房间 照片"]:
        try:
            r = route(q, top_k=3)
            s = r.get("strategies", [])
            check(f"[route] '{q[:15]:<15s}' strategies={s}", len(s) > 0)
        except Exception as e:
            check(f"[route] '{q[:15]}' error", False, str(e))

    # ── 5. Performance ──
    print(f"\n--- 5. Performance Benchmarks ---")
    t0 = time.time()
    bench = InvertedIndex()
    bench.build(dict(list(multi_docs.items())[:1000]))
    bt = time.time() - t0
    check(f"BM25 build 1000 docs < 1s", bt < 1.0, f"took {bt:.3f}s")

    qtimes = []
    for q in ["早餐","游泳池","服务","房间","噪音","停车场"]:
        for _ in range(3):
            t0 = time.time()
            bench.search(q, topk=10)
            qtimes.append((time.time() - t0) * 1000)
    avg = sum(qtimes) / len(qtimes)
    check(f"BM25 query avg < 10ms ({len(qtimes)} samples)",
          avg < 10.0, f"avg={avg:.2f}ms")

    # ── 6. Hotel Specific ──
    print(f"\n--- 6. Hotel-Specific Classification ---")
    for tc in queries:
        if tc["type"] != "hotel_specific":
            continue
        q = tc["query"]
        r = classify_intent_detailed(q)
        check(f"[{tc['id']}] {q[:24]:<24s} intent={r['primary']}",
              r["primary"] is not None)

    # ── Summary ──
    print(f"\n{'='*60}")
    total = passed + failed
    pct = passed / total * 100 if total > 0 else 0
    print(f"RESULTS: {passed}/{total} passed ({pct:.0f}%)")
    if errors:
        print(f"FAILURES ({failed}):")
        for e in errors:
            print(f"  {e}")
    print(f"{'='*60}")
    return {"passed": passed, "total": total, "failed": failed, "errors": errors}


if __name__ == "__main__":
    print("=" * 60)
    print("宿说 — 多酒店扩展测试套件")
    print("=" * 60)
    hotel_data = load_sampled_data()

    queries = build_test_queries(hotel_data)
    qpath = os.path.join(OUTPUT_DIR, "extended_queries.json")
    json.dump(queries, open(qpath, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[queries] Saved -> {qpath}")

    result = run_tests(queries, hotel_data)
    rpath = os.path.join(OUTPUT_DIR, "extended_results.json")
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hotels": len(SELECTED_HOTELS),
        "samples_per_hotel": SAMPLES_PER_HOTEL,
        "total_queries": len(queries),
        "passed": result["passed"],
        "total": result["total"],
        "failed": result["failed"],
        "errors": result["errors"],
    }, open(rpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[results] Saved -> {rpath}")
    sys.exit(0 if result["failed"] == 0 else 1)
