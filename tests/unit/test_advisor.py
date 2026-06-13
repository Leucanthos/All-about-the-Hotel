"""Advisor 模块测试 — recognizer + scenes + pipeline (~20 条检查点).

用法: python tests/unit/test_advisor.py
"""

import sys, os, json, time

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "scripts"))
os.chdir(PROJ)

OUTPUT_DIR = os.path.join("tests", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(OUTPUT_DIR, "test_advisor_results.json")

passed = 0
failed = 0
errors = []


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        msg = f"FAIL  {name}  {detail}"
        print(f"  {msg}")
        errors.append(msg)


print("=" * 60)
print("TEST SUITE: Advisor 模块")
print("=" * 60)

# ═══════════════════════════════════════════════════════
# 1. Recognizer — 场景识别
# ═══════════════════════════════════════════════════════
print("\n--- 1. Recognizer: Scene Classification ---")
from advisor.recognizer import recognize, _detect_practical

# 1a. 实务查询预检 — 原有覆盖
print("  1a. Practical pre-check (core)")
r = _detect_practical("半夜12:30到达可以入住吗")
check("半夜12:30 → practical", r is not None and r["scene"] == "practical",
      f"got {r}")
check("半夜12:30 → checkin subtype",
      r and r["specifics"].get("subtype") == "checkin",
      f"got {r.get('specifics',{}).get('subtype') if r else None}")

r = _detect_practical("能延时退房吗")
check("延时退房 → practical", r is not None and r["scene"] == "practical", f"got {r}")
r = _detect_practical("可以寄存行李吗")
check("寄存行李 → practical/luggage",
      r is not None and r["specifics"].get("subtype") == "luggage", f"got {r}")
r = _detect_practical("有接送机服务吗")
check("接送机 → practical/shuttle",
      r is not None and r["specifics"].get("subtype") == "shuttle", f"got {r}")
r = _detect_practical("能用信用卡支付吗")
check("信用卡 → practical/payment",
      r is not None and r["specifics"].get("subtype") == "payment", f"got {r}")
r = _detect_practical("可以带猴子入住吗")
check("带猴子 → practical/pet",
      r is not None and r["specifics"].get("subtype") == "pet", f"got {r}")
r = _detect_practical("让带宠物吗")
check("宠物 → practical/pet",
      r is not None and r["specifics"].get("subtype") == "pet", f"got {r}")

# 1b. 离谱查询预检 — 设施/饮食/健康/技术/杂项
print("  1b. Practical pre-check (crazy edge cases)")
cases = [
    ("能带大象入住吗", "pet"),
    ("凌晨3点还能叫room service吗", "checkin"),
    ("枕头是羽绒的还是乳胶的", "room_amenity"),
    ("有清真早餐吗", "dietary"),
    ("酒店有心脏除颤器吗", "health_safety"),
    ("淋浴水压够不够大", "room_amenity"),
    ("能借HDMI线吗", "tech_service"),
    ("房间里有异味怎么办", "health_safety"),
    ("盲人入住方便吗", "accessibility"),
    ("浴缸放满水要多久", "room_amenity"),
    ("马桶是智能的吗", "room_amenity"),
    ("窗户能打开吗", "room_amenity"),
    ("可以用国外手机号收验证码吗", "tech_service"),
    ("酒店有防蚊措施吗", "health_safety"),
    ("早餐有素食选项吗", "dietary"),
    ("能帮我收快递吗", "service_misc"),
    ("可以点外卖到房间吗", "service_misc"),
    ("酒店提供免费避孕套吗", "room_amenity"),
    ("有一次性牙刷和牙膏吗", "room_amenity"),
    ("生日前一天到达能帮忙布置房间吗", "special_req"),
    ("有加床服务吗多少钱", "special_req"),
    ("你们酒店能抽烟吗", "special_req"),
    ("能在房间烧香拜佛吗", "special_req"),
    ("我要在酒店里开30人的party", "special_req"),
    ("床垫适合腰椎间盘突出的人吗", "room_amenity"),
]
for q, expected_subtype in cases:
    r = _detect_practical(q)
    ok = r is not None and r["scene"] == "practical"
    subtype_ok = r and r["specifics"].get("subtype") == expected_subtype
    check(f"'{q[:20]}' → practical/{expected_subtype}",
          ok and subtype_ok,
          f"got scene={r['scene'] if r else None} subtype={r['specifics'].get('subtype') if r else None}")

# 1c. 非实务查询不应命中
print("  1c. Non-practical queries should not match")
r = _detect_practical("带5岁宝宝去住，推荐吗？")
check("亲子 → None", r is None, f"got {r}")
r = _detect_practical("下周出差开会合适吗")
check("出差 → None", r is None, f"got {r}")
r = _detect_practical("游泳池干净吗")
check("泳池 → None", r is None, f"got {r}")
r = _detect_practical("度蜜月适合吗")
check("蜜月 → None", r is None, f"got {r}")
r = _detect_practical("带70岁老人去，方便吗")
check("老人场景 → None (elder)", r is None, f"got {r}")

# 1d. 关键词回退 (不调 LLM，用 keyword_fallback 测试)
print("  1d. Keyword fallback scenes")
from advisor.recognizer import _keyword_fallback

kw = _keyword_fallback("带宝宝去住")
check("宝宝 → family", kw["scene"] == "family", f"got {kw['scene']}")

kw = _keyword_fallback("出差开会")
check("出差 → business", kw["scene"] == "business", f"got {kw['scene']}")

kw = _keyword_fallback("度蜜月去")
check("蜜月 → romance", kw["scene"] == "romance", f"got {kw['scene']}")

kw = _keyword_fallback("带老人去住")
check("老人 → elder", kw["scene"] == "elder", f"got {kw['scene']}")

kw = _keyword_fallback("和朋友聚会")
check("朋友 → friends", kw["scene"] == "friends", f"got {kw['scene']}")

kw = _keyword_fallback("一个人背包旅行")
check("一个人 → solo", kw["scene"] == "solo", f"got {kw['scene']}")

kw = _keyword_fallback("xyz神秘查询")
check("无意义 → general", kw["scene"] == "general", f"got {kw['scene']}")

# 1e. recognize() 端到端 (关键词路径，不调 LLM)
print("  1e. recognize() end-to-end")
r = recognize("半夜12:30到达可以入住吗")
check("recognize 半夜 → practical", r["scene"] == "practical",
      f"got {r['scene']} (router={r.get('router')})")

r = recognize("带宝宝去住")
check("recognize 宝宝 → family", r["scene"] == "family",
      f"got {r['scene']} (router={r.get('router')})")

r = recognize("游泳池干净吗")
check("recognize 泳池 → practical (facility query, LLM routed)",
      r["scene"] == "practical", f"got {r['scene']}")


# ═══════════════════════════════════════════════════════
# 2. Scenes — 场景-维度矩阵
# ═══════════════════════════════════════════════════════
print("\n--- 2. Scenes: Dimension Matrix ---")
from advisor.scenes import (get_dimensions_for_scene, get_scene_label,
                            SCENE_LABELS, SCENE_ORDER, DIMENSION_INFO)

# 2a. 每个场景有标签
for s in SCENE_ORDER:
    check(f"SCENE_LABELS['{s}'] exists", s in SCENE_LABELS,
          f"missing label for {s}")

# 2b. 每个场景有核心维度
for s in SCENE_ORDER:
    dims = get_dimensions_for_scene(s)
    core = [d for d in dims if d["level"] == "●"]
    check(f"{s}: {len(core)} core dims >= 1", len(core) >= 1,
          f"only {len(core)} core dims for {s}")

# 2c. 实务场景维度
practical_dims = get_dimensions_for_scene("practical")
check("practical scene has dimensions", len(practical_dims) > 0)
practical_keys = [d["key"] for d in practical_dims]
check("checkin in practical", "checkin" in practical_keys)
check("checkout in practical", "checkout" in practical_keys)
check("front_desk in practical", "front_desk" in practical_keys)

# 2d. 每个维度有 query 模板
from advisor.scenes import QUERY_TEMPLATES
for s in SCENE_ORDER:
    templates = QUERY_TEMPLATES.get(s, {})
    dims = get_dimensions_for_scene(s)
    for d in dims:
        if d["level"] == "●":
            check(f"{s}/{d['key']} has template",
                  d["key"] in templates and len(templates[d["key"]]) > 0,
                  f"missing template for {s}/{d['key']}")

# 2e. DIMENSION_INFO 完整
for key in practical_keys + ["network", "quiet", "safety", "pet"]:
    check(f"DIMENSION_INFO['{key}'] exists",
          key in DIMENSION_INFO, f"missing {key}")


# ═══════════════════════════════════════════════════════
# 3. Pipeline — advise() / advise_stream()
# ═══════════════════════════════════════════════════════
print("\n--- 3. Pipeline: advise() ---")
from advisor.agent import advise, advise_stream

# 3a. practical 查询走正确场景
r = advise("半夜12:30到达可以入住吗", verbose=False)
check("advise 半夜 → scene=practical",
      r["scene"] == "practical", f"got {r['scene']}")
check("advise 半夜 → has reply",
      len(r.get("reply", "")) > 0, f"reply empty")
check("advise 半夜 → no LLM synthesis (judgments empty)",
      r.get("judgments") == [], f"got judgments: {r.get('judgments')}")
check("advise 半夜 → no dimensions (direct lookup)",
      r.get("dimensions") == [], f"got dims: {r.get('dimensions')}")

# 3b. 离谱查询: 走 practical 直接检索，维度为空
crazy_tests = [
    "可以带猴子入住吗",
    "枕头是羽绒的还是乳胶的",
    "有清真早餐吗",
    "能借HDMI线吗",
    "酒店有防蚊措施吗",
    "能帮我收快递吗",
    "你们酒店能抽烟吗",
    "能在房间烧香拜佛吗",
    "可以带8个人入住双人房吗",
]
for q in crazy_tests:
    r = advise(q, verbose=False)
    check(f"advise '{q[:15]}' → practical",
          r["scene"] == "practical", f"got {r['scene']}")
    check(f"advise '{q[:15]}' has reply",
          len(r.get("reply", "")) > 0)
    check(f"advise '{q[:15]}' dims empty (direct lookup)",
          r.get("dimensions") == [],
          f"got dims: {r.get('dimensions',[])}")

# 3c. 正常场景仍走 LLM
r = advise("带宝宝去住", verbose=False)
check("advise 宝宝 → scene=family",
      r["scene"] in ("family", "general"),
      f"got {r['scene']}")

# 3d. general 不回 crash
r = advise("游泳池", verbose=False)
check("advise 泳池 → no crash", isinstance(r, dict) and "reply" in r,
      f"got {type(r)}")

# ═══════════════════════════════════════════════════════
print("\n--- 4. Pipeline: advise_stream() ---")
# 4a. 流式 yield 结构
events = list(advise_stream("半夜12:30到达可以入住吗"))
check("stream → has events", len(events) > 0)
check("stream → first: recognizing",
      events[0].get("phase") == "recognizing",
      f"got {events[0].get('phase')}" if events else "empty")
check("stream → first msg has no emoji",
      not any(0x1F300 <= ord(c) <= 0x1F9FF for c in events[0].get("msg", "")),
      f"msg: {events[0].get('msg','')}" if events else "")
done_events = [e for e in events if e.get("phase") == "done"]
check("stream → last is done",
      len(done_events) >= 1, f"got {len(done_events)} done events")
if done_events:
    check("stream → done has reply",
          len(done_events[0].get("reply", "")) > 0,
          "reply is empty")

# 4b. all phases are valid
valid_phases = {"recognizing", "scene_done", "dimensions", "query_rewrite",
                "search", "searching", "search_done", "synthesizing",
                "synthesize_done", "done"}
for e in events:
    phase = e.get("phase", "")
    check(f"stream phase '{phase}' is valid",
          phase in valid_phases, f"unknown phase: {phase}")

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
total = passed + failed
pct = passed / total * 100 if total > 0 else 0
print(f"RESULTS: {passed}/{total} passed ({pct:.0f}%)")
if errors:
    print(f"FAILURES ({failed}):")
    for e in errors:
        print(f"  {e}")
print(f"{'='*60}")

json.dump({
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "total": total, "passed": passed, "failed": failed, "errors": errors,
}, open(RESULTS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"Results saved: {RESULTS_PATH}")

sys.exit(0 if failed == 0 else 1)
