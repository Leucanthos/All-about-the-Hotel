import sys, os, json, time, warnings, logging
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from _shared.router import classify_intent_detailed, classify_intent, route

PROJ = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJ, "tests")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_CASES = [
    # facility
    {"id":"facility_1","query":"游泳池干净吗","exp_intent":"facility"},
    {"id":"facility_2","query":"有健身房吗","exp_intent":"facility"},
    {"id":"facility_3","query":"停车场收费吗","exp_intent":"facility"},
    # food
    {"id":"food_1","query":"早餐好吃吗","exp_intent":"food"},
    {"id":"food_2","query":"餐厅有什么推荐","exp_intent":"food"},
    {"id":"food_3","query":"自助餐怎么样","exp_intent":"food"},
    # price
    {"id":"price_1","query":"价格贵不贵","exp_intent":"price"},
    {"id":"price_2","query":"性价比怎么样","exp_intent":"price"},
    {"id":"price_3","query":"多少钱一晚","exp_intent":"price"},
    # location
    {"id":"location_1","query":"交通方便吗","exp_intent":"location"},
    {"id":"location_2","query":"离地铁多远","exp_intent":"location"},
    {"id":"location_3","query":"周边有商圈吗","exp_intent":"location"},
    # service
    {"id":"service_1","query":"前台服务态度好吗","exp_intent":"service"},
    {"id":"service_2","query":"入住效率高吗","exp_intent":"service"},
    {"id":"service_3","query":"工作人员热情吗","exp_intent":"service"},
    # cleanliness
    {"id":"clean_1","query":"房间干净吗","exp_intent":"cleanliness"},
    {"id":"clean_2","query":"卫生间卫生吗","exp_intent":"cleanliness"},
    # quiet
    {"id":"quiet_1","query":"隔音效果好吗","exp_intent":"quiet"},
    {"id":"quiet_2","query":"晚上吵不吵","exp_intent":"quiet"},
    # child
    {"id":"child_1","query":"适合带孩子吗","exp_intent":"child_friendly"},
    {"id":"child_2","query":"亲子设施怎么样","exp_intent":"child_friendly"},
    # elder
    {"id":"elder_1","query":"适合老人住吗","exp_intent":"elder_friendly"},
    # overall
    {"id":"overall_1","query":"酒店体验怎么样","exp_intent":"overall"},
    {"id":"overall_2","query":"值得推荐吗","exp_intent":"overall"},
    # mixed
    {"id":"mixed_1","query":"早餐好吃吗 游泳池怎么样","exp_intent":"facility"},
    {"id":"mixed_2","query":"房间装修豪华吗 服务态度好吗","exp_intent":"facility"},
    {"id":"mixed_3","query":"交通方便吗 早餐丰富吗","exp_intent":"location"},
    # edge
    {"id":"edge_short","query":"早餐","exp_intent":"food"},
    {"id":"edge_empty","query":"","exp_intent":"mixed"},
    {"id":"edge_special","query":"房间！！干净？？","exp_intent":"facility"},
    {"id":"edge_english","query":"breakfast","exp_intent":"mixed"},
    {"id":"edge_numbers","query":"123456","exp_intent":"mixed"},
    {"id":"edge_gibberish","query":"hskjdhfjksdhfksjdhf","exp_intent":"mixed"},
]

def evaluate():
    print("="*65)
    print("  路由器精度 + 耗时评测")
    print("="*65)

    # 1. 意图分类精度
    print(f"\n--- 1. 意图分类精度 ({len(TEST_CASES)} cases) ---")
    correct = 0
    failures = []
    lats = []
    for tc in TEST_CASES:
        q = tc["query"]
        t0 = time.perf_counter()
        r = classify_intent_detailed(q)
        el = (time.perf_counter()-t0)*1_000_000
        lats.append(el)
        ok = r["primary"] == tc["exp_intent"]
        if ok: correct += 1
        else: failures.append((tc["id"], q, tc["exp_intent"], r["primary"]))
        print(f"  {'O' if ok else 'X'} {tc['id']:<20s} {q[:20]:<20s} -> {r['primary']:<14s}")

    acc = correct/len(TEST_CASES)*100
    print(f"\n  准确率: {correct}/{len(TEST_CASES)} = {acc:.1f}%")
    if failures:
        print(f"  失败:")
        for tid,q,exp,got in failures:
            print(f"    {q[:24]:<24s} expected={exp:<14s} got={got}")

    # 2. 延迟
    print(f"\n--- 2. 延迟分析 ({len(lats)} samples) ---")
    la = np.array(lats)
    print(f"  分类 only: mean={la.mean():.1f}us median={np.median(la):.1f}us "
          f"min={la.min():.1f}us max={la.max():.1f}us")

    # route() 延迟 (不含缓存/模型加载)
    print(f"\n--- 3. route() 延迟 (不含检索, 跳过首次CLIP加载) ---")
    rlats = []
    for tc in TEST_CASES[:8]:
        t0 = time.perf_counter()
        r = route(tc["query"], top_k=3, strategies=None)
        el = (time.perf_counter()-t0)*1000
        if el < 1000:  # 忽略首次 CLIP 加载
            rlats.append(el)
        strat = r.get("strategies", [])
        print(f"  {tc['query'][:20]:<20s} -> {el:>8.3f}ms  strategies={strat}")

    if rlats:
        ra = np.array(rlats)
        print(f"  route() avg={ra.mean():.3f}ms min={ra.min():.3f}ms max={ra.max():.3f}ms")

    # 4. 旧版 vs dyx
    print(f"\n--- 4. 旧版英文 vs dyx 中文 ---")
    old_correct = 0
    for tc in TEST_CASES:
        got = classify_intent(tc["query"])
        # 映射到策略级别
        if got == "visual" and tc["exp_intent"] == "facility": old_correct += 1
        elif got == "qa" and tc["exp_intent"] != "facility" and tc["exp_intent"] != "mixed": old_correct += 1
        elif got == "mixed" and tc["exp_intent"] == "mixed": old_correct += 1
    old_acc = old_correct/len(TEST_CASES)*100
    print(f"  旧版英文 (qa/visual/mixed): {old_correct}/{len(TEST_CASES)} = {old_acc:.1f}%")
    print(f"  dyx 中文 (10维):           {correct}/{len(TEST_CASES)} = {acc:.1f}%")

    # 保存
    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases": len(TEST_CASES),
        "intent_accuracy_pct": round(acc, 1),
        "classification_latency_us": {
            "mean": round(float(la.mean()), 1),
            "median": round(float(np.median(la)), 1),
            "min": round(float(la.min()), 1),
            "max": round(float(la.max()), 1),
        },
        "route_latency_ms": {
            "mean": round(float(np.mean(rlats)), 3),
            "min": round(float(np.min(rlats)), 3),
            "max": round(float(np.max(rlats)), 3),
        } if rlats else None,
        "old_classifier_accuracy_pct": round(old_acc, 1),
        "dyx_classifier_accuracy_pct": round(acc, 1),
        "failures": [(tid, q[:30], f"exp={exp}, got={got}") for tid,q,exp,got in failures],
    }
    json.dump(out, open(os.path.join(OUTPUT_DIR, "router_eval_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[results] Saved")
    print(f"  意图分类: {acc:.1f}% | 分类延迟: {la.mean():.1f}us | "
          f"旧版->dyx: {old_acc:.1f}% -> {acc:.1f}%")

if __name__ == "__main__":
    evaluate()
