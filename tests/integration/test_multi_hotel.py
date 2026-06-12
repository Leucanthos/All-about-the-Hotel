import sys, os, json, time
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, 'scripts'))
os.chdir(PROJ)

# 加载查询集
qp = os.path.join(PROJ, 'tests', 'extended_queries_large.json')
with open(qp, encoding='utf-8') as f:
    data = json.load(f)

queries = data['queries']
print(f'Loaded {len(queries)} queries')
print(f'Breakdown: {data["breakdown"]}')

# 测试
from _shared.router import classify_intent_detailed
from _shared.bm25 import InvertedIndex

passed = 0
failed = 0
errors = []

def check(name, cond, detail=''):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        msg = f'FAIL {name}  {detail}'
        errors.append(msg)

# 建 BM25 索引
print('Building BM25 index...')
import csv
CSV_PATH = os.path.join(PROJ, 'comments.csv')
hotels = ['北京索菲特大酒店','上海浦东香格里拉大酒店','北京世纪金源大饭店',
          '北京大兴国际机场木棉花酒店','北京三里屯通盈中心洲际酒店',
          '北京远航国际酒店（首都机场新国展店）','朗丽兹太和府酒店（北京生命科学园地铁站店）',
          '桔子水晶北京南锣鼓巷酒店','北京万达文华酒店']
hotel_data = {h: [] for h in hotels}
with open(CSV_PATH, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        h = row['hotel_name']
        if h in hotel_data and len(hotel_data[h]) < 200:
            hotel_data[h].append(row['comment'])
idx = InvertedIndex()
docs = {}
for h, ds in hotel_data.items():
    for d in ds:
        if d.strip():
            docs[str(hash(d))] = d
idx.build(docs)

# 1. 意图分类
ni = data['breakdown'].get('intent', 0)
print(f'\n--- Intent ({ni} cases) ---')
count = 0
for tc in queries:
    if tc['type'] != 'intent': continue
    q = tc['query']
    r = classify_intent_detailed(q)
    exp = tc['expected_intent']
    check(f"[{tc['id']}] {q[:25]:<25s} -> expect={exp:<14s}", r['primary'] == exp, f"got={r['primary']}")
    count += 1
    if count % 100 == 0:
        print(f'  ... {count}/{ni}')

# 2. 酒店特定 (sample 100)
nh = data['breakdown'].get('hotel_specific', 0)
print(f'\n--- Hotel-Specific (sampled 100 of {nh}) ---')
count = 0
for tc in queries:
    if tc['type'] != 'hotel_specific': continue
    count += 1
    if count > 100: break
    q = tc['query']
    r = classify_intent_detailed(q)
    check(f"[{tc['id']}] {q[:25]:<25s}", r['primary'] is not None)

# 3. 混合意图
nm = data['breakdown'].get('mixed', 0)
print(f'\n--- Mixed ({nm} cases) ---')
for tc in queries:
    if tc['type'] != 'mixed': continue
    q = tc['query']
    r = classify_intent_detailed(q)
    exp = tc['expected_intent']
    check(f"[{tc['id']}] {q[:25]:<25s} -> expect={exp}", r['primary'] == exp, f"got={r['primary']}")

# 4. BM25
nb = data['breakdown'].get('bm25', 0)
print(f'\n--- BM25 Search ({nb} cases) ---')
for tc in queries:
    if tc['type'] != 'bm25': continue
    q = tc['query']
    t0 = time.time()
    res = idx.search(q, topk=10)
    ems = (time.time()-t0)*1000
    check(f"[{tc['id']}] {q[:20]:<20s} -> {len(res):<2d} hits in {ems:.1f}ms", len(res) > 0)

# 5. 边界
ne = data['breakdown'].get('edge', 0)
print(f'\n--- Edge ({ne} cases) ---')
for tc in queries:
    if tc['type'] != 'edge': continue
    q = tc['query']
    exp = tc['expected_intent']
    if not q.strip():
        r = classify_intent_detailed(q)
        check(f"[{tc['id']}] empty->mixed", r['primary'] == 'mixed')
    else:
        r = classify_intent_detailed(q)
        check(f"[{tc['id']}] {q[:20]:<20s} -> expect={exp:<12s}", r['primary'] == exp, f"got={r['primary']}")

# 汇总
print(f'\n{"="*60}')
total = passed + failed
pct = passed / total * 100 if total else 0
print(f'RESULTS: {passed}/{total} passed ({pct:.0f}%)')
if errors:
    shown = set()
    for e in errors:
        key = e.split('got=')[-1] if 'got=' in e else e
        if key not in shown:
            print(f'  {e}')
            shown.add(key)
            if len(shown) >= 15: break
    print(f'Total failures: {len(errors)}')
print(f'{"="*60}')

# 保存
rp = os.path.join(PROJ, 'tests', 'extended_large_results.json')
with open(rp, 'w', encoding='utf-8') as f:
    json.dump({'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
               'total_queries': len(queries), 'tested': total,
               'passed': passed, 'failed': failed, 'failures': errors[:100]},
              f, ensure_ascii=False, indent=2)
print(f'Results -> {rp}')
