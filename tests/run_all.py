"""宿说 — 系统化测试套件 (Run All)

用法:
    python tests/run_all.py              # 跑全部测试
    python tests/run_all.py unit         # 仅单元测试
    python tests/run_all.py integration  # 仅集成测试
    python tests/run_all.py eval         # 仅评测基准
"""

import sys, os, subprocess, time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJ)

TEST_SCRIPTS = {
    "unit": [
        ("单元测试 (30条)", "tests/unit/test_all.py"),
    ],
    "integration": [
        ("路由器测试 (28条)", "tests/integration/test_router.py"),
        ("多酒店测试 (48条)", "tests/integration/test_extended.py"),
        ("大规模多酒店 (725条)", "tests/integration/test_multi_hotel.py"),
    ],
    "eval": [
        ("召回指标基准", "tests/eval/recall_benchmark.py"),
    ],
}

def run_suite(name, scripts):
    print(f"\n{'='*60}")
    print(f"  [{name}] {' | '.join(s[0] for s in scripts)}")
    print(f"{'='*60}")
    passed = 0
    failed = 0
    for label, path in scripts:
        print(f"\n--- {label} ---")
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        t0 = time.time()
        ret = subprocess.run([sys.executable, path], capture_output=True, text=True)
        elapsed = time.time() - t0
        if ret.returncode == 0:
            passed += 1
            print(f"  [PASS] {elapsed:.1f}s")
        else:
            failed += 1
            print(f"  [FAIL] exit={ret.returncode} {elapsed:.1f}s")
            # 打印关键输出
            for line in ret.stdout.splitlines():
                if "FAIL" in line or "RESULTS" in line or "Error" in line:
                    print(f"    {line}")
    return passed, failed

if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else list(TEST_SCRIPTS.keys())
    
    print("=" * 60)
    print("  宿说 — 系统化测试套件")
    print(f"  目录: {PROJ}")
    print(f"  运行: {', '.join(args)}")
    print("=" * 60)
    print()
    print("  测试分类:")
    print("    tests/unit/          单元测试 (30条检查点)")
    print("    tests/integration/   集成测试 (路由/多酒店)")
    print("    tests/eval/          评测基准 (召回指标)")
    print("    tests/data/          测试数据集/ground truth")
    print("    tests/results/       评测结果存档")
    
    total_p = 0
    total_f = 0
    for suite in args:
        if suite in TEST_SCRIPTS:
            p, f = run_suite(suite, TEST_SCRIPTS[suite])
            total_p += p
            total_f += f
    
    print(f"\n{'='*60}")
    print(f"  汇总: {total_p} pass, {total_f} fail")
    print(f"{'='*60}")
    sys.exit(1 if total_f > 0 else 0)
