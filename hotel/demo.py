"""宿说 — 终端交互 Demo.

    python hotel/demo.py
"""

import sys, os

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "scripts"))
os.chdir(_PROJ)

BANNER = """
  ╔══════════════════════════════════════╗
  ║         宿 说                         ║
  ║  酒店评论智能顾问 · 场景化 Agent       ║
  ╚══════════════════════════════════════╝

  基于 6400+ 条广州花园酒店真实住客评论

  示例:
    · 带 5 岁宝宝去住，推荐吗？
    · 半夜 12:30 到达可以入住吗？
    · 枕头是乳胶的吗？
    · 可以点外卖到房间吗？

  输入 /quit 退出
"""

QUIT_WORDS = {"/quit", "/exit", "quit", "exit", "q"}


def main():
    # 预热模型
    print("  正在加载模型...", end="", flush=True)
    from _shared.cache import cache
    cache.get_bm25()
    cache.get_comments()
    print(" 就绪。")

    from advisor.agent import advise_stream

    print(BANNER)

    while True:
        try:
            user_input = input("  >>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in QUIT_WORDS:
            print("  再见！")
            break

        print()
        _stream(advise_stream, user_input)
        print()


def _stream(advise_stream, user_input: str):
    """流式输出各阶段，最后打印完整回复."""
    reply = ""

    try:
        for event in advise_stream(user_input):
            phase = event.get("phase", "")
            msg = event.get("msg", "")

            if phase == "recognizing":
                print(f"  {msg}")
            elif phase == "scene_done":
                print(f"  {msg}\n")
            elif phase == "dimensions":
                print(f"  {msg}")
            elif phase == "query_rewrite":
                print(f"    {msg}")
            elif phase == "search":
                print(f"\n  {msg}")
            elif phase in ("searching", "search_done"):
                print(f"    {msg}")
            elif phase == "synthesizing":
                print(f"\n  {msg}")
            elif phase == "synthesize_done":
                print(f"  {msg}\n")
            elif phase == "done":
                reply = event.get("reply", "")
                t = event.get("metrics", {}).get("total_latency", 0)
                print(f"  ({t:.1f}s)\n")
                print("-" * 50)
                print(reply)
                print("-" * 50)

    except Exception as e:
        print(f"  出错了: {e}")


if __name__ == "__main__":
    main()
