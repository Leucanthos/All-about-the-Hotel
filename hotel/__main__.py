"""宿说 — 场景顾问 Agent CLI.

用法:
    hotel               # 终端交互 Demo (默认)
    hotel advise "..."  # 单次查询 (显式调用)
"""

import sys, os as _os, argparse

_PROJ = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _os.path.join(_PROJ, "scripts"))


def cmd_demo(args):
    """终端交互 Demo (默认)."""
    from hotel.demo import main as demo_main
    demo_main()


def cmd_advise(args):
    """单次查询."""
    from advisor.agent import advise
    result = advise(args.query, verbose=False)
    reply = result.get("reply", "")
    # strip emoji
    import re
    reply = re.sub(r"[\U0001F300-\U0001F9FF\U0001FA00-\U0001FA6F"
                   r"\U0001FA70-\U0001FAFF\U00002702-\U000027B0"
                   r"\U000024C2-\U0001F251]+", "", reply)
    print()
    print(reply)
    print()
    print(f"  延迟 {result['metrics']['total_latency']}s  |  "
          f"场景 {result.get('scene_label','?')}  |  "
          f"置信度 {result.get('confidence',0):.0%}")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="hotel",
        description="宿说 — 酒店评论场景顾问 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            '  hotel                          # 终端交互 Demo\n'
            '  hotel advise "带宝宝去住推荐吗"  # 单次查询\n'
            "配置 DeepSeek API Key:\n"
            "  echo DEEPSEEK_API_KEY=sk-xxx > .env"
        ),
    )
    sub = parser.add_subparsers(dest="command")

    # demo 子命令
    d = sub.add_parser("demo", help="终端交互 Demo (默认)")
    d.set_defaults(func=cmd_demo)

    # advise 子命令
    a = sub.add_parser("advise", help="单次场景顾问查询")
    a.add_argument("query", help="自然语言查询")
    a.set_defaults(func=cmd_advise)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        # 无子命令 → 默认 demo
        cmd_demo(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
