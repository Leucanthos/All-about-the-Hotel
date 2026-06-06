import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(cmd):
    print("$", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    print(completed.stdout[:2000])
    if completed.stderr:
        print(completed.stderr[:1000])


def main():
    run(
        [
            sys.executable,
            "cli.py",
            "q16",
            "build-sft",
            "--max-samples",
            "40",
            "--out-dir",
            "../data/demo_run",
        ]
    )
    run([sys.executable, "cli.py", "q18", "ask", "早餐和前台服务怎么样？有哪些常见吐槽？", "--show-trace"])


if __name__ == "__main__":
    main()
