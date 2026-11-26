#!/usr/bin/env python3
"""bench_on_card.py <card-id> -- the awrise job behind decision card d-9gnf: the card RUNS on approve.

Each wake reads the card (awask show --json) and acts on the owner's answer, exactly once:
  run-now--fleet-dark-2-3-h        -> bench_all_cores.py --execute now
  run-tonight-at-01-00             -> run once local time is between 01:00 and 05:00
  rent-aws-nested-virt-box-instead -> record it; the AWS lane raises its own spend card
  not-now / cancelled              -> retire
A done-marker in ~/.aither makes a second wake a no-op. Exit 0 = nothing to do or done.
"""
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = os.path.join(os.path.expanduser("~"), ".aither", "awzeal-bench-card.done")


def main(card: str) -> int:
    if os.path.exists(MARK):
        print("already acted on", card)
        return 0
    out = subprocess.run(["awask", "show", card, "--json"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=60)
    if out.returncode != 0:
        print("could not read card:", out.stderr[-300:])
        return 2
    c = json.loads(out.stdout)
    status, answer = c.get("status"), c.get("answer")
    print(f"card {card}: status={status} answer={answer}")
    if status in ("cancelled", "retired") or answer == "not-now":
        open(MARK, "w").write(f"{card} {answer or status}\n")
        return 0
    if not answer:
        return 0
    if answer == "rent-aws-nested-virt-box-instead":
        open(MARK, "w").write(f"{card} aws: hand to the AWS lane\n")
        print("AWS chosen: the aws lane raises its own spend card; nothing runs locally")
        return 0
    if answer == "run-tonight-at-01-00" and not (1 <= datetime.datetime.now().hour < 5):
        print("approved for tonight; waiting for 01:00")
        return 0
    open(MARK, "w").write(f"{card} {answer} started {datetime.datetime.now().isoformat()}\n")
    env = dict(os.environ, AW_PLATFORM_FIRST_OK="1")
    rc = subprocess.run([sys.executable, os.path.join(HERE, "bench_all_cores.py"), "--execute"],
                        env=env).returncode
    with open(MARK, "a") as f:
        f.write(f"finished rc={rc} {datetime.datetime.now().isoformat()}\n")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "d-9gnf"))
