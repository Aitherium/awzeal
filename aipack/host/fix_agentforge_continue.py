#!/usr/bin/env python3
"""fix_agentforge_continue.py <src dir> -- rewrite AgentTick's 10 `continue`s as gotos
(ZealC has no `continue`). Byte-level, idempotent (no-op once the labels exist)."""
import sys

T, N = b"\t", b"\n"


def main(src: str) -> int:
    p = f"{src}/Kernel/AgentForge.ZC"
    b = open(p, "rb").read()
    if b"at_n1:" in b:
        print("already applied")
        return 0
    crlf = b"\r\n" in b[:4000]
    if crlf:
        b = b.replace(b"\r\n", b"\n")
    s = b.find(b"U0 AgentTick()")
    e = b.find(b"//Dispatch best task", s)
    if s < 0 or e < 0:
        raise SystemExit("AgentTick not found")
    body = b[s:e]

    def once(old, new):
        nonlocal body
        if body.count(old) != 1:
            raise SystemExit(f"expected one {old[:60]!r}, found {body.count(old)}")
        body = body.replace(old, new)

    # loop 1: metrics
    guard = (T * 2 + b"if (!agent) continue;" + N
             + T * 2 + b"if (!Bt(&agent->agent_flags, AGENTf_ACTIVE)) continue;" + N
             + T * 2 + b"if (Bt(&agent->agent_flags, AGENTf_SUSPENDED)) continue;" + N)

    def guard_to(label):
        return (T * 2 + b"if (!agent) goto " + label + b";" + N
                + T * 2 + b"if (!Bt(&agent->agent_flags, AGENTf_ACTIVE)) goto " + label + b";" + N
                + T * 2 + b"if (Bt(&agent->agent_flags, AGENTf_SUSPENDED)) goto " + label + b";" + N)

    i1 = body.find(guard)
    body = body[:i1] + guard_to(b"at_n1") + body[i1 + len(guard):]
    once(T * 2 + b"kernel_state.system_pain += agent->pain;" + N + T + b"}" + N,
         T * 2 + b"kernel_state.system_pain += agent->pain;" + N
         + b"at_n1:" + N + T * 2 + b"i = i;" + N + T + b"}" + N)
    # loop 2: scoring (outer)
    once(guard, guard_to(b"at_n2"))
    once(T * 3 + b"agent->cooldown--;" + N + T * 3 + b"continue;" + N,
         T * 3 + b"agent->cooldown--;" + N + T * 3 + b"goto at_n2;" + N)
    once(T * 2 + b"if (agent->hourly_used >= agent->hourly_budget)" + N + T * 3 + b"continue;" + N,
         T * 2 + b"if (agent->hourly_used >= agent->hourly_budget)" + N + T * 3 + b"goto at_n2;" + N)
    # inner task loop
    once(T * 3 + b"if (!at->action) continue;" + N, T * 3 + b"if (!at->action) goto at_n3;" + N)
    once(T * 4 + b"continue;" + N, T * 4 + b"goto at_n3;" + N)
    once(T * 4 + b"best_task = j;" + N + T * 3 + b"}" + N + T * 2 + b"}" + N + T + b"}" + N,
         T * 4 + b"best_task = j;" + N + T * 3 + b"}" + N
         + b"at_n3:" + N + T * 3 + b"j = j;" + N + T * 2 + b"}" + N
         + b"at_n2:" + N + T * 2 + b"i = i;" + N + T + b"}" + N)
    if b"continue" in body.replace(b"//", b"\0"):
        raise SystemExit("a continue survived in AgentTick")
    b = b[:s] + body + b[e:]
    if crlf:
        b = b.replace(b"\n", b"\r\n")
    open(p, "wb").write(b)
    print("patched", p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
