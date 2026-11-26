#!/usr/bin/env python3
"""fix_ai_kernel_holyc.py <src dir> -- byte-level fixes for holyc_lint findings in the
fork's kernel AI layer: struct assignment -> MemCopy, nested declarations -> function
top, `continue` -> goto. Idempotent: an edit is skipped once its new text is present."""
import sys

T, N = b"\t", b"\n"


def fix(path, edits):
    b = open(path, "rb").read()
    nl = b"\r\n" if b"\r\n" in b[:4000] else b"\n"
    for old, new in edits:
        old, new = old.replace(N, nl), new.replace(N, nl)
        if new and new in b:
            continue
        if b.count(old) == 1:
            b = b.replace(old, new)
            print(f"{path}: fixed {old.strip()[:50]!r}")
        elif new:
            raise SystemExit(f"{path}: anchor not found: {old[:60]!r}")
    open(path, "wb").write(b)


def loop_tail(var: bytes, arr: bytes, label: bytes) -> tuple:
    """Put `label:` (plus a no-op statement) at the end of the best_* scan loop."""
    old = (T * 3 + b"best_idx = i;" + N + T * 2 + b"}" + N + T + b"}" + N + N
           + T + b"if (best_idx >= 0)" + N + T * 2 + b"return &infer_queue." + arr + b"[best_idx];")
    new = (T * 3 + b"best_idx = i;" + N + T * 2 + b"}" + N + label + b":" + N
           + T * 2 + var + b" = " + var + b";" + N + T + b"}" + N + N
           + T + b"if (best_idx >= 0)" + N + T * 2 + b"return &infer_queue." + arr + b"[best_idx];")
    return old, new


def main(src):
    k = f"{src}/Kernel"
    fix(f"{k}/FluxIDT.ZC", [
        (b"\t\t\t\tfv->handlers[j] = fv->handlers[j + 1];",
         b"\t\t\t\tMemCopy(&fv->handlers[j], &fv->handlers[j + 1], sizeof(CFluxHandler));"),
    ])
    fix(f"{k}/CapEngine.ZC", [
        (b"\t\tchild->caps[i] = parent->caps[i];",
         b"\t\tMemCopy(&child->caps[i], &parent->caps[i], sizeof(CCapability));"),
        (b"{//Debug: print last N audit log entries.\n\tI64 i, idx;\n",
         b"{//Debug: print last N audit log entries.\n\tI64 i, idx;\n\tU8 *audit_res_st;\n"),
        (b"\t\tU8 *audit_res_st;\n", b""),
    ])
    fix(f"{k}/InferQueue.ZC", [
        (b"\tCInferReq\t*req;\n\tCModelEntry\t*model;\n",
         b"\tCInferReq\t*req;\n\tCModelEntry\t*model;\n\tI64\t\t\t latency;\n"),
        (b"\t\tI64 latency = req->complete_tsc - req->start_tsc;",
         b"\t\tlatency = req->complete_tsc - req->start_tsc;"),
        # ZealC has no `continue`: jump to a label at the end of the loop body
        (b"\t\tif (!m->available) continue;\n",
         b"\t\tif (!m->available)\n\t\t\tgoto ms_next;\n"),
        loop_tail(b"best_score", b"models", b"ms_next"),
        (b"\t\tif (req->status != INFER_STATUS_QUEUED) continue;\n",
         b"\t\tif (req->status != INFER_STATUS_QUEUED)\n\t\t\tgoto pn_next;\n"),
        loop_tail(b"best_priority", b"requests", b"pn_next"),
        (b"{//Print inference queue state.\n\tI64 i;\n\tCInferReq *req;\n",
         b"{//Print inference queue state.\n\tI64 i;\n\tCInferReq *req;\n\tU8 *avail_st;\n"),
        (b"\t\tU8 *avail_st;\n", b""),
    ])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
