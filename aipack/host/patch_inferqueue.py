#!/usr/bin/env python3
"""patch_inferqueue.py <src dir> -- wire the kernel InferQueue to a local CPU engine.

Byte-level edits (these files carry raw cp437 bytes; text-mode tools corrupt them):
  KernelA.HH : CInferQueue gains CInferResp responses[INFER_QUEUE_CAPACITY]
  InferQueue.ZC : infer_local_cpu_fp hook, InferResp(req), LOCAL_CPU backend calls the hook
Idempotent: exits 0 without changes when already applied.
"""
import sys

MARK = b"infer_local_cpu_fp"


def sub(data: bytes, old: bytes, new: bytes, what: str) -> bytes:
    if data.count(old) != 1:
        raise SystemExit(f"{what}: expected one match, found {data.count(old)}")
    return data.replace(old, new)


def main(src: str) -> int:
    ka = f"{src}/Kernel/KernelA.HH"
    iq = f"{src}/Kernel/InferQueue.ZC"
    a = open(ka, "rb").read()
    q = open(iq, "rb").read()
    if MARK in q:
        print("already applied")
        return 0
    nl = b"\r\n" if b"\r\n" in a[:2000] else b"\n"
    a = sub(a, b"\tCInferQueueStats stats;" + nl,
            b"\tCInferQueueStats stats;" + nl
            + b"\tCInferResp\t responses[INFER_QUEUE_CAPACITY]; //ZealOS-AI: one per request slot" + nl,
            "KernelA.HH CInferQueue")
    nq = b"\r\n" if b"\r\n" in q[:2000] else b"\n"
    q = sub(q, b"CInferQueue infer_queue;" + nq,
            b"CInferQueue infer_queue;" + nq
            + b"//Local CPU engine hook: set by /AIPack AILLM when an engine is loaded." + nq
            + b"//Fills InferResp(req) and returns TRUE on success." + nq
            + b"Bool (*infer_local_cpu_fp)(CInferReq *req, CInferResp *resp) = NULL;" + nq + nq
            + b"CInferResp *InferResp(CInferReq *req)" + nq
            + b"{//Response slot paired with a request." + nq
            + b"\treturn &infer_queue.responses[(req(U8 *) - infer_queue.requests(U8 *)) / sizeof(CInferReq)];" + nq
            + b"}" + nq,
            "InferQueue.ZC global")
    old_case = (b"\t\tcase INFER_BACKEND_LOCAL_CPU:" + nq
                + b"\t\t\t//Phase 2: llama.cpp inference" + nq
                + b"\t\t\treq->status = INFER_STATUS_FAILED;" + nq)
    new_case = (b"\t\tcase INFER_BACKEND_LOCAL_CPU:" + nq
                + b"\t\t\t//Phase 2: ring-0 engine (aipack AILLM) through infer_local_cpu_fp" + nq
                + b"\t\t\tif (infer_local_cpu_fp && (*infer_local_cpu_fp)(req, InferResp(req)))" + nq
                + b"\t\t\t{" + nq
                + b"\t\t\t\treq->status = INFER_STATUS_COMPLETE;" + nq
                + b"\t\t\t\tbreak;" + nq
                + b"\t\t\t}" + nq
                + b"\t\t\treq->status = INFER_STATUS_FAILED;" + nq)
    q = sub(q, old_case, new_case, "InferQueue.ZC LOCAL_CPU case")
    open(ka, "wb").write(a)
    open(iq, "wb").write(q)
    print("patched", ka, iq)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
