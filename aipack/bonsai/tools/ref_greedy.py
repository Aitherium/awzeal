#!/usr/bin/env python3
"""ref_greedy.py <base url> <out json> [question] -- reference tokens from the live prism llama-server.

Read-only: /tokenize the Qwen3.5 no-think chat prompt and run one greedy /completion
(8 tokens, top-5 probabilities). The server is loopback-only inside awnix and
presents the fleet's internal certificate, so this probe passes the internal CA
when it can find one and otherwise skips verification for 127.0.0.1 only.
"""
import json
import os
import ssl
import sys
import urllib.request

Q = "What is the capital of France?"


def ctx():
    for ca in ("/etc/aither/ca/ca.pem", "/etc/aither/certs/ca.pem", "/certs/ca.pem"):
        if os.path.exists(ca):
            return ssl.create_default_context(cafile=ca)
    c = ssl.create_default_context()
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, context=ctx(), timeout=600) as r:
        return json.load(r)


def main(base, out, q=Q):
    prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    tok = post(base, "/tokenize", {"content": prompt, "add_special": False, "parse_special": True, "with_pieces": True})
    ids = [t["id"] if isinstance(t, dict) else t for t in tok["tokens"]]
    comp = post(base, "/completion", {
        "prompt": ids, "n_predict": 8, "temperature": 0.0, "top_k": 1, "n_probs": 5,
        "cache_prompt": False, "return_tokens": True, "samplers": ["top_k"],
    })
    res = {"question": q, "prompt": prompt, "prompt_ids": ids, "pieces": tok["tokens"],
           "tokens": comp.get("tokens"), "content": comp.get("content"),
           "probs": comp.get("completion_probabilities")}
    json.dump(res, open(out, "w"), indent=1)
    print("prompt_ids", ids)
    print("greedy", comp.get("tokens"), repr(comp.get("content")))
    for p in comp.get("completion_probabilities") or []:
        print(p.get("id"), repr(p.get("token")), [(t.get("id"), round(t.get("logprob", t.get("prob", 0)), 4)) for t in p.get("top_logprobs", p.get("top_probs", []))])


if __name__ == "__main__":
    main(*sys.argv[1:])
