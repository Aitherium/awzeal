# awzeal inference pack

ZealOS runs a real LLM in ring 0. Qwen3-0.6B (Q8_0) answers on the ZealOS console with no Linux, no host round-trip and no network. The tokens are computed by the guest CPU inside ZealOS.

Measured 2026-11-26, QEMU KVM, Ryzen 9 9950X3D, one core:

```
[AILLM] ready: engine 552640 B, model 613 MiB, ctx 1024, vocab 151936, load 102.59s
The capital of France is Paris.
ZLOUT-END prefill=24 tok 1.34s decode=7 tok 1.88s (3.72 tok/s) ctx=31/1024
The capital of Germany is Berlin.          <- second turn, same KV cache
ZLOUT-END prefill=20 tok 0.47s decode=7 tok 0.37s (18.96 tok/s) ctx=60/1024
```

Screens: `evidence/p3-answers.png` (answers on the console), `evidence/p2-ready.png`.

## How it works

| piece | what it is |
|---|---|
| `engine/zeal_glue.asm` | Wraps the [llm.asm](https://github.com/alesha-pro/llm.asm) (MIT) shared Qwen3 engine (AVX2/FMA/F16C NASM) as a flat, RIP-relative image with a HolyC ABI: stack args, callee pops, all GPRs and MXCSR saved, private 512 KiB stack. It enables CR4.OSXSAVE and XCR0 on whatever core it runs on, because ZealOS never turns AVX on. |
| `engine/pack_engine.py` | Links the image twice (base 0 and base 0x10000000) and diffs the two to derive relocations. The count today is zero. Emits `ENGINE.BIN`. |
| `zealos/AILLM.ZC` | HolyC loader: reads ENGINE.BIN and the `.asmbin` model into 4 KiB-aligned sys_task memory, binds `ZlInit/ZlBegin/ZlNext/ZlEnd/ZlReset/ZlStat` as function pointers, then streams tokens. `AIAsk(prompt)` and `AIChat` (with `/reset`, `/think`, `/nothink`, `/quit`). Every answer is mirrored to COM1 between `ZLOUT-BEGIN` and `ZLOUT-END`. |
| `zealos/ONCE-AUTOSTART.ZC` | Replaces the live ISO's `/Home/Once.ZC` in place, same length (two copies; the T: drive runs the one with "Take Tour"). It mounts every SATA ATA port, because the live CD mounts only B: and T:, then runs `RUN.ZC` from the drive that holds it. |
| `host/` | Builds and boots everything on a Linux host with KVM: `fetch-and-export.sh` (weights to `.asmbin`), `mkdisk.sh` (FAT32 AI-pack disk), `patch_once.py`, and `zctl.py`, which boots, sends keys, takes screenshots and waits on COM1 markers. |

## Run it

```bash
bash aipack/host/fetch-and-export.sh     # Qwen3-0.6B -> models/qwen3-0.6b-q8_0.asmbin
bash aipack/host/z.sh engine             # ENGINE.BIN
bash aipack/host/z.sh disk               # aidisk.img (ENGINE.BIN, QWEN3.BIN, AILLM.ZC, RUN.ZC)
bash aipack/host/z.sh iso                # zealos-ai-live.iso from the upstream ZealOS BIOS ISO
bash aipack/host/z.sh run @aipack/steps/proof.txt
```

## Traps we hit

- The boot splash holds until any key; `Once.ZC` runs only after it. `zctl.py key:spc` handles that.
- `sendkey` needs a real display backend (`-display vnc=...`). With `-display none` the guest never sees a key.
- The header's `ATAMount` prototype is stale (5 arguments, while the kernel takes 3), so a call fails to compile. The autostart inlines its body instead.
- HolyC `try`/`catch` need braces.

## Next

- Done: every core works (`ZlWorker` tasks pinned per CPU). Agents infer locally through `InferQueue` (`AIINFER.ZC`, `AIKERNEL.ZC`).
- Bonsai 2.5 27B has its own engine in `bonsai/` (ENGINE2.BIN, same ABI); it is at parity with llama.cpp on Linux, and its in-guest boot is in progress.
