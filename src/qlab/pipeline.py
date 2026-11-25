"""Quantise GPT-2 block by block and measure perplexity.

    python -m qlab.pipeline --out results --device cuda          # wikitext-2 (needs datasets)
    python -m qlab.pipeline --out results --data stdlib          # offline fallback

Calibration follows GPTQ's sequential recipe. Block i's four linear layers (``c_attn``,
``attn.c_proj``, ``c_fc``, ``mlp.c_proj``) are calibrated on inputs produced by blocks
0…i−1 that are *already quantised*, so each block corrects for the errors upstream. Every
method sees the same 128 × 512-token calibration set and the same evaluation windows.
Finished configurations are kept in ``<out>/runs.jsonl``.

W8A8 configurations also quantise activations. Each input is quantised per tensor to 8 bits,
dynamically (absmax), before the matmul. SmoothQuant (Xiao et al. 2023) first divides
input channel j by ``s_j = max|x_j|^α / max|w_j|^(1−α)`` and multiplies the weights by it.
The product is unchanged, but activation outliers move into the weights, which tolerate
them better.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import sysconfig
import time
import zlib

import torch
from torch import nn

from .quant import awq, gptq, output_error, rtn

LAYERS = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")


class ActQuantLinear(nn.Module):
    """y = Q8(x / s) · W' + b, with W' = Q8_per_out_channel(s · W). HF Conv1D layout (in, out)."""

    def __init__(self, conv, s: torch.Tensor) -> None:
        super().__init__()
        W = conv.weight.detach().float() * s[:, None]  # (in, out)
        amax = W.abs().amax(0, keepdim=True).clamp(min=1e-8)
        W = torch.round(W / amax * 127).clamp(-127, 127) * amax / 127
        self.weight = nn.Parameter(W.to(conv.weight.dtype), requires_grad=False)
        self.bias = conv.bias
        self.register_buffer("inv_s", (1.0 / s).to(conv.weight.dtype))

    def forward(self, x):
        x = x * self.inv_s
        amax = x.abs().amax().clamp(min=1e-8)
        x = torch.round(x / amax * 127).clamp(-127, 127) * amax / 127
        return torch.addmm(self.bias, x.reshape(-1, x.shape[-1]), self.weight).view(
            *x.shape[:-1], self.weight.shape[1]
        )


def _get(block, name):
    m = block
    for p in name.split("."):
        m = getattr(m, p)
    return m


def _set(block, name, mod):
    parent, leaf = name.rsplit(".", 1)
    setattr(_get(block, parent), leaf, mod)


@torch.no_grad()
def _forward_all(model, ids, bs=8):
    for i in range(0, len(ids), bs):
        model(ids[i : i + bs])


@torch.no_grad()
def quantize_model(
    model,
    method: str,
    calib: torch.Tensor,
    bits: int = 4,
    group: int | None = 128,
    act_order: bool = False,
    alpha: float = 0.5,
    n_sample: int = 4096,
    seed: int = 0,
) -> dict:
    """Quantise ``model`` (HF GPT2LMHeadModel) in place. Returns per-layer diagnostics."""
    gen = torch.Generator().manual_seed(seed)
    blocks = model.transformer.h
    diag = {"layer_error": [], "awq_alpha": []}
    for block in blocks:
        stats = {n: {"H": None, "n": 0, "X": [], "amax": None} for n in LAYERS}

        def hook(name, stats=stats):
            def f(_m, inp, _out):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                st = stats[name]
                if method == "gptq":
                    h = x.t() @ x
                    st["H"] = h if st["H"] is None else st["H"] + h
                    st["n"] += x.shape[0]
                a = x.abs().amax(0)
                st["amax"] = a if st["amax"] is None else torch.maximum(st["amax"], a)
                k = max(1, n_sample * x.shape[0] // (len(calib) * calib.shape[1]))
                idx = torch.randint(0, x.shape[0], (k,), generator=gen).to(x.device)
                st["X"].append(x[idx])

            return f

        hs = [_get(block, n).register_forward_hook(hook(n)) for n in LAYERS]
        _forward_all(model, calib)
        for h in hs:
            h.remove()
        for name in LAYERS:
            conv, st = _get(block, name), stats[name]
            X = torch.cat(st["X"])
            W = conv.weight.detach().float().t()  # (out, in)
            if method == "rtn":
                Wq = rtn(W, bits, group)
            elif method == "gptq":
                Wq = gptq(W, 2 * st["H"] / st["n"], bits, group, act_order=act_order)
            elif method == "awq":
                Wq, a = awq(W, X, bits, group)
                diag["awq_alpha"].append(a)
            elif method in ("w8a8", "smoothquant"):
                if method == "smoothquant":
                    wmax = W.abs().amax(0).clamp(min=1e-5)
                    s = (st["amax"].clamp(min=1e-5) ** alpha / wmax ** (1 - alpha)).clamp(min=1e-5)
                else:
                    s = torch.ones(W.shape[1], device=W.device)
                q = ActQuantLinear(conv, s)
                _set(block, name, q)
                diag["layer_error"].append(_act_error(conv, q, X))
                continue
            else:
                raise ValueError(method)
            diag["layer_error"].append(output_error(W, Wq, X))
            conv.weight.copy_(Wq.t().to(conv.weight.dtype))
    return diag


def _act_error(conv, q, X):
    ref = X @ conv.weight.float() + conv.bias.float()
    got = q(X.to(q.weight.dtype)).float()
    return float((got - ref).pow(2).sum() / (ref - conv.bias.float()).pow(2).sum())


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, bs: int = 4) -> float:
    nll, n = 0.0, 0
    for i in range(0, len(ids), bs):
        x = ids[i : i + bs]
        logits = model(x).logits.float()
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum"
        )
        nll += float(loss)
        n += x[:, 1:].numel()
    return math.exp(nll / n)


def load_text(data: str):
    """(train text, test text)."""
    if data == "wikitext":
        from datasets import load_dataset

        d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        return "\n\n".join(d["train"]["text"]), "\n\n".join(d["test"]["text"])
    root = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    tr, te = [], []
    for f in files:
        key = zlib.crc32(os.path.relpath(f, root).replace("\\", "/").encode()) % 20
        (te if key == 0 else tr).append(open(f, encoding="utf-8", errors="ignore").read())
    return "\n".join(tr)[:4_000_000], "\n".join(te)[:2_000_000]


def windows(tokens: torch.Tensor, n: int, length: int, gen=None, random=False):
    if random:
        starts = torch.randint(0, len(tokens) - length, (n,), generator=gen)
    else:
        starts = torch.arange(n) * length
    return torch.stack([tokens[s : s + length] for s in starts])


CONFIGS = [
    ("fp16", {}),
    ("rtn w4 g128", dict(method="rtn", bits=4)),
    ("gptq w4 g128", dict(method="gptq", bits=4)),
    ("gptq w4 g128 act-order", dict(method="gptq", bits=4, act_order=True)),
    ("awq w4 g128", dict(method="awq", bits=4)),
    ("rtn w3 g128", dict(method="rtn", bits=3)),
    ("gptq w3 g128", dict(method="gptq", bits=3)),
    ("awq w3 g128", dict(method="awq", bits=3)),
    ("rtn w4 per-channel", dict(method="rtn", bits=4, group=None)),
    ("gptq w4 per-channel", dict(method="gptq", bits=4, group=None)),
    ("rtn w3 per-channel", dict(method="rtn", bits=3, group=None)),
    ("gptq w3 per-channel", dict(method="gptq", bits=3, group=None)),
    ("w8a8 per-tensor", dict(method="w8a8")),
    ("smoothquant w8a8 α=0.5", dict(method="smoothquant", alpha=0.5)),
]


def report(recs, out, meta):
    fp = next((r["ppl"] for r in recs if r["name"] == "fp16"), None)
    lines = [
        meta + "\n",
        "| config | ppl | Δ vs fp16 | mean layer output error | AWQ α |",
        "|---|---|---|---|---|",
    ]
    for r in recs:
        d = "–" if fp is None or r["name"] == "fp16" else f"{r['ppl'] - fp:+.2f}"
        e = r.get("mean_layer_error")
        a = r.get("mean_awq_alpha")
        lines.append(
            f"| {r['name']} | {r['ppl']:.2f} | {d} | "
            f"{'–' if e is None else f'{e:.2e}'} | {'–' if a is None else f'{a:.2f}'} |"
        )
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main(argv=None, model=None, tokens=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", choices=["wikitext", "stdlib"], default="wikitext")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--eval-len", type=int, default=1024)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "runs.jsonl")
    recs = [json.loads(x) for x in open(path)] if os.path.exists(path) else []
    done = {r["name"] for r in recs}
    if model is None:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        tok = GPT2TokenizerFast.from_pretrained(a.model)
        dt = torch.float16 if a.device == "cuda" else torch.float32
        model = GPT2LMHeadModel.from_pretrained(a.model, torch_dtype=dt).to(a.device).eval()
        train, test = load_text(a.data)
        tokens = (torch.tensor(tok(train)["input_ids"]), torch.tensor(tok(test)["input_ids"]))
    train_ids, test_ids = tokens
    gen = torch.Generator().manual_seed(0)
    calib = windows(train_ids, a.n_calib, a.calib_len, gen, random=True).to(a.device)
    n_eval = min(a.n_eval, len(test_ids) // a.eval_len)
    test = windows(test_ids, n_eval, a.eval_len).to(a.device)
    for name, kw in CONFIGS:
        if name in done or (a.only and name not in a.only):
            continue
        t0 = time.time()
        m = copy.deepcopy(model)
        diag = quantize_model(m, calib=calib, **kw) if kw else {}
        r = {"name": name, **kw, "ppl": perplexity(m, test), "quant_seconds": time.time() - t0}
        if diag.get("layer_error"):
            r["mean_layer_error"] = sum(diag["layer_error"]) / len(diag["layer_error"])
            r["layer_error"] = diag["layer_error"]
        if diag.get("awq_alpha"):
            r["mean_awq_alpha"] = sum(diag["awq_alpha"]) / len(diag["awq_alpha"])
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        recs.append(r)
        print(f"[{time.strftime('%H:%M:%S')}] {name}: ppl {r['ppl']:.2f}", flush=True)
        del m
    dev = torch.cuda.get_device_name() if a.device == "cuda" else "cpu"
    meta = (
        f"{a.model}, {a.data}, calibration {a.n_calib}×{a.calib_len} tokens, "
        f"eval {n_eval}×{a.eval_len} tokens (non-overlapping), {dev}."
    )
    print(report(recs, a.out, meta))


if __name__ == "__main__":
    main()
