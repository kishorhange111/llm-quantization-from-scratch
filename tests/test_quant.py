import copy

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from qlab.pipeline import ActQuantLinear, main, perplexity, quantize_model
from qlab.quant import awq, gptq, hessian, output_error, rtn


def _w(out=16, inp=64, seed=0):
    return torch.randn(out, inp, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _correlated_x(n=512, inp=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n, 8, generator=g, dtype=torch.float64)
    mix = torch.randn(8, inp, generator=g, dtype=torch.float64)
    return base @ mix + 0.1 * torch.randn(n, inp, generator=g, dtype=torch.float64)


@pytest.mark.parametrize("group", [None, 16])
def test_rtn_error_is_at_most_half_a_step_and_uses_few_levels(group):
    W = _w()
    Q = rtn(W, 3, group)
    span = 64 if group is None else group
    for i in range(0, 64, span):
        blk, qb = W[:, i : i + span], Q[:, i : i + span]
        step = (blk.amax(1).clamp(min=0) - blk.amin(1).clamp(max=0)) / 7
        assert torch.all((qb - blk).abs() <= step[:, None] / 2 + 1e-6)
        assert all(r.unique().numel() <= 8 for r in qb)


def test_hessian():
    X = _correlated_x()
    torch.testing.assert_close(hessian(X), 2 * X.t().float() @ X.float() / len(X))


@pytest.mark.parametrize("group", [None, 16])
def test_gptq_with_uncorrelated_inputs_is_rtn(group):
    """With H ∝ I, H⁻¹ is diagonal, so there is no error to propagate."""
    W = _w()
    torch.testing.assert_close(gptq(W, torch.eye(64), 4, group, block=16), rtn(W, 4, group))


def test_gptq_act_order_with_uncorrelated_inputs_is_rtn():
    W = _w()
    H = torch.diag(torch.rand(64, generator=torch.Generator().manual_seed(3)) + 0.5)
    torch.testing.assert_close(gptq(W, H, 4, None, act_order=True), rtn(W, 4, None))


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_gptq_beats_rtn_on_correlated_inputs(bits):
    W, X = _w(), _correlated_x()
    H = hessian(X)
    e_rtn = output_error(W, rtn(W, bits, None), X)
    e_gptq = output_error(W, gptq(W, H, bits, None, block=16), X)
    e_order = output_error(W, gptq(W, H, bits, None, block=16, act_order=True), X)
    assert e_gptq < 0.7 * e_rtn and e_order < e_rtn


def test_gptq_lazy_block_size_does_not_change_the_result():
    W, X = _w(), _correlated_x()
    H = hessian(X)
    torch.testing.assert_close(gptq(W, H, 3, None, block=8), gptq(W, H, 3, None, block=64))


def test_gptq_handles_dead_inputs():
    W, X = _w(), _correlated_x()
    X[:, 5] = 0
    Q = gptq(W, hessian(X), 4, None)
    assert torch.isfinite(Q).all() and (Q[:, 5] == 0).all()


def test_awq_picks_a_scale_no_worse_than_plain_rtn():
    W, X = _w(), _correlated_x()
    X[:, :4] *= 20  # a few outlier input channels, the case AWQ is designed for
    Wq, a = awq(W, X, 3, 16)
    assert 0.0 <= a <= 1.0
    assert output_error(W, Wq, X) <= output_error(W, rtn(W, 3, 16), X) + 1e-12
    assert a > 0  # with outliers, some scaling helps


def _tiny():
    torch.manual_seed(0)
    cfg = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=32,
        n_positions=128,
        vocab_size=200,
        bos_token_id=0,
        eos_token_id=0,
    )
    return GPT2LMHeadModel(cfg).eval()


def test_smoothing_is_exact_before_rounding():
    m = _tiny()
    conv = m.transformer.h[0].mlp.c_fc
    x = torch.randn(5, 32)
    s = torch.rand(32) + 0.5
    # x · W = (x / s) · (s W): what SmoothQuant relies on
    torch.testing.assert_close((x / s) @ (conv.weight * s[:, None]), x @ conv.weight)
    q = ActQuantLinear(conv, s)
    ref = conv(x)
    assert (q(x) - ref).abs().max() < 0.05 * ref.abs().max()


@pytest.mark.parametrize("method", ["rtn", "gptq", "awq", "w8a8", "smoothquant"])
def test_quantize_tiny_model(method):
    m = _tiny()
    ids = torch.randint(0, 200, (8, 64), generator=torch.Generator().manual_seed(4))
    base = perplexity(m, ids)
    q = copy.deepcopy(m)
    diag = quantize_model(
        q, method, ids[:4], bits=8 if method in ("rtn", "gptq") else 4, group=16, n_sample=64
    )
    assert len(diag["layer_error"]) == 8
    p = perplexity(q, ids)
    assert math_isfinite(p)
    if method in ("rtn", "gptq"):
        assert abs(p - base) / base < 0.01  # 8-bit weights: nearly lossless


def math_isfinite(x):
    return x == x and x != float("inf")


def test_pipeline_end_to_end_and_resume(tmp_path):
    m = _tiny()
    g = torch.Generator().manual_seed(5)
    toks = (
        torch.randint(0, 200, (20000,), generator=g),
        torch.randint(0, 200, (4000,), generator=g),
    )
    args = [
        "--device",
        "cpu",
        "--out",
        str(tmp_path),
        "--n-calib",
        "4",
        "--calib-len",
        "64",
        "--n-eval",
        "4",
        "--eval-len",
        "64",
        "--only",
        "fp16",
        "rtn w4 g128",
        "gptq w4 g128",
        "smoothquant w8a8 α=0.5",
    ]
    main(args, model=m, tokens=toks)
    lines = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    main(args, model=m, tokens=toks)
    assert len((tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()) == 4
    assert "Δ vs fp16" in (tmp_path / "summary.md").read_text(encoding="utf-8")
