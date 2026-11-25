# llm-quantization-from-scratch

[![CI](https://github.com/kishorhange111/llm-quantization-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/llm-quantization-from-scratch/actions/workflows/ci.yml)

GPT-2 is quantised to 4 and 3 bits with the methods production stacks use, each written from
its paper: **RTN**, **GPTQ**, **AWQ** and **SmoothQuant W8A8**. Every method gets the same
calibration data and evaluation windows.

## Methods ([`quant.py`](src/qlab/quant.py))

**RTN.** Asymmetric min/max scale and zero point per row, either per channel or per group of
128 input columns. This is the baseline everything else is measured against.

**GPTQ** (Frantar et al. 2023). It minimises the layer's output error `‖XWᵀ − XQᵀ‖²`, not
the weight error, using the Hessian `H = 2XᵀX/n` of the calibration inputs. Columns are
quantised one at a time. Each one's rounding error is pushed onto the columns not yet
quantised, in proportion to `H⁻¹` (the Optimal Brain Surgeon update). The paper's
engineering is included:
- the **upper Cholesky factor of H⁻¹** instead of repeated inverse updates;
- **lazy batch updates**, which apply the accumulated error to the rest of the matrix once
  per 128-column block, turning many rank-1 updates into one matmul;
- **dampening** of 1% of mean(diag H);
- **dead-input handling**;
- optional **act-order**, which quantises the columns with the largest `diag(H)` first;
- group grids computed from the *already-updated* weights.

**AWQ** (Lin et al. 2024). A few input channels carry much larger activations, and the
weights that meet them matter most. Scaling input channel j by `s_j = mean|x_j|^α` before
rounding gives those weights more grid resolution. The inverse scale is folded into the
activations: `y = (x/s)·Q(W·s)ᵀ`. α is grid-searched per layer on calibration outputs.

**SmoothQuant W8A8** (Xiao et al. 2023). This quantises *activations* too (8-bit,
per-tensor). Activation outliers would wreck a per-tensor scale, so they are moved into the
weights with `s_j = max|x_j|^α / max|w_j|^(1−α)`. That is exact before rounding:
`x·W = (x/s)·(sW)`.

Calibration is sequential, as in GPTQ. Block *i* is calibrated on inputs from blocks
0…*i*−1 that are already quantised, so each block corrects for errors upstream.

### Tests

There are 21, on CPU, with a tiny random GPT-2. They check:
- RTN error is at most half a step;
- **GPTQ with uncorrelated inputs (H ∝ I) reduces exactly to RTN**, with and without
  act-order, since there is no error to propagate;
- GPTQ beats RTN by over 30% on correlated inputs at 2–4 bits;
- the lazy block size doesn't change the result;
- dead inputs are handled;
- AWQ never does worse than RTN and picks α > 0 when there are outlier channels;
- smoothing is exact before rounding;
- 8-bit weights are nearly lossless end to end;
- the pipeline resumes.

## Evaluation ([`pipeline.py`](src/qlab/pipeline.py))

- **Model:** GPT-2 (124M).
- **Calibration:** 128 × 512 tokens from WikiText-2 train.
- **Perplexity:** 40 non-overlapping 1,024-token windows of WikiText-2 test.
- **Configurations:** fp16; RTN, GPTQ (± act-order) and AWQ at w4/w3 with group 128;
  RTN and GPTQ per channel; W8A8 with and without smoothing.
- **Also reported:** the mean per-layer relative output error.

```
pytest -q                                      # 21 tests, CPU
python -m qlab.pipeline --out results --device cuda
```

## Results (Tesla T4)

All rows use the same calibration and evaluation sets. The raw logs are in [`results/`](results).
"Layer error" is the mean relative output error of the quantised linear layers, measured on
the **calibration** inputs.

| config | ppl | Δ vs fp16 | layer error |
|---|---|---|---|
| fp16 | 31.33 | – | – |
| RTN w4 g128 | 34.71 | +3.38 | 1.17e-02 |
| **GPTQ w4 g128** | 32.64 | +1.31 | 4.01e-03 |
| **GPTQ w4 g128 act-order** | **32.56** | **+1.23** | 3.39e-03 |
| AWQ w4 g128 | 33.75 | +2.41 | 9.07e-03 |
| RTN w3 g128 | 66.72 | +35.39 | 4.96e-02 |
| **GPTQ w3 g128** | **40.56** | **+9.23** | 1.82e-02 |
| AWQ w3 g128 | 57.32 | +25.99 | 3.96e-02 |
| RTN w4 per-channel | 39.72 | +8.39 | 2.63e-02 |
| GPTQ w4 per-channel | 34.31 | +2.98 | 8.90e-03 |
| RTN w3 per-channel | 525.15 | +493.82 | 1.33e-01 |
| GPTQ w3 per-channel | 68.73 | +37.40 | 3.77e-02 |
| W8A8 per-tensor | 48.35 | +17.02 | 3.97e-02 |
| SmoothQuant W8A8 α=0.5 (`c_attn`, `c_fc`) | 44.85 | +13.52 | 3.61e-02 |
| SmoothQuant W8A8 α=0.5, all 4 linears | 2325.16 | +2293.83 | 1.30e-02 |

**GPTQ is the clear winner.** At 4 bits with groups of 128 it loses 1.3 perplexity points,
against 3.4 for RTN and 2.4 for AWQ. At 3 bits RTN collapses (+35) and GPTQ holds at +9.
The gap widens without groups: at w3 per channel, RTN goes to 525 and GPTQ stays at 69.
Act-order helps slightly at w4 (32.64 → 32.56).

**AWQ sits between RTN and GPTQ.** Activation-aware scaling recovers about 30% of RTN's
loss at w4, and its per-layer grid search picked a mean α of 0.31. On GPT-2, error feedback (GPTQ) is worth
more than protecting salient channels (AWQ).

**Activation quantisation is what hurts GPT-2.** Per-tensor 8-bit activations cost 17
points, more than 4-bit GPTQ weights. Smoothing the two layers that read LayerNorm output
(`attn.c_attn`, `mlp.c_fc`) recovers 3.5 of those points.

### Smoothing every linear layer breaks the model

The first run smoothed all four linear layers per block, and perplexity went to **2325**.
Its layer error (1.30e-02) was the *lowest* of the W8A8 rows, because that error is
measured on calibration data. A diagnostic run on a smaller evaluation set (fp16 32.97 there)
smoothed one layer type at a time:

| smoothed | ppl |
|---|---|
| `attn.c_attn` | 33.08 |
| `attn.c_proj` | 33.06 |
| `mlp.c_fc` | 33.01 |
| `mlp.c_proj` | **2089** |

The cause is in block 0. One input channel of `mlp.c_proj` (the GELU output) reaches 155×
its calibration maximum on evaluation text. Its smoothing factor was fitted to the small
calibration range. After dividing by it, that channel peaks at 50.5 on evaluation text,
against 8.5 in calibration. Activations share one dynamic absmax scale per tensor, so the
outlier coarsens the grid of every other channel in the layer. The SmoothQuant paper
smooths only the layers fed by LayerNorm (QKV and FC1), and the pipeline now does the same
(`SMOOTHABLE` in [`pipeline.py`](src/qlab/pipeline.py)). The all-layers configuration is
kept as a documented failure case.

The lesson: per-layer error on calibration data cannot catch a factor that is fitted to a
calibration range the evaluation data exceeds. Only the end-to-end evaluation shows it.
