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

There are 19, on CPU, with a tiny random GPT-2. They check:
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
pytest -q                                      # 19 tests, CPU
python -m qlab.pipeline --out results --device cuda
```

**Results are pending** from the Colab run.
