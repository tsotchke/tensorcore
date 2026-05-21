---
name: Performance regression
about: A kernel got slower
title: ''
labels: performance, regression
assignees: ''
---

## Numbers

| Bench | Last known good | Current |
|---|---|---|
| Shape / dtype | (e.g. `bench_gemm 4096³ fp16`) | |
| Throughput | (e.g. 17.88 TFLOPS) | |
| Backend | (e.g. `simdgroup_matrix`) | |

## Bisection

- Last known good commit (sha):
- First bad commit (sha):

## Environment

- Chip:
- macOS:
- Xcode SDK:
- `tc_version()`:

## Reproducer

```sh
# Exact command that produces the regressed number.
./build/bench/bench_gemm | grep "4096.*fp16"
```

## Hypothesis

<!-- Optional: what you think caused it. Kernel change? Tile-size change? SDK update? -->
