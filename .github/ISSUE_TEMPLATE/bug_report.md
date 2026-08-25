---
name: Bug report
about: Something behaves incorrectly
title: ''
labels: bug
assignees: ''
---

## What happened

<!-- Describe the failure. If a call returned the wrong status code, name the code. If the result was numerically wrong, give the rms_scaled error magnitude. -->

## Reproducer

```c
// Smallest C / Python / shell snippet that reproduces the bug.
```

## Environment

- Chip (run `system_profiler SPDisplaysDataType | head -3` or `./build/examples/hello_gemm`):
- macOS version (`sw_vers`):
- Xcode SDK (`xcrun --show-sdk-version`):
- `tc_version()`:
- `tc_device_info.family`:
- Build flags (`cmake -B build -D...`):

## Expected vs actual

- Expected:
- Actual:

## Diagnostic output

After the failing call, print `tc_backend_name(tc_last_backend())` and
paste the result. If it is one of `mps`, `accelerate_cpu`, or `none`
when you expected `simdgroup_matrix`, mention that.

## Anything else

<!-- Logs, ctest output, kernel patches you tried, etc. -->
