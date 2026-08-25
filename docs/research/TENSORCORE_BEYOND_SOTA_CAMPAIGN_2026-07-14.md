# TensorCore Beyond-SOTA research campaign

Assessment date: 2026-07-14
ICC task: `tensorcore-beyond-sota-research-20260714`
Machine-readable contract: `configs/beyond_sota_research.json`

## Outcome

"Beyond SOTA" is a falsifiable systems contract, not a blanket speed claim. TensorCore earns it only by holding a statistically significant Pareto lead on a fleet-relevant workload while matching the model, weights, quality target, hardware class, request trace, and measured system boundary of the comparator. Microbenchmarks remain useful engineering evidence but cannot support an end-to-end claim. Projected results and theoretical peaks are not evidence.

The current checkout does not yet satisfy that contract. It has promising local kernels and unusually broad backend ambitions, but its public record still mixes measured microbenchmarks, synthetic inference, projected CUDA results, incomplete A100/H100/ROCm coverage, and a scheduler view smaller than the actual computer mesh and Tailscale topology.

## What the frontier requires

1. **Architecture-specific pipeline co-design.** FlashAttention-4 reports that Blackwell's tensor throughput scales differently from shared-memory and exponential throughput; its response is joint MMA, data-movement, softmax, and tile scheduling rather than a single faster matmul. Nautilus and VDCores strengthen the case for correctness-constrained auto-scheduling and dependency-driven overlap instead of static environment-variable choices.

2. **Native low precision with quality proof.** Blackwell NVFP4 and Metal tensor formats expose block-scaled FP4/FP8 capabilities. Storage compression alone is insufficient: TensorCore needs native conversion and compute, model-level quality, end-to-end memory, tail latency, and energy evidence.

3. **Serving is a control-plane problem.** Mooncake and Dynamo make KV locality, transfer cost, TTFT, inter-token latency, overload, and request admission first-class. vLLM's current speculative-decoding matrix likewise shows that the best method depends on QPS, model support, and acceptance behavior.

4. **Communication is workload-shaped.** NCCL/RCCL provide topology-native homogeneous collectives; DeepEP treats MoE dispatch/combine, link domain, token skew, low precision, and SM occupation as a distinct problem. TensorCore needs vendor fast paths plus a portable, authenticated, failure-aware contract.

5. **Topology and identity are part of correctness.** Kubernetes DRA models typed devices, capacity, health, taints, and claims. Tailscale grants can enforce deny-by-default network policy and checked authorization tests, but SPIFFE-style workload identity is the relevant model for short-lived, rotated application peer identity.

6. **Claims need benchmark science.** MLPerf's core rules—fixed system boundary, shared implementation, quality and latency constraints, deterministic inputs, and mandatory replicability—are the minimum bar for TensorCore's public comparisons.

## Consequences for this fleet

- The complete topology authority comes first. A benchmark cannot characterize "our system" while the scheduler represents only a subset of computer_mesh and the active/inactive Tailscale profiles.
- The measured Atlas-to-cosbox path is about 103 MB/s over a 1 GbE ceiling. Dynamo explicitly warns that disaggregation without fast KV transfer can lose to colocated serving. Remote prefill/decode therefore stays disabled unless a live A/B admission gate proves at least 15% more SLO-qualified goodput.
- CUDA attention numbers marked projected or measured on fallback kernels must be replaced by native-path assertions and physical results. The same applies to M4/M5 Metal tensor operations and ROCm.
- Apple unified memory, CPU memory tiers, local NVMe, and remote nodes should be costed as one memory hierarchy, but placement must incorporate measured bandwidth, latency, contention, energy, identity, and failure domain.
- MoE expert parallelism belongs on high-bandwidth islands first. A 1 GbE tailnet is useful for control, checkpoint, and coarse-grained work; it is not a credible DeepEP-class data plane.

## Ranked engineering agenda

1. **Evidence and topology foundation:** finish `tensorcore-topology-authority`, add signed topology snapshots, and make every benchmark emit a machine-readable system/result bundle.
2. **Authenticated control and data planes:** finish `tensorcore-transport-identity-auth`; use Tailscale grants/tests for network policy and short-lived workload identities for application authentication.
3. **Adaptive kernel runtime:** build a per-device/per-shape tuning database across Metal, CUDA, HIP, and CPU; require held-out correctness and performance before selecting a schedule.
4. **Native precision frontier:** implement and prove NVFP4/FP8, Metal block-scaled tensor formats, and current AMD low-precision paths with model-quality and energy gates.
5. **End-to-end inference loop:** wire qLLM through stable TensorCore dispatch, then add QPS-aware speculative decoding and real-model serving benchmarks.
6. **Topology-shaped distribution:** replace centralized collectives, add vendor-native paths and MoE shapes, then enable KV disaggregation only on topology classes that pass the admission gate.
7. **Fleet Pareto certification:** publish no cross-system superlative until every eligible backend has physical evidence and at least one end-to-end fleet workload holds a reproducible Pareto lead.

## Research loop

The source and gate ledger expires every 45 days. Each refresh runs the dated ICC query batch, verifies every retained source at its primary URL, adds material new work, retires superseded baselines, and updates blockers only from checked evidence.

```sh
cd ~/Desktop/infinite_context_coder
bin/icc research-search \
  --batch ~/Desktop/tensorcore/docs/research/TENSORCORE_BEYOND_SOTA_QUERIES.txt \
  --top 5 --since 2024 --provider all --format json

cd ~/Desktop/tensorcore
python3 scripts/check_beyond_sota_research.py
python3 scripts/check_beyond_sota_research_selftest.py
```

The ICC academic sweep succeeded across OpenAlex, arXiv, Crossref, and web fallback. Semantic Scholar rate-limited unauthenticated requests; that provider failure is not treated as missing evidence because primary papers and official sources were verified directly.

## Primary-source baseline

- [FlashAttention-4](https://arxiv.org/abs/2603.05451)
- [Nautilus auto-scheduling compiler](https://arxiv.org/abs/2604.14825)
- [VDCores asynchronous GPU execution](https://arxiv.org/abs/2605.03190)
- [NVIDIA NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [Apple Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- [AMD hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/) and [RCCL](https://rocm.docs.amd.com/projects/rccl/en/latest/)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [Mooncake](https://arxiv.org/abs/2407.00079)
- [NVIDIA Dynamo disaggregated serving](https://docs.nvidia.com/dynamo/latest/user-guides/disaggregated-serving)
- [vLLM speculative decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/)
- [NVIDIA NCCL](https://docs.nvidia.com/deeplearning/nccl/)
- [Kubernetes Dynamic Resource Allocation](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/)
- [SPIFFE Workload API](https://spiffe.io/docs/latest/spiffe-specs/spiffe_workload_api/)
- [Tailscale policy syntax](https://tailscale.com/docs/reference/syntax/policy-file)
- [MLPerf Inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
