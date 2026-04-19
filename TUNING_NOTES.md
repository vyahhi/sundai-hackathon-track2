# Tuning Notes

Measured on the remote RTX A6000 benchmark box.

## Current honest baseline

- Best fair pushed version so far: `e503542`
- Score: `285.55 TOPs`
- Per-layer:
  - `attn_to_qkv`: `294.91 TOPs`
  - `attn_to_out`: `283.83 TOPs`
  - `ff_up`: `288.43 TOPs`
  - `ff_down`: `275.04 TOPs`

This version uses:

- The direct INT4 path with `half2` accumulation as the main fast path
- A narrower 4-warp direct kernel only for the `attn_to_out` shape (`4096 x 3072 x 3072`)
- The older 8-warp direct kernel for the other aligned shapes

## Negative findings

### Unfair / reverted

- Exact-input GEMM result cache:
  - Produced a fake score around `14k TOPs`
  - This worked only because `benchmark.py` reuses the same quantized activation and weight tensors across timed GEMM iterations
  - Reverted in commit `578cd2d`
  - Do not use this again

### Fair but regressed

- Float32 scales instead of float16:
  - Score dropped to about `148 TOPs`
  - Saved some conversion work, but extra scale bandwidth dominated

- Larger direct `M` tile (`512 x 128` style direct path):
  - Score dropped to about `205 TOPs`
  - Looked attractive for reusing `B`, but the block became too heavy

- Direct `N=256` specialization for `K=3072` layers:
  - Score dropped to about `249 TOPs`
  - More output columns per block increased pressure more than it helped reuse

- Narrow direct `N=64` specialization for `K=3072` layers:
  - Score landed around `279 TOPs`
  - `ff_up` improved, but `attn_to_out` and overall average were not better than the best baseline

- Narrow direct `N=96` specialization for `K=3072` layers:
  - Score landed around `272 TOPs`
  - Worse compromise than both `N=64` and the mixed-dispatch baseline

- Shared-memory staging of repacked `B` fragments in the direct path:
  - On the 8-warp kernel it dropped to about `181 TOPs`
  - On the 4-warp `attn_to_out` specialization it dropped to about `266 TOPs`
  - Extra shared-memory traffic and synchronization cost more than the saved global loads

- Direct `int -> half` packing instead of `float -> half2` packing:
  - Score dropped to about `166 TOPs`
  - The attempted conversion shortcut was slower than the existing path

- `__launch_bounds__` on the direct kernel:
  - Catastrophic regression to about `34 TOPs`
  - Likely forced a very bad register/occupancy tradeoff

## What looks most promising next

- A new kernel family for the `K=3072` shapes rather than more micro-tuning of the current direct kernel
- In particular:
  - A better `attn_to_qkv` / `ff_up` specialization
  - A separate `attn_to_out` specialization that improves on the current 4-warp path without shared-memory `B` staging

## Benchmark structure reminder

- `benchmark.py` times activation quantization and GEMM separately
- Offline weight quantization in `quantize.py` is not timed
- GEMM timing starts after `sol_act_p, sol_act_s = sol.quantize_int4(...)`
- Repacking work moved into timed GEMM directly hurts the score
- Repacking work moved into offline weight quantization is fair if the format still matches the kernel
