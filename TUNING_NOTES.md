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

## Recent fair iterations after `e503542`

These runs were measured fairly on the benchmark box, but they are still work-in-progress results rather than a new pushed kernel baseline.

### Weight-only coarser groups

The benchmark does not time offline weight quantization, so trying different weight group sizes in `quantize.py` is fair as long as the runtime kernel understands the resulting scale layout.

- All weights at `g128`:
  - Score reached about `296.57 TOPs`
  - Per-layer:
    - `attn_to_qkv`: `302.39 TOPs`
    - `attn_to_out`: `300.79 TOPs`
    - `ff_up`: `295.20 TOPs`
    - `ff_down`: `287.88 TOPs`

- Mixed weights with `attn_to_qkv` / `attn_to_out` at `g256` and `ff_up` / `ff_down` at `g512`:
  - Score reached about `298.75 TOPs`
  - Per-layer:
    - `attn_to_qkv`: `304.43 TOPs`
    - `attn_to_out`: `298.41 TOPs`
    - `ff_up`: `298.70 TOPs`
    - `ff_down`: `293.48 TOPs`

- More aggressive per-shape mix with `attn_to_qkv=g512`, `attn_to_out=g256`, `ff_up=g768`, `ff_down=g1024`:
  - Score reached about `299.50 TOPs`
  - Per-layer:
    - `attn_to_qkv`: `301.59 TOPs`
    - `attn_to_out`: `297.23 TOPs`
    - `ff_up`: `299.59 TOPs`
    - `ff_down`: `299.59 TOPs`

### Kernel support needed for the weight-group experiments

The original runtime assumed the activation and weight group sizes matched. That is too restrictive once weight groups get coarser.

- Add separate logical group counts for `A` and `B`:
  - `num_groups_A = scales_A.size(1)`
  - `num_groups_B = scales_B.size(1)`

- In the direct kernels, index `scales_B` using a weight-side stride:
  - `b_scale_stride = K / num_groups_B`
  - Use `kt / b_scale_stride` rather than assuming the activation-side grouping

- The fallback MMA path also needs separate `A` and `B` group indexing for correctness

This direction looks correct and helped unlock the better mixed-group results above.

### Small but real improvement from hoisting `scales_B`

After the mixed-group changes, hoisting the `scales_B` load out of the inner per-`kt` loop in the direct kernels helped a little more.

- Best short fair run from this direction reached about `299.80 TOPs`
- Per-layer:
  - `attn_to_qkv`: `300.39 TOPs`
  - `attn_to_out`: `301.99 TOPs`
  - `ff_up`: `299.59 TOPs`
  - `ff_down`: `297.23 TOPs`

This is the closest fair result so far to the `300+` target, but it still did not clear the current target cleanly.

### Dead end: 2-warp `attn_to_out` specialization

Trying to shrink the `attn_to_out` specialization from 4 warps to 2 warps was not viable.

- First failure mode:
  - `attn_to_out` produced `cosine=nan`
  - One bug was that `shared_scales_B` loading assumed 128 threads and needed a strided load for 64 threads

- After fixing the shared-scale loading:
  - `attn_to_out` still produced `NaN`s
  - So the problem was deeper than the obvious thread-count bug

- Conclusion:
  - Drop the 2-warp path for now
  - It is not a quick win and is likely to waste more time than it saves

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
