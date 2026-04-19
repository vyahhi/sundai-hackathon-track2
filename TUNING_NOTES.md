# Tuning Notes

Measured on the remote RTX A6000 benchmark box.

## Current honest baseline

- Best fair pushed version so far: `50b08b6`
- Score: `319.39 TOPs`
- Per-layer:
  - `attn_to_qkv`: `324.02 TOPs`
  - `attn_to_out`: `300.79 TOPs`
  - `ff_up`: `317.22 TOPs`
  - `ff_down`: `335.54 TOPs`

This version uses:

- The direct INT4 path with `half2` accumulation as the main fast path
- The 4-warp direct kernel for all benchmarked `K=3072` or `N=3072` shapes
- Warp-shuffle loads for `scales_A` inside the 4-warp kernel instead of per-tile shared-memory staging
- Aggressive per-shape offline weight groups:
  - `attn_to_qkv`: `g512`
  - `attn_to_out`: `g256`
  - `ff_up`: `g768`
  - `ff_down`: `g1536`

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

## Fair progress after the `298.25` baseline

These are the main steps that moved the score past `300` on full default benchmark runs.

### Dispatch changes

- `e8e7d07`:
  - `298.25 TOPs`
  - Mixed weight groups plus separate activation/weight group handling in the runtime

- `9acc711`:
  - `303.27 TOPs`
  - Expanded the 4-warp direct kernel to all `N=3072` shapes
  - Big gain on `ff_down`, but not yet enough to cleanly lift every `K=3072` layer

- `7097590`:
  - `306.57 TOPs`
  - Narrower split: keep the 4-warp path on `N=3072` shapes and add it to `attn_to_qkv`, but not `ff_up`
  - This was the first clean `306+` fair full-run result

- `df40404`:
  - `309.21 TOPs`
  - With the stronger weight map below, using the 4-warp path for all `K=3072` and `N=3072` shapes became net positive

### Weight-group changes

- `ef52566`:
  - `308.25 TOPs`
  - More aggressive per-shape weight groups:
    - `attn_to_qkv=g512`
    - `attn_to_out=g256`
    - `ff_up=g768`
    - `ff_down=g1024`

- `50b08b6`:
  - `319.39 TOPs`
  - Push `ff_down` further to `g1536`
  - This still clears correctness, but only barely:
    - `ff_down cosine = 0.977204`
    - threshold is `0.977`

### Kernel-internal change that mattered most

- `48abb32`:
  - `316.68 TOPs`
  - Replace shared-memory `scales_A` loads in the 4-warp direct kernel with warp-local loads plus `__shfl_sync`
  - This removes a per-`kt` shared-memory barrier for the 4-warp path
  - This was the biggest single fair kernel win after the initial mixed-group work

## Recent negative findings after `306+`

These are newer dead ends or marginal directions that should not be repeated without a different underlying idea.

- Broad 4-warp expansion before the later kernel/weight-map wins:
  - Short run looked strong at about `307.96 TOPs`
  - Full run only landed around `303.58 TOPs`
  - Conclusion: short runs were overestimating this variant at that stage

- `ff_up` at `g1024`:
  - Passed correctness, but only narrowly:
    - `ff_up cosine = 0.978154`
    - threshold is `0.978`
  - Short run landed around `319.05 TOPs`
  - Worse than the current best pushed baseline
  - Conclusion: not worth the reduced correctness margin

- `attn_to_qkv` at `g768`:
  - Passed correctness, but also narrowly:
    - `attn_to_qkv cosine = 0.989350`
    - threshold is `0.989`
  - Short run landed around `320.21 TOPs`
  - Too little upside for how close it runs to the floor
  - Conclusion: keep `attn_to_qkv` at `g512`

- More aggressive weight grouping in general:
  - The current best build is already using most of the safe headroom
  - The remaining margin is especially tight on `attn_to_out`, `ff_up`, and `ff_down`
  - Future gains are more likely to come from kernel changes than from another round of coarse grouping

## What looks most promising next

- `attn_to_out` is now the laggard layer in the best build
- The most credible next fair wins are kernel-side, not quantization-side:
  - Improve the 4-warp direct kernel for the `3072 x 3072` case specifically
  - Reduce remaining synchronization or scale-lookup overhead in the direct path
  - Try a shape-specific `attn_to_out` path that is different from the generic all-shapes 4-warp kernel

## Benchmark structure reminder

- `benchmark.py` times activation quantization and GEMM separately
- Offline weight quantization in `quantize.py` is not timed
- GEMM timing starts after `sol_act_p, sol_act_s = sol.quantize_int4(...)`
- Repacking work moved into timed GEMM directly hurts the score
- Repacking work moved into offline weight quantization is fair if the format still matches the kernel
