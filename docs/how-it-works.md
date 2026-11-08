# How it works

Technical notes on fitting MOSS-Image-SFT to the NPU of Ryzen AI 300-series processors. Numbers come from a Ryzen AI 7 PRO 350 with Ryzen AI Software 1.8.0.

## The model

The MOSS-Image-SFT transformer has 42 blocks of width 3072, 24 attention heads of 128 channels and an FFN of width 8192 with the activation `4·tanh(g/4)·sigmoid(g)`. The 4-step LoRA (rank 128) adds to the q, k, v, o and both FFN projections. The text encoder is Qwen3.5 and gives 4096-channel features per token. A 512² image is 1024 latent tokens (32×32), a 1024² image 4096 (64×64); text and image tokens go through the blocks together.

## Why DynamicDispatch

Ryzen AI Software 1.8.0 offers two ways to put such a model on the NPU:

- **VAIML**: the VitisAI execution provider compiles a general ONNX graph.
- **DynamicDispatch (DD)**: AMD's `onnx_utils` finds patterns in the graph (matrix products, attention with RoPE, RMSNorm, elementwise operations) and replaces them with hand-built BFP16 operators. AMD's SD 3.5 and FLUX.2-klein NPU releases use this path.

On block 0 at 512², DD took 0.7–0.8× the time of VAIML for the same part and compiled in under 2 seconds instead of 24–140 seconds. VAIML had about half the error on q/k and attention (0.6% against 1.2% for q/k), the FFN was about the same on both. DD was chosen.

A DD operator runs only if the instruction stream for its exact shape ships with Ryzen AI Software; there is no tool to make new ones. The shipped shapes are those of AMD's own models, so the model was rearranged to fit them:

| Need | Shipped shape used |
|---|---|
| Attention at 512² | sequence 1216 = 1024 image + 192 text rows |
| Attention at 1024² | sequence 4352 = 4096 image + 256 text rows |
| FFN | matrix products of 256 rows, 3072 → 9216 and 9216 → 3072 |
| RoPE | interleaved channel pairs |

## Two parts per block

Each block is two ONNX models, each mostly one DD region:

- **A (attention)**: normalization and modulation, the q/k/v projections, q/k normalization, RoPE and attention. 18–19 small operators stay on the CPU (normalization, modulation, the division described below).
- **T (FFN tile)**: for 256 rows, the o projection, residual, normalization and modulation, the FFN and its residual. Parts of the FFN activation stay on the CPU.

A runs once per block and step; T runs as many times as needed to cover all rows. The 42 blocks are 84 NPU sessions. At 1024², A is a separate part with the longer sequence; T is the same file for both sizes.

### Text of any length up to the frame

The text tokens sit in a fixed frame of 192 rows (512²) or 256 rows (1024²). Real tokens are marked 1, unused rows 0. After modulation the unused rows are multiplied by the mark, so they are exactly 0 going into the projections. They still take part in softmax, so attention is computed twice: once normally (y), and once with the mark in place of V, which gives the share of attention weight that fell on real tokens (a). The output is y ÷ a per query and head. On block 0, a stayed at 0.30 or above.

T only processes real rows, so its work grows with the prompt: at 512², up to 176 tokens fit in 5 tiles and 177–192 need 6; at 1024², up to 224 tokens fit in 18 tiles and 225–256 need 19.

### FFN width

The FFN width 8192 has no shipped operator, 9216 does. The up and gate projections are padded to 9216 outputs with zero weights, and the down projection to 9216 inputs.

### Discarded rows

When many sessions were loaded, T occasionally returned values about 1% off in the first 8 rows of a tile, and sometimes in row 160. Repeated runs with the same seed then gave different images. The cause is unknown. Rows 0–7 and 160–167 of every tile are now filled with dummy rows and their results discarded; each tile carries 240 real rows. Since every FFN row is computed independently, the real rows are unaffected. After the change, 4,000 calls of one layer and 30 passes through layers 21–41 showed no differing rows, and repeated generations are bit-identical.

## Weights and precision

The LoRA is merged into the weights in fp32, then `onnx_utils` packs the fp32 weights into BFP16 directly (no bf16 step). BFP16 groups 8 values under one 8-bit exponent with an 8-bit mantissa each, so all but the largest value in a group are coarser than bf16. Activations inside the DD operators are BFP16 too.

Error of one block's output against fp32, with the block's real input at 512²:

| Block | NPU vs fp32 | Official bf16 vs fp32 |
|---|---|---|
| 0 | 1.2% | 0.5% |
| 20 | 0.20–0.25% | 0.19–0.33% |
| 41 | 3.2–6.6% | 1.1–2.2% |

Chained through 42 blocks, these differences grow. The transformer's output at each step (the velocity, which the scheduler uses to move the latent) differs from the official bf16 run by 26–65% at 512² and 32–64% at 1024²; the fp32 run differs from the official bf16 run by 1.9–9.6%. In fp32 alone, adding 0.3% noise at block 0 moves the velocity by 7–15%: the later blocks amplify small changes. The images are complete, but their composition differs from the official pipeline for the same seed.

## Processes

`gen.ps1` runs three processes in turn: text (CPU), transformer (NPU), decoder. The text and decoder stages use the text environment; the transformer and the NPU decoder use the Ryzen AI environment.

Freeing DD sessions and creating them again in the same process crashed with an access violation, so when the blocks do not fit in one process, each step and each group of blocks runs in its own child process that creates its sessions once. At 512² the transformer keeps all 84 sessions in one process when 15 GB is free and otherwise uses two groups of 21 blocks; at 1024² it uses three groups of 14 blocks (one A and T pair takes about 0.75 GB at 1024², counting the first run).

CPU-side libraries use one thread each (PyTorch, the text encoder, the CPU decoder), so a run keeps less than one logical processor busy on average.

Windows takes a kernel live dump when a single NPU call runs longer than about 3 seconds. Each DD call here is much shorter, and no live dumps were recorded in the test runs.

## The 1024² decoder

MOSS-Image-SFT uses the FLUX.2 VAE design. At 1024² the decoder is AMD's compiled one from `amd/FLUX.2-klein-4B-amdnpu`, fed the latents after MOSS's batch-norm inverse and 2×2 unpacking ([1, 32, 128, 128]). Against the MOSS decoder on the CPU, over 6 images, PSNR is 39.3–46.3 dB. One decode takes about 1 second on the NPU and about 46 seconds on one CPU thread. AMD's compiled decoder is for 1024² only.

This decoder crashes during process shutdown. The decoder stage writes the images and a completion file, then ends its own process with `TerminateProcess`; `gen.ps1` checks the completion file.
