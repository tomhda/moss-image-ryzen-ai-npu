# MOSS-Image-SFT 4-step on the AMD Ryzen AI NPU

Text-to-image with [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) and the 4-step LoRA [akhaliq/MOSS-Image-SFT-4-steps](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps), with the diffusion transformer running on the NPU of an AMD Ryzen AI 300-series processor (XDNA 2). 512×512 and 1024×1024. The integrated GPU is not used. Unofficial; see [NOTICE.md](NOTICE.md).

The NPU parts are on Hugging Face: [Tomoali/MOSS-Image-SFT-4-steps-ryzen-ai-npu](https://huggingface.co/Tomoali/MOSS-Image-SFT-4-steps-ryzen-ai-npu). `setup.ps1` downloads them.

![Four images generated with this repository](docs/images/samples.jpg)

512² (left two, `examples/repro_512.json`) and 1024² (right two, from `examples/repro_1024.json`), generated with this repository.

## What runs where

| Stage | Runs on | Notes |
|---|---|---|
| Text encoder (Qwen3.5, bf16) | CPU, 1 thread | Prompts of 1–192 tokens at 512², 1–256 at 1024², counting the chat template |
| Diffusion transformer (42 blocks, 4 steps) | NPU | Matrix products, attention and RoPE on the NPU in BFP16. Normalization, modulation, residual adds and part of the FFN activation stay on the CPU |
| VAE decoder at 512² | CPU, 1 thread | The decoder of MOSS-Image-SFT, fp32 |
| VAE decoder at 1024² | NPU | AMD's compiled FLUX.2 decoder from [amd/FLUX.2-klein-4B-amdnpu](https://huggingface.co/amd/FLUX.2-klein-4B-amdnpu). `-Vae cpu` uses the MOSS decoder instead |

Each stage runs in its own process. The transformer blocks were converted with the DynamicDispatch tools of Ryzen AI Software 1.8.0, the same tools AMD uses for its FLUX.2-klein and SD 3.5 NPU releases. How the model was fitted to them is in [docs/how-it-works.md](docs/how-it-works.md).

## Verified configuration

| | |
|---|---|
| Processor | AMD Ryzen AI 7 PRO 350 (8 cores, 16 threads) |
| Memory | 32 GB |
| OS | Windows 11 Home, build 26200 |
| NPU driver | 32.0.203.329 |
| Ryzen AI Software | 1.8.0 |

All results below come from this one machine. Other XDNA 2 processors, machines with 16 GB of memory and other NPU driver versions have not been tried.

## Install

Requirements:

- [Ryzen AI Software 1.8.0](https://ryzenai.docs.amd.com/en/1.8/inst.html) with its NPU driver, installed to the default folder `C:\Program Files\RyzenAI\1.8.0` (or pass `-RyzenPath`)
- Miniforge or another conda, PowerShell 7, Git
- About 50 GB of free disk for 512² only, 54 GB for both sizes (see [Disk](#disk))

```powershell
git clone https://github.com/tomhda/moss-image-ryzen-ai-npu
cd moss-image-ryzen-ai-npu
pwsh -NoProfile -File .\setup.ps1 -Size 512
pwsh -NoProfile -File .\doctor.ps1 -Size 512
```

`-Size` is `512`, `1024` or `all`. `setup.ps1`:

1. Creates the conda environment `mossnpu-npu` from the `env.yaml` of Ryzen AI Software, installs all the AMD wheels in its folder, then `environments/npu-extra.txt`.
2. Creates `mossnpu-text` (Python 3.12.16, PyTorch 2.15.0 CPU, `environments/text.txt`).
3. Downloads the NPU parts for the chosen size from Hugging Face at a pinned commit and checks every file against the SHA256 in `manifest.json`.
4. Downloads the files of MOSS-Image-SFT that the pipeline needs (text encoder, transformer, VAE, tokenizer, scheduler) into `models\MOSS-Image-SFT`, and AMD's FLUX.2 decoder into `models\klein`, both at pinned commits.
5. Writes `configs\local.json`, which `gen.ps1` and `doctor.ps1` read.

To use copies you already have, pass `-MossPath` and `-KleinPath`; a folder that already has `transformer` and `text_encoder` is used as it is. `-NpuEnv` and `-TextEnv` change the environment names. You do not activate either environment; the scripts call their Python directly.

`doctor.ps1` prints the processor, memory, NPU driver, configured paths, package versions and the manifest check, then runs the first transformer block once on the NPU and checks that the output is finite. When everything works it prints `"ok": true` and `probe exit 0`.

## First image

```powershell
pwsh -NoProfile -File .\gen.ps1 -Items .\examples\one.json -Out .\outputs\first -Size 512
```

The items file is a JSON list:

```json
[
 {"id": "short-mug", "prompt": "A white coffee mug with the word 'HI' printed on it.", "seed": 7}
]
```

The image is written to `<Out>\<id>_s<seed>.png`. The same folder gets the final latents (`latents\`), the text-encoder outputs (`cache\`, reused when a prompt repeats) and `gen_summary.json` with wall and CPU time per stage.

| Option | Values |
|---|---|
| `-Size` | `512` (default) or `1024` |
| `-Vae` | `cpu` or `npu`. Default: `cpu` at 512², `npu` at 1024². The NPU decoder only works at 1024² |
| `-TextThreads`, `-VaeThreads` | CPU threads for the text encoder and the CPU decoder. Default 1 |

Exit codes: `0` all images done, `3` some prompts were longer than the limit and skipped (`SKIP id=... tokens=...`; the others are done), `2` bad input or the output folder is in use by another run, anything else a failure. A run never reuses images from an earlier run in the same folder.

Put several prompts in one items file: loading the text encoder and creating the NPU sessions happen once per run, so the time per image drops as the list grows.

## Measured

On the verified machine, with a fresh clone and environments created by `setup.ps1`, default options. Times include loading the models and creating the NPU sessions. CPU time is user + system time of all processes of the run.

| Run | Text | Transformer | Decoder | Total | Per image | CPU time / wall time |
|---|---|---|---|---|---|---|
| 512², 1 image (`one.json`) | 44.8 s | 153.5 s | 23.7 s (CPU) | 221.9 s | 221.9 s | 0.61 |
| 512², 2 images (`repro_512.json`) | 44.9 s | 136.0 s | 29.4 s (CPU) | 210.3 s | 105.2 s | 0.60 |
| 1024², 1 image (`one.json`) | 36.4 s | 228.7 s | 11.4 s (NPU) | 276.6 s | 276.6 s | 0.65 |
| 1024², 6 images (`repro_1024.json`) | 78.0 s | 503.2 s | 15.8 s (NPU) | 597.0 s | 99.5 s | 0.74 |

- CPU time / wall time is the average number of busy logical processors, out of 16.
- Integrated GPU: 0% on every engine for the processes of a 1024² run, read once per second from the Windows `GPU Engine` counters per process. The NPU appears in the same counters as a separate adapter with only a Compute engine.
- Lowest free physical memory during the 1-image runs, sampled every 5 seconds: 2.7 GB at 512², 3.9 GB at 1024².
- The transformer stage loads the 42 blocks as NPU sessions, about 0.35 GB each at 512² and 0.75 GB at 1024². At 512² it holds all 42 in one process when 15 GB is free and otherwise splits them into two processes of 21; at 1024² it always uses three processes of 14, one after another. Before creating the sessions it checks free memory and stops with a message if it is too low.
- Runs with the same inputs give bit-identical latents and PNGs.

### Disk

| | 512² | 1024² | Both |
|---|---|---|---|
| NPU parts (Hugging Face) | 12.9 GB | 13.1 GB | 17.0 GB |
| MOSS-Image-SFT files | 27.2 GB | 27.2 GB | 27.2 GB |
| AMD FLUX.2 decoder | 0.1 GB | 0.1 GB | 0.1 GB |
| Two conda environments | 8.7 GB | 8.7 GB | 8.7 GB |

The LoRA file is not downloaded: its weights are merged into the NPU parts. It is needed only to rebuild the parts.

## Differences from the official pipeline

- **Composition.** The same prompt and seed give a different image from the official bf16 pipeline. The NPU computes the matrix products in BFP16 (8 values share one exponent); each block's output differs from fp32 by 0.2–8%, and the later blocks of this model amplify small differences. The same happens between fp32 and the official bf16 run. Images are complete and not broken.
- **Text in images.** Results from the prompts in `examples/text_lengths.json` (512²) and `examples/repro_1024.json` (1024²), seed 7 unless noted:

  | id | Tokens | 512² | 1024² |
  |---|---|---|---|
  | short-mug | 24 | HI | HI |
  | short-stop | 19 | STOP | — |
  | mid-books | 89 | "BOOKSS & TEA" (wrong; official 512² correct) | — |
  | mid-station | 80 | 東京 12:30 | "東 京京 12:30" (wrong; official 1024² correct) |
  | long-market | 131 | NIGHT MARKET 2026 | NIGHT MARKET 2026 |
  | long-desk | 140 | 会議 10:00 | — |
  | max-magazine | 190 | all four text lines correct | headline split into "ISLAND / ICLAND ESCAPES" (official 1024² also wrong) |
  | max-map | 174 | ISLE OF MIST | — |
  | cafe-board (seed 1234) | 41 | — | COFFEE 450円 |
  | long-hd | 245 | — | all four text lines correct |

- **1024² decoder.** AMD's NPU decoder and the MOSS decoder on the CPU, given the same latents of 6 images, agree at PSNR 39.3–46.3 dB. This compares the two decoders, not the official image.

## Rebuilding the parts

`tools/convert/convert_all.py` builds all 126 parts (attention for 512², attention for 1024², FFN tiles shared by both, 42 layers each) from MOSS-Image-SFT and the LoRA file. For each part it exports the block from PyTorch, fixes the shapes, partitions it with AMD's `onnx_utils` and the strategies in `tools/convert/strategies/`, and compiles it on the NPU.

```powershell
hf download akhaliq/MOSS-Image-SFT-4-steps lora.safetensors --revision 2ded95c37a029360245926fe72e475e30039bbfb --local-dir .\models\lora
& <mossnpu-text>\python.exe -u .\tools\convert\convert_all.py --moss .\models\MOSS-Image-SFT --lora .\models\lora\lora.safetensors `
    --out .\rebuild --npu-python <mossnpu-npu>\python.exe --ryzen "C:\Program Files\RyzenAI\1.8.0"
```

The finished parts are in `rebuild\final` (same layout as the Hugging Face repository); point `parts_path` in `configs\local.json` there. On the verified machine all 126 took 33 minutes. Images generated with rebuilt parts were bit-identical to those from the downloaded parts. The compiled files themselves differ by 20 bytes in `.ddbundle` and `.state`, which record the folder they were compiled in.

## Repository layout

| Path | |
|---|---|
| `gen.ps1`, `setup.ps1`, `doctor.ps1` | Entry points (PowerShell 7) |
| `src/` | Text, transformer and decoder stages |
| `tools/convert/` | Rebuilding the NPU parts |
| `tools/download_models.py`, `tools/make_manifest.py` | Download with hash check; manifest of the Hugging Face files |
| `configs/example.json` | Pinned revisions and the shape of `local.json` |
| `environments/` | Pinned packages of the two environments |
| `examples/` | Prompt lists used above, and prompts at the token limits |
| `validation/` | Bit-level comparison of two output folders, manifest check |
| `docs/how-it-works.md` | Technical notes |

## License

Apache-2.0 ([LICENSE](LICENSE)). MOSS-Image-SFT, the 4-step LoRA and AMD's FLUX.2-klein NPU release are Apache-2.0. Ryzen AI Software is installed from AMD under its own license and is not included. See [NOTICE.md](NOTICE.md).
