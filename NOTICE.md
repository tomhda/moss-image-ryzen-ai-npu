# Notice

This project converts MOSS-Image-SFT with a 4-step LoRA to run on the NPU of AMD Ryzen AI processors. It consists of the code at https://github.com/tomhda/moss-image-ryzen-ai-npu and the compiled NPU parts at https://huggingface.co/Tomoali/MOSS-Image-SFT-4-steps-ryzen-ai-npu. It is not an official release of the OpenMOSS team, of the LoRA's author, or of AMD.

## Sources

| Source | Revision | License | Used for |
|---|---|---|---|
| [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) | `f972bd4b4b9e3852133cc5da954d87bc32cd8fb9` | Apache-2.0 | Transformer weights in the NPU parts. The text encoder, VAE, tokenizer and pipeline code are downloaded from this repository at run time and not redistributed |
| [akhaliq/MOSS-Image-SFT-4-steps](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps) | `2ded95c37a029360245926fe72e475e30039bbfb` | Apache-2.0 | `lora.safetensors`, merged into the NPU parts |
| [amd/FLUX.2-klein-4B-amdnpu](https://huggingface.co/amd/FLUX.2-klein-4B-amdnpu) | `45aeffe92ecf5ce0169c511c232c2f100be3cbe1` | Apache-2.0 | The compiled VAE decoder used at 1024², downloaded at setup and not redistributed |
| Ryzen AI Software 1.8.0 (AMD) | 1.8.0 | AMD's license | `onnx_utils` and DynamicDispatch, to build and run the parts. Not included; install it from AMD |

## Changes in the NPU parts

- The LoRA was merged in fp32 into the q, k, v, o and FFN projections of all 42 transformer blocks.
- Each block was split into an attention part and an FFN part, exported to ONNX and converted to AMD DynamicDispatch operators, which hold the weights in BFP16.
- Attention: within each head, the output channels of the q and k projections and the q/k normalization weights were reordered for the interleaved RoPE operator. The text tokens sit in a fixed frame of 192 (512²) or 256 (1024²) rows marked 1 for real tokens and 0 for unused rows; unused rows are zeroed after modulation. Attention is computed twice, the second time with the mark in place of V, and the output is divided by that weight sum so that unused rows do not contribute.
- FFN: the hidden width 8192 was padded with zero weights to 9216, a width the AMD operators support. The part processes 256 rows at a time.
- 512² and 1024² use separate attention parts (sequences of 1216 and 4352 rows); the FFN parts are shared.
- `tools/convert/strategies/*.yaml` are modified from the partition strategies of `ryzenai_onnx_utils` 1.8.0 and keep AMD's copyright notice. Each file lists its changes.

## Changes in the pipeline

- At run time the FFN part fills rows 0–7 and 160–167 of every 256-row tile with rows whose results are discarded, because the NPU occasionally returned wrong values in those rows.
- At 1024² the image is decoded by AMD's compiled FLUX.2 decoder, after the batch-norm inverse and 2×2 unpacking of MOSS-Image-SFT, instead of the decoder of MOSS-Image-SFT.

"AMD" and "Ryzen" are trademarks of Advanced Micro Devices, Inc. They are used here only to describe compatibility and origin.
