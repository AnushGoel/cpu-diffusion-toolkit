# Three Hugging Face Models on a CPU Budget

Text-to-image, image-to-video, and text classification, using only non-gated Hugging Face models, running end to end on a free Google Colab runtime with no GPU. Each task is timed, and all three are exposed through a single Gradio interface.

Coursework for AAI 6640 (Applied Deep Learning), Assignment #2.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/blob/main/AAI6640_Assignment2_CPU.ipynb)

---

## The constraint

The interesting part of this project is not the models. It is that none of the obvious choices fit.

A free Colab CPU runtime gives you roughly 2 vCPU and 12.7 GB of RAM. SDXL-Turbo needs about 10 GB for the UNet alone in fp32 and will OOM the runtime. LTX-Video pairs a 2B diffusion transformer with a 4.7B T5 text encoder and would take hours per clip. Stable Video Diffusion, the obvious image-to-video pick, is gated and so fails the assignment requirement outright.

Everything below was selected for the smallest compute that still does the task honestly.

## Models

| Task | Model | Params | License | Gated |
|---|---|---|---|---|
| Text to Image | [`stabilityai/sd-turbo`](https://huggingface.co/stabilityai/sd-turbo) | 865M UNet | `sai-nc-community` | No |
| Image to Video | [`ByteDance/AnimateDiff-Lightning`](https://huggingface.co/ByteDance/AnimateDiff-Lightning) on [SD 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | 860M + 453M | `creativeml-openrail-m` | No |
| Text Classification | [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions) | 125M | MIT | No |

Supporting component: [`madebyollin/taesd`](https://huggingface.co/madebyollin/taesd), a distilled stand-in for the Stable Diffusion VAE.

### Why these three

**SD-Turbo** is Stable Diffusion 2.1 distilled with Adversarial Diffusion Distillation. It reaches a usable image in a single denoising step instead of 25 to 50. On a GPU that is convenience. On a CPU it is the difference between 11 seconds and 5 minutes. Note that `guidance_scale` must be `0.0`, since the model was distilled without classifier-free guidance.

**AnimateDiff-Lightning** inserts temporal self-attention layers (motion modules) between the spatial layers of a frozen SD 1.5 UNet, then distills the result down to a 4-step schedule. Original AnimateDiff needs around 25 steps with CFG, which is 50 UNet passes per frame. This needs three, with no CFG.

**GoEmotions RoBERTa** is multi-label rather than multi-class. The head applies an independent sigmoid to each of 28 logits instead of a softmax across them, so scores do not sum to 1 and a sentence can be *admiration* and *pride* and *excitement* at once. The notebook passes `top_k=None` and exposes the decision threshold as a slider.

### How the image conditioning works

AnimateDiff is natively text-to-video. To drive it from a still, the pipeline used here is `AnimateDiffVideoToVideoPipeline`, fed the input image repeated N times as the input clip with `strength < 1.0`. The pipeline noises that clip partway and denoises from there, so the output keeps the composition and subject of the input while the temporal attention layers introduce coherent motion. Setting `strength=1.0` makes the input image stop mattering, which is a useful way to demonstrate that the conditioning is real.

Tuning: 0.5 to 0.75 is the usable range. Too low and the clip barely moves. Too high and it drifts away from your image.

---

## The four CPU optimisations

Together these are the difference between "runs" and "does not finish".

1. **Step count over model size.** Both diffusion models are distilled to 1 to 4 steps. Runtime on CPU is close to linear in step count, so this dominates every other choice.
2. **A tiny VAE.** TAESD replaces Stable Diffusion's variational autoencoder. The real decoder is one of the most expensive parts of the pipeline on CPU, and decoding eight video frames with it takes longer than denoising them. TAESD is roughly three orders of magnitude cheaper with little visible difference at these resolutions.
3. **No classifier-free guidance.** `guidance_scale` of 0.0 or 1.0 means one UNet forward pass per step rather than two. Both models were distilled without CFG, so this is free. It halves the work.
4. **fp32, not fp16.** PyTorch has no fast fp16 convolution kernels on CPU, so casting down makes things slower. fp16 is used only when a GPU is detected. See [Caveats](#caveats) for an exception on newer server CPUs.

---

## Results

Measured on a Colab host with 24 threads and 50.5 GB RAM, PyTorch 2.14.1+cpu, Python 3.13.16, fp32.

| Step | Time |
|---|---|
| Task 1 model load | 29.2 s |
| Task 1 inference, 384×384, 1 step | 10.7 s / 10.6 s / 17.4 s (avg 12.9 s) |
| Task 2 model load | 51.2 s |
| Task 2 inference, 8 frames | 301.1 s |
| Task 3 model load | 8.1 s |
| Task 3 inference, single sentence | 0.30 s to 0.70 s |

Step count against wall-clock time for SD-Turbo, which is the reason a distilled model was chosen:

| Steps | Time |
|---|---|
| 1 | 10.5 s |
| 2 | 26.8 s |
| 4 | 37.2 s |

Sample classification output:

| Text | 1st | 2nd | 3rd |
|---|---|---|---|
| Thank you so much, this genuinely made my week. | gratitude (0.991) | approval (0.014) | joy (0.014) |
| Hmm, I am not sure that is how the loss function is supposed to behave. | confusion (0.909) | neutral (0.092) | approval (0.037) |
| The flight was delayed four hours and nobody bothered to tell us anything. | disappointment (0.499) | neutral (0.289) | annoyance (0.262) |
| I cannot believe we finally shipped this. I am so proud of the whole team. | admiration (0.343) | pride (0.310) | excitement (0.110) |

---

## Running it

### Colab

Open the notebook, set `Runtime → Change runtime type → CPU` (or **T4 GPU** if one is available, which the notebook detects and uses automatically), then `Runtime → Run all`.

Do not use a TPU runtime. PyTorch cannot reach a TPU without `torch_xla` and diffusers will not use it regardless, so a TPU runtime just runs on that VM's CPU while consuming TPU quota.

First run downloads roughly 6 GB of weights.

### Locally

```bash
pip install diffusers transformers accelerate safetensors huggingface_hub
pip install gradio imageio imageio-ffmpeg psutil
jupyter notebook AAI6640_Assignment2_CPU.ipynb
```

### Quality switch

`FAST_MODE = True` in the first cell uses the smallest settings that still produce output, for verifying the notebook runs end to end. Set it to `False` for anything you intend to show. See [Caveats](#caveats) for why this matters more than it looks.

---

## Gradio interface

Three tabs in one `gr.Blocks` app, each reporting its own measured runtime.

- **Tab 1** generates an image, with a *Send to Task 2* button that pipes it straight into the video tab.
- **Tab 2** animates a conditioning image, with sliders for resolution, frame count, steps and strength.
- **Tab 3** classifies text, with a threshold slider so you can watch labels cross the decision boundary.

The `get_*` helpers load pipelines lazily and evict the other diffusion pipeline on a tab switch, which keeps peak memory to one pipeline at a time. The first click on each tab pays the model load; later clicks do not.

`demo.queue().launch(share=True)` prints a public `*.gradio.live` link that stays alive while the cell runs.

---

## Repository layout

```
.
├── AAI6640_Assignment2_CPU.ipynb          # the CPU notebook (start here)
├── AAI6640_Assignment2_HuggingFace.ipynb  # GPU variant: SDXL-Turbo + LTX-Video
├── requirements.txt
├── outputs/
│   ├── task1_*.png                        # generated images
│   ├── task2_output.mp4 / .gif            # generated clip
│   ├── task2_comparison.png               # input still beside output frames
│   └── timings.csv                        # measured runtimes
└── README.md
```

The GPU notebook uses `stabilityai/sdxl-turbo` and `Lightricks/LTX-Video` instead. Better output, but it will crash on a CPU runtime.

---

## Caveats

**Resolution is not a free knob.** SD 1.5 and the AnimateDiff motion module were both trained at 512×512. At 256×256 the latent is 32×32 and the model falls apart into incoherent blotches. This is the single most common way to get a bad result from this notebook, and it is why `FAST_MODE` output should never go in a report. Use 512 for video, or 384 at a push.

**bfloat16 may beat fp32 on newer hardware.** Optimisation 4 above assumes a CPU without AMX. Sapphire Rapids and later have hardware bfloat16, where `DTYPE = torch.bfloat16` can be meaningfully faster. Worth benchmarking a single conv layer before assuming.

**Batched classification needs an explicit batch size.** Passing a list to a `transformers` pipeline does not batch unless you also pass `batch_size=N`. Without it, per-sentence throughput can look worse than single-sentence timing.

**Timings are hardware-dependent.** Colab host specs vary considerably between sessions. The numbers above came from a 24-thread host; a 2 vCPU host will be several times slower. Always record the hardware alongside the duration.

---

## License

Notebook code is MIT. The models carry their own licenses, listed in the table above. `sai-nc-community` is non-commercial research only.

## Acknowledgements

Stability AI (SD-Turbo, SD 2.1), ByteDance (AnimateDiff-Lightning), Yuwei Guo et al. (AnimateDiff), Sam Lowe (GoEmotions fine-tune), Google Research (GoEmotions dataset), Ollin Boer Bohan (TAESD), and the Hugging Face `diffusers` and `transformers` teams.
