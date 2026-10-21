# Running Text-to-Image, Image-to-Video and Text Classification Models Under a CPU-Only Compute Budget

## Abstract

This report documents the selection, deployment and measurement of three open-access Hugging Face models — a text-to-image diffusion model, an image-to-video diffusion model, and a text classifier — under the constraint of a CPU-only Google Colab runtime. The constraint is treated as the design problem rather than an inconvenience, because the obvious candidate for each task either exceeds available memory or is access-gated. Four optimisations (step-count distillation, a distilled autoencoder, removal of classifier-free guidance, and precision selected by measurement) reduce the complete three-task pipeline to nine minutes of wall-clock time and 6.4 GB of cached weights. A regression over step count shows that distillation alone accounts for a 15× reduction in text-to-image latency, and a resolution comparison identifies the latent dimension below which the image-to-video model fails structurally.

---

## 1. Introduction

Diffusion models are conventionally evaluated on accelerators, where model capacity is the binding constraint and sampling cost is a secondary concern. Removing the GPU inverts that cost model. Arithmetic becomes the scarce resource, and architectural decisions that are invisible on a datacentre card — how many denoising steps a model needs, whether guidance doubles the forward passes, how expensive the decoder is — become the factors that decide whether a task completes at all.

Three candidate models were rejected before the final selection, each for a reason that illustrates the constraint. SDXL-Turbo carries a 2.6B-parameter UNet, roughly 10.4 GB in fp32, which exceeds the memory of a free-tier runtime. LTX-Video pairs a 2B diffusion transformer with a 4.7B T5 text encoder. Stable Video Diffusion [13], the most natural image-to-video choice, is access-gated and therefore fails the stated requirement outright.

All work was carried out on a Colab host with 24 CPU threads and 51 GB of RAM, PyTorch 2.14.1 (CPU build) and Python 3.13.16.

## 2. Model selection and access verification

The requirement was for *free access (not gated)* models. Rather than assert this from a screenshot, access status is queried programmatically through the Hub API: `model_info(repo).gated` returns `False` for an open repository and `"auto"` or `"manual"` for one behind an access request. An assertion in the notebook aborts the run if any selected repository is gated, making the claim reproducible rather than asserted.

**Table 1 — Access audit (queried at runtime)**

| Repository | Gated | License | Task |
|---|---|---|---|
| `stabilityai/sd-turbo` | False | `sai-nc-community` | text-to-image |
| `stable-diffusion-v1-5/stable-diffusion-v1-5` | False | `creativeml-openrail-m` | text-to-image (base) |
| `ByteDance/AnimateDiff-Lightning` | False | `creativeml-openrail-m` | text-to-video |
| `madebyollin/taesd` | False | MIT | autoencoder |
| `SamLowe/roberta-base-go_emotions` | False | MIT | text-classification |

## 3. Method

### 3.1 Pipelines

**Text to image.** SD-Turbo is Stable Diffusion 2.1 [1] distilled by Adversarial Diffusion Distillation [3], which combines an adversarial loss against a discriminator with a score-distillation loss against the frozen teacher. The student traverses most of the denoising trajectory in a single move, so one network evaluation yields a usable 512×512 image. Because the model was distilled without classifier-free guidance [4], `guidance_scale` must be fixed at 0.0 and `negative_prompt` is inert.

**Image to video.** AnimateDiff [5] inserts temporal self-attention modules between the spatial layers of a frozen SD 1.5 UNet; the motion module is trained on video while the image model remains untouched, so any compatible checkpoint becomes a video model. AnimateDiff-Lightning [6] distils that module to a four-step schedule using progressive adversarial distillation. Since AnimateDiff is natively text-to-video, image conditioning is obtained by passing the still repeated N times to `AnimateDiffVideoToVideoPipeline` with `strength = 0.75`: the pipeline noises that clip partway and denoises from there, preserving composition while temporal attention introduces motion. Sampling uses an Euler scheduler [14] with trailing timestep spacing, following the correction described by Lin et al. [15].

**Text classification.** A RoBERTa-base encoder [8] fine-tuned on GoEmotions [7], a corpus of 58k Reddit comments annotated across 27 emotions plus `neutral`. The head applies an independent sigmoid per logit rather than a softmax across them, so the task is multi-label: scores do not sum to one, and inference requires a decision threshold rather than an argmax.

All three run through the `diffusers` [9] and `transformers` [10] libraries and are exposed through a single Gradio [11] interface.

### 3.2 Optimisations

1. **Step count over parameter count.** Denoising diffusion originally required on the order of 1000 steps [2], and standard Stable Diffusion sampling uses 25–50. Both diffusion models here are distilled to 1–4. Section 4.1 quantifies why this dominates every other choice.
2. **Distilled autoencoder.** TAESD [12] replaces the Stable Diffusion VAE. The original decoder is among the most expensive components on CPU, and across eight video frames it costs more than the denoising itself.
3. **No classifier-free guidance.** A guidance scale of 0.0 (Turbo) or 1.0 (Lightning) means one UNet forward pass per step instead of two. Both models were distilled without guidance, so this is free.
4. **Precision chosen by measurement.** fp16 has no fast CPU convolution kernels, while bfloat16 is accelerated on AMX-equipped server CPUs. A one-second convolution benchmark selects between fp32 and bf16 at runtime rather than hard-coding either.

### 3.3 Memory and storage

A least-recently-used cache holds exactly one diffusion pipeline resident, so peak memory is bounded by the larger of the two regardless of execution order. Weights are downloaded as fp16 variants and upcast on load, halving transfer while preserving fp32 arithmetic. Passing the substitute VAE at construction prevents `diffusers` from fetching the original.

## 4. Results

**Table 2 — Measured runtimes (24 threads, fp32, `report` preset)**

| Task | Phase | Seconds |
|---|---|---|
| 1 | model load | 29.2 |
| 1 | generate 512px ×3 | 21.0 / 17.0 / 22.2 (mean 20.1) |
| 2 | model load | 49.2 |
| 2 | animate, 8 frames @ 512px | 287.4 |
| 3 | model load | 7.6 |
| 3 | classify, single sentence ×5 | mean 0.30 |
| — | **total** | **9.0 min** |

### 4.1 Step scaling and the value of distillation

Sweeping the denoising step count at fixed resolution gives 18.0 s, 33.1 s and 52.8 s for 1, 2 and 4 steps. Least-squares regression yields

> **t ≈ 8.2 + 11.3·k seconds** (R² = 0.99)

where *k* is the step count. The interpretation is direct: 11.3 s is the marginal cost of one UNet pass at 512×512 on this host, and the 8.2 s intercept is the fixed overhead of text encoding and TAESD decoding, which does not scale with *k*.

Extrapolating to the 25 steps a non-distilled SD 2.1 would require predicts 291 s per image, against 19.5 s for the one-step distilled model — a **14.9× reduction attributable to distillation alone**. No other optimisation in this study comes close to that magnitude.

### 4.2 Image-to-video and the resolution threshold

The SD 1.5 autoencoder downsamples by a factor of eight, so a 256×256 request produces a 32×32 latent, half the linear dimension the motion module was trained on. An initial configuration at 256 px produced frames with no recoverable structure. At 512 px the latent is 64×64, matching the training distribution, and output structure is preserved: the lighthouse, horizon and wave geometry of the conditioning image all survive into every generated frame.

The two runs are also informative about the optimisations, because both used three effective denoising steps over eight frames:

| Configuration | Resolution | Pixels | Time |
|---|---|---|---|
| Initial (slicing on, no `channels_last`) | 256 px | 1× | 301.1 s |
| Final (slicing off, `channels_last`, VAE at construction) | 512 px | 4× | 287.4 s |

Four times the pixel count at 0.95× the wall-clock time implies roughly a **4.2× throughput improvement**. This is not an isolated ablation — three changes were made together — but the magnitude indicates that memory-layout and decoder choices are consequential on CPU, not marginal tuning.

### 4.3 Classification throughput

Single-sentence inference averaged 299 ms. A more useful comparison concerns batching: passing a list to a `transformers` pipeline does not batch unless `batch_size` is set, in which case the call degenerates into a Python loop. Measured sequentially the cost was 318 ms per sentence; with `batch_size=5` it fell to 101 ms, a **3.16× improvement** from a single keyword argument. This is a common source of misreported throughput.

### 4.4 Resource footprint

Peak resident memory was 18.1 GB against 51 GB available, confirming the LRU cache bounds memory as intended. Total cached weights came to 6.4 GB. The fp16-variant download combined with constructor-level VAE substitution removed approximately 330 MB: the SD-Turbo fetch fell from 2.58 GB across 12 files to 2.41 GB across 10, and SD 1.5 from 2.74 GB across 15 files to 2.58 GB across 13.

## 5. Discussion and limitations

The precision benchmark selected fp32, measuring bfloat16 convolution as more than two orders of magnitude slower on this host, which indicates the absence of AMX dispatch. On a Sapphire Rapids or later host the decision would likely reverse. Because selection happens at runtime, the notebook remains portable without edits.

A strength of 0.75 preserves structure but amplifies stylisation relative to the conditioning image, which is expected when denoising begins from a heavily noised latent. Lowering strength toward 0.5 would track the input more closely at the cost of motion amplitude; the trade is exposed as a slider rather than fixed.

Four limitations bound these results. The comparison in Section 4.2 confounds three simultaneous changes. All measurements come from a single host and a single seed, and Colab host specifications vary between sessions. No quantitative image-quality metric was computed, so judgements of coherence are qualitative. Finally, eight frames at 8 fps is a one-second clip; temporal consistency over longer sequences was not assessed.

## 6. Conclusion

Three open-access models covering text-to-image, image-to-video and text classification were deployed and measured end to end on CPU-only hardware in nine minutes, using 6.4 GB of storage and 18.1 GB of peak memory. Access status was verified programmatically rather than claimed. The measurements support a general conclusion: under a CPU budget, sampling step count dominates parameter count, and a distilled 865M model is a more practical choice than a full-precision 2.6B one not because it is smaller but because it needs one denoising pass instead of twenty-five.

---

## References

[1] R. Rombach, A. Blattmann, D. Lorenz, P. Esser and B. Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models," *CVPR*, 2022. arXiv:2112.10752.

[2] J. Ho, A. Jain and P. Abbeel, "Denoising Diffusion Probabilistic Models," *NeurIPS*, 2020. arXiv:2006.11239.

[3] A. Sauer, D. Lorenz, A. Blattmann and R. Rombach, "Adversarial Diffusion Distillation," *ECCV*, 2024. arXiv:2311.17042.

[4] J. Ho and T. Salimans, "Classifier-Free Diffusion Guidance," 2022. arXiv:2207.12598.

[5] Y. Guo, C. Yang, A. Rao, Z. Liang, Y. Wang, Y. Qiao, M. Agrawala, D. Lin and B. Dai, "AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning," *ICLR*, 2024. arXiv:2307.04725.

[6] S. Lin and X. Yang, "AnimateDiff-Lightning: Cross-Model Diffusion Distillation," 2024. arXiv:2403.12706.

[7] D. Demszky, D. Movshovitz-Attias, J. Ko, A. Cowen, G. Nemade and S. Ravi, "GoEmotions: A Dataset of Fine-Grained Emotions," *ACL*, 2020. arXiv:2005.00547.

[8] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer and V. Stoyanov, "RoBERTa: A Robustly Optimized BERT Pretraining Approach," 2019. arXiv:1907.11692.

[9] P. von Platen et al., "Diffusers: State-of-the-art Diffusion Models," GitHub, 2022. https://github.com/huggingface/diffusers

[10] T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," *EMNLP: System Demonstrations*, 2020.

[11] A. Abid, A. Abdalla, A. Abid, D. Khan, A. Alfozan and J. Zou, "Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild," 2019. arXiv:1906.02569.

[12] O. Boer Bohan, "Tiny AutoEncoder for Stable Diffusion (TAESD)," GitHub, 2023. https://github.com/madebyollin/taesd

[13] A. Blattmann et al., "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets," 2023. arXiv:2311.15127.

[14] T. Karras, M. Aittala, T. Aila and S. Laine, "Elucidating the Design Space of Diffusion-Based Generative Models," *NeurIPS*, 2022. arXiv:2206.00364.

[15] S. Lin, B. Liu, J. Li and X. Yang, "Common Diffusion Noise Schedules and Sample Steps are Flawed," *WACV*, 2024.
