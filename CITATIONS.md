# Running citation list (for the write-up)

Methods and architecture
- Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
- Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", ICLR 2023.
- Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM), NeurIPS 2022 —
  preconditioning, loss weighting, DDPM++ 64x64 config, non-leaking augmentation, EMA half-life.
- Karras et al., "Analyzing and Improving the Training Dynamics of Diffusion Models" (EDM2), CVPR 2024 —
  ImageNet-64 noise distribution (P_mean = -0.8, P_std = 1.6).
- Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations" (DDPM++), ICLR 2021.
- Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis" (ADM), NeurIPS 2021.
- Ho & Salimans, "Classifier-Free Diffusion Guidance", 2022.
- Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3), ICML 2024 —
  logit-normal timestep sampling, timestep shift.
- Kynkäänniemi et al., "Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in
  Diffusion Models", NeurIPS 2024.
- Lu & Song, "Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models" (sCM), 2024 —
  low-frequency time embedding for JVP stability.
- Gu et al., "On Memorization in Diffusion Models", 2023 — nearest-neighbour ratio memorization metric.
- Lee, Lin & Fanti, "Improving the Training of Rectified Flows" (2-rectified flow++), NeurIPS 2024 — one Reflow
  round, initialization from the 1-rectified flow, U-shaped timestep distribution, pseudo-Huber premetric with
  c = 0.00054 d, real-data fine-tuning via ODE inversion, the "new" update rule for >= 3 steps. Their official code
  (github.com/sangyun884/rfpp) was read to pin down exact details (mirrored truncated-exponential t sampler,
  Huber constant and reduction); no code was copied.
- Song & Dhariwal, "Improved Techniques for Training Consistency Models" (iCT), ICLR 2024 — pseudo-Huber loss.
- Yin et al., "One-step Diffusion with Distribution Matching Distillation" (DMD), CVPR 2024 — distribution
  matching gradient from a frozen real score and an online fake score.
- Yin et al., "Improved Distribution Matching Distillation for Fast Image Synthesis" (DMD2), NeurIPS 2024 — no
  regression loss, two time-scale update rule (5 fake-score updates per generator update), GAN classifier on the
  fake score's bottleneck with noised real/fake inputs, backward simulation for multi-step generators; ImageNet-64
  hyper-parameters (lr 2e-6, AdamW, GAN weights 1e-2 / 3e-3, grad-clip 10). Their official code
  (github.com/tianweiy/DMD2, main/edm/) was read to pin down exact details (Karras timestep ranges 2%-98%,
  per-sample gradient normalization, loss reductions, classifier head layout); no code was copied.
- Wang et al., "Diffusion-GAN: Training GANs with Diffusion", ICLR 2023 — discriminator on diffusion-noised inputs.
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW), ICLR 2019.

Evaluation
- Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (FID), NeurIPS 2017.
- Szegedy et al., "Rethinking the Inception Architecture for Computer Vision", CVPR 2016.
- Parmar et al., "On Aliased Resizing and Surprising Subtleties in GAN Evaluation" (clean-fid), CVPR 2022.

Software and data
- PyTorch (Paszke et al., NeurIPS 2019).
- Pokemon Generation One - 22k, Kaggle (bhawks/pokemon-generation-one-22k).

No external code was copied; all modules in fastgen/ are our own implementations.
