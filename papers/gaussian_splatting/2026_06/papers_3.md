# gaussian splatting - 2026_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.02058v1">TIDES: Time-Derivative Event Simulation via Deformable Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-06-01
    </div>
    <details class="paper-abstract">
      Event cameras emit asynchronous events in response to environmental appearance changes. The scarcity of real-world event datasets makes simulation essential. However, most simulators infer event timestamps from frame sequences, forcing many threshold crossings to share a small set of discrete times; a failure mode we term timestamp batching that worsens under fast motion and occlusion. We present TIDES, a continuous-time event simulator built on dynamic Gaussian splatting. Because TIDES operates on an explicit 3D scene representation with learnt geometry and motion, it can derive per-pixel intensity dynamics directly from the scene, rather than by differencing rendered frames. This enables accurate threshold-crossing prediction, including multiple crossings per rendering step, without temporal upsampling or frame interpolation. The same 3D scene model reveals where objects partially occlude one another; TIDES uses this to guide adaptive time stepping, concentrating computation only in regions where occlusion dynamics make simple models of brightness change unreliable. Finally, we model finite sensor bandwidth using a tile-level arbiter whose throughput, jitter, and event drops reproduce realistic sensor artifacts. Across paired RGB-event benchmarks, TIDES attains state-of-the-art event-stream fidelity. We also show that events simulated by TIDES transfer more effectively to real downstream tasks than competitors'.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30855v2">Robust Dreamer: Deviation-Aware Latent Gaussian Memory for Action-Controlled AR Video Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-01
    </div>
    <details class="paper-abstract">
      Frame-wise action-controlled image-to-video generation is a promising paradigm for interactive world simulation, where each control signal should elicit an immediate visual response. However, maintaining visual fidelity and 3D consistency over long autoregressive rollouts remains challenging. Existing 3D-aware methods often suffer from catastrophic drift due to two impediments: information loss from \textit{Latent--RGB Cycling}, where generated latents are repeatedly decoded to RGB and re-encoded for future conditioning, and the training--inference gap induced by the \textit{error-free hypothesis}, where clean training memory fails to match prediction-corrupted inference memory. To address these challenges, we present \textbf{Robust Dreamer}, a memory-augmented framework built around how to design 3D memory and how to use it robustly. First, we introduce \textbf{Latent Gaussian Memory}, which anchors diffusion latents inherited from the generation process to Gaussian primitives and recalls them via latent-space Gaussian splatting. This provides dense, geometry-aware, view-aligned conditioning while avoiding accumulated degradation from repeated VAE conversion. Second, we propose \textbf{Deviation Learning with Dynamic Deviation Archive}, which synthesizes rollout-induced latent deviations through a one-step approximation, stores them by autoregressive stage and denoising timestamp, and injects them into historical memory during training. This exposes the generator to realistic corrupted memory states and teaches internal correction before inference. Experiments on ScanNet, DL3DV, and OmniWorldGame demonstrate state-of-the-art long-horizon performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.01950v1">Learning Action-Conditional and Object-Centric Gaussian Splatting World Models for Rigid Objects</a></div>
    <div class="paper-meta">
      📅 2026-06-01
    </div>
    <details class="paper-abstract">
      World models enable intelligent agents to predict the consequences of their actions on the environment. In this paper, we propose Multi Rigid Object Gaussian World Model (MRO-GWM), a novel model that learns action-conditional dynamics of rigid objects in 3D. By representing the scene by object-centric Gaussians, we can represent arbitrary object shapes and multi-object scenes. We develop a novel spatio-temporal transformer architecture that predicts future rigid body motion from a history of object Gaussians and future actions. Objects are represented by their Gaussians in a canonical frame, which allows for describing object motion as rigid body transformation. Our model is trained on reconstructions from multiple viewpoints, which requires the model to handle partial observations of objects due to occlusions. We analyze prediction performance of our approach on synthetic datasets composed of typical household objects with multi-object dynamics and interactions by a robot end effector. We also evaluate our model in model-predictive control for non-prehensile manipulation in simulation.
    </details>
</div>
