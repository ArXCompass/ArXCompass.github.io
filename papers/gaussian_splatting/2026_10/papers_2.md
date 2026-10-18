# gaussian splatting - 2026_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.11434v2">Hologram Representation via Quadratic Phase Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-01
      | 💬 SIGGRAPH Asia 2026 Technical Communications
    </div>
    <details class="paper-abstract">
      We introduce Complex-Valued Quadratic Phase Gaussian (CVQPG), a novel hologram representation method that augments each 2D Gaussian primitive with a quadratic phase profile controlled by a learnable curvature parameter. Against the planar Gaussian baseline, CVQPG improves the average PSNR of holographic reconstructions by 0.19 dB (RGB) and 0.33 dB (grayscale) at equal primitive counts, and by 0.05 dB (RGB) and 0.08 dB (grayscale) at equal parameter counts, where it still leads in all visual quality metrics. Our frequency-domain analysis shows that CVQPG better preserves the mid-to-high frequency band of natural images, where the reconstruction MSE drops by up to 11% (RGB) and 22% (grayscale), indicating that modulating primitive wavefronts is an effective and lightweight enhancement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.01876v1">EvenSplat: Coupled 2D-3D Decomposition for Gaussian Splatting under Exposure and Illumination Variation</a></div>
    <div class="paper-meta">
      📅 2026-10-01
    </div>
    <details class="paper-abstract">
      A surface photographed under even light presents nearly the same appearance from every angle; the same surface under uneven light does not. Exposure changes between views, illumination varies within a single image, and locally strong light sources leave one region bright and its neighbor in shadow. Multi-view reconstruction methods such as 3D Gaussian Splatting treat these lighting artifacts as if they were properties of the scene, entangling capture-specific illumination with the geometry and color they recover. We present EvenSplat, a framework that separates the two. EvenSplat couples an image-space illumination decomposition with an illumination field carried by the Gaussians, so that the same explanation of the lighting is shared between the two-dimensional and three-dimensional views of the scene; a camera-response network and a local exposure-compensation module absorb the global and residual differences that remain across training images. Through extensive experiments across multiple datasets and diverse forms of uneven illumination (cross-view exposure, spatial illumination variation, and high-contrast lighting) on both real-world captured and simulated benchmarks, EvenSplat generally outperforms state-of-the-art methods, particularly under high-contrast illumination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.01707v1">MEGA: Object-Level Mesh Extraction from 3D Gaussian Splatting via Spatial Visual Distillation</a></div>
    <div class="paper-meta">
      📅 2026-10-01
    </div>
    <details class="paper-abstract">
      Mesh extraction from 3D Gaussian Splatting (3DGS) aims to endow 3D Gaussians with accurate geometric structures, enabling explicit and precise 3D occupancy. However, existing methods primarily focus on scene-level mesh extraction, making them unable to represent object-level occupancy and often resulting in non-watertight surfaces. To overcome these limitations, we propose \textbf{MEGA} (\underline{M}esh \underline{E}xtraction from \underline{GA}ussians), a ``segment-then-mesh'' framework for extracting object-level, watertight meshes from complex 3DGS scenes. At the core of MEGA are \textbf{Spatial Visual Distillation (SVD)} and a mask-guided neural surface reconstruction module. SVD treats the 3DGS model as a teacher, sampling diverse camera poses and rendering the corresponding views of each segmented object. These observations are then used to train a mesh reconstruction model through photometric supervision. Extensive experiments on several widely used benchmarks demonstrate that MEGA achieves state-of-the-art performance in recovering accurate object-level 3D occupancy. Moreover, MEGA enables complex physical interactions by combining high-quality object-level meshes for geometric occupancy with 3DGS representations for photorealistic rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.01114v1">Affine-Aligned Atlas for Canonical Gaussian Construction in Video Representation</a></div>
    <div class="paper-meta">
      📅 2026-10-01
    </div>
    <details class="paper-abstract">
      Gaussian splatting has recently emerged as an efficient representation for images and videos due to its explicit structure and fast rendering capability. Existing Gaussian-based video representations often decompose a video into canonical Gaussians and temporal deformation. However, when a video contains large global motion such as camera movement, the canonical representation may become misaligned with individual frames, increasing the burden on the temporal deformation model. In this paper, we propose an affine-atlas canonical Gaussian representation, which constructs canonical Gaussians in a larger affine-aligned atlas space. Frame-wise affine transforms absorb global motion before canonical Gaussian construction, reducing the gap between the canonical representation and target frames. Since the proposed method only modifies the canonical construction stage, it can be integrated into existing canonical-Gaussian-based methods with negligible additional parameter cost. Experiments show that our method improves reconstruction quality especially for sequences with large camera motion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14929v2">Wrivinder: Towards Spatial Intelligence for Geo-locating Ground Images onto Satellite Imagery</a></div>
    <div class="paper-meta">
      📅 2026-10-01
    </div>
    <details class="paper-abstract">
      Aligning ground-level imagery with geo-registered satellite maps is crucial for mapping, navigation, and situational awareness, yet remains challenging under large viewpoint gaps or when GPS is unreliable. We introduce Wrivinder, a zero-shot, geometry-driven framework that aggregates multiple ground photographs to reconstruct a consistent 3D scene and align it with overhead satellite imagery. Wrivinder combines SfM reconstruction, 3D Gaussian Splatting, semantic grounding, and monocular depth--based metric cues to produce a stable zenith-view rendering that can be directly matched to satellite context for metrically accurate camera geo-localization. To support systematic evaluation of this task, which lacks suitable benchmarks, we also release MC-Sat, a curated dataset linking multi-view ground imagery with geo-registered satellite tiles across diverse outdoor environments. Together, Wrivinder and MC-Sat provide a first comprehensive baseline and testbed for studying geometry-centered cross-view alignment without paired supervision. In zero-shot experiments, Wrivinder achieves sub-30\,m geolocation accuracy across both dense and large-area scenes, highlighting the promise of geometry-based aggregation for robust ground-to-satellite localization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.07760v2">XClipGS: Exact Half-Space Clipping for Medical Volume Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-01
    </div>
    <details class="paper-abstract">
      Gaussian-splatting proxies enable interactive rendering of volumetric medical scans, but a clipping plane exposes anatomy not constrained by external-view training and intersects primitives that conventional splatting can only keep or drop whole. We present XClipGS (eXact Clipping), which treats these as two separate problems: the render-time clip operator and supervision of the hidden interior. Under the local affine model used by EWA splatting, the ray integral of a half-space-restricted Gaussian factorizes exactly into its ordinary 2D footprint and a conditional Gaussian CDF whose argument is affine in pixel coordinates. The resulting closed-form per-pixel operator introduces no learned clipping parameters or auxiliary network and remains differentiable with respect to the primitive and plane. We use multi-distance reference views with varied clipping-plane axes and offsets to supervise the interior through the same operator. We also introduce a paired clipped/unclipped cut-face protocol with difference-referenced cut error (CDE) and culled-side leakage (Leak), because global image metrics dilute errors near the plane. On eight CT and MRI volumes with plane offsets not used for training, XClipGS attains the highest PSNR on every volume (33.56 versus 32.34 dB for ClipGS) while rendering at over 650 FPS, far above real time, versus 278 FPS. On voxel-axis cut-face views, it raises average band SSIM from 0.809 to 0.860 and leaks roughly 40 times less. Without retraining, it also achieves the best average across all four metrics on arbitrary-normal planes; on a fixed interior, it matches RaRa's face fidelity with about 16 times less leakage. Project page: https://gaozhongpai.github.io/XClipGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.02382v1">FactorSplat: Appearance-Controllable Gaussian Proxies for Medical Volume Rendering</a></div>
    <div class="paper-meta">
      📅 2026-10-01
    </div>
    <details class="paper-abstract">
      Transfer functions (TFs) control color and visibility in medical volume rendering, but image-trained Gaussian proxies typically bake one transfer function into their appearance. We present FactorSplat, a per-scene N-dimensional Gaussian splatting (N-DGS) proxy that accepts region-specific intensity-to-RGBA curves at inference. A local lookup applies the authored color and opacity change, while a shared functional encoder and low-rank per-Gaussian factors learn the residual appearance response. Geometry and directional appearance remain shared across presets, with visibility control and TF-aware pruning preserving the ability to hide and reveal structures. On seven CT and MR scans, FactorSplat improves mean PSNR and changed-region error over region-aware VEG across validation, interpolation, unseen composition, and out-of-distribution (OOD) edits. Across these four splits, seven-scan mean PSNR gains over VEG range from 1.10 to 1.52 dB. One checkpoint per scan supports unseen edits without retraining. At $1600^2$, the cached fast renderer averages 524 FPS with 1.17 ms TF switches. Project page: https://gaozhongpai.github.io/FactorSplat/.
    </details>
</div>
