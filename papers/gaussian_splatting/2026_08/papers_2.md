# gaussian splatting - 2026_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01588v1">D^2-4DGS: Dual-Depth Guided Sparse-Camera 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-08-03
    </div>
    <details class="paper-abstract">
      Dynamic 4D Gaussian Splatting has emerged as an efficient representation for dynamic novel view synthesis through explicit scene modeling and real-time rendering. However, existing methods typically require dense multi-view videos for sufficient geometric constraints, making capture expensive and limiting sparse-camera deployment. Reducing input views lowers acquisition cost but weakens geometry supervision, often causing missing structures and floating Gaussians. Depth priors provide geometric cues, yet no single source offers both dense coverage and reliable geometry. Monocular depth provides dense structure but is scale-ambiguous and locally biased, whereas multi-view geometric depth provides incomplete anchors consistent with the reconstruction coordinate system. To exploit their complementarity, we propose D$^2$-4DGS, a sparse-camera dynamic 4D Gaussian Splatting framework guided by dual-source depth priors. We align monocular estimates with valid multi-view geometric depths and verify their consistency to identify reliable geometric anchors. These verified anchors support consistency-aware pruning and depth supervision, while verified geometric depths and aligned mono-only estimates provide candidate geometry for densification in under-reconstructed regions. Finally, RGB-D joint optimization improves appearance fidelity and geometric consistency under sparse-view supervision. Across all nine dataset--view settings, D$^2$-4DGS achieves the highest PSNR, improving by 1.33 dB on average over the best competing method in each setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06404v1">UAV3DCrop: Benchmarking 3D Reconstruction in Repeated Multi-Angle UAV Crop Surveys</a></div>
    <div class="paper-meta">
      📅 2026-08-03
      | 💬 22 pages, 7 figures. Dataset and project page: https://link-dev.github.io/UAV3DCrop/
    </div>
    <details class="paper-abstract">
      Accurate 3D crop monitoring underpins data-driven precision agriculture by enabling field-scale analysis of plant structure, growth dynamics, and management response. Modern 3D reconstruction methods perform strongly on generic benchmarks, but rendered appearance may not translate into metrically and agronomically useful geometry in crop fields. We introduce UAV3DCrop, a public benchmark of repeated multi-angle unmanned aerial vehicle (UAV) crop surveys. It contains 88,830 RGB images at $5280 \times 3956$ pixels, with a ground sampling distance of 3.6-5.8 mm, from 91 scenes spanning corn, soybean, wheat, and oat. Track A evaluates seven scene-optimized methods -- Neural Radiance Field (NeRF) and 3D Gaussian Splatting (3DGS) variants -- on held-out views, photogrammetry-referenced depth, and canopy-height recovery. Track B tests four pretrained feed-forward models on zero-shot camera-pose and geometry estimation. The scene-optimized methods rank differently across the three targets: Splatfacto-big leads appearance, whereas Scaffold-GS leads depth and is statistically tied with Splatfacto for canopy height. Among feed-forward models, MapAnything leads on seven of the eight metrics, while the remaining models vary more across crops and fail severely on absolute scale in a way that alignment conceals. Repeated acquisitions reveal further sensitivities that differ by output type and by model, associated with position within the acquisition sequence and with tie-point multiplicity. Current 3D reconstruction methods are therefore not yet interchangeable for agronomic use: no single method wins on appearance, geometry, and canopy height at once, and only one of four feed-forward models recovers usable metric scale. The dataset is publicly available at https://link-dev.github.io/UAV3DCrop/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01492v1">GaussianSelector: Lightweight Human-Guided Object Selection in 3D Gaussian Splatting with Graph Optimization</a></div>
    <div class="paper-meta">
      📅 2026-08-02
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Selecting a complete 3D object from a reconstructed scene with minimal user effort is essential for practical scene editing and embodied interaction. Existing 3DGS-based methods either retrain the Gaussian representation to embed per-object labels, or build dense multi-view SAM observations, both requiring heavy computation and dense viewpoint coverage that is rarely available in practice. We present GaussianSelector, a training-free framework for interactive 3D object selection from sparse views and sparse scribble guidance. Operating directly on native Gaussian primitives, we coarsen dense Gaussians into geometrically coherent superpoints and construct a continuity-weighted graph using appearance and spatial cues. Sparse user scribbles are lifted into 3D via visibility-aware transmittance coverage, and selection is solved as a global graph-cut energy minimization that propagates sparse evidence to a complete 3D object. This design naturally supports multi-round refinement, where users iteratively correct the selection from additional viewpoints to progressively improve the result. Experiments demonstrate that GaussianSelector achieves competitive selection quality against state-of-the-art multi-view SAM-based methods, while requiring significantly fewer interaction views and substantially lower computational overhead. These properties make it well suited for human-in-the-loop 3D scene editing and 3D asset extraction in real-world deployment scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01186v1">QuerySplat: Decoupling Geometry and Appearance Representations in 3DGS Prediction</a></div>
    <div class="paper-meta">
      📅 2026-08-02
    </div>
    <details class="paper-abstract">
      While feed-forward 3D Gaussian Splatting (3DGS) enables efficient 3D reconstruction, achieving high-fidelity rendering remains challenging. Existing pixel-aligned approaches suffer from spatial inflexibility and massive structural redundancy, whereas query-based methods lack 3D priors and entangle geometry with appearance, yielding blurry, pose-dependent results. To overcome these deficiencies, we propose \textbf{QuerySplat}, a feed-forward 3DGS framework driven by geometric priors and explicit appearance decoupling. Specifically, we design a dual-branch query-based decoder: the geometry branch leverages a pretrained Vision Geometric Model for spatial understanding, which intrinsically endows QuerySplat with pose-free modeling capabilities, while the appearance branch recovers high-frequency details through a dedicated pathway separated from geometric attribute regression. Extensive experiments demonstrate that QuerySplat mitigates the blurry rendering issues of early query-based models and consistently outperforms pixel-aligned approaches in rendering fidelity. On the challenging DL3DV benchmark, it achieves state-of-the-art novel view synthesis performance, with average PSNR gains of 2.30 dB and 1.04 dB over the best pose-free and pose-required baselines, respectively. Project Page: https://inspatio.github.io/querysplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01178v1">DynActiveGS: Active Gaussian Splatting for Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-08-02
      | 💬 Accepted to ACM Multimedia 2026
    </div>
    <details class="paper-abstract">
      We present DynActiveGS, a dynamic-aware active reconstruction framework based on 3D Gaussian Splatting (3DGS) for autonomous exploration in dynamic environments. The framework incrementally reconstructs a 3D Gaussian scene representation while suppressing motion-corrupted observations through online uncertainty prediction and uncertainty-weighted Gaussian optimization. A key component of DynActiveGS is the explicit decomposition of uncertainty into structural uncertainty and motion-induced uncertainty, which enables the system to distinguish under-reconstructed static regions from dynamically unreliable areas. Based on these uncertainty fields, DynActiveGS performs dynamic-aware viewpoint selection and dynamic-constrained path planning to favor informative yet stable observations during exploration. The resulting system forms a unified closed-loop pipeline for robust active reconstruction in dynamic scenes. Extensive experiments on challenging dynamic benchmarks demonstrate consistent improvements over existing active reconstruction baselines in reconstruction accuracy, completeness, rendering quality, and exploration efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25129v2">AirSplat: Alignment and Rating for Robust Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-08-02
      | 💬 Project page: https://kaist-viclab.github.io/airsplat-site, accepted to ECCV 2026
    </div>
    <details class="paper-abstract">
      While 3D Vision Foundation Models (3DVFMs) have demonstrated remarkable zero-shot capabilities in visual geometry estimation, their direct application to generalizable novel view synthesis (NVS) remains challenging. In this paper, we propose AirSplat, a novel training framework that effectively adapts the robust geometric priors of 3DVFMs into high-fidelity, pose-free NVS. Our approach introduces two key technical contributions: (1) Self-Consistent Pose Alignment (SCPA), a training-time feedback loop that ensures pixel-aligned supervision to resolve pose-geometry discrepancy; and (2) Rating-based Opacity Matching (ROM), which leverages the local 3D geometry consistency knowledge from a sparse-view NVS teacher model to filter out degraded primitives. Experimental results on large-scale benchmarks demonstrate that our method significantly outperforms state-of-the-art pose-free NVS approaches in reconstruction quality. Our AirSplat highlights the potential of adapting 3DVFMs to enable simultaneous visual geometry estimation and high-quality view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.00950v1">Swimm3R: Splatting with Medium-aware SfM for Underwater 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-08-02
      | 💬 Project Page: https://mnseong.github.io/swimm3r.github.io
    </div>
    <details class="paper-abstract">
      We propose Swimm3R, a unified framework that combines medium-aware structure-from-motion (SfM) with Underwater Beta Splatting to address scattering- and attenuation-induced failures in underwater 3D reconstruction. Swimm3R distills in-air geometric priors into a feed-forward backbone and uses a physics head to regress underwater image-formation parameters, camera poses, and restored point clouds. Additionally, we introduce Underwater Beta Splatting, which extends Gaussian splatting with Beta primitives and scattering-aware geometric gradients for stable underwater geometry representation. We further establish the Barbados underwater video dataset to demonstrate the effectiveness of our method in challenging underwater environments. On this dataset, Swimm3R robustly recovers underwater scene structure under challenging scattering conditions, yielding coherent seafloor geometry. Using these predicted point clouds, the proposed Underwater Beta Splatting improves average PSNR by $1.47$ dB over WaterSplatting while increasing downstream localization performance by $2.0$ and $2.4$ percentage points in RRA@15 and RTA@15, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.00931v1">Stipple: Real-Time Incremental Gaussian Splatting with Visual-Inertial Tracking</a></div>
    <div class="paper-meta">
      📅 2026-08-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) provides efficient rendering of photo-realistic scenes, but its heavy preprocessing and training steps make it a poor fit for applications that require real-time reconstruction in robotics or XR. This capability is important since it allows immediate feedback and interaction with new environments. Visual-inertial odometry (VIO) and simultaneous localization and mapping (VI-SLAM) systems, on the other hand, specifically target these real-time applications, which makes them a good choice for integration with 3DGS. We propose a new method that tracks and reconstructs simultaneously in real-time by leveraging an efficient visual-inertial tracking system based on Basalt together with a novel incremental method built on top of Brush, an efficient Rust-based GPU-vendor-agnostic implementation of 3D Gaussian Splatting. We show that many of the heavy preprocessing and training steps of 3DGS can be replaced with a more efficient incremental training strategy that has direct access to the information generated by the visual-inertial tracking system. Furthermore, we propose and combine multiple practical improvements to increase the efficiency of the training pipeline and adapt it to run in real-time, parallel to the tracking thread. This work highlights the value of exploiting the complementary nature of SLAM and 3DGS, and how that can lead to promising results for real-time 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.00463v1">Scene2Sound: Auditory-Grounded Soundscape Generation for 3D Gaussian Worlds</a></div>
    <div class="paper-meta">
      📅 2026-08-01
      | 💬 14 pages. Project page: https://masaki-lmd.github.io/scene2sound/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) turns captured or generated imagery into photorealistic 3D world simulations that users can freely explore, yet these worlds remain silent. Because existing audio generation methods condition on a single image or viewpoint, their sound is tied to that observation and cannot stay consistent while a listener moves. We introduce the task of generating a spatially consistent soundscape for a given 3DGS world through auditory grounding, identifying which objects in the world should emit sound and anchoring each to a persistent 3D position, and present Scene2Sound, a training-free framework built on this grounding. From the input world alone, our pipeline selects viewpoints that jointly cover the scene, identifies sound-emitting objects with a vision-language model, and associates the multi-view detections into 3D instances through Gaussian set matching, which measures the overlap between the Gaussian sets that render each detection. Each source then receives generated audio that a standard object-based audio engine spatializes in real time at arbitrary listener poses. We further propose two spatial-consistency metrics, one testing whether rendered audio responds consistently to listener motion and one testing whether the claimed sources are supported by views held out from their placement. On a curated set of generated 3DGS worlds and on 3DGS scenes generated from real-world 360-degree captures, Scene2Sound preserves the audio quality of strong per-viewpoint baselines while remaining spatially consistent where per-viewpoint and single-panorama pipelines do not, and a user study confirms the perceptual benefit. Project page: https://masaki-lmd.github.io/scene2sound/.
    </details>
</div>
