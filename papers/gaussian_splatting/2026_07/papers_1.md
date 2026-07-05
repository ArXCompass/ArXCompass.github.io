# gaussian splatting - 2026_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.01521v3">MiraGe: Editable 2D Images using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Implicit Neural Representations (INRs) approximate discrete data through continuous functions and are commonly used for encoding 2D images. Traditional image-based INRs employ neural networks to map pixel coordinates to RGB values, capturing shapes, colors, and textures within the network's weights. Recently, GaussianImage has been proposed as an alternative, using Gaussian functions instead of neural networks to achieve comparable quality and compression. Such a solution obtains a quality and compression ratio similar to classical INR models but does not allow image modification. In contrast, our work introduces a novel method, MiraGe, which uses mirror reflections to perceive 2D images in 3D space and employs flat-controlled Gaussians for precise 2D image editing. Our approach improves the rendering quality and allows realistic image modifications, including human-inspired perception of photos in the 3D world. Thanks to modeling images in 3D space, we obtain the illusion of 3D-based modification in 2D images. We also show that our Gaussian representation can be easily combined with a physics engine to produce physics-based modification of 2D images. Consequently, MiraGe allows for better quality than the standard approach and natural modification of 2D images
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02099v1">X-Splat: Gaussian Splatting for 3D CBCT Generation from Single Panoramic Radiograph</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 19 pages, 6 figures, including appendix. Under review
    </div>
    <details class="paper-abstract">
      Generating a 3D dental volume from a single panoramic radiograph (PXR) could provide a low-radiation alternative to Cone-Beam Computed Tomography (CBCT), but the problem is highly underdetermined: panoramic acquisition integrates 3D attenuation along curved X-ray paths into a 2D image, leaving depth-resolved anatomy unobserved. Existing implicit and generative approaches often produce oversmoothed geometry or anatomically inconsistent hallucinations, lacking geometry-driven supervision and relying on smooth representations unable to precisely localize sharp anatomical boundaries. We propose X-Splat, the first Gaussian Splatting framework for generating CBCT-like 3D dental volumes from a single PXR. X-Splat uses the known panoramic acquisition geometry as a generation scaffold: learnable anisotropic Gaussian primitives are initialized along the X-ray paths that formed the input image and adjusted in a single feed-forward pass, constrained by Beer-Lambert reprojection and multi-view radiographic training supervision. A lightweight residual refiner adds dataset-level anatomical priors without overriding the geometry already resolved by the Gaussians. We train on synthetic PXR-CBCT pairs, enabling direct volumetric supervision without paired real scans. We further introduce segmentation-based geometry-aware metrics, providing the first evaluation of PXR-based generation over maxillofacial anatomy. X-Splat outperforms NeRF- and GAN-based baselines, recovering individual teeth, cortical boundaries, and alveolar structure, including the mandibular canal which prior methods fail to reconstruct. Code will be available at https://github.com/tomek1911/X-Splat
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.24009v3">Learning 3D-Gaussian Simulators from RGB Videos</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Realistic simulation is critical for applications ranging from robotics to animation. Learned simulators have emerged as a possibility to capture real world physics directly from video data, but very often require privileged information such as depth information, particle tracks and hand-engineered features to maintain spatial and temporal consistency. These strong inductive biases or ground truth 3D information help in domains where data is sparse but limit scalability and generalization in data rich regimes. To overcome the key limitations, we propose 3DGSim, a learned 3D simulator that directly learns physical interactions from multi-view RGB videos. 3DGSim unifies 3D scene reconstruction, particle dynamics prediction and video synthesis into an end-to-end trained framework. It adopts MVSplat to learn a latent particle-based representation of 3D scenes, a Point Transformer for particle dynamics, a Temporal Merging module for consistent temporal aggregation and Gaussian Splatting to produce novel view renderings. By jointly training inverse rendering and dynamics forecasting, 3DGSim embeds the physical properties into point-wise latent features. This enables the model to capture diverse physical behaviors, from rigid to elastic, cloth-like dynamics, and boundary conditions (e.g. fixed cloth corner), along with realistic lighting effects that also generalize to unseen multibody interactions and novel scene edits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00595v2">GADA: Geometry-Aware Deformable Aggregation for Image-Based Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has achieved significant improvements by incorporating warping-based techniques. However, such methods suffer from pixel-level inaccuracies due to uncertain geometry. This uncertainty leads to spatial misalignments in the warped images, which disrupt residual learning used in warping-based methods and fundamentally limit the gains of correction, particularly on thin structures and high-frequency details. Driven by our insight that useful visual cues are not lost but locally preserved under slight displacement, we propose Geometry-Aware Deformable Aggregation (GADA). This method introduces an iterative refinement module with deformable offsets to actively correct spatial misalignments and recover these displaced cues. Furthermore, to address the limitations of standard pipelines where visibility checks (i.e., thresholding) often discard valid pixels and multi-view warped image fusion relies on naive mean aggregation, our module is coupled with an implicit confidence weighting mechanism that selectively suppresses unreliable evidence. Consequently, our approach outperforms prior warping-based Gaussian Splatting, preserving high-frequency quality while achieving 2.13 times faster FPS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01860v1">DL-SLAM: Enabling High-Fidelity Gaussian Splatting SLAM in Dynamic Environments based on Dual-Level Probability</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled significant progress in dense dynamic Simultaneous Localization And Mapping (SLAM). Prevailing methods typically discard predefined dynamic objects, ignoring that transiently static objects offer valuable geometric constraints for pose estimation. A recent work attempts to leverage this potential by employing per-pixel uncertainty maps to quantify the magnitude of motion. While this approach enables transiently static objects to enhance pose estimation, it erroneously integrates these objects into the static map, resulting in persistent artifacts. Moreover, its reliance on purely geometric information leads to ambiguous object boundaries in the uncertainty maps. To overcome these limitations, we present DL-SLAM, a monocular Gaussian Splatting SLAM system built upon a novel dual-level probabilistic framework. Our method computes dynamic probability maps by combining semantic and geometric information. These pixel-level probabilities are lifted to 3D and aggregated to derive an object-level dynamic probability for each instance. Object-level probability enables the categorical pruning of dynamic Gaussians, resulting in an artifact-free static map. The static map, in turn, provides a geometrically consistent guidance to refine the pixel-wise probabilities, enhancing their reliability. Experimental results demonstrate that DL-SLAM outperforms existing approaches, improving tracking accuracy by up to 13\% while generating high-fidelity semantic maps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01803v1">PixGS: Pixel-Space Diffusion for Direct 3D Gaussian Splat Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 Accepted at ECCV 2026
    </div>
    <details class="paper-abstract">
      Recent advances in 3D content generation from text or images have achieved impressive results, yet view inconsistency from 2D generators and the scarcity of high-quality 3D data remain significant bottlenecks. Existing solutions typically adapt large-scale pre-trained text-to-image latent diffusion models to generate 3D Gaussian Splats (3DGS). However, these approaches often rely on training complex cascade pipelines that are computationally expensive and scalability-limited. Most critically, the quality of generated 3D assets is inherently constrained by each component capacity and compressed latent space, leading to decoding artifacts and accumulated errors. To address these limitations, we propose PixGS, a single-stage pipeline for direct high-quality 3DGS generation, which leverages recent advances in pixel-space diffusion to bypass lossy latent compression while still benefiting from the vast 2D generative priors. By directly denoising 3D Gaussian attributes at each timestep, our method enables precise, splat-level regularization of both appearance and geometry. Furthermore, we introduce a comprehensive supervision strategy that incorporates surface normals, depth, and high-frequency structural information, which is often overlooked in prior works. Experiments demonstrate that PixGS outperforms current state-of-the-art methods while maintaining a fast inference speed (1s on a single A100 GPU), offering a robust and efficient alternative to multi-stage generation pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01753v1">The Turning Point of 3D Plant Phenotyping: 3D Foundation Models Enable Minute-to-Second Cross-Crop Reconstruction and Beyond</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 39 pages, 6 figures, 3 tables
    </div>
    <details class="paper-abstract">
      3D plant phenotyping is notoriously known to be procedure-complicated and of low throughput due to the extensive multi-view imaging, the fragile 3D reconstruction pipeline, and the additional cost from reconstructed geometry to phenotypic extraction. These limitations are further amplified in low-cost data acquisition, where smartphone videos or sparsely sampled multi-view images provide limited view overlap and self-occlusion. In this work, we show that the conventional 3D plant phenotyping pipeline could be streamlined and significantly accelerated with 3D Foundation Models (3DFMs), and particularly, present one of the first cross-crop 3D phenotyping frameworks powered by 3DFMs. The framework replaces COLMAP-style sparse initialization with 3DFM-based feed-forward geometric recovery, combines geometry-constrained 3D Gaussian Splatting for dense reconstruction, enables few-view reconstruction through iterative view synthesis and refinement, and converts reconstructed geometry into measurable organs through 2D-to-3D semantic transfer, metric scale recovery, and organ instance separation. We further construct a cross-crop dataset with smartphone-based image acquisition, diverse plant morphologies, and manual annotations for segmentation and phenotypic evaluation. Experiments across 26 plant sequences show that 3D Foundation Models reduce the average reconstruction time from 6.52 minutes to 1.58 seconds while maintaining high reconstruction quality and phenotyping accuracy. These results suggest a fresh technical route for high-throughput 3D plant phenotyping, from low-cost image acquisition to fast reconstruction, perception, scale recovery, and phenotypic measurement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01708v1">Consistent Scene Understanding in 3D Gaussian Splatting via Multi-Cue Mask Refinement</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 Accepted at ICPR 2026
    </div>
    <details class="paper-abstract">
      Reliable instance-level scene understanding is a fundamental prerequisite for object-level interactions and high-fidelity 3D representations. While current methods often leverage 2D foundation segmentation models to obtain these priors, their 2D-centric design typically yields fragmented masks and inconsistent predictions across different views. To address these issues, we propose a novel framework that produces consistent 2D instance masks to guide the optimization of 3D Gaussian Splatting (3DGS) feature fields. Our framework consists of three main stages. (1) Multi-Cue Extraction that generates synergistic semantic, geometric, and structural priors from input images. (2) Multi-Cue-Guided Mask Merging process that consolidates fragmented masks using a composite merge score derived from semantic, depth, and edge cues. (3) Cross-View Mask Matching that establishes globally consistent identity assignments across all viewpoints. By transforming viewpoint-specific segments into coherent 3D primitives, our approach enables stable 3D instance segmentation and effective downstream editing tasks. Experiments demonstrate that our method significantly improves cross-view consistency and segmentation stability over existing baselines while maintaining high-fidelity photometric reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01698v1">Structure-Aware Gaussian Splatting for Large-Scale Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has demonstrated remarkable potential in novel view synthesis. In contrast to small-scale scenes, large-scale scenes inevitably contain sparsely observed regions with excessively sparse initial points. In this case, supervising Gaussians initialized from low-frequency sparse points with high-frequency images often induces uncontrolled densification and redundant primitives, degrading both efficiency and quality. Intuitively, this issue can be mitigated with scheduling strategies, which can be categorized into two paradigms: modulating target signal frequency via densification and modulating sampling frequency via image resolution. However, previous scheduling strategies are primarily hardcoded, failing to perceive the convergence behavior of scene frequency. To address this, we reframe the scene reconstruction problem from the perspective of signal structure recovery and propose SIG, a novel scheduler that synchronizes image supervision with Gaussian frequencies. Specifically, we derive the average sampling frequency and bandwidth of 3D representations, and then regulate the training image resolution and the Gaussian densification process based on scene frequency convergence. Furthermore, we introduce Sphere-Constrained Gaussians, which leverage the spatial prior of initialized point clouds to control Gaussian optimization. Our framework enables frequency-consistent, geometry-aware, and floater-free training, achieving state-of-the-art performance by a substantial margin in both efficiency and rendering quality in large-scale scenes. The code is available at: https://github.com/weiyixue999/Signal_Structure_Aware_Gaussian
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01628v1">Online Segment 3D Gaussians via Launching Virtual Drones</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Interactive segmentation of 3D Gaussians offers a compelling opportunity for real-time manipulation of 3D scenes, thanks to the real-time rendering capability of 3D Gaussian Splatting (3DGS). However, existing methods require a time-consuming per-scene setup - typically tens of seconds or even minutes - before interactive segmentation can begin on a raw 3DGS scene. This setup involves multi-view mask preparation, mask lifting, and feature distillation, creating a major bottleneck for online applications. To address this limitation, we aim to completely eliminate the setup stage for interactive 3DGS segmentation while keeping the segmentation time practical (under 1 second). In this work, we present SAGO (Segment Any Gaussians Online), a novel setup-free framework for interactive 3DGS segmentation. By introducing virtual drones, our method reframes the 3D segmentation problem as an online Next-Best-View (NBV) planning task formulated within a Markov process. Extensive experiments demonstrate that SAGO can extract clean 3D assets directly from 3D Gaussians with sub-second latency, thereby enabling a broad range of downstream applications such as object manipulation and scene editing. Moreover, our method achieves over a 50x speedup compared to the previous setup-free 3DGS segmentation frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01578v1">MVFusion-GS: Motion-Variance Guided Temporal Attention for High-Quality Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables real-time novel view synthesis for static scenes. Extending it to dynamic scenes via deformation fields has recently attracted significant attention, particularly for dynamic scene reconstructionband distractor-free. However, existing deformation networks lack explicit motion awareness: they neither capture long-term motion intensity nor exploit short-term temporal coherence, leading to inaccurate foreground deformation and pseudo-static residuals in the background. We present MVFusion-GS, a method that enhances deformation networks with two complementary motion-aware mechanisms. The Motion-Variance Guided Refinement aggregates per-Gaussian deformation statistics across time to estimate motion variance and uses it to guide dynamic-static separation during deformation prediction. The MotionFormer Temporal Attention module applies Transformer self-attention over neighboring timesteps to model local motion dependencies and improve temporal consistency. Extensive experiments on both dynamic scene reconstruction and distractor-free reconstruction benchmarks demonstrate state-of-the-art performance, showing that explicit motion awareness improves both foreground motion modeling and static background reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01556v1">Mind the Gap: Standard 3DGS Evaluation Primarily Measures Near-Trajectory Interpolation</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Standard MipNeRF360-style 3D Gaussian Splatting (3DGS) evaluation holds out every N-th frame -- but these frames have trained neighbors on both sides, so the metric measures near-trajectory interpolation rather than spatial generalization. We introduce a fair matched-count protocol that isolates this effect: both arms train on the same number of images and differ only in whether the holdout is spread evenly (interpolation) or forms a contiguous spatial sector (extrapolation). Our primary finding is a large, consistent interpolation-extrapolation gap of 3~12dB -- several times the differences typically reported between competing methods. The gap is robust to training noise, is in two cases large enough to flip a method ranking under multi-seed confirmation, and -- crucially -- persists across three representation families, including a non-Gaussian volumetric neural radiance field (NeRF), so it reflects spatial coverage rather than any one representation. Diagnostically, it is dominated by a diffuse/geometry-proxy component and tracks each view's angular distance to its nearest training view, a zero-cost signal that also guides capture planning; loss-side regularization yields only marginal gains. Standard holdouts remain useful for near-trajectory rendering but should not, alone, be read as evidence of spatial generalization. Prior work notes protocol sensitivity; ours is, to our knowledge, the first to combine matched-count paired holdout, cross-representation quantification, and a diagnostic analysis Table 1. We describe a spatial-holdout benchmark toolkit with standardized splits and baselines for 16 scenes, which we are preparing for public release.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.29453v2">Resonant Brane Splatting for Arbitrary-Scale Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Arbitrary-Scale Super-Resolution (ASR) reconstructs images at continuous magnification factors. Recent methods accelerate inference by replacing computationally heavy implicit neural decoders with explicit 2D Gaussian Splatting (GS). However, since standard Gaussians are smooth low-pass primitives, modeling edges and fine textures requires multiple overlapping, well-aligned splats, which creates severe bottlenecks during rasterization. To address this, we introduce Resonant Brane Splatting (RBS), a feed-forward ASR framework. RBS replaces flat Gaussians with Branes: expressive primitives that emit spatially varying colors to natively model local contrast and complex textures within a single footprint. We achieve this by augmenting the standard Gaussian envelope with internal Gaussian-Hermite modes, assigning a distinct color coefficient to each. The zero-order mode recovers standard GS, while higher-order modes capture high frequencies. We predict Brane parameters directly from low-resolution features. Because Branes provide a mathematically richer formulation than simple Gaussians, far fewer primitives need to overlap to reconstruct a given target pixel. To exploit this, we introduce an efficient fully differentiable rasterizer with a precise culling strategy based on the classical quantum turning point. This allows us to safely skip negligible regions, drastically reducing the rendering overhead. Experiments on standard ASR benchmarks show that RBS improves reconstruction quality over implicit and GS baselines, while achieving superior speed-quality trade-off than prior GS methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01200v1">FastBridge: Closing the Model-Based Realization Gap in Safety Filters on 3D Gaussian Splatting for Fast Quadrotor Flight</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 preprint, 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Fast quadrotor flight requires safe obstacle avoidance under tight onboard compute limits. While 3D Gaussian Splatting (3DGS) provides a continuous, geometry-aware scene representation for perception-driven navigation, existing 3DGS safety filters use reduced-order models such as single- and double-integrators that ignore actuator limits and assume commanded accelerations are realized instantaneously. Building on an analytic collision cone barrier for 3DGS, we introduce a nonlinear, actuator-aware safety filter enforced through the full quadrotor dynamics. We derive a high-relative-degree collision cone exponential CBF and a backup CBF that preserves QP feasibility under input constraints using a forward-simulated backup policy. Compared with a state-of-the-art 3DGS safety filter, our approach reduces trajectory jerk by 47% and runs 2.25 times faster. We validate the method in simulation and on hardware for real-time navigation in cluttered, perception-derived environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00959v1">GaussianEmoTalker: Real-Time Emotional Talking Head Synthesis with Audio-Driven and Blendshape-Based 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Audio-driven talking head synthesis has achieved impressive progress in lip synchronization and visual quality, yet generating expressive emotional avatars with controllable intensity remains challenging, especially under real-time constraints. In this paper, we present GaussianEmoTalker, an audio-driven framework for real-time emotional talking head synthesis based on 3D Gaussian Splatting. Instead of directly predicting the final emotional avatar from speech, we formulate emotional animation as a neutral-to-emotional residual deformation problem. GaussianEmoTalker first constructs an identity-specific neutral talking space with GaussianBlendshapes, which provides high-fidelity Gaussian attributes and phoneme-synchronized neutral motion. It then predicts an emotion-conditioned residual deformation by combining mesh displacement cues, audio features, emotion categories, and intensity encodings. To fuse these heterogeneous signals, we introduce a spatial-audio-emotion attention module that estimates the offsets of Gaussian attributes for expressive and temporally stable rendering. Extensive experiments demonstrate that GaussianEmoTalker achieves competitive video quality, accurate lip synchronization, controllable emotional expression, and real-time rendering compared with recent emotional talking head methods. Our project page is available at https://njust-yang.github.io/GaussianEmoTalker.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00885v1">Improving Sparse-View 3DGS Generalization via Flat Minima Optimization</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 Accepted to ECCV 2026. Project Page: https://kangrnin.github.io/FlatMinGS
    </div>
    <details class="paper-abstract">
      Recent advances in neural rendering have established 3D Gaussian Splatting (3DGS) as a highly efficient representation for novel view synthesis, enabling fast training and real-time rendering with strong fidelity. However, when supervision is limited to sparse input views, 3DGS tends to overfit to the observed images and generalize poorly to unseen viewpoints. We address this challenge from the perspective of flat minima (FM) optimization, which seeks solutions that remain stable under small parameter perturbations. Viewing Gaussian parameters as trainable weights, we adapt FM principles to the geometric and dynamic nature of 3DGS with a lightweight training framework. Our method regularizes optimization with controlled Gaussian perturbations that account for each Gaussian's anisotropy and the training progress, preserving fine details while improving robustness to sparse-view overfitting. To further stabilize this flat minima optimization process, we introduce periodic reinitialization, which temporarily returns non-positional parameters to their initial states for a short window. Together, these techniques integrate seamlessly into existing 3DGS pipelines without architectural changes. Experiments on LLFF and Mip-NeRF360 datasets demonstrate improved quantitative metrics and perceptual quality under sparse-view supervision, producing reconstructions that are sharper, more stable, and better generalized to novel viewpoints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01290v1">AnchorSplat: Fast and Structure Consistent Detail Synthesis for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 Accepted by ECCV2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful representation for high-fidelity rendering. However, existing assets often suffer from quality bottlenecks such as missing details and texture noise. Prior attempts to enhance these assets via 2D image processing introduce multi-view inconsistencies and high computational costs. In this paper, we propose a novel 3D-native refinement paradigm named AnchorSplat. AnchorSplat is an end-to-end deep network operating directly on 3D structures, avoiding the expensive optimization overhead of traditional 3D-2D-3D pipelines. Crucially, AnchorSplat is a strictly source-free solution requiring no original multi-view images. Central to the proposed method is the Point Anchor Mechanism, which enforces geometric consistency via local offset constraints, mitigating ill-posed mapping and gradient confounding. Furthermore, AnchorSplat replaces iterative densification with a single-pass multiplication mechanism. To facilitate research, we construct 3DGS-SR, the first large-scale benchmark for this task. Experiments demonstrate state-of-the-art results on the 3DGS-SR dataset, with throughput up to $10^5$ times faster than optimization methods. Notably, AnchorSplat exhibits robust zero-shot generalization across diverse data distributions, including generative model outputs and real-world scans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08997v2">SkipGS: Post-Densification Backward Skipping for Efficient 3DGS Training</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 Code is available at https://github.com/ASU-ESIC-FAN-Lab/SkipGS
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves real-time novel-view synthesis by optimizing millions of anisotropic Gaussians, yet its training remains expensive, with the backward pass dominating runtime in the post-densification refinement phase. We observe substantial update redundancy in this phase: many sampled views have near-plateaued losses and provide diminishing gradient benefits, but standard training still runs full backpropagation. We propose SkipGS with a novel view-adaptive backward gating mechanism for efficient post-densification training. SkipGS always performs the forward pass to update per-view loss statistics, and selectively skips backward passes when the sampled view's loss is consistent with its recent per-view baseline, while enforcing a minimum backward budget for stable optimization. On Mip-NeRF 360, compared to 3DGS, SkipGS reduces end-to-end training time by 23.1%, driven by a 42.0% reduction in post-densification time, with comparable reconstruction quality. Because it only changes when to backpropagate without modifying the renderer, representation, or loss, SkipGS is plug-and-play and compatible with other complementary efficiency strategies, enabling additive speedups. Code is available at https://github.com/ASU-ESIC-FAN-Lab/SkipGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00673v1">Path Planning in Physically Viable World Models</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 18 pages, 7 figures, submitted to CORL
    </div>
    <details class="paper-abstract">
      Robots deployed in unstructured outdoor environments often plan from scene reconstructions collected before deployment because operators cannot remap large or remote sites before every mission. As a result, robots must make long-horizon planning decisions using stale maps that assume the terrain remains unchanged, even though physical changes to the environment may render previously feasible routes unsafe or unreachable at execution time. We present a physically viable world model for evaluating what-if queries for robot navigation under future terrain change. The system augments reconstructed 3D Gaussian splat scenes with physics-based simulation to generate physically modified versions of the same environment without recollecting sensor data or rebuilding the map. We then implement a terrain-aware planner that accounts for physical events, obstacles, and deformations that are simulated by the world model. This allows robots and human operators to evaluate whether planned routes remain feasible before committing to a planned route, particularly in constrained environments where retreat or recovery may become impossible once conditions change. We evaluate the system on a real outdoor field site in Central Texas using simulated flooding across multiple severity levels. We measure route and mission feasibility as terrain conditions deteriorate under physically simulated interventions. Our results show that physically viable world models expose long-horizon route failures and rerouting behavior that are not apparent when planning only on the original reconstructed environment, allowing robots to evaluate how future terrain changes may affect route feasibility before deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2408.16982v3">2DGH: 2D Gaussian-Hermite Splatting for High-quality Rendering and Better Geometry Features</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 12 pages, 11 figures
    </div>
    <details class="paper-abstract">
      2D Gaussian Splatting has recently emerged as a significant method in 3D reconstruction, enabling novel view synthesis and geometry reconstruction simultaneously. While the well-known Gaussian kernel is broadly used, its lack of anisotropy and deformation ability leads to dim and vague edges at object silhouettes, limiting the reconstruction quality of current Gaussian splatting methods. To enhance the representation power, we draw inspiration from quantum physics and propose to use the Gaussian-Hermite kernel as the new primitive in Gaussian splatting. The new kernel takes a unified mathematical form and extends the Gaussian function, which serves as the zero-rank special case in the updated general formulation. Our experiments demonstrate that the proposed Gaussian-Hermite kernel achieves improved performance over traditional Gaussian Splatting kernels on both geometry reconstruction and novel-view synthesis tasks. Specifically, on the DTU dataset, our method yields more accurate geometry reconstruction, while on datasets such as MipNeRF360 and our customized Detail dataset, it achieves better results in novel-view synthesis. These results highlight the potential of the Gaussian-Hermite kernel for high-quality 3D reconstruction and rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03337v3">FreeTimeGS++: Secrets of Dynamic Gaussian Splatting and Their Principles</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 Project page: https://yklcs.com/ftgspp
    </div>
    <details class="paper-abstract">
      Recent progress in 4D Gaussian Splatting (4DGS) has achieved impressive dynamic scene reconstruction results. While these methods demonstrate remarkable performance, the specific factors behind their gains remain underexplored, making a systematic understanding of the underlying principles challenging. In this paper, we perform a comprehensive analysis of these hidden factors to provide a clearer perspective on the 4DGS framework. We first establish a controlled baseline, FreeTimeGS_ours, by formalizing and reproducing the heuristics of the state-of-the-art FreeTimeGS. Using this framework, we examine 4DGS along its fundamental axes and identify practical secrets, including the emergent temporal partitioning driven by Gaussian durations and the decoupling between photometric fidelity and motion behavior. Based on these insights, we propose FreeTimeGS++, a principled method that employs gated marginalization, UFM-guided initialization, and color correction to improve stability and reproducibility. Our approach yields reproducible results with reduced run-to-run variance.
    </details>
</div>
