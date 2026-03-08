# gaussian splatting - 2026_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08491v2">Splat the Net: Radiance Fields with Splattable Neural Primitives</a></div>
    <div class="paper-meta">
      📅 2026-02-28
    </div>
    <details class="paper-abstract">
      Radiance fields have emerged as a predominant representation for modeling 3D scene appearance. Neural formulations such as Neural Radiance Fields provide high expressivity but require costly ray marching for rendering, whereas primitive-based methods such as 3D Gaussian Splatting offer real-time efficiency through splatting, yet at the expense of representational power. Inspired by advances in both these directions, we introduce splattable neural primitives, a new volumetric representation that reconciles the expressivity of neural models with the efficiency of primitive-based splatting. Each primitive encodes a bounded neural density field parameterized by a shallow neural network. Our formulation admits an exact analytical solution for line integrals, enabling efficient computation of perspectively accurate splatting kernels. As a result, our representation supports integration along view rays without the need for costly ray marching. The primitives flexibly adapt to scene geometry and, being larger than prior analytic primitives, reduce the number required per scene. On novel-view synthesis benchmarks, our approach matches the quality and speed of 3D Gaussian Splatting while using $10\times$ fewer primitives and $6\times$ fewer parameters. These advantages arise directly from the representation itself, without reliance on complex control or adaptation frameworks. The project page is https://vcai.mpi-inf.mpg.de/projects/SplatNet/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00697v1">TokenSplat: Token-aligned 3D Gaussian Splatting for Feed-forward Pose-free Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-28
    </div>
    <details class="paper-abstract">
      We present TokenSplat, a feed-forward framework for joint 3D Gaussian reconstruction and camera pose estimation from unposed multi-view images. At its core, TokenSplat introduces a Token-aligned Gaussian Prediction module that aligns semantically corresponding information across views directly in the feature space. Guided by coarse token positions and fusion confidence, it aggregates multi-scale contextual features to enable long-range cross-view reasoning and reduce redundancy from overlapping Gaussians. To further enhance pose robustness and disentangle viewpoint cues from scene semantics, TokenSplat employs learnable camera tokens and an Asymmetric Dual-Flow Decoder (ADF-Decoder) that enforces directionally constrained communication between camera and image tokens. This maintains clean factorization within a feed-forward architecture, enabling coherent reconstruction and stable pose estimation without iterative refinement. Extensive experiments demonstrate that TokenSplat achieves higher reconstruction fidelity and novel-view synthesis quality in pose-free settings, and significantly improves pose estimation accuracy compared to prior pose-free methods. Project page: https://kidleyh.github.io/tokensplat/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09663v2">PD$^{2}$GS: Part-Level Decoupling and Continuous Deformation of Articulated Objects via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-28
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Articulated objects are ubiquitous and important in robotics, AR/VR, and digital twins. Most self-supervised methods for articulated object modeling reconstruct discrete interaction states and relate them via cross-state geometric consistency, yielding representational fragmentation and drift that hinder smooth control of articulated configurations. We introduce PD$^{2}$GS, a novel framework that learns a shared canonical Gaussian field and models the arbitrary interaction state as its continuous deformation, jointly encoding geometry and kinematics. By associating each interaction state with a latent code and refining part boundaries using generic vision priors, PD$^{2}$GS enables accurate and reliable part-level decoupling while enforcing mutual exclusivity between parts and preserving scene-level coherence. This unified formulation supports part-aware reconstruction, fine-grained continuous control, and accurate kinematic modeling, all without manual supervision. To assess realism and generalization, we release RS-Art, a real-to-sim RGB-D dataset aligned with reverse-engineered 3D models, supporting real-world evaluation. Extensive experiments demonstrate that PD$^{2}$GS surpasses prior methods in geometric and kinematic accuracy, and in consistency under continuous control, both on synthetic and real data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00500v1">Zero-Shot Robotic Manipulation via 3D Gaussian Splatting-Enhanced Multimodal Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-28
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Existing end-to-end approaches of robotic manipulation often lack generalization to unseen objects or tasks due to limited data and poor interpretability. While recent Multimodal Large Language Models (MLLMs) demonstrate strong commonsense reasoning, they struggle with geometric and spatial understanding required for pose prediction. In this paper, we propose RobMRAG, a 3D Gaussian Splatting-Enhanced Multimodal Retrieval-Augmented Generation (MRAG) framework for zero-shot robotic manipulation. Specifically, we construct a multi-source manipulation knowledge base containing object contact frames, task completion frames, and pose parameters. During inference, a Hierarchical Multimodal Retrieval module first employs a three-priority hybrid retrieval strategy to find task-relevant object prototypes, then selects the geometrically closest reference example based on pixel-level similarity and Instance Matching Distance (IMD). We further introduce a 3D-Aware Pose Refinement module based on 3D Gaussian Splatting into the MRAG framework, which aligns the pose of the reference object to the target object in 3D space. The aligned results are reprojected onto the image plane and used as input to the MLLM to enhance the generation of the final pose parameters. Extensive experiments show that on a test set containing 30 categories of household objects, our method improves the success rate by 7.76% compared to the best-performing zero-shot baseline under the same setting, and by 6.54% compared to the state-of-the-art supervised baseline. Our results validate that RobMRAG effectively bridges the gap between high-level semantic reasoning and low-level geometric execution, enabling robotic systems that generalize to unseen objects while remaining inherently interpretable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00492v1">ArtiFixer: Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2026-02-28
      | 💬 Video results: https://artifixer2026.github.io/
    </div>
    <details class="paper-abstract">
      Per-scene optimization methods such as 3D Gaussian Splatting provide state-of-the-art novel view synthesis quality but extrapolate poorly to under-observed areas. Methods that leverage generative priors to correct artifacts in these areas hold promise but currently suffer from two shortcomings. The first is scalability, as existing methods use image diffusion models or bidirectional video models that are limited in the number of views they can generate in a single pass (and thus require a costly iterative distillation process for consistency). The second is quality itself, as generators used in prior work tend to produce outputs that are inconsistent with existing scene content and fail entirely in completely unobserved regions. To solve these, we propose a two-stage pipeline that leverages two key insights. First, we train a powerful bidirectional generative model with a novel opacity mixing strategy that encourages consistency with existing observations while retaining the model's ability to extrapolate novel content in unseen areas. Second, we distill it into a causal auto-regressive model that generates hundreds of frames in a single pass. This model can directly produce novel views or serve as pseudo-supervision to improve the underlying 3D representation in a simple and highly efficient manner. We evaluate our method extensively and demonstrate that it can generate plausible reconstructions in scenarios where existing approaches fail completely. When measured on commonly benchmarked datasets, we outperform existing all existing baselines by a wide margin, exceeding prior state-of-the-art methods by 1-3 dB PSNR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.26455v3">Stylos: Multi-View 3D Stylization with Single-Forward Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-28
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      We present Stylos, a single-forward 3D Gaussian framework for 3D style transfer that operates on unposed content, from a single image to a multi-view collection, conditioned on a separate reference style image. Stylos synthesizes a stylized 3D Gaussian scene without per-scene optimization or precomputed poses, achieving geometry-aware, view-consistent stylization that generalizes to unseen categories, scenes, and styles. At its core, Stylos adopts a Transformer backbone with two pathways: geometry predictions retain self-attention to preserve geometric fidelity, while style is injected via global cross-attention to enforce visual consistency across views. With the addition of a voxel-based 3D style loss that aligns aggregated scene features to style statistics, Stylos enforces view-consistent stylization while preserving geometry. Experiments across multiple datasets demonstrate that Stylos delivers high-quality zero-shot stylization, highlighting the effectiveness of global style-content coupling, the proposed 3D style loss, and the scalability of our framework from single view to large-scale multi-view settings. Our codes are available at https://github.com/HanzhouLiu/Stylos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00418v1">Station2Radar: query conditioned gaussian splatting for precipitation field</a></div>
    <div class="paper-meta">
      📅 2026-02-28
      | 💬 This paper was accepted to ICLR 2026
    </div>
    <details class="paper-abstract">
      Precipitation forecasting relies on heterogeneous data. Weather radar is accurate, but coverage is geographically limited and costly to maintain. Weather stations provide accurate but sparse point measurements, while satellites offer dense, high-resolution coverage without direct rainfall retrieval. To overcome these limitations, we propose Query-Conditioned Gaussian Splatting (QCGS), the first framework to fuse automatic weather station (AWS) observations with satellite imagery for generating precipitation fields. Unlike conventional 2D Gaussian splatting, which renders the entire image plane, QCGS selectively renders only queried precipitation regions, avoiding unnecessary computation in non-precipitating areas while preserving sharp precipitation structures. The framework combines a radar point proposal network that identifies rainfall-support locations with an implicit neural representation (INR) network that predicts Gaussian parameters for each point. QCGS enables efficient, resolution-flexible precipitation field generation in real time. Through extensive evaluation with benchmark precipitation products, QCGS demonstrates over 50\% improvement in RMSE compared to conventional gridded precipitation products, and consistently maintains high performance across multiple spatiotemporal scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.24136v1">Prune Wisely, Reconstruct Sharply: Compact 3D Gaussian Splatting via Adaptive Pruning and Difference-of-Gaussian Primitives</a></div>
    <div class="paper-meta">
      📅 2026-02-27
      | 💬 CVPR2026
    </div>
    <details class="paper-abstract">
      Recent significant advances in 3D scene representation have been driven by 3D Gaussian Splatting (3DGS), which has enabled real-time rendering with photorealistic quality. 3DGS often requires a large number of primitives to achieve high fidelity, leading to redundant representations and high resource consumption, thereby limiting its scalability for complex or large-scale scenes. Consequently, effective pruning strategies and more expressive primitives that can reduce redundancy while preserving visual quality are crucial for practical deployment. We propose an efficient, integrated reconstruction-aware pruning strategy that adaptively determines pruning timing and refining intervals based on reconstruction quality, thus reducing model size while enhancing rendering quality. Moreover, we introduce a 3D Difference-of-Gaussians primitive that jointly models both positive and negative densities in a single primitive, improving the expressiveness of Gaussians under compact configurations. Our method significantly improves model compactness, achieving up to 90\% reduction in Gaussian-count while delivering visual quality that is similar to, or in some cases better than, that produced by state-of-the-art methods. Code will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.07021v3">MEGS$^{2}$: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning</a></div>
    <div class="paper-meta">
      📅 2026-02-27
      | 💬 20 pages, 8 figures. Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a dominant novel-view synthesis technique, but its high memory consumption severely limits its applicability on edge devices. A growing number of 3DGS compression methods have been proposed to make 3DGS more efficient, yet most only focus on storage compression and fail to address the critical bottleneck of rendering memory. To address this problem, we introduce MEGS$^{2}$, a novel memory-efficient framework that tackles this challenge by jointly optimizing two key factors: the total primitive number and the parameters per primitive, achieving unprecedented memory compression. Specifically, we replace the memory-intensive spherical harmonics with lightweight, arbitrarily oriented spherical Gaussian lobes as our color representations. More importantly, we propose a unified soft pruning framework that models primitive-number and lobe-number pruning as a single constrained optimization problem. Experiments show that MEGS$^{2}$ achieves a 50% static VRAM reduction and a 40% rendering VRAM reduction compared to existing methods, while maintaining comparable rendering quality. Project page: https://megs-2.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.24020v1">SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-27
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      3D super-resolution (3DSR) aims to reconstruct high-resolution (HR) 3D scenes from low-resolution (LR) multi-view images. Existing methods rely on dense LR inputs and per-scene optimization, which restricts the high-frequency priors for constructing HR 3D Gaussian Splatting (3DGS) to those inherited from pretrained 2D super-resolution (2DSR) models. This severely limits reconstruction fidelity, cross-scene generalization, and real-time usability. We propose to reformulate 3DSR as a direct feed-forward mapping from sparse LR views to HR 3DGS representations, enabling the model to autonomously learn 3D-specific high-frequency geometry and appearance from large-scale, multi-scene data. This fundamentally changes how 3DSR acquires high-frequency knowledge and enables robust generalization to unseen scenes. Specifically, we introduce SR3R, a feed-forward framework that directly predicts HR 3DGS representations from sparse LR views via the learned mapping network. To further enhance reconstruction fidelity, we introduce Gaussian offset learning and feature refinement, which stabilize reconstruction and sharpen high-frequency details. SR3R is plug-and-play and can be paired with any feed-forward 3DGS reconstruction backbone: the backbone provides an LR 3DGS scaffold, and SR3R upscales it to an HR 3DGS. Extensive experiments across three 3D benchmarks demonstrate that SR3R surpasses state-of-the-art (SOTA) 3DSR methods and achieves strong zero-shot generalization, even outperforming SOTA per-scene optimization methods on unseen scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21644v2">DAGS-SLAM: Dynamic-Aware 3DGS SLAM via Spatiotemporal Motion Probability and Uncertainty-Aware Scheduling</a></div>
    <div class="paper-meta">
      📅 2026-02-27
    </div>
    <details class="paper-abstract">
      Mobile robots and IoT devices demand real-time localization and dense reconstruction under tight compute and energy budgets. While 3D Gaussian Splatting (3DGS) enables efficient dense SLAM, dynamic objects and occlusions still degrade tracking and mapping. Existing dynamic 3DGS-SLAM often relies on heavy optical flow and per-frame segmentation, which is costly for mobile deployment and brittle under challenging illumination. We present DAGS-SLAM, a dynamic-aware 3DGS-SLAM system that maintains a spatiotemporal motion probability (MP) state per Gaussian and triggers semantics on demand via an uncertainty-aware scheduler. DAGS-SLAM fuses lightweight YOLO instance priors with geometric cues to estimate and temporally update MP, propagates MP to the front-end for dynamic-aware correspondence selection, and suppresses dynamic artifacts in the back-end via MP-guided optimization. Experiments on public dynamic RGB-D benchmarks show improved reconstruction and robust tracking while sustaining real-time throughput on a commodity GPU, demonstrating a practical speed-accuracy tradeoff with reduced semantic invocations toward mobile deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06846v2">DynFOA: Generating First-Order Ambisonics with Conditional Diffusion for Dynamic and Acoustically Complex 360-Degree Videos</a></div>
    <div class="paper-meta">
      📅 2026-02-27
    </div>
    <details class="paper-abstract">
      Spatial audio is crucial for creating compelling immersive 360-degree video experiences. However, generating realistic spatial audio, such as first-order ambisonics (FOA), from 360-degree videos in complex acoustic scenes remains challenging. Existing methods often overlook the dynamic nature and acoustic complexity of 360-degree scenes, fail to fully account for dynamic sound sources, and neglect complex environmental effects such as occlusion, reflections, and reverberation, which are influenced by scene geometries and materials. We propose DynFOA, a framework based on dynamic acoustic perception and conditional diffusion, for generating high-fidelity FOA from 360-degree videos. DynFOA first performs visual processing via a video encoder, which detects and localizes multiple dynamic sound sources, estimates their depth and semantics, and reconstructs the scene geometry and materials using a 3D Gaussian Splatting. This reconstruction technique accurately models occlusion, reflections, and reverberation based on the geometries and materials of the reconstructed 3D scene and the listener's viewpoint. The audio encoder then captures the spatial motion and temporal 4D sound source trajectories to fine-tune the diffusion-based FOA generator. The fine-tuned FOA generator adjusts spatial cues in real time, ensuring consistent directional fidelity during listener head rotation and complex environmental changes. Extensive evaluations demonstrate that DynFOA consistently outperforms existing methods across metrics such as spatial accuracy, acoustic fidelity, and distribution matching, while also improving the user experience. Therefore, DynFOA provides a robust and scalable approach to rendering realistic dynamic spatial audio for VR and immersive media applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.12768v3">Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-27
      | 💬 Accepted to ICLR 2026. Project page: https://tamu-visual-ai.github.io/usplat4d/
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic 3D scenes from monocular input is fundamentally under-constrained, with ambiguities arising from occlusion and extreme novel views. While dynamic Gaussian Splatting offers an efficient representation, vanilla models optimize all Gaussian primitives uniformly, ignoring whether they are well or poorly observed. This limitation leads to motion drifts under occlusion and degraded synthesis when extrapolating to unseen views. We argue that uncertainty matters: Gaussians with recurring observations across views and time act as reliable anchors to guide motion, whereas those with limited visibility are treated as less reliable. To this end, we introduce USplat4D, a novel Uncertainty-aware dynamic Gaussian Splatting framework that propagates reliable motion cues to enhance 4D reconstruction. Our approach estimates time-varying per-Gaussian uncertainty and leverages it to construct a spatio-temporal graph for uncertainty-aware optimization. Experiments on diverse real and synthetic datasets show that explicitly modeling uncertainty consistently improves dynamic Gaussian Splatting models, yielding more stable geometry under occlusion and high-quality synthesis at extreme viewpoints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19766v2">One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image</a></div>
    <div class="paper-meta">
      📅 2026-02-27
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Generating explorable 3D scenes from a single image is a highly challenging problem in 3D vision. Existing methods struggle to support free exploration, often producing severe geometric distortions and noisy artifacts when the viewpoint moves far from the original perspective. We introduce \textbf{One2Scene}, an effective framework that decomposes this ill-posed problem into three tractable sub-tasks to enable immersive explorable scene generation. We first use a panorama generator to produce anchor views from a single input image as initialization. Then, we lift these 2D anchors into an explicit 3D geometric scaffold via a generalizable, feed-forward Gaussian Splatting network. Instead of treating the panorama as a single image for reconstruction, we project it into multiple sparse anchor views and reformulate the reconstruction task as multi-view stereo matching, which allows us to leverage robust geometric priors learned from large-scale multi-view datasets. A bidirectional feature fusion module is used to enforce cross-view consistency, yielding an efficient and geometrically reliable scaffold. Finally, the scaffold serves as a strong prior for a novel view generator to produce photorealistic and geometrically accurate views at arbitrary cameras. By explicitly conditioning on a 3D-consistent scaffold to perform reconstruction, One2Scene works stably under large camera motions, supporting immersive scene exploration. Extensive experiments show that One2Scene substantially outperforms state-of-the-art methods in panorama depth estimation, feed-forward 360° reconstruction, and explorable 3D scene generation. Project page: https://one2scene5406.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23559v1">No Calibration, No Depth, No Problem: Cross-Sensor View Synthesis with 3D Consistency</a></div>
    <div class="paper-meta">
      📅 2026-02-27
      | 💬 CVPR 2026 Main Conference. Project page: https://choyingw.github.io/3d-rgbx.github.io/
    </div>
    <details class="paper-abstract">
      We present the first study of cross-sensor view synthesis across different modalities. We examine a practical, fundamental, yet widely overlooked problem: getting aligned RGB-X data, where most RGB-X prior work assumes such pairs exist and focuses on modality fusion, but it empirically requires huge engineering effort in calibration. We propose a match-densify-consolidate method. First, we perform RGB-X image matching followed by guided point densification. Using the proposed confidence-aware densification and self-matching filtering, we attain better view synthesis and later consolidate them in 3D Gaussian Splatting (3DGS). Our method uses no 3D priors for X-sensor and only assumes nearly no-cost COLMAP for RGB. We aim to remove the cumbersome calibration for various RGB-X sensors and advance the popularity of cross-sensor learning by a scalable solution that breaks through the bottleneck in large-scale real-world RGB-X data collection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23172v1">Latent Gaussian Splatting for 4D Panoptic Occupancy Tracking</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Capturing 4D spatiotemporal surroundings is crucial for the safe and reliable operation of robots in dynamic environments. However, most existing methods address only one side of the problem: they either provide coarse geometric tracking via bounding boxes, or detailed 3D structures like voxel-based occupancy that lack explicit temporal association. In this work, we present Latent Gaussian Splatting for 4D Panoptic Occupancy Tracking (LaGS) that advances spatiotemporal scene understanding in a holistic direction. Our approach incorporates camera-based end-to-end tracking with mask-based multi-view panoptic occupancy prediction, and addresses the key challenge of efficiently aggregating multi-view information into 3D voxel grids via a novel latent Gaussian splatting approach. Specifically, we first fuse observations into 3D Gaussians that serve as a sparse point-centric latent representation of the 3D scene, and then splat the aggregated features onto a 3D voxel grid that is decoded by a mask-based segmentation head. We evaluate LaGS on the Occ3D nuScenes and Waymo datasets, achieving state-of-the-art performance for 4D panoptic occupancy tracking. We make our code available at https://lags.cs.uni-freiburg.de/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24421v3">Proxy-GS: Unified Occlusion Priors for Training and Inference in Structured 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 Project page: https://gyy456.github.io/Proxy-GS
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as an efficient approach for achieving photorealistic rendering. Recent MLP-based variants further improve visual fidelity but introduce substantial decoding overhead during rendering. To alleviate computation cost, several pruning strategies and level-of-detail (LOD) techniques have been introduced, aiming to effectively reduce the number of Gaussian primitives in large-scale scenes. However, our analysis reveals that significant redundancy still remains due to the lack of occlusion awareness. In this work, we propose Proxy-GS, a novel pipeline that exploits a proxy to introduce Gaussian occlusion awareness from any view. At the core of our approach is a fast proxy system capable of producing precise occlusion depth maps at a resolution of 1000x1000 under 1ms. This proxy serves two roles: first, it guides the culling of anchors and Gaussians to accelerate rendering speed. Second, it guides the densification towards surfaces during training, avoiding inconsistencies in occluded regions, and improving the rendering quality. In heavily occluded scenarios, such as the MatrixCity Streets dataset, Proxy-GS not only equips MLP-based Gaussian splatting with stronger rendering capability but also achieves faster rendering speed. Specifically, it achieves more than 2.5x speedup over Octree-GS, and consistently delivers substantially higher rendering quality. Code will be public upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23040v1">PackUV: Packed Gaussian UV Maps for 4D Volumetric Video</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 https://ivl.cs.brown.edu/packuv
    </div>
    <details class="paper-abstract">
      Volumetric videos offer immersive 4D experiences, but remain difficult to reconstruct, store, and stream at scale. Existing Gaussian Splatting based methods achieve high-quality reconstruction but break down on long sequences, temporal inconsistency, and fail under large motions and disocclusions. Moreover, their outputs are typically incompatible with conventional video coding pipelines, preventing practical applications. We introduce PackUV, a novel 4D Gaussian representation that maps all Gaussian attributes into a sequence of structured, multi-scale UV atlas, enabling compact, image-native storage. To fit this representation from multi-view videos, we propose PackUV-GS, a temporally consistent fitting method that directly optimizes Gaussian parameters in the UV domain. A flow-guided Gaussian labeling and video keyframing module identifies dynamic Gaussians, stabilizes static regions, and preserves temporal coherence even under large motions and disocclusions. The resulting UV atlas format is the first unified volumetric video representation compatible with standard video codecs (e.g., FFV1) without losing quality, enabling efficient streaming within existing multimedia infrastructure. To evaluate long-duration volumetric capture, we present PackUV-2B, the largest multi-view video dataset to date, featuring more than 50 synchronized cameras, substantial motion, and frequent disocclusions across 100 sequences and 2B (billion) frames. Extensive experiments demonstrate that our method surpasses existing baselines in rendering fidelity while scaling to sequences up to 30 minutes with consistent quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14510v3">Structured Image-based Coding for Efficient Gaussian Splatting Compression</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has recently emerged as a state-of-the-art representation for radiance fields, combining real-time rendering with high visual fidelity. However, GS models require storing millions of parameters, leading to large file sizes that impair their use in practical multimedia systems. To address this limitation, this paper introduces GS Image-based Compression (GSICO), a novel GS codec that efficiently compresses pre-trained GS models while preserving perceptual fidelity. The core contribution lies in a mapping procedure that arranges GS parameters into structured images, guided by a novel algorithm that enhances spatial coherence. These GS parameter images are then encoded using a conventional image codec. Experimental evaluations on Tanks and Temples, Deep Blending, and Mip-NeRF360 datasets show that GSICO achieves average compression factors of 20.2x with minimal loss in visual quality, as measured by PSNR, SSIM, and LPIPS. Compared with state-of-the-art GS compression methods, the proposed codec consistently yields superior rate-distortion (RD) trade-offs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16552v2">ST-GS: Vision-Based 3D Semantic Occupancy Prediction with Spatial-Temporal Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 Accepted by ICRA 2026
    </div>
    <details class="paper-abstract">
      3D occupancy prediction is critical for comprehensive scene understanding in vision-centric autonomous driving. Recent advances have explored utilizing 3D semantic Gaussians to model occupancy while reducing computational overhead, but they remain constrained by insufficient multi-view spatial interaction and limited multi-frame temporal consistency. To overcome these issues, in this paper, we propose a novel Spatial-Temporal Gaussian Splatting (ST-GS) framework to enhance both spatial and temporal modeling in existing Gaussian-based pipelines. Specifically, we develop a guidance-informed spatial aggregation strategy within a dual-mode attention mechanism to strengthen spatial interaction in Gaussian representations. Furthermore, we introduce a geometry-aware temporal fusion scheme that effectively leverages historical context to improve temporal continuity in scene completion. Extensive experiments on the large-scale nuScenes occupancy prediction benchmark showcase that our proposed approach not only achieves state-of-the-art performance but also delivers markedly better temporal consistency compared to existing Gaussian-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22800v1">GSTurb: Gaussian Splatting for Atmospheric Turbulence Mitigation</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Atmospheric turbulence causes significant image degradation due to pixel displacement (tilt) and blur, particularly in long-range imaging applications. In this paper, we propose a novel framework for atmospheric turbulence mitigation, GSTurb, which integrates optical flow-guided tilt correction and Gaussian splatting for modeling non-isoplanatic blur. The framework employs Gaussian parameters to represent tilt and blur, and optimizes them across multiple frames to enhance restoration. Experimental results on the ATSyn-static dataset demonstrate the effectiveness of our method, achieving a peak PSNR of 27.67 dB and SSIM of 0.8735. Compared to the state-of-the-art method, GSTurb improves PSNR by 1.3 dB (a 4.5% increase) and SSIM by 0.048 (a 5.8% increase). Additionally, on real datasets, including the TSRWGAN Real-World and CLEAR datasets, GSTurb outperforms existing methods, showing significant improvements in both qualitative and quantitative performance. These results highlight that combining optical flow-guided tilt correction with Gaussian splatting effectively enhances image restoration under both synthetic and real-world turbulence conditions. The code for this method will be available at https://github.com/DuhlLiamz/3DGS_turbulence/tree/main.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22731v1">Sapling-NeRF: Geo-Localised Sapling Reconstruction in Forests for Ecological Monitoring</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Saplings are key indicators of forest regeneration and overall forest health. However, their fine-scale architectural traits are difficult to capture with existing 3D sensing methods, which make quantitative evaluation difficult. Terrestrial Laser Scanners (TLS), Mobile Laser Scanners (MLS), or traditional photogrammetry approaches poorly reconstruct thin branches, dense foliage, and lack the scale consistency needed for long-term monitoring. Implicit 3D reconstruction methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) are promising alternatives, but cannot recover the true scale of a scene and lack any means to be accurately geo-localised. In this paper, we present a pipeline which fuses NeRF, LiDAR SLAM, and GNSS to enable repeatable, geo-localised ecological monitoring of saplings. Our system proposes a three-level representation: (i) coarse Earth-frame localisation using GNSS, (ii) LiDAR-based SLAM for centimetre-accurate localisation and reconstruction, and (iii) NeRF-derived object-centric dense reconstruction of individual saplings. This approach enables repeatable quantitative evaluation and long-term monitoring of sapling traits. Our experiments in forest plots in Wytham Woods (Oxford, UK) and Evo (Finland) show that stem height, branching patterns, and leaf-to-wood ratios can be captured with increased accuracy as compared to TLS. We demonstrate that accurate stem skeletons and leaf distributions can be measured for saplings with heights between 0.5m and 2m in situ, giving ecologists access to richer structural and quantitative data for analysing forest dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.12099v2">G4Splat: Geometry-Guided Gaussian Splatting with Generative Prior</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 ICLR'26. Project page: https://dali-jack.github.io/g4splat-web/
    </div>
    <details class="paper-abstract">
      Despite recent advances in leveraging generative prior from pre-trained diffusion models for 3D scene reconstruction, existing methods still face two critical limitations. First, due to the lack of reliable geometric supervision, they struggle to produce high-quality reconstructions even in observed regions, let alone in unobserved areas. Second, they lack effective mechanisms to mitigate multi-view inconsistencies in the generated images, leading to severe shape-appearance ambiguities and degraded scene geometry. In this paper, we identify accurate geometry as the fundamental prerequisite for effectively exploiting generative models to enhance 3D scene reconstruction. We first propose to leverage the prevalence of planar structures to derive accurate metric-scale depth maps, providing reliable supervision in both observed and unobserved regions. Furthermore, we incorporate this geometry guidance throughout the generative pipeline to improve visibility mask estimation, guide novel view selection, and enhance multi-view consistency when inpainting with video diffusion models, resulting in accurate and consistent scene completion. Extensive experiments on Replica, ScanNet++, DeepBlending and Mip-NeRF 360 show that our method consistently outperforms existing baselines in both geometry and appearance reconstruction, particularly for unobserved regions. Moreover, our method naturally supports single-view inputs and unposed videos, with strong generalizability in both indoor and outdoor scenarios with practical real-world applicability. The project page is available at https://dali-jack.github.io/g4splat-web/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22666v1">ArtPro: Self-Supervised Articulated Object Reconstruction with Adaptive Integration of Mobility Proposals</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Reconstructing articulated objects into high-fidelity digital twins is crucial for applications such as robotic manipulation and interactive simulation. Recent self-supervised methods using differentiable rendering frameworks like 3D Gaussian Splatting remain highly sensitive to the initial part segmentation. Their reliance on heuristic clustering or pre-trained models often causes optimization to converge to local minima, especially for complex multi-part objects. To address these limitations, we propose ArtPro, a novel self-supervised framework that introduces adaptive integration of mobility proposals. Our approach begins with an over-segmentation initialization guided by geometry features and motion priors, generating part proposals with plausible motion hypotheses. During optimization, we dynamically merge these proposals by analyzing motion consistency among spatial neighbors, while a collision-aware motion pruning mechanism prevents erroneous kinematic estimation. Extensive experiments on both synthetic and real-world objects demonstrate that ArtPro achieves robust reconstruction of complex multi-part objects, significantly outperforming existing methods in accuracy and stability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.15468v2">SplatSDF: Boosting SDF-NeRF via Architecture-Level Fusion with Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Signed distance-radiance field (SDF-NeRF) is a promising environment representation that offers both photo-realistic rendering and geometric reasoning such as proximity queries for collision avoidance. However, the slow training speed and convergence of SDF-NeRF hinder their use in practical robotic systems. We propose SplatSDF, a novel SDF-NeRF architecture that accelerates convergence using 3D Gaussian splats (3DGS), which can be quickly pre-trained. Unlike prior approaches that introduce a consistency loss between separate 3DGS and SDF-NeRF models, SplatSDF directly fuses 3DGS at an architectural level by consuming it as an input to SDF-NeRF during training. This is achieved using a novel sparse 3DGS fusion strategy that injects neural embeddings of 3DGS into SDF-NeRF around the object surface, while also permitting inference without 3DGS for minimal operation. Experimental results show SplatSDF achieves 3X faster convergence to the same geometric accuracy than the best baseline, and outperforms state-of-the-art SDF-NeRF methods in terms of chamfer distance and peak signal to noise ratio, unlike consistency loss-based approaches that in fact provide limited gains. We also present computational techniques for accelerating gradient and Hessian steps by 3X. We expect these improvements will contribute to deploying SDF-NeRF on practical systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22596v1">BetterScene: 3D Scene Synthesis with Representation-Aligned Generative Model</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      We present BetterScene, an approach to enhance novel view synthesis (NVS) quality for diverse real-world scenes using extremely sparse, unconstrained photos. BetterScene leverages the production-ready Stable Video Diffusion (SVD) model pretrained on billions of frames as a strong backbone, aiming to mitigate artifacts and recover view-consistent details at inference time. Conventional methods have developed similar diffusion-based solutions to address these challenges of novel view synthesis. Despite significant improvements, these methods typically rely on off-the-shelf pretrained diffusion priors and fine-tune only the UNet module while keeping other components frozen, which still leads to inconsistent details and artifacts even when incorporating geometry-aware regularizations like depth or semantic conditions. To address this, we investigate the latent space of the diffusion model and introduce two components: (1) temporal equivariance regularization and (2) vision foundation model-aligned representation, both applied to the variational autoencoder (VAE) module within the SVD pipeline. BetterScene integrates a feed-forward 3D Gaussian Splatting (3DGS) model to render features as inputs for the SVD enhancer and generate continuous, artifact-free, consistent novel views. We evaluate on the challenging DL3DV-10K dataset and demonstrate superior performance compared to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22571v1">GIFSplat: Generative Prior-Guided Iterative Feed-Forward 3D Gaussian Splatting from Sparse Views</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Feed-forward 3D reconstruction offers substantial runtime advantages over per-scene optimization, which remains slow at inference and often fragile under sparse views. However, existing feed-forward methods still have potential for further performance gains, especially for out-of-domain data, and struggle to retain second-level inference time once a generative prior is introduced. These limitations stem from the one-shot prediction paradigm in existing feed-forward pipeline: models are strictly bounded by capacity, lack inference-time refinement, and are ill-suited for continuously injecting generative priors. We introduce GIFSplat, a purely feed-forward iterative refinement framework for 3D Gaussian Splatting from sparse unposed views. A small number of forward-only residual updates progressively refine current 3D scene using rendering evidence, achieve favorable balance between efficiency and quality. Furthermore, we distill a frozen diffusion prior into Gaussian-level cues from enhanced novel renderings without gradient backpropagation or ever-increasing view-set expansion, thereby enabling per-scene adaptation with generative prior while preserving feed-forward efficiency. Across DL3DV, RealEstate10K, and DTU, GIFSplat consistently outperforms state-of-the-art feed-forward baselines, improving PSNR by up to +2.1 dB, and it maintains second-scale inference time without requiring camera poses or any test-time gradient optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.17605v3">Distractor-free Generalizable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      We present DGGS, a novel framework that addresses the previously unexplored challenge: $\textbf{Distractor-free Generalizable 3D Gaussian Splatting}$ (3DGS). It mitigates 3D inconsistency and training instability caused by distractor data in the cross-scenes generalizable train setting while enabling feedforward inference for 3DGS and distractor masks from references in the unseen scenes. To achieve these objectives, DGGS proposes a scene-agnostic reference-based mask prediction and refinement module during the training phase, effectively eliminating the impact of distractor on training stability. Moreover, we combat distractor-induced artifacts and holes at inference time through a novel two-stage inference framework for references scoring and re-selection, complemented by a distractor pruning mechanism that further removes residual distractor 3DGS-primitive influences. Extensive feedforward experiments on the real and our synthetic data show DGGS's reconstruction capability when dealing with novel distractor scenes. Moreover, our generalizable mask prediction even achieves an accuracy superior to existing scene-specific training methods. Homepage is https://github.com/bbbbby-99/DGGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22565v1">SwiftNDC: Fast Neural Depth Correction for High-Fidelity 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Depth-guided 3D reconstruction has gained popularity as a fast alternative to optimization-heavy approaches, yet existing methods still suffer from scale drift, multi-view inconsistencies, and the need for substantial refinement to achieve high-fidelity geometry. Here, we propose SwiftNDC, a fast and general framework built around a Neural Depth Correction field that produces cross-view consistent depth maps. From these refined depths, we generate a dense point cloud through back-projection and robust reprojection-error filtering, obtaining a clean and uniformly distributed geometric initialization for downstream reconstruction. This reliable dense geometry substantially accelerates 3D Gaussian Splatting (3DGS) for mesh reconstruction, enabling high-quality surfaces with significantly fewer optimization iterations. For novel-view synthesis, SwiftNDC can also improve 3DGS rendering quality, highlighting the benefits of strong geometric initialization. We conduct a comprehensive study across five datasets, including two for mesh reconstruction, as well as three for novel-view synthesis. SwiftNDC consistently reduces running time for accurate mesh reconstruction and boosts rendering fidelity for view synthesis, demonstrating the effectiveness of combining neural depth refinement with robust geometric initialization for high-fidelity and efficient 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03312v3">Universal Beta Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-26
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      We introduce Universal Beta Splatting (UBS), a unified framework that generalizes 3D Gaussian Splatting to N-dimensional anisotropic Beta kernels for explicit radiance field rendering. Unlike fixed Gaussian primitives, Beta kernels enable controllable dependency modeling across spatial, angular, and temporal dimensions within a single representation. Our unified approach captures complex light transport effects, handles anisotropic view-dependent appearance, and models scene dynamics without requiring auxiliary networks or specific color encodings. UBS maintains backward compatibility by approximating to Gaussian Splatting as a special case, guaranteeing plug-in usability and lower performance bounds. The learned Beta parameters naturally decompose scene properties into interpretable without explicit supervision: spatial (surface vs. texture), angular (diffuse vs. specular), and temporal (static vs. dynamic). Our CUDA-accelerated implementation achieves real-time rendering while consistently outperforming existing methods across static, view-dependent, and dynamic benchmarks, establishing Beta kernels as a scalable universal primitive for radiance field rendering. Our project website is available at https://rongliu-leo.github.io/universal-beta-splatting/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22376v1">AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 Accepted to CVPR 2026
    </div>
    <details class="paper-abstract">
      Recent advances in 4D scene reconstruction have significantly improved dynamic modeling across various domains. However, existing approaches remain limited under aerial conditions with single-view capture, wide spatial range, and dynamic objects of limited spatial footprint and large motion disparity. These challenges cause severe depth ambiguity and unstable motion estimation, making monocular aerial reconstruction inherently ill-posed. To this end, we present AeroDGS, a physics-guided 4D Gaussian splatting framework for monocular UAV videos. AeroDGS introduces a Monocular Geometry Lifting module that reconstructs reliable static and dynamic geometry from a single aerial sequence, providing a robust basis for dynamic estimation. To further resolve monocular ambiguity, we propose a Physics-Guided Optimization module that incorporates differentiable ground-support, upright-stability, and trajectory-smoothness priors, transforming ambiguous image cues into physically consistent motion. The framework jointly refines static backgrounds and dynamic entities with stable geometry and coherent temporal evolution. We additionally build a real-world UAV dataset that spans various altitudes and motion conditions to evaluate dynamic aerial reconstruction. Experiments on synthetic and real UAV scenes demonstrate that AeroDGS outperforms state-of-the-art methods, achieving superior reconstruction fidelity in dynamic aerial environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21874v1">Interactive Augmented Reality-enabled Outdoor Scene Visualization For Enhanced Real-time Disaster Response</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 6 pages, 2 figures
    </div>
    <details class="paper-abstract">
      A user-centered AR interface for disaster response is presented in this work that uses 3D Gaussian Splatting (3DGS) to visualize detailed scene reconstructions, while maintaining situational awareness and keeping cognitive load low. The interface relies on a lightweight interaction approach, combining World-in-Miniature (WIM) navigation with semantic Points of Interest (POIs) that can be filtered as needed, and it is supported by an architecture designed to stream updates as reconstructions evolve. User feedback from a preliminary evaluation indicates that this design is easy to use and supports real-time coordination, with participants highlighting the value of interaction and POIs for fast decision-making in context. Thorough user-centric performance evaluation demonstrates strong usability of the developed interface and high acceptance ratios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06830v2">MUGSQA: Novel Multi-Uncertainty-Based Gaussian Splatting Quality Assessment Method, Dataset, and Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 ICASSP 2026
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has recently emerged as a promising technique for 3D object reconstruction, delivering high-quality rendering results with significantly improved reconstruction speed. As variants continue to appear, assessing the perceptual quality of 3D objects reconstructed with different GS-based methods remains an open challenge. To address this issue, we first propose a unified multi-distance subjective quality assessment method that closely mimics human viewing behavior for objects reconstructed with GS-based methods in actual applications, thereby better collecting perceptual experiences. Based on it, we also construct a novel GS quality assessment dataset named MUGSQA, which is constructed considering multiple uncertainties of the input data. These uncertainties include the quantity and resolution of input views, the view distance, and the accuracy of the initial point cloud. Moreover, we construct two benchmarks: one to evaluate the robustness of various GS-based reconstruction methods under multiple uncertainties, and the other to evaluate the performance of existing quality assessment metrics. Our dataset and benchmark code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21668v1">Space-Time Forecasting of Dynamic Scenes with Motion-aware Gaussian Grouping</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 20 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Forecasting dynamic scenes remains a fundamental challenge in computer vision, as limited observations make it difficult to capture coherent object-level motion and long-term temporal evolution. We present Motion Group-aware Gaussian Forecasting (MoGaF), a framework for long-term scene extrapolation built upon the 4D Gaussian Splatting representation. MoGaF introduces motion-aware Gaussian grouping and group-wise optimization to enforce physically consistent motion across both rigid and non-rigid regions, yielding spatially coherent dynamic representations. Leveraging this structured space-time representation, a lightweight forecasting module predicts future motion, enabling realistic and temporally stable scene evolution. Experiments on synthetic and real-world datasets demonstrate that MoGaF consistently outperforms existing baselines in rendering quality, motion plausibility, and long-term forecasting stability. Our project page is available at https://slime0519.github.io/mogaf
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21644v1">DAGS-SLAM: Dynamic-Aware 3DGS SLAM via Spatiotemporal Motion Probability and Uncertainty-Aware Scheduling</a></div>
    <div class="paper-meta">
      📅 2026-02-25
    </div>
    <details class="paper-abstract">
      Mobile robots and IoT devices demand real-time localization and dense reconstruction under tight compute and energy budgets. While 3D Gaussian Splatting (3DGS) enables efficient dense SLAM, dynamic objects and occlusions still degrade tracking and mapping. Existing dynamic 3DGS-SLAM often relies on heavy optical flow and per-frame segmentation, which is costly for mobile deployment and brittle under challenging illumination. We present DAGS-SLAM, a dynamic-aware 3DGS-SLAM system that maintains a spatiotemporal motion probability (MP) state per Gaussian and triggers semantics on demand via an uncertainty-aware scheduler. DAGS-SLAM fuses lightweight YOLO instance priors with geometric cues to estimate and temporally update MP, propagates MP to the front-end for dynamic-aware correspondence selection, and suppresses dynamic artifacts in the back-end via MP-guided optimization. Experiments on public dynamic RGB-D benchmarks show improved reconstruction and robust tracking while sustaining real-time throughput on a commodity GPU, demonstrating a practical speed-accuracy tradeoff with reduced semantic invocations toward mobile deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21333v1">HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 Accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      Controllable driving scene generation is critical for realistic and scalable autonomous driving simulation, yet existing approaches struggle to jointly achieve photorealism and precise control. We introduce HorizonForge, a unified framework that reconstructs scenes as editable Gaussian Splats and Meshes, enabling fine-grained 3D manipulation and language-driven vehicle insertion. Edits are rendered through a noise-aware video diffusion process that enforces spatial and temporal consistency, producing diverse scene variations in a single feed-forward pass without per-trajectory optimization. To standardize evaluation, we further propose HorizonSuite, a comprehensive benchmark spanning ego- and agent-level editing tasks such as trajectory modifications and object manipulation. Extensive experiments show that Gaussian-Mesh representation delivers substantially higher fidelity than alternative 3D representations, and that temporal priors from video diffusion are essential for coherent synthesis. Combining these findings, HorizonForge establishes a simple yet powerful paradigm for photorealistic, controllable driving simulation, achieving an 83.4% user-preference gain and a 25.19% FID improvement over the second best state-of-the-art method. Project page: https://horizonforge.github.io/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21105v1">BrepGaussian: CAD reconstruction from Multi-View Images with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 Accepted to CVPR 2026
    </div>
    <details class="paper-abstract">
      The boundary representation (B-rep) models a 3D solid as its explicit boundaries: trimmed corners, edges, and faces. Recovering B-rep representation from unstructured data is a challenging and valuable task of computer vision and graphics. Recent advances in deep learning have greatly improved the recovery of 3D shape geometry, but still depend on dense and clean point clouds and struggle to generalize to novel shapes. We propose B-rep Gaussian Splatting (BrepGaussian), a novel framework that learns 3D parametric representations from 2D images. We employ a Gaussian Splatting renderer with learnable features, followed by a specific fitting strategy. To disentangle geometry reconstruction and feature learning, we introduce a two-stage learning framework that first captures geometry and edges and then refines patch features to achieve clean geometry and coherent instance representations. Extensive experiments demonstrate the superior performance of our approach to state-of-the-art methods. We will release our code and datasets upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20933v1">Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 Accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      Recent 3D Gaussian Splatting (3DGS) Dropout methods address overfitting under sparse-view conditions by randomly nullifying Gaussian opacities. However, we identify a neighbor compensation effect in these approaches: dropped Gaussians are often compensated by their neighbors, weakening the intended regularization. Moreover, these methods overlook the contribution of high-degree spherical harmonic coefficients (SH) to overfitting. To address these issues, we propose DropAnSH-GS, a novel anchor-based Dropout strategy. Rather than dropping Gaussians independently, our method randomly selects certain Gaussians as anchors and simultaneously removes their spatial neighbors. This effectively disrupts local redundancies near anchors and encourages the model to learn more robust, globally informed representations. Furthermore, we extend the Dropout to color attributes by randomly dropping higher-degree SH to concentrate appearance information in lower-degree SH. This strategy further mitigates overfitting and enables flexible post-training model compression via SH truncation. Experimental results demonstrate that DropAnSH-GS substantially outperforms existing Dropout methods with negligible computational overhead, and can be readily integrated into various 3DGS variants to enhance their performances. Project Website: https://sk-fun.fun/DropAnSH-GS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20807v1">RU4D-SLAM: Reweighting Uncertainty in Gaussian Splatting SLAM for 4D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Combining 3D Gaussian splatting with Simultaneous Localization and Mapping (SLAM) has gained popularity as it enables continuous 3D environment reconstruction during motion. However, existing methods struggle in dynamic environments, particularly moving objects complicate 3D reconstruction and, in turn, hinder reliable tracking. The emergence of 4D reconstruction, especially 4D Gaussian splatting, offers a promising direction for addressing these challenges, yet its potential for 4D-aware SLAM remains largely underexplored. Along this direction, we propose a robust and efficient framework, namely Reweighting Uncertainty in Gaussian Splatting SLAM (RU4D-SLAM) for 4D scene reconstruction, that introduces temporal factors into spatial 3D representation while incorporating uncertainty-aware perception of scene changes, blurred image synthesis, and dynamic scene reconstruction. We enhance dynamic scene representation by integrating motion blur rendering, and improve uncertainty-aware tracking by extending per-pixel uncertainty modeling, which is originally designed for static scenarios, to handle blurred images. Furthermore, we propose a semantic-guided reweighting mechanism for per-pixel uncertainty estimation in dynamic scenes, and introduce a learnable opacity weight to support adaptive 4D mapping. Extensive experiments on standard benchmarks demonstrate that our method substantially outperforms state-of-the-art approaches in both trajectory accuracy and 4D scene reconstruction, particularly in dynamic environments with moving objects and low-quality inputs. Code available: https://ru4d-slam.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20718v1">Monocular Endoscopic Tissue 3D Reconstruction with Multi-Level Geometry Regularization</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 ijcnn 2025
    </div>
    <details class="paper-abstract">
      Reconstructing deformable endoscopic tissues is crucial for achieving robot-assisted surgery. However, 3D Gaussian Splatting-based approaches encounter challenges in achieving consistent tissue surface reconstruction, while existing NeRF-based methods lack real-time rendering capabilities. In pursuit of both smooth deformable surfaces and real-time rendering, we introduce a novel approach based on 3D Gaussian Splatting. Specifically, we introduce surface-aware reconstruction, initially employing a Sign Distance Field-based method to construct a mesh, subsequently utilizing this mesh to constrain the Gaussian Splatting reconstruction process. Furthermore, to ensure the generation of physically plausible deformations, we incorporate local rigidity and global non-rigidity restrictions to guide Gaussian deformation, tailored for the highly deformable nature of soft endoscopic tissue. Based on 3D Gaussian Splatting, our proposed method delivers a fast rendering process and smooth surface appearances. Quantitative and qualitative analysis against alternative methodologies shows that our approach achieves solid reconstruction quality in both textures and geometries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16030v2">CuriGS: Curriculum-Guided Gaussian Splatting for Sparse View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as an efficient, high-fidelity representation for real-time scene reconstruction and rendering. However, extending 3DGS to sparse-view settings remains challenging because of supervision scarcity and overfitting caused by limited viewpoint coverage. In this paper, we present CuriGS, a curriculum-guided framework for sparse-view 3D reconstruction using 3DGS. CuriGS addresses the core challenge of sparse-view synthesis by introducing student views: pseudo-views sampled around ground-truth poses (teacher). For each teacher, we generate multiple groups of student views with different perturbation levels. During training, we follow a curriculum schedule that gradually unlocks higher perturbation level, randomly sampling candidate students from the active level to assist training. Each sampled student is regularized via depth-correlation and co-regularization, and evaluated using a multi-signal metric that combines SSIM, LPIPS, and an image-quality measure. For every teacher and perturbation level, we periodically retain the best-performing students and promote those that satisfy a predefined quality threshold to the training set, resulting in a stable augmentation of sparse training views. Experimental results show that CuriGS outperforms state-of-the-art baselines in both rendering fidelity and geometric consistency across various synthetic and real sparse-view scenes. Project page: https://zijian1026.github.io/CuriGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.14856v2">Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 10 pages, 4 figures in the main text. Published at ICLR 2026
    </div>
    <details class="paper-abstract">
      Some perspectives naturally provide more information than others. How can an AI system determine which viewpoint offers the most valuable insight for accurate and efficient 3D object reconstruction? Active view selection (AVS) for 3D reconstruction remains a fundamental challenge in computer vision. The aim is to identify the minimal set of views that yields the most accurate 3D reconstruction. Instead of learning radiance fields, like NeRF or 3D Gaussian Splatting, from a current observation and computing uncertainty for each candidate viewpoint, we introduce a novel AVS approach guided by neural uncertainty maps predicted by a lightweight feedforward deep neural network, named UPNet. UPNet takes a single input image of a 3D object and outputs a predicted uncertainty map, representing uncertainty values across all possible candidate viewpoints. By leveraging heuristics derived from observing many natural objects and their associated uncertainty patterns, we train UPNet to learn a direct mapping from viewpoint appearance to uncertainty in the underlying volumetric representations. Next, our approach aggregates all previously predicted neural uncertainty maps to suppress redundant candidate viewpoints and effectively select the most informative one. Using these selected viewpoints, we train 3D neural rendering models and evaluate the quality of novel view synthesis against other competitive AVS methods. Remarkably, despite using half of the viewpoints than the upper bound, our method achieves comparable reconstruction accuracy. In addition, it significantly reduces computational overhead during AVS, achieving up to a 400 times speedup along with over 50\% reductions in CPU, RAM, and GPU usage compared to baseline methods. Notably, our approach generalizes effectively to AVS tasks involving novel object categories, without requiring any additional training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20556v1">WildGHand: Learning Anti-Perturbation Gaussian Hand Avatars from Monocular In-the-Wild Videos</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Despite recent progress in 3D hand reconstruction from monocular videos, most existing methods rely on data captured in well-controlled environments and therefore degrade in real-world settings with severe perturbations, such as hand-object interactions, extreme poses, illumination changes, and motion blur. To tackle these issues, we introduce WildGHand, an optimization-based framework that enables self-adaptive 3D Gaussian splatting on in-the-wild videos and produces high-fidelity hand avatars. WildGHand incorporates two key components: (i) a dynamic perturbation disentanglement module that explicitly represents perturbations as time-varying biases on 3D Gaussian attributes during optimization, and (ii) a perturbation-aware optimization strategy that generates per-frame anisotropic weighted masks to guide optimization. Together, these components allow the framework to identify and suppress perturbations across both spatial and temporal dimensions. We further curate a dataset of monocular hand videos captured under diverse perturbations to benchmark in-the-wild hand avatar reconstruction. Extensive experiments on this dataset and two public datasets demonstrate that WildGHand achieves state-of-the-art performance and substantially improves over its base model across multiple metrics (e.g., up to a $15.8\%$ relative gain in PSNR and a $23.1\%$ relative reduction in LPIPS). Our implementation and dataset are available at https://github.com/XuanHuang0/WildGHand.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12370v3">Changes in Real Time: Online Scene Change Detection with Multi-View Fusion</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 Accepted at CVPR 2026. Project Page: https://chumsy0725.github.io/O-SCD/
    </div>
    <details class="paper-abstract">
      Online Scene Change Detection (SCD) is an extremely challenging problem that requires an agent to detect relevant changes on the fly while observing the scene from unconstrained viewpoints. Existing online SCD methods are significantly less accurate than offline approaches. We present the first online SCD approach that is pose-agnostic, label-free, and ensures multi-view consistency, while operating at over 10 FPS and achieving new state-of-the-art performance, surpassing even the best offline approaches. Our method introduces a new self-supervised fusion loss to infer scene changes from multiple cues and observations, PnP-based fast pose estimation against the reference scene, and a fast change-guided update strategy for the 3D Gaussian Splatting scene representation. Extensive experiments on complex real-world datasets demonstrate that our approach outperforms both online and offline baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00145v1">M-Gaussian: An Magnetic Gaussian Framework for Efficient Multi-Stack MRI Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 15 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Magnetic Resonance Imaging (MRI) is a crucial non-invasive imaging modality. In routine clinical practice, multi-stack thick-slice acquisitions are widely used to reduce scan time and motion sensitivity, particularly in challenging scenarios such as fetal brain imaging. However, the resulting severe through-plane anisotropy compromises volumetric analysis and downstream quantitative assessment, necessitating robust reconstruction of isotropic high-resolution volumes. Implicit neural representation methods, while achieving high quality, suffer from computational inefficiency due to complex network structures. We present M-Gaussian, adapting 3D Gaussian Splatting to MRI reconstruction. Our contributions include: (1) Magnetic Gaussian primitives with physics-consistent volumetric rendering, (2) neural residual field for high-frequency detail refinement, and (3) multi-resolution progressive training. Our method achieves an optimal balance between quality and speed. On the FeTA dataset, M-Gaussian achieves 40.31 dB PSNR while being 14 times faster, representing the first successful adaptation of 3D Gaussian Splatting to multi-stack MRI reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20363v1">Aesthetic Camera Viewpoint Suggestion with 3D Aesthetic Field</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 14 pages, 10 figures
    </div>
    <details class="paper-abstract">
      The aesthetic quality of a scene depends strongly on camera viewpoint. Existing approaches for aesthetic viewpoint suggestion are either single-view adjustments, predicting limited camera adjustments from a single image without understanding scene geometry, or 3D exploration approaches, which rely on dense captures or prebuilt 3D environments coupled with costly reinforcement learning (RL) searches. In this work, we introduce the notion of 3D aesthetic field that enables geometry-grounded aesthetic reasoning in 3D with sparse captures, allowing efficient viewpoint suggestions in contrast to costly RL searches. We opt to learn this 3D aesthetic field using a feedforward 3D Gaussian Splatting network that distills high-level aesthetic knowledge from a pretrained 2D aesthetic model into 3D space, enabling aesthetic prediction for novel viewpoints from only sparse input views. Building on this field, we propose a two-stage search pipeline that combines coarse viewpoint sampling with gradient-based refinement, efficiently identifying aesthetically appealing viewpoints without dense captures or RL exploration. Extensive experiments show that our method consistently suggests viewpoints with superior framing and composition compared to existing approaches, establishing a new direction toward 3D-aware aesthetic modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20342v1">Large-scale Photorealistic Outdoor 3D Scene Reconstruction from UAV Imagery Using Gaussian Splatting Techniques</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 7 pages, 2 figures
    </div>
    <details class="paper-abstract">
      In this study, we present an end-to-end pipeline capable of converting drone-captured video streams into high-fidelity 3D reconstructions with minimal latency. Unmanned aerial vehicles (UAVs) are extensively used in aerial real-time perception applications. Moreover, recent advances in 3D Gaussian Splatting (3DGS) have demonstrated significant potential for real-time neural rendering. However, their integration into end-to-end UAV-based reconstruction and visualization systems remains underexplored. Our goal is to propose an efficient architecture that combines live video acquisition via RTMP streaming, synchronized sensor fusion, camera pose estimation, and 3DGS optimization, achieving continuous model updates and low-latency deployment within interactive visualization environments that supports immersive augmented and virtual reality (AR/VR) applications. Experimental results demonstrate that the proposed method achieves competitive visual fidelity, while delivering significantly higher rendering performance and substantially reduced end-to-end latency, compared to NeRF-based approaches. Reconstruction quality remains within 4-7\% of high-fidelity offline references, confirming the suitability of the proposed system for real-time, scalable augmented perception from aerial platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20160v1">tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 Accepted by CVPR 2026. Project Page: https://cwchenwang.github.io/tttLRM
    </div>
    <details class="paper-abstract">
      We propose tttLRM, a novel large 3D reconstruction model that leverages a Test-Time Training (TTT) layer to enable long-context, autoregressive 3D reconstruction with linear computational complexity, further scaling the model's capability. Our framework efficiently compresses multiple image observations into the fast weights of the TTT layer, forming an implicit 3D representation in the latent space that can be decoded into various explicit formats, such as Gaussian Splats (GS) for downstream applications. The online learning variant of our model supports progressive 3D reconstruction and refinement from streaming observations. We demonstrate that pretraining on novel view synthesis tasks effectively transfers to explicit 3D modeling, resulting in improved reconstruction quality and faster convergence. Extensive experiments show that our method achieves superior performance in feedforward 3D Gaussian reconstruction compared to state-of-the-art approaches on both objects and scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19916v1">Augmented Radiance Field: A General Framework for Enhanced Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 Accepted to ICLR 2026. Project page: \url{https://xiaoxinyyx.github.io/augs}
    </div>
    <details class="paper-abstract">
      Due to the real-time rendering performance, 3D Gaussian Splatting (3DGS) has emerged as the leading method for radiance field reconstruction. However, its reliance on spherical harmonics for color encoding inherently limits its ability to separate diffuse and specular components, making it challenging to accurately represent complex reflections. To address this, we propose a novel enhanced Gaussian kernel that explicitly models specular effects through view-dependent opacity. Meanwhile, we introduce an error-driven compensation strategy to improve rendering quality in existing 3DGS scenes. Our method begins with 2D Gaussian initialization and then adaptively inserts and optimizes enhanced Gaussian kernels, ultimately producing an augmented radiance field. Experiments demonstrate that our method not only surpasses state-of-the-art NeRF methods in rendering performance but also achieves greater parameter efficiency. Project page at: https://xiaoxinyyx.github.io/augs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.06685v4">MOGS: Monocular Object-guided Gaussian Splatting in Large Scenes</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) deliver striking photorealism, and extending it to large scenes opens new opportunities for semantic reasoning and prediction in applications such as autonomous driving. Today's state-of-the-art systems for large scenes primarily originate from LiDAR-based pipelines that utilize long-range depth sensing. However, they require costly high-channel sensors whose dense point clouds strain memory and computation, limiting scalability, fleet deployment, and optimization speed. We present MOGS, a monocular 3DGS framework that replaces active LiDAR depth with object-anchored, metrized dense depth derived from sparse visual-inertial (VI) structure-from-motion (SfM) cues. Our key idea is to exploit image semantics to hypothesize per-object shape priors, anchor them with sparse but metrically reliable SfM points, and propagate the resulting metric constraints across each object to produce dense depth. To address two key challenges, i.e., insufficient SfM coverage within objects and cross-object geometric inconsistency, MOGS introduces (1) a multi-scale shape consensus module that adaptively merges small segments into coarse objects best supported by SfM and fits them with parametric shape models, and (2) a cross-object depth refinement module that optimizes per-pixel depth under a combinatorial objective combining geometric consistency, prior anchoring, and edge-aware smoothness. Experiments on public datasets show that, with a low-cost VI sensor suite, MOGS reduces training time by up to 30.4% and memory consumption by 19.8%, while achieving high-quality rendering competitive with costly LiDAR-based approaches in large scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19766v1">One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Generating explorable 3D scenes from a single image is a highly challenging problem in 3D vision. Existing methods struggle to support free exploration, often producing severe geometric distortions and noisy artifacts when the viewpoint moves far from the original perspective. We introduce \textbf{One2Scene}, an effective framework that decomposes this ill-posed problem into three tractable sub-tasks to enable immersive explorable scene generation. We first use a panorama generator to produce anchor views from a single input image as initialization. Then, we lift these 2D anchors into an explicit 3D geometric scaffold via a generalizable, feed-forward Gaussian Splatting network. Instead of treating the panorama as a single image for reconstruction, we project it into multiple sparse anchor views and reformulate the reconstruction task as multi-view stereo matching, which allows us to leverage robust geometric priors learned from large-scale multi-view datasets. A bidirectional feature fusion module is used to enforce cross-view consistency, yielding an efficient and geometrically reliable scaffold. Finally, the scaffold serves as a strong prior for a novel view generator to produce photorealistic and geometrically accurate views at arbitrary cameras. By explicitly conditioning on a 3D-consistent scaffold to perform reconstruction, One2Scene works stably under large camera motions, supporting immersive scene exploration. Extensive experiments show that One2Scene substantially outperforms state-of-the-art methods in panorama depth estimation, feed-forward 360° reconstruction, and explorable 3D scene generation. Code and models will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19753v1">RAP: Fast Feedforward Rendering-Free Attribute-Guided Primitive Importance Score Prediction for Efficient 3D Gaussian Splatting Processing</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 Accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a leading technology for high-quality 3D scene reconstruction. However, the iterative refinement and densification process leads to the generation of a large number of primitives, each contributing to the reconstruction to a substantially different extent. Estimating primitive importance is thus crucial, both for removing redundancy during reconstruction and for enabling efficient compression and transmission. Existing methods typically rely on rendering-based analyses, where each primitive is evaluated through its contribution across multiple camera viewpoints. However, such methods are sensitive to the number and selection of views, rely on specialized differentiable rasterizers, and have long calculation times that grow linearly with view count, making them difficult to integrate as plug-and-play modules and limiting scalability and generalization. To address these issues, we propose RAP, a fast feedforward rendering-free attribute-guided method for efficient importance score prediction in 3DGS. RAP infers primitive significance directly from intrinsic Gaussian attributes and local neighborhood statistics, avoiding rendering-based or visibility-dependent computations. A compact MLP predicts per-primitive importance scores using rendering loss, pruning-aware loss, and significance distribution regularization. After training on a small set of scenes, RAP generalizes effectively to unseen data and can be seamlessly integrated into reconstruction, compression, and transmission pipelines. Our code is publicly available at https://github.com/yyyykf/RAP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17354v3">PocketGS: On-Device Training of 3D Gaussian Splatting for High Perceptual Modeling</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      Efficient and high-fidelity 3D scene modeling is a long-standing pursuit in computer graphics. While recent 3D Gaussian Splatting (3DGS) methods achieve impressive real-time modeling performance, they rely on resource-unconstrained training assumptions that fail on mobile devices, which are limited by minute-scale training budgets and hardware-available peak-memory. We present PocketGS, a mobile scene modeling paradigm that enables on-device 3DGS training under these tightly coupled constraints while preserving high perceptual fidelity. Our method resolves the fundamental contradictions of standard 3DGS through three co-designed operators: G builds geometry-faithful point-cloud priors; I injects local surface statistics to seed anisotropic Gaussians, thereby reducing early conditioning gaps; and T unrolls alpha compositing with cached intermediates and index-mapped gradient scattering for stable mobile backpropagation. Collectively, these operators satisfy the competing requirements of training efficiency, memory compactness, and modeling fidelity. Extensive experiments demonstrate that PocketGS is able to outperform the powerful mainstream workstation 3DGS baseline to deliver high-quality reconstructions, enabling a fully on-device, practical capture-to-rendering workflow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19323v1">DefenseSplat: Enhancing the Robustness of 3D Gaussian Splatting via Frequency-Aware Filtering</a></div>
    <div class="paper-meta">
      📅 2026-02-22
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful paradigm for real-time and high-fidelity 3D reconstruction from posed images. However, recent studies reveal its vulnerability to adversarial corruptions in input views, where imperceptible yet consistent perturbations can drastically degrade rendering quality, increase training and rendering time, and inflate memory usage, even leading to server denial-of-service. In our work, to mitigate this issue, we begin by analyzing the distinct behaviors of adversarial perturbations in the low- and high-frequency components of input images using wavelet transforms. Based on this observation, we design a simple yet effective frequency-aware defense strategy that reconstructs training views by filtering high-frequency noise while preserving low-frequency content. This approach effectively suppresses adversarial artifacts while maintaining the authenticity of the original scene. Notably, it does not significantly impair training on clean data, achieving a desirable trade-off between robustness and performance on clean inputs. Through extensive experiments under a wide range of attack intensities on multiple benchmarks, we demonstrate that our method substantially enhances the robustness of 3DGS without access to clean ground-truth supervision. By highlighting and addressing the overlooked vulnerabilities of 3D Gaussian Splatting, our work paves the way for more robust and secure 3D reconstructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.18322v1">Unifying Color and Lightness Correction with View-Adaptive Curve Adjustment for Robust 3D Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-02-20
      | 💬 Journal extension version of CVPR 2025 paper: arXiv:2504.01503
    </div>
    <details class="paper-abstract">
      High-quality image acquisition in real-world environments remains challenging due to complex illumination variations and inherent limitations of camera imaging pipelines. These issues are exacerbated in multi-view capture, where differences in lighting, sensor responses, and image signal processor (ISP) configurations introduce photometric and chromatic inconsistencies that violate the assumptions of photometric consistency underlying modern 3D novel view synthesis (NVS) methods, including Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), leading to degraded reconstruction and rendering quality. We propose Luminance-GS++, a 3DGS-based framework for robust NVS under diverse illumination conditions. Our method combines a globally view-adaptive lightness adjustment with a local pixel-wise residual refinement for precise color correction. We further design unsupervised objectives that jointly enforce lightness correction and multi-view geometric and photometric consistency. Extensive experiments demonstrate state-of-the-art performance across challenging scenarios, including low-light, overexposure, and complex luminance and chromatic variations. Unlike prior approaches that modify the underlying representation, our method preserves the explicit 3DGS formulation, improving reconstruction fidelity while maintaining real-time rendering efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.18314v1">Diff2DGS: Reliable Reconstruction of Occluded Surgical Scenes via 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-20
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Real-time reconstruction of deformable surgical scenes is vital for advancing robotic surgery, improving surgeon guidance, and enabling automation. Recent methods achieve dense reconstructions from da Vinci robotic surgery videos, with Gaussian Splatting (GS) offering real-time performance via graphics acceleration. However, reconstruction quality in occluded regions remains limited, and depth accuracy has not been fully assessed, as benchmarks like EndoNeRF and StereoMIS lack 3D ground truth. We propose Diff2DGS, a novel two-stage framework for reliable 3D reconstruction of occluded surgical scenes. In the first stage, a diffusion-based video module with temporal priors inpaints tissue occluded by instruments with high spatial-temporal consistency. In the second stage, we adapt 2D Gaussian Splatting (2DGS) with a Learnable Deformation Model (LDM) to capture dynamic tissue deformation and anatomical geometry. We also extend evaluation beyond prior image-quality metrics by performing quantitative depth accuracy analysis on the SCARED dataset. Diff2DGS outperforms state-of-the-art approaches in both appearance and geometry, reaching 38.02 dB PSNR on EndoNeRF and 34.40 dB on StereoMIS. Furthermore, our experiments demonstrate that optimizing for image quality alone does not necessarily translate into optimal 3D reconstruction accuracy. To address this, we further optimize the depth quality of the reconstructed 3D results, ensuring more faithful geometry in addition to high-fidelity appearance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01844v2">CloDS: Visual-Only Unsupervised Cloth Dynamics Learning in Unknown Conditions</a></div>
    <div class="paper-meta">
      📅 2026-02-20
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Deep learning has demonstrated remarkable capabilities in simulating complex dynamic systems. However, existing methods require known physical properties as supervision or inputs, limiting their applicability under unknown conditions. To explore this challenge, we introduce Cloth Dynamics Grounding (CDG), a novel scenario for unsupervised learning of cloth dynamics from multi-view visual observations. We further propose Cloth Dynamics Splatting (CloDS), an unsupervised dynamic learning framework designed for CDG. CloDS adopts a three-stage pipeline that first performs video-to-geometry grounding and then trains a dynamics model on the grounded meshes. To cope with large non-linear deformations and severe self-occlusions during grounding, we introduce a dual-position opacity modulation that supports bidirectional mapping between 2D observations and 3D geometry via mesh-based Gaussian splatting in video-to-geometry grounding stage. It jointly considers the absolute and relative position of Gaussian components. Comprehensive experimental evaluations demonstrate that CloDS effectively learns cloth dynamics from visual data while maintaining strong generalization capabilities for unseen configurations. Our code is available at https://github.com/whynot-zyl/CloDS. Visualization results are available at https://github.com/whynot-zyl/CloDS_video}.%\footnote{As in this example.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02089v2">UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-20
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) enables high-quality, real-time rendering for bounded scenes, its extension to large-scale urban environments gives rise to critical challenges in terms of geometric consistency, memory efficiency, and computational scalability. To address these issues, we present UrbanGS, a scalable reconstruction framework that effectively tackles these challenges for city-scale applications. First, we propose a Depth-Consistent D-Normal Regularization module. Unlike existing approaches that rely solely on monocular normal estimators, which can effectively update rotation parameters yet struggle to update position parameters, our method integrates D-Normal constraints with external depth supervision. This allows for comprehensive updates of all geometric parameters. By further incorporating an adaptive confidence weighting mechanism based on gradient consistency and inverse depth deviation, our approach significantly enhances multi-view depth alignment and geometric coherence, which effectively resolves the issue of geometric accuracy in complex large-scale scenes. To improve scalability, we introduce a Spatially Adaptive Gaussian Pruning (SAGP) strategy, which dynamically adjusts Gaussian density based on local geometric complexity and visibility to reduce redundancy. Additionally, a unified partitioning and view assignment scheme is designed to eliminate boundary artifacts and optimize computational load. Extensive experiments on multiple urban datasets demonstrate that UrbanGS achieves superior performance in rendering quality, geometric accuracy, and memory efficiency, providing a systematic solution for high-fidelity large-scale scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17473v1">4D Monocular Surgical Reconstruction under Arbitrary Camera Motions</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 Due to the limitation "The abstract field cannot be longer than 1,920 characters", the abstract here is shorter than that in the PDF file Subjects
    </div>
    <details class="paper-abstract">
      Reconstructing deformable surgical scenes from endoscopic videos is challenging and clinically important. Recent state-of-the-art methods based on implicit neural representations or 3D Gaussian splatting have made notable progress. However, most are designed for deformable scenes with fixed endoscope viewpoints and rely on stereo depth priors or accurate structure-from-motion for initialization and optimization, limiting their ability to handle monocular sequences with large camera motion in real clinical settings. To address this, we propose Local-EndoGS, a high-quality 4D reconstruction framework for monocular endoscopic sequences with arbitrary camera motion. Local-EndoGS introduces a progressive, window-based global representation that allocates local deformable scene models to each observed window, enabling scalability to long sequences with substantial motion. To overcome unreliable initialization without stereo depth or accurate structure-from-motion, we design a coarse-to-fine strategy integrating multi-view geometry, cross-window information, and monocular depth priors, providing a robust foundation for optimization. We further incorporate long-range 2D pixel trajectory constraints and physical motion priors to improve deformation plausibility. Experiments on three public endoscopic datasets with deformable scenes and varying camera motions show that Local-EndoGS consistently outperforms state-of-the-art methods in appearance quality and geometry. Ablation studies validate the effectiveness of our key designs. Code will be released upon acceptance at: https://github.com/IRMVLab/Local-EndoGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17182v1">NRGS-SLAM: Monocular Non-Rigid SLAM for Endoscopy via Deformation-Aware 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Visual simultaneous localization and mapping (V-SLAM) is a fundamental capability for autonomous perception and navigation. However, endoscopic scenes violate the rigidity assumption due to persistent soft-tissue deformations, creating a strong coupling ambiguity between camera ego-motion and intrinsic deformation. Although recent monocular non-rigid SLAM methods have made notable progress, they often lack effective decoupling mechanisms and rely on sparse or low-fidelity scene representations, which leads to tracking drift and limited reconstruction quality. To address these limitations, we propose NRGS-SLAM, a monocular non-rigid SLAM system for endoscopy based on 3D Gaussian Splatting. To resolve the coupling ambiguity, we introduce a deformation-aware 3D Gaussian map that augments each Gaussian primitive with a learnable deformation probability, optimized via a Bayesian self-supervision strategy without requiring external non-rigidity labels. Building on this representation, we design a deformable tracking module that performs robust coarse-to-fine pose estimation by prioritizing low-deformation regions, followed by efficient per-frame deformation updates. A carefully designed deformable mapping module progressively expands and refines the map, balancing representational capacity and computational efficiency. In addition, a unified robust geometric loss incorporates external geometric priors to mitigate the inherent ill-posedness of monocular non-rigid SLAM. Extensive experiments on multiple public endoscopic datasets demonstrate that NRGS-SLAM achieves more accurate camera pose estimation (up to 50\% reduction in RMSE) and higher-quality photo-realistic reconstructions than state-of-the-art methods. Comprehensive ablation studies further validate the effectiveness of our key design choices. Source code will be publicly available upon paper acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17134v1">B$^3$-Seg: Camera-Free, Training-Free 3DGS Segmentation via Analytic EIG and Beta-Bernoulli Bayesian Updates</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 Project page: https://sony.github.io/B3-Seg-project/
    </div>
    <details class="paper-abstract">
      Interactive 3D Gaussian Splatting (3DGS) segmentation is essential for real-time editing of pre-reconstructed assets in film and game production. However, existing methods rely on predefined camera viewpoints, ground-truth labels, or costly retraining, making them impractical for low-latency use. We propose B$^3$-Seg (Beta-Bernoulli Bayesian Segmentation for 3DGS), a fast and theoretically grounded method for open-vocabulary 3DGS segmentation under camera-free and training-free conditions. Our approach reformulates segmentation as sequential Beta-Bernoulli Bayesian updates and actively selects the next view via analytic Expected Information Gain (EIG). This Bayesian formulation guarantees the adaptive monotonicity and submodularity of EIG, which produces a greedy $(1{-}1/e)$ approximation to the optimal view sampling policy. Experiments on multiple datasets show that B$^3$-Seg achieves competitive results to high-cost supervised methods while operating end-to-end segmentation within a few seconds. The results demonstrate that B$^3$-Seg enables practical, interactive 3DGS segmentation with provable information efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17124v1">3D Scene Rendering with Multimodal Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      3D scene reconstruction and rendering are core tasks in computer vision, with applications spanning industrial monitoring, robotics, and autonomous driving. Recent advances in 3D Gaussian Splatting (GS) and its variants have achieved impressive rendering fidelity while maintaining high computational and memory efficiency. However, conventional vision-based GS pipelines typically rely on a sufficient number of camera views to initialize the Gaussian primitives and train their parameters, typically incurring additional processing cost during initialization while falling short in conditions where visual cues are unreliable, such as adverse weather, low illumination, or partial occlusions. To cope with these challenges, and motivated by the robustness of radio-frequency (RF) signals to weather, lighting, and occlusions, we introduce a multimodal framework that integrates RF sensing, such as automotive radar, with GS-based rendering as a more efficient and robust alternative to vision-only GS rendering. The proposed approach enables efficient depth prediction from only sparse RF-based depth measurements, yielding a high-quality 3D point cloud for initializing Gaussian functions across diverse GS architectures. Numerical tests demonstrate the merits of judiciously incorporating RF sensing into GS pipelines, achieving high-fidelity 3D scene rendering driven by RF-informed structural accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17117v1">i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Physical simulation predicts future states of objects based on material properties and external loads, enabling blueprints for both Industry and Engineering to conduct risk management. Current 3D reconstruction-based simulators typically rely on explicit, step-wise updates, which are sensitive to step time and suffer from rapid accuracy degradation under complicated scenarios, such as high-stiffness materials or quasi-static movement. To address this, we introduce i-PhysGaussian, a framework that couples 3D Gaussian Splatting (3DGS) with an implicit Material Point Method (MPM) integrator. Unlike explicit methods, our solution obtains an end-of-step state by minimizing a momentum-balance residual through implicit Newton-type optimization with a GMRES solver. This formulation significantly reduces time-step sensitivity and ensures physical consistency. Our results demonstrate that i-PhysGaussian maintains stability at up to 20x larger time steps than explicit baselines, preserving structural coherence and smooth motion even in complex dynamic transitions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2404.19026v2">MeGA: Hybrid Mesh-Gaussian Head Avatar for High-Fidelity Rendering and Head Editing</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 Accepted by CVPR 2025. Project page: https://conallwang.github.io/MeGA_Pages/
    </div>
    <details class="paper-abstract">
      Creating high-fidelity head avatars from multi-view videos is a core issue for many AR/VR applications. However, existing methods usually struggle to obtain high-quality renderings for all different head components simultaneously since they use one single representation to model components with drastically different characteristics (e.g., skin vs. hair). In this paper, we propose a Hybrid Mesh-Gaussian Head Avatar (MeGA) that models different head components with more suitable representations. Specifically, we select an enhanced FLAME mesh as our facial representation and predict a UV displacement map to provide per-vertex offsets for improved personalized geometric details. To achieve photorealistic renderings, we obtain facial colors using deferred neural rendering and disentangle neural textures into three meaningful parts. For hair modeling, we first build a static canonical hair using 3D Gaussian Splatting. A rigid transformation and an MLP-based deformation field are further applied to handle complex dynamic expressions. Combined with our occlusion-aware blending, MeGA generates higher-fidelity renderings for the whole head and naturally supports more downstream tasks. Experiments on the NeRSemble dataset demonstrate the effectiveness of our designs, outperforming previous state-of-the-art methods and supporting various editing functionalities, including hairstyle alteration and texture editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24053v2">3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 Published at ICLR 2026. Project page and codes available at https://zixunh.github.io/3d-geer
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves an appealing balance between rendering quality and efficiency, but relies on approximating 3D Gaussians as 2D projections--an assumption that degrades accuracy, especially under generic large field-of-view (FoV) cameras. Despite recent extensions, no prior work has simultaneously achieved both projective exactness and real-time efficiency for general cameras. We introduce 3DGEER, a geometrically exact and efficient Gaussian rendering framework. From first principles, we derive a closed-form expression for integrating Gaussian density along a ray, enabling precise forward rendering and differentiable optimization under arbitrary camera models. To retain efficiency, we propose the Particle Bounding Frustum (PBF), which provides tight ray-Gaussian association without BVH traversal, and the Bipolar Equiangular Projection (BEAP), which unifies FoV representations, accelerates association, and improves reconstruction quality. Experiments on both pinhole and fisheye datasets show that 3DGEER outperforms prior methods across all metrics, runs 5x faster than existing projective exact ray-based baselines, and generalizes to wider FoVs unseen during training--establishing a new state of the art in real-time radiance field rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.12768v2">Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 Project page: https://tamu-visual-ai.github.io/usplat4d/
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic 3D scenes from monocular input is fundamentally under-constrained, with ambiguities arising from occlusion and extreme novel views. While dynamic Gaussian Splatting offers an efficient representation, vanilla models optimize all Gaussian primitives uniformly, ignoring whether they are well or poorly observed. This limitation leads to motion drifts under occlusion and degraded synthesis when extrapolating to unseen views. We argue that uncertainty matters: Gaussians with recurring observations across views and time act as reliable anchors to guide motion, whereas those with limited visibility are treated as less reliable. To this end, we introduce USplat4D, a novel Uncertainty-aware dynamic Gaussian Splatting framework that propagates reliable motion cues to enhance 4D reconstruction. Our approach estimates time-varying per-Gaussian uncertainty and leverages it to construct a spatio-temporal graph for uncertainty-aware optimization. Experiments on diverse real and synthetic datasets show that explicitly modeling uncertainty consistently improves dynamic Gaussian Splatting models, yielding more stable geometry under occlusion and high-quality synthesis at extreme viewpoints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07101v2">Zero-Shot UAV Navigation in Forests via Relightable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 12 pages, 8 figures
    </div>
    <details class="paper-abstract">
      UAV navigation in unstructured outdoor environments using passive monocular vision is hindered by the substantial visual domain gap between simulation and reality. While 3D Gaussian Splatting enables photorealistic scene reconstruction from real-world data, existing methods inherently couple static lighting with geometry, severely limiting policy generalization to dynamic real-world illumination. In this paper, we propose a novel end-to-end reinforcement learning framework designed for effective zero-shot transfer to unstructured outdoors. Within a high-fidelity simulation grounded in real-world data, our policy is trained to map raw monocular RGB observations directly to continuous control commands. To overcome photometric limitations, we introduce Relightable 3D Gaussian Splatting, which decomposes scene components to enable explicit, physically grounded editing of environmental lighting within the neural representation. By augmenting training with diverse synthesized lighting conditions ranging from strong directional sunlight to diffuse overcast skies, we compel the policy to learn robust, illumination-invariant visual features. Extensive real-world experiments demonstrate that a lightweight quadrotor achieves robust, collision-free navigation in complex forest environments at speeds up to 10 m/s, exhibiting significant resilience to drastic lighting variations without fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.01110v4">A LoD of Gaussians: Unified Training and Rendering for Ultra-Large Scale Reconstruction with External Memory</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has emerged as a high-performance technique for novel view synthesis, enabling real-time rendering and high-quality reconstruction of small scenes. However, scaling to larger environments has so far relied on partitioning the scene into chunks -- a strategy that introduces artifacts at chunk boundaries, complicates training across varying scales, and is poorly suited to unstructured scenarios such as city-scale flyovers combined with street-level views. Moreover, rendering remains fundamentally limited by GPU memory, as all visible chunks must reside in VRAM simultaneously. We introduce A LoD of Gaussians, a framework for training and rendering ultra-large-scale Gaussian scenes on a single consumer-grade GPU -- without partitioning. Our method stores the full scene out-of-core (e.g., in CPU memory) and trains a Level-of-Detail (LoD) representation directly, dynamically streaming only the relevant Gaussians. A hybrid data structure combining Gaussian hierarchies with Sequential Point Trees enables efficient, view-dependent LoD selection, while a lightweight caching and view scheduling system exploits temporal coherence to support real-time streaming and rendering. Together, these innovations enable seamless multi-scale reconstruction and interactive visualization of complex scenes -- from broad aerial views to fine-grained ground-level details.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15516v1">Semantic-Guided 3D Gaussian Splatting for Transient Object Removal</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      Transient objects in casual multi-view captures cause ghosting artifacts in 3D Gaussian Splatting (3DGS) reconstruction. Existing solutions relied on scene decomposition at significant memory cost or on motion-based heuristics that were vulnerable to parallax ambiguity. A semantic filtering framework was proposed for category-aware transient removal using vision-language models. CLIP similarity scores between rendered views and distractor text prompts were accumulated per-Gaussian across training iterations. Gaussians exceeding a calibrated threshold underwent opacity regularization and periodic pruning. Unlike motion-based approaches, semantic classification resolved parallax ambiguity by identifying object categories independently of motion patterns. Experiments on the RobustNeRF benchmark demonstrated consistent improvement in reconstruction quality over vanilla 3DGS across four sequences, while maintaining minimal memory overhead and real-time rendering performance. Threshold calibration and comparisons with baselines validated semantic guidance as a practical strategy for transient removal in scenarios with predictable distractor categories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.11762v3">GS-ProCams: Gaussian Splatting-based Projector-Camera Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 This version includes updated experimental results after an implementation fix
    </div>
    <details class="paper-abstract">
      We present GS-ProCams, the first Gaussian Splatting-based framework for projector-camera systems (ProCams). GS-ProCams is not only view-agnostic but also significantly enhances the efficiency of projection mapping (PM) that requires establishing geometric and radiometric mappings between the projector and the camera. Previous CNN-based ProCams are constrained to a specific viewpoint, limiting their applicability to novel perspectives. In contrast, NeRF-based ProCams support view-agnostic projection mapping, however, they require an additional co-located light source and demand significant computational and memory resources. To address this issue, we propose GS-ProCams that employs 2D Gaussian for scene representations, and enables efficient view-agnostic ProCams applications. In particular, we explicitly model the complex geometric and photometric mappings of ProCams using projector responses, the projection surface's geometry and materials represented by Gaussians, and the global illumination component. Then, we employ differentiable physically-based rendering to jointly estimate them from captured multi-view projections. Compared to state-of-the-art NeRF-based methods, our GS-ProCams eliminates the need for additional devices, achieving superior ProCams simulation quality. It also uses only 1/10 of the GPU memory for training and is 900 times faster in inference speed. Please refer to our project page for the code and dataset: https://realqingyue.github.io/GS-ProCams/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.12369v3">DARB-Splatting: Generalizing Splatting with Decaying Anisotropic Radial Basis Functions</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 Link to the project page: https://github.com/viruthshaan/darb-splatting/
    </div>
    <details class="paper-abstract">
      Splatting-based 3D reconstruction methods have gained popularity with the advent of 3D Gaussian Splatting, efficiently synthesizing high-quality novel views. These methods commonly resort to using exponential family functions, such as the Gaussian function, as reconstruction kernels due to their anisotropic nature, ease of projection, and differentiability in rasterization. However, the field remains restricted to variations within the exponential family, leaving generalized reconstruction kernels largely underexplored, partly due to the lack of easy integrability in 3D to 2D projections. In this light, we show that a class of decaying anisotropic radial basis functions (DARBFs), which are non-negative functions of the Mahalanobis distance, supports splatting by approximating the Gaussian function's closed-form integration advantage. With this fresh perspective, we demonstrate varying performances across selected DARB reconstruction kernels, achieving comparable training convergence and memory footprints, with on-par PSNR, SSIM, and LPIPS results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15355v1">DAV-GSWT: Diffusion-Active-View Sampling for Data-Efficient Gaussian Splatting Wang Tiles</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 16 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The emergence of 3D Gaussian Splatting has fundamentally redefined the capabilities of photorealistic neural rendering by enabling high-throughput synthesis of complex environments. While procedural methods like Wang Tiles have recently been integrated to facilitate the generation of expansive landscapes, these systems typically remain constrained by a reliance on densely sampled exemplar reconstructions. We present DAV-GSWT, a data-efficient framework that leverages diffusion priors and active view sampling to synthesize high-fidelity Gaussian Splatting Wang Tiles from minimal input observations. By integrating a hierarchical uncertainty quantification mechanism with generative diffusion models, our approach autonomously identifies the most informative viewpoints while hallucinating missing structural details to ensure seamless tile transitions. Experimental results indicate that our system significantly reduces the required data volume while maintaining the visual integrity and interactive performance necessary for large-scale virtual environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.13159v2">Digital Twin Generation from Visual Data: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      This survey examines recent advances in generating digital twins from visual data. These digital twins - virtual 3D replicas of physical assets - can be applied to robotics, media content creation, design or construction workflows. We analyze a range of approaches, including 3D Gaussian Splatting, generative inpainting, semantic segmentation, and foundation models, highlighting their respective advantages and limitations. In addition, we discuss key challenges such as occlusions, lighting variations, and scalability, as well as identify gaps, trends, and directions for future research. Overall, this survey aims to provide a comprehensive overview of state-of-the-art methodologies and their implications for real-world applications. Awesome Digital Twin: https://awesomedigitaltwin.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15181v1">Time-Archival Camera Virtualization for Sports and Visual Performances</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 Project Page: https://yunxiaozhangjack.com/tacv/; Under minor revision in Journal of Computer Vision and Image Understanding (CVIU); Special Issue: Computer Vision for Sports and Winter Sports. Outcome of a master and bachelor student project completed in Visual and Spatial AI Lab at TAMU
    </div>
    <details class="paper-abstract">
      Camera virtualization -- an emerging solution to novel view synthesis -- holds transformative potential for visual entertainment, live performances, and sports broadcasting by enabling the generation of photorealistic images from novel viewpoints using images from a limited set of calibrated multiple static physical cameras. Despite recent advances, achieving spatially and temporally coherent and photorealistic rendering of dynamic scenes with efficient time-archival capabilities, particularly in fast-paced sports and stage performances, remains challenging for existing approaches. Recent methods based on 3D Gaussian Splatting (3DGS) for dynamic scenes could offer real-time view-synthesis results. Yet, they are hindered by their dependence on accurate 3D point clouds from the structure-from-motion method and their inability to handle large, non-rigid, rapid motions of different subjects (e.g., flips, jumps, articulations, sudden player-to-player transitions). Moreover, independent motions of multiple subjects can break the Gaussian-tracking assumptions commonly used in 4DGS, ST-GS, and other dynamic splatting variants. This paper advocates reconsidering a neural volume rendering formulation for camera virtualization and efficient time-archival capabilities, making it useful for sports broadcasting and related applications. By modeling a dynamic scene as rigid transformations across multiple synchronized camera views at a given time, our method performs neural representation learning, providing enhanced visual rendering quality at test time. A key contribution of our approach is its support for time-archival, i.e., users can revisit any past temporal instance of a dynamic scene and can perform novel view synthesis, enabling retrospective rendering for replay, analysis, and archival of live events, a functionality absent in existing neural rendering approaches and novel view synthesis...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14929v1">Wrivinder: Towards Spatial Intelligence for Geo-locating Ground Images onto Satellite Imagery</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      Aligning ground-level imagery with geo-registered satellite maps is crucial for mapping, navigation, and situational awareness, yet remains challenging under large viewpoint gaps or when GPS is unreliable. We introduce Wrivinder, a zero-shot, geometry-driven framework that aggregates multiple ground photographs to reconstruct a consistent 3D scene and align it with overhead satellite imagery. Wrivinder combines SfM reconstruction, 3D Gaussian Splatting, semantic grounding, and monocular depth--based metric cues to produce a stable zenith-view rendering that can be directly matched to satellite context for metrically accurate camera geo-localization. To support systematic evaluation of this task, which lacks suitable benchmarks, we also release MC-Sat, a curated dataset linking multi-view ground imagery with geo-registered satellite tiles across diverse outdoor environments. Together, Wrivinder and MC-Sat provide a first comprehensive baseline and testbed for studying geometry-centered cross-view alignment without paired supervision. In zero-shot experiments, Wrivinder achieves sub-30\,m geolocation accuracy across both dense and large-area scenes, highlighting the promise of geometry-based aggregation for robust ground-to-satellite localization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.03407v2">Multi-Spectral Gaussian Splatting with Neural Color Representation</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 for project page, see https://meyerls.github.io/ms_splatting
    </div>
    <details class="paper-abstract">
      We present MS-Splatting -- a multi-spectral 3D Gaussian Splatting (3DGS) framework that is able to generate multi-view consistent novel views from images of multiple, independent cameras with different spectral domains. In contrast to previous approaches, our method does not require cross-modal camera calibration and is versatile enough to model a variety of different spectra, including thermal and near-infra red, without any algorithmic changes. Unlike existing 3DGS-based frameworks that treat each modality separately (by optimizing per-channel spherical harmonics) and therefore fail to exploit the underlying spectral and spatial correlations, our method leverages a novel neural color representation that encodes multi-spectral information into a learned, compact, per-splat feature embedding. A shallow multi-layer perceptron (MLP) then decodes this embedding to obtain spectral color values, enabling joint learning of all bands within a unified representation. Our experiments show that this simple yet effective strategy is able to improve multi-spectral rendering quality, while also leading to improved per-spectra rendering quality over state-of-the-art methods. We demonstrate the effectiveness of this new technique in agricultural applications to render vegetation indices, such as normalized difference vegetation index (NDVI).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14493v1">Gaussian Mesh Renderer for Lightweight Differentiable Rendering</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2026). GitHub: https://github.com/huntorochi/Gaussian-Mesh-Renderer
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has enabled high-fidelity virtualization with fast rendering and optimization for novel view synthesis. On the other hand, triangle mesh models still remain a popular choice for surface reconstruction but suffer from slow or heavy optimization in traditional mesh-based differentiable renderers. To address this problem, we propose a new lightweight differentiable mesh renderer leveraging the efficient rasterization process of 3DGS, named Gaussian Mesh Renderer (GMR), which tightly integrates the Gaussian and mesh representations. Each Gaussian primitive is analytically derived from the corresponding mesh triangle, preserving structural fidelity and enabling the gradient flow. Compared to the traditional mesh renderers, our method achieves smoother gradients, which especially contributes to better optimization using smaller batch sizes with limited memory. Our implementation is available in the public GitHub repository at https://github.com/huntorochi/Gaussian-Mesh-Renderer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14199v1">Learnable Multi-level Discrete Wavelet Transforms for 3D Gaussian Splatting Frequency Modulation</a></div>
    <div class="paper-meta">
      📅 2026-02-15
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful approach for novel view synthesis. However, the number of Gaussian primitives often grows substantially during training as finer scene details are reconstructed, leading to increased memory and storage costs. Recent coarse-to-fine strategies regulate Gaussian growth by modulating the frequency content of the ground-truth images. In particular, AutoOpti3DGS employs the learnable Discrete Wavelet Transform (DWT) to enable data-adaptive frequency modulation. Nevertheless, its modulation depth is limited by the 1-level DWT, and jointly optimizing wavelet regularization with 3D reconstruction introduces gradient competition that promotes excessive Gaussian densification. In this paper, we propose a multi-level DWT-based frequency modulation framework for 3DGS. By recursively decomposing the low-frequency subband, we construct a deeper curriculum that provides progressively coarser supervision during early training, consistently reducing Gaussian counts. Furthermore, we show that the modulation can be performed using only a single scaling parameter, rather than learning the full 2-tap high-pass filter. Experimental results on standard benchmarks demonstrate that our method further reduces Gaussian counts while maintaining competitive rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13909v1">High-fidelity 3D reconstruction for planetary exploration</a></div>
    <div class="paper-meta">
      📅 2026-02-14
      | 💬 7 pages, 3 figures, conference paper
    </div>
    <details class="paper-abstract">
      Planetary exploration increasingly relies on autonomous robotic systems capable of perceiving, interpreting, and reconstructing their surroundings in the absence of global positioning or real-time communication with Earth. Rovers operating on planetary surfaces must navigate under sever environmental constraints, limited visual redundancy, and communication delays, making onboard spatial awareness and visual localization key components for mission success. Traditional techniques based on Structure-from-Motion (SfM) and Simultaneous Localization and Mapping (SLAM) provide geometric consistency but struggle to capture radiometric detail or to scale efficiently in unstructured, low-texture terrains typical of extraterrestrial environments. This work explores the integration of radiance field-based methods - specifically Neural Radiance Fields (NeRF) and Gaussian Splatting - into a unified, automated environment reconstruction pipeline for planetary robotics. Our system combines the Nerfstudio and COLMAP frameworks with a ROS2-compatible workflow capable of processing raw rover data directly from rosbag recordings. This approach enables the generation of dense, photorealistic, and metrically consistent 3D representations from minimal visual input, supporting improved perception and planning for autonomous systems operating in planetary-like conditions. The resulting pipeline established a foundation for future research in radiance field-based mapping, bridging the gap between geometric and neural representations in planetary exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11575v2">ReaDy-Go: Real-to-Sim Dynamic 3D Gaussian Splatting Simulation for Environment-Specific Visual Navigation with Moving Obstacles</a></div>
    <div class="paper-meta">
      📅 2026-02-14
      | 💬 Project page: https://syeon-yoo.github.io/ready-go-site/
    </div>
    <details class="paper-abstract">
      Visual navigation models often struggle in real-world dynamic environments due to limited robustness to the sim-to-real gap and the difficulty of training policies tailored to target deployment environments (e.g., households, restaurants, and factories). Although real-to-sim navigation simulation using 3D Gaussian Splatting (GS) can mitigate these challenges, prior GS-based works have considered only static scenes or non-photorealistic human obstacles built from simulator assets, despite the importance of safe navigation in dynamic environments. To address these issues, we propose ReaDy-Go, a novel real-to-sim simulation pipeline that synthesizes photorealistic dynamic scenarios in target environments by augmenting a reconstructed static GS scene with dynamic human GS obstacles, and trains navigation policies using the generated datasets. The pipeline provides three key contributions: (1) a dynamic GS simulator that integrates static scene GS with a human animation module, enabling the insertion of animatable human GS avatars and the synthesis of plausible human motions from 2D trajectories, (2) a navigation dataset generation framework that leverages the simulator along with a robot expert planner designed for dynamic GS representations and a human planner, and (3) robust navigation policies to both the sim-to-real gap and moving obstacles. The proposed simulator generates thousands of photorealistic navigation scenarios with animatable human GS avatars from arbitrary viewpoints. ReaDy-Go outperforms baselines across target environments in both simulation and real-world experiments, demonstrating improved navigation performance even after sim-to-real transfer and in the presence of moving obstacles. Moreover, zero-shot sim-to-real deployment in an unseen environment indicates its generalization potential. Project page: https://syeon-yoo.github.io/ready-go-site/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13801v1">Joint Orientation and Weight Optimization for Robust Watertight Surface Reconstruction via Dirichlet-Regularized Winding Fields</a></div>
    <div class="paper-meta">
      📅 2026-02-14
    </div>
    <details class="paper-abstract">
      We propose Dirichlet Winding Reconstruction (DiWR), a robust method for reconstructing watertight surfaces from unoriented point clouds with non-uniform sampling, noise, and outliers. Our method uses the generalized winding number (GWN) field as the target implicit representation and jointly optimizes point orientations, per-point area weights, and confidence coefficients in a single pipeline. The optimization minimizes the Dirichlet energy of the induced winding field together with additional GWN-based constraints, allowing DiWR to compensate for non-uniform sampling, reduce the impact of noise, and downweight outliers during reconstruction, with no reliance on separate preprocessing. We evaluate DiWR on point clouds from 3D Gaussian Splatting, a computer-vision pipeline, and corrupted graphics benchmarks. Experiments show that DiWR produces plausible watertight surfaces on these challenging inputs and outperforms both traditional multi-stage pipelines and recent joint orientation-reconstruction methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13549v1">Nighttime Autonomous Driving Scene Reconstruction with Physically-Based Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-14
      | 💬 ICRA 2026
    </div>
    <details class="paper-abstract">
      This paper focuses on scene reconstruction under nighttime conditions in autonomous driving simulation. Recent methods based on Neural Radiance Fields (NeRFs) and 3D Gaussian Splatting (3DGS) have achieved photorealistic modeling in autonomous driving scene reconstruction, but they primarily focus on normal-light conditions. Low-light driving scenes are more challenging to model due to their complex lighting and appearance conditions, which often causes performance degradation of existing methods. To address this problem, this work presents a novel approach that integrates physically based rendering into 3DGS to enhance nighttime scene reconstruction for autonomous driving. Specifically, our approach integrates physically based rendering into composite scene Gaussian representations and jointly optimizes Bidirectional Reflectance Distribution Function (BRDF) based material properties. We explicitly model diffuse components through a global illumination module and specular components by anisotropic spherical Gaussians. As a result, our approach improves reconstruction quality for outdoor nighttime driving scenes, while maintaining real-time rendering. Extensive experiments across diverse nighttime scenarios on two real-world autonomous driving datasets, including nuScenes and Waymo, demonstrate that our approach outperforms the state-of-the-art methods both quantitatively and qualitatively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13444v1">FlowHOI: Flow-based Semantics-Grounded Generation of Hand-Object Interactions for Dexterous Robot Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-02-13
      | 💬 Project Page: https://huajian-zeng.github.io/projects/flowhoi/
    </div>
    <details class="paper-abstract">
      Recent vision-language-action (VLA) models can generate plausible end-effector motions, yet they often fail in long-horizon, contact-rich tasks because the underlying hand-object interaction (HOI) structure is not explicitly represented. An embodiment-agnostic interaction representation that captures this structure would make manipulation behaviors easier to validate and transfer across robots. We propose FlowHOI, a two-stage flow-matching framework that generates semantically grounded, temporally coherent HOI sequences, comprising hand poses, object poses, and hand-object contact states, conditioned on an egocentric observation, a language instruction, and a 3D Gaussian splatting (3DGS) scene reconstruction. We decouple geometry-centric grasping from semantics-centric manipulation, conditioning the latter on compact 3D scene tokens and employing a motion-text alignment loss to semantically ground the generated interactions in both the physical scene layout and the language instruction. To address the scarcity of high-fidelity HOI supervision, we introduce a reconstruction pipeline that recovers aligned hand-object trajectories and meshes from large-scale egocentric videos, yielding an HOI prior for robust generation. Across the GRAB and HOT3D benchmarks, FlowHOI achieves the highest action recognition accuracy and a 1.7$\times$ higher physics simulation success rate than the strongest diffusion-based baseline, while delivering a 40$\times$ inference speedup. We further demonstrate real-robot execution on four dexterous manipulation tasks, illustrating the feasibility of retargeting generated HOI representations to real-robot execution pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16736v3">A Step to Decouple Optimization in 3DGS</a></div>
    <div class="paper-meta">
      📅 2026-02-13
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for real-time novel view synthesis. As an explicit representation optimized through gradient propagation among primitives, optimization widely accepted in deep neural networks (DNNs) is actually adopted in 3DGS, such as synchronous weight updating and Adam with the adaptive gradient. However, considering the physical significance and specific design in 3DGS, there are two overlooked details in the optimization of 3DGS: (i) update step coupling, which induces optimizer state rescaling and costly attribute updates outside the viewpoints, and (ii) gradient coupling in the moment, which may lead to under- or over-effective regularization. Nevertheless, such a complex coupling is under-explored. After revisiting the optimization of 3DGS, we take a step to decouple it and recompose the process into: Sparse Adam, Re-State Regularization and Decoupled Attribute Regularization. Taking a large number of experiments under the 3DGS and 3DGS-MCMC frameworks, our work provides a deeper understanding of these components. Finally, based on the empirical analysis, we re-design the optimization and propose AdamW-GS by re-coupling the beneficial components, under which better optimization efficiency and representation effectiveness are achieved simultaneously.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12796v1">GSM-GS: Geometry-Constrained Single and Multi-view Gaussian Splatting for Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-13
      | 💬 https://aislab-sustech.github.io/GSM-GS/
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting has emerged as a prominent research direction owing to its ultrarapid training speed and high-fidelity rendering capabilities. However, the unstructured and irregular nature of Gaussian point clouds poses challenges to reconstruction accuracy. This limitation frequently causes high-frequency detail loss in complex surface microstructures when relying solely on routine strategies. To address this limitation, we propose GSM-GS: a synergistic optimization framework integrating single-view adaptive sub-region weighting constraints and multi-view spatial structure refinement. For single-view optimization, we leverage image gradient features to partition scenes into texture-rich and texture-less sub-regions. The reconstruction quality is enhanced through adaptive filtering mechanisms guided by depth discrepancy features. This preserves high-weight regions while implementing a dual-branch constraint strategy tailored to regional texture variations, thereby improving geometric detail characterization. For multi-view optimization, we introduce a geometry-guided cross-view point cloud association method combined with a dynamic weight sampling strategy. This constructs 3D structural normal constraints across adjacent point cloud frames, effectively reinforcing multi-view consistency and reconstruction fidelity. Extensive experiments on public datasets demonstrate that our method achieves both competitive rendering quality and geometric reconstruction. See our interactive project page
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22046v3">PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-13
      | 💬 Project page: https://city-super.github.io/PLANING/
    </div>
    <details class="paper-abstract">
      Streaming reconstruction from monocular image sequences remains challenging, as existing methods typically favor either high-quality rendering or accurate geometry, but rarely both. We present PLANING, an efficient on-the-fly reconstruction framework built on a hybrid representation that loosely couples explicit geometric primitives with neural Gaussians, enabling geometry and appearance to be modeled in a decoupled manner. This decoupling supports an online initialization and optimization strategy that separates geometry and appearance updates, yielding stable streaming reconstruction with substantially reduced structural redundancy. PLANING improves dense mesh Chamfer-L2 by 18.52% over PGSR, surpasses ARTDECO by 1.31 dB PSNR, and reconstructs ScanNetV2 scenes in under 100 seconds, over 5x faster than 2D Gaussian Splatting, while matching the quality of offline per-scene optimization. Beyond reconstruction quality, the structural clarity and computational efficiency of PLANING make it well suited for a broad range of downstream applications, such as enabling large-scale scene modeling and simulation-ready environments for embodied AI. Project page: https://city-super.github.io/PLANING/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11638v2">Variation-aware Flexible 3D Gaussian Editing</a></div>
    <div class="paper-meta">
      📅 2026-02-13
    </div>
    <details class="paper-abstract">
      Indirect editing methods for 3D Gaussian Splatting (3DGS) have recently witnessed significant advancements. These approaches operate by first applying edits in the rendered 2D space and subsequently projecting the modifications back into 3D. However, this paradigm inevitably introduces cross-view inconsistencies and constrains both the flexibility and efficiency of the editing process. To address these challenges, we present VF-Editor, which enables native editing of Gaussian primitives by predicting attribute variations in a feedforward manner. To accurately and efficiently estimate these variations, we design a novel variation predictor distilled from 2D editing knowledge. The predictor encodes the input to generate a variation field and employs two learnable, parallel decoding functions to iteratively infer attribute changes for each 3D Gaussian. Thanks to its unified design, VF-Editor can seamlessly distill editing knowledge from diverse 2D editors and strategies into a single predictor, allowing for flexible and effective knowledge transfer into the 3D domain. Extensive experiments on both public and private datasets reveal the inherent limitations of indirect editing pipelines and validate the effectiveness and flexibility of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.03886v2">ArmGS: Composite Gaussian Appearance Refinement for Modeling Dynamic Urban Environments</a></div>
    <div class="paper-meta">
      📅 2026-02-13
      | 💬 ICRA 2026
    </div>
    <details class="paper-abstract">
      This work focuses on modeling dynamic urban environments for autonomous driving simulation. Contemporary data-driven methods using neural radiance fields have achieved photorealistic driving scene modeling, but they suffer from low rendering efficacy. Recently, some approaches have explored 3D Gaussian splatting for modeling dynamic urban scenes, enabling high-fidelity reconstruction and real-time rendering. However, these approaches often neglect to model fine-grained variations between frames and camera viewpoints, leading to suboptimal results. In this work, we propose a new approach named ArmGS that exploits composite driving Gaussian splatting with multi-granularity appearance refinement for autonomous driving scene modeling. The core idea of our approach is devising a multi-level appearance modeling scheme to optimize a set of transformation parameters for composite Gaussian refinement from multiple granularities, ranging from local Gaussian level to global image level and dynamic actor level. This not only models global scene appearance variations between frames and camera viewpoints, but also models local fine-grained changes of background and objects. Extensive experiments on multiple challenging autonomous driving datasets, namely, Waymo, KITTI, NOTR and VKITTI2, demonstrate the superiority of our approach over the state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.13204v2">EDGS: Eliminating Densification for Efficient Convergence of 3DGS</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting reconstructs scenes by starting from a sparse Structure-from-Motion initialization and refining under-reconstructed regions. This process is slow, as it requires multiple densification steps where Gaussians are repeatedly split and adjusted, following a lengthy optimization path. Moreover, this incremental approach often yields suboptimal renderings in high-frequency regions. We propose a fundamentally different approach: eliminate densification with a one-step approximation of scene geometry using triangulated pixels from dense image correspondences. This dense initialization allows us to estimate the rough geometry of the scene while preserving rich details from input RGB images, providing each Gaussian with well-informed color, scale, and position. As a result, we dramatically shorten the optimization path and remove the need for densification. Unlike methods that rely on sparse keypoints, our dense initialization ensures uniform detail across the scene, even in high-frequency regions where other methods struggle. Moreover, since all splats are initialized in parallel at the start of optimization, we remove the need to wait for densification to adjust new Gaussians. EDGS reaches LPIPS and SSIM performance of standard 3DGS significantly faster than existing efficiency-focused approaches. When trained further, it exceeds the reconstruction quality of state-of-the-art models aimed at maximizing fidelity. Our method is fully compatible with other acceleration techniques, making it a versatile and efficient solution that can be integrated with existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12159v1">3DGSNav: Enhancing Vision-Language Model Reasoning for Object Navigation via Active 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Object navigation is a core capability of embodied intelligence, enabling an agent to locate target objects in unknown environments. Recent advances in vision-language models (VLMs) have facilitated zero-shot object navigation (ZSON). However, existing methods often rely on scene abstractions that convert environments into semantic maps or textual representations, causing high-level decision making to be constrained by the accuracy of low-level perception. In this work, we present 3DGSNav, a novel ZSON framework that embeds 3D Gaussian Splatting (3DGS) as persistent memory for VLMs to enhance spatial reasoning. Through active perception, 3DGSNav incrementally constructs a 3DGS representation of the environment, enabling trajectory-guided free-viewpoint rendering of frontier-aware first-person views. Moreover, we design structured visual prompts and integrate them with Chain-of-Thought (CoT) prompting to further improve VLM reasoning. During navigation, a real-time object detector filters potential targets, while VLM-driven active viewpoint switching performs target re-verification, ensuring efficient and reliable recognition. Extensive evaluations across multiple benchmarks and real-world experiments on a quadruped robot demonstrate that our method achieves robust and competitive performance against state-of-the-art approaches.The Project Page:https://aczheng-cai.github.io/3dgsnav.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.26455v2">Stylos: Multi-View 3D Stylization with Single-Forward Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      We present Stylos, a single-forward 3D Gaussian framework for 3D style transfer that operates on unposed content, from a single image to a multi-view collection, conditioned on a separate reference style image. Stylos synthesizes a stylized 3D Gaussian scene without per-scene optimization or precomputed poses, achieving geometry-aware, view-consistent stylization that generalizes to unseen categories, scenes, and styles. At its core, Stylos adopts a Transformer backbone with two pathways: geometry predictions retain self-attention to preserve geometric fidelity, while style is injected via global cross-attention to enforce visual consistency across views. With the addition of a voxel-based 3D style loss that aligns aggregated scene features to style statistics, Stylos enforces view-consistent stylization while preserving geometry. Experiments across multiple datasets demonstrate that Stylos delivers high-quality zero-shot stylization, highlighting the effectiveness of global style-content coupling, the proposed 3D style loss, and the scalability of our framework from single view to large-scale multi-view settings. Our codes are available at https://github.com/HanzhouLiu/Stylos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2210.00379v8">NeRF: Neural Radiance Field in 3D Vision: A Comprehensive Review (Updated Post-Gaussian Splatting)</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Updated Post-Gaussian Splatting
    </div>
    <details class="paper-abstract">
      In March 2020, Neural Radiance Field (NeRF) revolutionized Computer Vision, allowing for implicit, neural network-based scene representation and novel view synthesis. NeRF models have found diverse applications in robotics, urban mapping, autonomous navigation, virtual reality/augmented reality, and more. In August 2023, Gaussian Splatting, a direct competitor to the NeRF-based framework, was proposed, gaining tremendous momentum and overtaking NeRF-based research in terms of interest as the dominant framework for novel view synthesis. We present a comprehensive survey of NeRF papers from the past five years (2020-2025). These include papers from the pre-Gaussian Splatting era, where NeRF dominated the field for novel view synthesis and 3D implicit and hybrid representation neural field learning. We also include works from the post-Gaussian Splatting era where NeRF and implicit/hybrid neural fields found more niche applications. Our survey is organized into architecture and application-based taxonomies in the pre-Gaussian Splatting era, as well as a categorization of active research areas for NeRF, neural field, and implicit/hybrid neural representation methods. We provide an introduction to the theory of NeRF and its training via differentiable volume rendering. We also present a benchmark comparison of the performance and speed of classical NeRF, implicit and hybrid neural representation, and neural field models, and an overview of key datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11714v1">GSO-SLAM: Bidirectionally Coupled Gaussian Splatting and Direct Visual Odometry</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 8 pages, 6 figures, RA-L accepted
    </div>
    <details class="paper-abstract">
      We propose GSO-SLAM, a real-time monocular dense SLAM system that leverages Gaussian scene representation. Unlike existing methods that couple tracking and mapping with a unified scene, incurring computational costs, or loosely integrate them with well-structured tracking frameworks, introducing redundancies, our method bidirectionally couples Visual Odometry (VO) and Gaussian Splatting (GS). Specifically, our approach formulates joint optimization within an Expectation-Maximization (EM) framework, enabling the simultaneous refinement of VO-derived semi-dense depth estimates and the GS representation without additional computational overhead. Moreover, we present Gaussian Splat Initialization, which utilizes image information, keyframe poses, and pixel associations from VO to produce close approximations to the final Gaussian scene, thereby eliminating the need for heuristic methods. Through extensive experiments, we validate the effectiveness of our method, showing that it not only operates in real time but also achieves state-of-the-art geometric/photometric fidelity of the reconstructed scene and tracking accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11705v1">TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Accepted to AAAI 2026. Project page: https://vcc.tech/research/2026/TG-Field
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has revolutionized 3D scene representation with superior efficiency and quality. While recent adaptations for computed tomography (CT) show promise, they struggle with severe artifacts under highly sparse-view projections and dynamic motions. To address these challenges, we propose Tomographic Geometry Field (TG-Field), a geometry-aware Gaussian deformation framework tailored for both static and dynamic CT reconstruction. A multi-resolution hash encoder is employed to capture local spatial priors, regularizing primitive parameters under ultra-sparse settings. We further extend the framework to dynamic reconstruction by introducing time-conditioned representations and a spatiotemporal attention block to adaptively aggregate features, thereby resolving spatiotemporal ambiguities and enforcing temporal coherence. In addition, a motion-flow network models fine-grained respiratory motion to track local anatomical deformations. Extensive experiments on synthetic and real-world datasets demonstrate that TG-Field consistently outperforms existing methods, achieving state-of-the-art reconstruction accuracy under highly sparse-view conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11638v1">Variation-aware Flexible 3D Gaussian Editing</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Indirect editing methods for 3D Gaussian Splatting (3DGS) have recently witnessed significant advancements. These approaches operate by first applying edits in the rendered 2D space and subsequently projecting the modifications back into 3D. However, this paradigm inevitably introduces cross-view inconsistencies and constrains both the flexibility and efficiency of the editing process. To address these challenges, we present VF-Editor, which enables native editing of Gaussian primitives by predicting attribute variations in a feedforward manner. To accurately and efficiently estimate these variations, we design a novel variation predictor distilled from 2D editing knowledge. The predictor encodes the input to generate a variation field and employs two learnable, parallel decoding functions to iteratively infer attribute changes for each 3D Gaussian. Thanks to its unified design, VF-Editor can seamlessly distill editing knowledge from diverse 2D editors and strategies into a single predictor, allowing for flexible and effective knowledge transfer into the 3D domain. Extensive experiments on both public and private datasets reveal the inherent limitations of indirect editing pipelines and validate the effectiveness and flexibility of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00148v2">Learning Physics-Grounded 4D Dynamics with Neural Gaussian Force Fields</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 43 pages, ICLR 2026
    </div>
    <details class="paper-abstract">
      Predicting physical dynamics from raw visual data remains a major challenge in AI. While recent video generation models have achieved impressive visual quality, they still cannot consistently generate physically plausible videos due to a lack of modeling of physical laws. Recent approaches combining 3D Gaussian splatting and physics engines can produce physically plausible videos, but are hindered by high computational costs in both reconstruction and simulation, and often lack robustness in complex real-world scenarios. To address these issues, we introduce Neural Gaussian Force Field (NGFF), an end-to-end neural framework that integrates 3D Gaussian perception with physics-based dynamic modeling to generate interactive, physically realistic 4D videos from multi-view RGB inputs, achieving two orders of magnitude faster than prior Gaussian simulators. To support training, we also present GSCollision, a 4D Gaussian dataset featuring diverse materials, multi-object interactions, and complex scenes, totaling over 640k rendered physical videos (~4 TB). Evaluations on synthetic and real 3D scenarios show NGFF's strong generalization and robustness in physical reasoning, advancing video prediction towards physics-grounded world models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11577v1">LeafFit: Plant Assets Creation from 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Our source code is publicly available at https://github.com/netbeifeng/leaf_fit
    </div>
    <details class="paper-abstract">
      We propose LeafFit, a pipeline that converts 3D Gaussian Splatting (3DGS) of individual plants into editable, instanced mesh assets. While 3DGS faithfully captures complex foliage, its high memory footprint and lack of mesh topology make it incompatible with traditional game production workflows. We address this by leveraging the repetition of leaf shapes; our method segments leaves from the unstructured 3DGS, with optional user interaction included as a fallback. A representative leaf group is selected and converted into a thin, sharp mesh to serve as a template; this template is then fitted to all other leaves via differentiable Moving Least Squares (MLS) deformation. At runtime, the deformation is evaluated efficiently on-the-fly using a vertex shader to minimize storage requirements. Experiments demonstrate that LeafFit achieves higher segmentation quality and deformation accuracy than recent baselines while significantly reducing data size and enabling parameter-level editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11575v1">ReaDy-Go: Real-to-Sim Dynamic 3D Gaussian Splatting Simulation for Environment-Specific Visual Navigation with Moving Obstacles</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Project page: https://syeon-yoo.github.io/ready-go-site/
    </div>
    <details class="paper-abstract">
      Visual navigation models often struggle in real-world dynamic environments due to limited robustness to the sim-to-real gap and the difficulty of training policies tailored to target deployment environments (e.g., households, restaurants, and factories). Although real-to-sim navigation simulation using 3D Gaussian Splatting (GS) can mitigate this gap, prior works have assumed only static scenes or unrealistic dynamic obstacles, despite the importance of safe navigation in dynamic environments. To address these issues, we propose ReaDy-Go, a novel real-to-sim simulation pipeline that synthesizes photorealistic dynamic scenarios for target environments. ReaDy-Go generates photorealistic navigation datasets for dynamic environments by combining a reconstructed static GS scene with dynamic human GS obstacles, and trains policies robust to both the sim-to-real gap and moving obstacles. The pipeline consists of three components: (1) a dynamic GS simulator that integrates scene GS with a human animation module, enabling the insertion of animatable human GS avatars and the synthesis of plausible human motions from 2D trajectories, (2) navigation dataset generation for dynamic environments that leverages the simulator, a robot expert planner designed for dynamic GS representations, and a human planner, and (3) policy learning using the generated datasets. ReaDy-Go outperforms baselines across target environments in both simulation and real-world experiments, demonstrating improved navigation performance even after sim-to-real transfer and in the presence of moving obstacles. Moreover, zero-shot sim-to-real deployment in an unseen environment indicates its generalization potential. Project page: https://syeon-yoo.github.io/ready-go-site/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12314v1">LatentAM: Real-Time, Large-Scale Latent Gaussian Attention Mapping via Online Dictionary Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      We present LatentAM, an online 3D Gaussian Splatting (3DGS) mapping framework that builds scalable latent feature maps from streaming RGB-D observations for open-vocabulary robotic perception. Instead of distilling high-dimensional Vision-Language Model (VLM) embeddings using model-specific decoders, LatentAM proposes an online dictionary learning approach that is both model-agnostic and pretraining-free, enabling plug-and-play integration with different VLMs at test time. Specifically, our approach associates each Gaussian primitive with a compact query vector that can be converted into approximate VLM embeddings using an attention mechanism with a learnable dictionary. The dictionary is initialized efficiently from streaming observations and optimized online to adapt to evolving scene semantics under trust-region regularization. To scale to long trajectories and large environments, we further propose an efficient map management strategy based on voxel hashing, where optimization is restricted to an active local map on the GPU, while the global map is stored and indexed on the CPU to maintain bounded GPU memory usage. Experiments on public benchmarks and a large-scale custom dataset demonstrate that LatentAM attains significantly better feature reconstruction fidelity compared to state-of-the-art methods, while achieving near-real-time speed (12-35 FPS) on the evaluated datasets. Our project page is at: https://junwoonlee.github.io/projects/LatentAM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.06109v2">LighthouseGS: Indoor Structure-aware 3D Gaussian Splatting for Panorama-Style Mobile Captures</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 WACV 2026
    </div>
    <details class="paper-abstract">
      We introduce LighthouseGS, a practical novel view synthesis framework based on 3D Gaussian Splatting that utilizes simple panorama-style captures from a single mobile device. While convenient, this rotation-dominant motion and narrow baseline make accurate camera pose and 3D point estimation challenging, especially in textureless indoor scenes. To address these challenges, LighthouseGS leverages rough geometric priors, such as mobile device camera poses and monocular depth estimation, and utilizes indoor planar structures. Specifically, we propose a new initialization method called plane scaffold assembly to generate consistent 3D points on these structures, followed by a stable pruning strategy to enhance geometry and optimization stability. Additionally, we present geometric and photometric corrections to resolve inconsistencies from motion drift and auto-exposure in mobile devices. Tested on real and synthetic indoor scenes, LighthouseGS delivers photorealistic rendering, outperforming state-of-the-art methods and enabling applications like panoramic view synthesis and object placement. Project page: https://vision3d-lab.github.io/lighthousegs/
    </details>
</div>
