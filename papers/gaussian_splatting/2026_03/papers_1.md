# gaussian splatting - 2026_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08575v3">ReSplat: Learning Recurrent Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Project page: https://haofeixu.github.io/resplat/ Code: https://github.com/cvg/resplat
    </div>
    <details class="paper-abstract">
      While existing feed-forward Gaussian splatting models offer computational efficiency and can generalize to sparse view settings, their performance is fundamentally constrained by relying on a single forward pass for inference. We propose ReSplat, a feed-forward recurrent Gaussian splatting model that iteratively refines 3D Gaussians without explicitly computing gradients. Our key insight is that the Gaussian splatting rendering error serves as a rich feedback signal, guiding the recurrent network to learn effective Gaussian updates. This feedback signal naturally adapts to unseen data distributions at test time, enabling robust generalization across datasets, view counts, and image resolutions. To initialize the recurrent process, we introduce a compact reconstruction model that operates in a $16 \times$ subsampled space, producing $16 \times$ fewer Gaussians than previous per-pixel Gaussian models. This substantially reduces computational overhead and allows for efficient Gaussian updates. Extensive experiments across varying number of input views (2, 8, 16, 32), resolutions ($256 \times 256$ to $540 \times 960$), and datasets (DL3DV, RealEstate10K, and ACID) demonstrate that our method achieves state-of-the-art performance while significantly reducing the number of Gaussians and improving the rendering speed. Our project page is at https://haofeixu.github.io/resplat/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.19297v2">VolSplat: Rethinking Feed-Forward 3D Gaussian Splatting with Voxel-Aligned Prediction</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Project Page: https://lhmd.top/volsplat, Code: https://github.com/ziplab/VolSplat
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) has emerged as a highly effective solution for novel view synthesis. Existing methods predominantly rely on a \emph{pixel-aligned} Gaussian prediction paradigm, where each 2D pixel is mapped to a 3D Gaussian. We rethink this widely adopted formulation and identify several inherent limitations: it renders the reconstructed 3D models heavily dependent on the number of input views, leads to view-biased density distributions, and introduces alignment errors, particularly when source views contain occlusions or low texture. To address these challenges, we introduce VolSplat, a new multi-view feed-forward paradigm that replaces pixel alignment with voxel-aligned Gaussians. By directly predicting Gaussians from a predicted 3D voxel grid, it overcomes pixel alignment's reliance on error-prone 2D feature matching, ensuring robust multi-view consistency. Furthermore, it enables adaptive control over density based on 3D scene complexity, yielding more faithful Gaussians, improved geometric consistency, and enhanced novel-view rendering quality. Experiments on widely used benchmarks demonstrate that VolSplat achieves state-of-the-art performance, while producing more plausible and view-consistent results. The video results, code and trained models are available on our project page: https://lhmd.top/volsplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11969v1">AstroSplat: Physics-Based Gaussian Splatting for Rendering and Reconstruction of Small Celestial Bodies</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 10 pages, 6 figures, conference
    </div>
    <details class="paper-abstract">
      Image-based surface reconstruction and characterization are crucial for missions to small celestial bodies (e.g., asteroids), as it informs mission planning, navigation, and scientific analysis. Recent advances in Gaussian splatting enable high-fidelity neural scene representations but typically rely on a spherical harmonic intensity parameterization that is strictly appearance-based and does not explicitly model material properties or light-surface interactions. We introduce AstroSplat, a physics-based Gaussian splatting framework that integrates planetary reflectance models to improve the autonomous reconstruction and photometric characterization of small-body surfaces from in-situ imagery. The proposed framework is validated on real imagery taken by NASA's Dawn mission, where we demonstrate superior rendering performance and surface reconstruction accuracy compared to the typical spherical harmonic parameterization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.13639v5">4D Radar-Inertial Odometry based on Gaussian Modeling and Multi-Hypothesis Scan Matching</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Our code and results can be publicly accessed at: https://github.com/robotics-upo/gaussian-rio-cpp Accepted for publication in IEEE Robotics and Automation Letters
    </div>
    <details class="paper-abstract">
      4D millimeter-wave (mmWave) radars are sensors that provide robustness against adverse weather conditions (rain, snow, fog, etc.), and as such they are increasingly used for odometry and SLAM (Simultaneous Location and Mapping). However, the noisy and sparse nature of the returned scan data proves to be a challenging obstacle for existing registration algorithms, especially those originally intended for more accurate sensors such as LiDAR. Following the success of 3D Gaussian Splatting for vision, in this paper we propose a summarized representation for radar scenes based on global simultaneous optimization of 3D Gaussians as opposed to voxel-based approaches, and leveraging its inherent Probability Density Function (PDF) for registration. Moreover, we propose optimizing multiple registration hypotheses for better protection against local optima of the PDF. We evaluate our modeling and registration system against state of the art techniques, finding that our system provides richer models and more accurate registration results. Finally, we evaluate the effectiveness of our system in a real Radar-Inertial Odometry task. Experiments using publicly available 4D radar datasets show that our Gaussian approach is comparable to existing registration algorithms, outperforming them in several sequences. Copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09632v2">X-GS: An Extensible Open Framework for Perceiving and Thinking via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, subsequently extending into numerous spatial AI applications. However, most existing 3DGS methods operate in isolation, focusing on specific domains such as pose-free 3DGS, online SLAM, and semantic enrichment. In this paper, we introduce X-GS, an extensible open framework consisting of two major components: the X-GS-Perceiver, which unifies a broad range of 3DGS techniques to enable real-time online SLAM and distill semantic features; and the X-GS-Thinker, which interfaces with downstream multimodal models. In our implementation of the Perceiver, we integrate various 3DGS methods through three novel mechanisms: an online Vector Quantization (VQ) module, a GPU-accelerated grid-sampling scheme, and a highly parallelized pipeline design. The Thinker accommodates vision-language models and utilizes the resulting 3D semantic Gaussians, enabling downstream applications such as object detection, caption generation, and potentially embodied tasks. Experimental results on real-world datasets demonstrate the efficiency and newly unlocked multimodal capabilities of the X-GS framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24053v3">3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Published at ICLR 2026. Code is available at: https://github.com/boschresearch/3dgeer
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves an appealing balance between rendering quality and efficiency, but relies on approximating 3D Gaussians as 2D projections--an assumption that degrades accuracy, especially under generic large field-of-view (FoV) cameras. Despite recent extensions, no prior work has simultaneously achieved both projective exactness and real-time efficiency for general cameras. We introduce 3DGEER, a geometrically exact and efficient Gaussian rendering framework. From first principles, we derive a closed-form expression for integrating Gaussian density along a ray, enabling precise forward rendering and differentiable optimization under arbitrary camera models. To retain efficiency, we propose the Particle Bounding Frustum (PBF), which provides tight ray-Gaussian association without BVH traversal, and the Bipolar Equiangular Projection (BEAP), which unifies FoV representations, accelerates association, and improves reconstruction quality. Experiments on both pinhole and fisheye datasets show that 3DGEER outperforms prior methods across all metrics, runs 5x faster than existing projective exact ray-based baselines, and generalizes to wider FoVs unseen during training--establishing a new state of the art in real-time radiance field rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11543v1">Mango-GS: Enhancing Spatio-Temporal Consistency in Dynamic Scenes Reconstruction using Multi-Frame Node-Guided 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic 3D scenes with photorealistic detail and strong temporal coherence remains a significant challenge. Existing Gaussian splatting approaches for dynamic scene modeling often rely on per-frame optimization, which can overfit to instantaneous states instead of capturing underlying motion dynamics. To address this, we present Mango-GS, a multi-frame, node-guided framework for high-fidelity 4D reconstruction. Mango-GS leverages a temporal Transformer to model motion dependencies within a short window of frames, producing temporally consistent deformations. For efficiency, temporal modeling is confined to a sparse set of control nodes. Each node is represented by a decoupled canonical position and a latent code, providing a stable semantic anchor for motion propagation and preventing correspondence drift under large motion. Our framework is trained end-to-end, enhanced by an input masking strategy and two multi-frame losses to improve robustness. Extensive experiments demonstrate that Mango-GS achieves state-of-the-art reconstruction quality and real-time rendering speed, enabling high-fidelity reconstruction and interactive rendering of dynamic scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11531v1">Mobile-GS: Real-time Gaussian Splatting for Mobile Devices</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Project Page: https://xiaobiaodu.github.io/mobile-gs-project/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful representation for high-quality rendering across a wide range of applications.However, its high computational demands and large storage costs pose significant challenges for deployment on mobile devices. In this work, we propose a mobile-tailored real-time Gaussian Splatting method, dubbed Mobile-GS, enabling efficient inference of Gaussian Splatting on edge devices. Specifically, we first identify alpha blending as the primary computational bottleneck, since it relies on the time-consuming Gaussian depth sorting process. To solve this issue, we propose a depth-aware order-independent rendering scheme that eliminates the need for sorting, thereby substantially accelerating rendering. Although this order-independent rendering improves rendering speed, it may introduce transparency artifacts in regions with overlapping geometry due to the scarcity of rendering order. To address this problem, we propose a neural view-dependent enhancement strategy, enabling more accurate modeling of view-dependent effects conditioned on viewing direction, 3D Gaussian geometry, and appearance attributes. In this way, Mobile-GS can achieve both high-quality and real-time rendering. Furthermore, to facilitate deployment on memory-constrained mobile platforms, we also introduce first-order spherical harmonics distillation, a neural vector quantization technique, and a contribution-based pruning strategy to reduce the number of Gaussian primitives and compress the 3D Gaussian representation with the assistance of neural networks. Extensive experiments demonstrate that our proposed Mobile-GS achieves real-time rendering and compact model size while preserving high visual quality, making it well-suited for mobile applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10446v2">SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Generating natural and linguistically accurate sign language avatars remains a formidable challenge. Current Sign Language Production (SLP) frameworks face a stark trade-off: direct text-to-pose models suffer from regression-to-the-mean effects, while dictionary-retrieval methods produce robotic, disjointed transitions. To resolve this, we propose a novel training paradigm that leverages sparse keyframes to capture the true underlying kinematic distribution of human signing. By predicting dense motion from these discrete anchors, our approach mitigates regression-to-the-mean while ensuring fluid articulation. To realize this paradigm at scale, we first introduce FAST, an ultra-efficient sign segmentation model that automatically mines precise temporal boundaries. We then present SignSparK, a large-scale Conditional Flow Matching (CFM) framework that utilizes these extracted anchors to synthesize 3D signing sequences in SMPL-X and MANO spaces. This keyframe-driven formulation also uniquely unlocks Keyframe-to-Pose (KF2P) generation, making precise spatiotemporal editing of signing sequences possible. Furthermore, our adopted reconstruction-based CFM objective also enables high-fidelity synthesis in fewer than ten sampling steps; this allows SignSparK to scale across four distinct sign languages, establishing the largest multilingual SLP framework to date. Finally, by integrating 3D Gaussian Splatting for photorealistic rendering, we demonstrate through extensive evaluation that SignSparK establishes a new state-of-the-art across diverse SLP tasks and multilingual benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11298v1">InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      High dynamic range (HDR) novel view synthesis (NVS) aims to reconstruct HDR scenes from multi-exposure low dynamic range (LDR) images. Existing HDR pipelines heavily rely on known camera poses, well-initialized dense point clouds, and time-consuming per-scene optimization. Current feed-forward alternatives overlook the HDR problem by assuming exposure-invariant appearance. To bridge this gap, we propose InstantHDR, a feed-forward network that reconstructs 3D HDR scenes from uncalibrated multi-exposure LDR collections in a single forward pass. Specifically, we design a geometry-guided appearance modeling for multi-exposure fusion, and a meta-network for generalizable scene-specific tone mapping. Due to the lack of HDR scene data, we build a pre-training dataset, called HDR-Pretrain, for generalizable feed-forward HDR models, featuring 168 Blender-rendered scenes, diverse lighting types, and multiple camera response functions. Comprehensive experiments show that our InstantHDR delivers comparable synthesis performance to the state-of-the-art optimization-based HDR methods while enjoying $\sim700\times$ and $\sim20\times$ reconstruction speed improvement with our single-forward and post-optimization settings. All code, models, and datasets will be released after the review process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08958v3">Grow with the Flow: 4D Reconstruction of Growing Plants with Gaussian Flow Fields</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Project page: https://weihanluo.ca/growflow/
    </div>
    <details class="paper-abstract">
      Modeling the time-varying 3D appearance of plants during growth poses unique challenges: unlike most dynamic scenes, plants continuously generate new geometry as they expand, branch, and differentiate. Existing dynamic scene representations are ill-suited to this setting: deformation fields provide insufficient constraints to yield physically plausible scene dynamics, and 4D Gaussian splatting represents the same physical structures with different Gaussian primitives at different times, breaking temporal consistency. We introduce GrowFlow, a dynamic representation that couples 3D Gaussian primitives with a neural ordinary differential equation to model plant growth as a continuous flow field over geometric parameters (position, scale, and orientation). Our representation enables consistent appearance rendering and models nonlinear, continuous-time growth dynamics with full temporal correspondences for every primitive. To initialize a sufficient set of Gaussian primitives, we first reconstruct the mature plant and then learn a reverse-growth process, effectively simulating the plant's developmental history in reverse. GrowFlow achieves superior image quality and geometric coherence compared to prior methods on a new, multi-view timelapse dataset of plant growth, and provides the first temporally coherent representation for appearance modeling of growing 3D structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13310v2">InstantSfM: Towards GPU-Native SfM for the Deep Learning Era</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Structure-from-Motion (SfM) is a fundamental technique for recovering camera poses and scene structure from multi-view imagery, serving as a critical upstream component for applications ranging from 3D reconstruction to modern neural scene representations such as 3D Gaussian Splatting. However, most mature SfM systems remain CPU-centric and built upon traditional optimization toolchains, creating a growing mismatch with modern GPU-based, learning-driven pipelines and limiting scalability in large-scale scenes. While recent advances in GPU-accelerated bundle adjustment (BA) have demonstrated the potential of parallel sparse optimization, extending these techniques to build a complete global SfM system remains challenging due to unresolved issues in metric scale recovery and numerical robustness. In this paper, we implement a fully GPU-based and PyTorch-compatible global SfM system, named InstantSfM, to integrate seamlessly with modern learning pipelines. InstantSfM embeds metric depth priors directly into both global positioning and BA through a depth-constrained Jacobian structure, thereby resolving scale ambiguity within the optimization framework. To ensure numerical stability, we employ explicit filtering of under-constrained variables for the Jacobian matrix in an optimized GPU-friendly manner. Extensive experiments on diverse datasets demonstrate that InstantSfM achieves state-of-the-art efficiency while maintaining reconstruction accuracy comparable to both established classical pipelines and recent learning-based methods, showing up to ${\sim40\times}$ speedup over COLMAP on large-scale scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10893v1">S2D: Sparse to Dense Lifting for 3D Reconstruction with Minimal Inputs</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Explicit 3D representations have already become an essential medium for 3D simulation and understanding. However, the most commonly used point cloud and 3D Gaussian Splatting (3DGS) each suffer from non-photorealistic rendering and significant degradation under sparse inputs. In this paper, we introduce Sparse to Dense lifting (S2D), a novel pipeline that bridges the two representations and achieves high-quality 3DGS reconstruction with minimal inputs. Specifically, the S2D lifting is two-fold. We first present an efficient one-step diffusion model that lifts sparse point cloud for high-fidelity image artifact fixing. Meanwhile, to reconstruct 3D consistent scenes, we also design a corresponding reconstruction strategy with random sample drop and weighted gradient for robust model fitting from sparse input views to dense novel views. Extensive experiments show that S2D achieves the best consistency in generating novel view guidance and first-tier sparse view reconstruction quality under different input sparsity. By reconstructing stable scenes with the least possible captures among existing methods, S2D enables minimal input requirements for 3DGS applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.14373v3">SEGA: Drivable 3D Gaussian Head Avatar from a Single Image</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Creating photorealistic 3D head avatars from limited input has become increasingly important for applications in virtual reality, telepresence, and digital entertainment. While recent advances like neural rendering and 3D Gaussian splatting have enabled high-quality digital human avatar creation and animation, most methods rely on multiple images or multi-view inputs, limiting their practicality for real-world use. In this paper, we propose SEGA, a novel approach for Single-imagE-based 3D drivable Gaussian head Avatar creation that combines generalized prior models with a new hierarchical UV-space Gaussian Splatting framework. SEGA seamlessly combines priors derived from large-scale 2D datasets with 3D priors learned from multi-view, multi-expression, and multi-ID data, achieving robust generalization to unseen identities while ensuring 3D consistency across novel viewpoints and expressions. We further present a hierarchical UV-space Gaussian Splatting framework that leverages FLAME-based structural priors and employs a dual-branch architecture to disentangle dynamic and static facial components effectively. The dynamic branch encodes expression-driven fine details, while the static branch focuses on expression-invariant regions, enabling efficient parameter inference and precomputation. This design maximizes the utility of limited 3D data and achieves real-time performance for animation and rendering. Additionally, SEGA performs person-specific fine-tuning to further enhance the fidelity and realism of the generated avatars. Experiments show our method outperforms state-of-the-art approaches in generalization ability, identity preservation, and expression realism, advancing one-shot avatar creation for practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22046v4">PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Project page: https://city-super.github.io/PLANING/
    </div>
    <details class="paper-abstract">
      Streaming reconstruction from monocular image sequences remains challenging, as existing methods typically favor either high-quality rendering or accurate geometry, but rarely both. We present PLANING, an efficient on-the-fly reconstruction framework built on a hybrid representation that loosely couples explicit geometric primitives with neural Gaussians, enabling geometry and appearance to be modeled in a decoupled manner. This decoupling supports an online initialization and optimization strategy that separates geometry and appearance updates, yielding stable streaming reconstruction with substantially reduced structural redundancy. PLANING improves dense mesh Chamfer-L2 by 18.52% over PGSR, surpasses ARTDECO by 1.31 dB PSNR, and reconstructs ScanNetV2 scenes in under 100 seconds, over 5x faster than 2D Gaussian Splatting, while matching the quality of offline per-scene optimization. Beyond reconstruction quality, the structural clarity and computational efficiency of PLANING make it well suited for a broad range of downstream applications, such as enabling large-scale scene modeling and simulation-ready environments for embodied AI. Project page: https://city-super.github.io/PLANING/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10801v1">PolGS++: Physically-Guided Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 arXiv admin note: substantial text overlap with arXiv:2509.19726
    </div>
    <details class="paper-abstract">
      Accurate reconstruction of reflective surfaces remains a fundamental challenge in computer vision, with broad applications in real-time virtual reality and digital content creation. Although 3D Gaussian Splatting (3DGS) enables efficient novel-view rendering with explicit representations, its performance on reflective surfaces still lags behind implicit neural methods, especially in recovering fine geometry and surface normals. To address this gap, we propose PolGS++, a physically-guided polarimetric Gaussian Splatting framework for fast reflective surface reconstruction. Specifically, we integrate a polarized BRDF (pBRDF) model into 3DGS to explicitly decouple diffuse and specular components, providing physically grounded reflectance modeling and stronger geometric cues for reflective surface recovery. Furthermore, we introduce a depth-guided visibility mask acquisition mechanism that enables angle-of-polarization (AoP)-based tangent-space consistency constraints in Gaussian Splatting without costly ray-tracing intersections. This physically guided design improves reconstruction quality and efficiency, requiring only about 10 minutes of training. Extensive experiments on both synthetic and real-world datasets validate the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10638v1">Splat2Real: Novel-view Scaling for Physical AI with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Physical AI faces viewpoint shift between training and deployment, and novel-view robustness is essential for monocular RGB-to-3D perception. We cast Real2Render2Real monocular depth pretraining as imitation-learning-style supervision from a digital twin oracle: a student depth network imitates expert metric depth/visibility rendered from a scene mesh, while 3DGS supplies scalable novel-view observations. We present Splat2Real, centered on novel-view scaling: performance depends more on which views are added than on raw view count. We introduce CN-Coverage, a coverage+novelty curriculum that greedily selects views by geometry gain and an extrapolation penalty, plus a quality-aware guardrail fallback for low-reliability teachers. Across 20 TUM RGB-D sequences with step-matched budgets (N=0 to 2000 additional rendered views, with N unique <= 500 and resampling for larger budgets), naive scaling is unstable; CN-Coverage mitigates worst-case regressions relative to Robot/Coverage policies, and GOL-Gated CN-Coverage provides the strongest medium-high-budget stability with the lowest high-novelty tail error. Downstream control-proxy results versus N provides embodied-relevance evidence by shifting safety/progress trade-offs under viewpoint shift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10551v1">P-GSVC: Layered Progressive 2D Gaussian Splatting for Scalable Image and Video</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 MMSys 2026; Project Website: see https://longanwang-cs.github.io/PGSVC-webpage/
    </div>
    <details class="paper-abstract">
      Gaussian splatting has emerged as a competitive explicit representation for image and video reconstruction. In this work, we present P-GSVC, the first layered progressive 2D Gaussian splatting framework that provides a unified solution for scalable Gaussian representation in both images and videos. P-GSVC organizes 2D Gaussian splats into a base layer and successive enhancement layers, enabling coarse-to-fine reconstructions. To effectively optimize this layered representation, we propose a joint training strategy that simultaneously updates Gaussians across layers, aligning their optimization trajectories to ensure inter-layer compatibility and a stable progressive reconstruction. P-GSVC supports scalability in terms of both quality and resolution. Our experiments show that the joint training strategy can gain up to 1.9 dB improvement in PSNR for video and 2.6 dB improvement in PSNR for image when compared to methods that perform sequential layer-wise training. Project page: https://longanwang-cs.github.io/PGSVC-webpage/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16410v3">REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 CVPR 2026 Accepted
    </div>
    <details class="paper-abstract">
      Bridging the gap between complex human instructions and precise 3D object grounding remains a significant challenge in vision and robotics. Existing 3D segmentation methods often struggle to interpret ambiguous, reasoning-based instructions, while 2D vision-language models that excel at such reasoning lack intrinsic 3D spatial understanding. In this paper, we introduce REALM, an innovative MLLM-agent framework that enables open-world reasoning-based segmentation without requiring extensive 3D-specific post-training. We perform segmentation directly on 3D Gaussian Splatting representations, capitalizing on their ability to render photorealistic novel views that are highly suitable for MLLM comprehension. As directly feeding one or more rendered views to the MLLM can lead to high sensitivity to viewpoint selection, we propose a novel Global-to-Local Spatial Grounding strategy. Specifically, multiple global views are first fed into the MLLM agent in parallel for coarse-level localization, aggregating responses to robustly identify the target object. Then, several close-up novel views of the object are synthesized to perform fine-grained local segmentation, yielding accurate and consistent 3D masks. Extensive experiments show that REALM achieves remarkable performance in interpreting both explicit and implicit instructions across LERF, 3D-OVS, and our newly introduced REALM3D benchmarks. Furthermore, our agent framework seamlessly supports a range of 3D interaction tasks, including object removal, replacement, and style transfer, demonstrating its practical utility and versatility. Project page: https://ChangyueShi.github.io/REALM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09968v1">ReCoSplat: Autoregressive Feed-Forward Gaussian Splatting Using Render-and-Compare</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Online novel view synthesis remains challenging, requiring robust scene reconstruction from sequential, often unposed, observations. We present ReCoSplat, an autoregressive feed-forward Gaussian Splatting model supporting posed or unposed inputs, with or without camera intrinsics. While assembling local Gaussians using camera poses scales better than canonical-space prediction, it creates a dilemma during training: using ground-truth poses ensures stability but causes a distribution mismatch when predicted poses are used at inference. To address this, we introduce a Render-and-Compare (ReCo) module. ReCo renders the current reconstruction from the predicted viewpoint and compares it with the incoming observation, providing a stable conditioning signal that compensates for pose errors. To support long sequences, we propose a hybrid KV cache compression strategy combining early-layer truncation with chunk-level selective retention, reducing the KV cache size by over 90% for 100+ frames. ReCoSplat achieves state-of-the-art performance across different input settings on both in- and out-of-distribution benchmarks. Code and pretrained models will be released. Our project page is at https://freemancheng.com/ReCoSplat .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07789v2">SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      2D Gaussian Splatting has emerged as a novel image representation technique that can support efficient rendering on low-end devices. However, scaling to high-resolution images requires optimizing and storing millions of unstructured Gaussian primitives independently, leading to slow convergence and redundant parameters. To address this, we propose Structured Gaussian Image (SGI), a compact and efficient framework for representing high-resolution images. SGI decomposes a complex image into multi-scale local spaces defined by a set of seeds. Each seed corresponds to a spatially coherent region and, together with lightweight multi-layer perceptrons (MLPs), generates structured implicit 2D neural Gaussians. This seed-based formulation imposes structural regularity on otherwise unstructured Gaussian primitives, which facilitates entropy-based compression at the seed level to reduce the total storage. However, optimizing seed parameters directly on high-resolution images is a challenging and non-trivial task. Therefore, we designed a multi-scale fitting strategy that refines the seed representation in a coarse-to-fine manner, substantially accelerating convergence. Quantitative and qualitative evaluations demonstrate that SGI achieves up to 7.5x compression over prior non-quantized 2D Gaussian methods and 1.6x over quantized ones, while also delivering 1.6x and 6.5x faster optimization, respectively, without degrading, and often improving, image fidelity. Code is available at https://github.com/zx-pan/SGI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09718v1">GSStream: 3D Gaussian Splatting based Volumetric Scene Streaming System</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Recently, the 3D Gaussian splatting (3DGS) technique for real-time radiance field rendering has revolutionized the field of volumetric scene representation, providing users with an immersive experience. But in return, it also poses a large amount of data volume, which is extremely bandwidth-intensive. Cutting-edge researchers have tried to introduce different approaches and construct multiple variants for 3DGS to obtain a more compact scene representation, but it is still challenging for real-time distribution. In this paper, we propose GSStream, a novel volumetric scene streaming system to support 3DGS data format. Specifically, GSStream integrates a collaborative viewport prediction module to better predict users' future behaviors by learning collaborative priors and historical priors from multiple users and users' viewport sequences and a deep reinforcement learning (DRL)-based bitrate adaptation module to tackle the state and action space variability challenge of the bitrate adaptation problem, achieving efficient volumetric scene delivery. Besides, we first build a user viewport trajectory dataset for volumetric scenes to support the training and streaming simulation. Extensive experiments prove that our proposed GSStream system outperforms existing representative volumetric scene streaming systems in visual quality and network usage. Demo video: https://youtu.be/3WEe8PN8yvA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09703v1">ProGS: Towards Progressive Coding for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      With the emergence of 3D Gaussian Splatting (3DGS), numerous pioneering efforts have been made to address the effective compression issue of massive 3DGS data. 3DGS offers an efficient and scalable representation of 3D scenes by utilizing learnable 3D Gaussians, but the large size of the generated data has posed significant challenges for storage and transmission. Existing methods, however, have been limited by their inability to support progressive coding, a crucial feature in streaming applications with varying bandwidth. To tackle this limitation, this paper introduce a novel approach that organizes 3DGS data into an octree structure, enabling efficient progressive coding. The proposed ProGS is a streaming-friendly codec that facilitates progressive coding for 3D Gaussian splatting, and significantly improves both compression efficiency and visual fidelity. The proposed method incorporates mutual information enhancement mechanisms to mitigate structural redundancy, leveraging the relevance between nodes in the octree hierarchy. By adapting the octree structure and dynamically adjusting the anchor nodes, ProGS ensures scalable data compression without compromising the rendering quality. ProGS achieves a remarkable 45X reduction in file storage compared to the original 3DGS format, while simultaneously improving visual performance by over 10%. This demonstrates that ProGS can provide a robust solution for real-time applications with varying network conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09673v1">VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Accepted to CVPR 2026
    </div>
    <details class="paper-abstract">
      Simultaneous Localization and Mapping (SLAM) with 3D Gaussian Splatting (3DGS) enables fast, differentiable rendering and high-fidelity reconstruction across diverse real-world scenes. However, existing 3DGS-SLAM approaches handle measurement reliability implicitly, making pose estimation and global alignment susceptible to drift in low-texture regions, transparent surfaces, or areas with complex reflectance properties. To this end, we introduce VarSplat, an uncertainty-aware 3DGS-SLAM system that explicitly learns per-splat appearance variance. By using the law of total variance with alpha compositing, we then render differentiable per-pixel uncertainty map via efficient, single-pass rasterization. This map guides tracking, submap registration, and loop detection toward focusing on reliable regions and contributes to more stable optimization. Experimental results on Replica (synthetic) and TUM-RGBD, ScanNet, and ScanNet++ (real-world) show that VarSplat improves robustness and achieves competitive or superior tracking, mapping, and novel view synthesis rendering compared to existing studies for dense RGB-D SLAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09668v1">DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Accepted by ICLR 2026. Project page: https://zju3dv.github.io/DiffWind/
    </div>
    <details class="paper-abstract">
      Modeling wind-driven object dynamics from video observations is highly challenging due to the invisibility and spatio-temporal variability of wind, as well as the complex deformations of objects. We present DiffWind, a physics-informed differentiable framework that unifies wind-object interaction modeling, video-based reconstruction, and forward simulation. Specifically, we represent wind as a grid-based physical field and objects as particle systems derived from 3D Gaussian Splatting, with their interaction modeled by the Material Point Method (MPM). To recover wind-driven object dynamics, we introduce a reconstruction framework that jointly optimizes the spatio-temporal wind force field and object motion through differentiable rendering and simulation. To ensure physical validity, we incorporate the Lattice Boltzmann Method (LBM) as a physics-informed constraint, enforcing compliance with fluid dynamics laws. Beyond reconstruction, our method naturally supports forward simulation under novel wind conditions and enables new applications such as wind retargeting. We further introduce WD-Objects, a dataset of synthetic and real-world wind-driven scenes. Extensive experiments demonstrate that our method significantly outperforms prior dynamic scene modeling approaches in both reconstruction accuracy and simulation fidelity, opening a new avenue for video-based wind-object interaction modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.04859v3">CoRe-GS: Coarse-to-Refined Gaussian Splatting with Semantic Object Focus</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Fast and efficient 3D reconstruction is essential for time-critical robotic applications such as tele-guidance and disaster response, where operators must rapidly analyze specific points of interest (POIs). Existing semantic Gaussian Splatting (GS) approaches optimize the entire scene uniformly, incurring substantial computational cost even when only a small subset of the scene is operationally relevant. We propose CoRe-GS, a coarse-to-refine GS framework that enables task-driven POI-focused optimization. Our method first produces a segmentation-ready GS representation using a lightweight late-stage semantic refinement. Subsequently, only Gaussians associated with the selected POI are further optimized, reducing unnecessary background computation. To mitigate segmentation-induced outliers (floaters) during selective refinement, we introduce a color-based filtering mechanism that removes inconsistent Gaussians without requiring mask rasterization. We evaluate robustness multiple datasets. On LERF-Mask, our segmentation-ready representation achieves competitive mIoU using tremendously fewer optimization steps. Across synthetic and real-world datasets (NeRDS360, SCRREAM, Tanks and Temples), CoRe-GS drastically reduces training time compared to full semantic GS while improving POI reconstruction quality and mitigating floaters. These results demonstrate that task-aware selective refinement enables faster and higher-quality scene reconstruction tailored to robotic operational needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.18380v2">ARSGaussian: 3D Gaussian Splatting with LiDAR for Aerial Remote Sensing Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 This is the author's version of a work that was accepted for publication in [ISPRS]. Changes resulting from the publishing process... may not be reflected in this document
    </div>
    <details class="paper-abstract">
      Novel View Synthesis (NVS) can reconstruct scenes from multi-view images and synthesize novel images from new viewpoints, which provides technical support for tasks such as target recognition and environmental perception. Aerial remote sensing can conveniently capture a wealth of multi-view images with just a few flights. However, the challenges brought by large distances and sparse viewing angles during collection can cause the model to easily produce floaters and overgrowth issues due to geometric estimation errors. This results in low visual quality and a lack of precise geometric estimation capabilities. Therefore, this study presents ARSGaussian, an innovative novel view synthesis (NVS) method for aerial remote sensing. The method incorporates LiDAR point cloud as constraints into the 3D Gaussian Splatting approach, adaptively guiding the Gaussians to grow and split along geometric benchmarks, thereby addressing the overgrowth and floaters issues. Additionally, considering the geometric distortions arising from data acquisition, coordinate transformations with distortion parameters are integrated to replace the simple pinhole camera model parameters to achieve pixel-level alignment between LiDAR point cloud and multi-view optical images, facilitating the accurate fusion of heterogeneous data and achieving the high-precision geo-alignment. Moreover, depth, normal and scale consistency losses are introduced into the regularization process to guide Gaussians toward real depth and plane representations, significantly improving geometric estimation accuracy. To address the current lack of dense airborne hybrid datasets, we have established and released AIR-LONGYAN, an open-source dataset containing a dense LiDAR point cloud (8 pts/m) and multi-view optical images captured by airborne scanners and cameras in diverse scenes....
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09291v1">DenoiseSplat: Feed-Forward Gaussian Splatting for Noisy 3D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      3D scene reconstruction and novel-view synthesis are fundamental for VR, robotics, and content creation. However, most NeRF and 3D Gaussian Splatting pipelines assume clean inputs and degrade under real noise and artifacts. We therefore propose DenoiseSplat, a feed-forward 3D Gaussian splatting method for noisy multi-view images. We build a large-scale, scene-consistent noisy--clean benchmark on RE10K by injecting Gaussian, Poisson, speckle, and salt-and-pepper noise with controlled intensities. With a lightweight MVSplat-style feed-forward backbone, we train end-to-end using only clean 2D renderings as supervision and no 3D ground truth. On noisy RE10K, DenoiseSplat outperforms vanilla MVSplat and a strong two-stage baseline (IDF + MVSplat) in PSNR/SSIM and LPIPS across noise types and levels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09285v1">Learning Convex Decomposition via Feature Fields</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 14 pages, 12 figures
    </div>
    <details class="paper-abstract">
      This work proposes a new formulation to the long-standing problem of convex decomposition through learning feature fields, enabling the first feed-forward model for open-world convex decomposition. Our method produces high-quality decompositions of 3D shapes into a union of convex bodies, which are essential to accelerate collision detection in physical simulation, amongst many other applications. The key insight is to adopt a feature learning approach and learn a continuous feature field that can later be clustered to yield a good convex decomposition via our self-supervised, purely-geometric objective derived from the classical definition of convexity. Our formulation can be used for single shape optimization, but more importantly, feature prediction unlocks scalable, self-supervised learning on large datasets resulting in the first learned open-world model for convex decomposition. Experiments show that our decompositions are higher-quality than alternatives and generalize across open-world objects as well as across representations to meshes, CAD models, and even Gaussian splats. https://research.nvidia.com/labs/sil/projects/learning-convex-decomp/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09277v1">Speeding Up the Learning of 3D Gaussians with Much Shorter Gaussian Lists</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Accepted to CVPR 2026. Project page: https://github.com/MachinePerceptionLab/ShorterSplatting
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has become a vital tool for learning a radiance field from multiple posed images. Although 3DGS shows great advantages over NeRF in terms of rendering quality and efficiency, it remains a research challenge to further improve the efficiency of learning 3D Gaussians. To overcome this challenge, we propose novel training strategies and losses to shorten each Gaussian list used to render a pixel, which speeds up the splatting by involving fewer Gaussians along a ray. Specifically, we shrink the size of each Gaussian by resetting their scales regularly, encouraging smaller Gaussians to cover fewer nearby pixels, which shortens the Gaussian lists of pixels. Additionally, we introduce an entropy constraint on the alpha blending procedure to sharpen the weight distribution of Gaussians along each ray, which drives dominant weights larger while making minor weights smaller. As a result, each Gaussian becomes more focused on the pixels where it is dominant, which reduces its impact on nearby pixels, leading to even shorter Gaussian lists. Eventually, we integrate our method into a rendering resolution scheduler which further improves efficiency through progressive resolution increase. We evaluate our method by comparing it with state-of-the-art methods on widely used benchmarks. Our results show significant advantages over others in efficiency without sacrificing rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08997v1">SkipGS: Post-Densification Backward Skipping for Efficient 3DGS Training</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves real-time novel-view synthesis by optimizing millions of anisotropic Gaussians, yet its training remains expensive, with the backward pass dominating runtime in the post-densification refinement phase. We observe substantial update redundancy in this phase: many sampled views have near-plateaued losses and provide diminishing gradient benefits, but standard training still runs full backpropagation. We propose SkipGS with a novel view-adaptive backward gating mechanism for efficient post-densification training. SkipGS always performs the forward pass to update per-view loss statistics, and selectively skips backward passes when the sampled view's loss is consistent with its recent per-view baseline, while enforcing a minimum backward budget for stable optimization. On Mip-NeRF 360, compared to 3DGS, SkipGS reduces end-to-end training time by 23.1%, driven by a 42.0% reduction in post-densification time, with comparable reconstruction quality. Because it only changes when to backpropagate -- without modifying the renderer, representation, or loss -- SkipGS is plug-and-play and compatible with other complementary efficiency strategies for additive speedups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08983v1">SurgCalib: Gaussian Splatting-Based Hand-Eye Calibration for Robot-Assisted Minimally Invasive Surgery</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 9 pages, 7 figures
    </div>
    <details class="paper-abstract">
      We present a Gaussian Splatting-based framework for hand-eye calibration of the da Vinci surgical robot. In a vision-guided robotic system, accurate estimation of the rigid transformation between the robot base and the camera frame is essential for reliable closed-loop control. For cable-driven surgical robots, this task faces unique challenges. The encoders of surgical instruments often produce inaccurate proprioceptive measurements due to cable stretch and backlash. Conventional hand-eye calibration approaches typically rely on known fiducial patterns and solve the AX = XB formulation. While effective, introducing additional markers into the operating room (OR) environment can violate sterility protocols and disrupt surgical workflows. In this study, we propose SurgCalib, an automatic, markerless framework that has the potential to be used in the OR. SurgCalib first initializes the pose of the surgical instrument using raw kinematic measurements and subsequently refines this pose through a two-phase optimization procedure under the RCM constraint within a Gaussian Splatting-based differentiable rendering pipeline. We evaluate the proposed method on the public dVRK benchmark, SurgPose. The results demonstrate average 2D tool-tip reprojection errors of 12.24 px (2.06 mm) and 11.33 px (1.9 mm), and 3D tool-tip Euclidean distance errors of 5.98 mm and 4.75 mm, for the left and right instruments, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08809v1">Where, What, Why: Toward Explainable 3D-GS Watermarking</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      As 3D Gaussian Splatting becomes the de facto representation for interactive 3D assets, robust yet imperceptible watermarking is critical. We present a representation-native framework that separates where to write from how to preserve quality. A Trio-Experts module operates directly on Gaussian primitives to derive priors for carrier selection, while a Safety and Budget Aware Gate (SBAG) allocates Gaussians to watermark carriers, optimized for bit resilience under perturbation and bitrate budgets, and to visual compensators that are insulated from watermark loss. To maintain fidelity, we introduce a channel-wise group mask that controls gradient propagation for carriers and compensators, thereby limiting Gaussian parameter updates, repairing local artifacts, and preserving high-frequency details without increasing runtime. Our design yields view-consistent watermark persistence and strong robustness against common image distortions such as compression and noise, while achieving a favorable robustness-quality trade-off compared with prior methods. In addition, decoupled finetuning provides per-Gaussian attributions that reveal where the message is carried and why those carriers are selected, enabling auditable explainability. Compared with state-of-the-art methods, our approach achieves a PSNR improvement of +0.83 dB and a bit-accuracy gain of +1.24%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08661v1">ImprovedGS+: A High-Performance C++/CUDA Re-Implementation Strategy for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 6 pages, 1 figure. Technical Report. This work introduces ImprovedGS+, a library-free C++/CUDA implementation for 3D Gaussian Splatting within the LichtFeld-Studio framework. Source code available at https://github.com/jordizv/ImprovedGS-Plus
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3DGS) have shifted the focus toward balancing reconstruction fidelity with computational efficiency. In this work, we propose ImprovedGS+, a high-performance, low-level reinvention of the ImprovedGS strategy, implemented natively within the LichtFeld-Studio framework. By transitioning from high-level Python logic to hardware-optimized C++/CUDA kernels, we achieve a significant reduction in host-device synchronization and training latency. Our implementation introduces a Long-Axis-Split (LAS) CUDA kernel, custom Laplacian-based importance kernels with Non-Maximum Suppression (NMS) for edge scores, and an adaptive Exponential Scale Scheduler. Experimental results on the Mip-NeRF360 dataset demonstrate that ImprovedGS+ establishes a new Pareto-optimal front for scene reconstruction. Our 1M-budget variant outperforms the state-of-the-art MCMC baseline by achieving a 26.8% reduction in training time (saving 17 minutes per session) and utilizing 13.3% fewer Gaussians while maintaining superior visual quality. Furthermore, our full variant demonstrates a 1.28 dB PSNR increase over the ADC baseline with a 38.4% reduction in parametric complexity. These results validate ImprovedGS+ as a scalable, high-speed solution that upholds the core pillars of Speed, Quality, and Usability within the LichtFeld-Studio ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.23273v2">LIVE-GS: Online LiDAR-Inertial-Visual State Estimation and Globally Consistent Mapping with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) enabled photorealistic mapping, its integration into SLAM has largely followed traditional camera-centric pipelines. As a result, they inherit well-known weaknesses such as high computational load, failure in texture-poor or illumination-varying environments, and limited operational range, particularly for RGB-D setups. On the other hand, LiDAR emerges as a robust alternative, but its integration with 3DGS introduces new challenges, such as the need for tighter global alignment for photorealistic quality and prolonged optimization times caused by sparse data. To address these challenges, we propose LIVE-GS, an online LiDAR-Inertial Visual SLAM framework that tightly couples 3D Gaussian Splatting with LiDAR-based surfels to ensure high-precision map consistency through global geometric optimization. Particularly, to handle sparse data, our system employs a depth-invariant Gaussian initialization strategy for efficient representation and a bounded sigmoid constraint to prevent uncontrolled Gaussian growth. Experiments on public and our datasets demonstrate competitive performance in rendering quality and map-building efficiency compared with representative 3DGS SLAM baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08503v1">Spherical-GOF: Geometry-Aware Panoramic Gaussian Opacity Fields for 3D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 The source code and dataset will be released at https://github.com/1170632760/Spherical-GOF
    </div>
    <details class="paper-abstract">
      Omnidirectional images are increasingly used in robotics and vision due to their wide field of view. However, extending 3D Gaussian Splatting (3DGS) to panoramic camera models remains challenging, as existing formulations are designed for perspective projections and naive adaptations often introduce distortion and geometric inconsistencies. We present Spherical-GOF, an omnidirectional Gaussian rendering framework built upon Gaussian Opacity Fields (GOF). Unlike projection-based rasterization, Spherical-GOF performs GOF ray sampling directly on the unit sphere in spherical ray space, enabling consistent ray-Gaussian interactions for panoramic rendering. To make the spherical ray casting efficient and robust, we derive a conservative spherical bounding rule for fast ray-Gaussian culling and introduce a spherical filtering scheme that adapts Gaussian footprints to distortion-varying panoramic pixel sampling. Extensive experiments on standard panoramic benchmarks (OmniBlender and OmniPhotos) demonstrate competitive photometric quality and substantially improved geometric consistency. Compared with the strongest baseline, Spherical-GOF reduces depth reprojection error by 57% and improves cycle inlier ratio by 21%. Qualitative results show cleaner depth and more coherent normal maps, with strong robustness to global panorama rotations. We further validate generalization on OmniRob, a real-world robotic omnidirectional dataset introduced in this work, featuring UAV and quadruped platforms. The source code and the OmniRob dataset will be released at https://github.com/1170632760/Spherical-GOF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08499v1">Improving Continual Learning for Gaussian Splatting based Environments Reconstruction on Commercial Off-the-Shelf Edge Devices</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) is increasingly relevant for edge robotics, where compact and incrementally updatable 3D scene models are needed for SLAM, navigation, and inspection under tight memory and latency budgets. Variational Bayesian Gaussian Splatting (VBGS) enables replay-free continual updates for the 3DGS algorithm by maintaining a probabilistic scene model, but its high-precision computations and large intermediate tensors make on-device training impractical. We present a precision-adaptive optimization framework that enables VBGS training on resource-constrained hardware without altering its variational formulation. We (i) profile VBGS to identify memory/latency hotspots, (ii) fuse memory-dominant kernels to reduce materialized intermediate tensors, and (iii) automatically assign operation-level precisions via a mixed-precision search with bounded relative error. Across the Blender, Habitat, and Replica datasets, our optimised pipeline reduces peak memory from 9.44 GB to 1.11 GB and training time from ~234 min to ~61 min on an A5000 GPU, while preserving (and in some cases improving) reconstruction quality of the state-of-the-art VBGS baseline. We also enable for the first time NVS training on a commercial embedded platform, the Jetson Orin Nano, reducing per-frame latency by 19x compared to 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24758v5">ExGS: Extreme 3D Gaussian Compression with Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      Neural scene representations, such as 3D Gaussian Splatting (3DGS), have enabled high-quality neural rendering; however, their large storage and transmission costs hinder deployment in resource-constrained environments. Existing compression methods either rely on costly optimization, which is slow and scene-specific, or adopt training-free pruning and quantization, which degrade rendering quality under high compression ratios. In contrast, recent data-driven approaches provide a promising direction to overcome this trade-off, enabling efficient compression while preserving high rendering quality. We introduce ExGS, a novel feed-forward framework that unifies Universal Gaussian Compression (UGC) with GaussPainter for Extreme 3DGS compression. UGC performs re-optimization-free pruning to aggressively reduce Gaussian primitives while retaining only essential information, whereas GaussPainter leverages powerful diffusion priors with mask-guided refinement to restore high-quality renderings from heavily pruned Gaussian scenes. Unlike conventional inpainting, GaussPainter not only fills in missing regions but also enhances visible pixels, yielding substantial improvements in degraded renderings. To ensure practicality, it adopts a lightweight VAE and a one-step diffusion design, enabling real-time restoration. Our framework can even achieve over 100X compression (reducing a typical 354.77 MB model to about 3.31 MB) while preserving fidelity and significantly improving image quality under challenging conditions. These results highlight the central role of diffusion priors in bridging the gap between extreme compression and high-quality neural rendering. Our code repository will be released at: https://github.com/chenttt2001/ExGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08313v1">HDR-NSFF: High Dynamic Range Neural Scene Flow Fields</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 ICLR 2026. Project page: https://shin-dong-yeon.github.io/HDR-NSFF/
    </div>
    <details class="paper-abstract">
      Radiance of real-world scenes typically spans a much wider dynamic range than what standard cameras can capture. While conventional HDR methods merge alternating-exposure frames, these approaches are inherently constrained to 2D pixel-level alignment, often leading to ghosting artifacts and temporal inconsistency in dynamic scenes. To address these limitations, we present HDR-NSFF, a paradigm shift from 2D-based merging to 4D spatio-temporal modeling. Our framework reconstructs dynamic HDR radiance fields from alternating-exposure monocular videos by representing the scene as a continuous function of space and time, and is compatible with both neural radiance field and 4D Gaussian Splatting (4DGS) based dynamic representations. This unified end-to-end pipeline explicitly models HDR radiance, 3D scene flow, geometry, and tone-mapping, ensuring physical plausibility and global coherence. We further enhance robustness by (i) extending semantic-based optical flow with DINO features to achieve exposure-invariant motion estimation, and (ii) incorporating a generative prior as a regularizer to compensate for limited observation in monocular captures and saturation-induced information loss. To evaluate HDR space-time view synthesis, we present the first real-world HDR-GoPro dataset specifically designed for dynamic HDR scenes. Experiments demonstrate that HDR-NSFF recovers fine radiance details and coherent dynamics even under challenging exposure variations, thereby achieving state-of-the-art performance in novel space-time view synthesis. Project page: https://shin-dong-yeon.github.io/HDR-NSFF/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14191v3">MCGS-SLAM: A Multi-Camera SLAM Framework Using Gaussian Splatting for High-Fidelity Mapping</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026
    </div>
    <details class="paper-abstract">
      Recent progress in dense SLAM has primarily targeted monocular setups, often at the expense of robustness and geometric coverage. We present MCGS-SLAM, the first purely RGB-based multi-camera SLAM system built on 3D Gaussian Splatting (3DGS). Unlike prior methods relying on sparse maps or inertial data, MCGS-SLAM fuses dense RGB inputs from multiple viewpoints into a unified, continuously optimized Gaussian map. A multi-camera bundle adjustment (MCBA) jointly refines poses and depths via dense photometric and geometric residuals, while a scale consistency module enforces metric alignment across views using low-rank priors. The system supports RGB input and maintains real-time performance at large scale. Experiments on synthetic and real-world datasets show that MCGS-SLAM consistently yields accurate trajectories and photorealistic reconstructions, usually outperforming monocular baselines. Notably, the wide field of view from multi-camera input enables reconstruction of side-view regions that monocular setups miss, critical for safe autonomous operation. These results highlight the promise of multi-camera Gaussian Splatting SLAM for high-fidelity mapping in robotics and autonomous driving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08254v1">DynamicVGGT: Learning Dynamic Point Maps for 4D Scene Reconstruction in Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction in autonomous driving remains a fundamental challenge due to significant temporal variations, moving objects, and complex scene dynamics. Existing feed-forward 3D models have demonstrated strong performance in static reconstruction but still struggle to capture dynamic motion. To address these limitations, we propose DynamicVGGT, a unified feed-forward framework that extends VGGT from static 3D perception to dynamic 4D reconstruction. Our goal is to model point motion within feed-forward 3D models in a dynamic and temporally coherent manner. To this end, we jointly predict the current and future point maps within a shared reference coordinate system, allowing the model to implicitly learn dynamic point representations through temporal correspondence. To efficiently capture temporal dependencies, we introduce a Motion-aware Temporal Attention (MTA) module that learns motion continuity. Furthermore, we design a Dynamic 3D Gaussian Splatting Head that explicitly models point motion by predicting Gaussian velocities using learnable motion tokens under scene flow supervision. It refines dynamic geometry through continuous 3D Gaussian optimization. Extensive experiments on autonomous driving datasets demonstrate that DynamicVGGT significantly outperforms existing methods in reconstruction accuracy, achieving robust feed-forward 4D dynamic scene reconstruction under complex driving scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.13911v3">PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      Despite advances in physics-based 3D motion synthesis, current methods face key limitations: reliance on pre-reconstructed 3D Gaussian Splatting (3DGS) built from dense multi-view images with time-consuming per-scene optimization; physics integration via either inflexible, hand-specified attributes or unstable, optimization-heavy guidance from video models using Score Distillation Sampling (SDS); and naive concatenation of prebuilt 3DGS with physics modules, which ignores physical information embedded in appearance and yields suboptimal performance. To address these issues, we propose PhysGM, a feed-forward framework that jointly predicts 3D Gaussian representation and physical properties from a single image, enabling immediate simulation and high-fidelity 4D rendering. Unlike slow appearance-agnostic optimization methods, we first pre-train a physics-aware reconstruction model that directly infers both Gaussian and physical parameters. We further refine the model with Direct Preference Optimization (DPO), aligning simulations with the physically plausible reference videos and avoiding the high-cost SDS optimization. To address the absence of a supporting dataset for this task, we propose PhysAssets, a dataset of 50K+ 3D assets annotated with physical properties and corresponding reference videos. Experiments show that PhysGM produces high-fidelity 4D simulations from a single image in one minute, achieving a significant speedup over prior work while delivering realistic renderings.Our project page is at:https://hihixiaolv.github.io/PhysGM.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.17635v3">LangSurf: Language-Embedded Surface Gaussians for 3D Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 \url{https://langsurf.github.io}
    </div>
    <details class="paper-abstract">
      Applying Gaussian Splatting to perception tasks for 3D scene understanding is becoming increasingly popular. Most existing works primarily focus on rendering 2D feature maps from novel viewpoints, which leads to an imprecise 3D language field with outlier languages, ultimately failing to align objects in 3D space. By utilizing masked images for feature extraction, these approaches also lack essential contextual information, leading to inaccurate feature representation. To this end, we propose a Language-Embedded Surface Field (LangSurf), which accurately aligns the 3D language fields with the surface of objects, facilitating precise 2D and 3D segmentation with text query, widely expanding the downstream tasks such as removal and editing. The core of LangSurf is a joint training strategy that flattens the language Gaussian on the object surfaces using geometry supervision and contrastive losses to assign accurate language features to the Gaussians of objects. In addition, we also introduce the Hierarchical-Context Awareness Module to extract features at the image level for contextual information then perform hierarchical mask pooling using masks segmented by SAM to obtain fine-grained language features in different hierarchies. Extensive experiments on open-vocabulary 2D and 3D semantic segmentation demonstrate that LangSurf outperforms the previous state-of-the-art method LangSplat by a large margin. As shown in Fig. 1, our method is capable of segmenting objects in 3D space, thus boosting the effectiveness of our approach in instance recognition, removal, and editing, which is also supported by comprehensive experiments. https://langsurf.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07664v1">Ref-DGS: Reflective Dual Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-08
      | 💬 Project page: https://straybirdflower.github.io/Ref-DGS/
    </div>
    <details class="paper-abstract">
      Reflective appearance, especially strong and typically near-field specular reflections, poses a fundamental challenge for accurate surface reconstruction and novel view synthesis. Existing Gaussian splatting methods either fail to model near-field specular reflections or rely on explicit ray tracing at substantial computational cost. We present Ref-DGS, a reflective dual Gaussian splatting framework that addresses this trade-off by decoupling surface reconstruction from specular reflection within an efficient rasterization-based pipeline. Ref-DGS introduces a dual Gaussian scene representation consisting of geometry Gaussians and complementary local reflection Gaussians that capture near-field specular interactions without explicit ray tracing, along with a global environment reflection field for modeling far-field specular reflections. To predict specular radiance, we further propose a lightweight, physically-aware adaptive mixing shader that fuses global and local reflection features. Experiments demonstrate that Ref-DGS achieves state-of-the-art performance on reflective scenes while training substantially faster than ray-based Gaussian methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07660v1">Holi-Spatial: Evolving Video Streams into Holistic 3D Spatial Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-03-08
      | 💬 project page: https://visionary-laboratory.github.io/holi-spatial/
    </div>
    <details class="paper-abstract">
      The pursuit of spatial intelligence fundamentally relies on access to large-scale, fine-grained 3D data. However, existing approaches predominantly construct spatial understanding benchmarks by generating question-answer (QA) pairs from a limited number of manually annotated datasets, rather than systematically annotating new large-scale 3D scenes from raw web data. As a result, their scalability is severely constrained, and model performance is further hindered by domain gaps inherent in these narrowly curated datasets. In this work, we propose Holi-Spatial, the first fully automated, large-scale, spatially-aware multimodal dataset, constructed from raw video inputs without human intervention, using the proposed data curation pipeline. Holi-Spatial supports multi-level spatial supervision, ranging from geometrically accurate 3D Gaussian Splatting (3DGS) reconstructions with rendered depth maps to object-level and relational semantic annotations, together with corresponding spatial Question-Answer (QA) pairs. Following a principled and systematic pipeline, we further construct Holi-Spatial-4M, the first large-scale, high-quality 3D semantic dataset, containing 12K optimized 3DGS scenes, 1.3M 2D masks, 320K 3D bounding boxes, 320K instance captions, 1.2M 3D grounding instances, and 1.2M spatial QA pairs spanning diverse geometric, relational, and semantic reasoning tasks. Holi-Spatial demonstrates exceptional performance in data curation quality, significantly outperforming existing feed-forward and per-scene optimized methods on datasets such as ScanNet, ScanNet++, and DL3DV. Furthermore, fine-tuning Vision-Language Models (VLMs) on spatial reasoning tasks using this dataset has also led to substantial improvements in model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07604v1">EmbedTalk: Triplane-Free Talking Head Synthesis using Embedding-Driven Gaussian Deformation</a></div>
    <div class="paper-meta">
      📅 2026-03-08
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Real-time talking head synthesis increasingly relies on deformable 3D Gaussian Splatting (3DGS) due to its low latency. Tri-planes are the standard choice for encoding Gaussians prior to deformation, since they provide a continuous domain with explicit spatial relationships. However, tri-plane representations are limited by grid resolution and approximation errors introduced by projecting 3D volumetric fields onto 2D subspaces. Recent work has shown the superiority of learnt embeddings for driving temporal deformations in 4D scene reconstruction. We introduce $\textbf{EmbedTalk}$, which shows how such embeddings can be leveraged for modelling speech deformations in talking head synthesis. Through comprehensive experiments, we show that EmbedTalk outperforms existing 3DGS-based methods in rendering quality, lip synchronisation, and motion consistency, while remaining competitive with state-of-the-art generative models. Moreover, replacing the tri-plane encoding with learnt embeddings enables significantly more compact models that achieve over 60 FPS on a mobile GPU (RTX 2060 6 GB). Our code will be placed in the public domain on acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07587v1">3DGS-HPC: Distractor-free 3D Gaussian Splatting with Hybrid Patch-wise Classification</a></div>
    <div class="paper-meta">
      📅 2026-03-08
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated remarkable performance in novel view synthesis and 3D scene reconstruction, yet its quality often degrades in real-world environments due to transient distractors, such as moving objects and varying shadows. Existing methods commonly rely on semantic cues extracted from pre-trained vision models to identify and suppress these distractors, but such semantics are misaligned with the binary distinction between static and transient regions and remain fragile under the appearance perturbations introduced during 3DGS optimization. We propose 3DGS-HPC, a framework that circumvents these limitations by combining two complementary principles: a patch-wise classification strategy that leverages local spatial consistency for robust region-level decisions, and a hybrid classification metric that adaptively integrates photometric and perceptual cues for more reliable separation. Extensive experiments demonstrate the superiority and robustness of our method in mitigating distractors to improve 3DGS-based novel view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07552v1">ReconDrive: Fast Feed-Forward 4D Gaussian Splatting for Autonomous Driving Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-08
    </div>
    <details class="paper-abstract">
      High-fidelity visual reconstruction and novel-view synthesis are essential for realistic closed-loop evaluation in autonomous driving. While 4D Gaussian Splatting (4DGS) offers a promising balance of accuracy and efficiency, existing per-scene optimization methods require costly iterative refinement, rendering them unscalable for extensive urban environments. Conversely, current feed-forward approaches often suffer from degraded photometric quality. To address these limitations, we propose ReconDrive, a feed-forward framework that leverages and extends the 3D foundation model VGGT for rapid, high-fidelity 4DGS generation. Our architecture introduces two core adaptations to tailor the foundation model to dynamic driving scenes: (1) Hybrid Gaussian Prediction Heads, which decouple the regression of spatial coordinates and appearance attributes to overcome the photometric deficiencies inherent in generalized foundation features; and (2) a Static-Dynamic 4D Composition strategy that explicitly captures temporal motion via velocity modeling to represent complex dynamic environments. Benchmarked on nuScenes, ReconDrive significantly outperforms existing feed-forward baselines in reconstruction, novel-view synthesis, and 3D perception. It achieves performance competitive with per-scene optimization while being orders of magnitude faster, providing a scalable and practical solution for realistic driving simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06830v3">MUGSQA: Novel Multi-Uncertainty-Based Gaussian Splatting Quality Assessment Method, Dataset, and Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-03-08
      | 💬 ICASSP 2026
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has recently emerged as a promising technique for 3D object reconstruction, delivering high-quality rendering results with significantly improved reconstruction speed. As variants continue to appear, assessing the perceptual quality of 3D objects reconstructed with different GS-based methods remains an open challenge. To address this issue, we first propose a unified multi-distance subjective quality assessment method that closely mimics human viewing behavior for objects reconstructed with GS-based methods in actual applications, thereby better collecting perceptual experiences. Based on it, we also construct a novel GS quality assessment dataset named MUGSQA, which is constructed considering multiple uncertainties of the input data. These uncertainties include the quantity and resolution of input views, the view distance, and the accuracy of the initial point cloud. Moreover, we construct two benchmarks: one to evaluate the robustness of various GS-based reconstruction methods under multiple uncertainties, and the other to evaluate the performance of existing quality assessment metrics. Our dataset and code are available at https://github.com/Solivition/MUGSQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.24118v2">LagMemo: Language 3D Gaussian Splatting Memory for Multi-modal Open-vocabulary Multi-goal Visual Navigation</a></div>
    <div class="paper-meta">
      📅 2026-03-08
    </div>
    <details class="paper-abstract">
      Navigating to a designated goal using visual information is a fundamental capability for intelligent robots. To address the practical demands of multi-modal, open-vocabulary goal queries and multi-goal visual navigation, we propose LagMemo, a navigation system that leverages a language 3D Gaussian Splatting memory. During a one-time exploration, LagMemo constructs a unified 3D language memory with robust spatial-semantic correlations. With incoming task goals, the system efficiently queries the memory, predicts candidate goal locations, and integrates a local perception-based verification mechanism to dynamically match and validate goals. For fair and rigorous evaluation, we curate GOAT-Core, a high-quality core split distilled from GOAT-Bench. Experimental results show that LagMemo's memory module enables effective multi-modal open-vocabulary localization, and significantly outperforms state-of-the-art methods in multi-goal visual navigation. Project page: https://weekgoodday.github.io/lagmemo
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.06968v2">3D Gaussian Splatting with Fisheye Images: Field of View Analysis and Depth-Based Initialization</a></div>
    <div class="paper-meta">
      📅 2026-03-07
      | 💬 VISAPP 2026 Accepted Camera Ready Version
    </div>
    <details class="paper-abstract">
      We present the first evaluation of 3D Gaussian Splatting methods on real fisheye imagery with fields of view above 180\textdegree{}. Our study evaluates Fisheye-GS \cite{liao2024fisheyegslightweightextensiblegaussian} and 3DGUT \cite{wu20253dgut} on indoor and outdoor scenes captured with 200\textdegree{} fisheye cameras, with the aim of assessing the practicality of wide-angle reconstruction under severe distortion. By comparing reconstructions at 200\textdegree{}, 160\textdegree{}, and 120\textdegree{} field-of-view, we show that both methods achieve their best results at 160\textdegree{}, which balances scene coverage with image quality, while distortion at 200\textdegree{} degrades performance. To address the common failure of Structure-from-Motion (SfM) initialization at such wide angles, we introduce a depth-based alternative using UniK3D (Universal Camera Monocular 3D Estimation) \cite{piccinelli2025unik3d}. This represents the first application of UniK3D to fisheye imagery beyond 200\textdegree{}, despite the model not being trained on such data. With the number of predicted points controlled to match SfM for fairness, UniK3D produces geometrically accurate reconstructions that rival or surpass SfM, even in challenging scenes with fog, glare, or open sky. These results demonstrate the feasibility of fisheye-based 3D Gaussian Splatting and provides a benchmark for future research on wide-angle reconstruction from sparse and distorted inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.22276v2">GS-2M: Material-aware Gaussian Splatting for High-fidelity Mesh Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-07
      | 💬 This is the author's version of a paper accepted to Eurographics 2026, to appear in Computer Graphics Form. The final version will be available via Wiley
    </div>
    <details class="paper-abstract">
      We propose a material-aware optimization framework for high-fidelity mesh reconstruction from multi-view images based on 3D Gaussian Splatting, referred to as GS-2M. Previous works handle these tasks separately and struggle to reconstruct highly reflective surfaces, often relying on priors from external models to enhance the decomposition results. Conversely, our method addresses these two problems by jointly optimizing attributes relevant to the quality of rendered depth and normals, maintaining geometric details while being resilient to reflective surfaces. Although contemporary works effectively solve these tasks together, they often employ sophisticated neural components to learn scene properties, which hinders their performance at scale. To further eliminate these neural components, we propose a novel roughness supervision strategy based on multi-view photometric variation. When combined with a carefully designed loss and optimization process, our unified framework produces reconstruction results comparable to state-of-the-art methods, delivering accurate triangle meshes even for reflective surfaces. We validate the effectiveness of our approach with widely used datasets from previous works and qualitative comparisons with state-of-the-art surface reconstruction methods. Project page: https://ndming.github.io/publications/gs2m/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19210v2">MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-07
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Recent advances in dynamic scene reconstruction have significantly benefited from 3D Gaussian Splatting, yet existing methods show inconsistent performance across diverse scenes, indicating no single approach effectively handles all dynamic challenges. To overcome these limitations, we propose Mixture of Experts for Dynamic Gaussian Splatting (MoE-GS), a unified framework integrating multiple specialized experts via a novel Volume-aware Pixel Router. Unlike sparsity-oriented MoE architectures in large language models, MoE-GS is designed to improve dynamic novel view synthesis quality by combining heterogeneous deformation priors, rather than to reduce training or inference-time FLOPs. Our router adaptively blends expert outputs by projecting volumetric Gaussian-level weights into pixel space through differentiable weight splatting, ensuring spatially and temporally coherent results. Although MoE-GS improves rendering quality, the increased model capacity and reduced FPS are inherent to the MoE architecture. To mitigate this, we explore two complementary directions: (1) single-pass multi-expert rendering and gate-aware Gaussian pruning, which improve efficiency within the MoE framework, and (2) a distillation strategy that transfers MoE performance to individual experts, enabling lightweight deployment without architectural changes. To the best of our knowledge, MoE-GS is the first approach incorporating Mixture-of-Experts techniques into dynamic Gaussian splatting. Extensive experiments on the N3V and Technicolor datasets demonstrate that MoE-GS consistently outperforms state-of-the-art methods with improved efficiency. Video demonstrations are available at cvsp-lab.github.io/MoE-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.06989v1">MipSLAM: Alias-Free Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2026-03-07
      | 💬 Accepted to ICRA 2026
    </div>
    <details class="paper-abstract">
      This paper introduces MipSLAM, a frequency-aware 3D Gaussian Splatting (3DGS) SLAM framework capable of high-fidelity anti-aliased novel view synthesis and robust pose estimation under varying camera configurations. Existing 3DGS-based SLAM systems often suffer from aliasing artifacts and trajectory drift due to inadequate filtering and purely spatial optimization. To overcome these limitations, we propose an Elliptical Adaptive Anti-aliasing (EAA) algorithm that approximates Gaussian contributions via geometry-aware numerical integration, avoiding costly analytic computation. Furthermore, we present a Spectral-Aware Pose Graph Optimization (SA-PGO) module that reformulates trajectory estimation in the frequency domain, effectively suppressing high-frequency noise and drift through graph Laplacian analysis. A novel local frequency-domain perceptual loss is also introduced to enhance fine-grained geometric detail recovery. Extensive evaluations on Replica and TUM datasets demonstrate that MipSLAM achieves state-of-the-art rendering quality and localization accuracy across multiple resolutions while maintaining real-time capability. Code is available at https://github.com/yzli1998/MipSLAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23040v2">PackUV: Packed Gaussian UV Maps for 4D Volumetric Video</a></div>
    <div class="paper-meta">
      📅 2026-03-06
      | 💬 https://ivl.cs.brown.edu/packuv
    </div>
    <details class="paper-abstract">
      Volumetric videos offer immersive 4D experiences, but remain difficult to reconstruct, store, and stream at scale. Existing Gaussian Splatting based methods achieve high-quality reconstruction but break down on long sequences, temporal inconsistency, and fail under large motions and disocclusions. Moreover, their outputs are typically incompatible with conventional video coding pipelines, preventing practical applications. We introduce PackUV, a novel 4D Gaussian representation that maps all Gaussian attributes into a sequence of structured, multi-scale UV atlas, enabling compact, image-native storage. To fit this representation from multi-view videos, we propose PackUV-GS, a temporally consistent fitting method that directly optimizes Gaussian parameters in the UV domain. A flow-guided Gaussian labeling and video keyframing module identifies dynamic Gaussians, stabilizes static regions, and preserves temporal coherence even under large motions and disocclusions. The resulting UV atlas format is the first unified volumetric video representation compatible with standard video codecs (e.g., FFV1) without losing quality, enabling efficient streaming within existing multimedia infrastructure. To evaluate long-duration volumetric capture, we present PackUV-2B, the largest multi-view video dataset to date, featuring more than 50 synchronized cameras, substantial motion, and frequent disocclusions across 100 sequences and 2B (billion) frames. Extensive experiments demonstrate that our method surpasses existing baselines in rendering fidelity while scaling to sequences up to 30 minutes with consistent quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19255v2">Advances in 4D Representation: Geometry, Motion, and Interaction</a></div>
    <div class="paper-meta">
      📅 2026-03-06
      | 💬 21 pages. Project Page: https://mingrui-zhao.github.io/4DRep-GMI/
    </div>
    <details class="paper-abstract">
      We present a survey on 4D generation and reconstruction, a fast-evolving subfield of computer graphics whose developments have been propelled by recent advances in neural fields, geometric and motion deep learning, as well as 3D generative artificial intelligence (GenAI). While our survey is not the first of its kind, we build our coverage of the domain from a unique and distinctive perspective of 4D representations, to model 3D geometry evolving over time while exhibiting motion and interaction. Specifically, instead of offering an exhaustive enumeration of many works, we take a more selective approach by focusing on representative works to highlight both the desirable properties and ensuing challenges of each representation under different computation, application, and data scenarios. The main take-away message we aim to convey to the readers is on how to select and then customize the appropriate 4D representations for their tasks. Organizationally, we separate the 4D representations based on three key pillars: geometry, motion, and interaction. Our discourse will not only encompass the most popular representations of today, such as neural radiance fields (NeRFs) and 3D Gaussian Splatting (3DGS), but also bring attention to relatively under-explored representations in the 4D context, such as structured models and long-range motions. Throughout our survey, we will reprise the role of large language models (LLMs) and video foundational models (VFMs) in a variety of 4D applications, while steering our discussion towards their current limitations and how they can be addressed. We also provide a dedicated coverage on what 4D datasets are currently available, as well as what is lacking, in driving the subfield forward. Project page:https://mingrui-zhao.github.io/4DRep-GMI/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.06860v1">ColonSplat: Reconstruction of Peristaltic Motion in Colonoscopy with Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-06
    </div>
    <details class="paper-abstract">
      Accurate 3D reconstruction of colonoscopy data, accounting for complex peristaltic movements, is crucial for advanced surgical navigation and retrospective diagnostics. While recent novel view synthesis and 3D reconstruction methods have demonstrated remarkable success in general endoscopic scenarios, they struggle in the highly constrained environment of the colon. Due to the limited field of view of a camera moving through an actively deforming tubular structure, existing endoscopic methods reconstruct the colon appearance only for initial camera trajectory. However, the underlying anatomy remains largely static; instead of updating Gaussians' spatial coordinates (xyz), these methods encode deformation through either rotation, scale or opacity adjustments. In this paper, we first present a benchmark analysis of state-of-the-art dynamic endoscopic methods for realistic colonoscopic scenes, showing that they fail to model true anatomical motion. To enable rigorous evaluation of global reconstruction quality, we introduce DynamicColon, a synthetic dataset with ground-truth point clouds at every timestep. Building on these insights, we propose ColonSplat, a dynamic Gaussian Splatting framework that captures peristaltic-like motion while preserving global geometric consistency, achieving superior geometric fidelity on C3VDv2 and DynamicColon datasets. Project page: https://wmito.github.io/ColonSplat
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.06852v1">Active View Selection with Perturbed Gaussian Ensemble for Tomographic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-06
    </div>
    <details class="paper-abstract">
      Sparse-view computed tomography (CT) is critical for reducing radiation exposure to patients. Recent advances in radiative 3D Gaussian Splatting (3DGS) have enabled fast and accurate sparse-view CT reconstruction. Despite these algorithmic advancements, practical reconstruction fidelity remains fundamentally bounded by the quality of the captured data, raising the crucial yet underexplored problem of X-ray active view selection. Existing active view selection methods are primarily designed for natural-light scenes and fail to capture the unique geometric ambiguities and physical attenuation properties inherent in X-ray imaging. In this paper, we present Perturbed Gaussian Ensemble, an active view selection framework that integrates uncertainty modeling with sequential decision-making, tailored for X-ray Gaussian Splatting. Specifically, we identify low-density Gaussian primitives that are likely to be uncertain and apply stochastic density scaling to construct an ensemble of plausible Gaussian density fields. For each candidate projection, we measure the structural variance of the ensemble predictions and select the one with the highest variance as the next best view. Extensive experimental results on arbitrary-trajectory CT benchmarks demonstrate that our density-guided perturbation strategy effectively eliminates geometric artifacts and consistently outperforms existing baselines in progressive tomographic reconstruction under unified view selection protocols.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.18923v2">Gaussian Set Surface Reconstruction through Per-Gaussian Optimization</a></div>
    <div class="paper-meta">
      📅 2026-03-06
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) effectively synthesizes novel views through its flexible representation, yet fails to accurately reconstruct scene geometry. While modern variants like PGSR introduce additional losses to ensure proper depth and normal maps through Gaussian fusion, they still neglect individual placement optimization. This results in unevenly distributed Gaussians that deviate from the latent surface, complicating both reconstruction refinement and scene editing. Motivated by pioneering work on Point Set Surfaces, we propose Gaussian Set Surface Reconstruction (GSSR), a method designed to distribute Gaussians evenly along the latent surface while aligning their dominant normals with the surface normal. GSSR enforces fine-grained geometric alignment through a combination of pixel-level and Gaussian-level single-view normal consistency and multi-view photometric consistency, optimizing both local and global perspectives. To further refine the representation, we introduce an opacity regularization loss to eliminate redundant Gaussians and apply periodic depth- and normal-guided Gaussian reinitialization for a cleaner, more uniform spatial distribution. Our reconstruction results demonstrate significantly improved geometric precision in Gaussian placement, enabling intuitive scene editing and efficient generation of novel Gaussian-based 3D environments. Extensive experiments validate GSSR's effectiveness, showing enhanced geometric accuracy while preserving high-quality rendering performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.06216v1">EntON: Eigenentropy-Optimized Neighborhood Densification in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-06
      | 💬 Submitted to ISPRS Journal of Photogrammetry and Remote Sensing on 20 February 2026
    </div>
    <details class="paper-abstract">
      We present a novel Eigenentropy-optimized neighboorhood densification strategy EntON in 3D Gaussian Splatting (3DGS) for geometrically accurate and high-quality rendered 3D reconstruction. While standard 3DGS produces Gaussians whose centers and surfaces are poorly aligned with the underlying object geometry, surface-focused reconstruction methods frequently sacrifice photometric accuracy. In contrast to the conventional densification strategy, which relies on the magnitude of the view-space position gradient, our approach introduces a geometry-aware strategy to guide adaptive splitting and pruning. Specifically, we compute the 3D shape feature Eigenentropy from the eigenvalues of the covariance matrix in the k-nearest neighborhood of each Gaussian center, which quantifies the local structural order. These Eigenentropy values are integrated into an alternating optimization framework: During the optimization process, the algorithm alternates between (i) standard gradient-based densification, which refines regions via view-space gradients, and (ii) Eigenentropy-aware densification, which preferentially densifies Gaussians in low-Eigenentropy (ordered, flat) neighborhoods to better capture fine geometric details on the object surface, and prunes those in high-Eigenentropy (disordered, spherical) regions. We provide quantitative and qualitative evaluations on two benchmark datasets: small-scale DTU dataset and large-scale TUM2TWIN dataset, covering man-made objects and urban scenes. Experiments demonstrate that our Eigenentropy-aware alternating densification strategy improves geometric accuracy by up to 33% and rendering quality by up to 7%, while reducing the number of Gaussians by up to 50% and training time by up to 23%. Overall, EnTON achieves a favorable balance between geometric accuracy, rendering quality and efficiency by avoiding unnecessary scene expansion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.06210v1">VG3S: Visual Geometry Grounded Gaussian Splatting for Semantic Occupancy Prediction</a></div>
    <div class="paper-meta">
      📅 2026-03-06
    </div>
    <details class="paper-abstract">
      3D semantic occupancy prediction has become a crucial perception task for comprehensive scene understanding in autonomous driving. While recent advances have explored 3D Gaussian splatting for occupancy modeling to substantially reduce computational overhead, the generation of high-quality 3D Gaussians relies heavily on accurate geometric cues, which are often insufficient in purely vision-centric paradigms. To bridge this gap, we advocate for injecting the strong geometric grounding capability from Vision Foundation Models (VFMs) into occupancy prediction. In this regard, we introduce Visual Geometry Grounded Gaussian Splatting (VG3S), a novel framework that empowers Gaussian-based occupancy prediction with cross-view 3D geometric grounding. Specifically, to fully exploit the rich 3D geometric priors from a frozen VFM, we propose a plug-and-play hierarchical geometric feature adapter, which can effectively transform generic VFM tokens via feature aggregation, task-specific alignment, and multi-scale restructuring. Extensive experiments on the nuScenes occupancy benchmark demonstrate that VG3S achieves remarkable improvements of 12.6% in IoU and 7.5% in mIoU over the baseline. Furthermore, we show that VG3S generalizes seamlessly across diverse VFMs, consistently enhancing occupancy prediction accuracy and firmly underscoring the immense value of integrating priors derived from powerful, pre-trained geometry-grounded VFMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.04843v5">PoI: A Filter to Extract Pixel of Interest from Novel Views for Scene Coordinate Regression</a></div>
    <div class="paper-meta">
      📅 2026-03-06
    </div>
    <details class="paper-abstract">
      Neural View Synthesis (NVS) techniques such as NeRF and 3D Gaussian Splatting (3DGS) have enabled photorealistic rendering from novel viewpoints and are increasingly used to augment training data for visual localization. However, these methods fundamentally rely on observed geometry and radiance; they interpolate existing information but cannot hallucinate unseen 3D structures or recover missing content under sparse or extreme viewpoints. As a result, rendered views often exhibit blur, structural distortion, or incomplete geometry. While such imperfections may be tolerated by Camera Pose Regression (CPR) methods, they severely degrade Scene Coordinate Regression (SCR), which requires accurate per-pixel 3D supervision. To address this limitation, we introduce PoI (Pixel-of-Interest), a framework that enables effective NVS augmentation for SCR-based localization. We first employ 3DGS to render novel views and leverage a single-step diffusion model to refine them, allowing the synthesis of structurally plausible details beyond purely geometry-driven interpolation. However, even diffusion-refined views may contain unreliable pixels. Therefore, we propose a progressive pixel-level filtering strategy based on reprojection error to selectively retain trustworthy synthetic pixels during training while suppressing harmful ones. Extensive experiments on 7Scenes and Cambridge Landmarks demonstrate that our method consistently improves localization accuracy over strong SCR baselines and achieves state-of-the-art performance with competitive training efficiency. Our results reveal that, for SCR, the benefit of novel view augmentation depends not only on generative realism but also on explicit control of pixel-level reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.06061v1">Transforming Omnidirectional RGB-LiDAR data into 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-06
      | 💬 This work has been submitted to the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) for possible publication
    </div>
    <details class="paper-abstract">
      The demand for large-scale digital twins is rapidly growing in robotics and autonomous driving. However, constructing these environments with 3D Gaussian Splatting (3DGS) usually requires expensive, purpose-built data collection. Meanwhile, deployed platforms routinely collect extensive omnidirectional RGB and LiDAR logs, but a significant portion of these sensor data is directly discarded or strictly underutilized due to transmission constraints and the lack of scalable reuse pipeline. In this paper, we present an omnidirectional RGB-LiDAR reuse pipeline that transforms these archived logs into robust initialization assets for 3DGS. Direct conversion of such raw logs introduces practical bottlenecks: inherent non-linear distortion leads to unreliable Structure-from-Motion (SfM) tracking, and dense, unorganized LiDAR clouds cause computational overhead during 3DGS optimization. To overcome these challenges, our pipeline strategically integrates an ERP-to-cubemap conversion module for deterministic spatial anchoring, alongside PRISM-a color stratified downsampling strategy. By bridging these multi-modal inputs via Fast Point Feature Histograms (FPFH) based global registration and Iterative Closest Point (ICP), our pipeline successfully repurposes a considerable fraction of discarded data into usable SfM geometry. Furthermore, our LiDAR-reinforced initialization consistently enhances the final 3DGS rendering fidelity in structurally complex scenes compared to vision-only baselines. Ultimately, this work provides a deterministic workflow for creating simulation-grade digital twins from standard archived sensor logs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15355v2">DAV-GSWT: Diffusion-Active-View Sampling for Data-Efficient Gaussian Splatting Wang Tiles</a></div>
    <div class="paper-meta">
      📅 2026-03-06
      | 💬 16 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The emergence of 3D Gaussian Splatting has fundamentally redefined the capabilities of photorealistic neural rendering by enabling high-throughput synthesis of complex environments. While procedural methods like Wang Tiles have recently been integrated to facilitate the generation of expansive landscapes, these systems typically remain constrained by a reliance on densely sampled exemplar reconstructions. We present DAV-GSWT, a data-efficient framework that leverages diffusion priors and active view sampling to synthesize high-fidelity Gaussian Splatting Wang Tiles from minimal input observations. By integrating a hierarchical uncertainty quantification mechanism with generative diffusion models, our approach autonomously identifies the most informative viewpoints while hallucinating missing structural details to ensure seamless tile transitions. Experimental results indicate that our system significantly reduces the required data volume while maintaining the visual integrity and interactive performance necessary for large-scale virtual environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05932v1">FTSplat: Feed-forward Triangle Splatting Network</a></div>
    <div class="paper-meta">
      📅 2026-03-06
    </div>
    <details class="paper-abstract">
      High-fidelity three-dimensional (3D) reconstruction is essential for robotics and simulation. While Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) achieve impressive rendering quality, their reliance on time-consuming per-scene optimization limits real-time deployment. Emerging feed-forward Gaussian splatting methods improve efficiency but often lack explicit, manifold geometry required for direct simulation. To address these limitations, we propose a feed-forward framework for triangle primitive generation that directly predicts continuous triangle surfaces from calibrated multi-view images. Our method produces simulation-ready models in a single forward pass, obviating the need for per-scene optimization or post-processing. We introduce a pixel-aligned triangle generation module and incorporate relative 3D point cloud supervision to enhance geometric learning stability and consistency. Experiments demonstrate that our method achieves efficient reconstruction while maintaining seamless compatibility with standard graphics and robotic simulators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05882v1">CylinderSplat: 3D Gaussian Splatting with Cylindrical Triplanes for Panoramic Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-03-06
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) has shown great promise for real-time novel view synthesis, but its application to panoramic imagery remains challenging. Existing methods often rely on multi-view cost volumes for geometric refinement, which struggle to resolve occlusions in sparse-view scenarios. Furthermore, standard volumetric representations like Cartesian Triplanes are poor in capturing the inherent geometry of $360^\circ$ scenes, leading to distortion and aliasing. In this work, we introduce CylinderSplat, a feed-forward framework for panoramic 3DGS that addresses these limitations. The core of our method is a new {cylindrical Triplane} representation, which is better aligned with panoramic data and real-world structures adhering to the Manhattan-world assumption. We use a dual-branch architecture: a pixel-based branch reconstructs well-observed regions, while a volume-based branch leverages the cylindrical Triplane to complete occluded or sparsely-viewed areas. Our framework is designed to flexibly handle a variable number of input views, from single to multiple panoramas. Extensive experiments demonstrate that CylinderSplat achieves state-of-the-art results in both single-view and multi-view panoramic novel view synthesis, outperforming previous methods in both reconstruction quality and geometric accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11689v2">Phys2Real: Fusing VLM Priors with Interactive Online Adaptation for Uncertainty-Aware Sim-to-Real Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-03-06
      | 💬 Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026
    </div>
    <details class="paper-abstract">
      Learning robotic manipulation policies directly in the real world can be expensive and time-consuming. While reinforcement learning (RL) policies trained in simulation present a scalable alternative, effective sim-to-real transfer remains challenging, particularly for tasks that require precise dynamics. To address this, we propose Phys2Real, a real-to-sim-to-real RL pipeline that combines vision-language model (VLM)-inferred physical parameter estimates with interactive adaptation through uncertainty-aware fusion. Our approach consists of three core components: (1) high-fidelity geometric reconstruction with 3D Gaussian splatting, (2) VLM-inferred prior distributions over physical parameters, and (3) online physical parameter estimation from interaction data. Phys2Real conditions policies on interpretable physical parameters, refining VLM predictions with online estimates via ensemble-based uncertainty quantification. On planar pushing tasks of a T-block with varying center of mass (CoM) and a hammer with an off-center mass distribution, Phys2Real achieves substantial improvements over a domain randomization baseline: 100% vs 79% success rate for the bottom-weighted T-block, 57% vs 23% in the challenging top-weighted T-block, and 15% faster average task completion for hammer pushing. Ablation studies indicate that the combination of VLM and interaction information is essential for success. Project website: https://phys2real.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05152v1">SSR-GS: Separating Specular Reflection in Gaussian Splatting for Glossy Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Project page: https://gsflyer.github.io/SSR-GS/
    </div>
    <details class="paper-abstract">
      In recent years, 3D Gaussian splatting (3DGS) has achieved remarkable progress in novel view synthesis. However, accurately reconstructing glossy surfaces under complex illumination remains challenging, particularly in scenes with strong specular reflections and multi-surface interreflections. To address this issue, we propose SSR-GS, a specular reflection modeling framework for glossy surface reconstruction. Specifically, we introduce a prefiltered Mip-Cubemap to model direct specular reflections efficiently, and propose an IndiASG module to capture indirect specular reflections. Furthermore, we design Visual Geometry Priors (VGP) that couple a reflection-aware visual prior via a reflection score (RS) to downweight the photometric loss contribution of reflection-dominated regions, with geometry priors derived from VGGT, including progressively decayed depth supervision and transformed normal constraints. Extensive experiments on both synthetic and real-world datasets demonstrate that SSR-GS achieves state-of-the-art performance in glossy surface reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05108v1">GaussTwin: Unified Simulation and Correction with Gaussian Splatting for Robotic Digital Twins</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 8 pages, 4 figures, 3 tables, ICRA 2026
    </div>
    <details class="paper-abstract">
      Digital twins promise to enhance robotic manipulation by maintaining a consistent link between real-world perception and simulation. However, most existing systems struggle with the lack of a unified model, complex dynamic interactions, and the real-to-sim gap, which limits downstream applications such as model predictive control. Thus, we propose GaussTwin, a real-time digital twin that combines position-based dynamics with discrete Cosserat rod formulations for physically grounded simulation, and Gaussian splatting for efficient rendering and visual correction. By anchoring Gaussians to physical primitives and enforcing coherent SE(3) updates driven by photometric error and segmentation masks, GaussTwin achieves stable prediction-correction while preserving physical fidelity. Through experiments in both simulation and on a Franka Research 3 platform, we show that GaussTwin consistently improves tracking accuracy and robustness compared to shape-matching and rigid-only baselines, while also enabling downstream tasks such as push-based planning. These results highlight GaussTwin as a step toward unified, physically meaningful digital twins that can support closed-loop robotic interaction and learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.24096v2">DiffusionHarmonizer: Bridging Neural Reconstruction and Photorealistic Simulation with Online Diffusion Enhancer</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 For more details and updates, please visit our project website: https://research.nvidia.com/labs/sil/projects/diffusion-harmonizer
    </div>
    <details class="paper-abstract">
      Simulation is essential to the development and evaluation of autonomous robots such as self-driving vehicles. Neural reconstruction is emerging as a promising solution as it enables simulating a wide variety of scenarios from real-world data alone in an automated and scalable way. However, while methods such as NeRF and 3D Gaussian Splatting can produce visually compelling results, they often exhibit artifacts particularly when rendering novel views, and fail to realistically integrate inserted dynamic objects, especially when they were captured from different scenes. To overcome these limitations, we introduce DiffusionHarmonizer, an online generative enhancement framework that transforms renderings from such imperfect scenes into temporally consistent outputs while improving their realism. At its core is a single-step temporally-conditioned enhancer that is converted from a pretrained multi-step image diffusion model, capable of running in online simulators on a single GPU. The key to training it effectively is a custom data curation pipeline that constructs synthetic-real pairs emphasizing appearance harmonization, artifact correction, and lighting realism. The result is a scalable system that significantly elevates simulation fidelity in both research and production environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13454v2">Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 ICLR 2026 (Oral), Project page: https://gohyojun15.github.io/VIST3A/
    </div>
    <details class="paper-abstract">
      The rapid progress of large, pretrained models for both visual content generation and 3D reconstruction opens up new possibilities for text-to-3D generation. Intuitively, one could obtain a formidable 3D scene generator if one were able to combine the power of a modern latent text-to-video model as "generator" with the geometric abilities of a recent (feedforward) 3D reconstruction system as "decoder". We introduce VIST3A, a general framework that does just that, addressing two main challenges. First, the two components must be joined in a way that preserves the rich knowledge encoded in their weights. We revisit model stitching, i.e., we identify the layer in the 3D decoder that best matches the latent representation produced by the text-to-video generator and stitch the two parts together. That operation requires only a small dataset and no labels. Second, the text-to-video generator must be aligned with the stitched 3D decoder, to ensure that the generated latents are decodable into consistent, perceptually convincing 3D scene geometry. To that end, we adapt direct reward finetuning, a popular technique for human preference alignment. We evaluate the proposed VIST3A approach with different video generators and 3D reconstruction models. All tested pairings markedly improve over prior text-to-3D models that output Gaussian splats. Moreover, by choosing a suitable 3D base model, VIST3A also enables high-quality text-to-pointmap generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19854v3">STAvatar: Soft Binding and Temporal Density Control for Monocular 3D Head Avatars Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Accepted to CVPR 2026. Project page: https://jiankuozhao.github.io/STAvatar/
    </div>
    <details class="paper-abstract">
      Reconstructing high-fidelity and animatable 3D head avatars from monocular videos remains a challenging yet essential task. Existing methods based on 3D Gaussian Splatting typically bind Gaussians to mesh triangles and model deformations solely via Linear Blend Skinning, which results in rigid motion and limited expressiveness. Moreover, they lack specialized strategies to handle frequently occluded regions (e.g., mouth interiors, eyelids). To address these limitations, we propose STAvatar, which consists of two key components: (1) a UV-Adaptive Soft Binding framework that leverages both image-based and geometric priors to learn per-Gaussian feature offsets within the UV space. This UV representation supports dynamic resampling, ensuring full compatibility with Adaptive Density Control (ADC) and enhanced adaptability to shape and textural variations. (2) a Temporal ADC strategy, which first clusters structurally similar frames to facilitate more targeted computation of the densification criterion. It further introduces a novel fused perceptual error as clone criterion to jointly capture geometric and textural discrepancies, encouraging densification in regions requiring finer details. Extensive experiments on four benchmark datasets demonstrate that STAvatar achieves state-of-the-art reconstruction performance, especially in capturing fine-grained details and reconstructing frequently occluded regions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04847v1">GloSplat: Joint Pose-Appearance Optimization for Faster and More Accurate 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      Feature extraction, matching, structure from motion (SfM), and novel view synthesis (NVS) have traditionally been treated as separate problems with independent optimization objectives. We present GloSplat, a framework that performs \emph{joint pose-appearance optimization} during 3D Gaussian Splatting training. Unlike prior joint optimization methods (BARF, NeRF--, 3RGS) that rely purely on photometric gradients for pose refinement, GloSplat preserves \emph{explicit SfM feature tracks} as first-class entities throughout training: track 3D points are maintained as separate optimizable parameters from Gaussian primitives, providing persistent geometric anchors via a reprojection loss that operates alongside photometric supervision. This architectural choice prevents early-stage pose drift while enabling fine-grained refinement -- a capability absent in photometric-only approaches. We introduce two pipeline variants: (1) \textbf{GloSplat-F}, a COLMAP-free variant using retrieval-based pair selection for efficient reconstruction, and (2) \textbf{GloSplat-A}, an exhaustive matching variant for maximum quality. Both employ global SfM initialization followed by joint photometric-geometric optimization during 3DGS training. Experiments demonstrate that GloSplat-F achieves state-of-the-art among COLMAP-free methods while GloSplat-A surpasses all COLMAP-based baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.24290v2">UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 ICLR 2026, Project page: https://ufo-4d.github.io/
    </div>
    <details class="paper-abstract">
      Dense 4D reconstruction from unposed images remains a critical challenge, with current methods relying on slow test-time optimization or fragmented, task-specific feedforward models. We introduce UFO-4D, a unified feedforward framework to reconstruct a dense, explicit 4D representation from just a pair of unposed images. UFO-4D directly estimates dynamic 3D Gaussian Splats, enabling the joint and consistent estimation of 3D geometry, 3D motion, and camera pose in a feedforward manner. Our core insight is that differentiably rendering multiple signals from a single Dynamic 3D Gaussian representation offers major training advantages. This approach enables a self-supervised image synthesis loss while tightly coupling appearance, depth, and motion. Since all modalities share the same geometric primitives, supervising one inherently regularizes and improves the others. This synergy overcomes data scarcity, allowing UFO-4D to outperform prior work by up to 3 times in joint geometry, motion, and camera pose estimation. Our representation also enables high-fidelity 4D interpolation across novel views and time. Please visit our project page for visual results: https://ufo-4d.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04770v1">DSA-SRGS: Super-Resolution Gaussian Splatting for Dynamic Sparse-View DSA Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 11 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Digital subtraction angiography (DSA) is a key imaging technique for the auxiliary diagnosis and treatment of cerebrovascular diseases. Recent advancements in gaussian splatting and dynamic neural representations have enabled robust 3D vessel reconstruction from sparse dynamic inputs. However, these methods are fundamentally constrained by the resolution of input projections, where performing naive upsampling to enhance rendering resolution inevitably results in severe blurring and aliasing artifacts. Such lack of super-resolution capability prevents the reconstructed 4D models from recovering fine-grained vascular details and intricate branching structures, which restricts their application in precision diagnosis and treatment. To solve this problem, this paper proposes DSA-SRGS, the first super-resolution gaussian splatting framework for dynamic sparse-view DSA reconstruction. Specifically, we introduce a Multi-Fidelity Texture Learning Module that integrates high-quality priors from a fine-tuned DSA-specific super-resolution model, into the 4D reconstruction optimization. To mitigate potential hallucination artifacts from pseudo-labels, this module employs a Confidence-Aware Strategy to adaptively weight supervision signals between the original low-resolution projections and the generated high-resolution pseudo-labels. Furthermore, we develop Radiative Sub-Pixel Densification, an adaptive strategy that leverages gradient accumulation from high-resolution sub-pixel sampling to refine the 4D radiative gaussian kernels. Extensive experiments on two clinical DSA datasets demonstrate that DSA-SRGS significantly outperforms state-of-the-art methods in both quantitative metrics and qualitative visual fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18140v2">Observer-Actor: Active Vision Imitation Learning with Sparse-View Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 Accepted at ICRA 2026. Project Webpage: https://obact.github.io
    </div>
    <details class="paper-abstract">
      We propose Observer Actor (ObAct), a novel framework for active vision imitation learning in which the observer moves to optimal visual observations for the actor. We study ObAct on a dual-arm robotic system equipped with wrist-mounted cameras. At test time, ObAct dynamically assigns observer and actor roles: the observer arm constructs a 3D Gaussian Splatting (3DGS) representation from three images, virtually explores this to find an optimal camera pose, then moves to this pose; the actor arm then executes a policy using the observer's observations. This formulation enhances the clarity and visibility of both the object and the gripper in the policy's observations. As a result, we enable the training of ambidextrous policies on observations that remain closer to the occlusion-free training distribution, leading to more robust policies. We study this formulation with two existing imitation learning methods -- trajectory transfer and behavior cloning -- and experiments show that ObAct significantly outperforms static-camera setups: trajectory transfer improves by 145% without occlusion and 233% with occlusion, while behavior cloning improves by 75% and 143%, respectively. Videos are available at https://obact.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02887v2">Generalized non-exponential Gaussian splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 13 pages, 6 figures, 4 tables
    </div>
    <details class="paper-abstract">
      In this work we generalize 3D Gaussian splatting (3DGS) to a wider family of physically-based alpha-blending operators. 3DGS has become the standard de-facto for radiance field rendering and reconstruction, given its flexibility and efficiency. At its core, it is based on alpha-blending sorted semitransparent primitives, which in the limit converges to the classic radiative transfer function with exponential transmittance. Inspired by recent research on non-exponential radiative transfer, we generalize the image formation model of 3DGS to non-exponential regimes. Based on this generalization, we use a quadratic transmittance to define sub-linear, linear, and super-linear versions of 3DGS, which exhibit faster-than-exponential decay. We demonstrate that these new non-exponential variants achieve similar quality than the original 3DGS but significantly reduce the number of overdraws, which result on speed-ups of up to $4\times$ in complex real-world captures, on a ray-tracing-based renderer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02134v2">OnlineX: Unified Online 3D Reconstruction and Understanding with Active-to-Stable State Evolution</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 Accepted by CVPR Finding 2026 (Project page: https://xiac20.github.io/OnlineX/)
    </div>
    <details class="paper-abstract">
      Recent advances in generalizable 3D Gaussian Splatting (3DGS) have enabled rapid 3D scene reconstruction within seconds, eliminating the need for per-scene optimization. However, existing methods primarily follow an offline reconstruction paradigm, lacking the capacity for continuous reconstruction, which limits their applicability to online scenarios such as robotics and VR/AR. In this paper, we introduce OnlineX, a feed-forward framework that reconstructs both 3D visual appearance and language fields in an online manner using only streaming images. A key challenge in online formulation is the cumulative drift issue, which is rooted in the fundamental conflict between two opposing roles of the memory state: an active role that constantly refreshes to capture high-frequency local geometry, and a stable role that conservatively accumulates and preserves the long-term global structure. To address this, we introduce a decoupled active-to-stable state evolution paradigm. Our framework decouples the memory state into a dedicated active state and a persistent stable state, and then cohesively fuses the information from the former into the latter to achieve both fidelity and stability. Moreover, we jointly model visual appearance and language fields and incorporate an implicit Gaussian fusion module to enhance reconstruction quality. Experiments on mainstream datasets demonstrate that our method consistently outperforms prior work in novel view synthesis and semantic understanding, showcasing robust performance across input sequences of varying lengths with real-time inference speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02986v1">VIRGi: View-dependent Instant Recoloring of 3D Gaussians Splats</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 IEEE Transactions on Pattern Analysis and Machine Intelligence. 2026 Feb 24
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently transformed the fields of novel view synthesis and 3D reconstruction due to its ability to accurately model complex 3D scenes and its unprecedented rendering performance. However, a significant challenge persists: the absence of an efficient and photorealistic method for editing the appearance of the scene's content. In this paper we introduce VIRGi, a novel approach for rapidly editing the color of scenes modeled by 3DGS while preserving view-dependent effects such as specular highlights. Key to our method are a novel architecture that separates color into diffuse and view-dependent components, and a multi-view training strategy that integrates image patches from multiple viewpoints. Improving over the conventional single-view batch training, our 3DGS representation provides more accurate reconstruction and serves as a solid representation for the recoloring task. For 3DGS recoloring, we then introduce a rapid scheme requiring only one manually edited image of the scene from the end-user. By fine-tuning the weights of a single MLP, alongside a module for single-shot segmentation of the editable area, the color edits are seamlessly propagated to the entire scene in just two seconds, facilitating real-time interaction and providing control over the strength of the view-dependent effects. An exhaustive validation on diverse datasets demonstrates significant quantitative and qualitative advancements over competitors based on Neural Radiance Fields representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02893v1">Intrinsic Geometry-Appearance Consistency Optimization for Sparse-View Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      3D human reconstruction from a single image is a challenging problem and has been exclusively studied in the literature. Recently, some methods have resorted to diffusion models for guidance, optimizing a 3D representation via Score Distillation Sampling(SDS) or generating a back-view image for facilitating reconstruction. However, these methods tend to produce unsatisfactory artifacts (\textit{e.g.} flattened human structure or over-smoothing results caused by inconsistent priors from multiple views) and struggle with real-world generalization in the wild. In this work, we present \emph{MVD-HuGaS}, enabling free-view 3D human rendering from a single image via a multi-view human diffusion model. We first generate multi-view images from the single reference image with an enhanced multi-view diffusion model, which is well fine-tuned on high-quality 3D human datasets to incorporate 3D geometry priors and human structure priors. To infer accurate camera poses from the sparse generated multi-view images for reconstruction, an alignment module is introduced to facilitate joint optimization of 3D Gaussians and camera poses. Furthermore, we propose a depth-based Facial Distortion Mitigation module to refine the generated facial regions, thereby improving the overall fidelity of the reconstruction. Finally, leveraging the refined multi-view images, along with their accurate camera poses, MVD-HuGaS optimizes the 3D Gaussians of the target human for high-fidelity free-view renderings. Extensive experiments on Thuman2.0 and 2K2K datasets show that the proposed MVD-HuGaS achieves state-of-the-art performance on single-view 3D human rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02866v1">Multimodal-Prior-Guided Importance Sampling for Hierarchical Gaussian Splatting in Sparse-View Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      We present multimodal-prior-guided importance sampling as the central mechanism for hierarchical 3D Gaussian Splatting (3DGS) in sparse-view novel view synthesis. Our sampler fuses complementary cues { -- } photometric rendering residuals, semantic priors, and geometric priors { -- } to produce a robust, local recoverability estimate that directly drives where to inject fine Gaussians. Built around this sampling core, our framework comprises (1) a coarse-to-fine Gaussian representation that encodes global shape with a stable coarse layer and selectively adds fine primitives where the multimodal metric indicates recoverable detail; and (2) a geometric-aware sampling and retention policy that concentrates refinement on geometrically critical and complex regions while protecting newly added primitives in underconstrained areas from premature pruning. By prioritizing regions supported by consistent multimodal evidence rather than raw residuals alone, our method alleviates overfitting texture-induced errors and suppresses noise from pose/appearance inconsistencies. Experiments on diverse sparse-view benchmarks demonstrate state-of-the-art reconstructions, with up to +0.3 dB PSNR on DTU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02801v1">R3GW: Relightable 3D Gaussians for Outdoor Scenes in the Wild</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 Accepted at VISAPP 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has established itself as a leading technique for 3D reconstruction and novel view synthesis of static scenes, achieving outstanding rendering quality and fast training. However, the method does not explicitly model the scene illumination, making it unsuitable for relighting tasks. Furthermore, 3DGS struggles to reconstruct scenes captured in the wild by unconstrained photo collections featuring changing lighting conditions. In this paper, we present R3GW, a novel method that learns a relightable 3DGS representation of an outdoor scene captured in the wild. Our approach separates the scene into a relightable foreground and a non-reflective background (the sky), using two distinct sets of Gaussians. R3GW models view-dependent lighting effects in the foreground reflections by combining Physically Based Rendering with the 3DGS scene representation in a varying illumination setting. We evaluate our method quantitatively and qualitatively on the NeRF-OSR dataset, offering state-of-the-art performance and enhanced support for physically-based relighting of unconstrained scenes. Our method synthesizes photorealistic novel views under arbitrary illumination conditions. Additionally, our representation of the sky mitigates depth reconstruction artifacts, improving rendering quality at the sky-foreground boundary
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01099v2">HeroGS: Hierarchical Guidance for Robust 3D Gaussian Splatting under Sparse Views</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as a promising approach in novel view synthesis, combining photorealistic rendering with real-time efficiency. However, its success heavily relies on dense camera coverage; under sparse-view conditions, insufficient supervision leads to irregular Gaussian distributions, characterized by globally sparse coverage, blurred background, and distorted high-frequency areas. To address this, we propose HeroGS, Hierarchical Guidance for Robust 3D Gaussian Splatting, a unified framework that establishes hierarchical guidance across the image, feature, and parameter levels. At the image level, sparse supervision is converted into pseudo-dense guidance, globally regularizing the Gaussian distributions and forming a consistent foundation for subsequent optimization. Building upon this, Feature-Adaptive Densification and Pruning (FADP) at the feature level leverages low-level features to refine high-frequency details and adaptively densifies Gaussians in background regions. The optimized distributions then support Co-Pruned Geometry Consistency (CPG) at parameter level, which guides geometric consistency through parameter freezing and co-pruning, effectively removing inconsistent splats. The hierarchical guidance strategy effectively constrains and optimizes the overall Gaussian distributions, thereby enhancing both structural fidelity and rendering quality. Extensive experiments demonstrate that HeroGS achieves high-fidelity reconstructions and consistently surpasses state-of-the-art baselines under sparse-view conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24421v4">Proxy-GS: Unified Occlusion Priors for Training and Inference in Structured 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 Project page: https://visionary-laboratory.github.io/Proxy-GS
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as an efficient approach for achieving photorealistic rendering. Recent MLP-based variants further improve visual fidelity but introduce substantial decoding overhead during rendering. To alleviate computation cost, several pruning strategies and level-of-detail (LOD) techniques have been introduced, aiming to effectively reduce the number of Gaussian primitives in large-scale scenes. However, our analysis reveals that significant redundancy still remains due to the lack of occlusion awareness. In this work, we propose Proxy-GS, a novel pipeline that exploits a proxy to introduce Gaussian occlusion awareness from any view. At the core of our approach is a fast proxy system capable of producing precise occlusion depth maps at a resolution of 1000x1000 under 1ms. This proxy serves two roles: first, it guides the culling of anchors and Gaussians to accelerate rendering speed. Second, it guides the densification towards surfaces during training, avoiding inconsistencies in occluded regions, and improving the rendering quality. In heavily occluded scenarios, such as the MatrixCity Streets dataset, Proxy-GS not only equips MLP-based Gaussian splatting with stronger rendering capability but also achieves faster rendering speed. Specifically, it achieves more than 2.5x speedup over Octree-GS, and consistently delivers substantially higher rendering quality. Code will be public upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02548v1">SemGS: Feed-Forward Semantic 3D Gaussian Splatting from Sparse Views for Generalizable Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 ICRA 2026
    </div>
    <details class="paper-abstract">
      Semantic understanding of 3D scenes is essential for robots to operate effectively and safely in complex environments. Existing methods for semantic scene reconstruction and semantic-aware novel view synthesis often rely on dense multi-view inputs and require scene-specific optimization, limiting their practicality and scalability in real-world applications. To address these challenges, we propose SemGS, a feed-forward framework for reconstructing generalizable semantic fields from sparse image inputs. SemGS uses a dual-branch architecture to extract color and semantic features, where the two branches share shallow CNN layers, allowing semantic reasoning to leverage textural and structural cues in color appearance. We also incorporate a camera-aware attention mechanism into the feature extractor to explicitly model geometric relationships between camera viewpoints. The extracted features are decoded into dual-Gaussians that share geometric consistency while preserving branch-specific attributes, and further rasterized to synthesize semantic maps under novel viewpoints. Additionally, we introduce a regional smoothness loss to enhance semantic coherence. Experiments show that SemGS achieves state-of-the-art performance on benchmark datasets, while providing rapid inference and strong generalization capabilities across diverse synthetic and real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15151v2">Zero-shot CT Super-Resolution using Diffusion-based 2D Projection Priors and Signed 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Computed tomography (CT) is important in clinical diagnosis, but acquiring high-resolution (HR) CT is constrained by radiation exposure risks. While deep learning-based super-resolution (SR) methods have shown promise for reconstructing HR CT from low-resolution (LR) inputs, supervised approaches require paired datasets that are often unavailable. Zero-shot methods address this limitation by operating on single LR inputs; however, they frequently fail to recover fine structural details due to limited LR information within individual volumes. To overcome these limitations, we propose a novel zero-shot 3D CT SR framework that integrates diffusion-based upsampled 2D projection priors into the 3D reconstruction process. Specifically, our framework consists of two stages: (1) LR CT projection SR, training a diffusion model on abundant X-ray data to upsample LR projections, thereby enhancing the scarce information inherent in the LR inputs. (2) 3D CT volume reconstruction, using 3D Gaussian splatting with our novel Negative Alpha Blending (NAB-GS), which models positive and negative Gaussian densities to learn signed residuals between diffusion-generated HR and upsampled LR projections. Our framework demonstrates superior quantitative and qualitative performance on two public datasets, and expert evaluations present the framework's clinical potential at 4x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08862v2">StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted by ICLR 2026, Project page: https://streamsplat3d.github.io/
    </div>
    <details class="paper-abstract">
      Real-time reconstruction of dynamic 3D scenes from uncalibrated video streams demands robust online methods that recover scene dynamics from sparse observations under strict latency and memory constraints. Yet most dynamic reconstruction methods rely on hours of per-scene optimization under full-sequence access, limiting practical deployment. In this work, we introduce StreamSplat, a fully feed-forward framework that instantly transforms uncalibrated video streams of arbitrary length into dynamic 3D Gaussian Splatting (3DGS) representations in an online manner. It is achieved via three key technical innovations: 1) a probabilistic sampling mechanism that robustly predicts 3D Gaussians from uncalibrated inputs; 2) a bidirectional deformation field that yields reliable associations across frames and mitigates long-term error accumulation; 3) an adaptive Gaussian fusion operation that propagates persistent Gaussians while handling emerging and vanishing ones. Extensive experiments on standard dynamic and static benchmarks demonstrate that StreamSplat achieves state-of-the-art reconstruction quality and dynamic scene modeling. Uniquely, our method supports the online reconstruction of arbitrarily long video streams with a 1200x speedup over optimization-based methods. Our code and models are available at https://streamsplat3d.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20160v2">tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted by CVPR 2026. Project Page: https://cwchenwang.github.io/tttLRM
    </div>
    <details class="paper-abstract">
      We propose tttLRM, a novel large 3D reconstruction model that leverages a Test-Time Training (TTT) layer to enable long-context, autoregressive 3D reconstruction with linear computational complexity, further scaling the model's capability. Our framework efficiently compresses multiple image observations into the fast weights of the TTT layer, forming an implicit 3D representation in the latent space that can be decoded into various explicit formats, such as Gaussian Splats (GS) for downstream applications. The online learning variant of our model supports progressive 3D reconstruction and refinement from streaming observations. We demonstrate that pretraining on novel view synthesis tasks effectively transfers to explicit 3D modeling, resulting in improved reconstruction quality and faster convergence. Extensive experiments show that our method achieves superior performance in feedforward 3D Gaussian reconstruction compared to state-of-the-art approaches on both objects and scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02129v1">LiftAvatar: Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 19 pages, 11 figures
    </div>
    <details class="paper-abstract">
      We present LiftAvatar, a new paradigm that completes sparse monocular observations in kinematic space (e.g., facial expressions and head pose) and uses the completed signals to drive high-fidelity avatar animation. LiftAvatar is a fine-grained, expression-controllable large-scale video diffusion Transformer that synthesizes high-quality, temporally coherent expression sequences conditioned on single or multiple reference images. The key idea is to lift incomplete input data into a richer kinematic representation, thereby strengthening both reconstruction and animation in downstream 3D avatar pipelines. To this end, we introduce (i) a multi-granularity expression control scheme that combines shading maps with expression coefficients for precise and stable driving, and (ii) a multi-reference conditioning mechanism that aggregates complementary cues from multiple frames, enabling strong 3D consistency and controllability. As a plug-and-play enhancer, LiftAvatar directly addresses the limited expressiveness and reconstruction artifacts of 3D Gaussian Splatting-based avatars caused by sparse kinematic cues in everyday monocular videos. By expanding incomplete observations into diverse pose-expression variations, LiftAvatar also enables effective prior distillation from large-scale video generative models into 3D pipelines, leading to substantial gains. Extensive experiments show that LiftAvatar consistently boosts animation quality and quantitative metrics of state-of-the-art 3D avatar methods, especially under extreme, unseen expressions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01603v1">Sparse View Distractor-Free Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables efficient training and fast novel view synthesis in static environments. To address challenges posed by transient objects, distractor-free 3DGS methods have emerged and shown promising results when dense image captures are available. However, their performance degrades significantly under sparse input conditions. This limitation primarily stems from the reliance on the color residual heuristics to guide the training, which becomes unreliable with limited observations. In this work, we propose a framework to enhance distractor-free 3DGS under sparse-view conditions by incorporating rich prior information. Specifically, we first adopt the geometry foundation model VGGT to estimate camera parameters and generate a dense set of initial 3D points. Then, we harness the attention maps from VGGT for efficient and accurate semantic entity matching. Additionally, we utilize Vision-Language Models (VLMs) to further identify and preserve the large static regions in the scene. We also demonstrate how these priors can be seamlessly integrated into existing distractor-free 3DGS methods. Extensive experiments confirm the effectiveness and robustness of our approach in mitigating transient distractors for sparse-view 3DGS training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01491v1">Radiometrically Consistent Gaussian Surfels for Inverse Rendering</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 9 pages, 6 figures, ICLR 2026 Oral paper
    </div>
    <details class="paper-abstract">
      Inverse rendering with Gaussian Splatting has advanced rapidly, but accurately disentangling material properties from complex global illumination effects, particularly indirect illumination, remains a major challenge. Existing methods often query indirect radiance from Gaussian primitives pre-trained for novel-view synthesis. However, these pre-trained Gaussian primitives are supervised only towards limited training viewpoints, thus lack supervision for modeling indirect radiances from unobserved views. To address this issue, we introduce radiometric consistency, a novel physically-based constraint that provides supervision towards unobserved views by minimizing the residual between each Gaussian primitive's learned radiance and its physically-based rendered counterpart. Minimizing the residual for unobserved views establishes a self-correcting feedback loop that provides supervision from both physically-based rendering and novel-view synthesis, enabling accurate modeling of inter-reflection. We then propose Radiometrically Consistent Gaussian Surfels (RadioGS), an inverse rendering framework built upon our principle by efficiently integrating radiometric consistency by utilizing Gaussian surfels and 2D Gaussian ray tracing. We further propose a finetuning-based relighting strategy that adapts Gaussian surfel radiances to new illuminations within minutes, achieving low rendering cost (<10ms). Extensive experiments on existing inverse rendering benchmarks show that RadioGS outperforms existing Gaussian-based methods in inverse rendering, while retaining the computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01844v3">CloDS: Visual-Only Unsupervised Cloth Dynamics Learning in Unknown Conditions</a></div>
    <div class="paper-meta">
      📅 2026-03-01
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Deep learning has demonstrated remarkable capabilities in simulating complex dynamic systems. However, existing methods require known physical properties as supervision or inputs, limiting their applicability under unknown conditions. To explore this challenge, we introduce Cloth Dynamics Grounding (CDG), a novel scenario for unsupervised learning of cloth dynamics from multi-view visual observations. We further propose Cloth Dynamics Splatting (CloDS), an unsupervised dynamic learning framework designed for CDG. CloDS adopts a three-stage pipeline that first performs video-to-geometry grounding and then trains a dynamics model on the grounded meshes. To cope with large non-linear deformations and severe self-occlusions during grounding, we introduce a dual-position opacity modulation that supports bidirectional mapping between 2D observations and 3D geometry via mesh-based Gaussian splatting in video-to-geometry grounding stage. It jointly considers the absolute and relative position of Gaussian components. Comprehensive experimental evaluations demonstrate that CloDS effectively learns cloth dynamics from visual data while maintaining strong generalization capabilities for unseen configurations. Our code is available at https://github.com/whynot-zyl/CloDS. Visualization results are available at https://github.com/whynot-zyl/CloDS_video}.%\footnote{As in this example.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01158v1">FLICKER: A Fine-Grained Contribution-Aware Accelerator for Real-Time 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-01
      | 💬 Accepted at DATE 2026 (Design, Automation and Test in Europe Conference)
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has emerged as a mainstream rendering technique due to its photorealistic quality and low latency. However, processing massive numbers of non-contributing Gaussian points introduces significant computational overhead on resource-limited edge platforms, limiting its deployment in next-generation AR/VR devices. Contribution-based prior skipping alleviates this inefficiency, yet the resulting contribution-testing workload becomes prohibitive for edge execution. In this paper, we present FLICKER, a contribution-aware 3DGS accelerator based on hardware-software co-design. The proposed framework integrates adaptive leader pixels, pixel-rectangle grouping, hierarchical Gaussian testing, and a mixed-precision architecture to enable near pixel-level, contribution-driven rendering with minimal overhead. Experimental results demonstrate up to $1.5\times$ speedup, $2.6\times$ improvement in energy efficiency, and $14%$ area reduction compared with a state-of-the-art accelerator. Compared with a representative edge GPU, FLICKER achieves a $19.8\times$ speedup and $26.7\times$ higher energy efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01151v1">D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping</a></div>
    <div class="paper-meta">
      📅 2026-03-01
      | 💬 ICLR 2026 Poster
    </div>
    <details class="paper-abstract">
      Simulation provides a cost-effective and flexible platform for data generation and policy learning to develop robotic systems. However, bridging the gap between simulation and real-world dynamics remains a significant challenge, especially in physical parameter identification. In this work, we introduce a real-to-sim-to-real engine that leverages the Gaussian Splat representations to build a differentiable engine, enabling object mass identification from real-world visual observations and robot control signals, while enabling grasping policy learning simultaneously. Through optimizing the mass of the manipulated object, our method automatically builds high-fidelity and physically plausible digital twins. Additionally, we propose a novel approach to train force-aware grasping policies from limited data by transferring feasible human demonstrations into simulated robot demonstrations. Through comprehensive experiments, we demonstrate that our engine achieves accurate and robust performance in mass identification across various object geometries and mass values. Those optimized mass values facilitate force-aware policy learning, achieving superior and high performance in object grasping, effectively reducing the sim-to-real gap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.18041v7">Openfly: A comprehensive platform for aerial vision-language navigation</a></div>
    <div class="paper-meta">
      📅 2026-03-01
      | 💬 accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation (VLN) aims to guide agents by leveraging language instructions and visual cues, playing a pivotal role in embodied AI. Indoor VLN has been extensively studied, whereas outdoor aerial VLN remains underexplored. The potential reason is that outdoor aerial view encompasses vast areas, making data collection more challenging, which results in a lack of benchmarks. To address this problem, we propose OpenFly, a platform comprising various rendering engines, a versatile toolchain, and a large-scale benchmark for aerial VLN. Firstly, we integrate diverse rendering engines and advanced techniques for environment simulation, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of our environments. Secondly, we develop a highly automated toolchain for aerial VLN data collection, streamlining point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Thirdly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering diverse heights and lengths across 18 scenes. Moreover, we propose OpenFly-Agent, a keyframe-aware VLN model emphasizing key observations during flight. For benchmarking, extensive experiments and analyses are conducted, evaluating several recent VLN methods and showcasing the superiority of our OpenFly platform and agent. The toolchain, dataset, and codes will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18489v2">Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2026-03-01
      | 💬 This paper is accepted by ICLR 2026. Project page is available at https://liujf1226.github.io/Mono4DGS-HDR/
    </div>
    <details class="paper-abstract">
      We introduce Mono4DGS-HDR, the first system for reconstructing renderable 4D high dynamic range (HDR) scenes from unposed monocular low dynamic range (LDR) videos captured with alternating exposures. To tackle such a challenging problem, we present a unified framework with two-stage optimization approach based on Gaussian Splatting. The first stage learns a video HDR Gaussian representation in orthographic camera coordinate space, eliminating the need for camera poses and enabling robust initial HDR video reconstruction. The second stage transforms video Gaussians into world space and jointly refines the world Gaussians with camera poses. Furthermore, we propose a temporal luminance regularization strategy to enhance the temporal consistency of the HDR appearance. Since our task has not been studied before, we construct a new evaluation benchmark using publicly available datasets for HDR video reconstruction. Extensive experiments demonstrate that Mono4DGS-HDR significantly outperforms alternative solutions adapted from state-of-the-art methods in both rendering quality and speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.19754v2">FastAvatar: Towards Unified and Fast 3D Avatar Reconstruction with Large Gaussian Reconstruction Transformers</a></div>
    <div class="paper-meta">
      📅 2026-03-01
    </div>
    <details class="paper-abstract">
      Despite significant progress in 3D avatar reconstruction, it still faces challenges such as high time complexity, sensitivity to data quality, and low data utilization. We propose FastAvatar, a feedforward 3D avatar framework capable of flexibly leveraging diverse daily recordings (e.g., a single image, multi-view observations, or monocular video) to reconstruct a high-quality 3D Gaussian Splatting (3DGS) model within seconds, using only a single unified model. The core of FastAvatar is a Large Gaussian Reconstruction Transformer (LGRT) featuring three key designs: First, a 3DGS transformer aggregating multi-frame cues while injecting initial 3D prompt to predict the corresponding registered canonical 3DGS representations; Second, multi-granular guidance encoding (camera pose, expression coefficient, head pose) mitigating animation-induced misalignment for variable-length inputs; Third, incremental Gaussian aggregation via landmark tracking and sliced fusion losses. Integrating these features, FastAvatar enables incremental reconstruction, i.e., improving quality with more observations without wasting input data as in previous works. This yields a quality-speed-tunable paradigm for highly usable 3D avatar modeling. Extensive experiments show that FastAvatar has a higher quality and highly competitive speed compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00952v1">Decoupling Motion and Geometry in 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-01
    </div>
    <details class="paper-abstract">
      High-fidelity reconstruction of dynamic scenes is an important yet challenging problem. While recent 4D Gaussian Splatting (4DGS) has demonstrated the ability to model temporal dynamics, it couples Gaussian motion and geometric attributes within a single covariance formulation, which limits its expressiveness for complex motions and often leads to visual artifacts. To address this, we propose VeGaS, a novel velocity-based 4D Gaussian Splatting framework that decouples Gaussian motion and geometry. Specifically, we introduce a Galilean shearing matrix that explicitly incorporates time-varying velocity to flexibly model complex non-linear motions, while strictly isolating the effects of Gaussian motion from the geometry-related conditional Gaussian covariance. Furthermore, a Geometric Deformation Network is introduced to refine Gaussian shapes and orientations using spatio-temporal context and velocity cues, enhancing temporal geometric modeling. Extensive experiments on public datasets demonstrate that VeGaS achieves state-of-the-art performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21333v2">HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles</a></div>
    <div class="paper-meta">
      📅 2026-03-01
      | 💬 Accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      Controllable driving scene generation is critical for realistic and scalable autonomous driving simulation, yet existing approaches struggle to jointly achieve photorealism and precise control. We introduce HorizonForge, a unified framework that reconstructs scenes as editable Gaussian Splats and Meshes, enabling fine-grained 3D manipulation and language-driven vehicle insertion. Edits are rendered through a noise-aware video diffusion process that enforces spatial and temporal consistency, producing diverse scene variations in a single feed-forward pass without per-trajectory optimization. To standardize evaluation, we further propose HorizonSuite, a comprehensive benchmark spanning ego- and agent-level editing tasks such as trajectory modifications and object manipulation. Extensive experiments show that Gaussian-Mesh representation delivers substantially higher fidelity than alternative 3D representations, and that temporal priors from video diffusion are essential for coherent synthesis. Combining these findings, HorizonForge establishes a simple yet powerful paradigm for photorealistic, controllable driving simulation, achieving an 83.4% user-preference gain and a 25.19% FID improvement over the second best state-of-the-art method. Project page: https://horizonforge.github.io/ .
    </details>
</div>
