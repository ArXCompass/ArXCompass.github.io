# gaussian splatting - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04542v1">Gaussian Entropy Fields: Driving Adaptive Sparsity in 3D Gaussian Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 28 pages,11 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a leading technique for novel view synthesis, demonstrating exceptional rendering efficiency. \replaced[]{Well-reconstructed surfaces can be characterized by low configurational entropy, where dominant primitives clearly define surface geometry while redundant components are suppressed.}{The key insight is that well-reconstructed surfaces naturally exhibit low configurational entropy, where dominant primitives clearly define surface geometry while suppressing redundant components.} Three complementary technical contributions are introduced: (1) entropy-driven surface modeling via entropy minimization for low configurational entropy in primitive distributions; (2) adaptive spatial regularization using the Surface Neighborhood Redundancy Index (SNRI) and image entropy-guided weighting; (3) multi-scale geometric preservation through competitive cross-scale entropy alignment. Extensive experiments demonstrate that GEF achieves competitive geometric precision on DTU and T\&T benchmarks, while delivering superior rendering quality compared to existing methods on Mip-NeRF 360. Notably, superior Chamfer Distance (0.64) on DTU and F1 score (0.44) on T\&T are obtained, alongside the best SSIM (0.855) and LPIPS (0.136) among baselines on Mip-NeRF 360, validating the framework's ability to enhance surface reconstruction accuracy without compromising photometric fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.05168v3">SeeLe: A Unified Acceleration Framework for Real-Time Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a crucial rendering technique for many real-time applications. However, the limited hardware resources on today's mobile platforms hinder these applications, as they struggle to achieve real-time performance. In this paper, we propose SeeLe, a general framework designed to accelerate the 3DGS pipeline for resource-constrained mobile devices. Specifically, we propose two GPU-oriented techniques: hybrid preprocessing and contribution-aware rasterization. Hybrid preprocessing alleviates the GPU compute and memory pressure by reducing the number of irrelevant Gaussians during rendering. The key is to combine our view-dependent scene representation with online filtering. Meanwhile, contribution-aware rasterization improves the GPU utilization at the rasterization stage by prioritizing Gaussians with high contributions while reducing computations for those with low contributions. Both techniques can be seamlessly integrated into existing 3DGS pipelines with minimal fine-tuning. Collectively, our framework achieves 2.6$\times$ speedup and 32.3\% model reduction while achieving superior rendering quality compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04421v1">UTrice: Unifying Primitives in Differentiable Ray Tracing and Rasterization via Triangles for Particle-Based 3D Scenes</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 13 pages, 10 figures, submitted to CVPR2026
    </div>
    <details class="paper-abstract">
      Ray tracing 3D Gaussian particles enables realistic effects such as depth of field, refractions, and flexible camera modeling for novel-view synthesis. However, existing methods trace Gaussians through proxy geometry, which requires constructing complex intermediate meshes and performing costly intersection tests. This limitation arises because Gaussian-based particles are not well suited as unified primitives for both ray tracing and rasterization. In this work, we propose a differentiable triangle-based ray tracing pipeline that directly treats triangles as rendering primitives without relying on any proxy geometry. Our results show that the proposed method achieves significantly higher rendering quality than existing ray tracing approaches while maintaining real-time rendering performance. Moreover, our pipeline can directly render triangles optimized by the rasterization-based method Triangle Splatting, thus unifying the primitives used in novel-view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04315v1">SyncTrack4D: Cross-Video Motion Alignment and Video Synchronization for Multi-Video 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Modeling dynamic 3D scenes is challenging due to their high-dimensional nature, which requires aggregating information from multiple views to reconstruct time-evolving 3D geometry and motion. We present a novel multi-video 4D Gaussian Splatting (4DGS) approach designed to handle real-world, unsynchronized video sets. Our approach, SyncTrack4D, directly leverages dense 4D track representation of dynamic scene parts as cues for simultaneous cross-video synchronization and 4DGS reconstruction. We first compute dense per-video 4D feature tracks and cross-video track correspondences by Fused Gromov-Wasserstein optimal transport approach. Next, we perform global frame-level temporal alignment to maximize overlapping motion of matched 4D tracks. Finally, we achieve sub-frame synchronization through our multi-video 4D Gaussian splatting built upon a motion-spline scaffold representation. The final output is a synchronized 4DGS representation with dense, explicit 3D trajectories, and temporal offsets for each video. We evaluate our approach on the Panoptic Studio and SyncNeRF Blender, demonstrating sub-frame synchronization accuracy with an average temporal error below 0.26 frames, and high-fidelity 4D reconstruction reaching 26.3 PSNR scores on the Panoptic Studio dataset. To the best of our knowledge, our work is the first general 4D Gaussian Splatting approach for unsynchronized video sets, without assuming the existence of predefined scene objects or prior models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04313v1">Mind-to-Face: Neural-Driven Photorealistic Avatar Synthesis via EEG Decoding</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 16 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Current expressive avatar systems rely heavily on visual cues, failing when faces are occluded or when emotions remain internal. We present Mind-to-Face, the first framework that decodes non-invasive electroencephalogram (EEG) signals directly into high-fidelity facial expressions. We build a dual-modality recording setup to obtain synchronized EEG and multi-view facial video during emotion-eliciting stimuli, enabling precise supervision for neural-to-visual learning. Our model uses a CNN-Transformer encoder to map EEG signals into dense 3D position maps, capable of sampling over 65k vertices, capturing fine-scale geometry and subtle emotional dynamics, and renders them through a modified 3D Gaussian Splatting pipeline for photorealistic, view-consistent results. Through extensive evaluation, we show that EEG alone can reliably predict dynamic, subject-specific facial expressions, including subtle emotional responses, demonstrating that neural signals contain far richer affective and geometric information than previously assumed. Mind-to-Face establishes a new paradigm for neural-driven avatars, enabling personalized, emotion-aware telepresence and cognitive interaction in immersive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04021v1">C3G: Learning Compact 3D Representations with 2K Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 Project Page : https://cvlab-kaist.github.io/C3G/
    </div>
    <details class="paper-abstract">
      Reconstructing and understanding 3D scenes from unposed sparse views in a feed-forward manner remains as a challenging task in 3D computer vision. Recent approaches use per-pixel 3D Gaussian Splatting for reconstruction, followed by a 2D-to-3D feature lifting stage for scene understanding. However, they generate excessive redundant Gaussians, causing high memory overhead and sub-optimal multi-view feature aggregation, leading to degraded novel view synthesis and scene understanding performance. We propose C3G, a novel feed-forward framework that estimates compact 3D Gaussians only at essential spatial locations, minimizing redundancy while enabling effective feature lifting. We introduce learnable tokens that aggregate multi-view features through self-attention to guide Gaussian generation, ensuring each Gaussian integrates relevant visual features across views. We then exploit the learned attention patterns for Gaussian decoding to efficiently lift features. Extensive experiments on pose-free novel view synthesis, 3D open-vocabulary segmentation, and view-invariant feature aggregation demonstrate our approach's effectiveness. Results show that a compact yet geometrically meaningful representation is sufficient for high-quality scene reconstruction and understanding, achieving superior memory efficiency and feature fidelity compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13186v4">STT-GS: Sample-Then-Transmit Edge Gaussian Splatting with Joint Client Selection and Power Control</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Edge Gaussian splatting (EGS), which aggregates data from distributed clients (e.g., drones) and trains a global GS model at the edge (e.g., ground server), is an emerging paradigm for scene reconstruction in low-altitude economy. Unlike traditional edge resource management methods that emphasize communication throughput or general-purpose learning performance, EGS explicitly aims to maximize the GS qualities, rendering existing approaches inapplicable. To address this problem, this paper formulates a novel GS-oriented objective function that distinguishes the heterogeneous view contributions of different clients. However, evaluating this function in turn requires clients' images, leading to a causality dilemma. To this end, this paper further proposes a sample-then-transmit EGS (or STT-GS for short) strategy, which first samples a subset of images as pilot data from each client for loss prediction. Based on the first-stage evaluation, communication resources are then prioritized towards more valuable clients. To achieve efficient sampling, a feature-domain clustering (FDC) scheme is proposed to select the most representative data and pilot transmission time minimization (PTTM) is adopted to reduce the pilot overhead. Subsequently, we develop a joint client selection and power control (JCSPC) framework to maximize the GS-oriented function under communication resource constraints. Despite the nonconvexity of the problem, we propose a low-complexity efficient solution based on the penalty alternating majorization minimization (PAMM) algorithm. Experiments reveal that the proposed scheme significantly outperforms existing benchmarks on real-world datasets. The GS-oriented objective can be accurately predicted with low sampling ratios (e.g., 10%), and our method achieves an excellent tradeoff between view contributions and communication costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.15122v4">MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 This paper has been accepted to AAAI 2026. The first two authors contributed equally to this work (equal contribution). The last two authors are co-corresponding authors. Please visit our project page at https://kaist-viclab.github.io/mobgs-site/
    </div>
    <details class="paper-abstract">
      We present MoBGS, a novel motion deblurring 3D Gaussian Splatting (3DGS) framework capable of reconstructing sharp and high-quality novel spatio-temporal views from blurry monocular videos in an end-to-end manner. Existing dynamic novel view synthesis (NVS) methods are highly sensitive to motion blur in casually captured videos, resulting in significant degradation of rendering quality. While recent approaches address motion-blurred inputs for NVS, they primarily focus on static scene reconstruction and lack dedicated motion modeling for dynamic objects. To overcome these limitations, our MoBGS introduces a novel Blur-adaptive Latent Camera Estimation (BLCE) method using a proposed Blur-adaptive Neural Ordinary Differential Equation (ODE) solver for effective latent camera trajectory estimation, improving global camera motion deblurring. In addition, we propose a Latent Camera-induced Exposure Estimation (LCEE) method to ensure consistent deblurring of both a global camera and local object motions. Extensive experiments on the Stereo Blur dataset and real-world blurry videos show that our MoBGS significantly outperforms the very recent methods, achieving state-of-the-art performance for dynamic NVS under motion blur.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03601v1">Motion4D: Learning 3D-Consistent Motion and Semantics for 4D Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in foundation models for 2D vision have substantially improved the analysis of dynamic scenes from monocular videos. However, despite their strong generalization capabilities, these models often lack 3D consistency, a fundamental requirement for understanding scene geometry and motion, thereby causing severe spatial misalignment and temporal flickering in complex 3D environments. In this paper, we present Motion4D, a novel framework that addresses these challenges by integrating 2D priors from foundation models into a unified 4D Gaussian Splatting representation. Our method features a two-part iterative optimization framework: 1) Sequential optimization, which updates motion and semantic fields in consecutive stages to maintain local consistency, and 2) Global optimization, which jointly refines all attributes for long-term coherence. To enhance motion accuracy, we introduce a 3D confidence map that dynamically adjusts the motion priors, and an adaptive resampling process that inserts new Gaussians into under-represented regions based on per-pixel RGB and semantic errors. Furthermore, we enhance semantic coherence through an iterative refinement process that resolves semantic inconsistencies by alternately optimizing the semantic fields and updating prompts of SAM2. Extensive evaluations demonstrate that our Motion4D significantly outperforms both 2D foundation models and existing 3D-based approaches across diverse scene understanding tasks, including point-based tracking, video object segmentation, and novel view synthesis. Our code is available at https://hrzhou2.github.io/motion4d-web/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08710v2">SceneSplat++: A Large Dataset and Comprehensive Benchmark for Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 15 pages, codes, data and benchmark are released
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) serves as a highly performant and efficient encoding of scene geometry, appearance, and semantics. Moreover, grounding language in 3D scenes has proven to be an effective strategy for 3D scene understanding. Current Language Gaussian Splatting line of work fall into three main groups: (i) per-scene optimization-based, (ii) per-scene optimization-free, and (iii) generalizable approach. However, most of them are evaluated only on rendered 2D views of a handful of scenes and viewpoints close to the training views, limiting ability and insight into holistic 3D understanding. To address this gap, we propose the first large-scale benchmark that systematically assesses these three groups of methods directly in 3D space, evaluating on 1060 scenes across three indoor datasets and one outdoor dataset. Benchmark results demonstrate a clear advantage of the generalizable paradigm, particularly in relaxing the scene-specific limitation, enabling fast feed-forward inference on novel scenes, and achieving superior segmentation performance. We further introduce GaussianWorld-49K a carefully curated 3DGS dataset comprising around 49K diverse indoor and outdoor scenes obtained from multiple sources, with which we demonstrate the generalizable approach could harness strong data priors. Our codes, benchmark, and datasets are public to accelerate research in generalizable 3DGS scene understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.06859v2">ActiveInitSplat: How Active Image Selection Helps Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Gaussian splatting (GS) along with its extensions and variants provides outstanding performance in real-time scene rendering while meeting reduced storage demands and computational efficiency. While the selection of 2D images capturing the scene of interest is crucial for the proper initialization and training of GS, hence markedly affecting the rendering performance, prior works rely on passively and typically densely selected 2D images. In contrast, this paper proposes `ActiveInitSplat', a novel framework for active selection of training images for proper initialization and training of GS. ActiveInitSplat relies on density and occupancy criteria of the resultant 3D scene representation from the selected 2D images, to ensure that the latter are captured from diverse viewpoints leading to better scene coverage and that the initialized Gaussian functions are well aligned with the actual 3D structure. Numerical tests on well-known simulated and real environments demonstrate the merits of ActiveInitSplat resulting in significant GS rendering performance improvement over passive GS baselines in both dense- and sparse-view settings, in the widely adopted LPIPS, SSIM, and PSNR metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06517v3">GS4: Generalizable Sparse Splatting Semantic SLAM</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Traditional SLAM algorithms excel at camera tracking, but typically produce incomplete and low-resolution maps that are not tightly integrated with semantics prediction. Recent work integrates Gaussian Splatting (GS) into SLAM to enable dense, photorealistic 3D mapping, yet existing GS-based SLAM methods require per-scene optimization that is slow and consumes an excessive number of Gaussians. We present GS4, the first generalizable GS-based semantic SLAM system. Compared with prior approaches, GS4 runs 10x faster, uses 10x fewer Gaussians, and achieves state-of-the-art performance across color, depth, semantic mapping and camera tracking. From an RGB-D video stream, GS4 incrementally builds and updates a set of 3D Gaussians using a feed-forward network. First, the Gaussian Prediction Model estimates a sparse set of Gaussian parameters from input frame, which integrates both color and semantic prediction with the same backbone. Then, the Gaussian Refinement Network merges new Gaussians with the existing set while avoiding redundancy. Finally, when significant pose changes are detected, we perform only 1-5 iterations of joint Gaussian-pose optimization to correct drift, remove floaters, and further improve tracking accuracy. Experiments on the real-world ScanNet and ScanNet++ benchmarks demonstrate state-of-the-art semantic SLAM performance, with strong generalization capability shown through zero-shot transfer to the NYUv2 and TUM RGB-D datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03422v1">What Is The Best 3D Scene Representation for Robotics? From Geometric to Foundation Models</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      In this paper, we provide a comprehensive overview of existing scene representation methods for robotics, covering traditional representations such as point clouds, voxels, signed distance functions (SDF), and scene graphs, as well as more recent neural representations like Neural Radiance Fields (NeRF), 3D Gaussian Splatting (3DGS), and the emerging Foundation Models. While current SLAM and localization systems predominantly rely on sparse representations like point clouds and voxels, dense scene representations are expected to play a critical role in downstream tasks such as navigation and obstacle avoidance. Moreover, neural representations such as NeRF, 3DGS, and foundation models are well-suited for integrating high-level semantic features and language-based priors, enabling more comprehensive 3D scene understanding and embodied intelligence. In this paper, we categorized the core modules of robotics into five parts (Perception, Mapping, Localization, Navigation, Manipulation). We start by presenting the standard formulation of different scene representation methods and comparing the advantages and disadvantages of scene representation across different modules. This survey is centered around the question: What is the best 3D scene representation for robotics? We then discuss the future development trends of 3D scene representations, with a particular focus on how the 3D Foundation Model could replace current methods as the unified solution for future robotic applications. The remaining challenges in fully realizing this model are also explored. We aim to offer a valuable resource for both newcomers and experienced researchers to explore the future of 3D scene representations and their application in robotics. We have published an open-source project on GitHub and will continue to add new works and technologies to this project.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08136v2">FantasyStyle: Controllable Stylized Distillation for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      The success of 3DGS in generative and editing applications has sparked growing interest in 3DGS-based style transfer. However, current methods still face two major challenges: (1) multi-view inconsistency often leads to style conflicts, resulting in appearance smoothing and distortion; and (2) heavy reliance on VGG features, which struggle to disentangle style and content from style images, often causing content leakage and excessive stylization. To tackle these issues, we introduce \textbf{FantasyStyle}, a 3DGS-based style transfer framework, and the first to rely entirely on diffusion model distillation. It comprises two key components: (1) \textbf{Multi-View Frequency Consistency}. We enhance cross-view consistency by applying a 3D filter to multi-view noisy latent, selectively reducing low-frequency components to mitigate stylized prior conflicts. (2) \textbf{Controllable Stylized Distillation}. To suppress content leakage from style images, we introduce negative guidance to exclude undesired content. In addition, we identify the limitations of Score Distillation Sampling and Delta Denoising Score in 3D style transfer and remove the reconstruction term accordingly. Building on these insights, we propose a controllable stylized distillation that leverages negative guidance to more effectively optimize the 3D Gaussians. Extensive experiments demonstrate that our method consistently outperforms state-of-the-art approaches, achieving higher stylization quality and visual realism across various scenes and styles. The code is available at https://github.com/yangyt46/FantasyStyle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02621v1">Content-Aware Texturing for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 Project Page: https://repo-sam.inria.fr/nerphys/gs-texturing/
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has become the method of choice for 3D reconstruction and real-time rendering of captured real scenes. However, fine appearance details need to be represented as a large number of small Gaussian primitives, which can be wasteful when geometry and appearance exhibit different frequency characteristics. Inspired by the long tradition of texture mapping, we propose to use texture to represent detailed appearance where possible. Our main focus is to incorporate per-primitive texture maps that adapt to the scene in a principled manner during Gaussian Splatting optimization. We do this by proposing a new appearance representation for 2D Gaussian primitives with textures where the size of a texel is bounded by the image sampling frequency and adapted to the content of the input images. We achieve this by adaptively upscaling or downscaling the texture resolution during optimization. In addition, our approach enables control of the number of primitives during optimization based on texture resolution. We show that our approach performs favorably in image quality and total number of parameters used compared to alternative solutions for textured Gaussian primitives. Project page: https://repo-sam.inria.fr/nerphys/gs-texturing/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02482v1">G-SHARP: Gaussian Surgical Hardware Accelerated Real-time Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      We propose G-SHARP, a commercially compatible, real-time surgical scene reconstruction framework designed for minimally invasive procedures that require fast and accurate 3D modeling of deformable tissue. While recent Gaussian splatting approaches have advanced real-time endoscopic reconstruction, existing implementations often depend on non-commercial derivatives, limiting deployability. G-SHARP overcomes these constraints by being the first surgical pipeline built natively on the GSplat (Apache-2.0) differentiable Gaussian rasterizer, enabling principled deformation modeling, robust occlusion handling, and high-fidelity reconstructions on the EndoNeRF pulling benchmark. Our results demonstrate state-of-the-art reconstruction quality with strong speed-accuracy trade-offs suitable for intra-operative use. Finally, we provide a Holoscan SDK application that deploys G-SHARP on NVIDIA IGX Orin and Thor edge hardware, enabling real-time surgical visualization in practical operating-room settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.26219v2">Beyond Pixels: Efficient Dataset Distillation via Sparse Gaussian Representation</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 19 pages; Code is available on https://github.com/j-cyoung/GSDatasetDistillation
    </div>
    <details class="paper-abstract">
      Dataset distillation has emerged as a promising paradigm that synthesizes compact, informative datasets capable of retaining the knowledge of large-scale counterparts, thereby addressing the substantial computational and storage burdens of modern model training. Conventional approaches typically rely on dense pixel-level representations, which introduce redundancy and are difficult to scale up. In this work, we propose GSDD, a novel and efficient sparse representation for dataset distillation based on 2D Gaussians. Instead of representing all pixels equally, GSDD encodes critical discriminative information in a distilled image using only a small number of Gaussian primitives. This sparse representation could improve dataset diversity under the same storage budget, enhancing coverage of difficult samples and boosting distillation performance. To ensure both efficiency and scalability, we adapt CUDA-based splatting operators for parallel inference and training, enabling high-quality rendering with minimal computational and memory overhead. Our method is simple yet effective, broadly applicable to different distillation pipelines, and highly scalable. Experiments show that GSDD achieves state-of-the-art performance on CIFAR-10, CIFAR-100, and ImageNet subsets, while remaining highly efficient encoding and decoding cost. Our code is available at https://github.com/j-cyoung/GSDatasetDistillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02293v1">VIGS-SLAM: Visual Inertial Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 Project page: https://vigs-slam.github.io
    </div>
    <details class="paper-abstract">
      We present VIGS-SLAM, a visual-inertial 3D Gaussian Splatting SLAM system that achieves robust real-time tracking and high-fidelity reconstruction. Although recent 3DGS-based SLAM methods achieve dense and photorealistic mapping, their purely visual design degrades under motion blur, low texture, and exposure variations. Our method tightly couples visual and inertial cues within a unified optimization framework, jointly refining camera poses, depths, and IMU states. It features robust IMU initialization, time-varying bias modeling, and loop closure with consistent Gaussian updates. Experiments on four challenging datasets demonstrate our superiority over state-of-the-art methods. Project page: https://vigs-slam.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03210v1">Flux4D: Flow-based Unsupervised 4D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 NeurIPS 2025. Project page: https://waabi.ai/flux4d/
    </div>
    <details class="paper-abstract">
      Reconstructing large-scale dynamic scenes from visual observations is a fundamental challenge in computer vision, with critical implications for robotics and autonomous systems. While recent differentiable rendering methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have achieved impressive photorealistic reconstruction, they suffer from scalability limitations and require annotations to decouple actor motion. Existing self-supervised methods attempt to eliminate explicit annotations by leveraging motion cues and geometric priors, yet they remain constrained by per-scene optimization and sensitivity to hyperparameter tuning. In this paper, we introduce Flux4D, a simple and scalable framework for 4D reconstruction of large-scale dynamic scenes. Flux4D directly predicts 3D Gaussians and their motion dynamics to reconstruct sensor observations in a fully unsupervised manner. By adopting only photometric losses and enforcing an "as static as possible" regularization, Flux4D learns to decompose dynamic elements directly from raw data without requiring pre-trained supervised models or foundational priors simply by training across many scenes. Our approach enables efficient reconstruction of dynamic scenes within seconds, scales effectively to large datasets, and generalizes well to unseen environments, including rare and unknown objects. Experiments on outdoor driving datasets show Flux4D significantly outperforms existing methods in scalability, generalization, and reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03010v1">SurfFill: Completion of LiDAR Point Clouds via Gaussian Surfel Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 Project page: https://lfranke.github.io/surffill
    </div>
    <details class="paper-abstract">
      LiDAR-captured point clouds are often considered the gold standard in active 3D reconstruction. While their accuracy is exceptional in flat regions, the capturing is susceptible to miss small geometric structures and may fail with dark, absorbent materials. Alternatively, capturing multiple photos of the scene and applying 3D photogrammetry can infer these details as they often represent feature-rich regions. However, the accuracy of LiDAR for featureless regions is rarely reached. Therefore, we suggest combining the strengths of LiDAR and camera-based capture by introducing SurfFill: a Gaussian surfel-based LiDAR completion scheme. We analyze LiDAR capturings and attribute LiDAR beam divergence as a main factor for artifacts, manifesting mostly at thin structures and edges. We use this insight to introduce an ambiguity heuristic for completed scans by evaluating the change in density in the point cloud. This allows us to identify points close to missed areas, which we can then use to grow additional points from to complete the scan. For this point growing, we constrain Gaussian surfel reconstruction [Huang et al. 2024] to focus optimization and densification on these ambiguous areas. Finally, Gaussian primitives of the reconstruction in ambiguous areas are extracted and sampled for points to complete the point cloud. To address the challenges of large-scale reconstruction, we extend this pipeline with a divide-and-conquer scheme for building-sized point cloud completion. We evaluate on the task of LiDAR point cloud completion of synthetic and real-world scenes and find that our method outperforms previous reconstruction methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02932v1">EGGS: Exchangeable 2D/3D Gaussian Splatting for Geometry-Appearance Balanced Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) is crucial in computer vision and graphics, with wide applications in AR, VR, and autonomous driving. While 3D Gaussian Splatting (3DGS) enables real-time rendering with high appearance fidelity, it suffers from multi-view inconsistencies, limiting geometric accuracy. In contrast, 2D Gaussian Splatting (2DGS) enforces multi-view consistency but compromises texture details. To address these limitations, we propose Exchangeable Gaussian Splatting (EGGS), a hybrid representation that integrates 2D and 3D Gaussians to balance appearance and geometry. To achieve this, we introduce Hybrid Gaussian Rasterization for unified rendering, Adaptive Type Exchange for dynamic adaptation between 2D and 3D Gaussians, and Frequency-Decoupled Optimization that effectively exploits the strengths of each type of Gaussian representation. Our CUDA-accelerated implementation ensures efficient training and inference. Extensive experiments demonstrate that EGGS outperforms existing methods in rendering quality, geometric accuracy, and efficiency, providing a practical solution for high-quality NVS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.03659v6">DehazeGS: Seeing Through Fog with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 9 pages,5 figures. Accepted by AAAI2026. visualizations are available at https://dehazegs.github.io/
    </div>
    <details class="paper-abstract">
      Current novel view synthesis methods are typically designed for high-quality and clean input images. However, in foggy scenes, scattering and attenuation can significantly degrade the quality of rendering. Although NeRF-based dehazing approaches have been developed, their reliance on deep fully connected neural networks and per-ray sampling strategies leads to high computational costs. Furthermore, NeRF's implicit representation limits its ability to recover fine-grained details from hazy scenes. To overcome these limitations, we propose learning an explicit Gaussian representation to explain the formation mechanism of foggy images through a physically forward rendering process. Our method, DehazeGS, reconstructs and renders fog-free scenes using only multi-view foggy images as input. Specifically, based on the atmospheric scattering model, we simulate the formation of fog by establishing the transmission function directly onto Gaussian primitives via depth-to-transmission mapping. During training, we jointly learn the atmospheric light and scattering coefficients while optimizing the Gaussian representation of foggy scenes. At inference time, we remove the effects of scattering and attenuation in Gaussian distributions and directly render the scene to obtain dehazed views. Experiments on both real-world and synthetic foggy datasets demonstrate that DehazeGS achieves state-of-the-art performance. visualizations are available at https://dehazegs.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08294v2">SkelSplat: Robust Multi-view 3D Human Pose Estimation with Differentiable Gaussian Rendering</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 WACV 2026
    </div>
    <details class="paper-abstract">
      Accurate 3D human pose estimation is fundamental for applications such as augmented reality and human-robot interaction. State-of-the-art multi-view methods learn to fuse predictions across views by training on large annotated datasets, leading to poor generalization when the test scenario differs. To overcome these limitations, we propose SkelSplat, a novel framework for multi-view 3D human pose estimation based on differentiable Gaussian rendering. Human pose is modeled as a skeleton of 3D Gaussians, one per joint, optimized via differentiable rendering to enable seamless fusion of arbitrary camera views without 3D ground-truth supervision. Since Gaussian Splatting was originally designed for dense scene reconstruction, we propose a novel one-hot encoding scheme that enables independent optimization of human joints. SkelSplat outperforms approaches that do not rely on 3D ground truth in Human3.6M and CMU, while reducing the cross-dataset error up to 47.8% compared to learning-based methods. Experiments on Human3.6M-Occ and Occlusion-Person demonstrate robustness to occlusions, without scenario-specific fine-tuning. Our project page is available here: https://skelsplat.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02664v1">PolarGuide-GSDR: 3D Gaussian Splatting Driven by Polarization Priors and Deferred Reflection for Real-World Reflective Scenes</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Polarization-aware Neural Radiance Fields (NeRF) enable novel view synthesis of specular-reflection scenes but face challenges in slow training, inefficient rendering, and strong dependencies on material/viewpoint assumptions. However, 3D Gaussian Splatting (3DGS) enables real-time rendering yet struggles with accurate reflection reconstruction from reflection-geometry entanglement, adding a deferred reflection module introduces environment map dependence. We address these limitations by proposing PolarGuide-GSDR, a polarization-forward-guided paradigm establishing a bidirectional coupling mechanism between polarization and 3DGS: first 3DGS's geometric priors are leveraged to resolve polarization ambiguity, and then the refined polarization information cues are used to guide 3DGS's normal and spherical harmonic representation. This process achieves high-fidelity reflection separation and full-scene reconstruction without requiring environment maps or restrictive material assumptions. We demonstrate on public and self-collected datasets that PolarGuide-GSDR achieves state-of-the-art performance in specular reconstruction, normal estimation, and novel view synthesis, all while maintaining real-time rendering capabilities. To our knowledge, this is the first framework embedding polarization priors directly into 3DGS optimization, yielding superior interpretability and real-time performance for modeling complex reflective scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02648v1">PoreTrack3D: A Benchmark for Dynamic 3D Gaussian Splatting in Pore-Scale Facial Trajectory Tracking</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      We introduce PoreTrack3D, the first benchmark for dynamic 3D Gaussian splatting in pore-scale, non-rigid 3D facial trajectory tracking. It contains over 440,000 facial trajectories in total, among which more than 52,000 are longer than 10 frames, including 68 manually reviewed trajectories that span the entire 150 frames. To the best of our knowledge, PoreTrack3D is the first benchmark dataset to capture both traditional facial landmarks and pore-scale keypoints trajectory, advancing the study of fine-grained facial expressions through the analysis of subtle skin-surface motion. We systematically evaluate state-of-the-art dynamic 3D Gaussian splatting methods on PoreTrack3D, establishing the first performance baseline in this domain. Overall, the pipeline developed for this benchmark dataset's creation establishes a new framework for high-fidelity facial motion capture and dynamic 3D reconstruction. Our dataset are publicly available at: https://github.com/JHXion9/PoreTrack3D
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02443v2">PRIMU: Uncertainty Estimation for Novel Views in Gaussian Splatting from Primitive-Based Representations of Error and Coverage</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 Revised writing and figures; additional Gaussian Splatting experiments; added baselines and datasets; active view-selection experiments
    </div>
    <details class="paper-abstract">
      We introduce Primitive-based Representations of Uncertainty (PRIMU), a post-hoc uncertainty estimation (UE) framework for Gaussian Splatting (GS). Reliable UE is essential for deploying GS in safety-critical domains such as robotics and medicine. Existing approaches typically estimate Gaussian-primitive variances and rely on the rendering process to obtain pixel-wise uncertainties. In contrast, we construct primitive-level representations of error and visibility/coverage from training views, capturing interpretable uncertainty information. These representations are obtained by projecting view-dependent training errors and coverage statistics onto the primitives. Uncertainties for novel views are inferred by rendering these primitive-level representations, producing uncertainty feature maps, which are aggregate through pixel-wise regression on holdout data. We analyze combinations of uncertainty feature maps and regression models to understand how their interactions affect prediction accuracy and generalization. PRIMU also enables an effective active view selection strategy by directly leveraging these uncertainty feature maps. Additionally, we study the effect of separating splatting into foreground and background regions. Our estimates show strong correlations with true errors, outperforming state-of-the-art methods, especially for depth UE and foreground objects. Finally, our regression models show generalization capabilities to unseen scenes, enabling UE without additional holdout data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15719v1">A Fast Volumetric Capture and Reconstruction Pipeline for Dynamic Point Clouds and Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 ACM SIGGRAPH European Conference on Visual Media Production (CVMP) 2025. Code available at: https://github.com/irc-hslu/capturestudio
    </div>
    <details class="paper-abstract">
      We present a fast and efficient volumetric capture and reconstruction system that processes either RGB-D or RGB-only input to generate 3D representations in the form of point clouds and Gaussian splats. For Gaussian splat reconstructions, we took the GPS-Gaussian regressor and improved it, enabling high-quality reconstructions with minimal overhead. The system is designed for easy setup and deployment, supporting in-the-wild operation under uncontrolled illumination and arbitrary backgrounds, as well as flexible camera configurations, including sparse setups, arbitrary camera numbers and baselines. Captured data can be exported in standard formats such as PLY, MPEG V-PCC, and SPLAT, and visualized through a web-based viewer or Unity/Unreal plugins. A live on-location preview of both input and reconstruction is available at 5-10 FPS. We present qualitative findings focused on deployability and targeted ablations. The complete framework is open-source, facilitating reproducibility and further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01296v1">EGG-Fusion: Efficient 3D Reconstruction with Geometry-aware Gaussian Surfel on the Fly</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 SIGGRAPH ASIA 2025
    </div>
    <details class="paper-abstract">
      Real-time 3D reconstruction is a fundamental task in computer graphics. Recently, differentiable-rendering-based SLAM system has demonstrated significant potential, enabling photorealistic scene rendering through learnable scene representations such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). Current differentiable rendering methods face dual challenges in real-time computation and sensor noise sensitivity, leading to degraded geometric fidelity in scene reconstruction and limited practicality. To address these challenges, we propose a novel real-time system EGG-Fusion, featuring robust sparse-to-dense camera tracking and a geometry-aware Gaussian surfel mapping module, introducing an information filter-based fusion method that explicitly accounts for sensor noise to achieve high-precision surface reconstruction. The proposed differentiable Gaussian surfel mapping effectively models multi-view consistent surfaces while enabling efficient parameter optimization. Extensive experimental results demonstrate that the proposed system achieves a surface reconstruction error of 0.6\textit{cm} on standardized benchmark datasets including Replica and ScanNet++, representing over 20\% improvement in accuracy compared to state-of-the-art (SOTA) GS-based methods. Notably, the system maintains real-time processing capabilities at 24 FPS, establishing it as one of the most accurate differentiable-rendering-based real-time reconstruction systems. Project Page: https://zju3dv.github.io/eggfusion/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.01846v4">UVGS: Reimagining Unstructured 3D Gaussian Splatting using UV Mapping</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 https://ivl.cs.brown.edu/uvgs
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated superior quality in modeling 3D objects and scenes. However, generating 3DGS remains challenging due to their discrete, unstructured, and permutation-invariant nature. In this work, we present a simple yet effective method to overcome these challenges. We utilize spherical mapping to transform 3DGS into a structured 2D representation, termed UVGS. UVGS can be viewed as multi-channel images, with feature dimensions as a concatenation of Gaussian attributes such as position, scale, color, opacity, and rotation. We further find that these heterogeneous features can be compressed into a lower-dimensional (e.g., 3-channel) shared feature space using a carefully designed multi-branch network. The compressed UVGS can be treated as typical RGB images. Remarkably, we discover that typical VAEs trained with latent diffusion models can directly generalize to this new representation without additional training. Our novel representation makes it effortless to leverage foundational 2D models, such as diffusion models, to directly model 3DGS. Additionally, one can simply increase the 2D UV resolution to accommodate more Gaussians, making UVGS a scalable solution compared to typical 3D backbones. This approach immediately unlocks various novel generation applications of 3DGS by inherently utilizing the already developed superior 2D generation capabilities. In our experiments, we demonstrate various unconditional, conditional generation, and inpainting applications of 3DGS based on diffusion models, which were previously non-trivial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02172v1">SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 Project Page: https://splatsure.github.io/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables high-quality novel view synthesis, motivating interest in generating higher-resolution renders than those available during training. A natural strategy is to apply super-resolution (SR) to low-resolution (LR) input views, but independently enhancing each image introduces multi-view inconsistencies, leading to blurry renders. Prior methods attempt to mitigate these inconsistencies through learned neural components, temporally consistent video priors, or joint optimization on LR and SR views, but all uniformly apply SR across every image. In contrast, our key insight is that close-up LR views may contain high-frequency information for regions also captured in more distant views, and that we can use the camera pose relative to scene geometry to inform where to add SR content. Building from this insight, we propose SplatSuRe, a method that selectively applies SR content only in undersampled regions lacking high-frequency supervision, yielding sharper and more consistent results. Across Tanks & Temples, Deep Blending and Mip-NeRF 360, our approach surpasses baselines in both fidelity and perceptual quality. Notably, our gains are most significant in localized foreground regions where higher detail is desired.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02013v1">ManualVLA: A Unified VLA Model for Chain-of-Thought Manual Generation and Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models have recently emerged, demonstrating strong generalization in robotic scene understanding and manipulation. However, when confronted with long-horizon tasks that require defined goal states, such as LEGO assembly or object rearrangement, existing VLA models still face challenges in coordinating high-level planning with precise manipulation. Therefore, we aim to endow a VLA model with the capability to infer the "how" process from the "what" outcomes, transforming goal states into executable procedures. In this paper, we introduce ManualVLA, a unified VLA framework built upon a Mixture-of-Transformers (MoT) architecture, enabling coherent collaboration between multimodal manual generation and action execution. Unlike prior VLA models that directly map sensory inputs to actions, we first equip ManualVLA with a planning expert that generates intermediate manuals consisting of images, position prompts, and textual instructions. Building upon these multimodal manuals, we design a Manual Chain-of-Thought (ManualCoT) reasoning process that feeds them into the action expert, where each manual step provides explicit control conditions, while its latent representation offers implicit guidance for accurate manipulation. To alleviate the burden of data collection, we develop a high-fidelity digital-twin toolkit based on 3D Gaussian Splatting, which automatically generates manual data for planning expert training. ManualVLA demonstrates strong real-world performance, achieving an average success rate 32% higher than the previous hierarchical SOTA baseline on LEGO assembly and object rearrangement tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.12168v3">Sketch-guided Cage-based 3D Gaussian Splatting Deformation</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 10 pages, 9 figures, accepted at WACV 26, project page: https://tianhaoxie.github.io/project/gs_deform/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (GS) is one of the most promising novel 3D representations that has received great interest in computer graphics and computer vision. While various systems have introduced editing capabilities for 3D GS, such as those guided by text prompts, fine-grained control over deformation remains an open challenge. In this work, we present a novel sketch-guided 3D GS deformation system that allows users to intuitively modify the geometry of a 3D GS model by drawing a silhouette sketch from a single viewpoint. Our approach introduces a new deformation method that combines cage-based deformations with a variant of Neural Jacobian Fields, enabling precise, fine-grained control. Additionally, it leverages large-scale 2D diffusion priors and ControlNet to ensure the generated deformations are semantically plausible. Through a series of experiments, we demonstrate the effectiveness of our method and showcase its ability to animate static 3D GS models as one of its key applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22939v2">DenoiseGS: Gaussian Reconstruction Model for Burst Denoising</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 Update Abstract
    </div>
    <details class="paper-abstract">
      Burst denoising methods are crucial for enhancing images captured on handheld devices, but they often struggle with large motion or suffer from prohibitive computational costs. In this paper, we propose DenoiseGS, the first framework to leverage the efficiency of 3D Gaussian Splatting for burst denoising. Our approach addresses two key challenges when applying feedforward Gaussian reconsturction model to noisy inputs: the degradation of Gaussian point clouds and the loss of fine details. To this end, we propose a Gaussian self-consistency (GSC) loss, which regularizes the geometry predicted from noisy inputs with high-quality Gaussian point clouds. These point clouds are generated from clean inputs by the same model that we are training, thereby alleviating potential bias or domain gaps. Additionally, we introduce a log-weighted frequency (LWF) loss to strengthen supervision within the spectral domain, effectively preserving fine-grained details. The LWF loss adaptively weights frequency discrepancies in a logarithmic manner, emphasizing challenging high-frequency details. Extensive experiments demonstrate that DenoiseGS significantly exceeds the state-of-the-art NeRF-based methods on both burst denoising and novel view synthesis under noisy conditions, while achieving 250$\times$ faster inference speed. Code and models are released at https://github.com/yscheng04/DenoiseGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22056v2">EAST: Environment-Aware Stylized Transition Along the Reality-Virtuality Continuum</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      In the Virtual Reality (VR) gaming industry, maintaining immersion during real-world interruptions remains a challenge, particularly during transitions along the reality-virtuality continuum (RVC). Existing methods tend to rely on digital replicas or simple visual transitions, neglecting to address the aesthetic discontinuities between real and virtual environments, especially in highly stylized VR games. This paper introduces the Environment-Aware Stylized Transition (EAST) framework, which employs a novel style-transferred 3D Gaussian Splatting (3DGS) technique to transfer real-world interruptions into the virtual environment with seamless aesthetic consistency. Rather than merely transforming the real world into game-like visuals, EAST minimizes the disruptive impact of interruptions by integrating real-world elements within the framework. Qualitative user studies demonstrate significant enhancements in cognitive comfort and emotional continuity during transitions, while quantitative experiments highlight EAST's ability to maintain visual coherence across diverse VR styles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01329v1">TagSplat: Topology-Aware Gaussian Splatting for Dynamic Mesh Modeling and Tracking</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Topology-consistent dynamic model sequences are essential for applications such as animation and model editing. However, existing 4D reconstruction methods face challenges in generating high-quality topology-consistent meshes. To address this, we propose a topology-aware dynamic reconstruction framework based on Gaussian Splatting. We introduce a Gaussian topological structure that explicitly encodes spatial connectivity. This structure enables topology-aware densification and pruning, preserving the manifold consistency of the Gaussian representation. Temporal regularization terms further ensure topological coherence over time, while differentiable mesh rasterization improves mesh quality. Experimental results demonstrate that our method reconstructs topology-consistent mesh sequences with significantly higher accuracy than existing approaches. Moreover, the resulting meshes enable precise 3D keypoint tracking. Project page: https://haza628.github.io/tagSplat/
    </details>
</div>
