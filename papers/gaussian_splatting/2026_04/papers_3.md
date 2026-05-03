# gaussian splatting - 2026_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02586v1">TrackerSplat: Exploiting Point Tracking for Fast and Robust Dynamic 3D Gaussians Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-02
      | 💬 11 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3DGS) have demonstrated its potential for efficient and photorealistic 3D reconstructions, which is crucial for diverse applications such as robotics and immersive media. However, current Gaussian-based methods for dynamic scene reconstruction struggle with large inter-frame displacements, leading to artifacts and temporal inconsistencies under fast object motions. To address this, we introduce \textit{TrackerSplat}, a novel method that integrates advanced point tracking methods to enhance the robustness and scalability of 3DGS for dynamic scene reconstruction. TrackerSplat utilizes off-the-shelf point tracking models to extract pixel trajectories and triangulate per-view pixel trajectories onto 3D Gaussians to guide the relocation, rotation, and scaling of Gaussians before training. This strategy effectively handles large displacements between frames, dramatically reducing the fading and recoloring artifacts prevalent in prior methods. By accurately positioning Gaussians prior to gradient-based optimization, TrackerSplat overcomes the quality degradation associated with large frame gaps when processing multiple adjacent frames in parallel across multiple devices, thereby boosting reconstruction throughput while preserving rendering quality. Experiments on real-world datasets confirm the robustness of TrackerSplat in challenging scenarios with significant displacements, achieving superior throughput under parallel settings and maintaining visual quality compared to baselines. The code is available at https://github.com/yindaheng98/TrackerSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23292v3">FACT-GS: Frequency-Aligned Complexity-Aware Texture Reparameterization for 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-02
      | 💬 11 pages, 6 figures, CVPR 2026 Findings track. Project page: https://tianhaoxie.github.io/project/FACT-GS/
    </div>
    <details class="paper-abstract">
      Realistic scene appearance modeling has advanced rapidly with Gaussian Splatting, which enables real-time, high-quality rendering. Recent advances introduced per-primitive textures that incorporate spatial color variations within each Gaussian, improving their expressiveness. However, texture-based Gaussians parameterize appearance with a uniform per-Gaussian sampling grid, allocating equal sampling density regardless of local visual complexity, which leads to inefficient texture space utilization. We introduce FACT-GS, a Frequency-Aligned Complexity-aware Texture Gaussian Splatting framework that allocates texture sampling density according to local visual frequency. Grounded in adaptive sampling theory, FACT-GS reformulates texture parameterization as a differentiable sampling-density allocation problem, replacing the uniform textures with a learnable frequency-aware allocation strategy implemented via a deformation field whose Jacobian modulates local sampling density. Built on 2D Gaussian Splatting, FACT-GS performs non-uniform sampling on fixed-resolution texture grids, preserving real-time performance while recovering sharper high-frequency details under the same parameter budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02003v1">ProDiG: Progressive Diffusion-Guided Gaussian Splatting for Aerial to Ground Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-02
    </div>
    <details class="paper-abstract">
      Generating ground-level views and coherent 3D site models from aerial-only imagery is challenging due to extreme viewpoint changes, missing intermediate observations, and large scale variations. Existing methods either refine renderings post-hoc, often producing geometrically inconsistent results, or rely on multi-altitude ground-truth, which is rarely available. Gaussian Splatting and diffusion-based refinements improve fidelity under small variations but fail under wide aerial-to-ground gaps. To address these limitations, we introduce ProDiG (Progressive Altitude Gaussian Splatting), a diffusion-guided framework that progressively transforms aerial 3D representations toward ground-level fidelity. ProDiG synthesizes intermediate-altitude views and refines the Gaussian representation at each stage using a geometry-aware causal attention module that injects epipolar structure into reference-view diffusion. A distance-adaptive Gaussian module dynamically adjusts Gaussian scale and opacity based on camera distance, ensuring stable reconstruction across large viewpoint gaps. Together, these components enable progressive, geometrically grounded refinement without requiring additional ground-truth viewpoints. Extensive experiments on synthetic and real-world datasets demonstrate that ProDiG produces visually realistic ground-level renderings and coherent 3D geometry, significantly outperforming existing approaches in terms of visual quality, geometric consistency, and robustness to extreme viewpoint changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01994v1">Resonance4D: Frequency-Domain Motion Supervision for Preset-Free Physical Parameter Learning in 4D Dynamic Physical Scene Simulation</a></div>
    <div class="paper-meta">
      📅 2026-04-02
    </div>
    <details class="paper-abstract">
      Physics-driven 4D dynamic simulation from static 3D scenes remains constrained by an overlooked contradiction: reliable motion supervision often relies on online video diffusion or optical-flow pipelines whose computational cost exceeds that of the simulator itself. Existing methods further simplify inverse physical modeling by optimizing only partial material parameters, limiting realism in scenes with complex materials and dynamics. We present Resonance4D, a physics-driven 4D dynamic simulation framework that couples 3D Gaussian Splatting with the Material Point Method through lightweight yet physically expressive supervision. Our key insight is that dynamic consistency can be enforced without dense temporal generation by jointly constraining motion in complementary domains. To this end, we introduce Dual-domain Motion Supervision (DMS), which combines spatial structural consistency for local deformation with frequency-domain spectral consistency for oscillatory and global dynamic patterns, substantially reducing training cost and memory overhead while preserving physically meaningful motion cues. To enable stable full-parameter physical recovery, we further combine zero-shot text-prompted segmentation with simulation-guided initialization to automatically decompose Gaussians into object-part-level regions and support joint optimization of full material parameters. Experiments on both synthetic and real scenes show that Resonance4D achieves strong physical fidelity and motion consistency while reducing peak GPU memory from over 35\,GB to around 20\,GB, enabling high-fidelity physics-driven 4D simulation on a single consumer-grade GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17092v3">SPAGS: Sparse-View Articulated Object Reconstruction from Single State via Planar Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-02
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Articulated objects are ubiquitous in daily environments, and their 3D reconstruction holds great significance across various fields. However, existing articulated object reconstruction methods typically require costly inputs such as multi-stage and multi-view observations. To address the limitations, we propose a category-agnostic articulated object reconstruction framework via planar Gaussian Splatting, which only uses sparse-view RGB images from a single state. Specifically, we first introduce a Gaussian information field to perceive the optimal sparse viewpoints from candidate camera poses. To ensure precise geometric fidelity, we constrain traditional 3D Gaussians into planar primitives, facilitating accurate normal and depth estimation. The planar Gaussians are then optimized in a coarse-to-fine manner, regularized by depth smoothness and few-shot diffusion priors. Furthermore, we leverage a Vision-Language Model (VLM) via visual prompting to achieve open-vocabulary part segmentation and joint parameter estimation. Extensive experiments on both synthetic and real-world datasets demonstrate that our approach significantly outperforms existing baselines, achieving superior part-level surface reconstruction fidelity. Code and data are provided in the supplementary material.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01884v1">GS^2: Graph-based Spatial Distribution Optimization for Compact 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated breakthrough performance in novel view synthesis and real-time rendering. Nevertheless, its practicality is constrained by the high memory cost due to a huge number of Gaussian points. Many pruning-based 3DGS variants have been proposed for memory saving, but often compromise spatial consistency and may lead to rendering artifacts. To address this issue, we propose graph-based spatial distribution optimization for compact 3D Gaussian Splatting (GS\textasciicircum2), which enhances reconstruction quality by optimizing the spatial distribution of Gaussian points. Specifically, we introduce an evidence lower bound (ELBO)-based adaptive densification strategy that automatically controls the densification process. In addition, an opacity-aware progressive pruning strategy is proposed to further reduce memory consumption by dynamically removing low-opacity Gaussian points. Furthermore, we propose a graph-based feature encoding module to adjust the spatial distribution via feature-guided point shifting. Extensive experiments validate that GS\textasciicircum2 achieves a compact Gaussian representation while delivering superior rendering quality. Compared with 3DGS, it achieves higher PSNR with only about 12.5\% Gaussian points. Furthermore, it outperforms all compared baselines in both rendering quality and memory efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01844v1">FaCT-GS: Fast and Scalable CT Reconstruction with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-02
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has emerged as a dominating technique for image rendering and has quickly been adapted for the X-ray Computed Tomography (CT) reconstruction task. However, despite being on par or better than many of its predecessors, the benefits of GS are typically not substantial enough to motivate a transition from well-established reconstruction algorithms. This paper addresses the most significant remaining limitations of the GS-based approach by introducing FaCT-GS, a framework for fast and flexible CT reconstruction. Enabled by an in-depth optimization of the voxelization and rasterization pipelines, our new method is significantly faster than its predecessors and scales well with projection and output volume size. Furthermore, the improved voxelization enables rapid fitting of Gaussians to pre-existing volumes, which can serve as a prior for warm-starting the reconstruction, or simply as an alternative, compressed representation. FaCT-GS is over 4X faster than the State of the Art GS CT reconstruction on standard 512x512 projections, and over 13X faster on 2k projections. Implementation available at: https://github.com/PaPieta/fact-gs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19834v2">Fourier Splatting: Generalized Fourier encoded primitives for scalable radiance fields</a></div>
    <div class="paper-meta">
      📅 2026-04-02
    </div>
    <details class="paper-abstract">
      Novel view synthesis has recently been revolutionized by 3D Gaussian Splatting (3DGS), which enables real-time rendering through explicit primitive rasterization. However, existing methods tie visual fidelity strictly to the number of primitives: quality downscaling is achieved only through pruning primitives. We propose the first inherently scalable primitive for radiance field rendering. Fourier Splatting employs scalable primitives with arbitrary closed shapes obtained by parameterizing planar surfels with Fourier encoded descriptors. This formulation allows a single trained model to be rendered at varying levels of detail simply by truncating Fourier coefficients at runtime. To facilitate stable optimization, we employ a straight-through estimator for gradient extension beyond the primitive boundary, and introduce HYDRA, a densification strategy that decomposes complex primitives into simpler constituents within the MCMC framework. Our method achieves state-of-the-art rendering quality among planar-primitive frameworks and comparable perceptual metrics compared to leading volumetric representations on standard benchmarks, providing a versatile solution for bandwidth-constrained high-fidelity rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.22279v2">Learning Fine-Grained Geometry for Sparse-View Splatting via Cascade Depth Loss</a></div>
    <div class="paper-meta">
      📅 2026-04-02
    </div>
    <details class="paper-abstract">
      Novel view synthesis is a fundamental task in 3D computer vision that aims to reconstruct photorealistic images from novel viewpoints given a set of posed images. However, reconstruction quality degrades sharply under sparse-view conditions due to insufficient geometric cues. Existing methods, including Neural Radiance Fields (NeRF) and more recent 3D Gaussian Splatting (3DGS), often exhibit blurred details and structural artifacts when trained from sparse observations. Recent works have identified rendered depth quality as a key factor in mitigating these artifacts, as it directly affects geometric accuracy and view consistency. However, effectively leveraging depth under sparse views remains challenging. Depth priors can be noisy or misaligned with rendered geometry, and single-scale supervision often fails to capture both global structure and fine details. To address these challenges, we introduce Hierarchical Depth-Guided Splatting (HDGS), a depth supervision framework that progressively refines geometry from coarse to fine levels. Central to HDGS is our novel Cascade Pearson Correlation Loss (CPCL), which enforces consistency between rendered and estimated depth priors across multiple spatial scales. By enforcing multi-scale depth consistency, our method improves structural fidelity in sparse-view reconstruction. Experiments on LLFF and DTU demonstrate state-of-the-art performance under sparse-view settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01678v1">Director: Instance-aware Gaussian Splatting for Dynamic Scene Modeling and Understanding</a></div>
    <div class="paper-meta">
      📅 2026-04-02
      | 💬 Project page: https://caiyw2023.github.io/Director/
    </div>
    <details class="paper-abstract">
      Volumetric video seeks to model dynamic scenes as temporally coherent 4D representations. While recent Gaussian-based approaches achieve impressive rendering fidelity, they primarily emphasize appearance but are largely agnostic to instance-level structure, limiting stable tracking and semantic reasoning in highly dynamic scenarios. In this paper, we present Director, a unified spatio-temporal Gaussian representation that jointly models human performance, high-fidelity rendering, and instance-level semantics. Our key insight is that embedding instance-consistent semantics naturally complements 4D modeling, enabling more accurate scene decomposition while supporting robust dynamic scene understanding. To this end, we leverage temporally aligned instance masks and sentence embeddings derived from Multimodal Large Language Models to supervise the learnable semantic features of each Gaussian via two MLP decoders, enabling language-aligned 4D representations and enforcing identity consistency over time. To enhance temporal stability, we bridge 2D optical flow with 4D Gaussians and finetune their motions, yielding reliable initialization and reducing drift. For the training, we further introduce a geometry-aware SDF constraints, along with regularization terms that enforces surface continuity, enhancing temporal coherence in dynamic foreground modeling. Experiments demonstrate that Director achieves temporally coherent 4D reconstructions while simultaneously enabling instance segmentation and open-vocabulary querying.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01605v1">F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling</a></div>
    <div class="paper-meta">
      📅 2026-04-02
      | 💬 Accepted to the CVPR 2026 SPAR-3D Workshop
    </div>
    <details class="paper-abstract">
      We present F3DGS, a federated 3D Gaussian Splatting framework for decentralized multi-agent 3D reconstruction. Existing 3DGS pipelines assume centralized access to all observations, which limits their applicability in distributed robotic settings where agents operate independently, and centralized data aggregation may be restricted. Directly extending centralized training to multi-agent systems introduces communication overhead and geometric inconsistency. F3DGS first constructs a shared geometric scaffold by registering locally merged LiDAR point clouds from multiple clients to initialize a global 3DGS model. During federated optimization, Gaussian positions are fixed to preserve geometric alignment, while each client updates only appearance-related attributes, including covariance, opacity, and spherical harmonic coefficients. The server aggregates these updates using visibility-aware aggregation, weighting each client's contribution by how frequently it observed each Gaussian, resolving the partial-observability challenge inherent to multi-agent exploration. To evaluate decentralized reconstruction, we collect a multi-sequence indoor dataset with synchronized LiDAR, RGB, and IMU measurements. Experiments show that F3DGS achieves reconstruction quality comparable to centralized training while enabling distributed optimization across agents. The dataset, development kit, and source code will be publicly released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20331v3">GVGS: Gaussian Visibility-Aware Multi-View Geometry for Accurate Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables efficient rendering, yet accurate surface reconstruction remains challenging due to unreliable geometric supervision. Existing approaches predominantly rely on depth-based reprojection to infer visibility and enforce multi-view consistency, leading to a fundamental circular dependency: visibility estimation requires accurate depth, while depth supervision itself is conditioned on visibility. In this work, we revisit multi-view geometric supervision from the perspective of visibility modeling. Instead of inferring visibility from pixel-wise depth consistency, we explicitly model visibility at the level of Gaussian primitives. We introduce a Gaussian visibility-aware multi-view geometric consistency (GVMV) formulation, which aggregates cross-view visibility of shared Gaussians to construct reliable supervision over co-visible regions. To further incorporate monocular priors, we propose a progressive quadtree-calibrated depth alignment (QDC) strategy that performs block-wise affine calibration under visibility-aware guidance, effectively mitigating scale ambiguity while preserving local geometric structures. Extensive experiments on DTU and Tanks and Temples demonstrate that our method consistently improves reconstruction accuracy over prior Gaussian-based approaches. Our code is fully open-sourced and available at an anonymous repository: https://github.com/GVGScode/GVGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01551v1">ColorGradedGaussians: Palette-Based Color Grading for 3D Gaussian Splatting via View-Space Sparse Decomposition</a></div>
    <div class="paper-meta">
      📅 2026-04-02
      | 💬 9 pages, 2 figure pages
    </div>
    <details class="paper-abstract">
      Professional color editing requires precise control over both color (hue and saturation) and lightness, ideally through separate, independent controls. We present a real-time interactive color editing framework for 3D Gaussian Splatting (3DGS) that enables palette-based recoloring, per-palette tone curves for color-aware lightness adjustment, and accurate pixel-level constraints -- capabilities unavailable in prior palette-based 3DGS methods. Existing approaches decompose colors at the primitive level, optimizing per-Gaussian palette weights before splatting. However, sparse primitive-level weights do not guarantee sparse pixel-level decompositions after alpha-blending, causing palette edits to affect unintended regions and degrading editing quality. We address this through view-space palette decomposition, splatting weights instead of colors to optimize the observable appearance of the scene. We introduce a geometric loss using inverse barycentric coordinates to enforce consistent sparsity patterns, ensuring similar colors share similar decompositions. Our approach achieves superior editing quality compared to primitive-space methods, enabling professional color grading workflows for 3DGS scenes with real-time interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01388v1">LESV: Language Embedded Sparse Voxel Fusion for Open-Vocabulary 3D Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2026-04-01
    </div>
    <details class="paper-abstract">
      Recent advancements in open-vocabulary 3D scene understanding heavily rely on 3D Gaussian Splatting (3DGS) to register vision-language features into 3D space. However, we identify two critical limitations in these approaches: the spatial ambiguity arising from unstructured, overlapping Gaussians which necessitates probabilistic feature registration, and the multi-level semantic ambiguity caused by pooling features over object-level masks, which dilutes fine-grained details. To address these challenges, we present a novel framework that leverages Sparse Voxel Rasterization (SVRaster) as a structured, disjoint geometry representation. By regularizing SVRaster with monocular depth and normal priors, we establish a stable geometric foundation. This enables a deterministic, confidence-aware feature registration process and suppresses the semantic bleeding artifact common in 3DGS. Furthermore, we resolve multi-level ambiguity by exploiting the emerging dense alignment properties of foundation model AM-RADIO, avoiding the computational overhead of hierarchical training methods. Our approach achieves state-of-the-art performance on Open Vocabulary 3D Object Retrieval and Point Cloud Understanding benchmarks, particularly excelling on fine-grained queries where registration methods typically fail.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28431v3">LG-HCC: Local Geometry-Aware Hierarchical Context Compression for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-01
      | 💬 10
    </div>
    <details class="paper-abstract">
      Although 3D Gaussian Splatting (3DGS) enables high-fidelity real-time rendering, its prohibitive storage overhead severely hinders practical deployment. Recent anchor-based 3DGS compression schemes reduce gaussian redundancy through some advanced context models. However, they overlook explicit geometric dependencies, leading to structural degradation and suboptimal ratedistortion performance. In this paper, we propose a Local Geometry-aware Hierarchical Context Compression framework for 3DGS(LG-HCC) that incorporates inter-anchor geometric correlations into anchor pruning and entropy coding for compact representation. Specifically, we introduce an Neighborhood-Aware Anchor Pruning (NAAP) strategy, which evaluates anchor importance via weighted neighborhood feature aggregation and then merges low-contribution anchors into salient neighbors, yielding a compact yet geometry-consistent anchor set. Moreover, we further develop a hierarchical entropy coding scheme, in which coarse-to-fine priors are exploited through a lightweight Geometry-Guided Convolution(GG-Conv) operator to enable spatially adaptive context modeling and rate-distortion optimization. Extensive experiments show that LG-HCC effectively alleviates structural preservation issues,achieving superior geometric integrity and rendering fidelity while reducing storage by up to 30.85x compared to the Scaffold-GS baseline on the Mip-NeRF360 dataset
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01030v1">Diff3R: Feed-forward 3D Gaussian Splatting with Uncertainty-aware Differentiable Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-01
      | 💬 Project page: https://liu115.github.io/diff3r, Video: https://www.youtube.com/watch?v=IxzNSAdUY70
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) present two main directions: feed-forward models offer fast inference in sparse-view settings, while per-scene optimization yields high-quality renderings but is computationally expensive. To combine the benefits of both, we introduce Diff3R, a novel framework that explicitly bridges feed-forward prediction and test-time optimization. By incorporating a differentiable 3DGS optimization layer directly into the training loop, our network learns to predict an optimal initialization for test-time optimization rather than a conventional zero-shot result. To overcome the computational cost of backpropagating through the optimization steps, we propose computing gradients via the Implicit Function Theorem and a scalable, matrix-free PCG solver tailored for 3DGS optimization. Additionally, we incorporate a data-driven uncertainty model into the optimization process by adaptively controlling how much the parameters are allowed to change during optimization. This approach effectively mitigates overfitting in under-constrained regions and increases robustness against input outliers. Since our proposed optimization layer is model-agnostic, we show that it can be seamlessly integrated into existing feed-forward 3DGS architectures for both pose-given and pose-free methods, providing improvements for test-time optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09997v2">CLoD-GS: Continuous Level-of-Detail via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-01
      | 💬 Accepted by ICLR 2026 poster
    </div>
    <details class="paper-abstract">
      Level of Detail (LoD) is a fundamental technique in real-time computer graphics for managing the rendering costs of complex scenes while preserving visual fidelity. Traditionally, LoD is implemented using discrete levels (DLoD), where multiple, distinct versions of a model are swapped out at different distances. This long-standing paradigm, however, suffers from two major drawbacks: it requires significant storage for multiple model copies and causes jarring visual ``popping" artifacts during transitions, degrading the user experience. We argue that the explicit, primitive-based nature of the emerging 3D Gaussian Splatting (3DGS) technique enables a more ideal paradigm: Continuous LoD (CLoD). A CLoD approach facilitates smooth, seamless quality scaling within a single, unified model, thereby circumventing the core problems of DLOD. To this end, we introduce CLoD-GS, a framework that integrates a continuous LoD mechanism directly into a 3DGS representation. Our method introduces a learnable, distance-dependent decay parameter for each Gaussian primitive, which dynamically adjusts its opacity based on viewpoint proximity. This allows for the progressive and smooth filtering of less significant primitives, effectively creating a continuous spectrum of detail within one model. To train this model to be robust across all distances, we introduce a virtual distance scaling mechanism and a novel coarse-to-fine training strategy with rendered point count regularization. Our approach not only eliminates the storage overhead and visual artifacts of discrete methods but also reduces the primitive count and memory footprint of the final model. Extensive experiments demonstrate that CLoD-GS achieves smooth, quality-scalable rendering from a single model, delivering high-fidelity results across a wide range of performance targets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.00928v1">Autoregressive Appearance Prediction for 3D Gaussian Avatars</a></div>
    <div class="paper-meta">
      📅 2026-04-01
      | 💬 Project Page: https://steimich96.github.io/AAP-3DGA/
    </div>
    <details class="paper-abstract">
      A photorealistic and immersive human avatar experience demands capturing fine, person-specific details such as cloth and hair dynamics, subtle facial expressions, and characteristic motion patterns. Achieving this requires large, high-quality datasets, which often introduce ambiguities and spurious correlations when very similar poses correspond to different appearances. Models that fit these details during training can overfit and produce unstable, abrupt appearance changes for novel poses. We propose a 3D Gaussian Splatting avatar model with a spatial MLP backbone that is conditioned on both pose and an appearance latent. The latent is learned during training by an encoder, yielding a compact representation that improves reconstruction quality and helps disambiguate pose-driven renderings. At driving time, our predictor autoregressively infers the latent, producing temporally smooth appearance evolution and improved stability. Overall, our method delivers a robust and practical path to high-fidelity, stable avatar driving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.00804v1">Compact Keyframe-Optimized Multi-Agent Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2026-04-01
    </div>
    <details class="paper-abstract">
      Efficient multi-agent 3D mapping is essential for robotic teams operating in unknown environments, but dense representations hinder real-time exchange over constrained communication links. In multi-agent Simultaneous Localization and Mapping (SLAM), systems typically rely on a centralized server to merge and optimize the local maps produced by individual agents. However, sharing these large map representations, particularly those generated by recent methods such as Gaussian Splatting, becomes a bottleneck in real-world scenarios with limited bandwidth. We present an improved multi-agent RGB-D Gaussian Splatting SLAM framework that reduces communication load while preserving map fidelity. First, we incorporate a compaction step into our SLAM system to remove redundant 3D Gaussians, without degrading the rendering quality. Second, our approach performs centralized loop closure computation without initial guess, operating in two modes: a pure rendered-depth mode that requires no data beyond the 3D Gaussians, and a camera-depth mode that includes lightweight depth images for improved registration accuracy and additional Gaussian pruning. Evaluation on both synthetic and real-world datasets shows up to 85-95\% reduction in transmitted data compared to state-of-the-art approaches in both modes, bringing 3D Gaussian multi-agent SLAM closer to practical deployment in real-world scenarios. Code: https://github.com/lemonci/coko-slam
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.00648v1">DirectFisheye-GS: Enabling Native Fisheye Input in Gaussian Splatting with Cross-View Joint Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-01
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has enabled efficient 3D scene reconstruction from everyday images with real-time, high-fidelity rendering, greatly advancing VR/AR applications. Fisheye cameras, with their wider field of view (FOV), promise high-quality reconstructions from fewer inputs and have recently attracted much attention. However, since 3DGS relies on rasterization, most subsequent works involving fisheye camera inputs first undistort images before training, which introduces two problems: 1) Black borders at image edges cause information loss and negate the fisheye's large FOV advantage; 2) Undistortion's stretch-and-interpolate resampling spreads each pixel's value over a larger area, diluting detail density -- causes 3DGS overfitting these low-frequency zones, producing blur and floating artifacts. In this work, we integrate fisheye camera model into the original 3DGS framework, enabling native fisheye image input for training without preprocessing. Despite correct modeling, we observed that the reconstructed scenes still exhibit floaters at image edges: Distortion increases toward the periphery, and 3DGS's original per-iteration random-selecting-view optimization ignores the cross-view correlations of a Gaussian, leading to extreme shapes (e.g., oversized or elongated) that degrade reconstruction quality. To address this, we introduce a feature-overlap-driven cross-view joint optimization strategy that establishes consistent geometric and photometric constraints across views-a technique equally applicable to existing pinhole-camera-based pipelines. Our DirectFisheye-GS matches or surpasses state-of-the-art performance on public datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10075v2">Thinking Like Van Gogh: Structure-Aware Style Transfer via Flow-Guided 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-01
      | 💬 7 pages, 8 figures
    </div>
    <details class="paper-abstract">
      In 1888, Vincent van Gogh wrote, "I am seeking exaggeration in the essential." This principle, amplifying structural form while suppressing photographic detail, lies at the core of Post-Impressionist art. However, most existing 3D style transfer methods invert this philosophy, treating geometry as a rigid substrate for surface-level texture projection. To authentically reproduce Post-Impressionist stylization, geometric abstraction must be embraced as the primary vehicle of expression. We propose a flow-guided geometric advection framework for 3D Gaussian Splatting (3DGS) that operationalizes this principle in a mesh-free setting. Our method extracts directional flow fields from 2D paintings and back-propagates them into 3D space, rectifying Gaussian primitives to form flow-aligned brushstrokes that conform to scene topology without relying on explicit mesh priors. This enables expressive structural deformation driven directly by painterly motion rather than photometric constraints. Our contributions are threefold: (1) a projection-based, mesh-free flow guidance mechanism that transfers 2D artistic motion into 3D Gaussian geometry; (2) a luminance-structure decoupling strategy that isolates geometric deformation from color optimization, mitigating artifacts during aggressive structural abstraction; and (3) a VLM-as-a-Judge evaluation framework that assesses artistic authenticity through aesthetic judgment instead of conventional pixel-level metrics, explicitly addressing the subjective nature of artistic stylization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.00538v1">TRiGS: Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-01
      | 💬 Project page: https://wwwjjn.github.io/TRiGS-project_page/
    </div>
    <details class="paper-abstract">
      Recent 4D Gaussian Splatting (4DGS) methods achieve impressive dynamic scene reconstruction but often rely on piecewise linear velocity approximations and short temporal windows. This disjointed modeling leads to severe temporal fragmentation, forcing primitives to be repeatedly eliminated and regenerated to track complex nonlinear dynamics. This makeshift approximation eliminates the long-term temporal identity of objects and causes an inevitable proliferation of Gaussians, hindering scalability to extended video sequences. To address this, we propose TRiGS, a novel 4D representation that utilizes unified, continuous geometric transformations. By integrating $SE(3)$ transformations, hierarchical Bezier residuals, and learnable local anchors, TRiGS models geometrically consistent rigid motions for individual primitives. This continuous formulation preserves temporal identity and effectively mitigates unbounded memory growth. Extensive experiments demonstrate that TRiGS achieves high fidelity rendering on standard benchmarks while uniquely scaling to extended video sequences (e.g., 600 to 1200 frames) without severe memory bottlenecks, significantly outperforming prior works in temporal stability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.00509v1">RT-GS: Gaussian Splatting with Reflection and Transmittance Primitives</a></div>
    <div class="paper-meta">
      📅 2026-04-01
    </div>
    <details class="paper-abstract">
      Gaussian Splatting is a powerful tool for reconstructing diffuse scenes, but it struggles to simultaneously model specular reflections and the appearance of objects behind semi-transparent surfaces. These specular reflections and transmittance are essential for realistic novel view synthesis, and existing methods do not properly incorporate the underlying physical processes to simulate them. To address this issue, we propose RT-GS, a unified framework that integrates a microfacet material model and ray tracing to jointly model specular reflection and transmittance in Gaussian Splatting. We accomplish this by using separate Gaussian primitives for reflections and transmittance, which allow modeling distant reflections and reconstructing objects behind transparent surfaces concurrently. We utilize a differentiable ray tracing framework to obtain the specular reflection and transmittance appearance. Our experiments demonstrate that our method successfully produces reflections and recovers objects behind transparent surfaces in complex environments, achieving significant qualitative improvements over prior methods where these specular light interactions are prominent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18640v2">Geometric-Photometric Event-based 3D Gaussian Ray Tracing</a></div>
    <div class="paper-meta">
      📅 2026-04-01
      | 💬 15 pages, 12 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Event cameras offer a high temporal resolution over traditional frame-based cameras, which makes them suitable for motion and structure estimation. However, it has been unclear how event-based 3D Gaussian Splatting (3DGS) approaches could leverage fine-grained temporal information of sparse events. This work proposes GPERT, a framework to address the trade-off between accuracy and temporal resolution in event-based 3DGS. Our key idea is to decouple the rendering into two branches: event-by-event geometry (depth) rendering and snapshot-based radiance (intensity) rendering, by using ray-tracing and the image of warped events. The extensive evaluation shows that our method achieves state-of-the-art performance on the real-world datasets and competitive performance on the synthetic dataset. Also, the proposed method works without prior information (e.g., pretrained image reconstruction models) or COLMAP-based initialization, is more flexible in the event selection number, and achieves sharp reconstruction on scene edges with fast training time. We hope that this work deepens our understanding of the sparse nature of events for 3D reconstruction. https://github.com/e3ai/gpert
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.00494v1">ARGS: Auto-Regressive Gaussian Splatting via Parallel Progressive Next-Scale Prediction</a></div>
    <div class="paper-meta">
      📅 2026-04-01
    </div>
    <details class="paper-abstract">
      Auto-regressive frameworks for next-scale prediction of 2D images have demonstrated strong potential for producing diverse and sophisticated content by progressively refining a coarse input. However, extending this paradigm to 3D object generation remains largely unexplored. In this paper, we introduce auto-regressive Gaussian splatting (ARGS), a framework for making next-scale predictions in parallel for generation according to levels of detail. We propose a Gaussian simplification strategy and reverse the simplification to guide next-scale generation. Benefiting from the use of hierarchical trees, the generation process requires only \(\mathcal{O}(\log n)\) steps, where \(n\) is the number of points. Furthermore, we propose a tree-based transformer to predict the tree structure auto-regressively, allowing leaf nodes to attend to their internal ancestors to enhance structural consistency. Extensive experiments demonstrate that our approach effectively generates multi-scale Gaussian representations with controllable levels of detail, visual fidelity, and a manageable time consumption budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18739v2">Moving Light Adaptive Colonoscopy Reconstruction via Illumination-Attenuation-Aware 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-01
      | 💬 Accepted by ICME2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables real-time view synthesis in colonoscopy but assumes static illumination, making it incompatible with the strong photometric variations caused by the moving light source and camera. This mismatch leads existing methods to compensate for illumination attenuation with structure-violating Gaussians, degrading geometric fidelity. Prior work considers only distance-based attenuation and overlooks the physical characteristics of colonscopic lighting. In this paper, we propose ColIAGS, an improved 3DGS framework for colonoscopy. To mimic realistic appearance under varying illumination, we introduce a lighting model with two types of illumination attenuation factors. To satisfy this lighting model's approximation and effectively integrate it into the 3DGS framework, we design Improved Geometry Modeling to strengthen geometry details and Improved Appearance Modeling to implicitly optimize illumination attenuation solutions. Experimental results on standard benchmarks demonstrate that ColIAGS supports both high-quality novel-view synthesis and accurate geometry reconstruction, outperforming state-of-the-art methods in rendering fidelity and Depth MSE. Our code is available at https://github.com/haowang020110/ColIAGS.
    </details>
</div>
