# gaussian splatting - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18090v1">GeoSVR: Taming Sparse Voxels for Geometrically Accurate Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted at NeurIPS 2025 (Spotlight). Project page: https://fictionarry.github.io/GeoSVR-project/
    </div>
    <details class="paper-abstract">
      Reconstructing accurate surfaces with radiance fields has achieved remarkable progress in recent years. However, prevailing approaches, primarily based on Gaussian Splatting, are increasingly constrained by representational bottlenecks. In this paper, we introduce GeoSVR, an explicit voxel-based framework that explores and extends the under-investigated potential of sparse voxels for achieving accurate, detailed, and complete surface reconstruction. As strengths, sparse voxels support preserving the coverage completeness and geometric clarity, while corresponding challenges also arise from absent scene constraints and locality in surface refinement. To ensure correct scene convergence, we first propose a Voxel-Uncertainty Depth Constraint that maximizes the effect of monocular depth cues while presenting a voxel-oriented uncertainty to avoid quality degradation, enabling effective and robust scene constraints yet preserving highly accurate geometries. Subsequently, Sparse Voxel Surface Regularization is designed to enhance geometric consistency for tiny voxels and facilitate the voxel-based formation of sharp and accurate surfaces. Extensive experiments demonstrate our superior performance compared to existing methods across diverse challenging scenarios, excelling in geometric accuracy, detail preservation, and reconstruction completeness while maintaining high efficiency. Code is available at https://github.com/Fictionarry/GeoSVR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02045v2">Generating 360° Video is What You Need For a 3D Scene</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Generating 3D scenes is still a challenging task due to the lack of readily available scene data. Most existing methods only produce partial scenes and provide limited navigational freedom. We introduce a practical and scalable solution that uses 360{\deg} video as an intermediate scene representation, capturing the full-scene context and ensuring consistent visual content throughout the generation. We propose WorldPrompter, a generative pipeline that synthesizes traversable 3D scenes from text prompts. WorldPrompter incorporates a conditional 360{\deg} panoramic video generator, capable of producing a 128-frame video that simulates a person walking through and capturing a virtual environment. The resulting video is then reconstructed as Gaussian splats by a fast feedforward 3D reconstructor, enabling a true walkable experience within the 3D scene. Experiments demonstrate that our panoramic video generation model, trained with a mix of image and video data, achieves convincing spatial and temporal consistency for static scenes. This is validated by an average COLMAP matching rate of 94.6\%, allowing for high-quality panoramic Gaussian splat reconstruction and improved navigation throughout the scene. Qualitative and quantitative results also show it outperforms the state-of-the-art 360{\deg} video generators and 3D scene generation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15548v2">MS-GS: Multi-Appearance Sparse-View 3D Gaussian Splatting in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      In-the-wild photo collections often contain limited volumes of imagery and exhibit multiple appearances, e.g., taken at different times of day or seasons, posing significant challenges to scene reconstruction and novel view synthesis. Although recent adaptations of Neural Radiance Field (NeRF) and 3D Gaussian Splatting (3DGS) have improved in these areas, they tend to oversmooth and are prone to overfitting. In this paper, we present MS-GS, a novel framework designed with Multi-appearance capabilities in Sparse-view scenarios using 3DGS. To address the lack of support due to sparse initializations, our approach is built on the geometric priors elicited from monocular depth estimations. The key lies in extracting and utilizing local semantic regions with a Structure-from-Motion (SfM) points anchored algorithm for reliable alignment and geometry cues. Then, to introduce multi-view constraints, we propose a series of geometry-guided supervision at virtual views in a fine-grained and coarse scheme to encourage 3D consistency and reduce overfitting. We also introduce a dataset and an in-the-wild experiment setting to set up more realistic benchmarks. We demonstrate that MS-GS achieves photorealistic renderings under various challenging sparse-view and multi-appearance conditions and outperforms existing approaches significantly across different datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17889v1">GaussianPSL: A novel framework based on Gaussian Splatting for exploring the Pareto frontier in multi-criteria optimization</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Multi-objective optimization (MOO) is essential for solving complex real-world problems involving multiple conflicting objectives. However, many practical applications - including engineering design, autonomous systems, and machine learning - often yield non-convex, degenerate, or discontinuous Pareto frontiers, which involve traditional scalarization and Pareto Set Learning (PSL) methods that struggle to approximate accurately. Existing PSL approaches perform well on convex fronts but tend to fail in capturing the diversity and structure of irregular Pareto sets commonly observed in real-world scenarios. In this paper, we propose Gaussian-PSL, a novel framework that integrates Gaussian Splatting into PSL to address the challenges posed by non-convex Pareto frontiers. Our method dynamically partitions the preference vector space, enabling simple MLP networks to learn localized features within each region, which are then integrated by an additional MLP aggregator. This partition-aware strategy enhances both exploration and convergence, reduces sensi- tivity to initialization, and improves robustness against local optima. We first provide the mathematical formulation for controllable Pareto set learning using Gaussian Splat- ting. Then, we introduce the Gaussian-PSL architecture and evaluate its performance on synthetic and real-world multi-objective benchmarks. Experimental results demonstrate that our approach outperforms standard PSL models in learning irregular Pareto fronts while maintaining computational efficiency and model simplicity. This work offers a new direction for effective and scalable MOO under challenging frontier geometries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17864v1">ProDyG: Progressive Dynamic Scene Reconstruction via Gaussian Splatting from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Achieving truly practical dynamic 3D reconstruction requires online operation, global pose and map consistency, detailed appearance modeling, and the flexibility to handle both RGB and RGB-D inputs. However, existing SLAM methods typically merely remove the dynamic parts or require RGB-D input, while offline methods are not scalable to long video sequences, and current transformer-based feedforward methods lack global consistency and appearance details. To this end, we achieve online dynamic scene reconstruction by disentangling the static and dynamic parts within a SLAM system. The poses are tracked robustly with a novel motion masking strategy, and dynamic parts are reconstructed leveraging a progressive adaptation of a Motion Scaffolds graph. Our method yields novel view renderings competitive to offline methods and achieves on-par tracking with state-of-the-art dynamic SLAM methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17789v1">From Restoration to Reconstruction: Rethinking 3D Gaussian Splatting for Underwater Scenes</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Underwater image degradation poses significant challenges for 3D reconstruction, where simplified physical models often fail in complex scenes. We propose \textbf{R-Splatting}, a unified framework that bridges underwater image restoration (UIR) with 3D Gaussian Splatting (3DGS) to improve both rendering quality and geometric fidelity. Our method integrates multiple enhanced views produced by diverse UIR models into a single reconstruction pipeline. During inference, a lightweight illumination generator samples latent codes to support diverse yet coherent renderings, while a contrastive loss ensures disentangled and stable illumination representations. Furthermore, we propose \textit{Uncertainty-Aware Opacity Optimization (UAOO)}, which models opacity as a stochastic function to regularize training. This suppresses abrupt gradient responses triggered by illumination variation and mitigates overfitting to noisy or view-specific artifacts. Experiments on Seathru-NeRF and our new BlueCoral3D dataset demonstrate that R-Splatting outperforms strong baselines in both rendering quality and geometric accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17762v1">Neural-MMGS: Multi-modal Neural Gaussian Splats for Large-Scale Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      This paper proposes Neural-MMGS, a novel neural 3DGS framework for multimodal large-scale scene reconstruction that fuses multiple sensing modalities in a per-gaussian compact, learnable embedding. While recent works focusing on large-scale scene reconstruction have incorporated LiDAR data to provide more accurate geometric constraints, we argue that LiDAR's rich physical properties remain underexplored. Similarly, semantic information has been used for object retrieval, but could provide valuable high-level context for scene reconstruction. Traditional approaches append these properties to Gaussians as separate parameters, increasing memory usage and limiting information exchange across modalities. Instead, our approach fuses all modalities -- image, LiDAR, and semantics -- into a compact, learnable embedding that implicitly encodes optical, physical, and semantic features in each Gaussian. We then train lightweight neural decoders to map these embeddings to Gaussian parameters, enabling the reconstruction of each sensing modality with lower memory overhead and improved scalability. We evaluate Neural-MMGS on the Oxford Spires and KITTI-360 datasets. On Oxford Spires, we achieve higher-quality reconstructions, while on KITTI-360, our method reaches competitive results with less storage consumption compared with current approaches in LiDAR-based novel-view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11003v2">AD-GS: Alternating Densification for Sparse-Input 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 SIGGRAPH Asia 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown impressive results in real-time novel view synthesis. However, it often struggles under sparse-view settings, producing undesirable artifacts such as floaters, inaccurate geometry, and overfitting due to limited observations. We find that a key contributing factor is uncontrolled densification, where adding Gaussian primitives rapidly without guidance can harm geometry and cause artifacts. We propose AD-GS, a novel alternating densification framework that interleaves high and low densification phases. During high densification, the model densifies aggressively, followed by photometric loss based training to capture fine-grained scene details. Low densification then primarily involves aggressive opacity pruning of Gaussians followed by regularizing their geometry through pseudo-view consistency and edge-aware depth smoothness. This alternating approach helps reduce overfitting by carefully controlling model capacity growth while progressively refining the scene representation. Extensive experiments on challenging datasets demonstrate that AD-GS significantly improves rendering quality and geometric consistency compared to existing methods. The source code for our model can be found on our project page: https://gurutvapatle.github.io/publications/2025/ADGS.html .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17430v1">EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 16 pages, 18 figures, paper accepted at ICCV, 2025
    </div>
    <details class="paper-abstract">
      The field of Embodied AI predominantly relies on simulation for training and evaluation, often using either fully synthetic environments that lack photorealism or high-fidelity real-world reconstructions captured with expensive hardware. As a result, sim-to-real transfer remains a major challenge. In this paper, we introduce EmbodiedSplat, a novel approach that personalizes policy training by efficiently capturing the deployment environment and fine-tuning policies within the reconstructed scenes. Our method leverages 3D Gaussian Splatting (GS) and the Habitat-Sim simulator to bridge the gap between realistic scene capture and effective training environments. Using iPhone-captured deployment scenes, we reconstruct meshes via GS, enabling training in settings that closely approximate real-world conditions. We conduct a comprehensive analysis of training strategies, pre-training datasets, and mesh reconstruction techniques, evaluating their impact on sim-to-real predictivity in real-world scenarios. Experimental results demonstrate that agents fine-tuned with EmbodiedSplat outperform both zero-shot baselines pre-trained on large-scale real-world datasets (HM3D) and synthetically generated datasets (HSSD), achieving absolute success rate improvements of 20\% and 40\% on real-world Image Navigation task. Moreover, our approach yields a high sim-vs-real correlation (0.87--0.97) for the reconstructed meshes, underscoring its effectiveness in adapting policies to diverse environments with minimal effort. Project page: https://gchhablani.github.io/embodied-splat
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17390v1">FGGS-LiDAR: Ultra-Fast, GPU-Accelerated Simulation from General 3DGS Models to LiDAR</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has revolutionized photorealistic rendering, its vast ecosystem of assets remains incompatible with high-performance LiDAR simulation, a critical tool for robotics and autonomous driving. We present \textbf{FGGS-LiDAR}, a framework that bridges this gap with a truly plug-and-play approach. Our method converts \textit{any} pretrained 3DGS model into a high-fidelity, watertight mesh without requiring LiDAR-specific supervision or architectural alterations. This conversion is achieved through a general pipeline of volumetric discretization and Truncated Signed Distance Field (TSDF) extraction. We pair this with a highly optimized, GPU-accelerated ray-casting module that simulates LiDAR returns at over 500 FPS. We validate our approach on indoor and outdoor scenes, demonstrating exceptional geometric fidelity; By enabling the direct reuse of 3DGS assets for geometrically accurate depth sensing, our framework extends their utility beyond visualization and unlocks new capabilities for scalable, multimodal simulation. Our open-source implementation is available at https://github.com/TATP-233/FGGS-LiDAR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17329v1">SmokeSeer: 3D Gaussian Splatting for Smoke Removal and Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Project website: https://imaging.cs.cmu.edu/smokeseer
    </div>
    <details class="paper-abstract">
      Smoke in real-world scenes can severely degrade the quality of images and hamper visibility. Recent methods for image restoration either rely on data-driven priors that are susceptible to hallucinations, or are limited to static low-density smoke. We introduce SmokeSeer, a method for simultaneous 3D scene reconstruction and smoke removal from a video capturing multiple views of a scene. Our method uses thermal and RGB images, leveraging the fact that the reduced scattering in thermal images enables us to see through the smoke. We build upon 3D Gaussian splatting to fuse information from the two image modalities, and decompose the scene explicitly into smoke and non-smoke components. Unlike prior approaches, SmokeSeer handles a broad range of smoke densities and can adapt to temporally varying smoke. We validate our approach on synthetic data and introduce a real-world multi-view smoke dataset with RGB and thermal images. We provide open-source code and data at the project website.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17246v1">SPFSplatV2: Efficient Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views</a></div>
    <div class="paper-meta">
      📅 2025-09-21
    </div>
    <details class="paper-abstract">
      We introduce SPFSplatV2, an efficient feed-forward framework for 3D Gaussian splatting from sparse multi-view images, requiring no ground-truth poses during training and inference. It employs a shared feature extraction backbone, enabling simultaneous prediction of 3D Gaussian primitives and camera poses in a canonical space from unposed inputs. A masked attention mechanism is introduced to efficiently estimate target poses during training, while a reprojection loss enforces pixel-aligned Gaussian primitives, providing stronger geometric constraints. We further demonstrate the compatibility of our training framework with different reconstruction architectures, resulting in two model variants. Remarkably, despite the absence of pose supervision, our method achieves state-of-the-art performance in both in-domain and out-of-domain novel view synthesis, even under extreme viewpoint changes and limited image overlap, and surpasses recent methods that rely on geometric supervision for relative pose estimation. By eliminating dependence on ground-truth poses, our method offers the scalability to leverage larger and more diverse datasets. Code and pretrained models will be available on our project page: https://ranrhuang.github.io/spfsplatv2/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03526v3">Feed-Forward Bullet-Time Reconstruction of Dynamic Scenes from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 Project website: https://research.nvidia.com/labs/toronto-ai/bullet-timer/
    </div>
    <details class="paper-abstract">
      Recent advancements in static feed-forward scene reconstruction have demonstrated significant progress in high-quality novel view synthesis. However, these models often struggle with generalizability across diverse environments and fail to effectively handle dynamic content. We present BTimer (short for BulletTimer), the first motion-aware feed-forward model for real-time reconstruction and novel view synthesis of dynamic scenes. Our approach reconstructs the full scene in a 3D Gaussian Splatting representation at a given target ('bullet') timestamp by aggregating information from all the context frames. Such a formulation allows BTimer to gain scalability and generalization by leveraging both static and dynamic scene datasets. Given a casual monocular dynamic video, BTimer reconstructs a bullet-time scene within 150ms while reaching state-of-the-art performance on both static and dynamic scene datasets, even compared with optimization-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15376v3">DriveSplat: Decoupled Driving Scene Reconstruction with Geometry-enhanced Partitioned Neural Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-09-21
    </div>
    <details class="paper-abstract">
      In the realm of driving scenarios, the presence of rapidly moving vehicles, pedestrians in motion, and large-scale static backgrounds poses significant challenges for 3D scene reconstruction. Recent methods based on 3D Gaussian Splatting address the motion blur problem by decoupling dynamic and static components within the scene. However, these decoupling strategies overlook background optimization with adequate geometry relationships and rely solely on fitting each training view by adding Gaussians. Therefore, these models exhibit limited robustness in rendering novel views and lack an accurate geometric representation. To address the above issues, we introduce DriveSplat, a high-quality reconstruction method for driving scenarios based on neural Gaussian representations with dynamic-static decoupling. To better accommodate the predominantly linear motion patterns of driving viewpoints, a region-wise voxel initialization scheme is employed, which partitions the scene into near, middle, and far regions to enhance close-range detail representation. Deformable neural Gaussians are introduced to model non-rigid dynamic actors, whose parameters are temporally adjusted by a learnable deformation network. The entire framework is further supervised by depth and normal priors from pre-trained models, improving the accuracy of geometric structures. Our method has been rigorously evaluated on the Waymo and KITTI datasets, demonstrating state-of-the-art performance in novel-view synthesis for driving scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17083v1">HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-09-21
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has emerged as a powerful alternative to NeRF-based approaches, enabling real-time, high-quality novel view synthesis through explicit, optimizable 3D Gaussians. However, 3DGS suffers from significant memory overhead due to its reliance on per-Gaussian parameters to model view-dependent effects and anisotropic shapes. While recent works propose compressing 3DGS with neural fields, these methods struggle to capture high-frequency spatial variations in Gaussian properties, leading to degraded reconstruction of fine details. We present Hybrid Radiance Fields (HyRF), a novel scene representation that combines the strengths of explicit Gaussians and neural fields. HyRF decomposes the scene into (1) a compact set of explicit Gaussians storing only critical high-frequency parameters and (2) grid-based neural fields that predict remaining properties. To enhance representational capacity, we introduce a decoupled neural field architecture, separately modeling geometry (scale, opacity, rotation) and view-dependent color. Additionally, we propose a hybrid rendering scheme that composites Gaussian splatting with a neural field-predicted background, addressing limitations in distant scene representation. Experiments demonstrate that HyRF achieves state-of-the-art rendering quality while reducing model size by over 20 times compared to 3DGS and maintaining real-time performance. Our project page is available at https://wzpscott.github.io/hyrf/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17027v1">Efficient 3D Scene Reconstruction and Simulation from Sparse Endoscopic Views</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 Workshop Paper of AECAI@MICCAI 2025
    </div>
    <details class="paper-abstract">
      Surgical simulation is essential for medical training, enabling practitioners to develop crucial skills in a risk-free environment while improving patient safety and surgical outcomes. However, conventional methods for building simulation environments are cumbersome, time-consuming, and difficult to scale, often resulting in poor details and unrealistic simulations. In this paper, we propose a Gaussian Splatting-based framework to directly reconstruct interactive surgical scenes from endoscopic data while ensuring efficiency, rendering quality, and realism. A key challenge in this data-driven simulation paradigm is the restricted movement of endoscopic cameras, which limits viewpoint diversity. As a result, the Gaussian Splatting representation overfits specific perspectives, leading to reduced geometric accuracy. To address this issue, we introduce a novel virtual camera-based regularization method that adaptively samples virtual viewpoints around the scene and incorporates them into the optimization process to mitigate overfitting. An effective depth-based regularization is applied to both real and virtual views to further refine the scene geometry. To enable fast deformation simulation, we propose a sparse control node-based Material Point Method, which integrates physical properties into the reconstructed scene while significantly reducing computational costs. Experimental results on representative surgical data demonstrate that our method can efficiently reconstruct and simulate surgical scenes from sparse endoscopic views. Notably, our method takes only a few minutes to reconstruct the surgical scene and is able to produce physically plausible deformations in real-time with user-defined interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07493v2">Accurate and Complete Surface Reconstruction from 3D Gaussians via Direct SDF Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-21
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as a powerful paradigm for photorealistic view synthesis, representing scenes with spatially distributed Gaussian primitives. While highly effective for rendering, achieving accurate and complete surface reconstruction remains challenging due to the unstructured nature of the representation and the absence of explicit geometric supervision. In this work, we propose DiGS, a unified framework that embeds Signed Distance Field (SDF) learning directly into the 3DGS pipeline, thereby enforcing strong and interpretable surface priors. By associating each Gaussian with a learnable SDF value, DiGS explicitly aligns primitives with underlying geometry and improves cross-view consistency. To further ensure dense and coherent coverage, we design a geometry-guided grid growth strategy that adaptively distributes Gaussians along geometry-consistent regions under a multi-scale hierarchy. Extensive experiments on standard benchmarks, including DTU, Mip-NeRF 360, and Tanks& Temples, demonstrate that DiGS consistently improves reconstruction accuracy and completeness while retaining high rendering fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16922v1">PGSTalker: Real-Time Audio-Driven Talking Head Generation via 3D Gaussian Splatting with Pixel-Aware Density Control</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 Main paper (15 pages). Accepted for publication by ICONIP( International Conference on Neural Information Processing) 2025
    </div>
    <details class="paper-abstract">
      Audio-driven talking head generation is crucial for applications in virtual reality, digital avatars, and film production. While NeRF-based methods enable high-fidelity reconstruction, they suffer from low rendering efficiency and suboptimal audio-visual synchronization. This work presents PGSTalker, a real-time audio-driven talking head synthesis framework based on 3D Gaussian Splatting (3DGS). To improve rendering performance, we propose a pixel-aware density control strategy that adaptively allocates point density, enhancing detail in dynamic facial regions while reducing redundancy elsewhere. Additionally, we introduce a lightweight Multimodal Gated Fusion Module to effectively fuse audio and spatial features, thereby improving the accuracy of Gaussian deformation prediction. Extensive experiments on public datasets demonstrate that PGSTalker outperforms existing NeRF- and 3DGS-based approaches in rendering quality, lip-sync precision, and inference speed. Our method exhibits strong generalization capabilities and practical potential for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16863v1">ConfidentSplat: Confidence-Weighted Depth Fusion for Accurate 3D Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-09-21
    </div>
    <details class="paper-abstract">
      We introduce ConfidentSplat, a novel 3D Gaussian Splatting (3DGS)-based SLAM system for robust, highfidelity RGB-only reconstruction. Addressing geometric inaccuracies in existing RGB-only 3DGS SLAM methods that stem from unreliable depth estimation, ConfidentSplat incorporates a core innovation: a confidence-weighted fusion mechanism. This mechanism adaptively integrates depth cues from multiview geometry with learned monocular priors (Omnidata ViT), dynamically weighting their contributions based on explicit reliability estimates-derived predominantly from multi-view geometric consistency-to generate high-fidelity proxy depth for map supervision. The resulting proxy depth guides the optimization of a deformable 3DGS map, which efficiently adapts online to maintain global consistency following pose updates from a DROID-SLAM-inspired frontend and backend optimizations (loop closure, global bundle adjustment). Extensive validation on standard benchmarks (TUM-RGBD, ScanNet) and diverse custom mobile datasets demonstrates significant improvements in reconstruction accuracy (L1 depth error) and novel view synthesis fidelity (PSNR, SSIM, LPIPS) over baselines, particularly in challenging conditions. ConfidentSplat underscores the efficacy of principled, confidence-aware sensor fusion for advancing state-of-the-art dense visual SLAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16806v1">MedGS: Gaussian Splatting for Multi-Modal 3D Medical Imaging</a></div>
    <div class="paper-meta">
      📅 2025-09-20
    </div>
    <details class="paper-abstract">
      Multi-modal three-dimensional (3D) medical imaging data, derived from ultrasound, magnetic resonance imaging (MRI), and potentially computed tomography (CT), provide a widely adopted approach for non-invasive anatomical visualization. Accurate modeling, registration, and visualization in this setting depend on surface reconstruction and frame-to-frame interpolation. Traditional methods often face limitations due to image noise and incomplete information between frames. To address these challenges, we present MedGS, a semi-supervised neural implicit surface reconstruction framework that employs a Gaussian Splatting (GS)-based interpolation mechanism. In this framework, medical imaging data are represented as consecutive two-dimensional (2D) frames embedded in 3D space and modeled using Gaussian-based distributions. This representation enables robust frame interpolation and high-fidelity surface reconstruction across imaging modalities. As a result, MedGS offers more efficient training than traditional neural implicit methods. Its explicit GS-based representation enhances noise robustness, allows flexible editing, and supports precise modeling of complex anatomical structures with fewer artifacts. These features make MedGS highly suitable for scalable and practical applications in medical imaging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12720v3">Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 Accepted by NeurIPS 2025. Project page: https://chenkangjie1123.github.io/Co-Adaptation-3DGS/, Code at: https://github.com/chenkangjie1123/Co-Adaptation-of-3DGS
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated impressive performance in novel view synthesis under dense-view settings. However, in sparse-view scenarios, despite the realistic renderings in training views, 3DGS occasionally manifests appearance artifacts in novel views. This paper investigates the appearance artifacts in sparse-view 3DGS and uncovers a core limitation of current approaches: the optimized Gaussians are overly-entangled with one another to aggressively fit the training views, which leads to a neglect of the real appearance distribution of the underlying scene and results in appearance artifacts in novel views. The analysis is based on a proposed metric, termed Co-Adaptation Score (CA), which quantifies the entanglement among Gaussians, i.e., co-adaptation, by computing the pixel-wise variance across multiple renderings of the same viewpoint, with different random subsets of Gaussians. The analysis reveals that the degree of co-adaptation is naturally alleviated as the number of training views increases. Based on the analysis, we propose two lightweight strategies to explicitly mitigate the co-adaptation in sparse-view 3DGS: (1) random gaussian dropout; (2) multiplicative noise injection to the opacity. Both strategies are designed to be plug-and-play, and their effectiveness is validated across various methods and benchmarks. We hope that our insights into the co-adaptation effect will inspire the community to achieve a more comprehensive understanding of sparse-view 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02103v2">Multi-viewregulated gaussian splatting for novel view synthesis</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 Project Page:https://xiaobiaodu.github.io/mvgs-project/
    </div>
    <details class="paper-abstract">
      Recent works in volume rendering, \textit{e.g.} NeRF and 3D Gaussian Splatting (3DGS), significantly advance the rendering quality and efficiency with the help of the learned implicit neural radiance field or 3D Gaussians. Rendering on top of an explicit representation, the vanilla 3DGS and its variants deliver real-time efficiency by optimizing the parametric model with single-view supervision per iteration during training which is adopted from NeRF. Consequently, certain views are overfitted, leading to unsatisfying appearance in novel-view synthesis and imprecise 3D geometries. To solve aforementioned problems, we propose a new 3DGS optimization method embodying four key novel contributions: 1) We transform the conventional single-view training paradigm into a multi-view training strategy. With our proposed multi-view regulation, 3D Gaussian attributes are further optimized without overfitting certain training views. As a general solution, we improve the overall accuracy in a variety of scenarios and different Gaussian variants. 2) Inspired by the benefit introduced by additional views, we further propose a cross-intrinsic guidance scheme, leading to a coarse-to-fine training procedure concerning different resolutions. 3) Built on top of our multi-view regulated training, we further propose a cross-ray densification strategy, densifying more Gaussian kernels in the ray-intersect regions from a selection of views. 4) By further investigating the densification strategy, we found that the effect of densification should be enhanced when certain views are distinct dramatically. As a solution, we propose a novel multi-view augmented densification strategy, where 3D Gaussians are encouraged to get densified to a sufficient number accordingly, resulting in improved reconstruction accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16552v1">ST-GS: Vision-Based 3D Semantic Occupancy Prediction with Spatial-Temporal Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-20
    </div>
    <details class="paper-abstract">
      3D occupancy prediction is critical for comprehensive scene understanding in vision-centric autonomous driving. Recent advances have explored utilizing 3D semantic Gaussians to model occupancy while reducing computational overhead, but they remain constrained by insufficient multi-view spatial interaction and limited multi-frame temporal consistency. To overcome these issues, in this paper, we propose a novel Spatial-Temporal Gaussian Splatting (ST-GS) framework to enhance both spatial and temporal modeling in existing Gaussian-based pipelines. Specifically, we develop a guidance-informed spatial aggregation strategy within a dual-mode attention mechanism to strengthen spatial interaction in Gaussian representations. Furthermore, we introduce a geometry-aware temporal fusion scheme that effectively leverages historical context to improve temporal continuity in scene completion. Extensive experiments on the large-scale nuScenes occupancy prediction benchmark showcase that our proposed approach not only achieves state-of-the-art performance but also delivers markedly better temporal consistency compared to existing Gaussian-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16119v1">RadarGaussianDet3D: An Efficient and Effective Gaussian-based 3D Detector with 4D Automotive Radars</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      4D automotive radars have gained increasing attention for autonomous driving due to their low cost, robustness, and inherent velocity measurement capability. However, existing 4D radar-based 3D detectors rely heavily on pillar encoders for BEV feature extraction, where each point contributes to only a single BEV grid, resulting in sparse feature maps and degraded representation quality. In addition, they also optimize bounding box attributes independently, leading to sub-optimal detection accuracy. Moreover, their inference speed, while sufficient for high-end GPUs, may fail to meet the real-time requirement on vehicle-mounted embedded devices. To overcome these limitations, an efficient and effective Gaussian-based 3D detector, namely RadarGaussianDet3D is introduced, leveraging Gaussian primitives and distributions as intermediate representations for radar points and bounding boxes. In RadarGaussianDet3D, a novel Point Gaussian Encoder (PGE) is designed to transform each point into a Gaussian primitive after feature aggregation and employs the 3D Gaussian Splatting (3DGS) technique for BEV rasterization, yielding denser feature maps. PGE exhibits exceptionally low latency, owing to the optimized algorithm for point feature aggregation and fast rendering of 3DGS. In addition, a new Box Gaussian Loss (BGL) is proposed, which converts bounding boxes into 3D Gaussian distributions and measures their distance to enable more comprehensive and consistent optimization. Extensive experiments on TJ4DRadSet and View-of-Delft demonstrate that RadarGaussianDet3D achieves state-of-the-art detection accuracy while delivering substantially faster inference, highlighting its potential for real-time deployment in autonomous driving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15871v1">Zero-Shot Visual Grounding in 3D Gaussians via View Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      3D Visual Grounding (3DVG) aims to locate objects in 3D scenes based on text prompts, which is essential for applications such as robotics. However, existing 3DVG methods encounter two main challenges: first, they struggle to handle the implicit representation of spatial textures in 3D Gaussian Splatting (3DGS), making per-scene training indispensable; second, they typically require larges amounts of labeled data for effective training. To this end, we propose \underline{G}rounding via \underline{V}iew \underline{R}etrieval (GVR), a novel zero-shot visual grounding framework for 3DGS to transform 3DVG as a 2D retrieval task that leverages object-level view retrieval to collect grounding clues from multiple views, which not only avoids the costly process of 3D annotation, but also eliminates the need for per-scene training. Extensive experiments demonstrate that our method achieves state-of-the-art visual grounding performance while avoiding per-scene training, providing a solid foundation for zero-shot 3DVG research. Video demos can be found in https://github.com/leviome/GVR_demos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15677v1">Camera Splatting for Continuous View Optimization</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      We propose Camera Splatting, a novel view optimization framework for novel view synthesis. Each camera is modeled as a 3D Gaussian, referred to as a camera splat, and virtual cameras, termed point cameras, are placed at 3D points sampled near the surface to observe the distribution of camera splats. View optimization is achieved by continuously and differentiably refining the camera splats so that desirable target distributions are observed from the point cameras, in a manner similar to the original 3D Gaussian splatting. Compared to the Farthest View Sampling (FVS) approach, our optimized views demonstrate superior performance in capturing complex view-dependent phenomena, including intense metallic reflections and intricate textures such as text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15648v1">FingerSplat: Contactless Fingerprint 3D Reconstruction and Generation based on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      Researchers have conducted many pioneer researches on contactless fingerprints, yet the performance of contactless fingerprint recognition still lags behind contact-based methods primary due to the insufficient contactless fingerprint data with pose variations and lack of the usage of implicit 3D fingerprint representations. In this paper, we introduce a novel contactless fingerprint 3D registration, reconstruction and generation framework by integrating 3D Gaussian Splatting, with the goal of offering a new paradigm for contactless fingerprint recognition that integrates 3D fingerprint reconstruction and generation. To our knowledge, this is the first work to apply 3D Gaussian Splatting to the field of fingerprint recognition, and the first to achieve effective 3D registration and complete reconstruction of contactless fingerprints with sparse input images and without requiring camera parameters information. Experiments on 3D fingerprint registration, reconstruction, and generation prove that our method can accurately align and reconstruct 3D fingerprints from 2D images, and sequentially generates high-quality contactless fingerprints from 3D model, thus increasing the performances for contactless fingerprint recognition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15645v1">GS-Scale: Unlocking Large-Scale 3D Gaussian Splatting Training via Host Offloading</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      The advent of 3D Gaussian Splatting has revolutionized graphics rendering by delivering high visual quality and fast rendering speeds. However, training large-scale scenes at high quality remains challenging due to the substantial memory demands required to store parameters, gradients, and optimizer states, which can quickly overwhelm GPU memory. To address these limitations, we propose GS-Scale, a fast and memory-efficient training system for 3D Gaussian Splatting. GS-Scale stores all Gaussians in host memory, transferring only a subset to the GPU on demand for each forward and backward pass. While this dramatically reduces GPU memory usage, it requires frustum culling and optimizer updates to be executed on the CPU, introducing slowdowns due to CPU's limited compute and memory bandwidth. To mitigate this, GS-Scale employs three system-level optimizations: (1) selective offloading of geometric parameters for fast frustum culling, (2) parameter forwarding to pipeline CPU optimizer updates with GPU computation, and (3) deferred optimizer update to minimize unnecessary memory accesses for Gaussians with zero gradients. Our extensive evaluations on large-scale datasets demonstrate that GS-Scale significantly lowers GPU memory demands by 3.3-5.6x, while achieving training speeds comparable to GPU without host offloading. This enables large-scale 3D Gaussian Splatting training on consumer-grade GPUs; for instance, GS-Scale can scale the number of Gaussians from 4 million to 18 million on an RTX 4070 Mobile GPU, leading to 23-35% LPIPS (learned perceptual image patch similarity) improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15548v1">MS-GS: Multi-Appearance Sparse-View 3D Gaussian Splatting in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      In-the-wild photo collections often contain limited volumes of imagery and exhibit multiple appearances, e.g., taken at different times of day or seasons, posing significant challenges to scene reconstruction and novel view synthesis. Although recent adaptations of Neural Radiance Field (NeRF) and 3D Gaussian Splatting (3DGS) have improved in these areas, they tend to oversmooth and are prone to overfitting. In this paper, we present MS-GS, a novel framework designed with Multi-appearance capabilities in Sparse-view scenarios using 3DGS. To address the lack of support due to sparse initializations, our approach is built on the geometric priors elicited from monocular depth estimations. The key lies in extracting and utilizing local semantic regions with a Structure-from-Motion (SfM) points anchored algorithm for reliable alignment and geometry cues. Then, to introduce multi-view constraints, we propose a series of geometry-guided supervision at virtual views in a fine-grained and coarse scheme to encourage 3D consistency and reduce overfitting. We also introduce a dataset and an in-the-wild experiment setting to set up more realistic benchmarks. We demonstrate that MS-GS achieves photorealistic renderings under various challenging sparse-view and multi-appearance conditions and outperforms existing approaches significantly across different datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05075v2">GeoSplat: A Deep Dive into Geometry-Constrained Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      A few recent works explored incorporating geometric priors to regularize the optimization of Gaussian splatting, further improving its performance. However, those early studies mainly focused on the use of low-order geometric priors (e.g., normal vector), and they are also unreliably estimated by noise-sensitive methods, like local principal component analysis. To address their limitations, we first present GeoSplat, a general geometry-constrained optimization framework that exploits both first-order and second-order geometric quantities to improve the entire training pipeline of Gaussian splatting, including Gaussian initialization, gradient update, and densification. As an example, we initialize the scales of 3D Gaussian primitives in terms of principal curvatures, leading to a better coverage of the object surface than random initialization. Secondly, based on certain geometric structures (e.g., local manifold), we introduce efficient and noise-robust estimation methods that provide dynamic geometric priors for our framework. We conduct extensive experiments on multiple datasets for novel view synthesis, showing that our framework: GeoSplat, significantly improves the performance of Gaussian splatting and outperforms previous baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14135v3">GAF: Gaussian Action Field as a Dynamic World Model for Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 http://chaiying1.github.io/GAF.github.io/project_page/
    </div>
    <details class="paper-abstract">
      Accurate scene perception is critical for vision-based robotic manipulation. Existing approaches typically follow either a Vision-to-Action (V-A) paradigm, predicting actions directly from visual inputs, or a Vision-to-3D-to-Action (V-3D-A) paradigm, leveraging intermediate 3D representations. However, these methods often struggle with action inaccuracies due to the complexity and dynamic nature of manipulation scenes. In this paper, we adopt a V-4D-A framework that enables direct action reasoning from motion-aware 4D representations via a Gaussian Action Field (GAF). GAF extends 3D Gaussian Splatting (3DGS) by incorporating learnable motion attributes, allowing 4D modeling of dynamic scenes and manipulation actions. To learn time-varying scene geometry and action-aware robot motion, GAF provides three interrelated outputs: reconstruction of the current scene, prediction of future frames, and estimation of init action via Gaussian motion. Furthermore, we employ an action-vision-aligned denoising framework, conditioned on a unified representation that combines the init action and the Gaussian perception, both generated by the GAF, to further obtain more precise actions. Extensive experiments demonstrate significant improvements, with GAF achieving +11.5385 dB PSNR, +0.3864 SSIM and -0.5574 LPIPS improvements in reconstruction quality, while boosting the average +7.3% success rate in robotic manipulation tasks over state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09068v2">A new dataset and comparison for multi-camera frame synthesis</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 SPIE 2025 - Applications of Digital Image Processing XLVIII accepted manuscript, 13 pages
    </div>
    <details class="paper-abstract">
      Many methods exist for frame synthesis in image sequences but can be broadly categorised into frame interpolation and view synthesis techniques. Fundamentally, both frame interpolation and view synthesis tackle the same task, interpolating a frame given surrounding frames in time or space. However, most frame interpolation datasets focus on temporal aspects with single cameras moving through time and space, while view synthesis datasets are typically biased toward stereoscopic depth estimation use cases. This makes direct comparison between view synthesis and frame interpolation methods challenging. In this paper, we develop a novel multi-camera dataset using a custom-built dense linear camera array to enable fair comparison between these approaches. We evaluate classical and deep learning frame interpolators against a view synthesis method (3D Gaussian Splatting) for the task of view in-betweening. Our results reveal that deep learning methods do not significantly outperform classical methods on real image data, with 3D Gaussian Splatting actually underperforming frame interpolators by as much as 3.5 dB PSNR. However, in synthetic scenes, the situation reverses -- 3D Gaussian Splatting outperforms frame interpolation algorithms by almost 5 dB PSNR at a 95% confidence level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14739v1">FMGS-Avatar: Mesh-Guided 2D Gaussian Splatting with Foundation Model Priors for 3D Monocular Avatar Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      Reconstructing high-fidelity animatable human avatars from monocular videos remains challenging due to insufficient geometric information in single-view observations. While recent 3D Gaussian Splatting methods have shown promise, they struggle with surface detail preservation due to the free-form nature of 3D Gaussian primitives. To address both the representation limitations and information scarcity, we propose a novel method, \textbf{FMGS-Avatar}, that integrates two key innovations. First, we introduce Mesh-Guided 2D Gaussian Splatting, where 2D Gaussian primitives are attached directly to template mesh faces with constrained position, rotation, and movement, enabling superior surface alignment and geometric detail preservation. Second, we leverage foundation models trained on large-scale datasets, such as Sapiens, to complement the limited visual cues from monocular videos. However, when distilling multi-modal prior knowledge from foundation models, conflicting optimization objectives can emerge as different modalities exhibit distinct parameter sensitivities. We address this through a coordinated training strategy with selective gradient isolation, enabling each loss component to optimize its relevant parameters without interference. Through this combination of enhanced representation and coordinated information distillation, our approach significantly advances 3D monocular human avatar reconstruction. Experimental evaluation demonstrates superior reconstruction quality compared to existing methods, with notable gains in geometric accuracy and appearance fidelity while providing rich semantic information. Additionally, the distilled prior knowledge within a shared canonical space naturally enables spatially and temporally consistent rendering under novel views and poses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14687v1">RealMirror: A Comprehensive, Open-Source Vision-Language-Action Platform for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      The emerging field of Vision-Language-Action (VLA) for humanoid robots faces several fundamental challenges, including the high cost of data acquisition, the lack of a standardized benchmark, and the significant gap between simulation and the real world. To overcome these obstacles, we propose RealMirror, a comprehensive, open-source embodied AI VLA platform. RealMirror builds an efficient, low-cost data collection, model training, and inference system that enables end-to-end VLA research without requiring a real robot. To facilitate model evolution and fair comparison, we also introduce a dedicated VLA benchmark for humanoid robots, featuring multiple scenarios, extensive trajectories, and various VLA models. Furthermore, by integrating generative models and 3D Gaussian Splatting to reconstruct realistic environments and robot models, we successfully demonstrate zero-shot Sim2Real transfer, where models trained exclusively on simulation data can perform tasks on a real robot seamlessly, without any fine-tuning. In conclusion, with the unification of these critical components, RealMirror provides a robust framework that significantly accelerates the development of VLA models for humanoid robots. Project page: https://terminators2025.github.io/RealMirror.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06136v2">Roll Your Eyes: Gaze Redirection via Explicit 3D Eyeball Rotation</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 9 pages, 5 figures, ACM Multimeida 2025 accepted
    </div>
    <details class="paper-abstract">
      We propose a novel 3D gaze redirection framework that leverages an explicit 3D eyeball structure. Existing gaze redirection methods are typically based on neural radiance fields, which employ implicit neural representations via volume rendering. Unlike these NeRF-based approaches, where the rotation and translation of 3D representations are not explicitly modeled, we introduce a dedicated 3D eyeball structure to represent the eyeballs with 3D Gaussian Splatting (3DGS). Our method generates photorealistic images that faithfully reproduce the desired gaze direction by explicitly rotating and translating the 3D eyeball structure. In addition, we propose an adaptive deformation module that enables the replication of subtle muscle movements around the eyes. Through experiments conducted on the ETH-XGaze dataset, we demonstrate that our framework is capable of generating diverse novel gaze images, achieving superior image quality and gaze estimation accuracy compared to previous state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15249v1">Causal Reasoning Elicits Controllable 3D Scene Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      Existing 3D scene generation methods often struggle to model the complex logical dependencies and physical constraints between objects, limiting their ability to adapt to dynamic and realistic environments. We propose CausalStruct, a novel framework that embeds causal reasoning into 3D scene generation. Utilizing large language models (LLMs), We construct causal graphs where nodes represent objects and attributes, while edges encode causal dependencies and physical constraints. CausalStruct iteratively refines the scene layout by enforcing causal order to determine the placement order of objects and applies causal intervention to adjust the spatial configuration according to physics-driven constraints, ensuring consistency with textual descriptions and real-world dynamics. The refined scene causal graph informs subsequent optimization steps, employing a Proportional-Integral-Derivative(PID) controller to iteratively tune object scales and positions. Our method uses text or images to guide object placement and layout in 3D scenes, with 3D Gaussian Splatting and Score Distillation Sampling improving shape accuracy and rendering stability. Extensive experiments show that CausalStruct generates 3D scenes with enhanced logical coherence, realistic spatial interactions, and robust adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14191v1">MCGS-SLAM: A Multi-Camera SLAM Framework Using Gaussian Splatting for High-Fidelity Mapping</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Recent progress in dense SLAM has primarily targeted monocular setups, often at the expense of robustness and geometric coverage. We present MCGS-SLAM, the first purely RGB-based multi-camera SLAM system built on 3D Gaussian Splatting (3DGS). Unlike prior methods relying on sparse maps or inertial data, MCGS-SLAM fuses dense RGB inputs from multiple viewpoints into a unified, continuously optimized Gaussian map. A multi-camera bundle adjustment (MCBA) jointly refines poses and depths via dense photometric and geometric residuals, while a scale consistency module enforces metric alignment across views using low-rank priors. The system supports RGB input and maintains real-time performance at large scale. Experiments on synthetic and real-world datasets show that MCGS-SLAM consistently yields accurate trajectories and photorealistic reconstructions, usually outperforming monocular baselines. Notably, the wide field of view from multi-camera input enables reconstruction of side-view regions that monocular setups miss, critical for safe autonomous operation. These results highlight the promise of multi-camera Gaussian Splatting SLAM for high-fidelity mapping in robotics and autonomous driving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14171v2">Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      We introduce an image upscaling technique tailored for 3D Gaussian Splatting (3DGS) on lightweight GPUs. Compared to 3DGS, it achieves significantly higher rendering speeds and reduces artifacts commonly observed in 3DGS reconstructions. Our technique upscales low-resolution 3DGS renderings with a marginal increase in cost by directly leveraging the analytical image gradients of Gaussians for gradient-based bicubic spline interpolation. The technique is agnostic to the specific 3DGS implementation, achieving novel view synthesis at rates 3x-4x higher than the baseline implementation. Through extensive experiments on multiple datasets, we showcase the performance improvements and high reconstruction fidelity attainable with gradient-aware upscaling of 3DGS images. We further demonstrate the integration of gradient-aware upscaling into the gradient-based optimization of a 3DGS model and analyze its effects on reconstruction quality and performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13938v1">Plug-and-Play PDE Optimization for 3D Gaussian Splatting: Toward High-Quality Rendering and Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has revolutionized radiance field reconstruction by achieving high-quality novel view synthesis with fast rendering speed, introducing 3D Gaussian primitives to represent the scene. However, 3DGS encounters blurring and floaters when applied to complex scenes, caused by the reconstruction of redundant and ambiguous geometric structures. We attribute this issue to the unstable optimization of the Gaussians. To address this limitation, we present a plug-and-play PDE-based optimization method that overcomes the optimization constraints of 3DGS-based approaches in various tasks, such as novel view synthesis and surface reconstruction. Firstly, we theoretically derive that the 3DGS optimization procedure can be modeled as a PDE, and introduce a viscous term to ensure stable optimization. Secondly, we use the Material Point Method (MPM) to obtain a stable numerical solution of the PDE, which enhances both global and local constraints. Additionally, an effective Gaussian densification strategy and particle constraints are introduced to ensure fine-grained details. Extensive qualitative and quantitative experiments confirm that our method achieves state-of-the-art rendering and reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13863v1">LamiGauss: Pitching Radiative Gaussian for Sparse-View X-ray Laminography Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      X-ray Computed Laminography (CL) is essential for non-destructive inspection of plate-like structures in applications such as microchips and composite battery materials, where traditional computed tomography (CT) struggles due to geometric constraints. However, reconstructing high-quality volumes from laminographic projections remains challenging, particularly under highly sparse-view acquisition conditions. In this paper, we propose a reconstruction algorithm, namely LamiGauss, that combines Gaussian Splatting radiative rasterization with a dedicated detector-to-world transformation model incorporating the laminographic tilt angle. LamiGauss leverages an initialization strategy that explicitly filters out common laminographic artifacts from the preliminary reconstruction, preventing redundant Gaussians from being allocated to false structures and thereby concentrating model capacity on representing the genuine object. Our approach effectively optimizes directly from sparse projections, enabling accurate and efficient reconstruction with limited data. Extensive experiments on both synthetic and real datasets demonstrate the effectiveness and superiority of the proposed method over existing techniques. LamiGauss uses only 3$\%$ of full views to achieve superior performance over the iterative method optimized on a full dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17600v2">GWM: Towards Scalable Gaussian World Models for Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 Published at ICCV 2025. Project page: https://gaussian-world-model.github.io/
    </div>
    <details class="paper-abstract">
      Training robot policies within a learned world model is trending due to the inefficiency of real-world interactions. The established image-based world models and policies have shown prior success, but lack robust geometric information that requires consistent spatial and physical understanding of the three-dimensional world, even pre-trained on internet-scale video sources. To this end, we propose a novel branch of world model named Gaussian World Model (GWM) for robotic manipulation, which reconstructs the future state by inferring the propagation of Gaussian primitives under the effect of robot actions. At its core is a latent Diffusion Transformer (DiT) combined with a 3D variational autoencoder, enabling fine-grained scene-level future state reconstruction with Gaussian Splatting. GWM can not only enhance the visual representation for imitation learning agent by self-supervised future prediction training, but can serve as a neural simulator that supports model-based reinforcement learning. Both simulated and real-world experiments depict that GWM can precisely predict future scenes conditioned on diverse robot actions, and can be further utilized to train policies that outperform the state-of-the-art by impressive margins, showcasing the initial data scaling potential of 3D world model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14421v1">Perception-Integrated Safety Critical Control via Analytic Collision Cone Barrier Functions on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 Preprint for IEEE L-CSS/ACC
    </div>
    <details class="paper-abstract">
      We present a perception-driven safety filter that converts each 3D Gaussian Splat (3DGS) into a closed-form forward collision cone, which in turn yields a first-order control barrier function (CBF) embedded within a quadratic program (QP). By exploiting the analytic geometry of splats, our formulation provides a continuous, closed-form representation of collision constraints that is both simple and computationally efficient. Unlike distance-based CBFs, which tend to activate reactively only when an obstacle is already close, our collision-cone CBF activates proactively, allowing the robot to adjust earlier and thereby produce smoother and safer avoidance maneuvers at lower computational cost. We validate the method on a large synthetic scene with approximately 170k splats, where our filter reduces planning time by a factor of 3 and significantly decreased trajectory jerk compared to a state-of-the-art 3DGS planner, while maintaining the same level of safety. The approach is entirely analytic, requires no high-order CBF extensions (HOCBFs), and generalizes naturally to robots with physical extent through a principled Minkowski-sum inflation of the splats. These properties make the method broadly applicable to real-time navigation in cluttered, perception-derived extreme environments, including space robotics and satellite systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04929v2">CryoSplat: Gaussian Splatting for Cryo-EM Homogeneous Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      As a critical modality for structural biology, cryogenic electron microscopy (cryo-EM) facilitates the determination of macromolecular structures at near-atomic resolution. The core computational task in single-particle cryo-EM is to reconstruct the 3D electrostatic potential of a molecule from a large collection of noisy 2D projections acquired at unknown orientations. Gaussian mixture models (GMMs) provide a continuous, compact, and physically interpretable representation for molecular density and have recently gained interest in cryo-EM reconstruction. However, existing methods rely on external consensus maps or atomic models for initialization, limiting their use in self-contained pipelines. Addressing this issue, we introduce cryoGS, a GMM-based method that integrates Gaussian splatting with the physics of cryo-EM image formation. In particular, we develop an orthogonal projection-aware Gaussian splatting, with adaptations such as a normalization term and FFT-aligned coordinate system tailored for cryo-EM imaging. All these innovations enable stable and efficient homogeneous reconstruction directly from raw cryo-EM particle images using random initialization. Experimental results on real datasets validate the effectiveness and robustness of cryoGS over representative baselines. The code will be released upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01799v2">WorldExplorer: Towards Generating Fully Navigable 3D Scenes</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted to SIGGRAPH Asia 2025. Project page: see https://mschneider456.github.io/world-explorer, video: see https://youtu.be/N6NJsNyiv6I, code: https://github.com/mschneider456/WorldExplorer
    </div>
    <details class="paper-abstract">
      Generating 3D worlds from text is a highly anticipated goal in computer vision. Existing works are limited by the degree of exploration they allow inside of a scene, i.e., produce streched-out and noisy artifacts when moving beyond central or panoramic perspectives. To this end, we propose WorldExplorer, a novel method based on autoregressive video trajectory generation, which builds fully navigable 3D scenes with consistent visual quality across a wide range of viewpoints. We initialize our scenes by creating multi-view consistent images corresponding to a 360 degree panorama. Then, we expand it by leveraging video diffusion models in an iterative scene generation pipeline. Concretely, we generate multiple videos along short, pre-defined trajectories, that explore the scene in depth, including motion around objects. Our novel scene memory conditions each video on the most relevant prior views, while a collision-detection mechanism prevents degenerate results, like moving into objects. Finally, we fuse all generated views into a unified 3D representation via 3D Gaussian Splatting optimization. Compared to prior approaches, WorldExplorer produces high-quality scenes that remain stable under large camera motion, enabling for the first time realistic and unrestricted exploration. We believe this marks a significant step toward generating immersive and truly explorable virtual 3D environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13013v1">Dream3DAvatar: Text-Controlled 3D Avatar Reconstruction from a Single Image</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      With the rapid advancement of 3D representation techniques and generative models, substantial progress has been made in reconstructing full-body 3D avatars from a single image. However, this task remains fundamentally ill-posedness due to the limited information available from monocular input, making it difficult to control the geometry and texture of occluded regions during generation. To address these challenges, we redesign the reconstruction pipeline and propose Dream3DAvatar, an efficient and text-controllable two-stage framework for 3D avatar generation. In the first stage, we develop a lightweight, adapter-enhanced multi-view generation model. Specifically, we introduce the Pose-Adapter to inject SMPL-X renderings and skeletal information into SDXL, enforcing geometric and pose consistency across views. To preserve facial identity, we incorporate ID-Adapter-G, which injects high-resolution facial features into the generation process. Additionally, we leverage BLIP2 to generate high-quality textual descriptions of the multi-view images, enhancing text-driven controllability in occluded regions. In the second stage, we design a feedforward Transformer model equipped with a multi-view feature fusion module to reconstruct high-fidelity 3D Gaussian Splat representations (3DGS) from the generated images. Furthermore, we introduce ID-Adapter-R, which utilizes a gating mechanism to effectively fuse facial features into the reconstruction process, improving high-frequency detail recovery. Extensive experiments demonstrate that our method can generate realistic, animation-ready 3D avatars without any post-processing and consistently outperforms existing baselines across multiple evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12938v1">Beyond Averages: Open-Vocabulary 3D Scene Understanding with Gaussian Splatting and Bag of Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Novel view synthesis has seen significant advancements with 3D Gaussian Splatting (3DGS), enabling real-time photorealistic rendering. However, the inherent fuzziness of Gaussian Splatting presents challenges for 3D scene understanding, restricting its broader applications in AR/VR and robotics. While recent works attempt to learn semantics via 2D foundation model distillation, they inherit fundamental limitations: alpha blending averages semantics across objects, making 3D-level understanding impossible. We propose a paradigm-shifting alternative that bypasses differentiable rendering for semantics entirely. Our key insight is to leverage predecomposed object-level Gaussians and represent each object through multiview CLIP feature aggregation, creating comprehensive "bags of embeddings" that holistically describe objects. This allows: (1) accurate open-vocabulary object retrieval by comparing text queries to object-level (not Gaussian-level) embeddings, and (2) seamless task adaptation: propagating object IDs to pixels for 2D segmentation or to Gaussians for 3D extraction. Experiments demonstrate that our method effectively overcomes the challenges of 3D open-vocabulary object extraction while remaining comparable to state-of-the-art performance in 2D open-vocabulary segmentation, ensuring minimal compromise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04859v2">CoRe-GS: Coarse-to-Refined Gaussian Splatting with Semantic Object Focus</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Mobile reconstruction has the potential to support time-critical tasks such as tele-guidance and disaster response, where operators must quickly gain an accurate understanding of the environment. Full high-fidelity scene reconstruction is computationally expensive and often unnecessary when only specific points of interest (POIs) matter for timely decision making. We address this challenge with CoRe-GS, a semantic POI-focused extension of Gaussian Splatting (GS). Instead of optimizing every scene element uniformly, CoRe-GS first produces a fast segmentation-ready GS representation and then selectively refines splats belonging to semantically relevant POIs detected during data acquisition. This targeted refinement reduces training time to 25\% compared to full semantic GS while improving novel view synthesis quality in the areas that matter most. We validate CoRe-GS on both real-world (SCRREAM) and synthetic (NeRDS 360) datasets, demonstrating that prioritizing POIs enables faster and higher-quality mobile reconstruction tailored to operational needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12742v1">Effective Gaussian Management for High-fidelity Object Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      This paper proposes an effective Gaussian management approach for high-fidelity object reconstruction. Departing from recent Gaussian Splatting (GS) methods that employ indiscriminate attribute assignment, our approach introduces a novel densification strategy that dynamically activates spherical harmonics (SHs) or normals under the supervision of a surface reconstruction module, which effectively mitigates the gradient conflicts caused by dual supervision and achieves superior reconstruction results. To further improve representation efficiency, we develop a lightweight Gaussian representation that adaptively adjusts the SH orders of each Gaussian based on gradient magnitudes and performs task-decoupled pruning to remove Gaussian with minimal impact on a reconstruction task without sacrificing others, which balances the representational capacity with parameter quantity. Notably, our management approach is model-agnostic and can be seamlessly integrated into other frameworks, enhancing performance while reducing model size. Extensive experiments demonstrate that our approach consistently outperforms state-of-the-art approaches in both reconstruction quality and efficiency, achieving superior performance with significantly fewer parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13536v1">MemGS: Memory-Efficient Gaussian Splatting for Real-Time SLAM</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3DGS) have made a significant impact on rendering and reconstruction techniques. Current research predominantly focuses on improving rendering performance and reconstruction quality using high-performance desktop GPUs, largely overlooking applications for embedded platforms like micro air vehicles (MAVs). These devices, with their limited computational resources and memory, often face a trade-off between system performance and reconstruction quality. In this paper, we improve existing methods in terms of GPU memory usage while enhancing rendering quality. Specifically, to address redundant 3D Gaussian primitives in SLAM, we propose merging them in voxel space based on geometric similarity. This reduces GPU memory usage without impacting system runtime performance. Furthermore, rendering quality is improved by initializing 3D Gaussian primitives via Patch-Grid (PG) point sampling, enabling more accurate modeling of the entire scene. Quantitative and qualitative evaluations on publicly available datasets demonstrate the effectiveness of our improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13482v1">Improving 3D Gaussian Splatting Compression by Scene-Adaptive Lattice Vector Quantization</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Code available at https://github.com/hxu160/SALVQ
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is rapidly gaining popularity for its photorealistic rendering quality and real-time performance, but it generates massive amounts of data. Hence compressing 3DGS data is necessary for the cost effectiveness of 3DGS models. Recently, several anchor-based neural compression methods have been proposed, achieving good 3DGS compression performance. However, they all rely on uniform scalar quantization (USQ) due to its simplicity. A tantalizing question is whether more sophisticated quantizers can improve the current 3DGS compression methods with very little extra overhead and minimal change to the system. The answer is yes by replacing USQ with lattice vector quantization (LVQ). To better capture scene-specific characteristics, we optimize the lattice basis for each scene, improving LVQ's adaptability and R-D efficiency. This scene-adaptive LVQ (SALVQ) strikes a balance between the R-D efficiency of vector quantization and the low complexity of USQ. SALVQ can be seamlessly integrated into existing 3DGS compression architectures, enhancing their R-D performance with minimal modifications and computational overhead. Moreover, by scaling the lattice basis vectors, SALVQ can dynamically adjust lattice density, enabling a single model to accommodate multiple bit rate targets. This flexibility eliminates the need to train separate models for different compression levels, significantly reducing training time and memory consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12138v1">Distributed 3D Gaussian Splatting for High-Resolution Isosurface Visualization</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D-GS) has recently emerged as a powerful technique for real-time, photorealistic rendering by optimizing anisotropic Gaussian primitives from view-dependent images. While 3D-GS has been extended to scientific visualization, prior work remains limited to single-GPU settings, restricting scalability for large datasets on high-performance computing (HPC) systems. We present a distributed 3D-GS pipeline tailored for HPC. Our approach partitions data across nodes, trains Gaussian splats in parallel using multi-nodes and multi-GPUs, and merges splats for global rendering. To eliminate artifacts, we add ghost cells at partition boundaries and apply background masks to remove irrelevant pixels. Benchmarks on the Richtmyer-Meshkov datasets (about 106.7M Gaussians) show up to 3X speedup across 8 nodes on Polaris while preserving image quality. These results demonstrate that distributed 3D-GS enables scalable visualization of large-scale scientific data and provide a foundation for future in situ applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10241v2">On the Geometric Accuracy of Implicit and Primitive-based Representations Derived from View Rendering Constraints</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 9 pages, 3 figures, to be presented at ASTRA25,
    </div>
    <details class="paper-abstract">
      We present the first systematic comparison of implicit and explicit Novel View Synthesis methods for space-based 3D object reconstruction, evaluating the role of appearance embeddings. While embeddings improve photometric fidelity by modeling lighting variation, we show they do not translate into meaningful gains in geometric accuracy - a critical requirement for space robotics applications. Using the SPEED+ dataset, we compare K-Planes, Gaussian Splatting, and Convex Splatting, and demonstrate that embeddings primarily reduce the number of primitives needed for explicit methods rather than enhancing geometric fidelity. Moreover, convex splatting achieves more compact and clutter-free representations than Gaussian splatting, offering advantages for safety-critical applications such as interaction and collision avoidance. Our findings clarify the limits of appearance embeddings for geometry-centric tasks and highlight trade-offs between reconstruction quality and representation efficiency in space scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11853v1">Segmentation-Driven Initialization for Sparse-view 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Sparse-view synthesis remains a challenging problem due to the difficulty of recovering accurate geometry and appearance from limited observations. While recent advances in 3D Gaussian Splatting (3DGS) have enabled real-time rendering with competitive quality, existing pipelines often rely on Structure-from-Motion (SfM) for camera pose estimation, an approach that struggles in genuinely sparse-view settings. Moreover, several SfM-free methods replace SfM with multi-view stereo (MVS) models, but generate massive numbers of 3D Gaussians by back-projecting every pixel into 3D space, leading to high memory costs. We propose Segmentation-Driven Initialization for Gaussian Splatting (SDI-GS), a method that mitigates inefficiency by leveraging region-based segmentation to identify and retain only structurally significant regions. This enables selective downsampling of the dense point cloud, preserving scene fidelity while substantially reducing Gaussian count. Experiments across diverse benchmarks show that SDI-GS reduces Gaussian count by up to 50% and achieves comparable or superior rendering quality in PSNR and SSIM, with only marginal degradation in LPIPS. It further enables faster training and lower memory footprint, advancing the practicality of 3DGS for constrained-view scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06433v2">Real-time Photorealistic Mapping for Situational Awareness in Robot Teleoperation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Achieving efficient remote teleoperation is particularly challenging in unknown environments, as the teleoperator must rapidly build an understanding of the site's layout. Online 3D mapping is a proven strategy to tackle this challenge, as it enables the teleoperator to progressively explore the site from multiple perspectives. However, traditional online map-based teleoperation systems struggle to generate visually accurate 3D maps in real-time due to the high computational cost involved, leading to poor teleoperation performances. In this work, we propose a solution to improve teleoperation efficiency in unknown environments. Our approach proposes a novel, modular and efficient GPU-based integration between recent advancement in gaussian splatting SLAM and existing online map-based teleoperation systems. We compare the proposed solution against state-of-the-art teleoperation systems and validate its performances through real-world experiments using an aerial vehicle. The results show significant improvements in decision-making speed and more accurate interaction with the environment, leading to greater teleoperation efficiency. In doing so, our system enhances remote teleoperation by seamlessly integrating photorealistic mapping generation with real-time performances, enabling effective teleoperation in unfamiliar environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23716v2">AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Project page: https://city-super.github.io/anysplat/
    </div>
    <details class="paper-abstract">
      We introduce AnySplat, a feed forward network for novel view synthesis from uncalibrated image collections. In contrast to traditional neural rendering pipelines that demand known camera poses and per scene optimization, or recent feed forward methods that buckle under the computational weight of dense views, our model predicts everything in one shot. A single forward pass yields a set of 3D Gaussian primitives encoding both scene geometry and appearance, and the corresponding camera intrinsics and extrinsics for each input image. This unified design scales effortlessly to casually captured, multi view datasets without any pose annotations. In extensive zero shot evaluations, AnySplat matches the quality of pose aware baselines in both sparse and dense view scenarios while surpassing existing pose free approaches. Moreover, it greatly reduce rendering latency compared to optimization based neural fields, bringing real time novel view synthesis within reach for unconstrained capture settings.Project page: https://city-super.github.io/anysplat/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11624v1">A Controllable 3D Deepfake Generation Framework with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      We propose a novel 3D deepfake generation framework based on 3D Gaussian Splatting that enables realistic, identity-preserving face swapping and reenactment in a fully controllable 3D space. Compared to conventional 2D deepfake approaches that suffer from geometric inconsistencies and limited generalization to novel view, our method combines a parametric head model with dynamic Gaussian representations to support multi-view consistent rendering, precise expression control, and seamless background integration. To address editing challenges in point-based representations, we explicitly separate the head and background Gaussians and use pre-trained 2D guidance to optimize the facial region across views. We further introduce a repair module to enhance visual consistency under extreme poses and expressions. Experiments on NeRSemble and additional evaluation videos demonstrate that our method achieves comparable performance to state-of-the-art 2D approaches in identity preservation, as well as pose and expression consistency, while significantly outperforming them in multi-view rendering quality and 3D consistency. Our approach bridges the gap between 3D modeling and deepfake synthesis, enabling new directions for scene-aware, controllable, and immersive visual forgeries, revealing the threat that emerging 3D Gaussian Splatting technique could be used for manipulation attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00763v2">UnIRe: Unsupervised Instance Decomposition for Dynamic Urban Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Reconstructing and decomposing dynamic urban scenes is crucial for autonomous driving, urban planning, and scene editing. However, existing methods fail to perform instance-aware decomposition without manual annotations, which is crucial for instance-level scene editing.We propose UnIRe, a 3D Gaussian Splatting (3DGS) based approach that decomposes a scene into a static background and individual dynamic instances using only RGB images and LiDAR point clouds. At its core, we introduce 4D superpoints, a novel representation that clusters multi-frame LiDAR points in 4D space, enabling unsupervised instance separation based on spatiotemporal correlations. These 4D superpoints serve as the foundation for our decomposed 4D initialization, i.e., providing spatial and temporal initialization to train a dynamic 3DGS for arbitrary dynamic classes without requiring bounding boxes or object templates.Furthermore, we introduce a smoothness regularization strategy in both 2D and 3D space, further improving the temporal stability.Experiments on benchmark datasets show that our method outperforms existing methods in decomposed dynamic scene reconstruction while enabling accurate and flexible instance-level editing, making it a practical solution for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03017v2">SA-3DGS: A Self-Adaptive Compression Method for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 This paper is being withdrawn as the work is incomplete and requires substantial additional development before it can be presented
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting have enhanced efficient and high-quality novel view synthesis. However, representing scenes requires a large number of Gaussian points, leading to high storage demands and limiting practical deployment. The latest methods facilitate the compression of Gaussian models but struggle to identify truly insignificant Gaussian points in the scene, leading to a decline in subsequent Gaussian pruning, compression quality, and rendering performance. To address this issue, we propose SA-3DGS, a method that significantly reduces storage costs while maintaining rendering quality. SA-3DGS learns an importance score to automatically identify the least significant Gaussians in scene reconstruction, thereby enabling effective pruning and redundancy reduction. Next, the importance-aware clustering module compresses Gaussians attributes more accurately into the codebook, improving the codebook's expressive capability while reducing model size. Finally, the codebook repair module leverages contextual scene information to repair the codebook, thereby recovering the original Gaussian point attributes and mitigating the degradation in rendering quality caused by information loss. Experimental results on several benchmark datasets show that our method achieves up to 66x compression while maintaining or even improving rendering quality. The proposed Gaussian pruning approach is not only adaptable to but also improves other pruning-based methods (e.g., LightGaussian), showcasing excellent performance and strong generalization ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04612v2">Tool-as-Interface: Learning Robot Policies from Observing Human Tool Use</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 Accepted to CoRL 2025. Project page: https://tool-as-interface.github.io. 17 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Tool use is essential for enabling robots to perform complex real-world tasks, but learning such skills requires extensive datasets. While teleoperation is widely used, it is slow, delay-sensitive, and poorly suited for dynamic tasks. In contrast, human videos provide a natural way for data collection without specialized hardware, though they pose challenges on robot learning due to viewpoint variations and embodiment gaps. To address these challenges, we propose a framework that transfers tool-use knowledge from humans to robots. To improve the policy's robustness to viewpoint variations, we use two RGB cameras to reconstruct 3D scenes and apply Gaussian splatting for novel view synthesis. We reduce the embodiment gap using segmented observations and tool-centric, task-space actions to achieve embodiment-invariant visuomotor policy learning. We demonstrate our framework's effectiveness across a diverse suite of tool-use tasks, where our learned policy shows strong generalization and robustness to human perturbations, camera motion, and robot base movement. Our method achieves a 71\% improvement in task success over teleoperation-based diffusion policies and dramatically reduces data collection time by 77\% and 41\% compared to teleoperation and the state-of-the-art interface, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11411v1">On the Skinning of Gaussian Avatars</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Radiance field-based methods have recently been used to reconstruct human avatars, showing that we can significantly downscale the systems needed for creating animated human avatars. Although this progress has been initiated by neural radiance fields, their slow rendering and backward mapping from the observation space to the canonical space have been the main challenges. With Gaussian splatting overcoming both challenges, a new family of approaches has emerged that are faster to train and render, while also straightforward to implement using forward skinning from the canonical to the observation space. However, the linear blend skinning required for the deformation of the Gaussians does not provide valid results for their non-linear rotation properties. To address such artifacts, recent works use mesh properties to rotate the non-linear Gaussian properties or train models to predict corrective offsets. Instead, we propose a weighted rotation blending approach that leverages quaternion averaging. This leads to simpler vertex-based Gaussians that can be efficiently animated and integrated in any engine by only modifying the linear blend skinning technique, and using any Gaussian rasterizer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11275v1">ROSGS: Relightable Outdoor Scenes With Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Image data captured outdoors often exhibit unbounded scenes and unconstrained, varying lighting conditions, making it challenging to decompose them into geometry, reflectance, and illumination. Recent works have focused on achieving this decomposition using Neural Radiance Fields (NeRF) or the 3D Gaussian Splatting (3DGS) representation but remain hindered by two key limitations: the high computational overhead associated with neural networks of NeRF and the use of low-frequency lighting representations, which often result in inefficient rendering and suboptimal relighting accuracy. We propose ROSGS, a two-stage pipeline designed to efficiently reconstruct relightable outdoor scenes using the Gaussian Splatting representation. By leveraging monocular normal priors, ROSGS first reconstructs the scene's geometry with the compact 2D Gaussian Splatting (2DGS) representation, providing an efficient and accurate geometric foundation. Building upon this reconstructed geometry, ROSGS then decomposes the scene's texture and lighting through a hybrid lighting model. This model effectively represents typical outdoor lighting by employing a spherical Gaussian function to capture the directional, high-frequency components of sunlight, while learning a radiance transfer function via Spherical Harmonic coefficients to model the remaining low-frequency skylight comprehensively. Both quantitative metrics and qualitative comparisons demonstrate that ROSGS achieves state-of-the-art performance in relighting outdoor scenes and highlight its ability to deliver superior relighting accuracy and rendering efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11116v1">SVR-GS: Spatially Variant Regularization for Probabilistic Masks in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables fast, high-quality novel view synthesis but typically relies on densification followed by pruning to optimize the number of Gaussians. Existing mask-based pruning, such as MaskGS, regularizes the global mean of the mask, which is misaligned with the local per-pixel (per-ray) reconstruction loss that determines image quality along individual camera rays. This paper introduces SVR-GS, a spatially variant regularizer that renders a per-pixel spatial mask from each Gaussian's effective contribution along the ray, thereby applying sparsity pressure where it matters: on low-importance Gaussians. We explore three spatial-mask aggregation strategies, implement them in CUDA, and conduct a gradient analysis to motivate our final design. Extensive experiments on Tanks\&Temples, Deep Blending, and Mip-NeRF360 datasets demonstrate that, on average across the three datasets, the proposed SVR-GS reduces the number of Gaussians by 1.79\(\times\) compared to MaskGS and 5.63\(\times\) compared to 3DGS, while incurring only 0.50 dB and 0.40 dB PSNR drops, respectively. These gains translate into significantly smaller, faster, and more memory-efficient models, making them well-suited for real-time applications such as robotics, AR/VR, and mobile perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09040v3">Motion Blender Gaussian Splatting for Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 CoRL 2025
    </div>
    <details class="paper-abstract">
      Gaussian splatting has emerged as a powerful tool for high-fidelity reconstruction of dynamic scenes. However, existing methods primarily rely on implicit motion representations, such as encoding motions into neural networks or per-Gaussian parameters, which makes it difficult to further manipulate the reconstructed motions. This lack of explicit controllability limits existing methods to replaying recorded motions only, which hinders a wider application in robotics. To address this, we propose Motion Blender Gaussian Splatting (MBGS), a novel framework that uses motion graphs as an explicit and sparse motion representation. The motion of a graph's links is propagated to individual Gaussians via dual quaternion skinning, with learnable weight painting functions that determine the influence of each link. The motion graphs and 3D Gaussians are jointly optimized from input videos via differentiable rendering. Experiments show that MBGS achieves state-of-the-art performance on the highly challenging iPhone dataset while being competitive on HyperNeRF. We demonstrate the application potential of our method in animating novel object poses, synthesizing real robot demonstrations, and predicting robot actions through visual planning. The source code, models, video demonstrations can be found at http://mlzxy.github.io/motion-blender-gs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08137v3">Occlusion-Aware Temporally Consistent Amodal Completion for 3D Human-Object Interaction Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 ACM MM 2025
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for reconstructing dynamic human-object interactions from monocular video that overcomes challenges associated with occlusions and temporal inconsistencies. Traditional 3D reconstruction methods typically assume static objects or full visibility of dynamic subjects, leading to degraded performance when these assumptions are violated-particularly in scenarios where mutual occlusions occur. To address this, our framework leverages amodal completion to infer the complete structure of partially obscured regions. Unlike conventional approaches that operate on individual frames, our method integrates temporal context, enforcing coherence across video sequences to incrementally refine and stabilize reconstructions. This template-free strategy adapts to varying conditions without relying on predefined models, significantly enhancing the recovery of intricate details in dynamic scenes. We validate our approach using 3D Gaussian Splatting on challenging monocular videos, demonstrating superior precision in handling occlusions and maintaining temporal stability compared to existing techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11003v1">AD-GS: Alternating Densification for Sparse-Input 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-13
      | 💬 SIGGRAPH Asia 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown impressive results in real-time novel view synthesis. However, it often struggles under sparse-view settings, producing undesirable artifacts such as floaters, inaccurate geometry, and overfitting due to limited observations. We find that a key contributing factor is uncontrolled densification, where adding Gaussian primitives rapidly without guidance can harm geometry and cause artifacts. We propose AD-GS, a novel alternating densification framework that interleaves high and low densification phases. During high densification, the model densifies aggressively, followed by photometric loss based training to capture fine-grained scene details. Low densification then primarily involves aggressive opacity pruning of Gaussians followed by regularizing their geometry through pseudo-view consistency and edge-aware depth smoothness. This alternating approach helps reduce overfitting by carefully controlling model capacity growth while progressively refining the scene representation. Extensive experiments on challenging datasets demonstrate that AD-GS significantly improves rendering quality and geometric consistency compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10759v1">Every Camera Effect, Every Time, All at Once: 4D Gaussian Ray Tracing for Physics-based Camera Effect Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-13
    </div>
    <details class="paper-abstract">
      Common computer vision systems typically assume ideal pinhole cameras but fail when facing real-world camera effects such as fisheye distortion and rolling shutter, mainly due to the lack of learning from training data with camera effects. Existing data generation approaches suffer from either high costs, sim-to-real gaps or fail to accurately model camera effects. To address this bottleneck, we propose 4D Gaussian Ray Tracing (4D-GRT), a novel two-stage pipeline that combines 4D Gaussian Splatting with physically-based ray tracing for camera effect simulation. Given multi-view videos, 4D-GRT first reconstructs dynamic scenes, then applies ray tracing to generate videos with controllable, physically accurate camera effects. 4D-GRT achieves the fastest rendering speed while performing better or comparable rendering quality compared to existing baselines. Additionally, we construct eight synthetic dynamic scenes in indoor environments across four camera effects as a benchmark to evaluate generated videos with camera effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10241v1">On the Geometric Accuracy of Implicit and Primitive-based Representations Derived from View Rendering Constraints</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 9 pages, 3 figures, to be presented at ASTRA25,
    </div>
    <details class="paper-abstract">
      We present the first systematic comparison of implicit and explicit Novel View Synthesis methods for space-based 3D object reconstruction, evaluating the role of appearance embeddings. While embeddings improve photometric fidelity by modeling lighting variation, we show they do not translate into meaningful gains in geometric accuracy - a critical requirement for space robotics applications. Using the SPEED+ dataset, we compare K-Planes, Gaussian Splatting, and Convex Splatting, and demonstrate that embeddings primarily reduce the number of primitives needed for explicit methods rather than enhancing geometric fidelity. Moreover, convex splatting achieves more compact and clutter-free representations than Gaussian splatting, offering advantages for safety-critical applications such as interaction and collision avoidance. Our findings clarify the limits of appearance embeddings for geometry-centric tasks and highlight trade-offs between reconstruction quality and representation efficiency in space scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21152v2">Geometry and Perception Guided Gaussians for Multiview-consistent 3D Generation from a Single Image</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Generating realistic 3D objects from single-view images requires natural appearance, 3D consistency, and the ability to capture multiple plausible interpretations of unseen regions. Existing approaches often rely on fine-tuning pretrained 2D diffusion models or directly generating 3D information through fast network inference or 3D Gaussian Splatting, but their results generally suffer from poor multiview consistency and lack geometric detail. To tackle these issues, we present a novel method that seamlessly integrates geometry and perception information without requiring additional model training to reconstruct detailed 3D objects from a single image. Specifically, we incorporate geometry and perception priors to initialize the Gaussian branches and guide their parameter optimization. The geometry prior captures the rough 3D shapes, while the perception prior utilizes the 2D pretrained diffusion model to enhance multiview information. Subsequently, we introduce a stable Score Distillation Sampling for fine-grained prior distillation to ensure effective knowledge transfer. The model is further enhanced by a reprojection-based strategy that enforces depth consistency. Experimental results show that we outperform existing methods on novel view synthesis and 3D reconstruction, demonstrating robust and consistent 3D object generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10678v1">T2Bs: Text-to-Character Blendshapes via Video Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      We present T2Bs, a framework for generating high-quality, animatable character head morphable models from text by combining static text-to-3D generation with video diffusion. Text-to-3D models produce detailed static geometry but lack motion synthesis, while video diffusion models generate motion with temporal and multi-view geometric inconsistencies. T2Bs bridges this gap by leveraging deformable 3D Gaussian splatting to align static 3D assets with video outputs. By constraining motion with static geometry and employing a view-dependent deformation MLP, T2Bs (i) outperforms existing 4D generation methods in accuracy and expressiveness while reducing video artifacts and view inconsistencies, and (ii) reconstructs smooth, coherent, fully registered 3D geometries designed to scale for building morphable models with diverse, realistic facial motions. This enables synthesizing expressive, animatable character heads that surpass current 4D generation techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06179v3">ForestSplats: Deformable transient field for Gaussian Splatting in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3D-GS) has emerged, showing real-time rendering speeds and high-quality results in static scenes. Although 3D-GS shows effectiveness in static scenes, their performance significantly degrades in real-world environments due to transient objects, lighting variations, and diverse levels of occlusion. To tackle this, existing methods estimate occluders or transient elements by leveraging pre-trained models or integrating additional transient field pipelines. However, these methods still suffer from two defects: 1) Using semantic features from the Vision Foundation model (VFM) causes additional computational costs. 2) The transient field requires significant memory to handle transient elements with per-view Gaussians and struggles to define clear boundaries for occluders, solely relying on photometric errors. To address these problems, we propose ForestSplats, a novel approach that leverages the deformable transient field and a superpixel-aware mask to efficiently represent transient elements in the 2D scene across unconstrained image collections and effectively decompose static scenes from transient distractors without VFM. We designed the transient field to be deformable, capturing per-view transient elements. Furthermore, we introduce a superpixel-aware mask that clearly defines the boundaries of occluders by considering photometric errors and superpixels. Additionally, we propose uncertainty-aware densification to avoid generating Gaussians within the boundaries of occluders during densification. Through extensive experiments across several benchmark datasets, we demonstrate that ForestSplats outperforms existing methods without VFM and shows significant memory efficiency in representing transient elements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06685v3">VIM-GS: Visual-Inertial Monocular Gaussian Splatting via Object-level Guidance in Large Scenes</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 Withdrawn due to an error in the author list & incomplete experimental results
    </div>
    <details class="paper-abstract">
      VIM-GS is a Gaussian Splatting (GS) framework using monocular images for novel-view synthesis (NVS) in large scenes. GS typically requires accurate depth to initiate Gaussian ellipsoids using RGB-D/stereo cameras. Their limited depth sensing range makes it difficult for GS to work in large scenes. Monocular images, however, lack depth to guide the learning and lead to inferior NVS results. Although large foundation models (LFMs) for monocular depth estimation are available, they suffer from cross-frame inconsistency, inaccuracy for distant scenes, and ambiguity in deceptive texture cues. This paper aims to generate dense, accurate depth images from monocular RGB inputs for high-definite GS rendering. The key idea is to leverage the accurate but sparse depth from visual-inertial Structure-from-Motion (SfM) to refine the dense but coarse depth from LFMs. To bridge the sparse input and dense output, we propose an object-segmented depth propagation algorithm that renders the depth of pixels of structured objects. Then we develop a dynamic depth refinement module to handle the crippled SfM depth of dynamic objects and refine the coarse LFM depth. Experiments using public and customized datasets demonstrate the superior rendering quality of VIM-GS in large scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10546v2">The Oxford Spires Dataset: Benchmarking Large-Scale LiDAR-Visual Localisation, Reconstruction and Radiance Field Methods</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 Accepted by IJRR. Website: https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/
    </div>
    <details class="paper-abstract">
      This paper introduces a large-scale multi-modal dataset captured in and around well-known landmarks in Oxford using a custom-built multi-sensor perception unit as well as a millimetre-accurate map from a Terrestrial LiDAR Scanner (TLS). The perception unit includes three synchronised global shutter colour cameras, an automotive 3D LiDAR scanner, and an inertial sensor - all precisely calibrated. We also establish benchmarks for tasks involving localisation, reconstruction, and novel-view synthesis, which enable the evaluation of Simultaneous Localisation and Mapping (SLAM) methods, Structure-from-Motion (SfM) and Multi-view Stereo (MVS) methods as well as radiance field methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting. To evaluate 3D reconstruction the TLS 3D models are used as ground truth. Localisation ground truth is computed by registering the mobile LiDAR scans to the TLS 3D models. Radiance field methods are evaluated not only with poses sampled from the input trajectory, but also from viewpoints that are from trajectories which are distant from the training poses. Our evaluation demonstrates a key limitation of state-of-the-art radiance field methods: we show that they tend to overfit to the training poses/images and do not generalise well to out-of-sequence poses. They also underperform in 3D reconstruction compared to MVS systems using the same visual inputs. Our dataset and benchmarks are intended to facilitate better integration of radiance field methods and SLAM systems. The raw and processed data, along with software for parsing and evaluation, can be accessed at https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07809v1">SplatFill: 3D Scene Inpainting via Depth-Guided Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has enabled the creation of highly realistic 3D scene representations from sets of multi-view images. However, inpainting missing regions, whether due to occlusion or scene editing, remains a challenging task, often leading to blurry details, artifacts, and inconsistent geometry. In this work, we introduce SplatFill, a novel depth-guided approach for 3DGS scene inpainting that achieves state-of-the-art perceptual quality and improved efficiency. Our method combines two key ideas: (1) joint depth-based and object-based supervision to ensure inpainted Gaussians are accurately placed in 3D space and aligned with surrounding geometry, and (2) we propose a consistency-aware refinement scheme that selectively identifies and corrects inconsistent regions without disrupting the rest of the scene. Evaluations on the SPIn-NeRF dataset demonstrate that SplatFill not only surpasses existing NeRF-based and 3DGS-based inpainting methods in visual fidelity but also reduces training time by 24.5%. Qualitative results show our method delivers sharper details, fewer artifacts, and greater coherence across challenging viewpoints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07774v1">HairGS: Hair Strand Reconstruction based on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 This is the arXiv preprint of the paper "Hair Strand Reconstruction based on 3D Gaussian Splatting" published at BMVC 2025. Project website: https://yimin-pan.github.io/hair-gs/
    </div>
    <details class="paper-abstract">
      Human hair reconstruction is a challenging problem in computer vision, with growing importance for applications in virtual reality and digital human modeling. Recent advances in 3D Gaussians Splatting (3DGS) provide efficient and explicit scene representations that naturally align with the structure of hair strands. In this work, we extend the 3DGS framework to enable strand-level hair geometry reconstruction from multi-view images. Our multi-stage pipeline first reconstructs detailed hair geometry using a differentiable Gaussian rasterizer, then merges individual Gaussian segments into coherent strands through a novel merging scheme, and finally refines and grows the strands under photometric supervision. While existing methods typically evaluate reconstruction quality at the geometric level, they often neglect the connectivity and topology of hair strands. To address this, we propose a new evaluation metric that serves as a proxy for assessing topological accuracy in strand reconstruction. Extensive experiments on both synthetic and real-world datasets demonstrate that our method robustly handles a wide range of hairstyles and achieves efficient reconstruction, typically completing within one hour. The project page can be found at: https://yimin-pan.github.io/hair-gs/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05752v2">PINGS: Gaussian Splatting Meets Distance Fields within a Point-Based Implicit Neural Map</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 15 pages, 8 figures, presented at RSS 2025
    </div>
    <details class="paper-abstract">
      Robots benefit from high-fidelity reconstructions of their environment, which should be geometrically accurate and photorealistic to support downstream tasks. While this can be achieved by building distance fields from range sensors and radiance fields from cameras, realising scalable incremental mapping of both fields consistently and at the same time with high quality is challenging. In this paper, we propose a novel map representation that unifies a continuous signed distance field and a Gaussian splatting radiance field within an elastic and compact point-based implicit neural map. By enforcing geometric consistency between these fields, we achieve mutual improvements by exploiting both modalities. We present a novel LiDAR-visual SLAM system called PINGS using the proposed map representation and evaluate it on several challenging large-scale datasets. Experimental results demonstrate that PINGS can incrementally build globally consistent distance and radiance fields encoded with a compact set of neural points. Compared to state-of-the-art methods, PINGS achieves superior photometric and geometric rendering at novel views by constraining the radiance field with the distance field. Furthermore, by utilizing dense photometric cues and multi-view consistency from the radiance field, PINGS produces more accurate distance fields, leading to improved odometry estimation and mesh reconstruction. We also provide an open-source implementation of PING at: https://github.com/PRBonn/PINGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07493v1">DiGS: Accurate and Complete Surface Reconstruction from 3D Gaussians via Direct SDF Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as a powerful paradigm for photorealistic view synthesis, representing scenes with spatially distributed Gaussian primitives. While highly effective for rendering, achieving accurate and complete surface reconstruction remains challenging due to the unstructured nature of the representation and the absence of explicit geometric supervision. In this work, we propose DiGS, a unified framework that embeds Signed Distance Field (SDF) learning directly into the 3DGS pipeline, thereby enforcing strong and interpretable surface priors. By associating each Gaussian with a learnable SDF value, DiGS explicitly aligns primitives with underlying geometry and improves cross-view consistency. To further ensure dense and coherent coverage, we design a geometry-guided grid growth strategy that adaptively distributes Gaussians along geometry-consistent regions under a multi-scale hierarchy. Extensive experiments on standard benchmarks, including DTU, Mip-NeRF 360, and Tanks& Temples, demonstrate that DiGS consistently improves reconstruction accuracy and completeness while retaining high rendering fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07435v1">DreamLifting: A Plug-in Module Lifting MV Diffusion Models for 3D Asset Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 14 pages, 7 figures, project page: https://zx-yin.github.io/dreamlifting/
    </div>
    <details class="paper-abstract">
      The labor- and experience-intensive creation of 3D assets with physically based rendering (PBR) materials demands an autonomous 3D asset creation pipeline. However, most existing 3D generation methods focus on geometry modeling, either baking textures into simple vertex colors or leaving texture synthesis to post-processing with image diffusion models. To achieve end-to-end PBR-ready 3D asset generation, we present Lightweight Gaussian Asset Adapter (LGAA), a novel framework that unifies the modeling of geometry and PBR materials by exploiting multi-view (MV) diffusion priors from a novel perspective. The LGAA features a modular design with three components. Specifically, the LGAA Wrapper reuses and adapts network layers from MV diffusion models, which encapsulate knowledge acquired from billions of images, enabling better convergence in a data-efficient manner. To incorporate multiple diffusion priors for geometry and PBR synthesis, the LGAA Switcher aligns multiple LGAA Wrapper layers encapsulating different knowledge. Then, a tamed variational autoencoder (VAE), termed LGAA Decoder, is designed to predict 2D Gaussian Splatting (2DGS) with PBR channels. Finally, we introduce a dedicated post-processing procedure to effectively extract high-quality, relightable mesh assets from the resulting 2DGS. Extensive quantitative and qualitative experiments demonstrate the superior performance of LGAA with both text-and image-conditioned MV diffusion models. Additionally, the modular design enables flexible incorporation of multiple diffusion priors, and the knowledge-preserving scheme leads to efficient convergence trained on merely 69k multi-view instances. Our code, pre-trained weights, and the dataset used will be publicly available via our project page: https://zx-yin.github.io/dreamlifting/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06685v2">VIM-GS: Visual-Inertial Monocular Gaussian Splatting via Object-level Guidance in Large Scenes</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Withdrawn due to an error in the author list & incomplete experimental results
    </div>
    <details class="paper-abstract">
      VIM-GS is a Gaussian Splatting (GS) framework using monocular images for novel-view synthesis (NVS) in large scenes. GS typically requires accurate depth to initiate Gaussian ellipsoids using RGB-D/stereo cameras. Their limited depth sensing range makes it difficult for GS to work in large scenes. Monocular images, however, lack depth to guide the learning and lead to inferior NVS results. Although large foundation models (LFMs) for monocular depth estimation are available, they suffer from cross-frame inconsistency, inaccuracy for distant scenes, and ambiguity in deceptive texture cues. This paper aims to generate dense, accurate depth images from monocular RGB inputs for high-definite GS rendering. The key idea is to leverage the accurate but sparse depth from visual-inertial Structure-from-Motion (SfM) to refine the dense but coarse depth from LFMs. To bridge the sparse input and dense output, we propose an object-segmented depth propagation algorithm that renders the depth of pixels of structured objects. Then we develop a dynamic depth refinement module to handle the crippled SfM depth of dynamic objects and refine the coarse LFM depth. Experiments using public and customized datasets demonstrate the superior rendering quality of VIM-GS in large scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06685v1">VIM-GS: Visual-Inertial Monocular Gaussian Splatting via Object-level Guidance in Large Scenes</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      VIM-GS is a Gaussian Splatting (GS) framework using monocular images for novel-view synthesis (NVS) in large scenes. GS typically requires accurate depth to initiate Gaussian ellipsoids using RGB-D/stereo cameras. Their limited depth sensing range makes it difficult for GS to work in large scenes. Monocular images, however, lack depth to guide the learning and lead to inferior NVS results. Although large foundation models (LFMs) for monocular depth estimation are available, they suffer from cross-frame inconsistency, inaccuracy for distant scenes, and ambiguity in deceptive texture cues. This paper aims to generate dense, accurate depth images from monocular RGB inputs for high-definite GS rendering. The key idea is to leverage the accurate but sparse depth from visual-inertial Structure-from-Motion (SfM) to refine the dense but coarse depth from LFMs. To bridge the sparse input and dense output, we propose an object-segmented depth propagation algorithm that renders the depth of pixels of structured objects. Then we develop a dynamic depth refinement module to handle the crippled SfM depth of dynamic objects and refine the coarse LFM depth. Experiments using public and customized datasets demonstrate the superior rendering quality of VIM-GS in large scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06433v1">Real-time Photorealistic Mapping for Situational Awareness in Robot Teleoperation</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Achieving efficient remote teleoperation is particularly challenging in unknown environments, as the teleoperator must rapidly build an understanding of the site's layout. Online 3D mapping is a proven strategy to tackle this challenge, as it enables the teleoperator to progressively explore the site from multiple perspectives. However, traditional online map-based teleoperation systems struggle to generate visually accurate 3D maps in real-time due to the high computational cost involved, leading to poor teleoperation performances. In this work, we propose a solution to improve teleoperation efficiency in unknown environments. Our approach proposes a novel, modular and efficient GPU-based integration between recent advancement in gaussian splatting SLAM and existing online map-based teleoperation systems. We compare the proposed solution against state-of-the-art teleoperation systems and validate its performances through real-world experiments using an aerial vehicle. The results show significant improvements in decision-making speed and more accurate interaction with the environment, leading to greater teleoperation efficiency. In doing so, our system enhances remote teleoperation by seamlessly integrating photorealistic mapping generation with real-time performances, enabling effective teleoperation in unfamiliar environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06400v1">3DOF+Quantization: 3DGS quantization for large scenes with limited Degrees of Freedom</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a major breakthrough in 3D scene reconstruction. With a number of views of a given object or scene, the algorithm trains a model composed of 3D gaussians, which enables the production of novel views from arbitrary points of view. This freedom of movement is referred to as 6DoF for 6 degrees of freedom: a view is produced for any position (3 degrees), orientation of camera (3 other degrees). On large scenes, though, the input views are acquired from a limited zone in space, and the reconstruction is valuable for novel views from the same zone, even if the scene itself is almost unlimited in size. We refer to this particular case as 3DoF+, meaning that the 3 degrees of freedom of camera position are limited to small offsets around the central position. Considering the problem of coordinate quantization, the impact of position error on the projection error in pixels is studied. It is shown that the projection error is proportional to the squared inverse distance of the point being projected. Consequently, a new quantization scheme based on spherical coordinates is proposed. Rate-distortion performance of the proposed method are illustrated on the well-known Garden scene.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11854v2">ComplicitSplat: Downstream Models are Vulnerable to Blackbox Attacks by 3D Gaussian Splat Camouflages</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 7 pages, 6 figures
    </div>
    <details class="paper-abstract">
      As 3D Gaussian Splatting (3DGS) gains rapid adoption in safety-critical tasks for efficient novel-view synthesis from static images, how might an adversary tamper images to cause harm? We introduce ComplicitSplat, the first attack that exploits standard 3DGS shading methods to create viewpoint-specific camouflage - colors and textures that change with viewing angle - to embed adversarial content in scene objects that are visible only from specific viewpoints and without requiring access to model architecture or weights. Our extensive experiments show that ComplicitSplat generalizes to successfully attack a variety of popular detector - both single-stage, multi-stage, and transformer-based models on both real-world capture of physical objects and synthetic scenes. To our knowledge, this is the first black-box attack on downstream object detectors using 3DGS, exposing a novel safety risk for applications like autonomous navigation and other mission-critical robotic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07021v1">MEGS$^{2}$: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 14 pages, 4 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a dominant novel-view synthesis technique, but its high memory consumption severely limits its applicability on edge devices. A growing number of 3DGS compression methods have been proposed to make 3DGS more efficient, yet most only focus on storage compression and fail to address the critical bottleneck of rendering memory. To address this problem, we introduce MEGS$^{2}$, a novel memory-efficient framework that tackles this challenge by jointly optimizing two key factors: the total primitive number and the parameters per primitive, achieving unprecedented memory compression. Specifically, we replace the memory-intensive spherical harmonics with lightweight arbitrarily-oriented spherical Gaussian lobes as our color representations. More importantly, we propose a unified soft pruning framework that models primitive-number and lobe-number pruning as a single constrained optimization problem. Experiments show that MEGS$^{2}$ achieves a 50% static VRAM reduction and a 40% rendering VRAM reduction compared to existing methods, while maintaining comparable rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10906v2">ShapeSplat: A Large-scale Dataset of Gaussian Splats and Their Self-Supervised Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-09-06
      | 💬 Accepted as 3DV'25 Oral, project page: https://unique1i.github.io/ShapeSplat_webpage/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become the de facto method of 3D representation in many vision tasks. This calls for the 3D understanding directly in this representation space. To facilitate the research in this direction, we first build ShapeSplat, a large-scale dataset of 3DGS using the commonly used ShapeNet, ModelNet and Objaverse datasets. Our dataset ShapeSplat consists of 206K objects spanning over 87 unique categories, whose labels are in accordance with the respective datasets. The creation of this dataset utilized the compute equivalent of 3.8 GPU years on a TITAN XP GPU. We utilize our dataset for unsupervised pretraining and supervised finetuning for classification and segmentation tasks. To this end, we introduce Gaussian-MAE, which highlights the unique benefits of representation learning from Gaussian parameters. Through exhaustive experiments, we provide several valuable insights. In particular, we show that (1) the distribution of the optimized GS centroids significantly differs from the uniformly sampled point cloud (used for initialization) counterpart; (2) this change in distribution results in degradation in classification but improvement in segmentation tasks when using only the centroids; (3) to leverage additional Gaussian parameters, we propose Gaussian feature grouping in a normalized feature space, along with splats pooling layer, offering a tailored solution to effectively group and embed similar Gaussians, which leads to notable improvement in finetuning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05216v1">Toward Distributed 3D Gaussian Splatting for High-Resolution Isosurface Visualization</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      We present a multi-GPU extension of the 3D Gaussian Splatting (3D-GS) pipeline for scientific visualization. Building on previous work that demonstrated high-fidelity isosurface reconstruction using Gaussian primitives, we incorporate a multi-GPU training backend adapted from Grendel-GS to enable scalable processing of large datasets. By distributing optimization across GPUs, our method improves training throughput and supports high-resolution reconstructions that exceed single-GPU capacity. In our experiments, the system achieves a 5.6X speedup on the Kingsnake dataset (4M Gaussians) using four GPUs compared to a single-GPU baseline, and successfully trains the Miranda dataset (18M Gaussians) that is an infeasible task on a single A100 GPU. This work lays the groundwork for integrating 3D-GS into HPC-based scientific workflows, enabling real-time post hoc and in situ visualization of complex simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05075v1">GeoSplat: A Deep Dive into Geometry-Constrained Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      A few recent works explored incorporating geometric priors to regularize the optimization of Gaussian splatting, further improving its performance. However, those early studies mainly focused on the use of low-order geometric priors (e.g., normal vector), and they are also unreliably estimated by noise-sensitive methods, like local principal component analysis. To address their limitations, we first present GeoSplat, a general geometry-constrained optimization framework that exploits both first-order and second-order geometric quantities to improve the entire training pipeline of Gaussian splatting, including Gaussian initialization, gradient update, and densification. As an example, we initialize the scales of 3D Gaussian primitives in terms of principal curvatures, leading to a better coverage of the object surface than random initialization. Secondly, based on certain geometric structures (e.g., local manifold), we introduce efficient and noise-robust estimation methods that provide dynamic geometric priors for our framework. We conduct extensive experiments on multiple datasets for novel view synthesis, showing that our framework: GeoSplat, significantly improves the performance of Gaussian splatting and outperforms previous baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04859v1">CoRe-GS: Coarse-to-Refined Gaussian Splatting with Semantic Object Focus</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Mobile reconstruction for autonomous aerial robotics holds strong potential for critical applications such as tele-guidance and disaster response. These tasks demand both accurate 3D reconstruction and fast scene processing. Instead of reconstructing the entire scene in detail, it is often more efficient to focus on specific objects, i.e., points of interest (PoIs). Mobile robots equipped with advanced sensing can usually detect these early during data acquisition or preliminary analysis, reducing the need for full-scene optimization. Gaussian Splatting (GS) has recently shown promise in delivering high-quality novel view synthesis and 3D representation by an incremental learning process. Extending GS with scene editing, semantics adds useful per-splat features to isolate objects effectively. Semantic 3D Gaussian editing can already be achieved before the full training cycle is completed, reducing the overall training time. Moreover, the semantically relevant area, the PoI, is usually already known during capturing. To balance high-quality reconstruction with reduced training time, we propose CoRe-GS. We first generate a coarse segmentation-ready scene with semantic GS and then refine it for the semantic object using our novel color-based effective filtering for effective object isolation. This is speeding up the training process to be about a quarter less than a full training cycle for semantic GS. We evaluate our approach on two datasets, SCRREAM (real-world, outdoor) and NeRDS 360 (synthetic, indoor), showing reduced runtime and higher novel-view-synthesis quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17422v3">Multimodal LLM Guided Exploration and Active Mapping using Fisher Information</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 ICCV 2025
    </div>
    <details class="paper-abstract">
      We present an active mapping system that plans for both long-horizon exploration goals and short-term actions using a 3D Gaussian Splatting (3DGS) representation. Existing methods either do not take advantage of recent developments in multimodal Large Language Models (LLM) or do not consider challenges in localization uncertainty, which is critical in embodied agents. We propose employing multimodal LLMs for long-horizon planning in conjunction with detailed motion planning using our information-based objective. By leveraging high-quality view synthesis from our 3DGS representation, our method employs a multimodal LLM as a zero-shot planner for long-horizon exploration goals from the semantic perspective. We also introduce an uncertainty-aware path proposal and selection algorithm that balances the dual objectives of maximizing the information gain for the environment while minimizing the cost of localization errors. Experiments conducted on the Gibson and Habitat-Matterport 3D datasets demonstrate state-of-the-art results of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14014v2">Online 3D Gaussian Splatting Modeling with Novel View Selection</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      This study addresses the challenge of generating online 3D Gaussian Splatting (3DGS) models from RGB-only frames. Previous studies have employed dense SLAM techniques to estimate 3D scenes from keyframes for 3DGS model construction. However, these methods are limited by their reliance solely on keyframes, which are insufficient to capture an entire scene, resulting in incomplete reconstructions. Moreover, building a generalizable model requires incorporating frames from diverse viewpoints to achieve broader scene coverage. However, online processing restricts the use of many frames or extensive training iterations. Therefore, we propose a novel method for high-quality 3DGS modeling that improves model completeness through adaptive view selection. By analyzing reconstruction quality online, our approach selects optimal non-keyframes for additional training. By integrating both keyframes and selected non-keyframes, the method refines incomplete regions from diverse viewpoints, significantly enhancing completeness. We also present a framework that incorporates an online multi-view stereo approach, ensuring consistency in 3D information throughout the 3DGS modeling process. Experimental results demonstrate that our method outperforms state-of-the-art methods, delivering exceptional performance in complex outdoor scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05515v1">Visibility-Aware Language Aggregation for Open-Vocabulary Segmentation in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Recently, distilling open-vocabulary language features from 2D images into 3D Gaussians has attracted significant attention. Although existing methods achieve impressive language-based interactions of 3D scenes, we observe two fundamental issues: background Gaussians contributing negligibly to a rendered pixel get the same feature as the dominant foreground ones, and multi-view inconsistencies due to view-specific noise in language embeddings. We introduce Visibility-Aware Language Aggregation (VALA), a lightweight yet effective method that computes marginal contributions for each ray and applies a visibility-aware gate to retain only visible Gaussians. Moreover, we propose a streaming weighted geometric median in cosine space to merge noisy multi-view features. Our method yields a robust, view-consistent language feature embedding in a fast and memory-efficient manner. VALA improves open-vocabulary localization and segmentation across reference datasets, consistently surpassing existing works.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04379v1">SSGaussian: Semantic-Aware and Structure-Preserving 3D Style Transfer</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Recent advancements in neural representations, such as Neural Radiance Fields and 3D Gaussian Splatting, have increased interest in applying style transfer to 3D scenes. While existing methods can transfer style patterns onto 3D-consistent neural representations, they struggle to effectively extract and transfer high-level style semantics from the reference style image. Additionally, the stylized results often lack structural clarity and separation, making it difficult to distinguish between different instances or objects within the 3D scene. To address these limitations, we propose a novel 3D style transfer pipeline that effectively integrates prior knowledge from pretrained 2D diffusion models. Our pipeline consists of two key stages: First, we leverage diffusion priors to generate stylized renderings of key viewpoints. Then, we transfer the stylized key views onto the 3D representation. This process incorporates two innovative designs. The first is cross-view style alignment, which inserts cross-view attention into the last upsampling block of the UNet, allowing feature interactions across multiple key views. This ensures that the diffusion model generates stylized key views that maintain both style fidelity and instance-level consistency. The second is instance-level style transfer, which effectively leverages instance-level consistency across stylized key views and transfers it onto the 3D representation. This results in a more structured, visually coherent, and artistically enriched stylization. Extensive qualitative and quantitative experiments demonstrate that our 3D style transfer pipeline significantly outperforms state-of-the-art methods across a wide range of scenes, from forward-facing to challenging 360-degree environments. Visit our project page https://jm-xu.github.io/SSGaussian for immersive visualization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06897v3">ActiveGAMER: Active GAussian Mapping through Efficient Rendering</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 Accepted to CVPR2025. Project page: https://oppo-us-research.github.io/ActiveGAMER-website/. Code: https://github.com/oppo-us-research/ActiveGAMER
    </div>
    <details class="paper-abstract">
      We introduce ActiveGAMER, an active mapping system that utilizes 3D Gaussian Splatting (3DGS) to achieve high-quality, real-time scene mapping and exploration. Unlike traditional NeRF-based methods, which are computationally demanding and restrict active mapping performance, our approach leverages the efficient rendering capabilities of 3DGS, allowing effective and efficient exploration in complex environments. The core of our system is a rendering-based information gain module that dynamically identifies the most informative viewpoints for next-best-view planning, enhancing both geometric and photometric reconstruction accuracy. ActiveGAMER also integrates a carefully balanced framework, combining coarse-to-fine exploration, post-refinement, and a global-local keyframe selection strategy to maximize reconstruction completeness and fidelity. Our system autonomously explores and reconstructs environments with state-of-the-art geometric and photometric accuracy and completeness, significantly surpassing existing approaches in both aspects. Extensive evaluations on benchmark datasets such as Replica and MP3D highlight ActiveGAMER's effectiveness in active mapping tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06269v3">BayesSDF: Surface-Based Laplacian Uncertainty Estimation for 3D Geometry with Neural Signed Distance Fields</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 ICCV 2025 Workshops (11 Pages, 6 Figures, 2 Tables)
    </div>
    <details class="paper-abstract">
      Accurate surface estimation is critical for downstream tasks in scientific simulation, and quantifying uncertainty in implicit neural 3D representations still remains a substantial challenge due to computational inefficiencies, scalability issues, and geometric inconsistencies. However, current neural implicit surface models do not offer a principled way to quantify uncertainty, limiting their reliability in real-world applications. Inspired by recent probabilistic rendering approaches, we introduce BayesSDF, a novel probabilistic framework for uncertainty estimation in neural implicit 3D representations. Unlike radiance-based models such as Neural Radiance Fields (NeRF) or 3D Gaussian Splatting, Signed Distance Functions (SDFs) provide continuous, differentiable surface representations, making them especially well-suited for uncertainty-aware modeling. BayesSDF applies a Laplace approximation over SDF weights and derives Hessian-based metrics to estimate local geometric instability. We empirically demonstrate that these uncertainty estimates correlate strongly with surface reconstruction error across both synthetic and real-world benchmarks. By enabling surface-aware uncertainty quantification, BayesSDF lays the groundwork for more robust, interpretable, and actionable 3D perception systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00831v2">UPGS: Unified Pose-aware Gaussian Splatting for Dynamic Scene Deblurring</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic 3D scenes from monocular video has broad applications in AR/VR, robotics, and autonomous navigation, but often fails due to severe motion blur caused by camera and object motion. Existing methods commonly follow a two-step pipeline, where camera poses are first estimated and then 3D Gaussians are optimized. Since blurring artifacts usually undermine pose estimation, pose errors could be accumulated to produce inferior reconstruction results. To address this issue, we introduce a unified optimization framework by incorporating camera poses as learnable parameters complementary to 3DGS attributes for end-to-end optimization. Specifically, we recast camera and object motion as per-primitive SE(3) affine transformations on 3D Gaussians and formulate a unified optimization objective. For stable optimization, we introduce a three-stage training schedule that optimizes camera poses and Gaussians alternatively. Particularly, 3D Gaussians are first trained with poses being fixed, and then poses are optimized with 3D Gaussians being untouched. Finally, all learnable parameters are optimized together. Extensive experiments on the Stereo Blur dataset and challenging real-world sequences demonstrate that our method achieves significant gains in reconstruction quality and pose estimation accuracy over prior dynamic deblurring methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08624v2">Communication Efficient Robotic Mixed Reality with Gaussian Splatting Cross-Layer Optimization</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 14 pages, 18 figures, to appear in IEEE Transactions on Cognitive Communications and Networking
    </div>
    <details class="paper-abstract">
      Realizing low-cost communication in robotic mixed reality (RoboMR) systems presents a challenge, due to the necessity of uploading high-resolution images through wireless channels. This paper proposes Gaussian splatting (GS) RoboMR (GSMR), which enables the simulator to opportunistically render a photo-realistic view from the robot's pose by calling ``memory'' from a GS model, thus reducing the need for excessive image uploads. However, the GS model may involve discrepancies compared to the actual environments. To this end, a GS cross-layer optimization (GSCLO) framework is further proposed, which jointly optimizes content switching (i.e., deciding whether to upload image or not) and power allocation (i.e., adjusting to content profiles) across different frames by minimizing a newly derived GSMR loss function. The GSCLO problem is addressed by an accelerated penalty optimization (APO) algorithm that reduces computational complexity by over $10$x compared to traditional branch-and-bound and search algorithms. Moreover, variants of GSCLO are presented to achieve robust, low-power, and multi-robot GSMR. Extensive experiments demonstrate that the proposed GSMR paradigm and GSCLO method achieve significant improvements over existing benchmarks on both wheeled and legged robots in terms of diverse metrics in various scenarios. For the first time, it is found that RoboMR can be achieved with ultra-low communication costs, and mixture of data is useful for enhancing GS performance in dynamic scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00911v2">GS-TG: 3D Gaussian Splatting Accelerator with Tile Grouping for Reducing Redundant Sorting while Preserving Rasterization Efficiency</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 DAC 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D-GS) has emerged as a promising alternative to neural radiance fields (NeRF) as it offers high speed as well as high image quality in novel view synthesis. Despite these advancements, 3D-GS still struggles to meet the frames per second (FPS) demands of real-time applications. In this paper, we introduce GS-TG, a tile-grouping-based accelerator that enhances 3D-GS rendering speed by reducing redundant sorting operations and preserving rasterization efficiency. GS-TG addresses a critical trade-off issue in 3D-GS rendering: increasing the tile size effectively reduces redundant sorting operations, but it concurrently increases unnecessary rasterization computations. So, during sorting of the proposed approach, GS-TG groups small tiles (for making large tiles) to share sorting operations across tiles within each group, significantly reducing redundant computations. During rasterization, a bitmask assigned to each Gaussian identifies relevant small tiles, to enable efficient sharing of sorting results. Consequently, GS-TG enables sorting to be performed as if a large tile size is used by grouping tiles during the sorting stage, while allowing rasterization to proceed with the original small tiles by using bitmasks in the rasterization stage. GS-TG is a lossless method requiring no retraining or fine-tuning and it can be seamlessly integrated with previous 3D-GS optimization techniques. Experimental results show that GS-TG achieves an average speed-up of 1.54 times over state-of-the-art 3D-GS accelerators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03775v1">ContraGS: Codebook-Condensed and Trainable Gaussian Splatting for Fast, Memory-Efficient Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a state-of-art technique to model real-world scenes with high quality and real-time rendering. Typically, a higher quality representation can be achieved by using a large number of 3D Gaussians. However, using large 3D Gaussian counts significantly increases the GPU device memory for storing model parameters. A large model thus requires powerful GPUs with high memory capacities for training and has slower training/rendering latencies due to the inefficiencies of memory access and data movement. In this work, we introduce ContraGS, a method to enable training directly on compressed 3DGS representations without reducing the Gaussian Counts, and thus with a little loss in model quality. ContraGS leverages codebooks to compactly store a set of Gaussian parameter vectors throughout the training process, thereby significantly reducing memory consumption. While codebooks have been demonstrated to be highly effective at compressing fully trained 3DGS models, directly training using codebook representations is an unsolved challenge. ContraGS solves the problem of learning non-differentiable parameters in codebook-compressed representations by posing parameter estimation as a Bayesian inference problem. To this end, ContraGS provides a framework that effectively uses MCMC sampling to sample over a posterior distribution of these compressed representations. With ContraGS, we demonstrate that ContraGS significantly reduces the peak memory during training (on average 3.49X) and accelerated training and rendering (1.36X and 1.88X on average, respectively), while retraining close to state-of-art quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05740v2">Micro-splatting: Multistage Isotropy-informed Covariance Regularization Optimization for High-Fidelity 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 This work has been submitted to journal for potential publication
    </div>
    <details class="paper-abstract">
      High-fidelity 3D Gaussian Splatting methods excel at capturing fine textures but often overlook model compactness, resulting in massive splat counts, bloated memory, long training, and complex post-processing. We present Micro-Splatting: Two-Stage Adaptive Growth and Refinement, a unified, in-training pipeline that preserves visual detail while drastically reducing model complexity without any post-processing or auxiliary neural modules. In Stage I (Growth), we introduce a trace-based covariance regularization to maintain near-isotropic Gaussians, mitigating low-pass filtering in high-frequency regions and improving spherical-harmonic color fitting. We then apply gradient-guided adaptive densification that subdivides splats only in visually complex regions, leaving smooth areas sparse. In Stage II (Refinement), we prune low-impact splats using a simple opacity-scale importance score and merge redundant neighbors via lightweight spatial and feature thresholds, producing a lean yet detail-rich model. On four object-centric benchmarks, Micro-Splatting reduces splat count and model size by up to 60% and shortens training by 20%, while matching or surpassing state-of-the-art PSNR, SSIM, and LPIPS in real-time rendering. These results demonstrate that Micro-Splatting delivers both compactness and high fidelity in a single, efficient, end-to-end framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05254v3">CF3: Compact and Fast 3D Feature Fields</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 ICCV 2025, Project Page: https://jjoonii.github.io/cf3-website/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has begun incorporating rich information from 2D foundation models. However, most approaches rely on a bottom-up optimization process that treats raw 2D features as ground truth, incurring increased computational costs. We propose a top-down pipeline for constructing compact and fast 3D Gaussian feature fields, namely, CF3. We first perform a fast weighted fusion of multi-view 2D features with pre-trained Gaussians. This approach enables training a per-Gaussian autoencoder directly on the lifted features, instead of training autoencoders in the 2D domain. As a result, the autoencoder better aligns with the feature distribution. More importantly, we introduce an adaptive sparsification method that optimizes the Gaussian attributes of the feature field while pruning and merging the redundant Gaussians, constructing an efficient representation with preserved geometric details. Our approach achieves a competitive 3D feature field using as little as 5% of the Gaussians compared to Feature-3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18677v2">Reconstructing Tornadoes in 3D with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Accurately reconstructing the 3D structure of tornadoes is critically important for understanding and preparing for this highly destructive weather phenomenon. While modern 3D scene reconstruction techniques, such as 3D Gaussian splatting (3DGS), could provide a valuable tool for reconstructing the 3D structure of tornados, at present we are critically lacking a controlled tornado dataset with which to develop and validate these tools. In this work we capture and release a novel multiview dataset of a small lab-based tornado. We demonstrate one can effectively reconstruct and visualize the 3D structure of this tornado using 3DGS.
    </details>
</div>
