# gaussian splatting - 2025_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09663v1">Self-Supervised Multi-Part Articulated Objects Modeling via Deformable Gaussian Splatting and Progressive Primitive Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Articulated objects are ubiquitous in everyday life, and accurate 3D representations of their geometry and motion are critical for numerous applications. However, in the absence of human annotation, existing approaches still struggle to build a unified representation for objects that contain multiple movable parts. We introduce DeGSS, a unified framework that encodes articulated objects as deformable 3D Gaussian fields, embedding geometry, appearance, and motion in one compact representation. Each interaction state is modeled as a smooth deformation of a shared field, and the resulting deformation trajectories guide a progressive coarse-to-fine part segmentation that identifies distinct rigid components, all in an unsupervised manner. The refined field provides a spatially continuous, fully decoupled description of every part, supporting part-level reconstruction and precise modeling of their kinematic relationships. To evaluate generalization and realism, we enlarge the synthetic PartNet-Mobility benchmark and release RS-Art, a real-to-sim dataset that pairs RGB captures with accurately reverse-engineered 3D models. Extensive experiments demonstrate that our method outperforms existing methods in both accuracy and stability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13639v2">4D Radar-Inertial Odometry based on Gaussian Modeling and Multi-Hypothesis Scan Matching</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Our code and results can be publicly accessed at: https://github.com/robotics-upo/gaussian-rio-cpp
    </div>
    <details class="paper-abstract">
      4D millimeter-wave (mmWave) radars are sensors that provide robustness against adverse weather conditions (rain, snow, fog, etc.), and as such they are increasingly being used for odometry and SLAM applications. However, the noisy and sparse nature of the returned scan data proves to be a challenging obstacle for existing point cloud matching based solutions, especially those originally intended for more accurate sensors such as LiDAR. Inspired by visual odometry research around 3D Gaussian Splatting, in this paper we propose using freely positioned 3D Gaussians to create a summarized representation of a radar point cloud tolerant to sensor noise, and subsequently leverage its inherent probability distribution function for registration (similar to NDT). Moreover, we propose simultaneously optimizing multiple scan matching hypotheses in order to further increase the robustness of the system against local optima of the function. Finally, we fuse our Gaussian modeling and scan matching algorithms into an EKF radar-inertial odometry system designed after current best practices. Experiments using publicly available 4D radar datasets show that our Gaussian-based odometry is comparable to existing registration algorithms, outperforming them in several sequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09534v1">Gaussian Herding across Pens: An Optimal Transport Perspective on Global Gaussian Reduction for 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 18 pages, 8 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for radiance field rendering, but it typically requires millions of redundant Gaussian primitives, overwhelming memory and rendering budgets. Existing compaction approaches address this by pruning Gaussians based on heuristic importance scores, without global fidelity guarantee. To bridge this gap, we propose a novel optimal transport perspective that casts 3DGS compaction as global Gaussian mixture reduction. Specifically, we first minimize the composite transport divergence over a KD-tree partition to produce a compact geometric representation, and then decouple appearance from geometry by fine-tuning color and opacity attributes with far fewer Gaussian primitives. Experiments on benchmark datasets show that our method (i) yields negligible loss in rendering quality (PSNR, SSIM, LPIPS) compared to vanilla 3DGS with only 10% Gaussians; and (ii) consistently outperforms state-of-the-art 3DGS compaction techniques. Notably, our method is applicable to any stage of vanilla or accelerated 3DGS pipelines, providing an efficient and agnostic pathway to lightweight neural rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09518v1">HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic 3D scenes from monocular videos remains a fundamental challenge in 3D vision. While 3D Gaussian Splatting (3DGS) achieves real-time rendering in static settings, extending it to dynamic scenes is challenging due to the difficulty of learning structured and temporally consistent motion representations. This challenge often manifests as three limitations in existing methods: redundant Gaussian updates, insufficient motion supervision, and weak modeling of complex non-rigid deformations. These issues collectively hinder coherent and efficient dynamic reconstruction. To address these limitations, we propose HAIF-GS, a unified framework that enables structured and consistent dynamic modeling through sparse anchor-driven deformation. It first identifies motion-relevant regions via an Anchor Filter to suppresses redundant updates in static areas. A self-supervised Induced Flow-Guided Deformation module induces anchor motion using multi-frame feature aggregation, eliminating the need for explicit flow labels. To further handle fine-grained deformations, a Hierarchical Anchor Propagation mechanism increases anchor resolution based on motion complexity and propagates multi-level transformations. Extensive experiments on synthetic and real-world benchmarks validate that HAIF-GS significantly outperforms prior dynamic 3DGS methods in rendering quality, temporal coherence, and reconstruction efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08777v2">Gaussian2Scene: 3D Scene Representation Learning via Self-supervised Learning with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Self-supervised learning (SSL) for point cloud pre-training has become a cornerstone for many 3D vision tasks, enabling effective learning from large-scale unannotated data. At the scene level, existing SSL methods often incorporate volume rendering into the pre-training framework, using RGB-D images as reconstruction signals to facilitate cross-modal learning. This strategy promotes alignment between 2D and 3D modalities and enables the model to benefit from rich visual cues in the RGB-D inputs. However, these approaches are limited by their reliance on implicit scene representations and high memory demands. Furthermore, since their reconstruction objectives are applied only in 2D space, they often fail to capture underlying 3D geometric structures. To address these challenges, we propose Gaussian2Scene, a novel scene-level SSL framework that leverages the efficiency and explicit nature of 3D Gaussian Splatting (3DGS) for pre-training. The use of 3DGS not only alleviates the computational burden associated with volume rendering but also supports direct 3D scene reconstruction, thereby enhancing the geometric understanding of the backbone network. Our approach follows a progressive two-stage training strategy. In the first stage, a dual-branch masked autoencoder learns both 2D and 3D scene representations. In the second stage, we initialize training with reconstructed point clouds and further supervise learning using the geometric locations of Gaussian primitives and rendered RGB images. This process reinforces both geometric and cross-modal learning. We demonstrate the effectiveness of Gaussian2Scene across several downstream 3D object detection tasks, showing consistent improvements over existing pre-training methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09479v1">TinySplat: Feedforward Approach for Generating Compact 3D Scene Representation</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      The recent development of feedforward 3D Gaussian Splatting (3DGS) presents a new paradigm to reconstruct 3D scenes. Using neural networks trained on large-scale multi-view datasets, it can directly infer 3DGS representations from sparse input views. Although the feedforward approach achieves high reconstruction speed, it still suffers from the substantial storage cost of 3D Gaussians. Existing 3DGS compression methods relying on scene-wise optimization are not applicable due to architectural incompatibilities. To overcome this limitation, we propose TinySplat, a complete feedforward approach for generating compact 3D scene representations. Built upon standard feedforward 3DGS methods, TinySplat integrates a training-free compression framework that systematically eliminates key sources of redundancy. Specifically, we introduce View-Projection Transformation (VPT) to reduce geometric redundancy by projecting geometric parameters into a more compact space. We further present Visibility-Aware Basis Reduction (VABR), which mitigates perceptual redundancy by aligning feature energy along dominant viewing directions via basis transformation. Lastly, spatial redundancy is addressed through an off-the-shelf video codec. Comprehensive experimental results on multiple benchmark datasets demonstrate that TinySplat achieves over 100x compression for 3D Gaussian data generated by feedforward methods. Compared to the state-of-the-art compression approach, we achieve comparable quality with only 6% of the storage size. Meanwhile, our compression framework requires only 25% of the encoding time and 1% of the decoding time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12894v2">FLoD: Integrating Flexible Level of Detail into 3D Gaussian Splatting for Customizable Rendering</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Project page: https://3dgs-flod.github.io/flod/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) and its subsequent works are restricted to specific hardware setups, either on only low-cost or on only high-end configurations. Approaches aimed at reducing 3DGS memory usage enable rendering on low-cost GPU but compromise rendering quality, which fails to leverage the hardware capabilities in the case of higher-end GPU. Conversely, methods that enhance rendering quality require high-end GPU with large VRAM, making such methods impractical for lower-end devices with limited memory capacity. Consequently, 3DGS-based works generally assume a single hardware setup and lack the flexibility to adapt to varying hardware constraints. To overcome this limitation, we propose Flexible Level of Detail (FLoD) for 3DGS. FLoD constructs a multi-level 3DGS representation through level-specific 3D scale constraints, where each level independently reconstructs the entire scene with varying detail and GPU memory usage. A level-by-level training strategy is introduced to ensure structural consistency across levels. Furthermore, the multi-level structure of FLoD allows selective rendering of image regions at different detail levels, providing additional memory-efficient rendering options. To our knowledge, among prior works which incorporate the concept of Level of Detail (LoD) with 3DGS, FLoD is the first to follow the core principle of LoD by offering adjustable options for a broad range of GPU settings. Experiments demonstrate that FLoD provides various rendering options with trade-offs between quality and memory usage, enabling real-time rendering under diverse memory constraints. Furthermore, we show that FLoD generalizes to different 3DGS frameworks, indicating its potential for integration into future state-of-the-art developments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09378v1">UniForward: Unified 3D Scene and Semantic Field Reconstruction via Feed-Forward Gaussian Splatting from Only Sparse-View Images</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      We propose a feed-forward Gaussian Splatting model that unifies 3D scene and semantic field reconstruction. Combining 3D scenes with semantic fields facilitates the perception and understanding of the surrounding environment. However, key challenges include embedding semantics into 3D representations, achieving generalizable real-time reconstruction, and ensuring practical applicability by using only images as input without camera parameters or ground truth depth. To this end, we propose UniForward, a feed-forward model to predict 3D Gaussians with anisotropic semantic features from only uncalibrated and unposed sparse-view images. To enable the unified representation of the 3D scene and semantic field, we embed semantic features into 3D Gaussians and predict them through a dual-branch decoupled decoder. During training, we propose a loss-guided view sampler to sample views from easy to hard, eliminating the need for ground truth depth or masks required by previous methods and stabilizing the training process. The whole model can be trained end-to-end using a photometric loss and a distillation loss that leverages semantic features from a pre-trained 2D semantic model. At the inference stage, our UniForward can reconstruct 3D scenes and the corresponding semantic fields in real time from only sparse-view images. The reconstructed 3D scenes achieve high-quality rendering, and the reconstructed 3D semantic field enables the rendering of view-consistent semantic features from arbitrary views, which can be further decoded into dense segmentation masks in an open-vocabulary manner. Experiments on novel view synthesis and novel view segmentation demonstrate that our method achieves state-of-the-art performances for unifying 3D scene and semantic field reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08071v2">GigaSLAM: Large-Scale Monocular SLAM with Hierarchical Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2025-06-10
    </div>
    <details class="paper-abstract">
      Tracking and mapping in large-scale, unbounded outdoor environments using only monocular RGB input presents substantial challenges for existing SLAM systems. Traditional Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) SLAM methods are typically limited to small, bounded indoor settings. To overcome these challenges, we introduce GigaSLAM, the first RGB NeRF / 3DGS-based SLAM framework for kilometer-scale outdoor environments, as demonstrated on the KITTI, KITTI 360, 4 Seasons and A2D2 datasets. Our approach employs a hierarchical sparse voxel map representation, where Gaussians are decoded by neural networks at multiple levels of detail. This design enables efficient, scalable mapping and high-fidelity viewpoint rendering across expansive, unbounded scenes. For front-end tracking, GigaSLAM utilizes a metric depth model combined with epipolar geometry and PnP algorithms to accurately estimate poses, while incorporating a Bag-of-Words-based loop closure mechanism to maintain robust alignment over long trajectories. Consequently, GigaSLAM delivers high-precision tracking and visually faithful rendering on urban outdoor benchmarks, establishing a robust SLAM solution for large-scale, long-term scenarios, and significantly extending the applicability of Gaussian Splatting SLAM systems to unbounded outdoor environments. GitHub: https://github.com/DengKaiCQ/GigaSLAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08862v1">StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams</a></div>
    <div class="paper-meta">
      📅 2025-06-10
    </div>
    <details class="paper-abstract">
      Real-time reconstruction of dynamic 3D scenes from uncalibrated video streams is crucial for numerous real-world applications. However, existing methods struggle to jointly address three key challenges: 1) processing uncalibrated inputs in real time, 2) accurately modeling dynamic scene evolution, and 3) maintaining long-term stability and computational efficiency. To this end, we introduce StreamSplat, the first fully feed-forward framework that transforms uncalibrated video streams of arbitrary length into dynamic 3D Gaussian Splatting (3DGS) representations in an online manner, capable of recovering scene dynamics from temporally local observations. We propose two key technical innovations: a probabilistic sampling mechanism in the static encoder for 3DGS position prediction, and a bidirectional deformation field in the dynamic decoder that enables robust and efficient dynamic modeling. Extensive experiments on static and dynamic benchmarks demonstrate that StreamSplat consistently outperforms prior works in both reconstruction quality and dynamic scene modeling, while uniquely supporting online reconstruction of arbitrarily long video streams. Code and models are available at https://github.com/nickwzk/StreamSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08710v1">SceneSplat++: A Large Dataset and Comprehensive Benchmark for Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-10
      | 💬 15 pages, codes, data and benchmark will be released
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) serves as a highly performant and efficient encoding of scene geometry, appearance, and semantics. Moreover, grounding language in 3D scenes has proven to be an effective strategy for 3D scene understanding. Current Language Gaussian Splatting line of work fall into three main groups: (i) per-scene optimization-based, (ii) per-scene optimization-free, and (iii) generalizable approach. However, most of them are evaluated only on rendered 2D views of a handful of scenes and viewpoints close to the training views, limiting ability and insight into holistic 3D understanding. To address this gap, we propose the first large-scale benchmark that systematically assesses these three groups of methods directly in 3D space, evaluating on 1060 scenes across three indoor datasets and one outdoor dataset. Benchmark results demonstrate a clear advantage of the generalizable paradigm, particularly in relaxing the scene-specific limitation, enabling fast feed-forward inference on novel scenes, and achieving superior segmentation performance. We further introduce GaussianWorld-49K a carefully curated 3DGS dataset comprising around 49K diverse indoor and outdoor scenes obtained from multiple sources, with which we demonstrate the generalizable approach could harness strong data priors. Our codes, benchmark, and datasets will be made public to accelerate research in generalizable 3DGS scene understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08704v1">TraGraph-GS: Trajectory Graph-based Gaussian Splatting for Arbitrary Large-Scale Scene Rendering</a></div>
    <div class="paper-meta">
      📅 2025-06-10
    </div>
    <details class="paper-abstract">
      High-quality novel view synthesis for large-scale scenes presents a challenging dilemma in 3D computer vision. Existing methods typically partition large scenes into multiple regions, reconstruct a 3D representation using Gaussian splatting for each region, and eventually merge them for novel view rendering. They can accurately render specific scenes, yet they do not generalize effectively for two reasons: (1) rigid spatial partition techniques struggle with arbitrary camera trajectories, and (2) the merging of regions results in Gaussian overlap to distort texture details. To address these challenges, we propose TraGraph-GS, leveraging a trajectory graph to enable high-precision rendering for arbitrarily large-scale scenes. We present a spatial partitioning method for large-scale scenes based on graphs, which incorporates a regularization constraint to enhance the rendering of textures and distant objects, as well as a progressive rendering strategy to mitigate artifacts caused by Gaussian overlap. Experimental results demonstrate its superior performance both on four aerial and four ground datasets and highlight its remarkable efficiency: our method achieves an average improvement of 1.86 dB in PSNR on aerial datasets and 1.62 dB on ground datasets compared to state-of-the-art approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08350v1">Complex-Valued Holographic Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2025-06-10
      | 💬 28 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Modeling the full properties of light, including both amplitude and phase, in 3D representations is crucial for advancing physically plausible rendering, particularly in holographic displays. To support these features, we propose a novel representation that optimizes 3D scenes without relying on intensity-based intermediaries. We reformulate 3D Gaussian splatting with complex-valued Gaussian primitives, expanding support for rendering with light waves. By leveraging RGBD multi-view images, our method directly optimizes complex-valued Gaussians as a 3D holographic scene representation. This eliminates the need for computationally expensive hologram re-optimization. Compared with state-of-the-art methods, our method achieves 30x-10,000x speed improvements while maintaining on-par image quality, representing a first step towards geometrically aligned, physically plausible holographic scene representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07657v1">PIG: Physically-based Multi-Material Interaction with 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has achieved remarkable success in reconstructing both static and dynamic 3D scenes. However, in a scene represented by 3D Gaussian primitives, interactions between objects suffer from inaccurate 3D segmentation, imprecise deformation among different materials, and severe rendering artifacts. To address these challenges, we introduce PIG: Physically-Based Multi-Material Interaction with 3D Gaussians, a novel approach that combines 3D object segmentation with the simulation of interacting objects in high precision. Firstly, our method facilitates fast and accurate mapping from 2D pixels to 3D Gaussians, enabling precise 3D object-level segmentation. Secondly, we assign unique physical properties to correspondingly segmented objects within the scene for multi-material coupled interactions. Finally, we have successfully embedded constraint scales into deformation gradients, specifically clamping the scaling and rotation properties of the Gaussian primitives to eliminate artifacts and achieve geometric fidelity and visual consistency. Experimental results demonstrate that our method not only outperforms the state-of-the-art (SOTA) in terms of visual quality, but also opens up new directions and pipelines for the field of physically realistic scene generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21041v3">CityGo: Lightweight Urban Modeling and Rendering with Proxy Buildings and Residual Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      Accurate and efficient modeling of large-scale urban scenes is critical for applications such as AR navigation, UAV based inspection, and smart city digital twins. While aerial imagery offers broad coverage and complements limitations of ground-based data, reconstructing city-scale environments from such views remains challenging due to occlusions, incomplete geometry, and high memory demands. Recent advances like 3D Gaussian Splatting (3DGS) improve scalability and visual quality but remain limited by dense primitive usage, long training times, and poor suit ability for edge devices. We propose CityGo, a hybrid framework that combines textured proxy geometry with residual and surrounding 3D Gaussians for lightweight, photorealistic rendering of urban scenes from aerial perspectives. Our approach first extracts compact building proxy meshes from MVS point clouds, then uses zero order SH Gaussians to generate occlusion-free textures via image-based rendering and back-projection. To capture high-frequency details, we introduce residual Gaussians placed based on proxy-photo discrepancies and guided by depth priors. Broader urban context is represented by surrounding Gaussians, with importance-aware downsampling applied to non-critical regions to reduce redundancy. A tailored optimization strategy jointly refines proxy textures and Gaussian parameters, enabling real-time rendering of complex urban scenes on mobile GPUs with significantly reduced training and memory requirements. Extensive experiments on real-world aerial datasets demonstrate that our hybrid representation significantly reduces training time, achieving on average 1.4x speedup, while delivering comparable visual fidelity to pure 3D Gaussian Splatting approaches. Furthermore, CityGo enables real-time rendering of large-scale urban scenes on mobile consumer GPUs, with substantially reduced memory usage and energy consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07338v1">Hierarchical Scoring with 3D Gaussian Splatting for Instance Image-Goal Navigation</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      Instance Image-Goal Navigation (IIN) requires autonomous agents to identify and navigate to a target object or location depicted in a reference image captured from any viewpoint. While recent methods leverage powerful novel view synthesis (NVS) techniques, such as three-dimensional Gaussian splatting (3DGS), they typically rely on randomly sampling multiple viewpoints or trajectories to ensure comprehensive coverage of discriminative visual cues. This approach, however, creates significant redundancy through overlapping image samples and lacks principled view selection, substantially increasing both rendering and comparison overhead. In this paper, we introduce a novel IIN framework with a hierarchical scoring paradigm that estimates optimal viewpoints for target matching. Our approach integrates cross-level semantic scoring, utilizing CLIP-derived relevancy fields to identify regions with high semantic similarity to the target object class, with fine-grained local geometric scoring that performs precise pose estimation within promising regions. Extensive evaluations demonstrate that our method achieves state-of-the-art performance on simulated IIN benchmarks and real-world applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06598v3">Stochastic Ray Tracing of Transparent 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 10 pages, 7 figures, 5 tables
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting has been widely adopted as a 3D representation for novel-view synthesis, relighting, and 3D generation tasks. It delivers realistic and detailed results through a collection of explicit 3D Gaussian primitives, each carrying opacity and view-dependent color. However, efficient rendering of many transparent primitives remains a significant challenge. Existing approaches either rasterize the Gaussians with approximate per-view sorting or rely on high-end RTX GPUs. This paper proposes a stochastic ray-tracing method to render 3D clouds of transparent primitives. Instead of processing all ray-Gaussian intersections in sequential order, each ray traverses the acceleration structure only once, randomly accepting and shading a single intersection (or $N$ intersections, using a simple extension). This approach minimizes shading time and avoids primitive sorting along the ray, thereby minimizing register usage and maximizing parallelism even on low-end GPUs. The cost of rays through the Gaussian asset is comparable to that of standard mesh-intersection rays. The shading is unbiased and has low variance, as our stochastic acceptance achieves importance sampling based on accumulated weight. The alignment with Monte Carlo philosophy simplifies implementation and integration into a conventional path-tracing framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09070v1">STREAMINGGS: Voxel-Based Streaming 3D Gaussian Splatting with Memory Optimization and Architectural Support</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has gained popularity for its efficiency and sparse Gaussian-based representation. However, 3DGS struggles to meet the real-time requirement of 90 frames per second (FPS) on resource-constrained mobile devices, achieving only 2 to 9 FPS.Existing accelerators focus on compute efficiency but overlook memory efficiency, leading to redundant DRAM traffic. We introduce STREAMINGGS, a fully streaming 3DGS algorithm-architecture co-design that achieves fine-grained pipelining and reduces DRAM traffic by transforming from a tile-centric rendering to a memory-centric rendering. Results show that our design achieves up to 45.7 $\times$ speedup and 62.9 $\times$ energy savings over mobile Ampere GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07917v1">Speedy Deformable 3D Gaussian Splatting: Fast Rendering and Compression of Dynamic Scenes</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 Project Page: https://speede3dgs.github.io/
    </div>
    <details class="paper-abstract">
      Recent extensions of 3D Gaussian Splatting (3DGS) to dynamic scenes achieve high-quality novel view synthesis by using neural networks to predict the time-varying deformation of each Gaussian. However, performing per-Gaussian neural inference at every frame poses a significant bottleneck, limiting rendering speed and increasing memory and compute requirements. In this paper, we present Speedy Deformable 3D Gaussian Splatting (SpeeDe3DGS), a general pipeline for accelerating the rendering speed of dynamic 3DGS and 4DGS representations by reducing neural inference through two complementary techniques. First, we propose a temporal sensitivity pruning score that identifies and removes Gaussians with low contribution to the dynamic scene reconstruction. We also introduce an annealing smooth pruning mechanism that improves pruning robustness in real-world scenes with imprecise camera poses. Second, we propose GroupFlow, a motion analysis technique that clusters Gaussians by trajectory similarity and predicts a single rigid transformation per group instead of separate deformations for each Gaussian. Together, our techniques accelerate rendering by $10.37\times$, reduce model size by $7.71\times$, and shorten training time by $2.71\times$ on the NeRF-DS dataset. SpeeDe3DGS also improves rendering speed by $4.20\times$ and $58.23\times$ on the D-NeRF and HyperNeRF vrig datasets. Our methods are modular and can be integrated into any deformable 3DGS or 4DGS framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07897v1">GaussianVAE: Adaptive Learning Dynamics of 3D Gaussians for High-Fidelity Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      We present a novel approach for enhancing the resolution and geometric fidelity of 3D Gaussian Splatting (3DGS) beyond native training resolution. Current 3DGS methods are fundamentally limited by their input resolution, producing reconstructions that cannot extrapolate finer details than are present in the training views. Our work breaks this limitation through a lightweight generative model that predicts and refines additional 3D Gaussians where needed most. The key innovation is our Hessian-assisted sampling strategy, which intelligently identifies regions that are likely to benefit from densification, ensuring computational efficiency. Unlike computationally intensive GANs or diffusion approaches, our method operates in real-time (0.015s per inference on a single consumer-grade GPU), making it practical for interactive applications. Comprehensive experiments demonstrate significant improvements in both geometric accuracy and rendering quality compared to state-of-the-art methods, establishing a new paradigm for resolution-free 3D scene enhancement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07826v1">R3D2: Realistic 3D Asset Insertion via Diffusion for Autonomous Driving Simulation</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      Validating autonomous driving (AD) systems requires diverse and safety-critical testing, making photorealistic virtual environments essential. Traditional simulation platforms, while controllable, are resource-intensive to scale and often suffer from a domain gap with real-world data. In contrast, neural reconstruction methods like 3D Gaussian Splatting (3DGS) offer a scalable solution for creating photorealistic digital twins of real-world driving scenes. However, they struggle with dynamic object manipulation and reusability as their per-scene optimization-based methodology tends to result in incomplete object models with integrated illumination effects. This paper introduces R3D2, a lightweight, one-step diffusion model designed to overcome these limitations and enable realistic insertion of complete 3D assets into existing scenes by generating plausible rendering effects-such as shadows and consistent lighting-in real time. This is achieved by training R3D2 on a novel dataset: 3DGS object assets are generated from in-the-wild AD data using an image-conditioned 3D generative model, and then synthetically placed into neural rendering-based virtual environments, allowing R3D2 to learn realistic integration. Quantitative and qualitative evaluations demonstrate that R3D2 significantly enhances the realism of inserted assets, enabling use-cases like text-to-3D asset insertion and cross-scene/dataset object transfer, allowing for true scalability in AD validation. To promote further research in scalable and realistic AD simulation, we will release our dataset and code, see https://research.zenseact.com/publications/R3D2/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07697v1">OpenSplat3D: Open-Vocabulary 3D Instance Segmentation using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful representation for neural scene reconstruction, offering high-quality novel view synthesis while maintaining computational efficiency. In this paper, we extend the capabilities of 3DGS beyond pure scene representation by introducing an approach for open-vocabulary 3D instance segmentation without requiring manual labeling, termed OpenSplat3D. Our method leverages feature-splatting techniques to associate semantic information with individual Gaussians, enabling fine-grained scene understanding. We incorporate Segment Anything Model instance masks with a contrastive loss formulation as guidance for the instance features to achieve accurate instance-level segmentation. Furthermore, we utilize language embeddings of a vision-language model, allowing for flexible, text-driven instance identification. This combination enables our system to identify and segment arbitrary objects in 3D scenes based on natural language descriptions. We show results on LERF-mask and LERF-OVS as well as the full ScanNet++ validation set, demonstrating the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07670v1">ProSplat: Improved Feed-Forward 3D Gaussian Splatting for Wide-Baseline Sparse Views</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) has recently demonstrated promising results for novel view synthesis (NVS) from sparse input views, particularly under narrow-baseline conditions. However, its performance significantly degrades in wide-baseline scenarios due to limited texture details and geometric inconsistencies across views. To address these challenges, in this paper, we propose ProSplat, a two-stage feed-forward framework designed for high-fidelity rendering under wide-baseline conditions. The first stage involves generating 3D Gaussian primitives via a 3DGS generator. In the second stage, rendered views from these primitives are enhanced through an improvement model. Specifically, this improvement model is based on a one-step diffusion model, further optimized by our proposed Maximum Overlap Reference view Injection (MORI) and Distance-Weighted Epipolar Attention (DWEA). MORI supplements missing texture and color by strategically selecting a reference view with maximum viewpoint overlap, while DWEA enforces geometric consistency using epipolar constraints. Additionally, we introduce a divide-and-conquer training strategy that aligns data distributions between the two stages through joint optimization. We evaluate ProSplat on the RealEstate10K and DL3DV-10K datasets under wide-baseline settings. Experimental results demonstrate that ProSplat achieves an average improvement of 1 dB in PSNR compared to recent SOTA methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04120v2">Splatting Physical Scenes: End-to-End Real-to-Sim from Imperfect Robot Data</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 Updated version correcting inadvertent omission in author list
    </div>
    <details class="paper-abstract">
      Creating accurate, physical simulations directly from real-world robot motion holds great value for safe, scalable, and affordable robot learning, yet remains exceptionally challenging. Real robot data suffers from occlusions, noisy camera poses, dynamic scene elements, which hinder the creation of geometrically accurate and photorealistic digital twins of unseen objects. We introduce a novel real-to-sim framework tackling all these challenges at once. Our key insight is a hybrid scene representation merging the photorealistic rendering of 3D Gaussian Splatting with explicit object meshes suitable for physics simulation within a single representation. We propose an end-to-end optimization pipeline that leverages differentiable rendering and differentiable physics within MuJoCo to jointly refine all scene components - from object geometry and appearance to robot poses and physical parameters - directly from raw and imprecise robot trajectories. This unified optimization allows us to simultaneously achieve high-fidelity object mesh reconstruction, generate photorealistic novel views, and perform annotation-free robot pose calibration. We demonstrate the effectiveness of our approach both in simulation and on challenging real-world sequences using an ALOHA 2 bi-manual manipulator, enabling more practical and robust real-to-simulation pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18122v2">RT-GuIDE: Real-Time Gaussian splatting for Information-Driven Exploration</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      We propose a framework for active mapping and exploration that leverages Gaussian splatting for constructing dense maps. Further, we develop a GPU-accelerated motion planning algorithm that can exploit the Gaussian map for real-time navigation. The Gaussian map constructed onboard the robot is optimized for both photometric and geometric quality while enabling real-time situational awareness for autonomy. We show through simulation experiments that our method yields comparable Peak Signal-to-Noise Ratio (PSNR) and similar reconstruction error to state-of-the-art approaches, while being orders of magnitude faster to compute. In real-world experiments, our algorithm achieves better map quality (at least 0.8dB higher PSNR and more than 16% higher geometric reconstruction accuracy) than maps constructed by a state-of-the-art method, enabling semantic segmentation using off-the-shelf open-set models. Experiment videos and more details can be found on our project page: https://tyuezhan.github.io/RT_GuIDE/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07069v1">Accelerating 3D Gaussian Splatting with Neural Sorting and Axis-Oriented Rasterization</a></div>
    <div class="paper-meta">
      📅 2025-06-08
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently gained significant attention for high-quality and efficient view synthesis, making it widely adopted in fields such as AR/VR, robotics, and autonomous driving. Despite its impressive algorithmic performance, real-time rendering on resource-constrained devices remains a major challenge due to tight power and area budgets. This paper presents an architecture-algorithm co-design to address these inefficiencies. First, we reveal substantial redundancy caused by repeated computation of common terms/expressions during the conventional rasterization. To resolve this, we propose axis-oriented rasterization, which pre-computes and reuses shared terms along both the X and Y axes through a dedicated hardware design, effectively reducing multiply-and-add (MAC) operations by up to 63%. Second, by identifying the resource and performance inefficiency of the sorting process, we introduce a novel neural sorting approach that predicts order-independent blending weights using an efficient neural network, eliminating the need for costly hardware sorters. A dedicated training framework is also proposed to improve its algorithmic stability. Third, to uniformly support rasterization and neural network inference, we design an efficient reconfigurable processing array that maximizes hardware utilization and throughput. Furthermore, we introduce a $\pi$-trajectory tile schedule, inspired by Morton encoding and Hilbert curve, to optimize Gaussian reuse and reduce memory access overhead. Comprehensive experiments demonstrate that the proposed design preserves rendering quality while achieving a speedup of $23.4\sim27.8\times$ and energy savings of $28.8\sim51.4\times$ compared to edge GPUs for real-world scenes. We plan to open-source our design to foster further development in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06988v1">Hybrid Mesh-Gaussian Representation for Efficient Indoor Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has demonstrated exceptional performance in image-based 3D reconstruction and real-time rendering. However, regions with complex textures require numerous Gaussians to capture significant color variations accurately, leading to inefficiencies in rendering speed. To address this challenge, we introduce a hybrid representation for indoor scenes that combines 3DGS with textured meshes. Our approach uses textured meshes to handle texture-rich flat areas, while retaining Gaussians to model intricate geometries. The proposed method begins by pruning and refining the extracted mesh to eliminate geometrically complex regions. We then employ a joint optimization for 3DGS and mesh, incorporating a warm-up strategy and transmittance-aware supervision to balance their contributions seamlessly.Extensive experiments demonstrate that the hybrid representation maintains comparable rendering quality and achieves superior frames per second FPS with fewer Gaussian primitives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06909v1">Gaussian Mapping for Evolving Scenes</a></div>
    <div class="paper-meta">
      📅 2025-06-07
    </div>
    <details class="paper-abstract">
      Mapping systems with novel view synthesis (NVS) capabilities are widely used in computer vision, with augmented reality, robotics, and autonomous driving applications. Most notably, 3D Gaussian Splatting-based systems show high NVS performance; however, many current approaches are limited to static scenes. While recent works have started addressing short-term dynamics (motion within the view of the camera), long-term dynamics (the scene evolving through changes out of view) remain less explored. To overcome this limitation, we introduce a dynamic scene adaptation mechanism that continuously updates the 3D representation to reflect the latest changes. In addition, since maintaining geometric and semantic consistency remains challenging due to stale observations disrupting the reconstruction process, we propose a novel keyframe management mechanism that discards outdated observations while preserving as much information as possible. We evaluate Gaussian Mapping for Evolving Scenes (GaME) on both synthetic and real-world datasets and find it to be more accurate than the state of the art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06890v1">SPC to 3D: Novel View Synthesis from Binary SPC via I2I translation</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 Accepted for publication at ICIP 2025
    </div>
    <details class="paper-abstract">
      Single Photon Avalanche Diodes (SPADs) represent a cutting-edge imaging technology, capable of detecting individual photons with remarkable timing precision. Building on this sensitivity, Single Photon Cameras (SPCs) enable image capture at exceptionally high speeds under both low and high illumination. Enabling 3D reconstruction and radiance field recovery from such SPC data holds significant promise. However, the binary nature of SPC images leads to severe information loss, particularly in texture and color, making traditional 3D synthesis techniques ineffective. To address this challenge, we propose a modular two-stage framework that converts binary SPC images into high-quality colorized novel views. The first stage performs image-to-image (I2I) translation using generative models such as Pix2PixHD, converting binary SPC inputs into plausible RGB representations. The second stage employs 3D scene reconstruction techniques like Neural Radiance Fields (NeRF) or Gaussian Splatting (3DGS) to generate novel views. We validate our two-stage pipeline (Pix2PixHD + Nerf/3DGS) through extensive qualitative and quantitative experiments, demonstrating significant improvements in perceptual quality and geometric consistency over the alternative baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06846v1">Multi-StyleGS: Stylizing Gaussian Splatting with Multiple Styles</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 AAAI 2025
    </div>
    <details class="paper-abstract">
      In recent years, there has been a growing demand to stylize a given 3D scene to align with the artistic style of reference images for creative purposes. While 3D Gaussian Splatting(GS) has emerged as a promising and efficient method for realistic 3D scene modeling, there remains a challenge in adapting it to stylize 3D GS to match with multiple styles through automatic local style transfer or manual designation, while maintaining memory efficiency for stylization training. In this paper, we introduce a novel 3D GS stylization solution termed Multi-StyleGS to tackle these challenges. In particular, we employ a bipartite matching mechanism to au tomatically identify correspondences between the style images and the local regions of the rendered images. To facilitate local style transfer, we introduce a novel semantic style loss function that employs a segmentation network to apply distinct styles to various objects of the scene and propose a local-global feature matching to enhance the multi-view consistency. Furthermore, this technique can achieve memory efficient training, more texture details and better color match. To better assign a robust semantic label to each Gaussian, we propose several techniques to regularize the segmentation network. As demonstrated by our comprehensive experiments, our approach outperforms existing ones in producing plausible stylization results and offering flexible editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06822v1">Hi-LSplat: Hierarchical 3D Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-07
    </div>
    <details class="paper-abstract">
      Modeling 3D language fields with Gaussian Splatting for open-ended language queries has recently garnered increasing attention. However, recent 3DGS-based models leverage view-dependent 2D foundation models to refine 3D semantics but lack a unified 3D representation, leading to view inconsistencies. Additionally, inherent open-vocabulary challenges cause inconsistencies in object and relational descriptions, impeding hierarchical semantic understanding. In this paper, we propose Hi-LSplat, a view-consistent Hierarchical Language Gaussian Splatting work for 3D open-vocabulary querying. To achieve view-consistent 3D hierarchical semantics, we first lift 2D features to 3D features by constructing a 3D hierarchical semantic tree with layered instance clustering, which addresses the view inconsistency issue caused by 2D semantic features. Besides, we introduce instance-wise and part-wise contrastive losses to capture all-sided hierarchical semantic representations. Notably, we construct two hierarchical semantic datasets to better assess the model's ability to distinguish different semantic levels. Extensive experiments highlight our method's superiority in 3D open-vocabulary segmentation and localization. Its strong performance on hierarchical semantic datasets underscores its ability to capture complex hierarchical semantics within 3D scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04457v2">Monocular Dynamic Gaussian Splatting: Fast, Brittle, and Scene Complexity Rules</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 TMLR 2025. Project Website: https://brownvc.github.io/MonoDyGauBench.github.io/
    </div>
    <details class="paper-abstract">
      Gaussian splatting methods are emerging as a popular approach for converting multi-view image data into scene representations that allow view synthesis. In particular, there is interest in enabling view synthesis for dynamic scenes using only monocular input data -- an ill-posed and challenging problem. The fast pace of work in this area has produced multiple simultaneous papers that claim to work best, which cannot all be true. In this work, we organize, benchmark, and analyze many Gaussian-splatting-based methods, providing apples-to-apples comparisons that prior works have lacked. We use multiple existing datasets and a new instructive synthetic dataset designed to isolate factors that affect reconstruction quality. We systematically categorize Gaussian splatting methods into specific motion representation types and quantify how their differences impact performance. Empirically, we find that their rank order is well-defined in synthetic data, but the complexity of real-world data currently overwhelms the differences. Furthermore, the fast rendering speed of all Gaussian-based methods comes at the cost of brittleness in optimization. We summarize our experiments into a list of findings that can help to further progress in this lively problem setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17351v3">DOF-GS:Adjustable Depth-of-Field 3D Gaussian Splatting for Post-Capture Refocusing, Defocus Rendering and Blur Removal</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 Project page: https://dof-gs.github.io/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) techniques have recently enabled high-quality 3D scene reconstruction and real-time novel view synthesis. These approaches, however, are limited by the pinhole camera model and lack effective modeling of defocus effects. Departing from this, we introduce DOF-GS--a new 3DGS-based framework with a finite-aperture camera model and explicit, differentiable defocus rendering, enabling it to function as a post-capture control tool. By training with multi-view images with moderate defocus blur, DOF-GS learns inherent camera characteristics and reconstructs sharp details of the underlying scene, particularly, enabling rendering of varying DOF effects through on-demand aperture and focal distance control, post-capture and optimization. Additionally, our framework extracts circle-of-confusion cues during optimization to identify in-focus regions in input views, enhancing the reconstructed 3D scene details. Experimental results demonstrate that DOF-GS supports post-capture refocusing, adjustable defocus and high-quality all-in-focus rendering, from multi-view images with uncalibrated defocus blur.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06645v1">Parametric Gaussian Human Model: Generalizable Prior for Efficient and Realistic Human Avatar Modeling</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 Project Page: https://pengc02.github.io/pghm/
    </div>
    <details class="paper-abstract">
      Photorealistic and animatable human avatars are a key enabler for virtual/augmented reality, telepresence, and digital entertainment. While recent advances in 3D Gaussian Splatting (3DGS) have greatly improved rendering quality and efficiency, existing methods still face fundamental challenges, including time-consuming per-subject optimization and poor generalization under sparse monocular inputs. In this work, we present the Parametric Gaussian Human Model (PGHM), a generalizable and efficient framework that integrates human priors into 3DGS for fast and high-fidelity avatar reconstruction from monocular videos. PGHM introduces two core components: (1) a UV-aligned latent identity map that compactly encodes subject-specific geometry and appearance into a learnable feature tensor; and (2) a disentangled Multi-Head U-Net that predicts Gaussian attributes by decomposing static, pose-dependent, and view-dependent components via conditioned decoders. This design enables robust rendering quality under challenging poses and viewpoints, while allowing efficient subject adaptation without requiring multi-view capture or long optimization time. Experiments show that PGHM is significantly more efficient than optimization-from-scratch methods, requiring only approximately 20 minutes per subject to produce avatars with comparable visual quality, thereby demonstrating its practical applicability for real-world monocular avatar creation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08485v2">TT-Occ: Test-Time Compute for Self-Supervised Occupancy via Spatio-Temporal Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Self-supervised 3D occupancy prediction offers a promising solution for understanding complex driving scenes without requiring costly 3D annotations. However, training dense occupancy decoders to capture fine-grained geometry and semantics can demand hundreds of GPU hours, and once trained, such models struggle to adapt to varying voxel resolutions or novel object categories without extensive retraining. To overcome these limitations, we propose a practical and flexible test-time occupancy prediction framework termed TT-Occ. Our method incrementally constructs, optimizes and voxelizes time-aware 3D Gaussians from raw sensor streams by integrating vision foundation models (VLMs) at runtime. The flexible nature of 3D Gaussians allows voxelization at arbitrary user-specified resolutions, while the generalization ability of VLMs enables accurate perception and open-vocabulary recognition, without any network training or fine-tuning. Specifically, TT-Occ operates in a lift-track-voxelize symphony: We first lift the geometry and semantics of surrounding-view extracted from VLMs to instantiate Gaussians at 3D space; Next, we track dynamic Gaussians while accumulating static ones to complete the scene and enforce temporal consistency; Finally, we voxelize the optimized Gaussians to generate occupancy prediction. Optionally, inherent noise in VLM predictions and tracking is mitigated by periodically smoothing neighboring Gaussians during optimization. To validate the generality and effectiveness of our framework, we offer two variants: one LiDAR-based and one vision-centric, and conduct extensive experiments on Occ3D and nuCraft benchmarks with varying voxel resolutions. Code will be available at https://github.com/Xian-Bei/TT-Occ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05682v1">Lumina: Real-Time Mobile Neural Rendering by Exploiting Computational Redundancy</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has vastly advanced the pace of neural rendering, but it remains computationally demanding on today's mobile SoCs. To address this challenge, we propose Lumina, a hardware-algorithm co-designed system, which integrates two principal optimizations: a novel algorithm, S^2, and a radiance caching mechanism, RC, to improve the efficiency of neural rendering. S2 algorithm exploits temporal coherence in rendering to reduce the computational overhead, while RC leverages the color integration process of 3DGS to decrease the frequency of intensive rasterization computations. Coupled with these techniques, we propose an accelerator architecture, LuminCore, to further accelerate cache lookup and address the fundamental inefficiencies in Rasterization. We show that Lumina achieves 4.5x speedup and 5.3x energy reduction against a mobile Volta GPU, with a marginal quality loss (< 0.2 dB peak signal-to-noise ratio reduction) across synthetic and real-world datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06517v1">GS4: Generalizable Sparse Splatting Semantic SLAM</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Traditional SLAM algorithms are excellent at camera tracking but might generate lower resolution and incomplete 3D maps. Recently, Gaussian Splatting (GS) approaches have emerged as an option for SLAM with accurate, dense 3D map building. However, existing GS-based SLAM methods rely on per-scene optimization which is time-consuming and does not generalize to diverse scenes well. In this work, we introduce the first generalizable GS-based semantic SLAM algorithm that incrementally builds and updates a 3D scene representation from an RGB-D video stream using a learned generalizable network. Our approach starts from an RGB-D image recognition backbone to predict the Gaussian parameters from every downsampled and backprojected image location. Additionally, we seamlessly integrate 3D semantic segmentation into our GS framework, bridging 3D mapping and recognition through a shared backbone. To correct localization drifting and floaters, we propose to optimize the GS for only 1 iteration following global localization. We demonstrate state-of-the-art semantic SLAM performance on the real-world benchmark ScanNet with an order of magnitude fewer Gaussians compared to other recent GS-based methods, and showcase our model's generalization capability through zero-shot transfer to the NYUv2 and TUM RGB-D datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06462v1">Splat and Replace: 3D Reconstruction with Repetitive Elements</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 SIGGRAPH Conference Papers 2025. Project site: https://repo-sam.inria.fr/nerphys/splat-and-replace/
    </div>
    <details class="paper-abstract">
      We leverage repetitive elements in 3D scenes to improve novel view synthesis. Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have greatly improved novel view synthesis but renderings of unseen and occluded parts remain low-quality if the training views are not exhaustive enough. Our key observation is that our environment is often full of repetitive elements. We propose to leverage those repetitions to improve the reconstruction of low-quality parts of the scene due to poor coverage and occlusions. We propose a method that segments each repeated instance in a 3DGS reconstruction, registers them together, and allows information to be shared among instances. Our method improves the geometry while also accounting for appearance variations across instances. We demonstrate our method on a variety of synthetic and real scenes with typical repetitive elements, leading to a substantial improvement in the quality of novel view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09063v1">Reconstructing Heterogeneous Biomolecules via Hierarchical Gaussian Mixtures and Part Discovery</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 21 pages, 14 figures, Project Webpage: https://shekshaa.github.io/CryoSPIRE
    </div>
    <details class="paper-abstract">
      Cryo-EM is a transformational paradigm in molecular biology where computational methods are used to infer 3D molecular structure at atomic resolution from extremely noisy 2D electron microscope images. At the forefront of research is how to model the structure when the imaged particles exhibit non-rigid conformational flexibility and compositional variation where parts are sometimes missing. We introduce a novel 3D reconstruction framework with a hierarchical Gaussian mixture model, inspired in part by Gaussian Splatting for 4D scene reconstruction. In particular, the structure of the model is grounded in an initial process that infers a part-based segmentation of the particle, providing essential inductive bias in order to handle both conformational and compositional variability. The framework, called CryoSPIRE, is shown to reveal biologically meaningful structures on complex experimental datasets, and establishes a new state-of-the-art on CryoBench, a benchmark for cryo-EM heterogeneity methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05965v1">Dy3DGS-SLAM: Monocular 3D Gaussian Splatting SLAM for Dynamic Environments</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Current Simultaneous Localization and Mapping (SLAM) methods based on Neural Radiance Fields (NeRF) or 3D Gaussian Splatting excel in reconstructing static 3D scenes but struggle with tracking and reconstruction in dynamic environments, such as real-world scenes with moving elements. Existing NeRF-based SLAM approaches addressing dynamic challenges typically rely on RGB-D inputs, with few methods accommodating pure RGB input. To overcome these limitations, we propose Dy3DGS-SLAM, the first 3D Gaussian Splatting (3DGS) SLAM method for dynamic scenes using monocular RGB input. To address dynamic interference, we fuse optical flow masks and depth masks through a probabilistic model to obtain a fused dynamic mask. With only a single network iteration, this can constrain tracking scales and refine rendered geometry. Based on the fused dynamic mask, we designed a novel motion loss to constrain the pose estimation network for tracking. In mapping, we use the rendering loss of dynamic pixels, color, and depth to eliminate transient interference and occlusion caused by dynamic objects. Experimental results demonstrate that Dy3DGS-SLAM achieves state-of-the-art tracking and rendering in dynamic environments, outperforming or matching existing RGB-D methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05935v1">SurGSplat: Progressive Geometry-Constrained Gaussian Splatting for Surgical Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Intraoperative navigation relies heavily on precise 3D reconstruction to ensure accuracy and safety during surgical procedures. However, endoscopic scenarios present unique challenges, including sparse features and inconsistent lighting, which render many existing Structure-from-Motion (SfM)-based methods inadequate and prone to reconstruction failure. To mitigate these constraints, we propose SurGSplat, a novel paradigm designed to progressively refine 3D Gaussian Splatting (3DGS) through the integration of geometric constraints. By enabling the detailed reconstruction of vascular structures and other critical features, SurGSplat provides surgeons with enhanced visual clarity, facilitating precise intraoperative decision-making. Experimental evaluations demonstrate that SurGSplat achieves superior performance in both novel view synthesis (NVS) and pose estimation accuracy, establishing it as a high-fidelity and efficient solution for surgical scene reconstruction. More information and results can be found on the page https://surgsplat.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05280v2">Unifying Appearance Codes and Bilateral Grids for Driving Scene Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 Project page: https://bigcileng.github.io/bilateral-driving ; Code: https://github.com/BigCiLeng/bilateral-driving
    </div>
    <details class="paper-abstract">
      Neural rendering techniques, including NeRF and Gaussian Splatting (GS), rely on photometric consistency to produce high-quality reconstructions. However, in real-world scenarios, it is challenging to guarantee perfect photometric consistency in acquired images. Appearance codes have been widely used to address this issue, but their modeling capability is limited, as a single code is applied to the entire image. Recently, the bilateral grid was introduced to perform pixel-wise color mapping, but it is difficult to optimize and constrain effectively. In this paper, we propose a novel multi-scale bilateral grid that unifies appearance codes and bilateral grids. We demonstrate that this approach significantly improves geometric accuracy in dynamic, decoupled autonomous driving scene reconstruction, outperforming both appearance codes and bilateral grids. This is crucial for autonomous driving, where accurate geometry is important for obstacle avoidance and control. Our method shows strong results across four datasets: Waymo, NuScenes, Argoverse, and PandaSet. We further demonstrate that the improvement in geometry is driven by the multi-scale bilateral grid, which effectively reduces floaters caused by photometric inconsistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05327v1">Revisiting Depth Representations for Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Project page: https://aim-uofa.github.io/PMLoss
    </div>
    <details class="paper-abstract">
      Depth maps are widely used in feed-forward 3D Gaussian Splatting (3DGS) pipelines by unprojecting them into 3D point clouds for novel view synthesis. This approach offers advantages such as efficient training, the use of known camera poses, and accurate geometry estimation. However, depth discontinuities at object boundaries often lead to fragmented or sparse point clouds, degrading rendering quality -- a well-known limitation of depth-based representations. To tackle this issue, we introduce PM-Loss, a novel regularization loss based on a pointmap predicted by a pre-trained transformer. Although the pointmap itself may be less accurate than the depth map, it effectively enforces geometric smoothness, especially around object boundaries. With the improved depth map, our method significantly improves the feed-forward 3DGS across various architectures and scenes, delivering consistently better rendering results. Our project page: https://aim-uofa.github.io/PMLoss
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05280v1">Unifying Appearance Codes and Bilateral Grids for Driving Scene Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Project page: https://bigcileng.github.io/bilateral-driving; Code: https://github.com/BigCiLeng/bilateral-driving
    </div>
    <details class="paper-abstract">
      Neural rendering techniques, including NeRF and Gaussian Splatting (GS), rely on photometric consistency to produce high-quality reconstructions. However, in real-world scenarios, it is challenging to guarantee perfect photometric consistency in acquired images. Appearance codes have been widely used to address this issue, but their modeling capability is limited, as a single code is applied to the entire image. Recently, the bilateral grid was introduced to perform pixel-wise color mapping, but it is difficult to optimize and constrain effectively. In this paper, we propose a novel multi-scale bilateral grid that unifies appearance codes and bilateral grids. We demonstrate that this approach significantly improves geometric accuracy in dynamic, decoupled autonomous driving scene reconstruction, outperforming both appearance codes and bilateral grids. This is crucial for autonomous driving, where accurate geometry is important for obstacle avoidance and control. Our method shows strong results across four datasets: Waymo, NuScenes, Argoverse, and PandaSet. We further demonstrate that the improvement in geometry is driven by the multi-scale bilateral grid, which effectively reduces floaters caused by photometric inconsistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05092v1">Synthetic Dataset Generation for Autonomous Mobile Robots Using 3D Gaussian Splatting for Vision Training</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Annotated datasets are critical for training neural networks for object detection, yet their manual creation is time- and labour-intensive, subjective to human error, and often limited in diversity. This challenge is particularly pronounced in the domain of robotics, where diverse and dynamic scenarios further complicate the creation of representative datasets. To address this, we propose a novel method for automatically generating annotated synthetic data in Unreal Engine. Our approach leverages photorealistic 3D Gaussian splats for rapid synthetic data generation. We demonstrate that synthetic datasets can achieve performance comparable to that of real-world datasets while significantly reducing the time required to generate and annotate data. Additionally, combining real-world and synthetic data significantly increases object detection performance by leveraging the quality of real-world images with the easier scalability of synthetic data. To our knowledge, this is the first application of synthetic data for training object detection algorithms in the highly dynamic and varied environment of robot soccer. Validation experiments reveal that a detector trained on synthetic images performs on par with one trained on manually annotated real-world images when tested on robot soccer match scenarios. Our method offers a scalable and comprehensive alternative to traditional dataset creation, eliminating the labour-intensive error-prone manual annotation process. By generating datasets in a simulator where all elements are intrinsically known, we ensure accurate annotations while significantly reducing manual effort, which makes it particularly valuable for robotics applications requiring diverse and scalable training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05011v1">UAV4D: Dynamic Neural Rendering of Human-Centric UAV Imagery using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Despite significant advancements in dynamic neural rendering, existing methods fail to address the unique challenges posed by UAV-captured scenarios, particularly those involving monocular camera setups, top-down perspective, and multiple small, moving humans, which are not adequately represented in existing datasets. In this work, we introduce UAV4D, a framework for enabling photorealistic rendering for dynamic real-world scenes captured by UAVs. Specifically, we address the challenge of reconstructing dynamic scenes with multiple moving pedestrians from monocular video data without the need for additional sensors. We use a combination of a 3D foundation model and a human mesh reconstruction model to reconstruct both the scene background and humans. We propose a novel approach to resolve the scene scale ambiguity and place both humans and the scene in world coordinates by identifying human-scene contact points. Additionally, we exploit the SMPL model and background mesh to initialize Gaussian splats, enabling holistic scene rendering. We evaluated our method on three complex UAV-captured datasets: VisDrone, Manipal-UAV, and Okutama-Action, each with distinct characteristics and 10~50 humans. Our results demonstrate the benefits of our approach over existing methods in novel view synthesis, achieving a 1.5 dB PSNR improvement and superior visual sharpness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05009v1">Point Cloud Segmentation of Agricultural Vehicles using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Training neural networks for tasks such as 3D point cloud semantic segmentation demands extensive datasets, yet obtaining and annotating real-world point clouds is costly and labor-intensive. This work aims to introduce a novel pipeline for generating realistic synthetic data, by leveraging 3D Gaussian Splatting (3DGS) and Gaussian Opacity Fields (GOF) to generate 3D assets of multiple different agricultural vehicles instead of using generic models. These assets are placed in a simulated environment, where the point clouds are generated using a simulated LiDAR. This is a flexible approach that allows changing the LiDAR specifications without incurring additional costs. We evaluated the impact of synthetic data on segmentation models such as PointNet++, Point Transformer V3, and OACNN, by training and validating the models only on synthetic data. Remarkably, the PTv3 model had an mIoU of 91.35\%, a noteworthy result given that the model had neither been trained nor validated on any real data. Further studies even suggested that in certain scenarios the models trained only on synthetically generated data performed better than models trained on real-world data. Finally, experiments demonstrated that the models can generalize across semantic classes, enabling accurate predictions on mesh models they were never trained on.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04908v1">Generating Synthetic Stereo Datasets using 3D Gaussian Splatting and Expert Knowledge Transfer</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      In this paper, we introduce a 3D Gaussian Splatting (3DGS)-based pipeline for stereo dataset generation, offering an efficient alternative to Neural Radiance Fields (NeRF)-based methods. To obtain useful geometry estimates, we explore utilizing the reconstructed geometry from the explicit 3D representations as well as depth estimates from the FoundationStereo model in an expert knowledge transfer setup. We find that when fine-tuning stereo models on 3DGS-generated datasets, we demonstrate competitive performance in zero-shot generalization benchmarks. When using the reconstructed geometry directly, we observe that it is often noisy and contains artifacts, which propagate noise to the trained model. In contrast, we find that the disparity estimates from FoundationStereo are cleaner and consequently result in a better performance on the zero-shot generalization benchmarks. Our method highlights the potential for low-cost, high-fidelity dataset creation and fast fine-tuning for deep stereo models. Moreover, we also reveal that while the latest Gaussian Splatting based methods have achieved superior performance on established benchmarks, their robustness falls short in challenging in-the-wild settings warranting further exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04789v1">Object-X: Learning to Reconstruct Multi-Modal 3D Object Representations</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Learning effective multi-modal 3D representations of objects is essential for numerous applications, such as augmented reality and robotics. Existing methods often rely on task-specific embeddings that are tailored either for semantic understanding or geometric reconstruction. As a result, these embeddings typically cannot be decoded into explicit geometry and simultaneously reused across tasks. In this paper, we propose Object-X, a versatile multi-modal object representation framework capable of encoding rich object embeddings (e.g. images, point cloud, text) and decoding them back into detailed geometric and visual reconstructions. Object-X operates by geometrically grounding the captured modalities in a 3D voxel grid and learning an unstructured embedding fusing the information from the voxels with the object attributes. The learned embedding enables 3D Gaussian Splatting-based object reconstruction, while also supporting a range of downstream tasks, including scene alignment, single-image 3D object reconstruction, and localization. Evaluations on two challenging real-world datasets demonstrate that Object-X produces high-fidelity novel-view synthesis comparable to standard 3D Gaussian Splatting, while significantly improving geometric accuracy. Moreover, Object-X achieves competitive performance with specialized methods in scene alignment and localization. Critically, our object-centric descriptors require 3-4 orders of magnitude less storage compared to traditional image- or point cloud-based approaches, establishing Object-X as a scalable and highly practical solution for multi-modal 3D scene representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00625v3">Gaussian Building Mesh (GBM): Extract a Building's 3D Mesh with Google Earth and Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Recently released open-source pre-trained foundational image segmentation and object detection models (SAM2+GroundingDINO) allow for geometrically consistent segmentation of objects of interest in multi-view 2D images. Users can use text-based or click-based prompts to segment objects of interest without requiring labeled training datasets. Gaussian Splatting allows for the learning of the 3D representation of a scene's geometry and radiance based on 2D images. Combining Google Earth Studio, SAM2+GroundingDINO, 2D Gaussian Splatting, and our improvements in mask refinement based on morphological operations and contour simplification, we created a pipeline to extract the 3D mesh of any building based on its name, address, or geographic coordinates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05563v1">VoxelSplat: Dynamic Gaussian Splatting as an Effective Loss for Occupancy and Flow Prediction</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Accepted by CVPR 2025 Project Page: https://zzy816.github.io/VoxelSplat-Demo/
    </div>
    <details class="paper-abstract">
      Recent advancements in camera-based occupancy prediction have focused on the simultaneous prediction of 3D semantics and scene flow, a task that presents significant challenges due to specific difficulties, e.g., occlusions and unbalanced dynamic environments. In this paper, we analyze these challenges and their underlying causes. To address them, we propose a novel regularization framework called VoxelSplat. This framework leverages recent developments in 3D Gaussian Splatting to enhance model performance in two key ways: (i) Enhanced Semantics Supervision through 2D Projection: During training, our method decodes sparse semantic 3D Gaussians from 3D representations and projects them onto the 2D camera view. This provides additional supervision signals in the camera-visible space, allowing 2D labels to improve the learning of 3D semantics. (ii) Scene Flow Learning: Our framework uses the predicted scene flow to model the motion of Gaussians, and is thus able to learn the scene flow of moving objects in a self-supervised manner using the labels of adjacent frames. Our method can be seamlessly integrated into various existing occupancy models, enhancing performance without increasing inference time. Extensive experiments on benchmark datasets demonstrate the effectiveness of VoxelSplat in improving the accuracy of both semantic occupancy and scene flow estimation. The project page and codes are available at https://zzy816.github.io/VoxelSplat-Demo/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05558v1">On-the-fly Reconstruction for Large-Scale Novel View Synthesis from Unposed Images</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Radiance field methods such as 3D Gaussian Splatting (3DGS) allow easy reconstruction from photos, enabling free-viewpoint navigation. Nonetheless, pose estimation using Structure from Motion and 3DGS optimization can still each take between minutes and hours of computation after capture is complete. SLAM methods combined with 3DGS are fast but struggle with wide camera baselines and large scenes. We present an on-the-fly method to produce camera poses and a trained 3DGS immediately after capture. Our method can handle dense and wide-baseline captures of ordered photo sequences and large-scale scenes. To do this, we first introduce fast initial pose estimation, exploiting learned features and a GPU-friendly mini bundle adjustment. We then introduce direct sampling of Gaussian primitive positions and shapes, incrementally spawning primitives where required, significantly accelerating training. These two efficient steps allow fast and robust joint optimization of poses and Gaussian primitives. Our incremental approach handles large-scale scenes by introducing scalable radiance field construction, progressively clustering 3DGS primitives, storing them in anchors, and offloading them from the GPU. Clustered primitives are progressively merged, keeping the required scale of 3DGS at any viewpoint. We evaluate our solution on a variety of datasets and show that our solution can provide on-the-fly processing of all the capture scenarios and scene sizes we target while remaining competitive with other methods that only handle specific capture styles or scene sizes in speed, image quality, or both.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05480v1">ODE-GS: Latent ODEs for Dynamic Scene Extrapolation with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      We present ODE-GS, a novel method that unifies 3D Gaussian Splatting with latent neural ordinary differential equations (ODEs) to forecast dynamic 3D scenes far beyond the time span seen during training. Existing neural rendering systems - whether NeRF- or 3DGS-based - embed time directly in a deformation network and therefore excel at interpolation but collapse when asked to predict the future, where timestamps are strictly out-of-distribution. ODE-GS eliminates this dependency: after learning a high-fidelity, time-conditioned deformation model for the training window, we freeze it and train a Transformer encoder that summarizes past Gaussian trajectories into a latent state whose continuous evolution is governed by a neural ODE. Numerical integration of this latent flow yields smooth, physically plausible Gaussian trajectories that can be queried at any future instant and rendered in real time. Coupled with a variational objective and a lightweight second-derivative regularizer, ODE-GS attains state-of-the-art extrapolation on D-NeRF and NVFI benchmarks, improving PSNR by up to 10 dB and halving perceptual error (LPIPS) relative to the strongest baselines. Our results demonstrate that continuous-time latent dynamics are a powerful, practical route to photorealistic prediction of complex 3D scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04218v1">Pseudo-Simulation for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Existing evaluation paradigms for Autonomous Vehicles (AVs) face critical limitations. Real-world evaluation is often challenging due to safety concerns and a lack of reproducibility, whereas closed-loop simulation can face insufficient realism or high computational costs. Open-loop evaluation, while being efficient and data-driven, relies on metrics that generally overlook compounding errors. In this paper, we propose pseudo-simulation, a novel paradigm that addresses these limitations. Pseudo-simulation operates on real datasets, similar to open-loop evaluation, but augments them with synthetic observations generated prior to evaluation using 3D Gaussian Splatting. Our key idea is to approximate potential future states the AV might encounter by generating a diverse set of observations that vary in position, heading, and speed. Our method then assigns a higher importance to synthetic observations that best match the AV's likely behavior using a novel proximity-based weighting scheme. This enables evaluating error recovery and the mitigation of causal confusion, as in closed-loop benchmarks, without requiring sequential interactive simulation. We show that pseudo-simulation is better correlated with closed-loop simulations (R^2=0.8) than the best existing open-loop approach (R^2=0.7). We also establish a public leaderboard for the community to benchmark new methodologies with pseudo-simulation. Our code is available at https://github.com/autonomousvision/navsim.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04174v1">FlexGS: Train Once, Deploy Everywhere with Many-in-One Flexible 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 CVPR 2025; Project Page: https://flexgs.github.io
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has enabled various applications in 3D scene representation and novel view synthesis due to its efficient rendering capabilities. However, 3DGS demands relatively significant GPU memory, limiting its use on devices with restricted computational resources. Previous approaches have focused on pruning less important Gaussians, effectively compressing 3DGS but often requiring a fine-tuning stage and lacking adaptability for the specific memory needs of different devices. In this work, we present an elastic inference method for 3DGS. Given an input for the desired model size, our method selects and transforms a subset of Gaussians, achieving substantial rendering performance without additional fine-tuning. We introduce a tiny learnable module that controls Gaussian selection based on the input percentage, along with a transformation module that adjusts the selected Gaussians to complement the performance of the reduced model. Comprehensive experiments on ZipNeRF, MipNeRF and Tanks\&Temples scenes demonstrate the effectiveness of our approach. Code is available at https://flexgs.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04120v1">Splatting Physical Scenes: End-to-End Real-to-Sim from Imperfect Robot Data</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Creating accurate, physical simulations directly from real-world robot motion holds great value for safe, scalable, and affordable robot learning, yet remains exceptionally challenging. Real robot data suffers from occlusions, noisy camera poses, dynamic scene elements, which hinder the creation of geometrically accurate and photorealistic digital twins of unseen objects. We introduce a novel real-to-sim framework tackling all these challenges at once. Our key insight is a hybrid scene representation merging the photorealistic rendering of 3D Gaussian Splatting with explicit object meshes suitable for physics simulation within a single representation. We propose an end-to-end optimization pipeline that leverages differentiable rendering and differentiable physics within MuJoCo to jointly refine all scene components - from object geometry and appearance to robot poses and physical parameters - directly from raw and imprecise robot trajectories. This unified optimization allows us to simultaneously achieve high-fidelity object mesh reconstruction, generate photorealistic novel views, and perform annotation-free robot pose calibration. We demonstrate the effectiveness of our approach both in simulation and on challenging real-world sequences using an ALOHA 2 bi-manual manipulator, enabling more practical and robust real-to-simulation pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03872v1">JointSplat: Probabilistic Joint Flow-Depth Optimization for Sparse-View Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Reconstructing 3D scenes from sparse viewpoints is a long-standing challenge with wide applications. Recent advances in feed-forward 3D Gaussian sparse-view reconstruction methods provide an efficient solution for real-time novel view synthesis by leveraging geometric priors learned from large-scale multi-view datasets and computing 3D Gaussian centers via back-projection. Despite offering strong geometric cues, both feed-forward multi-view depth estimation and flow-depth joint estimation face key limitations: the former suffers from mislocation and artifact issues in low-texture or repetitive regions, while the latter is prone to local noise and global inconsistency due to unreliable matches when ground-truth flow supervision is unavailable. To overcome this, we propose JointSplat, a unified framework that leverages the complementarity between optical flow and depth via a novel probabilistic optimization mechanism. Specifically, this pixel-level mechanism scales the information fusion between depth and flow based on the matching probability of optical flow during training. Building upon the above mechanism, we further propose a novel multi-view depth-consistency loss to leverage the reliability of supervision while suppressing misleading gradients in uncertain areas. Evaluated on RealEstate10K and ACID, JointSplat consistently outperforms state-of-the-art (SOTA) methods, demonstrating the effectiveness and robustness of our proposed probabilistic joint flow-depth optimization approach for high-fidelity sparse-view 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03594v1">SplArt: Articulation Estimation and Part-Level Reconstruction with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 https://github.com/ripl/splart
    </div>
    <details class="paper-abstract">
      Reconstructing articulated objects prevalent in daily environments is crucial for applications in augmented/virtual reality and robotics. However, existing methods face scalability limitations (requiring 3D supervision or costly annotations), robustness issues (being susceptible to local optima), and rendering shortcomings (lacking speed or photorealism). We introduce SplArt, a self-supervised, category-agnostic framework that leverages 3D Gaussian Splatting (3DGS) to reconstruct articulated objects and infer kinematics from two sets of posed RGB images captured at different articulation states, enabling real-time photorealistic rendering for novel viewpoints and articulations. SplArt augments 3DGS with a differentiable mobility parameter per Gaussian, achieving refined part segmentation. A multi-stage optimization strategy is employed to progressively handle reconstruction, part segmentation, and articulation estimation, significantly enhancing robustness and accuracy. SplArt exploits geometric self-supervision, effectively addressing challenging scenarios without requiring 3D annotations or category-specific priors. Evaluations on established and newly proposed benchmarks, along with applications to real-world scenarios using a handheld RGB camera, demonstrate SplArt's state-of-the-art performance and real-world practicality. Code is publicly available at https://github.com/ripl/splart.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03538v1">Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      3D reconstruction from in-the-wild images remains a challenging task due to inconsistent lighting conditions and transient distractors. Existing methods typically rely on heuristic strategies to handle the low-quality training data, which often struggle to produce stable and consistent reconstructions, frequently resulting in visual artifacts. In this work, we propose Asymmetric Dual 3DGS, a novel framework that leverages the stochastic nature of these artifacts: they tend to vary across different training runs due to minor randomness. Specifically, our method trains two 3D Gaussian Splatting (3DGS) models in parallel, enforcing a consistency constraint that encourages convergence on reliable scene geometry while suppressing inconsistent artifacts. To prevent the two models from collapsing into similar failure modes due to confirmation bias, we introduce a divergent masking strategy that applies two complementary masks: a multi-cue adaptive mask and a self-supervised soft mask, which leads to an asymmetric training process of the two models, reducing shared error modes. In addition, to improve the efficiency of model training, we introduce a lightweight variant called Dynamic EMA Proxy, which replaces one of the two models with a dynamically updated Exponential Moving Average (EMA) proxy, and employs an alternating masking strategy to preserve divergence. Extensive experiments on challenging real-world datasets demonstrate that our method consistently outperforms existing approaches while achieving high efficiency. Codes and trained models will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02774v2">Voyager: Real-Time Splatting City-Scale 3D Gaussians on Your Phone</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is an emerging technique for photorealistic 3D scene rendering. However, rendering city-scale 3DGS scenes on mobile devices, e.g., your smartphones, remains a significant challenge due to the limited resources on mobile devices. A natural solution is to offload computation to the cloud; however, naively streaming rendered frames from the cloud to the client introduces high latency and requires bandwidth far beyond the capacity of current wireless networks. In this paper, we propose an effective solution to enable city-scale 3DGS rendering on mobile devices. Our key insight is that, under normal user motion, the number of newly visible Gaussians per second remains roughly constant. Leveraging this, we stream only the necessary Gaussians to the client. Specifically, on the cloud side, we propose asynchronous level-of-detail search to identify the necessary Gaussians for the client. On the client side, we accelerate rendering via a lookup table-based rasterization. Combined with holistic runtime optimizations, our system can deliver low-latency, city-scale 3DGS rendering on mobile devices. Compared to existing solutions, Voyager achieves over 100$\times$ reduction on data transfer and up to 8.9$\times$ speedup while retaining comparable rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04444v1">Photoreal Scene Reconstruction from an Egocentric Device</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Paper accepted to SIGGRAPH Conference Paper 2025
    </div>
    <details class="paper-abstract">
      In this paper, we investigate the challenges associated with using egocentric devices to photorealistic reconstruct the scene in high dynamic range. Existing methodologies typically assume using frame-rate 6DoF pose estimated from the device's visual-inertial odometry system, which may neglect crucial details necessary for pixel-accurate reconstruction. This study presents two significant findings. Firstly, in contrast to mainstream work treating RGB camera as global shutter frame-rate camera, we emphasize the importance of employing visual-inertial bundle adjustment (VIBA) to calibrate the precise timestamps and movement of the rolling shutter RGB sensing camera in a high frequency trajectory format, which ensures an accurate calibration of the physical properties of the rolling-shutter camera. Secondly, we incorporate a physical image formation model based into Gaussian Splatting, which effectively addresses the sensor characteristics, including the rolling-shutter effect of RGB cameras and the dynamic ranges measured by sensors. Our proposed formulation is applicable to the widely-used variants of Gaussian Splats representation. We conduct a comprehensive evaluation of our pipeline using the open-source Project Aria device under diverse indoor and outdoor lighting conditions, and further validate it on a Meta Quest3 device. Across all experiments, we observe a consistent visual enhancement of +1 dB in PSNR by incorporating VIBA, with an additional +1 dB achieved through our proposed image formation model. Our complete implementation, evaluation datasets, and recording profile are available at http://www.projectaria.com/photoreal-reconstruction/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04351v1">HuGeDiff: 3D Human Generation via Diffusion with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      3D human generation is an important problem with a wide range of applications in computer vision and graphics. Despite recent progress in generative AI such as diffusion models or rendering methods like Neural Radiance Fields or Gaussian Splatting, controlling the generation of accurate 3D humans from text prompts remains an open challenge. Current methods struggle with fine detail, accurate rendering of hands and faces, human realism, and controlability over appearance. The lack of diversity, realism, and annotation in human image data also remains a challenge, hindering the development of a foundational 3D human model. We present a weakly supervised pipeline that tries to address these challenges. In the first step, we generate a photorealistic human image dataset with controllable attributes such as appearance, race, gender, etc using a state-of-the-art image diffusion model. Next, we propose an efficient mapping approach from image features to 3D point clouds using a transformer-based architecture. Finally, we close the loop by training a point-cloud diffusion model that is conditioned on the same text prompts used to generate the original samples. We demonstrate orders-of-magnitude speed-ups in 3D human generation compared to the state-of-the-art approaches, along with significantly improved text-prompt alignment, realism, and rendering quality. We will make the code and dataset available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03073v1">LEG-SLAM: Real-Time Language-Enhanced Gaussian Splatting for SLAM</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Modern Gaussian Splatting methods have proven highly effective for real-time photorealistic rendering of 3D scenes. However, integrating semantic information into this representation remains a significant challenge, especially in maintaining real-time performance for SLAM (Simultaneous Localization and Mapping) applications. In this work, we introduce LEG-SLAM -- a novel approach that fuses an optimized Gaussian Splatting implementation with visual-language feature extraction using DINOv2 followed by a learnable feature compressor based on Principal Component Analysis, while enabling an online dense SLAM. Our method simultaneously generates high-quality photorealistic images and semantically labeled scene maps, achieving real-time scene reconstruction with more than 10 fps on the Replica dataset and 18 fps on ScanNet. Experimental results show that our approach significantly outperforms state-of-the-art methods in reconstruction speed while achieving competitive rendering quality. The proposed system eliminates the need for prior data preparation such as camera's ego motion or pre-computed static semantic maps. With its potential applications in autonomous robotics, augmented reality, and other interactive domains, LEG-SLAM represents a significant step forward in real-time semantic 3D Gaussian-based SLAM. Project page: https://titrom025.github.io/LEG-SLAM/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18052v2">SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Our code, model, and dataset will be released at https://unique1i.github.io/SceneSplat_webpage/
    </div>
    <details class="paper-abstract">
      Recognizing arbitrary or previously unseen categories is essential for comprehensive real-world 3D scene understanding. Currently, all existing methods rely on 2D or textual modalities during training or together at inference. This highlights the clear absence of a model capable of processing 3D data alone for learning semantics end-to-end, along with the necessary data to train such a model. Meanwhile, 3D Gaussian Splatting (3DGS) has emerged as the de facto standard for 3D scene representation across various vision tasks. However, effectively integrating semantic reasoning into 3DGS in a generalizable manner remains an open challenge. To address these limitations, we introduce SceneSplat, to our knowledge the first large-scale 3D indoor scene understanding approach that operates natively on 3DGS. Furthermore, we propose a self-supervised learning scheme that unlocks rich 3D feature learning from unlabeled scenes. To power the proposed methods, we introduce SceneSplat-7K, the first large-scale 3DGS dataset for indoor scenes, comprising 7916 scenes derived from seven established datasets, such as ScanNet and Matterport3D. Generating SceneSplat-7K required computational resources equivalent to 150 GPU days on an L4 GPU, enabling standardized benchmarking for 3DGS-based reasoning for indoor scenes. Our exhaustive experiments on SceneSplat-7K demonstrate the significant benefit of the proposed method over the established baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14147v2">HAMMER: Heterogeneous, Multi-Robot Semantic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting offers expressive scene reconstruction, modeling a broad range of visual, geometric, and semantic information. However, efficient real-time map reconstruction with data streamed from multiple robots and devices remains a challenge. To that end, we propose HAMMER, a server-based collaborative Gaussian Splatting method that leverages widely available ROS communication infrastructure to generate 3D, metric-semantic maps from asynchronous robot data-streams with no prior knowledge of initial robot positions and varying on-device pose estimators. HAMMER consists of (i) a frame alignment module that transforms local SLAM poses and image data into a global frame and requires no prior relative pose knowledge, and (ii) an online module for training semantic 3DGS maps from streaming data. HAMMER handles mixed perception modes, adjusts automatically for variations in image pre-processing among different devices, and distills CLIP semantic codes into the 3D scene for open-vocabulary language queries. In our real-world experiments, HAMMER creates higher-fidelity maps (2x) compared to competing baselines and is useful for downstream tasks, such as semantic goal-conditioned navigation (e.g., "go to the couch"). Accompanying content available at hammer-project.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02929v1">Large Processor Chip Model</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Computer System Architecture serves as a crucial bridge between software applications and the underlying hardware, encompassing components like compilers, CPUs, coprocessors, and RTL designs. Its development, from early mainframes to modern domain-specific architectures, has been driven by rising computational demands and advancements in semiconductor technology. However, traditional paradigms in computer system architecture design are confronting significant challenges, including a reliance on manual expertise, fragmented optimization across software and hardware layers, and high costs associated with exploring expansive design spaces. While automated methods leveraging optimization algorithms and machine learning have improved efficiency, they remain constrained by a single-stage focus, limited data availability, and a lack of comprehensive human domain knowledge. The emergence of large language models offers transformative opportunities for the design of computer system architecture. By leveraging the capabilities of LLMs in areas such as code generation, data analysis, and performance modeling, the traditional manual design process can be transitioned to a machine-based automated design approach. To harness this potential, we present the Large Processor Chip Model (LPCM), an LLM-driven framework aimed at achieving end-to-end automated computer architecture design. The LPCM is structured into three levels: Human-Centric; Agent-Orchestrated; and Model-Governed. This paper utilizes 3D Gaussian Splatting as a representative workload and employs the concept of software-hardware collaborative design to examine the implementation of the LPCM at Level 1, demonstrating the effectiveness of the proposed approach. Furthermore, this paper provides an in-depth discussion on the pathway to implementing Level 2 and Level 3 of the LPCM, along with an analysis of the existing challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02774v1">Voyager: Real-Time Splatting City-Scale 3D Gaussians on Your Phone</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is an emerging technique for photorealistic 3D scene rendering. However, rendering city-scale 3DGS scenes on mobile devices, e.g., your smartphones, remains a significant challenge due to the limited resources on mobile devices. A natural solution is to offload computation to the cloud; however, naively streaming rendered frames from the cloud to the client introduces high latency and requires bandwidth far beyond the capacity of current wireless networks. In this paper, we propose an effective solution to enable city-scale 3DGS rendering on mobile devices. Our key insight is that, under normal user motion, the number of newly visible Gaussians per second remains roughly constant. Leveraging this, we stream only the necessary Gaussians to the client. Specifically, on the cloud side, we propose asynchronous level-of-detail search to identify the necessary Gaussians for the client. On the client side, we accelerate rendering via a lookup table-based rasterization. Combined with holistic runtime optimizations, our system can deliver low-latency, city-scale 3DGS rendering on mobile devices. Compared to existing solutions, Voyager achieves over 100$\times$ reduction on data transfer and up to 8.9$\times$ speedup while retaining comparable rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02751v1">RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Project page: https://fcyycf.github.io/RobustSplat/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has gained significant attention for its real-time, photo-realistic rendering in novel-view synthesis and 3D modeling. However, existing methods struggle with accurately modeling scenes affected by transient objects, leading to artifacts in the rendered images. We identify that the Gaussian densification process, while enhancing scene detail capture, unintentionally contributes to these artifacts by growing additional Gaussians that model transient disturbances. To address this, we propose RobustSplat, a robust solution based on two critical designs. First, we introduce a delayed Gaussian growth strategy that prioritizes optimizing static scene structure before allowing Gaussian splitting/cloning, mitigating overfitting to transient objects in early optimization. Second, we design a scale-cascaded mask bootstrapping approach that first leverages lower-resolution feature similarity supervision for reliable initial transient mask estimation, taking advantage of its stronger semantic consistency and robustness to noise, and then progresses to high-resolution supervision to achieve more precise mask prediction. Extensive experiments on multiple challenging datasets show that our method outperforms existing methods, clearly demonstrating the robustness and effectiveness of our method. Our project page is https://fcyycf.github.io/RobustSplat/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01536v3">VR-Robo: A Real-to-Sim-to-Real Framework for Visual Robot Navigation and Locomotion</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Project Page: https://vr-robo.github.io/
    </div>
    <details class="paper-abstract">
      Recent success in legged robot locomotion is attributed to the integration of reinforcement learning and physical simulators. However, these policies often encounter challenges when deployed in real-world environments due to sim-to-real gaps, as simulators typically fail to replicate visual realism and complex real-world geometry. Moreover, the lack of realistic visual rendering limits the ability of these policies to support high-level tasks requiring RGB-based perception like ego-centric navigation. This paper presents a Real-to-Sim-to-Real framework that generates photorealistic and physically interactive "digital twin" simulation environments for visual navigation and locomotion learning. Our approach leverages 3D Gaussian Splatting (3DGS) based scene reconstruction from multi-view images and integrates these environments into simulations that support ego-centric visual perception and mesh-based physical interactions. To demonstrate its effectiveness, we train a reinforcement learning policy within the simulator to perform a visual goal-tracking task. Extensive experiments show that our framework achieves RGB-only sim-to-real policy transfer. Additionally, our framework facilitates the rapid adaptation of robot policies with effective exploration capability in complex new environments, highlighting its potential for applications in households and factories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02380v1">EyeNavGS: A 6-DoF Navigation Dataset and Record-n-Replay Software for Real-World 3DGS Scenes in VR</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is an emerging media representation that reconstructs real-world 3D scenes in high fidelity, enabling 6-degrees-of-freedom (6-DoF) navigation in virtual reality (VR). However, developing and evaluating 3DGS-enabled applications and optimizing their rendering performance, require realistic user navigation data. Such data is currently unavailable for photorealistic 3DGS reconstructions of real-world scenes. This paper introduces EyeNavGS (EyeNavGS), the first publicly available 6-DoF navigation dataset featuring traces from 46 participants exploring twelve diverse, real-world 3DGS scenes. The dataset was collected at two sites, using the Meta Quest Pro headsets, recording the head pose and eye gaze data for each rendered frame during free world standing 6-DoF navigation. For each of the twelve scenes, we performed careful scene initialization to correct for scene tilt and scale, ensuring a perceptually-comfortable VR experience. We also release our open-source SIBR viewer software fork with record-and-replay functionalities and a suite of utility tools for data processing, conversion, and visualization. The EyeNavGS dataset and its accompanying software tools provide valuable resources for advancing research in 6-DoF viewport prediction, adaptive streaming, 3D saliency, and foveated rendering for 3DGS scenes. The EyeNavGS dataset is available at: https://symmru.github.io/EyeNavGS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03407v1">Multi-Spectral Gaussian Splatting with Neural Color Representation</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      We present MS-Splatting -- a multi-spectral 3D Gaussian Splatting (3DGS) framework that is able to generate multi-view consistent novel views from images of multiple, independent cameras with different spectral domains. In contrast to previous approaches, our method does not require cross-modal camera calibration and is versatile enough to model a variety of different spectra, including thermal and near-infra red, without any algorithmic changes. Unlike existing 3DGS-based frameworks that treat each modality separately (by optimizing per-channel spherical harmonics) and therefore fail to exploit the underlying spectral and spatial correlations, our method leverages a novel neural color representation that encodes multi-spectral information into a learned, compact, per-splat feature embedding. A shallow multi-layer perceptron (MLP) then decodes this embedding to obtain spectral color values, enabling joint learning of all bands within a unified representation. Our experiments show that this simple yet effective strategy is able to improve multi-spectral rendering quality, while also leading to improved per-spectra rendering quality over state-of-the-art methods. We demonstrate the effectiveness of this new technique in agricultural applications to render vegetation indices, such as normalized difference vegetation index (NDVI).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05397v1">Gen4D: Synthesizing Humans and Scenes in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops
    </div>
    <details class="paper-abstract">
      Lack of input data for in-the-wild activities often results in low performance across various computer vision tasks. This challenge is particularly pronounced in uncommon human-centric domains like sports, where real-world data collection is complex and impractical. While synthetic datasets offer a promising alternative, existing approaches typically suffer from limited diversity in human appearance, motion, and scene composition due to their reliance on rigid asset libraries and hand-crafted rendering pipelines. To address this, we introduce Gen4D, a fully automated pipeline for generating diverse and photorealistic 4D human animations. Gen4D integrates expert-driven motion encoding, prompt-guided avatar generation using diffusion-based Gaussian splatting, and human-aware background synthesis to produce highly varied and lifelike human sequences. Based on Gen4D, we present SportPAL, a large-scale synthetic dataset spanning three sports: baseball, icehockey, and soccer. Together, Gen4D and SportPAL provide a scalable foundation for constructing synthetic datasets tailored to in-the-wild human-centric vision tasks, with no need for manual 3D modeling or scene design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08438v2">A Survey of 3D Reconstruction with Event Cameras</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 24 pages, 16 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Event cameras are rapidly emerging as powerful vision sensors for 3D reconstruction, uniquely capable of asynchronously capturing per-pixel brightness changes. Compared to traditional frame-based cameras, event cameras produce sparse yet temporally dense data streams, enabling robust and accurate 3D reconstruction even under challenging conditions such as high-speed motion, low illumination, and extreme dynamic range scenarios. These capabilities offer substantial promise for transformative applications across various fields, including autonomous driving, robotics, aerial navigation, and immersive virtual reality. In this survey, we present the first comprehensive review exclusively dedicated to event-based 3D reconstruction. Existing approaches are systematically categorised based on input modality into stereo, monocular, and multimodal systems, and further classified according to reconstruction methodologies, including geometry-based techniques, deep learning approaches, and neural rendering techniques such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). Within each category, methods are chronologically organised to highlight the evolution of key concepts and advancements. Furthermore, we provide a detailed summary of publicly available datasets specifically suited to event-based reconstruction tasks. Finally, we discuss significant open challenges in dataset availability, standardised evaluation, effective representation, and dynamic scene reconstruction, outlining insightful directions for future research. This survey aims to serve as an essential reference and provides a clear and motivating roadmap toward advancing the state of the art in event-driven 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19753v3">A Survey on Event-driven 3D Reconstruction: Development under Different Categories</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 We have decided not to submit this article and plan to withdraw it from public display. The content of this article will be presented in a more comprehensive form in another work
    </div>
    <details class="paper-abstract">
      Event cameras have gained increasing attention for 3D reconstruction due to their high temporal resolution, low latency, and high dynamic range. They capture per-pixel brightness changes asynchronously, allowing accurate reconstruction under fast motion and challenging lighting conditions. In this survey, we provide a comprehensive review of event-driven 3D reconstruction methods, including stereo, monocular, and multimodal systems. We further categorize recent developments based on geometric, learning-based, and hybrid approaches. Emerging trends, such as neural radiance fields and 3D Gaussian splatting with event data, are also covered. The related works are structured chronologically to illustrate the innovations and progression within the field. To support future research, we also highlight key research gaps and future research directions in dataset, experiment, evaluation, event representation, etc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17605v2">Distractor-free Generalizable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      We present DGGS, a novel framework that addresses the previously unexplored challenge: $\textbf{Distractor-free Generalizable 3D Gaussian Splatting}$ (3DGS). It mitigates 3D inconsistency and training instability caused by distractor data in the cross-scenes generalizable train setting while enabling feedforward inference for 3DGS and distractor masks from references in the unseen scenes. To achieve these objectives, DGGS proposes a scene-agnostic reference-based mask prediction and refinement module during the training phase, effectively eliminating the impact of distractor on training stability. Moreover, we combat distractor-induced artifacts and holes at inference time through a novel two-stage inference framework for references scoring and re-selection, complemented by a distractor pruning mechanism that further removes residual distractor 3DGS-primitive influences. Extensive feedforward experiments on the real and our synthetic data show DGGS's reconstruction capability when dealing with novel distractor scenes. Moreover, our generalizable mask prediction even achieves an accuracy superior to existing scene-specific training methods. Homepage is https://github.com/bbbbby-99/DGGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.05819v2">GASP: Gaussian Splatting for Physic-Based Simulations</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Physics simulation is paramount for modeling and utilization of 3D scenes in various real-world applications. However, its integration with state-of-the-art 3D scene rendering techniques such as Gaussian Splatting (GS) remains challenging. Existing models use additional meshing mechanisms, including triangle or tetrahedron meshing, marching cubes, or cage meshes. As an alternative, we can modify the physics grounded Newtonian dynamics to align with 3D Gaussian components. Current models take the first-order approximation of a deformation map, which locally approximates the dynamics by linear transformations. In contrast, our Gaussian Splatting for Physics-Based Simulations (GASP) model uses such a map (without any modifications) and flat Gaussian distributions, which are parameterized by three points (mesh faces). Subsequently, each 3D point (mesh face node) is treated as a discrete entity within a 3D space. Consequently, the problem of modeling Gaussian components is reduced to working with 3D points. Additionally, the information on mesh faces can be used to incorporate further properties into the physics model, facilitating the use of triangles. Resulting solution can be integrated into any physics engine that can be treated as a black box. As demonstrated in our studies, the proposed model exhibits superior performance on a diverse range of benchmark datasets designed for 3D object rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17345v2">SeaSplat: Representing Underwater Scenes with 3D Gaussian Splatting and a Physically Grounded Image Formation Model</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 ICRA 2025. Project page here: https://seasplat.github.io
    </div>
    <details class="paper-abstract">
      We introduce SeaSplat, a method to enable real-time rendering of underwater scenes leveraging recent advances in 3D radiance fields. Underwater scenes are challenging visual environments, as rendering through a medium such as water introduces both range and color dependent effects on image capture. We constrain 3D Gaussian Splatting (3DGS), a recent advance in radiance fields enabling rapid training and real-time rendering of full 3D scenes, with a physically grounded underwater image formation model. Applying SeaSplat to the real-world scenes from SeaThru-NeRF dataset, a scene collected by an underwater vehicle in the US Virgin Islands, and simulation-degraded real-world scenes, not only do we see increased quantitative performance on rendering novel viewpoints from the scene with the medium present, but are also able to recover the underlying true color of the scene and restore renders to be without the presence of the intervening medium. We show that the underwater image formation helps learn scene structure, with better depth maps, as well as show that our improvements maintain the significant computational improvements afforded by leveraging a 3D Gaussian representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01822v1">GSCodec Studio: A Modular Framework for Gaussian Splat Compression</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Repository of the project: https://github.com/JasonLSC/GSCodec_Studio
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting and its extension to 4D dynamic scenes enable photorealistic, real-time rendering from real-world captures, positioning Gaussian Splats (GS) as a promising format for next-generation immersive media. However, their high storage requirements pose significant challenges for practical use in sharing, transmission, and storage. Despite various studies exploring GS compression from different perspectives, these efforts remain scattered across separate repositories, complicating benchmarking and the integration of best practices. To address this gap, we present GSCodec Studio, a unified and modular framework for GS reconstruction, compression, and rendering. The framework incorporates a diverse set of 3D/4D GS reconstruction methods and GS compression techniques as modular components, facilitating flexible combinations and comprehensive comparisons. By integrating best practices from community research and our own explorations, GSCodec Studio supports the development of compact representation and compression solutions for static and dynamic Gaussian Splats, namely our Static and Dynamic GSCodec, achieving competitive rate-distortion performance in static and dynamic GS compression. The code for our framework is publicly available at https://github.com/JasonLSC/GSCodec_Studio , to advance the research on Gaussian Splats compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01799v1">WorldExplorer: Towards Generating Fully Navigable 3D Scenes</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 project page: see https://the-world-explorer.github.io/, video: see https://youtu.be/c1lBnwJWNmE
    </div>
    <details class="paper-abstract">
      Generating 3D worlds from text is a highly anticipated goal in computer vision. Existing works are limited by the degree of exploration they allow inside of a scene, i.e., produce streched-out and noisy artifacts when moving beyond central or panoramic perspectives. To this end, we propose WorldExplorer, a novel method based on autoregressive video trajectory generation, which builds fully navigable 3D scenes with consistent visual quality across a wide range of viewpoints. We initialize our scenes by creating multi-view consistent images corresponding to a 360 degree panorama. Then, we expand it by leveraging video diffusion models in an iterative scene generation pipeline. Concretely, we generate multiple videos along short, pre-defined trajectories, that explore the scene in depth, including motion around objects. Our novel scene memory conditions each video on the most relevant prior views, while a collision-detection mechanism prevents degenerate results, like moving into objects. Finally, we fuse all generated views into a unified 3D representation via 3D Gaussian Splatting optimization. Compared to prior approaches, WorldExplorer produces high-quality scenes that remain stable under large camera motion, enabling for the first time realistic and unrestricted exploration. We believe this marks a significant step toward generating immersive and truly explorable virtual 3D environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01600v1">WoMAP: World Models For Embodied Open-Vocabulary Object Localization</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Language-instructed active object localization is a critical challenge for robots, requiring efficient exploration of partially observable environments. However, state-of-the-art approaches either struggle to generalize beyond demonstration datasets (e.g., imitation learning methods) or fail to generate physically grounded actions (e.g., VLMs). To address these limitations, we introduce WoMAP (World Models for Active Perception): a recipe for training open-vocabulary object localization policies that: (i) uses a Gaussian Splatting-based real-to-sim-to-real pipeline for scalable data generation without the need for expert demonstrations, (ii) distills dense rewards signals from open-vocabulary object detectors, and (iii) leverages a latent world model for dynamics and rewards prediction to ground high-level action proposals at inference time. Rigorous simulation and hardware experiments demonstrate WoMAP's superior performance in a broad range of zero-shot object localization tasks, with more than 9x and 2x higher success rates compared to VLM and diffusion policy baselines, respectively. Further, we show that WoMAP achieves strong generalization and sim-to-real transfer on a TidyBot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01379v1">RadarSplat: Radar Gaussian Splatting for High-Fidelity Data Synthesis and 3D Reconstruction of Autonomous Driving Scenes</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      High-Fidelity 3D scene reconstruction plays a crucial role in autonomous driving by enabling novel data generation from existing datasets. This allows simulating safety-critical scenarios and augmenting training datasets without incurring further data collection costs. While recent advances in radiance fields have demonstrated promising results in 3D reconstruction and sensor data synthesis using cameras and LiDAR, their potential for radar remains largely unexplored. Radar is crucial for autonomous driving due to its robustness in adverse weather conditions like rain, fog, and snow, where optical sensors often struggle. Although the state-of-the-art radar-based neural representation shows promise for 3D driving scene reconstruction, it performs poorly in scenarios with significant radar noise, including receiver saturation and multipath reflection. Moreover, it is limited to synthesizing preprocessed, noise-excluded radar images, failing to address realistic radar data synthesis. To address these limitations, this paper proposes RadarSplat, which integrates Gaussian Splatting with novel radar noise modeling to enable realistic radar data synthesis and enhanced 3D reconstruction. Compared to the state-of-the-art, RadarSplat achieves superior radar image synthesis (+3.4 PSNR / 2.6x SSIM) and improved geometric reconstruction (-40% RMSE / 1.5x Accuracy), demonstrating its effectiveness in generating high-fidelity radar data and scene reconstruction. A project page is available at https://umautobots.github.io/radarsplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04343v2">Flash3D: Feed-Forward Generalisable 3D Scene Reconstruction from a Single Image</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Project page: https://www.robots.ox.ac.uk/~vgg/research/flash3d/
    </div>
    <details class="paper-abstract">
      We propose Flash3D, a method for scene reconstruction and novel view synthesis from a single image which is both very generalisable and efficient. For generalisability, we start from a "foundation" model for monocular depth estimation and extend it to a full 3D shape and appearance reconstructor. For efficiency, we base this extension on feed-forward Gaussian Splatting. Specifically, we predict a first layer of 3D Gaussians at the predicted depth, and then add additional layers of Gaussians that are offset in space, allowing the model to complete the reconstruction behind occlusions and truncations. Flash3D is very efficient, trainable on a single GPU in a day, and thus accessible to most researchers. It achieves state-of-the-art results when trained and tested on RealEstate10k. When transferred to unseen datasets like NYU it outperforms competitors by a large margin. More impressively, when transferred to KITTI, Flash3D achieves better PSNR than methods trained specifically on that dataset. In some instances, it even outperforms recent methods that use multiple views as input. Code, models, demo, and more results are available at https://www.robots.ox.ac.uk/~vgg/research/flash3d/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01109v1">CountingFruit: Real-Time 3D Fruit Counting with Language-Guided Semantic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Accurate fruit counting in real-world agricultural environments is a longstanding challenge due to visual occlusions, semantic ambiguity, and the high computational demands of 3D reconstruction. Existing methods based on neural radiance fields suffer from low inference speed, limited generalization, and lack support for open-set semantic control. This paper presents FruitLangGS, a real-time 3D fruit counting framework that addresses these limitations through spatial reconstruction, semantic embedding, and language-guided instance estimation. FruitLangGS first reconstructs orchard-scale scenes using an adaptive Gaussian splatting pipeline with radius-aware pruning and tile-based rasterization for efficient rendering. To enable semantic control, each Gaussian encodes a compressed CLIP-aligned language embedding, forming a compact and queryable 3D representation. At inference time, prompt-based semantic filtering is applied directly in 3D space, without relying on image-space segmentation or view-level fusion. The selected Gaussians are then converted into dense point clouds via distribution-aware sampling and clustered to estimate fruit counts. Experimental results on real orchard data demonstrate that FruitLangGS achieves higher rendering speed, semantic flexibility, and counting accuracy compared to prior approaches, offering a new perspective for language-driven, real-time neural rendering across open-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00970v1">Globally Consistent RGB-D SLAM with 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian splatting-based RGB-D SLAM displays remarkable performance of high-fidelity 3D reconstruction. However, the lack of depth rendering consistency and efficient loop closure limits the quality of its geometric reconstructions and its ability to perform globally consistent mapping online. In this paper, we present 2DGS-SLAM, an RGB-D SLAM system using 2D Gaussian splatting as the map representation. By leveraging the depth-consistent rendering property of the 2D variant, we propose an accurate camera pose optimization method and achieve geometrically accurate 3D reconstruction. In addition, we implement efficient loop detection and camera relocalization by leveraging MASt3R, a 3D foundation model, and achieve efficient map updates by maintaining a local active map. Experiments show that our 2DGS-SLAM approach achieves superior tracking accuracy, higher surface reconstruction quality, and more consistent global map reconstruction compared to existing rendering-based SLAM methods, while maintaining high-fidelity image rendering and improved computational efficiency.
    </details>
</div>
