# gaussian splatting - 2026_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03931v1">Sparse auto-regressive modeling for scene generation from multi-view images</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Accepted at ECCVV 2026
    </div>
    <details class="paper-abstract">
      Generating complete 3D scenes from sparse, unconstrained views is a fundamental challenge in 3D vision which requires reasoning beyond observed content while remaining computationally tractable. Existing feed-forward reconstruction methods are inherently limited to content visible in the input images, while 3D generative modeling is hindered by the high computational cost of dense volumetric representations and the scarcity of large-scale 3D supervision. We introduce SPAR3S, a sparse voxel-aligned 3D latent generative model for conditional scene completion without requiring ground-truth 3D data for supervision. Our key insight is to formulate 3D scene generation in a structured, compact, voxel-aligned 3D latent space where only occupied voxels are represented. We learn this sparse latent space directly from multi-view images using photometric supervision via differentiable 3D Gaussian Splatting. Given a partial set of observed voxels encoded from sparse input views, scene completion reduces to predicting the missing latent tokens and their spatial support within the voxel grid. To this end, we train a masked autoregressive transformer that jointly models voxel occupancy and latent token values, enabling efficient and spatially consistent generation of unseen regions. We demonstrate the effectiveness of our method on synthetic indoor scenes, achieving higher novel-view quality than prior work. We further validate its generalization on RealEstate10k, highlighting its applicability to real-world data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03897v1">Reparametrizing 3D Gaussian Splatting for Real-Time Palette-based Color and Luminance Editing</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 11 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Professional color editing requires precise control over both color (hue and saturation) and lightness, ideally through separate, independent controls. We present a real-time interactive color editing framework for 3D Gaussian Splatting that supports palette-based recoloring, per-palette tone curves for color-aware luminance adjustment, and pixel-level color constraints. Rather than training a new representation from scratch, we reparameterize the spherical harmonics of a pretrained vanilla 3DGS to encode view-dependent palette weights. We simultaneously solve for weights and palette colors via a loss based on image-space sparsity. Luminance editing is realized as a per-pixel weight shift along the achromatic axis, which we show is equivalent to a per-pixel palette-aware luminance edit. This view-space formulation addresses a core limitation of prior primitive-space methods, where alpha-blending breaks per-Gaussian sparsity and causes edits to bleed into unintended regions. Our edits run in tens of milliseconds via an iteratively reweighted least squares and damped block-coordinate descent that couples tone curves and palette shifts under view-space sparsity. Our representation can be efficiently baked back into a vanilla 3DGS, preserving compatibility with standard viewers. We demonstrate sparser, more localized edits than prior palette-based 3DGS methods, while enabling independent luminance control per palette color and view-consistent pixel-level constraints, capabilities previously unavailable for 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.21304v3">F4Splat: Feed-Forward Predictive Densification for Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Project Page: $\href{https://mlvlab.github.io/F4Splat}{\text{this http URL}}$
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting methods enable single-pass reconstruction and real-time rendering. However, they typically adopt rigid pixel-to-Gaussian or voxel-to-Gaussian pipelines that uniformly allocate Gaussians, leading to redundant Gaussians across views. Moreover, they lack an effective mechanism to control the total number of Gaussians while maintaining reconstruction fidelity. To address these limitations, we present F4Splat, which performs Feed-Forward predictive densification for Feed-Forward 3D Gaussian Splatting, introducing a densification-score-guided allocation strategy that adaptively distributes Gaussians according to spatial complexity and multi-view overlap. Our model predicts per-region densification scores to estimate the required Gaussian density and allows explicit control over the final Gaussian budget without retraining. This spatially adaptive allocation reduces redundancy in simple regions and minimizes duplicate Gaussians across overlapping views, producing compact yet high-quality 3D representations. Extensive experiments demonstrate that our model achieves superior novel-view synthesis performance compared to prior uncalibrated feed-forward methods, while using significantly fewer Gaussians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03657v1">Rethinking 3D Noise: Learning 3D-Aware Video Priors via Optimization-Free Morphological Perturbations</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      3D scene representations like NeRF and 3D Gaussian Splatting (3DGS) suffer severe artifacts in sparse-view settings. Recent generative 3D artifact fixers attempt to address this, but rely on paired corrupted and clean renders requiring costly, per-scene reconstructions across varying view configurations. While 2D image augmentations act as instant regularizers, no explicit equivalents exist for 3D representations to preserve spatial consistency across views, an essential property for 3D-aware training. We propose 3D Morphological Perturbations as an optimization-free regularizer that preserves spatial consistency. Leveraging explicit 3DGS, we treat each Gaussian as a fundamental building block - analogous to a 2D pixel - and apply perturbations across its morphological parameter space via scale, rotation, and pruning. Our method eliminates per-scene 3DGS optimization loops from dataset curation while enabling models to learn stronger geometric priors than sparse-view baselines in diagnostic ablations conducted on a lightweight video diffusion sandbox. Scaled to a 14B-parameter video model via ControlNet, our approach maintains visual fidelity while reducing mean depth error by 12.5% over state-of-the-art image-to-image 3D artifact refiners, ultimately boosting downstream robotics policy success rates by up to 8.0% across 3 of 4 manipulation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03613v1">TileGS: Tile-Local Depth Binning for Gaussian Splatting Rasterization</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Real-time 3D Gaussian Splatting (3DGS) achieves high rendering quality, but standard rasterization still traverses a globally sorted tile stream that creates long per-tile ranges and heavy geometry-attribute traffic. We present TileGS, a tile-local reorganization of Gaussian splatting. TileGS turns each long tile range into a sequence of shorter depth-local ranges, rasterizes those ranges in front-to-back order, and applies selective repair where coarse ordering is insufficient to match baseline compositing. Across a 9-scene benchmark on desktop and laptop Ada GPUs, our default No-GW (No Geometry-Write) variant delivers a mean 1.44x raster-kernel speedup on RTX 4090 and mean end-to-end frame speedups of 1.069x on RTX 4090 and 1.094x on RTX 1000 Ada over gsplat--a widely used optimized open-source 3DGS implementation--while matching the gsplat output up to numerical noise (|Delta PSNR| < 0.001 dB, |Delta SSIM| < 0.001, |Delta LPIPS| < 0.001). Full-suite RTX 4090 Nsight Compute profiling reveals TileGS is faster despite lower SM throughput, lower active-warp occupancy, and higher DRAM traffic, while total SASS thread instructions fall by 1.26x. Source-attributed profiling confirms that geometry attributes dominate the remaining memory pressure (85.8% of total raster traffic and 88.6% of excess sectors). Together, these counters support the interpretation that TileGS improves raster performance by reducing effective raster traversal work, rather than by reducing byte volume, improving coalescing, increasing occupancy, or directly reducing measured warp divergence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.15677v2">Camera Splatting for Continuous View Optimization</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 12 pages, Computer Graphics Forum (Proc. Pacific Graphics 2026)
    </div>
    <details class="paper-abstract">
      We propose Camera Splatting, a novel view optimization framework for novel view synthesis. Each camera is modeled as a 3D Gaussian, referred to as a camera splat, and virtual cameras, termed point cameras, are placed at 3D points sampled near the surface to observe the distribution of camera splats. View optimization is achieved by continuously and differentiably refining the camera splats so that desirable target distributions are observed from the point cameras, in a manner similar to the original 3D Gaussian splatting. Compared to the Farthest View Sampling (FVS) approach, our optimized views demonstrate superior performance in capturing complex view-dependent phenomena, including intense metallic reflections and intricate textures such as text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03534v1">TruncGradGS: Improved 3D Gaussian Splatting via Truncated Gradient Updates</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Accepted at Pacific Graphics 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has become a de facto scene representation for novel view synthesis, yet robustly learning 3D Gaussian primitives from visual input remains challenging. Standard optimization relies on gradient-based updates, but a common issue is the gradient vanishing phenomenon: a pixel far from a Gaussian primitive often has diminishing gradient magnitudes to influence primitive attributes, resulting in suboptimal scene reconstruction. In this paper, we propose a method to address gradient vanishing with a piecewise truncated gradient formulation that improves the optimization stability and robustness to initializations. We show that our method consistently improves 3D Gaussian Splatting with random and COLMAP initializations while being generalizable across static and dynamic Gaussian Splatting. As a by-product, we also examine the limitations of current benchmarks for dynamic scenes, and introduce a novel dataset for benchmarking dynamic Gaussian Splatting using synthetic 3D scenes. We demonstrate the effectiveness of our method in both static and dynamic settings for the public benchmarks and our proposed dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07604v2">EmbedTalk: Talking Head Synthesis using Gaussian Embeddings</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Deformable 3D Gaussian Splatting (3DGS) has emerged as a popular method for real-time talking head synthesis, offering high-quality renderings at low latency. Tri-planes are a common choice for encoding Gaussians prior to deformation since they provide a compact and continuous representation. However, tri-plane encodings are limited by grid resolution and approximation errors introduced by projecting 3D volumetric fields onto 2D subspaces. Recent work has demonstrated the effectiveness of per-Gaussian embeddings for driving temporal deformations in 4D scene reconstruction. We introduce EmbedTalk, which leverages these embeddings to model speech-driven facial deformations for talking head synthesis. Comprehensive experiments show that EmbedTalk improves rendering quality, lip synchronisation, and motion consistency over previous 3DGS-based methods, while remaining competitive with state-of-the-art generative models. Replacing tri-plane features with embeddings also yields significantly more compact models that achieve 60+ FPS on a laptop GPU (RTX 2060 6 GB). Our code will be placed in the public domain on acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03447v1">STARS-GS: Structure-Aware Regularized Gaussian Splatting for Large-Scale Aerial Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Large-scale 3D surface reconstruction from aerial imagery is fundamental to geospatial mapping and urban modeling. Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated considerable potential for this task. However, existing methods still face three major challenges in large and complex scenes: scene partitioning may split continuous scene elements across independently optimized sub-regions; geometric constraints mainly focus on the attributes of individual Gaussians while overlooking their local organization; and uniform regularization struggles to accommodate heterogeneous geometric structures. To address these issues, we propose STARS-GS, a structure-aware 3DGS framework for large-scale surface reconstruction. First, we introduce a structure-aware scene partitioning strategy that better preserves continuous scene structures during partitioning and reduces cross-region geometric inconsistencies and stitching artifacts through boundary refinement. Second, we develop neighborhood-aware Gaussian organization that extends geometric constraints from individual primitives to their neighborhood organization, encouraging Gaussians to better conform to local surface geometry. Third, we introduce adaptive surface regularization that adjusts the regularization strength according to local geometric characteristics, promoting geometric consistency in structured regions while preserving plausible variations in unstructured regions. Extensive experiments on large-scale aerial photogrammetry benchmarks demonstrate that STARS-GS consistently outperforms the evaluated Gaussian-based methods in surface reconstruction. It increases the average F1-score from 0.640 for the second-best method to 0.698, corresponding to a relative improvement of approximately 9.1\%, demonstrating effective improvements in geometric accuracy and surface completeness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03341v1">PointGT: Simultaneous Geometry and Texture Editing for Point-Based Representations</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Accepted to ECCV 2026. Project page: https://zvict.github.io/pointgt/
    </div>
    <details class="paper-abstract">
      We present PointGT, a point-based 3D representation that enables simultaneous editing of object geometry and appearance. Existing reconstruction and view synthesis techniques produce volumetric 3D representations that are high-quality and photorealistic, but are difficult to edit. In particular, recent efforts to enable texture editing for 3D Gaussian Splatting representations are not compatible with geometry edits and deformations. Our method combines a point-based representation that is well-suited for geometry deformations with a learned UV mapping technique that enables high-resolution texture editing. We show that PointGT enables fine-grained editing of both geometry and texture in point-based neural representations with high rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03334v1">Laplacian Frequency Hierarchies for Efficient 3D Gaussian Splatting Training</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Accepted to Pacific Graphics 2026 (conference track). Project page: https://sorenzhang574.github.io/Laplacian-GS/
    </div>
    <details class="paper-abstract">
      A key bottleneck in 3D Gaussian Splatting training is the continual growth of Gaussian primitives, which increases optimization cost and slows convergence, especially at high resolutions. We propose Laplacian Frequency Hierarchies, a simple yet efficient 3DGS scheme that combines Laplacian image decomposition with coarse-to-fine, frequency-staged training. After fitting lower-frequency structure, we archive the corresponding Gaussian field so that subsequent fields can optimize higher-frequency residuals without carrying the full primitive burden, and we compose the rendered components in the image domain via a Laplacian-style reconstruction at inference time. This design reduces the number of active Gaussians during training, thereby lowering optimization overhead and accelerating training. The proposed scheme is plug-and-play and orthogonal to prior 3DGS accelerations: it can be directly combined with strong backbones such as Taming-3DGS and FastGS to improve training speed with competitive reconstruction quality. It achieves average speedups of 1.73x and 1.21x at 1K setting, and 1.74x and 1.33x at 4K setting on Taming-3DGS and FastGS, with larger gains on more challenging scenes and increasingly pronounced benefits at higher resolutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06632v2">DIAL-GS: Dynamic Instance Aware Reconstruction for Label-free Street Scenes with 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 ICRA2026
    </div>
    <details class="paper-abstract">
      Urban scene reconstruction is critical for autonomous driving, enabling structured 3D representations for data synthesis and closed-loop testing. Supervised approaches rely on costly human annotations and lack scalability, while current self-supervised methods often confuse static and dynamic elements and fail to distinguish individual dynamic objects, limiting fine-grained editing. We propose DIAL-GS, a novel dynamic instance-aware reconstruction method for label-free street scenes with 4D Gaussian Splatting. We first accurately identify dynamic instances by exploiting appearance-position inconsistency between warped rendering and actual observation. Guided by instance-level dynamic perception, we employ instance-aware 4D Gaussians as the unified volumetric representation, realizing dynamic-adaptive and instance-aware reconstruction. Furthermore, we introduce a reciprocal mechanism through which identity and dynamics reinforce each other, enhancing both integrity and consistency. Experiments on urban driving scenarios show that DIAL-GS surpasses existing self-supervised baselines in reconstruction quality and instance-level editing, offering a concise yet powerful solution for urban scene modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03304v1">AnyGS2Mesh: Feed-Forward Mesh Reconstruction from 3D Gaussian Splatting with Arbitrary-Resolution Views</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Existing 3D mesh reconstruction methods from Gaussian scene representations predominantly rely on iterative optimization, resulting in slow inference and limited scalability to high-resolution inputs. In this paper, we present AnyGS2Mesh, the first feed-forward framework for directly reconstructing 3D meshes from 3D Gaussian Splatting representations with support for arbitrary input image resolutions. Our approach incorporates a Gaussian-Guided Transformer architecture that exploits explicit 3D geometric priors for efficient mesh generation. We introduce three key components: (1) a Gaussian-Guided Spatial Reasoning Transformer represents Gaussian primitives as structured 3D tokens and jointly reasons over Gaussian and image features; (2) a Streaming and Patchwise Geometry Encoder processes native-resolution views sequentially and aggregates information across variable-length view sets; (3) a Scale-Aligned Hybrid Depth Refiner uses a PatchFusion-style encoder--decoder to fuse RGB-conditioned predicted depth with Gaussian-rendered metric depth, combining fine local structures with globally consistent metric scale. The refined depth maps are integrated through TSDF fusion, followed by Marching Cubes for deterministic mesh extraction. Extensive experiments show that AnyGS2Mesh achieves state-of-the-art reconstruction quality while significantly reducing inference time compared with optimization-based baselines, enabling near-real-time, high-quality mesh reconstruction. Our results demonstrate the potential of combining Gaussian representations and feed-forward Transformer architectures for scalable 3D geometry reconstruction. The code will be made publicly available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09968v2">ReCoSplat: Online Feed-Forward Gaussian Splatting via Render-and-Compare</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 v2: Corrected OF3GS evaluation results after fixing an implementation bug, added baseline evaluations, updated efficiency benchmarks following a codebase refactor, and added code and pretrained model release links
    </div>
    <details class="paper-abstract">
      Online novel view synthesis requires a model to reconstruct a scene causally from a stream of observations while keeping it renderable at every moment. We present ReCoSplat, an online feed-forward Gaussian Splatting model supporting both posed and unposed inputs, with or without camera intrinsics. While assembling local Gaussians with camera poses scales better than canonical-space prediction, stable training requires ground-truth poses, creating a distribution mismatch when predicted poses are used at inference. To address this, we introduce a Render-and-Compare (ReCo) module. ReCo renders the accumulated scene from the viewpoint of the incoming observation, comparing the render with the observation to produce a stable conditioning signal that helps bridge the mismatch. To support long sequences, we propose a hybrid KV-cache compression strategy combining early-layer truncation with chunk-level selective retention, reducing the KV cache size by over 90% for 100 or more frames. ReCoSplat achieves state-of-the-art performance among online methods while processing 256-view streams at an average input throughput of 45.1 FPS, with an end-of-stream throughput of 41.1 FPS on an RTX 6000 Ada GPU. Code and pretrained models are released at https://freemancheng.com/ReCoSplat .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02747v1">InceptionGS: Generative Bootstrapping for Large-Scale Gaussian Splatting under Unstructured View Sampling</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Achieving truly immersive large-scale scene digitization necessitates consistent and visually pleasing rendering across all possible viewing perspectives. However, collecting multi-view images covering every fine detail of a large-scale scene is prohibitive due to scene complexity, capture cost, negligence, or accessibility constraints. As a result, the sampled views tend to be highly unstructured -- the majority of the scene is well covered yet certain regions inevitably lack sufficient observations. Existing reconstruction based methods are vulnerable to view scarcity while generation based approaches suffer from generalization, controllability, and 3D consistency issues. To address this challenge, we propose InceptionGS, which bootstraps Gaussian splatting by subtly balancing reconstruction and generation. Starting from an initial Gaussian splatting, InceptionGS reasonably rethinks and repairs problematic regions caused by view scarcity while preserving the quality elsewhere, by softly incorporating scene- and view-adaptive generative priors. Extensive experiments on real-world large-scale scenes demonstrate the superiority and broad applicability of our approach in handling unstructured imagery and boosting high-fidelity Gaussian splatting. Please refer to the supplementary video for better visual demonstrations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02543v1">LightBridge: Feed-Forward Generative Relighting for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 14pages, 8figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves high-quality, real-time novel view synthesis, but the resulting assets have baked-in illumination and cannot be easily relit. Inverse rendering methods optimize simplified reflectance and illumination models for each scene, limiting efficiency and relighting quality. Recent generative approaches leverage large diffusion models for realistic lighting edits, but applying them to 3DGS typically requires an additional per-scene optimization stage to bake the edited appearance into the representation. We present LightBridge, a feed-forward generative framework for controllable relighting of complete 3DGS assets in a single pass. To enable feed-forward training, we construct a large-scale Multi-Illumination Relighting Dataset with paired source and target observations of the same scenes. Latent Bridge Relighting Diffusion models relighting as source-to-target transport in latent space, enabling one-step extraction of 2D visual tokens without iterative diffusion sampling. A Gaussian Propagation Transformer uses a point transformer with sparse image-to-point self-attention followed by point-to-image cross-attention to efficiently propagate these cues across the complete 3DGS, while avoiding full attention over all image and Gaussian tokens. Experiments validate these designs, demonstrating competitive relighting quality and efficient single-pass prediction of complete relit 3DGS assets without scene-specific optimization. The code and dataset will be made publicly available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02352v1">Atlas: Algorithm-Hardware Co-Design for On-Device City-Scale 3D Gaussian Splatting in VR</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has drawn significant attention in the architectural community recently. However, enabling city scale 3DGS on mobile VR devices remains challenging, as the memory requirement of large scale scenes far exceeds the memory capacity of today's mobile GPUs. This paper presents Atlas, an on device city scale 3DGS rendering framework that enables scalable rendering without runtime Internet access. The key insight is that although the full 3DGS model is massive, each frame only requires a small subset of Gaussians under the current pose and level of detail requirement. Based on this insight, Atlas introduces a hierarchical memory offloading mechanism that dynamically loads only necessary Gaussian data into device memory. To further improve performance, Atlas proposes temporal aware LoD search and stereo rasterization to avoid redundant computation in VR. We further show that our technique can be integrated with existing 3DGS accelerators with negligible hardware overhead. Overall, Atlas achieves 18.5x speedup over the GPU baseline and 3.9x speedup over the state of the art 3DGS accelerators, with 92.4% energy savings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02184v1">CC-4DGS: Computational Deformation and Point-Cloud Compression for Storage-Efficient Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 16 pages, 8 figures, and 9 tables. Published in IEEE Transactions on Visualization and Computer Graphics. Code is available at https://github.com/KyungdaePark/CC-4DGS
    </div>
    <details class="paper-abstract">
      Dynamic four-dimensional (4D) Gaussian Splatting has emerged as a powerful explicit representation for high-quality view synthesis, yet existing methods still require tens to hundreds of megabytes per scene due to their heavy reliance on large multi-resolution hash tables and high-dimensional Gaussian attributes. This paper presents CC-4DGS, a storage-efficient and scalable framework that rethinks both deformation modeling and canonical attribute storage. First, we introduce a computational deformation field (CDF) that replaces large multi-resolution learnable hash tables with deterministic dense hash encoding and compact neural decoders, enabling on-the-fly synthesis of deformation features while reducing deformation storage to only 1--3 MB per scene. Second, we propose a compression of canonical point-cloud attributes (CCA) pipeline that compresses high-dimensional spherical harmonic appearance terms and auxiliary Gaussian attributes via conditional autoencoding, selective quantization, and residual codebooks, achieving 3--5$\times$ point-cloud reduction with negligible quality loss. Together, these components yield a unified representation that preserves real-time rendering performance while reducing total storage to 20--30 MB. Extensive experiments across the N3DV and Technicolor Light Field datasets demonstrate that CC-4DGS achieves reconstruction accuracy comparable to state-of-the-art methods such as Swift4D, while offering significantly improved storage efficiency and favorable runtime-memory trade-offs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00259v2">PointGauss: Point Cloud-Guided Multi-Object Segmentation for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 12 pages, 6 figures
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has established new standards for high-fidelity 3D scene modeling, interpreting massive, unstructured Gaussian primitives into meaningful geospatial entities remains a critical challenge for remote sensing and urban applications. Existing 2D-to-3D distillation methods suffer from projection ambiguities, geometric inconsistencies, and computational bottlenecks when applied to complex, large-scale topographies. Furthermore, current benchmarks lack natively aligned 2D-3D ground truth tailored for splatting-based representations. To address these limitations, we propose \textit{PointGauss}, a unified 3D-native framework for robust semantic parsing and instance segmentation. By treating Gaussian primitives as unstructured point sets, PointGauss leverages Point Transformer V3 (PTv3) to extract scale-invariant geometric features directly from Gaussian attributes, effectively resolving projection ambiguities. To ensure scalability in building-scale scenes, we introduce an adaptive region-of-interest cropping strategy and an instance-aware distance-constrained rasterization pipeline for pixel-level, view-consistent projection. Additionally, we present \textbf{SplatSeg-360}, the first rigorous cross-scale benchmark tailored for 3DGS, comprising 32 complex scenes with full 360$^\circ$ coverage and over 6,300 natively aligned 2D-3D masks. Extensive experiments demonstrate that PointGauss operates in real-time and achieves state-of-the-art performance. Notably, it attains approximately 90\% 3D-mIoU in large-scale building scenarios and roughly 80\% 2D-mIoU in view-consistent 2D instance segmentation, outperforming baseline methods by 16\%. (\href{https://github.com/hbycswt/pointgauss#}{Code})
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01698v1">VirSqueezer: Generating Realistic Deformations and Squeezing Dynamics in VR from Fine-Grained Squeezing Controls</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Squeezing is one of the most natural forms of hand manipulation, inherently involving fine-grained, temporally evolving, per-finger flexion. In VR content creation, squeezing plays a unique role in enabling particular visual effects such as localized deformations and dynamic behaviors, e.g., bursting a Coke can or juicing a fruit, thereby expanding the expressive possibilities of VR content. However, existing techniques, such as 3D Gaussian splatting-based methods and diffusion-based video generation models, are limited in their ability to simulate fine-grained virtual squeezing effects. We introduce VirSqueezer, a framework designed to generate both localized deformations (primary effects) and complex squeezing dynamics, such as rupture and overflow (secondary effects). VirSqueezer captures squeezing control signals using a SenseGlove and provides the user with inferred resistance force feedback during the squeezing process. By estimating object contact areas, inferring physical properties, and simulating physical responses, VirSqueezer computes conditions that guide generation models for visual effect generation, ensuring both visual coherence and temporal synchronization with the simulation. Consequently, VirSqueezer enables the generation of physically realistic visual effects directly from continuous, fine-grained squeezing control signals. Our extensive evaluation demonstrates VirSqueezer's ability to reproduce realistic localized deformations, generate convincing visual dynamics, and maintain consistency in fine-grained squeezing controls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01516v1">DualDiff3D: Dual Structure-Appearance Diffusion Priors for Reliability-Enhanced 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has revolutionized 3D reconstruction and novel-view synthesis, scenarios with limited input views often lead to poor reconstruction quality and artifacts in rendered novel views. Recent efforts attempt to utilize powerful diffusion priors, yet they typically process rendered and reference views concatenated along an additional dimension in a single network. These methods overlook an inherent nature that different views should maintain appearance similarity but differ in structure due to view shifts, leading to blur caused by conflicts between the two properties. In this paper, we propose DualDiff, a novel pipeline that leverages dual diffusion priors with a Structure-Appearance Attention (SAA) module to introduce reference guidance for refining low-quality novel views rendered from flawed 3D representations. Specifically, we retain one diffusion branch to focus on extracting structural information from the low-quality novel views, while introducing another branch to ensure appearance consistency with reference views. Furthermore, we present a 3D reconstruction framework named DualDiff3D, which integrates a reliability-enhanced Render-Refine-Optimize (RRO) loop to progressively and robustly incorporate the refined novel views, yielding more accurate 3DGS. Extensive experiments demonstrate that our approach outperforms state-of-the-art methods even in the inference-only setting, with further performance gains achievable through training. Our code and pre-trained weights are available at https://github.com/Akaneqwq/DualDiff3D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.10475v3">X-SG$^2$S: Safe and Generalizable Gaussian Splatting with X-dimensional Watermarks</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has been widely used in 3D reconstruction and 3D generation. However, the rapid adoption of 3D Gaussian Splatting raises growing concerns about information leakage and unauthorized use, urging the exploration of effective watermarking techniques. However, existing methods are limited by low capacity, fragility under geometric perturbations, and the infeasible requirement for costly fine-tuning or pipeline modifications, motivating the need for a generalizable, feed-forward framework capable of robust multi-modal embedding with minimal intrusion. In this paper, we propose a new framework X-SG$^2$S which can simultaneously inject 1D to 3D watermarks for copyright protection, while keeping the high fidelity of original 3DGS scenes. Specifically, we first split the watermarks into message patches. A self-adaptive gate is developed to select the injection positions of the watermark messages. Then, we use an XD (multi-dimensional) injection head to inject multi-modal messages into sorted 3DGS points. To restore watermarking messages, a learnable gate is developed to recognize the watermarked locations, from which our XD-extraction heads are used to restore hidden messages. X-SG$^2$S is the first framework to unify 1D-to-3D watermarking and enable simultaneous multi-modal watermark embedding in 3DGS, achieving this with minimal rendering interference and zero modifications to parameters or pipelines. Extensive experiments demonstrate that X-SG$^2$S effectively preserves consistency between the watermark and the original 3DGS, exhibits robustness against model degradation, and maintains accurate judgment capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.29048v2">DReSG: Diffusion Residuals for Stylized Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted to Pacific Graphics 2026 (Conference Track)
    </div>
    <details class="paper-abstract">
      Reference-guided stylization of scenes represented by 3D Gaussian Splatting (3DGS) is important for efficient and controllable 3D content creation. Existing VGG-feature-based 3D stylization methods provide stable rendered-view optimization, but often under-represent expressive reference style cues; diffusion models offer stronger image priors, yet direct per-view or score-based diffusion guidance can lead to view drift, local artifacts, and hard-to-control appearance updates. We present DReSG, a 3D-grounded residual-feedback framework for stylized Gaussian splatting. DReSG represents attention-guided diffusion proposals as residual targets relative to the current render, and progressively absorbs these residuals into a shared Gaussian scene through multi-view Gaussian feedback. To make this feedback stable and controllable, DReSG modulates residual strength during target construction and combines coverage-aware view selection with conflict-filtered color updates during multi-view fitting. Extensive experiments demonstrate that DReSG achieves competitive reference-guided stylization while better preserving scene structure and cross-view stability. Our project page is available at https://vpx-ecnu.github.io/DReSG-website/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00994v1">EvoGS: Modeling Deformation Evolution for Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted by Pacific Graphics 2026 (journal track)
    </div>
    <details class="paper-abstract">
      Recent extensions of 3D Gaussian Splatting (3DGS) enable real-time novel view synthesis in dynamic scenes by learning time-conditioned Gaussian deformations. However, existing MLP-based methods typically estimate deformations independently at each timestamp, making them less robust to large or abrupt motions. To address this issue, we propose \textbf{EvoGS}, a 3DGS-based dynamic reconstruction framework that models Gaussian deformation as a temporal evolution process. EvoGS maintains persistent deformation states for each Gaussian, extrapolates future states from historical deformation states, and corrects the predictions with MLP-derived observations. The correction is adaptively weighted using a temporal residual memory and evolution statistics such as deformation velocity and trajectory deviation. To further improve reconstruction quality, EvoGS introduces deformation-aware densification. Clone and split operations are performed along corrected deformation directions, while an uncertainty-aware strategy suppresses densification for Gaussians with unstable deformation histories. Experiments show that EvoGS improves dynamic novel view synthesis quality and achieves competitive performance across benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22228v3">3D-Consistent Multi-View Editing by Correspondence Guidance</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted to LoViF at ECCV 2026
    </div>
    <details class="paper-abstract">
      Recent advancements in diffusion and flow models have greatly improved text-based image editing, yet methods that edit images independently often produce geometrically and photometrically inconsistent results across different views of the same scene. Such inconsistencies are particularly problematic for editing of 3D representations such as NeRFs or Gaussian splat models. We propose a training-free guidance framework that enforces multi-view consistency during the image editing process. The key idea is that corresponding points should look similar after editing. To achieve this, we introduce a consistency loss that guides the denoising process toward coherent edits. The framework is flexible and can be combined with widely varying image editing methods, supporting both dense and sparse multi-view editing setups. Experimental results show that our approach significantly improves 3D consistency compared to existing multi-view editing methods. We also show that this increased consistency enables high-quality Gaussian splat editing with sharp details and strong fidelity to user-specified text prompts. Please refer to our project page for video results: https://3d-consistent-editing.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16103v3">NanoGS: Training-Free Gaussian Splat Simplification</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 ECCV 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splat (3DGS) enables high-fidelity, real-time novel view synthesis by representing scenes with large sets of anisotropic primitives, but often requires millions of Splats, incurring significant storage and transmission costs. Most existing compression methods rely on GPU-intensive post-training optimization with calibrated images, limiting practical deployment. We introduce \textbf{NanoGS}, a training-free and lightweight framework for Gaussian Splat simplification. Instead of relying on image-based rendering supervision, NanoGS formulates simplification as local pairwise merging over a sparse spatial graph. The method approximates a pair of Gaussians with a single primitive using mass preserved moment matching and evaluates merge quality through a principled merge cost between the original mixture and its approximation. By restricting merge candidates to local neighborhoods and selecting compatible pairs efficiently, NanoGS produces compact Gaussian representations while preserving scene structure and appearance. NanoGS operates directly on existing Gaussian Splat models, runs efficiently on CPU, and preserves the standard 3DGS parameterization, enabling seamless integration with existing rendering pipelines. Experiments demonstrate that NanoGS substantially reduces primitive count while maintaining high rendering fidelity, providing an efficient and practical solution for Gaussian Splat simplification. Our project website is available at \href{https://saliteta.github.io/NanoGS/}{https://saliteta.github.io/NanoGS/}.
    </details>
</div>
