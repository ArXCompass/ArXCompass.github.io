# gaussian splatting - 2026_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24316v2">Large-Scale Photogrammetric Documentation of St. John's Co-Cathedral: A Workflow for Cultural Heritage Preservation</a></div>
    <div class="paper-meta">
      📅 2026-05-14
    </div>
    <details class="paper-abstract">
      We present a comprehensive methodology for the large-scale photogrammetric documentation of St. John's Co-Cathedral in Valletta, Malta, a UNESCO World Heritage site renowned for its ornate Baroque architecture and Caravaggio masterpieces. Over seven nights of evening-only data collection, we captured 99,000 images using DSLR cameras, drone photography, and LIDAR scanning to create a highly detailed 3D reconstruction comprising 25-30 billion triangles. This paper documents our complete workflow for cultural heritage preservation, addressing the unique challenges of digitizing complex baroque architectural spaces with highly reflective metallic surfaces, dark materials, intricate tapestries, and restricted access. We detail our pipeline from multi-modal data acquisition through processing, including strategic image grading and AI-assisted denoising to address low-light grain, extensive LIDAR point cloud cleanup, hybrid photogrammetric reconstruction using RealityCapture, and mesh subdivision strategies for real-time visualization engines. Our methodology combines automated workflows with necessary manual intervention to handle the scale and complexity of the project, with particular attention to reflective surface challenges characteristic of baroque heritage sites. We also present preliminary experiments with Gaussian splatting as a complementary representation technique. The resulting digital archive serves multiple preservation purposes including disaster recovery documentation, conservation analysis, virtual tourism, and scholarly research. This work provides a detailed, replicable workflow for heritage professionals undertaking similar large-scale architectural documentation projects, addressing the practical challenges of applying photogrammetric methods in complex real-world heritage scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.14629v1">Efficient Dense Matching for Enhanced Gaussian Splatting Using AV1 Motion Vectors</a></div>
    <div class="paper-meta">
      📅 2026-05-14
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a prominent framework for real-time, photorealistic scene reconstruction, offering significant speed-ups over Neural Radiance Fields (NeRF). However, the fidelity of 3DGS representations remains heavily dependent on the quality of the initial point cloud. While standard Structure-from-Motion (SfM) pipelines using COLMAP provide adequate initialisation, they often suffer from high computational costs and sparsity in textureless regions, which degrades subsequent reconstruction accuracy and convergence speed. In this work, we introduce an AV1-based feature detection and matching pipeline that significantly reduces SfM processing overhead. By leveraging motion vectors inherent to the AV1 video codec, we bypass computationally expensive exhaustive matching while maintaining geometric robustness. Our pipeline produces substantially denser point clouds, with up to eight times as many points as classical SfM. We demonstrate that this enhanced initialisation directly improves 3DGS performance, yielding an 9-point increase in VMAF and a 63% average reduction in training time required to reach baseline quality. The project page: https://sigmedia.tv/AV1-3DGS.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02482v2">G-SHARP: Gaussian Surgical Hardware Accelerated Real-time Pipeline</a></div>
    <div class="paper-meta">
      📅 2026-05-14
    </div>
    <details class="paper-abstract">
      We propose G-SHARP, a commercially compatible, real-time surgical scene reconstruction framework designed for minimally invasive procedures that require fast and accurate 3D modeling of deformable tissue. While recent Gaussian splatting approaches have advanced real-time endoscopic reconstruction, existing implementations often depend on non-commercial derivatives, limiting deployability. G-SHARP overcomes these constraints by being the first surgical pipeline built natively on the GSplat (Apache-2.0) differentiable Gaussian rasterizer, enabling principled deformation modeling, robust occlusion handling, and high-fidelity reconstructions on the EndoNeRF pulling benchmark. Our results demonstrate state-of-the-art reconstruction quality with strong speed-accuracy trade-offs suitable for intra-operative use. Finally, we provide a Holoscan SDK application that deploys G-SHARP on NVIDIA IGX Orin and Thor edge hardware, enabling real-time surgical visualization in practical operating-room settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.14135v1">PanoPlane: Plane-Aware Panoramic Completion for Sparse-View Indoor 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-13
    </div>
    <details class="paper-abstract">
      We present PanoPlane, an approach for high-fidelity sparse-view indoor novel view synthesis that reconstructs closed room geometry via panoramic scene completion. Unlike perspective-based methods that generate training views from limited fields of view, PanoPlane leverages $360^{\circ}$ panoramic completion to condition the generative process on the full spatial layout. We propose Layout Anchored Attention Steering, a training-free mechanism that steers attention within the diffusion model's internal representation toward scene's detected planar surfaces at inference time. By directing each unobserved region's attention toward geometrically consistent observed content, our method replaces unconstrained hallucination with grounded surface extrapolation. The resulting panoramic completions provide supervision for 3D Gaussian Splatting, enabling accurate novel-view synthesis across unobserved regions from as few as three input views. Experiments on Replica, ScanNet++, and Matterport3D demonstrate state-of-the-art novel view synthesis quality across 3, 6, and 9 input views, achieving up to $+17.8\%$ improvement in PSNR over the current state-of-the-art baseline without any training or fine-tuning of the diffusion model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13794v1">BlitzGS: City-Scale Gaussian Splatting at Lightning Speed</a></div>
    <div class="paper-meta">
      📅 2026-05-13
    </div>
    <details class="paper-abstract">
      We present BlitzGS, a distributed 3DGS framework that reduces active Gaussian workload for fast city-scale reconstruction. BlitzGS manages this workload at three coupled levels. At the system level, the framework shards Gaussians across GPUs by index parity rather than spatial blocks. This approach mitigates the cross-block visibility redundancy inherent in spatial partitioning. Furthermore, it distributes each rendering step through a single cross-GPU exchange that routes projected Gaussians to their tile owners. At the model level, scheduled importance-scoring passes shrink the global Gaussian population. During these passes, the framework generates a per-Gaussian visibility weight to bias density-control updates toward contributing primitives and a per-view importance mask for the view-level renderer. At the view level, BlitzGS trims each camera's active set with a distance-based LOD gate to exclude excessively fine primitives for the current frustum and the importance-based culling mask to skip Gaussians with negligible cross-view contribution. On large-scale benchmarks, BlitzGS matches the rendering quality of recent large-scale baselines while delivering an order-of-magnitude speedup, training city-scale scenes in tens of minutes. Our code is available at https: //github.com/AkierRaee/BlitzGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13600v1">Sparse Code Uplifting for Efficient 3D Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-13
      | 💬 18 pages (9 pages main paper), 10 figures, preprint
    </div>
    <details class="paper-abstract">
      3D Language Gaussian Splatting (3DLGS) augments 3D Gaussian Splatting with language-aligned visual features for open-vocabulary 3D scene understanding. A core challenge is efficiently associating high-dimensional vision-language embeddings with millions of 3D Gaussians while preserving efficient feature rendering for text-based querying. Existing methods either store dense features directly on Gaussians, causing high storage costs and slow rendering, or learn compact representations through expensive per-scene optimization with repeated feature rasterization. No existing method simultaneously achieves fast 3D semantic reconstruction, efficient storage, and fast rendering. We propose SCOUP (Sparse COde UPlifting), which addresses all three by decoupling language representation learning from 3D Gaussian optimization. Rather than working directly in 3D, we learn sparse codebook-based representations entirely using features associated with 2D image regions, associating each region with a sparse set of codebook coefficients. We then uplift these coefficients to 3D Gaussians with our weighted sparse aggregation using Gaussian-to-pixel associations, where each Gaussian accumulates coefficients over codebook atoms across views. Top-$K$ filtering then extracts the most dominant multi-view coefficients per Gaussian, enabling efficient storage and fast rendering. Our method achieves up to $400\times$ training speedup while being $3\times$ more memory efficient during training compared to the state-of-the-art in rendering speed. Across multiple benchmarks, SCOUP matches or outperforms existing methods in open-vocabulary querying accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13591v1">Real2Sim: A Physics-driven and Editable Gaussian Splatting Framework for Autonomous Driving Scenes</a></div>
    <div class="paper-meta">
      📅 2026-05-13
    </div>
    <details class="paper-abstract">
      Reliable autonomous driving relies on large-scale, well-labeled data and robust models. However, manual data collection is resource-intensive, and traditional simulation suffers from a persistent reality gap. While recent generative frameworks and radiance-field methods improve visual fidelity, they still struggle with temporal and spatial consistency and cannot ensure physics-aware behavior, limiting their applicability to driving scenario generation. To address these challenges, we propose Real2Sim, an unified framework that combines 4D Gaussian Splatting (4DGS) with a differentiable Material Point Method (MPM) solver. Real2Sim explicitly reconstructs dynamic driving scenes as temporally continuous Gaussian primitives, supports instance-level editing, and simulates realistic object-object and object-environment interactions. This framework enables physics-aware, high-fidelity synthesis of diverse, editable scenarios, including challenging corner cases such as collisions and post-impact trajectories. Experiments on the Waymo Open Dataset validate Real2Sim's capabilities in rendering, reconstruction, editing, and physics simulation, demonstrating its potential as a scalable tool for data generation in downstream tasks such as perception, tracking, trajectory prediction, and end-to-end policy learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21238v3">3D-UIR: 3D Gaussian for Underwater 3D Scene Reconstruction via Physics Based Appearance-Medium Decoupling</a></div>
    <div class="paper-meta">
      📅 2026-05-13
      | 💬 Accepted to IEEE TIP 2026. Project webpage: https://bilityniu.github.io/3D-UIR
    </div>
    <details class="paper-abstract">
      Novel view synthesis for underwater scene reconstruction presents unique challenges due to complex light-media interactions. Optical scattering and absorption in water body bring inhomogeneous medium attenuation interference that disrupts conventional volume rendering assumptions of uniform propagation medium. While 3D Gaussian Splatting (3DGS) offers real-time rendering capabilities, it struggles with underwater inhomogeneous environments where scattering media introduces artifacts and inconsistent appearance. In this study, we propose a physics-based framework that disentangles object appearance from water medium effects through tailored Gaussian modeling. Our approach introduces appearance embeddings, which are explicit medium representations for backscatter and attenuation, enhancing scene consistency. In addition, we propose a depth-guided optimization strategy that leverages pseudo-depth maps as supervision with depth regularization and scale penalty terms to improve geometric fidelity. By integrating the proposed appearance and medium modeling components via an underwater imaging model, our approach achieves both high-quality novel view synthesis and physically accurate scene restoration. Experiments demonstrate our significant improvements in rendering quality and restoration accuracy over existing methods. The project page is available at https://bilityniu.github.io/3D-UIR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13465v1">Z-Order Transformer for Feed-Forward Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-13
      | 💬 Accept by CVPR 2026, Oral
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled significant progress in photorealistic novel view synthesis. However, traditional 3DGS relies on a slow, iterative optimization process, which limits its use in scenarios demanding real-time results. To overcome this bottleneck, recent feed-forward methods aim to predict Gaussian attributes directly from images, but they often struggle with the redundancy of Gaussian primitives and rendering quality. In this work, we introduce a transformer-based architecture specifically designed for feed-forward Gaussian Splatting. Our key insight is that spatial and semantic relationships among Gaussians can be effectively captured through a sparse attention mechanism, enabled by a Z-order strategy that organizes the unstructured Gaussian set into a spatially coherent sequence. Furthermore, we incorporate this Z-order strategy to adaptively suppress redundancy while preserving critical structural details. This allows the transformer to efficiently model context, compress Gaussian primitives, and predict Gaussian attributes in a single forward pass. Comprehensive experiments demonstrate that our method achieves fast and high-quality novel view synthesis with fewer Gaussian primitives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05876v3">3DSS: 3D Surface Splatting for Inverse Rendering</a></div>
    <div class="paper-meta">
      📅 2026-05-13
    </div>
    <details class="paper-abstract">
      We present 3D Surface Splatting (3DSS), the first differentiable surface splatting renderer for physically-based inverse rendering from multi-view images. Our central insight is that the surface separation problem at the heart of surface splatting admits a direct formulation in terms of the reconstruction kernels themselves. From this foundation we derive a coverage-based compositing model whose per-layer opacity arises directly from the accumulated Elliptical Weighted Average reconstruction weight, yielding anti-aliased silhouettes and informative visibility gradients at sparsely covered edges. Combined with forward microfacet shading under co-optimized HDR environment lighting and density-aware adaptive refinement, 3DSS jointly recovers shape, spatially-varying BRDF materials, and illumination. Because the optimized representation is a set of oriented surface samples, it bridges natively to mesh-based workflows via surface reconstruction from oriented point cloud methods. We evaluate 3DSS against mesh-based, implicit, and Gaussian-splatting baselines across geometry reconstruction, novel-view synthesis, and novel-illumination relighting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.09479v2">SkySplat: Generalizable 3D Gaussian Splatting from Multi-Temporal Sparse Satellite Images</a></div>
    <div class="paper-meta">
      📅 2026-05-13
      | 💬 AAAI 2026. Code is available at https://github.com/NanCheng2001/SkySplat-main
    </div>
    <details class="paper-abstract">
      Three-dimensional scene reconstruction from sparse-view satellite images is a long-standing and challenging task. While 3D Gaussian Splatting (3DGS) and its variants have recently attracted attention for its high efficiency, existing methods remain unsuitable for satellite images due to incompatibility with rational polynomial coefficient (RPC) models and limited generalization capability. Recent advances in generalizable 3DGS approaches show potential, but they perform poorly on multi-temporal sparse satellite images due to limited geometric constraints, transient objects, and radiometric inconsistencies. To address these limitations, we propose SkySplat, a novel self-supervised framework that integrates the RPC model into the generalizable 3DGS pipeline, enabling more effective use of sparse geometric cues for improved reconstruction. SkySplat relies only on RGB images and radiometric-robust relative height supervision, thereby eliminating the need for ground-truth height maps. Key components include a Cross-Self Consistency Module (CSCM), which mitigates transient object interference via consistency-based masking, and a multi-view consistency aggregation strategy that refines reconstruction results. Compared to per-scene optimization methods, SkySplat achieves an 86 times speedup over EOGS with higher accuracy. It also outperforms generalizable 3DGS baselines, reducing MAE from 13.18 m to 1.80 m on the DFC19 dataset significantly, and demonstrates strong cross-dataset generalization on the MVS3D benchmark. The is available at https://github.com/NanCheng2001/SkySplat-main
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.12072v2">PairDropGS: Paired Dropout-Induced Consistency Regularization for Sparse-View Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-13
      | 💬 11 pages,8 figures
    </div>
    <details class="paper-abstract">
      Dropout-based sparse-view 3D Gaussian Splatting (3DGS) methods alleviate overfitting by randomly suppressing Gaussian primitives during training. Existing methods mainly focus on designing increasingly sophisticated dropout strategies, while they overlook the resulting inconsistencies among different dropped Gaussian subsets. This oversight often leads to unstable reconstruction and suboptimal Gaussian representation learning.In this paper, we revisit dropout-based sparse-view 3DGS from a consistency regularization perspective and propose PairDropGS, a Paired Dropout-induced Consistency Regularization framework for sparse-view Gaussian splatting. Specifically, PairDropGS first constructs a pair of the dropped Gaussian subsets from a shared Gaussian field and designs a low-frequency consistency regularization to constrain their low-frequency rendered structures. This design encourages the shared Gaussian field to preserve stable scene layout and coarse geometry under different random dropouts, while avoiding excessive constraints on ambiguous high-frequency details. Moreover, we introduce a progressive consistency scheduling strategy to gradually strengthen the consistency regularization during training for stability and robustness of reconstruction. Extensive experiments on widely-used sparse-view benchmarks demonstrate that PairDropGS achieves superior training stability, significantly outperforms existing dropout-based 3DGS methods in reconstruction quality, while exhibiting the simplicity and plug-and-play nature for improving dropout-based optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13093v1">RoSplat: Robust Feed-Forward Pixel-wise Gaussian Splatting for Varying Input Views and High-Resolution Rendering</a></div>
    <div class="paper-meta">
      📅 2026-05-13
    </div>
    <details class="paper-abstract">
      Generalizable 3D Gaussian Splatting has recently emerged as an efficient approach for novel-view synthesis, enabling feed-forward synthesis from only a few input views. However, existing pixel-wise feed-forward methods suffer from over-bright renderings when the number of input views varies during inference, as well as insufficient supervision for accurate Gaussian scale estimation, which leads to hole artifacts, particularly in high-resolution renderings. To address these issues, we identify that the over-brightness is caused by the varying number of overlapping Gaussians and propose a simple alpha normalization strategy to maintain brightness consistency across different number of input views. In addition, we introduce an auxiliary 3D sampling-based regularizer to improve Gaussian scale estimation, thereby mitigating hole artifacts in high-resolution rendering. Experiments on benchmark datasets demonstrate that our method significantly improves baseline models under varying input-view and high-resolution rendering settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2403.11247v3">Compact 3D Gaussian Splatting For Dense Visual SLAM</a></div>
    <div class="paper-meta">
      📅 2026-05-13
      | 💬 Accepted by IJCV 2026
    </div>
    <details class="paper-abstract">
      Recent work has shown that 3D Gaussian-based SLAM enables high-quality reconstruction, accurate pose estimation, and real-time rendering of scenes. However, these approaches are built on a tremendous number of redundant 3D Gaussian ellipsoids, leading to high memory and storage costs, and slow training speed. To address the limitation, we propose a compact 3D Gaussian Splatting SLAM system that reduces the number and the parameter size of Gaussian ellipsoids. A sliding window-based masking strategy is first proposed to reduce the redundant ellipsoids. Then we observe that the covariance matrix (geometry) of most 3D Gaussian ellipsoids are extremely similar, which motivates a novel geometry codebook to compress 3D Gaussian geometric attributes, i.e., the parameters. Robust and accurate pose estimation is achieved by a global bundle adjustment method with reprojection loss. Extensive experiments demonstrate that our method achieves faster training and rendering speed while maintaining the state-of-the-art (SOTA) quality of the scene representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.12919v1">GuardMarkGS: Unified Ownership Tracing and Edit Deterrence for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-13
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is becoming a practical representation for novel view synthesis, but its growing adoption, together with rapid advances in instruction-driven 3DGS editing, also exposes a dual copyright risk: once a 3DGS-based asset is released, it can be used without permission and manipulated through 3D editing. Existing protection methods address only one side of this problem. Watermarking can trace ownership after unauthorized use, but it cannot prevent malicious editing. Adversarial edit-deterrence methods can disrupt editing, but they do not provide evidence of ownership. To the best of our knowledge, we present the first unified protection framework for 3DGS that jointly optimizes ownership tracing and unauthorized editing deterrence. Our framework combines a scene-wide watermarking objective over all Gaussians with an adversarial objective for edit deterrence. The adversarial branch combines latent-anchor separation, denoising-trajectory diversion, and cross-attention diversion to divert the editing trajectory, while an update-saliency-motivated Gaussian selection strategy assigns stronger adversarial updates to mask-selected Gaussians, improving the balance among watermark recovery, edit deterrence, and rendering fidelity. Experiments on scenes from Mip-NeRF 360 and Instruct-NeRF2NeRF demonstrate that the proposed framework achieves a favorable balance among bit accuracy, edit deterrence, and rendering quality. These results suggest that practical copyright protection of 3DGS-based assets can be more effectively addressed by integrating ownership tracing and unauthorized editing deterrence into a single optimization framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04506v2">Ilov3Splat: Instance-Level Open-Vocabulary 3D Scene Understanding in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-13
      | 💬 The International Conference on Pattern Recognition (ICPR) 2026
    </div>
    <details class="paper-abstract">
      We introduce Ilov3Splat, a novel framework for instance-level open-vocabulary 3D scene understanding built on 3D Gaussian Splatting (3D-GS). Most prior work depends on 2D rendering-based matching or point-level semantic association, which undermines cross-view consistency, lacks coherent instance-level reasoning, and limits precision in downstream 3D tasks. To address these limitations, our method jointly optimizes scene geometry and semantic representations by augmenting Gaussian splats with view-consistent feature fields. Specifically, we leverage multi-resolution hash embedding to efficiently encode language-aligned CLIP features, enabling dense and coherent language grounding in 3D space. We further train an instance feature field using contrastive loss over SAM masks, supporting fine-grained object distinction across views. At inference time, CLIP-encoded queries are matched against the learned features, followed by two-stage 3D clustering to retrieve relevant Gaussian groups. This enables our framework to identify arbitrary objects in 3D scenes based on natural language descriptions, without requiring category supervision or manual annotations. Experiments on standard benchmarks demonstrate that Ilov3Splat outperforms prior open-vocabulary 3D-GS methods in both object selection and instance segmentation, offering a flexible and accurate solution for language-driven 3D scene understanding. Project page: https://csiro-robotics.github.io/Ilov3Splat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.12494v1">Revisiting Photometric Ambiguity for Accurate Gaussian-Splatting Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-12
      | 💬 Accepted at ICML 2026. Project page: https://fictionarry.github.io/AmbiSuR-Proj/
    </div>
    <details class="paper-abstract">
      Surface reconstruction with differentiable rendering has achieved impressive performance in recent years, yet the pervasive photometric ambiguities have strictly bottlenecked existing approaches. This paper presents AmbiSuR, a framework that explores an intrinsic solution upon Gaussian Splatting for the photometric ambiguity-robust surface 3D reconstruction with high performance. Starting by revisiting the foundation, our investigation uncovers two built-in primitive-wise ambiguities in representation, while revealing an intrinsic potential for ambiguity self-indication in Gaussian Splatting. Stemming from these, a photometric disambiguation is first introduced, constraining ill-posed geometry solution for definite surface formation. Then, we propose an ambiguity indication module that unleashes the self-indication potential to identify and further guide correcting underconstrained reconstructions. Extensive experiments demonstrate our superior surface reconstructions compared to existing methods across various challenging scenarios, excelling in broad compatibility. Project: https://fictionarry.github.io/AmbiSuR-Proj/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.12437v1">3D Gaussian Splatting for Efficient Retrospective Dynamic Scene Novel View Synthesis with a Standardized Benchmark</a></div>
    <div class="paper-meta">
      📅 2026-05-12
      | 💬 Accepted for publication at CVPR 2026; 4D World Models Workshop. Draft info: 14 pages, 4 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Retrospective novel view synthesis (NVS) of dynamic scenes is fundamental to applications such as sports. Recent dynamic 3D Gaussian Splatting (3DGS) approaches introduce temporally coupled formulations to enforce motion coherence across time. In this paper, we argue that, in a synchronized multi-view (MV) setting typical of sports, the dynamic scene at each time step is already strongly geometrically constrained. We posit that the availability of calibrated, synchronized viewpoints provides sufficient spatial consistency, and therefore, explicit temporal coupling, or complex multi-body constraints seems unnecessary for retrospective NVS. To this end, we propose an approach tailored for synchronized MV dynamic scene. By initializing the SfM-derived point cloud at the start time and propagating optimized Gaussians over time, we show that efficient retrospective NVS can be achieved without imposing a temporal deformation constraint. Complementing our methodological contribution, we introduce a Dynamic MV dataset framework built on Blender for reproducible NeRF and 3DGS research. The framework generates high-quality, synchronized camera rigs and exports training-ready datasets in standard formats, eliminating inconsistencies in coordinate conventions and data pipelines. Using the framework, we construct a dynamic benchmark suite and evaluate representative NeRF and 3DGS approaches under controlled conditions. Together, we show that, under a synchronized MV setup, efficient retrospective dynamic scene NVS can be achieved using 3DGS. At the same time, the dataset-generation framework enables reproducible and principled benchmarking of dynamic NVS methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.12399v1">GeoQuery: Geometry-Query Diffusion for Sparse-View Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-12
      | 💬 Accept to SIGGRAPH 2026 Conference Track
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a prominent paradigm for 3D reconstruction and novel view synthesis. However, it remains vulnerable to severe artifacts when trained under sparse-view constraints. While recent methods attempt to rectify artifacts in rendered views using image diffusion models, they typically rely on multi-view self-attention to retrieve information from reference images. We observe that this mechanism often fails when the rendered novel views output by 3DGS are heavily corrupted: damaged query features lead to erroneous cross-view retrieval, resulting in inconsistent rendering refinement. To address this, we propose GeoQuery, a geometry-guided diffusion framework that integrates generative priors with explicit geometric cues via a novel Geometry-guided Cross-view Attention (GCA) mechanism. First, by leveraging predicted depth maps and camera poses, we construct a geometry-induced correspondence field to sample reference features, forming a geometry-aligned proxy query that replaces the corrupted rendering features. Furthermore, we design a new cross-view feature aggregation pipeline, in which we restrict the cross-view attention to a local window around each proxy query to effectively retrieve useful features while suppressing spurious matches. GeoQuery can be seamlessly integrated into existing diffusion-based pipelines, enabling robust reconstruction even under extreme view sparsity. Extensive experiments on sparse-view novel view synthesis and rendering artifact removal demonstrate the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.12144v1">PoseCompass: Intelligent Synthetic Pose Selection for Visual Localization</a></div>
    <div class="paper-meta">
      📅 2026-05-12
    </div>
    <details class="paper-abstract">
      In visual localization, Absolute Pose Regression (APR) enables real-time 6-DoF camera pose inference from single images, yet critically depends on fine-tuning data quality and coverage. While recent methods leverage 3D Gaussian Splatting (3DGS) for novel view synthesis-based data augmentation, random sampling generates redundant views and noisy samples from poorly reconstructed regions. To mitigate this research gap, we propose PoseCompass, an intelligent pose selection pipeline for 3DGS-based APR. PoseCompass formulates synthetic pose selection and derives a value-based pose ranking mechanism to identify informative poses. The ranking integrates three dimensions: Localization Difficulty, favoring challenging regions; Coverage Novelty, exploring under-sampled areas; and Rendering Observability, filtering artifacts and noise. PoseCompass then generates trajectory-constrained candidates, selects the top-K ranked poses, and synthesizes views using 3DGS with lightweight diffusion-based alignment. Finally, the pose regressor is fine-tuned on mixed real and synthetic data. We evaluate PoseCompass on 7-Scenes, where it reduces adaptation time from 15.2 to 5.1 minutes, a 3x speedup, while cutting median pose errors by 53.8 percent and significantly outperforming random baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10360v2">DySurface: Consistent 4D Surface Reconstruction via Bridging Explicit Gaussians and Implicit Functions</a></div>
    <div class="paper-meta">
      📅 2026-05-12
    </div>
    <details class="paper-abstract">
      While novel view synthesis (NVS) for dynamic scenes has seen significant progress, reconstructing temporally consistent geometric surfaces remains a challenge. Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) offer powerful dynamic scene rendering capabilities; however, relying solely on photometric optimization often leads to geometric ambiguities. This results in discontinuous surfaces, severe artifacts, and broken surfaces over time. To address these limitations, we present DySurface, a novel framework that bridges the effectiveness of explicit Gaussians with the geometric fidelity of implicit Signed Distance Functions (SDFs) in dynamic scenes. Our approach tackles the structural discrepancy between the forward deformation of 3DGS ($canonical \rightarrow dynamic$) and the backward deformation required for volumetric SDF rendering ($dynamic \rightarrow canonical$). Specifically, we propose the VoxGS-DSDF branch that leverages deformed Gaussians to construct a dynamic sparse voxel grid, providing explicit geometric guidance to the implicit SDF field. This explicit anchoring effectively regularizes the volumetric rendering process, significantly improving surface reconstruction quality, with watertight boundaries and detailed representations. Quantitative and qualitative experiments demonstrate that DySurface significantly outperforms state-of-the-art baselines in geometric accuracy metrics while maintaining competitive rendering performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.11594v1">PointForward: Feedforward Driving Reconstruction through Point-Aligned Representations</a></div>
    <div class="paper-meta">
      📅 2026-05-12
    </div>
    <details class="paper-abstract">
      High-fidelity reconstruction of driving scenes is crucial for autonomous driving. While recent feedforward 3D Gaussian Splatting (3DGS) methods enable fast reconstruction, their per-pixel Gaussian prediction paradigm often suffers from multi-view inconsistency and layering artifacts. Moreover, existing methods often model dynamic instances via dense flow prediction, which lacks explicit cross-view correspondence and instance-level consistency. In this paper, we propose PointForward, a feedforward driving reconstruction framework through point-aligned representations. Unlike pixel-aligned methods, we initialize sparse 3D queries in world space and aggregate multi-view image information via spatial-temporal fusion onto these queries, enforcing explicit cross-view consistency in a single feedforward pass. To handle scene dynamics, we introduce scene graphs that explicitly organize moving instances during reconstruction. By leveraging 3D bounding boxes, our method enables instance-level motion propagation and temporally consistent dynamic representations. Extensive experiments demonstrate that PointForward achieves state-of-the-art performance on large-scale driving benchmarks. The code will be available upon the publication of the paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.11520v1">PointGS: Semantic-Consistent Unsupervised 3D Point Cloud Segmentation with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-12
      | 💬 Accepted by Computer Vision and Pattern Recognition (CVPR) 2026
    </div>
    <details class="paper-abstract">
      Unsupervised point cloud segmentation is critical for embodied artificial intelligence and autonomous driving, as it mitigates the prohibitive cost of dense point-level annotations required by fully supervised methods. While integrating 2D pre-trained models such as the Segment Anything Model (SAM) to supplement semantic information is a natural choice, this approach faces a fundamental mismatch between discrete 3D points and continuous 2D images. This mismatch leads to inevitable projection overlap and complex modality alignment, resulting in compromised semantic consistency across 2D-3D transfer. To address these limitations, this paper proposes PointGS, a simple yet effective pipeline for unsupervised 3D point cloud segmentation. PointGS leverages 3D Gaussian Splatting as a unified intermediate representation to bridge the discrete-continuous domain gap. Input sparse point clouds are first reconstructed into dense 3D Gaussian spaces via multi-view observations, filling spatial gaps and encoding occlusion relationships to eliminate projection-induced semantic conflation. Multi-view dense images are rendered from the Gaussian space, with 2D semantic masks extracted via SAM, and semantics are distilled to 3D Gaussian primitives through contrastive learning to ensure consistent semantic assignments across different views. The Gaussian space is aligned with the original point cloud via two-step registration, and point semantics are assigned through nearest-neighbor search on labeled Gaussians. Experiments demonstrate that PointGS outperforms state-of-the-art unsupervised methods, achieving +0.9% mIoU on ScanNet-V2 and +2.8% mIoU on S3DIS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.11489v1">3DGS$^3$: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering</a></div>
    <div class="paper-meta">
      📅 2026-05-12
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables high-quality real-time 3D rendering but faces challenges in efficiently scaling to ultra-dense scenes and high-resolution due to computational bottlenecks that limit its use in latency-sensitive applications. Instead of optimizing the splatting pipeline itself, we propose \textbf{3DGS$^3$}, a unified post-rendering framework that jointly performs super sampling and frame interpolation through differentiable processing of low-resolution outputs to achieve both high-resolution and high-frame-rate rendering. Our \textbf{Gradient\- \-Aware Super Sampling (GASS)} module leverages the continuous differentiability of 3DGS to extract image gradients that guide a GRU-based refinement network to enable high-fidelity super sampling. Furthermore, a \textbf{Lightweight Temporal Frame Interpolation (LTFI)} module based on a compact U-Net-like backbone fuses temporal and differentiable spatial cues from consecutive frames to synthesize temporally coherent intermediate frames. Experiments on public datasets demonstrate that 3DGS$^3$ achieves superior rendering efficiency and visual quality when compared with state-of-the-art methods and remains compatible with existing 3DGS acceleration techniques. The code will be publicly released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14199v2">Learnable Multi-level Discrete Wavelet Transforms for 3D Gaussian Splatting Frequency Modulation</a></div>
    <div class="paper-meta">
      📅 2026-05-12
      | 💬 Accepted to EUSIPCO 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful approach for novel view synthesis. However, the number of Gaussian primitives often grows substantially during training as finer scene details are reconstructed, leading to increased memory and storage costs. Recent coarse-to-fine strategies regulate Gaussian growth by modulating the frequency content of the ground-truth images. In particular, AutoOpti3DGS employs the learnable Discrete Wavelet Transform (DWT) to enable data-adaptive frequency modulation. Nevertheless, its modulation depth is limited by the 1-level DWT, and jointly optimizing wavelet regularization with 3D reconstruction introduces gradient competition that promotes excessive Gaussian densification. In this paper, we propose a multi-level DWT-based frequency modulation framework for 3DGS. By recursively decomposing the low-frequency subband, we construct a deeper curriculum that provides progressively coarser supervision during early training, consistently reducing Gaussian counts. Furthermore, we show that the modulation can be performed using only a single scaling parameter, rather than learning the full 2-tap high-pass filter. Experimental results on standard benchmarks demonstrate that our method further reduces Gaussian counts while maintaining competitive rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.11427v1">PD-4DGS:Progressive Decomposition of 4D Gaussian Splatting for Bandwidth-Adaptive Dynamic Scene Streaming</a></div>
    <div class="paper-meta">
      📅 2026-05-12
    </div>
    <details class="paper-abstract">
      4D Gaussian Splatting (4DGS) enables high-quality dynamic novel view synthesis, yet current models remain monolithic bitstreams that clients must download in full before any frame can be rendered, causing black-screen waits of tens to hundreds of seconds on mobile bandwidth and leaving 4DGS incompatible with modern adaptive-bitrate delivery. Progressive 3DGS compression alleviates this for static scenes, but it acts only on spatial anchors and cannot partition the temporal deformation networks that dominate dynamic-scene size. We present PD-4DGS, the first framework for progressive compression and on-demand transmission of 4DGS. Hierarchical Deformation Decomposition (HDD) externalises the coarse-to-fine motion hierarchy already latent in 4DGS into three independently transmittable layers -- a static scaffold, a global deformation, and a local refinement -- so that any prefix of the bitstream is already renderable, turning a single training run into a scalable, DASH/HLS-compatible bitstream. A Gaussian-entropy attribute rate-distortion loss together with a temporal mask consistency regulariser shrink the base layer while suppressing low-bitrate flicker; a capacity-weighted rollout schedule, gated online by a learnt activation rate rho, then prevents deformation-network under-training without any per-scene hyperparameter. On the Dycheck iPhone benchmark, PD-4DGS cuts the streamed bitstream by >60% at matched rendering fidelity and reduces first-frame latency from 73--930 s to ~1.7 s on a 2 Mbps link, uniquely enabling true on-demand progressive streaming for 4DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.11424v1">VidSplat: Gaussian Splatting Reconstruction with Geometry-Guided Video Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2026-05-12
      | 💬 Accepted by SIGGRAPH Conference 2026. Project Page: https://tangjm24.github.io/VidSplat
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has achieved remarkable progress in multi-view surface reconstruction, yet it exhibits notable degradation when only few views are available. Although recent efforts alleviate this issue by enhancing multi-view consistency to produce plausible surfaces, they struggle to infer unseen, occluded, or weakly constrained regions beyond the input coverage. To address this limitation, we present VidSplat, a training-free generative reconstruction framework that leverages powerful video diffusion priors to iteratively synthesize novel views that compensate for missing input coverage, and thereby recover complete 3D scenes from sparse inputs. Specifically, we tackle two key challenges that enable the effective integration of generation and reconstruction. First, for 3D consistent generation, we elaborate a training-free, stage-wise denoising strategy that adaptively guides the denoising direction toward the underlying geometry using the rendered RGB and mask images. Second, to enhance the reconstruction, we develop an iterative mechanism that samples camera trajectories, explores unobserved regions, synthesizes novel views, and supplements training through confidence weighted refinement. VidSplat performs robustly to sparse input and even a single image. Extensive experiments on widely used benchmarks demonstrate our superior performance in sparse-view scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.11266v1">PG-3DGS: Optimizing 3D Gaussian Splatting to Satisfy Physics Objectives</a></div>
    <div class="paper-meta">
      📅 2026-05-11
      | 💬 Submitted to Artificial Intelligence. 52 pages
    </div>
    <details class="paper-abstract">
      Recent advances in Gaussian Splatting have enabled fast, high-fidelity 3D scene generation, yet these methods remain purely visual and lack an understanding of how shapes behave in the physical world. We introduce Physics-Guided 3D Gaussian Splatting (PG-3DGS), a framework that couples differentiable physics simulation with 3D Gaussian representations to generate 3D structures satisfying physics functionalities. By allowing physical objectives to guide the shape optimization process alongside visual losses, our approach produces geometries that are not only photometrically accurate but also physically functional. The model learns to adjust shapes so that the generated objects exhibit physically meaningful behaviors, for example, teapots that can pour and airplanes that can generate lift, without sacrificing visual quality. Experiments on pouring and aerodynamic lift tasks show that PG-3DGS improves physical functionality while preserving visual quality. In addition to simulation gains, bench-top physical lift tests with 3D-printed aircraft (Cessna, B-2 Spirit, and paper plane) under identical airflow conditions show higher scale-measured lift for PG-3DGS, generated structures than an appearance-matching baseline in all three cases. Our unified framework connects appearance-based reconstruction with physics-based reasoning, enabling end-to-end generation of 3D structures that both look realistic and function correctly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.11144v1">Forecast-aware Gaussian Splatting for Predictive 3D Representation in Language-Guided Pick-and-Place Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-05-11
    </div>
    <details class="paper-abstract">
      We introduce Forecast-aware Gaussian Splatting (Forecast-GS), a predictive 3D representation framework for language-conditioned robotic manipulation. While recent manipulation systems have made progress by grounding language instructions into robot affordances, value maps, or relational keypoint constraints, they usually reason over the current scene and do not explicitly model the task-completed state. This limitation is critical when success depends on satisfying spatial and semantic goals under partial observations, where the robot must evaluate whether a candidate action leads to a feasible task-consistent outcome. We validate Forecast-GS on real-world pick-and-place manipulation tasks, including Cutter-to-Box, Apple-to-Bowl, and Sponge-to-Tray. For each task, we conduct 25 real-world trials under varied initial object configurations using the same robot platform and sensing setup. Forecast-GS with automatic candidate selection achieves success rates of 21/25, 23/25, and 16/25 on the three tasks, respectively, outperforming the ReKep baseline, which achieves 15/25, 19/25, and 10/25. A diagnostic human-assisted setting further improves success rates to 23/25, 24/25, and 19/25, suggesting that candidate generation is effective while automatic ranking remains imperfect. These results suggest that explicitly forecasting task-completed 3D states enables more reliable action evaluation, while the gap between automatic and human-assisted selection indicates that robust final-state ranking remains an important challenge for fully autonomous manipulation. Overall, Forecast-GS provides an interpretable bridge between language understanding, 3D perception, and robotic manipulation planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10760v1">MAGS-SLAM: Monocular Multi-Agent Gaussian Splatting SLAM for Geometrically and Photometrically Consistent Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-11
    </div>
    <details class="paper-abstract">
      Collaborative photorealistic 3D reconstruction from multiple agents enables rapid large-scale scene capture for virtual production and cooperative multi-robot exploration. While recent 3D Gaussian Splatting (3DGS) SLAM algorithms can generate high-fidelity real-time mapping, most of the existing multi-agent Gaussian SLAM methods still rely on RGB-D sensors to obtain metric depth and simplify cross-agent alignment, which limits the deployment on lightweight, low-cost, or power-constrained robotic platforms. To address this challenge, we propose MAGS-SLAM, the first RGB-only multi-agent 3DGS SLAM framework for collaborative scene reconstruction. Each agent independently builds local monocular Gaussian submaps and transmits compact submap summaries rather than raw observations or dense maps. To facilitate robust collaboration in the presence of monocular scale ambiguity, our framework integrates compact submap communication, geometry- and appearance-aware loop verification, and occupancy-aware Gaussian fusion, enabling coherent global reconstruction without active depth sensors. We further introduce ReplicaMultiagent Plus benchmark for evaluating collaborative Gaussian SLAM. Intensive experiments on synthetic and real-world datasets show that MAGS-SLAM achieves competitive tracking accuracy and comparable or superior rendering quality to state-of-the-art RGB-D collaborative Gaussian SLAM methods while relying only RGB images.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10715v1">UAV-Assisted Scan-to-Simulation for Landslides Using Physics-Informed Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-11
    </div>
    <details class="paper-abstract">
      Landslide monitoring and simulation play an important role in urban safety assessment and disaster prevention. Existing landslide simulation pipelines typically rely on digital elevation model and mesh-based representations, which are suitable for geometric analysis, but often lack visual realism. This limitation reduces their effectiveness in interactive applications, hazard communication, and public education. In this paper, we propose a UAV-based scan-to-simulation framework that bridges photorealistic scene capture and physics-based landslide simulation through 3DGS. Specifically, our pipeline includes four stages: (1) UAV-based acquisition of slope imagery, (2) reconstruction of a low-anisotropy 3DGS scene representation, (3) volumetric conversion of the target simulation region by filling the interior of the surface-based model, and (4) integration with the Material Point Method (MPM) for landslide simulation. We validate the proposed framework on a real landslide site in Hong Kong that experienced a severe landslide event. The results show that our method supports both realistic visual reconstruction and effective simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10705v1">TransmissiveGS: Residual-Guided Disentangled Gaussian Splatting for Transmissive Scene Reconstruction and Rendering</a></div>
    <div class="paper-meta">
      📅 2026-05-11
    </div>
    <details class="paper-abstract">
      Transmissive scenes are ubiquitous in daily life, yet reconstructing and rendering them remains highly challenging due to the inherent entanglement between near-field reflections from the surrounding environment on the transmissive surface, and the transmitted content of the scene behind it. This coupling gives rise to dual surface geometries and dual radiance components within each observation, posing ambiguities for standard methods. We present TransmissiveGS, a novel framework for disentangled reconstruction and rendering of transmissive scenes. Specifically, we model the scene with a dual-Gaussian representation and introduce a deferred shading function to jointly render the two Gaussian components. To separate reflection and transmission, we exploit the inherent multi-view inconsistency of reflections and leverage the residuals from reconstructing multi-view consistent content as cues for disentangled geometry and appearance modeling. We further propose a reflection light field that enables high-fidelity estimation of near-field reflections. During training, we introduce a high-frequency regularization to preserve fine details. We also contribute a new synthetic dataset for evaluating transmissive surface reconstruction. Experiments on both synthetic and real-world scenes demonstrate that TransmissiveGS consistently outperforms prior Gaussian Splatting-based methods in both reconstruction and rendering quality for transmissive scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11969v2">AstroSplat: Physics-Based Gaussian Splatting for Rendering and Reconstruction of Small Celestial Bodies</a></div>
    <div class="paper-meta">
      📅 2026-05-11
      | 💬 10 pages, 6 figures, conference
    </div>
    <details class="paper-abstract">
      Image-based surface reconstruction and characterization are crucial for missions to small celestial bodies (e.g., asteroids), as it informs mission planning, navigation, and scientific analysis. Recent advances in Gaussian splatting enable high-fidelity neural scene representations but typically rely on a spherical harmonic intensity parameterization that is strictly appearance-based and does not explicitly model material properties or light-surface interactions. We introduce AstroSplat, a physics-based Gaussian splatting framework that integrates planetary reflectance models to improve the autonomous reconstruction and photometric characterization of small-body surfaces from in-situ imagery. The proposed framework is validated on real imagery taken by NASA's Dawn mission, where we demonstrate superior rendering performance and surface reconstruction accuracy compared to the typical spherical harmonic parameterization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10485v1">VEGA: Visual Encoder Grounding Alignment for Spatially-Aware Vision-Language-Action Models</a></div>
    <div class="paper-meta">
      📅 2026-05-11
    </div>
    <details class="paper-abstract">
      Precise spatial reasoning is fundamental to robotic manipulation, yet the visual backbones of current vision-language-action (VLA) models are predominantly pretrained on 2D image data without explicit 3D geometric supervision, resulting in representations that lack accurate spatial awareness. Existing implicit spatial grounding methods partially address this by aligning VLA features with those of 3D-aware foundation models, but they rely on empirical layer search and perform alignment on LLM-level visual tokens where spatial structure has already been entangled with linguistic semantics, limiting both generalizability and geometric interpretability. We propose VEGA (Visual Encoder Grounding Alignment), a simple yet effective framework that directly aligns the output of the VLA's visual encoder with spatially-aware features from DINOv2-FiT3D, a DINOv2 model fine-tuned with multi-view consistent 3D Gaussian Splatting supervision. By performing alignment at the visual encoder output level, VEGA grounds spatial awareness before any linguistic entanglement occurs, offering a more interpretable and principled alignment target. The alignment is implemented via a lightweight projector trained with a cosine similarity loss alongside the standard action prediction objective, and is discarded at inference time, introducing no additional computational overhead. Extensive experiments on simulation benchmark and real-world manipulation tasks demonstrate that VEGA consistently outperforms existing implicit spatial grounding baselines, establishing a new state-of-the-art among implicit spatial grounding methods for VLA models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10307v1">PaMoSplat: Part-Aware Motion-Guided Gaussian Splatting for Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-11
      | 💬 Accepted by TCSVT. Project Url: https://pamosplat.github.io
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction represents a fundamental yet demanding challenge in computer vision and robotics. While recent progress in 3DGS-based methods has advanced dynamic scene modeling, obtaining high-fidelity rendering and accurate tracking in scenarios with substantial, intricate motions remains significantly challenging. To address these challenges, we propose PaMoSplat, a novel dynamic Gaussian splatting framework incorporating part awareness and motion priors. Our approach is grounded in two key observations: 1) Parts serve as primitives for scene deformation, and 2) Motion cues from optical flow can effectively guide part motion. Specifically, PaMoSplat initializes by lifting multi-view segmentation masks into 3D space via graph clustering, establishing coherent Gaussian parts. For subsequent timestamps, we leverage a differential evolutionary algorithm to estimate the rigid motion of these parts using multi-view optical flow cues, providing a robust warm-start for further optimization. Additionally, PaMoSplat introduces an adaptive iteration count mechanism, internal learnable rigidity, and flow-supervised rendering loss to accelerate and optimize the training process. Comprehensive evaluations across diverse scenes, including real-world environments, demonstrate that PaMoSplat delivers superior rendering quality, improved tracking precision, and faster convergence compared to existing methods. Furthermore, it enables multiple part-level downstream applications, such as 4D scene editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.00408v2">Beyond Heuristics: Learnable Density Control for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-11
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has demonstrated impressive real-time rendering performance, its efficacy remains constrained by a reliance on heuristic density control. Despite numerous refinements to these handcrafted rules, such methods inherently lack the flexibility to adapt to diverse scenes with complex geometries. In this paper, we propose a paradigm shift for density control from rigid heuristics to fully learnable policies. Specifically, we introduce \textbf{LeGS}, a framework that reformulates density control as a parameterized policy network optimized via Reinforcement Learning (RL). Central to our approach is the tailored effective reward function grounded in sensitivity analysis, which precisely quantifies the marginal contribution of individual Gaussians to reconstruction quality. To maintain computational tractability, we derive a closed-form solution that reduces the complexity of reward calculation from $O(N^2)$ to $O(N)$. Extensive experiments on the Mip-NeRF 360, Tanks \& Temples, and Deep Blending datasets demonstrate that \textbf{LeGS} significantly outperforms state-of-the-art methods, striking a superior balance between reconstruction quality and efficiency. The code will be released at https://github.com/AaronNZH/LeGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00560v2">4D Neural Voxel Splatting: Dynamic Scene Rendering with Voxelized Guassian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-11
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Although 3D Gaussian Splatting (3D-GS) achieves efficient rendering for novel view synthesis, extending it to dynamic scenes still results in substantial memory overhead from replicating Gaussians across frames. To address this challenge, we propose 4D Neural Voxel Splatting (4D-NVS), which combines voxel-based representations with neural Gaussian splatting for efficient dynamic scene modeling. Instead of generating separate Gaussian sets per timestamp, our method employs a compact set of neural voxels with learned deformation fields to model temporal dynamics. The design greatly reduces memory consumption and accelerates training while preserving high image quality. We further introduce a novel view refinement stage that selectively improves challenging viewpoints through targeted optimization, maintaining global efficiency while enhancing rendering quality for difficult viewing angles. Experiments demonstrate that our method outperforms state-of-the-art approaches with significant memory reduction and faster training, enabling real-time rendering with superior visual fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09956v1">SDTalk: Structured Facial Priors and Dual-Branch Motion Fields for Generalizable Gaussian Talking Head Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-05-11
      | 💬 5 pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      High-quality, real-time talking head synthesis remains a fundamental challenge in computer vision. Existing reconstruction- and rendering-based methods typically rely on identity-specific models, limiting cross-identity generalization. To address this issue, we propose SDTalk, a one-shot 3D Gaussian Splatting (3DGS)-based framework that generalizes to unseen identities without personalized training or fine-tuning. Our framework comprises two modules with a two-stage training strategy. In the first stage, we incorporate structured facial priors into the reconstruction module and separately predict 3DGS parameters for visible and occluded regions, enabling complete head reconstruction from a single image. In the second stage, we introduce a dual-branch motion field to model coarse and fine facial dynamics, improving detail fidelity and lip synchronization. Experiments demonstrate that SDTalk surpasses existing methods in both visual quality and inference efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07203v2">From Pixels to Primitives: Scene Change Detection in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-11
      | 💬 Project Page: https://chumsy0725.github.io/GS-DIFF/
    </div>
    <details class="paper-abstract">
      Scene change detection methods built on Gaussian splatting universally follow a render-then-compare paradigm: the pre-change scene is rendered into 2D and compared against post-change images via pixel or feature residuals. This change detection problem with Gaussian Splatting has been treated as a question about pixels; we treat it as a question about primitives. We provide direct evidence that native primitive attributes alone -- position, anisotropic covariance, and color -- carry sufficient signal for scene change detection. What makes primitive-space comparison hard is the under-constrained nature of Gaussian splatting representation: independent optimizations yield primitive solutions whose count, positions, shapes, and colors differ even where nothing has changed. We address this challenge with anisotropic models of geometric and photometric drift, complemented by a per-primitive observability term that reflects the extent to which each Gaussian is constrained by the camera geometry. Operating directly on primitives gives our method, GD-DIFF, two properties that distinguish it from render-then-compare methods. First, change maps are multi-view consistent by construction, where prior work had to learn this through an additional optimization objective. Second, geometric and appearance changes are scored separately, identifying not just where but what kind of change occurred, distinguishing structural changes (e.g., an added object) from surface-level ones (e.g., a color change) without supervision or external model dependencies. On real-world benchmarks, GS-DIFF surpasses the prior state-of-the-art approach by $\sim$17% in mean Intersection over Union.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09688v1">ConFixGS: Learning to Fix Feedforward 3D Gaussian Splatting with Confidence-Aware Diffusion Priors in Driving Scenes</a></div>
    <div class="paper-meta">
      📅 2026-05-10
      | 💬 28 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Feedforward 3D Gaussian Splatting (3DGS) often struggles in trajectory-based sparse-view driving scenes. Existing Gaussian repair methods mainly target optimization-based 3DGS, while diffusion-based repair is typically restricted to iterative refinement near observed viewpoints, leaving feedforward 3DGS repair underexplored. We propose ConFixGS, a plug-and-play method that learns to fix feedforward 3DGS with confidence-aware diffusion priors. Starting from a pretrained feedforward model, ConFixGS generates diffusion-enhanced local pseudo-targets and validates them through reprojection-based cross-checking against support views. The resulting dense confidence maps guide refinement, enhancing reliable details while suppressing hallucinated or inconsistent evidence. On Waymo, nuScenes, and KITTI, ConFixGS improves challenging novel view synthesis, with PSNR gains of up to 3.68 dB and FID reduced by nearly half. Our results highlight confidence-aware fusion of generative priors and support-view consistency as a key principle for robust feedforward 3D driving scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09662v1">BEA-GS: BEyond RAdiance Supervision in 3DGS for Precise Object Extraction</a></div>
    <div class="paper-meta">
      📅 2026-05-10
      | 💬 CVPR 2026 Highlight
    </div>
    <details class="paper-abstract">
      Most Gaussian Splatting techniques that provide a 3D semantic representation of the scene do not optimize the underlying 3D geometry, making object-level editing or asset extraction challenging. Recent methods, such as COBGS, Trace3D, ObjectGS, acknowledge this limitation and propose approaches that modify the scene's geometry to represent the underlying semantics. We advance this concept further by proposing a novel solution that provides near perfect boundaries in object extraction. We do so by introducing two new losses in the optimization that take care of: 1) a loss that modifies the geometry of visible Gaussians to respect semantic boundaries, and 2) a loss that adjusts the geometry of non-visible Gaussians that appear once the object is extracted. Our first loss propagates gradients directly through the rasterization, allowing for seamless integration within the optimization of the Gaussian parameters. The second loss also propagates gradients to Gaussian parameters but does so without passing through the rasterization, enabling modification of the scene's geometry even when little transmittance reaches a Gaussian (partial or non-visible). Exhaustive comparisons with 12 state of the art methods across 4 datasets, using six metrics, demonstrate that our approach produces overall the best boundary segmentation to date.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09362v1">FrameTwin: Curve-Anchored Gaussian Alignment from Sparse Views for Adaptive Wireframe 3D Printing</a></div>
    <div class="paper-meta">
      📅 2026-05-10
    </div>
    <details class="paper-abstract">
      We present FrameTwin, a curve-anchored Gaussian alignment framework that uses sparse-view images to close the control loop for adaptive wireframe 3D printing. Our key idea is to capture the deformation of thin wireframe structures from sparse-view images using Gaussian kernels anchored to parametric curves, yielding a compact and geometry-aware encoding that explicitly captures strut topology. Driven by a differentiable rendering pipeline, FrameTwin estimates a neural deformation field that aligns the partially printed target model with the deformed structure observed during fabrication, where the optimized curve-Gaussian representation serves as a digital twin of the evolving wireframe. Unlike general Gaussian-splatting approaches, our formulation constrains kernel placement along parametric curves, substantially reducing the ambiguity inherent in sparse-view observations of thin structures. The resultant deformation-field alignment enforces global consistency across all struts. By using the estimated deformation field to blend the distorted printed geometry with the remaining unprinted geometry, FrameTwin enables adaptive updates to future printing trajectories. We demonstrate that FrameTwin can robustly capture and compensate for deformation in wireframe models fabricated using a robotized 3D printing system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09299v1">LagrangianSplats: Divergence-Free Transport of Gaussian Primitives for Fluid Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-10
    </div>
    <details class="paper-abstract">
      Reconstructing 3D fluid velocity fields from sparse 2D video observations is a highly ill-posed inverse problem, demanding both transport consistency with observed motion and physical validity under fluid laws. Existing methods typically impose these constraints through soft penalties, often leading to compromised accuracy and convergence issues. We introduce a reconstruction framework that structurally enforces both constraints. Specifically, we parameterize the reconstructed velocity using a continuous Divergence-Free Kernel representation, driving the advection of a Lagrangian 3D Gaussian Splatting representation. This formulation intrinsically guarantees both flow incompressibility and long-range transport coherence by construction. To enable the efficient optimization of such a constrained system, we introduce a novel Sliding Window scheme that propagates gradients over meaningful temporal horizons while maintaining tractable training costs. Experiments on synthetic and real-world datasets demonstrate that our method outperforms state-of-the-art baselines in both transport consistency and physical accuracy, enabling applications such as high-quality re-simulation and flow analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09279v1">CAGS: Color-Adaptive Volumetric Video Streaming with Dynamic 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-10
      | 💬 SIGGRAPH 2026 Conference Paper. Code is available at https://github.com/yindaheng98/ColorAdaptiveGaussianSplatting
    </div>
    <details class="paper-abstract">
      Volumetric video (VV) streaming enables real-time, immersive access to remote 3D environments, powering telepresence, ecological monitoring, and robotic teleoperation. These applications turn VV streaming into a real-time interface to remote physical environments, imposing new system-level demands for photorealistic scene representation, low-latency interaction, and robust performance under heterogeneous networks. 3D Gaussian Splatting (3DGS) has been widely used for real-time photorealistic rendering, offering superior visual quality and rendering performance, but it faces challenges due to bandwidth consumption. Furthermore, as the foundation of adaptive VV streaming, existing Levels of Detail (LoD) methods based on density are not well-suited to Gaussian representations, leading to visible gaps and severe quality degradation. Recent studies have also explored attribute compression techniques to reduce bandwidth consumption. Our preliminary studies reveal that aggressive attribute compression primarily causes color distortion, which can be effectively corrected in the rendered image using a reference image. Motivated by these findings, we propose a novel Color-Adaptive scheme for adaptive VV streaming that uses vector quantization (VQ) to establish LoDs and correct color distortions with low-resolution reference images. We further present CAGS, an adaptive VV streaming system compatible with diverse Gaussian representations, which integrates the Color-Adaptive scheme by rendering reference images on the streaming server and performing color restoration on the client. Extensive experiments on our prototype system demonstrate that CAGS outperforms the existing adaptive streaming systems in PSNR by 5$\sim$20 dB under fluctuating bandwidth, operates significantly faster than existing scalable Gaussian compression methods, and generalizes across different Gaussian representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.08699v1">Thin-Client Interactive Gaussian Adaptive Streaming over HTTP/3</a></div>
    <div class="paper-meta">
      📅 2026-05-09
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3DGS) have enabled photorealistic rendering of complex scenes, yet widespread adoption on mobile and Extended Reality (XR) devices is hindered by substantial computational and bandwidth requirements. While existing solutions often focus on model compression for client-side rendering, they still demand significant GPU power, limiting applicability on resource-constrained hardware. We propose TIGAS (Thin-client Interactive Gaussian Adaptive Streaming), a remote rendering framework offloading rasterization to a backend. To bypass the prohibitive latencies connected to fluctuating network conditions, TIGAS streams view-dependent 2D projections to a lightweight web client over QUIC, minimizing head-of-line (HoL) blocking. A dedicated ABR algorithm adapts rendering quality to fluctuating network conditions, maintaining motion-to-photon latency within strict 6DoF interactive constraints. Furthermore, we discuss the integration of an experimental WebGPU super-resolution pipeline to analyze the trade-offs between perceptual quality enhancements and thin-client processing bottlenecks. We extensively evaluate TIGAS across multi-continental environments using 14 3DGS models and real 6DoF EyeNavGS movement traces. Powered by a backend rendering frames in under 10 milliseconds, TIGAS maintains latency within interactive thresholds while achieving an average SSIM of 0.88, serving both as a robust testbed for 3DGS streaming research and a capable delivery system. The source code is available at: https://github.com/Rekenar/GaussianAdaptiveStreamer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09024v1">Relightable Gaussian Splatting for Virtual Production Using Image-Based Illumination</a></div>
    <div class="paper-meta">
      📅 2026-05-09
    </div>
    <details class="paper-abstract">
      Virtual production (VP) use LED walls to provide both background imagery and image-based lighting. While this enables on-set compositing, it couples lighting to background and scene appearance, limiting flexibility for downstream editing. In addition, inverse rendering conventionally relies on physically-based rendering to estimates 3D geometry and lighting, using environment maps. However, these maps are typically low-resolution and assume far-field lighting. In VP, with near-field and high-resolution image-based lighting, this can lead to inaccuracies and introduce complexities when editing. Addressing this, we propose a VP-specific framework for 3D reconstruction and relighting using Gaussian Splatting. This uses the known background imagery to condition the relighting process. This avoids relying on environment maps and reduces compositing to a background-image editing task. To realize our framework, we introduce a process (and associated dataset) that captures real VP scenes under varying background content and illumination conditions. This data is used to decompose a 3D scene into fixed appearance and variable lighting components. The variable lighting process simulates light transport by parameterizing each primitive with a UV coordinate, intensity value and resolution modifier. Using mipmaps, these directly sample the background texture in image space - implicitly capturing reflections and refractions without physically-based rendering. Combined with the fixed appearance component, this allows us to render relit scenes using a Gaussian Splatting rasterizer. Compared to baselines, our approach achieves higher-quality 3D reconstruction and controllable relighting. The method is efficient (<3 GB RAM, <5 GB VRAM, <2 hours training, ~35 FPS) and supports rendering useful arbitrary output variables including depth, lighting intensity, lighting color, and unlit renders.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04549v2">Nix and Fix: Targeting 1000x Compression of 3D Gaussian Splatting with Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2026-05-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) revolutionized novel view rendering. Instead of inferring from dense spatial points, as implicit representations do, 3DGS uses sparse Gaussians. This enables real-time performance but increases space requirements, hindering rate-constrained applications. 3DGS compression emerged as a field aimed at alleviating this issue. While impressive progress has been made, at low rates, compression introduces artifacts that degrade visual quality significantly. We introduce NiFi, a method for extreme 3DGS compression through restoration via artifact-aware, diffusion-based one-step distillation. We show that our method achieves state-of-the-art perceptual quality at extremely low rates, down to 0.1 MB, and towards 1000x rate improvement over 3DGS at comparable perceptual performance. Code is available at: https://github.com/ceteke/nifi
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12592v2">ELoG-GS: Dual-Branch Gaussian Splatting with Luminance-Guided Enhancement for Extreme Low-light 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-09
      | 💬 Our method achieved a ranking of 9 out of 148 participants in Track 1 of the NTIRE 3DRR Challenge, as reported on the official competition website: https://www.codabench.org/competitions/13854/
    </div>
    <details class="paper-abstract">
      This paper presents our approach to the NTIRE 2026 3D Restoration and Reconstruction Challenge (Track 1), which focuses on reconstructing high-quality 3D representations from degraded multi-view inputs. The challenge involves recovering geometrically consistent and photorealistic 3D scenes in extreme low-light environments. To address this task, we propose Extreme Low-light Optimized Gaussian Splatting (ELoG-GS), a robust low-light 3D reconstruction pipeline that integrates learning-based point cloud initialization and luminance-guided color enhancement for stable and photorealistic Gaussian Splatting. Our method incorporates both geometry-aware initialization and photometric adaptation strategies to improve reconstruction fidelity under challenging conditions. Extensive experiments on the NTIRE Track 1 benchmark demonstrate that our approach significantly improves reconstruction quality over the baselines, achieving superior visual fidelity and geometric consistency. The proposed method provides a practical solution for robust 3D reconstruction in real-world degraded scenarios. In the final testing phase, our method achieved a PSNR of 18.6626 and an SSIM of 0.6855 on the official platform leaderboard. Code is available at https://github.com/lyh120/FSGS_EAPGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.13547v3">Turbo-GS: Accelerating 3D Gaussian Fitting for High-Quality Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2026-05-09
      | 💬 Accepted to CVPR 2026. Project page: https://ivl.cs.brown.edu/research/turbo-gs
    </div>
    <details class="paper-abstract">
      Novel-view synthesis plays a crucial role in computer vision with applications in 3D reconstruction, mixed reality, and robotics. Recent approaches, such as 3D Gaussian Splatting (3DGS), have emerged as state-of-the-art solutions, offering high-quality novel view synthesis in real time. However, training 3DGS models remains slow, particularly for high-resolution images, often requiring hours to fit a scene with 200 views. In this work, we aim to accelerate the fitting process by reducing computational overhead and improving learning efficiency. Specifically, we introduce a dilated rendering technique that renders only a subset of pixels instead of the full image, significantly reducing computational costs. To enhance learning efficiency, we develop a convergence-aware budget control mechanism that balances the addition of new Gaussians with the optimization of existing ones. Additionally, to improve densification efficiency and prevent gradient vanishing, we incorporate both positional and appearance errors to improve the effectiveness of densification. With these improvements, we achieve fast 4K-resolution fitting while maintaining, or even improving, novel view rendering quality. Extensive experiments demonstrate that our method achieves significantly faster optimization than existing approaches while preserving high rendering fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.08739v1">ReorgGS: Equivalent Distribution Reorganization for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-09
    </div>
    <details class="paper-abstract">
      A converged 3D Gaussian Splatting (3DGS) model may approximate the target scene while remaining poorly parameterized for further optimization. We identify this failure mode as \emph{parameterization degeneration}: high-opacity floaters attenuate gradients to true surfaces through alpha compositing, and redundant overlapping clusters create strongly coupled parameter blocks with nearly collinear Jacobian responses. These effects explain why continued optimization can plateau even when the model still contains removable artifacts. We propose ReorgGS, an equivalent distribution reorganization method for converged 3DGS models. ReorgGS treats the existing Gaussian set as an empirical probability field, resamples centers from it, estimates local anisotropic covariances with kNN, initializes low opacity, and continues optimization with the original 3DGS renderer and loss. Unlike opacity reset, which only rescales opacity on the old overlap graph, ReorgGS rebuilds centers, covariances, and visibility structure, thereby changing the graph itself. Our analysis shows that distributional equivalence is not optimization equivalence. The reorganized model preserves scene support while improving gradient accessibility under alpha compositing and reducing opacity-weighted overlap, thereby weakening local parameter coupling during subsequent optimization. Under the same additional optimization budget, ReorgGS improves fitting quality at a fixed Gaussian count, suppresses persistent floaters, and reduces rendering overhead from redundant overlap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.08713v1">REAP: Reinforcement-Learning End-to-End Autonomous Parking with Gaussian Splatting Simulator for Real2Sim2Real Transfer</a></div>
    <div class="paper-meta">
      📅 2026-05-09
    </div>
    <details class="paper-abstract">
      In recent years, autonomous parking has made significant advances, yet parking tasks still face challenges in extreme scenarios such as mechanical and dead-end parking slots, often resulting in failures. This is mainly due to traditional parking methods adopting a multistage approach, lacking the ability to optimize the parking problem as a whole. End-to-end methods enable joint optimization across perception and planning modules to eliminate the accumulation of errors, enhancing algorithm performance in extreme scenarios. Although several end-to-end parking methods use imitation or reinforcement learning, the former is limited by data cost and distribution coverage, while the latter suffers from inefficient exploration. To address these challenges, we propose a Reinforcement learning End-to-end Autonomous Parking method (REAP). REAP employs Soft Actor-Critic (SAC) within an asymmetric reinforcement learning framework to improve training efficiency and inference performance. To accelerate model convergence, we distill the capabilities of a rule-based planner into the end-to-end network through behavior cloning. We further introduce a soft predictive collision penalty mechanism to reduce collision rates by penalizing obstacle-approaching actions. To ensure that the trained reinforcement learning network can directly transfer to real-world scenarios, we have established a Real2Sim2Real simulator. In the Real2Sim step, we use 3D Gaussian Splatting (3DGS) to transform real-world scenes into digital scenes. In the Sim2Real step, we deploy the end-to-end model onto the vehicle to bridge the Sim2Real gap. Trained in the 3DGS simulator and deployed on physical vehicles, REAP successfully parks in various types of parking spaces, especially demonstrating the feasibility of end-to-end RL parking in extremely narrow mechanical slots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16355v1">Generative 3D Gaussians with Learned Density Control</a></div>
    <div class="paper-meta">
      📅 2026-05-08
      | 💬 19 pages, 16 figures, SIGGRAPH Conference Papers '26
    </div>
    <details class="paper-abstract">
      We present Density-Sampled Gaussians (DeG), a novel 3D representation designed to bridge the gap between adaptive rendering primitives and scalable generative modeling. Unlike existing approaches that constrain 3D Gaussians to fixed voxel grids or arrays, DeG models Gaussian centers as samples from a learnable probability density function defined over an octree. This formulation provides a rigorous mathematical framework for adaptive density control: by jointly optimizing the spatial density and Gaussian attributes under rendering supervision, our model naturally concentrates primitives in regions of high geometric complexity. We achieve this via a new render loss contribution gradient that serves as a fully differentiable analogue to the discrete densification and pruning heuristics used in standard Gaussian Splatting. The resulting representation is highly flexible, supporting variable-resolution decoding from a single latent code by simply adjusting the sampling budget. To enable generative synthesis, we train a latent diffusion model on DeG. We identify a critical challenge in applying diffusion to unordered set-structured latents, which can significantly slow convergence, and propose VecSeq, a canonical re-indexing mechanism that anchors latent tokens to a deterministic 3D Sobol sequence. This transforms the ambiguous set-generation problem into a robust sequence modeling task. Extensive experiments demonstrate that our pipeline achieves state-of-the-art quality in single-image-to-3D generation, combining the structural adaptivity of unstructured primitives with the training stability of grid-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07781v1">Differentiable Ray Tracing with Gaussians for Unified Radio Propagation Simulation and View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-05-08
    </div>
    <details class="paper-abstract">
      Explicit neural representations such as 3D Gaussian Splatting (3DGS) enable high-fidelity and real-time novel view synthesis, yet optimize for alpha-composited optical appearance rather than ray-intersectable geometry. In contrast, radio-frequency (RF) digital twins require deterministic multi-bounce paths, where the geometry dictates trajectories and their associated attenuation and delay. We introduce a framework enabling differentiable RF propagation simulation directly within visually reconstructed neural scenes, allowing point-to-point path computation between arbitrary 3D locations while preserving high-quality visual rendering. Unlike conventional RF simulation pipelines that rely on manually constructed meshes, we embed Gaussian primitives into a hardware-accelerated ray tracing structure as the underlying spatial representation. By extracting physically meaningful channel impulse responses from visual-only reconstructions, we provide cross-modal evidence that neural reconstructions can serve as unified spatial representations for both electromagnetic propagation simulation and photorealistic view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02129v2">VDEGaussian: Video Diffusion Enhanced 4D Gaussian Splatting for Dynamic Urban Scenes Modeling</a></div>
    <div class="paper-meta">
      📅 2026-05-08
    </div>
    <details class="paper-abstract">
      Dynamic urban scene modeling is a rapidly evolving area with broad applications. While current approaches leveraging neural radiance fields or Gaussian Splatting have achieved fine-grained reconstruction and high-fidelity novel view synthesis, they still face significant limitations. These often stem from a dependence on pre-calibrated object tracks or difficulties in accurately modeling fast-moving objects from undersampled capture, particularly due to challenges in handling temporal discontinuities. To overcome these issues, we propose a novel video diffusion-enhanced 4D Gaussian Splatting framework. Our key insight is to distill robust, temporally consistent priors from a test-time adapted video diffusion model. To ensure precise pose alignment and effective integration of this denoised content, we introduce two core innovations: a joint timestamp optimization strategy that refines interpolated frame poses, and an uncertainty distillation method that adaptively extracts target content while preserving well-reconstructed regions. Extensive experiments demonstrate that our method significantly enhances dynamic modeling, especially for fast-moving objects, achieving an approximate PSNR gain of 2 dB for novel view synthesis over baseline approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12647v2">LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-08
      | 💬 8 pages, 7 figures, conference
    </div>
    <details class="paper-abstract">
      Recent 3D Gaussian Splatting (3DGS) methods have demonstrated the feasibility of self-driving scene reconstruction and novel view synthesis. However, most existing methods either rely solely on cameras or use LiDAR only for Gaussian initialization or depth supervision, while the rich scene information contained in point clouds, such as reflectance, and the complementarity between LiDAR and RGB have not been fully exploited, leading to degradation in challenging self-driving scenes, such as those with high ego-motion and complex lighting. To address these issues, we propose a robust and efficient LiDAR-reflectance-guided Salient Gaussian Splatting method (LR-SGS) for self-driving scenes, which introduces a structure-aware Salient Gaussian representation, initialized from geometric and reflectance feature points extracted from LiDAR and refined through a salient transform and improved density control to capture edge and planar structures. Furthermore, we calibrate LiDAR intensity into reflectance and attach it to each Gaussian as a lighting-invariant material channel, jointly aligned with RGB to enforce boundary consistency. Extensive experiments on the Waymo Open Dataset demonstrate that LR-SGS achieves superior reconstruction performance with fewer Gaussians and shorter training time. In particular, on Complex Lighting scenes, our method surpasses OmniRe by 1.18 dB PSNR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01195v2">TAIL-Safe: Task-Agnostic Safety Monitoring for Imitation Learning Policies</a></div>
    <div class="paper-meta">
      📅 2026-05-08
    </div>
    <details class="paper-abstract">
      Recent imitation learning (IL) algorithms such as flow-matching and diffusion policies demonstrate remarkable performance in learning complex manipulation tasks. However, these policies often fail even when operating within their training distribution due to extreme sensitivity to initial conditions and irreducible approximation errors that lead to compounding drift. This makes it unsafe to deploy IL policies in the field where out-of-distribution scenarios are prevalent. A prerequisite for safe deployment is enabling the policy to determine whether it can execute a task the way it was learned from demonstrations. This paper presents TAIL-Safe, a principled approach to identify, for a trained IL policy, a safe set from where the policy empirically succeeds in completing the learned task. We propose a Lipschitz-continuous Q-value function that maps state-action pairs to a long-term safety score based on three short-term task-agnostic criteria: visibility, recognizability, and graspability. The zero-superlevel set of this function characterizes an empirical control invariant set over state-action pairs. When the nominal policy proposes an action outside this set, we apply a recovery mechanism inspired by Nagumo's theorem that uses gradient ascent to the Q-function to steer the policy back to safety. To learn this Q-function, we construct a high-fidelity digital twin using Gaussian Splatting that enables systematic collection of failure data without risk to physical hardware. Experiments with a Franka Emika robot demonstrate that flow-matching policies, which fail under run-time perturbations, achieve consistent task success when guided by the proposed TAIL-Safe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07351v1">Disambiguating 2D-3D Correspondences in Gaussian Splatting-based Feature Fields for Visual Localization</a></div>
    <div class="paper-meta">
      📅 2026-05-08
    </div>
    <details class="paper-abstract">
      While Gaussian Splatting-based Feature Fields (GSFFs) have shown promise for visual localization, this paper highlights that photometrically optimized GSFFs are inherently ill-suited for 2D-3D matching. The volumetric extent of each Gaussian induces many-to-one pixel-to-point mappings that destabilize PnP-based pose estimation, while photometric optimization gives rise to superfluous Gaussians devoid of multi-view consistency. To address these issues, we propose SplitGS-Loc, a localization-specialized GSFFs construction framework that disambiguates 2D-3D correspondences by exploiting Gaussian attributes. Our key design, Mixture-of-Gaussians-based splitting, decomposes each Gaussian into smaller Gaussians, replacing ambiguous many-to-one with precise one-to-one correspondences. In parallel, we exploit composition weights from GS rasterization to select Gaussians that significantly and consistently contribute across multiple views and aggregate discriminative features through strong pixel-Gaussian associations, enforcing multi-view consistency. The resulting compact yet discriminative feature fields enable stable PnP convergence, achieving state-of-the-art performance on localization benchmarks. Extensive experiments validate that SplitGS-Loc extends the utility of photometric GSFFs to accurate and efficient localization by exploiting Gaussian attributes, without per-scene training or iterative pose refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07254v1">High-Fidelity Surface Splatting-Based 3D Reconstruction from Multi-View Images</a></div>
    <div class="paper-meta">
      📅 2026-05-08
      | 💬 19 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Multi-view mesh reconstruction remains a core challenge in computer graphics and vision, especially for recovering high-frequency geometry from sparse observations. Recent methods such as 3D Gaussian Splatting (3DGS) and Neural Radiance Fields (NeRF) rely on post-processing for mesh extraction, thereby limiting joint optimization of geometry and appearance. Implicit Moving Least Squares (IMLS) instead enables direct conversion of point clouds into signed distance and texture fields, supporting end-to-end reconstruction and rendering. However, existing IMLS formulations use exponential kernels that struggle with high-frequency detail. We introduce a compact polynomial kernel with local support and greater flexibility, allowing better control over frequency content and improved geometric fidelity. To further enhance fine details, we incorporate stochastic regularization with Laplacian filtering. Together, these improve the preservation of high-frequency structure while maintaining stable optimization. Experiments show state-of-the-art performance in both surface reconstruction and rendering, yielding more accurate geometry and sharper visuals from multi-view data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16736v5">A Step to Decouple Optimization in 3DGS</a></div>
    <div class="paper-meta">
      📅 2026-05-08
      | 💬 Accepted by ICLR 2026 (fixed typo)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for real-time novel view synthesis. As an explicit representation optimized through gradient propagation among primitives, optimization widely accepted in deep neural networks (DNNs) is actually adopted in 3DGS, such as synchronous weight updating and Adam with the adaptive gradient. However, considering the physical significance and specific design in 3DGS, there are two overlooked details in the optimization of 3DGS: (i) update step coupling, which induces optimizer state rescaling and costly attribute updates outside the viewpoints, and (ii) gradient coupling in the moment, which may lead to under- or over-effective regularization. Nevertheless, such a complex coupling is under-explored. After revisiting the optimization of 3DGS, we take a step to decouple it and recompose the process into: Sparse Adam, Re-State Regularization and Decoupled Attribute Regularization. Taking a large number of experiments under the 3DGS and 3DGS-MCMC frameworks, our work provides a deeper understanding of these components. Finally, based on the empirical analysis, we re-design the optimization and propose AdamW-GS by re-coupling the beneficial components, under which better optimization efficiency and representation effectiveness are achieved simultaneously.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07192v1">AsyncEvGS: Asynchronous Event-Assisted Gaussian Splatting for Handheld Motion-Blurred Scenes</a></div>
    <div class="paper-meta">
      📅 2026-05-08
    </div>
    <details class="paper-abstract">
      3D reconstruction methods such as 3D Gaussian Splatting (3DGS) and Neural Radiance Fields (NeRF) achieve impressive photorealism but fail when input images suffer from severe motion blur. While event cameras provide high-temporal-resolution motion cues, existing event-assisted approaches rely on low-resolution sensors and strict synchronization, limiting their practicality for handheld 3D capture on common devices, such as smartphones. We introduce a flexible, high-resolution asynchronous RGB-Event dual-camera system and a corresponding reconstruction framework. Our approach first reconstructs sharp images from the event data and then employs a cross-domain pose estimation module based on the Visual Geometry Transformer (VGGT) to obtain robust initialization for 3DGS. During optimization, we employ a structure-driven event loss and view-specific consistency regularizers to mitigate the ill-posed behavior of traditional event losses and deblurring losses, ensuring both stable and high-fidelity reconstruction. We further contribute AsyncEv-Deblur, a new high-resolution RGB-Event dataset captured with our asynchronous system. Experiments demonstrate that our method achieves state-of-the-art performance on both our challenging dataset and existing benchmarks, substantially improving reconstruction robustness under severe motion blur. Project page: https://openimaginglab.github.io/AsyncEvGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07181v1">SatSurfGS: Generalizable 2D Gaussian Splatting for Sparse-View Satellite Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-08
    </div>
    <details class="paper-abstract">
      Sparse-view satellite image surface reconstruction remains highly challenging, fundamentally because the reliability of multi-view matching under satellite imaging conditions is strongly spatially heterogeneous. Affected by large photometric differences, weak textures, and repetitive textures, multi-view geometric constraints are often sparse, unevenly distributed, and locally unreliable. Although 2D Gaussian Splatting (2DGS) is more suitable than 3D Gaussian Splatting (3DGS) for the explicit representation of continuous surfaces, research on generalizable feed-forward 2DGS frameworks for sparse-view satellite surface reconstruction is still lacking. To address this issue, we propose SatSurfGS, a generalizable sparse-view surface reconstruction method for satellite imagery based on 2DGS. The proposed method builds a coarse-to-fine Gaussian attribute prediction framework and explicitly models local geometric reliability at three levels: feature learning, Gaussian parameter estimation, and training optimization. Specifically, we propose a confidence-aware monocular multi-view feature fusion module to adaptively integrate monocular priors and multi-view matching features according to local confidence; a cross-stage self-consistency residual guidance module to stabilize stage-wise Gaussian parameter refinement using the residual between the rendered height map from the previous stage and the current-stage MVS height map, together with confidence information; and a confidence bidirectional routing loss to achieve differentiated allocation of geometric and appearance supervision. Experiments on satellite datasets show that the proposed method achieves improved rendering quality, surface reconstruction accuracy, cross-dataset generalization, and inference efficiency compared with representative generalizable baselines and competitive per-scene optimization methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06088v1">OpenGaFF: Open-Vocabulary Gaussian Feature Field with Codebook Attention</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Understanding open-vocabulary 3D scenes with Gaussian-based representations remains challenging due to fragmented and spatially inconsistent semantic predictions across multi-view observations. In this paper, we present OpenGaFF, a novel framework for open-vocabulary 3D scene understanding built upon 3D Gaussian Splatting. At the core of our method is a Gaussian Feature Field that models semantics as a continuous function of Gaussian geometry and appearance. By explicitly conditioning semantic predictions on geometric structure, this formulation strengthens the coupling between geometry and semantics, leading to improved spatial coherence across similar structures in 3D space. To further enforce object-level semantic consistency, we introduce a structured codebook that serves as a set of shared semantic primitives. Furthermore, a codebook-guided attention mechanism is proposed to retrieve language features via similarity matching between query embeddings and learned codebook entries, enabling robust open-vocabulary reasoning while reducing intra-object feature variance. Extensive experiments on standard 2D and 3D open-vocabulary benchmarks demonstrate that our method consistently outperforms prior approaches, achieving improved segmentation quality, stronger 3D semantic consistency and a semantically interpretable codebook that provides insight into the learned representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05876v1">3DSS: 3D Surface Splatting for Inverse Rendering</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      We present 3D Surface Splatting (3DSS), the first differentiable surface splatting renderer for physically-based inverse rendering from multi-view images. Our central insight is that the surface separation problem at the heart of surface splatting admits a direct formulation in terms of the reconstruction kernels themselves. From this foundation we derive a coverage-based compositing model whose per-layer opacity arises directly from the accumulated Elliptical Weighted Average reconstruction weight, yielding anti-aliased silhouettes and informative visibility gradients at sparsely covered edges. Combined with forward microfacet shading under co-optimized HDR environment lighting and density-aware adaptive refinement, 3DSS jointly recovers shape, spatially-varying BRDF materials, and illumination. Because the optimized representation is a set of oriented surface samples, it bridges natively to mesh-based workflows via surface reconstruction from oriented point cloud methods. We evaluate 3DSS against mesh-based, implicit, and Gaussian-splatting baselines across geometry reconstruction, novel-view synthesis, and novel-illumination relighting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26799v2">MesonGS++: Post-training Compression of 3D Gaussian Splatting with Hyperparameter Searching</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 https://github.com/mmlab-sigs/mesongs_plus
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves high-quality novel view synthesis with real-time rendering, but its storage cost remains prohibitive for practical deployment. Existing post-training compression methods still rely on many coupled hyperparameters across pruning, transformation, quantization, and entropy coding, making it difficult to control the final compressed size and fully exploit the rate-distortion trade-off. We propose MesonGS++, a size-aware post-training codec for 3D Gaussian compression. On the codec side, MesonGS++ combines joint importance-based pruning, octree geometry coding, attribute transformation, selective vector quantization for higher-degree spherical harmonics, and group-wise mixed-precision quantization with entropy coding. On the configuration side, it treats the reserve ratio and bit-width allocation as the dominant rate-distortion knobs and jointly optimizes them under a target storage budget via discrete sampling and 0--1 integer linear programming. We further propose a linear size estimator and a CUDA parallel quantization operator to accelerate the hyperparameter searching process. Extensive experiments show that MesonGS++ achieves over 34$\times$ compression while preserving rendering fidelity, outperforming state-of-the-art post-training methods and accurately meeting target size budgets. Remarkably, without any training, MesonGS++ can even surpass the PSNR of vanilla 3DGS at a 20$\times$ compression rate on the Stump scene. Our code is available at https://github.com/mmlab-sigs/mesongs_plus
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06876v1">AdpSplit: Error-Driven Adaptive Splitting for Faster Geometry Discovery in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Adaptive density control in 3D Gaussian Splatting (3DGS) repeatedly grows the Gaussian population through fixed-cardinality random splitting to discover useful scene structure. However, in vanilla 3DGS, its binary split operator requires many densification rounds to expose fine details, making it a bottleneck for efficient training schedules with fewer iterations. We introduce AdpSplit, an error-driven adaptive split operator that determines the number of split children and initializes the child parameters from L1-pixel-error region statistics, enabling fewer densification iterations, thus reduced training time, while preserving the rendering quality of full-schedule training. Across the MipNeRF360, Deep-Blending, and Tanks&Temples datasets, AdpSplit reduces the training time of multiple accelerated 3DGS pipelines by 9.2%-22.3% as a simple drop-in replacement for the standard split operator. With FastGS, AdpSplit matches the full-schedule PSNR on MipNeRF360 while reducing training time by 16.4%, corresponding to a 12.6x acceleration over vanilla 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.13549v2">KFC-W: Generating 3D-Consistent Videos from Unposed Internet Photos</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 project page: https://genechou.com/kfcw/
    </div>
    <details class="paper-abstract">
      We address the problem of generating videos from unposed internet photos. A handful of input images serve as keyframes, and our model interpolates between them to simulate a path moving between the cameras. Given random images, a model's ability to capture underlying geometry, recognize scene identity, and relate frames in terms of camera position and orientation reflects a fundamental understanding of 3D structure and scene layout. However, existing video models such as Luma Dream Machine fail at this task. We design a self-supervised method that takes advantage of the consistency of videos and variability of multiview internet photos to train a scalable, 3D-aware video model without any 3D annotations such as camera parameters. We validate that our method outperforms all baselines in terms of geometric and appearance consistency. We also show our model benefits applications that enable camera control, such as 3D Gaussian Splatting. Our results suggest that we can scale up scene-level 3D learning using only 2D data such as videos and multiview internet photos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05155v1">Aes3D: Aesthetic Assessment in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      As 3D Gaussian Splatting (3DGS) gains attention in immersive media and digital content creation, assessing the aesthetics of 3D scenes becomes important in helping creators build more visually compelling 3D content. However, existing evaluation methods for 3D scenes primarily emphasize reconstruction fidelity and perceptual realism, largely overlooking higher-level aesthetic attributes such as composition, harmony, and visual appeal. This limitation comes from two key challenges: (1) the absence of general 3DGS datasets with aesthetic annotations, and (2) the intrinsic nature of 3DGS as a low-level primitive representation, which makes it difficult to capture high-level aesthetic features. To address these challenges, we propose Aes3D, the first systematic framework for assessing the aesthetics of 3D neural rendering scenes. Aes3D includes Aesthetic3D, the first dataset dedicated to 3D scene aesthetic assessment, built on our proposed annotation strategy for 3D scene aesthetics. In addition, we present Aes3DGSNet, a lightweight model that directly predicts scene-level aesthetic scores from 3DGS representations. Notably, our model operates solely on 3D Gaussian primitives, eliminating the need for rendering multi-view images and thus reducing computational cost and hardware requirements. Through aesthetics-supervised learning on multi-view 3DGS scene representations, Aes3DGSNet effectively captures high-level aesthetic cues and accurately regresses aesthetic scores. Experimental results demonstrate that our approach achieves strong performance while maintaining a lightweight design, establishing a new benchmark for 3D scene aesthetic assessment. Code and datasets will be made available in a future version.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.13783v2">RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Accepted to CVPR2026
    </div>
    <details class="paper-abstract">
      Temporal retiming, the ability to reconstruct and render dynamic scenes at arbitrary timestamps, is crucial for applications such as slow-motion playback, temporal editing, and post-production. However, most existing 4D Gaussian Splatting (4DGS) methods overfit at discrete frame indices but struggle to represent continuous-time frames, leading to ghosting artifacts when interpolating between timestamps. We identify this limitation as a form of temporal aliasing and propose RetimeGS, a simple yet effective 4DGS representation that explicitly defines the temporal behavior of the 3D Gaussian and mitigates temporal aliasing. To achieve smooth and consistent interpolation, we incorporate optical flow-guided initialization and supervision, triple-rendering supervision, and other targeted strategies. Together, these components enable ghost-free, temporally coherent rendering even under large motions. Experiments on datasets featuring fast motion, non-rigid deformation, and severe occlusions demonstrate that RetimeGS achieves superior quality and coherence over state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09881v3">LTGS: Long-Term Gaussian Scene Chronology From Sparse View Updates</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Accepted to CVPR 2026 Findings. Project page: https://mkjjang3598.github.io/LTGS
    </div>
    <details class="paper-abstract">
      Recent advances in novel-view synthesis can create the photo-realistic visualization of real-world environments from conventional camera captures. However, the everyday environment experiences frequent scene changes, which require dense observations, both spatially and temporally, that an ordinary setup cannot cover. We propose long-term Gaussian scene chronology from sparse-view updates, coined LTGS, an efficient scene representation that can embrace everyday changes from highly under-constrained casual captures. Given an incomplete and unstructured 3D Gaussian Splatting (3DGS) representation obtained from an initial set of input images, we robustly model the long-term chronology of the scene despite abrupt movements and subtle environmental variations. We construct objects as template Gaussians, which serve as structural, reusable priors for shared object tracks. Then, the object templates undergo a further refinement pipeline that modulates the priors to adapt to temporally varying environments given few-shot observations. Once trained, our framework is generalizable across multiple time steps through simple transformations, significantly enhancing the scalability for a temporal evolution of 3D environments. As existing datasets do not explicitly represent the long-term real-world changes with a sparse capture setup, we collect real-world datasets to evaluate the practicality of our pipeline. Experiments demonstrate that our framework achieves superior reconstruction quality compared to other baselines while enabling fast and light-weight updates. Project page is available at: https://mkjjang3598.github.io/LTGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04844v1">QuadBox: Accelerating 3D Gaussian Splatting with Geometry-Aware Boxes</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 6 pages, 4 figures. Accepted by ICIP 26
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as an advanced technique for real-time novel view synthesis by representing scene geometry and appearance using differentiable Gaussian primitives. However, efficiently computing precise Gaussian-tile intersections remains a critical task in the rasterization pipeline. To this end, we propose QuadBox, a method that leverages four axis-aligned bounding boxes to tightly encapsulate projected Gaussians in a discrete manner. First, we derive a geometry-aware stretching factor that enables the construction of a tile-aligned QuadBox, which covers the elliptical projection and largely excludes irrelevant tiles. Second, we introduce QPass, a single-pass tile traversal algorithm that exhaustively exploits the discrete nature of QuadBox, ensuring that the tile intersection check is performed with simple interval tests. Experiments on public datasets show that our method accelerates the rendering speed of 3DGS by 1.85$\times$. Code is available at \href{https://github.com/Powertony102/QuadBox}{https://github.com/Powertony102/QuadBox}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04730v1">ULF-Loc: Unbiased Landmark Feature for Robust Visual Localization with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 published to CVPR (highlight)
    </div>
    <details class="paper-abstract">
      Visual localization is a core technology for augmented reality and autonomous navigation. Recent methods combine the efficient rendering of 3D Gaussian Splatting (3DGS) with feature-based localization. These methods rely on direct matching between 2D query features and the 3D Gaussian feature field, but this often results in mismatches due to an inherent bias in the learned Gaussian feature. We theoretically analyze the feature learning process in 3DGS, revealing that the widely adopted $α$-blending optimization inherently introduces bias into 3D point features. This bias stems from the entanglement between individual Gaussians and their neighboring Gaussians, making the learned features unsuitable for precise matching tasks. Motivated by these findings, we propose ULF-Loc, an unbiased landmark feature framework that replaces biased feature optimization with geometry-weighted feature fusion. We further introduce keypoint-consensus landmark sampling to select reliable Gaussians and local geometric consistency verification to reject mismatches caused by rendering artifacts. On the Cambridge Landmarks dataset, ULF-Loc reduces the mean median translation error by 17\% compared to the state-of-the-art, while achieving superior efficiency with only 1/10 the training time and 1/6 the GPU memory of STDLoc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04509v1">CoherentRaster: Efficient 3D Gaussian Splatting for Light Field Displays</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Light field displays (LFDs) require rendering an interlaced image that encodes many view-dependent observations. This multi-view requirement introduces substantial computational overhead, making real-time rendering difficult to achieve. While 3D Gaussian Splatting (3DGS) is efficient for single-view rendering on 2D displays, directly extending it to LFDs is computationally expensive. Moreover, prior accelerations either suffer from GPU inefficiency under spatially incoherent subpixel layouts or rely on computationally heavy multi-plane intermediates. In this paper, we propose CoherentRaster, a 3DGS-based light field rendering framework that performs subpixel-level rasterization. Our method employs Cross-view Coherent Attribute Reuse to eliminate redundant computation across neighboring viewpoints and applies View-coherent Remapping to restore warp-level memory efficiency degraded by the interlaced subpixel layout. Together, CoherentRaster provides an efficient pipeline for real-time, high-quality light field synthesis on consumer-grade hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04506v1">Ilov3Splat: Instance-Level Open-Vocabulary 3D Scene Understanding in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 The International Conference on Pattern Recognition (ICPR) 2026
    </div>
    <details class="paper-abstract">
      We introduce Ilov3Splat, a novel framework for instance-level open-vocabulary 3D scene understanding built on 3D Gaussian Splatting (3D-GS). Most prior work depends on 2D rendering-based matching or point-level semantic association, which undermines cross-view consistency, lacks coherent instance-level reasoning, and limits precision in downstream 3D tasks. To address these limitations, our method jointly optimizes scene geometry and semantic representations by augmenting Gaussian splats with view-consistent feature fields. Specifically, we leverage multi-resolution hash embedding to efficiently encode language-aligned CLIP features, enabling dense and coherent language grounding in 3D space. We further train an instance feature field using contrastive loss over SAM masks, supporting fine-grained object distinction across views. At inference time, CLIP-encoded queries are matched against the learned features, followed by two-stage 3D clustering to retrieve relevant Gaussian groups. This enables our framework to identify arbitrary objects in 3D scenes based on natural language descriptions, without requiring category supervision or manual annotations. Experiments on standard benchmarks demonstrate that Ilov3Splat outperforms prior open-vocabulary 3D-GS methods in both object selection and instance segmentation, offering a flexible and accurate solution for language-driven 3D scene understanding. Project page: https://csiro-robotics.github.io/Ilov3Splat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00285v2">SV-GS: Sparse View 4D Reconstruction with Skeleton-Driven Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Reconstructing a dynamic target moving over a large area is challenging. Standard approaches for dynamic object reconstruction require dense coverage in both the viewing space and the temporal dimension, typically relying on multi-view videos captured at each time step. However, such setups are only possible in constrained environments. In real-world scenarios, observations are often sparse over time and captured sparsely from diverse viewpoints (e.g., from security cameras), making dynamic reconstruction highly ill-posed. We present SV-GS, a framework that simultaneously estimates a deformation model and the object's motion over time under sparse observations. To initialize SV-GS, we leverage a rough skeleton graph and an initial static reconstruction as inputs to guide motion estimation. (Later, we show that this input requirement can be relaxed.) Our method optimizes a skeleton-driven deformation field composed of a coarse skeleton joint pose estimator and a module for fine-grained deformations. By making only the joint pose estimator time-dependent, our model enables smooth motion interpolation while preserving learned geometric details. Experiments on synthetic datasets show that our method outperforms existing approaches under sparse observations by up to 34% in PSNR, and achieves comparable performance to dense monocular video methods on real-world datasets despite using significantly fewer frames. Moreover, we demonstrate that the input initial static reconstruction can be replaced by a diffusion-based generative prior, making our method more practical for real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.03077v2">RoDyGS: Robust Dynamic Gaussian Splatting for Casual Videos</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 29 pages, 14 figures
    </div>
    <details class="paper-abstract">
      4D reconstruction from casually captured monocular videos is challenging due to inherent ambiguity in reconstructing dynamic 3D geometry. To address this challenge, we introduce Robust Dynamic Gaussian Splatting (RoDyGS), a method that reconstructs dynamic scene representation from casual monocular videos. RoDyGS explicitly separates static and dynamic scene elements, and applies spatiotemporal regularization to enforce physically plausible geometry and temporally consistent motion. Furthermore, we propose a comprehensive benchmark, Kubric-MRig, which provides extensive camera and object motion along with simultaneous multi-view capture, features that are absent in previous benchmarks. Experiments demonstrate that RoDyGS significantly outperforms previous pose-free dynamic novel view synthesis approaches and achieves competitive rendering quality compared to existing pose-free static novel view synthesis approaches. Our proejct page is available at https://rodygs.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04435v1">Ground4D: Spatially-Grounded Feedforward 4D Reconstruction for Unstructured Off-Road Scenes</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Feedforward Gaussian Splatting has recently emerged as an efficient paradigm for 4D reconstruction in autonomous driving. However, in unstructured off-road scenes, its performance degrades due to high-frequency geometry, ego-motion jitter, and increased non-rigid dynamics. These factors introduce conflicting Gaussian observations across timestamps, leading to either over-smoothed renderings or structural artifacts. To address this issue, we propose Ground4D, a spatially-grounded 4D feedforward framework for pose-free off-road reconstruction. The key idea is to resolve temporal conflicts through spatially localized conditioning. Specifically, we introduce voxel-grounded temporal Gaussian aggregation, which partitions the canonical Gaussian space into spatial voxels and performs query-conditioned temporal attention within each voxel. Intra-voxel softmax normalization ensures that temporal selectivity and spatial occupancy become mutually reinforcing rather than conflicting. We furthermore introduce surface normal cues as auxiliary geometric guidance to regularize the geometry of Gaussian primitives. Extensive experiments on ORAD-3D and RELLIS-3D demonstrate that Ground4D consistently outperforms existing feedforward methods in reconstruction quality and generalizes zero-shot to unseen off-road domains. Project page and code:https://github.com/wsnbws/Ground4D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00492v2">ArtiFixer: Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2026-05-05
      | 💬 Video results: https://research.nvidia.com/labs/sil/projects/artifixer/
    </div>
    <details class="paper-abstract">
      Per-scene optimization methods such as 3D Gaussian Splatting provide state-of-the-art novel view synthesis quality but extrapolate poorly to under-observed areas. Methods that leverage generative priors to correct artifacts in these areas hold promise but currently suffer from two shortcomings. The first is scalability, as existing methods use image diffusion models or bidirectional video models that are limited in the number of views they can generate in a single pass (and thus require a costly iterative distillation process for consistency). The second is quality itself, as generators used in prior work tend to produce outputs that are inconsistent with existing scene content and fail entirely in completely unobserved regions. To solve these, we propose a two-stage pipeline that leverages two key insights. First, we train a powerful bidirectional generative model with a novel opacity mixing strategy that encourages consistency with existing observations while retaining the model's ability to extrapolate novel content in unseen areas. Second, we distill it into a causal auto-regressive model that generates hundreds of frames in a single pass. This model can directly produce novel views or serve as pseudo-supervision to improve the underlying 3D representation in a simple and highly efficient manner. We evaluate our method extensively and demonstrate that it can generate plausible reconstructions in scenarios where existing approaches fail completely. When measured on commonly benchmarked datasets, we outperform all existing baselines by a wide margin, exceeding prior state-of-the-art methods by 1-3 dB PSNR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03337v1">FreeTimeGS++: Secrets of Dynamic Gaussian Splatting and Their Principles</a></div>
    <div class="paper-meta">
      📅 2026-05-05
      | 💬 22 pages, 8 figures
    </div>
    <details class="paper-abstract">
      The recent surge in 4D Gaussian Splatting (4DGS) has achieved impressive dynamic scene reconstruction. While these methods demonstrate remarkable performance, the specific drivers behind such gains remain less explored, making a systematic understanding of the underlying principles challenging. In this paper, we perform a comprehensive analysis of these hidden factors to provide a clearer perspective on the 4DGS framework. We first establish a controlled baseline, FreeTimeGS_ours, by formalizing and reproducing the heuristics of the state-of-the-art FreeTimeGS. Using this framework, we dissect 4DGS along its fundamental axes and uncover key secrets, including the emergent temporal partitioning driven by Gaussian durations and the discrepancy between photometric fidelity and spatiotemporal consistency. Based on these insights, we propose FreeTimeGS++, a principled method that employs gated marginalization and neural velocity fields to achieve superior stability and robust dynamic representations. Our approach yields reproducible results with reduced run-to-run variance. We will release our implementation to provide a reliable foundation for future 4DGS research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21874v2">Interactive Augmented Reality-enabled Outdoor Scene Visualization For Enhanced Real-time Disaster Response</a></div>
    <div class="paper-meta">
      📅 2026-05-04
      | 💬 6 pages, 2 figures
    </div>
    <details class="paper-abstract">
      A user-centered AR interface for disaster response is presented in this work that uses 3D Gaussian Splatting (3DGS) to visualize detailed scene reconstructions, while maintaining situational awareness and keeping cognitive load low. The interface relies on a lightweight interaction approach, combining World-in-Miniature (WIM) navigation with semantic Points of Interest (POIs) that can be filtered as needed, and it is supported by an architecture designed to stream updates as reconstructions evolve. User feedback from a preliminary evaluation indicates that this design is easy to use and supports real-time coordination, with participants highlighting the value of interaction and POIs for fast decision-making in context. Thorough user-centric performance evaluation demonstrates strong usability of the developed interface and high acceptance ratios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17817v3">Chorus: Multi-Teacher Pretraining for Holistic 3D Gaussian Scene Encoding</a></div>
    <div class="paper-meta">
      📅 2026-05-04
      | 💬 Project page at https://gaussianworld.github.io/Chorus
    </div>
    <details class="paper-abstract">
      While 3DGS has emerged as a high-fidelity scene representation, encoding rich, general-purpose features directly from its primitives remains under-explored. We address this gap by introducing Chorus, a multi-teacher pretraining framework that learns a holistic feed-forward 3D Gaussian Splatting (3DGS) scene encoder by distilling complementary signals from 2D foundation models. Chorus employs a shared 3D encoder and teacher-specific projectors to learn from language-aligned, generalist, and object-aware teachers, encouraging a shared embedding space that captures signals from high-level semantics to fine-grained structure. We evaluate Chorus on a wide range of tasks: open-vocabulary semantic and instance segmentation, linear and decoder probing, data-efficient supervision, as well as LLM-based Q&A. Besides 3DGS, we also test Chorus on several benchmarks that only support point clouds by pretraining a variant using only Gaussian centers, colors, and estimated normals. Surprisingly, this encoder shows strong transfer and outperforms the point-cloud baseline while using 39.9 times fewer training scenes. Finally, we propose a render-and-distill adaptation that facilitates out-of-domain finetuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17185v2">LGDWT-GS: Local and Global Discrete Wavelet-Regularized 3D Gaussian Splatting for Sparse-View Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-04
    </div>
    <details class="paper-abstract">
      We propose a new method for few-shot 3D reconstruction that integrates global and local frequency regularization to stabilize geometry and preserve fine details under sparse-view conditions, addressing a key limitation of existing 3D Gaussian Splatting (3DGS) models. We also introduce a new multispectral greenhouse dataset containing four spectral bands captured from diverse plant species under controlled conditions. Alongside the dataset, we release an open-source benchmarking package that defines standardized few-shot reconstruction protocols for evaluating 3DGS-based methods. Experiments on our multispectral dataset, as well as standard benchmarks, demonstrate that the proposed method achieves sharper, more stable, and spectrally consistent reconstructions than existing baselines. The dataset and code for this work are publicly available
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.02784v1">HumanSplatHMR: Closing the Loop Between Human Mesh Recovery and Gaussian Splatting Avatar</a></div>
    <div class="paper-meta">
      📅 2026-05-04
    </div>
    <details class="paper-abstract">
      Accurately recovering human pose and appearance from video is an essential component of scene reconstruction, with applications to motion capture, motion prediction, virtual reality, and digital twinning. Despite significant interest in building realistic human avatars from video, this paper demonstrates that existing methods do not accurately recover the 3D geometry of humans. ViT-based approaches are not consistently reliable and can overfit to 2D views, while NeRF- and Gaussian Splatting-based avatars treat pose and appearance separately, limiting rendering generalization to new poses. To resolve these shortcomings, this paper proposes HumanSplatHMR, a joint optimization framework that refines 3D human poses while simultaneously learning a high-fidelity avatar for novel-view and novel-pose synthesis. Our key insight is to close the loop between geometric pose estimation and differentiable rendering. Unlike prior human avatar methods that rely on accurate human pose obtained through motion capture systems or offline refinement, which are impractical in in-the-wild scenarios, our approach uses only human mesh estimates from a state-of-the-art human pose estimator to better reflect real-world conditions. Therefore, instead of using the human pose only as a deformation prior, HumanSplatHMR backpropagates photometric, segmentation, and depth losses through a differentiable renderer to the pose parameters and global position. This coupling refines the global 3D pose over time, improving accuracy and alignment while producing better renderings from novel views. Experiments show consistent improvements over pose recovery baselines that omit image-level refinement and avatar baselines that decouple pose estimation from avatar reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03200v2">A High-Fidelity Digital Twin for Robotic Manipulation Based on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-04
      | 💬 Accepted By Journal of Robot Learning
    </div>
    <details class="paper-abstract">
      Developing high-fidelity, interactive digital twins is crucial for enabling closed-loop motion planning and reliable real-world robot execution, which are essential to advancing sim-to-real transfer. However, existing approaches often suffer from slow reconstruction, limited visual fidelity, and difficulties in converting photorealistic models into planning-ready collision geometry. We present a practical framework that constructs high-quality digital twins within minutes from sparse RGB inputs. Our system employs 3D Gaussian Splatting (3DGS) for fast, photorealistic reconstruction as a unified scene representation. We enhance 3DGS with visibility-aware semantic fusion for accurate 3D labelling and introduce an efficient, filter-based geometry conversion method to produce collision-ready models seamlessly integrated with a Unity-ROS2-MoveIt physics engine. In experiments with a Franka Emika Panda robot performing pick-and-place tasks, we demonstrate that this enhanced geometric accuracy effectively supports robust manipulation in real-world trials. These results demonstrate that 3DGS-based digital twins, enriched with semantic and geometric consistency, offer a fast, reliable, and scalable path from perception to manipulation in unstructured environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21668v2">Space-Time Forecasting of Dynamic Scenes with Motion-aware Gaussian Grouping</a></div>
    <div class="paper-meta">
      📅 2026-05-04
      | 💬 20 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Forecasting dynamic scenes remains a fundamental challenge in computer vision, as limited observations make it difficult to capture coherent object-level motion and long-term temporal evolution. We present Motion Group-aware Gaussian Forecasting (MoGaF), a framework for long-term scene extrapolation built upon the 4D Gaussian Splatting representation. MoGaF introduces motion-aware Gaussian grouping and group-wise optimization to enforce physically consistent motion across both rigid and non-rigid regions, yielding spatially coherent dynamic representations. Leveraging this structured space-time representation, a lightweight forecasting module predicts future motion, enabling realistic and temporally stable scene evolution. Experiments on synthetic and real-world datasets demonstrate that MoGaF consistently outperforms existing baselines in rendering quality, motion plausibility, and long-term forecasting stability. Our project page is available at https://slime0519.github.io/mogaf
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.18966v2">SVGS: Enhancing Gaussian Splatting Using Primitives with Spatially Varying Colors</a></div>
    <div class="paper-meta">
      📅 2026-05-04
      | 💬 IEEE Transactions on Visualization and Computer Graphics
    </div>
    <details class="paper-abstract">
      Gaussian Splatting demonstrates impressive results in multi-view reconstruction based on Gaussian explicit representations. However, the current Gaussian primitives only have a single view-dependent color and an opacity to represent the appearance and geometry of the scene, resulting in a non-compact representation. In this paper, we introduce a new method called SVGS (Spatially Varying Gaussian Splatting) that utilizes spatially varying colors and opacity in a single Gaussian primitive to improve its representation ability. We have implemented bilinear interpolation, movable kernels, and tiny neural networks as spatially varying functions. SVGS employs 2D Gaussian surfels as primitives, which significantly enhances novel-view synthesis while maintaining high-quality geometric reconstruction. This approach is particularly effective in practical applications, as scenes combining complex textures with relatively simple geometry occur frequently in real-world environments. Quantitative and qualitative experimental results demonstrate that all three functions outperform the baseline, with the best movable kernels achieving superior novel view synthesis performance on multiple datasets, highlighting the strong potential of spatially varying functions. Project page: https://ruixu.me/html/SuperGaussians/index.html
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03210v2">Flux4D: Flow-based Unsupervised 4D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-04
      | 💬 NeurIPS 2025. Project page: https://waabi.ai/flux4d/
    </div>
    <details class="paper-abstract">
      Reconstructing large-scale dynamic scenes from visual observations is a fundamental challenge in computer vision, with critical implications for robotics and autonomous systems. While recent differentiable rendering methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have achieved impressive photorealistic reconstruction, they suffer from scalability limitations and require annotations to decouple actor motion. Existing self-supervised methods attempt to eliminate explicit annotations by leveraging motion cues and geometric priors, yet they remain constrained by per-scene optimization and sensitivity to hyperparameter tuning. In this paper, we introduce Flux4D, a simple and scalable framework for 4D reconstruction of large-scale dynamic scenes. Flux4D directly predicts 3D Gaussians and their motion dynamics to reconstruct sensor observations in a fully unsupervised manner. By adopting only photometric losses and enforcing an "as static as possible" regularization, Flux4D learns to decompose dynamic elements directly from raw data without requiring pre-trained supervised models or foundational priors simply by training across many scenes. Our approach enables efficient reconstruction of dynamic scenes within seconds, scales effectively to large datasets, and generalizes well to unseen environments, including rare and unknown objects. Experiments on outdoor driving datasets show Flux4D significantly outperforms existing methods in scalability, generalization, and reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01674v2">VRGaussianAvatar: Integrating 3D Gaussian Avatars into VR</a></div>
    <div class="paper-meta">
      📅 2026-05-04
      | 💬 Accepted as an IEEE TVCG paper at IEEE VR 2026 (journal track)
    </div>
    <details class="paper-abstract">
      We present VRGaussianAvatar, an integrated system that enables real-time full-body 3D Gaussian Splatting (3DGS) avatars in virtual reality using only head-mounted display (HMD) tracking signals. The system adopts a parallel pipeline with a VR Frontend and a GA Backend. The VR Frontend uses inverse kinematics to estimate full-body pose and streams the resulting pose along with stereo camera parameters to the backend. The GA Backend stereoscopically renders a 3DGS avatar reconstructed from a single image. To improve stereo rendering efficiency, we introduce Binocular Batching, which jointly processes left and right eye views in a single batched pass to reduce redundant computation and support high-resolution VR displays. We evaluate VRGaussianAvatar with quantitative performance tests and a within-subject user study against image- and video-based mesh avatar baselines. Results show that VRGaussianAvatar sustains interactive VR performance and yields higher perceived appearance similarity, embodiment, and plausibility. Project page and source code are available at https://vrgaussianavatar.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.02086v1">GETA-3DGS: Automatic Joint Structured Pruning and Quantization for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-03
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) is a state-of-the-art representation for real-time photorealistic novel-view synthesis, yet a single high-fidelity scene typically occupies hundreds of megabytes to several gigabytes, exceeding the budgets of mobile, immersive, and volumetric video platforms. Existing 3DGS compression methods (e.g., HAC++, FlexGaussian, LP-3DGS) treat pruning, quantization, and entropy coding as separate stages and rely on hand-tuned heuristics (opacity thresholds, fixed bit-widths, SH truncation), limiting cross-scene generalization and preventing users from specifying a target rate or quality budget. We propose GETA-3DGS, to our knowledge the first end-to-end automatic joint structured pruning and quantization framework for 3DGS. Building on GETA for joint pruning-quantization of deep networks, we contribute: (i) a 3DGS-aware quantization-aware dependency graph (QADG) treating each Gaussian primitive as a group with five attribute sub-nodes and degree-aware SH sub-nodes; (ii) a render-aware saliency fusing transmittance-weighted contribution, screen-space gradient, and pixel coverage into a Gaussian-level importance score; and (iii) a heterogeneous per-attribute mixed-precision scheme co-optimized with structural sparsity under a projected partial saliency-guided (PPSG) descent guarantee. On Mip-NeRF 360, Tanks and Temples, and Deep Blending, GETA-3DGS operates directly on raw Gaussian primitives rather than a post-hoc anchor representation, delivering ~5x storage reduction over Vanilla 3DGS with no per-scene thresholds. Bit-width policy is the dominant rate-distortion lever: a uniform 6-bit cap costs up to -6.74 dB on view-dependent scenes versus our heterogeneous allocation, matching an information-theoretic reverse-water-filling analysis we develop. GETA-3DGS is complementary to existing codecs: entropy coding (HAC++, CompGS) is downstream, so the two can be composed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01995v1">From Concept to Capability: Evaluating 3D Gaussian Splatting for Synthetic Scene Editing in Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-05-03
      | 💬 Accepted in the 45th International Conference on Computer Safety, Reliability and Security (SafeComp 2026)
    </div>
    <details class="paper-abstract">
      The perception of an Autonomous Driving System (ADS) critically depends on relevant, comprehensive, and diverse datasets to ensure its safety while operating in the environment. Field data collection lacks completeness with respect to the list of rare but still possible safety-related scenarios needed for the development, verification, and validation of the ADS. 3D Gaussian Splatting (3DGS) has shown promising capabilities for the reconstruction and editing of scenes based on data collected by cameras and LiDAR sensors. However, the industrial fidelity evaluation of reconstructions is underexplored, which is crucial when employing such methods in safety-related systems, especially for ADS. This becomes more challenging as ADS operates in a dynamic, uncontrolled environment with limited viewpoints and often partially occluded objects. This paper addresses this gap by proposing and implementing a framework (Fig. 1) to systematically analyze the capabilities and limitations of 3DGS for use in the reconstruction of safety-related scenes. It focuses on the quality of reconstruction for vehicles and pedestrians, which are the two most critical object classes for ADS. Our findings provide industry insights into the fidelity degradation of reconstructions from multiple novel viewpoints, both lateral and longitudinal, enabling the integration of these methods into real-world industrial AD software development and testing pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06846v3">DynFOA: Generating First-Order Ambisonics with Conditional Diffusion for Dynamic and Acoustically Complex 360-Degree Videos</a></div>
    <div class="paper-meta">
      📅 2026-05-03
      | 💬 Revised version with updated content. A duplicate submission, arXiv:2604.02781, was previously submitted by mistake and has been withdrawn. This submission is the intended replacement of arXiv:2602.06846
    </div>
    <details class="paper-abstract">
      Spatial audio is crucial for immersive 360-degree video experiences, yet most 360-degree videos lack it due to the difficulty of capturing spatial audio during recording. Automatically generating spatial audio such as first-order ambisonics (FOA) from video therefore remains an important but challenging problem. In complex scenes, sound perception depends not only on sound source locations but also on scene geometry, materials, and dynamic interactions with the environment. However, existing approaches only rely on visual cues and fail to model dynamic sources and acoustic effects such as occlusion, reflections, and reverberation. To address these challenges, we propose DynFOA, a generative framework that synthesizes FOA from 360-degree videos by integrating dynamic scene reconstruction with conditional diffusion modeling. DynFOA analyzes the input video to detect and localize dynamic sound sources, estimate depth and semantics, and reconstruct scene geometry and materials using 3D Gaussian Splatting (3DGS). The reconstructed scene representation provides physically grounded features that capture acoustic interactions between sources, environment, and listener viewpoint. Conditioned on these features, a diffusion model generates spatial audio consistent with the scene dynamics and acoustic context. We introduce M2G-360, a dataset of 600 real-world clips divided into MoveSources, Multi-Source, and Geometry subsets for evaluating robustness under diverse conditions. Experiments show that DynFOA consistently outperforms existing methods in spatial accuracy, acoustic fidelity, distribution matching, and perceived immersive experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01736v1">Multi-Scale Gaussian-Language Map for Zero-shot Embodied Navigation and Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-03
      | 💬 Accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      Understanding the geometric and semantic structure of environments is essential for embodied navigation and reasoning. Existing semantic mapping methods trade off between explicit geometry and multi-scale semantics, and lack a native interface for large models, thus requiring additional training of feature projection for semantic alignment. To this end, we propose the multi-scale Gaussian-Language Map (GLMap), which introduces three key designs: (1) explicit geometry, (2) multi-scale semantics covering both instance and region concepts, and (3) a dual-modality interface where each semantic unit jointly stores a natural language description and a 3D Gaussian representation. The 3D Gaussians enable compact storage and fast rendering of task-relevant images via Gaussian splatting. To enable efficient incremental construction, we further propose a Gaussian Estimator that analytically derives Gaussian parameters from dense point clouds without gradient-based optimization. Experiments on ObjectNav, InstNav, and SQA tasks show that GLMap effectively enhances target navigation and contextual reasoning, while remaining compatible with large-model-based methods in a zero-shot manner. The code is available at https://github.com/sx-zhang/GLMap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.19703v3">High-Quality Spatial Reconstruction and Orthoimage Generation Using Efficient 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-03
    </div>
    <details class="paper-abstract">
      Highly accurate geometric precision and dense image features characterize True Digital Orthophoto Maps (TDOMs), which are in great demand for applications such as urban planning, infrastructure management, and environmental monitoring. Traditional TDOM generation methods need sophisticated processes, such as Digital Surface Models (DSM) and occlusion detection, which are computationally expensive and prone to errors. This work presents an alternative technique rooted in 2D Gaussian Splatting (2DGS), free of explicit DSM and occlusion detection. With depth map generation, spatial information for every pixel within the TDOM is retrieved and can reconstruct the scene with high precision. Divide-and-conquer strategy achieves excellent GS training and rendering with high-resolution TDOMs at a lower resource cost, which preserves higher quality of rendering on complex terrain and thin structure without a decrease in efficiency. Experimental results demonstrate the efficiency of large-scale scene reconstruction and high-precision terrain modeling. This approach provides accurate spatial data, which assists users in better planning and decision-making based on maps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.18713v2">SaLF: Sparse Local Fields for Multi-Sensor Rendering in Real-Time</a></div>
    <div class="paper-meta">
      📅 2026-05-02
      | 💬 ICRA 2026. Project page: https://waabi.ai/salf/
    </div>
    <details class="paper-abstract">
      High-fidelity sensor simulation of light-based sensors such as cameras and LiDARs is critical for safe and accurate autonomy testing. Neural radiance field (NeRF)-based methods that reconstruct sensor observations via ray-casting of implicit representations have demonstrated accurate simulation of driving scenes, but are slow to train and render, hampering scalability. 3D Gaussian Splatting (3DGS) has demonstrated faster training and rendering times through rasterization, but is primarily restricted to pinhole camera sensors, preventing usage for realistic multi-sensor autonomy evaluation. Moreover, both NeRF and 3DGS couple the representation with the rendering procedure (implicit networks for ray-based evaluation, particles for rasterization), preventing interoperability, which is key for general usage. In this work, we present Sparse Local Fields (SaLF), a novel volumetric representation that supports rasterization and raytracing for unified multi-sensor simulation. SaLF represents volumes as a sparse set of 3D voxel primitives, where each voxel is a local implicit field. SaLF has fast training ($<$30 min) and rendering capabilities (50+ FPS for camera and 600+ FPS for LiDAR), has adaptive pruning and densification to easily handle large scenes, and can support non-pinhole cameras and spinning LiDARs. We demonstrate that SaLF has similar realism as existing self-driving sensor simulation methods while improving efficiency and enhancing capabilities, enabling more scalable simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.15491v4">GSDeformer: Direct, Real-time and Extensible Cage-based Deformation for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-02
      | 💬 Project Page: https://jhuangbu.github.io/gsdeformer, Video: https://www.youtube.com/watch?v=-ecrj48-MqM
    </div>
    <details class="paper-abstract">
      We present GSDeformer, a method that enables cage-based deformation on 3D Gaussian Splatting (3DGS). Our approach bridges cage-based deformation and 3DGS by using a proxy point-cloud representation. This point cloud is generated from 3D Gaussians, and deformations applied to the point cloud are translated into transformations on the 3D Gaussians. To handle potential bending caused by deformation, we incorporate a splitting process to approximate it. Our method does not modify or extend the core architecture of 3D Gaussian Splatting, making it compatible with any trained vanilla 3DGS or its variants. Additionally, we automate cage construction for 3DGS and its variants using a render-and-reconstruct approach. Experiments demonstrate that GSDeformer delivers superior deformation results compared to existing methods, is robust under extreme deformations, requires no retraining for editing, runs in real-time, and can be extended to other 3DGS variants. Project Page: https://jhuangbu.github.io/gsdeformer/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01466v1">SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion</a></div>
    <div class="paper-meta">
      📅 2026-05-02
      | 💬 Accepted as a Spotlight paper at ICML 2026; camera-ready version
    </div>
    <details class="paper-abstract">
      Although multi-modal learning has advanced point cloud completion, the theoretical mechanisms remain unclear. Recent works attribute success to the connection between modalities, yet we identify that standard hard projection severs this connection: projecting a sparse point cloud onto the image plane yields an extremely sparse support, which hinders visual prior propagation, a failure mode we term Cross-Modal Entropy Collapse. To address this practical limitation, we propose SplAttN, which replaces hard projection with Differentiable Gaussian Splatting to produce a dense, continuous image-plane representation. By reformulating projection as continuous density estimation, SplAttN avoids collapsed sparse support, facilitates gradient flow, and improves cross-modal connection learnability. Extensive experiments show that SplAttN achieves state-of-the-art performance on PCN and ShapeNet-55/34. Crucially, we utilize the real-world KITTI benchmark as a stress test for multi-modal reliance. Counter-factual evaluation reveals that while baselines degenerate into unimodal template retrievers insensitive to visual removal, SplAttN maintains a robust dependency on visual cues, validating that our method establishes an effective cross-modal connection. Code is available at https://github.com/zay002/SplAttN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.12001v4">3D Gaussian Splatting against Moving Objects for High-Fidelity Street Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-02
    </div>
    <details class="paper-abstract">
      The accurate reconstruction of dynamic street scenes is critical for applications in autonomous driving, augmented reality, and virtual reality. Traditional methods relying on dense point clouds and triangular meshes struggle with moving objects, occlusions, and real-time processing constraints, limiting their effectiveness in complex urban environments. While multi-view stereo and neural radiance fields have advanced 3D reconstruction, they face challenges in computational efficiency and handling scene dynamics. This paper proposes a novel 3D Gaussian point distribution method for dynamic street scene reconstruction. Our approach introduces an adaptive transparency mechanism that eliminates moving objects while preserving high-fidelity static scene details. Additionally, iterative refinement of Gaussian point distribution enhances geometric accuracy and texture representation. We integrate directional encoding with spatial position optimization to optimize storage and rendering efficiency, reducing redundancy while maintaining scene integrity. Experimental results demonstrate that our method achieves high reconstruction quality, improved rendering performance, and adaptability in large-scale dynamic environments. These contributions establish a robust framework for real-time, high-precision 3D reconstruction, advancing the practicality of dynamic scene modeling across multiple applications. The source code for this work is available to the public at https://github.com/okic-ca/3dgs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01232v1">A Principled Approach for Creating High-fidelity Synthetic Demonstrations for Imitation Learning</a></div>
    <div class="paper-meta">
      📅 2026-05-02
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled visually realistic demonstration generation from a single expert trajectory and a short multi-view scan. However, existing 3DGS-based synthesis pipelines typically generate new motions using sampling-based planners or trajectory optimization, which often deviate substantially from the expert's demonstrated path. While such deviations may be acceptable for tasks insensitive to motion shape, they discard subtle spatial and temporal structure that is critical for contact-rich and shape-sensitive manipulation, causing increased demonstration diversity to harm downstream policy learning. We argue that demonstration synthesis should treat the expert trajectory as a strong prior. Building on this principle, we propose a framework that synthesizes diverse task demonstrations while explicitly preserving expert motion structure. We model the expert trajectory using Dynamic Movement Primitives (DMPs) and retarget it to new goals, object configurations, and viewpoints within a reconstructed 3DGS scene, yielding phase-consistent, shape-preserving motion by construction. To safely realize this expert-preserving diversity in cluttered scenes, we introduce an analytic obstacle-aware DMP formulation that operates directly on the continuous density field induced by the 3DGS representation. This enables collision avoidance while minimally perturbing the nominal expert motion, unifying photorealistic rendering and geometric reasoning without additional scene representations. We evaluate our approach on a Spot mobile manipulator across three manipulation tasks with increasing sensitivity to trajectory fidelity. Compared to planner- and optimization-based synthesis, our method produces trajectories with lower deviation and collision rates and yields higher task success when training diffusion-based visuomotor policies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01195v1">TAIL-Safe: Task-Agnostic Safety Monitoring for Imitation Learning Policies</a></div>
    <div class="paper-meta">
      📅 2026-05-02
    </div>
    <details class="paper-abstract">
      Recent imitation learning (IL) algorithms such as flow-matching and diffusion policies demonstrate remarkable performance in learning complex manipulation tasks. However, these policies often fail even when operating within their training distribution due to extreme sensitivity to initial conditions and irreducible approximation errors that lead to compounding drift. This makes it unsafe to deploy IL policies in the field where out-of-distribution scenarios are prevalent. A prerequisite for safe deployment is enabling the policy to determine whether it can execute a task the way it was learned from demonstrations. This paper presents TAIL-Safe, a principled approach to identify, for a trained IL policy, a safe set from where the policy empirically succeeds in completing the learned task. We propose a Lipschitz-continuous Q-value function that maps state-action pairs to a long-term safety score based on three short-term task-agnostic criteria: visibility, recognizability, and graspability. The zero-superlevel set of this function characterizes an empirical control invariant set over state-action pairs. When the nominal policy proposes an action outside this set, we apply a recovery mechanism inspired by Nagumo's theorem that uses gradient ascent to the Q-function to steer the policy back to safety. To learn this Q-function, we construct a high-fidelity digital twin using Gaussian Splatting that enables systematic collection of failure data without risk to physical hardware. Experiments with a Franka Emika robot demonstrate that flow-matching policies, which fail under run-time perturbations, achieve consistent task success when guided by the proposed TAIL-Safe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.00569v1">2D-SuGaR: Surface-Aware Gaussian Splatting for Geometrically Accurate Mesh Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-01
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for generating photorealistic renderings of a scene in real-time. However, the volumetric nature of 3DGS limits its ability to accurately capture surface geometry. To address this, 2D Gaussian Splatting (2DGS) was proposed to enable view-consistent and geometrically accurate surface reconstruction from multi-view images. However, 2DGS can be sensitive to the initialization of the Gaussian primitives. Reliance on Structure-from-Motion (SfM) initializations, which can produce poor estimates on challenging image sets, may lead to subpar results. In this work, we enhance 2DGS by incorporating monocular depth and normal priors to improve both geometric accuracy and robustness. We propose a depth-guided initialization strategy for Gaussians and introduce a clustering-based technique for pruning degenerate Gaussians. We evaluate our method on the DTU dataset, where it achieves state-of-the-art results in mesh reconstruction while preserving high-quality novel view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12895v3">High Dynamic Range 3D Gaussian Splatting via Luminance-Chromaticity Decomposition</a></div>
    <div class="paper-meta">
      📅 2026-05-01
    </div>
    <details class="paper-abstract">
      High Dynamic Range (HDR) 3D reconstruction is pivotal for professional content creation in filmmaking and virtual production. Existing methods typically rely on multi-exposure Low Dynamic Range (LDR) supervision to constrain the learning process within vast brightness spaces, resulting in complex, dual-branch architectures. This work explores the feasibility of learning HDR 3D models exclusively in the HDR data space to simplify model design. By analyzing 3D Gaussian Splatting (3DGS) for HDR imagery, we reveal that its failure stems from the limited capacity of Spherical Harmonics (SHs) to capture extreme radiance variations across views, often biasing towards high-radiance observations and underfitting. While increasing the maximum SH degree improves training fitting, it leads to severe overfitting and excessive parameter overhead. To address this, we propose \textit{Luminance--Chromaticity Decomposition Gaussian Splatting} (LCD-GS). By decoupling luminance and chromaticity into independent parameters, LCD-GS significantly enhances learning flexibility with minimal parameter increase (\textit{e.g.}, one extra scalar per primitive). Notably, LCD-GS maintains the original training and inference pipeline, requiring only a change in color representation. This explicit decomposition naturally enables primitive-level local and global luminance editing during inference. Extensive experiments on synthetic and real datasets demonstrate that LCD-GS consistently outperforms state-of-the-art methods in reconstruction fidelity and dynamic-range preservation even with a simpler, more efficient architecture, providing an elegant paradigm for professional-grade HDR 3D modeling. Code and datasets will be released.
    </details>
</div>
