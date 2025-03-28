# gaussian splatting - 2025_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16502v3">GSplatLoc: Grounding Keypoint Descriptors into 3D Gaussian Splatting for Improved Visual Localization</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Project website at https://gsplatloc.github.io/
    </div>
    <details class="paper-abstract">
      Although various visual localization approaches exist, such as scene coordinate regression and camera pose regression, these methods often struggle with optimization complexity or limited accuracy. To address these challenges, we explore the use of novel view synthesis techniques, particularly 3D Gaussian Splatting (3DGS), which enables the compact encoding of both 3D geometry and scene appearance. We propose a two-stage procedure that integrates dense and robust keypoint descriptors from the lightweight XFeat feature extractor into 3DGS, enhancing performance in both indoor and outdoor environments. The coarse pose estimates are directly obtained via 2D-3D correspondences between the 3DGS representation and query image descriptors. In the second stage, the initial pose estimate is refined by minimizing the rendering-based photometric warp loss. Benchmarking on widely used indoor and outdoor datasets demonstrates improvements over recent neural rendering-based localization methods, such as NeRFMatch and PNeRFLoc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06903v2">Synthetic Prior for Few-Shot Drivable Head Avatar Inversion</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Accepted to CVPR25 Website: https://zielon.github.io/synshot/
    </div>
    <details class="paper-abstract">
      We present SynShot, a novel method for the few-shot inversion of a drivable head avatar based on a synthetic prior. We tackle three major challenges. First, training a controllable 3D generative network requires a large number of diverse sequences, for which pairs of images and high-quality tracked meshes are not always available. Second, the use of real data is strictly regulated (e.g., under the General Data Protection Regulation, which mandates frequent deletion of models and data to accommodate a situation when a participant's consent is withdrawn). Synthetic data, free from these constraints, is an appealing alternative. Third, state-of-the-art monocular avatar models struggle to generalize to new views and expressions, lacking a strong prior and often overfitting to a specific viewpoint distribution. Inspired by machine learning models trained solely on synthetic data, we propose a method that learns a prior model from a large dataset of synthetic heads with diverse identities, expressions, and viewpoints. With few input images, SynShot fine-tunes the pretrained synthetic prior to bridge the domain gap, modeling a photorealistic head avatar that generalizes to novel expressions and viewpoints. We model the head avatar using 3D Gaussian splatting and a convolutional encoder-decoder that outputs Gaussian parameters in UV texture space. To account for the different modeling complexities over parts of the head (e.g., skin vs hair), we embed the prior with explicit control for upsampling the number of per-part primitives. Compared to SOTA monocular and GAN-based methods, SynShot significantly improves novel view and expression synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04545v3">Gaussian Eigen Models for Human Heads</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Accepted to CVPR25 Website: https://zielon.github.io/gem/
    </div>
    <details class="paper-abstract">
      Current personalized neural head avatars face a trade-off: lightweight models lack detail and realism, while high-quality, animatable avatars require significant computational resources, making them unsuitable for commodity devices. To address this gap, we introduce Gaussian Eigen Models (GEM), which provide high-quality, lightweight, and easily controllable head avatars. GEM utilizes 3D Gaussian primitives for representing the appearance combined with Gaussian splatting for rendering. Building on the success of mesh-based 3D morphable face models (3DMM), we define GEM as an ensemble of linear eigenbases for representing the head appearance of a specific subject. In particular, we construct linear bases to represent the position, scale, rotation, and opacity of the 3D Gaussians. This allows us to efficiently generate Gaussian primitives of a specific head shape by a linear combination of the basis vectors, only requiring a low-dimensional parameter vector that contains the respective coefficients. We propose to construct these linear bases (GEM) by distilling high-quality compute-intense CNN-based Gaussian avatar models that can generate expression-dependent appearance changes like wrinkles. These high-quality models are trained on multi-view videos of a subject and are distilled using a series of principal component analyses. Once we have obtained the bases that represent the animatable appearance space of a specific human, we learn a regressor that takes a single RGB image as input and predicts the low-dimensional parameter vector that corresponds to the shown facial expression. In a series of experiments, we compare GEM's self-reenactment and cross-person reenactment results to state-of-the-art 3D avatar methods, demonstrating GEM's higher visual quality and better generalization to new expressions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03911v2">Multi-View Pose-Agnostic Change Localization with Zero Labels</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Accepted at CVPR 2025
    </div>
    <details class="paper-abstract">
      Autonomous agents often require accurate methods for detecting and localizing changes in their environment, particularly when observations are captured from unconstrained and inconsistent viewpoints. We propose a novel label-free, pose-agnostic change detection method that integrates information from multiple viewpoints to construct a change-aware 3D Gaussian Splatting (3DGS) representation of the scene. With as few as 5 images of the post-change scene, our approach can learn an additional change channel in a 3DGS and produce change masks that outperform single-view techniques. Our change-aware 3D scene representation additionally enables the generation of accurate change masks for unseen viewpoints. Experimental results demonstrate state-of-the-art performance in complex multi-object scenes, achieving a 1.7x and 1.5x improvement in Mean Intersection Over Union and F1 score respectively over other baselines. We also contribute a new real-world dataset to benchmark change detection in diverse challenging scenes in the presence of lighting variations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12552v2">MTGS: Multi-Traversal Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Multi-traversal data, commonly collected through daily commutes or by self-driving fleets, provides multiple viewpoints for scene reconstruction within a road block. This data offers significant potential for high-quality novel view synthesis, which is crucial for applications such as autonomous vehicle simulators. However, inherent challenges in multi-traversal data often result in suboptimal reconstruction quality, including variations in appearance and the presence of dynamic objects. To address these issues, we propose Multi-Traversal Gaussian Splatting (MTGS), a novel approach that reconstructs high-quality driving scenes from arbitrarily collected multi-traversal data by modeling a shared static geometry while separately handling dynamic elements and appearance variations. Our method employs a multi-traversal dynamic scene graph with a shared static node and traversal-specific dynamic nodes, complemented by color correction nodes with learnable spherical harmonics coefficient residuals. This approach enables high-fidelity novel view synthesis and provides flexibility to navigate any viewpoint. We conduct extensive experiments on a large-scale driving dataset, nuPlan, with multi-traversal data. Our results demonstrate that MTGS improves LPIPS by 23.5% and geometry accuracy by 46.3% compared to single-traversal baselines. The code and data would be available to the public.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15908v1">Enhancing Close-up Novel View Synthesis via Pseudo-labeling</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Accepted by AAAI 2025
    </div>
    <details class="paper-abstract">
      Recent methods, such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have demonstrated remarkable capabilities in novel view synthesis. However, despite their success in producing high-quality images for viewpoints similar to those seen during training, they struggle when generating detailed images from viewpoints that significantly deviate from the training set, particularly in close-up views. The primary challenge stems from the lack of specific training data for close-up views, leading to the inability of current methods to render these views accurately. To address this issue, we introduce a novel pseudo-label-based learning strategy. This approach leverages pseudo-labels derived from existing training data to provide targeted supervision across a wide range of close-up viewpoints. Recognizing the absence of benchmarks for this specific challenge, we also present a new dataset designed to assess the effectiveness of both current and future methods in this area. Our extensive experiments demonstrate the efficacy of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20031v3">MG-SLAM: Structure Gaussian Splatting SLAM with Manhattan World Hypothesis</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Gaussian Splatting SLAMs have made significant advancements in improving the efficiency and fidelity of real-time reconstructions. However, these systems often encounter incomplete reconstructions in complex indoor environments, characterized by substantial holes due to unobserved geometry caused by obstacles or limited view angles. To address this challenge, we present Manhattan Gaussian SLAM, an RGB-D system that leverages the Manhattan World hypothesis to enhance geometric accuracy and completeness. By seamlessly integrating fused line segments derived from structured scenes, our method ensures robust tracking in textureless indoor areas. Moreover, The extracted lines and planar surface assumption allow strategic interpolation of new Gaussians in regions of missing geometry, enabling efficient scene completion. Extensive experiments conducted on both synthetic and real-world scenes demonstrate that these advancements enable our method to achieve state-of-the-art performance, marking a substantial improvement in the capabilities of Gaussian SLAM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15855v1">VideoRFSplat: Direct Scene-Level Text-to-3D Gaussian Splatting Generation with Flexible Pose and Multi-View Joint Modeling</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Project page: https://gohyojun15.github.io/VideoRFSplat/
    </div>
    <details class="paper-abstract">
      We propose VideoRFSplat, a direct text-to-3D model leveraging a video generation model to generate realistic 3D Gaussian Splatting (3DGS) for unbounded real-world scenes. To generate diverse camera poses and unbounded spatial extent of real-world scenes, while ensuring generalization to arbitrary text prompts, previous methods fine-tune 2D generative models to jointly model camera poses and multi-view images. However, these methods suffer from instability when extending 2D generative models to joint modeling due to the modality gap, which necessitates additional models to stabilize training and inference. In this work, we propose an architecture and a sampling strategy to jointly model multi-view images and camera poses when fine-tuning a video generation model. Our core idea is a dual-stream architecture that attaches a dedicated pose generation model alongside a pre-trained video generation model via communication blocks, generating multi-view images and camera poses through separate streams. This design reduces interference between the pose and image modalities. Additionally, we propose an asynchronous sampling strategy that denoises camera poses faster than multi-view images, allowing rapidly denoised poses to condition multi-view generation, reducing mutual ambiguity and enhancing cross-modal consistency. Trained on multiple large-scale real-world datasets (RealEstate10K, MVImgNet, DL3DV-10K, ACID), VideoRFSplat outperforms existing text-to-3D direct generation methods that heavily depend on post-hoc refinement via score distillation sampling, achieving superior results without such refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15835v1">BARD-GS: Blur-Aware Reconstruction of Dynamic Scenes via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 CVPR2025. Project page at https://vulab-ai.github.io/BARD-GS/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown remarkable potential for static scene reconstruction, and recent advancements have extended its application to dynamic scenes. However, the quality of reconstructions depends heavily on high-quality input images and precise camera poses, which are not that trivial to fulfill in real-world scenarios. Capturing dynamic scenes with handheld monocular cameras, for instance, typically involves simultaneous movement of both the camera and objects within a single exposure. This combined motion frequently results in image blur that existing methods cannot adequately handle. To address these challenges, we introduce BARD-GS, a novel approach for robust dynamic scene reconstruction that effectively handles blurry inputs and imprecise camera poses. Our method comprises two main components: 1) camera motion deblurring and 2) object motion deblurring. By explicitly decomposing motion blur into camera motion blur and object motion blur and modeling them separately, we achieve significantly improved rendering results in dynamic regions. In addition, we collect a real-world motion blur dataset of dynamic scenes to evaluate our approach. Extensive experiments demonstrate that BARD-GS effectively reconstructs high-quality dynamic scenes under realistic conditions, significantly outperforming existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16747v1">SAGE: Semantic-Driven Adaptive Gaussian Splatting in Extended Reality</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has significantly improved the efficiency and realism of three-dimensional scene visualization in several applications, ranging from robotics to eXtended Reality (XR). This work presents SAGE (Semantic-Driven Adaptive Gaussian Splatting in Extended Reality), a novel framework designed to enhance the user experience by dynamically adapting the Level of Detail (LOD) of different 3DGS objects identified via a semantic segmentation. Experimental results demonstrate how SAGE effectively reduces memory and computational overhead while keeping a desired target visual quality, thus providing a powerful optimization for interactive XR applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01846v3">UVGS: Reimagining Unstructured 3D Gaussian Splatting using UV Mapping</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 https://aashishrai3799.github.io/uvgs
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated superior quality in modeling 3D objects and scenes. However, generating 3DGS remains challenging due to their discrete, unstructured, and permutation-invariant nature. In this work, we present a simple yet effective method to overcome these challenges. We utilize spherical mapping to transform 3DGS into a structured 2D representation, termed UVGS. UVGS can be viewed as multi-channel images, with feature dimensions as a concatenation of Gaussian attributes such as position, scale, color, opacity, and rotation. We further find that these heterogeneous features can be compressed into a lower-dimensional (e.g., 3-channel) shared feature space using a carefully designed multi-branch network. The compressed UVGS can be treated as typical RGB images. Remarkably, we discover that typical VAEs trained with latent diffusion models can directly generalize to this new representation without additional training. Our novel representation makes it effortless to leverage foundational 2D models, such as diffusion models, to directly model 3DGS. Additionally, one can simply increase the 2D UV resolution to accommodate more Gaussians, making UVGS a scalable solution compared to typical 3D backbones. This approach immediately unlocks various novel generation applications of 3DGS by inherently utilizing the already developed superior 2D generation capabilities. In our experiments, we demonstrate various unconditional, conditional generation, and inpainting applications of 3DGS based on diffusion models, which were previously non-trivial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16710v1">4D Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Simultaneously localizing camera poses and constructing Gaussian radiance fields in dynamic scenes establish a crucial bridge between 2D images and the 4D real world. Instead of removing dynamic objects as distractors and reconstructing only static environments, this paper proposes an efficient architecture that incrementally tracks camera poses and establishes the 4D Gaussian radiance fields in unknown scenarios by using a sequence of RGB-D images. First, by generating motion masks, we obtain static and dynamic priors for each pixel. To eliminate the influence of static scenes and improve the efficiency on learning the motion of dynamic objects, we classify the Gaussian primitives into static and dynamic Gaussian sets, while the sparse control points along with an MLP is utilized to model the transformation fields of the dynamic Gaussians. To more accurately learn the motion of dynamic Gaussians, a novel 2D optical flow map reconstruction algorithm is designed to render optical flows of dynamic objects between neighbor images, which are further used to supervise the 4D Gaussian radiance fields along with traditional photometric and geometric constraints. In experiments, qualitative and quantitative evaluation results show that the proposed method achieves robust tracking and high-quality view synthesis performance in real-world environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16681v1">GauRast: Enhancing GPU Triangle Rasterizers to Accelerate 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 DAC 2025
    </div>
    <details class="paper-abstract">
      3D intelligence leverages rich 3D features and stands as a promising frontier in AI, with 3D rendering fundamental to many downstream applications. 3D Gaussian Splatting (3DGS), an emerging high-quality 3D rendering method, requires significant computation, making real-time execution on existing GPU-equipped edge devices infeasible. Previous efforts to accelerate 3DGS rely on dedicated accelerators that require substantial integration overhead and hardware costs. This work proposes an acceleration strategy that leverages the similarities between the 3DGS pipeline and the highly optimized conventional graphics pipeline in modern GPUs. Instead of developing a dedicated accelerator, we enhance existing GPU rasterizer hardware to efficiently support 3DGS operations. Our results demonstrate a 23$\times$ increase in processing speed and a 24$\times$ reduction in energy consumption, with improvements yielding 6$\times$ faster end-to-end runtime for the original 3DGS algorithm and 4$\times$ for the latest efficiency-improved pipeline, achieving 24 FPS and 46 FPS respectively. These enhancements incur only a minimal area overhead of 0.2\% relative to the entire SoC chip area, underscoring the practicality and efficiency of our approach for enabling 3DGS rendering on resource-constrained platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05040v2">GaussRender: Learning 3D Occupancy with Gaussian Rendering</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Understanding the 3D geometry and semantics of driving scenes is critical for safe autonomous driving. Recent advances in 3D occupancy prediction have improved scene representation but often suffer from spatial inconsistencies, leading to floating artifacts and poor surface localization. Existing voxel-wise losses (e.g., cross-entropy) fail to enforce geometric coherence. In this paper, we propose GaussRender, a module that improves 3D occupancy learning by enforcing projective consistency. Our key idea is to project both predicted and ground-truth 3D occupancy into 2D camera views, where we apply supervision. Our method penalizes 3D configurations that produce inconsistent 2D projections, thereby enforcing a more coherent 3D structure. To achieve this efficiently, we leverage differentiable rendering with Gaussian splatting. GaussRender seamlessly integrates with existing architectures while maintaining efficiency and requiring no inference-time modifications. Extensive evaluations on multiple benchmarks (SurroundOcc-nuScenes, Occ3D-nuScenes, SSCBench-KITTI360) demonstrate that GaussRender significantly improves geometric fidelity across various 3D occupancy models (TPVFormer, SurroundOcc, Symphonies), achieving state-of-the-art results, particularly on surface-sensitive metrics. The code is open-sourced at https://github.com/valeoai/GaussRender.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08352v2">Mitigating Ambiguities in 3D Classification with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 Accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      3D classification with point cloud input is a fundamental problem in 3D vision. However, due to the discrete nature and the insufficient material description of point cloud representations, there are ambiguities in distinguishing wire-like and flat surfaces, as well as transparent or reflective objects. To address these issues, we propose Gaussian Splatting (GS) point cloud-based 3D classification. We find that the scale and rotation coefficients in the GS point cloud help characterize surface types. Specifically, wire-like surfaces consist of multiple slender Gaussian ellipsoids, while flat surfaces are composed of a few flat Gaussian ellipsoids. Additionally, the opacity in the GS point cloud represents the transparency characteristics of objects. As a result, ambiguities in point cloud-based 3D classification can be mitigated utilizing GS point cloud as input. To verify the effectiveness of GS point cloud input, we construct the first real-world GS point cloud dataset in the community, which includes 20 categories with 200 objects in each category. Experiments not only validate the superiority of GS point cloud input, especially in distinguishing ambiguous objects, but also demonstrate the generalization ability across different classification methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16898v2">MonoGSDF: Exploring Monocular Geometric Cues for Gaussian Splatting-Guided Implicit Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Accurate meshing from monocular images remains a key challenge in 3D vision. While state-of-the-art 3D Gaussian Splatting (3DGS) methods excel at synthesizing photorealistic novel views through rasterization-based rendering, their reliance on sparse, explicit primitives severely limits their ability to recover watertight and topologically consistent 3D surfaces.We introduce MonoGSDF, a novel method that couples Gaussian-based primitives with a neural Signed Distance Field (SDF) for high-quality reconstruction. During training, the SDF guides Gaussians' spatial distribution, while at inference, Gaussians serve as priors to reconstruct surfaces, eliminating the need for memory-intensive Marching Cubes. To handle arbitrary-scale scenes, we propose a scaling strategy for robust generalization. A multi-resolution training scheme further refines details and monocular geometric cues from off-the-shelf estimators enhance reconstruction quality. Experiments on real-world datasets show MonoGSDF outperforms prior methods while maintaining efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19459v2">ArtGS: Building Interactable Replicas of Complex Articulated Objects via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Building articulated objects is a key challenge in computer vision. Existing methods often fail to effectively integrate information across different object states, limiting the accuracy of part-mesh reconstruction and part dynamics modeling, particularly for complex multi-part articulated objects. We introduce ArtGS, a novel approach that leverages 3D Gaussians as a flexible and efficient representation to address these issues. Our method incorporates canonical Gaussians with coarse-to-fine initialization and updates for aligning articulated part information across different object states, and employs a skinning-inspired part dynamics modeling module to improve both part-mesh reconstruction and articulation learning. Extensive experiments on both synthetic and real-world datasets, including a new benchmark for complex multi-part objects, demonstrate that ArtGS achieves state-of-the-art performance in joint parameter estimation and part mesh reconstruction. Our approach significantly improves reconstruction quality and efficiency, especially for multi-part articulated objects. Additionally, we provide comprehensive analyses of our design choices, validating the effectiveness of each component to highlight potential areas for future improvement. Our work is made publicly available at: https://articulate-gs.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14475v1">Optimized 3D Gaussian Splatting using Coarse-to-Fine Image Frequency Modulation</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      The field of Novel View Synthesis has been revolutionized by 3D Gaussian Splatting (3DGS), which enables high-quality scene reconstruction that can be rendered in real-time. 3DGS-based techniques typically suffer from high GPU memory and disk storage requirements which limits their practical application on consumer-grade devices. We propose Opti3DGS, a novel frequency-modulated coarse-to-fine optimization framework that aims to minimize the number of Gaussian primitives used to represent a scene, thus reducing memory and storage demands. Opti3DGS leverages image frequency modulation, initially enforcing a coarse scene representation and progressively refining it by modulating frequency details in the training images. On the baseline 3DGS, we demonstrate an average reduction of 62% in Gaussians, a 40% reduction in the training GPU memory requirements and a 20% reduction in optimization time without sacrificing the visual quality. Furthermore, we show that our method integrates seamlessly with many 3DGS-based techniques, consistently reducing the number of Gaussian primitives while maintaining, and often improving, visual quality. Additionally, Opti3DGS inherently produces a level-of-detail scene representation at no extra cost, a natural byproduct of the optimization pipeline. Results and code will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02009v2">Morpheus: Text-Driven 3D Gaussian Splat Shape and Color Stylization</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Exploring real-world spaces using novel-view synthesis is fun, and reimagining those worlds in a different style adds another layer of excitement. Stylized worlds can also be used for downstream tasks where there is limited training data and a need to expand a model's training distribution. Most current novel-view synthesis stylization techniques lack the ability to convincingly change geometry. This is because any geometry change requires increased style strength which is often capped for stylization stability and consistency. In this work, we propose a new autoregressive 3D Gaussian Splatting stylization method. As part of this method, we contribute a new RGBD diffusion model that allows for strength control over appearance and shape stylization. To ensure consistency across stylized frames, we use a combination of novel depth-guided cross attention, feature injection, and a Warp ControlNet conditioned on composite frames for guiding the stylization of new frames. We validate our method via extensive qualitative results, quantitative experiments, and a user study. Code online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14274v1">Improving Adaptive Density Control for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become one of the most influential works in the past year. Due to its efficient and high-quality novel view synthesis capabilities, it has been widely adopted in many research fields and applications. Nevertheless, 3DGS still faces challenges to properly manage the number of Gaussian primitives that are used during scene reconstruction. Following the adaptive density control (ADC) mechanism of 3D Gaussian Splatting, new Gaussians in under-reconstructed regions are created, while Gaussians that do not contribute to the rendering quality are pruned. We observe that those criteria for densifying and pruning Gaussians can sometimes lead to worse rendering by introducing artifacts. We especially observe under-reconstructed background or overfitted foreground regions. To encounter both problems, we propose three new improvements to the adaptive density control mechanism. Those include a correction for the scene extent calculation that does not only rely on camera positions, an exponentially ascending gradient threshold to improve training convergence, and significance-aware pruning strategy to avoid background artifacts. With these adaptions, we show that the rendering quality improves while using the same number of Gaussians primitives. Furthermore, with our improvements, the training converges considerably faster, allowing for more than twice as fast training times while yielding better quality than 3DGS. Finally, our contributions are easily compatible with most existing derivative works of 3DGS making them relevant for future works.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14198v1">RoGSplat: Learning Robust Generalizable Human Gaussian Splatting from Sparse Multi-View Images</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 Accepted to CVPR2025
    </div>
    <details class="paper-abstract">
      This paper presents RoGSplat, a novel approach for synthesizing high-fidelity novel views of unseen human from sparse multi-view images, while requiring no cumbersome per-subject optimization. Unlike previous methods that typically struggle with sparse views with few overlappings and are less effective in reconstructing complex human geometry, the proposed method enables robust reconstruction in such challenging conditions. Our key idea is to lift SMPL vertices to dense and reliable 3D prior points representing accurate human body geometry, and then regress human Gaussian parameters based on the points. To account for possible misalignment between SMPL model and images, we propose to predict image-aligned 3D prior points by leveraging both pixel-level features and voxel-level features, from which we regress the coarse Gaussians. To enhance the ability to capture high-frequency details, we further render depth maps from the coarse 3D Gaussians to help regress fine-grained pixel-wise Gaussians. Experiments on several benchmark datasets demonstrate that our method outperforms state-of-the-art methods in novel view synthesis and cross-dataset generalization. Our code is available at https://github.com/iSEE-Laboratory/RoGSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14171v1">Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      We introduce an image upscaling technique tailored for 3D Gaussian Splatting (3DGS) on lightweight GPUs. Compared to 3DGS, it achieves significantly higher rendering speeds and reduces artifacts commonly observed in 3DGS reconstructions. Our technique upscales low-resolution 3DGS renderings with a marginal increase in cost by directly leveraging the analytical image gradients of Gaussians for gradient-based bicubic spline interpolation. The technique is agnostic to the specific 3DGS implementation, achieving novel view synthesis at rates 3x-4x higher than the baseline implementation. Through extensive experiments on multiple datasets, we showcase the performance improvements and high reconstruction fidelity attainable with gradient-aware upscaling of 3DGS images. We further demonstrate the integration of gradient-aware upscaling into the gradient-based optimization of a 3DGS model and analyze its effects on reconstruction quality and performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04013v2">3D-LMVIC: Learning-based Multi-View Image Coding with 3D Gaussian Geometric Priors</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 17 pages, 10 figures, conference
    </div>
    <details class="paper-abstract">
      Existing multi-view image compression methods often rely on 2D projection-based similarities between views to estimate disparities. While effective for small disparities, such as those in stereo images, these methods struggle with the more complex disparities encountered in wide-baseline multi-camera systems, commonly found in virtual reality and autonomous driving applications. To address this limitation, we propose 3D-LMVIC, a novel learning-based multi-view image compression framework that leverages 3D Gaussian Splatting to derive geometric priors for accurate disparity estimation. Furthermore, we introduce a depth map compression model to minimize geometric redundancy across views, along with a multi-view sequence ordering strategy based on a defined distance measure between views to enhance correlations between adjacent views. Experimental results demonstrate that 3D-LMVIC achieves superior performance compared to both traditional and learning-based methods. Additionally, it significantly improves disparity estimation accuracy over existing two-view approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14029v1">Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 CVPR 2025. The code is publicly available at this https URL (https://github.com/Runsong123/Unified-Lift)
    </div>
    <details class="paper-abstract">
      Lifting multi-view 2D instance segmentation to a radiance field has proven to be effective to enhance 3D understanding. Existing methods rely on direct matching for end-to-end lifting, yielding inferior results; or employ a two-stage solution constrained by complex pre- or post-processing. In this work, we design a new end-to-end object-aware lifting approach, named Unified-Lift that provides accurate 3D segmentation based on the 3D Gaussian representation. To start, we augment each Gaussian point with an additional Gaussian-level feature learned using a contrastive loss to encode instance information. Importantly, we introduce a learnable object-level codebook to account for individual objects in the scene for an explicit object-level understanding and associate the encoded object-level features with the Gaussian-level point features for segmentation predictions. While promising, achieving effective codebook learning is non-trivial and a naive solution leads to degraded performance. Therefore, we formulate the association learning module and the noisy label filtering module for effective and robust codebook learning. We conduct experiments on three benchmarks: LERF-Masked, Replica, and Messy Rooms datasets. Both qualitative and quantitative results manifest that our Unified-Lift clearly outperforms existing methods in terms of segmentation quality and time efficiency. The code is publicly available at \href{https://github.com/Runsong123/Unified-Lift}{https://github.com/Runsong123/Unified-Lift}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12335v2">GS-I$^{3}$: Gaussian Splatting for Surface Reconstruction from Illumination-Inconsistent Images</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 Comments: This work has been submitted to the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025) for possible publication
    </div>
    <details class="paper-abstract">
      Accurate geometric surface reconstruction, providing essential environmental information for navigation and manipulation tasks, is critical for enabling robotic self-exploration and interaction. Recently, 3D Gaussian Splatting (3DGS) has gained significant attention in the field of surface reconstruction due to its impressive geometric quality and computational efficiency. While recent relevant advancements in novel view synthesis under inconsistent illumination using 3DGS have shown promise, the challenge of robust surface reconstruction under such conditions is still being explored. To address this challenge, we propose a method called GS-3I. Specifically, to mitigate 3D Gaussian optimization bias caused by underexposed regions in single-view images, based on Convolutional Neural Network (CNN), a tone mapping correction framework is introduced. Furthermore, inconsistent lighting across multi-view images, resulting from variations in camera settings and complex scene illumination, often leads to geometric constraint mismatches and deviations in the reconstructed surface. To overcome this, we propose a normal compensation mechanism that integrates reference normals extracted from single-view image with normals computed from multi-view observations to effectively constrain geometric inconsistencies. Extensive experimental evaluations demonstrate that GS-3I can achieve robust and accurate surface reconstruction across complex illumination scenarios, highlighting its effectiveness and versatility in this critical challenge. https://github.com/TFwang-9527/GS-3I
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13948v1">Light4GS: Lightweight Compact 4D Gaussian Splatting Generation via Context Model</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as an efficient and high-fidelity paradigm for novel view synthesis. To adapt 3DGS for dynamic content, deformable 3DGS incorporates temporally deformable primitives with learnable latent embeddings to capture complex motions. Despite its impressive performance, the high-dimensional embeddings and vast number of primitives lead to substantial storage requirements. In this paper, we introduce a \textbf{Light}weight \textbf{4}D\textbf{GS} framework, called Light4GS, that employs significance pruning with a deep context model to provide a lightweight storage-efficient dynamic 3DGS representation. The proposed Light4GS is based on 4DGS that is a typical representation of deformable 3DGS. Specifically, our framework is built upon two core components: (1) a spatio-temporal significance pruning strategy that eliminates over 64\% of the deformable primitives, followed by an entropy-constrained spherical harmonics compression applied to the remainder; and (2) a deep context model that integrates intra- and inter-prediction with hyperprior into a coarse-to-fine context structure to enable efficient multiscale latent embedding compression. Our approach achieves over 120x compression and increases rendering FPS up to 20\% compared to the baseline 4DGS, and also superior to frame-wise state-of-the-art 3DGS compression methods, revealing the effectiveness of our Light4GS in terms of both intra- and inter-prediction methods without sacrificing rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12001v2">3D Gaussian Splatting against Moving Objects for High-Fidelity Street Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      The accurate reconstruction of dynamic street scenes is critical for applications in autonomous driving, augmented reality, and virtual reality. Traditional methods relying on dense point clouds and triangular meshes struggle with moving objects, occlusions, and real-time processing constraints, limiting their effectiveness in complex urban environments. While multi-view stereo and neural radiance fields have advanced 3D reconstruction, they face challenges in computational efficiency and handling scene dynamics. This paper proposes a novel 3D Gaussian point distribution method for dynamic street scene reconstruction. Our approach introduces an adaptive transparency mechanism that eliminates moving objects while preserving high-fidelity static scene details. Additionally, iterative refinement of Gaussian point distribution enhances geometric accuracy and texture representation. We integrate directional encoding with spatial position optimization to optimize storage and rendering efficiency, reducing redundancy while maintaining scene integrity. Experimental results demonstrate that our method achieves high reconstruction quality, improved rendering performance, and adaptability in large-scale dynamic environments. These contributions establish a robust framework for real-time, high-precision 3D reconstruction, advancing the practicality of dynamic scene modeling across multiple applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14736v1">HandSplat: Embedding-Driven Gaussian Splatting for High-Fidelity Hand Rendering</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Existing 3D Gaussian Splatting (3DGS) methods for hand rendering rely on rigid skeletal motion with an oversimplified non-rigid motion model, which fails to capture fine geometric and appearance details. Additionally, they perform densification based solely on per-point gradients and process poses independently, ignoring spatial and temporal correlations. These limitations lead to geometric detail loss, temporal instability, and inefficient point distribution. To address these issues, we propose HandSplat, a novel Gaussian Splatting-based framework that enhances both fidelity and stability for hand rendering. To improve fidelity, we extend standard 3DGS attributes with implicit geometry and appearance embeddings for finer non-rigid motion modeling while preserving the static hand characteristic modeled by original 3DGS attributes. Additionally, we introduce a local gradient-aware densification strategy that dynamically refines Gaussian density in high-variation regions. To improve stability, we incorporate pose-conditioned attribute regularization to encourage attribute consistency across similar poses, mitigating temporal artifacts. Extensive experiments on InterHand2.6M demonstrate that HandSplat surpasses existing methods in fidelity and stability while achieving real-time performance. We will release the code and pre-trained models upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14698v1">SplatVoxel: History-Aware Novel View Streaming without Temporal Training</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      We study the problem of novel view streaming from sparse-view videos, which aims to generate a continuous sequence of high-quality, temporally consistent novel views as new input frames arrive. However, existing novel view synthesis methods struggle with temporal coherence and visual fidelity, leading to flickering and inconsistency. To address these challenges, we introduce history-awareness, leveraging previous frames to reconstruct the scene and improve quality and stability. We propose a hybrid splat-voxel feed-forward scene reconstruction approach that combines Gaussian Splatting to propagate information over time, with a hierarchical voxel grid for temporal fusion. Gaussian primitives are efficiently warped over time using a motion graph that extends 2D tracking models to 3D motion, while a sparse voxel transformer integrates new temporal observations in an error-aware manner. Crucially, our method does not require training on multi-view video datasets, which are currently limited in size and diversity, and can be directly applied to sparse-view video streams in a history-aware manner at inference time. Our approach achieves state-of-the-art performance in both static and streaming scene reconstruction, effectively reducing temporal artifacts and visual artifacts while running at interactive rates (15 fps with 350ms delay) on a single H100 GPU. Project Page: https://19reborn.github.io/SplatVoxel/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19895v5">GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 This paper is accepted by the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR), 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently created impressive 3D assets for various applications. However, considering security, capacity, invisibility, and training efficiency, the copyright of 3DGS assets is not well protected as existing watermarking methods are unsuited for its rendering pipeline. In this paper, we propose GuardSplat, an innovative and efficient framework for watermarking 3DGS assets. Specifically, 1) We propose a CLIP-guided pipeline for optimizing the message decoder with minimal costs. The key objective is to achieve high-accuracy extraction by leveraging CLIP's aligning capability and rich representations, demonstrating exceptional capacity and efficiency. 2) We tailor a Spherical-Harmonic-aware (SH-aware) Message Embedding module for 3DGS, seamlessly embedding messages into the SH features of each 3D Gaussian while preserving the original 3D structure. This enables watermarking 3DGS assets with minimal fidelity trade-offs and prevents malicious users from removing the watermarks from the model files, meeting the demands for invisibility and security. 3) We present an Anti-distortion Message Extraction module to improve robustness against various distortions. Experiments demonstrate that GuardSplat outperforms state-of-the-art and achieves fast optimization speed. Project page is at https://narcissusex.github.io/GuardSplat, and Code is at https://github.com/NarcissusEx/GuardSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13272v1">Generative Gaussian Splatting: Generating 3D Scenes with Video Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Synthesizing consistent and photorealistic 3D scenes is an open problem in computer vision. Video diffusion models generate impressive videos but cannot directly synthesize 3D representations, i.e., lack 3D consistency in the generated sequences. In addition, directly training generative 3D models is challenging due to a lack of 3D training data at scale. In this work, we present Generative Gaussian Splatting (GGS) -- a novel approach that integrates a 3D representation with a pre-trained latent video diffusion model. Specifically, our model synthesizes a feature field parameterized via 3D Gaussian primitives. The feature field is then either rendered to feature maps and decoded into multi-view images, or directly upsampled into a 3D radiance field. We evaluate our approach on two common benchmark datasets for scene synthesis, RealEstate10K and ScanNet+, and find that our proposed GGS model significantly improves both the 3D consistency of the generated multi-view images, and the quality of the generated 3D scenes over all relevant baselines. Compared to a similar model without 3D representation, GGS improves FID on the generated 3D scenes by ~20% on both RealEstate10K and ScanNet+. Project page: https://katjaschwarz.github.io/ggs/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13176v1">DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Reconstructing clean, distractor-free 3D scenes from real-world captures remains a significant challenge, particularly in highly dynamic and cluttered settings such as egocentric videos. To tackle this problem, we introduce DeGauss, a simple and robust self-supervised framework for dynamic scene reconstruction based on a decoupled dynamic-static Gaussian Splatting design. DeGauss models dynamic elements with foreground Gaussians and static content with background Gaussians, using a probabilistic mask to coordinate their composition and enable independent yet complementary optimization. DeGauss generalizes robustly across a wide range of real-world scenarios, from casual image collections to long, dynamic egocentric videos, without relying on complex heuristics or extensive supervision. Experiments on benchmarks including NeRF-on-the-go, ADT, AEA, Hot3D, and EPIC-Fields demonstrate that DeGauss consistently outperforms existing methods, establishing a strong baseline for generalizable, distractor-free 3D reconstructionin highly dynamic, interaction-rich environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13086v1">Gaussian On-the-Fly Splatting: A Progressive Framework for Robust Near Real-Time 3DGS Optimization</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves high-fidelity rendering with fast real-time performance, but existing methods rely on offline training after full Structure-from-Motion (SfM) processing. In contrast, this work introduces On-the-Fly GS, a progressive framework enabling near real-time 3DGS optimization during image capture. As each image arrives, its pose and sparse points are updated via on-the-fly SfM, and newly optimized Gaussians are immediately integrated into the 3DGS field. We propose a progressive local optimization strategy to prioritize new images and their neighbors by their corresponding overlapping relationship, allowing the new image and its overlapping images to get more training. To further stabilize training across old and new images, an adaptive learning rate schedule balances the iterations and the learning rate. Moreover, to maintain overall quality of the 3DGS field, an efficient global optimization scheme prevents overfitting to the newly added images. Experiments on multiple benchmark datasets show that our On-the-Fly GS reduces training time significantly, optimizing each new image in seconds with minimal rendering loss, offering the first practical step toward rapid, progressive 3DGS reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12663v2">LAGA: Layered 3D Avatar Generation and Customization via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Creating and customizing a 3D clothed avatar from textual descriptions is a critical and challenging task. Traditional methods often treat the human body and clothing as inseparable, limiting users' ability to freely mix and match garments. In response to this limitation, we present LAyered Gaussian Avatar (LAGA), a carefully designed framework enabling the creation of high-fidelity decomposable avatars with diverse garments. By decoupling garments from avatar, our framework empowers users to conviniently edit avatars at the garment level. Our approach begins by modeling the avatar using a set of Gaussian points organized in a layered structure, where each layer corresponds to a specific garment or the human body itself. To generate high-quality garments for each layer, we introduce a coarse-to-fine strategy for diverse garment generation and a novel dual-SDS loss function to maintain coherence between the generated garments and avatar components, including the human body and other garments. Moreover, we introduce three regularization losses to guide the movement of Gaussians for garment transfer, allowing garments to be freely transferred to various avatars. Extensive experimentation demonstrates that our approach surpasses existing methods in the generation of 3D clothed humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03714v2">MoDec-GS: Global-to-Local Motion Decomposition and Temporal Interval Adjustment for Compact Dynamic 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 The last two authors are co-corresponding authors. Please visit our project page at https://kaist-viclab.github.io/MoDecGS-site/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has made significant strides in scene representation and neural rendering, with intense efforts focused on adapting it for dynamic scenes. Despite delivering remarkable rendering quality and speed, existing methods struggle with storage demands and representing complex real-world motions. To tackle these issues, we propose MoDecGS, a memory-efficient Gaussian splatting framework designed for reconstructing novel views in challenging scenarios with complex motions. We introduce GlobaltoLocal Motion Decomposition (GLMD) to effectively capture dynamic motions in a coarsetofine manner. This approach leverages Global Canonical Scaffolds (Global CS) and Local Canonical Scaffolds (Local CS), extending static Scaffold representation to dynamic video reconstruction. For Global CS, we propose Global Anchor Deformation (GAD) to efficiently represent global dynamics along complex motions, by directly deforming the implicit Scaffold attributes which are anchor position, offset, and local context features. Next, we finely adjust local motions via the Local Gaussian Deformation (LGD) of Local CS explicitly. Additionally, we introduce Temporal Interval Adjustment (TIA) to automatically control the temporal coverage of each Local CS during training, allowing MoDecGS to find optimal interval assignments based on the specified number of temporal segments. Extensive evaluations demonstrate that MoDecGS achieves an average 70% reduction in model size over stateoftheart methods for dynamic 3D Gaussians from realworld dynamic videos while maintaining or even improving rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00846v2">NeuSG: Neural Implicit Surface Reconstruction with 3D Gaussian Splatting Guidance</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Existing neural implicit surface reconstruction methods have achieved impressive performance in multi-view 3D reconstruction by leveraging explicit geometry priors such as depth maps or point clouds as regularization. However, the reconstruction results still lack fine details because of the over-smoothed depth map or sparse point cloud. In this work, we propose a neural implicit surface reconstruction pipeline with guidance from 3D Gaussian Splatting to recover highly detailed surfaces. The advantage of 3D Gaussian Splatting is that it can generate dense point clouds with detailed structure. Nonetheless, a naive adoption of 3D Gaussian Splatting can fail since the generated points are the centers of 3D Gaussians that do not necessarily lie on the surface. We thus introduce a scale regularizer to pull the centers close to the surface by enforcing the 3D Gaussians to be extremely thin. Moreover, we propose to refine the point cloud from 3D Gaussians Splatting with the normal priors from the surface predicted by neural implicit models instead of using a fixed set of points as guidance. Consequently, the quality of surface reconstruction improves from the guidance of the more accurate 3D Gaussian splatting. By jointly optimizing the 3D Gaussian Splatting and the neural implicit model, our approach benefits from both representations and generates complete surfaces with intricate details. Experiments on Tanks and Temples verify the effectiveness of our proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04459v3">Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 CVPR 2025; Project page at https://svraster.github.io/ ; Code at https://github.com/NVlabs/svraster
    </div>
    <details class="paper-abstract">
      We propose an efficient radiance field rendering algorithm that incorporates a rasterization process on adaptive sparse voxels without neural networks or 3D Gaussians. There are two key contributions coupled with the proposed system. The first is to adaptively and explicitly allocate sparse voxels to different levels of detail within scenes, faithfully reproducing scene details with $65536^3$ grid resolution while achieving high rendering frame rates. Second, we customize a rasterizer for efficient adaptive sparse voxels rendering. We render voxels in the correct depth order by using ray direction-dependent Morton ordering, which avoids the well-known popping artifact found in Gaussian splatting. Our method improves the previous neural-free voxel model by over 4db PSNR and more than 10x FPS speedup, achieving state-of-the-art comparable novel-view synthesis results. Additionally, our voxel representation is seamlessly compatible with grid-based 3D processing techniques such as Volume Fusion, Voxel Pooling, and Marching Cubes, enabling a wide range of future extensions and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12862v1">CAT-3DGS Pro: A New Benchmark for Efficient 3DGS Compression</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown immense potential for novel view synthesis. However, achieving rate-distortion-optimized compression of 3DGS representations for transmission and/or storage applications remains a challenge. CAT-3DGS introduces a context-adaptive triplane hyperprior for end-to-end optimized compression, delivering state-of-the-art coding performance. Despite this, it requires prolonged training and decoding time. To address these limitations, we propose CAT-3DGS Pro, an enhanced version of CAT-3DGS that improves both compression performance and computational efficiency. First, we introduce a PCA-guided vector-matrix hyperprior, which replaces the triplane-based hyperprior to reduce redundant parameters. To achieve a more balanced rate-distortion trade-off and faster encoding, we propose an alternate optimization strategy (A-RDO). Additionally, we refine the sampling rate optimization method in CAT-3DGS, leading to significant improvements in rate-distortion performance. These enhancements result in a 46.6% BD-rate reduction and 3x speedup in training time on BungeeNeRF, while achieving 5x acceleration in decoding speed for the Amsterdam scene compared to CAT-3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12836v1">CompMarkGS: Robust Watermarking for Compression 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 23 pages, 17 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables rapid differentiable rendering for 3D reconstruction and novel view synthesis, leading to its widespread commercial use. Consequently, copyright protection via watermarking has become critical. However, because 3DGS relies on millions of Gaussians, which require gigabytes of storage, efficient transfer and storage require compression. Existing 3DGS watermarking methods are vulnerable to quantization-based compression, often resulting in the loss of the embedded watermark. To address this challenge, we propose a novel watermarking method that ensures watermark robustness after model compression while maintaining high rendering quality. In detail, we incorporate a quantization distortion layer that simulates compression during training, preserving the watermark under quantization-based compression. Also, we propose a learnable watermark embedding feature that embeds the watermark into the anchor feature, ensuring structural consistency and seamless integration into the 3D scene. Furthermore, we present a frequency-aware anchor growing mechanism to enhance image quality in high-frequency regions by effectively identifying Guassians within these regions. Experimental results confirm that our method preserves the watermark and maintains superior image quality under high compression, validating it as a promising approach for a secure 3DGS model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12806v1">AV-Surf: Surface-Enhanced Geometry-Aware Novel-View Acoustic Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Accurately modeling sound propagation with complex real-world environments is essential for Novel View Acoustic Synthesis (NVAS). While previous studies have leveraged visual perception to estimate spatial acoustics, the combined use of surface normal and structural details from 3D representations in acoustic modeling has been underexplored. Given their direct impact on sound wave reflections and propagation, surface normals should be jointly modeled with structural details to achieve accurate spatial acoustics. In this paper, we propose a surface-enhanced geometry-aware approach for NVAS to improve spatial acoustic modeling. To achieve this, we exploit geometric priors, such as image, depth map, surface normals, and point clouds obtained using a 3D Gaussian Splatting (3DGS) based framework. We introduce a dual cross-attention-based transformer integrating geometrical constraints into frequency query to understand the surroundings of the emitter. Additionally, we design a ConvNeXt-based spectral features processing network called Spectral Refinement Network (SRN) to synthesize realistic binaural audio. Experimental results on the RWAVS and SoundSpace datasets highlight the necessity of our approach, as it surpasses existing methods in novel view acoustic synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08317v2">Uni-Gaussians: Unifying Camera and Lidar Simulation with Gaussians for Dynamic Driving Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Ensuring the safety of autonomous vehicles necessitates comprehensive simulation of multi-sensor data, encompassing inputs from both cameras and LiDAR sensors, across various dynamic driving scenarios. Neural rendering techniques, which utilize collected raw sensor data to simulate these dynamic environments, have emerged as a leading methodology. While NeRF-based approaches can uniformly represent scenes for rendering data from both camera and LiDAR, they are hindered by slow rendering speeds due to dense sampling. Conversely, Gaussian Splatting-based methods employ Gaussian primitives for scene representation and achieve rapid rendering through rasterization. However, these rasterization-based techniques struggle to accurately model non-linear optical sensors. This limitation restricts their applicability to sensors beyond pinhole cameras. To address these challenges and enable unified representation of dynamic driving scenarios using Gaussian primitives, this study proposes a novel hybrid approach. Our method utilizes rasterization for rendering image data while employing Gaussian ray-tracing for LiDAR data rendering. Experimental results on public datasets demonstrate that our approach outperforms current state-of-the-art methods. This work presents a unified and efficient solution for realistic simulation of camera and LiDAR data in autonomous driving scenarios using Gaussian primitives, offering significant advancements in both rendering quality and computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.20309v5">InstantSplat: Sparse-view Gaussian Splatting in Seconds</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 Project Page: https://instantsplat.github.io/
    </div>
    <details class="paper-abstract">
      While neural 3D reconstruction has advanced substantially, its performance significantly degrades with sparse-view data, which limits its broader applicability, since SfM is often unreliable in sparse-view scenarios where feature matches are scarce. In this paper, we introduce InstantSplat, a novel approach for addressing sparse-view 3D scene reconstruction at lightning-fast speed. InstantSplat employs a self-supervised framework that optimizes 3D scene representation and camera poses by unprojecting 2D pixels into 3D space and aligning them using differentiable neural rendering. The optimization process is initialized with a large-scale trained geometric foundation model, which provides dense priors that yield initial points through model inference, after which we further optimize all scene parameters using photometric errors. To mitigate redundancy introduced by the prior model, we propose a co-visibility-based geometry initialization, and a Gaussian-based bundle adjustment is employed to rapidly adapt both the scene representation and camera parameters without relying on a complex adaptive density control process. Overall, InstantSplat is compatible with multiple point-based representations for view synthesis and surface reconstruction. It achieves an acceleration of over 30x in reconstruction and improves visual quality (SSIM) from 0.3755 to 0.7624 compared to traditional SfM with 3D-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09868v2">SAFER-Splat: A Control Barrier Function for Safe Navigation with Online Gaussian Splatting Maps</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 Accepted to International Conference on Robotics and Automation
    </div>
    <details class="paper-abstract">
      SAFER-Splat (Simultaneous Action Filtering and Environment Reconstruction) is a real-time, scalable, and minimally invasive action filter, based on control barrier functions, for safe robotic navigation in a detailed map constructed at runtime using Gaussian Splatting (GSplat). We propose a novel Control Barrier Function (CBF) that not only induces safety with respect to all Gaussian primitives in the scene, but when synthesized into a controller, is capable of processing hundreds of thousands of Gaussians while maintaining a minimal memory footprint and operating at 15 Hz during online Splat training. Of the total compute time, a small fraction of it consumes GPU resources, enabling uninterrupted training. The safety layer is minimally invasive, correcting robot actions only when they are unsafe. To showcase the safety filter, we also introduce SplatBridge, an open-source software package built with ROS for real-time GSplat mapping for robots. We demonstrate the safety and robustness of our pipeline first in simulation, where our method is 20-50x faster, safer, and less conservative than competing methods based on neural radiance fields. Further, we demonstrate simultaneous GSplat mapping and safety filtering on a drone hardware platform using only on-board perception. We verify that under teleoperation a human pilot cannot invoke a collision. Our videos and codebase can be found at https://chengine.github.io/safer-splat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14423v2">Unleashing the Potential of Multi-modal Foundation Models and Video Diffusion for 4D Dynamic Physical Scene Simulation</a></div>
    <div class="paper-meta">
      📅 2025-03-16
      | 💬 CVPR 2025. Homepage: https://zhuomanliu.github.io/PhysFlow/
    </div>
    <details class="paper-abstract">
      Realistic simulation of dynamic scenes requires accurately capturing diverse material properties and modeling complex object interactions grounded in physical principles. However, existing methods are constrained to basic material types with limited predictable parameters, making them insufficient to represent the complexity of real-world materials. We introduce PhysFlow, a novel approach that leverages multi-modal foundation models and video diffusion to achieve enhanced 4D dynamic scene simulation. Our method utilizes multi-modal models to identify material types and initialize material parameters through image queries, while simultaneously inferring 3D Gaussian splats for detailed scene representation. We further refine these material parameters using video diffusion with a differentiable Material Point Method (MPM) and optical flow guidance rather than render loss or Score Distillation Sampling (SDS) loss. This integrated framework enables accurate prediction and realistic simulation of dynamic interactions in real-world scenarios, advancing both accuracy and flexibility in physics-based simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08920v3">AV-GS: Learning Material and Geometry Aware Priors for Novel View Acoustic Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-03-16
      | 💬 Accepted to NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Novel view acoustic synthesis (NVAS) aims to render binaural audio at any target viewpoint, given a mono audio emitted by a sound source at a 3D scene. Existing methods have proposed NeRF-based implicit models to exploit visual cues as a condition for synthesizing binaural audio. However, in addition to low efficiency originating from heavy NeRF rendering, these methods all have a limited ability of characterizing the entire scene environment such as room geometry, material properties, and the spatial relation between the listener and sound source. To address these issues, we propose a novel Audio-Visual Gaussian Splatting (AV-GS) model. To obtain a material-aware and geometry-aware condition for audio synthesis, we learn an explicit point-based scene representation with an audio-guidance parameter on locally initialized Gaussian points, taking into account the space relation from the listener and sound source. To make the visual scene model audio adaptive, we propose a point densification and pruning strategy to optimally distribute the Gaussian points, with the per-point contribution in sound propagation (e.g., more points needed for texture-less wall surfaces as they affect sound path diversion). Extensive experiments validate the superiority of our AV-GS over existing alternatives on the real-world RWAS and simulation-based SoundSpaces datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12572v1">Deblur Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      We present Deblur-SLAM, a robust RGB SLAM pipeline designed to recover sharp reconstructions from motion-blurred inputs. The proposed method bridges the strengths of both frame-to-frame and frame-to-model approaches to model sub-frame camera trajectories that lead to high-fidelity reconstructions in motion-blurred settings. Moreover, our pipeline incorporates techniques such as online loop closure and global bundle adjustment to achieve a dense and precise global trajectory. We model the physical image formation process of motion-blurred images and minimize the error between the observed blurry images and rendered blurry images obtained by averaging sharp virtual sub-frame images. Additionally, by utilizing a monocular depth estimator alongside the online deformation of Gaussians, we ensure precise mapping and enhanced image deblurring. The proposed SLAM pipeline integrates all these components to improve the results. We achieve state-of-the-art results for sharp map estimation and sub-frame trajectory recovery both on synthetic and real-world blurry input data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12552v1">MTGS: Multi-Traversal Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      Multi-traversal data, commonly collected through daily commutes or by self-driving fleets, provides multiple viewpoints for scene reconstruction within a road block. This data offers significant potential for high-quality novel view synthesis, which is crucial for applications such as autonomous vehicle simulators. However, inherent challenges in multi-traversal data often result in suboptimal reconstruction quality, including variations in appearance and the presence of dynamic objects. To address these issues, we propose Multi-Traversal Gaussian Splatting (MTGS), a novel approach that reconstructs high-quality driving scenes from arbitrarily collected multi-traversal data by modeling a shared static geometry while separately handling dynamic elements and appearance variations. Our method employs a multi-traversal dynamic scene graph with a shared static node and traversal-specific dynamic nodes, complemented by color correction nodes with learnable spherical harmonics coefficient residuals. This approach enables high-fidelity novel view synthesis and provides flexibility to navigate any viewpoint. We conduct extensive experiments on a large-scale driving dataset, nuPlan, with multi-traversal data. Our results demonstrate that MTGS improves LPIPS by 23.5% and geometry accuracy by 46.3% compared to single-traversal baselines. The code and data would be available to the public.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12535v1">SPC-GS: Gaussian Splatting with Semantic-Prompt Consistency for Indoor Open-World Free-view Synthesis from Sparse Inputs</a></div>
    <div class="paper-meta">
      📅 2025-03-16
      | 💬 Accepted by CVPR2025. The project page is available at https://gbliao.github.io/SPC-GS.github.io/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting-based indoor open-world free-view synthesis approaches have shown significant performance with dense input images. However, they exhibit poor performance when confronted with sparse inputs, primarily due to the sparse distribution of Gaussian points and insufficient view supervision. To relieve these challenges, we propose SPC-GS, leveraging Scene-layout-based Gaussian Initialization (SGI) and Semantic-Prompt Consistency (SPC) Regularization for open-world free view synthesis with sparse inputs. Specifically, SGI provides a dense, scene-layout-based Gaussian distribution by utilizing view-changed images generated from the video generation model and view-constraint Gaussian points densification. Additionally, SPC mitigates limited view supervision by employing semantic-prompt-based consistency constraints developed by SAM2. This approach leverages available semantics from training views, serving as instructive prompts, to optimize visually overlapping regions in novel views with 2D and 3D consistency constraints. Extensive experiments demonstrate the superior performance of SPC-GS across Replica and ScanNet benchmarks. Notably, our SPC-GS achieves a 3.06 dB gain in PSNR for reconstruction quality and a 7.3% improvement in mIoU for open-world semantic segmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06976v2">A Hierarchical Compression Technique for 3D Gaussian Splatting Compression</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (GS) demonstrates excellent rendering quality and generation speed in novel view synthesis. However, substantial data size poses challenges for storage and transmission, making 3D GS compression an essential technology. Current 3D GS compression research primarily focuses on developing more compact scene representations, such as converting explicit 3D GS data into implicit forms. In contrast, compression of the GS data itself has hardly been explored. To address this gap, we propose a Hierarchical GS Compression (HGSC) technique. Initially, we prune unimportant Gaussians based on importance scores derived from both global and local significance, effectively reducing redundancy while maintaining visual quality. An Octree structure is used to compress 3D positions. Based on the 3D GS Octree, we implement a hierarchical attribute compression strategy by employing a KD-tree to partition the 3D GS into multiple blocks. We apply farthest point sampling to select anchor primitives within each block and others as non-anchor primitives with varying Levels of Details (LoDs). Anchor primitives serve as reference points for predicting non-anchor primitives across different LoDs to reduce spatial redundancy. For anchor primitives, we use the region adaptive hierarchical transform to achieve near-lossless compression of various attributes. For non-anchor primitives, each is predicted based on the k-nearest anchor primitives. To further minimize prediction errors, the reconstructed LoD and anchor primitives are combined to form new anchor primitives to predict the next LoD. Our method notably achieves superior compression quality and a significant data size reduction of over 4.5 times compared to the state-of-the-art compression method on small scenes datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16053v2">UnitedVLN: Generalizable Gaussian Splatting for Continuous Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN), where an agent follows instructions to reach a target destination, has recently seen significant advancements. In contrast to navigation in discrete environments with predefined trajectories, VLN in Continuous Environments (VLN-CE) presents greater challenges, as the agent is free to navigate any unobstructed location and is more vulnerable to visual occlusions or blind spots. Recent approaches have attempted to address this by imagining future environments, either through predicted future visual images or semantic features, rather than relying solely on current observations. However, these RGB-based and feature-based methods lack intuitive appearance-level information or high-level semantic complexity crucial for effective navigation. To overcome these limitations, we introduce a novel, generalizable 3DGS-based pre-training paradigm, called UnitedVLN, which enables agents to better explore future environments by unitedly rendering high-fidelity 360 visual images and semantic features. UnitedVLN employs two key schemes: search-then-query sampling and separate-then-united rendering, which facilitate efficient exploitation of neural primitives, helping to integrate both appearance and semantic information for more robust navigation. Extensive experiments demonstrate that UnitedVLN outperforms state-of-the-art methods on existing VLN-CE benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12383v1">VRsketch2Gaussian: 3D VR Sketch Guided 3D Object Generation with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      We propose VRSketch2Gaussian, a first VR sketch-guided, multi-modal, native 3D object generation framework that incorporates a 3D Gaussian Splatting representation. As part of our work, we introduce VRSS, the first large-scale paired dataset containing VR sketches, text, images, and 3DGS, bridging the gap in multi-modal VR sketch-based generation. Our approach features the following key innovations: 1) Sketch-CLIP feature alignment. We propose a two-stage alignment strategy that bridges the domain gap between sparse VR sketch embeddings and rich CLIP embeddings, facilitating both VR sketch-based retrieval and generation tasks. 2) Fine-Grained multi-modal conditioning. We disentangle the 3D generation process by using explicit VR sketches for geometric conditioning and text descriptions for appearance control. To facilitate this, we propose a generalizable VR sketch encoder that effectively aligns different modalities. 3) Efficient and high-fidelity 3D native generation. Our method leverages a 3D-native generation approach that enables fast and texture-rich 3D object synthesis. Experiments conducted on our VRSS dataset demonstrate that our method achieves high-quality, multi-modal VR sketch-based 3D generation. We believe our VRSS dataset and VRsketch2Gaussian method will be beneficial for the 3D generation community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12343v1">TopoGaussian: Inferring Internal Topology Structures from Visual Clues</a></div>
    <div class="paper-meta">
      📅 2025-03-16
    </div>
    <details class="paper-abstract">
      We present TopoGaussian, a holistic, particle-based pipeline for inferring the interior structure of an opaque object from easily accessible photos and videos as input. Traditional mesh-based approaches require tedious and error-prone mesh filling and fixing process, while typically output rough boundary surface. Our pipeline combines Gaussian Splatting with a novel, versatile particle-based differentiable simulator that simultaneously accommodates constitutive model, actuator, and collision, without interference with mesh. Based on the gradients from this simulator, we provide flexible choice of topology representation for optimization, including particle, neural implicit surface, and quadratic surface. The resultant pipeline takes easily accessible photos and videos as input and outputs the topology that matches the physical characteristics of the input. We demonstrate the efficacy of our pipeline on a synthetic dataset and four real-world tasks with 3D-printed prototypes. Compared with existing mesh-based method, our pipeline is 5.26x faster on average with improved shape quality. These results highlight the potential of our pipeline in 3D vision, soft robotics, and manufacturing applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12335v1">GS-3I: Gaussian Splatting for Surface Reconstruction from Illumination-Inconsistent Images</a></div>
    <div class="paper-meta">
      📅 2025-03-16
      | 💬 This paper has been submitted to IROS 2025
    </div>
    <details class="paper-abstract">
      Accurate geometric surface reconstruction, providing essential environmental information for navigation and manipulation tasks, is critical for enabling robotic self-exploration and interaction. Recently, 3D Gaussian Splatting (3DGS) has gained significant attention in the field of surface reconstruction due to its impressive geometric quality and computational efficiency. While recent relevant advancements in novel view synthesis under inconsistent illumination using 3DGS have shown promise, the challenge of robust surface reconstruction under such conditions is still being explored. To address this challenge, we propose a method called GS-3I. Specifically, to mitigate 3D Gaussian optimization bias caused by underexposed regions in single-view images, based on Convolutional Neural Network (CNN), a tone mapping correction framework is introduced. Furthermore, inconsistent lighting across multi-view images, resulting from variations in camera settings and complex scene illumination, often leads to geometric constraint mismatches and deviations in the reconstructed surface. To overcome this, we propose a normal compensation mechanism that integrates reference normals extracted from single-view image with normals computed from multi-view observations to effectively constrain geometric inconsistencies. Extensive experimental evaluations demonstrate that GS-3I can achieve robust and accurate surface reconstruction across complex illumination scenarios, highlighting its effectiveness and versatility in this critical challenge. https://github.com/TFwang-9527/GS-3I
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12307v1">Swift4D:Adaptive divide-and-conquer Gaussian Splatting for compact and efficient reconstruction of dynamic scene</a></div>
    <div class="paper-meta">
      📅 2025-03-16
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Novel view synthesis has long been a practical but challenging task, although the introduction of numerous methods to solve this problem, even combining advanced representations like 3D Gaussian Splatting, they still struggle to recover high-quality results and often consume too much storage memory and training time. In this paper we propose Swift4D, a divide-and-conquer 3D Gaussian Splatting method that can handle static and dynamic primitives separately, achieving a good trade-off between rendering quality and efficiency, motivated by the fact that most of the scene is the static primitive and does not require additional dynamic properties. Concretely, we focus on modeling dynamic transformations only for the dynamic primitives which benefits both efficiency and quality. We first employ a learnable decomposition strategy to separate the primitives, which relies on an additional parameter to classify primitives as static or dynamic. For the dynamic primitives, we employ a compact multi-resolution 4D Hash mapper to transform these primitives from canonical space into deformation space at each timestamp, and then mix the static and dynamic primitives to produce the final output. This divide-and-conquer method facilitates efficient training and reduces storage redundancy. Our method not only achieves state-of-the-art rendering quality while being 20X faster in training than previous SOTA methods with a minimum storage requirement of only 30MB on real-world datasets. Code is available at https://github.com/WuJH2001/swift4d.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12284v1">REdiSplats: Ray Tracing for Editable Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has become one of the most important neural rendering algorithms. GS represents 3D scenes using Gaussian components with trainable color and opacity. This representation achieves high-quality renderings with fast inference. Regrettably, it is challenging to integrate such a solution with varying light conditions, including shadows and light reflections, manual adjustments, and a physical engine. Recently, a few approaches have appeared that incorporate ray-tracing or mesh primitives into GS to address some of these caveats. However, no such solution can simultaneously solve all the existing limitations of the classical GS. Consequently, we introduce REdiSplats, which employs ray tracing and a mesh-based representation of flat 3D Gaussians. In practice, we model the scene using flat Gaussian distributions parameterized by the mesh. We can leverage fast ray tracing and control Gaussian modification by adjusting the mesh vertices. Moreover, REdiSplats allows modeling of light conditions, manual adjustments, and physical simulation. Furthermore, we can render our models using 3D tools such as Blender or Nvdiffrast, which opens the possibility of integrating them with all existing 3D graphics techniques dedicated to mesh representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10148v2">3D Student Splatting and Scooping</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) provides a new framework for novel view synthesis, and has spiked a new wave of research in neural rendering and related applications. As 3DGS is becoming a foundational component of many models, any improvement on 3DGS itself can bring huge benefits. To this end, we aim to improve the fundamental paradigm and formulation of 3DGS. We argue that as an unnormalized mixture model, it needs to be neither Gaussians nor splatting. We subsequently propose a new mixture model consisting of flexible Student's t distributions, with both positive (splatting) and negative (scooping) densities. We name our model Student Splatting and Scooping, or SSS. When providing better expressivity, SSS also poses new challenges in learning. Therefore, we also propose a new principled sampling approach for optimization. Through exhaustive evaluation and comparison, across multiple datasets, settings, and metrics, we demonstrate that SSS outperforms existing methods in terms of quality and parameter efficiency, e.g. achieving matching or better quality with similar numbers of components, and obtaining comparable results while reducing the component number by as much as 82%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04082v2">Instrument-Splatting: Controllable Photorealistic Reconstruction of Surgical Instruments Using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-15
      | 💬 11 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Real2Sim is becoming increasingly important with the rapid development of surgical artificial intelligence (AI) and autonomy. In this work, we propose a novel Real2Sim methodology, Instrument-Splatting, that leverages 3D Gaussian Splatting to provide fully controllable 3D reconstruction of surgical instruments from monocular surgical videos. To maintain both high visual fidelity and manipulability, we introduce a geometry pre-training to bind Gaussian point clouds on part mesh with accurate geometric priors and define a forward kinematics to control the Gaussians as flexible as real instruments. Afterward, to handle unposed videos, we design a novel instrument pose tracking method leveraging semantics-embedded Gaussians to robustly refine per-frame instrument poses and joint states in a render-and-compare manner, which allows our instrument Gaussian to accurately learn textures and reach photorealistic rendering. We validated our method on 2 publicly released surgical videos and 4 videos collected on ex vivo tissues and green screens. Quantitative and qualitative evaluations demonstrate the effectiveness and superiority of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12001v1">3D Gaussian Splatting against Moving Objects for High-Fidelity Street Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      The accurate reconstruction of dynamic street scenes is critical for applications in autonomous driving, augmented reality, and virtual reality. Traditional methods relying on dense point clouds and triangular meshes struggle with moving objects, occlusions, and real-time processing constraints, limiting their effectiveness in complex urban environments. While multi-view stereo and neural radiance fields have advanced 3D reconstruction, they face challenges in computational efficiency and handling scene dynamics. This paper proposes a novel 3D Gaussian point distribution method for dynamic street scene reconstruction. Our approach introduces an adaptive transparency mechanism that eliminates moving objects while preserving high-fidelity static scene details. Additionally, iterative refinement of Gaussian point distribution enhances geometric accuracy and texture representation. We integrate directional encoding with spatial position optimization to optimize storage and rendering efficiency, reducing redundancy while maintaining scene integrity. Experimental results demonstrate that our method achieves high reconstruction quality, improved rendering performance, and adaptability in large-scale dynamic environments. These contributions establish a robust framework for real-time, high-precision 3D reconstruction, advancing the practicality of dynamic scene modeling across multiple applications. The source code for this work is available to the public at https://github.com/deepcoxcom/3dgs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11981v1">DecompDreamer: Advancing Structured 3D Asset Generation with Multi-Object Decomposition and Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      Text-to-3D generation saw dramatic advances in recent years by leveraging Text-to-Image models. However, most existing techniques struggle with compositional prompts, which describe multiple objects and their spatial relationships. They often fail to capture fine-grained inter-object interactions. We introduce DecompDreamer, a Gaussian splatting-based training routine designed to generate high-quality 3D compositions from such complex prompts. DecompDreamer leverages Vision-Language Models (VLMs) to decompose scenes into structured components and their relationships. We propose a progressive optimization strategy that first prioritizes joint relationship modeling before gradually shifting toward targeted object refinement. Our qualitative and quantitative evaluations against state-of-the-art text-to-3D models demonstrate that DecompDreamer effectively generates intricate 3D compositions with superior object disentanglement, offering enhanced control and flexibility in 3D generation. Project page : https://decompdreamer3d.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11979v1">DynaGSLAM: Real-Time Gaussian-Splatting SLAM for Online Rendering, Tracking, Motion Predictions of Moving Objects in Dynamic Scenes</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      Simultaneous Localization and Mapping (SLAM) is one of the most important environment-perception and navigation algorithms for computer vision, robotics, and autonomous cars/drones. Hence, high quality and fast mapping becomes a fundamental problem. With the advent of 3D Gaussian Splatting (3DGS) as an explicit representation with excellent rendering quality and speed, state-of-the-art (SOTA) works introduce GS to SLAM. Compared to classical pointcloud-SLAM, GS-SLAM generates photometric information by learning from input camera views and synthesize unseen views with high-quality textures. However, these GS-SLAM fail when moving objects occupy the scene that violate the static assumption of bundle adjustment. The failed updates of moving GS affects the static GS and contaminates the full map over long frames. Although some efforts have been made by concurrent works to consider moving objects for GS-SLAM, they simply detect and remove the moving regions from GS rendering ("anti'' dynamic GS-SLAM), where only the static background could benefit from GS. To this end, we propose the first real-time GS-SLAM, "DynaGSLAM'', that achieves high-quality online GS rendering, tracking, motion predictions of moving objects in dynamic scenes while jointly estimating accurate ego motion. Our DynaGSLAM outperforms SOTA static & "Anti'' dynamic GS-SLAM on three dynamic real datasets, while keeping speed and memory efficiency in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11601v1">Advancing 3D Gaussian Splatting Editing with Complementary and Consensus Information</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 7 pages, 9 figures
    </div>
    <details class="paper-abstract">
      We present a novel framework for enhancing the visual fidelity and consistency of text-guided 3D Gaussian Splatting (3DGS) editing. Existing editing approaches face two critical challenges: inconsistent geometric reconstructions across multiple viewpoints, particularly in challenging camera positions, and ineffective utilization of depth information during image manipulation, resulting in over-texture artifacts and degraded object boundaries. To address these limitations, we introduce: 1) A complementary information mutual learning network that enhances depth map estimation from 3DGS, enabling precise depth-conditioned 3D editing while preserving geometric structures. 2) A wavelet consensus attention mechanism that effectively aligns latent codes during the diffusion denoising process, ensuring multi-view consistency in the edited results. Through extensive experimentation, our method demonstrates superior performance in rendering quality and view consistency compared to state-of-the-art approaches. The results validate our framework as an effective solution for text-guided editing of 3D scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11345v1">EgoSplat: Open-Vocabulary Egocentric Scene Understanding with Language Embedded 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      Egocentric scenes exhibit frequent occlusions, varied viewpoints, and dynamic interactions compared to typical scene understanding tasks. Occlusions and varied viewpoints can lead to multi-view semantic inconsistencies, while dynamic objects may act as transient distractors, introducing artifacts into semantic feature modeling. To address these challenges, we propose EgoSplat, a language-embedded 3D Gaussian Splatting framework for open-vocabulary egocentric scene understanding. A multi-view consistent instance feature aggregation method is designed to leverage the segmentation and tracking capabilities of SAM2 to selectively aggregate complementary features across views for each instance, ensuring precise semantic representation of scenes. Additionally, an instance-aware spatial-temporal transient prediction module is constructed to improve spatial integrity and temporal continuity in predictions by incorporating spatial-temporal associations across multi-view instances, effectively reducing artifacts in the semantic reconstruction of egocentric scenes. EgoSplat achieves state-of-the-art performance in both localization and segmentation tasks on two datasets, outperforming existing methods with a 8.2% improvement in localization accuracy and a 3.7% improvement in segmentation mIoU on the ADT dataset, and setting a new benchmark in open-vocabulary egocentric scene understanding. The code will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10283v3">GauSTAR: Gaussian Surface Tracking and Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting techniques have enabled efficient photo-realistic rendering of static scenes. Recent works have extended these approaches to support surface reconstruction and tracking. However, tracking dynamic surfaces with 3D Gaussians remains challenging due to complex topology changes, such as surfaces appearing, disappearing, or splitting. To address these challenges, we propose GauSTAR, a novel method that achieves photo-realistic rendering, accurate surface reconstruction, and reliable 3D tracking for general dynamic scenes with changing topology. Given multi-view captures as input, GauSTAR binds Gaussians to mesh faces to represent dynamic objects. For surfaces with consistent topology, GauSTAR maintains the mesh topology and tracks the meshes using Gaussians. For regions where topology changes, GauSTAR adaptively unbinds Gaussians from the mesh, enabling accurate registration and generation of new surfaces based on these optimized Gaussians. Additionally, we introduce a surface-based scene flow method that provides robust initialization for tracking between frames. Experiments demonstrate that our method effectively tracks and reconstructs dynamic surfaces, enabling a range of applications. Our project page with the code release is available at https://eth-ait.github.io/GauSTAR/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05111v2">LiDAR-GS:Real-time LiDAR Re-Simulation using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      We present LiDAR-GS, a Gaussian Splatting (GS) method for real-time, high-fidelity re-simulation of LiDAR scans in public urban road scenes. Recent GS methods proposed for cameras have achieved significant advancements in real-time rendering beyond Neural Radiance Fields (NeRF). However, applying GS representation to LiDAR, an active 3D sensor type, poses several challenges that must be addressed to preserve high accuracy and unique characteristics. Specifically, LiDAR-GS designs a differentiable laser beam splatting, using range-view representation for precise surface splatting by projecting lasers onto micro cross-sections, effectively eliminating artifacts associated with local affine approximations. Furthermore, LiDAR-GS leverages Neural Gaussian Representation, which further integrate view-dependent clues, to represent key LiDAR properties that are influenced by the incident direction and external factors. Combining these practices with some essential adaptations, e.g., dynamic instances decomposition, LiDAR-GS succeeds in simultaneously re-simulating depth, intensity, and ray-drop channels, achieving state-of-the-art results in both rendering frame rate and quality on publically available large scene datasets when compared with the methods using explicit mesh or implicit NeRF. Our source code is publicly available at https://www.github.com/cqf7419/LiDAR-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11172v1">Uncertainty-Aware Normal-Guided Gaussian Splatting for Surface Reconstruction from Sparse Image Sequences</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 12 pages, 8 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has achieved impressive rendering performance in novel view synthesis. However, its efficacy diminishes considerably in sparse image sequences, where inherent data sparsity amplifies geometric uncertainty during optimization. This often leads to convergence at suboptimal local minima, resulting in noticeable structural artifacts in the reconstructed scenes.To mitigate these issues, we propose Uncertainty-aware Normal-Guided Gaussian Splatting (UNG-GS), a novel framework featuring an explicit Spatial Uncertainty Field (SUF) to quantify geometric uncertainty within the 3DGS pipeline. UNG-GS enables high-fidelity rendering and achieves high-precision reconstruction without relying on priors. Specifically, we first integrate Gaussian-based probabilistic modeling into the training of 3DGS to optimize the SUF, providing the model with adaptive error tolerance. An uncertainty-aware depth rendering strategy is then employed to weight depth contributions based on the SUF, effectively reducing noise while preserving fine details. Furthermore, an uncertainty-guided normal refinement method adjusts the influence of neighboring depth values in normal estimation, promoting robust results. Extensive experiments demonstrate that UNG-GS significantly outperforms state-of-the-art methods in both sparse and dense sequences. The code will be open-source.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19895v4">GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 This paper is accepted by the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR), 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently created impressive 3D assets for various applications. However, considering security, capacity, invisibility, and training efficiency, the copyright of 3DGS assets is not well protected as existing watermarking methods are unsuited for its rendering pipeline. In this paper, we propose GuardSplat, an innovative and efficient framework for watermarking 3DGS assets. Specifically, 1) We propose a CLIP-guided pipeline for optimizing the message decoder with minimal costs. The key objective is to achieve high-accuracy extraction by leveraging CLIP's aligning capability and rich representations, demonstrating exceptional capacity and efficiency. 2) We tailor a Spherical-Harmonic-aware (SH-aware) Message Embedding module for 3DGS, seamlessly embedding messages into the SH features of each 3D Gaussian while preserving the original 3D structure. This enables watermarking 3DGS assets with minimal fidelity trade-offs and prevents malicious users from removing the watermarks from the model files, meeting the demands for invisibility and security. 3) We present an Anti-distortion Message Extraction module to improve robustness against various distortions. Experiments demonstrate that GuardSplat outperforms state-of-the-art and achieves fast optimization speed. Project page is at https://narcissusex.github.io/GuardSplat, and Code is at https://github.com/NarcissusEx/GuardSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11731v1">Industrial-Grade Sensor Simulation via Gaussian Splatting: A Modular Framework for Scalable Editing and Full-Stack Validation</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      Sensor simulation is pivotal for scalable validation of autonomous driving systems, yet existing Neural Radiance Fields (NeRF) based methods face applicability and efficiency challenges in industrial workflows. This paper introduces a Gaussian Splatting (GS) based system to address these challenges: We first break down sensor simulator components and analyze the possible advantages of GS over NeRF. Then in practice, we refactor three crucial components through GS, to leverage its explicit scene representation and real-time rendering: (1) choosing the 2D neural Gaussian representation for physics-compliant scene and sensor modeling, (2) proposing a scene editing pipeline to leverage Gaussian primitives library for data augmentation, and (3) coupling a controllable diffusion model for scene expansion and harmonization. We implement this framework on a proprietary autonomous driving dataset supporting cameras and LiDAR sensors. We demonstrate through ablation studies that our approach reduces frame-wise simulation latency, achieves better geometric and photometric consistency, and enables interpretable explicit scene editing and expansion. Furthermore, we showcase how integrating such a GS-based sensor simulator with traffic and dynamic simulators enables full-stack testing of end-to-end autonomy algorithms. Our work provides both algorithmic insights and practical validation, establishing GS as a cornerstone for industrial-grade sensor simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10625v1">LHM: Large Animatable Human Reconstruction Model from a Single Image in Seconds</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 Project Page: https://lingtengqiu.github.io/LHM/
    </div>
    <details class="paper-abstract">
      Animatable 3D human reconstruction from a single image is a challenging problem due to the ambiguity in decoupling geometry, appearance, and deformation. Recent advances in 3D human reconstruction mainly focus on static human modeling, and the reliance of using synthetic 3D scans for training limits their generalization ability. Conversely, optimization-based video methods achieve higher fidelity but demand controlled capture conditions and computationally intensive refinement processes. Motivated by the emergence of large reconstruction models for efficient static reconstruction, we propose LHM (Large Animatable Human Reconstruction Model) to infer high-fidelity avatars represented as 3D Gaussian splatting in a feed-forward pass. Our model leverages a multimodal transformer architecture to effectively encode the human body positional features and image features with attention mechanism, enabling detailed preservation of clothing geometry and texture. To further boost the face identity preservation and fine detail recovery, we propose a head feature pyramid encoding scheme to aggregate multi-scale features of the head regions. Extensive experiments demonstrate that our LHM generates plausible animatable human in seconds without post-processing for face and hands, outperforming existing methods in both reconstruction accuracy and generalization ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10604v1">MuDG: Taming Multi-modal Diffusion with Gaussian Splatting for Urban Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      Recent breakthroughs in radiance fields have significantly advanced 3D scene reconstruction and novel view synthesis (NVS) in autonomous driving. Nevertheless, critical limitations persist: reconstruction-based methods exhibit substantial performance deterioration under significant viewpoint deviations from training trajectories, while generation-based techniques struggle with temporal coherence and precise scene controllability. To overcome these challenges, we present MuDG, an innovative framework that integrates Multi-modal Diffusion model with Gaussian Splatting (GS) for Urban Scene Reconstruction. MuDG leverages aggregated LiDAR point clouds with RGB and geometric priors to condition a multi-modal video diffusion model, synthesizing photorealistic RGB, depth, and semantic outputs for novel viewpoints. This synthesis pipeline enables feed-forward NVS without computationally intensive per-scene optimization, providing comprehensive supervision signals to refine 3DGS representations for rendering robustness enhancement under extreme viewpoint changes. Experiments on the Open Waymo Dataset demonstrate that MuDG outperforms existing methods in both reconstruction and synthesis quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08093v2">MVGSR: Multi-View Consistency Gaussian Splatting for Robust Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 project page https://mvgsr.github.io
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has gained significant attention for its high-quality rendering capabilities, ultra-fast training, and inference speeds. However, when we apply 3DGS to surface reconstruction tasks, especially in environments with dynamic objects and distractors, the method suffers from floating artifacts and color errors due to inconsistency from different viewpoints. To address this challenge, we propose Multi-View Consistency Gaussian Splatting for the domain of Robust Surface Reconstruction (\textbf{MVGSR}), which takes advantage of lightweight Gaussian models and a {heuristics-guided distractor masking} strategy for robust surface reconstruction in non-static environments. Compared to existing methods that rely on MLPs for distractor segmentation strategies, our approach separates distractors from static scene elements by comparing multi-view feature consistency, allowing us to obtain precise distractor masks early in training. Furthermore, we introduce a pruning measure based on multi-view contributions to reset transmittance, effectively reducing floating artifacts. Finally, a multi-view consistency loss is applied to achieve high-quality performance in surface reconstruction tasks. Experimental results demonstrate that MVGSR achieves competitive geometric accuracy and rendering fidelity compared to the state-of-the-art surface reconstruction algorithms. More information is available on our project page (https://mvgsr.github.io).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10437v1">4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 CVPR 2025. Project Page: https://4d-langsplat.github.io
    </div>
    <details class="paper-abstract">
      Learning 4D language fields to enable time-sensitive, open-ended language queries in dynamic scenes is essential for many real-world applications. While LangSplat successfully grounds CLIP features into 3D Gaussian representations, achieving precision and efficiency in 3D static scenes, it lacks the ability to handle dynamic 4D fields as CLIP, designed for static image-text tasks, cannot capture temporal dynamics in videos. Real-world environments are inherently dynamic, with object semantics evolving over time. Building a precise 4D language field necessitates obtaining pixel-aligned, object-wise video features, which current vision models struggle to achieve. To address these challenges, we propose 4D LangSplat, which learns 4D language fields to handle time-agnostic or time-sensitive open-vocabulary queries in dynamic scenes efficiently. 4D LangSplat bypasses learning the language field from vision features and instead learns directly from text generated from object-wise video captions via Multimodal Large Language Models (MLLMs). Specifically, we propose a multimodal object-wise video prompting method, consisting of visual and text prompts that guide MLLMs to generate detailed, temporally consistent, high-quality captions for objects throughout a video. These captions are encoded using a Large Language Model into high-quality sentence embeddings, which then serve as pixel-aligned, object-specific feature supervision, facilitating open-vocabulary text queries through shared embedding spaces. Recognizing that objects in 4D scenes exhibit smooth transitions across states, we further propose a status deformable network to model these continuous changes over time effectively. Our results across multiple benchmarks demonstrate that 4D LangSplat attains precise and efficient results for both time-sensitive and time-agnostic open-vocabulary queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16816v3">SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      Ensuring the safety of autonomous robots, such as self-driving vehicles, requires extensive testing across diverse driving scenarios. Simulation is a key ingredient for conducting such testing in a cost-effective and scalable way. Neural rendering methods have gained popularity, as they can build simulation environments from collected logs in a data-driven manner. However, existing neural radiance field (NeRF) methods for sensor-realistic rendering of camera and lidar data suffer from low rendering speeds, limiting their applicability for large-scale testing. While 3D Gaussian Splatting (3DGS) enables real-time rendering, current methods are limited to camera data and are unable to render lidar data essential for autonomous driving. To address these limitations, we propose SplatAD, the first 3DGS-based method for realistic, real-time rendering of dynamic scenes for both camera and lidar data. SplatAD accurately models key sensor-specific phenomena such as rolling shutter effects, lidar intensity, and lidar ray dropouts, using purpose-built algorithms to optimize rendering efficiency. Evaluation across three autonomous driving datasets demonstrates that SplatAD achieves state-of-the-art rendering quality with up to +2 PSNR for NVS and +3 PSNR for reconstruction while increasing rendering speed over NeRF-based methods by an order of magnitude. See https://research.zenseact.com/publications/splatad/ for our project page.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10286v1">VicaSplat: A Single Run is All You Need for 3D Gaussian Splatting and Camera Estimation from Unposed Video Frames</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      We present VicaSplat, a novel framework for joint 3D Gaussians reconstruction and camera pose estimation from a sequence of unposed video frames, which is a critical yet underexplored task in real-world 3D applications. The core of our method lies in a novel transformer-based network architecture. In particular, our model starts with an image encoder that maps each image to a list of visual tokens. All visual tokens are concatenated with additional inserted learnable camera tokens. The obtained tokens then fully communicate with each other within a tailored transformer decoder. The camera tokens causally aggregate features from visual tokens of different views, and further modulate them frame-wisely to inject view-dependent features. 3D Gaussian splats and camera pose parameters can then be estimated via different prediction heads. Experiments show that VicaSplat surpasses baseline methods for multi-view inputs, and achieves comparable performance to prior two-view approaches. Remarkably, VicaSplat also demonstrates exceptional cross-dataset generalization capability on the ScanNet benchmark, achieving superior performance without any fine-tuning. Project page: https://lizhiqi49.github.io/VicaSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10256v1">ROODI: Reconstructing Occluded Objects with Denoising Inpainters</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 Project page: https://yeonjin-chang.github.io/ROODI/
    </div>
    <details class="paper-abstract">
      While the quality of novel-view images has improved dramatically with 3D Gaussian Splatting, extracting specific objects from scenes remains challenging. Isolating individual 3D Gaussian primitives for each object and handling occlusions in scenes remain far from being solved. We propose a novel object extraction method based on two key principles: (1) being object-centric by pruning irrelevant primitives; and (2) leveraging generative inpainting to compensate for missing observations caused by occlusions. For pruning, we analyze the local structure of primitives using K-nearest neighbors, and retain only relevant ones. For inpainting, we employ an off-the-shelf diffusion-based inpainter combined with occlusion reasoning, utilizing the 3D representation of the entire scene. Our findings highlight the crucial synergy between pruning and inpainting, both of which significantly enhance extraction performance. We evaluate our method on a standard real-world dataset and introduce a synthetic dataset for quantitative analysis. Our approach outperforms the state-of-the-art, demonstrating its effectiveness in object extraction from complex scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10170v1">GS-SDF: LiDAR-Augmented Gaussian Splatting and Neural SDF for Geometrically Consistent Rendering and Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      Digital twins are fundamental to the development of autonomous driving and embodied artificial intelligence. However, achieving high-granularity surface reconstruction and high-fidelity rendering remains a challenge. Gaussian splatting offers efficient photorealistic rendering but struggles with geometric inconsistencies due to fragmented primitives and sparse observational data in robotics applications. Existing regularization methods, which rely on render-derived constraints, often fail in complex environments. Moreover, effectively integrating sparse LiDAR data with Gaussian splatting remains challenging. We propose a unified LiDAR-visual system that synergizes Gaussian splatting with a neural signed distance field. The accurate LiDAR point clouds enable a trained neural signed distance field to offer a manifold geometry field, This motivates us to offer an SDF-based Gaussian initialization for physically grounded primitive placement and a comprehensive geometric regularization for geometrically consistent rendering and reconstruction. Experiments demonstrate superior reconstruction accuracy and rendering quality across diverse trajectories. To benefit the community, the codes will be released at https://github.com/hku-mars/GS-SDF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10148v1">3D Student Splatting and Scooping</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) provides a new framework for novel view synthesis, and has spiked a new wave of research in neural rendering and related applications. As 3DGS is becoming a foundational component of many models, any improvement on 3DGS itself can bring huge benefits. To this end, we aim to improve the fundamental paradigm and formulation of 3DGS. We argue that as an unnormalized mixture model, it needs to be neither Gaussians nor splatting. We subsequently propose a new mixture model consisting of flexible Student's t distributions, with both positive (splatting) and negative (scooping) densities. We name our model Student Splatting and Scooping, or SSS. When providing better expressivity, SSS also poses new challenges in learning. Therefore, we also propose a new principled sampling approach for optimization. Through exhaustive evaluation and comparison, across multiple datasets, settings, and metrics, we demonstrate that SSS outperforms existing methods in terms of quality and parameter efficiency, e.g. achieving matching or better quality with similar numbers of components, and obtaining comparable results while reducing the component number by as much as 82%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10143v1">GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 This paper is accepted by CVPR 2025. Project page is available at https://liujf1226.github.io/GaussHDR
    </div>
    <details class="paper-abstract">
      High dynamic range (HDR) novel view synthesis (NVS) aims to reconstruct HDR scenes by leveraging multi-view low dynamic range (LDR) images captured at different exposure levels. Current training paradigms with 3D tone mapping often result in unstable HDR reconstruction, while training with 2D tone mapping reduces the model's capacity to fit LDR images. Additionally, the global tone mapper used in existing methods can impede the learning of both HDR and LDR representations. To address these challenges, we present GaussHDR, which unifies 3D and 2D local tone mapping through 3D Gaussian splatting. Specifically, we design a residual local tone mapper for both 3D and 2D tone mapping that accepts an additional context feature as input. We then propose combining the dual LDR rendering results from both 3D and 2D local tone mapping at the loss level. Finally, recognizing that different scenes may exhibit varying balances between the dual results, we introduce uncertainty learning and use the uncertainties for adaptive modulation. Extensive experiments demonstrate that GaussHDR significantly outperforms state-of-the-art methods in both synthetic and real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00746v3">DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3D-GS) have shown remarkable success in representing 3D scenes and generating high-quality, novel views in real-time. However, 3D-GS and its variants assume that input images are captured based on pinhole imaging and are fully in focus. This assumption limits their applicability, as real-world images often feature shallow depth-of-field (DoF). In this paper, we introduce DoF-Gaussian, a controllable depth-of-field method for 3D-GS. We develop a lens-based imaging model based on geometric optics principles to control DoF effects. To ensure accurate scene geometry, we incorporate depth priors adjusted per scene, and we apply defocus-to-focus adaptation to minimize the gap in the circle of confusion. We also introduce a synthetic dataset to assess refocusing capabilities and the model's ability to learn precise lens parameters. Our framework is customizable and supports various interactive applications. Extensive experiments confirm the effectiveness of our method. Our project is available at https://dof-gaussian.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10860v1">RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 Project page: https://people.engr.tamu.edu/nimak/Papers/RI3D, Code: https://github.com/avinashpaliwal/RI3D
    </div>
    <details class="paper-abstract">
      In this paper, we propose RI3D, a novel 3DGS-based approach that harnesses the power of diffusion models to reconstruct high-quality novel views given a sparse set of input images. Our key contribution is separating the view synthesis process into two tasks of reconstructing visible regions and hallucinating missing regions, and introducing two personalized diffusion models, each tailored to one of these tasks. Specifically, one model ('repair') takes a rendered image as input and predicts the corresponding high-quality image, which in turn is used as a pseudo ground truth image to constrain the optimization. The other model ('inpainting') primarily focuses on hallucinating details in unobserved areas. To integrate these models effectively, we introduce a two-stage optimization strategy: the first stage reconstructs visible areas using the repair model, and the second stage reconstructs missing regions with the inpainting model while ensuring coherence through further optimization. Moreover, we augment the optimization with a novel Gaussian initialization method that obtains per-image depth by combining 3D-consistent and smooth depth with highly detailed relative depth. We demonstrate that by separating the process into two tasks and addressing them with the repair and inpainting models, we produce results with detailed textures in both visible and missing regions that outperform state-of-the-art approaches on a diverse set of scenes with extremely sparse inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13995v3">Avatar Concept Slider: Controllable Editing of Concepts in 3D Human Avatars</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      Text-based editing of 3D human avatars to precisely match user requirements is challenging due to the inherent ambiguity and limited expressiveness of natural language. To overcome this, we propose the Avatar Concept Slider (ACS), a 3D avatar editing method that allows precise editing of semantic concepts in human avatars towards a specified intermediate point between two extremes of concepts, akin to moving a knob along a slider track. To achieve this, our ACS has three designs: Firstly, a Concept Sliding Loss based on linear discriminant analysis to pinpoint the concept-specific axes for precise editing. Secondly, an Attribute Preserving Loss based on principal component analysis for improved preservation of avatar identity during editing. We further propose a 3D Gaussian Splatting primitive selection mechanism based on concept-sensitivity, which updates only the primitives that are the most sensitive to our target concept, to improve efficiency. Results demonstrate that our ACS enables controllable 3D avatar editing, without compromising the avatar quality or its identifying attributes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09464v1">Hybrid Rendering for Multimodal Autonomous Driving: Merging Neural and Physics-Based Simulation</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Neural reconstruction models for autonomous driving simulation have made significant strides in recent years, with dynamic models becoming increasingly prevalent. However, these models are typically limited to handling in-domain objects closely following their original trajectories. We introduce a hybrid approach that combines the strengths of neural reconstruction with physics-based rendering. This method enables the virtual placement of traditional mesh-based dynamic agents at arbitrary locations, adjustments to environmental conditions, and rendering from novel camera viewpoints. Our approach significantly enhances novel view synthesis quality -- especially for road surfaces and lane markings -- while maintaining interactive frame rates through our novel training method, NeRF2GS. This technique leverages the superior generalization capabilities of NeRF-based methods and the real-time rendering speed of 3D Gaussian Splatting (3DGS). We achieve this by training a customized NeRF model on the original images with depth regularization derived from a noisy LiDAR point cloud, then using it as a teacher model for 3DGS training. This process ensures accurate depth, surface normals, and camera appearance modeling as supervision. With our block-based training parallelization, the method can handle large-scale reconstructions (greater than or equal to 100,000 square meters) and predict segmentation masks, surface normals, and depth maps. During simulation, it supports a rasterization-based rendering backend with depth-based composition and multiple camera models for real-time camera simulation, as well as a ray-traced backend for precise LiDAR simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09447v1">Online Language Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      To enable AI agents to interact seamlessly with both humans and 3D environments, they must not only perceive the 3D world accurately but also align human language with 3D spatial representations. While prior work has made significant progress by integrating language features into geometrically detailed 3D scene representations using 3D Gaussian Splatting (GS), these approaches rely on computationally intensive offline preprocessing of language features for each input image, limiting adaptability to new environments. In this work, we introduce Online Language Splatting, the first framework to achieve online, near real-time, open-vocabulary language mapping within a 3DGS-SLAM system without requiring pre-generated language features. The key challenge lies in efficiently fusing high-dimensional language features into 3D representations while balancing the computation speed, memory usage, rendering quality and open-vocabulary capability. To this end, we innovatively design: (1) a high-resolution CLIP embedding module capable of generating detailed language feature maps in 18ms per frame, (2) a two-stage online auto-encoder that compresses 768-dimensional CLIP features to 15 dimensions while preserving open-vocabulary capabilities, and (3) a color-language disentangled optimization approach to improve rendering quality. Experimental results show that our online method not only surpasses the state-of-the-art offline methods in accuracy but also achieves more than 40x efficiency boost, demonstrating the potential for dynamic and interactive AI applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08017v3">Fast Feedforward 3D Gaussian Splatting Compression</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 Project Page: https://yihangchen-ee.github.io/project_fcgs/ Code: https://github.com/yihangchen-ee/fcgs/
    </div>
    <details class="paper-abstract">
      With 3D Gaussian Splatting (3DGS) advancing real-time and high-fidelity rendering for novel view synthesis, storage requirements pose challenges for their widespread adoption. Although various compression techniques have been proposed, previous art suffers from a common limitation: for any existing 3DGS, per-scene optimization is needed to achieve compression, making the compression sluggish and slow. To address this issue, we introduce Fast Compression of 3D Gaussian Splatting (FCGS), an optimization-free model that can compress 3DGS representations rapidly in a single feed-forward pass, which significantly reduces compression time from minutes to seconds. To enhance compression efficiency, we propose a multi-path entropy module that assigns Gaussian attributes to different entropy constraint paths for balance between size and fidelity. We also carefully design both inter- and intra-Gaussian context models to remove redundancies among the unstructured Gaussian blobs. Overall, FCGS achieves a compression ratio of over 20X while maintaining fidelity, surpassing most per-scene SOTA optimization-based methods. Our code is available at: https://github.com/YihangChen-ee/FCGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09396v1">Close-up-GS: Enhancing Close-Up View Synthesis in 3D Gaussian Splatting with Progressive Self-Training</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated impressive performance in synthesizing novel views after training on a given set of viewpoints. However, its rendering quality deteriorates when the synthesized view deviates significantly from the training views. This decline occurs due to (1) the model's difficulty in generalizing to out-of-distribution scenarios and (2) challenges in interpolating fine details caused by substantial resolution changes and occlusions. A notable case of this limitation is close-up view generation--producing views that are significantly closer to the object than those in the training set. To tackle this issue, we propose a novel approach for close-up view generation based by progressively training the 3DGS model with self-generated data. Our solution is based on three key ideas. First, we leverage the See3D model, a recently introduced 3D-aware generative model, to enhance the details of rendered views. Second, we propose a strategy to progressively expand the ``trust regions'' of the 3DGS model and update a set of reference views for See3D. Finally, we introduce a fine-tuning strategy to carefully update the 3DGS model with training data generated from the above schemes. We further define metrics for close-up views evaluation to facilitate better research on this problem. By conducting evaluations on specifically selected scenarios for close-up views, our proposed approach demonstrates a clear advantage over competitive solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09342v1">GASPACHO: Gaussian Splatting for Controllable Humans and Objects</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      We present GASPACHO: a method for generating photorealistic controllable renderings of human-object interactions. Given a set of multi-view RGB images of human-object interactions, our method reconstructs animatable templates of the human and object as separate sets of Gaussians simultaneously. Different from existing work, which focuses on human reconstruction and ignores objects as background, our method explicitly reconstructs both humans and objects, thereby allowing for controllable renderings of novel human object interactions in different poses from novel-camera viewpoints. During reconstruction, we constrain the Gaussians that generate rendered images to be a linear function of a set of canonical Gaussians. By simply changing the parameters of the linear deformation functions after training, our method can generate renderings of novel human-object interaction in novel poses from novel camera viewpoints. We learn the 3D Gaussian properties of the canonical Gaussians on the underlying 2D manifold of the canonical human and object templates. This in turns requires a canonical object template with a fixed UV unwrapping. To define such an object template, we use a feature based representation to track the object across the multi-view sequence. We further propose an occlusion aware photometric loss that allows for reconstructions under significant occlusions. Several experiments on two human-object datasets - BEHAVE and DNA-Rendering - demonstrate that our method allows for high-quality reconstruction of human and object templates under significant occlusion and the synthesis of controllable renderings of novel human-object interactions in novel human poses from novel camera views.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00746v2">DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3D-GS) have shown remarkable success in representing 3D scenes and generating high-quality, novel views in real-time. However, 3D-GS and its variants assume that input images are captured based on pinhole imaging and are fully in focus. This assumption limits their applicability, as real-world images often feature shallow depth-of-field (DoF). In this paper, we introduce DoF-Gaussian, a controllable depth-of-field method for 3D-GS. We develop a lens-based imaging model based on geometric optics principles to control DoF effects. To ensure accurate scene geometry, we incorporate depth priors adjusted per scene, and we apply defocus-to-focus adaptation to minimize the gap in the circle of confusion. We also introduce a synthetic dataset to assess refocusing capabilities and the model's ability to learn precise lens parameters. Our framework is customizable and supports various interactive applications. Extensive experiments confirm the effectiveness of our method. Our project is available at https://dof-gaussian.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09332v1">SDD-4DGS: Static-Dynamic Aware Decoupling in Gaussian Splatting for 4D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Dynamic and static components in scenes often exhibit distinct properties, yet most 4D reconstruction methods treat them indiscriminately, leading to suboptimal performance in both cases. This work introduces SDD-4DGS, the first framework for static-dynamic decoupled 4D scene reconstruction based on Gaussian Splatting. Our approach is built upon a novel probabilistic dynamic perception coefficient that is naturally integrated into the Gaussian reconstruction pipeline, enabling adaptive separation of static and dynamic components. With carefully designed implementation strategies to realize this theoretical framework, our method effectively facilitates explicit learning of motion patterns for dynamic elements while maintaining geometric stability for static structures. Extensive experiments on five benchmark datasets demonstrate that SDD-4DGS consistently outperforms state-of-the-art methods in reconstruction fidelity, with enhanced detail restoration for static structures and precise modeling of dynamic motions. The code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01425v2">EVA-Gaussian: 3D Gaussian-based Real-time Human Novel View Synthesis under Diverse Multi-view Camera Settings</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Feed-forward based 3D Gaussian Splatting methods have demonstrated exceptional capability in real-time novel view synthesis for human models. However, current approaches are confined to either dense viewpoint configurations or restricted image resolutions. These limitations hinder their flexibility in free-viewpoint rendering across a wide range of camera view angle discrepancies, and also restrict their ability to recover fine-grained human details in real time using commonly available GPUs. To address these challenges, we propose a novel pipeline named EVA-Gaussian for 3D human novel view synthesis across diverse multi-view camera settings. Specifically, we first design an Efficient Cross-View Attention (EVA) module to effectively fuse cross-view information under high resolution inputs and sparse view settings, while minimizing temporal and computational overhead. Additionally, we introduce a feature refinement mechianism to predict the attributes of the 3D Gaussians and assign a feature value to each Gaussian, enabling the correction of artifacts caused by geometric inaccuracies in position estimation and enhancing overall visual fidelity. Experimental results on the THuman2.0 and THumansit datasets showcase the superiority of EVA-Gaussian in rendering quality across diverse camera settings. Project page: https://zhenliuzju.github.io/huyingdong/EVA-Gaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06838v4">Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Implicit Neural Representation (INR) has been successfully employed for Arbitrary-scale Super-Resolution (ASR). However, INR-based models need to query the multi-layer perceptron module numerous times and render a pixel in each query, resulting in insufficient representation capability and computational efficiency. Recently, Gaussian Splatting (GS) has shown its advantages over INR in both visual quality and rendering speed in 3D tasks, which motivates us to explore whether GS can be employed for the ASR task. However, directly applying GS to ASR is exceptionally challenging because the original GS is an optimization-based method through overfitting each single scene, while in ASR we aim to learn a single model that can generalize to different images and scaling factors. We overcome these challenges by developing two novel techniques. Firstly, to generalize GS for ASR, we elaborately design an architecture to predict the corresponding image-conditioned Gaussians of the input low-resolution image in a feed-forward manner. Each Gaussian can fit the shape and direction of an area of complex textures, showing powerful representation capability. Secondly, we implement an efficient differentiable 2D GPU/CUDA-based scale-aware rasterization to render super-resolved images by sampling discrete RGB values from the predicted continuous Gaussians. Via end-to-end training, our optimized network, namely GSASR, can perform ASR for any image and unseen scaling factors. Extensive experiments validate the effectiveness of our proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05757v3">Locality-aware Gaussian Compression for Fast and High-quality Rendering</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 Accepted to ICLR 2025. Project page: https://seungjooshin.github.io/LocoGS
    </div>
    <details class="paper-abstract">
      We present LocoGS, a locality-aware 3D Gaussian Splatting (3DGS) framework that exploits the spatial coherence of 3D Gaussians for compact modeling of volumetric scenes. To this end, we first analyze the local coherence of 3D Gaussian attributes, and propose a novel locality-aware 3D Gaussian representation that effectively encodes locally-coherent Gaussian attributes using a neural field representation with a minimal storage requirement. On top of the novel representation, LocoGS is carefully designed with additional components such as dense initialization, an adaptive spherical harmonics bandwidth scheme and different encoding schemes for different Gaussian attributes to maximize compression performance. Experimental results demonstrate that our approach outperforms the rendering quality of existing compact Gaussian representations for representative real-world 3D datasets while achieving from 54.6$\times$ to 96.6$\times$ compressed storage size and from 2.1$\times$ to 2.4$\times$ rendering speed than 3DGS. Even our approach also demonstrates an averaged 2.4$\times$ higher rendering speed than the state-of-the-art compression method with comparable compression performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02104v2">DynOMo: Online Point Tracking by Dynamic Online Monocular Gaussian Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 Accepted to 3DV 2025
    </div>
    <details class="paper-abstract">
      Reconstructing scenes and tracking motion are two sides of the same coin. Tracking points allow for geometric reconstruction [14], while geometric reconstruction of (dynamic) scenes allows for 3D tracking of points over time [24, 39]. The latter was recently also exploited for 2D point tracking to overcome occlusion ambiguities by lifting tracking directly into 3D [38]. However, above approaches either require offline processing or multi-view camera setups both unrealistic for real-world applications like robot navigation or mixed reality. We target the challenge of online 2D and 3D point tracking from unposed monocular camera input introducing Dynamic Online Monocular Reconstruction (DynOMo). We leverage 3D Gaussian splatting to reconstruct dynamic scenes in an online fashion. Our approach extends 3D Gaussians to capture new content and object motions while estimating camera movements from a single RGB frame. DynOMo stands out by enabling emergence of point trajectories through robust image feature reconstruction and a novel similarity-enhanced regularization term, without requiring any correspondence-level supervision. It sets the first baseline for online point tracking with monocular unposed cameras, achieving performance on par with existing methods. We aim to inspire the community to advance online point tracking and reconstruction, expanding the applicability to diverse real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19800v2">TrackGS: Optimizing COLMAP-Free 3D Gaussian Splatting with Global Track Constraints</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has advanced ability on novel view synthesis, it still depends on accurate pre-computaed camera parameters, which are hard to obtain and prone to noise. Previous COLMAP-Free methods optimize camera poses using local constraints, but they often struggle in complex scenarios. To address this, we introduce TrackGS, which incorporates feature tracks to globally constrain multi-view geometry. We select the Gaussians associated with each track, which will be trained and rescaled to an infinitesimally small size to guarantee the spatial accuracy. We also propose minimizing both reprojection and backprojection errors for better geometric consistency. Moreover, by deriving the gradient of intrinsics, we unify camera parameter estimation with 3DGS training into a joint optimization framework, achieving SOTA performance on challenging datasets with severe camera movements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07446v2">EigenGS Representation: From Eigenspace to Gaussian Image Space</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Principal Component Analysis (PCA), a classical dimensionality reduction technique, and 2D Gaussian representation, an adaptation of 3D Gaussian Splatting for image representation, offer distinct approaches to modeling visual data. We present EigenGS, a novel method that bridges these paradigms through an efficient transformation pipeline connecting eigenspace and image-space Gaussian representations. Our approach enables instant initialization of Gaussian parameters for new images without requiring per-image optimization from scratch, dramatically accelerating convergence. EigenGS introduces a frequency-aware learning mechanism that encourages Gaussians to adapt to different scales, effectively modeling varied spatial frequencies and preventing artifacts in high-resolution reconstruction. Extensive experiments demonstrate that EigenGS not only achieves superior reconstruction quality compared to direct 2D Gaussian fitting but also reduces necessary parameter count and training time. The results highlight EigenGS's effectiveness and generalization ability across images with varying resolutions and diverse categories, making Gaussian-based image representation both high-quality and viable for real-time applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09040v1">Motion Blender Gaussian Splatting for Dynamic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Gaussian splatting has emerged as a powerful tool for high-fidelity reconstruction of dynamic scenes. However, existing methods primarily rely on implicit motion representations, such as encoding motions into neural networks or per-Gaussian parameters, which makes it difficult to further manipulate the reconstructed motions. This lack of explicit controllability limits existing methods to replaying recorded motions only, which hinders a wider application. To address this, we propose Motion Blender Gaussian Splatting (MB-GS), a novel framework that uses motion graph as an explicit and sparse motion representation. The motion of graph links is propagated to individual Gaussians via dual quaternion skinning, with learnable weight painting functions determining the influence of each link. The motion graphs and 3D Gaussians are jointly optimized from input videos via differentiable rendering. Experiments show that MB-GS achieves state-of-the-art performance on the iPhone dataset while being competitive on HyperNeRF. Additionally, we demonstrate the application potential of our method in generating novel object motions and robot demonstrations through motion editing. Video demonstrations can be found at https://mlzxy.github.io/mbgs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11057v3">CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 Accepted to CVPR 2025
    </div>
    <details class="paper-abstract">
      We propose Covisibility Map-based Gaussian Splatting (CoMapGS), designed to recover underrepresented sparse regions in sparse novel view synthesis. CoMapGS addresses both high- and low-uncertainty regions by constructing covisibility maps, enhancing initial point clouds, and applying uncertainty-aware weighted supervision with a proximity classifier. Our contributions are threefold: (1) CoMapGS reframes novel view synthesis by leveraging covisibility maps as a core component to address region-specific uncertainty levels; (2) Enhanced initial point clouds for both low- and high-uncertainty regions compensate for sparse COLMAP-derived point clouds, improving reconstruction quality and benefiting few-shot 3DGS methods; (3) Adaptive supervision with covisibility-score-based weighting and proximity classification achieves consistent performance gains across scenes with various sparsity scores derived from covisibility maps. Experimental results demonstrate that CoMapGS outperforms state-of-the-art methods on datasets including Mip-NeRF 360 and LLFF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00144v3">Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated remarkable effectiveness in novel view synthesis (NVS). However, 3DGS tends to overfit when trained with sparse views, limiting its generalization to novel viewpoints. In this paper, we address this overfitting issue by introducing Self-Ensembling Gaussian Splatting (SE-GS). We achieve self-ensembling by incorporating an uncertainty-aware perturbation strategy during training. A $\mathbf{\Delta}$-model and a $\mathbf{\Sigma}$-model are jointly trained on the available images. The $\mathbf{\Delta}$-model is dynamically perturbed based on rendering uncertainty across training steps, generating diverse perturbed models with negligible computational overhead. Discrepancies between the $\mathbf{\Sigma}$-model and these perturbed models are minimized throughout training, forming a robust ensemble of 3DGS models. This ensemble, represented by the $\mathbf{\Sigma}$-model, is then used to generate novel-view images during inference. Experimental results on the LLFF, Mip-NeRF360, DTU, and MVImgNet datasets demonstrate that our approach enhances NVS quality under few-shot training conditions, outperforming existing state-of-the-art methods. The code is released at: https://sailor-z.github.io/projects/SEGS.html.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05511v2">Free Your Hands: Lightweight Relightable Turntable Capture Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) from multiple captured photos of an object is a widely studied problem. Achieving high quality typically requires dense sampling of input views, which can lead to frustrating and tedious manual labor. Manually positioning cameras to maintain an optimal desired distribution can be difficult for humans, and if a good distribution is found, it is not easy to replicate. Additionally, the captured data can suffer from motion blur and defocus due to human error. In this paper, we present a lightweight object capture pipeline to reduce the manual workload and standardize the acquisition setup. We use a consumer turntable to carry the object and a tripod to hold the camera. As the turntable rotates, we automatically capture dense samples from various views and lighting conditions; we can repeat this for several camera positions. This way, we can easily capture hundreds of valid images in several minutes without hands-on effort. However, in the object reference frame, the light conditions vary; this is harmful to a standard NVS method like 3D Gaussian splatting (3DGS) which assumes fixed lighting. We design a neural radiance representation conditioned on light rotations, which addresses this issue and allows relightability as an additional benefit. We demonstrate our pipeline using 3DGS as the underlying framework, achieving competitive quality compared to previous methods with exhaustive acquisition and showcasing its potential for relighting and harmonization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04981v2">AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Obtaining high-quality 3D semantic occupancy from raw sensor data remains an essential yet challenging task, often requiring extensive manual labeling. In this work, we propose AutoOcc, an vision-centric automated pipeline for open-ended semantic occupancy annotation that integrates differentiable Gaussian splatting guided by vision-language models. We formulate the open-ended semantic occupancy reconstruction task to automatically generate scene occupancy by combining attention maps from vision-language models and foundation vision models. We devise semantic-aware Gaussians as intermediate geometric descriptors and propose a cumulative Gaussian-to-voxel splatting algorithm that enables effective and efficient occupancy annotation. Our framework outperforms existing automated occupancy annotation methods without human labels. AutoOcc also enables open-ended semantic occupancy auto-labeling, achieving robust performance in both static and dynamically complex scenarios. All the source codes and trained models will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06677v2">REArtGS: Reconstructing and Generating Articulated Objects via 3D Gaussian Splatting with Geometric and Motion Constraints</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 11pages, 6 figures
    </div>
    <details class="paper-abstract">
      Articulated objects, as prevalent entities in human life, their 3D representations play crucial roles across various applications. However, achieving both high-fidelity textured surface reconstruction and dynamic generation for articulated objects remains challenging for existing methods. In this paper, we present REArtGS, a novel framework that introduces additional geometric and motion constraints to 3D Gaussian primitives, enabling high-quality textured surface reconstruction and generation for articulated objects. Specifically, given multi-view RGB images of arbitrary two states of articulated objects, we first introduce an unbiased Signed Distance Field (SDF) guidance to regularize Gaussian opacity fields, enhancing geometry constraints and improving surface reconstruction quality. Then we establish deformable fields for 3D Gaussians constrained by the kinematic structures of articulated objects, achieving unsupervised generation of surface meshes in unseen states. Extensive experiments on both synthetic and real datasets demonstrate our approach achieves high-quality textured surface reconstruction for given states, and enables high-fidelity surface generation for unseen states. Codes will be released within the next four months and the project website is at https://sites.google.com/view/reartgs/home.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13753v2">FAST-Splat: Fast, Ambiguity-Free Semantics Transfer in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      We present FAST-Splat for fast, ambiguity-free semantic Gaussian Splatting, which seeks to address the main limitations of existing semantic Gaussian Splatting methods, namely: slow training and rendering speeds; high memory usage; and ambiguous semantic object localization. We take a bottom-up approach in deriving FAST-Splat, dismantling the limitations of closed-set semantic distillation to enable open-set (open-vocabulary) semantic distillation. Ultimately, this key approach enables FAST-Splat to provide precise semantic object localization results, even when prompted with ambiguous user-provided natural-language queries. Further, by exploiting the explicit form of the Gaussian Splatting scene representation to the fullest extent, FAST-Splat retains the remarkable training and rendering speeds of Gaussian Splatting. Precisely, while existing semantic Gaussian Splatting methods distill semantics into a separate neural field or utilize neural models for dimensionality reduction, FAST-Splat directly augments each Gaussian with specific semantic codes, preserving the training, rendering, and memory-usage advantages of Gaussian Splatting over neural field methods. These Gaussian-specific semantic codes, together with a hash-table, enable semantic similarity to be measured with open-vocabulary user prompts and further enable FAST-Splat to respond with unambiguous semantic object labels and $3$D masks, unlike prior methods. In experiments, we demonstrate that FAST-Splat is 6x to 8x faster to train, achieves between 18x to 51x faster rendering speeds, and requires about 6x smaller GPU memory, compared to the best-competing semantic Gaussian Splatting methods. Further, FAST-Splat achieves relatively similar or better semantic segmentation performance compared to existing methods. After the review period, we will provide links to the project website and the codebase.
    </details>
</div>
