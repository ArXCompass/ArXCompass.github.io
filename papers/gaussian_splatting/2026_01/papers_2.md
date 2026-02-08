# gaussian splatting - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07484v1">R3-RECON: Radiance-Field-Free Active Reconstruction via Renderability</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 18 pages, 11 figures
    </div>
    <details class="paper-abstract">
      In active reconstruction, an embodied agent must decide where to look next to efficiently acquire views that support high-quality novel-view rendering. Recent work on active view planning for neural rendering largely derives next-best-view (NBV) criteria by backpropagating through radiance fields or estimating information entropy over 3D Gaussian primitives. While effective, these strategies tightly couple view selection to heavy, representation-specific mechanisms and fail to account for the computational and resource constraints required for lightweight online deployment. In this paper, we revisit active reconstruction from a renderability-centric perspective. We propose $\mathbb{R}^{3}$-RECON, a radiance-fields-free active reconstruction framework that induces an implicit, pose-conditioned renderability field over SE(3) from a lightweight voxel map. Our formulation aggregates per-voxel online observation statistics into a unified scalar renderability score that is cheap to update and can be queried in closed form at arbitrary candidate viewpoints in milliseconds, without requiring gradients or radiance-field training. This renderability field is strongly correlated with image-space reconstruction error, naturally guiding NBV selection. We further introduce a panoramic extension that estimates omnidirectional (360$^\circ$) view utility to accelerate candidate evaluation. In the standard indoor Replica dataset, $\mathbb{R}^{3}$-RECON achieves more uniform novel-view quality and higher 3D Gaussian splatting (3DGS) reconstruction accuracy than recent active GS baselines with matched view and time budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.19142v2">CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Recent works in 3D multimodal learning have made remarkable progress. However, typically 3D multimodal models are only capable of handling point clouds. Compared to the emerging 3D representation technique, 3D Gaussian Splatting (3DGS), the spatially sparse point cloud cannot depict the texture information of 3D objects, resulting in inferior reconstruction capabilities. This limitation constrains the potential of point cloud-based 3D multimodal representation learning. In this paper, we present CLIP-GS, a novel multimodal representation learning framework grounded in 3DGS. We introduce the GS Tokenizer to generate serialized gaussian tokens, which are then processed through transformer layers pre-initialized with weights from point cloud models, resulting in the 3DGS embeddings. CLIP-GS leverages contrastive loss between 3DGS and the visual-text embeddings of CLIP, and we introduce an image voting loss to guide the directionality and convergence of gradient optimization. Furthermore, we develop an efficient way to generate triplets of 3DGS, images, and text, facilitating CLIP-GS in learning unified multimodal representations. Leveraging the well-aligned multimodal representations, CLIP-GS demonstrates versatility and outperforms point cloud-based models on various 3D tasks, including multimodal retrieval, zero-shot, and few-shot classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.00800v2">SWAGSplatting: Semantic-guided Water-scene Augmented Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Accurate 3D reconstruction in underwater environments remains a challenging task due to light attenuation, scattering, and limited visibility. While recent AI-based approaches have advanced underwater imaging, they often overlook high-level semantic understanding, which is crucial for reconstructing complex scenes. In this paper, we propose SWAGSplatting, \textit{Semantic-guided Water-scene Augmented Gaussian Splatting}, a novel multimodal framework that integrates language and vision knowledge into 3D Gaussian Splatting for robust and high-fidelity underwater reconstruction. Each Gaussian primitive is augmented with a learnable semantic feature, supervised using CLIP-based embeddings extracted from region-level semantic cues. A dedicated semantic consistency loss enforces alignment between geometric reconstruction and scene semantics. In addition, a stage-wise optimisation strategy combining coarse-to-fine learning with late-stage parameter refinement improves training stability and visual quality. Furthermore, we propose a 3D Gaussian Primitives Reallocation strategy to address the imbalanced distribution of primitives introduced by naive point cloud densification. Extensive experiments on the SeaThru-NeRF and Submerged3D datasets demonstrate that SWAGSplatting consistently outperforms state-of-the-art methods across PSNR, SSIM, and LPIPS metrics, achieving up to a 3.48 dB improvement in PSNR, enabling more accurate and semantically coherent underwater scene reconstruction for applications in marine perception and exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.13339v2">Volume Encoding Gaussians: Transfer Function-Agnostic 3D Gaussians for Volume Rendering</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Visualizing the large-scale datasets output by HPC resources presents a difficult challenge, as the memory and compute power required become prohibitively expensive for end user systems. Novel view synthesis techniques can address this by producing a small, interactive model of the data, requiring only a set of training images to learn from. While these models allow accessible visualization of large data and complex scenes, they do not provide the interactions needed for scientific volumes, as they do not support interactive selection of transfer functions and lighting parameters. To address this, we introduce Volume Encoding Gaussians (VEG), a 3D Gaussian-based representation for volume visualization that supports arbitrary color and opacity mappings. Unlike prior 3D Gaussian Splatting (3DGS) methods that store color and opacity for each Gaussian, VEG decouple the visual appearance from the data representation by encoding only scalar values, enabling transfer function-agnostic rendering of 3DGS models. To ensure complete scalar field coverage, we introduce an opacity-guided training strategy, using differentiable rendering with multiple transfer functions to optimize our data representation. This allows VEG to preserve fine features across the full scalar range of a dataset while remaining independent of any specific transfer function. Across a diverse set of volume datasets, we demonstrate that our method outperforms the state-of-the-art on transfer functions unseen during training, while requiring a fraction of the memory and training time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07963v1">3DGS-Drag: Dragging Gaussians for Intuitive Point-Based 3D Editing</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      The transformative potential of 3D content creation has been progressively unlocked through advancements in generative models. Recently, intuitive drag editing with geometric changes has attracted significant attention in 2D editing yet remains challenging for 3D scenes. In this paper, we introduce 3DGS-Drag -- a point-based 3D editing framework that provides efficient, intuitive drag manipulation of real 3D scenes. Our approach bridges the gap between deformation-based and 2D-editing-based 3D editing methods, addressing their limitations to geometry-related content editing. We leverage two key innovations: deformation guidance utilizing 3D Gaussian Splatting for consistent geometric modifications and diffusion guidance for content correction and visual quality enhancement. A progressive editing strategy further supports aggressive 3D drag edits. Our method enables a wide range of edits, including motion change, shape adjustment, inpainting, and content extension. Experimental results demonstrate the effectiveness of 3DGS-Drag in various scenes, achieving state-of-the-art performance in geometry-related 3D content editing. Notably, the editing is efficient, taking 10 to 20 minutes on a single RTX 4090 GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.20031v4">MG-SLAM: Structure Gaussian Splatting SLAM with Manhattan World Hypothesis</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 IEEE Transactions on Automation Science and Engineering
    </div>
    <details class="paper-abstract">
      Gaussian Splatting SLAMs have made significant advancements in improving the efficiency and fidelity of real-time reconstructions. However, these systems often encounter incomplete reconstructions in complex indoor environments, characterized by substantial holes due to unobserved geometry caused by obstacles or limited view angles. To address this challenge, we present Manhattan Gaussian SLAM, an RGB-D system that leverages the Manhattan World hypothesis to enhance geometric accuracy and completeness. By seamlessly integrating fused line segments derived from structured scenes, our method ensures robust tracking in textureless indoor areas. Moreover, The extracted lines and planar surface assumption allow strategic interpolation of new Gaussians in regions of missing geometry, enabling efficient scene completion. Extensive experiments conducted on both synthetic and real-world scenes demonstrate that these advancements enable our method to achieve state-of-the-art performance, marking a substantial improvement in the capabilities of Gaussian SLAM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06831v1">SARA: Scene-Aware Reconstruction Accelerator</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 This work has been submitted to the 2026 International Conference on Pattern Recognition (ICPR) for possible publication
    </div>
    <details class="paper-abstract">
      We present SARA (Scene-Aware Reconstruction Accelerator), a geometry-driven pair selection module for Structure-from-Motion (SfM). Unlike conventional pipelines that select pairs based on visual similarity alone, SARA introduces geometry-first pair selection by scoring reconstruction informativeness - the product of overlap and parallax - before expensive matching. A lightweight pre-matching stage uses mutual nearest neighbors and RANSAC to estimate these cues, then constructs an Information-Weighted Spanning Tree (IWST) augmented with targeted edges for loop closure, long-baseline anchors, and weak-view reinforcement. Compared to exhaustive matching, SARA reduces rotation errors by 46.5+-5.5% and translation errors by 12.5+-6.5% across modern learned detectors, while achieving at most 50x speedup through 98% pair reduction (from 30,848 to 580 pairs). This reduces matching complexity from quadratic to quasi-linear, maintaining within +-3% of baseline reconstruction metrics for 3D Gaussian Splatting and SVRaster.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03422v2">What Is The Best 3D Scene Representation for Robotics? From Geometric to Foundation Models</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      In this paper, we provide a comprehensive overview of existing scene representation methods for robotics, covering traditional representations such as point clouds, voxels, signed distance functions (SDF), and scene graphs, as well as more recent neural representations like Neural Radiance Fields (NeRF), 3D Gaussian Splatting (3DGS), and the emerging Foundation Models. While current SLAM and localization systems predominantly rely on sparse representations like point clouds and voxels, dense scene representations are expected to play a critical role in downstream tasks such as navigation and obstacle avoidance. Moreover, neural representations such as NeRF, 3DGS, and foundation models are well-suited for integrating high-level semantic features and language-based priors, enabling more comprehensive 3D scene understanding and embodied intelligence. In this paper, we categorized the core modules of robotics into five parts (Perception, Mapping, Localization, Navigation, Manipulation). We start by presenting the standard formulation of different scene representation methods and comparing the advantages and disadvantages of scene representation across different modules. This survey is centered around the question: What is the best 3D scene representation for robotics? We then discuss the future development trends of 3D scene representations, with a particular focus on how the 3D Foundation Model could replace current methods as the unified solution for future robotic applications. The remaining challenges in fully realizing this model are also explored. We aim to offer a valuable resource for both newcomers and experienced researchers to explore the future of 3D scene representations and their application in robotics. We have published an open-source project on GitHub and will continue to add new works and technologies to this project.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.18992v2">VPGS-SLAM: Voxel-based Progressive 3D Gaussian SLAM in Large-Scale Scenes</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has recently shown promising results in dense visual SLAM. However, existing 3DGS-based SLAM methods are all constrained to small-room scenarios and struggle with memory explosion in large-scale scenes and long sequences. To this end, we propose VPGS-SLAM, the first 3DGS-based large-scale RGBD SLAM framework for both indoor and outdoor scenarios. We design a novel voxel-based progressive 3D Gaussian mapping method with multiple submaps for compact and accurate scene representation in large-scale and long-sequence scenes. This allows us to scale up to arbitrary scenes and improves robustness (even under pose drifts). In addition, we propose a 2D-3D fusion camera tracking method to achieve robust and accurate camera tracking in both indoor and outdoor large-scale scenes. Furthermore, we design a 2D-3D Gaussian loop closure method to eliminate pose drift. We further propose a submap fusion method with online distillation to achieve global consistency in large-scale scenes when detecting a loop. Experiments on various indoor and outdoor datasets demonstrate the superiority and generalizability of the proposed framework. The code will be open source on https://github.com/dtc111111/vpgs-slam.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06479v1">SRFlow: A Dataset and Regularization Model for High-Resolution Facial Optical Flow via Splatting Rasterization</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      Facial optical flow supports a wide range of tasks in facial motion analysis. However, the lack of high-resolution facial optical flow datasets has hindered progress in this area. In this paper, we introduce Splatting Rasterization Flow (SRFlow), a high-resolution facial optical flow dataset, and Splatting Rasterization Guided FlowNet (SRFlowNet), a facial optical flow model with tailored regularization losses. These losses constrain flow predictions using masks and gradients computed via difference or Sobel operator. This effectively suppresses high-frequency noise and large-scale errors in texture-less or repetitive-pattern regions, enabling SRFlowNet to be the first model explicitly capable of capturing high-resolution skin motion guided by Gaussian splatting rasterization. Experiments show that training with the SRFlow dataset improves facial optical flow estimation across various optical flow models, reducing end-point error (EPE) by up to 42% (from 0.5081 to 0.2953). Furthermore, when coupled with the SRFlow dataset, SRFlowNet achieves up to a 48% improvement in F1-score (from 0.4733 to 0.6947) on a composite of three micro-expression datasets. These results demonstrate the value of advancing both facial optical flow estimation and micro-expression recognition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05584v1">GS-DMSR: Dynamic Sensitive Multi-scale Manifold Enhancement for Accelerated High-Quality 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      In the field of 3D dynamic scene reconstruction, how to balance model convergence rate and rendering quality has long been a critical challenge that urgently needs to be addressed, particularly in high-precision modeling of scenes with complex dynamic motions. To tackle this issue, this study proposes the GS-DMSR method. By quantitatively analyzing the dynamic evolution process of Gaussian attributes, this mechanism achieves adaptive gradient focusing, enabling it to dynamically identify significant differences in the motion states of Gaussian models. It then applies differentiated optimization strategies to Gaussian models with varying degrees of significance, thereby significantly improving the model convergence rate. Additionally, this research integrates a multi-scale manifold enhancement module, which leverages the collaborative optimization of an implicit nonlinear decoder and an explicit deformation field to enhance the modeling efficiency for complex deformation scenes. Experimental results demonstrate that this method achieves a frame rate of up to 96 FPS on synthetic datasets, while effectively reducing both storage overhead and training time.Our code and data are available at https://anonymous.4open.science/r/GS-DMSR-2212.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.13713v5">SLAM&Render: A Benchmark for the Intersection Between Neural Rendering, Gaussian Splatting and SLAM</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 9 pages, 8 figures, submitted to The International Journal of Robotics Research (IJRR)
    </div>
    <details class="paper-abstract">
      Models and methods originally developed for Novel View Synthesis and Scene Rendering, such as Neural Radiance Fields (NeRF) and Gaussian Splatting, are increasingly being adopted as representations in Simultaneous Localization and Mapping (SLAM). However, existing datasets fail to include the specific challenges of both fields, such as sequential operations and, in many settings, multi-modality in SLAM or generalization across viewpoints and illumination conditions in neural rendering. Additionally, the data are often collected using sensors which are handheld or mounted on drones or mobile robots, which complicates the accurate reproduction of sensor motions. To bridge these gaps, we introduce SLAM&Render, a novel dataset designed to benchmark methods in the intersection between SLAM, Novel View Rendering and Gaussian Splatting. Recorded with a robot manipulator, it uniquely includes 40 sequences with time-synchronized RGB-D images, IMU readings, robot kinematic data, and ground-truth pose streams. By releasing robot kinematic data, the dataset also enables the assessment of recent integrations of SLAM paradigms within robotic applications. The dataset features five setups with consumer and industrial objects under four controlled lighting conditions, each with separate training and test trajectories. All sequences are static with different levels of object rearrangements and occlusions. Our experimental results, obtained with several baselines from the literature, validate SLAM&Render as a relevant benchmark for this emerging research area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05511v1">GaussianSwap: Animatable Video Face Swapping with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      We introduce GaussianSwap, a novel video face swapping framework that constructs a 3D Gaussian Splatting based face avatar from a target video while transferring identity from a source image to the avatar. Conventional video swapping frameworks are limited to generating facial representations in pixel-based formats. The resulting swapped faces exist merely as a set of unstructured pixels without any capacity for animation or interactive manipulation. Our work introduces a paradigm shift from conventional pixel-based video generation to the creation of high-fidelity avatar with swapped faces. The framework first preprocesses target video to extract FLAME parameters, camera poses and segmentation masks, and then rigs 3D Gaussian splats to the FLAME model across frames, enabling dynamic facial control. To ensure identity preserving, we propose an compound identity embedding constructed from three state-of-the-art face recognition models for avatar finetuning. Finally, we render the face-swapped avatar on the background frames to obtain the face-swapped video. Experimental results demonstrate that GaussianSwap achieves superior identity preservation, visual clarity and temporal consistency, while enabling previously unattainable interactive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06285v1">NAS-GS: Noise-Aware Sonar Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      Underwater sonar imaging plays a crucial role in various applications, including autonomous navigation in murky water, marine archaeology, and environmental monitoring. However, the unique characteristics of sonar images, such as complex noise patterns and the lack of elevation information, pose significant challenges for 3D reconstruction and novel view synthesis. In this paper, we present NAS-GS, a novel Noise-Aware Sonar Gaussian Splatting framework specifically designed to address these challenges. Our approach introduces a Two-Ways Splatting technique that accurately models the dual directions for intensity accumulation and transmittance calculation inherent in sonar imaging, significantly improving rendering speed without sacrificing quality. Moreover, we propose a Gaussian Mixture Model (GMM) based noise model that captures complex sonar noise patterns, including side-lobes, speckle, and multi-path noise. This model enhances the realism of synthesized images while preventing 3D Gaussian overfitting to noise, thereby improving reconstruction accuracy. We demonstrate state-of-the-art performance on both simulated and real-world large-scale offshore sonar scenarios, achieving superior results in novel view synthesis and 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.21226v2">Frequency-Aware Gaussian Splatting Decomposition</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 Accepted to the International Conference on 3D Vision (3DV) 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D-GS) enables efficient novel view synthesis, but treats all frequencies uniformly, making it difficult to separate coarse structure from fine detail. Recent works have started to exploit frequency signals, but lack explicit frequency decomposition of the 3D representation itself. We propose a frequency-aware decomposition that organizes 3D Gaussians into groups corresponding to Laplacian-pyramid subbands of the input images. Each group is trained with spatial frequency regularization to confine it to its target frequency, while higher-frequency bands use signed residual colors to capture fine details that may be missed by lower-frequency reconstructions. A progressive coarse-to-fine training schedule stabilizes the decomposition. Our method achieves state-of-the-art reconstruction quality and rendering speed among all LOD-capable methods. In addition to improved interpretability, our method enables dynamic level-of-detail rendering, progressive streaming, foveated rendering, promptable 3D focus, and artistic filtering. Our code will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05853v1">LayerGS: Decomposition and Inpainting of Layered 3D Human Avatars via 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      We propose a novel framework for decomposing arbitrarily posed humans into animatable multi-layered 3D human avatars, separating the body and garments. Conventional single-layer reconstruction methods lock clothing to one identity, while prior multi-layer approaches struggle with occluded regions. We overcome both limitations by encoding each layer as a set of 2D Gaussians for accurate geometry and photorealistic rendering, and inpainting hidden regions with a pretrained 2D diffusion model via score-distillation sampling (SDS). Our three-stage training strategy first reconstructs the coarse canonical garment via single-layer reconstruction, followed by multi-layer training to jointly recover the inner-layer body and outer-layer garment details. Experiments on two 3D human benchmark datasets (4D-Dress, Thuman2.0) show that our approach achieves better rendering quality and layer decomposition and recomposition than the previous state-of-the-art, enabling realistic virtual try-on under novel viewpoints and poses, and advancing practical creation of high-fidelity 3D human assets for immersive applications. Our code is available at https://github.com/RockyXu66/LayerGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05738v1">FeatureSLAM: Feature-enriched 3D gaussian splatting SLAM in real time</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      We present a real-time tracking SLAM system that unifies efficient camera tracking with photorealistic feature-enriched mapping using 3D Gaussian Splatting (3DGS). Our main contribution is integrating dense feature rasterization into the novel-view synthesis, aligned with a visual foundation model. This yields strong semantics, going beyond basic RGB-D input, aiding both tracking and mapping accuracy. Unlike previous semantic SLAM approaches (which embed pre-defined class labels) FeatureSLAM enables entirely new downstream tasks via free-viewpoint, open-set segmentation. Across standard benchmarks, our method achieves real-time tracking, on par with state-of-the-art systems while improving tracking stability and map fidelity without prohibitive compute. Quantitatively, we obtain 9\% lower pose error and 8\% higher mapping accuracy compared to recent fixed-set SLAM baselines. Our results confirm that real-time feature-embedded SLAM, is not only valuable for enabling new downstream applications. It also improves the performance of the underlying tracking and mapping subsystems, providing semantic and language masking results that are on-par with offline 3DGS models, alongside state-of-the-art tracking, depth and RGB rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04754v1">ProFuse: Efficient Cross-View Context Fusion for Open-Vocabulary 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      We present ProFuse, an efficient context-aware framework for open-vocabulary 3D scene understanding with 3D Gaussian Splatting (3DGS). The pipeline enhances cross-view consistency and intra-mask cohesion within a direct registration setup, adding minimal overhead and requiring no render-supervised fine-tuning. Instead of relying on a pretrained 3DGS scene, we introduce a dense correspondence-guided pre-registration phase that initializes Gaussians with accurate geometry while jointly constructing 3D Context Proposals via cross-view clustering. Each proposal carries a global feature obtained through weighted aggregation of member embeddings, and this feature is fused onto Gaussians during direct registration to maintain per-primitive language coherence across views. With associations established in advance, semantic fusion requires no additional optimization beyond standard reconstruction, and the model retains geometric refinement without densification. ProFuse achieves strong open-vocabulary 3DGS understanding while completing semantic attachment in about five minutes per scene, which is two times faster than SOTA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.13287v2">InnerGS: Internal Scenes Reconstruction and Segmentation via Factorized 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently gained popularity for efficient scene rendering by representing scenes as explicit sets of anisotropic 3D Gaussians. However, most existing work focuses primarily on modeling external surfaces. In this work, we target the reconstruction of internal scenes, which is crucial for applications that require a deep understanding of an object's interior. By directly modeling a continuous volumetric density through the inner 3D Gaussian distribution, our model effectively reconstructs smooth and detailed internal structures from sparse sliced data. Beyond high-fidelity reconstruction, we further demonstrate the framework's potential for downstream tasks such as segmentation. By integrating language features, we extend our approach to enable text-guided segmentation of medical scenes via natural language queries. Our approach eliminates the need for camera poses, is plug-and-play, and is inherently compatible with any data modalities. We provide cuda implementation at: https://github.com/Shuxin-Liang/InnerGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05368v1">MOSAIC-GS: Monocular Scene Reconstruction via Advanced Initialization for Complex Dynamic Environments</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      We present MOSAIC-GS, a novel, fully explicit, and computationally efficient approach for high-fidelity dynamic scene reconstruction from monocular videos using Gaussian Splatting. Monocular reconstruction is inherently ill-posed due to the lack of sufficient multiview constraints, making accurate recovery of object geometry and temporal coherence particularly challenging. To address this, we leverage multiple geometric cues, such as depth, optical flow, dynamic object segmentation, and point tracking. Combined with rigidity-based motion constraints, these cues allow us to estimate preliminary 3D scene dynamics during an initialization stage. Recovering scene dynamics prior to the photometric optimization reduces reliance on motion inference from visual appearance alone, which is often ambiguous in monocular settings. To enable compact representations, fast training, and real-time rendering while supporting non-rigid deformations, the scene is decomposed into static and dynamic components. Each Gaussian in the dynamic part of the scene is assigned a trajectory represented as time-dependent Poly-Fourier curve for parameter-efficient motion encoding. We demonstrate that MOSAIC-GS achieves substantially faster optimization and rendering compared to existing methods, while maintaining reconstruction quality on par with state-of-the-art approaches across standard monocular dynamic scene benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06212v1">Akasha 2: Hamiltonian State Space Duality and Visual-Language Joint Embedding Predictive Architectur</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 12 pages, 6 figures, 3 tables. Includes appendices with pseudocode and implementation details. Supplementary materials eventually at github.com/yanimeziani/akasha
    </div>
    <details class="paper-abstract">
      We present Akasha 2, a state-of-the-art multimodal architecture that integrates Hamiltonian State Space Duality (H-SSD) with Visual-Language Joint Embedding Predictive Architecture (VL-JEPA). The system leverages the Mamba-3 Selective State Space Model (SSM) augmented by a Sparse Mixture of Hamiltonian Experts (SMoE-HE) that enforces latent physical conservation laws through symplectic integration. For visual synthesis, we introduce Hamiltonian Flow Matching (HFM) and persistent 3D Gaussian Splatting (3DGS), enabling ultra-low latency (<50ms) on mobile hardware. This work establishes a new paradigm in latent world models, achieving unprecedented spatiotemporal coherence through a holographic memory architecture. Our approach demonstrates that incorporating physics-inspired inductive biases into neural architectures yields significant improvements: state-of-the-art video prediction (FVD: 287), 4x faster visual synthesis than diffusion models, and 3-18x inference speedup over transformer baselines while maintaining energy conservation over extended horizons.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04984v1">OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 Accepted to AAAI 2026. Project page: https://oceansplat.github.io
    </div>
    <details class="paper-abstract">
      We introduce OceanSplat, a novel 3D Gaussian Splatting-based approach for accurately representing 3D geometry in underwater scenes. To overcome multi-view inconsistencies caused by underwater optical degradation, our method enforces trinocular view consistency by rendering horizontally and vertically translated camera views relative to each input view and aligning them via inverse warping. Furthermore, these translated camera views are used to derive a synthetic epipolar depth prior through triangulation, which serves as a self-supervised depth regularizer. These geometric constraints facilitate the spatial optimization of 3D Gaussians and preserve scene structure in underwater environments. We also propose a depth-aware alpha adjustment that modulates the opacity of 3D Gaussians during early training based on their $z$-component and viewing direction, deterring the formation of medium-induced primitives. With our contributions, 3D Gaussians are disentangled from the scattering medium, enabling robust representation of object geometry and significantly reducing floating artifacts in reconstructed underwater scenes. Experiments on real-world underwater and simulated scenes demonstrate that OceanSplat substantially outperforms existing methods for both scene reconstruction and restoration in scattering media.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03824v1">IDESplat: Iterative Depth Probability Estimation for Generalizable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Generalizable 3D Gaussian Splatting aims to directly predict Gaussian parameters using a feed-forward network for scene reconstruction. Among these parameters, Gaussian means are particularly difficult to predict, so depth is usually estimated first and then unprojected to obtain the Gaussian sphere centers. Existing methods typically rely solely on a single warp to estimate depth probability, which hinders their ability to fully leverage cross-view geometric cues, resulting in unstable and coarse depth maps. To address this limitation, we propose IDESplat, which iteratively applies warp operations to boost depth probability estimation for accurate Gaussian mean prediction. First, to eliminate the inherent instability of a single warp, we introduce a Depth Probability Boosting Unit (DPBU) that integrates epipolar attention maps produced by cascading warp operations in a multiplicative manner. Next, we construct an iterative depth estimation process by stacking multiple DPBUs, progressively identifying potential depth candidates with high likelihood. As IDESplat iteratively boosts depth probability estimates and updates the depth candidates, the depth map is gradually refined, resulting in accurate Gaussian means. We conduct experiments on RealEstate10K, ACID, and DL3DV. IDESplat achieves outstanding reconstruction quality and state-of-the-art performance with real-time efficiency. On RE10K, it outperforms DepthSplat by 0.33 dB in PSNR, using only 10.7% of the parameters and 70% of the memory. Additionally, our IDESplat improves PSNR by 2.95 dB over DepthSplat on the DTU dataset in cross-dataset experiments, demonstrating its strong generalization ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03510v1">G2P: Gaussian-to-Point Attribute Alignment for Boundary-Aware 3D Semantic Segmentation</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Semantic segmentation on point clouds is critical for 3D scene understanding. However, sparse and irregular point distributions provide limited appearance evidence, making geometry-only features insufficient to distinguish objects with similar shapes but distinct appearances (e.g., color, texture, material). We propose Gaussian-to-Point (G2P), which transfers appearance-aware attributes from 3D Gaussian Splatting to point clouds for more discriminative and appearance-consistent segmentation. Our G2P address the misalignment between optimized Gaussians and original point geometry by establishing point-wise correspondences. By leveraging Gaussian opacity attributes, we resolve the geometric ambiguity that limits existing models. Additionally, Gaussian scale attributes enable precise boundary localization in complex 3D scenes. Extensive experiments demonstrate that our approach achieves superior performance on standard benchmarks and shows significant improvements on geometrically challenging classes, all without any 2D or language supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04348v1">SCAR-GS: Spatial Context Attention for Residuals in Progressive Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting have allowed for real-time, high-fidelity novel view synthesis. Nonetheless, these models have significant storage requirements for large and medium-sized scenes, hindering their deployment over cloud and streaming services. Some of the most recent progressive compression techniques for these models rely on progressive masking and scalar quantization techniques to reduce the bitrate of Gaussian attributes using spatial context models. While effective, scalar quantization may not optimally capture the correlations of high-dimensional feature vectors, which can potentially limit the rate-distortion performance. In this work, we introduce a novel progressive codec for 3D Gaussian Splatting that replaces traditional methods with a more powerful Residual Vector Quantization approach to compress the primitive features. Our key contribution is an auto-regressive entropy model, guided by a multi-resolution hash grid, that accurately predicts the conditional probability of each successive transmitted index, allowing for coarse and refinement layers to be compressed with high efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08219v2">SAGOnline: Segment Any Gaussians Online</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 11 pages, 6 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has emerged as a powerful paradigm for explicit 3D scene representation, yet achieving efficient and consistent 3D segmentation remains challenging. Existing segmentation approaches typically rely on high-dimensional feature lifting, which causes costly optimization, implicit semantics, and task-specific constraints. We present \textbf{Segment Any Gaussians Online (SAGOnline)}, a unified, zero-shot framework that achieves real-time, cross-view consistent segmentation without scene-specific training. SAGOnline decouples the monolithic segmentation problem into lightweight sub-tasks. By integrating video foundation models (e.g., SAM 2), we first generate temporally consistent 2D masks across rendered views. Crucially, instead of learning continuous feature fields, we introduce a \textbf{Rasterization-aware Geometric Consensus} mechanism that leverages the traceability of the Gaussian rasterization pipeline. This allows us to deterministically map 2D predictions to explicit, discrete 3D primitive labels in real-time. This discrete representation eliminates the memory and computational burden of feature distillation, enabling instant inference. Extensive evaluations on NVOS and SPIn-NeRF benchmarks demonstrate that SAGOnline achieves state-of-the-art accuracy (92.7\% and 95.2\% mIoU) while operating at the fastest speed at 27 ms per frame. By providing a flexible interface for diverse foundation models, our framework supports instant prompt, instance, and semantic segmentation, paving the way for interactive 3D understanding in AR/VR and robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03024v1">SA-ResGS: Self-Augmented Residual 3D Gaussian Splatting for Next Best View Selection</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      We propose Self-Augmented Residual 3D Gaussian Splatting (SA-ResGS), a novel framework to stabilize uncertainty quantification and enhancing uncertainty-aware supervision in next-best-view (NBV) selection for active scene reconstruction. SA-ResGS improves both the reliability of uncertainty estimates and their effectiveness for supervision by generating Self-Augmented point clouds (SA-Points) via triangulation between a training view and a rasterized extrapolated view, enabling efficient scene coverage estimation. While improving scene coverage through physically guided view selection, SA-ResGS also addresses the challenge of under-supervised Gaussians, exacerbated by sparse and wide-baseline views, by introducing the first residual learning strategy tailored for 3D Gaussian Splatting. This targeted supervision enhances gradient flow in high-uncertainty Gaussians by combining uncertainty-driven filtering with dropout- and hard-negative-mining-inspired sampling. Our contributions are threefold: (1) a physically grounded view selection strategy that promotes efficient and uniform scene coverage; (2) an uncertainty-aware residual supervision scheme that amplifies learning signals for weakly contributing Gaussians, improving training stability and uncertainty estimation across scenes with diverse camera distributions; (3) an implicit unbiasing of uncertainty quantification as a consequence of constrained view selection and residual supervision, which together mitigate conflicting effects of wide-baseline exploration and sparse-view ambiguity in NBV planning. Experiments on active view selection demonstrate that SA-ResGS outperforms state-of-the-art baselines in both reconstruction quality and view selection robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.09111v2">DenseSplat: Densifying Gaussian Splatting SLAM with Neural Radiance Prior</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 IEEE Transactions on Visualization and Computer Graphics
    </div>
    <details class="paper-abstract">
      Gaussian SLAM systems excel in real-time rendering and fine-grained reconstruction compared to NeRF-based systems. However, their reliance on extensive keyframes is impractical for deployment in real-world robotic systems, which typically operate under sparse-view conditions that can result in substantial holes in the map. To address these challenges, we introduce DenseSplat, the first SLAM system that effectively combines the advantages of NeRF and 3DGS. DenseSplat utilizes sparse keyframes and NeRF priors for initializing primitives that densely populate maps and seamlessly fill gaps. It also implements geometry-aware primitive sampling and pruning strategies to manage granularity and enhance rendering efficiency. Moreover, DenseSplat integrates loop closure and bundle adjustment, significantly enhancing frame-to-frame tracking accuracy. Extensive experiments on multiple large-scale datasets demonstrate that DenseSplat achieves superior performance in tracking and mapping compared to current state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02716v1">CAMO: Category-Agnostic 3D Motion Transfer from Monocular 2D Videos</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Project website: https://camo-project-page.github.io/
    </div>
    <details class="paper-abstract">
      Motion transfer from 2D videos to 3D assets is a challenging problem, due to inherent pose ambiguities and diverse object shapes, often requiring category-specific parametric templates. We propose CAMO, a category-agnostic framework that transfers motion to diverse target meshes directly from monocular 2D videos without relying on predefined templates or explicit 3D supervision. The core of CAMO is a morphology-parameterized articulated 3D Gaussian splatting model combined with dense semantic correspondences to jointly adapt shape and pose through optimization. This approach effectively alleviates shape-pose ambiguities, enabling visually faithful motion transfer for diverse categories. Experimental results demonstrate superior motion accuracy, efficiency, and visual coherence compared to existing methods, significantly advancing motion transfer in varied object categories and casual video scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03357v1">RelightAnyone: A Generalized Relightable 3D Gaussian Head Model</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a standard approach to reconstruct and render photorealistic 3D head avatars. A major challenge is to relight the avatars to match any scene illumination. For high quality relighting, existing methods require subjects to be captured under complex time-multiplexed illumination, such as one-light-at-a-time (OLAT). We propose a new generalized relightable 3D Gaussian head model that can relight any subject observed in a single- or multi-view images without requiring OLAT data for that subject. Our core idea is to learn a mapping from flat-lit 3DGS avatars to corresponding relightable Gaussian parameters for that avatar. Our model consists of two stages: a first stage that models flat-lit 3DGS avatars without OLAT lighting, and a second stage that learns the mapping to physically-based reflectance parameters for high-quality relighting. This two-stage design allows us to train the first stage across diverse existing multi-view datasets without OLAT lighting ensuring cross-subject generalization, where we learn a dataset-specific lighting code for self-supervised lighting alignment. Subsequently, the second stage can be trained on a significantly smaller dataset of subjects captured under OLAT illumination. Together, this allows our method to generalize well and relight any subject from the first stage as if we had captured them under OLAT lighting. Furthermore, we can fit our model to unseen subjects from as little as a single image, allowing several applications in novel view synthesis and relighting for digital avatars.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03319v1">CaricatureGS: Exaggerating 3D Gaussian Splatting Faces With Gaussian Curvature</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      A photorealistic and controllable 3D caricaturization framework for faces is introduced. We start with an intrinsic Gaussian curvature-based surface exaggeration technique, which, when coupled with texture, tends to produce over-smoothed renders. To address this, we resort to 3D Gaussian Splatting (3DGS), which has recently been shown to produce realistic free-viewpoint avatars. Given a multiview sequence, we extract a FLAME mesh, solve a curvature-weighted Poisson equation, and obtain its exaggerated form. However, directly deforming the Gaussians yields poor results, necessitating the synthesis of pseudo-ground-truth caricature images by warping each frame to its exaggerated 2D representation using local affine transformations. We then devise a training scheme that alternates real and synthesized supervision, enabling a single Gaussian collection to represent both natural and exaggerated avatars. This scheme improves fidelity, supports local edits, and allows continuous control over the intensity of the caricature. In order to achieve real-time deformations, an efficient interpolation between the original and exaggerated surfaces is introduced. We further analyze and show that it has a bounded deviation from closed-form solutions. In both quantitative and qualitative evaluations, our results outperform prior work, delivering photorealistic, geometry-controlled caricature avatars.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03200v1">A High-Fidelity Digital Twin for Robotic Manipulation Based on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Under review of Journal of Robot Learning
    </div>
    <details class="paper-abstract">
      Developing high-fidelity, interactive digital twins is crucial for enabling closed-loop motion planning and reliable real-world robot execution, which are essential to advancing sim-to-real transfer. However, existing approaches often suffer from slow reconstruction, limited visual fidelity, and difficulties in converting photorealistic models into planning-ready collision geometry. We present a practical framework that constructs high-quality digital twins within minutes from sparse RGB inputs. Our system employs 3D Gaussian Splatting (3DGS) for fast, photorealistic reconstruction as a unified scene representation. We enhance 3DGS with visibility-aware semantic fusion for accurate 3D labelling and introduce an efficient, filter-based geometry conversion method to produce collision-ready models seamlessly integrated with a Unity-ROS2-MoveIt physics engine. In experiments with a Franka Emika Panda robot performing pick-and-place tasks, we demonstrate that this enhanced geometric accuracy effectively supports robust manipulation in real-world trials. These results demonstrate that 3DGS-based digital twins, enriched with semantic and geometric consistency, offer a fast, reliable, and scalable path from perception to manipulation in unstructured environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01847v1">ESGaussianFace: Emotional and Stylized Audio-Driven Facial Animation via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 13 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Most current audio-driven facial animation research primarily focuses on generating videos with neutral emotions. While some studies have addressed the generation of facial videos driven by emotional audio, efficiently generating high-quality talking head videos that integrate both emotional expressions and style features remains a significant challenge. In this paper, we propose ESGaussianFace, an innovative framework for emotional and stylized audio-driven facial animation. Our approach leverages 3D Gaussian Splatting to reconstruct 3D scenes and render videos, ensuring efficient generation of 3D consistent results. We propose an emotion-audio-guided spatial attention method that effectively integrates emotion features with audio content features. Through emotion-guided attention, the model is able to reconstruct facial details across different emotional states more accurately. To achieve emotional and stylized deformations of the 3D Gaussian points through emotion and style features, we introduce two 3D Gaussian deformation predictors. Futhermore, we propose a multi-stage training strategy, enabling the step-by-step learning of the character's lip movements, emotional variations, and style features. Our generated results exhibit high efficiency, high quality, and 3D consistency. Extensive experimental results demonstrate that our method outperforms existing state-of-the-art techniques in terms of lip movement accuracy, expression variation, and style feature expressiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02103v1">HeadLighter: Disentangling Illumination in Generative 3D Gaussian Heads via Lightstage Captures</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Recent 3D-aware head generative models based on 3D Gaussian Splatting achieve real-time, photorealistic and view-consistent head synthesis. However, a fundamental limitation persists: the deep entanglement of illumination and intrinsic appearance prevents controllable relighting. Existing disentanglement methods rely on strong assumptions to enable weakly supervised learning, which restricts their capacity for complex illumination. To address this challenge, we introduce HeadLighter, a novel supervised framework that learns a physically plausible decomposition of appearance and illumination in head generative models. Specifically, we design a dual-branch architecture that separately models lighting-invariant head attributes and physically grounded rendering components. A progressive disentanglement training is employed to gradually inject head appearance priors into the generative architecture, supervised by multi-view images captured under controlled light conditions with a light stage setup. We further introduce a distillation strategy to generate high-quality normals for realistic rendering. Experiments demonstrate that our method preserves high-quality generation and real-time rendering, while simultaneously supporting explicit lighting and viewpoint editing. We will publicly release our code and dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02102v1">360-GeoGS: Geometrically Consistent Feed-Forward 3D Gaussian Splatting Reconstruction for 360 Images</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      3D scene reconstruction is fundamental for spatial intelligence applications such as AR, robotics, and digital twins. Traditional multi-view stereo struggles with sparse viewpoints or low-texture regions, while neural rendering approaches, though capable of producing high-quality results, require per-scene optimization and lack real-time efficiency. Explicit 3D Gaussian Splatting (3DGS) enables efficient rendering, but most feed-forward variants focus on visual quality rather than geometric consistency, limiting accurate surface reconstruction and overall reliability in spatial perception tasks. This paper presents a novel feed-forward 3DGS framework for 360 images, capable of generating geometrically consistent Gaussian primitives while maintaining high rendering quality. A Depth-Normal geometric regularization is introduced to couple rendered depth gradients with normal information, supervising Gaussian rotation, scale, and position to improve point cloud and surface accuracy. Experimental results show that the proposed method maintains high rendering quality while significantly improving geometric consistency, providing an effective solution for 3D reconstruction in spatial perception tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02098v1">InpaintHuman: Reconstructing Occluded Humans with Multi-Scale UV Mapping and Identity-Preserving Diffusion Inpainting</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Reconstructing complete and animatable 3D human avatars from monocular videos remains challenging, particularly under severe occlusions. While 3D Gaussian Splatting has enabled photorealistic human rendering, existing methods struggle with incomplete observations, often producing corrupted geometry and temporal inconsistencies. We present InpaintHuman, a novel method for generating high-fidelity, complete, and animatable avatars from occluded monocular videos. Our approach introduces two key innovations: (i) a multi-scale UV-parameterized representation with hierarchical coarse-to-fine feature interpolation, enabling robust reconstruction of occluded regions while preserving geometric details; and (ii) an identity-preserving diffusion inpainting module that integrates textual inversion with semantic-conditioned guidance for subject-specific, temporally coherent completion. Unlike SDS-based methods, our approach employs direct pixel-level supervision to ensure identity fidelity. Experiments on synthetic benchmarks (PeopleSnapshot, ZJU-MoCap) and real-world scenarios (OcMotion) demonstrate competitive performance with consistent improvements in reconstruction quality across diverse poses and viewpoints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02072v1">SketchRodGS: Sketch-based Extraction of Slender Geometries for Animating Gaussian Splatting Scenes</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Presented at SIGGRAPH Asia 2025 (Technical Communications). Best Technical Communications Award
    </div>
    <details class="paper-abstract">
      Physics simulation of slender elastic objects often requires discretization as a polyline. However, constructing a polyline from Gaussian splatting is challenging as Gaussian splatting lacks connectivity information and the configuration of Gaussian primitives contains much noise. This paper presents a method to extract a polyline representation of the slender part of the objects in a Gaussian splatting scene from the user's sketching input. Our method robustly constructs a polyline mesh that represents the slender parts using the screen-space shortest path analysis that can be efficiently solved using dynamic programming. We demonstrate the effectiveness of our approach in several in-the-wild examples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02660v3">PMGS: Reconstruction of Projectile Motion Across Large Spatiotemporal Spans via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Modeling complex rigid motion across large spatiotemporal spans remains an unresolved challenge in dynamic reconstruction. Existing paradigms are mainly confined to short-term, small-scale deformation and offer limited consideration for physical consistency. This study proposes PMGS, focusing on reconstructing Projectile Motion via 3D Gaussian Splatting. The workflow comprises two stages: 1) Target Modeling: achieving object-centralized reconstruction through dynamic scene decomposition and an improved point density control; 2) Motion Recovery: restoring full motion sequences by learning per-frame SE(3) poses. We introduce an acceleration consistency constraint to bridge Newtonian mechanics and pose estimation, and design a dynamic simulated annealing strategy that adaptively schedules learning rates based on motion states. Furthermore, we devise a Kalman fusion scheme to optimize error accumulation from multi-source observations to mitigate disturbances. Experiments show PMGS's superior performance in reconstructing high-speed nonlinear rigid motion compared to mainstream dynamic methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01660v1">Animated 3DGS Avatars in Diverse Scenes with Consistent Lighting and Shadows</a></div>
    <div class="paper-meta">
      📅 2026-01-04
      | 💬 Our project page is available at https://miraymen.github.io/dgsm
    </div>
    <details class="paper-abstract">
      We present a method for consistent lighting and shadows when animated 3D Gaussian Splatting (3DGS) avatars interact with 3DGS scenes or with dynamic objects inserted into otherwise static scenes. Our key contribution is Deep Gaussian Shadow Maps (DGSM), a modern analogue of the classical shadow mapping algorithm tailored to the volumetric 3DGS representation. Building on the classic deep shadow mapping idea, we show that 3DGS admits closed form light accumulation along light rays, enabling volumetric shadow computation without meshing. For each estimated light, we tabulate transmittance over concentric radial shells and store them in octahedral atlases, which modern GPUs can sample in real time per query to attenuate affected scene Gaussians and thus cast and receive shadows consistently. To relight moving avatars, we approximate the local environment illumination with HDRI probes represented in a spherical harmonic (SH) basis and apply a fast per Gaussian radiance transfer, avoiding explicit BRDF estimation or offline optimization. We demonstrate environment consistent lighting for avatars from AvatarX and ActorsHQ, composited into ScanNet++, DL3DV, and SuperSplat scenes, and show interactions with inserted objects. Across single and multi avatar settings, DGSM and SH relighting operate fully in the volumetric 3DGS representation, yielding coherent shadows and relighting while avoiding meshing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09827v2">AHA! Animating Human Avatars in Diverse Scenes with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-04
      | 💬 Project page available at: https://miraymen.github.io/aha/
    </div>
    <details class="paper-abstract">
      We present a novel framework for animating humans in 3D scenes using 3D Gaussian Splatting (3DGS), a neural scene representation that has recently achieved state-of-the-art photorealistic results for novel-view synthesis but remains under-explored for human-scene animation and interaction. Unlike existing animation pipelines that use meshes or point clouds as the underlying 3D representation, our approach introduces the use of 3DGS as the 3D representation for animating humans in scenes. By representing humans and scenes as Gaussians, our approach allows geometry-consistent free-viewpoint rendering of humans interacting with 3D scenes. Our key insight is that rendering can be decoupled from motion synthesis, and each sub-problem can be addressed independently without the need for paired human-scene data. Central to our method is a Gaussian-aligned motion module that synthesizes motion without explicit scene geometry, using opacity-based cues and projected Gaussian structures to guide human placement and pose alignment. To ensure natural interactions, we further propose a human-scene Gaussian refinement optimization that enforces realistic contact and navigation. We evaluate our approach on scenes from Scannet++ and the SuperSplat library, and on avatars reconstructed from sparse and dense multi-view human capture. Finally, we demonstrate that our framework enables novel applications such as geometry-consistent free-viewpoint rendering of edited monocular RGB videos with newly animated humans, showcasing the unique advantages of 3DGS for monocular video-based human animation. To assess the full quality of our results, we encourage readers to view the supplementary material available at https://miraymen.github.io/aha/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15169v3">MeSS: City Mesh-Guided Outdoor Scene Generation with Cross-View Consistent Diffusion</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      Mesh models have become increasingly accessible for numerous cities; however, the lack of realistic textures restricts their application in virtual urban navigation and autonomous driving. To address this, this paper proposes MeSS (Meshbased Scene Synthesis) for generating high-quality, styleconsistent outdoor scenes with city mesh models serving as the geometric prior. While image and video diffusion models can leverage spatial layouts (such as depth maps or HD maps) as control conditions to generate street-level perspective views, they are not directly applicable to 3D scene generation. Video diffusion models excel at synthesizing consistent view sequences that depict scenes but often struggle to adhere to predefined camera paths or align accurately with rendered control videos. In contrast, image diffusion models, though unable to guarantee cross-view visual consistency, can produce more geometry-aligned results when combined with ControlNet. Building on this insight, our approach enhances image diffusion models by improving cross-view consistency. The pipeline comprises three key stages: first, we generate geometrically consistent sparse views using Cascaded Outpainting ControlNets; second, we propagate denser intermediate views via a component dubbed AGInpaint; and third, we globally eliminate visual inconsistencies (e.g., varying exposure) using the GCAlign module. Concurrently with generation, a 3D Gaussian Splatting (3DGS) scene is reconstructed by initializing Gaussian balls on the mesh surface. Our method outperforms existing approaches in both geometric alignment and generation quality. Once synthesized, the scene can be rendered in diverse styles through relighting and style transfer techniques. project page: https://albertchen98.github.io/mess/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00939v1">ShadowGS: Shadow-Aware 3D Gaussian Splatting for Satellite Imagery</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a novel paradigm for 3D reconstruction from satellite imagery. However, in multi-temporal satellite images, prevalent shadows exhibit significant inconsistencies due to varying illumination conditions. To address this, we propose ShadowGS, a novel framework based on 3DGS. It leverages a physics-based rendering equation from remote sensing, combined with an efficient ray marching technique, to precisely model geometrically consistent shadows while maintaining efficient rendering. Additionally, it effectively disentangles different illumination components and apparent attributes in the scene. Furthermore, we introduce a shadow consistency constraint that significantly enhances the geometric accuracy of 3D reconstruction. We also incorporate a novel shadow map prior to improve performance with sparse-view inputs. Extensive experiments demonstrate that ShadowGS outperforms current state-of-the-art methods in shadow decoupling accuracy, 3D reconstruction precision, and novel view synthesis quality, with only a few minutes of training. ShadowGS exhibits robust performance across various settings, including RGB, pansharpened, and sparse-view satellite inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01386v1">ParkGaussian: Surround-view 3D Gaussian Splatting for Autonomous Parking</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      Parking is a critical task for autonomous driving systems (ADS), with unique challenges in crowded parking slots and GPS-denied environments. However, existing works focus on 2D parking slot perception, mapping, and localization, 3D reconstruction remains underexplored, which is crucial for capturing complex spatial geometry in parking scenarios. Naively improving the visual quality of reconstructed parking scenes does not directly benefit autonomous parking, as the key entry point for parking is the slots perception module. To address these limitations, we curate the first benchmark named ParkRecon3D, specifically designed for parking scene reconstruction. It includes sensor data from four surround-view fisheye cameras with calibrated extrinsics and dense parking slot annotations. We then propose ParkGaussian, the first framework that integrates 3D Gaussian Splatting (3DGS) for parking scene reconstruction. To further improve the alignment between reconstruction and downstream parking slot detection, we introduce a slot-aware reconstruction strategy that leverages existing parking perception methods to enhance the synthesis quality of slot regions. Experiments on ParkRecon3D demonstrate that ParkGaussian achieves state-of-the-art reconstruction quality and better preserves perception consistency for downstream tasks. The code and dataset will be released at: https://github.com/wm-research/ParkGaussian
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.22324v2">AH-GS: Augmented 3D Gaussian Splatting for High-Frequency Detail Representation</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 need to revsie
    </div>
    <details class="paper-abstract">
      The 3D Gaussian Splatting (3D-GS) is a novel method for scene representation and view synthesis. Although Scaffold-GS achieves higher quality real-time rendering compared to the original 3D-GS, its fine-grained rendering of the scene is extremely dependent on adequate viewing angles. The spectral bias of neural network learning results in Scaffold-GS's poor ability to perceive and learn high-frequency information in the scene. In this work, we propose enhancing the manifold complexity of input features and using network-based feature map loss to improve the image reconstruction quality of 3D-GS models. We introduce AH-GS, which enables 3D Gaussians in structurally complex regions to obtain higher-frequency encodings, allowing the model to more effectively learn the high-frequency information of the scene. Additionally, we incorporate high-frequency reinforce loss to further enhance the model's ability to capture detailed frequency information. Our result demonstrates that our model significantly improves rendering fidelity, and in specific scenarios (e.g., MipNeRf360-garden), our method exceeds the rendering quality of Scaffold-GS in just 15K iterations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.12905v2">Matrix-free Second-order Optimization of Gaussian Splats with Residual Sampling</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is widely used for novel view synthesis due to its high rendering quality and fast inference time. However, 3DGS predominantly relies on first-order optimizers such as Adam, which leads to long training times. To address this limitation, we propose a novel second-order optimization strategy based on Levenberg-Marquardt (LM) and Conjugate Gradient (CG), which we specifically tailor towards Gaussian Splatting. Our key insight is that the Jacobian in 3DGS exhibits significant sparsity since each Gaussian affects only a limited number of pixels. We exploit this sparsity by proposing a matrix-free and GPU-parallelized LM optimization. To further improve its efficiency, we propose sampling strategies for both the camera views and loss function and, consequently, the normal equation, significantly reducing the computational complexity. In addition, we increase the convergence rate of the second-order approximation by introducing an effective heuristic to determine the learning rate that avoids the expensive computation cost of line search methods. As a result, our method achieves a $3\times$ speedup over standard LM and outperforms Adam by $~6\times$ when the Gaussian count is low while remaining competitive for moderate counts. Project Page: https://vcai.mpi-inf.mpg.de/projects/LM-IS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.14921v2">Stereo-GS: Multi-View Stereo Vision Model for Generalizable 3D Gaussian Splatting Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 ACM Multimedia 2025
    </div>
    <details class="paper-abstract">
      Generalizable 3D Gaussian Splatting reconstruction showcases advanced Image-to-3D content creation but requires substantial computational resources and large datasets, posing challenges to training models from scratch. Current methods usually entangle the prediction of 3D Gaussian geometry and appearance, which rely heavily on data-driven priors and result in slow regression speeds. To address this, we propose \method, a disentangled framework for efficient 3D Gaussian prediction. Our method extracts features from local image pairs using a stereo vision backbone and fuses them via global attention blocks. Dedicated point and Gaussian prediction heads generate multi-view point-maps for geometry and Gaussian features for appearance, combined as GS-maps to represent the 3DGS object. A refinement network enhances these GS-maps for high-quality reconstruction. Unlike existing methods that depend on camera parameters, our approach achieves pose-free 3D reconstruction, improving robustness and practicality. By reducing resource demands while maintaining high-quality outputs, \method provides an efficient, scalable solution for real-world 3D content generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00285v1">SV-GS: Sparse View 4D Reconstruction with Skeleton-Driven Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      Reconstructing a dynamic target moving over a large area is challenging. Standard approaches for dynamic object reconstruction require dense coverage in both the viewing space and the temporal dimension, typically relying on multi-view videos captured at each time step. However, such setups are only possible in constrained environments. In real-world scenarios, observations are often sparse over time and captured sparsely from diverse viewpoints (e.g., from security cameras), making dynamic reconstruction highly ill-posed. We present SV-GS, a framework that simultaneously estimates a deformation model and the object's motion over time under sparse observations. To initialize SV-GS, we leverage a rough skeleton graph and an initial static reconstruction as inputs to guide motion estimation. (Later, we show that this input requirement can be relaxed.) Our method optimizes a skeleton-driven deformation field composed of a coarse skeleton joint pose estimator and a module for fine-grained deformations. By making only the joint pose estimator time-dependent, our model enables smooth motion interpolation while preserving learned geometric details. Experiments on synthetic datasets show that our method outperforms existing approaches under sparse observations by up to 34% in PSNR, and achieves comparable performance to dense monocular video methods on real-world datasets despite using significantly fewer frames. Moreover, we demonstrate that the input initial static reconstruction can be replaced by a diffusion-based generative prior, making our method more practical for real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07743v2">UltraGS: Real-Time Physically-Decoupled Gaussian Splatting for Ultrasound Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Ultrasound imaging is a cornerstone of non-invasive clinical diagnostics, yet its limited field of view poses challenges for novel view synthesis. We present UltraGS, a real-time framework that adapts Gaussian Splatting to sensorless ultrasound imaging by integrating explicit radiance fields with lightweight, physics-inspired acoustic modeling. UltraGS employs depth-aware Gaussian primitives with learnable fields of view to improve geometric consistency under unconstrained probe motion, and introduces PD Rendering, a differentiable acoustic operator that combines low-order spherical harmonics with first-order wave effects for efficient intensity synthesis. We further present a clinical ultrasound dataset acquired under real-world scanning protocols. Extensive evaluations across three datasets demonstrate that UltraGS establishes a new performance-efficiency frontier, achieving state-of-the-art results in PSNR (up to 29.55) and SSIM (up to 0.89) while achieving real-time synthesis at 64.69 fps on a single GPU. The code and dataset are open-sourced at: https://github.com/Bean-Young/UltraGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.09977v3">A Survey on 3D Gaussian Splatting Applications: Segmentation, Editing, and Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 GitHub Repo: https://github.com/heshuting555/Awesome-3DGS-Applications
    </div>
    <details class="paper-abstract">
      In the context of novel view synthesis, 3D Gaussian Splatting (3DGS) has recently emerged as an efficient and competitive counterpart to Neural Radiance Field (NeRF), enabling high-fidelity photorealistic rendering in real time. Beyond novel view synthesis, the explicit and compact nature of 3DGS enables a wide range of downstream applications that require geometric and semantic understanding. This survey provides a comprehensive overview of recent progress in 3DGS applications. It first introduces 2D foundation models that support semantic understanding and control in 3DGS applications, followed by a review of NeRF-based methods that inform their 3DGS counterparts. We then categorize 3DGS applications into three foundational tasks: segmentation, editing, and generation, alongside additional functional applications built upon or tightly coupled with these foundational capabilities. For each, we summarize representative methods, supervision strategies, and learning paradigms, highlighting shared design principles and emerging trends. Commonly used datasets and evaluation protocols are also summarized, along with comparative analyses of recent methods across public benchmarks. To support ongoing research and development, a continually updated repository of papers, code, and resources is maintained at https://github.com/heshuting555/Awesome-3DGS-Applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00913v1">Clean-GS: Semantic Mask-Guided Pruning for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting produces high-quality scene reconstructions but generates hundreds of thousands of spurious Gaussians (floaters) scattered throughout the environment. These artifacts obscure objects of interest and inflate model sizes, hindering deployment in bandwidth-constrained applications. We present Clean-GS, a method for removing background clutter and floaters from 3DGS reconstructions using sparse semantic masks. Our approach combines whitelist-based spatial filtering with color-guided validation and outlier removal to achieve 60-80\% model compression while preserving object quality. Unlike existing 3DGS pruning methods that rely on global importance metrics, Clean-GS uses semantic information from as few as 3 segmentation masks (1\% of views) to identify and remove Gaussians not belonging to the target object. Our multi-stage approach consisting of (1) whitelist filtering via projection to masked regions, (2) depth-buffered color validation, and (3) neighbor-based outlier removal isolates monuments and objects from complex outdoor scenes. Experiments on Tanks and Temples show that Clean-GS reduces file sizes from 125MB to 47MB while maintaining rendering quality, making 3DGS models practical for web deployment and AR/VR applications. Our code is available at https://github.com/smlab-niser/clean-gs
    </details>
</div>
