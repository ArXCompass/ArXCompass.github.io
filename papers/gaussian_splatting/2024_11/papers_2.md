# gaussian splatting - 2024_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10947v1">Direct and Explicit 3D Generation from a Single Image</a></div>
    <div class="paper-meta">
      📅 2024-11-17
      | 💬 3DV 2025, Project page: https://hao-yu-wu.github.io/gen3d/
    </div>
    <details class="paper-abstract">
      Current image-to-3D approaches suffer from high computational costs and lack scalability for high-resolution outputs. In contrast, we introduce a novel framework to directly generate explicit surface geometry and texture using multi-view 2D depth and RGB images along with 3D Gaussian features using a repurposed Stable Diffusion model. We introduce a depth branch into U-Net for efficient and high quality multi-view, cross-domain generation and incorporate epipolar attention into the latent-to-pixel decoder for pixel-level multi-view consistency. By back-projecting the generated depth pixels into 3D space, we create a structured 3D representation that can be either rendered via Gaussian splatting or extracted to high-quality meshes, thereby leveraging additional novel view synthesis loss to further improve our performance. Extensive experiments demonstrate that our method surpasses existing baselines in geometry and texture quality while achieving significantly faster generation time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.11024v1">VeGaS: Video Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-17
    </div>
    <details class="paper-abstract">
      Implicit Neural Representations (INRs) employ neural networks to approximate discrete data as continuous functions. In the context of video data, such models can be utilized to transform the coordinates of pixel locations along with frame occurrence times (or indices) into RGB color values. Although INRs facilitate effective compression, they are unsuitable for editing purposes. One potential solution is to use a 3D Gaussian Splatting (3DGS) based model, such as the Video Gaussian Representation (VGR), which is capable of encoding video as a multitude of 3D Gaussians and is applicable for numerous video processing operations, including editing. Nevertheless, in this case, the capacity for modification is constrained to a limited set of basic transformations. To address this issue, we introduce the Video Gaussian Splatting (VeGaS) model, which enables realistic modifications of video data. To construct VeGaS, we propose a novel family of Folded-Gaussian distributions designed to capture nonlinear dynamics in a video stream and model consecutive frames by 2D Gaussians obtained as respective conditional distributions. Our experiments demonstrate that VeGaS outperforms state-of-the-art solutions in frame reconstruction tasks and allows realistic modifications of video data. The code is available at: https://github.com/gmum/VeGaS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10722v1">DGS-SLAM: Gaussian Splatting SLAM in Dynamic Environment</a></div>
    <div class="paper-meta">
      📅 2024-11-16
      | 💬 Preprint, Under review
    </div>
    <details class="paper-abstract">
      We introduce Dynamic Gaussian Splatting SLAM (DGS-SLAM), the first dynamic SLAM framework built on the foundation of Gaussian Splatting. While recent advancements in dense SLAM have leveraged Gaussian Splatting to enhance scene representation, most approaches assume a static environment, making them vulnerable to photometric and geometric inconsistencies caused by dynamic objects. To address these challenges, we integrate Gaussian Splatting SLAM with a robust filtering process to handle dynamic objects throughout the entire pipeline, including Gaussian insertion and keyframe selection. Within this framework, to further improve the accuracy of dynamic object removal, we introduce a robust mask generation method that enforces photometric consistency across keyframes, reducing noise from inaccurate segmentation and artifacts such as shadows. Additionally, we propose the loop-aware window selection mechanism, which utilizes unique keyframe IDs of 3D Gaussians to detect loops between the current and past frames, facilitating joint optimization of the current camera poses and the Gaussian map. DGS-SLAM achieves state-of-the-art performance in both camera tracking and novel view synthesis on various dynamic SLAM benchmarks, proving its effectiveness in handling real-world dynamic scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.10722v1">DGS-SLAM: Gaussian Splatting SLAM in Dynamic Environment</a></div>
    <div class="paper-meta">
      📅 2024-11-16
      | 💬 Preprint, Under review
    </div>
    <details class="paper-abstract">
      We introduce Dynamic Gaussian Splatting SLAM (DGS-SLAM), the first dynamic SLAM framework built on the foundation of Gaussian Splatting. While recent advancements in dense SLAM have leveraged Gaussian Splatting to enhance scene representation, most approaches assume a static environment, making them vulnerable to photometric and geometric inconsistencies caused by dynamic objects. To address these challenges, we integrate Gaussian Splatting SLAM with a robust filtering process to handle dynamic objects throughout the entire pipeline, including Gaussian insertion and keyframe selection. Within this framework, to further improve the accuracy of dynamic object removal, we introduce a robust mask generation method that enforces photometric consistency across keyframes, reducing noise from inaccurate segmentation and artifacts such as shadows. Additionally, we propose the loop-aware window selection mechanism, which utilizes unique keyframe IDs of 3D Gaussians to detect loops between the current and past frames, facilitating joint optimization of the current camera poses and the Gaussian map. DGS-SLAM achieves state-of-the-art performance in both camera tracking and novel view synthesis on various dynamic SLAM benchmarks, proving its effectiveness in handling real-world dynamic scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10546v1">The Oxford Spires Dataset: Benchmarking Large-Scale LiDAR-Visual Localisation, Reconstruction and Radiance Field Methods</a></div>
    <div class="paper-meta">
      📅 2024-11-15
      | 💬 Website: https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/
    </div>
    <details class="paper-abstract">
      This paper introduces a large-scale multi-modal dataset captured in and around well-known landmarks in Oxford using a custom-built multi-sensor perception unit as well as a millimetre-accurate map from a Terrestrial LiDAR Scanner (TLS). The perception unit includes three synchronised global shutter colour cameras, an automotive 3D LiDAR scanner, and an inertial sensor - all precisely calibrated. We also establish benchmarks for tasks involving localisation, reconstruction, and novel-view synthesis, which enable the evaluation of Simultaneous Localisation and Mapping (SLAM) methods, Structure-from-Motion (SfM) and Multi-view Stereo (MVS) methods as well as radiance field methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting. To evaluate 3D reconstruction the TLS 3D models are used as ground truth. Localisation ground truth is computed by registering the mobile LiDAR scans to the TLS 3D models. Radiance field methods are evaluated not only with poses sampled from the input trajectory, but also from viewpoints that are from trajectories which are distant from the training poses. Our evaluation demonstrates a key limitation of state-of-the-art radiance field methods: we show that they tend to overfit to the training poses/images and do not generalise well to out-of-sequence poses. They also underperform in 3D reconstruction compared to MVS systems using the same visual inputs. Our dataset and benchmarks are intended to facilitate better integration of radiance field methods and SLAM systems. The raw and processed data, along with software for parsing and evaluation, can be accessed at https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10504v1">USP-Gaussian: Unifying Spike-based Image Reconstruction, Pose Correction and Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-15
    </div>
    <details class="paper-abstract">
      Spike cameras, as an innovative neuromorphic camera that captures scenes with the 0-1 bit stream at 40 kHz, are increasingly employed for the 3D reconstruction task via Neural Radiance Fields (NeRF) or 3D Gaussian Splatting (3DGS). Previous spike-based 3D reconstruction approaches often employ a casecased pipeline: starting with high-quality image reconstruction from spike streams based on established spike-to-image reconstruction algorithms, then progressing to camera pose estimation and 3D reconstruction. However, this cascaded approach suffers from substantial cumulative errors, where quality limitations of initial image reconstructions negatively impact pose estimation, ultimately degrading the fidelity of the 3D reconstruction. To address these issues, we propose a synergistic optimization framework, \textbf{USP-Gaussian}, that unifies spike-based image reconstruction, pose correction, and Gaussian splatting into an end-to-end framework. Leveraging the multi-view consistency afforded by 3DGS and the motion capture capability of the spike camera, our framework enables a joint iterative optimization that seamlessly integrates information between the spike-to-image network and 3DGS. Experiments on synthetic datasets with accurate poses demonstrate that our method surpasses previous approaches by effectively eliminating cascading errors. Moreover, we integrate pose optimization to achieve robust 3D reconstruction in real-world scenarios with inaccurate initial poses, outperforming alternative methods by effectively reducing noise and preserving fine texture details. Our code, data and trained models will be available at \url{https://github.com/chenkang455/USP-Gaussian}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12677v3">GSFusion: Online RGB-D Mapping Where Gaussian Splatting Meets TSDF Fusion</a></div>
    <div class="paper-meta">
      📅 2024-11-15
    </div>
    <details class="paper-abstract">
      Traditional volumetric fusion algorithms preserve the spatial structure of 3D scenes, which is beneficial for many tasks in computer vision and robotics. However, they often lack realism in terms of visualization. Emerging 3D Gaussian splatting bridges this gap, but existing Gaussian-based reconstruction methods often suffer from artifacts and inconsistencies with the underlying 3D structure, and struggle with real-time optimization, unable to provide users with immediate feedback in high quality. One of the bottlenecks arises from the massive amount of Gaussian parameters that need to be updated during optimization. Instead of using 3D Gaussian as a standalone map representation, we incorporate it into a volumetric mapping system to take advantage of geometric information and propose to use a quadtree data structure on images to drastically reduce the number of splats initialized. In this way, we simultaneously generate a compact 3D Gaussian map with fewer artifacts and a volumetric map on the fly. Our method, GSFusion, significantly enhances computational efficiency without sacrificing rendering quality, as demonstrated on both synthetic and real datasets. Code will be available at https://github.com/goldoak/GSFusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10033v1">GSEditPro: 3D Gaussian Splatting Editing with Attention-based Progressive Localization</a></div>
    <div class="paper-meta">
      📅 2024-11-15
      | 💬 Pacific Graphics 2024
    </div>
    <details class="paper-abstract">
      With the emergence of large-scale Text-to-Image(T2I) models and implicit 3D representations like Neural Radiance Fields (NeRF), many text-driven generative editing methods based on NeRF have appeared. However, the implicit encoding of geometric and textural information poses challenges in accurately locating and controlling objects during editing. Recently, significant advancements have been made in the editing methods of 3D Gaussian Splatting, a real-time rendering technology that relies on explicit representation. However, these methods still suffer from issues including inaccurate localization and limited manipulation over editing. To tackle these challenges, we propose GSEditPro, a novel 3D scene editing framework which allows users to perform various creative and precise editing using text prompts only. Leveraging the explicit nature of the 3D Gaussian distribution, we introduce an attention-based progressive localization module to add semantic labels to each Gaussian during rendering. This enables precise localization on editing areas by classifying Gaussians based on their relevance to the editing prompts derived from cross-attention layers of the T2I model. Furthermore, we present an innovative editing optimization method based on 3D Gaussian Splatting, obtaining stable and refined editing results through the guidance of Score Distillation Sampling and pseudo ground truth. We prove the efficacy of our method through extensive experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11356v2">GSORB-SLAM: Gaussian Splatting SLAM benefits from ORB features and Transmittance information</a></div>
    <div class="paper-meta">
      📅 2024-11-15
    </div>
    <details class="paper-abstract">
      The emergence of 3D Gaussian Splatting (3DGS) has recently sparked a renewed wave of dense visual SLAM research. However, current methods face challenges such as sensitivity to artifacts and noise, sub-optimal selection of training viewpoints, and a lack of light global optimization. In this paper, we propose a dense SLAM system that tightly couples 3DGS with ORB features. We design a joint optimization approach for robust tracking and effectively reducing the impact of noise and artifacts. This involves combining novel geometric observations, derived from accumulated transmittance, with ORB features extracted from pixel data. Furthermore, to improve mapping quality, we propose an adaptive Gaussian expansion and regularization method that enables Gaussian primitives to represent the scene compactly. This is coupled with a viewpoint selection strategy based on the hybrid graph to mitigate over-fitting effects and enhance convergence quality. Finally, our approach achieves compact and high-quality scene representations and accurate localization. GSORB-SLAM has been evaluated on different datasets, demonstrating outstanding performance. The code will be available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09952v1">GGAvatar: Reconstructing Garment-Separated 3D Gaussian Splatting Avatars from Monocular Video</a></div>
    <div class="paper-meta">
      📅 2024-11-15
      | 💬 MMAsia'24 Accepted
    </div>
    <details class="paper-abstract">
      Avatar modelling has broad applications in human animation and virtual try-ons. Recent advancements in this field have focused on high-quality and comprehensive human reconstruction but often overlook the separation of clothing from the body. To bridge this gap, this paper introduces GGAvatar (Garment-separated 3D Gaussian Splatting Avatar), which relies on monocular videos. Through advanced parameterized templates and unique phased training, this model effectively achieves decoupled, editable, and realistic reconstruction of clothed humans. Comparative evaluations with other costly models confirm GGAvatar's superior quality and efficiency in modelling both clothed humans and separable garments. The paper also showcases applications in clothing editing, as illustrated in Figure 1, highlighting the model's benefits and the advantages of effective disentanglement. The code is available at https://github.com/J-X-Chen/GGAvatar/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02968v2">GSGAN: Adversarial Learning for Hierarchical Generation of 3D Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2024-11-15
      | 💬 NeurIPS 2024 / Project page: https://hse1032.github.io/gsgan
    </div>
    <details class="paper-abstract">
      Most advances in 3D Generative Adversarial Networks (3D GANs) largely depend on ray casting-based volume rendering, which incurs demanding rendering costs. One promising alternative is rasterization-based 3D Gaussian Splatting (3D-GS), providing a much faster rendering speed and explicit 3D representation. In this paper, we exploit Gaussian as a 3D representation for 3D GANs by leveraging its efficient and explicit characteristics. However, in an adversarial framework, we observe that a na\"ive generator architecture suffers from training instability and lacks the capability to adjust the scale of Gaussians. This leads to model divergence and visual artifacts due to the absence of proper guidance for initialized positions of Gaussians and densification to manage their scales adaptively. To address these issues, we introduce a generator architecture with a hierarchical multi-scale Gaussian representation that effectively regularizes the position and scale of generated Gaussians. Specifically, we design a hierarchy of Gaussians where finer-level Gaussians are parameterized by their coarser-level counterparts; the position of finer-level Gaussians would be located near their coarser-level counterparts, and the scale would monotonically decrease as the level becomes finer, modeling both coarse and fine details of the 3D scene. Experimental results demonstrate that ours achieves a significantly faster rendering speed (x100) compared to state-of-the-art 3D consistent GANs with comparable 3D generation capability. Project page: https://hse1032.github.io/gsgan.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12592v1">SPARS3R: Semantic Prior Alignment and Regularization for Sparse 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-11-15
    </div>
    <details class="paper-abstract">
      Recent efforts in Gaussian-Splat-based Novel View Synthesis can achieve photorealistic rendering; however, such capability is limited in sparse-view scenarios due to sparse initialization and over-fitting floaters. Recent progress in depth estimation and alignment can provide dense point cloud with few views; however, the resulting pose accuracy is suboptimal. In this work, we present SPARS3R, which combines the advantages of accurate pose estimation from Structure-from-Motion and dense point cloud from depth estimation. To this end, SPARS3R first performs a Global Fusion Alignment process that maps a prior dense point cloud to a sparse point cloud from Structure-from-Motion based on triangulated correspondences. RANSAC is applied during this process to distinguish inliers and outliers. SPARS3R then performs a second, Semantic Outlier Alignment step, which extracts semantically coherent regions around the outliers and performs local alignment in these regions. Along with several improvements in the evaluation process, we demonstrate that SPARS3R can achieve photorealistic rendering with sparse images and significantly outperforms existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09749v1">Adversarial Attacks Using Differentiable Rendering: A Survey</a></div>
    <div class="paper-meta">
      📅 2024-11-14
    </div>
    <details class="paper-abstract">
      Differentiable rendering methods have emerged as a promising means for generating photo-realistic and physically plausible adversarial attacks by manipulating 3D objects and scenes that can deceive deep neural networks (DNNs). Recently, differentiable rendering capabilities have evolved significantly into a diverse landscape of libraries, such as Mitsuba, PyTorch3D, and methods like Neural Radiance Fields and 3D Gaussian Splatting for solving inverse rendering problems that share conceptually similar properties commonly used to attack DNNs, such as back-propagation and optimization. However, the adversarial machine learning research community has not yet fully explored or understood such capabilities for generating attacks. Some key reasons are that researchers often have different attack goals, such as misclassification or misdetection, and use different tasks to accomplish these goals by manipulating different representation in a scene, such as the mesh or texture of an object. This survey adopts a task-oriented unifying framework that systematically summarizes common tasks, such as manipulating textures, altering illumination, and modifying 3D meshes to exploit vulnerabilities in DNNs. Our framework enables easy comparison of existing works, reveals research gaps and spotlights exciting future research directions in this rapidly evolving field. Through focusing on how these tasks enable attacks on various DNNs such as image classification, facial recognition, object detection, optical flow and depth estimation, our survey helps researchers and practitioners better understand the vulnerabilities of computer vision systems against photorealistic adversarial attacks that could threaten real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06526v2">GaussianCity: Generative Gaussian Splatting for Unbounded 3D City Generation</a></div>
    <div class="paper-meta">
      📅 2024-11-14
    </div>
    <details class="paper-abstract">
      3D city generation with NeRF-based methods shows promising generation results but is computationally inefficient. Recently 3D Gaussian Splatting (3D-GS) has emerged as a highly efficient alternative for object-level 3D generation. However, adapting 3D-GS from finite-scale 3D objects and humans to infinite-scale 3D cities is non-trivial. Unbounded 3D city generation entails significant storage overhead (out-of-memory issues), arising from the need to expand points to billions, often demanding hundreds of Gigabytes of VRAM for a city scene spanning 10km^2. In this paper, we propose GaussianCity, a generative Gaussian Splatting framework dedicated to efficiently synthesizing unbounded 3D cities with a single feed-forward pass. Our key insights are two-fold: 1) Compact 3D Scene Representation: We introduce BEV-Point as a highly compact intermediate representation, ensuring that the growth in VRAM usage for unbounded scenes remains constant, thus enabling unbounded city generation. 2) Spatial-aware Gaussian Attribute Decoder: We present spatial-aware BEV-Point decoder to produce 3D Gaussian attributes, which leverages Point Serializer to integrate the structural and contextual characteristics of BEV points. Extensive experiments demonstrate that GaussianCity achieves state-of-the-art results in both drone-view and street-view 3D city generation. Notably, compared to CityDreamer, GaussianCity exhibits superior performance with a speedup of 60 times (10.72 FPS v.s. 0.18 FPS).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07579v3">Projecting Gaussian Ellipsoids While Avoiding Affine Projection Approximation</a></div>
    <div class="paper-meta">
      📅 2024-11-14
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting has dominated novel-view synthesis with its real-time rendering speed and state-of-the-art rendering quality. However, during the rendering process, the use of the Jacobian of the affine approximation of the projection transformation leads to inevitable errors, resulting in blurriness, artifacts and a lack of scene consistency in the final rendered images. To address this issue, we introduce an ellipsoid-based projection method to calculate the projection of Gaussian ellipsoid onto the image plane, which is the primitive of 3D Gaussian Splatting. As our proposed ellipsoid-based projection method cannot handle Gaussian ellipsoids with camera origins inside them or parts lying below $z=0$ plane in the camera space, we designed a pre-filtering strategy. Experiments over multiple widely adopted benchmark datasets show that our ellipsoid-based projection method can enhance the rendering quality of 3D Gaussian Splatting and its extensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2407.09733v3">Textured-GS: Gaussian Splatting with Spatially Defined Color and Opacity</a></div>
    <div class="paper-meta">
      📅 2024-11-14
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      In this paper, we introduce Textured-GS, an innovative method for rendering Gaussian splatting that incorporates spatially defined color and opacity variations using Spherical Harmonics (SH). This approach enables each Gaussian to exhibit a richer representation by accommodating varying colors and opacities across its surface, significantly enhancing rendering quality compared to traditional methods. To demonstrate the merits of our approach, we have adapted the Mini-Splatting architecture to integrate textured Gaussians without increasing the number of Gaussians. Our experiments across multiple real-world datasets show that Textured-GS consistently outperforms both the baseline Mini-Splatting and standard 3DGS in terms of visual fidelity. The results highlight the potential of Textured-GS to advance Gaussian-based rendering technologies, promising more efficient and high-quality scene reconstructions. Our implementation is available at https://github.com/ZhentaoHuang/Textured-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08879v1">4D Gaussian Splatting in the Wild with Uncertainty-Aware Regularization</a></div>
    <div class="paper-meta">
      📅 2024-11-13
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Novel view synthesis of dynamic scenes is becoming important in various applications, including augmented and virtual reality. We propose a novel 4D Gaussian Splatting (4DGS) algorithm for dynamic scenes from casually recorded monocular videos. To overcome the overfitting problem of existing work for these real-world videos, we introduce an uncertainty-aware regularization that identifies uncertain regions with few observations and selectively imposes additional priors based on diffusion models and depth smoothness on such regions. This approach improves both the performance of novel view synthesis and the quality of training image reconstruction. We also identify the initialization problem of 4DGS in fast-moving dynamic regions, where the Structure from Motion (SfM) algorithm fails to provide reliable 3D landmarks. To initialize Gaussian primitives in such regions, we present a dynamic region densification method using the estimated depth maps and scene flow. Our experiments show that the proposed method improves the performance of 4DGS reconstruction from a video captured by a handheld monocular camera and also exhibits promising results in few-shot static scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10259v4">GaussianObject: High-Quality 3D Object Reconstruction from Four Views with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-13
      | 💬 ACM Transactions on Graphics (SIGGRAPH Asia 2024). Project page: https://gaussianobject.github.io/ Code: https://github.com/chensjtu/GaussianObject
    </div>
    <details class="paper-abstract">
      Reconstructing and rendering 3D objects from highly sparse views is of critical importance for promoting applications of 3D vision techniques and improving user experience. However, images from sparse views only contain very limited 3D information, leading to two significant challenges: 1) Difficulty in building multi-view consistency as images for matching are too few; 2) Partially omitted or highly compressed object information as view coverage is insufficient. To tackle these challenges, we propose GaussianObject, a framework to represent and render the 3D object with Gaussian splatting that achieves high rendering quality with only 4 input images. We first introduce techniques of visual hull and floater elimination, which explicitly inject structure priors into the initial optimization process to help build multi-view consistency, yielding a coarse 3D Gaussian representation. Then we construct a Gaussian repair model based on diffusion models to supplement the omitted object information, where Gaussians are further refined. We design a self-generating strategy to obtain image pairs for training the repair model. We further design a COLMAP-free variant, where pre-given accurate camera poses are not required, which achieves competitive quality and facilitates wider applications. GaussianObject is evaluated on several challenging datasets, including MipNeRF360, OmniObject3D, OpenIllumination, and our-collected unposed images, achieving superior performance from only four views and significantly outperforming previous SOTA methods. Our demo is available at https://gaussianobject.github.io/, and the code has been released at https://github.com/GaussianObject/GaussianObject.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09733v3">Textured-GS: Gaussian Splatting with Spatially Defined Color and Opacity</a></div>
    <div class="paper-meta">
      📅 2024-11-13
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      In this paper, we introduce Textured-GS, an innovative method for rendering Gaussian splatting that incorporates spatially defined color and opacity variations using Spherical Harmonics (SH). This approach enables each Gaussian to exhibit a richer representation by accommodating varying colors and opacities across its surface, significantly enhancing rendering quality compared to traditional methods. To demonstrate the merits of our approach, we have adapted the Mini-Splatting architecture to integrate textured Gaussians without increasing the number of Gaussians. Our experiments across multiple real-world datasets show that Textured-GS consistently outperforms both the baseline Mini-Splatting and standard 3DGS in terms of visual fidelity. The results highlight the potential of Textured-GS to advance Gaussian-based rendering technologies, promising more efficient and high-quality scene reconstructions. Our implementation is available at https://github.com/ZhentaoHuang/Textured-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08642v1">Towards More Accurate Fake Detection on Images Generated from Advanced Generative and Neural Rendering Models</a></div>
    <div class="paper-meta">
      📅 2024-11-13
      | 💬 13 pages, 8 Figures
    </div>
    <details class="paper-abstract">
      The remarkable progress in neural-network-driven visual data generation, especially with neural rendering techniques like Neural Radiance Fields and 3D Gaussian splatting, offers a powerful alternative to GANs and diffusion models. These methods can produce high-fidelity images and lifelike avatars, highlighting the need for robust detection methods. In response, an unsupervised training technique is proposed that enables the model to extract comprehensive features from the Fourier spectrum magnitude, thereby overcoming the challenges of reconstructing the spectrum due to its centrosymmetric properties. By leveraging the spectral domain and dynamically combining it with spatial domain information, we create a robust multimodal detector that demonstrates superior generalization capabilities in identifying challenging synthetic images generated by the latest image synthesis techniques. To address the absence of a 3D neural rendering-based fake image database, we develop a comprehensive database that includes images generated by diverse neural rendering techniques, providing a robust foundation for evaluating and advancing detection methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08453v1">Biomass phenotyping of oilseed rape through UAV multi-view oblique imaging with 3DGS and SAM model</a></div>
    <div class="paper-meta">
      📅 2024-11-13
    </div>
    <details class="paper-abstract">
      Biomass estimation of oilseed rape is crucial for optimizing crop productivity and breeding strategies. While UAV-based imaging has advanced high-throughput phenotyping, current methods often rely on orthophoto images, which struggle with overlapping leaves and incomplete structural information in complex field environments. This study integrates 3D Gaussian Splatting (3DGS) with the Segment Anything Model (SAM) for precise 3D reconstruction and biomass estimation of oilseed rape. UAV multi-view oblique images from 36 angles were used to perform 3D reconstruction, with the SAM module enhancing point cloud segmentation. The segmented point clouds were then converted into point cloud volumes, which were fitted to ground-measured biomass using linear regression. The results showed that 3DGS (7k and 30k iterations) provided high accuracy, with peak signal-to-noise ratios (PSNR) of 27.43 and 29.53 and training times of 7 and 49 minutes, respectively. This performance exceeded that of structure from motion (SfM) and mipmap Neural Radiance Fields (Mip-NeRF), demonstrating superior efficiency. The SAM module achieved high segmentation accuracy, with a mean intersection over union (mIoU) of 0.961 and an F1-score of 0.980. Additionally, a comparison of biomass extraction models found the point cloud volume model to be the most accurate, with an determination coefficient (R2) of 0.976, root mean square error (RMSE) of 2.92 g/plant, and mean absolute percentage error (MAPE) of 6.81%, outperforming both the plot crop volume and individual crop volume models. This study highlights the potential of combining 3DGS with multi-view UAV imaging for improved biomass phenotyping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08373v1">DG-SLAM: Robust Dynamic Gaussian Splatting SLAM with Hybrid Pose Optimization</a></div>
    <div class="paper-meta">
      📅 2024-11-13
    </div>
    <details class="paper-abstract">
      Achieving robust and precise pose estimation in dynamic scenes is a significant research challenge in Visual Simultaneous Localization and Mapping (SLAM). Recent advancements integrating Gaussian Splatting into SLAM systems have proven effective in creating high-quality renderings using explicit 3D Gaussian models, significantly improving environmental reconstruction fidelity. However, these approaches depend on a static environment assumption and face challenges in dynamic environments due to inconsistent observations of geometry and photometry. To address this problem, we propose DG-SLAM, the first robust dynamic visual SLAM system grounded in 3D Gaussians, which provides precise camera pose estimation alongside high-fidelity reconstructions. Specifically, we propose effective strategies, including motion mask generation, adaptive Gaussian point management, and a hybrid camera tracking algorithm to improve the accuracy and robustness of pose estimation. Extensive experiments demonstrate that DG-SLAM delivers state-of-the-art performance in camera pose estimation, map reconstruction, and novel-view synthesis in dynamic scenes, outperforming existing methods meanwhile preserving real-time rendering ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08279v1">MBA-SLAM: Motion Blur Aware Dense Visual SLAM with Radiance Fields Representation</a></div>
    <div class="paper-meta">
      📅 2024-11-13
    </div>
    <details class="paper-abstract">
      Emerging 3D scene representations, such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have demonstrated their effectiveness in Simultaneous Localization and Mapping (SLAM) for photo-realistic rendering, particularly when using high-quality video sequences as input. However, existing methods struggle with motion-blurred frames, which are common in real-world scenarios like low-light or long-exposure conditions. This often results in a significant reduction in both camera localization accuracy and map reconstruction quality. To address this challenge, we propose a dense visual SLAM pipeline (i.e. MBA-SLAM) to handle severe motion-blurred inputs. Our approach integrates an efficient motion blur-aware tracker with either neural radiance fields or Gaussian Splatting based mapper. By accurately modeling the physical image formation process of motion-blurred images, our method simultaneously learns 3D scene representation and estimates the cameras' local trajectory during exposure time, enabling proactive compensation for motion blur caused by camera movement. In our experiments, we demonstrate that MBA-SLAM surpasses previous state-of-the-art methods in both camera localization and map reconstruction, showcasing superior performance across a range of datasets, including synthetic and real datasets featuring sharp images as well as those affected by motion blur, highlighting the versatility and robustness of our approach. Code is available at https://github.com/WU-CVGL/MBA-SLAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.09733v3">Textured-GS: Gaussian Splatting with Spatially Defined Color and Opacity</a></div>
    <div class="paper-meta">
      📅 2024-11-13
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      In this paper, we introduce Textured-GS, an innovative method for rendering Gaussian splatting that incorporates spatially defined color and opacity variations using Spherical Harmonics (SH). This approach enables each Gaussian to exhibit a richer representation by accommodating varying colors and opacities across its surface, significantly enhancing rendering quality compared to traditional methods. To demonstrate the merits of our approach, we have adapted the Mini-Splatting architecture to integrate textured Gaussians without increasing the number of Gaussians. Our experiments across multiple real-world datasets show that Textured-GS consistently outperforms both the baseline Mini-Splatting and standard 3DGS in terms of visual fidelity. The results highlight the potential of Textured-GS to advance Gaussian-based rendering technologies, promising more efficient and high-quality scene reconstructions. Our implementation is available at https://github.com/ZhentaoHuang/Textured-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.17245v6">LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS</a></div>
    <div class="paper-meta">
      📅 2024-11-12
      | 💬 NeurIPS 2024, Project page: https://lightgaussian.github.io/
    </div>
    <details class="paper-abstract">
      Recent advances in real-time neural rendering using point-based techniques have enabled broader adoption of 3D representations. However, foundational approaches like 3D Gaussian Splatting impose substantial storage overhead, as Structure-from-Motion (SfM) points can grow to millions, often requiring gigabyte-level disk space for a single unbounded scene. This growth presents scalability challenges and hinders splatting efficiency. To address this, we introduce LightGaussian, a method for transforming 3D Gaussians into a more compact format. Inspired by Network Pruning, LightGaussian identifies Gaussians with minimal global significance on scene reconstruction, and applies a pruning and recovery process to reduce redundancy while preserving visual quality. Knowledge distillation and pseudo-view augmentation then transfer spherical harmonic coefficients to a lower degree, yielding compact representations. Gaussian Vector Quantization, based on each Gaussian's global significance, further lowers bitwidth with minimal accuracy loss. LightGaussian achieves an average 15x compression rate while boosting FPS from 144 to 237 within the 3D-GS framework, enabling efficient complex scene representation on the Mip-NeRF 360 and Tank & Temple datasets. The proposed Gaussian pruning approach is also adaptable to other 3D representations (e.g., Scaffold-GS), demonstrating strong generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06390v2">SplatFormer: Point Transformer for Robust 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-12
      | 💬 Code and dataset: https://github.com/ChenYutongTHU/SplatFormer Project page: https://sergeyprokudin.github.io/splatformer/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently transformed photorealistic reconstruction, achieving high visual fidelity and real-time performance. However, rendering quality significantly deteriorates when test views deviate from the camera angles used during training, posing a major challenge for applications in immersive free-viewpoint rendering and navigation. In this work, we conduct a comprehensive evaluation of 3DGS and related novel view synthesis methods under out-of-distribution (OOD) test camera scenarios. By creating diverse test cases with synthetic and real-world datasets, we demonstrate that most existing methods, including those incorporating various regularization techniques and data-driven priors, struggle to generalize effectively to OOD views. To address this limitation, we introduce SplatFormer, the first point transformer model specifically designed to operate on Gaussian splats. SplatFormer takes as input an initial 3DGS set optimized under limited training views and refines it in a single forward pass, effectively removing potential artifacts in OOD test views. To our knowledge, this is the first successful application of point transformers directly on 3DGS sets, surpassing the limitations of previous multi-scene training methods, which could handle only a restricted number of input views during inference. Our model significantly improves rendering quality under extreme novel views, achieving state-of-the-art performance in these challenging scenarios and outperforming various 3DGS regularization techniques, multi-scene models tailored for sparse view synthesis, and diffusion-based frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07555v1">GaussianCut: Interactive segmentation via graph cut for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-12
    </div>
    <details class="paper-abstract">
      We introduce GaussianCut, a new method for interactive multiview segmentation of scenes represented as 3D Gaussians. Our approach allows for selecting the objects to be segmented by interacting with a single view. It accepts intuitive user input, such as point clicks, coarse scribbles, or text. Using 3D Gaussian Splatting (3DGS) as the underlying scene representation simplifies the extraction of objects of interest which are considered to be a subset of the scene's Gaussians. Our key idea is to represent the scene as a graph and use the graph-cut algorithm to minimize an energy function to effectively partition the Gaussians into foreground and background. To achieve this, we construct a graph based on scene Gaussians and devise a segmentation-aligned energy function on the graph to combine user inputs with scene properties. To obtain an initial coarse segmentation, we leverage 2D image/video segmentation models and further refine these coarse estimates using our graph construction. Our empirical evaluations show the adaptability of GaussianCut across a diverse set of scenes. GaussianCut achieves competitive performance with state-of-the-art approaches for 3D segmentation without requiring any additional segmentation-aware training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09227v2">DreamScape: 3D Scene Creation via Gaussian Splatting joint Correlation Modeling</a></div>
    <div class="paper-meta">
      📅 2024-11-12
    </div>
    <details class="paper-abstract">
      Recent progress in text-to-3D creation has been propelled by integrating the potent prior of Diffusion Models from text-to-image generation into the 3D domain. Nevertheless, generating 3D scenes characterized by multiple instances and intricate arrangements remains challenging. In this study, we present DreamScape, a method for creating highly consistent 3D scenes solely from textual descriptions, leveraging the strong 3D representation capabilities of Gaussian Splatting and the complex arrangement abilities of large language models (LLMs). Our approach involves a 3D Gaussian Guide ($3{DG^2}$) for scene representation, consisting of semantic primitives (objects) and their spatial transformations and relationships derived directly from text prompts using LLMs. This compositional representation allows for local-to-global optimization of the entire scene. A progressive scale control is tailored during local object generation, ensuring that objects of different sizes and densities adapt to the scene, which addresses training instability issue arising from simple blending in the subsequent global optimization stage. To mitigate potential biases of LLM priors, we model collision relationships between objects at the global level, enhancing physical correctness and overall realism. Additionally, to generate pervasive objects like rain and snow distributed extensively across the scene, we introduce a sparse initialization and densification strategy. Experiments demonstrate that DreamScape offers high usability and controllability, enabling the generation of high-fidelity 3D scenes from only text prompts and achieving state-of-the-art performance compared to other methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07478v1">GUS-IR: Gaussian Splatting with Unified Shading for Inverse Rendering</a></div>
    <div class="paper-meta">
      📅 2024-11-12
      | 💬 15 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Recovering the intrinsic physical attributes of a scene from images, generally termed as the inverse rendering problem, has been a central and challenging task in computer vision and computer graphics. In this paper, we present GUS-IR, a novel framework designed to address the inverse rendering problem for complicated scenes featuring rough and glossy surfaces. This paper starts by analyzing and comparing two prominent shading techniques popularly used for inverse rendering, forward shading and deferred shading, effectiveness in handling complex materials. More importantly, we propose a unified shading solution that combines the advantages of both techniques for better decomposition. In addition, we analyze the normal modeling in 3D Gaussian Splatting (3DGS) and utilize the shortest axis as normal for each particle in GUS-IR, along with a depth-related regularization, resulting in improved geometric representation and better shape reconstruction. Furthermore, we enhance the probe-based baking scheme proposed by GS-IR to achieve more accurate ambient occlusion modeling to better handle indirect illumination. Extensive experiments have demonstrated the superior performance of GUS-IR in achieving precise intrinsic decomposition and geometric representation, supporting many downstream tasks (such as relighting, retouching) in computer vision, graphics, and extended reality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08129v2">Efficient Perspective-Correct 3D Gaussian Splatting Using Hybrid Transparency</a></div>
    <div class="paper-meta">
      📅 2024-11-11
      | 💬 Project page: https://fhahlbohm.github.io/htgs/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splats (3DGS) have proven a versatile rendering primitive, both for inverse rendering as well as real-time exploration of scenes. In these applications, coherence across camera frames and multiple views is crucial, be it for robust convergence of a scene reconstruction or for artifact-free fly-throughs. Recent work started mitigating artifacts that break multi-view coherence, including popping artifacts due to inconsistent transparency sorting and perspective-correct outlines of (2D) splats. At the same time, real-time requirements forced such implementations to accept compromises in how transparency of large assemblies of 3D Gaussians is resolved, in turn breaking coherence in other ways. In our work, we aim at achieving maximum coherence, by rendering fully perspective-correct 3D Gaussians while using a high-quality approximation of accurate blending, hybrid transparency, on a per-pixel level, in order to retain real-time frame rates. Our fast and perspectively accurate approach for evaluation of 3D Gaussians does not require matrix inversions, thereby ensuring numerical stability and eliminating the need for special handling of degenerate splats, and the hybrid transparency formulation for blending maintains similar quality as fully resolved per-pixel transparencies at a fraction of the rendering costs. We further show that each of these two components can be independently integrated into Gaussian splatting systems. In combination, they achieve up to 2$\times$ higher frame rates, 2$\times$ faster optimization, and equal or better image quality with fewer rendering artifacts compared to traditional 3DGS on common benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06976v1">A Hierarchical Compression Technique for 3D Gaussian Splatting Compression</a></div>
    <div class="paper-meta">
      📅 2024-11-11
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (GS) demonstrates excellent rendering quality and generation speed in novel view synthesis. However, substantial data size poses challenges for storage and transmission, making 3D GS compression an essential technology. Current 3D GS compression research primarily focuses on developing more compact scene representations, such as converting explicit 3D GS data into implicit forms. In contrast, compression of the GS data itself has hardly been explored. To address this gap, we propose a Hierarchical GS Compression (HGSC) technique. Initially, we prune unimportant Gaussians based on importance scores derived from both global and local significance, effectively reducing redundancy while maintaining visual quality. An Octree structure is used to compress 3D positions. Based on the 3D GS Octree, we implement a hierarchical attribute compression strategy by employing a KD-tree to partition the 3D GS into multiple blocks. We apply farthest point sampling to select anchor primitives within each block and others as non-anchor primitives with varying Levels of Details (LoDs). Anchor primitives serve as reference points for predicting non-anchor primitives across different LoDs to reduce spatial redundancy. For anchor primitives, we use the region adaptive hierarchical transform to achieve near-lossless compression of various attributes. For non-anchor primitives, each is predicted based on the k-nearest anchor primitives. To further minimize prediction errors, the reconstructed LoD and anchor primitives are combined to form new anchor primitives to predict the next LoD. Our method notably achieves superior compression quality and a significant data size reduction of over 4.5 times compared to the state-of-the-art compression method on small scenes datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06602v1">Adaptive and Temporally Consistent Gaussian Surfels for Multi-view Dynamic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-11-10
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has recently achieved notable success in novel view synthesis for dynamic scenes and geometry reconstruction in static scenes. Building on these advancements, early methods have been developed for dynamic surface reconstruction by globally optimizing entire sequences. However, reconstructing dynamic scenes with significant topology changes, emerging or disappearing objects, and rapid movements remains a substantial challenge, particularly for long sequences. To address these issues, we propose AT-GS, a novel method for reconstructing high-quality dynamic surfaces from multi-view videos through per-frame incremental optimization. To avoid local minima across frames, we introduce a unified and adaptive gradient-aware densification strategy that integrates the strengths of conventional cloning and splitting techniques. Additionally, we reduce temporal jittering in dynamic surfaces by ensuring consistency in curvature maps across consecutive frames. Our method achieves superior accuracy and temporal coherence in dynamic surface reconstruction, delivering high-fidelity space-time novel view synthesis, even in complex and challenging scenes. Extensive experiments on diverse multi-view video datasets demonstrate the effectiveness of our approach, showing clear advantages over baseline methods. Project page: \url{https://fraunhoferhhi.github.io/AT-GS}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06365v1">Through the Curved Cover: Synthesizing Cover Aberrated Scenes with Refractive Field</a></div>
    <div class="paper-meta">
      📅 2024-11-10
      | 💬 WACV 2025
    </div>
    <details class="paper-abstract">
      Recent extended reality headsets and field robots have adopted covers to protect the front-facing cameras from environmental hazards and falls. The surface irregularities on the cover can lead to optical aberrations like blurring and non-parametric distortions. Novel view synthesis methods like NeRF and 3D Gaussian Splatting are ill-equipped to synthesize from sequences with optical aberrations. To address this challenge, we introduce SynthCover to enable novel view synthesis through protective covers for downstream extended reality applications. SynthCover employs a Refractive Field that estimates the cover's geometry, enabling precise analytical calculation of refracted rays. Experiments on synthetic and real-world scenes demonstrate our method's ability to accurately model scenes viewed through protective covers, achieving a significant improvement in rendering quality compared to prior methods. We also show that the model can adjust well to various cover geometries with synthetic sequences captured with covers of different surface curvatures. To motivate further studies on this problem, we provide the benchmarked dataset containing real and synthetic walkable scenes captured with protective cover optical aberrations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12369v3">AtomGS: Atomizing Gaussian Splatting for High-Fidelity Radiance Field</a></div>
    <div class="paper-meta">
      📅 2024-11-09
      | 💬 BMVC 2024
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently advanced radiance field reconstruction by offering superior capabilities for novel view synthesis and real-time rendering speed. However, its strategy of blending optimization and adaptive density control might lead to sub-optimal results; it can sometimes yield noisy geometry and blurry artifacts due to prioritizing optimizing large Gaussians at the cost of adequately densifying smaller ones. To address this, we introduce AtomGS, consisting of Atomized Proliferation and Geometry-Guided Optimization. The Atomized Proliferation constrains ellipsoid Gaussians of various sizes into more uniform-sized Atom Gaussians. The strategy enhances the representation of areas with fine features by placing greater emphasis on densification in accordance with scene details. In addition, we proposed a Geometry-Guided Optimization approach that incorporates an Edge-Aware Normal Loss. This optimization method effectively smooths flat surfaces while preserving intricate details. Our evaluation shows that AtomGS outperforms existing state-of-the-art methods in rendering quality. Additionally, it achieves competitive accuracy in geometry reconstruction and offers a significant improvement in training speed over other SDF-based methods. More interactive demos can be found in our website (https://rongliu-leo.github.io/AtomGS/).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10041v2">Implicit Gaussian Splatting with Efficient Multi-Level Tri-Plane Representation</a></div>
    <div class="paper-meta">
      📅 2024-11-09
      | 💬 Please note that the authors discovered configuration errors in the comparisons within the experiment section, resulting in unreliable quantitative results. We advise referencing the results in this paper with caution
    </div>
    <details class="paper-abstract">
      Recent advancements in photo-realistic novel view synthesis have been significantly driven by Gaussian Splatting (3DGS). Nevertheless, the explicit nature of 3DGS data entails considerable storage requirements, highlighting a pressing need for more efficient data representations. To address this, we present Implicit Gaussian Splatting (IGS), an innovative hybrid model that integrates explicit point clouds with implicit feature embeddings through a multi-level tri-plane architecture. This architecture features 2D feature grids at various resolutions across different levels, facilitating continuous spatial domain representation and enhancing spatial correlations among Gaussian primitives. Building upon this foundation, we introduce a level-based progressive training scheme, which incorporates explicit spatial regularization. This method capitalizes on spatial correlations to enhance both the rendering quality and the compactness of the IGS representation. Furthermore, we propose a novel compression pipeline tailored for both point clouds and 2D feature grids, considering the entropy variations across different levels. Extensive experimental evaluations demonstrate that our algorithm can deliver high-quality rendering using only a few MBs, effectively balancing storage efficiency and rendering fidelity, and yielding results that are competitive with the state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06067v1">AI-Driven Stylization of 3D Environments</a></div>
    <div class="paper-meta">
      📅 2024-11-09
    </div>
    <details class="paper-abstract">
      In this system, we discuss methods to stylize a scene of 3D primitive objects into a higher fidelity 3D scene using novel 3D representations like NeRFs and 3D Gaussian Splatting. Our approach leverages existing image stylization systems and image-to-3D generative models to create a pipeline that iteratively stylizes and composites 3D objects into scenes. We show our results on adding generated objects into a scene and discuss limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06019v1">GaussianSpa: An "Optimizing-Sparsifying" Simplification Framework for Compact and High-Quality 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-09
      | 💬 Project page at https://gaussianspa.github.io/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a mainstream for novel view synthesis, leveraging continuous aggregations of Gaussian functions to model scene geometry. However, 3DGS suffers from substantial memory requirements to store the multitude of Gaussians, hindering its practicality. To address this challenge, we introduce GaussianSpa, an optimization-based simplification framework for compact and high-quality 3DGS. Specifically, we formulate the simplification as an optimization problem associated with the 3DGS training. Correspondingly, we propose an efficient "optimizing-sparsifying" solution that alternately solves two independent sub-problems, gradually imposing strong sparsity onto the Gaussians in the training process. Our comprehensive evaluations on various datasets show the superiority of GaussianSpa over existing state-of-the-art approaches. Notably, GaussianSpa achieves an average PSNR improvement of 0.9 dB on the real-world Deep Blending dataset with 10$\times$ fewer Gaussians compared to the vanilla 3DGS. Our project page is available at https://gaussianspa.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03807v3">GS2Pose: Two-stage 6D Object Pose Estimation Guided by Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-08
    </div>
    <details class="paper-abstract">
      This paper proposes a new method for accurate and robust 6D pose estimation of novel objects, named GS2Pose. By introducing 3D Gaussian splatting, GS2Pose can utilize the reconstruction results without requiring a high-quality CAD model, which means it only requires segmented RGBD images as input. Specifically, GS2Pose employs a two-stage structure consisting of coarse estimation followed by refined estimation. In the coarse stage, a lightweight U-Net network with a polarization attention mechanism, called Pose-Net, is designed. By using the 3DGS model for supervised training, Pose-Net can generate NOCS images to compute a coarse pose. In the refinement stage, GS2Pose formulates a pose regression algorithm following the idea of reprojection or Bundle Adjustment (BA), referred to as GS-Refiner. By leveraging Lie algebra to extend 3DGS, GS-Refiner obtains a pose-differentiable rendering pipeline that refines the coarse pose by comparing the input images with the rendered images. GS-Refiner also selectively updates parameters in the 3DGS model to achieve environmental adaptation, thereby enhancing the algorithm's robustness and flexibility to illuminative variation, occlusion, and other challenging disruptive factors. GS2Pose was evaluated through experiments conducted on the LineMod dataset, where it was compared with similar algorithms, yielding highly competitive results. The code for GS2Pose will soon be released on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2403.17822v3">DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing</a></div>
    <div class="paper-meta">
      📅 2024-11-08
      | 💬 To be published in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)
    </div>
    <details class="paper-abstract">
      High-fidelity 3D reconstruction of common indoor scenes is crucial for VR and AR applications. 3D Gaussian splatting, a novel differentiable rendering technique, has achieved state-of-the-art novel view synthesis results with high rendering speeds and relatively low training times. However, its performance on scenes commonly seen in indoor datasets is poor due to the lack of geometric constraints during optimization. In this work, we explore the use of readily accessible geometric cues to enhance Gaussian splatting optimization in challenging, ill-posed, and textureless scenes. We extend 3D Gaussian splatting with depth and normal cues to tackle challenging indoor datasets and showcase techniques for efficient mesh extraction. Specifically, we regularize the optimization procedure with depth information, enforce local smoothness of nearby Gaussians, and use off-the-shelf monocular networks to achieve better alignment with the true scene geometry. We propose an adaptive depth loss based on the gradient of color images, improving depth estimation and novel view synthesis results over various baselines. Our simple yet effective regularization technique enables direct mesh extraction from the Gaussian representation, yielding more physically accurate reconstructions of indoor scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05006v1">ProEdit: Simple Progression is All You Need for High-Quality 3D Scene Editing</a></div>
    <div class="paper-meta">
      📅 2024-11-07
      | 💬 NeurIPS 2024. Project Page: https://immortalco.github.io/ProEdit/
    </div>
    <details class="paper-abstract">
      This paper proposes ProEdit - a simple yet effective framework for high-quality 3D scene editing guided by diffusion distillation in a novel progressive manner. Inspired by the crucial observation that multi-view inconsistency in scene editing is rooted in the diffusion model's large feasible output space (FOS), our framework controls the size of FOS and reduces inconsistency by decomposing the overall editing task into several subtasks, which are then executed progressively on the scene. Within this framework, we design a difficulty-aware subtask decomposition scheduler and an adaptive 3D Gaussian splatting (3DGS) training strategy, ensuring high quality and efficiency in performing each subtask. Extensive evaluation shows that our ProEdit achieves state-of-the-art results in various scenes and challenging editing tasks, all through a simple framework without any expensive or sophisticated add-ons like distillation losses, components, or training procedures. Notably, ProEdit also provides a new way to control, preview, and select the "aggressivity" of editing operation during the editing process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04924v1">MVSplat360: Feed-Forward 360 Scene Synthesis from Sparse Views</a></div>
    <div class="paper-meta">
      📅 2024-11-07
      | 💬 NeurIPS 2024, Project page: https://donydchen.github.io/mvsplat360, Code: https://github.com/donydchen/mvsplat360
    </div>
    <details class="paper-abstract">
      We introduce MVSplat360, a feed-forward approach for 360{\deg} novel view synthesis (NVS) of diverse real-world scenes, using only sparse observations. This setting is inherently ill-posed due to minimal overlap among input views and insufficient visual information provided, making it challenging for conventional methods to achieve high-quality results. Our MVSplat360 addresses this by effectively combining geometry-aware 3D reconstruction with temporally consistent video generation. Specifically, it refactors a feed-forward 3D Gaussian Splatting (3DGS) model to render features directly into the latent space of a pre-trained Stable Video Diffusion (SVD) model, where these features then act as pose and visual cues to guide the denoising process and produce photorealistic 3D-consistent views. Our model is end-to-end trainable and supports rendering arbitrary views with as few as 5 sparse input views. To evaluate MVSplat360's performance, we introduce a new benchmark using the challenging DL3DV-10K dataset, where MVSplat360 achieves superior visual quality compared to state-of-the-art methods on wide-sweeping or even 360{\deg} NVS tasks. Experiments on the existing benchmark RealEstate10K also confirm the effectiveness of our model. The video results are available on our project page: https://donydchen.github.io/mvsplat360.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.17822v3">DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing</a></div>
    <div class="paper-meta">
      📅 2024-11-07
      | 💬 To be published in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)
    </div>
    <details class="paper-abstract">
      High-fidelity 3D reconstruction of common indoor scenes is crucial for VR and AR applications. 3D Gaussian splatting, a novel differentiable rendering technique, has achieved state-of-the-art novel view synthesis results with high rendering speeds and relatively low training times. However, its performance on scenes commonly seen in indoor datasets is poor due to the lack of geometric constraints during optimization. In this work, we explore the use of readily accessible geometric cues to enhance Gaussian splatting optimization in challenging, ill-posed, and textureless scenes. We extend 3D Gaussian splatting with depth and normal cues to tackle challenging indoor datasets and showcase techniques for efficient mesh extraction. Specifically, we regularize the optimization procedure with depth information, enforce local smoothness of nearby Gaussians, and use off-the-shelf monocular networks to achieve better alignment with the true scene geometry. We propose an adaptive depth loss based on the gradient of color images, improving depth estimation and novel view synthesis results over various baselines. Our simple yet effective regularization technique enables direct mesh extraction from the Gaussian representation, yielding more physically accurate reconstructions of indoor scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2404.03202v5">OmniGS: Fast Radiance Field Reconstruction using Omnidirectional Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-07
      | 💬 8 pages, 6 figures, accepted by WACV 2025, project page: https://liquorleaf.github.io/research/OmniGS/
    </div>
    <details class="paper-abstract">
      Photorealistic reconstruction relying on 3D Gaussian Splatting has shown promising potential in various domains. However, the current 3D Gaussian Splatting system only supports radiance field reconstruction using undistorted perspective images. In this paper, we present OmniGS, a novel omnidirectional Gaussian splatting system, to take advantage of omnidirectional images for fast radiance field reconstruction. Specifically, we conduct a theoretical analysis of spherical camera model derivatives in 3D Gaussian Splatting. According to the derivatives, we then implement a new GPU-accelerated omnidirectional rasterizer that directly splats 3D Gaussians onto the equirectangular screen space for omnidirectional image rendering. We realize differentiable optimization of the omnidirectional radiance field without the requirement of cube-map rectification or tangent-plane approximation. Extensive experiments conducted in egocentric and roaming scenarios demonstrate that our method achieves state-of-the-art reconstruction quality and high rendering speed using omnidirectional images. The code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2411.02229v2">FewViewGS: Gaussian Splatting with Few View Matching and Multi-stage Training</a></div>
    <div class="paper-meta">
      📅 2024-11-07
      | 💬 Accepted by NeurIPS2024
    </div>
    <details class="paper-abstract">
      The field of novel view synthesis from images has seen rapid advancements with the introduction of Neural Radiance Fields (NeRF) and more recently with 3D Gaussian Splatting. Gaussian Splatting became widely adopted due to its efficiency and ability to render novel views accurately. While Gaussian Splatting performs well when a sufficient amount of training images are available, its unstructured explicit representation tends to overfit in scenarios with sparse input images, resulting in poor rendering performance. To address this, we present a 3D Gaussian-based novel view synthesis method using sparse input images that can accurately render the scene from the viewpoints not covered by the training images. We propose a multi-stage training scheme with matching-based consistency constraints imposed on the novel views without relying on pre-trained depth estimation or diffusion models. This is achieved by using the matches of the available training images to supervise the generation of the novel views sampled between the training frames with color, geometry, and semantic losses. In addition, we introduce a locality preserving regularization for 3D Gaussians which removes rendering artifacts by preserving the local color structure of the scene. Evaluation on synthetic and real-world datasets demonstrates competitive or superior performance of our method in few-shot novel view synthesis compared to existing state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2403.17822v3">DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing</a></div>
    <div class="paper-meta">
      📅 2024-11-07
      | 💬 To be published in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)
    </div>
    <details class="paper-abstract">
      High-fidelity 3D reconstruction of common indoor scenes is crucial for VR and AR applications. 3D Gaussian splatting, a novel differentiable rendering technique, has achieved state-of-the-art novel view synthesis results with high rendering speeds and relatively low training times. However, its performance on scenes commonly seen in indoor datasets is poor due to the lack of geometric constraints during optimization. In this work, we explore the use of readily accessible geometric cues to enhance Gaussian splatting optimization in challenging, ill-posed, and textureless scenes. We extend 3D Gaussian splatting with depth and normal cues to tackle challenging indoor datasets and showcase techniques for efficient mesh extraction. Specifically, we regularize the optimization procedure with depth information, enforce local smoothness of nearby Gaussians, and use off-the-shelf monocular networks to achieve better alignment with the true scene geometry. We propose an adaptive depth loss based on the gradient of color images, improving depth estimation and novel view synthesis results over various baselines. Our simple yet effective regularization technique enables direct mesh extraction from the Gaussian representation, yielding more physically accurate reconstructions of indoor scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02332v2">SplatOverflow: Asynchronous Hardware Troubleshooting</a></div>
    <div class="paper-meta">
      📅 2024-11-06
      | 💬 Our accompanying video figure is available at: https://youtu.be/m4TKeBDuZkU
    </div>
    <details class="paper-abstract">
      As tools for designing and manufacturing hardware become more accessible, smaller producers can develop and distribute novel hardware. However, there aren't established tools to support end-user hardware troubleshooting or routine maintenance. As a result, technical support for hardware remains ad-hoc and challenging to scale. Inspired by software troubleshooting workflows like StackOverflow, we propose a workflow for asynchronous hardware troubleshooting: SplatOverflow. SplatOverflow creates a novel boundary object, the SplatOverflow scene, that users reference to communicate about hardware. The scene comprises a 3D Gaussian Splat of the user's hardware registered onto the hardware's CAD model. The splat captures the current state of the hardware, and the registered CAD model acts as a referential anchor for troubleshooting instructions. With SplatOverflow, maintainers can directly address issues and author instructions in the user's workspace. The instructions define workflows that can easily be shared between users and recontextualized in new environments. In this paper, we describe the design of SplatOverflow, detail the workflows it enables, and illustrate its utility to different kinds of users. We also validate that non-experts can use SplatOverflow to troubleshoot common problems with a 3D printer in a user study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16147v3">Gaussian Deja-vu: Creating Controllable 3D Gaussian Head-Avatars with Enhanced Generalization and Personalization Abilities</a></div>
    <div class="paper-meta">
      📅 2024-11-06
      | 💬 11 pages, Accepted by WACV 2025 in Round 1
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3DGS) have unlocked significant potential for modeling 3D head avatars, providing greater flexibility than mesh-based methods and more efficient rendering compared to NeRF-based approaches. Despite these advancements, the creation of controllable 3DGS-based head avatars remains time-intensive, often requiring tens of minutes to hours. To expedite this process, we here introduce the "Gaussian Deja-vu" framework, which first obtains a generalized model of the head avatar and then personalizes the result. The generalized model is trained on large 2D (synthetic and real) image datasets. This model provides a well-initialized 3D Gaussian head that is further refined using a monocular video to achieve the personalized head avatar. For personalizing, we propose learnable expression-aware rectification blendmaps to correct the initial 3D Gaussians, ensuring rapid convergence without the reliance on neural networks. Experiments demonstrate that the proposed method meets its objectives. It outperforms state-of-the-art 3D Gaussian head avatars in terms of photorealistic quality as well as reduces training time consumption to at least a quarter of the existing methods, producing the avatar in minutes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.03202v5">OmniGS: Fast Radiance Field Reconstruction using Omnidirectional Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-06
      | 💬 8 pages, 6 figures, accepted by WACV 2025, project page: https://liquorleaf.github.io/research/OmniGS/
    </div>
    <details class="paper-abstract">
      Photorealistic reconstruction relying on 3D Gaussian Splatting has shown promising potential in various domains. However, the current 3D Gaussian Splatting system only supports radiance field reconstruction using undistorted perspective images. In this paper, we present OmniGS, a novel omnidirectional Gaussian splatting system, to take advantage of omnidirectional images for fast radiance field reconstruction. Specifically, we conduct a theoretical analysis of spherical camera model derivatives in 3D Gaussian Splatting. According to the derivatives, we then implement a new GPU-accelerated omnidirectional rasterizer that directly splats 3D Gaussians onto the equirectangular screen space for omnidirectional image rendering. We realize differentiable optimization of the omnidirectional radiance field without the requirement of cube-map rectification or tangent-plane approximation. Extensive experiments conducted in egocentric and roaming scenarios demonstrate that our method achieves state-of-the-art reconstruction quality and high rendering speed using omnidirectional images. The code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03637v1">Structure Consistent Gaussian Splatting with Matching Prior for Few-shot Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-11-06
      | 💬 NeurIPS 2024 Accepted
    </div>
    <details class="paper-abstract">
      Despite the substantial progress of novel view synthesis, existing methods, either based on the Neural Radiance Fields (NeRF) or more recently 3D Gaussian Splatting (3DGS), suffer significant degradation when the input becomes sparse. Numerous efforts have been introduced to alleviate this problem, but they still struggle to synthesize satisfactory results efficiently, especially in the large scene. In this paper, we propose SCGaussian, a Structure Consistent Gaussian Splatting method using matching priors to learn 3D consistent scene structure. Considering the high interdependence of Gaussian attributes, we optimize the scene structure in two folds: rendering geometry and, more importantly, the position of Gaussian primitives, which is hard to be directly constrained in the vanilla 3DGS due to the non-structure property. To achieve this, we present a hybrid Gaussian representation. Besides the ordinary non-structure Gaussian primitives, our model also consists of ray-based Gaussian primitives that are bound to matching rays and whose optimization of their positions is restricted along the ray. Thus, we can utilize the matching correspondence to directly enforce the position of these Gaussian primitives to converge to the surface points where rays intersect. Extensive experiments on forward-facing, surrounding, and complex large scenes show the effectiveness of our approach with state-of-the-art performance and high efficiency. Code is available at https://github.com/prstrive/SCGaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2404.03202v5">OmniGS: Fast Radiance Field Reconstruction using Omnidirectional Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-06
      | 💬 8 pages, 6 figures, accepted by WACV 2025, project page: https://liquorleaf.github.io/research/OmniGS/
    </div>
    <details class="paper-abstract">
      Photorealistic reconstruction relying on 3D Gaussian Splatting has shown promising potential in various domains. However, the current 3D Gaussian Splatting system only supports radiance field reconstruction using undistorted perspective images. In this paper, we present OmniGS, a novel omnidirectional Gaussian splatting system, to take advantage of omnidirectional images for fast radiance field reconstruction. Specifically, we conduct a theoretical analysis of spherical camera model derivatives in 3D Gaussian Splatting. According to the derivatives, we then implement a new GPU-accelerated omnidirectional rasterizer that directly splats 3D Gaussians onto the equirectangular screen space for omnidirectional image rendering. We realize differentiable optimization of the omnidirectional radiance field without the requirement of cube-map rectification or tangent-plane approximation. Extensive experiments conducted in egocentric and roaming scenarios demonstrate that our method achieves state-of-the-art reconstruction quality and high rendering speed using omnidirectional images. The code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03555v1">Object and Contact Point Tracking in Demonstrations Using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-05
      | 💬 CoRL 2024, Workshop on Lifelong Learning for Home Robots, Munich, Germany
    </div>
    <details class="paper-abstract">
      This paper introduces a method to enhance Interactive Imitation Learning (IIL) by extracting touch interaction points and tracking object movement from video demonstrations. The approach extends current IIL systems by providing robots with detailed knowledge of both where and how to interact with objects, particularly complex articulated ones like doors and drawers. By leveraging cutting-edge techniques such as 3D Gaussian Splatting and FoundationPose for tracking, this method allows robots to better understand and manipulate objects in dynamic environments. The research lays the foundation for more effective task learning and execution in autonomous robotic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02229v2">FewViewGS: Gaussian Splatting with Few View Matching and Multi-stage Training</a></div>
    <div class="paper-meta">
      📅 2024-11-05
      | 💬 Accepted by NeurIPS2024
    </div>
    <details class="paper-abstract">
      The field of novel view synthesis from images has seen rapid advancements with the introduction of Neural Radiance Fields (NeRF) and more recently with 3D Gaussian Splatting. Gaussian Splatting became widely adopted due to its efficiency and ability to render novel views accurately. While Gaussian Splatting performs well when a sufficient amount of training images are available, its unstructured explicit representation tends to overfit in scenarios with sparse input images, resulting in poor rendering performance. To address this, we present a 3D Gaussian-based novel view synthesis method using sparse input images that can accurately render the scene from the viewpoints not covered by the training images. We propose a multi-stage training scheme with matching-based consistency constraints imposed on the novel views without relying on pre-trained depth estimation or diffusion models. This is achieved by using the matches of the available training images to supervise the generation of the novel views sampled between the training frames with color, geometry, and semantic losses. In addition, we introduce a locality preserving regularization for 3D Gaussians which removes rendering artifacts by preserving the local color structure of the scene. Evaluation on synthetic and real-world datasets demonstrates competitive or superior performance of our method in few-shot novel view synthesis compared to existing state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17705v3">DC-Gaussian: Improving 3D Gaussian Splatting for Reflective Dash Cam Videos</a></div>
    <div class="paper-meta">
      📅 2024-11-05
      | 💬 10 pages,7 figures;project page: https://linhanwang.github.io/dcgaussian/; Accepted to NeurIPS 2024
    </div>
    <details class="paper-abstract">
      We present DC-Gaussian, a new method for generating novel views from in-vehicle dash cam videos. While neural rendering techniques have made significant strides in driving scenarios, existing methods are primarily designed for videos collected by autonomous vehicles. However, these videos are limited in both quantity and diversity compared to dash cam videos, which are more widely used across various types of vehicles and capture a broader range of scenarios. Dash cam videos often suffer from severe obstructions such as reflections and occlusions on the windshields, which significantly impede the application of neural rendering techniques. To address this challenge, we develop DC-Gaussian based on the recent real-time neural rendering technique 3D Gaussian Splatting (3DGS). Our approach includes an adaptive image decomposition module to model reflections and occlusions in a unified manner. Additionally, we introduce illumination-aware obstruction modeling to manage reflections and occlusions under varying lighting conditions. Lastly, we employ a geometry-guided Gaussian enhancement strategy to improve rendering details by incorporating additional geometry priors. Experiments on self-captured and public dash cam videos show that our method not only achieves state-of-the-art performance in novel view synthesis, but also accurately reconstructing captured scenes getting rid of obstructions. See the project page for code, data: https://linhanwang.github.io/dcgaussian/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03086v1">HFGaussian: Learning Generalizable Gaussian Human with Integrated Human Features</a></div>
    <div class="paper-meta">
      📅 2024-11-05
    </div>
    <details class="paper-abstract">
      Recent advancements in radiance field rendering show promising results in 3D scene representation, where Gaussian splatting-based techniques emerge as state-of-the-art due to their quality and efficiency. Gaussian splatting is widely used for various applications, including 3D human representation. However, previous 3D Gaussian splatting methods either use parametric body models as additional information or fail to provide any underlying structure, like human biomechanical features, which are essential for different applications. In this paper, we present a novel approach called HFGaussian that can estimate novel views and human features, such as the 3D skeleton, 3D key points, and dense pose, from sparse input images in real time at 25 FPS. The proposed method leverages generalizable Gaussian splatting technique to represent the human subject and its associated features, enabling efficient and generalizable reconstruction. By incorporating a pose regression network and the feature splatting technique with Gaussian splatting, HFGaussian demonstrates improved capabilities over existing 3D human methods, showcasing the potential of 3D human representations with integrated biomechanics. We thoroughly evaluate our HFGaussian method against the latest state-of-the-art techniques in human Gaussian splatting and pose estimation, demonstrating its real-time, state-of-the-art performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09510v4">3DGS.zip: A survey on 3D Gaussian Splatting Compression Methods</a></div>
    <div class="paper-meta">
      📅 2024-11-05
      | 💬 3D Gaussian Splatting compression survey; 3DGS compression; new approaches added
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a cutting-edge technique for real-time radiance field rendering, offering state-of-the-art performance in terms of both quality and speed. 3DGS models a scene as a collection of three-dimensional Gaussians, or splats, with additional attributes optimized to conform to the scene's geometric and visual properties. Despite its advantages in rendering speed and image fidelity, 3DGS is limited by its significant storage and memory demands. These high demands make 3DGS impractical for mobile devices or headsets, reducing its applicability in important areas of computer graphics. To address these challenges and advance the practicality of 3DGS, this survey provides a comprehensive and detailed examination of compression and compaction techniques developed to make 3DGS more efficient. We categorize current approaches into compression techniques, which aim at achieving the highest quality at minimal data size, and compaction techniques, which aim for optimal quality with the fewest Gaussians. We introduce the basic mathematical concepts underlying the analyzed methods, as well as key implementation details and design choices. Our report thoroughly discusses similarities and differences among the methods, as well as their respective advantages and disadvantages. We establish a consistent standard for comparing these methods based on key performance metrics and datasets. Specifically, since these methods have been developed in parallel and over a short period of time, currently, no comprehensive comparison exists. This survey, for the first time, presents a unified standard to evaluate 3DGS compression techniques. To facilitate the continuous monitoring of emerging methodologies, we maintain a dedicated website that will be regularly updated with new techniques and revisions of existing findings https://w-m.github.io/3dgs-compression-survey/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02703v1">LVI-GS: Tightly-coupled LiDAR-Visual-Inertial SLAM using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-05
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown its ability in rapid rendering and high-fidelity mapping. In this paper, we introduce LVI-GS, a tightly-coupled LiDAR-Visual-Inertial mapping framework with 3DGS, which leverages the complementary characteristics of LiDAR and image sensors to capture both geometric structures and visual details of 3D scenes. To this end, the 3D Gaussians are initialized from colourized LiDAR points and optimized using differentiable rendering. In order to achieve high-fidelity mapping, we introduce a pyramid-based training approach to effectively learn multi-level features and incorporate depth loss derived from LiDAR measurements to improve geometric feature perception. Through well-designed strategies for Gaussian-Map expansion, keyframe selection, thread management, and custom CUDA acceleration, our framework achieves real-time photo-realistic mapping. Numerical experiments are performed to evaluate the superior performance of our method compared to state-of-the-art 3D reconstruction systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2403.18784v3">SplatFace: Gaussian Splat Face Reconstruction Leveraging an Optimizable Surface</a></div>
    <div class="paper-meta">
      📅 2024-11-05
    </div>
    <details class="paper-abstract">
      We present SplatFace, a novel Gaussian splatting framework designed for 3D human face reconstruction without reliance on accurate pre-determined geometry. Our method is designed to simultaneously deliver both high-quality novel view rendering and accurate 3D mesh reconstructions. We incorporate a generic 3D Morphable Model (3DMM) to provide a surface geometric structure, making it possible to reconstruct faces with a limited set of input images. We introduce a joint optimization strategy that refines both the Gaussians and the morphable surface through a synergistic non-rigid alignment process. A novel distance metric, splat-to-surface, is proposed to improve alignment by considering both the Gaussian position and covariance. The surface information is also utilized to incorporate a world-space densification process, resulting in superior reconstruction quality. Our experimental analysis demonstrates that the proposed method is competitive with both other Gaussian splatting techniques in novel view synthesis and other 3D reconstruction methods in producing 3D face meshes with high geometric precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.02229v2">FewViewGS: Gaussian Splatting with Few View Matching and Multi-stage Training</a></div>
    <div class="paper-meta">
      📅 2024-11-05
      | 💬 Accepted by NeurIPS2024
    </div>
    <details class="paper-abstract">
      The field of novel view synthesis from images has seen rapid advancements with the introduction of Neural Radiance Fields (NeRF) and more recently with 3D Gaussian Splatting. Gaussian Splatting became widely adopted due to its efficiency and ability to render novel views accurately. While Gaussian Splatting performs well when a sufficient amount of training images are available, its unstructured explicit representation tends to overfit in scenarios with sparse input images, resulting in poor rendering performance. To address this, we present a 3D Gaussian-based novel view synthesis method using sparse input images that can accurately render the scene from the viewpoints not covered by the training images. We propose a multi-stage training scheme with matching-based consistency constraints imposed on the novel views without relying on pre-trained depth estimation or diffusion models. This is achieved by using the matches of the available training images to supervise the generation of the novel views sampled between the training frames with color, geometry, and semantic losses. In addition, we introduce a locality preserving regularization for 3D Gaussians which removes rendering artifacts by preserving the local color structure of the scene. Evaluation on synthetic and real-world datasets demonstrates competitive or superior performance of our method in few-shot novel view synthesis compared to existing state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02547v1">Modeling Uncertainty in 3D Gaussian Splatting through Continuous Semantic Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-04
    </div>
    <details class="paper-abstract">
      In this paper, we present a novel algorithm for probabilistically updating and rasterizing semantic maps within 3D Gaussian Splatting (3D-GS). Although previous methods have introduced algorithms which learn to rasterize features in 3D-GS for enhanced scene understanding, 3D-GS can fail without warning which presents a challenge for safety-critical robotic applications. To address this gap, we propose a method which advances the literature of continuous semantic mapping from voxels to ellipsoids, combining the precise structure of 3D-GS with the ability to quantify uncertainty of probabilistic robotic maps. Given a set of images, our algorithm performs a probabilistic semantic update directly on the 3D ellipsoids to obtain an expectation and variance through the use of conjugate priors. We also propose a probabilistic rasterization which returns per-pixel segmentation predictions with quantifiable uncertainty. We compare our method with similar probabilistic voxel-based methods to verify our extension to 3D ellipsoids, and perform ablation studies on uncertainty quantification and temporal smoothing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.14370v4">SyncTweedies: A General Generative Framework Based on Synchronized Diffusions</a></div>
    <div class="paper-meta">
      📅 2024-11-04
      | 💬 Project page: https://synctweedies.github.io/ (NeurIPS 2024)
    </div>
    <details class="paper-abstract">
      We introduce a general framework for generating diverse visual content, including ambiguous images, panorama images, mesh textures, and Gaussian splat textures, by synchronizing multiple diffusion processes. We present exhaustive investigation into all possible scenarios for synchronizing multiple diffusion processes through a canonical space and analyze their characteristics across applications. In doing so, we reveal a previously unexplored case: averaging the outputs of Tweedie's formula while conducting denoising in multiple instance spaces. This case also provides the best quality with the widest applicability to downstream tasks. We name this case SyncTweedies. In our experiments generating visual content aforementioned, we demonstrate the superior quality of generation by SyncTweedies compared to other synchronization methods, optimization-based and iterative-update-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06543v3">HDRGS: High Dynamic Range Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-03
    </div>
    <details class="paper-abstract">
      Recent years have witnessed substantial advancements in the field of 3D reconstruction from 2D images, particularly following the introduction of the neural radiance field (NeRF) technique. However, reconstructing a 3D high dynamic range (HDR) radiance field, which aligns more closely with real-world conditions, from 2D multi-exposure low dynamic range (LDR) images continues to pose significant challenges. Approaches to this issue fall into two categories: grid-based and implicit-based. Implicit methods, using multi-layer perceptrons (MLP), face inefficiencies, limited solvability, and overfitting risks. Conversely, grid-based methods require significant memory and struggle with image quality and long training times. In this paper, we introduce Gaussian Splatting-a recent, high-quality, real-time 3D reconstruction technique-into this domain. We further develop the High Dynamic Range Gaussian Splatting (HDR-GS) method, designed to address the aforementioned challenges. This method enhances color dimensionality by including luminance and uses an asymmetric grid for tone-mapping, swiftly and precisely converting pixel irradiance to color. Our approach improves HDR scene recovery accuracy and integrates a novel coarse-to-fine strategy to speed up model convergence, enhancing robustness against sparse viewpoints and exposure extremes, and preventing local optima. Extensive testing confirms that our method surpasses current state-of-the-art techniques in both synthetic and real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.18784v3">SplatFace: Gaussian Splat Face Reconstruction Leveraging an Optimizable Surface</a></div>
    <div class="paper-meta">
      📅 2024-11-02
    </div>
    <details class="paper-abstract">
      We present SplatFace, a novel Gaussian splatting framework designed for 3D human face reconstruction without reliance on accurate pre-determined geometry. Our method is designed to simultaneously deliver both high-quality novel view rendering and accurate 3D mesh reconstructions. We incorporate a generic 3D Morphable Model (3DMM) to provide a surface geometric structure, making it possible to reconstruct faces with a limited set of input images. We introduce a joint optimization strategy that refines both the Gaussians and the morphable surface through a synergistic non-rigid alignment process. A novel distance metric, splat-to-surface, is proposed to improve alignment by considering both the Gaussian position and covariance. The surface information is also utilized to incorporate a world-space densification process, resulting in superior reconstruction quality. Our experimental analysis demonstrates that the proposed method is competitive with both other Gaussian splatting techniques in novel view synthesis and other 3D reconstruction methods in producing 3D face meshes with high geometric precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01218v1">Real-Time Spatio-Temporal Reconstruction of Dynamic Endoscopic Scenes with 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-02
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction is essential in robotic minimally invasive surgery, providing crucial spatial information that enhances surgical precision and outcomes. However, existing methods struggle to address the complex, temporally dynamic nature of endoscopic scenes. This paper presents ST-Endo4DGS, a novel framework that models the spatio-temporal volume of dynamic endoscopic scenes using unbiased 4D Gaussian Splatting (4DGS) primitives, parameterized by anisotropic ellipses with flexible 4D rotations. This approach enables precise representation of deformable tissue dynamics, capturing intricate spatial and temporal correlations in real time. Additionally, we extend spherindrical harmonics to represent time-evolving appearance, achieving realistic adaptations to lighting and view changes. A new endoscopic normal alignment constraint (ENAC) further enhances geometric fidelity by aligning rendered normals with depth-derived geometry. Extensive evaluations show that ST-Endo4DGS outperforms existing methods in both visual quality and real-time performance, establishing a new state-of-the-art in dynamic scene reconstruction for endoscopic surgery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2403.18784v3">SplatFace: Gaussian Splat Face Reconstruction Leveraging an Optimizable Surface</a></div>
    <div class="paper-meta">
      📅 2024-11-02
    </div>
    <details class="paper-abstract">
      We present SplatFace, a novel Gaussian splatting framework designed for 3D human face reconstruction without reliance on accurate pre-determined geometry. Our method is designed to simultaneously deliver both high-quality novel view rendering and accurate 3D mesh reconstructions. We incorporate a generic 3D Morphable Model (3DMM) to provide a surface geometric structure, making it possible to reconstruct faces with a limited set of input images. We introduce a joint optimization strategy that refines both the Gaussians and the morphable surface through a synergistic non-rigid alignment process. A novel distance metric, splat-to-surface, is proposed to improve alignment by considering both the Gaussian position and covariance. The surface information is also utilized to incorporate a world-space densification process, resulting in superior reconstruction quality. Our experimental analysis demonstrates that the proposed method is competitive with both other Gaussian splatting techniques in novel view synthesis and other 3D reconstruction methods in producing 3D face meshes with high geometric precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00771v1">CityGaussianV2: Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes</a></div>
    <div class="paper-meta">
      📅 2024-11-01
      | 💬 Project Page: https://dekuliutesla.github.io/CityGaussianV2/
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has revolutionized radiance field reconstruction, manifesting efficient and high-fidelity novel view synthesis. However, accurately representing surfaces, especially in large and complex scenarios, remains a significant challenge due to the unstructured nature of 3DGS. In this paper, we present CityGaussianV2, a novel approach for large-scale scene reconstruction that addresses critical challenges related to geometric accuracy and efficiency. Building on the favorable generalization capabilities of 2D Gaussian Splatting (2DGS), we address its convergence and scalability issues. Specifically, we implement a decomposed-gradient-based densification and depth regression technique to eliminate blurry artifacts and accelerate convergence. To scale up, we introduce an elongation filter that mitigates Gaussian count explosion caused by 2DGS degeneration. Furthermore, we optimize the CityGaussian pipeline for parallel training, achieving up to 10$\times$ compression, at least 25% savings in training time, and a 50% decrease in memory usage. We also established standard geometry benchmarks under large-scale scenes. Experimental results demonstrate that our method strikes a promising balance between visual quality, geometric accuracy, as well as storage and training costs. The project page is available at https://dekuliutesla.github.io/CityGaussianV2/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.24204v2">GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering</a></div>
    <div class="paper-meta">
      📅 2024-11-01
      | 💬 Project page: https://pku-vcl-geometry.github.io/GeoSplatting/
    </div>
    <details class="paper-abstract">
      We consider the problem of physically-based inverse rendering using 3D Gaussian Splatting (3DGS) representations. While recent 3DGS methods have achieved remarkable results in novel view synthesis (NVS), accurately capturing high-fidelity geometry, physically interpretable materials and lighting remains challenging, as it requires precise geometry modeling to provide accurate surface normals, along with physically-based rendering (PBR) techniques to ensure correct material and lighting disentanglement. Previous 3DGS methods resort to approximating surface normals, but often struggle with noisy local geometry, leading to inaccurate normal estimation and suboptimal material-lighting decomposition. In this paper, we introduce GeoSplatting, a novel hybrid representation that augments 3DGS with explicit geometric guidance and differentiable PBR equations. Specifically, we bridge isosurface and 3DGS together, where we first extract isosurface mesh from a scalar field, then convert it into 3DGS points and formulate PBR equations for them in a fully differentiable manner. In GeoSplatting, 3DGS is grounded on the mesh geometry, enabling precise surface normal modeling, which facilitates the use of PBR frameworks for material decomposition. This approach further maintains the efficiency and quality of NVS from 3DGS while ensuring accurate geometry from the isosurface. Comprehensive evaluations across diverse datasets demonstrate the superiority of GeoSplatting, consistently outperforming existing methods both quantitatively and qualitatively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00632v1">PCoTTA: Continual Test-Time Adaptation for Multi-Task Point Cloud Understanding</a></div>
    <div class="paper-meta">
      📅 2024-11-01
      | 💬 Accepted to NeurIPS 2024
    </div>
    <details class="paper-abstract">
      In this paper, we present PCoTTA, an innovative, pioneering framework for Continual Test-Time Adaptation (CoTTA) in multi-task point cloud understanding, enhancing the model's transferability towards the continually changing target domain. We introduce a multi-task setting for PCoTTA, which is practical and realistic, handling multiple tasks within one unified model during the continual adaptation. Our PCoTTA involves three key components: automatic prototype mixture (APM), Gaussian Splatted feature shifting (GSFS), and contrastive prototype repulsion (CPR). Firstly, APM is designed to automatically mix the source prototypes with the learnable prototypes with a similarity balancing factor, avoiding catastrophic forgetting. Then, GSFS dynamically shifts the testing sample toward the source domain, mitigating error accumulation in an online manner. In addition, CPR is proposed to pull the nearest learnable prototype close to the testing feature and push it away from other prototypes, making each prototype distinguishable during the adaptation. Experimental comparisons lead to a new benchmark, demonstrating PCoTTA's superiority in boosting the model's transferability towards the continually changing target domain.
    </details>
</div>
