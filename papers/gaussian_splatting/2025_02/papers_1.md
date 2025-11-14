# gaussian splatting - 2025_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.21093v1">FlexDrive: Toward Trajectory Flexibility in Driving Scene Reconstruction and Rendering</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Driving scene reconstruction and rendering have advanced significantly using the 3D Gaussian Splatting. However, most prior research has focused on the rendering quality along a pre-recorded vehicle path and struggles to generalize to out-of-path viewpoints, which is caused by the lack of high-quality supervision in those out-of-path views. To address this issue, we introduce an Inverse View Warping technique to create compact and high-quality images as supervision for the reconstruction of the out-of-path views, enabling high-quality rendering results for those views. For accurate and robust inverse view warping, a depth bootstrap strategy is proposed to obtain on-the-fly dense depth maps during the optimization process, overcoming the sparsity and incompleteness of LiDAR depth data. Our method achieves superior in-path and out-of-path reconstruction and rendering performance on the widely used Waymo Open dataset. In addition, a simulator-based benchmark is proposed to obtain the out-of-path ground truth and quantitatively evaluate the performance of out-of-path rendering, where our method outperforms previous methods by a significant margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03844v4">HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 Accpeted by CVPR 2025. Project page: https://gujiaqivadin.github.io/hybridgs/ Code: https://github.com/Yeyuqqwx/HybridGS Data: https://huggingface.co/Eto63277/HybridGS/tree/main
    </div>
    <details class="paper-abstract">
      Generating high-quality novel view renderings of 3D Gaussian Splatting (3DGS) in scenes featuring transient objects is challenging. We propose a novel hybrid representation, termed as HybridGS, using 2D Gaussians for transient objects per image and maintaining traditional 3D Gaussians for the whole static scenes. Note that, the 3DGS itself is better suited for modeling static scenes that assume multi-view consistency, but the transient objects appear occasionally and do not adhere to the assumption, thus we model them as planar objects from a single view, represented with 2D Gaussians. Our novel representation decomposes the scene from the perspective of fundamental viewpoint consistency, making it more reasonable. Additionally, we present a novel multi-view regulated supervision method for 3DGS that leverages information from co-visible regions, further enhancing the distinctions between the transients and statics. Then, we propose a straightforward yet effective multi-stage training strategy to ensure robust training and high-quality view synthesis across various settings. Experiments on benchmark datasets show our state-of-the-art performance of novel view synthesis in both indoor and outdoor scenes, even in the presence of distracting elements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20669v1">EndoPBR: Material and Lighting Estimation for Photorealistic Surgical Simulations via Physically-based Rendering</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 10 pages, 3 figures
    </div>
    <details class="paper-abstract">
      The lack of labeled datasets in 3D vision for surgical scenes inhibits the development of robust 3D reconstruction algorithms in the medical domain. Despite the popularity of Neural Radiance Fields and 3D Gaussian Splatting in the general computer vision community, these systems have yet to find consistent success in surgical scenes due to challenges such as non-stationary lighting and non-Lambertian surfaces. As a result, the need for labeled surgical datasets continues to grow. In this work, we introduce a differentiable rendering framework for material and lighting estimation from endoscopic images and known geometry. Compared to previous approaches that model lighting and material jointly as radiance, we explicitly disentangle these scene properties for robust and photorealistic novel view synthesis. To disambiguate the training process, we formulate domain-specific properties inherent in surgical scenes. Specifically, we model the scene lighting as a simple spotlight and material properties as a bidirectional reflectance distribution function, parameterized by a neural network. By grounding color predictions in the rendering equation, we can generate photorealistic images at arbitrary camera poses. We evaluate our method with various sequences from the Colonoscopy 3D Video Dataset and show that our method produces competitive novel view synthesis results compared with other approaches. Furthermore, we demonstrate that synthetic data can be used to develop 3D vision algorithms by finetuning a depth estimation model with our rendered outputs. Overall, we see that the depth estimation performance is on par with fine-tuning with the original real images.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18041v2">OpenFly: A Versatile Toolchain and Large-scale Benchmark for Aerial Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation (VLN) aims to guide agents through an environment by leveraging both language instructions and visual cues, playing a pivotal role in embodied AI. Indoor VLN has been extensively studied, whereas outdoor aerial VLN remains underexplored. The potential reason is that outdoor aerial view encompasses vast areas, making data collection more challenging, which results in a lack of benchmarks. To address this problem, we propose OpenFly, a platform comprising a versatile toolchain and large-scale benchmark for aerial VLN. Firstly, we develop a highly automated toolchain for data collection, enabling automatic point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Secondly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering diverse heights and lengths across 18 scenes. The corresponding visual data are generated using various rendering engines and advanced techniques, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). All data exhibit high visual quality. Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of the dataset. Thirdly, we propose OpenFly-Agent, a keyframe-aware VLN model, which takes language instructions, current observations, and historical keyframes as input, and outputs flight actions directly. Extensive analyses and experiments are conducted, showcasing the superiority of our OpenFly platform and OpenFly-Agent. The toolchain, dataset, and codes will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18416v3">3D StreetUnveiler with Semantic-aware 2DGS -- a simple baseline</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 Project page: https://streetunveiler.github.io
    </div>
    <details class="paper-abstract">
      Unveiling an empty street from crowded observations captured by in-car cameras is crucial for autonomous driving. However, removing all temporarily static objects, such as stopped vehicles and standing pedestrians, presents a significant challenge. Unlike object-centric 3D inpainting, which relies on thorough observation in a small scene, street scene cases involve long trajectories that differ from previous 3D inpainting tasks. The camera-centric moving environment of captured videos further complicates the task due to the limited degree and time duration of object observation. To address these obstacles, we introduce StreetUnveiler to reconstruct an empty street. StreetUnveiler learns a 3D representation of the empty street from crowded observations. Our representation is based on the hard-label semantic 2D Gaussian Splatting (2DGS) for its scalability and ability to identify Gaussians to be removed. We inpaint rendered image after removing unwanted Gaussians to provide pseudo-labels and subsequently re-optimize the 2DGS. Given its temporal continuous movement, we divide the empty street scene into observed, partial-observed, and unobserved regions, which we propose to locate through a rendered alpha map. This decomposition helps us to minimize the regions that need to be inpainted. To enhance the temporal consistency of the inpainting, we introduce a novel time-reversal framework to inpaint frames in reverse order and use later frames as references for earlier frames to fully utilize the long-trajectory observations. Our experiments conducted on the street scene dataset successfully reconstructed a 3D representation of the empty street. The mesh representation of the empty street can be extracted for further applications. The project page and more visualizations can be found at: https://streetunveiler.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20386v1">ATLAS Navigator: Active Task-driven LAnguage-embedded Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      We address the challenge of task-oriented navigation in unstructured and unknown environments, where robots must incrementally build and reason on rich, metric-semantic maps in real time. Since tasks may require clarification or re-specification, it is necessary for the information in the map to be rich enough to enable generalization across a wide range of tasks. To effectively execute tasks specified in natural language, we propose a hierarchical representation built on language-embedded Gaussian splatting that enables both sparse semantic planning that lends itself to online operation and dense geometric representation for collision-free navigation. We validate the effectiveness of our method through real-world robot experiments conducted in both cluttered indoor and kilometer-scale outdoor environments, with a competitive ratio of about 60% against privileged baselines. Experiment videos and more details can be found on our project page: https://atlasnav.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20378v1">Efficient Gaussian Splatting for Monocular Dynamic Scene Rendering via Sparse Time-Variant Attribute Modeling</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 AAAI 2025
    </div>
    <details class="paper-abstract">
      Rendering dynamic scenes from monocular videos is a crucial yet challenging task. The recent deformable Gaussian Splatting has emerged as a robust solution to represent real-world dynamic scenes. However, it often leads to heavily redundant Gaussians, attempting to fit every training view at various time steps, leading to slower rendering speeds. Additionally, the attributes of Gaussians in static areas are time-invariant, making it unnecessary to model every Gaussian, which can cause jittering in static regions. In practice, the primary bottleneck in rendering speed for dynamic scenes is the number of Gaussians. In response, we introduce Efficient Dynamic Gaussian Splatting (EDGS), which represents dynamic scenes via sparse time-variant attribute modeling. Our approach formulates dynamic scenes using a sparse anchor-grid representation, with the motion flow of dense Gaussians calculated via a classical kernel representation. Furthermore, we propose an unsupervised strategy to efficiently filter out anchors corresponding to static areas. Only anchors associated with deformable objects are input into MLPs to query time-variant attributes. Experiments on two real-world datasets demonstrate that our EDGS significantly improves the rendering speed with superior rendering quality compared to previous state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06838v3">Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Implicit Neural Representation (INR) has been successfully employed for Arbitrary-scale Super-Resolution (ASR). However, INR-based models need to query the multi-layer perceptron module numerous times and render a pixel in each query, resulting in insufficient representation capability and computational efficiency. Recently, Gaussian Splatting (GS) has shown its advantages over INR in both visual quality and rendering speed in 3D tasks, which motivates us to explore whether GS can be employed for the ASR task. However, directly applying GS to ASR is exceptionally challenging because the original GS is an optimization-based method through overfitting each single scene, while in ASR we aim to learn a single model that can generalize to different images and scaling factors. We overcome these challenges by developing two novel techniques. Firstly, to generalize GS for ASR, we elaborately design an architecture to predict the corresponding image-conditioned Gaussians of the input low-resolution image in a feed-forward manner. Each Gaussian can fit the shape and direction of an area of complex textures, showing powerful representation capability. Secondly, we implement an efficient differentiable 2D GPU/CUDA-based scale-aware rasterization to render super-resolved images by sampling discrete RGB values from the predicted continuous Gaussians. Via end-to-end training, our optimized network, namely GSASR, can perform ASR for any image and unseen scaling factors. Extensive experiments validate the effectiveness of our proposed method. The code and models will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17860v2">UniGS: Unified Language-Image-3D Pretraining with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 ICLR 2025; Corrected citation of Uni3D;
    </div>
    <details class="paper-abstract">
      Recent advancements in multi-modal 3D pre-training methods have shown promising efficacy in learning joint representations of text, images, and point clouds. However, adopting point clouds as 3D representation fails to fully capture the intricacies of the 3D world and exhibits a noticeable gap between the discrete points and the dense 2D pixels of images. To tackle this issue, we propose UniGS, integrating 3D Gaussian Splatting (3DGS) into multi-modal pre-training to enhance the 3D representation. We first rely on the 3DGS representation to model the 3D world as a collection of 3D Gaussians with color and opacity, incorporating all the information of the 3D scene while establishing a strong connection with 2D images. Then, to achieve Language-Image-3D pertaining, UniGS starts with a pre-trained vision-language model to establish a shared visual and textual space through extensive real-world image-text pairs. Subsequently, UniGS employs a 3D encoder to align the optimized 3DGS with the Language-Image representations to learn unified multi-modal representations. To facilitate the extraction of global explicit 3D features by the 3D encoder and achieve better cross-modal alignment, we additionally introduce a novel Gaussian-Aware Guidance module that guides the learning of fine-grained representations of the 3D domain. Through extensive experiments across the Objaverse, ABO, MVImgNet and SUN RGBD datasets with zero-shot classification, text-driven retrieval and open-world understanding tasks, we demonstrate the effectiveness of UniGS in learning a more general and stronger aligned multi-modal representation. Specifically, UniGS achieves leading results across different 3D tasks with remarkable improvements over previous SOTA, Uni3D, including on zero-shot classification (+9.36%), text-driven retrieval (+4.3%) and open-world understanding (+7.92%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19800v1">No Parameters, No Problem: 3D Gaussian Splatting without Camera Intrinsics and Extrinsics</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has made significant progress in scene reconstruction and novel view synthesis, it still heavily relies on accurately pre-computed camera intrinsics and extrinsics, such as focal length and camera poses. In order to mitigate this dependency, the previous efforts have focused on optimizing 3DGS without the need for camera poses, yet camera intrinsics remain necessary. To further loose the requirement, we propose a joint optimization method to train 3DGS from an image collection without requiring either camera intrinsics or extrinsics. To achieve this goal, we introduce several key improvements during the joint training of 3DGS. We theoretically derive the gradient of the camera intrinsics, allowing the camera intrinsics to be optimized simultaneously during training. Moreover, we integrate global track information and select the Gaussian kernels associated with each track, which will be trained and automatically rescaled to an infinitesimally small size, closely approximating surface points, and focusing on enforcing multi-view consistency and minimizing reprojection errors, while the remaining kernels continue to serve their original roles. This hybrid training strategy nicely unifies the camera parameters estimation and 3DGS training. Extensive evaluations demonstrate that the proposed method achieves state-of-the-art (SOTA) performance on both public and synthetic datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19782v1">Open-Vocabulary Semantic Part Segmentation of 3D Human</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 3DV 2025
    </div>
    <details class="paper-abstract">
      3D part segmentation is still an open problem in the field of 3D vision and AR/VR. Due to limited 3D labeled data, traditional supervised segmentation methods fall short in generalizing to unseen shapes and categories. Recently, the advancement in vision-language models' zero-shot abilities has brought a surge in open-world 3D segmentation methods. While these methods show promising results for 3D scenes or objects, they do not generalize well to 3D humans. In this paper, we present the first open-vocabulary segmentation method capable of handling 3D human. Our framework can segment the human category into desired fine-grained parts based on the textual prompt. We design a simple segmentation pipeline, leveraging SAM to generate multi-view proposals in 2D and proposing a novel HumanCLIP model to create unified embeddings for visual and textual inputs. Compared with existing pre-trained CLIP models, the HumanCLIP model yields more accurate embeddings for human-centric contents. We also design a simple-yet-effective MaskFusion module, which classifies and fuses multi-view features into 3D semantic masks without complex voting and grouping mechanisms. The design of decoupling mask proposals and text input also significantly boosts the efficiency of per-prompt inference. Experimental results on various 3D human datasets show that our method outperforms current state-of-the-art open-vocabulary 3D segmentation methods by a large margin. In addition, we show that our method can be directly applied to various 3D representations including meshes, point clouds, and 3D Gaussian Splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06526v3">Generative Gaussian Splatting for Unbounded 3D City Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 CVPR 2025. Project Page: https://haozhexie.com/project/gaussian-city
    </div>
    <details class="paper-abstract">
      3D city generation with NeRF-based methods shows promising generation results but is computationally inefficient. Recently 3D Gaussian Splatting (3D-GS) has emerged as a highly efficient alternative for object-level 3D generation. However, adapting 3D-GS from finite-scale 3D objects and humans to infinite-scale 3D cities is non-trivial. Unbounded 3D city generation entails significant storage overhead (out-of-memory issues), arising from the need to expand points to billions, often demanding hundreds of Gigabytes of VRAM for a city scene spanning 10km^2. In this paper, we propose GaussianCity, a generative Gaussian Splatting framework dedicated to efficiently synthesizing unbounded 3D cities with a single feed-forward pass. Our key insights are two-fold: 1) Compact 3D Scene Representation: We introduce BEV-Point as a highly compact intermediate representation, ensuring that the growth in VRAM usage for unbounded scenes remains constant, thus enabling unbounded city generation. 2) Spatial-aware Gaussian Attribute Decoder: We present spatial-aware BEV-Point decoder to produce 3D Gaussian attributes, which leverages Point Serializer to integrate the structural and contextual characteristics of BEV points. Extensive experiments demonstrate that GaussianCity achieves state-of-the-art results in both drone-view and street-view 3D city generation. Notably, compared to CityDreamer, GaussianCity exhibits superior performance with a speedup of 60 times (10.72 FPS v.s. 0.18 FPS).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03844v3">HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 Accpeted by CVPR 2025. Project page: https://gujiaqivadin.github.io/hybridgs/ Code: https://github.com/Yeyuqqwx/HybridGS Data: https://huggingface.co/Eto63277/HybridGS/tree/main
    </div>
    <details class="paper-abstract">
      Generating high-quality novel view renderings of 3D Gaussian Splatting (3DGS) in scenes featuring transient objects is challenging. We propose a novel hybrid representation, termed as HybridGS, using 2D Gaussians for transient objects per image and maintaining traditional 3D Gaussians for the whole static scenes. Note that, the 3DGS itself is better suited for modeling static scenes that assume multi-view consistency, but the transient objects appear occasionally and do not adhere to the assumption, thus we model them as planar objects from a single view, represented with 2D Gaussians. Our novel representation decomposes the scene from the perspective of fundamental viewpoint consistency, making it more reasonable. Additionally, we present a novel multi-view regulated supervision method for 3DGS that leverages information from co-visible regions, further enhancing the distinctions between the transients and statics. Then, we propose a straightforward yet effective multi-stage training strategy to ensure robust training and high-quality view synthesis across various settings. Experiments on benchmark datasets show our state-of-the-art performance of novel view synthesis in both indoor and outdoor scenes, even in the presence of distracting elements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02720v3">3D-HGS: 3D Half-Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 8 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Photo-realistic image rendering from scene 3D reconstruction is a fundamental problem in 3D computer vision. This domain has seen considerable advancements owing to the advent of recent neural rendering techniques. These techniques predominantly aim to focus on learning volumetric representations of 3D scenes and refining these representations via loss functions derived from their rendering. Among these, 3D Gaussian Splatting (3D-GS) has emerged as a preferred method, surpassing Neural Radiance Fields' (NeRFs) quality and rendering speed. 3D-GS uses parameterized 3D Gaussians to model both spatial locations and color information, combined with a tile-based fast rendering technique. Despite its superior performance, using 3D Gaussian kernels has inherent limitations in accurately representing discontinuous functions, notably at edges and corners corresponding to shape discontinuities, and across varying textures due to color discontinuities. In this paper, we introduce 3D Half-Gaussian (\textbf{3D-HGS}) kernels, which can be used as a plug-and-play kernel, to address this issue. Our experiments demonstrate their capability to improve the performance of current 3D-GS related methods and achieve state-of-the-art rendering quality performance on various datasets without compromising their rendering speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2409.12954v3">GStex: Per-Primitive Texturing of 2D Gaussian Splatting for Decoupled Appearance and Geometry Modeling</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 Project page: https://lessvrong.com/cs/gstex. WACV 2025 camera-ready version
    </div>
    <details class="paper-abstract">
      Gaussian splatting has demonstrated excellent performance for view synthesis and scene reconstruction. The representation achieves photorealistic quality by optimizing the position, scale, color, and opacity of thousands to millions of 2D or 3D Gaussian primitives within a scene. However, since each Gaussian primitive encodes both appearance and geometry, these attributes are strongly coupled--thus, high-fidelity appearance modeling requires a large number of Gaussian primitives, even when the scene geometry is simple (e.g., for a textured planar surface). We propose to texture each 2D Gaussian primitive so that even a single Gaussian can be used to capture appearance details. By employing per-primitive texturing, our appearance representation is agnostic to the topology and complexity of the scene's geometry. We show that our approach, GStex, yields improved visual quality over prior work in texturing Gaussian splats. Furthermore, we demonstrate that our decoupling enables improved novel view synthesis performance compared to 2D Gaussian splatting when reducing the number of Gaussian primitives, and that GStex can be used for scene appearance editing and re-texturing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12954v3">GStex: Per-Primitive Texturing of 2D Gaussian Splatting for Decoupled Appearance and Geometry Modeling</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 Project page: https://lessvrong.com/cs/gstex. WACV 2025 camera-ready version
    </div>
    <details class="paper-abstract">
      Gaussian splatting has demonstrated excellent performance for view synthesis and scene reconstruction. The representation achieves photorealistic quality by optimizing the position, scale, color, and opacity of thousands to millions of 2D or 3D Gaussian primitives within a scene. However, since each Gaussian primitive encodes both appearance and geometry, these attributes are strongly coupled--thus, high-fidelity appearance modeling requires a large number of Gaussian primitives, even when the scene geometry is simple (e.g., for a textured planar surface). We propose to texture each 2D Gaussian primitive so that even a single Gaussian can be used to capture appearance details. By employing per-primitive texturing, our appearance representation is agnostic to the topology and complexity of the scene's geometry. We show that our approach, GStex, yields improved visual quality over prior work in texturing Gaussian splats. Furthermore, we demonstrate that our decoupling enables improved novel view synthesis performance compared to 2D Gaussian splatting when reducing the number of Gaussian primitives, and that GStex can be used for scene appearance editing and re-texturing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19318v1">Does 3D Gaussian Splatting Need Accurate Volumetric Rendering?</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 To be published in Eurogrpahics 2025, code: https://github.com/cg-tuwien/does_3d_gaussian_splatting_need_accurate_volumetric_rendering
    </div>
    <details class="paper-abstract">
      Since its introduction, 3D Gaussian Splatting (3DGS) has become an important reference method for learning 3D representations of a captured scene, allowing real-time novel-view synthesis with high visual quality and fast training times. Neural Radiance Fields (NeRFs), which preceded 3DGS, are based on a principled ray-marching approach for volumetric rendering. In contrast, while sharing a similar image formation model with NeRF, 3DGS uses a hybrid rendering solution that builds on the strengths of volume rendering and primitive rasterization. A crucial benefit of 3DGS is its performance, achieved through a set of approximations, in many cases with respect to volumetric rendering theory. A naturally arising question is whether replacing these approximations with more principled volumetric rendering solutions can improve the quality of 3DGS. In this paper, we present an in-depth analysis of the various approximations and assumptions used by the original 3DGS solution. We demonstrate that, while more accurate volumetric rendering can help for low numbers of primitives, the power of efficient optimization and the large number of Gaussians allows 3DGS to outperform volumetric rendering despite its approximations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05242v2">Scaffold-SLAM: Structured 3D Gaussians for Simultaneous Localization and Photorealistic Mapping</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 12 pages, 6 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently revolutionized novel view synthesis in the Simultaneous Localization and Mapping (SLAM). However, existing SLAM methods utilizing 3DGS have failed to provide high-quality novel view rendering for monocular, stereo, and RGB-D cameras simultaneously. Notably, some methods perform well for RGB-D cameras but suffer significant degradation in rendering quality for monocular cameras. In this paper, we present Scaffold-SLAM, which delivers simultaneous localization and high-quality photorealistic mapping across monocular, stereo, and RGB-D cameras. We introduce two key innovations to achieve this state-of-the-art visual quality. First, we propose Appearance-from-Motion embedding, enabling 3D Gaussians to better model image appearance variations across different camera poses. Second, we introduce a frequency regularization pyramid to guide the distribution of Gaussians, allowing the model to effectively capture finer details in the scene. Extensive experiments on monocular, stereo, and RGB-D datasets demonstrate that Scaffold-SLAM significantly outperforms state-of-the-art methods in photorealistic mapping quality, e.g., PSNR is 16.76% higher in the TUM RGB-D datasets for monocular cameras.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19459v1">Building Interactable Replicas of Complex Articulated Objects via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      Building articulated objects is a key challenge in computer vision. Existing methods often fail to effectively integrate information across different object states, limiting the accuracy of part-mesh reconstruction and part dynamics modeling, particularly for complex multi-part articulated objects. We introduce ArtGS, a novel approach that leverages 3D Gaussians as a flexible and efficient representation to address these issues. Our method incorporates canonical Gaussians with coarse-to-fine initialization and updates for aligning articulated part information across different object states, and employs a skinning-inspired part dynamics modeling module to improve both part-mesh reconstruction and articulation learning. Extensive experiments on both synthetic and real-world datasets, including a new benchmark for complex multi-part objects, demonstrate that ArtGS achieves state-of-the-art performance in joint parameter estimation and part mesh reconstruction. Our approach significantly improves reconstruction quality and efficiency, especially for multi-part articulated objects. Additionally, we provide comprehensive analyses of our design choices, validating the effectiveness of each component to highlight potential areas for future improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19457v1">Compression in 3D Gaussian Splatting: A Survey of Methods, Trends, and Future Directions</a></div>
    <div class="paper-meta">
      📅 2025-02-26
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as a pioneering approach in explicit scene rendering and computer graphics. Unlike traditional neural radiance field (NeRF) methods, which typically rely on implicit, coordinate-based models to map spatial coordinates to pixel values, 3DGS utilizes millions of learnable 3D Gaussians. Its differentiable rendering technique and inherent capability for explicit scene representation and manipulation positions 3DGS as a potential game-changer for the next generation of 3D reconstruction and representation technologies. This enables 3DGS to deliver real-time rendering speeds while offering unparalleled editability levels. However, despite its advantages, 3DGS suffers from substantial memory and storage requirements, posing challenges for deployment on resource-constrained devices. In this survey, we provide a comprehensive overview focusing on the scalability and compression of 3DGS. We begin with a detailed background overview of 3DGS, followed by a structured taxonomy of existing compression methods. Additionally, we analyze and compare current methods from the topological perspective, evaluating their strengths and limitations in terms of fidelity, compression ratios, and computational efficiency. Furthermore, we explore how advancements in efficient NeRF representations can inspire future developments in 3DGS optimization. Finally, we conclude with current research challenges and highlight key directions for future exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2502.17531v1">Laplace-Beltrami Operator for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      With the rising popularity of 3D Gaussian splatting and the expanse of applications from rendering to 3D reconstruction, there comes also a need for geometry processing applications directly on this new representation. While considering the centers of Gaussians as a point cloud or meshing them is an option that allows to apply existing algorithms, this might ignore information present in the data or be unnecessarily expensive. Additionally, Gaussian splatting tends to contain a large number of outliers which do not affect the rendering quality but need to be handled correctly in order not to produce noisy results in geometry processing applications. In this work, we propose a formulation to compute the Laplace-Beltrami operator, a widely used tool in geometry processing, directly on Gaussian splatting using the Mahalanobis distance. While conceptually similar to a point cloud Laplacian, our experiments show superior accuracy on the point clouds encoded in the Gaussian splatting centers and, additionally, the operator can be used to evaluate the quality of the output during optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.12954v3">GStex: Per-Primitive Texturing of 2D Gaussian Splatting for Decoupled Appearance and Geometry Modeling</a></div>
    <div class="paper-meta">
      📅 2025-02-26
      | 💬 Project page: https://lessvrong.com/cs/gstex. WACV 2025 camera-ready version
    </div>
    <details class="paper-abstract">
      Gaussian splatting has demonstrated excellent performance for view synthesis and scene reconstruction. The representation achieves photorealistic quality by optimizing the position, scale, color, and opacity of thousands to millions of 2D or 3D Gaussian primitives within a scene. However, since each Gaussian primitive encodes both appearance and geometry, these attributes are strongly coupled--thus, high-fidelity appearance modeling requires a large number of Gaussian primitives, even when the scene geometry is simple (e.g., for a textured planar surface). We propose to texture each 2D Gaussian primitive so that even a single Gaussian can be used to capture appearance details. By employing per-primitive texturing, our appearance representation is agnostic to the topology and complexity of the scene's geometry. We show that our approach, GStex, yields improved visual quality over prior work in texturing Gaussian splats. Furthermore, we demonstrate that our decoupling enables improved novel view synthesis performance compared to 2D Gaussian splatting when reducing the number of Gaussian primitives, and that GStex can be used for scene appearance editing and re-texturing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18041v1">OpenFly: A Versatile Toolchain and Large-scale Benchmark for Aerial Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation (VLN) aims to guide agents through an environment by leveraging both language instructions and visual cues, playing a pivotal role in embodied AI. Indoor VLN has been extensively studied, whereas outdoor aerial VLN remains underexplored. The potential reason is that outdoor aerial view encompasses vast areas, making data collection more challenging, which results in a lack of benchmarks. To address this problem, we propose OpenFly, a platform comprising a versatile toolchain and large-scale benchmark for aerial VLN. Firstly, we develop a highly automated toolchain for data collection, enabling automatic point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Secondly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering diverse heights and lengths across 18 scenes. The corresponding visual data are generated using various rendering engines and advanced techniques, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). All data exhibit high visual quality. Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of the dataset. Thirdly, we propose OpenFly-Agent, a keyframe-aware VLN model, which takes language instructions, current observations, and historical keyframes as input, and outputs flight actions directly. Extensive analyses and experiments are conducted, showcasing the superiority of our OpenFly platform and OpenFly-Agent. The toolchain, dataset, and codes will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17288v2">GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Occupancy estimation has become a prominent task in 3D computer vision, particularly within the autonomous driving community. In this paper, we present a novel approach to occupancy estimation, termed GaussianFlowOcc, which is inspired by Gaussian Splatting and replaces traditional dense voxel grids with a sparse 3D Gaussian representation. Our efficient model architecture based on a Gaussian Transformer significantly reduces computational and memory requirements by eliminating the need for expensive 3D convolutions used with inefficient voxel-based representations that predominantly represent empty 3D spaces. GaussianFlowOcc effectively captures scene dynamics by estimating temporal flow for each Gaussian during the overall network training process, offering a straightforward solution to a complex problem that is often neglected by existing methods. Moreover, GaussianFlowOcc is designed for scalability, as it employs weak supervision and does not require costly dense 3D voxel annotations based on additional data (e.g., LiDAR). Through extensive experimentation, we demonstrate that GaussianFlowOcc significantly outperforms all previous methods for weakly supervised occupancy estimation on the nuScenes dataset while featuring an inference speed that is 50 times faster than current SOTA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17860v1">UniGS: Unified Language-Image-3D Pretraining with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in multi-modal 3D pre-training methods have shown promising efficacy in learning joint representations of text, images, and point clouds. However, adopting point clouds as 3D representation fails to fully capture the intricacies of the 3D world and exhibits a noticeable gap between the discrete points and the dense 2D pixels of images. To tackle this issue, we propose UniGS, integrating 3D Gaussian Splatting (3DGS) into multi-modal pre-training to enhance the 3D representation. We first rely on the 3DGS representation to model the 3D world as a collection of 3D Gaussians with color and opacity, incorporating all the information of the 3D scene while establishing a strong connection with 2D images. Then, to achieve Language-Image-3D pertaining, UniGS starts with a pre-trained vision-language model to establish a shared visual and textual space through extensive real-world image-text pairs. Subsequently, UniGS employs a 3D encoder to align the optimized 3DGS with the Language-Image representations to learn unified multi-modal representations. To facilitate the extraction of global explicit 3D features by the 3D encoder and achieve better cross-modal alignment, we additionally introduce a novel Gaussian-Aware Guidance module that guides the learning of fine-grained representations of the 3D domain. Through extensive experiments across the Objaverse, ABO, MVImgNet and SUN RGBD datasets with zero-shot classification, text-driven retrieval and open-world understanding tasks, we demonstrate the effectiveness of UniGS in learning a more general and stronger aligned multi-modal representation. Specifically, UniGS achieves leading results across different 3D tasks with remarkable improvements over previous SOTA, Uni3D, including on zero-shot classification (+9.36%), text-driven retrieval (+4.3%) and open-world understanding (+7.92%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2403.17888v3">2D Gaussian Splatting for Geometrically Accurate Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 13 pages, 12 figures. Corrected Eq.7
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently revolutionized radiance field reconstruction, achieving high quality novel view synthesis and fast rendering speed without baking. However, 3DGS fails to accurately represent surfaces due to the multi-view inconsistent nature of 3D Gaussians. We present 2D Gaussian Splatting (2DGS), a novel approach to model and reconstruct geometrically accurate radiance fields from multi-view images. Our key idea is to collapse the 3D volume into a set of 2D oriented planar Gaussian disks. Unlike 3D Gaussians, 2D Gaussians provide view-consistent geometry while modeling surfaces intrinsically. To accurately recover thin surfaces and achieve stable optimization, we introduce a perspective-correct 2D splatting process utilizing ray-splat intersection and rasterization. Additionally, we incorporate depth distortion and normal consistency terms to further enhance the quality of the reconstructions. We demonstrate that our differentiable renderer allows for noise-free and detailed geometry reconstruction while maintaining competitive appearance quality, fast training speed, and real-time rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2502.16652v1">Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      We introduce Dr. Splat, a novel approach for open-vocabulary 3D scene understanding leveraging 3D Gaussian Splatting. Unlike existing language-embedded 3DGS methods, which rely on a rendering process, our method directly associates language-aligned CLIP embeddings with 3D Gaussians for holistic 3D scene understanding. The key of our method is a language feature registration technique where CLIP embeddings are assigned to the dominant Gaussians intersected by each pixel-ray. Moreover, we integrate Product Quantization (PQ) trained on general large-scale image data to compactly represent embeddings without per-scene optimization. Experiments demonstrate that our approach significantly outperforms existing approaches in 3D perception benchmarks, such as open-vocabulary 3D semantic segmentation, 3D object localization, and 3D object selection tasks. For video results, please visit : https://drsplat.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17377v1">Graph-Guided Scene Reconstruction from Images with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      This paper investigates an open research challenge of reconstructing high-quality, large 3D open scenes from images. It is observed existing methods have various limitations, such as requiring precise camera poses for input and dense viewpoints for supervision. To perform effective and efficient 3D scene reconstruction, we propose a novel graph-guided 3D scene reconstruction framework, GraphGS. Specifically, given a set of images captured by RGB cameras on a scene, we first design a spatial prior-based scene structure estimation method. This is then used to create a camera graph that includes information about the camera topology. Further, we propose to apply the graph-guided multi-view consistency constraint and adaptive sampling strategy to the 3D Gaussian Splatting optimization process. This greatly alleviates the issue of Gaussian points overfitting to specific sparse viewpoints and expedites the 3D reconstruction process. We demonstrate GraphGS achieves high-fidelity 3D reconstruction from images, which presents state-of-the-art performance through quantitative and qualitative evaluation across multiple datasets. Project Page: https://3dagentworld.github.io/graphgs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17288v1">GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Occupancy estimation has become a prominent task in 3D computer vision, particularly within the autonomous driving community. In this paper, we present a novel approach to occupancy estimation, termed GaussianFlowOcc, which is inspired by Gaussian Splatting and replaces traditional dense voxel grids with a sparse 3D Gaussian representation. Our efficient model architecture based on a Gaussian Transformer significantly reduces computational and memory requirements by eliminating the need for expensive 3D convolutions used with inefficient voxel-based representations that predominantly represent empty 3D spaces. GaussianFlowOcc effectively captures scene dynamics by estimating temporal flow for each Gaussian during the overall network training process, offering a straightforward solution to a complex problem that is often neglected by existing methods. Moreover, GaussianFlowOcc is designed for scalability, as it employs weak supervision and does not require costly dense 3D voxel annotations based on additional data (e.g., LiDAR). Through extensive experimentation, we demonstrate that GaussianFlowOcc significantly outperforms all previous methods for weakly supervised occupancy estimation on the nuScenes dataset while featuring an inference speed that is 50 times faster than current SOTA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17078v1">VR-Pipe: Streamlining Hardware Graphics Pipeline for Volume Rendering</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 To appear at the 31st International Symposium on High-Performance Computer Architecture (HPCA 2025)
    </div>
    <details class="paper-abstract">
      Graphics rendering that builds on machine learning and radiance fields is gaining significant attention due to its outstanding quality and speed in generating photorealistic images from novel viewpoints. However, prior work has primarily focused on evaluating its performance through software-based rendering on programmable shader cores, leaving its performance when exploiting fixed-function graphics units largely unexplored. In this paper, we investigate the performance implications of performing radiance field rendering on the hardware graphics pipeline. In doing so, we implement the state-of-the-art radiance field method, 3D Gaussian splatting, using graphics APIs and evaluate it across synthetic and real-world scenes on today's graphics hardware. Based on our analysis, we present VR-Pipe, which seamlessly integrates two innovations into graphics hardware to streamline the hardware pipeline for volume rendering, such as radiance field methods. First, we introduce native hardware support for early termination by repurposing existing special-purpose hardware in modern GPUs. Second, we propose multi-granular tile binning with quad merging, which opportunistically blends fragments in shader cores before passing them to fixed-function blending units. Our evaluation shows that VR-Pipe greatly improves rendering performance, achieving up to a 2.78x speedup over the conventional graphics pipeline with negligible hardware overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15309v2">DynamicGSG: Dynamic 3D Gaussian Scene Graphs for Environment Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      In real-world scenarios, environment changes caused by human or agent activities make it extremely challenging for robots to perform various long-term tasks. Recent works typically struggle to effectively understand and adapt to dynamic environments due to the inability to update their environment representations in memory according to environment changes and lack of fine-grained reconstruction of the environments. To address these challenges, we propose DynamicGSG, a dynamic, high-fidelity, open-vocabulary scene graph construction system leveraging Gaussian splatting. DynamicGSG builds hierarchical scene graphs using advanced vision language models to represent the spatial and semantic relationships between objects in the environments, utilizes a joint feature loss we designed to supervise Gaussian instance grouping while optimizing the Gaussian maps, and locally updates the Gaussian scene graphs according to real environment changes for long-term environment adaptation. Experiments and ablation studies demonstrate the performance and efficacy of our proposed method in terms of semantic segmentation, language-guided object retrieval, and reconstruction quality. Furthermore, we validate the dynamic updating capabilities of our system in real laboratory environments. The source code and supplementary experimental materials will be released at:~\href{https://github.com/GeLuzhou/Dynamic-GSG}{https://github.com/GeLuzhou/Dynamic-GSG}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17531v1">Laplace-Beltrami Operator for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      With the rising popularity of 3D Gaussian splatting and the expanse of applications from rendering to 3D reconstruction, there comes also a need for geometry processing applications directly on this new representation. While considering the centers of Gaussians as a point cloud or meshing them is an option that allows to apply existing algorithms, this might ignore information present in the data or be unnecessarily expensive. Additionally, Gaussian splatting tends to contain a large number of outliers which do not affect the rendering quality but need to be handled correctly in order not to produce noisy results in geometry processing applications. In this work, we propose a formulation to compute the Laplace-Beltrami operator, a widely used tool in geometry processing, directly on Gaussian splatting using the Mahalanobis distance. While conceptually similar to a point cloud Laplacian, our experiments show superior accuracy on the point clouds encoded in the Gaussian splatting centers and, additionally, the operator can be used to evaluate the quality of the output during optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.17531v1">Laplace-Beltrami Operator for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      With the rising popularity of 3D Gaussian splatting and the expanse of applications from rendering to 3D reconstruction, there comes also a need for geometry processing applications directly on this new representation. While considering the centers of Gaussians as a point cloud or meshing them is an option that allows to apply existing algorithms, this might ignore information present in the data or be unnecessarily expensive. Additionally, Gaussian splatting tends to contain a large number of outliers which do not affect the rendering quality but need to be handled correctly in order not to produce noisy results in geometry processing applications. In this work, we propose a formulation to compute the Laplace-Beltrami operator, a widely used tool in geometry processing, directly on Gaussian splatting using the Mahalanobis distance. While conceptually similar to a point cloud Laplacian, our experiments show superior accuracy on the point clouds encoded in the Gaussian splatting centers and, additionally, the operator can be used to evaluate the quality of the output during optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16748v1">GS-TransUNet: Integrated 2D Gaussian Splatting and Transformer UNet for Accurate Skin Lesion Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-23
      | 💬 12 pages, 7 figures, SPIE Medical Imaging 2025
    </div>
    <details class="paper-abstract">
      We can achieve fast and consistent early skin cancer detection with recent developments in computer vision and deep learning techniques. However, the existing skin lesion segmentation and classification prediction models run independently, thus missing potential efficiencies from their integrated execution. To unify skin lesion analysis, our paper presents the Gaussian Splatting - Transformer UNet (GS-TransUNet), a novel approach that synergistically combines 2D Gaussian splatting with the Transformer UNet architecture for automated skin cancer diagnosis. Our unified deep learning model efficiently delivers dual-function skin lesion classification and segmentation for clinical diagnosis. Evaluated on ISIC-2017 and PH2 datasets, our network demonstrates superior performance compared to existing state-of-the-art models across multiple metrics through 5-fold cross-validation. Our findings illustrate significant advancements in the precision of segmentation and classification. This integration sets new benchmarks in the field and highlights the potential for further research into multi-task medical image analysis methodologies, promising enhancements in automated diagnostic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16652v1">Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration</a></div>
    <div class="paper-meta">
      📅 2025-02-23
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      We introduce Dr. Splat, a novel approach for open-vocabulary 3D scene understanding leveraging 3D Gaussian Splatting. Unlike existing language-embedded 3DGS methods, which rely on a rendering process, our method directly associates language-aligned CLIP embeddings with 3D Gaussians for holistic 3D scene understanding. The key of our method is a language feature registration technique where CLIP embeddings are assigned to the dominant Gaussians intersected by each pixel-ray. Moreover, we integrate Product Quantization (PQ) trained on general large-scale image data to compactly represent embeddings without per-scene optimization. Experiments demonstrate that our approach significantly outperforms existing approaches in 3D perception benchmarks, such as open-vocabulary 3D semantic segmentation, 3D object localization, and 3D object selection tasks. For video results, please visit : https://drsplat.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16475v1">Dragen3D: Multiview Geometry Consistent 3D Gaussian Generation with Drag-Based Control</a></div>
    <div class="paper-meta">
      📅 2025-02-23
    </div>
    <details class="paper-abstract">
      Single-image 3D generation has emerged as a prominent research topic, playing a vital role in virtual reality, 3D modeling, and digital content creation. However, existing methods face challenges such as a lack of multi-view geometric consistency and limited controllability during the generation process, which significantly restrict their usability. % To tackle these challenges, we introduce Dragen3D, a novel approach that achieves geometrically consistent and controllable 3D generation leveraging 3D Gaussian Splatting (3DGS). We introduce the Anchor-Gaussian Variational Autoencoder (Anchor-GS VAE), which encodes a point cloud and a single image into anchor latents and decode these latents into 3DGS, enabling efficient latent-space generation. To enable multi-view geometry consistent and controllable generation, we propose a Seed-Point-Driven strategy: first generate sparse seed points as a coarse geometry representation, then map them to anchor latents via the Seed-Anchor Mapping Module. Geometric consistency is ensured by the easily learned sparse seed points, and users can intuitively drag the seed points to deform the final 3DGS geometry, with changes propagated through the anchor latents. To the best of our knowledge, we are the first to achieve geometrically controllable 3D Gaussian generation and editing without relying on 2D diffusion priors, delivering comparable 3D generation quality to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.16652v1">Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration</a></div>
    <div class="paper-meta">
      📅 2025-02-23
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      We introduce Dr. Splat, a novel approach for open-vocabulary 3D scene understanding leveraging 3D Gaussian Splatting. Unlike existing language-embedded 3DGS methods, which rely on a rendering process, our method directly associates language-aligned CLIP embeddings with 3D Gaussians for holistic 3D scene understanding. The key of our method is a language feature registration technique where CLIP embeddings are assigned to the dominant Gaussians intersected by each pixel-ray. Moreover, we integrate Product Quantization (PQ) trained on general large-scale image data to compactly represent embeddings without per-scene optimization. Experiments demonstrate that our approach significantly outperforms existing approaches in 3D perception benchmarks, such as open-vocabulary 3D semantic segmentation, 3D object localization, and 3D object selection tasks. For video results, please visit : https://drsplat.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11356v3">GSORB-SLAM: Gaussian Splatting SLAM benefits from ORB features and Transmittance information</a></div>
    <div class="paper-meta">
      📅 2025-02-22
    </div>
    <details class="paper-abstract">
      The emergence of 3D Gaussian Splatting (3DGS) has recently ignited a renewed wave of research in dense visual SLAM. However, existing approaches encounter challenges, including sensitivity to artifacts and noise, suboptimal selection of training viewpoints, and the absence of global optimization. In this paper, we propose GSORB-SLAM, a dense SLAM framework that integrates 3DGS with ORB features through a tightly coupled optimization pipeline. To mitigate the effects of noise and artifacts, we propose a novel geometric representation and optimization method for tracking, which significantly enhances localization accuracy and robustness. For high-fidelity mapping, we develop an adaptive Gaussian expansion and regularization method that facilitates compact yet expressive scene modeling while suppressing redundant primitives. Furthermore, we design a hybrid graph-based viewpoint selection mechanism that effectively reduces overfitting and accelerates convergence. Extensive evaluations across various datasets demonstrate that our system achieves state-of-the-art performance in both tracking precision-improving RMSE by 16.2% compared to ORB-SLAM2 baselines-and reconstruction quality-improving PSNR by 3.93 dB compared to 3DGS-SLAM baselines. The project: https://aczheng-cai.github.io/gsorb-slam.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.20291v2">RL-GSBridge: 3D Gaussian Splatting Based Real2Sim2Real Method for Robotic Manipulation Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 7 pages, 5 figures, 4 tables. Accepted by ICRA2025
    </div>
    <details class="paper-abstract">
      Sim-to-Real refers to the process of transferring policies learned in simulation to the real world, which is crucial for achieving practical robotics applications. However, recent Sim2real methods either rely on a large amount of augmented data or large learning models, which is inefficient for specific tasks. In recent years, with the emergence of radiance field reconstruction methods, especially 3D Gaussian splatting, it has become possible to construct realistic real-world scenes. To this end, we propose RL-GSBridge, a novel real-to-sim-to-real framework which incorporates 3D Gaussian Splatting into the conventional RL simulation pipeline, enabling zero-shot sim-to-real transfer for vision-based deep reinforcement learning. We introduce a mesh-based 3D GS method with soft binding constraints, enhancing the rendering quality of mesh models. Then utilizing a GS editing approach to synchronize the rendering with the physics simulator, RL-GSBridge could reflect the visual interactions of the physical robot accurately. Through a series of sim-to-real experiments, including grasping and pick-and-place tasks, we demonstrate that RL-GSBridge maintains a satisfactory success rate in real-world task completion during sim-to-real transfer. Furthermore, a series of rendering metrics and visualization results indicate that our proposed mesh-based 3D GS reduces artifacts in unstructured objects, demonstrating more realistic rendering performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.17888v3">2D Gaussian Splatting for Geometrically Accurate Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 13 pages, 12 figures. Corrected Eq.7
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently revolutionized radiance field reconstruction, achieving high quality novel view synthesis and fast rendering speed without baking. However, 3DGS fails to accurately represent surfaces due to the multi-view inconsistent nature of 3D Gaussians. We present 2D Gaussian Splatting (2DGS), a novel approach to model and reconstruct geometrically accurate radiance fields from multi-view images. Our key idea is to collapse the 3D volume into a set of 2D oriented planar Gaussian disks. Unlike 3D Gaussians, 2D Gaussians provide view-consistent geometry while modeling surfaces intrinsically. To accurately recover thin surfaces and achieve stable optimization, we introduce a perspective-correct 2D splatting process utilizing ray-splat intersection and rasterization. Additionally, we incorporate depth distortion and normal consistency terms to further enhance the quality of the reconstructions. We demonstrate that our differentiable renderer allows for noise-free and detailed geometry reconstruction while maintaining competitive appearance quality, fast training speed, and real-time rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.03890v5">A Survey on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 Ongoing project
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (GS) has emerged as a transformative technique in explicit radiance field and computer graphics. This innovative approach, characterized by the use of millions of learnable 3D Gaussians, represents a significant departure from mainstream neural radiance field approaches, which predominantly use implicit, coordinate-based models to map spatial coordinates to pixel values. 3D GS, with its explicit scene representation and differentiable rendering algorithm, not only promises real-time rendering capability but also introduces unprecedented levels of editability. This positions 3D GS as a potential game-changer for the next generation of 3D reconstruction and representation. In the present paper, we provide the first systematic overview of the recent developments and critical contributions in the domain of 3D GS. We begin with a detailed exploration of the underlying principles and the driving forces behind the emergence of 3D GS, laying the groundwork for understanding its significance. A focal point of our discussion is the practical applicability of 3D GS. By enabling unprecedented rendering speed, 3D GS opens up a plethora of applications, ranging from virtual reality to interactive media and beyond. This is complemented by a comparative analysis of leading 3D GS models, evaluated across various benchmark tasks to highlight their performance and practical utility. The survey concludes by identifying current challenges and suggesting potential avenues for future research. Through this survey, we aim to provide a valuable resource for both newcomers and seasoned researchers, fostering further exploration and advancement in explicit radiance field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2403.17888v3">2D Gaussian Splatting for Geometrically Accurate Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 13 pages, 12 figures. Corrected Eq.7
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently revolutionized radiance field reconstruction, achieving high quality novel view synthesis and fast rendering speed without baking. However, 3DGS fails to accurately represent surfaces due to the multi-view inconsistent nature of 3D Gaussians. We present 2D Gaussian Splatting (2DGS), a novel approach to model and reconstruct geometrically accurate radiance fields from multi-view images. Our key idea is to collapse the 3D volume into a set of 2D oriented planar Gaussian disks. Unlike 3D Gaussians, 2D Gaussians provide view-consistent geometry while modeling surfaces intrinsically. To accurately recover thin surfaces and achieve stable optimization, we introduce a perspective-correct 2D splatting process utilizing ray-splat intersection and rasterization. Additionally, we incorporate depth distortion and normal consistency terms to further enhance the quality of the reconstructions. We demonstrate that our differentiable renderer allows for noise-free and detailed geometry reconstruction while maintaining competitive appearance quality, fast training speed, and real-time rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04459v2">Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Project page at https://svraster.github.io/ Code at https://github.com/NVlabs/svraster
    </div>
    <details class="paper-abstract">
      We propose an efficient radiance field rendering algorithm that incorporates a rasterization process on adaptive sparse voxels without neural networks or 3D Gaussians. There are two key contributions coupled with the proposed system. The first is to adaptively and explicitly allocate sparse voxels to different levels of detail within scenes, faithfully reproducing scene details with $65536^3$ grid resolution while achieving high rendering frame rates. Second, we customize a rasterizer for efficient adaptive sparse voxels rendering. We render voxels in the correct depth order by using ray direction-dependent Morton ordering, which avoids the well-known popping artifact found in Gaussian splatting. Our method improves the previous neural-free voxel model by over 4db PSNR and more than 10x FPS speedup, achieving state-of-the-art comparable novel-view synthesis results. Additionally, our voxel representation is seamlessly compatible with grid-based 3D processing techniques such as Volume Fusion, Voxel Pooling, and Marching Cubes, enabling a wide range of future extensions and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15633v1">RGB-Only Gaussian Splatting SLAM for Unbounded Outdoor Scenes</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 ICRA 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a popular solution in SLAM, as it can produce high-fidelity novel views. However, previous GS-based methods primarily target indoor scenes and rely on RGB-D sensors or pre-trained depth estimation models, hence underperforming in outdoor scenarios. To address this issue, we propose a RGB-only gaussian splatting SLAM method for unbounded outdoor scenes--OpenGS-SLAM. Technically, we first employ a pointmap regression network to generate consistent pointmaps between frames for pose estimation. Compared to commonly used depth maps, pointmaps include spatial relationships and scene geometry across multiple views, enabling robust camera pose estimation. Then, we propose integrating the estimated camera poses with 3DGS rendering as an end-to-end differentiable pipeline. Our method achieves simultaneous optimization of camera poses and 3DGS scene parameters, significantly enhancing system tracking accuracy. Specifically, we also design an adaptive scale mapper for the pointmap regression network, which provides more accurate pointmap mapping to the 3DGS map representation. Our experiments on the Waymo dataset demonstrate that OpenGS-SLAM reduces tracking error to 9.8\% of previous 3DGS methods, and achieves state-of-the-art results in novel view synthesis. Project Page: https://3dagentworld.github.io/opengs-slam/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02332v3">SplatOverflow: Asynchronous Hardware Troubleshooting</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Our accompanying video figure is available at: https://youtu.be/rdtaUo2Lo38
    </div>
    <details class="paper-abstract">
      As tools for designing and manufacturing hardware become more accessible, smaller producers can develop and distribute novel hardware. However, processes for supporting end-user hardware troubleshooting or routine maintenance aren't well defined. As a result, providing technical support for hardware remains ad-hoc and challenging to scale. Inspired by patterns that helped scale software troubleshooting, we propose a workflow for asynchronous hardware troubleshooting: SplatOverflow. SplatOverflow creates a novel boundary object, the SplatOverflow scene, that users reference to communicate about hardware. A scene comprises a 3D Gaussian Splat of the user's hardware registered onto the hardware's CAD model. The splat captures the current state of the hardware, and the registered CAD model acts as a referential anchor for troubleshooting instructions. With SplatOverflow, remote maintainers can directly address issues and author instructions in the user's workspace. Workflows containing multiple instructions can easily be shared between users and recontextualized in new environments. In this paper, we describe the design of SplatOverflow, the workflows it enables, and its utility to different kinds of users. We also validate that non-experts can use SplatOverflow to troubleshoot common problems with a 3D printer in a usability study. Project Page: https://amritkwatra.com/research/splatoverflow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15309v1">DynamicGSG: Dynamic 3D Gaussian Scene Graphs for Environment Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      In real-world scenarios, the environment changes caused by agents or human activities make it extremely challenging for robots to perform various long-term tasks. To effectively understand and adapt to dynamic environments, the perception system of a robot needs to extract instance-level semantic information, reconstruct the environment in a fine-grained manner, and update its environment representation in memory according to environment changes. To address these challenges, We propose \textbf{DynamicGSG}, a dynamic, high-fidelity, open-vocabulary scene graph generation system leveraging Gaussian splatting. Our system comprises three key components: (1) constructing hierarchical scene graphs using advanced vision foundation models to represent the spatial and semantic relationships of objects in the environment, (2) designing a joint feature loss to optimize the Gaussian map for incremental high-fidelity reconstruction, and (3) updating the Gaussian map and scene graph according to real environment changes for long-term environment adaptation. Experiments and ablation studies demonstrate the performance and efficacy of the proposed method in terms of semantic segmentation, language-guided object retrieval, and reconstruction quality. Furthermore, we have validated the dynamic updating capabilities of our system in real laboratory environments. The source code will be released at:~\href{https://github.com/GeLuzhou/Dynamic-GSG}{https://github.com/GeLuzhou/DynamicGSG}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14684v1">CDGS: Confidence-Aware Depth Regularization for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown significant advantages in novel view synthesis (NVS), particularly in achieving high rendering speeds and high-quality results. However, its geometric accuracy in 3D reconstruction remains limited due to the lack of explicit geometric constraints during optimization. This paper introduces CDGS, a confidence-aware depth regularization approach developed to enhance 3DGS. We leverage multi-cue confidence maps of monocular depth estimation and sparse Structure-from-Motion depth to adaptively adjust depth supervision during the optimization process. Our method demonstrates improved geometric detail preservation in early training stages and achieves competitive performance in both NVS quality and geometric accuracy. Experiments on the publicly available Tanks and Temples benchmark dataset show that our method achieves more stable convergence behavior and more accurate geometric reconstruction results, with improvements of up to 2.31 dB in PSNR for NVS and consistently lower geometric errors in M3C2 distance metrics. Notably, our method reaches comparable F-scores to the original 3DGS with only 50% of the training iterations. We expect this work will facilitate the development of efficient and accurate 3D reconstruction systems for real-world applications such as digital twin creation, heritage preservation, or forestry applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00486v3">CaRtGS: Computational Alignment for Real-Time Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 Accepted by IEEE Robotics and Automation Letters (RA-L)
    </div>
    <details class="paper-abstract">
      Simultaneous Localization and Mapping (SLAM) is pivotal in robotics, with photorealistic scene reconstruction emerging as a key challenge. To address this, we introduce Computational Alignment for Real-Time Gaussian Splatting SLAM (CaRtGS), a novel method enhancing the efficiency and quality of photorealistic scene reconstruction in real-time environments. Leveraging 3D Gaussian Splatting (3DGS), CaRtGS achieves superior rendering quality and processing speed, which is crucial for scene photorealistic reconstruction. Our approach tackles computational misalignment in Gaussian Splatting SLAM (GS-SLAM) through an adaptive strategy that enhances optimization iterations, addresses long-tail optimization, and refines densification. Experiments on Replica, TUM-RGBD, and VECtor datasets demonstrate CaRtGS's effectiveness in achieving high-fidelity rendering with fewer Gaussian primitives. This work propels SLAM towards real-time, photorealistic dense rendering, significantly advancing photorealistic scene representation. For the benefit of the research community, we release the code and accompanying videos on our project website: https://dapengfeng.github.io/cartgs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12518v3">Hier-SLAM: Scaling-up Semantics in SLAM with a Hierarchically Categorical Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 Accepted for publication at ICRA 2025. Code will be released soon
    </div>
    <details class="paper-abstract">
      We propose Hier-SLAM, a semantic 3D Gaussian Splatting SLAM method featuring a novel hierarchical categorical representation, which enables accurate global 3D semantic mapping, scaling-up capability, and explicit semantic label prediction in the 3D world. The parameter usage in semantic SLAM systems increases significantly with the growing complexity of the environment, making it particularly challenging and costly for scene understanding. To address this problem, we introduce a novel hierarchical representation that encodes semantic information in a compact form into 3D Gaussian Splatting, leveraging the capabilities of large language models (LLMs). We further introduce a novel semantic loss designed to optimize hierarchical semantic information through both inter-level and cross-level optimization. Furthermore, we enhance the whole SLAM system, resulting in improved tracking and mapping performance. Our Hier-SLAM outperforms existing dense SLAM methods in both mapping and tracking accuracy, while achieving a 2x operation speed-up. Additionally, it exhibits competitive performance in rendering semantic segmentation in small synthetic scenes, with significantly reduced storage and training time requirements. Rendering FPS impressively reaches 2,000 with semantic information and 3,000 without it. Most notably, it showcases the capability of handling the complex real-world scene with more than 500 semantic classes, highlighting its valuable scaling-up capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14235v1">OG-Gaussian: Occupancy Based Street Gaussians for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Accurate and realistic 3D scene reconstruction enables the lifelike creation of autonomous driving simulation environments. With advancements in 3D Gaussian Splatting (3DGS), previous studies have applied it to reconstruct complex dynamic driving scenes. These methods typically require expensive LiDAR sensors and pre-annotated datasets of dynamic objects. To address these challenges, we propose OG-Gaussian, a novel approach that replaces LiDAR point clouds with Occupancy Grids (OGs) generated from surround-view camera images using Occupancy Prediction Network (ONet). Our method leverages the semantic information in OGs to separate dynamic vehicles from static street background, converting these grids into two distinct sets of initial point clouds for reconstructing both static and dynamic objects. Additionally, we estimate the trajectories and poses of dynamic objects through a learning-based approach, eliminating the need for complex manual annotations. Experiments on Waymo Open dataset demonstrate that OG-Gaussian is on par with the current state-of-the-art in terms of reconstruction quality and rendering speed, achieving an average PSNR of 35.13 and a rendering speed of 143 FPS, while significantly reducing computational costs and economic overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18974v2">3D-Adapter: Geometry-Consistent Multi-View Diffusion for High-Quality 3D Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 Project page: https://lakonik.github.io/3d-adapter/
    </div>
    <details class="paper-abstract">
      Multi-view image diffusion models have significantly advanced open-domain 3D object generation. However, most existing models rely on 2D network architectures that lack inherent 3D biases, resulting in compromised geometric consistency. To address this challenge, we introduce 3D-Adapter, a plug-in module designed to infuse 3D geometry awareness into pretrained image diffusion models. Central to our approach is the idea of 3D feedback augmentation: for each denoising step in the sampling loop, 3D-Adapter decodes intermediate multi-view features into a coherent 3D representation, then re-encodes the rendered RGBD views to augment the pretrained base model through feature addition. We study two variants of 3D-Adapter: a fast feed-forward version based on Gaussian splatting and a versatile training-free version utilizing neural fields and meshes. Our extensive experiments demonstrate that 3D-Adapter not only greatly enhances the geometry quality of text-to-multi-view models such as Instant3D and Zero123++, but also enables high-quality 3D generation using the plain text-to-image Stable Diffusion. Furthermore, we showcase the broad application potential of 3D-Adapter by presenting high quality results in text-to-3D, image-to-3D, text-to-texture, and text-to-avatar tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08449v3">OccGaussian: 3D Gaussian Splatting for Occluded Human Rendering</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 We have decided to withdraw this paper because the results require further verification or additional experimental data. We plan to resubmit an updated version once the necessary work is completed
    </div>
    <details class="paper-abstract">
      Rendering dynamic 3D human from monocular videos is crucial for various applications such as virtual reality and digital entertainment. Most methods assume the people is in an unobstructed scene, while various objects may cause the occlusion of body parts in real-life scenarios. Previous method utilizing NeRF for surface rendering to recover the occluded areas, but it requiring more than one day to train and several seconds to render, failing to meet the requirements of real-time interactive applications. To address these issues, we propose OccGaussian based on 3D Gaussian Splatting, which can be trained within 6 minutes and produces high-quality human renderings up to 160 FPS with occluded input. OccGaussian initializes 3D Gaussian distributions in the canonical space, and we perform occlusion feature query at occluded regions, the aggregated pixel-align feature is extracted to compensate for the missing information. Then we use Gaussian Feature MLP to further process the feature along with the occlusion-aware loss functions to better perceive the occluded area. Extensive experiments both in simulated and real-world occlusions, demonstrate that our method achieves comparable or even superior performance compared to the state-of-the-art method. And we improving training and inference speeds by 250x and 800x, respectively. Our code will be available for research purposes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14938v1">GS-Cache: A GS-Cache Inference Framework for Large-scale Gaussian Splatting Models</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Rendering large-scale 3D Gaussian Splatting (3DGS) model faces significant challenges in achieving real-time, high-fidelity performance on consumer-grade devices. Fully realizing the potential of 3DGS in applications such as virtual reality (VR) requires addressing critical system-level challenges to support real-time, immersive experiences. We propose GS-Cache, an end-to-end framework that seamlessly integrates 3DGS's advanced representation with a highly optimized rendering system. GS-Cache introduces a cache-centric pipeline to eliminate redundant computations, an efficiency-aware scheduler for elastic multi-GPU rendering, and optimized CUDA kernels to overcome computational bottlenecks. This synergy between 3DGS and system design enables GS-Cache to achieve up to 5.35x performance improvement, 35% latency reduction, and 42% lower GPU memory usage, supporting 2K binocular rendering at over 120 FPS with high visual quality. By bridging the gap between 3DGS's representation power and the demands of VR systems, GS-Cache establishes a scalable and efficient framework for real-time neural rendering in immersive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14931v1">Hier-SLAM++: Neuro-Symbolic Semantic SLAM with a Hierarchically Categorical Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 15 pages. Under review
    </div>
    <details class="paper-abstract">
      We propose Hier-SLAM++, a comprehensive Neuro-Symbolic semantic 3D Gaussian Splatting SLAM method with both RGB-D and monocular input featuring an advanced hierarchical categorical representation, which enables accurate pose estimation as well as global 3D semantic mapping. The parameter usage in semantic SLAM systems increases significantly with the growing complexity of the environment, making scene understanding particularly challenging and costly. To address this problem, we introduce a novel and general hierarchical representation that encodes both semantic and geometric information in a compact form into 3D Gaussian Splatting, leveraging the capabilities of large language models (LLMs) as well as the 3D generative model. By utilizing the proposed hierarchical tree structure, semantic information is symbolically represented and learned in an end-to-end manner. We further introduce a novel semantic loss designed to optimize hierarchical semantic information through both inter-level and cross-level optimization. Additionally, we propose an improved SLAM system to support both RGB-D and monocular inputs using a feed-forward model. To the best of our knowledge, this is the first semantic monocular Gaussian Splatting SLAM system, significantly reducing sensor requirements for 3D semantic understanding and broadening the applicability of semantic Gaussian SLAM system. We conduct experiments on both synthetic and real-world datasets, demonstrating superior or on-par performance with state-of-the-art NeRF-based and Gaussian-based SLAM systems, while significantly reducing storage and training time requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13803v1">3D Gaussian Splatting aided Localization for Large and Complex Indoor-Environments</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      The field of visual localization has been researched for several decades and has meanwhile found many practical applications. Despite the strong progress in this field, there are still challenging situations in which established methods fail. We present an approach to significantly improve the accuracy and reliability of established visual localization methods by adding rendered images. In detail, we first use a modern visual SLAM approach that provides a 3D Gaussian Splatting (3DGS) based map to create reference data. We demonstrate that enriching reference data with images rendered from 3DGS at randomly sampled poses significantly improves the performance of both geometry-based visual localization and Scene Coordinate Regression (SCR) methods. Through comprehensive evaluation in a large industrial environment, we analyze the performance impact of incorporating these additional rendered views.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11453v2">Hybrid Explicit Representation for Ultra-Realistic Head Avatars</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      We introduce a novel approach to creating ultra-realistic head avatars and rendering them in real-time (>30fps at $2048 \times 1334$ resolution). First, we propose a hybrid explicit representation that combines the advantages of two primitive-based efficient rendering techniques. UV-mapped 3D mesh is utilized to capture sharp and rich textures on smooth surfaces, while 3D Gaussian Splatting is employed to represent complex geometric structures. In the pipeline of modeling an avatar, after tracking parametric models based on captured multi-view RGB videos, our goal is to simultaneously optimize the texture and opacity map of mesh, as well as a set of 3D Gaussian splats localized and rigged onto the mesh facets. Specifically, we perform $\alpha$-blending on the color and opacity values based on the merged and re-ordered z-buffer from the rasterization results of mesh and 3DGS. This process involves the mesh and 3DGS adaptively fitting the captured visual information to outline a high-fidelity digital avatar. To avoid artifacts caused by Gaussian splats crossing the mesh facets, we design a stable hybrid depth sorting strategy. Experiments illustrate that our modeled results exceed those of state-of-the-art approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04832v2">WRF-GS: Wireless Radiation Field Reconstruction with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 accepted to the IEEE International Conference on Computer Communications (INFOCOM 2025)
    </div>
    <details class="paper-abstract">
      Wireless channel modeling plays a pivotal role in designing, analyzing, and optimizing wireless communication systems. Nevertheless, developing an effective channel modeling approach has been a longstanding challenge. This issue has been escalated due to the denser network deployment, larger antenna arrays, and wider bandwidth in 5G and beyond networks. To address this challenge, we put forth WRF-GS, a novel framework for channel modeling based on wireless radiation field (WRF) reconstruction using 3D Gaussian splatting. WRF-GS employs 3D Gaussian primitives and neural networks to capture the interactions between the environment and radio signals, enabling efficient WRF reconstruction and visualization of the propagation characteristics. The reconstructed WRF can then be used to synthesize the spatial spectrum for comprehensive wireless channel characterization. Notably, with a small number of measurements, WRF-GS can synthesize new spatial spectra within milliseconds for a given scene, thereby enabling latency-sensitive applications. Experimental results demonstrate that WRF-GS outperforms existing methods for spatial spectrum synthesis, such as ray tracing and other deep-learning approaches. Moreover, WRF-GS achieves superior performance in the channel state information prediction task, surpassing existing methods by a significant margin of more than 2.43 dB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14129v1">GlossGau: Efficient Inverse Rendering for Glossy Surface with Anisotropic Spherical Gaussian</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      The reconstruction of 3D objects from calibrated photographs represents a fundamental yet intricate challenge in the domains of computer graphics and vision. Although neural reconstruction approaches based on Neural Radiance Fields (NeRF) have shown remarkable capabilities, their processing costs remain substantial. Recently, the advent of 3D Gaussian Splatting (3D-GS) largely improves the training efficiency and facilitates to generate realistic rendering in real-time. However, due to the limited ability of Spherical Harmonics (SH) to represent high-frequency information, 3D-GS falls short in reconstructing glossy objects. Researchers have turned to enhance the specular expressiveness of 3D-GS through inverse rendering. Yet these methods often struggle to maintain the training and rendering efficiency, undermining the benefits of Gaussian Splatting techniques. In this paper, we introduce GlossGau, an efficient inverse rendering framework that reconstructs scenes with glossy surfaces while maintaining training and rendering speeds comparable to vanilla 3D-GS. Specifically, we explicitly model the surface normals, Bidirectional Reflectance Distribution Function (BRDF) parameters, as well as incident lights and use Anisotropic Spherical Gaussian (ASG) to approximate the per-Gaussian Normal Distribution Function under the microfacet model. We utilize 2D Gaussian Splatting (2D-GS) as foundational primitives and apply regularization to significantly alleviate the normal estimation challenge encountered in related works. Experiments demonstrate that GlossGau achieves competitive or superior reconstruction on datasets with glossy surfaces. Compared with previous GS-based works that address the specular surface, our optimization time is considerably less.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14004v1">Inter3D: A Benchmark and Strong Baseline for Human-Interactive 3D Object Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Recent advancements in implicit 3D reconstruction methods, e.g., neural rendering fields and Gaussian splatting, have primarily focused on novel view synthesis of static or dynamic objects with continuous motion states. However, these approaches struggle to efficiently model a human-interactive object with n movable parts, requiring 2^n separate models to represent all discrete states. To overcome this limitation, we propose Inter3D, a new benchmark and approach for novel state synthesis of human-interactive objects. We introduce a self-collected dataset featuring commonly encountered interactive objects and a new evaluation pipeline, where only individual part states are observed during training, while part combination states remain unseen. We also propose a strong baseline approach that leverages Space Discrepancy Tensors to efficiently modelling all states of an object. To alleviate the impractical constraints on camera trajectories across training states, we propose a Mutual State Regularization mechanism to enhance the spatial density consistency of movable parts. In addition, we explore two occupancy grid sampling strategies to facilitate training efficiency. We conduct extensive experiments on the proposed benchmark, showcasing the challenges of the task and the superiority of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03228v2">GARAD-SLAM: 3D GAussian splatting for Real-time Anti Dynamic SLAM</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 The paper was accepted by ICRA 2025
    </div>
    <details class="paper-abstract">
      The 3D Gaussian Splatting (3DGS)-based SLAM system has garnered widespread attention due to its excellent performance in real-time high-fidelity rendering. However, in real-world environments with dynamic objects, existing 3DGS-based SLAM systems often face mapping errors and tracking drift issues. To address these problems, we propose GARAD-SLAM, a real-time 3DGS-based SLAM system tailored for dynamic scenes. In terms of tracking, unlike traditional methods, we directly perform dynamic segmentation on Gaussians and map them back to the front-end to obtain dynamic point labels through a Gaussian pyramid network, achieving precise dynamic removal and robust tracking. For mapping, we impose rendering penalties on dynamically labeled Gaussians, which are updated through the network, to avoid irreversible erroneous removal caused by simple pruning. Our results on real-world datasets demonstrate that our method is competitive in tracking compared to baseline methods, generating fewer artifacts and higher-quality reconstructions in rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13196v1">GS-QA: Comprehensive Quality Assessment Benchmark for Gaussian Splatting View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) offers a promising alternative to Neural Radiance Fields (NeRF) for real-time 3D scene rendering. Using a set of 3D Gaussians to represent complex geometry and appearance, GS achieves faster rendering times and reduced memory consumption compared to the neural network approach used in NeRF. However, quality assessment of GS-generated static content is not yet explored in-depth. This paper describes a subjective quality assessment study that aims to evaluate synthesized videos obtained with several static GS state-of-the-art methods. The methods were applied to diverse visual scenes, covering both 360-degree and forward-facing (FF) camera trajectories. Moreover, the performance of 18 objective quality metrics was analyzed using the scores resulting from the subjective study, providing insights into their strengths, limitations, and alignment with human perception. All videos and scores are made available providing a comprehensive database that can be used as benchmark on GS view synthesis and objective quality metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11801v1">3D Gaussian Inpainting with Depth-Guided Cross-View Consistency</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      When performing 3D inpainting using novel-view rendering methods like Neural Radiance Field (NeRF) or 3D Gaussian Splatting (3DGS), how to achieve texture and geometry consistency across camera views has been a challenge. In this paper, we propose a framework of 3D Gaussian Inpainting with Depth-Guided Cross-View Consistency (3DGIC) for cross-view consistent 3D inpainting. Guided by the rendered depth information from each training view, our 3DGIC exploits background pixels visible across different views for updating the inpainting mask, allowing us to refine the 3DGS for inpainting purposes.Through extensive experiments on benchmark datasets, we confirm that our 3DGIC outperforms current state-of-the-art 3D inpainting methods quantitatively and qualitatively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11782v1">Exploring the Versal AI Engine for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Dataflow-oriented spatial architectures are the emerging paradigm for higher computation performance and efficiency. AMD Versal AI Engine is a commercial spatial architecture consisting of tiles of VLIW processors supporting SIMD operations arranged in a two-dimensional mesh. The architecture requires the explicit design of task assignments and dataflow configurations for each tile to maximize performance, demanding advanced techniques and meticulous design. However, a few works revealed the performance characteristics of the Versal AI Engine through practical workloads. In this work, we provide the comprehensive performance evaluation of the Versal AI Engine using Gaussian feature computation in 3D Gaussian splatting as a practical workload, and we then propose a novel dedicated algorithm to fully exploit the hardware architecture. The computations of 3D Gaussian splatting include matrix multiplications and color computations utilizing high-dimensional spherical harmonic coefficients. These tasks are processed efficiently by leveraging the SIMD capabilities and their instruction-level parallelism. Additionally, pipelined processing is achieved by assigning different tasks to individual cores, thereby fully exploiting the spatial parallelism of AI Engines. The proposed method demonstrated a 226-fold throughput increase in simulation-based evaluation, outperforming a naive approach. These findings provide valuable insights for application development that effectively harnesses the spatial and architectural advantages of AI Engines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11642v1">GaussianMotion: End-to-End Learning of Animatable Gaussian Avatars with Pose Guidance from Text</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      In this paper, we introduce GaussianMotion, a novel human rendering model that generates fully animatable scenes aligned with textual descriptions using Gaussian Splatting. Although existing methods achieve reasonable text-to-3D generation of human bodies using various 3D representations, they often face limitations in fidelity and efficiency, or primarily focus on static models with limited pose control. In contrast, our method generates fully animatable 3D avatars by combining deformable 3D Gaussian Splatting with text-to-3D score distillation, achieving high fidelity and efficient rendering for arbitrary poses. By densely generating diverse random poses during optimization, our deformable 3D human model learns to capture a wide range of natural motions distilled from a pose-conditioned diffusion model in an end-to-end manner. Furthermore, we propose Adaptive Score Distillation that effectively balances realistic detail and smoothness to achieve optimal 3D results. Experimental results demonstrate that our approach outperforms existing baselines by producing high-quality textures in both static and animated results, and by generating diverse 3D human models from various textual inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18643v2">3D Reconstruction of Shoes for Augmented Reality</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      This paper introduces a mobile-based solution that enhances online shoe shopping through 3D modeling and Augmented Reality (AR), leveraging the efficiency of 3D Gaussian Splatting. Addressing the limitations of static 2D images, the framework generates realistic 3D shoe models from 2D images, achieving an average Peak Signal-to-Noise Ratio (PSNR) of 32, and enables immersive AR interactions via smartphones. A custom shoe segmentation dataset of 3120 images was created, with the best-performing segmentation model achieving an Intersection over Union (IoU) score of 0.95. This paper demonstrates the potential of 3D modeling and AR to revolutionize online shopping by offering realistic virtual interactions, with applicability across broader fashion categories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12231v1">PUGS: Zero-shot Physical Understanding with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 ICRA 2025, Project page: https://evernorif.github.io/PUGS/
    </div>
    <details class="paper-abstract">
      Current robotic systems can understand the categories and poses of objects well. But understanding physical properties like mass, friction, and hardness, in the wild, remains challenging. We propose a new method that reconstructs 3D objects using the Gaussian splatting representation and predicts various physical properties in a zero-shot manner. We propose two techniques during the reconstruction phase: a geometry-aware regularization loss function to improve the shape quality and a region-aware feature contrastive loss function to promote region affinity. Two other new techniques are designed during inference: a feature-based property propagation module and a volume integration module tailored for the Gaussian representation. Our framework is named as zero-shot physical understanding with Gaussian splatting, or PUGS. PUGS achieves new state-of-the-art results on the standard benchmark of ABO-500 mass prediction. We provide extensive quantitative ablations and qualitative visualization to demonstrate the mechanism of our designs. We show the proposed methodology can help address challenging real-world grasping tasks. Our codes, data, and models are available at https://github.com/EverNorif/PUGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14895v1">High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian Representation</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Accepted as an Oral paper at ICLR 2025. Project page: https://ziyeeee.github.io/stcgs.github.io
    </div>
    <details class="paper-abstract">
      Weather nowcasting is an essential task that involves predicting future radar echo sequences based on current observations, offering significant benefits for disaster management, transportation, and urban planning. Current prediction methods are limited by training and storage efficiency, mainly focusing on 2D spatial predictions at specific altitudes. Meanwhile, 3D volumetric predictions at each timestamp remain largely unexplored. To address such a challenge, we introduce a comprehensive framework for 3D radar sequence prediction in weather nowcasting, using the newly proposed SpatioTemporal Coherent Gaussian Splatting (STC-GS) for dynamic radar representation and GauMamba for efficient and accurate forecasting. Specifically, rather than relying on a 4D Gaussian for dynamic scene reconstruction, STC-GS optimizes 3D scenes at each frame by employing a group of Gaussians while effectively capturing their movements across consecutive frames. It ensures consistent tracking of each Gaussian over time, making it particularly effective for prediction tasks. With the temporally correlated Gaussian groups established, we utilize them to train GauMamba, which integrates a memory mechanism into the Mamba framework. This allows the model to learn the temporal evolution of Gaussian groups while efficiently handling a large volume of Gaussian tokens. As a result, it achieves both efficiency and accuracy in forecasting a wide range of dynamic meteorological radar signals. The experimental results demonstrate that our STC-GS can efficiently represent 3D radar sequences with over $16\times$ higher spatial resolution compared with the existing 3D representation methods, while GauMamba outperforms state-of-the-art methods in forecasting a broad spectrum of high-dynamic weather conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10988v1">OMG: Opacity Matters in Material Modeling with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 Published as a conference paper at ICLR 2025
    </div>
    <details class="paper-abstract">
      Decomposing geometry, materials and lighting from a set of images, namely inverse rendering, has been a long-standing problem in computer vision and graphics. Recent advances in neural rendering enable photo-realistic and plausible inverse rendering results. The emergence of 3D Gaussian Splatting has boosted it to the next level by showing real-time rendering potentials. An intuitive finding is that the models used for inverse rendering do not take into account the dependency of opacity w.r.t. material properties, namely cross section, as suggested by optics. Therefore, we develop a novel approach that adds this dependency to the modeling itself. Inspired by radiative transfer, we augment the opacity term by introducing a neural network that takes as input material properties to provide modeling of cross section and a physically correct activation function. The gradients for material properties are therefore not only from color but also from opacity, facilitating a constraint for their optimization. Therefore, the proposed method incorporates more accurate physical properties compared to previous works. We implement our method into 3 different baselines that use Gaussian Splatting for inverse rendering and achieve significant improvements universally in terms of novel view synthesis and material modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10975v1">GS-GVINS: A Tightly-integrated GNSS-Visual-Inertial Navigation System Augmented by 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Recently, the emergence of 3D Gaussian Splatting (3DGS) has drawn significant attention in the area of 3D map reconstruction and visual SLAM. While extensive research has explored 3DGS for indoor trajectory tracking using visual sensor alone or in combination with Light Detection and Ranging (LiDAR) and Inertial Measurement Unit (IMU), its integration with GNSS for large-scale outdoor navigation remains underexplored. To address these concerns, we proposed GS-GVINS: a tightly-integrated GNSS-Visual-Inertial Navigation System augmented by 3DGS. This system leverages 3D Gaussian as a continuous differentiable scene representation in largescale outdoor environments, enhancing navigation performance through the constructed 3D Gaussian map. Notably, GS-GVINS is the first GNSS-Visual-Inertial navigation application that directly utilizes the analytical jacobians of SE3 camera pose with respect to 3D Gaussians. To maintain the quality of 3DGS rendering in extreme dynamic states, we introduce a motionaware 3D Gaussian pruning mechanism, updating the map based on relative pose translation and the accumulated opacity along the camera ray. For validation, we test our system under different driving environments: open-sky, sub-urban, and urban. Both self-collected and public datasets are used for evaluation. The results demonstrate the effectiveness of GS-GVINS in enhancing navigation accuracy across diverse driving environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10827v1">E-3DGS: Event-Based Novel View Rendering of Large-Scale Scenes Using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-15
      | 💬 15 pages, 10 figures and 3 tables; project page: https://4dqv.mpi-inf.mpg.de/E3DGS/; International Conference on 3D Vision (3DV) 2025
    </div>
    <details class="paper-abstract">
      Novel view synthesis techniques predominantly utilize RGB cameras, inheriting their limitations such as the need for sufficient lighting, susceptibility to motion blur, and restricted dynamic range. In contrast, event cameras are significantly more resilient to these limitations but have been less explored in this domain, particularly in large-scale settings. Current methodologies primarily focus on front-facing or object-oriented (360-degree view) scenarios. For the first time, we introduce 3D Gaussians for event-based novel view synthesis. Our method reconstructs large and unbounded scenes with high visual quality. We contribute the first real and synthetic event datasets tailored for this setting. Our method demonstrates superior novel view synthesis and consistently outperforms the baseline EventNeRF by a margin of 11-25% in PSNR (dB) while being orders of magnitude faster in reconstruction and rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06019v2">GaussianSpa: An "Optimizing-Sparsifying" Simplification Framework for Compact and High-Quality 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-15
      | 💬 Project page at https://noodle-lab.github.io/gaussianspa/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a mainstream for novel view synthesis, leveraging continuous aggregations of Gaussian functions to model scene geometry. However, 3DGS suffers from substantial memory requirements to store the multitude of Gaussians, hindering its practicality. To address this challenge, we introduce GaussianSpa, an optimization-based simplification framework for compact and high-quality 3DGS. Specifically, we formulate the simplification as an optimization problem associated with the 3DGS training. Correspondingly, we propose an efficient "optimizing-sparsifying" solution that alternately solves two independent sub-problems, gradually imposing strong sparsity onto the Gaussians in the training process. Our comprehensive evaluations on various datasets show the superiority of GaussianSpa over existing state-of-the-art approaches. Notably, GaussianSpa achieves an average PSNR improvement of 0.9 dB on the real-world Deep Blending dataset with 10$\times$ fewer Gaussians compared to the vanilla 3DGS. Our project page is available at https://noodle-lab.github.io/gaussianspa/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09563v1">Self-Calibrating Gaussian Splatting for Large Field of View Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Project Page: https://denghilbert.github.io/self-cali/
    </div>
    <details class="paper-abstract">
      In this paper, we present a self-calibrating framework that jointly optimizes camera parameters, lens distortion and 3D Gaussian representations, enabling accurate and efficient scene reconstruction. In particular, our technique enables high-quality scene reconstruction from Large field-of-view (FOV) imagery taken with wide-angle lenses, allowing the scene to be modeled from a smaller number of images. Our approach introduces a novel method for modeling complex lens distortions using a hybrid network that combines invertible residual networks with explicit grids. This design effectively regularizes the optimization process, achieving greater accuracy than conventional camera models. Additionally, we propose a cubemap-based resampling strategy to support large FOV images without sacrificing resolution or introducing distortion artifacts. Our method is compatible with the fast rasterization of Gaussian Splatting, adaptable to a wide variety of camera lens distortion, and demonstrates state-of-the-art performance on both synthetic and real-world datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10719v3">4-LEGS: 4D Language Embedded Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Eurographics 2025. Project webpage: https://tau-vailab.github.io/4-LEGS/
    </div>
    <details class="paper-abstract">
      The emergence of neural representations has revolutionized our means for digitally viewing a wide range of 3D scenes, enabling the synthesis of photorealistic images rendered from novel views. Recently, several techniques have been proposed for connecting these low-level representations with the high-level semantics understanding embodied within the scene. These methods elevate the rich semantic understanding from 2D imagery to 3D representations, distilling high-dimensional spatial features onto 3D space. In our work, we are interested in connecting language with a dynamic modeling of the world. We show how to lift spatio-temporal features to a 4D representation based on 3D Gaussian Splatting. This enables an interactive interface where the user can spatiotemporally localize events in the video from text prompts. We demonstrate our system on public 3D video datasets of people and animals performing various actions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01404v2">Gaussian-Det: Learning Closed-Surface Gaussians for 3D Object Detection</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Skins wrapping around our bodies, leathers covering over the sofa, sheet metal coating the car - it suggests that objects are enclosed by a series of continuous surfaces, which provides us with informative geometry prior for objectness deduction. In this paper, we propose Gaussian-Det which leverages Gaussian Splatting as surface representation for multi-view based 3D object detection. Unlike existing monocular or NeRF-based methods which depict the objects via discrete positional data, Gaussian-Det models the objects in a continuous manner by formulating the input Gaussians as feature descriptors on a mass of partial surfaces. Furthermore, to address the numerous outliers inherently introduced by Gaussian splatting, we accordingly devise a Closure Inferring Module (CIM) for the comprehensive surface-based objectness deduction. CIM firstly estimates the probabilistic feature residuals for partial surfaces given the underdetermined nature of Gaussian Splatting, which are then coalesced into a holistic representation on the overall surface closure of the object proposal. In this way, the surface information Gaussian-Det exploits serves as the prior on the quality and reliability of objectness and the information basis of proposal refinement. Experiments on both synthetic and real-world datasets demonstrate that Gaussian-Det outperforms various existing approaches, in terms of both average precision and recall.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09111v1">DenseSplat: Densifying Gaussian Splatting SLAM with Neural Radiance Prior</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Gaussian SLAM systems excel in real-time rendering and fine-grained reconstruction compared to NeRF-based systems. However, their reliance on extensive keyframes is impractical for deployment in real-world robotic systems, which typically operate under sparse-view conditions that can result in substantial holes in the map. To address these challenges, we introduce DenseSplat, the first SLAM system that effectively combines the advantages of NeRF and 3DGS. DenseSplat utilizes sparse keyframes and NeRF priors for initializing primitives that densely populate maps and seamlessly fill gaps. It also implements geometry-aware primitive sampling and pruning strategies to manage granularity and enhance rendering efficiency. Moreover, DenseSplat integrates loop closure and bundle adjustment, significantly enhancing frame-to-frame tracking accuracy. Extensive experiments on multiple large-scale datasets demonstrate that DenseSplat achieves superior performance in tracking and mapping compared to current state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09039v1">Large Images are Gaussians: High-Quality Large Image Representation with Levels of 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Accepted by 39th Annual AAAI Conference on Artificial Intelligence (AAAI 2025). 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      While Implicit Neural Representations (INRs) have demonstrated significant success in image representation, they are often hindered by large training memory and slow decoding speed. Recently, Gaussian Splatting (GS) has emerged as a promising solution in 3D reconstruction due to its high-quality novel view synthesis and rapid rendering capabilities, positioning it as a valuable tool for a broad spectrum of applications. In particular, a GS-based representation, 2DGS, has shown potential for image fitting. In our work, we present \textbf{L}arge \textbf{I}mages are \textbf{G}aussians (\textbf{LIG}), which delves deeper into the application of 2DGS for image representations, addressing the challenge of fitting large images with 2DGS in the situation of numerous Gaussian points, through two distinct modifications: 1) we adopt a variant of representation and optimization strategy, facilitating the fitting of a large number of Gaussian points; 2) we propose a Level-of-Gaussian approach for reconstructing both coarse low-frequency initialization and fine high-frequency details. Consequently, we successfully represent large images as Gaussian points and achieve high-quality large image representation, demonstrating its efficacy across various types of large images. Code is available at {\href{https://github.com/HKU-MedAI/LIG}{https://github.com/HKU-MedAI/LIG}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11356v2">RenderWorld: World Model with Self-Supervised 3D Label</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Accepted in 2025 IEEE International Conference on Robotics and Automation (ICRA)
    </div>
    <details class="paper-abstract">
      End-to-end autonomous driving with vision-only is not only more cost-effective compared to LiDAR-vision fusion but also more reliable than traditional methods. To achieve a economical and robust purely visual autonomous driving system, we propose RenderWorld, a vision-only end-to-end autonomous driving framework, which generates 3D occupancy labels using a self-supervised gaussian-based Img2Occ Module, then encodes the labels by AM-VAE, and uses world model for forecasting and planning. RenderWorld employs Gaussian Splatting to represent 3D scenes and render 2D images greatly improves segmentation accuracy and reduces GPU memory consumption compared with NeRF-based methods. By applying AM-VAE to encode air and non-air separately, RenderWorld achieves more fine-grained scene element representation, leading to state-of-the-art performance in both 4D occupancy forecasting and motion planning from autoregressive world model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10475v1">X-SG$^2$S: Safe and Generalizable Gaussian Splatting with X-dimensional Watermarks</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has been widely used in 3D reconstruction and 3D generation. Training to get a 3DGS scene often takes a lot of time and resources and even valuable inspiration. The increasing amount of 3DGS digital asset have brought great challenges to the copyright protection. However, it still lacks profound exploration targeted at 3DGS. In this paper, we propose a new framework X-SG$^2$S which can simultaneously watermark 1 to 3D messages while keeping the original 3DGS scene almost unchanged. Generally, we have a X-SG$^2$S injector for adding multi-modal messages simultaneously and an extractor for extract them. Specifically, we first split the watermarks into message patches in a fixed manner and sort the 3DGS points. A self-adaption gate is used to pick out suitable location for watermarking. Then use a XD(multi-dimension)-injection heads to add multi-modal messages into sorted 3DGS points. A learnable gate can recognize the location with extra messages and XD-extraction heads can restore hidden messages from the location recommended by the learnable gate. Extensive experiments demonstrated that the proposed X-SG$^2$S can effectively conceal multi modal messages without changing pretrained 3DGS pipeline or the original form of 3DGS parameters. Meanwhile, with simple and efficient model structure and high practicality, X-SG$^2$S still shows good performance in hiding and extracting multi-modal inner structured or unstructured messages. X-SG$^2$S is the first to unify 1 to 3D watermarking model for 3DGS and the first framework to add multi-modal watermarks simultaneous in one 3DGS which pave the wave for later researches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2404.09591v3">3D Gaussian Splatting as Markov Chain Monte Carlo</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting has recently become popular for neural rendering, current methods rely on carefully engineered cloning and splitting strategies for placing Gaussians, which can lead to poor-quality renderings, and reliance on a good initialization. In this work, we rethink the set of 3D Gaussians as a random sample drawn from an underlying probability distribution describing the physical representation of the scene-in other words, Markov Chain Monte Carlo (MCMC) samples. Under this view, we show that the 3D Gaussian updates can be converted as Stochastic Gradient Langevin Dynamics (SGLD) updates by simply introducing noise. We then rewrite the densification and pruning strategies in 3D Gaussian Splatting as simply a deterministic state transition of MCMC samples, removing these heuristics from the framework. To do so, we revise the 'cloning' of Gaussians into a relocalization scheme that approximately preserves sample probability. To encourage efficient use of Gaussians, we introduce a regularizer that promotes the removal of unused Gaussians. On various standard evaluation scenes, we show that our method provides improved rendering quality, easy control over the number of Gaussians, and robustness to initialization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01521v2">MiraGe: Editable 2D Images using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Implicit Neural Representations (INRs) approximate discrete data through continuous functions and are commonly used for encoding 2D images. Traditional image-based INRs employ neural networks to map pixel coordinates to RGB values, capturing shapes, colors, and textures within the network's weights. Recently, GaussianImage has been proposed as an alternative, using Gaussian functions instead of neural networks to achieve comparable quality and compression. Such a solution obtains a quality and compression ratio similar to classical INR models but does not allow image modification. In contrast, our work introduces a novel method, MiraGe, which uses mirror reflections to perceive 2D images in 3D space and employs flat-controlled Gaussians for precise 2D image editing. Our approach improves the rendering quality and allows realistic image modifications, including human-inspired perception of photos in the 3D world. Thanks to modeling images in 3D space, we obtain the illusion of 3D-based modification in 2D images. We also show that our Gaussian representation can be easily combined with a physics engine to produce physics-based modification of 2D images. Consequently, MiraGe allows for better quality than the standard approach and natural modification of 2D images
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09591v3">3D Gaussian Splatting as Markov Chain Monte Carlo</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting has recently become popular for neural rendering, current methods rely on carefully engineered cloning and splitting strategies for placing Gaussians, which can lead to poor-quality renderings, and reliance on a good initialization. In this work, we rethink the set of 3D Gaussians as a random sample drawn from an underlying probability distribution describing the physical representation of the scene-in other words, Markov Chain Monte Carlo (MCMC) samples. Under this view, we show that the 3D Gaussian updates can be converted as Stochastic Gradient Langevin Dynamics (SGLD) updates by simply introducing noise. We then rewrite the densification and pruning strategies in 3D Gaussian Splatting as simply a deterministic state transition of MCMC samples, removing these heuristics from the framework. To do so, we revise the 'cloning' of Gaussians into a relocalization scheme that approximately preserves sample probability. To encourage efficient use of Gaussians, we introduce a regularizer that promotes the removal of unused Gaussians. On various standard evaluation scenes, we show that our method provides improved rendering quality, easy control over the number of Gaussians, and robustness to initialization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18862v3">WeatherGS: 3D Scene Reconstruction in Adverse Weather Conditions via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has gained significant attention for 3D scene reconstruction, but still suffers from complex outdoor environments, especially under adverse weather. This is because 3DGS treats the artifacts caused by adverse weather as part of the scene and will directly reconstruct them, largely reducing the clarity of the reconstructed scene. To address this challenge, we propose WeatherGS, a 3DGS-based framework for reconstructing clear scenes from multi-view images under different weather conditions. Specifically, we explicitly categorize the multi-weather artifacts into the dense particles and lens occlusions that have very different characters, in which the former are caused by snowflakes and raindrops in the air, and the latter are raised by the precipitation on the camera lens. In light of this, we propose a dense-to-sparse preprocess strategy, which sequentially removes the dense particles by an Atmospheric Effect Filter (AEF) and then extracts the relatively sparse occlusion masks with a Lens Effect Detector (LED). Finally, we train a set of 3D Gaussians by the processed images and generated masks for excluding occluded areas, and accurately recover the underlying clear scene by Gaussian splatting. We conduct a diverse and challenging benchmark to facilitate the evaluation of 3D reconstruction under complex weather scenarios. Extensive experiments on this benchmark demonstrate that our WeatherGS consistently produces high-quality, clean scenes across various weather scenarios, outperforming existing state-of-the-art methods. See project page:https://jumponthemoon.github.io/weather-gs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08085v1">Interactive Holographic Visualization for 3D Facial Avatar</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Traditional methods for visualizing dynamic human expressions, particularly in medical training, often rely on flat-screen displays or static mannequins, which have proven inefficient for realistic simulation. In response, we propose a platform that leverages a 3D interactive facial avatar capable of displaying non-verbal feedback, including pain signals. This avatar is projected onto a stereoscopic, view-dependent 3D display, offering a more immersive and realistic simulated patient experience for pain assessment practice. However, there is no existing solution that dynamically predicts and projects interactive 3D facial avatars in real-time. To overcome this, we emphasize the need for a 3D display projection system that can project the facial avatar holographically, allowing users to interact with the avatar from any viewpoint. By incorporating 3D Gaussian Splatting (3DGS) and real-time view-dependent calibration, we significantly improve the training environment for accurate pain recognition and assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2404.09591v3">3D Gaussian Splatting as Markov Chain Monte Carlo</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting has recently become popular for neural rendering, current methods rely on carefully engineered cloning and splitting strategies for placing Gaussians, which can lead to poor-quality renderings, and reliance on a good initialization. In this work, we rethink the set of 3D Gaussians as a random sample drawn from an underlying probability distribution describing the physical representation of the scene-in other words, Markov Chain Monte Carlo (MCMC) samples. Under this view, we show that the 3D Gaussian updates can be converted as Stochastic Gradient Langevin Dynamics (SGLD) updates by simply introducing noise. We then rewrite the densification and pruning strategies in 3D Gaussian Splatting as simply a deterministic state transition of MCMC samples, removing these heuristics from the framework. To do so, we revise the 'cloning' of Gaussians into a relocalization scheme that approximately preserves sample probability. To encourage efficient use of Gaussians, we introduce a regularizer that promotes the removal of unused Gaussians. On various standard evaluation scenes, we show that our method provides improved rendering quality, easy control over the number of Gaussians, and robustness to initialization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01846v2">UVGS: Reimagining Unstructured 3D Gaussian Splatting using UV Mapping</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 https://aashishrai3799.github.io/uvgs
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated superior quality in modeling 3D objects and scenes. However, generating 3DGS remains challenging due to their discrete, unstructured, and permutation-invariant nature. In this work, we present a simple yet effective method to overcome these challenges. We utilize spherical mapping to transform 3DGS into a structured 2D representation, termed UVGS. UVGS can be viewed as multi-channel images, with feature dimensions as a concatenation of Gaussian attributes such as position, scale, color, opacity, and rotation. We further find that these heterogeneous features can be compressed into a lower-dimensional (e.g., 3-channel) shared feature space using a carefully designed multi-branch network. The compressed UVGS can be treated as typical RGB images. Remarkably, we discover that typical VAEs trained with latent diffusion models can directly generalize to this new representation without additional training. Our novel representation makes it effortless to leverage foundational 2D models, such as diffusion models, to directly model 3DGS. Additionally, one can simply increase the 2D UV resolution to accommodate more Gaussians, making UVGS a scalable solution compared to typical 3D backbones. This approach immediately unlocks various novel generation applications of 3DGS by inherently utilizing the already developed superior 2D generation capabilities. In our experiments, we demonstrate various unconditional, conditional generation, and inpainting applications of 3DGS based on diffusion models, which were previously non-trivial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07754v1">MeshSplats: Mesh-Based Rendering with Gaussian Splatting Initialization</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) is a recent and pivotal technique in 3D computer graphics. GS-based algorithms almost always bypass classical methods such as ray tracing, which offers numerous inherent advantages for rendering. For example, ray tracing is able to handle incoherent rays for advanced lighting effects, including shadows and reflections. To address this limitation, we introduce MeshSplats, a method which converts GS to a mesh-like format. Following the completion of training, MeshSplats transforms Gaussian elements into mesh faces, enabling rendering using ray tracing methods with all their associated benefits. Our model can be utilized immediately following transformation, yielding a mesh of slightly reduced quality without additional training. Furthermore, we can enhance the reconstruction quality through the application of a dedicated optimization algorithm that operates on mesh faces rather than Gaussian components. The efficacy of our method is substantiated by experimental results, underscoring its extensive applications in computer graphics and image processing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07615v1">Flow Distillation Sampling: Regularizing 3D Gaussians with Pre-trained Matching Priors</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 Accepted by ICLR 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has achieved excellent rendering quality with fast training and rendering speed. However, its optimization process lacks explicit geometric constraints, leading to suboptimal geometric reconstruction in regions with sparse or no observational input views. In this work, we try to mitigate the issue by incorporating a pre-trained matching prior to the 3DGS optimization process. We introduce Flow Distillation Sampling (FDS), a technique that leverages pre-trained geometric knowledge to bolster the accuracy of the Gaussian radiance field. Our method employs a strategic sampling technique to target unobserved views adjacent to the input views, utilizing the optical flow calculated from the matching model (Prior Flow) to guide the flow analytically calculated from the 3DGS geometry (Radiance Flow). Comprehensive experiments in depth rendering, mesh reconstruction, and novel view synthesis showcase the significant advantages of FDS over state-of-the-art methods. Additionally, our interpretive experiments and analysis aim to shed light on the effects of FDS on geometric accuracy and rendering quality, potentially providing readers with insights into its performance. Project page: https://nju-3dv.github.io/projects/fds
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12886v2">EdgeGaussians -- 3D Edge Mapping via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 To appear in the proceedings of WACV 2025
    </div>
    <details class="paper-abstract">
      With their meaningful geometry and their omnipresence in the 3D world, edges are extremely useful primitives in computer vision. 3D edges comprise of lines and curves, and methods to reconstruct them use either multi-view images or point clouds as input. State-of-the-art image-based methods first learn a 3D edge point cloud then fit 3D edges to it. The edge point cloud is obtained by learning a 3D neural implicit edge field from which the 3D edge points are sampled on a specific level set (0 or 1). However, such methods present two important drawbacks: i) it is not realistic to sample points on exact level sets due to float imprecision and training inaccuracies. Instead, they are sampled within a range of levels so the points do not lie accurately on the 3D edges and require further processing. ii) Such implicit representations are computationally expensive and require long training times. In this paper, we address these two limitations and propose a 3D edge mapping that is simpler, more efficient, and preserves accuracy. Our method learns explicitly the 3D edge points and their edge direction hence bypassing the need for point sampling. It casts a 3D edge point as the center of a 3D Gaussian and the edge direction as the principal axis of the Gaussian. Such a representation has the advantage of being not only geometrically meaningful but also compatible with the efficient training optimization defined in Gaussian Splatting. Results show that the proposed method produces edges as accurate and complete as the state-of-the-art while being an order of magnitude faster. Code is released at https://github.com/kunalchelani/EdgeGaussians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12255v4">HAC++: Towards 100X Compression of 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 Project Page: https://yihangchen-ee.github.io/project_hac++/ Code: https://github.com/YihangChen-ee/HAC-plus. This paper is a journal extension of HAC at arXiv:2403.14530 (ECCV 2024)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a promising framework for novel view synthesis, boasting rapid rendering speed with high fidelity. However, the substantial Gaussians and their associated attributes necessitate effective compression techniques. Nevertheless, the sparse and unorganized nature of the point cloud of Gaussians (or anchors in our paper) presents challenges for compression. To achieve a compact size, we propose HAC++, which leverages the relationships between unorganized anchors and a structured hash grid, utilizing their mutual information for context modeling. Additionally, HAC++ captures intra-anchor contextual relationships to further enhance compression performance. To facilitate entropy coding, we utilize Gaussian distributions to precisely estimate the probability of each quantized attribute, where an adaptive quantization module is proposed to enable high-precision quantization of these attributes for improved fidelity restoration. Moreover, we incorporate an adaptive masking strategy to eliminate invalid Gaussians and anchors. Overall, HAC++ achieves a remarkable size reduction of over 100X compared to vanilla 3DGS when averaged on all datasets, while simultaneously improving fidelity. It also delivers more than 20X size reduction compared to Scaffold-GS. Our code is available at https://github.com/YihangChen-ee/HAC-plus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04843v2">PoI: Pixel of Interest for Novel View Synthesis Assisted Scene Coordinate Regression</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      The task of estimating camera poses can be enhanced through novel view synthesis techniques such as NeRF and Gaussian Splatting to increase the diversity and extension of training data. However, these techniques often produce rendered images with issues like blurring and ghosting, which compromise their reliability. These issues become particularly pronounced for Scene Coordinate Regression (SCR) methods, which estimate 3D coordinates at the pixel level. To mitigate the problems associated with unreliable rendered images, we introduce a novel filtering approach, which selectively extracts well-rendered pixels while discarding the inferior ones. This filter simultaneously measures the SCR model's real-time reprojection loss and gradient during training. Building on this filtering technique, we also develop a new strategy to improve scene coordinate regression using sparse inputs, drawing on successful applications of sparse input techniques in novel view synthesis. Our experimental results validate the effectiveness of our method, demonstrating state-of-the-art performance on indoor and outdoor datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08508v3">BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      We present billboard Splatting (BBSplat) - a novel approach for 3D scene representation based on textured geometric primitives. BBSplat represents the scene as a set of optimizable textured planar primitives with learnable RGB textures and alpha-maps to control their shape. BBSplat primitives can be used in any Gaussian Splatting pipeline as drop-in replacements for Gaussians. The proposed primitives close the rendering quality gap between 2D and 3D Gaussian Splatting (GS), preserving the accurate mesh extraction ability of 2D primitives. Our novel regularization term encourages textures to have a sparser structure, unlocking an efficient compression that leads to a reduction in the storage space of the model. Our experiments show the efficiency of BBSplat on standard datasets of real indoor and outdoor scenes such as Tanks&Temples, DTU, and Mip-NeRF-360.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11394v3">DreamCatalyst: Fast and High-Quality 3D Editing via Controlling Editability and Identity Preservation</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Score distillation sampling (SDS) has emerged as an effective framework in text-driven 3D editing tasks, leveraging diffusion models for 3D-consistent editing. However, existing SDS-based 3D editing methods suffer from long training times and produce low-quality results. We identify that the root cause of this performance degradation is \textit{their conflict with the sampling dynamics of diffusion models}. Addressing this conflict allows us to treat SDS as a diffusion reverse process for 3D editing via sampling from data space. In contrast, existing methods naively distill the score function using diffusion models. From these insights, we propose DreamCatalyst, a novel framework that considers these sampling dynamics in the SDS framework. Specifically, we devise the optimization process of our DreamCatalyst to approximate the diffusion reverse process in editing tasks, thereby aligning with diffusion sampling dynamics. As a result, DreamCatalyst successfully reduces training time and improves editing quality. Our method offers two modes: (1) a fast mode that edits Neural Radiance Fields (NeRF) scenes approximately 23 times faster than current state-of-the-art NeRF editing methods, and (2) a high-quality mode that produces superior results about 8 times faster than these methods. Notably, our high-quality mode outperforms current state-of-the-art NeRF editing methods in terms of both speed and quality. DreamCatalyst also surpasses the state-of-the-art 3D Gaussian Splatting (3DGS) editing methods, establishing itself as an effective and model-agnostic 3D editing solution. See more extensive results on our project page: https://dream-catalyst.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05769v2">Digital Twin Buildings: 3D Modeling, GIS Integration, and Visual Descriptions Using Gaussian Splatting, ChatGPT/Deepseek, and Google Maps Platform</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 -Fixed minor typo
    </div>
    <details class="paper-abstract">
      Urban digital twins are virtual replicas of cities that use multi-source data and data analytics to optimize urban planning, infrastructure management, and decision-making. Towards this, we propose a framework focused on the single-building scale. By connecting to cloud mapping platforms such as Google Map Platforms APIs, by leveraging state-of-the-art multi-agent Large Language Models data analysis using ChatGPT(4o) and Deepseek-V3/R1, and by using our Gaussian Splatting-based mesh extraction pipeline, our Digital Twin Buildings framework can retrieve a building's 3D model, visual descriptions, and achieve cloud-based mapping integration with large language model-based data analytics using a building's address, postal code, or geographic coordinates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07840v1">TranSplat: Surface Embedding-guided 3D Gaussian Splatting for Transparent Object Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 7 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Transparent object manipulation remains a sig- nificant challenge in robotics due to the difficulty of acquiring accurate and dense depth measurements. Conventional depth sensors often fail with transparent objects, resulting in in- complete or erroneous depth data. Existing depth completion methods struggle with interframe consistency and incorrectly model transparent objects as Lambertian surfaces, leading to poor depth reconstruction. To address these challenges, we propose TranSplat, a surface embedding-guided 3D Gaussian Splatting method tailored for transparent objects. TranSplat uses a latent diffusion model to generate surface embeddings that provide consistent and continuous representations, making it robust to changes in viewpoint and lighting. By integrating these surface embeddings with input RGB images, TranSplat effectively captures the complexities of transparent surfaces, enhancing the splatting of 3D Gaussians and improving depth completion. Evaluations on synthetic and real-world transpar- ent object benchmarks, as well as robot grasping tasks, show that TranSplat achieves accurate and dense depth completion, demonstrating its effectiveness in practical applications. We open-source synthetic dataset and model: https://github. com/jeongyun0609/TranSplat
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2411.18311v2">Neural Surface Priors for Editable Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 9 pages, 7 figures
    </div>
    <details class="paper-abstract">
      In computer graphics and vision, recovering easily modifiable scene appearance from image data is crucial for applications such as content creation. We introduce a novel method that integrates 3D Gaussian Splatting with an implicit surface representation, enabling intuitive editing of recovered scenes through mesh manipulation. Starting with a set of input images and camera poses, our approach reconstructs the scene surface using a neural signed distance field. This neural surface acts as a geometric prior guiding the training of Gaussian Splatting components, ensuring their alignment with the scene geometry. To facilitate editing, we encode the visual and geometric information into a lightweight triangle soup proxy. Edits applied to the mesh extracted from the neural surface propagate seamlessly through this intermediate structure to update the recovered appearance. Unlike previous methods relying on the triangle soup proxy representation, our approach supports a wider range of modifications and fully leverages the mesh topology, enabling a more flexible and intuitive editing process. The complete source code for this project can be accessed at: https://github.com/WJakubowska/NeuralSurfacePriors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06519v1">SIREN: Semantic, Initialization-Free Registration of Multi-Robot Gaussian Splatting Maps</a></div>
    <div class="paper-meta">
      📅 2025-02-10
    </div>
    <details class="paper-abstract">
      We present SIREN for registration of multi-robot Gaussian Splatting (GSplat) maps, with zero access to camera poses, images, and inter-map transforms for initialization or fusion of local submaps. To realize these capabilities, SIREN harnesses the versatility and robustness of semantics in three critical ways to derive a rigorous registration pipeline for multi-robot GSplat maps. First, SIREN utilizes semantics to identify feature-rich regions of the local maps where the registration problem is better posed, eliminating the need for any initialization which is generally required in prior work. Second, SIREN identifies candidate correspondences between Gaussians in the local maps using robust semantic features, constituting the foundation for robust geometric optimization, coarsely aligning 3D Gaussian primitives extracted from the local maps. Third, this key step enables subsequent photometric refinement of the transformation between the submaps, where SIREN leverages novel-view synthesis in GSplat maps along with a semantics-based image filter to compute a high-accuracy non-rigid transformation for the generation of a high-fidelity fused map. We demonstrate the superior performance of SIREN compared to competing baselines across a range of real-world datasets, and in particular, across the most widely-used robot hardware platforms, including a manipulator, drone, and quadruped. In our experiments, SIREN achieves about 90x smaller rotation errors, 300x smaller translation errors, and 44x smaller scale errors in the most challenging scenes, where competing methods struggle. We will release the code and provide a link to the project page after the review process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01895v2">EnerVerse: Envisioning Embodied Future Space for Robotics Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-02-10
      | 💬 Website: https://sites.google.com/view/enerverse
    </div>
    <details class="paper-abstract">
      We introduce EnerVerse, a generative robotics foundation model that constructs and interprets embodied spaces. EnerVerse employs an autoregressive video diffusion framework to predict future embodied spaces from instructions, enhanced by a sparse context memory for long-term reasoning. To model the 3D robotics world, we propose Free Anchor Views (FAVs), a multi-view video representation offering flexible, task-adaptive perspectives to address challenges like motion ambiguity and environmental constraints. Additionally, we present EnerVerse-D, a data engine pipeline combining the generative model with 4D Gaussian Splatting, forming a self-reinforcing data loop to reduce the sim-to-real gap. Leveraging these innovations, EnerVerse translates 4D world representations into physical actions via a policy head (EnerVerse-A), enabling robots to execute task instructions. EnerVerse-A achieves state-of-the-art performance in both simulation and real-world settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.14823v2">LapisGS: Layered Progressive 3D Gaussian Splatting for Adaptive Streaming</a></div>
    <div class="paper-meta">
      📅 2025-02-10
      | 💬 3DV 2025; Project Page: https://yuang-ian.github.io/lapisgs/ ; Code: https://github.com/nus-vv-streams/lapis-gs
    </div>
    <details class="paper-abstract">
      The rise of Extended Reality (XR) requires efficient streaming of 3D online worlds, challenging current 3DGS representations to adapt to bandwidth-constrained environments. This paper proposes LapisGS, a layered 3DGS that supports adaptive streaming and progressive rendering. Our method constructs a layered structure for cumulative representation, incorporates dynamic opacity optimization to maintain visual fidelity, and utilizes occupancy maps to efficiently manage Gaussian splats. This proposed model offers a progressive representation supporting a continuous rendering quality adapted for bandwidth-aware streaming. Extensive experiments validate the effectiveness of our approach in balancing visual fidelity with the compactness of the model, with up to 50.71% improvement in SSIM, 286.53% improvement in LPIPS with 23% of the original model size, and shows its potential for bandwidth-adapted 3D streaming and rendering applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07007v1">Grounding Creativity in Physics: A Brief Survey of Physical Priors in AIGC</a></div>
    <div class="paper-meta">
      📅 2025-02-10
    </div>
    <details class="paper-abstract">
      Recent advancements in AI-generated content have significantly improved the realism of 3D and 4D generation. However, most existing methods prioritize appearance consistency while neglecting underlying physical principles, leading to artifacts such as unrealistic deformations, unstable dynamics, and implausible objects interactions. Incorporating physics priors into generative models has become a crucial research direction to enhance structural integrity and motion realism. This survey provides a review of physics-aware generative methods, systematically analyzing how physical constraints are integrated into 3D and 4D generation. First, we examine recent works in incorporating physical priors into static and dynamic 3D generation, categorizing methods based on representation types, including vision-based, NeRF-based, and Gaussian Splatting-based approaches. Second, we explore emerging techniques in 4D generation, focusing on methods that model temporal dynamics with physical simulations. Finally, we conduct a comparative analysis of major methods, highlighting their strengths, limitations, and suitability for different materials and motion dynamics. By presenting an in-depth analysis of physics-grounded AIGC, this survey aims to bridge the gap between generative models and physical realism, providing insights that inspire future research in physically consistent content generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09981v3">Controllable Text-to-3D Generation via Surface-Aligned Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-09
      | 💬 3DV-2025
    </div>
    <details class="paper-abstract">
      While text-to-3D and image-to-3D generation tasks have received considerable attention, one important but under-explored field between them is controllable text-to-3D generation, which we mainly focus on in this work. To address this task, 1) we introduce Multi-view ControlNet (MVControl), a novel neural network architecture designed to enhance existing pre-trained multi-view diffusion models by integrating additional input conditions, such as edge, depth, normal, and scribble maps. Our innovation lies in the introduction of a conditioning module that controls the base diffusion model using both local and global embeddings, which are computed from the input condition images and camera poses. Once trained, MVControl is able to offer 3D diffusion guidance for optimization-based 3D generation. And, 2) we propose an efficient multi-stage 3D generation pipeline that leverages the benefits of recent large reconstruction models and score distillation algorithm. Building upon our MVControl architecture, we employ a unique hybrid diffusion guidance method to direct the optimization process. In pursuit of efficiency, we adopt 3D Gaussians as our representation instead of the commonly used implicit representations. We also pioneer the use of SuGaR, a hybrid representation that binds Gaussians to mesh triangle faces. This approach alleviates the issue of poor geometry in 3D Gaussians and enables the direct sculpting of fine-grained geometry on the mesh. Extensive experiments demonstrate that our method achieves robust generalization and enables the controllable generation of high-quality 3D content. Project page: https://lizhiqi49.github.io/MVControl/.
    </details>
</div>
