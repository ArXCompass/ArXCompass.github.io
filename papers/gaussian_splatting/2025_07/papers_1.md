# gaussian splatting - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23785v1">Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 ICCV 2025. Project page: https://gvfdiffusion.github.io/
    </div>
    <details class="paper-abstract">
      In this paper, we present a novel framework for video-to-4D generation that creates high-quality dynamic 3D content from single video inputs. Direct 4D diffusion modeling is extremely challenging due to costly data construction and the high-dimensional nature of jointly representing 3D shape, appearance, and motion. We address these challenges by introducing a Direct 4DMesh-to-GS Variation Field VAE that directly encodes canonical Gaussian Splats (GS) and their temporal variations from 3D animation data without per-instance fitting, and compresses high-dimensional animations into a compact latent space. Building upon this efficient representation, we train a Gaussian Variation Field diffusion model with temporal-aware Diffusion Transformer conditioned on input videos and canonical GS. Trained on carefully-curated animatable 3D objects from the Objaverse dataset, our model demonstrates superior generation quality compared to existing methods. It also exhibits remarkable generalization to in-the-wild video inputs despite being trained exclusively on synthetic data, paving the way for generating high-quality animated 3D content. Project page: https://gvfdiffusion.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23772v1">SeqAffordSplat: Scene-level Sequential Affordance Reasoning on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      3D affordance reasoning, the task of associating human instructions with the functional regions of 3D objects, is a critical capability for embodied agents. Current methods based on 3D Gaussian Splatting (3DGS) are fundamentally limited to single-object, single-step interactions, a paradigm that falls short of addressing the long-horizon, multi-object tasks required for complex real-world applications. To bridge this gap, we introduce the novel task of Sequential 3D Gaussian Affordance Reasoning and establish SeqAffordSplat, a large-scale benchmark featuring 1800+ scenes to support research on long-horizon affordance understanding in complex 3DGS environments. We then propose SeqSplatNet, an end-to-end framework that directly maps an instruction to a sequence of 3D affordance masks. SeqSplatNet employs a large language model that autoregressively generates text interleaved with special segmentation tokens, guiding a conditional decoder to produce the corresponding 3D mask. To handle complex scene geometry, we introduce a pre-training strategy, Conditional Geometric Reconstruction, where the model learns to reconstruct complete affordance region masks from known geometric observations, thereby building a robust geometric prior. Furthermore, to resolve semantic ambiguities, we design a feature injection mechanism that lifts rich semantic features from 2D Vision Foundation Models (VFM) and fuses them into the 3D decoder at multiple scales. Extensive experiments demonstrate that our method sets a new state-of-the-art on our challenging benchmark, effectively advancing affordance reasoning from single-step interactions to complex, sequential tasks at the scene level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23704v1">Enhanced Velocity Field Modeling for Gaussian Video Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 17 pages, 8 figures
    </div>
    <details class="paper-abstract">
      High-fidelity 3D video reconstruction is essential for enabling real-time rendering of dynamic scenes with realistic motion in virtual and augmented reality (VR/AR). The deformation field paradigm of 3D Gaussian splatting has achieved near-photorealistic results in video reconstruction due to the great representation capability of deep deformation networks. However, in videos with complex motion and significant scale variations, deformation networks often overfit to irregular Gaussian trajectories, leading to suboptimal visual quality. Moreover, the gradient-based densification strategy designed for static scene reconstruction proves inadequate to address the absence of dynamic content. In light of these challenges, we propose a flow-empowered velocity field modeling scheme tailored for Gaussian video reconstruction, dubbed FlowGaussian-VR. It consists of two core components: a velocity field rendering (VFR) pipeline which enables optical flow-based optimization, and a flow-assisted adaptive densification (FAD) strategy that adjusts the number and size of Gaussians in dynamic regions. We validate our model's effectiveness on multi-view dynamic reconstruction and novel view synthesis with multiple real-world datasets containing challenging motion scenarios, demonstrating not only notable visual improvements (over 2.5 dB gain in PSNR) and less blurry artifacts in dynamic textures, but also regularized and trackable per-Gaussian trajectories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23683v1">I2V-GS: Infrastructure-to-Vehicle View Transformation with Gaussian Splatting for Autonomous Driving Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Vast and high-quality data are essential for end-to-end autonomous driving systems. However, current driving data is mainly collected by vehicles, which is expensive and inefficient. A potential solution lies in synthesizing data from real-world images. Recent advancements in 3D reconstruction demonstrate photorealistic novel view synthesis, highlighting the potential of generating driving data from images captured on the road. This paper introduces a novel method, I2V-GS, to transfer the Infrastructure view To the Vehicle view with Gaussian Splatting. Reconstruction from sparse infrastructure viewpoints and rendering under large view transformations is a challenging problem. We adopt the adaptive depth warp to generate dense training views. To further expand the range of views, we employ a cascade strategy to inpaint warped images, which also ensures inpainting content is consistent across views. To further ensure the reliability of the diffusion model, we utilize the cross-view information to perform a confidenceguided optimization. Moreover, we introduce RoadSight, a multi-modality, multi-view dataset from real scenarios in infrastructure views. To our knowledge, I2V-GS is the first framework to generate autonomous driving datasets with infrastructure-vehicle view transformation. Experimental results demonstrate that I2V-GS significantly improves synthesis quality under vehicle view, outperforming StreetGaussian in NTA-Iou, NTL-Iou, and FID by 45.7%, 34.2%, and 14.9%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23677v1">Stereo 3D Gaussian Splatting SLAM for Outdoor Urban Scenes</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently gained popularity in SLAM applications due to its fast rendering and high-fidelity representation. However, existing 3DGS-SLAM systems have predominantly focused on indoor environments and relied on active depth sensors, leaving a gap for large-scale outdoor applications. We present BGS-SLAM, the first binocular 3D Gaussian Splatting SLAM system designed for outdoor scenarios. Our approach uses only RGB stereo pairs without requiring LiDAR or active sensors. BGS-SLAM leverages depth estimates from pre-trained deep stereo networks to guide 3D Gaussian optimization with a multi-loss strategy enhancing both geometric consistency and visual quality. Experiments on multiple datasets demonstrate that BGS-SLAM achieves superior tracking accuracy and mapping performance compared to other 3DGS-based solutions in complex outdoor environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23569v1">Gaussian Splatting Feature Fields for Privacy-Preserving Visual Localization</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      Visual localization is the task of estimating a camera pose in a known environment. In this paper, we utilize 3D Gaussian Splatting (3DGS)-based representations for accurate and privacy-preserving visual localization. We propose Gaussian Splatting Feature Fields (GSFFs), a scene representation for visual localization that combines an explicit geometry model (3DGS) with an implicit feature field. We leverage the dense geometric information and differentiable rasterization algorithm from 3DGS to learn robust feature representations grounded in 3D. In particular, we align a 3D scale-aware feature field and a 2D feature encoder in a common embedding space through a contrastive framework. Using a 3D structure-informed clustering procedure, we further regularize the representation learning and seamlessly convert the features to segmentations, which can be used for privacy-preserving visual localization. Pose refinement, which involves aligning either feature maps or segmentations from a query image with those rendered from the GSFFs scene representation, is used to achieve localization. The resulting privacy- and non-privacy-preserving localization pipelines, evaluated on multiple real-world datasets, show state-of-the-art performances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18371v2">MVG4D: Image Matrix-Based Multi-View and Motion Generation for 4D Content Creation from a Single Image</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Advances in generative modeling have significantly enhanced digital content creation, extending from 2D images to complex 3D and 4D scenes. Despite substantial progress, producing high-fidelity and temporally consistent dynamic 4D content remains a challenge. In this paper, we propose MVG4D, a novel framework that generates dynamic 4D content from a single still image by combining multi-view synthesis with 4D Gaussian Splatting (4D GS). At its core, MVG4D employs an image matrix module that synthesizes temporally coherent and spatially diverse multi-view images, providing rich supervisory signals for downstream 3D and 4D reconstruction. These multi-view images are used to optimize a 3D Gaussian point cloud, which is further extended into the temporal domain via a lightweight deformation network. Our method effectively enhances temporal consistency, geometric fidelity, and visual realism, addressing key challenges in motion discontinuity and background degradation that affect prior 4D GS-based methods. Extensive experiments on the Objaverse dataset demonstrate that MVG4D outperforms state-of-the-art baselines in CLIP-I, PSNR, FVD, and time efficiency. Notably, it reduces flickering artifacts and sharpens structural details across views and time, enabling more immersive AR/VR experiences. MVG4D sets a new direction for efficient and controllable 4D generation from minimal inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21872v3">MultiEditor: Controllable Multimodal Object Editing for Driving Scenarios Using 3D Gaussian Splatting Priors</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Autonomous driving systems rely heavily on multimodal perception data to understand complex environments. However, the long-tailed distribution of real-world data hinders generalization, especially for rare but safety-critical vehicle categories. To address this challenge, we propose MultiEditor, a dual-branch latent diffusion framework designed to edit images and LiDAR point clouds in driving scenarios jointly. At the core of our approach is introducing 3D Gaussian Splatting (3DGS) as a structural and appearance prior for target objects. Leveraging this prior, we design a multi-level appearance control mechanism--comprising pixel-level pasting, semantic-level guidance, and multi-branch refinement--to achieve high-fidelity reconstruction across modalities. We further propose a depth-guided deformable cross-modality condition module that adaptively enables mutual guidance between modalities using 3DGS-rendered depth, significantly enhancing cross-modality consistency. Extensive experiments demonstrate that MultiEditor achieves superior performance in visual and geometric fidelity, editing controllability, and cross-modality consistency. Furthermore, generating rare-category vehicle data with MultiEditor substantially enhances the detection accuracy of perception models on underrepresented classes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23374v1">NeRF Is a Valuable Assistant for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 Accepted by ICCV
    </div>
    <details class="paper-abstract">
      We introduce NeRF-GS, a novel framework that jointly optimizes Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). This framework leverages the inherent continuous spatial representation of NeRF to mitigate several limitations of 3DGS, including sensitivity to Gaussian initialization, limited spatial awareness, and weak inter-Gaussian correlations, thereby enhancing its performance. In NeRF-GS, we revisit the design of 3DGS and progressively align its spatial features with NeRF, enabling both representations to be optimized within the same scene through shared 3D spatial information. We further address the formal distinctions between the two approaches by optimizing residual vectors for both implicit features and Gaussian positions to enhance the personalized capabilities of 3DGS. Experimental results on benchmark datasets show that NeRF-GS surpasses existing methods and achieves state-of-the-art performance. This outcome confirms that NeRF and 3DGS are complementary rather than competing, offering new insights into hybrid approaches that combine 3DGS and NeRF for efficient 3D scene representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23340v1">MagicRoad: Semantic-Aware 3D Road Surface Reconstruction via Obstacle Inpainting</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      Road surface reconstruction is essential for autonomous driving, supporting centimeter-accurate lane perception and high-definition mapping in complex urban environments.While recent methods based on mesh rendering or 3D Gaussian splatting (3DGS) achieve promising results under clean and static conditions, they remain vulnerable to occlusions from dynamic agents, visual clutter from static obstacles, and appearance degradation caused by lighting and weather changes. We present a robust reconstruction framework that integrates occlusion-aware 2D Gaussian surfels with semantic-guided color enhancement to recover clean, consistent road surfaces. Our method leverages a planar-adapted Gaussian representation for efficient large-scale modeling, employs segmentation-guided video inpainting to remove both dynamic and static foreground objects, and enhances color coherence via semantic-aware correction in HSV space. Extensive experiments on urban-scale datasets demonstrate that our framework produces visually coherent and geometrically faithful reconstructions, significantly outperforming prior methods under real-world conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18041v6">OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 20 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation (VLN) aims to guide agents by leveraging language instructions and visual cues, playing a pivotal role in embodied AI. Indoor VLN has been extensively studied, whereas outdoor aerial VLN remains underexplored. The potential reason is that outdoor aerial view encompasses vast areas, making data collection more challenging, which results in a lack of benchmarks. To address this problem, we propose OpenFly, a platform comprising various rendering engines, a versatile toolchain, and a large-scale benchmark for aerial VLN. Firstly, we integrate diverse rendering engines and advanced techniques for environment simulation, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of our environments. Secondly, we develop a highly automated toolchain for aerial VLN data collection, streamlining point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Thirdly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering diverse heights and lengths across 18 scenes. Moreover, we propose OpenFly-Agent, a keyframe-aware VLN model emphasizing key observations during flight. For benchmarking, extensive experiments and analyses are conducted, evaluating several recent VLN methods and showcasing the superiority of our OpenFly platform and agent. The toolchain, dataset, and codes will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01199v2">LiteGS: A High-performance Framework to Train 3DGS in Subminutes via System and Algorithm Codesign</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as promising alternative in 3D representation. However, it still suffers from high training cost. This paper introduces LiteGS, a high performance framework that systematically optimizes the 3DGS training pipeline from multiple aspects. At the low-level computation layer, we design a ``warp-based raster'' associated with two hardware-aware optimizations to significantly reduce gradient reduction overhead. At the mid-level data management layer, we introduce dynamic spatial sorting based on Morton coding to enable a performant ``Cluster-Cull-Compact'' pipeline and improve data locality, therefore reducing cache misses. At the top-level algorithm layer, we establish a new robust densification criterion based on the variance of the opacity gradient, paired with a more stable opacity control mechanism, to achieve more precise parameter growth. Experimental results demonstrate that LiteGS accelerates the original 3DGS training by up to 13.4x with comparable or superior quality and surpasses the current SOTA in lightweight models by up to 1.4x speedup. For high-quality reconstruction tasks, LiteGS sets a new accuracy record and decreases the training time by an order of magnitude.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23277v1">iLRM: An Iterative Large 3D Reconstruction Model</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 Project page: https://gynjn.github.io/iLRM/
    </div>
    <details class="paper-abstract">
      Feed-forward 3D modeling has emerged as a promising approach for rapid and high-quality 3D reconstruction. In particular, directly generating explicit 3D representations, such as 3D Gaussian splatting, has attracted significant attention due to its fast and high-quality rendering, as well as numerous applications. However, many state-of-the-art methods, primarily based on transformer architectures, suffer from severe scalability issues because they rely on full attention across image tokens from multiple input views, resulting in prohibitive computational costs as the number of views or image resolution increases. Toward a scalable and efficient feed-forward 3D reconstruction, we introduce an iterative Large 3D Reconstruction Model (iLRM) that generates 3D Gaussian representations through an iterative refinement mechanism, guided by three core principles: (1) decoupling the scene representation from input-view images to enable compact 3D representations; (2) decomposing fully-attentional multi-view interactions into a two-stage attention scheme to reduce computational costs; and (3) injecting high-resolution information at every layer to achieve high-fidelity reconstruction. Experimental results on widely used datasets, such as RE10K and DL3DV, demonstrate that iLRM outperforms existing methods in both reconstruction quality and speed. Notably, iLRM exhibits superior scalability, delivering significantly higher reconstruction quality under comparable computational cost by efficiently leveraging a larger number of input views.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23273v1">GSFusion:Globally Optimized LiDAR-Inertial-Visual Mapping for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has revolutionized photorealistic mapping, conventional approaches based on camera sensor, even RGB-D, suffer from fundamental limitations such as high computational load, failure in environments with poor texture or illumination, and short operational ranges. LiDAR emerges as a robust alternative, but its integration with 3DGS introduces new challenges, such as the need for exceptional global alignment for photorealistic quality and prolonged optimization times caused by sparse data. To address these challenges, we propose GSFusion, an online LiDAR-Inertial-Visual mapping system that ensures high-precision map consistency through a surfel-to-surfel constraint in the global pose-graph optimization. To handle sparse data, our system employs a pixel-aware Gaussian initialization strategy for efficient representation and a bounded sigmoid constraint to prevent uncontrolled Gaussian growth. Experiments on public and our datasets demonstrate our system outperforms existing 3DGS SLAM systems in terms of rendering quality and map-building efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02178v4">Sparfels: Fast Reconstruction from Sparse Unposed Imagery</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 ICCV 2025. Project page : https://shubhendu-jena.github.io/Sparfels-web/
    </div>
    <details class="paper-abstract">
      We present a method for Sparse view reconstruction with surface element splatting that runs within 3 minutes on a consumer grade GPU. While few methods address sparse radiance field learning from noisy or unposed sparse cameras, shape recovery remains relatively underexplored in this setting. Several radiance and shape learning test-time optimization methods address the sparse posed setting by learning data priors or using combinations of external monocular geometry priors. Differently, we propose an efficient and simple pipeline harnessing a single recent 3D foundation model. We leverage its various task heads, notably point maps and camera initializations to instantiate a bundle adjusting 2D Gaussian Splatting (2DGS) model, and image correspondences to guide camera optimization midst 2DGS training. Key to our contribution is a novel formulation of splatted color variance along rays, which can be computed efficiently. Reducing this moment in training leads to more accurate shape reconstructions. We demonstrate state-of-the-art performances in the sparse uncalibrated setting in reconstruction and novel view benchmarks based on established multi-view datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2505.08510v2">FOCI: Trajectory Optimization on Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 8 pages, 8 figures, Mario Gomez Andreu and Maximum Wilder-Smith contributed equally
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently gained popularity as a faster alternative to Neural Radiance Fields (NeRFs) in 3D reconstruction and view synthesis methods. Leveraging the spatial information encoded in 3DGS, this work proposes FOCI (Field Overlap Collision Integral), an algorithm that is able to optimize trajectories directly on the Gaussians themselves. FOCI leverages a novel and interpretable collision formulation for 3DGS using the notion of the overlap integral between Gaussians. Contrary to other approaches, which represent the robot with conservative bounding boxes that underestimate the traversability of the environment, we propose to represent the environment and the robot as Gaussian Splats. This not only has desirable computational properties, but also allows for orientation-aware planning, allowing the robot to pass through very tight and narrow spaces. We extensively test our algorithm in both synthetic and real Gaussian Splats, showcasing that collision-free trajectories for the ANYmal legged robot that can be computed in a few seconds, even with hundreds of thousands of Gaussians making up the environment. The project page and code are available at https://rffr.leggedrobotics.com/works/foci/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19451v2">GS-Occ3D: Scaling Vision-only Occupancy Reconstruction for Autonomous Driving with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 ICCV 2025. Project Page: https://gs-occ3d.github.io/
    </div>
    <details class="paper-abstract">
      Occupancy is crucial for autonomous driving, providing essential geometric priors for perception and planning. However, existing methods predominantly rely on LiDAR-based occupancy annotations, which limits scalability and prevents leveraging vast amounts of potential crowdsourced data for auto-labeling. To address this, we propose GS-Occ3D, a scalable vision-only framework that directly reconstructs occupancy. Vision-only occupancy reconstruction poses significant challenges due to sparse viewpoints, dynamic scene elements, severe occlusions, and long-horizon motion. Existing vision-based methods primarily rely on mesh representation, which suffer from incomplete geometry and additional post-processing, limiting scalability. To overcome these issues, GS-Occ3D optimizes an explicit occupancy representation using an Octree-based Gaussian Surfel formulation, ensuring efficiency and scalability. Additionally, we decompose scenes into static background, ground, and dynamic objects, enabling tailored modeling strategies: (1) Ground is explicitly reconstructed as a dominant structural element, significantly improving large-area consistency; (2) Dynamic vehicles are separately modeled to better capture motion-related occupancy patterns. Extensive experiments on the Waymo dataset demonstrate that GS-Occ3D achieves state-of-the-art geometry reconstruction results. By curating vision-only binary occupancy labels from diverse urban scenes, we show their effectiveness for downstream occupancy models on Occ3D-Waymo and superior zero-shot generalization on Occ3D-nuScenes. It highlights the potential of large-scale vision-based occupancy reconstruction as a new paradigm for scalable auto-labeling. Project Page: https://gs-occ3d.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06838v5">Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      Implicit Neural Representations (INR) have been successfully employed for Arbitrary-scale Super-Resolution (ASR). However, INR-based models need to query the multi-layer perceptron module numerous times and render a pixel in each query, resulting in insufficient representation capability and low computational efficiency. Recently, Gaussian Splatting (GS) has shown its advantages over INR in both visual quality and rendering speed in 3D tasks, which motivates us to explore whether GS can be employed for the ASR task. However, directly applying GS to ASR is exceptionally challenging because the original GS is an optimization-based method through overfitting each single scene, while in ASR we aim to learn a single model that can generalize to different images and scaling factors. We overcome these challenges by developing two novel techniques. Firstly, to generalize GS for ASR, we elaborately design an architecture to predict the corresponding image-conditioned Gaussians of the input low-resolution image in a feed-forward manner. Each Gaussian can fit the shape and direction of an area of complex textures, showing powerful representation capability. Secondly, we implement an efficient differentiable 2D GPU/CUDA-based scale-aware rasterization to render super-resolved images by sampling discrete RGB values from the predicted continuous Gaussians. Via end-to-end training, our optimized network, namely GSASR, can perform ASR for any image and unseen scaling factors. Extensive experiments validate the effectiveness of our proposed method. The code and models are available at https://github.com/ChrisDud0257/GSASR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13086v2">Gaussian On-the-Fly Splatting: A Progressive Framework for Robust Near Real-Time 3DGS Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves high-fidelity rendering with fast real-time performance, but existing methods rely on offline training after full Structure-from-Motion (SfM) processing. In contrast, this work introduces Gaussian on-the-fly Splatting (abbreviated as On-the-Fly GS), a progressive framework enabling near real-time 3DGS optimization during image capture. As each image arrives, its pose and sparse points are updated via On-the-Fly SfM, and newly optimized Gaussians are immediately integrated into the 3DGS field. To achieve this, we propose a progressive Local & Semi-Global optimization to prioritize the new image and its neighbors by their corresponding overlapping relationship, allowing the new image and its overlapping images to get more training. To further stabilize training across previous and new images, an adaptive learning rate schedule balances the iterations and the learning rate. Extensive experiments on multiple benchmarks show that our On-the-Fly GS reduces training time significantly, optimizing each new image in seconds with minimal rendering loss, offering one of the first practical steps toward rapid, progressive 3DGS reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21872v2">MultiEditor: Controllable Multimodal Object Editing for Driving Scenarios Using 3D Gaussian Splatting Priors</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Autonomous driving systems rely heavily on multimodal perception data to understand complex environments. However, the long-tailed distribution of real-world data hinders generalization, especially for rare but safety-critical vehicle categories. To address this challenge, we propose MultiEditor, a dual-branch latent diffusion framework designed to edit images and LiDAR point clouds in driving scenarios jointly. At the core of our approach is introducing 3D Gaussian Splatting (3DGS) as a structural and appearance prior for target objects. Leveraging this prior, we design a multi-level appearance control mechanism--comprising pixel-level pasting, semantic-level guidance, and multi-branch refinement--to achieve high-fidelity reconstruction across modalities. We further propose a depth-guided deformable cross-modality condition module that adaptively enables mutual guidance between modalities using 3DGS-rendered depth, significantly enhancing cross-modality consistency. Extensive experiments demonstrate that MultiEditor achieves superior performance in visual and geometric fidelity, editing controllability, and cross-modality consistency. Furthermore, generating rare-category vehicle data with MultiEditor substantially enhances the detection accuracy of perception models on underrepresented classes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21572v2">No Redundancy, No Stall: Lightweight Streaming 3D Gaussian Splatting for Real-time Rendering</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 Accepted by International Conference on Computer-Aided Design (ICCAD) 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables high-quality rendering of 3D scenes and is getting increasing adoption in domains like autonomous driving and embodied intelligence. However, 3DGS still faces major efficiency challenges when faced with high frame rate requirements and resource-constrained edge deployment. To enable efficient 3DGS, in this paper, we propose LS-Gaussian, an algorithm/hardware co-design framework for lightweight streaming 3D rendering. LS-Gaussian is motivated by the core observation that 3DGS suffers from substantial computation redundancy and stalls. On one hand, in practical scenarios, high-frame-rate 3DGS is often applied in settings where a camera observes and renders the same scene continuously but from slightly different viewpoints. Therefore, instead of rendering each frame separately, LS-Gaussian proposes a viewpoint transformation algorithm that leverages inter-frame continuity for efficient sparse rendering. On the other hand, as different tiles within an image are rendered in parallel but have imbalanced workloads, frequent hardware stalls also slow down the rendering process. LS-Gaussian predicts the workload for each tile based on viewpoint transformation to enable more balanced parallel computation and co-designs a customized 3DGS accelerator to support the workload-aware mapping in real-time. Experimental results demonstrate that LS-Gaussian achieves 5.41x speedup over the edge GPU baseline on average and up to 17.3x speedup with the customized accelerator, while incurring only minimal visual quality degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19856v2">RaGS: Unleashing 3D Gaussian Splatting from 4D Radar and Monocular Cues for 3D Object Detection</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 9 pages, 6 figures, conference
    </div>
    <details class="paper-abstract">
      4D millimeter-wave radar has emerged as a promising sensor for autonomous driving, but effective 3D object detection from both 4D radar and monocular images remains a challenge. Existing fusion approaches typically rely on either instance-based proposals or dense BEV grids, which either lack holistic scene understanding or are limited by rigid grid structures. To address these, we propose RaGS, the first framework to leverage 3D Gaussian Splatting (GS) as representation for fusing 4D radar and monocular cues in 3D object detection. 3D GS naturally suits 3D object detection by modeling the scene as a field of Gaussians, dynamically allocating resources on foreground objects and providing a flexible, resource-efficient solution. RaGS uses a cascaded pipeline to construct and refine the Gaussian field. It starts with the Frustum-based Localization Initiation (FLI), which unprojects foreground pixels to initialize coarse 3D Gaussians positions. Then, the Iterative Multimodal Aggregation (IMA) fuses semantics and geometry, refining the limited Gaussians to the regions of interest. Finally, the Multi-level Gaussian Fusion (MGF) renders the Gaussians into multi-level BEV features for 3D object detection. By dynamically focusing on sparse objects within scenes, RaGS enable object concentrating while offering comprehensive scene perception. Extensive experiments on View-of-Delft, TJ4DRadSet, and OmniHD-Scenes benchmarks demonstrate its state-of-the-art performance. Code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14501v2">Advances in Feed-Forward 3D Reconstruction and View Synthesis: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 A project page associated with this survey is available at https://fnzhan.com/projects/Feed-Forward-3D
    </div>
    <details class="paper-abstract">
      3D reconstruction and view synthesis are foundational problems in computer vision, graphics, and immersive technologies such as augmented reality (AR), virtual reality (VR), and digital twins. Traditional methods rely on computationally intensive iterative optimization in a complex chain, limiting their applicability in real-world scenarios. Recent advances in feed-forward approaches, driven by deep learning, have revolutionized this field by enabling fast and generalizable 3D reconstruction and view synthesis. This survey offers a comprehensive review of feed-forward techniques for 3D reconstruction and view synthesis, with a taxonomy according to the underlying representation architectures including point cloud, 3D Gaussian Splatting (3DGS), Neural Radiance Fields (NeRF), etc. We examine key tasks such as pose-free reconstruction, dynamic 3D reconstruction, and 3D-aware image and video synthesis, highlighting their applications in digital humans, SLAM, robotics, and beyond. In addition, we review commonly used datasets with detailed statistics, along with evaluation protocols for various downstream tasks. We conclude by discussing open research challenges and promising directions for future work, emphasizing the potential of feed-forward approaches to advance the state of the art in 3D vision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22342v1">UFV-Splatter: Pose-Free Feed-Forward 3D Gaussian Splatting Adapted to Unfavorable Views</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 Project page: https://yfujimura.github.io/UFV-Splatter_page/
    </div>
    <details class="paper-abstract">
      This paper presents a pose-free, feed-forward 3D Gaussian Splatting (3DGS) framework designed to handle unfavorable input views. A common rendering setup for training feed-forward approaches places a 3D object at the world origin and renders it from cameras pointed toward the origin -- i.e., from favorable views, limiting the applicability of these models to real-world scenarios involving varying and unknown camera poses. To overcome this limitation, we introduce a novel adaptation framework that enables pretrained pose-free feed-forward 3DGS models to handle unfavorable views. We leverage priors learned from favorable images by feeding recentered images into a pretrained model augmented with low-rank adaptation (LoRA) layers. We further propose a Gaussian adapter module to enhance the geometric consistency of the Gaussians derived from the recentered inputs, along with a Gaussian alignment method to render accurate target views for training. Additionally, we introduce a new training strategy that utilizes an off-the-shelf dataset composed solely of favorable images. Experimental results on both synthetic images from the Google Scanned Objects dataset and real images from the OmniObject3D dataset validate the effectiveness of our method in handling unfavorable input views.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08510v2">FOCI: Trajectory Optimization on Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 8 pages, 8 figures, Mario Gomez Andreu and Maximum Wilder-Smith contributed equally
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently gained popularity as a faster alternative to Neural Radiance Fields (NeRFs) in 3D reconstruction and view synthesis methods. Leveraging the spatial information encoded in 3DGS, this work proposes FOCI (Field Overlap Collision Integral), an algorithm that is able to optimize trajectories directly on the Gaussians themselves. FOCI leverages a novel and interpretable collision formulation for 3DGS using the notion of the overlap integral between Gaussians. Contrary to other approaches, which represent the robot with conservative bounding boxes that underestimate the traversability of the environment, we propose to represent the environment and the robot as Gaussian Splats. This not only has desirable computational properties, but also allows for orientation-aware planning, allowing the robot to pass through very tight and narrow spaces. We extensively test our algorithm in both synthetic and real Gaussian Splats, showcasing that collision-free trajectories for the ANYmal legged robot that can be computed in a few seconds, even with hundreds of thousands of Gaussians making up the environment. The project page and code are available at https://rffr.leggedrobotics.com/works/foci/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11692v2">FloPE: Flower Pose Estimation for Precision Pollination</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 Accepted to IROS 2025. Project page: https://wvu-irl.github.io/flope-irl/
    </div>
    <details class="paper-abstract">
      This study presents Flower Pose Estimation (FloPE), a real-time flower pose estimation framework for computationally constrained robotic pollination systems. Robotic pollination has been proposed to supplement natural pollination to ensure global food security due to the decreased population of natural pollinators. However, flower pose estimation for pollination is challenging due to natural variability, flower clusters, and high accuracy demands due to the flowers' fragility when pollinating. This method leverages 3D Gaussian Splatting to generate photorealistic synthetic datasets with precise pose annotations, enabling effective knowledge distillation from a high-capacity teacher model to a lightweight student model for efficient inference. The approach was evaluated on both single and multi-arm robotic platforms, achieving a mean pose estimation error of 0.6 cm and 19.14 degrees within a low computational cost. Our experiments validate the effectiveness of FloPE, achieving up to 78.75% pollination success rate and outperforming prior robotic pollination techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03984v3">GRaD-Nav: Efficiently Learning Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Autonomous visual navigation is an essential element in robot autonomy. Reinforcement learning (RL) offers a promising policy training paradigm. However existing RL methods suffer from high sample complexity, poor sim-to-real transfer, and limited runtime adaptability to navigation scenarios not seen during training. These problems are particularly challenging for drones, with complex nonlinear and unstable dynamics, and strong dynamic coupling between control and perception. In this paper, we propose a novel framework that integrates 3D Gaussian Splatting (3DGS) with differentiable deep reinforcement learning (DDRL) to train vision-based drone navigation policies. By leveraging high-fidelity 3D scene representations and differentiable simulation, our method improves sample efficiency and sim-to-real transfer. Additionally, we incorporate a Context-aided Estimator Network (CENet) to adapt to environmental variations at runtime. Moreover, by curriculum training in a mixture of different surrounding environments, we achieve in-task generalization, the ability to solve new instances of a task not seen during training. Drone hardware experiments demonstrate our method's high training efficiency compared to state-of-the-art RL methods, zero shot sim-to-real transfer for real robot deployment without fine tuning, and ability to adapt to new instances within the same task class (e.g. to fly through a gate at different locations with different distractors in the environment). Our simulator and training framework are open-sourced at: https://github.com/Qianzhong-Chen/grad_nav.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23006v1">Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      We present a framework that enables fast reconstruction and real-time rendering of urban-scale scenes while maintaining robustness against appearance variations across multi-view captures. Our approach begins with scene partitioning for parallel training, employing a visibility-based image selection strategy to optimize training efficiency. A controllable level-of-detail (LOD) strategy explicitly regulates Gaussian density under a user-defined budget, enabling efficient training and rendering while maintaining high visual fidelity. The appearance transformation module mitigates the negative effects of appearance inconsistencies across images while enabling flexible adjustments. Additionally, we utilize enhancement modules, such as depth regularization, scale regularization, and antialiasing, to improve reconstruction fidelity. Experimental results demonstrate that our method effectively reconstructs urban-scale scenes and outperforms previous approaches in both efficiency and quality. The source code is available at: https://yzslab.github.io/REUrbanGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21981v1">DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 8pages, IROS2025 (Camera Ready)
    </div>
    <details class="paper-abstract">
      We present the first unified, modular, open-source 3DGS-based simulation framework for Real2Sim2Real robot learning. It features a holistic Real2Sim pipeline that synthesizes hyper-realistic geometry and appearance of complex real-world scenarios, paving the way for analyzing and bridging the Sim2Real gap. Powered by Gaussian Splatting and MuJoCo, Discoverse enables massively parallel simulation of multiple sensor modalities and accurate physics, with inclusive supports for existing 3D assets, robot models, and ROS plugins, empowering large-scale robot learning and complex robotic benchmarks. Through extensive experiments on imitation learning, Discoverse demonstrates state-of-the-art zero-shot Sim2Real transfer performance compared to existing simulators. For code and demos: https://air-discoverse.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21872v1">MultiEditor: Controllable Multimodal Object Editing for Driving Scenarios Using 3D Gaussian Splatting Priors</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Autonomous driving systems rely heavily on multimodal perception data to understand complex environments. However, the long-tailed distribution of real-world data hinders generalization, especially for rare but safety-critical vehicle categories. To address this challenge, we propose MultiEditor, a dual-branch latent diffusion framework designed to edit images and LiDAR point clouds in driving scenarios jointly. At the core of our approach is introducing 3D Gaussian Splatting (3DGS) as a structural and appearance prior for target objects. Leveraging this prior, we design a multi-level appearance control mechanism--comprising pixel-level pasting, semantic-level guidance, and multi-branch refinement--to achieve high-fidelity reconstruction across modalities. We further propose a depth-guided deformable cross-modality condition module that adaptively enables mutual guidance between modalities using 3DGS-rendered depth, significantly enhancing cross-modality consistency. Extensive experiments demonstrate that MultiEditor achieves superior performance in visual and geometric fidelity, editing controllability, and cross-modality consistency. Furthermore, generating rare-category vehicle data with MultiEditor substantially enhances the detection accuracy of perception models on underrepresented classes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10170v2">GS-SDF: LiDAR-Augmented Gaussian Splatting and Neural SDF for Geometrically Consistent Rendering and Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 8 pages, IROS 2025
    </div>
    <details class="paper-abstract">
      Digital twins are fundamental to the development of autonomous driving and embodied artificial intelligence. However, achieving high-granularity surface reconstruction and high-fidelity rendering remains a challenge. Gaussian splatting offers efficient photorealistic rendering but struggles with geometric inconsistencies due to fragmented primitives and sparse observational data in robotics applications. Existing regularization methods, which rely on render-derived constraints, often fail in complex environments. Moreover, effectively integrating sparse LiDAR data with Gaussian splatting remains challenging. We propose a unified LiDAR-visual system that synergizes Gaussian splatting with a neural signed distance field. The accurate LiDAR point clouds enable a trained neural signed distance field to offer a manifold geometry field. This motivates us to offer an SDF-based Gaussian initialization for physically grounded primitive placement and a comprehensive geometric regularization for geometrically consistent rendering and reconstruction. Experiments demonstrate superior reconstruction accuracy and rendering quality across diverse trajectories. To benefit the community, the codes are released at https://github.com/hku-mars/GS-SDF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02751v3">RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 ICCV 2025. Project page: https://fcyycf.github.io/RobustSplat/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has gained significant attention for its real-time, photo-realistic rendering in novel-view synthesis and 3D modeling. However, existing methods struggle with accurately modeling scenes affected by transient objects, leading to artifacts in the rendered images. We identify that the Gaussian densification process, while enhancing scene detail capture, unintentionally contributes to these artifacts by growing additional Gaussians that model transient disturbances. To address this, we propose RobustSplat, a robust solution based on two critical designs. First, we introduce a delayed Gaussian growth strategy that prioritizes optimizing static scene structure before allowing Gaussian splitting/cloning, mitigating overfitting to transient objects in early optimization. Second, we design a scale-cascaded mask bootstrapping approach that first leverages lower-resolution feature similarity supervision for reliable initial transient mask estimation, taking advantage of its stronger semantic consistency and robustness to noise, and then progresses to high-resolution supervision to achieve more precise mask prediction. Extensive experiments on multiple challenging datasets show that our method outperforms existing methods, clearly demonstrating the robustness and effectiveness of our method. Our project page is https://fcyycf.github.io/RobustSplat/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21572v1">No Redundancy, No Stall: Lightweight Streaming 3D Gaussian Splatting for Real-time Rendering</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 Accepted by International Conference on Computer-Aided Design (ICCAD) 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables high-quality rendering of 3D scenes and is getting increasing adoption in domains like autonomous driving and embodied intelligence. However, 3DGS still faces major efficiency challenges when faced with high frame rate requirements and resource-constrained edge deployment. To enable efficient 3DGS, in this paper, we propose LS-Gaussian, an algorithm/hardware co-design framework for lightweight streaming 3D rendering. LS-Gaussian is motivated by the core observation that 3DGS suffers from substantial computation redundancy and stalls. On one hand, in practical scenarios, high-frame-rate 3DGS is often applied in settings where a camera observes and renders the same scene continuously but from slightly different viewpoints. Therefore, instead of rendering each frame separately, LS-Gaussian proposes a viewpoint transformation algorithm that leverages inter-frame continuity for efficient sparse rendering. On the other hand, as different tiles within an image are rendered in parallel but have imbalanced workloads, frequent hardware stalls also slow down the rendering process. LS-Gaussian predicts the workload for each tile based on viewpoint transformation to enable more balanced parallel computation and co-designs a customized 3DGS accelerator to support the workload-aware mapping in real-time. Experimental results demonstrate that LS-Gaussian achieves 5.41x speedup over the edge GPU baseline on average and up to 17.3x speedup with the customized accelerator, while incurring only minimal visual quality degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19141v2">DASH: 4D Hash Encoding with Self-Supervised Decomposition for Real-Time Dynamic Scene Rendering</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 ICCV 2025
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction is a long-term challenge in 3D vision. Existing plane-based methods in dynamic Gaussian splatting suffer from an unsuitable low-rank assumption, causing feature overlap and poor rendering quality. Although 4D hash encoding provides an explicit representation without low-rank constraints, directly applying it to the entire dynamic scene leads to substantial hash collisions and redundancy. To address these challenges, we present DASH, a real-time dynamic scene rendering framework that employs 4D hash encoding coupled with self-supervised decomposition. Our approach begins with a self-supervised decomposition mechanism that separates dynamic and static components without manual annotations or precomputed masks. Next, we introduce a multiresolution 4D hash encoder for dynamic elements, providing an explicit representation that avoids the low-rank assumption. Finally, we present a spatio-temporal smoothness regularization strategy to mitigate unstable deformation artifacts. Experiments on real-world datasets demonstrate that DASH achieves state-of-the-art dynamic rendering performance, exhibiting enhanced visual quality at real-time speeds of 264 FPS on a single 4090 GPU. Code: https://github.com/chenj02/DASH.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20331v2">From Gallery to Wrist: Realistic 3D Bracelet Insertion in Videos</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Inserting 3D objects into videos is a longstanding challenge in computer graphics with applications in augmented reality, virtual try-on, and video composition. Achieving both temporal consistency, or realistic lighting remains difficult, particularly in dynamic scenarios with complex object motion, perspective changes, and varying illumination. While 2D diffusion models have shown promise for producing photorealistic edits, they often struggle with maintaining temporal coherence across frames. Conversely, traditional 3D rendering methods excel in spatial and temporal consistency but fall short in achieving photorealistic lighting. In this work, we propose a hybrid object insertion pipeline that combines the strengths of both paradigms. Specifically, we focus on inserting bracelets into dynamic wrist scenes, leveraging the high temporal consistency of 3D Gaussian Splatting (3DGS) for initial rendering and refining the results using a 2D diffusion-based enhancement model to ensure realistic lighting interactions. Our method introduces a shading-driven pipeline that separates intrinsic object properties (albedo, shading, reflectance) and refines both shading and sRGB images for photorealism. To maintain temporal coherence, we optimize the 3DGS model with multi-frame weighted adjustments. This is the first approach to synergize 3D rendering and 2D diffusion for video object insertion, offering a robust solution for realistic and consistent video editing. Project Page: https://cjeen.github.io/BraceletPaper/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02178v3">Sparfels: Fast Reconstruction from Sparse Unposed Imagery</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 ICCV 2025. Project page : https://shubhendu-jena.github.io/Sparfels-web/
    </div>
    <details class="paper-abstract">
      We present a method for Sparse view reconstruction with surface element splatting that runs within 3 minutes on a consumer grade GPU. While few methods address sparse radiance field learning from noisy or unposed sparse cameras, shape recovery remains relatively underexplored in this setting. Several radiance and shape learning test-time optimization methods address the sparse posed setting by learning data priors or using combinations of external monocular geometry priors. Differently, we propose an efficient and simple pipeline harnessing a single recent 3D foundation model. We leverage its various task heads, notably point maps and camera initializations to instantiate a bundle adjusting 2D Gaussian Splatting (2DGS) model, and image correspondences to guide camera optimization midst 2DGS training. Key to our contribution is a novel formulation of splatted color variance along rays, which can be computed efficiently. Reducing this moment in training leads to more accurate shape reconstructions. We demonstrate state-of-the-art performances in the sparse uncalibrated setting in reconstruction and novel view benchmarks based on established multi-view datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16392v2">Quadratic Gaussian Splatting: High Quality Surface Reconstruction with Second-order Geometric Primitives</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 16pages,18figures
    </div>
    <details class="paper-abstract">
      We propose Quadratic Gaussian Splatting (QGS), a novel representation that replaces static primitives with deformable quadric surfaces (e.g., ellipse, paraboloids) to capture intricate geometry. Unlike prior works that rely on Euclidean distance for primitive density modeling--a metric misaligned with surface geometry under deformation--QGS introduces geodesic distance-based density distributions. This innovation ensures that density weights adapt intrinsically to the primitive curvature, preserving consistency during shape changes (e.g., from planar disks to curved paraboloids). By solving geodesic distances in closed form on quadric surfaces, QGS enables surface-aware splatting, where a single primitive can represent complex curvature that previously required dozens of planar surfels, potentially reducing memory usage while maintaining efficient rendering via fast ray-quadric intersection. Experiments on DTU, Tanks and Temples, and MipNeRF360 datasets demonstrate state-of-the-art surface reconstruction, with QGS reducing geometric error (chamfer distance) by 33% over 2DGS and 27% over GOF on the DTU dataset. Crucially, QGS retains competitive appearance quality, bridging the gap between geometric precision and visual fidelity for applications like robotics and immersive reality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15602v2">SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Accepted to ICCV 2025
    </div>
    <details class="paper-abstract">
      Surface reconstruction and novel view rendering from sparse-view images are challenging. Signed Distance Function (SDF)-based methods struggle with fine details, while 3D Gaussian Splatting (3DGS)-based approaches lack global geometry coherence. We propose a novel hybrid method that combines the strengths of both approaches: SDF captures coarse geometry to enhance 3DGS-based rendering, while newly rendered images from 3DGS refine the details of SDF for accurate surface reconstruction. As a result, our method surpasses state-of-the-art approaches in surface reconstruction and novel view synthesis on the DTU and MobileBrick datasets. Code will be released at https://github.com/aim-uofa/SurfaceSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03456v3">LM-Gaussian: Boost Sparse-view 3D Gaussian Splatting with Large Model Priors</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Project page: https://hanyangyu1021.github.io/lm-gaussian.github.io/
    </div>
    <details class="paper-abstract">
      We aim to address sparse-view reconstruction of a 3D scene by leveraging priors from large-scale vision models. While recent advancements such as 3D Gaussian Splatting (3DGS) have demonstrated remarkable successes in 3D reconstruction, these methods typically necessitate hundreds of input images that densely capture the underlying scene, making them time-consuming and impractical for real-world applications. However, sparse-view reconstruction is inherently ill-posed and under-constrained, often resulting in inferior and incomplete outcomes. This is due to issues such as failed initialization, overfitting on input images, and a lack of details. To mitigate these challenges, we introduce LM-Gaussian, a method capable of generating high-quality reconstructions from a limited number of images. Specifically, we propose a robust initialization module that leverages stereo priors to aid in the recovery of camera poses and the reliable point clouds. Additionally, a diffusion-based refinement is iteratively applied to incorporate image diffusion priors into the Gaussian optimization process to preserve intricate scene details. Finally, we utilize video diffusion priors to further enhance the rendered images for realistic visual effects. Overall, our approach significantly reduces the data acquisition requirements compared to previous 3DGS methods. We validate the effectiveness of our framework through experiments on various public datasets, demonstrating its potential for high-quality 360-degree scene reconstruction. Visual results are on our website.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20512v1">GaRe: Relightable 3D Gaussian Splatting for Outdoor Scenes from Unconstrained Photo Collections</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      We propose a 3D Gaussian splatting-based framework for outdoor relighting that leverages intrinsic image decomposition to precisely integrate sunlight, sky radiance, and indirect lighting from unconstrained photo collections. Unlike prior methods that compress the per-image global illumination into a single latent vector, our approach enables simultaneously diverse shading manipulation and the generation of dynamic shadow effects. This is achieved through three key innovations: (1) a residual-based sun visibility extraction method to accurately separate direct sunlight effects, (2) a region-based supervision framework with a structural consistency loss for physically interpretable and coherent illumination decomposition, and (3) a ray-tracing-based technique for realistic shadow simulation. Extensive experiments demonstrate that our framework synthesizes novel views with competitive fidelity against state-of-the-art relighting solutions and produces more natural and multifaceted illumination and shadow effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20480v1">Automated 3D-GS Registration and Fusion via Skeleton Alignment and Gaussian-Adaptive Features</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Accepted to IROS 2025
    </div>
    <details class="paper-abstract">
      In recent years, 3D Gaussian Splatting (3D-GS)-based scene representation demonstrates significant potential in real-time rendering and training efficiency. However, most existing methods primarily focus on single-map reconstruction, while the registration and fusion of multiple 3D-GS sub-maps remain underexplored. Existing methods typically rely on manual intervention to select a reference sub-map as a template and use point cloud matching for registration. Moreover, hard-threshold filtering of 3D-GS primitives often degrades rendering quality after fusion. In this paper, we present a novel approach for automated 3D-GS sub-map alignment and fusion, eliminating the need for manual intervention while enhancing registration accuracy and fusion quality. First, we extract geometric skeletons across multiple scenes and leverage ellipsoid-aware convolution to capture 3D-GS attributes, facilitating robust scene registration. Second, we introduce a multi-factor Gaussian fusion strategy to mitigate the scene element loss caused by rigid thresholding. Experiments on the ScanNet-GSReg and our Coord datasets demonstrate the effectiveness of the proposed method in registration and fusion. For registration, it achieves a 41.9\% reduction in RRE on complex scenes, ensuring more precise pose estimation. For fusion, it improves PSNR by 10.11 dB, highlighting superior structural preservation. These results confirm its ability to enhance scene alignment and reconstruction fidelity, ensuring more consistent and accurate 3D scene representation for robotic perception and autonomous navigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21311v1">VoluMe -- Authentic 3D Video Calls from Live Gaussian Splat Prediction</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Virtual 3D meetings offer the potential to enhance copresence, increase engagement and thus improve effectiveness of remote meetings compared to standard 2D video calls. However, representing people in 3D meetings remains a challenge; existing solutions achieve high quality by using complex hardware, making use of fixed appearance via enrolment, or by inverting a pre-trained generative model. These approaches lead to constraints that are unwelcome and ill-fitting for videoconferencing applications. We present the first method to predict 3D Gaussian reconstructions in real time from a single 2D webcam feed, where the 3D representation is not only live and realistic, but also authentic to the input video. By conditioning the 3D representation on each video frame independently, our reconstruction faithfully recreates the input video from the captured viewpoint (a property we call authenticity), while generalizing realistically to novel viewpoints. Additionally, we introduce a stability loss to obtain reconstructions that are temporally stable on video sequences. We show that our method delivers state-of-the-art accuracy in visual quality and stability metrics compared to existing methods, and demonstrate our approach in live one-to-one 3D meetings using only a standard 2D camera and display. This demonstrates that our approach can allow anyone to communicate volumetrically, via a method for 3D videoconferencing that is not only highly accessible, but also realistic and authentic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14462v5">LUDVIG: Learning-Free Uplifting of 2D Visual Features to Gaussian Splatting Scenes</a></div>
    <div class="paper-meta">
      📅 2025-07-27
      | 💬 Published at ICCV 2025. Project page: https://juliettemarrie.github.io/ludvig
    </div>
    <details class="paper-abstract">
      We address the problem of extending the capabilities of vision foundation models such as DINO, SAM, and CLIP, to 3D tasks. Specifically, we introduce a novel method to uplift 2D image features into Gaussian Splatting representations of 3D scenes. Unlike traditional approaches that rely on minimizing a reconstruction loss, our method employs a simpler and more efficient feature aggregation technique, augmented by a graph diffusion mechanism. Graph diffusion refines 3D features, such as coarse segmentation masks, by leveraging 3D geometry and pairwise similarities induced by DINOv2. Our approach achieves performance comparable to the state of the art on multiple downstream tasks while delivering significant speed-ups. Notably, we obtain competitive segmentation results using only generic DINOv2 features, despite DINOv2 not being trained on millions of annotated segmentation masks like SAM. When applied to CLIP features, our method demonstrates strong performance in open-vocabulary object segmentation tasks, highlighting the versatility of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20331v1">From Gallery to Wrist: Realistic 3D Bracelet Insertion in Videos</a></div>
    <div class="paper-meta">
      📅 2025-07-27
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Inserting 3D objects into videos is a longstanding challenge in computer graphics with applications in augmented reality, virtual try-on, and video composition. Achieving both temporal consistency, or realistic lighting remains difficult, particularly in dynamic scenarios with complex object motion, perspective changes, and varying illumination. While 2D diffusion models have shown promise for producing photorealistic edits, they often struggle with maintaining temporal coherence across frames. Conversely, traditional 3D rendering methods excel in spatial and temporal consistency but fall short in achieving photorealistic lighting. In this work, we propose a hybrid object insertion pipeline that combines the strengths of both paradigms. Specifically, we focus on inserting bracelets into dynamic wrist scenes, leveraging the high temporal consistency of 3D Gaussian Splatting (3DGS) for initial rendering and refining the results using a 2D diffusion-based enhancement model to ensure realistic lighting interactions. Our method introduces a shading-driven pipeline that separates intrinsic object properties (albedo, shading, reflectance) and refines both shading and sRGB images for photorealism. To maintain temporal coherence, we optimize the 3DGS model with multi-frame weighted adjustments. This is the first approach to synergize 3D rendering and 2D diffusion for video object insertion, offering a robust solution for realistic and consistent video editing. Project Page: https://cjeen.github.io/BraceletPaper/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13610v4">Video2BEV: Transforming Drone Videos to BEVs for Video-based Geo-localization</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Existing approaches to drone visual geo-localization predominantly adopt the image-based setting, where a single drone-view snapshot is matched with images from other platforms. Such task formulation, however, underutilizes the inherent video output of the drone and is sensitive to occlusions and viewpoint disparity. To address these limitations, we formulate a new video-based drone geo-localization task and propose the Video2BEV paradigm. This paradigm transforms the video into a Bird's Eye View (BEV), simplifying the subsequent \textbf{inter-platform} matching process. In particular, we employ Gaussian Splatting to reconstruct a 3D scene and obtain the BEV projection. Different from the existing transform methods, \eg, polar transform, our BEVs preserve more fine-grained details without significant distortion. To facilitate the discriminative \textbf{intra-platform} representation learning, our Video2BEV paradigm also incorporates a diffusion-based module for generating hard negative samples. To validate our approach, we introduce UniV, a new video-based geo-localization dataset that extends the image-based University-1652 dataset. UniV features flight paths at $30^\circ$ and $45^\circ$ elevation angles with increased frame rates of up to 10 frames per second (FPS). Extensive experiments on the UniV dataset show that our Video2BEV paradigm achieves competitive recall rates and outperforms conventional video-based methods. Compared to other competitive methods, our proposed approach exhibits robustness at lower elevations with more occlusions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20239v1">Decomposing Densification in Gaussian Splatting for Faster 3D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (GS) has emerged as a powerful representation for high-quality scene reconstruction, offering compelling rendering quality. However, the training process of GS often suffers from slow convergence due to inefficient densification and suboptimal spatial distribution of Gaussian primitives. In this work, we present a comprehensive analysis of the split and clone operations during the densification phase, revealing their distinct roles in balancing detail preservation and computational efficiency. Building upon this analysis, we propose a global-to-local densification strategy, which facilitates more efficient growth of Gaussians across the scene space, promoting both global coverage and local refinement. To cooperate with the proposed densification strategy and promote sufficient diffusion of Gaussian primitives in space, we introduce an energy-guided coarse-to-fine multi-resolution training framework, which gradually increases resolution based on energy density in 2D images. Additionally, we dynamically prune unnecessary Gaussian primitives to speed up the training. Extensive experiments on MipNeRF-360, Deep Blending, and Tanks & Temples datasets demonstrate that our approach significantly accelerates training,achieving over 2x speedup with fewer Gaussian primitives and superior reconstruction performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20200v1">Neural Shell Texture Splatting: More Details and Fewer Primitives</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Gaussian splatting techniques have shown promising results in novel view synthesis, achieving high fidelity and efficiency. However, their high reconstruction quality comes at the cost of requiring a large number of primitives. We identify this issue as stemming from the entanglement of geometry and appearance in Gaussian Splatting. To address this, we introduce a neural shell texture, a global representation that encodes texture information around the surface. We use Gaussian primitives as both a geometric representation and texture field samplers, efficiently splatting texture features into image space. Our evaluation demonstrates that this disentanglement enables high parameter efficiency, fine texture detail reconstruction, and easy textured mesh extraction, all while using significantly fewer primitives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11921v2">DeSiRe-GS: 4D Street Gaussians for Static-Dynamic Decomposition and Surface Reconstruction for Urban Driving Scenes</a></div>
    <div class="paper-meta">
      📅 2025-07-26
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      We present DeSiRe-GS, a self-supervised gaussian splatting representation, enabling effective static-dynamic decomposition and high-fidelity surface reconstruction in complex driving scenarios. Our approach employs a two-stage optimization pipeline of dynamic street Gaussians. In the first stage, we extract 2D motion masks based on the observation that 3D Gaussian Splatting inherently can reconstruct only the static regions in dynamic environments. These extracted 2D motion priors are then mapped into the Gaussian space in a differentiable manner, leveraging an efficient formulation of dynamic Gaussians in the second stage. Combined with the introduced geometric regularizations, our method are able to address the over-fitting issues caused by data sparsity in autonomous driving, reconstructing physically plausible Gaussians that align with object surfaces rather than floating in air. Furthermore, we introduce temporal cross-view consistency to ensure coherence across time and viewpoints, resulting in high-quality surface reconstruction. Comprehensive experiments demonstrate the efficiency and effectiveness of DeSiRe-GS, surpassing prior self-supervised arts and achieving accuracy comparable to methods relying on external 3D bounding box annotations. Code is available at https://github.com/chengweialan/DeSiRe-GS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00623v3">A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision</a></div>
    <div class="paper-meta">
      📅 2025-07-26
      | 💬 ICCV 2025
    </div>
    <details class="paper-abstract">
      We present a novel framework for training 3D image-conditioned diffusion models using only 2D supervision. Recovering 3D structure from 2D images is inherently ill-posed due to the ambiguity of possible reconstructions, making generative models a natural choice. However, most existing 3D generative models rely on full 3D supervision, which is impractical due to the scarcity of large-scale 3D datasets. To address this, we propose leveraging sparse-view supervision as a scalable alternative. While recent reconstruction models use sparse-view supervision with differentiable rendering to lift 2D images to 3D, they are predominantly deterministic, failing to capture the diverse set of plausible solutions and producing blurry predictions in uncertain regions. A key challenge in training 3D diffusion models with 2D supervision is that the standard training paradigm requires both the denoising process and supervision to be in the same modality. We address this by decoupling the noisy samples being denoised from the supervision signal, allowing the former to remain in 3D while the latter is provided in 2D. Our approach leverages suboptimal predictions from a deterministic image-to-3D model-acting as a "teacher"-to generate noisy 3D inputs, enabling effective 3D diffusion training without requiring full 3D ground truth. We validate our framework on both object-level and scene-level datasets, using two different 3D Gaussian Splat (3DGS) teachers. Our results show that our approach consistently improves upon these deterministic teachers, demonstrating its effectiveness in scalable and high-fidelity 3D generative modeling. See our project page at https://lesson-in-splats.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19856v1">RaGS: Unleashing 3D Gaussian Splatting from 4D Radar and Monocular Cues for 3D Object Detection</a></div>
    <div class="paper-meta">
      📅 2025-07-26
      | 💬 9 pages, 6 figures, conference
    </div>
    <details class="paper-abstract">
      4D millimeter-wave radar has emerged as a promising sensor for autonomous driving, but effective 3D object detection from both 4D radar and monocular images remains a challenge. Existing fusion approaches typically rely on either instance-based proposals or dense BEV grids, which either lack holistic scene understanding or are limited by rigid grid structures. To address these, we propose RaGS, the first framework to leverage 3D Gaussian Splatting (GS) as representation for fusing 4D radar and monocular cues in 3D object detection. 3D GS naturally suits 3D object detection by modeling the scene as a field of Gaussians, dynamically allocating resources on foreground objects and providing a flexible, resource-efficient solution. RaGS uses a cascaded pipeline to construct and refine the Gaussian field. It starts with the Frustum-based Localization Initiation (FLI), which unprojects foreground pixels to initialize coarse 3D Gaussians positions. Then, the Iterative Multimodal Aggregation (IMA) fuses semantics and geometry, refining the limited Gaussians to the regions of interest. Finally, the Multi-level Gaussian Fusion (MGF) renders the Gaussians into multi-level BEV features for 3D object detection. By dynamically focusing on sparse objects within scenes, RaGS enable object concentrating while offering comprehensive scene perception. Extensive experiments on View-of-Delft, TJ4DRadSet, and OmniHD-Scenes benchmarks demonstrate its state-of-the-art performance. Code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05152v3">GSplatVNM: Point-of-View Synthesis for Visual Navigation Models Using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-26
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach to image-goal navigation by integrating 3D Gaussian Splatting (3DGS) with Visual Navigation Models (VNMs), a method we refer to as GSplatVNM. VNMs offer a promising paradigm for image-goal navigation by guiding a robot through a sequence of point-of-view images without requiring metrical localization or environment-specific training. However, constructing a dense and traversable sequence of target viewpoints from start to goal remains a central challenge, particularly when the available image database is sparse. To address these challenges, we propose a 3DGS-based viewpoint synthesis framework for VNMs that synthesizes intermediate viewpoints to seamlessly bridge gaps in sparse data while significantly reducing storage overhead. Experimental results in a photorealistic simulator demonstrate that our approach not only enhances navigation efficiency but also exhibits robustness under varying levels of image database sparsity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19830v1">Taking Language Embedded 3D Gaussian Splatting into the Wild</a></div>
    <div class="paper-meta">
      📅 2025-07-26
      | 💬 Visit our project page at https://yuzewang1998.github.io/takinglangsplatw/
    </div>
    <details class="paper-abstract">
      Recent advances in leveraging large-scale Internet photo collections for 3D reconstruction have enabled immersive virtual exploration of landmarks and historic sites worldwide. However, little attention has been given to the immersive understanding of architectural styles and structural knowledge, which remains largely confined to browsing static text-image pairs. Therefore, can we draw inspiration from 3D in-the-wild reconstruction techniques and use unconstrained photo collections to create an immersive approach for understanding the 3D structure of architectural components? To this end, we extend language embedded 3D Gaussian splatting (3DGS) and propose a novel framework for open-vocabulary scene understanding from unconstrained photo collections. Specifically, we first render multiple appearance images from the same viewpoint as the unconstrained image with the reconstructed radiance field, then extract multi-appearance CLIP features and two types of language feature uncertainty maps-transient and appearance uncertainty-derived from the multi-appearance features to guide the subsequent optimization process. Next, we propose a transient uncertainty-aware autoencoder, a multi-appearance language field 3DGS representation, and a post-ensemble strategy to effectively compress, learn, and fuse language features from multiple appearances. Finally, to quantitatively evaluate our method, we introduce PT-OVS, a new benchmark dataset for assessing open-vocabulary segmentation performance on unconstrained photo collections. Experimental results show that our method outperforms existing methods, delivering accurate open-vocabulary segmentation and enabling applications such as interactive roaming with open-vocabulary queries, architectural style pattern recognition, and 3D scene editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12789v3">Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-26
      | 💬 ICCV 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D generation models have opened new possibilities for simulating dynamic 3D object movements and customizing behaviors, yet creating this content remains challenging. Current methods often require manual assignment of precise physical properties for simulations or rely on video generation models to predict them, which is computationally intensive. In this paper, we rethink the usage of multi-modal large language model (MLLM) in physics-based simulation, and present Sim Anything, a physics-based approach that endows static 3D objects with interactive dynamics. We begin with detailed scene reconstruction and object-level 3D open-vocabulary segmentation, progressing to multi-view image in-painting. Inspired by human visual reasoning, we propose MLLM-based Physical Property Perception (MLLM-P3) to predict mean physical properties of objects in a zero-shot manner. Based on the mean values and the object's geometry, the Material Property Distribution Prediction model (MPDP) model then estimates the full distribution, reformulating the problem as probability distribution estimation to reduce computational costs. Finally, we simulate objects in an open-world scene with particles sampled via the Physical-Geometric Adaptive Sampling (PGAS) strategy, efficiently capturing complex deformations and significantly reducing computational costs. Extensive experiments and user studies demonstrate our Sim Anything achieves more realistic motion than state-of-the-art methods within 2 minutes on a single GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19474v1">DINO-SLAM: DINO-informed RGB-D SLAM for Neural Implicit and Explicit Representations</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      This paper presents DINO-SLAM, a DINO-informed design strategy to enhance neural implicit (Neural Radiance Field -- NeRF) and explicit representations (3D Gaussian Splatting -- 3DGS) in SLAM systems through more comprehensive scene representations. Purposely, we rely on a Scene Structure Encoder (SSE) that enriches DINO features into Enhanced DINO ones (EDINO) to capture hierarchical scene elements and their structural relationships. Building upon it, we propose two foundational paradigms for NeRF and 3DGS SLAM systems integrating EDINO features. Our DINO-informed pipelines achieve superior performance on the Replica, ScanNet, and TUM compared to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19459v1">Fast Learning of Non-Cooperative Spacecraft 3D Models through Primitive Initialization</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      The advent of novel view synthesis techniques such as NeRF and 3D Gaussian Splatting (3DGS) has enabled learning precise 3D models only from posed monocular images. Although these methods are attractive, they hold two major limitations that prevent their use in space applications: they require poses during training, and have high computational cost at training and inference. To address these limitations, this work contributes: (1) a Convolutional Neural Network (CNN) based primitive initializer for 3DGS using monocular images; (2) a pipeline capable of training with noisy or implicit pose estimates; and (3) and analysis of initialization variants that reduce the training cost of precise 3D models. A CNN takes a single image as input and outputs a coarse 3D model represented as an assembly of primitives, along with the target's pose relative to the camera. This assembly of primitives is then used to initialize 3DGS, significantly reducing the number of training iterations and input images needed -- by at least an order of magnitude. For additional flexibility, the CNN component has multiple variants with different pose estimation techniques. This work performs a comparison between these variants, evaluating their effectiveness for downstream 3DGS training under noisy or implicit pose estimates. The results demonstrate that even with imperfect pose supervision, the pipeline is able to learn high-fidelity 3D representations, opening the door for the use of novel view synthesis in space applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19451v1">GS-Occ3D: Scaling Vision-only Occupancy Reconstruction for Autonomous Driving with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 ICCV 2025. Project Page: https://gs-occ3d.github.io/
    </div>
    <details class="paper-abstract">
      Occupancy is crucial for autonomous driving, providing essential geometric priors for perception and planning. However, existing methods predominantly rely on LiDAR-based occupancy annotations, which limits scalability and prevents leveraging vast amounts of potential crowdsourced data for auto-labeling. To address this, we propose GS-Occ3D, a scalable vision-only framework that directly reconstructs occupancy. Vision-only occupancy reconstruction poses significant challenges due to sparse viewpoints, dynamic scene elements, severe occlusions, and long-horizon motion. Existing vision-based methods primarily rely on mesh representation, which suffer from incomplete geometry and additional post-processing, limiting scalability. To overcome these issues, GS-Occ3D optimizes an explicit occupancy representation using an Octree-based Gaussian Surfel formulation, ensuring efficiency and scalability. Additionally, we decompose scenes into static background, ground, and dynamic objects, enabling tailored modeling strategies: (1) Ground is explicitly reconstructed as a dominant structural element, significantly improving large-area consistency; (2) Dynamic vehicles are separately modeled to better capture motion-related occupancy patterns. Extensive experiments on the Waymo dataset demonstrate that GS-Occ3D achieves state-of-the-art geometry reconstruction results. By curating vision-only binary occupancy labels from diverse urban scenes, we show their effectiveness for downstream occupancy models on Occ3D-Waymo and superior zero-shot generalization on Occ3D-nuScenes. It highlights the potential of large-scale vision-based occupancy reconstruction as a new paradigm for autonomous driving perception. Project Page: https://gs-occ3d.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19141v1">DASH: 4D Hash Encoding with Self-Supervised Decomposition for Real-Time Dynamic Scene Rendering</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction is a long-term challenge in 3D vision. Existing plane-based methods in dynamic Gaussian splatting suffer from an unsuitable low-rank assumption, causing feature overlap and poor rendering quality. Although 4D hash encoding provides an explicit representation without low-rank constraints, directly applying it to the entire dynamic scene leads to substantial hash collisions and redundancy. To address these challenges, we present DASH, a real-time dynamic scene rendering framework that employs 4D hash encoding coupled with self-supervised decomposition. Our approach begins with a self-supervised decomposition mechanism that separates dynamic and static components without manual annotations or precomputed masks. Next, we introduce a multiresolution 4D hash encoder for dynamic elements, providing an explicit representation that avoids the low-rank assumption. Finally, we present a spatio-temporal smoothness regularization strategy to mitigate unstable deformation artifacts. Experiments on real-world datasets demonstrate that DASH achieves state-of-the-art dynamic rendering performance, exhibiting enhanced visual quality at real-time speeds of 264 FPS on a single 4090 GPU. Code: https://github.com/chenj02/DASH.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19133v1">3DGauCIM: Accelerating Static/Dynamic 3D Gaussian Splatting via Digital CIM for High Frame Rate Real-Time Edge Rendering</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Dynamic 3D Gaussian splatting (3DGS) extends static 3DGS to render dynamic scenes, enabling AR/VR applications with moving objects. However, implementing dynamic 3DGS on edge devices faces challenges: (1) Loading all Gaussian parameters from DRAM for frustum culling incurs high energy costs. (2) Increased parameters for dynamic scenes elevate sorting latency and energy consumption. (3) Limited on-chip buffer capacity with higher parameters reduces buffer reuse, causing frequent DRAM access. (4) Dynamic 3DGS operations are not readily compatible with digital compute-in-memory (DCIM). These challenges hinder real-time performance and power efficiency on edge devices, leading to reduced battery life or requiring bulky batteries. To tackle these challenges, we propose algorithm-hardware co-design techniques. At the algorithmic level, we introduce three optimizations: (1) DRAM-access reduction frustum culling to lower DRAM access overhead, (2) Adaptive tile grouping to enhance on-chip buffer reuse, and (3) Adaptive interval initialization Bucket-Bitonic sort to reduce sorting latency. At the hardware level, we present a DCIM-friendly computation flow that is evaluated using the measured data from a 16nm DCIM prototype chip. Our experimental results on Large-Scale Real-World Static/Dynamic Datasets demonstrate the ability to achieve high frame rate real-time rendering exceeding 200 frame per second (FPS) with minimal power consumption, merely 0.28 W for static Large-Scale Real-World scenes and 0.63 W for dynamic Large-Scale Real-World scenes. This work successfully addresses the significant challenges of implementing static/dynamic 3DGS technology on resource-constrained edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18923v1">Gaussian Set Surface Reconstruction through Per-Gaussian Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) effectively synthesizes novel views through its flexible representation, yet fails to accurately reconstruct scene geometry. While modern variants like PGSR introduce additional losses to ensure proper depth and normal maps through Gaussian fusion, they still neglect individual placement optimization. This results in unevenly distributed Gaussians that deviate from the latent surface, complicating both reconstruction refinement and scene editing. Motivated by pioneering work on Point Set Surfaces, we propose Gaussian Set Surface Reconstruction (GSSR), a method designed to distribute Gaussians evenly along the latent surface while aligning their dominant normals with the surface normal. GSSR enforces fine-grained geometric alignment through a combination of pixel-level and Gaussian-level single-view normal consistency and multi-view photometric consistency, optimizing both local and global perspectives. To further refine the representation, we introduce an opacity regularization loss to eliminate redundant Gaussians and apply periodic depth- and normal-guided Gaussian reinitialization for a cleaner, more uniform spatial distribution. Our reconstruction results demonstrate significantly improved geometric precision in Gaussian placement, enabling intuitive scene editing and efficient generation of novel Gaussian-based 3D environments. Extensive experiments validate GSSR's effectiveness, showing enhanced geometric accuracy while preserving high-quality rendering performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14475v4">MagicDrive3D: Controllable 3D Generation for Any-View Rendering in Street Scenes</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 Project Page: https://flymin.github.io/magicdrive3d
    </div>
    <details class="paper-abstract">
      Controllable generative models for images and videos have seen significant success, yet 3D scene generation, especially in unbounded scenarios like autonomous driving, remains underdeveloped. Existing methods lack flexible controllability and often rely on dense view data collection in controlled environments, limiting their generalizability across common datasets (e.g., nuScenes). In this paper, we introduce MagicDrive3D, a novel framework for controllable 3D street scene generation that combines video-based view synthesis with 3D representation (3DGS) generation. It supports multi-condition control, including road maps, 3D objects, and text descriptions. Unlike previous approaches that require 3D representation before training, MagicDrive3D first trains a multi-view video generation model to synthesize diverse street views. This method utilizes routinely collected autonomous driving data, reducing data acquisition challenges and enriching 3D scene generation. In the 3DGS generation step, we introduce Fault-Tolerant Gaussian Splatting to address minor errors and use monocular depth for better initialization, alongside appearance modeling to manage exposure discrepancies across viewpoints. Experiments show that MagicDrive3D generates diverse, high-quality 3D driving scenes, supports any-view rendering, and enhances downstream tasks like BEV segmentation, demonstrating its potential for autonomous driving simulation and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15300v3">GCC: A 3DGS Inference Architecture with Gaussian-Wise and Cross-Stage Conditional Processing</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 Accepted to MICRO 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a leading neural rendering technique for high-fidelity view synthesis, prompting the development of dedicated 3DGS accelerators for resource-constrained platforms. The conventional decoupled preprocessing-rendering dataflow in existing accelerators has two major limitations: 1) a significant portion of preprocessed Gaussians are not used in rendering, and 2) the same Gaussian gets repeatedly loaded across different tile renderings, resulting in substantial computational and data movement overhead. To address these issues, we propose GCC, a novel accelerator designed for fast and energy-efficient 3DGS inference. GCC introduces a novel dataflow featuring: 1) \textit{cross-stage conditional processing}, which interleaves preprocessing and rendering to dynamically skip unnecessary Gaussian preprocessing; and 2) \textit{Gaussian-wise rendering}, ensuring that all rendering operations for a given Gaussian are completed before moving to the next, thereby eliminating duplicated Gaussian loading. We also propose an alpha-based boundary identification method to derive compact and accurate Gaussian regions, thereby reducing rendering costs. We implement our GCC accelerator in 28nm technology. Extensive experiments demonstrate that GCC significantly outperforms the state-of-the-art 3DGS inference accelerator, GSCore, in both performance and energy efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22800v3">RGE-GS: Reward-Guided Expansive Driving Scene Reconstruction via Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      A single-pass driving clip frequently results in incomplete scanning of the road structure, making reconstructed scene expanding a critical requirement for sensor simulators to effectively regress driving actions. Although contemporary 3D Gaussian Splatting (3DGS) techniques achieve remarkable reconstruction quality, their direct extension through the integration of diffusion priors often introduces cumulative physical inconsistencies and compromises training efficiency. To address these limitations, we present RGE-GS, a novel expansive reconstruction framework that synergizes diffusion-based generation with reward-guided Gaussian integration. The RGE-GS framework incorporates two key innovations: First, we propose a reward network that learns to identify and prioritize consistently generated patterns prior to reconstruction phases, thereby enabling selective retention of diffusion outputs for spatial stability. Second, during the reconstruction process, we devise a differentiated training strategy that automatically adjust Gaussian optimization progress according to scene converge metrics, which achieving better convergence than baseline methods. Extensive evaluations of publicly available datasets demonstrate that RGE-GS achieves state-of-the-art performance in reconstruction quality. Our source-code will be made publicly available at https://github.com/CN-ADLab/RGE-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19718v1">GSCache: Real-Time Radiance Caching for Volume Path Tracing using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Real-time path tracing is rapidly becoming the standard for rendering in entertainment and professional applications. In scientific visualization, volume rendering plays a crucial role in helping researchers analyze and interpret complex 3D data. Recently, photorealistic rendering techniques have gained popularity in scientific visualization, yet they face significant challenges. One of the most prominent issues is slow rendering performance and high pixel variance caused by Monte Carlo integration. In this work, we introduce a novel radiance caching approach for path-traced volume rendering. Our method leverages advances in volumetric scene representation and adapts 3D Gaussian splatting to function as a multi-level, path-space radiance cache. This cache is designed to be trainable on the fly, dynamically adapting to changes in scene parameters such as lighting configurations and transfer functions. By incorporating our cache, we achieve less noisy, higher-quality images without increasing rendering costs. To evaluate our approach, we compare it against a baseline path tracer that supports uniform sampling and next-event estimation and the state-of-the-art for neural radiance caching. Through both quantitative and qualitative analyses, we demonstrate that our path-space radiance cache is a robust solution that is easy to integrate and significantly enhances the rendering quality of volumetric visualization applications while maintaining comparable computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01804v6">EVER: Exact Volumetric Ellipsoid Rendering for Real-time View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 Project page: https://half-potato.gitlab.io/posts/ever
    </div>
    <details class="paper-abstract">
      We present Exact Volumetric Ellipsoid Rendering (EVER), a method for real-time differentiable emission-only volume rendering. Unlike recent rasterization based approach by 3D Gaussian Splatting (3DGS), our primitive based representation allows for exact volume rendering, rather than alpha compositing 3D Gaussian billboards. As such, unlike 3DGS our formulation does not suffer from popping artifacts and view dependent density, but still achieves frame rates of $\sim\!30$ FPS at 720p on an NVIDIA RTX4090. Since our approach is built upon ray tracing it enables effects such as defocus blur and camera distortion (e.g. such as from fisheye cameras), which are difficult to achieve by rasterization. We show that our method is more accurate with fewer blending issues than 3DGS and follow-up work on view-consistent rendering, especially on the challenging large-scale scenes from the Zip-NeRF dataset where it achieves sharpest results among real-time techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03737v2">Outdoor Monocular SLAM with Global Scale-Consistent 3D Gaussian Pointmaps</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 Accepted by ICCV2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a popular solution in SLAM due to its high-fidelity and real-time novel view synthesis performance. However, some previous 3DGS SLAM methods employ a differentiable rendering pipeline for tracking, lack geometric priors in outdoor scenes. Other approaches introduce separate tracking modules, but they accumulate errors with significant camera movement, leading to scale drift. To address these challenges, we propose a robust RGB-only outdoor 3DGS SLAM method: S3PO-GS. Technically, we establish a self-consistent tracking module anchored in the 3DGS pointmap, which avoids cumulative scale drift and achieves more precise and robust tracking with fewer iterations. Additionally, we design a patch-based pointmap dynamic mapping module, which introduces geometric priors while avoiding scale ambiguity. This significantly enhances tracking accuracy and the quality of scene reconstruction, making it particularly suitable for complex outdoor environments. Our experiments on the Waymo, KITTI, and DL3DV datasets demonstrate that S3PO-GS achieves state-of-the-art results in novel view synthesis and outperforms other 3DGS SLAM methods in tracking accuracy. Project page: https://3dagentworld.github.io/S3PO-GS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18541v1">Unposed 3DGS Reconstruction with Probabilistic Procrustes Mapping</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a core technique for 3D representation. Its effectiveness largely depends on precise camera poses and accurate point cloud initialization, which are often derived from pretrained Multi-View Stereo (MVS) models. However, in unposed reconstruction task from hundreds of outdoor images, existing MVS models may struggle with memory limits and lose accuracy as the number of input images grows. To address this limitation, we propose a novel unposed 3DGS reconstruction framework that integrates pretrained MVS priors with the probabilistic Procrustes mapping strategy. The method partitions input images into subsets, maps submaps into a global space, and jointly optimizes geometry and poses with 3DGS. Technically, we formulate the mapping of tens of millions of point clouds as a probabilistic Procrustes problem and solve a closed-form alignment. By employing probabilistic coupling along with a soft dustbin mechanism to reject uncertain correspondences, our method globally aligns point clouds and poses within minutes across hundreds of images. Moreover, we propose a joint optimization framework for 3DGS and camera poses. It constructs Gaussians from confidence-aware anchor points and integrates 3DGS differentiable rendering with an analytical Jacobian to jointly refine scene and poses, enabling accurate reconstruction and pose estimation. Experiments on Waymo and KITTI datasets show that our method achieves accurate reconstruction from unposed image sequences, setting a new state of the art for unposed 3DGS reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18473v1">CRUISE: Cooperative Reconstruction and Editing in V2X Scenarios using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 IROS 2025, Code: https://github.com/SainingZhang/CRUISE
    </div>
    <details class="paper-abstract">
      Vehicle-to-everything (V2X) communication plays a crucial role in autonomous driving, enabling cooperation between vehicles and infrastructure. While simulation has significantly contributed to various autonomous driving tasks, its potential for data generation and augmentation in V2X scenarios remains underexplored. In this paper, we introduce CRUISE, a comprehensive reconstruction-and-synthesis framework designed for V2X driving environments. CRUISE employs decomposed Gaussian Splatting to accurately reconstruct real-world scenes while supporting flexible editing. By decomposing dynamic traffic participants into editable Gaussian representations, CRUISE allows for seamless modification and augmentation of driving scenes. Furthermore, the framework renders images from both ego-vehicle and infrastructure views, enabling large-scale V2X dataset augmentation for training and evaluation. Our experimental results demonstrate that: 1) CRUISE reconstructs real-world V2X driving scenes with high fidelity; 2) using CRUISE improves 3D detection across ego-vehicle, infrastructure, and cooperative views, as well as cooperative 3D tracking on the V2X-Seq benchmark; and 3) CRUISE effectively generates challenging corner cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18371v1">MVG4D: Image Matrix-Based Multi-View and Motion Generation for 4D Content Creation from a Single Image</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Advances in generative modeling have significantly enhanced digital content creation, extending from 2D images to complex 3D and 4D scenes. Despite substantial progress, producing high-fidelity and temporally consistent dynamic 4D content remains a challenge. In this paper, we propose MVG4D, a novel framework that generates dynamic 4D content from a single still image by combining multi-view synthesis with 4D Gaussian Splatting (4D GS). At its core, MVG4D employs an image matrix module that synthesizes temporally coherent and spatially diverse multi-view images, providing rich supervisory signals for downstream 3D and 4D reconstruction. These multi-view images are used to optimize a 3D Gaussian point cloud, which is further extended into the temporal domain via a lightweight deformation network. Our method effectively enhances temporal consistency, geometric fidelity, and visual realism, addressing key challenges in motion discontinuity and background degradation that affect prior 4D GS-based methods. Extensive experiments on the Objaverse dataset demonstrate that MVG4D outperforms state-of-the-art baselines in CLIP-I, PSNR, FVD, and time efficiency. Notably, it reduces flickering artifacts and sharpens structural details across views and time, enabling more immersive AR/VR experiences. MVG4D sets a new direction for efficient and controllable 4D generation from minimal inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18344v1">G2S-ICP SLAM: Geometry-aware Gaussian Splatting ICP SLAM</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 8 pages, 6 figures
    </div>
    <details class="paper-abstract">
      In this paper, we present a novel geometry-aware RGB-D Gaussian Splatting SLAM system, named G2S-ICP SLAM. The proposed method performs high-fidelity 3D reconstruction and robust camera pose tracking in real-time by representing each scene element using a Gaussian distribution constrained to the local tangent plane. This effectively models the local surface as a 2D Gaussian disk aligned with the underlying geometry, leading to more consistent depth interpretation across multiple viewpoints compared to conventional 3D ellipsoid-based representations with isotropic uncertainty. To integrate this representation into the SLAM pipeline, we embed the surface-aligned Gaussian disks into a Generalized ICP framework by introducing anisotropic covariance prior without altering the underlying registration formulation. Furthermore we propose a geometry-aware loss that supervises photometric, depth, and normal consistency. Our system achieves real-time operation while preserving both visual and geometric fidelity. Extensive experiments on the Replica and TUM-RGBD datasets demonstrate that G2S-ICP SLAM outperforms prior SLAM systems in terms of localization accuracy, reconstruction completeness, while maintaining the rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22128v2">PF3plat: Pose-Free Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 Accepted by ICML'25
    </div>
    <details class="paper-abstract">
      We consider the problem of novel view synthesis from unposed images in a single feed-forward. Our framework capitalizes on fast speed, scalability, and high-quality 3D reconstruction and view synthesis capabilities of 3DGS, where we further extend it to offer a practical solution that relaxes common assumptions such as dense image views, accurate camera poses, and substantial image overlaps. We achieve this through identifying and addressing unique challenges arising from the use of pixel-aligned 3DGS: misaligned 3D Gaussians across different views induce noisy or sparse gradients that destabilize training and hinder convergence, especially when above assumptions are not met. To mitigate this, we employ pre-trained monocular depth estimation and visual correspondence models to achieve coarse alignments of 3D Gaussians. We then introduce lightweight, learnable modules to refine depth and pose estimates from the coarse alignments, improving the quality of 3D reconstruction and novel view synthesis. Furthermore, the refined estimates are leveraged to estimate geometry confidence scores, which assess the reliability of 3D Gaussian centers and condition the prediction of Gaussian parameters accordingly. Extensive evaluations on large-scale real-world datasets demonstrate that PF3plat sets a new state-of-the-art across all benchmarks, supported by comprehensive ablation studies validating our design choices. project page: https://cvlab-kaist.github.io/PF3plat/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18231v1">PS-GS: Gaussian Splatting for Multi-View Photometric Stereo</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Integrating inverse rendering with multi-view photometric stereo (MVPS) yields more accurate 3D reconstructions than the inverse rendering approaches that rely on fixed environment illumination. However, efficient inverse rendering with MVPS remains challenging. To fill this gap, we introduce the Gaussian Splatting for Multi-view Photometric Stereo (PS-GS), which efficiently and jointly estimates the geometry, materials, and lighting of the object that is illuminated by diverse directional lights (multi-light). Our method first reconstructs a standard 2D Gaussian splatting model as the initial geometry. Based on the initialization model, it then proceeds with the deferred inverse rendering by the full rendering equation containing a lighting-computing multi-layer perceptron. During the whole optimization, we regularize the rendered normal maps by the uncalibrated photometric stereo estimated normals. We also propose the 2D Gaussian ray-tracing for single directional light to refine the incident lighting. The regularizations and the use of multi-view and multi-light images mitigate the ill-posed problem of inverse rendering. After optimization, the reconstructed object can be used for novel-view synthesis, relighting, and material and shape editing. Experiments on both synthetic and real datasets demonstrate that our method outperforms prior works in terms of reconstruction accuracy and computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13176v3">DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      Reconstructing clean, distractor-free 3D scenes from real-world captures remains a significant challenge, particularly in highly dynamic and cluttered settings such as egocentric videos. To tackle this problem, we introduce DeGauss, a simple and robust self-supervised framework for dynamic scene reconstruction based on a decoupled dynamic-static Gaussian Splatting design. DeGauss models dynamic elements with foreground Gaussians and static content with background Gaussians, using a probabilistic mask to coordinate their composition and enable independent yet complementary optimization. DeGauss generalizes robustly across a wide range of real-world scenarios, from casual image collections to long, dynamic egocentric videos, without relying on complex heuristics or extensive supervision. Experiments on benchmarks including NeRF-on-the-go, ADT, AEA, Hot3D, and EPIC-Fields demonstrate that DeGauss consistently outperforms existing methods, establishing a strong baseline for generalizable, distractor-free 3D reconstructionin highly dynamic, interaction-rich environments. Project page: https://batfacewayne.github.io/DeGauss.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18155v1">GeoAvatar: Adaptive Geometrical Gaussian Splatting for 3D Head Avatar</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 ICCV 2025, Project page: https://hahminlew.github.io/geoavatar/
    </div>
    <details class="paper-abstract">
      Despite recent progress in 3D head avatar generation, balancing identity preservation, i.e., reconstruction, with novel poses and expressions, i.e., animation, remains a challenge. Existing methods struggle to adapt Gaussians to varying geometrical deviations across facial regions, resulting in suboptimal quality. To address this, we propose GeoAvatar, a framework for adaptive geometrical Gaussian Splatting. GeoAvatar leverages Adaptive Pre-allocation Stage (APS), an unsupervised method that segments Gaussians into rigid and flexible sets for adaptive offset regularization. Then, based on mouth anatomy and dynamics, we introduce a novel mouth structure and the part-wise deformation strategy to enhance the animation fidelity of the mouth. Finally, we propose a regularization loss for precise rigging between Gaussians and 3DMM faces. Moreover, we release DynamicFace, a video dataset with highly expressive facial motions. Extensive experiments show the superiority of GeoAvatar compared to state-of-the-art methods in reconstruction and novel animation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18023v1">High-fidelity 3D Gaussian Inpainting: preserving multi-view consistency and photorealistic details</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Recent advancements in multi-view 3D reconstruction and novel-view synthesis, particularly through Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have greatly enhanced the fidelity and efficiency of 3D content creation. However, inpainting 3D scenes remains a challenging task due to the inherent irregularity of 3D structures and the critical need for maintaining multi-view consistency. In this work, we propose a novel 3D Gaussian inpainting framework that reconstructs complete 3D scenes by leveraging sparse inpainted views. Our framework incorporates an automatic Mask Refinement Process and region-wise Uncertainty-guided Optimization. Specifically, we refine the inpainting mask using a series of operations, including Gaussian scene filtering and back-projection, enabling more accurate localization of occluded regions and realistic boundary restoration. Furthermore, our Uncertainty-guided Fine-grained Optimization strategy, which estimates the importance of each region across multi-view images during training, alleviates multi-view inconsistencies and enhances the fidelity of fine details in the inpainted results. Comprehensive experiments conducted on diverse datasets demonstrate that our approach outperforms existing state-of-the-art methods in both visual quality and view consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18713v1">SaLF: Sparse Local Fields for Multi-Sensor Rendering in Real-Time</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      High-fidelity sensor simulation of light-based sensors such as cameras and LiDARs is critical for safe and accurate autonomy testing. Neural radiance field (NeRF)-based methods that reconstruct sensor observations via ray-casting of implicit representations have demonstrated accurate simulation of driving scenes, but are slow to train and render, hampering scale. 3D Gaussian Splatting (3DGS) has demonstrated faster training and rendering times through rasterization, but is primarily restricted to pinhole camera sensors, preventing usage for realistic multi-sensor autonomy evaluation. Moreover, both NeRF and 3DGS couple the representation with the rendering procedure (implicit networks for ray-based evaluation, particles for rasterization), preventing interoperability, which is key for general usage. In this work, we present Sparse Local Fields (SaLF), a novel volumetric representation that supports rasterization and raytracing. SaLF represents volumes as a sparse set of 3D voxel primitives, where each voxel is a local implicit field. SaLF has fast training (<30 min) and rendering capabilities (50+ FPS for camera and 600+ FPS LiDAR), has adaptive pruning and densification to easily handle large scenes, and can support non-pinhole cameras and spinning LiDARs. We demonstrate that SaLF has similar realism as existing self-driving sensor simulation methods while improving efficiency and enhancing capabilities, enabling more scalable simulation. https://waabi.ai/salf/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17032v2">TaoAvatar: Real-Time Lifelike Full-Body Talking Avatars for Augmented Reality via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 Accepted by CVPR 2025 (Highlight), project page: https://PixelAI-Team.github.io/TaoAvatar
    </div>
    <details class="paper-abstract">
      Realistic 3D full-body talking avatars hold great potential in AR, with applications ranging from e-commerce live streaming to holographic communication. Despite advances in 3D Gaussian Splatting (3DGS) for lifelike avatar creation, existing methods struggle with fine-grained control of facial expressions and body movements in full-body talking tasks. Additionally, they often lack sufficient details and cannot run in real-time on mobile devices. We present TaoAvatar, a high-fidelity, lightweight, 3DGS-based full-body talking avatar driven by various signals. Our approach starts by creating a personalized clothed human parametric template that binds Gaussians to represent appearances. We then pre-train a StyleUnet-based network to handle complex pose-dependent non-rigid deformation, which can capture high-frequency appearance details but is too resource-intensive for mobile devices. To overcome this, we "bake" the non-rigid deformations into a lightweight MLP-based network using a distillation technique and develop blend shapes to compensate for details. Extensive experiments show that TaoAvatar achieves state-of-the-art rendering quality while running in real-time across various devices, maintaining 90 FPS on high-definition stereo devices such as the Apple Vision Pro.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17336v1">Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 21 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Dynamic 4D Gaussian Splatting (4DGS) effectively extends the high-speed rendering capabilities of 3D Gaussian Splatting (3DGS) to represent volumetric videos. However, the large number of Gaussians, substantial temporal redundancies, and especially the absence of an entropy-aware compression framework result in large storage requirements. Consequently, this poses significant challenges for practical deployment, efficient edge-device processing, and data transmission. In this paper, we introduce a novel end-to-end RD-optimized compression framework tailored for 4DGS, aiming to enable flexible, high-fidelity rendering across varied computational platforms. Leveraging Fully Explicit Dynamic Gaussian Splatting (Ex4DGS), one of the state-of-the-art 4DGS methods, as our baseline, we start from the existing 3DGS compression methods for compatibility while effectively addressing additional challenges introduced by the temporal axis. In particular, instead of storing motion trajectories independently per point, we employ a wavelet transform to reflect the real-world smoothness prior, significantly enhancing storage efficiency. This approach yields significantly improved compression ratios and provides a user-controlled balance between compression efficiency and rendering quality. Extensive experiments demonstrate the effectiveness of our method, achieving up to 91x compression compared to the original Ex4DGS model while maintaining high visual fidelity. These results highlight the applicability of our framework for real-time dynamic scene rendering in diverse scenarios, from resource-constrained edge devices to high-performance environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06488v2">NVS-SQA: Exploring Self-Supervised Quality Representation Learning for Neurally Synthesized Scenes without References</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Neural View Synthesis (NVS), such as NeRF and 3D Gaussian Splatting, effectively creates photorealistic scenes from sparse viewpoints, typically evaluated by quality assessment methods like PSNR, SSIM, and LPIPS. However, these full-reference methods, which compare synthesized views to reference views, may not fully capture the perceptual quality of neurally synthesized scenes (NSS), particularly due to the limited availability of dense reference views. Furthermore, the challenges in acquiring human perceptual labels hinder the creation of extensive labeled datasets, risking model overfitting and reduced generalizability. To address these issues, we propose NVS-SQA, a NSS quality assessment method to learn no-reference quality representations through self-supervision without reliance on human labels. Traditional self-supervised learning predominantly relies on the "same instance, similar representation" assumption and extensive datasets. However, given that these conditions do not apply in NSS quality assessment, we employ heuristic cues and quality scores as learning objectives, along with a specialized contrastive pair preparation process to improve the effectiveness and efficiency of learning. The results show that NVS-SQA outperforms 17 no-reference methods by a large margin (i.e., on average 109.5% in SRCC, 98.6% in PLCC, and 91.5% in KRCC over the second best) and even exceeds 16 full-reference methods across all evaluation metrics (i.e., 22.9% in SRCC, 19.1% in PLCC, and 18.6% in KRCC over the second best).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16535v2">EarthCrafter: Scalable 3D Earth Generation via Dual-Sparse Latent Diffusion</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 Models and codes will be released at this https URL: https://github.com/whiteinblue/EarthCrafter
    </div>
    <details class="paper-abstract">
      Despite the remarkable developments achieved by recent 3D generation works, scaling these methods to geographic extents, such as modeling thousands of square kilometers of Earth's surface, remains an open challenge. We address this through a dual innovation in data infrastructure and model architecture. First, we introduce Aerial-Earth3D, the largest 3D aerial dataset to date, consisting of 50k curated scenes (each measuring 600m x 600m) captured across the U.S. mainland, comprising 45M multi-view Google Earth frames. Each scene provides pose-annotated multi-view images, depth maps, normals, semantic segmentation, and camera poses, with explicit quality control to ensure terrain diversity. Building on this foundation, we propose EarthCrafter, a tailored framework for large-scale 3D Earth generation via sparse-decoupled latent diffusion. Our architecture separates structural and textural generation: 1) Dual sparse 3D-VAEs compress high-resolution geometric voxels and textural 2D Gaussian Splats (2DGS) into compact latent spaces, largely alleviating the costly computation suffering from vast geographic scales while preserving critical information. 2) We propose condition-aware flow matching models trained on mixed inputs (semantics, images, or neither) to flexibly model latent geometry and texture features independently. Extensive experiments demonstrate that EarthCrafter performs substantially better in extremely large-scale generation. The framework further supports versatile applications, from semantic-guided urban layout generation to unconditional terrain synthesis, while maintaining geographic plausibility through our rich data priors from Aerial-Earth3D. Our project page is available at https://whiteinblue.github.io/earthcrafter/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21401v3">Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted by ICCV 2025, Code: https://github.com/zhirui-gao/Curve-Gaussian
    </div>
    <details class="paper-abstract">
      This paper presents an end-to-end framework for reconstructing 3D parametric curves directly from multi-view edge maps. Contrasting with existing two-stage methods that follow a sequential ``edge point cloud reconstruction and parametric curve fitting'' pipeline, our one-stage approach optimizes 3D parametric curves directly from 2D edge maps, eliminating error accumulation caused by the inherent optimization gap between disconnected stages. However, parametric curves inherently lack suitability for rendering-based multi-view optimization, necessitating a complementary representation that preserves their geometric properties while enabling differentiable rendering. We propose a novel bi-directional coupling mechanism between parametric curves and edge-oriented Gaussian components. This tight correspondence formulates a curve-aware Gaussian representation, \textbf{CurveGaussian}, that enables differentiable rendering of 3D curves, allowing direct optimization guided by multi-view evidence. Furthermore, we introduce a dynamically adaptive topology optimization framework during training to refine curve structures through linearization, merging, splitting, and pruning operations. Comprehensive evaluations on the ABC dataset and real-world benchmarks demonstrate our one-stage method's superiority over two-stage alternatives, particularly in producing cleaner and more robust reconstructions. Additionally, by directly optimizing parametric curves, our method significantly reduces the parameter count during training, achieving both higher efficiency and superior performance compared to existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15300v2">GCC: A 3DGS Inference Architecture with Gaussian-Wise and Cross-Stage Conditional Processing</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a leading neural rendering technique for high-fidelity view synthesis, prompting the development of dedicated 3DGS accelerators for mobile applications. Through in-depth analysis, we identify two major limitations in the conventional decoupled preprocessing-rendering dataflow adopted by existing accelerators: 1) a significant portion of preprocessed Gaussians are not used in rendering, and 2) the same Gaussian gets repeatedly loaded across different tile renderings, resulting in substantial computational and data movement overhead. To address these issues, we propose GCC, a novel accelerator designed for fast and energy-efficient 3DGS inference. At the dataflow level, GCC introduces: 1) cross-stage conditional processing, which interleaves preprocessing and rendering to dynamically skip unnecessary Gaussian preprocessing; and 2) Gaussian-wise rendering, ensuring that all rendering operations for a given Gaussian are completed before moving to the next, thereby eliminating duplicated Gaussian loading. We also propose an alpha-based boundary identification method to derive compact and accurate Gaussian regions, thereby reducing rendering costs. We implement our GCC accelerator in 28nm technology. Extensive experiments demonstrate that GCC significantly outperforms the state-of-the-art 3DGS inference accelerator, GSCore, in both performance and energy efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16535v1">EarthCrafter: Scalable 3D Earth Generation via Dual-Sparse Latent Diffusion</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Despite the remarkable developments achieved by recent 3D generation works, scaling these methods to geographic extents, such as modeling thousands of square kilometers of Earth's surface, remains an open challenge. We address this through a dual innovation in data infrastructure and model architecture. First, we introduce Aerial-Earth3D, the largest 3D aerial dataset to date, consisting of 50k curated scenes (each measuring 600m x 600m) captured across the U.S. mainland, comprising 45M multi-view Google Earth frames. Each scene provides pose-annotated multi-view images, depth maps, normals, semantic segmentation, and camera poses, with explicit quality control to ensure terrain diversity. Building on this foundation, we propose EarthCrafter, a tailored framework for large-scale 3D Earth generation via sparse-decoupled latent diffusion. Our architecture separates structural and textural generation: 1) Dual sparse 3D-VAEs compress high-resolution geometric voxels and textural 2D Gaussian Splats (2DGS) into compact latent spaces, largely alleviating the costly computation suffering from vast geographic scales while preserving critical information. 2) We propose condition-aware flow matching models trained on mixed inputs (semantics, images, or neither) to flexibly model latent geometry and texture features independently. Extensive experiments demonstrate that EarthCrafter performs substantially better in extremely large-scale generation. The framework further supports versatile applications, from semantic-guided urban layout generation to unconditional terrain synthesis, while maintaining geographic plausibility through our rich data priors from Aerial-Earth3D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05182v2">MGSR: 2D/3D Mutual-boosted Gaussian Splatting for High-fidelity Surface Reconstruction under Various Light Conditions</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted at ICCV'25
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) and surface reconstruction (SR) are essential tasks in 3D Gaussian Splatting (3D-GS). Despite recent progress, these tasks are often addressed independently, with GS-based rendering methods struggling under diverse light conditions and failing to produce accurate surfaces, while GS-based reconstruction methods frequently compromise rendering quality. This raises a central question: must rendering and reconstruction always involve a trade-off? To address this, we propose MGSR, a 2D/3D Mutual-boosted Gaussian splatting for Surface Reconstruction that enhances both rendering quality and 3D reconstruction accuracy. MGSR introduces two branches--one based on 2D-GS and the other on 3D-GS. The 2D-GS branch excels in surface reconstruction, providing precise geometry information to the 3D-GS branch. Leveraging this geometry, the 3D-GS branch employs a geometry-guided illumination decomposition module that captures reflected and transmitted components, enabling realistic rendering under varied light conditions. Using the transmitted component as supervision, the 2D-GS branch also achieves high-fidelity surface reconstruction. Throughout the optimization process, the 2D-GS and 3D-GS branches undergo alternating optimization, providing mutual supervision. Prior to this, each branch completes an independent warm-up phase, with an early stopping strategy implemented to reduce computational costs. We evaluate MGSR on a diverse set of synthetic and real-world datasets, at both object and scene levels, demonstrating strong performance in rendering and surface reconstruction. Code is available at https://github.com/TsingyuanChou/MGSR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13613v3">MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      4D Gaussian Splatting (4DGS) has recently emerged as a promising technique for capturing complex dynamic 3D scenes with high fidelity. It utilizes a 4D Gaussian representation and a GPU-friendly rasterizer, enabling rapid rendering speeds. Despite its advantages, 4DGS faces significant challenges, notably the requirement of millions of 4D Gaussians, each with extensive associated attributes, leading to substantial memory and storage cost. This paper introduces a memory-efficient framework for 4DGS. We streamline the color attribute by decomposing it into a per-Gaussian direct color component with only 3 parameters and a shared lightweight alternating current color predictor. This approach eliminates the need for spherical harmonics coefficients, which typically involve up to 144 parameters in classic 4DGS, thereby creating a memory-efficient 4D Gaussian representation. Furthermore, we introduce an entropy-constrained Gaussian deformation technique that uses a deformation field to expand the action range of each Gaussian and integrates an opacity-based entropy loss to limit the number of Gaussians, thus forcing our model to use as few Gaussians as possible to fit a dynamic scene well. With simple half-precision storage and zip compression, our framework achieves a storage reduction by approximately 190$\times$ and 125$\times$ on the Technicolor and Neural 3D Video datasets, respectively, compared to the original 4DGS. Meanwhile, it maintains comparable rendering speeds and scene representation quality, setting a new standard in the field. Code is available at https://github.com/Xinjie-Q/MEGA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16406v1">Sparse-View 3D Reconstruction: Recent Advances and Open Challenges</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 30 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Sparse-view 3D reconstruction is essential for applications in which dense image acquisition is impractical, such as robotics, augmented/virtual reality (AR/VR), and autonomous systems. In these settings, minimal image overlap prevents reliable correspondence matching, causing traditional methods, such as structure-from-motion (SfM) and multiview stereo (MVS), to fail. This survey reviews the latest advances in neural implicit models (e.g., NeRF and its regularized versions), explicit point-cloud-based approaches (e.g., 3D Gaussian Splatting), and hybrid frameworks that leverage priors from diffusion and vision foundation models (VFMs).We analyze how geometric regularization, explicit shape modeling, and generative inference are used to mitigate artifacts such as floaters and pose ambiguities in sparse-view settings. Comparative results on standard benchmarks reveal key trade-offs between the reconstruction accuracy, efficiency, and generalization. Unlike previous reviews, our survey provides a unified perspective on geometry-based, neural implicit, and generative (diffusion-based) methods. We highlight the persistent challenges in domain generalization and pose-free reconstruction and outline future directions for developing 3D-native generative priors and achieving real-time, unconstrained sparse-view reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16144v1">LongSplat: Online Generalizable 3D Gaussian Splatting from Long Sequence Images</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting achieves high-fidelity novel view synthesis, but its application to online long-sequence scenarios is still limited. Existing methods either rely on slow per-scene optimization or fail to provide efficient incremental updates, hindering continuous performance. In this paper, we propose LongSplat, an online real-time 3D Gaussian reconstruction framework designed for long-sequence image input. The core idea is a streaming update mechanism that incrementally integrates current-view observations while selectively compressing redundant historical Gaussians. Crucial to this mechanism is our Gaussian-Image Representation (GIR), a representation that encodes 3D Gaussian parameters into a structured, image-like 2D format. GIR simultaneously enables efficient fusion of current-view and historical Gaussians and identity-aware redundancy compression. These functions enable online reconstruction and adapt the model to long sequences without overwhelming memory or computational costs. Furthermore, we leverage an existing image compression method to guide the generation of more compact and higher-quality 3D Gaussians. Extensive evaluations demonstrate that LongSplat achieves state-of-the-art efficiency-quality trade-offs in real-time novel view synthesis, delivering real-time reconstruction while reducing Gaussian counts by 44\% compared to existing per-pixel Gaussian prediction methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17029v1">StreamME: Simplify 3D Gaussian Avatar within Live Stream</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 12 pages, 15 Figures
    </div>
    <details class="paper-abstract">
      We propose StreamME, a method focuses on fast 3D avatar reconstruction. The StreamME synchronously records and reconstructs a head avatar from live video streams without any pre-cached data, enabling seamless integration of the reconstructed appearance into downstream applications. This exceptionally fast training strategy, which we refer to as on-the-fly training, is central to our approach. Our method is built upon 3D Gaussian Splatting (3DGS), eliminating the reliance on MLPs in deformable 3DGS and relying solely on geometry, which significantly improves the adaptation speed to facial expression. To further ensure high efficiency in on-the-fly training, we introduced a simplification strategy based on primary points, which distributes the point clouds more sparsely across the facial surface, optimizing points number while maintaining rendering quality. Leveraging the on-the-fly training capabilities, our method protects the facial privacy and reduces communication bandwidth in VR system or online conference. Additionally, it can be directly applied to downstream application such as animation, toonify, and relighting. Please refer to our project page for more details: https://songluchuan.github.io/StreamME/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2507.15629v1">Gaussian Splatting with Discretized SDF for Relightable Assets</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has shown its detailed expressive ability and highly efficient rendering speed in the novel view synthesis (NVS) task. The application to inverse rendering still faces several challenges, as the discrete nature of Gaussian primitives makes it difficult to apply geometry constraints. Recent works introduce the signed distance field (SDF) as an extra continuous representation to regularize the geometry defined by Gaussian primitives. It improves the decomposition quality, at the cost of increasing memory usage and complicating training. Unlike these works, we introduce a discretized SDF to represent the continuous SDF in a discrete manner by encoding it within each Gaussian using a sampled value. This approach allows us to link the SDF with the Gaussian opacity through an SDF-to-opacity transformation, enabling rendering the SDF via splatting and avoiding the computational cost of ray marching.The key challenge is to regularize the discrete samples to be consistent with the underlying SDF, as the discrete representation can hardly apply the gradient-based constraints (\eg Eikonal loss). For this, we project Gaussians onto the zero-level set of SDF and enforce alignment with the surface from splatting, namely a projection-based consistency loss. Thanks to the discretized SDF, our method achieves higher relighting quality, while requiring no extra memory beyond GS and avoiding complex manually designed optimization. The experiments reveal that our method outperforms existing Gaussian-based inverse rendering methods. Our code is available at https://github.com/NK-CS-ZZL/DiscretizedSDF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15748v1">Appearance Harmonization via Bilateral Grid Prediction with Transformers for 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 10 pages, 3 figures, NeurIPS 2025 under review
    </div>
    <details class="paper-abstract">
      Modern camera pipelines apply extensive on-device processing, such as exposure adjustment, white balance, and color correction, which, while beneficial individually, often introduce photometric inconsistencies across views. These appearance variations violate multi-view consistency and degrade the quality of novel view synthesis. Joint optimization of scene representations and per-image appearance embeddings has been proposed to address this issue, but at the cost of increased computational complexity and slower training. In this work, we propose a transformer-based method that predicts spatially adaptive bilateral grids to correct photometric variations in a multi-view consistent manner, enabling robust cross-scene generalization without the need for scene-specific retraining. By incorporating the learned grids into the 3D Gaussian Splatting pipeline, we improve reconstruction quality while maintaining high training efficiency. Extensive experiments show that our approach outperforms or matches existing scene-specific optimization methods in reconstruction fidelity and convergence speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15690v1">DWTGS: Rethinking Frequency Regularization for Sparse-view 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 6 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Sparse-view 3D Gaussian Splatting (3DGS) presents significant challenges in reconstructing high-quality novel views, as it often overfits to the widely-varying high-frequency (HF) details of the sparse training views. While frequency regularization can be a promising approach, its typical reliance on Fourier transforms causes difficult parameter tuning and biases towards detrimental HF learning. We propose DWTGS, a framework that rethinks frequency regularization by leveraging wavelet-space losses that provide additional spatial supervision. Specifically, we supervise only the low-frequency (LF) LL subbands at multiple DWT levels, while enforcing sparsity on the HF HH subband in a self-supervised manner. Experiments across benchmarks show that DWTGS consistently outperforms Fourier-based counterparts, as this LF-centric strategy improves generalization and reduces HF hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15683v1">Hi^2-GSLoc: Dual-Hierarchical Gaussian-Specific Visual Relocalization for Remote Sensing</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 17 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Visual relocalization, which estimates the 6-degree-of-freedom (6-DoF) camera pose from query images, is fundamental to remote sensing and UAV applications. Existing methods face inherent trade-offs: image-based retrieval and pose regression approaches lack precision, while structure-based methods that register queries to Structure-from-Motion (SfM) models suffer from computational complexity and limited scalability. These challenges are particularly pronounced in remote sensing scenarios due to large-scale scenes, high altitude variations, and domain gaps of existing visual priors. To overcome these limitations, we leverage 3D Gaussian Splatting (3DGS) as a novel scene representation that compactly encodes both 3D geometry and appearance. We introduce $\mathrm{Hi}^2$-GSLoc, a dual-hierarchical relocalization framework that follows a sparse-to-dense and coarse-to-fine paradigm, fully exploiting the rich semantic information and geometric constraints inherent in Gaussian primitives. To handle large-scale remote sensing scenarios, we incorporate partitioned Gaussian training, GPU-accelerated parallel matching, and dynamic memory management strategies. Our approach consists of two stages: (1) a sparse stage featuring a Gaussian-specific consistent render-aware sampling strategy and landmark-guided detector for robust and accurate initial pose estimation, and (2) a dense stage that iteratively refines poses through coarse-to-fine dense rasterization matching while incorporating reliability verification. Through comprehensive evaluation on simulation data, public datasets, and real flight experiments, we demonstrate that our method delivers competitive localization accuracy, recall rate, and computational efficiency while effectively filtering unreliable pose estimates. The results confirm the effectiveness of our approach for practical remote sensing applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12137v2">AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      Modeling and rendering dynamic urban driving scenes is crucial for self-driving simulation. Current high-quality methods typically rely on costly manual object tracklet annotations, while self-supervised approaches fail to capture dynamic object motions accurately and decompose scenes properly, resulting in rendering artifacts. We introduce AD-GS, a novel self-supervised framework for high-quality free-viewpoint rendering of driving scenes from a single log. At its core is a novel learnable motion model that integrates locality-aware B-spline curves with global-aware trigonometric functions, enabling flexible yet precise dynamic object modeling. Rather than requiring comprehensive semantic labeling, AD-GS automatically segments scenes into objects and background with the simplified pseudo 2D segmentation, representing objects using dynamic Gaussians and bidirectional temporal visibility masks. Further, our model incorporates visibility reasoning and physically rigid regularization to enhance robustness. Extensive evaluations demonstrate that our annotation-free model significantly outperforms current state-of-the-art annotation-free methods and is competitive with annotation-dependent approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15629v1">Gaussian Splatting with Discretized SDF for Relightable Assets</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has shown its detailed expressive ability and highly efficient rendering speed in the novel view synthesis (NVS) task. The application to inverse rendering still faces several challenges, as the discrete nature of Gaussian primitives makes it difficult to apply geometry constraints. Recent works introduce the signed distance field (SDF) as an extra continuous representation to regularize the geometry defined by Gaussian primitives. It improves the decomposition quality, at the cost of increasing memory usage and complicating training. Unlike these works, we introduce a discretized SDF to represent the continuous SDF in a discrete manner by encoding it within each Gaussian using a sampled value. This approach allows us to link the SDF with the Gaussian opacity through an SDF-to-opacity transformation, enabling rendering the SDF via splatting and avoiding the computational cost of ray marching.The key challenge is to regularize the discrete samples to be consistent with the underlying SDF, as the discrete representation can hardly apply the gradient-based constraints (\eg Eikonal loss). For this, we project Gaussians onto the zero-level set of SDF and enforce alignment with the surface from splatting, namely a projection-based consistency loss. Thanks to the discretized SDF, our method achieves higher relighting quality, while requiring no extra memory beyond GS and avoiding complex manually designed optimization. The experiments reveal that our method outperforms existing Gaussian-based inverse rendering methods. Our code is available at https://github.com/NK-CS-ZZL/DiscretizedSDF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15602v1">SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Surface reconstruction and novel view rendering from sparse-view images are challenging. Signed Distance Function (SDF)-based methods struggle with fine details, while 3D Gaussian Splatting (3DGS)-based approaches lack global geometry coherence. We propose a novel hybrid method that combines the strengths of both approaches: SDF captures coarse geometry to enhance 3DGS-based rendering, while newly rendered images from 3DGS refine the details of SDF for accurate surface reconstruction. As a result, our method surpasses state-of-the-art approaches in surface reconstruction and novel view synthesis on the DTU and MobileBrick datasets. Code will be released at https://github.com/Gaozihui/SurfaceSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06060v2">VisualSpeaker: Visually-Guided 3D Avatar Lip Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Accepted in International Conference on Computer Vision (ICCV) Workshops
    </div>
    <details class="paper-abstract">
      Realistic, high-fidelity 3D facial animations are crucial for expressive avatar systems in human-computer interaction and accessibility. Although prior methods show promising quality, their reliance on the mesh domain limits their ability to fully leverage the rapid visual innovations seen in 2D computer vision and graphics. We propose VisualSpeaker, a novel method that bridges this gap using photorealistic differentiable rendering, supervised by visual speech recognition, for improved 3D facial animation. Our contribution is a perceptual lip-reading loss, derived by passing photorealistic 3D Gaussian Splatting avatar renders through a pre-trained Visual Automatic Speech Recognition model during training. Evaluation on the MEAD dataset demonstrates that VisualSpeaker improves both the standard Lip Vertex Error metric by 56.1% and the perceptual quality of the generated animations, while retaining the controllability of mesh-driven animation. This perceptual focus naturally supports accurate mouthings, essential cues that disambiguate similar manual signs in sign language avatars.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11061v2">Robust 3D-Masked Part-level Editing in 3D Gaussian Splatting with Regularized Score Distillation Sampling</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Recent advances in 3D neural representations and instance-level editing models have enabled the efficient creation of high-quality 3D content. However, achieving precise local 3D edits remains challenging, especially for Gaussian Splatting, due to inconsistent multi-view 2D part segmentations and inherently ambiguous nature of Score Distillation Sampling (SDS) loss. To address these limitations, we propose RoMaP, a novel local 3D Gaussian editing framework that enables precise and drastic part-level modifications. First, we introduce a robust 3D mask generation module with our 3D-Geometry Aware Label Prediction (3D-GALP), which uses spherical harmonics (SH) coefficients to model view-dependent label variations and soft-label property, yielding accurate and consistent part segmentations across viewpoints. Second, we propose a regularized SDS loss that combines the standard SDS loss with additional regularizers. In particular, an L1 anchor loss is introduced via our Scheduled Latent Mixing and Part (SLaMP) editing method, which generates high-quality part-edited 2D images and confines modifications only to the target region while preserving contextual coherence. Additional regularizers, such as Gaussian prior removal, further improve flexibility by allowing changes beyond the existing context, and robust 3D masking prevents unintended edits. Experimental results demonstrate that our RoMaP achieves state-of-the-art local 3D editing on both reconstructed and generated Gaussian scenes and objects qualitatively and quantitatively, making it possible for more robust and flexible part-level 3D Gaussian editing. Code is available at https://janeyeon.github.io/romap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15454v1">ObjectGS: Object-aware Scene Reconstruction and Scene Understanding via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting is renowned for its high-fidelity reconstructions and real-time novel view synthesis, yet its lack of semantic understanding limits object-level perception. In this work, we propose ObjectGS, an object-aware framework that unifies 3D scene reconstruction with semantic understanding. Instead of treating the scene as a unified whole, ObjectGS models individual objects as local anchors that generate neural Gaussians and share object IDs, enabling precise object-level reconstruction. During training, we dynamically grow or prune these anchors and optimize their features, while a one-hot ID encoding with a classification loss enforces clear semantic constraints. We show through extensive experiments that ObjectGS not only outperforms state-of-the-art methods on open-vocabulary and panoptic segmentation tasks, but also integrates seamlessly with applications like mesh extraction and scene editing. Project page: https://ruijiezhu94.github.io/ObjectGS_page
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08366v2">In-2-4D: Inbetweening from Two Single-View Images to 4D Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      We propose a new problem, In-2-4D, for generative 4D (i.e., 3D + motion) inbetweening from a minimalistic input setting: two single-view images capturing an object in two distinct motion states. Given two images representing the start and end states of an object in motion, our goal is to generate and reconstruct the motion in 4D. We utilize a video interpolation model to predict the motion, but large frame-to-frame motions can lead to ambiguous interpretations. To overcome this, we employ a hierarchical approach to identify keyframes that are visually close to the input states and show significant motion, then generate smooth fragments between them. For each fragment, we construct the 3D representation of the keyframe using Gaussian Splatting. The temporal frames within the fragment guide the motion, enabling their transformation into dynamic Gaussians through a deformation field. To improve temporal consistency and refine 3D motion, we expand the self-attention of multi-view diffusion across timesteps and apply rigid transformation regularization. Finally, we merge the independently generated 3D motion segments by interpolating boundary deformation fields and optimizing them to align with the guiding video, ensuring smooth and flicker-free transitions. Through extensive qualitative and quantitiave experiments as well as a user study, we show the effectiveness of our method and its components. The project page is available at https://in-2-4d.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15300v1">GCC: A 3DGS Inference Architecture with Gaussian-Wise and Cross-Stage Conditional Processing</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a leading neural rendering technique for high-fidelity view synthesis, prompting the development of dedicated 3DGS accelerators for mobile applications. Through in-depth analysis, we identify two major limitations in the conventional decoupled preprocessing-rendering dataflow adopted by existing accelerators: 1) a significant portion of preprocessed Gaussians are not used in rendering, and 2) the same Gaussian gets repeatedly loaded across different tile renderings, resulting in substantial computational and data movement overhead. To address these issues, we propose GCC, a novel accelerator designed for fast and energy-efficient 3DGS inference. At the dataflow level, GCC introduces: 1) cross-stage conditional processing, which interleaves preprocessing and rendering to dynamically skip unnecessary Gaussian preprocessing; and 2) Gaussian-wise rendering, ensuring that all rendering operations for a given Gaussian are completed before moving to the next, thereby eliminating duplicated Gaussian loading. We also propose an alpha-based boundary identification method to derive compact and accurate Gaussian regions, thereby reducing rendering costs. We implement our GCC accelerator in 28nm technology. Extensive experiments demonstrate that GCC significantly outperforms the state-of-the-art 3DGS inference accelerator, GSCore, in both performance and energy efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13891v2">PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      COLMAP-free 3D Gaussian Splatting (3D-GS) has recently attracted increasing attention due to its remarkable performance in reconstructing high-quality 3D scenes from unposed images or videos. However, it often struggles to handle scenes with complex camera trajectories as featured by drastic rotation and translation across adjacent camera views, leading to degraded estimation of camera poses and further local minima in joint optimization of camera poses and 3D-GS. We propose PCR-GS, an innovative COLMAP-free 3DGS technique that achieves superior 3D scene modeling and camera pose estimation via camera pose co-regularization. PCR-GS achieves regularization from two perspectives. The first is feature reprojection regularization which extracts view-robust DINO features from adjacent camera views and aligns their semantic information for camera pose regularization. The second is wavelet-based frequency regularization which exploits discrepancy in high-frequency details to further optimize the rotation matrix in camera poses. Extensive experiments over multiple real-world scenes show that the proposed PCR-GS achieves superior pose-free 3D-GS scene modeling under dramatic changes of camera trajectories.
    </details>
</div>
