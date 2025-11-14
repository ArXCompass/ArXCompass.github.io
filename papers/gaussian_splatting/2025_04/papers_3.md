# gaussian splatting - 2025_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02158v1">UAVTwin: Neural Digital Twins for UAVs using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      We present UAVTwin, a method for creating digital twins from real-world environments and facilitating data augmentation for training downstream models embedded in unmanned aerial vehicles (UAVs). Specifically, our approach focuses on synthesizing foreground components, such as various human instances in motion within complex scene backgrounds, from UAV perspectives. This is achieved by integrating 3D Gaussian Splatting (3DGS) for reconstructing backgrounds along with controllable synthetic human models that display diverse appearances and actions in multiple poses. To the best of our knowledge, UAVTwin is the first approach for UAV-based perception that is capable of generating high-fidelity digital twins based on 3DGS. The proposed work significantly enhances downstream models through data augmentation for real-world environments with multiple dynamic objects and significant appearance variations-both of which typically introduce artifacts in 3DGS-based modeling. To tackle these challenges, we propose a novel appearance modeling strategy and a mask refinement module to enhance the training of 3D Gaussian Splatting. We demonstrate the high quality of neural rendering by achieving a 1.23 dB improvement in PSNR compared to recent methods. Furthermore, we validate the effectiveness of data augmentation by showing a 2.5% to 13.7% improvement in mAP for the human detection task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20522v2">MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 CVPR 2025; Project page:https://maskgaussian.github.io/
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has demonstrated remarkable performance in novel view synthesis and real-time rendering, the high memory consumption due to the use of millions of Gaussians limits its practicality. To mitigate this issue, improvements have been made by pruning unnecessary Gaussians, either through a hand-crafted criterion or by using learned masks. However, these methods deterministically remove Gaussians based on a snapshot of the pruning moment, leading to sub-optimized reconstruction performance from a long-term perspective. To address this issue, we introduce MaskGaussian, which models Gaussians as probabilistic entities rather than permanently removing them, and utilize them according to their probability of existence. To achieve this, we propose a masked-rasterization technique that enables unused yet probabilistically existing Gaussians to receive gradients, allowing for dynamic assessment of their contribution to the evolving scene and adjustment of their probability of existence. Hence, the importance of Gaussians iteratively changes and the pruned Gaussians are selected diversely. Extensive experiments demonstrate the superiority of the proposed method in achieving better rendering quality with fewer Gaussians than previous pruning methods, pruning over 60% of Gaussians on average with only a 0.02 PSNR decline. Our code can be found at: https://github.com/kaikai23/MaskGaussian
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02045v1">WorldPrompter: Traversable Text-to-Scene Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Scene-level 3D generation is a challenging research topic, with most existing methods generating only partial scenes and offering limited navigational freedom. We introduce WorldPrompter, a novel generative pipeline for synthesizing traversable 3D scenes from text prompts. We leverage panoramic videos as an intermediate representation to model the 360{\deg} details of a scene. WorldPrompter incorporates a conditional 360{\deg} panoramic video generator, capable of producing a 128-frame video that simulates a person walking through and capturing a virtual environment. The resulting video is then reconstructed as Gaussian splats by a fast feedforward 3D reconstructor, enabling a true walkable experience within the 3D scene. Experiments demonstrate that our panoramic video generation model achieves convincing view consistency across frames, enabling high-quality panoramic Gaussian splat reconstruction and facilitating traversal over an area of the scene. Qualitative and quantitative results also show it outperforms the state-of-the-art 360{\deg} video generators and 3D scene generation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01960v1">Diffusion-Guided Gaussian Splatting for Large-Scale Unconstrained 3D Reconstruction and Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 WACV ULTRRA Workshop 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3DGS) and Neural Radiance Fields (NeRF) have achieved impressive results in real-time 3D reconstruction and novel view synthesis. However, these methods struggle in large-scale, unconstrained environments where sparse and uneven input coverage, transient occlusions, appearance variability, and inconsistent camera settings lead to degraded quality. We propose GS-Diff, a novel 3DGS framework guided by a multi-view diffusion model to address these limitations. By generating pseudo-observations conditioned on multi-view inputs, our method transforms under-constrained 3D reconstruction problems into well-posed ones, enabling robust optimization even with sparse data. GS-Diff further integrates several enhancements, including appearance embedding, monocular depth priors, dynamic object modeling, anisotropy regularization, and advanced rasterization techniques, to tackle geometric and photometric challenges in real-world settings. Experiments on four benchmarks demonstrate that GS-Diff consistently outperforms state-of-the-art baselines by significant margins.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01844v1">BOGausS: Better Optimized Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) proposes an efficient solution for novel view synthesis. Its framework provides fast and high-fidelity rendering. Although less complex than other solutions such as Neural Radiance Fields (NeRF), there are still some challenges building smaller models without sacrificing quality. In this study, we perform a careful analysis of 3DGS training process and propose a new optimization methodology. Our Better Optimized Gaussian Splatting (BOGausS) solution is able to generate models up to ten times lighter than the original 3DGS with no quality degradation, thus significantly boosting the performance of Gaussian Splatting compared to the state of the art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09227v3">DreamScape: 3D Scene Creation via Gaussian Splatting joint Correlation Modeling</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Recent advances in text-to-3D creation integrate the potent prior of Diffusion Models from text-to-image generation into 3D domain. Nevertheless, generating 3D scenes with multiple objects remains challenging. Therefore, we present DreamScape, a method for generating 3D scenes from text. Utilizing Gaussian Splatting for 3D representation, DreamScape introduces 3D Gaussian Guide that encodes semantic primitives, spatial transformations and relationships from text using LLMs, enabling local-to-global optimization. Progressive scale control is tailored during local object generation, addressing training instability issue arising from simple blending in the global optimization stage. Collision relationships between objects are modeled at the global level to mitigate biases in LLMs priors, ensuring physical correctness. Additionally, to generate pervasive objects like rain and snow distributed extensively across the scene, we design specialized sparse initialization and densification strategy. Experiments demonstrate that DreamScape achieves state-of-the-art performance, enabling high-fidelity, controllable 3D scene generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01732v1">FIORD: A Fisheye Indoor-Outdoor Dataset with LIDAR Ground Truth for 3D Scene Reconstruction and Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 SCIA 2025
    </div>
    <details class="paper-abstract">
      The development of large-scale 3D scene reconstruction and novel view synthesis methods mostly rely on datasets comprising perspective images with narrow fields of view (FoV). While effective for small-scale scenes, these datasets require large image sets and extensive structure-from-motion (SfM) processing, limiting scalability. To address this, we introduce a fisheye image dataset tailored for scene reconstruction tasks. Using dual 200-degree fisheye lenses, our dataset provides full 360-degree coverage of 5 indoor and 5 outdoor scenes. Each scene has sparse SfM point clouds and precise LIDAR-derived dense point clouds that can be used as geometric ground-truth, enabling robust benchmarking under challenging conditions such as occlusions and reflections. While the baseline experiments focus on vanilla Gaussian Splatting and NeRF based Nerfacto methods, the dataset supports diverse approaches for scene reconstruction, novel view synthesis, and image-based rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01647v1">FlowR: Flowing from Sparse to Dense 3D Reconstructions</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 Project page is available at https://tobiasfshr.github.io/pub/flowr
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting enables high-quality novel view synthesis (NVS) at real-time frame rates. However, its quality drops sharply as we depart from the training views. Thus, dense captures are needed to match the high-quality expectations of some applications, e.g. Virtual Reality (VR). However, such dense captures are very laborious and expensive to obtain. Existing works have explored using 2D generative models to alleviate this requirement by distillation or generating additional training views. These methods are often conditioned only on a handful of reference input views and thus do not fully exploit the available 3D information, leading to inconsistent generation results and reconstruction artifacts. To tackle this problem, we propose a multi-view, flow matching model that learns a flow to connect novel view renderings from possibly sparse reconstructions to renderings that we expect from dense reconstructions. This enables augmenting scene captures with novel, generated views to improve reconstruction quality. Our model is trained on a novel dataset of 3.6M image pairs and can process up to 45 views at 540x960 resolution (91K tokens) on one H100 GPU in a single forward pass. Our pipeline consistently improves NVS in sparse- and dense-view scenarios, leading to higher-quality reconstructions than prior works across multiple, widely-used NVS benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01619v1">3DBonsai: Structure-Aware Bonsai Modeling Using Conditioned 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 Accepted by ICME 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in text-to-3D generation have shown remarkable results by leveraging 3D priors in combination with 2D diffusion. However, previous methods utilize 3D priors that lack detailed and complex structural information, limiting them to generating simple objects and presenting challenges for creating intricate structures such as bonsai. In this paper, we propose 3DBonsai, a novel text-to-3D framework for generating 3D bonsai with complex structures. Technically, we first design a trainable 3D space colonization algorithm to produce bonsai structures, which are then enhanced through random sampling and point cloud augmentation to serve as the 3D Gaussian priors. We introduce two bonsai generation pipelines with distinct structural levels: fine structure conditioned generation, which initializes 3D Gaussians using a 3D structure prior to produce detailed and complex bonsai, and coarse structure conditioned generation, which employs a multi-view structure consistency module to align 2D and 3D structures. Moreover, we have compiled a unified 2D and 3D Chinese-style bonsai dataset. Our experimental results demonstrate that 3DBonsai significantly outperforms existing methods, providing a new benchmark for structure-aware 3D bonsai generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01559v1">RealityAvatar: Towards Realistic Loose Clothing Modeling in Animatable 3D Gaussian Avatars</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Modeling animatable human avatars from monocular or multi-view videos has been widely studied, with recent approaches leveraging neural radiance fields (NeRFs) or 3D Gaussian Splatting (3DGS) achieving impressive results in novel-view and novel-pose synthesis. However, existing methods often struggle to accurately capture the dynamics of loose clothing, as they primarily rely on global pose conditioning or static per-frame representations, leading to oversmoothing and temporal inconsistencies in non-rigid regions. To address this, We propose RealityAvatar, an efficient framework for high-fidelity digital human modeling, specifically targeting loosely dressed avatars. Our method leverages 3D Gaussian Splatting to capture complex clothing deformations and motion dynamics while ensuring geometric consistency. By incorporating a motion trend module and a latentbone encoder, we explicitly model pose-dependent deformations and temporal variations in clothing behavior. Extensive experiments on benchmark datasets demonstrate the effectiveness of our approach in capturing fine-grained clothing deformations and motion-driven shape variations. Our method significantly enhances structural fidelity and perceptual quality in dynamic human reconstruction, particularly in non-rigid regions, while achieving better consistency across temporal frames.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01512v1">High-fidelity 3D Object Generation from Single Image with RGBN-Volume Gaussian Reconstruction Model</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Recently single-view 3D generation via Gaussian splatting has emerged and developed quickly. They learn 3D Gaussians from 2D RGB images generated from pre-trained multi-view diffusion (MVD) models, and have shown a promising avenue for 3D generation through a single image. Despite the current progress, these methods still suffer from the inconsistency jointly caused by the geometric ambiguity in the 2D images, and the lack of structure of 3D Gaussians, leading to distorted and blurry 3D object generation. In this paper, we propose to fix these issues by GS-RGBN, a new RGBN-volume Gaussian Reconstruction Model designed to generate high-fidelity 3D objects from single-view images. Our key insight is a structured 3D representation can simultaneously mitigate the afore-mentioned two issues. To this end, we propose a novel hybrid Voxel-Gaussian representation, where a 3D voxel representation contains explicit 3D geometric information, eliminating the geometric ambiguity from 2D images. It also structures Gaussians during learning so that the optimization tends to find better local optima. Our 3D voxel representation is obtained by a fusion module that aligns RGB features and surface normal features, both of which can be estimated from 2D images. Extensive experiments demonstrate the superiority of our methods over prior works in terms of high-quality reconstruction results, robust generalization, and good efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01503v1">Luminance-GS: Adapting 3D Gaussian Splatting to Challenging Lighting Conditions with View-Adaptive Curve Adjustment</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 CVPR 2025, project page: https://cuiziteng.github.io/Luminance_GS_web/
    </div>
    <details class="paper-abstract">
      Capturing high-quality photographs under diverse real-world lighting conditions is challenging, as both natural lighting (e.g., low-light) and camera exposure settings (e.g., exposure time) significantly impact image quality. This challenge becomes more pronounced in multi-view scenarios, where variations in lighting and image signal processor (ISP) settings across viewpoints introduce photometric inconsistencies. Such lighting degradations and view-dependent variations pose substantial challenges to novel view synthesis (NVS) frameworks based on Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). To address this, we introduce Luminance-GS, a novel approach to achieving high-quality novel view synthesis results under diverse challenging lighting conditions using 3DGS. By adopting per-view color matrix mapping and view-adaptive curve adjustments, Luminance-GS achieves state-of-the-art (SOTA) results across various lighting conditions -- including low-light, overexposure, and varying exposure -- while not altering the original 3DGS explicit representation. Compared to previous NeRF- and 3DGS-based baselines, Luminance-GS provides real-time rendering speed with improved reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01358v1">3D Gaussian Inverse Rendering with Approximated Global Illumination</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting shows great potential in reconstructing photo-realistic 3D scenes. However, these methods typically bake illumination into their representations, limiting their use for physically-based rendering and scene editing. Although recent inverse rendering approaches aim to decompose scenes into material and lighting components, they often rely on simplifying assumptions that fail when editing. We present a novel approach that enables efficient global illumination for 3D Gaussians Splatting through screen-space ray tracing. Our key insight is that a substantial amount of indirect light can be traced back to surfaces visible within the current view frustum. Leveraging this observation, we augment the direct shading computed by 3D Gaussians with Monte-Carlo screen-space ray-tracing to capture one-bounce indirect illumination. In this way, our method enables realistic global illumination without sacrificing the computational efficiency and editability benefits of 3D Gaussians. Through experiments, we show that the screen-space approximation we utilize allows for indirect illumination and supports real-time rendering and editing. Code, data, and models will be made available at our project page: https://wuzirui.github.io/gs-ssr.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2504.00773v1">DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 Accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian splatting (3DGS) has gained considerable attentions in the field of novel view synthesis due to its fast performance while yielding the excellent image quality. However, 3DGS in sparse-view settings (e.g., three-view inputs) often faces with the problem of overfitting to training views, which significantly drops the visual quality of novel view images. Many existing approaches have tackled this issue by using strong priors, such as 2D generative contextual information and external depth signals. In contrast, this paper introduces a prior-free method, so-called DropGaussian, with simple changes in 3D Gaussian splatting. Specifically, we randomly remove Gaussians during the training process in a similar way of dropout, which allows non-excluded Gaussians to have larger gradients while improving their visibility. This makes the remaining Gaussians to contribute more to the optimization process for rendering with sparse input views. Such simple operation effectively alleviates the overfitting problem and enhances the quality of novel view synthesis. By simply applying DropGaussian to the original 3DGS framework, we can achieve the competitive performance with existing prior-based 3DGS methods in sparse-view settings of benchmark datasets without any additional complexity. The code and model are publicly available at: https://github.com/DCVL-3D/DropGaussian release.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.01844v1">BOGausS: Better Optimized Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) proposes an efficient solution for novel view synthesis. Its framework provides fast and high-fidelity rendering. Although less complex than other solutions such as Neural Radiance Fields (NeRF), there are still some challenges building smaller models without sacrificing quality. In this study, we perform a careful analysis of 3DGS training process and propose a new optimization methodology. Our Better Optimized Gaussian Splatting (BOGausS) solution is able to generate models up to ten times lighter than the original 3DGS with no quality degradation, thus significantly boosting the performance of Gaussian Splatting compared to the state of the art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00665v1">Monocular and Generalizable Gaussian Talking Head Animation</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      In this work, we introduce Monocular and Generalizable Gaussian Talking Head Animation (MGGTalk), which requires monocular datasets and generalizes to unseen identities without personalized re-training. Compared with previous 3D Gaussian Splatting (3DGS) methods that requires elusive multi-view datasets or tedious personalized learning/inference, MGGtalk enables more practical and broader applications. However, in the absence of multi-view and personalized training data, the incompleteness of geometric and appearance information poses a significant challenge. To address these challenges, MGGTalk explores depth information to enhance geometric and facial symmetry characteristics to supplement both geometric and appearance features. Initially, based on the pixel-wise geometric information obtained from depth estimation, we incorporate symmetry operations and point cloud filtering techniques to ensure a complete and precise position parameter for 3DGS. Subsequently, we adopt a two-stage strategy with symmetric priors for predicting the remaining 3DGS parameters. We begin by predicting Gaussian parameters for the visible facial regions of the source image. These parameters are subsequently utilized to improve the prediction of Gaussian parameters for the non-visible regions. Extensive experiments demonstrate that MGGTalk surpasses previous state-of-the-art methods, achieving superior performance across various metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00525v1">Robust LiDAR-Camera Calibration with 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted in IEEE Robotics and Automation Letters. Code available at: https://github.com/ShuyiZhou495/RobustCalibration
    </div>
    <details class="paper-abstract">
      LiDAR-camera systems have become increasingly popular in robotics recently. A critical and initial step in integrating the LiDAR and camera data is the calibration of the LiDAR-camera system. Most existing calibration methods rely on auxiliary target objects, which often involve complex manual operations, whereas targetless methods have yet to achieve practical effectiveness. Recognizing that 2D Gaussian Splatting (2DGS) can reconstruct geometric information from camera image sequences, we propose a calibration method that estimates LiDAR-camera extrinsic parameters using geometric constraints. The proposed method begins by reconstructing colorless 2DGS using LiDAR point clouds. Subsequently, we update the colors of the Gaussian splats by minimizing the photometric loss. The extrinsic parameters are optimized during this process. Additionally, we address the limitations of the photometric loss by incorporating the reprojection and triangulation losses, thereby enhancing the calibration robustness and accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00437v1">ADGaussian: Generalizable Gaussian Splatting for Autonomous Driving with Multi-modal Inputs</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 The project page can be found at https://maggiesong7.github.io/research/ADGaussian/
    </div>
    <details class="paper-abstract">
      We present a novel approach, termed ADGaussian, for generalizable street scene reconstruction. The proposed method enables high-quality rendering from single-view input. Unlike prior Gaussian Splatting methods that primarily focus on geometry refinement, we emphasize the importance of joint optimization of image and depth features for accurate Gaussian prediction. To this end, we first incorporate sparse LiDAR depth as an additional input modality, formulating the Gaussian prediction process as a joint learning framework of visual information and geometric clue. Furthermore, we propose a multi-modal feature matching strategy coupled with a multi-scale Gaussian decoding model to enhance the joint refinement of multi-modal features, thereby enabling efficient multi-modal Gaussian learning. Extensive experiments on two large-scale autonomous driving datasets, Waymo and KITTI, demonstrate that our ADGaussian achieves state-of-the-art performance and exhibits superior zero-shot generalization capabilities in novel-view shifting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00387v1">Scene4U: Hierarchical Layered 3D Scene Reconstruction from Single Panoramic Image for Your Immerse Exploration</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 CVPR 2025, 11 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The reconstruction of immersive and realistic 3D scenes holds significant practical importance in various fields of computer vision and computer graphics. Typically, immersive and realistic scenes should be free from obstructions by dynamic objects, maintain global texture consistency, and allow for unrestricted exploration. The current mainstream methods for image-driven scene construction involves iteratively refining the initial image using a moving virtual camera to generate the scene. However, previous methods struggle with visual discontinuities due to global texture inconsistencies under varying camera poses, and they frequently exhibit scene voids caused by foreground-background occlusions. To this end, we propose a novel layered 3D scene reconstruction framework from panoramic image, named Scene4U. Specifically, Scene4U integrates an open-vocabulary segmentation model with a large language model to decompose a real panorama into multiple layers. Then, we employs a layered repair module based on diffusion model to restore occluded regions using visual cues and depth information, generating a hierarchical representation of the scene. The multi-layer panorama is then initialized as a 3D Gaussian Splatting representation, followed by layered optimization, which ultimately produces an immersive 3D scene with semantic and structural consistency that supports free exploration. Scene4U outperforms state-of-the-art method, improving by 24.24% in LPIPS and 24.40% in BRISQUE, while also achieving the fastest training speed. Additionally, to demonstrate the robustness of Scene4U and allow users to experience immersive scenes from various landmarks, we build WorldVista3D dataset for 3D scene reconstruction, which contains panoramic images of globally renowned sites. The implementation code and dataset will be released at https://github.com/LongHZ140516/Scene4U .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06897v2">ActiveGAMER: Active GAussian Mapping through Efficient Rendering</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted to CVPR2025
    </div>
    <details class="paper-abstract">
      We introduce ActiveGAMER, an active mapping system that utilizes 3D Gaussian Splatting (3DGS) to achieve high-quality, real-time scene mapping and exploration. Unlike traditional NeRF-based methods, which are computationally demanding and restrict active mapping performance, our approach leverages the efficient rendering capabilities of 3DGS, allowing effective and efficient exploration in complex environments. The core of our system is a rendering-based information gain module that dynamically identifies the most informative viewpoints for next-best-view planning, enhancing both geometric and photometric reconstruction accuracy. ActiveGAMER also integrates a carefully balanced framework, combining coarse-to-fine exploration, post-refinement, and a global-local keyframe selection strategy to maximize reconstruction completeness and fidelity. Our system autonomously explores and reconstructs environments with state-of-the-art geometric and photometric accuracy and completeness, significantly surpassing existing approaches in both aspects. Extensive evaluations on benchmark datasets such as Replica and MP3D highlight ActiveGAMER's effectiveness in active mapping tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21442v2">RainyGS: Efficient Rain Synthesis with Physically-Based Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      We consider the problem of adding dynamic rain effects to in-the-wild scenes in a physically-correct manner. Recent advances in scene modeling have made significant progress, with NeRF and 3DGS techniques emerging as powerful tools for reconstructing complex scenes. However, while effective for novel view synthesis, these methods typically struggle with challenging scene editing tasks, such as physics-based rain simulation. In contrast, traditional physics-based simulations can generate realistic rain effects, such as raindrops and splashes, but they often rely on skilled artists to carefully set up high-fidelity scenes. This process lacks flexibility and scalability, limiting its applicability to broader, open-world environments. In this work, we introduce RainyGS, a novel approach that leverages the strengths of both physics-based modeling and 3DGS to generate photorealistic, dynamic rain effects in open-world scenes with physical accuracy. At the core of our method is the integration of physically-based raindrop and shallow water simulation techniques within the fast 3DGS rendering framework, enabling realistic and efficient simulations of raindrop behavior, splashes, and reflections. Our method supports synthesizing rain effects at over 30 fps, offering users flexible control over rain intensity -- from light drizzles to heavy downpours. We demonstrate that RainyGS performs effectively for both real-world outdoor scenes and large-scale driving scenarios, delivering more photorealistic and physically-accurate rain effects compared to state-of-the-art methods. Project page can be found at https://pku-vcl-geometry.github.io/RainyGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19671v2">GaussianRoom: Improving 3D Gaussian Splatting with SDF Guidance and Monocular Cues for Indoor Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Embodied intelligence requires precise reconstruction and rendering to simulate large-scale real-world data. Although 3D Gaussian Splatting (3DGS) has recently demonstrated high-quality results with real-time performance, it still faces challenges in indoor scenes with large, textureless regions, resulting in incomplete and noisy reconstructions due to poor point cloud initialization and underconstrained optimization. Inspired by the continuity of signed distance field (SDF), which naturally has advantages in modeling surfaces, we propose a unified optimization framework that integrates neural signed distance fields (SDFs) with 3DGS for accurate geometry reconstruction and real-time rendering. This framework incorporates a neural SDF field to guide the densification and pruning of Gaussians, enabling Gaussians to model scenes accurately even with poor initialized point clouds. Simultaneously, the geometry represented by Gaussians improves the efficiency of the SDF field by piloting its point sampling. Additionally, we introduce two regularization terms based on normal and edge priors to resolve geometric ambiguities in textureless areas and enhance detail accuracy. Extensive experiments in ScanNet and ScanNet++ show that our method achieves state-of-the-art performance in both surface reconstruction and novel view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03526v2">Feed-Forward Bullet-Time Reconstruction of Dynamic Scenes from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Project website: https://research.nvidia.com/labs/toronto-ai/bullet-timer/
    </div>
    <details class="paper-abstract">
      Recent advancements in static feed-forward scene reconstruction have demonstrated significant progress in high-quality novel view synthesis. However, these models often struggle with generalizability across diverse environments and fail to effectively handle dynamic content. We present BTimer (short for BulletTimer), the first motion-aware feed-forward model for real-time reconstruction and novel view synthesis of dynamic scenes. Our approach reconstructs the full scene in a 3D Gaussian Splatting representation at a given target ('bullet') timestamp by aggregating information from all the context frames. Such a formulation allows BTimer to gain scalability and generalization by leveraging both static and dynamic scene datasets. Given a casual monocular dynamic video, BTimer reconstructs a bullet-time scene within 150ms while reaching state-of-the-art performance on both static and dynamic scene datasets, even compared with optimization-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.24270v2">Visual Acoustic Fields</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Objects produce different sounds when hit, and humans can intuitively infer how an object might sound based on its appearance and material properties. Inspired by this intuition, we propose Visual Acoustic Fields, a framework that bridges hitting sounds and visual signals within a 3D space using 3D Gaussian Splatting (3DGS). Our approach features two key modules: sound generation and sound localization. The sound generation module leverages a conditional diffusion model, which takes multiscale features rendered from a feature-augmented 3DGS to generate realistic hitting sounds. Meanwhile, the sound localization module enables querying the 3D scene, represented by the feature-augmented 3DGS, to localize hitting positions based on the sound sources. To support this framework, we introduce a novel pipeline for collecting scene-level visual-sound sample pairs, achieving alignment between captured images, impact locations, and corresponding sounds. To the best of our knowledge, this is the first dataset to connect visual and acoustic signals in a 3D context. Extensive experiments on our dataset demonstrate the effectiveness of Visual Acoustic Fields in generating plausible impact sounds and accurately localizing impact sources. Our project page is at https://yuelei0428.github.io/projects/Visual-Acoustic-Fields/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10437v2">4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 CVPR 2025. Project Page: https://4d-langsplat.github.io
    </div>
    <details class="paper-abstract">
      Learning 4D language fields to enable time-sensitive, open-ended language queries in dynamic scenes is essential for many real-world applications. While LangSplat successfully grounds CLIP features into 3D Gaussian representations, achieving precision and efficiency in 3D static scenes, it lacks the ability to handle dynamic 4D fields as CLIP, designed for static image-text tasks, cannot capture temporal dynamics in videos. Real-world environments are inherently dynamic, with object semantics evolving over time. Building a precise 4D language field necessitates obtaining pixel-aligned, object-wise video features, which current vision models struggle to achieve. To address these challenges, we propose 4D LangSplat, which learns 4D language fields to handle time-agnostic or time-sensitive open-vocabulary queries in dynamic scenes efficiently. 4D LangSplat bypasses learning the language field from vision features and instead learns directly from text generated from object-wise video captions via Multimodal Large Language Models (MLLMs). Specifically, we propose a multimodal object-wise video prompting method, consisting of visual and text prompts that guide MLLMs to generate detailed, temporally consistent, high-quality captions for objects throughout a video. These captions are encoded using a Large Language Model into high-quality sentence embeddings, which then serve as pixel-aligned, object-specific feature supervision, facilitating open-vocabulary text queries through shared embedding spaces. Recognizing that objects in 4D scenes exhibit smooth transitions across states, we further propose a status deformable network to model these continuous changes over time effectively. Our results across multiple benchmarks demonstrate that 4D LangSplat attains precise and efficient results for both time-sensitive and time-agnostic open-vocabulary queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00370v2">Scalable Real2Sim: Physics-Aware Asset Generation Via Robotic Pick-and-Place Setups</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Website: https://scalable-real2sim.github.io/
    </div>
    <details class="paper-abstract">
      Simulating object dynamics from real-world perception shows great promise for digital twins and robotic manipulation but often demands labor-intensive measurements and expertise. We present a fully automated Real2Sim pipeline that generates simulation-ready assets for real-world objects through robotic interaction. Using only a robot's joint torque sensors and an external camera, the pipeline identifies visual geometry, collision geometry, and physical properties such as inertial parameters. Our approach introduces a general method for extracting high-quality, object-centric meshes from photometric reconstruction techniques (e.g., NeRF, Gaussian Splatting) by employing alpha-transparent training while explicitly distinguishing foreground occlusions from background subtraction. We validate the full pipeline through extensive experiments, demonstrating its effectiveness across diverse objects. By eliminating the need for manual intervention or environment modifications, our pipeline can be integrated directly into existing pick-and-place setups, enabling scalable and efficient dataset creation. Project page (with code and data): https://scalable-real2sim.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22876v2">VizFlyt: Perception-centric Pedagogical Framework For Autonomous Aerial Robots</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted at ICRA 2025. Projected Page: https://pear.wpi.edu/research/vizflyt.html
    </div>
    <details class="paper-abstract">
      Autonomous aerial robots are becoming commonplace in our lives. Hands-on aerial robotics courses are pivotal in training the next-generation workforce to meet the growing market demands. Such an efficient and compelling course depends on a reliable testbed. In this paper, we present VizFlyt, an open-source perception-centric Hardware-In-The-Loop (HITL) photorealistic testing framework for aerial robotics courses. We utilize pose from an external localization system to hallucinate real-time and photorealistic visual sensors using 3D Gaussian Splatting. This enables stress-free testing of autonomy algorithms on aerial robots without the risk of crashing into obstacles. We achieve over 100Hz of system update rate. Lastly, we build upon our past experiences of offering hands-on aerial robotics courses and propose a new open-source and open-hardware curriculum based on VizFlyt for the future. We test our framework on various course projects in real-world HITL experiments and present the results showing the efficacy of such a system and its large potential use cases. Code, datasets, hardware guides and demo videos are available at https://pear.wpi.edu/research/vizflyt.html
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00773v1">DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian splatting (3DGS) has gained considerable attentions in the field of novel view synthesis due to its fast performance while yielding the excellent image quality. However, 3DGS in sparse-view settings (e.g., three-view inputs) often faces with the problem of overfitting to training views, which significantly drops the visual quality of novel view images. Many existing approaches have tackled this issue by using strong priors, such as 2D generative contextual information and external depth signals. In contrast, this paper introduces a prior-free method, so-called DropGaussian, with simple changes in 3D Gaussian splatting. Specifically, we randomly remove Gaussians during the training process in a similar way of dropout, which allows non-excluded Gaussians to have larger gradients while improving their visibility. This makes the remaining Gaussians to contribute more to the optimization process for rendering with sparse input views. Such simple operation effectively alleviates the overfitting problem and enhances the quality of novel view synthesis. By simply applying DropGaussian to the original 3DGS framework, we can achieve the competitive performance with existing prior-based 3DGS methods in sparse-view settings of benchmark datasets without any additional complexity. The code and model are publicly available at: https://github.com/DCVL-3D/DropGaussian release.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00763v1">UnIRe: Unsupervised Instance Decomposition for Dynamic Urban Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Reconstructing and decomposing dynamic urban scenes is crucial for autonomous driving, urban planning, and scene editing. However, existing methods fail to perform instance-aware decomposition without manual annotations, which is crucial for instance-level scene editing.We propose UnIRe, a 3D Gaussian Splatting (3DGS) based approach that decomposes a scene into a static background and individual dynamic instances using only RGB images and LiDAR point clouds. At its core, we introduce 4D superpoints, a novel representation that clusters multi-frame LiDAR points in 4D space, enabling unsupervised instance separation based on spatiotemporal correlations. These 4D superpoints serve as the foundation for our decomposed 4D initialization, i.e., providing spatial and temporal initialization to train a dynamic 3DGS for arbitrary dynamic classes without requiring bounding boxes or object templates.Furthermore, we introduce a smoothness regularization strategy in both 2D and 3D space, further improving the temporal stability.Experiments on benchmark datasets show that our method outperforms existing methods in decomposed dynamic scene reconstruction while enabling accurate and flexible instance-level editing, making it a practical solution for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2503.24366v1">StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) is a popular radiance field method, with many application-specific extensions. Most variants rely on the same core algorithm: depth-sorting of Gaussian splats then rasterizing in primitive order. This ensures correct alpha compositing, but can cause rendering artifacts due to built-in approximations. Moreover, for a fixed representation, sorted rendering offers little control over render cost and visual fidelity. For example, and counter-intuitively, rendering a lower-resolution image is not necessarily faster. In this work, we address the above limitations by combining 3D Gaussian splatting with stochastic rasterization. Concretely, we leverage an unbiased Monte Carlo estimator of the volume rendering equation. This removes the need for sorting, and allows for accurate 3D blending of overlapping Gaussians. The number of Monte Carlo samples further imbues 3DGS with a way to trade off computation time and quality. We implement our method using OpenGL shaders, enabling efficient rendering on modern GPU hardware. At a reasonable visual quality, our method renders more than four times faster than sorted rasterization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.00773v1">DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian splatting (3DGS) has gained considerable attentions in the field of novel view synthesis due to its fast performance while yielding the excellent image quality. However, 3DGS in sparse-view settings (e.g., three-view inputs) often faces with the problem of overfitting to training views, which significantly drops the visual quality of novel view images. Many existing approaches have tackled this issue by using strong priors, such as 2D generative contextual information and external depth signals. In contrast, this paper introduces a prior-free method, so-called DropGaussian, with simple changes in 3D Gaussian splatting. Specifically, we randomly remove Gaussians during the training process in a similar way of dropout, which allows non-excluded Gaussians to have larger gradients while improving their visibility. This makes the remaining Gaussians to contribute more to the optimization process for rendering with sparse input views. Such simple operation effectively alleviates the overfitting problem and enhances the quality of novel view synthesis. By simply applying DropGaussian to the original 3DGS framework, we can achieve the competitive performance with existing prior-based 3DGS methods in sparse-view settings of benchmark datasets without any additional complexity. The code and model are publicly available at: https://github.com/DCVL-3D/DropGaussian release.
    </details>
</div>
