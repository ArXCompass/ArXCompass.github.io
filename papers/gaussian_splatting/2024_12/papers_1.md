# gaussian splatting - 2024_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04831v4">AugGS: Self-augmented Gaussians with Structural Masks for Sparse-view 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-12-31
    </div>
    <details class="paper-abstract">
      Sparse-view 3D reconstruction is a major challenge in computer vision, aiming to create complete three-dimensional models from limited viewing angles. Key obstacles include: 1) a small number of input images with inconsistent information; 2) dependence on input image quality; and 3) large model parameter sizes. To tackle these issues, we propose a self-augmented two-stage Gaussian splatting framework enhanced with structural masks for sparse-view 3D reconstruction. Initially, our method generates a basic 3D Gaussian representation from sparse inputs and renders multi-view images. We then fine-tune a pre-trained 2D diffusion model to enhance these images, using them as augmented data to further optimize the 3D Gaussians. Additionally, a structural masking strategy during training enhances the model's robustness to sparse inputs and noise. Experiments on benchmarks like MipNeRF360, OmniObject3D, and OpenIllumination demonstrate that our approach achieves state-of-the-art performance in perceptual quality and multi-view consistency with sparse inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00352v1">PanoSLAM: Panoptic 3D Scene Reconstruction via Gaussian SLAM</a></div>
    <div class="paper-meta">
      📅 2024-12-31
    </div>
    <details class="paper-abstract">
      Understanding geometric, semantic, and instance information in 3D scenes from sequential video data is essential for applications in robotics and augmented reality. However, existing Simultaneous Localization and Mapping (SLAM) methods generally focus on either geometric or semantic reconstruction. In this paper, we introduce PanoSLAM, the first SLAM system to integrate geometric reconstruction, 3D semantic segmentation, and 3D instance segmentation within a unified framework. Our approach builds upon 3D Gaussian Splatting, modified with several critical components to enable efficient rendering of depth, color, semantic, and instance information from arbitrary viewpoints. To achieve panoptic 3D scene reconstruction from sequential RGB-D videos, we propose an online Spatial-Temporal Lifting (STL) module that transfers 2D panoptic predictions from vision models into 3D Gaussian representations. This STL module addresses the challenges of label noise and inconsistencies in 2D predictions by refining the pseudo labels across multi-view inputs, creating a coherent 3D representation that enhances segmentation accuracy. Our experiments show that PanoSLAM outperforms recent semantic SLAM methods in both mapping and tracking accuracy. For the first time, it achieves panoptic 3D reconstruction of open-world environments directly from the RGB-D video. (https://github.com/runnanchen/PanoSLAM)
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00342v1">SG-Splatting: Accelerating 3D Gaussian Splatting with Spherical Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-12-31
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting is emerging as a state-of-the-art technique in novel view synthesis, recognized for its impressive balance between visual quality, speed, and rendering efficiency. However, reliance on third-degree spherical harmonics for color representation introduces significant storage demands and computational overhead, resulting in a large memory footprint and slower rendering speed. We introduce SG-Splatting with Spherical Gaussians based color representation, a novel approach to enhance rendering speed and quality in novel view synthesis. Our method first represents view-dependent color using Spherical Gaussians, instead of three degree spherical harmonics, which largely reduces the number of parameters used for color representation, and significantly accelerates the rendering process. We then develop an efficient strategy for organizing multiple Spherical Gaussians, optimizing their arrangement to achieve a balanced and accurate scene representation. To further improve rendering quality, we propose a mixed representation that combines Spherical Gaussians with low-degree spherical harmonics, capturing both high- and low-frequency color information effectively. SG-Splatting also has plug-and-play capability, allowing it to be easily integrated into existing systems. This approach improves computational efficiency and overall visual fidelity, making it a practical solution for real-time applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.21206v1">PERSE: Personalized 3D Generative Avatars from A Single Portrait</a></div>
    <div class="paper-meta">
      📅 2024-12-30
      | 💬 Project Page: https://hyunsoocha.github.io/perse/
    </div>
    <details class="paper-abstract">
      We present PERSE, a method for building an animatable personalized generative avatar from a reference portrait. Our avatar model enables facial attribute editing in a continuous and disentangled latent space to control each facial attribute, while preserving the individual's identity. To achieve this, our method begins by synthesizing large-scale synthetic 2D video datasets, where each video contains consistent changes in the facial expression and viewpoint, combined with a variation in a specific facial attribute from the original input. We propose a novel pipeline to produce high-quality, photorealistic 2D videos with facial attribute editing. Leveraging this synthetic attribute dataset, we present a personalized avatar creation method based on the 3D Gaussian Splatting, learning a continuous and disentangled latent space for intuitive facial attribute manipulation. To enforce smooth transitions in this latent space, we introduce a latent space regularization technique by using interpolated 2D faces as supervision. Compared to previous approaches, we demonstrate that PERSE generates high-quality avatars with interpolated attributes while preserving identity of reference person.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20767v1">KeyGS: A Keyframe-Centric Gaussian Splatting Method for Monocular Image Sequences</a></div>
    <div class="paper-meta">
      📅 2024-12-30
      | 💬 AAAI 2025
    </div>
    <details class="paper-abstract">
      Reconstructing high-quality 3D models from sparse 2D images has garnered significant attention in computer vision. Recently, 3D Gaussian Splatting (3DGS) has gained prominence due to its explicit representation with efficient training speed and real-time rendering capabilities. However, existing methods still heavily depend on accurate camera poses for reconstruction. Although some recent approaches attempt to train 3DGS models without the Structure-from-Motion (SfM) preprocessing from monocular video datasets, these methods suffer from prolonged training times, making them impractical for many applications. In this paper, we present an efficient framework that operates without any depth or matching model. Our approach initially uses SfM to quickly obtain rough camera poses within seconds, and then refines these poses by leveraging the dense representation in 3DGS. This framework effectively addresses the issue of long training times. Additionally, we integrate the densification process with joint refinement and propose a coarse-to-fine frequency-aware densification to reconstruct different levels of details. This approach prevents camera pose estimation from being trapped in local minima or drifting due to high-frequency signals. Our method significantly reduces training time from hours to minutes while achieving more accurate novel view synthesis and camera pose estimation compared to previous methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18862v2">WeatherGS: 3D Scene Reconstruction in Adverse Weather Conditions via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-30
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has gained significant attention for 3D scene reconstruction, but still suffers from complex outdoor environments, especially under adverse weather. This is because 3DGS treats the artifacts caused by adverse weather as part of the scene and will directly reconstruct them, largely reducing the clarity of the reconstructed scene. To address this challenge, we propose WeatherGS, a 3DGS-based framework for reconstructing clear scenes from multi-view images under different weather conditions. Specifically, we explicitly categorize the multi-weather artifacts into the dense particles and lens occlusions that have very different characters, in which the former are caused by snowflakes and raindrops in the air, and the latter are raised by the precipitation on the camera lens. In light of this, we propose a dense-to-sparse preprocess strategy, which sequentially removes the dense particles by an Atmospheric Effect Filter (AEF) and then extracts the relatively sparse occlusion masks with a Lens Effect Detector (LED). Finally, we train a set of 3D Gaussians by the processed images and generated masks for excluding occluded areas, and accurately recover the underlying clear scene by Gaussian splatting. We conduct a diverse and challenging benchmark to facilitate the evaluation of 3D reconstruction under complex weather scenarios. Extensive experiments on this benchmark demonstrate that our WeatherGS consistently produces high-quality, clean scenes across various weather scenarios, outperforming existing state-of-the-art methods. See project page:https://jumponthemoon.github.io/weather-gs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20720v1">4D Gaussian Splatting: Modeling Dynamic Scenes with Native 4D Primitives</a></div>
    <div class="paper-meta">
      📅 2024-12-30
      | 💬 Journal extension of ICLR 2024. arXiv admin note: text overlap with arXiv:2310.10642
    </div>
    <details class="paper-abstract">
      Dynamic 3D scene representation and novel view synthesis from captured videos are crucial for enabling immersive experiences required by AR/VR and metaverse applications. However, this task is challenging due to the complexity of unconstrained real-world scenes and their temporal dynamics. In this paper, we frame dynamic scenes as a spatio-temporal 4D volume learning problem, offering a native explicit reformulation with minimal assumptions about motion, which serves as a versatile dynamic scene learning framework. Specifically, we represent a target dynamic scene using a collection of 4D Gaussian primitives with explicit geometry and appearance features, dubbed as 4D Gaussian splatting (4DGS). This approach can capture relevant information in space and time by fitting the underlying spatio-temporal volume. Modeling the spacetime as a whole with 4D Gaussians parameterized by anisotropic ellipses that can rotate arbitrarily in space and time, our model can naturally learn view-dependent and time-evolved appearance with 4D spherindrical harmonics. Notably, our 4DGS model is the first solution that supports real-time rendering of high-resolution, photorealistic novel views for complex dynamic scenes. To enhance efficiency, we derive several compact variants that effectively reduce memory footprint and mitigate the risk of overfitting. Extensive experiments validate the superiority of 4DGS in terms of visual quality and efficiency across a range of dynamic scene-related tasks (e.g., novel view synthesis, 4D generation, scene understanding) and scenarios (e.g., single object, indoor scenes, driving environments, synthetic and real data).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2412.18816v1">GSAVS: Gaussian Splatting-based Autonomous Vehicle Simulator</a></div>
    <div class="paper-meta">
      📅 2024-12-30
    </div>
    <details class="paper-abstract">
      Modern autonomous vehicle simulators feature an ever-growing library of assets, including vehicles, buildings, roads, pedestrians, and more. While this level of customization proves beneficial when creating virtual urban environments, this process becomes cumbersome when intending to train within a digital twin or a duplicate of a real scene. Gaussian splatting emerged as a powerful technique in scene reconstruction and novel view synthesis, boasting high fidelity and rendering speeds. In this paper, we introduce GSAVS, an autonomous vehicle simulator that supports the creation and development of autonomous vehicle models. Every asset within the simulator is a 3D Gaussian splat, including the vehicles and the environment. However, the simulator runs within a classical 3D engine, rendering 3D Gaussian splats in real-time. This allows the simulator to utilize the photorealism that 3D Gaussian splatting boasts while providing the customization and ease of use of a classical 3D engine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20522v1">MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks</a></div>
    <div class="paper-meta">
      📅 2024-12-29
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has demonstrated remarkable performance in novel view synthesis and real-time rendering, the high memory consumption due to the use of millions of Gaussians limits its practicality. To mitigate this issue, improvements have been made by pruning unnecessary Gaussians, either through a hand-crafted criterion or by using learned masks. However, these methods deterministically remove Gaussians based on a snapshot of the pruning moment, leading to sub-optimized reconstruction performance from a long-term perspective. To address this issue, we introduce MaskGaussian, which models Gaussians as probabilistic entities rather than permanently removing them, and utilize them according to their probability of existence. To achieve this, we propose a masked-rasterization technique that enables unused yet probabilistically existing Gaussians to receive gradients, allowing for dynamic assessment of their contribution to the evolving scene and adjustment of their probability of existence. Hence, the importance of Gaussians iteratively changes and the pruned Gaussians are selected diversely. Extensive experiments demonstrate the superiority of the proposed method in achieving better rendering quality with fewer Gaussians than previous pruning methods, pruning over 60% of Gaussians on average with only a 0.02 PSNR decline. Our code can be found at: https://github.com/kaikai23/MaskGaussian
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20148v1">DEGSTalk: Decomposed Per-Embedding Gaussian Fields for Hair-Preserving Talking Face Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-12-28
      | 💬 Accepted by ICASSP 2025
    </div>
    <details class="paper-abstract">
      Accurately synthesizing talking face videos and capturing fine facial features for individuals with long hair presents a significant challenge. To tackle these challenges in existing methods, we propose a decomposed per-embedding Gaussian fields (DEGSTalk), a 3D Gaussian Splatting (3DGS)-based talking face synthesis method for generating realistic talking faces with long hairs. Our DEGSTalk employs Deformable Pre-Embedding Gaussian Fields, which dynamically adjust pre-embedding Gaussian primitives using implicit expression coefficients. This enables precise capture of dynamic facial regions and subtle expressions. Additionally, we propose a Dynamic Hair-Preserving Portrait Rendering technique to enhance the realism of long hair motions in the synthesized videos. Results show that DEGSTalk achieves improved realism and synthesis quality compared to existing approaches, particularly in handling complex facial dynamics and hair preservation. Our code will be publicly available at https://github.com/CVI-SZU/DEGSTalk.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20056v1">GSplatLoc: Ultra-Precise Camera Localization via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-28
      | 💬 11 pages, 2 figures. Code available at https://github.com/AtticusZeller/GsplatLoc
    </div>
    <details class="paper-abstract">
      We present GSplatLoc, a camera localization method that leverages the differentiable rendering capabilities of 3D Gaussian splatting for ultra-precise pose estimation. By formulating pose estimation as a gradient-based optimization problem that minimizes discrepancies between rendered depth maps from a pre-existing 3D Gaussian scene and observed depth images, GSplatLoc achieves translational errors within 0.01 cm and near-zero rotational errors on the Replica dataset - significantly outperforming existing methods. Evaluations on the Replica and TUM RGB-D datasets demonstrate the method's robustness in challenging indoor environments with complex camera motions. GSplatLoc sets a new benchmark for localization in dense mapping, with important implications for applications requiring accurate real-time localization, such as robotics and augmented reality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19584v1">DAS3R: Dynamics-Aware Gaussian Splatting for Static Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-12-27
    </div>
    <details class="paper-abstract">
      We propose a novel framework for scene decomposition and static background reconstruction from everyday videos. By integrating the trained motion masks and modeling the static scene as Gaussian splats with dynamics-aware optimization, our method achieves more accurate background reconstruction results than previous works. Our proposed method is termed DAS3R, an abbreviation for Dynamics-Aware Gaussian Splatting for Static Scene Reconstruction. Compared to existing methods, DAS3R is more robust in complex motion scenarios, capable of handling videos where dynamic objects occupy a significant portion of the scene, and does not require camera pose inputs or point cloud data from SLAM-based methods. We compared DAS3R against recent distractor-free approaches on the DAVIS and Sintel datasets; DAS3R demonstrates enhanced performance and robustness with a margin of more than 2 dB in PSNR. The project's webpage can be accessed via \url{https://kai422.github.io/DAS3R/}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19518v1">Dust to Tower: Coarse-to-Fine Photo-Realistic Scene Reconstruction from Sparse Uncalibrated Images</a></div>
    <div class="paper-meta">
      📅 2024-12-27
    </div>
    <details class="paper-abstract">
      Photo-realistic scene reconstruction from sparse-view, uncalibrated images is highly required in practice. Although some successes have been made, existing methods are either Sparse-View but require accurate camera parameters (i.e., intrinsic and extrinsic), or SfM-free but need densely captured images. To combine the advantages of both methods while addressing their respective weaknesses, we propose Dust to Tower (D2T), an accurate and efficient coarse-to-fine framework to optimize 3DGS and image poses simultaneously from sparse and uncalibrated images. Our key idea is to first construct a coarse model efficiently and subsequently refine it using warped and inpainted images at novel viewpoints. To do this, we first introduce a Coarse Construction Module (CCM) which exploits a fast Multi-View Stereo model to initialize a 3D Gaussian Splatting (3DGS) and recover initial camera poses. To refine the 3D model at novel viewpoints, we propose a Confidence Aware Depth Alignment (CADA) module to refine the coarse depth maps by aligning their confident parts with estimated depths by a Mono-depth model. Then, a Warped Image-Guided Inpainting (WIGI) module is proposed to warp the training images to novel viewpoints by the refined depth maps, and inpainting is applied to fulfill the ``holes" in the warped images caused by view-direction changes, providing high-quality supervision to further optimize the 3D model and the camera poses. Extensive experiments and ablation studies demonstrate the validity of D2T and its design choices, achieving state-of-the-art performance in both tasks of novel view synthesis and pose estimation while keeping high efficiency. Codes will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19483v1">Learning Radiance Fields from a Single Snapshot Compressive Image</a></div>
    <div class="paper-meta">
      📅 2024-12-27
    </div>
    <details class="paper-abstract">
      In this paper, we explore the potential of Snapshot Compressive Imaging (SCI) technique for recovering the underlying 3D scene structure from a single temporal compressed image. SCI is a cost-effective method that enables the recording of high-dimensional data, such as hyperspectral or temporal information, into a single image using low-cost 2D imaging sensors. To achieve this, a series of specially designed 2D masks are usually employed, reducing storage and transmission requirements and offering potential privacy protection. Inspired by this, we take one step further to recover the encoded 3D scene information leveraging powerful 3D scene representation capabilities of neural radiance fields (NeRF). Specifically, we propose SCINeRF, in which we formulate the physical imaging process of SCI as part of the training of NeRF, allowing us to exploit its impressive performance in capturing complex scene structures. In addition, we further integrate the popular 3D Gaussian Splatting (3DGS) framework and propose SCISplat to improve 3D scene reconstruction quality and training/rendering speed by explicitly optimizing point clouds into 3D Gaussian representations. To assess the effectiveness of our method, we conduct extensive evaluations using both synthetic data and real data captured by our SCI system. Experimental results demonstrate that our proposed approach surpasses the state-of-the-art methods in terms of image reconstruction and novel view synthesis. Moreover, our method also exhibits the ability to render high frame-rate multi-view consistent images in real time by leveraging SCI and the rendering capabilities of 3DGS. Codes will be available at: https://github.com/WU- CVGL/SCISplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19282v1">Reflective Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-26
      | 💬 17 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Novel view synthesis has experienced significant advancements owing to increasingly capable NeRF- and 3DGS-based methods. However, reflective object reconstruction remains challenging, lacking a proper solution to achieve real-time, high-quality rendering while accommodating inter-reflection. To fill this gap, we introduce a Reflective Gaussian splatting (\textbf{Ref-Gaussian}) framework characterized with two components: (I) {\em Physically based deferred rendering} that empowers the rendering equation with pixel-level material properties via formulating split-sum approximation; (II) {\em Gaussian-grounded inter-reflection} that realizes the desired inter-reflection function within a Gaussian splatting paradigm for the first time. To enhance geometry modeling, we further introduce material-aware normal propagation and an initial per-Gaussian shading stage, along with 2D Gaussian primitives. Extensive experiments on standard datasets demonstrate that Ref-Gaussian surpasses existing approaches in terms of quantitative metrics, visual quality, and compute efficiency. Further, we show that our method serves as a unified solution for both reflective and non-reflective scenes, going beyond the previous alternatives focusing on only reflective scenes. Also, we illustrate that Ref-Gaussian supports more applications such as relighting and editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15447v2">LiHi-GS: LiDAR-Supervised Gaussian Splatting for Highway Driving Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-12-26
    </div>
    <details class="paper-abstract">
      Photorealistic 3D scene reconstruction plays an important role in autonomous driving, enabling the generation of novel data from existing datasets to simulate safety-critical scenarios and expand training data without additional acquisition costs. Gaussian Splatting (GS) facilitates real-time, photorealistic rendering with an explicit 3D Gaussian representation of the scene, providing faster processing and more intuitive scene editing than the implicit Neural Radiance Fields (NeRFs). While extensive GS research has yielded promising advancements in autonomous driving applications, they overlook two critical aspects: First, existing methods mainly focus on low-speed and feature-rich urban scenes and ignore the fact that highway scenarios play a significant role in autonomous driving. Second, while LiDARs are commonplace in autonomous driving platforms, existing methods learn primarily from images and use LiDAR only for initial estimates or without precise sensor modeling, thus missing out on leveraging the rich depth information LiDAR offers and limiting the ability to synthesize LiDAR data. In this paper, we propose a novel GS method for dynamic scene synthesis and editing with improved scene reconstruction through LiDAR supervision and support for LiDAR rendering. Unlike prior works that are tested mostly on urban datasets, to the best of our knowledge, we are the first to focus on the more challenging and highly relevant highway scenes for autonomous driving, with sparse sensor views and monotone backgrounds. Visit our project page at: https://umautobots.github.io/lihi_gs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19149v1">Generating Editable Head Avatars with 3D Gaussian GANs</a></div>
    <div class="paper-meta">
      📅 2024-12-26
    </div>
    <details class="paper-abstract">
      Generating animatable and editable 3D head avatars is essential for various applications in computer vision and graphics. Traditional 3D-aware generative adversarial networks (GANs), often using implicit fields like Neural Radiance Fields (NeRF), achieve photorealistic and view-consistent 3D head synthesis. However, these methods face limitations in deformation flexibility and editability, hindering the creation of lifelike and easily modifiable 3D heads. We propose a novel approach that enhances the editability and animation control of 3D head avatars by incorporating 3D Gaussian Splatting (3DGS) as an explicit 3D representation. This method enables easier illumination control and improved editability. Central to our approach is the Editable Gaussian Head (EG-Head) model, which combines a 3D Morphable Model (3DMM) with texture maps, allowing precise expression control and flexible texture editing for accurate animation while preserving identity. To capture complex non-facial geometries like hair, we use an auxiliary set of 3DGS and tri-plane features. Extensive experiments demonstrate that our approach delivers high-quality 3D-aware synthesis with state-of-the-art controllability. Our code and models are available at https://github.com/liguohao96/EGG3D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19142v1">CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-26
    </div>
    <details class="paper-abstract">
      Recent works in 3D multimodal learning have made remarkable progress. However, typically 3D multimodal models are only capable of handling point clouds. Compared to the emerging 3D representation technique, 3D Gaussian Splatting (3DGS), the spatially sparse point cloud cannot depict the texture information of 3D objects, resulting in inferior reconstruction capabilities. This limitation constrains the potential of point cloud-based 3D multimodal representation learning. In this paper, we present CLIP-GS, a novel multimodal representation learning framework grounded in 3DGS. We introduce the GS Tokenizer to generate serialized gaussian tokens, which are then processed through transformer layers pre-initialized with weights from point cloud models, resulting in the 3DGS embeddings. CLIP-GS leverages contrastive loss between 3DGS and the visual-text embeddings of CLIP, and we introduce an image voting loss to guide the directionality and convergence of gradient optimization. Furthermore, we develop an efficient way to generate triplets of 3DGS, images, and text, facilitating CLIP-GS in learning unified multimodal representations. Leveraging the well-aligned multimodal representations, CLIP-GS demonstrates versatility and outperforms point cloud-based models on various 3D tasks, including multimodal retrieval, zero-shot, and few-shot classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19130v1">MVS-GS: High-Quality 3D Gaussian Splatting Mapping via Online Multi-View Stereo</a></div>
    <div class="paper-meta">
      📅 2024-12-26
      | 💬 7 pages, 6 figures, submitted to IEEE ICRA 2025
    </div>
    <details class="paper-abstract">
      This study addresses the challenge of online 3D model generation for neural rendering using an RGB image stream. Previous research has tackled this issue by incorporating Neural Radiance Fields (NeRF) or 3D Gaussian Splatting (3DGS) as scene representations within dense SLAM methods. However, most studies focus primarily on estimating coarse 3D scenes rather than achieving detailed reconstructions. Moreover, depth estimation based solely on images is often ambiguous, resulting in low-quality 3D models that lead to inaccurate renderings. To overcome these limitations, we propose a novel framework for high-quality 3DGS modeling that leverages an online multi-view stereo (MVS) approach. Our method estimates MVS depth using sequential frames from a local time window and applies comprehensive depth refinement techniques to filter out outliers, enabling accurate initialization of Gaussians in 3DGS. Furthermore, we introduce a parallelized backend module that optimizes the 3DGS model efficiently, ensuring timely updates with each new keyframe. Experimental results demonstrate that our method outperforms state-of-the-art dense SLAM methods, particularly excelling in challenging outdoor environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20815v3">Grid4D: 4D Decomposed Hash Encoding for High-Fidelity Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-25
      | 💬 Accepted by NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Recently, Gaussian splatting has received more and more attention in the field of static scene rendering. Due to the low computational overhead and inherent flexibility of explicit representations, plane-based explicit methods are popular ways to predict deformations for Gaussian-based dynamic scene rendering models. However, plane-based methods rely on the inappropriate low-rank assumption and excessively decompose the space-time 4D encoding, resulting in overmuch feature overlap and unsatisfactory rendering quality. To tackle these problems, we propose Grid4D, a dynamic scene rendering model based on Gaussian splatting and employing a novel explicit encoding method for the 4D input through the hash encoding. Different from plane-based explicit representations, we decompose the 4D encoding into one spatial and three temporal 3D hash encodings without the low-rank assumption. Additionally, we design a novel attention module that generates the attention scores in a directional range to aggregate the spatial and temporal features. The directional attention enables Grid4D to more accurately fit the diverse deformations across distinct scene components based on the spatial encoded features. Moreover, to mitigate the inherent lack of smoothness in explicit representation methods, we introduce a smooth regularization term that keeps our model from the chaos of deformation prediction. Our experiments demonstrate that Grid4D significantly outperforms the state-of-the-art models in visual quality and rendering speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16619v2">Topology-Aware 3D Gaussian Splatting: Leveraging Persistent Homology for Optimized Structural Integrity</a></div>
    <div class="paper-meta">
      📅 2024-12-25
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has emerged as a crucial technique for representing discrete volumetric radiance fields. It leverages unique parametrization to mitigate computational demands in scene optimization. This work introduces Topology-Aware 3D Gaussian Splatting (Topology-GS), which addresses two key limitations in current approaches: compromised pixel-level structural integrity due to incomplete initial geometric coverage, and inadequate feature-level integrity from insufficient topological constraints during optimization. To overcome these limitations, Topology-GS incorporates a novel interpolation strategy, Local Persistent Voronoi Interpolation (LPVI), and a topology-focused regularization term based on persistent barcodes, named PersLoss. LPVI utilizes persistent homology to guide adaptive interpolation, enhancing point coverage in low-curvature areas while preserving topological structure. PersLoss aligns the visual perceptual similarity of rendered images with ground truth by constraining distances between their topological features. Comprehensive experiments on three novel-view synthesis benchmarks demonstrate that Topology-GS outperforms existing methods in terms of PSNR, SSIM, and LPIPS metrics, while maintaining efficient memory usage. This study pioneers the integration of topology with 3D-GS, laying the groundwork for future research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18816v1">GSAVS: Gaussian Splatting-based Autonomous Vehicle Simulator</a></div>
    <div class="paper-meta">
      📅 2024-12-25
    </div>
    <details class="paper-abstract">
      Modern autonomous vehicle simulators feature an ever-growing library of assets, including vehicles, buildings, roads, pedestrians, and more. While this level of customization proves beneficial when creating virtual urban environments, this process becomes cumbersome when intending to train within a digital twin or a duplicate of a real scene. Gaussian splatting emerged as a powerful technique in scene reconstruction and novel view synthesis, boasting high fidelity and rendering speeds. In this paper, we introduce GSAVS, an autonomous vehicle simulator that supports the creation and development of autonomous vehicle models. Every asset within the simulator is a 3D Gaussian splat, including the vehicles and the environment. However, the simulator runs within a classical 3D engine, rendering 3D Gaussian splats in real-time. This allows the simulator to utilize the photorealism that 3D Gaussian splatting boasts while providing the customization and ease of use of a classical 3D engine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18783v1">ArtNVG: Content-Style Separated Artistic Neighboring-View Gaussian Stylization</a></div>
    <div class="paper-meta">
      📅 2024-12-25
    </div>
    <details class="paper-abstract">
      As demand from the film and gaming industries for 3D scenes with target styles grows, the importance of advanced 3D stylization techniques increases. However, recent methods often struggle to maintain local consistency in color and texture throughout stylized scenes, which is essential for maintaining aesthetic coherence. To solve this problem, this paper introduces ArtNVG, an innovative 3D stylization framework that efficiently generates stylized 3D scenes by leveraging reference style images. Built on 3D Gaussian Splatting (3DGS), ArtNVG achieves rapid optimization and rendering while upholding high reconstruction quality. Our framework realizes high-quality 3D stylization by incorporating two pivotal techniques: Content-Style Separated Control and Attention-based Neighboring-View Alignment. Content-Style Separated Control uses the CSGO model and the Tile ControlNet to decouple the content and style control, reducing risks of information leakage. Concurrently, Attention-based Neighboring-View Alignment ensures consistency of local colors and textures across neighboring views, significantly improving visual quality. Extensive experiments validate that ArtNVG surpasses existing methods, delivering superior results in content preservation, style alignment, and local consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18584v1">Resolution-Robust 3D MRI Reconstruction with 2D Diffusion Priors: Diverse-Resolution Training Outperforms Interpolation</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      Deep learning-based 3D imaging, in particular magnetic resonance imaging (MRI), is challenging because of limited availability of 3D training data. Therefore, 2D diffusion models trained on 2D slices are starting to be leveraged for 3D MRI reconstruction. However, as we show in this paper, existing methods pertain to a fixed voxel size, and performance degrades when the voxel size is varied, as it is often the case in clinical practice. In this paper, we propose and study several approaches for resolution-robust 3D MRI reconstruction with 2D diffusion priors. As a result of this investigation, we obtain a simple resolution-robust variational 3D reconstruction approach based on diffusion-guided regularization of randomly sampled 2D slices. This method provides competitive reconstruction quality compared to posterior sampling baselines. Towards resolving the sensitivity to resolution-shifts, we investigate state-of-the-art model-based approaches including Gaussian splatting, neural representations, and infinite-dimensional diffusion models, as well as a simple data-centric approach of training the diffusion model on several resolutions. Our experiments demonstrate that the model-based approaches fail to close the performance gap in 3D MRI. In contrast, the data-centric approach of training the diffusion model on various resolutions effectively provides a resolution-robust method without compromising accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18380v1">RSGaussian:3D Gaussian Splatting with LiDAR for Aerial Remote Sensing Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      This study presents RSGaussian, an innovative novel view synthesis (NVS) method for aerial remote sensing scenes that incorporate LiDAR point cloud as constraints into the 3D Gaussian Splatting method, which ensures that Gaussians grow and split along geometric benchmarks, addressing the overgrowth and floaters issues occurs. Additionally, the approach introduces coordinate transformations with distortion parameters for camera models to achieve pixel-level alignment between LiDAR point clouds and 2D images, facilitating heterogeneous data fusion and achieving the high-precision geo-alignment required in aerial remote sensing. Depth and plane consistency losses are incorporated into the loss function to guide Gaussians towards real depth and plane representations, significantly improving depth estimation accuracy. Experimental results indicate that our approach has achieved novel view synthesis that balances photo-realistic visual quality and high-precision geometric estimation under aerial remote sensing datasets. Finally, we have also established and open-sourced a dense LiDAR point cloud dataset along with its corresponding aerial multi-view images, AIR-LONGYAN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19841v1">FlameGS: Reconstruct flame light field via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      To address the time-consuming and computationally intensive issues of traditional ART algorithms for flame combustion diagnosis, inspired by flame simulation technology, we propose a novel representation method for flames. By modeling the luminous process of flames and utilizing 2D projection images for supervision, our experimental validation shows that this model achieves an average structural similarity index of 0.96 between actual images and predicted 2D projections, along with a Peak Signal-to-Noise Ratio of 39.05. Additionally, it saves approximately 34 times the computation time and about 10 times the memory compared to traditional algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03771v4">SpikeGS: Reconstruct 3D scene via fast-moving bio-inspired sensors</a></div>
    <div class="paper-meta">
      📅 2024-12-24
      | 💬 Accepted by AAAI2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) demonstrates unparalleled superior performance in 3D scene reconstruction. However, 3DGS heavily relies on the sharp images. Fulfilling this requirement can be challenging in real-world scenarios especially when the camera moves fast, which severely limits the application of 3DGS. To address these challenges, we proposed Spike Gausian Splatting (SpikeGS), the first framework that integrates the spike streams into 3DGS pipeline to reconstruct 3D scenes via a fast-moving bio-inspired camera. With accumulation rasterization, interval supervision, and a specially designed pipeline, SpikeGS extracts detailed geometry and texture from high temporal resolution but texture lacking spike stream, reconstructs 3D scenes captured in 1 second. Extensive experiments on multiple synthetic and real-world datasets demonstrate the superiority of SpikeGS compared with existing spike-based and deblur 3D scene reconstruction methods. Codes and data will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17635v2">LangSurf: Language-Embedded Surface Gaussians for 3D Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2024-12-24
      | 💬 \url{https://langsurf.github.io}
    </div>
    <details class="paper-abstract">
      Applying Gaussian Splatting to perception tasks for 3D scene understanding is becoming increasingly popular. Most existing works primarily focus on rendering 2D feature maps from novel viewpoints, which leads to an imprecise 3D language field with outlier languages, ultimately failing to align objects in 3D space. By utilizing masked images for feature extraction, these approaches also lack essential contextual information, leading to inaccurate feature representation. To this end, we propose a Language-Embedded Surface Field (LangSurf), which accurately aligns the 3D language fields with the surface of objects, facilitating precise 2D and 3D segmentation with text query, widely expanding the downstream tasks such as removal and editing. The core of LangSurf is a joint training strategy that flattens the language Gaussian on the object surfaces using geometry supervision and contrastive losses to assign accurate language features to the Gaussians of objects. In addition, we also introduce the Hierarchical-Context Awareness Module to extract features at the image level for contextual information then perform hierarchical mask pooling using masks segmented by SAM to obtain fine-grained language features in different hierarchies. Extensive experiments on open-vocabulary 2D and 3D semantic segmentation demonstrate that LangSurf outperforms the previous state-of-the-art method LangSplat by a large margin. As shown in Fig. 1, our method is capable of segmenting objects in 3D space, thus boosting the effectiveness of our approach in instance recognition, removal, and editing, which is also supported by comprehensive experiments. \url{https://langsurf.github.io}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2412.16809v1">GeoTexDensifier: Geometry-Texture-Aware Densification for High-Quality Photorealistic 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-24
      | 💬 12 pages, 8 figures, 1 table
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently attracted wide attentions in various areas such as 3D navigation, Virtual Reality (VR) and 3D simulation, due to its photorealistic and efficient rendering performance. High-quality reconstrution of 3DGS relies on sufficient splats and a reasonable distribution of these splats to fit real geometric surface and texture details, which turns out to be a challenging problem. We present GeoTexDensifier, a novel geometry-texture-aware densification strategy to reconstruct high-quality Gaussian splats which better comply with the geometric structure and texture richness of the scene. Specifically, our GeoTexDensifier framework carries out an auxiliary texture-aware densification method to produce a denser distribution of splats in fully textured areas, while keeping sparsity in low-texture regions to maintain the quality of Gaussian point cloud. Meanwhile, a geometry-aware splitting strategy takes depth and normal priors to guide the splitting sampling and filter out the noisy splats whose initial positions are far from the actual geometric surfaces they aim to fit, under a Validation of Depth Ratio Change checking. With the help of relative monocular depth prior, such geometry-aware validation can effectively reduce the influence of scattered Gaussians to the final rendering quality, especially in regions with weak textures or without sufficient training views. The texture-aware densification and geometry-aware splitting strategies are fully combined to obtain a set of high-quality Gaussian splats. We experiment our GeoTexDensifier framework on various datasets and compare our Novel View Synthesis results to other state-of-the-art 3DGS approaches, with detailed quantitative and qualitative evaluations to demonstrate the effectiveness of our method in producing more photorealistic 3DGS models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07759v2">SwinGS: Sliding Window Gaussian Splatting for Volumetric Video Streaming with Arbitrary Length</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have garnered significant attention in computer vision and computer graphics due to its high rendering speed and remarkable quality. While extant research has endeavored to extend the application of 3DGS from static to dynamic scenes, such efforts have been consistently impeded by excessive model sizes, constraints on video duration, and content deviation. These limitations significantly compromise the streamability of dynamic 3D Gaussian models, thereby restricting their utility in downstream applications, including volumetric video, autonomous vehicle, and immersive technologies such as virtual, augmented, and mixed reality. This paper introduces SwinGS, a novel framework for training, delivering, and rendering volumetric video in a real-time streaming fashion. To address the aforementioned challenges and enhance streamability, SwinGS integrates spacetime Gaussian with Markov Chain Monte Carlo (MCMC) to adapt the model to fit various 3D scenes across frames, in the meantime employing a sliding window captures Gaussian snapshots for each frame in an accumulative way. We implement a prototype of SwinGS and demonstrate its streamability across various datasets and scenes. Additionally, we develop an interactive WebGL viewer enabling real-time volumetric video playback on most devices with modern browsers, including smartphones and tablets. Experimental results show that SwinGS reduces transmission costs by 83.6% compared to previous work with ignorable compromise in PSNR. Moreover, SwinGS easily scales to long video sequences without compromising quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17812v1">FaceLift: Single Image to 3D Head with View Generation and GS-LRM</a></div>
    <div class="paper-meta">
      📅 2024-12-23
      | 💬 Project page: https://weijielyu.github.io/FaceLift
    </div>
    <details class="paper-abstract">
      We present FaceLift, a feed-forward approach for rapid, high-quality, 360-degree head reconstruction from a single image. Our pipeline begins by employing a multi-view latent diffusion model that generates consistent side and back views of the head from a single facial input. These generated views then serve as input to a GS-LRM reconstructor, which produces a comprehensive 3D representation using Gaussian splats. To train our system, we develop a dataset of multi-view renderings using synthetic 3D human head as-sets. The diffusion-based multi-view generator is trained exclusively on synthetic head images, while the GS-LRM reconstructor undergoes initial training on Objaverse followed by fine-tuning on synthetic head data. FaceLift excels at preserving identity and maintaining view consistency across views. Despite being trained solely on synthetic data, FaceLift demonstrates remarkable generalization to real-world images. Through extensive qualitative and quantitative evaluations, we show that FaceLift outperforms state-of-the-art methods in 3D head reconstruction, highlighting its practical applicability and robust performance on real-world images. In addition to single image reconstruction, FaceLift supports video inputs for 4D novel view synthesis and seamlessly integrates with 2D reanimation techniques to enable 3D facial animation. Project page: https://weijielyu.github.io/FaceLift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17769v1">ActiveGS: Active Scene Reconstruction using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      Robotics applications often rely on scene reconstructions to enable downstream tasks. In this work, we tackle the challenge of actively building an accurate map of an unknown scene using an on-board RGB-D camera. We propose a hybrid map representation that combines a Gaussian splatting map with a coarse voxel map, leveraging the strengths of both representations: the high-fidelity scene reconstruction capabilities of Gaussian splatting and the spatial modelling strengths of the voxel map. The core of our framework is an effective confidence modelling technique for the Gaussian splatting map to identify under-reconstructed areas, while utilising spatial information from the voxel map to target unexplored areas and assist in collision-free path planning. By actively collecting scene information in under-reconstructed and unexplored areas for map updates, our approach achieves superior Gaussian splatting reconstruction results compared to state-of-the-art approaches. Additionally, we demonstrate the applicability of our active scene reconstruction framework in the real world using an unmanned aerial vehicle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17715v1">GaussianPainter: Painting Point Cloud into 3D Gaussians with Normal Guidance</a></div>
    <div class="paper-meta">
      📅 2024-12-23
      | 💬 To appear in AAAI 2025
    </div>
    <details class="paper-abstract">
      In this paper, we present GaussianPainter, the first method to paint a point cloud into 3D Gaussians given a reference image. GaussianPainter introduces an innovative feed-forward approach to overcome the limitations of time-consuming test-time optimization in 3D Gaussian splatting. Our method addresses a critical challenge in the field: the non-uniqueness problem inherent in the large parameter space of 3D Gaussian splatting. This space, encompassing rotation, anisotropic scales, and spherical harmonic coefficients, introduces the challenge of rendering similar images from substantially different Gaussian fields. As a result, feed-forward networks face instability when attempting to directly predict high-quality Gaussian fields, struggling to converge on consistent parameters for a given output. To address this issue, we propose to estimate a surface normal for each point to determine its Gaussian rotation. This strategy enables the network to effectively predict the remaining Gaussian parameters in the constrained space. We further enhance our approach with an appearance injection module, incorporating reference image appearance into Gaussian fields via a multiscale triplane representation. Our method successfully balances efficiency and fidelity in 3D Gaussian generation, achieving high-quality, diverse, and robust 3D content creation from point clouds in a single forward pass.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17612v1">CoSurfGS:Collaborative 3D Surface Gaussian Splatting with Distributed Learning for Large Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-12-23
      | 💬 Our project page is available at \url{https://gyy456.github.io/CoSurfGS}
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated impressive performance in scene reconstruction. However, most existing GS-based surface reconstruction methods focus on 3D objects or limited scenes. Directly applying these methods to large-scale scene reconstruction will pose challenges such as high memory costs, excessive time consumption, and lack of geometric detail, which makes it difficult to implement in practical applications. To address these issues, we propose a multi-agent collaborative fast 3DGS surface reconstruction framework based on distributed learning for large-scale surface reconstruction. Specifically, we develop local model compression (LMC) and model aggregation schemes (MAS) to achieve high-quality surface representation of large scenes while reducing GPU memory consumption. Extensive experiments on Urban3d, MegaNeRF, and BlendedMVS demonstrate that our proposed method can achieve fast and scalable high-fidelity surface reconstruction and photorealistic rendering. Our project page is available at \url{https://gyy456.github.io/CoSurfGS}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17532v1">Exploring Dynamic Novel View Synthesis Technologies for Cinematography</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) has shown significant promise for applications in cinematographic production, particularly through the exploitation of Neural Radiance Fields (NeRF) and Gaussian Splatting (GS). These methods model real 3D scenes, enabling the creation of new shots that are challenging to capture in the real world due to set topology or expensive equipment requirement. This innovation also offers cinematographic advantages such as smooth camera movements, virtual re-shoots, slow-motion effects, etc. This paper explores dynamic NVS with the aim of facilitating the model selection process. We showcase its potential through a short montage filmed using various NVS models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01517v2">UW-GS: Distractor-Aware 3D Gaussian Splatting for Enhanced Underwater Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-12-23
      | 💬 Accepted at IEEE/CVF WACV 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) offers the capability to achieve real-time high quality 3D scene rendering. However, 3DGS assumes that the scene is in a clear medium environment and struggles to generate satisfactory representations in underwater scenes, where light absorption and scattering are prevalent and moving objects are involved. To overcome these, we introduce a novel Gaussian Splatting-based method, UW-GS, designed specifically for underwater applications. It introduces a color appearance that models distance-dependent color variation, employs a new physics-based density control strategy to enhance clarity for distant objects, and uses a binary motion mask to handle dynamic content. Optimized with a well-designed loss function supporting for scattering media and strengthened by pseudo-depth maps, UW-GS outperforms existing methods with PSNR gains up to 1.26dB. To fully verify the effectiveness of the model, we also developed a new underwater dataset, S-UW, with dynamic object masks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.02218v4">WavePlanes: Compact Hex Planes for Dynamic Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      Dynamic Novel View Synthesis (Dynamic NVS) enhances NVS technologies to model moving 3-D scenes. However, current methods are resource intensive and challenging to compress. To address this, we present WavePlanes, a fast and more compact hex plane representation, applicable to both Neural Radiance Fields and Gaussian Splatting methods. Rather than modeling many feature scales separately (as done previously), we use the inverse discrete wavelet transform to reconstruct features at varying scales. This leads to a more compact representation and allows us to explore wavelet-based compression schemes for further gains. The proposed compression scheme exploits the sparsity of wavelet coefficients, by applying hard thresholding to the wavelet planes and storing nonzero coefficients and their locations on each plane in a Hash Map. Compared to the state-of-the-art (SotA), WavePlanes is significantly smaller, less resource demanding and competitive in reconstruction quality. Compared to small SotA models, WavePlanes outperforms methods in both model size and quality of novel views.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13222v3">3D-GSW: 3D Gaussian Splatting for Robust Watermarking</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      As 3D Gaussian Splatting~(3D-GS) gains significant attention and its commercial usage increases, the need for watermarking technologies to prevent unauthorized use of the 3D-GS models and rendered images has become increasingly important. In this paper, we introduce a robust watermarking method for 3D-GS that secures ownership of both the model and its rendered images. Our proposed method remains robust against distortions in rendered images and model attacks while maintaining high rendering quality. To achieve these objectives, we present Frequency-Guided Densification~(FGD), which removes 3D Gaussians based on their contribution to rendering quality, enhancing real-time rendering and the robustness of the message. FGD utilizes Discrete Fourier Transform to split 3D Gaussians in high-frequency areas, improving rendering quality. Furthermore, we employ a gradient mask for 3D Gaussians and design a wavelet-subband loss to enhance rendering quality. Our experiments show that our method embeds the message in the rendered images invisibly and robustly against various attacks, including model distortion. Our method achieves state-of-the-art performance. Project page: https://kuai-lab.github.io/3dgsw2024/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03910v2">DGNS: Deformable Gaussian Splatting and Dynamic Neural Surface for Monocular Dynamic 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction from monocular video is critical for real-world applications. This paper tackles the dual challenges of dynamic novel-view synthesis and 3D geometry reconstruction by introducing a hybrid framework: Deformable Gaussian Splatting and Dynamic Neural Surfaces (DGNS), in which both modules can leverage each other for both tasks. During training, depth maps generated by the deformable Gaussian splatting module guide the ray sampling for faster processing and provide depth supervision within the dynamic neural surface module to improve geometry reconstruction. Simultaneously, the dynamic neural surface directs the distribution of Gaussian primitives around the surface, enhancing rendering quality. To further refine depth supervision, we introduce a depth-filtering process on depth maps derived from Gaussian rasterization. Extensive experiments on public datasets demonstrate that DGNS achieves state-of-the-art performance in both novel-view synthesis and 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2412.15400v1">SolidGS: Consolidating Gaussian Surfel Splatting for Sparse-View Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-12-23
      | 💬 Project page: https://mickshen7558.github.io/projects/SolidGS/
    </div>
    <details class="paper-abstract">
      Gaussian splatting has achieved impressive improvements for both novel-view synthesis and surface reconstruction from multi-view images. However, current methods still struggle to reconstruct high-quality surfaces from only sparse view input images using Gaussian splatting. In this paper, we propose a novel method called SolidGS to address this problem. We observed that the reconstructed geometry can be severely inconsistent across multi-views, due to the property of Gaussian function in geometry rendering. This motivates us to consolidate all Gaussians by adopting a more solid kernel function, which effectively improves the surface reconstruction quality. With the additional help of geometrical regularization and monocular normal estimation, our method achieves superior performance on the sparse view surface reconstruction than all the Gaussian splatting methods and neural field methods on the widely used DTU, Tanks-and-Temples, and LLFF datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07541v2">HiCoM: Hierarchical Coherent Motion for Streamable Dynamic Scene with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-22
      | 💬 Accepted to NeurIPS 2024; Code is avaliable at https://github.com/gqk/HiCoM
    </div>
    <details class="paper-abstract">
      The online reconstruction of dynamic scenes from multi-view streaming videos faces significant challenges in training, rendering and storage efficiency. Harnessing superior learning speed and real-time rendering capabilities, 3D Gaussian Splatting (3DGS) has recently demonstrated considerable potential in this field. However, 3DGS can be inefficient in terms of storage and prone to overfitting by excessively growing Gaussians, particularly with limited views. This paper proposes an efficient framework, dubbed HiCoM, with three key components. First, we construct a compact and robust initial 3DGS representation using a perturbation smoothing strategy. Next, we introduce a Hierarchical Coherent Motion mechanism that leverages the inherent non-uniform distribution and local consistency of 3D Gaussians to swiftly and accurately learn motions across frames. Finally, we continually refine the 3DGS with additional Gaussians, which are later merged into the initial 3DGS to maintain consistency with the evolving scene. To preserve a compact representation, an equivalent number of low-opacity Gaussians that minimally impact the representation are removed before processing subsequent frames. Extensive experiments conducted on two widely used datasets show that our framework improves learning efficiency of the state-of-the-art methods by about $20\%$ and reduces the data storage by $85\%$, achieving competitive free-viewpoint video synthesis quality but with higher robustness and stability. Moreover, by parallel learning multiple frames simultaneously, our HiCoM decreases the average training wall time to $<2$ seconds per frame with negligible performance degradation, substantially boosting real-world applicability and responsiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16932v1">GSemSplat: Generalizable Semantic 3D Gaussian Splatting from Uncalibrated Image Pairs</a></div>
    <div class="paper-meta">
      📅 2024-12-22
    </div>
    <details class="paper-abstract">
      Modeling and understanding the 3D world is crucial for various applications, from augmented reality to robotic navigation. Recent advancements based on 3D Gaussian Splatting have integrated semantic information from multi-view images into Gaussian primitives. However, these methods typically require costly per-scene optimization from dense calibrated images, limiting their practicality. In this paper, we consider the new task of generalizable 3D semantic field modeling from sparse, uncalibrated image pairs. Building upon the Splatt3R architecture, we introduce GSemSplat, a framework that learns open-vocabulary semantic representations linked to 3D Gaussians without the need for per-scene optimization, dense image collections or calibration. To ensure effective and reliable learning of semantic features in 3D space, we employ a dual-feature approach that leverages both region-specific and context-aware semantic features as supervision in the 2D space. This allows us to capitalize on their complementary strengths. Experimental results on the ScanNet++ dataset demonstrate the effectiveness and superiority of our approach compared to the traditional scene-specific method. We hope our work will inspire more research into generalizable 3D understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16809v1">GeoTexDensifier: Geometry-Texture-Aware Densification for High-Quality Photorealistic 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-22
      | 💬 12 pages, 8 figures, 1 table
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently attracted wide attentions in various areas such as 3D navigation, Virtual Reality (VR) and 3D simulation, due to its photorealistic and efficient rendering performance. High-quality reconstrution of 3DGS relies on sufficient splats and a reasonable distribution of these splats to fit real geometric surface and texture details, which turns out to be a challenging problem. We present GeoTexDensifier, a novel geometry-texture-aware densification strategy to reconstruct high-quality Gaussian splats which better comply with the geometric structure and texture richness of the scene. Specifically, our GeoTexDensifier framework carries out an auxiliary texture-aware densification method to produce a denser distribution of splats in fully textured areas, while keeping sparsity in low-texture regions to maintain the quality of Gaussian point cloud. Meanwhile, a geometry-aware splitting strategy takes depth and normal priors to guide the splitting sampling and filter out the noisy splats whose initial positions are far from the actual geometric surfaces they aim to fit, under a Validation of Depth Ratio Change checking. With the help of relative monocular depth prior, such geometry-aware validation can effectively reduce the influence of scattered Gaussians to the final rendering quality, especially in regions with weak textures or without sufficient training views. The texture-aware densification and geometry-aware splitting strategies are fully combined to obtain a set of high-quality Gaussian splats. We experiment our GeoTexDensifier framework on various datasets and compare our Novel View Synthesis results to other state-of-the-art 3DGS approaches, with detailed quantitative and qualitative evaluations to demonstrate the effectiveness of our method in producing more photorealistic 3DGS models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16604v1">OmniSplat: Taming Feed-Forward 3D Gaussian Splatting for Omnidirectional Images with Editable Capabilities</a></div>
    <div class="paper-meta">
      📅 2024-12-21
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) models have gained significant popularity due to their ability to generate scenes immediately without needing per-scene optimization. Although omnidirectional images are getting more popular since they reduce the computation for image stitching to composite a holistic scene, existing feed-forward models are only designed for perspective images. The unique optical properties of omnidirectional images make it difficult for feature encoders to correctly understand the context of the image and make the Gaussian non-uniform in space, which hinders the image quality synthesized from novel views. We propose OmniSplat, a pioneering work for fast feed-forward 3DGS generation from a few omnidirectional images. We introduce Yin-Yang grid and decompose images based on it to reduce the domain gap between omnidirectional and perspective images. The Yin-Yang grid can use the existing CNN structure as it is, but its quasi-uniform characteristic allows the decomposed image to be similar to a perspective image, so it can exploit the strong prior knowledge of the learned feed-forward network. OmniSplat demonstrates higher reconstruction accuracy than existing feed-forward networks trained on perspective images. Furthermore, we enhance the segmentation consistency between omnidirectional images by leveraging attention from the encoder of OmniSplat, providing fast and clean 3DGS editing results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12753v2">DrivingForward: Feed-forward 3D Gaussian Splatting for Driving Scene Reconstruction from Flexible Surround-view Input</a></div>
    <div class="paper-meta">
      📅 2024-12-21
      | 💬 Accept by AAAI 2025. Project Page: https://fangzhou2000.github.io/projects/drivingforward/
    </div>
    <details class="paper-abstract">
      We propose DrivingForward, a feed-forward Gaussian Splatting model that reconstructs driving scenes from flexible surround-view input. Driving scene images from vehicle-mounted cameras are typically sparse, with limited overlap, and the movement of the vehicle further complicates the acquisition of camera extrinsics. To tackle these challenges and achieve real-time reconstruction, we jointly train a pose network, a depth network, and a Gaussian network to predict the Gaussian primitives that represent the driving scenes. The pose network and depth network determine the position of the Gaussian primitives in a self-supervised manner, without using depth ground truth and camera extrinsics during training. The Gaussian network independently predicts primitive parameters from each input image, including covariance, opacity, and spherical harmonics coefficients. At the inference stage, our model can achieve feed-forward reconstruction from flexible multi-frame surround-view input. Experiments on the nuScenes dataset show that our model outperforms existing state-of-the-art feed-forward and scene-optimized reconstruction methods in terms of reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.06149v2">HeadStudio: Text to Animatable Head Avatars with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-21
      | 💬 26 pages, 18 figures, accepted by ECCV 2024
    </div>
    <details class="paper-abstract">
      Creating digital avatars from textual prompts has long been a desirable yet challenging task. Despite the promising results achieved with 2D diffusion priors, current methods struggle to create high-quality and consistent animated avatars efficiently. Previous animatable head models like FLAME have difficulty in accurately representing detailed texture and geometry. Additionally, high-quality 3D static representations face challenges in semantically driving with dynamic priors. In this paper, we introduce \textbf{HeadStudio}, a novel framework that utilizes 3D Gaussian splatting to generate realistic and animatable avatars from text prompts. Firstly, we associate 3D Gaussians with animatable head prior model, facilitating semantic animation on high-quality 3D representations. To ensure consistent animation, we further enhance the optimization from initialization, distillation, and regularization to jointly learn the shape, texture, and animation. Extensive experiments demonstrate the efficacy of HeadStudio in generating animatable avatars from textual prompts, exhibiting appealing appearances. The avatars are capable of rendering high-quality real-time ($\geq 40$ fps) novel views at a resolution of 1024. Moreover, These avatars can be smoothly driven by real-world speech and video. We hope that HeadStudio can enhance digital avatar creation and gain popularity in the community. Code is at: https://github.com/ZhenglinZhou/HeadStudio.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15171v2">SqueezeMe: Efficient Gaussian Avatars for VR</a></div>
    <div class="paper-meta">
      📅 2024-12-21
      | 💬 v2
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has enabled real-time 3D human avatars with unprecedented levels of visual quality. While previous methods require a desktop GPU for real-time inference of a single avatar, we aim to squeeze multiple Gaussian avatars onto a portable virtual reality headset with real-time drivable inference. We begin by training a previous work, Animatable Gaussians, on a high quality dataset captured with 512 cameras. The Gaussians are animated by controlling base set of Gaussians with linear blend skinning (LBS) motion and then further adjusting the Gaussians with a neural network decoder to correct their appearance. When deploying the model on a Meta Quest 3 VR headset, we find two major computational bottlenecks: the decoder and the rendering. To accelerate the decoder, we train the Gaussians in UV-space instead of pixel-space, and we distill the decoder to a single neural network layer. Further, we discover that neighborhoods of Gaussians can share a single corrective from the decoder, which provides an additional speedup. To accelerate the rendering, we develop a custom pipeline in Vulkan that runs on the mobile GPU. Putting it all together, we run 3 Gaussian avatars concurrently at 72 FPS on a VR headset. Demo videos are at https://forresti.github.io/squeezeme.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16346v1">SOUS VIDE: Cooking Visual Drone Navigation Policies in a Gaussian Splatting Vacuum</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      We propose a new simulator, training approach, and policy architecture, collectively called SOUS VIDE, for end-to-end visual drone navigation. Our trained policies exhibit zero-shot sim-to-real transfer with robust real-world performance using only on-board perception and computation. Our simulator, called FiGS, couples a computationally simple drone dynamics model with a high visual fidelity Gaussian Splatting scene reconstruction. FiGS can quickly simulate drone flights producing photorealistic images at up to 130 fps. We use FiGS to collect 100k-300k observation-action pairs from an expert MPC with privileged state and dynamics information, randomized over dynamics parameters and spatial disturbances. We then distill this expert MPC into an end-to-end visuomotor policy with a lightweight neural architecture, called SV-Net. SV-Net processes color image, optical flow and IMU data streams into low-level body rate and thrust commands at 20Hz onboard a drone. Crucially, SV-Net includes a Rapid Motor Adaptation (RMA) module that adapts at runtime to variations in drone dynamics. In a campaign of 105 hardware experiments, we show SOUS VIDE policies to be robust to 30% mass variations, 40 m/s wind gusts, 60% changes in ambient brightness, shifting or removing objects from the scene, and people moving aggressively through the drone's visual field. Code, data, and experiment videos can be found on our project page: https://stanfordmsl.github.io/SousVide/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16028v1">CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images</a></div>
    <div class="paper-meta">
      📅 2024-12-20
      | 💬 Project Page: https://Jho-Yonsei.github.io/CoCoGaussian/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has attracted significant attention for its high-quality novel view rendering, inspiring research to address real-world challenges. While conventional methods depend on sharp images for accurate scene reconstruction, real-world scenarios are often affected by defocus blur due to finite depth of field, making it essential to account for realistic 3D scene representation. In this study, we propose CoCoGaussian, a Circle of Confusion-aware Gaussian Splatting that enables precise 3D scene representation using only defocused images. CoCoGaussian addresses the challenge of defocus blur by modeling the Circle of Confusion (CoC) through a physically grounded approach based on the principles of photographic defocus. Exploiting 3D Gaussians, we compute the CoC diameter from depth and learnable aperture information, generating multiple Gaussians to precisely capture the CoC shape. Furthermore, we introduce a learnable scaling factor to enhance robustness and provide more flexibility in handling unreliable depth in scenes with reflective or refractive surfaces. Experiments on both synthetic and real-world datasets demonstrate that CoCoGaussian achieves state-of-the-art performance across multiple benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15867v1">IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing</a></div>
    <div class="paper-meta">
      📅 2024-12-20
      | 💬 Project page: https://fudan-zvg.github.io/IRGS
    </div>
    <details class="paper-abstract">
      In inverse rendering, accurately modeling visibility and indirect radiance for incident light is essential for capturing secondary effects. Due to the absence of a powerful Gaussian ray tracer, previous 3DGS-based methods have either adopted a simplified rendering equation or used learnable parameters to approximate incident light, resulting in inaccurate material and lighting estimations. To this end, we introduce inter-reflective Gaussian splatting (IRGS) for inverse rendering. To capture inter-reflection, we apply the full rendering equation without simplification and compute incident radiance on the fly using the proposed differentiable 2D Gaussian ray tracing. Additionally, we present an efficient optimization scheme to handle the computational demands of Monte Carlo sampling for rendering equation evaluation. Furthermore, we introduce a novel strategy for querying the indirect radiance of incident light when relighting the optimized scenes. Extensive experiments on multiple standard benchmarks validate the effectiveness of IRGS, demonstrating its capability to accurately model complex inter-reflection effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15723v3">GSurf: 3D Reconstruction via Signed Distance Fields with Direct Gaussian Supervision</a></div>
    <div class="paper-meta">
      📅 2024-12-20
      | 💬 see https://github.com/xubaixinxbx/Gsurf
    </div>
    <details class="paper-abstract">
      Surface reconstruction from multi-view images is a core challenge in 3D vision. Recent studies have explored signed distance fields (SDF) within Neural Radiance Fields (NeRF) to achieve high-fidelity surface reconstructions. However, these approaches often suffer from slow training and rendering speeds compared to 3D Gaussian splatting (3DGS). Current state-of-the-art techniques attempt to fuse depth information to extract geometry from 3DGS, but frequently result in incomplete reconstructions and fragmented surfaces. In this paper, we introduce GSurf, a novel end-to-end method for learning a signed distance field directly from Gaussian primitives. The continuous and smooth nature of SDF addresses common issues in the 3DGS family, such as holes resulting from noisy or missing depth data. By using Gaussian splatting for rendering, GSurf avoids the redundant volume rendering typically required in other GS and SDF integrations. Consequently, GSurf achieves faster training and rendering speeds while delivering 3D reconstruction quality comparable to neural implicit surface methods, such as VolSDF and NeuS. Experimental results across various benchmark datasets demonstrate the effectiveness of our method in producing high-fidelity 3D reconstructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15609v1">AvatarPerfect: User-Assisted 3D Gaussian Splatting Avatar Refinement with Automatic Pose Suggestion</a></div>
    <div class="paper-meta">
      📅 2024-12-20
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Creating high-quality 3D avatars using 3D Gaussian Splatting (3DGS) from a monocular video benefits virtual reality and telecommunication applications. However, existing automatic methods exhibit artifacts under novel poses due to limited information in the input video. We propose AvatarPerfect, a novel system that allows users to iteratively refine 3DGS avatars by manually editing the rendered avatar images. In each iteration, our system suggests a new body and camera pose to help users identify and correct artifacts. The edited images are then used to update the current avatar, and our system suggests the next body and camera pose for further refinement. To investigate the effectiveness of AvatarPerfect, we conducted a user study comparing our method to an existing 3DGS editor SuperSplat, which allows direct manipulation of Gaussians without automatic pose suggestions. The results indicate that our system enables users to obtain higher quality refined 3DGS avatars than the existing 3DGS editor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16253v1">Interactive Scene Authoring with Specialized Generative Primitives</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      Generating high-quality 3D digital assets often requires expert knowledge of complex design tools. We introduce Specialized Generative Primitives, a generative framework that allows non-expert users to author high-quality 3D scenes in a seamless, lightweight, and controllable manner. Each primitive is an efficient generative model that captures the distribution of a single exemplar from the real world. With our framework, users capture a video of an environment, which we turn into a high-quality and explicit appearance model thanks to 3D Gaussian Splatting. Users then select regions of interest guided by semantically-aware features. To create a generative primitive, we adapt Generative Cellular Automata to single-exemplar training and controllable generation. We decouple the generative task from the appearance model by operating on sparse voxels and we recover a high-quality output with a subsequent sparse patch consistency step. Each primitive can be trained within 10 minutes and used to author new scenes interactively in a fully compositional manner. We showcase interactive sessions where various primitives are extracted from real-world scenes and controlled to create 3D assets and scenes in a few minutes. We also demonstrate additional capabilities of our primitives: handling various 3D representations to control generation, transferring appearances, and editing geometries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15550v1">EGSRAL: An Enhanced 3D Gaussian Splatting based Renderer with Automated Labeling for Large-Scale Driving Scene</a></div>
    <div class="paper-meta">
      📅 2024-12-20
      | 💬 AAAI2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D GS) has gained popularity due to its faster rendering speed and high-quality novel view synthesis. Some researchers have explored using 3D GS for reconstructing driving scenes. However, these methods often rely on various data types, such as depth maps, 3D boxes, and trajectories of moving objects. Additionally, the lack of annotations for synthesized images limits their direct application in downstream tasks. To address these issues, we propose EGSRAL, a 3D GS-based method that relies solely on training images without extra annotations. EGSRAL enhances 3D GS's capability to model both dynamic objects and static backgrounds and introduces a novel adaptor for auto labeling, generating corresponding annotations based on existing annotations. We also propose a grouping strategy for vanilla 3D GS to address perspective issues in rendering large-scale, complex scenes. Our method achieves state-of-the-art performance on multiple datasets without any extra annotation. For example, the PSNR metric reaches 29.04 on the nuScenes dataset. Moreover, our automated labeling can significantly improve the performance of 2D/3D detection tasks. Code is available at https://github.com/jiangxb98/EGSRAL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15400v1">SolidGS: Consolidating Gaussian Surfel Splatting for Sparse-View Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-12-19
      | 💬 Project page: https://mickshen7558.github.io/projects/SolidGS/
    </div>
    <details class="paper-abstract">
      Gaussian splatting has achieved impressive improvements for both novel-view synthesis and surface reconstruction from multi-view images. However, current methods still struggle to reconstruct high-quality surfaces from only sparse view input images using Gaussian splatting. In this paper, we propose a novel method called SolidGS to address this problem. We observed that the reconstructed geometry can be severely inconsistent across multi-views, due to the property of Gaussian function in geometry rendering. This motivates us to consolidate all Gaussians by adopting a more solid kernel function, which effectively improves the surface reconstruction quality. With the additional help of geometrical regularization and monocular normal estimation, our method achieves superior performance on the sparse view surface reconstruction than all the Gaussian splatting methods and neural field methods on the widely used DTU, Tanks-and-Temples, and LLFF datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14957v1">Dream to Manipulate: Compositional World Models Empowering Robot Imitation Learning with Imagination</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      A world model provides an agent with a representation of its environment, enabling it to predict the causal consequences of its actions. Current world models typically cannot directly and explicitly imitate the actual environment in front of a robot, often resulting in unrealistic behaviors and hallucinations that make them unsuitable for real-world applications. In this paper, we introduce a new paradigm for constructing world models that are explicit representations of the real world and its dynamics. By integrating cutting-edge advances in real-time photorealism with Gaussian Splatting and physics simulators, we propose the first compositional manipulation world model, which we call DreMa. DreMa replicates the observed world and its dynamics, allowing it to imagine novel configurations of objects and predict the future consequences of robot actions. We leverage this capability to generate new data for imitation learning by applying equivariant transformations to a small set of demonstrations. Our evaluations across various settings demonstrate significant improvements in both accuracy and robustness by incrementing actions and object distributions, reducing the data needed to learn a policy and improving the generalization of the agents. As a highlight, we show that a real Franka Emika Panda robot, powered by DreMa's imagination, can successfully learn novel physical tasks from just a single example per task variation (one-shot policy learning). Our project page and source code can be found in https://leobarcellona.github.io/DreamToManipulate/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14579v1">GSRender: Deduplicated Occupancy Prediction via Weakly Supervised 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      3D occupancy perception is gaining increasing attention due to its capability to offer detailed and precise environment representations. Previous weakly-supervised NeRF methods balance efficiency and accuracy, with mIoU varying by 5-10 points due to sampling count along camera rays. Recently, real-time Gaussian splatting has gained widespread popularity in 3D reconstruction, and the occupancy prediction task can also be viewed as a reconstruction task. Consequently, we propose GSRender, which naturally employs 3D Gaussian Splatting for occupancy prediction, simplifying the sampling process. In addition, the limitations of 2D supervision result in duplicate predictions along the same camera ray. We implemented the Ray Compensation (RC) module, which mitigates this issue by compensating for features from adjacent frames. Finally, we redesigned the loss to eliminate the impact of dynamic objects from adjacent frames. Extensive experiments demonstrate that our approach achieves SOTA (state-of-the-art) results in RayIoU (+6.0), while narrowing the gap with 3D supervision methods. Our code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14568v1">Improving Geometry in Sparse-View 3DGS via Reprojection-based DoF Separation</a></div>
    <div class="paper-meta">
      📅 2024-12-19
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Recent learning-based Multi-View Stereo models have demonstrated state-of-the-art performance in sparse-view 3D reconstruction. However, directly applying 3D Gaussian Splatting (3DGS) as a refinement step following these models presents challenges. We hypothesize that the excessive positional degrees of freedom (DoFs) in Gaussians induce geometry distortion, fitting color patterns at the cost of structural fidelity. To address this, we propose reprojection-based DoF separation, a method distinguishing positional DoFs in terms of uncertainty: image-plane-parallel DoFs and ray-aligned DoF. To independently manage each DoF, we introduce a reprojection process along with tailored constraints for each DoF. Through experiments across various datasets, we confirm that separating the positional DoFs of Gaussians and applying targeted constraints effectively suppresses geometric artifacts, producing reconstruction results that are both visually and geometrically plausible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12262v2">3D Gaussian Splatting in Robotics: A Survey</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      Dense 3D representations of the environment have been a long-term goal in the robotics field. While previous Neural Radiance Fields (NeRF) representation have been prevalent for its implicit, coordinate-based model, the recent emergence of 3D Gaussian Splatting (3DGS) has demonstrated remarkable potential in its explicit radiance field representation. By leveraging 3D Gaussian primitives for explicit scene representation and enabling differentiable rendering, 3DGS has shown significant advantages over other radiance fields in real-time rendering and photo-realistic performance, which is beneficial for robotic applications. In this survey, we provide a comprehensive understanding of 3DGS in the field of robotics. We divide our discussion of the related works into two main categories: the application of 3DGS and the advancements in 3DGS techniques. In the application section, we explore how 3DGS has been utilized in various robotics tasks from scene understanding and interaction perspectives. The advance of 3DGS section focuses on the improvements of 3DGS own properties in its adaptability and efficiency, aiming to enhance its performance in robotics. We then summarize the most commonly used datasets and evaluation metrics in robotics. Finally, we identify the challenges and limitations of current 3DGS methods and discuss the future development of 3DGS in robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13983v1">GraphAvatar: Compact Head Avatars with GNN-Generated 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 accepted by AAAI2025
    </div>
    <details class="paper-abstract">
      Rendering photorealistic head avatars from arbitrary viewpoints is crucial for various applications like virtual reality. Although previous methods based on Neural Radiance Fields (NeRF) can achieve impressive results, they lack fidelity and efficiency. Recent methods using 3D Gaussian Splatting (3DGS) have improved rendering quality and real-time performance but still require significant storage overhead. In this paper, we introduce a method called GraphAvatar that utilizes Graph Neural Networks (GNN) to generate 3D Gaussians for the head avatar. Specifically, GraphAvatar trains a geometric GNN and an appearance GNN to generate the attributes of the 3D Gaussians from the tracked mesh. Therefore, our method can store the GNN models instead of the 3D Gaussians, significantly reducing the storage overhead to just 10MB. To reduce the impact of face-tracking errors, we also present a novel graph-guided optimization module to refine face-tracking parameters during training. Finally, we introduce a 3D-aware enhancer for post-processing to enhance the rendering quality. We conduct comprehensive experiments to demonstrate the advantages of GraphAvatar, surpassing existing methods in visual fidelity and storage consumption. The ablation study sheds light on the trade-offs between rendering quality and model size. The code will be released at: https://github.com/ucwxb/GraphAvatar
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13654v1">GAGS: Granularity-Aware Feature Distillation for Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 Project page: https://pz0826.github.io/GAGS-Webpage/
    </div>
    <details class="paper-abstract">
      3D open-vocabulary scene understanding, which accurately perceives complex semantic properties of objects in space, has gained significant attention in recent years. In this paper, we propose GAGS, a framework that distills 2D CLIP features into 3D Gaussian splatting, enabling open-vocabulary queries for renderings on arbitrary viewpoints. The main challenge of distilling 2D features for 3D fields lies in the multiview inconsistency of extracted 2D features, which provides unstable supervision for the 3D feature field. GAGS addresses this challenge with two novel strategies. First, GAGS associates the prompt point density of SAM with the camera distances, which significantly improves the multiview consistency of segmentation results. Second, GAGS further decodes a granularity factor to guide the distillation process and this granularity factor can be learned in a unsupervised manner to only select the multiview consistent 2D features in the distillation process. Experimental results on two datasets demonstrate significant performance and stability improvements of GAGS in visual grounding and semantic segmentation, with an inference speed 2$\times$ faster than baseline methods. The code and additional results are available at https://pz0826.github.io/GAGS-Webpage/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13639v1">4D Radar-Inertial Odometry based on Gaussian Modeling and Multi-Hypothesis Scan Matching</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 Our code and results can be publicly accessed at: https://github.com/robotics-upo/gaussian-rio
    </div>
    <details class="paper-abstract">
      4D millimeter-wave (mmWave) radars are sensors that provide robustness against adverse weather conditions (rain, snow, fog, etc.), and as such they are increasingly being used for odometry and SLAM applications. However, the noisy and sparse nature of the returned scan data proves to be a challenging obstacle for existing point cloud matching based solutions, especially those originally intended for more accurate sensors such as LiDAR. Inspired by visual odometry research around 3D Gaussian Splatting, in this paper we propose using freely positioned 3D Gaussians to create a summarized representation of a radar point cloud tolerant to sensor noise, and subsequently leverage its inherent probability distribution function for registration (similar to NDT). Moreover, we propose simultaneously optimizing multiple scan matching hypotheses in order to further increase the robustness of the system against local optima of the function. Finally, we fuse our Gaussian modeling and scan matching algorithms into an EKF radar-inertial odometry system designed after current best practices. Experiments show that our Gaussian-based odometry is able to outperform current baselines on a well-known 4D radar dataset used for evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13547v1">Turbo-GS: Accelerating 3D Gaussian Fitting for High-Quality Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 Project page: https://ivl.cs.brown.edu/research/turbo-gs
    </div>
    <details class="paper-abstract">
      Novel-view synthesis is an important problem in computer vision with applications in 3D reconstruction, mixed reality, and robotics. Recent methods like 3D Gaussian Splatting (3DGS) have become the preferred method for this task, providing high-quality novel views in real time. However, the training time of a 3DGS model is slow, often taking 30 minutes for a scene with 200 views. In contrast, our goal is to reduce the optimization time by training for fewer steps while maintaining high rendering quality. Specifically, we combine the guidance from both the position error and the appearance error to achieve a more effective densification. To balance the rate between adding new Gaussians and fitting old Gaussians, we develop a convergence-aware budget control mechanism. Moreover, to make the densification process more reliable, we selectively add new Gaussians from mostly visited regions. With these designs, we reduce the Gaussian optimization steps to one-third of the previous approach while achieving a comparable or even better novel view rendering quality. To further facilitate the rapid fitting of 4K resolution images, we introduce a dilation-based rendering technique. Our method, Turbo-GS, speeds up optimization for typical scenes and scales well to high-resolution (4K) scenarios on standard datasets. Through extensive experiments, we show that our method is significantly faster in optimization than other methods while retaining quality. Project page: https://ivl.cs.brown.edu/research/turbo-gs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13509v1">Vivar: A Generative AR System for Intuitive Multi-Modal Sensor Data Presentation</a></div>
    <div class="paper-meta">
      📅 2024-12-18
    </div>
    <details class="paper-abstract">
      Understanding sensor data can be challenging for non-experts because of the complexity and unique semantic meanings of sensor modalities. This calls for intuitive and effective methods to present sensor information. However, creating intuitive sensor data visualizations presents three key challenges: the variability of sensor readings, gaps in domain comprehension, and the dynamic nature of sensor data. To address these issues, we develop Vivar, a novel AR system that integrates multi-modal sensor data and presents 3D volumetric content for visualization. In particular, we introduce a cross-modal embedding approach that maps sensor data into a pre-trained visual embedding space through barycentric interpolation. This allows for accurate and continuous integration of multi-modal sensor information. Vivar also incorporates sensor-aware AR scene generation using foundation models and 3D Gaussian Splatting (3DGS) without requiring domain expertise. In addition, Vivar leverages latent reuse and caching strategies to accelerate 2D and AR content generation. Our extensive experiments demonstrate that our system achieves 11$\times$ latency reduction without compromising quality. A user study involving over 485 participants, including domain experts, demonstrates Vivar's effectiveness in accuracy, consistency, and real-world applicability, paving the way for more intuitive sensor data visualization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09982v2">SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 The first two authors contributed equally to this work (equal contribution). The last two authors advised equally to this work. Please visit our project page at this https://kaist-viclab.github.io/splinegs-site/
    </div>
    <details class="paper-abstract">
      Synthesizing novel views from in-the-wild monocular videos is challenging due to scene dynamics and the lack of multi-view cues. To address this, we propose SplineGS, a COLMAP-free dynamic 3D Gaussian Splatting (3DGS) framework for high-quality reconstruction and fast rendering from monocular videos. At its core is a novel Motion-Adaptive Spline (MAS) method, which represents continuous dynamic 3D Gaussian trajectories using cubic Hermite splines with a small number of control points. For MAS, we introduce a Motion-Adaptive Control points Pruning (MACP) method to model the deformation of each dynamic 3D Gaussian across varying motions, progressively pruning control points while maintaining dynamic modeling integrity. Additionally, we present a joint optimization strategy for camera parameter estimation and 3D Gaussian attributes, leveraging photometric and geometric consistency. This eliminates the need for Structure-from-Motion preprocessing and enhances SplineGS's robustness in real-world conditions. Experiments show that SplineGS significantly outperforms state-of-the-art methods in novel view synthesis quality for dynamic scenes from monocular videos, achieving thousands times faster rendering speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2412.12734v1">Gaussian Billboards: Expressive 2D Gaussian Splatting with Textures</a></div>
    <div class="paper-meta">
      📅 2024-12-18
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has recently emerged as the go-to representation for reconstructing and rendering 3D scenes. The transition from 3D to 2D Gaussian primitives has further improved multi-view consistency and surface reconstruction accuracy. In this work we highlight the similarity between 2D Gaussian Splatting (2DGS) and billboards from traditional computer graphics. Both use flat semi-transparent 2D geometry that is positioned, oriented and scaled in 3D space. However 2DGS uses a solid color per splat and an opacity modulated by a Gaussian distribution, where billboards are more expressive, modulating the color with a uv-parameterized texture. We propose to unify these concepts by presenting Gaussian Billboards, a modification of 2DGS to add spatially-varying color achieved using per-splat texture interpolation. The result is a mixture of the two representations, which benefits from both the robust scene optimization power of 2DGS and the expressiveness of texture mapping. We show that our method can improve the sharpness and quality of the scene representation in a wide range of qualitative and quantitative evaluations compared to the original 2DGS implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.20309v4">InstantSplat: Sparse-view SfM-free Gaussian Splatting in Seconds</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 Project Page: https://instantsplat.github.io/
    </div>
    <details class="paper-abstract">
      While neural 3D reconstruction has advanced substantially, it typically requires densely captured multi-view data with carefully initialized poses (e.g., using COLMAP). However, this requirement limits its broader applicability, as Structure-from-Motion (SfM) is often unreliable in sparse-view scenarios where feature matches are limited, resulting in cumulative errors. In this paper, we introduce InstantSplat, a novel and lightning-fast neural reconstruction system that builds accurate 3D representations from as few as 2-3 images. InstantSplat adopts a self-supervised framework that bridges the gap between 2D images and 3D representations using Gaussian Bundle Adjustment (GauBA) and can be optimized in an end-to-end manner. InstantSplat integrates dense stereo priors and co-visibility relationships between frames to initialize pixel-aligned geometry by progressively expanding the scene avoiding redundancy. Gaussian Bundle Adjustment is used to adapt both the scene representation and camera parameters quickly by minimizing gradient-based photometric error. Overall, InstantSplat achieves large-scale 3D reconstruction in mere seconds by reducing the required number of input views. It achieves an acceleration of over 20 times in reconstruction, improves visual quality (SSIM) from 0.3755 to 0.7624 than COLMAP with 3D-GS, and is compatible with multiple 3D representations (3D-GS, 2D-GS, and Mip-Splatting).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13183v1">Real-time Free-view Human Rendering from Sparse-view RGB Videos using Double Unprojected Textures</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 Project page: https://vcai.mpi-inf.mpg.de/projects/DUT/
    </div>
    <details class="paper-abstract">
      Real-time free-view human rendering from sparse-view RGB inputs is a challenging task due to the sensor scarcity and the tight time budget. To ensure efficiency, recent methods leverage 2D CNNs operating in texture space to learn rendering primitives. However, they either jointly learn geometry and appearance, or completely ignore sparse image information for geometry estimation, significantly harming visual quality and robustness to unseen body poses. To address these issues, we present Double Unprojected Textures, which at the core disentangles coarse geometric deformation estimation from appearance synthesis, enabling robust and photorealistic 4K rendering in real-time. Specifically, we first introduce a novel image-conditioned template deformation network, which estimates the coarse deformation of the human template from a first unprojected texture. This updated geometry is then used to apply a second and more accurate texture unprojection. The resulting texture map has fewer artifacts and better alignment with input views, which benefits our learning of finer-level geometry and appearance represented by Gaussian splats. We validate the effectiveness and efficiency of the proposed method in quantitative and qualitative experiments, which significantly surpasses other state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13047v1">EOGS: Gaussian Splatting for Earth Observation</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      Recently, Gaussian splatting has emerged as a strong alternative to NeRF, demonstrating impressive 3D modeling capabilities while requiring only a fraction of the training and rendering time. In this paper, we show how the standard Gaussian splatting framework can be adapted for remote sensing, retaining its high efficiency. This enables us to achieve state-of-the-art performance in just a few minutes, compared to the day-long optimization required by the best-performing NeRF-based Earth observation methods. The proposed framework incorporates remote-sensing improvements from EO-NeRF, such as radiometric correction and shadow modeling, while introducing novel components, including sparsity, view consistency, and opacity regularizations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12919v1">4DRGS: 4D Radiative Gaussian Splatting for Efficient 3D Vessel Reconstruction from Sparse-View Dynamic DSA Images</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 Zhentao Liu and Ruyi Zha made equal contributions
    </div>
    <details class="paper-abstract">
      Reconstructing 3D vessel structures from sparse-view dynamic digital subtraction angiography (DSA) images enables accurate medical assessment while reducing radiation exposure. Existing methods often produce suboptimal results or require excessive computation time. In this work, we propose 4D radiative Gaussian splatting (4DRGS) to achieve high-quality reconstruction efficiently. In detail, we represent the vessels with 4D radiative Gaussian kernels. Each kernel has time-invariant geometry parameters, including position, rotation, and scale, to model static vessel structures. The time-dependent central attenuation of each kernel is predicted from a compact neural network to capture the temporal varying response of contrast agent flow. We splat these Gaussian kernels to synthesize DSA images via X-ray rasterization and optimize the model with real captured ones. The final 3D vessel volume is voxelized from the well-trained kernels. Moreover, we introduce accumulated attenuation pruning and bounded scaling activation to improve reconstruction quality. Extensive experiments on real-world patient data demonstrate that 4DRGS achieves impressive results in 5 minutes training, which is 32x faster than the state-of-the-art method. This underscores the potential of 4DRGS for real-world clinics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12906v1">CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      Recently, generalizable feed-forward methods based on 3D Gaussian Splatting have gained significant attention for their potential to reconstruct 3D scenes using finite resources. These approaches create a 3D radiance field, parameterized by per-pixel 3D Gaussian primitives, from just a few images in a single forward pass. However, unlike multi-view methods that benefit from cross-view correspondences, 3D scene reconstruction with a single-view image remains an underexplored area. In this work, we introduce CATSplat, a novel generalizable transformer-based framework designed to break through the inherent constraints in monocular settings. First, we propose leveraging textual guidance from a visual-language model to complement insufficient information from a single image. By incorporating scene-specific contextual details from text embeddings through cross-attention, we pave the way for context-aware 3D scene reconstruction beyond relying solely on visual cues. Moreover, we advocate utilizing spatial guidance from 3D point features toward comprehensive geometric understanding under single-view settings. With 3D priors, image features can capture rich structural insights for predicting 3D Gaussians without multi-view techniques. Extensive experiments on large-scale datasets demonstrate the state-of-the-art performance of CATSplat in single-view 3D scene reconstruction with high-quality novel view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14322v2">SplatR : Experience Goal Visual Rearrangement with 3D Gaussian Splatting and Dense Feature Matching</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      Experience Goal Visual Rearrangement task stands as a foundational challenge within Embodied AI, requiring an agent to construct a robust world model that accurately captures the goal state. The agent uses this world model to restore a shuffled scene to its original configuration, making an accurate representation of the world essential for successfully completing the task. In this work, we present a novel framework that leverages on 3D Gaussian Splatting as a 3D scene representation for experience goal visual rearrangement task. Recent advances in volumetric scene representation like 3D Gaussian Splatting, offer fast rendering of high quality and photo-realistic novel views. Our approach enables the agent to have consistent views of the current and the goal setting of the rearrangement task, which enables the agent to directly compare the goal state and the shuffled state of the world in image space. To compare these views, we propose to use a dense feature matching method with visual features extracted from a foundation model, leveraging its advantages of a more universal feature representation, which facilitates robustness, and generalization. We validate our approach on the AI2-THOR rearrangement challenge benchmark and demonstrate improvements over the current state of the art methods
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12849v1">HyperGS: Hyperspectral 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      We introduce HyperGS, a novel framework for Hyperspectral Novel View Synthesis (HNVS), based on a new latent 3D Gaussian Splatting (3DGS) technique. Our approach enables simultaneous spatial and spectral renderings by encoding material properties from multi-view 3D hyperspectral datasets. HyperGS reconstructs high-fidelity views from arbitrary perspectives with improved accuracy and speed, outperforming currently existing methods. To address the challenges of high-dimensional data, we perform view synthesis in a learned latent space, incorporating a pixel-wise adaptive density function and a pruning technique for increased training stability and efficiency. Additionally, we introduce the first HNVS benchmark, implementing a number of new baselines based on recent SOTA RGB-NVS techniques, alongside the small number of prior works on HNVS. We demonstrate HyperGS's robustness through extensive evaluation of real and simulated hyperspectral scenes with a 14db accuracy improvement upon previously published models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12734v1">Gaussian Billboards: Expressive 2D Gaussian Splatting with Textures</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has recently emerged as the go-to representation for reconstructing and rendering 3D scenes. The transition from 3D to 2D Gaussian primitives has further improved multi-view consistency and surface reconstruction accuracy. In this work we highlight the similarity between 2D Gaussian Splatting (2DGS) and billboards from traditional computer graphics. Both use flat semi-transparent 2D geometry that is positioned, oriented and scaled in 3D space. However 2DGS uses a solid color per splat and an opacity modulated by a Gaussian distribution, where billboards are more expressive, modulating the color with a uv-parameterized texture. We propose to unify these concepts by presenting Gaussian Billboards, a modification of 2DGS to add spatially-varying color achieved using per-splat texture interpolation. The result is a mixture of the two representations, which benefits from both the robust scene optimization power of 2DGS and the expressiveness of texture mapping. We show that our method can improve the sharpness and quality of the scene representation in a wide range of qualitative and quantitative evaluations compared to the original 2DGS implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.17418v2">3D Gaussian Splatting: Survey, Technologies, Challenges, and Opportunities</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a prominent technique with the potential to become a mainstream method for 3D representations. It can effectively transform multi-view images into explicit 3D Gaussian through efficient training, and achieve real-time rendering of novel views. This survey aims to analyze existing 3DGS-related works from multiple intersecting perspectives, including related tasks, technologies, challenges, and opportunities. The primary objective is to provide newcomers with a rapid understanding of the field and to assist researchers in methodically organizing existing technologies and challenges. Specifically, we delve into the optimization, application, and extension of 3DGS, categorizing them based on their focuses or motivations. Additionally, we summarize and classify nine types of technical modules and corresponding improvements identified in existing works. Based on these analyses, we further examine the common challenges and technologies across various tasks, proposing potential research opportunities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12507v1">3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown great potential for efficient reconstruction and high-fidelity real-time rendering of complex scenes on consumer hardware. However, due to its rasterization-based formulation, 3DGS is constrained to ideal pinhole cameras and lacks support for secondary lighting effects. Recent methods address these limitations by tracing volumetric particles instead, however, this comes at the cost of significantly slower rendering speeds. In this work, we propose 3D Gaussian Unscented Transform (3DGUT), replacing the EWA splatting formulation in 3DGS with the Unscented Transform that approximates the particles through sigma points, which can be projected exactly under any nonlinear projection function. This modification enables trivial support of distorted cameras with time dependent effects such as rolling shutter, while retaining the efficiency of rasterization. Additionally, we align our rendering formulation with that of tracing-based methods, enabling secondary ray tracing required to represent phenomena such as reflections and refraction within the same 3D representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2412.01682v3">Diffusion Models with Anisotropic Gaussian Splatting for Image Inpainting</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      Image inpainting is a fundamental task in computer vision, aiming to restore missing or corrupted regions in images realistically. While recent deep learning approaches have significantly advanced the state-of-the-art, challenges remain in maintaining structural continuity and generating coherent textures, particularly in large missing areas. Diffusion models have shown promise in generating high-fidelity images but often lack the structural guidance necessary for realistic inpainting. We propose a novel inpainting method that combines diffusion models with anisotropic Gaussian splatting to capture both local structures and global context effectively. By modeling missing regions using anisotropic Gaussian functions that adapt to local image gradients, our approach provides structural guidance to the diffusion-based inpainting network. The Gaussian splat maps are integrated into the diffusion process, enhancing the model's ability to generate high-fidelity and structurally coherent inpainting results. Extensive experiments demonstrate that our method outperforms state-of-the-art techniques, producing visually plausible results with enhanced structural integrity and texture realism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12096v1">PanSplat: 4K Panorama Synthesis with Feed-Forward Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 Project Page: https://chengzhag.github.io/publication/pansplat/ Code: https://github.com/chengzhag/PanSplat
    </div>
    <details class="paper-abstract">
      With the advent of portable 360{\deg} cameras, panorama has gained significant attention in applications like virtual reality (VR), virtual tours, robotics, and autonomous driving. As a result, wide-baseline panorama view synthesis has emerged as a vital task, where high resolution, fast inference, and memory efficiency are essential. Nevertheless, existing methods are typically constrained to lower resolutions (512 $\times$ 1024) due to demanding memory and computational requirements. In this paper, we present PanSplat, a generalizable, feed-forward approach that efficiently supports resolution up to 4K (2048 $\times$ 4096). Our approach features a tailored spherical 3D Gaussian pyramid with a Fibonacci lattice arrangement, enhancing image quality while reducing information redundancy. To accommodate the demands of high resolution, we propose a pipeline that integrates a hierarchical spherical cost volume and Gaussian heads with local operations, enabling two-step deferred backpropagation for memory-efficient training on a single A100 GPU. Experiments demonstrate that PanSplat achieves state-of-the-art results with superior efficiency and image quality across both synthetic and real-world datasets. Code will be available at \url{https://github.com/chengzhag/PanSplat}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12091v1">Wonderland: Navigating 3D Scenes from a Single Image</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 Project page: https://snap-research.github.io/wonderland/
    </div>
    <details class="paper-abstract">
      This paper addresses a challenging question: How can we efficiently create high-quality, wide-scope 3D scenes from a single arbitrary image? Existing methods face several constraints, such as requiring multi-view data, time-consuming per-scene optimization, low visual quality in backgrounds, and distorted reconstructions in unseen areas. We propose a novel pipeline to overcome these limitations. Specifically, we introduce a large-scale reconstruction model that uses latents from a video diffusion model to predict 3D Gaussian Splattings for the scenes in a feed-forward manner. The video diffusion model is designed to create videos precisely following specified camera trajectories, allowing it to generate compressed video latents that contain multi-view information while maintaining 3D consistency. We train the 3D reconstruction model to operate on the video latent space with a progressive training strategy, enabling the efficient generation of high-quality, wide-scope, and generic 3D scenes. Extensive evaluations across various datasets demonstrate that our model significantly outperforms existing methods for single-view 3D scene generation, particularly with out-of-domain images. For the first time, we demonstrate that a 3D reconstruction model can be effectively built upon the latent space of a diffusion model to realize efficient 3D scene generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11762v1">GS-ProCams: Gaussian Splatting-based Projector-Camera Systems</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      We present GS-ProCams, the first Gaussian Splatting-based framework for projector-camera systems (ProCams). GS-ProCams significantly enhances the efficiency of projection mapping (PM) that requires establishing geometric and radiometric mappings between the projector and the camera. Previous CNN-based ProCams are constrained to a specific viewpoint, limiting their applicability to novel perspectives. In contrast, NeRF-based ProCams support view-agnostic projection mapping, however, they require an additional colocated light source and demand significant computational and memory resources. To address this issue, we propose GS-ProCams that employs 2D Gaussian for scene representations, and enables efficient view-agnostic ProCams applications. In particular, we explicitly model the complex geometric and photometric mappings of ProCams using projector responses, the target surface's geometry and materials represented by Gaussians, and global illumination component. Then, we employ differentiable physically-based rendering to jointly estimate them from captured multi-view projections. Compared to state-of-the-art NeRF-based methods, our GS-ProCams eliminates the need for additional devices, achieving superior ProCams simulation quality. It is also 600 times faster and uses only 1/10 of the GPU memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11752v1">Deformable Radial Kernel Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      Recently, Gaussian splatting has emerged as a robust technique for representing 3D scenes, enabling real-time rasterization and high-fidelity rendering. However, Gaussians' inherent radial symmetry and smoothness constraints limit their ability to represent complex shapes, often requiring thousands of primitives to approximate detailed geometry. We introduce Deformable Radial Kernel (DRK), which extends Gaussian splatting into a more general and flexible framework. Through learnable radial bases with adjustable angles and scales, DRK efficiently models diverse shape primitives while enabling precise control over edge sharpness and boundary curvature. iven DRK's planar nature, we further develop accurate ray-primitive intersection computation for depth sorting and introduce efficient kernel culling strategies for improved rasterization efficiency. Extensive experiments demonstrate that DRK outperforms existing methods in both representation efficiency and rendering quality, achieving state-of-the-art performance while dramatically reducing primitive count.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19271v2">AGS-Mesh: Adaptive Gaussian Splatting and Meshing with Geometric Priors for Indoor Room Reconstruction Using Smartphones</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      Geometric priors are often used to enhance 3D reconstruction. With many smartphones featuring low-resolution depth sensors and the prevalence of off-the-shelf monocular geometry estimators, incorporating geometric priors as regularization signals has become common in 3D vision tasks. However, the accuracy of depth estimates from mobile devices is typically poor for highly detailed geometry, and monocular estimators often suffer from poor multi-view consistency and precision. In this work, we propose an approach for joint surface depth and normal refinement of Gaussian Splatting methods for accurate 3D reconstruction of indoor scenes. We develop supervision strategies that adaptively filters low-quality depth and normal estimates by comparing the consistency of the priors during optimization. We mitigate regularization in regions where prior estimates have high uncertainty or ambiguities. Our filtering strategy and optimization design demonstrate significant improvements in both mesh estimation and novel-view synthesis for both 3D and 2D Gaussian Splatting-based methods on challenging indoor room datasets. Furthermore, we explore the use of alternative meshing strategies for finer geometry extraction. We develop a scale-aware meshing strategy inspired by TSDF and octree-based isosurface extraction, which recovers finer details from Gaussian models compared to other commonly used open-source meshing tools. Our code is released in https://xuqianren.github.io/ags_mesh_website/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11579v1">SweepEvGS: Event-Based 3D Gaussian Splatting for Macro and Micro Radiance Field Rendering from a Single Sweep</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3D-GS) have demonstrated the potential of using 3D Gaussian primitives for high-speed, high-fidelity, and cost-efficient novel view synthesis from continuously calibrated input views. However, conventional methods require high-frame-rate dense and high-quality sharp images, which are time-consuming and inefficient to capture, especially in dynamic environments. Event cameras, with their high temporal resolution and ability to capture asynchronous brightness changes, offer a promising alternative for more reliable scene reconstruction without motion blur. In this paper, we propose SweepEvGS, a novel hardware-integrated method that leverages event cameras for robust and accurate novel view synthesis across various imaging settings from a single sweep. SweepEvGS utilizes the initial static frame with dense event streams captured during a single camera sweep to effectively reconstruct detailed scene views. We also introduce different real-world hardware imaging systems for real-world data collection and evaluation for future research. We validate the robustness and efficiency of SweepEvGS through experiments in three different imaging settings: synthetic objects, real-world macro-level, and real-world micro-level view synthesis. Our results demonstrate that SweepEvGS surpasses existing methods in visual rendering quality, rendering speed, and computational efficiency, highlighting its potential for dynamic practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11520v1">EditSplat: Multi-View Fusion and Attention-Guided Optimization for View-Consistent 3D Scene Editing with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D editing have highlighted the potential of text-driven methods in real-time, user-friendly AR/VR applications. However, current methods rely on 2D diffusion models without adequately considering multi-view information, resulting in multi-view inconsistency. While 3D Gaussian Splatting (3DGS) significantly improves rendering quality and speed, its 3D editing process encounters difficulties with inefficient optimization, as pre-trained Gaussians retain excessive source information, hindering optimization. To address these limitations, we propose \textbf{EditSplat}, a novel 3D editing framework that integrates Multi-view Fusion Guidance (MFG) and Attention-Guided Trimming (AGT). Our MFG ensures multi-view consistency by incorporating essential multi-view information into the diffusion process, leveraging classifier-free guidance from the text-to-image diffusion model and the geometric properties of 3DGS. Additionally, our AGT leverages the explicit representation of 3DGS to selectively prune and optimize 3D Gaussians, enhancing optimization efficiency and enabling precise, semantically rich local edits. Through extensive qualitative and quantitative evaluations, EditSplat achieves superior multi-view consistency and editing quality over existing methods, significantly enhancing overall efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06367v3">MVGamba: Unify 3D Content Generation as State Space Sequence Modeling</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 Accepted by NeurIPS 2024. Code is included in https://github.com/SkyworkAI/MVGamba
    </div>
    <details class="paper-abstract">
      Recent 3D large reconstruction models (LRMs) can generate high-quality 3D content in sub-seconds by integrating multi-view diffusion models with scalable multi-view reconstructors. Current works further leverage 3D Gaussian Splatting as 3D representation for improved visual quality and rendering efficiency. However, we observe that existing Gaussian reconstruction models often suffer from multi-view inconsistency and blurred textures. We attribute this to the compromise of multi-view information propagation in favor of adopting powerful yet computationally intensive architectures (e.g., Transformers). To address this issue, we introduce MVGamba, a general and lightweight Gaussian reconstruction model featuring a multi-view Gaussian reconstructor based on the RNN-like State Space Model (SSM). Our Gaussian reconstructor propagates causal context containing multi-view information for cross-view self-refinement while generating a long sequence of Gaussians for fine-detail modeling with linear complexity. With off-the-shelf multi-view diffusion models integrated, MVGamba unifies 3D generation tasks from a single image, sparse images, or text prompts. Extensive experiments demonstrate that MVGamba outperforms state-of-the-art baselines in all 3D content generation scenarios with approximately only $0.1\times$ of the model size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10972v1">DCSEG: Decoupled 3D Open-Set Segmentation using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-14
    </div>
    <details class="paper-abstract">
      Open-set 3D segmentation represents a major point of interest for multiple downstream robotics and augmented/virtual reality applications. Recent advances introduce 3D Gaussian Splatting as a computationally efficient representation of the underlying scene. They enable the rendering of novel views while achieving real-time display rates and matching the quality of computationally far more expensive methods. We present a decoupled 3D segmentation pipeline to ensure modularity and adaptability to novel 3D representations and semantic segmentation foundation models. The pipeline proposes class-agnostic masks based on a 3D reconstruction of the scene. Given the resulting class-agnostic masks, we use a class-aware 2D foundation model to add class annotations to the 3D masks. We test this pipeline with 3D Gaussian Splatting and different 2D segmentation models and achieve better performance than more tailored approaches while also significantly increasing the modularity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01682v3">Diffusion Models with Anisotropic Gaussian Splatting for Image Inpainting</a></div>
    <div class="paper-meta">
      📅 2024-12-14
    </div>
    <details class="paper-abstract">
      Image inpainting is a fundamental task in computer vision, aiming to restore missing or corrupted regions in images realistically. While recent deep learning approaches have significantly advanced the state-of-the-art, challenges remain in maintaining structural continuity and generating coherent textures, particularly in large missing areas. Diffusion models have shown promise in generating high-fidelity images but often lack the structural guidance necessary for realistic inpainting. We propose a novel inpainting method that combines diffusion models with anisotropic Gaussian splatting to capture both local structures and global context effectively. By modeling missing regions using anisotropic Gaussian functions that adapt to local image gradients, our approach provides structural guidance to the diffusion-based inpainting network. The Gaussian splat maps are integrated into the diffusion process, enhancing the model's ability to generate high-fidelity and structurally coherent inpainting results. Extensive experiments demonstrate that our method outperforms state-of-the-art techniques, producing visually plausible results with enhanced structural integrity and texture realism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08475v2">Human-3Diffusion: Realistic Avatar Creation via Explicit 3D Consistent Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2024-12-14
      | 💬 Accepted to NeurIPS2024. Project Page: https://yuxuan-xue.com/human-3diffusion
    </div>
    <details class="paper-abstract">
      Creating realistic avatars from a single RGB image is an attractive yet challenging problem. Due to its ill-posed nature, recent works leverage powerful prior from 2D diffusion models pretrained on large datasets. Although 2D diffusion models demonstrate strong generalization capability, they cannot provide multi-view shape priors with guaranteed 3D consistency. We propose Human 3Diffusion: Realistic Avatar Creation via Explicit 3D Consistent Diffusion. Our key insight is that 2D multi-view diffusion and 3D reconstruction models provide complementary information for each other, and by coupling them in a tight manner, we can fully leverage the potential of both models. We introduce a novel image-conditioned generative 3D Gaussian Splats reconstruction model that leverages the priors from 2D multi-view diffusion models, and provides an explicit 3D representation, which further guides the 2D reverse sampling process to have better 3D consistency. Experiments show that our proposed framework outperforms state-of-the-art methods and enables the creation of realistic avatars from a single RGB image, achieving high-fidelity in both geometry and appearance. Extensive ablations also validate the efficacy of our design, (1) multi-view 2D priors conditioning in generative 3D reconstruction and (2) consistency refinement of sampling trajectory via the explicit 3D representation. Our code and models will be released on https://yuxuan-xue.com/human-3diffusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10231v1">SuperGSeg: Open-Vocabulary 3D Segmentation with Structured Super-Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-12-13
      | 💬 13 pages, 8 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has recently gained traction for its efficient training and real-time rendering. While the vanilla Gaussian Splatting representation is mainly designed for view synthesis, more recent works investigated how to extend it with scene understanding and language features. However, existing methods lack a detailed comprehension of scenes, limiting their ability to segment and interpret complex structures. To this end, We introduce SuperGSeg, a novel approach that fosters cohesive, context-aware scene representation by disentangling segmentation and language field distillation. SuperGSeg first employs neural Gaussians to learn instance and hierarchical segmentation features from multi-view images with the aid of off-the-shelf 2D masks. These features are then leveraged to create a sparse set of what we call Super-Gaussians. Super-Gaussians facilitate the distillation of 2D language features into 3D space. Through Super-Gaussians, our method enables high-dimensional language feature rendering without extreme increases in GPU memory. Extensive experiments demonstrate that SuperGSeg outperforms prior works on both open-vocabulary object localization and semantic segmentation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10209v1">GAF: Gaussian Avatar Reconstruction from Monocular Videos via Multi-view Diffusion</a></div>
    <div class="paper-meta">
      📅 2024-12-13
      | 💬 Paper Video: https://youtu.be/QuIYTljvhyg Project Page: https://tangjiapeng.github.io/projects/GAF
    </div>
    <details class="paper-abstract">
      We propose a novel approach for reconstructing animatable 3D Gaussian avatars from monocular videos captured by commodity devices like smartphones. Photorealistic 3D head avatar reconstruction from such recordings is challenging due to limited observations, which leaves unobserved regions under-constrained and can lead to artifacts in novel views. To address this problem, we introduce a multi-view head diffusion model, leveraging its priors to fill in missing regions and ensure view consistency in Gaussian splatting renderings. To enable precise viewpoint control, we use normal maps rendered from FLAME-based head reconstruction, which provides pixel-aligned inductive biases. We also condition the diffusion model on VAE features extracted from the input image to preserve details of facial identity and appearance. For Gaussian avatar reconstruction, we distill multi-view diffusion priors by using iteratively denoised images as pseudo-ground truths, effectively mitigating over-saturation issues. To further improve photorealism, we apply latent upsampling to refine the denoised latent before decoding it into an image. We evaluate our method on the NeRSemble dataset, showing that GAF outperforms the previous state-of-the-art methods in novel view synthesis by a 5.34\% higher SSIM score. Furthermore, we demonstrate higher-fidelity avatar reconstructions from monocular videos captured on commodity devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10051v1">TSGaussian: Semantic and Depth-Guided Target-Specific Gaussian Splatting from Sparse Views</a></div>
    <div class="paper-meta">
      📅 2024-12-13
    </div>
    <details class="paper-abstract">
      Recent advances in Gaussian Splatting have significantly advanced the field, achieving both panoptic and interactive segmentation of 3D scenes. However, existing methodologies often overlook the critical need for reconstructing specified targets with complex structures from sparse views. To address this issue, we introduce TSGaussian, a novel framework that combines semantic constraints with depth priors to avoid geometry degradation in challenging novel view synthesis tasks. Our approach prioritizes computational resources on designated targets while minimizing background allocation. Bounding boxes from YOLOv9 serve as prompts for Segment Anything Model to generate 2D mask predictions, ensuring semantic accuracy and cost efficiency. TSGaussian effectively clusters 3D gaussians by introducing a compact identity encoding for each Gaussian ellipsoid and incorporating 3D spatial consistency regularization. Leveraging these modules, we propose a pruning strategy to effectively reduce redundancy in 3D gaussians. Extensive experiments demonstrate that TSGaussian outperforms state-of-the-art methods on three standard datasets and a new challenging dataset we collected, achieving superior results in novel view synthesis of specific objects. Code is available at: https://github.com/leon2000-ai/TSGaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00477v3">LineGS : 3D Line Segment Representation on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-13
    </div>
    <details class="paper-abstract">
      Abstract representations of 3D scenes play a crucial role in computer vision, enabling a wide range of applications such as mapping, localization, surface reconstruction, and even advanced tasks like SLAM and rendering. Among these representations, line segments are widely used because of their ability to succinctly capture the structural features of a scene. However, existing 3D reconstruction methods often face significant challenges. Methods relying on 2D projections suffer from instability caused by errors in multi-view matching and occlusions, while direct 3D approaches are hampered by noise and sparsity in 3D point cloud data. This paper introduces LineGS, a novel method that combines geometry-guided 3D line reconstruction with a 3D Gaussian splatting model to address these challenges and improve representation ability. The method leverages the high-density Gaussian point distributions along the edge of the scene to refine and optimize initial line segments generated from traditional geometric approaches. By aligning these segments with the underlying geometric features of the scene, LineGS achieves a more precise and reliable representation of 3D structures. The results show significant improvements in both geometric accuracy and model compactness compared to baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09868v1">RP-SLAM: Real-time Photorealistic SLAM with Efficient 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-13
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has emerged as a promising technique for high-quality 3D rendering, leading to increasing interest in integrating 3DGS into realism SLAM systems. However, existing methods face challenges such as Gaussian primitives redundancy, forgetting problem during continuous optimization, and difficulty in initializing primitives in monocular case due to lack of depth information. In order to achieve efficient and photorealistic mapping, we propose RP-SLAM, a 3D Gaussian splatting-based vision SLAM method for monocular and RGB-D cameras. RP-SLAM decouples camera poses estimation from Gaussian primitives optimization and consists of three key components. Firstly, we propose an efficient incremental mapping approach to achieve a compact and accurate representation of the scene through adaptive sampling and Gaussian primitives filtering. Secondly, a dynamic window optimization method is proposed to mitigate the forgetting problem and improve map consistency. Finally, for the monocular case, a monocular keyframe initialization method based on sparse point cloud is proposed to improve the initialization accuracy of Gaussian primitives, which provides a geometric basis for subsequent optimization. The results of numerous experiments demonstrate that RP-SLAM achieves state-of-the-art map rendering accuracy while ensuring real-time performance and model compactness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09723v1">MAC-Ego3D: Multi-Agent Gaussian Consensus for Real-Time Collaborative Ego-Motion and Photorealistic 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-12-12
      | 💬 27 pages, 25 figures
    </div>
    <details class="paper-abstract">
      Real-time multi-agent collaboration for ego-motion estimation and high-fidelity 3D reconstruction is vital for scalable spatial intelligence. However, traditional methods produce sparse, low-detail maps, while recent dense mapping approaches struggle with high latency. To overcome these challenges, we present MAC-Ego3D, a novel framework for real-time collaborative photorealistic 3D reconstruction via Multi-Agent Gaussian Consensus. MAC-Ego3D enables agents to independently construct, align, and iteratively refine local maps using a unified Gaussian splat representation. Through Intra-Agent Gaussian Consensus, it enforces spatial coherence among neighboring Gaussian splats within an agent. For global alignment, parallelized Inter-Agent Gaussian Consensus, which asynchronously aligns and optimizes local maps by regularizing multi-agent Gaussian splats, seamlessly integrates them into a high-fidelity 3D model. Leveraging Gaussian primitives, MAC-Ego3D supports efficient RGB-D rendering, enabling rapid inter-agent Gaussian association and alignment. MAC-Ego3D bridges local precision and global coherence, delivering higher efficiency, largely reducing localization error, and improving mapping fidelity. It establishes a new SOTA on synthetic and real-world benchmarks, achieving a 15x increase in inference speed, order-of-magnitude reductions in ego-motion estimation error for partial cases, and RGB PSNR gains of 4 to 10 dB. Our code will be made publicly available at https://github.com/Xiaohao-Xu/MAC-Ego3D .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09680v1">PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields</a></div>
    <div class="paper-meta">
      📅 2024-12-12
      | 💬 16 pages, 7 figures. Code is publicly available at https://github.com/s3anwu/pbrnerf
    </div>
    <details class="paper-abstract">
      We tackle the ill-posed inverse rendering problem in 3D reconstruction with a Neural Radiance Field (NeRF) approach informed by Physics-Based Rendering (PBR) theory, named PBR-NeRF. Our method addresses a key limitation in most NeRF and 3D Gaussian Splatting approaches: they estimate view-dependent appearance without modeling scene materials and illumination. To address this limitation, we present an inverse rendering (IR) model capable of jointly estimating scene geometry, materials, and illumination. Our model builds upon recent NeRF-based IR approaches, but crucially introduces two novel physics-based priors that better constrain the IR estimation. Our priors are rigorously formulated as intuitive loss terms and achieve state-of-the-art material estimation without compromising novel view synthesis quality. Our method is easily adaptable to other inverse rendering and 3D reconstruction frameworks that require material estimation. We demonstrate the importance of extending current neural rendering approaches to fully model scene properties beyond geometry and view-dependent appearance. Code is publicly available at https://github.com/s3anwu/pbrnerf
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09606v1">Feat2GS: Probing Visual Foundation Models with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-12
      | 💬 Project Page: https://fanegg.github.io/Feat2GS/
    </div>
    <details class="paper-abstract">
      Given that visual foundation models (VFMs) are trained on extensive datasets but often limited to 2D images, a natural question arises: how well do they understand the 3D world? With the differences in architecture and training protocols (i.e., objectives, proxy tasks), a unified framework to fairly and comprehensively probe their 3D awareness is urgently needed. Existing works on 3D probing suggest single-view 2.5D estimation (e.g., depth and normal) or two-view sparse 2D correspondence (e.g., matching and tracking). Unfortunately, these tasks ignore texture awareness, and require 3D data as ground-truth, which limits the scale and diversity of their evaluation set. To address these issues, we introduce Feat2GS, which readout 3D Gaussians attributes from VFM features extracted from unposed images. This allows us to probe 3D awareness for geometry and texture via novel view synthesis, without requiring 3D data. Additionally, the disentanglement of 3DGS parameters - geometry ($\boldsymbol{x}, \alpha, \Sigma$) and texture ($\boldsymbol{c}$) - enables separate analysis of texture and geometry awareness. Under Feat2GS, we conduct extensive experiments to probe the 3D awareness of several VFMs, and investigate the ingredients that lead to a 3D aware VFM. Building on these findings, we develop several variants that achieve state-of-the-art across diverse datasets. This makes Feat2GS useful for probing VFMs, and as a simple-yet-effective baseline for novel-view synthesis. Code and data will be made available at https://fanegg.github.io/Feat2GS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09597v1">LiftImage3D: Lifting Any Single Image to 3D Gaussians with Video Generation Priors</a></div>
    <div class="paper-meta">
      📅 2024-12-12
      | 💬 Project page: https://liftimage3d.github.io/
    </div>
    <details class="paper-abstract">
      Single-image 3D reconstruction remains a fundamental challenge in computer vision due to inherent geometric ambiguities and limited viewpoint information. Recent advances in Latent Video Diffusion Models (LVDMs) offer promising 3D priors learned from large-scale video data. However, leveraging these priors effectively faces three key challenges: (1) degradation in quality across large camera motions, (2) difficulties in achieving precise camera control, and (3) geometric distortions inherent to the diffusion process that damage 3D consistency. We address these challenges by proposing LiftImage3D, a framework that effectively releases LVDMs' generative priors while ensuring 3D consistency. Specifically, we design an articulated trajectory strategy to generate video frames, which decomposes video sequences with large camera motions into ones with controllable small motions. Then we use robust neural matching models, i.e. MASt3R, to calibrate the camera poses of generated frames and produce corresponding point clouds. Finally, we propose a distortion-aware 3D Gaussian splatting representation, which can learn independent distortions between frames and output undistorted canonical Gaussians. Extensive experiments demonstrate that LiftImage3D achieves state-of-the-art performance on two challenging datasets, i.e. LLFF, DL3DV, and Tanks and Temples, and generalizes well to diverse in-the-wild images, from cartoon illustrations to complex real-world scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09573v1">FreeSplatter: Pose-free Gaussian Splatting for Sparse-view 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-12-12
      | 💬 Project page: https://bluestyle97.github.io/projects/freesplatter/
    </div>
    <details class="paper-abstract">
      Existing sparse-view reconstruction models heavily rely on accurate known camera poses. However, deriving camera extrinsics and intrinsics from sparse-view images presents significant challenges. In this work, we present FreeSplatter, a highly scalable, feed-forward reconstruction framework capable of generating high-quality 3D Gaussians from uncalibrated sparse-view images and recovering their camera parameters in mere seconds. FreeSplatter is built upon a streamlined transformer architecture, comprising sequential self-attention blocks that facilitate information exchange among multi-view image tokens and decode them into pixel-wise 3D Gaussian primitives. The predicted Gaussian primitives are situated in a unified reference frame, allowing for high-fidelity 3D modeling and instant camera parameter estimation using off-the-shelf solvers. To cater to both object-centric and scene-level reconstruction, we train two model variants of FreeSplatter on extensive datasets. In both scenarios, FreeSplatter outperforms state-of-the-art baselines in terms of reconstruction quality and pose estimation accuracy. Furthermore, we showcase FreeSplatter's potential in enhancing the productivity of downstream applications, such as text/image-to-3D content creation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09511v1">GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency</a></div>
    <div class="paper-meta">
      📅 2024-12-12
      | 💬 22 pages, 8 figures, 12 tables; Project Page at https://dylanorange.github.io/projects/geal
    </div>
    <details class="paper-abstract">
      Identifying affordance regions on 3D objects from semantic cues is essential for robotics and human-machine interaction. However, existing 3D affordance learning methods struggle with generalization and robustness due to limited annotated data and a reliance on 3D backbones focused on geometric encoding, which often lack resilience to real-world noise and data corruption. We propose GEAL, a novel framework designed to enhance the generalization and robustness of 3D affordance learning by leveraging large-scale pre-trained 2D models. We employ a dual-branch architecture with Gaussian splatting to establish consistent mappings between 3D point clouds and 2D representations, enabling realistic 2D renderings from sparse point clouds. A granularity-adaptive fusion module and a 2D-3D consistency alignment module further strengthen cross-modal alignment and knowledge transfer, allowing the 3D branch to benefit from the rich semantics and generalization capacity of 2D models. To holistically assess the robustness, we introduce two new corruption-based benchmarks: PIAD-C and LASO-C. Extensive experiments on public datasets and our benchmarks show that GEAL consistently outperforms existing methods across seen and novel object categories, as well as corrupted data, demonstrating robust and adaptable affordance prediction under diverse conditions. Code and corruption datasets have been made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11447v3">GaussianOcc: Fully Self-supervised and Efficient 3D Occupancy Estimation with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-12
      | 💬 Project page: https://ganwanshui.github.io/GaussianOcc/
    </div>
    <details class="paper-abstract">
      We introduce GaussianOcc, a systematic method that investigates the two usages of Gaussian splatting for fully self-supervised and efficient 3D occupancy estimation in surround views. First, traditional methods for self-supervised 3D occupancy estimation still require ground truth 6D poses from sensors during training. To address this limitation, we propose Gaussian Splatting for Projection (GSP) module to provide accurate scale information for fully self-supervised training from adjacent view projection. Additionally, existing methods rely on volume rendering for final 3D voxel representation learning using 2D signals (depth maps, semantic maps), which is both time-consuming and less effective. We propose Gaussian Splatting from Voxel space (GSV) to leverage the fast rendering properties of Gaussian splatting. As a result, the proposed GaussianOcc method enables fully self-supervised (no ground truth pose) 3D occupancy estimation in competitive performance with low computational cost (2.7 times faster in training and 5 times faster in rendering). The relevant code is available in https://github.com/GANWANSHUI/GaussianOcc.git.
    </details>
</div>
