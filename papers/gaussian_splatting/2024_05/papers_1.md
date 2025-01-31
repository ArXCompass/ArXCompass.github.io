# gaussian splatting - 2024_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20721v1">ContextGS: Compact 3D Gaussian Splatting with Anchor Level Context Model</a></div>
    <div class="paper-meta">
      📅 2024-05-31
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has become a promising framework for novel view synthesis, offering fast rendering speeds and high fidelity. However, the large number of Gaussians and their associated attributes require effective compression techniques. Existing methods primarily compress neural Gaussians individually and independently, i.e., coding all the neural Gaussians at the same time, with little design for their interactions and spatial dependence. Inspired by the effectiveness of the context model in image compression, we propose the first autoregressive model at the anchor level for 3DGS compression in this work. We divide anchors into different levels and the anchors that are not coded yet can be predicted based on the already coded ones in all the coarser levels, leading to more accurate modeling and higher coding efficiency. To further improve the efficiency of entropy coding, e.g., to code the coarsest level with no already coded anchors, we propose to introduce a low-dimensional quantized feature as the hyperprior for each anchor, which can be effectively compressed. Our work pioneers the context model in the anchor level for 3DGS representation, yielding an impressive size reduction of over 100 times compared to vanilla 3DGS and 15 times compared to the most recent state-of-the-art work Scaffold-GS, while achieving comparable or even higher rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11129v2">MotionGS : Compact Gaussian Splatting SLAM by Motion Filter</a></div>
    <div class="paper-meta">
      📅 2024-05-31
    </div>
    <details class="paper-abstract">
      With their high-fidelity scene representation capability, the attention of SLAM field is deeply attracted by the Neural Radiation Field (NeRF) and 3D Gaussian Splatting (3DGS). Recently, there has been a surge in NeRF-based SLAM, while 3DGS-based SLAM is sparse. A novel 3DGS-based SLAM approach with a fusion of deep visual feature, dual keyframe selection and 3DGS is presented in this paper. Compared with the existing methods, the proposed tracking is achieved by feature extraction and motion filter on each frame. The joint optimization of poses and 3D Gaussians runs through the entire mapping process. Additionally, the coarse-to-fine pose estimation and compact Gaussian scene representation are implemented by dual keyframe selection and novel loss functions. Experimental results demonstrate that the proposed algorithm not only outperforms the existing methods in tracking and mapping, but also has less memory usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20323v1">$\textit{S}^3$Gaussian: Self-Supervised Street Gaussians for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2024-05-30
      | 💬 Code is available at: https://github.com/nnanhuang/S3Gaussian/
    </div>
    <details class="paper-abstract">
      Photorealistic 3D reconstruction of street scenes is a critical technique for developing real-world simulators for autonomous driving. Despite the efficacy of Neural Radiance Fields (NeRF) for driving scenes, 3D Gaussian Splatting (3DGS) emerges as a promising direction due to its faster speed and more explicit representation. However, most existing street 3DGS methods require tracked 3D vehicle bounding boxes to decompose the static and dynamic elements for effective reconstruction, limiting their applications for in-the-wild scenarios. To facilitate efficient 3D scene reconstruction without costly annotations, we propose a self-supervised street Gaussian ($\textit{S}^3$Gaussian) method to decompose dynamic and static elements from 4D consistency. We represent each scene with 3D Gaussians to preserve the explicitness and further accompany them with a spatial-temporal field network to compactly model the 4D dynamics. We conduct extensive experiments on the challenging Waymo-Open dataset to evaluate the effectiveness of our method. Our $\textit{S}^3$Gaussian demonstrates the ability to decompose static and dynamic scenes and achieves the best performance without using 3D annotations. Code is available at: https://github.com/nnanhuang/S3Gaussian/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18163v2">NegGS: Negative Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-05-30
    </div>
    <details class="paper-abstract">
      One of the key advantages of 3D rendering is its ability to simulate intricate scenes accurately. One of the most widely used methods for this purpose is Gaussian Splatting, a novel approach that is known for its rapid training and inference capabilities. In essence, Gaussian Splatting involves incorporating data about the 3D objects of interest into a series of Gaussian distributions, each of which can then be depicted in 3D in a manner analogous to traditional meshes. It is regrettable that the use of Gaussians in Gaussian Splatting is currently somewhat restrictive due to their perceived linear nature. In practice, 3D objects are often composed of complex curves and highly nonlinear structures. This issue can to some extent be alleviated by employing a multitude of Gaussian components to reflect the complex, nonlinear structures accurately. However, this approach results in a considerable increase in time complexity. This paper introduces the concept of negative Gaussians, which are interpreted as items with negative colors. The rationale behind this approach is based on the density distribution created by dividing the probability density functions (PDFs) of two Gaussians, which we refer to as Diff-Gaussian. Such a distribution can be used to approximate structures such as donut and moon-shaped datasets. Experimental findings indicate that the application of these techniques enhances the modeling of high-frequency elements with rapid color transitions. Additionally, it improves the representation of shadows. To the best of our knowledge, this is the first paper to extend the simple elipsoid shapes of Gaussian Splatting to more complex nonlinear structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17835v3">Deform3DGS: Flexible Deformation for Fast Surgical Scene Reconstruction with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-05-30
      | 💬 Early accepted at MICCAI 2024, 10 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Tissue deformation poses a key challenge for accurate surgical scene reconstruction. Despite yielding high reconstruction quality, existing methods suffer from slow rendering speeds and long training times, limiting their intraoperative applicability. Motivated by recent progress in 3D Gaussian Splatting, an emerging technology in real-time 3D rendering, this work presents a novel fast reconstruction framework, termed Deform3DGS, for deformable tissues during endoscopic surgery. Specifically, we introduce 3D GS into surgical scenes by integrating a point cloud initialization to improve reconstruction. Furthermore, we propose a novel flexible deformation modeling scheme (FDM) to learn tissue deformation dynamics at the level of individual Gaussians. Our FDM can model the surface deformation with efficient representations, allowing for real-time rendering performance. More importantly, FDM significantly accelerates surgical scene reconstruction, demonstrating considerable clinical values, particularly in intraoperative settings where time efficiency is crucial. Experiments on DaVinci robotic surgery videos indicate the efficacy of our approach, showcasing superior reconstruction fidelity PSNR: (37.90) and rendering speed (338.8 FPS) while substantially reducing training time to only 1 minute/scene. Our code is available at https://github.com/jinlab-imvr/Deform3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18416v2">3D StreetUnveiler with Semantic-Aware 2DGS</a></div>
    <div class="paper-meta">
      📅 2024-05-30
      | 💬 Project page: https://streetunveiler.github.io
    </div>
    <details class="paper-abstract">
      Unveiling an empty street from crowded observations captured by in-car cameras is crucial for autonomous driving. However, removing all temporarily static objects, such as stopped vehicles and standing pedestrians, presents a significant challenge. Unlike object-centric 3D inpainting, which relies on thorough observation in a small scene, street scene cases involve long trajectories that differ from previous 3D inpainting tasks. The camera-centric moving environment of captured videos further complicates the task due to the limited degree and time duration of object observation. To address these obstacles, we introduce StreetUnveiler to reconstruct an empty street. StreetUnveiler learns a 3D representation of the empty street from crowded observations. Our representation is based on the hard-label semantic 2D Gaussian Splatting (2DGS) for its scalability and ability to identify Gaussians to be removed. We inpaint rendered image after removing unwanted Gaussians to provide pseudo-labels and subsequently re-optimize the 2DGS. Given its temporal continuous movement, we divide the empty street scene into observed, partial-observed, and unobserved regions, which we propose to locate through a rendered alpha map. This decomposition helps us to minimize the regions that need to be inpainted. To enhance the temporal consistency of the inpainting, we introduce a novel time-reversal framework to inpaint frames in reverse order and use later frames as references for earlier frames to fully utilize the long-trajectory observations. Our experiments conducted on the street scene dataset successfully reconstructed a 3D representation of the empty street. The mesh representation of the empty street can be extracted for further applications. The project page and more visualizations can be found at: https://streetunveiler.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19671v1">GaussianRoom: Improving 3D Gaussian Splatting with SDF Guidance and Monocular Cues for Indoor Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-05-30
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting(3DGS) has revolutionized neural rendering with its high-quality rendering and real-time speed. However, when it comes to indoor scenes with a significant number of textureless areas, 3DGS yields incomplete and noisy reconstruction results due to the poor initialization of the point cloud and under-constrained optimization. Inspired by the continuity of signed distance field (SDF), which naturally has advantages in modeling surfaces, we present a unified optimizing framework integrating neural SDF with 3DGS. This framework incorporates a learnable neural SDF field to guide the densification and pruning of Gaussians, enabling Gaussians to accurately model scenes even with poor initialized point clouds. At the same time, the geometry represented by Gaussians improves the efficiency of the SDF field by piloting its point sampling. Additionally, we regularize the optimization with normal and edge priors to eliminate geometry ambiguity in textureless areas and improve the details. Extensive experiments in ScanNet and ScanNet++ show that our method achieves state-of-the-art performance in both surface reconstruction and novel view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19657v1">Uncertainty-guided Optimal Transport in Depth Supervised Sparse-View 3D Gaussian</a></div>
    <div class="paper-meta">
      📅 2024-05-30
      | 💬 10pages
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting has demonstrated impressive performance in real-time novel view synthesis. However, achieving successful reconstruction from RGB images generally requires multiple input views captured under static conditions. To address the challenge of sparse input views, previous approaches have incorporated depth supervision into the training of 3D Gaussians to mitigate overfitting, using dense predictions from pretrained depth networks as pseudo-ground truth. Nevertheless, depth predictions from monocular depth estimation models inherently exhibit significant uncertainty in specific areas. Relying solely on pixel-wise L2 loss may inadvertently incorporate detrimental noise from these uncertain areas. In this work, we introduce a novel method to supervise the depth distribution of 3D Gaussians, utilizing depth priors with integrated uncertainty estimates. To address these localized errors in depth predictions, we integrate a patch-wise optimal transport strategy to complement traditional L2 loss in depth supervision. Extensive experiments conducted on the LLFF, DTU, and Blender datasets demonstrate that our approach, UGOT, achieves superior novel view synthesis and consistently outperforms state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19614v1">TAMBRIDGE: Bridging Frame-Centered Tracking and 3D Gaussian Splatting for Enhanced SLAM</a></div>
    <div class="paper-meta">
      📅 2024-05-30
    </div>
    <details class="paper-abstract">
      The limited robustness of 3D Gaussian Splatting (3DGS) to motion blur and camera noise, along with its poor real-time performance, restricts its application in robotic SLAM tasks. Upon analysis, the primary causes of these issues are the density of views with motion blur and the cumulative errors in dense pose estimation from calculating losses based on noisy original images and rendering results, which increase the difficulty of 3DGS rendering convergence. Thus, a cutting-edge 3DGS-based SLAM system is introduced, leveraging the efficiency and flexibility of 3DGS to achieve real-time performance while remaining robust against sensor noise, motion blur, and the challenges posed by long-session SLAM. Central to this approach is the Fusion Bridge module, which seamlessly integrates tracking-centered ORB Visual Odometry with mapping-centered online 3DGS. Precise pose initialization is enabled by this module through joint optimization of re-projection and rendering loss, as well as strategic view selection, enhancing rendering convergence in large-scale scenes. Extensive experiments demonstrate state-of-the-art rendering quality and localization accuracy, positioning this system as a promising solution for real-world robotics applications that require stable, near-real-time performance. Our project is available at https://ZeldaFromHeaven.github.io/TAMBRIDGE/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17187v2">Memorize What Matters: Emergent Scene Decomposition from Multitraverse</a></div>
    <div class="paper-meta">
      📅 2024-05-29
      | 💬 Project page: https://3d-gaussian-mapping.github.io; Code and data: https://github.com/NVlabs/3DGM
    </div>
    <details class="paper-abstract">
      Humans naturally retain memories of permanent elements, while ephemeral moments often slip through the cracks of memory. This selective retention is crucial for robotic perception, localization, and mapping. To endow robots with this capability, we introduce 3D Gaussian Mapping (3DGM), a self-supervised, camera-only offline mapping framework grounded in 3D Gaussian Splatting. 3DGM converts multitraverse RGB videos from the same region into a Gaussian-based environmental map while concurrently performing 2D ephemeral object segmentation. Our key observation is that the environment remains consistent across traversals, while objects frequently change. This allows us to exploit self-supervision from repeated traversals to achieve environment-object decomposition. More specifically, 3DGM formulates multitraverse environmental mapping as a robust differentiable rendering problem, treating pixels of the environment and objects as inliers and outliers, respectively. Using robust feature distillation, feature residuals mining, and robust optimization, 3DGM jointly performs 2D segmentation and 3D mapping without human intervention. We build the Mapverse benchmark, sourced from the Ithaca365 and nuPlan datasets, to evaluate our method in unsupervised 2D segmentation, 3D reconstruction, and neural rendering. Extensive results verify the effectiveness and potential of our method for self-driving and robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16829v3">PyGS: Large-scale Scene Representation with Pyramidal 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-05-29
    </div>
    <details class="paper-abstract">
      Neural Radiance Fields (NeRFs) have demonstrated remarkable proficiency in synthesizing photorealistic images of large-scale scenes. However, they are often plagued by a loss of fine details and long rendering durations. 3D Gaussian Splatting has recently been introduced as a potent alternative, achieving both high-fidelity visual results and accelerated rendering performance. Nonetheless, scaling 3D Gaussian Splatting is fraught with challenges. Specifically, large-scale scenes grapples with the integration of objects across multiple scales and disparate viewpoints, which often leads to compromised efficacy as the Gaussians need to balance between detail levels. Furthermore, the generation of initialization points via COLMAP from large-scale dataset is both computationally demanding and prone to incomplete reconstructions. To address these challenges, we present Pyramidal 3D Gaussian Splatting (PyGS) with NeRF Initialization. Our approach represent the scene with a hierarchical assembly of Gaussians arranged in a pyramidal fashion. The top level of the pyramid is composed of a few large Gaussians, while each subsequent layer accommodates a denser collection of smaller Gaussians. We effectively initialize these pyramidal Gaussians through sampling a rapidly trained grid-based NeRF at various frequencies. We group these pyramidal Gaussians into clusters and use a compact weighting network to dynamically determine the influence of each pyramid level of each cluster considering camera viewpoint during rendering. Our method achieves a significant performance leap across multiple large-scale datasets and attains a rendering time that is over 400 times faster than current state-of-the-art approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.07494v3">SemGauss-SLAM: Dense Semantic Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2024-05-29
    </div>
    <details class="paper-abstract">
      We propose SemGauss-SLAM, a dense semantic SLAM system utilizing 3D Gaussian representation, that enables accurate 3D semantic mapping, robust camera tracking, and high-quality rendering simultaneously. In this system, we incorporate semantic feature embedding into 3D Gaussian representation, which effectively encodes semantic information within the spatial layout of the environment for precise semantic scene representation. Furthermore, we propose feature-level loss for updating 3D Gaussian representation, enabling higher-level guidance for 3D Gaussian optimization. In addition, to reduce cumulative drift in tracking and improve semantic reconstruction accuracy, we introduce semantic-informed bundle adjustment leveraging multi-frame semantic associations for joint optimization of 3D Gaussian representation and camera poses, leading to low-drift tracking and accurate mapping. Our SemGauss-SLAM method demonstrates superior performance over existing radiance field-based SLAM methods in terms of mapping and tracking accuracy on Replica and ScanNet datasets, while also showing excellent capabilities in high-precision semantic segmentation and dense semantic mapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18784v1">LP-3DGS: Learning to Prune 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-05-29
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has become one of the mainstream methodologies for novel view synthesis (NVS) due to its high quality and fast rendering speed. However, as a point-based scene representation, 3DGS potentially generates a large number of Gaussians to fit the scene, leading to high memory usage. Improvements that have been proposed require either an empirical and preset pruning ratio or importance score threshold to prune the point cloud. Such hyperparamter requires multiple rounds of training to optimize and achieve the maximum pruning ratio, while maintaining the rendering quality for each scene. In this work, we propose learning-to-prune 3DGS (LP-3DGS), where a trainable binary mask is applied to the importance score that can find optimal pruning ratio automatically. Instead of using the traditional straight-through estimator (STE) method to approximate the binary mask gradient, we redesign the masking function to leverage the Gumbel-Sigmoid method, making it differentiable and compatible with the existing training process of 3DGS. Extensive experiments have shown that LP-3DGS consistently produces a good balance that is both efficient and high quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.17089v2">Multi-Scale 3D Gaussian Splatting for Anti-Aliased Rendering</a></div>
    <div class="paper-meta">
      📅 2024-05-29
      | 💬 CVPR 2024
    </div>
    <details class="paper-abstract">
      3D Gaussians have recently emerged as a highly efficient representation for 3D reconstruction and rendering. Despite its high rendering quality and speed at high resolutions, they both deteriorate drastically when rendered at lower resolutions or from far away camera position. During low resolution or far away rendering, the pixel size of the image can fall below the Nyquist frequency compared to the screen size of each splatted 3D Gaussian and leads to aliasing effect. The rendering is also drastically slowed down by the sequential alpha blending of more splatted Gaussians per pixel. To address these issues, we propose a multi-scale 3D Gaussian splatting algorithm, which maintains Gaussians at different scales to represent the same scene. Higher-resolution images are rendered with more small Gaussians, and lower-resolution images are rendered with fewer larger Gaussians. With similar training time, our algorithm can achieve 13\%-66\% PSNR and 160\%-2400\% rendering speed improvement at 4$\times$-128$\times$ scale rendering on Mip-NeRF360 dataset compared to the single scale 3D Gaussian splitting. Our code and more results are available on our project website https://jokeryan.github.io/projects/ms-gs/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18424v1">3DitScene: Editing Any Scene via Language-guided Disentangled Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-05-28
    </div>
    <details class="paper-abstract">
      Scene image editing is crucial for entertainment, photography, and advertising design. Existing methods solely focus on either 2D individual object or 3D global scene editing. This results in a lack of a unified approach to effectively control and manipulate scenes at the 3D level with different levels of granularity. In this work, we propose 3DitScene, a novel and unified scene editing framework leveraging language-guided disentangled Gaussian Splatting that enables seamless editing from 2D to 3D, allowing precise control over scene composition and individual objects. We first incorporate 3D Gaussians that are refined through generative priors and optimization techniques. Language features from CLIP then introduce semantics into 3D geometry for object disentanglement. With the disentangled Gaussians, 3DitScene allows for manipulation at both the global and individual levels, revolutionizing creative expression and empowering control over scenes and objects. Experimental results demonstrate the effectiveness and versatility of 3DitScene in scene image editing. Code and online demo can be found at our project homepage: https://zqh0253.github.io/3DitScene/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17083v2">F-3DGS: Factorized Coordinates and Representations for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-05-28
      | 💬 Our project page including code is available at https://xiangyu1sun.github.io/Factorize-3DGS/
    </div>
    <details class="paper-abstract">
      The neural radiance field (NeRF) has made significant strides in representing 3D scenes and synthesizing novel views. Despite its advancements, the high computational costs of NeRF have posed challenges for its deployment in resource-constrained environments and real-time applications. As an alternative to NeRF-like neural rendering methods, 3D Gaussian Splatting (3DGS) offers rapid rendering speeds while maintaining excellent image quality. However, as it represents objects and scenes using a myriad of Gaussians, it requires substantial storage to achieve high-quality representation. To mitigate the storage overhead, we propose Factorized 3D Gaussian Splatting (F-3DGS), a novel approach that drastically reduces storage requirements while preserving image quality. Inspired by classical matrix and tensor factorization techniques, our method represents and approximates dense clusters of Gaussians with significantly fewer Gaussians through efficient factorization. We aim to efficiently represent dense 3D Gaussians by approximating them with a limited amount of information for each axis and their combinations. This method allows us to encode a substantially large number of Gaussians along with their essential attributes -- such as color, scale, and rotation -- necessary for rendering using a relatively small number of elements. Extensive experimental results demonstrate that F-3DGS achieves a significant reduction in storage costs while maintaining comparable quality in rendered images.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09413v2">Relaxing Accurate Initialization Constraint for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-05-28
      | 💬 Project Page: https://ku-cvlab.github.io/RAIN-GS
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has recently demonstrated impressive capabilities in real-time novel view synthesis and 3D reconstruction. However, 3DGS heavily depends on the accurate initialization derived from Structure-from-Motion (SfM) methods. When the quality of the initial point cloud deteriorates, such as in the presence of noise or when using randomly initialized point cloud, 3DGS often undergoes large performance drops. To address this limitation, we propose a novel optimization strategy dubbed RAIN-GS (Relaing Accurate Initialization Constraint for 3D Gaussian Splatting). Our approach is based on an in-depth analysis of the original 3DGS optimization scheme and the analysis of the SfM initialization in the frequency domain. Leveraging simple modifications based on our analyses, RAIN-GS successfully trains 3D Gaussians from sub-optimal point cloud (e.g., randomly initialized point cloud), effectively relaxing the need for accurate initialization. We demonstrate the efficacy of our strategy through quantitative and qualitative comparisons on multiple datasets, where RAIN-GS trained with random point cloud achieves performance on-par with or even better than 3DGS trained with accurate SfM point cloud. Our project page and code can be found at https://ku-cvlab.github.io/RAIN-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18133v1">A Grid-Free Fluid Solver based on Gaussian Spatial Representation</a></div>
    <div class="paper-meta">
      📅 2024-05-28
    </div>
    <details class="paper-abstract">
      We present a grid-free fluid solver featuring a novel Gaussian representation. Drawing inspiration from the expressive capabilities of 3D Gaussian Splatting in multi-view image reconstruction, we model the continuous flow velocity as a weighted sum of multiple Gaussian functions. Leveraging this representation, we derive differential operators for the field and implement a time-dependent PDE solver using the traditional operator splitting method. Compared to implicit neural representations as another continuous spatial representation with increasing attention, our method with flexible 3D Gaussians presents enhanced accuracy on vorticity preservation. Moreover, we apply physics-driven strategies to accelerate the optimization-based time integration of Gaussian functions. This temporal evolution surpasses previous work based on implicit neural representation with reduced computational time and memory. Although not surpassing the quality of state-of-the-art Eulerian methods in fluid simulation, experiments and ablation studies indicate the potential of our memory-efficient representation. With enriched spatial information, our method exhibits a distinctive perspective combining the advantages of Eulerian and Lagrangian approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18132v1">EG4D: Explicit Generation of 4D Object without Score Distillation</a></div>
    <div class="paper-meta">
      📅 2024-05-28
    </div>
    <details class="paper-abstract">
      In recent years, the increasing demand for dynamic 3D assets in design and gaming applications has given rise to powerful generative pipelines capable of synthesizing high-quality 4D objects. Previous methods generally rely on score distillation sampling (SDS) algorithm to infer the unseen views and motion of 4D objects, thus leading to unsatisfactory results with defects like over-saturation and Janus problem. Therefore, inspired by recent progress of video diffusion models, we propose to optimize a 4D representation by explicitly generating multi-view videos from one input image. However, it is far from trivial to handle practical challenges faced by such a pipeline, including dramatic temporal inconsistency, inter-frame geometry and texture diversity, and semantic defects brought by video generation results. To address these issues, we propose DG4D, a novel multi-stage framework that generates high-quality and consistent 4D assets without score distillation. Specifically, collaborative techniques and solutions are developed, including an attention injection strategy to synthesize temporal-consistent multi-view videos, a robust and efficient dynamic reconstruction method based on Gaussian Splatting, and a refinement stage with diffusion prior for semantic restoration. The qualitative results and user preference study demonstrate that our framework outperforms the baselines in generation quality by a considerable margin. Code will be released at \url{https://github.com/jasongzy/EG4D}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16923v2">SA-GS: Semantic-Aware Gaussian Splatting for Large Scene Reconstruction with Geometry Constrain</a></div>
    <div class="paper-meta">
      📅 2024-05-28
      | 💬 Might need more comparison, will be add later
    </div>
    <details class="paper-abstract">
      With the emergence of Gaussian Splats, recent efforts have focused on large-scale scene geometric reconstruction. However, most of these efforts either concentrate on memory reduction or spatial space division, neglecting information in the semantic space. In this paper, we propose a novel method, named SA-GS, for fine-grained 3D geometry reconstruction using semantic-aware 3D Gaussian Splats. Specifically, we leverage prior information stored in large vision models such as SAM and DINO to generate semantic masks. We then introduce a geometric complexity measurement function to serve as soft regularization, guiding the shape of each Gaussian Splat within specific semantic areas. Additionally, we present a method that estimates the expected number of Gaussian Splats in different semantic areas, effectively providing a lower bound for Gaussian Splats in these areas. Subsequently, we extract the point cloud using a novel probability density-based extraction method, transforming Gaussian Splats into a point cloud crucial for downstream tasks. Our method also offers the potential for detailed semantic inquiries while maintaining high image-based reconstruction results. We provide extensive experiments on publicly available large-scale scene reconstruction datasets with highly accurate point clouds as ground truth and our novel dataset. Our results demonstrate the superiority of our method over current state-of-the-art Gaussian Splats reconstruction methods by a significant margin in terms of geometric-based measurement metrics. Code and additional results will soon be available on our project page.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17891v1">A Refined 3D Gaussian Representation for High-Quality Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-05-28
    </div>
    <details class="paper-abstract">
      In recent years, Neural Radiance Fields (NeRF) has revolutionized three-dimensional (3D) reconstruction with its implicit representation. Building upon NeRF, 3D Gaussian Splatting (3D-GS) has departed from the implicit representation of neural networks and instead directly represents scenes as point clouds with Gaussian-shaped distributions. While this shift has notably elevated the rendering quality and speed of radiance fields but inevitably led to a significant increase in memory usage. Additionally, effectively rendering dynamic scenes in 3D-GS has emerged as a pressing challenge. To address these concerns, this paper purposes a refined 3D Gaussian representation for high-quality dynamic scene reconstruction. Firstly, we use a deformable multi-layer perceptron (MLP) network to capture the dynamic offset of Gaussian points and express the color features of points through hash encoding and a tiny MLP to reduce storage requirements. Subsequently, we introduce a learnable denoising mask coupled with denoising loss to eliminate noise points from the scene, thereby further compressing 3D Gaussian model. Finally, motion noise of points is mitigated through static constraints and motion consistency constraints. Experimental results demonstrate that our method surpasses existing approaches in rendering quality and speed, while significantly reducing the memory usage associated with 3D-GS, making it highly suitable for various tasks such as novel view synthesis, and dynamic mapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17811v1">Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh</a></div>
    <div class="paper-meta">
      📅 2024-05-28
      | 💬 Project page here: https://gaoxiangjun.github.io/mani_gs/
    </div>
    <details class="paper-abstract">
      Neural 3D representations such as Neural Radiance Fields (NeRF), excel at producing photo-realistic rendering results but lack the flexibility for manipulation and editing which is crucial for content creation. Previous works have attempted to address this issue by deforming a NeRF in canonical space or manipulating the radiance field based on an explicit mesh. However, manipulating NeRF is not highly controllable and requires a long training and inference time. With the emergence of 3D Gaussian Splatting (3DGS), extremely high-fidelity novel view synthesis can be achieved using an explicit point-based 3D representation with much faster training and rendering speed. However, there is still a lack of effective means to manipulate 3DGS freely while maintaining rendering quality. In this work, we aim to tackle the challenge of achieving manipulable photo-realistic rendering. We propose to utilize a triangular mesh to manipulate 3DGS directly with self-adaptation. This approach reduces the need to design various algorithms for different types of Gaussian manipulation. By utilizing a triangle shape-aware Gaussian binding and adapting method, we can achieve 3DGS manipulation and preserve high-fidelity rendering after manipulation. Our approach is capable of handling large deformations, local manipulations, and soft body simulations while keeping high-quality rendering. Furthermore, we demonstrate that our method is also effective with inaccurate meshes extracted from 3DGS. Experiments conducted demonstrate the effectiveness of our method and its superiority over baseline approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17351v1">DOF-GS: Adjustable Depth-of-Field 3D Gaussian Splatting for Refocusing,Defocus Rendering and Blur Removal</a></div>
    <div class="paper-meta">
      📅 2024-05-27
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting-based techniques have recently advanced 3D scene reconstruction and novel view synthesis, achieving high-quality real-time rendering. However, these approaches are inherently limited by the underlying pinhole camera assumption in modeling the images and hence only work for All-in-Focus (AiF) sharp image inputs. This severely affects their applicability in real-world scenarios where images often exhibit defocus blur due to the limited depth-of-field (DOF) of imaging devices. Additionally, existing 3D Gaussian Splatting (3DGS) methods also do not support rendering of DOF effects. To address these challenges, we introduce DOF-GS that allows for rendering adjustable DOF effects, removing defocus blur as well as refocusing of 3D scenes, all from multi-view images degraded by defocus blur. To this end, we re-imagine the traditional Gaussian Splatting pipeline by employing a finite aperture camera model coupled with explicit, differentiable defocus rendering guided by the Circle-of-Confusion (CoC). The proposed framework provides for dynamic adjustment of DOF effects by changing the aperture and focal distance of the underlying camera model on-demand. It also enables rendering varying DOF effects of 3D scenes post-optimization, and generating AiF images from defocused training images. Furthermore, we devise a joint optimization strategy to further enhance details in the reconstructed scenes by jointly optimizing rendered defocused and AiF images. Our experimental results indicate that DOF-GS produces high-quality sharp all-in-focus renderings conditioned on inputs compromised by defocus blur, with the training process incurring only a modest increase in GPU memory consumption. We further demonstrate the applications of the proposed method for adjustable defocus rendering and refocusing of the 3D scene from input images degraded by defocus blur.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00860v2">Segment Any 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-05-27
      | 💬 Work in progress. Project page: https://jumpat.github.io/SAGA
    </div>
    <details class="paper-abstract">
      This paper presents SAGA (Segment Any 3D GAussians), a highly efficient 3D promptable segmentation method based on 3D Gaussian Splatting (3D-GS). Given 2D visual prompts as input, SAGA can segment the corresponding 3D target represented by 3D Gaussians within 4 ms. This is achieved by attaching an scale-gated affinity feature to each 3D Gaussian to endow it a new property towards multi-granularity segmentation. Specifically, a scale-aware contrastive training strategy is proposed for the scale-gated affinity feature learning. It 1) distills the segmentation capability of the Segment Anything Model (SAM) from 2D masks into the affinity features and 2) employs a soft scale gate mechanism to deal with multi-granularity ambiguity in 3D segmentation through adjusting the magnitude of each feature channel according to a specified 3D physical scale. Evaluations demonstrate that SAGA achieves real-time multi-granularity segmentation with quality comparable to state-of-the-art methods. As one of the first methods addressing promptable segmentation in 3D-GS, the simplicity and effectiveness of SAGA pave the way for future advancements in this field. Our code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16645v1">Diffusion4D: Fast Spatial-temporal Consistent 4D Generation via Video Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2024-05-26
      | 💬 Project page: https://vita-group.github.io/Diffusion4D
    </div>
    <details class="paper-abstract">
      The availability of large-scale multimodal datasets and advancements in diffusion models have significantly accelerated progress in 4D content generation. Most prior approaches rely on multiple image or video diffusion models, utilizing score distillation sampling for optimization or generating pseudo novel views for direct supervision. However, these methods are hindered by slow optimization speeds and multi-view inconsistency issues. Spatial and temporal consistency in 4D geometry has been extensively explored respectively in 3D-aware diffusion models and traditional monocular video diffusion models. Building on this foundation, we propose a strategy to migrate the temporal consistency in video diffusion models to the spatial-temporal consistency required for 4D generation. Specifically, we present a novel framework, \textbf{Diffusion4D}, for efficient and scalable 4D content generation. Leveraging a meticulously curated dynamic 3D dataset, we develop a 4D-aware video diffusion model capable of synthesizing orbital views of dynamic 3D assets. To control the dynamic strength of these assets, we introduce a 3D-to-4D motion magnitude metric as guidance. Additionally, we propose a novel motion magnitude reconstruction loss and 3D-aware classifier-free guidance to refine the learning and generation of motion dynamics. After obtaining orbital views of the 4D asset, we perform explicit 4D construction with Gaussian splatting in a coarse-to-fine manner. The synthesized multi-view consistent 4D image set enables us to swiftly generate high-fidelity and diverse 4D assets within just several minutes. Extensive experiments demonstrate that our method surpasses prior state-of-the-art techniques in terms of generation efficiency and 4D geometry consistency across various prompt modalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16544v1">Splat-SLAM: Globally Optimized RGB-only SLAM with 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-05-26
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has emerged as a powerful representation of geometry and appearance for RGB-only dense Simultaneous Localization and Mapping (SLAM), as it provides a compact dense map representation while enabling efficient and high-quality map rendering. However, existing methods show significantly worse reconstruction quality than competing methods using other 3D representations, e.g. neural points clouds, since they either do not employ global map and pose optimization or make use of monocular depth. In response, we propose the first RGB-only SLAM system with a dense 3D Gaussian map representation that utilizes all benefits of globally optimized tracking by adapting dynamically to keyframe pose and depth updates by actively deforming the 3D Gaussian map. Moreover, we find that refining the depth updates in inaccurate areas with a monocular depth estimator further improves the accuracy of the 3D reconstruction. Our experiments on the Replica, TUM-RGBD, and ScanNet datasets indicate the effectiveness of globally optimized 3D Gaussians, as the approach achieves superior or on par performance with existing RGB-only SLAM methods methods in tracking, mapping and rendering accuracy while yielding small map sizes and fast runtimes. The source code is available at https://github.com/eriksandstroem/Splat-SLAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.16096v4">Animatable and Relightable Gaussians for High-fidelity Human Avatar Modeling</a></div>
    <div class="paper-meta">
      📅 2024-05-25
      | 💬 An extended version of Animatable Gaussians, Projectpage: https://animatable-gaussians.github.io/relight
    </div>
    <details class="paper-abstract">
      Modeling animatable human avatars from RGB videos is a long-standing and challenging problem. Recent works usually adopt MLP-based neural radiance fields (NeRF) to represent 3D humans, but it remains difficult for pure MLPs to regress pose-dependent garment details. To this end, we introduce Animatable Gaussians, a new avatar representation that leverages powerful 2D CNNs and 3D Gaussian splatting to create high-fidelity avatars. To associate 3D Gaussians with the animatable avatar, we learn a parametric template from the input videos, and then parameterize the template on two front & back canonical Gaussian maps where each pixel represents a 3D Gaussian. The learned template is adaptive to the wearing garments for modeling looser clothes like dresses. Such template-guided 2D parameterization enables us to employ a powerful StyleGAN-based CNN to learn the pose-dependent Gaussian maps for modeling detailed dynamic appearances. Furthermore, we introduce a pose projection strategy for better generalization given novel poses. To tackle the realistic relighting of animatable avatars, we introduce physically-based rendering into the avatar representation for decomposing avatar materials and environment illumination. Overall, our method can create lifelike avatars with dynamic, realistic, generalized and relightable appearances. Experiments show that our method outperforms other state-of-the-art approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.18795v3">Gamba: Marry Gaussian Splatting with Mamba for single view 3D reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-05-24
      | 💬 project page: https://florinshen.github.io/gamba-project
    </div>
    <details class="paper-abstract">
      We tackle the challenge of efficiently reconstructing a 3D asset from a single image at millisecond speed. Existing methods for single-image 3D reconstruction are primarily based on Score Distillation Sampling (SDS) with Neural 3D representations. Despite promising results, these approaches encounter practical limitations due to lengthy optimizations and significant memory consumption. In this work, we introduce Gamba, an end-to-end 3D reconstruction model from a single-view image, emphasizing two main insights: (1) Efficient Backbone Design: introducing a Mamba-based GambaFormer network to model 3D Gaussian Splatting (3DGS) reconstruction as sequential prediction with linear scalability of token length, thereby accommodating a substantial number of Gaussians; (2) Robust Gaussian Constraints: deriving radial mask constraints from multi-view masks to eliminate the need for warmup supervision of 3D point clouds in training. We trained Gamba on Objaverse and assessed it against existing optimization-based and feed-forward 3D reconstruction approaches on the GSO Dataset, among which Gamba is the only end-to-end trained single-view reconstruction model with 3DGS. Experimental results demonstrate its competitive generation capabilities both qualitatively and quantitatively and highlight its remarkable speed: Gamba completes reconstruction within 0.05 seconds on a single NVIDIA A100 GPU, which is about $1,000\times$ faster than optimization-based methods. Please see our project page at https://florinshen.github.io/gamba-project.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10128v2">GES: Generalized Exponential Splatting for Efficient Radiance Field Rendering</a></div>
    <div class="paper-meta">
      📅 2024-05-24
      | 💬 CVPR 2024 paper. project website https://abdullahamdi.com/ges
    </div>
    <details class="paper-abstract">
      Advancements in 3D Gaussian Splatting have significantly accelerated 3D reconstruction and generation. However, it may require a large number of Gaussians, which creates a substantial memory footprint. This paper introduces GES (Generalized Exponential Splatting), a novel representation that employs Generalized Exponential Function (GEF) to model 3D scenes, requiring far fewer particles to represent a scene and thus significantly outperforming Gaussian Splatting methods in efficiency with a plug-and-play replacement ability for Gaussian-based utilities. GES is validated theoretically and empirically in both principled 1D setup and realistic 3D scenes. It is shown to represent signals with sharp edges more accurately, which are typically challenging for Gaussians due to their inherent low-pass characteristics. Our empirical analysis demonstrates that GEF outperforms Gaussians in fitting natural-occurring signals (e.g. squares, triangles, and parabolic signals), thereby reducing the need for extensive splitting operations that increase the memory footprint of Gaussian Splatting. With the aid of a frequency-modulated loss, GES achieves competitive performance in novel-view synthesis benchmarks while requiring less than half the memory storage of Gaussian Splatting and increasing the rendering speed by up to 39%. The code is available on the project website https://abdullahamdi.com/ges .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.10142v2">GS-Planner: A Gaussian-Splatting-based Planning Framework for Active High-Fidelity Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-05-24
    </div>
    <details class="paper-abstract">
      Active reconstruction technique enables robots to autonomously collect scene data for full coverage, relieving users from tedious and time-consuming data capturing process. However, designed based on unsuitable scene representations, existing methods show unrealistic reconstruction results or the inability of online quality evaluation. Due to the recent advancements in explicit radiance field technology, online active high-fidelity reconstruction has become achievable. In this paper, we propose GS-Planner, a planning framework for active high-fidelity reconstruction using 3D Gaussian Splatting. With improvement on 3DGS to recognize unobserved regions, we evaluate the reconstruction quality and completeness of 3DGS map online to guide the robot. Then we design a sampling-based active reconstruction strategy to explore the unobserved areas and improve the reconstruction geometric and textural quality. To establish a complete robot active reconstruction system, we choose quadrotor as the robotic platform for its high agility. Then we devise a safety constraint with 3DGS to generate executable trajectories for quadrotor navigation in the 3DGS map. To validate the effectiveness of our method, we conduct extensive experiments and ablation studies in highly realistic simulation scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.12547v3">Evaluating Alternatives to SFM Point Cloud Initialization for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-05-23
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has recently been embraced as a versatile and effective method for scene reconstruction and novel view synthesis, owing to its high-quality results and compatibility with hardware rasterization. Despite its advantages, Gaussian Splatting's reliance on high-quality point cloud initialization by Structure-from-Motion (SFM) algorithms is a significant limitation to be overcome. To this end, we investigate various initialization strategies for Gaussian Splatting and delve into how volumetric reconstructions from Neural Radiance Fields (NeRF) can be utilized to bypass the dependency on SFM data. Our findings demonstrate that random initialization can perform much better if carefully designed and that by employing a combination of improved initialization strategies and structure distillation from low-cost NeRF models, it is possible to achieve equivalent results, or at times even superior, to those obtained from SFM initialization. Source code is available at https://theialab.github.io/nerf-3dgs .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14866v1">Tele-Aloha: A Low-budget and High-authenticity Telepresence System Using Sparse RGB Cameras</a></div>
    <div class="paper-meta">
      📅 2024-05-23
      | 💬 Paper accepted by SIGGRAPH 2024. Project page: http://118.178.32.38/c/Tele-Aloha/
    </div>
    <details class="paper-abstract">
      In this paper, we present a low-budget and high-authenticity bidirectional telepresence system, Tele-Aloha, targeting peer-to-peer communication scenarios. Compared to previous systems, Tele-Aloha utilizes only four sparse RGB cameras, one consumer-grade GPU, and one autostereoscopic screen to achieve high-resolution (2048x2048), real-time (30 fps), low-latency (less than 150ms) and robust distant communication. As the core of Tele-Aloha, we propose an efficient novel view synthesis algorithm for upper-body. Firstly, we design a cascaded disparity estimator for obtaining a robust geometry cue. Additionally a neural rasterizer via Gaussian Splatting is introduced to project latent features onto target view and to decode them into a reduced resolution. Further, given the high-quality captured data, we leverage weighted blending mechanism to refine the decoded image into the final resolution of 2K. Exploiting world-leading autostereoscopic display and low-latency iris tracking, users are able to experience a strong three-dimensional sense even without any wearable head-mounted display device. Altogether, our telepresence system demonstrates the sense of co-presence in real-life experiments, inspiring the next generation of communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.07472v2">GaussianVTON: 3D Human Virtual Try-ON via Multi-Stage Gaussian Splatting Editing with Image Prompting</a></div>
    <div class="paper-meta">
      📅 2024-05-23
      | 💬 On-going work
    </div>
    <details class="paper-abstract">
      The increasing prominence of e-commerce has underscored the importance of Virtual Try-On (VTON). However, previous studies predominantly focus on the 2D realm and rely heavily on extensive data for training. Research on 3D VTON primarily centers on garment-body shape compatibility, a topic extensively covered in 2D VTON. Thanks to advances in 3D scene editing, a 2D diffusion model has now been adapted for 3D editing via multi-viewpoint editing. In this work, we propose GaussianVTON, an innovative 3D VTON pipeline integrating Gaussian Splatting (GS) editing with 2D VTON. To facilitate a seamless transition from 2D to 3D VTON, we propose, for the first time, the use of only images as editing prompts for 3D editing. To further address issues, e.g., face blurring, garment inaccuracy, and degraded viewpoint quality during editing, we devise a three-stage refinement strategy to gradually mitigate potential issues. Furthermore, we introduce a new editing strategy termed Edit Recall Reconstruction (ERR) to tackle the limitations of previous editing strategies in leading to complex geometric changes. Our comprehensive experiments demonstrate the superiority of GaussianVTON, offering a novel perspective on 3D VTON while also establishing a novel starting point for image-prompting 3D scene editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13694v1">Gaussian Time Machine: A Real-Time Rendering Methodology for Time-Variant Appearances</a></div>
    <div class="paper-meta">
      📅 2024-05-22
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in neural rendering techniques have significantly enhanced the fidelity of 3D reconstruction. Notably, the emergence of 3D Gaussian Splatting (3DGS) has marked a significant milestone by adopting a discrete scene representation, facilitating efficient training and real-time rendering. Several studies have successfully extended the real-time rendering capability of 3DGS to dynamic scenes. However, a challenge arises when training images are captured under vastly differing weather and lighting conditions. This scenario poses a challenge for 3DGS and its variants in achieving accurate reconstructions. Although NeRF-based methods (NeRF-W, CLNeRF) have shown promise in handling such challenging conditions, their computational demands hinder real-time rendering capabilities. In this paper, we present Gaussian Time Machine (GTM) which models the time-dependent attributes of Gaussian primitives with discrete time embedding vectors decoded by a lightweight Multi-Layer-Perceptron(MLP). By adjusting the opacity of Gaussian primitives, we can reconstruct visibility changes of objects. We further propose a decomposed color model for improved geometric consistency. GTM achieved state-of-the-art rendering fidelity on 3 datasets and is 100 times faster than NeRF-based counterparts in rendering. Moreover, GTM successfully disentangles the appearance changes and renders smooth appearance interpolation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12069v2">Gaussian Head & Shoulders: High Fidelity Neural Upper Body Avatars with Anchor Gaussian Guided Texture Warping</a></div>
    <div class="paper-meta">
      📅 2024-05-21
      | 💬 Project Page: https://gaussian-head-shoulders.netlify.app/
    </div>
    <details class="paper-abstract">
      By equipping the most recent 3D Gaussian Splatting representation with head 3D morphable models (3DMM), existing methods manage to create head avatars with high fidelity. However, most existing methods only reconstruct a head without the body, substantially limiting their application scenarios. We found that naively applying Gaussians to model the clothed chest and shoulders tends to result in blurry reconstruction and noisy floaters under novel poses. This is because of the fundamental limitation of Gaussians and point clouds -- each Gaussian or point can only have a single directional radiance without spatial variance, therefore an unnecessarily large number of them is required to represent complicated spatially varying texture, even for simple geometry. In contrast, we propose to model the body part with a neural texture that consists of coarse and pose-dependent fine colors. To properly render the body texture for each view and pose without accurate geometry nor UV mapping, we optimize another sparse set of Gaussians as anchors that constrain the neural warping field that maps image plane coordinates to the texture space. We demonstrate that Gaussian Head & Shoulders can fit the high-frequency details on the clothed upper body with high fidelity and potentially improve the accuracy and fidelity of the head region. We evaluate our method with casual phone-captured and internet videos and show our method archives superior reconstruction quality and robustness in both self and cross reenactment tasks. To fully utilize the efficient rendering speed of Gaussian splatting, we additionally propose an accelerated inference method of our trained model without Multi-Layer Perceptron (MLP) queries and reach a stable rendering speed of around 130 FPS for any subjects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12663v1">LAGA: Layered 3D Avatar Generation and Customization via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-05-21
    </div>
    <details class="paper-abstract">
      Creating and customizing a 3D clothed avatar from textual descriptions is a critical and challenging task. Traditional methods often treat the human body and clothing as inseparable, limiting users' ability to freely mix and match garments. In response to this limitation, we present LAyered Gaussian Avatar (LAGA), a carefully designed framework enabling the creation of high-fidelity decomposable avatars with diverse garments. By decoupling garments from avatar, our framework empowers users to conviniently edit avatars at the garment level. Our approach begins by modeling the avatar using a set of Gaussian points organized in a layered structure, where each layer corresponds to a specific garment or the human body itself. To generate high-quality garments for each layer, we introduce a coarse-to-fine strategy for diverse garment generation and a novel dual-SDS loss function to maintain coherence between the generated garments and avatar components, including the human body and other garments. Moreover, we introduce three regularization losses to guide the movement of Gaussians for garment transfer, allowing garments to be freely transferred to various avatars. Extensive experimentation demonstrates that our approach surpasses existing methods in the generation of 3D clothed humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.05154v2">GSEdit: Efficient Text-Guided Editing of 3D Objects via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-05-21
      | 💬 8 pages, 8 figures
    </div>
    <details class="paper-abstract">
      We present GSEdit, a pipeline for text-guided 3D object editing based on Gaussian Splatting models. Our method enables the editing of the style and appearance of 3D objects without altering their main details, all in a matter of minutes on consumer hardware. We tackle the problem by leveraging Gaussian splatting to represent 3D scenes, and we optimize the model while progressively varying the image supervision by means of a pretrained image-based diffusion model. The input object may be given as a 3D triangular mesh, or directly provided as Gaussians from a generative model such as DreamGaussian. GSEdit ensures consistency across different viewpoints, maintaining the integrity of the original object's information. Compared to previously proposed methods relying on NeRF-like MLP models, GSEdit stands out for its efficiency, making 3D editing tasks much faster. Our editing process is refined via the application of the SDS loss, ensuring that our edits are both precise and accurate. Our comprehensive evaluation demonstrates that GSEdit effectively alters object shape and appearance following the given textual instructions while preserving their coherence and detail.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12420v1">GarmentDreamer: 3DGS Guided Garment Synthesis with Diverse Geometry and Texture Details</a></div>
    <div class="paper-meta">
      📅 2024-05-20
    </div>
    <details class="paper-abstract">
      Traditional 3D garment creation is labor-intensive, involving sketching, modeling, UV mapping, and texturing, which are time-consuming and costly. Recent advances in diffusion-based generative models have enabled new possibilities for 3D garment generation from text prompts, images, and videos. However, existing methods either suffer from inconsistencies among multi-view images or require additional processes to separate cloth from the underlying human model. In this paper, we propose GarmentDreamer, a novel method that leverages 3D Gaussian Splatting (GS) as guidance to generate wearable, simulation-ready 3D garment meshes from text prompts. In contrast to using multi-view images directly predicted by generative models as guidance, our 3DGS guidance ensures consistent optimization in both garment deformation and texture synthesis. Our method introduces a novel garment augmentation module, guided by normal and RGBA information, and employs implicit Neural Texture Fields (NeTF) combined with Score Distillation Sampling (SDS) to generate diverse geometric and texture details. We validate the effectiveness of our approach through comprehensive qualitative and quantitative experiments, showcasing the superior performance of GarmentDreamer over state-of-the-art alternatives. Our project page is available at: https://xuan-li.github.io/GarmentDreamerDemo/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11921v1">MirrorGaussian: Reflecting 3D Gaussians for Reconstructing Mirror Reflections</a></div>
    <div class="paper-meta">
      📅 2024-05-20
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting showcases notable advancements in photo-realistic and real-time novel view synthesis. However, it faces challenges in modeling mirror reflections, which exhibit substantial appearance variations from different viewpoints. To tackle this problem, we present MirrorGaussian, the first method for mirror scene reconstruction with real-time rendering based on 3D Gaussian Splatting. The key insight is grounded on the mirror symmetry between the real-world space and the virtual mirror space. We introduce an intuitive dual-rendering strategy that enables differentiable rasterization of both the real-world 3D Gaussians and the mirrored counterpart obtained by reflecting the former about the mirror plane. All 3D Gaussians are jointly optimized with the mirror plane in an end-to-end framework. MirrorGaussian achieves high-quality and real-time rendering in scenes with mirrors, empowering scene editing like adding new mirrors and objects. Comprehensive experiments on multiple datasets demonstrate that our approach significantly outperforms existing methods, achieving state-of-the-art results. Project page: https://mirror-gaussian.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11252v1">Dreamer XL: Towards High-Resolution Text-to-3D Generation via Trajectory Score Matching</a></div>
    <div class="paper-meta">
      📅 2024-05-18
    </div>
    <details class="paper-abstract">
      In this work, we propose a novel Trajectory Score Matching (TSM) method that aims to solve the pseudo ground truth inconsistency problem caused by the accumulated error in Interval Score Matching (ISM) when using the Denoising Diffusion Implicit Models (DDIM) inversion process. Unlike ISM which adopts the inversion process of DDIM to calculate on a single path, our TSM method leverages the inversion process of DDIM to generate two paths from the same starting point for calculation. Since both paths start from the same starting point, TSM can reduce the accumulated error compared to ISM, thus alleviating the problem of pseudo ground truth inconsistency. TSM enhances the stability and consistency of the model's generated paths during the distillation process. We demonstrate this experimentally and further show that ISM is a special case of TSM. Furthermore, to optimize the current multi-stage optimization process from high-resolution text to 3D generation, we adopt Stable Diffusion XL for guidance. In response to the issues of abnormal replication and splitting caused by unstable gradients during the 3D Gaussian splatting process when using Stable Diffusion XL, we propose a pixel-by-pixel gradient clipping method. Extensive experiments show that our model significantly surpasses the state-of-the-art models in terms of visual quality and performance. Code: \url{https://github.com/xingy038/Dreamer-XL}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.10508v1">ART3D: 3D Gaussian Splatting for Text-Guided Artistic Scenes Generation</a></div>
    <div class="paper-meta">
      📅 2024-05-17
      | 💬 Accepted at CVPR 2024 Workshop on AI3DG
    </div>
    <details class="paper-abstract">
      In this paper, we explore the existing challenges in 3D artistic scene generation by introducing ART3D, a novel framework that combines diffusion models and 3D Gaussian splatting techniques. Our method effectively bridges the gap between artistic and realistic images through an innovative image semantic transfer algorithm. By leveraging depth information and an initial artistic image, we generate a point cloud map, addressing domain differences. Additionally, we propose a depth consistency module to enhance 3D scene consistency. Finally, the 3D scene serves as initial points for optimizing Gaussian splats. Experimental results demonstrate ART3D's superior performance in both content and structural consistency metrics when compared to existing methods. ART3D significantly advances the field of AI in art creation by providing an innovative solution for generating high-quality 3D artistic scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.12365v2">GaussianFlow: Splatting Gaussian Dynamics for 4D Content Creation</a></div>
    <div class="paper-meta">
      📅 2024-05-13
    </div>
    <details class="paper-abstract">
      Creating 4D fields of Gaussian Splatting from images or videos is a challenging task due to its under-constrained nature. While the optimization can draw photometric reference from the input videos or be regulated by generative models, directly supervising Gaussian motions remains underexplored. In this paper, we introduce a novel concept, Gaussian flow, which connects the dynamics of 3D Gaussians and pixel velocities between consecutive frames. The Gaussian flow can be efficiently obtained by splatting Gaussian dynamics into the image space. This differentiable process enables direct dynamic supervision from optical flow. Our method significantly benefits 4D dynamic content generation and 4D novel view synthesis with Gaussian Splatting, especially for contents with rich motions that are hard to be handled by existing methods. The common color drifting issue that happens in 4D generation is also resolved with improved Guassian dynamics. Superior visual quality on extensive experiments demonstrates our method's effectiveness. Quantitative and qualitative evaluations show that our method achieves state-of-the-art results on both tasks of 4D generation and 4D novel view synthesis. Project page: https://zerg-overmind.github.io/GaussianFlow.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.08529v3">GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2024-05-13
      | 💬 CVPR 2024, Project page: https://taoranyi.com/gaussiandreamer/
    </div>
    <details class="paper-abstract">
      In recent times, the generation of 3D assets from text prompts has shown impressive results. Both 2D and 3D diffusion models can help generate decent 3D objects based on prompts. 3D diffusion models have good 3D consistency, but their quality and generalization are limited as trainable 3D data is expensive and hard to obtain. 2D diffusion models enjoy strong abilities of generalization and fine generation, but 3D consistency is hard to guarantee. This paper attempts to bridge the power from the two types of diffusion models via the recent explicit and efficient 3D Gaussian splatting representation. A fast 3D object generation framework, named as GaussianDreamer, is proposed, where the 3D diffusion model provides priors for initialization and the 2D diffusion model enriches the geometry and appearance. Operations of noisy point growing and color perturbation are introduced to enhance the initialized Gaussians. Our GaussianDreamer can generate a high-quality 3D instance or 3D avatar within 15 minutes on one GPU, much faster than previous methods, while the generated instances can be directly rendered in real time. Demos and code are available at https://taoranyi.com/gaussiandreamer/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00206v2">SparseGS: Real-Time 360° Sparse View Synthesis using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-05-13
      | 💬 This is a revised version which includes multiple new components. Project page: https://github.com/ForMyCat/SparseGS
    </div>
    <details class="paper-abstract">
      The problem of novel view synthesis has grown significantly in popularity recently with the introduction of Neural Radiance Fields (NeRFs) and other implicit scene representation methods. A recent advance, 3D Gaussian Splatting (3DGS), leverages an explicit representation to achieve real-time rendering with high-quality results. However, 3DGS still requires an abundance of training views to generate a coherent scene representation. In few shot settings, similar to NeRF, 3DGS tends to overfit to training views, causing background collapse and excessive floaters, especially as the number of training views are reduced. We propose a method to enable training coherent 3DGS-based radiance fields of 360-degree scenes from sparse training views. We integrate depth priors with generative and explicit constraints to reduce background collapse, remove floaters, and enhance consistency from unseen viewpoints. Experiments show that our method outperforms base 3DGS by 6.4% in LPIPS and by 12.2% in PSNR, and NeRF-based methods by at least 17.6% in LPIPS on the MipNeRF-360 dataset with substantially less training and inference cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18669v2">Bootstrap 3D Reconstructed Scenes from 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-05-12
    </div>
    <details class="paper-abstract">
      Recent developments in neural rendering techniques have greatly enhanced the rendering of photo-realistic 3D scenes across both academic and commercial fields. The latest method, known as 3D Gaussian Splatting (3D-GS), has set new benchmarks for rendering quality and speed. Nevertheless, the limitations of 3D-GS become pronounced in synthesizing new viewpoints, especially for views that greatly deviate from those seen during training. Additionally, issues such as dilation and aliasing arise when zooming in or out. These challenges can all be traced back to a single underlying issue: insufficient sampling. In our paper, we present a bootstrapping method that significantly addresses this problem. This approach employs a diffusion model to enhance the rendering of novel views using trained 3D-GS, thereby streamlining the training process. Our results indicate that bootstrapping effectively reduces artifacts, as well as clear enhancements on the evaluation metrics. Furthermore, we show that our method is versatile and can be easily integrated, allowing various 3D reconstruction projects to benefit from our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06547v1">OneTo3D: One Image to Re-editable Dynamic 3D Model and Video Generation</a></div>
    <div class="paper-meta">
      📅 2024-05-10
      | 💬 24 pages, 13 figures, 2 tables
    </div>
    <details class="paper-abstract">
      One image to editable dynamic 3D model and video generation is novel direction and change in the research area of single image to 3D representation or 3D reconstruction of image. Gaussian Splatting has demonstrated its advantages in implicit 3D reconstruction, compared with the original Neural Radiance Fields. As the rapid development of technologies and principles, people tried to used the Stable Diffusion models to generate targeted models with text instructions. However, using the normal implicit machine learning methods is hard to gain the precise motions and actions control, further more, it is difficult to generate a long content and semantic continuous 3D video. To address this issue, we propose the OneTo3D, a method and theory to used one single image to generate the editable 3D model and generate the targeted semantic continuous time-unlimited 3D video. We used a normal basic Gaussian Splatting model to generate the 3D model from a single image, which requires less volume of video memory and computer calculation ability. Subsequently, we designed an automatic generation and self-adaptive binding mechanism for the object armature. Combined with the re-editable motions and actions analyzing and controlling algorithm we proposed, we can achieve a better performance than the SOTA projects in the area of building the 3D model precise motions and actions control, and generating a stable semantic continuous time-unlimited 3D video with the input text instructions. Here we will analyze the detailed implementation methods and theories analyses. Relative comparisons and conclusions will be presented. The project code is open source.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06408v1">I3DGS: Improve 3D Gaussian Splatting from Multiple Dimensions</a></div>
    <div class="paper-meta">
      📅 2024-05-10
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting is a novel method for 3D view synthesis, which can gain an implicit neural learning rendering result than the traditional neural rendering technology but keep the more high-definition fast rendering speed. But it is still difficult to achieve a fast enough efficiency on 3D Gaussian Splatting for the practical applications. To Address this issue, we propose the I3DS, a synthetic model performance improvement evaluation solution and experiments test. From multiple and important levels or dimensions of the original 3D Gaussian Splatting, we made more than two thousand various kinds of experiments to test how the selected different items and components can make an impact on the training efficiency of the 3D Gaussian Splatting model. In this paper, we will share abundant and meaningful experiences and methods about how to improve the training, performance and the impacts caused by different items of the model. A special but normal Integer compression in base 95 and a floating-point compression in base 94 with ASCII encoding and decoding mechanism is presented. Many real and effective experiments and test results or phenomena will be recorded. After a series of reasonable fine-tuning, I3DS can gain excellent performance improvements than the previous one. The project code is available as open source.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05800v1">DragGaussian: Enabling Drag-style Manipulation on 3D Gaussian Representation</a></div>
    <div class="paper-meta">
      📅 2024-05-09
    </div>
    <details class="paper-abstract">
      User-friendly 3D object editing is a challenging task that has attracted significant attention recently. The limitations of direct 3D object editing without 2D prior knowledge have prompted increased attention towards utilizing 2D generative models for 3D editing. While existing methods like Instruct NeRF-to-NeRF offer a solution, they often lack user-friendliness, particularly due to semantic guided editing. In the realm of 3D representation, 3D Gaussian Splatting emerges as a promising approach for its efficiency and natural explicit property, facilitating precise editing tasks. Building upon these insights, we propose DragGaussian, a 3D object drag-editing framework based on 3D Gaussian Splatting, leveraging diffusion models for interactive image editing with open-vocabulary input. This framework enables users to perform drag-based editing on pre-trained 3D Gaussian object models, producing modified 2D images through multi-view consistent editing. Our contributions include the introduction of a new task, the development of DragGaussian for interactive point-based 3D editing, and comprehensive validation of its effectiveness through qualitative and quantitative experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05768v1">FastScene: Text-Driven Fast 3D Indoor Scene Generation via Panoramic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-05-09
      | 💬 Accepted by IJCAI-2024
    </div>
    <details class="paper-abstract">
      Text-driven 3D indoor scene generation holds broad applications, ranging from gaming and smart homes to AR/VR applications. Fast and high-fidelity scene generation is paramount for ensuring user-friendly experiences. However, existing methods are characterized by lengthy generation processes or necessitate the intricate manual specification of motion parameters, which introduces inconvenience for users. Furthermore, these methods often rely on narrow-field viewpoint iterative generations, compromising global consistency and overall scene quality. To address these issues, we propose FastScene, a framework for fast and higher-quality 3D scene generation, while maintaining the scene consistency. Specifically, given a text prompt, we generate a panorama and estimate its depth, since the panorama encompasses information about the entire scene and exhibits explicit geometric constraints. To obtain high-quality novel views, we introduce the Coarse View Synthesis (CVS) and Progressive Novel View Inpainting (PNVI) strategies, ensuring both scene consistency and view quality. Subsequently, we utilize Multi-View Projection (MVP) to form perspective views, and apply 3D Gaussian Splatting (3DGS) for scene reconstruction. Comprehensive experiments demonstrate FastScene surpasses other methods in both generation speed and quality with better scene consistency. Notably, guided only by a text prompt, FastScene can generate a 3D scene within a mere 15 minutes, which is at least one hour faster than state-of-the-art methods, making it a paradigm for user-friendly scene generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.19706v3">RTG-SLAM: Real-time 3D Reconstruction at Scale using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-05-09
      | 💬 To be published in ACM SIGGRAPH 2024
    </div>
    <details class="paper-abstract">
      We present Real-time Gaussian SLAM (RTG-SLAM), a real-time 3D reconstruction system with an RGBD camera for large-scale environments using Gaussian splatting. The system features a compact Gaussian representation and a highly efficient on-the-fly Gaussian optimization scheme. We force each Gaussian to be either opaque or nearly transparent, with the opaque ones fitting the surface and dominant colors, and transparent ones fitting residual colors. By rendering depth in a different way from color rendering, we let a single opaque Gaussian well fit a local surface region without the need of multiple overlapping Gaussians, hence largely reducing the memory and computation cost. For on-the-fly Gaussian optimization, we explicitly add Gaussians for three types of pixels per frame: newly observed, with large color errors, and with large depth errors. We also categorize all Gaussians into stable and unstable ones, where the stable Gaussians are expected to well fit previously observed RGBD images and otherwise unstable. We only optimize the unstable Gaussians and only render the pixels occupied by unstable Gaussians. In this way, both the number of Gaussians to be optimized and pixels to be rendered are largely reduced, and the optimization can be done in real time. We show real-time reconstructions of a variety of large scenes. Compared with the state-of-the-art NeRF-based RGBD SLAM, our system achieves comparable high-quality reconstruction but with around twice the speed and half the memory cost, and shows superior performance in the realism of novel view synthesis and camera tracking accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05446v1">GDGS: Gradient Domain Gaussian Splatting for Sparse Representation of Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2024-05-08
      | 💬 arXiv admin note: text overlap with arXiv:2404.09105
    </div>
    <details class="paper-abstract">
      The 3D Gaussian splatting methods are getting popular. However, they work directly on the signal, leading to a dense representation of the signal. Even with some techniques such as pruning or distillation, the results are still dense. In this paper, we propose to model the gradient of the original signal. The gradients are much sparser than the original signal. Therefore, the gradients use much less Gaussian splats, leading to the more efficient storage and thus higher computational performance during both training and rendering. Thanks to the sparsity, during the view synthesis, only a small mount of pixels are needed, leading to much higher computational performance ($100\sim 1000\times$ faster). And the 2D image can be recovered from the gradients via solving a Poisson equation with linear computation complexity. Several experiments are performed to confirm the sparseness of the gradients and the computation performance of the proposed method. The method can be applied various applications, such as human body modeling and indoor environment modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.03417v1">Gaussian Splatting: 3D Reconstruction and Novel View Synthesis, a Review</a></div>
    <div class="paper-meta">
      📅 2024-05-06
      | 💬 24 pages
    </div>
    <details class="paper-abstract">
      Image-based 3D reconstruction is a challenging task that involves inferring the 3D shape of an object or scene from a set of input images. Learning-based methods have gained attention for their ability to directly estimate 3D shapes. This review paper focuses on state-of-the-art techniques for 3D reconstruction, including the generation of novel, unseen views. An overview of recent developments in the Gaussian Splatting method is provided, covering input types, model structures, output representations, and training strategies. Unresolved challenges and future directions are also discussed. Given the rapid progress in this domain and the numerous opportunities for enhancing 3D reconstruction methods, a comprehensive examination of algorithms appears essential. Consequently, this study offers a thorough overview of the latest advancements in Gaussian Splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09412v2">DeferredGS: Decoupled and Editable Gaussian Splatting with Deferred Shading</a></div>
    <div class="paper-meta">
      📅 2024-05-06
    </div>
    <details class="paper-abstract">
      Reconstructing and editing 3D objects and scenes both play crucial roles in computer graphics and computer vision. Neural radiance fields (NeRFs) can achieve realistic reconstruction and editing results but suffer from inefficiency in rendering. Gaussian splatting significantly accelerates rendering by rasterizing Gaussian ellipsoids. However, Gaussian splatting utilizes a single Spherical Harmonic (SH) function to model both texture and lighting, limiting independent editing capabilities of these components. Recently, attempts have been made to decouple texture and lighting with the Gaussian splatting representation but may fail to produce plausible geometry and decomposition results on reflective scenes. Additionally, the forward shading technique they employ introduces noticeable blending artifacts during relighting, as the geometry attributes of Gaussians are optimized under the original illumination and may not be suitable for novel lighting conditions. To address these issues, we introduce DeferredGS, a method for decoupling and editing the Gaussian splatting representation using deferred shading. To achieve successful decoupling, we model the illumination with a learnable environment map and define additional attributes such as texture parameters and normal direction on Gaussians, where the normal is distilled from a jointly trained signed distance function. More importantly, we apply deferred shading, resulting in more realistic relighting effects compared to previous methods. Both qualitative and quantitative experiments demonstrate the superior performance of DeferredGS in novel view synthesis and editing tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.16663v2">VR-GS: A Physical Dynamics-Aware Interactive Gaussian Splatting System in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2024-05-04
    </div>
    <details class="paper-abstract">
      As consumer Virtual Reality (VR) and Mixed Reality (MR) technologies gain momentum, there's a growing focus on the development of engagements with 3D virtual content. Unfortunately, traditional techniques for content creation, editing, and interaction within these virtual spaces are fraught with difficulties. They tend to be not only engineering-intensive but also require extensive expertise, which adds to the frustration and inefficiency in virtual object manipulation. Our proposed VR-GS system represents a leap forward in human-centered 3D content interaction, offering a seamless and intuitive user experience. By developing a physical dynamics-aware interactive Gaussian Splatting in a Virtual Reality setting, and constructing a highly efficient two-level embedding strategy alongside deformable body simulations, VR-GS ensures real-time execution with highly realistic dynamic responses. The components of our Virtual Reality system are designed for high efficiency and effectiveness, starting from detailed scene reconstruction and object segmentation, advancing through multi-view image in-painting, and extending to interactive physics-based editing. The system also incorporates real-time deformation embedding and dynamic shadow casting, ensuring a comprehensive and engaging virtual experience.Our project page is available at: https://yingjiang96.github.io/VR-GS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.01970v2">FMGS: Foundation Model Embedded 3D Gaussian Splatting for Holistic 3D Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2024-05-03
      | 💬 Project page: https://xingxingzuo.github.io/fmgs
    </div>
    <details class="paper-abstract">
      Precisely perceiving the geometric and semantic properties of real-world 3D objects is crucial for the continued evolution of augmented reality and robotic applications. To this end, we present Foundation Model Embedded Gaussian Splatting (FMGS), which incorporates vision-language embeddings of foundation models into 3D Gaussian Splatting (GS). The key contribution of this work is an efficient method to reconstruct and represent 3D vision-language models. This is achieved by distilling feature maps generated from image-based foundation models into those rendered from our 3D model. To ensure high-quality rendering and fast training, we introduce a novel scene representation by integrating strengths from both GS and multi-resolution hash encodings (MHE). Our effective training procedure also introduces a pixel alignment loss that makes the rendered feature distance of the same semantic entities close, following the pixel-level semantic boundaries. Our results demonstrate remarkable multi-view semantic consistency, facilitating diverse downstream tasks, beating state-of-the-art methods by 10.2 percent on open-vocabulary language-based object detection, despite that we are 851X faster for inference. This research explores the intersection of vision, language, and 3D scene representation, paving the way for enhanced scene understanding in uncontrolled real-world environments. We plan to release the code on the project page.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.02005v1">HoloGS: Instant Depth-based 3D Gaussian Splatting with Microsoft HoloLens 2</a></div>
    <div class="paper-meta">
      📅 2024-05-03
      | 💬 8 pages, 9 figures, 2 tables. Will be published in the ISPRS The International Archives of Photogrammetry, Remote Sensing and Spatial Information Sciences
    </div>
    <details class="paper-abstract">
      In the fields of photogrammetry, computer vision and computer graphics, the task of neural 3D scene reconstruction has led to the exploration of various techniques. Among these, 3D Gaussian Splatting stands out for its explicit representation of scenes using 3D Gaussians, making it appealing for tasks like 3D point cloud extraction and surface reconstruction. Motivated by its potential, we address the domain of 3D scene reconstruction, aiming to leverage the capabilities of the Microsoft HoloLens 2 for instant 3D Gaussian Splatting. We present HoloGS, a novel workflow utilizing HoloLens sensor data, which bypasses the need for pre-processing steps like Structure from Motion by instantly accessing the required input data i.e. the images, camera poses and the point cloud from depth sensing. We provide comprehensive investigations, including the training process and the rendering quality, assessed through the Peak Signal-to-Noise Ratio, and the geometric 3D accuracy of the densified point cloud from Gaussian centers, measured by Chamfer Distance. We evaluate our approach on two self-captured scenes: An outdoor scene of a cultural heritage statue and an indoor scene of a fine-structured plant. Our results show that the HoloLens data, including RGB images, corresponding camera poses, and depth sensing based point clouds to initialize the Gaussians, are suitable as input for 3D Gaussian Splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.13299v2">Compact 3D Scene Representation via Self-Organizing Gaussian Grids</a></div>
    <div class="paper-meta">
      📅 2024-05-02
      | 💬 Added compression of spherical harmonics, updated compression method with improved results (all attributes compressed with JPEG XL now), added qualitative comparison of additional scenes, moved compression explanation and comparison to main paper, added comparison with "Making Gaussian Splats smaller"
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has recently emerged as a highly promising technique for modeling of static 3D scenes. In contrast to Neural Radiance Fields, it utilizes efficient rasterization allowing for very fast rendering at high-quality. However, the storage size is significantly higher, which hinders practical deployment, e.g. on resource constrained devices. In this paper, we introduce a compact scene representation organizing the parameters of 3D Gaussian Splatting (3DGS) into a 2D grid with local homogeneity, ensuring a drastic reduction in storage requirements without compromising visual quality during rendering. Central to our idea is the explicit exploitation of perceptual redundancies present in natural scenes. In essence, the inherent nature of a scene allows for numerous permutations of Gaussian parameters to equivalently represent it. To this end, we propose a novel highly parallel algorithm that regularly arranges the high-dimensional Gaussian parameters into a 2D grid while preserving their neighborhood structure. During training, we further enforce local smoothness between the sorted parameters in the grid. The uncompressed Gaussians use the same structure as 3DGS, ensuring a seamless integration with established renderers. Our method achieves a reduction factor of 17x to 42x in size for complex scenes with no increase in training time, marking a substantial leap forward in the domain of 3D scene distribution and consumption. Additional information can be found on our project page: https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.19398v2">3D Gaussian Blendshapes for Head Avatar Animation</a></div>
    <div class="paper-meta">
      📅 2024-05-02
      | 💬 ACM SIGGRAPH Conference Proceedings 2024
    </div>
    <details class="paper-abstract">
      We introduce 3D Gaussian blendshapes for modeling photorealistic head avatars. Taking a monocular video as input, we learn a base head model of neutral expression, along with a group of expression blendshapes, each of which corresponds to a basis expression in classical parametric face models. Both the neutral model and expression blendshapes are represented as 3D Gaussians, which contain a few properties to depict the avatar appearance. The avatar model of an arbitrary expression can be effectively generated by combining the neutral model and expression blendshapes through linear blending of Gaussians with the expression coefficients. High-fidelity head avatar animations can be synthesized in real time using Gaussian splatting. Compared to state-of-the-art methods, our Gaussian blendshape representation better captures high-frequency details exhibited in input video, and achieves superior rendering performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.00676v1">Spectrally Pruned Gaussian Fields with Neural Compensation</a></div>
    <div class="paper-meta">
      📅 2024-05-01
      | 💬 Code: https://github.com/RunyiYang/SUNDAE Project page: https://runyiyang.github.io/projects/SUNDAE/
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting, as a novel 3D representation, has garnered attention for its fast rendering speed and high rendering quality. However, this comes with high memory consumption, e.g., a well-trained Gaussian field may utilize three million Gaussian primitives and over 700 MB of memory. We credit this high memory footprint to the lack of consideration for the relationship between primitives. In this paper, we propose a memory-efficient Gaussian field named SUNDAE with spectral pruning and neural compensation. On one hand, we construct a graph on the set of Gaussian primitives to model their relationship and design a spectral down-sampling module to prune out primitives while preserving desired signals. On the other hand, to compensate for the quality loss of pruning Gaussians, we exploit a lightweight neural network head to mix splatted features, which effectively compensates for quality losses while capturing the relationship between primitives in its weights. We demonstrate the performance of SUNDAE with extensive results. For example, SUNDAE can achieve 26.80 PSNR at 145 FPS using 104 MB memory while the vanilla Gaussian splatting algorithm achieves 25.60 PSNR at 160 FPS using 523 MB memory, on the Mip-NeRF360 dataset. Codes are publicly available at https://runyiyang.github.io/projects/SUNDAE/.
    </details>
</div>
