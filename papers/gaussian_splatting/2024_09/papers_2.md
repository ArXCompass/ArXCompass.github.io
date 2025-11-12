# gaussian splatting - 2024_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2408.15695v2">G-Style: Stylized Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-06
    </div>
    <details class="paper-abstract">
      We introduce G-Style, a novel algorithm designed to transfer the style of an image onto a 3D scene represented using Gaussian Splatting. Gaussian Splatting is a powerful 3D representation for novel view synthesis, as -- compared to other approaches based on Neural Radiance Fields -- it provides fast scene renderings and user control over the scene. Recent pre-prints have demonstrated that the style of Gaussian Splatting scenes can be modified using an image exemplar. However, since the scene geometry remains fixed during the stylization process, current solutions fall short of producing satisfactory results. Our algorithm aims to address these limitations by following a three-step process: In a pre-processing step, we remove undesirable Gaussians with large projection areas or highly elongated shapes. Subsequently, we combine several losses carefully designed to preserve different scales of the style in the image, while maintaining as much as possible the integrity of the original scene content. During the stylization process and following the original design of Gaussian Splatting, we split Gaussians where additional detail is necessary within our scene by tracking the gradient of the stylized color. Our experiments demonstrate that G-Style generates high-quality stylizations within just a few minutes, outperforming existing methods both qualitatively and quantitatively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2407.13520v3">EaDeblur-GS: Event assisted 3D Deblur Reconstruction with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-06
    </div>
    <details class="paper-abstract">
      3D deblurring reconstruction techniques have recently seen significant advancements with the development of Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). Although these techniques can recover relatively clear 3D reconstructions from blurry image inputs, they still face limitations in handling severe blurring and complex camera motion. To address these issues, we propose Event-assisted 3D Deblur Reconstruction with Gaussian Splatting (EaDeblur-GS), which integrates event camera data to enhance the robustness of 3DGS against motion blur. By employing an Adaptive Deviation Estimator (ADE) network to estimate Gaussian center deviations and using novel loss functions, EaDeblur-GS achieves sharp 3D reconstructions in real-time, demonstrating performance comparable to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.13520v3">EaDeblur-GS: Event assisted 3D Deblur Reconstruction with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-05
    </div>
    <details class="paper-abstract">
      3D deblurring reconstruction techniques have recently seen significant advancements with the development of Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). Although these techniques can recover relatively clear 3D reconstructions from blurry image inputs, they still face limitations in handling severe blurring and complex camera motion. To address these issues, we propose Event-assisted 3D Deblur Reconstruction with Gaussian Splatting (EaDeblur-GS), which integrates event camera data to enhance the robustness of 3DGS against motion blur. By employing an Adaptive Deviation Estimator (ADE) network to estimate Gaussian center deviations and using novel loss functions, EaDeblur-GS achieves sharp 3D reconstructions in real-time, demonstrating performance comparable to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.15695v2">G-Style: Stylized Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-05
    </div>
    <details class="paper-abstract">
      We introduce G-Style, a novel algorithm designed to transfer the style of an image onto a 3D scene represented using Gaussian Splatting. Gaussian Splatting is a powerful 3D representation for novel view synthesis, as -- compared to other approaches based on Neural Radiance Fields -- it provides fast scene renderings and user control over the scene. Recent pre-prints have demonstrated that the style of Gaussian Splatting scenes can be modified using an image exemplar. However, since the scene geometry remains fixed during the stylization process, current solutions fall short of producing satisfactory results. Our algorithm aims to address these limitations by following a three-step process: In a pre-processing step, we remove undesirable Gaussians with large projection areas or highly elongated shapes. Subsequently, we combine several losses carefully designed to preserve different scales of the style in the image, while maintaining as much as possible the integrity of the original scene content. During the stylization process and following the original design of Gaussian Splatting, we split Gaussians where additional detail is necessary within our scene by tracking the gradient of the stylized color. Our experiments demonstrate that G-Style generates high-quality stylizations within just a few minutes, outperforming existing methods both qualitatively and quantitatively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03213v1">Optimizing 3D Gaussian Splatting for Sparse Viewpoint Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-09-05
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a promising approach for 3D scene representation, offering a reduction in computational overhead compared to Neural Radiance Fields (NeRF). However, 3DGS is susceptible to high-frequency artifacts and demonstrates suboptimal performance under sparse viewpoint conditions, thereby limiting its applicability in robotics and computer vision. To address these limitations, we introduce SVS-GS, a novel framework for Sparse Viewpoint Scene reconstruction that integrates a 3D Gaussian smoothing filter to suppress artifacts. Furthermore, our approach incorporates a Depth Gradient Profile Prior (DGPP) loss with a dynamic depth mask to sharpen edges and 2D diffusion with Score Distillation Sampling (SDS) loss to enhance geometric consistency in novel view synthesis. Experimental evaluations on the MipNeRF-360 and SeaThru-NeRF datasets demonstrate that SVS-GS markedly improves 3D reconstruction from sparse viewpoints, offering a robust and efficient solution for scene understanding in robotics and computer vision applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02382v1">GGS: Generalizable Gaussian Splatting for Lane Switching in Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2024-09-04
    </div>
    <details class="paper-abstract">
      We propose GGS, a Generalizable Gaussian Splatting method for Autonomous Driving which can achieve realistic rendering under large viewpoint changes. Previous generalizable 3D gaussian splatting methods are limited to rendering novel views that are very close to the original pair of images, which cannot handle large differences in viewpoint. Especially in autonomous driving scenarios, images are typically collected from a single lane. The limited training perspective makes rendering images of a different lane very challenging. To further improve the rendering capability of GGS under large viewpoint changes, we introduces a novel virtual lane generation module into GSS method to enables high-quality lane switching even without a multi-lane dataset. Besides, we design a diffusion loss to supervise the generation of virtual lane image to further address the problem of lack of data in the virtual lanes. Finally, we also propose a depth refinement module to optimize depth estimation in the GSS model. Extensive validation of our method, compared to existing approaches, demonstrates state-of-the-art performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02851v1">Human-VDM: Learning Single-Image 3D Human Gaussian Splatting from Video Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2024-09-04
      | 💬 14 Pages, 8 figures, Project page: https://human-vdm.github.io/Human-VDM/
    </div>
    <details class="paper-abstract">
      Generating lifelike 3D humans from a single RGB image remains a challenging task in computer vision, as it requires accurate modeling of geometry, high-quality texture, and plausible unseen parts. Existing methods typically use multi-view diffusion models for 3D generation, but they often face inconsistent view issues, which hinder high-quality 3D human generation. To address this, we propose Human-VDM, a novel method for generating 3D human from a single RGB image using Video Diffusion Models. Human-VDM provides temporally consistent views for 3D human generation using Gaussian Splatting. It consists of three modules: a view-consistent human video diffusion module, a video augmentation module, and a Gaussian Splatting module. First, a single image is fed into a human video diffusion module to generate a coherent human video. Next, the video augmentation module applies super-resolution and video interpolation to enhance the textures and geometric smoothness of the generated video. Finally, the 3D Human Gaussian Splatting module learns lifelike humans under the guidance of these high-resolution and view-consistent images. Experiments demonstrate that Human-VDM achieves high-quality 3D human from a single image, outperforming state-of-the-art methods in both generation quality and quantity. Project page: https://human-vdm.github.io/Human-VDM/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02581v1">Object Gaussian for Monocular 6D Pose Estimation from Sparse Views</a></div>
    <div class="paper-meta">
      📅 2024-09-04
    </div>
    <details class="paper-abstract">
      Monocular object pose estimation, as a pivotal task in computer vision and robotics, heavily depends on accurate 2D-3D correspondences, which often demand costly CAD models that may not be readily available. Object 3D reconstruction methods offer an alternative, among which recent advancements in 3D Gaussian Splatting (3DGS) afford a compelling potential. Yet its performance still suffers and tends to overfit with fewer input views. Embracing this challenge, we introduce SGPose, a novel framework for sparse view object pose estimation using Gaussian-based methods. Given as few as ten views, SGPose generates a geometric-aware representation by starting with a random cuboid initialization, eschewing reliance on Structure-from-Motion (SfM) pipeline-derived geometry as required by traditional 3DGS methods. SGPose removes the dependence on CAD models by regressing dense 2D-3D correspondences between images and the reconstructed model from sparse input and random initialization, while the geometric-consistent depth supervision and online synthetic view warping are key to the success. Experiments on typical benchmarks, especially on the Occlusion LM-O dataset, demonstrate that SGPose outperforms existing methods even under sparse view constraints, under-scoring its potential in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02104v1">DynOMo: Online Point Tracking by Dynamic Online Monocular Gaussian Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-09-03
    </div>
    <details class="paper-abstract">
      Reconstructing scenes and tracking motion are two sides of the same coin. Tracking points allow for geometric reconstruction [14], while geometric reconstruction of (dynamic) scenes allows for 3D tracking of points over time [24, 39]. The latter was recently also exploited for 2D point tracking to overcome occlusion ambiguities by lifting tracking directly into 3D [38]. However, above approaches either require offline processing or multi-view camera setups both unrealistic for real-world applications like robot navigation or mixed reality. We target the challenge of online 2D and 3D point tracking from unposed monocular camera input introducing Dynamic Online Monocular Reconstruction (DynOMo). We leverage 3D Gaussian splatting to reconstruct dynamic scenes in an online fashion. Our approach extends 3D Gaussians to capture new content and object motions while estimating camera movements from a single RGB frame. DynOMo stands out by enabling emergence of point trajectories through robust image feature reconstruction and a novel similarity-enhanced regularization term, without requiring any correspondence-level supervision. It sets the first baseline for online point tracking with monocular unposed cameras, achieving performance on par with existing methods. We aim to inspire the community to advance online point tracking and reconstruction, expanding the applicability to diverse real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01761v1">PRoGS: Progressive Rendering of Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2024-09-03
    </div>
    <details class="paper-abstract">
      Over the past year, 3D Gaussian Splatting (3DGS) has received significant attention for its ability to represent 3D scenes in a perceptually accurate manner. However, it can require a substantial amount of storage since each splat's individual data must be stored. While compression techniques offer a potential solution by reducing the memory footprint, they still necessitate retrieving the entire scene before any part of it can be rendered. In this work, we introduce a novel approach for progressively rendering such scenes, aiming to display visible content that closely approximates the final scene as early as possible without loading the entire scene into memory. This approach benefits both on-device rendering applications limited by memory constraints and streaming applications where minimal bandwidth usage is preferred. To achieve this, we approximate the contribution of each Gaussian to the final scene and construct an order of prioritization on their inclusion in the rendering process. Additionally, we demonstrate that our approach can be combined with existing compression methods to progressively render (and stream) 3DGS scenes, optimizing bandwidth usage by focusing on the most important splats within a scene. Overall, our work establishes a foundation for making remotely hosted 3DGS content more quickly accessible to end-users in over-the-top consumption scenarios, with our results showing significant improvements in quality across all metrics compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01581v1">GaussianPU: A Hybrid 2D-3D Upsampling Framework for Enhancing Color Point Clouds via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-03
      | 💬 7 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Dense colored point clouds enhance visual perception and are of significant value in various robotic applications. However, existing learning-based point cloud upsampling methods are constrained by computational resources and batch processing strategies, which often require subdividing point clouds into smaller patches, leading to distortions that degrade perceptual quality. To address this challenge, we propose a novel 2D-3D hybrid colored point cloud upsampling framework (GaussianPU) based on 3D Gaussian Splatting (3DGS) for robotic perception. This approach leverages 3DGS to bridge 3D point clouds with their 2D rendered images in robot vision systems. A dual scale rendered image restoration network transforms sparse point cloud renderings into dense representations, which are then input into 3DGS along with precise robot camera poses and interpolated sparse point clouds to reconstruct dense 3D point clouds. We have made a series of enhancements to the vanilla 3DGS, enabling precise control over the number of points and significantly boosting the quality of the upsampled point cloud for robotic scene understanding. Our framework supports processing entire point clouds on a single consumer-grade GPU, such as the NVIDIA GeForce RTX 3090, eliminating the need for segmentation and thus producing high-quality, dense colored point clouds with millions of points for robot navigation and manipulation tasks. Extensive experimental results on generating million-level point cloud data validate the effectiveness of our method, substantially improving the quality of colored point clouds and demonstrating significant potential for applications involving large-scale point clouds in autonomous robotics and human-robot interaction scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2408.16982v1">2DGH: 2D Gaussian-Hermite Splatting for High-quality Rendering and Better Geometry Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-09-02
    </div>
    <details class="paper-abstract">
      2D Gaussian Splatting has recently emerged as a significant method in 3D reconstruction, enabling novel view synthesis and geometry reconstruction simultaneously. While the well-known Gaussian kernel is broadly used, its lack of anisotropy and deformation ability leads to dim and vague edges at object silhouettes, limiting the reconstruction quality of current Gaussian splatting methods. To enhance the representation power, we draw inspiration from quantum physics and propose to use the Gaussian-Hermite kernel as the new primitive in Gaussian splatting. The new kernel takes a unified mathematical form and extends the Gaussian function, which serves as the zero-rank term in the updated formulation. Our experiments demonstrate the extraordinary performance of Gaussian-Hermite kernel in both geometry reconstruction and novel-view synthesis tasks. The proposed kernel outperforms traditional Gaussian Splatting kernels, showcasing its potential for high-quality 3D reconstruction and rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11679v3">NEDS-SLAM: A Neural Explicit Dense Semantic SLAM Framework using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-01
      | 💬 accepted by RA-L, IEEE Robotics and Automation Letters
    </div>
    <details class="paper-abstract">
      We propose NEDS-SLAM, a dense semantic SLAM system based on 3D Gaussian representation, that enables robust 3D semantic mapping, accurate camera tracking, and high-quality rendering in real-time. In the system, we propose a Spatially Consistent Feature Fusion model to reduce the effect of erroneous estimates from pre-trained segmentation head on semantic reconstruction, achieving robust 3D semantic Gaussian mapping. Additionally, we employ a lightweight encoder-decoder to compress the high-dimensional semantic features into a compact 3D Gaussian representation, mitigating the burden of excessive memory consumption. Furthermore, we leverage the advantage of 3D Gaussian splatting, which enables efficient and differentiable novel view rendering, and propose a Virtual Camera View Pruning method to eliminate outlier gaussians, thereby effectively enhancing the quality of scene representations. Our NEDS-SLAM method demonstrates competitive performance over existing dense semantic SLAM methods in terms of mapping and tracking accuracy on Replica and ScanNet datasets, while also showing excellent capabilities in 3D dense semantic mapping.
    </details>
</div>
