# gaussian splatting - 2024_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

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
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11679v3">NEDS-SLAM: A Neural Explicit Dense Semantic SLAM Framework using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-01
      | 💬 accepted by RA-L, IEEE Robotics and Automation Letters
    </div>
    <details class="paper-abstract">
      We propose NEDS-SLAM, a dense semantic SLAM system based on 3D Gaussian representation, that enables robust 3D semantic mapping, accurate camera tracking, and high-quality rendering in real-time. In the system, we propose a Spatially Consistent Feature Fusion model to reduce the effect of erroneous estimates from pre-trained segmentation head on semantic reconstruction, achieving robust 3D semantic Gaussian mapping. Additionally, we employ a lightweight encoder-decoder to compress the high-dimensional semantic features into a compact 3D Gaussian representation, mitigating the burden of excessive memory consumption. Furthermore, we leverage the advantage of 3D Gaussian splatting, which enables efficient and differentiable novel view rendering, and propose a Virtual Camera View Pruning method to eliminate outlier gaussians, thereby effectively enhancing the quality of scene representations. Our NEDS-SLAM method demonstrates competitive performance over existing dense semantic SLAM methods in terms of mapping and tracking accuracy on Replica and ScanNet datasets, while also showing excellent capabilities in 3D dense semantic mapping.
    </details>
</div>
