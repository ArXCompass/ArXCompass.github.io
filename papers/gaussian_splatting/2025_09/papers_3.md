# gaussian splatting - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.05740v2">Micro-splatting: Multistage Isotropy-informed Covariance Regularization Optimization for High-Fidelity 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 This work has been submitted to journal for potential publication
    </div>
    <details class="paper-abstract">
      High-fidelity 3D Gaussian Splatting methods excel at capturing fine textures but often overlook model compactness, resulting in massive splat counts, bloated memory, long training, and complex post-processing. We present Micro-Splatting: Two-Stage Adaptive Growth and Refinement, a unified, in-training pipeline that preserves visual detail while drastically reducing model complexity without any post-processing or auxiliary neural modules. In Stage I (Growth), we introduce a trace-based covariance regularization to maintain near-isotropic Gaussians, mitigating low-pass filtering in high-frequency regions and improving spherical-harmonic color fitting. We then apply gradient-guided adaptive densification that subdivides splats only in visually complex regions, leaving smooth areas sparse. In Stage II (Refinement), we prune low-impact splats using a simple opacity-scale importance score and merge redundant neighbors via lightweight spatial and feature thresholds, producing a lean yet detail-rich model. On four object-centric benchmarks, Micro-Splatting reduces splat count and model size by up to 60% and shortens training by 20%, while matching or surpassing state-of-the-art PSNR, SSIM, and LPIPS in real-time rendering. These results demonstrate that Micro-Splatting delivers both compactness and high fidelity in a single, efficient, end-to-end framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.01964v1">2D Gaussian Splatting with Semantic Alignment for Image Inpainting</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS), a recent technique for converting discrete points into continuous spatial representations, has shown promising results in 3D scene modeling and 2D image super-resolution. In this paper, we explore its untapped potential for image inpainting, which demands both locally coherent pixel synthesis and globally consistent semantic restoration. We propose the first image inpainting framework based on 2D Gaussian Splatting, which encodes incomplete images into a continuous field of 2D Gaussian splat coefficients and reconstructs the final image via a differentiable rasterization process. The continuous rendering paradigm of GS inherently promotes pixel-level coherence in the inpainted results. To improve efficiency and scalability, we introduce a patch-wise rasterization strategy that reduces memory overhead and accelerates inference. For global semantic consistency, we incorporate features from a pretrained DINO model. We observe that DINO's global features are naturally robust to small missing regions and can be effectively adapted to guide semantic alignment in large-mask scenarios, ensuring that the inpainted content remains contextually consistent with the surrounding scene. Extensive experiments on standard benchmarks demonstrate that our method achieves competitive performance in both quantitative metrics and perceptual quality, establishing a new direction for applying Gaussian Splatting to 2D image processing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09573v2">FreeSplatter: Pose-free Gaussian Splatting for Sparse-view 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Project page: https://bluestyle97.github.io/projects/freesplatter/
    </div>
    <details class="paper-abstract">
      Sparse-view reconstruction models typically require precise camera poses, yet obtaining these parameters from sparse-view images remains challenging. We introduce FreeSplatter, a scalable feed-forward framework that generates high-quality 3D Gaussians from uncalibrated sparse-view images while estimating camera parameters within seconds. Our approach employs a streamlined transformer architecture where self-attention blocks facilitate information exchange among multi-view image tokens, decoding them into pixel-aligned 3D Gaussian primitives within a unified reference frame. This representation enables both high-fidelity 3D modeling and efficient camera parameter estimation using off-the-shelf solvers. We develop two specialized variants--for object-centric and scene-level reconstruction--trained on comprehensive datasets. Remarkably, FreeSplatter outperforms several pose-dependent Large Reconstruction Models (LRMs) by a notable margin while achieving comparable or even better pose estimation accuracy compared to state-of-the-art pose-free reconstruction approach MASt3R in challenging benchmarks. Beyond technical benchmarks, FreeSplatter streamlines text/image-to-3D content creation pipelines, eliminating the complexity of camera pose management while delivering exceptional visual fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10462v2">BloomScene: Lightweight Structured 3D Gaussian Splatting for Crossmodal Scene Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Accepted by AAAI 2025. Code: https://github.com/SparklingH/BloomScene
    </div>
    <details class="paper-abstract">
      With the widespread use of virtual reality applications, 3D scene generation has become a new challenging research frontier. 3D scenes have highly complex structures and need to ensure that the output is dense, coherent, and contains all necessary structures. Many current 3D scene generation methods rely on pre-trained text-to-image diffusion models and monocular depth estimators. However, the generated scenes occupy large amounts of storage space and often lack effective regularisation methods, leading to geometric distortions. To this end, we propose BloomScene, a lightweight structured 3D Gaussian splatting for crossmodal scene generation, which creates diverse and high-quality 3D scenes from text or image inputs. Specifically, a crossmodal progressive scene generation framework is proposed to generate coherent scenes utilizing incremental point cloud reconstruction and 3D Gaussian splatting. Additionally, we propose a hierarchical depth prior-based regularization mechanism that utilizes multi-level constraints on depth accuracy and smoothness to enhance the realism and continuity of the generated scenes. Ultimately, we propose a structured context-guided compression mechanism that exploits structured hash grids to model the context of unorganized anchor attributes, which significantly eliminates structural redundancy and reduces storage overhead. Comprehensive experiments across multiple scenes demonstrate the significant potential and advantages of our framework compared with several baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01681v1">GaussianGAN: Real-Time Photorealistic controllable Human Avatars</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 IEEE conference series on Automatic Face and Gesture Recognition 2025
    </div>
    <details class="paper-abstract">
      Photorealistic and controllable human avatars have gained popularity in the research community thanks to rapid advances in neural rendering, providing fast and realistic synthesis tools. However, a limitation of current solutions is the presence of noticeable blurring. To solve this problem, we propose GaussianGAN, an animatable avatar approach developed for photorealistic rendering of people in real-time. We introduce a novel Gaussian splatting densification strategy to build Gaussian points from the surface of cylindrical structures around estimated skeletal limbs. Given the camera calibration, we render an accurate semantic segmentation with our novel view segmentation module. Finally, a UNet generator uses the rendered Gaussian splatting features and the segmentation maps to create photorealistic digital avatars. Our method runs in real-time with a rendering speed of 79 FPS. It outperforms previous methods regarding visual perception and quality, achieving a state-of-the-art results in terms of a pixel fidelity of 32.94db on the ZJU Mocap dataset and 33.39db on the Thuman4 dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01469v1">Im2Haircut: Single-view Strand-based Hair Reconstruction for Human Avatars</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 For more results please refer to the project page https://im2haircut.is.tue.mpg.de
    </div>
    <details class="paper-abstract">
      We present a novel approach for 3D hair reconstruction from single photographs based on a global hair prior combined with local optimization. Capturing strand-based hair geometry from single photographs is challenging due to the variety and geometric complexity of hairstyles and the lack of ground truth training data. Classical reconstruction methods like multi-view stereo only reconstruct the visible hair strands, missing the inner structure of hairstyles and hampering realistic hair simulation. To address this, existing methods leverage hairstyle priors trained on synthetic data. Such data, however, is limited in both quantity and quality since it requires manual work from skilled artists to model the 3D hairstyles and create near-photorealistic renderings. To address this, we propose a novel approach that uses both, real and synthetic data to learn an effective hairstyle prior. Specifically, we train a transformer-based prior model on synthetic data to obtain knowledge of the internal hairstyle geometry and introduce real data in the learning process to model the outer structure. This training scheme is able to model the visible hair strands depicted in an input image, while preserving the general 3D structure of hairstyles. We exploit this prior to create a Gaussian-splatting-based reconstruction method that creates hairstyles from one or more images. Qualitative and quantitative comparisons with existing reconstruction pipelines demonstrate the effectiveness and superior performance of our method for capturing detailed hair orientation, overall silhouette, and backside consistency. For additional results and code, please refer to https://im2haircut.is.tue.mpg.de.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2508.21542v1">Complete Gaussian Splats from a Single Image with Denoising Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Main paper: 11 pages; Supplementary materials: 7 pages
    </div>
    <details class="paper-abstract">
      Gaussian splatting typically requires dense observations of the scene and can fail to reconstruct occluded and unobserved areas. We propose a latent diffusion model to reconstruct a complete 3D scene with Gaussian splats, including the occluded parts, from only a single image during inference. Completing the unobserved surfaces of a scene is challenging due to the ambiguity of the plausible surfaces. Conventional methods use a regression-based formulation to predict a single "mode" for occluded and out-of-frustum surfaces, leading to blurriness, implausibility, and failure to capture multiple possible explanations. Thus, they often address this problem partially, focusing either on objects isolated from the background, reconstructing only visible surfaces, or failing to extrapolate far from the input views. In contrast, we propose a generative formulation to learn a distribution of 3D representations of Gaussian splats conditioned on a single input image. To address the lack of ground-truth training data, we propose a Variational AutoReconstructor to learn a latent space only from 2D images in a self-supervised manner, over which a diffusion model is trained. Our method generates faithful reconstructions and diverse samples with the ability to complete the occluded surfaces for high-quality 360-degree renderings.
    </details>
</div>
