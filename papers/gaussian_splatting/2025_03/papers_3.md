# gaussian splatting - 2025_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01199v1">LiteGS: A High-Performance Modular Framework for Gaussian Splatting Training</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Gaussian splatting has emerged as a powerful technique for reconstruction of 3D scenes in computer graphics and vision. However, conventional implementations often suffer from inefficiencies, limited flexibility, and high computational overhead, which constrain their adaptability to diverse applications. In this paper, we present LiteGS,a high-performance and modular framework that enhances both the efficiency and usability of Gaussian splatting. LiteGS achieves a 3.4x speedup over the original 3DGS implementation while reducing GPU memory usage by approximately 30%. Its modular design decomposes the splatting process into multiple highly optimized operators, and it provides dual API support via a script-based interface and a CUDA-based interface. The script-based interface, in combination with autograd, enables rapid prototyping and straightforward customization of new ideas, while the CUDA-based interface delivers optimal training speeds for performance-critical applications. LiteGS retains the core algorithm of 3DGS, ensuring compatibility. Comprehensive experiments on the Mip-NeRF 360 dataset demonstrate that LiteGS accelerates training without compromising accuracy, making it an ideal solution for both rapid prototyping and production environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01109v1">FGS-SLAM: Fourier-based Gaussian Splatting for Real-time SLAM with Sparse and Dense Map Fusion</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      3D gaussian splatting has advanced simultaneous localization and mapping (SLAM) technology by enabling real-time positioning and the construction of high-fidelity maps. However, the uncertainty in gaussian position and initialization parameters introduces challenges, often requiring extensive iterative convergence and resulting in redundant or insufficient gaussian representations. To address this, we introduce a novel adaptive densification method based on Fourier frequency domain analysis to establish gaussian priors for rapid convergence. Additionally, we propose constructing independent and unified sparse and dense maps, where a sparse map supports efficient tracking via Generalized Iterative Closest Point (GICP) and a dense map creates high-fidelity visual representations. This is the first SLAM system leveraging frequency domain analysis to achieve high-quality gaussian mapping in real-time. Experimental results demonstrate an average frame rate of 36 FPS on Replica and TUM RGB-D datasets, achieving competitive accuracy in both localization and mapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00868v1">Vid2Fluid: 3D Dynamic Fluid Assets from Single-View Videos with Generative Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      The generation of 3D content from single-view images has been extensively studied, but 3D dynamic scene generation with physical consistency from videos remains in its early stages. We propose a novel framework leveraging generative 3D Gaussian Splatting (3DGS) models to extract 3D dynamic fluid objects from single-view videos. The fluid geometry represented by 3DGS is initially generated from single-frame images, then denoised, densified, and aligned across frames. We estimate the fluid surface velocity using optical flow and compute the mainstream of the fluid to refine it. The 3D volumetric velocity field is then derived from the enclosed surface. The velocity field is then converted into a divergence-free, grid-based representation, enabling the optimization of simulation parameters through its differentiability across frames. This process results in simulation-ready fluid assets with physical dynamics closely matching those observed in the source video. Our approach is applicable to various fluid types, including gas, liquid, and viscous fluids, and allows users to edit the output geometry or extend movement durations seamlessly. Our automatic method for creating 3D dynamic fluid assets from single-view videos, easily obtainable from the internet, shows great potential for generating large-scale 3D fluid assets at a low cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00848v1">PSRGS:Progressive Spectral Residual of 3D Gaussian for High-Frequency Recovery</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D GS) achieves impressive results in novel view synthesis for small, single-object scenes through Gaussian ellipsoid initialization and adaptive density control. However, when applied to large-scale remote sensing scenes, 3D GS faces challenges: the point clouds generated by Structure-from-Motion (SfM) are often sparse, and the inherent smoothing behavior of 3D GS leads to over-reconstruction in high-frequency regions, where have detailed textures and color variations. This results in the generation of large, opaque Gaussian ellipsoids that cause gradient artifacts. Moreover, the simultaneous optimization of both geometry and texture may lead to densification of Gaussian ellipsoids at incorrect geometric locations, resulting in artifacts in other views. To address these issues, we propose PSRGS, a progressive optimization scheme based on spectral residual maps. Specifically, we create a spectral residual significance map to separate low-frequency and high-frequency regions. In the low-frequency region, we apply depth-aware and depth-smooth losses to initialize the scene geometry with low threshold. For the high-frequency region, we use gradient features with higher threshold to split and clone ellipsoids, refining the scene. The sampling rate is determined by feature responses and gradient loss. Finally, we introduce a pre-trained network that jointly computes perceptual loss from multiple views, ensuring accurate restoration of high-frequency details in both Gaussian ellipsoids geometry and color. We conduct experiments on multiple datasets to assess the effectiveness of our method, which demonstrates competitive rendering quality, especially in recovering texture details in high-frequency regions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00746v1">DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3D-GS) have shown remarkable success in representing 3D scenes and generating high-quality, novel views in real-time. However, 3D-GS and its variants assume that input images are captured based on pinhole imaging and are fully in focus. This assumption limits their applicability, as real-world images often feature shallow depth-of-field (DoF). In this paper, we introduce DoF-Gaussian, a controllable depth-of-field method for 3D-GS. We develop a lens-based imaging model based on geometric optics principles to control DoF effects. To ensure accurate scene geometry, we incorporate depth priors adjusted per scene, and we apply defocus-to-focus adaptation to minimize the gap in the circle of confusion. We also introduce a synthetic dataset to assess refocusing capabilities and the model's ability to learn precise lens parameters. Our framework is customizable and supports various interactive applications. Extensive experiments confirm the effectiveness of our method. Our project is available at https://dof-gaussian.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00726v1">Enhancing Monocular 3D Scene Completion with Diffusion Model</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 All authors had equal contribution
    </div>
    <details class="paper-abstract">
      3D scene reconstruction is essential for applications in virtual reality, robotics, and autonomous driving, enabling machines to understand and interact with complex environments. Traditional 3D Gaussian Splatting techniques rely on images captured from multiple viewpoints to achieve optimal performance, but this dependence limits their use in scenarios where only a single image is available. In this work, we introduce FlashDreamer, a novel approach for reconstructing a complete 3D scene from a single image, significantly reducing the need for multi-view inputs. Our approach leverages a pre-trained vision-language model to generate descriptive prompts for the scene, guiding a diffusion model to produce images from various perspectives, which are then fused to form a cohesive 3D reconstruction. Extensive experiments show that our method effectively and robustly expands single-image inputs into a comprehensive 3D scene, extending monocular 3D reconstruction capabilities without further training. Our code is available https://github.com/CharlieSong1999/FlashDreamer/tree/main.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18672v3">Drag Your Gaussian: Effective Drag-Based Editing with Score Distillation for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 Visit our project page at https://quyans.github.io/Drag-Your-Gaussian
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D scene editing have been propelled by the rapid development of generative models. Existing methods typically utilize generative models to perform text-guided editing on 3D representations, such as 3D Gaussian Splatting (3DGS). However, these methods are often limited to texture modifications and fail when addressing geometric changes, such as editing a character's head to turn around. Moreover, such methods lack accurate control over the spatial position of editing results, as language struggles to precisely describe the extent of edits. To overcome these limitations, we introduce DYG, an effective 3D drag-based editing method for 3D Gaussian Splatting. It enables users to conveniently specify the desired editing region and the desired dragging direction through the input of 3D masks and pairs of control points, thereby enabling precise control over the extent of editing. DYG integrates the strengths of the implicit triplane representation to establish the geometric scaffold of the editing results, effectively overcoming suboptimal editing outcomes caused by the sparsity of 3DGS in the desired editing regions. Additionally, we incorporate a drag-based Latent Diffusion Model into our method through the proposed Drag-SDS loss function, enabling flexible, multi-view consistent, and fine-grained editing. Extensive experiments demonstrate that DYG conducts effective drag-based editing guided by control point prompts, surpassing other baselines in terms of editing effect and quality, both qualitatively and quantitatively. Visit our project page at https://quyans.github.io/Drag-Your-Gaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05702v7">NGM-SLAM: Gaussian Splatting SLAM with Radiance Field Submap</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 9pages, 4 figures
    </div>
    <details class="paper-abstract">
      SLAM systems based on Gaussian Splatting have garnered attention due to their capabilities for rapid real-time rendering and high-fidelity mapping. However, current Gaussian Splatting SLAM systems usually struggle with large scene representation and lack effective loop closure detection. To address these issues, we introduce NGM-SLAM, the first 3DGS based SLAM system that utilizes neural radiance field submaps for progressive scene expression, effectively integrating the strengths of neural radiance fields and 3D Gaussian Splatting. We utilize neural radiance field submaps as supervision and achieve high-quality scene expression and online loop closure adjustments through Gaussian rendering of fused submaps. Our results on multiple real-world scenes and large-scale scene datasets demonstrate that our method can achieve accurate hole filling and high-quality scene expression, supporting monocular, stereo, and RGB-D inputs, and achieving state-of-the-art scene reconstruction and tracking performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02283v3">GP-GS: Gaussian Processes for Enhanced Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 14 pages,11 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has emerged as an efficient photorealistic novel view synthesis method. However, its reliance on sparse Structure-from-Motion (SfM) point clouds consistently compromises the scene reconstruction quality. To address these limitations, this paper proposes a novel 3D reconstruction framework Gaussian Processes Gaussian Splatting (GP-GS), where a multi-output Gaussian Process model is developed to achieve adaptive and uncertainty-guided densification of sparse SfM point clouds. Specifically, we propose a dynamic sampling and filtering pipeline that adaptively expands the SfM point clouds by leveraging GP-based predictions to infer new candidate points from the input 2D pixels and depth maps. The pipeline utilizes uncertainty estimates to guide the pruning of high-variance predictions, ensuring geometric consistency and enabling the generation of dense point clouds. The densified point clouds provide high-quality initial 3D Gaussians to enhance reconstruction performance. Extensive experiments conducted on synthetic and real-world datasets across various scales validate the effectiveness and practicality of the proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00881v1">Evolving High-Quality Rendering and Reconstruction in a Unified Framework with Contribution-Adaptive Regularization</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      Representing 3D scenes from multiview images is a core challenge in computer vision and graphics, which requires both precise rendering and accurate reconstruction. Recently, 3D Gaussian Splatting (3DGS) has garnered significant attention for its high-quality rendering and fast inference speed. Yet, due to the unstructured and irregular nature of Gaussian point clouds, ensuring accurate geometry reconstruction remains difficult. Existing methods primarily focus on geometry regularization, with common approaches including primitive-based and dual-model frameworks. However, the former suffers from inherent conflicts between rendering and reconstruction, while the latter is computationally and storage-intensive. To address these challenges, we propose CarGS, a unified model leveraging Contribution-adaptive regularization to achieve simultaneous, high-quality rendering and surface reconstruction. The essence of our framework is learning adaptive contribution for Gaussian primitives by squeezing the knowledge from geometry regularization into a compact MLP. Additionally, we introduce a geometry-guided densification strategy with clues from both normals and Signed Distance Fields (SDF) to improve the capability of capturing high-frequency details. Our design improves the mutual learning of the two tasks, meanwhile its unified structure does not require separate models as in dual-model based approaches, guaranteeing efficiency. Extensive experiments demonstrate the ability to achieve state-of-the-art (SOTA) results in both rendering fidelity and reconstruction accuracy while maintaining real-time speed and minimal storage size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11085v4">GS-CPR: Efficient Camera Pose Refinement via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 Accepted to International Conference on Learning Representations (ICLR) 2025. During the ICLR review process, we changed the name of our framework from GSLoc to GS-CPR (Camera Pose Refinement), according to reviewers' comments. The project page is available at https://xrim-lab.github.io/GS-CPR/
    </div>
    <details class="paper-abstract">
      We leverage 3D Gaussian Splatting (3DGS) as a scene representation and propose a novel test-time camera pose refinement (CPR) framework, GS-CPR. This framework enhances the localization accuracy of state-of-the-art absolute pose regression and scene coordinate regression methods. The 3DGS model renders high-quality synthetic images and depth maps to facilitate the establishment of 2D-3D correspondences. GS-CPR obviates the need for training feature extractors or descriptors by operating directly on RGB images, utilizing the 3D foundation model, MASt3R, for precise 2D matching. To improve the robustness of our model in challenging outdoor environments, we incorporate an exposure-adaptive module within the 3DGS framework. Consequently, GS-CPR enables efficient one-shot pose refinement given a single RGB query and a coarse initial pose estimation. Our proposed approach surpasses leading NeRF-based optimization methods in both accuracy and runtime across indoor and outdoor visual localization benchmarks, achieving new state-of-the-art accuracy on two indoor datasets. The project page is available at https://xrim-lab.github.io/GS-CPR/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15355v2">UniGaussian: Driving Scene Reconstruction from Multiple Camera Models via Unified Gaussian Representations</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 Technical report
    </div>
    <details class="paper-abstract">
      Urban scene reconstruction is crucial for real-world autonomous driving simulators. Although existing methods have achieved photorealistic reconstruction, they mostly focus on pinhole cameras and neglect fisheye cameras. In fact, how to effectively simulate fisheye cameras in driving scene remains an unsolved problem. In this work, we propose UniGaussian, a novel approach that learns a unified 3D Gaussian representation from multiple camera models for urban scene reconstruction in autonomous driving. Our contributions are two-fold. First, we propose a new differentiable rendering method that distorts 3D Gaussians using a series of affine transformations tailored to fisheye camera models. This addresses the compatibility issue of 3D Gaussian splatting with fisheye cameras, which is hindered by light ray distortion caused by lenses or mirrors. Besides, our method maintains real-time rendering while ensuring differentiability. Second, built on the differentiable rendering method, we design a new framework that learns a unified Gaussian representation from multiple camera models. By applying affine transformations to adapt different camera models and regularizing the shared Gaussians with supervision from different modalities, our framework learns a unified 3D Gaussian representation with input data from multiple sources and achieves holistic driving scene understanding. As a result, our approach models multiple sensors (pinhole and fisheye cameras) and modalities (depth, semantic, normal and LiDAR point clouds). Our experiments show that our method achieves superior rendering quality and fast rendering speed for driving scene simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00531v1">GaussianSeal: Rooting Adaptive Watermarks for 3D Gaussian Generation Model</a></div>
    <div class="paper-meta">
      📅 2025-03-01
    </div>
    <details class="paper-abstract">
      With the advancement of AIGC technologies, the modalities generated by models have expanded from images and videos to 3D objects, leading to an increasing number of works focused on 3D Gaussian Splatting (3DGS) generative models. Existing research on copyright protection for generative models has primarily concentrated on watermarking in image and text modalities, with little exploration into the copyright protection of 3D object generative models. In this paper, we propose the first bit watermarking framework for 3DGS generative models, named GaussianSeal, to enable the decoding of bits as copyright identifiers from the rendered outputs of generated 3DGS. By incorporating adaptive bit modulation modules into the generative model and embedding them into the network blocks in an adaptive way, we achieve high-precision bit decoding with minimal training overhead while maintaining the fidelity of the model's outputs. Experiments demonstrate that our method outperforms post-processing watermarking approaches for 3DGS objects, achieving superior performance of watermark decoding accuracy and preserving the quality of the generated results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00370v1">Scalable Real2Sim: Physics-Aware Asset Generation Via Robotic Pick-and-Place Setups</a></div>
    <div class="paper-meta">
      📅 2025-03-01
    </div>
    <details class="paper-abstract">
      Simulating object dynamics from real-world perception shows great promise for digital twins and robotic manipulation but often demands labor-intensive measurements and expertise. We present a fully automated Real2Sim pipeline that generates simulation-ready assets for real-world objects through robotic interaction. Using only a robot's joint torque sensors and an external camera, the pipeline identifies visual geometry, collision geometry, and physical properties such as inertial parameters. Our approach introduces a general method for extracting high-quality, object-centric meshes from photometric reconstruction techniques (e.g., NeRF, Gaussian Splatting) by employing alpha-transparent training while explicitly distinguishing foreground occlusions from background subtraction. We validate the full pipeline through extensive experiments, demonstrating its effectiveness across diverse objects. By eliminating the need for manual intervention or environment modifications, our pipeline can be integrated directly into existing pick-and-place setups, enabling scalable and efficient dataset creation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00357v1">CAT-3DGS: A Context-Adaptive Triplane Approach to Rate-Distortion-Optimized 3DGS Compression</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 Accepted for Publication in International Conference on Learning Representations (ICLR)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as a promising 3D representation. Much research has been focused on reducing its storage requirements and memory footprint. However, the needs to compress and transmit the 3DGS representation to the remote side are overlooked. This new application calls for rate-distortion-optimized 3DGS compression. How to quantize and entropy encode sparse Gaussian primitives in the 3D space remains largely unexplored. Few early attempts resort to the hyperprior framework from learned image compression. But, they fail to utilize fully the inter and intra correlation inherent in Gaussian primitives. Built on ScaffoldGS, this work, termed CAT-3DGS, introduces a context-adaptive triplane approach to their rate-distortion-optimized coding. It features multi-scale triplanes, oriented according to the principal axes of Gaussian primitives in the 3D space, to capture their inter correlation (i.e. spatial correlation) for spatial autoregressive coding in the projected 2D planes. With these triplanes serving as the hyperprior, we further perform channel-wise autoregressive coding to leverage the intra correlation within each individual Gaussian primitive. Our CAT-3DGS incorporates a view frequency-aware masking mechanism. It actively skips from coding those Gaussian primitives that potentially have little impact on the rendering quality. When trained end-to-end to strike a good rate-distortion trade-off, our CAT-3DGS achieves the state-of-the-art compression performance on the commonly used real-world datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00260v1">Seeing A 3D World in A Grain of Sand</a></div>
    <div class="paper-meta">
      📅 2025-03-01
    </div>
    <details class="paper-abstract">
      We present a snapshot imaging technique for recovering 3D surrounding views of miniature scenes. Due to their intricacy, miniature scenes with objects sized in millimeters are difficult to reconstruct, yet miniatures are common in life and their 3D digitalization is desirable. We design a catadioptric imaging system with a single camera and eight pairs of planar mirrors for snapshot 3D reconstruction from a dollhouse perspective. We place paired mirrors on nested pyramid surfaces for capturing surrounding multi-view images in a single shot. Our mirror design is customizable based on the size of the scene for optimized view coverage. We use the 3D Gaussian Splatting (3DGS) representation for scene reconstruction and novel view synthesis. We overcome the challenge posed by our sparse view input by integrating visual hull-derived depth constraint. Our method demonstrates state-of-the-art performance on a variety of synthetic and real miniature scenes.
    </details>
</div>
