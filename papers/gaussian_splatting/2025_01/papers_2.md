# gaussian splatting - 2025_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19370v2">BeSplat: Gaussian Splatting from a Single Blurry Image and Event Stream</a></div>
    <div class="paper-meta">
      📅 2025-01-05
      | 💬 Accepted for publication at EVGEN2025, WACV-25 Workshop
    </div>
    <details class="paper-abstract">
      Novel view synthesis has been greatly enhanced by the development of radiance field methods. The introduction of 3D Gaussian Splatting (3DGS) has effectively addressed key challenges, such as long training times and slow rendering speeds, typically associated with Neural Radiance Fields (NeRF), while maintaining high-quality reconstructions. In this work (BeSplat), we demonstrate the recovery of sharp radiance field (Gaussian splats) from a single motion-blurred image and its corresponding event stream. Our method jointly learns the scene representation via Gaussian Splatting and recovers the camera motion through Bezier SE(3) formulation effectively, minimizing discrepancies between synthesized and real-world measurements of both blurry image and corresponding event stream. We evaluate our approach on both synthetic and real datasets, showcasing its ability to render view-consistent, sharp images from the learned radiance field and the estimated camera trajectory. To the best of our knowledge, ours is the first work to address this highly challenging ill-posed problem in a Gaussian Splatting framework with the effective incorporation of temporal information captured using the event stream.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.19370v2">BeSplat: Gaussian Splatting from a Single Blurry Image and Event Stream</a></div>
    <div class="paper-meta">
      📅 2025-01-05
      | 💬 Accepted for publication at EVGEN2025, WACV-25 Workshop
    </div>
    <details class="paper-abstract">
      Novel view synthesis has been greatly enhanced by the development of radiance field methods. The introduction of 3D Gaussian Splatting (3DGS) has effectively addressed key challenges, such as long training times and slow rendering speeds, typically associated with Neural Radiance Fields (NeRF), while maintaining high-quality reconstructions. In this work (BeSplat), we demonstrate the recovery of sharp radiance field (Gaussian splats) from a single motion-blurred image and its corresponding event stream. Our method jointly learns the scene representation via Gaussian Splatting and recovers the camera motion through Bezier SE(3) formulation effectively, minimizing discrepancies between synthesized and real-world measurements of both blurry image and corresponding event stream. We evaluate our approach on both synthetic and real datasets, showcasing its ability to render view-consistent, sharp images from the learned radiance field and the estimated camera trajectory. To the best of our knowledge, ours is the first work to address this highly challenging ill-posed problem in a Gaussian Splatting framework with the effective incorporation of temporal information captured using the event stream.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2501.00342v1">SG-Splatting: Accelerating 3D Gaussian Splatting with Spherical Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-01-03
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting is emerging as a state-of-the-art technique in novel view synthesis, recognized for its impressive balance between visual quality, speed, and rendering efficiency. However, reliance on third-degree spherical harmonics for color representation introduces significant storage demands and computational overhead, resulting in a large memory footprint and slower rendering speed. We introduce SG-Splatting with Spherical Gaussians based color representation, a novel approach to enhance rendering speed and quality in novel view synthesis. Our method first represents view-dependent color using Spherical Gaussians, instead of three degree spherical harmonics, which largely reduces the number of parameters used for color representation, and significantly accelerates the rendering process. We then develop an efficient strategy for organizing multiple Spherical Gaussians, optimizing their arrangement to achieve a balanced and accurate scene representation. To further improve rendering quality, we propose a mixed representation that combines Spherical Gaussians with low-degree spherical harmonics, capturing both high- and low-frequency color information effectively. SG-Splatting also has plug-and-play capability, allowing it to be easily integrated into existing systems. This approach improves computational efficiency and overall visual fidelity, making it a practical solution for real-time applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00601v2">DreamDrive: Generative 4D Scene Modeling from Street View Images</a></div>
    <div class="paper-meta">
      📅 2025-01-03
      | 💬 Project page: https://pointscoder.github.io/DreamDrive/
    </div>
    <details class="paper-abstract">
      Synthesizing photo-realistic visual observations from an ego vehicle's driving trajectory is a critical step towards scalable training of self-driving models. Reconstruction-based methods create 3D scenes from driving logs and synthesize geometry-consistent driving videos through neural rendering, but their dependence on costly object annotations limits their ability to generalize to in-the-wild driving scenarios. On the other hand, generative models can synthesize action-conditioned driving videos in a more generalizable way but often struggle with maintaining 3D visual consistency. In this paper, we present DreamDrive, a 4D spatial-temporal scene generation approach that combines the merits of generation and reconstruction, to synthesize generalizable 4D driving scenes and dynamic driving videos with 3D consistency. Specifically, we leverage the generative power of video diffusion models to synthesize a sequence of visual references and further elevate them to 4D with a novel hybrid Gaussian representation. Given a driving trajectory, we then render 3D-consistent driving videos via Gaussian splatting. The use of generative priors allows our method to produce high-quality 4D scenes from in-the-wild driving data, while neural rendering ensures 3D-consistent video generation from the 4D scenes. Extensive experiments on nuScenes and street view images demonstrate that DreamDrive can generate controllable and generalizable 4D driving scenes, synthesize novel views of driving videos with high fidelity and 3D consistency, decompose static and dynamic elements in a self-supervised manner, and enhance perception and planning tasks for autonomous driving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01895v1">EnerVerse: Envisioning Embodied Future Space for Robotics Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-01-03
      | 💬 Website: https://sites.google.com/view/enerverse
    </div>
    <details class="paper-abstract">
      We introduce EnerVerse, a comprehensive framework for embodied future space generation specifically designed for robotic manipulation tasks. EnerVerse seamlessly integrates convolutional and bidirectional attention mechanisms for inner-chunk space modeling, ensuring low-level consistency and continuity. Recognizing the inherent redundancy in video data, we propose a sparse memory context combined with a chunkwise unidirectional generative paradigm to enable the generation of infinitely long sequences. To further augment robotic capabilities, we introduce the Free Anchor View (FAV) space, which provides flexible perspectives to enhance observation and analysis. The FAV space mitigates motion modeling ambiguity, removes physical constraints in confined environments, and significantly improves the robot's generalization and adaptability across various tasks and settings. To address the prohibitive costs and labor intensity of acquiring multi-camera observations, we present a data engine pipeline that integrates a generative model with 4D Gaussian Splatting (4DGS). This pipeline leverages the generative model's robust generalization capabilities and the spatial constraints provided by 4DGS, enabling an iterative enhancement of data quality and diversity, thus creating a data flywheel effect that effectively narrows the sim-to-real gap. Finally, our experiments demonstrate that the embodied future space generation prior substantially enhances policy predictive capabilities, resulting in improved overall performance, particularly in long-range robotic manipulation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01715v1">Cloth-Splatting: 3D Cloth State Estimation from RGB Supervision</a></div>
    <div class="paper-meta">
      📅 2025-01-03
      | 💬 Accepted at the 8th Conference on Robot Learning (CoRL 2024). Code and videos available at: kth-rpl.github.io/cloth-splatting
    </div>
    <details class="paper-abstract">
      We introduce Cloth-Splatting, a method for estimating 3D states of cloth from RGB images through a prediction-update framework. Cloth-Splatting leverages an action-conditioned dynamics model for predicting future states and uses 3D Gaussian Splatting to update the predicted states. Our key insight is that coupling a 3D mesh-based representation with Gaussian Splatting allows us to define a differentiable map between the cloth state space and the image space. This enables the use of gradient-based optimization techniques to refine inaccurate state estimates using only RGB supervision. Our experiments demonstrate that Cloth-Splatting not only improves state estimation accuracy over current baselines but also reduces convergence time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01695v1">CrossView-GS: Cross-view Gaussian Splatting For Large-scale Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-01-03
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a prominent method for scene representation and reconstruction, leveraging densely distributed Gaussian primitives to enable real-time rendering of high-resolution images. While existing 3DGS methods perform well in scenes with minor view variation, large view changes in cross-view scenes pose optimization challenges for these methods. To address these issues, we propose a novel cross-view Gaussian Splatting method for large-scale scene reconstruction, based on dual-branch fusion. Our method independently reconstructs models from aerial and ground views as two independent branches to establish the baselines of Gaussian distribution, providing reliable priors for cross-view reconstruction during both initialization and densification. Specifically, a gradient-aware regularization strategy is introduced to mitigate smoothing issues caused by significant view disparities. Additionally, a unique Gaussian supplementation strategy is utilized to incorporate complementary information of dual-branch into the cross-view model. Extensive experiments on benchmark datasets demonstrate that our method achieves superior performance in novel view synthesis compared to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01677v1">PG-SAG: Parallel Gaussian Splatting for Fine-Grained Large-Scale Urban Buildings Reconstruction via Semantic-Aware Grouping</a></div>
    <div class="paper-meta">
      📅 2025-01-03
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a transformative method in the field of real-time novel synthesis. Based on 3DGS, recent advancements cope with large-scale scenes via spatial-based partition strategy to reduce video memory and optimization time costs. In this work, we introduce a parallel Gaussian splatting method, termed PG-SAG, which fully exploits semantic cues for both partitioning and Gaussian kernel optimization, enabling fine-grained building surface reconstruction of large-scale urban areas without downsampling the original image resolution. First, the Cross-modal model - Language Segment Anything is leveraged to segment building masks. Then, the segmented building regions is grouped into sub-regions according to the visibility check across registered images. The Gaussian kernels for these sub-regions are optimized in parallel with masked pixels. In addition, the normal loss is re-formulated for the detected edges of masks to alleviate the ambiguities in normal vectors on edges. Finally, to improve the optimization of 3D Gaussians, we introduce a gradient-constrained balance-load loss that accounts for the complexity of the corresponding scenes, effectively minimizing the thread waiting time in the pixel-parallel rendering stage as well as the reconstruction lost. Extensive experiments are tested on various urban datasets, the results demonstrated the superior performance of our PG-SAG on building surface reconstruction, compared to several state-of-the-art 3DGS-based methods. Project Web:https://github.com/TFWang-9527/PG-SAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01101v1">Deformable Gaussian Splatting for Efficient and High-Fidelity Reconstruction of Surgical Scenes</a></div>
    <div class="paper-meta">
      📅 2025-01-02
      | 💬 7 pages, 4 figures, submitted to ICRA 2025
    </div>
    <details class="paper-abstract">
      Efficient and high-fidelity reconstruction of deformable surgical scenes is a critical yet challenging task. Building on recent advancements in 3D Gaussian splatting, current methods have seen significant improvements in both reconstruction quality and rendering speed. However, two major limitations remain: (1) difficulty in handling irreversible dynamic changes, such as tissue shearing, which are common in surgical scenes; and (2) the lack of hierarchical modeling for surgical scene deformation, which reduces rendering speed. To address these challenges, we introduce EH-SurGS, an efficient and high-fidelity reconstruction algorithm for deformable surgical scenes. We propose a deformation modeling approach that incorporates the life cycle of 3D Gaussians, effectively capturing both regular and irreversible deformations, thus enhancing reconstruction quality. Additionally, we present an adaptive motion hierarchy strategy that distinguishes between static and deformable regions within the surgical scene. This strategy reduces the number of 3D Gaussians passing through the deformation field, thereby improving rendering speed. Extensive experiments demonstrate that our method surpasses existing state-of-the-art approaches in both reconstruction quality and rendering speed. Ablation studies further validate the effectiveness and necessity of our proposed components. We will open-source our code upon acceptance of the paper.
    </details>
</div>
