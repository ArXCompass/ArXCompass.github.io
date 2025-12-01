# gaussian splatting - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04297v2">MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 This work is accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      We present Multi-Baseline Gaussian Splatting (MuGS), a generalized feed-forward approach for novel view synthesis that effectively handles diverse baseline settings, including sparse input views with both small and large baselines. Specifically, we integrate features from Multi-View Stereo (MVS) and Monocular Depth Estimation (MDE) to enhance feature representations for generalizable reconstruction. Next, We propose a projection-and-sampling mechanism for deep depth fusion, which constructs a fine probability volume to guide the regression of the feature map. Furthermore, We introduce a reference-view loss to improve geometry and optimization efficiency. We leverage 3D Gaussian representations to accelerate training and inference time while enhancing rendering quality. MuGS achieves state-of-the-art performance across multiple baseline settings and diverse scenarios ranging from simple objects (DTU) to complex indoor and outdoor scenes (RealEstate10K). We also demonstrate promising zero-shot performance on the LLFF and Mip-NeRF 360 datasets. Code is available at https://github.com/EuclidLou/MuGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2510.19210v1">MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      Recent advances in dynamic scene reconstruction have significantly benefited from 3D Gaussian Splatting, yet existing methods show inconsistent performance across diverse scenes, indicating no single approach effectively handles all dynamic challenges. To overcome these limitations, we propose Mixture of Experts for Dynamic Gaussian Splatting (MoE-GS), a unified framework integrating multiple specialized experts via a novel Volume-aware Pixel Router. Our router adaptively blends expert outputs by projecting volumetric Gaussian-level weights into pixel space through differentiable weight splatting, ensuring spatially and temporally coherent results. Although MoE-GS improves rendering quality, the increased model capacity and reduced FPS are inherent to the MoE architecture. To mitigate this, we explore two complementary directions: (1) single-pass multi-expert rendering and gate-aware Gaussian pruning, which improve efficiency within the MoE framework, and (2) a distillation strategy that transfers MoE performance to individual experts, enabling lightweight deployment without architectural changes. To the best of our knowledge, MoE-GS is the first approach incorporating Mixture-of-Experts techniques into dynamic Gaussian splatting. Extensive experiments on the N3V and Technicolor datasets demonstrate that MoE-GS consistently outperforms state-of-the-art methods with improved efficiency. Video demonstrations are available at https://anonymous.4open.science/w/MoE-GS-68BA/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04297v2">MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 This work is accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      We present Multi-Baseline Gaussian Splatting (MuGS), a generalized feed-forward approach for novel view synthesis that effectively handles diverse baseline settings, including sparse input views with both small and large baselines. Specifically, we integrate features from Multi-View Stereo (MVS) and Monocular Depth Estimation (MDE) to enhance feature representations for generalizable reconstruction. Next, We propose a projection-and-sampling mechanism for deep depth fusion, which constructs a fine probability volume to guide the regression of the feature map. Furthermore, We introduce a reference-view loss to improve geometry and optimization efficiency. We leverage 3D Gaussian representations to accelerate training and inference time while enhancing rendering quality. MuGS achieves state-of-the-art performance across multiple baseline settings and diverse scenarios ranging from simple objects (DTU) to complex indoor and outdoor scenes (RealEstate10K). We also demonstrate promising zero-shot performance on the LLFF and Mip-NeRF 360 datasets. Code is available at https://github.com/EuclidLou/MuGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20813v1">GSWorld: Closed-Loop Photo-Realistic Simulation Suite for Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      This paper presents GSWorld, a robust, photo-realistic simulator for robotics manipulation that combines 3D Gaussian Splatting with physics engines. Our framework advocates "closing the loop" of developing manipulation policies with reproducible evaluation of policies learned from real-robot data and sim2real policy training without using real robots. To enable photo-realistic rendering of diverse scenes, we propose a new asset format, which we term GSDF (Gaussian Scene Description File), that infuses Gaussian-on-Mesh representation with robot URDF and other objects. With a streamlined reconstruction pipeline, we curate a database of GSDF that contains 3 robot embodiments for single-arm and bimanual manipulation, as well as more than 40 objects. Combining GSDF with physics engines, we demonstrate several immediate interesting applications: (1) learning zero-shot sim2real pixel-to-action manipulation policy with photo-realistic rendering, (2) automated high-quality DAgger data collection for adapting policies to deployment environments, (3) reproducible benchmarking of real-robot manipulation policies in simulation, (4) simulation data collection by virtual teleoperation, and (5) zero-shot sim2real visual reinforcement learning. Website: https://3dgsworld.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14475v2">Optimized 3D Gaussian Splatting using Coarse-to-Fine Image Frequency Modulation</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      The field of Novel View Synthesis has been revolutionized by 3D Gaussian Splatting (3DGS), which enables high-quality scene reconstruction that can be rendered in real-time. 3DGS-based techniques typically suffer from high GPU memory and disk storage requirements which limits their practical application on consumer-grade devices. We propose Opti3DGS, a novel frequency-modulated coarse-to-fine optimization framework that aims to minimize the number of Gaussian primitives used to represent a scene, thus reducing memory and storage demands. Opti3DGS leverages image frequency modulation, initially enforcing a coarse scene representation and progressively refining it by modulating frequency details in the training images. On the baseline 3DGS, we demonstrate an average reduction of 62% in Gaussians, a 40% reduction in the training GPU memory requirements and a 20% reduction in optimization time without sacrificing the visual quality. Furthermore, we show that our method integrates seamlessly with many 3DGS-based techniques, consistently reducing the number of Gaussian primitives while maintaining, and often improving, visual quality. Additionally, Opti3DGS inherently produces a level-of-detail scene representation at no extra cost, a natural byproduct of the optimization pipeline. Results and code will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04844v4">Discretized Gaussian Representation for Tomographic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Accepted to ICCV 2025
    </div>
    <details class="paper-abstract">
      Computed Tomography (CT) enables detailed cross-sectional imaging but continues to face challenges in balancing reconstruction quality and computational efficiency. While deep learning-based methods have significantly improved image quality and noise reduction, they typically require large-scale training data and intensive computation. Recent advances in scene reconstruction, such as Neural Radiance Fields and 3D Gaussian Splatting, offer alternative perspectives but are not well-suited for direct volumetric CT reconstruction. In this work, we propose Discretized Gaussian Representation (DGR), a novel framework that reconstructs the 3D volume directly using a set of discretized Gaussian functions in an end-to-end manner. To further enhance efficiency, we introduce Fast Volume Reconstruction, a highly parallelized technique that aggregates Gaussian contributions into the voxel grid with minimal overhead. Extensive experiments on both real-world and synthetic datasets demonstrate that DGR achieves superior reconstruction quality and runtime performance across various CT reconstruction scenarios. Our code is publicly available at https://github.com/wskingdom/DGR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19578v1">VGD: Visual Geometry Gaussian Splatting for Feed-Forward Surround-view Driving Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Feed-forward surround-view autonomous driving scene reconstruction offers fast, generalizable inference ability, which faces the core challenge of ensuring generalization while elevating novel view quality. Due to the surround-view with minimal overlap regions, existing methods typically fail to ensure geometric consistency and reconstruction quality for novel views. To tackle this tension, we claim that geometric information must be learned explicitly, and the resulting features should be leveraged to guide the elevating of semantic quality in novel views. In this paper, we introduce \textbf{Visual Gaussian Driving (VGD)}, a novel feed-forward end-to-end learning framework designed to address this challenge. To achieve generalizable geometric estimation, we design a lightweight variant of the VGGT architecture to efficiently distill its geometric priors from the pre-trained VGGT to the geometry branch. Furthermore, we design a Gaussian Head that fuses multi-scale geometry tokens to predict Gaussian parameters for novel view rendering, which shares the same patch backbone as the geometry branch. Finally, we integrate multi-scale features from both geometry and Gaussian head branches to jointly supervise a semantic refinement model, optimizing rendering quality through feature-consistent learning. Experiments on nuScenes demonstrate that our approach significantly outperforms state-of-the-art methods in both objective metrics and subjective quality under various settings, which validates VGD's scalability and high-fidelity surround-view reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19255v1">Advances in 4D Representation: Geometry, Motion, and Interaction</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 21 pages. Project Page: https://mingrui-zhao.github.io/4DRep-GMI/
    </div>
    <details class="paper-abstract">
      We present a survey on 4D generation and reconstruction, a fast-evolving subfield of computer graphics whose developments have been propelled by recent advances in neural fields, geometric and motion deep learning, as well 3D generative artificial intelligence (GenAI). While our survey is not the first of its kind, we build our coverage of the domain from a unique and distinctive perspective of 4D representations\/}, to model 3D geometry evolving over time while exhibiting motion and interaction. Specifically, instead of offering an exhaustive enumeration of many works, we take a more selective approach by focusing on representative works to highlight both the desirable properties and ensuing challenges of each representation under different computation, application, and data scenarios. The main take-away message we aim to convey to the readers is on how to select and then customize the appropriate 4D representations for their tasks. Organizationally, we separate the 4D representations based on three key pillars: geometry, motion, and interaction. Our discourse will not only encompass the most popular representations of today, such as neural radiance fields (NeRFs) and 3D Gaussian Splatting (3DGS), but also bring attention to relatively under-explored representations in the 4D context, such as structured models and long-range motions. Throughout our survey, we will reprise the role of large language models (LLMs) and video foundational models (VFMs) in a variety of 4D applications, while steering our discussion towards their current limitations and how they can be addressed. We also provide a dedicated coverage on what 4D datasets are currently available, as well as what is lacking, in driving the subfield forward. Project page:https://mingrui-zhao.github.io/4DRep-GMI/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19210v1">MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Recent advances in dynamic scene reconstruction have significantly benefited from 3D Gaussian Splatting, yet existing methods show inconsistent performance across diverse scenes, indicating no single approach effectively handles all dynamic challenges. To overcome these limitations, we propose Mixture of Experts for Dynamic Gaussian Splatting (MoE-GS), a unified framework integrating multiple specialized experts via a novel Volume-aware Pixel Router. Our router adaptively blends expert outputs by projecting volumetric Gaussian-level weights into pixel space through differentiable weight splatting, ensuring spatially and temporally coherent results. Although MoE-GS improves rendering quality, the increased model capacity and reduced FPS are inherent to the MoE architecture. To mitigate this, we explore two complementary directions: (1) single-pass multi-expert rendering and gate-aware Gaussian pruning, which improve efficiency within the MoE framework, and (2) a distillation strategy that transfers MoE performance to individual experts, enabling lightweight deployment without architectural changes. To the best of our knowledge, MoE-GS is the first approach incorporating Mixture-of-Experts techniques into dynamic Gaussian splatting. Extensive experiments on the N3V and Technicolor datasets demonstrate that MoE-GS consistently outperforms state-of-the-art methods with improved efficiency. Video demonstrations are available at https://anonymous.4open.science/w/MoE-GS-68BA/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19200v1">GRASPLAT: Enabling dexterous grasping through novel view synthesis</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Accepted IROS 2025
    </div>
    <details class="paper-abstract">
      Achieving dexterous robotic grasping with multi-fingered hands remains a significant challenge. While existing methods rely on complete 3D scans to predict grasp poses, these approaches face limitations due to the difficulty of acquiring high-quality 3D data in real-world scenarios. In this paper, we introduce GRASPLAT, a novel grasping framework that leverages consistent 3D information while being trained solely on RGB images. Our key insight is that by synthesizing physically plausible images of a hand grasping an object, we can regress the corresponding hand joints for a successful grasp. To achieve this, we utilize 3D Gaussian Splatting to generate high-fidelity novel views of real hand-object interactions, enabling end-to-end training with RGB data. Unlike prior methods, our approach incorporates a photometric loss that refines grasp predictions by minimizing discrepancies between rendered and real images. We conduct extensive experiments on both synthetic and real-world grasping datasets, demonstrating that GRASPLAT improves grasp success rates up to 36.9% over existing image-based methods. Project page: https://mbortolon97.github.io/grasplat/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20027v1">Extreme Views: 3DGS Filter for Novel View Synthesis from Out-of-Distribution Camera Poses</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      When viewing a 3D Gaussian Splatting (3DGS) model from camera positions significantly outside the training data distribution, substantial visual noise commonly occurs. These artifacts result from the lack of training data in these extrapolated regions, leading to uncertain density, color, and geometry predictions from the model. To address this issue, we propose a novel real-time render-aware filtering method. Our approach leverages sensitivity scores derived from intermediate gradients, explicitly targeting instabilities caused by anisotropic orientations rather than isotropic variance. This filtering method directly addresses the core issue of generative uncertainty, allowing 3D reconstruction systems to maintain high visual fidelity even when users freely navigate outside the original training viewpoints. Experimental evaluation demonstrates that our method substantially improves visual quality, realism, and consistency compared to existing Neural Radiance Field (NeRF)-based approaches such as BayesRays. Critically, our filter seamlessly integrates into existing 3DGS rendering pipelines in real-time, unlike methods that require extensive post-hoc retraining or fine-tuning. Code and results at https://damian-bowness.github.io/EV3DGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2505.22978v3">Pose-free 3D Gaussian splatting via shape-ray estimation</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 ICIP 2025 (Best Student Paper Award) Code available at: https://github.com/youngju-na/SHARE
    </div>
    <details class="paper-abstract">
      While generalizable 3D Gaussian splatting enables efficient, high-quality rendering of unseen scenes, it heavily depends on precise camera poses for accurate geometry. In real-world scenarios, obtaining accurate poses is challenging, leading to noisy pose estimates and geometric misalignments. To address this, we introduce SHARE, a pose-free, feed-forward Gaussian splatting framework that overcomes these ambiguities by joint shape and camera rays estimation. Instead of relying on explicit 3D transformations, SHARE builds a pose-aware canonical volume representation that seamlessly integrates multi-view information, reducing misalignment caused by inaccurate pose estimates. Additionally, anchor-aligned Gaussian prediction enhances scene reconstruction by refining local geometry around coarse anchors, allowing for more precise Gaussian placement. Extensive experiments on diverse real-world datasets show that our method achieves robust performance in pose-free generalizable Gaussian splatting. Code is avilable at https://github.com/youngju-na/SHARE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2510.18101v1">From Volume Rendering to 3D Gaussian Splatting: Theory and Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Accepted at the Conference on Graphics, Patterns and Images (SIBGRAPI), math focused, 5 equations, 5 Figure, 5 pages of text and 1 of bibligraphy
    </div>
    <details class="paper-abstract">
      The problem of 3D reconstruction from posed images is undergoing a fundamental transformation, driven by continuous advances in 3D Gaussian Splatting (3DGS). By modeling scenes explicitly as collections of 3D Gaussians, 3DGS enables efficient rasterization through volumetric splatting, offering thus a seamless integration with common graphics pipelines. Despite its real-time rendering capabilities for novel view synthesis, 3DGS suffers from a high memory footprint, the tendency to bake lighting effects directly into its representation, and limited support for secondary-ray effects. This tutorial provides a concise yet comprehensive overview of the 3DGS pipeline, starting from its splatting formulation and then exploring the main efforts in addressing its limitations. Finally, we survey a range of applications that leverage 3DGS for surface reconstruction, avatar modeling, animation, and content generation-highlighting its efficient rendering and suitability for feed-forward pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19210v1">MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Recent advances in dynamic scene reconstruction have significantly benefited from 3D Gaussian Splatting, yet existing methods show inconsistent performance across diverse scenes, indicating no single approach effectively handles all dynamic challenges. To overcome these limitations, we propose Mixture of Experts for Dynamic Gaussian Splatting (MoE-GS), a unified framework integrating multiple specialized experts via a novel Volume-aware Pixel Router. Our router adaptively blends expert outputs by projecting volumetric Gaussian-level weights into pixel space through differentiable weight splatting, ensuring spatially and temporally coherent results. Although MoE-GS improves rendering quality, the increased model capacity and reduced FPS are inherent to the MoE architecture. To mitigate this, we explore two complementary directions: (1) single-pass multi-expert rendering and gate-aware Gaussian pruning, which improve efficiency within the MoE framework, and (2) a distillation strategy that transfers MoE performance to individual experts, enabling lightweight deployment without architectural changes. To the best of our knowledge, MoE-GS is the first approach incorporating Mixture-of-Experts techniques into dynamic Gaussian splatting. Extensive experiments on the N3V and Technicolor datasets demonstrate that MoE-GS consistently outperforms state-of-the-art methods with improved efficiency. Video demonstrations are available at https://anonymous.4open.science/w/MoE-GS-68BA/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20027v1">Extreme Views: 3DGS Filter for Novel View Synthesis from Out-of-Distribution Camera Poses</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      When viewing a 3D Gaussian Splatting (3DGS) model from camera positions significantly outside the training data distribution, substantial visual noise commonly occurs. These artifacts result from the lack of training data in these extrapolated regions, leading to uncertain density, color, and geometry predictions from the model. To address this issue, we propose a novel real-time render-aware filtering method. Our approach leverages sensitivity scores derived from intermediate gradients, explicitly targeting instabilities caused by anisotropic orientations rather than isotropic variance. This filtering method directly addresses the core issue of generative uncertainty, allowing 3D reconstruction systems to maintain high visual fidelity even when users freely navigate outside the original training viewpoints. Experimental evaluation demonstrates that our method substantially improves visual quality, realism, and consistency compared to existing Neural Radiance Field (NeRF)-based approaches such as BayesRays. Critically, our filter seamlessly integrates into existing 3DGS rendering pipelines in real-time, unlike methods that require extensive post-hoc retraining or fine-tuning. Code and results at https://damian-bowness.github.io/EV3DGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.14475v2">Optimized 3D Gaussian Splatting using Coarse-to-Fine Image Frequency Modulation</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      The field of Novel View Synthesis has been revolutionized by 3D Gaussian Splatting (3DGS), which enables high-quality scene reconstruction that can be rendered in real-time. 3DGS-based techniques typically suffer from high GPU memory and disk storage requirements which limits their practical application on consumer-grade devices. We propose Opti3DGS, a novel frequency-modulated coarse-to-fine optimization framework that aims to minimize the number of Gaussian primitives used to represent a scene, thus reducing memory and storage demands. Opti3DGS leverages image frequency modulation, initially enforcing a coarse scene representation and progressively refining it by modulating frequency details in the training images. On the baseline 3DGS, we demonstrate an average reduction of 62% in Gaussians, a 40% reduction in the training GPU memory requirements and a 20% reduction in optimization time without sacrificing the visual quality. Furthermore, we show that our method integrates seamlessly with many 3DGS-based techniques, consistently reducing the number of Gaussian primitives while maintaining, and often improving, visual quality. Additionally, Opti3DGS inherently produces a level-of-detail scene representation at no extra cost, a natural byproduct of the optimization pipeline. Results and code will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.04844v4">Discretized Gaussian Representation for Tomographic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Accepted to ICCV 2025
    </div>
    <details class="paper-abstract">
      Computed Tomography (CT) enables detailed cross-sectional imaging but continues to face challenges in balancing reconstruction quality and computational efficiency. While deep learning-based methods have significantly improved image quality and noise reduction, they typically require large-scale training data and intensive computation. Recent advances in scene reconstruction, such as Neural Radiance Fields and 3D Gaussian Splatting, offer alternative perspectives but are not well-suited for direct volumetric CT reconstruction. In this work, we propose Discretized Gaussian Representation (DGR), a novel framework that reconstructs the 3D volume directly using a set of discretized Gaussian functions in an end-to-end manner. To further enhance efficiency, we introduce Fast Volume Reconstruction, a highly parallelized technique that aggregates Gaussian contributions into the voxel grid with minimal overhead. Extensive experiments on both real-world and synthetic datasets demonstrate that DGR achieves superior reconstruction quality and runtime performance across various CT reconstruction scenarios. Our code is publicly available at https://github.com/wskingdom/DGR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19578v1">VGD: Visual Geometry Gaussian Splatting for Feed-Forward Surround-view Driving Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Feed-forward surround-view autonomous driving scene reconstruction offers fast, generalizable inference ability, which faces the core challenge of ensuring generalization while elevating novel view quality. Due to the surround-view with minimal overlap regions, existing methods typically fail to ensure geometric consistency and reconstruction quality for novel views. To tackle this tension, we claim that geometric information must be learned explicitly, and the resulting features should be leveraged to guide the elevating of semantic quality in novel views. In this paper, we introduce \textbf{Visual Gaussian Driving (VGD)}, a novel feed-forward end-to-end learning framework designed to address this challenge. To achieve generalizable geometric estimation, we design a lightweight variant of the VGGT architecture to efficiently distill its geometric priors from the pre-trained VGGT to the geometry branch. Furthermore, we design a Gaussian Head that fuses multi-scale geometry tokens to predict Gaussian parameters for novel view rendering, which shares the same patch backbone as the geometry branch. Finally, we integrate multi-scale features from both geometry and Gaussian head branches to jointly supervise a semantic refinement model, optimizing rendering quality through feature-consistent learning. Experiments on nuScenes demonstrate that our approach significantly outperforms state-of-the-art methods in both objective metrics and subjective quality under various settings, which validates VGD's scalability and high-fidelity surround-view reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19255v1">Advances in 4D Representation: Geometry, Motion, and Interaction</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 21 pages. Project Page: https://mingrui-zhao.github.io/4DRep-GMI/
    </div>
    <details class="paper-abstract">
      We present a survey on 4D generation and reconstruction, a fast-evolving subfield of computer graphics whose developments have been propelled by recent advances in neural fields, geometric and motion deep learning, as well 3D generative artificial intelligence (GenAI). While our survey is not the first of its kind, we build our coverage of the domain from a unique and distinctive perspective of 4D representations\/}, to model 3D geometry evolving over time while exhibiting motion and interaction. Specifically, instead of offering an exhaustive enumeration of many works, we take a more selective approach by focusing on representative works to highlight both the desirable properties and ensuing challenges of each representation under different computation, application, and data scenarios. The main take-away message we aim to convey to the readers is on how to select and then customize the appropriate 4D representations for their tasks. Organizationally, we separate the 4D representations based on three key pillars: geometry, motion, and interaction. Our discourse will not only encompass the most popular representations of today, such as neural radiance fields (NeRFs) and 3D Gaussian Splatting (3DGS), but also bring attention to relatively under-explored representations in the 4D context, such as structured models and long-range motions. Throughout our survey, we will reprise the role of large language models (LLMs) and video foundational models (VFMs) in a variety of 4D applications, while steering our discussion towards their current limitations and how they can be addressed. We also provide a dedicated coverage on what 4D datasets are currently available, as well as what is lacking, in driving the subfield forward. Project page:https://mingrui-zhao.github.io/4DRep-GMI/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19200v1">GRASPLAT: Enabling dexterous grasping through novel view synthesis</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Accepted IROS 2025
    </div>
    <details class="paper-abstract">
      Achieving dexterous robotic grasping with multi-fingered hands remains a significant challenge. While existing methods rely on complete 3D scans to predict grasp poses, these approaches face limitations due to the difficulty of acquiring high-quality 3D data in real-world scenarios. In this paper, we introduce GRASPLAT, a novel grasping framework that leverages consistent 3D information while being trained solely on RGB images. Our key insight is that by synthesizing physically plausible images of a hand grasping an object, we can regress the corresponding hand joints for a successful grasp. To achieve this, we utilize 3D Gaussian Splatting to generate high-fidelity novel views of real hand-object interactions, enabling end-to-end training with RGB data. Unlike prior methods, our approach incorporates a photometric loss that refines grasp predictions by minimizing discrepancies between rendered and real images. We conduct extensive experiments on both synthetic and real-world grasping datasets, demonstrating that GRASPLAT improves grasp success rates up to 36.9% over existing image-based methods. Project page: https://mbortolon97.github.io/grasplat/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18739v1">Moving Light Adaptive Colonoscopy Reconstruction via Illumination-Attenuation-Aware 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a pivotal technique for real-time view synthesis in colonoscopy, enabling critical applications such as virtual colonoscopy and lesion tracking. However, the vanilla 3DGS assumes static illumination and that observed appearance depends solely on viewing angle, which causes incompatibility with the photometric variations in colonoscopic scenes induced by dynamic light source/camera. This mismatch forces most 3DGS methods to introduce structure-violating vaporous Gaussian blobs between the camera and tissues to compensate for illumination attenuation, ultimately degrading the quality of 3D reconstructions. Previous works only consider the illumination attenuation caused by light distance, ignoring the physical characters of light source and camera. In this paper, we propose ColIAGS, an improved 3DGS framework tailored for colonoscopy. To mimic realistic appearance under varying illumination, we introduce an Improved Appearance Modeling with two types of illumination attenuation factors, which enables Gaussians to adapt to photometric variations while preserving geometry accuracy. To ensure the geometry approximation condition of appearance modeling, we propose an Improved Geometry Modeling using high-dimensional view embedding to enhance Gaussian geometry attribute prediction. Furthermore, another cosine embedding input is leveraged to generate illumination attenuation solutions in an implicit manner. Comprehensive experimental results on standard benchmarks demonstrate that our proposed ColIAGS achieves the dual capabilities of novel view synthesis and accurate geometric reconstruction. It notably outperforms other state-of-the-art methods by achieving superior rendering fidelity while significantly reducing Depth MSE. Code will be available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10759v2">Every Camera Effect, Every Time, All at Once: 4D Gaussian Ray Tracing for Physics-based Camera Effect Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Paper accepted to NeurIPS 2025 Workshop SpaVLE. Project page: https://shigon255.github.io/4DGRT-project-page/
    </div>
    <details class="paper-abstract">
      Common computer vision systems typically assume ideal pinhole cameras but fail when facing real-world camera effects such as fisheye distortion and rolling shutter, mainly due to the lack of learning from training data with camera effects. Existing data generation approaches suffer from either high costs, sim-to-real gaps or fail to accurately model camera effects. To address this bottleneck, we propose 4D Gaussian Ray Tracing (4D-GRT), a novel two-stage pipeline that combines 4D Gaussian Splatting with physically-based ray tracing for camera effect simulation. Given multi-view videos, 4D-GRT first reconstructs dynamic scenes, then applies ray tracing to generate videos with controllable, physically accurate camera effects. 4D-GRT achieves the fastest rendering speed while performing better or comparable rendering quality compared to existing baselines. Additionally, we construct eight synthetic dynamic scenes in indoor environments across four camera effects as a benchmark to evaluate generated videos with camera effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22978v3">Pose-free 3D Gaussian splatting via shape-ray estimation</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 ICIP 2025 (Best Student Paper Award) Code available at: https://github.com/youngju-na/SHARE
    </div>
    <details class="paper-abstract">
      While generalizable 3D Gaussian splatting enables efficient, high-quality rendering of unseen scenes, it heavily depends on precise camera poses for accurate geometry. In real-world scenarios, obtaining accurate poses is challenging, leading to noisy pose estimates and geometric misalignments. To address this, we introduce SHARE, a pose-free, feed-forward Gaussian splatting framework that overcomes these ambiguities by joint shape and camera rays estimation. Instead of relying on explicit 3D transformations, SHARE builds a pose-aware canonical volume representation that seamlessly integrates multi-view information, reducing misalignment caused by inaccurate pose estimates. Additionally, anchor-aligned Gaussian prediction enhances scene reconstruction by refining local geometry around coarse anchors, allowing for more precise Gaussian placement. Extensive experiments on diverse real-world datasets show that our method achieves robust performance in pose-free generalizable Gaussian splatting. Code is avilable at https://github.com/youngju-na/SHARE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18489v1">Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Project page is available at https://liujf1226.github.io/Mono4DGS-HDR/
    </div>
    <details class="paper-abstract">
      We introduce Mono4DGS-HDR, the first system for reconstructing renderable 4D high dynamic range (HDR) scenes from unposed monocular low dynamic range (LDR) videos captured with alternating exposures. To tackle such a challenging problem, we present a unified framework with two-stage optimization approach based on Gaussian Splatting. The first stage learns a video HDR Gaussian representation in orthographic camera coordinate space, eliminating the need for camera poses and enabling robust initial HDR video reconstruction. The second stage transforms video Gaussians into world space and jointly refines the world Gaussians with camera poses. Furthermore, we propose a temporal luminance regularization strategy to enhance the temporal consistency of the HDR appearance. Since our task has not been studied before, we construct a new evaluation benchmark using publicly available datasets for HDR video reconstruction. Extensive experiments demonstrate that Mono4DGS-HDR significantly outperforms alternative solutions adapted from state-of-the-art methods in both rendering quality and speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13036v3">H3D-DGS: Exploring Heterogeneous 3D Motion Representation for Deformable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction poses a persistent challenge in 3D vision. Deformable 3D Gaussian Splatting has emerged as an effective method for this task, offering real-time rendering and high visual fidelity. This approach decomposes a dynamic scene into a static representation in a canonical space and time-varying scene motion. Scene motion is defined as the collective movement of all Gaussian points, and for compactness, existing approaches commonly adopt implicit neural fields or sparse control points. However, these methods predominantly rely on gradient-based optimization for all motion information. Due to the high degree of freedom, they struggle to converge on real-world datasets exhibiting complex motion. To preserve the compactness of motion representation and address convergence challenges, this paper proposes heterogeneous 3D control points, termed \textbf{H3D control points}, whose attributes are obtained using a hybrid strategy combining optical flow back-projection and gradient-based methods. This design decouples directly observable motion components from those that are geometrically occluded. Specifically, components of 3D motion that project onto the image plane are directly acquired via optical flow back projection, while unobservable portions are refined through gradient-based optimization. Experiments on the Neu3DV and CMU-Panoptic datasets demonstrate that our method achieves superior performance over state-of-the-art deformable 3D Gaussian splatting techniques. Remarkably, our method converges within just 100 iterations and achieves a per-frame processing speed of 2 seconds on a single NVIDIA RTX 4070 GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18253v1">OpenInsGaussian: Open-vocabulary Instance Gaussian Segmentation with Context-aware Cross-view Fusion</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Understanding 3D scenes is pivotal for autonomous driving, robotics, and augmented reality. Recent semantic Gaussian Splatting approaches leverage large-scale 2D vision models to project 2D semantic features onto 3D scenes. However, they suffer from two major limitations: (1) insufficient contextual cues for individual masks during preprocessing and (2) inconsistencies and missing details when fusing multi-view features from these 2D models. In this paper, we introduce \textbf{OpenInsGaussian}, an \textbf{Open}-vocabulary \textbf{Ins}tance \textbf{Gaussian} segmentation framework with Context-aware Cross-view Fusion. Our method consists of two modules: Context-Aware Feature Extraction, which augments each mask with rich semantic context, and Attention-Driven Feature Aggregation, which selectively fuses multi-view features to mitigate alignment errors and incompleteness. Through extensive experiments on benchmark datasets, OpenInsGaussian achieves state-of-the-art results in open-vocabulary 3D Gaussian segmentation, outperforming existing baselines by a large margin. These findings underscore the robustness and generality of our proposed approach, marking a significant step forward in 3D scene understanding and its practical deployment across diverse real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19653v1">Re-Activating Frozen Primitives for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D-GS) achieves real-time photorealistic novel view synthesis, yet struggles with complex scenes due to over-reconstruction artifacts, manifesting as local blurring and needle-shape distortions. While recent approaches attribute these issues to insufficient splitting of large-scale Gaussians, we identify two fundamental limitations: gradient magnitude dilution during densification and the primitive frozen phenomenon, where essential Gaussian densification is inhibited in complex regions while suboptimally scaled Gaussians become trapped in local optima. To address these challenges, we introduce ReAct-GS, a method founded on the principle of re-activation. Our approach features: (1) an importance-aware densification criterion incorporating $\alpha$-blending weights from multiple viewpoints to re-activate stalled primitive growth in complex regions, and (2) a re-activation mechanism that revitalizes frozen primitives through adaptive parameter perturbations. Comprehensive experiments across diverse real-world datasets demonstrate that ReAct-GS effectively eliminates over-reconstruction artifacts and achieves state-of-the-art performance on standard novel view synthesis metrics while preserving intricate geometric details. Additionally, our re-activation mechanism yields consistent improvements when integrated with other 3D-GS variants such as Pixel-GS, demonstrating its broad applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2504.10809v4">GaSLight: Gaussian Splats for Spatially-Varying Lighting in HDR</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      We present GaSLight, a method that generates spatially-varying lighting from regular images. Our method proposes using HDR Gaussian Splats as light source representation, marking the first time regular images can serve as light sources in a 3D renderer. Our two-stage process first enhances the dynamic range of images plausibly and accurately by leveraging the priors embedded in diffusion models. Next, we employ Gaussian Splats to model 3D lighting, achieving spatially variant lighting. Our approach yields state-of-the-art results on HDR estimations and their applications in illuminating virtual objects and scenes. To facilitate the benchmarking of images as light sources, we introduce a novel dataset of calibrated and unsaturated HDR to evaluate images as light sources. We assess our method using a combination of this novel dataset and an existing dataset from the literature. Project page: https://lvsn.github.io/gaslight/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.22978v3">Pose-free 3D Gaussian splatting via shape-ray estimation</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 ICIP 2025 (Best Student Paper Award) Code available at: https://github.com/youngju-na/SHARE
    </div>
    <details class="paper-abstract">
      While generalizable 3D Gaussian splatting enables efficient, high-quality rendering of unseen scenes, it heavily depends on precise camera poses for accurate geometry. In real-world scenarios, obtaining accurate poses is challenging, leading to noisy pose estimates and geometric misalignments. To address this, we introduce SHARE, a pose-free, feed-forward Gaussian splatting framework that overcomes these ambiguities by joint shape and camera rays estimation. Instead of relying on explicit 3D transformations, SHARE builds a pose-aware canonical volume representation that seamlessly integrates multi-view information, reducing misalignment caused by inaccurate pose estimates. Additionally, anchor-aligned Gaussian prediction enhances scene reconstruction by refining local geometry around coarse anchors, allowing for more precise Gaussian placement. Extensive experiments on diverse real-world datasets show that our method achieves robust performance in pose-free generalizable Gaussian splatting. Code is avilable at https://github.com/youngju-na/SHARE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18739v1">Moving Light Adaptive Colonoscopy Reconstruction via Illumination-Attenuation-Aware 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a pivotal technique for real-time view synthesis in colonoscopy, enabling critical applications such as virtual colonoscopy and lesion tracking. However, the vanilla 3DGS assumes static illumination and that observed appearance depends solely on viewing angle, which causes incompatibility with the photometric variations in colonoscopic scenes induced by dynamic light source/camera. This mismatch forces most 3DGS methods to introduce structure-violating vaporous Gaussian blobs between the camera and tissues to compensate for illumination attenuation, ultimately degrading the quality of 3D reconstructions. Previous works only consider the illumination attenuation caused by light distance, ignoring the physical characters of light source and camera. In this paper, we propose ColIAGS, an improved 3DGS framework tailored for colonoscopy. To mimic realistic appearance under varying illumination, we introduce an Improved Appearance Modeling with two types of illumination attenuation factors, which enables Gaussians to adapt to photometric variations while preserving geometry accuracy. To ensure the geometry approximation condition of appearance modeling, we propose an Improved Geometry Modeling using high-dimensional view embedding to enhance Gaussian geometry attribute prediction. Furthermore, another cosine embedding input is leveraged to generate illumination attenuation solutions in an implicit manner. Comprehensive experimental results on standard benchmarks demonstrate that our proposed ColIAGS achieves the dual capabilities of novel view synthesis and accurate geometric reconstruction. It notably outperforms other state-of-the-art methods by achieving superior rendering fidelity while significantly reducing Depth MSE. Code will be available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19653v1">Re-Activating Frozen Primitives for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D-GS) achieves real-time photorealistic novel view synthesis, yet struggles with complex scenes due to over-reconstruction artifacts, manifesting as local blurring and needle-shape distortions. While recent approaches attribute these issues to insufficient splitting of large-scale Gaussians, we identify two fundamental limitations: gradient magnitude dilution during densification and the primitive frozen phenomenon, where essential Gaussian densification is inhibited in complex regions while suboptimally scaled Gaussians become trapped in local optima. To address these challenges, we introduce ReAct-GS, a method founded on the principle of re-activation. Our approach features: (1) an importance-aware densification criterion incorporating $α$-blending weights from multiple viewpoints to re-activate stalled primitive growth in complex regions, and (2) a re-activation mechanism that revitalizes frozen primitives through adaptive parameter perturbations. Comprehensive experiments across diverse real-world datasets demonstrate that ReAct-GS effectively eliminates over-reconstruction artifacts and achieves state-of-the-art performance on standard novel view synthesis metrics while preserving intricate geometric details. Additionally, our re-activation mechanism yields consistent improvements when integrated with other 3D-GS variants such as Pixel-GS, demonstrating its broad applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10759v2">Every Camera Effect, Every Time, All at Once: 4D Gaussian Ray Tracing for Physics-based Camera Effect Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Paper accepted to NeurIPS 2025 Workshop SpaVLE. Project page: https://shigon255.github.io/4DGRT-project-page/
    </div>
    <details class="paper-abstract">
      Common computer vision systems typically assume ideal pinhole cameras but fail when facing real-world camera effects such as fisheye distortion and rolling shutter, mainly due to the lack of learning from training data with camera effects. Existing data generation approaches suffer from either high costs, sim-to-real gaps or fail to accurately model camera effects. To address this bottleneck, we propose 4D Gaussian Ray Tracing (4D-GRT), a novel two-stage pipeline that combines 4D Gaussian Splatting with physically-based ray tracing for camera effect simulation. Given multi-view videos, 4D-GRT first reconstructs dynamic scenes, then applies ray tracing to generate videos with controllable, physically accurate camera effects. 4D-GRT achieves the fastest rendering speed while performing better or comparable rendering quality compared to existing baselines. Additionally, we construct eight synthetic dynamic scenes in indoor environments across four camera effects as a benchmark to evaluate generated videos with camera effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18489v1">Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Project page is available at https://liujf1226.github.io/Mono4DGS-HDR/
    </div>
    <details class="paper-abstract">
      We introduce Mono4DGS-HDR, the first system for reconstructing renderable 4D high dynamic range (HDR) scenes from unposed monocular low dynamic range (LDR) videos captured with alternating exposures. To tackle such a challenging problem, we present a unified framework with two-stage optimization approach based on Gaussian Splatting. The first stage learns a video HDR Gaussian representation in orthographic camera coordinate space, eliminating the need for camera poses and enabling robust initial HDR video reconstruction. The second stage transforms video Gaussians into world space and jointly refines the world Gaussians with camera poses. Furthermore, we propose a temporal luminance regularization strategy to enhance the temporal consistency of the HDR appearance. Since our task has not been studied before, we construct a new evaluation benchmark using publicly available datasets for HDR video reconstruction. Extensive experiments demonstrate that Mono4DGS-HDR significantly outperforms alternative solutions adapted from state-of-the-art methods in both rendering quality and speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2408.13036v3">H3D-DGS: Exploring Heterogeneous 3D Motion Representation for Deformable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction poses a persistent challenge in 3D vision. Deformable 3D Gaussian Splatting has emerged as an effective method for this task, offering real-time rendering and high visual fidelity. This approach decomposes a dynamic scene into a static representation in a canonical space and time-varying scene motion. Scene motion is defined as the collective movement of all Gaussian points, and for compactness, existing approaches commonly adopt implicit neural fields or sparse control points. However, these methods predominantly rely on gradient-based optimization for all motion information. Due to the high degree of freedom, they struggle to converge on real-world datasets exhibiting complex motion. To preserve the compactness of motion representation and address convergence challenges, this paper proposes heterogeneous 3D control points, termed \textbf{H3D control points}, whose attributes are obtained using a hybrid strategy combining optical flow back-projection and gradient-based methods. This design decouples directly observable motion components from those that are geometrically occluded. Specifically, components of 3D motion that project onto the image plane are directly acquired via optical flow back projection, while unobservable portions are refined through gradient-based optimization. Experiments on the Neu3DV and CMU-Panoptic datasets demonstrate that our method achieves superior performance over state-of-the-art deformable 3D Gaussian splatting techniques. Remarkably, our method converges within just 100 iterations and achieves a per-frame processing speed of 2 seconds on a single NVIDIA RTX 4070 GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18253v1">OpenInsGaussian: Open-vocabulary Instance Gaussian Segmentation with Context-aware Cross-view Fusion</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Understanding 3D scenes is pivotal for autonomous driving, robotics, and augmented reality. Recent semantic Gaussian Splatting approaches leverage large-scale 2D vision models to project 2D semantic features onto 3D scenes. However, they suffer from two major limitations: (1) insufficient contextual cues for individual masks during preprocessing and (2) inconsistencies and missing details when fusing multi-view features from these 2D models. In this paper, we introduce \textbf{OpenInsGaussian}, an \textbf{Open}-vocabulary \textbf{Ins}tance \textbf{Gaussian} segmentation framework with Context-aware Cross-view Fusion. Our method consists of two modules: Context-Aware Feature Extraction, which augments each mask with rich semantic context, and Attention-Driven Feature Aggregation, which selectively fuses multi-view features to mitigate alignment errors and incompleteness. Through extensive experiments on benchmark datasets, OpenInsGaussian achieves state-of-the-art results in open-vocabulary 3D Gaussian segmentation, outperforming existing baselines by a large margin. These findings underscore the robustness and generality of our proposed approach, marking a significant step forward in 3D scene understanding and its practical deployment across diverse real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17783v1">Botany-Bot: Digital Twin Monitoring of Occluded and Underleaf Plant Structures with Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
    </div>
    <details class="paper-abstract">
      Commercial plant phenotyping systems using fixed cameras cannot perceive many plant details due to leaf occlusion. In this paper, we present Botany-Bot, a system for building detailed "annotated digital twins" of living plants using two stereo cameras, a digital turntable inside a lightbox, an industrial robot arm, and 3D segmentated Gaussian Splat models. We also present robot algorithms for manipulating leaves to take high-resolution indexable images of occluded details such as stem buds and the underside/topside of leaves. Results from experiments suggest that Botany-Bot can segment leaves with 90.8% accuracy, detect leaves with 86.2% accuracy, lift/push leaves with 77.9% accuracy, and take detailed overside/underside images with 77.3% accuracy. Code, videos, and datasets are available at https://berkeleyautomation.github.io/Botany-Bot/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17719v1">Raindrop GS: A Benchmark for 3D Gaussian Splatting under Raindrop Conditions</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) under raindrop conditions suffers from severe occlusions and optical distortions caused by raindrop contamination on the camera lens, substantially degrading reconstruction quality. Existing benchmarks typically evaluate 3DGS using synthetic raindrop images with known camera poses (constrained images), assuming ideal conditions. However, in real-world scenarios, raindrops often interfere with accurate camera pose estimation and point cloud initialization. Moreover, a significant domain gap between synthetic and real raindrops further impairs generalization. To tackle these issues, we introduce RaindropGS, a comprehensive benchmark designed to evaluate the full 3DGS pipeline-from unconstrained, raindrop-corrupted images to clear 3DGS reconstructions. Specifically, the whole benchmark pipeline consists of three parts: data preparation, data processing, and raindrop-aware 3DGS evaluation, including types of raindrop interference, camera pose estimation and point cloud initialization, single image rain removal comparison, and 3D Gaussian training comparison. First, we collect a real-world raindrop reconstruction dataset, in which each scene contains three aligned image sets: raindrop-focused, background-focused, and rain-free ground truth, enabling a comprehensive evaluation of reconstruction quality under different focus conditions. Through comprehensive experiments and analyses, we reveal critical insights into the performance limitations of existing 3DGS methods on unconstrained raindrop images and the varying impact of different pipeline components: the impact of camera focus position on 3DGS reconstruction performance, and the interference caused by inaccurate pose and point cloud initialization on reconstruction. These insights establish clear directions for developing more robust 3DGS methods under raindrop conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17479v1">Initialize to Generalize: A Stronger Initialization Pipeline for Sparse-View 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 A preprint paper
    </div>
    <details class="paper-abstract">
      Sparse-view 3D Gaussian Splatting (3DGS) often overfits to the training views, leading to artifacts like blurring in novel view rendering. Prior work addresses it either by enhancing the initialization (\emph{i.e.}, the point cloud from Structure-from-Motion (SfM)) or by adding training-time constraints (regularization) to the 3DGS optimization. Yet our controlled ablations reveal that initialization is the decisive factor: it determines the attainable performance band in sparse-view 3DGS, while training-time constraints yield only modest within-band improvements at extra cost. Given initialization's primacy, we focus our design there. Although SfM performs poorly under sparse views due to its reliance on feature matching, it still provides reliable seed points. Thus, building on SfM, our effort aims to supplement the regions it fails to cover as comprehensively as possible. Specifically, we design: (i) frequency-aware SfM that improves low-texture coverage via low-frequency view augmentation and relaxed multi-view correspondences; (ii) 3DGS self-initialization that lifts photometric supervision into additional points, compensating SfM-sparse regions with learned Gaussian centers; and (iii) point-cloud regularization that enforces multi-view consistency and uniform spatial coverage through simple geometric/visibility priors, yielding a clean and reliable point cloud. Our experiments on LLFF and Mip-NeRF360 demonstrate consistent gains in sparse-view settings, establishing our approach as a stronger initialization strategy. Code is available at https://github.com/zss171999645/ItG-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13639v3">4D Radar-Inertial Odometry based on Gaussian Modeling and Multi-Hypothesis Scan Matching</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Our code and results can be publicly accessed at: https://github.com/robotics-upo/gaussian-rio-cpp
    </div>
    <details class="paper-abstract">
      4D millimeter-wave (mmWave) radars are sensors that provide robustness against adverse weather conditions (rain, snow, fog, etc.), and as such they are increasingly used for odometry and SLAM (Simultaneous Location and Mapping). However, the noisy and sparse nature of the returned scan data proves to be a challenging obstacle for existing registration algorithms, especially those originally intended for more accurate sensors such as LiDAR. Following the success of 3D Gaussian Splatting for vision, in this paper we propose a summarized representation for radar scenes based on global simultaneous optimization of 3D Gaussians as opposed to voxel-based approaches, and leveraging its inherent probability distribution function for registration. Moreover, we propose tackling the problem of radar noise by optimizing multiple scan matching hypotheses in order to further increase the robustness of the system against local optima of the function. Finally, following existing practice we implement an Extended Kalman Filter-based Radar-Inertial Odometry pipeline in order to evaluate the effectiveness of our system. Experiments using publicly available 4D radar datasets show that our Gaussian approach is comparable to existing registration algorithms, outperforming them in several sequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17095v1">GSPlane: Concise and Accurate Planar Reconstruction via Structured Representation</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Planes are fundamental primitives of 3D sences, especially in man-made environments such as indoor spaces and urban streets. Representing these planes in a structured and parameterized format facilitates scene editing and physical simulations in downstream applications. Recently, Gaussian Splatting (GS) has demonstrated remarkable effectiveness in the Novel View Synthesis task, with extensions showing great potential in accurate surface reconstruction. However, even state-of-the-art GS representations often struggle to reconstruct planar regions with sufficient smoothness and precision. To address this issue, we propose GSPlane, which recovers accurate geometry and produces clean and well-structured mesh connectivity for plane regions in the reconstructed scene. By leveraging off-the-shelf segmentation and normal prediction models, GSPlane extracts robust planar priors to establish structured representations for planar Gaussian coordinates, which help guide the training process by enforcing geometric consistency. To further enhance training robustness, a Dynamic Gaussian Re-classifier is introduced to adaptively reclassify planar Gaussians with persistently high gradients as non-planar, ensuring more reliable optimization. Furthermore, we utilize the optimized planar priors to refine the mesh layouts, significantly improving topological structure while reducing the number of vertices and faces. We also explore applications of the structured planar representation, which enable decoupling and flexible manipulation of objects on supportive planes. Extensive experiments demonstrate that, with no sacrifice in rendering quality, the introduction of planar priors significantly improves the geometric accuracy of the extracted meshes across various baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18101v1">From Volume Rendering to 3D Gaussian Splatting: Theory and Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Accepted at the Conference on Graphics, Patterns and Images (SIBGRAPI), math focused, 5 equations, 5 Figure, 5 pages of text and 1 of bibligraphy
    </div>
    <details class="paper-abstract">
      The problem of 3D reconstruction from posed images is undergoing a fundamental transformation, driven by continuous advances in 3D Gaussian Splatting (3DGS). By modeling scenes explicitly as collections of 3D Gaussians, 3DGS enables efficient rasterization through volumetric splatting, offering thus a seamless integration with common graphics pipelines. Despite its real-time rendering capabilities for novel view synthesis, 3DGS suffers from a high memory footprint, the tendency to bake lighting effects directly into its representation, and limited support for secondary-ray effects. This tutorial provides a concise yet comprehensive overview of the 3DGS pipeline, starting from its splatting formulation and then exploring the main efforts in addressing its limitations. Finally, we survey a range of applications that leverage 3DGS for surface reconstruction, avatar modeling, animation, and content generation-highlighting its efficient rendering and suitability for feed-forward pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18054v1">HouseTour: A Virtual Real Estate A(I)gent</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Published on ICCV 2025
    </div>
    <details class="paper-abstract">
      We introduce HouseTour, a method for spatially-aware 3D camera trajectory and natural language summary generation from a collection of images depicting an existing 3D space. Unlike existing vision-language models (VLMs), which struggle with geometric reasoning, our approach generates smooth video trajectories via a diffusion process constrained by known camera poses and integrates this information into the VLM for 3D-grounded descriptions. We synthesize the final video using 3D Gaussian splatting to render novel views along the trajectory. To support this task, we present the HouseTour dataset, which includes over 1,200 house-tour videos with camera poses, 3D reconstructions, and real estate descriptions. Experiments demonstrate that incorporating 3D camera trajectories into the text generation process improves performance over methods handling each task independently. We evaluate both individual and end-to-end performance, introducing a new joint metric. Our work enables automated, professional-quality video creation for real estate and touristic applications without requiring specialized expertise or equipment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18101v1">From Volume Rendering to 3D Gaussian Splatting: Theory and Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Accepted at the Conference on Graphics, Patterns and Images (SIBGRAPI), math focused, 5 equations, 5 Figure, 5 pages of text and 1 of bibligraphy
    </div>
    <details class="paper-abstract">
      The problem of 3D reconstruction from posed images is undergoing a fundamental transformation, driven by continuous advances in 3D Gaussian Splatting (3DGS). By modeling scenes explicitly as collections of 3D Gaussians, 3DGS enables efficient rasterization through volumetric splatting, offering thus a seamless integration with common graphics pipelines. Despite its real-time rendering capabilities for novel view synthesis, 3DGS suffers from a high memory footprint, the tendency to bake lighting effects directly into its representation, and limited support for secondary-ray effects. This tutorial provides a concise yet comprehensive overview of the 3DGS pipeline, starting from its splatting formulation and then exploring the main efforts in addressing its limitations. Finally, we survey a range of applications that leverage 3DGS for surface reconstruction, avatar modeling, animation, and content generation-highlighting its efficient rendering and suitability for feed-forward pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18054v1">HouseTour: A Virtual Real Estate A(I)gent</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Published on ICCV 2025
    </div>
    <details class="paper-abstract">
      We introduce HouseTour, a method for spatially-aware 3D camera trajectory and natural language summary generation from a collection of images depicting an existing 3D space. Unlike existing vision-language models (VLMs), which struggle with geometric reasoning, our approach generates smooth video trajectories via a diffusion process constrained by known camera poses and integrates this information into the VLM for 3D-grounded descriptions. We synthesize the final video using 3D Gaussian splatting to render novel views along the trajectory. To support this task, we present the HouseTour dataset, which includes over 1,200 house-tour videos with camera poses, 3D reconstructions, and real estate descriptions. Experiments demonstrate that incorporating 3D camera trajectories into the text generation process improves performance over methods handling each task independently. We evaluate both individual and end-to-end performance, introducing a new joint metric. Our work enables automated, professional-quality video creation for real estate and touristic applications without requiring specialized expertise or equipment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17783v1">Botany-Bot: Digital Twin Monitoring of Occluded and Underleaf Plant Structures with Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
    </div>
    <details class="paper-abstract">
      Commercial plant phenotyping systems using fixed cameras cannot perceive many plant details due to leaf occlusion. In this paper, we present Botany-Bot, a system for building detailed "annotated digital twins" of living plants using two stereo cameras, a digital turntable inside a lightbox, an industrial robot arm, and 3D segmentated Gaussian Splat models. We also present robot algorithms for manipulating leaves to take high-resolution indexable images of occluded details such as stem buds and the underside/topside of leaves. Results from experiments suggest that Botany-Bot can segment leaves with 90.8% accuracy, detect leaves with 86.2% accuracy, lift/push leaves with 77.9% accuracy, and take detailed overside/underside images with 77.3% accuracy. Code, videos, and datasets are available at https://berkeleyautomation.github.io/Botany-Bot/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17719v1">Raindrop GS: A Benchmark for 3D Gaussian Splatting under Raindrop Conditions</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) under raindrop conditions suffers from severe occlusions and optical distortions caused by raindrop contamination on the camera lens, substantially degrading reconstruction quality. Existing benchmarks typically evaluate 3DGS using synthetic raindrop images with known camera poses (constrained images), assuming ideal conditions. However, in real-world scenarios, raindrops often interfere with accurate camera pose estimation and point cloud initialization. Moreover, a significant domain gap between synthetic and real raindrops further impairs generalization. To tackle these issues, we introduce RaindropGS, a comprehensive benchmark designed to evaluate the full 3DGS pipeline-from unconstrained, raindrop-corrupted images to clear 3DGS reconstructions. Specifically, the whole benchmark pipeline consists of three parts: data preparation, data processing, and raindrop-aware 3DGS evaluation, including types of raindrop interference, camera pose estimation and point cloud initialization, single image rain removal comparison, and 3D Gaussian training comparison. First, we collect a real-world raindrop reconstruction dataset, in which each scene contains three aligned image sets: raindrop-focused, background-focused, and rain-free ground truth, enabling a comprehensive evaluation of reconstruction quality under different focus conditions. Through comprehensive experiments and analyses, we reveal critical insights into the performance limitations of existing 3DGS methods on unconstrained raindrop images and the varying impact of different pipeline components: the impact of camera focus position on 3DGS reconstruction performance, and the interference caused by inaccurate pose and point cloud initialization on reconstruction. These insights establish clear directions for developing more robust 3DGS methods under raindrop conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17479v1">Initialize to Generalize: A Stronger Initialization Pipeline for Sparse-View 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 A preprint paper
    </div>
    <details class="paper-abstract">
      Sparse-view 3D Gaussian Splatting (3DGS) often overfits to the training views, leading to artifacts like blurring in novel view rendering. Prior work addresses it either by enhancing the initialization (\emph{i.e.}, the point cloud from Structure-from-Motion (SfM)) or by adding training-time constraints (regularization) to the 3DGS optimization. Yet our controlled ablations reveal that initialization is the decisive factor: it determines the attainable performance band in sparse-view 3DGS, while training-time constraints yield only modest within-band improvements at extra cost. Given initialization's primacy, we focus our design there. Although SfM performs poorly under sparse views due to its reliance on feature matching, it still provides reliable seed points. Thus, building on SfM, our effort aims to supplement the regions it fails to cover as comprehensively as possible. Specifically, we design: (i) frequency-aware SfM that improves low-texture coverage via low-frequency view augmentation and relaxed multi-view correspondences; (ii) 3DGS self-initialization that lifts photometric supervision into additional points, compensating SfM-sparse regions with learned Gaussian centers; and (iii) point-cloud regularization that enforces multi-view consistency and uniform spatial coverage through simple geometric/visibility priors, yielding a clean and reliable point cloud. Our experiments on LLFF and Mip-NeRF360 demonstrate consistent gains in sparse-view settings, establishing our approach as a stronger initialization strategy. Code is available at https://github.com/zss171999645/ItG-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17095v1">GSPlane: Concise and Accurate Planar Reconstruction via Structured Representation</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Planes are fundamental primitives of 3D sences, especially in man-made environments such as indoor spaces and urban streets. Representing these planes in a structured and parameterized format facilitates scene editing and physical simulations in downstream applications. Recently, Gaussian Splatting (GS) has demonstrated remarkable effectiveness in the Novel View Synthesis task, with extensions showing great potential in accurate surface reconstruction. However, even state-of-the-art GS representations often struggle to reconstruct planar regions with sufficient smoothness and precision. To address this issue, we propose GSPlane, which recovers accurate geometry and produces clean and well-structured mesh connectivity for plane regions in the reconstructed scene. By leveraging off-the-shelf segmentation and normal prediction models, GSPlane extracts robust planar priors to establish structured representations for planar Gaussian coordinates, which help guide the training process by enforcing geometric consistency. To further enhance training robustness, a Dynamic Gaussian Re-classifier is introduced to adaptively reclassify planar Gaussians with persistently high gradients as non-planar, ensuring more reliable optimization. Furthermore, we utilize the optimized planar priors to refine the mesh layouts, significantly improving topological structure while reducing the number of vertices and faces. We also explore applications of the structured planar representation, which enable decoupling and flexible manipulation of objects on supportive planes. Extensive experiments demonstrate that, with no sacrifice in rendering quality, the introduction of planar priors significantly improves the geometric accuracy of the extracted meshes across various baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16837v1">2DGS-R: Revisiting the Normal Consistency Regularization in 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3DGS) have greatly influenced neural fields, as it enables high-fidelity rendering with impressive visual quality. However, 3DGS has difficulty accurately representing surfaces. In contrast, 2DGS transforms the 3D volume into a collection of 2D planar Gaussian disks. Despite advancements in geometric fidelity, rendering quality remains compromised, highlighting the challenge of achieving both high-quality rendering and precise geometric structures. This indicates that optimizing both geometric and rendering quality in a single training stage is currently unfeasible. To overcome this limitation, we present 2DGS-R, a new method that uses a hierarchical training approach to improve rendering quality while maintaining geometric accuracy. 2DGS-R first trains the original 2D Gaussians with the normal consistency regularization. Then 2DGS-R selects the 2D Gaussians with inadequate rendering quality and applies a novel in-place cloning operation to enhance the 2D Gaussians. Finally, we fine-tune the 2DGS-R model with opacity frozen. Experimental results show that compared to the original 2DGS, our method requires only 1\% more storage and minimal additional training time. Despite this negligible overhead, it achieves high-quality rendering results while preserving fine geometric structures. These findings indicate that our approach effectively balances efficiency with performance, leading to improvements in both visual fidelity and geometric reconstruction accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16777v1">GS2POSE: Marry Gaussian Splatting to 6D Object Pose Estimation</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Accurate 6D pose estimation of 3D objects is a fundamental task in computer vision, and current research typically predicts the 6D pose by establishing correspondences between 2D image features and 3D model features. However, these methods often face difficulties with textureless objects and varying illumination conditions. To overcome these limitations, we propose GS2POSE, a novel approach for 6D object pose estimation. GS2POSE formulates a pose regression algorithm inspired by the principles of Bundle Adjustment (BA). By leveraging Lie algebra, we extend the capabilities of 3DGS to develop a pose-differentiable rendering pipeline, which iteratively optimizes the pose by comparing the input image to the rendered image. Additionally, GS2POSE updates color parameters within the 3DGS model, enhancing its adaptability to changes in illumination. Compared to previous models, GS2POSE demonstrates accuracy improvements of 1.4\%, 2.8\% and 2.5\% on the T-LESS, LineMod-Occlusion and LineMod datasets, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11387v2">MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Accepted by NeurIPS 2025. Project Page: https://wen-yuan-zhang.github.io/MaterialRefGS
    </div>
    <details class="paper-abstract">
      Modeling reflections from 2D images is essential for photorealistic rendering and novel view synthesis. Recent approaches enhance Gaussian primitives with reflection-related material attributes to enable physically based rendering (PBR) with Gaussian Splatting. However, the material inference often lacks sufficient constraints, especially under limited environment modeling, resulting in illumination aliasing and reduced generalization. In this work, we revisit the problem from a multi-view perspective and show that multi-view consistent material inference with more physically-based environment modeling is key to learning accurate reflections with Gaussian Splatting. To this end, we enforce 2D Gaussians to produce multi-view consistent material maps during deferred shading. We also track photometric variations across views to identify highly reflective regions, which serve as strong priors for reflection strength terms. To handle indirect illumination caused by inter-object occlusions, we further introduce an environment modeling strategy through ray tracing with 2DGS, enabling photorealistic rendering of indirect radiance. Experiments on widely used benchmarks show that our method faithfully recovers both illumination and geometry, achieving state-of-the-art rendering quality in novel views synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16837v1">2DGS-R: Revisiting the Normal Consistency Regularization in 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3DGS) have greatly influenced neural fields, as it enables high-fidelity rendering with impressive visual quality. However, 3DGS has difficulty accurately representing surfaces. In contrast, 2DGS transforms the 3D volume into a collection of 2D planar Gaussian disks. Despite advancements in geometric fidelity, rendering quality remains compromised, highlighting the challenge of achieving both high-quality rendering and precise geometric structures. This indicates that optimizing both geometric and rendering quality in a single training stage is currently unfeasible. To overcome this limitation, we present 2DGS-R, a new method that uses a hierarchical training approach to improve rendering quality while maintaining geometric accuracy. 2DGS-R first trains the original 2D Gaussians with the normal consistency regularization. Then 2DGS-R selects the 2D Gaussians with inadequate rendering quality and applies a novel in-place cloning operation to enhance the 2D Gaussians. Finally, we fine-tune the 2DGS-R model with opacity frozen. Experimental results show that compared to the original 2DGS, our method requires only 1\% more storage and minimal additional training time. Despite this negligible overhead, it achieves high-quality rendering results while preserving fine geometric structures. These findings indicate that our approach effectively balances efficiency with performance, leading to improvements in both visual fidelity and geometric reconstruction accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16777v1">GS2POSE: Marry Gaussian Splatting to 6D Object Pose Estimation</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Accurate 6D pose estimation of 3D objects is a fundamental task in computer vision, and current research typically predicts the 6D pose by establishing correspondences between 2D image features and 3D model features. However, these methods often face difficulties with textureless objects and varying illumination conditions. To overcome these limitations, we propose GS2POSE, a novel approach for 6D object pose estimation. GS2POSE formulates a pose regression algorithm inspired by the principles of Bundle Adjustment (BA). By leveraging Lie algebra, we extend the capabilities of 3DGS to develop a pose-differentiable rendering pipeline, which iteratively optimizes the pose by comparing the input image to the rendered image. Additionally, GS2POSE updates color parameters within the 3DGS model, enhancing its adaptability to changes in illumination. Compared to previous models, GS2POSE demonstrates accuracy improvements of 1.4\%, 2.8\% and 2.5\% on the T-LESS, LineMod-Occlusion and LineMod datasets, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11387v2">MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Accepted by NeurIPS 2025. Project Page: https://wen-yuan-zhang.github.io/MaterialRefGS
    </div>
    <details class="paper-abstract">
      Modeling reflections from 2D images is essential for photorealistic rendering and novel view synthesis. Recent approaches enhance Gaussian primitives with reflection-related material attributes to enable physically based rendering (PBR) with Gaussian Splatting. However, the material inference often lacks sufficient constraints, especially under limited environment modeling, resulting in illumination aliasing and reduced generalization. In this work, we revisit the problem from a multi-view perspective and show that multi-view consistent material inference with more physically-based environment modeling is key to learning accurate reflections with Gaussian Splatting. To this end, we enforce 2D Gaussians to produce multi-view consistent material maps during deferred shading. We also track photometric variations across views to identify highly reflective regions, which serve as strong priors for reflection strength terms. To handle indirect illumination caused by inter-object occlusions, we further introduce an environment modeling strategy through ray tracing with 2DGS, enabling photorealistic rendering of indirect radiance. Experiments on widely used benchmarks show that our method faithfully recovers both illumination and geometry, achieving state-of-the-art rendering quality in novel views synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16463v1">HGC-Avatar: Hierarchical Gaussian Compression for Streamable Dynamic 3D Avatars</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 ACM International Conference on Multimedia 2025
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled fast, photorealistic rendering of dynamic 3D scenes, showing strong potential in immersive communication. However, in digital human encoding and transmission, the compression methods based on general 3DGS representations are limited by the lack of human priors, resulting in suboptimal bitrate efficiency and reconstruction quality at the decoder side, which hinders their application in streamable 3D avatar systems. We propose HGC-Avatar, a novel Hierarchical Gaussian Compression framework designed for efficient transmission and high-quality rendering of dynamic avatars. Our method disentangles the Gaussian representation into a structural layer, which maps poses to Gaussians via a StyleUNet-based generator, and a motion layer, which leverages the SMPL-X model to represent temporal pose variations compactly and semantically. This hierarchical design supports layer-wise compression, progressive decoding, and controllable rendering from diverse pose inputs such as video sequences or text. Since people are most concerned with facial realism, we incorporate a facial attention mechanism during StyleUNet training to preserve identity and expression details under low-bitrate constraints. Experimental results demonstrate that HGC-Avatar provides a streamable solution for rapid 3D avatar rendering, while significantly outperforming prior methods in both visual quality and compression efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16410v1">REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Bridging the gap between complex human instructions and precise 3D object grounding remains a significant challenge in vision and robotics. Existing 3D segmentation methods often struggle to interpret ambiguous, reasoning-based instructions, while 2D vision-language models that excel at such reasoning lack intrinsic 3D spatial understanding. In this paper, we introduce REALM, an innovative MLLM-agent framework that enables open-world reasoning-based segmentation without requiring extensive 3D-specific post-training. We perform segmentation directly on 3D Gaussian Splatting representations, capitalizing on their ability to render photorealistic novel views that are highly suitable for MLLM comprehension. As directly feeding one or more rendered views to the MLLM can lead to high sensitivity to viewpoint selection, we propose a novel Global-to-Local Spatial Grounding strategy. Specifically, multiple global views are first fed into the MLLM agent in parallel for coarse-level localization, aggregating responses to robustly identify the target object. Then, several close-up novel views of the object are synthesized to perform fine-grained local segmentation, yielding accurate and consistent 3D masks. Extensive experiments show that REALM achieves remarkable performance in interpreting both explicit and implicit instructions across LERF, 3D-OVS, and our newly introduced REALM3D benchmarks. Furthermore, our agent framework seamlessly supports a range of 3D interaction tasks, including object removal, replacement, and style transfer, demonstrating its practical utility and versatility. Project page: https://ChangyueShi.github.io/REALM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21152v3">Geometry and Perception Guided Gaussians for Multiview-consistent 3D Generation from a Single Image</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Generating realistic 3D objects from single-view images requires natural appearance, 3D consistency, and the ability to capture multiple plausible interpretations of unseen regions. Existing approaches often rely on fine-tuning pretrained 2D diffusion models or directly generating 3D information through fast network inference or 3D Gaussian Splatting, but their results generally suffer from poor multiview consistency and lack geometric detail. To tackle these issues, we present a novel method that seamlessly integrates geometry and perception information without requiring additional model training to reconstruct detailed 3D objects from a single image. Specifically, we incorporate geometry and perception priors to initialize the Gaussian branches and guide their parameter optimization. The geometry prior captures the rough 3D shapes, while the perception prior utilizes the 2D pretrained diffusion model to enhance multiview information. Subsequently, we introduce a stable Score Distillation Sampling for fine-grained prior distillation to ensure effective knowledge transfer. The model is further enhanced by a reprojection-based strategy that enforces depth consistency. Experimental results show that we outperform existing methods on novel view synthesis and 3D reconstruction, demonstrating robust and consistent 3D object generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16463v1">HGC-Avatar: Hierarchical Gaussian Compression for Streamable Dynamic 3D Avatars</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 ACM International Conference on Multimedia 2025
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled fast, photorealistic rendering of dynamic 3D scenes, showing strong potential in immersive communication. However, in digital human encoding and transmission, the compression methods based on general 3DGS representations are limited by the lack of human priors, resulting in suboptimal bitrate efficiency and reconstruction quality at the decoder side, which hinders their application in streamable 3D avatar systems. We propose HGC-Avatar, a novel Hierarchical Gaussian Compression framework designed for efficient transmission and high-quality rendering of dynamic avatars. Our method disentangles the Gaussian representation into a structural layer, which maps poses to Gaussians via a StyleUNet-based generator, and a motion layer, which leverages the SMPL-X model to represent temporal pose variations compactly and semantically. This hierarchical design supports layer-wise compression, progressive decoding, and controllable rendering from diverse pose inputs such as video sequences or text. Since people are most concerned with facial realism, we incorporate a facial attention mechanism during StyleUNet training to preserve identity and expression details under low-bitrate constraints. Experimental results demonstrate that HGC-Avatar provides a streamable solution for rapid 3D avatar rendering, while significantly outperforming prior methods in both visual quality and compression efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16410v1">REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Bridging the gap between complex human instructions and precise 3D object grounding remains a significant challenge in vision and robotics. Existing 3D segmentation methods often struggle to interpret ambiguous, reasoning-based instructions, while 2D vision-language models that excel at such reasoning lack intrinsic 3D spatial understanding. In this paper, we introduce REALM, an innovative MLLM-agent framework that enables open-world reasoning-based segmentation without requiring extensive 3D-specific post-training. We perform segmentation directly on 3D Gaussian Splatting representations, capitalizing on their ability to render photorealistic novel views that are highly suitable for MLLM comprehension. As directly feeding one or more rendered views to the MLLM can lead to high sensitivity to viewpoint selection, we propose a novel Global-to-Local Spatial Grounding strategy. Specifically, multiple global views are first fed into the MLLM agent in parallel for coarse-level localization, aggregating responses to robustly identify the target object. Then, several close-up novel views of the object are synthesized to perform fine-grained local segmentation, yielding accurate and consistent 3D masks. Extensive experiments show that REALM achieves remarkable performance in interpreting both explicit and implicit instructions across LERF, 3D-OVS, and our newly introduced REALM3D benchmarks. Furthermore, our agent framework seamlessly supports a range of 3D interaction tasks, including object removal, replacement, and style transfer, demonstrating its practical utility and versatility. Project page: https://ChangyueShi.github.io/REALM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.21152v3">Geometry and Perception Guided Gaussians for Multiview-consistent 3D Generation from a Single Image</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Generating realistic 3D objects from single-view images requires natural appearance, 3D consistency, and the ability to capture multiple plausible interpretations of unseen regions. Existing approaches often rely on fine-tuning pretrained 2D diffusion models or directly generating 3D information through fast network inference or 3D Gaussian Splatting, but their results generally suffer from poor multiview consistency and lack geometric detail. To tackle these issues, we present a novel method that seamlessly integrates geometry and perception information without requiring additional model training to reconstruct detailed 3D objects from a single image. Specifically, we incorporate geometry and perception priors to initialize the Gaussian branches and guide their parameter optimization. The geometry prior captures the rough 3D shapes, while the perception prior utilizes the 2D pretrained diffusion model to enhance multiview information. Subsequently, we introduce a stable Score Distillation Sampling for fine-grained prior distillation to ensure effective knowledge transfer. The model is further enhanced by a reprojection-based strategy that enforces depth consistency. Experimental results show that we outperform existing methods on novel view synthesis and 3D reconstruction, demonstrating robust and consistent 3D object generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02493v3">Low-Frequency First: Eliminating Floating Artifacts in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Our paper has been accepted by the 24th International Conference on Cyberworlds and recieved the Best Paper Honorable Mention
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a powerful and computationally efficient representation for 3D reconstruction. Despite its strengths, 3DGS often produces floating artifacts, which are erroneous structures detached from the actual geometry and significantly degrade visual fidelity. The underlying mechanisms causing these artifacts, particularly in low-quality initialization scenarios, have not been fully explored. In this paper, we investigate the origins of floating artifacts from a frequency-domain perspective and identify under-optimized Gaussians as the primary source. Based on our analysis, we propose \textit{Eliminating-Floating-Artifacts} Gaussian Splatting (EFA-GS), which selectively expands under-optimized Gaussians to prioritize accurate low-frequency learning. Additionally, we introduce complementary depth-based and scale-based strategies to dynamically refine Gaussian expansion, effectively mitigating detail erosion. Extensive experiments on both synthetic and real-world datasets demonstrate that EFA-GS substantially reduces floating artifacts while preserving high-frequency details, achieving an improvement of 1.68 dB in PSNR over baseline method on our RWLQ dataset. Furthermore, we validate the effectiveness of our approach in downstream 3D editing tasks. Project Website: https://jcwang-gh.github.io/EFA-GS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14081v2">Capture, Canonicalize, Splat: Zero-Shot 3D Gaussian Avatars from Unstructured Phone Images</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      We present a novel, zero-shot pipeline for creating hyperrealistic, identity-preserving 3D avatars from a few unstructured phone images. Existing methods face several challenges: single-view approaches suffer from geometric inconsistencies and hallucinations, degrading identity preservation, while models trained on synthetic data fail to capture high-frequency details like skin wrinkles and fine hair, limiting realism. Our method introduces two key contributions: (1) a generative canonicalization module that processes multiple unstructured views into a standardized, consistent representation, and (2) a transformer-based model trained on a new, large-scale dataset of high-fidelity Gaussian splatting avatars derived from dome captures of real people. This "Capture, Canonicalize, Splat" pipeline produces static quarter-body avatars with compelling realism and robust identity preservation from unstructured photos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15386v1">PFGS: Pose-Fused 3D Gaussian Splatting for Complete Multi-Pose Object Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled high-quality, real-time novel-view synthesis from multi-view images. However, most existing methods assume the object is captured in a single, static pose, resulting in incomplete reconstructions that miss occluded or self-occluded regions. We introduce PFGS, a pose-aware 3DGS framework that addresses the practical challenge of reconstructing complete objects from multi-pose image captures. Given images of an object in one main pose and several auxiliary poses, PFGS iteratively fuses each auxiliary set into a unified 3DGS representation of the main pose. Our pose-aware fusion strategy combines global and local registration to merge views effectively and refine the 3DGS model. While recent advances in 3D foundation models have improved registration robustness and efficiency, they remain limited by high memory demands and suboptimal accuracy. PFGS overcomes these challenges by incorporating them more intelligently into the registration process: it leverages background features for per-pose camera pose estimation and employs foundation models for cross-pose registration. This design captures the best of both approaches while resolving background inconsistency issues. Experimental results demonstrate that PFGS consistently outperforms strong baselines in both qualitative and quantitative evaluations, producing more complete reconstructions and higher-fidelity 3DGS models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15352v1">GaussGym: An open-source real-to-sim framework for learning locomotion from pixels</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      We present a novel approach for photorealistic robot simulation that integrates 3D Gaussian Splatting as a drop-in renderer within vectorized physics simulators such as IsaacGym. This enables unprecedented speed -- exceeding 100,000 steps per second on consumer GPUs -- while maintaining high visual fidelity, which we showcase across diverse tasks. We additionally demonstrate its applicability in a sim-to-real robotics setting. Beyond depth-based sensing, our results highlight how rich visual semantics improve navigation and decision-making, such as avoiding undesirable regions. We further showcase the ease of incorporating thousands of environments from iPhone scans, large-scale scene datasets (e.g., GrandTour, ARKit), and outputs from generative video models like Veo, enabling rapid creation of realistic training worlds. This work bridges high-throughput simulation and high-fidelity perception, advancing scalable and generalizable robot learning. All code and data will be open-sourced for the community to build upon. Videos, code, and data available at https://escontrela.me/gauss_gym/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21779v2">X$^{2}$-Gaussian: 4D Radiative Gaussian Splatting for Continuous-time Tomographic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Project Page: https://x2-gaussian.github.io/
    </div>
    <details class="paper-abstract">
      Four-dimensional computed tomography (4D CT) reconstruction is crucial for capturing dynamic anatomical changes but faces inherent limitations from conventional phase-binning workflows. Current methods discretize temporal resolution into fixed phases with respiratory gating devices, introducing motion misalignment and restricting clinical practicality. In this paper, We propose X$^2$-Gaussian, a novel framework that enables continuous-time 4D-CT reconstruction by integrating dynamic radiative Gaussian splatting with self-supervised respiratory motion learning. Our approach models anatomical dynamics through a spatiotemporal encoder-decoder architecture that predicts time-varying Gaussian deformations, eliminating phase discretization. To remove dependency on external gating devices, we introduce a physiology-driven periodic consistency loss that learns patient-specific breathing cycles directly from projections via differentiable optimization. Extensive experiments demonstrate state-of-the-art performance, achieving a 9.93 dB PSNR gain over traditional methods and 2.25 dB improvement against prior Gaussian splatting techniques. By unifying continuous motion modeling with hardware-free period learning, X$^2$-Gaussian advances high-fidelity 4D CT reconstruction for dynamic clinical imaging. Code is publicly available at: https://x2-gaussian.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12493v2">BSGS: Bi-stage 3D Gaussian Splatting for Camera Motion Deblurring</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Accept by ACM MM 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has exhibited remarkable capabilities in 3D scene reconstruction. However, reconstructing high-quality 3D scenes from motion-blurred images caused by camera motion poses a significant challenge.The performance of existing 3DGS-based deblurring methods are limited due to their inherent mechanisms, such as extreme dependence on the accuracy of camera poses and inability to effectively control erroneous Gaussian primitives densification caused by motion blur. To solve these problems, we introduce a novel framework, Bi-Stage 3D Gaussian Splatting, to accurately reconstruct 3D scenes from motion-blurred images. BSGS contains two stages. First, Camera Pose Refinement roughly optimizes camera poses to reduce motion-induced distortions. Second, with fixed rough camera poses, Global RigidTransformation further corrects motion-induced blur distortions. To alleviate multi-subframe gradient conflicts, we propose a subframe gradient aggregation strategy to optimize both stages. Furthermore, a space-time bi-stage optimization strategy is introduced to dynamically adjust primitive densification thresholds and prevent premature noisy Gaussian generation in blurred regions. Comprehensive experiments verify the effectiveness of our proposed deblurring method and show its superiority over the state of the arts.Our source code is available at https://github.com/wsxujm/bsgs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23277v2">iLRM: An Iterative Large 3D Reconstruction Model</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Project page: https://gynjn.github.io/iLRM/
    </div>
    <details class="paper-abstract">
      Feed-forward 3D modeling has emerged as a promising approach for rapid and high-quality 3D reconstruction. In particular, directly generating explicit 3D representations, such as 3D Gaussian splatting, has attracted significant attention due to its fast and high-quality rendering, as well as numerous applications. However, many state-of-the-art methods, primarily based on transformer architectures, suffer from severe scalability issues because they rely on full attention across image tokens from multiple input views, resulting in prohibitive computational costs as the number of views or image resolution increases. Toward a scalable and efficient feed-forward 3D reconstruction, we introduce an iterative Large 3D Reconstruction Model (iLRM) that generates 3D Gaussian representations through an iterative refinement mechanism, guided by three core principles: (1) decoupling the scene representation from input-view images to enable compact 3D representations; (2) decomposing fully-attentional multi-view interactions into a two-stage attention scheme to reduce computational costs; and (3) injecting high-resolution information at every layer to achieve high-fidelity reconstruction. Experimental results on widely used datasets, such as RE10K and DL3DV, demonstrate that iLRM outperforms existing methods in both reconstruction quality and speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16272v1">Proactive Scene Decomposition and Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Human behaviors are the major causes of scene dynamics and inherently contain rich cues regarding the dynamics. This paper formalizes a new task of proactive scene decomposition and reconstruction, an online approach that leverages human-object interactions to iteratively disassemble and reconstruct the environment. By observing these intentional interactions, we can dynamically refine the decomposition and reconstruction process, addressing inherent ambiguities in static object-level reconstruction. The proposed system effectively integrates multiple tasks in dynamic environments such as accurate camera and object pose estimation, instance decomposition, and online map updating, capitalizing on cues from human-object interactions in egocentric live streams for a flexible, progressive alternative to conventional object-level reconstruction methods. Aided by the Gaussian splatting technique, accurate and consistent dynamic scene modeling is achieved with photorealistic and efficient rendering. The efficacy is validated in multiple real-world scenarios with promising advantages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10809v4">GaSLight: Gaussian Splats for Spatially-Varying Lighting in HDR</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      We present GaSLight, a method that generates spatially-varying lighting from regular images. Our method proposes using HDR Gaussian Splats as light source representation, marking the first time regular images can serve as light sources in a 3D renderer. Our two-stage process first enhances the dynamic range of images plausibly and accurately by leveraging the priors embedded in diffusion models. Next, we employ Gaussian Splats to model 3D lighting, achieving spatially variant lighting. Our approach yields state-of-the-art results on HDR estimations and their applications in illuminating virtual objects and scenes. To facilitate the benchmarking of images as light sources, we introduce a novel dataset of calibrated and unsaturated HDR to evaluate images as light sources. We assess our method using a combination of this novel dataset and an existing dataset from the literature. Project page: https://lvsn.github.io/gaslight/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2506.21117v2">CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 ICCV 2025, Project Page: https://cl-splats.github.io
    </div>
    <details class="paper-abstract">
      In dynamic 3D environments, accurately updating scene representations over time is crucial for applications in robotics, mixed reality, and embodied AI. As scenes evolve, efficient methods to incorporate changes are needed to maintain up-to-date, high-quality reconstructions without the computational overhead of re-optimizing the entire scene. This paper introduces CL-Splats, which incrementally updates Gaussian splatting-based 3D representations from sparse scene captures. CL-Splats integrates a robust change-detection module that segments updated and static components within the scene, enabling focused, local optimization that avoids unnecessary re-computation. Moreover, CL-Splats supports storing and recovering previous scene states, facilitating temporal segmentation and new scene-analysis applications. Our extensive experiments demonstrate that CL-Splats achieves efficient updates with improved reconstruction quality over the state-of-the-art. This establishes a robust foundation for future real-time adaptation in 3D scene reconstruction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.10809v4">GaSLight: Gaussian Splats for Spatially-Varying Lighting in HDR</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      We present GaSLight, a method that generates spatially-varying lighting from regular images. Our method proposes using HDR Gaussian Splats as light source representation, marking the first time regular images can serve as light sources in a 3D renderer. Our two-stage process first enhances the dynamic range of images plausibly and accurately by leveraging the priors embedded in diffusion models. Next, we employ Gaussian Splats to model 3D lighting, achieving spatially variant lighting. Our approach yields state-of-the-art results on HDR estimations and their applications in illuminating virtual objects and scenes. To facilitate the benchmarking of images as light sources, we introduce a novel dataset of calibrated and unsaturated HDR to evaluate images as light sources. We assess our method using a combination of this novel dataset and an existing dataset from the literature. Project page: https://lvsn.github.io/gaslight/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16272v1">Proactive Scene Decomposition and Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Human behaviors are the major causes of scene dynamics and inherently contain rich cues regarding the dynamics. This paper formalizes a new task of proactive scene decomposition and reconstruction, an online approach that leverages human-object interactions to iteratively disassemble and reconstruct the environment. By observing these intentional interactions, we can dynamically refine the decomposition and reconstruction process, addressing inherent ambiguities in static object-level reconstruction. The proposed system effectively integrates multiple tasks in dynamic environments such as accurate camera and object pose estimation, instance decomposition, and online map updating, capitalizing on cues from human-object interactions in egocentric live streams for a flexible, progressive alternative to conventional object-level reconstruction methods. Aided by the Gaussian splatting technique, accurate and consistent dynamic scene modeling is achieved with photorealistic and efficient rendering. The efficacy is validated in multiple real-world scenarios with promising advantages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15736v1">Fix False Transparency by Noise Guided Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Opaque objects reconstructed by 3DGS often exhibit a falsely transparent surface, leading to inconsistent background and internal patterns under camera motion in interactive viewing. This issue stems from the ill-posed optimization in 3DGS. During training, background and foreground Gaussians are blended via alpha-compositing and optimized solely against the input RGB images using a photometric loss. As this process lacks an explicit constraint on surface opacity, the optimization may incorrectly assign transparency to opaque regions, resulting in view-inconsistent and falsely transparent. This issue is difficult to detect in standard evaluation settings but becomes particularly evident in object-centric reconstructions under interactive viewing. Although other causes of view-inconsistency have been explored recently, false transparency has not been explicitly identified. To the best of our knowledge, we are the first to identify, characterize, and develop solutions for this artifact, an underreported artifact in 3DGS. Our strategy, NGS, encourages surface Gaussians to adopt higher opacity by injecting opaque noise Gaussians in the object volume during training, requiring only minimal modifications to the existing splatting process. To quantitatively evaluate false transparency in static renderings, we propose a transmittance-based metric that measures the severity of this artifact. In addition, we introduce a customized, high-quality object-centric scan dataset exhibiting pronounced transparency issues, and we augment popular existing datasets with complementary infill noise specifically designed to assess the robustness of 3D reconstruction methods to false transparency. Experiments across multiple datasets show that NGS substantially reduces false transparency while maintaining competitive performance on standard rendering metrics, demonstrating its overall effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02493v3">Low-Frequency First: Eliminating Floating Artifacts in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Our paper has been accepted by the 24th International Conference on Cyberworlds and recieved the Best Paper Honorable Mention
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a powerful and computationally efficient representation for 3D reconstruction. Despite its strengths, 3DGS often produces floating artifacts, which are erroneous structures detached from the actual geometry and significantly degrade visual fidelity. The underlying mechanisms causing these artifacts, particularly in low-quality initialization scenarios, have not been fully explored. In this paper, we investigate the origins of floating artifacts from a frequency-domain perspective and identify under-optimized Gaussians as the primary source. Based on our analysis, we propose \textit{Eliminating-Floating-Artifacts} Gaussian Splatting (EFA-GS), which selectively expands under-optimized Gaussians to prioritize accurate low-frequency learning. Additionally, we introduce complementary depth-based and scale-based strategies to dynamically refine Gaussian expansion, effectively mitigating detail erosion. Extensive experiments on both synthetic and real-world datasets demonstrate that EFA-GS substantially reduces floating artifacts while preserving high-frequency details, achieving an improvement of 1.68 dB in PSNR over baseline method on our RWLQ dataset. Furthermore, we validate the effectiveness of our approach in downstream 3D editing tasks. Project Website: https://jcwang-gh.github.io/EFA-GS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15386v1">PFGS: Pose-Fused 3D Gaussian Splatting for Complete Multi-Pose Object Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled high-quality, real-time novel-view synthesis from multi-view images. However, most existing methods assume the object is captured in a single, static pose, resulting in incomplete reconstructions that miss occluded or self-occluded regions. We introduce PFGS, a pose-aware 3DGS framework that addresses the practical challenge of reconstructing complete objects from multi-pose image captures. Given images of an object in one main pose and several auxiliary poses, PFGS iteratively fuses each auxiliary set into a unified 3DGS representation of the main pose. Our pose-aware fusion strategy combines global and local registration to merge views effectively and refine the 3DGS model. While recent advances in 3D foundation models have improved registration robustness and efficiency, they remain limited by high memory demands and suboptimal accuracy. PFGS overcomes these challenges by incorporating them more intelligently into the registration process: it leverages background features for per-pose camera pose estimation and employs foundation models for cross-pose registration. This design captures the best of both approaches while resolving background inconsistency issues. Experimental results demonstrate that PFGS consistently outperforms strong baselines in both qualitative and quantitative evaluations, producing more complete reconstructions and higher-fidelity 3DGS models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15352v1">GaussGym: An open-source real-to-sim framework for learning locomotion from pixels</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      We present a novel approach for photorealistic robot simulation that integrates 3D Gaussian Splatting as a drop-in renderer within vectorized physics simulators such as IsaacGym. This enables unprecedented speed -- exceeding 100,000 steps per second on consumer GPUs -- while maintaining high visual fidelity, which we showcase across diverse tasks. We additionally demonstrate its applicability in a sim-to-real robotics setting. Beyond depth-based sensing, our results highlight how rich visual semantics improve navigation and decision-making, such as avoiding undesirable regions. We further showcase the ease of incorporating thousands of environments from iPhone scans, large-scale scene datasets (e.g., GrandTour, ARKit), and outputs from generative video models like Veo, enabling rapid creation of realistic training worlds. This work bridges high-throughput simulation and high-fidelity perception, advancing scalable and generalizable robot learning. All code and data will be open-sourced for the community to build upon. Videos, code, and data available at https://escontrela.me/gauss_gym/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.21779v2">X$^{2}$-Gaussian: 4D Radiative Gaussian Splatting for Continuous-time Tomographic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Project Page: https://x2-gaussian.github.io/
    </div>
    <details class="paper-abstract">
      Four-dimensional computed tomography (4D CT) reconstruction is crucial for capturing dynamic anatomical changes but faces inherent limitations from conventional phase-binning workflows. Current methods discretize temporal resolution into fixed phases with respiratory gating devices, introducing motion misalignment and restricting clinical practicality. In this paper, We propose X$^2$-Gaussian, a novel framework that enables continuous-time 4D-CT reconstruction by integrating dynamic radiative Gaussian splatting with self-supervised respiratory motion learning. Our approach models anatomical dynamics through a spatiotemporal encoder-decoder architecture that predicts time-varying Gaussian deformations, eliminating phase discretization. To remove dependency on external gating devices, we introduce a physiology-driven periodic consistency loss that learns patient-specific breathing cycles directly from projections via differentiable optimization. Extensive experiments demonstrate state-of-the-art performance, achieving a 9.93 dB PSNR gain over traditional methods and 2.25 dB improvement against prior Gaussian splatting techniques. By unifying continuous motion modeling with hardware-free period learning, X$^2$-Gaussian advances high-fidelity 4D CT reconstruction for dynamic clinical imaging. Code is publicly available at: https://x2-gaussian.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.12493v2">BSGS: Bi-stage 3D Gaussian Splatting for Camera Motion Deblurring</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Accept by ACM MM 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has exhibited remarkable capabilities in 3D scene reconstruction. However, reconstructing high-quality 3D scenes from motion-blurred images caused by camera motion poses a significant challenge.The performance of existing 3DGS-based deblurring methods are limited due to their inherent mechanisms, such as extreme dependence on the accuracy of camera poses and inability to effectively control erroneous Gaussian primitives densification caused by motion blur. To solve these problems, we introduce a novel framework, Bi-Stage 3D Gaussian Splatting, to accurately reconstruct 3D scenes from motion-blurred images. BSGS contains two stages. First, Camera Pose Refinement roughly optimizes camera poses to reduce motion-induced distortions. Second, with fixed rough camera poses, Global RigidTransformation further corrects motion-induced blur distortions. To alleviate multi-subframe gradient conflicts, we propose a subframe gradient aggregation strategy to optimize both stages. Furthermore, a space-time bi-stage optimization strategy is introduced to dynamically adjust primitive densification thresholds and prevent premature noisy Gaussian generation in blurred regions. Comprehensive experiments verify the effectiveness of our proposed deblurring method and show its superiority over the state of the arts.Our source code is available at https://github.com/wsxujm/bsgs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.23277v2">iLRM: An Iterative Large 3D Reconstruction Model</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Project page: https://gynjn.github.io/iLRM/
    </div>
    <details class="paper-abstract">
      Feed-forward 3D modeling has emerged as a promising approach for rapid and high-quality 3D reconstruction. In particular, directly generating explicit 3D representations, such as 3D Gaussian splatting, has attracted significant attention due to its fast and high-quality rendering, as well as numerous applications. However, many state-of-the-art methods, primarily based on transformer architectures, suffer from severe scalability issues because they rely on full attention across image tokens from multiple input views, resulting in prohibitive computational costs as the number of views or image resolution increases. Toward a scalable and efficient feed-forward 3D reconstruction, we introduce an iterative Large 3D Reconstruction Model (iLRM) that generates 3D Gaussian representations through an iterative refinement mechanism, guided by three core principles: (1) decoupling the scene representation from input-view images to enable compact 3D representations; (2) decomposing fully-attentional multi-view interactions into a two-stage attention scheme to reduce computational costs; and (3) injecting high-resolution information at every layer to achieve high-fidelity reconstruction. Experimental results on widely used datasets, such as RE10K and DL3DV, demonstrate that iLRM outperforms existing methods in both reconstruction quality and speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15072v1">SaLon3R: Structure-aware Long-term Generalizable 3D Reconstruction from Unposed Images</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled generalizable, on-the-fly reconstruction of sequential input views. However, existing methods often predict per-pixel Gaussians and combine Gaussians from all views as the scene representation, leading to substantial redundancies and geometric inconsistencies in long-duration video sequences. To address this, we propose SaLon3R, a novel framework for Structure-aware, Long-term 3DGS Reconstruction. To our best knowledge, SaLon3R is the first online generalizable GS method capable of reconstructing over 50 views in over 10 FPS, with 50% to 90% redundancy removal. Our method introduces compact anchor primitives to eliminate redundancy through differentiable saliency-aware Gaussian quantization, coupled with a 3D Point Transformer that refines anchor attributes and saliency to resolve cross-frame geometric and photometric inconsistencies. Specifically, we first leverage a 3D reconstruction backbone to predict dense per-pixel Gaussians and a saliency map encoding regional geometric complexity. Redundant Gaussians are compressed into compact anchors by prioritizing high-complexity regions. The 3D Point Transformer then learns spatial structural priors in 3D space from training data to refine anchor attributes and saliency, enabling regionally adaptive Gaussian decoding for geometric fidelity. Without known camera parameters or test-time optimization, our approach effectively resolves artifacts and prunes the redundant 3DGS in a single feed-forward pass. Experiments on multiple datasets demonstrate our state-of-the-art performance on both novel view synthesis and depth estimation, demonstrating superior efficiency, robustness, and generalization ability for long-term generalizable 3D reconstruction. Project Page: https://wrld.github.io/SaLon3R/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14705v1">Leveraging Learned Image Prior for 3D Gaussian Compression</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted to ICCV 2025 Workshop on ECLR
    </div>
    <details class="paper-abstract">
      Compression techniques for 3D Gaussian Splatting (3DGS) have recently achieved considerable success in minimizing storage overhead for 3D Gaussians while preserving high rendering quality. Despite the impressive storage reduction, the lack of learned priors restricts further advances in the rate-distortion trade-off for 3DGS compression tasks. To address this, we introduce a novel 3DGS compression framework that leverages the powerful representational capacity of learned image priors to recover compression-induced quality degradation. Built upon initially compressed Gaussians, our restoration network effectively models the compression artifacts in the image space between degraded and original Gaussians. To enhance the rate-distortion performance, we provide coarse rendering residuals into the restoration network as side information. By leveraging the supervision of restored images, the compressed Gaussians are refined, resulting in a highly compact representation with enhanced rendering performance. Our framework is designed to be compatible with existing Gaussian compression methods, making it broadly applicable across different baselines. Extensive experiments validate the effectiveness of our framework, demonstrating superior rate-distortion performance and outperforming the rendering quality of state-of-the-art 3DGS compression methods while requiring substantially less storage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14564v1">BalanceGS: Algorithm-System Co-design for Efficient 3D Gaussian Splatting Training on GPU</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted by ASP-DAC 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a promising 3D reconstruction technique. The traditional 3DGS training pipeline follows three sequential steps: Gaussian densification, Gaussian projection, and color splatting. Despite its promising reconstruction quality, this conventional approach suffers from three critical inefficiencies: (1) Skewed density allocation during Gaussian densification, (2) Imbalanced computation workload during Gaussian projection and (3) Fragmented memory access during color splatting. To tackle the above challenges, we introduce BalanceGS, the algorithm-system co-design for efficient training in 3DGS. (1) At the algorithm level, we propose heuristic workload-sensitive Gaussian density control to automatically balance point distributions - removing 80% redundant Gaussians in dense regions while filling gaps in sparse areas. (2) At the system level, we propose Similarity-based Gaussian sampling and merging, which replaces the static one-to-one thread-pixel mapping with adaptive workload distribution - threads now dynamically process variable numbers of Gaussians based on local cluster density. (3) At the mapping level, we propose reordering-based memory access mapping strategy that restructures RGB storage and enables batch loading in shared memory. Extensive experiments demonstrate that compared with 3DGS, our approach achieves a 1.44$\times$ training speedup on a NVIDIA A100 GPU with negligible quality degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14270v1">GauSSmart: Enhanced 3D Reconstruction through 2D Foundation Models and Geometric Filtering</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Scene reconstruction has emerged as a central challenge in computer vision, with approaches such as Neural Radiance Fields (NeRF) and Gaussian Splatting achieving remarkable progress. While Gaussian Splatting demonstrates strong performance on large-scale datasets, it often struggles to capture fine details or maintain realism in regions with sparse coverage, largely due to the inherent limitations of sparse 3D training data. In this work, we propose GauSSmart, a hybrid method that effectively bridges 2D foundational models and 3D Gaussian Splatting reconstruction. Our approach integrates established 2D computer vision techniques, including convex filtering and semantic feature supervision from foundational models such as DINO, to enhance Gaussian-based scene reconstruction. By leveraging 2D segmentation priors and high-dimensional feature embeddings, our method guides the densification and refinement of Gaussian splats, improving coverage in underrepresented areas and preserving intricate structural details. We validate our approach across three datasets, where GauSSmart consistently outperforms existing Gaussian Splatting in the majority of evaluated scenes. Our results demonstrate the significant potential of hybrid 2D-3D approaches, highlighting how the thoughtful combination of 2D foundational models with 3D reconstruction pipelines can overcome the limitations inherent in either approach alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07944v2">CVD-STORM: Cross-View Video Diffusion with Spatial-Temporal Reconstruction Model for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Generative models have been widely applied to world modeling for environment simulation and future state prediction. With advancements in autonomous driving, there is a growing demand not only for high-fidelity video generation under various controls, but also for producing diverse and meaningful information such as depth estimation. To address this, we propose CVD-STORM, a cross-view video diffusion model utilizing a spatial-temporal reconstruction Variational Autoencoder (VAE) that generates long-term, multi-view videos with 4D reconstruction capabilities under various control inputs. Our approach first fine-tunes the VAE with an auxiliary 4D reconstruction task, enhancing its ability to encode 3D structures and temporal dynamics. Subsequently, we integrate this VAE into the video diffusion process to significantly improve generation quality. Experimental results demonstrate that our model achieves substantial improvements in both FID and FVD metrics. Additionally, the jointly-trained Gaussian Splatting Decoder effectively reconstructs dynamic scenes, providing valuable geometric information for comprehensive scene understanding. Our project page is https://sensetime-fvg.github.io/CVD-STORM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15008v2">HuGDiffusion: Generalizable Single-Image Human Rendering via 3D Gaussian Diffusion</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      We present HuGDiffusion, a generalizable 3D Gaussian splatting (3DGS) learning pipeline to achieve novel view synthesis (NVS) of human characters from single-view input images. Existing approaches typically require monocular videos or calibrated multi-view images as inputs, whose applicability could be weakened in real-world scenarios with arbitrary and/or unknown camera poses. In this paper, we aim to generate the set of 3DGS attributes via a diffusion-based framework conditioned on human priors extracted from a single image. Specifically, we begin with carefully integrated human-centric feature extraction procedures to deduce informative conditioning signals. Based on our empirical observations that jointly learning the whole 3DGS attributes is challenging to optimize, we design a multi-stage generation strategy to obtain different types of 3DGS attributes. To facilitate the training process, we investigate constructing proxy ground-truth 3D Gaussian attributes as high-quality attribute-level supervision signals. Through extensive experiments, our HuGDiffusion shows significant performance improvements over the state-of-the-art methods. Our code will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14179v1">Virtually Being: Customizing Camera-Controllable Video Diffusion Models with Multi-View Performance Captures</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted to SIGGRAPH Asia 2025
    </div>
    <details class="paper-abstract">
      We introduce a framework that enables both multi-view character consistency and 3D camera control in video diffusion models through a novel customization data pipeline. We train the character consistency component with recorded volumetric capture performances re-rendered with diverse camera trajectories via 4D Gaussian Splatting (4DGS), lighting variability obtained with a video relighting model. We fine-tune state-of-the-art open-source video diffusion models on this data to provide strong multi-view identity preservation, precise camera control, and lighting adaptability. Our framework also supports core capabilities for virtual production, including multi-subject generation using two approaches: joint training and noise blending, the latter enabling efficient composition of independently customized models at inference time; it also achieves scene and real-life video customization as well as control over motion and spatial layout during customization. Extensive experiments show improved video quality, higher personalization accuracy, and enhanced camera control and lighting adaptability, advancing the integration of video generation into virtual production. Our project page is available at: https://eyeline-labs.github.io/Virtually-Being.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01978v2">ROI-GS: Interest-based Local Quality 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 4 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      We tackle the challenge of efficiently reconstructing 3D scenes with high detail on objects of interest. Existing 3D Gaussian Splatting (3DGS) methods allocate resources uniformly across the scene, limiting fine detail to Regions Of Interest (ROIs) and leading to inflated model size. We propose ROI-GS, an object-aware framework that enhances local details through object-guided camera selection, targeted Object training, and seamless integration of high-fidelity object of interest reconstructions into the global scene. Our method prioritizes higher resolution details on chosen objects while maintaining real-time performance. Experiments show that ROI-GS significantly improves local quality (up to 2.96 dB PSNR), while reducing overall model size by $\approx 17\%$ of baseline and achieving faster training for a scene with a single object of interest, outperforming existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.01978v2">ROI-GS: Interest-based Local Quality 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 4 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      We tackle the challenge of efficiently reconstructing 3D scenes with high detail on objects of interest. Existing 3D Gaussian Splatting (3DGS) methods allocate resources uniformly across the scene, limiting fine detail to Regions Of Interest (ROIs) and leading to inflated model size. We propose ROI-GS, an object-aware framework that enhances local details through object-guided camera selection, targeted Object training, and seamless integration of high-fidelity object of interest reconstructions into the global scene. Our method prioritizes higher resolution details on chosen objects while maintaining real-time performance. Experiments show that ROI-GS significantly improves local quality (up to 2.96 dB PSNR), while reducing overall model size by $\approx 17\%$ of baseline and achieving faster training for a scene with a single object of interest, outperforming existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15072v1">SaLon3R: Structure-aware Long-term Generalizable 3D Reconstruction from Unposed Images</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled generalizable, on-the-fly reconstruction of sequential input views. However, existing methods often predict per-pixel Gaussians and combine Gaussians from all views as the scene representation, leading to substantial redundancies and geometric inconsistencies in long-duration video sequences. To address this, we propose SaLon3R, a novel framework for Structure-aware, Long-term 3DGS Reconstruction. To our best knowledge, SaLon3R is the first online generalizable GS method capable of reconstructing over 50 views in over 10 FPS, with 50% to 90% redundancy removal. Our method introduces compact anchor primitives to eliminate redundancy through differentiable saliency-aware Gaussian quantization, coupled with a 3D Point Transformer that refines anchor attributes and saliency to resolve cross-frame geometric and photometric inconsistencies. Specifically, we first leverage a 3D reconstruction backbone to predict dense per-pixel Gaussians and a saliency map encoding regional geometric complexity. Redundant Gaussians are compressed into compact anchors by prioritizing high-complexity regions. The 3D Point Transformer then learns spatial structural priors in 3D space from training data to refine anchor attributes and saliency, enabling regionally adaptive Gaussian decoding for geometric fidelity. Without known camera parameters or test-time optimization, our approach effectively resolves artifacts and prunes the redundant 3DGS in a single feed-forward pass. Experiments on multiple datasets demonstrate our state-of-the-art performance on both novel view synthesis and depth estimation, demonstrating superior efficiency, robustness, and generalization ability for long-term generalizable 3D reconstruction. Project Page: https://wrld.github.io/SaLon3R/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14705v1">Leveraging Learned Image Prior for 3D Gaussian Compression</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted to ICCV 2025 Workshop on ECLR
    </div>
    <details class="paper-abstract">
      Compression techniques for 3D Gaussian Splatting (3DGS) have recently achieved considerable success in minimizing storage overhead for 3D Gaussians while preserving high rendering quality. Despite the impressive storage reduction, the lack of learned priors restricts further advances in the rate-distortion trade-off for 3DGS compression tasks. To address this, we introduce a novel 3DGS compression framework that leverages the powerful representational capacity of learned image priors to recover compression-induced quality degradation. Built upon initially compressed Gaussians, our restoration network effectively models the compression artifacts in the image space between degraded and original Gaussians. To enhance the rate-distortion performance, we provide coarse rendering residuals into the restoration network as side information. By leveraging the supervision of restored images, the compressed Gaussians are refined, resulting in a highly compact representation with enhanced rendering performance. Our framework is designed to be compatible with existing Gaussian compression methods, making it broadly applicable across different baselines. Extensive experiments validate the effectiveness of our framework, demonstrating superior rate-distortion performance and outperforming the rendering quality of state-of-the-art 3DGS compression methods while requiring substantially less storage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14564v1">BalanceGS: Algorithm-System Co-design for Efficient 3D Gaussian Splatting Training on GPU</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted by ASP-DAC 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a promising 3D reconstruction technique. The traditional 3DGS training pipeline follows three sequential steps: Gaussian densification, Gaussian projection, and color splatting. Despite its promising reconstruction quality, this conventional approach suffers from three critical inefficiencies: (1) Skewed density allocation during Gaussian densification, (2) Imbalanced computation workload during Gaussian projection and (3) Fragmented memory access during color splatting. To tackle the above challenges, we introduce BalanceGS, the algorithm-system co-design for efficient training in 3DGS. (1) At the algorithm level, we propose heuristic workload-sensitive Gaussian density control to automatically balance point distributions - removing 80% redundant Gaussians in dense regions while filling gaps in sparse areas. (2) At the system level, we propose Similarity-based Gaussian sampling and merging, which replaces the static one-to-one thread-pixel mapping with adaptive workload distribution - threads now dynamically process variable numbers of Gaussians based on local cluster density. (3) At the mapping level, we propose reordering-based memory access mapping strategy that restructures RGB storage and enables batch loading in shared memory. Extensive experiments demonstrate that compared with 3DGS, our approach achieves a 1.44$\times$ training speedup on a NVIDIA A100 GPU with negligible quality degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.07944v2">CVD-STORM: Cross-View Video Diffusion with Spatial-Temporal Reconstruction Model for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Generative models have been widely applied to world modeling for environment simulation and future state prediction. With advancements in autonomous driving, there is a growing demand not only for high-fidelity video generation under various controls, but also for producing diverse and meaningful information such as depth estimation. To address this, we propose CVD-STORM, a cross-view video diffusion model utilizing a spatial-temporal reconstruction Variational Autoencoder (VAE) that generates long-term, multi-view videos with 4D reconstruction capabilities under various control inputs. Our approach first fine-tunes the VAE with an auxiliary 4D reconstruction task, enhancing its ability to encode 3D structures and temporal dynamics. Subsequently, we integrate this VAE into the video diffusion process to significantly improve generation quality. Experimental results demonstrate that our model achieves substantial improvements in both FID and FVD metrics. Additionally, the jointly-trained Gaussian Splatting Decoder effectively reconstructs dynamic scenes, providing valuable geometric information for comprehensive scene understanding. Our project page is https://sensetime-fvg.github.io/CVD-STORM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.15008v2">HuGDiffusion: Generalizable Single-Image Human Rendering via 3D Gaussian Diffusion</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      We present HuGDiffusion, a generalizable 3D Gaussian splatting (3DGS) learning pipeline to achieve novel view synthesis (NVS) of human characters from single-view input images. Existing approaches typically require monocular videos or calibrated multi-view images as inputs, whose applicability could be weakened in real-world scenarios with arbitrary and/or unknown camera poses. In this paper, we aim to generate the set of 3DGS attributes via a diffusion-based framework conditioned on human priors extracted from a single image. Specifically, we begin with carefully integrated human-centric feature extraction procedures to deduce informative conditioning signals. Based on our empirical observations that jointly learning the whole 3DGS attributes is challenging to optimize, we design a multi-stage generation strategy to obtain different types of 3DGS attributes. To facilitate the training process, we investigate constructing proxy ground-truth 3D Gaussian attributes as high-quality attribute-level supervision signals. Through extensive experiments, our HuGDiffusion shows significant performance improvements over the state-of-the-art methods. Our code will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14179v1">Virtually Being: Customizing Camera-Controllable Video Diffusion Models with Multi-View Performance Captures</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted to SIGGRAPH Asia 2025
    </div>
    <details class="paper-abstract">
      We introduce a framework that enables both multi-view character consistency and 3D camera control in video diffusion models through a novel customization data pipeline. We train the character consistency component with recorded volumetric capture performances re-rendered with diverse camera trajectories via 4D Gaussian Splatting (4DGS), lighting variability obtained with a video relighting model. We fine-tune state-of-the-art open-source video diffusion models on this data to provide strong multi-view identity preservation, precise camera control, and lighting adaptability. Our framework also supports core capabilities for virtual production, including multi-subject generation using two approaches: joint training and noise blending, the latter enabling efficient composition of independently customized models at inference time; it also achieves scene and real-life video customization as well as control over motion and spatial layout during customization. Extensive experiments show improved video quality, higher personalization accuracy, and enhanced camera control and lighting adaptability, advancing the integration of video generation into virtual production. Our project page is available at: https://eyeline-labs.github.io/Virtually-Being.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14081v1">Capture, Canonicalize, Splat: Zero-Shot 3D Gaussian Avatars from Unstructured Phone Images</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      We present a novel, zero-shot pipeline for creating hyperrealistic, identity-preserving 3D avatars from a few unstructured phone images. Existing methods face several challenges: single-view approaches suffer from geometric inconsistencies and hallucinations, degrading identity preservation, while models trained on synthetic data fail to capture high-frequency details like skin wrinkles and fine hair, limiting realism. Our method introduces two key contributions: (1) a generative canonicalization module that processes multiple unstructured views into a standardized, consistent representation, and (2) a transformer-based model trained on a new, large-scale dataset of high-fidelity Gaussian splatting avatars derived from dome captures of real people. This "Capture, Canonicalize, Splat" pipeline produces static quarter-body avatars with compelling realism and robust identity preservation from unstructured photos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13978v1">Instant Skinned Gaussian Avatars for Web, Mobile and VR Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Accepted to SUI 2025 Demo Track
    </div>
    <details class="paper-abstract">
      We present Instant Skinned Gaussian Avatars, a real-time and cross-platform 3D avatar system. Many approaches have been proposed to animate Gaussian Splatting, but they often require camera arrays, long preprocessing times, or high-end GPUs. Some methods attempt to convert Gaussian Splatting into mesh-based representations, achieving lightweight performance but sacrificing visual fidelity. In contrast, our system efficiently animates Gaussian Splatting by leveraging parallel splat-wise processing to dynamically follow the underlying skinned mesh in real time while preserving high visual fidelity. From smartphone-based 3D scanning to on-device preprocessing, the entire process takes just around five minutes, with the avatar generation step itself completed in only about 30 seconds. Our system enables users to instantly transform their real-world appearance into a 3D avatar, making it ideal for seamless integration with social media and metaverse applications. Website: https://sites.google.com/view/gaussian-vrm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17864v1">InsideOut: Integrated RGB-Radiative Gaussian Splatting for Comprehensive 3D Object Representation</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Published at ICCV 2025
    </div>
    <details class="paper-abstract">
      We introduce InsideOut, an extension of 3D Gaussian splatting (3DGS) that bridges the gap between high-fidelity RGB surface details and subsurface X-ray structures. The fusion of RGB and X-ray imaging is invaluable in fields such as medical diagnostics, cultural heritage restoration, and manufacturing. We collect new paired RGB and X-ray data, perform hierarchical fitting to align RGB and X-ray radiative Gaussian splats, and propose an X-ray reference loss to ensure consistent internal structures. InsideOut effectively addresses the challenges posed by disparate data representations between the two modalities and limited paired datasets. This approach significantly extends the applicability of 3DGS, enhancing visualization, simulation, and non-destructive testing capabilities across various domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13454v1">VIST3A: Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Project page: https://gohyojun15.github.io/VIST3A/
    </div>
    <details class="paper-abstract">
      The rapid progress of large, pretrained models for both visual content generation and 3D reconstruction opens up new possibilities for text-to-3D generation. Intuitively, one could obtain a formidable 3D scene generator if one were able to combine the power of a modern latent text-to-video model as "generator" with the geometric abilities of a recent (feedforward) 3D reconstruction system as "decoder". We introduce VIST3A, a general framework that does just that, addressing two main challenges. First, the two components must be joined in a way that preserves the rich knowledge encoded in their weights. We revisit model stitching, i.e., we identify the layer in the 3D decoder that best matches the latent representation produced by the text-to-video generator and stitch the two parts together. That operation requires only a small dataset and no labels. Second, the text-to-video generator must be aligned with the stitched 3D decoder, to ensure that the generated latents are decodable into consistent, perceptually convincing 3D scene geometry. To that end, we adapt direct reward finetuning, a popular technique for human preference alignment. We evaluate the proposed VIST3A approach with different video generators and 3D reconstruction models. All tested pairings markedly improve over prior text-to-3D models that output Gaussian splats. Moreover, by choosing a suitable 3D base model, VIST3A also enables high-quality text-to-pointmap generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13381v1">Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Accepted at ICCV-2025, project page: https://dynamic-ugsdf.github.io/
    </div>
    <details class="paper-abstract">
      Dynamic scene rendering and reconstruction play a crucial role in computer vision and augmented reality. Recent methods based on 3D Gaussian Splatting (3DGS), have enabled accurate modeling of dynamic urban scenes, but for urban scenes they require both camera and LiDAR data, ground-truth 3D segmentations and motion data in the form of tracklets or pre-defined object templates such as SMPL. In this work, we explore whether a combination of 2D object agnostic priors in the form of depth and point tracking coupled with a signed distance function (SDF) representation for dynamic objects can be used to relax some of these requirements. We present a novel approach that integrates Signed Distance Functions (SDFs) with 3D Gaussian Splatting (3DGS) to create a more robust object representation by harnessing the strengths of both methods. Our unified optimization framework enhances the geometric accuracy of 3D Gaussian splatting and improves deformation modeling within the SDF, resulting in a more adaptable and precise representation. We demonstrate that our method achieves state-of-the-art performance in rendering metrics even without LiDAR data on urban scenes. When incorporating LiDAR, our approach improved further in reconstructing and generating novel views across diverse object categories, without ground-truth 3D motion annotation. Additionally, our method enables various scene editing tasks, including scene decomposition, and scene composition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13186v1">STT-GS: Sample-Then-Transmit Edge Gaussian Splatting with Joint Client Selection and Power Control</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Edge Gaussian splatting (EGS), which aggregates data from distributed clients and trains a global GS model at the edge server, is an emerging paradigm for scene reconstruction. Unlike traditional edge resource management methods that emphasize communication throughput or general-purpose learning performance, EGS explicitly aims to maximize the GS qualities, rendering existing approaches inapplicable. To address this problem, this paper formulates a novel GS-oriented objective function that distinguishes the heterogeneous view contributions of different clients. However, evaluating this function in turn requires clients' images, leading to a causality dilemma. To this end, this paper further proposes a sample-then-transmit EGS (or STT-GS for short) strategy, which first samples a subset of images as pilot data from each client for loss prediction. Based on the first-stage evaluation, communication resources are then prioritized towards more valuable clients. To achieve efficient sampling, a feature-domain clustering (FDC) scheme is proposed to select the most representative data and pilot transmission time minimization (PTTM) is adopted to reduce the pilot overhead.Subsequently, we develop a joint client selection and power control (JCSPC) framework to maximize the GS-oriented function under communication resource constraints. Despite the nonconvexity of the problem, we propose a low-complexity efficient solution based on the penalty alternating majorization minimization (PAMM) algorithm. Experiments unveil that the proposed scheme significantly outperforms existing benchmarks on real-world datasets. It is found that the GS-oriented objective can be accurately predicted with low sampling ratios (e.g.,10%), and our method achieves an excellent tradeoff between view contributions and communication costs.
    </details>
</div>
