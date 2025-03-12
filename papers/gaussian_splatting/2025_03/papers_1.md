# gaussian splatting - 2025_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.07199v2">RealmDreamer: Text-Driven 3D Scene Generation with Inpainting and Depth Diffusion</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Published at 3DV 2025
    </div>
    <details class="paper-abstract">
      We introduce RealmDreamer, a technique for generating forward-facing 3D scenes from text descriptions. Our method optimizes a 3D Gaussian Splatting representation to match complex text prompts using pretrained diffusion models. Our key insight is to leverage 2D inpainting diffusion models conditioned on an initial scene estimate to provide low variance supervision for unknown regions during 3D distillation. In conjunction, we imbue high-fidelity geometry with geometric distillation from a depth diffusion model, conditioned on samples from the inpainting model. We find that the initialization of the optimization is crucial, and provide a principled methodology for doing so. Notably, our technique doesn't require video or multi-view data and can synthesize various high-quality 3D scenes in different styles with complex layouts. Further, the generality of our method allows 3D synthesis from a single image. As measured by a comprehensive user study, our method outperforms all existing approaches, preferred by 88-95%. Project Page: https://realmdreamer.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08511v1">PCGS: Progressive Compression of 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Project Page: https://yihangchen-ee.github.io/project_pcgs/ Code: https://github.com/YihangChen-ee/PCGS
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves impressive rendering fidelity and speed for novel view synthesis. However, its substantial data size poses a significant challenge for practical applications. While many compression techniques have been proposed, they fail to efficiently utilize existing bitstreams in on-demand applications due to their lack of progressivity, leading to a waste of resource. To address this issue, we propose PCGS (Progressive Compression of 3D Gaussian Splatting), which adaptively controls both the quantity and quality of Gaussians (or anchors) to enable effective progressivity for on-demand applications. Specifically, for quantity, we introduce a progressive masking strategy that incrementally incorporates new anchors while refining existing ones to enhance fidelity. For quality, we propose a progressive quantization approach that gradually reduces quantization step sizes to achieve finer modeling of Gaussian attributes. Furthermore, to compact the incremental bitstreams, we leverage existing quantization results to refine probability prediction, improving entropy coding efficiency across progressive levels. Overall, PCGS achieves progressivity while maintaining compression performance comparable to SoTA non-progressive methods. Code available at: github.com/YihangChen-ee/PCGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08485v1">TT-GaussOcc: Test-Time Compute for Self-Supervised Occupancy Prediction via Spatio-Temporal Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Self-supervised 3D occupancy prediction offers a promising solution for understanding complex driving scenes without requiring costly 3D annotations. However, training dense voxel decoders to capture fine-grained geometry and semantics can demand hundreds of GPU hours, and such models often fail to adapt to varying voxel resolutions or new classes without extensive retraining. To overcome these limitations, we propose a practical and flexible test-time occupancy prediction framework termed TT-GaussOcc. Our approach incrementally optimizes time-aware 3D Gaussians instantiated from raw sensor streams at runtime, enabling voxelization at arbitrary user-specified resolution. Specifically, TT-GaussOcc operates in a "lift-move-voxel" symphony: we first "lift" surrounding-view semantics obtained from 2D vision foundation models (VLMs) to instantiate Gaussians at non-empty 3D space; Next, we "move" dynamic Gaussians from previous frames along estimated Gaussian scene flow to complete appearance and eliminate trailing artifacts of fast-moving objects, while accumulating static Gaussians to enforce temporal consistency; Finally, we mitigate inherent noises in semantic predictions and scene flow vectors by periodically smoothing neighboring Gaussians during optimization, using proposed trilateral RBF kernels that jointly consider color, semantic, and spatial affinities. The historical static and current dynamic Gaussians are then combined and voxelized to generate occupancy prediction. Extensive experiments on Occ3D and nuCraft with varying voxel resolutions demonstrate that TT-GaussOcc surpasses self-supervised baselines by 46% on mIoU without any offline training, and supports finer voxel resolutions at 2.6 FPS inference speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08352v1">Mitigating Ambiguities in 3D Classification with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      3D classification with point cloud input is a fundamental problem in 3D vision. However, due to the discrete nature and the insufficient material description of point cloud representations, there are ambiguities in distinguishing wire-like and flat surfaces, as well as transparent or reflective objects. To address these issues, we propose Gaussian Splatting (GS) point cloud-based 3D classification. We find that the scale and rotation coefficients in the GS point cloud help characterize surface types. Specifically, wire-like surfaces consist of multiple slender Gaussian ellipsoids, while flat surfaces are composed of a few flat Gaussian ellipsoids. Additionally, the opacity in the GS point cloud represents the transparency characteristics of objects. As a result, ambiguities in point cloud-based 3D classification can be mitigated utilizing GS point cloud as input. To verify the effectiveness of GS point cloud input, we construct the first real-world GS point cloud dataset in the community, which includes 20 categories with 200 objects in each category. Experiments not only validate the superiority of GS point cloud input, especially in distinguishing ambiguous objects, but also demonstrate the generalization ability across different classification methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08317v1">Uni-Gaussians: Unifying Camera and Lidar Simulation with Gaussians for Dynamic Driving Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Ensuring the safety of autonomous vehicles necessitates comprehensive simulation of multi-sensor data, encompassing inputs from both cameras and LiDAR sensors, across various dynamic driving scenarios. Neural rendering techniques, which utilize collected raw sensor data to simulate these dynamic environments, have emerged as a leading methodology. While NeRF-based approaches can uniformly represent scenes for rendering data from both camera and LiDAR, they are hindered by slow rendering speeds due to dense sampling. Conversely, Gaussian Splatting-based methods employ Gaussian primitives for scene representation and achieve rapid rendering through rasterization. However, these rasterization-based techniques struggle to accurately model non-linear optical sensors. This limitation restricts their applicability to sensors beyond pinhole cameras. To address these challenges and enable unified representation of dynamic driving scenarios using Gaussian primitives, this study proposes a novel hybrid approach. Our method utilizes rasterization for rendering image data while employing Gaussian ray-tracing for LiDAR data rendering. Experimental results on public datasets demonstrate that our approach outperforms current state-of-the-art methods. This work presents a unified and efficient solution for realistic simulation of camera and LiDAR data in autonomous driving scenarios using Gaussian primitives, offering significant advancements in both rendering quality and computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08305v1">ELECTRA: A Symmetry-breaking Cartesian Network for Charge Density Prediction with Floating Orbitals</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 8 pages, 3 figures, 1 table
    </div>
    <details class="paper-abstract">
      We present the Electronic Tensor Reconstruction Algorithm (ELECTRA) - an equivariant model for predicting electronic charge densities using "floating" orbitals. Floating orbitals are a long-standing idea in the quantum chemistry community that promises more compact and accurate representations by placing orbitals freely in space, as opposed to centering all orbitals at the position of atoms. Finding ideal placements of these orbitals requires extensive domain knowledge though, which thus far has prevented widespread adoption. We solve this in a data-driven manner by training a Cartesian tensor network to predict orbital positions along with orbital coefficients. This is made possible through a symmetry-breaking mechanism that is used to learn position displacements with lower symmetry than the input molecule while preserving the rotation equivariance of the charge density itself. Inspired by recent successes of Gaussian Splatting in representing densities in space, we are using Gaussians as our orbitals and predict their weights and covariance matrices. Our method achieves a state-of-the-art balance between computational efficiency and predictive accuracy on established benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08224v1">HRAvatar: High-Quality and Relightable Gaussian Head Avatar</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Project page: https://eastbeanzhang.github.io/HRAvatar
    </div>
    <details class="paper-abstract">
      Reconstructing animatable and high-quality 3D head avatars from monocular videos, especially with realistic relighting, is a valuable task. However, the limited information from single-view input, combined with the complex head poses and facial movements, makes this challenging. Previous methods achieve real-time performance by combining 3D Gaussian Splatting with a parametric head model, but the resulting head quality suffers from inaccurate face tracking and limited expressiveness of the deformation model. These methods also fail to produce realistic effects under novel lighting conditions. To address these issues, we propose HRAvatar, a 3DGS-based method that reconstructs high-fidelity, relightable 3D head avatars. HRAvatar reduces tracking errors through end-to-end optimization and better captures individual facial deformations using learnable blendshapes and learnable linear blend skinning. Additionally, it decomposes head appearance into several physical properties and incorporates physically-based shading to account for environmental lighting. Extensive experiments demonstrate that HRAvatar not only reconstructs superior-quality heads but also achieves realistic visual effects under varying lighting conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08217v1">S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has reshaped the field of photorealistic 3D reconstruction, achieving impressive rendering quality and speed. However, when applied to large-scale street scenes, existing methods suffer from rapidly escalating per-viewpoint reconstruction costs as scene size increases, leading to significant computational overhead. After revisiting the conventional pipeline, we identify three key factors accounting for this issue: unnecessary local-to-global transformations, excessive 3D-to-2D projections, and inefficient rendering of distant content. To address these challenges, we propose S3R-GS, a 3DGS framework that Streamlines the pipeline for large-scale Street Scene Reconstruction, effectively mitigating these limitations. Moreover, most existing street 3DGS methods rely on ground-truth 3D bounding boxes to separate dynamic and static components, but 3D bounding boxes are difficult to obtain, limiting real-world applicability. To address this, we propose an alternative solution with 2D boxes, which are easier to annotate or can be predicted by off-the-shelf vision foundation models. Such designs together make S3R-GS readily adapt to large, in-the-wild scenarios. Extensive experiments demonstrate that S3R-GS enhances rendering quality and significantly accelerates reconstruction. Remarkably, when applied to videos from the challenging Argoverse2 dataset, it achieves state-of-the-art PSNR and SSIM, reducing reconstruction time to below 50%--and even 20%--of competing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08166v1">Dynamic Scene Reconstruction: Recent Advance in Real-time Rendering and Streaming</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 20 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Representing and rendering dynamic scenes from 2D images is a fundamental yet challenging problem in computer vision and graphics. This survey provides a comprehensive review of the evolution and advancements in dynamic scene representation and rendering, with a particular emphasis on recent progress in Neural Radiance Fields based and 3D Gaussian Splatting based reconstruction methods. We systematically summarize existing approaches, categorize them according to their core principles, compile relevant datasets, compare the performance of various methods on these benchmarks, and explore the challenges and future research directions in this rapidly evolving field. In total, we review over 170 relevant papers, offering a broad perspective on the state of the art in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08135v1">ArticulatedGS: Self-supervised Digital Twin Modeling of Articulated Objects using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      We tackle the challenge of concurrent reconstruction at the part level with the RGB appearance and estimation of motion parameters for building digital twins of articulated objects using the 3D Gaussian Splatting (3D-GS) method. With two distinct sets of multi-view imagery, each depicting an object in separate static articulation configurations, we reconstruct the articulated object in 3D Gaussian representations with both appearance and geometry information at the same time. Our approach decoupled multiple highly interdependent parameters through a multi-step optimization process, thereby achieving a stable optimization procedure and high-quality outcomes. We introduce ArticulatedGS, a self-supervised, comprehensive framework that autonomously learns to model shapes and appearances at the part level and synchronizes the optimization of motion parameters, all without reliance on 3D supervision, motion cues, or semantic labels. Our experimental results demonstrate that, among comparable methodologies, our approach has achieved optimal outcomes in terms of part segmentation accuracy, motion estimation accuracy, and visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06714v3">F3D-Gaus: Feed-forward 3D-aware Generation on ImageNet with Cycle-Aggregative Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Project Page: https://w-ted.github.io/publications/F3D-Gaus
    </div>
    <details class="paper-abstract">
      This paper tackles the problem of generalizable 3D-aware generation from monocular datasets, e.g., ImageNet. The key challenge of this task is learning a robust 3D-aware representation without multi-view or dynamic data, while ensuring consistent texture and geometry across different viewpoints. Although some baseline methods are capable of 3D-aware generation, the quality of the generated images still lags behind state-of-the-art 2D generation approaches, which excel in producing high-quality, detailed images. To address this severe limitation, we propose a novel feed-forward pipeline based on pixel-aligned Gaussian Splatting, coined as F3D-Gaus, which can produce more realistic and reliable 3D renderings from monocular inputs. In addition, we introduce a self-supervised cycle-aggregative constraint to enforce cross-view consistency in the learned 3D representation. This training strategy naturally allows aggregation of multiple aligned Gaussian primitives and significantly alleviates the interpolation limitations inherent in single-view pixel-aligned Gaussian Splatting. Furthermore, we incorporate video model priors to perform geometry-aware refinement, enhancing the generation of fine details in wide-viewpoint scenarios and improving the model's capability to capture intricate 3D textures. Extensive experiments demonstrate that our approach not only achieves high-quality, multi-view consistent 3D-aware generation from monocular datasets, but also significantly improves training and inference efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10133v3">Efficient Density Control for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated outstanding performance in novel view synthesis, achieving a balance between rendering quality and real-time performance. 3DGS employs Adaptive Density Control (ADC) to increase the number of Gaussians. However, the clone and split operations within ADC are not sufficiently efficient, impacting optimization speed and detail recovery. Additionally, overfitted Gaussians that affect rendering quality may exist, and the original ADC is unable to remove them. To address these issues, we propose two key innovations: (1) Long-Axis Split, which precisely controls the position, shape, and opacity of child Gaussians to minimize the difference before and after splitting. (2) Recovery-Aware Pruning, which leverages differences in recovery speed after resetting opacity to prune overfitted Gaussians, thereby improving generalization performance. Experimental results show that our method significantly enhances rendering quality. Code is available at https://github.com/XiaoBin2001/EDC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18197v3">Make-It-Animatable: An Efficient Framework for Authoring Animation-Ready 3D Characters</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 CVPR 2025. Project page: https://jasongzy.github.io/Make-It-Animatable/
    </div>
    <details class="paper-abstract">
      3D characters are essential to modern creative industries, but making them animatable often demands extensive manual work in tasks like rigging and skinning. Existing automatic rigging tools face several limitations, including the necessity for manual annotations, rigid skeleton topologies, and limited generalization across diverse shapes and poses. An alternative approach is to generate animatable avatars pre-bound to a rigged template mesh. However, this method often lacks flexibility and is typically limited to realistic human shapes. To address these issues, we present Make-It-Animatable, a novel data-driven method to make any 3D humanoid model ready for character animation in less than one second, regardless of its shapes and poses. Our unified framework generates high-quality blend weights, bones, and pose transformations. By incorporating a particle-based shape autoencoder, our approach supports various 3D representations, including meshes and 3D Gaussian splats. Additionally, we employ a coarse-to-fine representation and a structure-aware modeling strategy to ensure both accuracy and robustness, even for characters with non-standard skeleton structures. We conducted extensive experiments to validate our framework's effectiveness. Compared to existing methods, our approach demonstrates significant improvements in both quality and speed. More demos and code are available at https://jasongzy.github.io/Make-It-Animatable/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08093v1">MVGSR: Multi-View Consistency Gaussian Splatting for Robust Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 project page \url{https://mvgsr.github.io}
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has gained significant attention for its high-quality rendering capabilities, ultra-fast training, and inference speeds. However, when we apply 3DGS to surface reconstruction tasks, especially in environments with dynamic objects and distractors, the method suffers from floating artifacts and color errors due to inconsistency from different viewpoints. To address this challenge, we propose Multi-View Consistency Gaussian Splatting for the domain of Robust Surface Reconstruction (\textbf{MVGSR}), which takes advantage of lightweight Gaussian models and a {heuristics-guided distractor masking} strategy for robust surface reconstruction in non-static environments. Compared to existing methods that rely on MLPs for distractor segmentation strategies, our approach separates distractors from static scene elements by comparing multi-view feature consistency, allowing us to obtain precise distractor masks early in training. Furthermore, we introduce a pruning measure based on multi-view contributions to reset transmittance, effectively reducing floating artifacts. Finally, a multi-view consistency loss is applied to achieve high-quality performance in surface reconstruction tasks. Experimental results demonstrate that MVGSR achieves competitive geometric accuracy and rendering fidelity compared to the state-of-the-art surface reconstruction algorithms. More information is available on our project page (\href{https://mvgsr.github.io}{this url})
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08071v1">GigaSLAM: Large-Scale Monocular SLAM with Hierachical Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Tracking and mapping in large-scale, unbounded outdoor environments using only monocular RGB input presents substantial challenges for existing SLAM systems. Traditional Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) SLAM methods are typically limited to small, bounded indoor settings. To overcome these challenges, we introduce GigaSLAM, the first NeRF/3DGS-based SLAM framework for kilometer-scale outdoor environments, as demonstrated on the KITTI and KITTI 360 datasets. Our approach employs a hierarchical sparse voxel map representation, where Gaussians are decoded by neural networks at multiple levels of detail. This design enables efficient, scalable mapping and high-fidelity viewpoint rendering across expansive, unbounded scenes. For front-end tracking, GigaSLAM utilizes a metric depth model combined with epipolar geometry and PnP algorithms to accurately estimate poses, while incorporating a Bag-of-Words-based loop closure mechanism to maintain robust alignment over long trajectories. Consequently, GigaSLAM delivers high-precision tracking and visually faithful rendering on urban outdoor benchmarks, establishing a robust SLAM solution for large-scale, long-term scenarios, and significantly extending the applicability of Gaussian Splatting SLAM systems to unbounded outdoor environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22070v2">FreeGaussian: Annotation-free Controllable 3D Gaussian Splats with Flow Derivatives</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Reconstructing controllable Gaussian splats from monocular video is a challenging task due to its inherently insufficient constraints. Widely adopted approaches supervise complex interactions with additional masks and control signal annotations, limiting their real-world applications. In this paper, we propose an annotation guidance-free method, dubbed FreeGaussian, that mathematically derives dynamic Gaussian motion from optical flow and camera motion using novel dynamic Gaussian constraints. By establishing a connection between 2D flows and 3D Gaussian dynamic control, our method enables self-supervised optimization and continuity of dynamic Gaussian motions from flow priors. Furthermore, we introduce a 3D spherical vector controlling scheme, which represents the state with a 3D Gaussian trajectory, thereby eliminating the need for complex 1D control signal calculations and simplifying controllable Gaussian modeling. Quantitative and qualitative evaluations on extensive experiments demonstrate the state-of-the-art visual performance and control capability of our method. Project page: https://freegaussian.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12789v2">Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D generation models have opened new possibilities for simulating dynamic 3D object movements and customizing behaviors, yet creating this content remains challenging. Current methods often require manual assignment of precise physical properties for simulations or rely on video generation models to predict them, which is computationally intensive. In this paper, we rethink the usage of multi-modal large language model (MLLM) in physics-based simulation, and present Sim Anything, a physics-based approach that endows static 3D objects with interactive dynamics. We begin with detailed scene reconstruction and object-level 3D open-vocabulary segmentation, progressing to multi-view image in-painting. Inspired by human visual reasoning, we propose MLLM-based Physical Property Perception (MLLM-P3) to predict mean physical properties of objects in a zero-shot manner. Based on the mean values and the object's geometry, the Material Property Distribution Prediction model (MPDP) model then estimates the full distribution, reformulating the problem as probability distribution estimation to reduce computational costs. Finally, we simulate objects in an open-world scene with particles sampled via the Physical-Geometric Adaptive Sampling (PGAS) strategy, efficiently capturing complex deformations and significantly reducing computational costs. Extensive experiments and user studies demonstrate our Sim Anything achieves more realistic motion than state-of-the-art methods within 2 minutes on a single GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07946v1">7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Real-time rendering of dynamic scenes with view-dependent effects remains a fundamental challenge in computer graphics. While recent advances in Gaussian Splatting have shown promising results separately handling dynamic scenes (4DGS) and view-dependent effects (6DGS), no existing method unifies these capabilities while maintaining real-time performance. We present 7D Gaussian Splatting (7DGS), a unified framework representing scene elements as seven-dimensional Gaussians spanning position (3D), time (1D), and viewing direction (3D). Our key contribution is an efficient conditional slicing mechanism that transforms 7D Gaussians into view- and time-conditioned 3D Gaussians, maintaining compatibility with existing 3D Gaussian Splatting pipelines while enabling joint optimization. Experiments demonstrate that 7DGS outperforms prior methods by up to 7.36 dB in PSNR while achieving real-time rendering (401 FPS) on challenging dynamic scenes with complex view-dependent effects. The project page is: https://gaozhongpai.github.io/7dgs/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04974v3">6DGS: Enhanced Direction-Aware Gaussian Splatting for Volumetric Rendering</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Accepted by ICLR2025
    </div>
    <details class="paper-abstract">
      Novel view synthesis has advanced significantly with the development of neural radiance fields (NeRF) and 3D Gaussian splatting (3DGS). However, achieving high quality without compromising real-time rendering remains challenging, particularly for physically-based ray tracing with view-dependent effects. Recently, N-dimensional Gaussians (N-DG) introduced a 6D spatial-angular representation to better incorporate view-dependent effects, but the Gaussian representation and control scheme are sub-optimal. In this paper, we revisit 6D Gaussians and introduce 6D Gaussian Splatting (6DGS), which enhances color and opacity representations and leverages the additional directional information in the 6D space for optimized Gaussian control. Our approach is fully compatible with the 3DGS framework and significantly improves real-time radiance field rendering by better modeling view-dependent effects and fine details. Experiments demonstrate that 6DGS significantly outperforms 3DGS and N-DG, achieving up to a 15.73 dB improvement in PSNR with a reduction of 66.5% Gaussian points compared to 3DGS. The project page is: https://gaozhongpai.github.io/6dgs/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07476v1">SOGS: Second-Order Anchor for Advanced 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 Accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      Anchor-based 3D Gaussian splatting (3D-GS) exploits anchor features in 3D Gaussian prediction, which has achieved impressive 3D rendering quality with reduced Gaussian redundancy. On the other hand, it often encounters the dilemma among anchor features, model size, and rendering quality - large anchor features lead to large 3D models and high-quality rendering whereas reducing anchor features degrades Gaussian attribute prediction which leads to clear artifacts in the rendered textures and geometries. We design SOGS, an anchor-based 3D-GS technique that introduces second-order anchors to achieve superior rendering quality and reduced anchor features and model size simultaneously. Specifically, SOGS incorporates covariance-based second-order statistics and correlation across feature dimensions to augment features within each anchor, compensating for the reduced feature size and improving rendering quality effectively. In addition, it introduces a selective gradient loss to enhance the optimization of scene textures and scene geometries, leading to high-quality rendering with small anchor features. Extensive experiments over multiple widely adopted benchmarks show that SOGS achieves superior rendering quality in novel view synthesis with clearly reduced model size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07446v1">EigenGS Representation: From Eigenspace to Gaussian Image Space</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Principal Component Analysis (PCA), a classical dimensionality reduction technique, and 2D Gaussian representation, an adaptation of 3D Gaussian Splatting for image representation, offer distinct approaches to modeling visual data. We present EigenGS, a novel method that bridges these paradigms through an efficient transformation pipeline connecting eigenspace and image-space Gaussian representations. Our approach enables instant initialization of Gaussian parameters for new images without requiring per-image optimization from scratch, dramatically accelerating convergence. EigenGS introduces a frequency-aware learning mechanism that encourages Gaussians to adapt to different scales, effectively modeling varied spatial frequencies and preventing artifacts in high-resolution reconstruction. Extensive experiments demonstrate that EigenGS not only achieves superior reconstruction quality compared to direct 2D Gaussian fitting but also reduces necessary parameter count and training time. The results highlight EigenGS's effectiveness and generalization ability across images with varying resolutions and diverse categories, making Gaussian-based image representation both high-quality and viable for real-time applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08129v3">Efficient Perspective-Correct 3D Gaussian Splatting Using Hybrid Transparency</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 Project page: https://fhahlbohm.github.io/htgs/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splats (3DGS) have proven a versatile rendering primitive, both for inverse rendering as well as real-time exploration of scenes. In these applications, coherence across camera frames and multiple views is crucial, be it for robust convergence of a scene reconstruction or for artifact-free fly-throughs. Recent work started mitigating artifacts that break multi-view coherence, including popping artifacts due to inconsistent transparency sorting and perspective-correct outlines of (2D) splats. At the same time, real-time requirements forced such implementations to accept compromises in how transparency of large assemblies of 3D Gaussians is resolved, in turn breaking coherence in other ways. In our work, we aim at achieving maximum coherence, by rendering fully perspective-correct 3D Gaussians while using a high-quality approximation of accurate blending, hybrid transparency, on a per-pixel level, in order to retain real-time frame rates. Our fast and perspectively accurate approach for evaluation of 3D Gaussians does not require matrix inversions, thereby ensuring numerical stability and eliminating the need for special handling of degenerate splats, and the hybrid transparency formulation for blending maintains similar quality as fully resolved per-pixel transparencies at a fraction of the rendering costs. We further show that each of these two components can be independently integrated into Gaussian splatting systems. In combination, they achieve up to 2$\times$ higher frame rates, 2$\times$ faster optimization, and equal or better image quality with fewer rendering artifacts compared to traditional 3DGS on common benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13654v2">GAGS: Granularity-Aware Feature Distillation for Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 Project page: https://pz0826.github.io/GAGS-Webpage/
    </div>
    <details class="paper-abstract">
      3D open-vocabulary scene understanding, which accurately perceives complex semantic properties of objects in space, has gained significant attention in recent years. In this paper, we propose GAGS, a framework that distills 2D CLIP features into 3D Gaussian splatting, enabling open-vocabulary queries for renderings on arbitrary viewpoints. The main challenge of distilling 2D features for 3D fields lies in the multiview inconsistency of extracted 2D features, which provides unstable supervision for the 3D feature field. GAGS addresses this challenge with two novel strategies. First, GAGS associates the prompt point density of SAM with the camera distances, which significantly improves the multiview consistency of segmentation results. Second, GAGS further decodes a granularity factor to guide the distillation process and this granularity factor can be learned in a unsupervised manner to only select the multiview consistent 2D features in the distillation process. Experimental results on two datasets demonstrate significant performance and stability improvements of GAGS in visual grounding and semantic segmentation, with an inference speed 2$\times$ faster than baseline methods. The code and additional results are available at https://pz0826.github.io/GAGS-Webpage/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08508v4">BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      We present billboard Splatting (BBSplat) - a novel approach for novel view synthesis based on textured geometric primitives. BBSplat represents the scene as a set of optimizable textured planar primitives with learnable RGB textures and alpha-maps to control their shape. BBSplat primitives can be used in any Gaussian Splatting pipeline as drop-in replacements for Gaussians. The proposed primitives close the rendering quality gap between 2D and 3D Gaussian Splatting (GS), enabling the accurate extraction of 3D mesh as in the 2DGS framework. Additionally, the explicit nature of planar primitives enables the use of the ray-tracing effects in rasterization. Our novel regularization term encourages textures to have a sparser structure, enabling an efficient compression that leads to a reduction in the storage space of the model up to x17 times compared to 3DGS. Our experiments show the efficiency of BBSplat on standard datasets of real indoor and outdoor scenes such as Tanks&Temples, DTU, and Mip-NeRF-360. Namely, we achieve a state-of-the-art PSNR of 29.72 for DTU at Full HD resolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07191v1">All That Glitters Is Not Gold: Key-Secured 3D Secrets within 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have revolutionized scene reconstruction, opening new possibilities for 3D steganography by hiding 3D secrets within 3D covers. The key challenge in steganography is ensuring imperceptibility while maintaining high-fidelity reconstruction. However, existing methods often suffer from detectability risks and utilize only suboptimal 3DGS features, limiting their full potential. We propose a novel end-to-end key-secured 3D steganography framework (KeySS) that jointly optimizes a 3DGS model and a key-secured decoder for secret reconstruction. Our approach reveals that Gaussian features contribute unequally to secret hiding. The framework incorporates a key-controllable mechanism enabling multi-secret hiding and unauthorized access prevention, while systematically exploring optimal feature update to balance fidelity and security. To rigorously evaluate steganographic imperceptibility beyond conventional 2D metrics, we introduce 3D-Sinkhorn distance analysis, which quantifies distributional differences between original and steganographic Gaussian parameters in the representation space. Extensive experiments demonstrate that our method achieves state-of-the-art performance in both cover and secret reconstruction while maintaining high security levels, advancing the field of 3D steganography. Code is available at https://github.com/RY-Paper/KeySS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07190v1">Multi-Modal 3D Mesh Reconstruction from Images and Text</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 under review
    </div>
    <details class="paper-abstract">
      6D object pose estimation for unseen objects is essential in robotics but traditionally relies on trained models that require large datasets, high computational costs, and struggle to generalize. Zero-shot approaches eliminate the need for training but depend on pre-existing 3D object models, which are often impractical to obtain. To address this, we propose a language-guided few-shot 3D reconstruction method, reconstructing a 3D mesh from few input images. In the proposed pipeline, receives a set of input images and a language query. A combination of GroundingDINO and Segment Anything Model outputs segmented masks from which a sparse point cloud is reconstructed with VGGSfM. Subsequently, the mesh is reconstructed with the Gaussian Splatting method SuGAR. In a final cleaning step, artifacts are removed, resulting in the final 3D mesh of the queried object. We evaluate the method in terms of accuracy and quality of the geometry and texture. Furthermore, we study the impact of imaging conditions such as viewing angle, number of input images, and image overlap on 3D object reconstruction quality, efficiency, and computational scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14957v2">Dream to Manipulate: Compositional World Models Empowering Robot Imitation Learning with Imagination</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      A world model provides an agent with a representation of its environment, enabling it to predict the causal consequences of its actions. Current world models typically cannot directly and explicitly imitate the actual environment in front of a robot, often resulting in unrealistic behaviors and hallucinations that make them unsuitable for real-world robotics applications. To overcome those challenges, we propose to rethink robot world models as learnable digital twins. We introduce DreMa, a new approach for constructing digital twins automatically using learned explicit representations of the real world and its dynamics, bridging the gap between traditional digital twins and world models. DreMa replicates the observed world and its structure by integrating Gaussian Splatting and physics simulators, allowing robots to imagine novel configurations of objects and to predict the future consequences of robot actions thanks to its compositionality. We leverage this capability to generate new data for imitation learning by applying equivariant transformations to a small set of demonstrations. Our evaluations across various settings demonstrate significant improvements in accuracy and robustness by incrementing actions and object distributions, reducing the data needed to learn a policy and improving the generalization of the agents. As a highlight, we show that a real Franka Emika Panda robot, powered by DreMa's imagination, can successfully learn novel physical tasks from just a single example per task variation (one-shot policy learning). Our project page can be found in: https://dreamtomanipulate.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06390v3">SplatFormer: Point Transformer for Robust 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently transformed photorealistic reconstruction, achieving high visual fidelity and real-time performance. However, rendering quality significantly deteriorates when test views deviate from the camera angles used during training, posing a major challenge for applications in immersive free-viewpoint rendering and navigation. In this work, we conduct a comprehensive evaluation of 3DGS and related novel view synthesis methods under out-of-distribution (OOD) test camera scenarios. By creating diverse test cases with synthetic and real-world datasets, we demonstrate that most existing methods, including those incorporating various regularization techniques and data-driven priors, struggle to generalize effectively to OOD views. To address this limitation, we introduce SplatFormer, the first point transformer model specifically designed to operate on Gaussian splats. SplatFormer takes as input an initial 3DGS set optimized under limited training views and refines it in a single forward pass, effectively removing potential artifacts in OOD test views. To our knowledge, this is the first successful application of point transformers directly on 3DGS sets, surpassing the limitations of previous multi-scene training methods, which could handle only a restricted number of input views during inference. Our model significantly improves rendering quality under extreme novel views, achieving state-of-the-art performance in these challenging scenarios and outperforming various 3DGS regularization techniques, multi-scene models tailored for sparse view synthesis, and diffusion-based frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07000v1">Frequency-Aware Density Control via Reparameterization for High-Quality Rendering of 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 Accepted to AAAI2025
    </div>
    <details class="paper-abstract">
      By adaptively controlling the density and generating more Gaussians in regions with high-frequency information, 3D Gaussian Splatting (3DGS) can better represent scene details. From the signal processing perspective, representing details usually needs more Gaussians with relatively smaller scales. However, 3DGS currently lacks an explicit constraint linking the density and scale of 3D Gaussians across the domain, leading to 3DGS using improper-scale Gaussians to express frequency information, resulting in the loss of accuracy. In this paper, we propose to establish a direct relation between density and scale through the reparameterization of the scaling parameters and ensure the consistency between them via explicit constraints (i.e., density responds well to changes in frequency). Furthermore, we develop a frequency-aware density control strategy, consisting of densification and deletion, to improve representation quality with fewer Gaussians. A dynamic threshold encourages densification in high-frequency regions, while a scale-based filter deletes Gaussians with improper scale. Experimental results on various datasets demonstrate that our method outperforms existing state-of-the-art methods quantitatively and qualitatively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05168v2">SeeLe: A Unified Acceleration Framework for Real-Time Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a crucial rendering technique for many real-time applications. However, the limited hardware resources on today's mobile platforms hinder these applications, as they struggle to achieve real-time performance. In this paper, we propose SeeLe, a general framework designed to accelerate the 3DGS pipeline for resource-constrained mobile devices. Specifically, we propose two GPU-oriented techniques: hybrid preprocessing and contribution-aware rasterization. Hybrid preprocessing alleviates the GPU compute and memory pressure by reducing the number of irrelevant Gaussians during rendering. The key is to combine our view-dependent scene representation with online filtering. Meanwhile, contribution-aware rasterization improves the GPU utilization at the rasterization stage by prioritizing Gaussians with high contributions while reducing computations for those with low contributions. Both techniques can be seamlessly integrated into existing 3DGS pipelines with minimal fine-tuning. Collectively, our framework achieves 2.6$\times$ speedup and 32.3\% model reduction while achieving superior rendering quality compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12518v4">Hier-SLAM: Scaling-up Semantics in SLAM with a Hierarchically Categorical Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 Accepted for publication at ICRA 2025. Code is available at https://github.com/LeeBY68/Hier-SLAM
    </div>
    <details class="paper-abstract">
      We propose Hier-SLAM, a semantic 3D Gaussian Splatting SLAM method featuring a novel hierarchical categorical representation, which enables accurate global 3D semantic mapping, scaling-up capability, and explicit semantic label prediction in the 3D world. The parameter usage in semantic SLAM systems increases significantly with the growing complexity of the environment, making it particularly challenging and costly for scene understanding. To address this problem, we introduce a novel hierarchical representation that encodes semantic information in a compact form into 3D Gaussian Splatting, leveraging the capabilities of large language models (LLMs). We further introduce a novel semantic loss designed to optimize hierarchical semantic information through both inter-level and cross-level optimization. Furthermore, we enhance the whole SLAM system, resulting in improved tracking and mapping performance. Our \MethodName{} outperforms existing dense SLAM methods in both mapping and tracking accuracy, while achieving a 2x operation speed-up. Additionally, it achieves on-par semantic rendering performance compared to existing methods while significantly reducing storage and training time requirements. Rendering FPS impressively reaches 2,000 with semantic information and 3,000 without it. Most notably, it showcases the capability of handling the complex real-world scene with more than 500 semantic classes, highlighting its valuable scaling-up capability. The open-source code is available at https://github.com/LeeBY68/Hier-SLAM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05152v2">GSplatVNM: Point-of-View Synthesis for Visual Navigation Models Using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach to image-goal navigation by integrating 3D Gaussian Splatting (3DGS) with Visual Navigation Models (VNMs), a method we refer to as GSplatVNM. VNMs offer a promising paradigm for image-goal navigation by guiding a robot through a sequence of point-of-view images without requiring metrical localization or environment-specific training. However, constructing a dense and traversable sequence of target viewpoints from start to goal remains a central challenge, particularly when the available image database is sparse. To address these challenges, we propose a 3DGS-based viewpoint synthesis framework for VNMs that synthesizes intermediate viewpoints to seamlessly bridge gaps in sparse data while significantly reducing storage overhead. Experimental results in a photorealistic simulator demonstrate that our approach not only enhances navigation efficiency but also exhibits robustness under varying levels of image database sparsity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06900v1">DirectTriGS: Triplane-based Gaussian Splatting Field Representation for 3D Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 Accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      We present DirectTriGS, a novel framework designed for 3D object generation with Gaussian Splatting (GS). GS-based rendering for 3D content has gained considerable attention recently. However, there has been limited exploration in directly generating 3D Gaussians compared to traditional generative modeling approaches. The main challenge lies in the complex data structure of GS represented by discrete point clouds with multiple channels. To overcome this challenge, we propose employing the triplane representation, which allows us to represent Gaussian Splatting as an image-like continuous field. This representation effectively encodes both the geometry and texture information, enabling smooth transformation back to Gaussian point clouds and rendering into images by a TriRenderer, with only 2D supervisions. The proposed TriRenderer is fully differentiable, so that the rendering loss can supervise both texture and geometry encoding. Furthermore, the triplane representation can be compressed using a Variational Autoencoder (VAE), which can subsequently be utilized in latent diffusion to generate 3D objects. The experiments demonstrate that the proposed generation framework can produce high-quality 3D object geometry and rendering results in the text-to-3D task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06859v1">ActiveInitSplat: How Active Image Selection Helps Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Gaussian splatting (GS) along with its extensions and variants provides outstanding performance in real-time scene rendering while meeting reduced storage demands and computational efficiency. While the selection of 2D images capturing the scene of interest is crucial for the proper initialization and training of GS, hence markedly affecting the rendering performance, prior works rely on passively and typically densely selected 2D images. In contrast, this paper proposes `ActiveInitSplat', a novel framework for active selection of training images for proper initialization and training of GS. ActiveInitSplat relies on density and occupancy criteria of the resultant 3D scene representation from the selected 2D images, to ensure that the latter are captured from diverse viewpoints leading to better scene coverage and that the initialized Gaussian functions are well aligned with the actual 3D structure. Numerical tests on well-known simulated and real environments demonstrate the merits of ActiveInitSplat resulting in significant GS rendering performance improvement over passive GS baselines, in the widely adopted LPIPS, SSIM, and PSNR metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00486v4">CaRtGS: Computational Alignment for Real-Time Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 Accepted by IEEE Robotics and Automation Letters (RA-L)
    </div>
    <details class="paper-abstract">
      Simultaneous Localization and Mapping (SLAM) is pivotal in robotics, with photorealistic scene reconstruction emerging as a key challenge. To address this, we introduce Computational Alignment for Real-Time Gaussian Splatting SLAM (CaRtGS), a novel method enhancing the efficiency and quality of photorealistic scene reconstruction in real-time environments. Leveraging 3D Gaussian Splatting (3DGS), CaRtGS achieves superior rendering quality and processing speed, which is crucial for scene photorealistic reconstruction. Our approach tackles computational misalignment in Gaussian Splatting SLAM (GS-SLAM) through an adaptive strategy that enhances optimization iterations, addresses long-tail optimization, and refines densification. Experiments on Replica, TUM-RGBD, and VECtor datasets demonstrate CaRtGS's effectiveness in achieving high-fidelity rendering with fewer Gaussian primitives. This work propels SLAM towards real-time, photorealistic dense rendering, significantly advancing photorealistic scene representation. For the benefit of the research community, we release the code and accompanying videos on our project website: https://dapengfeng.github.io/cartgs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07819v1">POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      In this paper, we present a novel algorithm for quantifying uncertainty and information gained within 3D Gaussian Splatting (3D-GS) through P-Optimality. While 3D-GS has proven to be a useful world model with high-quality rasterizations, it does not natively quantify uncertainty. Quantifying uncertainty in parameters of 3D-GS is necessary to understand the information gained from acquiring new images as in active perception, or identify redundant images which can be removed from memory due to resource constraints in online 3D-GS SLAM. We propose to quantify uncertainty and information gain in 3D-GS by reformulating the problem through the lens of optimal experimental design, which is a classical solution to measuring information gain. By restructuring information quantification of 3D-GS through optimal experimental design, we arrive at multiple solutions, of which T-Optimality and D-Optimality perform the best quantitatively and qualitatively as measured on two popular datasets. Additionally, we propose a block diagonal approximation of the 3D-GS uncertainty, which provides a measure of correlation for computing more accurate information gain, at the expense of a greater computation cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06762v1">Gaussian RBFNet: Gaussian Radial Basis Functions for Fast and Accurate Representation and Reconstruction of Neural Fields</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 Our code is available at https://grbfnet.github.io/
    </div>
    <details class="paper-abstract">
      Neural fields such as DeepSDF and Neural Radiance Fields have recently revolutionized novel-view synthesis and 3D reconstruction from RGB images and videos. However, achieving high-quality representation, reconstruction, and rendering requires deep neural networks, which are slow to train and evaluate. Although several acceleration techniques have been proposed, they often trade off speed for memory. Gaussian splatting-based methods, on the other hand, accelerate the rendering time but remain costly in terms of training speed and memory needed to store the parameters of a large number of Gaussians. In this paper, we introduce a novel neural representation that is fast, both at training and inference times, and lightweight. Our key observation is that the neurons used in traditional MLPs perform simple computations (a dot product followed by ReLU activation) and thus one needs to use either wide and deep MLPs or high-resolution and high-dimensional feature grids to parameterize complex nonlinear functions. We show in this paper that by replacing traditional neurons with Radial Basis Function (RBF) kernels, one can achieve highly accurate representation of 2D (RGB images), 3D (geometry), and 5D (radiance fields) signals with just a single layer of such neurons. The representation is highly parallelizable, operates on low-resolution feature grids, and is compact and memory-efficient. We demonstrate that the proposed novel representation can be trained for 3D geometry representation in less than 15 seconds and for novel view synthesis in less than 15 mins. At runtime, it can synthesize novel views at more than 60 fps without sacrificing quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06744v1">CoDa-4DGS: Dynamic Gaussian Splatting with Context and Deformation Awareness for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-03-09
    </div>
    <details class="paper-abstract">
      Dynamic scene rendering opens new avenues in autonomous driving by enabling closed-loop simulations with photorealistic data, which is crucial for validating end-to-end algorithms. However, the complex and highly dynamic nature of traffic environments presents significant challenges in accurately rendering these scenes. In this paper, we introduce a novel 4D Gaussian Splatting (4DGS) approach, which incorporates context and temporal deformation awareness to improve dynamic scene rendering. Specifically, we employ a 2D semantic segmentation foundation model to self-supervise the 4D semantic features of Gaussians, ensuring meaningful contextual embedding. Simultaneously, we track the temporal deformation of each Gaussian across adjacent frames. By aggregating and encoding both semantic and temporal deformation features, each Gaussian is equipped with cues for potential deformation compensation within 3D space, facilitating a more precise representation of dynamic scenes. Experimental results show that our method improves 4DGS's ability to capture fine details in dynamic scene rendering for autonomous driving and outperforms other self-supervised methods in 4D reconstruction and novel view synthesis. Furthermore, CoDa-4DGS deforms semantic features with each Gaussian, enabling broader applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06740v1">D3DR: Lighting-Aware Object Insertion in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-09
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has become a popular technique for various 3D Computer Vision tasks, including novel view synthesis, scene reconstruction, and dynamic scene rendering. However, the challenge of natural-looking object insertion, where the object's appearance seamlessly matches the scene, remains unsolved. In this work, we propose a method, dubbed D3DR, for inserting a 3DGS-parametrized object into 3DGS scenes while correcting its lighting, shadows, and other visual artifacts to ensure consistency, a problem that has not been successfully addressed before. We leverage advances in diffusion models, which, trained on real-world data, implicitly understand correct scene lighting. After inserting the object, we optimize a diffusion-based Delta Denoising Score (DDS)-inspired objective to adjust its 3D Gaussian parameters for proper lighting correction. Utilizing diffusion model personalization techniques to improve optimization quality, our approach ensures seamless object insertion and natural appearance. Finally, we demonstrate the method's effectiveness by comparing it to existing approaches, achieving 0.5 PSNR and 0.15 SSIM improvements in relighting quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06677v1">REArtGS: Reconstructing and Generating Articulated Objects via 3D Gaussian Splatting with Geometric and Motion Constraints</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 11pages, 6 figures
    </div>
    <details class="paper-abstract">
      Articulated objects, as prevalent entities in human life, their 3D representations play crucial roles across various applications. However, achieving both high-fidelity textured surface reconstruction and dynamic generation for articulated objects remains challenging for existing methods. In this paper, we present REArtGS, a novel framework that introduces additional geometric and motion constraints to 3D Gaussian primitives, enabling high-quality textured surface reconstruction and generation for articulated objects. Specifically, given multi-view RGB images of arbitrary two states of articulated objects, we first introduce an unbiased Signed Distance Field (SDF) guidance to regularize Gaussian opacity fields, enhancing geometry constraints and improving surface reconstruction quality. Then we establish deformable fields for 3D Gaussians constrained by the kinematic structures of articulated objects, achieving unsupervised generation of surface meshes in unseen states. Extensive experiments on both synthetic and real datasets demonstrate our approach achieves high-quality textured surface reconstruction for given states, and enables high-fidelity surface generation for unseen states. Codes will be released within the next four months.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06617v1">Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 Tech Report
    </div>
    <details class="paper-abstract">
      Arbitrary-scale super-resolution (ASSR) aims to reconstruct high-resolution (HR) images from low-resolution (LR) inputs with arbitrary upsampling factors using a single model, addressing the limitations of traditional SR methods constrained to fixed-scale factors (\textit{e.g.}, $\times$ 2). Recent advances leveraging implicit neural representation (INR) have achieved great progress by modeling coordinate-to-pixel mappings. However, the efficiency of these methods may suffer from repeated upsampling and decoding, while their reconstruction fidelity and quality are constrained by the intrinsic representational limitations of coordinate-based functions. To address these challenges, we propose a novel ContinuousSR framework with a Pixel-to-Gaussian paradigm, which explicitly reconstructs 2D continuous HR signals from LR images using Gaussian Splatting. This approach eliminates the need for time-consuming upsampling and decoding, enabling extremely fast arbitrary-scale super-resolution. Once the Gaussian field is built in a single pass, ContinuousSR can perform arbitrary-scale rendering in just 1ms per scale. Our method introduces several key innovations. Through statistical ana
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06587v1">Introducing Unbiased Depth into 2D Gaussian Splatting for High-accuracy Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-09
    </div>
    <details class="paper-abstract">
      Recently, 2D Gaussian Splatting (2DGS) has demonstrated superior geometry reconstruction quality than the popular 3DGS by using 2D surfels to approximate thin surfaces. However, it falls short when dealing with glossy surfaces, resulting in visible holes in these areas. We found the reflection discontinuity causes the issue. To fit the jump from diffuse to specular reflection at different viewing angles, depth bias is introduced in the optimized Gaussian primitives. To address that, we first replace the depth distortion loss in 2DGS with a novel depth convergence loss, which imposes a strong constraint on depth continuity. Then, we rectified the depth criterion in determining the actual surface, which fully accounts for all the intersecting Gaussians along the ray. Qualitative and quantitative evaluations across various datasets reveal that our method significantly improves reconstruction quality, with more complete and accurate surfaces than 2DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14514v5">NexusSplats: Efficient 3D Gaussian Splatting in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 Project page: https://nexus-splats.github.io/
    </div>
    <details class="paper-abstract">
      Photorealistic 3D reconstruction of unstructured real-world scenes remains challenging due to complex illumination variations and transient occlusions. Existing methods based on Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) struggle with inefficient light decoupling and structure-agnostic occlusion handling. To address these limitations, we propose NexusSplats, an approach tailored for efficient and high-fidelity 3D scene reconstruction under complex lighting and occlusion conditions. In particular, NexusSplats leverages a hierarchical light decoupling strategy that performs centralized appearance learning, efficiently and effectively decoupling varying lighting conditions. Furthermore, a structure-aware occlusion handling mechanism is developed, establishing a nexus between 3D and 2D structures for fine-grained occlusion handling. Experimental results demonstrate that NexusSplats achieves state-of-the-art rendering quality and reduces the number of total parameters by 65.4\%, leading to 2.7$\times$ faster reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06462v1">StructGS: Adaptive Spherical Harmonics and Rendering Enhancements for Superior 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-09
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D reconstruction coupled with neural rendering techniques have greatly improved the creation of photo-realistic 3D scenes, influencing both academic research and industry applications. The technique of 3D Gaussian Splatting and its variants incorporate the strengths of both primitive-based and volumetric representations, achieving superior rendering quality. While 3D Geometric Scattering (3DGS) and its variants have advanced the field of 3D representation, they fall short in capturing the stochastic properties of non-local structural information during the training process. Additionally, the initialisation of spherical functions in 3DGS-based methods often fails to engage higher-order terms in early training rounds, leading to unnecessary computational overhead as training progresses. Furthermore, current 3DGS-based approaches require training on higher resolution images to render higher resolution outputs, significantly increasing memory demands and prolonging training durations. We introduce StructGS, a framework that enhances 3D Gaussian Splatting (3DGS) for improved novel-view synthesis in 3D reconstruction. StructGS innovatively incorporates a patch-based SSIM loss, dynamic spherical harmonics initialisation and a Multi-scale Residual Network (MSRN) to address the above-mentioned limitations, respectively. Our framework significantly reduces computational redundancy, enhances detail capture and supports high-resolution rendering from low-resolution inputs. Experimentally, StructGS demonstrates superior performance over state-of-the-art (SOTA) models, achieving higher quality and more detailed renderings with fewer artifacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04680v4">Next Best Sense: Guiding Vision and Touch with FisherRF for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 To appear in International Conference on Robotics and Automation (ICRA) 2025
    </div>
    <details class="paper-abstract">
      We propose a framework for active next best view and touch selection for robotic manipulators using 3D Gaussian Splatting (3DGS). 3DGS is emerging as a useful explicit 3D scene representation for robotics, as it has the ability to represent scenes in a both photorealistic and geometrically accurate manner. However, in real-world, online robotic scenes where the number of views is limited given efficiency requirements, random view selection for 3DGS becomes impractical as views are often overlapping and redundant. We address this issue by proposing an end-to-end online training and active view selection pipeline, which enhances the performance of 3DGS in few-view robotics settings. We first elevate the performance of few-shot 3DGS with a novel semantic depth alignment method using Segment Anything Model 2 (SAM2) that we supplement with Pearson depth and surface normal loss to improve color and depth reconstruction of real-world scenes. We then extend FisherRF, a next-best-view selection method for 3DGS, to select views and touch poses based on depth uncertainty. We perform online view selection on a real robot system during live 3DGS training. We motivate our improvements to few-shot GS scenes, and extend depth-based FisherRF to them, where we demonstrate both qualitative and quantitative improvements on challenging robot scenes. For more information, please see our project page at https://arm.stanford.edu/next-best-sense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14384v3">Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 A novel one-stage 3DGS-based diffusion for 3D object generation and scene reconstruction from a single view in ~6 seconds
    </div>
    <details class="paper-abstract">
      Existing feedforward image-to-3D methods mainly rely on 2D multi-view diffusion models that cannot guarantee 3D consistency. These methods easily collapse when changing the prompt view direction and mainly handle object-centric cases. In this paper, we propose a novel single-stage 3D diffusion model, DiffusionGS, for object generation and scene reconstruction from a single view. DiffusionGS directly outputs 3D Gaussian point clouds at each timestep to enforce view consistency and allow the model to generate robustly given prompt views of any directions, beyond object-centric inputs. Plus, to improve the capability and generality of DiffusionGS, we scale up 3D training data by developing a scene-object mixed training strategy. Experiments show that DiffusionGS yields improvements of 2.20 dB/23.25 and 1.34 dB/19.16 in PSNR/FID for objects and scenes than the state-of-the-art methods, without depth estimator. Plus, our method enjoys over 5$\times$ faster speed ($\sim$6s on an A100 GPU). Our Project page at https://caiyuanhao1998.github.io/project/DiffusionGS/ shows the video and interactive results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06271v1">SplatTalk: 3D VQA with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      Language-guided 3D scene understanding is important for advancing applications in robotics, AR/VR, and human-computer interaction, enabling models to comprehend and interact with 3D environments through natural language. While 2D vision-language models (VLMs) have achieved remarkable success in 2D VQA tasks, progress in the 3D domain has been significantly slower due to the complexity of 3D data and the high cost of manual annotations. In this work, we introduce SplatTalk, a novel method that uses a generalizable 3D Gaussian Splatting (3DGS) framework to produce 3D tokens suitable for direct input into a pretrained LLM, enabling effective zero-shot 3D visual question answering (3D VQA) for scenes with only posed images. During experiments on multiple benchmarks, our approach outperforms both 3D models trained specifically for the task and previous 2D-LMM-based models utilizing only images (our setting), while achieving competitive performance with state-of-the-art 3D LMMs that additionally utilize 3D inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01807v2">Occam's LGS: An Efficient Approach for Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 Project Page: https://insait-institute.github.io/OccamLGS/
    </div>
    <details class="paper-abstract">
      TL;DR: Gaussian Splatting is a widely adopted approach for 3D scene representation, offering efficient, high-quality reconstruction and rendering. A key reason for its success is the simplicity of representing scenes with sets of Gaussians, making it interpretable and adaptable. To enhance understanding beyond visual representation, recent approaches extend Gaussian Splatting with semantic vision-language features, enabling open-set tasks. Typically, these language features are aggregated from multiple 2D views, however, existing methods rely on cumbersome techniques, resulting in high computational costs and longer training times. In this work, we show that the complicated pipelines for language 3D Gaussian Splatting are simply unnecessary. Instead, we follow a probabilistic formulation of Language Gaussian Splatting and apply Occam's razor to the task at hand, leading to a highly efficient weighted multi-view feature aggregation technique. Doing so offers us state-of-the-art results with a speed-up of two orders of magnitude without any compression, allowing for easy scene manipulation. Project Page: https://insait-institute.github.io/OccamLGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06235v1">StreamGS: Online Generalizable Gaussian Splatting Reconstruction for Unposed Image Streams</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      The advent of 3D Gaussian Splatting (3DGS) has advanced 3D scene reconstruction and novel view synthesis. With the growing interest of interactive applications that need immediate feedback, online 3DGS reconstruction in real-time is in high demand. However, none of existing methods yet meet the demand due to three main challenges: the absence of predetermined camera parameters, the need for generalizable 3DGS optimization, and the necessity of reducing redundancy. We propose StreamGS, an online generalizable 3DGS reconstruction method for unposed image streams, which progressively transform image streams to 3D Gaussian streams by predicting and aggregating per-frame Gaussians. Our method overcomes the limitation of the initial point reconstruction \cite{dust3r} in tackling out-of-domain (OOD) issues by introducing a content adaptive refinement. The refinement enhances cross-frame consistency by establishing reliable pixel correspondences between adjacent frames. Such correspondences further aid in merging redundant Gaussians through cross-frame feature aggregation. The density of Gaussians is thereby reduced, empowering online reconstruction by significantly lowering computational and memory costs. Extensive experiments on diverse datasets have demonstrated that StreamGS achieves quality on par with optimization-based approaches but does so 150 times faster, and exhibits superior generalizability in handling OOD scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00155v2">T-3DGS: Removing Transient Objects for 3D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 Project website at https://transient-3dgs.github.io/
    </div>
    <details class="paper-abstract">
      Transient objects in video sequences can significantly degrade the quality of 3D scene reconstructions. To address this challenge, we propose T-3DGS, a novel framework that robustly filters out transient distractors during 3D reconstruction using Gaussian Splatting. Our framework consists of two steps. First, we employ an unsupervised classification network that distinguishes transient objects from static scene elements by leveraging their distinct training dynamics within the reconstruction process. Second, we refine these initial detections by integrating an off-the-shelf segmentation method with a bidirectional tracking module, which together enhance boundary accuracy and temporal coherence. Evaluations on both sparsely and densely captured video datasets demonstrate that T-3DGS significantly outperforms state-of-the-art approaches, enabling high-fidelity 3D reconstructions in challenging, real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13610v2">Video2BEV: Transforming Drone Videos to BEVs for Video-based Geo-localization</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      Existing approaches to drone visual geo-localization predominantly adopt the image-based setting, where a single drone-view snapshot is matched with images from other platforms. Such task formulation, however, underutilizes the inherent video output of the drone and is sensitive to occlusions and viewpoint disparity. To address these limitations, we formulate a new video-based drone geo-localization task and propose the Video2BEV paradigm. This paradigm transforms the video into a Bird's Eye View (BEV), simplifying the subsequent \textbf{inter-platform} matching process. In particular, we employ Gaussian Splatting to reconstruct a 3D scene and obtain the BEV projection. Different from the existing transform methods, \eg, polar transform, our BEVs preserve more fine-grained details without significant distortion. To facilitate the discriminative \textbf{intra-platform} representation learning, our Video2BEV paradigm also incorporates a diffusion-based module for generating hard negative samples. To validate our approach, we introduce UniV, a new video-based geo-localization dataset that extends the image-based University-1652 dataset. UniV features flight paths at $30^\circ$ and $45^\circ$ elevation angles with increased frame rates of up to 10 frames per second (FPS). Extensive experiments on the UniV dataset show that our Video2BEV paradigm achieves competitive recall rates and outperforms conventional video-based methods. Compared to other competitive methods, our proposed approach exhibits robustness at lower elevations with more occlusions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06179v1">ForestSplats: Deformable transient field for Gaussian Splatting in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3D-GS) has emerged, showing real-time rendering speeds and high-quality results in static scenes. Although 3D-GS shows effectiveness in static scenes, their performance significantly degrades in real-world environments due to transient objects, lighting variations, and diverse levels of occlusion. To tackle this, existing methods estimate occluders or transient elements by leveraging pre-trained models or integrating additional transient field pipelines. However, these methods still suffer from two defects: 1) Using semantic features from the Vision Foundation model (VFM) causes additional computational costs. 2) The transient field requires significant memory to handle transient elements with per-view Gaussians and struggles to define clear boundaries for occluders, solely relying on photometric errors. To address these problems, we propose ForestSplats, a novel approach that leverages the deformable transient field and a superpixel-aware mask to efficiently represent transient elements in the 2D scene across unconstrained image collections and effectively decompose static scenes from transient distractors without VFM. We designed the transient field to be deformable, capturing per-view transient elements. Furthermore, we introduce a superpixel-aware mask that clearly defines the boundaries of occluders by considering photometric errors and superpixels. Additionally, we propose uncertainty-aware densification to avoid generating Gaussians within the boundaries of occluders during densification. Through extensive experiments across several benchmark datasets, we demonstrate that ForestSplats outperforms existing methods without VFM and shows significant memory efficiency in representing transient elements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06161v1">Feature-EndoGaussian: Feature Distilled Gaussian Splatting in Surgical Deformable Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 14 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Minimally invasive surgery (MIS) has transformed clinical practice by reducing recovery times, minimizing complications, and enhancing precision. Nonetheless, MIS inherently relies on indirect visualization and precise instrument control, posing unique challenges. Recent advances in artificial intelligence have enabled real-time surgical scene understanding through techniques such as image classification, object detection, and segmentation, with scene reconstruction emerging as a key element for enhanced intraoperative guidance. Although neural radiance fields (NeRFs) have been explored for this purpose, their substantial data requirements and slow rendering inhibit real-time performance. In contrast, 3D Gaussian Splatting (3DGS) offers a more efficient alternative, achieving state-of-the-art performance in dynamic surgical scene reconstruction. In this work, we introduce Feature-EndoGaussian (FEG), an extension of 3DGS that integrates 2D segmentation cues into 3D rendering to enable real-time semantic and scene reconstruction. By leveraging pretrained segmentation foundation models, FEG incorporates semantic feature distillation within the Gaussian deformation framework, thereby enhancing both reconstruction fidelity and segmentation accuracy. On the EndoNeRF dataset, FEG achieves superior performance (SSIM of 0.97, PSNR of 39.08, and LPIPS of 0.03) compared to leading methods. Additionally, on the EndoVis18 dataset, FEG demonstrates competitive class-wise segmentation metrics while balancing model size and real-time performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18041v4">OpenFly: A Versatile Toolchain and Large-scale Benchmark for Aerial Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation (VLN) aims to guide agents through an environment by leveraging both language instructions and visual cues, playing a pivotal role in embodied AI. Indoor VLN has been extensively studied, whereas outdoor aerial VLN remains underexplored. The potential reason is that outdoor aerial view encompasses vast areas, making data collection more challenging, which results in a lack of benchmarks. To address this problem, we propose OpenFly, a platform comprising a versatile toolchain and large-scale benchmark for aerial VLN. Firstly, we develop a highly automated toolchain for data collection, enabling automatic point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Secondly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering diverse heights and lengths across 18 scenes. The corresponding visual data are generated using various rendering engines and advanced techniques, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). All data exhibit high visual quality. Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of the dataset. Thirdly, we propose OpenFly-Agent, a keyframe-aware VLN model, which takes language instructions, current observations, and historical keyframes as input, and outputs flight actions directly. Extensive analyses and experiments are conducted, showcasing the superiority of our OpenFly platform and OpenFly-Agent. The toolchain, dataset, and codes will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06136v1">GSV3D: Gaussian Splatting-based Geometric Distillation with Stable Video Diffusion for Single-Image 3D Object Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      Image-based 3D generation has vast applications in robotics and gaming, where high-quality, diverse outputs and consistent 3D representations are crucial. However, existing methods have limitations: 3D diffusion models are limited by dataset scarcity and the absence of strong pre-trained priors, while 2D diffusion-based approaches struggle with geometric consistency. We propose a method that leverages 2D diffusion models' implicit 3D reasoning ability while ensuring 3D consistency via Gaussian-splatting-based geometric distillation. Specifically, the proposed Gaussian Splatting Decoder enforces 3D consistency by transforming SV3D latent outputs into an explicit 3D representation. Unlike SV3D, which only relies on implicit 2D representations for video generation, Gaussian Splatting explicitly encodes spatial and appearance attributes, enabling multi-view consistency through geometric constraints. These constraints correct view inconsistencies, ensuring robust geometric consistency. As a result, our approach simultaneously generates high-quality, multi-view-consistent images and accurate 3D models, providing a scalable solution for single-image-based 3D generation and bridging the gap between 2D Diffusion diversity and 3D structural coherence. Experimental results demonstrate state-of-the-art multi-view consistency and strong generalization across diverse datasets. The code will be made publicly available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06118v1">SecureGS: Boosting the Security and Fidelity of 3D Gaussian Splatting Steganography</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 Accepted by ICLR 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a premier method for 3D representation due to its real-time rendering and high-quality outputs, underscoring the critical need to protect the privacy of 3D assets. Traditional NeRF steganography methods fail to address the explicit nature of 3DGS since its point cloud files are publicly accessible. Existing GS steganography solutions mitigate some issues but still struggle with reduced rendering fidelity, increased computational demands, and security flaws, especially in the security of the geometric structure of the visualized point cloud. To address these demands, we propose a SecureGS, a secure and efficient 3DGS steganography framework inspired by Scaffold-GS's anchor point design and neural decoding. SecureGS uses a hybrid decoupled Gaussian encryption mechanism to embed offsets, scales, rotations, and RGB attributes of the hidden 3D Gaussian points in anchor point features, retrievable only by authorized users through privacy-preserving neural networks. To further enhance security, we propose a density region-aware anchor growing and pruning strategy that adaptively locates optimal hiding regions without exposing hidden information. Extensive experiments show that SecureGS significantly surpasses existing GS steganography methods in rendering fidelity, speed, and security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05600v1">D2GV: Deformable 2D Gaussian Splatting for Video Representation in 400FPS</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Implicit Neural Representations (INRs) have emerged as a powerful approach for video representation, offering versatility across tasks such as compression and inpainting. However, their implicit formulation limits both interpretability and efficacy, undermining their practicality as a comprehensive solution. We propose a novel video representation based on deformable 2D Gaussian splatting, dubbed D2GV, which aims to achieve three key objectives: 1) improved efficiency while delivering superior quality; 2) enhanced scalability and interpretability; and 3) increased friendliness for downstream tasks. Specifically, we initially divide the video sequence into fixed-length Groups of Pictures (GoP) to allow parallel training and linear scalability with video length. For each GoP, D2GV represents video frames by applying differentiable rasterization to 2D Gaussians, which are deformed from a canonical space into their corresponding timestamps. Notably, leveraging efficient CUDA-based rasterization, D2GV converges fast and decodes at speeds exceeding 400 FPS, while delivering quality that matches or surpasses state-of-the-art INRs. Moreover, we incorporate a learnable pruning and quantization strategy to streamline D2GV into a more compact representation. We demonstrate D2GV's versatility in tasks including video interpolation, inpainting and denoising, underscoring its potential as a promising solution for video representation. Code is available at: \href{https://github.com/Evan-sudo/D2GV}{https://github.com/Evan-sudo/D2GV}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05511v1">Free Your Hands: Lightweight Relightable Turntable Capture Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) from multiple captured photos of an object is a widely studied problem. Achieving high quality typically requires dense sampling of input views, which can lead to frustrating and tedious manual labor. Manually positioning cameras to maintain an optimal desired distribution can be difficult for humans, and if a good distribution is found, it is not easy to replicate. Additionally, the captured data can suffer from motion blur and defocus due to human error. In this paper, we present a lightweight object capture pipeline to reduce the manual workload and standardize the acquisition setup. We use a consumer turntable to carry the object and a tripod to hold the camera. As the turntable rotates, we automatically capture dense samples from various views and lighting conditions; we can repeat this for several camera positions. This way, we can easily capture hundreds of valid images in several minutes without hands-on effort. However, in the object reference frame, the light conditions vary; this is harmful to a standard NVS method like 3D Gaussian splatting (3DGS) which assumes fixed lighting. We design a neural radiance representation conditioned on light rotations, which addresses this issue and allows relightability as an additional benefit. We demonstrate our pipeline using 3DGS as the underlying framework, achieving competitive quality compared to previous methods with exhaustive acquisition and showcasing its potential for relighting and harmonization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05425v1">LiDAR-enhanced 3D Gaussian Splatting Mapping</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Accepted by ICRA 2025
    </div>
    <details class="paper-abstract">
      This paper introduces LiGSM, a novel LiDAR-enhanced 3D Gaussian Splatting (3DGS) mapping framework that improves the accuracy and robustness of 3D scene mapping by integrating LiDAR data. LiGSM constructs joint loss from images and LiDAR point clouds to estimate the poses and optimize their extrinsic parameters, enabling dynamic adaptation to variations in sensor alignment. Furthermore, it leverages LiDAR point clouds to initialize 3DGS, providing a denser and more reliable starting points compared to sparse SfM points. In scene rendering, the framework augments standard image-based supervision with depth maps generated from LiDAR projections, ensuring an accurate scene representation in both geometry and photometry. Experiments on public and self-collected datasets demonstrate that LiGSM outperforms comparative methods in pose tracking and scene rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05398v1">Self-Modeling Robots by Photographing</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Self-modeling enables robots to build task-agnostic models of their morphology and kinematics based on data that can be automatically collected, with minimal human intervention and prior information, thereby enhancing machine intelligence. Recent research has highlighted the potential of data-driven technology in modeling the morphology and kinematics of robots. However, existing self-modeling methods suffer from either low modeling quality or excessive data acquisition costs. Beyond morphology and kinematics, texture is also a crucial component of robots, which is challenging to model and remains unexplored. In this work, a high-quality, texture-aware, and link-level method is proposed for robot self-modeling. We utilize three-dimensional (3D) Gaussians to represent the static morphology and texture of robots, and cluster the 3D Gaussians to construct neural ellipsoid bones, whose deformations are controlled by the transformation matrices generated by a kinematic neural network. The 3D Gaussians and kinematic neural network are trained using data pairs composed of joint angles, camera parameters and multi-view images without depth information. By feeding the kinematic neural network with joint angles, we can utilize the well-trained model to describe the corresponding morphology, kinematics and texture of robots at the link level, and render robot images from different perspectives with the aid of 3D Gaussian splatting. Furthermore, we demonstrate that the established model can be exploited to perform downstream tasks such as motion planning and inverse kinematics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.03890v6">A Survey on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Ongoing project. Paper list: https://github.com/guikunchen/Awesome3DGS ; Benchmark: https://github.com/guikunchen/3DGS-Benchmarks
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (GS) has emerged as a transformative technique in explicit radiance field and computer graphics. This innovative approach, characterized by the use of millions of learnable 3D Gaussians, represents a significant departure from mainstream neural radiance field approaches, which predominantly use implicit, coordinate-based models to map spatial coordinates to pixel values. 3D GS, with its explicit scene representation and differentiable rendering algorithm, not only promises real-time rendering capability but also introduces unprecedented levels of editability. This positions 3D GS as a potential game-changer for the next generation of 3D reconstruction and representation. In the present paper, we provide the first systematic overview of the recent developments and critical contributions in the domain of 3D GS. We begin with a detailed exploration of the underlying principles and the driving forces behind the emergence of 3D GS, laying the groundwork for understanding its significance. A focal point of our discussion is the practical applicability of 3D GS. By enabling unprecedented rendering speed, 3D GS opens up a plethora of applications, ranging from virtual reality to interactive media and beyond. This is complemented by a comparative analysis of leading 3D GS models, evaluated across various benchmark tasks to highlight their performance and practical utility. The survey concludes by identifying current challenges and suggesting potential avenues for future research. Through this survey, we aim to provide a valuable resource for both newcomers and seasoned researchers, fostering further exploration and advancement in explicit radiance field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19233v2">Gaussians-to-Life: Text-Driven Animation of 3D Gaussian Splatting Scenes</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Project website at https://wimmerth.github.io/gaussians2life.html. Accepted to 3DV 2025
    </div>
    <details class="paper-abstract">
      State-of-the-art novel view synthesis methods achieve impressive results for multi-view captures of static 3D scenes. However, the reconstructed scenes still lack "liveliness," a key component for creating engaging 3D experiences. Recently, novel video diffusion models generate realistic videos with complex motion and enable animations of 2D images, however they cannot naively be used to animate 3D scenes as they lack multi-view consistency. To breathe life into the static world, we propose Gaussians2Life, a method for animating parts of high-quality 3D scenes in a Gaussian Splatting representation. Our key idea is to leverage powerful video diffusion models as the generative component of our model and to combine these with a robust technique to lift 2D videos into meaningful 3D motion. We find that, in contrast to prior work, this enables realistic animations of complex, pre-existing 3D scenes and further enables the animation of a large variety of object classes, while related work is mostly focused on prior-based character animation, or single 3D objects. Our model enables the creation of consistent, immersive 3D experiences for arbitrary scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05332v1">CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Revised Version of CRiM-GS, Github: https://github.com/Jho-Yonsei/CoMoGaussian
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has gained significant attention for their high-quality novel view rendering, motivating research to address real-world challenges. A critical issue is the camera motion blur caused by movement during exposure, which hinders accurate 3D scene reconstruction. In this study, we propose CoMoGaussian, a Continuous Motion-Aware Gaussian Splatting that reconstructs precise 3D scenes from motion-blurred images while maintaining real-time rendering speed. Considering the complex motion patterns inherent in real-world camera movements, we predict continuous camera trajectories using neural ordinary differential equations (ODEs). To ensure accurate modeling, we employ rigid body transformations, preserving the shape and size of the object but rely on the discrete integration of sampled frames. To better approximate the continuous nature of motion blur, we introduce a continuous motion refinement (CMR) transformation that refines rigid transformations by incorporating additional learnable parameters. By revisiting fundamental camera theory and leveraging advanced neural ODE techniques, we achieve precise modeling of continuous camera trajectories, leading to improved reconstruction accuracy. Extensive experiments demonstrate state-of-the-art performance both quantitatively and qualitatively on benchmark datasets, which include a wide range of motion blur scenarios, from moderate to extreme blur.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05196v1">STGA: Selective-Training Gaussian Head Avatars</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      We propose selective-training Gaussian head avatars (STGA) to enhance the details of dynamic head Gaussian. The dynamic head Gaussian model is trained based on the FLAME parameterized model. Each Gaussian splat is embedded within the FLAME mesh to achieve mesh-based animation of the Gaussian model. Before training, our selection strategy calculates the 3D Gaussian splat to be optimized in each frame. The parameters of these 3D Gaussian splats are optimized in the training of each frame, while those of the other splats are frozen. This means that the splats participating in the optimization process differ in each frame, to improve the realism of fine details. Compared with network-based methods, our method achieves better results with shorter training time. Compared with mesh-based methods, our method produces more realistic details within the same training time. Additionally, the ablation experiment confirms that our method effectively enhances the quality of details.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09740v3">Gaussian Splatting Visual MPC for Granular Media Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 project website https://weichengtseng.github.io/gs-granular-mani/
    </div>
    <details class="paper-abstract">
      Recent advancements in learned 3D representations have enabled significant progress in solving complex robotic manipulation tasks, particularly for rigid-body objects. However, manipulating granular materials such as beans, nuts, and rice, remains challenging due to the intricate physics of particle interactions, high-dimensional and partially observable state, inability to visually track individual particles in a pile, and the computational demands of accurate dynamics prediction. Current deep latent dynamics models often struggle to generalize in granular material manipulation due to a lack of inductive biases. In this work, we propose a novel approach that learns a visual dynamics model over Gaussian splatting representations of scenes and leverages this model for manipulating granular media via Model-Predictive Control. Our method enables efficient optimization for complex manipulation tasks on piles of granular media. We evaluate our approach in both simulated and real-world settings, demonstrating its ability to solve unseen planning tasks and generalize to new environments in a zero-shot transfer. We also show significant prediction and manipulation performance improvements compared to existing granular media manipulation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05189v1">Persistent Object Gaussian Splat (POGS) for Tracking Human and Robot Manipulation of Irregularly Shaped Objects</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Accepted to ICRA 2025
    </div>
    <details class="paper-abstract">
      Tracking and manipulating irregularly-shaped, previously unseen objects in dynamic environments is important for robotic applications in manufacturing, assembly, and logistics. Recently introduced Gaussian Splats efficiently model object geometry, but lack persistent state estimation for task-oriented manipulation. We present Persistent Object Gaussian Splat (POGS), a system that embeds semantics, self-supervised visual features, and object grouping features into a compact representation that can be continuously updated to estimate the pose of scanned objects. POGS updates object states without requiring expensive rescanning or prior CAD models of objects. After an initial multi-view scene capture and training phase, POGS uses a single stereo camera to integrate depth estimates along with self-supervised vision encoder features for object pose estimation. POGS supports grasping, reorientation, and natural language-driven manipulation by refining object pose estimates, facilitating sequential object reset operations with human-induced object perturbations and tool servoing, where robots recover tool pose despite tool perturbations of up to 30{\deg}. POGS achieves up to 12 consecutive successful object resets and recovers from 80% of in-grasp tool perturbations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05182v1">MGSR: 2D/3D Mutual-boosted Gaussian Splatting for High-fidelity Surface Reconstruction under Various Light Conditions</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 11 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) and surface reconstruction (SR) are essential tasks in 3D Gaussian Splatting (3D-GS). Despite recent progress, these tasks are often addressed independently, with GS-based rendering methods struggling under diverse light conditions and failing to produce accurate surfaces, while GS-based reconstruction methods frequently compromise rendering quality. This raises a central question: must rendering and reconstruction always involve a trade-off? To address this, we propose MGSR, a 2D/3D Mutual-boosted Gaussian splatting for Surface Reconstruction that enhances both rendering quality and 3D reconstruction accuracy. MGSR introduces two branches--one based on 2D-GS and the other on 3D-GS. The 2D-GS branch excels in surface reconstruction, providing precise geometry information to the 3D-GS branch. Leveraging this geometry, the 3D-GS branch employs a geometry-guided illumination decomposition module that captures reflected and transmitted components, enabling realistic rendering under varied light conditions. Using the transmitted component as supervision, the 2D-GS branch also achieves high-fidelity surface reconstruction. Throughout the optimization process, the 2D-GS and 3D-GS branches undergo alternating optimization, providing mutual supervision. Prior to this, each branch completes an independent warm-up phase, with an early stopping strategy implemented to reduce computational costs. We evaluate MGSR on a diverse set of synthetic and real-world datasets, at both object and scene levels, demonstrating strong performance in rendering and surface reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05174v1">SplatPose: Geometry-Aware 6-DoF Pose Estimation from Single RGB Image via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Submitted to IROS 2025
    </div>
    <details class="paper-abstract">
      6-DoF pose estimation is a fundamental task in computer vision with wide-ranging applications in augmented reality and robotics. Existing single RGB-based methods often compromise accuracy due to their reliance on initial pose estimates and susceptibility to rotational ambiguity, while approaches requiring depth sensors or multi-view setups incur significant deployment costs. To address these limitations, we introduce SplatPose, a novel framework that synergizes 3D Gaussian Splatting (3DGS) with a dual-branch neural architecture to achieve high-precision pose estimation using only a single RGB image. Central to our approach is the Dual-Attention Ray Scoring Network (DARS-Net), which innovatively decouples positional and angular alignment through geometry-domain attention mechanisms, explicitly modeling directional dependencies to mitigate rotational ambiguity. Additionally, a coarse-to-fine optimization pipeline progressively refines pose estimates by aligning dense 2D features between query images and 3DGS-synthesized views, effectively correcting feature misalignment and depth errors from sparse ray sampling. Experiments on three benchmark datasets demonstrate that SplatPose achieves state-of-the-art 6-DoF pose estimation accuracy in single RGB settings, rivaling approaches that depend on depth or multi-view images.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05168v1">SeeLe: A Unified Acceleration Framework for Real-Time Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a crucial rendering technique for many real-time applications. However, the limited hardware resources on today's mobile platforms hinder these applications, as they struggle to achieve real-time performance. In this paper, we propose SeeLe, a general framework designed to accelerate the 3DGS pipeline for resource-constrained mobile devices. Specifically, we propose two GPU-oriented techniques: hybrid preprocessing and contribution-aware rasterization. Hybrid preprocessing alleviates the GPU compute and memory pressure by reducing the number of irrelevant Gaussians during rendering. The key is to combine our view-dependent scene representation with online filtering. Meanwhile, contribution-aware rasterization improves the GPU utilization at the rasterization stage by prioritizing Gaussians with high contributions while reducing computations for those with low contributions. Both techniques can be seamlessly integrated into existing 3DGS pipelines with minimal fine-tuning. Collectively, our framework achieves 2.6$\times$ speedup and 32.3\% model reduction while achieving superior rendering quality compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00357v2">CAT-3DGS: A Context-Adaptive Triplane Approach to Rate-Distortion-Optimized 3DGS Compression</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Accepted for Publication in International Conference on Learning Representations (ICLR)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as a promising 3D representation. Much research has been focused on reducing its storage requirements and memory footprint. However, the needs to compress and transmit the 3DGS representation to the remote side are overlooked. This new application calls for rate-distortion-optimized 3DGS compression. How to quantize and entropy encode sparse Gaussian primitives in the 3D space remains largely unexplored. Few early attempts resort to the hyperprior framework from learned image compression. But, they fail to utilize fully the inter and intra correlation inherent in Gaussian primitives. Built on ScaffoldGS, this work, termed CAT-3DGS, introduces a context-adaptive triplane approach to their rate-distortion-optimized coding. It features multi-scale triplanes, oriented according to the principal axes of Gaussian primitives in the 3D space, to capture their inter correlation (i.e. spatial correlation) for spatial autoregressive coding in the projected 2D planes. With these triplanes serving as the hyperprior, we further perform channel-wise autoregressive coding to leverage the intra correlation within each individual Gaussian primitive. Our CAT-3DGS incorporates a view frequency-aware masking mechanism. It actively skips from coding those Gaussian primitives that potentially have little impact on the rendering quality. When trained end-to-end to strike a good rate-distortion trade-off, our CAT-3DGS achieves the state-of-the-art compression performance on the commonly used real-world datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06234v3">Generative Densification: Learning to Densify Gaussians for High-Fidelity Generalizable 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Project page: https://stnamjef.github.io/GenerativeDensification/
    </div>
    <details class="paper-abstract">
      Generalized feed-forward Gaussian models have achieved significant progress in sparse-view 3D reconstruction by leveraging prior knowledge from large multi-view datasets. However, these models often struggle to represent high-frequency details due to the limited number of Gaussians. While the densification strategy used in per-scene 3D Gaussian splatting (3D-GS) optimization can be adapted to the feed-forward models, it may not be ideally suited for generalized scenarios. In this paper, we propose Generative Densification, an efficient and generalizable method to densify Gaussians generated by feed-forward models. Unlike the 3D-GS densification strategy, which iteratively splits and clones raw Gaussian parameters, our method up-samples feature representations from the feed-forward models and generates their corresponding fine Gaussians in a single forward pass, leveraging the embedded prior knowledge for enhanced generalization. Experimental results on both object-level and scene-level reconstruction tasks demonstrate that our method outperforms state-of-the-art approaches with comparable or smaller model sizes, achieving notable improvements in representing fine details.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05162v1">EvolvingGS: High-Fidelity Streamable Volumetric Video via Evolving 3D Gaussian Representation</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      We have recently seen great progress in 3D scene reconstruction through explicit point-based 3D Gaussian Splatting (3DGS), notable for its high quality and fast rendering speed. However, reconstructing dynamic scenes such as complex human performances with long durations remains challenging. Prior efforts fall short of modeling a long-term sequence with drastic motions, frequent topology changes or interactions with props, and resort to segmenting the whole sequence into groups of frames that are processed independently, which undermines temporal stability and thereby leads to an unpleasant viewing experience and inefficient storage footprint. In view of this, we introduce EvolvingGS, a two-stage strategy that first deforms the Gaussian model to coarsely align with the target frame, and then refines it with minimal point addition/subtraction, particularly in fast-changing areas. Owing to the flexibility of the incrementally evolving representation, our method outperforms existing approaches in terms of both per-frame and temporal quality metrics while maintaining fast rendering through its purely explicit representation. Moreover, by exploiting temporal coherence between successive frames, we propose a simple yet effective compression algorithm that achieves over 50x compression rate. Extensive experiments on both public benchmarks and challenging custom datasets demonstrate that our method significantly advances the state-of-the-art in dynamic scene reconstruction, particularly for extended sequences with complex human performances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05161v1">GaussianCAD: Robust Self-Supervised CAD Reconstruction from Three Orthographic Views Using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      The automatic reconstruction of 3D computer-aided design (CAD) models from CAD sketches has recently gained significant attention in the computer vision community. Most existing methods, however, rely on vector CAD sketches and 3D ground truth for supervision, which are often difficult to be obtained in industrial applications and are sensitive to noise inputs. We propose viewing CAD reconstruction as a specific instance of sparse-view 3D reconstruction to overcome these limitations. While this reformulation offers a promising perspective, existing 3D reconstruction methods typically require natural images and corresponding camera poses as inputs, which introduces two major significant challenges: (1) modality discrepancy between CAD sketches and natural images, and (2) difficulty of accurate camera pose estimation for CAD sketches. To solve these issues, we first transform the CAD sketches into representations resembling natural images and extract corresponding masks. Next, we manually calculate the camera poses for the orthographic views to ensure accurate alignment within the 3D coordinate system. Finally, we employ a customized sparse-view 3D reconstruction method to achieve high-quality reconstructions from aligned orthographic views. By leveraging raster CAD sketches for self-supervision, our approach eliminates the reliance on vector CAD sketches and 3D ground truth. Experiments on the Sub-Fusion360 dataset demonstrate that our proposed method significantly outperforms previous approaches in CAD reconstruction performance and exhibits strong robustness to noisy inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05152v1">GSplatVNM: Point-of-View Synthesis for Visual Navigation Models Using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach to image-goal navigation by integrating 3D Gaussian Splatting (3DGS) with Visual Navigation Models (VNMs), a method we refer to as GSplatVNM. VNMs offer a promising paradigm for image-goal navigation by guiding a robot through a sequence of point-of-view images without requiring metrical localization or environment-specific training. However, constructing a dense and traversable sequence of target viewpoints from start to goal remains a central challenge, particularly when the available image database is sparse. To address these challenges, we propose a 3DGS-based viewpoint synthesis framework for VNMs that synthesizes intermediate viewpoints to seamlessly bridge gaps in sparse data while significantly reducing storage overhead. Experimental results in a photorealistic simulator demonstrate that our approach not only enhances navigation efficiency but also exhibits robustness under varying levels of image database sparsity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05082v1">Taming Video Diffusion Prior with Scene-Grounding Guidance for 3D Gaussian Splatting from Sparse Inputs</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Accepted by CVPR2025. The project page is available at https://zhongyingji.github.io/guidevd-3dgs/
    </div>
    <details class="paper-abstract">
      Despite recent successes in novel view synthesis using 3D Gaussian Splatting (3DGS), modeling scenes with sparse inputs remains a challenge. In this work, we address two critical yet overlooked issues in real-world sparse-input modeling: extrapolation and occlusion. To tackle these issues, we propose to use a reconstruction by generation pipeline that leverages learned priors from video diffusion models to provide plausible interpretations for regions outside the field of view or occluded. However, the generated sequences exhibit inconsistencies that do not fully benefit subsequent 3DGS modeling. To address the challenge of inconsistencies, we introduce a novel scene-grounding guidance based on rendered sequences from an optimized 3DGS, which tames the diffusion model to generate consistent sequences. This guidance is training-free and does not require any fine-tuning of the diffusion model. To facilitate holistic scene modeling, we also propose a trajectory initialization method. It effectively identifies regions that are outside the field of view and occluded. We further design a scheme tailored for 3DGS optimization with generated sequences. Experiments demonstrate that our method significantly improves upon the baseline and achieves state-of-the-art performance on challenging benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05949v1">Bayesian Fields: Task-driven Open-Set Semantic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Open-set semantic mapping requires (i) determining the correct granularity to represent the scene (e.g., how should objects be defined), and (ii) fusing semantic knowledge across multiple 2D observations into an overall 3D reconstruction -ideally with a high-fidelity yet low-memory footprint. While most related works bypass the first issue by grouping together primitives with similar semantics (according to some manually tuned threshold), we recognize that the object granularity is task-dependent, and develop a task-driven semantic mapping approach. To address the second issue, current practice is to average visual embedding vectors over multiple views. Instead, we show the benefits of using a probabilistic approach based on the properties of the underlying visual-language foundation model, and leveraging Bayesian updating to aggregate multiple observations of the scene. The result is Bayesian Fields, a task-driven and probabilistic approach for open-set semantic mapping. To enable high-fidelity objects and a dense scene representation, Bayesian Fields uses 3D Gaussians which we cluster into task-relevant objects, allowing for both easy 3D object extraction and reduced memory usage. We release Bayesian Fields open-source at https: //github.com/MIT-SPARK/Bayesian-Fields.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00299v2">GSPR: Multimodal Place Recognition Using 3D Gaussian Splatting for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 8 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Place recognition is a crucial component that enables autonomous vehicles to obtain localization results in GPS-denied environments. In recent years, multimodal place recognition methods have gained increasing attention. They overcome the weaknesses of unimodal sensor systems by leveraging complementary information from different modalities. However, most existing methods explore cross-modality correlations through feature-level or descriptor-level fusion, suffering from a lack of interpretability. Conversely, the recently proposed 3D Gaussian Splatting provides a new perspective on multimodal fusion by harmonizing different modalities into an explicit scene representation. In this paper, we propose a 3D Gaussian Splatting-based multimodal place recognition network dubbed GSPR. It explicitly combines multi-view RGB images and LiDAR point clouds into a spatio-temporally unified scene representation with the proposed Multimodal Gaussian Splatting. A network composed of 3D graph convolution and transformer is designed to extract spatio-temporal features and global descriptors from the Gaussian scenes for place recognition. Extensive evaluations on three datasets demonstrate that our method can effectively leverage complementary strengths of both multi-view cameras and LiDAR, achieving SOTA place recognition performance while maintaining solid generalization ability. Our open-source code will be released at https://github.com/QiZS-BIT/GSPR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04333v1">GaussianVideo: Efficient Video Representation and Compression by Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Implicit Neural Representation for Videos (NeRV) has introduced a novel paradigm for video representation and compression, outperforming traditional codecs. As model size grows, however, slow encoding and decoding speed and high memory consumption hinder its application in practice. To address these limitations, we propose a new video representation and compression method based on 2D Gaussian Splatting to efficiently handle video data. Our proposed deformable 2D Gaussian Splatting dynamically adapts the transformation of 2D Gaussians at each frame, significantly reducing memory cost. Equipped with a multi-plane-based spatiotemporal encoder and a lightweight decoder, it predicts changes in color, coordinates, and shape of initialized Gaussians, given the time step. By leveraging temporal gradients, our model effectively captures temporal redundancy at negligible cost, significantly enhancing video representation efficiency. Our method reduces GPU memory usage by up to 78.4%, and significantly expedites video processing, achieving 5.5x faster training and 12.5x faster decoding compared to the state-of-the-art NeRV methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04314v1">S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      In this paper, we aim ambitiously for a realistic yet challenging problem, namely, how to reconstruct high-quality 3D scenes from sparse low-resolution views that simultaneously suffer from deficient perspectives and clarity. Whereas existing methods only deal with either sparse views or low-resolution observations, they fail to handle such hybrid and complicated scenarios. To this end, we propose a novel Sparse-view Super-resolution 3D Gaussian Splatting framework, dubbed S2Gaussian, that can reconstruct structure-accurate and detail-faithful 3D scenes with only sparse and low-resolution views. The S2Gaussian operates in a two-stage fashion. In the first stage, we initially optimize a low-resolution Gaussian representation with depth regularization and densify it to initialize the high-resolution Gaussians through a tailored Gaussian Shuffle Split operation. In the second stage, we refine the high-resolution Gaussians with the super-resolved images generated from both original sparse views and pseudo-views rendered by the low-resolution Gaussians. In which a customized blur-free inconsistency modeling scheme and a 3D robust optimization strategy are elaborately designed to mitigate multi-view inconsistency and eliminate erroneous updates caused by imperfect supervision. Extensive experiments demonstrate superior results and in particular establishing new state-of-the-art performances with more consistent geometry and finer details.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18672v4">Drag Your Gaussian: Effective Drag-Based Editing with Score Distillation for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Visit our project page at https://quyans.github.io/Drag-Your-Gaussian
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D scene editing have been propelled by the rapid development of generative models. Existing methods typically utilize generative models to perform text-guided editing on 3D representations, such as 3D Gaussian Splatting (3DGS). However, these methods are often limited to texture modifications and fail when addressing geometric changes, such as editing a character's head to turn around. Moreover, such methods lack accurate control over the spatial position of editing results, as language struggles to precisely describe the extent of edits. To overcome these limitations, we introduce DYG, an effective 3D drag-based editing method for 3D Gaussian Splatting. It enables users to conveniently specify the desired editing region and the desired dragging direction through the input of 3D masks and pairs of control points, thereby enabling precise control over the extent of editing. DYG integrates the strengths of the implicit triplane representation to establish the geometric scaffold of the editing results, effectively overcoming suboptimal editing outcomes caused by the sparsity of 3DGS in the desired editing regions. Additionally, we incorporate a drag-based Latent Diffusion Model into our method through the proposed Drag-SDS loss function, enabling flexible, multi-view consistent, and fine-grained editing. Extensive experiments demonstrate that DYG conducts effective drag-based editing guided by control point prompts, surpassing other baselines in terms of editing effect and quality, both qualitatively and quantitatively. Visit our project page at https://quyans.github.io/Drag-Your-Gaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04082v1">Instrument-Splatting: Controllable Photorealistic Reconstruction of Surgical Instruments Using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 11 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Real2Sim is becoming increasingly important with the rapid development of surgical artificial intelligence (AI) and autonomy. In this work, we propose a novel Real2Sim methodology, \textit{Instrument-Splatting}, that leverages 3D Gaussian Splatting to provide fully controllable 3D reconstruction of surgical instruments from monocular surgical videos. To maintain both high visual fidelity and manipulability, we introduce a geometry pre-training to bind Gaussian point clouds on part mesh with accurate geometric priors and define a forward kinematics to control the Gaussians as flexible as real instruments. Afterward, to handle unposed videos, we design a novel instrument pose tracking method leveraging semantics-embedded Gaussians to robustly refine per-frame instrument poses and joint states in a render-and-compare manner, which allows our instrument Gaussian to accurately learn textures and reach photorealistic rendering. We validated our method on 2 publicly released surgical videos and 4 videos collected on ex vivo tissues and green screens. Quantitative and qualitative evaluations demonstrate the effectiveness and superiority of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04037v1">Beyond Existance: Fulfill 3D Reconstructed Scenes with Pseudo Details</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      The emergence of 3D Gaussian Splatting (3D-GS) has significantly advanced 3D reconstruction by providing high fidelity and fast training speeds across various scenarios. While recent efforts have mainly focused on improving model structures to compress data volume or reduce artifacts during zoom-in and zoom-out operations, they often overlook an underlying issue: training sampling deficiency. In zoomed-in views, Gaussian primitives can appear unregulated and distorted due to their dilation limitations and the insufficient availability of scale-specific training samples. Consequently, incorporating pseudo-details that ensure the completeness and alignment of the scene becomes essential. In this paper, we introduce a new training method that integrates diffusion models and multi-scale training using pseudo-ground-truth data. This approach not only notably mitigates the dilation and zoomed-in artifacts but also enriches reconstructed scenes with precise details out of existing scenarios. Our method achieves state-of-the-art performance across various benchmarks and extends the capabilities of 3D reconstruction beyond training datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04034v1">GaussianGraph: 3D Gaussian-based Scene Graph Generation for Open-world Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting(3DGS) have significantly improved semantic scene understanding, enabling natural language queries to localize objects within a scene. However, existing methods primarily focus on embedding compressed CLIP features to 3D Gaussians, suffering from low object segmentation accuracy and lack spatial reasoning capabilities. To address these limitations, we propose GaussianGraph, a novel framework that enhances 3DGS-based scene understanding by integrating adaptive semantic clustering and scene graph generation. We introduce a "Control-Follow" clustering strategy, which dynamically adapts to scene scale and feature distribution, avoiding feature compression and significantly improving segmentation accuracy. Additionally, we enrich scene representation by integrating object attributes and spatial relations extracted from 2D foundation models. To address inaccuracies in spatial relationships, we propose 3D correction modules that filter implausible relations through spatial consistency verification, ensuring reliable scene graph construction. Extensive experiments on three datasets demonstrate that GaussianGraph outperforms state-of-the-art methods in both semantic segmentation and object grounding tasks, providing a robust solution for complex scene understanding and interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03984v1">GRaD-Nav: Efficiently Learning Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Autonomous visual navigation is an essential element in robot autonomy. Reinforcement learning (RL) offers a promising policy training paradigm. However existing RL methods suffer from high sample complexity, poor sim-to-real transfer, and limited runtime adaptability to navigation scenarios not seen during training. These problems are particularly challenging for drones, with complex nonlinear and unstable dynamics, and strong dynamic coupling between control and perception. In this paper, we propose a novel framework that integrates 3D Gaussian Splatting (3DGS) with differentiable deep reinforcement learning (DDRL) to train vision-based drone navigation policies. By leveraging high-fidelity 3D scene representations and differentiable simulation, our method improves sample efficiency and sim-to-real transfer. Additionally, we incorporate a Context-aided Estimator Network (CENet) to adapt to environmental variations at runtime. Moreover, by curriculum training in a mixture of different surrounding environments, we achieve in-task generalization, the ability to solve new instances of a task not seen during training. Drone hardware experiments demonstrate our method's high training efficiency compared to state-of-the-art RL methods, zero shot sim-to-real transfer for real robot deployment without fine tuning, and ability to adapt to new instances within the same task class (e.g. to fly through a gate at different locations with different distractors in the environment).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13335v2">Deblur-Avatar: Animatable Avatars from Motion-Blurred Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for modeling high-fidelity, animatable 3D human avatars from motion-blurred monocular video inputs. Motion blur is prevalent in real-world dynamic video capture, especially due to human movements in 3D human avatar modeling. Existing methods either (1) assume sharp image inputs, failing to address the detail loss introduced by motion blur, or (2) mainly consider blur by camera movements, neglecting the human motion blur which is more common in animatable avatars. Our proposed approach integrates a human movement-based motion blur model into 3D Gaussian Splatting (3DGS). By explicitly modeling human motion trajectories during exposure time, we jointly optimize the trajectories and 3D Gaussians to reconstruct sharp, high-quality human avatars. We employ a pose-dependent fusion mechanism to distinguish moving body regions, optimizing both blurred and sharp areas effectively. Extensive experiments on synthetic and real-world datasets demonstrate that our method significantly outperforms existing methods in rendering quality and quantitative metrics, producing sharp avatar reconstructions and enabling real-time rendering under challenging motion blur conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16502v2">GSplatLoc: Grounding Keypoint Descriptors into 3D Gaussian Splatting for Improved Visual Localization</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 Project website at https://gsplatloc.github.io/
    </div>
    <details class="paper-abstract">
      Although various visual localization approaches exist, such as scene coordinate regression and camera pose regression, these methods often struggle with optimization complexity or limited accuracy. To address these challenges, we explore the use of novel view synthesis techniques, particularly 3D Gaussian Splatting (3DGS), which enables the compact encoding of both 3D geometry and scene appearance. We propose a two-stage procedure that integrates dense and robust keypoint descriptors from the lightweight XFeat feature extractor into 3DGS, enhancing performance in both indoor and outdoor environments. The coarse pose estimates are directly obtained via 2D-3D correspondences between the 3DGS representation and query image descriptors. In the second stage, the initial pose estimate is refined by minimizing the rendering-based photometric warp loss. Benchmarking on widely used indoor and outdoor datasets demonstrates improvements over recent neural rendering-based localization methods, such as NeRFMatch and PNeRFLoc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18783v2">ArtNVG: Content-Style Separated Artistic Neighboring-View Gaussian Stylization</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      As demand from the film and gaming industries for 3D scenes with target styles grows, the importance of advanced 3D stylization techniques increases. However, recent methods often struggle to maintain local consistency in color and texture throughout stylized scenes, which is essential for maintaining aesthetic coherence. To solve this problem, this paper introduces ArtNVG, an innovative 3D stylization framework that efficiently generates stylized 3D scenes by leveraging reference style images. Built on 3D Gaussian Splatting (3DGS), ArtNVG achieves rapid optimization and rendering while upholding high reconstruction quality. Our framework realizes high-quality 3D stylization by incorporating two pivotal techniques: Content-Style Separated Control and Attention-based Neighboring-View Alignment. Content-Style Separated Control uses the CSGO model and the Tile ControlNet to decouple the content and style control, reducing risks of information leakage. Concurrently, Attention-based Neighboring-View Alignment ensures consistency of local colors and textures across neighboring views, significantly improving visual quality. Extensive experiments validate that ArtNVG surpasses existing methods, delivering superior results in content preservation, style alignment, and local consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10860v2">Sim2Real within 5 Minutes: Efficient Domain Transfer with Stylized Gaussian Splatting for Endoscopic Images</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 Accepted by ICRA 2025
    </div>
    <details class="paper-abstract">
      Robot assisted endoluminal intervention is an emerging technique for both benign and malignant luminal lesions. With vision-based navigation, when combined with pre-operative imaging data as priors, it is possible to recover position and pose of the endoscope without the need of additional sensors. In practice, however, aligning pre-operative and intra-operative domains is complicated by significant texture differences. Although methods such as style transfer can be used to address this issue, they require large datasets from both source and target domains with prolonged training times. This paper proposes an efficient domain transfer method based on stylized Gaussian splatting, only requiring a few of real images (10 images) with very fast training time. Specifically, the transfer process includes two phases. In the first phase, the 3D models reconstructed from CT scans are represented as differential Gaussian point clouds. In the second phase, only color appearance related parameters are optimized to transfer the style and preserve the visual content. A novel structure consistency loss is applied to latent features and depth levels to enhance the stability of the transferred images. Detailed validation was performed to demonstrate the performance advantages of the proposed method compared to that of the current state-of-the-art, highlighting the potential for intra-operative surgical navigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09510v5">3DGS.zip: A survey on 3D Gaussian Splatting Compression Methods</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 3D Gaussian Splatting compression survey; 3DGS compression; updated discussion; new approaches added; new illustrations
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a cutting-edge technique for real-time radiance field rendering, offering state-of-the-art performance in terms of both quality and speed. 3DGS models a scene as a collection of three-dimensional Gaussians, with additional attributes optimized to conform to the scene's geometric and visual properties. Despite its advantages in rendering speed and image fidelity, 3DGS is limited by its significant storage and memory demands. These high demands make 3DGS impractical for mobile devices or headsets, reducing its applicability in important areas of computer graphics. To address these challenges and advance the practicality of 3DGS, this survey provides a comprehensive and detailed examination of compression and compaction techniques developed to make 3DGS more efficient. We classify existing methods into two categories: compression, which focuses on reducing file size, and compaction, which aims to minimize the number of Gaussians. Both methods aim to maintain or improve quality, each by minimizing its respective attribute: file size for compression and Gaussian count for compaction. We introduce the basic mathematical concepts underlying the analyzed methods, as well as key implementation details and design choices. Our report thoroughly discusses similarities and differences among the methods, as well as their respective advantages and disadvantages. We establish a consistent framework for comparing the surveyed methods based on key performance metrics and datasets. Specifically, since these methods have been developed in parallel and over a short period of time, currently, no comprehensive comparison exists. This survey, for the first time, presents a unified framework to evaluate 3DGS compression techniques. We maintain a website that will be regularly updated with emerging methods: https://w-m.github.io/3dgs-compression-survey/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03115v1">NTR-Gaussian: Nighttime Dynamic Thermal Reconstruction with 4D Gaussian Splatting Based on Thermodynamics</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 IEEE Conference on Computer Vision and Pattern Recognition 2025
    </div>
    <details class="paper-abstract">
      Thermal infrared imaging offers the advantage of all-weather capability, enabling non-intrusive measurement of an object's surface temperature. Consequently, thermal infrared images are employed to reconstruct 3D models that accurately reflect the temperature distribution of a scene, aiding in applications such as building monitoring and energy management. However, existing approaches predominantly focus on static 3D reconstruction for a single time period, overlooking the impact of environmental factors on thermal radiation and failing to predict or analyze temperature variations over time. To address these challenges, we propose the NTR-Gaussian method, which treats temperature as a form of thermal radiation, incorporating elements like convective heat transfer and radiative heat dissipation. Our approach utilizes neural networks to predict thermodynamic parameters such as emissivity, convective heat transfer coefficient, and heat capacity. By integrating these predictions, we can accurately forecast thermal temperatures at various times throughout a nighttime scene. Furthermore, we introduce a dynamic dataset specifically for nighttime thermal imagery. Extensive experiments and evaluations demonstrate that NTR-Gaussian significantly outperforms comparison methods in thermal reconstruction, achieving a predicted temperature error within 1 degree Celsius.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19895v3">GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 This paper is accepted by the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR), 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently created impressive 3D assets for various applications. However, the copyright of these assets is not well protected as existing watermarking methods are not suited for the 3DGS rendering pipeline considering security, capacity, and invisibility. Besides, these methods often require hours or even days for optimization, limiting the application scenarios. In this paper, we propose GuardSplat, an innovative and efficient framework that effectively protects the copyright of 3DGS assets. Specifically, 1) We first propose a CLIP-guided Message Decoupling Optimization module for training the message decoder, leveraging CLIP's aligning capability and rich representations to achieve a high extraction accuracy with minimal optimization costs, presenting exceptional capacity and efficiency. 2) Then, we propose a Spherical-harmonic-aware (SH-aware) Message Embedding module tailored for 3DGS, which employs a set of SH offsets to seamlessly embed the message into the SH features of each 3D Gaussian while maintaining the original 3D structure. It enables the 3DGS assets to be watermarked with minimal fidelity trade-offs and also prevents malicious users from removing the messages from the model files, meeting the demands for invisibility and security. 3) We further propose an Anti-distortion Message Extraction module to improve robustness against various visual distortions. Extensive experiments demonstrate that GuardSplat outperforms state-of-the-art and achieves fast optimization speed. Project page: https://narcissusex.github.io/GuardSplat, and Code: https://github.com/NarcissusEx/GuardSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03890v1">LensDFF: Language-enhanced Sparse Feature Distillation for Efficient Few-Shot Dexterous Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Learning dexterous manipulation from few-shot demonstrations is a significant yet challenging problem for advanced, human-like robotic systems. Dense distilled feature fields have addressed this challenge by distilling rich semantic features from 2D visual foundation models into the 3D domain. However, their reliance on neural rendering models such as Neural Radiance Fields (NeRF) or Gaussian Splatting results in high computational costs. In contrast, previous approaches based on sparse feature fields either suffer from inefficiencies due to multi-view dependencies and extensive training or lack sufficient grasp dexterity. To overcome these limitations, we propose Language-ENhanced Sparse Distilled Feature Field (LensDFF), which efficiently distills view-consistent 2D features onto 3D points using our novel language-enhanced feature fusion strategy, thereby enabling single-view few-shot generalization. Based on LensDFF, we further introduce a few-shot dexterous manipulation framework that integrates grasp primitives into the demonstrations to generate stable and highly dexterous grasps. Moreover, we present a real2sim grasp evaluation pipeline for efficient grasp assessment and hyperparameter tuning. Through extensive simulation experiments based on the real2sim pipeline and real-world experiments, our approach achieves competitive grasping performance, outperforming state-of-the-art approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17792v2">CrowdSplat: Exploring Gaussian Splatting For Crowd Rendering</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 4 pages, 4 figures
    </div>
    <details class="paper-abstract">
      We present CrowdSplat, a novel approach that leverages 3D Gaussian Splatting for real-time, high-quality crowd rendering. Our method utilizes 3D Gaussian functions to represent animated human characters in diverse poses and outfits, which are extracted from monocular videos. We integrate Level of Detail (LoD) rendering to optimize computational efficiency and quality. The CrowdSplat framework consists of two stages: (1) avatar reconstruction and (2) crowd synthesis. The framework is also optimized for GPU memory usage to enhance scalability. Quantitative and qualitative evaluations show that CrowdSplat achieves good levels of rendering quality, memory efficiency, and computational performance. Through these experiments, we demonstrate that CrowdSplat is a viable solution for dynamic, realistic crowd simulation in real-time applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17085v2">Evaluating CrowdSplat: Perceived Level of Detail for Gaussian Crowds</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 5 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Efficient and realistic crowd rendering is an important element of many real-time graphics applications such as Virtual Reality (VR) and games. To this end, Levels of Detail (LOD) avatar representations such as polygonal meshes, image-based impostors, and point clouds have been proposed and evaluated. More recently, 3D Gaussian Splatting has been explored as a potential method for real-time crowd rendering. In this paper, we present a two-alternative forced choice (2AFC) experiment that aims to determine the perceived quality of 3D Gaussian avatars. Three factors were explored: Motion, LOD (i.e., #Gaussians), and the avatar height in Pixels (corresponding to the viewing distance). Participants viewed pairs of animated 3D Gaussian avatars and were tasked with choosing the most detailed one. Our findings can inform the optimization of LOD strategies in Gaussian-based crowd rendering, thereby helping to achieve efficient rendering while maintaining visual quality in real-time applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02452v1">2DGS-Avatar: Animatable High-fidelity Clothed Avatar via 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 ICVRV 2024
    </div>
    <details class="paper-abstract">
      Real-time rendering of high-fidelity and animatable avatars from monocular videos remains a challenging problem in computer vision and graphics. Over the past few years, the Neural Radiance Field (NeRF) has made significant progress in rendering quality but behaves poorly in run-time performance due to the low efficiency of volumetric rendering. Recently, methods based on 3D Gaussian Splatting (3DGS) have shown great potential in fast training and real-time rendering. However, they still suffer from artifacts caused by inaccurate geometry. To address these problems, we propose 2DGS-Avatar, a novel approach based on 2D Gaussian Splatting (2DGS) for modeling animatable clothed avatars with high-fidelity and fast training performance. Given monocular RGB videos as input, our method generates an avatar that can be driven by poses and rendered in real-time. Compared to 3DGS-based methods, our 2DGS-Avatar retains the advantages of fast training and rendering while also capturing detailed, dynamic, and photo-realistic appearances. We conduct abundant experiments on popular datasets such as AvatarRex and THuman4.0, demonstrating impressive performance in both qualitative and quantitative metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18041v3">OpenFly: A Versatile Toolchain and Large-scale Benchmark for Aerial Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation (VLN) aims to guide agents through an environment by leveraging both language instructions and visual cues, playing a pivotal role in embodied AI. Indoor VLN has been extensively studied, whereas outdoor aerial VLN remains underexplored. The potential reason is that outdoor aerial view encompasses vast areas, making data collection more challenging, which results in a lack of benchmarks. To address this problem, we propose OpenFly, a platform comprising a versatile toolchain and large-scale benchmark for aerial VLN. Firstly, we develop a highly automated toolchain for data collection, enabling automatic point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Secondly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering diverse heights and lengths across 18 scenes. The corresponding visual data are generated using various rendering engines and advanced techniques, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). All data exhibit high visual quality. Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of the dataset. Thirdly, we propose OpenFly-Agent, a keyframe-aware VLN model, which takes language instructions, current observations, and historical keyframes as input, and outputs flight actions directly. Extensive analyses and experiments are conducted, showcasing the superiority of our OpenFly platform and OpenFly-Agent. The toolchain, dataset, and codes will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00308v2">Abstract Rendering: Computing All that is Seen in Gaussian Splat Scenes</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      We introduce abstract rendering, a method for computing a set of images by rendering a scene from a continuously varying range of camera positions. The resulting abstract image-which encodes an infinite collection of possible renderings-is represented using constraints on the image matrix, enabling rigorous uncertainty propagation through the rendering process. This capability is particularly valuable for the formal verification of vision-based autonomous systems and other safety-critical applications. Our approach operates on Gaussian splat scenes, an emerging representation in computer vision and robotics. We leverage efficient piecewise linear bound propagation to abstract fundamental rendering operations, while addressing key challenges that arise in matrix inversion and depth sorting-two operations not directly amenable to standard approximations. To handle these, we develop novel linear relational abstractions that maintain precision while ensuring computational efficiency. These abstractions not only power our abstract rendering algorithm but also provide broadly applicable tools for other rendering problems. Our implementation, AbstractSplat, is optimized for scalability, handling up to 750k Gaussians while allowing users to balance memory and runtime through tile and batch-based computation. Compared to the only existing abstract image method for mesh-based scenes, AbstractSplat achieves 2-14x speedups while preserving precision. Our results demonstrate that continuous camera motion, rotations, and scene variations can be rigorously analyzed at scale, making abstract rendering a powerful tool for uncertainty-aware vision applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02223v1">DQO-MAP: Dual Quadrics Multi-Object mapping with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Accurate object perception is essential for robotic applications such as object navigation. In this paper, we propose DQO-MAP, a novel object-SLAM system that seamlessly integrates object pose estimation and reconstruction. We employ 3D Gaussian Splatting for high-fidelity object reconstruction and leverage quadrics for precise object pose estimation. Both of them management is handled on the CPU, while optimization is performed on the GPU, significantly improving system efficiency. By associating objects with unique IDs, our system enables rapid object extraction from the scene. Extensive experimental results on object reconstruction and pose estimation demonstrate that DQO-MAP achieves outstanding performance in terms of precision, reconstruction quality, and computational efficiency. The code and dataset are available at: https://github.com/LiHaoy-ux/DQO-MAP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00771v2">CityGaussianV2: Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 Accepted by ICLR2025
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has revolutionized radiance field reconstruction, manifesting efficient and high-fidelity novel view synthesis. However, accurately representing surfaces, especially in large and complex scenarios, remains a significant challenge due to the unstructured nature of 3DGS. In this paper, we present CityGaussianV2, a novel approach for large-scale scene reconstruction that addresses critical challenges related to geometric accuracy and efficiency. Building on the favorable generalization capabilities of 2D Gaussian Splatting (2DGS), we address its convergence and scalability issues. Specifically, we implement a decomposed-gradient-based densification and depth regression technique to eliminate blurry artifacts and accelerate convergence. To scale up, we introduce an elongation filter that mitigates Gaussian count explosion caused by 2DGS degeneration. Furthermore, we optimize the CityGaussian pipeline for parallel training, achieving up to 10$\times$ compression, at least 25% savings in training time, and a 50% decrease in memory usage. We also established standard geometry benchmarks under large-scale scenes. Experimental results demonstrate that our method strikes a promising balance between visual quality, geometric accuracy, as well as storage and training costs. The project page is available at https://dekuliutesla.github.io/CityGaussianV2/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18669v3">Bootstrap-GS: Self-Supervised Augmentation for High-Fidelity Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3D-GS) have established new benchmarks for rendering quality and efficiency in 3D reconstruction. However, 3D-GS faces critical limitations when generating novel views that significantly deviate from those encountered during training. Moreover, issues such as dilation and aliasing arise during zoom operations. These challenges stem from a fundamental issue: training sampling deficiency. In this paper, we introduce a bootstrapping framework to address this problem. Our approach synthesizes pseudo-ground truth from novel views that align with the limited training set and reintegrates these synthesized views into the training pipeline. Experimental results demonstrate that our bootstrapping technique not only reduces artifacts but also improves quantitative metrics. Furthermore, our technique is highly adaptable, allowing various Gaussian-based method to benefit from its integration.
    </details>
</div>
