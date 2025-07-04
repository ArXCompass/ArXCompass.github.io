# gaussian splatting - 2025_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20194v2">Radiance Fields for Robotic Teleoperation</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 8 pages, 10 figures, Accepted to IROS 2024
    </div>
    <details class="paper-abstract">
      Radiance field methods such as Neural Radiance Fields (NeRFs) or 3D Gaussian Splatting (3DGS), have revolutionized graphics and novel view synthesis. Their ability to synthesize new viewpoints with photo-realistic quality, as well as capture complex volumetric and specular scenes, makes them an ideal visualization for robotic teleoperation setups. Direct camera teleoperation provides high-fidelity operation at the cost of maneuverability, while reconstruction-based approaches offer controllable scenes with lower fidelity. With this in mind, we propose replacing the traditional reconstruction-visualization components of the robotic teleoperation pipeline with online Radiance Fields, offering highly maneuverable scenes with photorealistic quality. As such, there are three main contributions to state of the art: (1) online training of Radiance Fields using live data from multiple cameras, (2) support for a variety of radiance methods including NeRF and 3DGS, (3) visualization suite for these methods including a virtual reality scene. To enable seamless integration with existing setups, these components were tested with multiple robots in multiple configurations and were displayed using traditional tools as well as the VR headset. The results across methods and robots were compared quantitatively to a baseline of mesh reconstruction, and a user study was conducted to compare the different visualization methods. For videos and code, check out https://rffr.leggedrobotics.com/works/teleoperation/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10923v1">GrowSplat: Constructing Temporal Digital Twins of Plants with Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Accurate temporal reconstructions of plant growth are essential for plant phenotyping and breeding, yet remain challenging due to complex geometries, occlusions, and non-rigid deformations of plants. We present a novel framework for building temporal digital twins of plants by combining 3D Gaussian Splatting with a robust sample alignment pipeline. Our method begins by reconstructing Gaussian Splats from multi-view camera data, then leverages a two-stage registration approach: coarse alignment through feature-based matching and Fast Global Registration, followed by fine alignment with Iterative Closest Point. This pipeline yields a consistent 4D model of plant development in discrete time steps. We evaluate the approach on data from the Netherlands Plant Eco-phenotyping Center, demonstrating detailed temporal reconstructions of Sequoia and Quinoa species. Videos and Images can be seen at https://berkeleyautomation.github.io/GrowSplat/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10787v1">EA-3DGS: Efficient and Adaptive 3D Gaussians with Highly Enhanced Quality for outdoor scenes</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Efficient scene representations are essential for many real-world applications, especially those involving spatial measurement. Although current NeRF-based methods have achieved impressive results in reconstructing building-scale scenes, they still suffer from slow training and inference speeds due to time-consuming stochastic sampling. Recently, 3D Gaussian Splatting (3DGS) has demonstrated excellent performance with its high-quality rendering and real-time speed, especially for objects and small-scale scenes. However, in outdoor scenes, its point-based explicit representation lacks an effective adjustment mechanism, and the millions of Gaussian points required often lead to memory constraints during training. To address these challenges, we propose EA-3DGS, a high-quality real-time rendering method designed for outdoor scenes. First, we introduce a mesh structure to regulate the initialization of Gaussian components by leveraging an adaptive tetrahedral mesh that partitions the grid and initializes Gaussian components on each face, effectively capturing geometric structures in low-texture regions. Second, we propose an efficient Gaussian pruning strategy that evaluates each 3D Gaussian's contribution to the view and prunes accordingly. To retain geometry-critical Gaussian points, we also present a structure-aware densification strategy that densifies Gaussian points in low-curvature regions. Additionally, we employ vector quantization for parameter quantization of Gaussian components, significantly reducing disk space requirements with only a minimal impact on rendering quality. Extensive experiments on 13 scenes, including eight from four public datasets (MatrixCity-Aerial, Mill-19, Tanks \& Temples, WHU) and five self-collected scenes acquired through UAV photogrammetry measurement from SCUT-CA and plateau regions, further demonstrate the superiority of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19409v2">GSFF-SLAM: 3D Semantic Gaussian Splatting SLAM via Feature Field</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Semantic-aware 3D scene reconstruction is essential for autonomous robots to perform complex interactions. Semantic SLAM, an online approach, integrates pose tracking, geometric reconstruction, and semantic mapping into a unified framework, shows significant potential. However, existing systems, which rely on 2D ground truth priors for supervision, are often limited by the sparsity and noise of these signals in real-world environments. To address this challenge, we propose GSFF-SLAM, a novel dense semantic SLAM system based on 3D Gaussian Splatting that leverages feature fields to achieve joint rendering of appearance, geometry, and N-dimensional semantic features. By independently optimizing feature gradients, our method supports semantic reconstruction using various forms of 2D priors, particularly sparse and noisy signals. Experimental results demonstrate that our approach outperforms previous methods in both tracking accuracy and photorealistic rendering quality. When utilizing 2D ground truth priors, GSFF-SLAM achieves state-of-the-art semantic segmentation performance with 95.03\% mIoU, while achieving up to 2.9$\times$ speedup with only marginal performance degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10473v1">Consistent Quantity-Quality Control across Scenes for Deployment-Aware Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      To reduce storage and computational costs, 3D Gaussian splatting (3DGS) seeks to minimize the number of Gaussians used while preserving high rendering quality, introducing an inherent trade-off between Gaussian quantity and rendering quality. Existing methods strive for better quantity-quality performance, but lack the ability for users to intuitively adjust this trade-off to suit practical needs such as model deployment under diverse hardware and communication constraints. Here, we present ControlGS, a 3DGS optimization method that achieves semantically meaningful and cross-scene consistent quantity-quality control while maintaining strong quantity-quality performance. Through a single training run using a fixed setup and a user-specified hyperparameter reflecting quantity-quality preference, ControlGS can automatically find desirable quantity-quality trade-off points across diverse scenes, from compact objects to large outdoor scenes. It also outperforms baselines by achieving higher rendering quality with fewer Gaussians, and supports a broad adjustment range with stepless control over the trade-off.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05731v3">PEP-GS: Perceptually-Enhanced Precise Structured 3D Gaussians for View-Adaptive Rendering</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3D-GS) has achieved significant success in real-time, high-quality 3D scene rendering. However, it faces several challenges, including Gaussian redundancy, limited ability to capture view-dependent effects, and difficulties in handling complex lighting and specular reflections. Additionally, methods that use spherical harmonics for color representation often struggle to effectively capture anisotropic components, especially when modeling view-dependent colors under complex lighting conditions, leading to insufficient contrast and unnatural color saturation. To address these limitations, we introduce PEP-GS, a perceptually-enhanced framework that dynamically predicts Gaussian attributes, including opacity, color, and covariance. We replace traditional spherical harmonics with a Hierarchical Granular-Structural Attention mechanism, which enables more accurate modeling of complex view-dependent color effects. By employing a stable and interpretable framework for opacity and covariance estimation, PEP-GS avoids the removal of essential Gaussians prematurely, ensuring a more accurate scene representation. Furthermore, perceptual optimization is applied to the final rendered images, enhancing perceptual consistency across different views and ensuring high-quality renderings with improved texture fidelity and fine-scale detail preservation. Experimental results demonstrate that PEP-GS outperforms state-of-the-art methods, particularly in challenging scenarios involving view-dependent effects and fine-scale details.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10144v1">VRSplat: Fast and Robust Gaussian Splatting for Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 I3D'25 (PACMCGIT); Project Page: https://cekavis.site/VRSplat/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has rapidly become a leading technique for novel-view synthesis, providing exceptional performance through efficient software-based GPU rasterization. Its versatility enables real-time applications, including on mobile and lower-powered devices. However, 3DGS faces key challenges in virtual reality (VR): (1) temporal artifacts, such as popping during head movements, (2) projection-based distortions that result in disturbing and view-inconsistent floaters, and (3) reduced framerates when rendering large numbers of Gaussians, falling below the critical threshold for VR. Compared to desktop environments, these issues are drastically amplified by large field-of-view, constant head movements, and high resolution of head-mounted displays (HMDs). In this work, we introduce VRSplat: we combine and extend several recent advancements in 3DGS to address challenges of VR holistically. We show how the ideas of Mini-Splatting, StopThePop, and Optimal Projection can complement each other, by modifying the individual techniques and core 3DGS rasterizer. Additionally, we propose an efficient foveated rasterizer that handles focus and peripheral areas in a single GPU launch, avoiding redundant computations and improving GPU utilization. Our method also incorporates a fine-tuning step that optimizes Gaussian parameters based on StopThePop depth evaluations and Optimal Projection. We validate our method through a controlled user study with 25 participants, showing a strong preference for VRSplat over other configurations of Mini-Splatting. VRSplat is the first, systematically evaluated 3DGS approach capable of supporting modern VR applications, achieving 72+ FPS while eliminating popping and stereo-disrupting floaters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16028v2">CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 CVPR 2025, Project Page: https://Jho-Yonsei.github.io/CoCoGaussian/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has attracted significant attention for its high-quality novel view rendering, inspiring research to address real-world challenges. While conventional methods depend on sharp images for accurate scene reconstruction, real-world scenarios are often affected by defocus blur due to finite depth of field, making it essential to account for realistic 3D scene representation. In this study, we propose CoCoGaussian, a Circle of Confusion-aware Gaussian Splatting that enables precise 3D scene representation using only defocused images. CoCoGaussian addresses the challenge of defocus blur by modeling the Circle of Confusion (CoC) through a physically grounded approach based on the principles of photographic defocus. Exploiting 3D Gaussians, we compute the CoC diameter from depth and learnable aperture information, generating multiple Gaussians to precisely capture the CoC shape. Furthermore, we introduce a learnable scaling factor to enhance robustness and provide more flexibility in handling unreliable depth in scenes with reflective or refractive surfaces. Experiments on both synthetic and real-world datasets demonstrate that CoCoGaussian achieves state-of-the-art performance across multiple benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10049v1">Advances in Radiance Field for Dynamic Scene: From Neural Field to Gaussian Field</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Dynamic scene representation and reconstruction have undergone transformative advances in recent years, catalyzed by breakthroughs in neural radiance fields and 3D Gaussian splatting techniques. While initially developed for static environments, these methodologies have rapidly evolved to address the complexities inherent in 4D dynamic scenes through an expansive body of research. Coupled with innovations in differentiable volumetric rendering, these approaches have significantly enhanced the quality of motion representation and dynamic scene reconstruction, thereby garnering substantial attention from the computer vision and graphics communities. This survey presents a systematic analysis of over 200 papers focused on dynamic scene representation using radiance field, spanning the spectrum from implicit neural representations to explicit Gaussian primitives. We categorize and evaluate these works through multiple critical lenses: motion representation paradigms, reconstruction techniques for varied scene dynamics, auxiliary information integration strategies, and regularization approaches that ensure temporal consistency and physical plausibility. We organize diverse methodological approaches under a unified representational framework, concluding with a critical examination of persistent challenges and promising research directions. By providing this comprehensive overview, we aim to establish a definitive reference for researchers entering this rapidly evolving field while offering experienced practitioners a systematic understanding of both conceptual principles and practical frontiers in dynamic scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09915v1">Large-Scale Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      The recently developed Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have shown encouraging and impressive results for visual SLAM. However, most representative methods require RGBD sensors and are only available for indoor environments. The robustness of reconstruction in large-scale outdoor scenarios remains unexplored. This paper introduces a large-scale 3DGS-based visual SLAM with stereo cameras, termed LSG-SLAM. The proposed LSG-SLAM employs a multi-modality strategy to estimate prior poses under large view changes. In tracking, we introduce feature-alignment warping constraints to alleviate the adverse effects of appearance similarity in rendering losses. For the scalability of large-scale scenarios, we introduce continuous Gaussian Splatting submaps to tackle unbounded scenes with limited memory. Loops are detected between GS submaps by place recognition and the relative pose between looped keyframes is optimized utilizing rendering and feature warping losses. After the global optimization of camera poses and Gaussian points, a structure refinement module enhances the reconstruction quality. With extensive evaluations on the EuRoc and KITTI datasets, LSG-SLAM achieves superior performance over existing Neural, 3DGS-based, and even traditional approaches. Project page: https://lsg-slam.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08712v2">NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 Project Page: https://wzcai99.github.io/navigation-diffusion-policy.github.io/
    </div>
    <details class="paper-abstract">
      Learning navigation in dynamic open-world environments is an important yet challenging skill for robots. Most previous methods rely on precise localization and mapping or learn from expensive real-world demonstrations. In this paper, we propose the Navigation Diffusion Policy (NavDP), an end-to-end framework trained solely in simulation and can zero-shot transfer to different embodiments in diverse real-world environments. The key ingredient of NavDP's network is the combination of diffusion-based trajectory generation and a critic function for trajectory selection, which are conditioned on only local observation tokens encoded from a shared policy transformer. Given the privileged information of the global environment in simulation, we scale up the demonstrations of good quality to train the diffusion policy and formulate the critic value function targets with contrastive negative samples. Our demonstration generation approach achieves about 2,500 trajectories/GPU per day, 20$\times$ more efficient than real-world data collection, and results in a large-scale navigation dataset with 363.2km trajectories across 1244 scenes. Trained with this simulation dataset, NavDP achieves state-of-the-art performance and consistently outstanding generalization capability on quadruped, wheeled, and humanoid robots in diverse indoor and outdoor environments. In addition, we present a preliminary attempt at using Gaussian Splatting to make in-domain real-to-sim fine-tuning to further bridge the sim-to-real gap. Experiments show that adding such real-to-sim data can improve the success rate by 30\% without hurting its generalization capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09601v1">Real2Render2Real: Scaling Robot Data Without Dynamics Simulation or Robot Hardware</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Scaling robot learning requires vast and diverse datasets. Yet the prevailing data collection paradigm-human teleoperation-remains costly and constrained by manual effort and physical robot access. We introduce Real2Render2Real (R2R2R), a novel approach for generating robot training data without relying on object dynamics simulation or teleoperation of robot hardware. The input is a smartphone-captured scan of one or more objects and a single video of a human demonstration. R2R2R renders thousands of high visual fidelity robot-agnostic demonstrations by reconstructing detailed 3D object geometry and appearance, and tracking 6-DoF object motion. R2R2R uses 3D Gaussian Splatting (3DGS) to enable flexible asset generation and trajectory synthesis for both rigid and articulated objects, converting these representations to meshes to maintain compatibility with scalable rendering engines like IsaacLab but with collision modeling off. Robot demonstration data generated by R2R2R integrates directly with models that operate on robot proprioceptive states and image observations, such as vision-language-action models (VLA) and imitation learning policies. Physical experiments suggest that models trained on R2R2R data from a single human demonstration can match the performance of models trained on 150 human teleoperation demonstrations. Project page: https://real2render2real.com
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09324v1">Neural Video Compression using 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-14
      | 💬 9 pages, 8 figures
    </div>
    <details class="paper-abstract">
      The computer vision and image processing research community has been involved in standardizing video data communications for the past many decades, leading to standards such as AVC, HEVC, VVC, AV1, AV2, etc. However, recent groundbreaking works have focused on employing deep learning-based techniques to replace the traditional video codec pipeline to a greater affect. Neural video codecs (NVC) create an end-to-end ML-based solution that does not rely on any handcrafted features (motion or edge-based) and have the ability to learn content-aware compression strategies, offering better adaptability and higher compression efficiency than traditional methods. This holds a great potential not only for hardware design, but also for various video streaming platforms and applications, especially video conferencing applications such as MS-Teams or Zoom that have found extensive usage in classrooms and workplaces. However, their high computational demands currently limit their use in real-time applications like video conferencing. To address this, we propose a region-of-interest (ROI) based neural video compression model that leverages 2D Gaussian Splatting. Unlike traditional codecs, 2D Gaussian Splatting is capable of real-time decoding and can be optimized using fewer data points, requiring only thousands of Gaussians for decent quality outputs as opposed to millions in 3D scenes. In this work, we designed a video pipeline that speeds up the encoding time of the previous Gaussian splatting-based image codec by 88% by using a content-aware initialization strategy paired with a novel Gaussian inter-frame redundancy-reduction mechanism, enabling Gaussian splatting to be used for a video-codec solution, the first of its kind solution in this neural video codec space.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02126v2">GarmentGS: Point-Cloud Guided Gaussian Splatting for High-Fidelity Non-Watertight 3D Garment Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-05-14
      | 💬 Accepted by ICMR 2025
    </div>
    <details class="paper-abstract">
      Traditional 3D garment creation requires extensive manual operations, resulting in time and labor costs. Recently, 3D Gaussian Splatting has achieved breakthrough progress in 3D scene reconstruction and rendering, attracting widespread attention and opening new pathways for 3D garment reconstruction. However, due to the unstructured and irregular nature of Gaussian primitives, it is difficult to reconstruct high-fidelity, non-watertight 3D garments. In this paper, we present GarmentGS, a dense point cloud-guided method that can reconstruct high-fidelity garment surfaces with high geometric accuracy and generate non-watertight, single-layer meshes. Our method introduces a fast dense point cloud reconstruction module that can complete garment point cloud reconstruction in 10 minutes, compared to traditional methods that require several hours. Furthermore, we use dense point clouds to guide the movement, flattening, and rotation of Gaussian primitives, enabling better distribution on the garment surface to achieve superior rendering effects and geometric accuracy. Through numerical and visual comparisons, our method achieves fast training and real-time rendering while maintaining competitive quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10578v1">ExploreGS: a vision-based low overhead framework for 3D scene reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      This paper proposes a low-overhead, vision-based 3D scene reconstruction framework for drones, named ExploreGS. By using RGB images, ExploreGS replaces traditional lidar-based point cloud acquisition process with a vision model, achieving a high-quality reconstruction at a lower cost. The framework integrates scene exploration and model reconstruction, and leverags a Bag-of-Words(BoW) model to enable real-time processing capabilities, therefore, the 3D Gaussian Splatting (3DGS) training can be executed on-board. Comprehensive experiments in both simulation and real-world environments demonstrate the efficiency and applicability of the ExploreGS framework on resource-constrained devices, while maintaining reconstruction quality comparable to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08712v1">NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Learning navigation in dynamic open-world environments is an important yet challenging skill for robots. Most previous methods rely on precise localization and mapping or learn from expensive real-world demonstrations. In this paper, we propose the Navigation Diffusion Policy (NavDP), an end-to-end framework trained solely in simulation and can zero-shot transfer to different embodiments in diverse real-world environments. The key ingredient of NavDP's network is the combination of diffusion-based trajectory generation and a critic function for trajectory selection, which are conditioned on only local observation tokens encoded from a shared policy transformer. Given the privileged information of the global environment in simulation, we scale up the demonstrations of good quality to train the diffusion policy and formulate the critic value function targets with contrastive negative samples. Our demonstration generation approach achieves about 2,500 trajectories/GPU per day, 20$\times$ more efficient than real-world data collection, and results in a large-scale navigation dataset with 363.2km trajectories across 1244 scenes. Trained with this simulation dataset, NavDP achieves state-of-the-art performance and consistently outstanding generalization capability on quadruped, wheeled, and humanoid robots in diverse indoor and outdoor environments. In addition, we present a preliminary attempt at using Gaussian Splatting to make in-domain real-to-sim fine-tuning to further bridge the sim-to-real gap. Experiments show that adding such real-to-sim data can improve the success rate by 30\% without hurting its generalization capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02283v5">GP-GS: Gaussian Processes for Enhanced Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 12 pages, 7 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has emerged as an efficient photorealistic novel view synthesis method. However, its reliance on sparse Structure-from-Motion (SfM) point clouds often limits scene reconstruction quality. To address the limitation, this paper proposes a novel 3D reconstruction framework, Gaussian Processes enhanced Gaussian Splatting (GP-GS), in which a multi-output Gaussian Process model is developed to enable adaptive and uncertainty-guided densification of sparse SfM point clouds. Specifically, we propose a dynamic sampling and filtering pipeline that adaptively expands the SfM point clouds by leveraging GP-based predictions to infer new candidate points from the input 2D pixels and depth maps. The pipeline utilizes uncertainty estimates to guide the pruning of high-variance predictions, ensuring geometric consistency and enabling the generation of dense point clouds. These densified point clouds provide high-quality initial 3D Gaussians, enhancing reconstruction performance. Extensive experiments conducted on synthetic and real-world datasets across various scales validate the effectiveness and practicality of the proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08644v1">DLO-Splatting: Tracking Deformable Linear Objects Using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 5 pages, 2 figures, presented at the 2025 5th Workshop: Reflections on Representations and Manipulating Deformable Objects at the IEEE International Conference on Robotics and Automation. RMDO workshop (https://deformable-workshop.github.io/icra2025/)
    </div>
    <details class="paper-abstract">
      This work presents DLO-Splatting, an algorithm for estimating the 3D shape of Deformable Linear Objects (DLOs) from multi-view RGB images and gripper state information through prediction-update filtering. The DLO-Splatting algorithm uses a position-based dynamics model with shape smoothness and rigidity dampening corrections to predict the object shape. Optimization with a 3D Gaussian Splatting-based rendering loss iteratively renders and refines the prediction to align it with the visual observations in the update step. Initial experiments demonstrate promising results in a knot tying scenario, which is challenging for existing vision-only methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19703v2">High-Quality Spatial Reconstruction and Orthoimage Generation Using Efficient 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Highly accurate geometric precision and dense image features characterize True Digital Orthophoto Maps (TDOMs), which are in great demand for applications such as urban planning, infrastructure management, and environmental monitoring.Traditional TDOM generation methods need sophisticated processes, such as Digital Surface Models (DSM) and occlusion detection, which are computationally expensive and prone to errors.This work presents an alternative technique rooted in 2D Gaussian Splatting (2DGS), free of explicit DSM and occlusion detection. With depth map generation, spatial information for every pixel within the TDOM is retrieved and can reconstruct the scene with high precision. Divide-and-conquer strategy achieves excellent GS training and rendering with high-resolution TDOMs at a lower resource cost, which preserves higher quality of rendering on complex terrain and thin structure without a decrease in efficiency. Experimental results demonstrate the efficiency of large-scale scene reconstruction and high-precision terrain modeling. This approach provides accurate spatial data, which assists users in better planning and decision-making based on maps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08510v1">FOCI: Trajectory Optimization on Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 7 pages, 8 figures, Mario Gomez Andreu and Maximum Wilder-Smith contributed equally
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently gained popularity as a faster alternative to Neural Radiance Fields (NeRFs) in 3D reconstruction and view synthesis methods. Leveraging the spatial information encoded in 3DGS, this work proposes FOCI (Field Overlap Collision Integral), an algorithm that is able to optimize trajectories directly on the Gaussians themselves. FOCI leverages a novel and interpretable collision formulation for 3DGS using the notion of the overlap integral between Gaussians. Contrary to other approaches, which represent the robot with conservative bounding boxes that underestimate the traversability of the environment, we propose to represent the environment and the robot as Gaussian Splats. This not only has desirable computational properties, but also allows for orientation-aware planning, allowing the robot to pass through very tight and narrow spaces. We extensively test our algorithm in both synthetic and real Gaussian Splats, showcasing that collision-free trajectories for the ANYmal legged robot that can be computed in a few seconds, even with hundreds of thousands of Gaussians making up the environment. The project page and code are available at https://rffr.leggedrobotics.com/works/foci/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07396v2">TUM2TWIN: Introducing the Large-Scale Multimodal Urban Digital Twin Benchmark Dataset</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 Submitted to the ISPRS Journal of Photogrammetry and Remote Sensing
    </div>
    <details class="paper-abstract">
      Urban Digital Twins (UDTs) have become essential for managing cities and integrating complex, heterogeneous data from diverse sources. Creating UDTs involves challenges at multiple process stages, including acquiring accurate 3D source data, reconstructing high-fidelity 3D models, maintaining models' updates, and ensuring seamless interoperability to downstream tasks. Current datasets are usually limited to one part of the processing chain, hampering comprehensive UDTs validation. To address these challenges, we introduce the first comprehensive multimodal Urban Digital Twin benchmark dataset: TUM2TWIN. This dataset includes georeferenced, semantically aligned 3D models and networks along with various terrestrial, mobile, aerial, and satellite observations boasting 32 data subsets over roughly 100,000 $m^2$ and currently 767 GB of data. By ensuring georeferenced indoor-outdoor acquisition, high accuracy, and multimodal data integration, the benchmark supports robust analysis of sensors and the development of advanced reconstruction methods. Additionally, we explore downstream tasks demonstrating the potential of TUM2TWIN, including novel view synthesis of NeRF and Gaussian Splatting, solar potential analysis, point cloud semantic segmentation, and LoD3 building reconstruction. We are convinced this contribution lays a foundation for overcoming current limitations in UDT creation, fostering new research directions and practical solutions for smarter, data-driven urban environments. The project is available under: https://tum2t.win
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08438v1">A Survey of 3D Reconstruction with Event Cameras: From Event-based Geometry to Neural 3D Rendering</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 35 pages, 12 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Event cameras have emerged as promising sensors for 3D reconstruction due to their ability to capture per-pixel brightness changes asynchronously. Unlike conventional frame-based cameras, they produce sparse and temporally rich data streams, which enable more accurate 3D reconstruction and open up the possibility of performing reconstruction in extreme environments such as high-speed motion, low light, or high dynamic range scenes. In this survey, we provide the first comprehensive review focused exclusively on 3D reconstruction using event cameras. The survey categorises existing works into three major types based on input modality - stereo, monocular, and multimodal systems, and further classifies them by reconstruction approach, including geometry-based, deep learning-based, and recent neural rendering techniques such as Neural Radiance Fields and 3D Gaussian Splatting. Methods with a similar research focus were organised chronologically into the most subdivided groups. We also summarise public datasets relevant to event-based 3D reconstruction. Finally, we highlight current research limitations in data availability, evaluation, representation, and dynamic scene handling, and outline promising future research directions. This survey aims to serve as a comprehensive reference and a roadmap for future developments in event-driven 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21650v2">HoloTime: Taming Video Diffusion Models for Panoramic 4D Scene Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 Project Homepage: https://zhouhyocean.github.io/holotime/ Code: https://github.com/PKU-YuanGroup/HoloTime
    </div>
    <details class="paper-abstract">
      The rapid advancement of diffusion models holds the promise of revolutionizing the application of VR and AR technologies, which typically require scene-level 4D assets for user experience. Nonetheless, existing diffusion models predominantly concentrate on modeling static 3D scenes or object-level dynamics, constraining their capacity to provide truly immersive experiences. To address this issue, we propose HoloTime, a framework that integrates video diffusion models to generate panoramic videos from a single prompt or reference image, along with a 360-degree 4D scene reconstruction method that seamlessly transforms the generated panoramic video into 4D assets, enabling a fully immersive 4D experience for users. Specifically, to tame video diffusion models for generating high-fidelity panoramic videos, we introduce the 360World dataset, the first comprehensive collection of panoramic videos suitable for downstream 4D scene reconstruction tasks. With this curated dataset, we propose Panoramic Animator, a two-stage image-to-video diffusion model that can convert panoramic images into high-quality panoramic videos. Following this, we present Panoramic Space-Time Reconstruction, which leverages a space-time depth estimation method to transform the generated panoramic videos into 4D point clouds, enabling the optimization of a holistic 4D Gaussian Splatting representation to reconstruct spatially and temporally consistent 4D scenes. To validate the efficacy of our method, we conducted a comparative analysis with existing approaches, revealing its superiority in both panoramic video generation and 4D scene reconstruction. This demonstrates our method's capability to create more engaging and realistic immersive environments, thereby enhancing user experiences in VR and AR applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08196v1">ADC-GS: Anchor-Driven Deformable and Compressed Gaussian Splatting for Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Existing 4D Gaussian Splatting methods rely on per-Gaussian deformation from a canonical space to target frames, which overlooks redundancy among adjacent Gaussian primitives and results in suboptimal performance. To address this limitation, we propose Anchor-Driven Deformable and Compressed Gaussian Splatting (ADC-GS), a compact and efficient representation for dynamic scene reconstruction. Specifically, ADC-GS organizes Gaussian primitives into an anchor-based structure within the canonical space, enhanced by a temporal significance-based anchor refinement strategy. To reduce deformation redundancy, ADC-GS introduces a hierarchical coarse-to-fine pipeline that captures motions at varying granularities. Moreover, a rate-distortion optimization is adopted to achieve an optimal balance between bitrate consumption and representation fidelity. Experimental results demonstrate that ADC-GS outperforms the per-Gaussian deformation approaches in rendering speed by 300%-800% while achieving state-of-the-art storage efficiency without compromising rendering quality. The code is released at https://github.com/H-Huang774/ADC-GS.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02283v4">GP-GS: Gaussian Processes for Enhanced Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 12 pages, 7 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has emerged as an efficient photorealistic novel view synthesis method. However, its reliance on sparse Structure-from-Motion (SfM) point clouds often limits scene reconstruction quality. To address the limitation, this paper proposes a novel 3D reconstruction framework, Gaussian Processes enhanced Gaussian Splatting (GP-GS), in which a multi-output Gaussian Process model is developed to enable adaptive and uncertainty-guided densification of sparse SfM point clouds. Specifically, we propose a dynamic sampling and filtering pipeline that adaptively expands the SfM point clouds by leveraging GP-based predictions to infer new candidate points from the input 2D pixels and depth maps. The pipeline utilizes uncertainty estimates to guide the pruning of high-variance predictions, ensuring geometric consistency and enabling the generation of dense point clouds. These densified point clouds provide high-quality initial 3D Gaussians, enhancing reconstruction performance. Extensive experiments conducted on synthetic and real-world datasets across various scales validate the effectiveness and practicality of the proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06587v2">Introducing Unbiased Depth into 2D Gaussian Splatting for High-accuracy Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 We found a major error in Sec. 4.3 Novel View Synthesis. We mistakenly used the test-set images in training for NVS experiments, making our results look better than they actually are
    </div>
    <details class="paper-abstract">
      Recently, 2D Gaussian Splatting (2DGS) has demonstrated superior geometry reconstruction quality than the popular 3DGS by using 2D surfels to approximate thin surfaces. However, it falls short when dealing with glossy surfaces, resulting in visible holes in these areas. We found the reflection discontinuity causes the issue. To fit the jump from diffuse to specular reflection at different viewing angles, depth bias is introduced in the optimized Gaussian primitives. To address that, we first replace the depth distortion loss in 2DGS with a novel depth convergence loss, which imposes a strong constraint on depth continuity. Then, we rectified the depth criterion in determining the actual surface, which fully accounts for all the intersecting Gaussians along the ray. Qualitative and quantitative evaluations across various datasets reveal that our method significantly improves reconstruction quality, with more complete and accurate surfaces than 2DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07539v1">GIFStream: 4D Gaussian-based Immersive Video with Feature Stream</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 14 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Immersive video offers a 6-Dof-free viewing experience, potentially playing a key role in future video technology. Recently, 4D Gaussian Splatting has gained attention as an effective approach for immersive video due to its high rendering efficiency and quality, though maintaining quality with manageable storage remains challenging. To address this, we introduce GIFStream, a novel 4D Gaussian representation using a canonical space and a deformation field enhanced with time-dependent feature streams. These feature streams enable complex motion modeling and allow efficient compression by leveraging temporal correspondence and motion-aware pruning. Additionally, we incorporate both temporal and spatial compression networks for end-to-end compression. Experimental results show that GIFStream delivers high-quality immersive video at 30 Mbps, with real-time rendering and fast decoding on an RTX 4090. Project page: https://xdimlab.github.io/GIFStream
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07396v1">TUM2TWIN: Introducing the Large-Scale Multimodal Urban Digital Twin Benchmark Dataset</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 Submitted to the ISPRS Journal of Photogrammetry and Remote Sensing
    </div>
    <details class="paper-abstract">
      Urban Digital Twins (UDTs) have become essential for managing cities and integrating complex, heterogeneous data from diverse sources. Creating UDTs involves challenges at multiple process stages, including acquiring accurate 3D source data, reconstructing high-fidelity 3D models, maintaining models' updates, and ensuring seamless interoperability to downstream tasks. Current datasets are usually limited to one part of the processing chain, hampering comprehensive UDTs validation. To address these challenges, we introduce the first comprehensive multimodal Urban Digital Twin benchmark dataset: TUM2TWIN. This dataset includes georeferenced, semantically aligned 3D models and networks along with various terrestrial, mobile, aerial, and satellite observations boasting 32 data subsets over roughly 100,000 $m^2$ and currently 767 GB of data. By ensuring georeferenced indoor-outdoor acquisition, high accuracy, and multimodal data integration, the benchmark supports robust analysis of sensors and the development of advanced reconstruction methods. Additionally, we explore downstream tasks demonstrating the potential of TUM2TWIN, including novel view synthesis of NeRF and Gaussian Splatting, solar potential analysis, point cloud semantic segmentation, and LoD3 building reconstruction. We are convinced this contribution lays a foundation for overcoming current limitations in UDT creation, fostering new research directions and practical solutions for smarter, data-driven urban environments. The project is available under: https://tum2t.win
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08124v1">SLAG: Scalable Language-Augmented Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Language-augmented scene representations hold great promise for large-scale robotics applications such as search-and-rescue, smart cities, and mining. Many of these scenarios are time-sensitive, requiring rapid scene encoding while also being data-intensive, necessitating scalable solutions. Deploying these representations on robots with limited computational resources further adds to the challenge. To address this, we introduce SLAG, a multi-GPU framework for language-augmented Gaussian splatting that enhances the speed and scalability of embedding large scenes. Our method integrates 2D visual-language model features into 3D scenes using SAM and CLIP. Unlike prior approaches, SLAG eliminates the need for a loss function to compute per-Gaussian language embeddings. Instead, it derives embeddings from 3D Gaussian scene parameters via a normalized weighted average, enabling highly parallelized scene encoding. Additionally, we introduce a vector database for efficient embedding storage and retrieval. Our experiments show that SLAG achieves an 18 times speedup in embedding computation on a 16-GPU setup compared to OpenGaussian, while preserving embedding quality on the ScanNet and LERF datasets. For more details, visit our project website: https://slag-project.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09040v2">Motion Blender Gaussian Splatting for Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Gaussian splatting has emerged as a powerful tool for high-fidelity reconstruction of dynamic scenes. However, existing methods primarily rely on implicit motion representations, such as encoding motions into neural networks or per-Gaussian parameters, which makes it difficult to further manipulate the reconstructed motions. This lack of explicit controllability limits existing methods to replaying recorded motions only, which hinders a wider application in robotics. To address this, we propose Motion Blender Gaussian Splatting (MBGS), a novel framework that uses motion graphs as an explicit and sparse motion representation. The motion of a graph's links is propagated to individual Gaussians via dual quaternion skinning, with learnable weight painting functions that determine the influence of each link. The motion graphs and 3D Gaussians are jointly optimized from input videos via differentiable rendering. Experiments show that MBGS achieves state-of-the-art performance on the highly challenging iPhone dataset while being competitive on HyperNeRF. We demonstrate the application potential of our method in animating novel object poses, synthesizing real robot demonstrations, and predicting robot actions through visual planning. The source code, models, video demonstrations can be found at http://mlzxy.github.io/motion-blender-gs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08811v1">TUGS: Physics-based Compact Representation of Underwater Scenes by Tensorized Gaussian</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Underwater 3D scene reconstruction is crucial for undewater robotic perception and navigation. However, the task is significantly challenged by the complex interplay between light propagation, water medium, and object surfaces, with existing methods unable to model their interactions accurately. Additionally, expensive training and rendering costs limit their practical application in underwater robotic systems. Therefore, we propose Tensorized Underwater Gaussian Splatting (TUGS), which can effectively solve the modeling challenges of the complex interactions between object geometries and water media while achieving significant parameter reduction. TUGS employs lightweight tensorized higher-order Gaussians with a physics-based underwater Adaptive Medium Estimation (AME) module, enabling accurate simulation of both light attenuation and backscatter effects in underwater environments. Compared to other NeRF-based and GS-based methods designed for underwater, TUGS is able to render high-quality underwater images with faster rendering speeds and less memory usage. Extensive experiments on real-world underwater datasets have demonstrated that TUGS can efficiently achieve superior reconstruction quality using a limited number of parameters, making it particularly suitable for memory-constrained underwater UAV applications
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06523v1">Virtualized 3D Gaussians: Flexible Cluster-based Level-of-Detail System for Real-Time Rendering of Composed Scenes</a></div>
    <div class="paper-meta">
      📅 2025-05-10
      | 💬 project page: https://xijie-yang.github.io/V3DG/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables the reconstruction of intricate digital 3D assets from multi-view images by leveraging a set of 3D Gaussian primitives for rendering. Its explicit and discrete representation facilitates the seamless composition of complex digital worlds, offering significant advantages over previous neural implicit methods. However, when applied to large-scale compositions, such as crowd-level scenes, it can encompass numerous 3D Gaussians, posing substantial challenges for real-time rendering. To address this, inspired by Unreal Engine 5's Nanite system, we propose Virtualized 3D Gaussians (V3DG), a cluster-based LOD solution that constructs hierarchical 3D Gaussian clusters and dynamically selects only the necessary ones to accelerate rendering speed. Our approach consists of two stages: (1) Offline Build, where hierarchical clusters are generated using a local splatting method to minimize visual differences across granularities, and (2) Online Selection, where footprint evaluation determines perceptible clusters for efficient rasterization during rendering. We curate a dataset of synthetic and real-world scenes, including objects, trees, people, and buildings, each requiring 0.1 billion 3D Gaussians to capture fine details. Experiments show that our solution balances rendering efficiency and visual quality across user-defined tolerances, facilitating downstream interactive applications that compose extensive 3DGS assets for consistent rendering performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05475v1">SVAD: From Single Image to 3D Avatar via Synthetic Data Generation with Video Diffusion and Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 Accepted by CVPR 2025 SyntaGen Workshop, Project Page: https://yc4ny.github.io/SVAD/
    </div>
    <details class="paper-abstract">
      Creating high-quality animatable 3D human avatars from a single image remains a significant challenge in computer vision due to the inherent difficulty of reconstructing complete 3D information from a single viewpoint. Current approaches face a clear limitation: 3D Gaussian Splatting (3DGS) methods produce high-quality results but require multiple views or video sequences, while video diffusion models can generate animations from single images but struggle with consistency and identity preservation. We present SVAD, a novel approach that addresses these limitations by leveraging complementary strengths of existing techniques. Our method generates synthetic training data through video diffusion, enhances it with identity preservation and image restoration modules, and utilizes this refined data to train 3DGS avatars. Comprehensive evaluations demonstrate that SVAD outperforms state-of-the-art (SOTA) single-image methods in maintaining identity consistency and fine details across novel poses and viewpoints, while enabling real-time rendering capabilities. Through our data augmentation pipeline, we overcome the dependency on dense monocular or multi-view training data typically required by traditional 3DGS approaches. Extensive quantitative, qualitative comparisons show our method achieves superior performance across multiple metrics against baseline models. By effectively combining the generative power of diffusion models with both the high-quality results and rendering efficiency of 3DGS, our work establishes a new approach for high-fidelity avatar generation from a single image input.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05356v1">Time of the Flight of the Gaussians: Optimizing Depth Indirectly in Dynamic Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      We present a method to reconstruct dynamic scenes from monocular continuous-wave time-of-flight (C-ToF) cameras using raw sensor samples that achieves similar or better accuracy than neural volumetric approaches and is 100x faster. Quickly achieving high-fidelity dynamic 3D reconstruction from a single viewpoint is a significant challenge in computer vision. In C-ToF radiance field reconstruction, the property of interest-depth-is not directly measured, causing an additional challenge. This problem has a large and underappreciated impact upon the optimization when using a fast primitive-based scene representation like 3D Gaussian splatting, which is commonly used with multi-view data to produce satisfactory results and is brittle in its optimization otherwise. We incorporate two heuristics into the optimization to improve the accuracy of scene geometry represented by Gaussians. Experimental results show that our approach produces accurate reconstructions under constrained C-ToF sensing conditions, including for fast motions like swinging baseball bats. https://visual.cs.brown.edu/gftorf
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17378v4">Balanced 3DGS: Gaussian-wise Parallelism Rendering with Fine-Grained Tiling</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is increasingly attracting attention in both academia and industry owing to its superior visual quality and rendering speed. However, training a 3DGS model remains a time-intensive task, especially in load imbalance scenarios where workload diversity among pixels and Gaussian spheres causes poor renderCUDA kernel performance. We introduce Balanced 3DGS, a Gaussian-wise parallelism rendering with fine-grained tiling approach in 3DGS training process, perfectly solving load-imbalance issues. First, we innovatively introduce the inter-block dynamic workload distribution technique to map workloads to Streaming Multiprocessor(SM) resources within a single GPU dynamically, which constitutes the foundation of load balancing. Second, we are the first to propose the Gaussian-wise parallel rendering technique to significantly reduce workload divergence inside a warp, which serves as a critical component in addressing load imbalance. Based on the above two methods, we further creatively put forward the fine-grained combined load balancing technique to uniformly distribute workload across all SMs, which boosts the forward renderCUDA kernel performance by up to 7.52x. Besides, we present a self-adaptive render kernel selection strategy during the 3DGS training process based on different load-balance situations, which effectively improves training efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14423v4">PhysFlow: Unleashing the Potential of Multi-modal Foundation Models and Video Diffusion for 4D Dynamic Physical Scene Simulation</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 CVPR 2025. Homepage: https://zhuomanliu.github.io/PhysFlow/
    </div>
    <details class="paper-abstract">
      Realistic simulation of dynamic scenes requires accurately capturing diverse material properties and modeling complex object interactions grounded in physical principles. However, existing methods are constrained to basic material types with limited predictable parameters, making them insufficient to represent the complexity of real-world materials. We introduce PhysFlow, a novel approach that leverages multi-modal foundation models and video diffusion to achieve enhanced 4D dynamic scene simulation. Our method utilizes multi-modal models to identify material types and initialize material parameters through image queries, while simultaneously inferring 3D Gaussian splats for detailed scene representation. We further refine these material parameters using video diffusion with a differentiable Material Point Method (MPM) and optical flow guidance rather than render loss or Score Distillation Sampling (SDS) loss. This integrated framework enables accurate prediction and realistic simulation of dynamic interactions in real-world scenarios, advancing both accuracy and flexibility in physics-based simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05672v1">TeGA: Texture Space Gaussian Avatars for High-Resolution Dynamic Head Modeling</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 10 pages, 9 figures, supplementary results found at: https://syntec-research.github.io/UVGA/, to be published in SIGGRAPH 2025
    </div>
    <details class="paper-abstract">
      Sparse volumetric reconstruction and rendering via 3D Gaussian splatting have recently enabled animatable 3D head avatars that are rendered under arbitrary viewpoints with impressive photorealism. Today, such photoreal avatars are seen as a key component in emerging applications in telepresence, extended reality, and entertainment. Building a photoreal avatar requires estimating the complex non-rigid motion of different facial components as seen in input video images; due to inaccurate motion estimation, animatable models typically present a loss of fidelity and detail when compared to their non-animatable counterparts, built from an individual facial expression. Also, recent state-of-the-art models are often affected by memory limitations that reduce the number of 3D Gaussians used for modeling, leading to lower detail and quality. To address these problems, we present a new high-detail 3D head avatar model that improves upon the state of the art, largely increasing the number of 3D Gaussians and modeling quality for rendering at 4K resolution. Our high-quality model is reconstructed from multiview input video and builds on top of a mesh-based 3D morphable model, which provides a coarse deformation layer for the head. Photoreal appearance is modelled by 3D Gaussians embedded within the continuous UVD tangent space of this mesh, allowing for more effective densification where most needed. Additionally, these Gaussians are warped by a novel UVD deformation field to capture subtle, localized motion. Our key contribution is the novel deformable Gaussian encoding and overall fitting procedure that allows our head model to preserve appearance detail, while capturing facial motion and other transient high-frequency features such as skin wrinkling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05643v1">UltraGauss: Ultrafast Gaussian Reconstruction of 3D Ultrasound Volumes</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      Ultrasound imaging is widely used due to its safety, affordability, and real-time capabilities, but its 2D interpretation is highly operator-dependent, leading to variability and increased cognitive demand. 2D-to-3D reconstruction mitigates these challenges by providing standardized volumetric views, yet existing methods are often computationally expensive, memory-intensive, or incompatible with ultrasound physics. We introduce UltraGauss: the first ultrasound-specific Gaussian Splatting framework, extending view synthesis techniques to ultrasound wave propagation. Unlike conventional perspective-based splatting, UltraGauss models probe-plane intersections in 3D, aligning with acoustic image formation. We derive an efficient rasterization boundary formulation for GPU parallelization and introduce a numerically stable covariance parametrization, improving computational efficiency and reconstruction accuracy. On real clinical ultrasound data, UltraGauss achieves state-of-the-art reconstructions in 5 minutes, and reaching 0.99 SSIM within 20 minutes on a single GPU. A survey of expert clinicians confirms UltraGauss' reconstructions are the most realistic among competing methods. Our CUDA implementation will be released upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05591v1">QuickSplat: Fast 3D Surface Reconstruction via Learned Gaussian Initialization</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 Project page: https://liu115.github.io/quicksplat, Video: https://youtu.be/2IA_gnFvFG8
    </div>
    <details class="paper-abstract">
      Surface reconstruction is fundamental to computer vision and graphics, enabling applications in 3D modeling, mixed reality, robotics, and more. Existing approaches based on volumetric rendering obtain promising results, but optimize on a per-scene basis, resulting in a slow optimization that can struggle to model under-observed or textureless regions. We introduce QuickSplat, which learns data-driven priors to generate dense initializations for 2D gaussian splatting optimization of large-scale indoor scenes. This provides a strong starting point for the reconstruction, which accelerates the convergence of the optimization and improves the geometry of flat wall structures. We further learn to jointly estimate the densification and update of the scene parameters during each iteration; our proposed densifier network predicts new Gaussians based on the rendering gradients of existing ones, removing the needs of heuristics for densification. Extensive experiments on large-scale indoor scene reconstruction demonstrate the superiority of our data-driven optimization. Concretely, we accelerate runtime by 8x, while decreasing depth errors by up to 48% in comparison to state of the art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05587v1">Steepest Descent Density Control for Compact 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 CVPR 2025, Project page: https://vita-group.github.io/SteepGS/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for real-time, high-resolution novel view synthesis. By representing scenes as a mixture of Gaussian primitives, 3DGS leverages GPU rasterization pipelines for efficient rendering and reconstruction. To optimize scene coverage and capture fine details, 3DGS employs a densification algorithm to generate additional points. However, this process often leads to redundant point clouds, resulting in excessive memory usage, slower performance, and substantial storage demands - posing significant challenges for deployment on resource-constrained devices. To address this limitation, we propose a theoretical framework that demystifies and improves density control in 3DGS. Our analysis reveals that splitting is crucial for escaping saddle points. Through an optimization-theoretic approach, we establish the necessary conditions for densification, determine the minimal number of offspring Gaussians, identify the optimal parameter update direction, and provide an analytical solution for normalizing off-spring opacity. Building on these insights, we introduce SteepGS, incorporating steepest density control, a principled strategy that minimizes loss while maintaining a compact point cloud. SteepGS achieves a ~50% reduction in Gaussian points without compromising rendering quality, significantly enhancing both efficiency and scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04262v1">Bridging Geometry-Coherent Text-to-3D Generation with Multi-View Diffusion Priors and Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Score Distillation Sampling (SDS) leverages pretrained 2D diffusion models to advance text-to-3D generation but neglects multi-view correlations, being prone to geometric inconsistencies and multi-face artifacts in the generated 3D content. In this work, we propose Coupled Score Distillation (CSD), a framework that couples multi-view joint distribution priors to ensure geometrically consistent 3D generation while enabling the stable and direct optimization of 3D Gaussian Splatting. Specifically, by reformulating the optimization as a multi-view joint optimization problem, we derive an effective optimization rule that effectively couples multi-view priors to guide optimization across different viewpoints while preserving the diversity of generated 3D assets. Additionally, we propose a framework that directly optimizes 3D Gaussian Splatting (3D-GS) with random initialization to generate geometrically consistent 3D content. We further employ a deformable tetrahedral grid, initialized from 3D-GS and refined through CSD, to produce high-quality, refined meshes. Quantitative and qualitative experimental results demonstrate the efficiency and competitive quality of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22676v2">TranSplat: Lighting-Consistent Cross-Scene Object Transfer with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      We present TranSplat, a 3D scene rendering algorithm that enables realistic cross-scene object transfer (from a source to a target scene) based on the Gaussian Splatting framework. Our approach addresses two critical challenges: (1) precise 3D object extraction from the source scene, and (2) faithful relighting of the transferred object in the target scene without explicit material property estimation. TranSplat fits a splatting model to the source scene, using 2D object masks to drive fine-grained 3D segmentation. Following user-guided insertion of the object into the target scene, along with automatic refinement of position and orientation, TranSplat derives per-Gaussian radiance transfer functions via spherical harmonic analysis to adapt the object's appearance to match the target scene's lighting environment. This relighting strategy does not require explicitly estimating physical scene properties such as BRDFs. Evaluated on several synthetic and real-world scenes and objects, TranSplat yields excellent 3D object extractions and relighting performance compared to recent baseline methods and visually convincing cross-scene object transfers. We conclude by discussing the limitations of the approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00814v4">VR-Doh: Hands-on 3D Modeling in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      We introduce VR-Doh, an open-source, hands-on 3D modeling system that enables intuitive creation and manipulation of elastoplastic objects in Virtual Reality (VR). By customizing the Material Point Method (MPM) for real-time simulation of hand-induced large deformations and enhancing 3D Gaussian Splatting for seamless rendering, VR-Doh provides an interactive and immersive 3D modeling experience. Users can naturally sculpt, deform, and edit objects through both contact- and gesture-based hand-object interactions. To achieve real-time performance, our system incorporates localized simulation techniques, particle-level collision handling, and the decoupling of physical and appearance representations, ensuring smooth and responsive interactions. VR-Doh supports both object creation and editing, enabling diverse modeling tasks such as designing food items, characters, and interlocking structures, all resulting in simulation-ready assets. User studies with both novice and experienced participants highlight the system's intuitive design, immersive feedback, and creative potential. Compared to existing geometric modeling tools, VR-Doh offers enhanced accessibility and natural interaction, making it a powerful tool for creative exploration in VR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04668v1">SGCR: Spherical Gaussians for Efficient 3D Curve Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 The IEEE/CVF Conference on Computer Vision and Pattern Recognition 2025, 8 pages
    </div>
    <details class="paper-abstract">
      Neural rendering techniques have made substantial progress in generating photo-realistic 3D scenes. The latest 3D Gaussian Splatting technique has achieved high quality novel view synthesis as well as fast rendering speed. However, 3D Gaussians lack proficiency in defining accurate 3D geometric structures despite their explicit primitive representations. This is due to the fact that Gaussian's attributes are primarily tailored and fine-tuned for rendering diverse 2D images by their anisotropic nature. To pave the way for efficient 3D reconstruction, we present Spherical Gaussians, a simple and effective representation for 3D geometric boundaries, from which we can directly reconstruct 3D feature curves from a set of calibrated multi-view images. Spherical Gaussians is optimized from grid initialization with a view-based rendering loss, where a 2D edge map is rendered at a specific view and then compared to the ground-truth edge map extracted from the corresponding image, without the need for any 3D guidance or supervision. Given Spherical Gaussians serve as intermedia for the robust edge representation, we further introduce a novel optimization-based algorithm called SGCR to directly extract accurate parametric curves from aligned Spherical Gaussians. We demonstrate that SGCR outperforms existing state-of-the-art methods in 3D edge reconstruction while enjoying great efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04659v1">GSsplat: Generalizable Semantic Gaussian Splatting for Novel-view Synthesis in 3D Scenes</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      The semantic synthesis of unseen scenes from multiple viewpoints is crucial for research in 3D scene understanding. Current methods are capable of rendering novel-view images and semantic maps by reconstructing generalizable Neural Radiance Fields. However, they often suffer from limitations in speed and segmentation performance. We propose a generalizable semantic Gaussian Splatting method (GSsplat) for efficient novel-view synthesis. Our model predicts the positions and attributes of scene-adaptive Gaussian distributions from once input, replacing the densification and pruning processes of traditional scene-specific Gaussian Splatting. In the multi-task framework, a hybrid network is designed to extract color and semantic information and predict Gaussian parameters. To augment the spatial perception of Gaussians for high-quality rendering, we put forward a novel offset learning module through group-based supervision and a point-level interaction module with spatial unit aggregation. When evaluated with varying numbers of multi-view inputs, GSsplat achieves state-of-the-art performance for semantic synthesis at the fastest speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00814v3">VR-Doh: Hands-on 3D Modeling in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      We introduce VR-Doh, an open-source, hands-on 3D modeling system that enables intuitive creation and manipulation of elastoplastic objects in Virtual Reality (VR). By customizing the Material Point Method (MPM) for real-time simulation of hand-induced large deformations and enhancing 3D Gaussian Splatting for seamless rendering, VR-Doh provides an interactive and immersive 3D modeling experience. Users can naturally sculpt, deform, and edit objects through both contact- and gesture-based hand-object interactions. To achieve real-time performance, our system incorporates localized simulation techniques, particle-level collision handling, and the decoupling of physical and appearance representations, ensuring smooth and responsive interactions. VR-Doh supports both object creation and editing, enabling diverse modeling tasks such as designing food items, characters, and interlocking structures, all resulting in simulation-ready assets. User studies with both novice and experienced participants highlight the system's intuitive design, immersive feedback, and creative potential. Compared to existing geometric modeling tools, VR-Doh offers enhanced accessibility and natural interaction, making it a powerful tool for creative exploration in VR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03310v1">3D Gaussian Splatting Data Compression with Mixture of Priors</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) data compression is crucial for enabling efficient storage and transmission in 3D scene modeling. However, its development remains limited due to inadequate entropy models and suboptimal quantization strategies for both lossless and lossy compression scenarios, where existing methods have yet to 1) fully leverage hyperprior information to construct robust conditional entropy models, and 2) apply fine-grained, element-wise quantization strategies for improved compression granularity. In this work, we propose a novel Mixture of Priors (MoP) strategy to simultaneously address these two challenges. Specifically, inspired by the Mixture-of-Experts (MoE) paradigm, our MoP approach processes hyperprior information through multiple lightweight MLPs to generate diverse prior features, which are subsequently integrated into the MoP feature via a gating mechanism. To enhance lossless compression, the resulting MoP feature is utilized as a hyperprior to improve conditional entropy modeling. Meanwhile, for lossy compression, we employ the MoP feature as guidance information in an element-wise quantization procedure, leveraging a prior-guided Coarse-to-Fine Quantization (C2FQ) strategy with a predefined quantization step value. Specifically, we expand the quantization step value into a matrix and adaptively refine it from coarse to fine granularity, guided by the MoP feature, thereby obtaining a quantization step matrix that facilitates element-wise quantization. Extensive experiments demonstrate that our proposed 3DGS data compression framework achieves state-of-the-art performance across multiple benchmarks, including Mip-NeRF360, BungeeNeRF, DeepBlending, and Tank&Temples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18630v2">Deformable Beta Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 SIGGRAPH 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has advanced radiance field reconstruction by enabling real-time rendering. However, its reliance on Gaussian kernels for geometry and low-order Spherical Harmonics (SH) for color encoding limits its ability to capture complex geometries and diverse colors. We introduce Deformable Beta Splatting (DBS), a deformable and compact approach that enhances both geometry and color representation. DBS replaces Gaussian kernels with deformable Beta Kernels, which offer bounded support and adaptive frequency control to capture fine geometric details with higher fidelity while achieving better memory efficiency. In addition, we extended the Beta Kernel to color encoding, which facilitates improved representation of diffuse and specular components, yielding superior results compared to SH-based methods. Furthermore, Unlike prior densification techniques that depend on Gaussian properties, we mathematically prove that adjusting regularized opacity alone ensures distribution-preserved Markov chain Monte Carlo (MCMC), independent of the splatting kernel type. Experimental results demonstrate that DBS achieves state-of-the-art visual quality while utilizing only 45% of the parameters and rendering 1.5x faster than 3DGS-MCMC, highlighting the superior performance of DBS for real-time radiance field rendering. Interactive demonstrations and source code are available on our project website: https://rongliu-leo.github.io/beta-splatting/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07577v2">3D Vision-Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 Accepted at ICLR 2025. Main paper + supplementary material
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D reconstruction methods and vision-language models have propelled the development of multi-modal 3D scene understanding, which has vital applications in robotics, autonomous driving, and virtual/augmented reality. However, current multi-modal scene understanding approaches have naively embedded semantic representations into 3D reconstruction methods without striking a balance between visual and language modalities, which leads to unsatisfying semantic rasterization of translucent or reflective objects, as well as over-fitting on color modality. To alleviate these limitations, we propose a solution that adequately handles the distinct visual and semantic modalities, i.e., a 3D vision-language Gaussian splatting model for scene understanding, to put emphasis on the representation learning of language modality. We propose a novel cross-modal rasterizer, using modality fusion along with a smoothed semantic indicator for enhancing semantic rasterization. We also employ a camera-view blending technique to improve semantic consistency between existing and synthesized views, thereby effectively mitigating over-fitting. Extensive experiments demonstrate that our method achieves state-of-the-art performance in open-vocabulary semantic segmentation, surpassing existing methods by a significant margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18468v3">RGS-DR: Reflective Gaussian Surfels with Deferred Rendering for Shiny Objects</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      We introduce RGS-DR, a novel inverse rendering method for reconstructing and rendering glossy and reflective objects with support for flexible relighting and scene editing. Unlike existing methods (e.g., NeRF and 3D Gaussian Splatting), which struggle with view-dependent effects, RGS-DR utilizes a 2D Gaussian surfel representation to accurately estimate geometry and surface normals, an essential property for high-quality inverse rendering. Our approach explicitly models geometric and material properties through learnable primitives rasterized into a deferred shading pipeline, effectively reducing rendering artifacts and preserving sharp reflections. By employing a multi-level cube mipmap, RGS-DR accurately approximates environment lighting integrals, facilitating high-quality reconstruction and relighting. A residual pass with spherical-mipmap-based directional encoding further refines the appearance modeling. Experiments demonstrate that RGS-DR achieves high-quality reconstruction and rendering quality for shiny objects, often outperforming reconstruction-exclusive state-of-the-art methods incapable of relighting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00159v2">SonarSplat: Novel View Synthesis of Imaging Sonar via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-04
    </div>
    <details class="paper-abstract">
      In this paper, we present SonarSplat, a novel Gaussian splatting framework for imaging sonar that demonstrates realistic novel view synthesis and models acoustic streaking phenomena. Our method represents the scene as a set of 3D Gaussians with acoustic reflectance and saturation properties. We develop a novel method to efficiently rasterize Gaussians to produce a range/azimuth image that is faithful to the acoustic image formation model of imaging sonar. In particular, we develop a novel approach to model azimuth streaking in a Gaussian splatting framework. We evaluate SonarSplat using real-world datasets of sonar images collected from an underwater robotic platform in a controlled test tank and in a real-world river environment. Compared to the state-of-the-art, SonarSplat offers improved image synthesis capabilities (+3.2 dB PSNR) and more accurate 3D reconstruction (52% lower Chamfer Distance). We also demonstrate that SonarSplat can be leveraged for azimuth streak removal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02178v1">Sparfels: Fast Reconstruction from Sparse Unposed Imagery</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 Project page : https://shubhendu-jena.github.io/Sparfels/
    </div>
    <details class="paper-abstract">
      We present a method for Sparse view reconstruction with surface element splatting that runs within 3 minutes on a consumer grade GPU. While few methods address sparse radiance field learning from noisy or unposed sparse cameras, shape recovery remains relatively underexplored in this setting. Several radiance and shape learning test-time optimization methods address the sparse posed setting by learning data priors or using combinations of external monocular geometry priors. Differently, we propose an efficient and simple pipeline harnessing a single recent 3D foundation model. We leverage its various task heads, notably point maps and camera initializations to instantiate a bundle adjusting 2D Gaussian Splatting (2DGS) model, and image correspondences to guide camera optimization midst 2DGS training. Key to our contribution is a novel formulation of splatted color variance along rays, which can be computed efficiently. Reducing this moment in training leads to more accurate shape reconstructions. We demonstrate state-of-the-art performances in the sparse uncalibrated setting in reconstruction and novel view benchmarks based on established multi-view datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02175v1">SparSplat: Fast Multi-View Reconstruction with Generalizable 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 Project page : https://shubhendu-jena.github.io/SparSplat/
    </div>
    <details class="paper-abstract">
      Recovering 3D information from scenes via multi-view stereo reconstruction (MVS) and novel view synthesis (NVS) is inherently challenging, particularly in scenarios involving sparse-view setups. The advent of 3D Gaussian Splatting (3DGS) enabled real-time, photorealistic NVS. Following this, 2D Gaussian Splatting (2DGS) leveraged perspective accurate 2D Gaussian primitive rasterization to achieve accurate geometry representation during rendering, improving 3D scene reconstruction while maintaining real-time performance. Recent approaches have tackled the problem of sparse real-time NVS using 3DGS within a generalizable, MVS-based learning framework to regress 3D Gaussian parameters. Our work extends this line of research by addressing the challenge of generalizable sparse 3D reconstruction and NVS jointly, and manages to perform successfully at both tasks. We propose an MVS-based learning pipeline that regresses 2DGS surface element parameters in a feed-forward fashion to perform 3D shape reconstruction and NVS from sparse-view images. We further show that our generalizable pipeline can benefit from preexisting foundational multi-view deep visual features. The resulting model attains the state-of-the-art results on the DTU sparse 3D reconstruction benchmark in terms of Chamfer distance to ground-truth, as-well as state-of-the-art NVS. It also demonstrates strong generalization on the BlendedMVS and Tanks and Temples datasets. We note that our model outperforms the prior state-of-the-art in feed-forward sparse view reconstruction based on volume rendering of implicit representations, while offering an almost 2 orders of magnitude higher inference speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02126v1">GarmentGS: Point-Cloud Guided Gaussian Splatting for High-Fidelity Non-Watertight 3D Garment Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-05-04
    </div>
    <details class="paper-abstract">
      Traditional 3D garment creation requires extensive manual operations, resulting in time and labor costs. Recently, 3D Gaussian Splatting has achieved breakthrough progress in 3D scene reconstruction and rendering, attracting widespread attention and opening new pathways for 3D garment reconstruction. However, due to the unstructured and irregular nature of Gaussian primitives, it is difficult to reconstruct high-fidelity, non-watertight 3D garments. In this paper, we present GarmentGS, a dense point cloud-guided method that can reconstruct high-fidelity garment surfaces with high geometric accuracy and generate non-watertight, single-layer meshes. Our method introduces a fast dense point cloud reconstruction module that can complete garment point cloud reconstruction in 10 minutes, compared to traditional methods that require several hours. Furthermore, we use dense point clouds to guide the movement, flattening, and rotation of Gaussian primitives, enabling better distribution on the garment surface to achieve superior rendering effects and geometric accuracy. Through numerical and visual comparisons, our method achieves fast training and real-time rendering while maintaining competitive quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02108v1">SignSplat: Rendering Sign Language via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-04
    </div>
    <details class="paper-abstract">
      State-of-the-art approaches for conditional human body rendering via Gaussian splatting typically focus on simple body motions captured from many views. This is often in the context of dancing or walking. However, for more complex use cases, such as sign language, we care less about large body motion and more about subtle and complex motions of the hands and face. The problems of building high fidelity models are compounded by the complexity of capturing multi-view data of sign. The solution is to make better use of sequence data, ensuring that we can overcome the limited information from only a few views by exploiting temporal variability. Nevertheless, learning from sequence-level data requires extremely accurate and consistent model fitting to ensure that appearance is consistent across complex motions. We focus on how to achieve this, constraining mesh parameters to build an accurate Gaussian splatting framework from few views capable of modelling subtle human motion. We leverage regularization techniques on the Gaussian parameters to mitigate overfitting and rendering artifacts. Additionally, we propose a new adaptive control method to densify Gaussians and prune splat points on the mesh surface. To demonstrate the accuracy of our approach, we render novel sequences of sign language video, building on neural machine translation approaches to sign stitching. On benchmark datasets, our approach achieves state-of-the-art performance; and on highly articulated and complex sign language motion, we significantly outperform competing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02720v4">3D-HGS: 3D Half-Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 8 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Photo-realistic image rendering from 3D scene reconstruction has advanced significantly with neural rendering techniques. Among these, 3D Gaussian Splatting (3D-GS) outperforms Neural Radiance Fields (NeRFs) in quality and speed but struggles with shape and color discontinuities. We propose 3D Half-Gaussian (3D-HGS) kernels as a plug-and-play solution to address these limitations. Our experiments show that 3D-HGS enhances existing 3D-GS methods, achieving state-of-the-art rendering quality without compromising speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01938v1">HybridGS: High-Efficiency Gaussian Splatting Data Compression using Dual-Channel Sparse Representation and Point Cloud Encoder</a></div>
    <div class="paper-meta">
      📅 2025-05-03
      | 💬 Accepted by ICML2025
    </div>
    <details class="paper-abstract">
      Most existing 3D Gaussian Splatting (3DGS) compression schemes focus on producing compact 3DGS representation via implicit data embedding. They have long coding times and highly customized data format, making it difficult for widespread deployment. This paper presents a new 3DGS compression framework called HybridGS, which takes advantage of both compact generation and standardized point cloud data encoding. HybridGS first generates compact and explicit 3DGS data. A dual-channel sparse representation is introduced to supervise the primitive position and feature bit depth. It then utilizes a canonical point cloud encoder to perform further data compression and form standard output bitstreams. A simple and effective rate control scheme is proposed to pivot the interpretable data compression scheme. At the current stage, HybridGS does not include any modules aimed at improving 3DGS quality during generation. But experiment results show that it still provides comparable reconstruction performance against state-of-the-art methods, with evidently higher encoding and decoding speed. The code is publicly available at https://github.com/Qi-Yangsjtu/HybridGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01928v1">GenSync: A Generalized Talking Head Framework for Audio-driven Multi-Subject Lip-Sync using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-03
    </div>
    <details class="paper-abstract">
      We introduce GenSync, a novel framework for multi-identity lip-synced video synthesis using 3D Gaussian Splatting. Unlike most existing 3D methods that require training a new model for each identity , GenSync learns a unified network that synthesizes lip-synced videos for multiple speakers. By incorporating a Disentanglement Module, our approach separates identity-specific features from audio representations, enabling efficient multi-identity video synthesis. This design reduces computational overhead and achieves 6.8x faster training compared to state-of-the-art models, while maintaining high lip-sync accuracy and visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01869v1">Visual enhancement and 3D representation for underwater scenes: a review</a></div>
    <div class="paper-meta">
      📅 2025-05-03
    </div>
    <details class="paper-abstract">
      Underwater visual enhancement (UVE) and underwater 3D reconstruction pose significant challenges in computer vision and AI-based tasks due to complex imaging conditions in aquatic environments. Despite the development of numerous enhancement algorithms, a comprehensive and systematic review covering both UVE and underwater 3D reconstruction remains absent. To advance research in these areas, we present an in-depth review from multiple perspectives. First, we introduce the fundamental physical models, highlighting the peculiarities that challenge conventional techniques. We survey advanced methods for visual enhancement and 3D reconstruction specifically designed for underwater scenarios. The paper assesses various approaches from non-learning methods to advanced data-driven techniques, including Neural Radiance Fields and 3D Gaussian Splatting, discussing their effectiveness in handling underwater distortions. Finally, we conduct both quantitative and qualitative evaluations of state-of-the-art UVE and underwater 3D reconstruction algorithms across multiple benchmark datasets. Finally, we highlight key research directions for future advancements in underwater vision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16693v2">PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-05-03
      | 💬 Robotics: Science and Systems 2025
    </div>
    <details class="paper-abstract">
      While non-prehensile manipulation (e.g., controlled pushing/poking) constitutes a foundational robotic skill, its learning remains challenging due to the high sensitivity to complex physical interactions involving friction and restitution. To achieve robust policy learning and generalization, we opt to learn a world model of the 3D rigid body dynamics involved in non-prehensile manipulations and use it for model-based reinforcement learning. We propose PIN-WM, a Physics-INformed World Model that enables efficient end-to-end identification of a 3D rigid body dynamical system from visual observations. Adopting differentiable physics simulation, PIN-WM can be learned with only few-shot and task-agnostic physical interaction trajectories. Further, PIN-WM is learned with observational loss induced by Gaussian Splatting without needing state estimation. To bridge Sim2Real gaps, we turn the learned PIN-WM into a group of Digital Cousins via physics-aware randomizations which perturb physics and rendering parameters to generate diverse and meaningful variations of the PIN-WM. Extensive evaluations on both simulation and real-world tests demonstrate that PIN-WM, enhanced with physics-aware digital cousins, facilitates learning robust non-prehensile manipulation skills with Sim2Real transfer, surpassing the Real2Sim2Real state-of-the-arts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01799v1">AquaGS: Fast Underwater Scene Reconstruction with SfM-Free Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-03
    </div>
    <details class="paper-abstract">
      Underwater scene reconstruction is a critical tech-nology for underwater operations, enabling the generation of 3D models from images captured by underwater platforms. However, the quality of underwater images is often degraded due to medium interference, which limits the effectiveness of Structure-from-Motion (SfM) pose estimation, leading to subsequent reconstruction failures. Additionally, SfM methods typically operate at slower speeds, further hindering their applicability in real-time scenarios. In this paper, we introduce AquaGS, an SfM-free underwater scene reconstruction model based on the SeaThru algorithm, which facilitates rapid and accurate separation of scene details and medium features. Our approach initializes Gaussians by integrating state-of-the-art multi-view stereo (MVS) technology, employs implicit Neural Radiance Fields (NeRF) for rendering translucent media and utilizes the latest explicit 3D Gaussian Splatting (3DGS) technique to render object surfaces, which effectively addresses the limitations of traditional methods and accurately simulates underwater optical phenomena. Experimental results on the data set and the robot platform show that our model can complete high-precision reconstruction in 30 seconds with only 3 image inputs, significantly enhancing the practical application of the algorithm in robotic platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01383v1">FalconWing: An Open-Source Platform for Ultra-Light Fixed-Wing Aircraft Research</a></div>
    <div class="paper-meta">
      📅 2025-05-02
    </div>
    <details class="paper-abstract">
      We present FalconWing -- an open-source, ultra-lightweight (150 g) fixed-wing platform for autonomy research. The hardware platform integrates a small camera, a standard airframe, offboard computation, and radio communication for manual overrides. We demonstrate FalconWing's capabilities by developing and deploying a purely vision-based control policy for autonomous landing (without IMU or motion capture) using a novel real-to-sim-to-real learning approach. Our learning approach: (1) constructs a photorealistic simulation environment via 3D Gaussian splatting trained on real-world images; (2) identifies nonlinear dynamics from vision-estimated real-flight data; and (3) trains a multi-modal Vision Transformer (ViT) policy through simulation-only imitation learning. The ViT architecture fuses single RGB image with the history of control actions via self-attention, preserving temporal context while maintaining real-time 20 Hz inference. When deployed zero-shot on the hardware platform, this policy achieves an 80% success rate in vision-based autonomous landings. Together with the hardware specifications, we also open-source the system dynamics, the software for photorealistic simulator and the learning approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01235v1">Compensating Spatiotemporally Inconsistent Observations for Online Dynamic 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-02
      | 💬 SIGGRAPH 2025, Project page: https://bbangsik13.github.io/OR2
    </div>
    <details class="paper-abstract">
      Online reconstruction of dynamic scenes is significant as it enables learning scenes from live-streaming video inputs, while existing offline dynamic reconstruction methods rely on recorded video inputs. However, previous online reconstruction approaches have primarily focused on efficiency and rendering quality, overlooking the temporal consistency of their results, which often contain noticeable artifacts in static regions. This paper identifies that errors such as noise in real-world recordings affect temporal inconsistency in online reconstruction. We propose a method that enhances temporal consistency in online reconstruction from observations with temporal inconsistency which is inevitable in cameras. We show that our method restores the ideal observation by subtracting the learned error. We demonstrate that applying our method to various baselines significantly enhances both temporal consistency and rendering quality across datasets. Code, video results, and checkpoints are available at https://bbangsik13.github.io/OR2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00362v2">UDGS-SLAM : UniDepth Assisted Gaussian Splatting for Monocular SLAM</a></div>
    <div class="paper-meta">
      📅 2025-05-02
    </div>
    <details class="paper-abstract">
      Recent advancements in monocular neural depth estimation, particularly those achieved by the UniDepth network, have prompted the investigation of integrating UniDepth within a Gaussian splatting framework for monocular SLAM. This study presents UDGS-SLAM, a novel approach that eliminates the necessity of RGB-D sensors for depth estimation within Gaussian splatting framework. UDGS-SLAM employs statistical filtering to ensure local consistency of the estimated depth and jointly optimizes camera trajectory and Gaussian scene representation parameters. The proposed method achieves high-fidelity rendered images and low ATERMSE of the camera trajectory. The performance of UDGS-SLAM is rigorously evaluated using the TUM RGB-D dataset and benchmarked against several baseline methods, demonstrating superior performance across various scenarios. Additionally, an ablation study is conducted to validate design choices and investigate the impact of different network backbone encoders on system performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15122v3">MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video</a></div>
    <div class="paper-meta">
      📅 2025-05-02
      | 💬 The first two authors contributed equally to this work (equal contribution). The last two authors are co-corresponding authors. Please visit our project page at https://kaist-viclab.github.io/mobgs-site/
    </div>
    <details class="paper-abstract">
      We present MoBGS, a novel deblurring dynamic 3D Gaussian Splatting (3DGS) framework capable of reconstructing sharp and high-quality novel spatio-temporal views from blurry monocular videos in an end-to-end manner. Existing dynamic novel view synthesis (NVS) methods are highly sensitive to motion blur in casually captured videos, resulting in significant degradation of rendering quality. While recent approaches address motion-blurred inputs for NVS, they primarily focus on static scene reconstruction and lack dedicated motion modeling for dynamic objects. To overcome these limitations, our MoBGS introduces a novel Blur-adaptive Latent Camera Estimation (BLCE) method for effective latent camera trajectory estimation, improving global camera motion deblurring. In addition, we propose a physically-inspired Latent Camera-induced Exposure Estimation (LCEE) method to ensure consistent deblurring of both global camera and local object motion. Our MoBGS framework ensures the temporal consistency of unseen latent timestamps and robust motion decomposition of static and dynamic regions. Extensive experiments on the Stereo Blur dataset and real-world blurry videos show that our MoBGS significantly outperforms the very recent advanced methods (DyBluRF and Deblur4DGS), achieving state-of-the-art performance for dynamic NVS under motion blur.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00421v1">Real-Time Animatable 2DGS-Avatars with Detail Enhancement from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      High-quality, animatable 3D human avatar reconstruction from monocular videos offers significant potential for reducing reliance on complex hardware, making it highly practical for applications in game development, augmented reality, and social media. However, existing methods still face substantial challenges in capturing fine geometric details and maintaining animation stability, particularly under dynamic or complex poses. To address these issues, we propose a novel real-time framework for animatable human avatar reconstruction based on 2D Gaussian Splatting (2DGS). By leveraging 2DGS and global SMPL pose parameters, our framework not only aligns positional and rotational discrepancies but also enables robust and natural pose-driven animation of the reconstructed avatars. Furthermore, we introduce a Rotation Compensation Network (RCN) that learns rotation residuals by integrating local geometric features with global pose parameters. This network significantly improves the handling of non-rigid deformations and ensures smooth, artifact-free pose transitions during animation. Experimental results demonstrate that our method successfully reconstructs realistic and highly animatable human avatars from monocular videos, effectively preserving fine-grained details while ensuring stable and natural pose variation. Our approach surpasses current state-of-the-art methods in both reconstruction quality and animation robustness on public benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18768v2">TransparentGS: Fast Inverse Rendering of Transparent Objects with Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 accepted by SIGGRAPH 2025; https://letianhuang.github.io/transparentgs/
    </div>
    <details class="paper-abstract">
      The emergence of neural and Gaussian-based radiance field methods has led to considerable advancements in novel view synthesis and 3D object reconstruction. Nonetheless, specular reflection and refraction continue to pose significant challenges due to the instability and incorrect overfitting of radiance fields to high-frequency light variations. Currently, even 3D Gaussian Splatting (3D-GS), as a powerful and efficient tool, falls short in recovering transparent objects with nearby contents due to the existence of apparent secondary ray effects. To address this issue, we propose TransparentGS, a fast inverse rendering pipeline for transparent objects based on 3D-GS. The main contributions are three-fold. Firstly, an efficient representation of transparent objects, transparent Gaussian primitives, is designed to enable specular refraction through a deferred refraction strategy. Secondly, we leverage Gaussian light field probes (GaussProbe) to encode both ambient light and nearby contents in a unified framework. Thirdly, a depth-based iterative probes query (IterQuery) algorithm is proposed to reduce the parallax errors in our probe-based framework. Experiments demonstrate the speed and accuracy of our approach in recovering transparent objects from complex environments, as well as several applications in computer graphics and vision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20379v2">GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      In this paper, we present a method for localizing a query image with respect to a precomputed 3D Gaussian Splatting (3DGS) scene representation. First, the method uses 3DGS to render a synthetic RGBD image at some initial pose estimate. Second, it establishes 2D-2D correspondences between the query image and this synthetic image. Third, it uses the depth map to lift the 2D-2D correspondences to 2D-3D correspondences and solves a perspective-n-point (PnP) problem to produce a final pose estimate. Results from evaluation across three existing datasets with 38 scenes and over 2,700 test images show that our method significantly reduces both inference time (by over two orders of magnitude, from more than 10 seconds to as fast as 0.1 seconds) and estimation error compared to baseline methods that use photometric loss minimization. Results also show that our method tolerates large errors in the initial pose estimate of up to 55{\deg} in rotation and 1.1 units in translation (normalized by scene scale), achieving final pose errors of less than 5{\deg} in rotation and 0.05 units in translation on 90% of images from the Synthetic NeRF and Mip-NeRF360 datasets and on 42% of images from the more challenging Tanks and Temples dataset.
    </details>
</div>
