# gaussian splatting - 2025_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13167v2">ODHSR: Online Dense 3D Reconstruction of Humans and Scenes from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 Accepted at CVPR 2025
    </div>
    <details class="paper-abstract">
      Creating a photorealistic scene and human reconstruction from a single monocular in-the-wild video figures prominently in the perception of a human-centric 3D world. Recent neural rendering advances have enabled holistic human-scene reconstruction but require pre-calibrated camera and human poses, and days of training time. In this work, we introduce a novel unified framework that simultaneously performs camera tracking, human pose estimation and human-scene reconstruction in an online fashion. 3D Gaussian Splatting is utilized to learn Gaussian primitives for humans and scenes efficiently, and reconstruction-based camera tracking and human pose estimation modules are designed to enable holistic understanding and effective disentanglement of pose and appearance. Specifically, we design a human deformation module to reconstruct the details and enhance generalizability to out-of-distribution poses faithfully. Aiming to learn the spatial correlation between human and scene accurately, we introduce occlusion-aware human silhouette rendering and monocular geometric priors, which further improve reconstruction quality. Experiments on the EMDB and NeuMan datasets demonstrate superior or on-par performance with existing methods in camera tracking, human pose estimation, novel view synthesis and runtime. Our project page is at https://eth-ait.github.io/ODHSR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13713v1">SLAM&Render: A Benchmark for the Intersection Between Neural Rendering, Gaussian Splatting and SLAM</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Models and methods originally developed for novel view synthesis and scene rendering, such as Neural Radiance Fields (NeRF) and Gaussian Splatting, are increasingly being adopted as representations in Simultaneous Localization and Mapping (SLAM). However, existing datasets fail to include the specific challenges of both fields, such as multimodality and sequentiality in SLAM or generalization across viewpoints and illumination conditions in neural rendering. To bridge this gap, we introduce SLAM&Render, a novel dataset designed to benchmark methods in the intersection between SLAM and novel view rendering. It consists of 40 sequences with synchronized RGB, depth, IMU, robot kinematic data, and ground-truth pose streams. By releasing robot kinematic data, the dataset also enables the assessment of novel SLAM strategies when applied to robot manipulators. The dataset sequences span five different setups featuring consumer and industrial objects under four different lighting conditions, with separate training and test trajectories per scene, as well as object rearrangements. Our experimental results, obtained with several baselines from the literature, validate SLAM&Render as a relevant benchmark for this emerging research area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13697v1">Green Robotic Mixed Reality with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 6 pages, 5 figures, accepted by IEEE INFOCOM 2025 Workshop on Networked Robotics and Communication Systems
    </div>
    <details class="paper-abstract">
      Realizing green communication in robotic mixed reality (RoboMR) systems presents a challenge, due to the necessity of uploading high-resolution images at high frequencies through wireless channels. This paper proposes Gaussian splatting (GS) RoboMR (GSRMR), which achieves a lower energy consumption and makes a concrete step towards green RoboMR. The crux to GSRMR is to build a GS model which enables the simulator to opportunistically render a photo-realistic view from the robot's pose, thereby reducing the need for excessive image uploads. Since the GS model may involve discrepancies compared to the actual environments, a GS cross-layer optimization (GSCLO) framework is further proposed, which jointly optimizes content switching (i.e., deciding whether to upload image or not) and power allocation across different frames. The GSCLO problem is solved by an accelerated penalty optimization (APO) algorithm. Experiments demonstrate that the proposed GSRMR reduces the communication energy by over 10x compared with RoboMR. Furthermore, the proposed GSRMR with APO outperforms extensive baseline schemes, in terms of peak signal-to-noise ratio (PSNR) and structural similarity index measure (SSIM).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13540v1">EG-Gaussian: Epipolar Geometry and Graph Network Enhanced 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      In this paper, we explore an open research problem concerning the reconstruction of 3D scenes from images. Recent methods have adopt 3D Gaussian Splatting (3DGS) to produce 3D scenes due to its efficient training process. However, these methodologies may generate incomplete 3D scenes or blurred multiviews. This is because of (1) inaccurate 3DGS point initialization and (2) the tendency of 3DGS to flatten 3D Gaussians with the sparse-view input. To address these issues, we propose a novel framework EG-Gaussian, which utilizes epipolar geometry and graph networks for 3D scene reconstruction. Initially, we integrate epipolar geometry into the 3DGS initialization phase to enhance initial 3DGS point construction. Then, we specifically design a graph learning module to refine 3DGS spatial features, in which we incorporate both spatial coordinates and angular relationships among neighboring points. Experiments on indoor and outdoor benchmark datasets demonstrate that our approach significantly improves reconstruction accuracy compared to 3DGS-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13175v1">Novel Demonstration Generation with Gaussian Splatting Enables Robust One-Shot Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 Published at Robotics: Science and Systems (RSS) 2025
    </div>
    <details class="paper-abstract">
      Visuomotor policies learned from teleoperated demonstrations face challenges such as lengthy data collection, high costs, and limited data diversity. Existing approaches address these issues by augmenting image observations in RGB space or employing Real-to-Sim-to-Real pipelines based on physical simulators. However, the former is constrained to 2D data augmentation, while the latter suffers from imprecise physical simulation caused by inaccurate geometric reconstruction. This paper introduces RoboSplat, a novel method that generates diverse, visually realistic demonstrations by directly manipulating 3D Gaussians. Specifically, we reconstruct the scene through 3D Gaussian Splatting (3DGS), directly edit the reconstructed scene, and augment data across six types of generalization with five techniques: 3D Gaussian replacement for varying object types, scene appearance, and robot embodiments; equivariant transformations for different object poses; visual attribute editing for various lighting conditions; novel view synthesis for new camera perspectives; and 3D content generation for diverse object types. Comprehensive real-world experiments demonstrate that RoboSplat significantly enhances the generalization of visuomotor policies under diverse disturbances. Notably, while policies trained on hundreds of real-world demonstrations with additional 2D data augmentation achieve an average success rate of 57.2%, RoboSplat attains 87.8% in one-shot settings across six types of generalization in the real world.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13167v1">ODHSR: Online Dense 3D Reconstruction of Humans and Scenes from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 Accepted at CVPR 2025
    </div>
    <details class="paper-abstract">
      Creating a photorealistic scene and human reconstruction from a single monocular in-the-wild video figures prominently in the perception of a human-centric 3D world. Recent neural rendering advances have enabled holistic human-scene reconstruction but require pre-calibrated camera and human poses, and days of training time. In this work, we introduce a novel unified framework that simultaneously performs camera tracking, human pose estimation and human-scene reconstruction in an online fashion. 3D Gaussian Splatting is utilized to learn Gaussian primitives for humans and scenes efficiently, and reconstruction-based camera tracking and human pose estimation modules are designed to enable holistic understanding and effective disentanglement of pose and appearance. Specifically, we design a human deformation module to reconstruct the details and enhance generalizability to out-of-distribution poses faithfully. Aiming to learn the spatial correlation between human and scene accurately, we introduce occlusion-aware human silhouette rendering and monocular geometric priors, which further improve reconstruction quality. Experiments on the EMDB and NeuMan datasets demonstrate superior or on-par performance with existing methods in camera tracking, human pose estimation, novel view synthesis and runtime. Our project page is at https://eth-ait.github.io/ODHSR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13159v1">Digital Twin Generation from Visual Data: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      This survey explores recent developments in generating digital twins from videos. Such digital twins can be used for robotics application, media content creation, or design and construction works. We analyze various approaches, including 3D Gaussian Splatting, generative in-painting, semantic segmentation, and foundation models highlighting their advantages and limitations. Additionally, we discuss challenges such as occlusions, lighting variations, and scalability, as well as potential future research directions. This survey aims to provide a comprehensive overview of state-of-the-art methodologies and their implications for real-world applications. Awesome list: https://github.com/ndrwmlnk/awesome-digital-twins
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13153v1">Training-Free Hierarchical Scene Understanding for Gaussian Splatting with Superpoint Graphs</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Bridging natural language and 3D geometry is a crucial step toward flexible, language-driven scene understanding. While recent advances in 3D Gaussian Splatting (3DGS) have enabled fast and high-quality scene reconstruction, research has also explored incorporating open-vocabulary understanding into 3DGS. However, most existing methods require iterative optimization over per-view 2D semantic feature maps, which not only results in inefficiencies but also leads to inconsistent 3D semantics across views. To address these limitations, we introduce a training-free framework that constructs a superpoint graph directly from Gaussian primitives. The superpoint graph partitions the scene into spatially compact and semantically coherent regions, forming view-consistent 3D entities and providing a structured foundation for open-vocabulary understanding. Based on the graph structure, we design an efficient reprojection strategy that lifts 2D semantic features onto the superpoints, avoiding costly multi-view iterative training. The resulting representation ensures strong 3D semantic coherence and naturally supports hierarchical understanding, enabling both coarse- and fine-grained open-vocabulary perception within a unified semantic field. Extensive experiments demonstrate that our method achieves state-of-the-art open-vocabulary segmentation performance, with semantic field reconstruction completed over $30\times$ faster. Our code will be available at https://github.com/Atrovast/THGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13022v1">CompGS++: Compressed Gaussian Splatting for Static and Dynamic Scene Representation</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 Submitted to a journal
    </div>
    <details class="paper-abstract">
      Gaussian splatting demonstrates proficiency for 3D scene modeling but suffers from substantial data volume due to inherent primitive redundancy. To enable future photorealistic 3D immersive visual communication applications, significant compression is essential for transmission over the existing Internet infrastructure. Hence, we propose Compressed Gaussian Splatting (CompGS++), a novel framework that leverages compact Gaussian primitives to achieve accurate 3D modeling with substantial size reduction for both static and dynamic scenes. Our design is based on the principle of eliminating redundancy both between and within primitives. Specifically, we develop a comprehensive prediction paradigm to address inter-primitive redundancy through spatial and temporal primitive prediction modules. The spatial primitive prediction module establishes predictive relationships for scene primitives and enables most primitives to be encoded as compact residuals, substantially reducing the spatial redundancy. We further devise a temporal primitive prediction module to handle dynamic scenes, which exploits primitive correlations across timestamps to effectively reduce temporal redundancy. Moreover, we devise a rate-constrained optimization module that jointly minimizes reconstruction error and rate consumption. This module effectively eliminates parameter redundancy within primitives and enhances the overall compactness of scene representations. Comprehensive evaluations across multiple benchmark datasets demonstrate that CompGS++ significantly outperforms existing methods, achieving superior compression performance while preserving accurate scene modeling. Our implementation will be made publicly available on GitHub to facilitate further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12999v1">GSAC: Leveraging Gaussian Splatting for Photorealistic Avatar Creation with Unity Integration</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Photorealistic avatars have become essential for immersive applications in virtual reality (VR) and augmented reality (AR), enabling lifelike interactions in areas such as training simulations, telemedicine, and virtual collaboration. These avatars bridge the gap between the physical and digital worlds, improving the user experience through realistic human representation. However, existing avatar creation techniques face significant challenges, including high costs, long creation times, and limited utility in virtual applications. Manual methods, such as MetaHuman, require extensive time and expertise, while automatic approaches, such as NeRF-based pipelines often lack efficiency, detailed facial expression fidelity, and are unable to be rendered at a speed sufficent for real-time applications. By involving several cutting-edge modern techniques, we introduce an end-to-end 3D Gaussian Splatting (3DGS) avatar creation pipeline that leverages monocular video input to create a scalable and efficient photorealistic avatar directly compatible with the Unity game engine. Our pipeline incorporates a novel Gaussian splatting technique with customized preprocessing that enables the user of "in the wild" monocular video capture, detailed facial expression reconstruction and embedding within a fully rigged avatar model. Additionally, we present a Unity-integrated Gaussian Splatting Avatar Editor, offering a user-friendly environment for VR/AR application development. Experimental results validate the effectiveness of our preprocessing pipeline in standardizing custom data for 3DGS training and demonstrate the versatility of Gaussian avatars in Unity, highlighting the scalability and practicality of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12905v1">Second-order Optimization of Gaussian Splats with Importance Sampling</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is widely used for novel view synthesis due to its high rendering quality and fast inference time. However, 3DGS predominantly relies on first-order optimizers such as Adam, which leads to long training times. To address this limitation, we propose a novel second-order optimization strategy based on Levenberg-Marquardt (LM) and Conjugate Gradient (CG), which we specifically tailor towards Gaussian Splatting. Our key insight is that the Jacobian in 3DGS exhibits significant sparsity since each Gaussian affects only a limited number of pixels. We exploit this sparsity by proposing a matrix-free and GPU-parallelized LM optimization. To further improve its efficiency, we propose sampling strategies for both the camera views and loss function and, consequently, the normal equation, significantly reducing the computational complexity. In addition, we increase the convergence rate of the second-order approximation by introducing an effective heuristic to determine the learning rate that avoids the expensive computation cost of line search methods. As a result, our method achieves a $3\times$ speedup over standard LM and outperforms Adam by $~6\times$ when the Gaussian count is low while remaining competitive for moderate counts. Project Page: https://vcai.mpi-inf.mpg.de/projects/LM-IS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12811v1">AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Although 3D Gaussian Splatting (3DGS) has revolutionized 3D reconstruction, it still faces challenges such as aliasing, projection artifacts, and view inconsistencies, primarily due to the simplification of treating splats as 2D entities. We argue that incorporating full 3D evaluation of Gaussians throughout the 3DGS pipeline can effectively address these issues while preserving rasterization efficiency. Specifically, we introduce an adaptive 3D smoothing filter to mitigate aliasing and present a stable view-space bounding method that eliminates popping artifacts when Gaussians extend beyond the view frustum. Furthermore, we promote tile-based culling to 3D with screen-space planes, accelerating rendering and reducing sorting costs for hierarchical rasterization. Our method achieves state-of-the-art quality on in-distribution evaluation sets and significantly outperforms other approaches for out-of-distribution views. Our qualitative evaluations further demonstrate the effective removal of aliasing, distortions, and popping artifacts, ensuring real-time, artifact-free rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12800v1">CAGE-GS: High-fidelity Cage Based 3D Gaussian Splatting Deformation</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      As 3D Gaussian Splatting (3DGS) gains popularity as a 3D representation of real scenes, enabling user-friendly deformation to create novel scenes while preserving fine details from the original 3DGS has attracted significant research attention. We introduce CAGE-GS, a cage-based 3DGS deformation method that seamlessly aligns a source 3DGS scene with a user-defined target shape. Our approach learns a deformation cage from the target, which guides the geometric transformation of the source scene. While the cages effectively control structural alignment, preserving the textural appearance of 3DGS remains challenging due to the complexity of covariance parameters. To address this, we employ a Jacobian matrix-based strategy to update the covariance parameters of each Gaussian, ensuring texture fidelity post-deformation. Our method is highly flexible, accommodating various target shape representations, including texts, images, point clouds, meshes and 3DGS models. Extensive experiments and ablation studies on both public datasets and newly proposed scenes demonstrate that our method significantly outperforms existing techniques in both efficiency and deformation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12799v1">TSGS: Improving Gaussian Splatting for Transparent Surface Reconstruction via Normal and De-lighting Priors</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 Project page: https://longxiang-ai.github.io/TSGS/
    </div>
    <details class="paper-abstract">
      Reconstructing transparent surfaces is essential for tasks such as robotic manipulation in labs, yet it poses a significant challenge for 3D reconstruction techniques like 3D Gaussian Splatting (3DGS). These methods often encounter a transparency-depth dilemma, where the pursuit of photorealistic rendering through standard $\alpha$-blending undermines geometric precision, resulting in considerable depth estimation errors for transparent materials. To address this issue, we introduce Transparent Surface Gaussian Splatting (TSGS), a new framework that separates geometry learning from appearance refinement. In the geometry learning stage, TSGS focuses on geometry by using specular-suppressed inputs to accurately represent surfaces. In the second stage, TSGS improves visual fidelity through anisotropic specular modeling, crucially maintaining the established opacity to ensure geometric accuracy. To enhance depth inference, TSGS employs a first-surface depth extraction method. This technique uses a sliding window over $\alpha$-blending weights to pinpoint the most likely surface location and calculates a robust weighted average depth. To evaluate the transparent surface reconstruction task under realistic conditions, we collect a TransLab dataset that includes complex transparent laboratory glassware. Extensive experiments on TransLab show that TSGS achieves accurate geometric reconstruction and realistic rendering of transparent objects simultaneously within the efficient 3DGS framework. Specifically, TSGS significantly surpasses current leading methods, achieving a 37.3% reduction in chamfer distance and an 8.0% improvement in F1 score compared to the top baseline. The code and dataset will be released at https://longxiang-ai.github.io/TSGS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12788v1">ARAP-GS: Drag-driven As-Rigid-As-Possible 3D Gaussian Splatting Editing with Diffusion Prior</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Drag-driven editing has become popular among designers for its ability to modify complex geometric structures through simple and intuitive manipulation, allowing users to adjust and reshape content with minimal technical skill. This drag operation has been incorporated into numerous methods to facilitate the editing of 2D images and 3D meshes in design. However, few studies have explored drag-driven editing for the widely-used 3D Gaussian Splatting (3DGS) representation, as deforming 3DGS while preserving shape coherence and visual continuity remains challenging. In this paper, we introduce ARAP-GS, a drag-driven 3DGS editing framework based on As-Rigid-As-Possible (ARAP) deformation. Unlike previous 3DGS editing methods, we are the first to apply ARAP deformation directly to 3D Gaussians, enabling flexible, drag-driven geometric transformations. To preserve scene appearance after deformation, we incorporate an advanced diffusion prior for image super-resolution within our iterative optimization process. This approach enhances visual quality while maintaining multi-view consistency in the edited results. Experiments show that ARAP-GS outperforms current methods across diverse 3D scenes, demonstrating its effectiveness and superiority for drag-driven 3DGS editing. Additionally, our method is highly efficient, requiring only 10 to 20 minutes to edit a scene on a single RTX 3090 GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10809v2">GaSLight: Gaussian Splats for Spatially-Varying Lighting in HDR</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      We present GaSLight, a method that generates spatially-varying lighting from regular images. Our method proposes using HDR Gaussian Splats as light source representation, marking the first time regular images can serve as light sources in a 3D renderer. Our two-stage process first enhances the dynamic range of images plausibly and accurately by leveraging the priors embedded in diffusion models. Next, we employ Gaussian Splats to model 3D lighting, achieving spatially variant lighting. Our approach yields state-of-the-art results on HDR estimations and their applications in illuminating virtual objects and scenes. To facilitate the benchmarking of images as light sources, we introduce a novel dataset of calibrated and unsaturated HDR to evaluate images as light sources. We assess our method using a combination of this novel dataset and an existing dataset from the literature. Project page: https://lvsn.github.io/gaslight/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13339v1">Volume Encoding Gaussians: Transfer Function-Agnostic 3D Gaussians for Volume Rendering</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      While HPC resources are increasingly being used to produce adaptively refined or unstructured volume datasets, current research in applying machine learning-based representation to visualization has largely ignored this type of data. To address this, we introduce Volume Encoding Gaussians (VEG), a novel 3D Gaussian-based representation for scientific volume visualization focused on unstructured volumes. Unlike prior 3D Gaussian Splatting (3DGS) methods that store view-dependent color and opacity for each Gaussian, VEG decouple the visual appearance from the data representation by encoding only scalar values, enabling transfer-function-agnostic rendering of 3DGS models for interactive scientific visualization. VEG are directly initialized from volume datasets, eliminating the need for structure-from-motion pipelines like COLMAP. To ensure complete scalar field coverage, we introduce an opacity-guided training strategy, using differentiable rendering with multiple transfer functions to optimize our data representation. This allows VEG to preserve fine features across the full scalar range of a dataset while remaining independent of any specific transfer function. Each Gaussian is scaled and rotated to adapt to local geometry, allowing for efficient representation of unstructured meshes without storing mesh connectivity and while using far fewer primitives. Across a diverse set of data, VEG achieve high reconstruction quality, compress large volume datasets by up to 3600x, and support lightning-fast rendering on commodity GPUs, enabling interactive visualization of large-scale structured and unstructured volumes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11520v2">EditSplat: Multi-View Fusion and Attention-Guided Optimization for View-Consistent 3D Scene Editing with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D editing have highlighted the potential of text-driven methods in real-time, user-friendly AR/VR applications. However, current methods rely on 2D diffusion models without adequately considering multi-view information, resulting in multi-view inconsistency. While 3D Gaussian Splatting (3DGS) significantly improves rendering quality and speed, its 3D editing process encounters difficulties with inefficient optimization, as pre-trained Gaussians retain excessive source information, hindering optimization. To address these limitations, we propose EditSplat, a novel text-driven 3D scene editing framework that integrates Multi-view Fusion Guidance (MFG) and Attention-Guided Trimming (AGT). Our MFG ensures multi-view consistency by incorporating essential multi-view information into the diffusion process, leveraging classifier-free guidance from the text-to-image diffusion model and the geometric structure inherent to 3DGS. Additionally, our AGT utilizes the explicit representation of 3DGS to selectively prune and optimize 3D Gaussians, enhancing optimization efficiency and enabling precise, semantically rich local editing. Through extensive qualitative and quantitative evaluations, EditSplat achieves state-of-the-art performance, establishing a new benchmark for text-driven 3D scene editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04196v2">GST: Precise 3D Human Body from a Single Image with Gaussian Splatting Transformers</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 Camera ready for CVSports workshop at CVPR 2025
    </div>
    <details class="paper-abstract">
      Reconstructing posed 3D human models from monocular images has important applications in the sports industry, including performance tracking, injury prevention and virtual training. In this work, we combine 3D human pose and shape estimation with 3D Gaussian Splatting (3DGS), a representation of the scene composed of a mixture of Gaussians. This allows training or fine-tuning a human model predictor on multi-view images alone, without 3D ground truth. Predicting such mixtures for a human from a single input image is challenging due to self-occlusions and dependence on articulations, while also needing to retain enough flexibility to accommodate a variety of clothes and poses. Our key observation is that the vertices of standardized human meshes (such as SMPL) can provide an adequate spatial density and approximate initial position for the Gaussians. We can then train a transformer model to jointly predict comparatively small adjustments to these positions, as well as the other 3DGS attributes and the SMPL parameters. We show empirically that this combination (using only multi-view supervision) can achieve near real-time inference of 3D human models from a single image without expensive diffusion models or 3D points supervision, thus making it ideal for the sport industry at any level. More importantly, rendering is an effective auxiliary objective to refine 3D pose estimation by accounting for clothes and other geometric variations. The code is available at https://github.com/prosperolo/GST.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04857v2">3D Gaussian Particle Approximation of VDB Datasets: A Study for Scientific Visualization</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      The complexity and scale of Volumetric and Simulation datasets for Scientific Visualization(SciVis) continue to grow. And the approaches and advantages of memory-efficient data formats and storage techniques for such datasets vary. OpenVDB library and its VDB data format excels in memory efficiency through its hierarchical and dynamic tree structure, with active and inactive sub-trees for data storage. It is heavily used in current production renderers for both animation and rendering stages in VFX pipelines and photorealistic rendering of volumes and fluids. However, it still remains to be fully leveraged in SciVis where domains dealing with sparse scalar fields like porous media, time varying volumes such as tornado and weather simulation or high resolution simulation of Computational Fluid Dynamics present ample number of large challenging data sets. The goal of this paper hence is not only to explore the use of OpenVDB in SciVis but also to explore a level of detail(LOD) technique using 3D Gaussian particles approximating voxel regions. For rendering, we utilize NVIDIA OptiX library for ray marching through the Gaussians particles. Data modeling using 3D Gaussians has been very popular lately due to success in stereoscopic image to 3D scene conversion using Gaussian Splatting and Gaussian approximation and mixture models aren't entirely new in SciVis as well. Our work explores the integration with rendering software libraries like OpenVDB and OptiX to take advantage of their built-in memory compaction and hardware acceleration features, while also leveraging the performance capabilities of modern GPUs. Thus, we present a SciVis rendering approach that uses 3D Gaussians at varying LOD in a lossy scheme derived from VDB datasets, rather than focusing on photorealistic volume rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16443v3">SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 Project Page: https://gohyojun15.github.io/SplatFlow/
    </div>
    <details class="paper-abstract">
      Text-based generation and editing of 3D scenes hold significant potential for streamlining content creation through intuitive user interactions. While recent advances leverage 3D Gaussian Splatting (3DGS) for high-fidelity and real-time rendering, existing methods are often specialized and task-focused, lacking a unified framework for both generation and editing. In this paper, we introduce SplatFlow, a comprehensive framework that addresses this gap by enabling direct 3DGS generation and editing. SplatFlow comprises two main components: a multi-view rectified flow (RF) model and a Gaussian Splatting Decoder (GSDecoder). The multi-view RF model operates in latent space, generating multi-view images, depths, and camera poses simultaneously, conditioned on text prompts, thus addressing challenges like diverse scene scales and complex camera trajectories in real-world settings. Then, the GSDecoder efficiently translates these latent outputs into 3DGS representations through a feed-forward 3DGS method. Leveraging training-free inversion and inpainting techniques, SplatFlow enables seamless 3DGS editing and supports a broad range of 3D tasks-including object editing, novel view synthesis, and camera pose estimation-within a unified framework without requiring additional complex pipelines. We validate SplatFlow's capabilities on the MVImgNet and DL3DV-7K datasets, demonstrating its versatility and effectiveness in various 3D generation, editing, and inpainting-based tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11893v1">CAGS: Open-Vocabulary 3D Scene Understanding with Context-Aware Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Open-vocabulary 3D scene understanding is crucial for applications requiring natural language-driven spatial interpretation, such as robotics and augmented reality. While 3D Gaussian Splatting (3DGS) offers a powerful representation for scene reconstruction, integrating it with open-vocabulary frameworks reveals a key challenge: cross-view granularity inconsistency. This issue, stemming from 2D segmentation methods like SAM, results in inconsistent object segmentations across views (e.g., a "coffee set" segmented as a single entity in one view but as "cup + coffee + spoon" in another). Existing 3DGS-based methods often rely on isolated per-Gaussian feature learning, neglecting the spatial context needed for cohesive object reasoning, leading to fragmented representations. We propose Context-Aware Gaussian Splatting (CAGS), a novel framework that addresses this challenge by incorporating spatial context into 3DGS. CAGS constructs local graphs to propagate contextual features across Gaussians, reducing noise from inconsistent granularity, employs mask-centric contrastive learning to smooth SAM-derived features across views, and leverages a precomputation strategy to reduce computational cost by precomputing neighborhood relationships, enabling efficient training in large-scale scenes. By integrating spatial context, CAGS significantly improves 3D instance segmentation and reduces fragmentation errors on datasets like LERF-OVS and ScanNet, enabling robust language-guided 3D scene understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10001v3">GaussVideoDreamer: 3D Scene Generation with Video Diffusion and Inconsistency-Aware Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Single-image 3D scene reconstruction presents significant challenges due to its inherently ill-posed nature and limited input constraints. Recent advances have explored two promising directions: multiview generative models that train on 3D consistent datasets but struggle with out-of-distribution generalization, and 3D scene inpainting and completion frameworks that suffer from cross-view inconsistency and suboptimal error handling, as they depend exclusively on depth data or 3D smoothness, which ultimately degrades output quality and computational performance. Building upon these approaches, we present GaussVideoDreamer, which advances generative multimedia approaches by bridging the gap between image, video, and 3D generation, integrating their strengths through two key innovations: (1) A progressive video inpainting strategy that harnesses temporal coherence for improved multiview consistency and faster convergence. (2) A 3D Gaussian Splatting consistency mask to guide the video diffusion with 3D consistent multiview evidence. Our pipeline combines three core components: a geometry-aware initialization protocol, Inconsistency-Aware Gaussian Splatting, and a progressive video inpainting strategy. Experimental results demonstrate that our approach achieves 32% higher LLaVA-IQA scores and at least 2x speedup compared to existing methods while maintaining robust performance across diverse scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11218v2">3DAffordSplat: Efficient Affordance Reasoning with 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 The first large-scale 3D Gaussians Affordance Reasoning Benchmark
    </div>
    <details class="paper-abstract">
      3D affordance reasoning is essential in associating human instructions with the functional regions of 3D objects, facilitating precise, task-oriented manipulations in embodied AI. However, current methods, which predominantly depend on sparse 3D point clouds, exhibit limited generalizability and robustness due to their sensitivity to coordinate variations and the inherent sparsity of the data. By contrast, 3D Gaussian Splatting (3DGS) delivers high-fidelity, real-time rendering with minimal computational overhead by representing scenes as dense, continuous distributions. This positions 3DGS as a highly effective approach for capturing fine-grained affordance details and improving recognition accuracy. Nevertheless, its full potential remains largely untapped due to the absence of large-scale, 3DGS-specific affordance datasets. To overcome these limitations, we present 3DAffordSplat, the first large-scale, multi-modal dataset tailored for 3DGS-based affordance reasoning. This dataset includes 23,677 Gaussian instances, 8,354 point cloud instances, and 6,631 manually annotated affordance labels, encompassing 21 object categories and 18 affordance types. Building upon this dataset, we introduce AffordSplatNet, a novel model specifically designed for affordance reasoning using 3DGS representations. AffordSplatNet features an innovative cross-modal structure alignment module that exploits structural consistency priors to align 3D point cloud and 3DGS representations, resulting in enhanced affordance recognition accuracy. Extensive experiments demonstrate that the 3DAffordSplat dataset significantly advances affordance learning within the 3DGS domain, while AffordSplatNet consistently outperforms existing methods across both seen and unseen settings, highlighting its robust generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13207v1">BEV-GS: Feed-forward Gaussian Splatting in Bird's-Eye-View for Road Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Road surface is the sole contact medium for wheels or robot feet. Reconstructing road surface is crucial for unmanned vehicles and mobile robots. Recent studies on Neural Radiance Fields (NeRF) and Gaussian Splatting (GS) have achieved remarkable results in scene reconstruction. However, they typically rely on multi-view image inputs and require prolonged optimization times. In this paper, we propose BEV-GS, a real-time single-frame road surface reconstruction method based on feed-forward Gaussian splatting. BEV-GS consists of a prediction module and a rendering module. The prediction module introduces separate geometry and texture networks following Bird's-Eye-View paradigm. Geometric and texture parameters are directly estimated from a single frame, avoiding per-scene optimization. In the rendering module, we utilize grid Gaussian for road surface representation and novel view synthesis, which better aligns with road surface characteristics. Our method achieves state-of-the-art performance on the real-world dataset RSRD. The road elevation error reduces to 1.73 cm, and the PSNR of novel view synthesis reaches 28.36 dB. The prediction and rendering FPS is 26, and 2061, respectively, enabling high-accuracy and real-time applications. The code will be available at: \href{https://github.com/cat-wwh/BEV-GS}{\texttt{https://github.com/cat-wwh/BEV-GS}}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10331v2">LL-Gaussian: Low-Light Scene Reconstruction and Enhancement via Gaussian Splatting for Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) in low-light scenes remains a significant challenge due to degraded inputs characterized by severe noise, low dynamic range (LDR) and unreliable initialization. While recent NeRF-based approaches have shown promising results, most suffer from high computational costs, and some rely on carefully captured or pre-processed data--such as RAW sensor inputs or multi-exposure sequences--which severely limits their practicality. In contrast, 3D Gaussian Splatting (3DGS) enables real-time rendering with competitive visual fidelity; however, existing 3DGS-based methods struggle with low-light sRGB inputs, resulting in unstable Gaussian initialization and ineffective noise suppression. To address these challenges, we propose LL-Gaussian, a novel framework for 3D reconstruction and enhancement from low-light sRGB images, enabling pseudo normal-light novel view synthesis. Our method introduces three key innovations: 1) an end-to-end Low-Light Gaussian Initialization Module (LLGIM) that leverages dense priors from learning-based MVS approach to generate high-quality initial point clouds; 2) a dual-branch Gaussian decomposition model that disentangles intrinsic scene properties (reflectance and illumination) from transient interference, enabling stable and interpretable optimization; 3) an unsupervised optimization strategy guided by both physical constrains and diffusion prior to jointly steer decomposition and enhancement. Additionally, we contribute a challenging dataset collected in extreme low-light environments and demonstrate the effectiveness of LL-Gaussian. Compared to state-of-the-art NeRF-based methods, LL-Gaussian achieves up to 2,000 times faster inference and reduces training time to just 2%, while delivering superior reconstruction and rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11218v1">3DAffordSplat: Efficient Affordance Reasoning with 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 The first large-scale 3D Gaussians Affordance Reasoning Benchmark
    </div>
    <details class="paper-abstract">
      3D affordance reasoning is essential in associating human instructions with the functional regions of 3D objects, facilitating precise, task-oriented manipulations in embodied AI. However, current methods, which predominantly depend on sparse 3D point clouds, exhibit limited generalizability and robustness due to their sensitivity to coordinate variations and the inherent sparsity of the data. By contrast, 3D Gaussian Splatting (3DGS) delivers high-fidelity, real-time rendering with minimal computational overhead by representing scenes as dense, continuous distributions. This positions 3DGS as a highly effective approach for capturing fine-grained affordance details and improving recognition accuracy. Nevertheless, its full potential remains largely untapped due to the absence of large-scale, 3DGS-specific affordance datasets. To overcome these limitations, we present 3DAffordSplat, the first large-scale, multi-modal dataset tailored for 3DGS-based affordance reasoning. This dataset includes 23,677 Gaussian instances, 8,354 point cloud instances, and 6,631 manually annotated affordance labels, encompassing 21 object categories and 18 affordance types. Building upon this dataset, we introduce AffordSplatNet, a novel model specifically designed for affordance reasoning using 3DGS representations. AffordSplatNet features an innovative cross-modal structure alignment module that exploits structural consistency priors to align 3D point cloud and 3DGS representations, resulting in enhanced affordance recognition accuracy. Extensive experiments demonstrate that the 3DAffordSplat dataset significantly advances affordance learning within the 3DGS domain, while AffordSplatNet consistently outperforms existing methods across both seen and unseen settings, highlighting its robust generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19235v2">InstanceGaussian: Appearance-Semantic Joint Gaussian Representation for 3D Instance-Level Perception</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 14 pages, accepted by CVPR 2025 as poster
    </div>
    <details class="paper-abstract">
      3D scene understanding has become an essential area of research with applications in autonomous driving, robotics, and augmented reality. Recently, 3D Gaussian Splatting (3DGS) has emerged as a powerful approach, combining explicit modeling with neural adaptability to provide efficient and detailed scene representations. However, three major challenges remain in leveraging 3DGS for scene understanding: 1) an imbalance between appearance and semantics, where dense Gaussian usage for fine-grained texture modeling does not align with the minimal requirements for semantic attributes; 2) inconsistencies between appearance and semantics, as purely appearance-based Gaussians often misrepresent object boundaries; and 3) reliance on top-down instance segmentation methods, which struggle with uneven category distributions, leading to over- or under-segmentation. In this work, we propose InstanceGaussian, a method that jointly learns appearance and semantic features while adaptively aggregating instances. Our contributions include: i) a novel Semantic-Scaffold-GS representation balancing appearance and semantics to improve feature representations and boundary delineation; ii) a progressive appearance-semantic joint training strategy to enhance stability and segmentation accuracy; and iii) a bottom-up, category-agnostic instance aggregation approach that addresses segmentation challenges through farthest point sampling and connected component analysis. Our approach achieves state-of-the-art performance in category-agnostic, open-vocabulary 3D point-level segmentation, highlighting the effectiveness of the proposed representation and training strategies. Project page: https://lhj-git.github.io/InstanceGaussian/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17932v2">VR-Splatting: Foveated Radiance Field Rendering via 3D Gaussian Splatting and Neural Points</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      Recent advances in novel view synthesis have demonstrated impressive results in fast photorealistic scene rendering through differentiable point rendering, either via Gaussian Splatting (3DGS) [Kerbl and Kopanas et al. 2023] or neural point rendering [Aliev et al. 2020]. Unfortunately, these directions require either a large number of small Gaussians or expensive per-pixel post-processing for reconstructing fine details, which negatively impacts rendering performance. To meet the high performance demands of virtual reality (VR) systems, primitive or pixel counts therefore must be kept low, affecting visual quality. In this paper, we propose a novel hybrid approach based on foveated rendering as a promising solution that combines the strengths of both point rendering directions regarding performance sweet spots. Analyzing the compatibility with the human visual system, we find that using a low-detailed, few primitive smooth Gaussian representation for the periphery is cheap to compute and meets the perceptual demands of peripheral vision. For the fovea only, we use neural points with a convolutional neural network for the small pixel footprint, which provides sharp, detailed output within the rendering budget. This combination also allows for synergistic method accelerations with point occlusion culling and reducing the demands on the neural network. Our evaluation confirms that our approach increases sharpness and details compared to a standard VR-ready 3DGS configuration, and participants of a user study overwhelmingly preferred our method. Our system meets the necessary performance requirements for real-time VR interactions, ultimately enhancing the user's immersive experience. The project page can be found at: https://lfranke.github.io/vr_splatting
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11024v1">Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      The increasing availability of digital 3D environments, whether through image-based 3D reconstruction, generation, or scans obtained by robots, is driving innovation across various applications. These come with a significant demand for 3D interaction, such as 3D Interactive Segmentation, which is useful for tasks like object selection and manipulation. Additionally, there is a persistent need for solutions that are efficient, precise, and performing well across diverse settings, particularly in unseen environments and with unfamiliar objects. In this work, we introduce a 3D interactive segmentation method that consistently surpasses previous state-of-the-art techniques on both in-domain and out-of-domain datasets. Our simple approach integrates a voxel-based sparse encoder with a lightweight transformer-based decoder that implements implicit click fusion, achieving superior performance and maximizing efficiency. Our method demonstrates substantial improvements on benchmark datasets, including ScanNet, ScanNet++, S3DIS, and KITTI-360, and also on unseen geometric distributions such as the ones obtained by Gaussian Splatting. The project web-page is available at https://simonelli-andrea.github.io/easy3d.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11003v1">3D Gabor Splatting: Reconstruction of High-frequency Surface Texture using Gabor Noise</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 4 pages, 5 figures, Eurographics 2025 Short Paper
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting has experienced explosive popularity in the past few years in the field of novel view synthesis. The lightweight and differentiable representation of the radiance field using the Gaussian enables rapid and high-quality reconstruction and fast rendering. However, reconstructing objects with high-frequency surface textures (e.g., fine stripes) requires many skinny Gaussian kernels because each Gaussian represents only one color if viewed from one direction. Thus, reconstructing the stripes pattern, for example, requires Gaussians for at least the number of stripes. We present 3D Gabor splatting, which augments the Gaussian kernel to represent spatially high-frequency signals using Gabor noise. The Gabor kernel is a combination of a Gaussian term and spatially fluctuating wave functions, making it suitable for representing spatial high-frequency texture. We demonstrate that our 3D Gabor splatting can reconstruct various high-frequency textures on the objects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09048v2">BlockGaussian: Efficient Large-Scale Scene Novel View Synthesis via Adaptive Block-Based Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 https://github.com/SunshineWYC/BlockGaussian
    </div>
    <details class="paper-abstract">
      The recent advancements in 3D Gaussian Splatting (3DGS) have demonstrated remarkable potential in novel view synthesis tasks. The divide-and-conquer paradigm has enabled large-scale scene reconstruction, but significant challenges remain in scene partitioning, optimization, and merging processes. This paper introduces BlockGaussian, a novel framework incorporating a content-aware scene partition strategy and visibility-aware block optimization to achieve efficient and high-quality large-scale scene reconstruction. Specifically, our approach considers the content-complexity variation across different regions and balances computational load during scene partitioning, enabling efficient scene reconstruction. To tackle the supervision mismatch issue during independent block optimization, we introduce auxiliary points during individual block optimization to align the ground-truth supervision, which enhances the reconstruction quality. Furthermore, we propose a pseudo-view geometry constraint that effectively mitigates rendering degradation caused by airspace floaters during block merging. Extensive experiments on large-scale scenes demonstrate that our approach achieves state-of-the-art performance in both reconstruction efficiency and rendering quality, with a 5x speedup in optimization and an average PSNR improvement of 1.21 dB on multiple benchmarks. Notably, BlockGaussian significantly reduces computational requirements, enabling large-scale scene reconstruction on a single 24GB VRAM device. The project page is available at https://github.com/SunshineWYC/BlockGaussian
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05511v3">Free Your Hands: Lightweight Relightable Turntable Capture Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) from multiple captured photos of an object is a widely studied problem. Achieving high quality typically requires dense sampling of input views, which can lead to frustrating and tedious manual labor. Manually positioning cameras to maintain an optimal desired distribution can be difficult for humans, and if a good distribution is found, it is not easy to replicate. Additionally, the captured data can suffer from motion blur and defocus due to human error. In this paper, we present a lightweight object capture pipeline to reduce the manual workload and standardize the acquisition setup. We use a consumer turntable to carry the object and a tripod to hold the camera. As the turntable rotates, we automatically capture dense samples from various views and lighting conditions; we can repeat this for several camera positions. This way, we can easily capture hundreds of valid images in several minutes without hands-on effort. However, in the object reference frame, the light conditions vary; this is harmful to a standard NVS method like 3D Gaussian splatting (3DGS) which assumes fixed lighting. We design a neural radiance representation conditioned on light rotations, which addresses this issue and allows relightability as an additional benefit. We demonstrate our pipeline using 3DGS as the underlying framework, achieving competitive quality compared to previous methods with exhaustive acquisition and showcasing its potential for relighting and harmonization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10001v2">GaussVideoDreamer: 3D Scene Generation with Video Diffusion and Inconsistency-Aware Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      Single-image 3D scene reconstruction presents significant challenges due to its inherently ill-posed nature and limited input constraints. Recent advances have explored two promising directions: multiview generative models that train on 3D consistent datasets but struggle with out-of-distribution generalization, and 3D scene inpainting and completion frameworks that suffer from cross-view inconsistency and suboptimal error handling, as they depend exclusively on depth data or 3D smoothness, which ultimately degrades output quality and computational performance. Building upon these approaches, we present GaussVideoDreamer, which advances generative multimedia approaches by bridging the gap between image, video, and 3D generation, integrating their strengths through two key innovations: (1) A progressive video inpainting strategy that harnesses temporal coherence for improved multiview consistency and faster convergence. (2) A 3D Gaussian Splatting consistency mask to guide the video diffusion with 3D consistent multiview evidence. Our pipeline combines three core components: a geometry-aware initialization protocol, Inconsistency-Aware Gaussian Splatting, and a progressive video inpainting strategy. Experimental results demonstrate that our approach achieves 32% higher LLaVA-IQA scores and at least 2x speedup compared to existing methods while maintaining robust performance across diverse scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10809v1">GaSLight: Gaussian Splats for Spatially-Varying Lighting in HDR</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      We present GaSLight, a method that generates spatially-varying lighting from regular images. Our method proposes using HDR Gaussian Splats as light source representation, marking the first time regular images can serve as light sources in a 3D renderer. Our two-stage process first enhances the dynamic range of images plausibly and accurately by leveraging the priors embedded in diffusion models. Next, we employ Gaussian Splats to model 3D lighting, achieving spatially variant lighting. Our approach yields state-of-the-art results on HDR estimations and their applications in illuminating virtual objects and scenes. To facilitate the benchmarking of images as light sources, we introduce a novel dataset of calibrated and unsaturated HDR to evaluate images as light sources. We assess our method using a combination of this novel dataset and an existing dataset from the literature. The code to reproduce our method will be available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13204v1">EDGS: Eliminating Densification for Efficient Convergence of 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting reconstructs scenes by starting from a sparse Structure-from-Motion initialization and iteratively refining under-reconstructed regions. This process is inherently slow, as it requires multiple densification steps where Gaussians are repeatedly split and adjusted, following a lengthy optimization path. Moreover, this incremental approach often leads to suboptimal renderings, particularly in high-frequency regions where detail is critical. We propose a fundamentally different approach: we eliminate densification process with a one-step approximation of scene geometry using triangulated pixels from dense image correspondences. This dense initialization allows us to estimate rough geometry of the scene while preserving rich details from input RGB images, providing each Gaussian with well-informed colors, scales, and positions. As a result, we dramatically shorten the optimization path and remove the need for densification. Unlike traditional methods that rely on sparse keypoints, our dense initialization ensures uniform detail across the scene, even in high-frequency regions where 3DGS and other methods struggle. Moreover, since all splats are initialized in parallel at the start of optimization, we eliminate the need to wait for densification to adjust new Gaussians. Our method not only outperforms speed-optimized models in training efficiency but also achieves higher rendering quality than state-of-the-art approaches, all while using only half the splats of standard 3DGS. It is fully compatible with other 3DGS acceleration techniques, making it a versatile and efficient solution that can be integrated with existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10486v1">DNF-Avatar: Distilling Neural Fields for Real-time Animatable Avatar Relighting</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 16 pages, 8 figures, Project pages: https://jzr99.github.io/DNF-Avatar/
    </div>
    <details class="paper-abstract">
      Creating relightable and animatable human avatars from monocular videos is a rising research topic with a range of applications, e.g. virtual reality, sports, and video games. Previous works utilize neural fields together with physically based rendering (PBR), to estimate geometry and disentangle appearance properties of human avatars. However, one drawback of these methods is the slow rendering speed due to the expensive Monte Carlo ray tracing. To tackle this problem, we proposed to distill the knowledge from implicit neural fields (teacher) to explicit 2D Gaussian splatting (student) representation to take advantage of the fast rasterization property of Gaussian splatting. To avoid ray-tracing, we employ the split-sum approximation for PBR appearance. We also propose novel part-wise ambient occlusion probes for shadow computation. Shadow prediction is achieved by querying these probes only once per pixel, which paves the way for real-time relighting of avatars. These techniques combined give high-quality relighting results with realistic shadow effects. Our experiments demonstrate that the proposed student model achieves comparable or even better relighting results with our teacher model while being 370 times faster at inference time, achieving a 67 FPS rendering speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.15856v3">SplatMesh: Interactive 3D Segmentation and Editing Using Mesh-Based Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      A key challenge in fine-grained 3D-based interactive editing is the absence of an efficient representation that balances diverse modifications with high-quality view synthesis under a given memory constraint. While 3D meshes provide robustness for various modifications, they often yield lower-quality view synthesis compared to 3D Gaussian Splatting, which, in turn, suffers from instability during extensive editing. A straightforward combination of these two representations results in suboptimal performance and fails to meet memory constraints. In this paper, we introduce SplatMesh, a novel fine-grained interactive 3D segmentation and editing algorithm that integrates 3D Gaussian Splat with a precomputed mesh and could adjust the memory request based on the requirement. Specifically, given a mesh, \method simplifies it while considering both color and shape, ensuring it meets memory constraints. Then, SplatMesh aligns Gaussian splats with the simplified mesh by treating each triangle as a new reference point. By segmenting and editing the simplified mesh, we can effectively edit the Gaussian splats as well, which will lead to extensive experiments on real and synthetic datasets, coupled with illustrative visual examples, highlighting the superiority of our approach in terms of representation quality and editing performance. Code of our paper can be found here: https://github.com/kaichen-z/SplatMesh.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10209v2">GAF: Gaussian Avatar Reconstruction from Monocular Videos via Multi-view Diffusion</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 Paper Video: https://youtu.be/QuIYTljvhyg Project Page: https://tangjiapeng.github.io/projects/GAF
    </div>
    <details class="paper-abstract">
      We propose a novel approach for reconstructing animatable 3D Gaussian avatars from monocular videos captured by commodity devices like smartphones. Photorealistic 3D head avatar reconstruction from such recordings is challenging due to limited observations, which leaves unobserved regions under-constrained and can lead to artifacts in novel views. To address this problem, we introduce a multi-view head diffusion model, leveraging its priors to fill in missing regions and ensure view consistency in Gaussian splatting renderings. To enable precise viewpoint control, we use normal maps rendered from FLAME-based head reconstruction, which provides pixel-aligned inductive biases. We also condition the diffusion model on VAE features extracted from the input image to preserve facial identity and appearance details. For Gaussian avatar reconstruction, we distill multi-view diffusion priors by using iteratively denoised images as pseudo-ground truths, effectively mitigating over-saturation issues. To further improve photorealism, we apply latent upsampling priors to refine the denoised latent before decoding it into an image. We evaluate our method on the NeRSemble dataset, showing that GAF outperforms previous state-of-the-art methods in novel view synthesis. Furthermore, we demonstrate higher-fidelity avatar reconstructions from monocular videos captured on commodity devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10331v1">LL-Gaussian: Low-Light Scene Reconstruction and Enhancement via Gaussian Splatting for Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) in low-light scenes remains a significant challenge due to degraded inputs characterized by severe noise, low dynamic range (LDR) and unreliable initialization. While recent NeRF-based approaches have shown promising results, most suffer from high computational costs, and some rely on carefully captured or pre-processed data--such as RAW sensor inputs or multi-exposure sequences--which severely limits their practicality. In contrast, 3D Gaussian Splatting (3DGS) enables real-time rendering with competitive visual fidelity; however, existing 3DGS-based methods struggle with low-light sRGB inputs, resulting in unstable Gaussian initialization and ineffective noise suppression. To address these challenges, we propose LL-Gaussian, a novel framework for 3D reconstruction and enhancement from low-light sRGB images, enabling pseudo normal-light novel view synthesis. Our method introduces three key innovations: 1) an end-to-end Low-Light Gaussian Initialization Module (LLGIM) that leverages dense priors from learning-based MVS approach to generate high-quality initial point clouds; 2) a dual-branch Gaussian decomposition model that disentangles intrinsic scene properties (reflectance and illumination) from transient interference, enabling stable and interpretable optimization; 3) an unsupervised optimization strategy guided by both physical constrains and diffusion prior to jointly steer decomposition and enhancement. Additionally, we contribute a challenging dataset collected in extreme low-light environments and demonstrate the effectiveness of LL-Gaussian. Compared to state-of-the-art NeRF-based methods, LL-Gaussian achieves up to 2,000 times faster inference and reduces training time to just 2%, while delivering superior reconstruction and rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17636v3">HOMER: Homography-Based Efficient Multi-view 3D Object Removal</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      3D object removal is an important sub-task in 3D scene editing, with broad applications in scene understanding, augmented reality, and robotics. However, existing methods struggle to achieve a desirable balance among consistency, usability, and computational efficiency in multi-view settings. These limitations are primarily due to unintuitive user interaction in the source view, inefficient multi-view object mask generation, computationally expensive inpainting procedures, and a lack of applicability across different radiance field representations. To address these challenges, we propose a novel pipeline that improves the quality and efficiency of multi-view object mask generation and inpainting. Our method introduces an intuitive region-based interaction mechanism in the source view and eliminates the need for camera poses or extra model training. Our lightweight HoMM module is employed to achieve high-quality multi-view mask propagation with enhanced efficiency. In the inpainting stage, we further reduce computational costs by performing inpainting only on selected key views and propagating the results to other views via homography-based mapping. Our pipeline is compatible with a variety of radiance field frameworks, including NeRF and 3D Gaussian Splatting, demonstrating improved generalizability and practicality in real-world scenarios. Additionally, we present a new 3D multi-object removal dataset with greater object diversity and viewpoint variation than existing datasets. Experiments on public benchmarks and our proposed dataset show that our method achieves state-of-the-art performance while reducing runtime to one-fifth of that required by leading baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10316v1">ESCT3D: Efficient and Selectively Controllable Text-Driven 3D Content Generation with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      In recent years, significant advancements have been made in text-driven 3D content generation. However, several challenges remain. In practical applications, users often provide extremely simple text inputs while expecting high-quality 3D content. Generating optimal results from such minimal text is a difficult task due to the strong dependency of text-to-3D models on the quality of input prompts. Moreover, the generation process exhibits high variability, making it difficult to control. Consequently, multiple iterations are typically required to produce content that meets user expectations, reducing generation efficiency. To address this issue, we propose GPT-4V for self-optimization, which significantly enhances the efficiency of generating satisfactory content in a single attempt. Furthermore, the controllability of text-to-3D generation methods has not been fully explored. Our approach enables users to not only provide textual descriptions but also specify additional conditions, such as style, edges, scribbles, poses, or combinations of multiple conditions, allowing for more precise control over the generated 3D content. Additionally, during training, we effectively integrate multi-view information, including multi-view depth, masks, features, and images, to address the common Janus problem in 3D content generation. Extensive experiments demonstrate that our method achieves robust generalization, facilitating the efficient and controllable generation of high-quality 3D content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13183v2">Real-time Free-view Human Rendering from Sparse-view RGB Videos using Double Unprojected Textures</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 Accepted at CVPR 2025, Project page: https://vcai.mpi-inf.mpg.de/projects/DUT/
    </div>
    <details class="paper-abstract">
      Real-time free-view human rendering from sparse-view RGB inputs is a challenging task due to the sensor scarcity and the tight time budget. To ensure efficiency, recent methods leverage 2D CNNs operating in texture space to learn rendering primitives. However, they either jointly learn geometry and appearance, or completely ignore sparse image information for geometry estimation, significantly harming visual quality and robustness to unseen body poses. To address these issues, we present Double Unprojected Textures, which at the core disentangles coarse geometric deformation estimation from appearance synthesis, enabling robust and photorealistic 4K rendering in real-time. Specifically, we first introduce a novel image-conditioned template deformation network, which estimates the coarse deformation of the human template from a first unprojected texture. This updated geometry is then used to apply a second and more accurate texture unprojection. The resulting texture map has fewer artifacts and better alignment with input views, which benefits our learning of finer-level geometry and appearance represented by Gaussian splats. We validate the effectiveness and efficiency of the proposed method in quantitative and qualitative experiments, which significantly surpasses other state-of-the-art methods. Project page: https://vcai.mpi-inf.mpg.de/projects/DUT/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10012v1">EBAD-Gaussian: Event-driven Bundle Adjusted Deblur Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3D-GS) achieves photorealistic novel view synthesis, its performance degrades with motion blur. In scenarios with rapid motion or low-light conditions, existing RGB-based deblurring methods struggle to model camera pose and radiance changes during exposure, reducing reconstruction accuracy. Event cameras, capturing continuous brightness changes during exposure, can effectively assist in modeling motion blur and improving reconstruction quality. Therefore, we propose Event-driven Bundle Adjusted Deblur Gaussian Splatting (EBAD-Gaussian), which reconstructs sharp 3D Gaussians from event streams and severely blurred images. This method jointly learns the parameters of these Gaussians while recovering camera motion trajectories during exposure time. Specifically, we first construct a blur loss function by synthesizing multiple latent sharp images during the exposure time, minimizing the difference between real and synthesized blurred images. Then we use event stream to supervise the light intensity changes between latent sharp images at any time within the exposure period, supplementing the light intensity dynamic changes lost in RGB images. Furthermore, we optimize the latent sharp images at intermediate exposure times based on the event-based double integral (EDI) prior, applying consistency constraints to enhance the details and texture information of the reconstructed images. Extensive experiments on synthetic and real-world datasets show that EBAD-Gaussian can achieve high-quality 3D scene reconstruction under the condition of blurred images and event stream inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10001v1">GaussVideoDreamer: 3D Scene Generation with Video Diffusion and Inconsistency-Aware Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Single-image 3D scene reconstruction presents significant challenges due to its inherently ill-posed nature and limited input constraints. Recent advances have explored two promising directions: multiview generative models that train on 3D consistent datasets but struggle with out-of-distribution generalization, and 3D scene inpainting and completion frameworks that suffer from cross-view inconsistency and suboptimal error handling, as they depend exclusively on depth data or 3D smoothness, which ultimately degrades output quality and computational performance. Building upon these approaches, we present GaussVideoDreamer, which advances generative multimedia approaches by bridging the gap between image, video, and 3D generation, integrating their strengths through two key innovations: (1) A progressive video inpainting strategy that harnesses temporal coherence for improved multiview consistency and faster convergence. (2) A 3D Gaussian Splatting consistency mask to guide the video diffusion with 3D consistent multiview evidence. Our pipeline combines three core components: a geometry-aware initialization protocol, Inconsistency-Aware Gaussian Splatting, and a progressive video inpainting strategy. Experimental results demonstrate that our approach achieves 32% higher LLaVA-IQA scores and at least 2x speedup compared to existing methods while maintaining robust performance across diverse scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09878v1">MCBlock: Boosting Neural Radiance Field Training Speed by MCTS-based Dynamic-Resolution Ray Sampling</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Neural Radiance Field (NeRF) is widely known for high-fidelity novel view synthesis. However, even the state-of-the-art NeRF model, Gaussian Splatting, requires minutes for training, far from the real-time performance required by multimedia scenarios like telemedicine. One of the obstacles is its inefficient sampling, which is only partially addressed by existing works. Existing point-sampling algorithms uniformly sample simple-texture regions (easy to fit) and complex-texture regions (hard to fit), while existing ray-sampling algorithms sample these regions all in the finest granularity (i.e. the pixel level), both wasting GPU training resources. Actually, regions with different texture intensities require different sampling granularities. To this end, we propose a novel dynamic-resolution ray-sampling algorithm, MCBlock, which employs Monte Carlo Tree Search (MCTS) to partition each training image into pixel blocks with different sizes for active block-wise training. Specifically, the trees are initialized according to the texture of training images to boost the initialization speed, and an expansion/pruning module dynamically optimizes the block partition. MCBlock is implemented in Nerfstudio, an open-source toolset, and achieves a training acceleration of up to 2.33x, surpassing other ray-sampling algorithms. We believe MCBlock can apply to any cone-tracing NeRF model and contribute to the multimedia community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00905v2">Ref-GS: Directional Factorization for 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-13
      | 💬 CVPR 2025. Project page: https://ref-gs.github.io/
    </div>
    <details class="paper-abstract">
      In this paper, we introduce Ref-GS, a novel approach for directional light factorization in 2D Gaussian splatting, which enables photorealistic view-dependent appearance rendering and precise geometry recovery. Ref-GS builds upon the deferred rendering of Gaussian splatting and applies directional encoding to the deferred-rendered surface, effectively reducing the ambiguity between orientation and viewing angle. Next, we introduce a spherical Mip-grid to capture varying levels of surface roughness, enabling roughness-aware Gaussian shading. Additionally, we propose a simple yet efficient geometry-lighting factorization that connects geometry and lighting via the vector outer product, significantly reducing renderer overhead when integrating volumetric attributes. Our method achieves superior photorealistic rendering for a range of open-world scenes while also accurately recovering geometry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09588v1">TextSplat: Text-Guided Semantic Fusion for Generalizable Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Recent advancements in Generalizable Gaussian Splatting have enabled robust 3D reconstruction from sparse input views by utilizing feed-forward Gaussian Splatting models, achieving superior cross-scene generalization. However, while many methods focus on geometric consistency, they often neglect the potential of text-driven guidance to enhance semantic understanding, which is crucial for accurately reconstructing fine-grained details in complex scenes. To address this limitation, we propose TextSplat--the first text-driven Generalizable Gaussian Splatting framework. By employing a text-guided fusion of diverse semantic cues, our framework learns robust cross-modal feature representations that improve the alignment of geometric and semantic information, producing high-fidelity 3D reconstructions. Specifically, our framework employs three parallel modules to obtain complementary representations: the Diffusion Prior Depth Estimator for accurate depth information, the Semantic Aware Segmentation Network for detailed semantic information, and the Multi-View Interaction Network for refined cross-view features. Then, in the Text-Guided Semantic Fusion Module, these representations are integrated via the text-guided and attention-based feature aggregation mechanism, resulting in enhanced 3D Gaussian parameters enriched with detailed semantic cues. Experimental results on various benchmark datasets demonstrate improved performance compared to existing methods across multiple evaluation metrics, validating the effectiveness of our framework. The code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09491v1">DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering</a></div>
    <div class="paper-meta">
      📅 2025-04-13
      | 💬 Accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      Although 3D Gaussian Splatting (3DGS) has demonstrated promising results in novel view synthesis, its performance degrades dramatically with sparse inputs and generates undesirable artifacts. As the number of training views decreases, the novel view synthesis task degrades to a highly under-determined problem such that existing methods suffer from the notorious overfitting issue. Interestingly, we observe that models with fewer Gaussian primitives exhibit less overfitting under sparse inputs. Inspired by this observation, we propose a Random Dropout Regularization (RDR) to exploit the advantages of low-complexity models to alleviate overfitting. In addition, to remedy the lack of high-frequency details for these models, an Edge-guided Splitting Strategy (ESS) is developed. With these two techniques, our method (termed DropoutGS) provides a simple yet effective plug-in approach to improve the generalization performance of existing 3DGS methods. Extensive experiments show that our DropoutGS produces state-of-the-art performance under sparse views on benchmark datasets including Blender, LLFF, and DTU. The project page is at: https://xuyx55.github.io/DropoutGS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18394v3">Reconstructing Satellites in 3D from Amateur Telescope Images</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Monitoring space objects is crucial for space situational awareness, yet reconstructing 3D satellite models from ground-based telescope images is challenging due to atmospheric turbulence, long observation distances, limited viewpoints, and low signal-to-noise ratios. In this paper, we propose a novel computational imaging framework that overcomes these obstacles by integrating a hybrid image pre-processing pipeline with a joint pose estimation and 3D reconstruction module based on controlled Gaussian Splatting (GS) and Branch-and-Bound (BnB) search. We validate our approach on both synthetic satellite datasets and on-sky observations of China's Tiangong Space Station and the International Space Station, achieving robust 3D reconstructions of low-Earth orbit satellites from ground-based data. Quantitative evaluations using SSIM, PSNR, LPIPS, and Chamfer Distance demonstrate that our method outperforms state-of-the-art NeRF-based approaches, and ablation studies confirm the critical role of each component. Our framework enables high-fidelity 3D satellite monitoring from Earth, offering a cost-effective alternative for space situational awareness. Project page: https://ai4scientificimaging.org/ReconstructingSatellites
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09129v1">A Constrained Optimization Approach for Gaussian Splatting from Coarsely-posed Images and Noisy Lidar Point Clouds</a></div>
    <div class="paper-meta">
      📅 2025-04-12
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a powerful reconstruction technique, but it needs to be initialized from accurate camera poses and high-fidelity point clouds. Typically, the initialization is taken from Structure-from-Motion (SfM) algorithms; however, SfM is time-consuming and restricts the application of 3DGS in real-world scenarios and large-scale scene reconstruction. We introduce a constrained optimization method for simultaneous camera pose estimation and 3D reconstruction that does not require SfM support. Core to our approach is decomposing a camera pose into a sequence of camera-to-(device-)center and (device-)center-to-world optimizations. To facilitate, we propose two optimization constraints conditioned to the sensitivity of each parameter group and restricts each parameter's search space. In addition, as we learn the scene geometry directly from the noisy point clouds, we propose geometric constraints to improve the reconstruction quality. Experiments demonstrate that the proposed method significantly outperforms the existing (multi-modal) 3DGS baseline and methods supplemented by COLMAP on both our collected dataset and two public benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09097v1">BIGS: Bimanual Category-agnostic Interaction Reconstruction from Monocular Videos via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-12
      | 💬 Accepted to CVPR 2025
    </div>
    <details class="paper-abstract">
      Reconstructing 3Ds of hand-object interaction (HOI) is a fundamental problem that can find numerous applications. Despite recent advances, there is no comprehensive pipeline yet for bimanual class-agnostic interaction reconstruction from a monocular RGB video, where two hands and an unknown object are interacting with each other. Previous works tackled the limited hand-object interaction case, where object templates are pre-known or only one hand is involved in the interaction. The bimanual interaction reconstruction exhibits severe occlusions introduced by complex interactions between two hands and an object. To solve this, we first introduce BIGS (Bimanual Interaction 3D Gaussian Splatting), a method that reconstructs 3D Gaussians of hands and an unknown object from a monocular video. To robustly obtain object Gaussians avoiding severe occlusions, we leverage prior knowledge of pre-trained diffusion model with score distillation sampling (SDS) loss, to reconstruct unseen object parts. For hand Gaussians, we exploit the 3D priors of hand model (i.e., MANO) and share a single Gaussian for two hands to effectively accumulate hand 3D information, given limited views. To further consider the 3D alignment between hands and objects, we include the interacting-subjects optimization step during Gaussian optimization. Our method achieves the state-of-the-art accuracy on two challenging datasets, in terms of 3D hand pose estimation (MPJPE), 3D object reconstruction (CDh, CDo, F10), and rendering quality (PSNR, SSIM, LPIPS), respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09062v1">You Need a Transition Plane: Bridging Continuous Panoramic 3D Reconstruction with Perspective Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-12
    </div>
    <details class="paper-abstract">
      Recently, reconstructing scenes from a single panoramic image using advanced 3D Gaussian Splatting (3DGS) techniques has attracted growing interest. Panoramic images offer a 360$\times$ 180 field of view (FoV), capturing the entire scene in a single shot. However, panoramic images introduce severe distortion, making it challenging to render 3D Gaussians into 2D distorted equirectangular space directly. Converting equirectangular images to cubemap projections partially alleviates this problem but introduces new challenges, such as projection distortion and discontinuities across cube-face boundaries. To address these limitations, we present a novel framework, named TPGS, to bridge continuous panoramic 3D scene reconstruction with perspective Gaussian splatting. Firstly, we introduce a Transition Plane between adjacent cube faces to enable smoother transitions in splatting directions and mitigate optimization ambiguity in the boundary region. Moreover, an intra-to-inter face optimization strategy is proposed to enhance local details and restore visual consistency across cube-face boundaries. Specifically, we optimize 3D Gaussians within individual cube faces and then fine-tune them in the stitched panoramic space. Additionally, we introduce a spherical sampling technique to eliminate visible stitching seams. Extensive experiments on indoor and outdoor, egocentric, and roaming benchmark datasets demonstrate that our approach outperforms existing state-of-the-art methods. Code and models will be available at https://github.com/zhijieshen-bjtu/TPGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12323v2">Depth Estimation Based on 3D Gaussian Splatting Siamese Defocus</a></div>
    <div class="paper-meta">
      📅 2025-04-12
    </div>
    <details class="paper-abstract">
      Depth estimation is a fundamental task in 3D geometry. While stereo depth estimation can be achieved through triangulation methods, it is not as straightforward for monocular methods, which require the integration of global and local information. The Depth from Defocus (DFD) method utilizes camera lens models and parameters to recover depth information from blurred images and has been proven to perform well. However, these methods rely on All-In-Focus (AIF) images for depth estimation, which is nearly impossible to obtain in real-world applications. To address this issue, we propose a self-supervised framework based on 3D Gaussian splatting and Siamese networks. By learning the blur levels at different focal distances of the same scene in the focal stack, the framework predicts the defocus map and Circle of Confusion (CoC) from a single defocused image, using the defocus map as input to DepthNet for monocular depth estimation. The 3D Gaussian splatting model renders defocused images using the predicted CoC, and the differences between these and the real defocused images provide additional supervision signals for the Siamese Defocus self-supervised network. This framework has been validated on both artificially synthesized and real blurred datasets. Subsequent quantitative and visualization experiments demonstrate that our proposed framework is highly effective as a DFD method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09048v1">BlockGaussian: Efficient Large-Scale Scene NovelView Synthesis via Adaptive Block-Based Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-12
      | 💬 https://github.com/SunshineWYC/BlockGaussian
    </div>
    <details class="paper-abstract">
      The recent advancements in 3D Gaussian Splatting (3DGS) have demonstrated remarkable potential in novel view synthesis tasks. The divide-and-conquer paradigm has enabled large-scale scene reconstruction, but significant challenges remain in scene partitioning, optimization, and merging processes. This paper introduces BlockGaussian, a novel framework incorporating a content-aware scene partition strategy and visibility-aware block optimization to achieve efficient and high-quality large-scale scene reconstruction. Specifically, our approach considers the content-complexity variation across different regions and balances computational load during scene partitioning, enabling efficient scene reconstruction. To tackle the supervision mismatch issue during independent block optimization, we introduce auxiliary points during individual block optimization to align the ground-truth supervision, which enhances the reconstruction quality. Furthermore, we propose a pseudo-view geometry constraint that effectively mitigates rendering degradation caused by airspace floaters during block merging. Extensive experiments on large-scale scenes demonstrate that our approach achieves state-of-the-art performance in both reconstruction efficiency and rendering quality, with a 5x speedup in optimization and an average PSNR improvement of 1.21 dB on multiple benchmarks. Notably, BlockGaussian significantly reduces computational requirements, enabling large-scale scene reconstruction on a single 24GB VRAM device. The project page is available at https://github.com/SunshineWYC/BlockGaussian
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08581v1">FMLGS: Fast Multilevel Language Embedded Gaussians for Part-level Interactive Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      The semantically interactive radiance field has long been a promising backbone for 3D real-world applications, such as embodied AI to achieve scene understanding and manipulation. However, multi-granularity interaction remains a challenging task due to the ambiguity of language and degraded quality when it comes to queries upon object components. In this work, we present FMLGS, an approach that supports part-level open-vocabulary query within 3D Gaussian Splatting (3DGS). We propose an efficient pipeline for building and querying consistent object- and part-level semantics based on Segment Anything Model 2 (SAM2). We designed a semantic deviation strategy to solve the problem of language ambiguity among object parts, which interpolates the semantic features of fine-grained targets for enriched information. Once trained, we can query both objects and their describable parts using natural language. Comparisons with other state-of-the-art methods prove that our method can not only better locate specified part-level targets, but also achieve first-place performance concerning both speed and accuracy, where FMLGS is 98 x faster than LERF, 4 x faster than LangSplat and 2.5 x faster than LEGaussians. Meanwhile, we further integrate FMLGS as a virtual agent that can interactively navigate through 3D scenes, locate targets, and respond to user demands through a chat interface, which demonstrates the potential of our work to be further expanded and applied in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16938v2">Generative Object Insertion in Gaussian Splatting with a Multi-View Diffusion Model</a></div>
    <div class="paper-meta">
      📅 2025-04-11
      | 💬 Accepted by Visual Informatics. Project Page: https://github.com/JiuTongBro/MultiView_Inpaint
    </div>
    <details class="paper-abstract">
      Generating and inserting new objects into 3D content is a compelling approach for achieving versatile scene recreation. Existing methods, which rely on SDS optimization or single-view inpainting, often struggle to produce high-quality results. To address this, we propose a novel method for object insertion in 3D content represented by Gaussian Splatting. Our approach introduces a multi-view diffusion model, dubbed MVInpainter, which is built upon a pre-trained stable video diffusion model to facilitate view-consistent object inpainting. Within MVInpainter, we incorporate a ControlNet-based conditional injection module to enable controlled and more predictable multi-view generation. After generating the multi-view inpainted results, we further propose a mask-aware 3D reconstruction technique to refine Gaussian Splatting reconstruction from these sparse inpainted views. By leveraging these fabricate techniques, our approach yields diverse results, ensures view-consistent and harmonious insertions, and produces better object quality. Extensive experiments demonstrate that our approach outperforms existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08473v1">Cut-and-Splat: Leveraging Gaussian Splatting for Synthetic Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-11
      | 💬 Accepted at the International Conference on Robotics, Computer Vision and Intelligent Systems 2025 (ROBOVIS)
    </div>
    <details class="paper-abstract">
      Generating synthetic images is a useful method for cheaply obtaining labeled data for training computer vision models. However, obtaining accurate 3D models of relevant objects is necessary, and the resulting images often have a gap in realism due to challenges in simulating lighting effects and camera artifacts. We propose using the novel view synthesis method called Gaussian Splatting to address these challenges. We have developed a synthetic data pipeline for generating high-quality context-aware instance segmentation training data for specific objects. This process is fully automated, requiring only a video of the target object. We train a Gaussian Splatting model of the target object and automatically extract the object from the video. Leveraging Gaussian Splatting, we then render the object on a random background image, and monocular depth estimation is employed to place the object in a believable pose. We introduce a novel dataset to validate our approach and show superior performance over other data generation approaches, such as Cut-and-Paste and Diffusion model-based generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08366v1">In-2-4D: Inbetweening from Two Single-View Images to 4D Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-11
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      We propose a new problem, In-2-4D, for generative 4D (i.e., 3D + motion) inbetweening from a minimalistic input setting: two single-view images capturing an object in two distinct motion states. Given two images representing the start and end states of an object in motion, our goal is to generate and reconstruct the motion in 4D. We utilize a video interpolation model to predict the motion, but large frame-to-frame motions can lead to ambiguous interpretations. To overcome this, we employ a hierarchical approach to identify keyframes that are visually close to the input states and show significant motion, then generate smooth fragments between them. For each fragment, we construct the 3D representation of the keyframe using Gaussian Splatting. The temporal frames within the fragment guide the motion, enabling their transformation into dynamic Gaussians through a deformation field. To improve temporal consistency and refine 3D motion, we expand the self-attention of multi-view diffusion across timesteps and apply rigid transformation regularization. Finally, we merge the independently generated 3D motion segments by interpolating boundary deformation fields and optimizing them to align with the guiding video, ensuring smooth and flicker-free transitions. Through extensive qualitative and quantitiave experiments as well as a user study, we show the effectiveness of our method and its components. The project page is available at https://in-2-4d.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10148v4">3D Student Splatting and Scooping</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) provides a new framework for novel view synthesis, and has spiked a new wave of research in neural rendering and related applications. As 3DGS is becoming a foundational component of many models, any improvement on 3DGS itself can bring huge benefits. To this end, we aim to improve the fundamental paradigm and formulation of 3DGS. We argue that as an unnormalized mixture model, it needs to be neither Gaussians nor splatting. We subsequently propose a new mixture model consisting of flexible Student's t distributions, with both positive (splatting) and negative (scooping) densities. We name our model Student Splatting and Scooping, or SSS. When providing better expressivity, SSS also poses new challenges in learning. Therefore, we also propose a new principled sampling approach for optimization. Through exhaustive evaluation and comparison, across multiple datasets, settings, and metrics, we demonstrate that SSS outperforms existing methods in terms of quality and parameter efficiency, e.g. achieving matching or better quality with similar numbers of components, and obtaining comparable results while reducing the component number by as much as 82%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16995v2">E-3DGS: Gaussian Splatting with Exposure and Motion Events</a></div>
    <div class="paper-meta">
      📅 2025-04-11
      | 💬 Accepted to Applied Optics (AO). The source code and dataset will be available at https://github.com/MasterHow/E-3DGS
    </div>
    <details class="paper-abstract">
      Achieving 3D reconstruction from images captured under optimal conditions has been extensively studied in the vision and imaging fields. However, in real-world scenarios, challenges such as motion blur and insufficient illumination often limit the performance of standard frame-based cameras in delivering high-quality images. To address these limitations, we incorporate a transmittance adjustment device at the hardware level, enabling event cameras to capture both motion and exposure events for diverse 3D reconstruction scenarios. Motion events (triggered by camera or object movement) are collected in fast-motion scenarios when the device is inactive, while exposure events (generated through controlled camera exposure) are captured during slower motion to reconstruct grayscale images for high-quality training and optimization of event-based 3D Gaussian Splatting (3DGS). Our framework supports three modes: High-Quality Reconstruction using exposure events, Fast Reconstruction relying on motion events, and Balanced Hybrid optimizing with initial exposure events followed by high-speed motion events. On the EventNeRF dataset, we demonstrate that exposure events significantly improve fine detail reconstruction compared to motion events and outperform frame-based cameras under challenging conditions such as low illumination and overexposure. Furthermore, we introduce EME-3D, a real-world 3D dataset with exposure events, motion events, camera calibration parameters, and sparse point clouds. Our method achieves faster and higher-quality reconstruction than event-based NeRF and is more cost-effective than methods combining event and RGB data. E-3DGS sets a new benchmark for event-based 3D reconstruction with robust performance in challenging conditions and lower hardware demands. The source code and dataset will be available at https://github.com/MasterHow/E-3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07015v3">SplatMAP: Online Dense Monocular SLAM with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Achieving high-fidelity 3D reconstruction from monocular video remains challenging due to the inherent limitations of traditional methods like Structure-from-Motion (SfM) and monocular SLAM in accurately capturing scene details. While differentiable rendering techniques such as Neural Radiance Fields (NeRF) address some of these challenges, their high computational costs make them unsuitable for real-time applications. Additionally, existing 3D Gaussian Splatting (3DGS) methods often focus on photometric consistency, neglecting geometric accuracy and failing to exploit SLAM's dynamic depth and pose updates for scene refinement. We propose a framework integrating dense SLAM with 3DGS for real-time, high-fidelity dense reconstruction. Our approach introduces SLAM-Informed Adaptive Densification, which dynamically updates and densifies the Gaussian model by leveraging dense point clouds from SLAM. Additionally, we incorporate Geometry-Guided Optimization, which combines edge-aware geometric constraints and photometric consistency to jointly optimize the appearance and geometry of the 3DGS scene representation, enabling detailed and accurate SLAM mapping reconstruction. Experiments on the Replica and TUM-RGBD datasets demonstrate the effectiveness of our approach, achieving state-of-the-art results among monocular systems. Specifically, our method achieves a PSNR of 36.864, SSIM of 0.985, and LPIPS of 0.040 on Replica, representing improvements of 10.7%, 6.4%, and 49.4%, respectively, over the previous SOTA. On TUM-RGBD, our method outperforms the closest baseline by 10.2%, 6.6%, and 34.7% in the same metrics. These results highlight the potential of our framework in bridging the gap between photometric and geometric dense 3D scene representations, paving the way for practical and efficient monocular dense reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07949v1">InteractAvatar: Modeling Hand-Face Interaction in Photorealistic Avatars with Deformable Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      With the rising interest from the community in digital avatars coupled with the importance of expressions and gestures in communication, modeling natural avatar behavior remains an important challenge across many industries such as teleconferencing, gaming, and AR/VR. Human hands are the primary tool for interacting with the environment and essential for realistic human behavior modeling, yet existing 3D hand and head avatar models often overlook the crucial aspect of hand-body interactions, such as between hand and face. We present InteracttAvatar, the first model to faithfully capture the photorealistic appearance of dynamic hand and non-rigid hand-face interactions. Our novel Dynamic Gaussian Hand model, combining template model and 3D Gaussian Splatting as well as a dynamic refinement module, captures pose-dependent change, e.g. the fine wrinkles and complex shadows that occur during articulation. Importantly, our hand-face interaction module models the subtle geometry and appearance dynamics that underlie common gestures. Through experiments of novel view synthesis, self reenactment and cross-identity reenactment, we demonstrate that InteracttAvatar can reconstruct hand and hand-face interactions from monocular or multiview videos with high-fidelity details and be animated with novel poses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06598v2">Stochastic Ray Tracing of 3D Transparent Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-04-10
      | 💬 10 pages, 6 figures, 5 tables
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting has recently been widely adopted as a 3D representation for novel-view synthesis, relighting, and text-to-3D generation tasks, offering realistic and detailed results through a collection of explicit 3D Gaussians carrying opacities and view-dependent colors. However, efficient rendering of many transparent primitives remains a significant challenge. Existing approaches either rasterize the 3D Gaussians with approximate sorting per view or rely on high-end RTX GPUs to exhaustively process all ray-Gaussian intersections (bounding Gaussians by meshes). This paper proposes a stochastic ray tracing method to render 3D clouds of transparent primitives. Instead of processing all ray-Gaussian intersections in sequential order, each ray traverses the acceleration structure only once, randomly accepting and shading a single intersection (or N intersections, using a simple extension). This approach minimizes shading time and avoids sorting the Gaussians along the ray while minimizing the register usage and maximizing parallelism even on low-end GPUs. The cost of rays through the Gaussian asset is comparable to that of standard mesh-intersection rays. While our method introduces noise, the shading is unbiased, and the variance is slight, as stochastic acceptance is importance-sampled based on accumulated opacity. The alignment with the Monte Carlo philosophy simplifies implementation and easily integrates our method into a conventional path-tracing framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15491v3">GSDeformer: Direct, Real-time and Extensible Cage-based Deformation for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-10
      | 💬 Project Page: https://jhuangbu.github.io/gsdeformer, Video: https://www.youtube.com/watch?v=-ecrj48-MqM
    </div>
    <details class="paper-abstract">
      We present GSDeformer, a method that enables cage-based deformation on 3D Gaussian Splatting (3DGS). Our approach bridges cage-based deformation and 3DGS by using a proxy point-cloud representation. This point cloud is generated from 3D Gaussians, and deformations applied to the point cloud are translated into transformations on the 3D Gaussians. To handle potential bending caused by deformation, we incorporate a splitting process to approximate it. Our method does not modify or extend the core architecture of 3D Gaussian Splatting, making it compatible with any trained vanilla 3DGS or its variants. Additionally, we automate cage construction for 3DGS and its variants using a render-and-reconstruct approach. Experiments demonstrate that GSDeformer delivers superior deformation results compared to existing methods, is robust under extreme deformations, requires no retraining for editing, runs in real-time, and can be extended to other 3DGS variants. Project Page: https://jhuangbu.github.io/gsdeformer/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07370v1">View-Dependent Uncertainty Estimation of 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become increasingly popular in 3D scene reconstruction for its high visual accuracy. However, uncertainty estimation of 3DGS scenes remains underexplored and is crucial to downstream tasks such as asset extraction and scene completion. Since the appearance of 3D gaussians is view-dependent, the color of a gaussian can thus be certain from an angle and uncertain from another. We thus propose to model uncertainty in 3DGS as an additional view-dependent per-gaussian feature that can be modeled with spherical harmonics. This simple yet effective modeling is easily interpretable and can be integrated into the traditional 3DGS pipeline. It is also significantly faster than ensemble methods while maintaining high accuracy, as demonstrated in our experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06019v3">GaussianSpa: An "Optimizing-Sparsifying" Simplification Framework for Compact and High-Quality 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-10
      | 💬 CVPR 2025. Project page at https://noodle-lab.github.io/gaussianspa/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a mainstream for novel view synthesis, leveraging continuous aggregations of Gaussian functions to model scene geometry. However, 3DGS suffers from substantial memory requirements to store the multitude of Gaussians, hindering its practicality. To address this challenge, we introduce GaussianSpa, an optimization-based simplification framework for compact and high-quality 3DGS. Specifically, we formulate the simplification as an optimization problem associated with the 3DGS training. Correspondingly, we propose an efficient "optimizing-sparsifying" solution that alternately solves two independent sub-problems, gradually imposing strong sparsity onto the Gaussians in the training process. Our comprehensive evaluations on various datasets show the superiority of GaussianSpa over existing state-of-the-art approaches. Notably, GaussianSpa achieves an average PSNR improvement of 0.9 dB on the real-world Deep Blending dataset with 10$\times$ fewer Gaussians compared to the vanilla 3DGS. Our project page is available at https://noodle-lab.github.io/gaussianspa/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08100v1">ContrastiveGaussian: High-Fidelity 3D Generation with Contrastive Learning and Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-10
      | 💬 Code will be available at https://github.com/YaNLlan-ljb/ContrastiveGaussian
    </div>
    <details class="paper-abstract">
      Creating 3D content from single-view images is a challenging problem that has attracted considerable attention in recent years. Current approaches typically utilize score distillation sampling (SDS) from pre-trained 2D diffusion models to generate multi-view 3D representations. Although some methods have made notable progress by balancing generation speed and model quality, their performance is often limited by the visual inconsistencies of the diffusion model outputs. In this work, we propose ContrastiveGaussian, which integrates contrastive learning into the generative process. By using a perceptual loss, we effectively differentiate between positive and negative samples, leveraging the visual inconsistencies to improve 3D generation quality. To further enhance sample differentiation and improve contrastive learning, we incorporate a super-resolution model and introduce another Quantity-Aware Triplet Loss to address varying sample distributions during training. Our experiments demonstrate that our approach achieves superior texture fidelity and improved geometric consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16681v2">GauRast: Enhancing GPU Triangle Rasterizers to Accelerate 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-10
      | 💬 DAC 2025
    </div>
    <details class="paper-abstract">
      3D intelligence leverages rich 3D features and stands as a promising frontier in AI, with 3D rendering fundamental to many downstream applications. 3D Gaussian Splatting (3DGS), an emerging high-quality 3D rendering method, requires significant computation, making real-time execution on existing GPU-equipped edge devices infeasible. Previous efforts to accelerate 3DGS rely on dedicated accelerators that require substantial integration overhead and hardware costs. This work proposes an acceleration strategy that leverages the similarities between the 3DGS pipeline and the highly optimized conventional graphics pipeline in modern GPUs. Instead of developing a dedicated accelerator, we enhance existing GPU rasterizer hardware to efficiently support 3DGS operations. Our results demonstrate a 23$\times$ increase in processing speed and a 24$\times$ reduction in energy consumption, with improvements yielding 6$\times$ faster end-to-end runtime for the original 3DGS algorithm and 4$\times$ for the latest efficiency-improved pipeline, achieving 24 FPS and 46 FPS respectively. These enhancements incur only a minimal area overhead of 0.2\% relative to the entire SoC chip area, underscoring the practicality and efficiency of our approach for enabling 3DGS rendering on resource-constrained platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06978v1">Wheat3DGS: In-field 3D Reconstruction, Instance Segmentation and Phenotyping of Wheat Heads with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 Copyright 2025 IEEE. This is the author's version of the work. It is posted here for your personal use. Not for redistribution. The definitive version is published in the 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)
    </div>
    <details class="paper-abstract">
      Automated extraction of plant morphological traits is crucial for supporting crop breeding and agricultural management through high-throughput field phenotyping (HTFP). Solutions based on multi-view RGB images are attractive due to their scalability and affordability, enabling volumetric measurements that 2D approaches cannot directly capture. While advanced methods like Neural Radiance Fields (NeRFs) have shown promise, their application has been limited to counting or extracting traits from only a few plants or organs. Furthermore, accurately measuring complex structures like individual wheat heads-essential for studying crop yields-remains particularly challenging due to occlusions and the dense arrangement of crop canopies in field conditions. The recent development of 3D Gaussian Splatting (3DGS) offers a promising alternative for HTFP due to its high-quality reconstructions and explicit point-based representation. In this paper, we present Wheat3DGS, a novel approach that leverages 3DGS and the Segment Anything Model (SAM) for precise 3D instance segmentation and morphological measurement of hundreds of wheat heads automatically, representing the first application of 3DGS to HTFP. We validate the accuracy of wheat head extraction against high-resolution laser scan data, obtaining per-instance mean absolute percentage errors of 15.1%, 18.3%, and 40.2% for length, width, and volume. We provide additional comparisons to NeRF-based approaches and traditional Muti-View Stereo (MVS), demonstrating superior results. Our approach enables rapid, non-destructive measurements of key yield-related traits at scale, with significant implications for accelerating crop breeding and improving our understanding of wheat development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01732v2">FIORD: A Fisheye Indoor-Outdoor Dataset with LIDAR Ground Truth for 3D Scene Reconstruction and Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 SCIA 2025
    </div>
    <details class="paper-abstract">
      The development of large-scale 3D scene reconstruction and novel view synthesis methods mostly rely on datasets comprising perspective images with narrow fields of view (FoV). While effective for small-scale scenes, these datasets require large image sets and extensive structure-from-motion (SfM) processing, limiting scalability. To address this, we introduce a fisheye image dataset tailored for scene reconstruction tasks. Using dual 200-degree fisheye lenses, our dataset provides full 360-degree coverage of 5 indoor and 5 outdoor scenes. Each scene has sparse SfM point clouds and precise LIDAR-derived dense point clouds that can be used as geometric ground-truth, enabling robust benchmarking under challenging conditions such as occlusions and reflections. While the baseline experiments focus on vanilla Gaussian Splatting and NeRF based Nerfacto methods, the dataset supports diverse approaches for scene reconstruction, novel view synthesis, and image-based rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06827v1">IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      This work presents IAAO, a novel framework that builds an explicit 3D model for intelligent agents to gain understanding of articulated objects in their environment through interaction. Unlike prior methods that rely on task-specific networks and assumptions about movable parts, our IAAO leverages large foundation models to estimate interactive affordances and part articulations in three stages. We first build hierarchical features and label fields for each object state using 3D Gaussian Splatting (3DGS) by distilling mask features and view-consistent labels from multi-view images. We then perform object- and part-level queries on the 3D Gaussian primitives to identify static and articulated elements, estimating global transformations and local articulation parameters along with affordances. Finally, scenes from different states are merged and refined based on the estimated transformations, enabling robust affordance-based interaction and manipulation of objects. Experimental results demonstrate the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06815v1">SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Reconstructing 3D assets from images, known as inverse rendering (IR), remains a challenging task due to its ill-posed nature. 3D Gaussian Splatting (3DGS) has demonstrated impressive capabilities for novel view synthesis (NVS) tasks. Methods apply it to relighting by separating radiance into BRDF parameters and lighting, yet produce inferior relighting quality with artifacts and unnatural indirect illumination due to the limited capability of each Gaussian, which has constant material parameters and normal, alongside the absence of physical constraints for indirect lighting. In this paper, we present a novel framework called Spatially-vayring Gaussian Inverse Rendering (SVG-IR), aimed at enhancing both NVS and relighting quality. To this end, we propose a new representation-Spatially-varying Gaussian (SVG)-that allows per-Gaussian spatially varying parameters. This enhanced representation is complemented by a SVG splatting scheme akin to vertex/fragment shading in traditional graphics pipelines. Furthermore, we integrate a physically-based indirect lighting model, enabling more realistic relighting. The proposed SVG-IR framework significantly improves rendering quality, outperforming state-of-the-art NeRF-based methods by 2.5 dB in peak signal-to-noise ratio (PSNR) and surpassing existing Gaussian-based techniques by 3.5 dB in relighting tasks, all while maintaining a real-time rendering speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06716v1">GSta: Efficient Training Scheme with Siestaed Gaussians for Monocular 3D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 9 pages. In submission to an IEEE conference
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) is a popular approach for 3D reconstruction, mostly due to its ability to converge reasonably fast, faithfully represent the scene and render (novel) views in a fast fashion. However, it suffers from large storage and memory requirements, and its training speed still lags behind the hash-grid based radiance field approaches (e.g. Instant-NGP), which makes it especially difficult to deploy them in robotics scenarios, where 3D reconstruction is crucial for accurate operation. In this paper, we propose GSta that dynamically identifies Gaussians that have converged well during training, based on their positional and color gradient norms. By forcing such Gaussians into a siesta and stopping their updates (freezing) during training, we improve training speed with competitive accuracy compared to state of the art. We also propose an early stopping mechanism based on the PSNR values computed on a subset of training images. Combined with other improvements, such as integrating a learning rate scheduler, GSta achieves an improved Pareto front in convergence speed, memory and storage requirements, while preserving quality. We also show that GSta can improve other methods and complement orthogonal approaches in efficiency improvement; once combined with Trick-GS, GSta achieves up to 5x faster training, 16x smaller disk size compared to vanilla GS, while having comparable accuracy and consuming only half the peak memory. More visualisations are available at https://anilarmagan.github.io/SRUK-GSta.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06651v1">Collision avoidance from monocular vision trained with novel view synthesis</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Collision avoidance can be checked in explicit environment models such as elevation maps or occupancy grids, yet integrating such models with a locomotion policy requires accurate state estimation. In this work, we consider the question of collision avoidance from an implicit environment model. We use monocular RGB images as inputs and train a collisionavoidance policy from photorealistic images generated by 2D Gaussian splatting. We evaluate the resulting pipeline in realworld experiments under velocity commands that bring the robot on an intercept course with obstacles. Our results suggest that RGB images can be enough to make collision-avoidance decisions, both in the room where training data was collected and in out-of-distribution environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16323v4">LeanGaussian: Breaking Pixel or Point Cloud Correspondence in Modeling 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Recently, Gaussian splatting has demonstrated significant success in novel view synthesis. Current methods often regress Gaussians with pixel or point cloud correspondence, linking each Gaussian with a pixel or a 3D point. This leads to the redundancy of Gaussians being used to overfit the correspondence rather than the objects represented by the 3D Gaussians themselves, consequently wasting resources and lacking accurate geometries or textures. In this paper, we introduce LeanGaussian, a novel approach that treats each query in deformable Transformer as one 3D Gaussian ellipsoid, breaking the pixel or point cloud correspondence constraints. We leverage deformable decoder to iteratively refine the Gaussians layer-by-layer with the image features as keys and values. Notably, the center of each 3D Gaussian is defined as 3D reference points, which are then projected onto the image for deformable attention in 2D space. On both the ShapeNet SRN dataset (category level) and the Google Scanned Objects dataset (open-category level, trained with the Objaverse dataset), our approach, outperforms prior methods by approximately 6.1%, achieving a PSNR of 25.44 and 22.36, respectively. Additionally, our method achieves a 3D reconstruction speed of 7.2 FPS and rendering speed 500 FPS. Codes are available at https://github.com/jwubz123/LeanGaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06598v1">Stochastic Ray Tracing of 3D Transparent Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 10 pages, 6 figures, 5 tables
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting has recently been widely adopted as a 3D representation for novel-view synthesis, relighting, and text-to-3D generation tasks, offering realistic and detailed results through a collection of explicit 3D Gaussians carrying opacities and view-dependent colors. However, efficient rendering of many transparent primitives remains a significant challenge. Existing approaches either rasterize the 3D Gaussians with approximate sorting per view or rely on high-end RTX GPUs to exhaustively process all ray-Gaussian intersections (bounding Gaussians by meshes). This paper proposes a stochastic ray tracing method to render 3D clouds of transparent primitives. Instead of processing all ray-Gaussian intersections in sequential order, each ray traverses the acceleration structure only once, randomly accepting and shading a single intersection (or N intersections, using a simple extension). This approach minimizes shading time and avoids sorting the Gaussians along the ray while minimizing the register usage and maximizing parallelism even on low-end GPUs. The cost of rays through the Gaussian asset is comparable to that of standard mesh-intersection rays. While our method introduces noise, the shading is unbiased, and the variance is slight, as stochastic acceptance is importance-sampled based on accumulated opacity. The alignment with the Monte Carlo philosophy simplifies implementation and easily integrates our method into a conventional path-tracing framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18931v2">Sort-free Gaussian Splatting via Weighted Sum Rendering</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has emerged as a significant advancement in 3D scene reconstruction, attracting considerable attention due to its ability to recover high-fidelity details while maintaining low complexity. Despite the promising results achieved by 3DGS, its rendering performance is constrained by its dependence on costly non-commutative alpha-blending operations. These operations mandate complex view dependent sorting operations that introduce computational overhead, especially on the resource-constrained platforms such as mobile phones. In this paper, we propose Weighted Sum Rendering, which approximates alpha blending with weighted sums, thereby removing the need for sorting. This simplifies implementation, delivers superior performance, and eliminates the "popping" artifacts caused by sorting. Experimental results show that optimizing a generalized Gaussian splatting formulation to the new differentiable rendering yields competitive image quality. The method was implemented and tested in a mobile device GPU, achieving on average $1.23\times$ faster rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03371v2">SGSST: Scaling Gaussian Splatting StyleTransfer</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Applying style transfer to a full 3D environment is a challenging task that has seen many developments since the advent of neural rendering. 3D Gaussian splatting (3DGS) has recently pushed further many limits of neural rendering in terms of training speed and reconstruction quality. This work introduces SGSST: Scaling Gaussian Splatting Style Transfer, an optimization-based method to apply style transfer to pretrained 3DGS scenes. We demonstrate that a new multiscale loss based on global neural statistics, that we name SOS for Simultaneously Optimized Scales, enables style transfer to ultra-high resolution 3D scenes. Not only SGSST pioneers 3D scene style transfer at such high image resolutions, it also produces superior visual quality as assessed by thorough qualitative, quantitative and perceptual comparisons.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17769v2">ActiveGS: Active Scene Reconstruction Using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 Accepted to IEEE Robotics and Automation Letters
    </div>
    <details class="paper-abstract">
      Robotics applications often rely on scene reconstructions to enable downstream tasks. In this work, we tackle the challenge of actively building an accurate map of an unknown scene using an RGB-D camera on a mobile platform. We propose a hybrid map representation that combines a Gaussian splatting map with a coarse voxel map, leveraging the strengths of both representations: the high-fidelity scene reconstruction capabilities of Gaussian splatting and the spatial modelling strengths of the voxel map. At the core of our framework is an effective confidence modelling technique for the Gaussian splatting map to identify under-reconstructed areas, while utilising spatial information from the voxel map to target unexplored areas and assist in collision-free path planning. By actively collecting scene information in under-reconstructed and unexplored areas for map updates, our approach achieves superior Gaussian splatting reconstruction results compared to state-of-the-art approaches. Additionally, we demonstrate the real-world applicability of our framework using an unmanned aerial vehicle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17486v3">ProtoGS: Efficient and High-Quality Rendering with 3D Gaussian Prototypes</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has made significant strides in novel view synthesis but is limited by the substantial number of Gaussian primitives required, posing challenges for deployment on lightweight devices. Recent methods address this issue by compressing the storage size of densified Gaussians, yet fail to preserve rendering quality and efficiency. To overcome these limitations, we propose ProtoGS to learn Gaussian prototypes to represent Gaussian primitives, significantly reducing the total Gaussian amount without sacrificing visual quality. Our method directly uses Gaussian prototypes to enable efficient rendering and leverage the resulting reconstruction loss to guide prototype learning. To further optimize memory efficiency during training, we incorporate structure-from-motion (SfM) points as anchor points to group Gaussian primitives. Gaussian prototypes are derived within each group by clustering of K-means, and both the anchor points and the prototypes are optimized jointly. Our experiments on real-world and synthetic datasets prove that we outperform existing methods, achieving a substantial reduction in the number of Gaussians, and enabling high rendering speed while maintaining or even enhancing rendering fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05740v1">Micro-splatting: Maximizing Isotropic Constraints for Refined Optimization in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting have achieved impressive scalability and real-time rendering for large-scale scenes but often fall short in capturing fine-grained details. Conventional approaches that rely on relatively large covariance parameters tend to produce blurred representations, while directly reducing covariance sizes leads to sparsity. In this work, we introduce Micro-splatting (Maximizing Isotropic Constraints for Refined Optimization in 3D Gaussian Splatting), a novel framework designed to overcome these limitations. Our approach leverages a covariance regularization term to penalize excessively large Gaussians to ensure each splat remains compact and isotropic. This work implements an adaptive densification strategy that dynamically refines regions with high image gradients by lowering the splitting threshold, followed by loss function enhancement. This strategy results in a denser and more detailed gaussian means where needed, without sacrificing rendering efficiency. Quantitative evaluations using metrics such as L1, L2, PSNR, SSIM, and LPIPS, alongside qualitative comparisons demonstrate that our method significantly enhances fine-details in 3D reconstructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03059v2">Compressing 3D Gaussian Splatting by Noise-Substituted Vector Quantization</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 Appearing in Scandinavian Conference on Image Analysis (SCIA) 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated remarkable effectiveness in 3D reconstruction, achieving high-quality results with real-time radiance field rendering. However, a key challenge is the substantial storage cost: reconstructing a single scene typically requires millions of Gaussian splats, each represented by 59 floating-point parameters, resulting in approximately 1 GB of memory. To address this challenge, we propose a compression method by building separate attribute codebooks and storing only discrete code indices. Specifically, we employ noise-substituted vector quantization technique to jointly train the codebooks and model features, ensuring consistency between gradient descent optimization and parameter discretization. Our method reduces the memory consumption efficiently (around $45\times$) while maintaining competitive reconstruction quality on standard 3D benchmark scenes. Experiments on different codebook sizes show the trade-off between compression ratio and image quality. Furthermore, the trained compressed model remains fully compatible with popular 3DGS viewers and enables faster rendering speed, making it well-suited for practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10972v2">DCSEG: Decoupled 3D Open-Set Segmentation using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 To be published in CVPR Workshop on Open-World 3D Scene Understanding with Foundation Models
    </div>
    <details class="paper-abstract">
      Open-set 3D segmentation represents a major point of interest for multiple downstream robotics and augmented/virtual reality applications. We present a decoupled 3D segmentation pipeline to ensure modularity and adaptability to novel 3D representations as well as semantic segmentation foundation models. We first reconstruct a scene with 3D Gaussians and learn class-agnostic features through contrastive supervision from a 2D instance proposal network. These 3D features are then clustered to form coarse object- or part-level masks. Finally, we match each 3D cluster to class-aware masks predicted by a 2D open-vocabulary segmentation model, assigning semantic labels without retraining the 3D representation. Our decoupled design (1) provides a plug-and-play interface for swapping different 2D or 3D modules, (2) ensures multi-object instance segmentation at no extra cost, and (3) leverages rich 3D geometry for robust scene understanding. We evaluate on synthetic and real-world indoor datasets, demonstrating improved performance over comparable NeRF-based pipelines on mIoU and mAcc, particularly for challenging or long-tail classes. We also show how varying the 2D backbone affects the final segmentation, highlighting the modularity of our framework. These results confirm that decoupling 3D mask proposal and semantic classification can deliver flexible, efficient, and open-vocabulary 3D segmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05296v1">Let it Snow! Animating Static Gaussian Scenes With Dynamic Weather Effects</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 Project webpage: https://galfiebelman.github.io/let-it-snow/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has recently enabled fast and photorealistic reconstruction of static 3D scenes. However, introducing dynamic elements that interact naturally with such static scenes remains challenging. Accordingly, we present a novel hybrid framework that combines Gaussian-particle representations for incorporating physically-based global weather effects into static 3D Gaussian Splatting scenes, correctly handling the interactions of dynamic elements with the static scene. We follow a three-stage process: we first map static 3D Gaussians to a particle-based representation. We then introduce dynamic particles and simulate their motion using the Material Point Method (MPM). Finally, we map the simulated particles back to the Gaussian domain while introducing appearance parameters tailored for specific effects. To correctly handle the interactions of dynamic elements with the static scene, we introduce specialized collision handling techniques. Our approach supports a variety of weather effects, including snowfall, rainfall, fog, and sandstorms, and can also support falling objects, all with physically plausible motion and appearance. Experiments demonstrate that our method significantly outperforms existing approaches in both visual quality and physical realism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05152v1">PanoDreamer: Consistent Text to 360-Degree Scene Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 Accepted by CVPR 2025 Workshop on Computer Vision for Metaverse
    </div>
    <details class="paper-abstract">
      Automatically generating a complete 3D scene from a text description, a reference image, or both has significant applications in fields like virtual reality and gaming. However, current methods often generate low-quality textures and inconsistent 3D structures. This is especially true when extrapolating significantly beyond the field of view of the reference image. To address these challenges, we propose PanoDreamer, a novel framework for consistent, 3D scene generation with flexible text and image control. Our approach employs a large language model and a warp-refine pipeline, first generating an initial set of images and then compositing them into a 360-degree panorama. This panorama is then lifted into 3D to form an initial point cloud. We then use several approaches to generate additional images, from different viewpoints, that are consistent with the initial point cloud and expand/refine the initial point cloud. Given the resulting set of images, we utilize 3D Gaussian Splatting to create the final 3D scene, which can then be rendered from different viewpoints. Experiments demonstrate the effectiveness of PanoDreamer in generating high-quality, geometrically consistent 3D scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19702v5">RNG: Relightable Neural Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 Camera-ready version. Proceedings of CVPR 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown impressive results for the novel view synthesis task, where lighting is assumed to be fixed. However, creating relightable 3D assets, especially for objects with ill-defined shapes (fur, fabric, etc.), remains a challenging task. The decomposition between light, geometry, and material is ambiguous, especially if either smooth surface assumptions or surfacebased analytical shading models do not apply. We propose Relightable Neural Gaussians (RNG), a novel 3DGS-based framework that enables the relighting of objects with both hard surfaces or soft boundaries, while avoiding assumptions on the shading model. We condition the radiance at each point on both view and light directions. We also introduce a shadow cue, as well as a depth refinement network to improve shadow accuracy. Finally, we propose a hybrid forward-deferred fitting strategy to balance geometry and appearance quality. Our method achieves significantly faster training (1.3 hours) and rendering (60 frames per second) compared to a prior method based on neural radiance fields and produces higher-quality shadows than a concurrent 3DGS-based method. Project page: https://www.whois-jiahui.fun/project_pages/RNG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04857v1">3D Gaussian Particle Approximation of VDB Datasets: A Study for Scientific Visualization</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      The complexity and scale of Volumetric and Simulation datasets for Scientific Visualization(SciVis) continue to grow. And the approaches and advantages of memory-efficient data formats and storage techniques for such datasets vary. OpenVDB library and its VDB data format excels in memory efficiency through its hierarchical and dynamic tree structure, with active and inactive sub-trees for data storage. It is heavily used in current production renderers for both animation and rendering stages in VFX pipelines and photorealistic rendering of volumes and fluids. However, it still remains to be fully leveraged in SciVis where domains dealing with sparse scalar fields like porous media, time varying volumes such as tornado and weather simulation or high resolution simulation of Computational Fluid Dynamics present ample number of large challenging data sets.Goal of this paper is not only to explore the use of OpenVDB in SciVis but also to explore a level of detail(LOD) technique using 3D Gaussian particles approximating voxel regions. For rendering, we utilize NVIDIA OptiX library for ray marching through the Gaussians particles. Data modeling using 3D Gaussians has been very popular lately due to success in stereoscopic image to 3D scene conversion using Gaussian Splatting and Gaussian approximation and mixture models aren't entirely new in SciVis as well. Our work explores the integration with rendering software libraries like OpenVDB and OptiX to take advantage of their built-in memory compaction and hardware acceleration features, while also leveraging the performance capabilities of modern GPUs. Thus, we present a SciVis rendering approach that uses 3D Gaussians at varying LOD in a lossy scheme derived from VDB datasets, rather than focusing on photorealistic volume rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04844v1">Embracing Dynamics: Dynamics-aware 4D Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 This paper is currently under reviewed for IROS 2025
    </div>
    <details class="paper-abstract">
      Simultaneous localization and mapping (SLAM) technology now has photorealistic mapping capabilities thanks to the real-time high-fidelity rendering capability of 3D Gaussian splatting (3DGS). However, due to the static representation of scenes, current 3DGS-based SLAM encounters issues with pose drift and failure to reconstruct accurate maps in dynamic environments. To address this problem, we present D4DGS-SLAM, the first SLAM method based on 4DGS map representation for dynamic environments. By incorporating the temporal dimension into scene representation, D4DGS-SLAM enables high-quality reconstruction of dynamic scenes. Utilizing the dynamics-aware InfoModule, we can obtain the dynamics, visibility, and reliability of scene points, and filter stable static points for tracking accordingly. When optimizing Gaussian points, we apply different isotropic regularization terms to Gaussians with varying dynamic characteristics. Experimental results on real-world dynamic scene datasets demonstrate that our method outperforms state-of-the-art approaches in both camera pose tracking and map quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16323v3">LeanGaussian: Breaking Pixel or Point Cloud Correspondence in Modeling 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Recently, Gaussian splatting has demonstrated significant success in novel view synthesis. Current methods often regress Gaussians with pixel or point cloud correspondence, linking each Gaussian with a pixel or a 3D point. This leads to the redundancy of Gaussians being used to overfit the correspondence rather than the objects represented by the 3D Gaussians themselves, consequently wasting resources and lacking accurate geometries or textures. In this paper, we introduce LeanGaussian, a novel approach that treats each query in deformable Transformer as one 3D Gaussian ellipsoid, breaking the pixel or point cloud correspondence constraints. We leverage deformable decoder to iteratively refine the Gaussians layer-by-layer with the image features as keys and values. Notably, the center of each 3D Gaussian is defined as 3D reference points, which are then projected onto the image for deformable attention in 2D space. On both the ShapeNet SRN dataset (category level) and the Google Scanned Objects dataset (open-category level, trained with the Objaverse dataset), our approach, outperforms prior methods by approximately 6.1%, achieving a PSNR of 25.44 and 22.36, respectively. Additionally, our method achieves a 3D reconstruction speed of 7.2 FPS and rendering speed 500 FPS. Codes are available at https://github.com/jwubz123/LeanGaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04679v1">DeclutterNeRF: Generative-Free 3D Scene Recovery for Occlusion Removal</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 Accepted by CVPR 2025 4th CV4Metaverse Workshop. 15 pages, 10 figures. Code and data at: https://github.com/wanzhouliu/declutter-nerf
    </div>
    <details class="paper-abstract">
      Recent novel view synthesis (NVS) techniques, including Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have greatly advanced 3D scene reconstruction with high-quality rendering and realistic detail recovery. Effectively removing occlusions while preserving scene details can further enhance the robustness and applicability of these techniques. However, existing approaches for object and occlusion removal predominantly rely on generative priors, which, despite filling the resulting holes, introduce new artifacts and blurriness. Moreover, existing benchmark datasets for evaluating occlusion removal methods lack realistic complexity and viewpoint variations. To address these issues, we introduce DeclutterSet, a novel dataset featuring diverse scenes with pronounced occlusions distributed across foreground, midground, and background, exhibiting substantial relative motion across viewpoints. We further introduce DeclutterNeRF, an occlusion removal method free from generative priors. DeclutterNeRF introduces joint multi-view optimization of learnable camera parameters, occlusion annealing regularization, and employs an explainable stochastic structural similarity loss, ensuring high-quality, artifact-free reconstructions from incomplete images. Experiments demonstrate that DeclutterNeRF significantly outperforms state-of-the-art methods on our proposed DeclutterSet, establishing a strong baseline for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09680v2">PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 CVPR 2025. 16 pages, 7 figures. Code is publicly available at https://github.com/s3anwu/pbrnerf
    </div>
    <details class="paper-abstract">
      We tackle the ill-posed inverse rendering problem in 3D reconstruction with a Neural Radiance Field (NeRF) approach informed by Physics-Based Rendering (PBR) theory, named PBR-NeRF. Our method addresses a key limitation in most NeRF and 3D Gaussian Splatting approaches: they estimate view-dependent appearance without modeling scene materials and illumination. To address this limitation, we present an inverse rendering (IR) model capable of jointly estimating scene geometry, materials, and illumination. Our model builds upon recent NeRF-based IR approaches, but crucially introduces two novel physics-based priors that better constrain the IR estimation. Our priors are rigorously formulated as intuitive loss terms and achieve state-of-the-art material estimation without compromising novel view synthesis quality. Our method is easily adaptable to other inverse rendering and 3D reconstruction frameworks that require material estimation. We demonstrate the importance of extending current neural rendering approaches to fully model scene properties beyond geometry and view-dependent appearance. Code is publicly available at https://github.com/s3anwu/pbrnerf
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05544v1">View-Dependent Deformation Fields for 2D Editing of 3D Models</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      We propose a method for authoring non-realistic 3D objects (represented as either 3D Gaussian Splats or meshes), that comply with 2D edits from specific viewpoints. Namely, given a 3D object, a user chooses different viewpoints and interactively deforms the object in the 2D image plane of each view. The method then produces a "deformation field" - an interpolation between those 2D deformations in a smooth manner as the viewpoint changes. Our core observation is that the 2D deformations do not need to be tied to an underlying object, nor share the same deformation space. We use this observation to devise a method for authoring view-dependent deformations, holding several technical contributions: first, a novel way to compositionality-blend between the 2D deformations after lifting them to 3D - this enables the user to "stack" the deformations similarly to layers in an editing software, each deformation operating on the results of the previous; second, a novel method to apply the 3D deformation to 3D Gaussian Splats; third, an approach to author the 2D deformations, by deforming a 2D mesh encapsulating a rendered image of the object. We show the versatility and efficacy of our method by adding cartoonish effects to objects, providing means to modify human characters, fitting 3D models to given 2D sketches and caricatures, resolving occlusions, and recreating classic non-realistic paintings as 3D models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05517v1">L3GS: Layered 3D Gaussian Splats for Efficient 3D Scene Delivery</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Traditional 3D content representations include dense point clouds that consume large amounts of data and hence network bandwidth, while newer representations such as neural radiance fields suffer from poor frame rates due to their non-standard volumetric rendering pipeline. 3D Gaussian splats (3DGS) can be seen as a generalization of point clouds that meet the best of both worlds, with high visual quality and efficient rendering for real-time frame rates. However, delivering 3DGS scenes from a hosting server to client devices is still challenging due to high network data consumption (e.g., 1.5 GB for a single scene). The goal of this work is to create an efficient 3D content delivery framework that allows users to view high quality 3D scenes with 3DGS as the underlying data representation. The main contributions of the paper are: (1) Creating new layered 3DGS scenes for efficient delivery, (2) Scheduling algorithms to choose what splats to download at what time, and (3) Trace-driven experiments from users wearing virtual reality headsets to evaluate the visual quality and latency. Our system for Layered 3D Gaussian Splats delivery L3GS demonstrates high visual quality, achieving 16.9% higher average SSIM compared to baselines, and also works with other compressed 3DGS representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04612v1">Tool-as-Interface: Learning Robot Policies from Human Tool Usage through Imitation Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-06
      | 💬 Project Page: https://tool-as-interface.github.io. 17 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Tool use is critical for enabling robots to perform complex real-world tasks, and leveraging human tool-use data can be instrumental for teaching robots. However, existing data collection methods like teleoperation are slow, prone to control delays, and unsuitable for dynamic tasks. In contrast, human natural data, where humans directly perform tasks with tools, offers natural, unstructured interactions that are both efficient and easy to collect. Building on the insight that humans and robots can share the same tools, we propose a framework to transfer tool-use knowledge from human data to robots. Using two RGB cameras, our method generates 3D reconstruction, applies Gaussian splatting for novel view augmentation, employs segmentation models to extract embodiment-agnostic observations, and leverages task-space tool-action representations to train visuomotor policies. We validate our approach on diverse real-world tasks, including meatball scooping, pan flipping, wine bottle balancing, and other complex tasks. Our method achieves a 71\% higher average success rate compared to diffusion policies trained with teleoperation data and reduces data collection time by 77\%, with some tasks solvable only by our framework. Compared to hand-held gripper, our method cuts data collection time by 41\%. Additionally, our method bridges the embodiment gap, improves robustness to variations in camera viewpoints and robot configurations, and generalizes effectively across objects and spatial setups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04448v1">Thermoxels: a voxel-based method to generate simulation-ready 3D thermal models</a></div>
    <div class="paper-meta">
      📅 2025-04-06
      | 💬 7 pages, 2 figures
    </div>
    <details class="paper-abstract">
      In the European Union, buildings account for 42% of energy use and 35% of greenhouse gas emissions. Since most existing buildings will still be in use by 2050, retrofitting is crucial for emissions reduction. However, current building assessment methods rely mainly on qualitative thermal imaging, which limits data-driven decisions for energy savings. On the other hand, quantitative assessments using finite element analysis (FEA) offer precise insights but require manual CAD design, which is tedious and error-prone. Recent advances in 3D reconstruction, such as Neural Radiance Fields (NeRF) and Gaussian Splatting, enable precise 3D modeling from sparse images but lack clearly defined volumes and the interfaces between them needed for FEA. We propose Thermoxels, a novel voxel-based method able to generate FEA-compatible models, including both geometry and temperature, from a sparse set of RGB and thermal images. Using pairs of RGB and thermal images as input, Thermoxels represents a scene's geometry as a set of voxels comprising color and temperature information. After optimization, a simple process is used to transform Thermoxels' models into tetrahedral meshes compatible with FEA. We demonstrate Thermoxels' capability to generate RGB+Thermal meshes of 3D scenes, surpassing other state-of-the-art methods. To showcase the practical applications of Thermoxels' models, we conduct a simple heat conduction simulation using FEA, achieving convergence from an initial state defined by Thermoxels' thermal reconstruction. Additionally, we compare Thermoxels' image synthesis abilities with current state-of-the-art methods, showing competitive results, and discuss the limitations of existing metrics in assessing mesh quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17190v5">SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-06
      | 💬 Project page: https://gynjn.github.io/selfsplat/
    </div>
    <details class="paper-abstract">
      We propose SelfSplat, a novel 3D Gaussian Splatting model designed to perform pose-free and 3D prior-free generalizable 3D reconstruction from unposed multi-view images. These settings are inherently ill-posed due to the lack of ground-truth data, learned geometric information, and the need to achieve accurate 3D reconstruction without finetuning, making it difficult for conventional methods to achieve high-quality results. Our model addresses these challenges by effectively integrating explicit 3D representations with self-supervised depth and pose estimation techniques, resulting in reciprocal improvements in both pose accuracy and 3D reconstruction quality. Furthermore, we incorporate a matching-aware pose estimation network and a depth refinement module to enhance geometry consistency across views, ensuring more accurate and stable 3D reconstructions. To present the performance of our method, we evaluated it on large-scale real-world datasets, including RealEstate10K, ACID, and DL3DV. SelfSplat achieves superior results over previous state-of-the-art methods in both appearance and geometry quality, also demonstrates strong cross-dataset generalization capabilities. Extensive ablation studies and analysis also validate the effectiveness of our proposed methods. Code and pretrained models are available at https://gynjn.github.io/selfsplat/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04294v1">3R-GS: Best Practice in Optimizing Camera Poses Along with 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has revolutionized neural rendering with its efficiency and quality, but like many novel view synthesis methods, it heavily depends on accurate camera poses from Structure-from-Motion (SfM) systems. Although recent SfM pipelines have made impressive progress, questions remain about how to further improve both their robust performance in challenging conditions (e.g., textureless scenes) and the precision of camera parameter estimation simultaneously. We present 3R-GS, a 3D Gaussian Splatting framework that bridges this gap by jointly optimizing 3D Gaussians and camera parameters from large reconstruction priors MASt3R-SfM. We note that naively performing joint 3D Gaussian and camera optimization faces two challenges: the sensitivity to the quality of SfM initialization, and its limited capacity for global optimization, leading to suboptimal reconstruction results. Our 3R-GS, overcomes these issues by incorporating optimized practices, enabling robust scene reconstruction even with imperfect camera registration. Extensive experiments demonstrate that 3R-GS delivers high-quality novel view synthesis and precise camera pose estimation while remaining computationally efficient. Project page: https://zsh523.github.io/3R-GS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04190v1">Interpretable Single-View 3D Gaussian Splatting using Unsupervised Hierarchical Disentangled Representation Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has recently marked a significant advancement in 3D reconstruction, delivering both rapid rendering and high-quality results. However, existing 3DGS methods pose challenges in understanding underlying 3D semantics, which hinders model controllability and interpretability. To address it, we propose an interpretable single-view 3DGS framework, termed 3DisGS, to discover both coarse- and fine-grained 3D semantics via hierarchical disentangled representation learning (DRL). Specifically, the model employs a dual-branch architecture, consisting of a point cloud initialization branch and a triplane-Gaussian generation branch, to achieve coarse-grained disentanglement by separating 3D geometry and visual appearance features. Subsequently, fine-grained semantic representations within each modality are further discovered through DRL-based encoder-adapters. To our knowledge, this is the first work to achieve unsupervised interpretable 3DGS. Evaluations indicate that our model achieves 3D disentanglement while preserving high-quality and rapid reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15966v2">Gaussian Scenes: Pose-Free Sparse-View Scene Reconstruction using Depth-Enhanced Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2025-04-05
      | 💬 Project page is available at https://gaussianscenes.github.io/
    </div>
    <details class="paper-abstract">
      In this work, we introduce a generative approach for pose-free (without camera parameters) reconstruction of 360 scenes from a sparse set of 2D images. Pose-free scene reconstruction from incomplete, pose-free observations is usually regularized with depth estimation or 3D foundational priors. While recent advances have enabled sparse-view reconstruction of large complex scenes (with high degree of foreground and background detail) with known camera poses using view-conditioned generative priors, these methods cannot be directly adapted for the pose-free setting when ground-truth poses are not available during evaluation. To address this, we propose an image-to-image generative model designed to inpaint missing details and remove artifacts in novel view renders and depth maps of a 3D scene. We introduce context and geometry conditioning using Feature-wise Linear Modulation (FiLM) modulation layers as a lightweight alternative to cross-attention and also propose a novel confidence measure for 3D Gaussian splat representations to allow for better detection of these artifacts. By progressively integrating these novel views in a Gaussian-SLAM-inspired process, we achieve a multi-view-consistent 3D representation. Evaluations on the MipNeRF360 and DL3DV-10K benchmark datasets demonstrate that our method surpasses existing pose-free techniques and performs competitively with state-of-the-art posed (precomputed camera parameters are given) reconstruction methods in complex 360 scenes.
    </details>
</div>
