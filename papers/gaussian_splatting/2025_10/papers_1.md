# gaussian splatting - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10691v3">Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      This paper presents a unified framework that allows high-quality dynamic Gaussian Splatting from both defocused and motion-blurred monocular videos. Due to the significant difference between the formation processes of defocus blur and motion blur, existing methods are tailored for either one of them, lacking the ability to simultaneously deal with both of them. Although the two can be jointly modeled as blur kernel-based convolution, the inherent difficulty in estimating accurate blur kernels greatly limits the progress in this direction. In this work, we go a step further towards this direction. Particularly, we propose to estimate per-pixel reliable blur kernels using a blur prediction network that exploits blur-related scene and camera information and is subject to a blur-aware sparsity constraint. Besides, we introduce a dynamic Gaussian densification strategy to mitigate the lack of Gaussians for incomplete regions, and boost the performance of novel view synthesis by incorporating unseen view information to constrain scene optimization. Extensive experiments show that our method outperforms the state-of-the-art methods in generating photorealistic novel view synthesis from defocused and motion-blurred monocular videos. Our code is available at https://github.com/hhhddddddd/dydeblur.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.05819v3">GASP: Gaussian Splatting for Physic-Based Simulations</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      Physics simulation is paramount for modeling and utilizing 3D scenes in various real-world applications. However, integrating with state-of-the-art 3D scene rendering techniques such as Gaussian Splatting (GS) remains challenging. Existing models use additional meshing mechanisms, including triangle or tetrahedron meshing, marching cubes, or cage meshes. Alternatively, we can modify the physics-grounded Newtonian dynamics to align with 3D Gaussian components. Current models take the first-order approximation of a deformation map, which locally approximates the dynamics by linear transformations. In contrast, our GS for Physics-Based Simulations (GASP) pipeline uses parametrized flat Gaussian distributions. Consequently, the problem of modeling Gaussian components using the physics engine is reduced to working with 3D points. In our work, we present additional rules for manipulating Gaussians, demonstrating how to adapt the pipeline to incorporate meshes, control Gaussian sizes during simulations, and enhance simulation efficiency. This is achieved through the Gaussian grouping strategy, which implements hierarchical structuring and enables simulations to be performed exclusively on selected Gaussians. The resulting solution can be integrated into any physics engine that can be treated as a black box. As demonstrated in our studies, the proposed pipeline exhibits superior performance on a diverse range of benchmark datasets designed for 3D object rendering. The project webpage, which includes additional visualizations, can be found at https://waczjoan.github.io/GASP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27318v1">SAGS: Self-Adaptive Alias-Free Gaussian Splatting for Dynamic Surgical Endoscopic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      Surgical reconstruction of dynamic tissues from endoscopic videos is a crucial technology in robot-assisted surgery. The development of Neural Radiance Fields (NeRFs) has greatly advanced deformable tissue reconstruction, achieving high-quality results from video and image sequences. However, reconstructing deformable endoscopic scenes remains challenging due to aliasing and artifacts caused by tissue movement, which can significantly degrade visualization quality. The introduction of 3D Gaussian Splatting (3DGS) has improved reconstruction efficiency by enabling a faster rendering pipeline. Nevertheless, existing 3DGS methods often prioritize rendering speed while neglecting these critical issues. To address these challenges, we propose SAGS, a self-adaptive alias-free Gaussian splatting framework. We introduce an attention-driven, dynamically weighted 4D deformation decoder, leveraging 3D smoothing filters and 2D Mip filters to mitigate artifacts in deformable tissue reconstruction and better capture the fine details of tissue movement. Experimental results on two public benchmarks, EndoNeRF and SCARED, demonstrate that our method achieves superior performance in all metrics of PSNR, SSIM, and LPIPS compared to the state of the art while also delivering better visualization quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15392v4">EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 Accepted to NeurIPS 2025,Project Page: https://lbh666.github.io/ef-3dgs/
    </div>
    <details class="paper-abstract">
      Scene reconstruction from casually captured videos has wide applications in real-world scenarios. With recent advancements in differentiable rendering techniques, several methods have attempted to simultaneously optimize scene representations (NeRF or 3DGS) and camera poses. Despite recent progress, existing methods relying on traditional camera input tend to fail in high-speed (or equivalently low-frame-rate) scenarios. Event cameras, inspired by biological vision, record pixel-wise intensity changes asynchronously with high temporal resolution, providing valuable scene and motion information in blind inter-frame intervals. In this paper, we introduce the event camera to aid scene construction from a casually captured video for the first time, and propose Event-Aided Free-Trajectory 3DGS, called EF-3DGS, which seamlessly integrates the advantages of event cameras into 3DGS through three key components. First, we leverage the Event Generation Model (EGM) to fuse events and frames, supervising the rendered views observed by the event stream. Second, we adopt the Contrast Maximization (CMax) framework in a piece-wise manner to extract motion information by maximizing the contrast of the Image of Warped Events (IWE), thereby calibrating the estimated poses. Besides, based on the Linear Event Generation Model (LEGM), the brightness information encoded in the IWE is also utilized to constrain the 3DGS in the gradient domain. Third, to mitigate the absence of color information of events, we introduce photometric bundle adjustment (PBA) to ensure view consistency across events and frames. We evaluate our method on the public Tanks and Temples benchmark and a newly collected real-world dataset, RealEv-DAVIS. Our project page is https://lbh666.github.io/ef-3dgs/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04955v3">MixedGaussianAvatar: Realistically and Geometrically Accurate Head Avatar via Mixed 2D-3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      Reconstructing high-fidelity 3D head avatars is crucial in various applications such as virtual reality. The pioneering methods reconstruct realistic head avatars with Neural Radiance Fields (NeRF), which have been limited by training and rendering speed. Recent methods based on 3D Gaussian Splatting (3DGS) significantly improve the efficiency of training and rendering. However, the surface inconsistency of 3DGS results in subpar geometric accuracy; later, 2DGS uses 2D surfels to enhance geometric accuracy at the expense of rendering fidelity. To leverage the benefits of both 2DGS and 3DGS, we propose a novel method named MixedGaussianAvatar for realistically and geometrically accurate head avatar reconstruction. Our main idea is to utilize 2D Gaussians to reconstruct the surface of the 3D head, ensuring geometric accuracy. We attach the 2D Gaussians to the triangular mesh of the FLAME model and connect additional 3D Gaussians to those 2D Gaussians where the rendering quality of 2DGS is inadequate, creating a mixed 2D-3D Gaussian representation. These 2D-3D Gaussians can then be animated using FLAME parameters. We further introduce a progressive training strategy that first trains the 2D Gaussians and then fine-tunes the mixed 2D-3D Gaussians. We use a unified mixed Gaussian representation to integrate the two modalities of 2D image and 3D mesh. Furthermore, the comprehensive experiments demonstrate the superiority of MixedGaussianAvatar. The code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27133v1">WildfireX-SLAM: A Large-scale Low-altitude RGB-D Dataset for Wildfire SLAM and Beyond</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 This paper has been accepted by MMM 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) and its subsequent variants have led to remarkable progress in simultaneous localization and mapping (SLAM). While most recent 3DGS-based SLAM works focus on small-scale indoor scenes, developing 3DGS-based SLAM methods for large-scale forest scenes holds great potential for many real-world applications, especially for wildfire emergency response and forest management. However, this line of research is impeded by the absence of a comprehensive and high-quality dataset, and collecting such a dataset over real-world scenes is costly and technically infeasible. To this end, we have built a large-scale, comprehensive, and high-quality synthetic dataset for SLAM in wildfire and forest environments. Leveraging the Unreal Engine 5 Electric Dreams Environment Sample Project, we developed a pipeline to easily collect aerial and ground views, including ground-truth camera poses and a range of additional data modalities from unmanned aerial vehicle. Our pipeline also provides flexible controls on environmental factors such as light, weather, and types and conditions of wildfire, supporting the need for various tasks covering forest mapping, wildfire emergency response, and beyond. The resulting pilot dataset, WildfireX-SLAM, contains 5.5k low-altitude RGB-D aerial images from a large-scale forest map with a total size of 16 km2. On top of WildfireX-SLAM, a thorough benchmark is also conducted, which not only reveals the unique challenges of 3DGS-based SLAM in the forest but also highlights potential improvements for future works. The dataset and code will be publicly available. Project page: https://zhicongsun.github.io/wildfirexslam.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26786v1">HEIR: Learning Graph-Based Motion Hierarchies</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Code link: https://github.com/princeton-computational-imaging/HEIR
    </div>
    <details class="paper-abstract">
      Hierarchical structures of motion exist across research fields, including computer vision, graphics, and robotics, where complex dynamics typically arise from coordinated interactions among simpler motion components. Existing methods to model such dynamics typically rely on manually-defined or heuristic hierarchies with fixed motion primitives, limiting their generalizability across different tasks. In this work, we propose a general hierarchical motion modeling method that learns structured, interpretable motion relationships directly from data. Our method represents observed motions using graph-based hierarchies, explicitly decomposing global absolute motions into parent-inherited patterns and local motion residuals. We formulate hierarchy inference as a differentiable graph learning problem, where vertices represent elemental motions and directed edges capture learned parent-child dependencies through graph neural networks. We evaluate our hierarchical reconstruction approach on three examples: 1D translational motion, 2D rotational motion, and dynamic 3D scene deformation via Gaussian splatting. Experimental results show that our method reconstructs the intrinsic motion hierarchy in 1D and 2D cases, and produces more realistic and interpretable deformations compared to the baseline on dynamic 3D Gaussian splatting scenes. By providing an adaptable, data-driven hierarchical modeling paradigm, our method offers a formulation applicable to a broad range of motion-centric tasks. Project Page: https://light.princeton.edu/HEIR/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26694v1">The Impact and Outlook of 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Article written for Frontiers of Science Award, International Congress on Basic Science, 2025
    </div>
    <details class="paper-abstract">
      Since its introduction, 3D Gaussian Splatting (3DGS) has rapidly transformed the landscape of 3D scene representations, inspiring an extensive body of associated research. Follow-up work includes analyses and contributions that enhance the efficiency, scalability, and real-world applicability of 3DGS. In this summary, we present an overview of several key directions that have emerged in the wake of 3DGS. We highlight advances enabling resource-efficient training and rendering, the evolution toward dynamic (or four-dimensional, 4DGS) representations, and deeper exploration of the mathematical foundations underlying its appearance modeling and rendering process. Furthermore, we examine efforts to bring 3DGS to mobile and virtual reality platforms, its extension to massive-scale environments, and recent progress toward near-instant radiance field reconstruction via feed-forward or distributed computation. Collectively, these developments illustrate how 3DGS has evolved from a breakthrough representation into a versatile and foundational tool for 3D vision and graphics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26358v1">AgriGS-SLAM: Orchard Mapping Across Seasons via Multi-View Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Autonomous robots in orchards require real-time 3D scene understanding despite repetitive row geometry, seasonal appearance changes, and wind-driven foliage motion. We present AgriGS-SLAM, a Visual--LiDAR SLAM framework that couples direct LiDAR odometry and loop closures with multi-camera 3D Gaussian Splatting (3DGS) rendering. Batch rasterization across complementary viewpoints recovers orchard structure under occlusions, while a unified gradient-driven map lifecycle executed between keyframes preserves fine details and bounds memory. Pose refinement is guided by a probabilistic LiDAR-based depth consistency term, back-propagated through the camera projection to tighten geometry-appearance coupling. We deploy the system on a field platform in apple and pear orchards across dormancy, flowering, and harvesting, using a standardized trajectory protocol that evaluates both training-view and novel-view synthesis to reduce 3DGS overfitting in evaluation. Across seasons and sites, AgriGS-SLAM delivers sharper, more stable reconstructions and steadier trajectories than recent state-of-the-art 3DGS-SLAM baselines while maintaining real-time performance on-tractor. While demonstrated in orchard monitoring, the approach can be applied to other outdoor domains requiring robust multimodal perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22159v3">Disentangled 4D Gaussian Splatting: Rendering High-Resolution Dynamic World at 343 FPS</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      While dynamic novel view synthesis from 2D videos has seen progress, achieving efficient reconstruction and rendering of dynamic scenes remains a challenging task. In this paper, we introduce Disentangled 4D Gaussian Splatting (Disentangled4DGS), a novel representation and rendering pipeline that achieves real-time performance without compromising visual fidelity. Disentangled4DGS decouples the temporal and spatial components of 4D Gaussians, avoiding the need for slicing first and four-dimensional matrix calculations in prior methods. By projecting temporal and spatial deformations into dynamic 2D Gaussians and deferring temporal processing, we minimize redundant computations of 4DGS. Our approach also features a gradient-guided flow loss and temporal splitting strategy to reduce artifacts. Experiments demonstrate a significant improvement in rendering speed and quality, achieving 343 FPS when render 1352*1014 resolution images on a single RTX3090 while reducing storage requirements by at least 4.5%. Our approach sets a new benchmark for dynamic novel view synthesis, outperforming existing methods on both multi-view and monocular dynamic scene datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26117v1">JOGS: Joint Optimization of Pose Estimation and 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Traditional novel view synthesis methods heavily rely on external camera pose estimation tools such as COLMAP, which often introduce computational bottlenecks and propagate errors. To address these challenges, we propose a unified framework that jointly optimizes 3D Gaussian points and camera poses without requiring pre-calibrated inputs. Our approach iteratively refines 3D Gaussian parameters and updates camera poses through a novel co-optimization strategy, ensuring simultaneous improvements in scene reconstruction fidelity and pose accuracy. The key innovation lies in decoupling the joint optimization into two interleaved phases: first, updating 3D Gaussian parameters via differentiable rendering with fixed poses, and second, refining camera poses using a customized 3D optical flow algorithm that incorporates geometric and photometric constraints. This formulation progressively reduces projection errors, particularly in challenging scenarios with large viewpoint variations and sparse feature distributions, where traditional methods struggle. Extensive evaluations on multiple datasets demonstrate that our approach significantly outperforms existing COLMAP-free techniques in reconstruction quality, and also surpasses the standard COLMAP-based baseline in general.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00868v2">3D Dynamic Fluid Assets from Single-View Videos with Generative Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      While the generation of 3D content from single-view images has been extensively studied, the creation of physically consistent 3D dynamic scenes from videos remains in its early stages. We propose a novel framework leveraging generative 3D Gaussian Splatting (3DGS) models to extract and re-simulate 3D dynamic fluid objects from single-view videos using simulation methods. The fluid geometry represented by 3DGS is initially generated and optimized from single-view images, then denoised, densified, and aligned across frames. We estimate the fluid surface velocity using optical flow, propose a mainstream extraction algorithm to refine it. The 3D volumetric velocity field is then derived from the velocity of the fluid's enclosed surface. The velocity field is therewith converted into a divergence-free, grid-based representation, enabling the optimization of simulation parameters through its differentiability across frames. This process outputs simulation-ready fluid assets with physical dynamics closely matching those observed in the source video. Our approach is applicable to various liquid fluids, including inviscid and viscous types, and allows users to edit the output geometry or extend movement durations seamlessly. This automatic method for creating 3D dynamic fluid assets from single-view videos, easily obtainable from the internet, shows great potential for generating large-scale 3D fluid assets at a low cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26921v1">DC4GS: Directional Consistency-Driven Adaptive Density Control for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Accepted to NeurIPS 2025 / Project page: https://github.com/cgskku/dc4gs
    </div>
    <details class="paper-abstract">
      We present a Directional Consistency (DC)-driven Adaptive Density Control (ADC) for 3D Gaussian Splatting (DC4GS). Whereas the conventional ADC bases its primitive splitting on the magnitudes of positional gradients, we further incorporate the DC of the gradients into ADC, and realize it through the angular coherence of the gradients. Our DC better captures local structural complexities in ADC, avoiding redundant splitting. When splitting is required, we again utilize the DC to define optimal split positions so that sub-primitives best align with the local structures than the conventional random placement. As a consequence, our DC4GS greatly reduces the number of primitives (up to 30% in our experiments) than the existing ADC, and also enhances reconstruction fidelity greatly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.24096v2">MILo: Mesh-In-the-Loop Gaussian Splatting for Detailed and Efficient Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 10 pages. A presentation video of our approach is available at https://youtu.be/_SGNhhNz0fE
    </div>
    <details class="paper-abstract">
      While recent advances in Gaussian Splatting have enabled fast reconstruction of high-quality 3D scenes from images, extracting accurate surface meshes remains a challenge. Current approaches extract the surface through costly post-processing steps, resulting in the loss of fine geometric details or requiring significant time and leading to very dense meshes with millions of vertices. More fundamentally, the a posteriori conversion from a volumetric to a surface representation limits the ability of the final mesh to preserve all geometric structures captured during training. We present MILo, a novel Gaussian Splatting framework that bridges the gap between volumetric and surface representations by differentiably extracting a mesh from the 3D Gaussians. We design a fully differentiable procedure that constructs the mesh-including both vertex locations and connectivity-at every iteration directly from the parameters of the Gaussians, which are the only quantities optimized during training. Our method introduces three key technical contributions: a bidirectional consistency framework ensuring both representations-Gaussians and the extracted mesh-capture the same underlying geometry during training; an adaptive mesh extraction process performed at each training iteration, which uses Gaussians as differentiable pivots for Delaunay triangulation; a novel method for computing signed distance values from the 3D Gaussians that enables precise surface extraction while avoiding geometric erosion. Our approach can reconstruct complete scenes, including backgrounds, with state-of-the-art quality while requiring an order of magnitude fewer mesh vertices than previous methods. Due to their light weight and empty interior, our meshes are well suited for downstream applications such as physics simulations or animation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09518v2">HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Accepted to NeurIPS 2025. Project page: https://echopickle.github.io/HAIF-GS.github.io/
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic 3D scenes from monocular videos remains a fundamental challenge in 3D vision. While 3D Gaussian Splatting (3DGS) achieves real-time rendering in static settings, extending it to dynamic scenes is challenging due to the difficulty of learning structured and temporally consistent motion representations. This challenge often manifests as three limitations in existing methods: redundant Gaussian updates, insufficient motion supervision, and weak modeling of complex non-rigid deformations. These issues collectively hinder coherent and efficient dynamic reconstruction. To address these limitations, we propose HAIF-GS, a unified framework that enables structured and consistent dynamic modeling through sparse anchor-driven deformation. It first identifies motion-relevant regions via an Anchor Filter to suppress redundant updates in static areas. A self-supervised Induced Flow-Guided Deformation module induces anchor motion using multi-frame feature aggregation, eliminating the need for explicit flow labels. To further handle fine-grained deformations, a Hierarchical Anchor Propagation mechanism increases anchor resolution based on motion complexity and propagates multi-level transformations. Extensive experiments on synthetic and real-world benchmarks validate that HAIF-GS significantly outperforms prior dynamic 3DGS methods in rendering quality, temporal coherence, and reconstruction efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12015v2">InstDrive: Instance-Aware 3D Gaussian Splatting for Driving Scenes</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic driving scenes from dashcam videos has attracted increasing attention due to its significance in autonomous driving and scene understanding. While recent advances have made impressive progress, most methods still unify all background elements into a single representation, hindering both instance-level understanding and flexible scene editing. Some approaches attempt to lift 2D segmentation into 3D space, but often rely on pre-processed instance IDs or complex pipelines to map continuous features to discrete identities. Moreover, these methods are typically designed for indoor scenes with rich viewpoints, making them less applicable to outdoor driving scenarios. In this paper, we present InstDrive, an instance-aware 3D Gaussian Splatting framework tailored for the interactive reconstruction of dynamic driving scene. We use masks generated by SAM as pseudo ground-truth to guide 2D feature learning via contrastive loss and pseudo-supervised objectives. At the 3D level, we introduce regularization to implicitly encode instance identities and enforce consistency through a voxel-based loss. A lightweight static codebook further bridges continuous features and discrete identities without requiring data pre-processing or complex optimization. Quantitative and qualitative experiments demonstrate the effectiveness of InstDrive, and to the best of our knowledge, it is the first framework to achieve 3D instance segmentation in dynamic, open-world driving scenes.More visualizations are available at our project page.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25173v1">$D^2GS$: Dense Depth Regularization for LiDAR-free Urban Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Recently, Gaussian Splatting (GS) has shown great potential for urban scene reconstruction in the field of autonomous driving. However, current urban scene reconstruction methods often depend on multimodal sensors as inputs, \textit{i.e.} LiDAR and images. Though the geometry prior provided by LiDAR point clouds can largely mitigate ill-posedness in reconstruction, acquiring such accurate LiDAR data is still challenging in practice: i) precise spatiotemporal calibration between LiDAR and other sensors is required, as they may not capture data simultaneously; ii) reprojection errors arise from spatial misalignment when LiDAR and cameras are mounted at different locations. To avoid the difficulty of acquiring accurate LiDAR depth, we propose $D^2GS$, a LiDAR-free urban scene reconstruction framework. In this work, we obtain geometry priors that are as effective as LiDAR while being denser and more accurate. $\textbf{First}$, we initialize a dense point cloud by back-projecting multi-view metric depth predictions. This point cloud is then optimized by a Progressive Pruning strategy to improve the global consistency. $\textbf{Second}$, we jointly refine Gaussian geometry and predicted dense metric depth via a Depth Enhancer. Specifically, we leverage diffusion priors from a depth foundation model to enhance the depth maps rendered by Gaussians. In turn, the enhanced depths provide stronger geometric constraints during Gaussian training. $\textbf{Finally}$, we improve the accuracy of ground geometry by constraining the shape and normal attributes of Gaussians within road regions. Extensive experiments on the Waymo dataset demonstrate that our method consistently outperforms state-of-the-art methods, producing more accurate geometry even when compared with those using ground-truth LiDAR data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25129v1">AtlasGS: Atlanta-world Guided Surface Reconstruction with Implicit Structured Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 18 pages, 11 figures. NeurIPS 2025; Project page: https://zju3dv.github.io/AtlasGS/
    </div>
    <details class="paper-abstract">
      3D reconstruction of indoor and urban environments is a prominent research topic with various downstream applications. However, existing geometric priors for addressing low-texture regions in indoor and urban settings often lack global consistency. Moreover, Gaussian Splatting and implicit SDF fields often suffer from discontinuities or exhibit computational inefficiencies, resulting in a loss of detail. To address these issues, we propose an Atlanta-world guided implicit-structured Gaussian Splatting that achieves smooth indoor and urban scene reconstruction while preserving high-frequency details and rendering efficiency. By leveraging the Atlanta-world model, we ensure the accurate surface reconstruction for low-texture regions, while the proposed novel implicit-structured GS representations provide smoothness without sacrificing efficiency and high-frequency details. Specifically, we propose a semantic GS representation to predict the probability of all semantic regions and deploy a structure plane regularization with learnable plane indicators for global accurate surface reconstruction. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches in both indoor and urban scenes, delivering superior surface reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23158v2">LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 NeurIPS 2025; Web: https://lodge-gs.github.io/
    </div>
    <details class="paper-abstract">
      In this work, we present a novel level-of-detail (LOD) method for 3D Gaussian Splatting that enables real-time rendering of large-scale scenes on memory-constrained devices. Our approach introduces a hierarchical LOD representation that iteratively selects optimal subsets of Gaussians based on camera distance, thus largely reducing both rendering time and GPU memory usage. We construct each LOD level by applying a depth-aware 3D smoothing filter, followed by importance-based pruning and fine-tuning to maintain visual fidelity. To further reduce memory overhead, we partition the scene into spatial chunks and dynamically load only relevant Gaussians during rendering, employing an opacity-blending mechanism to avoid visual artifacts at chunk boundaries. Our method achieves state-of-the-art performance on both outdoor (Hierarchical 3DGS) and indoor (Zip-NeRF) datasets, delivering high-quality renderings with reduced latency and memory requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17345v2">NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 NeurIPS 2025 D&B; Web: https://jkulhanek.com/nerfbaselines
    </div>
    <details class="paper-abstract">
      Novel view synthesis is an important problem with many applications, including AR/VR, gaming, and robotic simulations. With the recent rapid development of Neural Radiance Fields (NeRFs) and 3D Gaussian Splatting (3DGS) methods, it is becoming difficult to keep track of the current state of the art (SoTA) due to methods using different evaluation protocols, codebases being difficult to install and use, and methods not generalizing well to novel 3D scenes. In our experiments, we show that even tiny differences in the evaluation protocols of various methods can artificially boost the performance of these methods. This raises questions about the validity of quantitative comparisons performed in the literature. To address these questions, we propose NerfBaselines, an evaluation framework which provides consistent benchmarking tools, ensures reproducibility, and simplifies the installation and use of various methods. We validate our implementation experimentally by reproducing the numbers reported in the original papers. For improved accessibility, we release a web platform that compares commonly used methods on standard benchmarks. We strongly believe NerfBaselines is a valuable contribution to the community as it ensures that quantitative results are comparable and thus truly measure progress in the field of novel view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24335v1">NVSim: Novel View Synthesis Simulator for Large Scale Indoor Navigation</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 9 pages, 10 figures
    </div>
    <details class="paper-abstract">
      We present NVSim, a framework that automatically constructs large-scale, navigable indoor simulators from only common image sequences, overcoming the cost and scalability limitations of traditional 3D scanning. Our approach adapts 3D Gaussian Splatting to address visual artifacts on sparsely observed floors a common issue in robotic traversal data. We introduce Floor-Aware Gaussian Splatting to ensure a clean, navigable ground plane, and a novel mesh-free traversability checking algorithm that constructs a topological graph by directly analyzing rendered views. We demonstrate our system's ability to generate valid, large-scale navigation graphs from real-world data. A video demonstration is avilable at https://youtu.be/tTiIQt6nXC8
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24118v1">LagMemo: Language 3D Gaussian Splatting Memory for Multi-modal Open-vocabulary Multi-goal Visual Navigation</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Navigating to a designated goal using visual information is a fundamental capability for intelligent robots. Most classical visual navigation methods are restricted to single-goal, single-modality, and closed set goal settings. To address the practical demands of multi-modal, open-vocabulary goal queries and multi-goal visual navigation, we propose LagMemo, a navigation system that leverages a language 3D Gaussian Splatting memory. During exploration, LagMemo constructs a unified 3D language memory. With incoming task goals, the system queries the memory, predicts candidate goal locations, and integrates a local perception-based verification mechanism to dynamically match and validate goals during navigation. For fair and rigorous evaluation, we curate GOAT-Core, a high-quality core split distilled from GOAT-Bench tailored to multi-modal open-vocabulary multi-goal visual navigation. Experimental results show that LagMemo's memory module enables effective multi-modal open-vocabulary goal localization, and that LagMemo outperforms state-of-the-art methods in multi-goal visual navigation. Project page: https://weekgoodday.github.io/lagmemo
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06517v2">GS4: Generalizable Sparse Splatting Semantic SLAM</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 17 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Traditional SLAM algorithms excel at camera tracking, but typically produce incomplete and low-resolution maps that are not tightly integrated with semantics prediction. Recent work integrates Gaussian Splatting (GS) into SLAM to enable dense, photorealistic 3D mapping, yet existing GS-based SLAM methods require per-scene optimization that is slow and consumes an excessive number of Gaussians. We present GS4, the first generalizable GS-based semantic SLAM system. Compared with prior approaches, GS4 runs 10x faster, uses 10x fewer Gaussians, and achieves state-of-the-art performance across color, depth, semantic mapping and camera tracking. From an RGB-D video stream, GS4 incrementally builds and updates a set of 3D Gaussians using a feed-forward network. First, the Gaussian Prediction Model estimates a sparse set of Gaussian parameters from input frame, which integrates both color and semantic prediction with the same backbone. Then, the Gaussian Refinement Network merges new Gaussians with the existing set while avoiding redundancy. Finally, we propose to optimize GS for only 1-5 iterations that corrects drift and floaters when significant pose changes are detected. Experiments on the real-world ScanNet and ScanNet++ benchmarks demonstrate state-of-the-art semantic SLAM performance, with strong generalization capability shown through zero-shot transfer to the NYUv2 and TUM RGB-D datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23988v1">A Survey on Collaborative SLAM with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      This survey comprehensively reviews the evolving field of multi-robot collaborative Simultaneous Localization and Mapping (SLAM) using 3D Gaussian Splatting (3DGS). As an explicit scene representation, 3DGS has enabled unprecedented real-time, high-fidelity rendering, ideal for robotics. However, its use in multi-robot systems introduces significant challenges in maintaining global consistency, managing communication, and fusing data from heterogeneous sources. We systematically categorize approaches by their architecture -- centralized, distributed -- and analyze core components like multi-agent consistency and alignment, communication-efficient, Gaussian representation, semantic distillation, fusion and pose optimization, and real-time scalability. In addition, a summary of critical datasets and evaluation metrics is provided to contextualize performance. Finally, we identify key open challenges and chart future research directions, including lifelong mapping, semantic association and mapping, multi-model for robustness, and bridging the Sim2Real gap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2503.22204v2">Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 NeurIPS 2025. Project page: https://vulab-ai.github.io/Segment-then-Splat/
    </div>
    <details class="paper-abstract">
      Open-vocabulary querying in 3D space is crucial for enabling more intelligent perception in applications such as robotics, autonomous systems, and augmented reality. However, most existing methods rely on 2D pixel-level parsing, leading to multi-view inconsistencies and poor 3D object retrieval. Moreover, they are limited to static scenes and struggle with dynamic scenes due to the complexities of motion modeling. In this paper, we propose Segment then Splat, a 3D-aware open vocabulary segmentation approach for both static and dynamic scenes based on Gaussian Splatting. Segment then Splat reverses the long established approach of "segmentation after reconstruction" by dividing Gaussians into distinct object sets before reconstruction. Once reconstruction is complete, the scene is naturally segmented into individual objects, achieving true 3D segmentation. This design eliminates both geometric and semantic ambiguities, as well as Gaussian-object misalignment issues in dynamic scenes. It also accelerates the optimization process, as it eliminates the need for learning a separate language field. After optimization, a CLIP embedding is assigned to each object to enable open-vocabulary querying. Extensive experiments one various datasets demonstrate the effectiveness of our proposed method in both static and dynamic scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23521v1">Explicit Memory through Online 3D Gaussian Splatting Improves Class-Agnostic Video Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted in IEEE Robotics and Automation Letters September 2025
    </div>
    <details class="paper-abstract">
      Remembering where object segments were predicted in the past is useful for improving the accuracy and consistency of class-agnostic video segmentation algorithms. Existing video segmentation algorithms typically use either no object-level memory (e.g. FastSAM) or they use implicit memories in the form of recurrent neural network features (e.g. SAM2). In this paper, we augment both types of segmentation models using an explicit 3D memory and show that the resulting models have more accurate and consistent predictions. For this, we develop an online 3D Gaussian Splatting (3DGS) technique to store predicted object-level segments generated throughout the duration of a video. Based on this 3DGS representation, a set of fusion techniques are developed, named FastSAM-Splat and SAM2-Splat, that use the explicit 3DGS memory to improve their respective foundation models' predictions. Ablation experiments are used to validate the proposed techniques' design and hyperparameter settings. Results from both real-world and simulated benchmarking experiments show that models which use explicit 3D memories result in more accurate and consistent predictions than those which use no memory or only implicit neural network memories. Project Page: https://topipari.com/projects/FastSAM-Splat/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14081v3">Capture, Canonicalize, Splat: Zero-Shot 3D Gaussian Avatars from Unstructured Phone Images</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 This work received the Best Paper Honorable Mention at the AMFG Workshop, ICCV 2025
    </div>
    <details class="paper-abstract">
      We present a novel, zero-shot pipeline for creating hyperrealistic, identity-preserving 3D avatars from a few unstructured phone images. Existing methods face several challenges: single-view approaches suffer from geometric inconsistencies and hallucinations, degrading identity preservation, while models trained on synthetic data fail to capture high-frequency details like skin wrinkles and fine hair, limiting realism. Our method introduces two key contributions: (1) a generative canonicalization module that processes multiple unstructured views into a standardized, consistent representation, and (2) a transformer-based model trained on a new, large-scale dataset of high-fidelity Gaussian splatting avatars derived from dome captures of real people. This "Capture, Canonicalize, Splat" pipeline produces static quarter-body avatars with compelling realism and robust identity preservation from unstructured photos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23205v1">VR-Drive: Viewpoint-Robust End-to-End Driving with Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted by NeurIPS2025
    </div>
    <details class="paper-abstract">
      End-to-end autonomous driving (E2E-AD) has emerged as a promising paradigm that unifies perception, prediction, and planning into a holistic, data-driven framework. However, achieving robustness to varying camera viewpoints, a common real-world challenge due to diverse vehicle configurations, remains an open problem. In this work, we propose VR-Drive, a novel E2E-AD framework that addresses viewpoint generalization by jointly learning 3D scene reconstruction as an auxiliary task to enable planning-aware view synthesis. Unlike prior scene-specific synthesis approaches, VR-Drive adopts a feed-forward inference strategy that supports online training-time augmentation from sparse views without additional annotations. To further improve viewpoint consistency, we introduce a viewpoint-mixed memory bank that facilitates temporal interaction across multiple viewpoints and a viewpoint-consistent distillation strategy that transfers knowledge from original to synthesized views. Trained in a fully end-to-end manner, VR-Drive effectively mitigates synthesis-induced noise and improves planning under viewpoint shifts. In addition, we release a new benchmark dataset to evaluate E2E-AD performance under novel camera viewpoints, enabling comprehensive analysis. Our results demonstrate that VR-Drive is a scalable and robust solution for the real-world deployment of end-to-end autonomous driving systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23087v1">EndoWave: Rational-Wavelet 4D Gaussian Splatting for Endoscopic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      In robot-assisted minimally invasive surgery, accurate 3D reconstruction from endoscopic video is vital for downstream tasks and improved outcomes. However, endoscopic scenarios present unique challenges, including photometric inconsistencies, non-rigid tissue motion, and view-dependent highlights. Most 3DGS-based methods that rely solely on appearance constraints for optimizing 3DGS are often insufficient in this context, as these dynamic visual artifacts can mislead the optimization process and lead to inaccurate reconstructions. To address these limitations, we present EndoWave, a unified spatio-temporal Gaussian Splatting framework by incorporating an optical flow-based geometric constraint and a multi-resolution rational wavelet supervision. First, we adopt a unified spatio-temporal Gaussian representation that directly optimizes primitives in a 4D domain. Second, we propose a geometric constraint derived from optical flow to enhance temporal coherence and effectively constrain the 3D structure of the scene. Third, we propose a multi-resolution rational orthogonal wavelet as a constraint, which can effectively separate the details of the endoscope and enhance the rendering performance. Extensive evaluations on two real surgical datasets, EndoNeRF and StereoMIS, demonstrate that our method EndoWave achieves state-of-the-art reconstruction quality and visual accuracy compared to the baseline method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22973v1">Scaling Up Occupancy-centric Driving Scene Generation: Dataset and Method</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation/tree/v2
    </div>
    <details class="paper-abstract">
      Driving scene generation is a critical domain for autonomous driving, enabling downstream applications, including perception and planning evaluation. Occupancy-centric methods have recently achieved state-of-the-art results by offering consistent conditioning across frames and modalities; however, their performance heavily depends on annotated occupancy data, which still remains scarce. To overcome this limitation, we curate Nuplan-Occ, the largest semantic occupancy dataset to date, constructed from the widely used Nuplan benchmark. Its scale and diversity facilitate not only large-scale generative modeling but also autonomous driving downstream applications. Based on this dataset, we develop a unified framework that jointly synthesizes high-quality semantic occupancy, multi-view videos, and LiDAR point clouds. Our approach incorporates a spatio-temporal disentangled architecture to support high-fidelity spatial expansion and temporal forecasting of 4D dynamic occupancy. To bridge modal gaps, we further propose two novel techniques: a Gaussian splatting-based sparse point map rendering strategy that enhances multi-view video generation, and a sensor-aware embedding strategy that explicitly models LiDAR sensor properties for realistic multi-LiDAR simulation. Extensive experiments demonstrate that our method achieves superior generation fidelity and scalability compared to existing approaches, and validates its practical value in downstream tasks. Repo: https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation/tree/v2
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22930v1">Gen-LangSplat: Generalized Language Gaussian Splatting with Pre-Trained Feature Compression</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Modeling open-vocabulary language fields in 3D is essential for intuitive human-AI interaction and querying within physical environments. State-of-the-art approaches, such as LangSplat, leverage 3D Gaussian Splatting to efficiently construct these language fields, encoding features distilled from high-dimensional models like CLIP. However, this efficiency is currently offset by the requirement to train a scene-specific language autoencoder for feature compression, introducing a costly, per-scene optimization bottleneck that hinders deployment scalability. In this work, we introduce Gen-LangSplat, that eliminates this requirement by replacing the scene-wise autoencoder with a generalized autoencoder, pre-trained extensively on the large-scale ScanNet dataset. This architectural shift enables the use of a fixed, compact latent space for language features across any new scene without any scene-specific training. By removing this dependency, our entire language field construction process achieves a efficiency boost while delivering querying performance comparable to, or exceeding, the original LangSplat method. To validate our design choice, we perform a thorough ablation study empirically determining the optimal latent embedding dimension and quantifying representational fidelity using Mean Squared Error and cosine similarity between the original and reprojected 512-dimensional CLIP embeddings. Our results demonstrate that generalized embeddings can efficiently and accurately support open-vocabulary querying in novel 3D scenes, paving the way for scalable, real-time interactive 3D AI applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10097v2">Gesplat: Robust Pose-Free 3D Reconstruction via Geometry-Guided Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have advanced 3D reconstruction and novel view synthesis, but remain heavily dependent on accurate camera poses and dense viewpoint coverage. These requirements limit their applicability in sparse-view settings, where pose estimation becomes unreliable and supervision is insufficient. To overcome these challenges, we introduce Gesplat, a 3DGS-based framework that enables robust novel view synthesis and geometrically consistent reconstruction from unposed sparse images. Unlike prior works that rely on COLMAP for sparse point cloud initialization, we leverage the VGGT foundation model to obtain more reliable initial poses and dense point clouds. Our approach integrates several key innovations: 1) a hybrid Gaussian representation with dual position-shape optimization enhanced by inter-view matching consistency; 2) a graph-guided attribute refinement module to enhance scene details; and 3) flow-based depth regularization that improves depth estimation accuracy for more effective supervision. Comprehensive quantitative and qualitative experiments demonstrate that our approach achieves more robust performance on both forward-facing and large-scale complex datasets compared to other pose-free methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10691v2">Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      This paper presents a unified framework that allows high-quality dynamic Gaussian Splatting from both defocused and motion-blurred monocular videos. Due to the significant difference between the formation processes of defocus blur and motion blur, existing methods are tailored for either one of them, lacking the ability to simultaneously deal with both of them. Although the two can be jointly modeled as blur kernel-based convolution, the inherent difficulty in estimating accurate blur kernels greatly limits the progress in this direction. In this work, we go a step further towards this direction. Particularly, we propose to estimate per-pixel reliable blur kernels using a blur prediction network that exploits blur-related scene and camera information and is subject to a blur-aware sparsity constraint. Besides, we introduce a dynamic Gaussian densification strategy to mitigate the lack of Gaussians for incomplete regions, and boost the performance of novel view synthesis by incorporating unseen view information to constrain scene optimization. Extensive experiments show that our method outperforms the state-of-the-art methods in generating photorealistic novel view synthesis from defocused and motion-blurred monocular videos. Our code is available at \href{https://github.com/hhhddddddd/dydeblur}{\textcolor{cyan}{https://github.com/hhhddddddd/dydeblur}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23930v1">PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted by NeurIPS 2025. Project page: https://planargs.github.io
    </div>
    <details class="paper-abstract">
      Three-dimensional Gaussian Splatting (3DGS) has recently emerged as an efficient representation for novel-view synthesis, achieving impressive visual quality. However, in scenes dominated by large and low-texture regions, common in indoor environments, the photometric loss used to optimize 3DGS yields ambiguous geometry and fails to recover high-fidelity 3D surfaces. To overcome this limitation, we introduce PlanarGS, a 3DGS-based framework tailored for indoor scene reconstruction. Specifically, we design a pipeline for Language-Prompted Planar Priors (LP3) that employs a pretrained vision-language segmentation model and refines its region proposals via cross-view fusion and inspection with geometric priors. 3D Gaussians in our framework are optimized with two additional terms: a planar prior supervision term that enforces planar consistency, and a geometric prior supervision term that steers the Gaussians toward the depth and normal cues. We have conducted extensive experiments on standard indoor benchmarks. The results show that PlanarGS reconstructs accurate and detailed 3D surfaces, consistently outperforming state-of-the-art methods by a large margin. Project page: https://planargs.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22204v2">Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 NeurIPS 2025. Project page: https://vulab-ai.github.io/Segment-then-Splat/
    </div>
    <details class="paper-abstract">
      Open-vocabulary querying in 3D space is crucial for enabling more intelligent perception in applications such as robotics, autonomous systems, and augmented reality. However, most existing methods rely on 2D pixel-level parsing, leading to multi-view inconsistencies and poor 3D object retrieval. Moreover, they are limited to static scenes and struggle with dynamic scenes due to the complexities of motion modeling. In this paper, we propose Segment then Splat, a 3D-aware open vocabulary segmentation approach for both static and dynamic scenes based on Gaussian Splatting. Segment then Splat reverses the long established approach of "segmentation after reconstruction" by dividing Gaussians into distinct object sets before reconstruction. Once reconstruction is complete, the scene is naturally segmented into individual objects, achieving true 3D segmentation. This design eliminates both geometric and semantic ambiguities, as well as Gaussian-object misalignment issues in dynamic scenes. It also accelerates the optimization process, as it eliminates the need for learning a separate language field. After optimization, a CLIP embedding is assigned to each object to enable open-vocabulary querying. Extensive experiments one various datasets demonstrate the effectiveness of our proposed method in both static and dynamic scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22812v1">Region-Adaptive Learned Hierarchical Encoding for 3D Gaussian Splatting Data</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 10 Pages, 5 Figures
    </div>
    <details class="paper-abstract">
      We introduce Region-Adaptive Learned Hierarchical Encoding (RALHE) for 3D Gaussian Splatting (3DGS) data. While 3DGS has recently become popular for novel view synthesis, the size of trained models limits its deployment in bandwidth-constrained applications such as volumetric media streaming. To address this, we propose a learned hierarchical latent representation that builds upon the principles of "overfitted" learned image compression (e.g., Cool-Chic and C3) to efficiently encode 3DGS attributes. Unlike images, 3DGS data have irregular spatial distributions of Gaussians (geometry) and consist of multiple attributes (signals) defined on the irregular geometry. Our codec is designed to account for these differences between images and 3DGS. Specifically, we leverage the octree structure of the voxelized 3DGS geometry to obtain a hierarchical multi-resolution representation. Our approach overfits latents to each Gaussian attribute under a global rate constraint. These latents are decoded independently through a lightweight decoder network. To estimate the bitrate during training, we employ an autoregressive probability model that leverages octree-derived contexts from the 3D point structure. The multi-resolution latents, decoder, and autoregressive entropy coding networks are jointly optimized for each Gaussian attribute. Experiments demonstrate that the proposed RALHE compression framework achieves a rendering PSNR gain of up to 2dB at low bitrates (less than 1 MB) compared to the baseline 3DGS compression methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22718v1">Edge Collaborative Gaussian Splatting with Integrated Rendering and Communication</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 5 pages and 7 figures, submitted for possible publication
    </div>
    <details class="paper-abstract">
      Gaussian splatting (GS) struggles with degraded rendering quality on low-cost devices. To address this issue, we present edge collaborative GS (ECO-GS), where each user can switch between a local small GS model to guarantee timeliness and a remote large GS model to guarantee fidelity. However, deciding how to engage the large GS model is nontrivial, due to the interdependency between rendering requirements and resource conditions. To this end, we propose integrated rendering and communication (IRAC), which jointly optimizes collaboration status (i.e., deciding whether to engage large GS) and edge power allocation (i.e., enabling remote rendering) under communication constraints across different users by minimizing a newly-derived GS switching function. Despite the nonconvexity of the problem, we propose an efficient penalty majorization minimization (PMM) algorithm to obtain the critical point solution. Furthermore, we develop an imitation learning optimization (ILO) algorithm, which reduces the computational time by over 100x compared to PMM. Experiments demonstrate the superiority of PMM and the real-time execution capability of ILO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22669v1">LVD-GS: Gaussian Splatting SLAM for Dynamic Scenes via Hierarchical Explicit-Implicit Representation Collaboration Rendering</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting SLAM has emerged as a widely used technique for high-fidelity mapping in spatial intelligence. However, existing methods often rely on a single representation scheme, which limits their performance in large-scale dynamic outdoor scenes and leads to cumulative pose errors and scale ambiguity. To address these challenges, we propose \textbf{LVD-GS}, a novel LiDAR-Visual 3D Gaussian Splatting SLAM system. Motivated by the human chain-of-thought process for information seeking, we introduce a hierarchical collaborative representation module that facilitates mutual reinforcement for mapping optimization, effectively mitigating scale drift and enhancing reconstruction robustness. Furthermore, to effectively eliminate the influence of dynamic objects, we propose a joint dynamic modeling module that generates fine-grained dynamic masks by fusing open-world segmentation with implicit residual constraints, guided by uncertainty estimates from DINO-Depth features. Extensive evaluations on KITTI, nuScenes, and self-collected datasets demonstrate that our approach achieves state-of-the-art performance compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04789v2">Object-X: Learning to Reconstruct Multi-Modal 3D Object Representations</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      Learning effective multi-modal 3D representations of objects is essential for numerous applications, such as augmented reality and robotics. Existing methods often rely on task-specific embeddings that are tailored either for semantic understanding or geometric reconstruction. As a result, these embeddings typically cannot be decoded into explicit geometry and simultaneously reused across tasks. In this paper, we propose Object-X, a versatile multi-modal object representation framework capable of encoding rich object embeddings (e.g. images, point cloud, text) and decoding them back into detailed geometric and visual reconstructions. Object-X operates by geometrically grounding the captured modalities in a 3D voxel grid and learning an unstructured embedding fusing the information from the voxels with the object attributes. The learned embedding enables 3D Gaussian Splatting-based object reconstruction, while also supporting a range of downstream tasks, including scene alignment, single-image 3D object reconstruction, and localization. Evaluations on two challenging real-world datasets demonstrate that Object-X produces high-fidelity novel-view synthesis comparable to standard 3D Gaussian Splatting, while significantly improving geometric accuracy. Moreover, Object-X achieves competitive performance with specialized methods in scene alignment and localization. Critically, our object-centric descriptors require 3-4 orders of magnitude less storage compared to traditional image- or point cloud-based approaches, establishing Object-X as a scalable and highly practical solution for multi-modal 3D scene representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06488v3">NVS-SQA: Exploring Self-Supervised Quality Representation Learning for Neurally Synthesized Scenes without References</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 Accepted by TPAMI
    </div>
    <details class="paper-abstract">
      Neural View Synthesis (NVS), such as NeRF and 3D Gaussian Splatting, effectively creates photorealistic scenes from sparse viewpoints, typically evaluated by quality assessment methods like PSNR, SSIM, and LPIPS. However, these full-reference methods, which compare synthesized views to reference views, may not fully capture the perceptual quality of neurally synthesized scenes (NSS), particularly due to the limited availability of dense reference views. Furthermore, the challenges in acquiring human perceptual labels hinder the creation of extensive labeled datasets, risking model overfitting and reduced generalizability. To address these issues, we propose NVS-SQA, a NSS quality assessment method to learn no-reference quality representations through self-supervision without reliance on human labels. Traditional self-supervised learning predominantly relies on the "same instance, similar representation" assumption and extensive datasets. However, given that these conditions do not apply in NSS quality assessment, we employ heuristic cues and quality scores as learning objectives, along with a specialized contrastive pair preparation process to improve the effectiveness and efficiency of learning. The results show that NVS-SQA outperforms 17 no-reference methods by a large margin (i.e., on average 109.5% in SRCC, 98.6% in PLCC, and 91.5% in KRCC over the second best) and even exceeds 16 full-reference methods across all evaluation metrics (i.e., 22.9% in SRCC, 19.1% in PLCC, and 18.6% in KRCC over the second best).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22600v1">RoGER-SLAM: A Robust Gaussian Splatting SLAM System for Noisy and Low-light Environment Resilience</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 13 pages, 11 figures, under review
    </div>
    <details class="paper-abstract">
      The reliability of Simultaneous Localization and Mapping (SLAM) is severely constrained in environments where visual inputs suffer from noise and low illumination. Although recent 3D Gaussian Splatting (3DGS) based SLAM frameworks achieve high-fidelity mapping under clean conditions, they remain vulnerable to compounded degradations that degrade mapping and tracking performance. A key observation underlying our work is that the original 3DGS rendering pipeline inherently behaves as an implicit low-pass filter, attenuating high-frequency noise but also risking over-smoothing. Building on this insight, we propose RoGER-SLAM, a robust 3DGS SLAM system tailored for noise and low-light resilience. The framework integrates three innovations: a Structure-Preserving Robust Fusion (SP-RoFusion) mechanism that couples rendered appearance, depth, and edge cues; an adaptive tracking objective with residual balancing regularization; and a Contrastive Language-Image Pretraining (CLIP)-based enhancement module, selectively activated under compounded degradations to restore semantic and structural fidelity. Comprehensive experiments on Replica, TUM, and real-world sequences show that RoGER-SLAM consistently improves trajectory accuracy and reconstruction quality compared with other 3DGS-SLAM systems, especially under adverse imaging conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11762v2">GS-ProCams: Gaussian Splatting-based Projector-Camera Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      We present GS-ProCams, the first Gaussian Splatting-based framework for projector-camera systems (ProCams). GS-ProCams is not only view-agnostic but also significantly enhances the efficiency of projection mapping (PM) that requires establishing geometric and radiometric mappings between the projector and the camera. Previous CNN-based ProCams are constrained to a specific viewpoint, limiting their applicability to novel perspectives. In contrast, NeRF-based ProCams support view-agnostic projection mapping, however, they require an additional co-located light source and demand significant computational and memory resources. To address this issue, we propose GS-ProCams that employs 2D Gaussian for scene representations, and enables efficient view-agnostic ProCams applications. In particular, we explicitly model the complex geometric and photometric mappings of ProCams using projector responses, the projection surface's geometry and materials represented by Gaussians, and the global illumination component. Then, we employ differentiable physically-based rendering to jointly estimate them from captured multi-view projections. Compared to state-of-the-art NeRF-based methods, our GS-ProCams eliminates the need for additional devices, achieving superior ProCams simulation quality. It also uses only 1/10 of the GPU memory for training and is 900 times faster in inference speed. Please refer to our project page for the code and dataset: https://realqingyue.github.io/GS-ProCams/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22473v1">DynaPose4D: High-Quality 4D Dynamic Content Generation via Pose Alignment Loss</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      Recent advancements in 2D and 3D generative models have expanded the capabilities of computer vision. However, generating high-quality 4D dynamic content from a single static image remains a significant challenge. Traditional methods have limitations in modeling temporal dependencies and accurately capturing dynamic geometry changes, especially when considering variations in camera perspective. To address this issue, we propose DynaPose4D, an innovative solution that integrates 4D Gaussian Splatting (4DGS) techniques with Category-Agnostic Pose Estimation (CAPE) technology. This framework uses 3D Gaussian Splatting to construct a 3D model from single images, then predicts multi-view pose keypoints based on one-shot support from a chosen view, leveraging supervisory signals to enhance motion consistency. Experimental results show that DynaPose4D achieves excellent coherence, consistency, and fluidity in dynamic motion generation. These findings not only validate the efficacy of the DynaPose4D framework but also indicate its potential applications in the domains of computer vision and animation production.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.22204v2">Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 NeurIPS 2025. Project page: https://vulab-ai.github.io/Segment-then-Splat/
    </div>
    <details class="paper-abstract">
      Open-vocabulary querying in 3D space is crucial for enabling more intelligent perception in applications such as robotics, autonomous systems, and augmented reality. However, most existing methods rely on 2D pixel-level parsing, leading to multi-view inconsistencies and poor 3D object retrieval. Moreover, they are limited to static scenes and struggle with dynamic scenes due to the complexities of motion modeling. In this paper, we propose Segment then Splat, a 3D-aware open vocabulary segmentation approach for both static and dynamic scenes based on Gaussian Splatting. Segment then Splat reverses the long established approach of "segmentation after reconstruction" by dividing Gaussians into distinct object sets before reconstruction. Once reconstruction is complete, the scene is naturally segmented into individual objects, achieving true 3D segmentation. This design eliminates both geometric and semantic ambiguities, as well as Gaussian-object misalignment issues in dynamic scenes. It also accelerates the optimization process, as it eliminates the need for learning a separate language field. After optimization, a CLIP embedding is assigned to each object to enable open-vocabulary querying. Extensive experiments one various datasets demonstrate the effectiveness of our proposed method in both static and dynamic scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22213v1">DynamicTree: Interactive Real Tree Animation via Sparse Voxel Spectrum</a></div>
    <div class="paper-meta">
      📅 2025-10-25
      | 💬 Project Page: https://dynamictree-dev.github.io/DynamicTree.github.io/
    </div>
    <details class="paper-abstract">
      Generating dynamic and interactive 3D objects, such as trees, has wide applications in virtual reality, games, and world simulation. Nevertheless, existing methods still face various challenges in generating realistic 4D motion for complex real trees. In this paper, we propose DynamicTree, the first framework that can generate long-term, interactive animation of 3D Gaussian Splatting trees. Unlike prior optimization-based methods, our approach generates dynamics in a fast feed-forward manner. The key success of our approach is the use of a compact sparse voxel spectrum to represent the tree movement. Given a 3D tree from Gaussian Splatting reconstruction, our pipeline first generates mesh motion using the sparse voxel spectrum and then binds Gaussians to deform the mesh. Additionally, the proposed sparse voxel spectrum can also serve as a basis for fast modal analysis under external forces, allowing real-time interactive responses. To train our model, we also introduce 4DTree, the first large-scale synthetic 4D tree dataset containing 8,786 animated tree meshes with semantic labels and 100-frame motion sequences. Extensive experiments demonstrate that our method achieves realistic and responsive tree animations, significantly outperforming existing approaches in both visual quality and computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12945v2">Metropolis-Hastings Sampling for 3D Gaussian Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-24
      | 💬 NeurIPS 2025. Project Page: https://hjhyunjinkim.github.io/MH-3DGS
    </div>
    <details class="paper-abstract">
      We propose an adaptive sampling framework for 3D Gaussian Splatting (3DGS) that leverages comprehensive multi-view photometric error signals within a unified Metropolis-Hastings approach. Vanilla 3DGS heavily relies on heuristic-based density-control mechanisms (e.g., cloning, splitting, and pruning), which can lead to redundant computations or premature removal of beneficial Gaussians. Our framework overcomes these limitations by reformulating densification and pruning as a probabilistic sampling process, dynamically inserting and relocating Gaussians based on aggregated multi-view errors and opacity scores. Guided by Bayesian acceptance tests derived from these error-based importance scores, our method substantially reduces reliance on heuristics, offers greater flexibility, and adaptively infers Gaussian distributions without requiring predefined scene complexity. Experiments on benchmark datasets, including Mip-NeRF360, Tanks and Temples and Deep Blending, show that our approach reduces the number of Gaussians needed, achieving faster convergence while matching or modestly surpassing the view-synthesis quality of state-of-the-art models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22854v2">CLIPGaussian: Universal and Multimodal Style Transfer Based on Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-24
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has recently emerged as an efficient representation for rendering 3D scenes from 2D images and has been extended to images, videos, and dynamic 4D content. However, applying style transfer to GS-based representations, especially beyond simple color changes, remains challenging. In this work, we introduce CLIPGaussian, the first unified style transfer framework that supports text- and image-guided stylization across multiple modalities: 2D images, videos, 3D objects, and 4D scenes. Our method operates directly on Gaussian primitives and integrates into existing GS pipelines as a plug-in module, without requiring large generative models or retraining from scratch. The CLIPGaussian approach enables joint optimization of color and geometry in 3D and 4D settings, and achieves temporal coherence in videos, while preserving the model size. We demonstrate superior style fidelity and consistency across all tasks, validating CLIPGaussian as a universal and efficient solution for multimodal style transfer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21307v1">Towards Physically Executable 3D Gaussian for Embodied Navigation</a></div>
    <div class="paper-meta">
      📅 2025-10-24
      | 💬 Download link of InteriorGS: https://huggingface.co/datasets/spatialverse/InteriorGS
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS), a 3D representation method with photorealistic real-time rendering capabilities, is regarded as an effective tool for narrowing the sim-to-real gap. However, it lacks fine-grained semantics and physical executability for Visual-Language Navigation (VLN). To address this, we propose SAGE-3D (Semantically and Physically Aligned Gaussian Environments for 3D Navigation), a new paradigm that upgrades 3DGS into an executable, semantically and physically aligned environment. It comprises two components: (1) Object-Centric Semantic Grounding, which adds object-level fine-grained annotations to 3DGS; and (2) Physics-Aware Execution Jointing, which embeds collision objects into 3DGS and constructs rich physical interfaces. We release InteriorGS, containing 1K object-annotated 3DGS indoor scene data, and introduce SAGE-Bench, the first 3DGS-based VLN benchmark with 2M VLN data. Experiments show that 3DGS scene data is more difficult to converge, while exhibiting strong generalizability, improving baseline performance by 31% on the VLN-CE Unseen task. The data and code will be available soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15548v4">MS-GS: Multi-Appearance Sparse-View 3D Gaussian Splatting in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-10-24
    </div>
    <details class="paper-abstract">
      In-the-wild photo collections often contain limited volumes of imagery and exhibit multiple appearances, e.g., taken at different times of day or seasons, posing significant challenges to scene reconstruction and novel view synthesis. Although recent adaptations of Neural Radiance Field (NeRF) and 3D Gaussian Splatting (3DGS) have improved in these areas, they tend to oversmooth and are prone to overfitting. In this paper, we present MS-GS, a novel framework designed with Multi-appearance capabilities in Sparse-view scenarios using 3DGS. To address the lack of support due to sparse initializations, our approach is built on the geometric priors elicited from monocular depth estimations. The key lies in extracting and utilizing local semantic regions with a Structure-from-Motion (SfM) points anchored algorithm for reliable alignment and geometry cues. Then, to introduce multi-view constraints, we propose a series of geometry-guided supervision steps at virtual views in pixel and feature levels to encourage 3D consistency and reduce overfitting. We also introduce a dataset and an in-the-wild experiment setting to set up more realistic benchmarks. We demonstrate that MS-GS achieves photorealistic renderings under various challenging sparse-view and multi-appearance conditions, and outperforms existing approaches significantly across different datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20813v1">GSWorld: Closed-Loop Photo-Realistic Simulation Suite for Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      This paper presents GSWorld, a robust, photo-realistic simulator for robotics manipulation that combines 3D Gaussian Splatting with physics engines. Our framework advocates "closing the loop" of developing manipulation policies with reproducible evaluation of policies learned from real-robot data and sim2real policy training without using real robots. To enable photo-realistic rendering of diverse scenes, we propose a new asset format, which we term GSDF (Gaussian Scene Description File), that infuses Gaussian-on-Mesh representation with robot URDF and other objects. With a streamlined reconstruction pipeline, we curate a database of GSDF that contains 3 robot embodiments for single-arm and bimanual manipulation, as well as more than 40 objects. Combining GSDF with physics engines, we demonstrate several immediate interesting applications: (1) learning zero-shot sim2real pixel-to-action manipulation policy with photo-realistic rendering, (2) automated high-quality DAgger data collection for adapting policies to deployment environments, (3) reproducible benchmarking of real-robot manipulation policies in simulation, (4) simulation data collection by virtual teleoperation, and (5) zero-shot sim2real visual reinforcement learning. Website: https://3dgsworld.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20335v1">Dino-Diffusion Modular Designs Bridge the Cross-Domain Gap in Autonomous Parking</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 Code is at https://github.com/ChampagneAndfragrance/Dino_Diffusion_Parking_Official
    </div>
    <details class="paper-abstract">
      Parking is a critical pillar of driving safety. While recent end-to-end (E2E) approaches have achieved promising in-domain results, robustness under domain shifts (e.g., weather and lighting changes) remains a key challenge. Rather than relying on additional data, in this paper, we propose Dino-Diffusion Parking (DDP), a domain-agnostic autonomous parking pipeline that integrates visual foundation models with diffusion-based planning to enable generalized perception and robust motion planning under distribution shifts. We train our pipeline in CARLA at regular setting and transfer it to more adversarial settings in a zero-shot fashion. Our model consistently achieves a parking success rate above 90% across all tested out-of-distribution (OOD) scenarios, with ablation studies confirming that both the network architecture and algorithmic design significantly enhance cross-domain performance over existing baselines. Furthermore, testing in a 3D Gaussian splatting (3DGS) environment reconstructed from a real-world parking lot demonstrates promising sim-to-real transfer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13978v2">Instant Skinned Gaussian Avatars for Web, Mobile and VR Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 Accepted to SUI 2025 Demo Track
    </div>
    <details class="paper-abstract">
      We present Instant Skinned Gaussian Avatars, a real-time and cross-platform 3D avatar system. Many approaches have been proposed to animate Gaussian Splatting, but they often require camera arrays, long preprocessing times, or high-end GPUs. Some methods attempt to convert Gaussian Splatting into mesh-based representations, achieving lightweight performance but sacrificing visual fidelity. In contrast, our system efficiently animates Gaussian Splatting by leveraging parallel splat-wise processing to dynamically follow the underlying skinned mesh in real time while preserving high visual fidelity. From smartphone-based 3D scanning to on-device preprocessing, the entire process takes just around five minutes, with the avatar generation step itself completed in only about 30 seconds. Our system enables users to instantly transform their real-world appearance into a 3D avatar, making it ideal for seamless integration with social media and metaverse applications. Website: https://gaussian-vrm.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20238v1">COS3D: Collaborative Open-Vocabulary 3D Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 NeurIPS 2025. The code is publicly available at \href{https://github.com/Runsong123/COS3D}{https://github.com/Runsong123/COS3D}
    </div>
    <details class="paper-abstract">
      Open-vocabulary 3D segmentation is a fundamental yet challenging task, requiring a mutual understanding of both segmentation and language. However, existing Gaussian-splatting-based methods rely either on a single 3D language field, leading to inferior segmentation, or on pre-computed class-agnostic segmentations, suffering from error accumulation. To address these limitations, we present COS3D, a new collaborative prompt-segmentation framework that contributes to effectively integrating complementary language and segmentation cues throughout its entire pipeline. We first introduce the new concept of collaborative field, comprising an instance field and a language field, as the cornerstone for collaboration. During training, to effectively construct the collaborative field, our key idea is to capture the intrinsic relationship between the instance field and language field, through a novel instance-to-language feature mapping and designing an efficient two-stage training strategy. During inference, to bridge distinct characteristics of the two fields, we further design an adaptive language-to-instance prompt refinement, promoting high-quality prompt-segmentation inference. Extensive experiments not only demonstrate COS3D's leading performance over existing methods on two widely-used benchmarks but also show its high potential to various applications,~\ie, novel image-based 3D segmentation, hierarchical segmentation, and robotics. The code is publicly available at \href{https://github.com/Runsong123/COS3D}{https://github.com/Runsong123/COS3D}.
    </details>
</div>
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01978v2">ROI-GS: Interest-based Local Quality 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 4 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      We tackle the challenge of efficiently reconstructing 3D scenes with high detail on objects of interest. Existing 3D Gaussian Splatting (3DGS) methods allocate resources uniformly across the scene, limiting fine detail to Regions Of Interest (ROIs) and leading to inflated model size. We propose ROI-GS, an object-aware framework that enhances local details through object-guided camera selection, targeted Object training, and seamless integration of high-fidelity object of interest reconstructions into the global scene. Our method prioritizes higher resolution details on chosen objects while maintaining real-time performance. Experiments show that ROI-GS significantly improves local quality (up to 2.96 dB PSNR), while reducing overall model size by $\approx 17\%$ of baseline and achieving faster training for a scene with a single object of interest, outperforming existing methods.
    </details>
</div>
