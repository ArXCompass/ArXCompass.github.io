# gaussian splatting - 2026_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2401.03890v9">A Survey on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Accepted by ACM Computing Surveys; Paper list: https://github.com/guikunchen/Awesome3DGS ; Benchmark: https://github.com/guikunchen/3DGS-Benchmarks
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (GS) has emerged as a transformative technique in radiance fields. Unlike mainstream implicit neural models, 3D GS uses millions of learnable 3D Gaussians for an explicit scene representation. Paired with a differentiable rendering algorithm, this approach achieves real-time rendering and unprecedented editability, making it a potential game-changer for 3D reconstruction and representation. In the present paper, we provide the first systematic overview of the recent developments and critical contributions in 3D GS. We begin with a detailed exploration of the underlying principles and the driving forces behind the emergence of 3D GS, laying the groundwork for understanding its significance. A focal point of our discussion is the practical applicability of 3D GS. By enabling unprecedented rendering speed, 3D GS opens up a plethora of applications, ranging from virtual reality to interactive media and beyond. This is complemented by a comparative analysis of leading 3D GS models, evaluated across various benchmark tasks to highlight their performance and practical utility. The survey concludes by identifying current challenges and suggesting potential avenues for future research. Through this survey, we aim to provide a valuable resource for both newcomers and seasoned researchers, fostering further exploration and advancement in explicit radiance field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04929v4">CryoSplat: Gaussian Splatting for Cryo-EM Homogeneous Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Accepted at ICLR 2026 (Camera-ready). Code available at https://github.com/Chen-Suyi/cryosplat
    </div>
    <details class="paper-abstract">
      As a critical modality for structural biology, cryogenic electron microscopy (cryo-EM) facilitates the determination of macromolecular structures at near-atomic resolution. The core computational task in single-particle cryo-EM is to reconstruct the 3D electrostatic potential of a molecule from noisy 2D projections acquired at unknown orientations. Gaussian mixture models (GMMs) provide a continuous, compact, and physically interpretable representation for molecular density and have recently gained interest in cryo-EM reconstruction. However, existing methods rely on external consensus maps or atomic models for initialization, limiting their use in self-contained pipelines. In parallel, differentiable rendering techniques such as Gaussian splatting have demonstrated remarkable scalability and efficiency for volumetric representations, suggesting a natural fit for GMM-based cryo-EM reconstruction. However, off-the-shelf Gaussian splatting methods are designed for photorealistic view synthesis and remain incompatible with cryo-EM due to mismatches in the image formation physics, reconstruction objectives, and coordinate systems. Addressing these issues, we propose cryoSplat, a GMM-based method that integrates Gaussian splatting with the physics of cryo-EM image formation. In particular, we develop an orthogonal projection-aware Gaussian splatting, with adaptations such as a view-dependent normalization term and FFT-aligned coordinate system tailored for cryo-EM imaging. These innovations enable stable and efficient homogeneous reconstruction directly from raw cryo-EM particle images using random initialization. Experimental results on real datasets validate the effectiveness and robustness of cryoSplat over representative baselines. The code will be released at https://github.com/Chen-Suyi/cryosplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.09640v2">Physically Plausible Human-Object Rendering from Sparse Views via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 16 pages, 14 figures, accepted by IEEE Transactions on Image Processing (TIP)
    </div>
    <details class="paper-abstract">
      Rendering realistic human-object interactions (HOIs) from sparse-view inputs is a challenging yet crucial task for various real-world applications. Existing methods often struggle to simultaneously achieve high rendering quality, physical plausibility, and computational efficiency. To address these limitations, we propose HOGS (Human-Object Rendering via 3D Gaussian Splatting), a novel framework for efficient HOI rendering with physically plausible geometric constraints from sparse views. HOGS represents both humans and objects as dynamic 3D Gaussians. Central to HOGS is a novel optimization process that operates directly on these Gaussians to enforce geometric consistency (i.e., preventing inter-penetration or floating contacts) to achieve physical plausibility. To support this core optimization under sparse-view ambiguity, our framework incorporates two pre-trained modules: an optimization-guided Human Pose Refiner for robust estimation under sparse-view occlusions, and a Human-Object Contact Predictor that efficiently identifies interaction regions to guide our novel contact and separation losses. Extensive experiments on both human-object and hand-object interaction datasets demonstrate that HOGS achieves state-of-the-art rendering quality and maintains high computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07728v1">GEAR: GEometry-motion Alternating Refinement for Articulated Object Modeling with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Accepted to CVPRF2026
    </div>
    <details class="paper-abstract">
      High-fidelity interactive digital assets are essential for embodied intelligence and robotic interaction, yet articulated objects remain challenging to reconstruct due to their complex structures and coupled geometry-motion relationships. Existing methods suffer from instability in geometry-motion joint optimization, while their generalization remains limited on complex multi-joint or out-of-distribution objects. To address these challenges, we propose GEAR, an EM-style alternating optimization framework that jointly models geometry and motion as interdependent components within a Gaussian Splatting representation. GEAR treats part segmentation as a latent variable and joint motion parameters as explicit variables, alternately refining them for improved convergence and geometric-motion consistency. To enhance part segmentation quality without sacrificing generalization, we leverage a vanilla 2D segmentation model to provide multi-view part priors, and employ a weakly supervised constraint to regularize the latent variable. Experiments on multiple benchmarks and our newly constructed dataset GEAR-Multi demonstrate that GEAR achieves state-of-the-art results in geometric reconstruction and motion parameters estimation, particularly on complex articulated objects with multiple movable parts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.17212v2">Part$^{2}$GS: Part-aware Modeling of Articulated Objects using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Articulated objects are common in the real world, yet modeling their structure and motion remains a challenging task for 3D reconstruction methods. In this work, we introduce Part$^{2}$GS, a novel framework for modeling articulated digital twins of multi-part objects with high-fidelity geometry and physically consistent articulation. Part$^{2}$GS leverages a part-aware 3D Gaussian representation that encodes articulated components with learnable attributes, enabling structured, disentangled transformations that preserve high-fidelity geometry. To ensure physically consistent motion, we propose a motion-aware canonical representation guided by physics-based constraints, including contact enforcement, velocity consistency, and vector-field alignment. Furthermore, we introduce a field of repel points to prevent part collisions and maintain stable articulation paths, significantly improving motion coherence over baselines. Extensive evaluations on both synthetic and real-world datasets show that Part$^{2}$GS consistently outperforms state-of-the-art methods by up to 10$\times$ in Chamfer Distance for movable parts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08760v1">SIC3D: Style Image Conditioned Text-to-3D Gaussian Splatting Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Recent progress in text-to-3D object generation enables the synthesis of detailed geometry from text input by leveraging 2D diffusion models and differentiable 3D representations. However, the approaches often suffer from limited controllability and texture ambiguity due to the limitation of the text modality. To address this, we present SIC3D, a controllable image-conditioned text-to-3D generation pipeline with 3D Gaussian Splatting (3DGS). There are two stages in SIC3D. The first stage generates the 3D object content from text with a text-to-3DGS generation model. The second stage transfers style from a reference image to the 3DGS. Within this stylization stage, we introduce a novel Variational Stylized Score Distillation (VSSD) loss to effectively capture both global and local texture patterns while mitigating conflicts between geometry and appearance. A scaling regularization is further applied to prevent the emergence of artifacts and preserve the pattern from the style image. Extensive experiments demonstrate that SIC3D enhances geometric fidelity and style adherence, outperforming prior approaches in both qualitative and quantitative evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01204v2">Neural Harmonic Textures for High-Quality Primitive Based Neural Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Primitive-based methods such as 3D Gaussian Splatting have recently become the state-of-the-art for novel-view synthesis and related reconstruction tasks. Compared to neural fields, these representations are more flexible, adaptive, and scale better to large scenes. However, the limited expressivity of individual primitives makes modeling high-frequency detail challenging. We introduce Neural Harmonic Textures, a neural representation approach that anchors latent feature vectors on a virtual scaffold surrounding each primitive. These features are interpolated within the primitive at ray intersection points. Inspired by Fourier analysis, we apply periodic activations to the interpolated features, turning alpha blending into a weighted sum of harmonic components. The resulting signal is then decoded in a single deferred pass using a small neural network, significantly reducing computational cost. Neural Harmonic Textures yield state-of-the-art results in real-time novel view synthesis while bridging the gap between primitive- and neural-field-based reconstruction. Our method integrates seamlessly into existing primitive-based pipelines such as 3DGUT, Triangle Splatting, and 2DGS. We further demonstrate its generality with applications to 2D image fitting and semantic reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07337v1">From Blobs to Spokes: High-Fidelity Surface Reconstruction via Oriented Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Our project page is available in http://diego1401.github.io/BlobsToSpokesWebsite/index.html
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has revolutionized fast novel view synthesis, yet its opacity-based formulation makes surface extraction fundamentally difficult. Unlike implicit methods built on Signed Distance Fields or occupancy, 3DGS lacks a global geometric field, forcing existing approaches to resort to heuristics such as TSDF fusion of blended depth maps. Inspired by the Objects as Volumes framework, we derive a principled occupancy field for Gaussian Splatting and show how it can be used to extract highly accurate watertight meshes of complex scenes. Our key contribution is to introduce a learnable oriented normal at each Gaussian element and to define an adapted attenuation formulation, which leads to closed-form expressions for both the normal and occupancy fields at arbitrary locations in space. We further introduce a novel consistency loss and a dedicated densification strategy to enforce Gaussians to wrap the entire surface by closing geometric holes, ensuring a complete shell of oriented primitives. We modify the differentiable rasterizer to output depth as an isosurface of our continuous model, and introduce Primal Adaptive Meshing for Region-of-Interest meshing at arbitrary resolution. We additionally expose fundamental biases in standard surface evaluation protocols and propose two more rigorous alternatives. Overall, our method Gaussian Wrapping sets a new state-of-the-art on DTU and Tanks and Temples, producing complete, watertight meshes at a fraction of the size of concurrent work-recovering thin structures such as the notoriously elusive bicycle spokes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18525v2">Splatblox: Traversability-Aware Gaussian Splatting for Outdoor Robot Navigation</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      We present Splatblox, a real-time system for autonomous navigation in outdoor environments with dense vegetation, irregular obstacles, and complex terrain. Our method fuses segmented RGB images and LiDAR point clouds using Gaussian Splatting to construct a traversability-aware Euclidean Signed Distance Field (ESDF) that jointly encodes geometry and semantics. Updated online, this field enables semantic reasoning to distinguish traversable vegetation (e.g., tall grass) from rigid obstacles (e.g., trees), while LiDAR ensures 360-degree geometric coverage for extended planning horizons. We validate Splatblox on a quadruped robot and demonstrate transfer to a wheeled platform. In field trials across vegetation-rich scenarios, it outperforms state-of-the-art methods with over 50% higher success rate, 40% fewer freezing incidents, 5% shorter paths, and up to 13% faster time to goal, while supporting long-range missions up to 100 meters. Experiment videos and more details can be found on our project page: https://splatblox.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21105v3">BrepGaussian: CAD reconstruction from Multi-View Images with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted to CVPR 2026
    </div>
    <details class="paper-abstract">
      The boundary representation (B-Rep) models a 3D solid as its explicit boundaries: trimmed corners, edges, and faces. Recovering B-Rep representation from unstructured data is a challenging and valuable task of computer vision and graphics. Recent advances in deep learning have greatly improved the recovery of 3D shape geometry, but still depend on dense and clean point clouds and struggle to generalize to novel shapes. We propose B-Rep Gaussian Splatting (BrepGaussian), a novel framework that learns 3D parametric representations from 2D images. We employ a Gaussian Splatting renderer with learnable features, followed by a specific fitting strategy. To disentangle geometry reconstruction and feature learning, we introduce a two-stage learning framework that first captures geometry and edges and then refines patch features to achieve clean geometry and coherent instance representations. Extensive experiments demonstrate the superior performance of our approach to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07177v1">Splats under Pressure: Exploring Performance-Energy Trade-offs in Real-Time 3D Gaussian Splatting under Constrained GPU Budgets</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      We investigate the feasibility of real-time 3D Gaussian Splatting (3DGS) rasterisation on edge clients with varying Gaussian splat counts and GPU computational budgets. Instead of evaluating multiple physical devices, we adopt an emulation-based approach that approximates different GPU capability tiers on a single high-end GPU. By systematically under-clocking the GPU core frequency and applying power caps, we emulate a controlled range of floating-point performance levels that approximate different GPU capability tiers. At each point in this range, we measure frame rate, runtime behaviour, and power consumption across scenes of varying complexity, pipelines, and optimisations, enabling analysis of power-performance relationships such as FPS-power curves, energy per frame, and performance per watt. This method allows us to approximate the performance envelope of a diverse class of GPUs, from embedded and mobile-class devices to high-end consumer-grade systems. Our objective is to explore the practical lower bounds of client-side 3DGS rasterisation and assess its potential for deployment in energy-constrained environments, including standalone headsets and thin clients. Through this analysis, we provide early insights into the performance-energy trade-offs that govern the viability of edge-deployed 3DGS systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07105v1">Genie Sim PanoRecon: Fast Immersive Scene Generation from Single-View Panorama</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      We present Genie Sim PanoRecon, a feed-forward Gaussian-splatting pipeline that delivers high-fidelity, low-cost 3D scenes for robotic manipulation simulation. The panorama input is decomposed into six non-overlapping cube-map faces, processed in parallel, and seamlessly reassembled. To guarantee geometric consistency across views, we devise a depth-aware fusion strategy coupled with a training-free depth-injection module that steers the monocular feed-forward network to generate coherent 3D Gaussians. The whole system reconstructs photo-realistic scenes in seconds and has been integrated into Genie Sim - a LLM-driven simulation platform for embodied synthetic data generation and evaluation - to provide scalable backgrounds for manipulation tasks. For code details, please refer to: https://github.com/AgibotTech/genie_sim/tree/main/source/geniesim_world.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06739v1">DOC-GS: Dual-Domain Observation and Calibration for Reliable Sparse-View Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Sparse-view reconstruction with 3D Gaussian Splatting (3DGS) is fundamentally ill-posed due to insufficient geometric supervision, often leading to severe overfitting and the emergence of structural distortions and translucent haze-like artifacts. While existing approaches attempt to alleviate this issue via dropout-based regularization, they are largely heuristic and lack a unified understanding of artifact formation. In this paper, we revisit sparse-view 3DGS reconstruction from a new perspective and identify the core challenge as the unobservability of Gaussian primitive reliability. Unreliable Gaussians are insufficiently constrained during optimization and accumulate as haze-like degradations in rendered images. Motivated by this observation, we propose a unified Dual-domain Observation and Calibration (DOC-GS) framework that models and corrects Gaussian reliability through the synergy of optimization-domain inductive bias and observation-domain evidence. Specifically, in the optimization domain, we characterize Gaussian reliability by the degree to which each primitive is constrained during training, and instantiate this signal via a Continuous Depth-Guided Dropout (CDGD) strategy, where the dropout probability serves as an explicit proxy for primitive reliability. This imposes a smooth depth-aware inductive bias to suppress weakly constrained Gaussians and improve optimization stability. In the observation domain, we establish a connection between floater artifacts and atmospheric scattering, and leverage the Dark Channel Prior (DCP) as a structural consistency cue to identify and accumulate anomalous regions. Based on cross-view aggregated evidence, we further design a reliability-driven geometric pruning strategy to remove low-confidence Gaussians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06671v1">4D Vessel Reconstruction for Benchtop Thrombectomy Analysis</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 20 pages, 10 figures, 1 table, supplementary material (3 tables, 3 figures, and 11 videos). Project page: https://ethanuser.github.io/vessel4D/
    </div>
    <details class="paper-abstract">
      Introduction: Mechanical thrombectomy can cause vessel deformation and procedure-related injury. Benchtop models are widely used for device testing, but time-resolved, full-field 3D vessel-motion measurements remain limited. Methods: We developed a nine-camera, low-cost multi-view workflow for benchtop thrombectomy in silicone middle cerebral artery phantoms (2160p, 20 fps). Multi-view videos were calibrated, segmented, and reconstructed with 4D Gaussian Splatting. Reconstructed point clouds were converted to fixed-connectivity edge graphs for region-of-interest (ROI) displacement tracking and a relative surface-based stress proxy. Stress-proxy values were derived from edge stretch using a Neo-Hookean mapping and reported as comparative surface metrics. A synthetic Blender pipeline with known deformation provided geometric and temporal validation. Results: In synthetic bulk translation, the stress proxy remained near zero for most edges (median $\approx$ 0 MPa; 90th percentile 0.028 MPa), with sparse outliers. In synthetic pulling (1-5 mm), reconstruction showed close geometric and temporal agreement with ground truth, with symmetric Chamfer distance of 1.714-1.815 mm and precision of 0.964-0.972 at $τ= 1$ mm. In preliminary benchtop comparative trials (one trial per condition), cervical aspiration catheter placement showed higher max-median ROI displacement and stress-proxy values than internal carotid artery terminus placement. Conclusion: The proposed protocol provides standardized, time-resolved surface kinematics and comparative relative displacement and stress proxy measurements for thrombectomy benchtop studies. The framework supports condition-to-condition comparisons and methods validation, while remaining distinct from absolute wall-stress estimation. Implementation code and example data are available at https://ethanuser.github.io/vessel4D
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02996v2">Rendering Multi-Human and Multi-Object with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 8 pages, 4 figures, accepted by ICRA 2026
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic scenes with multiple interacting humans and objects from sparse-view inputs is a critical yet challenging task, essential for creating high-fidelity digital twins for robotics and VR/AR. This problem, which we term Multi-Human Multi-Object (MHMO) rendering, presents two significant obstacles: achieving view-consistent representations for individual instances under severe mutual occlusion, and explicitly modeling the complex and combinatorial dependencies that arise from their interactions. To overcome these challenges, we propose MM-GS, a novel hierarchical framework built upon 3D Gaussian Splatting. Our method first employs a Per-Instance Multi-View Fusion module to establish a robust and consistent representation for each instance by aggregating visual information across all available views. Subsequently, a Scene-Level Instance Interaction module operates on a global scene graph to reason about relationships between all participants, refining their attributes to capture subtle interaction effects. Extensive experiments on challenging datasets demonstrate that our method significantly outperforms strong baselines, producing state-of-the-art results with high-fidelity details and plausible inter-instance contacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06358v1">GS-Surrogate: Deformable Gaussian Splatting for Parameter Space Exploration of Ensemble Simulations</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Exploring ensemble simulations is increasingly important across many scientific domains. However, supporting flexible post-hoc exploration remains challenging due to the trade-off between storing the expensive raw data and flexibly adjusting visualization settings. Existing visualization surrogate models have improved this workflow, but they either operate in image space without an explicit 3D representation or rely on neural radiance fields that are computationally expensive for interactive exploration and encode all parameter-driven variations within a single implicit field. In this work, we introduce GS-Surrogate, a deformable Gaussian Splatting-based visualization surrogate for parameter-space exploration. Our method first constructs a canonical Gaussian field as a base 3D representation and adapts it through sequential parameter-conditioned deformations. By separating simulation-related variations from visualization-specific changes, this explicit formulation enables efficient and controllable adaptation to different visualization tasks, such as isosurface extraction and transfer function editing. We evaluate our framework on a range of simulation datasets, demonstrating that GS-Surrogate enables real-time and flexible exploration across both simulation and visualization parameter spaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19172v3">MetroGS: Efficient and Stable Reconstruction of Geometrically Accurate High-Fidelity Large-Scale Scenes</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Project page: https://m3phist0.github.io/MetroGS
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting and its derivatives have achieved significant breakthroughs in large-scale scene reconstruction. However, how to efficiently and stably achieve high-quality geometric fidelity remains a core challenge. To address this issue, we introduce MetroGS, a novel Gaussian Splatting framework for efficient and robust reconstruction in complex urban environments. Our method is built upon a distributed 2D Gaussian Splatting representation as the core foundation, serving as a unified backbone for subsequent modules. To handle potential sparse regions in complex scenes, we propose a structured dense enhancement scheme that utilizes SfM priors and a pointmap model to achieve a denser initialization, while incorporating a sparsity compensation mechanism to improve reconstruction completeness. Furthermore, we design a progressive hybrid geometric optimization strategy that organically integrates monocular and multi-view optimization to achieve efficient and accurate geometric refinement. Finally, to address the appearance inconsistency commonly observed in large-scale scenes, we introduce a depth-guided appearance modeling approach that learns spatial features with 3D consistency, facilitating effective decoupling between geometry and appearance and further enhancing reconstruction stability. Experiments on large-scale urban datasets demonstrate that MetroGS achieves superior geometric accuracy, rendering quality, offering a unified solution for high-fidelity large-scale scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05908v1">Appearance Decomposition Gaussian Splatting for Multi-Traversal Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Multi-traversal scene reconstruction is important for high-fidelity autonomous driving simulation and digital twin construction. This task involves integrating multiple sequences captured from the same geographical area at different times. In this context, a primary challenge is the significant appearance inconsistency across traversals caused by varying illumination and environmental conditions, despite the shared underlying geometry. This paper presents ADM-GS (Appearance Decomposition Gaussian Splatting for Multi-Traversal Reconstruction), a framework that applies an explicit appearance decomposition to the static background to alleviate appearance entanglement across traversals. For the static background, we decompose the appearance into traversal-invariant material, representing intrinsic material properties, and traversal-dependent illumination, capturing lighting variations. Specifically, we propose a neural light field that utilizes a frequency-separated hybrid encoding strategy. By incorporating surface normals and explicit reflection vectors, this design separately captures low-frequency diffuse illumination and high-frequency specular reflections. Quantitative evaluations on the Argoverse 2 and Waymo Open datasets demonstrate the effectiveness of ADM-GS. In multi-traversal experiments, our method achieves a +0.98 dB PSNR improvement over existing latent-based baselines while producing more consistent appearance across traversals. Code will be available at https://github.com/IRMVLab/ADM-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05721v1">GaussianGrow: Geometry-aware Gaussian Growing from 3D Point Clouds with Text Guidance</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Accepted by CVPR 2026. Project page: https://weiqi-zhang.github.io/GaussianGrow
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has demonstrated superior performance in rendering efficiency and quality, yet the generation of 3D Gaussians still remains a challenge without proper geometric priors. Existing methods have explored predicting point maps as geometric references for inferring Gaussian primitives, while the unreliable estimated geometries may lead to poor generations. In this work, we introduce GaussianGrow, a novel approach that generates 3D Gaussians by learning to grow them from easily accessible 3D point clouds, naturally enforcing geometric accuracy in Gaussian generation. Specifically, we design a text-guided Gaussian growing scheme that leverages a multi-view diffusion model to synthesize consistent appearances from input point clouds for supervision. To mitigate artifacts caused by fusing neighboring views, we constrain novel views generated at non-preset camera poses identified in overlapping regions across different views. For completing the hard-to-observe regions, we propose to iteratively detect the camera pose by observing the largest un-grown regions in point clouds and inpainting them by inpainting the rendered view with a pretrained 2D diffusion model. The process continues until complete Gaussians are generated. We extensively evaluate GaussianGrow on text-guided Gaussian generation from synthetic and even real-scanned point clouds. Project Page: https://weiqi-zhang.github.io/GaussianGrow
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05715v1">In Depth We Trust: Reliable Monocular Depth Supervision for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 accepted to CVPR 3DMV Workshop
    </div>
    <details class="paper-abstract">
      Using accurate depth priors in 3D Gaussian Splatting helps mitigate artifacts caused by sparse training data and textureless surfaces. However, acquiring accurate depth maps requires specialized acquisition systems. Foundation monocular depth estimation models offer a cost-effective alternative, but they suffer from scale ambiguity, multi-view inconsistency, and local geometric inaccuracies, which can degrade rendering performance when applied naively. This paper addresses the challenge of reliably leveraging monocular depth priors for Gaussian Splatting (GS) rendering enhancement. To this end, we introduce a training framework integrating scale-ambiguous and noisy depth priors into geometric supervision. We highlight the importance of learning from weakly aligned depth variations. We introduce a method to isolate ill-posed geometry for selective monocular depth regularization, restricting the propagation of depth inaccuracies into well-reconstructed 3D structures. Extensive experiments across diverse datasets show consistent improvements in geometric accuracy, leading to more faithful depth estimation and higher rendering quality across different GS variants and monocular depth backbones tested.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05687v1">3D Smoke Scene Reconstruction Guided by Vision Priors from Multimodal Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Reconstructing 3D scenes from smoke-degraded multi-view images is particularly difficult because smoke introduces strong scattering effects, view-dependent appearance changes, and severe degradation of cross-view consistency. To address these issues, we propose a framework that integrates visual priors with efficient 3D scene modeling. We employ Nano-Banana-Pro to enhance smoke-degraded images and provide clearer visual observations for reconstruction and develop Smoke-GS, a medium-aware 3D Gaussian Splatting framework for smoke scene reconstruction and restoration-oriented novel view synthesis. Smoke-GS models the scene using explicit 3D Gaussians and introduces a lightweight view-dependent medium branch to capture direction-dependent appearance variations caused by smoke. Our method preserves the rendering efficiency of 3D Gaussian Splatting while improving robustness to smoke-induced degradation. Results demonstrate the effectiveness of our method for generating consistent and visually clear novel views in challenging smoke environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05638v1">PanopticQuery: Unified Query-Time Reasoning for 4D Scenes</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Understanding dynamic 4D environments through natural language queries requires not only accurate scene reconstruction but also robust semantic grounding across space, time, and viewpoints. While recent methods using neural representations have advanced 4D reconstruction, they remain limited in contextual reasoning, especially for complex semantics such as interactions, temporal actions, and spatial relations. A key challenge lies in transforming noisy, view-dependent predictions into globally consistent 4D interpretations. We introduce PanopticQuery, a framework for unified query-time reasoning in 4D scenes. Our approach builds on 4D Gaussian Splatting for high-fidelity dynamic reconstruction and introduces a multi-view semantic consensus mechanism that grounds natural language queries by aggregating 2D semantic predictions across multiple views and time frames. This process filters inconsistent outputs, enforces geometric consistency, and lifts 2D semantics into structured 4D groundings via neural field optimization. To support evaluation, we present Panoptic-L4D, a new benchmark for language-based querying in dynamic scenes. Experiments demonstrate that PanopticQuery sets a new state of the art on complex language queries, effectively handling attributes, actions, spatial relationships, and multi-object interactions. A video demonstration is available in the supplementary materials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05402v1">LSGS-Loc: Towards Robust 3DGS-Based Visual Localization for Large-Scale UAV Scenarios</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 This paper is under reviewed by RA-L. The copyright might be transferred upon acceptance
    </div>
    <details class="paper-abstract">
      Visual localization in large-scale UAV scenarios is a critical capability for autonomous systems, yet it remains challenging due to geometric complexity and environmental variations. While 3D Gaussian Splatting (3DGS) has emerged as a promising scene representation, existing 3DGS-based visual localization methods struggle with robust pose initialization and sensitivity to rendering artifacts in large-scale settings. To address these limitations, we propose LSGS-Loc, a novel visual localization pipeline tailored for large-scale 3DGS scenes. Specifically, we introduce a scale-aware pose initialization strategy that combines scene-agnostic relative pose estimation with explicit 3DGS scale constraints, enabling geometrically grounded localization without scene-specific training. Furthermore, in the pose refinement, to mitigate the impact of reconstruction artifacts such as blur and floaters, we develop a Laplacian-based reliability masking mechanism that guides photometric refinement toward high-quality regions. Extensive experiments on large-scale UAV benchmarks demonstrate that our method achieves state-of-the-art accuracy and robustness for unordered image queries, significantly outperforming existing 3DGS-based approaches. Code is available at: https://github.com/xzhang-z/LSGS-Loc
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05366v1">3DTurboQuant: Training-Free Near-Optimal Quantization for 3D Reconstruction Models</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Every existing method for compressing 3D Gaussian Splatting, NeRF, or transformer-based 3D reconstructors requires learning a data-dependent codebook through per-scene fine-tuning. We show this is unnecessary. The parameter vectors that dominate storage in these models, 45-dimensional spherical harmonics in 3DGS and 1024-dimensional key-value vectors in DUSt3R, fall in a dimension range where a single random rotation transforms any input into coordinates with a known Beta distribution. This makes precomputed, data-independent Lloyd-Max quantization near-optimal, within a factor of 2.7 of the information-theoretic lower bound. We develop 3D, deriving (1) a dimension-dependent criterion that predicts which parameters can be quantized and at what bit-width before running any experiment, (2) norm-separation bounds connecting quantization MSE to rendering PSNR per scene, (3) an entry-grouping strategy extending rotation-based quantization to 2-dimensional hash grid features, and (4) a composable pruning-quantization pipeline with a closed-form compression ratio. On NeRF Synthetic, 3DTurboQuant compresses 3DGS by 3.5x with 0.02dB PSNR loss and DUSt3R KV caches by 7.9x with 39.7dB pointmap fidelity. No training, no codebook learning, no calibration data. Compression takes seconds. The code will be released (https://github.com/JaeLee18/3DTurboQuant)
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.04576v2">PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Accepted at CVPR 2026. Project Page: https://kakaomacao.github.io/pr-iqa-project-page/
    </div>
    <details class="paper-abstract">
      Diffusion models are promising for sparse-view novel view synthesis (NVS), as they can generate pseudo-ground-truth views to aid 3D reconstruction pipelines like 3D Gaussian Splatting (3DGS). However, these synthesized images often contain photometric and geometric inconsistencies, and their direct use for supervision can impair reconstruction. To address this, we propose Partial-Reference Image Quality Assessment (PR-IQA), a framework that evaluates diffusion-generated views using reference images from different poses, eliminating the need for ground truth. PR-IQA first computes a geometrically consistent partial quality map in overlapping regions. It then performs quality completion to inpaint this partial map into a dense, full-image map. This completion is achieved via a cross-attention mechanism that incorporates reference-view context, ensuring cross-view consistency and enabling thorough quality assessment. When integrated into a diffusion-augmented 3DGS pipeline, PR-IQA restricts supervision to high-confidence regions identified by its quality maps. Experiments demonstrate that PR-IQA outperforms existing IQA methods, achieving full-reference-level accuracy without ground-truth supervision. Thus, our quality-aware 3DGS approach more effectively filters inconsistencies, producing superior 3D reconstructions and NVS results. The project page is available at https://kakaomacao.github.io/pr-iqa-project-page/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05316v1">Indoor Asset Detection in Large Scale 360° Drone-Captured Imagery via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Accepted to CVPR 2026 3DMV Workshop
    </div>
    <details class="paper-abstract">
      We present an approach for object-level detection and segmentation of target indoor assets in 3D Gaussian Splatting (3DGS) scenes, reconstructed from 360° drone-captured imagery. We introduce a 3D object codebook that jointly leverages mask semantics and spatial information of their corresponding Gaussian primitives to guide multi-view mask association and indoor asset detection. By integrating 2D object detection and segmentation models with semantically and spatially constrained merging procedures, our method aggregates masks from multiple views into coherent 3D object instances. Experiments on two large indoor scenes demonstrate reliable multi-view mask consistency, improving F1 score by 65% over state-of-the-art baselines, and accurate object-level 3D indoor asset detection, achieving an 11% mAP gain over baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05301v1">SmokeGS-R: Physics-Guided Pseudo-Clean 3DGS for Real-World Multi-View Smoke Restoration</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Lab Report for NTIRE 2026 3DRR Track 2
    </div>
    <details class="paper-abstract">
      Real-world smoke simultaneously attenuates scene radiance, adds airlight, and destabilizes multi-view appearance consistency, making robust 3D reconstruction particularly difficult. We present \textbf{SmokeGS-R}, a practical pipeline developed for the NTIRE 2026 3D Restoration and Reconstruction Track 2 challenge. The key idea is to decouple geometry recovery from appearance correction: we generate physics-guided pseudo-clean supervision with a refined dark channel prior and guided filtering, train a sharp clean-only 3D Gaussian Splatting source model, and then harmonize its renderings with a donor ensemble using geometric-mean reference aggregation, LAB-space Reinhard transfer, and light Gaussian smoothing. On the official challenge testing leaderboard, the final submission achieved \mbox{PSNR $=15.217$} and \mbox{SSIM $=0.666$}. After the public release of RealX3D, we re-evaluated the same frozen result on the seven released challenge scenes without retraining and obtained \mbox{PSNR $=15.209$}, \mbox{SSIM $=0.644$}, and \mbox{LPIPS $=0.551$}, outperforming the strongest official baseline average on the same scenes by $+3.68$ dB PSNR. These results suggest that a geometry-first reconstruction strategy combined with stable post-render appearance harmonization is an effective recipe for real-world multi-view smoke restoration. The code is available at https://github.com/windrise/3drr_Track2_SmokeGS-R.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05062v1">GaussFly: Contrastive Reinforcement Learning for Visuomotor Policies in 3D Gaussian Fields</a></div>
    <div class="paper-meta">
      📅 2026-04-06
    </div>
    <details class="paper-abstract">
      Learning visuomotor policies for Autonomous Aerial Vehicles (AAVs) relying solely on monocular vision is an attractive yet highly challenging paradigm. Existing end-to-end learning approaches directly map high-dimensional RGB observations to action commands, which frequently suffer from low sample efficiency and severe sim-to-real gaps due to the visual discrepancy between simulation and physical domains. To address these long-standing challenges, we propose GaussFly, a novel framework that explicitly decouples representation learning from policy optimization through a cohesive real-to-sim-to-real paradigm. First, to achieve a high-fidelity real-to-sim transition, we reconstruct training scenes using 3D Gaussian Splatting (3DGS) augmented with explicit geometric constraints. Second, to ensure robust sim-to-real transfer, we leverage these photorealistic simulated environments and employ contrastive representation learning to extract compact, noise-resilient latent features from the rendered RGB images. By utilizing this pre-trained encoder to provide low-dimensional feature inputs, the computational burden on the visuomotor policy is significantly reduced while its resistance against visual noise is inherently enhanced. Extensive experiments in simulated and real-world environments demonstrate that GaussFly achieves superior sample efficiency and asymptotic performance compared to baselines. Crucially, it enables robust and zero-shot policy transfer to unseen real-world environments with complex textures, effectively bridging the sim-to-real gap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16736v4">A Step to Decouple Optimization in 3DGS</a></div>
    <div class="paper-meta">
      📅 2026-04-06
      | 💬 Accepted by ICLR 2026 (fixed typo)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for real-time novel view synthesis. As an explicit representation optimized through gradient propagation among primitives, optimization widely accepted in deep neural networks (DNNs) is actually adopted in 3DGS, such as synchronous weight updating and Adam with the adaptive gradient. However, considering the physical significance and specific design in 3DGS, there are two overlooked details in the optimization of 3DGS: (i) update step coupling, which induces optimizer state rescaling and costly attribute updates outside the viewpoints, and (ii) gradient coupling in the moment, which may lead to under- or over-effective regularization. Nevertheless, such a complex coupling is under-explored. After revisiting the optimization of 3DGS, we take a step to decouple it and recompose the process into: Sparse Adam, Re-State Regularization and Decoupled Attribute Regularization. Taking a large number of experiments under the 3DGS and 3DGS-MCMC frameworks, our work provides a deeper understanding of these components. Finally, based on the empirical analysis, we re-design the optimization and propose AdamW-GS by re-coupling the beneficial components, under which better optimization efficiency and representation effectiveness are achieved simultaneously.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.04787v1">AvatarPointillist: AutoRegressive 4D Gaussian Avatarization</a></div>
    <div class="paper-meta">
      📅 2026-04-06
      | 💬 Accepted by the CVPR 2026 main conference. Project page: https://kumapowerliu.github.io/AvatarPointillist/
    </div>
    <details class="paper-abstract">
      We introduce AvatarPointillist, a novel framework for generating dynamic 4D Gaussian avatars from a single portrait image. At the core of our method is a decoder-only Transformer that autoregressively generates a point cloud for 3D Gaussian Splatting. This sequential approach allows for precise, adaptive construction, dynamically adjusting point density and the total number of points based on the subject's complexity. During point generation, the AR model also jointly predicts per-point binding information, enabling realistic animation. After generation, a dedicated Gaussian decoder converts the points into complete, renderable Gaussian attributes. We demonstrate that conditioning the decoder on the latent features from the AR generator enables effective interaction between stages and markedly improves fidelity. Extensive experiments validate that AvatarPointillist produces high-quality, photorealistic, and controllable avatars. We believe this autoregressive formulation represents a new paradigm for avatar generation, and we will release our code inspire future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.04693v1">3D Gaussian Splatting for Annular Dark Field Scanning Transmission Electron Microscopy Tomography Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-06
    </div>
    <details class="paper-abstract">
      Analytical Dark Field Scanning Transmission Electron Microscopy (ADF-STEM) tomography reconstructs nanoscale materials in 3D by integrating multi-view tilt-series images, enabling precise analysis of their structural and compositional features. Although integrating more tilt views improves 3D reconstruction, it requires extended electron exposure that risks damaging dose-sensitive materials and introduces drift and misalignment, making it difficult to balance reconstruction fidelity with sample preservation. In practice, sparse-view acquisition is frequently required, yet conventional ADF-STEM methods degrade under limited views, exhibiting artifacts and reduced structural fidelity. To resolve these issues, in this paper, we adapt 3D GS to this domain with three key components. We first model the local scattering strength as a learnable scalar field, denza, to address the mismatch between 3DGS and ADF-STEM imaging physics. Then we introduce a coefficient $γ$ to stabilize scattering across tilt angles, ensuring consistent denza via scattering view normalization. Finally, We incorporate a loss function that includes a 2D Fourier amplitude term to suppress missing wedge artifacts in sparse-view reconstruction. Experiments on 45-view and 15-view tilt series show that DenZa-Gaussian produces high-fidelity reconstructions and 2D projections that align more closely with original tilts, demonstrating superior robustness under sparse-view conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16806v3">MedGS: Gaussian Splatting for Multi-Modal 3D Medical Imaging</a></div>
    <div class="paper-meta">
      📅 2026-04-06
    </div>
    <details class="paper-abstract">
      Endoluminal endoscopic procedures are essential for diagnosing colorectal cancer and other severe conditions in the digestive tract, urogenital system, and airways. 3D reconstruction and novel-view synthesis from endoscopic images are promising tools for enhancing diagnosis. Moreover, integrating physiological deformations and interaction with the endoscope enables the development of simulation tools from real video data. However, constrained camera trajectories and view-dependent lighting create artifacts, leading to inaccurate or overfitted reconstructions. We present MedGS, a novel 3D reconstruction framework leveraging the unique property of endoscopic imaging, where a single light source is closely aligned with the camera. Our method separates light effects from tissue properties. MedGS enhances 3D Gaussian Splatting with a physically based relightable model. We boost the traditional light transport formulation with a specialized MLP capturing complex light-related effects while ensuring reduced artifacts and better generalization across novel views. MedGS achieves superior reconstruction quality compared to baseline methods on both public and in-house datasets. Unlike existing approaches, MedGS enables tissue modifications while preserving a physically accurate response to light, making it closer to real-world clinical use. Repository: https://github.com/gmum/MedGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02120v2">GEMM-GS: Accelerating 3D Gaussian Splatting on Tensor Cores with GEMM-Compatible Blending</a></div>
    <div class="paper-meta">
      📅 2026-04-06
      | 💬 Accepted by the 63rd Design Automation Conference (DAC 2026)
    </div>
    <details class="paper-abstract">
      Neural Radiance Fields (NeRF) enables 3D scene reconstruction from several 2D images but incurs high rendering latency via its point-sampling design. 3D Gaussian Splatting (3DGS) improves on NeRF with explicit scene representation and an optimized pipeline yet still fails to meet practical real-time demands. Existing acceleration works overlook the evolving Tensor Cores of modern GPUs because 3DGS pipeline lacks General Matrix Multiplication (GEMM) operations. This paper proposes GEMM-GS, an acceleration approach utilizing tensor cores on GPUs via GEMM-friendly blending transformation. It equivalently reformulates the 3DGS blending process into a GEMM-compatible form to utilize Tensor Cores. A high-performance CUDA kernel is designed, integrating a three-stage double-buffered pipeline that overlaps computation and memory access. Extensive experiments show that GEMM-GS achieves $1.42\times$ speedup over vanilla 3DGS and provides an additional $1.47\times$ speedup on average when combining with existing acceleration approaches. Code is released at https://github.com/shieldforever/GEMM-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.02794v3">PhysGaia: A Physics-Aware Benchmark with Multi-Body Interactions for Dynamic Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-04-06
      | 💬 Accepted at CVPR 2026 Project page: http://cvlab.snu.ac.kr/research/PhysGaia Dataset: https://huggingface.co/datasets/mijeongkim/PhysGaia/tree/main
    </div>
    <details class="paper-abstract">
      We introduce PhysGaia, a novel physics-aware benchmark for Dynamic Novel View Synthesis (DyNVS) that encompasses both structured objects and unstructured physical phenomena. While existing datasets primarily focus on photorealistic appearance, PhysGaia is specifically designed to support physics-consistent dynamic reconstruction. Our benchmark features complex scenarios with rich multi-body interactions, where objects realistically collide and exchange forces. Furthermore, it incorporates a diverse range of materials, including liquid, gas, textile, and rheological substance, moving beyond the rigid-body assumptions prevalent in prior work. To ensure physical fidelity, all scenes in PhysGaia are generated using material-specific physics solvers that strictly adhere to fundamental physical laws. We provide comprehensive ground-truth information, including 3D particle trajectories and physical parameters (e.g., viscosity), enabling the quantitative evaluation of physical modeling. To facilitate research adoption, we also provide integration pipelines for recent 4D Gaussian Splatting models along with our dataset and their results. By addressing the critical shortage of physics-aware benchmarks, PhysGaia can significantly advance research in dynamic view synthesis, physics-based scene understanding, and the integration of deep learning with physical simulation, ultimately enabling more faithful reconstruction and interpretation of complex dynamic scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.07435v2">DreamLifting: A Plug-in Module Lifting MV Diffusion Models for 3D Asset Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-06
      | 💬 16 pages, 9 figures, TVCG 2026, project page: https://zx-yin.github.io/dreamlifting/
    </div>
    <details class="paper-abstract">
      The labor- and experience-intensive creation of 3D assets with physically based rendering (PBR) materials demands an autonomous 3D asset creation pipeline. However, most existing 3D generation methods focus on geometry modeling, either baking textures into simple vertex colors or leaving texture synthesis to post-processing with image diffusion models. To achieve end-to-end PBR-ready 3D asset generation, we present Lightweight Gaussian Asset Adapter (LGAA), a novel framework that unifies the modeling of geometry and PBR materials by exploiting multi-view (MV) diffusion priors from a novel perspective. The LGAA features a modular design with three components. Specifically, the LGAA Wrapper reuses and adapts network layers from MV diffusion models, which encapsulate knowledge acquired from billions of images, enabling better convergence in a data-efficient manner. To incorporate multiple diffusion priors for geometry and PBR synthesis, the LGAA Switcher aligns multiple LGAA Wrapper layers encapsulating different knowledge. Then, a tamed variational autoencoder (VAE), termed LGAA Decoder, is designed to predict 2D Gaussian Splatting (2DGS) with PBR channels. Finally, we introduce a dedicated post-processing procedure to effectively extract high-quality, relightable mesh assets from the resulting 2DGS. Extensive quantitative and qualitative experiments demonstrate the superior performance of LGAA with both text- and image-conditioned MV diffusion models. Additionally, the modular design enables flexible incorporation of multiple diffusion priors, and the knowledge-preserving scheme effectively preseves the 2D priors learned on massive image dataset, which leads to data efficient finetuning to lift the MV diffuison models for 3D generation with merely 69k multi-view instances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.04331v1">GA-GS: Generation-Assisted Gaussian Splatting for Static Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-06
    </div>
    <details class="paper-abstract">
      Reconstructing static 3D scene from monocular video with dynamic objects is important for numerous applications such as virtual reality and autonomous driving. Current approaches typically rely on background for static scene reconstruction, limiting the ability to recover regions occluded by dynamic objects. In this paper, we propose GA-GS, a Generation-Assisted Gaussian Splatting method for Static Scene Reconstruction. The key innovation of our work lies in leveraging generation to assist in reconstructing occluded regions. We employ a motion-aware module to segment and remove dynamic regions, and thenuse a diffusion model to inpaint the occluded areas, providing pseudo-ground-truth supervision. To balance contributions from real background and generated region, we introduce a learnable authenticity scalar for each Gaussian primitive, which dynamically modulates opacity during splatting for authenticity-aware rendering and supervision. Since no existing dataset provides ground-truth static scene of video with dynamic objects, we construct a dataset named Trajectory-Match, using a fixed-path robot to record each scene with/without dynamic objects, enabling quantitative evaluation in reconstruction of occluded regions. Extensive experiments on both the DAVIS and our dataset show that GA-GS achieves state-of-the-art performance in static scene reconstruction, especially in challenging scenarios with large-scale, persistent occlusions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17541v2">FLEG: Feed-Forward Language Embedded Gaussian Splatting from Any Views via Compact Semantic Representation</a></div>
    <div class="paper-meta">
      📅 2026-04-05
      | 💬 Project page: https://fangzhou2000.github.io/projects/fleg
    </div>
    <details class="paper-abstract">
      We present FLEG, a feed-forward network that reconstructs language-embedded 3D Gaussians from arbitrary views. Previous feed-forward language-embedded Gaussian reconstruction methods are restricted to a fixed number of input views and typically attach a language-aligned semantic embedding to each Gaussian, resulting in impractical input settings and semantic redundancy. In contrast, we introduce a geometry-semantic dual-branch distillation framework that enables flexible input from arbitrary multi-view images without camera parameters. We also propose a novel-view-based distillation strategy during training that mitigates overfitting to input views. In addition, we observe that semantic representations are significantly sparser than geometric ones, and per-Gaussian language embedding is unnecessary. To exploit this sparsity, we design a decoupled language embedding strategy that represents language information with a sparse set of semantic Gaussians, rather than attaching embeddings to every Gaussian. Compared with dense pixel-aligned per-Gaussian embedding schemes, our method uses only 5\% of the language embeddings while maintaining comparable semantic fidelity, effectively reducing storage costs. Extensive experiments demonstrate that FLEG outperforms state-of-the-art feed-forward reconstruction and language-embedded Gaussian methods in both reconstruction quality and language-aligned semantic representation. Project page: https://fangzhou2000.github.io/projects/fleg.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.04063v1">4C4D: 4 Camera 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-05
      | 💬 Accepted by CVPR 2026. Project page: https://junshengzhou.github.io/4C4D
    </div>
    <details class="paper-abstract">
      This paper tackles the challenge of recovering 4D dynamic scenes from videos captured by as few as four portable cameras. Learning to model scene dynamics for temporally consistent novel-view rendering is a foundational task in computer graphics, where previous works often require dense multi-view captures using camera arrays of dozens or even hundreds of views. We propose \textbf{4C4D}, a novel framework that enables high-fidelity 4D Gaussian Splatting from video captures of extremely sparse cameras. Our key insight lies that the geometric learning under sparse settings is substantially more difficult than modeling appearance. Driven by this observation, we introduce a Neural Decaying Function on Gaussian opacities for enhancing the geometric modeling capability of 4D Gaussians. This design mitigates the inherent imbalance between geometry and appearance modeling in 4DGS by encouraging the 4DGS gradients to focus more on geometric learning. Extensive experiments across sparse-view datasets with varying camera overlaps show that 4C4D achieves superior performance over prior art. Project page at: https://junshengzhou.github.io/4C4D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.04016v1">HOIGS: Human-Object Interaction Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-05
      | 💬 24 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic scenes with complex human-object interactions is a fundamental challenge in computer vision and graphics. Existing Gaussian Splatting methods either rely on human pose priors while neglecting dynamic objects, or approximate all motions within a single field, limiting their ability to capture interaction-rich dynamics. To address this gap, we propose Human-Object Interaction Gaussian Splatting (HOIGS), which explicitly models interaction-induced deformation between humans and objects through a cross-attention-based HOI module. Distinct deformation baselines are employed to extract features: HexPlane for humans and Cubic Hermite Spline (CHS) for objects. By integrating these heterogeneous features, HOIGS effectively captures interdependent motions and improves deformation estimation in scenarios involving occlusion, contact, and object manipulation. Comprehensive experiments on multiple datasets demonstrate that our method consistently outperforms state-of-the-art human-centric and 4D Gaussian approaches, highlighting the importance of explicitly modeling human-object interactions for high-fidelity reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22262v2">Can Protective Watermarking Safeguard the Copyright of 3D Gaussian Splatting?</a></div>
    <div class="paper-meta">
      📅 2026-04-05
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful representation for 3D scenes, widely adopted due to its exceptional efficiency and high-fidelity visual quality. Given the significant value of 3DGS assets, recent works have introduced specialized watermarking schemes to ensure copyright protection and ownership verification. However, can existing 3D Gaussian watermarking approaches genuinely guarantee robust protection of the 3D assets? In this paper, for the first time, we systematically explore and validate possible vulnerabilities of 3DGS watermarking frameworks. We demonstrate that conventional watermark removal techniques designed for 2D images do not effectively generalize to the 3DGS scenario due to the specialized rendering pipeline and unique attributes of each gaussian primitives. Motivated by this insight, we propose GSPure, the first watermark purification framework specifically for 3DGS watermarking representations. By analyzing view-dependent rendering contributions and exploiting geometrically accurate feature clustering, GSPure precisely isolates and effectively removes watermark-related Gaussian primitives while preserving scene integrity. Extensive experiments demonstrate that our GSPure achieves the best watermark purification performance, reducing watermark PSNR by up to 16.34dB while minimizing degradation to original scene fidelity with less than 1dB PSNR loss. Moreover, it consistently outperforms existing methods in both effectiveness and generalization. Our code is available at https://github.com/insightlab-CG-3DV/GSPure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.03773v1">M2StyleGS: Multi-Modality 3D Style Transfer with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-04
    </div>
    <details class="paper-abstract">
      Conventional 3D style transfer methods rely on a fixed reference image to apply artistic patterns to 3D scenes. However, in practical applications such as virtual or augmented reality, users often prefer more flexible inputs, including textual descriptions and diverse imagery. In this work, we introduce a novel real-time styling technique M2StyleGS to generate a sequence of precisely color-mapped views. It utilizes 3D Gaussian Splatting (3DGS) as a 3D presentation and multi-modality knowledge refined by CLIP as a reference style. M2StyleGS resolves the abnormal transformation issue by employing a precise feature alignment, namely subdivisive flow, it strengthens the projection of the mapped CLIP text-visual combination feature to the VGG style feature. In addition, we introduce observation loss, which assists in the stylized scene better matching the reference style during the generation, and suppression loss, which suppresses the offset of reference color information throughout the decoding process. By integrating these approaches, M2StyleGS can employ text or images as references to generate a set of style-enhanced novel views. Our experiments show that M2StyleGS achieves better visual quality and surpasses the previous work by up to 32.92% in terms of consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.03716v1">CGHair: Compact Gaussian Hair Reconstruction with Card Clustering</a></div>
    <div class="paper-meta">
      📅 2026-04-04
      | 💬 Accepted to CVPR 2026. This arXiv version is not the final published version
    </div>
    <details class="paper-abstract">
      We present a compact pipeline for high-fidelity hair reconstruction from multi-view images. While recent 3D Gaussian Splatting (3DGS) methods achieve realistic results, they often require millions of primitives, leading to high storage and rendering costs. Observing that hair exhibits structural and visual similarities across a hairstyle, we cluster strands into representative hair cards and group these into shared texture codebooks. Our approach integrates this structure with 3DGS rendering, significantly reducing reconstruction time and storage while maintaining comparable visual quality. In addition, we propose a generative prior accelerated method to reconstruct the initial strand geometry from a set of images. Our experiments demonstrate a 4-fold reduction in strand reconstruction time and achieve comparable rendering performance with over 200x lower memory footprint.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.03462v1">SpectralSplat: Appearance-Disentangled Feed-Forward Gaussian Splatting for Driving Scenes</a></div>
    <div class="paper-meta">
      📅 2026-04-03
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting methods have achieved impressive reconstruction quality for autonomous driving scenes, yet they entangle scene geometry with transient appearance properties such as lighting, weather, and time of day. This coupling prevents relighting, appearance transfer, and consistent rendering across multi-traversal data captured under varying environmental conditions. We present SpectralSplat, a method that disentangles appearance from geometry within a feed-forward Gaussian Splatting framework. Our key insight is to factor color prediction into an appearance-agnostic base stream and and appearance-conditioned adapted stream, both produced by a shared MLP conditioned on a global appearance embedding derived from DINOv2 features. To enforce disentanglement, we train with paired observations generated by a hybrid relighting pipeline that combines physics-based intrinsic decomposition with diffusion based generative refinement, and supervise with complementary consistency, reconstruction, cross-appearance, and base color losses. We further introduce an appearance-adaptable temporal history that stores appearance-agnostic features, enabling accumulated Gaussians to be re-rendered under arbitrary target appearances. Experiments demonstrate that SpectralSplat preserves the reconstruction quality of the underlying backbone while enabling controllable appearance transfer and temporally consistent relighting across driving sequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.03092v1">Flash-Mono: Feed-Forward Accelerated Gaussian Splatting Monocular SLAM</a></div>
    <div class="paper-meta">
      📅 2026-04-03
    </div>
    <details class="paper-abstract">
      Monocular 3D Gaussian Splatting SLAM suffers from critical limitations in time efficiency, geometric accuracy, and multi-view consistency. These issues stem from the time-consuming $\textit{Train-from-Scratch}$ optimization and the lack of inter-frame scale consistency from single-frame geometry priors. We contend that a feed-forward paradigm, leveraging multi-frame context to predict Gaussian attributes directly, is crucial for addressing these challenges. We present Flash-Mono, a system composed of three core modules: a feed-forward prediction frontend, a 2D Gaussian Splatting mapping backend, and an efficient hidden-state-based loop closure module. We trained a recurrent feed-forward frontend model that progressively aggregates multi-frame visual features into a hidden state via cross attention and jointly predicts camera poses and per-pixel Gaussian properties. By directly predicting Gaussian attributes, our method bypasses the burdensome per-frame optimization required in optimization-based GS-SLAM, achieving a $\textbf{10x}$ speedup while ensuring high-quality rendering. The power of our recurrent architecture extends beyond efficient prediction. The hidden states act as compact submap descriptors, facilitating efficient loop closure and global $\mathrm{Sim}(3)$ optimization to mitigate the long-standing challenge of drift. For enhanced geometric fidelity, we replace conventional 3D Gaussian ellipsoids with 2D Gaussian surfels. Extensive experiments demonstrate that Flash-Mono achieves state-of-the-art performance in both tracking and mapping quality, highlighting its potential for embodied perception and real-time reconstruction applications. Project page: https://victkk.github.io/flash-mono.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.03069v1">SparseSplat: Towards Applicable Feed-Forward 3D Gaussian Splatting with Pixel-Unaligned Prediction</a></div>
    <div class="paper-meta">
      📅 2026-04-03
    </div>
    <details class="paper-abstract">
      Recent progress in feed-forward 3D Gaussian Splatting (3DGS) has notably improved rendering quality. However, the spatially uniform and highly redundant 3DGS map generated by previous feed-forward 3DGS methods limits their integration into downstream reconstruction tasks. We propose SparseSplat, the first feed-forward 3DGS model that adaptively adjusts Gaussian density according to scene structure and information richness of local regions, yielding highly compact 3DGS maps. To achieve this, we propose entropy-based probabilistic sampling, generating large, sparse Gaussians in textureless areas and assigning small, dense Gaussians to regions with rich information. Additionally, we designed a specialized point cloud network that efficiently encodes local context and decodes it into 3DGS attributes, addressing the receptive field mismatch between the general 3DGS optimization pipeline and feed-forward models. Extensive experimental results demonstrate that SparseSplat can achieve state-of-the-art rendering quality with only 22% of the Gaussians and maintain reasonable rendering quality with only 1.5% of the Gaussians. Project page: https://victkk.github.io/SparseSplat-page/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.26584v2">Scene Grounding In the Wild</a></div>
    <div class="paper-meta">
      📅 2026-04-03
      | 💬 CVPR 2026. Project page at https://tau-vailab.github.io/SceneGround/
    </div>
    <details class="paper-abstract">
      Reconstructing accurate 3D models of large-scale real-world scenes from unstructured, in-the-wild imagery remains a core challenge in computer vision, especially when the input views have little or no overlap. In such cases, existing reconstruction pipelines often produce multiple disconnected partial reconstructions or erroneously merge non-overlapping regions into overlapping geometry. In this work, we propose a framework that grounds each partial reconstruction to a complete reference model of the scene, enabling globally consistent alignment even in the absence of visual overlap. We obtain reference models from dense, geospatially accurate pseudo-synthetic renderings derived from Google Earth Studio. These renderings provide full scene coverage but differ substantially in appearance from real-world photographs. Our key insight is that, despite this significant domain gap, both domains share the same underlying scene semantics. We represent the reference model using 3D Gaussian Splatting, augmenting each Gaussian with semantic features, and formulate alignment as an inverse feature-based optimization scheme that estimates a global 6DoF pose and scale while keeping the reference model fixed. Furthermore, we introduce the WikiEarth dataset, which registers existing partial 3D reconstructions with pseudo-synthetic reference models. We demonstrate that our approach consistently improves global alignment when initialized with various classical and learning-based pipelines, while mitigating failure modes of state-of-the-art end-to-end models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06343v2">Uncertainty-Aware 4D Gaussian Splatting for Monocular Occluded Human Rendering</a></div>
    <div class="paper-meta">
      📅 2026-04-03
    </div>
    <details class="paper-abstract">
      High-fidelity rendering of dynamic humans from monocular videos typically degrades catastrophically under occlusions. Existing solutions incorporate external priors-either hallucinating missing content via generative models, which induces severe temporal flickering, or imposing rigid geometric heuristics that fail to capture diverse appearances. To this end, we reformulate the task as a Maximum A Posteriori estimation problem under heteroscedastic observation noise. In this paper, we propose U-4DGS, a framework integrating a Probabilistic Deformation Network and a Joint Rasterization pipeline. This architecture renders pixel-aligned uncertainty maps that act as an adaptive gradient modulator, automatically attenuating artifacts from unreliable observations. Furthermore, to prevent geometric drift in regions lacking reliable visual cues, we enforce Confidence-Aware Regularizations, which leverage the learned uncertainty to selectively propagate spatial-temporal validity. Extensive experiments on the ZJU-MoCap and OcMotion datasets demonstrate that U-4DGS achieves state-of-the-art rendering fidelity and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02915v1">GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes</a></div>
    <div class="paper-meta">
      📅 2026-04-03
      | 💬 CVPR 2026, Page: https://cv.snu.ac.kr/research/GP4DGS
    </div>
    <details class="paper-abstract">
      We present GP-4DGS, a novel framework that integrates Gaussian Processes (GPs) into 4D Gaussian Splatting (4DGS) for principled probabilistic modeling of dynamic scenes. While existing 4DGS methods focus on deterministic reconstruction, they are inherently limited in capturing motion ambiguity and lack mechanisms to assess prediction reliability. By leveraging the kernel-based probabilistic nature of GPs, our approach introduces three key capabilities: (i) uncertainty quantification for motion predictions, (ii) motion estimation for unobserved or sparsely sampled regions, and (iii) temporal extrapolation beyond observed training frames. To scale GPs to the large number of Gaussian primitives in 4DGS, we design spatio-temporal kernels that capture the correlation structure of deformation fields and adopt variational Gaussian Processes with inducing points for tractable inference. Our experiments show that GP-4DGS enhances reconstruction quality while providing reliable uncertainty estimates that effectively identify regions of high motion ambiguity. By addressing these challenges, our work takes a meaningful step toward bridging probabilistic modeling and neural graphics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02851v1">Streaming Real-Time Rendered Scenes as 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-04-03
    </div>
    <details class="paper-abstract">
      Cloud rendering is widely used in gaming and XR to overcome limited client-side GPU resources and to support heterogeneous devices. Existing systems typically deliver the rendered scene as a 2D video stream, which tightly couples the transmitted content to the server-rendered viewpoint and limits latency compensation to image-space reprojection or warping. In this paper, we investigate an alternative approach based on streaming a live 3D Gaussian Splatting (3DGS) scene representation instead of only rendered video. We present a Unity-based prototype in which a server constructs and continuously optimizes a 3DGS model from real-time rendered reference views, while streaming the evolving representation to remote clients using full model snapshots and incremental updates supporting relighting and rigid object dynamics. The clients reconstruct the streamed Gaussian model locally and render their current viewpoint from the received representation. This approach aims to improve viewpoint flexibility for latency compensation and to better amortize server-side scene modeling across multiple users than per-user rendering and video streaming. We describe the system design, evaluate it, and compare it with conventional image warping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02828v1">NavCrafter: Exploring 3D Scenes from a Single Image</a></div>
    <div class="paper-meta">
      📅 2026-04-03
      | 💬 8 pages accepted by ICRA 2026
    </div>
    <details class="paper-abstract">
      Creating flexible 3D scenes from a single image is vital when direct 3D data acquisition is costly or impractical. We introduce NavCrafter, a novel framework that explores 3D scenes from a single image by synthesizing novel-view video sequences with camera controllability and temporal-spatial consistency. NavCrafter leverages video diffusion models to capture rich 3D priors and adopts a geometry-aware expansion strategy to progressively extend scene coverage. To enable controllable multi-view synthesis, we introduce a multi-stage camera control mechanism that conditions diffusion models with diverse trajectories via dual-branch camera injection and attention modulation. We further propose a collision-aware camera trajectory planner and an enhanced 3D Gaussian Splatting (3DGS) pipeline with depth-aligned supervision, structural regularization and refinement. Extensive experiments demonstrate that NavCrafter achieves state-of-the-art novel-view synthesis under large viewpoint shifts and substantially improves 3D reconstruction fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02799v1">UNICA: A Unified Neural Framework for Controllable 3D Avatars</a></div>
    <div class="paper-meta">
      📅 2026-04-03
      | 💬 Opensource code: https://github.com/zjh21/UNICA
    </div>
    <details class="paper-abstract">
      Controllable 3D human avatars have found widespread applications in 3D games, the metaverse, and AR/VR scenarios. The conventional approach to creating such a 3D avatar requires a lengthy, intricate pipeline encompassing appearance modeling, motion planning, rigging, and physical simulation. In this paper, we introduce UNICA (UNIfied neural Controllable Avatar), a skeleton-free generative model that unifies all avatar control components into a single neural framework. Given keyboard inputs akin to video game controls, UNICA generates the next frame of a 3D avatar's geometry through an action-conditioned diffusion model operating on 2D position maps. A point transformer then maps the resulting geometry to 3D Gaussian Splatting for high-fidelity free-view rendering. Our approach naturally captures hair and loose clothing dynamics without manually designed physical simulation, and supports extra-long autoregressive generation. To the best of our knowledge, UNICA is the first model to unify the workflow of "motion planning, rigging, physical simulation, and rendering". Code is released at https://github.com/zjh21/UNICA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16672v2">ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-03
      | 💬 Accepted to CVPR 2026
    </div>
    <details class="paper-abstract">
      High-quality 3D garment reconstruction plays a crucial role in mitigating the sim-to-real gap in applications such as digital avatars, virtual try-on and robotic manipulation. However, existing garment reconstruction methods typically rely on unstructured representations, such as 3D Gaussian Splats, struggling to provide accurate reconstructions of garment topology and sewing structures. As a result, the reconstructed outputs are often unsuitable for high-fidelity physical simulation. We propose ReWeaver, a novel framework for topology-accurate 3D garment and sewing pattern reconstruction from sparse multi-view RGB images. Given as few as four input views, ReWeaver predicts seams and panels as well as their connectivities in both the 2D UV space and the 3D space. The predicted seams and panels align precisely with the multi-view images, yielding structured 2D--3D garment representations suitable for 3D perception, high-fidelity physical simulation, and robotic manipulation. To enable effective training, we construct a large-scale dataset GCD-TS, comprising multi-view RGB images, 3D garment geometries, textured human body meshes and annotated sewing patterns. The dataset contains over 100,000 synthetic samples covering a wide range of complex geometries and topologies. Extensive experiments show that ReWeaver consistently outperforms existing methods in terms of topology accuracy, geometry alignment and seam-panel consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02781v1">DynFOA: Generating First-Order Ambisonics with Conditional Diffusion for Dynamic and Acoustically Complex 360-Degree Videos</a></div>
    <div class="paper-meta">
      📅 2026-04-03
      | 💬 arXiv admin note: text overlap with arXiv:2602.06846
    </div>
    <details class="paper-abstract">
      Spatial audio is crucial for immersive 360-degree video experiences, yet most 360-degree videos lack it due to the difficulty of capturing spatial audio during recording. Automatically generating spatial audio such as first-order ambisonics (FOA) from video therefore remains an important but challenging problem. In complex scenes, sound perception depends not only on sound source locations but also on scene geometry, materials, and dynamic interactions with the environment. However, existing approaches only rely on visual cues and fail to model dynamic sources and acoustic effects such as occlusion, reflections, and reverberation. To address these challenges, we propose DynFOA, a generative framework that synthesizes FOA from 360-degree videos by integrating dynamic scene reconstruction with conditional diffusion modeling. DynFOA analyzes the input video to detect and localize dynamic sound sources, estimate depth and semantics, and reconstruct scene geometry and materials using 3D Gaussian Splatting (3DGS). The reconstructed scene representation provides physically grounded features that capture acoustic interactions between sources, environment, and listener viewpoint. Conditioned on these features, a diffusion model generates spatial audio consistent with the scene dynamics and acoustic context. We introduce M2G-360, a dataset of 600 real-world clips divided into MoveSources, Multi-Source, and Geometry subsets for evaluating robustness under diverse conditions. Experiments show that DynFOA consistently outperforms existing methods in spatial accuracy, acoustic fidelity, distribution matching, and perceived immersive experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02752v1">Differentiable Stroke Planning with Dual Parameterization for Efficient and High-Fidelity Painting Creation</a></div>
    <div class="paper-meta">
      📅 2026-04-03
    </div>
    <details class="paper-abstract">
      In stroke-based rendering, search methods often get trapped in local minima due to discrete stroke placement, while differentiable optimizers lack structural awareness and produce unstructured layouts. To bridge this gap, we propose a dual representation that couples discrete polylines with continuous Bézier control points via a bidirectional mapping mechanism. This enables collaborative optimization: local gradients refine global stroke structures, while content-aware stroke proposals help escape poor local optima. Our representation further supports Gaussian-splatting-inspired initialization, enabling highly parallel stroke optimization across the image. Experiments show that our approach reduces the number of strokes by 30-50%, achieves more structurally coherent layouts, and improves reconstruction quality, while cutting optimization time by 30-40% compared to existing differentiable vectorization methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02696v1">VBGS-SLAM: Variational Bayesian Gaussian Splatting Simultaneous Localization and Mapping</a></div>
    <div class="paper-meta">
      📅 2026-04-03
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown promising results for 3D scene modeling using mixtures of Gaussians, yet its existing simultaneous localization and mapping (SLAM) variants typically rely on direct, deterministic pose optimization against the splat map, making them sensitive to initialization and susceptible to catastrophic forgetting as map evolves. We propose Variational Bayesian Gaussian Splatting SLAM (VBGS-SLAM), a novel framework that couples the splat map refinement and camera pose tracking in a generative probabilistic form. By leveraging conjugate properties of multivariate Gaussians and variational inference, our method admits efficient closed-form updates and explicitly maintains posterior uncertainty over both poses and scene parameters. This uncertainty-aware method mitigates drift and enhances robustness in challenging conditions, while preserving the efficiency and rendering quality of existing 3DGS. Our experiments demonstrate superior tracking performance and robustness in long sequence prediction, alongside efficient, high-quality novel view synthesis across diverse synthetic and real-world scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01581v2">Satellite-Free Training for Drone-View Geo-Localization</a></div>
    <div class="paper-meta">
      📅 2026-04-03
    </div>
    <details class="paper-abstract">
      Drone-view geo-localization (DVGL) aims to determine the location of drones in GPS-denied environments by retrieving the corresponding geotagged satellite tile from a reference gallery given UAV observations of a location. In many existing formulations, these observations are represented by a single oblique UAV image. In contrast, our satellite-free setting is designed for multi-view UAV sequences, which are used to construct a geometry-normalized UAV-side location representation before cross-view retrieval. Existing approaches rely on satellite imagery during training, either through paired supervision or unsupervised alignment, which limits practical deployment when satellite data are unavailable or restricted. In this paper, we propose a satellite-free training (SFT) framework that converts drone imagery into cross-view compatible representations through three main stages: drone-side 3D scene reconstruction, geometry-based pseudo-orthophoto generation, and satellite-free feature aggregation for retrieval. Specifically, we first reconstruct dense 3D scenes from multi-view drone images using 3D Gaussian splatting and project the reconstructed geometry into pseudo-orthophotos via PCA-guided orthographic projection. This rendering stage operates directly on reconstructed scene geometry without requiring camera parameters at rendering time. Next, we refine these orthophotos with lightweight geometry-guided inpainting to obtain texture-complete drone-side views. Finally, we extract DINOv3 patch features from the generated orthophotos, learn a Fisher vector aggregation model solely from drone data, and reuse it at test time to encode satellite tiles for cross-view retrieval. Experimental results on University-1652 and SUES-200 show that our SFT framework substantially outperforms satellite-free generalization baselines and narrows the gap to methods trained with satellite imagery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01447v2">Better Rigs, Not Bigger Networks: A Body Model Ablation for Gaussian Avatars</a></div>
    <div class="paper-meta">
      📅 2026-04-03
    </div>
    <details class="paper-abstract">
      Recent 3D Gaussian splatting methods built atop SMPL achieve remarkable visual fidelity while continually increasing the complexity of the overall training architecture. We demonstrate that much of this complexity is unnecessary: by replacing SMPL with the Momentum Human Rig (MHR), estimated via SAM-3D-Body, a minimal pipeline with no learned deformations or pose-dependent corrections achieves the highest reported PSNR and competitive or superior LPIPS and SSIM on PeopleSnapshot and ZJU-MoCap. To disentangle pose estimation quality from body model representational capacity, we perform two controlled ablations: translating SAM-3D-Body meshes to SMPL-X, and translating the original dataset's SMPL poses into MHR both retrained under identical conditions. These ablations confirm that body model expressiveness has been a primary bottleneck in avatar reconstruction, with both mesh representational capacity and pose estimation quality contributing meaningfully to the full pipeline's gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02586v1">TrackerSplat: Exploiting Point Tracking for Fast and Robust Dynamic 3D Gaussians Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-02
      | 💬 11 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3DGS) have demonstrated its potential for efficient and photorealistic 3D reconstructions, which is crucial for diverse applications such as robotics and immersive media. However, current Gaussian-based methods for dynamic scene reconstruction struggle with large inter-frame displacements, leading to artifacts and temporal inconsistencies under fast object motions. To address this, we introduce \textit{TrackerSplat}, a novel method that integrates advanced point tracking methods to enhance the robustness and scalability of 3DGS for dynamic scene reconstruction. TrackerSplat utilizes off-the-shelf point tracking models to extract pixel trajectories and triangulate per-view pixel trajectories onto 3D Gaussians to guide the relocation, rotation, and scaling of Gaussians before training. This strategy effectively handles large displacements between frames, dramatically reducing the fading and recoloring artifacts prevalent in prior methods. By accurately positioning Gaussians prior to gradient-based optimization, TrackerSplat overcomes the quality degradation associated with large frame gaps when processing multiple adjacent frames in parallel across multiple devices, thereby boosting reconstruction throughput while preserving rendering quality. Experiments on real-world datasets confirm the robustness of TrackerSplat in challenging scenarios with significant displacements, achieving superior throughput under parallel settings and maintaining visual quality compared to baselines. The code is available at https://github.com/yindaheng98/TrackerSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23292v3">FACT-GS: Frequency-Aligned Complexity-Aware Texture Reparameterization for 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-02
      | 💬 11 pages, 6 figures, CVPR 2026 Findings track. Project page: https://tianhaoxie.github.io/project/FACT-GS/
    </div>
    <details class="paper-abstract">
      Realistic scene appearance modeling has advanced rapidly with Gaussian Splatting, which enables real-time, high-quality rendering. Recent advances introduced per-primitive textures that incorporate spatial color variations within each Gaussian, improving their expressiveness. However, texture-based Gaussians parameterize appearance with a uniform per-Gaussian sampling grid, allocating equal sampling density regardless of local visual complexity, which leads to inefficient texture space utilization. We introduce FACT-GS, a Frequency-Aligned Complexity-aware Texture Gaussian Splatting framework that allocates texture sampling density according to local visual frequency. Grounded in adaptive sampling theory, FACT-GS reformulates texture parameterization as a differentiable sampling-density allocation problem, replacing the uniform textures with a learnable frequency-aware allocation strategy implemented via a deformation field whose Jacobian modulates local sampling density. Built on 2D Gaussian Splatting, FACT-GS performs non-uniform sampling on fixed-resolution texture grids, preserving real-time performance while recovering sharper high-frequency details under the same parameter budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02003v1">ProDiG: Progressive Diffusion-Guided Gaussian Splatting for Aerial to Ground Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-02
    </div>
    <details class="paper-abstract">
      Generating ground-level views and coherent 3D site models from aerial-only imagery is challenging due to extreme viewpoint changes, missing intermediate observations, and large scale variations. Existing methods either refine renderings post-hoc, often producing geometrically inconsistent results, or rely on multi-altitude ground-truth, which is rarely available. Gaussian Splatting and diffusion-based refinements improve fidelity under small variations but fail under wide aerial-to-ground gaps. To address these limitations, we introduce ProDiG (Progressive Altitude Gaussian Splatting), a diffusion-guided framework that progressively transforms aerial 3D representations toward ground-level fidelity. ProDiG synthesizes intermediate-altitude views and refines the Gaussian representation at each stage using a geometry-aware causal attention module that injects epipolar structure into reference-view diffusion. A distance-adaptive Gaussian module dynamically adjusts Gaussian scale and opacity based on camera distance, ensuring stable reconstruction across large viewpoint gaps. Together, these components enable progressive, geometrically grounded refinement without requiring additional ground-truth viewpoints. Extensive experiments on synthetic and real-world datasets demonstrate that ProDiG produces visually realistic ground-level renderings and coherent 3D geometry, significantly outperforming existing approaches in terms of visual quality, geometric consistency, and robustness to extreme viewpoint changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01994v1">Resonance4D: Frequency-Domain Motion Supervision for Preset-Free Physical Parameter Learning in 4D Dynamic Physical Scene Simulation</a></div>
    <div class="paper-meta">
      📅 2026-04-02
    </div>
    <details class="paper-abstract">
      Physics-driven 4D dynamic simulation from static 3D scenes remains constrained by an overlooked contradiction: reliable motion supervision often relies on online video diffusion or optical-flow pipelines whose computational cost exceeds that of the simulator itself. Existing methods further simplify inverse physical modeling by optimizing only partial material parameters, limiting realism in scenes with complex materials and dynamics. We present Resonance4D, a physics-driven 4D dynamic simulation framework that couples 3D Gaussian Splatting with the Material Point Method through lightweight yet physically expressive supervision. Our key insight is that dynamic consistency can be enforced without dense temporal generation by jointly constraining motion in complementary domains. To this end, we introduce Dual-domain Motion Supervision (DMS), which combines spatial structural consistency for local deformation with frequency-domain spectral consistency for oscillatory and global dynamic patterns, substantially reducing training cost and memory overhead while preserving physically meaningful motion cues. To enable stable full-parameter physical recovery, we further combine zero-shot text-prompted segmentation with simulation-guided initialization to automatically decompose Gaussians into object-part-level regions and support joint optimization of full material parameters. Experiments on both synthetic and real scenes show that Resonance4D achieves strong physical fidelity and motion consistency while reducing peak GPU memory from over 35\,GB to around 20\,GB, enabling high-fidelity physics-driven 4D simulation on a single consumer-grade GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17092v3">SPAGS: Sparse-View Articulated Object Reconstruction from Single State via Planar Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-02
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Articulated objects are ubiquitous in daily environments, and their 3D reconstruction holds great significance across various fields. However, existing articulated object reconstruction methods typically require costly inputs such as multi-stage and multi-view observations. To address the limitations, we propose a category-agnostic articulated object reconstruction framework via planar Gaussian Splatting, which only uses sparse-view RGB images from a single state. Specifically, we first introduce a Gaussian information field to perceive the optimal sparse viewpoints from candidate camera poses. To ensure precise geometric fidelity, we constrain traditional 3D Gaussians into planar primitives, facilitating accurate normal and depth estimation. The planar Gaussians are then optimized in a coarse-to-fine manner, regularized by depth smoothness and few-shot diffusion priors. Furthermore, we leverage a Vision-Language Model (VLM) via visual prompting to achieve open-vocabulary part segmentation and joint parameter estimation. Extensive experiments on both synthetic and real-world datasets demonstrate that our approach significantly outperforms existing baselines, achieving superior part-level surface reconstruction fidelity. Code and data are provided in the supplementary material.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01884v1">GS^2: Graph-based Spatial Distribution Optimization for Compact 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated breakthrough performance in novel view synthesis and real-time rendering. Nevertheless, its practicality is constrained by the high memory cost due to a huge number of Gaussian points. Many pruning-based 3DGS variants have been proposed for memory saving, but often compromise spatial consistency and may lead to rendering artifacts. To address this issue, we propose graph-based spatial distribution optimization for compact 3D Gaussian Splatting (GS\textasciicircum2), which enhances reconstruction quality by optimizing the spatial distribution of Gaussian points. Specifically, we introduce an evidence lower bound (ELBO)-based adaptive densification strategy that automatically controls the densification process. In addition, an opacity-aware progressive pruning strategy is proposed to further reduce memory consumption by dynamically removing low-opacity Gaussian points. Furthermore, we propose a graph-based feature encoding module to adjust the spatial distribution via feature-guided point shifting. Extensive experiments validate that GS\textasciicircum2 achieves a compact Gaussian representation while delivering superior rendering quality. Compared with 3DGS, it achieves higher PSNR with only about 12.5\% Gaussian points. Furthermore, it outperforms all compared baselines in both rendering quality and memory efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01844v1">FaCT-GS: Fast and Scalable CT Reconstruction with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-02
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has emerged as a dominating technique for image rendering and has quickly been adapted for the X-ray Computed Tomography (CT) reconstruction task. However, despite being on par or better than many of its predecessors, the benefits of GS are typically not substantial enough to motivate a transition from well-established reconstruction algorithms. This paper addresses the most significant remaining limitations of the GS-based approach by introducing FaCT-GS, a framework for fast and flexible CT reconstruction. Enabled by an in-depth optimization of the voxelization and rasterization pipelines, our new method is significantly faster than its predecessors and scales well with projection and output volume size. Furthermore, the improved voxelization enables rapid fitting of Gaussians to pre-existing volumes, which can serve as a prior for warm-starting the reconstruction, or simply as an alternative, compressed representation. FaCT-GS is over 4X faster than the State of the Art GS CT reconstruction on standard 512x512 projections, and over 13X faster on 2k projections. Implementation available at: https://github.com/PaPieta/fact-gs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19834v2">Fourier Splatting: Generalized Fourier encoded primitives for scalable radiance fields</a></div>
    <div class="paper-meta">
      📅 2026-04-02
    </div>
    <details class="paper-abstract">
      Novel view synthesis has recently been revolutionized by 3D Gaussian Splatting (3DGS), which enables real-time rendering through explicit primitive rasterization. However, existing methods tie visual fidelity strictly to the number of primitives: quality downscaling is achieved only through pruning primitives. We propose the first inherently scalable primitive for radiance field rendering. Fourier Splatting employs scalable primitives with arbitrary closed shapes obtained by parameterizing planar surfels with Fourier encoded descriptors. This formulation allows a single trained model to be rendered at varying levels of detail simply by truncating Fourier coefficients at runtime. To facilitate stable optimization, we employ a straight-through estimator for gradient extension beyond the primitive boundary, and introduce HYDRA, a densification strategy that decomposes complex primitives into simpler constituents within the MCMC framework. Our method achieves state-of-the-art rendering quality among planar-primitive frameworks and comparable perceptual metrics compared to leading volumetric representations on standard benchmarks, providing a versatile solution for bandwidth-constrained high-fidelity rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.22279v2">Learning Fine-Grained Geometry for Sparse-View Splatting via Cascade Depth Loss</a></div>
    <div class="paper-meta">
      📅 2026-04-02
    </div>
    <details class="paper-abstract">
      Novel view synthesis is a fundamental task in 3D computer vision that aims to reconstruct photorealistic images from novel viewpoints given a set of posed images. However, reconstruction quality degrades sharply under sparse-view conditions due to insufficient geometric cues. Existing methods, including Neural Radiance Fields (NeRF) and more recent 3D Gaussian Splatting (3DGS), often exhibit blurred details and structural artifacts when trained from sparse observations. Recent works have identified rendered depth quality as a key factor in mitigating these artifacts, as it directly affects geometric accuracy and view consistency. However, effectively leveraging depth under sparse views remains challenging. Depth priors can be noisy or misaligned with rendered geometry, and single-scale supervision often fails to capture both global structure and fine details. To address these challenges, we introduce Hierarchical Depth-Guided Splatting (HDGS), a depth supervision framework that progressively refines geometry from coarse to fine levels. Central to HDGS is our novel Cascade Pearson Correlation Loss (CPCL), which enforces consistency between rendered and estimated depth priors across multiple spatial scales. By enforcing multi-scale depth consistency, our method improves structural fidelity in sparse-view reconstruction. Experiments on LLFF and DTU demonstrate state-of-the-art performance under sparse-view settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01678v1">Director: Instance-aware Gaussian Splatting for Dynamic Scene Modeling and Understanding</a></div>
    <div class="paper-meta">
      📅 2026-04-02
      | 💬 Project page: https://caiyw2023.github.io/Director/
    </div>
    <details class="paper-abstract">
      Volumetric video seeks to model dynamic scenes as temporally coherent 4D representations. While recent Gaussian-based approaches achieve impressive rendering fidelity, they primarily emphasize appearance but are largely agnostic to instance-level structure, limiting stable tracking and semantic reasoning in highly dynamic scenarios. In this paper, we present Director, a unified spatio-temporal Gaussian representation that jointly models human performance, high-fidelity rendering, and instance-level semantics. Our key insight is that embedding instance-consistent semantics naturally complements 4D modeling, enabling more accurate scene decomposition while supporting robust dynamic scene understanding. To this end, we leverage temporally aligned instance masks and sentence embeddings derived from Multimodal Large Language Models to supervise the learnable semantic features of each Gaussian via two MLP decoders, enabling language-aligned 4D representations and enforcing identity consistency over time. To enhance temporal stability, we bridge 2D optical flow with 4D Gaussians and finetune their motions, yielding reliable initialization and reducing drift. For the training, we further introduce a geometry-aware SDF constraints, along with regularization terms that enforces surface continuity, enhancing temporal coherence in dynamic foreground modeling. Experiments demonstrate that Director achieves temporally coherent 4D reconstructions while simultaneously enabling instance segmentation and open-vocabulary querying.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01605v1">F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling</a></div>
    <div class="paper-meta">
      📅 2026-04-02
      | 💬 Accepted to the CVPR 2026 SPAR-3D Workshop
    </div>
    <details class="paper-abstract">
      We present F3DGS, a federated 3D Gaussian Splatting framework for decentralized multi-agent 3D reconstruction. Existing 3DGS pipelines assume centralized access to all observations, which limits their applicability in distributed robotic settings where agents operate independently, and centralized data aggregation may be restricted. Directly extending centralized training to multi-agent systems introduces communication overhead and geometric inconsistency. F3DGS first constructs a shared geometric scaffold by registering locally merged LiDAR point clouds from multiple clients to initialize a global 3DGS model. During federated optimization, Gaussian positions are fixed to preserve geometric alignment, while each client updates only appearance-related attributes, including covariance, opacity, and spherical harmonic coefficients. The server aggregates these updates using visibility-aware aggregation, weighting each client's contribution by how frequently it observed each Gaussian, resolving the partial-observability challenge inherent to multi-agent exploration. To evaluate decentralized reconstruction, we collect a multi-sequence indoor dataset with synchronized LiDAR, RGB, and IMU measurements. Experiments show that F3DGS achieves reconstruction quality comparable to centralized training while enabling distributed optimization across agents. The dataset, development kit, and source code will be publicly released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20331v3">GVGS: Gaussian Visibility-Aware Multi-View Geometry for Accurate Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables efficient rendering, yet accurate surface reconstruction remains challenging due to unreliable geometric supervision. Existing approaches predominantly rely on depth-based reprojection to infer visibility and enforce multi-view consistency, leading to a fundamental circular dependency: visibility estimation requires accurate depth, while depth supervision itself is conditioned on visibility. In this work, we revisit multi-view geometric supervision from the perspective of visibility modeling. Instead of inferring visibility from pixel-wise depth consistency, we explicitly model visibility at the level of Gaussian primitives. We introduce a Gaussian visibility-aware multi-view geometric consistency (GVMV) formulation, which aggregates cross-view visibility of shared Gaussians to construct reliable supervision over co-visible regions. To further incorporate monocular priors, we propose a progressive quadtree-calibrated depth alignment (QDC) strategy that performs block-wise affine calibration under visibility-aware guidance, effectively mitigating scale ambiguity while preserving local geometric structures. Extensive experiments on DTU and Tanks and Temples demonstrate that our method consistently improves reconstruction accuracy over prior Gaussian-based approaches. Our code is fully open-sourced and available at an anonymous repository: https://github.com/GVGScode/GVGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01551v1">ColorGradedGaussians: Palette-Based Color Grading for 3D Gaussian Splatting via View-Space Sparse Decomposition</a></div>
    <div class="paper-meta">
      📅 2026-04-02
      | 💬 9 pages, 2 figure pages
    </div>
    <details class="paper-abstract">
      Professional color editing requires precise control over both color (hue and saturation) and lightness, ideally through separate, independent controls. We present a real-time interactive color editing framework for 3D Gaussian Splatting (3DGS) that enables palette-based recoloring, per-palette tone curves for color-aware lightness adjustment, and accurate pixel-level constraints -- capabilities unavailable in prior palette-based 3DGS methods. Existing approaches decompose colors at the primitive level, optimizing per-Gaussian palette weights before splatting. However, sparse primitive-level weights do not guarantee sparse pixel-level decompositions after alpha-blending, causing palette edits to affect unintended regions and degrading editing quality. We address this through view-space palette decomposition, splatting weights instead of colors to optimize the observable appearance of the scene. We introduce a geometric loss using inverse barycentric coordinates to enforce consistent sparsity patterns, ensuring similar colors share similar decompositions. Our approach achieves superior editing quality compared to primitive-space methods, enabling professional color grading workflows for 3DGS scenes with real-time interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01388v1">LESV: Language Embedded Sparse Voxel Fusion for Open-Vocabulary 3D Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2026-04-01
    </div>
    <details class="paper-abstract">
      Recent advancements in open-vocabulary 3D scene understanding heavily rely on 3D Gaussian Splatting (3DGS) to register vision-language features into 3D space. However, we identify two critical limitations in these approaches: the spatial ambiguity arising from unstructured, overlapping Gaussians which necessitates probabilistic feature registration, and the multi-level semantic ambiguity caused by pooling features over object-level masks, which dilutes fine-grained details. To address these challenges, we present a novel framework that leverages Sparse Voxel Rasterization (SVRaster) as a structured, disjoint geometry representation. By regularizing SVRaster with monocular depth and normal priors, we establish a stable geometric foundation. This enables a deterministic, confidence-aware feature registration process and suppresses the semantic bleeding artifact common in 3DGS. Furthermore, we resolve multi-level ambiguity by exploiting the emerging dense alignment properties of foundation model AM-RADIO, avoiding the computational overhead of hierarchical training methods. Our approach achieves state-of-the-art performance on Open Vocabulary 3D Object Retrieval and Point Cloud Understanding benchmarks, particularly excelling on fine-grained queries where registration methods typically fail.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28431v3">LG-HCC: Local Geometry-Aware Hierarchical Context Compression for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-01
      | 💬 10
    </div>
    <details class="paper-abstract">
      Although 3D Gaussian Splatting (3DGS) enables high-fidelity real-time rendering, its prohibitive storage overhead severely hinders practical deployment. Recent anchor-based 3DGS compression schemes reduce gaussian redundancy through some advanced context models. However, they overlook explicit geometric dependencies, leading to structural degradation and suboptimal ratedistortion performance. In this paper, we propose a Local Geometry-aware Hierarchical Context Compression framework for 3DGS(LG-HCC) that incorporates inter-anchor geometric correlations into anchor pruning and entropy coding for compact representation. Specifically, we introduce an Neighborhood-Aware Anchor Pruning (NAAP) strategy, which evaluates anchor importance via weighted neighborhood feature aggregation and then merges low-contribution anchors into salient neighbors, yielding a compact yet geometry-consistent anchor set. Moreover, we further develop a hierarchical entropy coding scheme, in which coarse-to-fine priors are exploited through a lightweight Geometry-Guided Convolution(GG-Conv) operator to enable spatially adaptive context modeling and rate-distortion optimization. Extensive experiments show that LG-HCC effectively alleviates structural preservation issues,achieving superior geometric integrity and rendering fidelity while reducing storage by up to 30.85x compared to the Scaffold-GS baseline on the Mip-NeRF360 dataset
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01030v1">Diff3R: Feed-forward 3D Gaussian Splatting with Uncertainty-aware Differentiable Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-01
      | 💬 Project page: https://liu115.github.io/diff3r, Video: https://www.youtube.com/watch?v=IxzNSAdUY70
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) present two main directions: feed-forward models offer fast inference in sparse-view settings, while per-scene optimization yields high-quality renderings but is computationally expensive. To combine the benefits of both, we introduce Diff3R, a novel framework that explicitly bridges feed-forward prediction and test-time optimization. By incorporating a differentiable 3DGS optimization layer directly into the training loop, our network learns to predict an optimal initialization for test-time optimization rather than a conventional zero-shot result. To overcome the computational cost of backpropagating through the optimization steps, we propose computing gradients via the Implicit Function Theorem and a scalable, matrix-free PCG solver tailored for 3DGS optimization. Additionally, we incorporate a data-driven uncertainty model into the optimization process by adaptively controlling how much the parameters are allowed to change during optimization. This approach effectively mitigates overfitting in under-constrained regions and increases robustness against input outliers. Since our proposed optimization layer is model-agnostic, we show that it can be seamlessly integrated into existing feed-forward 3DGS architectures for both pose-given and pose-free methods, providing improvements for test-time optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09997v2">CLoD-GS: Continuous Level-of-Detail via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-01
      | 💬 Accepted by ICLR 2026 poster
    </div>
    <details class="paper-abstract">
      Level of Detail (LoD) is a fundamental technique in real-time computer graphics for managing the rendering costs of complex scenes while preserving visual fidelity. Traditionally, LoD is implemented using discrete levels (DLoD), where multiple, distinct versions of a model are swapped out at different distances. This long-standing paradigm, however, suffers from two major drawbacks: it requires significant storage for multiple model copies and causes jarring visual ``popping" artifacts during transitions, degrading the user experience. We argue that the explicit, primitive-based nature of the emerging 3D Gaussian Splatting (3DGS) technique enables a more ideal paradigm: Continuous LoD (CLoD). A CLoD approach facilitates smooth, seamless quality scaling within a single, unified model, thereby circumventing the core problems of DLOD. To this end, we introduce CLoD-GS, a framework that integrates a continuous LoD mechanism directly into a 3DGS representation. Our method introduces a learnable, distance-dependent decay parameter for each Gaussian primitive, which dynamically adjusts its opacity based on viewpoint proximity. This allows for the progressive and smooth filtering of less significant primitives, effectively creating a continuous spectrum of detail within one model. To train this model to be robust across all distances, we introduce a virtual distance scaling mechanism and a novel coarse-to-fine training strategy with rendered point count regularization. Our approach not only eliminates the storage overhead and visual artifacts of discrete methods but also reduces the primitive count and memory footprint of the final model. Extensive experiments demonstrate that CLoD-GS achieves smooth, quality-scalable rendering from a single model, delivering high-fidelity results across a wide range of performance targets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.00928v1">Autoregressive Appearance Prediction for 3D Gaussian Avatars</a></div>
    <div class="paper-meta">
      📅 2026-04-01
      | 💬 Project Page: https://steimich96.github.io/AAP-3DGA/
    </div>
    <details class="paper-abstract">
      A photorealistic and immersive human avatar experience demands capturing fine, person-specific details such as cloth and hair dynamics, subtle facial expressions, and characteristic motion patterns. Achieving this requires large, high-quality datasets, which often introduce ambiguities and spurious correlations when very similar poses correspond to different appearances. Models that fit these details during training can overfit and produce unstable, abrupt appearance changes for novel poses. We propose a 3D Gaussian Splatting avatar model with a spatial MLP backbone that is conditioned on both pose and an appearance latent. The latent is learned during training by an encoder, yielding a compact representation that improves reconstruction quality and helps disambiguate pose-driven renderings. At driving time, our predictor autoregressively infers the latent, producing temporally smooth appearance evolution and improved stability. Overall, our method delivers a robust and practical path to high-fidelity, stable avatar driving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.00804v1">Compact Keyframe-Optimized Multi-Agent Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2026-04-01
    </div>
    <details class="paper-abstract">
      Efficient multi-agent 3D mapping is essential for robotic teams operating in unknown environments, but dense representations hinder real-time exchange over constrained communication links. In multi-agent Simultaneous Localization and Mapping (SLAM), systems typically rely on a centralized server to merge and optimize the local maps produced by individual agents. However, sharing these large map representations, particularly those generated by recent methods such as Gaussian Splatting, becomes a bottleneck in real-world scenarios with limited bandwidth. We present an improved multi-agent RGB-D Gaussian Splatting SLAM framework that reduces communication load while preserving map fidelity. First, we incorporate a compaction step into our SLAM system to remove redundant 3D Gaussians, without degrading the rendering quality. Second, our approach performs centralized loop closure computation without initial guess, operating in two modes: a pure rendered-depth mode that requires no data beyond the 3D Gaussians, and a camera-depth mode that includes lightweight depth images for improved registration accuracy and additional Gaussian pruning. Evaluation on both synthetic and real-world datasets shows up to 85-95\% reduction in transmitted data compared to state-of-the-art approaches in both modes, bringing 3D Gaussian multi-agent SLAM closer to practical deployment in real-world scenarios. Code: https://github.com/lemonci/coko-slam
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.00648v1">DirectFisheye-GS: Enabling Native Fisheye Input in Gaussian Splatting with Cross-View Joint Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-01
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has enabled efficient 3D scene reconstruction from everyday images with real-time, high-fidelity rendering, greatly advancing VR/AR applications. Fisheye cameras, with their wider field of view (FOV), promise high-quality reconstructions from fewer inputs and have recently attracted much attention. However, since 3DGS relies on rasterization, most subsequent works involving fisheye camera inputs first undistort images before training, which introduces two problems: 1) Black borders at image edges cause information loss and negate the fisheye's large FOV advantage; 2) Undistortion's stretch-and-interpolate resampling spreads each pixel's value over a larger area, diluting detail density -- causes 3DGS overfitting these low-frequency zones, producing blur and floating artifacts. In this work, we integrate fisheye camera model into the original 3DGS framework, enabling native fisheye image input for training without preprocessing. Despite correct modeling, we observed that the reconstructed scenes still exhibit floaters at image edges: Distortion increases toward the periphery, and 3DGS's original per-iteration random-selecting-view optimization ignores the cross-view correlations of a Gaussian, leading to extreme shapes (e.g., oversized or elongated) that degrade reconstruction quality. To address this, we introduce a feature-overlap-driven cross-view joint optimization strategy that establishes consistent geometric and photometric constraints across views-a technique equally applicable to existing pinhole-camera-based pipelines. Our DirectFisheye-GS matches or surpasses state-of-the-art performance on public datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10075v2">Thinking Like Van Gogh: Structure-Aware Style Transfer via Flow-Guided 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-01
      | 💬 7 pages, 8 figures
    </div>
    <details class="paper-abstract">
      In 1888, Vincent van Gogh wrote, "I am seeking exaggeration in the essential." This principle, amplifying structural form while suppressing photographic detail, lies at the core of Post-Impressionist art. However, most existing 3D style transfer methods invert this philosophy, treating geometry as a rigid substrate for surface-level texture projection. To authentically reproduce Post-Impressionist stylization, geometric abstraction must be embraced as the primary vehicle of expression. We propose a flow-guided geometric advection framework for 3D Gaussian Splatting (3DGS) that operationalizes this principle in a mesh-free setting. Our method extracts directional flow fields from 2D paintings and back-propagates them into 3D space, rectifying Gaussian primitives to form flow-aligned brushstrokes that conform to scene topology without relying on explicit mesh priors. This enables expressive structural deformation driven directly by painterly motion rather than photometric constraints. Our contributions are threefold: (1) a projection-based, mesh-free flow guidance mechanism that transfers 2D artistic motion into 3D Gaussian geometry; (2) a luminance-structure decoupling strategy that isolates geometric deformation from color optimization, mitigating artifacts during aggressive structural abstraction; and (3) a VLM-as-a-Judge evaluation framework that assesses artistic authenticity through aesthetic judgment instead of conventional pixel-level metrics, explicitly addressing the subjective nature of artistic stylization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.00538v1">TRiGS: Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-01
      | 💬 Project page: https://wwwjjn.github.io/TRiGS-project_page/
    </div>
    <details class="paper-abstract">
      Recent 4D Gaussian Splatting (4DGS) methods achieve impressive dynamic scene reconstruction but often rely on piecewise linear velocity approximations and short temporal windows. This disjointed modeling leads to severe temporal fragmentation, forcing primitives to be repeatedly eliminated and regenerated to track complex nonlinear dynamics. This makeshift approximation eliminates the long-term temporal identity of objects and causes an inevitable proliferation of Gaussians, hindering scalability to extended video sequences. To address this, we propose TRiGS, a novel 4D representation that utilizes unified, continuous geometric transformations. By integrating $SE(3)$ transformations, hierarchical Bezier residuals, and learnable local anchors, TRiGS models geometrically consistent rigid motions for individual primitives. This continuous formulation preserves temporal identity and effectively mitigates unbounded memory growth. Extensive experiments demonstrate that TRiGS achieves high fidelity rendering on standard benchmarks while uniquely scaling to extended video sequences (e.g., 600 to 1200 frames) without severe memory bottlenecks, significantly outperforming prior works in temporal stability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.00509v1">RT-GS: Gaussian Splatting with Reflection and Transmittance Primitives</a></div>
    <div class="paper-meta">
      📅 2026-04-01
    </div>
    <details class="paper-abstract">
      Gaussian Splatting is a powerful tool for reconstructing diffuse scenes, but it struggles to simultaneously model specular reflections and the appearance of objects behind semi-transparent surfaces. These specular reflections and transmittance are essential for realistic novel view synthesis, and existing methods do not properly incorporate the underlying physical processes to simulate them. To address this issue, we propose RT-GS, a unified framework that integrates a microfacet material model and ray tracing to jointly model specular reflection and transmittance in Gaussian Splatting. We accomplish this by using separate Gaussian primitives for reflections and transmittance, which allow modeling distant reflections and reconstructing objects behind transparent surfaces concurrently. We utilize a differentiable ray tracing framework to obtain the specular reflection and transmittance appearance. Our experiments demonstrate that our method successfully produces reflections and recovers objects behind transparent surfaces in complex environments, achieving significant qualitative improvements over prior methods where these specular light interactions are prominent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18640v2">Geometric-Photometric Event-based 3D Gaussian Ray Tracing</a></div>
    <div class="paper-meta">
      📅 2026-04-01
      | 💬 15 pages, 12 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Event cameras offer a high temporal resolution over traditional frame-based cameras, which makes them suitable for motion and structure estimation. However, it has been unclear how event-based 3D Gaussian Splatting (3DGS) approaches could leverage fine-grained temporal information of sparse events. This work proposes GPERT, a framework to address the trade-off between accuracy and temporal resolution in event-based 3DGS. Our key idea is to decouple the rendering into two branches: event-by-event geometry (depth) rendering and snapshot-based radiance (intensity) rendering, by using ray-tracing and the image of warped events. The extensive evaluation shows that our method achieves state-of-the-art performance on the real-world datasets and competitive performance on the synthetic dataset. Also, the proposed method works without prior information (e.g., pretrained image reconstruction models) or COLMAP-based initialization, is more flexible in the event selection number, and achieves sharp reconstruction on scene edges with fast training time. We hope that this work deepens our understanding of the sparse nature of events for 3D reconstruction. https://github.com/e3ai/gpert
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.00494v1">ARGS: Auto-Regressive Gaussian Splatting via Parallel Progressive Next-Scale Prediction</a></div>
    <div class="paper-meta">
      📅 2026-04-01
    </div>
    <details class="paper-abstract">
      Auto-regressive frameworks for next-scale prediction of 2D images have demonstrated strong potential for producing diverse and sophisticated content by progressively refining a coarse input. However, extending this paradigm to 3D object generation remains largely unexplored. In this paper, we introduce auto-regressive Gaussian splatting (ARGS), a framework for making next-scale predictions in parallel for generation according to levels of detail. We propose a Gaussian simplification strategy and reverse the simplification to guide next-scale generation. Benefiting from the use of hierarchical trees, the generation process requires only \(\mathcal{O}(\log n)\) steps, where \(n\) is the number of points. Furthermore, we propose a tree-based transformer to predict the tree structure auto-regressively, allowing leaf nodes to attend to their internal ancestors to enhance structural consistency. Extensive experiments demonstrate that our approach effectively generates multi-scale Gaussian representations with controllable levels of detail, visual fidelity, and a manageable time consumption budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18739v2">Moving Light Adaptive Colonoscopy Reconstruction via Illumination-Attenuation-Aware 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-01
      | 💬 Accepted by ICME2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables real-time view synthesis in colonoscopy but assumes static illumination, making it incompatible with the strong photometric variations caused by the moving light source and camera. This mismatch leads existing methods to compensate for illumination attenuation with structure-violating Gaussians, degrading geometric fidelity. Prior work considers only distance-based attenuation and overlooks the physical characteristics of colonscopic lighting. In this paper, we propose ColIAGS, an improved 3DGS framework for colonoscopy. To mimic realistic appearance under varying illumination, we introduce a lighting model with two types of illumination attenuation factors. To satisfy this lighting model's approximation and effectively integrate it into the 3DGS framework, we design Improved Geometry Modeling to strengthen geometry details and Improved Appearance Modeling to implicitly optimize illumination attenuation solutions. Experimental results on standard benchmarks demonstrate that ColIAGS supports both high-quality novel-view synthesis and accurate geometry reconstruction, outperforming state-of-the-art methods in rendering fidelity and Depth MSE. Our code is available at https://github.com/haowang020110/ColIAGS.
    </details>
</div>
