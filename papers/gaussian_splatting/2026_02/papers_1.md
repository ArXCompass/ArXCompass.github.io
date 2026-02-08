# gaussian splatting - 2026_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05822v1">NVS-HO: A Benchmark for Novel View Synthesis of Handheld Objects</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      We propose NVS-HO, the first benchmark designed for novel view synthesis of handheld objects in real-world environments using only RGB inputs. Each object is recorded in two complementary RGB sequences: (1) a handheld sequence, where the object is manipulated in front of a static camera, and (2) a board sequence, where the object is fixed on a ChArUco board to provide accurate camera poses via marker detection. The goal of NVS-HO is to learn a NVS model that captures the full appearance of an object from (1), whereas (2) provides the ground-truth images used for evaluation. To establish baselines, we consider both a classical SfM pipeline and a state-of-the-art pre-trained feed-forward neural network (VGGT) as pose estimators, and train NVS models based on NeRF and Gaussian Splatting. Our experiments reveal significant performance gaps in current methods under unconstrained handheld conditions, highlighting the need for more robust approaches. NVS-HO thus offers a challenging real-world benchmark to drive progress in RGB-based novel view synthesis of handheld objects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02989v2">SharpTimeGS: Sharp and Stable Dynamic Gaussian Splatting via Lifespan Modulation</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Novel view synthesis of dynamic scenes is fundamental to achieving photorealistic 4D reconstruction and immersive visual experiences. Recent progress in Gaussian-based representations has significantly improved real-time rendering quality, yet existing methods still struggle to maintain a balance between long-term static and short-term dynamic regions in both representation and optimization. To address this, we present SharpTimeGS, a lifespan-aware 4D Gaussian framework that achieves temporally adaptive modeling of both static and dynamic regions under a unified representation. Specifically, we introduce a learnable lifespan parameter that reformulates temporal visibility from a Gaussian-shaped decay into a flat-top profile, allowing primitives to remain consistently active over their intended duration and avoiding redundant densification. In addition, the learned lifespan modulates each primitives' motion, reducing drift in long-lived static points while retaining unrestricted motion for short-lived dynamic ones. This effectively decouples motion magnitude from temporal duration, improving long-term stability without compromising dynamic fidelity. Moreover, we design a lifespan-velocity-aware densification strategy that mitigates optimization imbalance between static and dynamic regions by allocating more capacity to regions with pronounced motion while keeping static areas compact and stable. Extensive experiments on multiple benchmarks demonstrate that our method achieves state-of-the-art performance while supporting real-time rendering up to 4K resolution at 100 FPS on one RTX 4090.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17059v3">REArtGS++: Generalizable Articulation Reconstruction with Temporal Geometry Constraint via Planar Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Articulated objects are pervasive in daily environments, such as drawers and refrigerators. Towards their part-level surface reconstruction and joint parameter estimation, REArtGS introduces a category-agnostic approach using multi-view RGB images at two different states. However, we observe that REArtGS still struggles with screw-joint or multi-part objects and lacks geometric constraints for unseen states. In this paper, we propose REArtGS++, a novel method towards generalizable articulated object reconstruction with temporal geometry constraint and planar Gaussian splatting. We first model a decoupled screw motion for each joint without type prior, and jointly optimize part-aware Gaussians with joint parameters through part motion blending. To introduce time-continuous geometric constraint for articulated modeling, we encourage Gaussians to be planar and propose a temporally consistent regularization between planar normal and depth through Taylor first-order expansion. Extensive experiments on both synthetic and real-world articulated objects demonstrate our superiority in generalizable part-level surface reconstruction and joint parameter estimation, compared to existing approaches. Project Site: https://sites.google.com/view/reartgs2/home.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.12788v2">Efficient Scene Modeling via Structure-Aware and Region-Prioritized 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Reconstructing 3D scenes with high fidelity and efficiency remains a central pursuit in computer vision and graphics. Recent advances in 3D Gaussian Splatting (3DGS) enable photorealistic rendering with Gaussian primitives, yet the modeling process remains governed predominantly by photometric supervision. This reliance often leads to irregular spatial distribution and indiscriminate primitive adjustments that largely ignore underlying geometric context. In this work, we rethink Gaussian modeling from a geometric standpoint and introduce Mini-Splatting2, an efficient scene modeling framework that couples structure-aware distribution and region-prioritized optimization, driving 3DGS into a geometry-regulated paradigm. The structure-aware distribution enforces spatial regularity through structured reorganization and representation sparsity, ensuring balanced structural coverage for compact organization. The region-prioritized optimization improves training discrimination through geometric saliency and computational selectivity, fostering appropriate structural emergence for fast convergence. These mechanisms alleviate the long-standing tension among representation compactness, convergence acceleration, and rendering fidelity. Extensive experiments demonstrate that Mini-Splatting2 achieves up to 4$\times$ fewer Gaussians and 3$\times$ faster optimization while maintaining state-of-the-art visual quality, paving the way towards structured and efficient 3D Gaussian modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.02103v3">MVGS: Multi-view Regulated Gaussian Splatting for Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 Project Page:https://xiaobiaodu.github.io/mvgs-project/
    </div>
    <details class="paper-abstract">
      Recent works in volume rendering, \textit{e.g.} NeRF and 3D Gaussian Splatting (3DGS), significantly advance the rendering quality and efficiency with the help of the learned implicit neural radiance field or 3D Gaussians. Rendering on top of an explicit representation, the vanilla 3DGS and its variants deliver real-time efficiency by optimizing the parametric model with single-view supervision per iteration during training which is adopted from NeRF. Consequently, certain views are overfitted, leading to unsatisfying appearance in novel-view synthesis and imprecise 3D geometries. To solve aforementioned problems, we propose a new 3DGS optimization method embodying four key novel contributions: 1) We transform the conventional single-view training paradigm into a multi-view training strategy. With our proposed multi-view regulation, 3D Gaussian attributes are further optimized without overfitting certain training views. As a general solution, we improve the overall accuracy in a variety of scenarios and different Gaussian variants. 2) Inspired by the benefit introduced by additional views, we further propose a cross-intrinsic guidance scheme, leading to a coarse-to-fine training procedure concerning different resolutions. 3) Built on top of our multi-view regulated training, we further propose a cross-ray densification strategy, densifying more Gaussian kernels in the ray-intersect regions from a selection of views. 4) By further investigating the densification strategy, we found that the effect of densification should be enhanced when certain views are distinct dramatically. As a solution, we propose a novel multi-view augmented densification strategy, where 3D Gaussians are encouraged to get densified to a sufficient number accordingly, resulting in improved reconstruction accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.15281v2">StyleMe3D: Stylization with Disentangled Priors by Multiple Encoders on 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 18 pages; Project page: https://styleme3d.github.io/
    </div>
    <details class="paper-abstract">
      Current 3D Gaussian Splatting stylization approaches are limited in their ability to represent diverse artistic styles, frequently defaulting to low-level texture replacement or yielding semantically inconsistent outputs. In this paper, we introduce StyleMe3D, a novel hierarchical framework that achieves comprehensive, high-fidelity stylization by disentangling multi-level style representations while preserving geometric fidelity. The cornerstone of StyleMe3D is Dynamic Style Score Distillation (DSSD), which harnesses latent priors from a style-aware diffusion model to provide high-level semantic guidance, ensuring robust and expressive style transfer. To further refine this distillation process, we propose a multi-modal alignment strategy using the CLIP latent space: a CLIP-based style stream evaluator (Contrastive Style Descriptor) that enforces middle-level stylistic similarity, and a CLIP-based content stream evaluator (3D Gaussian Quality Assessment) that acts as a global regularizer to mitigate typical GS quality degradation. Finally, a VGG-based Simultaneously Optimized Scale module is integrated to refine fine-grained texture details at the low-level. Extensive experiments demonstrate that our method consistently preserves intricate geometric details and achieves coherent stylistic effects across entire scenes, significantly surpassing state-of-the-art baselines in both qualitative and quantitative evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05190v1">PoseGaussian: Pose-Driven Novel View Synthesis for Robust 3D Human Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      We propose PoseGaussian, a pose-guided Gaussian Splatting framework for high-fidelity human novel view synthesis. Human body pose serves a dual purpose in our design: as a structural prior, it is fused with a color encoder to refine depth estimation; as a temporal cue, it is processed by a dedicated pose encoder to enhance temporal consistency across frames. These components are integrated into a fully differentiable, end-to-end trainable pipeline. Unlike prior works that use pose only as a condition or for warping, PoseGaussian embeds pose signals into both geometric and temporal stages to improve robustness and generalization. It is specifically designed to address challenges inherent in dynamic human scenes, such as articulated motion and severe self-occlusion. Notably, our framework achieves real-time rendering at 100 FPS, maintaining the efficiency of standard Gaussian Splatting pipelines. We validate our approach on ZJU-MoCap, THuman2.0, and in-house datasets, demonstrating state-of-the-art performance in perceptual quality and structural accuracy (PSNR 30.86, SSIM 0.979, LPIPS 0.028).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05047v1">QuantumGS: Quantum Encoding Framework for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Recent advances in neural rendering, particularly 3D Gaussian Splatting (3DGS), have enabled real-time rendering of complex scenes. However, standard 3DGS relies on spherical harmonics, which often struggle to accurately capture high-frequency view-dependent effects such as sharp reflections and transparency. While hybrid approaches like Viewing Direction Gaussian Splatting (VDGS) mitigate this limitation using classical Multi-Layer Perceptrons (MLPs), they remain limited by the expressivity of classical networks in low-parameter regimes. In this paper, we introduce QuantumGS, a novel hybrid framework that integrates Variational Quantum Circuits (VQC) into the Gaussian Splatting pipeline. We propose a unique encoding strategy that maps the viewing direction directly onto the Bloch sphere, leveraging the natural geometry of qubits to represent 3D directional data. By replacing classical color-modulating networks with quantum circuits generated via a hypernetwork or conditioning mechanism, we achieve higher expressivity and better generalization. Source code is available in the supplementary material. Code is available at https://github.com/gwilczynski95/QuantumGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04549v1">Nix and Fix: Targeting 1000x Compression of 3D Gaussian Splatting with Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) revolutionized novel view rendering. Instead of inferring from dense spatial points, as implicit representations do, 3DGS uses sparse Gaussians. This enables real-time performance but increases space requirements, hindering applications such as immersive communication. 3DGS compression emerged as a field aimed at alleviating this issue. While impressive progress has been made, at low rates, compression introduces artifacts that degrade visual quality significantly. We introduce NiFi, a method for extreme 3DGS compression through restoration via artifact-aware, diffusion-based one-step distillation. We show that our method achieves state-of-the-art perceptual quality at extremely low rates, down to 0.1 MB, and towards 1000x rate improvement over 3DGS at comparable perceptual performance. The code will be open-sourced upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04349v1">VecSet-Edit: Unleashing Pre-trained LRM for Mesh Editing from Single Image</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      3D editing has emerged as a critical research area to provide users with flexible control over 3D assets. While current editing approaches predominantly focus on 3D Gaussian Splatting or multi-view images, the direct editing of 3D meshes remains underexplored. Prior attempts, such as VoxHammer, rely on voxel-based representations that suffer from limited resolution and necessitate labor-intensive 3D mask. To address these limitations, we propose \textbf{VecSet-Edit}, the first pipeline that leverages the high-fidelity VecSet Large Reconstruction Model (LRM) as a backbone for mesh editing. Our approach is grounded on a analysis of the spatial properties in VecSet tokens, revealing that token subsets govern distinct geometric regions. Based on this insight, we introduce Mask-guided Token Seeding and Attention-aligned Token Gating strategies to precisely localize target regions using only 2D image conditions. Also, considering the difference between VecSet diffusion process versus voxel we design a Drift-aware Token Pruning to reject geometric outliers during the denoising process. Finally, our Detail-preserving Texture Baking module ensures that we not only preserve the geometric details of original mesh but also the textural information. More details can be found in our project page: https://github.com/BlueDyee/VecSet-Edit/tree/main
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04251v1">Towards Next-Generation SLAM: A Survey on 3DGS-SLAM Focusing on Performance, Robustness, and Future Directions</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Traditional Simultaneous Localization and Mapping (SLAM) systems often face limitations including coarse rendering quality, insufficient recovery of scene details, and poor robustness in dynamic environments. 3D Gaussian Splatting (3DGS), with its efficient explicit representation and high-quality rendering capabilities, offers a new reconstruction paradigm for SLAM. This survey comprehensively reviews key technical approaches for integrating 3DGS with SLAM. We analyze performance optimization of representative methods across four critical dimensions: rendering quality, tracking accuracy, reconstruction speed, and memory consumption, delving into their design principles and breakthroughs. Furthermore, we examine methods for enhancing the robustness of 3DGS-SLAM in complex environments such as motion blur and dynamic environments. Finally, we discuss future challenges and development trends in this area. This survey aims to provide a technical reference for researchers and foster the development of next-generation SLAM systems characterized by high fidelity, efficiency, and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04043v1">AnyStyle: Single-Pass Multimodal Stylization for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      The growing demand for rapid and scalable 3D asset creation has driven interest in feed-forward 3D reconstruction methods, with 3D Gaussian Splatting (3DGS) emerging as an effective scene representation. While recent approaches have demonstrated pose-free reconstruction from unposed image collections, integrating stylization or appearance control into such pipelines remains underexplored. Existing attempts largely rely on image-based conditioning, which limits both controllability and flexibility. In this work, we introduce AnyStyle, a feed-forward 3D reconstruction and stylization framework that enables pose-free, zero-shot stylization through multimodal conditioning. Our method supports both textual and visual style inputs, allowing users to control the scene appearance using natural language descriptions or reference images. We propose a modular stylization architecture that requires only minimal architectural modifications and can be integrated into existing feed-forward 3D reconstruction backbones. Experiments demonstrate that AnyStyle improves style controllability over prior feed-forward stylization methods while preserving high-quality geometric reconstruction. A user study further confirms that AnyStyle achieves superior stylization quality compared to an existing state-of-the-art approach. Repository: https://github.com/joaxkal/AnyStyle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03538v1">Constrained Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      While Dynamic Gaussian Splatting enables high-fidelity 4D reconstruction, its deployment is severely hindered by a fundamental dilemma: unconstrained densification leads to excessive memory consumption incompatible with edge devices, whereas heuristic pruning fails to achieve optimal rendering quality under preset Gaussian budgets. In this work, we propose Constrained Dynamic Gaussian Splatting (CDGS), a novel framework that formulates dynamic scene reconstruction as a budget-constrained optimization problem to enforce a strict, user-defined Gaussian budget during training. Our key insight is to introduce a differentiable budget controller as the core optimization driver. Guided by a multi-modal unified importance score, this controller fuses geometric, motion, and perceptual cues for precise capacity regulation. To maximize the utility of this fixed budget, we further decouple the optimization of static and dynamic elements, employing an adaptive allocation mechanism that dynamically distributes capacity based on motion complexity. Furthermore, we implement a three-phase training strategy to seamlessly integrate these constraints, ensuring precise adherence to the target count. Coupled with a dual-mode hybrid compression scheme, CDGS not only strictly adheres to hardware constraints (error < 2%}) but also pushes the Pareto frontier of rate-distortion performance. Extensive experiments demonstrate that CDGS delivers optimal rendering quality under varying capacity limits, achieving over 3x compression compared to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20348v3">Material-informed Gaussian Splatting for 3D World Reconstruction in a Digital Twin</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 8 pages, 5 figures. Accepted to IEEE Intelligent Vehicles Symposium (IV) 2026. Revised version (v3) presents camera-ready publication
    </div>
    <details class="paper-abstract">
      3D reconstruction for Digital Twins often relies on LiDAR-based methods, which provide accurate geometry but lack the semantics and textures naturally captured by cameras. Traditional LiDAR-camera fusion approaches require complex calibration and still struggle with certain materials like glass, which are visible in images but poorly represented in point clouds. We propose a camera-only pipeline that reconstructs scenes using 3D Gaussian Splatting from multi-view images, extracts semantic material masks via vision models, converts Gaussian representations to mesh surfaces with projected material labels, and assigns physics-based material properties for accurate sensor simulation in modern graphics engines and simulators. This approach combines photorealistic reconstruction with physics-based material assignment, providing sensor simulation fidelity comparable to LiDAR-camera fusion while eliminating hardware complexity and calibration requirements. We validate our camera-only method using an internal dataset from an instrumented test vehicle, leveraging LiDAR as ground truth for reflectivity validation alongside image similarity metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03327v1">Pi-GS: Sparse-View Gaussian Splatting with Dense π^3 Initialization</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Novel view synthesis has evolved rapidly, advancing from Neural Radiance Fields to 3D Gaussian Splatting (3DGS), which offers real-time rendering and rapid training without compromising visual fidelity. However, 3DGS relies heavily on accurate camera poses and high-quality point cloud initialization, which are difficult to obtain in sparse-view scenarios. While traditional Structure from Motion (SfM) pipelines often fail in these settings, existing learning-based point estimation alternatives typically require reliable reference views and remain sensitive to pose or depth errors. In this work, we propose a robust method utilizing π^3, a reference-free point cloud estimation network. We integrate dense initialization from π^3 with a regularization scheme designed to mitigate geometric inaccuracies. Specifically, we employ uncertainty-guided depth supervision, normal consistency loss, and depth warping. Experimental results demonstrate that our approach achieves state-of-the-art performance on the Tanks and Temples, LLFF, DTU, and MipNeRF360 datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02000v2">SurfSplat: Conquering Feedforward 2D Gaussian Splatting with Surface Continuity Priors</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 ICLR 2026; Project Page: https://hebing-sjtu.github.io/SurfSplat-website/
    </div>
    <details class="paper-abstract">
      Reconstructing 3D scenes from sparse images remains a challenging task due to the difficulty of recovering accurate geometry and texture without optimization. Recent approaches leverage generalizable models to generate 3D scenes using 3D Gaussian Splatting (3DGS) primitive. However, they often fail to produce continuous surfaces and instead yield discrete, color-biased point clouds that appear plausible at normal resolution but reveal severe artifacts under close-up views. To address this issue, we present SurfSplat, a feedforward framework based on 2D Gaussian Splatting (2DGS) primitive, which provides stronger anisotropy and higher geometric precision. By incorporating a surface continuity prior and a forced alpha blending strategy, SurfSplat reconstructs coherent geometry together with faithful textures. Furthermore, we introduce High-Resolution Rendering Consistency (HRRC), a new evaluation metric designed to evaluate high-resolution reconstruction quality. Extensive experiments on RealEstate10K, DL3DV, and ScanNet demonstrate that SurfSplat consistently outperforms prior methods on both standard metrics and HRRC, establishing a robust solution for high-fidelity 3D reconstruction from sparse inputs. Project page: https://hebing-sjtu.github.io/SurfSplat-website/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03207v1">WebSplatter: Enabling Cross-Device Efficient Gaussian Splatting in Web Browsers via WebGPU</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      We present WebSplatter, an end-to-end GPU rendering pipeline for the heterogeneous web ecosystem. Unlike naive ports, WebSplatter introduces a wait-free hierarchical radix sort that circumvents the lack of global atomics in WebGPU, ensuring deterministic execution across diverse hardware. Furthermore, we propose an opacity-aware geometry culling stage that dynamically prunes splats before rasterization, significantly reducing overdraw and peak memory footprint. Evaluation demonstrates that WebSplatter consistently achieves 1.2$\times$ to 4.5$\times$ speedups over state-of-the-art web viewers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02402v1">SoMA: A Real-to-Sim Neural Simulator for Robotic Soft-body Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-02-02
      | 💬 Project page: https://city-super.github.io/SoMA/
    </div>
    <details class="paper-abstract">
      Simulating deformable objects under rich interactions remains a fundamental challenge for real-to-sim robot manipulation, with dynamics jointly driven by environmental effects and robot actions. Existing simulators rely on predefined physics or data-driven dynamics without robot-conditioned control, limiting accuracy, stability, and generalization. This paper presents SoMA, a 3D Gaussian Splat simulator for soft-body manipulation. SoMA couples deformable dynamics, environmental forces, and robot joint actions in a unified latent neural space for end-to-end real-to-sim simulation. Modeling interactions over learned Gaussian splats enables controllable, stable long-horizon manipulation and generalization beyond observed trajectories without predefined physical models. SoMA improves resimulation accuracy and generalization on real-world robot manipulation by 20%, enabling stable simulation of complex tasks such as long-horizon cloth folding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.17982v3">HI-SLAM2: Geometry-Aware Gaussian SLAM for Fast Monocular Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-02
    </div>
    <details class="paper-abstract">
      We present HI-SLAM2, a geometry-aware Gaussian SLAM system that achieves fast and accurate monocular scene reconstruction using only RGB input. Existing Neural SLAM or 3DGS-based SLAM methods often trade off between rendering quality and geometry accuracy, our research demonstrates that both can be achieved simultaneously with RGB input alone. The key idea of our approach is to enhance the ability for geometry estimation by combining easy-to-obtain monocular priors with learning-based dense SLAM, and then using 3D Gaussian splatting as our core map representation to efficiently model the scene. Upon loop closure, our method ensures on-the-fly global consistency through efficient pose graph bundle adjustment and instant map updates by explicitly deforming the 3D Gaussian units based on anchored keyframe updates. Furthermore, we introduce a grid-based scale alignment strategy to maintain improved scale consistency in prior depths for finer depth details. Through extensive experiments on Replica, ScanNet, and ScanNet++, we demonstrate significant improvements over existing Neural SLAM methods and even surpass RGB-D-based methods in both reconstruction and rendering quality. The project page and source code will be made available at https://hi-slam2.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03878v1">Intellectual Property Protection for 3D Gaussian Splatting Assets: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-02-02
      | 💬 A collection of relevant papers is summarized and will be continuously updated at \url{https://github.com/tmllab/Awesome-3DGS-IP-Protection}
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a mainstream representation for real-time 3D scene synthesis, enabling applications in virtual and augmented reality, robotics, and 3D content creation. Its rising commercial value and explicit parametric structure raise emerging intellectual property (IP) protection concerns, prompting a surge of research on 3DGS IP protection. However, current progress remains fragmented, lacking a unified view of the underlying mechanisms, protection paradigms, and robustness challenges. To address this gap, we present the first systematic survey on 3DGS IP protection and introduce a bottom-up framework that examines (i) underlying Gaussian-based perturbation mechanisms, (ii) passive and active protection paradigms, and (iii) robustness threats under emerging generative AI era, revealing gaps in technical foundations and robustness characterization and indicating opportunities for deeper investigation. Finally, we outline six research directions across robustness, efficiency, and protection paradigms, offering a roadmap toward reliable and trustworthy IP protection for 3DGS assets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02089v1">UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-02
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) enables high-quality, real-time rendering for bounded scenes, its extension to large-scale urban environments gives rise to critical challenges in terms of geometric consistency, memory efficiency, and computational scalability. To address these issues, we present UrbanGS, a scalable reconstruction framework that effectively tackles these challenges for city-scale applications. First, we propose a Depth-Consistent D-Normal Regularization module. Unlike existing approaches that rely solely on monocular normal estimators, which can effectively update rotation parameters yet struggle to update position parameters, our method integrates D-Normal constraints with external depth supervision. This allows for comprehensive updates of all geometric parameters. By further incorporating an adaptive confidence weighting mechanism based on gradient consistency and inverse depth deviation, our approach significantly enhances multi-view depth alignment and geometric coherence, which effectively resolves the issue of geometric accuracy in complex large-scale scenes. To improve scalability, we introduce a Spatially Adaptive Gaussian Pruning (SAGP) strategy, which dynamically adjusts Gaussian density based on local geometric complexity and visibility to reduce redundancy. Additionally, a unified partitioning and view assignment scheme is designed to eliminate boundary artifacts and optimize computational load. Extensive experiments on multiple urban datasets demonstrate that UrbanGS achieves superior performance in rendering quality, geometric accuracy, and memory efficiency, providing a systematic solution for high-fidelity large-scale scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.09606v2">Feat2GS: Probing Visual Foundation Models with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-02
      | 💬 Project Page: https://fanegg.github.io/Feat2GS/
    </div>
    <details class="paper-abstract">
      Given that visual foundation models (VFMs) are trained on extensive datasets but often limited to 2D images, a natural question arises: how well do they understand the 3D world? With the differences in architecture and training protocols (i.e., objectives, proxy tasks), a unified framework to fairly and comprehensively probe their 3D awareness is urgently needed. Existing works on 3D probing suggest single-view 2.5D estimation (e.g., depth and normal) or two-view sparse 2D correspondence (e.g., matching and tracking). Unfortunately, these tasks ignore texture awareness, and require 3D data as ground-truth, which limits the scale and diversity of their evaluation set. To address these issues, we introduce Feat2GS, which readout 3D Gaussians attributes from VFM features extracted from unposed images. This allows us to probe 3D awareness for geometry and texture via novel view synthesis, without requiring 3D data. Additionally, the disentanglement of 3DGS parameters - geometry ($\boldsymbol{x}$, $α$, $Σ$) and texture ($\boldsymbol{c}$) - enables separate analysis of texture and geometry awareness. Under Feat2GS, we conduct extensive experiments to probe the 3D awareness of several VFMs, and investigate the ingredients that lead to a 3D aware VFM. Building on these findings, we develop several variants that achieve state-of-the-art across diverse datasets. This makes Feat2GS useful for probing VFMs, and as a simple-yet-effective baseline for novel-view synthesis. Code and data are available at https://fanegg.github.io/Feat2GS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01844v1">CloDS: Visual-Only Unsupervised Cloth Dynamics Learning in Unknown Conditions</a></div>
    <div class="paper-meta">
      📅 2026-02-02
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Deep learning has demonstrated remarkable capabilities in simulating complex dynamic systems. However, existing methods require known physical properties as supervision or inputs, limiting their applicability under unknown conditions. To explore this challenge, we introduce Cloth Dynamics Grounding (CDG), a novel scenario for unsupervised learning of cloth dynamics from multi-view visual observations. We further propose Cloth Dynamics Splatting (CloDS), an unsupervised dynamic learning framework designed for CDG. CloDS adopts a three-stage pipeline that first performs video-to-geometry grounding and then trains a dynamics model on the grounded meshes. To cope with large non-linear deformations and severe self-occlusions during grounding, we introduce a dual-position opacity modulation that supports bidirectional mapping between 2D observations and 3D geometry via mesh-based Gaussian splatting in video-to-geometry grounding stage. It jointly considers the absolute and relative position of Gaussian components. Comprehensive experimental evaluations demonstrate that CloDS effectively learns cloth dynamics from visual data while maintaining strong generalization capabilities for unseen configurations. Our code is available at https://github.com/whynot-zyl/CloDS. Visualization results are available at https://github.com/whynot-zyl/CloDS_video}.%\footnote{As in this example.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.14552v2">From Slices to Structures: Unsupervised 3D Reconstruction of Female Pelvic Anatomy from Freehand Transvaginal Ultrasound</a></div>
    <div class="paper-meta">
      📅 2026-02-02
    </div>
    <details class="paper-abstract">
      Volumetric ultrasound has the potential to significantly improve diagnostic accuracy and clinical decision-making, yet its widespread adoption remains limited by dependence on specialized hardware and restrictive acquisition protocols. In this work, we present a novel unsupervised framework for reconstructing 3D anatomical structures from freehand 2D transvaginal ultrasound sweeps, without requiring external tracking or learned pose estimators. Our method, TVGS, adapts the principles of Gaussian Splatting to the domain of ultrasound, introducing a slice-aware, differentiable rasterizer tailored to the unique physics and geometry of ultrasound imaging. We model anatomy as a collection of anisotropic 3D Gaussians and optimize their parameters directly from image-level supervision. To ensure robustness against irregular probe motion, we introduce a joint optimization scheme that refines slice poses alongside anatomical structure. The result is a compact, flexible, and memory-efficient volumetric representation that captures anatomical detail with high spatial fidelity. This work demonstrates that accurate 3D reconstruction from 2D ultrasound images can be achieved through purely computational means, offering a scalable alternative to conventional 3D systems and enabling new opportunities for AI-assisted analysis and diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01723v1">FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-02
    </div>
    <details class="paper-abstract">
      Extending 3D Gaussian Splatting (3DGS) to 4D physical simulation remains challenging. Based on the Material Point Method (MPM), existing methods either rely on manual parameter tuning or distill dynamics from video diffusion models, limiting the generalization and optimization efficiency. Recent attempts using LLMs/VLMs suffer from a text/image-to-3D perceptual gap, yielding unstable physics behavior. In addition, they often ignore the surface structure of 3DGS, leading to implausible motion. We propose FastPhysGS, a fast and robust framework for physics-based dynamic 3DGS simulation:(1) Instance-aware Particle Filling (IPF) with Monte Carlo Importance Sampling (MCIS) to efficiently populate interior particles while preserving geometric fidelity; (2) Bidirectional Graph Decoupling Optimization (BGDO), an adaptive strategy that rapidly optimizes material parameters predicted from a VLM. Experiments show FastPhysGS achieves high-fidelity physical simulation in 1 minute using only 7 GB runtime memory, outperforming prior works with broad potential applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01674v1">VRGaussianAvatar: Integrating 3D Gaussian Avatars into VR</a></div>
    <div class="paper-meta">
      📅 2026-02-02
      | 💬 Accepted as an IEEE TVCG paper at IEEE VR 2026 (journal track)
    </div>
    <details class="paper-abstract">
      We present VRGaussianAvatar, an integrated system that enables real-time full-body 3D Gaussian Splatting (3DGS) avatars in virtual reality using only head-mounted display (HMD) tracking signals. The system adopts a parallel pipeline with a VR Frontend and a GA Backend. The VR Frontend uses inverse kinematics to estimate full-body pose and streams the resulting pose along with stereo camera parameters to the backend. The GA Backend stereoscopically renders a 3DGS avatar reconstructed from a single image. To improve stereo rendering efficiency, we introduce Binocular Batching, which jointly processes left and right eye views in a single batched pass to reduce redundant computation and support high-resolution VR displays. We evaluate VRGaussianAvatar with quantitative performance tests and a within-subject user study against image- and video-based mesh avatar baselines. Results show that VRGaussianAvatar sustains interactive VR performance and yields higher perceived appearance similarity, embodiment, and plausibility. Project page and source code are available at https://vrgaussianavatar.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01513v1">MarkCleaner: High-Fidelity Watermark Removal via Imperceptible Micro-Geometric Perturbation</a></div>
    <div class="paper-meta">
      📅 2026-02-02
    </div>
    <details class="paper-abstract">
      Semantic watermarks exhibit strong robustness against conventional image-space attacks. In this work, we show that such robustness does not survive under micro-geometric perturbations: spatial displacements can remove watermarks by breaking the phase alignment. Motivated by this observation, we introduce MarkCleaner, a watermark removal framework that avoids semantic drift caused by regeneration-based watermark removal. Specifically, MarkCleaner is trained with micro-geometry-perturbed supervision, which encourages the model to separate semantic content from strict spatial alignment and enables robust reconstruction under subtle geometric displacements. The framework adopts a mask-guided encoder that learns explicit spatial representations and a 2D Gaussian Splatting-based decoder that explicitly parameterizes geometric perturbations while preserving semantic content. Extensive experiments demonstrate that MarkCleaner achieves superior performance in both watermark removal effectiveness and visual fidelity, while enabling efficient real-time inference. Our code will be made available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02602v1">Position: 3D Gaussian Splatting Watermarking Should Be Scenario-Driven and Threat-Model Explicit</a></div>
    <div class="paper-meta">
      📅 2026-02-01
    </div>
    <details class="paper-abstract">
      3D content acquisition and creation are expanding rapidly in the new era of machine learning and AI. 3D Gaussian Splatting (3DGS) has become a promising high-fidelity and real-time representation for 3D content. Similar to the initial wave of digital audio-visual content at the turn of the millennium, the demand for intellectual property protection is also increasing, since explicit and editable 3D parameterization makes unauthorized use and dissemination easier. In this position paper, we argue that effective progress in watermarking 3D assets requires articulated security objectives and realistic threat models, incorporating the lessons learned from digital audio-visual asset protection over the past decades. To address this gap in security specification and evaluation, we advocate a scenario-driven formulation, in which adversarial capabilities are formalized through a security model. Based on this formulation, we construct a reference framework that organizes existing methods and clarifies how specific design choices map to corresponding adversarial assumptions. Within this framework, we also examine a legacy spread-spectrum embedding scheme, characterizing its advantages and limitations and highlighting the important trade-offs it entails. Overall, this work aims to foster effective intellectual property protection for 3D assets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03809v1">Split&Splat: Zero-Shot Panoptic Segmentation via Explicit Instance Modeling and 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-01
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (GS) enables fast and high-quality scene reconstruction, but it lacks an object-consistent and semantically aware structure. We propose Split&Splat, a framework for panoptic scene reconstruction using 3DGS. Our approach explicitly models object instances. It first propagates instance masks across views using depth, thus producing view-consistent 2D masks. Each object is then reconstructed independently and merged back into the scene while refining its boundaries. Finally, instance-level semantic descriptors are embedded in the reconstructed objects, supporting various applications, including panoptic segmentation, object retrieval, and 3D editing. Unlike existing methods, Split&Splat tackles the problem by first segmenting the scene and then reconstructing each object individually. This design naturally supports downstream tasks and allows Split&Splat to achieve state-of-the-art performance on the ScanNetv2 segmentation benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01057v1">Radioactive 3D Gaussian Ray Tracing for Tomographic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-01
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged in computer vision as a promising rendering technique. By adapting the principles of Elliptical Weighted Average (EWA) splatting to a modern differentiable pipeline, 3DGS enables real-time, high-quality novel view synthesis. Building upon this, R2-Gaussian extended the 3DGS paradigm to tomographic reconstruction by rectifying integration bias, achieving state-of-the-art performance in computed tomography (CT). To enable differentiability, R2-Gaussian adopts a local affine approximation: each 3D Gaussian is locally mapped to a 2D Gaussian on the detector and composed via alpha blending to form projections. However, the affine approximation can degrade reconstruction quantitative accuracy and complicate the incorporation of nonlinear geometric corrections. To address these limitations, we propose a tomographic reconstruction framework based on 3D Gaussian ray tracing. Our approach provides two key advantages over splatting-based models: (i) it computes the line integral through 3D Gaussian primitives analytically, avoiding the local affine collapse and thus yielding a more physically consistent forward projection model; and (ii) the ray-tracing formulation gives explicit control over ray origins and directions, which facilitates the precise application of nonlinear geometric corrections, e.g., arc-correction used in positron emission tomography (PET). These properties extend the applicability of Gaussian-based reconstruction to a wider range of realistic tomography systems while improving projection accuracy.
    </details>
</div>
