# gaussian splatting - 2026_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16806v4">MedGS: Gaussian Splatting for Multi-Modal 3D Medical Imaging</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Endoluminal endoscopic procedures are essential for diagnosing colorectal cancer and other severe conditions in the digestive tract, urogenital system, and airways. 3D reconstruction and novel-view synthesis from endoscopic images are promising tools for enhancing diagnosis. Moreover, integrating physiological deformations and interaction with the endoscope enables the development of simulation tools from real video data. However, constrained camera trajectories and view-dependent lighting create artifacts, leading to inaccurate or overfitted reconstructions. We present MedGS, a novel 3D reconstruction framework leveraging the unique property of endoscopic imaging, where a single light source is closely aligned with the camera. Our method separates light effects from tissue properties. MedGS enhances 3D Gaussian Splatting with a physically based relightable model. We boost the traditional light transport formulation with a specialized MLP capturing complex light-related effects while ensuring reduced artifacts and better generalization across novel views. MedGS achieves superior reconstruction quality compared to baseline methods on both public and in-house datasets. Unlike existing approaches, MedGS enables tissue modifications while preserving a physically accurate response to light, making it closer to real-world clinical use. Repository: https://github.com/gmum/MedGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12626v1">Habitat-GS: A High-Fidelity Navigation Simulator with Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Project page: https://zju3dv.github.io/habitat-gs/
    </div>
    <details class="paper-abstract">
      Training embodied AI agents depends critically on the visual fidelity of simulation environments and the ability to model dynamic humans. Current simulators rely on mesh-based rasterization with limited visual realism, and their support for dynamic human avatars, where available, is constrained to mesh representations, hindering agent generalization to human-populated real-world scenarios. We present Habitat-GS, a navigation-centric embodied AI simulator extended from Habitat-Sim that integrates 3D Gaussian Splatting scene rendering and drivable gaussian avatars while maintaining full compatibility with the Habitat ecosystem. Our system implements a 3DGS renderer for real-time photorealistic rendering and supports scalable 3DGS asset import from diverse sources. For dynamic human modeling, we introduce a gaussian avatar module that enables each avatar to simultaneously serve as a photorealistic visual entity and an effective navigation obstacle, allowing agents to learn human-aware behaviors in realistic settings. Experiments on point-goal navigation demonstrate that agents trained on 3DGS scenes achieve stronger cross-domain generalization, with mixed-domain training being the most effective strategy. Evaluations on avatar-aware navigation further confirm that gaussian avatars enable effective human-aware navigation. Finally, performance benchmarks validate the system's scalability across varying scene complexity and avatar counts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10578v2">Rein3D: Reinforced 3D Indoor Scene Generation with Panoramic Video Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      The growing demand for Embodied AI and VR applications has highlighted the need for synthesizing high-quality 3D indoor scenes from sparse inputs. However, existing approaches struggle to infer massive amounts of missing geometry in large unseen areas while maintaining global consistency, often producing locally plausible but globally inconsistent reconstructions. We present Rein3D, a framework that reconstructs full 360-degree indoor environments by coupling explicit 3D Gaussian Splatting (3DGS) with temporally coherent priors from video diffusion models. Our approach follows a "restore-and-refine" paradigm: we employ a radial exploration strategy to render imperfect panoramic videos along trajectories starting from the origin, effectively uncovering occluded regions from a coarse 3DGS initialization. These sequences are restored by a panoramic video-to-video diffusion model and further enhanced via video super-resolution to synthesize high-fidelity geometry and textures. Finally, these refined videos serve as pseudo-ground truths to update the global 3D Gaussian field. To support this task, we construct PanoV2V-15K, a dataset of over 15K paired clean and degraded panoramic videos for diffusion-based scene restoration. Experiments demonstrate that Rein3D produces photorealistic and globally consistent 3D scenes and significantly improves long-range camera exploration compared with existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12592v1">ELoG-GS: Dual-Branch Gaussian Splatting with Luminance-Guided Enhancement for Extreme Low-light 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Our method achieved a ranking of 9 out of 148 participants in Track 1 of the NTIRE 3DRR Challenge, as reported on the official competition website: https://www.codabench.org/competitions/13854/
    </div>
    <details class="paper-abstract">
      This paper presents our approach to the NTIRE 2026 3D Restoration and Reconstruction Challenge (Track 1), which focuses on reconstructing high-quality 3D representations from degraded multi-view inputs. The challenge involves recovering geometrically consistent and photorealistic 3D scenes in extreme low-light environments. To address this task, we propose Extreme Low-light Optimized Gaussian Splatting (ELoG-GS), a robust low-light 3D reconstruction pipeline that integrates learning-based point cloud initialization and luminance-guided color enhancement for stable and photorealistic Gaussian Splatting. Our method incorporates both geometry-aware initialization and photometric adaptation strategies to improve reconstruction fidelity under challenging conditions. Extensive experiments on the NTIRE Track 1 benchmark demonstrate that our approach significantly improves reconstruction quality over the baselines, achieving superior visual fidelity and geometric consistency. The proposed method provides a practical solution for robust 3D reconstruction in real-world degraded scenarios. In the final testing phase, our method achieved a PSNR of 18.6626 and an SSIM of 0.6855 on the official platform leaderboard. Code is available at https://github.com/lyh120/FSGS_EAPGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15355v3">DAV-GSWT: Diffusion-Active-View Sampling for Data-Efficient Gaussian Splatting Wang Tiles</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 16 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The emergence of 3D Gaussian Splatting has fundamentally redefined the capabilities of photorealistic neural rendering by enabling high-throughput synthesis of complex environments. While procedural methods like Wang Tiles have recently been integrated to facilitate the generation of expansive landscapes, these systems typically remain constrained by a reliance on densely sampled exemplar reconstructions. We present DAV-GSWT, a data-efficient framework that leverages diffusion priors and active view sampling to synthesize high-fidelity Gaussian Splatting Wang Tiles from minimal input observations. By integrating a hierarchical uncertainty quantification mechanism with generative diffusion models, our approach autonomously identifies the most informative viewpoints while hallucinating missing structural details to ensure seamless tile transitions. Experimental results indicate that our system significantly reduces the required data volume while maintaining the visual integrity and interactive performance necessary for large-scale virtual environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08410v2">BLaDA: Bridging Language to Functional Dexterous Actions within 3DGS Fields</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Code will be publicly available at https://github.com/PopeyePxx/BLaDA
    </div>
    <details class="paper-abstract">
      In unstructured environments, functional dexterous grasping calls for the tight integration of semantic understanding, precise 3D functional localization, and physically interpretable execution. Modular hierarchical methods are more controllable and interpretable than end-to-end VLA approaches, but existing ones still rely on predefined affordance labels and lack the tight semantic--pose coupling needed for functional dexterous manipulation. To address this, we propose BLaDA (Bridging Language to Dexterous Actions in 3DGS fields), an interpretable zero-shot framework that grounds open-vocabulary instructions as perceptual and control constraints for functional dexterous manipulation. BLaDA establishes an interpretable reasoning chain by first parsing natural language into a structured sextuple of manipulation constraints via a Knowledge-guided Language Parsing (KLP) module. To achieve pose-consistent spatial reasoning, we introduce the Triangular Functional Point Localization (TriLocation) module, which utilizes 3D Gaussian Splatting as a continuous scene representation and identifies functional regions under triangular geometric constraints. Finally, the 3D Keypoint Grasp Matrix Transformation Execution (KGT3D+) module decodes these semantic-geometric constraints into physically plausible wrist poses and finger-level commands. Extensive experiments on complex benchmarks demonstrate that BLaDA significantly outperforms existing methods in both affordance grounding precision and the success rate of functional manipulation across diverse categories and tasks. Code will be publicly available at https://github.com/PopeyePxx/BLaDA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12251v1">ArtifactWorld: Scaling 3D Gaussian Splatting Artifact Restoration via Video Generation Models</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 The second author is the corresponding author
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) delivers high-fidelity real-time rendering but suffers from geometric and photometric degradations under sparse-view constraints. Current generative restoration approaches are often limited by insufficient temporal coherence, a lack of explicit spatial constraints, and a lack of large-scale training data, resulting in multi-view inconsistencies, erroneous geometric hallucinations, and limited generalization to diverse real-world artifact distributions. In this paper, we present ArtifactWorld, a framework that resolves 3DGS artifact repair through systematic data expansion and a homogeneous dual-model paradigm. To address the data bottleneck, we establish a fine-grained phenomenological taxonomy of 3DGS artifacts and construct a comprehensive training set of 107.5K diverse paired video clips to enhance model robustness. Architecturally, we unify the restoration process within a video diffusion backbone, utilizing an isomorphic predictor to localize structural defects via an artifact heatmap. This heatmap then guides the restoration through an Artifact-Aware Triplet Fusion mechanism, enabling precise, intensity-guided spatio-temporal repair within native self-attention. Extensive experiments demonstrate that ArtifactWorld achieves state-of-the-art performance in sparse novel view synthesis and robust 3D reconstruction. Code and dataset will be made public.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10910v2">STGV: Spatio-Temporal Hash Encoding for Gaussian-based Video Representation</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      2D Gaussian Splatting (2DGS) has recently become a promising paradigm for high-quality video representation. However, existing methods employ content-agnostic or spatio-temporal feature overlapping embeddings to predict canonical Gaussian primitive deformations, which entangles static and dynamic components in videos and prevents modeling their distinct properties effectively. These result in inaccurate predictions for spatio-temporal deformations and unsatisfactory representation quality. To address these problems, this paper proposes a Spatio-Temporal hash encoding framework for Gaussian-based Video representation (STGV). By decomposing video features into learnable 2D spatial and 3D temporal hash encodings, STGV effectively facilitates the learning of motion patterns for dynamic components while maintaining background details for static elements. In addition, we construct a more stable and consistent initial canonical Gaussian representation through a key frame canonical initialization strategy, preventing from feature overlapping and a structurally incoherent geometry representation. Experimental results demonstrate that our method attains better video representation quality (+0.98 PSNR) against other Gaussian-based methods and achieves competitive performance in downstream video tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12217v1">VVGT: Visual Volume-Grounded Transformer</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Volumetric visualization has long been dominated by Direct Volume Rendering (DVR), which operates on dense voxel grids and suffers from limited scalability as resolution and interactivity demands increase. Recent advances in 3D Gaussian Splatting (3DGS) offer a representation-centric alternative; however, existing volumetric extensions still depend on costly per-scene optimization, limiting scalability and interactivity. We present VVGT (Visual Volume-Grounded Transformer), a feed-forward, representation-first framework that directly maps volumetric data to a 3D Gaussian Splatting representation, advancing a new paradigm for volumetric visualization beyond DVR. Unlike prior feed-forward 3DGS methods designed for surface-centric reconstruction, VVGT explicitly accounts for volumetric rendering, where each pixel aggregates contributions along a ray. VVGT employs a dual-transformer network and introduces Volume Geometry Forcing, an epipolar cross-attention mechanism that integrates multi-view observations into distributed 3D Gaussian primitives without surface assumptions. This design eliminates per-scene optimization while enabling accurate volumetric representations. Extensive experiments show that VVGT achieves high-quality visualization with orders-of-magnitude faster conversion, improved geometric consistency, and strong zero-shot generalization across diverse datasets, enabling truly interactive and scalable volumetric visualization. The code will be publicly released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11992v1">ReefMapGS: Enabling Large-Scale Underwater Reconstruction by Closing the Loop Between Multimodal SLAM and Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-13
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting is a powerful visual representation, providing high-quality and efficient 3D scene reconstruction, but it is crucially dependent on accurate camera poses typically obtained from computationally intensive processes like structure-from-motion that are unsuitable for field robot applications. However, in these domains, multimodal sensor data from acoustic, inertial, pressure, and visual sensors are available and suitable for pose-graph optimization-based SLAM methods that can estimate the vehicle's trajectory and thus our needed camera poses while providing uncertainty. We propose a 3DGS-based incremental reconstruction framework, ReefMapGS, that builds an initial model from a high certainty region and progressively expands to incorporate the whole scene. We reconstruct the scene incrementally by interleaving local tracking of new image observations with optimization of the underlying 3DGS scene. These refined poses are integrated back into the pose-graph to globally optimize the whole trajectory. We show COLMAP-free 3D reconstruction of two underwater reef sites with complex geometry as well as more accurate global pose estimation of our AUV over survey trajectories spanning up to 700 m.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11685v1">Unfolding 3D Gaussian Splatting via Iterative Gaussian Synopsis</a></div>
    <div class="paper-meta">
      📅 2026-04-13
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a state-of-the-art framework for real-time, high-fidelity novel view synthesis. However, its substantial storage requirements and inherently unstructured representation pose challenges for deployment in streaming and resource-constrained environments. Existing Level-of-Detail (LOD) strategies, particularly those based on bottom-up construction, often introduce redundancy or lead to fidelity degradation. To overcome these limitations, we propose Iterative Gaussian Synopsis, a novel framework for compact and progressive rendering through a top-down "unfolding" scheme. Our approach begins with a full-resolution 3DGS model and iteratively derives coarser LODs using an adaptive, learnable mask-based pruning mechanism. This process constructs a multi-level hierarchy that preserves visual quality while improving efficiency. We integrate hierarchical spatial grids, which capture the global scene structure, with a shared Anchor Codebook that models localized details. This combination produces a compact yet expressive feature representation, designed to minimize redundancy and support efficient, level-specific adaptation. The unfolding mechanism promotes inter-layer reusability and requires only minimal data overhead for progressive refinement. Experiments show that our method maintains high rendering quality across all LODs while achieving substantial storage reduction. These results demonstrate the practicality and scalability of our approach for real-time 3DGS rendering in bandwidth- and memory-constrained scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19172v4">MetroGS: Efficient and Stable Reconstruction of Geometrically Accurate High-Fidelity Large-Scale Scenes</a></div>
    <div class="paper-meta">
      📅 2026-04-13
      | 💬 Accepted by CVPR26; Project page: https://m3phist0.github.io/MetroGS
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting and its derivatives have achieved significant breakthroughs in large-scale scene reconstruction. However, how to efficiently and stably achieve high-quality geometric fidelity remains a core challenge. To address this issue, we introduce MetroGS, a novel Gaussian Splatting framework for efficient and robust reconstruction in complex urban environments. Our method is built upon a distributed 2D Gaussian Splatting representation as the core foundation, serving as a unified backbone for subsequent modules. To handle potential sparse regions in complex scenes, we propose a structured dense enhancement scheme that utilizes SfM priors and a pointmap model to achieve a denser initialization, while incorporating a sparsity compensation mechanism to improve reconstruction completeness. Furthermore, we design a progressive hybrid geometric optimization strategy that organically integrates monocular and multi-view optimization to achieve efficient and accurate geometric refinement. Finally, to address the appearance inconsistency commonly observed in large-scale scenes, we introduce a depth-guided appearance modeling approach that learns spatial features with 3D consistency, facilitating effective decoupling between geometry and appearance and further enhancing reconstruction stability. Experiments on large-scale urban datasets demonstrate that MetroGS achieves superior geometric accuracy, rendering quality, offering a unified solution for high-fidelity large-scale scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11401v1">GS4City: Hierarchical Semantic Gaussian Splatting via City-Model Priors</a></div>
    <div class="paper-meta">
      📅 2026-04-13
    </div>
    <details class="paper-abstract">
      Recent semantic 3D Gaussian Splatting (3DGS) methods primarily rely on 2D foundation models, often yielding ambiguous boundaries and limited support for structured urban semantics. While city models such as CityGML encode hierarchically organized semantics together with building geometry, these labels cannot be directly mapped to Gaussian primitives. We present GS4City, a hierarchical semantic Gaussian Splatting method that incorporates city-model priors for urban scene understanding. GS4City derives reliable image-aligned masks from Level of Detail (LoD) 3 CityGML models via two-pass raycasting, explicitly using parent-child relations to validate and recover fine-grained facade elements. It then fuses these geometry-grounded masks with foundation-model predictions to establish scene-consistent instance correspondences, and learns a compact identity encoding for each Gaussian under joint 2D identity supervision and 3D spatial regularization. Experiments on the TUM2TWIN and Gold Coast datasets show that GS4City effectively incorporates structured building semantics into Gaussian scene representations, outperforming existing 2D-driven semantic 3DGS baselines, including LangSplat and Gaga, by up to 15.8 IoU points in coarse building segmentation and 14.2 mIoU points in fine-grained semantic segmentation. By bridging structured city models and photorealistic Gaussian scene representations, GS4City enables semantically queryable and structure-aware urban reconstruction. Code is available at https://github.com/Jinyzzz/GS4City.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11142v1">Naka-GS: A Bionics-inspired Dual-Branch Naka Correction and Progressive Point Pruning for Low-Light 3DGS</a></div>
    <div class="paper-meta">
      📅 2026-04-13
    </div>
    <details class="paper-abstract">
      Low-light conditions severely hinder 3D restoration and reconstruction by degrading image visibility, introducing color distortions, and contaminating geometric priors for downstream optimization. We present NAKA-GS, a bionics-inspired framework for low-light 3D Gaussian Splatting that jointly improves photometric restoration and geometric initialization. Our method starts with a Naka-guided chroma-correction network, which combines physics-prior low-light enhancement, dual-branch input modeling, frequency-decoupled correction, and mask-guided optimization to suppress bright-region chromatic artifacts and edge-structure errors. The enhanced images are then fed into a feed-forward multi-view reconstruction model to produce dense scene priors. To further improve Gaussian initialization, we introduce a lightweight Point Preprocessing Module (PPM) that performs coordinate alignment, voxel pooling, and distance-adaptive progressive pruning to remove noisy and redundant points while preserving representative structures. Without introducing heavy inference overhead, NAKA-GS improves restoration quality, training stability, and optimization efficiency for low-light 3D reconstruction. The proposed method was presented in the NTIRE 3D Restoration and Reconstruction (3DRR) Challenge, and outperformed the baseline methods by a large margin. The code is available at https://github.com/RunyuZhu/Naka-GS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11138v1">ViserDex: Visual Sim-to-Real for Robust Dexterous In-hand Reorientation</a></div>
    <div class="paper-meta">
      📅 2026-04-13
    </div>
    <details class="paper-abstract">
      In-hand object reorientation requires precise estimation of the object pose to handle complex task dynamics. While RGB sensing offers rich semantic cues for pose tracking, existing solutions rely on multi-camera setups or costly ray tracing. We present a sim-to-real framework for monocular RGB in-hand reorientation that integrates 3D Gaussian Splatting (3DGS) to bridge the visual sim-to-real gap. Our key insight is performing domain randomization in the Gaussian representation space: by applying physically consistent, pre-rendering augmentations to 3D Gaussians, we generate photorealistic, randomized visual data for object pose estimation. The manipulation policy is trained using curriculum-based reinforcement learning with teacher-student distillation, enabling efficient learning of complex behaviors. Importantly, both perception and control models can be trained independently on consumer-grade hardware, eliminating the need for large compute clusters. Experiments show that the pose estimator trained with 3DGS data outperforms those trained using conventional rendering data in challenging visual environments. We validate the system on a physical multi-fingered hand equipped with an RGB camera, demonstrating robust reorientation of five diverse objects even under challenging lighting conditions. Our results highlight Gaussian splatting as a practical path for RGB-only dexterous manipulation. For videos of the hardware deployments and additional supplementary materials, please refer to the project website: https://rffr.leggedrobotics.com/works/viserdex/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12895v2">HDR 3D Gaussian Splatting via Luminance-Chromaticity Decomposition</a></div>
    <div class="paper-meta">
      📅 2026-04-13
    </div>
    <details class="paper-abstract">
      High Dynamic Range (HDR) 3D reconstruction is pivotal for professional content creation in filmmaking and virtual production. Existing methods typically rely on multi-exposure Low Dynamic Range (LDR) supervision to constrain the learning process within vast brightness spaces, resulting in complex, dual-branch architectures. This work explores the feasibility of learning HDR 3D models exclusively in the HDR data space to simplify model design. By analyzing 3D Gaussian Splatting (3DGS) for HDR imagery, we reveal that its failure stems from the limited capacity of Spherical Harmonics (SH) to capture extreme radiance variations across views, often biasing towards high-radiance observations. While increasing SH orders improves training fitting, it leads to severe overfitting and excessive parameter overhead. To address this, we propose \textit{Luminance-Chromaticity Decomposition 3DGS} (LCD-GS). By decoupling luminance and chromaticity into independent parameters, LCD-GS significantly enhances learning flexibility with minimal parameter increase (\textit{e.g.}, one extra scalar per primitive). Notably, LCD-GS maintains the original training and inference pipeline, requiring only a change in color representation. Extensive experiments on synthetic and real datasets demonstrate that LCD-GS consistently outperforms state-of-the-art methods in reconstruction fidelity and dynamic-range preservation even with a simpler, more efficient architecture, providing an elegant paradigm for professional-grade HDR 3D modeling. Code and datasets will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11098v1">Efficient Transceiver Design for Aerial Image Transmission and Large-scale Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-13
      | 💬 6 pages, 6 figures, submitted to IEEE ISIT-w
    </div>
    <details class="paper-abstract">
      Large-scale three-dimensional (3D) scene reconstruction in low-altitude intelligent networks (LAIN) demands highly efficient wireless image transmission. However, existing schemes struggle to balance severe pilot overhead with the transmission accuracy required to maintain reconstruction fidelity. To strike a balance between efficiency and reliability, this paper proposes a novel deep learning-based end-to-end (E2E) transceiver design that integrates 3D Gaussian Splatting (3DGS) directly into the training process. By jointly optimizing the communication modules via the combined 3DGS rendering loss, our approach explicitly improves scene recovery quality. Furthermore, this task-driven framework enables the use of a sparse pilot scheme, significantly reducing transmission overhead while maintaining robust image recovery under low-altitude channel conditions. Extensive experiments on real-world aerial image datasets demonstrate that the proposed E2E design significantly outperforms existing baselines, delivering superior transmission performance and accurate 3D scene reconstructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.11931v2">Dark-EvGS: Event Camera as an Eye for Radiance Field in the Dark</a></div>
    <div class="paper-meta">
      📅 2026-04-13
    </div>
    <details class="paper-abstract">
      In low-light environments, conventional cameras often struggle to capture clear multi-view images of objects due to dynamic range limitations and motion blur caused by long exposure. Event cameras, with their high-dynamic range and high-speed properties, have the potential to mitigate these issues. Additionally, 3D Gaussian Splatting (GS) enables radiance field reconstruction, facilitating bright frame synthesis from multiple viewpoints in low-light conditions. However, naively using an event-assisted 3D GS approach still faced challenges because, in low light, events are noisy, frames lack quality, and the color tone may be inconsistent. To address these issues, we propose Dark-EvGS, the first event-assisted 3D GS framework that enables the reconstruction of bright frames from arbitrary viewpoints along the camera trajectory. Triplet-level supervision is proposed to gain holistic knowledge, granular details, and sharp scene rendering. The color tone matching block is proposed to guarantee the color consistency of the rendered frames. Furthermore, we introduce the first real-captured dataset for the event-guided bright frame synthesis task via 3D GS-based radiance field reconstruction. Experiments demonstrate that our method achieves better results than existing methods, conquering radiance field reconstruction under challenging low-light conditions. The code and sample data are included in the supplementary material.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10994v1">LumiMotion: Improving Gaussian Relighting with Scene Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-04-13
      | 💬 CVPR2026
    </div>
    <details class="paper-abstract">
      In 3D reconstruction, the problem of inverse rendering, namely recovering the illumination of the scene and the material properties, is fundamental. Existing Gaussian Splatting-based methods primarily target static scenes and often assume simplified or moderate lighting to avoid entangling shadows with surface appearance. This limits their ability to accurately separate lighting effects from material properties, particularly in real-world conditions. We address this limitation by leveraging dynamic elements - regions of the scene that undergo motion - as a supervisory signal for inverse rendering. Motion reveals the same surfaces under varying lighting conditions, providing stronger cues for disentangling material and illumination. This thesis is supported by our experimental results which show we improve LPIPS by 23% for albedo estimation and by 15% for scene relighting relative to next-best baseline. To this end, we introduce LumiMotion, the first Gaussian-based approach that leverages dynamics for inverse rendering and operates in arbitrary dynamic scenes. Our method learns a dynamic 2D Gaussian Splatting representation that employs a set of novel constraints which encourage the dynamic regions of the scene to deform, while keeping static regions stable. As we demonstrate, this separation is crucial for correct optimization of the albedo. Finally, we release a new synthetic benchmark comprising five scenes under four lighting conditions, each in both static and dynamic variants, for the first time enabling systematic evaluation of inverse rendering methods in dynamic environments and challenging lighting. Link to project page: https://joaxkal.github.io/LumiMotion/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10982v1">Ψ-Map: Panoptic Surface Integrated Mapping Enables Real2Sim Transfer</a></div>
    <div class="paper-meta">
      📅 2026-04-13
    </div>
    <details class="paper-abstract">
      Open-vocabulary panoptic reconstruction is essential for advanced robotics perception and simulation. However, existing methods based on 3D Gaussian Splatting (3DGS) often struggle to simultaneously achieve geometric accuracy, coherent panoptic understanding, and real-time inference frequency in large-scale scenes. In this paper, we propose a comprehensive framework that integrates geometric reinforcement, end-to-end panoptic learning, and efficient rendering. First, to ensure physical realism in large-scale environments, we leverage LiDAR data to construct plane-constrained multimodal Gaussian Mixture Models (GMMs) and employ 2D Gaussian surfels as the map representation, enabling high-precision surface alignment and continuous geometric supervision. Building upon this, to overcome the error accumulation and cumbersome cross-frame association inherent in traditional multi-stage panoptic segmentation pipelines, we design a query-guided end-to-end learning architecture. By utilizing a local cross-attention mechanism within the view frustum, the system lifts 2D mask features directly into 3D space, achieving globally consistent panoptic understanding. Finally, addressing the computational bottlenecks caused by high-dimensional semantic features, we introduce Precise Tile Intersection and a Top-K Hard Selection strategy to optimize the rendering pipeline. Experimental results demonstrate that our system achieves superior geometric and panoptic reconstruction quality in large-scale scenes while maintaining an inference rate exceeding 40 FPS, meeting the real-time requirements of robotic control loops.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10951v1">Fast-SegSim: Real-Time Open-Vocabulary Segmentation for Robotics in Simulation</a></div>
    <div class="paper-meta">
      📅 2026-04-13
    </div>
    <details class="paper-abstract">
      Open-vocabulary panoptic reconstruction is crucial for advanced robotics and simulation. However, existing 3D reconstruction methods, such as NeRF or Gaussian Splatting variants, often struggle to achieve the real-time inference frequency required by robotic control loops. Existing methods incur prohibitive latency when processing the high-dimensional features required for robust open-vocabulary segmentation. We propose Fast-SegSim, a novel, simple, and end-to-end framework built upon 2D Gaussian Splatting, designed to realize real-time, high-fidelity, and 3D-consistent open-vocabulary segmentation reconstruction. Our core contribution is a highly optimized rendering pipeline that specifically addresses the computational bottleneck of high-channel segmentation feature accumulation. We introduce two key optimizations: Precise Tile Intersection to reduce rasterization redundancy, and a novel Top-K Hard Selection strategy. This strategy leverages the geometric sparsity inherent in the 2D Gaussian representation to greatly simplify feature accumulation and alleviate bandwidth limitations, achieving render rates exceeding 40 FPS. Fast-SegSim provides critical value in robotic applications: it serves both as a high-frequency sensor input for simulation platforms like Gazebo, and its 3D-consistent outputs provide essential multi-view 'ground truth' labels for fine-tuning downstream perception tasks. We demonstrate this utility by using the generated labels to fine-tune the perception module in object goal navigation, successfully doubling the navigation success rate. Our superior rendering speed and practical utility underscore Fast-SegSim's potential to bridge the sim-to-real gap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10809v1">WARPED: Wrist-Aligned Rendering for Robot Policy Learning from Egocentric Human Demonstrations</a></div>
    <div class="paper-meta">
      📅 2026-04-12
    </div>
    <details class="paper-abstract">
      Recent advancements in learning from human demonstration have shown promising results in addressing the scalability and high cost of data collection required to train robust visuomotor policies. However, existing approaches are often constrained by a reliance on multiview camera setups, depth sensors, or custom hardware and are typically limited to policy execution from third-person or egocentric cameras. In this paper, we present WARPED, a framework designed to synthesize realistic wrist-view observations from human demonstration videos to facilitate the training of visuomotor policies using only monocular RGB data. With data collected from an egocentric RGB camera, our system leverages vision foundation models to initialize the interactive scene. A hand-object interaction pipeline is then employed to track the hand and manipulated object and retarget the trajectories to a robotic end-effector. Lastly, photo-realistic wrist-view observations are synthesized via Gaussian Splatting to directly train a robotic policy. We demonstrate that WARPED achieves success rates comparable to policies trained on teleoperated demonstration data for five tabletop manipulation tasks, while requiring 5-8x less data collection time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10593v1">MonoEM-GS: Monocular Expectation-Maximization Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2026-04-12
    </div>
    <details class="paper-abstract">
      Feed-forward geometric foundation models can infer dense point clouds and camera motion directly from RGB streams, providing priors for monocular SLAM. However, their predictions are often view-dependent and noisy: geometry can vary across viewpoints and under image transformations, and local metric properties may drift between frames. We present MonoEM-GS, a monocular mapping pipeline that integrates such geometric predictions into a global Gaussian Splatting representation while explicitly addressing these inconsistencies. MonoEM-GS couples Gaussian Splatting with an Expectation--Maximization formulation to stabilize geometry, and employs ICP-based alignment for monocular pose estimation. Beyond geometry, MonoEM-GS parameterizes Gaussians with multi-modal features, enabling in-place open-set segmentation and other downstream queries directly on the reconstructed map. We evaluate MonoEM-GS on 7-Scenes, TUM RGB-D and Replica, and compare against recent baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10573v1">Learning 3D Representations for Spatial Intelligence from Unposed Multi-View Images</a></div>
    <div class="paper-meta">
      📅 2026-04-12
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      Robust 3D representation learning forms the perceptual foundation of spatial intelligence, enabling downstream tasks in scene understanding and embodied AI. However, learning such representations directly from unposed multi-view images remains challenging. Recent self-supervised methods attempt to unify geometry, appearance, and semantics in a feed-forward manner, but they often suffer from weak geometry induction, limited appearance detail, and inconsistencies between geometry and semantics. We introduce UniSplat, a feed-forward framework designed to address these limitations through three complementary components. First, we propose a dual-masking strategy that strengthens geometry induction in the encoder. By masking both encoder and decoder tokens, and targeting decoder masks toward geometry-rich regions, the model is forced to infer structural information from incomplete visual cues, yielding geometry-aware representations even under unposed inputs. Second, we develop a coarse-to-fine Gaussian splatting strategy that reduces appearance-semantics inconsistencies by progressively refining the radiance field. Finally, to enforce geometric-semantic consistency, we introduce a pose-conditioned recalibration mechanism that interrelates the outputs of multiple heads by re-projecting predicted 3D point and semantic maps into the image plane using estimated camera parameters, and aligning them with corresponding RGB and semantic predictions to ensure cross-task consistency, thereby resolving geometry-semantic mismatches. Together, these components yield unified 3D representations that are robust to unposed, sparse-view inputs and generalize across diverse tasks, laying a perceptual foundation for spatial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10512v1">FreeScale: Scaling 3D Scenes via Certainty-Aware Free-View Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-12
      | 💬 CVPR2026
    </div>
    <details class="paper-abstract">
      The development of generalizable Novel View Synthesis (NVS) models is critically limited by the scarcity of large-scale training data featuring diverse and precise camera trajectories. While real-world captures are photorealistic, they are typically sparse and discrete. Conversely, synthetic data scales but suffers from a domain gap and often lacks realistic semantics. We introduce FreeScale, a novel framework that leverages the power of scene reconstruction to transform limited real-world image sequences into a scalable source of high-quality training data. Our key insight is that an imperfect reconstructed scene serves as a rich geometric proxy, but naively sampling from it amplifies artifacts. To this end, we propose a certainty-aware free-view sampling strategy identifying novel viewpoints that are both semantically meaningful and minimally affected by reconstruction errors. We demonstrate FreeScale's effectiveness by scaling up the training of feedforward NVS models, achieving a notable gain of 2.7 dB in PSNR on challenging out-of-distribution benchmarks. Furthermore, we show that the generated data can actively enhance per-scene 3D Gaussian Splatting optimization, leading to consistent improvements across multiple datasets. Our work provides a practical and powerful data generation engine to overcome a fundamental bottleneck in 3D vision. Project page: https://mvp-ai-lab.github.io/FreeScale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10259v1">Real-Time Human Reconstruction and Animation using Feed-Forward Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-11
    </div>
    <details class="paper-abstract">
      We present a generalizable feed-forward Gaussian splatting framework for human 3D reconstruction and real-time animation that operates directly on multi-view RGB images and their associated SMPL-X poses. Unlike prior methods that rely on depth supervision, fixed input views, UV map, or repeated feed-forward inference for each target view or pose, our approach predicts, in a canonical pose, a set of 3D Gaussian primitives associated with each SMPL-X vertex. One Gaussian is regularized to remain close to the SMPL-X surface, providing a strong geometric prior and stable correspondence to the parametric body model, while an additional small set of unconstrained Gaussians per vertex allows the representation to capture geometric structures that deviate from the parametric surface, such as clothing and hair. In contrast to recent approaches such as HumanRAM, which require repeated network inference to synthesize novel poses, our method produces an animatable human representation from a single forward pass; by explicitly associating Gaussian primitives with SMPL-X vertices, the reconstructed model can be efficiently animated via linear blend skinning without further network evaluation. We evaluate our method on the THuman 2.1, AvatarReX and THuman 4.0 datasets, where it achieves reconstruction quality comparable to state-of-the-art methods while uniquely supporting real-time animation and interactive applications. Code and pre-trained models are available at https://github.com/Devdoot57/HumanGS .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10223v1">A 129FPS Full HD Real-Time Accelerator for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-11
    </div>
    <details class="paper-abstract">
      Rendering large-scale, unbounded scenes on AR/VR-class devices is constrained by the computation, bandwidth, and storage cost of 3D Gaussian Splatting (3DGS). We propose a low-power, low-cost 3DGS hardware accelerator that renders full-HD images in real time, together with a hardware-friendly compression pipeline that combines iterative Gaussian pruning and fine-tuning, progressive spherical harmonics (SH) degree reduction, and vector quantization of all SH coefficients and colors. The scheme achieves a $51.6\times$ model-size reduction with a 0.743 dB PSNR loss. The accelerator uses a frame-level pipeline that integrates point-based culling and projection with tile-based sorting and rasterization, skips zero-Jacobian matrix multiplications (reducing processing elements by 63\% and computation by 53\%), and adopts comparison-free tile-based sorting with deterministic latency. Implemented in a TSMC 28-nm process at 800 MHz, the design occupies $0.66~\text{mm}^2$ with 1.1438 M gates and 120 kB SRAM, consumes 0.219 W, and delivers 1219 Mpixels/J at 267.5 Mpixels/s, enabling 1080p at 129 FPS. Overall, it is $5.98\times$ smaller in area, $5.94\times$ higher throughput, and delivers $7.5\times$ higher energy efficiency than prior 3DGS accelerators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.09977v4">A Survey on 3D Gaussian Splatting Applications: Segmentation, Editing, and Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-11
      | 💬 GitHub Repo: https://github.com/heshuting555/Awesome-3DGS-Applications
    </div>
    <details class="paper-abstract">
      In the context of novel view synthesis, 3D Gaussian Splatting (3DGS) has recently emerged as an efficient and competitive counterpart to Neural Radiance Field (NeRF), enabling high-fidelity photorealistic rendering in real time. Beyond novel view synthesis, the explicit and compact nature of 3DGS enables a wide range of downstream applications that require geometric and semantic understanding. This survey provides a comprehensive overview of recent progress in 3DGS applications. It first reviews the reconstruction preliminaries of 3DGS, followed by the problem formulation, 2D foundation models, and related NeRF-based research areas that inform downstream 3DGS applications. We then categorize 3DGS applications into three foundational tasks: segmentation, editing, and generation, alongside additional functional applications built upon or tightly coupled with these foundational capabilities. For each, we summarize representative methods, supervision strategies, and learning paradigms, highlighting shared design principles and emerging trends. Commonly used datasets and evaluation protocols are also summarized, along with comparative analyses of recent methods across public benchmarks. To support ongoing research and development, a continually updated repository of papers, code, and resources is maintained at https://github.com/heshuting555/Awesome-3DGS-Applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.09903v1">PointSplat: Efficient Geometry-Driven Pruning and Transformer Refinement for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-10
      | 💬 Accepted to CVPRW 2026 (3DMV)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently unlocked real-time, high-fidelity novel view synthesis by representing scenes using explicit 3D primitives. However, traditional methods often require millions of Gaussians to capture complex scenes, leading to significant memory and storage demands. Recent approaches have addressed this issue through pruning and per-scene fine-tuning of Gaussian parameters, thereby reducing the model size while maintaining visual quality. These strategies typically rely on 2D images to compute important scores followed by scene-specific optimization. In this work, we introduce PointSplat, 3D geometry-driven prune-and-refine framework that bridges previously disjoint directions of gaussian pruning and transformer refinement. Our method includes two key components: (1) an efficient geometry-driven strategy that ranks Gaussians based solely on their 3D attributes, removing reliance on 2D images during pruning stage, and (2) a dual-branch encoder that separates, re-weights geometric and appearance to avoid feature imbalance. Extensive experiments on ScanNet++ and Replica across varying sparsity levels demonstrate that PointSplat consistently achieves competitive rendering quality and superior efficiency without additional per-scene optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02003v2">ProDiG: Progressive Diffusion-Guided Gaussian Splatting for Aerial to Ground Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-10
      | 💬 CVPR Findings 2026
    </div>
    <details class="paper-abstract">
      Generating ground-level views and coherent 3D site models from aerial-only imagery is challenging due to extreme viewpoint changes, missing intermediate observations, and large scale variations. Existing methods either refine renderings post-hoc, often producing geometrically inconsistent results, or rely on multi-altitude ground-truth, which is rarely available. Gaussian Splatting and diffusion-based refinements improve fidelity under small variations but fail under wide aerial-toground gaps. To address these limitations, we introduce ProDiG (Progressive Diffusion-Guided Gaussian Splatting for Aerial to Ground Reconstruction), a diffusionguided framework that progressively transforms aerial 3D representations toward ground-level fidelity. ProDiG synthesizes intermediate-altitude views and refines the Gaussian representation at each stage using a geometry-aware causal attention module that injects epipolar structure into reference-view diffusion. A distance-adaptive Gaussian module dynamically adjusts Gaussian scale and opacity based on camera distance, ensuring stable reconstruction across large viewpoint gaps. Together, these components enable progressive, geometrically grounded refinement without requiring additional ground-truth viewpoints. Extensive experiments on synthetic and real-world datasets demonstrate that ProDiG produces visually realistic ground-level renderings and coherent 3D geometry, significantly outperforming existing approaches in terms of visual quality, geometric consistency, and robustness to extreme viewpoint changes. Project Page: https://sirsh07.github.io/research/prodig
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.09835v1">F3G-Avatar : Face Focused Full-body Gaussian Avatar</a></div>
    <div class="paper-meta">
      📅 2026-04-10
      | 💬 CVPRW 3DMV, 10 pages
    </div>
    <details class="paper-abstract">
      Existing full-body Gaussian avatar methods primarily optimize global reconstruction quality and often fail to preserve fine-grained facial geometry and expression details. This challenge arises from limited facial representational capacity that causes difficulties in modeling high-frequency pose-dependent deformations. To address this, we propose F3G-Avatar, a full-body, face-aware avatar synthesis method that reconstructs animatable human representations from multi-view RGB video and regressed pose/shape parameters. Starting from a clothed Momentum Human Rig (MHR) template, front/back positional maps are rendered and decoded into 3D Gaussians through a two-branch architecture: a body branch that captures pose-dependent non-rigid deformations and a face-focused deformation branch that refines head geometry and appearance. The predicted Gaussians are fused, posed with linear blend skinning (LBS), and rendered with differentiable Gaussian splatting. Training combines reconstruction and perceptual objectives with a face-specific adversarial loss to enhance realism in close-up views. Experiments demonstrate strong rendering quality, with face-view performance reaching PSNR/SSIM/LPIPS of 26.243/0.964/0.084 on the AvatarReX dataset. Ablations further highlight contributions of the MHR template and the face-focused deformation. F3G-Avatar provides a practical, high-quality pipeline for realistic, animatable full-body avatar synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02781v2">DynFOA: Generating First-Order Ambisonics with Conditional Diffusion for Dynamic and Acoustically Complex 360-Degree Videos</a></div>
    <div class="paper-meta">
      📅 2026-04-10
      | 💬 Accidental duplicate submission. This paper was intended to be a replacement (v2) for arXiv:2602.06846
    </div>
    <details class="paper-abstract">
      Spatial audio is crucial for immersive 360-degree video experiences, yet most 360-degree videos lack it due to the difficulty of capturing spatial audio during recording. Automatically generating spatial audio such as first-order ambisonics (FOA) from video therefore remains an important but challenging problem. In complex scenes, sound perception depends not only on sound source locations but also on scene geometry, materials, and dynamic interactions with the environment. However, existing approaches only rely on visual cues and fail to model dynamic sources and acoustic effects such as occlusion, reflections, and reverberation. To address these challenges, we propose DynFOA, a generative framework that synthesizes FOA from 360-degree videos by integrating dynamic scene reconstruction with conditional diffusion modeling. DynFOA analyzes the input video to detect and localize dynamic sound sources, estimate depth and semantics, and reconstruct scene geometry and materials using 3D Gaussian Splatting (3DGS). The reconstructed scene representation provides physically grounded features that capture acoustic interactions between sources, environment, and listener viewpoint. Conditioned on these features, a diffusion model generates spatial audio consistent with the scene dynamics and acoustic context. We introduce M2G-360, a dataset of 600 real-world clips divided into MoveSources, Multi-Source, and Geometry subsets for evaluating robustness under diverse conditions. Experiments show that DynFOA consistently outperforms existing methods in spatial accuracy, acoustic fidelity, distribution matching, and perceived immersive experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07928v2">Generative 3D Gaussian Splatting for Arbitrary-ResolutionAtmospheric Downscaling and Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-04-10
      | 💬 20 pages, 13 figures
    </div>
    <details class="paper-abstract">
      While AI-based numerical weather prediction (NWP) enables rapid forecasting, generating high-resolution outputs remains computationally demanding due to limited multi-scale adaptability and inefficient data representations. We propose the 3D Gaussian splatting-based scale-aware vision transformer (GSSA-ViT), a novel framework for arbitrary-resolution forecasting and flexible downscaling of high-dimensional atmospheric fields. Specifically, latitude-longitude grid points are treated as centers of 3D Gaussians. A generative 3D Gaussian prediction scheme is introduced to estimate key parameters, including covariance, attributes, and opacity, for unseen samples, improving generalization and mitigating overfitting. In addition, a scale-aware attention module is designed to capture cross-scale dependencies, enabling the model to effectively integrate information across varying downscaling ratios and support continuous resolution adaptation. To our knowledge, this is the first NWP approach that combines generative 3D Gaussian modeling with scale-aware attention for unified multi-scale prediction. Experiments on ERA5 show that the proposed method accurately forecasts 87 atmospheric variables at arbitrary resolutions, while evaluations on ERA5 and CMIP6 demonstrate its superior performance in downscaling tasks. The proposed framework provides an efficient and scalable solution for high-resolution, multi-scale atmospheric prediction and downscaling. Code is available at: https://github.com/binbin2xs/weather-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.09324v1">Structure-Aware Fine-Grained Gaussian Splatting for Expressive Avatar Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-10
      | 💬 The code is on Github: https://github.com/Su245811YZ/SFGS
    </div>
    <details class="paper-abstract">
      Reconstructing photorealistic and topology-aware human avatars from monocular videos remains a significant challenge in the fields of computer vision and graphics. While existing 3D human avatar modeling approaches can effectively capture body motion, they often fail to accurately model fine details such as hand movements and facial expressions. To address this, we propose Structure-aware Fine-grained Gaussian Splatting (SFGS), a novel method for reconstructing expressive and coherent full-body 3D human avatars from a monocular video sequence. The SFGS use both spatial-only triplane and time-aware hexplane to capture dynamic features across consecutive frames. A structure-aware gaussian module is designed to capture pose-dependent details in a spatially coherent manner and improve pose and texture expression. To better model hand deformations, we also propose a residual refinement module based on fine-grained hand reconstruction. Our method requires only a single-stage training and outperforms state-of-the-art baselines in both quantitative and qualitative evaluations, generating high-fidelity avatars with natural motion and fine details. The code is on Github: https://github.com/Su245811YZ/SFGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.01767v2">LoBE-GS: Load-Balanced and Efficient 3D Gaussian Splatting for Large-Scale Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-10
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has established itself as an efficient representation for real-time, high-fidelity 3D scene reconstruction. However, scaling 3DGS to large and unbounded scenes such as city blocks remains difficult. Existing divide-and-conquer methods alleviate memory pressure by partitioning the scene into blocks and training on multiple, non-communicating GPUs, but introduce new bottlenecks: (i) partitions suffer from severe load imbalance since uniform or heuristic splits do not reflect actual computational demands, and (ii) coarse-to-fine pipelines fail to exploit the coarse stage efficiently, often reloading the entire model and incurring high overhead. In this work, we introduce LoBE-GS, a novel Load-Balanced and Efficient 3D Gaussian Splatting framework, that re-engineers the large-scale 3DGS pipeline. Specifically, LoBE-GS introduces a load-balanced KD-tree scene partitioning scheme with optimized cutlines that balance per-block camera counts. To accelerate preprocessing, it employs depth-based back-projection for fast camera assignment, reducing processing time from hours to minutes. It further reduces training cost through two lightweight techniques: visibility cropping and selective densification. Evaluations on large-scale urban and outdoor datasets show that LoBE-GS consistently achieves up to 2 times faster end-to-end training time than state-of-the-art baselines, while maintaining reconstruction quality and enabling scalability to scenes infeasible with vanilla 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.09045v1">Scene-Agnostic Object-Centric Representation Learning for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-10
    </div>
    <details class="paper-abstract">
      Recent works on 3D scene understanding leverage 2D masks from visual foundation models (VFMs) to supervise radiance fields, enabling instance-level 3D segmentation. However, the supervision signals from foundation models are not fundamentally object-centric and often require additional mask pre/post-processing or specialized training and loss design to resolve mask identity conflicts across views. The learned identity of the 3D scene is scene-dependent, limiting generalizability across scenes. Therefore, we propose a dataset-level, object-centric supervision scheme to learn object representations in 3D Gaussian Splatting (3DGS). Building on a pre-trained slot attention-based Global Object Centric Learning (GOCL) module, we learn a scene-agnostic object codebook that provides consistent, identity-anchored representations across views and scenes. By coupling the codebook with the module's unsupervised object masks, we can directly supervise the identity features of 3D Gaussians without additional mask pre-/post-processing or explicit multi-view alignment. The learned scene-agnostic codebook enables object supervision and identification without per-scene fine-tuning or retraining. Our method thus introduces unsupervised object-centric learning (OCL) into 3DGS, yielding more structured representations and better generalization for downstream tasks such as robotic interaction, scene understanding, and cross-scene generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08967v1">AudioGS: Spectrogram-Based Audio Gaussian Splatting for Sound Field Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-10
    </div>
    <details class="paper-abstract">
      Spatial audio is fundamental to immersive virtual experiences, yet synthesizing high-fidelity binaural audio from sparse observations remains a significant challenge. Existing methods typically rely on implicit neural representations conditioned on visual priors, which often struggle to capture fine-grained acoustic structures. Inspired by 3D Gaussian Splatting (3DGS), we introduce AudioGS, a novel visual-free framework that explicitly encodes the sound field as a set of Audio Gaussians based on spectrograms. AudioGS associates each time-frequency bin with an Audio Gaussian equipped with dual Spherical Harmonic (SH) coefficients and a decay coefficient. For a target pose, we render binaural audio by evaluating the SH field to capture directionality, incorporating geometry-guided distance attenuation and phase correction, and reconstructing the waveform. Experiments on the Replay-NVAS dataset demonstrate that AudioGS successfully captures complex spatial cues and outperforms state-of-the-art visual-dependent baselines. Specifically, AudioGS reduces the magnitude reconstruction error (MAG) by over 14% and reduces the perceptual quality metric (DPAM) by approximately 25% compared to the best performing visual-guided method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08410v1">BLaDA: Bridging Language to Functional Dexterous Actions within 3DGS Fields</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Code will be publicly available at https://github.com/PopeyePxx/BLaDA
    </div>
    <details class="paper-abstract">
      In unstructured environments, functional dexterous grasping calls for the tight integration of semantic understanding, precise 3D functional localization, and physically interpretable execution. Modular hierarchical methods are more controllable and interpretable than end-to-end VLA approaches, but existing ones still rely on predefined affordance labels and lack the tight semantic--pose coupling needed for functional dexterous manipulation. To address this, we propose BLaDA (Bridging Language to Dexterous Actions in 3DGS fields), an interpretable zero-shot framework that grounds open-vocabulary instructions as perceptual and control constraints for functional dexterous manipulation. BLaDA establishes an interpretable reasoning chain by first parsing natural language into a structured sextuple of manipulation constraints via a Knowledge-guided Language Parsing (KLP) module. To achieve pose-consistent spatial reasoning, we introduce the Triangular Functional Point Localization (TriLocation) module, which utilizes 3D Gaussian Splatting as a continuous scene representation and identifies functional regions under triangular geometric constraints. Finally, the 3D Keypoint Grasp Matrix Transformation Execution (KGT3D+) module decodes these semantic-geometric constraints into physically plausible wrist poses and finger-level commands. Extensive experiments on complex benchmarks demonstrate that BLaDA significantly outperforms existing methods in both affordance grounding precision and the success rate of functional manipulation across diverse categories and tasks. Code will be publicly available at https://github.com/PopeyePxx/BLaDA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08370v1">SurfelSplat: Learning Efficient and Generalizable Gaussian Surfel Representations for Sparse-View Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Code is available at https://github.com/Simon-Dcs/Surfel_Splat
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated impressive performance in 3D scene reconstruction. Beyond novel view synthesis, it shows great potential for multi-view surface reconstruction. Existing methods employ optimization-based reconstruction pipelines that achieve precise and complete surface extractions. However, these approaches typically require dense input views and high time consumption for per-scene optimization. To address these limitations, we propose SurfelSplat, a feed-forward framework that generates efficient and generalizable pixel-aligned Gaussian surfel representations from sparse-view images. We observe that conventional feed-forward structures struggle to recover accurate geometric attributes of Gaussian surfels because the spatial frequency of pixel-aligned primitives exceeds Nyquist sampling rates. Therefore, we propose a cross-view feature aggregation module based on the Nyquist sampling theorem. Specifically, we first adapt the geometric forms of Gaussian surfels with spatial sampling rate-guided low-pass filters. We then project the filtered surfels across all input views to obtain cross-view feature correlations. By processing these correlations through a specially designed feature fusion network, we can finally regress Gaussian surfels with precise geometry. Extensive experiments on DTU reconstruction benchmarks demonstrate that our model achieves comparable results with state-of-the-art methods, and predict Gaussian surfels within 1 second, offering a 100x speedup without costly per-scene training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07053v2">AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      Recent feed-forward Gaussian reconstruction models adopt a pixel-aligned formulation that maps each 2D pixel to a 3D Gaussian, entangling Gaussian representations tightly with the input images. In this paper, we propose AnchorSplat, a novel feed-forward 3DGS framework for scene-level reconstruction that represents the scene directly in 3D space. AnchorSplat introduces an anchor-aligned Gaussian representation guided by 3D geometric priors (e.g., sparse point clouds, voxels, or RGB-D point clouds), enabling a more geometry-aware renderable 3D Gaussians that is independent of image resolution and number of views. This design substantially reduces the number of required Gaussians, improving computational efficiency while enhancing reconstruction fidelity. Beyond the anchor-aligned design, we utilize a Gaussian Refiner to adjust the intermediate Gaussiansy via merely a few forward passes. Experiments on the ScanNet++ v2 NVS benchmark demonstrate the SOTA performance, outperforming previous methods with more view-consistent and substantially fewer Gaussian primitives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07928v1">Generative 3D Gaussian Splatting for Arbitrary-ResolutionAtmospheric Downscaling and Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 20 pages, 13 figures
    </div>
    <details class="paper-abstract">
      While AI-based numerical weather prediction (NWP) enables rapid forecasting, generating high-resolution outputs remains computationally demanding due to limited multi-scale adaptability and inefficient data representations. We propose the 3D Gaussian splatting-based scale-aware vision transformer (GSSA-ViT), a novel framework for arbitrary-resolution forecasting and flexible downscaling of high-dimensional atmospheric fields. Specifically, latitude-longitude grid points are treated as centers of 3D Gaussians. A generative 3D Gaussian prediction scheme is introduced to estimate key parameters, including covariance, attributes, and opacity, for unseen samples, improving generalization and mitigating overfitting. In addition, a scale-aware attention module is designed to capture cross-scale dependencies, enabling the model to effectively integrate information across varying downscaling ratios and support continuous resolution adaptation. To our knowledge, this is the first NWP approach that combines generative 3D Gaussian modeling with scale-aware attention for unified multi-scale prediction. Experiments on ERA5 show that the proposed method accurately forecasts 87 atmospheric variables at arbitrary resolutions, while evaluations on ERA5 and CMIP6 demonstrate its superior performance in downscaling tasks. The proposed framework provides an efficient and scalable solution for high-resolution, multi-scale atmospheric prediction and downscaling. Code is available at: https://github.com/binbin2xs/weather-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07882v1">ReconPhys: Reconstruct Appearance and Physical Attributes from Single Video</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Reconstructing non-rigid objects with physical plausibility remains a significant challenge. Existing approaches leverage differentiable rendering for per-scene optimization, recovering geometry and dynamics but requiring expensive tuning or manual annotation, which limits practicality and generalizability. To address this, we propose ReconPhys, the first feedforward framework that jointly learns physical attribute estimation and 3D Gaussian Splatting reconstruction from a single monocular video. Our method employs a dual-branch architecture trained via a self-supervised strategy, eliminating the need for ground-truth physics labels. Given a video sequence, ReconPhys simultaneously infers geometry, appearance, and physical attributes. Experiments on a large-scale synthetic dataset demonstrate superior performance: our method achieves 21.64 PSNR in future prediction compared to 13.27 by state-of-the-art optimization baselines, while reducing Chamfer Distance from 0.349 to 0.004. Crucially, ReconPhys enables fast inference (<1 second) versus hours required by existing methods, facilitating rapid generation of simulation-ready assets for robotics and graphics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06774v2">RDSplat: Robust Watermarking for 3D Gaussian Splatting Against 2D and 3D Diffusion Editing</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a leading representation for high-fidelity 3D assets, yet protecting these assets via digital watermarking remains an open challenge. Existing 3DGS watermarking methods are robust only to classical distortions and fail under diffusion editing, which operates at both the 2D image level and the 3D scene level, covertly erasing embedded watermarks while preserving visual plausibility. We present RDSplat, the first 3DGS watermarking framework designed to withstand both 2D and 3D diffusion editing. Our key observation is that diffusion models act as low-pass filters that preserve low-frequency structures while regenerating high-frequency details. RDSplat exploits this by embedding 100-bit watermarks exclusively into low-frequency Gaussian primitives identified through Frequency-Aware Primitive Selection (FAPS), which combines the Mip score and directional balance score, while freezing all other primitives. Training efficiency is achieved through a surrogate strategy that replaces costly diffusion forward passes with Gaussian blur augmentation. A dedicated decoder, GeoMark, built on ViT-S/16 with spatially periodic secret embedding, jointly resists diffusion editing and the geometric transformations inherent to novel-view rendering. Extensive experiments on four benchmarks under seven 2D diffusion attacks and iterative 3D editing demonstrate strong classical robustness (bit accuracy 0.811) and competitive diffusion robustness (bit accuracy 0.701) at 100-bit capacity, while completing fine-tuning in 3 to 7 minutes on a single RTX 4090 GPU.
    </details>
</div>
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
