# gaussian splatting - 2026_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.29734v1">GRVS: a Generalizable and Recurrent Approach to Monocular Dynamic View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-03-31
      | 💬 CVPR Findings 2026
    </div>
    <details class="paper-abstract">
      Synthesizing novel views from monocular videos of dynamic scenes remains a challenging problem. Scene-specific methods that optimize 4D representations with explicit motion priors often break down in highly dynamic regions where multi-view information is hard to exploit. Diffusion-based approaches that integrate camera control into large pre-trained models can produce visually plausible videos but frequently suffer from geometric inconsistencies across both static and dynamic areas. Both families of methods also require substantial computational resources. Building on the success of generalizable models for static novel view synthesis, we adapt the framework to dynamic inputs and propose a new model with two key components: (1) a recurrent loop that enables unbounded and asynchronous mapping between input and target videos and (2) an efficient use of plane sweeps over dynamic inputs to disentangle camera and scene motion, and achieve fine-grained, six-degrees-of-freedom camera controls. We train and evaluate our model on the UCSD dataset and on Kubric-4D-dyn, a new monocular dynamic dataset featuring longer, higher resolution sequences with more complex scene dynamics than existing alternatives. Our model outperforms four Gaussian Splatting-based scene-specific approaches, as well as two diffusion-based approaches in reconstructing fine-grained geometric details across both static and dynamic regions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.08811v2">TUGS: Physics-based Compact Representation of Underwater Scenes by Tensorized Gaussian</a></div>
    <div class="paper-meta">
      📅 2026-03-31
    </div>
    <details class="paper-abstract">
      Underwater 3D scene reconstruction is crucial for multimedia applications in adverse environments, such as underwater robotic perception and navigation. However, the complexity of interactions between light propagation, water medium, and object surfaces poses significant difficulties for existing methods in accurately simulating their interplay. Additionally, expensive training and rendering costs limit their practical application. Therefore, we propose Tensorized Underwater Gaussian Splatting (TUGS), a compact underwater 3D representation based on physical modeling of complex underwater light fields. TUGS includes a physics-based underwater Adaptive Medium Estimation (AME) module, enabling accurate simulation of both light attenuation and backscatter effects in underwater environments, and introduces Tensorized Densification Strategies (TDS) to efficiently refine the tensorized representation during optimization. TUGS is able to render high-quality underwater images with faster rendering speeds and less memory usage. Extensive experiments on real-world underwater datasets have demonstrated that TUGS can efficiently achieve superior reconstruction quality using a limited number of parameters. The code is available at https://liamlian0727.github.io/TUGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.27516v2">SGS-Intrinsic: Semantic-Invariant Gaussian Splatting for Sparse-View Indoor Inverse Rendering</a></div>
    <div class="paper-meta">
      📅 2026-03-31
      | 💬 CVPR2026
    </div>
    <details class="paper-abstract">
      We present SGS-Intrinsic, an indoor inverse rendering framework that works well for sparse-view images. Unlike existing 3D Gaussian Splatting (3DGS) based methods that focus on object-centric reconstruction and fail to work under sparse view settings, our method allows to achieve high-quality geometry reconstruction and accurate disentanglement of material and illumination. The core idea is to construct a dense and geometry-consistent Gaussian semantic field guided by semantic and geometric priors, providing a reliable foundation for subsequent inverse rendering. Building upon this, we perform material-illumination disentanglement by combining a hybrid illumination model and material prior to effectively capture illumination-material interactions. To mitigate the impact of cast shadows and enhance the robustness of material recovery, we introduce illumination-invariant material constraint together with a deshadowing model. Extensive experiments on benchmark datasets show that our method consistently improves both reconstruction fidelity and inverse rendering quality over existing 3DGS-based inverse rendering approaches. Our code is available at https://github.com/GrumpySloths/SGS_Intrinsic.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.03309v1">TreeGaussian: Tree-Guided Cascaded Contrastive Learning for Hierarchical Consistent 3D Gaussian Scene Segmentation and Understanding</a></div>
    <div class="paper-meta">
      📅 2026-03-31
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a real-time, differentiable representation for neural scene understanding. However, existing 3DGS-based methods struggle to represent hierarchical 3D semantic structures and capture whole-part relationships in complex scenes. Moreover, dense pairwise comparisons and inconsistent hierarchical labels from 2D priors hinder feature learning, resulting in suboptimal segmentation. To address these limitations, we introduce TreeGaussian, a tree-guided cascaded contrastive learning framework that explicitly models hierarchical semantic relationships and reduces redundancy in contrastive supervision. By constructing a multi-level object tree, TreeGaussian enables structured learning across object-part hierarchies. In addition, we propose a two-stage cascaded contrastive learning strategy that progressively refines feature representations from global to local, mitigating saturation and stabilizing training. A Consistent Segmentation Detection (CSD) mechanism and a graph-based denoising module are further introduced to align segmentation modes across views while suppressing unstable Gaussian points, enhancing segmentation consistency and quality. Extensive experiments, including open-vocabulary 3D object selection, 3D point cloud understanding, and ablation studies, demonstrate the effectiveness and robustness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.29394v1">AA-Splat: Anti-Aliased Feed-forward Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-31
      | 💬 Please visit our project page at https://kaist-viclab.github.io/aasplat-site/
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (FF-3DGS) emerges as a fast and robust solution for sparse-view 3D reconstruction and novel view synthesis (NVS). However, existing FF-3DGS methods are built on incorrect screen-space dilation filters, causing severe rendering artifacts when rendering at out-of-distribution sampling rates. We firstly propose an FF-3DGS model, called AA-Splat, to enable robust anti-aliased rendering at any resolution. AA-Splat utilizes an opacity-balanced band-limiting (OBBL) design, which combines two components: a 3D band-limiting post-filter integrates multi-view maximal frequency bounds into the feed-forward reconstruction pipeline, effectively band-limiting the resulting 3D scene representations and eliminating degenerate Gaussians; an Opacity Balancing (OB) to seamlessly integrate all pixel-aligned Gaussian primitives into the rendering process, compensating for the increased overlap between expanded Gaussian primitives. AA-Splat demonstrates drastic improvements with average 5.4$\sim$7.5dB PSNR gains on NVS performance over a state-of-the-art (SOTA) baseline, DepthSplat, at all resolutions, between $4\times$ and $1/4\times$. Code will be made available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.29296v1">MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-31
      | 💬 Accepted to CVPR 2026
    </div>
    <details class="paper-abstract">
      Realistic reconstruction of dynamic 4D scenes from monocular videos is essential for understanding the physical world. Despite recent progress in neural rendering, existing methods often struggle to recover accurate 3D geometry and temporally consistent motion in complex environments. To address these challenges, we propose MotionScale, a 4D Gaussian Splatting framework that scales efficiently to large scenes and extended sequences while maintaining high-fidelity structural and motion coherence. At the core of our approach is a scalable motion field parameterized by cluster-centric basis transformations that adaptively expand to capture diverse and evolving motion patterns. To ensure robust reconstruction over long durations, we introduce a progressive optimization strategy comprising two decoupled propagation stages: 1) A background extension stage that adapts to newly visible regions, refines camera poses, and explicitly models transient shadows; 2) A foreground propagation stage that enforces motion consistency through a specialized three-stage refinement process. Extensive experiments on challenging real-world benchmarks demonstrate that MotionScale significantly outperforms state-of-the-art methods in both reconstruction quality and temporal stability. Project page: https://hrzhou2.github.io/motion-scale-web/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.29209v1">LightHarmony3D: Harmonizing Illumination and Shadows for Object Insertion in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-31
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables high-fidelity reconstruction of scene geometry and appearance. Building on this capability, inserting external mesh objects into reconstructed 3DGS scenes enables interactive editing and content augmentation for immersive applications such as AR/VR, virtual staging, and digital content creation. However, achieving physically consistent lighting and shadows for mesh insertion remains challenging, as it requires accurate scene illumination estimation and multi-view consistent rendering. To address this challenge, we present LightHarmony3D, a novel framework for illumination-consistent mesh insertion in 3DGS scenes. Central to our approach is our proposed generative module that predicts a full 360° HDR environment map at the insertion location via a single forward pass. By leveraging generative priors instead of iterative optimization, our method efficiently captures dominant scene illumination and enables physically grounded shading and shadows for inserted meshes while maintaining multi-view coherence. Furthermore, we introduce the first dedicated benchmark for mesh insertion in 3DGS, providing a standardized evaluation framework for assessing lighting consistency and photorealism. Extensive experiments across multiple real-world reconstruction datasets demonstrate that LightHarmony3D achieves state-of-the-art realism and multi-view consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.29192v1">Efficient Camera Pose Augmentation for View Generalization in Robotic Policy Learning</a></div>
    <div class="paper-meta">
      📅 2026-03-31
    </div>
    <details class="paper-abstract">
      Prevailing 2D-centric visuomotor policies exhibit a pronounced deficiency in novel view generalization, as their reliance on static observations hinders consistent action mapping across unseen views. In response, we introduce GenSplat, a feed-forward 3D Gaussian Splatting framework that facilitates view-generalized policy learning through novel view rendering. GenSplat employs a permutation-equivariant architecture to reconstruct high-fidelity 3D scenes from sparse, uncalibrated inputs in a single forward pass. To ensure structural integrity, we design a 3D-prior distillation strategy that regularizes the 3DGS optimization, preventing the geometric collapse typical of purely photometric supervision. By rendering diverse synthetic views from these stable 3D representations, we systematically augment the observational manifold during training. This augmentation forces the policy to ground its decisions in underlying 3D structures, thereby ensuring robust execution under severe spatial perturbations where baselines severely degrade.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.29185v1">Hierarchical Visual Relocalization with Nearest View Synthesis from Feature Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-31
      | 💬 Accepted to CVPR 2026
    </div>
    <details class="paper-abstract">
      Visual relocalization is a fundamental task in the field of 3D computer vision, estimating a camera's pose when it revisits a previously known scene. While point-based hierarchical relocalization methods have shown strong scalability and efficiency, they are often limited by sparse image observations and weak feature matching. In this work, we propose SplatHLoc, a novel hierarchical visual relocalization framework that uses Feature Gaussian Splatting as the scene representation. To address the sparsity of database images, we propose an adaptive viewpoint retrieval method that synthesizes virtual candidates with viewpoints more closely aligned with the query, thereby improving the accuracy of initial pose estimation. For feature matching, we observe that Gaussian-rendered features and those extracted directly from images exhibit different strengths across the two-stage matching process: the former performs better in the coarse stage, while the latter proves more effective in the fine stage. Therefore, we introduce a hybrid feature matching strategy, enabling more accurate and efficient pose estimation. Extensive experiments on both indoor and outdoor datasets show that SplatHLoc enhances the robustness of visual relocalization, setting a new state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15508v2">Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-30
      | 💬 CVPR 2026 camera ready version
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) models enable real-time scene generation but are hindered by suboptimal pixel-aligned primitive placement, which relies on a dense, rigid grid that limits both quality and efficiency. We introduce a new feed-forward architecture that detects 3D Gaussian primitives at a sub-pixel level, replacing the pixel grid with an adaptive, ``Off-The-Grid" distribution. Inspired by keypoint detection, our decoder learns to locally distribute primitives across image patches. We also provide an Adaptive Density mechanism by assigning varying number of primitives per patch based on Shannon entropy. We combine the proposed decoder with a pre-trained 3D reconstruction backbone and train them end-to-end using photometric supervision without any 3D annotation. The resulting pose-free model generates photorealistic 3DGS scenes in seconds, achieving state-of-the-art novel view synthesis for feed-forward models. It outperforms competitors while using far fewer primitives, demonstrating a more accurate and efficient allocation that captures fine details and reduces artifacts. Project page: https://arthurmoreau.github.io/OffTheGrid/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00255v2">Relightable Holoported Characters: Capturing and Relighting Dynamic Human Performance from Sparse Views</a></div>
    <div class="paper-meta">
      📅 2026-03-30
    </div>
    <details class="paper-abstract">
      We present Relightable Holoported Characters (RHC), a novel person-specific method for free-view rendering and relighting of full-body and highly dynamic humans solely observed from sparse-view RGB videos at inference. In contrast to classical one-light-at-a-time (OLAT)-based human relighting, our transformer-based RelightNet predicts relit appearance within a single network pass, avoiding costly OLAT-basis capture and generation. For training such a model, we introduce a new capture strategy and dataset recorded in a multi-view lightstage, where we alternate frames lit by random environment maps with uniformly lit tracking frames, simultaneously enabling accurate motion tracking and diverse illumination as well as dynamics coverage. Inspired by the rendering equation, we derive physics-informed features that encode geometry, albedo, shading, and the virtual camera view from a coarse human mesh proxy and the input views. Our RelightNet then takes these features as input and cross-attends them with a novel lighting condition, and regresses the relit appearance in the form of texel-aligned 3D Gaussian splats attached to the coarse mesh proxy. Consequently, our RelightNet implicitly learns to efficiently compute the rendering equation for novel lighting conditions within a single feed-forward pass. Experiments demonstrate our method's superior visual fidelity and lighting reproduction compared to state-of-the-art approaches. Project page: https://vcai.mpi-inf.mpg.de/projects/RHC/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28152v1">ObjectMorpher: 3D-Aware Image Editing via Deformable 3DGS Models</a></div>
    <div class="paper-meta">
      📅 2026-03-30
      | 💬 11 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Achieving precise, object-level control in image editing remains challenging: 2D methods lack 3D awareness and often yield ambiguous or implausible results, while existing 3D-aware approaches rely on heavy optimization or incomplete monocular reconstructions. We present ObjectMorpher, a unified, interactive framework that converts ambiguous 2D edits into geometry-grounded operations. ObjectMorpher lifts target instances with an image-to-3D generator into editable 3D Gaussian Splatting (3DGS), enabling fast, identity-preserving manipulation. Users drag control points; a graph-based non-rigid deformation with as-rigid-as-possible (ARAP) constraints ensures physically sensible shape and pose changes. A composite diffusion module harmonizes lighting, color, and boundaries for seamless reintegration. Across diverse categories, ObjectMorpher delivers fine-grained, photorealistic edits with superior controllability and efficiency, outperforming 2D drag and 3D-aware baselines on KID, LPIPS, SIFID, and user preference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28126v1">SVGS: Single-View to 3D Object Editing via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-30
    </div>
    <details class="paper-abstract">
      Text-driven 3D scene editing has attracted considerable interest due to its convenience and user-friendliness. However, methods that rely on implicit 3D representations, such as Neural Radiance Fields (NeRF), while effective in rendering complex scenes, are hindered by slow processing speeds and limited control over specific regions of the scene. Moreover, existing approaches, including Instruct-NeRF2NeRF and GaussianEditor, which utilize multi-view editing strategies, frequently produce inconsistent results across different views when executing text instructions. This inconsistency can adversely affect the overall performance of the model, complicating the task of balancing the consistency of editing results with editing efficiency. To address these challenges, we propose a novel method termed Single-View to 3D Object Editing via Gaussian Splatting (SVGS), which is a single-view text-driven editing technique based on 3D Gaussian Splatting (3DGS). Specifically, in response to text instructions, we introduce a single-view editing strategy grounded in multi-view diffusion models, which reconstructs 3D scenes by leveraging only those views that yield consistent editing results. Additionally, we employ sparse 3D Gaussian Splatting as the 3D representation, which significantly enhances editing efficiency. We conducted a comparative analysis of SVGS against existing baseline methods across various scene settings, and the results indicate that SVGS outperforms its counterparts in both editing capability and processing speed, representing a significant advancement in 3D editing technology. For further details, please visit our project page at: https://amateurc.github.io/svgs.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28064v1">\textit{4DSurf}: High-Fidelity Dynamic Scene Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-30
      | 💬 Accepted to CVPR 2026
    </div>
    <details class="paper-abstract">
      This paper addresses the problem of dynamic scene surface reconstruction using Gaussian Splatting (GS), aiming to recover temporally consistent geometry. While existing GS-based dynamic surface reconstruction methods can yield superior reconstruction, they are typically limited to either a single object or objects with only small deformations, struggling to maintain temporally consistent surface reconstruction of large deformations over time. We propose ``\textit{4DSurf}'', a novel and unified framework for generic dynamic surface reconstruction that does not require specifying the number or types of objects in the scene, can handle large surface deformations and temporal inconsistency in reconstruction. The key innovation of our framework is the introduction of Gaussian deformations induced Signed Distance Function Flow Regularization that constrains the motion of Gaussians to align with the evolving surface. To handle large deformations, we introduce an Overlapping Segment Partitioning strategy that divides the sequence into overlapping segments with small deformations and incrementally passes geometric information across segments through the shared overlapping timestep. Experiments on two challenging dynamic scene datasets, Hi4D and CMU Panoptic, demonstrate that our method outperforms state-of-the-art surface reconstruction methods by 49\% and 19\% in Chamfer distance, respectively, and achieves superior temporal consistency under sparse-view settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28020v1">Physically Inspired Gaussian Splatting for HDR Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-03-30
      | 💬 Accepted to CVPR 2026
    </div>
    <details class="paper-abstract">
      High dynamic range novel view synthesis (HDR-NVS) reconstructs scenes with dynamic details by fusing multi-exposure low dynamic range (LDR) views, yet it struggles to capture ambient illumination-dependent appearance. Implicitly supervising HDR content by constraining tone-mapped results fails in correcting abnormal HDR values, and results in limited gradients for Gaussians in under/over-exposed regions. To this end, we introduce PhysHDR-GS, a physically inspired HDR-NVS framework that models scene appearance via intrinsic reflectance and adjustable ambient illumination. PhysHDR-GS employs a complementary image-exposure (IE) branch and Gaussian-illumination (GI) branch to faithfully reproduce standard camera observations and capture illumination-dependent appearance changes, respectively. During training, the proposed cross-branch HDR consistency loss provides explicit supervision for HDR content, while an illumination-guided gradient scaling strategy mitigates exposure-biased gradient starvation and reduces under-densified representations. Experimental results across realistic and synthetic datasets demonstrate our superiority in reconstructing HDR details (e.g., a PSNR gain of 2.04 dB over HDR-GS), while maintaining real-time rendering speed (up to 76 FPS). Code and models are available at https://huimin-zeng.github.io/PhysHDR-GS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.27781v1">GS3LAM: Gaussian Semantic Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2026-03-29
      | 💬 Accepted by ACM MM 2024
    </div>
    <details class="paper-abstract">
      Recently, the multi-modal fusion of RGB, depth, and semantics has shown great potential in dense Simultaneous Localization and Mapping (SLAM). However, a prerequisite for generating consistent semantic maps is the availability of dense, efficient, and scalable scene representations. Existing semantic SLAM systems based on explicit representations are often limited by resolution and an inability to predict unknown areas. Conversely, implicit representations typically rely on time-consuming ray tracing, failing to meet real-time requirements. Fortunately, 3D Gaussian Splatting (3DGS) has emerged as a promising representation that combines the efficiency of point-based methods with the continuity of geometric structures. To this end, we propose GS3LAM, a Gaussian Semantic Splatting SLAM framework that processes multimodal data to render consistent, dense semantic maps in real-time. GS3LAM models the scene as a Semantic Gaussian Field (SG-Field) and jointly optimizes camera poses and the field via multimodal error constraints. Furthermore, a Depth-adaptive Scale Regularization (DSR) scheme is introduced to resolve misalignments between scale-invariant Gaussians and geometric surfaces. To mitigate catastrophic forgetting, we propose a Random Sampling-based Keyframe Mapping (RSKM) strategy, which demonstrates superior performance over common local covisibility optimization methods. Extensive experiments on benchmark datasets show that GS3LAM achieves increased tracking robustness, superior rendering quality, and enhanced semantic precision compared to state-of-the-art methods. Source code is available at https://github.com/lif314/GS3LAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.07754v2">MeshSplats: Mesh-Based Rendering with Gaussian Splatting Initialization</a></div>
    <div class="paper-meta">
      📅 2026-03-29
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) is a recent and pivotal technique in 3D computer graphics. GS-based algorithms almost always bypass classical methods such as ray tracing, which offer numerous inherent advantages for rendering. For example, ray tracing can handle incoherent rays for advanced lighting effects, including shadows and reflections. To address this limitation, we introduce MeshSplats, a method which converts GS to a mesh-like format. Following the completion of training, MeshSplats transforms Gaussian elements into mesh faces, enabling rendering using ray tracing methods with all their associated benefits. Our model can be utilized immediately following transformation, yielding a mesh of slightly reduced reconstruction quality without additional training. Furthermore, we can enhance the quality by applying a dedicated optimization algorithm that operates on mesh faces rather than Gaussian components. Importantly, MeshSplats acts as a wrapper, converting pre-trained GS models into a ray-traceable format. The efficacy of our method is substantiated by experimental results, underscoring its extensive applications in computer graphics and image processing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18600v3">NeAR: Coupled Neural Asset-Renderer Stack</a></div>
    <div class="paper-meta">
      📅 2026-03-29
      | 💬 Accepted by CVPR 2026. The project page: https://near-project.github.io/
    </div>
    <details class="paper-abstract">
      Neural asset authoring and neural rendering have traditionally evolved as disjoint paradigms: one generates digital assets for fixed graphics pipelines, while the other maps conventional assets to images. However, treating them as independent entities limits the potential for end-to-end optimization in fidelity and consistency. In this paper, we bridge this gap with NeAR, a Coupled Neural Asset--Renderer Stack. We argue that co-designing the asset representation and the renderer creates a robust "contract" for superior generation. On the asset side, we introduce the Lighting-Homogenized SLAT (LH-SLAT). Leveraging a rectified-flow model, NeAR lifts casually lit single images into a canonical, illumination-invariant latent space, effectively suppressing baked-in shadows and highlights. On the renderer side, we design a lighting-aware neural decoder tailored to interpret these homogenized latents. Conditioned on HDR environment maps and camera views, it synthesizes relightable 3D Gaussian splats in real-time without per-object optimization. We validate NeAR on four tasks: (1) G-buffer-based forward rendering, (2) random-lit reconstruction, (3) unknown-lit relighting, and (4) novel-view relighting. Extensive experiments demonstrate that our coupled stack outperforms state-of-the-art baselines in both quantitative metrics and perceptual quality. We hope this coupled asset-renderer perspective inspires future graphics stacks that view neural assets and renderers as co-designed components instead of independent entities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17227v2">Adaptive Anchor Policies for Efficient 4D Gaussian Streaming</a></div>
    <div class="paper-meta">
      📅 2026-03-28
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction with Gaussian Splatting has enabled efficient streaming for real-time rendering and free-viewpoint video. However, most pipelines rely on fixed anchor selection such as Farthest Point Sampling (FPS), typically using 8,192 anchors regardless of scene complexity, which over-allocates computation under strict budgets. We propose Efficient Gaussian Streaming (EGS), a plug-in, budget-aware anchor sampler that replaces FPS with a reinforcement-learned policy while keeping the Gaussian streaming reconstruction backbone unchanged. The policy jointly selects an anchor budget and a subset of informative anchors under discrete constraints, balancing reconstruction quality and runtime using spatial features of the Gaussian representation. We evaluate EGS in two settings: fast rendering, which prioritizes runtime efficiency, and high-quality refinement, which enables additional optimization. Experiments on dynamic multi-view datasets show consistent improvements in the quality--efficiency trade-off over FPS sampling. On unseen data, in fast rendering at 256 anchors ($32\times$ fewer than 8,192), EGS improves PSNR by $+0.52$--$0.61$\,dB while running $1.29$--$1.35\times$ faster than IGS@8192 (N3DV and MeetingRoom). In high-quality refinement, EGS remains competitive with the full-anchor baseline at substantially lower anchor budgets. \emph{Code and pretrained checkpoints will be released upon acceptance.} \keywords{4D Gaussian Splatting \and 4D Gaussian Streaming \and Reinforcement Learning}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02172v2">SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-28
      | 💬 Project Page: https://splatsure.github.io/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables high-quality novel view synthesis, motivating interest in generating higher-resolution renders than those available during training. A natural strategy is to apply super-resolution (SR) to low-resolution (LR) input views, but independently enhancing each image introduces multi-view inconsistencies, leading to blurry renders. Prior methods attempt to mitigate these inconsistencies through learned neural components, temporally consistent video priors, or joint optimization on LR and SR views, but all uniformly apply SR across every image. In contrast, our key insight is that close-up LR views may contain high-frequency information for regions also captured in more distant views and that we can use the camera pose relative to scene geometry to inform where to add SR content. Building on this insight, we propose SplatSuRe, a method that selectively applies SR content only in undersampled regions lacking high-frequency supervision, yielding sharper and more consistent results. Across Tanks & Temples, Deep Blending, and Mip-NeRF 360, our approach surpasses baselines in both fidelity and perceptual quality. Notably, our gains are most significant in localized foreground regions where higher detail is desired.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.27151v1">DiffSoup: Direct Differentiable Rasterization of Triangle Soup for Extreme Radiance Field Simplification</a></div>
    <div class="paper-meta">
      📅 2026-03-28
    </div>
    <details class="paper-abstract">
      Radiance field reconstruction aims to recover high-quality 3D representations from multi-view RGB images. Recent advances, such as 3D Gaussian splatting, enable real-time rendering with high visual fidelity on sufficiently powerful graphics hardware. However, efficient online transmission and rendering across diverse platforms requires drastic model simplification, reducing the number of primitives by several orders of magnitude. We introduce DiffSoup, a radiance field representation that employs a soup (i.e., a highly unstructured set) of a small number of triangles with neural textures and binary opacity. We show that this binary opacity representation is directly differentiable via stochastic opacity masking, enabling stable training without a mollifier (i.e., smooth rasterization). DiffSoup can be rasterized using standard depth testing, enabling seamless integration into traditional graphics pipelines and interactive rendering on consumer-grade laptops and mobile devices. Code is available at https://github.com/kenji-tojo/diffsoup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17354v4">PocketGS: On-Device Training of 3D Gaussian Splatting for High Perceptual Modeling</a></div>
    <div class="paper-meta">
      📅 2026-03-28
    </div>
    <details class="paper-abstract">
      Efficient and high-fidelity 3D scene modeling is a long-standing pursuit in computer graphics. While recent 3D Gaussian Splatting (3DGS) methods achieve impressive real-time modeling performance, they rely on resource-unconstrained training assumptions that fail on mobile devices, which are limited by minute-scale training budgets and hardware-available peak-memory. We present PocketGS, a mobile scene modeling paradigm that enables on-device 3DGS training under these tightly coupled constraints while preserving high perceptual fidelity. Our method resolves the fundamental contradictions of standard 3DGS through three co-designed operators: G builds geometry-faithful point-cloud priors; I injects local surface statistics to seed anisotropic Gaussians, thereby reducing early conditioning gaps; and T unrolls alpha compositing with cached intermediates and index-mapped gradient scattering for stable mobile backpropagation. Collectively, these operators satisfy the competing requirements of training efficiency, memory compactness, and modeling fidelity. Extensive experiments demonstrate that PocketGS is able to outperform the powerful mainstream workstation 3DGS baseline to deliver high-quality reconstructions, enabling a fully on-device, practical capture-to-rendering workflow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.21332v2">EmoTaG: Emotion-Aware Talking Head Synthesis on Gaussian Splatting with Few-Shot Personalization</a></div>
    <div class="paper-meta">
      📅 2026-03-28
      | 💬 Accepted by CVPR 2026. Page: https://emotag26.github.io/
    </div>
    <details class="paper-abstract">
      Audio-driven 3D talking head synthesis has advanced rapidly with Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). By leveraging rich pre-trained priors, few-shot methods enable instant personalization from just a few seconds of video. However, under expressive facial motion, existing few-shot approaches often suffer from geometric instability and audio-emotion mismatch, highlighting the need for more effective emotion-aware motion modeling. In this work, we present EmoTaG, a few-shot emotion-aware 3D talking head synthesis framework built on the Pretrain-and-Adapt paradigm. Our key insight is to reformulate motion prediction in a structured FLAME parameter space rather than directly deforming 3D Gaussians, thereby introducing explicit geometric priors that improve motion stability. Building upon this, we propose a Gated Residual Motion Network (GRMN), which captures emotional prosody from audio while supplementing head pose and upper-face cues absent from audio, enabling expressive and coherent motion generation. Extensive experiments demonstrate that EmoTaG achieves state-of-the-art performance in emotional expressiveness, lip synchronization, visual realism, and motion stability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.07917v4">SpeeDe3DGS: Speedy Deformable 3D Gaussian Splatting with Temporal Pruning and Motion Grouping</a></div>
    <div class="paper-meta">
      📅 2026-03-27
      | 💬 Project Page: https://speede3dgs.github.io/
    </div>
    <details class="paper-abstract">
      Dynamic extensions of 3D Gaussian Splatting (3DGS) achieve high-quality reconstructions through neural motion fields, but per-Gaussian neural inference makes these models computationally expensive. Building on DeformableGS, we introduce Speedy Deformable 3D Gaussian Splatting (SpeeDe3DGS), which bridges this efficiency-fidelity gap through three complementary modules: Temporal Sensitivity Pruning (TSP) removes low-impact Gaussians via temporally aggregated sensitivity analysis, Temporal Sensitivity Sampling (TSS) perturbs timestamps to suppress floaters and improve temporal coherence, and GroupFlow distills the learned deformation field into shared SE(3) transformations for efficient groupwise motion. On the 50 dynamic scenes in MonoDyGauBench, integrating TSP and TSS into DeformableGS accelerates rendering by 6.78$\times$ on average while maintaining neural-field fidelity and using 10$\times$ fewer primitives. Adding GroupFlow culminates in 13.71$\times$ faster rendering and 2.53$\times$ shorter training, surpassing all baselines in speed while preserving superior image quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25053v2">GaussFusion: Improving 3D Reconstruction in the Wild with A Geometry-Informed Video Generator</a></div>
    <div class="paper-meta">
      📅 2026-03-27
      | 💬 CVPR 2026 main paper camera-ready. Project page: http://research.zhuliyuan.net/projects/GaussFusion/
    </div>
    <details class="paper-abstract">
      We present GaussFusion, a novel approach for improving 3D Gaussian splatting (3DGS) reconstructions in the wild through geometry-informed video generation. GaussFusion mitigates common 3DGS artifacts, including floaters, flickering, and blur caused by camera pose errors, incomplete coverage, and noisy geometry initialization. Unlike prior RGB-based approaches limited to a single reconstruction pipeline, our method introduces a geometry-informed video-to-video generator that refines 3DGS renderings across both optimization-based and feed-forward methods. Given an existing reconstruction, we render a Gaussian primitive video buffer encoding depth, normals, opacity, and covariance, which the generator refines to produce temporally coherent, artifact-free frames. We further introduce an artifact synthesis pipeline that simulates diverse degradation patterns, ensuring robustness and generalization. GaussFusion achieves state-of-the-art performance on novel-view synthesis benchmarks, and an efficient variant runs in real time at 15 FPS while maintaining similar performance, enabling interactive 3D applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.26665v1">Detailed Geometry and Appearance from Opportunistic Motion</a></div>
    <div class="paper-meta">
      📅 2026-03-27
    </div>
    <details class="paper-abstract">
      Reconstructing 3D geometry and appearance from a sparse set of fixed cameras is a foundational task with broad applications, yet it remains fundamentally constrained by the limited viewpoints. We show that this bound can be broken by exploiting opportunistic object motion: as a person manipulates an object~(e.g., moving a chair or lifting a mug), the static cameras effectively ``orbit'' the object in its local coordinate frame, providing additional virtual viewpoints. Harnessing this object motion, however, poses two challenges: the tight coupling of object pose and geometry estimation and the complex appearance variations of a moving object under static illumination. We address these by formulating a joint pose and shape optimization using 2D Gaussian splatting with alternating minimization of 6DoF trajectories and primitive parameters, and by introducing a novel appearance model that factorizes diffuse and specular components with reflected directional probing within the spherical harmonics space. Extensive experiments on synthetic and real-world datasets with extremely sparse viewpoints demonstrate that our method recovers significantly more accurate geometry and appearance than state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.26638v1">Drive-Through 3D Vehicle Exterior Reconstruction via Dynamic-Scene SfM and Distortion-Aware Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-27
      | 💬 8 pages, 7 figures, Submitted to IEEE IROS 2026 (under review)
    </div>
    <details class="paper-abstract">
      High-fidelity 3D reconstruction of vehicle exteriors improves buyer confidence in online automotive marketplaces, but generating these models in cluttered dealership drive-throughs presents severe technical challenges. Unlike static-scene photogrammetry, this setting features a dynamic vehicle moving against heavily cluttered, static backgrounds. This problem is further compounded by wide-angle lens distortion, specular automotive paint, and non-rigid wheel rotations that violate classical epipolar constraints. We propose an end-to-end pipeline utilizing a two-pillar camera rig. First, we resolve dynamic-scene ambiguities by coupling SAM 3 for instance segmentation with motion-gating to cleanly isolate the moving vehicle, explicitly masking out non-rigid wheels to enforce strict epipolar geometry. Second, we extract robust correspondences directly on raw, distorted 4K imagery using the RoMa v2 learned matcher guided by semantic confidence masks. Third, these matches are integrated into a rig-aware SfM optimization that utilizes CAD-derived relative pose priors to eliminate scale drift. Finally, we use a distortion-aware 3D Gaussian Splatting framework (3DGUT) coupled with a stochastic Markov Chain Monte Carlo (MCMC) densification strategy to render reflective surfaces. Evaluations on 25 real-world vehicles across 10 dealerships demonstrate that our full pipeline achieves a PSNR of 28.66 dB, an SSIM of 0.89, and an LPIPS of 0.21 on held-out views, representing a 3.85 dB improvement over standard 3D-GS, delivering inspection-grade interactive 3D models without controlled studio infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16542v2">EOGS++: Earth Observation Gaussian Splatting with Internal Camera Refinement and Direct Panchromatic Rendering</a></div>
    <div class="paper-meta">
      📅 2026-03-27
      | 💬 8 pages, ISPRS
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting has been introduced as a compelling alternative to NeRF for Earth observation, offering competitive reconstruction quality with significantly reduced training times. In this work, we extend the Earth Observation Gaussian Splatting (EOGS) framework to propose EOGS++, a novel method tailored for satellite imagery that directly operates on raw high-resolution panchromatic data without requiring external preprocessing. Furthermore, leveraging optical flow techniques we embed bundle adjustment directly within the training process, avoiding reliance on external optimization tools while improving camera pose estimation. We also introduce several improvements to the original implementation, including early stopping and TSDF post-processing, all contributing to sharper reconstructions and better geometric accuracy. Experiments on the IARPA 2016 and DFC2019 datasets demonstrate that EOGS++ achieves state-of-the-art performance in terms of reconstruction quality and efficiency, outperforming the original EOGS method and other NeRF-based methods while maintaining the computational advantages of Gaussian Splatting. Our model demonstrates an improvement from 1.33 to 1.19 mean MAE errors on buildings compared to the original EOGS models
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06909v3">Gaussian Mapping for Evolving Scenes</a></div>
    <div class="paper-meta">
      📅 2026-03-27
    </div>
    <details class="paper-abstract">
      Mapping systems with novel view synthesis (NVS) capabilities, most notably 3D Gaussian Splatting (3DGS), are widely used in computer vision, as well as in various applications, including augmented reality, robotics, and autonomous driving. However, many current approaches are limited to static scenes. While recent works have begun addressing short-term dynamics (motion within the camera's view), long-term dynamics (the scene evolving through changes out of view) remain less explored. To overcome this limitation, we introduce a dynamic scene adaptation mechanism to continuously update 3DGS to reflect the latest changes. Since maintaining consistency remains challenging due to stale observations disrupting the reconstruction process, we further propose a novel keyframe management mechanism that discards outdated observations while preserving as much information as possible. We thoroughly evaluate Gaussian Mapping for Evolving Scenes (GaME) on both synthetic and real-world datasets, achieving a 29.7% improvement in PSNR and a 3 times improvement in L1 depth error over the most competitive baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.22225v2">ExtrinSplat: Decoupling Geometry and Semantics for Open-Vocabulary Understanding in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-27
      | 💬 Accepted to CVPR 2026
    </div>
    <details class="paper-abstract">
      Lifting 2D open-vocabulary understanding into 3D Gaussian Splatting (3DGS) scenes is a critical challenge. Mainstream methods, built on an embedding paradigm, suffer from three key flaws: (i) geometry-semantic inconsistency, where points, rather than objects, serve as the semantic basis, limiting semantic fidelity; (ii) semantic bloat from injecting gigabytes of feature data into the geometry; and (iii) semantic rigidity, as one feature per Gaussian struggles to capture rich polysemy. To overcome these limitations, we introduce ExtrinSplat, a framework built on the extrinsic paradigm that decouples geometry from semantics. Instead of embedding features, ExtrinSplat clusters Gaussians into multi-granularity, overlapping 3D object groups. A Vision-Language Model (VLM) then interprets these groups to generate lightweight textual hypotheses, creating an extrinsic index layer that natively supports complex polysemy. By replacing costly feature embedding with lightweight indices, ExtrinSplat reduces scene adaptation time from hours to minutes and lowers storage overhead by several orders of magnitude. On benchmark tasks for open-vocabulary 3D object selection and semantic segmentation, ExtrinSplat outperforms established embedding-based frameworks, validating the efficacy and efficiency of the proposed extrinsic paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00850v2">Smol-GS: Compact Representations for Abstract 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-27
    </div>
    <details class="paper-abstract">
      We present Smol-GS, a novel method for learning compact representations for 3D Gaussian Splatting (3DGS). Our approach learns highly efficient splat-wise features to model 3D space which capture abstracted cues, including color, opacity, transformation, and material properties. We propose octree-derived positional encoding, which explicitly models spatial locality and enhances representation efficiency. We further apply entropy-based compression to exploit feature redundancy, and compress splat coordinates using a recursive voxel hierarchy. This design enables orders-of-magnitude storage reduction while preserving representation flexibility. Smol-GS achieves state-of-the-art compression performance on standard benchmarks with high-level rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.26181v1">GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport</a></div>
    <div class="paper-meta">
      📅 2026-03-27
      | 💬 CVPR 2026, Project page: https://youngju-na.github.io/GLINT
    </div>
    <details class="paper-abstract">
      While 3D Gaussian splatting has emerged as a powerful paradigm, it fundamentally fails to model transparency such as glass panels. The core challenge lies in decoupling the intertwined radiance contributions from transparent interfaces and the transmitted geometry observed through the glass. We present GLINT, a framework that models scene-scale transparency through explicit decomposed Gaussian representation. GLINT reconstructs the primary interface and models reflected and transmitted radiance separately, enabling consistent radiance transport. During optimization, GLINT bootstraps transparency localization from geometry-separation cues induced by the decomposition, together with geometry and material priors from a pre-trained video relighting model. Extensive experiments demonstrate consistent improvements over prior methods for reconstructing complex transparent scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.24994v2">Relaxed Rigidity with Ray-based Grouping for Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-27
      | 💬 24 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The reconstruction of dynamic 3D scenes using 3D Gaussian Splatting has shown significant promise. A key challenge, however, remains in modeling realistic motion, as most methods fail to align the motion of Gaussians with real-world physical dynamics. This misalignment is particularly problematic for monocular video datasets, where failing to maintain coherent motion undermines local geometric structure, ultimately leading to degraded reconstruction quality. Consequently, many state-of-the-art approaches rely heavily on external priors, such as optical flow or 2D tracks, to enforce temporal coherence. In this work, we propose a novel method to explicitly preserve the local geometric structure of Gaussians across time in 4D scenes. Our core idea is to introduce a view-space ray grouping strategy that clusters Gaussians intersected by the same ray, considering only those whose $α$-blending weights exceed a threshold. We then apply constraints to these groups to maintain a consistent spatial distribution, effectively preserving their local geometry. This approach enforces a more physically plausible motion model by ensuring that local geometry remains stable over time, eliminating the reliance on external guidance. We demonstrate the efficacy of our method by integrating it into two distinct baseline models. Extensive experiments on challenging monocular datasets show that our approach significantly outperforms existing methods, achieving superior temporal consistency and reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09962v2">VG-Mapping: Variation-aware Density Control for Online 3D Gaussian Mapping in Semi-static Scenes</a></div>
    <div class="paper-meta">
      📅 2026-03-27
    </div>
    <details class="paper-abstract">
      Maintaining an up-to-date map that accurately reflects recent changes in the environment is crucial, especially for robots that repeatedly traverse the same space. Failing to promptly update the changed regions can degrade map quality, resulting in poor localization, inefficient operations, and even lost robots. 3D Gaussian Splatting (3DGS) has recently seen widespread adoption in online map reconstruction due to its dense, differentiable, and photorealistic properties, yet accurately and efficiently updating the regions of change remains a challenge. In this paper, we propose VG-Mapping, a novel online 3DGS-based mapping system tailored for such semi-static scenes. Our approach introduces a variation-aware density control strategy that decouples Gaussian density regulation from optimization. Specifically, we identify regions with variation to guide initialization and pruning, which avoids the use of stale information in defining the starting point for the subsequent optimization. Furthermore, to address the absence of public benchmarks for this task, we construct a RGB-D dataset comprising both synthetic and real-world semi-static environments. Experimental results demonstrate that our method substantially improves the rendering quality and map update efficiency in semi-static scenes. The code and dataset are available at https://github.com/heyicheng-never/VG-Mapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.26067v1">R-PGA: Robust Physical Adversarial Camouflage Generation via Relightable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-27
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Physical adversarial camouflage poses a severe security threat to autonomous driving systems by mapping adversarial textures onto 3D objects. Nevertheless, current methods remain brittle in complex dynamic scenarios, failing to generalize across diverse geometric (e.g., viewing configurations) and radiometric (e.g., dynamic illumination, atmospheric scattering) variations. We attribute this deficiency to two fundamental limitations in simulation and optimization. First, the reliance on coarse, oversimplified simulations (e.g., via CARLA) induces a significant domain gap, confining optimization to a biased feature space. Second, standard strategies targeting average performance result in a rugged loss landscape, leaving the camouflage vulnerable to configuration shifts.To bridge these gaps, we propose the Relightable Physical 3D Gaussian Splatting (3DGS) based Attack framework (R-PGA). Technically, to address the simulation fidelity issue, we leverage 3DGS to ensure photo-realistic reconstruction and augment it with physically disentangled attributes to decouple intrinsic material from lighting. Furthermore, we design a hybrid rendering pipeline that leverages precise Relightable 3DGS for foreground rendering, while employing a pre-trained image translation model to synthesize plausible relighted backgrounds that align with the relighted foreground.To address the optimization robustness issue, we propose the Hard Physical Configuration Mining (HPCM) module, designed to actively mine worst-case physical configurations and suppress their corresponding loss peaks. This strategy not only diminishes the overall loss magnitude but also effectively flattens the rugged loss landscape, ensuring consistent adversarial effectiveness and robustness across varying physical configurations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25745v1">Less Gaussians, Texture More: 4K Feed-Forward Textured Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-26
    </div>
    <details class="paper-abstract">
      Existing feed-forward 3D Gaussian Splatting methods predict pixel-aligned primitives, leading to a quadratic growth in primitive count as resolution increases. This fundamentally limits their scalability, making high-resolution synthesis such as 4K intractable. We introduce LGTM (Less Gaussians, Texture More), a feed-forward framework that overcomes this resolution scaling barrier. By predicting compact Gaussian primitives coupled with per-primitive textures, LGTM decouples geometric complexity from rendering resolution. This approach enables high-fidelity 4K novel view synthesis without per-scene optimization, a capability previously out of reach for feed-forward methods, all while using significantly fewer Gaussian primitives. Project page: https://yxlao.github.io/lgtm/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.21936v2">Cross-Instance Gaussian Splatting Registration via Geometry-Aware Feature-Guided Alignment</a></div>
    <div class="paper-meta">
      📅 2026-03-26
      | 💬 Accepted to CVPR 2026
    </div>
    <details class="paper-abstract">
      We present Gaussian Splatting Alignment (GSA), a novel method for aligning two independent 3D Gaussian Splatting (3DGS) models via a similarity transformation (rotation, translation, and scale), even when they are of different objects in the same category (e.g., different cars). In contrast, existing methods can only align 3DGS models of the same object (e.g., the same car) and often must be given true scale as input, while we estimate it successfully. GSA leverages viewpoint-guided spherical map features to obtain robust correspondences and introduces a two-step optimization framework that aligns 3DGS models while keeping them fixed. First, we apply an iterative feature-guided absolute orientation solver as our coarse registration, which is robust to poor initialization (e.g., 180 degrees misalignment or a 10x scale gap). Next, we use a fine registration step that enforces multi-view feature consistency, inspired by inverse radiance-field formulations. The first step already achieves state-of-the-art performance, and the second further improves results. In the same-object case, GSA outperforms prior works, often by a large margin, even when the other methods are given the true scale. In the harder case of different objects in the same category, GSA vastly surpasses them, providing the first effective solution for category-level 3DGS registration and unlocking new applications. Project webpage: https://bgu-cs-vil.github.io/GSA-project/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.26814v1">arg-VU: Affordance Reasoning with Physics-Aware 3D Geometry for Visual Understanding in Robotic Surgery</a></div>
    <div class="paper-meta">
      📅 2026-03-26
    </div>
    <details class="paper-abstract">
      Affordance reasoning provides a principled link between perception and action, yet remains underexplored in surgical robotics, where tissues are highly deformable, compliant, and dynamically coupled with tool motion. We present arg-VU, a physics-aware affordance reasoning framework that integrates temporally consistent geometry tracking with constraint-induced mechanical modeling for surgical visual understanding. Surgical scenes are reconstructed using 3D Gaussian Splatting (3DGS) and converted into a temporally tracked surface representation. Extended Position-Based Dynamics (XPBD) embeds local deformation constraints and produces representative geometry points (RGPs) whose constraint sensitivities define anisotropic stiffness metrics capturing the local constraint-manifold geometry. Robotic tool poses in SE(3) are incorporated to compute rigidly induced displacements at RGPs, from which we derive two complementary measures: a physics-aware compliance energy that evaluates mechanical feasibility with respect to local deformation constraints, and a positional agreement score that captures motion alignment (as kinematic motion baseline). Experiments on surgical video datasets show that arg-VU yields more stable, physically consistent, and interpretable affordance predictions than kinematic baselines. These results demonstrate that physics-aware geometric representations enable reliable affordance reasoning for deformable surgical environments and support embodied robotic interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19682v2">3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-26
      | 💬 Accepted by CVPR 2026. Project page: https://takeshie.github.io/GSPrior
    </div>
    <details class="paper-abstract">
      Rendering 3D surfaces has been revolutionized within the modeling of radiance fields through either 3DGS or NeRF. Although 3DGS has shown advantages over NeRF in terms of rendering quality or speed, there is still room for improvement in recovering high fidelity surfaces through 3DGS. To resolve this issue, we propose a self-constrained prior to constrain the learning of 3D Gaussians, aiming for more accurate depth rendering. Our self-constrained prior is derived from a TSDF grid that is obtained by fusing the depth maps rendered with current 3D Gaussians. The prior measures a distance field around the estimated surface, offering a band centered at the surface for imposing more specific constraints on 3D Gaussians, such as removing Gaussians outside the band, moving Gaussians closer to the surface, and encouraging larger or smaller opacity in a geometry-aware manner. More importantly, our prior can be regularly updated by the most recent depth images which are usually more accurate and complete. In addition, the prior can also progressively narrow the band to tighten the imposed constraints. We justify our idea and report our superiority over the state-of-the-art methods in evaluations on widely used benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.23324v2">Pose-Free Omnidirectional Gaussian Splatting for 360-Degree Videos with Consistent Depth Priors</a></div>
    <div class="paper-meta">
      📅 2026-03-26
    </div>
    <details class="paper-abstract">
      Omnidirectional 3D Gaussian Splatting with panoramas is a key technique for 3D scene representation, and existing methods typically rely on slow SfM to provide camera poses and sparse points priors. In this work, we propose a pose-free omnidirectional 3DGS method, named PFGS360, that reconstructs 3D Gaussians from unposed omnidirectional videos. To achieve accurate camera pose estimation, we first construct a spherical consistency-aware pose estimation module, which recovers poses by establishing consistent 2D-3D correspondences between the reconstructed Gaussians and the unposed images using Gaussians' internal depth priors. Besides, to enhance the fidelity of novel view synthesis, we introduce a depth-inlier-aware densification module to extract depth inliers and Gaussian outliers with consistent monocular depth priors, enabling efficient Gaussian densification and achieving photorealistic novel view synthesis. The experiments show significant outperformance over existing pose-free and pose-aware 3DGS methods on both real-world and synthetic 360-degree videos. Code is available at https://github.com/zcq15/PFGS360.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.23637v2">Stochastic Ray Tracing for the Reconstruction of 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-26
      | 💬 Project Page: https://xupaya.github.io/stoch3DGS/
    </div>
    <details class="paper-abstract">
      Ray-tracing-based 3D Gaussian splatting (3DGS) methods overcome the limitations of rasterization -- rigid pinhole camera assumptions, inaccurate shadows, and lack of native reflection or refraction -- but remain slower due to the cost of sorting all intersecting Gaussians along every ray. Moreover, existing ray-tracing methods still rely on rasterization-style approximations such as shadow mapping for relightable scenes, undermining the generality that ray tracing promises. We present a differentiable, sorting-free stochastic formulation for ray-traced 3DGS -- the first framework that uses stochastic ray tracing to both reconstruct and render standard and relightable 3DGS scenes. At its core is an unbiased Monte Carlo estimator for pixel-color gradients that evaluates only a small sampled subset of Gaussians per ray, bypassing the need for sorting. For standard 3DGS, our method matches the reconstruction quality and speed of rasterization-based 3DGS while substantially outperforming sorting-based ray tracing. For relightable 3DGS, the same stochastic estimator drives per-Gaussian shading with fully ray-traced shadow rays, delivering notably higher reconstruction fidelity than prior work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25265v1">ViewSplat: View-Adaptive Dynamic Gaussian Splatting for Feed-Forward Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-03-26
      | 💬 24 pages, 10 figures
    </div>
    <details class="paper-abstract">
      We present ViewSplat, a view-adaptive 3D Gaussian splatting network for novel view synthesis from unposed images. While recent feed-forward 3D Gaussian splatting has significantly accelerated 3D scene reconstruction by bypassing per-scene optimization, a fundamental fidelity gap remains. We attribute this bottleneck to the limited capacity of single-step feed-forward networks to regress static Gaussian primitives that satisfy all viewpoints. To address this limitation, we shift the paradigm from static primitive regression to view-adaptive dynamic splatting. Instead of a rigid Gaussian representation, our pipeline learns a view-adaptable latent representation. Specifically, ViewSplat initially predicts base Gaussian primitives alongside the weights of dynamic MLPs. During rendering, these MLPs take target view coordinates as input and predict view-dependent residual updates for each Gaussian attribute (i.e., 3D position, scale, rotation, opacity, and color). This mechanism, which we term view-adaptive dynamic splatting, allows each primitive to rectify initial estimation errors, effectively capturing high-fidelity appearances. Extensive experiments demonstrate that ViewSplat achieves state-of-the-art fidelity while maintaining fast inference (17 FPS) and real-time rendering (154 FPS).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09270v2">MoRel: Long-Range Flicker-Free 4D Motion Modeling via Anchor Relay-based Bidirectional Blending with Hierarchical Densification</a></div>
    <div class="paper-meta">
      📅 2026-03-26
      | 💬 CVPR 2026 (camera ready ver.). The first two authors contributed equally to this work (equal contribution). Please visit our project page at https://cmlab-korea.github.io/MoRel/
    </div>
    <details class="paper-abstract">
      Recent advances in 4D Gaussian Splatting (4DGS) have extended the high-speed rendering capability of 3D Gaussian Splatting (3DGS) into the temporal domain, enabling real-time rendering of dynamic scenes. However, one of the major remaining challenges lies in modeling long-range motion-contained dynamic videos, where a naive extension of existing methods leads to severe memory explosion, temporal flickering, and failure to handle appearing or disappearing occlusions over time. To address these challenges, we propose a novel 4DGS framework characterized by an Anchor Relay-based Bidirectional Blending (ARBB) mechanism, named MoRel, which enables temporally consistent and memory-efficient modeling of long-range dynamic scenes. Our method progressively constructs locally canonical anchor spaces at key-frame time index and models inter-frame deformations at the anchor level, enhancing temporal coherence. By learning bidirectional deformations between KfA and adaptively blending them through learnable opacity control, our approach mitigates temporal discontinuities and flickering artifacts. We further introduce a Feature-variance-guided Hierarchical Densification (FHD) scheme that effectively densifies KfA's while keeping rendering quality, based on an assigned level of feature-variance. To effectively evaluate our model's capability to handle real-world long-range 4D motion, we newly compose long-range 4D motion-contained dataset, called SelfCap$_{\text{LR}}$. It has larger average dynamic motion magnitude, captured at spatially wider spaces, compared to previous dynamic video datasets. Overall, our MoRel achieves temporally coherent and flicker-free long-range 4D reconstruction while maintaining bounded memory usage, demonstrating both scalability and efficiency in dynamic Gaussian-based representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25129v1">AirSplat: Alignment and Rating for Robust Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-26
      | 💬 Project page: https://kaist-viclab.github.io/airsplat-site
    </div>
    <details class="paper-abstract">
      While 3D Vision Foundation Models (3DVFMs) have demonstrated remarkable zero-shot capabilities in visual geometry estimation, their direct application to generalizable novel view synthesis (NVS) remains challenging. In this paper, we propose AirSplat, a novel training framework that effectively adapts the robust geometric priors of 3DVFMs into high-fidelity, pose-free NVS. Our approach introduces two key technical contributions: (1) Self-Consistent Pose Alignment (SCPA), a training-time feedback loop that ensures pixel-aligned supervision to resolve pose-geometry discrepancy; and (2) Rating-based Opacity Matching (ROM), which leverages the local 3D geometry consistency knowledge from a sparse-view NVS teacher model to filter out degraded primitives. Experimental results on large-scale benchmarks demonstrate that our method significantly outperforms state-of-the-art pose-free NVS approaches in reconstruction quality. Our AirSplat highlights the potential of adapting 3DVFMs to enable simultaneous visual geometry estimation and high-quality view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03824v3">IDESplat: Iterative Depth Probability Estimation for Generalizable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-26
    </div>
    <details class="paper-abstract">
      Generalizable 3D Gaussian Splatting aims to directly predict Gaussian parameters using a feed-forward network for scene reconstruction. Among these parameters, Gaussian means are particularly difficult to predict, so depth is usually estimated first and then unprojected to obtain the Gaussian sphere centers. Existing methods typically rely solely on a single warp to estimate depth probability, which hinders their ability to fully leverage cross-view geometric cues, resulting in unstable and coarse depth maps. To address this limitation, we propose IDESplat, which iteratively applies warp operations to boost depth probability estimation for accurate Gaussian mean prediction. First, to eliminate the inherent instability of a single warp, we introduce a Depth Probability Boosting Unit (DPBU) that integrates epipolar attention maps produced by cascading warp operations in a multiplicative manner. Next, we construct an iterative depth estimation process by stacking multiple DPBUs, progressively identifying potential depth candidates with high likelihood. As IDESplat iteratively boosts depth probability estimates and updates the depth candidates, the depth map is gradually refined, resulting in accurate Gaussian means. We conduct experiments on RealEstate10K, ACID, and DL3DV. IDESplat achieves outstanding reconstruction quality and state-of-the-art performance with real-time efficiency. On RE10K, it outperforms DepthSplat by 0.33 dB in PSNR, using only 10.7% of the parameters and 70% of the memory. Additionally, our IDESplat improves PSNR by 2.95 dB over DepthSplat on the DTU dataset in cross-dataset experiments, demonstrating its strong generalization ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22666v2">ArtPro: Self-Supervised Articulated Object Reconstruction with Adaptive Integration of Mobility Proposals</a></div>
    <div class="paper-meta">
      📅 2026-03-26
    </div>
    <details class="paper-abstract">
      Reconstructing articulated objects into high-fidelity digital twins is crucial for applications such as robotic manipulation and interactive simulation. Recent self-supervised methods using differentiable rendering frameworks like 3D Gaussian Splatting remain highly sensitive to the initial part segmentation. Their reliance on heuristic clustering or pre-trained models often causes optimization to converge to local minima, especially for complex multi-part objects. To address these limitations, we propose ArtPro, a novel self-supervised framework that introduces adaptive integration of mobility proposals. Our approach begins with an over-segmentation initialization guided by geometry features and motion priors, generating part proposals with plausible motion hypotheses. During optimization, we dynamically merge these proposals by analyzing motion consistency among spatial neighbors, while a collision-aware motion pruning mechanism prevents erroneous kinematic estimation. Extensive experiments on both synthetic and real-world objects demonstrate that ArtPro achieves robust reconstruction of complex multi-part objects, significantly outperforming existing methods in accuracy and stability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25058v1">Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2026-03-26
      | 💬 Accepted to CVPR 2026
    </div>
    <details class="paper-abstract">
      We present an approach for high-quality dynamic Gaussian Splatting from monocular videos. To this end, we in this work go one step further beyond previous methods to explicitly model continuous position and orientation deformation of dynamic Gaussians, using an SE(3) B-spline motion bases with a compact set of control points. To improve computational efficiency while enhancing the ability to model complex motions, an adaptive control mechanism is devised to dynamically adjust the number of motion bases and control points. Besides, we develop a soft segment reconstruction strategy to mitigate long-interval motion interference, and employ a multi-view diffusion model to provide multi-view cues for avoiding overfitting to training views. Extensive experiments demonstrate that our method outperforms state-of-the-art methods in novel view synthesis. Our code is available at https://github.com/hhhddddddd/se3bsplinegs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25042v1">MoRGS: Efficient Per-Gaussian Motion Reasoning for Streamable Dynamic 3D Scenes</a></div>
    <div class="paper-meta">
      📅 2026-03-26
    </div>
    <details class="paper-abstract">
      Online reconstruction of dynamic scenes aims to learn from streaming multi-view inputs under low-latency constraints. The fast training and real-time rendering capabilities of 3D Gaussian Splatting have made on-the-fly reconstruction practically feasible, enabling online 4D reconstruction. However, existing online approaches, despite their efficiency and visual quality, fail to learn per-Gaussian motion that reflects true scene dynamics. Without explicit motion cues, appearance and motion are optimized solely under photometric loss, causing per-Gaussian motion to chase pixel residuals rather than true 3D motion. To address this, we propose MoRGS, an efficient online per-Gaussian motion reasoning framework that explicitly models per-Gaussian motion to improve 4D reconstruction quality. Specifically, we leverage optical flow on a sparse set of key views as lightweight motion cues that regularize per-Gaussian motion beyond photometric supervision. To compensate for the sparsity of flow supervision, we learn a per-Gaussian motion offset field that reconciles discrepancies between projected 3D motion and observed flow across views and time. In addition, we introduce a per-Gaussian motion confidence that separates dynamic from static Gaussians and weights Gaussian attribute residual updates, thereby suppressing redundant motion in static regions for better temporal consistency and accelerating the modeling of large motions. Extensive experiments demonstrate that MoRGS achieves state-of-the-art reconstruction quality and motion fidelity among online methods, while maintaining streamable performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25038v1">$π$, But Make It Fly: Physics-Guided Transfer of VLA Models to Aerial Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-03-26
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models such as $π_0$ have demonstrated remarkable generalization across diverse fixed-base manipulators. However, transferring these foundation models to aerial platforms remains an open challenge due to the fundamental mismatch between the quasi-static dynamics of fixed-base arms and the underactuated, highly dynamic nature of flight. In this work, we introduce AirVLA, a system that investigates the transferability of manipulation-pretrained VLAs to aerial pick-and-place tasks. We find that while visual representations transfer effectively, the specific control dynamics required for flight do not. To bridge this "dynamics gap" without retraining the foundation model, we introduce a Payload-Aware Guidance mechanism that injects payload constraints directly into the policy's flow-matching sampling process. To overcome data scarcity, we further utilize a Gaussian Splatting pipeline to synthesize navigation training data. We evaluate our method through a cumulative 460 real-world experiments which demonstrate that this synthetic data is a key enabler of performance, unlocking 100% success in navigation tasks where directly fine-tuning on teleoperation data alone attains 81% success. Our inference-time intervention, Payload-Aware Guidance, increases real-world pick-and-place task success from 23% to 50%. Finally, we evaluate the model on a long-horizon compositional task, achieving a 62% overall success rate. These results suggest that pre-trained manipulation VLAs, with appropriate data augmentation and physics-informed guidance, can transfer to aerial manipulation and navigation, as well as the composition of these tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.24725v1">Confidence-Based Mesh Extraction from 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-03-25
      | 💬 Project Page: https://r4dl.github.io/CoMe/
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) greatly accelerated mesh extraction from posed images due to its explicit representation and fast software rasterization. While the addition of geometric losses and other priors has improved the accuracy of extracted surfaces, mesh extraction remains difficult in scenes with abundant view-dependent effects. To resolve the resulting ambiguities, prior works rely on multi-view techniques, iterative mesh extraction, or large pre-trained models, sacrificing the inherent efficiency of 3DGS. In this work, we present a simple and efficient alternative by introducing a self-supervised confidence framework to 3DGS: within this framework, learnable confidence values dynamically balance photometric and geometric supervision. Extending our confidence-driven formulation, we introduce losses which penalize per-primitive color and normal variance and demonstrate their benefits to surface extraction. Finally, we complement the above with an improved appearance model, by decoupling the individual terms of the D-SSIM loss. Our final approach delivers state-of-the-art results for unbounded meshes while remaining highly efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.24716v1">Accurate Point Measurement in 3DGS -- A New Alternative to Traditional Stereoscopic-View Based Measurements</a></div>
    <div class="paper-meta">
      📅 2026-03-25
      | 💬 Accepted to the 2026 ISPRS Congress
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has revolutionized real-time rendering with its state-of-the-art novel view synthesis, but its utility for accurate geometric measurement remains underutilized. Compared to multi-view stereo (MVS) point clouds or meshes, 3DGS rendered views present superior visual quality and completeness. However, current point measurement methods still rely on demanding stereoscopic workstations or direct picking on often-incomplete and inaccurate 3D meshes. As a novel view synthesizer, 3DGS renders exact source views and smoothly interpolates in-between views. This allows users to intuitively pick congruent points across different views while operating 3DGS models. By triangulating these congruent points, one can precisely generate 3D point measurements. This approach mimics traditional stereoscopic measurement but is significantly less demanding: it requires neither a stereo workstation nor specialized operator stereoscopic capability. Furthermore, it enables multi-view intersection (more than two views) for higher measurement accuracy. We implemented a web-based application to demonstrate this proof-of-concept (PoC). Using several UAV aerial datasets, we show this PoC allows users to successfully perform highly accurate point measurements, achieving accuracy matching or exceeding traditional stereoscopic methods on standard hardware. Specifically, our approach significantly outperforms direct mesh-based measurements. Quantitatively, our method achieves RMSEs in the 1-2 cm range on well-defined points. More critically, on challenging thin structures where mesh-based RMSE was 0.062 m, our method achieved 0.037 m. On sharp corners poorly reconstructed in the mesh, our method successfully measured all points with a 0.013 m RMSE, whereas the mesh method failed entirely. Code is available at: https://github.com/GDAOSU/3dgs_measurement_tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.02803v3">HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars</a></div>
    <div class="paper-meta">
      📅 2026-03-25
      | 💬 CVPR 2026, Project page: https://gserifi.github.io/HyperGaussians, Code: https://github.com/gserifi/HyperGaussians
    </div>
    <details class="paper-abstract">
      We introduce HyperGaussians, a novel extension of 3D Gaussian Splatting for high-quality animatable face avatars. Creating such detailed face avatars from videos is a challenging problem and has numerous applications in augmented and virtual reality. While tremendous successes have been achieved for static faces, animatable avatars from monocular videos still fall in the uncanny valley. The de facto standard, 3D Gaussian Splatting (3DGS), represents a face through a collection of 3D Gaussian primitives. 3DGS excels at rendering static faces, but the state-of-the-art still struggles with nonlinear deformations, complex lighting effects, and fine details. While most related works focus on predicting better Gaussian parameters from expression codes, we rethink the 3D Gaussian representation itself and how to make it more expressive. Our insights lead to a novel extension of 3D Gaussians to high-dimensional multivariate Gaussians, dubbed 'HyperGaussians'. The higher dimensionality increases expressivity through conditioning on a learnable local embedding. However, splatting HyperGaussians is computationally expensive because it requires inverting a high-dimensional covariance matrix. We solve this by reparameterizing the covariance matrix, dubbed the 'inverse covariance trick'. This trick boosts the efficiency so that HyperGaussians can be seamlessly integrated into existing models. To demonstrate this, we plug in HyperGaussians into the state-of-the-art in fast monocular face avatars: FlashAvatar. Our evaluation on 19 subjects from 4 face datasets shows that HyperGaussians outperform 3DGS numerically and visually, particularly for high-frequency details like eyeglass frames, teeth, complex facial movements, and specular reflections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.05296v4">Let it Snow! Animating 3D Gaussian Scenes with Dynamic Weather Effects via Physics-Guided Score Distillation</a></div>
    <div class="paper-meta">
      📅 2026-03-25
      | 💬 Accepted to CVPR 2026. Project webpage: https://galfiebelman.github.io/let-it-snow/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has recently enabled fast and photorealistic reconstruction of static 3D scenes. However, dynamic editing of such scenes remains a significant challenge. We introduce a novel framework, Physics-Guided Score Distillation, to address a fundamental conflict: physics simulation provides a strong motion prior that is insufficient for photorealism , while video-based Score Distillation Sampling (SDS) alone cannot generate coherent motion for complex, multi-particle scenarios. We resolve this through a unified optimization framework where physics simulation guides Score Distillation to jointly refine the motion prior for photorealism while simultaneously optimizing appearance. Specifically, we learn a neural dynamics model that predicts particle motion and appearance, optimized end-to-end via a combined loss integrating Video-SDS for photorealism with our physics-guidance prior. This allows for photorealistic refinements while ensuring the dynamics remain plausible. Our framework enables scene-wide dynamic weather effects, including snowfall, rainfall, fog, and sandstorms, with physically plausible motion. Experiments demonstrate our physics-guided approach significantly outperforms baselines, with ablations confirming this joint refinement is essential for generating coherent, high-fidelity dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08334v3">HybridSplat: Fast Reflection-baked Gaussian Tracing using Hybrid Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-25
      | 💬 The authors have decided to withdraw this manuscript to undergo a comprehensive revision of the methodology and data analysis. The current version no longer accurately reflects the final scope and quality of our ongoing research
    </div>
    <details class="paper-abstract">
      Rendering complex reflection of real-world scenes using 3D Gaussian splatting has been a quite promising solution for photorealistic novel view synthesis, but still faces bottlenecks especially in rendering speed and memory storage. This paper proposes a new Hybrid Splatting(HybridSplat) mechanism for Gaussian primitives. Our key idea is a new reflection-baked Gaussian tracing, which bakes the view-dependent reflection within each Gaussian primitive while rendering the reflection using tile-based Gaussian splatting. Then we integrate the reflective Gaussian primitives with base Gaussian primitives using a unified hybrid splatting framework for high-fidelity scene reconstruction. Moreover, we further introduce a pipeline-level acceleration for the hybrid splatting, and reflection-sensitive Gaussian pruning to reduce the model size, thus achieving much faster rendering speed and lower memory storage while preserving the reflection rendering quality. By extensive evaluation, our HybridSplat accelerates about 7x rendering speed across complex reflective scenes from Ref-NeRF, NeRF-Casting with 4x fewer Gaussian primitives than similar ray-tracing based Gaussian splatting baselines, serving as a new state-of-the-art method especially for complex reflective scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.20714v3">Wideband RF Radiance Field Modeling Using Frequency-embedded 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-25
      | 💬 This paper is withdrawn because the technical approach has been significantly updated. The methods and results in this version are no longer representative of the latest research progress
    </div>
    <details class="paper-abstract">
      Indoor environments typically contain diverse RF signals distributed across multiple frequency bands, including NB-IoT, Wi-Fi, and millimeter-wave. Consequently, wideband RF modeling is essential for practical applications such as joint deployment of heterogeneous RF systems, cross-band communication, and distributed RF sensing. Although 3D Gaussian Splatting (3DGS) techniques effectively reconstruct RF radiance fields at a single frequency, they cannot model fields at arbitrary or unknown frequencies across a wide range. In this paper, we present a novel 3DGS algorithm for unified wideband RF radiance field modeling. RF wave propagation depends on signal frequency and the 3D spatial environment, including geometry and material electromagnetic (EM) properties. To address these factors, we introduce a frequency-embedded EM feature network that utilizes 3D Gaussian spheres at each spatial location to learn the relationship between frequency and transmission characteristics, such as attenuation and radiance intensity. With a dataset containing sparse frequency samples in a specific 3D environment, our model can efficiently reconstruct RF radiance fields at arbitrary and unseen frequencies. To assess our approach, we introduce a large-scale power angular spectrum (PAS) dataset with 50,000 samples spanning 1 to 94 GHz across six indoor environments. Experimental results show that the proposed model trained on multiple frequencies achieves a Structural Similarity Index Measure (SSIM) of 0.922 for PAS reconstruction, surpassing state-of-the-art single-frequency 3DGS models with SSIM of 0.863.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.24036v1">SpectralSplats: Robust Differentiable Tracking via Spectral Moment Supervision</a></div>
    <div class="paper-meta">
      📅 2026-03-25
      | 💬 Project page: https://avigailco.github.io/SpectralSplats/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables real-time, photorealistic novel view synthesis, making it a highly attractive representation for model-based video tracking. However, leveraging the differentiability of the 3DGS renderer "in the wild" remains notoriously fragile. A fundamental bottleneck lies in the compact, local support of the Gaussian primitives. Standard photometric objectives implicitly rely on spatial overlap; if severe camera misalignment places the rendered object outside the target's local footprint, gradients strictly vanish, leaving the optimizer stranded. We introduce SpectralSplats, a robust tracking framework that resolves this "vanishing gradient" problem by shifting the optimization objective from the spatial to the frequency domain. By supervising the rendered image via a set of global complex sinusoidal features (Spectral Moments), we construct a global basin of attraction, ensuring that a valid, directional gradient toward the target exists across the entire image domain, even when pixel overlap is completely nonexistent. To harness this global basin without introducing periodic local minima associated with high frequencies, we derive a principled Frequency Annealing schedule from first principles, gracefully transitioning the optimizer from global convexity to precise spatial alignment. We demonstrate that SpectralSplats acts as a seamless, drop-in replacement for spatial losses across diverse deformation parameterizations (from MLPs to sparse control points), successfully recovering complex deformations even from severely misaligned initializations where standard appearance-based tracking catastrophically fails.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.21304v2">F4Splat: Feed-Forward Predictive Densification for Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-25
      | 💬 Project Page: $\href{https://mlvlab.github.io/F4Splat}{\text{this http URL}}$
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting methods enable single-pass reconstruction and real-time rendering. However, they typically adopt rigid pixel-to-Gaussian or voxel-to-Gaussian pipelines that uniformly allocate Gaussians, leading to redundant Gaussians across views. Moreover, they lack an effective mechanism to control the total number of Gaussians while maintaining reconstruction fidelity. To address these limitations, we present F4Splat, which performs Feed-Forward predictive densification for Feed-Forward 3D Gaussian Splatting, introducing a densification-score-guided allocation strategy that adaptively distributes Gaussians according to spatial complexity and multi-view overlap. Our model predicts per-region densification scores to estimate the required Gaussian density and allows explicit control over the final Gaussian budget without retraining. This spatially adaptive allocation reduces redundancy in simple regions and minimizes duplicate Gaussians across overlapping views, producing compact yet high-quality 3D representations. Extensive experiments demonstrate that our model achieves superior novel-view synthesis performance compared to prior uncalibrated feed-forward methods, while using significantly fewer Gaussians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.22792v2">Instrument-Splatting++: Towards Controllable Surgical Instrument Digital Twin Using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-25
      | 💬 10 pages, 9 figures
    </div>
    <details class="paper-abstract">
      High-quality and controllable digital twins of surgical instruments are critical for Real2Sim in robot-assisted surgery, as they enable realistic simulation, synthetic data generation, and perception learning under novel poses. We present Instrument-Splatting++, a monocular 3D Gaussian Splatting (3DGS) framework that reconstructs surgical instruments as a fully controllable Gaussian asset with high fidelity. Our pipeline starts with part-wise geometry pretraining that injects CAD priors into Gaussian primitives and equips the representation with part-aware semantic rendering. Built on the pretrained model, we propose a semantics-aware pose estimation and tracking (SAPET) method to recover per-frame 6-DoF pose and joint angles from unposed endoscopic videos, where a gripper-tip network trained purely from synthetic semantics provides robust supervision and a loose regularization suppresses singular articulations. Finally, we introduce Robust Texture Learning (RTL), which alternates pose refinement and robust appearance optimization, mitigating pose noise during texture learning. The proposed framework can perform pose estimation and learn realistic texture from unposed videos. We validate our method on sequences extracted from EndoVis17/18, SAR-RARP, and an in-house dataset, showing superior photometric quality and improved geometric accuracy over state-of-the-art baselines. We further demonstrate a downstream keypoint detection task where unseen-pose data augmentation from our controllable instrument Gaussian improves performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.23891v1">FilterGS: Traversal-Free Parallel Filtering and Adaptive Shrinking for Large-Scale LoD 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-25
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has revolutionized neural rendering with real-time performance. However, scaling this approach to large scenes using Level-of-Detail methods faces critical challenges: inefficient serial traversal consuming over 60\% of rendering time, and redundant Gaussian-tile pairs that incur unnecessary processing overhead. To address these limitations, we introduce FilterGS, featuring a parallel filtering mechanism with two complementary filters that select Gaussian elements efficiently without tree traversal. Additionally, we propose a novel GTC metric that quantifies the redundancy of Gaussian-tile key-value pairs. Based on this metric, we introduce a scene-adaptive Gaussian shrinking strategy that effectively reduces redundant pairs. Extensive experiments demonstrate that FilterGS achieves state-of-the-art rendering speeds while maintaining competitive visual quality across multiple large-scale datasets. Project page: https://github.com/xenon-w/FilterGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.23686v1">AdvSplat: Adversarial Attacks on Feed-Forward Gaussian Splatting Models</a></div>
    <div class="paper-meta">
      📅 2026-03-24
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is increasingly recognized as a powerful paradigm for real-time, high-fidelity 3D reconstruction. However, its per-scene optimization pipeline limits scalability and generalization, and prevents efficient inference. Recently emerged feed-forward 3DGS models address these limitations by enabling fast reconstruction from a few input views after large-scale pretraining, without scene-specific optimization. Despite their advantages and strong potential for commercial deployment, the use of neural networks as the backbone also amplifies the risk of adversarial manipulation. In this paper, we introduce AdvSplat, the first systematic study of adversarial attacks on feed-forward 3DGS. We first employ white-box attacks to reveal fundamental vulnerabilities of this model family. We then develop two improved, practically relevant, query-efficient black-box algorithms that optimize pixel-space perturbations via a frequency-domain parameterization: one based on gradient estimation and the other gradient-free, without requiring any access to model internals. Extensive experiments across multiple datasets demonstrate that AdvSplat can significantly disrupt reconstruction results by injecting imperceptible perturbations into the input images. Our findings surface an overlooked yet urgent problem in this domain, and we hope to draw the community's attention to this emerging security and robustness challenge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06438v3">AGORA: Adversarial Generation Of Real-time Animatable 3D Gaussian Head Avatars</a></div>
    <div class="paper-meta">
      📅 2026-03-24
      | 💬 Extended the method to support mobile devices; updated experiments, results and supplementary
    </div>
    <details class="paper-abstract">
      The generation of high-fidelity, animatable 3D human avatars remains a core challenge in computer graphics and vision, with applications in VR, telepresence, and entertainment. Existing approaches based on implicit representations like NeRFs suffer from slow rendering and dynamic inconsistencies, while 3D Gaussian Splatting (3DGS) methods are typically limited to static head generation, lacking dynamic control. We bridge this gap by introducing AGORA, a novel framework that extends 3DGS within a generative adversarial network to produce animatable avatars. Our formulation combines spatial shape conditioning with a dual-discriminator training strategy that supervises both rendered appearance and synthetic geometry cues, improving expression fidelity and controllability. To enable practical deployment, we further introduce a simple inference-time approach that extracts Gaussian blendshapes and reuses them for animation on-device. AGORA generates avatars that are visually realistic, precisely controllable, and achieves state-of-the-art performance among animatable generative head-avatar methods. Quantitatively, we render at 560 FPS on a single GPU and 60 FPS on mobile phones, marking a significant step toward practical, high-performance digital humans. Project website: https://ramazan793.github.io/AGORA/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.23192v1">GTLR-GS: Geometry-Texture Aware LiDAR-Regularized 3D Gaussian Splatting for Realistic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-24
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled real-time, photorealistic scene reconstruction. However, conventional 3DGS frameworks typically rely on sparse point clouds derived from Structure-from-Motion (SfM), which inherently suffer from scale ambiguity, limited geometric consistency, and strong view dependency due to the lack of geometric priors. In this work, a LiDAR-centric 3D Gaussian Splatting framework is proposed that explicitly incorporates metric geometric priors into the entire Gaussian optimization process. Instead of treating LiDAR data as a passive initialization source, 3DGS optimization is reformulated as a geometry-conditioned allocation and refinement problem under a fixed representational budget. Specifically, this work introduces (i) a geometry-texture-aware allocation strategy that selectively assigns Gaussian primitives to regions with high structural or appearance complexity, (ii) a curvature-adaptive refinement mechanism that dynamically guides Gaussian splitting toward geometrically complex areas during training, and (iii) a confidence-aware metric depth regularization that anchors the reconstructed geometry to absolute scale using LiDAR measurements while maintaining optimization stability. Extensive experiments on the ScanNet++ dataset and a custom real-world dataset validate the proposed approach. The results demonstrate state-of-the-art performance in metric-scale reconstruction with high geometric fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.21064v2">2Xplat: Two Experts Are Better Than One Generalist</a></div>
    <div class="paper-meta">
      📅 2026-03-24
      | 💬 Project page: https://hwasikjeong.github.io/2Xplat
    </div>
    <details class="paper-abstract">
      Pose-free feed-forward 3D Gaussian Splatting (3DGS) has opened a new frontier for rapid 3D modeling, enabling high-quality Gaussian representations to be generated from uncalibrated multi-view images in a single forward pass. The dominant approach in this space adopts unified monolithic architectures, often built on geometry-centric 3D foundation models, to jointly estimate camera poses and synthesize 3DGS representations within a single network. While architecturally streamlined, such "all-in-one" designs may be suboptimal for high-fidelity 3DGS generation, as they entangle geometric reasoning and appearance modeling within a shared representation. In this work, we introduce 2Xplat, a pose-free feed-forward 3DGS framework based on a two-expert design that explicitly separates geometry estimation from Gaussian generation. A dedicated geometry expert first predicts camera poses, which are then explicitly passed to a powerful appearance expert that synthesizes 3D Gaussians. Despite its conceptual simplicity, being largely underexplored in prior works, the proposed approach proves highly effective. In fewer than 5K training iterations, the proposed two-experts pipeline substantially outperforms prior pose-free feed-forward 3DGS approaches and achieves performance on par with state-of-the-art posed methods. These results challenge the prevailing unified paradigm and suggest the potential advantages of modular design principles for complex 3D geometric estimation and appearance synthesis tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08809v2">Where, What, Why: Toward Explainable 3D-GS Watermarking</a></div>
    <div class="paper-meta">
      📅 2026-03-24
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      As 3D Gaussian Splatting becomes the de facto representation for interactive 3D assets, robust yet imperceptible watermarking is critical. We present a representation-native framework that separates where to write from how to preserve quality. A Trio-Experts module operates directly on Gaussian primitives to derive priors for carrier selection, while a Safety and Budget Aware Gate (SBAG) allocates Gaussians to watermark carriers, optimized for bit resilience under perturbation and bitrate budgets, and to visual compensators that are insulated from watermark loss. To maintain fidelity, we introduce a channel-wise group mask that controls gradient propagation for carriers and compensators, thereby limiting Gaussian parameter updates, repairing local artifacts, and preserving high-frequency details without increasing runtime. Our design yields view-consistent watermark persistence and strong robustness against common image distortions such as compression and noise, while achieving a favorable robustness-quality trade-off compared with prior methods. In addition, decoupled finetuning provides per-Gaussian attributions that reveal where the message is carried and why those carriers are selected, enabling auditable explainability. Compared with state-of-the-art methods, our approach achieves a PSNR improvement of +0.83 dB and a bit-accuracy gain of +1.24%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.03643v4">Uni3R: Unified 3D Reconstruction and Semantic Understanding via Generalizable Gaussian Splatting from Unposed Multi-View Images</a></div>
    <div class="paper-meta">
      📅 2026-03-24
      | 💬 The code is available at https://github.com/HorizonRobotics/Uni3R
    </div>
    <details class="paper-abstract">
      Reconstructing and semantically interpreting 3D scenes from sparse 2D views remains a fundamental challenge in computer vision. Conventional methods often decouple semantic understanding from reconstruction or necessitate costly per-scene optimization, thereby restricting their scalability and generalizability. In this paper, we introduce Uni3R, a novel feed-forward framework that jointly reconstructs a unified 3D scene representation enriched with open-vocabulary semantics, directly from unposed multi-view images. Our approach leverages a Cross-View Transformer to robustly integrate information across arbitrary multi-view inputs, which then regresses a set of 3D Gaussian primitives endowed with semantic feature fields. This unified representation facilitates high-fidelity novel view synthesis, open-vocabulary 3D semantic segmentation, and depth prediction, all within a single, feed-forward pass. Extensive experiments demonstrate that Uni3R establishes a new state-of-the-art across multiple benchmarks, including 25.07 PSNR on RE10K and 55.84 mIoU on ScanNet. Our work signifies a novel paradigm towards generalizable, unified 3D scene reconstruction and understanding. The code is available at https://github.com/HorizonRobotics/Uni3R.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.22796v1">PhotoAgent: A Robotic Photographer with Spatial and Aesthetic Understanding</a></div>
    <div class="paper-meta">
      📅 2026-03-24
      | 💬 Accepted to the IEEE International Conference on Robotics and Automation (ICRA) 2026
    </div>
    <details class="paper-abstract">
      Embodied agents for creative tasks like photography must bridge the semantic gap between high-level language commands and geometric control. We introduce PhotoAgent, an agent that achieves this by integrating Large Multimodal Models (LMMs) reasoning with a novel control paradigm. PhotoAgent first translates subjective aesthetic goals into solvable geometric constraints via LMM-driven, chain-of-thought (CoT) reasoning, allowing an analytical solver to compute a high-quality initial viewpoint. This initial pose is then iteratively refined through visual reflection within a photorealistic internal world model built with 3D Gaussian Splatting (3DGS). This ``mental simulation'' replaces costly and slow physical trial-and-error, enabling rapid convergence to aesthetically superior results. Evaluations confirm that PhotoAgent excels in spatial reasoning and achieves superior final image quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.22786v1">Predictive Photometric Uncertainty in Gaussian Splatting for Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-03-24
      | 💬 Project Page: https://chumsy0725.github.io/GS-U/
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting have enabled impressive photorealistic novel view synthesis. However, to transition from a pure rendering engine to a reliable spatial map for autonomous agents and safety-critical applications, knowing where the representation is uncertain is as important as the rendering fidelity itself. We bridge this critical gap by introducing a lightweight, plug-and-play framework for pixel-wise, view-dependent predictive uncertainty estimation. Our post-hoc method formulates uncertainty as a Bayesian-regularized linear least-squares optimization over reconstruction residuals. This architecture-agnostic approach extracts a per-primitive uncertainty channel without modifying the underlying scene representation or degrading baseline visual fidelity. Crucially, we demonstrate that providing this actionable reliability signal successfully translates 3D Gaussian splatting into a trustworthy spatial map, further improving state-of-the-art performance across three critical downstream perception tasks: active view selection, pose-agnostic scene change detection, and pose-agnostic anomaly detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.23297v1">Drop-In Perceptual Optimization for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-23
      | 💬 Project page: https://apple.github.io/ml-perceptual-3dgs
    </div>
    <details class="paper-abstract">
      Despite their output being ultimately consumed by human viewers, 3D Gaussian Splatting (3DGS) methods often rely on ad-hoc combinations of pixel-level losses, resulting in blurry renderings. To address this, we systematically explore perceptual optimization strategies for 3DGS by searching over a diverse set of distortion losses. We conduct the first-of-its-kind large-scale human subjective study on 3DGS, involving 39,320 pairwise ratings across several datasets and 3DGS frameworks. A regularized version of Wasserstein Distortion, which we call WD-R, emerges as the clear winner, excelling at recovering fine textures without incurring a higher splat count. WD-R is preferred by raters more than $2.3\times$ over the original 3DGS loss, and $1.5\times$ over current best method Perceptual-GS. WD-R also consistently achieves state-of-the-art LPIPS, DISTS, and FID scores across various datasets, and generalizes across recent frameworks, such as Mip-Splatting and Scaffold-GS, where replacing the original loss with WD-R consistently enhances perceptual quality within a similar resource budget (number of splats for Mip-Splatting, model size for Scaffold-GS), and leads to reconstructions being preferred by human raters $1.8\times$ and $3.6\times$, respectively. We also find that this carries over to the task of 3DGS scene compression, with $\approx 50\%$ bitrate savings for comparable perceptual metric performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.22102v1">FreeArtGS: Articulated Gaussian Splatting Under Free-moving Scenario</a></div>
    <div class="paper-meta">
      📅 2026-03-23
      | 💬 Accepted to CVPR 2026
    </div>
    <details class="paper-abstract">
      The increasing demand for augmented reality and robotics is driving the need for articulated object reconstruction with high scalability. However, existing settings for reconstructing from discrete articulation states or casual monocular videos require non-trivial axis alignment or suffer from insufficient coverage, limiting their applicability. In this paper, we introduce FreeArtGS, a novel method for reconstructing articulated objects under free-moving scenario, a new setting with a simple setup and high scalability. FreeArtGS combines free-moving part segmentation with joint estimation and end-to-end optimization, taking only a monocular RGB-D video as input. By optimizing with the priors from off-the-shelf point-tracking and feature models, the free-moving part segmentation module identifies rigid parts from relative motion under unconstrained capture. The joint estimation module calibrates the unified object-to-camera poses and recovers joint type and axis robustly from part segmentation. Finally, 3DGS-based end-to-end optimization is implemented to jointly reconstruct visual textures, geometry, and joint angles of the articulated object. We conduct experiments on two benchmarks and real-world free-moving articulated objects. Experimental results demonstrate that FreeArtGS consistently excels in reconstructing free-moving articulated objects and remains highly competitive in previous reconstruction settings, proving itself a practical and effective solution for realistic asset generation. The project page is available at: https://freeartgs.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.01583v3">3DSceneEditor: Controllable 3D Scene Editing with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-23
      | 💬 Accepted by WACV 2026, Project Page: https://ziyangyan.github.io/3DSceneEditor
    </div>
    <details class="paper-abstract">
      The creation of 3D scenes has traditionally been both labor-intensive and costly, requiring designers to meticulously configure 3D assets and environments. Recent advancements in generative AI, including text-to-3D and image-to-3D methods, have dramatically reduced the complexity and cost of this process. However, current techniques for editing complex 3D scenes continue to rely on generally interactive multi-step, 2D-to-3D projection methods and diffusion-based techniques, which often lack precision in control and hamper interactive-rate performance. In this work, we propose ***3DSceneEditor***, a fully 3D-based paradigm for interactive-rate, precise editing of intricate 3D scenes using Gaussian Splatting. Unlike conventional methods, 3DSceneEditor operates through a streamlined 3D pipeline, enabling direct Gaussian-based manipulation for efficient, high-quality edits based on input prompts. The proposed framework (i) integrates a pre-trained instance segmentation model for semantic labeling; (ii) employs a zero-shot grounding approach with CLIP to align target objects with user prompts; and (iii) applies scene modifications, such as object addition, repositioning, recoloring, replacing, and removal--directly on Gaussians. Extensive experimental results show that 3DSceneEditor surpasses existing state-of-the-art techniques in terms of both editing precision and efficiency, establishing a new benchmark for efficient and interactive 3D scene customization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.21933v1">Camera-Agnostic Pruning of 3D Gaussian Splats via Descriptor-Based Beta Evidence</a></div>
    <div class="paper-meta">
      📅 2026-03-23
      | 💬 14 pages, 3 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The pruning of 3D Gaussian splats is essential for reducing their complexity to enable efficient storage, transmission, and downstream processing. However, most of the existing pruning strategies depend on camera parameters, rendered images, or view-dependent measures. This dependency becomes a hindrance in emerging camera-agnostic exchange settings, where splats are shared directly as point-based representations (e.g., .ply). In this paper, we propose a camera-agnostic, one-shot, post-training pruning method for 3D Gaussian splats that relies solely on attribute-derived neighbourhood descriptors. As our primary contribution, we introduce a hybrid descriptor framework that captures structural and appearance consistency directly from the splat representation. Building on these descriptors, we formulate pruning as a statistical evidence estimation problem and introduce a Beta evidence model that quantifies per-splat reliability through a probabilistic confidence score. Experiments conducted on standardized test sequences defined by the ISO/IEC MPEG Common Test Conditions (CTC) demonstrate that our approach achieves substantial pruning while preserving reconstruction quality, establishing a practical and generalizable alternative to existing camera-dependent pruning strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19235v2">IDSplat: Instance-Decomposed 3D Gaussian Splatting for Driving Scenes</a></div>
    <div class="paper-meta">
      📅 2026-03-23
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic driving scenes is essential for developing autonomous systems through sensor-realistic simulation. Although recent methods achieve high-fidelity reconstructions, they either rely on costly human annotations for object trajectories or use time-varying representations without explicit object-level decomposition, leading to intertwined static and dynamic elements that hinder scene separation. We present IDSplat, a self-supervised 3D Gaussian Splatting framework that reconstructs dynamic scenes with explicit instance decomposition and learnable motion trajectories, without requiring human annotations. Our key insight is to model dynamic objects as coherent instances undergoing rigid transformations, rather than unstructured time-varying primitives. For instance decomposition, we employ zero-shot, language-grounded video tracking anchored to 3D using lidar, and estimate consistent poses via feature correspondences. We introduce a coordinated-turn smoothing scheme to obtain temporally and physically consistent motion trajectories, mitigating pose misalignments and tracking failures, followed by joint optimization of object poses and Gaussian parameters. Experiments on the Waymo Open Dataset demonstrate that our method achieves competitive reconstruction quality while maintaining instance-level decomposition and generalizes across diverse sequences and view densities without retraining, making it practical for large-scale autonomous driving applications. Code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.21695v1">RefracGS: Novel View Synthesis Through Refractive Water Surfaces with 3D Gaussian Ray Tracing</a></div>
    <div class="paper-meta">
      📅 2026-03-23
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) through non-planar refractive surfaces presents fundamental challenges due to severe, spatially varying optical distortions. While recent representations like NeRF and 3D Gaussian Splatting (3DGS) excel at NVS, their assumption of straight-line ray propagation fails under these conditions, leading to significant artifacts. To overcome this limitation, we introduce RefracGS, a framework that jointly reconstructs the refractive water surface and the scene beneath the interface. Our key insight is to explicitly decouple the refractive boundary from the target objects: the refractive surface is modeled via a neural height field, capturing wave geometry, while the underlying scene is represented as a 3D Gaussian field. We formulate a refraction-aware Gaussian ray tracing approach that accurately computes non-linear ray trajectories using Snell's law and efficiently renders the underlying Gaussian field while backpropagating the loss gradients to the parameterized refractive surface. Through end-to-end joint optimization of both representations, our method ensures high-fidelity NVS and view-consistent surface recovery. Experiments on both synthetic and real-world scenes with complex waves demonstrate that RefracGS outperforms prior refractive methods in visual quality, while achieving 15x faster training and real-time rendering at 200 FPS. The project page for RefracGS is available at https://yimgshao.github.io/refracgs/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17779v2">CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image</a></div>
    <div class="paper-meta">
      📅 2026-03-23
      | 💬 Accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      Single-view 3D human reconstruction has garnered significant attention in recent years. Despite numerous advancements, prior research has concentrated on reconstructing 3D models from clear, close-up images of individual subjects, often yielding subpar results in the more prevalent multi-person scenarios. Reconstructing 3D human crowd models is a highly intricate task, laden with challenges such as: 1) extensive occlusions, 2) low clarity, and 3) numerous and various appearances. To address this task, we propose CrowdGaussian, a unified framework that directly reconstructs multi-person 3D Gaussian Splatting (3DGS) representations from single-image inputs. To handle occlusions, we devise a self-supervised adaptation pipeline that enables the pretrained large human model to reconstruct complete 3D humans with plausible geometry and appearance from heavily occluded inputs. Furthermore, we introduce Self-Calibrated Learning (SCL). This training strategy enables single-step diffusion models to adaptively refine coarse renderings to optimal quality by blending identity-preserving samples with clean/corrupted image pairs. The outputs can be distilled back to enhance the quality of multi-person 3DGS representations. Extensive experiments demonstrate that CrowdGaussian generates photorealistic, geometrically coherent reconstructions of multi-person scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10446v3">SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning</a></div>
    <div class="paper-meta">
      📅 2026-03-23
    </div>
    <details class="paper-abstract">
      Generating natural and linguistically accurate sign language avatars remains a formidable challenge. Current Sign Language Production (SLP) frameworks face a stark trade-off: direct text-to-pose models suffer from regression-to-the-mean effects, while dictionary-retrieval methods produce robotic, disjointed transitions. To resolve this, we propose a novel training paradigm that leverages sparse keyframes to capture the true underlying kinematic distribution of human signing. By predicting dense motion from these discrete anchors, our approach mitigates regression-to-the-mean while ensuring fluid articulation. To realize this paradigm at scale, we first introduce FAST, an ultra-efficient sign segmentation model that automatically mines precise temporal boundaries. We then present SignSparK, a large-scale Conditional Flow Matching (CFM) framework that utilizes these extracted anchors to synthesize 3D signing sequences in SMPL-X and MANO spaces. This keyframe-driven formulation also uniquely unlocks Keyframe-to-Pose (KF2P) generation, making precise spatiotemporal editing of signing sequences possible. Furthermore, our adopted reconstruction-based CFM objective also enables high-fidelity synthesis in fewer than ten sampling steps; this allows SignSparK to scale across four distinct sign languages, establishing the largest multilingual SLP framework to date. Finally, by integrating 3D Gaussian Splatting for photorealistic rendering, we demonstrate through extensive evaluation that SignSparK establishes a new state-of-the-art across diverse SLP tasks and multilingual benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.06989v2">MipSLAM: Alias-Free Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2026-03-23
      | 💬 Accepted to ICRA 2026
    </div>
    <details class="paper-abstract">
      This paper introduces MipSLAM, a frequency-aware 3D Gaussian Splatting (3DGS) SLAM framework capable of high-fidelity anti-aliased novel view synthesis and robust pose estimation under varying camera configurations. Existing 3DGS-based SLAM systems often suffer from aliasing artifacts and trajectory drift due to inadequate filtering and purely spatial optimization. To overcome these limitations, we propose an Elliptical Adaptive Anti-aliasing (EAA) algorithm that approximates Gaussian contributions via geometry-aware numerical integration, avoiding costly analytic computation. Furthermore, we present a Spectral-Aware Pose Graph Optimization (SA-PGO) module that reformulates trajectory estimation in the frequency domain, effectively suppressing high-frequency noise and drift through graph Laplacian analysis. Extensive evaluations on Replica and TUM datasets demonstrate that MipSLAM achieves state-of-the-art rendering quality and localization accuracy across multiple resolutions. Code is available at https://github.com/yzli1998/MipSLAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.21055v1">SGAD-SLAM: Splatting Gaussians at Adjusted Depth for Better Radiance Fields in RGBD SLAM</a></div>
    <div class="paper-meta">
      📅 2026-03-22
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has made remarkable progress in RGBD SLAM. Current methods usually use 3D Gaussians or view-tied 3D Gaussians to represent radiance fields in tracking and mapping. However, these Gaussians are either too flexible or too limited in movements, resulting in slow convergence or limited rendering quality. To resolve this issue, we adopt pixel-aligned Gaussians but allow each Gaussian to adjust its position along its ray to maximize the rendering quality, even if Gaussians are simplified to improve system scalability. To speed up the tracking, we model the depth distribution around each pixel as a Gaussian distribution, and then use these distributions to align each frame to the 3D scene quickly. We report our evaluations on widely used benchmarks, justify our designs, and show advantages over the latest methods in view rendering, camera tracking, runtime, and storage complexity. Please see our project page for code and videos at https://machineperceptionlab.github.io/SGAD-SLAM-Project .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20857v1">Fast and Robust Deformable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-21
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has demonstrated remarkable real-time rendering capabilities and superior visual quality in novel view synthesis for static scenes. Building upon these advantages, researchers have progressively extended 3D Gaussians to dynamic scene reconstruction. Deformation field-based methods have emerged as a promising approach among various techniques. These methods maintain 3D Gaussian attributes in a canonical field and employ the deformation field to transform this field across temporal sequences. Nevertheless, these approaches frequently encounter challenges such as suboptimal rendering speeds, significant dependence on initial point clouds, and vulnerability to local optima in dim scenes. To overcome these limitations, we present FRoG, an efficient and robust framework for high-quality dynamic scene reconstruction. FRoG integrates per-Gaussian embedding with a coarse-to-fine temporal embedding strategy, accelerating rendering through the early fusion of temporal embeddings. Moreover, to enhance robustness against sparse initializations, we introduce a novel depth- and error-guided sampling strategy. This strategy populates the canonical field with new 3D Gaussians at low-deviation initial positions, significantly reducing the optimization burden on the deformation field and improving detail reconstruction in both static and dynamic regions. Furthermore, by modulating opacity variations, we mitigate the local optima problem in dim scenes, improving color fidelity. Comprehensive experimental results validate that our method achieves accelerated rendering speeds while maintaining state-of-the-art visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20714v1">The Role and Relationship of Initialization and Densification in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-21
      | 💬 Sources will be made publicly available
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become the method of choice for photo-realistic 3D reconstruction of scenes, due to being able to efficiently and accurately recover the scene appearance and geometry from images. 3DGS represents the scene through a set of 3D Gaussians, parameterized by their position, spatial extent, and view-dependent color. Starting from an initial point cloud, 3DGS refines the Gaussians' parameters as to reconstruct a set of training images as accurately as possible. Typically, a sparse Structure-from-Motion point cloud is used as initialization. In order to obtain dense Gaussian clouds, 3DGS methods thus rely on a densification stage. In this paper, we systematically study the relation between densification and initialization. Proposing a new benchmark, we study combinations of different types of initializations (dense laser scans, dense (multi-view) stereo point clouds, dense monocular depth estimates, sparse SfM point clouds) and different densification schemes. We show that current densification approaches are not able to take full advantage of dense initialization as they are often unable to (significantly) improve over sparse SfM-based initialization. We will make our benchmark publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20611v1">GaussianPile: A Unified Sparse Gaussian Splatting Framework for Slice-based Volumetric Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-21
      | 💬 Accepted by IEEE/CVF Conference on Computer Vision and Pattern Recognition 2026 (CVPR 2026)
    </div>
    <details class="paper-abstract">
      Slice-based volumetric imaging is widely applied and it demands representations that compress aggressively while preserving internal structure for analysis. We introduce GaussianPile, unifying 3D Gaussian splatting with an imaging system-aware focus model to address this challenge. Our proposed method introduces three key innovations: (i) a slice-aware piling strategy that positions anisotropic 3D Gaussians to model through-slice contributions, (ii) a differentiable projection operator that encodes the finite-thickness point spread function of the imaging acquisition system, and (iii) a compact encoding and joint optimization pipeline that simultaneously reconstructs and compresses the Gaussian sets. Our CUDA-based design retains the compression and real-time rendering efficiency of Gaussian primitives while preserving high-frequency internal volumetric detail. Experiments on microscopy and ultrasound datasets demonstrate that our method reduces storage and reconstruction cost, sustains diagnostic fidelity, and enables fast 2D visualization, along with 3D voxelization. In practice, it delivers high-quality results in as few as 3 minutes, up to 11x faster than NeRF-based approaches, and achieves consistent 16x compression over voxel grids, offering a practical path to deployable compression and exploration of slice-based volumetric datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20560v1">Nevis Digital Twin: Photogrammetry and Immersive Visualization of Historical Sites</a></div>
    <div class="paper-meta">
      📅 2026-03-20
      | 💬 ARCHERIX Workshop - IEEE VR 2026
    </div>
    <details class="paper-abstract">
      In this work, we present a multimodal data acquisition workflow for the digital preservation and virtual reconstruction of at-risk historical sites in the island of Nevis. Facing threats from coastal erosion, rising sea levels, and aggressive vegetation, the archaeological heritage of Nevis requires documentation strategies that bridge the gap between high-cost professional surveying and consumer accessibility. Experimental test compared acquisition variables, specifically camera height (1m vs. 3m) and operator trajectory against high-resolution control data. Moreover, we explore the virtual reconstruction between mesh reconstruction and 3D gaussian splatting to serve as different modalities for documentation. The resulting data is fused into immersive virtual reality (VR) environments, offering a scalable, non-proprietary model for democratizing digital heritage in the Caribbean.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20443v1">TRGS-SLAM: IMU-Aided Gaussian Splatting SLAM for Blurry, Rolling Shutter, and Noisy Thermal Images</a></div>
    <div class="paper-meta">
      📅 2026-03-20
      | 💬 Project page: https://umautobots.github.io/trgs_slam
    </div>
    <details class="paper-abstract">
      Thermal cameras offer several advantages for simultaneous localization and mapping (SLAM) with mobile robots: they provide a passive, low-power solution to operating in darkness, are invariant to rapidly changing or high dynamic range illumination, and can see through fog, dust, and smoke. However, uncooled microbolometer thermal cameras, the only practical option in most robotics applications, suffer from significant motion blur, rolling shutter distortions, and fixed pattern noise. In this paper, we present TRGS-SLAM, a 3D Gaussian Splatting (3DGS) based thermal inertial SLAM system uniquely capable of handling these degradations. To overcome the challenges of thermal data, we introduce a model-aware 3DGS rendering method and several general innovations to 3DGS SLAM, including B-spline trajectory optimization with a two-stage IMU loss, view-diversity-based opacity resetting, and pose drift correction schemes. Our system demonstrates accurate tracking on real-world, fast motion, and high-noise thermal data that causes all other tested SLAM methods to fail. Moreover, through offline refinement of our SLAM results, we demonstrate thermal image restoration competitive with prior work that required ground truth poses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04218v3">Pseudo-Simulation for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-03-20
      | 💬 CoRL 2025, updated with leaderboard snapshot from March 2026
    </div>
    <details class="paper-abstract">
      Existing evaluation paradigms for Autonomous Vehicles (AVs) face critical limitations. Real-world evaluation is often challenging due to safety concerns and a lack of reproducibility, whereas closed-loop simulation can face insufficient realism or high computational costs. Open-loop evaluation, while being efficient and data-driven, relies on metrics that generally overlook compounding errors. In this paper, we propose pseudo-simulation, a novel paradigm that addresses these limitations. Pseudo-simulation operates on real datasets, similar to open-loop evaluation, but augments them with synthetic observations generated prior to evaluation using 3D Gaussian Splatting. Our key idea is to approximate potential future states the AV might encounter by generating a diverse set of observations that vary in position, heading, and speed. Our method then assigns a higher importance to synthetic observations that best match the AV's likely behavior using a novel proximity-based weighting scheme. This enables evaluating error recovery and the mitigation of causal confusion, as in closed-loop benchmarks, without requiring sequential interactive simulation. We show that pseudo-simulation is better correlated with closed-loop simulations ($R^2=0.8$) than the best existing open-loop approach ($R^2=0.7$). We also establish a public leaderboard for the community to benchmark new methodologies with pseudo-simulation. Our code is available at https://github.com/autonomousvision/navsim.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22228v2">3D-Consistent Multi-View Editing by Correspondence Guidance</a></div>
    <div class="paper-meta">
      📅 2026-03-20
      | 💬 Added experiments with FLUX.1 editing method
    </div>
    <details class="paper-abstract">
      Recent advancements in diffusion and flow models have greatly improved text-based image editing, yet methods that edit images independently often produce geometrically and photometrically inconsistent results across different views of the same scene. Such inconsistencies are particularly problematic for editing of 3D representations such as NeRFs or Gaussian splat models. We propose a training-free guidance framework that enforces multi-view consistency during the image editing process. The key idea is that corresponding points should look similar after editing. To achieve this, we introduce a consistency loss that guides the denoising process toward coherent edits. The framework is flexible and can be combined with widely varying image editing methods, supporting both dense and sparse multi-view editing setups. Experimental results show that our approach significantly improves 3D consistency compared to existing multi-view editing methods. We also show that this increased consistency enables high-quality Gaussian splat editing with sharp details and strong fidelity to user-specified text prompts. Please refer to our project page for video results: https://3d-consistent-editing.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19234v2">Matryoshka Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-20
      | 💬 project page: https://zhilinguo.github.io/MGS
    </div>
    <details class="paper-abstract">
      The ability to render scenes at adjustable fidelity from a single model, known as level of detail (LoD), is crucial for practical deployment of 3D Gaussian Splatting (3DGS). Existing discrete LoD methods expose only a limited set of operating points, while concurrent continuous LoD approaches enable smoother scaling but often suffer noticeable quality degradation at full capacity, making LoD a costly design decision. We introduce Matryoshka Gaussian Splatting (MGS), a training framework that enables continuous LoD for standard 3DGS pipelines without sacrificing full-capacity rendering quality. MGS learns a single ordered set of Gaussians such that rendering any prefix, the first k splats, produces a coherent reconstruction whose fidelity improves smoothly with increasing budget. Our key idea is stochastic budget training: each iteration samples a random splat budget and optimises both the corresponding prefix and the full set. This strategy requires only two forward passes and introduces no architectural modifications. Experiments across four benchmarks and six baselines show that MGS matches the full-capacity performance of its backbone while enabling a continuous speed-quality trade-off from a single model. Extensive ablations on ordering strategies, training objectives, and model capacity further validate the designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19822v1">HUGE-Bench: A Benchmark for High-Level UAV Vision-Language-Action Tasks</a></div>
    <div class="paper-meta">
      📅 2026-03-20
    </div>
    <details class="paper-abstract">
      Existing UAV vision-language navigation (VLN) benchmarks have enabled language-guided flight, but they largely focus on long, step-wise route descriptions with goal-centric evaluation, making them less diagnostic for real operations where brief, high-level commands must be grounded into safe multi-stage behaviors. We present HUGE-Bench, a benchmark for High-Level UAV Vision-Language-Action (HL-VLA) tasks that tests whether an agent can interpret concise language and execute complex, process-oriented trajectories with safety awareness. HUGE-Bench comprises 4 real-world digital twin scenes, 8 high-level tasks, and 2.56M meters of trajectories, and is built on an aligned 3D Gaussian Splatting (3DGS)-Mesh representation that combines photorealistic rendering with collision-capable geometry for scalable generation and collision-aware evaluation. We introduce process-oriented and collision-aware metrics to assess process fidelity, terminal accuracy, and safety. Experiments on representative state-of-the-art VLA models reveal significant gaps in high-level semantic completion and safe execution, highlighting HUGE-Bench as a diagnostic testbed for high-level UAV autonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2404.10318v3">SRGS: Super-Resolution 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-20
      | 💬 The first to focus on the HRNVS of 3DGS
    </div>
    <details class="paper-abstract">
      Low-resolution (LR) multi-view capture limits the fidelity of 3D Gaussian Splatting (3DGS). 3DGS super-resolution (SR) is therefore important, yet challenging because it must recover missing high-frequency details while enforcing cross-view geometric consistency. We revisit SRGS, a simple baseline that couples plug-in 2D SR priors with geometry-aware cross-view regularization, and observe that most subsequent advances follow the same paradigm, either strengthening prior injection, refining cross-view constraints, or modulating the objective. However, this shared structure is rarely formalized as a unified objective with explicit modules, limiting principled attribution of improvements and reusable design guidance. In this paper, we formalize SRGS as a unified modular framework that factorizes 3DGS SR into two components, prior injection and cross-view regularization, within a joint objective. This abstraction subsumes a broad family of recent methods as instantiations of the same recipe, enabling analysis beyond single-method innovation. Across five public benchmarks, we consolidate nine representative follow-up methods and trace reported improvements to specific modules and settings. Ablations disentangle the roles of priors and consistency, and stress tests under sparse-view input and challenging capture conditions characterize robustness. Overall, our study consolidates 3DGS SR into a coherent foundation and offers practical guidance for robust, comparable 3DGS SR methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19552v1">StreetForward: Perceiving Dynamic Street with Feedforward Causal Attention</a></div>
    <div class="paper-meta">
      📅 2026-03-20
    </div>
    <details class="paper-abstract">
      Feedforward reconstruction is crucial for autonomous driving applications, where rapid scene reconstruction enables efficient utilization of large-scale driving datasets in closed-loop simulation and other downstream tasks, eliminating the need for time-consuming per-scene optimization. We present StreetForward, a pose-free and tracker-free feedforward framework for dynamic street reconstruction. Building upon the alternating attention mechanism from Visual Geometry Grounded Transformer (VGGT), we propose a simple yet effective temporal mask attention module that captures dynamic motion information from image sequences and produces motion-aware latent representations. Static content and dynamic instances are represented uniformly with 3D Gaussian Splatting, and are optimized jointly by cross-frame rendering with spatio-temporal consistency, allowing the model to infer per-pixel velocities and produce high-fidelity novel views at new poses and times. We train and evaluate our model on the Waymo Open Dataset, demonstrating superior performance on novel view synthesis and depth estimation compared to existing methods. Furthermore, zero-shot inference on CARLA and other datasets validates the generalization capability of our approach. More visualizations are available on our project page: https://streetforward.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19234v1">Matryoshka Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 project page: https://zhilinguo.github.io/MGS
    </div>
    <details class="paper-abstract">
      The ability to render scenes at adjustable fidelity from a single model, known as level of detail (LoD), is crucial for practical deployment of 3D Gaussian Splatting (3DGS). Existing discrete LoD methods expose only a limited set of operating points, while concurrent continuous LoD approaches enable smoother scaling but often suffer noticeable quality degradation at full capacity, making LoD a costly design decision. We introduce Matryoshka Gaussian Splatting (MGS), a training framework that enables continuous LoD for standard 3DGS pipelines without sacrificing full-capacity rendering quality. MGS learns a single ordered set of Gaussians such that rendering any prefix, the first k splats, produces a coherent reconstruction whose fidelity improves smoothly with increasing budget. Our key idea is stochastic budget training: each iteration samples a random splat budget and optimises both the corresponding prefix and the full set. This strategy requires only two forward passes and introduces no architectural modifications. Experiments across four benchmarks and six baselines show that MGS matches the full-capacity performance of its backbone while enabling a continuous speed-quality trade-off from a single model. Extensive ablations on ordering strategies, training objectives, and model capacity further validate the designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19193v1">Reconstruction Matters: Learning Geometry-Aligned BEV Representation through 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 Project page at https://vulab-ai.github.io/Splat2BEV/
    </div>
    <details class="paper-abstract">
      Bird's-Eye-View (BEV) perception serves as a cornerstone for autonomous driving, offering a unified spatial representation that fuses surrounding-view images to enable reasoning for various downstream tasks, such as semantic segmentation, 3D object detection, and motion prediction. However, most existing BEV perception frameworks adopt an end-to-end training paradigm, where image features are directly transformed into the BEV space and optimized solely through downstream task supervision. This formulation treats the entire perception process as a black box, often lacking explicit 3D geometric understanding and interpretability, leading to suboptimal performance. In this paper, we claim that an explicit 3D representation matters for accurate BEV perception, and we propose Splat2BEV, a Gaussian Splatting-assisted framework for BEV tasks. Splat2BEV aims to learn BEV feature representations that are both semantically rich and geometrically precise. We first pre-train a Gaussian generator that explicitly reconstructs 3D scenes from multi-view inputs, enabling the generation of geometry-aligned feature representations. These representations are then projected into the BEV space to serve as inputs for downstream tasks. Extensive experiments on nuScenes and argoverse dataset demonstrate that Splat2BEV achieves state-of-the-art performance and validate the effectiveness of incorporating explicit 3D reconstruction into BEV perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19137v1">GSMem: 3D Gaussian Splatting as Persistent Spatial Memory for Zero-Shot Embodied Exploration and Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 Project page at https://vulab-ai.github.io/GSMem/
    </div>
    <details class="paper-abstract">
      Effective embodied exploration requires agents to accumulate and retain spatial knowledge over time. However, existing scene representations, such as discrete scene graphs or static view-based snapshots, lack \textit{post-hoc re-observability}. If an initial observation misses a target, the resulting memory omission is often irrecoverable. To bridge this gap, we propose \textbf{GSMem}, a zero-shot embodied exploration and reasoning framework built upon 3D Gaussian Splatting (3DGS). By explicitly parameterizing continuous geometry and dense appearance, 3DGS serves as a persistent spatial memory that endows the agent with \textit{Spatial Recollection}: the ability to render photorealistic novel views from optimal, previously unoccupied viewpoints. To operationalize this, GSMem employs a retrieval mechanism that simultaneously leverages parallel object-level scene graphs and semantic-level language fields. This complementary design robustly localizes target regions, enabling the agent to ``hallucinate'' optimal views for high-fidelity Vision-Language Model (VLM) reasoning. Furthermore, we introduce a hybrid exploration strategy that combines VLM-driven semantic scoring with a 3DGS-based coverage objective, balancing task-aware exploration with geometric coverage. Extensive experiments on embodied question answering and lifelong navigation demonstrate the robustness and effectiveness of our framework
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18912v1">GHOST: Fast Category-agnostic Hand-Object Interaction Reconstruction from RGB Videos using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Understanding realistic hand-object interactions from monocular RGB videos is essential for AR/VR, robotics, and embodied AI. Existing methods rely on category-specific templates or heavy computation, yet still produce physically inconsistent hand-object alignment in 3D. We introduce GHOST (Gaussian Hand-Object Splatting), a fast, category-agnostic framework for reconstructing dynamic hand-object interactions using 2D Gaussian Splatting. GHOST represents both hands and objects as dense, view-consistent Gaussian discs and introduces three key innovations: (1) a geometric-prior retrieval and consistency loss that completes occluded object regions, (2) a grasp-aware alignment that refines hand translations and object scale to ensure realistic contact, and (3) a hand-aware background loss that prevents penalizing hand-occluded object regions. GHOST achieves complete, physically consistent, and animatable reconstructions from a single RGB video while running an order of magnitude faster than prior category-agnostic methods. Extensive experiments on ARCTIC, HO3D, and in-the-wild datasets demonstrate state-of-the-art accuracy in 3D reconstruction and 2D rendering quality, establishing GHOST as an efficient and robust solution for realistic hand-object interaction modeling. Code is available at https://github.com/ATAboukhadra/GHOST.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09162v3">GTAvatar: Bridging Gaussian Splatting and Texture Mapping for Relightable and Editable Gaussian Avatars</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 Accepted to Eurographics 2026. Project page: https://kelianb.github.io/GTAvatar/
    </div>
    <details class="paper-abstract">
      Recent advancements in Gaussian Splatting have enabled increasingly accurate reconstruction of photorealistic head avatars, opening the door to numerous applications in visual effects, videoconferencing, and virtual reality. This, however, comes with the lack of intuitive editability offered by traditional triangle mesh-based methods. In contrast, we propose a method that combines the accuracy and fidelity of 2D Gaussian Splatting with the intuitiveness of UV texture mapping. By embedding each canonical Gaussian primitive's local frame into a patch in the UV space of a template mesh in a computationally efficient manner, we reconstruct continuous editable material head textures from a single monocular video on a conventional UV domain. Furthermore, we leverage an efficient physically based reflectance model to enable relighting and editing of these intrinsic material maps. Through extensive comparisons with state-of-the-art methods, we demonstrate the accuracy of our reconstructions, the quality of our relighting results, and the ability to provide intuitive controls for modifying an avatar's appearance and geometry via texture mapping without additional optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.11431v2">Remove360: Benchmarking Residuals After Object Removal in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      An object can disappear from a 3D scene, yet still be detectable. Even after visual removal, modern vision models may infer what was originally present. In this work, we introduce a novel benchmark and evaluation framework to quantify semantic residuals, the unintended cues left behind after object removal in 3D Gaussian Splatting. We conduct experiments across a diverse set of indoor and outdoor scenes, showing that current methods often preserve semantic information despite the absence of visual geometry. Notably, even when removal is followed by inpainting, residual cues frequently remain detectable by foundation models. We also present Remove360, a real-world dataset of pre- and post-removal RGB captures with object-level masks. Unlike prior datasets focused on isolated object instances, Remove360 contains complex, cluttered scenes that enable evaluation of object removal in full-scene settings. By leveraging the ground-truth post-removal images, we directly assess whether semantic presence is eliminated and whether downstream models can still infer what was removed. Our results reveal a consistent gap between geometric removal and semantic erasure, exposing critical limitations in existing 3D editing pipelines and highlighting the need for privacy-aware removal methods that eliminate recoverable cues, not only visible geometry. Dataset and evaluation code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18707v1">From ex(p) to poly: Gaussian Splatting with Polynomial Kernels</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Recent advancements in Gaussian Splatting (3DGS) have introduced various modifications to the original kernel, resulting in significant performance improvements. However, many of these kernel changes are incompatible with existing datasets optimized for the original Gaussian kernel, presenting a challenge for widespread adoption. In this work, we address this challenge by proposing an alternative kernel that maintains compatibility with existing datasets while improving computational efficiency. Specifically, we replace the original exponential kernel with a polynomial approximation combined with a ReLU function. This modification allows for more aggressive culling of Gaussians, leading to enhanced performance across different 3DGS implementations. Our results show a notable performance improvement of 4 to 15% with negligible impact on image quality. We also provide a detailed mathematical analysis of the new kernel and discuss its potential benefits for 3DGS implementations on NPU hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15376v4">DriveSplat: Unified Neural Gaussian Reconstruction for Dynamic Driving Scenes</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Reconstructing large-scale dynamic driving scenes remains challenging due to the coexistence of static environments with extreme depth variation and diverse dynamic actors exhibiting complex motions. Existing Gaussian Splatting based methods have primarily focused on limited-scale or object-centric settings, and their applicability to large-scale dynamic driving scenes remains underexplored, particularly in the presence of extreme scale variation and non-rigid motions. In this work, we propose DriveSplat, a unified neural Gaussian framework for reconstructing dynamic driving scenes within a unified Gaussian-based representation. For static backgrounds, we introduce a scene-aware learnable level-of-detail (LOD) modeling strategy that explicitly accounts for near, intermediate, and far depth ranges in driving environments, enabling adaptive multi-scale Gaussian allocation. For dynamic actors, we use an object-centric formulation with neural Gaussian primitives, modeling motion through a global rigid transformation and handling non-rigid dynamics via a two-stage deformation that first adjusts anchors and subsequently updates the Gaussians. To further regularize the optimization, we incorporate dense depth and surface normal priors from pre-trained models as auxiliary supervision. Extensive experiments on the Waymo and KITTI benchmarks demonstrate that DriveSplat achieves state-of-the-art performance in novel-view synthesis while producing temporally stable and geometrically consistent reconstructions of dynamic driving scenes. Project page: https://physwm.github.io/drivesplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02691v3">FSFSplatter: Build Surface and Novel Views with Sparse-Views within 2min</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has become a leading reconstruction technique, known for its high-quality novel view synthesis and detailed reconstruction. However, most existing methods require dense, calibrated views. Reconstructing from free sparse images often leads to poor surface due to limited overlap and overfitting. We introduce FSFSplatter, a new approach for fast surface reconstruction from free sparse images. Our method integrates end-to-end dense Gaussian initialization, camera parameter estimation, and geometry-enhanced scene optimization. Specifically, FSFSplatter employs a large Transformer to encode multi-view images and generates a dense and geometrically consistent Gaussian scene initialization via a self-splitting Gaussian head. It eliminates local floaters through contribution-based pruning and mitigates overfitting to limited views by leveraging depth and multi-view feature supervision with differentiable camera parameters during rapid optimization. FSFSplatter outperforms current state-of-the-art methods on widely used DTU, Replica, and BlendedMVS datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18510v1">OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      Open-vocabulary scene understanding with online panoptic mapping is essential for embodied applications to perceive and interact with environments. However, existing methods are predominantly offline or lack instance-level understanding, limiting their applicability to real-world robotic tasks. In this paper, we propose OnlinePG, a novel and effective system that integrates geometric reconstruction and open-vocabulary perception using 3D Gaussian Splatting in an online setting. Technically, to achieve online panoptic mapping, we employ an efficient local-to-global paradigm with a sliding window. To build local consistency map, we construct a 3D segment clustering graph that jointly leverages geometric and semantic cues, fusing inconsistent segments within sliding window into complete instances. Subsequently, to update the global map, we construct explicit grids with spatial attributes for the local 3D Gaussian map and fuse them into the global map via robust bidirectional bipartite 3D Gaussian instance matching. Finally, we utilize the fused VLM features inside the 3D spatial attribute grids to achieve open-vocabulary scene understanding. Extensive experiments on widely used datasets demonstrate that our method achieves better performance among online approaches, while maintaining real-time efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18402v1">Inst4DGS: Instance-Decomposed 4D Gaussian Splatting with Multi-Video Label Permutation Learning</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      We present Inst4DGS, an instance-decomposed 4D Gaussian Splatting (4DGS) approach with long-horizon per-Gaussian trajectories. While dynamic 4DGS has advanced rapidly, instance-decomposed 4DGS remains underexplored, largely due to the difficulty of associating inconsistent instance labels across independently segmented multi-view videos. We address this challenge by introducing per-video label-permutation latents that learn cross-video instance matches through a differentiable Sinkhorn layer, enabling direct multi-view supervision with consistent identity preservation. This explicit label alignment yields sharp decision boundaries and temporally stable identities without identity drift. To further improve efficiency, we propose instance-decomposed motion scaffolds that provide low-dimensional motion bases per object for long-horizon trajectory optimization. Experiments on Panoptic Studio and Neural3DV show that Inst4DGS jointly supports tracking and instance decomposition while achieving state-of-the-art rendering and segmentation quality. On the Panoptic Studio dataset, Inst4DGS improves PSNR from 26.10 to 28.36, and instance mIoU from 0.6310 to 0.9129, over the strongest baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03992v2">Camera-Aware Cross-View Alignment for Referring 3D Gaussian Splatting Segmentation</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 Accepted to ICME 2026
    </div>
    <details class="paper-abstract">
      Referring 3D Gaussian Splatting Segmentation (R3DGS) aims to ground free-form language queries in 3D Gaussian fields. However, existing methods rely on single-view pseudo supervision, leading to viewpoint drift and inconsistent predictions across views. We propose CaRF (Camera-aware Referring Field), a camera-aware cross-view alignment framework for view-consistent referring in 3D Gaussian splatting. CaRF introduces Camera-conditioned Alignment Modulation (CAM) to inject camera geometry into Gaussian-text interactions, and Gaussian-level Cross-view Logit Alignment (GCLA) to explicitly align referring responses of the same Gaussians across calibrated views during training. By turning cross-view discrepancy into an optimizable objective, CaRF enables geometry-aware and view-consistent reasoning directly in the Gaussian space. Extensive experiments on three benchmarks demonstrate that CaRF achieves state-of-the-art performance, improving mIoU by 16.8%, 4.3%, and 2.0% on Ref-LERF, LERF-OVS, and 3D-OVS, respectively. Our code is available at https://github.com/eR3R3/CaRF.
    </details>
</div>
