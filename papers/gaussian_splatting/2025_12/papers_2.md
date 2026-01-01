# gaussian splatting - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.21520v2">MADrive: Memory-Augmented Driving Scene Modeling</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Recent advances in scene reconstruction have pushed toward highly realistic modeling of autonomous driving (AD) environments using 3D Gaussian splatting. However, the resulting reconstructions remain closely tied to the original observations and struggle to support photorealistic synthesis of significantly altered or novel driving scenarios. This work introduces MADrive, a memory-augmented reconstruction framework designed to extend the capabilities of existing scene reconstruction methods by replacing observed vehicles with visually similar 3D assets retrieved from a large-scale external memory bank. Specifically, we release MAD-Cars, a curated dataset of ${\sim}70$K 360° car videos captured in the wild and present a retrieval module that finds the most similar car instances in the memory bank, reconstructs the corresponding 3D assets from video, and integrates them into the target scene through orientation alignment and relighting. The resulting replacements provide complete multi-view representations of vehicles in the scene, enabling photorealistic synthesis of substantially altered configurations, as demonstrated in our experiments. Project page: https://yandex-research.github.io/madrive/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10095v1">TraceFlow: Dynamic 3D Reconstruction of Specular Scenes Driven by Ray Tracing</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      We present TraceFlow, a novel framework for high-fidelity rendering of dynamic specular scenes by addressing two key challenges: precise reflection direction estimation and physically accurate reflection modeling. To achieve this, we propose a Residual Material-Augmented 2D Gaussian Splatting representation that models dynamic geometry and material properties, allowing accurate reflection ray computation. Furthermore, we introduce a Dynamic Environment Gaussian and a hybrid rendering pipeline that decomposes rendering into diffuse and specular components, enabling physically grounded specular synthesis via rasterization and ray tracing. Finally, we devise a coarse-to-fine training strategy to improve optimization stability and promote physically meaningful decomposition. Extensive experiments on dynamic scene benchmarks demonstrate that TraceFlow outperforms prior methods both quantitatively and qualitatively, producing sharper and more realistic specular reflections in complex dynamic environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.17329v2">SmokeSeer: 3D Gaussian Splatting for Smoke Removal and Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 Project website: https://imaging.cs.cmu.edu/smokeseer
    </div>
    <details class="paper-abstract">
      Smoke in real-world scenes can severely degrade image quality and hamper visibility. Recent image restoration methods either rely on data-driven priors that are susceptible to hallucinations, or are limited to static low-density smoke. We introduce SmokeSeer, a method for simultaneous 3D scene reconstruction and smoke removal from multi-view video sequences. Our method uses thermal and RGB images, leveraging the reduced scattering in thermal images to see through smoke. We build upon 3D Gaussian splatting to fuse information from the two image modalities, and decompose the scene into smoke and non-smoke components. Unlike prior work, SmokeSeer handles a broad range of smoke densities and adapts to temporally varying smoke. We validate our method on synthetic data and a new real-world smoke dataset with RGB and thermal images. We provide an open-source implementation and data on the project website.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05991v2">EmoDiffTalk:Emotion-aware Diffusion for Editable 3D Gaussian Talking Head</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Recent photo-realistic 3D talking head via 3D Gaussian Splatting still has significant shortcoming in emotional expression manipulation, especially for fine-grained and expansive dynamics emotional editing using multi-modal control. This paper introduces a new editable 3D Gaussian talking head, i.e. EmoDiffTalk. Our key idea is a novel Emotion-aware Gaussian Diffusion, which includes an action unit (AU) prompt Gaussian diffusion process for fine-grained facial animator, and moreover an accurate text-to-AU emotion controller to provide accurate and expansive dynamic emotional editing using text input. Experiments on public EmoTalk3D and RenderMe-360 datasets demonstrate superior emotional subtlety, lip-sync fidelity, and controllability of our EmoDiffTalk over previous works, establishing a principled pathway toward high-quality, diffusion-driven, multimodal editable 3D talking-head synthesis. To our best knowledge, our EmoDiffTalk is one of the first few 3D Gaussian Splatting talking-head generation framework, especially supporting continuous, multimodal emotional editing within the AU-based expression space.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06438v2">AGORA: Adversarial Generation Of Real-time Animatable 3D Gaussian Head Avatars</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      The generation of high-fidelity, animatable 3D human avatars remains a core challenge in computer graphics and vision, with applications in VR, telepresence, and entertainment. Existing approaches based on implicit representations like NeRFs suffer from slow rendering and dynamic inconsistencies, while 3D Gaussian Splatting (3DGS) methods are typically limited to static head generation, lacking dynamic control. We bridge this gap by introducing AGORA, a novel framework that extends 3DGS within a generative adversarial network to produce animatable avatars. Our key contribution is a lightweight, FLAME-conditioned deformation branch that predicts per-Gaussian residuals, enabling identity-preserving, fine-grained expression control while allowing real-time inference. Expression fidelity is enforced via a dual-discriminator training scheme leveraging synthetic renderings of the parametric mesh. AGORA generates avatars that are not only visually realistic but also precisely controllable. Quantitatively, we outperform state-of-the-art NeRF-based methods on expression accuracy while rendering at 250+ FPS on a single GPU, and, notably, at $\sim$9 FPS under CPU-only inference - representing, to our knowledge, the first demonstration of practical CPU-only animatable 3DGS avatar synthesis. This work represents a significant step toward practical, high-performance digital humans. Project website: https://ramazan793.github.io/AGORA/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09925v1">GAINS: Gaussian-based Inverse Rendering from Sparse Multi-View Captures</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 23 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Recent advances in Gaussian Splatting-based inverse rendering extend Gaussian primitives with shading parameters and physically grounded light transport, enabling high-quality material recovery from dense multi-view captures. However, these methods degrade sharply under sparse-view settings, where limited observations lead to severe ambiguity between geometry, reflectance, and lighting. We introduce GAINS (Gaussian-based Inverse rendering from Sparse multi-view captures), a two-stage inverse rendering framework that leverages learning-based priors to stabilize geometry and material estimation. GAINS first refines geometry using monocular depth/normal and diffusion priors, then employs segmentation, intrinsic image decomposition (IID), and diffusion priors to regularize material recovery. Extensive experiments on synthetic and real-world datasets show that GAINS significantly improves material parameter accuracy, relighting quality, and novel-view synthesis compared to state-of-the-art Gaussian-based inverse rendering methods, especially under sparse-view settings. Project page: https://patrickbail.github.io/gains/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09923v1">Splatent: Splatting Diffusion Latents for Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Radiance field representations have recently been explored in the latent space of VAEs that are commonly used by diffusion models. This direction offers efficient rendering and seamless integration with diffusion-based pipelines. However, these methods face a fundamental limitation: The VAE latent space lacks multi-view consistency, leading to blurred textures and missing details during 3D reconstruction. Existing approaches attempt to address this by fine-tuning the VAE, at the cost of reconstruction quality, or by relying on pre-trained diffusion models to recover fine-grained details, at the risk of some hallucinations. We present Splatent, a diffusion-based enhancement framework designed to operate on top of 3D Gaussian Splatting (3DGS) in the latent space of VAEs. Our key insight departs from the conventional 3D-centric view: rather than reconstructing fine-grained details in 3D space, we recover them in 2D from input views through multi-view attention mechanisms. This approach preserves the reconstruction quality of pretrained VAEs while achieving faithful detail recovery. Evaluated across multiple benchmarks, Splatent establishes a new state-of-the-art for VAE latent radiance field reconstruction. We further demonstrate that integrating our method with existing feed-forward frameworks, consistently improves detail preservation, opening new possibilities for high-quality sparse-view 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09903v1">YOPO-Nav: Visual Navigation using 3DGS Graphs from One-Pass Videos</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Visual navigation has emerged as a practical alternative to traditional robotic navigation pipelines that rely on detailed mapping and path planning. However, constructing and maintaining 3D maps is often computationally expensive and memory-intensive. We address the problem of visual navigation when exploration videos of a large environment are available. The videos serve as a visual reference, allowing a robot to retrace the explored trajectories without relying on metric maps. Our proposed method, YOPO-Nav (You Only Pass Once), encodes an environment into a compact spatial representation composed of interconnected local 3D Gaussian Splatting (3DGS) models. During navigation, the framework aligns the robot's current visual observation with this representation and predicts actions that guide it back toward the demonstrated trajectory. YOPO-Nav employs a hierarchical design: a visual place recognition (VPR) module provides coarse localization, while the local 3DGS models refine the goal and intermediate poses to generate control actions. To evaluate our approach, we introduce the YOPO-Campus dataset, comprising 4 hours of egocentric video and robot controller inputs from over 6 km of human-teleoperated robot trajectories. We benchmark recent visual navigation methods on trajectories from YOPO-Campus using a Clearpath Jackal robot. Experimental results show YOPO-Nav provides excellent performance in image-goal navigation for real-world scenes on a physical robot. The dataset and code will be made publicly available for visual navigation and scene representation research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09656v1">ReMoSPLAT: Reactive Mobile Manipulation Control on a Gaussian Splat</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Reactive control can gracefully coordinate the motion of the base and the arm of a mobile manipulator. However, incorporating an accurate representation of the environment to avoid obstacles without involving costly planning remains a challenge. In this work, we present ReMoSPLAT, a reactive controller based on a quadratic program formulation for mobile manipulation that leverages a Gaussian Splat representation for collision avoidance. By integrating additional constraints and costs into the optimisation formulation, a mobile manipulator platform can reach its intended end effector pose while avoiding obstacles, even in cluttered scenes. We investigate the trade-offs of two methods for efficiently calculating robot-obstacle distances, comparing a purely geometric approach with a rasterisation-based approach. Our experiments in simulation on both synthetic and real-world scans demonstrate the feasibility of our method, showing that the proposed approach achieves performance comparable to controllers that rely on perfect ground-truth information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09411v1">D$^2$GSLAM: 4D Dynamic Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Recent advances in Dense Simultaneous Localization and Mapping (SLAM) have demonstrated remarkable performance in static environments. However, dense SLAM in dynamic environments remains challenging. Most methods directly remove dynamic objects and focus solely on static scene reconstruction, which ignores the motion information contained in these dynamic objects. In this paper, we present D$^2$GSLAM, a novel dynamic SLAM system utilizing Gaussian representation, which simultaneously performs accurate dynamic reconstruction and robust tracking within dynamic environments. Our system is composed of four key components: (i) We propose a geometric-prompt dynamic separation method to distinguish between static and dynamic elements of the scene. This approach leverages the geometric consistency of Gaussian representation and scene geometry to obtain coarse dynamic regions. The regions then serve as prompts to guide the refinement of the coarse mask for achieving accurate motion mask. (ii) To facilitate accurate and efficient mapping of the dynamic scene, we introduce dynamic-static composite representation that integrates static 3D Gaussians with dynamic 4D Gaussians. This representation allows for modeling the transitions between static and dynamic states of objects in the scene for composite mapping and optimization. (iii) We employ a progressive pose refinement strategy that leverages both the multi-view consistency of static scene geometry and motion information from dynamic objects to achieve accurate camera tracking. (iv) We introduce a motion consistency loss, which leverages the temporal continuity in object motions for accurate dynamic modeling. Our D$^2$GSLAM demonstrates superior performance on dynamic scenes in terms of mapping and tracking accuracy, while also showing capability in accurate dynamic modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.02316v3">ConsDreamer: Advancing Multi-View Consistency for Zero-Shot Text-to-3D Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 13 pages, 14 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Recent advances in zero-shot text-to-3D generation have revolutionized 3D content creation by enabling direct synthesis from textual descriptions. While state-of-the-art methods leverage 3D Gaussian Splatting with score distillation to enhance multi-view rendering through pre-trained text-to-image (T2I) models, they suffer from inherent prior view biases in T2I priors. These biases lead to inconsistent 3D generation, particularly manifesting as the multi-face Janus problem, where objects exhibit conflicting features across views. To address this fundamental challenge, we propose ConsDreamer, a novel method that mitigates view bias by refining both the conditional and unconditional terms in the score distillation process: (1) a View Disentanglement Module (VDM) that eliminates viewpoint biases in conditional prompts by decoupling irrelevant view components and injecting precise view control; and (2) a similarity-based partial order loss that enforces geometric consistency in the unconditional term by aligning cosine similarities with azimuth relationships. Extensive experiments demonstrate that ConsDreamer can be seamlessly integrated into various 3D representations and score distillation paradigms, effectively mitigating the multi-face Janus problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09335v1">Relightable and Dynamic Gaussian Avatar Reconstruction from Monocular Video</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 8 pages, 9 figures, published in ACM MM 2025
    </div>
    <details class="paper-abstract">
      Modeling relightable and animatable human avatars from monocular video is a long-standing and challenging task. Recently, Neural Radiance Field (NeRF) and 3D Gaussian Splatting (3DGS) methods have been employed to reconstruct the avatars. However, they often produce unsatisfactory photo-realistic results because of insufficient geometrical details related to body motion, such as clothing wrinkles. In this paper, we propose a 3DGS-based human avatar modeling framework, termed as Relightable and Dynamic Gaussian Avatar (RnD-Avatar), that presents accurate pose-variant deformation for high-fidelity geometrical details. To achieve this, we introduce dynamic skinning weights that define the human avatar's articulation based on pose while also learning additional deformations induced by body motion. We also introduce a novel regularization to capture fine geometric details under sparse visual cues. Furthermore, we present a new multi-view dataset with varied lighting conditions to evaluate relight. Our framework enables realistic rendering of novel poses and views while supporting photo-realistic lighting effects under arbitrary lighting conditions. Our method achieves state-of-the-art performance in novel view synthesis, novel pose rendering, and relighting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09270v1">MoRel: Long-Range Flicker-Free 4D Motion Modeling via Anchor Relay-based Bidirectional Blending with Hierarchical Densification</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 Please visit our project page at https://cmlab-korea.github.io/MoRel/
    </div>
    <details class="paper-abstract">
      Recent advances in 4D Gaussian Splatting (4DGS) have extended the high-speed rendering capability of 3D Gaussian Splatting (3DGS) into the temporal domain, enabling real-time rendering of dynamic scenes. However, one of the major remaining challenges lies in modeling long-range motion-contained dynamic videos, where a naive extension of existing methods leads to severe memory explosion, temporal flickering, and failure to handle appearing or disappearing occlusions over time. To address these challenges, we propose a novel 4DGS framework characterized by an Anchor Relay-based Bidirectional Blending (ARBB) mechanism, named MoRel, which enables temporally consistent and memory-efficient modeling of long-range dynamic scenes. Our method progressively constructs locally canonical anchor spaces at key-frame time index and models inter-frame deformations at the anchor level, enhancing temporal coherence. By learning bidirectional deformations between KfA and adaptively blending them through learnable opacity control, our approach mitigates temporal discontinuities and flickering artifacts. We further introduce a Feature-variance-guided Hierarchical Densification (FHD) scheme that effectively densifies KfA's while keeping rendering quality, based on an assigned level of feature-variance. To effectively evaluate our model's capability to handle real-world long-range 4D motion, we newly compose long-range 4D motion-contained dataset, called SelfCap$_{\text{LR}}$. It has larger average dynamic motion magnitude, captured at spatially wider spaces, compared to previous dynamic video datasets. Overall, our MoRel achieves temporally coherent and flicker-free long-range 4D reconstruction while maintaining bounded memory usage, demonstrating both scalability and efficiency in dynamic Gaussian-based representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08478v1">Visionary: The World Model Carrier Built on WebGPU-Powered Gaussian Splatting Platform</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 Project page: https://visionary-laboratory.github.io/visionary
    </div>
    <details class="paper-abstract">
      Neural rendering, particularly 3D Gaussian Splatting (3DGS), has evolved rapidly and become a key component for building world models. However, existing viewer solutions remain fragmented, heavy, or constrained by legacy pipelines, resulting in high deployment friction and limited support for dynamic content and generative models. In this work, we present Visionary, an open, web-native platform for real-time various Gaussian Splatting and meshes rendering. Built on an efficient WebGPU renderer with per-frame ONNX inference, Visionary enables dynamic neural processing while maintaining a lightweight, "click-to-run" browser experience. It introduces a standardized Gaussian Generator contract, which not only supports standard 3DGS rendering but also allows plug-and-play algorithms to generate or update Gaussians each frame. Such inference also enables us to apply feedforward generative post-processing. The platform further offers a plug in three.js library with a concise TypeScript API for seamless integration into existing web applications. Experiments show that, under identical 3DGS assets, Visionary achieves superior rendering efficiency compared to current Web viewers due to GPU-based primitive sorting. It already supports multiple variants, including MLP-based 3DGS, 4DGS, neural avatars, and style transformation or enhancement networks. By unifying inference and rendering directly in the browser, Visionary significantly lowers the barrier to reproduction, comparison, and deployment of 3DGS-family methods, serving as a unified World Model Carrier for both reconstructive and generative paradigms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.02316v2">ConsDreamer: Advancing Multi-View Consistency for Zero-Shot Text-to-3D Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 13 pages, 14 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Recent advances in zero-shot text-to-3D generation have revolutionized 3D content creation by enabling direct synthesis from textual descriptions. While state-of-the-art methods leverage 3D Gaussian Splatting with score distillation to enhance multi-view rendering through pre-trained text-to-image (T2I) models, they suffer from inherent prior view biases in T2I priors. These biases lead to inconsistent 3D generation, particularly manifesting as the multi-face Janus problem, where objects exhibit conflicting features across views. To address this fundamental challenge, we propose ConsDreamer, a novel method that mitigates view bias by refining both the conditional and unconditional terms in the score distillation process: (1) a View Disentanglement Module (VDM) that eliminates viewpoint biases in conditional prompts by decoupling irrelevant view components and injecting precise view control; and (2) a similarity-based partial order loss that enforces geometric consistency in the unconditional term by aligning cosine similarities with azimuth relationships. Extensive experiments demonstrate that ConsDreamer can be seamlessly integrated into various 3D representations and score distillation paradigms, effectively mitigating the multi-face Janus problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08334v1">HybridSplat: Fast Reflection-baked Gaussian Tracing using Hybrid Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Rendering complex reflection of real-world scenes using 3D Gaussian splatting has been a quite promising solution for photorealistic novel view synthesis, but still faces bottlenecks especially in rendering speed and memory storage. This paper proposes a new Hybrid Splatting(HybridSplat) mechanism for Gaussian primitives. Our key idea is a new reflection-baked Gaussian tracing, which bakes the view-dependent reflection within each Gaussian primitive while rendering the reflection using tile-based Gaussian splatting. Then we integrate the reflective Gaussian primitives with base Gaussian primitives using a unified hybrid splatting framework for high-fidelity scene reconstruction. Moreover, we further introduce a pipeline-level acceleration for the hybrid splatting, and reflection-sensitive Gaussian pruning to reduce the model size, thus achieving much faster rendering speed and lower memory storage while preserving the reflection rendering quality. By extensive evaluation, our HybridSplat accelerates about 7x rendering speed across complex reflective scenes from Ref-NeRF, NeRF-Casting with 4x fewer Gaussian primitives than similar ray-tracing based Gaussian splatting baselines, serving as a new state-of-the-art method especially for complex reflective scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08271v1">Zero-Splat TeleAssist: A Zero-Shot Pose Estimation Framework for Semantic Teleoperation</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 Published and Presented at 3rd Workshop on Human-Centric Multilateral Teleoperation in ICRA 2025
    </div>
    <details class="paper-abstract">
      We introduce Zero-Splat TeleAssist, a zero-shot sensor-fusion pipeline that transforms commodity CCTV streams into a shared, 6-DoF world model for multilateral teleoperation. By integrating vision-language segmentation, monocular depth, weighted-PCA pose extraction, and 3D Gaussian Splatting (3DGS), TeleAssist provides every operator with real-time global positions and orientations of multiple robots without fiducials or depth sensors in an interaction-centric teleoperation setup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07107v2">COREA: Coarse-to-Fine 3D Representation Alignment Between Relightable 3D Gaussians and SDF via Bidirectional 3D-to-3D Supervision</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 Project page: https://cau-vilab.github.io/COREA/
    </div>
    <details class="paper-abstract">
      We present COREA, the first unified framework that jointly learns relightable 3D Gaussians and a Signed Distance Field (SDF) for accurate geometry reconstruction and faithful relighting. While recent 3D Gaussian Splatting (3DGS) methods have extended toward mesh reconstruction and physically-based rendering (PBR), their geometry is still learned from 2D renderings, leading to coarse surfaces and unreliable BRDF-lighting decomposition. To address these limitations, COREA introduces a coarse-to-fine bidirectional 3D-to-3D alignment strategy that allows geometric signals to be learned directly in 3D space. Within this strategy, depth provides coarse alignment between the two representations, while depth gradients and normals refine fine-scale structure, and the resulting geometry supports stable BRDF-lighting decomposition. A density-control mechanism further stabilizes Gaussian growth, balancing geometric fidelity with memory efficiency. Experiments on standard benchmarks demonstrate that COREA achieves superior performance in novel-view synthesis, mesh reconstruction, and PBR within a unified framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09162v1">GTAvatar: Bridging Gaussian Splatting and Texture Mapping for Relightable and Editable Gaussian Avatars</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Recent advancements in Gaussian Splatting have enabled increasingly accurate reconstruction of photorealistic head avatars, opening the door to numerous applications in visual effects, videoconferencing, and virtual reality. This, however, comes with the lack of intuitive editability offered by traditional triangle mesh-based methods. In contrast, we propose a method that combines the accuracy and fidelity of 2D Gaussian Splatting with the intuitiveness of UV texture mapping. By embedding each canonical Gaussian primitive's local frame into a patch in the UV space of a template mesh in a computationally efficient manner, we reconstruct continuous editable material head textures from a single monocular video on a conventional UV domain. Furthermore, we leverage an efficient physically based reflectance model to enable relighting and editing of these intrinsic material maps. Through extensive comparisons with state-of-the-art methods, we demonstrate the accuracy of our reconstructions, the quality of our relighting results, and the ability to provide intuitive controls for modifying an avatar's appearance and geometry via texture mapping without additional optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.22676v4">TranSplat: Instant Cross-Scene Object Relighting in Gaussian Splatting via Spherical Harmonic Transfer</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      We present TranSplat, a method for fast and accurate object relighting for the 3D Gaussian Splatting (GS) framework when transferring a 3D object from a source GS scene to a target GS scene. TranSplat is based on a theoretical radiance transfer identity for cross-scene relighting of objects with radially symmetric BRDFs that involves only taking simple products of spherical harmonic appearance coefficients of the object, source, and target environment maps without any explicit computation of scene quantities (e.g., the BRDFs themselves). TranSplat is the first method to demonstrate how this theoretical identity may be used to perform relighting within the GS framework, and furthermore, by automatically inferring unknown source and target environment maps directly from the source and target scene GS representations. We evaluated TranSplat on several synthetic and real-world scenes and objects, demonstrating comparable 3D object relighting performance to recent conventional inverse rendering-based GS methods with a fraction of their runtime. While TranSplat is theoretically best-suited for radially symmetric BRDFs, results demonstrate that TranSplat still offers perceptually realistic renderings on real scenes and opens a valuable, lightweight path forward to relighting with the GS framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08625v1">OpenMonoGS-SLAM: Monocular Gaussian Splatting SLAM with Open-set Semantics</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Simultaneous Localization and Mapping (SLAM) is a foundational component in robotics, AR/VR, and autonomous systems. With the rising focus on spatial AI in recent years, combining SLAM with semantic understanding has become increasingly important for enabling intelligent perception and interaction. Recent efforts have explored this integration, but they often rely on depth sensors or closed-set semantic models, limiting their scalability and adaptability in open-world environments. In this work, we present OpenMonoGS-SLAM, the first monocular SLAM framework that unifies 3D Gaussian Splatting (3DGS) with open-set semantic understanding. To achieve our goal, we leverage recent advances in Visual Foundation Models (VFMs), including MASt3R for visual geometry and SAM and CLIP for open-vocabulary semantics. These models provide robust generalization across diverse tasks, enabling accurate monocular camera tracking and mapping, as well as a rich understanding of semantics in open-world environments. Our method operates without any depth input or 3D semantic ground truth, relying solely on self-supervised learning objectives. Furthermore, we propose a memory mechanism specifically designed to manage high-dimensional semantic features, which effectively constructs Gaussian semantic feature maps, leading to strong overall performance. Experimental results demonstrate that our approach achieves performance comparable to or surpassing existing baselines in both closed-set and open-set segmentation tasks, all without relying on supplementary sensors such as depth maps or semantic annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02069v2">Spec-Gloss Surfels and Normal-Diffuse Priors for Relightable Glossy Objects</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Accurate reconstruction and relighting of glossy objects remains a longstanding challenge, as object shape, material properties, and illumination are inherently difficult to disentangle. Existing neural rendering approaches often rely on simplified BRDF models or parameterizations that couple diffuse and specular components, which restrict faithful material recovery and limit relighting fidelity. We propose a relightable framework that integrates a microfacet BRDF with the specular-glossiness parameterization into 2D Gaussian Splatting with deferred shading. This formulation enables more physically consistent material decomposition, while diffusion-based priors for surface normals and diffuse color guide early-stage optimization and mitigate ambiguity. A coarse-to-fine environment map optimization accelerates convergence, and negative-only environment map clipping preserves high-dynamic-range specular reflections. Extensive experiments on complex, glossy scenes demonstrate that our method achieves high-quality geometry and material reconstruction, delivering substantially more realistic and consistent relighting under novel illumination compared to existing Gaussian splatting methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.18468v6">RGS-DR: Deferred Reflections and Residual Shading in 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      In this work, we address specular appearance in inverse rendering using 2D Gaussian splatting with deferred shading and argue for a refinement stage to improve specular detail, thereby bridging the gap with reconstruction-only methods. Our pipeline estimates editable material properties and environment illumination while employing a directional residual pass that captures leftover view-dependent effects for further refining novel view synthesis. In contrast to per-Gaussian shading with shortest-axis normals and normal residuals, which tends to result in more noisy geometry and specular appearance, a pixel-deferred surfel formulation with specular residuals yields sharper highlights, cleaner materials, and improved editability. We evaluate our approach on rendering and reconstruction quality on three popular datasets featuring glossy objects, and also demonstrate high-quality relighting and material editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08498v1">On-the-fly Large-scale 3D Reconstruction from Multi-Camera Rigs</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled efficient free-viewpoint rendering and photorealistic scene reconstruction. While on-the-fly extensions of 3DGS have shown promise for real-time reconstruction from monocular RGB streams, they often fail to achieve complete 3D coverage due to the limited field of view (FOV). Employing a multi-camera rig fundamentally addresses this limitation. In this paper, we present the first on-the-fly 3D reconstruction framework for multi-camera rigs. Our method incrementally fuses dense RGB streams from multiple overlapping cameras into a unified Gaussian representation, achieving drift-free trajectory estimation and efficient online reconstruction. We propose a hierarchical camera initialization scheme that enables coarse inter-camera alignment without calibration, followed by a lightweight multi-camera bundle adjustment that stabilizes trajectories while maintaining real-time performance. Furthermore, we introduce a redundancy-free Gaussian sampling strategy and a frequency-aware optimization scheduler to reduce the number of Gaussian primitives and the required optimization iterations, thereby maintaining both efficiency and reconstruction fidelity. Our method reconstructs hundreds of meters of 3D scenes within just 2 minutes using only raw multi-camera video streams, demonstrating unprecedented speed, robustness, and Fidelity for on-the-fly 3D scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.22676v3">TranSplat: Instant Cross-Scene Object Relighting in Gaussian Splatting via Spherical Harmonic Transfer</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      We present TranSplat, a method for fast and accurate object relighting for the 3D Gaussian Splatting (GS) framework when transferring a 3D object from a source GS scene to a target GS scene. TranSplat is based on a theoretical radiance transfer identity for cross-scene relighting of objects with radially symmetric BRDFs that involves only taking simple products of spherical harmonic appearance coefficients of the object, source, and target environment maps without any explicit computation of scene quantities (e.g., the BRDFs themselves). TranSplat is the first method to demonstrate how this theoretical identity may be used to perform relighting within the GS framework, and furthermore, by automatically inferring unknown source and target environment maps directly from the source and target scene GS representations. We evaluated TranSplat on several synthetic and real-world scenes and objects, demonstrating comparable 3D object relighting performance to recent conventional inverse rendering-based GS methods with a fraction of their runtime. While TranSplat is theoretically best-suited for radially symmetric BRDFs, results demonstrate that TranSplat still offers perceptually realistic renderings on real scenes and opens a valuable, lightweight path forward to relighting with the GS framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07381v1">Tessellation GS: Neural Mesh Gaussians for Robust Monocular Reconstruction of Dynamic Objects</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (GS) enables highly photorealistic scene reconstruction from posed image sequences but struggles with viewpoint extrapolation due to its anisotropic nature, leading to overfitting and poor generalization, particularly in sparse-view and dynamic scene reconstruction. We propose Tessellation GS, a structured 2D GS approach anchored on mesh faces, to reconstruct dynamic scenes from a single continuously moving or static camera. Our method constrains 2D Gaussians to localized regions and infers their attributes via hierarchical neural features on mesh faces. Gaussian subdivision is guided by an adaptive face subdivision strategy driven by a detail-aware loss function. Additionally, we leverage priors from a reconstruction foundation model to initialize Gaussian deformations, enabling robust reconstruction of general dynamic objects from a single static camera, previously extremely challenging for optimization-based methods. Our method outperforms previous SOTA method, reducing LPIPS by 29.1% and Chamfer distance by 49.2% on appearance and mesh reconstruction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07345v1">Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 15 pages, 8 figures, 5 tables, 2 algorithms, Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Versatile 3D tasks (e.g., generation or editing) that distill from Text-to-Image (T2I) diffusion models have attracted significant research interest for not relying on extensive 3D training data. However, T2I models exhibit limitations resulting from prior view bias, which produces conflicting appearances between different views of an object. This bias causes subject-words to preferentially activate prior view features during cross-attention (CA) computation, regardless of the target view condition. To overcome this limitation, we conduct a comprehensive mathematical analysis to reveal the root cause of the prior view bias in T2I models. Moreover, we find different UNet layers show different effects of prior view in CA. Therefore, we propose a novel framework, TD-Attn, which addresses multi-view inconsistency via two key components: (1) the 3D-Aware Attention Guidance Module (3D-AAG) constructs a view-consistent 3D attention Gaussian for subject-words to enforce spatial consistency across attention-focused regions, thereby compensating for the limited spatial information in 2D individual view CA maps; (2) the Hierarchical Attention Modulation Module (HAM) utilizes a Semantic Guidance Tree (SGT) to direct the Semantic Response Profiler (SRP) in localizing and modulating CA layers that are highly responsive to view conditions, where the enhanced CA maps further support the construction of more consistent 3D attention Gaussians. Notably, HAM facilitates semantic-specific interventions, enabling controllable and precise 3D editing. Extensive experiments firmly establish that TD-Attn has the potential to serve as a universal plugin, significantly enhancing multi-view consistency across 3D tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05113v2">Splannequin: Freezing Monocular Mannequin-Challenge Footage with Dual-Detection Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 WACV 2026. Project page: https://chien90190.github.io/splannequin/
    </div>
    <details class="paper-abstract">
      Synthesizing high-fidelity frozen 3D scenes from monocular Mannequin-Challenge (MC) videos is a unique problem distinct from standard dynamic scene reconstruction. Instead of focusing on modeling motion, our goal is to create a frozen scene while strategically preserving subtle dynamics to enable user-controlled instant selection. To achieve this, we introduce a novel application of dynamic Gaussian splatting: the scene is modeled dynamically, which retains nearby temporal variation, and a static scene is rendered by fixing the model's time parameter. However, under this usage, monocular capture with sparse temporal supervision introduces artifacts like ghosting and blur for Gaussians that become unobserved or occluded at weakly supervised timestamps. We propose Splannequin, an architecture-agnostic regularization that detects two states of Gaussian primitives, hidden and defective, and applies temporal anchoring. Under predominantly forward camera motion, hidden states are anchored to their recent well-observed past states, while defective states are anchored to future states with stronger supervision. Our method integrates into existing dynamic Gaussian pipelines via simple loss terms, requires no architectural changes, and adds zero inference overhead. This results in markedly improved visual quality, enabling high-fidelity, user-selectable frozen-time renderings, validated by a 96% user preference. Project page: https://chien90190.github.io/splannequin/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07247v1">AdLift: Lifting Adversarial Perturbations to Safeguard 3D Gaussian Splatting Assets Against Instruction-Driven Editing</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 40 pages, 34 figures, 18 tables
    </div>
    <details class="paper-abstract">
      Recent studies have extended diffusion-based instruction-driven 2D image editing pipelines to 3D Gaussian Splatting (3DGS), enabling faithful manipulation of 3DGS assets and greatly advancing 3DGS content creation. However, it also exposes these assets to serious risks of unauthorized editing and malicious tampering. Although imperceptible adversarial perturbations against diffusion models have proven effective for protecting 2D images, applying them to 3DGS encounters two major challenges: view-generalizable protection and balancing invisibility with protection capability. In this work, we propose the first editing safeguard for 3DGS, termed AdLift, which prevents instruction-driven editing across arbitrary views and dimensions by lifting strictly bounded 2D adversarial perturbations into 3D Gaussian-represented safeguard. To ensure both adversarial perturbations effectiveness and invisibility, these safeguard Gaussians are progressively optimized across training views using a tailored Lifted PGD, which first conducts gradient truncation during back-propagation from the editing model at the rendered image and applies projected gradients to strictly constrain the image-level perturbation. Then, the resulting perturbation is backpropagated to the safeguard Gaussian parameters via an image-to-Gaussian fitting operation. We alternate between gradient truncation and image-to-Gaussian fitting, yielding consistent adversarial-based protection performance across different viewpoints and generalizes to novel views. Empirically, qualitative and quantitative results demonstrate that AdLift effectively protects against state-of-the-art instruction-driven 2D image and 3DGS editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07230v1">STRinGS: Selective Text Refinement in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 Accepted to WACV 2026. Project Page, see https://STRinGS-official.github.io
    </div>
    <details class="paper-abstract">
      Text as signs, labels, or instructions is a critical element of real-world scenes as they can convey important contextual information. 3D representations such as 3D Gaussian Splatting (3DGS) struggle to preserve fine-grained text details, while achieving high visual fidelity. Small errors in textual element reconstruction can lead to significant semantic loss. We propose STRinGS, a text-aware, selective refinement framework to address this issue for 3DGS reconstruction. Our method treats text and non-text regions separately, refining text regions first and merging them with non-text regions later for full-scene optimization. STRinGS produces sharp, readable text even in challenging configurations. We introduce a text readability measure OCR Character Error Rate (CER) to evaluate the efficacy on text regions. STRinGS results in a 63.6% relative improvement over 3DGS at just 7K iterations. We also introduce a curated dataset STRinGS-360 with diverse text scenarios to evaluate text readability in 3D reconstruction. Our method and dataset together push the boundaries of 3D scene understanding in text-rich environments, paving the way for more robust text-aware reconstruction methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07197v1">SUCCESS-GS: Survey of Compactness and Compression for Efficient Static and Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 The first three authors contributed equally to this work. The last two authors are co-corresponding authors. Please visit our project page at https://cmlab-korea.github.io/Awesome-Efficient-GS/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful explicit representation enabling real-time, high-fidelity 3D reconstruction and novel view synthesis. However, its practical use is hindered by the massive memory and computational demands required to store and render millions of Gaussians. These challenges become even more severe in 4D dynamic scenes. To address these issues, the field of Efficient Gaussian Splatting has rapidly evolved, proposing methods that reduce redundancy while preserving reconstruction quality. This survey provides the first unified overview of efficient 3D and 4D Gaussian Splatting techniques. For both 3D and 4D settings, we systematically categorize existing methods into two major directions, Parameter Compression and Restructuring Compression, and comprehensively summarize the core ideas and methodological trends within each category. We further cover widely used datasets, evaluation metrics, and representative benchmark comparisons. Finally, we discuss current limitations and outline promising research directions toward scalable, compact, and real-time Gaussian Splatting for both static and dynamic 3D scene representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07165v1">MuSASplat: Efficient Sparse-View 3D Gaussian Splats via Lightweight Multi-Scale Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Sparse-view 3D Gaussian splatting seeks to render high-quality novel views of 3D scenes from a limited set of input images. While recent pose-free feed-forward methods leveraging pre-trained 3D priors have achieved impressive results, most of them rely on full fine-tuning of large Vision Transformer (ViT) backbones and incur substantial GPU costs. In this work, we introduce MuSASplat, a novel framework that dramatically reduces the computational burden of training pose-free feed-forward 3D Gaussian splats models with little compromise of rendering quality. Central to our approach is a lightweight Multi-Scale Adapter that enables efficient fine-tuning of ViT-based architectures with only a small fraction of training parameters. This design avoids the prohibitive GPU overhead associated with previous full-model adaptation techniques while maintaining high fidelity in novel view synthesis, even with very sparse input views. In addition, we introduce a Feature Fusion Aggregator that integrates features across input views effectively and efficiently. Unlike widely adopted memory banks, the Feature Fusion Aggregator ensures consistent geometric integration across input views and meanwhile mitigates the memory usage, training complexity, and computational costs significantly. Extensive experiments across diverse datasets show that MuSASplat achieves state-of-the-art rendering quality but has significantly reduced parameters and training resource requirements as compared with existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07107v1">COREA: Coarse-to-Fine 3D Representation Alignment Between Relightable 3D Gaussians and SDF via Bidirectional 3D-to-3D Supervision</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 Project page: https://vilab-cau.github.io/COREA/
    </div>
    <details class="paper-abstract">
      We present COREA, the first unified framework that jointly learns relightable 3D Gaussians and a Signed Distance Field (SDF) for accurate geometry reconstruction and faithful relighting. While recent 3D Gaussian Splatting (3DGS) methods have extended toward mesh reconstruction and physically-based rendering (PBR), their geometry is still learned from 2D renderings, leading to coarse surfaces and unreliable BRDF-lighting decomposition. To address these limitations, COREA introduces a coarse-to-fine bidirectional 3D-to-3D alignment strategy that allows geometric signals to be learned directly in 3D space. Within this strategy, depth provides coarse alignment between the two representations, while depth gradients and normals refine fine-scale structure, and the resulting geometry supports stable BRDF-lighting decomposition. A density-control mechanism further stabilizes Gaussian growth, balancing geometric fidelity with memory efficiency. Experiments on standard benchmarks demonstrate that COREA achieves superior performance in novel-view synthesis, mesh reconstruction, and PBR within a unified framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07806v1">Multi-view Pyramid Transformer: Look Coarser to See Broader</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 Project page: see https://gynjn.github.io/MVP/
    </div>
    <details class="paper-abstract">
      We propose Multi-view Pyramid Transformer (MVP), a scalable multi-view transformer architecture that directly reconstructs large 3D scenes from tens to hundreds of images in a single forward pass. Drawing on the idea of ``looking broader to see the whole, looking finer to see the details," MVP is built on two core design principles: 1) a local-to-global inter-view hierarchy that gradually broadens the model's perspective from local views to groups and ultimately the full scene, and 2) a fine-to-coarse intra-view hierarchy that starts from detailed spatial representations and progressively aggregates them into compact, information-dense tokens. This dual hierarchy achieves both computational efficiency and representational richness, enabling fast reconstruction of large and complex scenes. We validate MVP on diverse datasets and show that, when coupled with 3D Gaussian Splatting as the underlying 3D representation, it achieves state-of-the-art generalizable reconstruction quality while maintaining high efficiency and scalability across a wide range of view configurations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07052v1">RAVE: Rate-Adaptive Visual Encoding for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Recent advances in neural scene representations have transformed immersive multimedia, with 3D Gaussian Splatting (3DGS) enabling real-time photorealistic rendering. Despite its efficiency, 3DGS suffers from large memory requirements and costly training procedures, motivating efforts toward compression. Existing approaches, however, operate at fixed rates, limiting adaptability to varying bandwidth and device constraints. In this work, we propose a flexible compression scheme for 3DGS that supports interpolation at any rate between predefined bounds. Our method is computationally lightweight, requires no retraining for any rate, and preserves rendering quality across a broad range of operating points. Experiments demonstrate that the approach achieves efficient, high-quality compression while offering dynamic rate control, making it suitable for practical deployment in immersive applications. The code will be provided open-source upon acceptance of the work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06818v1">MeshSplatting: Differentiable Rendering with Opaque Meshes</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Primitive-based splatting methods like 3D Gaussian Splatting have revolutionized novel view synthesis with real-time rendering. However, their point-based representations remain incompatible with mesh-based pipelines that power AR/VR and game engines. We present MeshSplatting, a mesh-based reconstruction approach that jointly optimizes geometry and appearance through differentiable rendering. By enforcing connectivity via restricted Delaunay triangulation and refining surface consistency, MeshSplatting creates end-to-end smooth, visually high-quality meshes that render efficiently in real-time 3D engines. On Mip-NeRF360, it boosts PSNR by +0.69 dB over the current state-of-the-art MiLo for mesh-based novel view synthesis, while training 2x faster and using 2x less memory, bridging neural rendering and interactive 3D graphics for seamless real-time scene interaction. The project page is available at https://meshsplatting.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06774v1">RDSplat: Robust Watermarking Against Diffusion Editing for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has enabled the creation of digital assets and downstream applications, underscoring the need for robust copyright protection via digital watermarking. However, existing 3DGS watermarking methods remain highly vulnerable to diffusion-based editing, which can easily erase embedded provenance. This challenge highlights the urgent need for 3DGS watermarking techniques that are intrinsically resilient to diffusion-based editing. In this paper, we introduce RDSplat, a Robust watermarking paradigm against Diffusion editing for 3D Gaussian Splatting. RDSplat embeds watermarks into 3DGS components that diffusion-based editing inherently preserve, achieved through (i) proactively targeting low-frequency Gaussians and (ii) adversarial training with a diffusion proxy. Specifically, we introduce a multi-domain framework that operates natively in 3DGS space and embeds watermarks into diffusion-editing-preserved low-frequency Gaussians via coordinated covariance regularization and 2D filtering. In addition, we exploit the low-pass filtering behavior of diffusion-based editing by using Gaussian blur as an efficient training surrogate, enabling adversarial fine-tuning that further enhances watermark robustness against diffusion-based editing. Empirically, comprehensive quantitative and qualitative evaluations on three benchmark datasets demonstrate that RDSplat not only maintains superior robustness under diffusion-based editing, but also preserves watermark invisibility, achieving state-of-the-art performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06684v1">EMGauss: Continuous Slice-to-3D Reconstruction via Dynamic Gaussian Modeling in Volume Electron Microscopy</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Volume electron microscopy (vEM) enables nanoscale 3D imaging of biological structures but remains constrained by acquisition trade-offs, leading to anisotropic volumes with limited axial resolution. Existing deep learning methods seek to restore isotropy by leveraging lateral priors, yet their assumptions break down for morphologically anisotropic structures. We present EMGauss, a general framework for 3D reconstruction from planar scanned 2D slices with applications in vEM, which circumvents the inherent limitations of isotropy-based approaches. Our key innovation is to reframe slice-to-3D reconstruction as a 3D dynamic scene rendering problem based on Gaussian splatting, where the progression of axial slices is modeled as the temporal evolution of 2D Gaussian point clouds. To enhance fidelity in data-sparse regimes, we incorporate a Teacher-Student bootstrapping mechanism that uses high-confidence predictions on unobserved slices as pseudo-supervisory signals. Compared with diffusion- and GAN-based reconstruction methods, EMGauss substantially improves interpolation quality, enables continuous slice synthesis, and eliminates the need for large-scale pretraining. Beyond vEM, it potentially provides a generalizable slice-to-3D solution across diverse imaging domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.17792v3">CrowdSplat: Exploring Gaussian Splatting For Crowd Rendering</a></div>
    <div class="paper-meta">
      📅 2025-12-06
      | 💬 4 pages, 4 figures
    </div>
    <details class="paper-abstract">
      We present CrowdSplat, a novel approach that leverages 3D Gaussian Splatting for real-time, high-quality crowd rendering. Our method utilizes 3D Gaussian functions to represent animated human characters in diverse poses and outfits, which are extracted from monocular videos. We integrate Level of Detail (LoD) rendering to optimize computational efficiency and quality. The CrowdSplat framework consists of two stages: (1) avatar reconstruction and (2) crowd synthesis. The framework is also optimized for GPU memory usage to enhance scalability. Quantitative and qualitative evaluations show that CrowdSplat achieves good levels of rendering quality, memory efficiency, and computational performance. Through the.se experiments, we demonstrate that CrowdSplat is a viable solution for dynamic, realistic crowd simulation in real-time applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23292v2">FACT-GS: Frequency-Aligned Complexity-Aware Texture Reparameterization for 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-06
      | 💬 11 pages, 6 figures, preprint
    </div>
    <details class="paper-abstract">
      Realistic scene appearance modeling has advanced rapidly with Gaussian Splatting, which enables real-time, high-quality rendering. Recent advances introduced per-primitive textures that incorporate spatial color variations within each Gaussian, improving their expressiveness. However, texture-based Gaussians parameterize appearance with a uniform per-Gaussian sampling grid, allocating equal sampling density regardless of local visual complexity. This leads to inefficient texture space utilization, where high-frequency regions are under-sampled and smooth regions waste capacity, causing blurred appearance and loss of fine structural detail. We introduce FACT-GS, a Frequency-Aligned Complexity-aware Texture Gaussian Splatting framework that allocates texture sampling density according to local visual frequency. Grounded in adaptive sampling theory, FACT-GS reformulates texture parameterization as a differentiable sampling-density allocation problem, replacing the uniform textures with a learnable frequency-aware allocation strategy implemented via a deformation field whose Jacobian modulates local sampling density. Built on 2D Gaussian Splatting, FACT-GS performs non-uniform sampling on fixed-resolution texture grids, preserving real-time performance while recovering sharper high-frequency details under the same parameter budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06438v1">AGORA: Adversarial Generation Of Real-time Animatable 3D Gaussian Head Avatars</a></div>
    <div class="paper-meta">
      📅 2025-12-06
    </div>
    <details class="paper-abstract">
      The generation of high-fidelity, animatable 3D human avatars remains a core challenge in computer graphics and vision, with applications in VR, telepresence, and entertainment. Existing approaches based on implicit representations like NeRFs suffer from slow rendering and dynamic inconsistencies, while 3D Gaussian Splatting (3DGS) methods are typically limited to static head generation, lacking dynamic control. We bridge this gap by introducing AGORA, a novel framework that extends 3DGS within a generative adversarial network to produce animatable avatars. Our key contribution is a lightweight, FLAME-conditioned deformation branch that predicts per-Gaussian residuals, enabling identity-preserving, fine-grained expression control while allowing real-time inference. Expression fidelity is enforced via a dual-discriminator training scheme leveraging synthetic renderings of the parametric mesh. AGORA generates avatars that are not only visually realistic but also precisely controllable. Quantitatively, we outperform state-of-the-art NeRF-based methods on expression accuracy while rendering at 250+ FPS on a single GPU, and, notably, at $\sim$9 FPS under CPU-only inference - representing, to our knowledge, the first demonstration of practical CPU-only animatable 3DGS avatar synthesis. This work represents a significant step toward practical, high-performance digital humans. Project website: https://ramazan793.github.io/AGORA/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08575v2">ReSplat: Learning Recurrent Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2025-12-06
      | 💬 Project page: https://haofeixu.github.io/resplat/
    </div>
    <details class="paper-abstract">
      While feed-forward Gaussian splatting models offer computational efficiency and can generalize to sparse input settings, their performance is fundamentally constrained by relying on a single forward pass for inference. We propose ReSplat, a feed-forward recurrent Gaussian splatting model that iteratively refines 3D Gaussians without explicitly computing gradients. Our key insight is that the Gaussian splatting rendering error serves as a rich feedback signal, guiding the recurrent network to learn effective Gaussian updates. This feedback signal naturally adapts to unseen data distributions at test time, enabling robust generalization across datasets, view counts and image resolutions. To initialize the recurrent process, we introduce a compact reconstruction model that operates in a $16 \times$ subsampled space, producing $16 \times$ fewer Gaussians than previous per-pixel Gaussian models. This substantially reduces computational overhead and allows for efficient Gaussian updates. Extensive experiments across varying of input views (2, 8, 16, 32), resolutions ($256 \times 256$ to $540 \times 960$), and datasets (DL3DV, RealEstate10K and ACID) demonstrate that our method achieves state-of-the-art performance while significantly reducing the number of Gaussians and improving the rendering speed. Our project page is at https://haofeixu.github.io/resplat/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06269v1">TriaGS: Differentiable Triangulation-Guided Geometric Consistency for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-06
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting is crucial for real-time novel view synthesis due to its efficiency and ability to render photorealistic images. However, building a 3D Gaussian is guided solely by photometric loss, which can result in inconsistencies in reconstruction. This under-constrained process often results in "floater" artifacts and unstructured geometry, preventing the extraction of high-fidelity surfaces. To address this issue, our paper introduces a novel method that improves reconstruction by enforcing global geometry consistency through constrained multi-view triangulation. Our approach aims to achieve a consensus on 3D representation in the physical world by utilizing various estimated views. We optimize this process by penalizing the deviation of a rendered 3D point from a robust consensus point, which is re-triangulated from a bundle of neighboring views in a self-supervised fashion. We demonstrate the effectiveness of our method across multiple datasets, achieving state-of-the-art results. On the DTU dataset, our method attains a mean Chamfer Distance of 0.50 mm, outperforming comparable explicit methods. We will make our code open-source to facilitate community validation and ensure reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04283v3">FastGS: Training 3D Gaussian Splatting in 100 Seconds</a></div>
    <div class="paper-meta">
      📅 2025-12-06
      | 💬 Project page: https://fastgs.github.io/
    </div>
    <details class="paper-abstract">
      The dominant 3D Gaussian splatting (3DGS) acceleration methods fail to properly regulate the number of Gaussians during training, causing redundant computational time overhead. In this paper, we propose FastGS, a novel, simple, and general acceleration framework that fully considers the importance of each Gaussian based on multi-view consistency, efficiently solving the trade-off between training time and rendering quality. We innovatively design a densification and pruning strategy based on multi-view consistency, dispensing with the budgeting mechanism. Extensive experiments on Mip-NeRF 360, Tanks & Temples, and Deep Blending datasets demonstrate that our method significantly outperforms the state-of-the-art methods in training speed, achieving a 3.32$\times$ training acceleration and comparable rendering quality compared with DashGaussian on the Mip-NeRF 360 dataset and a 15.45$\times$ acceleration compared with vanilla 3DGS on the Deep Blending dataset. We demonstrate that FastGS exhibits strong generality, delivering 2-7$\times$ training acceleration across various tasks, including dynamic scene reconstruction, surface reconstruction, sparse-view reconstruction, large-scale reconstruction, and simultaneous localization and mapping. The project page is available at https://fastgs.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.13348v2">TextureSplat: Per-Primitive Texture Mapping for Reflective Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-05
      | 💬 3DV 2026
    </div>
    <details class="paper-abstract">
      Gaussian Splatting have demonstrated remarkable novel view synthesis performance at high rendering frame rates. Optimization-based inverse rendering within complex capture scenarios remains however a challenging problem. A particular case is modelling complex surface light interactions for highly reflective scenes, which results in intricate high frequency specular radiance components. We hypothesize that such challenging settings can benefit from increased representation power. We hence propose a method that tackles this issue through a geometrically and physically grounded Gaussian Splatting borne radiance field, where normals and material properties are spatially variable in the primitive's local space. Using per-primitive texture maps for this purpose, we also propose to harness the GPU hardware to accelerate rendering at test time via unified material texture atlas. Code will be available at https://github.com/maeyounes/TextureSplat
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.07118v3">DexFruit: Dexterous Manipulation and Gaussian Splatting Inspection of Fruit</a></div>
    <div class="paper-meta">
      📅 2025-12-05
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      DexFruit is a robotic manipulation framework that enables gentle, autonomous handling of fragile fruit and precise evaluation of damage. Many fruits are fragile and prone to bruising, thus requiring humans to manually harvest them with care. In this work, we demonstrate by using optical tactile sensing, autonomous manipulation of fruit with minimal damage can be achieved. We show that our tactile informed diffusion policies outperform baselines in both reduced bruising and pick-and-place success rate across three fruits: strawberries, tomatoes, and blackberries. In addition, we introduce FruitSplat, a novel technique to represent and quantify visual damage in high-resolution 3D representation via 3D Gaussian Splatting (3DGS). Existing metrics for measuring damage lack quantitative rigor or require expensive equipment. With FruitSplat, we distill a 2D strawberry mask as well as a 2D bruise segmentation mask into the 3DGS representation. Furthermore, this representation is modular and general, compatible with any relevant 2D model. Overall, we demonstrate a 92% grasping policy success rate, up to a 20% reduction in visual bruising, and up to an 31% improvement in grasp success rate on challenging fruit compared to our baselines across our three tested fruits. We rigorously evaluate this result with over 630 trials. Please checkout our website at https://dex-fruit.github.io .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06158v1">Tracking-Guided 4D Generation: Foundation-Tracker Motion Priors for 3D Model Animation</a></div>
    <div class="paper-meta">
      📅 2025-12-05
      | 💬 15 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Generating dynamic 4D objects from sparse inputs is difficult because it demands joint preservation of appearance and motion coherence across views and time while suppressing artifacts and temporal drift. We hypothesize that the view discrepancy arises from supervision limited to pixel- or latent-space video-diffusion losses, which lack explicitly temporally aware, feature-level tracking guidance. We present \emph{Track4DGen}, a two-stage framework that couples a multi-view video diffusion model with a foundation point tracker and a hybrid 4D Gaussian Splatting (4D-GS) reconstructor. The central idea is to explicitly inject tracker-derived motion priors into intermediate feature representations for both multi-view video generation and 4D-GS. In Stage One, we enforce dense, feature-level point correspondences inside the diffusion generator, producing temporally consistent features that curb appearance drift and enhance cross-view coherence. In Stage Two, we reconstruct a dynamic 4D-GS using a hybrid motion encoding that concatenates co-located diffusion features (carrying Stage-One tracking priors) with Hex-plane features, and augment them with 4D Spherical Harmonics for higher-fidelity dynamics modeling. \emph{Track4DGen} surpasses baselines on both multi-view video generation and 4D generation benchmarks, yielding temporally stable, text-editable 4D assets. Lastly, we curate \emph{Sketchfab28}, a high-quality dataset for benchmarking object-centric 4D generation and fostering future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05446v1">TED-4DGS: Temporally Activated and Embedding-based Deformation for 4DGS Compression</a></div>
    <div class="paper-meta">
      📅 2025-12-05
    </div>
    <details class="paper-abstract">
      Building on the success of 3D Gaussian Splatting (3DGS) in static 3D scene representation, its extension to dynamic scenes, commonly referred to as 4DGS or dynamic 3DGS, has attracted increasing attention. However, designing more compact and efficient deformation schemes together with rate-distortion-optimized compression strategies for dynamic 3DGS representations remains an underexplored area. Prior methods either rely on space-time 4DGS with overspecified, short-lived Gaussian primitives or on canonical 3DGS with deformation that lacks explicit temporal control. To address this, we present TED-4DGS, a temporally activated and embedding-based deformation scheme for rate-distortion-optimized 4DGS compression that unifies the strengths of both families. TED-4DGS is built on a sparse anchor-based 3DGS representation. Each canonical anchor is assigned learnable temporal-activation parameters to specify its appearance and disappearance transitions over time, while a lightweight per-anchor temporal embedding queries a shared deformation bank to produce anchor-specific deformation. For rate-distortion compression, we incorporate an implicit neural representation (INR)-based hyperprior to model anchor attribute distributions, along with a channel-wise autoregressive model to capture intra-anchor correlations. With these novel elements, our scheme achieves state-of-the-art rate-distortion performance on several real-world datasets. To the best of our knowledge, this work represents one of the first attempts to pursue a rate-distortion-optimized compression framework for dynamic 3DGS representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05354v1">SplatPainter: Interactive Authoring of 3D Gaussians from 2D Edits via Test-Time Training</a></div>
    <div class="paper-meta">
      📅 2025-12-05
      | 💬 project page https://y-zheng18.github.io/SplatPainter/
    </div>
    <details class="paper-abstract">
      The rise of 3D Gaussian Splatting has revolutionized photorealistic 3D asset creation, yet a critical gap remains for their interactive refinement and editing. Existing approaches based on diffusion or optimization are ill-suited for this task, as they are often prohibitively slow, destructive to the original asset's identity, or lack the precision for fine-grained control. To address this, we introduce \ourmethod, a state-aware feedforward model that enables continuous editing of 3D Gaussian assets from user-provided 2D view(s). Our method directly predicts updates to the attributes of a compact, feature-rich Gaussian representation and leverages Test-Time Training to create a state-aware, iterative workflow. The versatility of our approach allows a single architecture to perform diverse tasks, including high-fidelity local detail refinement, local paint-over, and consistent global recoloring, all at interactive speeds, paving the way for fluid and intuitive 3D content authoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04542v1">Gaussian Entropy Fields: Driving Adaptive Sparsity in 3D Gaussian Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 28 pages,11 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a leading technique for novel view synthesis, demonstrating exceptional rendering efficiency. \replaced[]{Well-reconstructed surfaces can be characterized by low configurational entropy, where dominant primitives clearly define surface geometry while redundant components are suppressed.}{The key insight is that well-reconstructed surfaces naturally exhibit low configurational entropy, where dominant primitives clearly define surface geometry while suppressing redundant components.} Three complementary technical contributions are introduced: (1) entropy-driven surface modeling via entropy minimization for low configurational entropy in primitive distributions; (2) adaptive spatial regularization using the Surface Neighborhood Redundancy Index (SNRI) and image entropy-guided weighting; (3) multi-scale geometric preservation through competitive cross-scale entropy alignment. Extensive experiments demonstrate that GEF achieves competitive geometric precision on DTU and T\&T benchmarks, while delivering superior rendering quality compared to existing methods on Mip-NeRF 360. Notably, superior Chamfer Distance (0.64) on DTU and F1 score (0.44) on T\&T are obtained, alongside the best SSIM (0.855) and LPIPS (0.136) among baselines on Mip-NeRF 360, validating the framework's ability to enhance surface reconstruction accuracy without compromising photometric fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.05168v3">SeeLe: A Unified Acceleration Framework for Real-Time Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a crucial rendering technique for many real-time applications. However, the limited hardware resources on today's mobile platforms hinder these applications, as they struggle to achieve real-time performance. In this paper, we propose SeeLe, a general framework designed to accelerate the 3DGS pipeline for resource-constrained mobile devices. Specifically, we propose two GPU-oriented techniques: hybrid preprocessing and contribution-aware rasterization. Hybrid preprocessing alleviates the GPU compute and memory pressure by reducing the number of irrelevant Gaussians during rendering. The key is to combine our view-dependent scene representation with online filtering. Meanwhile, contribution-aware rasterization improves the GPU utilization at the rasterization stage by prioritizing Gaussians with high contributions while reducing computations for those with low contributions. Both techniques can be seamlessly integrated into existing 3DGS pipelines with minimal fine-tuning. Collectively, our framework achieves 2.6$\times$ speedup and 32.3\% model reduction while achieving superior rendering quality compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04421v1">UTrice: Unifying Primitives in Differentiable Ray Tracing and Rasterization via Triangles for Particle-Based 3D Scenes</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 13 pages, 10 figures, submitted to CVPR2026
    </div>
    <details class="paper-abstract">
      Ray tracing 3D Gaussian particles enables realistic effects such as depth of field, refractions, and flexible camera modeling for novel-view synthesis. However, existing methods trace Gaussians through proxy geometry, which requires constructing complex intermediate meshes and performing costly intersection tests. This limitation arises because Gaussian-based particles are not well suited as unified primitives for both ray tracing and rasterization. In this work, we propose a differentiable triangle-based ray tracing pipeline that directly treats triangles as rendering primitives without relying on any proxy geometry. Our results show that the proposed method achieves significantly higher rendering quality than existing ray tracing approaches while maintaining real-time rendering performance. Moreover, our pipeline can directly render triangles optimized by the rasterization-based method Triangle Splatting, thus unifying the primitives used in novel-view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05113v1">Splannequin: Freezing Monocular Mannequin-Challenge Footage with Dual-Detection Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 WACV 2025. Project page: https://chien90190.github.io/splannequin/
    </div>
    <details class="paper-abstract">
      Synthesizing high-fidelity frozen 3D scenes from monocular Mannequin-Challenge (MC) videos is a unique problem distinct from standard dynamic scene reconstruction. Instead of focusing on modeling motion, our goal is to create a frozen scene while strategically preserving subtle dynamics to enable user-controlled instant selection. To achieve this, we introduce a novel application of dynamic Gaussian splatting: the scene is modeled dynamically, which retains nearby temporal variation, and a static scene is rendered by fixing the model's time parameter. However, under this usage, monocular capture with sparse temporal supervision introduces artifacts like ghosting and blur for Gaussians that become unobserved or occluded at weakly supervised timestamps. We propose Splannequin, an architecture-agnostic regularization that detects two states of Gaussian primitives, hidden and defective, and applies temporal anchoring. Under predominantly forward camera motion, hidden states are anchored to their recent well-observed past states, while defective states are anchored to future states with stronger supervision. Our method integrates into existing dynamic Gaussian pipelines via simple loss terms, requires no architectural changes, and adds zero inference overhead. This results in markedly improved visual quality, enabling high-fidelity, user-selectable frozen-time renderings, validated by a 96% user preference. Project page: https://chien90190.github.io/splannequin/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05060v1">4DLangVGGT: 4D Language-Visual Geometry Grounded Transformer</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 Code: https://github.com/hustvl/4DLangVGGT, Webpage: https://hustvl.github.io/4DLangVGGT
    </div>
    <details class="paper-abstract">
      Constructing 4D language fields is crucial for embodied AI, augmented/virtual reality, and 4D scene understanding, as they provide enriched semantic representations of dynamic environments and enable open-vocabulary querying in complex scenarios. However, existing approaches to 4D semantic field construction primarily rely on scene-specific Gaussian splatting, which requires per-scene optimization, exhibits limited generalization, and is difficult to scale to real-world applications. To address these limitations, we propose 4DLangVGGT, the first Transformer-based feed-forward unified framework for 4D language grounding, that jointly integrates geometric perception and language alignment within a single architecture. 4DLangVGGT has two key components: the 4D Visual Geometry Transformer, StreamVGGT, which captures spatio-temporal geometric representations of dynamic scenes; and the Semantic Bridging Decoder (SBD), which projects geometry-aware features into a language-aligned semantic space, thereby enhancing semantic interpretability while preserving structural fidelity. Unlike prior methods that depend on costly per-scene optimization, 4DLangVGGT can be jointly trained across multiple dynamic scenes and directly applied during inference, achieving both deployment efficiency and strong generalization. This design significantly improves the practicality of large-scale deployment and establishes a new paradigm for open-vocabulary 4D scene understanding. Experiments on HyperNeRF and Neu3D datasets demonstrate that our approach not only generalizes effectively but also achieves state-of-the-art performance, achieving up to 2% gains under per-scene training and 1% improvements under multi-scene training. Our code released in https://github.com/hustvl/4DLangVGGT
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04815v1">RobustSplat++: Decoupling Densification, Dynamics, and Illumination for In-the-Wild 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 arXiv admin note: substantial text overlap with arXiv:2506.02751
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has gained significant attention for its real-time, photo-realistic rendering in novel-view synthesis and 3D modeling. However, existing methods struggle with accurately modeling in-the-wild scenes affected by transient objects and illuminations, leading to artifacts in the rendered images. We identify that the Gaussian densification process, while enhancing scene detail capture, unintentionally contributes to these artifacts by growing additional Gaussians that model transient disturbances and illumination variations. To address this, we propose RobustSplat++, a robust solution based on several critical designs. First, we introduce a delayed Gaussian growth strategy that prioritizes optimizing static scene structure before allowing Gaussian splitting/cloning, mitigating overfitting to transient objects in early optimization. Second, we design a scale-cascaded mask bootstrapping approach that first leverages lower-resolution feature similarity supervision for reliable initial transient mask estimation, taking advantage of its stronger semantic consistency and robustness to noise, and then progresses to high-resolution supervision to achieve more precise mask prediction. Third, we incorporate the delayed Gaussian growth strategy and mask bootstrapping with appearance modeling to handling in-the-wild scenes including transients and illuminations. Extensive experiments on multiple challenging datasets show that our method outperforms existing methods, clearly demonstrating the robustness and effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.13547v2">Turbo-GS: Accelerating 3D Gaussian Fitting for High-Quality Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 Project page: https://ivl.cs.brown.edu/research/turbo-gs
    </div>
    <details class="paper-abstract">
      Novel-view synthesis is an important problem in computer vision with applications in 3D reconstruction, mixed reality, and robotics. Recent methods like 3D Gaussian Splatting (3DGS) have become the preferred method for this task, providing high-quality novel views in real time. However, the training time of a 3DGS model is slow, often taking 30 minutes for a scene with 200 views. In contrast, our goal is to reduce the optimization time by training for fewer steps while maintaining high rendering quality. Specifically, we combine the guidance from both the position error and the appearance error to achieve a more effective densification. To balance the rate between adding new Gaussians and fitting old Gaussians, we develop a convergence-aware budget control mechanism. Moreover, to make the densification process more reliable, we selectively add new Gaussians from mostly visited regions. With these designs, we reduce the Gaussian optimization steps to one-third of the previous approach while achieving a comparable or even better novel view rendering quality. To further facilitate the rapid fitting of 4K resolution images, we introduce a dilation-based rendering technique. Our method, Turbo-GS, speeds up optimization for typical scenes and scales well to high-resolution (4K) scenarios on standard datasets. Through extensive experiments, we show that our method is significantly faster in optimization than other methods while retaining quality. Project page: https://ivl.cs.brown.edu/research/turbo-gs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04731v1">Bridging Simulation and Reality: Cross-Domain Transfer with Semantic 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Cross-domain transfer in robotic manipulation remains a longstanding challenge due to the significant domain gap between simulated and real-world environments. Existing methods such as domain randomization, adaptation, and sim-real calibration often require extensive tuning or fail to generalize to unseen scenarios. To address this issue, we observe that if domain-invariant features are utilized during policy training in simulation, and the same features can be extracted and provided as the input to policy during real-world deployment, the domain gap can be effectively bridged, leading to significantly improved policy generalization. Accordingly, we propose Semantic 2D Gaussian Splatting (S2GS), a novel representation method that extracts object-centric, domain-invariant spatial features. S2GS constructs multi-view 2D semantic fields and projects them into a unified 3D space via feature-level Gaussian splatting. A semantic filtering mechanism removes irrelevant background content, ensuring clean and consistent inputs for policy learning. To evaluate the effectiveness of S2GS, we adopt Diffusion Policy as the downstream learning algorithm and conduct experiments in the ManiSkill simulation environment, followed by real-world deployment. Results demonstrate that S2GS significantly improves sim-to-real transferability, maintaining high and stable task performance in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04315v1">SyncTrack4D: Cross-Video Motion Alignment and Video Synchronization for Multi-Video 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Modeling dynamic 3D scenes is challenging due to their high-dimensional nature, which requires aggregating information from multiple views to reconstruct time-evolving 3D geometry and motion. We present a novel multi-video 4D Gaussian Splatting (4DGS) approach designed to handle real-world, unsynchronized video sets. Our approach, SyncTrack4D, directly leverages dense 4D track representation of dynamic scene parts as cues for simultaneous cross-video synchronization and 4DGS reconstruction. We first compute dense per-video 4D feature tracks and cross-video track correspondences by Fused Gromov-Wasserstein optimal transport approach. Next, we perform global frame-level temporal alignment to maximize overlapping motion of matched 4D tracks. Finally, we achieve sub-frame synchronization through our multi-video 4D Gaussian splatting built upon a motion-spline scaffold representation. The final output is a synchronized 4DGS representation with dense, explicit 3D trajectories, and temporal offsets for each video. We evaluate our approach on the Panoptic Studio and SyncNeRF Blender, demonstrating sub-frame synchronization accuracy with an average temporal error below 0.26 frames, and high-fidelity 4D reconstruction reaching 26.3 PSNR scores on the Panoptic Studio dataset. To the best of our knowledge, our work is the first general 4D Gaussian Splatting approach for unsynchronized video sets, without assuming the existence of predefined scene objects or prior models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04313v1">Mind-to-Face: Neural-Driven Photorealistic Avatar Synthesis via EEG Decoding</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 16 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Current expressive avatar systems rely heavily on visual cues, failing when faces are occluded or when emotions remain internal. We present Mind-to-Face, the first framework that decodes non-invasive electroencephalogram (EEG) signals directly into high-fidelity facial expressions. We build a dual-modality recording setup to obtain synchronized EEG and multi-view facial video during emotion-eliciting stimuli, enabling precise supervision for neural-to-visual learning. Our model uses a CNN-Transformer encoder to map EEG signals into dense 3D position maps, capable of sampling over 65k vertices, capturing fine-scale geometry and subtle emotional dynamics, and renders them through a modified 3D Gaussian Splatting pipeline for photorealistic, view-consistent results. Through extensive evaluation, we show that EEG alone can reliably predict dynamic, subject-specific facial expressions, including subtle emotional responses, demonstrating that neural signals contain far richer affective and geometric information than previously assumed. Mind-to-Face establishes a new paradigm for neural-driven avatars, enabling personalized, emotion-aware telepresence and cognitive interaction in immersive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04021v1">C3G: Learning Compact 3D Representations with 2K Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 Project Page : https://cvlab-kaist.github.io/C3G/
    </div>
    <details class="paper-abstract">
      Reconstructing and understanding 3D scenes from unposed sparse views in a feed-forward manner remains as a challenging task in 3D computer vision. Recent approaches use per-pixel 3D Gaussian Splatting for reconstruction, followed by a 2D-to-3D feature lifting stage for scene understanding. However, they generate excessive redundant Gaussians, causing high memory overhead and sub-optimal multi-view feature aggregation, leading to degraded novel view synthesis and scene understanding performance. We propose C3G, a novel feed-forward framework that estimates compact 3D Gaussians only at essential spatial locations, minimizing redundancy while enabling effective feature lifting. We introduce learnable tokens that aggregate multi-view features through self-attention to guide Gaussian generation, ensuring each Gaussian integrates relevant visual features across views. We then exploit the learned attention patterns for Gaussian decoding to efficiently lift features. Extensive experiments on pose-free novel view synthesis, 3D open-vocabulary segmentation, and view-invariant feature aggregation demonstrate our approach's effectiveness. Results show that a compact yet geometrically meaningful representation is sufficient for high-quality scene reconstruction and understanding, achieving superior memory efficiency and feature fidelity compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13186v4">STT-GS: Sample-Then-Transmit Edge Gaussian Splatting with Joint Client Selection and Power Control</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Edge Gaussian splatting (EGS), which aggregates data from distributed clients (e.g., drones) and trains a global GS model at the edge (e.g., ground server), is an emerging paradigm for scene reconstruction in low-altitude economy. Unlike traditional edge resource management methods that emphasize communication throughput or general-purpose learning performance, EGS explicitly aims to maximize the GS qualities, rendering existing approaches inapplicable. To address this problem, this paper formulates a novel GS-oriented objective function that distinguishes the heterogeneous view contributions of different clients. However, evaluating this function in turn requires clients' images, leading to a causality dilemma. To this end, this paper further proposes a sample-then-transmit EGS (or STT-GS for short) strategy, which first samples a subset of images as pilot data from each client for loss prediction. Based on the first-stage evaluation, communication resources are then prioritized towards more valuable clients. To achieve efficient sampling, a feature-domain clustering (FDC) scheme is proposed to select the most representative data and pilot transmission time minimization (PTTM) is adopted to reduce the pilot overhead. Subsequently, we develop a joint client selection and power control (JCSPC) framework to maximize the GS-oriented function under communication resource constraints. Despite the nonconvexity of the problem, we propose a low-complexity efficient solution based on the penalty alternating majorization minimization (PAMM) algorithm. Experiments reveal that the proposed scheme significantly outperforms existing benchmarks on real-world datasets. The GS-oriented objective can be accurately predicted with low sampling ratios (e.g., 10%), and our method achieves an excellent tradeoff between view contributions and communication costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.15122v4">MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 This paper has been accepted to AAAI 2026. The first two authors contributed equally to this work (equal contribution). The last two authors are co-corresponding authors. Please visit our project page at https://kaist-viclab.github.io/mobgs-site/
    </div>
    <details class="paper-abstract">
      We present MoBGS, a novel motion deblurring 3D Gaussian Splatting (3DGS) framework capable of reconstructing sharp and high-quality novel spatio-temporal views from blurry monocular videos in an end-to-end manner. Existing dynamic novel view synthesis (NVS) methods are highly sensitive to motion blur in casually captured videos, resulting in significant degradation of rendering quality. While recent approaches address motion-blurred inputs for NVS, they primarily focus on static scene reconstruction and lack dedicated motion modeling for dynamic objects. To overcome these limitations, our MoBGS introduces a novel Blur-adaptive Latent Camera Estimation (BLCE) method using a proposed Blur-adaptive Neural Ordinary Differential Equation (ODE) solver for effective latent camera trajectory estimation, improving global camera motion deblurring. In addition, we propose a Latent Camera-induced Exposure Estimation (LCEE) method to ensure consistent deblurring of both a global camera and local object motions. Extensive experiments on the Stereo Blur dataset and real-world blurry videos show that our MoBGS significantly outperforms the very recent methods, achieving state-of-the-art performance for dynamic NVS under motion blur.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03601v1">Motion4D: Learning 3D-Consistent Motion and Semantics for 4D Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in foundation models for 2D vision have substantially improved the analysis of dynamic scenes from monocular videos. However, despite their strong generalization capabilities, these models often lack 3D consistency, a fundamental requirement for understanding scene geometry and motion, thereby causing severe spatial misalignment and temporal flickering in complex 3D environments. In this paper, we present Motion4D, a novel framework that addresses these challenges by integrating 2D priors from foundation models into a unified 4D Gaussian Splatting representation. Our method features a two-part iterative optimization framework: 1) Sequential optimization, which updates motion and semantic fields in consecutive stages to maintain local consistency, and 2) Global optimization, which jointly refines all attributes for long-term coherence. To enhance motion accuracy, we introduce a 3D confidence map that dynamically adjusts the motion priors, and an adaptive resampling process that inserts new Gaussians into under-represented regions based on per-pixel RGB and semantic errors. Furthermore, we enhance semantic coherence through an iterative refinement process that resolves semantic inconsistencies by alternately optimizing the semantic fields and updating prompts of SAM2. Extensive evaluations demonstrate that our Motion4D significantly outperforms both 2D foundation models and existing 3D-based approaches across diverse scene understanding tasks, including point-based tracking, video object segmentation, and novel view synthesis. Our code is available at https://hrzhou2.github.io/motion4d-web/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08710v2">SceneSplat++: A Large Dataset and Comprehensive Benchmark for Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 15 pages, codes, data and benchmark are released
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) serves as a highly performant and efficient encoding of scene geometry, appearance, and semantics. Moreover, grounding language in 3D scenes has proven to be an effective strategy for 3D scene understanding. Current Language Gaussian Splatting line of work fall into three main groups: (i) per-scene optimization-based, (ii) per-scene optimization-free, and (iii) generalizable approach. However, most of them are evaluated only on rendered 2D views of a handful of scenes and viewpoints close to the training views, limiting ability and insight into holistic 3D understanding. To address this gap, we propose the first large-scale benchmark that systematically assesses these three groups of methods directly in 3D space, evaluating on 1060 scenes across three indoor datasets and one outdoor dataset. Benchmark results demonstrate a clear advantage of the generalizable paradigm, particularly in relaxing the scene-specific limitation, enabling fast feed-forward inference on novel scenes, and achieving superior segmentation performance. We further introduce GaussianWorld-49K a carefully curated 3DGS dataset comprising around 49K diverse indoor and outdoor scenes obtained from multiple sources, with which we demonstrate the generalizable approach could harness strong data priors. Our codes, benchmark, and datasets are public to accelerate research in generalizable 3DGS scene understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.06859v2">ActiveInitSplat: How Active Image Selection Helps Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Gaussian splatting (GS) along with its extensions and variants provides outstanding performance in real-time scene rendering while meeting reduced storage demands and computational efficiency. While the selection of 2D images capturing the scene of interest is crucial for the proper initialization and training of GS, hence markedly affecting the rendering performance, prior works rely on passively and typically densely selected 2D images. In contrast, this paper proposes `ActiveInitSplat', a novel framework for active selection of training images for proper initialization and training of GS. ActiveInitSplat relies on density and occupancy criteria of the resultant 3D scene representation from the selected 2D images, to ensure that the latter are captured from diverse viewpoints leading to better scene coverage and that the initialized Gaussian functions are well aligned with the actual 3D structure. Numerical tests on well-known simulated and real environments demonstrate the merits of ActiveInitSplat resulting in significant GS rendering performance improvement over passive GS baselines in both dense- and sparse-view settings, in the widely adopted LPIPS, SSIM, and PSNR metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06517v3">GS4: Generalizable Sparse Splatting Semantic SLAM</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Traditional SLAM algorithms excel at camera tracking, but typically produce incomplete and low-resolution maps that are not tightly integrated with semantics prediction. Recent work integrates Gaussian Splatting (GS) into SLAM to enable dense, photorealistic 3D mapping, yet existing GS-based SLAM methods require per-scene optimization that is slow and consumes an excessive number of Gaussians. We present GS4, the first generalizable GS-based semantic SLAM system. Compared with prior approaches, GS4 runs 10x faster, uses 10x fewer Gaussians, and achieves state-of-the-art performance across color, depth, semantic mapping and camera tracking. From an RGB-D video stream, GS4 incrementally builds and updates a set of 3D Gaussians using a feed-forward network. First, the Gaussian Prediction Model estimates a sparse set of Gaussian parameters from input frame, which integrates both color and semantic prediction with the same backbone. Then, the Gaussian Refinement Network merges new Gaussians with the existing set while avoiding redundancy. Finally, when significant pose changes are detected, we perform only 1-5 iterations of joint Gaussian-pose optimization to correct drift, remove floaters, and further improve tracking accuracy. Experiments on the real-world ScanNet and ScanNet++ benchmarks demonstrate state-of-the-art semantic SLAM performance, with strong generalization capability shown through zero-shot transfer to the NYUv2 and TUM RGB-D datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03422v1">What Is The Best 3D Scene Representation for Robotics? From Geometric to Foundation Models</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      In this paper, we provide a comprehensive overview of existing scene representation methods for robotics, covering traditional representations such as point clouds, voxels, signed distance functions (SDF), and scene graphs, as well as more recent neural representations like Neural Radiance Fields (NeRF), 3D Gaussian Splatting (3DGS), and the emerging Foundation Models. While current SLAM and localization systems predominantly rely on sparse representations like point clouds and voxels, dense scene representations are expected to play a critical role in downstream tasks such as navigation and obstacle avoidance. Moreover, neural representations such as NeRF, 3DGS, and foundation models are well-suited for integrating high-level semantic features and language-based priors, enabling more comprehensive 3D scene understanding and embodied intelligence. In this paper, we categorized the core modules of robotics into five parts (Perception, Mapping, Localization, Navigation, Manipulation). We start by presenting the standard formulation of different scene representation methods and comparing the advantages and disadvantages of scene representation across different modules. This survey is centered around the question: What is the best 3D scene representation for robotics? We then discuss the future development trends of 3D scene representations, with a particular focus on how the 3D Foundation Model could replace current methods as the unified solution for future robotic applications. The remaining challenges in fully realizing this model are also explored. We aim to offer a valuable resource for both newcomers and experienced researchers to explore the future of 3D scene representations and their application in robotics. We have published an open-source project on GitHub and will continue to add new works and technologies to this project.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08136v2">FantasyStyle: Controllable Stylized Distillation for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      The success of 3DGS in generative and editing applications has sparked growing interest in 3DGS-based style transfer. However, current methods still face two major challenges: (1) multi-view inconsistency often leads to style conflicts, resulting in appearance smoothing and distortion; and (2) heavy reliance on VGG features, which struggle to disentangle style and content from style images, often causing content leakage and excessive stylization. To tackle these issues, we introduce \textbf{FantasyStyle}, a 3DGS-based style transfer framework, and the first to rely entirely on diffusion model distillation. It comprises two key components: (1) \textbf{Multi-View Frequency Consistency}. We enhance cross-view consistency by applying a 3D filter to multi-view noisy latent, selectively reducing low-frequency components to mitigate stylized prior conflicts. (2) \textbf{Controllable Stylized Distillation}. To suppress content leakage from style images, we introduce negative guidance to exclude undesired content. In addition, we identify the limitations of Score Distillation Sampling and Delta Denoising Score in 3D style transfer and remove the reconstruction term accordingly. Building on these insights, we propose a controllable stylized distillation that leverages negative guidance to more effectively optimize the 3D Gaussians. Extensive experiments demonstrate that our method consistently outperforms state-of-the-art approaches, achieving higher stylization quality and visual realism across various scenes and styles. The code is available at https://github.com/yangyt46/FantasyStyle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02621v1">Content-Aware Texturing for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 Project Page: https://repo-sam.inria.fr/nerphys/gs-texturing/
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has become the method of choice for 3D reconstruction and real-time rendering of captured real scenes. However, fine appearance details need to be represented as a large number of small Gaussian primitives, which can be wasteful when geometry and appearance exhibit different frequency characteristics. Inspired by the long tradition of texture mapping, we propose to use texture to represent detailed appearance where possible. Our main focus is to incorporate per-primitive texture maps that adapt to the scene in a principled manner during Gaussian Splatting optimization. We do this by proposing a new appearance representation for 2D Gaussian primitives with textures where the size of a texel is bounded by the image sampling frequency and adapted to the content of the input images. We achieve this by adaptively upscaling or downscaling the texture resolution during optimization. In addition, our approach enables control of the number of primitives during optimization based on texture resolution. We show that our approach performs favorably in image quality and total number of parameters used compared to alternative solutions for textured Gaussian primitives. Project page: https://repo-sam.inria.fr/nerphys/gs-texturing/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02482v1">G-SHARP: Gaussian Surgical Hardware Accelerated Real-time Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      We propose G-SHARP, a commercially compatible, real-time surgical scene reconstruction framework designed for minimally invasive procedures that require fast and accurate 3D modeling of deformable tissue. While recent Gaussian splatting approaches have advanced real-time endoscopic reconstruction, existing implementations often depend on non-commercial derivatives, limiting deployability. G-SHARP overcomes these constraints by being the first surgical pipeline built natively on the GSplat (Apache-2.0) differentiable Gaussian rasterizer, enabling principled deformation modeling, robust occlusion handling, and high-fidelity reconstructions on the EndoNeRF pulling benchmark. Our results demonstrate state-of-the-art reconstruction quality with strong speed-accuracy trade-offs suitable for intra-operative use. Finally, we provide a Holoscan SDK application that deploys G-SHARP on NVIDIA IGX Orin and Thor edge hardware, enabling real-time surgical visualization in practical operating-room settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.26219v2">Beyond Pixels: Efficient Dataset Distillation via Sparse Gaussian Representation</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 19 pages; Code is available on https://github.com/j-cyoung/GSDatasetDistillation
    </div>
    <details class="paper-abstract">
      Dataset distillation has emerged as a promising paradigm that synthesizes compact, informative datasets capable of retaining the knowledge of large-scale counterparts, thereby addressing the substantial computational and storage burdens of modern model training. Conventional approaches typically rely on dense pixel-level representations, which introduce redundancy and are difficult to scale up. In this work, we propose GSDD, a novel and efficient sparse representation for dataset distillation based on 2D Gaussians. Instead of representing all pixels equally, GSDD encodes critical discriminative information in a distilled image using only a small number of Gaussian primitives. This sparse representation could improve dataset diversity under the same storage budget, enhancing coverage of difficult samples and boosting distillation performance. To ensure both efficiency and scalability, we adapt CUDA-based splatting operators for parallel inference and training, enabling high-quality rendering with minimal computational and memory overhead. Our method is simple yet effective, broadly applicable to different distillation pipelines, and highly scalable. Experiments show that GSDD achieves state-of-the-art performance on CIFAR-10, CIFAR-100, and ImageNet subsets, while remaining highly efficient encoding and decoding cost. Our code is available at https://github.com/j-cyoung/GSDatasetDistillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02293v1">VIGS-SLAM: Visual Inertial Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 Project page: https://vigs-slam.github.io
    </div>
    <details class="paper-abstract">
      We present VIGS-SLAM, a visual-inertial 3D Gaussian Splatting SLAM system that achieves robust real-time tracking and high-fidelity reconstruction. Although recent 3DGS-based SLAM methods achieve dense and photorealistic mapping, their purely visual design degrades under motion blur, low texture, and exposure variations. Our method tightly couples visual and inertial cues within a unified optimization framework, jointly refining camera poses, depths, and IMU states. It features robust IMU initialization, time-varying bias modeling, and loop closure with consistent Gaussian updates. Experiments on four challenging datasets demonstrate our superiority over state-of-the-art methods. Project page: https://vigs-slam.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03210v1">Flux4D: Flow-based Unsupervised 4D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 NeurIPS 2025. Project page: https://waabi.ai/flux4d/
    </div>
    <details class="paper-abstract">
      Reconstructing large-scale dynamic scenes from visual observations is a fundamental challenge in computer vision, with critical implications for robotics and autonomous systems. While recent differentiable rendering methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have achieved impressive photorealistic reconstruction, they suffer from scalability limitations and require annotations to decouple actor motion. Existing self-supervised methods attempt to eliminate explicit annotations by leveraging motion cues and geometric priors, yet they remain constrained by per-scene optimization and sensitivity to hyperparameter tuning. In this paper, we introduce Flux4D, a simple and scalable framework for 4D reconstruction of large-scale dynamic scenes. Flux4D directly predicts 3D Gaussians and their motion dynamics to reconstruct sensor observations in a fully unsupervised manner. By adopting only photometric losses and enforcing an "as static as possible" regularization, Flux4D learns to decompose dynamic elements directly from raw data without requiring pre-trained supervised models or foundational priors simply by training across many scenes. Our approach enables efficient reconstruction of dynamic scenes within seconds, scales effectively to large datasets, and generalizes well to unseen environments, including rare and unknown objects. Experiments on outdoor driving datasets show Flux4D significantly outperforms existing methods in scalability, generalization, and reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03010v1">SurfFill: Completion of LiDAR Point Clouds via Gaussian Surfel Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 Project page: https://lfranke.github.io/surffill
    </div>
    <details class="paper-abstract">
      LiDAR-captured point clouds are often considered the gold standard in active 3D reconstruction. While their accuracy is exceptional in flat regions, the capturing is susceptible to miss small geometric structures and may fail with dark, absorbent materials. Alternatively, capturing multiple photos of the scene and applying 3D photogrammetry can infer these details as they often represent feature-rich regions. However, the accuracy of LiDAR for featureless regions is rarely reached. Therefore, we suggest combining the strengths of LiDAR and camera-based capture by introducing SurfFill: a Gaussian surfel-based LiDAR completion scheme. We analyze LiDAR capturings and attribute LiDAR beam divergence as a main factor for artifacts, manifesting mostly at thin structures and edges. We use this insight to introduce an ambiguity heuristic for completed scans by evaluating the change in density in the point cloud. This allows us to identify points close to missed areas, which we can then use to grow additional points from to complete the scan. For this point growing, we constrain Gaussian surfel reconstruction [Huang et al. 2024] to focus optimization and densification on these ambiguous areas. Finally, Gaussian primitives of the reconstruction in ambiguous areas are extracted and sampled for points to complete the point cloud. To address the challenges of large-scale reconstruction, we extend this pipeline with a divide-and-conquer scheme for building-sized point cloud completion. We evaluate on the task of LiDAR point cloud completion of synthetic and real-world scenes and find that our method outperforms previous reconstruction methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02932v1">EGGS: Exchangeable 2D/3D Gaussian Splatting for Geometry-Appearance Balanced Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) is crucial in computer vision and graphics, with wide applications in AR, VR, and autonomous driving. While 3D Gaussian Splatting (3DGS) enables real-time rendering with high appearance fidelity, it suffers from multi-view inconsistencies, limiting geometric accuracy. In contrast, 2D Gaussian Splatting (2DGS) enforces multi-view consistency but compromises texture details. To address these limitations, we propose Exchangeable Gaussian Splatting (EGGS), a hybrid representation that integrates 2D and 3D Gaussians to balance appearance and geometry. To achieve this, we introduce Hybrid Gaussian Rasterization for unified rendering, Adaptive Type Exchange for dynamic adaptation between 2D and 3D Gaussians, and Frequency-Decoupled Optimization that effectively exploits the strengths of each type of Gaussian representation. Our CUDA-accelerated implementation ensures efficient training and inference. Extensive experiments demonstrate that EGGS outperforms existing methods in rendering quality, geometric accuracy, and efficiency, providing a practical solution for high-quality NVS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.03659v6">DehazeGS: Seeing Through Fog with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 9 pages,5 figures. Accepted by AAAI2026. visualizations are available at https://dehazegs.github.io/
    </div>
    <details class="paper-abstract">
      Current novel view synthesis methods are typically designed for high-quality and clean input images. However, in foggy scenes, scattering and attenuation can significantly degrade the quality of rendering. Although NeRF-based dehazing approaches have been developed, their reliance on deep fully connected neural networks and per-ray sampling strategies leads to high computational costs. Furthermore, NeRF's implicit representation limits its ability to recover fine-grained details from hazy scenes. To overcome these limitations, we propose learning an explicit Gaussian representation to explain the formation mechanism of foggy images through a physically forward rendering process. Our method, DehazeGS, reconstructs and renders fog-free scenes using only multi-view foggy images as input. Specifically, based on the atmospheric scattering model, we simulate the formation of fog by establishing the transmission function directly onto Gaussian primitives via depth-to-transmission mapping. During training, we jointly learn the atmospheric light and scattering coefficients while optimizing the Gaussian representation of foggy scenes. At inference time, we remove the effects of scattering and attenuation in Gaussian distributions and directly render the scene to obtain dehazed views. Experiments on both real-world and synthetic foggy datasets demonstrate that DehazeGS achieves state-of-the-art performance. visualizations are available at https://dehazegs.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08294v2">SkelSplat: Robust Multi-view 3D Human Pose Estimation with Differentiable Gaussian Rendering</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 WACV 2026
    </div>
    <details class="paper-abstract">
      Accurate 3D human pose estimation is fundamental for applications such as augmented reality and human-robot interaction. State-of-the-art multi-view methods learn to fuse predictions across views by training on large annotated datasets, leading to poor generalization when the test scenario differs. To overcome these limitations, we propose SkelSplat, a novel framework for multi-view 3D human pose estimation based on differentiable Gaussian rendering. Human pose is modeled as a skeleton of 3D Gaussians, one per joint, optimized via differentiable rendering to enable seamless fusion of arbitrary camera views without 3D ground-truth supervision. Since Gaussian Splatting was originally designed for dense scene reconstruction, we propose a novel one-hot encoding scheme that enables independent optimization of human joints. SkelSplat outperforms approaches that do not rely on 3D ground truth in Human3.6M and CMU, while reducing the cross-dataset error up to 47.8% compared to learning-based methods. Experiments on Human3.6M-Occ and Occlusion-Person demonstrate robustness to occlusions, without scenario-specific fine-tuning. Our project page is available here: https://skelsplat.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02664v1">PolarGuide-GSDR: 3D Gaussian Splatting Driven by Polarization Priors and Deferred Reflection for Real-World Reflective Scenes</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Polarization-aware Neural Radiance Fields (NeRF) enable novel view synthesis of specular-reflection scenes but face challenges in slow training, inefficient rendering, and strong dependencies on material/viewpoint assumptions. However, 3D Gaussian Splatting (3DGS) enables real-time rendering yet struggles with accurate reflection reconstruction from reflection-geometry entanglement, adding a deferred reflection module introduces environment map dependence. We address these limitations by proposing PolarGuide-GSDR, a polarization-forward-guided paradigm establishing a bidirectional coupling mechanism between polarization and 3DGS: first 3DGS's geometric priors are leveraged to resolve polarization ambiguity, and then the refined polarization information cues are used to guide 3DGS's normal and spherical harmonic representation. This process achieves high-fidelity reflection separation and full-scene reconstruction without requiring environment maps or restrictive material assumptions. We demonstrate on public and self-collected datasets that PolarGuide-GSDR achieves state-of-the-art performance in specular reconstruction, normal estimation, and novel view synthesis, all while maintaining real-time rendering capabilities. To our knowledge, this is the first framework embedding polarization priors directly into 3DGS optimization, yielding superior interpretability and real-time performance for modeling complex reflective scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02648v1">PoreTrack3D: A Benchmark for Dynamic 3D Gaussian Splatting in Pore-Scale Facial Trajectory Tracking</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      We introduce PoreTrack3D, the first benchmark for dynamic 3D Gaussian splatting in pore-scale, non-rigid 3D facial trajectory tracking. It contains over 440,000 facial trajectories in total, among which more than 52,000 are longer than 10 frames, including 68 manually reviewed trajectories that span the entire 150 frames. To the best of our knowledge, PoreTrack3D is the first benchmark dataset to capture both traditional facial landmarks and pore-scale keypoints trajectory, advancing the study of fine-grained facial expressions through the analysis of subtle skin-surface motion. We systematically evaluate state-of-the-art dynamic 3D Gaussian splatting methods on PoreTrack3D, establishing the first performance baseline in this domain. Overall, the pipeline developed for this benchmark dataset's creation establishes a new framework for high-fidelity facial motion capture and dynamic 3D reconstruction. Our dataset are publicly available at: https://github.com/JHXion9/PoreTrack3D
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02443v2">PRIMU: Uncertainty Estimation for Novel Views in Gaussian Splatting from Primitive-Based Representations of Error and Coverage</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 Revised writing and figures; additional Gaussian Splatting experiments; added baselines and datasets; active view-selection experiments
    </div>
    <details class="paper-abstract">
      We introduce Primitive-based Representations of Uncertainty (PRIMU), a post-hoc uncertainty estimation (UE) framework for Gaussian Splatting (GS). Reliable UE is essential for deploying GS in safety-critical domains such as robotics and medicine. Existing approaches typically estimate Gaussian-primitive variances and rely on the rendering process to obtain pixel-wise uncertainties. In contrast, we construct primitive-level representations of error and visibility/coverage from training views, capturing interpretable uncertainty information. These representations are obtained by projecting view-dependent training errors and coverage statistics onto the primitives. Uncertainties for novel views are inferred by rendering these primitive-level representations, producing uncertainty feature maps, which are aggregate through pixel-wise regression on holdout data. We analyze combinations of uncertainty feature maps and regression models to understand how their interactions affect prediction accuracy and generalization. PRIMU also enables an effective active view selection strategy by directly leveraging these uncertainty feature maps. Additionally, we study the effect of separating splatting into foreground and background regions. Our estimates show strong correlations with true errors, outperforming state-of-the-art methods, especially for depth UE and foreground objects. Finally, our regression models show generalization capabilities to unseen scenes, enabling UE without additional holdout data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15719v1">A Fast Volumetric Capture and Reconstruction Pipeline for Dynamic Point Clouds and Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 ACM SIGGRAPH European Conference on Visual Media Production (CVMP) 2025. Code available at: https://github.com/irc-hslu/capturestudio
    </div>
    <details class="paper-abstract">
      We present a fast and efficient volumetric capture and reconstruction system that processes either RGB-D or RGB-only input to generate 3D representations in the form of point clouds and Gaussian splats. For Gaussian splat reconstructions, we took the GPS-Gaussian regressor and improved it, enabling high-quality reconstructions with minimal overhead. The system is designed for easy setup and deployment, supporting in-the-wild operation under uncontrolled illumination and arbitrary backgrounds, as well as flexible camera configurations, including sparse setups, arbitrary camera numbers and baselines. Captured data can be exported in standard formats such as PLY, MPEG V-PCC, and SPLAT, and visualized through a web-based viewer or Unity/Unreal plugins. A live on-location preview of both input and reconstruction is available at 5-10 FPS. We present qualitative findings focused on deployability and targeted ablations. The complete framework is open-source, facilitating reproducibility and further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01296v1">EGG-Fusion: Efficient 3D Reconstruction with Geometry-aware Gaussian Surfel on the Fly</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 SIGGRAPH ASIA 2025
    </div>
    <details class="paper-abstract">
      Real-time 3D reconstruction is a fundamental task in computer graphics. Recently, differentiable-rendering-based SLAM system has demonstrated significant potential, enabling photorealistic scene rendering through learnable scene representations such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). Current differentiable rendering methods face dual challenges in real-time computation and sensor noise sensitivity, leading to degraded geometric fidelity in scene reconstruction and limited practicality. To address these challenges, we propose a novel real-time system EGG-Fusion, featuring robust sparse-to-dense camera tracking and a geometry-aware Gaussian surfel mapping module, introducing an information filter-based fusion method that explicitly accounts for sensor noise to achieve high-precision surface reconstruction. The proposed differentiable Gaussian surfel mapping effectively models multi-view consistent surfaces while enabling efficient parameter optimization. Extensive experimental results demonstrate that the proposed system achieves a surface reconstruction error of 0.6\textit{cm} on standardized benchmark datasets including Replica and ScanNet++, representing over 20\% improvement in accuracy compared to state-of-the-art (SOTA) GS-based methods. Notably, the system maintains real-time processing capabilities at 24 FPS, establishing it as one of the most accurate differentiable-rendering-based real-time reconstruction systems. Project Page: https://zju3dv.github.io/eggfusion/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.01846v4">UVGS: Reimagining Unstructured 3D Gaussian Splatting using UV Mapping</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 https://ivl.cs.brown.edu/uvgs
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated superior quality in modeling 3D objects and scenes. However, generating 3DGS remains challenging due to their discrete, unstructured, and permutation-invariant nature. In this work, we present a simple yet effective method to overcome these challenges. We utilize spherical mapping to transform 3DGS into a structured 2D representation, termed UVGS. UVGS can be viewed as multi-channel images, with feature dimensions as a concatenation of Gaussian attributes such as position, scale, color, opacity, and rotation. We further find that these heterogeneous features can be compressed into a lower-dimensional (e.g., 3-channel) shared feature space using a carefully designed multi-branch network. The compressed UVGS can be treated as typical RGB images. Remarkably, we discover that typical VAEs trained with latent diffusion models can directly generalize to this new representation without additional training. Our novel representation makes it effortless to leverage foundational 2D models, such as diffusion models, to directly model 3DGS. Additionally, one can simply increase the 2D UV resolution to accommodate more Gaussians, making UVGS a scalable solution compared to typical 3D backbones. This approach immediately unlocks various novel generation applications of 3DGS by inherently utilizing the already developed superior 2D generation capabilities. In our experiments, we demonstrate various unconditional, conditional generation, and inpainting applications of 3DGS based on diffusion models, which were previously non-trivial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02172v1">SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 Project Page: https://splatsure.github.io/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables high-quality novel view synthesis, motivating interest in generating higher-resolution renders than those available during training. A natural strategy is to apply super-resolution (SR) to low-resolution (LR) input views, but independently enhancing each image introduces multi-view inconsistencies, leading to blurry renders. Prior methods attempt to mitigate these inconsistencies through learned neural components, temporally consistent video priors, or joint optimization on LR and SR views, but all uniformly apply SR across every image. In contrast, our key insight is that close-up LR views may contain high-frequency information for regions also captured in more distant views, and that we can use the camera pose relative to scene geometry to inform where to add SR content. Building from this insight, we propose SplatSuRe, a method that selectively applies SR content only in undersampled regions lacking high-frequency supervision, yielding sharper and more consistent results. Across Tanks & Temples, Deep Blending and Mip-NeRF 360, our approach surpasses baselines in both fidelity and perceptual quality. Notably, our gains are most significant in localized foreground regions where higher detail is desired.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02013v1">ManualVLA: A Unified VLA Model for Chain-of-Thought Manual Generation and Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models have recently emerged, demonstrating strong generalization in robotic scene understanding and manipulation. However, when confronted with long-horizon tasks that require defined goal states, such as LEGO assembly or object rearrangement, existing VLA models still face challenges in coordinating high-level planning with precise manipulation. Therefore, we aim to endow a VLA model with the capability to infer the "how" process from the "what" outcomes, transforming goal states into executable procedures. In this paper, we introduce ManualVLA, a unified VLA framework built upon a Mixture-of-Transformers (MoT) architecture, enabling coherent collaboration between multimodal manual generation and action execution. Unlike prior VLA models that directly map sensory inputs to actions, we first equip ManualVLA with a planning expert that generates intermediate manuals consisting of images, position prompts, and textual instructions. Building upon these multimodal manuals, we design a Manual Chain-of-Thought (ManualCoT) reasoning process that feeds them into the action expert, where each manual step provides explicit control conditions, while its latent representation offers implicit guidance for accurate manipulation. To alleviate the burden of data collection, we develop a high-fidelity digital-twin toolkit based on 3D Gaussian Splatting, which automatically generates manual data for planning expert training. ManualVLA demonstrates strong real-world performance, achieving an average success rate 32% higher than the previous hierarchical SOTA baseline on LEGO assembly and object rearrangement tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.12168v3">Sketch-guided Cage-based 3D Gaussian Splatting Deformation</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 10 pages, 9 figures, accepted at WACV 26, project page: https://tianhaoxie.github.io/project/gs_deform/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (GS) is one of the most promising novel 3D representations that has received great interest in computer graphics and computer vision. While various systems have introduced editing capabilities for 3D GS, such as those guided by text prompts, fine-grained control over deformation remains an open challenge. In this work, we present a novel sketch-guided 3D GS deformation system that allows users to intuitively modify the geometry of a 3D GS model by drawing a silhouette sketch from a single viewpoint. Our approach introduces a new deformation method that combines cage-based deformations with a variant of Neural Jacobian Fields, enabling precise, fine-grained control. Additionally, it leverages large-scale 2D diffusion priors and ControlNet to ensure the generated deformations are semantically plausible. Through a series of experiments, we demonstrate the effectiveness of our method and showcase its ability to animate static 3D GS models as one of its key applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22939v2">DenoiseGS: Gaussian Reconstruction Model for Burst Denoising</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 Update Abstract
    </div>
    <details class="paper-abstract">
      Burst denoising methods are crucial for enhancing images captured on handheld devices, but they often struggle with large motion or suffer from prohibitive computational costs. In this paper, we propose DenoiseGS, the first framework to leverage the efficiency of 3D Gaussian Splatting for burst denoising. Our approach addresses two key challenges when applying feedforward Gaussian reconsturction model to noisy inputs: the degradation of Gaussian point clouds and the loss of fine details. To this end, we propose a Gaussian self-consistency (GSC) loss, which regularizes the geometry predicted from noisy inputs with high-quality Gaussian point clouds. These point clouds are generated from clean inputs by the same model that we are training, thereby alleviating potential bias or domain gaps. Additionally, we introduce a log-weighted frequency (LWF) loss to strengthen supervision within the spectral domain, effectively preserving fine-grained details. The LWF loss adaptively weights frequency discrepancies in a logarithmic manner, emphasizing challenging high-frequency details. Extensive experiments demonstrate that DenoiseGS significantly exceeds the state-of-the-art NeRF-based methods on both burst denoising and novel view synthesis under noisy conditions, while achieving 250$\times$ faster inference speed. Code and models are released at https://github.com/yscheng04/DenoiseGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22056v2">EAST: Environment-Aware Stylized Transition Along the Reality-Virtuality Continuum</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      In the Virtual Reality (VR) gaming industry, maintaining immersion during real-world interruptions remains a challenge, particularly during transitions along the reality-virtuality continuum (RVC). Existing methods tend to rely on digital replicas or simple visual transitions, neglecting to address the aesthetic discontinuities between real and virtual environments, especially in highly stylized VR games. This paper introduces the Environment-Aware Stylized Transition (EAST) framework, which employs a novel style-transferred 3D Gaussian Splatting (3DGS) technique to transfer real-world interruptions into the virtual environment with seamless aesthetic consistency. Rather than merely transforming the real world into game-like visuals, EAST minimizes the disruptive impact of interruptions by integrating real-world elements within the framework. Qualitative user studies demonstrate significant enhancements in cognitive comfort and emotional continuity during transitions, while quantitative experiments highlight EAST's ability to maintain visual coherence across diverse VR styles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01329v1">TagSplat: Topology-Aware Gaussian Splatting for Dynamic Mesh Modeling and Tracking</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Topology-consistent dynamic model sequences are essential for applications such as animation and model editing. However, existing 4D reconstruction methods face challenges in generating high-quality topology-consistent meshes. To address this, we propose a topology-aware dynamic reconstruction framework based on Gaussian Splatting. We introduce a Gaussian topological structure that explicitly encodes spatial connectivity. This structure enables topology-aware densification and pruning, preserving the manifold consistency of the Gaussian representation. Temporal regularization terms further ensure topological coherence over time, while differentiable mesh rasterization improves mesh quality. Experimental results demonstrate that our method reconstructs topology-consistent mesh sequences with significantly higher accuracy than existing approaches. Moreover, the resulting meshes enable precise 3D keypoint tracking. Project page: https://haza628.github.io/tagSplat/
    </details>
</div>
