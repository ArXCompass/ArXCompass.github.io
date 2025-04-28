# gaussian splatting - 2025_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18468v1">RGS-DR: Reflective Gaussian Surfels with Deferred Rendering for Shiny Objects</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      We introduce RGS-DR, a novel inverse rendering method for reconstructing and rendering glossy and reflective objects with support for flexible relighting and scene editing. Unlike existing methods (e.g., NeRF and 3D Gaussian Splatting), which struggle with view-dependent effects, RGS-DR utilizes a 2D Gaussian surfel representation to accurately estimate geometry and surface normals, an essential property for high-quality inverse rendering. Our approach explicitly models geometric and material properties through learnable primitives rasterized into a deferred shading pipeline, effectively reducing rendering artifacts and preserving sharp reflections. By employing a multi-level cube mipmap, RGS-DR accurately approximates environment lighting integrals, facilitating high-quality reconstruction and relighting. A residual pass with spherical-mipmap-based directional encoding further refines the appearance modeling. Experiments demonstrate that RGS-DR achieves high-quality reconstruction and rendering quality for shiny objects, often outperforming reconstruction-exclusive state-of-the-art methods incapable of relighting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16915v2">Let's Make a Splan: Risk-Aware Trajectory Optimization in a Normalized Gaussian Splat</a></div>
    <div class="paper-meta">
      📅 2025-04-25
      | 💬 First two authors contributed equally. Project Page: https://roahmlab.github.io/splanning
    </div>
    <details class="paper-abstract">
      Neural Radiance Fields and Gaussian Splatting have recently transformed computer vision by enabling photo-realistic representations of complex scenes. However, they have seen limited application in real-world robotics tasks such as trajectory optimization. This is due to the difficulty in reasoning about collisions in radiance models and the computational complexity associated with operating in dense models. This paper addresses these challenges by proposing SPLANNING, a risk-aware trajectory optimizer operating in a Gaussian Splatting model. This paper first derives a method to rigorously upper-bound the probability of collision between a robot and a radiance field. Then, this paper introduces a normalized reformulation of Gaussian Splatting that enables efficient computation of this collision bound. Finally, this paper presents a method to optimize trajectories that avoid collisions in a Gaussian Splat. Experiments show that SPLANNING outperforms state-of-the-art methods in generating collision-free trajectories in cluttered environments. The proposed system is also tested on a real-world robot manipulator. A project page is available at https://roahmlab.github.io/splanning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18318v1">STP4D: Spatio-Temporal-Prompt Consistent Modeling for Text-to-4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      Text-to-4D generation is rapidly developing and widely applied in various scenarios. However, existing methods often fail to incorporate adequate spatio-temporal modeling and prompt alignment within a unified framework, resulting in temporal inconsistencies, geometric distortions, or low-quality 4D content that deviates from the provided texts. Therefore, we propose STP4D, a novel approach that aims to integrate comprehensive spatio-temporal-prompt consistency modeling for high-quality text-to-4D generation. Specifically, STP4D employs three carefully designed modules: Time-varying Prompt Embedding, Geometric Information Enhancement, and Temporal Extension Deformation, which collaborate to accomplish this goal. Furthermore, STP4D is among the first methods to exploit the Diffusion model to generate 4D Gaussians, combining the fine-grained modeling capabilities and the real-time rendering process of 4DGS with the rapid inference speed of the Diffusion model. Extensive experiments demonstrate that STP4D excels in generating high-fidelity 4D content with exceptional efficiency (approximately 4.6s per asset), surpassing existing methods in both quality and speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18165v1">PerfCam: Digital Twinning for Production Lines Using 3D Gaussian Splatting and Vision Models</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      We introduce PerfCam, an open source Proof-of-Concept (PoC) digital twinning framework that combines camera and sensory data with 3D Gaussian Splatting and computer vision models for digital twinning, object tracking, and Key Performance Indicators (KPIs) extraction in industrial production lines. By utilizing 3D reconstruction and Convolutional Neural Networks (CNNs), PerfCam offers a semi-automated approach to object tracking and spatial mapping, enabling digital twins that capture real-time KPIs such as availability, performance, Overall Equipment Effectiveness (OEE), and rate of conveyor belts in the production line. We validate the effectiveness of PerfCam through a practical deployment within realistic test production lines in the pharmaceutical industry and contribute an openly published dataset to support further research and development in the field. The results demonstrate PerfCam's ability to deliver actionable insights through its precise digital twin capabilities, underscoring its value as an effective tool for developing usable digital twins in smart manufacturing environments and extracting operational analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17728v1">CasualHDRSplat: Robust High Dynamic Range 3D Gaussian Splatting from Casually Captured Videos</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 Source Code: https://github.com/WU-CVGL/CasualHDRSplat
    </div>
    <details class="paper-abstract">
      Recently, photo-realistic novel view synthesis from multi-view images, such as neural radiance field (NeRF) and 3D Gaussian Splatting (3DGS), have garnered widespread attention due to their superior performance. However, most works rely on low dynamic range (LDR) images, which limits the capturing of richer scene details. Some prior works have focused on high dynamic range (HDR) scene reconstruction, typically require capturing of multi-view sharp images with different exposure times at fixed camera positions during exposure times, which is time-consuming and challenging in practice. For a more flexible data acquisition, we propose a one-stage method: \textbf{CasualHDRSplat} to easily and robustly reconstruct the 3D HDR scene from casually captured videos with auto-exposure enabled, even in the presence of severe motion blur and varying unknown exposure time. \textbf{CasualHDRSplat} contains a unified differentiable physical imaging model which first applies continuous-time trajectory constraint to imaging process so that we can jointly optimize exposure time, camera response function (CRF), camera poses, and sharp 3D HDR scene. Extensive experiments demonstrate that our approach outperforms existing methods in terms of robustness and rendering quality. Our source code will be available at https://github.com/WU-CVGL/CasualHDRSplat
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14423v3">PhysFlow: Unleashing the Potential of Multi-modal Foundation Models and Video Diffusion for 4D Dynamic Physical Scene Simulation</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 CVPR 2025. Homepage: https://zhuomanliu.github.io/PhysFlow/
    </div>
    <details class="paper-abstract">
      Realistic simulation of dynamic scenes requires accurately capturing diverse material properties and modeling complex object interactions grounded in physical principles. However, existing methods are constrained to basic material types with limited predictable parameters, making them insufficient to represent the complexity of real-world materials. We introduce PhysFlow, a novel approach that leverages multi-modal foundation models and video diffusion to achieve enhanced 4D dynamic scene simulation. Our method utilizes multi-modal models to identify material types and initialize material parameters through image queries, while simultaneously inferring 3D Gaussian splats for detailed scene representation. We further refine these material parameters using video diffusion with a differentiable Material Point Method (MPM) and optical flow guidance rather than render loss or Score Distillation Sampling (SDS) loss. This integrated framework enables accurate prediction and realistic simulation of dynamic interactions in real-world scenarios, advancing both accuracy and flexibility in physics-based simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05702v8">NGM-SLAM: Gaussian Splatting SLAM with Radiance Field Submap</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 9pages, 4 figures
    </div>
    <details class="paper-abstract">
      SLAM systems based on Gaussian Splatting have garnered attention due to their capabilities for rapid real-time rendering and high-fidelity mapping. However, current Gaussian Splatting SLAM systems usually struggle with large scene representation and lack effective loop closure detection. To address these issues, we introduce NGM-SLAM, the first 3DGS based SLAM system that utilizes neural radiance field submaps for progressive scene expression, effectively integrating the strengths of neural radiance fields and 3D Gaussian Splatting. We utilize neural radiance field submaps as supervision and achieve high-quality scene expression and online loop closure adjustments through Gaussian rendering of fused submaps. Our results on multiple real-world scenes and large-scale scene datasets demonstrate that our method can achieve accurate hole filling and high-quality scene expression, supporting monocular, stereo, and RGB-D inputs, and achieving state-of-the-art scene reconstruction and tracking performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17954v1">iVR-GS: Inverse Volume Rendering for Explorable Visualization via Editable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 Accepted by IEEE Transactions on Visualization and Computer Graphics (TVCG)
    </div>
    <details class="paper-abstract">
      In volume visualization, users can interactively explore the three-dimensional data by specifying color and opacity mappings in the transfer function (TF) or adjusting lighting parameters, facilitating meaningful interpretation of the underlying structure. However, rendering large-scale volumes demands powerful GPUs and high-speed memory access for real-time performance. While existing novel view synthesis (NVS) methods offer faster rendering speeds with lower hardware requirements, the visible parts of a reconstructed scene are fixed and constrained by preset TF settings, significantly limiting user exploration. This paper introduces inverse volume rendering via Gaussian splatting (iVR-GS), an innovative NVS method that reduces the rendering cost while enabling scene editing for interactive volume exploration. Specifically, we compose multiple iVR-GS models associated with basic TFs covering disjoint visible parts to make the entire volumetric scene visible. Each basic model contains a collection of 3D editable Gaussians, where each Gaussian is a 3D spatial point that supports real-time scene rendering and editing. We demonstrate the superior reconstruction quality and composability of iVR-GS against other NVS solutions (Plenoxels, CCNeRF, and base 3DGS) on various volume datasets. The code is available at https://github.com/TouKaienn/iVR-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01535v3">GaussianBlock: Building Part-Aware Compositional and Editable 3D Scene by Primitives and Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Recently, with the development of Neural Radiance Fields and Gaussian Splatting, 3D reconstruction techniques have achieved remarkably high fidelity. However, the latent representations learnt by these methods are highly entangled and lack interpretability. In this paper, we propose a novel part-aware compositional reconstruction method, called GaussianBlock, that enables semantically coherent and disentangled representations, allowing for precise and physical editing akin to building blocks, while simultaneously maintaining high fidelity. Our GaussianBlock introduces a hybrid representation that leverages the advantages of both primitives, known for their flexible actionability and editability, and 3D Gaussians, which excel in reconstruction quality. Specifically, we achieve semantically coherent primitives through a novel attention-guided centering loss derived from 2D semantic priors, complemented by a dynamic splitting and fusion strategy. Furthermore, we utilize 3D Gaussians that hybridize with primitives to refine structural details and enhance fidelity. Additionally, a binding inheritance strategy is employed to strengthen and maintain the connection between the two. Our reconstructed scenes are evidenced to be disentangled, compositional, and compact across diverse benchmarks, enabling seamless, direct and precise editing while maintaining high quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01503v2">Luminance-GS: Adapting 3D Gaussian Splatting to Challenging Lighting Conditions with View-Adaptive Curve Adjustment</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 CVPR 2025, project page: https://cuiziteng.github.io/Luminance_GS_web/
    </div>
    <details class="paper-abstract">
      Capturing high-quality photographs under diverse real-world lighting conditions is challenging, as both natural lighting (e.g., low-light) and camera exposure settings (e.g., exposure time) significantly impact image quality. This challenge becomes more pronounced in multi-view scenarios, where variations in lighting and image signal processor (ISP) settings across viewpoints introduce photometric inconsistencies. Such lighting degradations and view-dependent variations pose substantial challenges to novel view synthesis (NVS) frameworks based on Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). To address this, we introduce Luminance-GS, a novel approach to achieving high-quality novel view synthesis results under diverse challenging lighting conditions using 3DGS. By adopting per-view color matrix mapping and view-adaptive curve adjustments, Luminance-GS achieves state-of-the-art (SOTA) results across various lighting conditions -- including low-light, overexposure, and varying exposure -- while not altering the original 3DGS explicit representation. Compared to previous NeRF- and 3DGS-based baselines, Luminance-GS provides real-time rendering speed with improved reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16740v1">Gaussian Splatting is an Effective Data Generator for 3D Object Detection</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      We investigate data augmentation for 3D object detection in autonomous driving. We utilize recent advancements in 3D reconstruction based on Gaussian Splatting for 3D object placement in driving scenes. Unlike existing diffusion-based methods that synthesize images conditioned on BEV layouts, our approach places 3D objects directly in the reconstructed 3D space with explicitly imposed geometric transformations. This ensures both the physical plausibility of object placement and highly accurate 3D pose and position annotations. Our experiments demonstrate that even by integrating a limited number of external 3D objects into real scenes, the augmented data significantly enhances 3D object detection performance and outperforms existing diffusion-based 3D augmentation for object detection. Extensive testing on the nuScenes dataset reveals that imposing high geometric diversity in object placement has a greater impact compared to the appearance diversity of objects. Additionally, we show that generating hard examples, either by maximizing detection loss or imposing high visual occlusion in camera images, does not lead to more efficient 3D data augmentation for camera-based 3D object detection in autonomous driving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16693v1">PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      While non-prehensile manipulation (e.g., controlled pushing/poking) constitutes a foundational robotic skill, its learning remains challenging due to the high sensitivity to complex physical interactions involving friction and restitution. To achieve robust policy learning and generalization, we opt to learn a world model of the 3D rigid body dynamics involved in non-prehensile manipulations and use it for model-based reinforcement learning. We propose PIN-WM, a Physics-INformed World Model that enables efficient end-to-end identification of a 3D rigid body dynamical system from visual observations. Adopting differentiable physics simulation, PIN-WM can be learned with only few-shot and task-agnostic physical interaction trajectories. Further, PIN-WM is learned with observational loss induced by Gaussian Splatting without needing state estimation. To bridge Sim2Real gaps, we turn the learned PIN-WM into a group of Digital Cousins via physics-aware randomizations which perturb physics and rendering parameters to generate diverse and meaningful variations of the PIN-WM. Extensive evaluations on both simulation and real-world tests demonstrate that PIN-WM, enhanced with physics-aware digital cousins, facilitates learning robust non-prehensile manipulation skills with Sim2Real transfer, surpassing the Real2Sim2Real state-of-the-arts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14373v2">SEGA: Drivable 3D Gaussian Head Avatar from a Single Image</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Creating photorealistic 3D head avatars from limited input has become increasingly important for applications in virtual reality, telepresence, and digital entertainment. While recent advances like neural rendering and 3D Gaussian splatting have enabled high-quality digital human avatar creation and animation, most methods rely on multiple images or multi-view inputs, limiting their practicality for real-world use. In this paper, we propose SEGA, a novel approach for Single-imagE-based 3D drivable Gaussian head Avatar creation that combines generalized prior models with a new hierarchical UV-space Gaussian Splatting framework. SEGA seamlessly combines priors derived from large-scale 2D datasets with 3D priors learned from multi-view, multi-expression, and multi-ID data, achieving robust generalization to unseen identities while ensuring 3D consistency across novel viewpoints and expressions. We further present a hierarchical UV-space Gaussian Splatting framework that leverages FLAME-based structural priors and employs a dual-branch architecture to disentangle dynamic and static facial components effectively. The dynamic branch encodes expression-driven fine details, while the static branch focuses on expression-invariant regions, enabling efficient parameter inference and precomputation. This design maximizes the utility of limited 3D data and achieves real-time performance for animation and rendering. Additionally, SEGA performs person-specific fine-tuning to further enhance the fidelity and realism of the generated avatars. Experiments show our method outperforms state-of-the-art approaches in generalization ability, identity preservation, and expression realism, advancing one-shot avatar creation for practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16606v1">HUG: Hierarchical Urban Gaussian Splatting with Block-Based Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      As urban 3D scenes become increasingly complex and the demand for high-quality rendering grows, efficient scene reconstruction and rendering techniques become crucial. We present HUG, a novel approach to address inefficiencies in handling large-scale urban environments and intricate details based on 3D Gaussian splatting. Our method optimizes data partitioning and the reconstruction pipeline by incorporating a hierarchical neural Gaussian representation. We employ an enhanced block-based reconstruction pipeline focusing on improving reconstruction quality within each block and reducing the need for redundant training regions around block boundaries. By integrating neural Gaussian representation with a hierarchical architecture, we achieve high-quality scene rendering at a low computational cost. This is demonstrated by our state-of-the-art results on public benchmarks, which prove the effectiveness and advantages in large-scale urban scene representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16545v1">ToF-Splatting: Dense SLAM using Sparse Time-of-Flight Depth and Multi-Frame Integration</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Time-of-Flight (ToF) sensors provide efficient active depth sensing at relatively low power budgets; among such designs, only very sparse measurements from low-resolution sensors are considered to meet the increasingly limited power constraints of mobile and AR/VR devices. However, such extreme sparsity levels limit the seamless usage of ToF depth in SLAM. In this work, we propose ToF-Splatting, the first 3D Gaussian Splatting-based SLAM pipeline tailored for using effectively very sparse ToF input data. Our approach improves upon the state of the art by introducing a multi-frame integration module, which produces dense depth maps by merging cues from extremely sparse ToF depth, monocular color, and multi-view geometry. Extensive experiments on both synthetic and real sparse ToF datasets demonstrate the viability of our approach, as it achieves state-of-the-art tracking and mapping performances on reference datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10475v2">X-SG$^2$S: Safe and Generalizable Gaussian Splatting with X-dimensional Watermarks</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has been widely used in 3D reconstruction and 3D generation. Training to get a 3DGS scene often takes a lot of time and resources and even valuable inspiration. The increasing amount of 3DGS digital asset have brought great challenges to the copyright protection. However, it still lacks profound exploration targeted at 3DGS. In this paper, we propose a new framework X-SG$^2$S which can simultaneously watermark 1 to 3D messages while keeping the original 3DGS scene almost unchanged. Generally, we have a X-SG$^2$S injector for adding multi-modal messages simultaneously and an extractor for extract them. Specifically, we first split the watermarks into message patches in a fixed manner and sort the 3DGS points. A self-adaption gate is used to pick out suitable location for watermarking. Then use a XD(multi-dimension)-injection heads to add multi-modal messages into sorted 3DGS points. A learnable gate can recognize the location with extra messages and XD-extraction heads can restore hidden messages from the location recommended by the learnable gate. Extensive experiments demonstrated that the proposed X-SG$^2$S can effectively conceal multi modal messages without changing pretrained 3DGS pipeline or the original form of 3DGS parameters. Meanwhile, with simple and efficient model structure and high practicality, X-SG$^2$S still shows good performance in hiding and extracting multi-modal inner structured or unstructured messages. X-SG$^2$S is the first to unify 1 to 3D watermarking model for 3DGS and the first framework to add multi-modal watermarks simultaneous in one 3DGS which pave the wave for later researches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01552v4">GFreeDet: Exploiting Gaussian Splatting and Foundation Models for Model-free Unseen Object Detection in the BOP Challenge 2024</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 CVPR 2025 CV4MR Workshop (citation style changed)
    </div>
    <details class="paper-abstract">
      We present GFreeDet, an unseen object detection approach that leverages Gaussian splatting and vision Foundation models under model-free setting. Unlike existing methods that rely on predefined CAD templates, GFreeDet reconstructs objects directly from reference videos using Gaussian splatting, enabling robust detection of novel objects without prior 3D models. Evaluated on the BOP-H3 benchmark, GFreeDet achieves comparable performance to CAD-based methods, demonstrating the viability of model-free detection for mixed reality (MR) applications. Notably, GFreeDet won the best overall method and the best fast method awards in the model-free 2D detection track at BOP Challenge 2024.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17815v1">Visibility-Uncertainty-guided 3D Gaussian Inpainting via Scene Conceptional Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 14 pages, 12 figures, ICCV
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful and efficient 3D representation for novel view synthesis. This paper extends 3DGS capabilities to inpainting, where masked objects in a scene are replaced with new contents that blend seamlessly with the surroundings. Unlike 2D image inpainting, 3D Gaussian inpainting (3DGI) is challenging in effectively leveraging complementary visual and semantic cues from multiple input views, as occluded areas in one view may be visible in others. To address this, we propose a method that measures the visibility uncertainties of 3D points across different input views and uses them to guide 3DGI in utilizing complementary visual cues. We also employ uncertainties to learn a semantic concept of scene without the masked object and use a diffusion model to fill masked objects in input images based on the learned concept. Finally, we build a novel 3DGI framework, VISTA, by integrating VISibility-uncerTainty-guided 3DGI with scene conceptuAl learning. VISTA generates high-quality 3DGS models capable of synthesizing artifact-free and naturally inpainted novel views. Furthermore, our approach extends to handling dynamic distractors arising from temporal object changes, enhancing its versatility in diverse scene reconstruction scenarios. We demonstrate the superior performance of our method over state-of-the-art techniques using two challenging datasets: the SPIn-NeRF dataset, featuring 10 diverse static 3D inpainting scenes, and an underwater 3D inpainting dataset derived from UTB180, including fast-moving fish as inpainting targets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18473v2">HEMGS: A Hybrid Entropy Model for 3D Gaussian Splatting Data Compression</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      In this work, we propose a novel compression framework for 3D Gaussian Splatting (3DGS) data. Building on anchor-based 3DGS methodologies, our approach compresses all attributes within each anchor by introducing a novel Hybrid Entropy Model for 3D Gaussian Splatting (HEMGS) to achieve hybrid lossy-lossless compression. It consists of three main components: a variable-rate predictor, a hyperprior network, and an autoregressive network. First, unlike previous methods that adopt multiple models to achieve multi-rate lossy compression, thereby increasing training overhead, our variable-rate predictor enables variable-rate compression with a single model and a hyperparameter $\lambda$ by producing a learned Quantization Step feature for versatile lossy compression. Second, to improve lossless compression, the hyperprior network captures both scene-agnostic and scene-specific features to generate a prior feature, while the autoregressive network employs an adaptive context selection algorithm with flexible receptive fields to produce a contextual feature. By integrating these two features, HEMGS can accurately estimate the distribution of the current coding element within each attribute, enabling improved entropy coding and reduced storage. We integrate HEMGS into a compression framework, and experimental results on four benchmarks indicate that HEMGS achieves about a 40% average reduction in size while maintaining rendering quality over baseline methods and achieving state-of-the-art compression results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.15676v2">3DGR-CT: Sparse-View CT Reconstruction with a 3D Gaussian Representation</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Sparse-view computed tomography (CT) reduces radiation exposure by acquiring fewer projections, making it a valuable tool in clinical scenarios where low-dose radiation is essential. However, this often results in increased noise and artifacts due to limited data. In this paper we propose a novel 3D Gaussian representation (3DGR) based method for sparse-view CT reconstruction. Inspired by recent success in novel view synthesis driven by 3D Gaussian splatting, we leverage the efficiency and expressiveness of 3D Gaussian representation as an alternative to implicit neural representation. To unleash the potential of 3DGR for CT imaging scenario, we propose two key innovations: (i) FBP-image-guided Guassian initialization and (ii) efficient integration with a differentiable CT projector. Extensive experiments and ablations on diverse datasets demonstrate the proposed 3DGR-CT consistently outperforms state-of-the-art counterpart methods, achieving higher reconstruction accuracy with faster convergence. Furthermore, we showcase the potential of 3DGR-CT for real-time physical simulation, which holds important clinical applications while challenging for implicit neural representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07200v2">ThermalGaussian: Thermal 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Thermography is especially valuable for the military and other users of surveillance cameras. Some recent methods based on Neural Radiance Fields (NeRF) are proposed to reconstruct the thermal scenes in 3D from a set of thermal and RGB images. However, unlike NeRF, 3D Gaussian splatting (3DGS) prevails due to its rapid training and real-time rendering. In this work, we propose ThermalGaussian, the first thermal 3DGS approach capable of rendering high-quality images in RGB and thermal modalities. We first calibrate the RGB camera and the thermal camera to ensure that both modalities are accurately aligned. Subsequently, we use the registered images to learn the multimodal 3D Gaussians. To prevent the overfitting of any single modality, we introduce several multimodal regularization constraints. We also develop smoothing constraints tailored to the physical characteristics of the thermal modality. Besides, we contribute a real-world dataset named RGBT-Scenes, captured by a hand-hold thermal-infrared camera, facilitating future research on thermal scene reconstruction. We conduct comprehensive experiments to show that ThermalGaussian achieves photorealistic rendering of thermal images and improves the rendering quality of RGB images. With the proposed multimodal regularization constraints, we also reduced the model's storage cost by 90%. Our project page is at https://thermalgaussian.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.07608v2">Faster and Better 3D Splatting via Group Training</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, demonstrating remarkable capability in high-fidelity scene reconstruction through its Gaussian primitive representations. However, the computational overhead induced by the massive number of primitives poses a significant bottleneck to training efficiency. To overcome this challenge, we propose Group Training, a simple yet effective strategy that organizes Gaussian primitives into manageable groups, optimizing training efficiency and improving rendering quality. This approach shows universal compatibility with existing 3DGS frameworks, including vanilla 3DGS and Mip-Splatting, consistently achieving accelerated training while maintaining superior synthesis quality. Extensive experiments reveal that our straightforward Group Training strategy achieves up to 30% faster convergence and improved rendering quality across diverse scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17810v1">SmallGS: Gaussian Splatting-based Camera Pose Estimation for Small-Baseline Videos</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 10 pages, 4 figures, Accepted by CVPR workshop
    </div>
    <details class="paper-abstract">
      Dynamic videos with small baseline motions are ubiquitous in daily life, especially on social media. However, these videos present a challenge to existing pose estimation frameworks due to ambiguous features, drift accumulation, and insufficient triangulation constraints. Gaussian splatting, which maintains an explicit representation for scenes, provides a reliable novel view rasterization when the viewpoint change is small. Inspired by this, we propose SmallGS, a camera pose estimation framework that is specifically designed for small-baseline videos. SmallGS optimizes sequential camera poses using Gaussian splatting, which reconstructs the scene from the first frame in each video segment to provide a stable reference for the rest. The temporal consistency of Gaussian splatting within limited viewpoint differences reduced the requirement of sufficient depth variations in traditional camera pose estimation. We further incorporate pretrained robust visual features, e.g. DINOv2, into Gaussian splatting, where high-dimensional feature map rendering enhances the robustness of camera pose estimation. By freezing the Gaussian splatting and optimizing camera viewpoints based on rasterized features, SmallGS effectively learns camera poses without requiring explicit feature correspondences or strong parallax motion. We verify the effectiveness of SmallGS in small-baseline videos in TUM-Dynamics sequences, which achieves impressive accuracy in camera pose estimation compared to MonST3R and DORID-SLAM for small-baseline videos in dynamic scenes. Our project page is at: https://yuxinyao620.github.io/SmallGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15281v1">StyleMe3D: Stylization with Disentangled Priors by Multiple Encoders on 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 16 pages; Project page: https://styleme3d.github.io/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) excels in photorealistic scene reconstruction but struggles with stylized scenarios (e.g., cartoons, games) due to fragmented textures, semantic misalignment, and limited adaptability to abstract aesthetics. We propose StyleMe3D, a holistic framework for 3D GS style transfer that integrates multi-modal style conditioning, multi-level semantic alignment, and perceptual quality enhancement. Our key insights include: (1) optimizing only RGB attributes preserves geometric integrity during stylization; (2) disentangling low-, medium-, and high-level semantics is critical for coherent style transfer; (3) scalability across isolated objects and complex scenes is essential for practical deployment. StyleMe3D introduces four novel components: Dynamic Style Score Distillation (DSSD), leveraging Stable Diffusion's latent space for semantic alignment; Contrastive Style Descriptor (CSD) for localized, content-aware texture transfer; Simultaneously Optimized Scale (SOS) to decouple style details and structural coherence; and 3D Gaussian Quality Assessment (3DG-QA), a differentiable aesthetic prior trained on human-rated data to suppress artifacts and enhance visual harmony. Evaluated on NeRF synthetic dataset (objects) and tandt db (scenes) datasets, StyleMe3D outperforms state-of-the-art methods in preserving geometric details (e.g., carvings on sculptures) and ensuring stylistic consistency across scenes (e.g., coherent lighting in landscapes), while maintaining real-time rendering. This work bridges photorealistic 3D GS and artistic stylization, unlocking applications in gaming, virtual worlds, and digital art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15229v1">Immersive Teleoperation Framework for Locomanipulation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 CASE2025, 8 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in robotic loco-manipulation have leveraged Virtual Reality (VR) to enhance the precision and immersiveness of teleoperation systems, significantly outperforming traditional methods reliant on 2D camera feeds and joystick controls. Despite these advancements, challenges remain, particularly concerning user experience across different setups. This paper introduces a novel VR-based teleoperation framework designed for a robotic manipulator integrated onto a mobile platform. Central to our approach is the application of Gaussian splatting, a technique that abstracts the manipulable scene into a VR environment, thereby enabling more intuitive and immersive interactions. Users can navigate and manipulate within the virtual scene as if interacting with a real robot, enhancing both the engagement and efficacy of teleoperation tasks. An extensive user study validates our approach, demonstrating significant usability and efficiency improvements. Two-thirds (66%) of participants completed tasks faster, achieving an average time reduction of 43%. Additionally, 93% preferred the Gaussian Splat interface overall, with unanimous (100%) recommendations for future use, highlighting improvements in precision, responsiveness, and situational awareness. Finally, we demonstrate the effectiveness of our framework through real-world experiments in two distinct application scenarios, showcasing the practical capabilities and versatility of the Splat-based VR interface.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15122v1">MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 The first two authors contributed equally to this work (equal contribution). The last two authors advised equally to this work
    </div>
    <details class="paper-abstract">
      We present MoBGS, a novel deblurring dynamic 3D Gaussian Splatting (3DGS) framework capable of reconstructing sharp and high-quality novel spatio-temporal views from blurry monocular videos in an end-to-end manner. Existing dynamic novel view synthesis (NVS) methods are highly sensitive to motion blur in casually captured videos, resulting in significant degradation of rendering quality. While recent approaches address motion-blurred inputs for NVS, they primarily focus on static scene reconstruction and lack dedicated motion modeling for dynamic objects. To overcome these limitations, our MoBGS introduces a novel Blur-adaptive Latent Camera Estimation (BLCE) method for effective latent camera trajectory estimation, improving global camera motion deblurring. In addition, we propose a physically-inspired Latent Camera-induced Exposure Estimation (LCEE) method to ensure consistent deblurring of both global camera and local object motion. Our MoBGS framework ensures the temporal consistency of unseen latent timestamps and robust motion decomposition of static and dynamic regions. Extensive experiments on the Stereo Blur dataset and real-world blurry videos show that our MoBGS significantly outperforms the very recent advanced methods (DyBluRF and Deblur4DGS), achieving state-of-the-art performance for dynamic NVS under motion blur.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06945v3">Direct Learning of Mesh and Appearance via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Accurately reconstructing a 3D scene including explicit geometry information is both attractive and challenging. Geometry reconstruction can benefit from incorporating differentiable appearance models, such as Neural Radiance Fields and 3D Gaussian Splatting (3DGS). However, existing methods encounter efficiency issues due to indirect geometry learning and the paradigm of separately modeling geometry and surface appearance. In this work, we propose a learnable scene model that incorporates 3DGS with an explicit geometry representation, namely a mesh. Our model learns the mesh and appearance in an end-to-end manner, where we bind 3D Gaussians to the mesh faces and perform differentiable rendering of 3DGS to obtain photometric supervision. The model creates an effective information pathway to supervise the learning of both 3DGS and mesh. Experimental results demonstrate that the learned scene model not only improves efficiency and rendering quality but also enables manipulation via the explicit mesh. In addition, our model has a unique advantage in adapting to scene updates, thanks to the end-to-end learning of both mesh and appearance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12369v2">DARB-Splatting: Generalizing Splatting with Decaying Anisotropic Radial Basis Functions</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 Link to the project page: https://randomnerds.github.io/darbs.github.io/
    </div>
    <details class="paper-abstract">
      Splatting-based 3D reconstruction methods have gained popularity with the advent of 3D Gaussian Splatting, efficiently synthesizing high-quality novel views. These methods commonly resort to using exponential family functions, such as the Gaussian function, as reconstruction kernels due to their anisotropic nature, ease of projection, and differentiability in rasterization. However, the field remains restricted to variations within the exponential family, leaving generalized reconstruction kernels largely underexplored, partly due to the lack of easy integrability in 3D to 2D projections. In this light, we show that a class of decaying anisotropic radial basis functions (DARBFs), which are non-negative functions of the Mahalanobis distance, supports splatting by approximating the Gaussian function's closed-form integration advantage. With this fresh perspective, we demonstrate up to 34% faster convergence during training and a 45% reduction in memory consumption across various DARB reconstruction kernels, while maintaining comparable PSNR, SSIM, and LPIPS results. We will make the code available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13713v2">SLAM&Render: A Benchmark for the Intersection Between Neural Rendering, Gaussian Splatting and SLAM</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 8 pages, 8 figures, RA-L submission
    </div>
    <details class="paper-abstract">
      Models and methods originally developed for novel view synthesis and scene rendering, such as Neural Radiance Fields (NeRF) and Gaussian Splatting, are increasingly being adopted as representations in Simultaneous Localization and Mapping (SLAM). However, existing datasets fail to include the specific challenges of both fields, such as multimodality and sequentiality in SLAM or generalization across viewpoints and illumination conditions in neural rendering. To bridge this gap, we introduce SLAM&Render, a novel dataset designed to benchmark methods in the intersection between SLAM and novel view rendering. It consists of 40 sequences with synchronized RGB, depth, IMU, robot kinematic data, and ground-truth pose streams. By releasing robot kinematic data, the dataset also enables the assessment of novel SLAM strategies when applied to robot manipulators. The dataset sequences span five different setups featuring consumer and industrial objects under four different lighting conditions, with separate training and test trajectories per scene, as well as object rearrangements. Our experimental results, obtained with several baselines from the literature, validate SLAM&Render as a relevant benchmark for this emerging research area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08224v2">HRAvatar: High-Quality and Relightable Gaussian Head Avatar</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 Accepted to CVPR 2025,Project page: https://eastbeanzhang.github.io/HRAvatar
    </div>
    <details class="paper-abstract">
      Reconstructing animatable and high-quality 3D head avatars from monocular videos, especially with realistic relighting, is a valuable task. However, the limited information from single-view input, combined with the complex head poses and facial movements, makes this challenging. Previous methods achieve real-time performance by combining 3D Gaussian Splatting with a parametric head model, but the resulting head quality suffers from inaccurate face tracking and limited expressiveness of the deformation model. These methods also fail to produce realistic effects under novel lighting conditions. To address these issues, we propose HRAvatar, a 3DGS-based method that reconstructs high-fidelity, relightable 3D head avatars. HRAvatar reduces tracking errors through end-to-end optimization and better captures individual facial deformations using learnable blendshapes and learnable linear blend skinning. Additionally, it decomposes head appearance into several physical properties and incorporates physically-based shading to account for environmental lighting. Extensive experiments demonstrate that HRAvatar not only reconstructs superior-quality heads but also achieves realistic visual effects under varying lighting conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00387v2">Scene4U: Hierarchical Layered 3D Scene Reconstruction from Single Panoramic Image for Your Immerse Exploration</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 CVPR 2025, 11 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The reconstruction of immersive and realistic 3D scenes holds significant practical importance in various fields of computer vision and computer graphics. Typically, immersive and realistic scenes should be free from obstructions by dynamic objects, maintain global texture consistency, and allow for unrestricted exploration. The current mainstream methods for image-driven scene construction involves iteratively refining the initial image using a moving virtual camera to generate the scene. However, previous methods struggle with visual discontinuities due to global texture inconsistencies under varying camera poses, and they frequently exhibit scene voids caused by foreground-background occlusions. To this end, we propose a novel layered 3D scene reconstruction framework from panoramic image, named Scene4U. Specifically, Scene4U integrates an open-vocabulary segmentation model with a large language model to decompose a real panorama into multiple layers. Then, we employs a layered repair module based on diffusion model to restore occluded regions using visual cues and depth information, generating a hierarchical representation of the scene. The multi-layer panorama is then initialized as a 3D Gaussian Splatting representation, followed by layered optimization, which ultimately produces an immersive 3D scene with semantic and structural consistency that supports free exploration. Scene4U outperforms state-of-the-art method, improving by 24.24% in LPIPS and 24.40% in BRISQUE, while also achieving the fastest training speed. Additionally, to demonstrate the robustness of Scene4U and allow users to experience immersive scenes from various landmarks, we build WorldVista3D dataset for 3D scene reconstruction, which contains panoramic images of globally renowned sites. The implementation code and dataset will be released at https://github.com/LongHZ140516/Scene4U .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05769v3">Digital Twin Buildings: 3D Modeling, GIS Integration, and Visual Descriptions Using Gaussian Splatting, ChatGPT/Deepseek, and Google Maps Platform</a></div>
    <div class="paper-meta">
      📅 2025-04-20
      | 💬 -Fixed minor typo
    </div>
    <details class="paper-abstract">
      Urban digital twins are virtual replicas of cities that use multi-source data and data analytics to optimize urban planning, infrastructure management, and decision-making. Towards this, we propose a framework focused on the single-building scale. By connecting to cloud mapping platforms such as Google Map Platforms APIs, by leveraging state-of-the-art multi-agent Large Language Models data analysis using ChatGPT(4o) and Deepseek-V3/R1, and by using our Gaussian Splatting-based mesh extraction pipeline, our Digital Twin Buildings framework can retrieve a building's 3D model, visual descriptions, and achieve cloud-based mapping integration with large language model-based data analytics using a building's address, postal code, or geographic coordinates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14699v1">IXGS-Intraoperative 3D Reconstruction from Sparse, Arbitrarily Posed Real X-rays</a></div>
    <div class="paper-meta">
      📅 2025-04-20
    </div>
    <details class="paper-abstract">
      Spine surgery is a high-risk intervention demanding precise execution, often supported by image-based navigation systems. Recently, supervised learning approaches have gained attention for reconstructing 3D spinal anatomy from sparse fluoroscopic data, significantly reducing reliance on radiation-intensive 3D imaging systems. However, these methods typically require large amounts of annotated training data and may struggle to generalize across varying patient anatomies or imaging conditions. Instance-learning approaches like Gaussian splatting could offer an alternative by avoiding extensive annotation requirements. While Gaussian splatting has shown promise for novel view synthesis, its application to sparse, arbitrarily posed real intraoperative X-rays has remained largely unexplored. This work addresses this limitation by extending the $R^2$-Gaussian splatting framework to reconstruct anatomically consistent 3D volumes under these challenging conditions. We introduce an anatomy-guided radiographic standardization step using style transfer, improving visual consistency across views, and enhancing reconstruction quality. Notably, our framework requires no pretraining, making it inherently adaptable to new patients and anatomies. We evaluated our approach using an ex-vivo dataset. Expert surgical evaluation confirmed the clinical utility of the 3D reconstructions for navigation, especially when using 20 to 30 views, and highlighted the standardization's benefit for anatomical clarity. Benchmarking via quantitative 2D metrics (PSNR/SSIM) confirmed performance trade-offs compared to idealized settings, but also validated the improvement gained from standardization over raw inputs. This work demonstrates the feasibility of instance-based volumetric reconstruction from arbitrary sparse-view X-rays, advancing intraoperative 3D imaging for surgical navigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14638v1">NVSMask3D: Hard Visual Prompting with Camera Pose Interpolation for 3D Open Vocabulary Instance Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-04-20
      | 💬 15 pages, 4 figures, Scandinavian Conference on Image Analysis 2025
    </div>
    <details class="paper-abstract">
      Vision-language models (VLMs) have demonstrated impressive zero-shot transfer capabilities in image-level visual perception tasks. However, they fall short in 3D instance-level segmentation tasks that require accurate localization and recognition of individual objects. To bridge this gap, we introduce a novel 3D Gaussian Splatting based hard visual prompting approach that leverages camera interpolation to generate diverse viewpoints around target objects without any 2D-3D optimization or fine-tuning. Our method simulates realistic 3D perspectives, effectively augmenting existing hard visual prompts by enforcing geometric consistency across viewpoints. This training-free strategy seamlessly integrates with prior hard visual prompts, enriching object-descriptive features and enabling VLMs to achieve more robust and accurate 3D instance segmentation in diverse 3D scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01552v3">GFreeDet: Exploiting Gaussian Splatting and Foundation Models for Model-free Unseen Object Detection in the BOP Challenge 2024</a></div>
    <div class="paper-meta">
      📅 2025-04-20
      | 💬 CVPR 2025 CV4MR Workshop
    </div>
    <details class="paper-abstract">
      We present GFreeDet, an unseen object detection approach that leverages Gaussian splatting and vision Foundation models under model-free setting. Unlike existing methods that rely on predefined CAD templates, GFreeDet reconstructs objects directly from reference videos using Gaussian splatting, enabling robust detection of novel objects without prior 3D models. Evaluated on the BOP-H3 benchmark, GFreeDet achieves comparable performance to CAD-based methods, demonstrating the viability of model-free detection for mixed reality (MR) applications. Notably, GFreeDet won the best overall method and the best fast method awards in the model-free 2D detection track at BOP Challenge 2024.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14548v1">VGNC: Reducing the Overfitting of Sparse-view 3DGS via Validation-guided Gaussian Number Control</a></div>
    <div class="paper-meta">
      📅 2025-04-20
      | 💬 10 pages,8 figures
    </div>
    <details class="paper-abstract">
      Sparse-view 3D reconstruction is a fundamental yet challenging task in practical 3D reconstruction applications. Recently, many methods based on the 3D Gaussian Splatting (3DGS) framework have been proposed to address sparse-view 3D reconstruction. Although these methods have made considerable advancements, they still show significant issues with overfitting. To reduce the overfitting, we introduce VGNC, a novel Validation-guided Gaussian Number Control (VGNC) approach based on generative novel view synthesis (NVS) models. To the best of our knowledge, this is the first attempt to alleviate the overfitting issue of sparse-view 3DGS with generative validation images. Specifically, we first introduce a validation image generation method based on a generative NVS model. We then propose a Gaussian number control strategy that utilizes generated validation images to determine the optimal Gaussian numbers, thereby reducing the issue of overfitting. We conducted detailed experiments on various sparse-view 3DGS baselines and datasets to evaluate the effectiveness of VGNC. Extensive experiments show that our approach not only reduces overfitting but also improves rendering quality on the test set while decreasing the number of Gaussian points. This reduction lowers storage demands and accelerates both training and rendering. The code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14460v1">Metamon-GS: Enhancing Representability with Variance-Guided Densification and Light Encoding</a></div>
    <div class="paper-meta">
      📅 2025-04-20
    </div>
    <details class="paper-abstract">
      The introduction of 3D Gaussian Splatting (3DGS) has advanced novel view synthesis by utilizing Gaussians to represent scenes. Encoding Gaussian point features with anchor embeddings has significantly enhanced the performance of newer 3DGS variants. While significant advances have been made, it is still challenging to boost rendering performance. Feature embeddings have difficulty accurately representing colors from different perspectives under varying lighting conditions, which leads to a washed-out appearance. Another reason is the lack of a proper densification strategy that prevents Gaussian point growth in thinly initialized areas, resulting in blurriness and needle-shaped artifacts. To address them, we propose Metamon-GS, from innovative viewpoints of variance-guided densification strategy and multi-level hash grid. The densification strategy guided by variance specifically targets Gaussians with high gradient variance in pixels and compensates for the importance of regions with extra Gaussians to improve reconstruction. The latter studies implicit global lighting conditions and accurately interprets color from different perspectives and feature embeddings. Our thorough experiments on publicly available datasets show that Metamon-GS surpasses its baseline model and previous versions, delivering superior quality in rendering novel views.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04251v3">Improving Gaussian Splatting with Localized Points Management</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 CVPR 2025 (Highlight). Github: https://happy-hsy.github.io/projects/LPM/
    </div>
    <details class="paper-abstract">
      Point management is critical for optimizing 3D Gaussian Splatting models, as point initiation (e.g., via structure from motion) is often distributionally inappropriate. Typically, Adaptive Density Control (ADC) algorithm is adopted, leveraging view-averaged gradient magnitude thresholding for point densification, opacity thresholding for pruning, and regular all-points opacity reset. We reveal that this strategy is limited in tackling intricate/special image regions (e.g., transparent) due to inability of identifying all 3D zones requiring point densification, and lacking an appropriate mechanism to handle ill-conditioned points with negative impacts (e.g., occlusion due to false high opacity). To address these limitations, we propose a Localized Point Management (LPM) strategy, capable of identifying those error-contributing zones in greatest need for both point addition and geometry calibration. Zone identification is achieved by leveraging the underlying multiview geometry constraints, subject to image rendering errors. We apply point densification in the identified zones and then reset the opacity of the points in front of these regions, creating a new opportunity to correct poorly conditioned points. Serving as a versatile plugin, LPM can be seamlessly integrated into existing static 3D and dynamic 4D Gaussian Splatting models with minimal additional cost. Experimental evaluations validate the efficacy of our LPM in boosting a variety of existing 3D/4D models both quantitatively and qualitatively. Notably, LPM improves both static 3DGS and dynamic SpaceTimeGS to achieve state-of-the-art rendering quality while retaining real-time speeds, excelling on challenging datasets such as Tanks & Temples and the Neural 3D Video dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14373v1">SEGA: Drivable 3D Gaussian Head Avatar from a Single Image</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      Creating photorealistic 3D head avatars from limited input has become increasingly important for applications in virtual reality, telepresence, and digital entertainment. While recent advances like neural rendering and 3D Gaussian splatting have enabled high-quality digital human avatar creation and animation, most methods rely on multiple images or multi-view inputs, limiting their practicality for real-world use. In this paper, we propose SEGA, a novel approach for Single-imagE-based 3D drivable Gaussian head Avatar creation that combines generalized prior models with a new hierarchical UV-space Gaussian Splatting framework. SEGA seamlessly combines priors derived from large-scale 2D datasets with 3D priors learned from multi-view, multi-expression, and multi-ID data, achieving robust generalization to unseen identities while ensuring 3D consistency across novel viewpoints and expressions. We further present a hierarchical UV-space Gaussian Splatting framework that leverages FLAME-based structural priors and employs a dual-branch architecture to disentangle dynamic and static facial components effectively. The dynamic branch encodes expression-driven fine details, while the static branch focuses on expression-invariant regions, enabling efficient parameter inference and precomputation. This design maximizes the utility of limited 3D data and achieves real-time performance for animation and rendering. Additionally, SEGA performs person-specific fine-tuning to further enhance the fidelity and realism of the generated avatars. Experiments show our method outperforms state-of-the-art approaches in generalization ability, identity preservation, and expression realism, advancing one-shot avatar creation for practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05176v3">AuraFusion360: Augmented Unseen Region Alignment for Reference-based 360° Unbounded Scene Inpainting</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 Paper accepted to CVPR 2025. Project page: https://kkennethwu.github.io/aurafusion360/
    </div>
    <details class="paper-abstract">
      Three-dimensional scene inpainting is crucial for applications from virtual reality to architectural visualization, yet existing methods struggle with view consistency and geometric accuracy in 360{\deg} unbounded scenes. We present AuraFusion360, a novel reference-based method that enables high-quality object removal and hole filling in 3D scenes represented by Gaussian Splatting. Our approach introduces (1) depth-aware unseen mask generation for accurate occlusion identification, (2) Adaptive Guided Depth Diffusion, a zero-shot method for accurate initial point placement without requiring additional training, and (3) SDEdit-based detail enhancement for multi-view coherence. We also introduce 360-USID, the first comprehensive dataset for 360{\deg} unbounded scene inpainting with ground truth. Extensive experiments demonstrate that AuraFusion360 significantly outperforms existing methods, achieving superior perceptual quality while maintaining geometric accuracy across dramatic viewpoint changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10331v3">LL-Gaussian: Low-Light Scene Reconstruction and Enhancement via Gaussian Splatting for Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 Project page: https://sunhao242.github.io/LL-Gaussian_web.github.io/
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) in low-light scenes remains a significant challenge due to degraded inputs characterized by severe noise, low dynamic range (LDR) and unreliable initialization. While recent NeRF-based approaches have shown promising results, most suffer from high computational costs, and some rely on carefully captured or pre-processed data--such as RAW sensor inputs or multi-exposure sequences--which severely limits their practicality. In contrast, 3D Gaussian Splatting (3DGS) enables real-time rendering with competitive visual fidelity; however, existing 3DGS-based methods struggle with low-light sRGB inputs, resulting in unstable Gaussian initialization and ineffective noise suppression. To address these challenges, we propose LL-Gaussian, a novel framework for 3D reconstruction and enhancement from low-light sRGB images, enabling pseudo normal-light novel view synthesis. Our method introduces three key innovations: 1) an end-to-end Low-Light Gaussian Initialization Module (LLGIM) that leverages dense priors from learning-based MVS approach to generate high-quality initial point clouds; 2) a dual-branch Gaussian decomposition model that disentangles intrinsic scene properties (reflectance and illumination) from transient interference, enabling stable and interpretable optimization; 3) an unsupervised optimization strategy guided by both physical constrains and diffusion prior to jointly steer decomposition and enhancement. Additionally, we contribute a challenging dataset collected in extreme low-light environments and demonstrate the effectiveness of LL-Gaussian. Compared to state-of-the-art NeRF-based methods, LL-Gaussian achieves up to 2,000 times faster inference and reduces training time to just 2%, while delivering superior reconstruction and rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01402v2">ULSR-GS: Ultra Large-scale Surface Reconstruction Gaussian Splatting with Multi-View Geometric Consistency</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 Project page: https://ulsrgs.github.io
    </div>
    <details class="paper-abstract">
      While Gaussian Splatting (GS) demonstrates efficient and high-quality scene rendering and small area surface extraction ability, it falls short in handling large-scale aerial image surface extraction tasks. To overcome this, we present ULSR-GS, a framework dedicated to high-fidelity surface extraction in ultra-large-scale scenes, addressing the limitations of existing GS-based mesh extraction methods. Specifically, we propose a point-to-photo partitioning approach combined with a multi-view optimal view matching principle to select the best training images for each sub-region. Additionally, during training, ULSR-GS employs a densification strategy based on multi-view geometric consistency to enhance surface extraction details. Experimental results demonstrate that ULSR-GS outperforms other state-of-the-art GS-based works on large-scale aerial photogrammetry benchmark datasets, significantly improving surface extraction accuracy in complex urban environments. Project page: https://ulsrgs.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.16760v2">OmniRe: Omni Urban Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 See the project page for code, video results and demos: https://ziyc.github.io/omnire/
    </div>
    <details class="paper-abstract">
      We introduce OmniRe, a comprehensive system for efficiently creating high-fidelity digital twins of dynamic real-world scenes from on-device logs. Recent methods using neural fields or Gaussian Splatting primarily focus on vehicles, hindering a holistic framework for all dynamic foregrounds demanded by downstream applications, e.g., the simulation of human behavior. OmniRe extends beyond vehicle modeling to enable accurate, full-length reconstruction of diverse dynamic objects in urban scenes. Our approach builds scene graphs on 3DGS and constructs multiple Gaussian representations in canonical spaces that model various dynamic actors, including vehicles, pedestrians, cyclists, and others. OmniRe allows holistically reconstructing any dynamic object in the scene, enabling advanced simulations (~60Hz) that include human-participated scenarios, such as pedestrian behavior simulation and human-vehicle interaction. This comprehensive simulation capability is unmatched by existing methods. Extensive evaluations on the Waymo dataset show that our approach outperforms prior state-of-the-art methods quantitatively and qualitatively by a large margin. We further extend our results to 5 additional popular driving datasets to demonstrate its generalizability on common urban scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07759v3">SwinGS: Sliding Window Gaussian Splatting for Volumetric Video Streaming with Arbitrary Length</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have garnered significant attention in computer vision and computer graphics due to its high rendering speed and remarkable quality. While extant research has endeavored to extend the application of 3DGS from static to dynamic scenes, such efforts have been consistently impeded by excessive model sizes, constraints on video duration, and content deviation. These limitations significantly compromise the streamability of dynamic 3D Gaussian models, thereby restricting their utility in downstream applications, including volumetric video, autonomous vehicle, and immersive technologies such as virtual, augmented, and mixed reality. This paper introduces SwinGS, a novel framework for training, delivering, and rendering volumetric video in a real-time streaming fashion. To address the aforementioned challenges and enhance streamability, SwinGS integrates spacetime Gaussian with Markov Chain Monte Carlo (MCMC) to adapt the model to fit various 3D scenes across frames, in the meantime employing a sliding window captures Gaussian snapshots for each frame in an accumulative way. We implement a prototype of SwinGS and demonstrate its streamability across various datasets and scenes. Additionally, we develop an interactive WebGL viewer enabling real-time volumetric video playback on most devices with modern browsers, including smartphones and tablets. Experimental results show that SwinGS reduces transmission costs by 83.6% compared to previous work and could be easily scaled to volumetric videos with arbitrary length with no increasing of required GPU resources.
    </details>
</div>
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
