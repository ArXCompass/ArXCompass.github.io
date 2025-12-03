# gaussian splatting - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

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
