# gaussian splatting - 2025_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12307v1">Swift4D:Adaptive divide-and-conquer Gaussian Splatting for compact and efficient reconstruction of dynamic scene</a></div>
    <div class="paper-meta">
      📅 2025-03-16
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Novel view synthesis has long been a practical but challenging task, although the introduction of numerous methods to solve this problem, even combining advanced representations like 3D Gaussian Splatting, they still struggle to recover high-quality results and often consume too much storage memory and training time. In this paper we propose Swift4D, a divide-and-conquer 3D Gaussian Splatting method that can handle static and dynamic primitives separately, achieving a good trade-off between rendering quality and efficiency, motivated by the fact that most of the scene is the static primitive and does not require additional dynamic properties. Concretely, we focus on modeling dynamic transformations only for the dynamic primitives which benefits both efficiency and quality. We first employ a learnable decomposition strategy to separate the primitives, which relies on an additional parameter to classify primitives as static or dynamic. For the dynamic primitives, we employ a compact multi-resolution 4D Hash mapper to transform these primitives from canonical space into deformation space at each timestamp, and then mix the static and dynamic primitives to produce the final output. This divide-and-conquer method facilitates efficient training and reduces storage redundancy. Our method not only achieves state-of-the-art rendering quality while being 20X faster in training than previous SOTA methods with a minimum storage requirement of only 30MB on real-world datasets. Code is available at https://github.com/WuJH2001/swift4d.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12284v1">REdiSplats: Ray Tracing for Editable Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has become one of the most important neural rendering algorithms. GS represents 3D scenes using Gaussian components with trainable color and opacity. This representation achieves high-quality renderings with fast inference. Regrettably, it is challenging to integrate such a solution with varying light conditions, including shadows and light reflections, manual adjustments, and a physical engine. Recently, a few approaches have appeared that incorporate ray-tracing or mesh primitives into GS to address some of these caveats. However, no such solution can simultaneously solve all the existing limitations of the classical GS. Consequently, we introduce REdiSplats, which employs ray tracing and a mesh-based representation of flat 3D Gaussians. In practice, we model the scene using flat Gaussian distributions parameterized by the mesh. We can leverage fast ray tracing and control Gaussian modification by adjusting the mesh vertices. Moreover, REdiSplats allows modeling of light conditions, manual adjustments, and physical simulation. Furthermore, we can render our models using 3D tools such as Blender or Nvdiffrast, which opens the possibility of integrating them with all existing 3D graphics techniques dedicated to mesh representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10148v2">3D Student Splatting and Scooping</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) provides a new framework for novel view synthesis, and has spiked a new wave of research in neural rendering and related applications. As 3DGS is becoming a foundational component of many models, any improvement on 3DGS itself can bring huge benefits. To this end, we aim to improve the fundamental paradigm and formulation of 3DGS. We argue that as an unnormalized mixture model, it needs to be neither Gaussians nor splatting. We subsequently propose a new mixture model consisting of flexible Student's t distributions, with both positive (splatting) and negative (scooping) densities. We name our model Student Splatting and Scooping, or SSS. When providing better expressivity, SSS also poses new challenges in learning. Therefore, we also propose a new principled sampling approach for optimization. Through exhaustive evaluation and comparison, across multiple datasets, settings, and metrics, we demonstrate that SSS outperforms existing methods in terms of quality and parameter efficiency, e.g. achieving matching or better quality with similar numbers of components, and obtaining comparable results while reducing the component number by as much as 82%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04082v2">Instrument-Splatting: Controllable Photorealistic Reconstruction of Surgical Instruments Using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-15
      | 💬 11 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Real2Sim is becoming increasingly important with the rapid development of surgical artificial intelligence (AI) and autonomy. In this work, we propose a novel Real2Sim methodology, Instrument-Splatting, that leverages 3D Gaussian Splatting to provide fully controllable 3D reconstruction of surgical instruments from monocular surgical videos. To maintain both high visual fidelity and manipulability, we introduce a geometry pre-training to bind Gaussian point clouds on part mesh with accurate geometric priors and define a forward kinematics to control the Gaussians as flexible as real instruments. Afterward, to handle unposed videos, we design a novel instrument pose tracking method leveraging semantics-embedded Gaussians to robustly refine per-frame instrument poses and joint states in a render-and-compare manner, which allows our instrument Gaussian to accurately learn textures and reach photorealistic rendering. We validated our method on 2 publicly released surgical videos and 4 videos collected on ex vivo tissues and green screens. Quantitative and qualitative evaluations demonstrate the effectiveness and superiority of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12001v1">3D Gaussian Splatting against Moving Objects for High-Fidelity Street Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      The accurate reconstruction of dynamic street scenes is critical for applications in autonomous driving, augmented reality, and virtual reality. Traditional methods relying on dense point clouds and triangular meshes struggle with moving objects, occlusions, and real-time processing constraints, limiting their effectiveness in complex urban environments. While multi-view stereo and neural radiance fields have advanced 3D reconstruction, they face challenges in computational efficiency and handling scene dynamics. This paper proposes a novel 3D Gaussian point distribution method for dynamic street scene reconstruction. Our approach introduces an adaptive transparency mechanism that eliminates moving objects while preserving high-fidelity static scene details. Additionally, iterative refinement of Gaussian point distribution enhances geometric accuracy and texture representation. We integrate directional encoding with spatial position optimization to optimize storage and rendering efficiency, reducing redundancy while maintaining scene integrity. Experimental results demonstrate that our method achieves high reconstruction quality, improved rendering performance, and adaptability in large-scale dynamic environments. These contributions establish a robust framework for real-time, high-precision 3D reconstruction, advancing the practicality of dynamic scene modeling across multiple applications. The source code for this work is available to the public at https://github.com/deepcoxcom/3dgs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11981v1">DecompDreamer: Advancing Structured 3D Asset Generation with Multi-Object Decomposition and Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      Text-to-3D generation saw dramatic advances in recent years by leveraging Text-to-Image models. However, most existing techniques struggle with compositional prompts, which describe multiple objects and their spatial relationships. They often fail to capture fine-grained inter-object interactions. We introduce DecompDreamer, a Gaussian splatting-based training routine designed to generate high-quality 3D compositions from such complex prompts. DecompDreamer leverages Vision-Language Models (VLMs) to decompose scenes into structured components and their relationships. We propose a progressive optimization strategy that first prioritizes joint relationship modeling before gradually shifting toward targeted object refinement. Our qualitative and quantitative evaluations against state-of-the-art text-to-3D models demonstrate that DecompDreamer effectively generates intricate 3D compositions with superior object disentanglement, offering enhanced control and flexibility in 3D generation. Project page : https://decompdreamer3d.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11979v1">DynaGSLAM: Real-Time Gaussian-Splatting SLAM for Online Rendering, Tracking, Motion Predictions of Moving Objects in Dynamic Scenes</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      Simultaneous Localization and Mapping (SLAM) is one of the most important environment-perception and navigation algorithms for computer vision, robotics, and autonomous cars/drones. Hence, high quality and fast mapping becomes a fundamental problem. With the advent of 3D Gaussian Splatting (3DGS) as an explicit representation with excellent rendering quality and speed, state-of-the-art (SOTA) works introduce GS to SLAM. Compared to classical pointcloud-SLAM, GS-SLAM generates photometric information by learning from input camera views and synthesize unseen views with high-quality textures. However, these GS-SLAM fail when moving objects occupy the scene that violate the static assumption of bundle adjustment. The failed updates of moving GS affects the static GS and contaminates the full map over long frames. Although some efforts have been made by concurrent works to consider moving objects for GS-SLAM, they simply detect and remove the moving regions from GS rendering ("anti'' dynamic GS-SLAM), where only the static background could benefit from GS. To this end, we propose the first real-time GS-SLAM, "DynaGSLAM'', that achieves high-quality online GS rendering, tracking, motion predictions of moving objects in dynamic scenes while jointly estimating accurate ego motion. Our DynaGSLAM outperforms SOTA static & "Anti'' dynamic GS-SLAM on three dynamic real datasets, while keeping speed and memory efficiency in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11731v1">Industrial-Grade Sensor Simulation via Gaussian Splatting: A Modular Framework for Scalable Editing and Full-Stack Validation</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      Sensor simulation is pivotal for scalable validation of autonomous driving systems, yet existing Neural Radiance Fields (NeRF) based methods face applicability and efficiency challenges in industrial workflows. This paper introduces a Gaussian Splatting (GS) based system to address these challenges: We first break down sensor simulator components and analyze the possible advantages of GS over NeRF. Then in practice, we refactor three crucial components through GS, to leverage its explicit scene representation and real-time rendering: (1) choosing the 2D neural Gaussian representation for physics-compliant scene and sensor modeling, (2) proposing a scene editing pipeline to leverage Gaussian primitives library for data augmentation, and (3) coupling a controllable diffusion model for scene expansion and harmonization. We implement this framework on a proprietary autonomous driving dataset supporting cameras and LiDAR sensors. We demonstrate through ablation studies that our approach reduces frame-wise simulation latency, achieves better geometric and photometric consistency, and enables interpretable explicit scene editing and expansion. Furthermore, we showcase how integrating such a GS-based sensor simulator with traffic and dynamic simulators enables full-stack testing of end-to-end autonomy algorithms. Our work provides both algorithmic insights and practical validation, establishing GS as a cornerstone for industrial-grade sensor simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11601v1">Advancing 3D Gaussian Splatting Editing with Complementary and Consensus Information</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 7 pages, 9 figures
    </div>
    <details class="paper-abstract">
      We present a novel framework for enhancing the visual fidelity and consistency of text-guided 3D Gaussian Splatting (3DGS) editing. Existing editing approaches face two critical challenges: inconsistent geometric reconstructions across multiple viewpoints, particularly in challenging camera positions, and ineffective utilization of depth information during image manipulation, resulting in over-texture artifacts and degraded object boundaries. To address these limitations, we introduce: 1) A complementary information mutual learning network that enhances depth map estimation from 3DGS, enabling precise depth-conditioned 3D editing while preserving geometric structures. 2) A wavelet consensus attention mechanism that effectively aligns latent codes during the diffusion denoising process, ensuring multi-view consistency in the edited results. Through extensive experimentation, our method demonstrates superior performance in rendering quality and view consistency compared to state-of-the-art approaches. The results validate our framework as an effective solution for text-guided editing of 3D scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11345v1">EgoSplat: Open-Vocabulary Egocentric Scene Understanding with Language Embedded 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      Egocentric scenes exhibit frequent occlusions, varied viewpoints, and dynamic interactions compared to typical scene understanding tasks. Occlusions and varied viewpoints can lead to multi-view semantic inconsistencies, while dynamic objects may act as transient distractors, introducing artifacts into semantic feature modeling. To address these challenges, we propose EgoSplat, a language-embedded 3D Gaussian Splatting framework for open-vocabulary egocentric scene understanding. A multi-view consistent instance feature aggregation method is designed to leverage the segmentation and tracking capabilities of SAM2 to selectively aggregate complementary features across views for each instance, ensuring precise semantic representation of scenes. Additionally, an instance-aware spatial-temporal transient prediction module is constructed to improve spatial integrity and temporal continuity in predictions by incorporating spatial-temporal associations across multi-view instances, effectively reducing artifacts in the semantic reconstruction of egocentric scenes. EgoSplat achieves state-of-the-art performance in both localization and segmentation tasks on two datasets, outperforming existing methods with a 8.2% improvement in localization accuracy and a 3.7% improvement in segmentation mIoU on the ADT dataset, and setting a new benchmark in open-vocabulary egocentric scene understanding. The code will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10283v3">GauSTAR: Gaussian Surface Tracking and Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting techniques have enabled efficient photo-realistic rendering of static scenes. Recent works have extended these approaches to support surface reconstruction and tracking. However, tracking dynamic surfaces with 3D Gaussians remains challenging due to complex topology changes, such as surfaces appearing, disappearing, or splitting. To address these challenges, we propose GauSTAR, a novel method that achieves photo-realistic rendering, accurate surface reconstruction, and reliable 3D tracking for general dynamic scenes with changing topology. Given multi-view captures as input, GauSTAR binds Gaussians to mesh faces to represent dynamic objects. For surfaces with consistent topology, GauSTAR maintains the mesh topology and tracks the meshes using Gaussians. For regions where topology changes, GauSTAR adaptively unbinds Gaussians from the mesh, enabling accurate registration and generation of new surfaces based on these optimized Gaussians. Additionally, we introduce a surface-based scene flow method that provides robust initialization for tracking between frames. Experiments demonstrate that our method effectively tracks and reconstructs dynamic surfaces, enabling a range of applications. Our project page with the code release is available at https://eth-ait.github.io/GauSTAR/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05111v2">LiDAR-GS:Real-time LiDAR Re-Simulation using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      We present LiDAR-GS, a Gaussian Splatting (GS) method for real-time, high-fidelity re-simulation of LiDAR scans in public urban road scenes. Recent GS methods proposed for cameras have achieved significant advancements in real-time rendering beyond Neural Radiance Fields (NeRF). However, applying GS representation to LiDAR, an active 3D sensor type, poses several challenges that must be addressed to preserve high accuracy and unique characteristics. Specifically, LiDAR-GS designs a differentiable laser beam splatting, using range-view representation for precise surface splatting by projecting lasers onto micro cross-sections, effectively eliminating artifacts associated with local affine approximations. Furthermore, LiDAR-GS leverages Neural Gaussian Representation, which further integrate view-dependent clues, to represent key LiDAR properties that are influenced by the incident direction and external factors. Combining these practices with some essential adaptations, e.g., dynamic instances decomposition, LiDAR-GS succeeds in simultaneously re-simulating depth, intensity, and ray-drop channels, achieving state-of-the-art results in both rendering frame rate and quality on publically available large scene datasets when compared with the methods using explicit mesh or implicit NeRF. Our source code is publicly available at https://www.github.com/cqf7419/LiDAR-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11172v1">Uncertainty-Aware Normal-Guided Gaussian Splatting for Surface Reconstruction from Sparse Image Sequences</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 12 pages, 8 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has achieved impressive rendering performance in novel view synthesis. However, its efficacy diminishes considerably in sparse image sequences, where inherent data sparsity amplifies geometric uncertainty during optimization. This often leads to convergence at suboptimal local minima, resulting in noticeable structural artifacts in the reconstructed scenes.To mitigate these issues, we propose Uncertainty-aware Normal-Guided Gaussian Splatting (UNG-GS), a novel framework featuring an explicit Spatial Uncertainty Field (SUF) to quantify geometric uncertainty within the 3DGS pipeline. UNG-GS enables high-fidelity rendering and achieves high-precision reconstruction without relying on priors. Specifically, we first integrate Gaussian-based probabilistic modeling into the training of 3DGS to optimize the SUF, providing the model with adaptive error tolerance. An uncertainty-aware depth rendering strategy is then employed to weight depth contributions based on the SUF, effectively reducing noise while preserving fine details. Furthermore, an uncertainty-guided normal refinement method adjusts the influence of neighboring depth values in normal estimation, promoting robust results. Extensive experiments demonstrate that UNG-GS significantly outperforms state-of-the-art methods in both sparse and dense sequences. The code will be open-source.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19895v4">GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-14
      | 💬 This paper is accepted by the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR), 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently created impressive 3D assets for various applications. However, considering security, capacity, invisibility, and training efficiency, the copyright of 3DGS assets is not well protected as existing watermarking methods are unsuited for its rendering pipeline. In this paper, we propose GuardSplat, an innovative and efficient framework for watermarking 3DGS assets. Specifically, 1) We propose a CLIP-guided pipeline for optimizing the message decoder with minimal costs. The key objective is to achieve high-accuracy extraction by leveraging CLIP's aligning capability and rich representations, demonstrating exceptional capacity and efficiency. 2) We tailor a Spherical-Harmonic-aware (SH-aware) Message Embedding module for 3DGS, seamlessly embedding messages into the SH features of each 3D Gaussian while preserving the original 3D structure. This enables watermarking 3DGS assets with minimal fidelity trade-offs and prevents malicious users from removing the watermarks from the model files, meeting the demands for invisibility and security. 3) We present an Anti-distortion Message Extraction module to improve robustness against various distortions. Experiments demonstrate that GuardSplat outperforms state-of-the-art and achieves fast optimization speed. Project page is at https://narcissusex.github.io/GuardSplat, and Code is at https://github.com/NarcissusEx/GuardSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10625v1">LHM: Large Animatable Human Reconstruction Model from a Single Image in Seconds</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 Project Page: https://lingtengqiu.github.io/LHM/
    </div>
    <details class="paper-abstract">
      Animatable 3D human reconstruction from a single image is a challenging problem due to the ambiguity in decoupling geometry, appearance, and deformation. Recent advances in 3D human reconstruction mainly focus on static human modeling, and the reliance of using synthetic 3D scans for training limits their generalization ability. Conversely, optimization-based video methods achieve higher fidelity but demand controlled capture conditions and computationally intensive refinement processes. Motivated by the emergence of large reconstruction models for efficient static reconstruction, we propose LHM (Large Animatable Human Reconstruction Model) to infer high-fidelity avatars represented as 3D Gaussian splatting in a feed-forward pass. Our model leverages a multimodal transformer architecture to effectively encode the human body positional features and image features with attention mechanism, enabling detailed preservation of clothing geometry and texture. To further boost the face identity preservation and fine detail recovery, we propose a head feature pyramid encoding scheme to aggregate multi-scale features of the head regions. Extensive experiments demonstrate that our LHM generates plausible animatable human in seconds without post-processing for face and hands, outperforming existing methods in both reconstruction accuracy and generalization ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10604v1">MuDG: Taming Multi-modal Diffusion with Gaussian Splatting for Urban Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      Recent breakthroughs in radiance fields have significantly advanced 3D scene reconstruction and novel view synthesis (NVS) in autonomous driving. Nevertheless, critical limitations persist: reconstruction-based methods exhibit substantial performance deterioration under significant viewpoint deviations from training trajectories, while generation-based techniques struggle with temporal coherence and precise scene controllability. To overcome these challenges, we present MuDG, an innovative framework that integrates Multi-modal Diffusion model with Gaussian Splatting (GS) for Urban Scene Reconstruction. MuDG leverages aggregated LiDAR point clouds with RGB and geometric priors to condition a multi-modal video diffusion model, synthesizing photorealistic RGB, depth, and semantic outputs for novel viewpoints. This synthesis pipeline enables feed-forward NVS without computationally intensive per-scene optimization, providing comprehensive supervision signals to refine 3DGS representations for rendering robustness enhancement under extreme viewpoint changes. Experiments on the Open Waymo Dataset demonstrate that MuDG outperforms existing methods in both reconstruction and synthesis quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08093v2">MVGSR: Multi-View Consistency Gaussian Splatting for Robust Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 project page https://mvgsr.github.io
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has gained significant attention for its high-quality rendering capabilities, ultra-fast training, and inference speeds. However, when we apply 3DGS to surface reconstruction tasks, especially in environments with dynamic objects and distractors, the method suffers from floating artifacts and color errors due to inconsistency from different viewpoints. To address this challenge, we propose Multi-View Consistency Gaussian Splatting for the domain of Robust Surface Reconstruction (\textbf{MVGSR}), which takes advantage of lightweight Gaussian models and a {heuristics-guided distractor masking} strategy for robust surface reconstruction in non-static environments. Compared to existing methods that rely on MLPs for distractor segmentation strategies, our approach separates distractors from static scene elements by comparing multi-view feature consistency, allowing us to obtain precise distractor masks early in training. Furthermore, we introduce a pruning measure based on multi-view contributions to reset transmittance, effectively reducing floating artifacts. Finally, a multi-view consistency loss is applied to achieve high-quality performance in surface reconstruction tasks. Experimental results demonstrate that MVGSR achieves competitive geometric accuracy and rendering fidelity compared to the state-of-the-art surface reconstruction algorithms. More information is available on our project page (https://mvgsr.github.io).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10437v1">4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 CVPR 2025. Project Page: https://4d-langsplat.github.io
    </div>
    <details class="paper-abstract">
      Learning 4D language fields to enable time-sensitive, open-ended language queries in dynamic scenes is essential for many real-world applications. While LangSplat successfully grounds CLIP features into 3D Gaussian representations, achieving precision and efficiency in 3D static scenes, it lacks the ability to handle dynamic 4D fields as CLIP, designed for static image-text tasks, cannot capture temporal dynamics in videos. Real-world environments are inherently dynamic, with object semantics evolving over time. Building a precise 4D language field necessitates obtaining pixel-aligned, object-wise video features, which current vision models struggle to achieve. To address these challenges, we propose 4D LangSplat, which learns 4D language fields to handle time-agnostic or time-sensitive open-vocabulary queries in dynamic scenes efficiently. 4D LangSplat bypasses learning the language field from vision features and instead learns directly from text generated from object-wise video captions via Multimodal Large Language Models (MLLMs). Specifically, we propose a multimodal object-wise video prompting method, consisting of visual and text prompts that guide MLLMs to generate detailed, temporally consistent, high-quality captions for objects throughout a video. These captions are encoded using a Large Language Model into high-quality sentence embeddings, which then serve as pixel-aligned, object-specific feature supervision, facilitating open-vocabulary text queries through shared embedding spaces. Recognizing that objects in 4D scenes exhibit smooth transitions across states, we further propose a status deformable network to model these continuous changes over time effectively. Our results across multiple benchmarks demonstrate that 4D LangSplat attains precise and efficient results for both time-sensitive and time-agnostic open-vocabulary queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16816v3">SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      Ensuring the safety of autonomous robots, such as self-driving vehicles, requires extensive testing across diverse driving scenarios. Simulation is a key ingredient for conducting such testing in a cost-effective and scalable way. Neural rendering methods have gained popularity, as they can build simulation environments from collected logs in a data-driven manner. However, existing neural radiance field (NeRF) methods for sensor-realistic rendering of camera and lidar data suffer from low rendering speeds, limiting their applicability for large-scale testing. While 3D Gaussian Splatting (3DGS) enables real-time rendering, current methods are limited to camera data and are unable to render lidar data essential for autonomous driving. To address these limitations, we propose SplatAD, the first 3DGS-based method for realistic, real-time rendering of dynamic scenes for both camera and lidar data. SplatAD accurately models key sensor-specific phenomena such as rolling shutter effects, lidar intensity, and lidar ray dropouts, using purpose-built algorithms to optimize rendering efficiency. Evaluation across three autonomous driving datasets demonstrates that SplatAD achieves state-of-the-art rendering quality with up to +2 PSNR for NVS and +3 PSNR for reconstruction while increasing rendering speed over NeRF-based methods by an order of magnitude. See https://research.zenseact.com/publications/splatad/ for our project page.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10286v1">VicaSplat: A Single Run is All You Need for 3D Gaussian Splatting and Camera Estimation from Unposed Video Frames</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      We present VicaSplat, a novel framework for joint 3D Gaussians reconstruction and camera pose estimation from a sequence of unposed video frames, which is a critical yet underexplored task in real-world 3D applications. The core of our method lies in a novel transformer-based network architecture. In particular, our model starts with an image encoder that maps each image to a list of visual tokens. All visual tokens are concatenated with additional inserted learnable camera tokens. The obtained tokens then fully communicate with each other within a tailored transformer decoder. The camera tokens causally aggregate features from visual tokens of different views, and further modulate them frame-wisely to inject view-dependent features. 3D Gaussian splats and camera pose parameters can then be estimated via different prediction heads. Experiments show that VicaSplat surpasses baseline methods for multi-view inputs, and achieves comparable performance to prior two-view approaches. Remarkably, VicaSplat also demonstrates exceptional cross-dataset generalization capability on the ScanNet benchmark, achieving superior performance without any fine-tuning. Project page: https://lizhiqi49.github.io/VicaSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10256v1">ROODI: Reconstructing Occluded Objects with Denoising Inpainters</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 Project page: https://yeonjin-chang.github.io/ROODI/
    </div>
    <details class="paper-abstract">
      While the quality of novel-view images has improved dramatically with 3D Gaussian Splatting, extracting specific objects from scenes remains challenging. Isolating individual 3D Gaussian primitives for each object and handling occlusions in scenes remain far from being solved. We propose a novel object extraction method based on two key principles: (1) being object-centric by pruning irrelevant primitives; and (2) leveraging generative inpainting to compensate for missing observations caused by occlusions. For pruning, we analyze the local structure of primitives using K-nearest neighbors, and retain only relevant ones. For inpainting, we employ an off-the-shelf diffusion-based inpainter combined with occlusion reasoning, utilizing the 3D representation of the entire scene. Our findings highlight the crucial synergy between pruning and inpainting, both of which significantly enhance extraction performance. We evaluate our method on a standard real-world dataset and introduce a synthetic dataset for quantitative analysis. Our approach outperforms the state-of-the-art, demonstrating its effectiveness in object extraction from complex scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10170v1">GS-SDF: LiDAR-Augmented Gaussian Splatting and Neural SDF for Geometrically Consistent Rendering and Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      Digital twins are fundamental to the development of autonomous driving and embodied artificial intelligence. However, achieving high-granularity surface reconstruction and high-fidelity rendering remains a challenge. Gaussian splatting offers efficient photorealistic rendering but struggles with geometric inconsistencies due to fragmented primitives and sparse observational data in robotics applications. Existing regularization methods, which rely on render-derived constraints, often fail in complex environments. Moreover, effectively integrating sparse LiDAR data with Gaussian splatting remains challenging. We propose a unified LiDAR-visual system that synergizes Gaussian splatting with a neural signed distance field. The accurate LiDAR point clouds enable a trained neural signed distance field to offer a manifold geometry field, This motivates us to offer an SDF-based Gaussian initialization for physically grounded primitive placement and a comprehensive geometric regularization for geometrically consistent rendering and reconstruction. Experiments demonstrate superior reconstruction accuracy and rendering quality across diverse trajectories. To benefit the community, the codes will be released at https://github.com/hku-mars/GS-SDF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10148v1">3D Student Splatting and Scooping</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) provides a new framework for novel view synthesis, and has spiked a new wave of research in neural rendering and related applications. As 3DGS is becoming a foundational component of many models, any improvement on 3DGS itself can bring huge benefits. To this end, we aim to improve the fundamental paradigm and formulation of 3DGS. We argue that as an unnormalized mixture model, it needs to be neither Gaussians nor splatting. We subsequently propose a new mixture model consisting of flexible Student's t distributions, with both positive (splatting) and negative (scooping) densities. We name our model Student Splatting and Scooping, or SSS. When providing better expressivity, SSS also poses new challenges in learning. Therefore, we also propose a new principled sampling approach for optimization. Through exhaustive evaluation and comparison, across multiple datasets, settings, and metrics, we demonstrate that SSS outperforms existing methods in terms of quality and parameter efficiency, e.g. achieving matching or better quality with similar numbers of components, and obtaining comparable results while reducing the component number by as much as 82%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10143v1">GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 This paper is accepted by CVPR 2025. Project page is available at https://liujf1226.github.io/GaussHDR
    </div>
    <details class="paper-abstract">
      High dynamic range (HDR) novel view synthesis (NVS) aims to reconstruct HDR scenes by leveraging multi-view low dynamic range (LDR) images captured at different exposure levels. Current training paradigms with 3D tone mapping often result in unstable HDR reconstruction, while training with 2D tone mapping reduces the model's capacity to fit LDR images. Additionally, the global tone mapper used in existing methods can impede the learning of both HDR and LDR representations. To address these challenges, we present GaussHDR, which unifies 3D and 2D local tone mapping through 3D Gaussian splatting. Specifically, we design a residual local tone mapper for both 3D and 2D tone mapping that accepts an additional context feature as input. We then propose combining the dual LDR rendering results from both 3D and 2D local tone mapping at the loss level. Finally, recognizing that different scenes may exhibit varying balances between the dual results, we introduce uncertainty learning and use the uncertainties for adaptive modulation. Extensive experiments demonstrate that GaussHDR significantly outperforms state-of-the-art methods in both synthetic and real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00746v3">DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3D-GS) have shown remarkable success in representing 3D scenes and generating high-quality, novel views in real-time. However, 3D-GS and its variants assume that input images are captured based on pinhole imaging and are fully in focus. This assumption limits their applicability, as real-world images often feature shallow depth-of-field (DoF). In this paper, we introduce DoF-Gaussian, a controllable depth-of-field method for 3D-GS. We develop a lens-based imaging model based on geometric optics principles to control DoF effects. To ensure accurate scene geometry, we incorporate depth priors adjusted per scene, and we apply defocus-to-focus adaptation to minimize the gap in the circle of confusion. We also introduce a synthetic dataset to assess refocusing capabilities and the model's ability to learn precise lens parameters. Our framework is customizable and supports various interactive applications. Extensive experiments confirm the effectiveness of our method. Our project is available at https://dof-gaussian.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10860v1">RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 Project page: https://people.engr.tamu.edu/nimak/Papers/RI3D, Code: https://github.com/avinashpaliwal/RI3D
    </div>
    <details class="paper-abstract">
      In this paper, we propose RI3D, a novel 3DGS-based approach that harnesses the power of diffusion models to reconstruct high-quality novel views given a sparse set of input images. Our key contribution is separating the view synthesis process into two tasks of reconstructing visible regions and hallucinating missing regions, and introducing two personalized diffusion models, each tailored to one of these tasks. Specifically, one model ('repair') takes a rendered image as input and predicts the corresponding high-quality image, which in turn is used as a pseudo ground truth image to constrain the optimization. The other model ('inpainting') primarily focuses on hallucinating details in unobserved areas. To integrate these models effectively, we introduce a two-stage optimization strategy: the first stage reconstructs visible areas using the repair model, and the second stage reconstructs missing regions with the inpainting model while ensuring coherence through further optimization. Moreover, we augment the optimization with a novel Gaussian initialization method that obtains per-image depth by combining 3D-consistent and smooth depth with highly detailed relative depth. We demonstrate that by separating the process into two tasks and addressing them with the repair and inpainting models, we produce results with detailed textures in both visible and missing regions that outperform state-of-the-art approaches on a diverse set of scenes with extremely sparse inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13995v3">Avatar Concept Slider: Controllable Editing of Concepts in 3D Human Avatars</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      Text-based editing of 3D human avatars to precisely match user requirements is challenging due to the inherent ambiguity and limited expressiveness of natural language. To overcome this, we propose the Avatar Concept Slider (ACS), a 3D avatar editing method that allows precise editing of semantic concepts in human avatars towards a specified intermediate point between two extremes of concepts, akin to moving a knob along a slider track. To achieve this, our ACS has three designs: Firstly, a Concept Sliding Loss based on linear discriminant analysis to pinpoint the concept-specific axes for precise editing. Secondly, an Attribute Preserving Loss based on principal component analysis for improved preservation of avatar identity during editing. We further propose a 3D Gaussian Splatting primitive selection mechanism based on concept-sensitivity, which updates only the primitives that are the most sensitive to our target concept, to improve efficiency. Results demonstrate that our ACS enables controllable 3D avatar editing, without compromising the avatar quality or its identifying attributes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2411.13753v2">FAST-Splat: Fast, Ambiguity-Free Semantics Transfer in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      We present FAST-Splat for fast, ambiguity-free semantic Gaussian Splatting, which seeks to address the main limitations of existing semantic Gaussian Splatting methods, namely: slow training and rendering speeds; high memory usage; and ambiguous semantic object localization. We take a bottom-up approach in deriving FAST-Splat, dismantling the limitations of closed-set semantic distillation to enable open-set (open-vocabulary) semantic distillation. Ultimately, this key approach enables FAST-Splat to provide precise semantic object localization results, even when prompted with ambiguous user-provided natural-language queries. Further, by exploiting the explicit form of the Gaussian Splatting scene representation to the fullest extent, FAST-Splat retains the remarkable training and rendering speeds of Gaussian Splatting. Precisely, while existing semantic Gaussian Splatting methods distill semantics into a separate neural field or utilize neural models for dimensionality reduction, FAST-Splat directly augments each Gaussian with specific semantic codes, preserving the training, rendering, and memory-usage advantages of Gaussian Splatting over neural field methods. These Gaussian-specific semantic codes, together with a hash-table, enable semantic similarity to be measured with open-vocabulary user prompts and further enable FAST-Splat to respond with unambiguous semantic object labels and $3$D masks, unlike prior methods. In experiments, we demonstrate that FAST-Splat is 6x to 8x faster to train, achieves between 18x to 51x faster rendering speeds, and requires about 6x smaller GPU memory, compared to the best-competing semantic Gaussian Splatting methods. Further, FAST-Splat achieves relatively similar or better semantic segmentation performance compared to existing methods. After the review period, we will provide links to the project website and the codebase.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.08017v3">Fast Feedforward 3D Gaussian Splatting Compression</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 Project Page: https://yihangchen-ee.github.io/project_fcgs/ Code: https://github.com/yihangchen-ee/fcgs/
    </div>
    <details class="paper-abstract">
      With 3D Gaussian Splatting (3DGS) advancing real-time and high-fidelity rendering for novel view synthesis, storage requirements pose challenges for their widespread adoption. Although various compression techniques have been proposed, previous art suffers from a common limitation: for any existing 3DGS, per-scene optimization is needed to achieve compression, making the compression sluggish and slow. To address this issue, we introduce Fast Compression of 3D Gaussian Splatting (FCGS), an optimization-free model that can compress 3DGS representations rapidly in a single feed-forward pass, which significantly reduces compression time from minutes to seconds. To enhance compression efficiency, we propose a multi-path entropy module that assigns Gaussian attributes to different entropy constraint paths for balance between size and fidelity. We also carefully design both inter- and intra-Gaussian context models to remove redundancies among the unstructured Gaussian blobs. Overall, FCGS achieves a compression ratio of over 20X while maintaining fidelity, surpassing most per-scene SOTA optimization-based methods. Our code is available at: https://github.com/YihangChen-ee/FCGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09342v1">GASPACHO: Gaussian Splatting for Controllable Humans and Objects</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      We present GASPACHO: a method for generating photorealistic controllable renderings of human-object interactions. Given a set of multi-view RGB images of human-object interactions, our method reconstructs animatable templates of the human and object as separate sets of Gaussians simultaneously. Different from existing work, which focuses on human reconstruction and ignores objects as background, our method explicitly reconstructs both humans and objects, thereby allowing for controllable renderings of novel human object interactions in different poses from novel-camera viewpoints. During reconstruction, we constrain the Gaussians that generate rendered images to be a linear function of a set of canonical Gaussians. By simply changing the parameters of the linear deformation functions after training, our method can generate renderings of novel human-object interaction in novel poses from novel camera viewpoints. We learn the 3D Gaussian properties of the canonical Gaussians on the underlying 2D manifold of the canonical human and object templates. This in turns requires a canonical object template with a fixed UV unwrapping. To define such an object template, we use a feature based representation to track the object across the multi-view sequence. We further propose an occlusion aware photometric loss that allows for reconstructions under significant occlusions. Several experiments on two human-object datasets - BEHAVE and DNA-Rendering - demonstrate that our method allows for high-quality reconstruction of human and object templates under significant occlusion and the synthesis of controllable renderings of novel human-object interactions in novel human poses from novel camera views.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00746v2">DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3D-GS) have shown remarkable success in representing 3D scenes and generating high-quality, novel views in real-time. However, 3D-GS and its variants assume that input images are captured based on pinhole imaging and are fully in focus. This assumption limits their applicability, as real-world images often feature shallow depth-of-field (DoF). In this paper, we introduce DoF-Gaussian, a controllable depth-of-field method for 3D-GS. We develop a lens-based imaging model based on geometric optics principles to control DoF effects. To ensure accurate scene geometry, we incorporate depth priors adjusted per scene, and we apply defocus-to-focus adaptation to minimize the gap in the circle of confusion. We also introduce a synthetic dataset to assess refocusing capabilities and the model's ability to learn precise lens parameters. Our framework is customizable and supports various interactive applications. Extensive experiments confirm the effectiveness of our method. Our project is available at https://dof-gaussian.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09332v1">SDD-4DGS: Static-Dynamic Aware Decoupling in Gaussian Splatting for 4D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Dynamic and static components in scenes often exhibit distinct properties, yet most 4D reconstruction methods treat them indiscriminately, leading to suboptimal performance in both cases. This work introduces SDD-4DGS, the first framework for static-dynamic decoupled 4D scene reconstruction based on Gaussian Splatting. Our approach is built upon a novel probabilistic dynamic perception coefficient that is naturally integrated into the Gaussian reconstruction pipeline, enabling adaptive separation of static and dynamic components. With carefully designed implementation strategies to realize this theoretical framework, our method effectively facilitates explicit learning of motion patterns for dynamic elements while maintaining geometric stability for static structures. Extensive experiments on five benchmark datasets demonstrate that SDD-4DGS consistently outperforms state-of-the-art methods in reconstruction fidelity, with enhanced detail restoration for static structures and precise modeling of dynamic motions. The code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01425v2">EVA-Gaussian: 3D Gaussian-based Real-time Human Novel View Synthesis under Diverse Multi-view Camera Settings</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Feed-forward based 3D Gaussian Splatting methods have demonstrated exceptional capability in real-time novel view synthesis for human models. However, current approaches are confined to either dense viewpoint configurations or restricted image resolutions. These limitations hinder their flexibility in free-viewpoint rendering across a wide range of camera view angle discrepancies, and also restrict their ability to recover fine-grained human details in real time using commonly available GPUs. To address these challenges, we propose a novel pipeline named EVA-Gaussian for 3D human novel view synthesis across diverse multi-view camera settings. Specifically, we first design an Efficient Cross-View Attention (EVA) module to effectively fuse cross-view information under high resolution inputs and sparse view settings, while minimizing temporal and computational overhead. Additionally, we introduce a feature refinement mechianism to predict the attributes of the 3D Gaussians and assign a feature value to each Gaussian, enabling the correction of artifacts caused by geometric inaccuracies in position estimation and enhancing overall visual fidelity. Experimental results on the THuman2.0 and THumansit datasets showcase the superiority of EVA-Gaussian in rendering quality across diverse camera settings. Project page: https://zhenliuzju.github.io/huyingdong/EVA-Gaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06838v4">Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Implicit Neural Representation (INR) has been successfully employed for Arbitrary-scale Super-Resolution (ASR). However, INR-based models need to query the multi-layer perceptron module numerous times and render a pixel in each query, resulting in insufficient representation capability and computational efficiency. Recently, Gaussian Splatting (GS) has shown its advantages over INR in both visual quality and rendering speed in 3D tasks, which motivates us to explore whether GS can be employed for the ASR task. However, directly applying GS to ASR is exceptionally challenging because the original GS is an optimization-based method through overfitting each single scene, while in ASR we aim to learn a single model that can generalize to different images and scaling factors. We overcome these challenges by developing two novel techniques. Firstly, to generalize GS for ASR, we elaborately design an architecture to predict the corresponding image-conditioned Gaussians of the input low-resolution image in a feed-forward manner. Each Gaussian can fit the shape and direction of an area of complex textures, showing powerful representation capability. Secondly, we implement an efficient differentiable 2D GPU/CUDA-based scale-aware rasterization to render super-resolved images by sampling discrete RGB values from the predicted continuous Gaussians. Via end-to-end training, our optimized network, namely GSASR, can perform ASR for any image and unseen scaling factors. Extensive experiments validate the effectiveness of our proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05757v3">Locality-aware Gaussian Compression for Fast and High-quality Rendering</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 Accepted to ICLR 2025. Project page: https://seungjooshin.github.io/LocoGS
    </div>
    <details class="paper-abstract">
      We present LocoGS, a locality-aware 3D Gaussian Splatting (3DGS) framework that exploits the spatial coherence of 3D Gaussians for compact modeling of volumetric scenes. To this end, we first analyze the local coherence of 3D Gaussian attributes, and propose a novel locality-aware 3D Gaussian representation that effectively encodes locally-coherent Gaussian attributes using a neural field representation with a minimal storage requirement. On top of the novel representation, LocoGS is carefully designed with additional components such as dense initialization, an adaptive spherical harmonics bandwidth scheme and different encoding schemes for different Gaussian attributes to maximize compression performance. Experimental results demonstrate that our approach outperforms the rendering quality of existing compact Gaussian representations for representative real-world 3D datasets while achieving from 54.6$\times$ to 96.6$\times$ compressed storage size and from 2.1$\times$ to 2.4$\times$ rendering speed than 3DGS. Even our approach also demonstrates an averaged 2.4$\times$ higher rendering speed than the state-of-the-art compression method with comparable compression performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02104v2">DynOMo: Online Point Tracking by Dynamic Online Monocular Gaussian Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 Accepted to 3DV 2025
    </div>
    <details class="paper-abstract">
      Reconstructing scenes and tracking motion are two sides of the same coin. Tracking points allow for geometric reconstruction [14], while geometric reconstruction of (dynamic) scenes allows for 3D tracking of points over time [24, 39]. The latter was recently also exploited for 2D point tracking to overcome occlusion ambiguities by lifting tracking directly into 3D [38]. However, above approaches either require offline processing or multi-view camera setups both unrealistic for real-world applications like robot navigation or mixed reality. We target the challenge of online 2D and 3D point tracking from unposed monocular camera input introducing Dynamic Online Monocular Reconstruction (DynOMo). We leverage 3D Gaussian splatting to reconstruct dynamic scenes in an online fashion. Our approach extends 3D Gaussians to capture new content and object motions while estimating camera movements from a single RGB frame. DynOMo stands out by enabling emergence of point trajectories through robust image feature reconstruction and a novel similarity-enhanced regularization term, without requiring any correspondence-level supervision. It sets the first baseline for online point tracking with monocular unposed cameras, achieving performance on par with existing methods. We aim to inspire the community to advance online point tracking and reconstruction, expanding the applicability to diverse real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19800v2">TrackGS: Optimizing COLMAP-Free 3D Gaussian Splatting with Global Track Constraints</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has advanced ability on novel view synthesis, it still depends on accurate pre-computaed camera parameters, which are hard to obtain and prone to noise. Previous COLMAP-Free methods optimize camera poses using local constraints, but they often struggle in complex scenarios. To address this, we introduce TrackGS, which incorporates feature tracks to globally constrain multi-view geometry. We select the Gaussians associated with each track, which will be trained and rescaled to an infinitesimally small size to guarantee the spatial accuracy. We also propose minimizing both reprojection and backprojection errors for better geometric consistency. Moreover, by deriving the gradient of intrinsics, we unify camera parameter estimation with 3DGS training into a joint optimization framework, achieving SOTA performance on challenging datasets with severe camera movements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07446v2">EigenGS Representation: From Eigenspace to Gaussian Image Space</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Principal Component Analysis (PCA), a classical dimensionality reduction technique, and 2D Gaussian representation, an adaptation of 3D Gaussian Splatting for image representation, offer distinct approaches to modeling visual data. We present EigenGS, a novel method that bridges these paradigms through an efficient transformation pipeline connecting eigenspace and image-space Gaussian representations. Our approach enables instant initialization of Gaussian parameters for new images without requiring per-image optimization from scratch, dramatically accelerating convergence. EigenGS introduces a frequency-aware learning mechanism that encourages Gaussians to adapt to different scales, effectively modeling varied spatial frequencies and preventing artifacts in high-resolution reconstruction. Extensive experiments demonstrate that EigenGS not only achieves superior reconstruction quality compared to direct 2D Gaussian fitting but also reduces necessary parameter count and training time. The results highlight EigenGS's effectiveness and generalization ability across images with varying resolutions and diverse categories, making Gaussian-based image representation both high-quality and viable for real-time applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09040v1">Motion Blender Gaussian Splatting for Dynamic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Gaussian splatting has emerged as a powerful tool for high-fidelity reconstruction of dynamic scenes. However, existing methods primarily rely on implicit motion representations, such as encoding motions into neural networks or per-Gaussian parameters, which makes it difficult to further manipulate the reconstructed motions. This lack of explicit controllability limits existing methods to replaying recorded motions only, which hinders a wider application. To address this, we propose Motion Blender Gaussian Splatting (MB-GS), a novel framework that uses motion graph as an explicit and sparse motion representation. The motion of graph links is propagated to individual Gaussians via dual quaternion skinning, with learnable weight painting functions determining the influence of each link. The motion graphs and 3D Gaussians are jointly optimized from input videos via differentiable rendering. Experiments show that MB-GS achieves state-of-the-art performance on the iPhone dataset while being competitive on HyperNeRF. Additionally, we demonstrate the application potential of our method in generating novel object motions and robot demonstrations through motion editing. Video demonstrations can be found at https://mlzxy.github.io/mbgs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11057v3">CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 Accepted to CVPR 2025
    </div>
    <details class="paper-abstract">
      We propose Covisibility Map-based Gaussian Splatting (CoMapGS), designed to recover underrepresented sparse regions in sparse novel view synthesis. CoMapGS addresses both high- and low-uncertainty regions by constructing covisibility maps, enhancing initial point clouds, and applying uncertainty-aware weighted supervision with a proximity classifier. Our contributions are threefold: (1) CoMapGS reframes novel view synthesis by leveraging covisibility maps as a core component to address region-specific uncertainty levels; (2) Enhanced initial point clouds for both low- and high-uncertainty regions compensate for sparse COLMAP-derived point clouds, improving reconstruction quality and benefiting few-shot 3DGS methods; (3) Adaptive supervision with covisibility-score-based weighting and proximity classification achieves consistent performance gains across scenes with various sparsity scores derived from covisibility maps. Experimental results demonstrate that CoMapGS outperforms state-of-the-art methods on datasets including Mip-NeRF 360 and LLFF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00144v3">Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated remarkable effectiveness in novel view synthesis (NVS). However, 3DGS tends to overfit when trained with sparse views, limiting its generalization to novel viewpoints. In this paper, we address this overfitting issue by introducing Self-Ensembling Gaussian Splatting (SE-GS). We achieve self-ensembling by incorporating an uncertainty-aware perturbation strategy during training. A $\mathbf{\Delta}$-model and a $\mathbf{\Sigma}$-model are jointly trained on the available images. The $\mathbf{\Delta}$-model is dynamically perturbed based on rendering uncertainty across training steps, generating diverse perturbed models with negligible computational overhead. Discrepancies between the $\mathbf{\Sigma}$-model and these perturbed models are minimized throughout training, forming a robust ensemble of 3DGS models. This ensemble, represented by the $\mathbf{\Sigma}$-model, is then used to generate novel-view images during inference. Experimental results on the LLFF, Mip-NeRF360, DTU, and MVImgNet datasets demonstrate that our approach enhances NVS quality under few-shot training conditions, outperforming existing state-of-the-art methods. The code is released at: https://sailor-z.github.io/projects/SEGS.html.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05511v2">Free Your Hands: Lightweight Relightable Turntable Capture Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) from multiple captured photos of an object is a widely studied problem. Achieving high quality typically requires dense sampling of input views, which can lead to frustrating and tedious manual labor. Manually positioning cameras to maintain an optimal desired distribution can be difficult for humans, and if a good distribution is found, it is not easy to replicate. Additionally, the captured data can suffer from motion blur and defocus due to human error. In this paper, we present a lightweight object capture pipeline to reduce the manual workload and standardize the acquisition setup. We use a consumer turntable to carry the object and a tripod to hold the camera. As the turntable rotates, we automatically capture dense samples from various views and lighting conditions; we can repeat this for several camera positions. This way, we can easily capture hundreds of valid images in several minutes without hands-on effort. However, in the object reference frame, the light conditions vary; this is harmful to a standard NVS method like 3D Gaussian splatting (3DGS) which assumes fixed lighting. We design a neural radiance representation conditioned on light rotations, which addresses this issue and allows relightability as an additional benefit. We demonstrate our pipeline using 3DGS as the underlying framework, achieving competitive quality compared to previous methods with exhaustive acquisition and showcasing its potential for relighting and harmonization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04981v2">AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Obtaining high-quality 3D semantic occupancy from raw sensor data remains an essential yet challenging task, often requiring extensive manual labeling. In this work, we propose AutoOcc, an vision-centric automated pipeline for open-ended semantic occupancy annotation that integrates differentiable Gaussian splatting guided by vision-language models. We formulate the open-ended semantic occupancy reconstruction task to automatically generate scene occupancy by combining attention maps from vision-language models and foundation vision models. We devise semantic-aware Gaussians as intermediate geometric descriptors and propose a cumulative Gaussian-to-voxel splatting algorithm that enables effective and efficient occupancy annotation. Our framework outperforms existing automated occupancy annotation methods without human labels. AutoOcc also enables open-ended semantic occupancy auto-labeling, achieving robust performance in both static and dynamically complex scenarios. All the source codes and trained models will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06677v2">REArtGS: Reconstructing and Generating Articulated Objects via 3D Gaussian Splatting with Geometric and Motion Constraints</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 11pages, 6 figures
    </div>
    <details class="paper-abstract">
      Articulated objects, as prevalent entities in human life, their 3D representations play crucial roles across various applications. However, achieving both high-fidelity textured surface reconstruction and dynamic generation for articulated objects remains challenging for existing methods. In this paper, we present REArtGS, a novel framework that introduces additional geometric and motion constraints to 3D Gaussian primitives, enabling high-quality textured surface reconstruction and generation for articulated objects. Specifically, given multi-view RGB images of arbitrary two states of articulated objects, we first introduce an unbiased Signed Distance Field (SDF) guidance to regularize Gaussian opacity fields, enhancing geometry constraints and improving surface reconstruction quality. Then we establish deformable fields for 3D Gaussians constrained by the kinematic structures of articulated objects, achieving unsupervised generation of surface meshes in unseen states. Extensive experiments on both synthetic and real datasets demonstrate our approach achieves high-quality textured surface reconstruction for given states, and enables high-fidelity surface generation for unseen states. Codes will be released within the next four months and the project website is at https://sites.google.com/view/reartgs/home.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13753v2">FAST-Splat: Fast, Ambiguity-Free Semantics Transfer in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      We present FAST-Splat for fast, ambiguity-free semantic Gaussian Splatting, which seeks to address the main limitations of existing semantic Gaussian Splatting methods, namely: slow training and rendering speeds; high memory usage; and ambiguous semantic object localization. We take a bottom-up approach in deriving FAST-Splat, dismantling the limitations of closed-set semantic distillation to enable open-set (open-vocabulary) semantic distillation. Ultimately, this key approach enables FAST-Splat to provide precise semantic object localization results, even when prompted with ambiguous user-provided natural-language queries. Further, by exploiting the explicit form of the Gaussian Splatting scene representation to the fullest extent, FAST-Splat retains the remarkable training and rendering speeds of Gaussian Splatting. Precisely, while existing semantic Gaussian Splatting methods distill semantics into a separate neural field or utilize neural models for dimensionality reduction, FAST-Splat directly augments each Gaussian with specific semantic codes, preserving the training, rendering, and memory-usage advantages of Gaussian Splatting over neural field methods. These Gaussian-specific semantic codes, together with a hash-table, enable semantic similarity to be measured with open-vocabulary user prompts and further enable FAST-Splat to respond with unambiguous semantic object labels and $3$D masks, unlike prior methods. In experiments, we demonstrate that FAST-Splat is 6x to 8x faster to train, achieves between 18x to 51x faster rendering speeds, and requires about 6x smaller GPU memory, compared to the best-competing semantic Gaussian Splatting methods. Further, FAST-Splat achieves relatively similar or better semantic segmentation performance compared to existing methods. After the review period, we will provide links to the project website and the codebase.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10782v2">3DArticCyclists: Generating Synthetic Articulated 8D Pose-Controllable Cyclist Data for Computer Vision Applications</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      In Autonomous Driving (AD) Perception, cyclists are considered safety-critical scene objects. Commonly used publicly-available AD datasets typically contain large amounts of car and vehicle object instances but a low number of cyclist instances, usually with limited appearance and pose diversity. This cyclist training data scarcity problem not only limits the generalization of deep-learning perception models for cyclist semantic segmentation, pose estimation, and cyclist crossing intention prediction, but also limits research on new cyclist-related tasks such as fine-grained cyclist pose estimation and spatio-temporal analysis under complex interactions between humans and articulated objects. To address this data scarcity problem, in this paper we propose a framework to generate synthetic dynamic 3D cyclist data assets that can be used to generate training data for different tasks. In our framework, we designed a methodology for creating a new part-based multi-view articulated synthetic 3D bicycle dataset that we call 3DArticBikes that we use to train a 3D Gaussian Splatting (3DGS)-based reconstruction and image rendering method. We then propose a parametric bicycle 3DGS composition model to assemble 8-DoF pose-controllable 3D bicycles. Finally, using dynamic information from cyclist videos, we build a complete synthetic dynamic 3D cyclist (rider pedaling a bicycle) by re-posing a selectable synthetic 3D person, while automatically placing the rider onto one of our new articulated 3D bicycles using a proposed 3D Keypoint optimization-based Inverse Kinematics pose refinement. We present both, qualitative and quantitative results where we compare our generated cyclists against those from a recent stable diffusion-based method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05256v3">Extrapolated Urban View Synthesis Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 Project page: https://ai4ce.github.io/EUVS-Benchmark/
    </div>
    <details class="paper-abstract">
      Photorealistic simulators are essential for the training and evaluation of vision-centric autonomous vehicles (AVs). At their core is Novel View Synthesis (NVS), a crucial capability that generates diverse unseen viewpoints to accommodate the broad and continuous pose distribution of AVs. Recent advances in radiance fields, such as 3D Gaussian Splatting, achieve photorealistic rendering at real-time speeds and have been widely used in modeling large-scale driving scenes. However, their performance is commonly evaluated using an interpolated setup with highly correlated training and test views. In contrast, extrapolation, where test views largely deviate from training views, remains underexplored, limiting progress in generalizable simulation technology. To address this gap, we leverage publicly available AV datasets with multiple traversals, multiple vehicles, and multiple cameras to build the first Extrapolated Urban View Synthesis (EUVS) benchmark. Meanwhile, we conduct both quantitative and qualitative evaluations of state-of-the-art NVS methods across different evaluation settings. Our results show that current NVS methods are prone to overfitting to training views. Besides, incorporating diffusion priors and improving geometry cannot fundamentally improve NVS under large view changes, highlighting the need for more robust approaches and large-scale training. We will release the data to help advance self-driving and urban robotics simulation technology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09640v1">Physics-Aware Human-Object Rendering from Sparse Views via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Rendering realistic human-object interactions (HOIs) from sparse-view inputs is challenging due to occlusions and incomplete observations, yet crucial for various real-world applications. Existing methods always struggle with either low rendering qualities (\eg, visual fidelity and physically plausible HOIs) or high computational costs. To address these limitations, we propose HOGS (Human-Object Rendering via 3D Gaussian Splatting), a novel framework for efficient and physically plausible HOI rendering from sparse views. Specifically, HOGS combines 3D Gaussian Splatting with a physics-aware optimization process. It incorporates a Human Pose Refinement module for accurate pose estimation and a Sparse-View Human-Object Contact Prediction module for efficient contact region identification. This combination enables coherent joint rendering of human and object Gaussians while enforcing physically plausible interactions. Extensive experiments on the HODome dataset demonstrate that HOGS achieves superior rendering quality, efficiency, and physical plausibility compared to existing methods. We further show its extensibility to hand-object grasp rendering tasks, presenting its broader applicability to articulated object interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09464v1">Hybrid Rendering for Multimodal Autonomous Driving: Merging Neural and Physics-Based Simulation</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Neural reconstruction models for autonomous driving simulation have made significant strides in recent years, with dynamic models becoming increasingly prevalent. However, these models are typically limited to handling in-domain objects closely following their original trajectories. We introduce a hybrid approach that combines the strengths of neural reconstruction with physics-based rendering. This method enables the virtual placement of traditional mesh-based dynamic agents at arbitrary locations, adjustments to environmental conditions, and rendering from novel camera viewpoints. Our approach significantly enhances novel view synthesis quality -- especially for road surfaces and lane markings -- while maintaining interactive frame rates through our novel training method, NeRF2GS. This technique leverages the superior generalization capabilities of NeRF-based methods and the real-time rendering speed of 3D Gaussian Splatting (3DGS). We achieve this by training a customized NeRF model on the original images with depth regularization derived from a noisy LiDAR point cloud, then using it as a teacher model for 3DGS training. This process ensures accurate depth, surface normals, and camera appearance modeling as supervision. With our block-based training parallelization, the method can handle large-scale reconstructions (greater than or equal to 100,000 square meters) and predict segmentation masks, surface normals, and depth maps. During simulation, it supports a rasterization-based rendering backend with depth-based composition and multiple camera models for real-time camera simulation, as well as a ray-traced backend for precise LiDAR simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09447v1">Online Language Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      To enable AI agents to interact seamlessly with both humans and 3D environments, they must not only perceive the 3D world accurately but also align human language with 3D spatial representations. While prior work has made significant progress by integrating language features into geometrically detailed 3D scene representations using 3D Gaussian Splatting (GS), these approaches rely on computationally intensive offline preprocessing of language features for each input image, limiting adaptability to new environments. In this work, we introduce Online Language Splatting, the first framework to achieve online, near real-time, open-vocabulary language mapping within a 3DGS-SLAM system without requiring pre-generated language features. The key challenge lies in efficiently fusing high-dimensional language features into 3D representations while balancing the computation speed, memory usage, rendering quality and open-vocabulary capability. To this end, we innovatively design: (1) a high-resolution CLIP embedding module capable of generating detailed language feature maps in 18ms per frame, (2) a two-stage online auto-encoder that compresses 768-dimensional CLIP features to 15 dimensions while preserving open-vocabulary capabilities, and (3) a color-language disentangled optimization approach to improve rendering quality. Experimental results show that our online method not only surpasses the state-of-the-art offline methods in accuracy but also achieves more than 40x efficiency boost, demonstrating the potential for dynamic and interactive AI applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08017v3">Fast Feedforward 3D Gaussian Splatting Compression</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 Project Page: https://yihangchen-ee.github.io/project_fcgs/ Code: https://github.com/yihangchen-ee/fcgs/
    </div>
    <details class="paper-abstract">
      With 3D Gaussian Splatting (3DGS) advancing real-time and high-fidelity rendering for novel view synthesis, storage requirements pose challenges for their widespread adoption. Although various compression techniques have been proposed, previous art suffers from a common limitation: for any existing 3DGS, per-scene optimization is needed to achieve compression, making the compression sluggish and slow. To address this issue, we introduce Fast Compression of 3D Gaussian Splatting (FCGS), an optimization-free model that can compress 3DGS representations rapidly in a single feed-forward pass, which significantly reduces compression time from minutes to seconds. To enhance compression efficiency, we propose a multi-path entropy module that assigns Gaussian attributes to different entropy constraint paths for balance between size and fidelity. We also carefully design both inter- and intra-Gaussian context models to remove redundancies among the unstructured Gaussian blobs. Overall, FCGS achieves a compression ratio of over 20X while maintaining fidelity, surpassing most per-scene SOTA optimization-based methods. Our code is available at: https://github.com/YihangChen-ee/FCGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09396v1">Close-up-GS: Enhancing Close-Up View Synthesis in 3D Gaussian Splatting with Progressive Self-Training</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated impressive performance in synthesizing novel views after training on a given set of viewpoints. However, its rendering quality deteriorates when the synthesized view deviates significantly from the training views. This decline occurs due to (1) the model's difficulty in generalizing to out-of-distribution scenarios and (2) challenges in interpolating fine details caused by substantial resolution changes and occlusions. A notable case of this limitation is close-up view generation--producing views that are significantly closer to the object than those in the training set. To tackle this issue, we propose a novel approach for close-up view generation based by progressively training the 3DGS model with self-generated data. Our solution is based on three key ideas. First, we leverage the See3D model, a recently introduced 3D-aware generative model, to enhance the details of rendered views. Second, we propose a strategy to progressively expand the ``trust regions'' of the 3DGS model and update a set of reference views for See3D. Finally, we introduce a fine-tuning strategy to carefully update the 3DGS model with training data generated from the above schemes. We further define metrics for close-up views evaluation to facilitate better research on this problem. By conducting evaluations on specifically selected scenarios for close-up views, our proposed approach demonstrates a clear advantage over competitive solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2503.07946v1">7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Real-time rendering of dynamic scenes with view-dependent effects remains a fundamental challenge in computer graphics. While recent advances in Gaussian Splatting have shown promising results separately handling dynamic scenes (4DGS) and view-dependent effects (6DGS), no existing method unifies these capabilities while maintaining real-time performance. We present 7D Gaussian Splatting (7DGS), a unified framework representing scene elements as seven-dimensional Gaussians spanning position (3D), time (1D), and viewing direction (3D). Our key contribution is an efficient conditional slicing mechanism that transforms 7D Gaussians into view- and time-conditioned 3D Gaussians, maintaining compatibility with existing 3D Gaussian Splatting pipelines while enabling joint optimization. Experiments demonstrate that 7DGS outperforms prior methods by up to 7.36 dB in PSNR while achieving real-time rendering (401 FPS) on challenging dynamic scenes with complex view-dependent effects. The project page is: https://gaozhongpai.github.io/7dgs/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2503.08511v1">PCGS: Progressive Compression of 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 Project Page: https://yihangchen-ee.github.io/project_pcgs/ Code: https://github.com/YihangChen-ee/PCGS
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves impressive rendering fidelity and speed for novel view synthesis. However, its substantial data size poses a significant challenge for practical applications. While many compression techniques have been proposed, they fail to efficiently utilize existing bitstreams in on-demand applications due to their lack of progressivity, leading to a waste of resource. To address this issue, we propose PCGS (Progressive Compression of 3D Gaussian Splatting), which adaptively controls both the quantity and quality of Gaussians (or anchors) to enable effective progressivity for on-demand applications. Specifically, for quantity, we introduce a progressive masking strategy that incrementally incorporates new anchors while refining existing ones to enhance fidelity. For quality, we propose a progressive quantization approach that gradually reduces quantization step sizes to achieve finer modeling of Gaussian attributes. Furthermore, to compact the incremental bitstreams, we leverage existing quantization results to refine probability prediction, improving entropy coding efficiency across progressive levels. Overall, PCGS achieves progressivity while maintaining compression performance comparable to SoTA non-progressive methods. Code available at: github.com/YihangChen-ee/PCGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.04974v3">6DGS: Enhanced Direction-Aware Gaussian Splatting for Volumetric Rendering</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 Accepted by ICLR2025
    </div>
    <details class="paper-abstract">
      Novel view synthesis has advanced significantly with the development of neural radiance fields (NeRF) and 3D Gaussian splatting (3DGS). However, achieving high quality without compromising real-time rendering remains challenging, particularly for physically-based ray tracing with view-dependent effects. Recently, N-dimensional Gaussians (N-DG) introduced a 6D spatial-angular representation to better incorporate view-dependent effects, but the Gaussian representation and control scheme are sub-optimal. In this paper, we revisit 6D Gaussians and introduce 6D Gaussian Splatting (6DGS), which enhances color and opacity representations and leverages the additional directional information in the 6D space for optimized Gaussian control. Our approach is fully compatible with the 3DGS framework and significantly improves real-time radiance field rendering by better modeling view-dependent effects and fine details. Experiments demonstrate that 6DGS significantly outperforms 3DGS and N-DG, achieving up to a 15.73 dB improvement in PSNR with a reduction of 66.5% Gaussian points compared to 3DGS. The project page is: https://gaozhongpai.github.io/6dgs/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08093v1">MVGSR: Multi-View Consistency Gaussian Splatting for Robust Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 project page \url{https://mvgsr.github.io}
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has gained significant attention for its high-quality rendering capabilities, ultra-fast training, and inference speeds. However, when we apply 3DGS to surface reconstruction tasks, especially in environments with dynamic objects and distractors, the method suffers from floating artifacts and color errors due to inconsistency from different viewpoints. To address this challenge, we propose Multi-View Consistency Gaussian Splatting for the domain of Robust Surface Reconstruction (\textbf{MVGSR}), which takes advantage of lightweight Gaussian models and a {heuristics-guided distractor masking} strategy for robust surface reconstruction in non-static environments. Compared to existing methods that rely on MLPs for distractor segmentation strategies, our approach separates distractors from static scene elements by comparing multi-view feature consistency, allowing us to obtain precise distractor masks early in training. Furthermore, we introduce a pruning measure based on multi-view contributions to reset transmittance, effectively reducing floating artifacts. Finally, a multi-view consistency loss is applied to achieve high-quality performance in surface reconstruction tasks. Experimental results demonstrate that MVGSR achieves competitive geometric accuracy and rendering fidelity compared to the state-of-the-art surface reconstruction algorithms. More information is available on our project page (\href{https://mvgsr.github.io}{this url})
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08071v1">GigaSLAM: Large-Scale Monocular SLAM with Hierachical Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Tracking and mapping in large-scale, unbounded outdoor environments using only monocular RGB input presents substantial challenges for existing SLAM systems. Traditional Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) SLAM methods are typically limited to small, bounded indoor settings. To overcome these challenges, we introduce GigaSLAM, the first NeRF/3DGS-based SLAM framework for kilometer-scale outdoor environments, as demonstrated on the KITTI and KITTI 360 datasets. Our approach employs a hierarchical sparse voxel map representation, where Gaussians are decoded by neural networks at multiple levels of detail. This design enables efficient, scalable mapping and high-fidelity viewpoint rendering across expansive, unbounded scenes. For front-end tracking, GigaSLAM utilizes a metric depth model combined with epipolar geometry and PnP algorithms to accurately estimate poses, while incorporating a Bag-of-Words-based loop closure mechanism to maintain robust alignment over long trajectories. Consequently, GigaSLAM delivers high-precision tracking and visually faithful rendering on urban outdoor benchmarks, establishing a robust SLAM solution for large-scale, long-term scenarios, and significantly extending the applicability of Gaussian Splatting SLAM systems to unbounded outdoor environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22070v2">FreeGaussian: Annotation-free Controllable 3D Gaussian Splats with Flow Derivatives</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Reconstructing controllable Gaussian splats from monocular video is a challenging task due to its inherently insufficient constraints. Widely adopted approaches supervise complex interactions with additional masks and control signal annotations, limiting their real-world applications. In this paper, we propose an annotation guidance-free method, dubbed FreeGaussian, that mathematically derives dynamic Gaussian motion from optical flow and camera motion using novel dynamic Gaussian constraints. By establishing a connection between 2D flows and 3D Gaussian dynamic control, our method enables self-supervised optimization and continuity of dynamic Gaussian motions from flow priors. Furthermore, we introduce a 3D spherical vector controlling scheme, which represents the state with a 3D Gaussian trajectory, thereby eliminating the need for complex 1D control signal calculations and simplifying controllable Gaussian modeling. Quantitative and qualitative evaluations on extensive experiments demonstrate the state-of-the-art visual performance and control capability of our method. Project page: https://freegaussian.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12789v2">Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D generation models have opened new possibilities for simulating dynamic 3D object movements and customizing behaviors, yet creating this content remains challenging. Current methods often require manual assignment of precise physical properties for simulations or rely on video generation models to predict them, which is computationally intensive. In this paper, we rethink the usage of multi-modal large language model (MLLM) in physics-based simulation, and present Sim Anything, a physics-based approach that endows static 3D objects with interactive dynamics. We begin with detailed scene reconstruction and object-level 3D open-vocabulary segmentation, progressing to multi-view image in-painting. Inspired by human visual reasoning, we propose MLLM-based Physical Property Perception (MLLM-P3) to predict mean physical properties of objects in a zero-shot manner. Based on the mean values and the object's geometry, the Material Property Distribution Prediction model (MPDP) model then estimates the full distribution, reformulating the problem as probability distribution estimation to reduce computational costs. Finally, we simulate objects in an open-world scene with particles sampled via the Physical-Geometric Adaptive Sampling (PGAS) strategy, efficiently capturing complex deformations and significantly reducing computational costs. Extensive experiments and user studies demonstrate our Sim Anything achieves more realistic motion than state-of-the-art methods within 2 minutes on a single GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07946v1">7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Real-time rendering of dynamic scenes with view-dependent effects remains a fundamental challenge in computer graphics. While recent advances in Gaussian Splatting have shown promising results separately handling dynamic scenes (4DGS) and view-dependent effects (6DGS), no existing method unifies these capabilities while maintaining real-time performance. We present 7D Gaussian Splatting (7DGS), a unified framework representing scene elements as seven-dimensional Gaussians spanning position (3D), time (1D), and viewing direction (3D). Our key contribution is an efficient conditional slicing mechanism that transforms 7D Gaussians into view- and time-conditioned 3D Gaussians, maintaining compatibility with existing 3D Gaussian Splatting pipelines while enabling joint optimization. Experiments demonstrate that 7DGS outperforms prior methods by up to 7.36 dB in PSNR while achieving real-time rendering (401 FPS) on challenging dynamic scenes with complex view-dependent effects. The project page is: https://gaozhongpai.github.io/7dgs/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04974v3">6DGS: Enhanced Direction-Aware Gaussian Splatting for Volumetric Rendering</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Accepted by ICLR2025
    </div>
    <details class="paper-abstract">
      Novel view synthesis has advanced significantly with the development of neural radiance fields (NeRF) and 3D Gaussian splatting (3DGS). However, achieving high quality without compromising real-time rendering remains challenging, particularly for physically-based ray tracing with view-dependent effects. Recently, N-dimensional Gaussians (N-DG) introduced a 6D spatial-angular representation to better incorporate view-dependent effects, but the Gaussian representation and control scheme are sub-optimal. In this paper, we revisit 6D Gaussians and introduce 6D Gaussian Splatting (6DGS), which enhances color and opacity representations and leverages the additional directional information in the 6D space for optimized Gaussian control. Our approach is fully compatible with the 3DGS framework and significantly improves real-time radiance field rendering by better modeling view-dependent effects and fine details. Experiments demonstrate that 6DGS significantly outperforms 3DGS and N-DG, achieving up to a 15.73 dB improvement in PSNR with a reduction of 66.5% Gaussian points compared to 3DGS. The project page is: https://gaozhongpai.github.io/6dgs/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09635v1">FPGS: Feed-Forward Semantic-aware Photorealistic Style Transfer of Large-Scale Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Project page: https://kim-geonu.github.io/FPGS/. arXiv admin note: substantial text overlap with arXiv:2401.05516
    </div>
    <details class="paper-abstract">
      We present FPGS, a feed-forward photorealistic style transfer method of large-scale radiance fields represented by Gaussian Splatting. FPGS, stylizes large-scale 3D scenes with arbitrary, multiple style reference images without additional optimization while preserving multi-view consistency and real-time rendering speed of 3D Gaussians. Prior arts required tedious per-style optimization or time-consuming per-scene training stage and were limited to small-scale 3D scenes. FPGS efficiently stylizes large-scale 3D scenes by introducing a style-decomposed 3D feature field, which inherits AdaIN's feed-forward stylization machinery, supporting arbitrary style reference images. Furthermore, FPGS supports multi-reference stylization with the semantic correspondence matching and local AdaIN, which adds diverse user control for 3D scene styles. FPGS also preserves multi-view consistency by applying semantic matching and style transfer processes directly onto queried features in 3D space. In experiments, we demonstrate that FPGS achieves favorable photorealistic quality scene stylization for large-scale static and dynamic 3D scenes with diverse reference images. Project page: https://kim-geonu.github.io/FPGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.07199v2">RealmDreamer: Text-Driven 3D Scene Generation with Inpainting and Depth Diffusion</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Published at 3DV 2025
    </div>
    <details class="paper-abstract">
      We introduce RealmDreamer, a technique for generating forward-facing 3D scenes from text descriptions. Our method optimizes a 3D Gaussian Splatting representation to match complex text prompts using pretrained diffusion models. Our key insight is to leverage 2D inpainting diffusion models conditioned on an initial scene estimate to provide low variance supervision for unknown regions during 3D distillation. In conjunction, we imbue high-fidelity geometry with geometric distillation from a depth diffusion model, conditioned on samples from the inpainting model. We find that the initialization of the optimization is crucial, and provide a principled methodology for doing so. Notably, our technique doesn't require video or multi-view data and can synthesize various high-quality 3D scenes in different styles with complex layouts. Further, the generality of our method allows 3D synthesis from a single image. As measured by a comprehensive user study, our method outperforms all existing approaches, preferred by 88-95%. Project Page: https://realmdreamer.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08511v1">PCGS: Progressive Compression of 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Project Page: https://yihangchen-ee.github.io/project_pcgs/ Code: https://github.com/YihangChen-ee/PCGS
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves impressive rendering fidelity and speed for novel view synthesis. However, its substantial data size poses a significant challenge for practical applications. While many compression techniques have been proposed, they fail to efficiently utilize existing bitstreams in on-demand applications due to their lack of progressivity, leading to a waste of resource. To address this issue, we propose PCGS (Progressive Compression of 3D Gaussian Splatting), which adaptively controls both the quantity and quality of Gaussians (or anchors) to enable effective progressivity for on-demand applications. Specifically, for quantity, we introduce a progressive masking strategy that incrementally incorporates new anchors while refining existing ones to enhance fidelity. For quality, we propose a progressive quantization approach that gradually reduces quantization step sizes to achieve finer modeling of Gaussian attributes. Furthermore, to compact the incremental bitstreams, we leverage existing quantization results to refine probability prediction, improving entropy coding efficiency across progressive levels. Overall, PCGS achieves progressivity while maintaining compression performance comparable to SoTA non-progressive methods. Code available at: github.com/YihangChen-ee/PCGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08485v1">TT-GaussOcc: Test-Time Compute for Self-Supervised Occupancy Prediction via Spatio-Temporal Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Self-supervised 3D occupancy prediction offers a promising solution for understanding complex driving scenes without requiring costly 3D annotations. However, training dense voxel decoders to capture fine-grained geometry and semantics can demand hundreds of GPU hours, and such models often fail to adapt to varying voxel resolutions or new classes without extensive retraining. To overcome these limitations, we propose a practical and flexible test-time occupancy prediction framework termed TT-GaussOcc. Our approach incrementally optimizes time-aware 3D Gaussians instantiated from raw sensor streams at runtime, enabling voxelization at arbitrary user-specified resolution. Specifically, TT-GaussOcc operates in a "lift-move-voxel" symphony: we first "lift" surrounding-view semantics obtained from 2D vision foundation models (VLMs) to instantiate Gaussians at non-empty 3D space; Next, we "move" dynamic Gaussians from previous frames along estimated Gaussian scene flow to complete appearance and eliminate trailing artifacts of fast-moving objects, while accumulating static Gaussians to enforce temporal consistency; Finally, we mitigate inherent noises in semantic predictions and scene flow vectors by periodically smoothing neighboring Gaussians during optimization, using proposed trilateral RBF kernels that jointly consider color, semantic, and spatial affinities. The historical static and current dynamic Gaussians are then combined and voxelized to generate occupancy prediction. Extensive experiments on Occ3D and nuCraft with varying voxel resolutions demonstrate that TT-GaussOcc surpasses self-supervised baselines by 46% on mIoU without any offline training, and supports finer voxel resolutions at 2.6 FPS inference speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08352v1">Mitigating Ambiguities in 3D Classification with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      3D classification with point cloud input is a fundamental problem in 3D vision. However, due to the discrete nature and the insufficient material description of point cloud representations, there are ambiguities in distinguishing wire-like and flat surfaces, as well as transparent or reflective objects. To address these issues, we propose Gaussian Splatting (GS) point cloud-based 3D classification. We find that the scale and rotation coefficients in the GS point cloud help characterize surface types. Specifically, wire-like surfaces consist of multiple slender Gaussian ellipsoids, while flat surfaces are composed of a few flat Gaussian ellipsoids. Additionally, the opacity in the GS point cloud represents the transparency characteristics of objects. As a result, ambiguities in point cloud-based 3D classification can be mitigated utilizing GS point cloud as input. To verify the effectiveness of GS point cloud input, we construct the first real-world GS point cloud dataset in the community, which includes 20 categories with 200 objects in each category. Experiments not only validate the superiority of GS point cloud input, especially in distinguishing ambiguous objects, but also demonstrate the generalization ability across different classification methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08317v1">Uni-Gaussians: Unifying Camera and Lidar Simulation with Gaussians for Dynamic Driving Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Ensuring the safety of autonomous vehicles necessitates comprehensive simulation of multi-sensor data, encompassing inputs from both cameras and LiDAR sensors, across various dynamic driving scenarios. Neural rendering techniques, which utilize collected raw sensor data to simulate these dynamic environments, have emerged as a leading methodology. While NeRF-based approaches can uniformly represent scenes for rendering data from both camera and LiDAR, they are hindered by slow rendering speeds due to dense sampling. Conversely, Gaussian Splatting-based methods employ Gaussian primitives for scene representation and achieve rapid rendering through rasterization. However, these rasterization-based techniques struggle to accurately model non-linear optical sensors. This limitation restricts their applicability to sensors beyond pinhole cameras. To address these challenges and enable unified representation of dynamic driving scenarios using Gaussian primitives, this study proposes a novel hybrid approach. Our method utilizes rasterization for rendering image data while employing Gaussian ray-tracing for LiDAR data rendering. Experimental results on public datasets demonstrate that our approach outperforms current state-of-the-art methods. This work presents a unified and efficient solution for realistic simulation of camera and LiDAR data in autonomous driving scenarios using Gaussian primitives, offering significant advancements in both rendering quality and computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08305v1">ELECTRA: A Symmetry-breaking Cartesian Network for Charge Density Prediction with Floating Orbitals</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 8 pages, 3 figures, 1 table
    </div>
    <details class="paper-abstract">
      We present the Electronic Tensor Reconstruction Algorithm (ELECTRA) - an equivariant model for predicting electronic charge densities using "floating" orbitals. Floating orbitals are a long-standing idea in the quantum chemistry community that promises more compact and accurate representations by placing orbitals freely in space, as opposed to centering all orbitals at the position of atoms. Finding ideal placements of these orbitals requires extensive domain knowledge though, which thus far has prevented widespread adoption. We solve this in a data-driven manner by training a Cartesian tensor network to predict orbital positions along with orbital coefficients. This is made possible through a symmetry-breaking mechanism that is used to learn position displacements with lower symmetry than the input molecule while preserving the rotation equivariance of the charge density itself. Inspired by recent successes of Gaussian Splatting in representing densities in space, we are using Gaussians as our orbitals and predict their weights and covariance matrices. Our method achieves a state-of-the-art balance between computational efficiency and predictive accuracy on established benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08224v1">HRAvatar: High-Quality and Relightable Gaussian Head Avatar</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Project page: https://eastbeanzhang.github.io/HRAvatar
    </div>
    <details class="paper-abstract">
      Reconstructing animatable and high-quality 3D head avatars from monocular videos, especially with realistic relighting, is a valuable task. However, the limited information from single-view input, combined with the complex head poses and facial movements, makes this challenging. Previous methods achieve real-time performance by combining 3D Gaussian Splatting with a parametric head model, but the resulting head quality suffers from inaccurate face tracking and limited expressiveness of the deformation model. These methods also fail to produce realistic effects under novel lighting conditions. To address these issues, we propose HRAvatar, a 3DGS-based method that reconstructs high-fidelity, relightable 3D head avatars. HRAvatar reduces tracking errors through end-to-end optimization and better captures individual facial deformations using learnable blendshapes and learnable linear blend skinning. Additionally, it decomposes head appearance into several physical properties and incorporates physically-based shading to account for environmental lighting. Extensive experiments demonstrate that HRAvatar not only reconstructs superior-quality heads but also achieves realistic visual effects under varying lighting conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08217v1">S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has reshaped the field of photorealistic 3D reconstruction, achieving impressive rendering quality and speed. However, when applied to large-scale street scenes, existing methods suffer from rapidly escalating per-viewpoint reconstruction costs as scene size increases, leading to significant computational overhead. After revisiting the conventional pipeline, we identify three key factors accounting for this issue: unnecessary local-to-global transformations, excessive 3D-to-2D projections, and inefficient rendering of distant content. To address these challenges, we propose S3R-GS, a 3DGS framework that Streamlines the pipeline for large-scale Street Scene Reconstruction, effectively mitigating these limitations. Moreover, most existing street 3DGS methods rely on ground-truth 3D bounding boxes to separate dynamic and static components, but 3D bounding boxes are difficult to obtain, limiting real-world applicability. To address this, we propose an alternative solution with 2D boxes, which are easier to annotate or can be predicted by off-the-shelf vision foundation models. Such designs together make S3R-GS readily adapt to large, in-the-wild scenarios. Extensive experiments demonstrate that S3R-GS enhances rendering quality and significantly accelerates reconstruction. Remarkably, when applied to videos from the challenging Argoverse2 dataset, it achieves state-of-the-art PSNR and SSIM, reducing reconstruction time to below 50%--and even 20%--of competing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08166v1">Dynamic Scene Reconstruction: Recent Advance in Real-time Rendering and Streaming</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 20 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Representing and rendering dynamic scenes from 2D images is a fundamental yet challenging problem in computer vision and graphics. This survey provides a comprehensive review of the evolution and advancements in dynamic scene representation and rendering, with a particular emphasis on recent progress in Neural Radiance Fields based and 3D Gaussian Splatting based reconstruction methods. We systematically summarize existing approaches, categorize them according to their core principles, compile relevant datasets, compare the performance of various methods on these benchmarks, and explore the challenges and future research directions in this rapidly evolving field. In total, we review over 170 relevant papers, offering a broad perspective on the state of the art in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08135v1">ArticulatedGS: Self-supervised Digital Twin Modeling of Articulated Objects using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      We tackle the challenge of concurrent reconstruction at the part level with the RGB appearance and estimation of motion parameters for building digital twins of articulated objects using the 3D Gaussian Splatting (3D-GS) method. With two distinct sets of multi-view imagery, each depicting an object in separate static articulation configurations, we reconstruct the articulated object in 3D Gaussian representations with both appearance and geometry information at the same time. Our approach decoupled multiple highly interdependent parameters through a multi-step optimization process, thereby achieving a stable optimization procedure and high-quality outcomes. We introduce ArticulatedGS, a self-supervised, comprehensive framework that autonomously learns to model shapes and appearances at the part level and synchronizes the optimization of motion parameters, all without reliance on 3D supervision, motion cues, or semantic labels. Our experimental results demonstrate that, among comparable methodologies, our approach has achieved optimal outcomes in terms of part segmentation accuracy, motion estimation accuracy, and visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06714v3">F3D-Gaus: Feed-forward 3D-aware Generation on ImageNet with Cycle-Aggregative Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Project Page: https://w-ted.github.io/publications/F3D-Gaus
    </div>
    <details class="paper-abstract">
      This paper tackles the problem of generalizable 3D-aware generation from monocular datasets, e.g., ImageNet. The key challenge of this task is learning a robust 3D-aware representation without multi-view or dynamic data, while ensuring consistent texture and geometry across different viewpoints. Although some baseline methods are capable of 3D-aware generation, the quality of the generated images still lags behind state-of-the-art 2D generation approaches, which excel in producing high-quality, detailed images. To address this severe limitation, we propose a novel feed-forward pipeline based on pixel-aligned Gaussian Splatting, coined as F3D-Gaus, which can produce more realistic and reliable 3D renderings from monocular inputs. In addition, we introduce a self-supervised cycle-aggregative constraint to enforce cross-view consistency in the learned 3D representation. This training strategy naturally allows aggregation of multiple aligned Gaussian primitives and significantly alleviates the interpolation limitations inherent in single-view pixel-aligned Gaussian Splatting. Furthermore, we incorporate video model priors to perform geometry-aware refinement, enhancing the generation of fine details in wide-viewpoint scenarios and improving the model's capability to capture intricate 3D textures. Extensive experiments demonstrate that our approach not only achieves high-quality, multi-view consistent 3D-aware generation from monocular datasets, but also significantly improves training and inference efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10133v3">Efficient Density Control for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated outstanding performance in novel view synthesis, achieving a balance between rendering quality and real-time performance. 3DGS employs Adaptive Density Control (ADC) to increase the number of Gaussians. However, the clone and split operations within ADC are not sufficiently efficient, impacting optimization speed and detail recovery. Additionally, overfitted Gaussians that affect rendering quality may exist, and the original ADC is unable to remove them. To address these issues, we propose two key innovations: (1) Long-Axis Split, which precisely controls the position, shape, and opacity of child Gaussians to minimize the difference before and after splitting. (2) Recovery-Aware Pruning, which leverages differences in recovery speed after resetting opacity to prune overfitted Gaussians, thereby improving generalization performance. Experimental results show that our method significantly enhances rendering quality. Code is available at https://github.com/XiaoBin2001/EDC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18197v3">Make-It-Animatable: An Efficient Framework for Authoring Animation-Ready 3D Characters</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 CVPR 2025. Project page: https://jasongzy.github.io/Make-It-Animatable/
    </div>
    <details class="paper-abstract">
      3D characters are essential to modern creative industries, but making them animatable often demands extensive manual work in tasks like rigging and skinning. Existing automatic rigging tools face several limitations, including the necessity for manual annotations, rigid skeleton topologies, and limited generalization across diverse shapes and poses. An alternative approach is to generate animatable avatars pre-bound to a rigged template mesh. However, this method often lacks flexibility and is typically limited to realistic human shapes. To address these issues, we present Make-It-Animatable, a novel data-driven method to make any 3D humanoid model ready for character animation in less than one second, regardless of its shapes and poses. Our unified framework generates high-quality blend weights, bones, and pose transformations. By incorporating a particle-based shape autoencoder, our approach supports various 3D representations, including meshes and 3D Gaussian splats. Additionally, we employ a coarse-to-fine representation and a structure-aware modeling strategy to ensure both accuracy and robustness, even for characters with non-standard skeleton structures. We conducted extensive experiments to validate our framework's effectiveness. Compared to existing methods, our approach demonstrates significant improvements in both quality and speed. More demos and code are available at https://jasongzy.github.io/Make-It-Animatable/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2412.01807v2">Occam's LGS: An Efficient Approach for Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Project Page: https://insait-institute.github.io/OccamLGS/
    </div>
    <details class="paper-abstract">
      TL;DR: Gaussian Splatting is a widely adopted approach for 3D scene representation, offering efficient, high-quality reconstruction and rendering. A key reason for its success is the simplicity of representing scenes with sets of Gaussians, making it interpretable and adaptable. To enhance understanding beyond visual representation, recent approaches extend Gaussian Splatting with semantic vision-language features, enabling open-set tasks. Typically, these language features are aggregated from multiple 2D views, however, existing methods rely on cumbersome techniques, resulting in high computational costs and longer training times. In this work, we show that the complicated pipelines for language 3D Gaussian Splatting are simply unnecessary. Instead, we follow a probabilistic formulation of Language Gaussian Splatting and apply Occam's razor to the task at hand, leading to a highly efficient weighted multi-view feature aggregation technique. Doing so offers us state-of-the-art results with a speed-up of two orders of magnitude without any compression, allowing for easy scene manipulation. Project Page: https://insait-institute.github.io/OccamLGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.00486v4">CaRtGS: Computational Alignment for Real-Time Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Accepted by IEEE Robotics and Automation Letters (RA-L)
    </div>
    <details class="paper-abstract">
      Simultaneous Localization and Mapping (SLAM) is pivotal in robotics, with photorealistic scene reconstruction emerging as a key challenge. To address this, we introduce Computational Alignment for Real-Time Gaussian Splatting SLAM (CaRtGS), a novel method enhancing the efficiency and quality of photorealistic scene reconstruction in real-time environments. Leveraging 3D Gaussian Splatting (3DGS), CaRtGS achieves superior rendering quality and processing speed, which is crucial for scene photorealistic reconstruction. Our approach tackles computational misalignment in Gaussian Splatting SLAM (GS-SLAM) through an adaptive strategy that enhances optimization iterations, addresses long-tail optimization, and refines densification. Experiments on Replica, TUM-RGBD, and VECtor datasets demonstrate CaRtGS's effectiveness in achieving high-fidelity rendering with fewer Gaussian primitives. This work propels SLAM towards real-time, photorealistic dense rendering, significantly advancing photorealistic scene representation. For the benefit of the research community, we release the code and accompanying videos on our project website: https://dapengfeng.github.io/cartgs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.08129v3">Efficient Perspective-Correct 3D Gaussian Splatting Using Hybrid Transparency</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Project page: https://fhahlbohm.github.io/htgs/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splats (3DGS) have proven a versatile rendering primitive, both for inverse rendering as well as real-time exploration of scenes. In these applications, coherence across camera frames and multiple views is crucial, be it for robust convergence of a scene reconstruction or for artifact-free fly-throughs. Recent work started mitigating artifacts that break multi-view coherence, including popping artifacts due to inconsistent transparency sorting and perspective-correct outlines of (2D) splats. At the same time, real-time requirements forced such implementations to accept compromises in how transparency of large assemblies of 3D Gaussians is resolved, in turn breaking coherence in other ways. In our work, we aim at achieving maximum coherence, by rendering fully perspective-correct 3D Gaussians while using a high-quality approximation of accurate blending, hybrid transparency, on a per-pixel level, in order to retain real-time frame rates. Our fast and perspectively accurate approach for evaluation of 3D Gaussians does not require matrix inversions, thereby ensuring numerical stability and eliminating the need for special handling of degenerate splats, and the hybrid transparency formulation for blending maintains similar quality as fully resolved per-pixel transparencies at a fraction of the rendering costs. We further show that each of these two components can be independently integrated into Gaussian splatting systems. In combination, they achieve up to 2$\times$ higher frame rates, 2$\times$ faster optimization, and equal or better image quality with fewer rendering artifacts compared to traditional 3DGS on common benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08508v4">BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      We present billboard Splatting (BBSplat) - a novel approach for novel view synthesis based on textured geometric primitives. BBSplat represents the scene as a set of optimizable textured planar primitives with learnable RGB textures and alpha-maps to control their shape. BBSplat primitives can be used in any Gaussian Splatting pipeline as drop-in replacements for Gaussians. The proposed primitives close the rendering quality gap between 2D and 3D Gaussian Splatting (GS), enabling the accurate extraction of 3D mesh as in the 2DGS framework. Additionally, the explicit nature of planar primitives enables the use of the ray-tracing effects in rasterization. Our novel regularization term encourages textures to have a sparser structure, enabling an efficient compression that leads to a reduction in the storage space of the model up to x17 times compared to 3DGS. Our experiments show the efficiency of BBSplat on standard datasets of real indoor and outdoor scenes such as Tanks&Temples, DTU, and Mip-NeRF-360. Namely, we achieve a state-of-the-art PSNR of 29.72 for DTU at Full HD resolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07191v1">All That Glitters Is Not Gold: Key-Secured 3D Secrets within 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have revolutionized scene reconstruction, opening new possibilities for 3D steganography by hiding 3D secrets within 3D covers. The key challenge in steganography is ensuring imperceptibility while maintaining high-fidelity reconstruction. However, existing methods often suffer from detectability risks and utilize only suboptimal 3DGS features, limiting their full potential. We propose a novel end-to-end key-secured 3D steganography framework (KeySS) that jointly optimizes a 3DGS model and a key-secured decoder for secret reconstruction. Our approach reveals that Gaussian features contribute unequally to secret hiding. The framework incorporates a key-controllable mechanism enabling multi-secret hiding and unauthorized access prevention, while systematically exploring optimal feature update to balance fidelity and security. To rigorously evaluate steganographic imperceptibility beyond conventional 2D metrics, we introduce 3D-Sinkhorn distance analysis, which quantifies distributional differences between original and steganographic Gaussian parameters in the representation space. Extensive experiments demonstrate that our method achieves state-of-the-art performance in both cover and secret reconstruction while maintaining high security levels, advancing the field of 3D steganography. Code is available at https://github.com/RY-Paper/KeySS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07190v1">Multi-Modal 3D Mesh Reconstruction from Images and Text</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 under review
    </div>
    <details class="paper-abstract">
      6D object pose estimation for unseen objects is essential in robotics but traditionally relies on trained models that require large datasets, high computational costs, and struggle to generalize. Zero-shot approaches eliminate the need for training but depend on pre-existing 3D object models, which are often impractical to obtain. To address this, we propose a language-guided few-shot 3D reconstruction method, reconstructing a 3D mesh from few input images. In the proposed pipeline, receives a set of input images and a language query. A combination of GroundingDINO and Segment Anything Model outputs segmented masks from which a sparse point cloud is reconstructed with VGGSfM. Subsequently, the mesh is reconstructed with the Gaussian Splatting method SuGAR. In a final cleaning step, artifacts are removed, resulting in the final 3D mesh of the queried object. We evaluate the method in terms of accuracy and quality of the geometry and texture. Furthermore, we study the impact of imaging conditions such as viewing angle, number of input images, and image overlap on 3D object reconstruction quality, efficiency, and computational scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14957v2">Dream to Manipulate: Compositional World Models Empowering Robot Imitation Learning with Imagination</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      A world model provides an agent with a representation of its environment, enabling it to predict the causal consequences of its actions. Current world models typically cannot directly and explicitly imitate the actual environment in front of a robot, often resulting in unrealistic behaviors and hallucinations that make them unsuitable for real-world robotics applications. To overcome those challenges, we propose to rethink robot world models as learnable digital twins. We introduce DreMa, a new approach for constructing digital twins automatically using learned explicit representations of the real world and its dynamics, bridging the gap between traditional digital twins and world models. DreMa replicates the observed world and its structure by integrating Gaussian Splatting and physics simulators, allowing robots to imagine novel configurations of objects and to predict the future consequences of robot actions thanks to its compositionality. We leverage this capability to generate new data for imitation learning by applying equivariant transformations to a small set of demonstrations. Our evaluations across various settings demonstrate significant improvements in accuracy and robustness by incrementing actions and object distributions, reducing the data needed to learn a policy and improving the generalization of the agents. As a highlight, we show that a real Franka Emika Panda robot, powered by DreMa's imagination, can successfully learn novel physical tasks from just a single example per task variation (one-shot policy learning). Our project page can be found in: https://dreamtomanipulate.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06390v3">SplatFormer: Point Transformer for Robust 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently transformed photorealistic reconstruction, achieving high visual fidelity and real-time performance. However, rendering quality significantly deteriorates when test views deviate from the camera angles used during training, posing a major challenge for applications in immersive free-viewpoint rendering and navigation. In this work, we conduct a comprehensive evaluation of 3DGS and related novel view synthesis methods under out-of-distribution (OOD) test camera scenarios. By creating diverse test cases with synthetic and real-world datasets, we demonstrate that most existing methods, including those incorporating various regularization techniques and data-driven priors, struggle to generalize effectively to OOD views. To address this limitation, we introduce SplatFormer, the first point transformer model specifically designed to operate on Gaussian splats. SplatFormer takes as input an initial 3DGS set optimized under limited training views and refines it in a single forward pass, effectively removing potential artifacts in OOD test views. To our knowledge, this is the first successful application of point transformers directly on 3DGS sets, surpassing the limitations of previous multi-scene training methods, which could handle only a restricted number of input views during inference. Our model significantly improves rendering quality under extreme novel views, achieving state-of-the-art performance in these challenging scenarios and outperforming various 3DGS regularization techniques, multi-scene models tailored for sparse view synthesis, and diffusion-based frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07000v1">Frequency-Aware Density Control via Reparameterization for High-Quality Rendering of 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 Accepted to AAAI2025
    </div>
    <details class="paper-abstract">
      By adaptively controlling the density and generating more Gaussians in regions with high-frequency information, 3D Gaussian Splatting (3DGS) can better represent scene details. From the signal processing perspective, representing details usually needs more Gaussians with relatively smaller scales. However, 3DGS currently lacks an explicit constraint linking the density and scale of 3D Gaussians across the domain, leading to 3DGS using improper-scale Gaussians to express frequency information, resulting in the loss of accuracy. In this paper, we propose to establish a direct relation between density and scale through the reparameterization of the scaling parameters and ensure the consistency between them via explicit constraints (i.e., density responds well to changes in frequency). Furthermore, we develop a frequency-aware density control strategy, consisting of densification and deletion, to improve representation quality with fewer Gaussians. A dynamic threshold encourages densification in high-frequency regions, while a scale-based filter deletes Gaussians with improper scale. Experimental results on various datasets demonstrate that our method outperforms existing state-of-the-art methods quantitatively and qualitatively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05168v2">SeeLe: A Unified Acceleration Framework for Real-Time Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a crucial rendering technique for many real-time applications. However, the limited hardware resources on today's mobile platforms hinder these applications, as they struggle to achieve real-time performance. In this paper, we propose SeeLe, a general framework designed to accelerate the 3DGS pipeline for resource-constrained mobile devices. Specifically, we propose two GPU-oriented techniques: hybrid preprocessing and contribution-aware rasterization. Hybrid preprocessing alleviates the GPU compute and memory pressure by reducing the number of irrelevant Gaussians during rendering. The key is to combine our view-dependent scene representation with online filtering. Meanwhile, contribution-aware rasterization improves the GPU utilization at the rasterization stage by prioritizing Gaussians with high contributions while reducing computations for those with low contributions. Both techniques can be seamlessly integrated into existing 3DGS pipelines with minimal fine-tuning. Collectively, our framework achieves 2.6$\times$ speedup and 32.3\% model reduction while achieving superior rendering quality compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12518v4">Hier-SLAM: Scaling-up Semantics in SLAM with a Hierarchically Categorical Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 Accepted for publication at ICRA 2025. Code is available at https://github.com/LeeBY68/Hier-SLAM
    </div>
    <details class="paper-abstract">
      We propose Hier-SLAM, a semantic 3D Gaussian Splatting SLAM method featuring a novel hierarchical categorical representation, which enables accurate global 3D semantic mapping, scaling-up capability, and explicit semantic label prediction in the 3D world. The parameter usage in semantic SLAM systems increases significantly with the growing complexity of the environment, making it particularly challenging and costly for scene understanding. To address this problem, we introduce a novel hierarchical representation that encodes semantic information in a compact form into 3D Gaussian Splatting, leveraging the capabilities of large language models (LLMs). We further introduce a novel semantic loss designed to optimize hierarchical semantic information through both inter-level and cross-level optimization. Furthermore, we enhance the whole SLAM system, resulting in improved tracking and mapping performance. Our \MethodName{} outperforms existing dense SLAM methods in both mapping and tracking accuracy, while achieving a 2x operation speed-up. Additionally, it achieves on-par semantic rendering performance compared to existing methods while significantly reducing storage and training time requirements. Rendering FPS impressively reaches 2,000 with semantic information and 3,000 without it. Most notably, it showcases the capability of handling the complex real-world scene with more than 500 semantic classes, highlighting its valuable scaling-up capability. The open-source code is available at https://github.com/LeeBY68/Hier-SLAM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05152v2">GSplatVNM: Point-of-View Synthesis for Visual Navigation Models Using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach to image-goal navigation by integrating 3D Gaussian Splatting (3DGS) with Visual Navigation Models (VNMs), a method we refer to as GSplatVNM. VNMs offer a promising paradigm for image-goal navigation by guiding a robot through a sequence of point-of-view images without requiring metrical localization or environment-specific training. However, constructing a dense and traversable sequence of target viewpoints from start to goal remains a central challenge, particularly when the available image database is sparse. To address these challenges, we propose a 3DGS-based viewpoint synthesis framework for VNMs that synthesizes intermediate viewpoints to seamlessly bridge gaps in sparse data while significantly reducing storage overhead. Experimental results in a photorealistic simulator demonstrate that our approach not only enhances navigation efficiency but also exhibits robustness under varying levels of image database sparsity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06900v1">DirectTriGS: Triplane-based Gaussian Splatting Field Representation for 3D Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 Accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      We present DirectTriGS, a novel framework designed for 3D object generation with Gaussian Splatting (GS). GS-based rendering for 3D content has gained considerable attention recently. However, there has been limited exploration in directly generating 3D Gaussians compared to traditional generative modeling approaches. The main challenge lies in the complex data structure of GS represented by discrete point clouds with multiple channels. To overcome this challenge, we propose employing the triplane representation, which allows us to represent Gaussian Splatting as an image-like continuous field. This representation effectively encodes both the geometry and texture information, enabling smooth transformation back to Gaussian point clouds and rendering into images by a TriRenderer, with only 2D supervisions. The proposed TriRenderer is fully differentiable, so that the rendering loss can supervise both texture and geometry encoding. Furthermore, the triplane representation can be compressed using a Variational Autoencoder (VAE), which can subsequently be utilized in latent diffusion to generate 3D objects. The experiments demonstrate that the proposed generation framework can produce high-quality 3D object geometry and rendering results in the text-to-3D task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06859v1">ActiveInitSplat: How Active Image Selection Helps Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Gaussian splatting (GS) along with its extensions and variants provides outstanding performance in real-time scene rendering while meeting reduced storage demands and computational efficiency. While the selection of 2D images capturing the scene of interest is crucial for the proper initialization and training of GS, hence markedly affecting the rendering performance, prior works rely on passively and typically densely selected 2D images. In contrast, this paper proposes `ActiveInitSplat', a novel framework for active selection of training images for proper initialization and training of GS. ActiveInitSplat relies on density and occupancy criteria of the resultant 3D scene representation from the selected 2D images, to ensure that the latter are captured from diverse viewpoints leading to better scene coverage and that the initialized Gaussian functions are well aligned with the actual 3D structure. Numerical tests on well-known simulated and real environments demonstrate the merits of ActiveInitSplat resulting in significant GS rendering performance improvement over passive GS baselines, in the widely adopted LPIPS, SSIM, and PSNR metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00486v4">CaRtGS: Computational Alignment for Real-Time Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 Accepted by IEEE Robotics and Automation Letters (RA-L)
    </div>
    <details class="paper-abstract">
      Simultaneous Localization and Mapping (SLAM) is pivotal in robotics, with photorealistic scene reconstruction emerging as a key challenge. To address this, we introduce Computational Alignment for Real-Time Gaussian Splatting SLAM (CaRtGS), a novel method enhancing the efficiency and quality of photorealistic scene reconstruction in real-time environments. Leveraging 3D Gaussian Splatting (3DGS), CaRtGS achieves superior rendering quality and processing speed, which is crucial for scene photorealistic reconstruction. Our approach tackles computational misalignment in Gaussian Splatting SLAM (GS-SLAM) through an adaptive strategy that enhances optimization iterations, addresses long-tail optimization, and refines densification. Experiments on Replica, TUM-RGBD, and VECtor datasets demonstrate CaRtGS's effectiveness in achieving high-fidelity rendering with fewer Gaussian primitives. This work propels SLAM towards real-time, photorealistic dense rendering, significantly advancing photorealistic scene representation. For the benefit of the research community, we release the code and accompanying videos on our project website: https://dapengfeng.github.io/cartgs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07819v1">POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      In this paper, we present a novel algorithm for quantifying uncertainty and information gained within 3D Gaussian Splatting (3D-GS) through P-Optimality. While 3D-GS has proven to be a useful world model with high-quality rasterizations, it does not natively quantify uncertainty. Quantifying uncertainty in parameters of 3D-GS is necessary to understand the information gained from acquiring new images as in active perception, or identify redundant images which can be removed from memory due to resource constraints in online 3D-GS SLAM. We propose to quantify uncertainty and information gain in 3D-GS by reformulating the problem through the lens of optimal experimental design, which is a classical solution to measuring information gain. By restructuring information quantification of 3D-GS through optimal experimental design, we arrive at multiple solutions, of which T-Optimality and D-Optimality perform the best quantitatively and qualitatively as measured on two popular datasets. Additionally, we propose a block diagonal approximation of the 3D-GS uncertainty, which provides a measure of correlation for computing more accurate information gain, at the expense of a greater computation cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07476v1">SOGS: Second-Order Anchor for Advanced 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 Accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      Anchor-based 3D Gaussian splatting (3D-GS) exploits anchor features in 3D Gaussian prediction, which has achieved impressive 3D rendering quality with reduced Gaussian redundancy. On the other hand, it often encounters the dilemma among anchor features, model size, and rendering quality - large anchor features lead to large 3D models and high-quality rendering whereas reducing anchor features degrades Gaussian attribute prediction which leads to clear artifacts in the rendered textures and geometries. We design SOGS, an anchor-based 3D-GS technique that introduces second-order anchors to achieve superior rendering quality and reduced anchor features and model size simultaneously. Specifically, SOGS incorporates covariance-based second-order statistics and correlation across feature dimensions to augment features within each anchor, compensating for the reduced feature size and improving rendering quality effectively. In addition, it introduces a selective gradient loss to enhance the optimization of scene textures and scene geometries, leading to high-quality rendering with small anchor features. Extensive experiments over multiple widely adopted benchmarks show that SOGS achieves superior rendering quality in novel view synthesis with clearly reduced model size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07446v1">EigenGS Representation: From Eigenspace to Gaussian Image Space</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Principal Component Analysis (PCA), a classical dimensionality reduction technique, and 2D Gaussian representation, an adaptation of 3D Gaussian Splatting for image representation, offer distinct approaches to modeling visual data. We present EigenGS, a novel method that bridges these paradigms through an efficient transformation pipeline connecting eigenspace and image-space Gaussian representations. Our approach enables instant initialization of Gaussian parameters for new images without requiring per-image optimization from scratch, dramatically accelerating convergence. EigenGS introduces a frequency-aware learning mechanism that encourages Gaussians to adapt to different scales, effectively modeling varied spatial frequencies and preventing artifacts in high-resolution reconstruction. Extensive experiments demonstrate that EigenGS not only achieves superior reconstruction quality compared to direct 2D Gaussian fitting but also reduces necessary parameter count and training time. The results highlight EigenGS's effectiveness and generalization ability across images with varying resolutions and diverse categories, making Gaussian-based image representation both high-quality and viable for real-time applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08129v3">Efficient Perspective-Correct 3D Gaussian Splatting Using Hybrid Transparency</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 Project page: https://fhahlbohm.github.io/htgs/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splats (3DGS) have proven a versatile rendering primitive, both for inverse rendering as well as real-time exploration of scenes. In these applications, coherence across camera frames and multiple views is crucial, be it for robust convergence of a scene reconstruction or for artifact-free fly-throughs. Recent work started mitigating artifacts that break multi-view coherence, including popping artifacts due to inconsistent transparency sorting and perspective-correct outlines of (2D) splats. At the same time, real-time requirements forced such implementations to accept compromises in how transparency of large assemblies of 3D Gaussians is resolved, in turn breaking coherence in other ways. In our work, we aim at achieving maximum coherence, by rendering fully perspective-correct 3D Gaussians while using a high-quality approximation of accurate blending, hybrid transparency, on a per-pixel level, in order to retain real-time frame rates. Our fast and perspectively accurate approach for evaluation of 3D Gaussians does not require matrix inversions, thereby ensuring numerical stability and eliminating the need for special handling of degenerate splats, and the hybrid transparency formulation for blending maintains similar quality as fully resolved per-pixel transparencies at a fraction of the rendering costs. We further show that each of these two components can be independently integrated into Gaussian splatting systems. In combination, they achieve up to 2$\times$ higher frame rates, 2$\times$ faster optimization, and equal or better image quality with fewer rendering artifacts compared to traditional 3DGS on common benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13654v2">GAGS: Granularity-Aware Feature Distillation for Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 Project page: https://pz0826.github.io/GAGS-Webpage/
    </div>
    <details class="paper-abstract">
      3D open-vocabulary scene understanding, which accurately perceives complex semantic properties of objects in space, has gained significant attention in recent years. In this paper, we propose GAGS, a framework that distills 2D CLIP features into 3D Gaussian splatting, enabling open-vocabulary queries for renderings on arbitrary viewpoints. The main challenge of distilling 2D features for 3D fields lies in the multiview inconsistency of extracted 2D features, which provides unstable supervision for the 3D feature field. GAGS addresses this challenge with two novel strategies. First, GAGS associates the prompt point density of SAM with the camera distances, which significantly improves the multiview consistency of segmentation results. Second, GAGS further decodes a granularity factor to guide the distillation process and this granularity factor can be learned in a unsupervised manner to only select the multiview consistent 2D features in the distillation process. Experimental results on two datasets demonstrate significant performance and stability improvements of GAGS in visual grounding and semantic segmentation, with an inference speed 2$\times$ faster than baseline methods. The code and additional results are available at https://pz0826.github.io/GAGS-Webpage/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06762v1">Gaussian RBFNet: Gaussian Radial Basis Functions for Fast and Accurate Representation and Reconstruction of Neural Fields</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 Our code is available at https://grbfnet.github.io/
    </div>
    <details class="paper-abstract">
      Neural fields such as DeepSDF and Neural Radiance Fields have recently revolutionized novel-view synthesis and 3D reconstruction from RGB images and videos. However, achieving high-quality representation, reconstruction, and rendering requires deep neural networks, which are slow to train and evaluate. Although several acceleration techniques have been proposed, they often trade off speed for memory. Gaussian splatting-based methods, on the other hand, accelerate the rendering time but remain costly in terms of training speed and memory needed to store the parameters of a large number of Gaussians. In this paper, we introduce a novel neural representation that is fast, both at training and inference times, and lightweight. Our key observation is that the neurons used in traditional MLPs perform simple computations (a dot product followed by ReLU activation) and thus one needs to use either wide and deep MLPs or high-resolution and high-dimensional feature grids to parameterize complex nonlinear functions. We show in this paper that by replacing traditional neurons with Radial Basis Function (RBF) kernels, one can achieve highly accurate representation of 2D (RGB images), 3D (geometry), and 5D (radiance fields) signals with just a single layer of such neurons. The representation is highly parallelizable, operates on low-resolution feature grids, and is compact and memory-efficient. We demonstrate that the proposed novel representation can be trained for 3D geometry representation in less than 15 seconds and for novel view synthesis in less than 15 mins. At runtime, it can synthesize novel views at more than 60 fps without sacrificing quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06744v1">CoDa-4DGS: Dynamic Gaussian Splatting with Context and Deformation Awareness for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-03-09
    </div>
    <details class="paper-abstract">
      Dynamic scene rendering opens new avenues in autonomous driving by enabling closed-loop simulations with photorealistic data, which is crucial for validating end-to-end algorithms. However, the complex and highly dynamic nature of traffic environments presents significant challenges in accurately rendering these scenes. In this paper, we introduce a novel 4D Gaussian Splatting (4DGS) approach, which incorporates context and temporal deformation awareness to improve dynamic scene rendering. Specifically, we employ a 2D semantic segmentation foundation model to self-supervise the 4D semantic features of Gaussians, ensuring meaningful contextual embedding. Simultaneously, we track the temporal deformation of each Gaussian across adjacent frames. By aggregating and encoding both semantic and temporal deformation features, each Gaussian is equipped with cues for potential deformation compensation within 3D space, facilitating a more precise representation of dynamic scenes. Experimental results show that our method improves 4DGS's ability to capture fine details in dynamic scene rendering for autonomous driving and outperforms other self-supervised methods in 4D reconstruction and novel view synthesis. Furthermore, CoDa-4DGS deforms semantic features with each Gaussian, enabling broader applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06740v1">D3DR: Lighting-Aware Object Insertion in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-09
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has become a popular technique for various 3D Computer Vision tasks, including novel view synthesis, scene reconstruction, and dynamic scene rendering. However, the challenge of natural-looking object insertion, where the object's appearance seamlessly matches the scene, remains unsolved. In this work, we propose a method, dubbed D3DR, for inserting a 3DGS-parametrized object into 3DGS scenes while correcting its lighting, shadows, and other visual artifacts to ensure consistency, a problem that has not been successfully addressed before. We leverage advances in diffusion models, which, trained on real-world data, implicitly understand correct scene lighting. After inserting the object, we optimize a diffusion-based Delta Denoising Score (DDS)-inspired objective to adjust its 3D Gaussian parameters for proper lighting correction. Utilizing diffusion model personalization techniques to improve optimization quality, our approach ensures seamless object insertion and natural appearance. Finally, we demonstrate the method's effectiveness by comparing it to existing approaches, achieving 0.5 PSNR and 0.15 SSIM improvements in relighting quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06677v1">REArtGS: Reconstructing and Generating Articulated Objects via 3D Gaussian Splatting with Geometric and Motion Constraints</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 11pages, 6 figures
    </div>
    <details class="paper-abstract">
      Articulated objects, as prevalent entities in human life, their 3D representations play crucial roles across various applications. However, achieving both high-fidelity textured surface reconstruction and dynamic generation for articulated objects remains challenging for existing methods. In this paper, we present REArtGS, a novel framework that introduces additional geometric and motion constraints to 3D Gaussian primitives, enabling high-quality textured surface reconstruction and generation for articulated objects. Specifically, given multi-view RGB images of arbitrary two states of articulated objects, we first introduce an unbiased Signed Distance Field (SDF) guidance to regularize Gaussian opacity fields, enhancing geometry constraints and improving surface reconstruction quality. Then we establish deformable fields for 3D Gaussians constrained by the kinematic structures of articulated objects, achieving unsupervised generation of surface meshes in unseen states. Extensive experiments on both synthetic and real datasets demonstrate our approach achieves high-quality textured surface reconstruction for given states, and enables high-fidelity surface generation for unseen states. Codes will be released within the next four months.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06617v1">Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 Tech Report
    </div>
    <details class="paper-abstract">
      Arbitrary-scale super-resolution (ASSR) aims to reconstruct high-resolution (HR) images from low-resolution (LR) inputs with arbitrary upsampling factors using a single model, addressing the limitations of traditional SR methods constrained to fixed-scale factors (\textit{e.g.}, $\times$ 2). Recent advances leveraging implicit neural representation (INR) have achieved great progress by modeling coordinate-to-pixel mappings. However, the efficiency of these methods may suffer from repeated upsampling and decoding, while their reconstruction fidelity and quality are constrained by the intrinsic representational limitations of coordinate-based functions. To address these challenges, we propose a novel ContinuousSR framework with a Pixel-to-Gaussian paradigm, which explicitly reconstructs 2D continuous HR signals from LR images using Gaussian Splatting. This approach eliminates the need for time-consuming upsampling and decoding, enabling extremely fast arbitrary-scale super-resolution. Once the Gaussian field is built in a single pass, ContinuousSR can perform arbitrary-scale rendering in just 1ms per scale. Our method introduces several key innovations. Through statistical ana
    </details>
</div>
