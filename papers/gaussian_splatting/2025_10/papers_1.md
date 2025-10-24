# gaussian splatting - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

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
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14705v1">Leveraging Learned Image Prior for 3D Gaussian Compression</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted to ICCV 2025 Workshop on ECLR
    </div>
    <details class="paper-abstract">
      Compression techniques for 3D Gaussian Splatting (3DGS) have recently achieved considerable success in minimizing storage overhead for 3D Gaussians while preserving high rendering quality. Despite the impressive storage reduction, the lack of learned priors restricts further advances in the rate-distortion trade-off for 3DGS compression tasks. To address this, we introduce a novel 3DGS compression framework that leverages the powerful representational capacity of learned image priors to recover compression-induced quality degradation. Built upon initially compressed Gaussians, our restoration network effectively models the compression artifacts in the image space between degraded and original Gaussians. To enhance the rate-distortion performance, we provide coarse rendering residuals into the restoration network as side information. By leveraging the supervision of restored images, the compressed Gaussians are refined, resulting in a highly compact representation with enhanced rendering performance. Our framework is designed to be compatible with existing Gaussian compression methods, making it broadly applicable across different baselines. Extensive experiments validate the effectiveness of our framework, demonstrating superior rate-distortion performance and outperforming the rendering quality of state-of-the-art 3DGS compression methods while requiring substantially less storage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14564v1">BalanceGS: Algorithm-System Co-design for Efficient 3D Gaussian Splatting Training on GPU</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted by ASP-DAC 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a promising 3D reconstruction technique. The traditional 3DGS training pipeline follows three sequential steps: Gaussian densification, Gaussian projection, and color splatting. Despite its promising reconstruction quality, this conventional approach suffers from three critical inefficiencies: (1) Skewed density allocation during Gaussian densification, (2) Imbalanced computation workload during Gaussian projection and (3) Fragmented memory access during color splatting. To tackle the above challenges, we introduce BalanceGS, the algorithm-system co-design for efficient training in 3DGS. (1) At the algorithm level, we propose heuristic workload-sensitive Gaussian density control to automatically balance point distributions - removing 80% redundant Gaussians in dense regions while filling gaps in sparse areas. (2) At the system level, we propose Similarity-based Gaussian sampling and merging, which replaces the static one-to-one thread-pixel mapping with adaptive workload distribution - threads now dynamically process variable numbers of Gaussians based on local cluster density. (3) At the mapping level, we propose reordering-based memory access mapping strategy that restructures RGB storage and enables batch loading in shared memory. Extensive experiments demonstrate that compared with 3DGS, our approach achieves a 1.44$\times$ training speedup on a NVIDIA A100 GPU with negligible quality degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14270v1">GauSSmart: Enhanced 3D Reconstruction through 2D Foundation Models and Geometric Filtering</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Scene reconstruction has emerged as a central challenge in computer vision, with approaches such as Neural Radiance Fields (NeRF) and Gaussian Splatting achieving remarkable progress. While Gaussian Splatting demonstrates strong performance on large-scale datasets, it often struggles to capture fine details or maintain realism in regions with sparse coverage, largely due to the inherent limitations of sparse 3D training data. In this work, we propose GauSSmart, a hybrid method that effectively bridges 2D foundational models and 3D Gaussian Splatting reconstruction. Our approach integrates established 2D computer vision techniques, including convex filtering and semantic feature supervision from foundational models such as DINO, to enhance Gaussian-based scene reconstruction. By leveraging 2D segmentation priors and high-dimensional feature embeddings, our method guides the densification and refinement of Gaussian splats, improving coverage in underrepresented areas and preserving intricate structural details. We validate our approach across three datasets, where GauSSmart consistently outperforms existing Gaussian Splatting in the majority of evaluated scenes. Our results demonstrate the significant potential of hybrid 2D-3D approaches, highlighting how the thoughtful combination of 2D foundational models with 3D reconstruction pipelines can overcome the limitations inherent in either approach alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07944v2">CVD-STORM: Cross-View Video Diffusion with Spatial-Temporal Reconstruction Model for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Generative models have been widely applied to world modeling for environment simulation and future state prediction. With advancements in autonomous driving, there is a growing demand not only for high-fidelity video generation under various controls, but also for producing diverse and meaningful information such as depth estimation. To address this, we propose CVD-STORM, a cross-view video diffusion model utilizing a spatial-temporal reconstruction Variational Autoencoder (VAE) that generates long-term, multi-view videos with 4D reconstruction capabilities under various control inputs. Our approach first fine-tunes the VAE with an auxiliary 4D reconstruction task, enhancing its ability to encode 3D structures and temporal dynamics. Subsequently, we integrate this VAE into the video diffusion process to significantly improve generation quality. Experimental results demonstrate that our model achieves substantial improvements in both FID and FVD metrics. Additionally, the jointly-trained Gaussian Splatting Decoder effectively reconstructs dynamic scenes, providing valuable geometric information for comprehensive scene understanding. Our project page is https://sensetime-fvg.github.io/CVD-STORM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15008v2">HuGDiffusion: Generalizable Single-Image Human Rendering via 3D Gaussian Diffusion</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      We present HuGDiffusion, a generalizable 3D Gaussian splatting (3DGS) learning pipeline to achieve novel view synthesis (NVS) of human characters from single-view input images. Existing approaches typically require monocular videos or calibrated multi-view images as inputs, whose applicability could be weakened in real-world scenarios with arbitrary and/or unknown camera poses. In this paper, we aim to generate the set of 3DGS attributes via a diffusion-based framework conditioned on human priors extracted from a single image. Specifically, we begin with carefully integrated human-centric feature extraction procedures to deduce informative conditioning signals. Based on our empirical observations that jointly learning the whole 3DGS attributes is challenging to optimize, we design a multi-stage generation strategy to obtain different types of 3DGS attributes. To facilitate the training process, we investigate constructing proxy ground-truth 3D Gaussian attributes as high-quality attribute-level supervision signals. Through extensive experiments, our HuGDiffusion shows significant performance improvements over the state-of-the-art methods. Our code will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14179v1">Virtually Being: Customizing Camera-Controllable Video Diffusion Models with Multi-View Performance Captures</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted to SIGGRAPH Asia 2025
    </div>
    <details class="paper-abstract">
      We introduce a framework that enables both multi-view character consistency and 3D camera control in video diffusion models through a novel customization data pipeline. We train the character consistency component with recorded volumetric capture performances re-rendered with diverse camera trajectories via 4D Gaussian Splatting (4DGS), lighting variability obtained with a video relighting model. We fine-tune state-of-the-art open-source video diffusion models on this data to provide strong multi-view identity preservation, precise camera control, and lighting adaptability. Our framework also supports core capabilities for virtual production, including multi-subject generation using two approaches: joint training and noise blending, the latter enabling efficient composition of independently customized models at inference time; it also achieves scene and real-life video customization as well as control over motion and spatial layout during customization. Extensive experiments show improved video quality, higher personalization accuracy, and enhanced camera control and lighting adaptability, advancing the integration of video generation into virtual production. Our project page is available at: https://eyeline-labs.github.io/Virtually-Being.
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
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15072v1">SaLon3R: Structure-aware Long-term Generalizable 3D Reconstruction from Unposed Images</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled generalizable, on-the-fly reconstruction of sequential input views. However, existing methods often predict per-pixel Gaussians and combine Gaussians from all views as the scene representation, leading to substantial redundancies and geometric inconsistencies in long-duration video sequences. To address this, we propose SaLon3R, a novel framework for Structure-aware, Long-term 3DGS Reconstruction. To our best knowledge, SaLon3R is the first online generalizable GS method capable of reconstructing over 50 views in over 10 FPS, with 50% to 90% redundancy removal. Our method introduces compact anchor primitives to eliminate redundancy through differentiable saliency-aware Gaussian quantization, coupled with a 3D Point Transformer that refines anchor attributes and saliency to resolve cross-frame geometric and photometric inconsistencies. Specifically, we first leverage a 3D reconstruction backbone to predict dense per-pixel Gaussians and a saliency map encoding regional geometric complexity. Redundant Gaussians are compressed into compact anchors by prioritizing high-complexity regions. The 3D Point Transformer then learns spatial structural priors in 3D space from training data to refine anchor attributes and saliency, enabling regionally adaptive Gaussian decoding for geometric fidelity. Without known camera parameters or test-time optimization, our approach effectively resolves artifacts and prunes the redundant 3DGS in a single feed-forward pass. Experiments on multiple datasets demonstrate our state-of-the-art performance on both novel view synthesis and depth estimation, demonstrating superior efficiency, robustness, and generalization ability for long-term generalizable 3D reconstruction. Project Page: https://wrld.github.io/SaLon3R/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13454v1">VIST3A: Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Project page: https://gohyojun15.github.io/VIST3A/
    </div>
    <details class="paper-abstract">
      The rapid progress of large, pretrained models for both visual content generation and 3D reconstruction opens up new possibilities for text-to-3D generation. Intuitively, one could obtain a formidable 3D scene generator if one were able to combine the power of a modern latent text-to-video model as "generator" with the geometric abilities of a recent (feedforward) 3D reconstruction system as "decoder". We introduce VIST3A, a general framework that does just that, addressing two main challenges. First, the two components must be joined in a way that preserves the rich knowledge encoded in their weights. We revisit model stitching, i.e., we identify the layer in the 3D decoder that best matches the latent representation produced by the text-to-video generator and stitch the two parts together. That operation requires only a small dataset and no labels. Second, the text-to-video generator must be aligned with the stitched 3D decoder, to ensure that the generated latents are decodable into consistent, perceptually convincing 3D scene geometry. To that end, we adapt direct reward finetuning, a popular technique for human preference alignment. We evaluate the proposed VIST3A approach with different video generators and 3D reconstruction models. All tested pairings markedly improve over prior text-to-3D models that output Gaussian splats. Moreover, by choosing a suitable 3D base model, VIST3A also enables high-quality text-to-pointmap generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13381v1">Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Accepted at ICCV-2025, project page: https://dynamic-ugsdf.github.io/
    </div>
    <details class="paper-abstract">
      Dynamic scene rendering and reconstruction play a crucial role in computer vision and augmented reality. Recent methods based on 3D Gaussian Splatting (3DGS), have enabled accurate modeling of dynamic urban scenes, but for urban scenes they require both camera and LiDAR data, ground-truth 3D segmentations and motion data in the form of tracklets or pre-defined object templates such as SMPL. In this work, we explore whether a combination of 2D object agnostic priors in the form of depth and point tracking coupled with a signed distance function (SDF) representation for dynamic objects can be used to relax some of these requirements. We present a novel approach that integrates Signed Distance Functions (SDFs) with 3D Gaussian Splatting (3DGS) to create a more robust object representation by harnessing the strengths of both methods. Our unified optimization framework enhances the geometric accuracy of 3D Gaussian splatting and improves deformation modeling within the SDF, resulting in a more adaptable and precise representation. We demonstrate that our method achieves state-of-the-art performance in rendering metrics even without LiDAR data on urban scenes. When incorporating LiDAR, our approach improved further in reconstructing and generating novel views across diverse object categories, without ground-truth 3D motion annotation. Additionally, our method enables various scene editing tasks, including scene decomposition, and scene composition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13186v1">STT-GS: Sample-Then-Transmit Edge Gaussian Splatting with Joint Client Selection and Power Control</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Edge Gaussian splatting (EGS), which aggregates data from distributed clients and trains a global GS model at the edge server, is an emerging paradigm for scene reconstruction. Unlike traditional edge resource management methods that emphasize communication throughput or general-purpose learning performance, EGS explicitly aims to maximize the GS qualities, rendering existing approaches inapplicable. To address this problem, this paper formulates a novel GS-oriented objective function that distinguishes the heterogeneous view contributions of different clients. However, evaluating this function in turn requires clients' images, leading to a causality dilemma. To this end, this paper further proposes a sample-then-transmit EGS (or STT-GS for short) strategy, which first samples a subset of images as pilot data from each client for loss prediction. Based on the first-stage evaluation, communication resources are then prioritized towards more valuable clients. To achieve efficient sampling, a feature-domain clustering (FDC) scheme is proposed to select the most representative data and pilot transmission time minimization (PTTM) is adopted to reduce the pilot overhead.Subsequently, we develop a joint client selection and power control (JCSPC) framework to maximize the GS-oriented function under communication resource constraints. Despite the nonconvexity of the problem, we propose a low-complexity efficient solution based on the penalty alternating majorization minimization (PAMM) algorithm. Experiments unveil that the proposed scheme significantly outperforms existing benchmarks on real-world datasets. It is found that the GS-oriented objective can be accurately predicted with low sampling ratios (e.g.,10%), and our method achieves an excellent tradeoff between view contributions and communication costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21117v2">CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 ICCV 2025, Project Page: https://cl-splats.github.io
    </div>
    <details class="paper-abstract">
      In dynamic 3D environments, accurately updating scene representations over time is crucial for applications in robotics, mixed reality, and embodied AI. As scenes evolve, efficient methods to incorporate changes are needed to maintain up-to-date, high-quality reconstructions without the computational overhead of re-optimizing the entire scene. This paper introduces CL-Splats, which incrementally updates Gaussian splatting-based 3D representations from sparse scene captures. CL-Splats integrates a robust change-detection module that segments updated and static components within the scene, enabling focused, local optimization that avoids unnecessary re-computation. Moreover, CL-Splats supports storing and recovering previous scene states, facilitating temporal segmentation and new scene-analysis applications. Our extensive experiments demonstrate that CL-Splats achieves efficient updates with improved reconstruction quality over the state-of-the-art. This establishes a robust foundation for future real-time adaptation in 3D scene reconstruction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14081v1">Capture, Canonicalize, Splat: Zero-Shot 3D Gaussian Avatars from Unstructured Phone Images</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      We present a novel, zero-shot pipeline for creating hyperrealistic, identity-preserving 3D avatars from a few unstructured phone images. Existing methods face several challenges: single-view approaches suffer from geometric inconsistencies and hallucinations, degrading identity preservation, while models trained on synthetic data fail to capture high-frequency details like skin wrinkles and fine hair, limiting realism. Our method introduces two key contributions: (1) a generative canonicalization module that processes multiple unstructured views into a standardized, consistent representation, and (2) a transformer-based model trained on a new, large-scale dataset of high-fidelity Gaussian splatting avatars derived from dome captures of real people. This "Capture, Canonicalize, Splat" pipeline produces static quarter-body avatars with compelling realism and robust identity preservation from unstructured photos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13978v1">Instant Skinned Gaussian Avatars for Web, Mobile and VR Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Accepted to SUI 2025 Demo Track
    </div>
    <details class="paper-abstract">
      We present Instant Skinned Gaussian Avatars, a real-time and cross-platform 3D avatar system. Many approaches have been proposed to animate Gaussian Splatting, but they often require camera arrays, long preprocessing times, or high-end GPUs. Some methods attempt to convert Gaussian Splatting into mesh-based representations, achieving lightweight performance but sacrificing visual fidelity. In contrast, our system efficiently animates Gaussian Splatting by leveraging parallel splat-wise processing to dynamically follow the underlying skinned mesh in real time while preserving high visual fidelity. From smartphone-based 3D scanning to on-device preprocessing, the entire process takes just around five minutes, with the avatar generation step itself completed in only about 30 seconds. Our system enables users to instantly transform their real-world appearance into a 3D avatar, making it ideal for seamless integration with social media and metaverse applications. Website: https://sites.google.com/view/gaussian-vrm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17864v1">InsideOut: Integrated RGB-Radiative Gaussian Splatting for Comprehensive 3D Object Representation</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Published at ICCV 2025
    </div>
    <details class="paper-abstract">
      We introduce InsideOut, an extension of 3D Gaussian splatting (3DGS) that bridges the gap between high-fidelity RGB surface details and subsurface X-ray structures. The fusion of RGB and X-ray imaging is invaluable in fields such as medical diagnostics, cultural heritage restoration, and manufacturing. We collect new paired RGB and X-ray data, perform hierarchical fitting to align RGB and X-ray radiative Gaussian splats, and propose an X-ray reference loss to ensure consistent internal structures. InsideOut effectively addresses the challenges posed by disparate data representations between the two modalities and limited paired datasets. This approach significantly extends the applicability of 3DGS, enhancing visualization, simulation, and non-destructive testing capabilities across various domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12768v1">Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Project page: https://tamu-visual-ai.github.io/usplat4d/
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic 3D scenes from monocular input is fundamentally under-constrained, with ambiguities arising from occlusion and extreme novel views. While dynamic Gaussian Splatting offers an efficient representation, vanilla models optimize all Gaussian primitives uniformly, ignoring whether they are well or poorly observed. This limitation leads to motion drifts under occlusion and degraded synthesis when extrapolating to unseen views. We argue that uncertainty matters: Gaussians with recurring observations across views and time act as reliable anchors to guide motion, whereas those with limited visibility are treated as less reliable. To this end, we introduce USplat4D, a novel Uncertainty-aware dynamic Gaussian Splatting framework that propagates reliable motion cues to enhance 4D reconstruction. Our key insight is to estimate time-varying per-Gaussian uncertainty and leverages it to construct a spatio-temporal graph for uncertainty-aware optimization. Experiments on diverse real and synthetic datasets show that explicitly modeling uncertainty consistently improves dynamic Gaussian Splatting models, yielding more stable geometry under occlusion and high-quality synthesis at extreme viewpoints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12493v1">BSGS: Bi-stage 3D Gaussian Splatting for Camera Motion Deblurring</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has exhibited remarkable capabilities in 3D scene reconstruction.However, reconstructing high-quality 3D scenes from motion-blurred images caused by camera motion poses a significant challenge.The performance of existing 3DGS-based deblurring methods are limited due to their inherent mechanisms, such as extreme dependence on the accuracy of camera poses and inability to effectively control erroneous Gaussian primitives densification caused by motion blur.To solve these problems, we introduce a novel framework, Bi-Stage 3D Gaussian Splatting, to accurately reconstruct 3D scenes from motion-blurred images.BSGS contains two stages. First, Camera Pose Refinement roughly optimizes camera poses to reduce motion-induced distortions. Second, with fixed rough camera poses, Global RigidTransformation further corrects motion-induced blur distortions.To alleviate multi-subframe gradient conflicts, we propose a subframe gradient aggregation strategy to optimize both stages.Furthermore, a space-time bi-stage optimization strategy is introduced to dynamically adjust primitive densification thresholds and prevent premature noisy Gaussian generation in blurred regions. Comprehensive experiments verify the effectiveness of our proposed deblurring method and show its superiority over the state of the arts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12308v1">Hybrid Gaussian Splatting for Novel Urban View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 ICCV 2025 RealADSim Workshop
    </div>
    <details class="paper-abstract">
      This paper describes the Qualcomm AI Research solution to the RealADSim-NVS challenge, hosted at the RealADSim Workshop at ICCV 2025. The challenge concerns novel view synthesis in street scenes, and participants are required to generate, starting from car-centric frames captured during some training traversals, renders of the same urban environment as viewed from a different traversal (e.g. different street lane or car direction). Our solution is inspired by hybrid methods in scene generation and generative simulators merging gaussian splatting and diffusion models, and it is composed of two stages: First, we fit a 3D reconstruction of the scene and render novel views as seen from the target cameras. Then, we enhance the resulting frames with a dedicated single-step diffusion model. We discuss specific choices made in the initialization of gaussian primitives as well as the finetuning of the enhancer model and its training data curation. We report the performance of our model design and we ablate its components in terms of novel view quality as measured by PSNR, SSIM and LPIPS. On the public leaderboard reporting test results, our proposal reaches an aggregated score of 0.432, achieving the second place overall.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12282v1">PAGS: Priority-Adaptive Gaussian Splatting for Dynamic Driving Scenes</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic 3D urban scenes is crucial for autonomous driving, yet current methods face a stark trade-off between fidelity and computational cost. This inefficiency stems from their semantically agnostic design, which allocates resources uniformly, treating static backgrounds and safety-critical objects with equal importance. To address this, we introduce Priority-Adaptive Gaussian Splatting (PAGS), a framework that injects task-aware semantic priorities directly into the 3D reconstruction and rendering pipeline. PAGS introduces two core contributions: (1) Semantically-Guided Pruning and Regularization strategy, which employs a hybrid importance metric to aggressively simplify non-critical scene elements while preserving fine-grained details on objects vital for navigation. (2) Priority-Driven Rendering pipeline, which employs a priority-based depth pre-pass to aggressively cull occluded primitives and accelerate the final shading computations. Extensive experiments on the Waymo and KITTI datasets demonstrate that PAGS achieves exceptional reconstruction quality, particularly on safety-critical objects, while significantly reducing training time and boosting rendering speeds to over 350 FPS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12174v1">UniGS: Unified Geometry-Aware Gaussian Splatting for Multimodal Rendering</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      In this paper, we propose UniGS, a unified map representation and differentiable framework for high-fidelity multimodal 3D reconstruction based on 3D Gaussian Splatting. Our framework integrates a CUDA-accelerated rasterization pipeline capable of rendering photo-realistic RGB images, geometrically accurate depth maps, consistent surface normals, and semantic logits simultaneously. We redesign the rasterization to render depth via differentiable ray-ellipsoid intersection rather than using Gaussian centers, enabling effective optimization of rotation and scale attribute through analytic depth gradients. Furthermore, we derive the analytic gradient formulation for surface normal rendering, ensuring geometric consistency among reconstructed 3D scenes. To improve computational and storage efficiency, we introduce a learnable attribute that enables differentiable pruning of Gaussians with minimal contribution during training. Quantitative and qualitative experiments demonstrate state-of-the-art reconstruction accuracy across all modalities, validating the efficacy of our geometry-aware paradigm. Source code and multimodal viewer will be available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12099v1">G4Splat: Geometry-Guided Gaussian Splatting with Generative Prior</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Project page: https://dali-jack.github.io/g4splat-web/
    </div>
    <details class="paper-abstract">
      Despite recent advances in leveraging generative prior from pre-trained diffusion models for 3D scene reconstruction, existing methods still face two critical limitations. First, due to the lack of reliable geometric supervision, they struggle to produce high-quality reconstructions even in observed regions, let alone in unobserved areas. Second, they lack effective mechanisms to mitigate multi-view inconsistencies in the generated images, leading to severe shape-appearance ambiguities and degraded scene geometry. In this paper, we identify accurate geometry as the fundamental prerequisite for effectively exploiting generative models to enhance 3D scene reconstruction. We first propose to leverage the prevalence of planar structures to derive accurate metric-scale depth maps, providing reliable supervision in both observed and unobserved regions. Furthermore, we incorporate this geometry guidance throughout the generative pipeline to improve visibility mask estimation, guide novel view selection, and enhance multi-view consistency when inpainting with video diffusion models, resulting in accurate and consistent scene completion. Extensive experiments on Replica, ScanNet++, and DeepBlending show that our method consistently outperforms existing baselines in both geometry and appearance reconstruction, particularly for unobserved regions. Moreover, our method naturally supports single-view inputs and unposed videos, with strong generalizability in both indoor and outdoor scenarios with practical real-world applicability. The project page is available at https://dali-jack.github.io/g4splat-web/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11717v1">Ev4DGS: Novel-view Rendering of Non-Rigid Objects from Monocular Event Streams</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Event cameras offer various advantages for novel view rendering compared to synchronously operating RGB cameras, and efficient event-based techniques supporting rigid scenes have been recently demonstrated in the literature. In the case of non-rigid objects, however, existing approaches additionally require sparse RGB inputs, which can be a substantial practical limitation; it remains unknown if similar models could be learned from event streams only. This paper sheds light on this challenging open question and introduces Ev4DGS, i.e., the first approach for novel view rendering of non-rigidly deforming objects in the explicit observation space (i.e., as RGB or greyscale images) from monocular event streams. Our method regresses a deformable 3D Gaussian Splatting representation through 1) a loss relating the outputs of the estimated model with the 2D event observation space, and 2) a coarse 3D deformation model trained from binary masks generated from events. We perform experimental comparisons on existing synthetic and newly recorded real datasets with non-rigid objects. The results demonstrate the validity of Ev4DGS and its superior performance compared to multiple naive baselines that can be applied in our setting. We will release our models and the datasets used in the evaluation for research purposes; see the project webpage: https://4dqv.mpi-inf.mpg.de/Ev4DGS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11689v1">Phys2Real: Fusing VLM Priors with Interactive Online Adaptation for Uncertainty-Aware Sim-to-Real Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Learning robotic manipulation policies directly in the real world can be expensive and time-consuming. While reinforcement learning (RL) policies trained in simulation present a scalable alternative, effective sim-to-real transfer remains challenging, particularly for tasks that require precise dynamics. To address this, we propose Phys2Real, a real-to-sim-to-real RL pipeline that combines vision-language model (VLM)-inferred physical parameter estimates with interactive adaptation through uncertainty-aware fusion. Our approach consists of three core components: (1) high-fidelity geometric reconstruction with 3D Gaussian splatting, (2) VLM-inferred prior distributions over physical parameters, and (3) online physical parameter estimation from interaction data. Phys2Real conditions policies on interpretable physical parameters, refining VLM predictions with online estimates via ensemble-based uncertainty quantification. On planar pushing tasks of a T-block with varying center of mass (CoM) and a hammer with an off-center mass distribution, Phys2Real achieves substantial improvements over a domain randomization baseline: 100% vs 79% success rate for the bottom-weighted T-block, 57% vs 23% in the challenging top-weighted T-block, and 15% faster average task completion for hammer pushing. Ablation studies indicate that the combination of VLM and interaction information is essential for success. Project website: https://phys2real.github.io/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11473v1">VA-GS: Enhancing the Geometric Representation of Gaussian Splatting via View Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has recently emerged as an efficient solution for high-quality and real-time novel view synthesis. However, its capability for accurate surface reconstruction remains underexplored. Due to the discrete and unstructured nature of Gaussians, supervision based solely on image rendering loss often leads to inaccurate geometry and inconsistent multi-view alignment. In this work, we propose a novel method that enhances the geometric representation of 3D Gaussians through view alignment (VA). Specifically, we incorporate edge-aware image cues into the rendering loss to improve surface boundary delineation. To enforce geometric consistency across views, we introduce a visibility-aware photometric alignment loss that models occlusions and encourages accurate spatial relationships among Gaussians. To further mitigate ambiguities caused by lighting variations, we incorporate normal-based constraints to refine the spatial orientation of Gaussians and improve local surface estimation. Additionally, we leverage deep image feature embeddings to enforce cross-view consistency, enhancing the robustness of the learned geometry under varying viewpoints and illumination. Extensive experiments on standard benchmarks demonstrate that our method achieves state-of-the-art performance in both surface reconstruction and novel view synthesis. The source code is available at https://github.com/LeoQLi/VA-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11387v1">MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Accepted by NeurIPS 2025. Project Page: https://wen-yuan-zhang.github.io/MaterialRefGS
    </div>
    <details class="paper-abstract">
      Modeling reflections from 2D images is essential for photorealistic rendering and novel view synthesis. Recent approaches enhance Gaussian primitives with reflection-related material attributes to enable physically based rendering (PBR) with Gaussian Splatting. However, the material inference often lacks sufficient constraints, especially under limited environment modeling, resulting in illumination aliasing and reduced generalization. In this work, we revisit the problem from a multi-view perspective and show that multi-view consistent material inference with more physically-based environment modeling is key to learning accurate reflections with Gaussian Splatting. To this end, we enforce 2D Gaussians to produce multi-view consistent material maps during deferred shading. We also track photometric variations across views to identify highly reflective regions, which serve as strong priors for reflection strength terms. To handle indirect illumination caused by inter-object occlusions, we further introduce an environment modeling strategy through ray tracing with 2DGS, enabling photorealistic rendering of indirect radiance. Experiments on widely used benchmarks show that our method faithfully recovers both illumination and geometry, achieving state-of-the-art rendering quality in novel views synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05111v3">LiDAR-GS:Real-time LiDAR Re-Simulation using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      We present LiDAR-GS, a Gaussian Splatting (GS) method for real-time, high-fidelity re-simulation of LiDAR scans in public urban road scenes. Recent GS methods proposed for cameras have achieved significant advancements in real-time rendering beyond Neural Radiance Fields (NeRF). However, applying GS representation to LiDAR, an active 3D sensor type, poses several challenges that must be addressed to preserve high accuracy and unique characteristics. Specifically, LiDAR-GS designs a differentiable laser beam splatting, using range-view representation for precise surface splatting by projecting lasers onto micro cross-sections, effectively eliminating artifacts associated with local affine approximations. Furthermore, LiDAR-GS leverages Neural Gaussian Representation, which further integrate view-dependent clues, to represent key LiDAR properties that are influenced by the incident direction and external factors. Combining these practices with some essential adaptations, e.g., dynamic instances decomposition, LiDAR-GS succeeds in simultaneously re-simulating depth, intensity, and ray-drop channels, achieving state-of-the-art results in both rendering frame rate and quality on publically available large scene datasets when compared with the methods using explicit mesh or implicit NeRF. Our source code is publicly available at https://www.github.com/cqf7419/LiDAR-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11878v1">GS-Verse: Mesh-based Gaussian Splatting for Physics-aware Interaction in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      As the demand for immersive 3D content grows, the need for intuitive and efficient interaction methods becomes paramount. Current techniques for physically manipulating 3D content within Virtual Reality (VR) often face significant limitations, including reliance on engineering-intensive processes and simplified geometric representations, such as tetrahedral cages, which can compromise visual fidelity and physical accuracy. In this paper, we introduce \our{} (\textbf{G}aussian \textbf{S}platting for \textbf{V}irtual \textbf{E}nvironment \textbf{R}endering and \textbf{S}cene \textbf{E}diting), a novel method designed to overcome these challenges by directly integrating an object's mesh with a Gaussian Splatting (GS) representation. Our approach enables more precise surface approximation, leading to highly realistic deformations and interactions. By leveraging existing 3D mesh assets, \our{} facilitates seamless content reuse and simplifies the development workflow. Moreover, our system is designed to be physics-engine-agnostic, granting developers robust deployment flexibility. This versatile architecture delivers a highly realistic, adaptable, and intuitive approach to interactive 3D manipulation. We rigorously validate our method against the current state-of-the-art technique that couples VR with GS in a comparative user study involving 18 participants. Specifically, we demonstrate that our approach is statistically significantly better for physics-aware stretching manipulation and is also more consistent in other physics-based manipulations like twisting and shaking. Further evaluation across various interactions and scenes confirms that our method consistently delivers high and reliable performance, showing its potential as a plausible alternative to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10691v1">Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      This paper presents a unified framework that allows high-quality dynamic Gaussian Splatting from both defocused and motion-blurred monocular videos. Due to the significant difference between the formation processes of defocus blur and motion blur, existing methods are tailored for either one of them, lacking the ability to simultaneously deal with both of them. Although the two can be jointly modeled as blur kernel-based convolution, the inherent difficulty in estimating accurate blur kernels greatly limits the progress in this direction. In this work, we go a step further towards this direction. Particularly, we propose to estimate per-pixel reliable blur kernels using a blur prediction network that exploits blur-related scene and camera information and is subject to a blur-aware sparsity constraint. Besides, we introduce a dynamic Gaussian densification strategy to mitigate the lack of Gaussians for incomplete regions, and boost the performance of novel view synthesis by incorporating unseen view information to constrain scene optimization. Extensive experiments show that our method outperforms the state-of-the-art methods in generating photorealistic novel view synthesis from defocused and motion-blurred monocular videos. Our code and trained model will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10637v1">High-Fidelity Simulated Data Generation for Real-World Zero-Shot Robotic Manipulation Learning with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The scalability of robotic learning is fundamentally bottlenecked by the significant cost and labor of real-world data collection. While simulated data offers a scalable alternative, it often fails to generalize to the real world due to significant gaps in visual appearance, physical properties, and object interactions. To address this, we propose RoboSimGS, a novel Real2Sim2Real framework that converts multi-view real-world images into scalable, high-fidelity, and physically interactive simulation environments for robotic manipulation. Our approach reconstructs scenes using a hybrid representation: 3D Gaussian Splatting (3DGS) captures the photorealistic appearance of the environment, while mesh primitives for interactive objects ensure accurate physics simulation. Crucially, we pioneer the use of a Multi-modal Large Language Model (MLLM) to automate the creation of physically plausible, articulated assets. The MLLM analyzes visual data to infer not only physical properties (e.g., density, stiffness) but also complex kinematic structures (e.g., hinges, sliding rails) of objects. We demonstrate that policies trained entirely on data generated by RoboSimGS achieve successful zero-shot sim-to-real transfer across a diverse set of real-world manipulation tasks. Furthermore, data from RoboSimGS significantly enhances the performance and generalization capabilities of SOTA methods. Our results validate RoboSimGS as a powerful and scalable solution for bridging the sim-to-real gap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06927v4">CULTURE3D: A Large-Scale and Diverse Dataset of Cultural Landmarks and Terrains for Gaussian-Based Scene Rendering</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Current state-of-the-art 3D reconstruction models face limitations in building extra-large scale outdoor scenes, primarily due to the lack of sufficiently large-scale and detailed datasets. In this paper, we present a extra-large fine-grained dataset with 10 billion points composed of 41,006 drone-captured high-resolution aerial images, covering 20 diverse and culturally significant scenes from worldwide locations such as Cambridge Uni main buildings, the Pyramids, and the Forbidden City Palace. Compared to existing datasets, ours offers significantly larger scale and higher detail, uniquely suited for fine-grained 3D applications. Each scene contains an accurate spatial layout and comprehensive structural information, supporting detailed 3D reconstruction tasks. By reconstructing environments using these detailed images, our dataset supports multiple applications, including outputs in the widely adopted COLMAP format, establishing a novel benchmark for evaluating state-of-the-art large-scale Gaussian Splatting methods.The dataset's flexibility encourages innovations and supports model plug-ins, paving the way for future 3D breakthroughs. All datasets and code will be open-sourced for community use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02691v2">FSFSplatter: Build Surface and Novel Views with Sparse-Views within 2min</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has become a leading reconstruction technique, known for its high-quality novel view synthesis and detailed reconstruction. However, most existing methods require dense, calibrated views. Reconstructing from free sparse images often leads to poor surface due to limited overlap and overfitting. We introduce FSFSplatter, a new approach for fast surface reconstruction from free sparse images. Our method integrates end-to-end dense Gaussian initialization, camera parameter estimation, and geometry-enhanced scene optimization. Specifically, FSFSplatter employs a large Transformer to encode multi-view images and generates a dense and geometrically consistent Gaussian scene initialization via a self-splitting Gaussian head. It eliminates local floaters through contribution-based pruning and mitigates overfitting to limited views by leveraging depth and multi-view feature supervision with differentiable camera parameters during rapid optimization. FSFSplatter outperforms current state-of-the-art methods on widely used DTU, Replica, and BlendedMVS datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10492v1">Towards Efficient 3D Gaussian Human Avatar Compression: A Prior-Guided Framework</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      This paper proposes an efficient 3D avatar coding framework that leverages compact human priors and canonical-to-target transformation to enable high-quality 3D human avatar video compression at ultra-low bit rates. The framework begins by training a canonical Gaussian avatar using articulated splatting in a network-free manner, which serves as the foundation for avatar appearance modeling. Simultaneously, a human-prior template is employed to capture temporal body movements through compact parametric representations. This decomposition of appearance and temporal evolution minimizes redundancy, enabling efficient compression: the canonical avatar is shared across the sequence, requiring compression only once, while the temporal parameters, consisting of just 94 parameters per frame, are transmitted with minimal bit-rate. For each frame, the target human avatar is generated by deforming canonical avatar via Linear Blend Skinning transformation, facilitating temporal coherent video reconstruction and novel view synthesis. Experimental results demonstrate that the proposed method significantly outperforms conventional 2D/3D codecs and existing learnable dynamic 3D Gaussian splatting compression method in terms of rate-distortion performance on mainstream multi-view human video datasets, paving the way for seamless immersive multimedia experiences in meta-verse applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14384v5">Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 ICCV 2025; A novel one-stage 3DGS-based diffusion for 3D object generation and scene reconstruction from a single view in ~6 seconds
    </div>
    <details class="paper-abstract">
      Existing feedforward image-to-3D methods mainly rely on 2D multi-view diffusion models that cannot guarantee 3D consistency. These methods easily collapse when changing the prompt view direction and mainly handle object-centric cases. In this paper, we propose a novel single-stage 3D diffusion model, DiffusionGS, for object generation and scene reconstruction from a single view. DiffusionGS directly outputs 3D Gaussian point clouds at each timestep to enforce view consistency and allow the model to generate robustly given prompt views of any directions, beyond object-centric inputs. Plus, to improve the capability and generality of DiffusionGS, we scale up 3D training data by developing a scene-object mixed training strategy. Experiments show that DiffusionGS yields improvements of 2.20 dB/23.25 and 1.34 dB/19.16 in PSNR/FID for objects and scenes than the state-of-the-art methods, without depth estimator. Plus, our method enjoys over 5$\times$ faster speed ($\sim$6s on an A100 GPU). Our Project page at https://caiyuanhao1998.github.io/project/DiffusionGS/ shows the video and interactive results. The code and models are publicly available at https://github.com/caiyuanhao1998/Open-DiffusionGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10257v1">Opacity-Gradient Driven Density Control for Compact and Efficient Few-Shot 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) struggles in few-shot scenarios, where its standard adaptive density control (ADC) can lead to overfitting and bloated reconstructions. While state-of-the-art methods like FSGS improve quality, they often do so by significantly increasing the primitive count. This paper presents a framework that revises the core 3DGS optimization to prioritize efficiency. We replace the standard positional gradient heuristic with a novel densification trigger that uses the opacity gradient as a lightweight proxy for rendering error. We find this aggressive densification is only effective when paired with a more conservative pruning schedule, which prevents destructive optimization cycles. Combined with a standard depth-correlation loss for geometric guidance, our framework demonstrates a fundamental improvement in efficiency. On the 3-view LLFF dataset, our model is over 40% more compact (32k vs. 57k primitives) than FSGS, and on the Mip-NeRF 360 dataset, it achieves a reduction of approximately 70%. This dramatic gain in compactness is achieved with a modest trade-off in reconstruction metrics, establishing a new state-of-the-art on the quality-vs-efficiency Pareto frontier for few-shot view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10152v1">Color3D: Controllable and Consistent 3D Colorization with Personalized Colorizer</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 Project Page https://yecongwan.github.io/Color3D/
    </div>
    <details class="paper-abstract">
      In this work, we present Color3D, a highly adaptable framework for colorizing both static and dynamic 3D scenes from monochromatic inputs, delivering visually diverse and chromatically vibrant reconstructions with flexible user-guided control. In contrast to existing methods that focus solely on static scenarios and enforce multi-view consistency by averaging color variations which inevitably sacrifice both chromatic richness and controllability, our approach is able to preserve color diversity and steerability while ensuring cross-view and cross-time consistency. In particular, the core insight of our method is to colorize only a single key view and then fine-tune a personalized colorizer to propagate its color to novel views and time steps. Through personalization, the colorizer learns a scene-specific deterministic color mapping underlying the reference view, enabling it to consistently project corresponding colors to the content in novel views and video frames via its inherent inductive bias. Once trained, the personalized colorizer can be applied to infer consistent chrominance for all other images, enabling direct reconstruction of colorful 3D scenes with a dedicated Lab color space Gaussian splatting representation. The proposed framework ingeniously recasts complicated 3D colorization as a more tractable single image paradigm, allowing seamless integration of arbitrary image colorization models with enhanced flexibility and controllability. Extensive experiments across diverse static and dynamic 3D colorization benchmarks substantiate that our method can deliver more consistent and chromatically rich renderings with precise user control. Project Page https://yecongwan.github.io/Color3D/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24796v2">TC-GS: A Faster Gaussian Splatting Module Utilizing Tensor Cores</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) renders pixels by rasterizing Gaussian primitives, where conditional alpha-blending dominates the computational cost in the rendering pipeline. This paper proposes TC-GS, an algorithm-independent universal module that expands the applicability of Tensor Core (TCU) for 3DGS, leading to substantial speedups and seamless integration into existing 3DGS optimization frameworks. The key innovation lies in mapping alpha computation to matrix multiplication, fully utilizing otherwise idle TCUs in existing 3DGS implementations. TC-GS provides plug-and-play acceleration for existing top-tier acceleration algorithms and integrates seamlessly with rendering pipeline designs, such as Gaussian compression and redundancy elimination algorithms. Additionally, we introduce a global-to-local coordinate transformation to mitigate rounding errors from quadratic terms of pixel coordinates caused by Tensor Core half-precision computation. Extensive experiments demonstrate that our method maintains rendering quality while providing an additional 2.18x speedup over existing Gaussian acceleration algorithms, thereby achieving a total acceleration of up to 5.6x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10097v1">Gesplat: Robust Pose-Free 3D Reconstruction via Geometry-Guided Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have advanced 3D reconstruction and novel view synthesis, but remain heavily dependent on accurate camera poses and dense viewpoint coverage. These requirements limit their applicability in sparse-view settings, where pose estimation becomes unreliable and supervision is insufficient. To overcome these challenges, we introduce Gesplat, a 3DGS-based framework that enables robust novel view synthesis and geometrically consistent reconstruction from unposed sparse images. Unlike prior works that rely on COLMAP for sparse point cloud initialization, we leverage the VGGT foundation model to obtain more reliable initial poses and dense point clouds. Our approach integrates several key innovations: 1) a hybrid Gaussian representation with dual position-shape optimization enhanced by inter-view matching consistency; 2) a graph-guided attribute refinement module to enhance scene details; and 3) flow-based depth regularization that improves depth estimation accuracy for more effective supervision. Comprehensive quantitative and qualitative experiments demonstrate that our approach achieves more robust performance on both forward-facing and large-scale complex datasets compared to other pose-free methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10030v1">P-4DGS: Predictive 4D Gaussian Splatting with 90$\times$ Compression</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has garnered significant attention due to its superior scene representation fidelity and real-time rendering performance, especially for dynamic 3D scene reconstruction (\textit{i.e.}, 4D reconstruction). However, despite achieving promising results, most existing algorithms overlook the substantial temporal and spatial redundancies inherent in dynamic scenes, leading to prohibitive memory consumption. To address this, we propose P-4DGS, a novel dynamic 3DGS representation for compact 4D scene modeling. Inspired by intra- and inter-frame prediction techniques commonly used in video compression, we first design a 3D anchor point-based spatial-temporal prediction module to fully exploit the spatial-temporal correlations across different 3D Gaussian primitives. Subsequently, we employ an adaptive quantization strategy combined with context-based entropy coding to further reduce the size of the 3D anchor points, thereby achieving enhanced compression efficiency. To evaluate the rate-distortion performance of our proposed P-4DGS in comparison with other dynamic 3DGS representations, we conduct extensive experiments on both synthetic and real-world datasets. Experimental results demonstrate that our approach achieves state-of-the-art reconstruction quality and the fastest rendering speed, with a remarkably low storage footprint (around \textbf{1MB} on average), achieving up to \textbf{40$\times$} and \textbf{90$\times$} compression on synthetic and real-world scenes, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09997v1">CLoD-GS: Continuous Level-of-Detail via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Level of Detail (LoD) is a fundamental technique in real-time computer graphics for managing the rendering costs of complex scenes while preserving visual fidelity. Traditionally, LoD is implemented using discrete levels (DLoD), where multiple, distinct versions of a model are swapped out at different distances. This long-standing paradigm, however, suffers from two major drawbacks: it requires significant storage for multiple model copies and causes jarring visual ``popping" artifacts during transitions, degrading the user experience. We argue that the explicit, primitive-based nature of the emerging 3D Gaussian Splatting (3DGS) technique enables a more ideal paradigm: Continuous LoD (CLoD). A CLoD approach facilitates smooth, seamless quality scaling within a single, unified model, thereby circumventing the core problems of DLOD. To this end, we introduce CLoD-GS, a framework that integrates a continuous LoD mechanism directly into a 3DGS representation. Our method introduces a learnable, distance-dependent decay parameter for each Gaussian primitive, which dynamically adjusts its opacity based on viewpoint proximity. This allows for the progressive and smooth filtering of less significant primitives, effectively creating a continuous spectrum of detail within one model. To train this model to be robust across all distances, we introduce a virtual distance scaling mechanism and a novel coarse-to-fine training strategy with rendered point count regularization. Our approach not only eliminates the storage overhead and visual artifacts of discrete methods but also reduces the primitive count and memory footprint of the final model. Extensive experiments demonstrate that CLoD-GS achieves smooth, quality-scalable rendering from a single model, delivering high-fidelity results across a wide range of performance targets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25626v2">LLM-Powered Code Analysis and Optimization for Gaussian Splatting Kernels</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 Updated Figure 12
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) is a transformative technique with profound implications on novel view synthesis and real-time rendering. Given its importance, there have been many attempts to improve its performance. However, with the increasing complexity of GPU architectures and the vast search space of performance-tuning parameters, it is a challenging task. Although manual optimizations have achieved remarkable speedups, they require domain expertise and the optimization process can be highly time consuming and error prone. In this paper, we propose to exploit large language models (LLMs) to analyze and optimize Gaussian splatting kernels. To our knowledge, this is the first work to use LLMs to optimize highly specialized real-world GPU kernels. We reveal the intricacies of using LLMs for code optimization and analyze the code optimization techniques from the LLMs. We also propose ways to collaborate with LLMs to further leverage their capabilities. For the original 3DGS code on the MipNeRF360 datasets, LLMs achieve significant speedups, 19% with Deepseek and 24% with GPT-5, demonstrating the different capabilities of different LLMs. By feeding additional information from performance profilers, the performance improvement from LLM-optimized code is enhanced to up to 42% and 38% on average. In comparison, our best-effort manually optimized version can achieve a performance improvement up to 48% and 39% on average, showing that there are still optimizations beyond the capabilities of current LLMs. On the other hand, even upon a newly proposed 3DGS framework with algorithmic optimizations, Seele, LLMs can still further enhance its performance by 6%, showing that there are optimization opportunities missed by domain experts. This highlights the potential of collaboration between domain experts and LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09962v1">VG-Mapping: Variation-Aware 3D Gaussians for Online Semi-static Scene Mapping</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Maintaining an up-to-date map that accurately reflects recent changes in the environment is crucial, especially for robots that repeatedly traverse the same space. Failing to promptly update the changed regions can degrade map quality, resulting in poor localization, inefficient operations, and even lost robots. 3D Gaussian Splatting (3DGS) has recently seen widespread adoption in online map reconstruction due to its dense, differentiable, and photorealistic properties, yet accurately and efficiently updating the regions of change remains a challenge. In this paper, we propose VG-Mapping, a novel online 3DGS-based mapping system tailored for such semi-static scenes. Our approach introduces a hybrid representation that augments 3DGS with a TSDF-based voxel map to efficiently identify changed regions in a scene, along with a variation-aware density control strategy that inserts or deletes Gaussian primitives in regions undergoing change. Furthermore, to address the absence of public benchmarks for this task, we construct a RGB-D dataset comprising both synthetic and real-world semi-static environments. Experimental results demonstrate that our method substantially improves the rendering quality and map update efficiency in semi-static scenes. The code and dataset are available at https://github.com/heyicheng-never/VG-Mapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15966v3">Gaussian Scenes: Pose-Free Sparse-View Scene Reconstruction using Depth-Enhanced Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted to TMLR 2025. Code and models released at https://gaussianscenes.github.io/
    </div>
    <details class="paper-abstract">
      In this work, we introduce a generative approach for pose-free (without camera parameters) reconstruction of 360 scenes from a sparse set of 2D images. Pose-free scene reconstruction from incomplete, pose-free observations is usually regularized with depth estimation or 3D foundational priors. While recent advances have enabled sparse-view reconstruction of large complex scenes (with high degree of foreground and background detail) with known camera poses using view-conditioned generative priors, these methods cannot be directly adapted for the pose-free setting when ground-truth poses are not available during evaluation. To address this, we propose an image-to-image generative model designed to inpaint missing details and remove artifacts in novel view renders and depth maps of a 3D scene. We introduce context and geometry conditioning using Feature-wise Linear Modulation (FiLM) modulation layers as a lightweight alternative to cross-attention and also propose a novel confidence measure for 3D Gaussian splat representations to allow for better detection of these artifacts. By progressively integrating these novel views in a Gaussian-SLAM-inspired process, we achieve a multi-view-consistent 3D representation. Evaluations on the MipNeRF360 and DL3DV-10K benchmark dataset demonstrate that our method surpasses existing pose-free techniques and performs competitively with state-of-the-art posed (precomputed camera parameters are given) reconstruction methods in complex 360 scenes. Our project page provides additional results, videos, and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09586v1">Vision Language Models: A Survey of 26K Papers</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 VLM/LLM Learning Notes
    </div>
    <details class="paper-abstract">
      We present a transparent, reproducible measurement of research trends across 26,104 accepted papers from CVPR, ICLR, and NeurIPS spanning 2023-2025. Titles and abstracts are normalized, phrase-protected, and matched against a hand-crafted lexicon to assign up to 35 topical labels and mine fine-grained cues about tasks, architectures, training regimes, objectives, datasets, and co-mentioned modalities. The analysis quantifies three macro shifts: (1) a sharp rise of multimodal vision-language-LLM work, which increasingly reframes classic perception as instruction following and multi-step reasoning; (2) steady expansion of generative methods, with diffusion research consolidating around controllability, distillation, and speed; and (3) resilient 3D and video activity, with composition moving from NeRFs to Gaussian splatting and a growing emphasis on human- and agent-centric understanding. Within VLMs, parameter-efficient adaptation like prompting/adapters/LoRA and lightweight vision-language bridges dominate; training practice shifts from building encoders from scratch to instruction tuning and finetuning strong backbones; contrastive objectives recede relative to cross-entropy/ranking and distillation. Cross-venue comparisons show CVPR has a stronger 3D footprint and ICLR the highest VLM share, while reliability themes such as efficiency or robustness diffuse across areas. We release the lexicon and methodology to enable auditing and extension. Limitations include lexicon recall and abstract-only scope, but the longitudinal signals are consistent across venues and years.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09537v1">FLOWING: Implicit Neural Flows for Structure-Preserving Morphing</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 10 pages main paper; 9 pages references and appendix
    </div>
    <details class="paper-abstract">
      Morphing is a long-standing problem in vision and computer graphics, requiring a time-dependent warping for feature alignment and a blending for smooth interpolation. Recently, multilayer perceptrons (MLPs) have been explored as implicit neural representations (INRs) for modeling such deformations, due to their meshlessness and differentiability; however, extracting coherent and accurate morphings from standard MLPs typically relies on costly regularizations, which often lead to unstable training and prevent effective feature alignment. To overcome these limitations, we propose FLOWING (FLOW morphING), a framework that recasts warping as the construction of a differential vector flow, naturally ensuring continuity, invertibility, and temporal coherence by encoding structural flow properties directly into the network architectures. This flow-centric approach yields principled and stable transformations, enabling accurate and structure-preserving morphing of both 2D images and 3D shapes. Extensive experiments across a range of applications - including face and image morphing, as well as Gaussian Splatting morphing - show that FLOWING achieves state-of-the-art morphing quality with faster convergence. Code and pretrained models are available at http://schardong.github.io/flowing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09489v1">Two-Stage Gaussian Splatting Optimization for Outdoor Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Outdoor scene reconstruction remains challenging due to the stark contrast between well-textured, nearby regions and distant backgrounds dominated by low detail, uneven illumination, and sky effects. We introduce a two-stage Gaussian Splatting framework that explicitly separates and optimizes these regions, yielding higher-fidelity novel view synthesis. In stage one, background primitives are initialized within a spherical shell and optimized using a loss that combines a background-only photometric term with two geometric regularizers: one constraining Gaussians to remain inside the shell, and another aligning them with local tangential planes. In stage two, foreground Gaussians are initialized from a Structure-from-Motion reconstruction, added and refined using the standard rendering loss, while the background set remains fixed but contributes to the final image formation. Experiments on diverse outdoor datasets show that our method reduces background artifacts and improves perceptual quality compared to state-of-the-art baselines. Moreover, the explicit background separation enables automatic, object-free environment map estimation, opening new possibilities for photorealistic outdoor rendering and mixed-reality applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09364v1">Visibility-Aware Densification for 3D Gaussian Splatting in Dynamic Urban Scenes</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has demonstrated impressive performance in synthesizing high-fidelity novel views. Nonetheless, its effectiveness critically depends on the quality of the initialized point cloud. Specifically, achieving uniform and complete point coverage over the underlying scene structure requires overlapping observation frustums, an assumption that is often violated in unbounded, dynamic urban environments. Training Gaussian models with partially initialized point clouds often leads to distortions and artifacts, as camera rays may fail to intersect valid surfaces, resulting in incorrect gradient propagation to Gaussian primitives associated with occluded or invisible geometry. Additionally, existing densification strategies simply clone and split Gaussian primitives from existing ones, incapable of reconstructing missing structures. To address these limitations, we propose VAD-GS, a 3DGS framework tailored for geometry recovery in challenging urban scenes. Our method identifies unreliable geometry structures via voxel-based visibility reasoning, selects informative supporting views through diversity-aware view selection, and recovers missing structures via patch matching-based multi-view stereo reconstruction. This design enables the generation of new Gaussian primitives guided by reliable geometric priors, even in regions lacking initial points. Extensive experiments on the Waymo and nuScenes datasets demonstrate that VAD-GS outperforms state-of-the-art 3DGS approaches and significantly improves the quality of reconstructed geometry for both static and dynamic objects. Source code will be released upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09881v1">LTGS: Long-Term Gaussian Scene Chronology From Sparse View Updates</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Recent advances in novel-view synthesis can create the photo-realistic visualization of real-world environments from conventional camera captures. However, acquiring everyday environments from casual captures faces challenges due to frequent scene changes, which require dense observations both spatially and temporally. We propose long-term Gaussian scene chronology from sparse-view updates, coined LTGS, an efficient scene representation that can embrace everyday changes from highly under-constrained casual captures. Given an incomplete and unstructured Gaussian splatting representation obtained from an initial set of input images, we robustly model the long-term chronology of the scene despite abrupt movements and subtle environmental variations. We construct objects as template Gaussians, which serve as structural, reusable priors for shared object tracks. Then, the object templates undergo a further refinement pipeline that modulates the priors to adapt to temporally varying environments based on few-shot observations. Once trained, our framework is generalizable across multiple time steps through simple transformations, significantly enhancing the scalability for a temporal evolution of 3D environments. As existing datasets do not explicitly represent the long-term real-world changes with a sparse capture setup, we collect real-world datasets to evaluate the practicality of our pipeline. Experiments demonstrate that our framework achieves superior reconstruction quality compared to other baselines while enabling fast and light-weight updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08575v1">ReSplat: Learning Recurrent Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Project page: https://haofeixu.github.io/resplat/
    </div>
    <details class="paper-abstract">
      While feed-forward Gaussian splatting models provide computational efficiency and effectively handle sparse input settings, their performance is fundamentally limited by the reliance on a single forward pass during inference. We propose ReSplat, a feed-forward recurrent Gaussian splatting model that iteratively refines 3D Gaussians without explicitly computing gradients. Our key insight is that the Gaussian splatting rendering error serves as a rich feedback signal, guiding the recurrent network to learn effective Gaussian updates. This feedback signal naturally adapts to unseen data distributions at test time, enabling robust generalization. To initialize the recurrent process, we introduce a compact reconstruction model that operates in a $16 \times$ subsampled space, producing $16 \times$ fewer Gaussians than previous per-pixel Gaussian models. This substantially reduces computational overhead and allows for efficient Gaussian updates. Extensive experiments across varying of input views (2, 8, 16), resolutions ($256 \times 256$ to $540 \times 960$), and datasets (DL3DV and RealEstate10K) demonstrate that our method achieves state-of-the-art performance while significantly reducing the number of Gaussians and improving the rendering speed. Our project page is at https://haofeixu.github.io/resplat/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08566v1">D$^2$GS: Depth-and-Density Guided Gaussian Splatting for Stable and Accurate Sparse-View Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) enable real-time, high-fidelity novel view synthesis (NVS) with explicit 3D representations. However, performance degradation and instability remain significant under sparse-view conditions. In this work, we identify two key failure modes under sparse-view conditions: overfitting in regions with excessive Gaussian density near the camera, and underfitting in distant areas with insufficient Gaussian coverage. To address these challenges, we propose a unified framework D$^2$GS, comprising two key components: a Depth-and-Density Guided Dropout strategy that suppresses overfitting by adaptively masking redundant Gaussians based on density and depth, and a Distance-Aware Fidelity Enhancement module that improves reconstruction quality in under-fitted far-field areas through targeted supervision. Moreover, we introduce a new evaluation metric to quantify the stability of learned Gaussian distributions, providing insights into the robustness of the sparse-view 3DGS. Extensive experiments on multiple datasets demonstrate that our method significantly improves both visual quality and robustness under sparse view conditions. The project page can be found at: https://insta360-research-team.github.io/DDGS-website/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08491v1">Splat the Net: Radiance Fields with Splattable Neural Primitives</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Radiance fields have emerged as a predominant representation for modeling 3D scene appearance. Neural formulations such as Neural Radiance Fields provide high expressivity but require costly ray marching for rendering, whereas primitive-based methods such as 3D Gaussian Splatting offer real-time efficiency through splatting, yet at the expense of representational power. Inspired by advances in both these directions, we introduce splattable neural primitives, a new volumetric representation that reconciles the expressivity of neural models with the efficiency of primitive-based splatting. Each primitive encodes a bounded neural density field parameterized by a shallow neural network. Our formulation admits an exact analytical solution for line integrals, enabling efficient computation of perspectively accurate splatting kernels. As a result, our representation supports integration along view rays without the need for costly ray marching. The primitives flexibly adapt to scene geometry and, being larger than prior analytic primitives, reduce the number required per scene. On novel-view synthesis benchmarks, our approach matches the quality and speed of 3D Gaussian Splatting while using $10\times$ fewer primitives and $6\times$ fewer parameters. These advantages arise directly from the representation itself, without reliance on complex control or adaptation frameworks. The project page is https://vcai.mpi-inf.mpg.de/projects/SplatNet/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16898v3">MonoGSDF: Exploring Monocular Geometric Cues for Gaussian Splatting-Guided Implicit Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Accurate meshing from monocular images remains a key challenge in 3D vision. While state-of-the-art 3D Gaussian Splatting (3DGS) methods excel at synthesizing photorealistic novel views through rasterization-based rendering, their reliance on sparse, explicit primitives severely limits their ability to recover watertight and topologically consistent 3D surfaces.We introduce MonoGSDF, a novel method that couples Gaussian-based primitives with a neural Signed Distance Field (SDF) for high-quality reconstruction. During training, the SDF guides Gaussians' spatial distribution, while at inference, Gaussians serve as priors to reconstruct surfaces, eliminating the need for memory-intensive Marching Cubes. To handle arbitrary-scale scenes, we propose a scaling strategy for robust generalization. A multi-resolution training scheme further refines details and monocular geometric cues from off-the-shelf estimators enhance reconstruction quality. Experiments on real-world datasets show MonoGSDF outperforms prior methods while maintaining efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08096v1">Efficient Label Refinement for Face Parsing Under Extreme Poses Using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted to VCIP 2025 (International Conference on Visual Communications and Image Processing 2025)
    </div>
    <details class="paper-abstract">
      Accurate face parsing under extreme viewing angles remains a significant challenge due to limited labeled data in such poses. Manual annotation is costly and often impractical at scale. We propose a novel label refinement pipeline that leverages 3D Gaussian Splatting (3DGS) to generate accurate segmentation masks from noisy multiview predictions. By jointly fitting two 3DGS models, one to RGB images and one to their initial segmentation maps, our method enforces multiview consistency through shared geometry, enabling the synthesis of pose-diverse training data with only minimal post-processing. Fine-tuning a face parsing model on this refined dataset significantly improves accuracy on challenging head poses, while maintaining strong performance on standard views. Extensive experiments, including human evaluations, demonstrate that our approach achieves superior results compared to state-of-the-art methods, despite requiring no ground-truth 3D annotations and using only a small set of initial images. Our method offers a scalable and effective solution for improving face parsing robustness in real-world settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07944v1">CVD-STORM: Cross-View Video Diffusion with Spatial-Temporal Reconstruction Model for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Generative models have been widely applied to world modeling for environment simulation and future state prediction. With advancements in autonomous driving, there is a growing demand not only for high-fidelity video generation under various controls, but also for producing diverse and meaningful information such as depth estimation. To address this, we propose CVD-STORM, a cross-view video diffusion model utilizing a spatial-temporal reconstruction Variational Autoencoder (VAE) that generates long-term, multi-view videos with 4D reconstruction capabilities under various control inputs. Our approach first fine-tunes the VAE with an auxiliary 4D reconstruction task, enhancing its ability to encode 3D structures and temporal dynamics. Subsequently, we integrate this VAE into the video diffusion process to significantly improve generation quality. Experimental results demonstrate that our model achieves substantial improvements in both FID and FVD metrics. Additionally, the jointly-trained Gaussian Splatting Decoder effectively reconstructs dynamic scenes, providing valuable geometric information for comprehensive scene understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07830v1">PrismGS: Physically-Grounded Anti-Aliasing for High-Fidelity Large-Scale 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently enabled real-time photorealistic rendering in compact scenes, but scaling to large urban environments introduces severe aliasing artifacts and optimization instability, especially under high-resolution (e.g., 4K) rendering. These artifacts, manifesting as flickering textures and jagged edges, arise from the mismatch between Gaussian primitives and the multi-scale nature of urban geometry. While existing ``divide-and-conquer'' pipelines address scalability, they fail to resolve this fidelity gap. In this paper, we propose PrismGS, a physically-grounded regularization framework that improves the intrinsic rendering behavior of 3D Gaussians. PrismGS integrates two synergistic regularizers. The first is pyramidal multi-scale supervision, which enforces consistency by supervising the rendering against a pre-filtered image pyramid. This compels the model to learn an inherently anti-aliased representation that remains coherent across different viewing scales, directly mitigating flickering textures. This is complemented by an explicit size regularization that imposes a physically-grounded lower bound on the dimensions of the 3D Gaussians. This prevents the formation of degenerate, view-dependent primitives, leading to more stable and plausible geometric surfaces and reducing jagged edges. Our method is plug-and-play and compatible with existing pipelines. Extensive experiments on MatrixCity, Mill-19, and UrbanScene3D demonstrate that PrismGS achieves state-of-the-art performance, yielding significant PSNR gains around 1.5 dB against CityGaussian, while maintaining its superior quality and robustness under demanding 4K rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04597v2">Targetless LiDAR-Camera Calibration with Neural Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Project page: https://zang09.github.io/tlc-calib-site
    </div>
    <details class="paper-abstract">
      Accurate LiDAR-camera calibration is crucial for multi-sensor systems. However, traditional methods often rely on physical targets, which are impractical for real-world deployment. Moreover, even carefully calibrated extrinsics can degrade over time due to sensor drift or external disturbances, necessitating periodic recalibration. To address these challenges, we present a Targetless LiDAR-Camera Calibration (TLC-Calib) that jointly optimizes sensor poses with a neural Gaussian-based scene representation. Reliable LiDAR points are frozen as anchor Gaussians to preserve global structure, while auxiliary Gaussians prevent local overfitting under noisy initialization. Our fully differentiable pipeline with photometric and geometric regularization achieves robust and generalizable calibration, consistently outperforming existing targetless methods on KITTI-360, Waymo, and FAST-LIVO2, and surpassing even the provided calibrations in rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07752v1">DEGS: Deformable Event-based 3D Gaussian Splatting from RGB and Event Stream</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted by TVCG
    </div>
    <details class="paper-abstract">
      Reconstructing Dynamic 3D Gaussian Splatting (3DGS) from low-framerate RGB videos is challenging. This is because large inter-frame motions will increase the uncertainty of the solution space. For example, one pixel in the first frame might have more choices to reach the corresponding pixel in the second frame. Event cameras can asynchronously capture rapid visual changes and are robust to motion blur, but they do not provide color information. Intuitively, the event stream can provide deterministic constraints for the inter-frame large motion by the event trajectories. Hence, combining low-temporal-resolution images with high-framerate event streams can address this challenge. However, it is challenging to jointly optimize Dynamic 3DGS using both RGB and event modalities due to the significant discrepancy between these two data modalities. This paper introduces a novel framework that jointly optimizes dynamic 3DGS from the two modalities. The key idea is to adopt event motion priors to guide the optimization of the deformation fields. First, we extract the motion priors encoded in event streams by using the proposed LoCM unsupervised fine-tuning framework to adapt an event flow estimator to a certain unseen scene. Then, we present the geometry-aware data association method to build the event-Gaussian motion correspondence, which is the primary foundation of the pipeline, accompanied by two useful strategies, namely motion decomposition and inter-frame pseudo-label. Extensive experiments show that our method outperforms existing image and event-based approaches across synthetic and real scenes and prove that our method can effectively optimize dynamic 3DGS with the help of event data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07729v1">ComGS: Efficient 3D Object-Scene Composition via Surface Octahedral Probes</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) enables immersive rendering, but realistic 3D object-scene composition remains challenging. Baked appearance and shadow information in GS radiance fields cause inconsistencies when combining objects and scenes. Addressing this requires relightable object reconstruction and scene lighting estimation. For relightable object reconstruction, existing Gaussian-based inverse rendering methods often rely on ray tracing, leading to low efficiency. We introduce Surface Octahedral Probes (SOPs), which store lighting and occlusion information and allow efficient 3D querying via interpolation, avoiding expensive ray tracing. SOPs provide at least a 2x speedup in reconstruction and enable real-time shadow computation in Gaussian scenes. For lighting estimation, existing Gaussian-based inverse rendering methods struggle to model intricate light transport and often fail in complex scenes, while learning-based methods predict lighting from a single image and are viewpoint-sensitive. We observe that 3D object-scene composition primarily concerns the object's appearance and nearby shadows. Thus, we simplify the challenging task of full scene lighting estimation by focusing on the environment lighting at the object's placement. Specifically, we capture a 360 degrees reconstructed radiance field of the scene at the location and fine-tune a diffusion model to complete the lighting. Building on these advances, we propose ComGS, a novel 3D object-scene composition framework. Our method achieves high-quality, real-time rendering at around 28 FPS, produces visually harmonious results with vivid shadows, and requires only 36 seconds for editing. Code and dataset are available at https://nju-3dv.github.io/projects/ComGS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06644v2">RTGS: Real-Time 3D Gaussian Splatting SLAM via Multi-Level Redundancy Reduction</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted by MICRO2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) based Simultaneous Localization and Mapping (SLAM) systems can largely benefit from 3DGS's state-of-the-art rendering efficiency and accuracy, but have not yet been adopted in resource-constrained edge devices due to insufficient speed. Addressing this, we identify notable redundancies across the SLAM pipeline for acceleration. While conceptually straightforward, practical approaches are required to minimize the overhead associated with identifying and eliminating these redundancies. In response, we propose RTGS, an algorithm-hardware co-design framework that comprehensively reduces the redundancies for real-time 3DGS-SLAM on edge. To minimize the overhead, RTGS fully leverages the characteristics of the 3DGS-SLAM pipeline. On the algorithm side, we introduce (1) an adaptive Gaussian pruning step to remove the redundant Gaussians by reusing gradients computed during backpropagation; and (2) a dynamic downsampling technique that directly reuses the keyframe identification and alpha computing steps to eliminate redundant pixels. On the hardware side, we propose (1) a subtile-level streaming strategy and a pixel-level pairwise scheduling strategy that mitigates workload imbalance via a Workload Scheduling Unit (WSU) guided by previous iteration information; (2) a Rendering and Backpropagation (R&B) Buffer that accelerates the rendering backpropagation by reusing intermediate data computed during rendering; and (3) a Gradient Merging Unit (GMU) to reduce intensive memory accesses caused by atomic operations while enabling pipelined aggregation. Integrated into an edge GPU, RTGS achieves real-time performance (>= 30 FPS) on four datasets and three algorithms, with up to 82.5x energy efficiency over the baseline and negligible quality loss. Code is available at https://github.com/UMN-ZhaoLab/RTGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06644v1">RTGS: Real-Time 3D Gaussian Splatting SLAM via Multi-Level Redundancy Reduction</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted by MICRO2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) based Simultaneous Localization and Mapping (SLAM) systems can largely benefit from 3DGS's state-of-the-art rendering efficiency and accuracy, but have not yet been adopted in resource-constrained edge devices due to insufficient speed. Addressing this, we identify notable redundancies across the SLAM pipeline for acceleration. While conceptually straightforward, practical approaches are required to minimize the overhead associated with identifying and eliminating these redundancies. In response, we propose RTGS, an algorithm-hardware co-design framework that comprehensively reduces the redundancies for real-time 3DGS-SLAM on edge. To minimize the overhead, RTGS fully leverages the characteristics of the 3DGS-SLAM pipeline. On the algorithm side, we introduce (1) an adaptive Gaussian pruning step to remove the redundant Gaussians by reusing gradients computed during backpropagation; and (2) a dynamic downsampling technique that directly reuses the keyframe identification and alpha computing steps to eliminate redundant pixels. On the hardware side, we propose (1) a subtile-level streaming strategy and a pixel-level pairwise scheduling strategy that mitigates workload imbalance via a Workload Scheduling Unit (WSU) guided by previous iteration information; (2) a Rendering and Backpropagation (R&B) Buffer that accelerates the rendering backpropagation by reusing intermediate data computed during rendering; and (3) a Gradient Merging Unit (GMU) to reduce intensive memory accesses caused by atomic operations while enabling pipelined aggregation. Integrated into an edge GPU, RTGS achieves real-time performance (>= 30 FPS) on four datasets and three algorithms, with up to 82.5x energy efficiency over the baseline and negligible quality loss. Code is available at https://github.com/UMN-ZhaoLab/RTGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07136v2">LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted by NeurIPS 2025. Project Page: https://langsplat-v2.github.io
    </div>
    <details class="paper-abstract">
      In this paper, we introduce LangSplatV2, which achieves high-dimensional feature splatting at 476.2 FPS and 3D open-vocabulary text querying at 384.6 FPS for high-resolution images, providing a 42 $\times$ speedup and a 47 $\times$ boost over LangSplat respectively, along with improved query accuracy. LangSplat employs Gaussian Splatting to embed 2D CLIP language features into 3D, significantly enhancing speed and learning a precise 3D language field with SAM semantics. Such advancements in 3D language fields are crucial for applications that require language interaction within complex scenes. However, LangSplat does not yet achieve real-time inference performance (8.2 FPS), even with advanced A100 GPUs, severely limiting its broader application. In this paper, we first conduct a detailed time analysis of LangSplat, identifying the heavyweight decoder as the primary speed bottleneck. Our solution, LangSplatV2 assumes that each Gaussian acts as a sparse code within a global dictionary, leading to the learning of a 3D sparse coefficient field that entirely eliminates the need for a heavyweight decoder. By leveraging this sparsity, we further propose an efficient sparse coefficient splatting method with CUDA optimization, rendering high-dimensional feature maps at high quality while incurring only the time cost of splatting an ultra-low-dimensional feature. Our experimental results demonstrate that LangSplatV2 not only achieves better or competitive query accuracy but is also significantly faster. Codes and demos are available at our project page: https://langsplat-v2.github.io.
    </details>
</div>
