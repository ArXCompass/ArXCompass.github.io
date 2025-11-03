# gaussian splatting - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

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
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24734v1">DrivingScene: A Multi-Task Online Feed-Forward 3D Gaussian Splatting Method for Dynamic Driving Scenes</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Autonomous Driving, Novel view Synthesis, Multi task Learning
    </div>
    <details class="paper-abstract">
      Real-time, high-fidelity reconstruction of dynamic driving scenes is challenged by complex dynamics and sparse views, with prior methods struggling to balance quality and efficiency. We propose DrivingScene, an online, feed-forward framework that reconstructs 4D dynamic scenes from only two consecutive surround-view images. Our key innovation is a lightweight residual flow network that predicts the non-rigid motion of dynamic objects per camera on top of a learned static scene prior, explicitly modeling dynamics via scene flow. We also introduce a coarse-to-fine training paradigm that circumvents the instabilities common to end-to-end approaches. Experiments on nuScenes dataset show our image-only method simultaneously generates high-quality depth, scene flow, and 3D Gaussian point clouds online, significantly outperforming state-of-the-art methods in both dynamic reconstruction and novel view synthesis.
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11878v1">GS-Verse: Mesh-based Gaussian Splatting for Physics-aware Interaction in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      As the demand for immersive 3D content grows, the need for intuitive and efficient interaction methods becomes paramount. Current techniques for physically manipulating 3D content within Virtual Reality (VR) often face significant limitations, including reliance on engineering-intensive processes and simplified geometric representations, such as tetrahedral cages, which can compromise visual fidelity and physical accuracy. In this paper, we introduce \our{} (\textbf{G}aussian \textbf{S}platting for \textbf{V}irtual \textbf{E}nvironment \textbf{R}endering and \textbf{S}cene \textbf{E}diting), a novel method designed to overcome these challenges by directly integrating an object's mesh with a Gaussian Splatting (GS) representation. Our approach enables more precise surface approximation, leading to highly realistic deformations and interactions. By leveraging existing 3D mesh assets, \our{} facilitates seamless content reuse and simplifies the development workflow. Moreover, our system is designed to be physics-engine-agnostic, granting developers robust deployment flexibility. This versatile architecture delivers a highly realistic, adaptable, and intuitive approach to interactive 3D manipulation. We rigorously validate our method against the current state-of-the-art technique that couples VR with GS in a comparative user study involving 18 participants. Specifically, we demonstrate that our approach is statistically significantly better for physics-aware stretching manipulation and is also more consistent in other physics-based manipulations like twisting and shaking. Further evaluation across various interactions and scenes confirms that our method consistently delivers high and reliable performance, showing its potential as a plausible alternative to existing methods.
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03538v3">Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 NeurIPS 2025 Spotlight; Project page: https://steveli88.github.io/AsymGS/
    </div>
    <details class="paper-abstract">
      3D reconstruction from in-the-wild images remains a challenging task due to inconsistent lighting conditions and transient distractors. Existing methods typically rely on heuristic strategies to handle the low-quality training data, which often struggle to produce stable and consistent reconstructions, frequently resulting in visual artifacts. In this work, we propose \modelname{}, a novel framework that leverages the stochastic nature of these artifacts: they tend to vary across different training runs due to minor randomness. Specifically, our method trains two 3D Gaussian Splatting (3DGS) models in parallel, enforcing a consistency constraint that encourages convergence on reliable scene geometry while suppressing inconsistent artifacts. To prevent the two models from collapsing into similar failure modes due to confirmation bias, we introduce a divergent masking strategy that applies two complementary masks: a multi-cue adaptive mask and a self-supervised soft mask, which leads to an asymmetric training process of the two models, reducing shared error modes. In addition, to improve the efficiency of model training, we introduce a lightweight variant called Dynamic EMA Proxy, which replaces one of the two models with a dynamically updated Exponential Moving Average (EMA) proxy, and employs an alternating masking strategy to preserve divergence. Extensive experiments on challenging real-world datasets demonstrate that our method consistently outperforms existing approaches while achieving high efficiency. See the project website at https://steveli88.github.io/AsymGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18468v5">RGS-DR: Deferred Reflections and Residual Shading in 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      In this work, we address specular appearance in inverse rendering using 2D Gaussian splatting with deferred shading and argue for a refinement stage to improve specular detail, thereby bridging the gap with reconstruction-only methods. Our pipeline estimates editable material properties and environment illumination while employing a directional residual pass that captures leftover view-dependent effects for further refining novel view synthesis. In contrast to per-Gaussian shading with shortest-axis normals and normal residuals, which tends to result in more noisy geometry and specular appearance, a pixel-deferred surfel formulation with specular residuals yields sharper highlights, cleaner materials, and improved editability. We evaluate our approach on rendering and reconstruction quality on three popular datasets featuring glossy objects, and also demonstrate high-quality relighting and material editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24893v3">HBSplat: Robust Sparse-View Gaussian Reconstruction with Hybrid-Loss Guided Depth and Bidirectional Warping</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 14 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Novel View Synthesis (NVS) from sparse views presents a formidable challenge in 3D reconstruction, where limited multi-view constraints lead to severe overfitting, geometric distortion, and fragmented scenes. While 3D Gaussian Splatting (3DGS) delivers real-time, high-fidelity rendering, its performance drastically deteriorates under sparse inputs, plagued by floating artifacts and structural failures. To address these challenges, we introduce HBSplat, a unified framework that elevates 3DGS by seamlessly integrating robust structural cues, virtual view constraints, and occluded region completion. Our core contributions are threefold: a Hybrid-Loss Depth Estimation module that ensures multi-view consistency by leveraging dense matching priors and integrating reprojection, point propagation, and smoothness constraints; a Bidirectional Warping Virtual View Synthesis method that enforces substantially stronger constraints by creating high-fidelity virtual views through bidirectional depth-image warping and multi-view fusion; and an Occlusion-Aware Reconstruction component that recovers occluded areas using a depth-difference mask and a learning-based inpainting model. Extensive evaluations on LLFF, Blender, and DTU benchmarks validate that HBSplat sets a new state-of-the-art, achieving up to 21.13 dB PSNR and 0.189 LPIPS, while maintaining real-time inference. Code is available at: https://github.com/eternalland/HBSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06967v1">Generating Surface for Text-to-3D using 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Recent advancements in Text-to-3D modeling have shown significant potential for the creation of 3D content. However, due to the complex geometric shapes of objects in the natural world, generating 3D content remains a challenging task. Current methods either leverage 2D diffusion priors to recover 3D geometry, or train the model directly based on specific 3D representations. In this paper, we propose a novel method named DirectGaussian, which focuses on generating the surfaces of 3D objects represented by surfels. In DirectGaussian, we utilize conditional text generation models and the surface of a 3D object is rendered by 2D Gaussian splatting with multi-view normal and texture priors. For multi-view geometric consistency problems, DirectGaussian incorporates curvature constraints on the generated surface during optimization process. Through extensive experiments, we demonstrate that our framework is capable of achieving diverse and high-fidelity 3D content creation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21633v2">SAR-GS: Gaussian Splatting based SAR Images Rendering and Target Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Three-dimensional target reconstruction from synthetic aperture radar (SAR) imagery is crucial for interpreting complex scattering information in SAR data. However, the intricate electromagnetic scattering mechanisms inherent to SAR imaging pose significant reconstruction challenges. Inspired by the remarkable success of 3D Gaussian Splatting (3D-GS) in optical domain reconstruction, this paper presents a novel SAR Differentiable Gaussian Splatting Rasterizer (SDGR) specifically designed for SAR target reconstruction. Our approach combines Gaussian splatting with the Mapping and Projection Algorithm to compute scattering intensities of Gaussian primitives and generate simulated SAR images through SDGR. Subsequently, the loss function between the rendered image and the ground truth image is computed to optimize the Gaussian primitive parameters representing the scene, while a custom CUDA gradient flow is employed to replace automatic differentiation for accelerated gradient computation. Through experiments involving the rendering of simplified architectural targets and SAR images of multiple vehicle targets, we validate the imaging rationality of SDGR on simulated SAR imagery. Furthermore, the effectiveness of our method for target reconstruction is demonstrated on both simulated and real-world datasets containing multiple vehicle targets, with quantitative evaluations conducted to assess its reconstruction performance. Experimental results indicate that our approach can effectively reconstruct the geometric structures and scattering properties of targets, thereby providing a novel solution for 3D reconstruction in the field of SAR imaging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06802v1">Capture and Interact: Rapid 3D Object Acquisition and Rendering with Gaussian Splatting in Unity</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Capturing and rendering three-dimensional (3D) objects in real time remain a significant challenge, yet hold substantial potential for applications in augmented reality, digital twin systems, remote collaboration and prototyping. We present an end-to-end pipeline that leverages 3D Gaussian Splatting (3D GS) to enable rapid acquisition and interactive rendering of real-world objects using a mobile device, cloud processing and a local computer. Users scan an object with a smartphone video, upload it for automated 3D reconstruction, and visualize it interactively in Unity at an average of 150 frames per second (fps) on a laptop. The system integrates mobile capture, cloud-based 3D GS and Unity rendering to support real-time telepresence. Our experiments show that the pipeline processes scans in approximately 10 minutes on a graphics processing unit (GPU) achieving real-time rendering on the laptop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06694v1">SCas4D: Structural Cascaded Optimization for Boosting Persistent 4D Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Published in Transactions on Machine Learning Research (06/2025)
    </div>
    <details class="paper-abstract">
      Persistent dynamic scene modeling for tracking and novel-view synthesis remains challenging due to the difficulty of capturing accurate deformations while maintaining computational efficiency. We propose SCas4D, a cascaded optimization framework that leverages structural patterns in 3D Gaussian Splatting for dynamic scenes. The key idea is that real-world deformations often exhibit hierarchical patterns, where groups of Gaussians share similar transformations. By progressively refining deformations from coarse part-level to fine point-level, SCas4D achieves convergence within 100 iterations per time frame and produces results comparable to existing methods with only one-twentieth of the training iterations. The approach also demonstrates effectiveness in self-supervised articulated object segmentation, novel view synthesis, and dense point tracking tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15690v3">DWTGS: Rethinking Frequency Regularization for Sparse-view 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted to VCIP 2025
    </div>
    <details class="paper-abstract">
      Sparse-view 3D Gaussian Splatting (3DGS) presents significant challenges in reconstructing high-quality novel views, as it often overfits to the widely-varying high-frequency (HF) details of the sparse training views. While frequency regularization can be a promising approach, its typical reliance on Fourier transforms causes difficult parameter tuning and biases towards detrimental HF learning. We propose DWTGS, a framework that rethinks frequency regularization by leveraging wavelet-space losses that provide additional spatial supervision. Specifically, we supervise only the low-frequency (LF) LL subbands at multiple DWT levels, while enforcing sparsity on the HF HH subband in a self-supervised manner. Experiments across benchmarks show that DWTGS consistently outperforms Fourier-based counterparts, as this LF-centric strategy improves generalization and reduces HF hallucinations.
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
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06481v1">Active Next-Best-View Optimization for Risk-Averse Path Planning</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Safe navigation in uncertain environments requires planning methods that integrate risk aversion with active perception. In this work, we present a unified framework that refines a coarse reference path by constructing tail-sensitive risk maps from Average Value-at-Risk statistics on an online-updated 3D Gaussian-splat Radiance Field. These maps enable the generation of locally safe and feasible trajectories. In parallel, we formulate Next-Best-View (NBV) selection as an optimization problem on the SE(3) pose manifold, where Riemannian gradient descent maximizes an expected information gain objective to reduce uncertainty most critical for imminent motion. Our approach advances the state-of-the-art by coupling risk-averse path refinement with NBV planning, while introducing scalable gradient decompositions that support efficient online updates in complex environments. We demonstrate the effectiveness of the proposed framework through extensive computational studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24893v2">HBSplat: Robust Sparse-View Gaussian Reconstruction with Hybrid-Loss Guided Depth and Bidirectional Warping</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 14 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Novel View Synthesis (NVS) from sparse views presents a formidable challenge in 3D reconstruction, where limited multi-view constraints lead to severe overfitting, geometric distortion, and fragmented scenes. While 3D Gaussian Splatting (3DGS) delivers real-time, high-fidelity rendering, its performance drastically deteriorates under sparse inputs, plagued by floating artifacts and structural failures. To address these challenges, we introduce HBSplat, a unified framework that elevates 3DGS by seamlessly integrating robust structural cues, virtual view constraints, and occluded region completion. Our core contributions are threefold: a Hybrid-Loss Depth Estimation module that ensures multi-view consistency by leveraging dense matching priors and integrating reprojection, point propagation, and smoothness constraints; a Bidirectional Warping Virtual View Synthesis method that enforces substantially stronger constraints by creating high-fidelity virtual views through bidirectional depth-image warping and multi-view fusion; and an Occlusion-Aware Reconstruction component that recovers occluded areas using a depth-difference mask and a learning-based inpainting model. Extensive evaluations on LLFF, Blender, and DTU benchmarks validate that HBSplat sets a new state-of-the-art, achieving up to 21.13 dB PSNR and 0.189 LPIPS, while maintaining real-time inference. Code is available at: https://github.com/eternalland/HBSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03538v2">Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 NeurIPS 2025 Spotlight; Project page: https://steveli88.github.io/AsymGS/
    </div>
    <details class="paper-abstract">
      3D reconstruction from in-the-wild images remains a challenging task due to inconsistent lighting conditions and transient distractors. Existing methods typically rely on heuristic strategies to handle the low-quality training data, which often struggle to produce stable and consistent reconstructions, frequently resulting in visual artifacts.In this work, we propose \modelname{}, a novel framework that leverages the stochastic nature of these artifacts: they tend to vary across different training runs due to minor randomness. Specifically, our method trains two 3D Gaussian Splatting (3DGS) models in parallel, enforcing a consistency constraint that encourages convergence on reliable scene geometry while suppressing inconsistent artifacts. To prevent the two models from collapsing into similar failure modes due to confirmation bias, we introduce a divergent masking strategy that applies two complementary masks: a multi-cue adaptive mask and a self-supervised soft mask, which leads to an asymmetric training process of the two models, reducing shared error modes. In addition, to improve the efficiency of model training, we introduce a lightweight variant called Dynamic EMA Proxy, which replaces one of the two models with a dynamically updated Exponential Moving Average (EMA) proxy, and employs an alternating masking strategy to preserve divergence. Extensive experiments on challenging real-world datasets demonstrate that our method consistently outperforms existing approaches while achieving high efficiency. See the project website at https://steveli88.github.io/AsymGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24758v4">ExGS: Extreme 3D Gaussian Compression with Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Neural scene representations, such as 3D Gaussian Splatting (3DGS), have enabled high-quality neural rendering; however, their large storage and transmission costs hinder deployment in resource-constrained environments. Existing compression methods either rely on costly optimization, which is slow and scene-specific, or adopt training-free pruning and quantization, which degrade rendering quality under high compression ratios. In contrast, recent data-driven approaches provide a promising direction to overcome this trade-off, enabling efficient compression while preserving high rendering quality. We introduce ExGS, a novel feed-forward framework that unifies Universal Gaussian Compression (UGC) with GaussPainter for Extreme 3DGS compression. UGC performs re-optimization-free pruning to aggressively reduce Gaussian primitives while retaining only essential information, whereas GaussPainter leverages powerful diffusion priors with mask-guided refinement to restore high-quality renderings from heavily pruned Gaussian scenes. Unlike conventional inpainting, GaussPainter not only fills in missing regions but also enhances visible pixels, yielding substantial improvements in degraded renderings. To ensure practicality, it adopts a lightweight VAE and a one-step diffusion design, enabling real-time restoration. Our framework can even achieve over 100X compression (reducing a typical 354.77 MB model to about 3.31 MB) while preserving fidelity and significantly improving image quality under challenging conditions. These results highlight the central role of diffusion priors in bridging the gap between extreme compression and high-quality neural rendering. Our code repository will be released at: https://github.com/chenttt2001/ExGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05488v1">ArchitectHead: Continuous Level of Detail Control for 3D Gaussian Head Avatars</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has enabled photorealistic and real-time rendering of 3D head avatars. Existing 3DGS-based avatars typically rely on tens of thousands of 3D Gaussian points (Gaussians), with the number of Gaussians fixed after training. However, many practical applications require adjustable levels of detail (LOD) to balance rendering efficiency and visual quality. In this work, we propose "ArchitectHead", the first framework for creating 3D Gaussian head avatars that support continuous control over LOD. Our key idea is to parameterize the Gaussians in a 2D UV feature space and propose a UV feature field composed of multi-level learnable feature maps to encode their latent features. A lightweight neural network-based decoder then transforms these latent features into 3D Gaussian attributes for rendering. ArchitectHead controls the number of Gaussians by dynamically resampling feature maps from the UV feature field at the desired resolutions. This method enables efficient and continuous control of LOD without retraining. Experimental results show that ArchitectHead achieves state-of-the-art (SOTA) quality in self and cross-identity reenactment tasks at the highest LOD, while maintaining near SOTA performance at lower LODs. At the lowest LOD, our method uses only 6.2\% of the Gaussians while the quality degrades moderately (L1 Loss +7.9\%, PSNR --0.97\%, SSIM --0.6\%, LPIPS Loss +24.1\%), and the rendering speed nearly doubles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.03890v8">A Survey on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Ongoing project; Paper list: https://github.com/guikunchen/Awesome3DGS ; Benchmark: https://github.com/guikunchen/3DGS-Benchmarks
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (GS) has emerged as a transformative technique in radiance fields. Unlike mainstream implicit neural models, 3D GS uses millions of learnable 3D Gaussians for an explicit scene representation. Paired with a differentiable rendering algorithm, this approach achieves real-time rendering and unprecedented editability, making it a potential game-changer for 3D reconstruction and representation. In the present paper, we provide the first systematic overview of the recent developments and critical contributions in 3D GS. We begin with a detailed exploration of the underlying principles and the driving forces behind the emergence of 3D GS, laying the groundwork for understanding its significance. A focal point of our discussion is the practical applicability of 3D GS. By enabling unprecedented rendering speed, 3D GS opens up a plethora of applications, ranging from virtual reality to interactive media and beyond. This is complemented by a comparative analysis of leading 3D GS models, evaluated across various benchmark tasks to highlight their performance and practical utility. The survey concludes by identifying current challenges and suggesting potential avenues for future research. Through this survey, we aim to provide a valuable resource for both newcomers and seasoned researchers, fostering further exploration and advancement in explicit radiance field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24758v3">ExGS: Extreme 3D Gaussian Compression with Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Neural scene representations, such as 3D Gaussian Splatting (3DGS), have enabled high-quality neural rendering; however, their large storage and transmission costs hinder deployment in resource-constrained environments. Existing compression methods either rely on costly optimization, which is slow and scene-specific, or adopt training-free pruning and quantization, which degrade rendering quality under high compression ratios. In contrast, recent data-driven approaches provide a promising direction to overcome this trade-off, enabling efficient compression while preserving high rendering quality.We introduce ExGS, a novel feed-forward framework that unifies Universal Gaussian Compression (UGC) with GaussPainter for Extreme 3DGS compression. UGC performs re-optimization-free pruning to aggressively reduce Gaussian primitives while retaining only essential information, whereas GaussPainter leverages powerful diffusion priors with mask-guided refinement to restore high-quality renderings from heavily pruned Gaussian scenes. Unlike conventional inpainting, GaussPainter not only fills in missing regions but also enhances visible pixels, yielding substantial improvements in degraded renderings.To ensure practicality, it adopts a lightweight VAE and a one-step diffusion design, enabling real-time restoration. Our framework can even achieve over 100X compression (reducing a typical 354.77 MB model to about 3.31 MB) while preserving fidelity and significantly improving image quality under challenging conditions. These results highlight the central role of diffusion priors in bridging the gap between extreme compression and high-quality neural rendering.Our code repository will be released at: https://github.com/chenttt2001/ExGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11941v2">TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 ICCV 2025
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction is a long-term challenge in 3D vision. Recent methods extend 3D Gaussian Splatting to dynamic scenes via additional deformation fields and apply explicit constraints like motion flow to guide the deformation. However, they learn motion changes from individual timestamps independently, making it challenging to reconstruct complex scenes, particularly when dealing with violent movement, extreme-shaped geometries, or reflective surfaces. To address the above issue, we design a plug-and-play module called TimeFormer to enable existing deformable 3D Gaussians reconstruction methods with the ability to implicitly model motion patterns from a learning perspective. Specifically, TimeFormer includes a Cross-Temporal Transformer Encoder, which adaptively learns the temporal relationships of deformable 3D Gaussians. Furthermore, we propose a two-stream optimization strategy that transfers the motion knowledge learned from TimeFormer to the base stream during the training phase. This allows us to remove TimeFormer during inference, thereby preserving the original rendering speed. Extensive experiments in the multi-view and monocular dynamic scenes validate qualitative and quantitative improvement brought by TimeFormer. Project Page: https://patrickddj.github.io/TimeFormer/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18122v3">RT-GuIDE: Real-Time Gaussian Splatting for Information-Driven Exploration</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      We propose a framework for active mapping and exploration that leverages Gaussian splatting for constructing dense maps. Further, we develop a GPU-accelerated motion planning algorithm that can exploit the Gaussian map for real-time navigation. The Gaussian map constructed onboard the robot is optimized for both photometric and geometric quality while enabling real-time situational awareness for autonomy. We show through viewpoint selection experiments that our method yields comparable Peak Signal-to-Noise Ratio (PSNR) and similar reconstruction error to state-of-the-art approaches, while being orders of magnitude faster to compute. In closed-loop physics-based simulation and real-world experiments, our algorithm achieves better map quality (at least 0.8dB higher PSNR and more than 16% higher geometric reconstruction accuracy) than maps constructed by a state-of-the-art method, enabling semantic segmentation using off-the-shelf open-set models. Experiment videos and more details can be found on our project page: https://tyuezhan.github.io/RT GuIDE/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03857v1">Optimized Minimal 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 17 pages, 8 figures
    </div>
    <details class="paper-abstract">
      4D Gaussian Splatting has emerged as a new paradigm for dynamic scene representation, enabling real-time rendering of scenes with complex motions. However, it faces a major challenge of storage overhead, as millions of Gaussians are required for high-fidelity reconstruction. While several studies have attempted to alleviate this memory burden, they still face limitations in compression ratio or visual quality. In this work, we present OMG4 (Optimized Minimal 4D Gaussian Splatting), a framework that constructs a compact set of salient Gaussians capable of faithfully representing 4D Gaussian models. Our method progressively prunes Gaussians in three stages: (1) Gaussian Sampling to identify primitives critical to reconstruction fidelity, (2) Gaussian Pruning to remove redundancies, and (3) Gaussian Merging to fuse primitives with similar characteristics. In addition, we integrate implicit appearance compression and generalize Sub-Vector Quantization (SVQ) to 4D representations, further reducing storage while preserving quality. Extensive experiments on standard benchmark datasets demonstrate that OMG4 significantly outperforms recent state-of-the-art methods, reducing model sizes by over 60% while maintaining reconstruction quality. These results position OMG4 as a significant step forward in compact 4D scene representation, opening new possibilities for a wide range of applications. Our source code is available at https://minshirley.github.io/OMG4/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23258v2">OracleGS: Grounding Generative Priors for Sparse-View Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 Project page available at: https://atakan-topaloglu.github.io/oraclegs/
    </div>
    <details class="paper-abstract">
      Sparse-view novel view synthesis is fundamentally ill-posed due to severe geometric ambiguity. Current methods are caught in a trade-off: regressive models are geometrically faithful but incomplete, whereas generative models can complete scenes but often introduce structural inconsistencies. We propose OracleGS, a novel framework that reconciles generative completeness with regressive fidelity for sparse view Gaussian Splatting. Instead of using generative models to patch incomplete reconstructions, our "propose-and-validate" framework first leverages a pre-trained 3D-aware diffusion model to synthesize novel views to propose a complete scene. We then repurpose a multi-view stereo (MVS) model as a 3D-aware oracle to validate the 3D uncertainties of generated views, using its attention maps to reveal regions where the generated views are well-supported by multi-view evidence versus where they fall into regions of high uncertainty due to occlusion, lack of texture, or direct inconsistency. This uncertainty signal directly guides the optimization of a 3D Gaussian Splatting model via an uncertainty-weighted loss. Our approach conditions the powerful generative prior on multi-view geometric evidence, filtering hallucinatory artifacts while preserving plausible completions in under-constrained regions, outperforming state-of-the-art methods on datasets including Mip-NeRF 360 and NeRF Synthetic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09040v4">Motion Blender Gaussian Splatting for Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 CoRL 2025
    </div>
    <details class="paper-abstract">
      Gaussian splatting has emerged as a powerful tool for high-fidelity reconstruction of dynamic scenes. However, existing methods primarily rely on implicit motion representations, such as encoding motions into neural networks or per-Gaussian parameters, which makes it difficult to further manipulate the reconstructed motions. This lack of explicit controllability limits existing methods to replaying recorded motions only, which hinders a wider application in robotics. To address this, we propose Motion Blender Gaussian Splatting (MBGS), a novel framework that uses motion graphs as an explicit and sparse motion representation. The motion of a graph's links is propagated to individual Gaussians via dual quaternion skinning, with learnable weight painting functions that determine the influence of each link. The motion graphs and 3D Gaussians are jointly optimized from input videos via differentiable rendering. Experiments show that MBGS achieves state-of-the-art performance on the highly challenging iPhone dataset while being competitive on HyperNeRF. We demonstrate the application potential of our method in animating novel object poses, synthesizing real robot demonstrations, and predicting robot actions through visual planning. The source code, models, video demonstrations can be found at http://mlzxy.github.io/motion-blender-gs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02884v1">GS-Share: Enabling High-fidelity Map Sharing with Incremental Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 11 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Constructing and sharing 3D maps is essential for many applications, including autonomous driving and augmented reality. Recently, 3D Gaussian splatting has emerged as a promising approach for accurate 3D reconstruction. However, a practical map-sharing system that features high-fidelity, continuous updates, and network efficiency remains elusive. To address these challenges, we introduce GS-Share, a photorealistic map-sharing system with a compact representation. The core of GS-Share includes anchor-based global map construction, virtual-image-based map enhancement, and incremental map update. We evaluate GS-Share against state-of-the-art methods, demonstrating that our system achieves higher fidelity, particularly for extrapolated views, with improvements of 11%, 22%, and 74% in PSNR, LPIPS, and Depth L1, respectively. Furthermore, GS-Share is significantly more compact, reducing map transmission overhead by 36%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05398v2">Learning High-Fidelity Robot Self-Model with Articulated 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 This paper is accepted by IJRR. The code will be open-sourced on GitHub as soon as possible after the paper is officially published
    </div>
    <details class="paper-abstract">
      Self-modeling enables robots to build task-agnostic models of their morphology and kinematics based on data that can be automatically collected, with minimal human intervention and prior information, thereby enhancing machine intelligence. Recent research has highlighted the potential of data-driven technology in modeling the morphology and kinematics of robots. However, existing self-modeling methods suffer from either low modeling quality or excessive data acquisition costs. Beyond morphology and kinematics, texture is also a crucial component of robots, which is challenging to model and remains unexplored. In this work, a high-quality, texture-aware, and link-level method is proposed for robot self-modeling. We utilize three-dimensional (3D) Gaussians to represent the static morphology and texture of robots, and cluster the 3D Gaussians to construct neural ellipsoid bones, whose deformations are controlled by the transformation matrices generated by a kinematic neural network. The 3D Gaussians and kinematic neural network are trained using data pairs composed of joint angles, camera parameters and multi-view images without depth information. By feeding the kinematic neural network with joint angles, we can utilize the well-trained model to describe the corresponding morphology, kinematics and texture of robots at the link level, and render robot images from different perspectives with the aid of 3D Gaussian splatting. Furthermore, we demonstrate that the established model can be exploited to perform downstream tasks such as motion planning and inverse kinematics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02732v1">From Tokens to Nodes: Semantic-Guided Motion Control for Dynamic 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Dynamic 3D reconstruction from monocular videos remains difficult due to the ambiguity inferring 3D motion from limited views and computational demands of modeling temporally varying scenes. While recent sparse control methods alleviate computation by reducing millions of Gaussians to thousands of control points, they suffer from a critical limitation: they allocate points purely by geometry, leading to static redundancy and dynamic insufficiency. We propose a motion-adaptive framework that aligns control density with motion complexity. Leveraging semantic and motion priors from vision foundation models, we establish patch-token-node correspondences and apply motion-adaptive compression to concentrate control points in dynamic regions while suppressing redundancy in static backgrounds. Our approach achieves flexible representational density adaptation through iterative voxelization and motion tendency scoring, directly addressing the fundamental mismatch between control point allocation and motion complexity. To capture temporal evolution, we introduce spline-based trajectory parameterization initialized by 2D tracklets, replacing MLP-based deformation fields to achieve smoother motion representation and more stable optimization. Extensive experiments demonstrate significant improvements in reconstruction quality and efficiency over existing state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02691v1">FSFSplatter: Build Surface and Novel Views with Sparse-Views within 3min</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has become a leading reconstruction technique, known for its high-quality novel view synthesis and detailed reconstruction. However, most existing methods require dense, calibrated views. Reconstructing from free sparse images often leads to poor surface due to limited overlap and overfitting. We introduce FSFSplatter, a new approach for fast surface reconstruction from free sparse images. Our method integrates end-to-end dense Gaussian initialization, camera parameter estimation, and geometry-enhanced scene optimization. Specifically, FSFSplatter employs a large Transformer to encode multi-view images and generates a dense and geometrically consistent Gaussian scene initialization via a self-splitting Gaussian head. It eliminates local floaters through contribution-based pruning and mitigates overfitting to limited views by leveraging depth and multi-view feature supervision with differentiable camera parameters during rapid optimization. FSFSplatter outperforms current state-of-the-art methods on widely used DTU and Replica.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24758v2">ExGS: Extreme 3D Gaussian Compression with Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Neural scene representations, such as 3D Gaussian Splatting (3DGS), have enabled high-quality neural rendering; however, their large storage and transmission costs hinder deployment in resource-constrained environments. Existing compression methods either rely on costly optimization, which is slow and scene-specific, or adopt training-free pruning and quantization, which degrade rendering quality under high compression ratios. In contrast, recent data-driven approaches provide a promising direction to overcome this trade-off, enabling efficient compression while preserving high rendering quality.We introduce ExGS, a novel feed-forward framework that unifies Universal Gaussian Compression (UGC) with GaussPainter for Extreme 3DGS compression. UGC performs re-optimization-free pruning to aggressively reduce Gaussian primitives while retaining only essential information, whereas GaussPainter leverages powerful diffusion priors with mask-guided refinement to restore high-quality renderings from heavily pruned Gaussian scenes. Unlike conventional inpainting, GaussPainter not only fills in missing regions but also enhances visible pixels, yielding substantial improvements in degraded renderings.To ensure practicality, it adopts a lightweight VAE and a one-step diffusion design, enabling real-time restoration. Our framework can even achieve over 100X compression (reducing a typical 354.77 MB model to about 3.31 MB) while preserving fidelity and significantly improving image quality under challenging conditions. These results highlight the central role of diffusion priors in bridging the gap between extreme compression and high-quality neural rendering.Our code repository will be released at: https://github.com/chenttt2001/ExGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14191v2">MCGS-SLAM: A Multi-Camera SLAM Framework Using Gaussian Splatting for High-Fidelity Mapping</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Recent progress in dense SLAM has primarily targeted monocular setups, often at the expense of robustness and geometric coverage. We present MCGS-SLAM, the first purely RGB-based multi-camera SLAM system built on 3D Gaussian Splatting (3DGS). Unlike prior methods relying on sparse maps or inertial data, MCGS-SLAM fuses dense RGB inputs from multiple viewpoints into a unified, continuously optimized Gaussian map. A multi-camera bundle adjustment (MCBA) jointly refines poses and depths via dense photometric and geometric residuals, while a scale consistency module enforces metric alignment across views using low-rank priors. The system supports RGB input and maintains real-time performance at large scale. Experiments on synthetic and real-world datasets show that MCGS-SLAM consistently yields accurate trajectories and photorealistic reconstructions, usually outperforming monocular baselines. Notably, the wide field of view from multi-camera input enables reconstruction of side-view regions that monocular setups miss, critical for safe autonomous operation. These results highlight the promise of multi-camera Gaussian Splatting SLAM for high-fidelity mapping in robotics and autonomous driving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01826v2">GSRF: Complex-Valued 3D Gaussian Splatting for Efficient Radio-Frequency Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Synthesizing radio-frequency (RF) data given the transmitter and receiver positions, e.g., received signal strength indicator (RSSI), is critical for wireless networking and sensing applications, such as indoor localization. However, it remains challenging due to complex propagation interactions, including reflection, diffraction, and scattering. State-of-the-art neural radiance field (NeRF)-based methods achieve high-fidelity RF data synthesis but are limited by long training times and high inference latency. We introduce GSRF, a framework that extends 3D Gaussian Splatting (3DGS) from the optical domain to the RF domain, enabling efficient RF data synthesis. GSRF realizes this adaptation through three key innovations: First, it introduces complex-valued 3D Gaussians with a hybrid Fourier-Legendre basis to model directional and phase-dependent radiance. Second, it employs orthographic splatting for efficient ray-Gaussian intersection identification. Third, it incorporates a complex-valued ray tracing algorithm, executed on RF-customized CUDA kernels and grounded in wavefront propagation principles, to synthesize RF data in real time. Evaluated across various RF technologies, GSRF preserves high-fidelity RF data synthesis while achieving significant improvements in training efficiency, shorter training time, and reduced inference latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03545v1">SketchPlan: Diffusion Based Drone Planning From Human Sketches</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Code available at https://github.com/sixnor/SketchPlan
    </div>
    <details class="paper-abstract">
      We propose SketchPlan, a diffusion-based planner that interprets 2D hand-drawn sketches over depth images to generate 3D flight paths for drone navigation. SketchPlan comprises two components: a SketchAdapter that learns to map the human sketches to projected 2D paths, and DiffPath, a diffusion model that infers 3D trajectories from 2D projections and a first person view depth image. Our model achieves zero-shot sim-to-real transfer, generating accurate and safe flight paths in previously unseen real-world environments. To train the model, we build a synthetic dataset of 32k flight paths using a diverse set of photorealistic 3D Gaussian Splatting scenes. We automatically label the data by computing 2D projections of the 3D flight paths onto the camera plane, and use this to train the DiffPath diffusion model. However, since real human 2D sketches differ significantly from ideal 2D projections, we additionally label 872 of the 3D flight paths with real human sketches and use this to train the SketchAdapter to infer the 2D projection from the human sketch. We demonstrate SketchPlan's effectiveness in both simulated and real-world experiments, and show through ablations that training on a mix of human labeled and auto-labeled data together with a modular design significantly boosts its capabilities to correctly interpret human intent and infer 3D paths. In real-world drone tests, SketchPlan achieved 100\% success in low/medium clutter and 40\% in unseen high-clutter environments, outperforming key ablations by 20-60\% in task completion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08587v1">EGSTalker: Real-Time Audio-Driven Talking Head Generation with Efficient Gaussian Deformation</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Main paper (6 pages). Accepted for publication by IEEE International Conference on Systems, Man, and Cybernetics 2025
    </div>
    <details class="paper-abstract">
      This paper presents EGSTalker, a real-time audio-driven talking head generation framework based on 3D Gaussian Splatting (3DGS). Designed to enhance both speed and visual fidelity, EGSTalker requires only 3-5 minutes of training video to synthesize high-quality facial animations. The framework comprises two key stages: static Gaussian initialization and audio-driven deformation. In the first stage, a multi-resolution hash triplane and a Kolmogorov-Arnold Network (KAN) are used to extract spatial features and construct a compact 3D Gaussian representation. In the second stage, we propose an Efficient Spatial-Audio Attention (ESAA) module to fuse audio and spatial cues, while KAN predicts the corresponding Gaussian deformations. Extensive experiments demonstrate that EGSTalker achieves rendering quality and lip-sync accuracy comparable to state-of-the-art methods, while significantly outperforming them in inference speed. These results highlight EGSTalker's potential for real-time multimedia applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03104v1">Geometry Meets Vision: Revisiting Pretrained Semantics in Distilled Fields</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Semantic distillation in radiance fields has spurred significant advances in open-vocabulary robot policies, e.g., in manipulation and navigation, founded on pretrained semantics from large vision models. While prior work has demonstrated the effectiveness of visual-only semantic features (e.g., DINO and CLIP) in Gaussian Splatting and neural radiance fields, the potential benefit of geometry-grounding in distilled fields remains an open question. In principle, visual-geometry features seem very promising for spatial tasks such as pose estimation, prompting the question: Do geometry-grounded semantic features offer an edge in distilled fields? Specifically, we ask three critical questions: First, does spatial-grounding produce higher-fidelity geometry-aware semantic features? We find that image features from geometry-grounded backbones contain finer structural details compared to their counterparts. Secondly, does geometry-grounding improve semantic object localization? We observe no significant difference in this task. Thirdly, does geometry-grounding enable higher-accuracy radiance field inversion? Given the limitations of prior work and their lack of semantics integration, we propose a novel framework SPINE for inverting radiance fields without an initial guess, consisting of two core components: coarse inversion using distilled semantics, and fine inversion using photometric-based optimization. Surprisingly, we find that the pose estimation accuracy decreases with geometry-grounded features. Our results suggest that visual-only features offer greater versatility for a broader range of downstream tasks, although geometry-grounded features contain more geometric detail. Notably, our findings underscore the necessity of future research on effective strategies for geometry-grounding that augment the versatility and performance of pretrained semantic features.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02069v1">Spec-Gloss Surfels and Normal-Diffuse Priors for Relightable Glossy Objects</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Accurate reconstruction and relighting of glossy objects remain a longstanding challenge, as object shape, material properties, and illumination are inherently difficult to disentangle. Existing neural rendering approaches often rely on simplified BRDF models or parameterizations that couple diffuse and specular components, which restricts faithful material recovery and limits relighting fidelity. We propose a relightable framework that integrates a microfacet BRDF with the specular-glossiness parameterization into 2D Gaussian Splatting with deferred shading. This formulation enables more physically consistent material decomposition, while diffusion-based priors for surface normals and diffuse color guide early-stage optimization and mitigate ambiguity. A coarse-to-fine optimization of the environment map accelerates convergence and preserves high-dynamic-range specular reflections. Extensive experiments on complex, glossy scenes demonstrate that our method achieves high-quality geometry and material reconstruction, delivering substantially more realistic and consistent relighting under novel illumination compared to existing Gaussian splatting methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02034v1">GaussianMorphing: Mesh-Guided 3D Gaussians for Semantic-Aware Object Morphing</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Project page: https://baiyunshu.github.io/GAUSSIANMORPHING.github.io/
    </div>
    <details class="paper-abstract">
      We introduce GaussianMorphing, a novel framework for semantic-aware 3D shape and texture morphing from multi-view images. Previous approaches usually rely on point clouds or require pre-defined homeomorphic mappings for untextured data. Our method overcomes these limitations by leveraging mesh-guided 3D Gaussian Splatting (3DGS) for high-fidelity geometry and appearance modeling. The core of our framework is a unified deformation strategy that anchors 3DGaussians to reconstructed mesh patches, ensuring geometrically consistent transformations while preserving texture fidelity through topology-aware constraints. In parallel, our framework establishes unsupervised semantic correspondence by using the mesh topology as a geometric prior and maintains structural integrity via physically plausible point trajectories. This integrated approach preserves both local detail and global semantic coherence throughout the morphing process with out requiring labeled data. On our proposed TexMorph benchmark, GaussianMorphing substantially outperforms prior 2D/3D methods, reducing color consistency error ($\Delta E$) by 22.2% and EI by 26.2%. Project page: https://baiyunshu.github.io/GAUSSIANMORPHING.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01991v1">4DGS-Craft: Consistent and Interactive 4D Gaussian Splatting Editing</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Recent advances in 4D Gaussian Splatting (4DGS) editing still face challenges with view, temporal, and non-editing region consistency, as well as with handling complex text instructions. To address these issues, we propose 4DGS-Craft, a consistent and interactive 4DGS editing framework. We first introduce a 4D-aware InstructPix2Pix model to ensure both view and temporal consistency. This model incorporates 4D VGGT geometry features extracted from the initial scene, enabling it to capture underlying 4D geometric structures during editing. We further enhance this model with a multi-view grid module that enforces consistency by iteratively refining multi-view input images while jointly optimizing the underlying 4D scene. Furthermore, we preserve the consistency of non-edited regions through a novel Gaussian selection mechanism, which identifies and optimizes only the Gaussians within the edited regions. Beyond consistency, facilitating user interaction is also crucial for effective 4DGS editing. Therefore, we design an LLM-based module for user intent understanding. This module employs a user instruction template to define atomic editing operations and leverages an LLM for reasoning. As a result, our framework can interpret user intent and decompose complex instructions into a logical sequence of atomic operations, enabling it to handle intricate user commands and further enhance editing performance. Compared to related works, our approach enables more consistent and controllable 4D scene editing. Our code will be made available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01978v1">ROI-GS: Interest-based Local Quality 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 4 pages, 3 figures, 2 tables
    </div>
    <details class="paper-abstract">
      We tackle the challenge of efficiently reconstructing 3D scenes with high detail on objects of interest. Existing 3D Gaussian Splatting (3DGS) methods allocate resources uniformly across the scene, limiting fine detail to Regions Of Interest (ROIs) and leading to inflated model size. We propose ROI-GS, an object-aware framework that enhances local details through object-guided camera selection, targeted Object training, and seamless integration of high-fidelity object of interest reconstructions into the global scene. Our method prioritizes higher resolution details on chosen objects while maintaining real-time performance. Experiments show that ROI-GS significantly improves local quality (up to 2.96 dB PSNR), while reducing overall model size by $\approx 17\%$ of baseline and achieving faster training for a scene with a single object of interest, outperforming existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18468v4">RGS-DR: Deferred Reflections and Residual Shading in 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      In this work, we address specular appearance in inverse rendering using 2D Gaussian splatting with deferred shading and argue for a refinement stage to improve specular detail, thereby bridging the gap with reconstruction-only methods. Our pipeline estimates editable material properties and environment illumination while employing a directional residual pass that captures leftover view-dependent effects for further refining novel view synthesis. In contrast to per-Gaussian shading with shortest-axis normals and normal residuals, which tends to result in more noisy geometry and specular appearance, a pixel-deferred surfel formulation with specular residuals yields sharper highlights, cleaner materials, and improved editability. We evaluate our approach on rendering and reconstruction quality on three popular datasets featuring glossy objects, and also demonstrate high-quality relighting and material editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01848v1">GreenhouseSplat: A Dataset of Photorealistic Greenhouse Simulations for Mobile Robotics</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Simulating greenhouse environments is critical for developing and evaluating robotic systems for agriculture, yet existing approaches rely on simplistic or synthetic assets that limit simulation-to-real transfer. Recent advances in radiance field methods, such as Gaussian splatting, enable photorealistic reconstruction but have so far been restricted to individual plants or controlled laboratory conditions. In this work, we introduce GreenhouseSplat, a framework and dataset for generating photorealistic greenhouse assets directly from inexpensive RGB images. The resulting assets are integrated into a ROS-based simulation with support for camera and LiDAR rendering, enabling tasks such as localization with fiducial markers. We provide a dataset of 82 cucumber plants across multiple row configurations and demonstrate its utility for robotics evaluation. GreenhouseSplat represents the first step toward greenhouse-scale radiance-field simulation and offers a foundation for future research in agricultural robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01767v1">LOBE-GS: Load-Balanced and Efficient 3D Gaussian Splatting for Large-Scale Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has established itself as an efficient representation for real-time, high-fidelity 3D scene reconstruction. However, scaling 3DGS to large and unbounded scenes such as city blocks remains difficult. Existing divide-and-conquer methods alleviate memory pressure by partitioning the scene into blocks, but introduce new bottlenecks: (i) partitions suffer from severe load imbalance since uniform or heuristic splits do not reflect actual computational demands, and (ii) coarse-to-fine pipelines fail to exploit the coarse stage efficiently, often reloading the entire model and incurring high overhead. In this work, we introduce LoBE-GS, a novel Load-Balanced and Efficient 3D Gaussian Splatting framework, that re-engineers the large-scale 3DGS pipeline. LoBE-GS introduces a depth-aware partitioning method that reduces preprocessing from hours to minutes, an optimization-based strategy that balances visible Gaussians -- a strong proxy for computational load -- across blocks, and two lightweight techniques, visibility cropping and selective densification, to further reduce training cost. Evaluations on large-scale urban and outdoor datasets show that LoBE-GS consistently achieves up to $2\times$ faster end-to-end training time than state-of-the-art baselines, while maintaining reconstruction quality and enabling scalability to scenes infeasible with vanilla 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01619v1">MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      While there has been significant progress in the field of 3D avatar creation from visual observations, modeling physically plausible dynamics of humans with loose garments remains a challenging problem. Although a few existing works address this problem by leveraging physical simulation, they suffer from limited accuracy or robustness to novel animation inputs. In this work, we present MPMAvatar, a framework for creating 3D human avatars from multi-view videos that supports highly realistic, robust animation, as well as photorealistic rendering from free viewpoints. For accurate and robust dynamics modeling, our key idea is to use a Material Point Method-based simulator, which we carefully tailor to model garments with complex deformations and contact with the underlying body by incorporating an anisotropic constitutive model and a novel collision handling algorithm. We combine this dynamics modeling scheme with our canonical avatar that can be rendered using 3D Gaussian Splatting with quasi-shadowing, enabling high-fidelity rendering for physically realistic animations. In our experiments, we demonstrate that MPMAvatar significantly outperforms the existing state-of-the-art physics-based avatar in terms of (1) dynamics modeling accuracy, (2) rendering accuracy, and (3) robustness and efficiency. Additionally, we present a novel application in which our avatar generalizes to unseen interactions in a zero-shot manner-which was not achievable with previous learning-based methods due to their limited simulation generalizability. Our project page is at: https://KAISTChangmin.github.io/MPMAvatar/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02469v1">SIMSplat: Predictive Driving Scene Editing with Language-aligned 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Driving scene manipulation with sensor data is emerging as a promising alternative to traditional virtual driving simulators. However, existing frameworks struggle to generate realistic scenarios efficiently due to limited editing capabilities. To address these challenges, we present SIMSplat, a predictive driving scene editor with language-aligned Gaussian splatting. As a language-controlled editor, SIMSplat enables intuitive manipulation using natural language prompts. By aligning language with Gaussian-reconstructed scenes, it further supports direct querying of road objects, allowing precise and flexible editing. Our method provides detailed object-level editing, including adding new objects and modifying the trajectories of both vehicles and pedestrians, while also incorporating predictive path refinement through multi-agent motion prediction to generate realistic interactions among all agents in the scene. Experiments on the Waymo dataset demonstrate SIMSplat's extensive editing capabilities and adaptability across a wide range of scenarios. Project page: https://sungyeonparkk.github.io/simsplat/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02314v1">StealthAttack: Robust 3D Gaussian Splatting Poisoning via Density-Guided Illusions</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 ICCV 2025. Project page: https://hentci.github.io/stealthattack/
    </div>
    <details class="paper-abstract">
      3D scene representation methods like Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have significantly advanced novel view synthesis. As these methods become prevalent, addressing their vulnerabilities becomes critical. We analyze 3DGS robustness against image-level poisoning attacks and propose a novel density-guided poisoning method. Our method strategically injects Gaussian points into low-density regions identified via Kernel Density Estimation (KDE), embedding viewpoint-dependent illusory objects clearly visible from poisoned views while minimally affecting innocent views. Additionally, we introduce an adaptive noise strategy to disrupt multi-view consistency, further enhancing attack effectiveness. We propose a KDE-based evaluation protocol to assess attack difficulty systematically, enabling objective benchmarking for future research. Extensive experiments demonstrate our method's superior performance compared to state-of-the-art techniques. Project page: https://hentci.github.io/stealthattack/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02248v1">Performance-Guided Refinement for Visual Aerial Navigation using Editable Gaussian Splatting in FalconGym 2.0</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Visual policy design is crucial for aerial navigation. However, state-of-the-art visual policies often overfit to a single track and their performance degrades when track geometry changes. We develop FalconGym 2.0, a photorealistic simulation framework built on Gaussian Splatting (GSplat) with an Edit API that programmatically generates diverse static and dynamic tracks in milliseconds. Leveraging FalconGym 2.0's editability, we propose a Performance-Guided Refinement (PGR) algorithm, which concentrates visual policy's training on challenging tracks while iteratively improving its performance. Across two case studies (fixed-wing UAVs and quadrotors) with distinct dynamics and environments, we show that a single visual policy trained with PGR in FalconGym 2.0 outperforms state-of-the-art baselines in generalization and robustness: it generalizes to three unseen tracks with 100% success without per-track retraining and maintains higher success rates under gate-pose perturbations. Finally, we demonstrate that the visual policy trained with PGR in FalconGym 2.0 can be zero-shot sim-to-real transferred to a quadrotor hardware, achieving a 98.6% success rate (69 / 70 gates) over 30 trials spanning two three-gate tracks and a moving-gate track.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25075v2">GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Cryo-electron microscopy (cryo-EM) has become a central tool for high-resolution structural biology, yet the massive scale of datasets (often exceeding 100k particle images) renders 3D reconstruction both computationally expensive and memory intensive. Traditional Fourier-space methods are efficient but lose fidelity due to repeated transforms, while recent real-space approaches based on neural radiance fields (NeRFs) improve accuracy but incur cubic memory and computation overhead. Therefore, we introduce GEM, a novel cryo-EM reconstruction framework built on 3D Gaussian Splatting (3DGS) that operates directly in real-space while maintaining high efficiency. Instead of modeling the entire density volume, GEM represents proteins with compact 3D Gaussians, each parameterized by only 11 values. To further improve the training efficiency, we designed a novel gradient computation to 3D Gaussians that contribute to each voxel. This design substantially reduced both memory footprint and training cost. On standard cryo-EM benchmarks, GEM achieves up to 48% faster training and 12% lower memory usage compared to state-of-the-art methods, while improving local resolution by as much as 38.8%. These results establish GEM as a practical and scalable paradigm for cryo-EM reconstruction, unifying speed, efficiency, and high-resolution accuracy. Our code is available at https://github.com/UNITES-Lab/GEM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01383v2">FalconWing: An Ultra-Light Indoor Fixed-Wing UAV Platform for Vision-Based Autonomy</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      We introduce FalconWing, an ultra-light (150 g) indoor fixed-wing UAV platform for vision-based autonomy. Controlled indoor environment enables year-round repeatable UAV experiment but imposes strict weight and maneuverability limits on the UAV, motivating our ultra-light FalconWing design. FalconWing couples a lightweight hardware stack (137g airframe with a 9g camera) and offboard computation with a software stack featuring a photorealistic 3D Gaussian Splat (GSplat) simulator for developing and evaluating vision-based controllers. We validate FalconWing on two challenging vision-based aerial case studies. In the leader-follower case study, our best vision-based controller, trained via imitation learning on GSplat-rendered data augmented with domain randomization, achieves 100% tracking success across 3 types of leader maneuvers over 30 trials and shows robustness to leader's appearance shifts in simulation. In the autonomous landing case study, our vision-based controller trained purely in simulation transfers zero-shot to real hardware, achieving an 80% success rate over ten landing trials. We will release hardware designs, GSplat scenes, and dynamics models upon publication to make FalconWing an open-source flight kit for engineering students and research labs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24421v2">Proxy-GS: Efficient 3D Gaussian Splatting via Proxy Mesh</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as an efficient approach for achieving photorealistic rendering. Recent MLP-based variants further improve visual fidelity but introduce substantial decoding overhead during rendering. To alleviate computation cost, several pruning strategies and level-of-detail (LOD) techniques have been introduced, aiming to effectively reduce the number of Gaussian primitives in large-scale scenes. However, our analysis reveals that significant redundancy still remains due to the lack of occlusion awareness. In this work, we propose Proxy-GS, a novel pipeline that exploits a proxy to introduce Gaussian occlusion awareness from any view. At the core of our approach is a fast proxy system capable of producing precise occlusion depth maps at a resolution of 1000x1000 under 1ms. This proxy serves two roles: first, it guides the culling of anchors and Gaussians to accelerate rendering speed. Second, it guides the densification towards surfaces during training, avoiding inconsistencies in occluded regions, and improving the rendering quality. In heavily occluded scenarios, such as the MatrixCity Streets dataset, Proxy-GS not only equips MLP-based Gaussian splatting with stronger rendering capability but also achieves faster rendering speed. Specifically, it achieves more than 2.5x speedup over Octree-GS, and consistently delivers substantially higher rendering quality. Code will be public upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01119v1">Instant4D: 4D Gaussian Splatting in Minutes</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Accepted by NeurIPS 25
    </div>
    <details class="paper-abstract">
      Dynamic view synthesis has seen significant advances, yet reconstructing scenes from uncalibrated, casual video remains challenging due to slow optimization and complex parameter estimation. In this work, we present Instant4D, a monocular reconstruction system that leverages native 4D representation to efficiently process casual video sequences within minutes, without calibrated cameras or depth sensors. Our method begins with geometric recovery through deep visual SLAM, followed by grid pruning to optimize scene representation. Our design significantly reduces redundancy while maintaining geometric integrity, cutting model size to under 10% of its original footprint. To handle temporal dynamics efficiently, we introduce a streamlined 4D Gaussian representation, achieving a 30x speed-up and reducing training time to within two minutes, while maintaining competitive performance across several benchmarks. Our method reconstruct a single video within 10 minutes on the Dycheck dataset or for a typical 200-frame video. We further apply our model to in-the-wild videos, showcasing its generalizability. Our project website is published at https://instant4d.github.io/.
    </details>
</div>
