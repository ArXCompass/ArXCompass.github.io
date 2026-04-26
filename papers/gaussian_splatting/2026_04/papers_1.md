# gaussian splatting - 2026_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21631v1">DualSplat: Robust 3D Gaussian Splatting via Pseudo-Mask Bootstrapping from Reconstruction Failures</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 10 pages,6 figures, accepted to Computer Vision and Pattern Recognition Conference 2026
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) achieves real-time photorealistic rendering, its performance degrades significantly when training images contain transient objects that violate multi-view consistency. Existing methods face a circular dependency: accurate transient detection requires a well-reconstructed static scene, while clean reconstruction itself depends on reliable transient masks. We address this challenge with DualSplat, a Failure-to-Prior framework that converts first-pass reconstruction failures into explicit priors for a second reconstruction stage. We observe that transients, which appear in only a subset of views, often manifest as incomplete fragments during conservative initial training. We exploit these failures to construct object-level pseudo-masks by combining photometric residuals, feature mismatches, and SAM2 instance boundaries. These pseudo-masks then guide a clean second-pass 3DGS optimization, while a lightweight MLP refines them online by gradually shifting from prior supervision to self-consistency. Experiments on RobustNeRF and NeRF On-the-go show that DualSplat outperforms existing baselines, demonstrating particularly clear advantages in transient-heavy scenes and transient regions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21400v1">You Only Gaussian Once: Controllable 3D Gaussian Splatting for Ultra-Densely Sampled Scenes</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 17 pages, 5 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has revolutionized neural rendering, yet existing methods remain predominantly research prototypes ill-suited for production-level deployment. We identify a critical "Industry-Academia Gap" hindering real-world application: unpredictable resource consumption from heuristic Gaussian growth, the "sparsity shield" of current benchmarks that rewards hallucination over physical fidelity, and severe multi-sensor data pollution. To bridge this gap, we propose YOGO (You Only Gaussian Once), a system-level framework that reformulates the stochastic growth process into a deterministic, budget-aware equilibrium. YOGO integrates a novel budget controller for hardware-constrained resource allocation and an availability-registration protocol for robust multi-sensor fusion. To push the boundaries of reconstruction fidelity, we introduce Immersion v1.0, the first ultra-dense indoor dataset specifically designed to break the "sparsity shield." By providing saturated viewpoint coverage, Immersion v1.0 forces algorithms to focus on extreme physical fidelity rather than viewpoint interpolation, and enables the community to focus on the upper limits of high-fidelity reconstruction. Extensive experiments demonstrate that YOGO achieves state-of-the-art visual quality while maintaining a strictly deterministic profile, establishing a new standard for production-grade 3DGS. To facilitate reproducibility, part scenes of Immersion v1.0 dataset and source code of YOGO has been publicly released. The project link is https://jjrcn.github.io/YOGO/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.17969v2">E3VS-Bench: A Benchmark for Viewpoint-Dependent Active Perception in 3D Gaussian Splatting Scenes</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 Project page: https://k0uya.github.io/e3vs-proj/
    </div>
    <details class="paper-abstract">
      Visual search in 3D environments requires embodied agents to actively explore their surroundings and acquire task-relevant evidence. However, existing visual search and embodied AI benchmarks, including EQA, typically rely on static observations or constrained egocentric motion, and thus do not explicitly evaluate fine-grained viewpoint-dependent phenomena that arise under unrestricted 5-DoF viewpoint control in real-world 3D environments, such as visibility changes caused by vertical viewpoint shifts, revealing contents inside containers, and disambiguating object attributes that are only observable from specific angles. To address this limitation, we introduce {E3VS-Bench}, a benchmark for embodied 3D visual search where agents must control their viewpoints in 5-DoF to gather viewpoint-dependent evidence for question answering. E3VS-Bench consists of 99 high-fidelity 3D scenes reconstructed using 3D Gaussian Splatting and 2,014 question-driven episodes. 3D Gaussian Splatting enables photorealistic free-viewpoint rendering that preserves fine-grained visual details (e.g., small text and subtle attributes) often degraded in mesh-based simulators, thereby allowing the construction of questions that cannot be answered from a single view and instead require active inspection across viewpoints in 5-DoF. We evaluate multiple state-of-the-art VLMs and compare their performance with humans. Despite strong 2D reasoning ability, all models exhibit a substantial gap from humans, highlighting limitations in active perception and coherent viewpoint planning specifically under full 5-DoF viewpoint changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13589v3">Dehaze-then-Splat: Generative Dehazing with Physics-Informed 3D Gaussian Splatting for Smoke-Free Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      We present Dehaze-then-Splat, a two-stage pipeline for multi-view smoke removal and novel view synthesis developed for Track~2 of the NTIRE 2026 3D Restoration and Reconstruction Challenge. In the first stage, we produce pseudo-clean training images via per-frame generative dehazing using Nano Banana Pro, followed by brightness normalization. In the second stage, we train 3D Gaussian Splatting (3DGS) with physics-informed auxiliary losses -- depth supervision via Pearson correlation with pseudo-depth, dark channel prior regularization, and dual-source gradient matching -- that compensate for cross-view inconsistencies inherent in frame-wise generative processing. We identify a fundamental tension in dehaze-then-reconstruct pipelines: per-image restoration quality does not guarantee multi-view consistency, and such inconsistency manifests as blurred renders and structural instability in downstream 3D reconstruction.Our analysis shows that MCMC-based densification with early stopping, combined with depth and haze-suppression priors, effectively mitigates these artifacts. On the Akikaze validation scene, our pipeline achieves 20.98\,dB PSNR and 0.683 SSIM for novel view synthesis, a +1.50\,dB improvement over the unregularized baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21182v1">WildSplatter: Feed-forward 3D Gaussian Splatting with Appearance Control from Unconstrained Images</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 Project page: https://github.com/yfujimura/WildSplatter
    </div>
    <details class="paper-abstract">
      We propose WildSplatter, a feed-forward 3D Gaussian Splatting (3DGS) model for unconstrained images with unknown camera parameters and varying lighting conditions. 3DGS is an effective scene representation that enables high-quality, real-time rendering; however, it typically requires iterative optimization and multi-view images captured under consistent lighting with known camera parameters. WildSplatter is trained on unconstrained photo collections and jointly learns 3D Gaussians and appearance embeddings conditioned on input images. This design enables flexible modulation of Gaussian colors to represent significant variations in lighting and appearance. Our method reconstructs 3D Gaussians from sparse input views in under one second, while also enabling appearance control under diverse lighting conditions. Experimental results demonstrate that our approach outperforms existing pose-free 3DGS methods on challenging real-world datasets with varying illumination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22793v2">GSpaRC: Gaussian Splatting for Real-time Reconstruction of RF Channels</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Channel state information (CSI) is essential for adaptive beamforming and maintaining robust links in wireless communication systems. However, acquiring CSI incurs significant overhead, consuming up to 25\% of spectrum resources in 5G networks due to frequent pilot transmissions at sub-millisecond intervals. Recent approaches aim to reduce this burden by reconstructing CSI from spatiotemporal RF measurements, such as signal strength and direction-of-arrival. While effective in offline settings, these methods often suffer from inference latencies in the 5--100~ms range, making them impractical for real-time systems. We present GSpaRC: Gaussian Splatting for Real-time Reconstruction of RF Channels, the first algorithm to break the 1 ms latency barrier while maintaining high accuracy. GSpaRC represents the RF environment using a compact set of 3D Gaussian primitives, each parameterized by a lightweight neural model augmented with physics-informed features such as distance-based attenuation. Unlike traditional vision-based splatting pipelines, GSpaRC is tailored for RF reception: it employs an equirectangular projection onto a hemispherical surface centered at the receiver to reflect omnidirectional antenna behavior. A custom CUDA pipeline enables fully parallelized directional sorting, splatting, and rendering across frequency and spatial dimensions. Evaluated on multiple RF datasets, GSpaRC achieves similar CSI reconstruction fidelity to recent state-of-the-art methods while reducing training and inference time by over an order of magnitude. By trading modest GPU computation for a substantial reduction in pilot overhead, GSpaRC enables scalable, low-latency channel estimation suitable for deployment in 5G and future wireless systems. The code is available here: \href{https://github.com/Nbhavyasai/GSpaRC-WirelessGaussianSplatting.git}{GSpaRC}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11098v2">Efficient Transceiver Design for Aerial Image Transmission and Large-scale Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 6 pages, 6 figures, Accepted in ISIT 2026 IEEE International Symposium on Information Theory-w
    </div>
    <details class="paper-abstract">
      Large-scale three-dimensional (3D) scene reconstruction in low-altitude intelligent networks (LAIN) demands highly efficient wireless image transmission. However, existing schemes struggle to balance severe pilot overhead with the transmission accuracy required to maintain reconstruction fidelity. To strike a balance between efficiency and reliability, this paper proposes a novel deep learning-based end-to-end (E2E) transceiver design that integrates 3D Gaussian Splatting (3DGS) directly into the training process. By jointly optimizing the communication modules via the combined 3DGS rendering loss, our approach explicitly improves scene recovery quality. Furthermore, this task-driven framework enables the use of a sparse pilot scheme, significantly reducing transmission overhead while maintaining robust image recovery under low-altitude channel conditions. Extensive experiments on real-world aerial image datasets demonstrate that the proposed E2E design significantly outperforms existing baselines, delivering superior transmission performance and accurate 3D scene reconstructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20714v2">The Role and Relationship of Initialization and Densification in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Sources are available at https://github.com/deivse/ivd_splat . Changes in this version: fixed wrong graphs being used in Fig. 6 (b), Fig. 10 (a,c,d) due to compilation issue; results with EDGS* are now using splat scale increase when reducing init. size (previously reported results without scale increase, but conclusions remain unchanged)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become the method of choice for photo-realistic 3D reconstruction of scenes, due to being able to efficiently and accurately recover the scene appearance and geometry from images. 3DGS represents the scene through a set of 3D Gaussians, parameterized by their position, spatial extent, and view-dependent color. Starting from an initial point cloud, 3DGS refines the Gaussians' parameters as to reconstruct a set of training images as accurately as possible. Typically, a sparse Structure-from-Motion point cloud is used as initialization. In order to obtain dense Gaussian clouds, 3DGS methods thus rely on a densification stage. In this paper, we systematically study the relation between densification and initialization. Proposing a new benchmark, we study combinations of different types of initializations (dense laser scans, dense (multi-view) stereo point clouds, dense monocular depth estimates, sparse SfM point clouds) and different densification schemes. We show that current densification approaches are not able to take full advantage of dense initialization as they are often unable to (significantly) improve over sparse SfM-based initialization. We will make our benchmark publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.24725v2">Confidence-Based Mesh Extraction from 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Project Page: https://r4dl.github.io/CoMe/
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) greatly accelerated mesh extraction from posed images due to its explicit representation and fast software rasterization. While the addition of geometric losses and other priors has improved the accuracy of extracted surfaces, mesh extraction remains difficult in scenes with abundant view-dependent effects. To resolve the resulting ambiguities, prior works rely on multi-view techniques, iterative mesh extraction, or large pre-trained models, sacrificing the inherent efficiency of 3DGS. In this work, we present a simple and efficient alternative by introducing a self-supervised confidence framework to 3DGS: within this framework, learnable confidence values dynamically balance photometric and geometric supervision. Extending our confidence-driven formulation, we introduce losses which penalize per-primitive color and normal variance and demonstrate their benefits to surface extraction. Finally, we complement the above with an improved appearance model, by decoupling the individual terms of the D-SSIM loss. Our final approach delivers state-of-the-art results for unbounded meshes while remaining highly efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05687v2">3D Smoke Scene Reconstruction Guided by Vision Priors from Multimodal Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Reconstructing 3D scenes from smoke-degraded multi-view images is particularly difficult because smoke introduces strong scattering effects, view-dependent appearance changes, and severe degradation of cross-view consistency. To address these issues, we propose a framework that integrates visual priors with efficient 3D scene modeling. We employ Nano-Banana-Pro to enhance smoke-degraded images and provide clearer visual observations for reconstruction and develop Smoke-GS, a medium-aware 3D Gaussian Splatting framework for smoke scene reconstruction and restoration-oriented novel view synthesis. Smoke-GS models the scene using explicit 3D Gaussians and introduces a lightweight view-dependent medium branch to capture direction-dependent appearance variations caused by smoke. Our method preserves the rendering efficiency of 3D Gaussian Splatting while improving robustness to smoke-induced degradation. Results demonstrate the effectiveness of our method for generating consistent and visually clear novel views in challenging smoke environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20155v1">GSCompleter: A Distillation-Free Plugin for Metric-Aware 3D Gaussian Splatting Completion in Seconds</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has revolutionized real-time rendering, its performance degrades significantly under sparse-view extrapolation, manifesting as severe geometric voids and artifacts. Existing solutions primarily rely on an iterative "Repair-then-Distill" paradigm, which is inherently unstable and prone to overfitting. In this work, we propose GSCompleter, a distillation-free plugin that shifts scene completion to a stable "Generate-then-Register" workflow. Our approach first synthesizes plausible 2D reference images and explicitly lifts them into metric-scale 3D primitives via a robust Stereo-Anchor mechanism. These primitives are then seamlessly integrated into the global context through a novel Ray-Constrained Registration strategy. This shift to a rapid registration paradigm delivers superior 3DGS completion performance across three distinct benchmarks, enhancing the quality and efficiency of various baselines and achieving new SOTA results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20046v1">Gaussians on a Diet: High-Quality Memory-Bounded 3D Gaussian Splatting Training</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has revolutionized novel view synthesis with high-quality rendering through continuous aggregations of millions of 3D Gaussian primitives. However, it suffers from a substantial memory footprint, particularly during training due to uncontrolled densification, posing a critical bottleneck for deployment on memory-constrained edge devices. While existing methods prune redundant Gaussians post-training, they fail to address the peak memory spikes caused by the abrupt growth of Gaussians early in the training process. To solve the training memory consumption problem, we propose a systematic memory-bounded training framework that dynamically optimizes Gaussians through iterative growth and pruning. In other words, the proposed framework alternates between incremental pruning of low-impact Gaussians and strategic growing of new primitives with an adaptive Gaussian compensation, maintaining a near-constant low memory usage while progressively refining rendering fidelity. We comprehensively evaluate the proposed training framework on various real-world datasets under strict memory constraints, showing significant improvements over existing state-of-the-art methods. Particularly, our proposed method practically enables memory-efficient 3DGS training on NVIDIA Jetson AGX Xavier, achieving similar visual quality with up to 80% lower peak training memory consumption than the original 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20038v1">FluSplat: Sparse-View 3D Editing without Test-Time Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Recent advances in text-guided image editing and 3D Gaussian Splatting (3DGS) have enabled high-quality 3D scene manipulation. However, existing pipelines rely on iterative edit-and-fit optimization at test time, alternating between 2D diffusion editing and 3D reconstruction. This process is computationally expensive, scene-specific, and prone to cross-view inconsistencies. We propose a feed-forward framework for cross-view consistent 3D scene editing from sparse views. Instead of enforcing consistency through iterative 3D refinement, we introduce a cross-view regularization scheme in the image domain during training. By jointly supervising multi-view edits with geometric alignment constraints, our model produces view-consistent results without per-scene optimization at inference. The edited views are then lifted into 3D via a feedforward 3DGS model, yielding a coherent 3DGS representation in a single forward pass. Experiments demonstrate competitive editing fidelity and substantially improved cross-view consistency compared to optimization-based methods, while reducing inference time by orders of magnitude.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.00800v3">Semantic-guided Gaussian Splatting for High-Fidelity Underwater Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Accurate 3D reconstruction in degraded imaging conditions remains a key challenge in photogrammetry and neural rendering. In underwater environments, spatially varying visibility caused by scattering, attenuation, and sparse observations leads to highly non-uniform information quality. Existing 3D Gaussian Splatting (3DGS) methods typically optimize primitives based on photometric signals alone, resulting in imbalanced representation, with overfitting in well-observed regions and insufficient reconstruction in degraded areas. In this paper, we propose SWAGSplatting (Semantic-guided Water-scene Augmented Gaussian Splatting), a multimodal framework that integrates semantic priors into 3DGS for robust, high-fidelity underwater reconstruction. Each Gaussian primitive is augmented with a learnable semantic feature, supervised by CLIP-based embeddings derived from region-level cues. A semantic consistency loss is introduced to align geometric reconstruction with high-level semantics, improving structural coherence and preserving salient object boundaries under challenging conditions. Furthermore, we propose an adaptive Gaussian primitive reallocation strategy that redistributes representation capacity based on both primitive importance and reconstruction error, mitigating the imbalance introduced by conventional densification. This enables more effective modeling of low-visibility regions without increasing computational cost. Extensive experiments on real-world datasets, including SeaThru-NeRF, Submerged3D, and S-UW, demonstrate that the proposed method consistently outperforms state-of-the-art approaches in terms of average PSNR, SSIM, and LPIPS. The results validate the effectiveness of integrating semantic priors for high-fidelity underwater scene reconstruction. Code is available at https://github.com/theflash987/SWAGSplatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.22650v2">MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Accepted at CVPR 2026 (Oral). Project webpage: https://shiyao-li.github.io/magician/
    </div>
    <details class="paper-abstract">
      Active mapping aims to determine how an agent should move to efficiently reconstruct unknown environments. Most existing approaches rely on greedy next-best-view prediction, resulting in inefficient exploration and incomplete reconstruction. To address this, we introduce MAGICIAN, a novel long-term planning framework that maximizes accumulated surface coverage gain through Imagined Gaussians, a scene representation based on 3D Gaussian Splatting, derived from a pre-trained occupancy network with strong structural priors. This representation enables efficient coverage gain computation for any novel viewpoint via fast volumetric rendering, allowing its integration into a tree-search algorithm for long-horizon planning. We update Imagined Gaussians and refine the trajectory in a closed loop. Our method achieves state-of-the-art performance across indoor and outdoor benchmarks with varying action spaces, highlighting the advantage of long-term planning in active mapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19571v1">TransSplat: Unbalanced Semantic Transport for Language-Driven 3DGS Editing</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Language-driven 3D Gaussian Splatting (3DGS) editing provides a more convenient approach for modifying complex scenes in VR/AR. Standard pipelines typically adopt a two-stage strategy: first editing multiple 2D views, and then optimizing the 3D representation to match these edited observations. Existing methods mainly improve view consistency through multi-view feature fusion, attention filtering, or iterative recalibration. However, they fail to explicitly address a more fundamental issue: the semantic correspondence between edited 2D evidence and 3D Gaussians. To tackle this problem, we propose TransSplat, which formulates language-driven 3DGS editing as a multi-view unbalanced semantic transport problem. Specifically, our method establishes correspondences between visible Gaussians and view-specific editing prototypes, thereby explicitly characterizing the semantic relationship between edited 2D evidence and 3D Gaussians. It further recovers a cross-view shared canonical 3D edit field to guide unified 3D appearance updates. In addition, we use transport residuals to suppress erroneous edits in non-target regions, mitigating edit leakage and improving local control precision. Qualitative and quantitative results show that, compared with existing 3D editing methods centered on enhancing view consistency, TransSplat achieves superior performance in local editing accuracy and structural consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12942v2">RMGS-SLAM: Real-time Multi-sensor Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 The manuscript has been improved, with refined content and updated and corrected experimental results
    </div>
    <details class="paper-abstract">
      Achieving real-time Simultaneous Localization and Mapping (SLAM) based on 3D Gaussian splatting (3DGS) in large-scale real-world environments remains challenging, as existing methods still struggle to jointly achieve low-latency pose estimation, continuous 3D Gaussian reconstruction, and long-term global consistency. In this paper, we present a tightly coupled LiDAR-Inertial-Visual 3DGS-based SLAM framework for real-time pose estimation and photorealistic mapping in large-scale real-world scenes. The system executes state estimation and 3D Gaussian primitive initialization in parallel with global Gaussian optimization, enabling continuous dense mapping. To improve Gaussian initialization quality and accelerate optimization convergence, we introduce a cascaded strategy that combines feed-forward predictions with geometric priors derived from voxel-based principal component analysis. To enhance global consistency, we perform loop closure directly on the optimized global Gaussian map by estimating loop constraints through Gaussian-based Generalized Iterative Closest Point registration, followed by pose-graph optimization. We also collect challenging large-scale looped outdoor sequences with hardware-synchronized LiDAR-camera-IMU and ground-truth trajectories for realistic evaluation. Extensive experiments on both public datasets and our dataset demonstrate that the proposed method achieves a state of the art among real-time efficiency, localization accuracy, and rendering quality across diverse real-world scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05301v2">SmokeGS-R: Physics-Guided Pseudo-Clean 3DGS for Real-World Multi-View Smoke Restoration</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Lab Report for NTIRE 2026 3DRR Track 2
    </div>
    <details class="paper-abstract">
      Real-world smoke simultaneously attenuates scene radiance, adds airlight, and destabilizes multi-view appearance consistency, making robust 3D reconstruction particularly difficult. We present \textbf{SmokeGS-R}, a practical pipeline developed for the NTIRE 2026 3D Restoration and Reconstruction Track 2 challenge. The key idea is to decouple geometry recovery from appearance correction: we generate physics-guided pseudo-clean supervision with a refined dark channel prior and guided filtering, train a sharp clean-only 3D Gaussian Splatting source model, and then harmonize its renderings with a donor ensemble using geometric-mean reference aggregation, LAB-space Reinhard transfer, and light Gaussian smoothing. On the official challenge testing leaderboard, the final submission achieved \mbox{PSNR $=15.217$} and \mbox{SSIM $=0.666$}. After the public release of RealX3D, we re-evaluated the same frozen result on the seven released challenge scenes without retraining and obtained \mbox{PSNR $=15.209$}, \mbox{SSIM $=0.644$}, and \mbox{LPIPS $=0.551$}, outperforming the strongest official baseline average on the same scenes by $+3.68$ dB PSNR. These results suggest that a geometry-first reconstruction strategy combined with stable post-render appearance harmonization is an effective recipe for real-world multi-view smoke restoration. The code is available at https://github.com/windrise/3drr_Track2_SmokeGS-R.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19216v1">An Object-Centered Data Acquisition Method for 3D Gaussian Splatting using Mobile Phones</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Data acquisition through mobile phones remains a challenge for 3D Gaussian Splatting (3DGS). In this work we target the object-centered scenario and enable reliable mobile acquisition by providing on-device capture guidance and recording onboard sensor signals for offline reconstruction. After the calibration step, the device orientations are aligned to a baseline frame to obtain relative poses, and the optical axis of the camera is mapped to an object-centered spherical grid for uniform viewpoint indexing. To curb polar sampling bias, we compute area-weighted spherical coverage in real-time and guide the user's motion accordingly. We compare the proposed method with RealityScan and the free-capture strategy. Our method achieves superior reconstruction quality using fewer input images compared to free capture and RealityScan. Further analysis shows that the proposed method is able to obtain more comprehensive and uniform viewpoint coverage during object-centered acquisition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19202v1">SketchFaceGS: Real-Time Sketch-Driven Face Editing and Generation with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Accepted to CVPR 2026 as a Highlight. Jittor implementation: https://github.com/gogoneural/SketchFaceGS_jittor. (C) 2026 IEEE. Personal use of this material is permitted
    </div>
    <details class="paper-abstract">
      3D Gaussian representations have emerged as a powerful paradigm for digital head modeling, achieving photorealistic quality with real-time rendering. However, intuitive and interactive creation or editing of 3D Gaussian head models remains challenging. Although 2D sketches provide an ideal interaction modality for fast, intuitive conceptual design, they are sparse, depth-ambiguous, and lack high-frequency appearance cues, making it difficult to infer dense, geometrically consistent 3D Gaussian structures from strokes - especially under real-time constraints. To address these challenges, we propose SketchFaceGS, the first sketch-driven framework for real-time generation and editing of photorealistic 3D Gaussian head models from 2D sketches. Our method uses a feed-forward, coarse-to-fine architecture. A Transformer-based UV feature-prediction module first reconstructs a coarse but geometrically consistent UV feature map from the input sketch, and then a 3D UV feature enhancement module refines it with high-frequency, photorealistic detail to produce a high-fidelity 3D head. For editing, we introduce a UV Mask Fusion technique combined with a layer-by-layer feature-fusion strategy, enabling precise, real-time, free-viewpoint modifications. Extensive experiments show that SketchFaceGS outperforms existing methods in both generation fidelity and editing flexibility, producing high-quality, editable 3D heads from sketches in a single forward pass.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19133v1">BALTIC: A Benchmark and Cross-Domain Strategy for 3D Reconstruction Across Air and Underwater Domains Under Varying Illumination</a></div>
    <div class="paper-meta">
      📅 2026-04-21
    </div>
    <details class="paper-abstract">
      Robust 3D reconstruction across varying environmental conditions remains a critical challenge for robotic perception, particularly when transitioning between air and water. To address this, we introduce BALTIC, a controlled benchmark designed to systematically evaluate modern 3D reconstruction methods under variations in medium and lighting. The benchmark comprises 13 datasets spanning two media (air and water) and three lighting conditions (ambient, artificial, and mixed), with additional variations in motion type, scanning pattern, and initialization trajectory, resulting in a diverse set of sequences. Our experimental setup features a custom water tank equipped with a monocular camera and an HTC Vive tracker, enabling accurate ground-truth pose estimation. We further investigate cross-domain reconstruction by augmenting underwater image sequences with a small number of in-air views captured under similar lighting conditions. We evaluate Structure-from-Motion reconstruction using COLMAP in terms of both trajectory accuracy and scene geometry, and use these reconstructions as input to Neural Radiance Fields and 3D Gaussian Splatting methods. The resulting models are assessed against ground-truth trajectories and in-air references, while rendered outputs are compared using perceptual and photometric metrics. Additionally, we perform a color restoration analysis to evaluate radiometric consistency across domains. Our results show that under controlled, texture-consistent conditions, Gaussian Splatting with simple preprocessing (e.g., white balance correction) can achieve performance comparable to specialized underwater methods, although its robustness decreases in more complex and heterogeneous real-world environments
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19127v1">OT-UVGS: Revisiting UV Mapping for Gaussian Splatting as a Capacity Allocation Problem</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 Accepted to Eurographics 2026 Short Papers
    </div>
    <details class="paper-abstract">
      UV-parameterized Gaussian Splatting (UVGS) maps an unstructured set of 3D Gaussians to a regular UV tensor, enabling compact storage and explicit control of representation capacity. Existing UVGS, however, uses a deterministic spherical pro- jection to assign Gaussians to UV locations. Because this mapping ignores the global Gaussian distribution, it often leaves many UV slots empty while causing frequent collisions in dense regions. We reinterpret UV mapping as a capacity-allocation problem under a fixed UV budget and propose OT-UVGS, a lightweight, separable one-dimensional optimal-transport-inspired mapping that globally couples assignments while preserving the original UVGS representation. The method is implemented with rank-based sorting, has O(N log N) complexity for N Gaussians, and can be used as a drop-in replacement for spherical UVGS. Across 184 object-centric scenes and the Mip-NeRF dataset, OT-UVGS consistently improves peak signal-to-noise ratio (PSNR), structural similarity (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS) under the same UV resolution and per-slot capacity (K=1). These gains are accompanied by substantially better UV utilization, including higher non-empty slot ratios, fewer collisions, and higher Gaussian retention. Our results show that revisiting the mapping alone can unlock a significant fraction of the latent capacity of UVGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.18980v1">AdaGScale: Viewpoint-Adaptive Gaussian Scaling in 3D Gaussian Splatting to Reduce Gaussian-Tile Pairs</a></div>
    <div class="paper-meta">
      📅 2026-04-21
      | 💬 DAC 2026
    </div>
    <details class="paper-abstract">
      Reducing the number of Gaussian-tile pairs is one of the most promising approaches to improve 3D Gaussian Splatting (3D-GS) rendering speed on GPUs. However, the importance difference existing among Gaussian-tile pairs has never been considered in the previous works. In this paper, we propose AdaGScale, a novel viewpoint-adaptive Gaussian scaling technique for reducing the number of Gaussian-tile pairs. AdaGScale is based on the observation that the peripheral tiles located far from Gaussian center contribute negligibly to pixel color accumulation. This suggests an opportunity for reducing the number of Gaussian-tile pairs based on color contribution. AdaGScale efficiently estimates the color contribution in the peripheral region of each Gaussian during a preprocessing stage and adaptively scales its size based on the peripheral score. As a result, Gaussians with lower importance intersect with fewer tiles during the intersection test, which improves rendering speed while maintaining image quality. The adjusted size is used only for tile intersection test, and the original size is retained during color accumulation to preserve visual fidelity. Experimental results show that AdaGScale achieves a geometric mean speedup of 13.8x over original 3D-GS on a GPU, with only about 0.5 dB degradation in PSNR on city-scale scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21714v1">High-Fidelity 3D Gaussian Human Reconstruction via Region-Aware Initialization and Geometric Priors</a></div>
    <div class="paper-meta">
      📅 2026-04-20
    </div>
    <details class="paper-abstract">
      Real-time, high-fidelity 3D human reconstruction from RGB images is essential for interactive applications such as virtual reality and gaming, yet remains challenging due to the complex non-rigid deformations of dynamic human bodies. Although 3D Gaussian Splatting enables efficient rendering, existing methods struggle to capture fine geometric details and often produce artifacts such as fused fingers and over-smoothed faces. Moreover, conventional spatial-field-based dynamic modeling faces a trade-off between reconstruction fidelity and GPU memory consumption. To address these issues, we propose a novel 3D Gaussian human reconstruction framework that combines region-aware initialization with rich geometric priors. Specifically, we leverage the expressive SMPL-X model to initialize both 3D Gaussians and skinning weights, providing a robust geometric foundation for precise reconstruction. We further introduce a region-aware density initialization strategy and a geometry-aware multi-scale hash encoding module to improve local detail recovery while maintaining computational efficiency.Experiments on PeopleSnapshot and GalaBasketball show that our method achieves superior reconstruction quality and finer detail preservation under complex motions, while maintaining real-time rendering speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16988v2">PhysMorph-GS: Render-Guided Volumetric Morphing with Differentiable Physics</a></div>
    <div class="paper-meta">
      📅 2026-04-20
      | 💬 12pages, 11figures
    </div>
    <details class="paper-abstract">
      Differentiable particle-based simulation can produce physically plausible motion, but target-driven volumetric shape morphing remains underconstrained: physics-only mass matching captures coarse global structure yet struggles with fine geometric detail, while naive image-space coupling destabilizes elastic dynamics. We present PhysMorph-GS, a render-guided morphing framework that couples material point method simulation with differentiable 3D Gaussian splatting. The key idea is to inject visual supervision through the deformation gradient $\mathbf{F}$ rather than particle positions, so render gradients act as control-space guidance while trajectories remain governed by physics. We further introduce phased Chamfer-guided plasticity that delays rest-state migration until coarse structure has formed; in practice, rendering is evaluated on a surface-focused particle subset for efficiency and gradient concentration. Relative to a physics-only baseline, our method reduces silhouette error by 25.8\%, 10.8\%, and 49.9\% on representative examples, with the largest gains on models with thin features. These results suggest that the main challenge in render-guided differentiable morphing is not simply adding stronger image losses, but injecting visual guidance in a way that remains compatible with elastic simulation. We further observe that plasticity-driven rest-state migration drives different sources toward a shared target-determined attractor, distinguishing physics-based morphing from interpolation between registered shape pairs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.04787v2">AvatarPointillist: AutoRegressive 4D Gaussian Avatarization</a></div>
    <div class="paper-meta">
      📅 2026-04-20
      | 💬 Accepted by the CVPR 2026 main conference. Project page: https://kumapowerliu.github.io/AvatarPointillist/
    </div>
    <details class="paper-abstract">
      We introduce AvatarPointillist, a novel framework for generating dynamic 4D Gaussian avatars from a single portrait image. At the core of our method is a decoder-only Transformer that autoregressively generates a point cloud for 3D Gaussian Splatting. This sequential approach allows for precise, adaptive construction, dynamically adjusting point density and the total number of points based on the subject's complexity. During point generation, the AR model also jointly predicts per-point binding information, enabling realistic animation. After generation, a dedicated Gaussian decoder converts the points into complete, renderable Gaussian attributes. We demonstrate that conditioning the decoder on the latent features from the AR generator enables effective interaction between stages and markedly improves fidelity. Extensive experiments validate that AvatarPointillist produces high-quality, photorealistic, and controllable avatars. We believe this autoregressive formulation represents a new paradigm for avatar generation, and we will release our code inspire future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19202v2">NVGS: Neural Visibility for Occlusion Culling in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-20
      | 💬 17 pages, 15 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting can exploit frustum culling and level-of-detail strategies to accelerate rendering of scenes containing a large number of primitives. However, the semi-transparent nature of Gaussians prevents the application of another highly effective technique: occlusion culling. We address this limitation by proposing a novel method to learn the viewpoint-dependent visibility function of all Gaussians in a trained model using a small, shared MLP across instances of an asset in a scene. By querying it for Gaussians within the viewing frustum prior to rasterization, our method can discard occluded primitives during rendering. Leveraging Tensor Cores for efficient computation, we integrate these neural queries directly into a novel instanced software rasterizer. Our approach outperforms the current state of the art for composed scenes in terms of VRAM usage and image quality, utilizing a combination of our instanced rasterizer and occlusion culling MLP, and exhibits complementary properties to existing LoD techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.18205v1">A Comparative Evaluation of Geometric Accuracy in NeRF and Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-20
    </div>
    <details class="paper-abstract">
      Recent advances in neural rendering have introduced numerous 3D scene representations. Although standard computer vision metrics evaluate the visual quality of generated images, they often overlook the fidelity of surface geometry. This limitation is particularly critical in robotics, where accurate geometry is essential for tasks such as grasping and object manipulation. In this paper, we present an evaluation pipeline for neural rendering methods that focuses on geometric accuracy, along with a benchmark comprising 19 diverse scenes. Our approach enables a systematic assessment of reconstruction methods in terms of surface and shape fidelity, complementing traditional visual metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02034v2">SemMorph3D: Unsupervised Semantic-Aware 3D Morphing via Mesh-Guided Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-04-20
      | 💬 Project page: https://baiyunshu.github.io/GAUSSIANMORPHING.github.io/
    </div>
    <details class="paper-abstract">
      We introduce METHODNAME, a novel framework for semantic-aware 3D shape and texture morphing directly from multi-view images. While 3D Gaussian Splatting (3DGS) enables photorealistic rendering, its unstructured nature often leads to catastrophic geometric fragmentation during morphing. Conversely, traditional mesh-based morphing enforces structural integrity but mandates pristine input topology and struggles with complex appearances. Our method resolves this dichotomy by employing a mesh-guided strategy where a coarse, extracted base mesh acts as a flexible geometric anchor. This anchor provides the necessary topological scaffolding to guide unstructured Gaussians, successfully compensating for mesh extraction artifacts and topological limitations. Furthermore, we propose a novel dual-domain optimization strategy that leverages this hybrid representation to establish unsupervised semantic correspondence, synergizing geodesic regularizations for shape preservation with texture-aware constraints for coherent color evolution. This integrated approach ensures stable, physically plausible transformations without requiring labeled data, specialized 3D assets, or category-specific templates. On the proposed TexMorph benchmark, METHODNAME substantially outperforms prior 2D and 3D methods, yielding fully textured, topologically robust 3D morphing while reducing color consistency error (Delta E) by 22.2% and EI by 26.2%. Project page: https://baiyunshu.github.io/GAUSSIANMORPHING.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05152v2">Splatography: Sparse multi-view dynamic Gaussian Splatting for filmmaking challenges</a></div>
    <div class="paper-meta">
      📅 2026-04-20
      | 💬 Accepted to IEEE International Conference on 3DV (2026)
    </div>
    <details class="paper-abstract">
      Deformable Gaussian Splatting (GS) accomplishes photorealistic dynamic 3-D reconstruction from dense multi-view video (MVV) by learning to deform a canonical GS representation. However, in filmmaking, tight budgets can result in sparse camera configurations, which limits state-of-the-art (SotA) methods when capturing complex dynamic features. To address this issue, we introduce an approach that splits the canonical Gaussians and deformation field into foreground and background components using a sparse set of masks for frames at t=0. Each representation is separately trained on different loss functions during canonical pre-training. Then, during dynamic training, different parameters are modeled for each deformation field following common filmmaking practices. The foreground stage contains diverse dynamic features so changes in color, position and rotation are learned. While, the background containing film-crew and equipment, is typically dimmer and less dynamic so only changes in point position are learned. Experiments on 3-D and 2.5-D entertainment datasets show that our method produces SotA qualitative and quantitative results; up to 3 PSNR higher with half the model size on 3-D scenes. Unlike the SotA and without the need for dense mask supervision, our method also produces segmented dynamic reconstructions including transparent and dynamic textures. Code and video comparisons are available online: https://azzarelli.github.io/splatographypage/index.html
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.18047v1">GS-STVSR: Ultra-Efficient Continuous Spatio-Temporal Video Super-Resolution via 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-20
    </div>
    <details class="paper-abstract">
      Continuous Spatio-Temporal Video Super-Resolution (C-STVSR) aims to simultaneously enhance the spatial resolution and frame rate of videos by arbitrary scale factors, offering greater flexibility than fixed-scale methods that are constrained by predefined upsampling ratios. In recent years, methods based on Implicit Neural Representations (INR) have made significant progress in C-STVSR by learning continuous mappings from spatio-temporal coordinates to pixel values. However, these methods fundamentally rely on dense pixel-wise grid queries, causing computational cost to scale linearly with the number of interpolated frames and severely limiting inference efficiency. We propose GS-STVSR, an ultra-efficient C-STVSR framework based on 2D Gaussian Splatting (2D-GS) that drives the spatiotemporal evolution of Gaussian kernels through continuous motion modeling, bypassing dense grid queries entirely. We exploit the strong temporal stability of covariance parameters for lightweight intermediate fitting, design an optical flow-guided motion module to derive Gaussian position and color at arbitrary time steps, introduce a Covariance resampling alignment module to prevent covariance drift, and propose an adaptive offset window for large-scale motion. Extensive experiments on Vid4, GoPro, and Adobe240 show that GS-STVSR achieves state-of-the-art quality across all benchmarks. Moreover, its inference time remains nearly constant at conventional temporal scales (X2--X8) and delivers over X3 speedup at extreme scales X32, demonstrating strong practical applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.17727v1">Voronoi-guided Bilateral 2D Gaussian Splatting for Arbitrary-Scale Hyperspectral Image Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2026-04-20
    </div>
    <details class="paper-abstract">
      Most existing hyperspectral image super-resolution methods require modifications for different scales, limiting their flexibility in arbitrary-scale reconstruction. 2D Gaussian splatting provides a continuous representation that is compatible with arbitrary-scale super-resolution. Existing methods often rely on rasterization strategies, which may limit flexible spatial modeling. Extending them to hyperspectral image super-resolution remains challenging, as the task requires adaptive spatial reconstruction while preserving spectral fidelity. This paper proposes GaussianHSI, a Gaussian-Splatting-based framework for arbitrary-scale hyperspectral image super-resolution. We develop a Voronoi-Guided Bilateral 2D Gaussian Splatting for spatial reconstruction. After predicting a set of Gaussian functions to represent the input, it associates each target pixel with relevant Gaussian functions through Voronoi-guided selection. The target pixel is then reconstructed by aggregating the selected Gaussian functions with reference-aware bilateral weighting, which considers both geometric relevance and consistency with low-resolution features. We further introduce a Spectral Detail Enhancement module to improve spectral reconstruction. Extensive experiments on benchmark datasets demonstrate the effectiveness of GaussianHSI over state-of-the-art methods for arbitrary-scale hyperspectral image super-resolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.07826v2">R3D2: Realistic 3D Asset Insertion via Diffusion for Autonomous Driving Simulation</a></div>
    <div class="paper-meta">
      📅 2026-04-19
    </div>
    <details class="paper-abstract">
      Validating autonomous driving (AD) systems requires diverse and safety-critical testing, making photorealistic virtual environments essential. Traditional simulation platforms, while controllable, are resource-intensive to scale and often suffer from a domain gap with real-world data. In contrast, neural reconstruction methods like 3D Gaussian Splatting (3DGS) offer a scalable solution for creating photorealistic digital twins of real-world driving scenes. However, they struggle with dynamic object manipulation and reusability as their per-scene optimization-based methodology tends to result in incomplete object models with integrated illumination effects. This paper introduces R3D2, a lightweight, one-step diffusion model designed to overcome these limitations and enable realistic insertion of complete 3D assets into existing scenes by generating plausible rendering effects-such as shadows and consistent lighting-in real time. This is achieved by training R3D2 on a novel dataset: 3DGS object assets are generated from in-the-wild AD data using an image-conditioned 3D generative model, and then synthetically placed into neural rendering-based virtual environments, allowing R3D2 to learn realistic integration. Quantitative and qualitative evaluations demonstrate that R3D2 significantly enhances the realism of inserted assets, enabling use-cases like text-to-3D asset insertion and cross-scene/dataset object transfer, allowing for true scalability in AD validation. To promote further research in scalable and realistic AD simulation, we release our code, see https://research.zenseact.com/publications/R3D2/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.17155v1">Instant Colorization of Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2026-04-18
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has recently become one of the most popular frameworks for photorealistic 3D scene reconstruction and rendering. While current rasterizers allow for efficient mappings of 3D Gaussian splats onto 2D camera views, this work focuses on mapping 2D image information (e.g. color, neural features or segmentation masks) efficiently back onto an existing scene of Gaussian splats. This 'opposite' direction enables applications ranging from scene relighting and stylization to 3D semantic segmentation, but also introduces challenges, such as view-dependent colorization and occlusion handling. Our approach tackles these challenges using the normal equation to solve a visibility-weighted least squares problem for every Gaussian and can be implemented efficiently with existing differentiable rasterizers. We demonstrate the effectiveness of our approach on scene relighting, feature enrichment and 3D semantic segmentation tasks, achieving up to an order of magnitude speedup compared to gradient descent-based baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.16910v1">LAGS: Low-Altitude Gaussian Splatting with Groupwise Heterogeneous Graph Learning</a></div>
    <div class="paper-meta">
      📅 2026-04-18
      | 💬 5 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Low-altitude Gaussian splatting (LAGS) facilitates 3D scene reconstruction by aggregating aerial images from distributed drones. However, as LAGS prioritizes maximizing reconstruction quality over communication throughput, existing low-altitude resource allocation schemes become inefficient. This inefficiency stems from their failure to account for image diversity introduced by varying viewpoints. To fill this gap, we propose a groupwise heterogeneous graph neural network (GW-HGNN) for LAGS resource allocation. GW-HGNN explicitly models the non-uniform contribution of different image groups to the reconstruction process, thus automatically balancing data fidelity and transmission cost. The key insight of GW-HGNN is to transform LAGS losses and communication constraints into graph learning costs for dual-level message passing. Experiments on real-world LAGS datasets demonstrate that GW-HGNN significantly outperforms state-of-the-art benchmarks across key rendering metrics, including PSNR, SSIM, and LPIPS. Furthermore, GW-HGNN reduces computational latency by approximately 100x compared to the widely-used MOSEK solver, achieving millisecond-level inference suitable for real-time deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.16747v1">Incoherent Deformation, Not Capacity: Diagnosing and Mitigating Overfitting in Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-17
      | 💬 10 pages, 6 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Dynamic 3D Gaussian Splatting methods achieve strong training-view PSNR on monocular video but generalize poorly: on the D-NeRF benchmark we measure an average train-test PSNR gap of 6.18 dB, rising to 11 dB on individual scenes. We report two findings that together account for most of that gap. Finding 1 (the role of splitting). A systematic ablation of the Adaptive Density Control pipeline (split, clone, prune, frequency, threshold, schedule) shows that splitting is responsible for over 80% of the gap: disabling split collapses the cloud from 44K to 3K Gaussians and the gap from 6.18 dB to 1.15 dB. Across all threshold-varying ablations, gap is log-linear in count (r = 0.995, bootstrap 95% CI [0.99, 1.00]), which suggests a capacity-based explanation. Finding 2 (the role of deformation coherence). We show that the capacity explanation is incomplete. A local-smoothness penalty on the per-Gaussian deformation field -- Elastic Energy Regularization (EER) -- reduces the gap by 40.8% while growing the cloud by 85%. Measuring per-Gaussian strain directly on trained checkpoints, EER reduces mean strain by 99.72% (median 99.80%) across all 8 scenes; on 8/8 scenes the median Gaussian under EER is less strained than the 1st-percentile (best-behaved) Gaussian under baseline. Alongside EER, we evaluate two further regularizers: GAD, a loss-rate-aware densification threshold, and PTDrop, a jitter-weighted Gaussian dropout. GAD+EER reduces the gap by 48%; adding PTDrop and a soft growth cap reaches 57%. We confirm that coherence generalizes to (a) a different deformation architecture (Deformable-3DGS, +40.6% gap reduction at re-tuned lambda), and (b) real monocular video (4 HyperNeRF scenes, reducing the mean PSNR gap by 14.9% at the same lambda as D-NeRF, with near-zero quality cost). The overfitting in dynamic 3DGS is driven by incoherent deformation, not parameter count.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15941v1">Neural Gabor Splatting: Enhanced Gaussian Splatting with Neural Gabor for High-frequency Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-17
      | 💬 Accepted to CVPR 2026
    </div>
    <details class="paper-abstract">
      Recent years have witnessed the rapid emergence of 3D Gaussian splatting (3DGS) as a powerful approach for 3D reconstruction and novel view synthesis. Its explicit representation with Gaussian primitives enables fast training, real-time rendering, and convenient post-processing such as editing and surface reconstruction. However, 3DGS suffers from a critical drawback: the number of primitives grows drastically for scenes with high-frequency appearance details, since each primitive can represent only a single color, requiring multiple primitives for every sharp color transition. To overcome this limitation, we propose neural Gabor splatting, which augments each Gaussian primitive with a lightweight multi-layer perceptron that models a wide range of color variations within a single primitive. To further control primitive numbers, we introduce a frequency-aware densification strategy that selects mismatch primitives for pruning and cloning based on frequency energy. Our method achieves accurate reconstruction of challenging high-frequency surfaces. We demonstrate its effectiveness through extensive experiments on both standard benchmarks, such as Mip-NeRF360 and High-Frequency datasets (e.g., checkered patterns), supported by comprehensive ablation studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15875v1">CLOTH-HUGS: Cloth Aware Human Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-17
    </div>
    <details class="paper-abstract">
      We present Cloth-HUGS, a Gaussian Splatting based neural rendering framework for photorealistic clothed human reconstruction that explicitly disentangles body and clothing. Unlike prior methods that absorb clothing into a single body representation and struggle with loose garments and complex deformations, Cloth-HUGS represents the performer using separate Gaussian layers for body and cloth within a shared canonical space. The canonical volume jointly encodes body, cloth, and scene primitives and is deformed through SMPL-driven articulation with learned linear blend skinning weights. To improve cloth realism, we initialize cloth Gaussians from mesh topology and apply physics-inspired constraints, including simulation-consistency, ARAP regularization, and mask supervision. We further introduce a depth-aware multi-pass rendering strategy for robust body-cloth-scene compositing, enabling real-time rendering at over 60 FPS. Experiments on multiple benchmarks show that Cloth-HUGS improves perceptual quality and geometric fidelity over state-of-the-art baselines, reducing LPIPS by up to 28% while producing temporally coherent cloth dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15862v1">Splats in Splats++: Robust and Generalizable 3D Gaussian Splatting Steganography</a></div>
    <div class="paper-meta">
      📅 2026-04-17
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently redefined the paradigm of 3D reconstruction, striking an unprecedented balance between visual fidelity and computational efficiency. As its adoption proliferates, safeguarding the copyright of explicit 3DGS assets has become paramount. However, existing invisible message embedding frameworks struggle to reconcile secure and high-capacity data embedding with intrinsic asset utility, often disrupting the native rendering pipeline or exhibiting vulnerability to structural perturbations. In this work, we present \textbf{\textit{Splats in Splats++}}, a unified and pipeline-agnostic steganography framework that seamlessly embeds high-capacity 3D/4D content directly within the native 3DGS representation. Grounded in a principled analysis of the frequency distribution of Spherical Harmonics (SH), we propose an importance-graded SH coefficient encryption scheme that achieves imperceptible embedding without compromising the original expressive power. To fundamentally resolve the geometric ambiguities that lead to message leakage, we introduce a \textbf{Hash-Grid Guided Opacity Mapping} mechanism. Coupled with a novel \textbf{Gradient-Gated Opacity Consistency Loss}, our formulation enforces a stringent spatial-attribute coupling between the original and hidden scenes, effectively projecting the discrete attribute mapping into a continuous, attack-resilient latent manifold. Extensive experiments demonstrate that our method substantially outperforms existing approaches, achieving up to \textbf{6.28 db} higher message fidelity, \textbf{3$\times$} faster rendering, and exceptional robustness against aggressive 3D-targeted structural attacks (e.g., GSPure). Furthermore, our framework exhibits remarkable versatility, generalizing seamlessly to 2D image embedding, 4D dynamic scene steganography, and diverse downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15284v2">GlobalSplat: Efficient Feed-Forward 3D Gaussian Splatting via Global Scene Tokens</a></div>
    <div class="paper-meta">
      📅 2026-04-17
    </div>
    <details class="paper-abstract">
      The efficient spatial allocation of primitives serves as the foundation of 3D Gaussian Splatting, as it directly dictates the synergy between representation compactness, reconstruction speed, and rendering fidelity. Previous solutions, whether based on iterative optimization or feed-forward inference, suffer from significant trade-offs between these goals, mainly due to the reliance on local, heuristic-driven allocation strategies that lack global scene awareness. Specifically, current feed-forward methods are largely pixel-aligned or voxel-aligned. By unprojecting pixels into dense, view-aligned primitives, they bake redundancy into the 3D asset. As more input views are added, the representation size increases and global consistency becomes fragile. To this end, we introduce GlobalSplat, a framework built on the principle of align first, decode later. Our approach learns a compact, global, latent scene representation that encodes multi-view input and resolves cross-view correspondences before decoding any explicit 3D geometry. Crucially, this formulation enables compact, globally consistent reconstructions without relying on pretrained pixel-prediction backbones or reusing latent features from dense baselines. Utilizing a coarse-to-fine training curriculum that gradually increases decoded capacity, GlobalSplat natively prevents representation bloat. On RealEstate10K and ACID, our model achieves competitive novel-view synthesis performance while utilizing as few as 16K Gaussians, significantly less than required by dense pipelines, obtaining a light 4MB footprint. Further, GlobalSplat enables significantly faster inference than the baselines, operating under 78 milliseconds in a single forward pass. Project page is available at https://r-itk.github.io/globalsplat/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15612v1">GaussianFlow SLAM: Monocular Gaussian Splatting SLAM Guided by GaussianFlow</a></div>
    <div class="paper-meta">
      📅 2026-04-17
      | 💬 8 pages, 5 figures, 7 tables, accepted to IEEE RA-L
    </div>
    <details class="paper-abstract">
      Gaussian splatting has recently gained traction as a compelling map representation for SLAM systems, enabling dense and photo-realistic scene modeling. However, its application to monocular SLAM remains challenging due to the lack of reliable geometric cues from monocular input. Without geometric supervision, mapping or tracking could fall in local-minima, resulting in structural degeneracies and inaccuracies. To address this challenge, we propose GaussianFlow SLAM, a monocular 3DGS-SLAM that leverages optical flow as a geometry-aware cue to guide the optimization of both the scene structure and camera poses. By encouraging the projected motion of Gaussians, termed GaussianFlow, to align with the optical flow, our method introduces consistent structural cues to regularize both map reconstruction and pose estimation. Furthermore, we introduce normalized error-based densification and pruning modules to refine inactive and unstable Gaussians, thereby contributing to improved map quality and pose accuracy. Experiments conducted on public datasets demonstrate that our method achieves superior rendering quality and tracking accuracy compared with state-of-the-art algorithms. The source code is available at: https://github.com/url-kaist/gaussianflow-slam.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15284v1">GlobalSplat: Efficient Feed-Forward 3D Gaussian Splatting via Global Scene Tokens</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      The efficient spatial allocation of primitives serves as the foundation of 3D Gaussian Splatting, as it directly dictates the synergy between representation compactness, reconstruction speed, and rendering fidelity. Previous solutions, whether based on iterative optimization or feed-forward inference, suffer from significant trade-offs between these goals, mainly due to the reliance on local, heuristic-driven allocation strategies that lack global scene awareness. Specifically, current feed-forward methods are largely pixel-aligned or voxel-aligned. By unprojecting pixels into dense, view-aligned primitives, they bake redundancy into the 3D asset. As more input views are added, the representation size increases and global consistency becomes fragile. To this end, we introduce GlobalSplat, a framework built on the principle of align first, decode later. Our approach learns a compact, global, latent scene representation that encodes multi-view input and resolves cross-view correspondences before decoding any explicit 3D geometry. Crucially, this formulation enables compact, globally consistent reconstructions without relying on pretrained pixel-prediction backbones or reusing latent features from dense baselines. Utilizing a coarse-to-fine training curriculum that gradually increases decoded capacity, GlobalSplat natively prevents representation bloat. On RealEstate10K and ACID, our model achieves competitive novel-view synthesis performance while utilizing as few as 16K Gaussians, significantly less than required by dense pipelines, obtaining a light 4MB footprint. Further, GlobalSplat enables significantly faster inference than the baselines, operating under 78 milliseconds in a single forward pass. Project page is available at https://r-itk.github.io/globalsplat/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15239v1">TokenGS: Decoupling 3D Gaussian Prediction from Pixels with Learnable Tokens</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 Project page: https://research.nvidia.com/labs/toronto-ai/tokengs
    </div>
    <details class="paper-abstract">
      In this work, we revisit several key design choices of modern Transformer-based approaches for feed-forward 3D Gaussian Splatting (3DGS) prediction. We argue that the common practice of regressing Gaussian means as depths along camera rays is suboptimal, and instead propose to directly regress 3D mean coordinates using only a self-supervised rendering loss. This formulation allows us to move from the standard encoder-only design to an encoder-decoder architecture with learnable Gaussian tokens, thereby unbinding the number of predicted primitives from input image resolution and number of views. Our resulting method, TokenGS, demonstrates improved robustness to pose noise and multiview inconsistencies, while naturally supporting efficient test-time optimization in token space without degrading learned priors. TokenGS achieves state-of-the-art feed-forward reconstruction performance on both static and dynamic scenes, producing more regularized geometry and more balanced 3DGS distribution, while seamlessly recovering emergent scene attributes such as static-dynamic decomposition and scene flow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14782v1">One-shot Compositional 3D Head Avatars with Deformable Hair</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 project page: https://yuansun-xjtu.github.io/CompHairHead.io
    </div>
    <details class="paper-abstract">
      We propose a compositional method for constructing a complete 3D head avatar from a single image. Prior one-shot holistic approaches frequently fail to produce realistic hair dynamics during animation, largely due to inadequate decoupling of hair from the facial region, resulting in entangled geometry and unnatural deformations. Our method explicitly decouples hair from the face, modeling these components using distinct deformation paradigms while integrating them into a unified rendering pipeline. Furthermore, by leveraging image-to-3D lifting techniques, we preserve fine-grained textures from the input image to the greatest extent possible, effectively mitigating the common issue of high-frequency information loss in generalized models. Specifically, given a frontal portrait image, we first perform hair removal to obtain a bald image. Both the original image and the bald image are then lifted to dense, detail-rich 3D Gaussian Splatting (3DGS) representations. For the bald 3DGS, we rig it to a FLAME mesh via non-rigid registration with a prior model, enabling natural deformation that follows the mesh triangles during animation. For the hair component, we employ semantic label supervision combined with a boundary-aware reassignment strategy to extract a clean and isolated set of hair Gaussians. To control hair deformation, we introduce a cage structure that supports Position-Based Dynamics (PBD) simulation, allowing realistic and physically plausible transformations of the hair Gaussian primitives under head motion, gravity, and inertial effects. Striking qualitative results, including dynamic animations under diverse head motions, gravity effects, and expressions, showcase substantially more realistic hair behavior alongside faithfully preserved facial details, outperforming state-of-the-art one-shot methods in perceptual realism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14706v1">NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 Accepted to CVPR 2026 (Highlight)
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled highly efficient and photorealistic novel view synthesis. However, segmenting objects accurately in 3DGS remains challenging due to the discrete nature of Gaussian representations, which often leads to aliasing and artifacts at object boundaries. In this paper, we introduce NG-GS, a novel framework for high-quality object segmentation in 3DGS that explicitly addresses boundary discretization. Our approach begins by automatically identifying ambiguous Gaussians at object boundaries using mask variance analysis. We then apply radial basis function (RBF) interpolation to construct a spatially continuous feature field, enhanced by multi-resolution hash encoding for efficient multi-scale representation. A joint optimization strategy aligns 3DGS with a lightweight NeRF module through alignment and spatial continuity losses, ensuring smooth and consistent segmentation boundaries. Extensive experiments on NVOS, LERF-OVS, and ScanNet benchmarks demonstrate that our method achieves state-of-the-art performance, with significant gains in boundary mIoU. Code is available at https://github.com/BJTU-KD3D/NG-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12580v2">PDF-GS: Progressive Distractor Filtering for Robust 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-16
      | 💬 Accepted to CVPR Findings 2026. Project Page: https://kangrnin.github.io/PDF-GS
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled impressive real-time photorealistic rendering. However, conventional training pipelines inherently assume full multi-view consistency among input images, which makes them sensitive to distractors that violate this assumption and cause visual artifacts. In this work, we revisit an underexplored aspect of 3DGS: its inherent ability to suppress inconsistent signals. Building on this insight, we propose PDF-GS (Progressive Distractor Filtering for Robust 3D Gaussian Splatting), a framework that amplifies this self-filtering property through a progressive multi-phase optimization. The progressive filtering phases gradually remove distractors by exploiting discrepancy cues, while the following reconstruction phase restores fine-grained, view-consistent details from the purified Gaussian representation. Through this iterative refinement, PDF-GS achieves robust, high-fidelity, and distractor-free reconstructions, consistently outperforming baselines across diverse datasets and challenging real-world conditions. Moreover, our approach is lightweight and easily adaptable to existing 3DGS frameworks, requiring no architectural changes or additional inference overhead, leading to a new state-of-the-art performance. The code is publicly available at https://github.com/kangrnin/PDF-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14268v1">HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 Project Page: https://3d-models.hunyuan.tencent.com/world/ ; Code: https://github.com/Tencent-Hunyuan/HY-World-2.0
    </div>
    <details class="paper-abstract">
      We introduce HY-World 2.0, a multi-modal world model framework that advances our prior project HY-World 1.0. HY-World 2.0 accommodates diverse input modalities, including text prompts, single-view images, multi-view images, and videos, and produces 3D world representations. With text or single-view image inputs, the model performs world generation, synthesizing high-fidelity, navigable 3D Gaussian Splatting (3DGS) scenes. This is achieved through a four-stage method: a) Panorama Generation with HY-Pano 2.0, b) Trajectory Planning with WorldNav, c) World Expansion with WorldStereo 2.0, and d) World Composition with WorldMirror 2.0. Specifically, we introduce key innovations to enhance panorama fidelity, enable 3D scene understanding and planning, and upgrade WorldStereo, our keyframe-based view generation model with consistent memory. We also upgrade WorldMirror, a feed-forward model for universal 3D prediction, by refining model architecture and learning strategy, enabling world reconstruction from multi-view images or videos. Also, we introduce WorldLens, a high-performance 3DGS rendering platform featuring a flexible engine-agnostic architecture, automatic IBL lighting, efficient collision detection, and training-rendering co-design, enabling interactive exploration of 3D worlds with character support. Extensive experiments demonstrate that HY-World 2.0 achieves state-of-the-art performance on several benchmarks among open-source approaches, delivering results comparable to the closed-source model Marble. We release all model weights, code, and technical details to facilitate reproducibility and support further research on 3D world models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13746v1">ClipGStream: Clip-Stream Gaussian Splatting for Any Length and Any Motion Multi-View Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 CVPR 2026, Project pages: https://liangjie1999.github.io/ClipGStreamWeb/
    </div>
    <details class="paper-abstract">
      Dynamic 3D scene reconstruction is essential for immersive media such as VR, MR, and XR, yet remains challenging for long multi-view sequences with large-scale motion. Existing dynamic Gaussian approaches are either Frame-Stream, offering scalability but poor temporal stability, or Clip, achieving local consistency at the cost of high memory and limited sequence length. We propose ClipGStream, a hybrid reconstruction framework that performs stream optimization at the clip level rather than the frame level. The sequence is divided into short clips, where dynamic motion is modeled using clip-independent spatio-temporal fields and residual anchor compensation to capture local variations efficiently, while inter-clip inherited anchors and decoders maintain structural consistency across clips. This Clip-Stream design enables scalable, flicker-free reconstruction of long dynamic videos with high temporal coherence and reduced memory overhead. Extensive experiments demonstrate that ClipGStream achieves state-of-the-art reconstruction quality and efficiency. The project page is available at: https://liangjie1999.github.io/ClipGStreamWeb/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13589v1">Dehaze-then-Splat: Generative Dehazing with Physics-Informed 3D Gaussian Splatting for Smoke-Free Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      We present Dehaze-then-Splat, a two-stage pipeline for multi-view smoke removal and novel view synthesis developed for Track~2 of the NTIRE 2026 3D Restoration and Reconstruction Challenge. In the first stage, we produce pseudo-clean training images via per-frame generative dehazing using Nano Banana Pro, followed by brightness normalization. In the second stage, we train 3D Gaussian Splatting (3DGS) with physics-informed auxiliary losses -- depth supervision via Pearson correlation with pseudo-depth, dark channel prior regularization, and dual-source gradient matching -- that compensate for cross-view inconsistencies inherent in frame-wise generative processing. We identify a fundamental tension in dehaze-then-reconstruct pipelines: per-image restoration quality does not guarantee multi-view consistency, and such inconsistency manifests as blurred renders and structural instability in downstream 3D reconstruction.Our analysis shows that MCMC-based densification with early stopping, combined with depth and haze-suppression priors, effectively mitigates these artifacts. On the Akikaze validation scene, our pipeline achieves 20.98\,dB PSNR and 0.683 SSIM for novel view synthesis, a +1.50\,dB improvement over the unregularized baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.09176v2">LIVE-GS: LLM Powers Interactive VR Experience with Physics-Aware Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      As 3D Gaussian Splatting (3DGS) emerges as a leading approach for novel view synthesis and scene reconstruction, its potential in digital asset creation has gained significant attention. An increasing number of asset libraries based on GS are being established. However, generating physics-based dynamic assets remains a time-consuming and expertise-intensive task, especially for non-experts. In this paper, we propose LIVE-GS, a highly realistic Virtual Reality (VR) system powered by Large Language Models (LLMs), which enables rapid creation of dynamic Gaussian assets and real-time VR interactions. To inform our system design, we conducted interviews to examine challenges faced by current GS-based VR systems and the specific demands of users. Based on these insights, we employed GPT-4o to analyze key physical properties of objects that significantly impact user interactions, ensuring physics-based interactions in VR align with real-world phenomena. A key innovation of LIVE-GS is its ability to predict reasonable parameters in just 10 seconds from static Gaussian assets while maintaining high-quality VR interactions. To validate our approach, we invited participants experienced in physical simulation to manually adjust physical parameters, providing a baseline for comparison in both asset quality and authoring efficiency. We also conducted a comprehensive user study to evaluate system usability and user satisfaction. Experimental results demonstrate that LIVE-GS, leveraging LLMs' scene understanding capabilities, can achieve efficient physical scene creation and natural interactions without requiring manual design or annotation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13492v1">RadarSplat-RIO: Indoor Radar-Inertial Odometry with Gaussian Splatting-Based Radar Bundle Adjustment</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      Radar is more resilient to adverse weather and lighting conditions than visual and Lidar simultaneous localization and mapping (SLAM). However, most radar SLAM pipelines still rely heavily on frame-to-frame odometry, which leads to substantial drift. While loop closure can correct long-term errors, it requires revisiting places and relies on robust place recognition. In contrast, visual odometry methods typically leverage bundle adjustment (BA) to jointly optimize poses and map within a local window. However, an equivalent BA formulation for radar has remained largely unexplored. We present the first radar BA framework enabled by Gaussian Splatting (GS), a dense and differentiable scene representation. Our method jointly optimizes radar sensor poses and scene geometry using full range-azimuth-Doppler data, bringing the benefits of multi-frame BA to radar for the first time. When integrated with an existing radar-inertial odometry frontend, our approach significantly reduces pose drift and improves robustness. Across multiple indoor scenes, our radar BA achieves substantial gains over the prior radar-inertial odometry, reducing average absolute translational and rotational errors by 90% and 80%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13416v1">DF3DV-1K: A Large-Scale Dataset and Benchmark for Distractor-Free Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      Advances in radiance fields have enabled photorealistic novel view synthesis. In several domains, large-scale real-world datasets have been developed to support comprehensive benchmarking and to facilitate progress beyond scene-specific reconstruction. However, for distractor-free radiance fields, a large-scale dataset with clean and cluttered images per scene remains lacking, limiting the development. To address this gap, we introduce DF3DV-1K, a large-scale real-world dataset comprising 1,048 scenes, each providing clean and cluttered image sets for benchmarking. In total, the dataset contains 89,924 images captured using consumer cameras to mimic casual capture, spanning 128 distractor types and 161 scene themes across indoor and outdoor environments. A curated subset of 41 scenes, DF3DV-41, is systematically designed to evaluate the robustness of distractor-free radiance field methods under challenging scenarios. Using DF3DV-1K, we benchmark nine recent distractor-free radiance field methods and 3D Gaussian Splatting, identifying the most robust methods and the most challenging scenarios. Beyond benchmarking, we demonstrate an application of DF3DV-1K by fine-tuning a diffusion-based 2D enhancer to improve radiance field methods, achieving average improvements of 0.96 dB PSNR and 0.057 LPIPS on the held-out set (e.g., DF3DV-41) and the On-the-go dataset. We hope DF3DV-1K facilitates the development of distractor-free vision and promotes progress beyond scene-specific approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13340v1">MSGS: Multispectral 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Published in IEEE ISMAR 2025 Adjunct
    </div>
    <details class="paper-abstract">
      We present a multispectral extension to 3D Gaussian Splatting (3DGS) for wavelength-aware view synthesis. Each Gaussian is augmented with spectral radiance, represented via per-band spherical harmonics, and optimized under a dual-loss supervision scheme combining RGB and multispectral signals. To improve rendering fidelity, we perform spectral-to-RGB conversion at the pixel level, allowing richer spectral cues to be retained during optimization. Our method is evaluated on both public and self-captured real-world datasets, demonstrating consistent improvements over the RGB-only 3DGS baseline in terms of image quality and spectral consistency. Notably, it excels in challenging scenes involving translucent materials and anisotropic reflections. The proposed approach maintains the compactness and real-time efficiency of 3DGS while laying the foundation for future integration with physically based shading models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13333v1">SSD-GS: Scattering and Shadow Decomposition for Relightable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Accepted to ICLR 2026. Code available at: https://github.com/irisfreesiri/SSD-GS
    </div>
    <details class="paper-abstract">
      We present SSD-GS, a physically-based relighting framework built upon 3D Gaussian Splatting (3DGS) that achieves high-quality reconstruction and photorealistic relighting under novel lighting conditions. In physically-based relighting, accurately modeling light-material interactions is essential for faithful appearance reproduction. However, existing 3DGS-based relighting methods adopt coarse shading decompositions, either modeling only diffuse and specular reflections or relying on neural networks to approximate shadows and scattering. This leads to limited fidelity and poor physical interpretability, particularly for anisotropic metals and translucent materials. To address these limitations, SSD-GS decomposes reflectance into four components: diffuse, specular, shadow, and subsurface scattering. We introduce a learnable dipole-based scattering module for subsurface transport, an occlusion-aware shadow formulation that integrates visibility estimates with a refinement network, and an enhanced specular component with an anisotropic Fresnel-based model. Through progressive integration of all components during training, SSD-GS effectively disentangles lighting and material properties, even for unseen illumination conditions, as demonstrated on the challenging OLAT dataset. Experiments demonstrate superior quantitative and perceptual relighting quality compared to prior methods and pave the way for downstream tasks, including controllable light source editing and interactive scene relighting. The source code is available at: https://github.com/irisfreesiri/SSD-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13153v1">PatchPoison: Poisoning Multi-View Datasets to Degrade 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 CVPR Workshop on Security, Privacy, and Adversarial Robustness in 3D Generative Vision Models (SPAR-3D), 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently enabled highly photorealistic 3D reconstruction from casually captured multi-view images. However, this accessibility raises a privacy concern: publicly available images or videos can be exploited to reconstruct detailed 3D models of scenes or objects without the owner's consent. We present PatchPoison, a lightweight dataset-poisoning method that prevents unauthorized 3D reconstruction. Unlike global perturbations, PatchPoison injects a small high-frequency adversarial patch, a structured checkerboard, into the periphery of each image in a multi-view dataset. The patch is designed to corrupt the feature-matching stage of Structure-from-Motion (SfM) pipelines such as COLMAP by introducing spurious correspondences that systematically misalign estimated camera poses. Consequently, downstream 3DGS optimization diverges from the correct scene geometry. On the NeRF-Synthetic benchmark, inserting a 12 X 12 pixel patch increases reconstruction error by 6.8x in LPIPS, while the poisoned images remain unobtrusive to human viewers. PatchPoison requires no pipeline modifications, offering a practical, "drop-in" preprocessing step for content creators to protect their multi-view data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12942v1">RMGS-SLAM: Real-time Multi-sensor Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Real-time 3D Gaussian splatting (3DGS)-based Simultaneous Localization and Mapping (SLAM) in large-scale real-world environments remains challenging, as existing methods often struggle to jointly achieve low-latency pose estimation, 3D Gaussian reconstruction in step with incoming sensor streams, and long-term global consistency. In this paper, we present a tightly coupled LiDAR-Inertial-Visual (LIV) 3DGS-based SLAM framework for real-time pose estimation and photorealistic mapping in large-scale real-world scenes. The system executes state estimation and 3D Gaussian primitive initialization in parallel with global Gaussian optimization, thereby enabling continuous dense mapping. To improve Gaussian initialization quality and accelerate optimization convergence, we introduce a cascaded strategy that combines feed-forward predictions with voxel-based principal component analysis (voxel-PCA) geometric priors. To enhance global consistency in large scenes, we further perform loop closure directly on the optimized global Gaussian map by estimating loop constraints through Gaussian-based Generalized Iterative Closest Point (GICP) registration, followed by pose-graph optimization. In addition, we collected challenging large-scale looped outdoor SLAM sequences with hardware-synchronized LiDAR-camera-IMU and ground-truth trajectories to support realistic and comprehensive evaluation. Extensive experiments on both public datasets and our dataset demonstrate that the proposed method achieves a strong balance among real-time efficiency, localization accuracy, and rendering quality across diverse and challenging real-world scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12837v1">GGD-SLAM: Monocular 3DGS SLAM Powered by Generalizable Motion Model for Dynamic Environments</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 8 pages, Accepted by ICRA 2026
    </div>
    <details class="paper-abstract">
      Visual SLAM algorithms achieve significant improvements through the exploration of 3D Gaussian Splatting (3DGS) representations, particularly in generating high-fidelity dense maps. However, they depend on a static environment assumption and experience significant performance degradation in dynamic environments. This paper presents GGD-SLAM, a framework that employs a generalizable motion model to address the challenges of localization and dense mapping in dynamic environments - without predefined semantic annotations or depth input. Specifically, the proposed system employs a First-In-First-Out (FIFO) queue to manage incoming frames, facilitating dynamic semantic feature extraction through a sequential attention mechanism. This is integrated with a dynamic feature enhancer to separate static and dynamic components. Additionally, to minimize dynamic distractors' impact on the static components, we devise a method to fill occluded areas via static information sampling and design a distractor-adaptive Structure Similarity Index Measure (SSIM) loss tailored for dynamic environments, significantly enhancing the system's resilience. Experiments conducted on real-world dynamic datasets demonstrate that the proposed system achieves state-of-the-art performance in camera pose estimation and dense reconstruction in dynamic scenes.
    </details>
</div>
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
