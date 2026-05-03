# gaussian splatting - 2026_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.28179v1">Stop Holding Your Breath: CT-Informed Gaussian Splatting for Dynamic Bronchoscopy</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Bronchoscopic navigation relies on registering endoscopic video to a preoperative CT scan, but respiratory motion deforms the airway by 5-20 mm, creating CT-to-body divergence that limits localization accuracy. In practice, this is mitigated through breath-hold protocols, which attempt to match the intraoperative anatomy to a static CT, but are difficult to reproduce and disrupt clinical workflow. We propose to eliminate the need for breath-hold protocols by leveraging patient-specific respiratory modeling. Paired inhale-exhale CT scans, already acquired for planning, implicitly define the patient-specific deformation space of the breathing airway. By registering these scans, we reduce respiratory motion to a single scalar breathing phase per frame, constraining all reconstructions to anatomically observed configurations. We embed this representation within a mesh-anchored Gaussian splatting framework, where a lightweight estimator infers breathing phase directly from endoscopic RGB, enabling continuous, deformation-aware reconstruction throughout the respiratory cycle without breath-holds or external sensing. To enable quantitative evaluation, we introduce RESPIRE, a physically grounded bronchoscopy simulation pipeline with per-frame ground truth for geometry, pose, breathing phase, and deformation. Experiments on RESPIRE show that our approach achieves geometrically faithful reconstruction, over 20x faster training, and 1.22 mm target localization accuracy (within the 3mm clinically relevant tolerances) outperforming unconstrained single-CT baselines. Please check out our website for additional visuals: https://asdunnbe.github.io/RESPIRE/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.28111v1">GSDrive: Reinforcing Driving Policies by Multi-mode Trajectory Probing with 3D Gaussian Splatting Environment</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 initial version
    </div>
    <details class="paper-abstract">
      End-to-end (E2E) autonomous driving presents a promising approach for translating perceptual inputs directly into driving actions. However, prohibitive annotation costs and temporal data quality degradation hinder long-term real-world deployment. While combining imitation learning (IL) and reinforcement learning (RL) is a common strategy for policy improvement, conventional RL training relies on delayed, event-based rewards-policies learn only from catastrophic outcomes such as collisions, leading to premature convergence to suboptimal behaviors. To address these limitations, we introduce GSDrive, a framework that exploits 3D Gaussian Splatting (3DGS) for differentiable, physics-based reward shaping in E2E driving policy improvement. Our method incorporates a flow matching-based trajectory predictor within the 3DGS simulator, enabling multi-mode trajectory probing where candidate trajectories are rolled out to assess prospective rewards. This establishes a bidirectional knowledge exchange between IL and RL by grounding reward functions in physically simulated interaction signals, offering immediate dense feedback instead of sparse catastrophic events. Evaluated on the reconstructed nuScenes dataset, our method surpasses existing simulation-based RL driving approaches in closed-loop experiments. Code is available at https://github.com/ZionGo6/GSDrive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.28016v1">Faster 3D Gaussian Splatting Convergence via Structure-Aware Densification</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 Siggraph 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has emerged as a powerful scene representation for real-time novel-view synthesis. However, its standard adaptive density control relies on screen-space positional gradients, which do not distinguish between geometric misplacement and frequency aliasing, often leading to either over-blurred high-frequency textures or inefficient over-densification. We present a structure-aware densification framework. Our key insight is that the decision to subdivide a Gaussian should be driven by an explicit comparison between its projected screen-space extent and the local structure of the texture it seeks to represent. We introduce a multi-scale frequency analysis combining structure tensors with Laplacian scale space analysis to estimate the dominant frequency at each pixel, enabling robust supervision across varying texture scales. Based on this analysis, we define $η$, a per-Gaussian, per-axis frequency violation metric that indicates when a primitive may be under-resolving local texture details. Unlike methods that perform isotropic splitting (e.g., splitting each Gaussian into two smaller ones with uniform shape), our approach performs anisotropic splitting. For each axis with high $η$, we compute a split factor to better resolve the local frequency content. We further introduce a multiview consistency criterion that aggregates $η$ observations across multiple views. By performing densification early and faster, we skip the lengthy iterative densification phases required by baseline methods and achieve significantly faster convergence. Experiments on standard benchmarks demonstrate that our method also achieves superior reconstruction quality, particularly in high-frequency regions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27590v1">Fake3DGS: A Benchmark for 3D Manipulation Detection in Neural Rendering</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 Accepted at ICPR 2026. Code and data: https://github.com/iot-unimore/Fake3DGS
    </div>
    <details class="paper-abstract">
      Recent advances in 3D reconstruction and neural rendering,particularly 3D Gaussian Splatting, make it feasible and simple to edit 3D scenes and re-render them as highly realistic images. Therefore, security concerns arise regarding the authenticity of 3D content. Despite this threat, 3D fake detection remains largely unexplored in the literature, and most existing work is limited to 2D space. Therefore, in this paper, we formalize the concept of 3D fake detection and introduce Fake3DGS, a dataset of 3D Gaussian splatting scenes and corresponding rendered views, where fake images are produced by controlled manipulations of geometry, appearance, and spatial layout, while preserving high visual realism. Using this benchmark, we demonstrate that current state-of-the-art 2D detectors struggle to distinguish between original and 3D manipulated images. To bridge this gap, we introduce a 3D-aware detection method that leverages multi-view coherence and features derived from the Gaussian splatting representation. Experimental results demonstrate a substantial improvement in recognizing modified 3D content, underscoring the validity of the new dataset and the necessity for authenticity assessment techniques that extend beyond 2D evidence. Code and data are publicly released for future investigations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27572v1">SandSim: Curve-Guided Gaussian Splatting for Reconstructing Sand Painting Processes</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Sand painting is a process-driven art where visual appearance emerges from granular accumulation. Given a single image, reconstructing a plausible sand painting process requires modeling coherent stroke structures and material-dependent effects. Existing methods, including stroke-based optimization and diffusion-based video synthesis, often lack structural coherence and material consistency, leading to unrealistic drawing sequences. We present SandSim, a framework that reconstructs a sand painting process from a single image. We introduce a curve-guided Gaussian representation that models strokes as sequences of anisotropic primitives along continuous trajectories, whose smooth kernels capture the soft boundaries of sand strokes and enable coherent stroke formation. We further adopt a subtractive compositing scheme to model light attenuation during sand accumulation. We incorporate a semantic-guided planning module for scene decomposition and drawing order inference. Our framework jointly optimizes stroke geometry and appearance and can be integrated with a physics-based simulator for interactive sand dynamics and editing. Experiments show that our method produces temporally coherent and visually realistic results, achieving improved reconstruction quality and perceptual fidelity compared to existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27552v1">Residual Gaussian Splatting for Ultra Sparse-View CBCT Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      While 3D Gaussian splatting (3DGS) offers explicit and efficient scene representations for cone-beam computed tomography reconstruction, conventional photometric optimization inherently suffers from spectral bias under ultra sparse-view conditions, leading to over-smoothing and a loss of high-frequency anatomical details. Since wavelet transforms provide rich high-frequency information and have been widely utilized to enhance sparse reconstruction, this work integrates wavelet multi-resolution analysis with 3DGS. To circumvent the mathematical mismatch between the strict non-negativity of physical X-ray attenuation and the bipolar nature of high-frequency wavelet coefficients, we propose Residual Gaussian Splatting (RGS). Methodologically, we introduce a spectrally-decoupled Gaussian representation that stratifies the volumetric field into a geometric base component and a residual detail component. This decomposition systematically transforms explicit high-frequency fitting into a physically consistent, implicit residual compensation task. Furthermore, we devise a spectral-spatial collaborative optimization strategy to coordinate the interplay between geometric anchoring and texture refinement, effectively preventing spectral crosstalk. Extensive experiments on clinical datasets demonstrate that RGS enables the reconstructed images to capture highly refined geometric textures. It successfully resolves the trade-off between artifact suppression and detail preservation, yielding superior visual fidelity in complex trabecular and vascular structures compared to existing neural rendering baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.22676v5">TranSplat: Instant Object Relighting in Gaussian Splatting via Spherical Harmonic Radiance Transfer</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      We present TranSplat, a method for instant, accurate object relighting within the Gaussian Splatting (GS) framework. Rather than relying on costly inverse rendering routines, we propose a BRDF-free radiance transfer strategy that analytically modulates the spherical harmonic (SH) appearance coefficients of an object's 2D Gaussian surfels using per-normal irradiance ratios derived from source and target environment maps. To handle view-dependent and glossy appearances without explicit material estimation, we introduce a specularity-aware dual-path SH transfer strategy that adapts higher-order SH bands in the reflection domain. Additionally, we propose a lightweight SH-domain self-shadowing module to ensure physically realistic occlusion without explicit mesh raycasting. Operating as a post-processing step, TranSplat requires no additional GS retraining for a pair of source and target scenes. Evaluations on synthetic and real-world objects demonstrate state-of-the-art accuracy, outperforming recent inverse-rendering and diffusion-based GS relighting methods across most conditions, all while completing relighting operations in under one second. Although bounded by radially symmetric BRDF approximations and the low-pass nature of the SH basis, TranSplat produces perceptually realistic renderings even for glossy, complex materials, establishing a valuable, lightweight path forward for GS relighting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27437v1">Softmax-GS: Generalized Gaussians Learning When to Blend or Bound</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D GS) is widely adopted for novel view synthesis due to its high training and rendering efficiency. However, its efficiency relies on the key assumption that Gaussians do not overlap in the 3D space, which leads to noticeable artifacts and view inconsistencies. In addition, the inherently diffuse boundaries of Gaussians hinder accurate reconstruction of sharp object edges. We propose Softmax-GS, a unified solution that addresses both the view-inconsistency and the diffuse-boundary problem by enforcing a softmax-based competition in overlapping regions between two Gaussians. With learnable parameters controlling the strength of the competition, it enables a continuous spectrum from smooth color blending to crisp, well-defined boundaries. Our formulation explicitly preserves order invariance for any two overlapping Gaussians and ensures that the output transmittance remains unchanged irrespective of the extent of overlapping, preventing undesirable discontinuities in the rendered output. Ablation experiments on simple geometries demonstrate the effectiveness of each component of Softmax-GS, and evaluations on real-world benchmarks show that it achieves state-of-the-art performance, improving both reconstruction quality and parameter efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27422v1">Sparse-View 3D Gaussian Splatting in the Wild</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 18 pages, 14 figures, and 14 tables
    </div>
    <details class="paper-abstract">
      We propose a 3D novel sparse-view synthesis framework for unconstrained real-world scenarios that contain distractors. Unlike existing methods that primarily perform novel-view synthesis from a sparse set of constrained images without transient elements or leverage unconstrained dense image collections to enhance 3D representation in real-world scenarios, our method not only effectively tackles sparse unconstrained image collections, but also shows high-quality 3D rendering results. To do this, we introduce reference-guided view refinement with a diffusion model using a transient mask and a reference image to enhance the 3D representation and mitigate artifacts in rendered views. Furthermore, we address sparse regions in the Gaussian field via pseudo-view generation along with a sparsity-aware Gaussian replication strategy to amplify Gaussians in the sparse regions. Extensive experiments on publicly available datasets demonstrate that our methodology consistently outperforms existing methods (e.g., PSNR - 17.2%, SSIM - 10.8%, LPIPS - 4.0%) and provides high-fidelity 3D rendering results. This advancement paves the way for realizing unconstrained real-world scenarios without labor-intensive data acquisition. Our project page is available at $\href{https://robotic-vision-lab.github.io/SaveWildGS/}{here}$
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19571v2">TransSplat: Unbalanced Semantic Transport for Language-Driven 3DGS Editing</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Language-driven 3D Gaussian Splatting (3DGS) editing provides a more convenient approach for modifying complex scenes in VR/AR. Standard pipelines typically adopt a two-stage strategy: first editing multiple 2D views, and then optimizing the 3D representation to match these edited observations. Existing methods mainly improve view consistency through multi-view feature fusion, attention filtering, or iterative recalibration. However, they fail to explicitly address a more fundamental issue: the semantic correspondence between edited 2D evidence and 3D Gaussians. To tackle this problem, we propose TransSplat, which formulates language-driven 3DGS editing as a multi-view unbalanced semantic transport problem. Specifically, our method establishes correspondences between visible Gaussians and view-specific editing prototypes, thereby explicitly characterizing the semantic relationship between edited 2D evidence and 3D Gaussians. It further recovers a cross-view shared canonical 3D edit field to guide unified 3D appearance updates. In addition, we use transport residuals to suppress erroneous edits in non-target regions, mitigating edit leakage and improving local control precision. Qualitative and quantitative results show that, compared with existing 3D editing methods centered on enhancing view consistency, TransSplat achieves superior performance in local editing accuracy and structural consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26920v1">Color-Encoded Illumination for High-Speed Volumetric Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 accepted to IEEE CVPR 2026 as a highlight
    </div>
    <details class="paper-abstract">
      The task of capturing and rendering 3D dynamic scenes from 2D images has become increasingly popular in recent years. However, most conventional cameras are bandwidth-limited to 30-60 FPS, restricting these methods to static or slowly evolving scenes. While overcoming bandwidth limitations is difficult for general scenes, recent years have seen a flurry of computational imaging methods that yield high-speed videos using conventional cameras for specific applications (e.g., motion capture and particle image velocimetry). However, most of these methods require modifications to a camera's optics or the addition of mechanically moving components, limiting them to a single-view high-speed capture. Consequently, these methods cannot be readily used to capture a 3D representation of rapid scene motion. In this paper, we propose a novel method to capture and reconstruct a volumetric representation of a high-speed scene using only unaugmented low-speed cameras. Instead of modifying the hardware or optics of each individual camera, we encode high-speed scene dynamics by illuminating the scene with a rapid, sequential color-coded sequence. This results in simultaneous multi-view capture of the scene, where high-speed temporal information is encoded in the spatial intensity and color variations of the captured images. To construct a high-speed volumetric representation of the dynamic scene, we develop a novel dynamic Gaussian Splatting-based approach that decodes the temporal information from the images. We evaluate our approach on simulated scenes and real-world experiments using a multi-camera imaging setup, showing first-of-a-kind high-speed volumetric scene reconstructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.16747v2">Incoherent Deformation, Not Capacity: Diagnosing and Mitigating Overfitting in Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 10 pages, 6 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Dynamic 3D Gaussian Splatting methods achieve strong training-view PSNR on monocular video but generalize poorly: on the D-NeRF benchmark we measure an average train-test PSNR gap of 6.18 dB, rising to 11 dB on individual scenes. We report two findings that together account for most of that gap. Finding 1 (the role of splitting). A systematic ablation of the Adaptive Density Control pipeline (split, clone, prune, frequency, threshold, schedule) shows that splitting is responsible for over 80% of the gap: disabling split collapses the cloud from 44K to 3K Gaussians and the gap from 6.18 dB to 1.15 dB. Across all threshold-varying ablations, gap is log-linear in count (r = 0.995, bootstrap 95% CI [0.99, 1.00]), which suggests a capacity-based explanation. Finding 2 (the role of deformation coherence). We show that the capacity explanation is incomplete. A local-smoothness penalty on the per-Gaussian deformation field -- Elastic Energy Regularization (EER) -- reduces the gap by 40.8% while growing the cloud by 85%. Measuring per-Gaussian strain directly on trained checkpoints, EER reduces mean strain by 99.72% (median 99.80%) across all 8 scenes; on 8/8 scenes the median Gaussian under EER is less strained than the 1st-percentile (best-behaved) Gaussian under baseline. Alongside EER, we evaluate two further regularizers: GAD, a loss-rate-aware densification threshold, and PTDrop, a jitter-weighted Gaussian dropout. GAD+EER reduces the gap by 48%; adding PTDrop and a soft growth cap reaches 57%. We confirm that coherence generalizes to (a) a different deformation architecture (Deformable-3DGS, +40.6% gap reduction at re-tuned lambda), and (b) real monocular video (4 HyperNeRF scenes, reducing the mean PSNR gap by 14.9% at the same lambda as D-NeRF, with near-zero quality cost). The overfitting in dynamic 3DGS is driven by incoherent deformation, not parameter count.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26799v1">MesonGS++: Post-training Compression of 3D Gaussian Splatting with Hyperparameter Searching</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 https://github.com/mmlab-sigs/mesongs_plus
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves high-quality novel view synthesis with real-time rendering, but its storage cost remains prohibitive for practical deployment. Existing post-training compression methods still rely on many coupled hyperparameters across pruning, transformation, quantization, and entropy coding, making it difficult to control the final compressed size and fully exploit the rate-distortion trade-off. We propose MesonGS++, a size-aware post-training codec for 3D Gaussian compression. On the codec side, MesonGS++ combines joint importance-based pruning, octree geometry coding, attribute transformation, selective vector quantization for higher-degree spherical harmonics, and group-wise mixed-precision quantization with entropy coding. On the configuration side, it treats the reserve ratio and bit-width allocation as the dominant rate-distortion knobs and jointly optimizes them under a target storage budget via discrete sampling and 0--1 integer linear programming. We further propose a linear size estimator and a CUDA parallel quantization operator to accelerate the hyperparameter searching process. Extensive experiments show that MesonGS++ achieves over 34$\times$ compression while preserving rendering fidelity, outperforming state-of-the-art post-training methods and accurately meeting target size budgets. Remarkably, without any training, MesonGS++ can even surpass the PSNR of vanilla 3DGS at a 20$\times$ compression rate on the Stump scene. Our code is available at https://github.com/mmlab-sigs/mesongs_plus
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19216v2">Bridging Visual and Wireless Sensing via a Unified Radiation Field for 3D Radio Map Construction</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 The code for this work will be publicly available at: https://github.com/wenchaozheng/URF-GS
    </div>
    <details class="paper-abstract">
      The emerging applications of next-generation wireless networks demand high-fidelity environmental intelligence. 3D radio maps bridge physical environments and electromagnetic propagation for spectrum planning and environment-aware sensing. However, most existing methods treat visual and wireless data as independent modalities and fail to leverage shared electromagnetic propagation principles. To bridge this gap, we propose URF-GS, a unified radio-optical radiation field framework based on 3D Gaussian splatting and inverse rendering for 3D radio map construction. By fusing cross-modal observations, our method recovers scene geometry and material properties to predict radio signals under arbitrary transceiver configurations without retraining. Experiments demonstrate up to a 24.7% improvement in spatial spectrum accuracy and a 10x increase in sample efficiency compared with NeRF-based methods. We further showcase URF-GS in Wi-Fi AP deployment and robot path planning tasks. This unified visual-wireless representation supports holistic radiation field modeling for future wireless communication systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26262v1">Semantic Foam: Unifying Spatial and Semantic Scene Decomposition</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 15 pages, 10 figures, Accepted to CVPR 2026 (Highlight) , Project page: http://semanticfoam.github.io/
    </div>
    <details class="paper-abstract">
      Modern scene reconstruction methods, such as 3D Gaussian Splatting, enable photo-realistic novel view synthesis at real-time speeds. However, their adoption in interactive graphics applications remains limited due to the difficulty of interacting with these representations compared to traditional, human-authored 3D assets. While prior work has attempted to impose semantic decomposition on these models, significant challenges remain in segmentation quality and cross-view consistency.To address these limitations, we introduce Semantic Foam, which extends the recently proposed Radiant Foam representation to semantic decomposition tasks. Our approach leverages the inherent spatial structure of Radiant Foam's volumetric Voronoi mesh and augments it with an explicit semantic feature field defined at the cell level. This design enables direct spatial regularization, improving consistency across views and mitigating artifacts caused by occlusion and inconsistent supervision, which are common issues in point-based representations.Experimental results demonstrate that our method achieves superior object-level segmentation performance compared to state-of-the-art approaches such as Gaussian Grouping and SAGA.Project page: http://semanticfoam.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26238v1">EnerGS: Energy-Based Gaussian Splatting with Partial Geometric Priors</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has been widely adopted for scene reconstruction, where training inherently constitutes a highly coupled and non-convex optimization problem. Recent works commonly incorporate geometric priors, such as LiDAR measurements, either for initialization or as training constraints, with the goal of improving photometric reconstruction quality. However, in large-scale outdoor scenarios, such geometric supervision is often spatially incomplete and uneven, which limits its effectiveness as a reliable prior and can even be detrimental to the final reconstruction. To address this challenge, we model partially observable geometry as a continuous energy field induced by geometric evidence and propose EnerGS. Rather than enforcing geometry as a hard constraint, EnerGS provides a soft geometric guidance for the optimization of Gaussian primitives, allowing geometric information to steer the optimization process without directly restricting the solution space. Extensive experiments on large-scale outdoor scenes demonstrate that, under both sparse multi-view and monocular settings, EnerGS consistently improves photometric quality and geometric stability, while effectively mitigating overfitting during 3DGS training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07743v3">UltraGS: Real-Time Physically-Decoupled Gaussian Splatting for Ultrasound Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 Accepted by ICME 2026
    </div>
    <details class="paper-abstract">
      Ultrasound imaging is a cornerstone of non-invasive clinical diagnostics, yet its limited field of view poses challenges for novel view synthesis. We present UltraGS, a real-time framework that adapts Gaussian Splatting to sensorless ultrasound imaging by integrating explicit radiance fields with lightweight, physics-inspired acoustic modeling. UltraGS employs depth-aware Gaussian primitives with learnable fields of view to improve geometric consistency under unconstrained probe motion, and introduces PD Rendering, a differentiable acoustic operator that combines low-order spherical harmonics with first-order wave effects for efficient intensity synthesis. We further present a clinical ultrasound dataset acquired under real-world scanning protocols. Extensive evaluations across three datasets demonstrate that UltraGS establishes a new performance-efficiency frontier, achieving state-of-the-art results in PSNR (up to 29.55) and SSIM (up to 0.89) while achieving real-time synthesis at 64.69 fps on a single GPU. The code and dataset are open-sourced at: https://github.com/Bean-Young/UltraGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09923v2">Splatent: Splatting Diffusion Latents for Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 CVPR 2026. Project's webpage at https://orhir.github.io/Splatent/
    </div>
    <details class="paper-abstract">
      Radiance field representations have recently been explored in the latent space of VAEs that are commonly used by diffusion models. This direction offers efficient rendering and seamless integration with diffusion-based pipelines. However, these methods face a fundamental limitation: The VAE latent space lacks multi-view consistency, leading to blurred textures and missing details during 3D reconstruction. Existing approaches attempt to address this by fine-tuning the VAE, at the cost of reconstruction quality, or by relying on pre-trained diffusion models to recover fine-grained details, at the risk of some hallucinations. We present Splatent, a diffusion-based enhancement framework designed to operate on top of 3D Gaussian Splatting (3DGS) in the latent space of VAEs. Our key insight departs from the conventional 3D-centric view: rather than reconstructing fine-grained details in 3D space, we recover them in 2D from input views through multi-view attention mechanisms. This approach preserves the reconstruction quality of pretrained VAEs while achieving faithful detail recovery. Evaluated across multiple benchmarks, Splatent establishes a new state-of-the-art for VAE latent radiance field reconstruction. We further demonstrate that integrating our method with existing feed-forward frameworks, consistently improves detail preservation, opening new possibilities for high-quality sparse-view 3D reconstruction. Code is available on our project page: https://orhir.github.io/Splatent/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04021v2">C3G: Learning Compact 3D Representations with 2K Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 Project Page : https://cvlab-kaist.github.io/C3G/
    </div>
    <details class="paper-abstract">
      Reconstructing and understanding 3D scenes from unposed sparse views in a feed-forward manner remains as a challenging task in 3D computer vision. Recent approaches use per-pixel 3D Gaussian Splatting for reconstruction, followed by a 2D-to-3D feature lifting stage for scene understanding. However, they generate excessive redundant Gaussians, causing high memory overhead and sub-optimal multi-view feature aggregation, leading to degraded novel view synthesis and scene understanding performance. We propose C3G, a novel feed-forward framework that estimates compact 3D Gaussians only at essential spatial locations, minimizing redundancy while enabling effective feature lifting. We introduce learnable tokens that aggregate multi-view features through self-attention to guide Gaussian generation, ensuring each Gaussian integrates relevant visual features across views. We then exploit the learned attention patterns for Gaussian decoding to efficiently lift features. Extensive experiments on pose-free novel view synthesis, 3D open-vocabulary segmentation, and view-invariant feature aggregation demonstrate our approach's effectiveness. Results show that a compact yet geometrically meaningful representation is sufficient for high-quality scene reconstruction and understanding, achieving superior memory efficiency and feature fidelity compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25466v1">Generalizable Human Gaussian Splatting via Multi-view Semantic Consistency</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 10 pages, 8 figures, CVPR 2026 Findings
    </div>
    <details class="paper-abstract">
      Recently, generalizable human Gaussian splatting from sparse-view inputs has been actively studied for the photorealistic human rendering. Most existing methods rely on explicit geometric constraints or predefined structural representations to accurately position 3D Gaussians. Although these approaches have shown the remarkable progress in this field, they still suffer from inconsistent feature representations across multi-view inputs due to complex articulations of the human body and limited overlaps between different views. To address this problem, we propose a novel method to accurately localize 3D Gaussians and ultimately improve the quality of human rendering. The key idea is to unproject latent embeddings encoded from each viewpoint into a shared 3D space through predicted depth maps and recalibrate them belonging to the same body part based on cross-view attention. This helps the model resolve the spatial ambiguity occurring in highly textured regions as well as occluded body parts, thus leading to the accurate localization of 3D Gaussians. Experimental results on benchmark datasets show that the proposed method efficiently improves the performance of generalizable human Gaussian splatting from sparse-view inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25459v1">GS-Playground: A High-Throughput Photorealistic Simulator for Vision-Informed Robot Learning</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 Robotics: Science and Systems 2026
    </div>
    <details class="paper-abstract">
      Embodied AI research is undergoing a shift toward vision-centric perceptual paradigms. While massively parallel simulators have catalyzed breakthroughs in proprioception-based locomotion, their potential remains largely untapped for vision-informed tasks due to the prohibitive computational overhead of large-scale photorealistic rendering. Furthermore, the creation of simulation-ready 3D assets heavily relies on labor-intensive manual modeling, while the significant sim-to-real physical gap hinders the transfer of contact-rich manipulation policies. To address these bottlenecks, we propose GS-Playground, a multi-modal simulation framework designed to accelerate end-to-end perceptual learning. We develop a novel high-performance parallel physics engine, specifically designed to integrate with a batch 3D Gaussian Splatting (3DGS) rendering pipeline to ensure high-fidelity synchronization. Our system achieves a breakthrough throughput of 10^4 FPS at 640x480 resolution, significantly lowering the barrier for large-scale visual RL. Additionally, we introduce an automated Real2Sim workflow that reconstructs photorealistic, physically consistent, and memory-efficient environments, streamlining the generation of complex simulation-ready scenes. Extensive experiments on locomotion, navigation, and manipulation demonstrate that GS-Playground effectively bridges the perceptual and physical gaps across diverse embodied tasks. Project homepage: https://gsplayground.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.22339v2">Flow4DGS-SLAM: Optical Flow-Guided 4D Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      Handling the dynamic environments is a significant research challenge in Visual Simultaneous Localization and Mapping (SLAM). Recent research combines 3D Gaussian Splatting (3DGS) with SLAM to achieve both robust camera pose estimation and photorealistic renderings. However, using SLAM to efficiently reconstruct both static and dynamic regions remains challenging. In this work, we propose an efficient framework for dynamic 3DGS SLAM guided by optical flow. Using the input depth and prior optical flow, we first propose a category-agnostic motion mask generation strategy by fitting a camera ego-motion model to decompose the optical flow. This module separates dynamic and static Gaussians and simultaneously provides flow-guided camera pose initialization. We boost the training speed of dynamic 3DGS by explicitly modeling their temporal centers at keyframes. These centers are propagated using 3D scene flow priors and are dynamically initialized with an adaptive insertion strategy. Alongside this, we model the temporal opacity and rotation using a Gaussian Mixture Model (GMM) to adaptively learn the complex dynamics. The empirical results demonstrate our state-of-the-art performance in tracking, dynamic reconstruction, and training efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19133v2">BALTIC: A Benchmark and Cross-Domain Strategy for 3D Reconstruction Across Air and Underwater Domains Under Varying Illumination</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      Robust 3D reconstruction across varying environmental conditions remains a critical challenge for robotic perception, particularly when transitioning between air and water. To address this, we introduce BALTIC, a controlled benchmark designed to systematically evaluate modern 3D reconstruction methods under variations in medium and lighting. The benchmark comprises 13 datasets spanning two media (air and water) and three lighting conditions (ambient, artificial, and mixed), with additional variations in motion type, scanning pattern, and initialization trajectory, resulting in a diverse set of sequences. Our experimental setup features a custom water tank equipped with a monocular camera and an HTC Vive tracker, enabling accurate ground-truth pose estimation. We further investigate cross-domain reconstruction by augmenting underwater image sequences with a small number of in-air views captured under similar lighting conditions. We evaluate Structure-from-Motion reconstruction using COLMAP in terms of both trajectory accuracy and scene geometry, and use these reconstructions as input to Neural Radiance Fields and 3D Gaussian Splatting methods. The resulting models are assessed against ground-truth trajectories and in-air references, while rendered outputs are compared using perceptual and photometric metrics. Additionally, we perform a color restoration analysis to evaluate radiometric consistency across domains. Our results show that under controlled, texture-consistent conditions, Gaussian Splatting with simple preprocessing (e.g., white balance correction) can achieve performance comparable to specialized underwater methods, although its robustness decreases in more complex and heterogeneous real-world environments
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07105v3">Genie Sim PanoRecon: Fast Immersive Scene Generation from Single-View Panorama</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      We present Genie Sim PanoRecon, a feed-forward Gaussian-splatting pipeline that delivers high-fidelity, low-cost 3D scenes for robotic manipulation simulation. The panorama input is decomposed into six non-overlapping cube-map faces, processed in parallel, and seamlessly reassembled. To guarantee geometric consistency across views, we devise a depth-aware fusion strategy coupled with a training-free depth-injection module that steers the monocular feed-forward network to generate coherent 3D Gaussians. The whole system reconstructs photo-realistic scenes in seconds and has been integrated into Genie Sim - a LLM-driven simulation platform for embodied synthetic data generation and evaluation - to provide scalable backgrounds for manipulation tasks. For code details, please refer to: https://github.com/AgibotTech/genie_sim/tree/main/source/geniesim_world.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22793v3">GSpaRC: Gaussian Splatting for Real-time Reconstruction of RF Channels</a></div>
    <div class="paper-meta">
      📅 2026-04-27
      | 💬 Project website: https://nbhavyasai.github.io/GSpaRC/
    </div>
    <details class="paper-abstract">
      Channel state information (CSI) is essential for adaptive beamforming and maintaining robust links in wireless communication systems. However, acquiring CSI incurs significant overhead, consuming up to 25% of spectrum resources in 5G networks due to frequent pilot transmissions at millisecond-scale intervals. Recent approaches aim to reduce this burden by reconstructing CSI from spatiotemporal RF measurements, such as signal strength and direction-of-arrival. While effective in offline settings, these methods often suffer from inference latencies in the 5-100 ms range, making them impractical for real-time systems. We present GSpaRC: Gaussian Splatting for Real-time Reconstruction of RF Channels, a method that achieves accurate channel reconstruction with latency in the low-millisecond regime or below. GSpaRC represents the RF environment using a compact set of 3D Gaussian primitives, each parameterized by a lightweight neural model augmented with physics-informed features such as distance-based attenuation. Unlike traditional vision-based splatting pipelines, GSpaRC is tailored for RF reception: it employs an equirectangular projection onto a hemispherical surface centered at the receiver to reflect omnidirectional antenna behavior. A custom CUDA pipeline enables fully parallelized directional sorting, splatting, and rendering across frequency and spatial dimensions. Evaluated on multiple RF datasets, GSpaRC achieves similar CSI reconstruction fidelity to recent state-of-the-art methods while reducing training and inference time by over an order of magnitude. These results illustrate that modest GPU computation can substantially reduce pilot overhead, making GSpaRC a scalable low-latency approach for channel estimation in 5G and future wireless systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24316v1">Large-Scale Photogrammetric Documentation of St. John's Co-Cathedral: A Workflow for Cultural Heritage Preservation</a></div>
    <div class="paper-meta">
      📅 2026-04-27
    </div>
    <details class="paper-abstract">
      We present a comprehensive methodology for the large-scale photogrammetric documentation of St. John's Co-Cathedral in Valletta, Malta, a UNESCO World Heritage site renowned for its ornate Baroque architecture and Caravaggio masterpieces. Over seven nights of evening-only data collection, we captured 99,000 images using DSLR cameras, drone photography, and LIDAR scanning to create a highly detailed 3D reconstruction comprising 25-30 billion triangles. This paper documents our complete workflow for cultural heritage preservation, addressing the unique challenges of digitizing complex baroque architectural spaces with highly reflective metallic surfaces, dark materials, intricate tapestries, and restricted access. We detail our pipeline from multi-modal data acquisition through processing, including strategic image grading and AI-assisted denoising to address low-light grain, extensive LIDAR point cloud cleanup, hybrid photogrammetric reconstruction using RealityCapture, and mesh subdivision strategies for real-time visualization engines. Our methodology combines automated workflows with necessary manual intervention to handle the scale and complexity of the project, with particular attention to reflective surface challenges characteristic of baroque heritage sites. We also present preliminary experiments with Gaussian splatting as a complementary representation technique. The resulting digital archive serves multiple preservation purposes including disaster recovery documentation, conservation analysis, virtual tourism, and scholarly research. This work provides a detailed, replicable workflow for heritage professionals undertaking similar large-scale architectural documentation projects, addressing the practical challenges of applying photogrammetric methods in complex real-world heritage scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24053v1">Light 'em Up: Enabling Few-Shot Low-Light 3D Gaussian Splatting with Multi-Scale Explicit Retinex Illumination Decoupling</a></div>
    <div class="paper-meta">
      📅 2026-04-27
      | 💬 19 pages, 5 figures. Code available at https://github.com/YhuoyuH/MERID-GS
    </div>
    <details class="paper-abstract">
      Full 360$^\circ$ novel view synthesis under low-light conditions remains challenging. Insufficient illumination, noise amplification, and view-dependent photometric inconsistencies prevent existing methods from jointly preserving geometric consistency and photorealism. Unsupervised approaches often exhibit color drift under large viewpoint variations, while supervised low-light enhancement models, though effective for 2D tasks, struggle to generalize to new scenes and typically require retraining. To address this issue, we propose MERID-GS, a Multi-Scale Explicit Retinex Illumination-Decoupled Gaussian framework for low-light 360$^\circ$ synthesis. Based on Retinex theory, the method explicitly separates illumination and reflectance, and suppresses noise propagation while enhancing dark-region structures via a learnable gain and Illumination-State-Guided Frequency Gating. Combined with lightweight Reflection Head and 3D Gaussian Splatting, MERID-GS adapts to new scenes with only a few shots and enables stable low-light novel view synthesis from sparse-view observations. In addition, we construct a low-light multi-view dataset covering full 360$^\circ$ scenes for joint evaluation. Thorough experiments across multiple datasets in this area demonstrate that MERID-GS achieves SOTA performance, exhibiting superior cross-scene generalization and view consistency. The source code and pre-trained models are available at https://github.com/YhuoyuH/MERID-GS..
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.20066v2">Learning Scene-Level Signed Directional Distance Function with Ellipsoidal Priors and Neural Residuals</a></div>
    <div class="paper-meta">
      📅 2026-04-27
    </div>
    <details class="paper-abstract">
      Dense reconstruction and differentiable rendering are fundamental tightly connected operations in 3D vision and computer graphics. Recent neural implicit representations demonstrate compelling advantages in reconstruction fidelity and differentiability over conventional discrete representations such as meshes, point clouds, and voxels. However, many neural implicit models, such as neural radiance fields (NeRF) and signed distance function (SDF) networks, are inefficient in rendering due to the need to perform multiple queries along each camera ray. Moreover, NeRF and Gaussian Splatting methods offer impressive photometric reconstruction but often require careful supervision to achieve accurate geometric reconstruction. To address these challenges, we propose a novel representation called signed directional distance function (SDDF). Unlike SDF and similar to NeRF, SDDF has a position and viewing direction as input. Like SDF and unlike NeRF, SDDF directly provides distance to the observed surface rather than integrating along the view ray. As a result, SDDF achieves accurate geometric reconstruction and efficient differentiable directional distance prediction. To learn and predict scene-level SDDF efficiently, we develop a differentiable hybrid representation that combines explicit ellipsoid priors and implicit neural residuals. This allows the model to handle distance discontinuities around obstacle boundaries effectively while preserving the ability for dense high-fidelity distance prediction. Through extensive evaluation against state-of-the-art representations, we show that SDDF achieves (i) competitive SDDF prediction accuracy, (ii) faster prediction speed than SDF and NeRF, and (iii) superior geometric consistency compared to NeRF and Gaussian Splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17092v4">SPAGS: Sparse-View Articulated Object Reconstruction from Single State via Planar Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-27
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Articulated objects are ubiquitous in daily environments, and their 3D reconstruction holds great significance across various fields. However, existing articulated object reconstruction methods typically require costly inputs such as multi-stage and multi-view observations. To address the limitations, we propose a category-agnostic articulated object reconstruction framework via planar Gaussian Splatting, which only uses sparse-view RGB images from a single state. Specifically, we first introduce a Gaussian information field to perceive the optimal sparse viewpoints from candidate camera poses. To ensure precise geometric fidelity, we constrain traditional 3D Gaussians into planar primitives, facilitating accurate normal and depth estimation. The planar Gaussians are then optimized in a coarse-to-fine manner, regularized by depth smoothness and few-shot diffusion priors. Furthermore, we leverage a Vision-Language Model (VLM) via visual prompting to achieve open-vocabulary part segmentation and joint parameter estimation. Extensive experiments on both synthetic and real-world datasets demonstrate that our approach significantly outperforms existing baselines, achieving superior part-level surface reconstruction fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.13713v6">SLAM&Render: A Benchmark for the Intersection Between Neural Rendering, Gaussian Splatting and SLAM</a></div>
    <div class="paper-meta">
      📅 2026-04-26
      | 💬 9 pages, 8 figures, 7 tables. Submitted to IROS 2026
    </div>
    <details class="paper-abstract">
      Models and methods originally developed for Novel View Synthesis and Scene Rendering, such as Neural Radiance Fields (NeRF) and Gaussian Splatting, are increasingly being adopted as representations in Simultaneous Localization and Mapping (SLAM). However, existing datasets fail to include the specific challenges of both fields, such as sequential operations and, in many settings, multi-modality in SLAM or generalization across viewpoints and illumination conditions in neural rendering. Additionally, the data are often collected using sensors which are handheld or mounted on drones or mobile robots, which complicates the accurate reproduction of sensor motions. To bridge these gaps, we introduce SLAM&Render, a novel dataset designed to benchmark methods in the intersection between SLAM, Novel View Rendering and Gaussian Splatting. Recorded with a robot manipulator, it uniquely includes 40 sequences with time-synchronized RGB-D images, IMU readings, robot kinematic data, and ground-truth pose streams. By releasing robot kinematic data, the dataset also enables the assessment of recent integrations of SLAM paradigms within robotic applications. The dataset features five setups with consumer and industrial objects under four controlled lighting conditions, each with separate training and test trajectories. All sequences are static with different levels of object rearrangements and occlusions. Our experimental results, obtained with several baselines from the literature, validate SLAM&Render as a relevant benchmark for this emerging research area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23803v1">Bringing a Personal Point of View: Evaluating Dynamic 3D Gaussian Splatting for Egocentric Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-26
      | 💬 Accepted at the EgoVis Workshop at CVPR 2026
    </div>
    <details class="paper-abstract">
      Egocentric video provides a unique view into human perception and interaction, with growing relevance for augmented reality, robotics, and assistive technologies. However, rapid camera motion and complex scene dynamics pose major challenges for 3D reconstruction from this perspective. While 3D Gaussian Splatting (3DGS) has become a state-of-the-art method for efficient, high-quality novel view synthesis, variants, that focus on reconstructing dynamic scenes from monocular video are rarely evaluated on egocentric video. It remains unclear whether existing models generalize to this setting or if egocentric-specific solutions are needed. In this work, we evaluate dynamic monocular 3DGS models on egocentric and exocentric video using paired ego-exo recordings from the EgoExo4D dataset. We find that reconstruction quality is consistently lower in egocentric views. Analysis reveals that the difference in reconstruction quality, measured in peak signal-to-noise ratio, stems from the reconstruction of static, not dynamic, content. Our findings underscore current limitations and motivate the development of egocentric-specific approaches, while also highlighting the value of separately evaluating static and dynamic regions of a video.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07940v2">ISExplore:Informative Segment Selection for Efficient Personalized 3D Talking Face Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-26
    </div>
    <details class="paper-abstract">
      Talking Face Generation (TFG) methods based on Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have recently achieved impressive progress in personalized talking head synthesis. However, existing methods typically require several minutes of reference video for meticulous preprocessing and fitting, resulting in hours of preparation time and limiting their practical applicability. In this paper, we revisit a fundamental yet underexplored question: do high-quality personalized TFG models truly require minutes-long reference videos? Our exploratory study reveals that a carefully selected reference segment of only a few seconds can often achieve performance comparable to that of using the full reference video. This finding suggests that the informativeness of reference data is more critical than its duration. Motivated by this observation, we propose ISExplore (Informative Segment Explore), a simple yet effective segment selection strategy that automatically identifies the most informative short reference segment based on three key data quality dimensions: audio feature diversity, lip movement amplitude, and viewpoint diversity. Extensive experiments demonstrate that ISExplore reduces data processing and training time by over 5x for both NeRF- and 3DGS-based methods, while preserving high-fidelity generation quality. Our method provides a practical and efficient solution for personalized TFG and offers new insights into data efficiency in 3D talking face generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23675v1">GS-DOT: Gaussian splatting-based image reconstruction for diffuse optical tomography</a></div>
    <div class="paper-meta">
      📅 2026-04-26
    </div>
    <details class="paper-abstract">
      This work presents GS-DOT, a novel image reconstruction framework based on Gaussian Splatting (GS) for diffuse optical tomography (DOT). Inspired by GS for rendering applications, absorption coefficients are represented as a sparse sum of anisotropic Gaussian primitives optimized to fit measured time-resolved point-spread functions through analytic gradients and Adam optimization. This is the first adaptation of GS algorithms in the photon diffusion regime, where the ray transport function is replaced by the diffusion functions to enable accurate modeling of light transport in highly scattering media. Validation on synthetic tissue models demonstrate high accuracy in localization and quantification of reconstructed absorption maps for both clean and noisy signals. GS-DOT has demonstrated high robustness to noise and showed a huge reduction in memory demand.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23551v1">Spatiotemporal Degradation-Aware 3D Gaussian Splatting for Realistic Underwater Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-26
      | 💬 12 pages, 10 figures, 6 tables. Author version of the paper published in Proceedings of ACM Multimedia 2025
    </div>
    <details class="paper-abstract">
      Reconstructing realistic underwater scenes from underwater video remains a meaningful yet challenging task in the multimedia domain. The inherent spatiotemporal degradations in underwater imaging, including caustics, flickering, attenuation, and backscattering, frequently result in inaccurate geometry and appearance in existing 3D reconstruction methods. While a few recent works have explored underwater degradation-aware reconstruction, they often address either spatial or temporal degradation alone, falling short in more real-world underwater scenarios where both types of degradation occur. We propose MarineSTD-GS, a novel 3D Gaussian Splatting-based framework that explicitly models both temporal and spatial degradations for realistic underwater scene reconstruction. Specifically, we introduce two paired Gaussian primitives: Intrinsic Gaussians represent the true scene, while Degraded Gaussians render the degraded observations. The color of each Degraded Gaussian is physically derived from its paired Intrinsic Gaussian via a Spatiotemporal Degradation Modeling (SDM) module, enabling self-supervised disentanglement of realistic appearance from degraded images. To ensure stable training and accurate geometry, we further propose a Depth-Guided Geometry Loss and a Multi-Stage Optimization strategy. We also construct a simulated benchmark with diverse spatial and temporal degradations and ground-truth appearances for comprehensive evaluation. Experiments on both simulated and real-world datasets show that MarineSTD-GS robustly handles spatiotemporal degradations and outperforms existing methods in novel view synthesis with realistic, water-free scene appearances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05480v4">ODE-GS: Latent ODEs for Dynamic Scene Extrapolation with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-26
      | 💬 Accepted at ICLR 2026
    </div>
    <details class="paper-abstract">
      We introduce ODE-GS, a novel approach that integrates 3D Gaussian Splatting with latent neural ordinary differential equations (ODEs) to enable future extrapolation of dynamic 3D scenes. Unlike existing dynamic scene reconstruction methods, which rely on time-conditioned deformation networks and are limited to interpolation within a fixed time window, ODE-GS eliminates timestamp dependency by modeling Gaussian parameter trajectories as continuous-time latent dynamics. Our approach first learns an interpolation model to generate accurate Gaussian trajectories within the observed window, then trains a Transformer encoder to aggregate past trajectories into a latent state evolved via a neural ODE. Finally, numerical integration produces smooth, physically plausible future Gaussian trajectories, enabling rendering at arbitrary future timestamps. On the D-NeRF, NVFi, and HyperNeRF benchmarks, ODE-GS achieves state-of-the-art extrapolation performance, improving metrics by 19.8% compared to leading baselines, demonstrating its ability to accurately represent and predict 3D scene dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19843v2">Graphical X Splatting (GraphiXS): A Graphical Model for 4D Gaussian Splatting under Uncertainty</a></div>
    <div class="paper-meta">
      📅 2026-04-25
    </div>
    <details class="paper-abstract">
      We propose a new framework to systematically incorporate data uncertainty in Gaussian Splatting. Being the new paradigm of neural rendering, Gaussian Splatting has been investigated in many applications, with the main effort in extending its representation, improving its optimization process, and accelerating its speed. However, one orthogonal, much needed, but under-explored area is data uncertainty. In standard 4D Gaussian Splatting, data uncertainty can manifest as view sparsity, missing frames, camera asynchronization, etc. So far, there has been little research to holistically incorporating various types of data uncertainty under a single framework. To this end, we propose Graphical X Splatting, or GraphiXS, a new probabilistic framework that considers multiple types of data uncertainty, aiming for a fundamental augmentation of the current 4D Gaussian Splatting paradigm into a probabilistic setting. GraphiXS is general and can be instantiated with a range of primitives, e.g. Gaussians, Student's-t. Furthermore, GraphiXS can be used to `upgrade' existing methods to accommodate data uncertainty. Through exhaustive evaluation and comparison, we demonstrate that GraphiXS can systematically model various uncertainties in data, outperform existing methods in many settings where data are missing or polluted in space and time, and therefore is a major generalization of the current 4D Gaussian Splatting research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.02316v4">ConsDreamer: Advancing Multi-View Consistency for Zero-Shot Text-to-3D Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-25
      | 💬 Accepted by IEEE Transactions on Image Processing, Code is available at: https://github.com/GAInuist/ConsDreamer
    </div>
    <details class="paper-abstract">
      Recent advances in zero-shot text-to-3D generation have revolutionized 3D content creation by enabling direct synthesis from textual descriptions. While state-of-the-art methods leverage 3D Gaussian Splatting with score distillation to enhance multi-view rendering through pre-trained text-to-image (T2I) models, they suffer from inherent prior view biases in T2I priors. These biases lead to inconsistent 3D generation, particularly manifesting as the multi-face Janus problem, where objects exhibit conflicting features across views. To address this fundamental challenge, we propose ConsDreamer, a novel method that mitigates view bias by refining both the conditional and unconditional terms in the score distillation process: (1) a View Disentanglement Module (VDM) that eliminates viewpoint biases in conditional prompts by decoupling irrelevant view components and injecting precise view control; and (2) a similarity-based partial order loss that enforces geometric consistency in the unconditional term by aligning cosine similarities with azimuth relationships. Extensive experiments demonstrate that ConsDreamer can be seamlessly integrated into various 3D representations and score distillation paradigms, effectively mitigating the multi-face Janus problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.22439v1">NRGS: Neural Regularization for Robust 3D Semantic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-24
    </div>
    <details class="paper-abstract">
      We propose a neural regularization method that refines the noisy 3D semantic field produced by lifting multi-view inconsistent 2D features, in order to obtain an accurate and robust 3D semantic Gaussian Splatting. The 2D features extracted from vision foundation models suffer from multi-view inconsistency due to a lack of cross-view constraints. Lifting these inconsistent features directly into 3D Gaussians results in a noisy semantic field, which degrades the performance of downstream tasks. Previous methods either focus on obtaining consistent multi-view features in the preprocessing stage or aim to mitigate noise through improved optimization strategies, often at the cost of increased preprocessing time or expensive computational overhead. In contrast, we introduce a variance-aware conditional MLP that operates directly on the 3D Gaussians, leveraging their geometric and appearance attributes to correct semantic errors in 3D space. Experiments on different datasets show that our method enhances the accuracy of lifted semantics, providing an efficient and effective approach to robust 3D semantic Gaussian Splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.22183v1">EvFlow-GS: Event Enhanced Motion Deblurring with Optical Flow for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-24
      | 💬 Accepted by ICME 2026
    </div>
    <details class="paper-abstract">
      Achieving sharp 3D reconstruction from motion-blurred images alone becomes challenging, motivating recent methods to incorporate event cameras, benefiting from microsecond temporal resolution. However, they suffer from residual artifacts and blurry texture details due to misleading supervision from inaccurate event double integral priors and noisy, blurry events. In this study, we propose EvFlow-GS, a unified framework that leverages event streams and optical flow to optimize an end-to-end learnable double integral (LDI), camera poses, and 3D Gaussian Splatting (3DGS) jointly on-the-fly. Specifically, we first extract edge information from the events using optical flow and then formulate a novel event-based loss applied separately to different modules. Additionally, we exploit a novel event-residual prior to strengthen the supervision of intensity changes between images rendered from 3DGS. Finally, we integrate the outputs of both 3DGS and LDI into a joint loss, enabling their optimization to mutually facilitate each other. Experiments demonstrate the leading performance of our EvFlow-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21400v2">You Only Gaussian Once: Controllable 3D Gaussian Splatting for Ultra-Densely Sampled Scenes</a></div>
    <div class="paper-meta">
      📅 2026-04-24
      | 💬 17 pages, 5 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has revolutionized neural rendering, yet existing methods remain predominantly research prototypes ill-suited for production-level deployment. We identify a critical "Industry-Academia Gap" hindering real-world application: unpredictable resource consumption from heuristic Gaussian growth, the "sparsity shield" of current benchmarks that rewards hallucination over physical fidelity, and severe multi-sensor data pollution. To bridge this gap, we propose YOGO (You Only Gaussian Once), a system-level framework that reformulates the stochastic growth process into a deterministic, budget-aware equilibrium. YOGO integrates a novel budget controller for hardware-constrained resource allocation and an availability-registration protocol for robust multi-sensor fusion. To push the boundaries of reconstruction fidelity, we introduce Immersion v1.0, the first ultra-dense indoor dataset specifically designed to break the "sparsity shield." By providing saturated viewpoint coverage, Immersion v1.0 forces algorithms to focus on extreme physical fidelity rather than viewpoint interpolation, and enables the community to focus on the upper limits of high-fidelity reconstruction. Extensive experiments demonstrate that YOGO achieves state-of-the-art visual quality while maintaining a strictly deterministic profile, establishing a new standard for production-grade 3DGS. To facilitate reproducibility, part scenes of Immersion v1.0 dataset and source code of YOGO has been publicly released. The project link is https://jjrcn.github.io/yogo-project-home/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.22129v1">PAGaS: Pixel-Aligned 1DoF Gaussian Splatting for Depth Refinement</a></div>
    <div class="paper-meta">
      📅 2026-04-24
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has emerged as an efficient approach for high-quality novel view synthesis. While early GS variants struggled to accurately model the scene's geometry, recent advancements constraining the Gaussians' spread and shapes, such as 2D Gaussian Splatting, have significantly improved geometric fidelity. In this paper, we present Pixel-Aligned 1DoF Gaussian Splatting (PAGaS) that adapts the GS representation from novel view synthesis to the multi-view stereo depth task. Our key contribution is modeling a pixel's depth using one-degree-of-freedom (1DoF) Gaussians that remain tightly constrained during optimization. Unlike existing approaches, our Gaussians' positions and sizes are restricted by the back-projected pixel volumes, leaving depth as the sole degree of freedom to optimize. PAGaS produces highly detailed depths, as illustrated in Figure 1. We quantitatively validate these improvements on top of reference geometric and learning-based multi-view stereo baselines on challenging 3D reconstruction benchmarks. Code: davidrecasens.github.io/pagas
    </details>
</div>
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
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20046v2">Gaussians on a Diet: High-Quality Memory-Bounded 3D Gaussian Splatting Training</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has revolutionized novel view synthesis with high-quality rendering through continuous aggregations of millions of 3D Gaussian primitives. However, it suffers from a substantial memory footprint, particularly during training due to uncontrolled densification, posing a critical bottleneck for deployment on memory-constrained edge devices. While existing methods prune redundant Gaussians post-training, they fail to address the peak memory spikes caused by the abrupt growth of Gaussians early in the training process. To solve the training memory consumption problem, we propose a systematic memory-bounded training framework that dynamically optimizes Gaussians through iterative growth and pruning. In other words, the proposed framework alternates between incremental pruning of low-impact Gaussians and strategic growing of new primitives with an adaptive Gaussian compensation, maintaining a near-constant low memory usage while progressively refining rendering fidelity. We comprehensively evaluate the proposed training framework on various real-world datasets under strict memory constraints, showing significant improvements over existing state-of-the-art methods. Particularly, our proposed method practically enables memory-efficient 3DGS training on NVIDIA Jetson AGX Xavier, achieving similar visual quality with up to 80% lower peak training memory consumption than the original 3DGS.
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
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25945v1">Planar Gaussian Splatting with Bilinear Spatial Transformer for Wireless Radiance Field Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-17
      | 💬 Accepted to IEEE ICC 2026 Workshop
    </div>
    <details class="paper-abstract">
      Wireless radiance field (WRF) reconstruction aims to learn a continuous, queryable representation of radio frequency characteristics over 3D space and direction, from which specific quantities, such as the spatial power spectrum (SPS) at a receiver given a transmitter position, can be predicted. While Gaussian splatting (GS)-based method has surpassed Neural Radiance Fields (NeRF)-based method for this task, existing adaptations largely transplant vision pipelines, limiting physical interpretability and accuracy. We introduce BiSplat-WRF, a planar GS framework that retains the expressiveness of 3D GS while removing unnecessary projections and incorporating global EM coupling and mutual scattering among primitives. Each primitive is a 2D planar Gaussian with 3D coordinates, rendered directly on the angular domain of the SPS. A bilinear spatial transformer (BST) aggregates inter-primitive relations on an angular grid and, via attention, captures long-range electromagnetic dependencies, thereby enforcing globally aware EM interactions that reflect the complex physics of the wireless environment. On spatial spectrum synthesis task, BiSplat-WRF surpasses NeRF-based and prior GS-based baselines with respect to the Structural Similarity Index (SSIM); comprehensive ablation studies validate the contribution of BST. We also provide a larger BiSplat-WRF+ variant that further increases SSIM at a higher computation cost, serving as a strong reference for future studies.
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
