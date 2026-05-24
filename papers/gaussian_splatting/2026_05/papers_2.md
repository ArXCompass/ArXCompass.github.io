# gaussian splatting - 2026_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.08699v1">Thin-Client Interactive Gaussian Adaptive Streaming over HTTP/3</a></div>
    <div class="paper-meta">
      📅 2026-05-09
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3DGS) have enabled photorealistic rendering of complex scenes, yet widespread adoption on mobile and Extended Reality (XR) devices is hindered by substantial computational and bandwidth requirements. While existing solutions often focus on model compression for client-side rendering, they still demand significant GPU power, limiting applicability on resource-constrained hardware. We propose TIGAS (Thin-client Interactive Gaussian Adaptive Streaming), a remote rendering framework offloading rasterization to a backend. To bypass the prohibitive latencies connected to fluctuating network conditions, TIGAS streams view-dependent 2D projections to a lightweight web client over QUIC, minimizing head-of-line (HoL) blocking. A dedicated ABR algorithm adapts rendering quality to fluctuating network conditions, maintaining motion-to-photon latency within strict 6DoF interactive constraints. Furthermore, we discuss the integration of an experimental WebGPU super-resolution pipeline to analyze the trade-offs between perceptual quality enhancements and thin-client processing bottlenecks. We extensively evaluate TIGAS across multi-continental environments using 14 3DGS models and real 6DoF EyeNavGS movement traces. Powered by a backend rendering frames in under 10 milliseconds, TIGAS maintains latency within interactive thresholds while achieving an average SSIM of 0.88, serving both as a robust testbed for 3DGS streaming research and a capable delivery system. The source code is available at: https://github.com/Rekenar/GaussianAdaptiveStreamer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16355v1">Generative 3D Gaussians with Learned Density Control</a></div>
    <div class="paper-meta">
      📅 2026-05-08
      | 💬 19 pages, 16 figures, SIGGRAPH Conference Papers '26
    </div>
    <details class="paper-abstract">
      We present Density-Sampled Gaussians (DeG), a novel 3D representation designed to bridge the gap between adaptive rendering primitives and scalable generative modeling. Unlike existing approaches that constrain 3D Gaussians to fixed voxel grids or arrays, DeG models Gaussian centers as samples from a learnable probability density function defined over an octree. This formulation provides a rigorous mathematical framework for adaptive density control: by jointly optimizing the spatial density and Gaussian attributes under rendering supervision, our model naturally concentrates primitives in regions of high geometric complexity. We achieve this via a new render loss contribution gradient that serves as a fully differentiable analogue to the discrete densification and pruning heuristics used in standard Gaussian Splatting. The resulting representation is highly flexible, supporting variable-resolution decoding from a single latent code by simply adjusting the sampling budget. To enable generative synthesis, we train a latent diffusion model on DeG. We identify a critical challenge in applying diffusion to unordered set-structured latents, which can significantly slow convergence, and propose VecSeq, a canonical re-indexing mechanism that anchors latent tokens to a deterministic 3D Sobol sequence. This transforms the ambiguous set-generation problem into a robust sequence modeling task. Extensive experiments demonstrate that our pipeline achieves state-of-the-art quality in single-image-to-3D generation, combining the structural adaptivity of unstructured primitives with the training stability of grid-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07781v1">Differentiable Ray Tracing with Gaussians for Unified Radio Propagation Simulation and View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-05-08
    </div>
    <details class="paper-abstract">
      Explicit neural representations such as 3D Gaussian Splatting (3DGS) enable high-fidelity and real-time novel view synthesis, yet optimize for alpha-composited optical appearance rather than ray-intersectable geometry. In contrast, radio-frequency (RF) digital twins require deterministic multi-bounce paths, where the geometry dictates trajectories and their associated attenuation and delay. We introduce a framework enabling differentiable RF propagation simulation directly within visually reconstructed neural scenes, allowing point-to-point path computation between arbitrary 3D locations while preserving high-quality visual rendering. Unlike conventional RF simulation pipelines that rely on manually constructed meshes, we embed Gaussian primitives into a hardware-accelerated ray tracing structure as the underlying spatial representation. By extracting physically meaningful channel impulse responses from visual-only reconstructions, we provide cross-modal evidence that neural reconstructions can serve as unified spatial representations for both electromagnetic propagation simulation and photorealistic view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02129v2">VDEGaussian: Video Diffusion Enhanced 4D Gaussian Splatting for Dynamic Urban Scenes Modeling</a></div>
    <div class="paper-meta">
      📅 2026-05-08
    </div>
    <details class="paper-abstract">
      Dynamic urban scene modeling is a rapidly evolving area with broad applications. While current approaches leveraging neural radiance fields or Gaussian Splatting have achieved fine-grained reconstruction and high-fidelity novel view synthesis, they still face significant limitations. These often stem from a dependence on pre-calibrated object tracks or difficulties in accurately modeling fast-moving objects from undersampled capture, particularly due to challenges in handling temporal discontinuities. To overcome these issues, we propose a novel video diffusion-enhanced 4D Gaussian Splatting framework. Our key insight is to distill robust, temporally consistent priors from a test-time adapted video diffusion model. To ensure precise pose alignment and effective integration of this denoised content, we introduce two core innovations: a joint timestamp optimization strategy that refines interpolated frame poses, and an uncertainty distillation method that adaptively extracts target content while preserving well-reconstructed regions. Extensive experiments demonstrate that our method significantly enhances dynamic modeling, especially for fast-moving objects, achieving an approximate PSNR gain of 2 dB for novel view synthesis over baseline approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12647v2">LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-08
      | 💬 8 pages, 7 figures, conference
    </div>
    <details class="paper-abstract">
      Recent 3D Gaussian Splatting (3DGS) methods have demonstrated the feasibility of self-driving scene reconstruction and novel view synthesis. However, most existing methods either rely solely on cameras or use LiDAR only for Gaussian initialization or depth supervision, while the rich scene information contained in point clouds, such as reflectance, and the complementarity between LiDAR and RGB have not been fully exploited, leading to degradation in challenging self-driving scenes, such as those with high ego-motion and complex lighting. To address these issues, we propose a robust and efficient LiDAR-reflectance-guided Salient Gaussian Splatting method (LR-SGS) for self-driving scenes, which introduces a structure-aware Salient Gaussian representation, initialized from geometric and reflectance feature points extracted from LiDAR and refined through a salient transform and improved density control to capture edge and planar structures. Furthermore, we calibrate LiDAR intensity into reflectance and attach it to each Gaussian as a lighting-invariant material channel, jointly aligned with RGB to enforce boundary consistency. Extensive experiments on the Waymo Open Dataset demonstrate that LR-SGS achieves superior reconstruction performance with fewer Gaussians and shorter training time. In particular, on Complex Lighting scenes, our method surpasses OmniRe by 1.18 dB PSNR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01195v2">TAIL-Safe: Task-Agnostic Safety Monitoring for Imitation Learning Policies</a></div>
    <div class="paper-meta">
      📅 2026-05-08
    </div>
    <details class="paper-abstract">
      Recent imitation learning (IL) algorithms such as flow-matching and diffusion policies demonstrate remarkable performance in learning complex manipulation tasks. However, these policies often fail even when operating within their training distribution due to extreme sensitivity to initial conditions and irreducible approximation errors that lead to compounding drift. This makes it unsafe to deploy IL policies in the field where out-of-distribution scenarios are prevalent. A prerequisite for safe deployment is enabling the policy to determine whether it can execute a task the way it was learned from demonstrations. This paper presents TAIL-Safe, a principled approach to identify, for a trained IL policy, a safe set from where the policy empirically succeeds in completing the learned task. We propose a Lipschitz-continuous Q-value function that maps state-action pairs to a long-term safety score based on three short-term task-agnostic criteria: visibility, recognizability, and graspability. The zero-superlevel set of this function characterizes an empirical control invariant set over state-action pairs. When the nominal policy proposes an action outside this set, we apply a recovery mechanism inspired by Nagumo's theorem that uses gradient ascent to the Q-function to steer the policy back to safety. To learn this Q-function, we construct a high-fidelity digital twin using Gaussian Splatting that enables systematic collection of failure data without risk to physical hardware. Experiments with a Franka Emika robot demonstrate that flow-matching policies, which fail under run-time perturbations, achieve consistent task success when guided by the proposed TAIL-Safe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07351v1">Disambiguating 2D-3D Correspondences in Gaussian Splatting-based Feature Fields for Visual Localization</a></div>
    <div class="paper-meta">
      📅 2026-05-08
    </div>
    <details class="paper-abstract">
      While Gaussian Splatting-based Feature Fields (GSFFs) have shown promise for visual localization, this paper highlights that photometrically optimized GSFFs are inherently ill-suited for 2D-3D matching. The volumetric extent of each Gaussian induces many-to-one pixel-to-point mappings that destabilize PnP-based pose estimation, while photometric optimization gives rise to superfluous Gaussians devoid of multi-view consistency. To address these issues, we propose SplitGS-Loc, a localization-specialized GSFFs construction framework that disambiguates 2D-3D correspondences by exploiting Gaussian attributes. Our key design, Mixture-of-Gaussians-based splitting, decomposes each Gaussian into smaller Gaussians, replacing ambiguous many-to-one with precise one-to-one correspondences. In parallel, we exploit composition weights from GS rasterization to select Gaussians that significantly and consistently contribute across multiple views and aggregate discriminative features through strong pixel-Gaussian associations, enforcing multi-view consistency. The resulting compact yet discriminative feature fields enable stable PnP convergence, achieving state-of-the-art performance on localization benchmarks. Extensive experiments validate that SplitGS-Loc extends the utility of photometric GSFFs to accurate and efficient localization by exploiting Gaussian attributes, without per-scene training or iterative pose refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07254v1">High-Fidelity Surface Splatting-Based 3D Reconstruction from Multi-View Images</a></div>
    <div class="paper-meta">
      📅 2026-05-08
      | 💬 19 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Multi-view mesh reconstruction remains a core challenge in computer graphics and vision, especially for recovering high-frequency geometry from sparse observations. Recent methods such as 3D Gaussian Splatting (3DGS) and Neural Radiance Fields (NeRF) rely on post-processing for mesh extraction, thereby limiting joint optimization of geometry and appearance. Implicit Moving Least Squares (IMLS) instead enables direct conversion of point clouds into signed distance and texture fields, supporting end-to-end reconstruction and rendering. However, existing IMLS formulations use exponential kernels that struggle with high-frequency detail. We introduce a compact polynomial kernel with local support and greater flexibility, allowing better control over frequency content and improved geometric fidelity. To further enhance fine details, we incorporate stochastic regularization with Laplacian filtering. Together, these improve the preservation of high-frequency structure while maintaining stable optimization. Experiments show state-of-the-art performance in both surface reconstruction and rendering, yielding more accurate geometry and sharper visuals from multi-view data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16736v5">A Step to Decouple Optimization in 3DGS</a></div>
    <div class="paper-meta">
      📅 2026-05-08
      | 💬 Accepted by ICLR 2026 (fixed typo)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for real-time novel view synthesis. As an explicit representation optimized through gradient propagation among primitives, optimization widely accepted in deep neural networks (DNNs) is actually adopted in 3DGS, such as synchronous weight updating and Adam with the adaptive gradient. However, considering the physical significance and specific design in 3DGS, there are two overlooked details in the optimization of 3DGS: (i) update step coupling, which induces optimizer state rescaling and costly attribute updates outside the viewpoints, and (ii) gradient coupling in the moment, which may lead to under- or over-effective regularization. Nevertheless, such a complex coupling is under-explored. After revisiting the optimization of 3DGS, we take a step to decouple it and recompose the process into: Sparse Adam, Re-State Regularization and Decoupled Attribute Regularization. Taking a large number of experiments under the 3DGS and 3DGS-MCMC frameworks, our work provides a deeper understanding of these components. Finally, based on the empirical analysis, we re-design the optimization and propose AdamW-GS by re-coupling the beneficial components, under which better optimization efficiency and representation effectiveness are achieved simultaneously.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07192v1">AsyncEvGS: Asynchronous Event-Assisted Gaussian Splatting for Handheld Motion-Blurred Scenes</a></div>
    <div class="paper-meta">
      📅 2026-05-08
    </div>
    <details class="paper-abstract">
      3D reconstruction methods such as 3D Gaussian Splatting (3DGS) and Neural Radiance Fields (NeRF) achieve impressive photorealism but fail when input images suffer from severe motion blur. While event cameras provide high-temporal-resolution motion cues, existing event-assisted approaches rely on low-resolution sensors and strict synchronization, limiting their practicality for handheld 3D capture on common devices, such as smartphones. We introduce a flexible, high-resolution asynchronous RGB-Event dual-camera system and a corresponding reconstruction framework. Our approach first reconstructs sharp images from the event data and then employs a cross-domain pose estimation module based on the Visual Geometry Transformer (VGGT) to obtain robust initialization for 3DGS. During optimization, we employ a structure-driven event loss and view-specific consistency regularizers to mitigate the ill-posed behavior of traditional event losses and deblurring losses, ensuring both stable and high-fidelity reconstruction. We further contribute AsyncEv-Deblur, a new high-resolution RGB-Event dataset captured with our asynchronous system. Experiments demonstrate that our method achieves state-of-the-art performance on both our challenging dataset and existing benchmarks, substantially improving reconstruction robustness under severe motion blur. Project page: https://openimaginglab.github.io/AsyncEvGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07181v1">SatSurfGS: Generalizable 2D Gaussian Splatting for Sparse-View Satellite Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-08
    </div>
    <details class="paper-abstract">
      Sparse-view satellite image surface reconstruction remains highly challenging, fundamentally because the reliability of multi-view matching under satellite imaging conditions is strongly spatially heterogeneous. Affected by large photometric differences, weak textures, and repetitive textures, multi-view geometric constraints are often sparse, unevenly distributed, and locally unreliable. Although 2D Gaussian Splatting (2DGS) is more suitable than 3D Gaussian Splatting (3DGS) for the explicit representation of continuous surfaces, research on generalizable feed-forward 2DGS frameworks for sparse-view satellite surface reconstruction is still lacking. To address this issue, we propose SatSurfGS, a generalizable sparse-view surface reconstruction method for satellite imagery based on 2DGS. The proposed method builds a coarse-to-fine Gaussian attribute prediction framework and explicitly models local geometric reliability at three levels: feature learning, Gaussian parameter estimation, and training optimization. Specifically, we propose a confidence-aware monocular multi-view feature fusion module to adaptively integrate monocular priors and multi-view matching features according to local confidence; a cross-stage self-consistency residual guidance module to stabilize stage-wise Gaussian parameter refinement using the residual between the rendered height map from the previous stage and the current-stage MVS height map, together with confidence information; and a confidence bidirectional routing loss to achieve differentiated allocation of geometric and appearance supervision. Experiments on satellite datasets show that the proposed method achieves improved rendering quality, surface reconstruction accuracy, cross-dataset generalization, and inference efficiency compared with representative generalizable baselines and competitive per-scene optimization methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06088v1">OpenGaFF: Open-Vocabulary Gaussian Feature Field with Codebook Attention</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Understanding open-vocabulary 3D scenes with Gaussian-based representations remains challenging due to fragmented and spatially inconsistent semantic predictions across multi-view observations. In this paper, we present OpenGaFF, a novel framework for open-vocabulary 3D scene understanding built upon 3D Gaussian Splatting. At the core of our method is a Gaussian Feature Field that models semantics as a continuous function of Gaussian geometry and appearance. By explicitly conditioning semantic predictions on geometric structure, this formulation strengthens the coupling between geometry and semantics, leading to improved spatial coherence across similar structures in 3D space. To further enforce object-level semantic consistency, we introduce a structured codebook that serves as a set of shared semantic primitives. Furthermore, a codebook-guided attention mechanism is proposed to retrieve language features via similarity matching between query embeddings and learned codebook entries, enabling robust open-vocabulary reasoning while reducing intra-object feature variance. Extensive experiments on standard 2D and 3D open-vocabulary benchmarks demonstrate that our method consistently outperforms prior approaches, achieving improved segmentation quality, stronger 3D semantic consistency and a semantically interpretable codebook that provides insight into the learned representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05876v1">3DSS: 3D Surface Splatting for Inverse Rendering</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      We present 3D Surface Splatting (3DSS), the first differentiable surface splatting renderer for physically-based inverse rendering from multi-view images. Our central insight is that the surface separation problem at the heart of surface splatting admits a direct formulation in terms of the reconstruction kernels themselves. From this foundation we derive a coverage-based compositing model whose per-layer opacity arises directly from the accumulated Elliptical Weighted Average reconstruction weight, yielding anti-aliased silhouettes and informative visibility gradients at sparsely covered edges. Combined with forward microfacet shading under co-optimized HDR environment lighting and density-aware adaptive refinement, 3DSS jointly recovers shape, spatially-varying BRDF materials, and illumination. Because the optimized representation is a set of oriented surface samples, it bridges natively to mesh-based workflows via surface reconstruction from oriented point cloud methods. We evaluate 3DSS against mesh-based, implicit, and Gaussian-splatting baselines across geometry reconstruction, novel-view synthesis, and novel-illumination relighting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26799v2">MesonGS++: Post-training Compression of 3D Gaussian Splatting with Hyperparameter Searching</a></div>
    <div class="paper-meta">
      📅 2026-05-07
      | 💬 https://github.com/mmlab-sigs/mesongs_plus
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves high-quality novel view synthesis with real-time rendering, but its storage cost remains prohibitive for practical deployment. Existing post-training compression methods still rely on many coupled hyperparameters across pruning, transformation, quantization, and entropy coding, making it difficult to control the final compressed size and fully exploit the rate-distortion trade-off. We propose MesonGS++, a size-aware post-training codec for 3D Gaussian compression. On the codec side, MesonGS++ combines joint importance-based pruning, octree geometry coding, attribute transformation, selective vector quantization for higher-degree spherical harmonics, and group-wise mixed-precision quantization with entropy coding. On the configuration side, it treats the reserve ratio and bit-width allocation as the dominant rate-distortion knobs and jointly optimizes them under a target storage budget via discrete sampling and 0--1 integer linear programming. We further propose a linear size estimator and a CUDA parallel quantization operator to accelerate the hyperparameter searching process. Extensive experiments show that MesonGS++ achieves over 34$\times$ compression while preserving rendering fidelity, outperforming state-of-the-art post-training methods and accurately meeting target size budgets. Remarkably, without any training, MesonGS++ can even surpass the PSNR of vanilla 3DGS at a 20$\times$ compression rate on the Stump scene. Our code is available at https://github.com/mmlab-sigs/mesongs_plus
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06876v1">AdpSplit: Error-Driven Adaptive Splitting for Faster Geometry Discovery in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Adaptive density control in 3D Gaussian Splatting (3DGS) repeatedly grows the Gaussian population through fixed-cardinality random splitting to discover useful scene structure. However, in vanilla 3DGS, its binary split operator requires many densification rounds to expose fine details, making it a bottleneck for efficient training schedules with fewer iterations. We introduce AdpSplit, an error-driven adaptive split operator that determines the number of split children and initializes the child parameters from L1-pixel-error region statistics, enabling fewer densification iterations, thus reduced training time, while preserving the rendering quality of full-schedule training. Across the MipNeRF360, Deep-Blending, and Tanks&Temples datasets, AdpSplit reduces the training time of multiple accelerated 3DGS pipelines by 9.2%-22.3% as a simple drop-in replacement for the standard split operator. With FastGS, AdpSplit matches the full-schedule PSNR on MipNeRF360 while reducing training time by 16.4%, corresponding to a 12.6x acceleration over vanilla 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.13549v2">KFC-W: Generating 3D-Consistent Videos from Unposed Internet Photos</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 project page: https://genechou.com/kfcw/
    </div>
    <details class="paper-abstract">
      We address the problem of generating videos from unposed internet photos. A handful of input images serve as keyframes, and our model interpolates between them to simulate a path moving between the cameras. Given random images, a model's ability to capture underlying geometry, recognize scene identity, and relate frames in terms of camera position and orientation reflects a fundamental understanding of 3D structure and scene layout. However, existing video models such as Luma Dream Machine fail at this task. We design a self-supervised method that takes advantage of the consistency of videos and variability of multiview internet photos to train a scalable, 3D-aware video model without any 3D annotations such as camera parameters. We validate that our method outperforms all baselines in terms of geometric and appearance consistency. We also show our model benefits applications that enable camera control, such as 3D Gaussian Splatting. Our results suggest that we can scale up scene-level 3D learning using only 2D data such as videos and multiview internet photos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05155v1">Aes3D: Aesthetic Assessment in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      As 3D Gaussian Splatting (3DGS) gains attention in immersive media and digital content creation, assessing the aesthetics of 3D scenes becomes important in helping creators build more visually compelling 3D content. However, existing evaluation methods for 3D scenes primarily emphasize reconstruction fidelity and perceptual realism, largely overlooking higher-level aesthetic attributes such as composition, harmony, and visual appeal. This limitation comes from two key challenges: (1) the absence of general 3DGS datasets with aesthetic annotations, and (2) the intrinsic nature of 3DGS as a low-level primitive representation, which makes it difficult to capture high-level aesthetic features. To address these challenges, we propose Aes3D, the first systematic framework for assessing the aesthetics of 3D neural rendering scenes. Aes3D includes Aesthetic3D, the first dataset dedicated to 3D scene aesthetic assessment, built on our proposed annotation strategy for 3D scene aesthetics. In addition, we present Aes3DGSNet, a lightweight model that directly predicts scene-level aesthetic scores from 3DGS representations. Notably, our model operates solely on 3D Gaussian primitives, eliminating the need for rendering multi-view images and thus reducing computational cost and hardware requirements. Through aesthetics-supervised learning on multi-view 3DGS scene representations, Aes3DGSNet effectively captures high-level aesthetic cues and accurately regresses aesthetic scores. Experimental results demonstrate that our approach achieves strong performance while maintaining a lightweight design, establishing a new benchmark for 3D scene aesthetic assessment. Code and datasets will be made available in a future version.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.13783v2">RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Accepted to CVPR2026
    </div>
    <details class="paper-abstract">
      Temporal retiming, the ability to reconstruct and render dynamic scenes at arbitrary timestamps, is crucial for applications such as slow-motion playback, temporal editing, and post-production. However, most existing 4D Gaussian Splatting (4DGS) methods overfit at discrete frame indices but struggle to represent continuous-time frames, leading to ghosting artifacts when interpolating between timestamps. We identify this limitation as a form of temporal aliasing and propose RetimeGS, a simple yet effective 4DGS representation that explicitly defines the temporal behavior of the 3D Gaussian and mitigates temporal aliasing. To achieve smooth and consistent interpolation, we incorporate optical flow-guided initialization and supervision, triple-rendering supervision, and other targeted strategies. Together, these components enable ghost-free, temporally coherent rendering even under large motions. Experiments on datasets featuring fast motion, non-rigid deformation, and severe occlusions demonstrate that RetimeGS achieves superior quality and coherence over state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09881v3">LTGS: Long-Term Gaussian Scene Chronology From Sparse View Updates</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Accepted to CVPR 2026 Findings. Project page: https://mkjjang3598.github.io/LTGS
    </div>
    <details class="paper-abstract">
      Recent advances in novel-view synthesis can create the photo-realistic visualization of real-world environments from conventional camera captures. However, the everyday environment experiences frequent scene changes, which require dense observations, both spatially and temporally, that an ordinary setup cannot cover. We propose long-term Gaussian scene chronology from sparse-view updates, coined LTGS, an efficient scene representation that can embrace everyday changes from highly under-constrained casual captures. Given an incomplete and unstructured 3D Gaussian Splatting (3DGS) representation obtained from an initial set of input images, we robustly model the long-term chronology of the scene despite abrupt movements and subtle environmental variations. We construct objects as template Gaussians, which serve as structural, reusable priors for shared object tracks. Then, the object templates undergo a further refinement pipeline that modulates the priors to adapt to temporally varying environments given few-shot observations. Once trained, our framework is generalizable across multiple time steps through simple transformations, significantly enhancing the scalability for a temporal evolution of 3D environments. As existing datasets do not explicitly represent the long-term real-world changes with a sparse capture setup, we collect real-world datasets to evaluate the practicality of our pipeline. Experiments demonstrate that our framework achieves superior reconstruction quality compared to other baselines while enabling fast and light-weight updates. Project page is available at: https://mkjjang3598.github.io/LTGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04844v1">QuadBox: Accelerating 3D Gaussian Splatting with Geometry-Aware Boxes</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 6 pages, 4 figures. Accepted by ICIP 26
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as an advanced technique for real-time novel view synthesis by representing scene geometry and appearance using differentiable Gaussian primitives. However, efficiently computing precise Gaussian-tile intersections remains a critical task in the rasterization pipeline. To this end, we propose QuadBox, a method that leverages four axis-aligned bounding boxes to tightly encapsulate projected Gaussians in a discrete manner. First, we derive a geometry-aware stretching factor that enables the construction of a tile-aligned QuadBox, which covers the elliptical projection and largely excludes irrelevant tiles. Second, we introduce QPass, a single-pass tile traversal algorithm that exhaustively exploits the discrete nature of QuadBox, ensuring that the tile intersection check is performed with simple interval tests. Experiments on public datasets show that our method accelerates the rendering speed of 3DGS by 1.85$\times$. Code is available at \href{https://github.com/Powertony102/QuadBox}{https://github.com/Powertony102/QuadBox}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04730v1">ULF-Loc: Unbiased Landmark Feature for Robust Visual Localization with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 published to CVPR (highlight)
    </div>
    <details class="paper-abstract">
      Visual localization is a core technology for augmented reality and autonomous navigation. Recent methods combine the efficient rendering of 3D Gaussian Splatting (3DGS) with feature-based localization. These methods rely on direct matching between 2D query features and the 3D Gaussian feature field, but this often results in mismatches due to an inherent bias in the learned Gaussian feature. We theoretically analyze the feature learning process in 3DGS, revealing that the widely adopted $α$-blending optimization inherently introduces bias into 3D point features. This bias stems from the entanglement between individual Gaussians and their neighboring Gaussians, making the learned features unsuitable for precise matching tasks. Motivated by these findings, we propose ULF-Loc, an unbiased landmark feature framework that replaces biased feature optimization with geometry-weighted feature fusion. We further introduce keypoint-consensus landmark sampling to select reliable Gaussians and local geometric consistency verification to reject mismatches caused by rendering artifacts. On the Cambridge Landmarks dataset, ULF-Loc reduces the mean median translation error by 17\% compared to the state-of-the-art, while achieving superior efficiency with only 1/10 the training time and 1/6 the GPU memory of STDLoc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04509v1">CoherentRaster: Efficient 3D Gaussian Splatting for Light Field Displays</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Light field displays (LFDs) require rendering an interlaced image that encodes many view-dependent observations. This multi-view requirement introduces substantial computational overhead, making real-time rendering difficult to achieve. While 3D Gaussian Splatting (3DGS) is efficient for single-view rendering on 2D displays, directly extending it to LFDs is computationally expensive. Moreover, prior accelerations either suffer from GPU inefficiency under spatially incoherent subpixel layouts or rely on computationally heavy multi-plane intermediates. In this paper, we propose CoherentRaster, a 3DGS-based light field rendering framework that performs subpixel-level rasterization. Our method employs Cross-view Coherent Attribute Reuse to eliminate redundant computation across neighboring viewpoints and applies View-coherent Remapping to restore warp-level memory efficiency degraded by the interlaced subpixel layout. Together, CoherentRaster provides an efficient pipeline for real-time, high-quality light field synthesis on consumer-grade hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04506v1">Ilov3Splat: Instance-Level Open-Vocabulary 3D Scene Understanding in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 The International Conference on Pattern Recognition (ICPR) 2026
    </div>
    <details class="paper-abstract">
      We introduce Ilov3Splat, a novel framework for instance-level open-vocabulary 3D scene understanding built on 3D Gaussian Splatting (3D-GS). Most prior work depends on 2D rendering-based matching or point-level semantic association, which undermines cross-view consistency, lacks coherent instance-level reasoning, and limits precision in downstream 3D tasks. To address these limitations, our method jointly optimizes scene geometry and semantic representations by augmenting Gaussian splats with view-consistent feature fields. Specifically, we leverage multi-resolution hash embedding to efficiently encode language-aligned CLIP features, enabling dense and coherent language grounding in 3D space. We further train an instance feature field using contrastive loss over SAM masks, supporting fine-grained object distinction across views. At inference time, CLIP-encoded queries are matched against the learned features, followed by two-stage 3D clustering to retrieve relevant Gaussian groups. This enables our framework to identify arbitrary objects in 3D scenes based on natural language descriptions, without requiring category supervision or manual annotations. Experiments on standard benchmarks demonstrate that Ilov3Splat outperforms prior open-vocabulary 3D-GS methods in both object selection and instance segmentation, offering a flexible and accurate solution for language-driven 3D scene understanding. Project page: https://csiro-robotics.github.io/Ilov3Splat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00285v2">SV-GS: Sparse View 4D Reconstruction with Skeleton-Driven Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Reconstructing a dynamic target moving over a large area is challenging. Standard approaches for dynamic object reconstruction require dense coverage in both the viewing space and the temporal dimension, typically relying on multi-view videos captured at each time step. However, such setups are only possible in constrained environments. In real-world scenarios, observations are often sparse over time and captured sparsely from diverse viewpoints (e.g., from security cameras), making dynamic reconstruction highly ill-posed. We present SV-GS, a framework that simultaneously estimates a deformation model and the object's motion over time under sparse observations. To initialize SV-GS, we leverage a rough skeleton graph and an initial static reconstruction as inputs to guide motion estimation. (Later, we show that this input requirement can be relaxed.) Our method optimizes a skeleton-driven deformation field composed of a coarse skeleton joint pose estimator and a module for fine-grained deformations. By making only the joint pose estimator time-dependent, our model enables smooth motion interpolation while preserving learned geometric details. Experiments on synthetic datasets show that our method outperforms existing approaches under sparse observations by up to 34% in PSNR, and achieves comparable performance to dense monocular video methods on real-world datasets despite using significantly fewer frames. Moreover, we demonstrate that the input initial static reconstruction can be replaced by a diffusion-based generative prior, making our method more practical for real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.03077v2">RoDyGS: Robust Dynamic Gaussian Splatting for Casual Videos</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 29 pages, 14 figures
    </div>
    <details class="paper-abstract">
      4D reconstruction from casually captured monocular videos is challenging due to inherent ambiguity in reconstructing dynamic 3D geometry. To address this challenge, we introduce Robust Dynamic Gaussian Splatting (RoDyGS), a method that reconstructs dynamic scene representation from casual monocular videos. RoDyGS explicitly separates static and dynamic scene elements, and applies spatiotemporal regularization to enforce physically plausible geometry and temporally consistent motion. Furthermore, we propose a comprehensive benchmark, Kubric-MRig, which provides extensive camera and object motion along with simultaneous multi-view capture, features that are absent in previous benchmarks. Experiments demonstrate that RoDyGS significantly outperforms previous pose-free dynamic novel view synthesis approaches and achieves competitive rendering quality compared to existing pose-free static novel view synthesis approaches. Our proejct page is available at https://rodygs.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04435v1">Ground4D: Spatially-Grounded Feedforward 4D Reconstruction for Unstructured Off-Road Scenes</a></div>
    <div class="paper-meta">
      📅 2026-05-06
    </div>
    <details class="paper-abstract">
      Feedforward Gaussian Splatting has recently emerged as an efficient paradigm for 4D reconstruction in autonomous driving. However, in unstructured off-road scenes, its performance degrades due to high-frequency geometry, ego-motion jitter, and increased non-rigid dynamics. These factors introduce conflicting Gaussian observations across timestamps, leading to either over-smoothed renderings or structural artifacts. To address this issue, we propose Ground4D, a spatially-grounded 4D feedforward framework for pose-free off-road reconstruction. The key idea is to resolve temporal conflicts through spatially localized conditioning. Specifically, we introduce voxel-grounded temporal Gaussian aggregation, which partitions the canonical Gaussian space into spatial voxels and performs query-conditioned temporal attention within each voxel. Intra-voxel softmax normalization ensures that temporal selectivity and spatial occupancy become mutually reinforcing rather than conflicting. We furthermore introduce surface normal cues as auxiliary geometric guidance to regularize the geometry of Gaussian primitives. Extensive experiments on ORAD-3D and RELLIS-3D demonstrate that Ground4D consistently outperforms existing feedforward methods in reconstruction quality and generalizes zero-shot to unseen off-road domains. Project page and code:https://github.com/wsnbws/Ground4D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00492v2">ArtiFixer: Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2026-05-05
      | 💬 Video results: https://research.nvidia.com/labs/sil/projects/artifixer/
    </div>
    <details class="paper-abstract">
      Per-scene optimization methods such as 3D Gaussian Splatting provide state-of-the-art novel view synthesis quality but extrapolate poorly to under-observed areas. Methods that leverage generative priors to correct artifacts in these areas hold promise but currently suffer from two shortcomings. The first is scalability, as existing methods use image diffusion models or bidirectional video models that are limited in the number of views they can generate in a single pass (and thus require a costly iterative distillation process for consistency). The second is quality itself, as generators used in prior work tend to produce outputs that are inconsistent with existing scene content and fail entirely in completely unobserved regions. To solve these, we propose a two-stage pipeline that leverages two key insights. First, we train a powerful bidirectional generative model with a novel opacity mixing strategy that encourages consistency with existing observations while retaining the model's ability to extrapolate novel content in unseen areas. Second, we distill it into a causal auto-regressive model that generates hundreds of frames in a single pass. This model can directly produce novel views or serve as pseudo-supervision to improve the underlying 3D representation in a simple and highly efficient manner. We evaluate our method extensively and demonstrate that it can generate plausible reconstructions in scenarios where existing approaches fail completely. When measured on commonly benchmarked datasets, we outperform all existing baselines by a wide margin, exceeding prior state-of-the-art methods by 1-3 dB PSNR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03337v1">FreeTimeGS++: Secrets of Dynamic Gaussian Splatting and Their Principles</a></div>
    <div class="paper-meta">
      📅 2026-05-05
      | 💬 22 pages, 8 figures
    </div>
    <details class="paper-abstract">
      The recent surge in 4D Gaussian Splatting (4DGS) has achieved impressive dynamic scene reconstruction. While these methods demonstrate remarkable performance, the specific drivers behind such gains remain less explored, making a systematic understanding of the underlying principles challenging. In this paper, we perform a comprehensive analysis of these hidden factors to provide a clearer perspective on the 4DGS framework. We first establish a controlled baseline, FreeTimeGS_ours, by formalizing and reproducing the heuristics of the state-of-the-art FreeTimeGS. Using this framework, we dissect 4DGS along its fundamental axes and uncover key secrets, including the emergent temporal partitioning driven by Gaussian durations and the discrepancy between photometric fidelity and spatiotemporal consistency. Based on these insights, we propose FreeTimeGS++, a principled method that employs gated marginalization and neural velocity fields to achieve superior stability and robust dynamic representations. Our approach yields reproducible results with reduced run-to-run variance. We will release our implementation to provide a reliable foundation for future 4DGS research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21874v2">Interactive Augmented Reality-enabled Outdoor Scene Visualization For Enhanced Real-time Disaster Response</a></div>
    <div class="paper-meta">
      📅 2026-05-04
      | 💬 6 pages, 2 figures
    </div>
    <details class="paper-abstract">
      A user-centered AR interface for disaster response is presented in this work that uses 3D Gaussian Splatting (3DGS) to visualize detailed scene reconstructions, while maintaining situational awareness and keeping cognitive load low. The interface relies on a lightweight interaction approach, combining World-in-Miniature (WIM) navigation with semantic Points of Interest (POIs) that can be filtered as needed, and it is supported by an architecture designed to stream updates as reconstructions evolve. User feedback from a preliminary evaluation indicates that this design is easy to use and supports real-time coordination, with participants highlighting the value of interaction and POIs for fast decision-making in context. Thorough user-centric performance evaluation demonstrates strong usability of the developed interface and high acceptance ratios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17817v3">Chorus: Multi-Teacher Pretraining for Holistic 3D Gaussian Scene Encoding</a></div>
    <div class="paper-meta">
      📅 2026-05-04
      | 💬 Project page at https://gaussianworld.github.io/Chorus
    </div>
    <details class="paper-abstract">
      While 3DGS has emerged as a high-fidelity scene representation, encoding rich, general-purpose features directly from its primitives remains under-explored. We address this gap by introducing Chorus, a multi-teacher pretraining framework that learns a holistic feed-forward 3D Gaussian Splatting (3DGS) scene encoder by distilling complementary signals from 2D foundation models. Chorus employs a shared 3D encoder and teacher-specific projectors to learn from language-aligned, generalist, and object-aware teachers, encouraging a shared embedding space that captures signals from high-level semantics to fine-grained structure. We evaluate Chorus on a wide range of tasks: open-vocabulary semantic and instance segmentation, linear and decoder probing, data-efficient supervision, as well as LLM-based Q&A. Besides 3DGS, we also test Chorus on several benchmarks that only support point clouds by pretraining a variant using only Gaussian centers, colors, and estimated normals. Surprisingly, this encoder shows strong transfer and outperforms the point-cloud baseline while using 39.9 times fewer training scenes. Finally, we propose a render-and-distill adaptation that facilitates out-of-domain finetuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17185v2">LGDWT-GS: Local and Global Discrete Wavelet-Regularized 3D Gaussian Splatting for Sparse-View Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-04
    </div>
    <details class="paper-abstract">
      We propose a new method for few-shot 3D reconstruction that integrates global and local frequency regularization to stabilize geometry and preserve fine details under sparse-view conditions, addressing a key limitation of existing 3D Gaussian Splatting (3DGS) models. We also introduce a new multispectral greenhouse dataset containing four spectral bands captured from diverse plant species under controlled conditions. Alongside the dataset, we release an open-source benchmarking package that defines standardized few-shot reconstruction protocols for evaluating 3DGS-based methods. Experiments on our multispectral dataset, as well as standard benchmarks, demonstrate that the proposed method achieves sharper, more stable, and spectrally consistent reconstructions than existing baselines. The dataset and code for this work are publicly available
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.02784v1">HumanSplatHMR: Closing the Loop Between Human Mesh Recovery and Gaussian Splatting Avatar</a></div>
    <div class="paper-meta">
      📅 2026-05-04
    </div>
    <details class="paper-abstract">
      Accurately recovering human pose and appearance from video is an essential component of scene reconstruction, with applications to motion capture, motion prediction, virtual reality, and digital twinning. Despite significant interest in building realistic human avatars from video, this paper demonstrates that existing methods do not accurately recover the 3D geometry of humans. ViT-based approaches are not consistently reliable and can overfit to 2D views, while NeRF- and Gaussian Splatting-based avatars treat pose and appearance separately, limiting rendering generalization to new poses. To resolve these shortcomings, this paper proposes HumanSplatHMR, a joint optimization framework that refines 3D human poses while simultaneously learning a high-fidelity avatar for novel-view and novel-pose synthesis. Our key insight is to close the loop between geometric pose estimation and differentiable rendering. Unlike prior human avatar methods that rely on accurate human pose obtained through motion capture systems or offline refinement, which are impractical in in-the-wild scenarios, our approach uses only human mesh estimates from a state-of-the-art human pose estimator to better reflect real-world conditions. Therefore, instead of using the human pose only as a deformation prior, HumanSplatHMR backpropagates photometric, segmentation, and depth losses through a differentiable renderer to the pose parameters and global position. This coupling refines the global 3D pose over time, improving accuracy and alignment while producing better renderings from novel views. Experiments show consistent improvements over pose recovery baselines that omit image-level refinement and avatar baselines that decouple pose estimation from avatar reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03200v2">A High-Fidelity Digital Twin for Robotic Manipulation Based on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-04
      | 💬 Accepted By Journal of Robot Learning
    </div>
    <details class="paper-abstract">
      Developing high-fidelity, interactive digital twins is crucial for enabling closed-loop motion planning and reliable real-world robot execution, which are essential to advancing sim-to-real transfer. However, existing approaches often suffer from slow reconstruction, limited visual fidelity, and difficulties in converting photorealistic models into planning-ready collision geometry. We present a practical framework that constructs high-quality digital twins within minutes from sparse RGB inputs. Our system employs 3D Gaussian Splatting (3DGS) for fast, photorealistic reconstruction as a unified scene representation. We enhance 3DGS with visibility-aware semantic fusion for accurate 3D labelling and introduce an efficient, filter-based geometry conversion method to produce collision-ready models seamlessly integrated with a Unity-ROS2-MoveIt physics engine. In experiments with a Franka Emika Panda robot performing pick-and-place tasks, we demonstrate that this enhanced geometric accuracy effectively supports robust manipulation in real-world trials. These results demonstrate that 3DGS-based digital twins, enriched with semantic and geometric consistency, offer a fast, reliable, and scalable path from perception to manipulation in unstructured environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21668v2">Space-Time Forecasting of Dynamic Scenes with Motion-aware Gaussian Grouping</a></div>
    <div class="paper-meta">
      📅 2026-05-04
      | 💬 20 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Forecasting dynamic scenes remains a fundamental challenge in computer vision, as limited observations make it difficult to capture coherent object-level motion and long-term temporal evolution. We present Motion Group-aware Gaussian Forecasting (MoGaF), a framework for long-term scene extrapolation built upon the 4D Gaussian Splatting representation. MoGaF introduces motion-aware Gaussian grouping and group-wise optimization to enforce physically consistent motion across both rigid and non-rigid regions, yielding spatially coherent dynamic representations. Leveraging this structured space-time representation, a lightweight forecasting module predicts future motion, enabling realistic and temporally stable scene evolution. Experiments on synthetic and real-world datasets demonstrate that MoGaF consistently outperforms existing baselines in rendering quality, motion plausibility, and long-term forecasting stability. Our project page is available at https://slime0519.github.io/mogaf
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.18966v2">SVGS: Enhancing Gaussian Splatting Using Primitives with Spatially Varying Colors</a></div>
    <div class="paper-meta">
      📅 2026-05-04
      | 💬 IEEE Transactions on Visualization and Computer Graphics
    </div>
    <details class="paper-abstract">
      Gaussian Splatting demonstrates impressive results in multi-view reconstruction based on Gaussian explicit representations. However, the current Gaussian primitives only have a single view-dependent color and an opacity to represent the appearance and geometry of the scene, resulting in a non-compact representation. In this paper, we introduce a new method called SVGS (Spatially Varying Gaussian Splatting) that utilizes spatially varying colors and opacity in a single Gaussian primitive to improve its representation ability. We have implemented bilinear interpolation, movable kernels, and tiny neural networks as spatially varying functions. SVGS employs 2D Gaussian surfels as primitives, which significantly enhances novel-view synthesis while maintaining high-quality geometric reconstruction. This approach is particularly effective in practical applications, as scenes combining complex textures with relatively simple geometry occur frequently in real-world environments. Quantitative and qualitative experimental results demonstrate that all three functions outperform the baseline, with the best movable kernels achieving superior novel view synthesis performance on multiple datasets, highlighting the strong potential of spatially varying functions. Project page: https://ruixu.me/html/SuperGaussians/index.html
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03210v2">Flux4D: Flow-based Unsupervised 4D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-04
      | 💬 NeurIPS 2025. Project page: https://waabi.ai/flux4d/
    </div>
    <details class="paper-abstract">
      Reconstructing large-scale dynamic scenes from visual observations is a fundamental challenge in computer vision, with critical implications for robotics and autonomous systems. While recent differentiable rendering methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have achieved impressive photorealistic reconstruction, they suffer from scalability limitations and require annotations to decouple actor motion. Existing self-supervised methods attempt to eliminate explicit annotations by leveraging motion cues and geometric priors, yet they remain constrained by per-scene optimization and sensitivity to hyperparameter tuning. In this paper, we introduce Flux4D, a simple and scalable framework for 4D reconstruction of large-scale dynamic scenes. Flux4D directly predicts 3D Gaussians and their motion dynamics to reconstruct sensor observations in a fully unsupervised manner. By adopting only photometric losses and enforcing an "as static as possible" regularization, Flux4D learns to decompose dynamic elements directly from raw data without requiring pre-trained supervised models or foundational priors simply by training across many scenes. Our approach enables efficient reconstruction of dynamic scenes within seconds, scales effectively to large datasets, and generalizes well to unseen environments, including rare and unknown objects. Experiments on outdoor driving datasets show Flux4D significantly outperforms existing methods in scalability, generalization, and reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01674v2">VRGaussianAvatar: Integrating 3D Gaussian Avatars into VR</a></div>
    <div class="paper-meta">
      📅 2026-05-04
      | 💬 Accepted as an IEEE TVCG paper at IEEE VR 2026 (journal track)
    </div>
    <details class="paper-abstract">
      We present VRGaussianAvatar, an integrated system that enables real-time full-body 3D Gaussian Splatting (3DGS) avatars in virtual reality using only head-mounted display (HMD) tracking signals. The system adopts a parallel pipeline with a VR Frontend and a GA Backend. The VR Frontend uses inverse kinematics to estimate full-body pose and streams the resulting pose along with stereo camera parameters to the backend. The GA Backend stereoscopically renders a 3DGS avatar reconstructed from a single image. To improve stereo rendering efficiency, we introduce Binocular Batching, which jointly processes left and right eye views in a single batched pass to reduce redundant computation and support high-resolution VR displays. We evaluate VRGaussianAvatar with quantitative performance tests and a within-subject user study against image- and video-based mesh avatar baselines. Results show that VRGaussianAvatar sustains interactive VR performance and yields higher perceived appearance similarity, embodiment, and plausibility. Project page and source code are available at https://vrgaussianavatar.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.02086v1">GETA-3DGS: Automatic Joint Structured Pruning and Quantization for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-03
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) is a state-of-the-art representation for real-time photorealistic novel-view synthesis, yet a single high-fidelity scene typically occupies hundreds of megabytes to several gigabytes, exceeding the budgets of mobile, immersive, and volumetric video platforms. Existing 3DGS compression methods (e.g., HAC++, FlexGaussian, LP-3DGS) treat pruning, quantization, and entropy coding as separate stages and rely on hand-tuned heuristics (opacity thresholds, fixed bit-widths, SH truncation), limiting cross-scene generalization and preventing users from specifying a target rate or quality budget. We propose GETA-3DGS, to our knowledge the first end-to-end automatic joint structured pruning and quantization framework for 3DGS. Building on GETA for joint pruning-quantization of deep networks, we contribute: (i) a 3DGS-aware quantization-aware dependency graph (QADG) treating each Gaussian primitive as a group with five attribute sub-nodes and degree-aware SH sub-nodes; (ii) a render-aware saliency fusing transmittance-weighted contribution, screen-space gradient, and pixel coverage into a Gaussian-level importance score; and (iii) a heterogeneous per-attribute mixed-precision scheme co-optimized with structural sparsity under a projected partial saliency-guided (PPSG) descent guarantee. On Mip-NeRF 360, Tanks and Temples, and Deep Blending, GETA-3DGS operates directly on raw Gaussian primitives rather than a post-hoc anchor representation, delivering ~5x storage reduction over Vanilla 3DGS with no per-scene thresholds. Bit-width policy is the dominant rate-distortion lever: a uniform 6-bit cap costs up to -6.74 dB on view-dependent scenes versus our heterogeneous allocation, matching an information-theoretic reverse-water-filling analysis we develop. GETA-3DGS is complementary to existing codecs: entropy coding (HAC++, CompGS) is downstream, so the two can be composed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01995v1">From Concept to Capability: Evaluating 3D Gaussian Splatting for Synthetic Scene Editing in Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-05-03
      | 💬 Accepted in the 45th International Conference on Computer Safety, Reliability and Security (SafeComp 2026)
    </div>
    <details class="paper-abstract">
      The perception of an Autonomous Driving System (ADS) critically depends on relevant, comprehensive, and diverse datasets to ensure its safety while operating in the environment. Field data collection lacks completeness with respect to the list of rare but still possible safety-related scenarios needed for the development, verification, and validation of the ADS. 3D Gaussian Splatting (3DGS) has shown promising capabilities for the reconstruction and editing of scenes based on data collected by cameras and LiDAR sensors. However, the industrial fidelity evaluation of reconstructions is underexplored, which is crucial when employing such methods in safety-related systems, especially for ADS. This becomes more challenging as ADS operates in a dynamic, uncontrolled environment with limited viewpoints and often partially occluded objects. This paper addresses this gap by proposing and implementing a framework (Fig. 1) to systematically analyze the capabilities and limitations of 3DGS for use in the reconstruction of safety-related scenes. It focuses on the quality of reconstruction for vehicles and pedestrians, which are the two most critical object classes for ADS. Our findings provide industry insights into the fidelity degradation of reconstructions from multiple novel viewpoints, both lateral and longitudinal, enabling the integration of these methods into real-world industrial AD software development and testing pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06846v3">DynFOA: Generating First-Order Ambisonics with Conditional Diffusion for Dynamic and Acoustically Complex 360-Degree Videos</a></div>
    <div class="paper-meta">
      📅 2026-05-03
      | 💬 Revised version with updated content. A duplicate submission, arXiv:2604.02781, was previously submitted by mistake and has been withdrawn. This submission is the intended replacement of arXiv:2602.06846
    </div>
    <details class="paper-abstract">
      Spatial audio is crucial for immersive 360-degree video experiences, yet most 360-degree videos lack it due to the difficulty of capturing spatial audio during recording. Automatically generating spatial audio such as first-order ambisonics (FOA) from video therefore remains an important but challenging problem. In complex scenes, sound perception depends not only on sound source locations but also on scene geometry, materials, and dynamic interactions with the environment. However, existing approaches only rely on visual cues and fail to model dynamic sources and acoustic effects such as occlusion, reflections, and reverberation. To address these challenges, we propose DynFOA, a generative framework that synthesizes FOA from 360-degree videos by integrating dynamic scene reconstruction with conditional diffusion modeling. DynFOA analyzes the input video to detect and localize dynamic sound sources, estimate depth and semantics, and reconstruct scene geometry and materials using 3D Gaussian Splatting (3DGS). The reconstructed scene representation provides physically grounded features that capture acoustic interactions between sources, environment, and listener viewpoint. Conditioned on these features, a diffusion model generates spatial audio consistent with the scene dynamics and acoustic context. We introduce M2G-360, a dataset of 600 real-world clips divided into MoveSources, Multi-Source, and Geometry subsets for evaluating robustness under diverse conditions. Experiments show that DynFOA consistently outperforms existing methods in spatial accuracy, acoustic fidelity, distribution matching, and perceived immersive experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01736v1">Multi-Scale Gaussian-Language Map for Zero-shot Embodied Navigation and Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-03
      | 💬 Accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      Understanding the geometric and semantic structure of environments is essential for embodied navigation and reasoning. Existing semantic mapping methods trade off between explicit geometry and multi-scale semantics, and lack a native interface for large models, thus requiring additional training of feature projection for semantic alignment. To this end, we propose the multi-scale Gaussian-Language Map (GLMap), which introduces three key designs: (1) explicit geometry, (2) multi-scale semantics covering both instance and region concepts, and (3) a dual-modality interface where each semantic unit jointly stores a natural language description and a 3D Gaussian representation. The 3D Gaussians enable compact storage and fast rendering of task-relevant images via Gaussian splatting. To enable efficient incremental construction, we further propose a Gaussian Estimator that analytically derives Gaussian parameters from dense point clouds without gradient-based optimization. Experiments on ObjectNav, InstNav, and SQA tasks show that GLMap effectively enhances target navigation and contextual reasoning, while remaining compatible with large-model-based methods in a zero-shot manner. The code is available at https://github.com/sx-zhang/GLMap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.19703v3">High-Quality Spatial Reconstruction and Orthoimage Generation Using Efficient 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-03
    </div>
    <details class="paper-abstract">
      Highly accurate geometric precision and dense image features characterize True Digital Orthophoto Maps (TDOMs), which are in great demand for applications such as urban planning, infrastructure management, and environmental monitoring. Traditional TDOM generation methods need sophisticated processes, such as Digital Surface Models (DSM) and occlusion detection, which are computationally expensive and prone to errors. This work presents an alternative technique rooted in 2D Gaussian Splatting (2DGS), free of explicit DSM and occlusion detection. With depth map generation, spatial information for every pixel within the TDOM is retrieved and can reconstruct the scene with high precision. Divide-and-conquer strategy achieves excellent GS training and rendering with high-resolution TDOMs at a lower resource cost, which preserves higher quality of rendering on complex terrain and thin structure without a decrease in efficiency. Experimental results demonstrate the efficiency of large-scale scene reconstruction and high-precision terrain modeling. This approach provides accurate spatial data, which assists users in better planning and decision-making based on maps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.18713v2">SaLF: Sparse Local Fields for Multi-Sensor Rendering in Real-Time</a></div>
    <div class="paper-meta">
      📅 2026-05-02
      | 💬 ICRA 2026. Project page: https://waabi.ai/salf/
    </div>
    <details class="paper-abstract">
      High-fidelity sensor simulation of light-based sensors such as cameras and LiDARs is critical for safe and accurate autonomy testing. Neural radiance field (NeRF)-based methods that reconstruct sensor observations via ray-casting of implicit representations have demonstrated accurate simulation of driving scenes, but are slow to train and render, hampering scalability. 3D Gaussian Splatting (3DGS) has demonstrated faster training and rendering times through rasterization, but is primarily restricted to pinhole camera sensors, preventing usage for realistic multi-sensor autonomy evaluation. Moreover, both NeRF and 3DGS couple the representation with the rendering procedure (implicit networks for ray-based evaluation, particles for rasterization), preventing interoperability, which is key for general usage. In this work, we present Sparse Local Fields (SaLF), a novel volumetric representation that supports rasterization and raytracing for unified multi-sensor simulation. SaLF represents volumes as a sparse set of 3D voxel primitives, where each voxel is a local implicit field. SaLF has fast training ($<$30 min) and rendering capabilities (50+ FPS for camera and 600+ FPS for LiDAR), has adaptive pruning and densification to easily handle large scenes, and can support non-pinhole cameras and spinning LiDARs. We demonstrate that SaLF has similar realism as existing self-driving sensor simulation methods while improving efficiency and enhancing capabilities, enabling more scalable simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.15491v4">GSDeformer: Direct, Real-time and Extensible Cage-based Deformation for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-02
      | 💬 Project Page: https://jhuangbu.github.io/gsdeformer, Video: https://www.youtube.com/watch?v=-ecrj48-MqM
    </div>
    <details class="paper-abstract">
      We present GSDeformer, a method that enables cage-based deformation on 3D Gaussian Splatting (3DGS). Our approach bridges cage-based deformation and 3DGS by using a proxy point-cloud representation. This point cloud is generated from 3D Gaussians, and deformations applied to the point cloud are translated into transformations on the 3D Gaussians. To handle potential bending caused by deformation, we incorporate a splitting process to approximate it. Our method does not modify or extend the core architecture of 3D Gaussian Splatting, making it compatible with any trained vanilla 3DGS or its variants. Additionally, we automate cage construction for 3DGS and its variants using a render-and-reconstruct approach. Experiments demonstrate that GSDeformer delivers superior deformation results compared to existing methods, is robust under extreme deformations, requires no retraining for editing, runs in real-time, and can be extended to other 3DGS variants. Project Page: https://jhuangbu.github.io/gsdeformer/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01466v1">SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion</a></div>
    <div class="paper-meta">
      📅 2026-05-02
      | 💬 Accepted as a Spotlight paper at ICML 2026; camera-ready version
    </div>
    <details class="paper-abstract">
      Although multi-modal learning has advanced point cloud completion, the theoretical mechanisms remain unclear. Recent works attribute success to the connection between modalities, yet we identify that standard hard projection severs this connection: projecting a sparse point cloud onto the image plane yields an extremely sparse support, which hinders visual prior propagation, a failure mode we term Cross-Modal Entropy Collapse. To address this practical limitation, we propose SplAttN, which replaces hard projection with Differentiable Gaussian Splatting to produce a dense, continuous image-plane representation. By reformulating projection as continuous density estimation, SplAttN avoids collapsed sparse support, facilitates gradient flow, and improves cross-modal connection learnability. Extensive experiments show that SplAttN achieves state-of-the-art performance on PCN and ShapeNet-55/34. Crucially, we utilize the real-world KITTI benchmark as a stress test for multi-modal reliance. Counter-factual evaluation reveals that while baselines degenerate into unimodal template retrievers insensitive to visual removal, SplAttN maintains a robust dependency on visual cues, validating that our method establishes an effective cross-modal connection. Code is available at https://github.com/zay002/SplAttN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.12001v4">3D Gaussian Splatting against Moving Objects for High-Fidelity Street Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-02
    </div>
    <details class="paper-abstract">
      The accurate reconstruction of dynamic street scenes is critical for applications in autonomous driving, augmented reality, and virtual reality. Traditional methods relying on dense point clouds and triangular meshes struggle with moving objects, occlusions, and real-time processing constraints, limiting their effectiveness in complex urban environments. While multi-view stereo and neural radiance fields have advanced 3D reconstruction, they face challenges in computational efficiency and handling scene dynamics. This paper proposes a novel 3D Gaussian point distribution method for dynamic street scene reconstruction. Our approach introduces an adaptive transparency mechanism that eliminates moving objects while preserving high-fidelity static scene details. Additionally, iterative refinement of Gaussian point distribution enhances geometric accuracy and texture representation. We integrate directional encoding with spatial position optimization to optimize storage and rendering efficiency, reducing redundancy while maintaining scene integrity. Experimental results demonstrate that our method achieves high reconstruction quality, improved rendering performance, and adaptability in large-scale dynamic environments. These contributions establish a robust framework for real-time, high-precision 3D reconstruction, advancing the practicality of dynamic scene modeling across multiple applications. The source code for this work is available to the public at https://github.com/okic-ca/3dgs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01232v1">A Principled Approach for Creating High-fidelity Synthetic Demonstrations for Imitation Learning</a></div>
    <div class="paper-meta">
      📅 2026-05-02
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled visually realistic demonstration generation from a single expert trajectory and a short multi-view scan. However, existing 3DGS-based synthesis pipelines typically generate new motions using sampling-based planners or trajectory optimization, which often deviate substantially from the expert's demonstrated path. While such deviations may be acceptable for tasks insensitive to motion shape, they discard subtle spatial and temporal structure that is critical for contact-rich and shape-sensitive manipulation, causing increased demonstration diversity to harm downstream policy learning. We argue that demonstration synthesis should treat the expert trajectory as a strong prior. Building on this principle, we propose a framework that synthesizes diverse task demonstrations while explicitly preserving expert motion structure. We model the expert trajectory using Dynamic Movement Primitives (DMPs) and retarget it to new goals, object configurations, and viewpoints within a reconstructed 3DGS scene, yielding phase-consistent, shape-preserving motion by construction. To safely realize this expert-preserving diversity in cluttered scenes, we introduce an analytic obstacle-aware DMP formulation that operates directly on the continuous density field induced by the 3DGS representation. This enables collision avoidance while minimally perturbing the nominal expert motion, unifying photorealistic rendering and geometric reasoning without additional scene representations. We evaluate our approach on a Spot mobile manipulator across three manipulation tasks with increasing sensitivity to trajectory fidelity. Compared to planner- and optimization-based synthesis, our method produces trajectories with lower deviation and collision rates and yields higher task success when training diffusion-based visuomotor policies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01195v1">TAIL-Safe: Task-Agnostic Safety Monitoring for Imitation Learning Policies</a></div>
    <div class="paper-meta">
      📅 2026-05-02
    </div>
    <details class="paper-abstract">
      Recent imitation learning (IL) algorithms such as flow-matching and diffusion policies demonstrate remarkable performance in learning complex manipulation tasks. However, these policies often fail even when operating within their training distribution due to extreme sensitivity to initial conditions and irreducible approximation errors that lead to compounding drift. This makes it unsafe to deploy IL policies in the field where out-of-distribution scenarios are prevalent. A prerequisite for safe deployment is enabling the policy to determine whether it can execute a task the way it was learned from demonstrations. This paper presents TAIL-Safe, a principled approach to identify, for a trained IL policy, a safe set from where the policy empirically succeeds in completing the learned task. We propose a Lipschitz-continuous Q-value function that maps state-action pairs to a long-term safety score based on three short-term task-agnostic criteria: visibility, recognizability, and graspability. The zero-superlevel set of this function characterizes an empirical control invariant set over state-action pairs. When the nominal policy proposes an action outside this set, we apply a recovery mechanism inspired by Nagumo's theorem that uses gradient ascent to the Q-function to steer the policy back to safety. To learn this Q-function, we construct a high-fidelity digital twin using Gaussian Splatting that enables systematic collection of failure data without risk to physical hardware. Experiments with a Franka Emika robot demonstrate that flow-matching policies, which fail under run-time perturbations, achieve consistent task success when guided by the proposed TAIL-Safe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.00569v1">2D-SuGaR: Surface-Aware Gaussian Splatting for Geometrically Accurate Mesh Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-01
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for generating photorealistic renderings of a scene in real-time. However, the volumetric nature of 3DGS limits its ability to accurately capture surface geometry. To address this, 2D Gaussian Splatting (2DGS) was proposed to enable view-consistent and geometrically accurate surface reconstruction from multi-view images. However, 2DGS can be sensitive to the initialization of the Gaussian primitives. Reliance on Structure-from-Motion (SfM) initializations, which can produce poor estimates on challenging image sets, may lead to subpar results. In this work, we enhance 2DGS by incorporating monocular depth and normal priors to improve both geometric accuracy and robustness. We propose a depth-guided initialization strategy for Gaussians and introduce a clustering-based technique for pruning degenerate Gaussians. We evaluate our method on the DTU dataset, where it achieves state-of-the-art results in mesh reconstruction while preserving high-quality novel view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12895v3">High Dynamic Range 3D Gaussian Splatting via Luminance-Chromaticity Decomposition</a></div>
    <div class="paper-meta">
      📅 2026-05-01
    </div>
    <details class="paper-abstract">
      High Dynamic Range (HDR) 3D reconstruction is pivotal for professional content creation in filmmaking and virtual production. Existing methods typically rely on multi-exposure Low Dynamic Range (LDR) supervision to constrain the learning process within vast brightness spaces, resulting in complex, dual-branch architectures. This work explores the feasibility of learning HDR 3D models exclusively in the HDR data space to simplify model design. By analyzing 3D Gaussian Splatting (3DGS) for HDR imagery, we reveal that its failure stems from the limited capacity of Spherical Harmonics (SHs) to capture extreme radiance variations across views, often biasing towards high-radiance observations and underfitting. While increasing the maximum SH degree improves training fitting, it leads to severe overfitting and excessive parameter overhead. To address this, we propose \textit{Luminance--Chromaticity Decomposition Gaussian Splatting} (LCD-GS). By decoupling luminance and chromaticity into independent parameters, LCD-GS significantly enhances learning flexibility with minimal parameter increase (\textit{e.g.}, one extra scalar per primitive). Notably, LCD-GS maintains the original training and inference pipeline, requiring only a change in color representation. This explicit decomposition naturally enables primitive-level local and global luminance editing during inference. Extensive experiments on synthetic and real datasets demonstrate that LCD-GS consistently outperforms state-of-the-art methods in reconstruction fidelity and dynamic-range preservation even with a simpler, more efficient architecture, providing an elegant paradigm for professional-grade HDR 3D modeling. Code and datasets will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.00498v1">GOR-IS: 3D Gaussian Object Removal in the Intrinsic Space</a></div>
    <div class="paper-meta">
      📅 2026-05-01
    </div>
    <details class="paper-abstract">
      Recent advances in Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have made it standard practice to reconstruct 3D scenes from multi-view images. Removing objects from such 3D representations is a fundamental editing task that requires complete and seamless inpainting of occluded regions, ensuring consistency in geometry and appearance. Although existing methods have made notable progress in improving inpainting consistency, they often neglect global lighting effects, leading to physically implausible results. Moreover, these methods struggle with view-dependent non-Lambertian surfaces, where appearance varies across viewpoints, leading to unreliable inpainting. In this paper, we present 3D Gaussian Object Removal in the Intrinsic Space (GOR-IS), a novel framework for physically consistent and visually coherent 3D object removal. Our approach decomposes the scene into intrinsic components and explicitly models light transport to maintain global lighting effects consistency. Furthermore, we introduce an intrinsic-space inpainting module that operates directly in the material and lighting domains, effectively addressing the challenges posed by non-Lambertian surfaces. Extensive experiments on both synthetic and real-world datasets demonstrate that our framework substantially improves the physical consistency and visual coherence of object removal, outperforming existing methods by 13% in perceptual similarity (LPIPS) and 2dB in peak signal-to-noise ratio (PSNR). Code is publicly available at https://applezyh.github.io/GOR-IS-project-page/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04349v2">VecSet-Edit: Unleashing Pre-trained LRM for Mesh Editing from Single Image</a></div>
    <div class="paper-meta">
      📅 2026-05-01
    </div>
    <details class="paper-abstract">
      3D editing has emerged as a critical research area to provide users with flexible control over 3D assets. While current editing approaches predominantly focus on 3D Gaussian Splatting or multi-view images, the direct editing of 3D meshes remains underexplored. Prior attempts, such as VoxHammer, rely on voxel-based representations that suffer from limited resolution and necessitate labor-intensive 3D mask. To address these limitations, we propose \textbf{VecSet-Edit}, the first pipeline that leverages the high-fidelity VecSet Large Reconstruction Model (LRM) as a backbone for mesh editing. Our approach is grounded on a analysis of the spatial properties in VecSet tokens, revealing that token subsets govern distinct geometric regions. Based on this insight, we introduce Mask-guided Token Seeding and Attention-aligned Token Gating strategies to precisely localize target regions using only 2D image conditions. Also, considering the difference between VecSet diffusion process versus voxel we design a Drift-aware Token Pruning to reject geometric outliers during the denoising process. Finally, our Detail-preserving Texture Baking module ensures that we not only preserve the geometric details of original mesh but also the textural information. More details can be found in our project page: https://github.com/BlueDyee/VecSet-Edit/tree/main
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26262v2">Semantic Foam: Unifying Spatial and Semantic Scene Decomposition</a></div>
    <div class="paper-meta">
      📅 2026-05-01
      | 💬 15 pages, 10 figures, Accepted to CVPR 2026 (Highlight) , Project page: http://semanticfoam.github.io/
    </div>
    <details class="paper-abstract">
      Modern scene reconstruction methods, such as 3D Gaussian Splatting, deliver photo-realistic novel view synthesis at real-time speeds, yet their adoption in interactive graphics applications has been limited. A major bottleneck is the difficulty of interacting with these representations compared to traditional, human-authored 3D assets. While previous research has attempted to impose semantic decomposition on these models, significant challenges remain regarding segmentation quality and consistency. To address this, we introduce Semantic Foam, extending the recently proposed Radiant Foam representations to semantic decomposition tasks. Our approach integrates the natural spatial volumetric decomposition of Radiant Foam's Voronoi mesh with an explicit semantic feature field parameterized at the cell level. This explicit structure enables direct spatial regularization, which prevents artifacts caused by occlusion or inconsistent supervision across views - common pitfalls for other point-based representations. Experimental results show that our method achieves superior object-level segmentation performance compared to state-of-the-art methods like Gaussian Grouping and SAGA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.28111v2">GSDrive: Reinforcing Driving Policies by Multi-mode Trajectory Probing with 3D Gaussian Splatting Environment</a></div>
    <div class="paper-meta">
      📅 2026-05-01
      | 💬 initial version
    </div>
    <details class="paper-abstract">
      End-to-end (E2E) autonomous driving presents a promising approach for translating perceptual inputs directly into driving actions. However, prohibitive annotation costs and temporal data quality degradation hinder long-term real-world deployment. While combining imitation learning (IL) and reinforcement learning (RL) is a common strategy for policy improvement, conventional RL training relies on delayed, event-based rewards-policies learn only from catastrophic outcomes such as collisions, leading to premature convergence to suboptimal behaviors. To address these limitations, we introduce GSDrive, a framework that exploits 3D Gaussian Splatting (3DGS) for differentiable, physics-based reward shaping in E2E driving policy improvement. Our method incorporates a flow matching-based trajectory predictor within the 3DGS simulator, enabling multi-mode trajectory probing where candidate trajectories are rolled out to assess prospective rewards. This establishes a bidirectional knowledge exchange between IL and RL by grounding reward functions in physically simulated interaction signals, offering immediate dense feedback instead of sparse catastrophic events. Evaluated on the reconstructed nuScenes dataset, our method surpasses existing simulation-based RL driving approaches in closed-loop experiments. Code is available at https://github.com/ZionGo6/GSDrive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.00408v1">Beyond Heuristics: Learnable Density Control for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-01
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has demonstrated impressive real-time rendering performance, its efficacy remains constrained by a reliance on heuristic density control. Despite numerous refinements to these handcrafted rules, such methods inherently lack the flexibility to adapt to diverse scenes with complex geometries. In this paper, we propose a paradigm shift for density control from rigid heuristics to fully learnable policies. Specifically, we introduce \textbf{LeGS}, a framework that reformulates density control as a parameterized policy network optimized via Reinforcement Learning (RL). Central to our approach is the tailored effective reward function grounded in sensitivity analysis, which precisely quantifies the marginal contribution of individual Gaussians to reconstruction quality. To maintain computational tractability, we derive a closed-form solution that reduces the complexity of reward calculation from $O(N^2)$ to $O(N)$. Extensive experiments on the Mip-NeRF 360, Tanks \& Temples, and Deep Blending datasets demonstrate that \textbf{LeGS} significantly outperforms state-of-the-art methods, striking a superior balance between reconstruction quality and efficiency. The code will be released at https://github.com/AaronNZH/LeGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04929v5">CryoSplat: Gaussian Splatting for Cryo-EM Homogeneous Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-01
      | 💬 Published at ICLR 2026 (Camera-ready). Code available at https://github.com/Chen-Suyi/cryosplat
    </div>
    <details class="paper-abstract">
      As a critical modality for structural biology, cryogenic electron microscopy (cryo-EM) facilitates the determination of macromolecular structures at near-atomic resolution. The core computational task in single-particle cryo-EM is to reconstruct the 3D electrostatic potential of a molecule from noisy 2D projections acquired at unknown orientations. Gaussian mixture models (GMMs) provide a continuous, compact, and physically interpretable representation for molecular density and have recently gained interest in cryo-EM reconstruction. However, existing methods rely on external consensus maps or atomic models for initialization, limiting their use in self-contained pipelines. In parallel, differentiable rendering techniques such as Gaussian splatting have demonstrated remarkable scalability and efficiency for volumetric representations, suggesting a natural fit for GMM-based cryo-EM reconstruction. However, off-the-shelf Gaussian splatting methods are designed for photorealistic view synthesis and remain incompatible with cryo-EM due to mismatches in the image formation physics, reconstruction objectives, and coordinate systems. Addressing these issues, we propose cryoSplat, a GMM-based method that integrates Gaussian splatting with the physics of cryo-EM image formation. In particular, we develop an orthogonal projection-aware Gaussian splatting, with adaptations such as a view-dependent normalization term and FFT-aligned coordinate system tailored for cryo-EM imaging. These innovations enable stable and efficient homogeneous reconstruction directly from raw cryo-EM particle images using random initialization. Experimental results on real datasets validate the effectiveness and robustness of cryoSplat over representative baselines. The code will be released at https://github.com/Chen-Suyi/cryosplat.
    </details>
</div>
