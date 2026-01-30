# gaussian splatting - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22046v1">PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Streaming reconstruction from monocular image sequences remains challenging, as existing methods typically favor either high-quality rendering or accurate geometry, but rarely both. We present PLANING, an efficient on-the-fly reconstruction framework built on a hybrid representation that loosely couples explicit geometric primitives with neural Gaussians, enabling geometry and appearance to be modeled in a decoupled manner. This decoupling supports an online initialization and optimization strategy that separates geometry and appearance updates, yielding stable streaming reconstruction with substantially reduced structural redundancy. PLANING improves dense mesh Chamfer-L2 by 18.52% over PGSR, surpasses ARTDECO by 1.31 dB PSNR, and reconstructs ScanNetV2 scenes in under 100 seconds, over 5x faster than 2D Gaussian Splatting, while matching the quality of offline per-scene optimization. Beyond reconstruction quality, the structural clarity and computational efficiency of \modelname~make it well suited for a broad range of downstream applications, such as enabling large-scale scene modeling and simulation-ready environments for embodied AI. Project page: https://city-super.github.io/PLANING/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22026v1">Hybrid Foveated Path Tracing with Peripheral Gaussians for Immersive Anatomy</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Scheduled for publication in the Proceedings of IEEE VR 2026
    </div>
    <details class="paper-abstract">
      Volumetric medical imaging offers great potential for understanding complex pathologies. Yet, traditional 2D slices provide little support for interpreting spatial relationships, forcing users to mentally reconstruct anatomy into three dimensions. Direct volumetric path tracing and VR rendering can improve perception but are computationally expensive, while precomputed representations, like Gaussian Splatting, require planning ahead. Both approaches limit interactive use. We propose a hybrid rendering approach for high-quality, interactive, and immersive anatomical visualization. Our method combines streamed foveated path tracing with a lightweight Gaussian Splatting approximation of the periphery. The peripheral model generation is optimized with volume data and continuously refined using foveal renderings, enabling interactive updates. Depth-guided reprojection further improves robustness to latency and allows users to balance fidelity with refresh rate. We compare our method against direct path tracing and Gaussian Splatting. Our results highlight how their combination can preserve strengths in visual quality while re-generating the peripheral model in under a second, eliminating extensive preprocessing and approximations. This opens new options for interactive medical visualization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22718v2">Edge Collaborative Gaussian Splatting with Integrated Rendering and Communication</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 IEEE ICASSP, Barcelona, Spain, 2026
    </div>
    <details class="paper-abstract">
      Gaussian splatting (GS) struggles with degraded rendering quality on low-cost devices. To address this issue, we present edge collaborative GS (ECO-GS), where each user can switch between a local small GS model to guarantee timeliness and a remote large GS model to guarantee fidelity. However, deciding how to engage the large GS model is nontrivial, due to the interdependency between rendering requirements and resource conditions. To this end, we propose integrated rendering and communication (IRAC), which jointly optimizes collaboration status (i.e., deciding whether to engage large GS) and edge power allocation (i.e., enabling remote rendering) under communication constraints across different users by minimizing a newly-derived GS switching function. Despite the nonconvexity of the problem, we propose an efficient penalty majorization minimization (PMM) algorithm to obtain the critical point solution. Furthermore, we develop an imitation learning optimization (ILO) algorithm, which reduces the computational time by over 100x compared to PMM. Experiments demonstrate the superiority of PMM and the real-time execution capability of ILO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2401.08742v5">Efficient4D: Fast Dynamic 3D Object Generation from a Single-view Video</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 IJCV version
    </div>
    <details class="paper-abstract">
      Generating dynamic 3D object from a single-view video is challenging due to the lack of 4D labeled data. An intuitive approach is to extend previous image-to-3D pipelines by transferring off-the-shelf image generation models such as score distillation sampling.However, this approach would be slow and expensive to scale due to the need for back-propagating the information-limited supervision signals through a large pretrained model. To address this, we propose an efficient video-to-4D object generation framework called Efficient4D. It generates high-quality spacetime-consistent images under different camera views, and then uses them as labeled data to directly reconstruct the 4D content through a 4D Gaussian splatting model. Importantly, our method can achieve real-time rendering under continuous camera trajectories. To enable robust reconstruction under sparse views, we introduce inconsistency-aware confidence-weighted loss design, along with a lightly weighted score distillation loss. Extensive experiments on both synthetic and real videos show that Efficient4D offers a remarkable 10-fold increase in speed when compared to prior art alternatives while preserving the quality of novel view synthesis. For example, Efficient4D takes only 10 minutes to model a dynamic object, vs 120 minutes by the previous art model Consistent4D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07052v2">RAVE: Rate-Adaptive Visual Encoding for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Recent advances in neural scene representations have transformed immersive multimedia, with 3D Gaussian Splatting (3DGS) enabling real-time photorealistic rendering. Despite its efficiency, 3DGS suffers from large memory requirements and costly training procedures, motivating efforts toward compression. Existing approaches, however, operate at fixed rates, limiting adaptability to varying bandwidth and device constraints. In this work, we propose a flexible compression scheme for 3DGS that supports interpolation at any rate between predefined bounds. Our method is computationally lightweight, requires no retraining for any rate, and preserves rendering quality across a broad range of operating points. Experiments demonstrate that the approach achieves efficient, high-quality compression while offering dynamic rate control, making it suitable for practical deployment in immersive applications. The code is available at https://github.com/inspiros/RAVE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.05600v2">Progressively Deformable 2D Gaussian Splatting for Video Representation at Arbitrary Resolutions</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Implicit neural representations (INRs) enable fast video compression and effective video processing, but a single model rarely offers scalable decoding across rates and resolutions. In practice, multi-resolution typically relies on retraining or multi-branch designs, and structured pruning failed to provide a permutation-invariant progressive transmission order. Motivated by the explicit structure and efficiency of Gaussian splatting, we propose D2GV-AR, a deformable 2D Gaussian video representation that enables \emph{arbitrary-scale} rendering and \emph{any-ratio} progressive coding within a single model. We partition each video into fixed-length Groups of Pictures and represent each group with a canonical set of 2D Gaussian primitives, whose temporal evolution is modeled by a neural ordinary differential equation. During training and rendering, we apply scale-aware grouping according to Nyquist sampling theorem to form a nested hierarchy across resolutions. Once trained, primitives can be pruned via a D-optimal subset objective to enable any-ratio progressive coding. Extensive experiments show that D2GV-AR renders at over 250 FPS while matching or surpassing recent INR baselines, enabling multiscale continuous rate--quality adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20857v1">FreeFix: Boosting 3D Gaussian Splatting via Fine-Tuning-Free Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 Our project page is at https://xdimlab.github.io/freefix
    </div>
    <details class="paper-abstract">
      Neural Radiance Fields and 3D Gaussian Splatting have advanced novel view synthesis, yet still rely on dense inputs and often degrade at extrapolated views. Recent approaches leverage generative models, such as diffusion models, to provide additional supervision, but face a trade-off between generalization and fidelity: fine-tuning diffusion models for artifact removal improves fidelity but risks overfitting, while fine-tuning-free methods preserve generalization but often yield lower fidelity. We introduce FreeFix, a fine-tuning-free approach that pushes the boundary of this trade-off by enhancing extrapolated rendering with pretrained image diffusion models. We present an interleaved 2D-3D refinement strategy, showing that image diffusion models can be leveraged for consistent refinement without relying on costly video diffusion models. Furthermore, we take a closer look at the guidance signal for 2D refinement and propose a per-pixel confidence mask to identify uncertain regions for targeted improvement. Experiments across multiple datasets show that FreeFix improves multi-frame consistency and achieves performance comparable to or surpassing fine-tuning-based methods, while retaining strong generalization ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.14277v3">Dense-SfM: Structure from Motion with Dense Consistent Matching</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      We present Dense-SfM, a novel Structure from Motion (SfM) framework designed for dense and accurate 3D reconstruction from multi-view images. Sparse keypoint matching, which traditional SfM methods often rely on, limits both accuracy and point density, especially in texture-less areas. Dense-SfM addresses this limitation by integrating dense matching with a Gaussian Splatting (GS) based track extension which gives more consistent, longer feature tracks. To further improve reconstruction accuracy, Dense-SfM is equipped with a multi-view kernelized matching module leveraging transformer and Gaussian Process architectures, for robust track refinement across multi-views. Evaluations on the ETH3D and Texture-Poor SfM datasets show that Dense-SfM offers significant improvements in accuracy and density over state-of-the-art methods. Project page: https://icetea-cv.github.io/densesfm/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.19073v3">WaveletGaussian: Wavelet-domain Diffusion for Sparse-view 3D Gaussian Object Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 Accepted to ICASSP 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a powerful representation for image-based object reconstruction, yet its performance drops sharply in sparse-view settings. Prior works address this limitation by employing diffusion models to repair corrupted renders, subsequently using them as pseudo ground truths for later optimization. While effective, such approaches incur heavy computation from the diffusion fine-tuning and repair steps. We present WaveletGaussian, a framework for more efficient sparse-view 3D Gaussian object reconstruction. Our key idea is to shift diffusion into the wavelet domain: diffusion is applied only to the low-resolution LL subband, while high-frequency subbands are refined with a lightweight network. We further propose an efficient online random masking strategy to curate training pairs for diffusion fine-tuning, replacing the commonly used, but inefficient, leave-one-out strategy. Experiments across two benchmark datasets, Mip-NeRF 360 and OmniObject3D, show WaveletGaussian achieves competitive rendering quality while substantially reducing training time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20429v1">GRTX: Efficient Ray Tracing for 3D Gaussian-Based Rendering</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 To appear at the 32nd International Symposium on High-Performance Computer Architecture (HPCA 2026)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has gained widespread adoption across diverse applications due to its exceptional rendering performance and visual quality. While most existing methods rely on rasterization to render Gaussians, recent research has started investigating ray tracing approaches to overcome the fundamental limitations inherent in rasterization. However, current Gaussian ray tracing methods suffer from inefficiencies such as bloated acceleration structures and redundant node traversals, which greatly degrade ray tracing performance. In this work, we present GRTX, a set of software and hardware optimizations that enable efficient ray tracing for 3D Gaussian-based rendering. First, we introduce a novel approach for constructing streamlined acceleration structures for Gaussian primitives. Our key insight is that anisotropic Gaussians can be treated as unit spheres through ray space transformations, which substantially reduces BVH size and traversal overhead. Second, we propose dedicated hardware support for traversal checkpointing within ray tracing units. This eliminates redundant node visits during multi-round tracing by resuming traversal from checkpointed nodes rather than restarting from the root node in each subsequent round. Our evaluation shows that GRTX significantly improves ray tracing performance compared to the baseline ray tracing method with a negligible hardware cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19489v2">Fast Converging 3D Gaussian Splatting for 1-Minute Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 First Rank of SIGGRAPH Asia 2025 3DGS Challenge. Code available at https://github.com/will-zzy/siggraph_asia
    </div>
    <details class="paper-abstract">
      We present a fast 3DGS reconstruction pipeline designed to converge within one minute, developed for the SIGGRAPH Asia 3DGS Fast Reconstruction Challenge. The challenge consists of an initial round using SLAM-generated camera poses (with noisy trajectories) and a final round using COLMAP poses (highly accurate). To robustly handle these heterogeneous settings, we develop a two-stage solution. In the first round, we use reverse per-Gaussian parallel optimization and compact forward splatting based on Taming-GS and Speedy-splat, load-balanced tiling, an anchor-based Neural-Gaussian representation enabling rapid convergence with fewer learnable parameters, initialization from monocular depth and partially from feed-forward 3DGS models, and a global pose refinement module for noisy SLAM trajectories. In the final round, the accurate COLMAP poses change the optimization landscape; we disable pose refinement, revert from Neural-Gaussians back to standard 3DGS to eliminate MLP inference overhead, introduce multi-view consistency-guided Gaussian splitting inspired by Fast-GS, and introduce a depth estimator to supervise the rendered depth. Together, these techniques enable high-fidelity reconstruction under a strict one-minute budget. Our method achieved the top performance with a PSNR of 28.43 and ranked first in the competition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20331v1">GVGS: Gaussian Visibility-Aware Multi-View Geometry for Accurate Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting enables efficient optimization and high-quality rendering, yet accurate surface reconstruction remains challenging. Prior methods improve surface reconstruction by refining Gaussian depth estimates, either via multi-view geometric consistency or through monocular depth priors. However, multi-view constraints become unreliable under large geometric discrepancies, while monocular priors suffer from scale ambiguity and local inconsistency, ultimately leading to inaccurate Gaussian depth supervision. To address these limitations, we introduce a Gaussian visibility-aware multi-view geometric consistency constraint that aggregates the visibility of shared Gaussian primitives across views, enabling more accurate and stable geometric supervision. In addition, we propose a progressive quadtree-calibrated Monocular depth constraint that performs block-wise affine calibration from coarse to fine spatial scales, mitigating the scale ambiguity of depth priors while preserving fine-grained surface details. Extensive experiments on DTU and TNT datasets demonstrate consistent improvements in geometric accuracy over prior Gaussian-based and implicit surface reconstruction methods. Codes are available at an anonymous repository: https://github.com/GVGScode/GVGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21853v3">Dynamic Novel View Synthesis in High Dynamic Range</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 It has been accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      High Dynamic Range Novel View Synthesis (HDR NVS) seeks to learn an HDR 3D model from Low Dynamic Range (LDR) training images captured under conventional imaging conditions. Current methods primarily focus on static scenes, implicitly assuming all scene elements remain stationary and non-living. However, real-world scenarios frequently feature dynamic elements, such as moving objects, varying lighting conditions, and other temporal events, thereby presenting a significantly more challenging scenario. To address this gap, we propose a more realistic problem named HDR Dynamic Novel View Synthesis (HDR DNVS), where the additional dimension ``Dynamic'' emphasizes the necessity of jointly modeling temporal radiance variations alongside sophisticated 3D translation between LDR and HDR. To tackle this complex, intertwined challenge, we introduce HDR-4DGS, a Gaussian Splatting-based architecture featured with an innovative dynamic tone-mapping module that explicitly connects HDR and LDR domains, maintaining temporal radiance coherence by dynamically adapting tone-mapping functions according to the evolving radiance distributions across the temporal dimension. As a result, HDR-4DGS achieves both temporal radiance consistency and spatially accurate color translation, enabling photorealistic HDR renderings from arbitrary viewpoints and time instances. Extensive experiments demonstrate that HDR-4DGS surpasses existing state-of-the-art methods in both quantitative performance and visual fidelity. Source code is available at https://github.com/prinasi/HDR-4DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17354v2">PocketGS: On-Device Training of 3D Gaussian Splatting for High Perceptual Modeling</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Efficient and high-fidelity 3D scene modeling is a long-standing pursuit in computer graphics. While recent 3D Gaussian Splatting (3DGS) methods achieve impressive real-time modeling performance, they rely on resource-unconstrained training assumptions that fail on mobile devices, which are limited by minute-scale training budgets and hardware-available peak-memory. We present PocketGS, a mobile scene modeling paradigm that enables on-device 3DGS training under these tightly coupled constraints while preserving high perceptual fidelity. Our method resolves the fundamental contradictions of standard 3DGS through three co-designed operators: G builds geometry-faithful point-cloud priors; I injects local surface statistics to seed anisotropic Gaussians, thereby reducing early conditioning gaps; and T unrolls alpha compositing with cached intermediates and index-mapped gradient scattering for stable mobile backpropagation. Collectively, these operators satisfy the competing requirements of training efficiency, memory compactness, and modeling fidelity. Extensive experiments demonstrate that PocketGS is able to outperform the powerful mainstream workstation 3DGS baseline to deliver high-quality reconstructions, enabling a fully on-device, practical capture-to-rendering workflow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19843v1">Graphical X Splatting (GraphiXS): A Graphical Model for 4D Gaussian Splatting under Uncertainty</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      We propose a new framework to systematically incorporate data uncertainty in Gaussian Splatting. Being the new paradigm of neural rendering, Gaussian Splatting has been investigated in many applications, with the main effort in extending its representation, improving its optimization process, and accelerating its speed. However, one orthogonal, much needed, but under-explored area is data uncertainty. In standard 4D Gaussian Splatting, data uncertainty can manifest as view sparsity, missing frames, camera asynchronization, etc. So far, there has been little research to holistically incorporating various types of data uncertainty under a single framework. To this end, we propose Graphical X Splatting, or GraphiXS, a new probabilistic framework that considers multiple types of data uncertainty, aiming for a fundamental augmentation of the current 4D Gaussian Splatting paradigm into a probabilistic setting. GraphiXS is general and can be instantiated with a range of primitives, e.g. Gaussians, Student's-t. Furthermore, GraphiXS can be used to `upgrade' existing methods to accommodate data uncertainty. Through exhaustive evaluation and comparison, we demonstrate that GraphiXS can systematically model various uncertainties in data, outperform existing methods in many settings where data are missing or polluted in space and time, and therefore is a major generalization of the current 4D Gaussian Splatting research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19753v1">WaterClear-GS: Optical-Aware Gaussian Splatting for Underwater Reconstruction and Restoration</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Underwater 3D reconstruction and appearance restoration are hindered by the complex optical properties of water, such as wavelength-dependent attenuation and scattering. Existing Neural Radiance Fields (NeRF)-based methods struggle with slow rendering speeds and suboptimal color restoration, while 3D Gaussian Splatting (3DGS) inherently lacks the capability to model complex volumetric scattering effects. To address these issues, we introduce WaterClear-GS, the first pure 3DGS-based framework that explicitly integrates underwater optical properties of local attenuation and scattering into Gaussian primitives, eliminating the need for an auxiliary medium network. Our method employs a dual-branch optimization strategy to ensure underwater photometric consistency while naturally recovering water-free appearances. This strategy is enhanced by depth-guided geometry regularization and perception-driven image loss, together with exposure constraints, spatially-adaptive regularization, and physically guided spectral regularization, which collectively enforce local 3D coherence and maintain natural visual perception. Experiments on standard benchmarks and our newly collected dataset demonstrate that WaterClear-GS achieves outstanding performance on both novel view synthesis (NVS) and underwater image restoration (UIR) tasks, while maintaining real-time rendering. The code will be available at https://buaaxrzhang.github.io/WaterClear-GS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17835v2">Geometry-Grounded Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 16 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has demonstrated impressive quality and efficiency in novel view synthesis. However, shape extraction from Gaussian primitives remains an open problem. Due to inadequate geometry parameterization and approximation, existing shape reconstruction methods suffer from poor multi-view consistency and are sensitive to floaters. In this paper, we present a rigorous theoretical derivation that establishes Gaussian primitives as a specific type of stochastic solids. This theoretical framework provides a principled foundation for Geometry-Grounded Gaussian Splatting by enabling the direct treatment of Gaussian primitives as explicit geometric representations. Using the volumetric nature of stochastic solids, our method efficiently renders high-quality depth maps for fine-grained geometry extraction. Experiments show that our method achieves the best shape reconstruction results among all Gaussian Splatting-based methods on public datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19489v1">Fast Converging 3D Gaussian Splatting for 1-Minute Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 First Rank of SIGGRAPH Asia 2025 3DGS Challenge. Code available at
    </div>
    <details class="paper-abstract">
      We present a fast 3DGS reconstruction pipeline designed to converge within one minute, developed for the SIGGRAPH Asia 3DGS Fast Reconstruction Challenge. The challenge consists of an initial round using SLAM-generated camera poses (with noisy trajectories) and a final round using COLMAP poses (highly accurate). To robustly handle these heterogeneous settings, we develop a two-stage solution. In the first round, we use reverse per-Gaussian parallel optimization and compact forward splatting based on Taming-GS and Speedy-splat, load-balanced tiling, an anchor-based Neural-Gaussian representation enabling rapid convergence with fewer learnable parameters, initialization from monocular depth and partially from feed-forward 3DGS models, and a global pose refinement module for noisy SLAM trajectories. In the final round, the accurate COLMAP poses change the optimization landscape; we disable pose refinement, revert from Neural-Gaussians back to standard 3DGS to eliminate MLP inference overhead, introduce multi-view consistency-guided Gaussian splitting inspired by Fast-GS, and introduce a depth estimator to supervise the rendered depth. Together, these techniques enable high-fidelity reconstruction under a strict one-minute budget. Our method achieved the top performance with a PSNR of 28.43 and ranked first in the competition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15766v2">LL-GaussianMap: Zero-shot Low-Light Image Enhancement via 2D Gaussian Splatting Guided Gain Maps</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Significant progress has been made in low-light image enhancement with respect to visual quality. However, most existing methods primarily operate in the pixel domain or rely on implicit feature representations. As a result, the intrinsic geometric structural priors of images are often neglected. 2D Gaussian Splatting (2DGS) has emerged as a prominent explicit scene representation technique characterized by superior structural fitting capabilities and high rendering efficiency. Despite these advantages, the utilization of 2DGS in low-level vision tasks remains unexplored. To bridge this gap, LL-GaussianMap is proposed as the first unsupervised framework incorporating 2DGS into low-light image enhancement. Distinct from conventional methodologies, the enhancement task is formulated as a gain map generation process guided by 2DGS primitives. The proposed method comprises two primary stages. First, high-fidelity structural reconstruction is executed utilizing 2DGS. Then, data-driven enhancement dictionary coefficients are rendered via the rasterization mechanism of Gaussian splatting through an innovative unified enhancement module. This design effectively incorporates the structural perception capabilities of 2DGS into gain map generation, thereby preserving edges and suppressing artifacts during enhancement. Additionally, the reliance on paired data is circumvented through unsupervised learning. Experimental results demonstrate that LL-GaussianMap achieves superior enhancement performance with an extremely low storage footprint, highlighting the effectiveness of explicit Gaussian representations for image enhancement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.09239v2">Gradient-Direction-Aware Density Control for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      The emergence of 3D Gaussian Splatting (3DGS) has significantly advanced Novel View Synthesis (NVS) through explicit scene representation, enabling real-time photorealistic rendering. However, existing approaches manifest two critical limitations in complex scenarios: (1) Over-reconstruction occurs when persistent large Gaussians cannot meet adaptive splitting thresholds during density control. This is exacerbated by conflicting gradient directions that prevent effective splitting of these Gaussians; (2) Over-densification of Gaussians occurs in regions with aligned gradient aggregation, leading to redundant component proliferation. This redundancy significantly increases memory overhead due to unnecessary data retention. We present Gradient-Direction-Aware Gaussian Splatting (GDAGS) to address these challenges. Our key innovations: the Gradient Coherence Ratio (GCR), computed through normalized gradient vector norms, which explicitly discriminates Gaussians with concordant versus conflicting gradient directions; and a nonlinear dynamic weighting mechanism leverages the GCR to enable gradient-direction-aware density control. Specifically, GDAGS prioritizes conflicting-gradient Gaussians during splitting operations to enhance geometric details while suppressing redundant concordant-direction Gaussians. Conversely, in cloning processes, GDAGS promotes concordant-direction Gaussian densification for structural completion while preventing conflicting-direction Gaussian overpopulation. Comprehensive evaluations across diverse real-world benchmarks demonstrate that GDAGS achieves superior rendering quality while effectively mitigating over-reconstruction, suppressing over-densification, and constructing compact scene representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19310v1">ClipGS-VR: Immersive and Interactive Cinematic Visualization of Volumetric Medical Data in Mobile Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 IEEE VR 2026 Posters
    </div>
    <details class="paper-abstract">
      High-fidelity cinematic medical visualization on mobile virtual reality (VR) remains challenging. Although ClipGS enables cross-sectional exploration via 3D Gaussian Splatting, it lacks arbitrary-angle slicing on consumer-grade VR headsets. To achieve real-time interactive performance, we introduce ClipGS-VR and restructure ClipGS's neural inference into a consolidated dataset, integrating high-fidelity layers from multiple pre-computed slicing states into a unified rendering structure. Our framework further supports arbitrary-angle slicing via gradient-based opacity modulation for smooth, visually coherent rendering. Evaluations confirm our approach maintains visual fidelity comparable to offline results while offering superior usability and interaction efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19247v1">TIGaussian: Disentangle Gaussians for Spatial-Awared Text-Image-3D Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      While visual-language models have profoundly linked features between texts and images, the incorporation of 3D modality data, such as point clouds and 3D Gaussians, further enables pretraining for 3D-related tasks, e.g., cross-modal retrieval, zero-shot classification, and scene recognition. As challenges remain in extracting 3D modal features and bridging the gap between different modalities, we propose TIGaussian, a framework that harnesses 3D Gaussian Splatting (3DGS) characteristics to strengthen cross-modality alignment through multi-branch 3DGS tokenizer and modality-specific 3D feature alignment strategies. Specifically, our multi-branch 3DGS tokenizer decouples the intrinsic properties of 3DGS structures into compact latent representations, enabling more generalizable feature extraction. To further bridge the modality gap, we develop a bidirectional cross-modal alignment strategies: a multi-view feature fusion mechanism that leverages diffusion priors to resolve perspective ambiguity in image-3D alignment, while a text-3D projection module adaptively maps 3D features to text embedding space for better text-3D alignment. Extensive experiments on various datasets demonstrate the state-of-the-art performance of TIGaussian in multiple tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19233v1">UniMGS: Unifying Mesh and 3D Gaussian Splatting with Single-Pass Rasterization and Proxy-Based Deformation</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 conference
    </div>
    <details class="paper-abstract">
      Joint rendering and deformation of mesh and 3D Gaussian Splatting (3DGS) have significant value as both representa tions offer complementary advantages for graphics applica tions. However, due to differences in representation and ren dering pipelines, existing studies render meshes and 3DGS separately, making it difficult to accurately handle occlusions and transparency. Moreover, the deformed 3DGS still suffers from visual artifacts due to the sensitivity to the topology quality of the proxy mesh. These issues pose serious obsta cles to the joint use of 3DGS and meshes, making it diffi cult to adapt 3DGS to conventional mesh-oriented graphics pipelines. We propose UniMGS, the first unified framework for rasterizing mesh and 3DGS in a single-pass anti-aliased manner, with a novel binding strategy for 3DGS deformation based on proxy mesh. Our key insight is to blend the col ors of both triangle and Gaussian fragments by anti-aliased α-blending in a single pass, achieving visually coherent re sults with precise handling of occlusion and transparency. To improve the visual appearance of the deformed 3DGS, our Gaussian-centric binding strategy employs a proxy mesh and spatially associates Gaussians with the mesh faces, signifi cantly reducing rendering artifacts. With these two compo nents, UniMGS enables the visualization and manipulation of 3D objects represented by mesh or 3DGS within a unified framework, opening up new possibilities in embodied AI, vir tual reality, and gaming. We will release our source code to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19216v1">Bridging Visual and Wireless Sensing: A Unified Radiation Field for 3D Radio Map Construction</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 The code for this work will be publicly available at: https://github.com/wenchaozheng/URF-GS
    </div>
    <details class="paper-abstract">
      The emerging applications of next-generation wireless networks (e.g., immersive 3D communication, low-altitude networks, and integrated sensing and communication) necessitate high-fidelity environmental intelligence. 3D radio maps have emerged as a critical tool for this purpose, enabling spectrum-aware planning and environment-aware sensing by bridging the gap between physical environments and electromagnetic signal propagation. However, constructing accurate 3D radio maps requires fine-grained 3D geometric information and a profound understanding of electromagnetic wave propagation. Existing approaches typically treat optical and wireless knowledge as distinct modalities, failing to exploit the fundamental physical principles governing both light and electromagnetic propagation. To bridge this gap, we propose URF-GS, a unified radio-optical radiation field representation framework for accurate and generalizable 3D radio map construction based on 3D Gaussian splatting (3D-GS) and inverse rendering. By fusing visual and wireless sensing observations, URF-GS recovers scene geometry and material properties while accurately predicting radio signal behavior at arbitrary transmitter-receiver (Tx-Rx) configurations. Experimental results demonstrate that URF-GS achieves up to a 24.7% improvement in spatial spectrum prediction accuracy and a 10x increase in sample efficiency for 3D radio map construction compared with neural radiance field (NeRF)-based methods. This work establishes a foundation for next-generation wireless networks by integrating perception, interaction, and communication through holistic radiation field reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03312v2">Universal Beta Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      We introduce Universal Beta Splatting (UBS), a unified framework that generalizes 3D Gaussian Splatting to N-dimensional anisotropic Beta kernels for explicit radiance field rendering. Unlike fixed Gaussian primitives, Beta kernels enable controllable dependency modeling across spatial, angular, and temporal dimensions within a single representation. Our unified approach captures complex light transport effects, handles anisotropic view-dependent appearance, and models scene dynamics without requiring auxiliary networks or specific color encodings. UBS maintains backward compatibility by approximating to Gaussian Splatting as a special case, guaranteeing plug-in usability and lower performance bounds. The learned Beta parameters naturally decompose scene properties into interpretable without explicit supervision: spatial (surface vs. texture), angular (diffuse vs. specular), and temporal (static vs. dynamic). Our CUDA-accelerated implementation achieves real-time rendering while consistently outperforming existing methods across static, view-dependent, and dynamic benchmarks, establishing Beta kernels as a scalable universal primitive for radiance field rendering. Our project website is available at https://rongliu-leo.github.io/universal-beta-splatting/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02103v2">HeadLighter: Disentangling Illumination in Generative 3D Gaussian Heads via Lightstage Captures</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Recent 3D-aware head generative models based on 3D Gaussian Splatting achieve real-time, photorealistic and view-consistent head synthesis. However, a fundamental limitation persists: the deep entanglement of illumination and intrinsic appearance prevents controllable relighting. Existing disentanglement methods rely on strong assumptions to enable weakly supervised learning, which restricts their capacity for complex illumination. To address this challenge, we introduce HeadLighter, a novel supervised framework that learns a physically plausible decomposition of appearance and illumination in head generative models. Specifically, we design a dual-branch architecture that separately models lighting-invariant head attributes and physically grounded rendering components. A progressive disentanglement training is employed to gradually inject head appearance priors into the generative architecture, supervised by multi-view images captured under controlled light conditions with a light stage setup. We further introduce a distillation strategy to generate high-quality normals for realistic rendering. Experiments demonstrate that our method preserves high-quality generation and real-time rendering, while simultaneously supporting explicit lighting and viewpoint editing. We will publicly release our code and dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18633v1">Splat-Portrait: Generalizing Talking Heads with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Talking Head Generation aims at synthesizing natural-looking talking videos from speech and a single portrait image. Previous 3D talking head generation methods have relied on domain-specific heuristics such as warping-based facial motion representation priors to animate talking motions, yet still produce inaccurate 3D avatar reconstructions, thus undermining the realism of generated animations. We introduce Splat-Portrait, a Gaussian-splatting-based method that addresses the challenges of 3D head reconstruction and lip motion synthesis. Our approach automatically learns to disentangle a single portrait image into a static 3D reconstruction represented as static Gaussian Splatting, and a predicted whole-image 2D background. It then generates natural lip motion conditioned on input audio, without any motion driven priors. Training is driven purely by 2D reconstruction and score-distillation losses, without 3D supervision nor landmarks. Experimental results demonstrate that Splat-Portrait exhibits superior performance on talking head generation and novel view synthesis, achieving better visual quality compared to previous works. Our project code and supplementary documents are public available at https://github.com/stonewalking/Splat-portrait.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18629v1">ExoGS: A 4D Real-to-Sim-to-Real Framework for Scalable Manipulation Data Collection</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Real-to-Sim-to-Real technique is gaining increasing interest for robotic manipulation, as it can generate scalable data in simulation while having narrower sim-to-real gap. However, previous methods mainly focused on environment-level visual real-to-sim transfer, ignoring the transfer of interactions, which could be challenging and inefficient to obtain purely in simulation especially for contact-rich tasks. We propose ExoGS, a robot-free 4D Real-to-Sim-to-Real framework that captures both static environments and dynamic interactions in the real world and transfers them seamlessly to a simulated environment. It provides a new solution for scalable manipulation data collection and policy learning. ExoGS employs a self-designed robot-isomorphic passive exoskeleton AirExo-3 to capture kinematically consistent trajectories with millimeter-level accuracy and synchronized RGB observations during direct human demonstrations. The robot, objects, and environment are reconstructed as editable 3D Gaussian Splatting assets, enabling geometry-consistent replay and large-scale data augmentation. Additionally, a lightweight Mask Adapter injects instance-level semantics into the policy to enhance robustness under visual domain shifts. Real-world experiments demonstrate that ExoGS significantly improves data efficiency and policy generalization compared to teleoperation-based baselines. Code and hardware files have been released on https://github.com/zaixiabalala/ExoGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18475v1">LoD-Structured 3D Gaussian Splatting for Streaming Video Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Free-Viewpoint Video (FVV) reconstruction enables photorealistic and interactive 3D scene visualization; however, real-time streaming is often bottlenecked by sparse-view inputs, prohibitive training costs, and bandwidth constraints. While recent 3D Gaussian Splatting (3DGS) has advanced FVV due to its superior rendering speed, Streaming Free-Viewpoint Video (SFVV) introduces additional demands for rapid optimization, high-fidelity reconstruction under sparse constraints, and minimal storage footprints. To bridge this gap, we propose StreamLoD-GS, an LoD-based Gaussian Splatting framework designed specifically for SFVV. Our approach integrates three core innovations: 1) an Anchor- and Octree-based LoD-structured 3DGS with a hierarchical Gaussian dropout technique to ensure efficient and stable optimization while maintaining high-quality rendering; 2) a GMM-based motion partitioning mechanism that separates dynamic and static content, refining dynamic regions while preserving background stability; and 3) a quantized residual refinement framework that significantly reduces storage requirements without compromising visual fidelity. Extensive experiments demonstrate that StreamLoD-GS achieves competitive or state-of-the-art performance in terms of quality, efficiency, and storage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16736v2">A Step to Decouple Optimization in 3DGS</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for real-time novel view synthesis. As an explicit representation optimized through gradient propagation among primitives, optimization widely accepted in deep neural networks (DNNs) is actually adopted in 3DGS, such as synchronous weight updating and Adam with the adaptive gradient. However, considering the physical significance and specific design in 3DGS, there are two overlooked details in the optimization of 3DGS: (i) update step coupling, which induces optimizer state rescaling and costly attribute updates outside the viewpoints, and (ii) gradient coupling in the moment, which may lead to under- or over-effective regularization. Nevertheless, such a complex coupling is under-explored. After revisiting the optimization of 3DGS, we take a step to decouple it and recompose the process into: Sparse Adam, Re-State Regularization and Decoupled Attribute Regularization. Taking a large number of experiments under the 3DGS and 3DGS-MCMC frameworks, our work provides a deeper understanding of these components. Finally, based on the empirical analysis, we re-design the optimization and propose AdamW-GS by re-coupling the beneficial components, under which better optimization efficiency and representation effectiveness are achieved simultaneously.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17835v1">Geometry-Grounded Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 16 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has demonstrated impressive quality and efficiency in novel view synthesis. However, shape extraction from Gaussian primitives remains an open problem. Due to inadequate geometry parameterization and approximation, existing shape reconstruction methods suffer from poor multi-view consistency and are sensitive to floaters. In this paper, we present a rigorous theoretical derivation that establishes Gaussian primitives as a specific type of stochastic solids. This theoretical framework provides a principled foundation for Geometry-Grounded Gaussian Splatting by enabling the direct treatment of Gaussian primitives as explicit geometric representations. Using the volumetric nature of stochastic solids, our method efficiently renders high-quality depth maps for fine-grained geometry extraction. Experiments show that our method achieves the best shape reconstruction results among all Gaussian Splatting-based methods on public datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08710v4">SceneSplat++: A Large Dataset and Comprehensive Benchmark for Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 15 pages, codes, data and benchmark are released at https://scenesplatpp.gaussianworld.ai/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) serves as a highly performant and efficient encoding of scene geometry, appearance, and semantics. Moreover, grounding language in 3D scenes has proven to be an effective strategy for 3D scene understanding. Current Language Gaussian Splatting line of work fall into three main groups: (i) per-scene optimization-based, (ii) per-scene optimization-free, and (iii) generalizable approach. However, most of them are evaluated only on rendered 2D views of a handful of scenes and viewpoints close to the training views, limiting ability and insight into holistic 3D understanding. To address this gap, we propose the first large-scale benchmark that systematically assesses these three groups of methods directly in 3D space, evaluating on 1060 scenes across three indoor datasets and one outdoor dataset. Benchmark results demonstrate a clear advantage of the generalizable paradigm, particularly in relaxing the scene-specific limitation, enabling fast feed-forward inference on novel scenes, and achieving superior segmentation performance. We further introduce GaussianWorld-49K a carefully curated 3DGS dataset comprising around 49K diverse indoor and outdoor scenes obtained from multiple sources, with which we demonstrate the generalizable approach could harness strong data priors. Our codes, benchmark, and datasets are released at https://scenesplatpp.gaussianworld.ai/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.12905v3">Matrix-free Second-order Optimization of Gaussian Splats with Residual Sampling</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is widely used for novel view synthesis due to its high rendering quality and fast inference time. However, 3DGS predominantly relies on first-order optimizers such as Adam, which leads to long training times. To address this limitation, we propose a novel second-order optimization strategy based on Levenberg-Marquardt (LM) and Conjugate Gradient (CG), specifically tailored towards Gaussian Splatting. Our key insight is that the Jacobian in 3DGS exhibits significant sparsity since each Gaussian affects only a limited number of pixels. We exploit this sparsity by proposing a matrix-free and GPU-parallelized LM optimization. To further improve its efficiency, we propose sampling strategies for both camera views and loss function and, consequently, the normal equation, significantly reducing the computational complexity. In addition, we increase the convergence rate of the second-order approximation by introducing an effective heuristic to determine the learning rate that avoids the expensive computation cost of line search methods. As a result, our method achieves a 4x speedup over standard LM and outperforms Adam by ~5x when the Gaussian count is low while providing ~1.3x speed in moderate counts. In addition, our matrix-free implementation achieves 2x speedup over the concurrent second-order optimizer 3DGS-LM, while using 3.5x less memory. Project Page: https://vcai.mpi-inf.mpg.de/projects/LM-RS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17720v1">Advancing Structured Priors for Sparse-Voxel Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Reconstructing accurate surfaces with radiance fields has progressed rapidly, yet two promising explicit representations, 3D Gaussian Splatting and sparse-voxel rasterization, exhibit complementary strengths and weaknesses. 3D Gaussian Splatting converges quickly and carries useful geometric priors, but surface fidelity is limited by its point-like parameterization. Sparse-voxel rasterization provides continuous opacity fields and crisp geometry, but its typical uniform dense-grid initialization slows convergence and underutilizes scene structure. We combine the advantages of both by introducing a voxel initialization method that places voxels at plausible locations and with appropriate levels of detail, yielding a strong starting point for per-scene optimization. To further enhance depth consistency without blurring edges, we propose refined depth geometry supervision that converts multi-view cues into direct per-ray depth regularization. Experiments on standard benchmarks demonstrate improvements over prior methods in geometric accuracy, better fine-structure recovery, and more complete surfaces, while maintaining fast convergence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12823v2">TreeDGS: Aerial Gaussian Splatting for Distant DBH Measurement</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Aerial remote sensing enables efficient large-area surveying, but accurate direct object-level measurement remains difficult in complex natural scenes. Recent advancements in 3D vision, particularly learned radiance-field representations such as NeRF and 3D Gaussian Splatting, have begun to raise the ceiling on reconstruction fidelity and densifiable geometry from posed imagery. Nevertheless, direct aerial measurement of important natural attributes such as tree diameter at breast height (DBH) remains challenging. Trunks in aerial forest scans are distant and sparsely observed in image views: at typical operating altitudes, stems may span only a few pixels. With these constraints, conventional reconstruction methods leave breast-height trunk geometry weakly constrained. We present TreeDGS, an aerial image reconstruction method that leverages 3D Gaussian Splatting as a continuous, densifiable scene representation for trunk measurement. After SfM--MVS initialization and Gaussian optimization, we extract a dense point set from the Gaussian field using RaDe-GS's depth-aware cumulative-opacity integration and associate each sample with a multi-view opacity reliability score. Then, we estimate DBH from trunk-isolated points using opacity-weighted solid-circle fitting. Evaluated on 10 plots with field-measured DBH, TreeDGS reaches 4.79,cm RMSE (about 2.6 pixels at this GSD) and outperforms a state-of-the-art LiDAR baseline (7.91,cm RMSE). This shows that TreeDGS can enable accurate, low-cost aerial DBH measurement
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.08305v5">ELECTRA: A Cartesian Network for 3D Charge Density Prediction with Floating Orbitals</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 10 pages, 4 figures, 5 tables, NeurIPS 2025
    </div>
    <details class="paper-abstract">
      We present the Electronic Tensor Reconstruction Algorithm (ELECTRA) - an equivariant model for predicting electronic charge densities using floating orbitals. Floating orbitals are a long-standing concept in the quantum chemistry community that promises more compact and accurate representations by placing orbitals freely in space, as opposed to centering all orbitals at the position of atoms. Finding the ideal placement of these orbitals requires extensive domain knowledge, though, which thus far has prevented widespread adoption. We solve this in a data-driven manner by training a Cartesian tensor network to predict the orbital positions along with orbital coefficients. This is made possible through a symmetry-breaking mechanism that is used to learn position displacements with lower symmetry than the input molecule while preserving the rotation equivariance of the charge density itself. Inspired by recent successes of Gaussian Splatting in representing densities in space, we are using Gaussian orbitals and predicting their weights and covariance matrices. Our method achieves a state-of-the-art balance between computational efficiency and predictive accuracy on established benchmarks. Furthermore, ELECTRA is able to lower the compute time required to arrive at converged DFT solutions - initializing calculations using our predicted densities yields an average 50.72 % reduction in self-consistent field (SCF) iterations on unseen molecules.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17354v1">PocketGS: On-Device Training of 3D Gaussian Splatting for High Perceptual Modeling</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Efficient and high-fidelity 3D scene modeling is a long-standing pursuit in computer graphics. While recent 3D Gaussian Splatting (3DGS) methods achieve impressive real-time modeling performance, they rely on resource-unconstrained training assumptions that fail on mobile devices, which are limited by minute-scale training budgets and hardware-available peak-memory. We present PocketGS, a mobile scene modeling paradigm that enables on-device 3DGS training under these tightly coupled constraints while preserving high perceptual fidelity. Our method resolves the fundamental contradictions of standard 3DGS through three co-designed operators: G builds geometry-faithful point-cloud priors; I injects local surface statistics to seed anisotropic Gaussians, thereby reducing early conditioning gaps; and T unrolls alpha compositing with cached intermediates and index-mapped gradient scattering for stable mobile backpropagation. Collectively, these operators satisfy the competing requirements of training efficiency, memory compactness, and modeling fidelity. Extensive experiments demonstrate that PocketGS is able to outperform the powerful mainstream workstation 3DGS baseline to deliver high-quality reconstructions, enabling a fully on-device, practical capture-to-rendering workflow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16736v1">A Step to Decouple Optimization in 3DGS</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for real-time novel view synthesis. As an explicit representation optimized through gradient propagation among primitives, optimization widely accepted in deep neural networks (DNNs) is actually adopted in 3DGS, such as synchronous weight updating and Adam with the adaptive gradient. However, considering the physical significance and specific design in 3DGS, there are two overlooked details in the optimization of 3DGS: (i) update step coupling, which induces optimizer state rescaling and costly attribute updates outside the viewpoints, and (ii) gradient coupling in the moment, which may lead to under- or over-effective regularization. Nevertheless, such a complex coupling is under-explored. After revisiting the optimization of 3DGS, we take a step to decouple it and recompose the process into: Sparse Adam, Re-State Regularization and Decoupled Attribute Regularization. Taking a large number of experiments under the 3DGS and 3DGS-MCMC frameworks, our work provides a deeper understanding of these components. Finally, based on the empirical analysis, we re-design the optimization and propose AdamW-GS by re-coupling the beneficial components, under which better optimization efficiency and representation effectiveness are achieved simultaneously.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16672v1">ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 15 pages, 8 figures, Submitted to CVPR 2026
    </div>
    <details class="paper-abstract">
      High-quality 3D garment reconstruction plays a crucial role in mitigating the sim-to-real gap in applications such as digital avatars, virtual try-on and robotic manipulation. However, existing garment reconstruction methods typically rely on unstructured representations, such as 3D Gaussian Splats, struggling to provide accurate reconstructions of garment topology and sewing structures. As a result, the reconstructed outputs are often unsuitable for high-fidelity physical simulation. We propose ReWeaver, a novel framework for topology-accurate 3D garment and sewing pattern reconstruction from sparse multi-view RGB images. Given as few as four input views, ReWeaver predicts seams and panels as well as their connectivities in both the 2D UV space and the 3D space. The predicted seams and panels align precisely with the multi-view images, yielding structured 2D--3D garment representations suitable for 3D perception, high-fidelity physical simulation, and robotic manipulation. To enable effective training, we construct a large-scale dataset GCD-TS, comprising multi-view RGB images, 3D garment geometries, textured human body meshes and annotated sewing patterns. The dataset contains over 100,000 synthetic samples covering a wide range of complex geometries and topologies. Extensive experiments show that ReWeaver consistently outperforms existing methods in terms of topology accuracy, geometry alignment and seam-panel consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17185v1">LGDWT-GS: Local and Global Discrete Wavelet-Regularized 3D Gaussian Splatting for Sparse-View Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      We propose a new method for few-shot 3D reconstruction that integrates global and local frequency regularization to stabilize geometry and preserve fine details under sparse-view conditions, addressing a key limitation of existing 3D Gaussian Splatting (3DGS) models. We also introduce a new multispectral greenhouse dataset containing four spectral bands captured from diverse plant species under controlled conditions. Alongside the dataset, we release an open-source benchmarking package that defines standardized few-shot reconstruction protocols for evaluating 3DGS-based methods. Experiments on our multispectral dataset, as well as standard benchmarks, demonstrate that the proposed method achieves sharper, more stable, and spectrally consistent reconstructions than existing baselines. The dataset and code for this work are publicly available
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17590v3">CGS-GAN: 3D Consistent Gaussian Splatting GANs for High Resolution Human Head Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Main paper 12 pages, supplementary materials 8 pages
    </div>
    <details class="paper-abstract">
      Recently, 3D GANs based on 3D Gaussian splatting have been proposed for high quality synthesis of human heads. However, existing methods stabilize training and enhance rendering quality from steep viewpoints by conditioning the random latent vector on the current camera position. This compromises 3D consistency, as we observe significant identity changes when re-synthesizing the 3D head with each camera shift. Conversely, fixing the camera to a single viewpoint yields high-quality renderings for that perspective but results in poor performance for novel views. Removing view-conditioning typically destabilizes GAN training, often causing the training to collapse. In response to these challenges, we introduce CGS-GAN, a novel 3D Gaussian Splatting GAN framework that enables stable training and high-quality 3D-consistent synthesis of human heads without relying on view-conditioning. To ensure training stability, we introduce a multi-view regularization technique that enhances generator convergence with minimal computational overhead. Additionally, we adapt the conditional loss used in existing 3D Gaussian splatting GANs and propose a generator architecture designed to not only stabilize training but also facilitate efficient rendering and straightforward scaling, enabling output resolutions up to $2048^2$. To evaluate the capabilities of CGS-GAN, we curate a new dataset derived from FFHQ. This dataset enables very high resolutions, focuses on larger portions of the human head, reduces view-dependent artifacts for improved 3D consistency, and excludes images where subjects are obscured by hands or other objects. As a result, our approach achieves very high rendering quality, supported by competitive FID scores, while ensuring consistent 3D scene generation. Check our our project page here: https://fraunhoferhhi.github.io/cgs-gan/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15951v1">EVolSplat4D: Efficient Volume-based Gaussian Splatting for 4D Urban Scene Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) of static and dynamic urban scenes is essential for autonomous driving simulation, yet existing methods often struggle to balance reconstruction time with quality. While state-of-the-art neural radiance fields and 3D Gaussian Splatting approaches achieve photorealism, they often rely on time-consuming per-scene optimization. Conversely, emerging feed-forward methods frequently adopt per-pixel Gaussian representations, which lead to 3D inconsistencies when aggregating multi-view predictions in complex, dynamic environments. We propose EvolSplat4D, a feed-forward framework that moves beyond existing per-pixel paradigms by unifying volume-based and pixel-based Gaussian prediction across three specialized branches. For close-range static regions, we predict consistent geometry of 3D Gaussians over multiple frames directly from a 3D feature volume, complemented by a semantically-enhanced image-based rendering module for predicting their appearance. For dynamic actors, we utilize object-centric canonical spaces and a motion-adjusted rendering module to aggregate temporal features, ensuring stable 4D reconstruction despite noisy motion priors. Far-Field scenery is handled by an efficient per-pixel Gaussian branch to ensure full-scene coverage. Experimental results on the KITTI-360, KITTI, Waymo, and PandaSet datasets show that EvolSplat4D reconstructs both static and dynamic environments with superior accuracy and consistency, outperforming both per-scene optimization and state-of-the-art feed-forward baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15897v1">ThermoSplat: Cross-Modal 3D Gaussian Splatting with Feature Modulation and Geometry Decoupling</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Multi-modal scene reconstruction integrating RGB and thermal infrared data is essential for robust environmental perception across diverse lighting and weather conditions. However, extending 3D Gaussian Splatting (3DGS) to multi-spectral scenarios remains challenging. Current approaches often struggle to fully leverage the complementary information of multi-modal data, typically relying on mechanisms that either tend to neglect cross-modal correlations or leverage shared representations that fail to adaptively handle the complex structural correlations and physical discrepancies between spectrums. To address these limitations, we propose ThermoSplat, a novel framework that enables deep spectral-aware reconstruction through active feature modulation and adaptive geometry decoupling. First, we introduce a Cross-Modal FiLM Modulation mechanism that dynamically conditions shared latent features on thermal structural priors, effectively guiding visible texture synthesis with reliable cross-modal geometric cues. Second, to accommodate modality-specific geometric inconsistencies, we propose a Modality-Adaptive Geometric Decoupling scheme that learns independent opacity offsets and executes an independent rasterization pass for the thermal branch. Additionally, a hybrid rendering pipeline is employed to integrate explicit Spherical Harmonics with implicit neural decoding, ensuring both semantic consistency and high-frequency detail preservation. Extensive experiments on the RGBT-Scenes dataset demonstrate that ThermoSplat achieves state-of-the-art rendering quality across both visible and thermal spectrums.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14510v2">Structured Image-based Coding for Efficient Gaussian Splatting Compression</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has recently emerged as a state-of-the-art representation for radiance fields, combining real-time rendering with high visual fidelity. However, GS models require storing millions of parameters, leading to large file sizes that impair their use in practical multimedia systems. To address this limitation, this paper introduces GS Image-based Compression (GSICO), a novel GS codec that efficiently compresses pre-trained GS models while preserving perceptual fidelity. The core contribution lies in a mapping procedure that arranges GS parameters into structured images, guided by a novel algorithm that enhances spatial coherence. These GS parameter images are then encoded using a conventional image codec. Experimental evaluations on Tanks and Temples, Deep Blending, and Mip-NeRF360 datasets show that GSICO achieves average compression factors of 20.2x with minimal loss in visual quality, as measured by PSNR, SSIM, and LPIPS. Compared with state-of-the-art GS compression methods, the proposed codec consistently yields superior rate-distortion (RD) trade-offs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.12787v3">Rasterizing Wireless Radiance Field via Deformable 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Modeling the wireless radiance field (WRF) is fundamental to modern communication systems, enabling key tasks such as localization, sensing, and channel estimation. Traditional approaches, which rely on empirical formulas or physical simulations, often suffer from limited accuracy or require strong scene priors. Recent neural radiance field (NeRF-based) methods improve reconstruction fidelity through differentiable volumetric rendering, but their reliance on computationally expensive multilayer perceptron (MLP) queries hinders real-time deployment. To overcome these challenges, we introduce Gaussian splatting (GS) to the wireless domain, leveraging its efficiency in modeling optical radiance fields to enable compact and accurate WRF reconstruction. Specifically, we propose SwiftWRF, a deformable 2D Gaussian splatting framework that synthesizes WRF spectra at arbitrary positions under single-sided transceiver mobility. SwiftWRF employs CUDA-accelerated rasterization to render spectra at over 100000 fps and uses a lightweight MLP to model the deformation of 2D Gaussians, effectively capturing mobility-induced WRF variations. In addition to novel spectrum synthesis, the efficacy of SwiftWRF is further underscored in its applications in angle-of-arrival (AoA) and received signal strength indicator (RSSI) prediction. Experiments conducted on both real-world and synthetic indoor scenes demonstrate that SwiftWRF can reconstruct WRF spectra up to 500x faster than existing state-of-the-art methods, while significantly enhancing its signal quality. The project page is https://evan-sudo.github.io/swiftwrf/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15772v1">LL-GaussianImage: Efficient Image Representation for Zero-shot Low-Light Enhancement with 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      2D Gaussian Splatting (2DGS) is an emerging explicit scene representation method with significant potential for image compression due to high fidelity and high compression ratios. However, existing low-light enhancement algorithms operate predominantly within the pixel domain. Processing 2DGS-compressed images necessitates a cumbersome decompression-enhancement-recompression pipeline, which compromises efficiency and introduces secondary degradation. To address these limitations, we propose LL-GaussianImage, the first zero-shot unsupervised framework designed for low-light enhancement directly within the 2DGS compressed representation domain. Three primary advantages are offered by this framework. First, a semantic-guided Mixture-of-Experts enhancement framework is designed. Dynamic adaptive transformations are applied to the sparse attribute space of 2DGS using rendered images as guidance to enable compression-as-enhancement without full decompression to a pixel grid. Second, a multi-objective collaborative loss function system is established to strictly constrain smoothness and fidelity during enhancement, suppressing artifacts while improving visual quality. Third, a two-stage optimization process is utilized to achieve reconstruction-as-enhancement. The accuracy of the base representation is ensured through single-scale reconstruction and network robustness is enhanced. High-quality enhancement of low-light images is achieved while high compression ratios are maintained. The feasibility and superiority of the paradigm for direct processing within the compressed representation domain are validated through experimental results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15766v1">LL-GaussianMap: Zero-shot Low-Light Image Enhancement via 2D Gaussian Splatting Guided Gain Maps</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Significant progress has been made in low-light image enhancement with respect to visual quality. However, most existing methods primarily operate in the pixel domain or rely on implicit feature representations. As a result, the intrinsic geometric structural priors of images are often neglected. 2D Gaussian Splatting (2DGS) has emerged as a prominent explicit scene representation technique characterized by superior structural fitting capabilities and high rendering efficiency. Despite these advantages, the utilization of 2DGS in low-level vision tasks remains unexplored. To bridge this gap, LL-GaussianMap is proposed as the first unsupervised framework incorporating 2DGS into low-light image enhancement. Distinct from conventional methodologies, the enhancement task is formulated as a gain map generation process guided by 2DGS primitives. The proposed method comprises two primary stages. First, high-fidelity structural reconstruction is executed utilizing 2DGS. Then, data-driven enhancement dictionary coefficients are rendered via the rasterization mechanism of Gaussian splatting through an innovative unified enhancement module. This design effectively incorporates the structural perception capabilities of 2DGS into gain map generation, thereby preserving edges and suppressing artifacts during enhancement. Additionally, the reliance on paired data is circumvented through unsupervised learning. Experimental results demonstrate that LL-GaussianMap achieves superior enhancement performance with an extremely low storage footprint, highlighting the effectiveness of explicit Gaussian representations for image enhancement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15283v1">LuxRemix: Lighting Decomposition and Remixing for Indoor Scenes</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Project page: https://luxremix.github.io
    </div>
    <details class="paper-abstract">
      We present a novel approach for interactive light editing in indoor scenes from a single multi-view scene capture. Our method leverages a generative image-based light decomposition model that factorizes complex indoor scene illumination into its constituent light sources. This factorization enables independent manipulation of individual light sources, specifically allowing control over their state (on/off), chromaticity, and intensity. We further introduce multi-view lighting harmonization to ensure consistent propagation of the lighting decomposition across all scene views. This is integrated into a relightable 3D Gaussian splatting representation, providing real-time interactive control over the individual light sources. Our results demonstrate highly photorealistic lighting decomposition and relighting outcomes across diverse indoor scenes. We evaluate our method on both synthetic and real-world datasets and provide a quantitative and qualitative comparison to state-of-the-art techniques. For video results and interactive demos, see https://luxremix.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14821v1">POTR: Post-Training 3DGS Compression</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 15 pages, 12 figures. Submitted to IEEE TCSVT, under review
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as a promising contender to Neural Radiance Fields (NeRF) in 3D scene reconstruction and real-time novel view synthesis. 3DGS outperforms NeRF in training and inference speed but has substantially higher storage requirements. To remedy this downside, we propose POTR, a post-training 3DGS codec built on two novel techniques. First, POTR introduces a novel pruning approach that uses a modified 3DGS rasterizer to efficiently calculate every splat's individual removal effect simultaneously. This technique results in 2-4x fewer splats than other post-training pruning techniques and as a result also significantly accelerates inference with experiments demonstrating 1.5-2x faster inference than other compressed models. Second, we propose a novel method to recompute lighting coefficients, significantly reducing their entropy without using any form of training. Our fast and highly parallel approach especially increases AC lighting coefficient sparsity, with experiments demonstrating increases from 70% to 97%, with minimal loss in quality. Finally, we extend POTR with a simple fine-tuning scheme to further enhance pruning, inference, and rate-distortion performance. Experiments demonstrate that POTR, even without fine-tuning, consistently outperforms all other post-training compression techniques in both rate-distortion performance and inference speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06810v2">ConeGS: Error-Guided Densification Using Pixel Cones for Improved Reconstruction With Fewer Primitives</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves state-of-the-art image quality and real-time performance in novel view synthesis but often suffers from a suboptimal spatial distribution of primitives. This issue stems from cloning-based densification, which propagates Gaussians along existing geometry, limiting exploration and requiring many primitives to adequately cover the scene. We present ConeGS, an image-space-informed densification framework that is independent of existing scene geometry state. ConeGS first creates a fast Instant Neural Graphics Primitives (iNGP) reconstruction as a geometric proxy to estimate per-pixel depth. During the subsequent 3DGS optimization, it identifies high-error pixels and inserts new Gaussians along the corresponding viewing cones at the predicted depth values, initializing their size according to the cone diameter. A pre-activation opacity penalty rapidly removes redundant Gaussians, while a primitive budgeting strategy controls the total number of primitives, either by a fixed budget or by adapting to scene complexity, ensuring high reconstruction quality. Experiments show that ConeGS consistently enhances reconstruction quality and rendering performance across Gaussian budgets, with especially strong gains under tight primitive constraints where efficient placement is crucial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.10860v2">RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 ICCV 2025, Project page: https://people.engr.tamu.edu/nimak/Papers/RI3D, Code: https://github.com/avinashpaliwal/RI3D
    </div>
    <details class="paper-abstract">
      In this paper, we propose RI3D, a novel 3DGS-based approach that harnesses the power of diffusion models to reconstruct high-quality novel views given a sparse set of input images. Our key contribution is separating the view synthesis process into two tasks of reconstructing visible regions and hallucinating missing regions, and introducing two personalized diffusion models, each tailored to one of these tasks. Specifically, one model ('repair') takes a rendered image as input and predicts the corresponding high-quality image, which in turn is used as a pseudo ground truth image to constrain the optimization. The other model ('inpainting') primarily focuses on hallucinating details in unobserved areas. To integrate these models effectively, we introduce a two-stage optimization strategy: the first stage reconstructs visible areas using the repair model, and the second stage reconstructs missing regions with the inpainting model while ensuring coherence through further optimization. Moreover, we augment the optimization with a novel Gaussian initialization method that obtains per-image depth by combining 3D-consistent and smooth depth with highly detailed relative depth. We demonstrate that by separating the process into two tasks and addressing them with the repair and inpainting models, we produce results with detailed textures in both visible and missing regions that outperform state-of-the-art approaches on a diverse set of scenes with extremely sparse inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15431v1">SplatBus: A Gaussian Splatting Viewer Framework via GPU Interprocess Communication</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Radiance field-based rendering methods have attracted significant interest from the computer vision and computer graphics communities. They enable high-fidelity rendering with complex real-world lighting effects, but at the cost of high rendering time. 3D Gaussian Splatting solves this issue with a rasterisation-based approach for real-time rendering, enabling applications such as autonomous driving, robotics, virtual reality, and extended reality. However, current 3DGS implementations are difficult to integrate into traditional mesh-based rendering pipelines, which is a common use case for interactive applications and artistic exploration. To address this limitation, this software solution uses Nvidia's interprocess communication (IPC) APIs to easily integrate into implementations and allow the results to be viewed in external clients such as Unity, Blender, Unreal Engine, and OpenGL viewers. The code is available at https://github.com/RockyXu66/splatbus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14208v1">Rig-Aware 3D Reconstruction of Vehicle Undercarriages using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 8 pages, 9 figures, Conference: IEEE International Conference on Machine Learning and Applications 2025 (ICMLA 2025): https://www.icmla-conference.org/icmla25/
    </div>
    <details class="paper-abstract">
      Inspecting the undercarriage of used vehicles is a labor-intensive task that requires inspectors to crouch or crawl underneath each vehicle to thoroughly examine it. Additionally, online buyers rarely see undercarriage photos. We present an end-to-end pipeline that utilizes a three-camera rig to capture videos of the undercarriage as the vehicle drives over it, and produces an interactive 3D model of the undercarriage. The 3D model enables inspectors and customers to rotate, zoom, and slice through the undercarriage, allowing them to detect rust, leaks, or impact damage in seconds, thereby improving both workplace safety and buyer confidence. Our primary contribution is a rig-aware Structure-from-Motion (SfM) pipeline specifically designed to overcome the challenges of wide-angle lens distortion and low-parallax scenes. Our method overcomes the challenges of wide-angle lens distortion and low-parallax scenes by integrating precise camera calibration, synchronized video streams, and strong geometric priors from the camera rig. We use a constrained matching strategy with learned components, the DISK feature extractor, and the attention-based LightGlue matcher to generate high-quality sparse point clouds that are often unattainable with standard SfM pipelines. These point clouds seed the Gaussian splatting process to generate photorealistic undercarriage models that render in real-time. Our experiments and ablation studies demonstrate that our design choices are essential to achieve state-of-the-art quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.10231v3">SuperGSeg: Open-Vocabulary 3D Segmentation with Structured Super-Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 13 pages, 8 figures. Project page: supergseg.github.io
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has recently gained traction for its efficient training and real-time rendering. While its vanilla representation is mainly designed for view synthesis, recent works extended it to scene understanding with language features. However, storing additional high-dimensional features per Gaussian for semantic information is memory-intensive, which limits their ability to segment and interpret challenging scenes. To this end, we introduce SuperGSeg, a novel approach that fosters cohesive, context-aware hierarchical scene representation by disentangling segmentation and language field distillation. SuperGSeg first employs neural 3D Gaussians to learn geometry, instance and hierarchical segmentation features from multi-view images with the aid of off-the-shelf 2D masks. These features are then leveraged to create a sparse set of \acrlong{superg}s. \acrlong{superg}s facilitate the lifting and distillation of 2D language features into 3D space. They enable hierarchical scene understanding with high-dimensional language feature rendering at moderate GPU memory costs. Extensive experiments demonstrate that SuperGSeg achieves remarkable performance on both open-vocabulary object selection and semantic segmentation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14161v1">One-Shot Refiner: Boosting Feed-forward Novel View Synthesis via One-Step Diffusion</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      We present a novel framework for high-fidelity novel view synthesis (NVS) from sparse images, addressing key limitations in recent feed-forward 3D Gaussian Splatting (3DGS) methods built on Vision Transformer (ViT) backbones. While ViT-based pipelines offer strong geometric priors, they are often constrained by low-resolution inputs due to computational costs. Moreover, existing generative enhancement methods tend to be 3D-agnostic, resulting in inconsistent structures across views, especially in unseen regions. To overcome these challenges, we design a Dual-Domain Detail Perception Module, which enables handling high-resolution images without being limited by the ViT backbone, and endows Gaussians with additional features to store high-frequency details. We develop a feature-guided diffusion network, which can preserve high-frequency details during the restoration process. We introduce a unified training strategy that enables joint optimization of the ViT-based geometric backbone and the diffusion-based refinement module. Experiments demonstrate that our method can maintain superior generation quality across multiple datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03824v2">IDESplat: Iterative Depth Probability Estimation for Generalizable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Generalizable 3D Gaussian Splatting aims to directly predict Gaussian parameters using a feed-forward network for scene reconstruction. Among these parameters, Gaussian means are particularly difficult to predict, so depth is usually estimated first and then unprojected to obtain the Gaussian sphere centers. Existing methods typically rely solely on a single warp to estimate depth probability, which hinders their ability to fully leverage cross-view geometric cues, resulting in unstable and coarse depth maps. To address this limitation, we propose IDESplat, which iteratively applies warp operations to boost depth probability estimation for accurate Gaussian mean prediction. First, to eliminate the inherent instability of a single warp, we introduce a Depth Probability Boosting Unit (DPBU) that integrates epipolar attention maps produced by cascading warp operations in a multiplicative manner. Next, we construct an iterative depth estimation process by stacking multiple DPBUs, progressively identifying potential depth candidates with high likelihood. As IDESplat iteratively boosts depth probability estimates and updates the depth candidates, the depth map is gradually refined, resulting in accurate Gaussian means. We conduct experiments on RealEstate10K, ACID, and DL3DV. IDESplat achieves outstanding reconstruction quality and state-of-the-art performance with real-time efficiency. On RE10K, it outperforms DepthSplat by 0.33 dB in PSNR, using only 10.7% of the parameters and 70% of the memory. Additionally, our IDESplat improves PSNR by 2.95 dB over DepthSplat on the DTU dataset in cross-dataset experiments, demonstrating its strong generalization ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13706v1">ParkingTwin: Training-Free Streaming 3D Reconstruction for Parking-Lot Digital Twins</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 35 pages, 10 figures. Submitted to ISPRS Journal of Photogrammetry and Remote Sensing. Under review
    </div>
    <details class="paper-abstract">
      High-fidelity parking-lot digital twins provide essential priors for path planning, collision checking, and perception validation in Automated Valet Parking (AVP). Yet robot-oriented reconstruction faces a trilemma: sparse forward-facing views cause weak parallax and ill-posed geometry; dynamic occlusions and extreme lighting hinder stable texture fusion; and neural rendering typically needs expensive offline optimization, violating edge-side streaming constraints. We propose ParkingTwin, a training-free, lightweight system for online streaming 3D reconstruction. First, OSM-prior-driven geometric construction uses OpenStreetMap semantic topology to directly generate a metric-consistent TSDF, replacing blind geometric search with deterministic mapping and avoiding costly optimization. Second, geometry-aware dynamic filtering employs a quad-modal constraint field (normal/height/depth consistency) to reject moving vehicles and transient occlusions in real time. Third, illumination-robust fusion in CIELAB decouples luminance and chromaticity via adaptive L-channel weighting and depth-gradient suppression, reducing seams under abrupt lighting changes. ParkingTwin runs at 30+ FPS on an entry-level GTX 1660. On a 68,000 m^2 real-world dataset, it achieves SSIM 0.87 (+16.0%), delivers about 15x end-to-end speedup, and reduces GPU memory by 83.3% compared with state-of-the-art 3D Gaussian Splatting (3DGS) that typically requires high-end GPUs (RTX 4090D). The system outputs explicit triangle meshes compatible with Unity/Unreal digital-twin pipelines. Project page: https://mihoutao-liu.github.io/ParkingTwin/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.06424v3">Deblur4DGS: 4D Gaussian Splatting from Blurry Monocular Video</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Recent 4D reconstruction methods have yielded impressive results but rely on sharp videos as supervision. However, motion blur often occurs in videos due to camera shake and object movement, while existing methods render blurry results when using such videos for reconstructing 4D models. Although a few approaches attempted to address the problem, they struggled to produce high-quality results, due to the inaccuracy in estimating continuous dynamic representations within the exposure time. Encouraged by recent works in 3D motion trajectory modeling using 3D Gaussian Splatting (3DGS), we take 3DGS as the scene representation manner, and propose Deblur4DGS to reconstruct a high-quality 4D model from blurry monocular video. Specifically, we transform continuous dynamic representations estimation within an exposure time into the exposure time estimation. Moreover, we introduce the exposure regularization term, multi-frame, and multi-resolution consistency regularization term to avoid trivial solutions. Furthermore, to better represent objects with large motion, we suggest blur-aware variable canonical Gaussians. Beyond novel-view synthesis, Deblur4DGS can be applied to improve blurry video from multiple perspectives, including deblurring, frame interpolation, and video stabilization. Extensive experiments in both synthetic and real-world data on the above four tasks show that Deblur4DGS outperforms state-of-the-art 4D reconstruction methods. The codes are available at https://github.com/ZcsrenlongZ/Deblur4DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.13948v2">Light4GS: Lightweight Compact 4D Gaussian Splatting Generation via Context Model</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as an efficient and high-fidelity paradigm for novel view synthesis. To adapt 3DGS for dynamic content, deformable 3DGS incorporates temporally deformable primitives with learnable latent embeddings to capture complex motions. Despite its impressive performance, the high-dimensional embeddings and vast number of primitives lead to substantial storage requirements. In this paper, we introduce a \textbf{Light}weight \textbf{4}D\textbf{GS} framework, called Light4GS, that employs significance pruning with a deep context model to provide a lightweight storage-efficient dynamic 3DGS representation. The proposed Light4GS is based on 4DGS that is a typical representation of deformable 3DGS. Specifically, our framework is built upon two core components: (1) a spatio-temporal significance pruning strategy that eliminates over 64\% of the deformable primitives, followed by an entropy-constrained spherical harmonics compression applied to the remainder; and (2) a deep context model that integrates intra- and inter-prediction with hyperprior into a coarse-to-fine context structure to enable efficient multiscale latent embedding compression. Our approach achieves over 120x compression and increases rendering FPS up to 20\% compared to the baseline 4DGS, and also superior to frame-wise state-of-the-art 3DGS compression methods, revealing the effectiveness of our Light4GS in terms of both intra- and inter-prediction methods without sacrificing rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14510v1">Structured Image-based Coding for Efficient Gaussian Splatting Compression</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has recently emerged as a state-of-the-art representation for radiance fields, combining real-time rendering with high visual fidelity. However, GS models require storing millions of parameters, leading to large file sizes that impair their use in practical multimedia systems. To address this limitation, this paper introduces GS Image-based Compression (GSICO), a novel GS codec that efficiently compresses pre-trained GS models while preserving perceptual fidelity. The core contribution lies in a mapping procedure that arranges GS parameters into structured images, guided by a novel algorithm that enhances spatial coherence. These GS parameter images are then encoded using a conventional image codec. Experimental evaluations on Tanks and Temples, Deep Blending, and Mip-NeRF360 datasets show that GSICO achieves average compression factors of 20.2x with minimal loss in visual quality, as measured by PSNR, SSIM, and LPIPS. Compared with state-of-the-art GS compression methods, the proposed codec consistently yields superior rate-distortion (RD) trade-offs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04984v2">OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 Accepted to AAAI 2026. Project page: https://oceansplat.github.io
    </div>
    <details class="paper-abstract">
      We introduce OceanSplat, a novel 3D Gaussian Splatting-based approach for high-fidelity underwater scene reconstruction. To overcome multi-view inconsistencies caused by scattering media, we design a trinocular setup for each camera pose by rendering from horizontally and vertically translated virtual viewpoints, enforcing view consistency to facilitate spatial optimization of 3D Gaussians. Furthermore, we derive synthetic epipolar depth priors from the virtual viewpoints, which serve as self-supervised depth regularizers to compensate for the limited geometric cues in degraded underwater scenes. We also propose a depth-aware alpha adjustment that modulates the opacity of 3D Gaussians during early training based on their depth along the viewing direction, deterring the formation of medium-induced primitives. Our approach promotes the disentanglement of 3D Gaussians from the scattering medium through effective geometric constraints, enabling accurate representation of scene structure and significantly reducing floating artifacts. Experiments on real-world underwater and simulated scenes demonstrate that OceanSplat substantially outperforms existing methods for both scene reconstruction and restoration in scattering media.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13132v1">GaussExplorer: 3D Gaussian Splatting for Embodied Exploration and Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 Project page: https://gaussexplorer.github.io/
    </div>
    <details class="paper-abstract">
      We present GaussExplorer, a framework for embodied exploration and reasoning built on 3D Gaussian Splatting (3DGS). While prior approaches to language-embedded 3DGS have made meaningful progress in aligning simple text queries with Gaussian embeddings, they are generally optimized for relatively simple queries and struggle to interpret more complex, compositional language queries. Alternative studies based on object-centric RGB-D structured memories provide spatial grounding but are constrained by pre-fixed viewpoints. To address these issues, GaussExplorer introduces Vision-Language Models (VLMs) on top of 3DGS to enable question-driven exploration and reasoning within 3D scenes. We first identify pre-captured images that are most correlated with the query question, and subsequently adjust them into novel viewpoints to more accurately capture visual information for better reasoning by VLMs. Experiments show that ours outperforms existing methods on several benchmarks, demonstrating the effectiveness of integrating VLM-based reasoning with 3DGS for embodied tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12823v1">TreeDGS: Aerial Gaussian Splatting for Distant DBH Measurement</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Aerial remote sensing enables efficient large-area surveying, but accurate direct object-level measurement remains difficult in complex natural scenes. Recent advancements in 3D vision, particularly learned radiance-field representations such as NeRF and 3D Gaussian Splatting, have begun to raise the ceiling on reconstruction fidelity and densifiable geometry from posed imagery. Nevertheless, direct aerial measurement of important natural attributes such as tree diameter at breast height (DBH) remains challenging. Trunks in aerial forest scans are distant and sparsely observed in image views: at typical operating altitudes, stems may span only a few pixels. With these constraints, conventional reconstruction methods leave breast-height trunk geometry weakly constrained. We present TreeDGS, an aerial image reconstruction method that leverages 3D Gaussian Splatting as a continuous, densifiable scene representation for trunk measurement. After SfM-MVS initialization and Gaussian optimization, we extract a dense point set from the Gaussian field using RaDe-GS's depth-aware cumulative-opacity integration and associate each sample with a multi-view opacity reliability score. We then estimate DBH from trunk-isolated points using opacity-weighted solid-circle fitting. Evaluated on 10 plots with field-measured DBH, TreeDGS reaches 4.79,cm RMSE (about 2.6 pixels at this GSD) and outperforms a state-of-the-art LiDAR baseline (7.91,cm RMSE), demonstrating that densified splat-based geometry can enable accurate, low-cost aerial DBH measurement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12814v1">CSGaussian: Progressive Rate-Distortion Compression and Segmentation for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 Accepted at WACV 2026
    </div>
    <details class="paper-abstract">
      We present the first unified framework for rate-distortion-optimized compression and segmentation of 3D Gaussian Splatting (3DGS). While 3DGS has proven effective for both real-time rendering and semantic scene understanding, prior works have largely treated these tasks independently, leaving their joint consideration unexplored. Inspired by recent advances in rate-distortion-optimized 3DGS compression, this work integrates semantic learning into the compression pipeline to support decoder-side applications--such as scene editing and manipulation--that extend beyond traditional scene reconstruction and view synthesis. Our scheme features a lightweight implicit neural representation-based hyperprior, enabling efficient entropy coding of both color and semantic attributes while avoiding costly grid-based hyperprior as seen in many prior works. To facilitate compression and segmentation, we further develop compression-guided segmentation learning, consisting of quantization-aware training to enhance feature separability and a quality-aware weighting mechanism to suppress unreliable Gaussian primitives. Extensive experiments on the LERF and 3D-OVS datasets demonstrate that our approach significantly reduces transmission cost while preserving high rendering quality and strong segmentation performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12736v1">KaoLRM: Repurposing Pre-trained Large Reconstruction Models for Parametric 3D Face Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      We propose KaoLRM to re-target the learned prior of the Large Reconstruction Model (LRM) for parametric 3D face reconstruction from single-view images. Parametric 3D Morphable Models (3DMMs) have been widely used for facial reconstruction due to their compact and interpretable parameterization, yet existing 3DMM regressors often exhibit poor consistency across varying viewpoints. To address this, we harness the pre-trained 3D prior of LRM and incorporate FLAME-based 2D Gaussian Splatting into LRM's rendering pipeline. Specifically, KaoLRM projects LRM's pre-trained triplane features into the FLAME parameter space to recover geometry, and models appearance via 2D Gaussian primitives that are tightly coupled to the FLAME mesh. The rich prior enables the FLAME regressor to be aware of the 3D structure, leading to accurate and robust reconstructions under self-occlusions and diverse viewpoints. Experiments on both controlled and in-the-wild benchmarks demonstrate that KaoLRM achieves superior reconstruction accuracy and cross-view consistency, while existing methods remain sensitive to viewpoint variations. The code is released at https://github.com/CyberAgentAILab/KaoLRM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09291v2">TIDI-GS: Floater Suppression in 3D Gaussian Splatting for Enhanced Indoor Scene Fidelity</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a technique to create high-quality, real-time 3D scenes from images. This method often produces visual artifacts known as floaters--nearly transparent, disconnected elements that drift in space away from the actual surface. This geometric inaccuracy undermines the reliability of these models for practical applications, which is critical. To address this issue, we introduce TIDI-GS, a new training framework designed to eliminate these floaters. A key benefit of our approach is that it functions as a lightweight plugin for the standard 3DGS pipeline, requiring no major architectural changes and adding minimal overhead to the training process. The core of our method is a floater pruning algorithm--TIDI--that identifies and removes floaters based on several criteria: their consistency across multiple viewpoints, their spatial relationship to other elements, and an importance score learned during training. The framework includes a mechanism to preserve fine details, ensuring that important high-frequency elements are not mistakenly removed. This targeted cleanup is supported by a monocular depth-based loss function that helps improve the overall geometric structure of the scene. Our experiments demonstrate that TIDI-GS improves both the perceptual quality and geometric integrity of reconstructions, transforming them into robust digital assets, suitable for high-fidelity applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04754v2">ProFuse: Efficient Cross-View Context Fusion for Open-Vocabulary 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-18
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      We present ProFuse, an efficient context-aware framework for open-vocabulary 3D scene understanding with 3D Gaussian Splatting (3DGS). The pipeline enhances cross-view consistency and intra-mask cohesion within a direct registration setup, adding minimal overhead and requiring no render-supervised fine-tuning. Instead of relying on a pretrained 3DGS scene, we introduce a dense correspondence-guided pre-registration phase that initializes Gaussians with accurate geometry while jointly constructing 3D Context Proposals via cross-view clustering. Each proposal carries a global feature obtained through weighted aggregation of member embeddings, and this feature is fused onto Gaussians during direct registration to maintain per-primitive language coherence across views. With associations established in advance, semantic fusion requires no additional optimization beyond standard reconstruction, and the model retains geometric refinement without densification. ProFuse achieves strong open-vocabulary 3DGS understanding while completing semantic attachment in about five minutes per scene, which is two times faster than SOTA. Additional details are available at our project page https://chiou1203.github.io/ProFuse/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03024v2">SA-ResGS: Self-Augmented Residual 3D Gaussian Splatting for Next Best View Selection</a></div>
    <div class="paper-meta">
      📅 2026-01-18
    </div>
    <details class="paper-abstract">
      We propose Self-Augmented Residual 3D Gaussian Splatting (SA-ResGS), a novel framework to stabilize uncertainty quantification and enhancing uncertainty-aware supervision in next-best-view (NBV) selection for active scene reconstruction. SA-ResGS improves both the reliability of uncertainty estimates and their effectiveness for supervision by generating Self-Augmented point clouds (SA-Points) via triangulation between a training view and a rasterized extrapolated view, enabling efficient scene coverage estimation. While improving scene coverage through physically guided view selection, SA-ResGS also addresses the challenge of under-supervised Gaussians, exacerbated by sparse and wide-baseline views, by introducing the first residual learning strategy tailored for 3D Gaussian Splatting. This targeted supervision enhances gradient flow in high-uncertainty Gaussians by combining uncertainty-driven filtering with dropout- and hard-negative-mining-inspired sampling. Our contributions are threefold: (1) a physically grounded view selection strategy that promotes efficient and uniform scene coverage; (2) an uncertainty-aware residual supervision scheme that amplifies learning signals for weakly contributing Gaussians, improving training stability and uncertainty estimation across scenes with diverse camera distributions; (3) an implicit unbiasing of uncertainty quantification as a consequence of constrained view selection and residual supervision, which together mitigate conflicting effects of wide-baseline exploration and sparse-view ambiguity in NBV planning. Experiments on active view selection demonstrate that SA-ResGS outperforms state-of-the-art baselines in both reconstruction quality and view selection robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12122v1">Active Semantic Mapping of Horticultural Environments Using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-17
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Semantic reconstruction of agricultural scenes plays a vital role in tasks such as phenotyping and yield estimation. However, traditional approaches that rely on manual scanning or fixed camera setups remain a major bottleneck in this process. In this work, we propose an active 3D reconstruction framework for horticultural environments using a mobile manipulator. The proposed system integrates the classical Octomap representation with 3D Gaussian Splatting to enable accurate and efficient target-aware mapping. While a low-resolution Octomap provides probabilistic occupancy information for informative viewpoint selection and collision-free planning, 3D Gaussian Splatting leverages geometric, photometric, and semantic information to optimize a set of 3D Gaussians for high-fidelity scene reconstruction. We further introduce simple yet effective strategies to enhance robustness against segmentation noise and reduce memory consumption. Simulation experiments demonstrate that our method outperforms purely occupancy-based approaches in both runtime efficiency and reconstruction accuracy, enabling precise fruit counting and volume estimation. Compared to a 0.01m-resolution Octomap, our approach achieves an improvement of 6.6% in fruit-level F1 score under noise-free conditions, and up to 28.6% under segmentation noise. Additionally, it achieves a 50% reduction in runtime, highlighting its potential for scalable, real-time semantic reconstruction in agricultural robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11772v1">studentSplat: Your Student Model Learns Single-view 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Recent advance in feed-forward 3D Gaussian splatting has enable remarkable multi-view 3D scene reconstruction or single-view 3D object reconstruction but single-view 3D scene reconstruction remain under-explored due to inherited ambiguity in single-view. We present \textbf{studentSplat}, a single-view 3D Gaussian splatting method for scene reconstruction. To overcome the scale ambiguity and extrapolation problems inherent in novel-view supervision from a single input, we introduce two techniques: 1) a teacher-student architecture where a multi-view teacher model provides geometric supervision to the single-view student during training, addressing scale ambiguity and encourage geometric validity; and 2) an extrapolation network that completes missing scene context, enabling high-quality extrapolation. Extensive experiments show studentSplat achieves state-of-the-art single-view novel-view reconstruction quality and comparable performance to multi-view methods at the scene level. Furthermore, studentSplat demonstrates competitive performance as a self-supervised single-view depth estimation method, highlighting its potential for general single-view 3D understanding tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10606v1">RSATalker: Realistic Socially-Aware Talking Head Generation for Multi-Turn Conversation</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Talking head generation is increasingly important in virtual reality (VR), especially for social scenarios involving multi-turn conversation. Existing approaches face notable limitations: mesh-based 3D methods can model dual-person dialogue but lack realistic textures, while large-model-based 2D methods produce natural appearances but incur prohibitive computational costs. Recently, 3D Gaussian Splatting (3DGS) based methods achieve efficient and realistic rendering but remain speaker-only and ignore social relationships. We introduce RSATalker, the first framework that leverages 3DGS for realistic and socially-aware talking head generation with support for multi-turn conversation. Our method first drives mesh-based 3D facial motion from speech, then binds 3D Gaussians to mesh facets to render high-fidelity 2D avatar videos. To capture interpersonal dynamics, we propose a socially-aware module that encodes social relationships, including blood and non-blood as well as equal and unequal, into high-level embeddings through a learnable query mechanism. We design a three-stage training paradigm and construct the RSATalker dataset with speech-mesh-image triplets annotated with social relationships. Extensive experiments demonstrate that RSATalker achieves state-of-the-art performance in both realism and social awareness. The code and dataset will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00705v3">RGS-SLAM: Robust Gaussian Splatting SLAM with One-Shot Dense Initialization</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 10 pages, 9 figures
    </div>
    <details class="paper-abstract">
      We introduce RGS-SLAM, a robust Gaussian-splatting SLAM framework that replaces the residual-driven densification stage of GS-SLAM with a training-free correspondence-to-Gaussian initialization. Instead of progressively adding Gaussians as residuals reveal missing geometry, RGS-SLAM performs a one-shot triangulation of dense multi-view correspondences derived from DINOv3 descriptors refined through a confidence-aware inlier classifier, generating a well-distributed and structure-aware Gaussian seed prior to optimization. This initialization stabilizes early mapping and accelerates convergence by roughly 20\%, yielding higher rendering fidelity in texture-rich and cluttered scenes while remaining fully compatible with existing GS-SLAM pipelines. Evaluated on the TUM RGB-D and Replica datasets, RGS-SLAM achieves competitive or superior localization and reconstruction accuracy compared with state-of-the-art Gaussian and point-based SLAM systems, sustaining real-time mapping performance at up to 925 FPS. Additional details and resources are available at this URL: https://breeze1124.github.io/rgs-slam-project-page/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10075v1">Thinking Like Van Gogh: Structure-Aware Style Transfer via Flow-Guided 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 7 pages, 8 figures
    </div>
    <details class="paper-abstract">
      In 1888, Vincent van Gogh wrote, "I am seeking exaggeration in the essential." This principle, amplifying structural form while suppressing photographic detail, lies at the core of Post-Impressionist art. However, most existing 3D style transfer methods invert this philosophy, treating geometry as a rigid substrate for surface-level texture projection. To authentically reproduce Post-Impressionist stylization, geometric abstraction must be embraced as the primary vehicle of expression. We propose a flow-guided geometric advection framework for 3D Gaussian Splatting (3DGS) that operationalizes this principle in a mesh-free setting. Our method extracts directional flow fields from 2D paintings and back-propagates them into 3D space, rectifying Gaussian primitives to form flow-aligned brushstrokes that conform to scene topology without relying on explicit mesh priors. This enables expressive structural deformation driven directly by painterly motion rather than photometric constraints. Our contributions are threefold: (1) a projection-based, mesh-free flow guidance mechanism that transfers 2D artistic motion into 3D Gaussian geometry; (2) a luminance-structure decoupling strategy that isolates geometric deformation from color optimization, mitigating artifacts during aggressive structural abstraction; and (3) a VLM-as-a-Judge evaluation framework that assesses artistic authenticity through aesthetic judgment instead of conventional pixel-level metrics, explicitly addressing the subjective nature of artistic stylization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.26117v2">JOGS: Joint Optimization of Pose Estimation and 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Traditional novel view synthesis methods heavily rely on external camera pose estimation tools such as COLMAP, which often introduce computational bottlenecks and propagate errors. To address these challenges, we propose a unified framework that jointly optimizes 3D Gaussian points and camera poses without requiring pre-calibrated inputs. Our approach iteratively refines 3D Gaussian parameters and updates camera poses through a novel co-optimization strategy, ensuring simultaneous improvements in scene reconstruction fidelity and pose estimation accuracy. The key innovation lies in decoupling the joint optimization into two interleaved phases: first, updating 3D Gaussian parameters via differentiable rendering with fixed poses, and second, refining camera poses using a customized 3D optical flow algorithm that incorporates geometric and photometric constraints. This formulation progressively reduces projection errors, particularly in challenging scenarios with large viewpoint variations and sparse feature distributions, where traditional methods struggle. Extensive evaluations on multiple datasets demonstrate that our approach significantly outperforms existing COLMAP-free techniques in reconstruction quality, and also surpasses the standard COLMAP-based baseline in general.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09417v1">Variable Basis Mapping for Real-Time Volumetric Visualization</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 11 pages. Under review
    </div>
    <details class="paper-abstract">
      Real-time visualization of large-scale volumetric data remains challenging, as direct volume rendering and voxel-based methods suffer from prohibitively high computational cost. We propose Variable Basis Mapping (VBM), a framework that transforms volumetric fields into 3D Gaussian Splatting (3DGS) representations through wavelet-domain analysis. First, we precompute a compact Wavelet-to-Gaussian Transition Bank that provides optimal Gaussian surrogates for canonical wavelet atoms across multiple scales. Second, we perform analytical Gaussian construction that maps discrete wavelet coefficients directly to 3DGS parameters using a closed-form, mathematically principled rule. Finally, a lightweight image-space fine-tuning stage further refines the representation to improve rendering fidelity. Experiments on diverse datasets demonstrate that VBM significantly accelerates convergence and enhances rendering quality, enabling real-time volumetric visualization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09291v1">TIDI-GS: Floater Suppression in 3D Gaussian Splatting for Enhanced Indoor Scene Fidelity</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a technique to create high-quality, real-time 3D scenes from images. This method often produces visual artifacts known as floaters--nearly transparent, disconnected elements that drift in space away from the actual surface. This geometric inaccuracy undermines the reliability of these models for practical applications, which is critical. To address this issue, we introduce TIDI-GS, a new training framework designed to eliminate these floaters. A key benefit of our approach is that it functions as a lightweight plugin for the standard 3DGS pipeline, requiring no major architectural changes and adding minimal overhead to the training process. The core of our method is a floater pruning algorithm--TIDI--that identifies and removes floaters based on several criteria: their consistency across multiple viewpoints, their spatial relationship to other elements, and an importance score learned during training. The framework includes a mechanism to preserve fine details, ensuring that important high-frequency elements are not mistakenly removed. This targeted cleanup is supported by a monocular depth-based loss function that helps improve the overall geometric structure of the scene. Our experiments demonstrate that TIDI-GS improves both the perceptual quality and geometric integrity of reconstructions, transforming them into robust digital assets, suitable for high-fidelity applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09265v1">GaussianFluent: Gaussian Simulation for Dynamic Scenes with Mixed Materials</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a prominent 3D representation for high-fidelity and real-time rendering. Prior work has coupled physics simulation with Gaussians, but predominantly targets soft, deformable materials, leaving brittle fracture largely unresolved. This stems from two key obstacles: the lack of volumetric interiors with coherent textures in GS representation, and the absence of fracture-aware simulation methods for Gaussians. To address these challenges, we introduce GaussianFluent, a unified framework for realistic simulation and rendering of dynamic object states. First, it synthesizes photorealistic interiors by densifying internal Gaussians guided by generative models. Second, it integrates an optimized Continuum Damage Material Point Method (CD-MPM) to enable brittle fracture simulation at remarkably high speed. Our approach handles complex scenarios including mixed-material objects and multi-stage fracture propagation, achieving results infeasible with previous methods. Experiments clearly demonstrate GaussianFluent's capability for photo-realistic, real-time rendering with structurally consistent interiors, highlighting its potential for downstream application, such as VR and Robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09243v1">A$^2$TG: Adaptive Anisotropic Textured Gaussians for Efficient 3D Scene Representation</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has emerged as a powerful representation for high-quality, real-time 3D scene rendering. While recent works extend Gaussians with learnable textures to enrich visual appearance, existing approaches allocate a fixed square texture per primitive, leading to inefficient memory usage and limited adaptability to scene variability. In this paper, we introduce adaptive anisotropic textured Gaussians (A$^2$TG), a novel representation that generalizes textured Gaussians by equipping each primitive with an anisotropic texture. Our method employs a gradient-guided adaptive rule to jointly determine texture resolution and aspect ratio, enabling non-uniform, detail-aware allocation that aligns with the anisotropic nature of Gaussian splats. This design significantly improves texture efficiency, reducing memory consumption while enhancing image quality. Experiments on multiple benchmark datasets demonstrate that A TG consistently outperforms fixed-texture Gaussian Splatting methods, achieving comparable rendering fidelity with substantially lower memory requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.17336v3">Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 24 pages, 10 figures, NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Dynamic 4D Gaussian Splatting (4DGS) effectively extends the high-speed rendering capabilities of 3D Gaussian Splatting (3DGS) to represent volumetric videos. However, the large number of Gaussians, substantial temporal redundancies, and especially the absence of an entropy-aware compression framework result in large storage requirements. Consequently, this poses significant challenges for practical deployment, efficient edge-device processing, and data transmission. In this paper, we introduce a novel end-to-end RD-optimized compression framework tailored for 4DGS, aiming to enable flexible, high-fidelity rendering across varied computational platforms. Leveraging Fully Explicit Dynamic Gaussian Splatting (Ex4DGS), one of the state-of-the-art 4DGS methods, as our baseline, we start from the existing 3DGS compression methods for compatibility while effectively addressing additional challenges introduced by the temporal axis. In particular, instead of storing motion trajectories independently per point, we employ a wavelet transform to reflect the real-world smoothness prior, significantly enhancing storage efficiency. This approach yields significantly improved compression ratios and provides a user-controlled balance between compression efficiency and rendering quality. Extensive experiments demonstrate the effectiveness of our method, achieving up to 91$\times$ compression compared to the original Ex4DGS model while maintaining high visual fidelity. These results highlight the applicability of our framework for real-time dynamic scene rendering in diverse scenarios, from resource-constrained edge devices to high-performance environments. The source code is available at https://github.com/HyeongminLEE/RD4DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05280v4">Unifying Appearance Codes and Bilateral Grids for Driving Scene Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Accepted to NeurIPS 2025 ; Project page: https://bigcileng.github.io/bilateral-driving ; Code: https://github.com/BigCiLeng/bilateral-driving
    </div>
    <details class="paper-abstract">
      Neural rendering techniques, including NeRF and Gaussian Splatting (GS), rely on photometric consistency to produce high-quality reconstructions. However, in real-world scenarios, it is challenging to guarantee perfect photometric consistency in acquired images. Appearance codes have been widely used to address this issue, but their modeling capability is limited, as a single code is applied to the entire image. Recently, the bilateral grid was introduced to perform pixel-wise color mapping, but it is difficult to optimize and constrain effectively. In this paper, we propose a novel multi-scale bilateral grid that unifies appearance codes and bilateral grids. We demonstrate that this approach significantly improves geometric accuracy in dynamic, decoupled autonomous driving scene reconstruction, outperforming both appearance codes and bilateral grids. This is crucial for autonomous driving, where accurate geometry is important for obstacle avoidance and control. Our method shows strong results across four datasets: Waymo, NuScenes, Argoverse, and PandaSet. We further demonstrate that the improvement in geometry is driven by the multi-scale bilateral grid, which effectively reduces floaters caused by photometric inconsistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00705v2">RGS-SLAM: Robust Gaussian Splatting SLAM with One-Shot Dense Initialization</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 10 pages, 9 figures
    </div>
    <details class="paper-abstract">
      We introduce RGS-SLAM, a robust Gaussian-splatting SLAM framework that replaces the residual-driven densification stage of GS-SLAM with a training-free correspondence-to-Gaussian initialization. Instead of progressively adding Gaussians as residuals reveal missing geometry, RGS-SLAM performs a one-shot triangulation of dense multi-view correspondences derived from DINOv3 descriptors refined through a confidence-aware inlier classifier, generating a well-distributed and structure-aware Gaussian seed prior to optimization. This initialization stabilizes early mapping and accelerates convergence by roughly 20\%, yielding higher rendering fidelity in texture-rich and cluttered scenes while remaining fully compatible with existing GS-SLAM pipelines. Evaluated on the TUM RGB-D and Replica datasets, RGS-SLAM achieves competitive or superior localization and reconstruction accuracy compared with state-of-the-art Gaussian and point-based SLAM systems, sustaining real-time mapping performance at up to 925 FPS. Project page:https://breeze1124.github.io/rgs-slam-project-page/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2408.15235v3">Learning-based Multi-View Stereo: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Accepted to IEEE T-PAMI 2026
    </div>
    <details class="paper-abstract">
      3D reconstruction aims to recover the dense 3D structure of a scene. It plays an essential role in various applications such as Augmented/Virtual Reality (AR/VR), autonomous driving and robotics. Leveraging multiple views of a scene captured from different viewpoints, Multi-View Stereo (MVS) algorithms synthesize a comprehensive 3D representation, enabling precise reconstruction in complex environments. Due to its efficiency and effectiveness, MVS has become a pivotal method for image-based 3D reconstruction. Recently, with the success of deep learning, many learning-based MVS methods have been proposed, achieving impressive performance against traditional methods. We categorize these learning-based methods as: depth map-based, voxel-based, NeRF-based, 3D Gaussian Splatting-based, and large feed-forward methods. Among these, we focus significantly on depth map-based methods, which are the main family of MVS due to their conciseness, flexibility and scalability. In this survey, we provide a comprehensive review of the literature at the time of this writing. We investigate these learning-based methods, summarize their performances on popular benchmarks, and discuss promising future research directions in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07540v2">ViewMorpher3D: A 3D-aware Diffusion Framework for Multi-Camera Novel View Synthesis in Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Paper and supplementary materials
    </div>
    <details class="paper-abstract">
      Autonomous driving systems rely heavily on multi-view images to ensure accurate perception and robust decision-making. To effectively develop and evaluate perception stacks and planning algorithms, realistic closed-loop simulators are indispensable. While 3D reconstruction techniques such as Gaussian Splatting offer promising avenues for simulator construction, the rendered novel views often exhibit artifacts, particularly in extrapolated perspectives or when available observations are sparse. We introduce ViewMorpher3D, a multi-view image enhancement framework based on image diffusion models, designed to elevate photorealism and multi-view coherence in driving scenes. Unlike single-view approaches, ViewMorpher3D jointly processes a set of rendered views conditioned on camera poses, 3D geometric priors, and temporally adjacent or spatially overlapping reference views. This enables the model to infer missing details, suppress rendering artifacts, and enforce cross-view consistency. Our framework accommodates variable numbers of cameras and flexible reference/target view configurations, making it adaptable to diverse sensor setups. Experiments on real-world driving datasets demonstrate substantial improvements in image quality metrics, effectively reducing artifacts while preserving geometric fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11048v2">PINGS-X: Physics-Informed Normalized Gaussian Splatting with Axes Alignment for Efficient Super-Resolution of 4D Flow MRI</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Accepted at AAAI 2026. Supplementary material included after references. 27 pages, 21 figures, 11 tables
    </div>
    <details class="paper-abstract">
      4D flow magnetic resonance imaging (MRI) is a reliable, non-invasive approach for estimating blood flow velocities, vital for cardiovascular diagnostics. Unlike conventional MRI focused on anatomical structures, 4D flow MRI requires high spatiotemporal resolution for early detection of critical conditions such as stenosis or aneurysms. However, achieving such resolution typically results in prolonged scan times, creating a trade-off between acquisition speed and prediction accuracy. Recent studies have leveraged physics-informed neural networks (PINNs) for super-resolution of MRI data, but their practical applicability is limited as the prohibitively slow training process must be performed for each patient. To overcome this limitation, we propose PINGS-X, a novel framework modeling high-resolution flow velocities using axes-aligned spatiotemporal Gaussian representations. Inspired by the effectiveness of 3D Gaussian splatting (3DGS) in novel view synthesis, PINGS-X extends this concept through several non-trivial novel innovations: (i) normalized Gaussian splatting with a formal convergence guarantee, (ii) axes-aligned Gaussians that simplify training for high-dimensional data while preserving accuracy and the convergence guarantee, and (iii) a Gaussian merging procedure to prevent degenerate solutions and boost computational efficiency. Experimental results on computational fluid dynamics (CFD) and real 4D flow MRI datasets demonstrate that PINGS-X substantially reduces training time while achieving superior super-resolution accuracy. Our code and datasets are available at https://github.com/SpatialAILab/PINGS-X.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.02283v6">GP-GS: Gaussian Processes Densification for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 11 pages, 8 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables photorealistic rendering but suffers from artefacts due to sparse Structure-from-Motion (SfM) initialisation. To address this limitation, we propose GP-GS, a Gaussian Process (GP) based densification framework for 3DGS optimisation. GP-GS formulates point cloud densification as a continuous regression problem, where a GP learns a local mapping from 2D pixel coordinates to 3D position and colour attributes. An adaptive neighbourhood-based sampling strategy generates candidate pixels for inference, while GP-predicted uncertainty is used to filter unreliable predictions, reducing noise and preserving geometric structure. Extensive experiments on synthetic and real-world benchmarks demonstrate that GP-GS consistently improves reconstruction quality and rendering fidelity, achieving up to 1.12 dB PSNR improvement over strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.10231v2">SuperGSeg: Open-Vocabulary 3D Segmentation with Structured Super-Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 13 pages, 8 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has recently gained traction for its efficient training and real-time rendering. While the vanilla Gaussian Splatting representation is mainly designed for view synthesis, more recent works investigated how to extend it with scene understanding and language features. However, existing methods lack a detailed comprehension of scenes, limiting their ability to segment and interpret complex structures. To this end, We introduce SuperGSeg, a novel approach that fosters cohesive, context-aware scene representation by disentangling segmentation and language field distillation. SuperGSeg first employs neural Gaussians to learn instance and hierarchical segmentation features from multi-view images with the aid of off-the-shelf 2D masks. These features are then leveraged to create a sparse set of what we call Super-Gaussians. Super-Gaussians facilitate the distillation of 2D language features into 3D space. Through Super-Gaussians, our method enables high-dimensional language feature rendering without extreme increases in GPU memory. Extensive experiments demonstrate that SuperGSeg outperforms prior works on both open-vocabulary object localization and semantic segmentation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07540v1">ViewMorpher3D: A 3D-aware Diffusion Framework for Multi-Camera Novel View Synthesis in Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 Paper and supplementary materials
    </div>
    <details class="paper-abstract">
      Autonomous driving systems rely heavily on multi-view images to ensure accurate perception and robust decision-making. To effectively develop and evaluate perception stacks and planning algorithms, realistic closed-loop simulators are indispensable. While 3D reconstruction techniques such as Gaussian Splatting offer promising avenues for simulator construction, the rendered novel views often exhibit artifacts, particularly in extrapolated perspectives or when available observations are sparse. We introduce ViewMorpher3D, a multi-view image enhancement framework based on image diffusion models, designed to elevate photorealism and multi-view coherence in driving scenes. Unlike single-view approaches, ViewMorpher3D jointly processes a set of rendered views conditioned on camera poses, 3D geometric priors, and temporally adjacent or spatially overlapping reference views. This enables the model to infer missing details, suppress rendering artifacts, and enforce cross-view consistency. Our framework accommodates variable numbers of cameras and flexible reference/target view configurations, making it adaptable to diverse sensor setups. Experiments on real-world driving datasets demonstrate substantial improvements in image quality metrics, effectively reducing artifacts while preserving geometric fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07518v1">Mon3tr: Monocular 3D Telepresence with Pre-built Gaussian Avatars as Amortization</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Immersive telepresence aims to transform human interaction in AR/VR applications by enabling lifelike full-body holographic representations for enhanced remote collaboration. However, existing systems rely on hardware-intensive multi-camera setups and demand high bandwidth for volumetric streaming, limiting their real-time performance on mobile devices. To overcome these challenges, we propose Mon3tr, a novel Monocular 3D telepresence framework that integrates 3D Gaussian splatting (3DGS) based parametric human modeling into telepresence for the first time. Mon3tr adopts an amortized computation strategy, dividing the process into a one-time offline multi-view reconstruction phase to build a user-specific avatar and a monocular online inference phase during live telepresence sessions. A single monocular RGB camera is used to capture body motions and facial expressions in real time to drive the 3DGS-based parametric human model, significantly reducing system complexity and cost. The extracted motion and appearance features are transmitted at < 0.2 Mbps over WebRTC's data channel, allowing robust adaptation to network fluctuations. On the receiver side, e.g., Meta Quest 3, we develop a lightweight 3DGS attribute deformation network to dynamically generate corrective 3DGS attribute adjustments on the pre-built avatar, synthesizing photorealistic motion and appearance at ~ 60 FPS. Extensive experiments demonstrate the state-of-the-art performance of our method, achieving a PSNR of > 28 dB for novel poses, an end-to-end latency of ~ 80 ms, and > 1000x bandwidth reduction compared to point-cloud streaming, while supporting real-time operation from monocular inputs across diverse scenarios. Our demos can be found at https://mon3tr3d.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07484v1">R3-RECON: Radiance-Field-Free Active Reconstruction via Renderability</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 18 pages, 11 figures
    </div>
    <details class="paper-abstract">
      In active reconstruction, an embodied agent must decide where to look next to efficiently acquire views that support high-quality novel-view rendering. Recent work on active view planning for neural rendering largely derives next-best-view (NBV) criteria by backpropagating through radiance fields or estimating information entropy over 3D Gaussian primitives. While effective, these strategies tightly couple view selection to heavy, representation-specific mechanisms and fail to account for the computational and resource constraints required for lightweight online deployment. In this paper, we revisit active reconstruction from a renderability-centric perspective. We propose $\mathbb{R}^{3}$-RECON, a radiance-fields-free active reconstruction framework that induces an implicit, pose-conditioned renderability field over SE(3) from a lightweight voxel map. Our formulation aggregates per-voxel online observation statistics into a unified scalar renderability score that is cheap to update and can be queried in closed form at arbitrary candidate viewpoints in milliseconds, without requiring gradients or radiance-field training. This renderability field is strongly correlated with image-space reconstruction error, naturally guiding NBV selection. We further introduce a panoramic extension that estimates omnidirectional (360$^\circ$) view utility to accelerate candidate evaluation. In the standard indoor Replica dataset, $\mathbb{R}^{3}$-RECON achieves more uniform novel-view quality and higher 3D Gaussian splatting (3DGS) reconstruction accuracy than recent active GS baselines with matched view and time budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.19142v2">CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Recent works in 3D multimodal learning have made remarkable progress. However, typically 3D multimodal models are only capable of handling point clouds. Compared to the emerging 3D representation technique, 3D Gaussian Splatting (3DGS), the spatially sparse point cloud cannot depict the texture information of 3D objects, resulting in inferior reconstruction capabilities. This limitation constrains the potential of point cloud-based 3D multimodal representation learning. In this paper, we present CLIP-GS, a novel multimodal representation learning framework grounded in 3DGS. We introduce the GS Tokenizer to generate serialized gaussian tokens, which are then processed through transformer layers pre-initialized with weights from point cloud models, resulting in the 3DGS embeddings. CLIP-GS leverages contrastive loss between 3DGS and the visual-text embeddings of CLIP, and we introduce an image voting loss to guide the directionality and convergence of gradient optimization. Furthermore, we develop an efficient way to generate triplets of 3DGS, images, and text, facilitating CLIP-GS in learning unified multimodal representations. Leveraging the well-aligned multimodal representations, CLIP-GS demonstrates versatility and outperforms point cloud-based models on various 3D tasks, including multimodal retrieval, zero-shot, and few-shot classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.00800v2">SWAGSplatting: Semantic-guided Water-scene Augmented Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Accurate 3D reconstruction in underwater environments remains a challenging task due to light attenuation, scattering, and limited visibility. While recent AI-based approaches have advanced underwater imaging, they often overlook high-level semantic understanding, which is crucial for reconstructing complex scenes. In this paper, we propose SWAGSplatting, \textit{Semantic-guided Water-scene Augmented Gaussian Splatting}, a novel multimodal framework that integrates language and vision knowledge into 3D Gaussian Splatting for robust and high-fidelity underwater reconstruction. Each Gaussian primitive is augmented with a learnable semantic feature, supervised using CLIP-based embeddings extracted from region-level semantic cues. A dedicated semantic consistency loss enforces alignment between geometric reconstruction and scene semantics. In addition, a stage-wise optimisation strategy combining coarse-to-fine learning with late-stage parameter refinement improves training stability and visual quality. Furthermore, we propose a 3D Gaussian Primitives Reallocation strategy to address the imbalanced distribution of primitives introduced by naive point cloud densification. Extensive experiments on the SeaThru-NeRF and Submerged3D datasets demonstrate that SWAGSplatting consistently outperforms state-of-the-art methods across PSNR, SSIM, and LPIPS metrics, achieving up to a 3.48 dB improvement in PSNR, enabling more accurate and semantically coherent underwater scene reconstruction for applications in marine perception and exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.13339v2">Volume Encoding Gaussians: Transfer Function-Agnostic 3D Gaussians for Volume Rendering</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Visualizing the large-scale datasets output by HPC resources presents a difficult challenge, as the memory and compute power required become prohibitively expensive for end user systems. Novel view synthesis techniques can address this by producing a small, interactive model of the data, requiring only a set of training images to learn from. While these models allow accessible visualization of large data and complex scenes, they do not provide the interactions needed for scientific volumes, as they do not support interactive selection of transfer functions and lighting parameters. To address this, we introduce Volume Encoding Gaussians (VEG), a 3D Gaussian-based representation for volume visualization that supports arbitrary color and opacity mappings. Unlike prior 3D Gaussian Splatting (3DGS) methods that store color and opacity for each Gaussian, VEG decouple the visual appearance from the data representation by encoding only scalar values, enabling transfer function-agnostic rendering of 3DGS models. To ensure complete scalar field coverage, we introduce an opacity-guided training strategy, using differentiable rendering with multiple transfer functions to optimize our data representation. This allows VEG to preserve fine features across the full scalar range of a dataset while remaining independent of any specific transfer function. Across a diverse set of volume datasets, we demonstrate that our method outperforms the state-of-the-art on transfer functions unseen during training, while requiring a fraction of the memory and training time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07963v1">3DGS-Drag: Dragging Gaussians for Intuitive Point-Based 3D Editing</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      The transformative potential of 3D content creation has been progressively unlocked through advancements in generative models. Recently, intuitive drag editing with geometric changes has attracted significant attention in 2D editing yet remains challenging for 3D scenes. In this paper, we introduce 3DGS-Drag -- a point-based 3D editing framework that provides efficient, intuitive drag manipulation of real 3D scenes. Our approach bridges the gap between deformation-based and 2D-editing-based 3D editing methods, addressing their limitations to geometry-related content editing. We leverage two key innovations: deformation guidance utilizing 3D Gaussian Splatting for consistent geometric modifications and diffusion guidance for content correction and visual quality enhancement. A progressive editing strategy further supports aggressive 3D drag edits. Our method enables a wide range of edits, including motion change, shape adjustment, inpainting, and content extension. Experimental results demonstrate the effectiveness of 3DGS-Drag in various scenes, achieving state-of-the-art performance in geometry-related 3D content editing. Notably, the editing is efficient, taking 10 to 20 minutes on a single RTX 4090 GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.20031v4">MG-SLAM: Structure Gaussian Splatting SLAM with Manhattan World Hypothesis</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 IEEE Transactions on Automation Science and Engineering
    </div>
    <details class="paper-abstract">
      Gaussian Splatting SLAMs have made significant advancements in improving the efficiency and fidelity of real-time reconstructions. However, these systems often encounter incomplete reconstructions in complex indoor environments, characterized by substantial holes due to unobserved geometry caused by obstacles or limited view angles. To address this challenge, we present Manhattan Gaussian SLAM, an RGB-D system that leverages the Manhattan World hypothesis to enhance geometric accuracy and completeness. By seamlessly integrating fused line segments derived from structured scenes, our method ensures robust tracking in textureless indoor areas. Moreover, The extracted lines and planar surface assumption allow strategic interpolation of new Gaussians in regions of missing geometry, enabling efficient scene completion. Extensive experiments conducted on both synthetic and real-world scenes demonstrate that these advancements enable our method to achieve state-of-the-art performance, marking a substantial improvement in the capabilities of Gaussian SLAM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06831v1">SARA: Scene-Aware Reconstruction Accelerator</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 This work has been submitted to the 2026 International Conference on Pattern Recognition (ICPR) for possible publication
    </div>
    <details class="paper-abstract">
      We present SARA (Scene-Aware Reconstruction Accelerator), a geometry-driven pair selection module for Structure-from-Motion (SfM). Unlike conventional pipelines that select pairs based on visual similarity alone, SARA introduces geometry-first pair selection by scoring reconstruction informativeness - the product of overlap and parallax - before expensive matching. A lightweight pre-matching stage uses mutual nearest neighbors and RANSAC to estimate these cues, then constructs an Information-Weighted Spanning Tree (IWST) augmented with targeted edges for loop closure, long-baseline anchors, and weak-view reinforcement. Compared to exhaustive matching, SARA reduces rotation errors by 46.5+-5.5% and translation errors by 12.5+-6.5% across modern learned detectors, while achieving at most 50x speedup through 98% pair reduction (from 30,848 to 580 pairs). This reduces matching complexity from quadratic to quasi-linear, maintaining within +-3% of baseline reconstruction metrics for 3D Gaussian Splatting and SVRaster.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06479v1">SRFlow: A Dataset and Regularization Model for High-Resolution Facial Optical Flow via Splatting Rasterization</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      Facial optical flow supports a wide range of tasks in facial motion analysis. However, the lack of high-resolution facial optical flow datasets has hindered progress in this area. In this paper, we introduce Splatting Rasterization Flow (SRFlow), a high-resolution facial optical flow dataset, and Splatting Rasterization Guided FlowNet (SRFlowNet), a facial optical flow model with tailored regularization losses. These losses constrain flow predictions using masks and gradients computed via difference or Sobel operator. This effectively suppresses high-frequency noise and large-scale errors in texture-less or repetitive-pattern regions, enabling SRFlowNet to be the first model explicitly capable of capturing high-resolution skin motion guided by Gaussian splatting rasterization. Experiments show that training with the SRFlow dataset improves facial optical flow estimation across various optical flow models, reducing end-point error (EPE) by up to 42% (from 0.5081 to 0.2953). Furthermore, when coupled with the SRFlow dataset, SRFlowNet achieves up to a 48% improvement in F1-score (from 0.4733 to 0.6947) on a composite of three micro-expression datasets. These results demonstrate the value of advancing both facial optical flow estimation and micro-expression recognition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03422v2">What Is The Best 3D Scene Representation for Robotics? From Geometric to Foundation Models</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      In this paper, we provide a comprehensive overview of existing scene representation methods for robotics, covering traditional representations such as point clouds, voxels, signed distance functions (SDF), and scene graphs, as well as more recent neural representations like Neural Radiance Fields (NeRF), 3D Gaussian Splatting (3DGS), and the emerging Foundation Models. While current SLAM and localization systems predominantly rely on sparse representations like point clouds and voxels, dense scene representations are expected to play a critical role in downstream tasks such as navigation and obstacle avoidance. Moreover, neural representations such as NeRF, 3DGS, and foundation models are well-suited for integrating high-level semantic features and language-based priors, enabling more comprehensive 3D scene understanding and embodied intelligence. In this paper, we categorized the core modules of robotics into five parts (Perception, Mapping, Localization, Navigation, Manipulation). We start by presenting the standard formulation of different scene representation methods and comparing the advantages and disadvantages of scene representation across different modules. This survey is centered around the question: What is the best 3D scene representation for robotics? We then discuss the future development trends of 3D scene representations, with a particular focus on how the 3D Foundation Model could replace current methods as the unified solution for future robotic applications. The remaining challenges in fully realizing this model are also explored. We aim to offer a valuable resource for both newcomers and experienced researchers to explore the future of 3D scene representations and their application in robotics. We have published an open-source project on GitHub and will continue to add new works and technologies to this project.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.18992v2">VPGS-SLAM: Voxel-based Progressive 3D Gaussian SLAM in Large-Scale Scenes</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has recently shown promising results in dense visual SLAM. However, existing 3DGS-based SLAM methods are all constrained to small-room scenarios and struggle with memory explosion in large-scale scenes and long sequences. To this end, we propose VPGS-SLAM, the first 3DGS-based large-scale RGBD SLAM framework for both indoor and outdoor scenarios. We design a novel voxel-based progressive 3D Gaussian mapping method with multiple submaps for compact and accurate scene representation in large-scale and long-sequence scenes. This allows us to scale up to arbitrary scenes and improves robustness (even under pose drifts). In addition, we propose a 2D-3D fusion camera tracking method to achieve robust and accurate camera tracking in both indoor and outdoor large-scale scenes. Furthermore, we design a 2D-3D Gaussian loop closure method to eliminate pose drift. We further propose a submap fusion method with online distillation to achieve global consistency in large-scale scenes when detecting a loop. Experiments on various indoor and outdoor datasets demonstrate the superiority and generalizability of the proposed framework. The code will be open source on https://github.com/dtc111111/vpgs-slam.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05853v1">LayerGS: Decomposition and Inpainting of Layered 3D Human Avatars via 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      We propose a novel framework for decomposing arbitrarily posed humans into animatable multi-layered 3D human avatars, separating the body and garments. Conventional single-layer reconstruction methods lock clothing to one identity, while prior multi-layer approaches struggle with occluded regions. We overcome both limitations by encoding each layer as a set of 2D Gaussians for accurate geometry and photorealistic rendering, and inpainting hidden regions with a pretrained 2D diffusion model via score-distillation sampling (SDS). Our three-stage training strategy first reconstructs the coarse canonical garment via single-layer reconstruction, followed by multi-layer training to jointly recover the inner-layer body and outer-layer garment details. Experiments on two 3D human benchmark datasets (4D-Dress, Thuman2.0) show that our approach achieves better rendering quality and layer decomposition and recomposition than the previous state-of-the-art, enabling realistic virtual try-on under novel viewpoints and poses, and advancing practical creation of high-fidelity 3D human assets for immersive applications. Our code is available at https://github.com/RockyXu66/LayerGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05738v1">FeatureSLAM: Feature-enriched 3D gaussian splatting SLAM in real time</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      We present a real-time tracking SLAM system that unifies efficient camera tracking with photorealistic feature-enriched mapping using 3D Gaussian Splatting (3DGS). Our main contribution is integrating dense feature rasterization into the novel-view synthesis, aligned with a visual foundation model. This yields strong semantics, going beyond basic RGB-D input, aiding both tracking and mapping accuracy. Unlike previous semantic SLAM approaches (which embed pre-defined class labels) FeatureSLAM enables entirely new downstream tasks via free-viewpoint, open-set segmentation. Across standard benchmarks, our method achieves real-time tracking, on par with state-of-the-art systems while improving tracking stability and map fidelity without prohibitive compute. Quantitatively, we obtain 9\% lower pose error and 8\% higher mapping accuracy compared to recent fixed-set SLAM baselines. Our results confirm that real-time feature-embedded SLAM, is not only valuable for enabling new downstream applications. It also improves the performance of the underlying tracking and mapping subsystems, providing semantic and language masking results that are on-par with offline 3DGS models, alongside state-of-the-art tracking, depth and RGB rendering.
    </details>
</div>
