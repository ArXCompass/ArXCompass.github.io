# gaussian splatting - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

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
