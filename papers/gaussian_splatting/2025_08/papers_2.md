# gaussian splatting - 2025_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02261v1">SplatSSC: Decoupled Depth-Guided Gaussian Splatting for Semantic Scene Completion</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Monocular 3D Semantic Scene Completion (SSC) is a challenging yet promising task that aims to infer dense geometric and semantic descriptions of a scene from a single image. While recent object-centric paradigms significantly improve efficiency by leveraging flexible 3D Gaussian primitives, they still rely heavily on a large number of randomly initialized primitives, which inevitably leads to 1) inefficient primitive initialization and 2) outlier primitives that introduce erroneous artifacts. In this paper, we propose SplatSSC, a novel framework that resolves these limitations with a depth-guided initialization strategy and a principled Gaussian aggregator. Instead of random initialization, SplatSSC utilizes a dedicated depth branch composed of a Group-wise Multi-scale Fusion (GMF) module, which integrates multi-scale image and depth features to generate a sparse yet representative set of initial Gaussian primitives. To mitigate noise from outlier primitives, we develop the Decoupled Gaussian Aggregator (DGA), which enhances robustness by decomposing geometric and semantic predictions during the Gaussian-to-voxel splatting process. Complemented with a specialized Probability Scale Loss, our method achieves state-of-the-art performance on the Occ-ScanNet dataset, outperforming prior approaches by over 6.3% in IoU and 4.1% in mIoU, while reducing both latency and memory consumption by more than 9.3%. The code will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02172v1">GaussianCross: Cross-modal Self-supervised 3D Representation Learning via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 14 pages, 8 figures, accepted by MM'25
    </div>
    <details class="paper-abstract">
      The significance of informative and robust point representations has been widely acknowledged for 3D scene understanding. Despite existing self-supervised pre-training counterparts demonstrating promising performance, the model collapse and structural information deficiency remain prevalent due to insufficient point discrimination difficulty, yielding unreliable expressions and suboptimal performance. In this paper, we present GaussianCross, a novel cross-modal self-supervised 3D representation learning architecture integrating feed-forward 3D Gaussian Splatting (3DGS) techniques to address current challenges. GaussianCross seamlessly converts scale-inconsistent 3D point clouds into a unified cuboid-normalized Gaussian representation without missing details, enabling stable and generalizable pre-training. Subsequently, a tri-attribute adaptive distillation splatting module is incorporated to construct a 3D feature field, facilitating synergetic feature capturing of appearance, geometry, and semantic cues to maintain cross-modal consistency. To validate GaussianCross, we perform extensive evaluations on various benchmarks, including ScanNet, ScanNet200, and S3DIS. In particular, GaussianCross shows a prominent parameter and data efficiency, achieving superior performance through linear probing (<0.1% parameters) and limited data training (1% of scenes) compared to state-of-the-art methods. Furthermore, GaussianCross demonstrates strong generalization capabilities, improving the full fine-tuning accuracy by 9.3% mIoU and 6.1% AP$_{50}$ on ScanNet200 semantic and instance segmentation tasks, respectively, supporting the effectiveness of our approach. The code, weights, and visualizations are publicly available at \href{https://rayyoh.github.io/GaussianCross/}{https://rayyoh.github.io/GaussianCross/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02146v1">ScrewSplat: An End-to-End Method for Articulated Object Recognition</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 26 pages, 12 figures, Conference on Robot Learning (CoRL) 2025
    </div>
    <details class="paper-abstract">
      Articulated object recognition -- the task of identifying both the geometry and kinematic joints of objects with movable parts -- is essential for enabling robots to interact with everyday objects such as doors and laptops. However, existing approaches often rely on strong assumptions, such as a known number of articulated parts; require additional inputs, such as depth images; or involve complex intermediate steps that can introduce potential errors -- limiting their practicality in real-world settings. In this paper, we introduce ScrewSplat, a simple end-to-end method that operates solely on RGB observations. Our approach begins by randomly initializing screw axes, which are then iteratively optimized to recover the object's underlying kinematic structure. By integrating with Gaussian Splatting, we simultaneously reconstruct the 3D geometry and segment the object into rigid, movable parts. We demonstrate that our method achieves state-of-the-art recognition accuracy across a diverse set of articulated objects, and further enables zero-shot, text-guided manipulation using the recovered kinematic model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02129v1">VDEGaussian: Video Diffusion Enhanced 4D Gaussian Splatting for Dynamic Urban Scenes Modeling</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Dynamic urban scene modeling is a rapidly evolving area with broad applications. While current approaches leveraging neural radiance fields or Gaussian Splatting have achieved fine-grained reconstruction and high-fidelity novel view synthesis, they still face significant limitations. These often stem from a dependence on pre-calibrated object tracks or difficulties in accurately modeling fast-moving objects from undersampled capture, particularly due to challenges in handling temporal discontinuities. To overcome these issues, we propose a novel video diffusion-enhanced 4D Gaussian Splatting framework. Our key insight is to distill robust, temporally consistent priors from a test-time adapted video diffusion model. To ensure precise pose alignment and effective integration of this denoised content, we introduce two core innovations: a joint timestamp optimization strategy that refines interpolated frame poses, and an uncertainty distillation method that adaptively extracts target content while preserving well-reconstructed regions. Extensive experiments demonstrate that our method significantly enhances dynamic modeling, especially for fast-moving objects, achieving an approximate PSNR gain of 2 dB for novel view synthesis over baseline approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08524v2">GS-ID: Illumination Decomposition on Gaussian Splatting via Adaptive Light Aggregation and Diffusion-Guided Material Priors</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 17 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has emerged as an effective representation for photorealistic rendering, but the underlying geometry, material, and lighting remain entangled, hindering scene editing. Existing GS-based methods struggle to disentangle these components under non-Lambertian conditions, especially in the presence of specularities and shadows. We propose \textbf{GS-ID}, an end-to-end framework for illumination decomposition that integrates adaptive light aggregation with diffusion-based material priors. In addition to a learnable environment map for ambient illumination, we model spatially-varying local lighting using anisotropic spherical Gaussian mixtures (SGMs) that are jointly optimized with scene content. To better capture cast shadows, we associate each splat with a learnable unit vector that encodes shadow directions from multiple light sources, further improving material and lighting estimation. By combining SGMs with intrinsic priors from diffusion models, GS-ID significantly reduces ambiguity in light-material-geometry interactions and achieves state-of-the-art performance on inverse rendering and relighting benchmarks. Experiments also demonstrate the effectiveness of GS-ID for downstream applications such as relighting and scene composition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18458v3">StableGS: A Floater-Free Framework for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) reconstructions are plagued by stubborn ``floater" artifacts that degrade their geometric and visual fidelity. We are the first to reveal the root cause: a fundamental conflict in the 3DGS optimization process where the opacity gradients of floaters vanish when their blended color reaches a pseudo-equilibrium of canceling errors against the background, trapping them in a spurious local minimum. To resolve this, we propose StableGS, a novel framework that decouples geometric regularization from final appearance rendering. Its core is a Dual Opacity architecture that creates two separate rendering paths: a ``Geometric Regularization Path" to bear strong depth-based constraints for structural correctness, and an ``Appearance Refinement Path" to generate high-fidelity details upon this stable foundation. We complement this with a synergistic set of geometric constraints: a self-supervised depth consistency loss and an external geometric prior enabled by our efficient global scale optimization algorithm. Experiments on multiple benchmarks show StableGS not only eliminates floaters but also resolves the common blur-artifact trade-off, achieving state-of-the-art geometric accuracy and visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08137v2">Occlusion-Aware Temporally Consistent Amodal Completion for 3D Human-Object Interaction Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 ACM MM 2025
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for reconstructing dynamic human-object interactions from monocular video that overcomes challenges associated with occlusions and temporal inconsistencies. Traditional 3D reconstruction methods typically assume static objects or full visibility of dynamic subjects, leading to degraded performance when these assumptions are violated-particularly in scenarios where mutual occlusions occur. To address this, our framework leverages amodal completion to infer the complete structure of partially obscured regions. Unlike conventional approaches that operate on individual frames, our method integrates temporal context, enforcing coherence across video sequences to incrementally refine and stabilize reconstructions. This template-free strategy adapts to varying conditions without relying on predefined models, significantly enhancing the recovery of intricate details in dynamic scenes. We validate our approach using 3D Gaussian Splatting on challenging monocular videos, demonstrating superior precision in handling occlusions and maintaining temporal stability compared to existing techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01965v1">From Photons to Physics: Autonomous Indoor Drones and the Future of Objective Property Assessment</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 63 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The convergence of autonomous indoor drones with physics-aware sensing technologies promises to transform property assessment from subjective visual inspection to objective, quantitative measurement. This comprehensive review examines the technical foundations enabling this paradigm shift across four critical domains: (1) platform architectures optimized for indoor navigation, where weight constraints drive innovations in heterogeneous computing, collision-tolerant design, and hierarchical control systems; (2) advanced sensing modalities that extend perception beyond human vision, including hyperspectral imaging for material identification, polarimetric sensing for surface characterization, and computational imaging with metaphotonics enabling radical miniaturization; (3) intelligent autonomy through active reconstruction algorithms, where drones equipped with 3D Gaussian Splatting make strategic decisions about viewpoint selection to maximize information gain within battery constraints; and (4) integration pathways with existing property workflows, including Building Information Modeling (BIM) systems and industry standards like Uniform Appraisal Dataset (UAD) 3.6.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01647v2">FlowR: Flowing from Sparse to Dense 3D Reconstructions</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 ICCV 2025 Highlight. Project page is available at https://tobiasfshr.github.io/pub/flowr
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting enables high-quality novel view synthesis (NVS) at real-time frame rates. However, its quality drops sharply as we depart from the training views. Thus, dense captures are needed to match the high-quality expectations of applications like Virtual Reality (VR). However, such dense captures are very laborious and expensive to obtain. Existing works have explored using 2D generative models to alleviate this requirement by distillation or generating additional training views. These models typically rely on a noise-to-data generative process conditioned only on a handful of reference input views, leading to hallucinations, inconsistent generation results, and subsequent reconstruction artifacts. Instead, we propose a multi-view, flow matching model that learns a flow to directly connect novel view renderings from possibly sparse reconstructions to renderings that we expect from dense reconstructions. This enables augmenting scene captures with consistent, generated views to improve reconstruction quality. Our model is trained on a novel dataset of 3.6M image pairs and can process up to 45 views at 540x960 resolution (91K tokens) on one H100 GPU in a single forward pass. Our pipeline consistently improves NVS in sparse- and dense-view scenarios, leading to higher-quality reconstructions than prior works across multiple, widely-used NVS benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02831v1">GENIE: Gaussian Encoding for Neural Radiance Fields Interactive Editing</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Neural Radiance Fields (NeRF) and Gaussian Splatting (GS) have recently transformed 3D scene representation and rendering. NeRF achieves high-fidelity novel view synthesis by learning volumetric representations through neural networks, but its implicit encoding makes editing and physical interaction challenging. In contrast, GS represents scenes as explicit collections of Gaussian primitives, enabling real-time rendering, faster training, and more intuitive manipulation. This explicit structure has made GS particularly well-suited for interactive editing and integration with physics-based simulation. In this paper, we introduce GENIE (Gaussian Encoding for Neural Radiance Fields Interactive Editing), a hybrid model that combines the photorealistic rendering quality of NeRF with the editable and structured representation of GS. Instead of using spherical harmonics for appearance modeling, we assign each Gaussian a trainable feature embedding. These embeddings are used to condition a NeRF network based on the k nearest Gaussians to each query point. To make this conditioning efficient, we introduce Ray-Traced Gaussian Proximity Search (RT-GPS), a fast nearest Gaussian search based on a modified ray-tracing pipeline. We also integrate a multi-resolution hash grid to initialize and update Gaussian features. Together, these components enable real-time, locality-aware editing: as Gaussian primitives are repositioned or modified, their interpolated influence is immediately reflected in the rendered output. By combining the strengths of implicit and explicit representations, GENIE supports intuitive scene manipulation, dynamic interaction, and compatibility with physical simulation, bridging the gap between geometry-based editing and neural rendering. The code can be found under (https://github.com/MikolajZielinski/genie)
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17505v2">PLGS: Robust Panoptic Lifting with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      Previous methods utilize the Neural Radiance Field (NeRF) for panoptic lifting, while their training and rendering speed are unsatisfactory. In contrast, 3D Gaussian Splatting (3DGS) has emerged as a prominent technique due to its rapid training and rendering speed. However, unlike NeRF, the conventional 3DGS may not satisfy the basic smoothness assumption as it does not rely on any parameterized structures to render (e.g., MLPs). Consequently, the conventional 3DGS is, in nature, more susceptible to noisy 2D mask supervision. In this paper, we propose a new method called PLGS that enables 3DGS to generate consistent panoptic segmentation masks from noisy 2D segmentation masks while maintaining superior efficiency compared to NeRF-based methods. Specifically, we build a panoptic-aware structured 3D Gaussian model to introduce smoothness and design effective noise reduction strategies. For the semantic field, instead of initialization with structure from motion, we construct reliable semantic anchor points to initialize the 3D Gaussians. We then use these anchor points as smooth regularization during training. Additionally, we present a self-training approach using pseudo labels generated by merging the rendered masks with the noisy masks to enhance the robustness of PLGS. For the instance field, we project the 2D instance masks into 3D space and match them with oriented bounding boxes to generate cross-view consistent instance masks for supervision. Experiments on various benchmarks demonstrate that our method outperforms previous state-of-the-art methods in terms of both segmentation quality and speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05256v2">SegmentDreamer: Towards High-fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation</a></div>
    <div class="paper-meta">
      📅 2025-08-03
      | 💬 Accepted by ICCV 2025, project page: https://zjhjojo.github.io/segmentdreamer/
    </div>
    <details class="paper-abstract">
      Recent advancements in text-to-3D generation improve the visual quality of Score Distillation Sampling (SDS) and its variants by directly connecting Consistency Distillation (CD) to score distillation. However, due to the imbalance between self-consistency and cross-consistency, these CD-based methods inherently suffer from improper conditional guidance, leading to sub-optimal generation results. To address this issue, we present SegmentDreamer, a novel framework designed to fully unleash the potential of consistency models for high-fidelity text-to-3D generation. Specifically, we reformulate SDS through the proposed Segmented Consistency Trajectory Distillation (SCTD), effectively mitigating the imbalance issues by explicitly defining the relationship between self- and cross-consistency. Moreover, SCTD partitions the Probability Flow Ordinary Differential Equation (PF-ODE) trajectory into multiple sub-trajectories and ensures consistency within each segment, which can theoretically provide a significantly tighter upper bound on distillation error. Additionally, we propose a distillation pipeline for a more swift and stable generation. Extensive experiments demonstrate that our SegmentDreamer outperforms state-of-the-art methods in visual quality, enabling high-fidelity 3D asset creation through 3D Gaussian Splatting (3DGS).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01740v1">AG$^2$aussian: Anchor-Graph Structured Gaussian Splatting for Instance-Level 3D Scene Understanding and Editing</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has witnessed exponential adoption across diverse applications, driving a critical need for semantic-aware 3D Gaussian representations to enable scene understanding and editing tasks. Existing approaches typically attach semantic features to a collection of free Gaussians and distill the features via differentiable rendering, leading to noisy segmentation and a messy selection of Gaussians. In this paper, we introduce AG$^2$aussian, a novel framework that leverages an anchor-graph structure to organize semantic features and regulate Gaussian primitives. Our anchor-graph structure not only promotes compact and instance-aware Gaussian distributions, but also facilitates graph-based propagation, achieving a clean and accurate instance-level Gaussian selection. Extensive validation across four applications, i.e. interactive click-based query, open-vocabulary text-driven query, object removal editing, and physics simulation, demonstrates the advantages of our approach and its benefits to various applications. The experiments and ablation studies further evaluate the effectiveness of the key designs of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01704v1">LT-Gaussian: Long-Term Map Update Using 3D Gaussian Splatting for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-08-03
      | 💬 Accepted by IV 2025
    </div>
    <details class="paper-abstract">
      Maps play an important role in autonomous driving systems. The recently proposed 3D Gaussian Splatting (3D-GS) produces rendering-quality explicit scene reconstruction results, demonstrating the potential for map construction in autonomous driving scenarios. However, because of the time and computational costs involved in generating Gaussian scenes, how to update the map becomes a significant challenge. In this paper, we propose LT-Gaussian, a map update method for 3D-GS-based maps. LT-Gaussian consists of three main components: Multimodal Gaussian Splatting, Structural Change Detection Module, and Gaussian-Map Update Module. Firstly, the Gaussian map of the old scene is generated using our proposed Multimodal Gaussian Splatting. Subsequently, during the map update process, we compare the outdated Gaussian map with the current LiDAR data stream to identify structural changes. Finally, we perform targeted updates to the Gaussian-map to generate an up-to-date map. We establish a benchmark for map updating on the nuScenes dataset to quantitatively evaluate our method. The experimental results show that LT-Gaussian can effectively and efficiently update the Gaussian-map, handling common environmental changes in autonomous driving scenarios. Furthermore, by taking full advantage of information from both new and old scenes, LT-Gaussian is able to produce higher quality reconstruction results compared to map update strategies that reconstruct maps from scratch. Our open-source code is available at https://github.com/ChengLuqi/LT-gaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01684v1">DisCo3D: Distilling Multi-View Consistency for 3D Scene Editing</a></div>
    <div class="paper-meta">
      📅 2025-08-03
      | 💬 17 pages, 7 figures
    </div>
    <details class="paper-abstract">
      While diffusion models have demonstrated remarkable progress in 2D image generation and editing, extending these capabilities to 3D editing remains challenging, particularly in maintaining multi-view consistency. Classical approaches typically update 3D representations through iterative refinement based on a single editing view. However, these methods often suffer from slow convergence and blurry artifacts caused by cross-view inconsistencies. Recent methods improve efficiency by propagating 2D editing attention features, yet still exhibit fine-grained inconsistencies and failure modes in complex scenes due to insufficient constraints. To address this, we propose \textbf{DisCo3D}, a novel framework that distills 3D consistency priors into a 2D editor. Our method first fine-tunes a 3D generator using multi-view inputs for scene adaptation, then trains a 2D editor through consistency distillation. The edited multi-view outputs are finally optimized into 3D representations via Gaussian Splatting. Experimental results show DisCo3D achieves stable multi-view consistency and outperforms state-of-the-art methods in editing quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11839v2">RoboGSim: A Real2Sim2Real Robotic Gaussian Splatting Simulator</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      Efficient acquisition of real-world embodied data has been increasingly critical. However, large-scale demonstrations captured by remote operation tend to take extremely high costs and fail to scale up the data size in an efficient manner. Sampling the episodes under a simulated environment is a promising way for large-scale collection while existing simulators fail to high-fidelity modeling on texture and physics. To address these limitations, we introduce the RoboGSim, a real2sim2real robotic simulator, powered by 3D Gaussian Splatting and the physics engine. RoboGSim mainly includes four parts: Gaussian Reconstructor, Digital Twins Builder, Scene Composer, and Interactive Engine. It can synthesize the simulated data with novel views, objects, trajectories, and scenes. RoboGSim also provides an online, reproducible, and safe evaluation for different manipulation policies. The real2sim and sim2real transfer experiments show a high consistency in the texture and physics. We compared the test results of RoboGSim data and real robot data on both RoboGSim and real robot platforms. The experimental results show that the RoboGSim data model can achieve zero-shot performance on the real robot, with results comparable to real robot data. Additionally, in experiments with novel perspectives and novel scenes, the RoboGSim data model performed even better on the real robot than the real robot data model. This not only helps reduce the sim2real gap but also addresses the limitations of real robot data collection, such as its single-source and high cost. We hope RoboGSim serves as a closed-loop simulator for fair comparison on policy learning. More information can be found on our project page https://robogsim.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12799v2">TSGS: Improving Gaussian Splatting for Transparent Surface Reconstruction via Normal and De-lighting Priors</a></div>
    <div class="paper-meta">
      📅 2025-08-03
      | 💬 Project page: https://longxiang-ai.github.io/TSGS/ . Accepted by ACM MM 2025
    </div>
    <details class="paper-abstract">
      Reconstructing transparent surfaces is essential for tasks such as robotic manipulation in labs, yet it poses a significant challenge for 3D reconstruction techniques like 3D Gaussian Splatting (3DGS). These methods often encounter a transparency-depth dilemma, where the pursuit of photorealistic rendering through standard $\alpha$-blending undermines geometric precision, resulting in considerable depth estimation errors for transparent materials. To address this issue, we introduce Transparent Surface Gaussian Splatting (TSGS), a new framework that separates geometry learning from appearance refinement. In the geometry learning stage, TSGS focuses on geometry by using specular-suppressed inputs to accurately represent surfaces. In the second stage, TSGS improves visual fidelity through anisotropic specular modeling, crucially maintaining the established opacity to ensure geometric accuracy. To enhance depth inference, TSGS employs a first-surface depth extraction method. This technique uses a sliding window over $\alpha$-blending weights to pinpoint the most likely surface location and calculates a robust weighted average depth. To evaluate the transparent surface reconstruction task under realistic conditions, we collect a TransLab dataset that includes complex transparent laboratory glassware. Extensive experiments on TransLab show that TSGS achieves accurate geometric reconstruction and realistic rendering of transparent objects simultaneously within the efficient 3DGS framework. Specifically, TSGS significantly surpasses current leading methods, achieving a 37.3% reduction in chamfer distance and an 8.0% improvement in F1 score compared to the top baseline. The code and dataset are available at https://longxiang-ai.github.io/TSGS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.08742v4">Efficient4D: Fast Dynamic 3D Object Generation from a Single-view Video</a></div>
    <div class="paper-meta">
      📅 2025-08-03
      | 💬 Technical report
    </div>
    <details class="paper-abstract">
      Generating dynamic 3D object from a single-view video is challenging due to the lack of 4D labeled data. An intuitive approach is to extend previous image-to-3D pipelines by transferring off-the-shelf image generation models such as score distillation sampling.However, this approach would be slow and expensive to scale due to the need for back-propagating the information-limited supervision signals through a large pretrained model. To address this, we propose an efficient video-to-4D object generation framework called Efficient4D. It generates high-quality spacetime-consistent images under different camera views, and then uses them as labeled data to directly reconstruct the 4D content through a 4D Gaussian splatting model. Importantly, our method can achieve real-time rendering under continuous camera trajectories. To enable robust reconstruction under sparse views, we introduce inconsistency-aware confidence-weighted loss design, along with a lightly weighted score distillation loss. Extensive experiments on both synthetic and real videos show that Efficient4D offers a remarkable 10-fold increase in speed when compared to prior art alternatives while preserving the quality of novel view synthesis. For example, Efficient4D takes only 10 minutes to model a dynamic object, vs 120 minutes by the previous art model Consistent4D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19718v2">GSCache: Real-Time Radiance Caching for Volume Path Tracing using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-02
    </div>
    <details class="paper-abstract">
      Real-time path tracing is rapidly becoming the standard for rendering in entertainment and professional applications. In scientific visualization, volume rendering plays a crucial role in helping researchers analyze and interpret complex 3D data. Recently, photorealistic rendering techniques have gained popularity in scientific visualization, yet they face significant challenges. One of the most prominent issues is slow rendering performance and high pixel variance caused by Monte Carlo integration. In this work, we introduce a novel radiance caching approach for path-traced volume rendering. Our method leverages advances in volumetric scene representation and adapts 3D Gaussian splatting to function as a multi-level, path-space radiance cache. This cache is designed to be trainable on the fly, dynamically adapting to changes in scene parameters such as lighting configurations and transfer functions. By incorporating our cache, we achieve less noisy, higher-quality images without increasing rendering costs. To evaluate our approach, we compare it against a baseline path tracer that supports uniform sampling and next-event estimation and the state-of-the-art for neural radiance caching. Through both quantitative and qualitative analyses, we demonstrate that our path-space radiance cache is a robust solution that is easy to integrate and significantly enhances the rendering quality of volumetric visualization applications while maintaining comparable computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01464v1">Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-08-02
    </div>
    <details class="paper-abstract">
      3D generation has made significant progress, however, it still largely remains at the object-level. Feedforward 3D scene-level generation has been rarely explored due to the lack of models capable of scaling-up latent representation learning on 3D scene-level data. Unlike object-level generative models, which are trained on well-labeled 3D data in a bounded canonical space, scene-level generations with 3D scenes represented by 3D Gaussian Splatting (3DGS) are unbounded and exhibit scale inconsistency across different scenes, making unified latent representation learning for generative purposes extremely challenging. In this paper, we introduce Can3Tok, the first 3D scene-level variational autoencoder (VAE) capable of encoding a large number of Gaussian primitives into a low-dimensional latent embedding, which effectively captures both semantic and spatial information of the inputs. Beyond model design, we propose a general pipeline for 3D scene data processing to address scale inconsistency issue. We validate our method on the recent scene-level 3D dataset DL3DV-10K, where we found that only Can3Tok successfully generalizes to novel 3D scenes, while compared methods fail to converge on even a few hundred scene inputs during training and exhibit zero generalization ability during inference. Finally, we demonstrate image-to-3DGS and text-to-3DGS generation as our applications to demonstrate its ability to facilitate downstream generation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19451v3">GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-02
      | 💬 ICCV 2025. Project Page: https://gs-occ3d.github.io/
    </div>
    <details class="paper-abstract">
      Occupancy is crucial for autonomous driving, providing essential geometric priors for perception and planning. However, existing methods predominantly rely on LiDAR-based occupancy annotations, which limits scalability and prevents leveraging vast amounts of potential crowdsourced data for auto-labeling. To address this, we propose GS-Occ3D, a scalable vision-only framework that directly reconstructs occupancy. Vision-only occupancy reconstruction poses significant challenges due to sparse viewpoints, dynamic scene elements, severe occlusions, and long-horizon motion. Existing vision-based methods primarily rely on mesh representation, which suffer from incomplete geometry and additional post-processing, limiting scalability. To overcome these issues, GS-Occ3D optimizes an explicit occupancy representation using an Octree-based Gaussian Surfel formulation, ensuring efficiency and scalability. Additionally, we decompose scenes into static background, ground, and dynamic objects, enabling tailored modeling strategies: (1) Ground is explicitly reconstructed as a dominant structural element, significantly improving large-area consistency; (2) Dynamic vehicles are separately modeled to better capture motion-related occupancy patterns. Extensive experiments on the Waymo dataset demonstrate that GS-Occ3D achieves state-of-the-art geometry reconstruction results. By curating vision-only binary occupancy labels from diverse urban scenes, we show their effectiveness for downstream occupancy models on Occ3D-Waymo and superior zero-shot generalization on Occ3D-nuScenes. It highlights the potential of large-scale vision-based occupancy reconstruction as a new paradigm for scalable auto-labeling. Project Page: https://gs-occ3d.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04981v3">AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-02
      | 💬 ICCV 2025 Hightlight (main conference)
    </div>
    <details class="paper-abstract">
      Obtaining high-quality 3D semantic occupancy from raw sensor data remains an essential yet challenging task, often requiring extensive manual labeling. In this work, we propose AutoOcc, a vision-centric automated pipeline for open-ended semantic occupancy annotation that integrates differentiable Gaussian splatting guided by vision-language models. We formulate the open-ended semantic 3D occupancy reconstruction task to automatically generate scene occupancy by combining attention maps from vision-language models and foundation vision models. We devise semantic-aware Gaussians as intermediate geometric descriptors and propose a cumulative Gaussian-to-voxel splatting algorithm that enables effective and efficient occupancy annotation. Our framework outperforms existing automated occupancy annotation methods without human labels. AutoOcc also enables open-ended semantic occupancy auto-labeling, achieving robust performance in both static and dynamically complex scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01239v1">OCSplats: Observation Completeness Quantification and Label Noise Separation in 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-08-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become one of the most promising 3D reconstruction technologies. However, label noise in real-world scenarios-such as moving objects, non-Lambertian surfaces, and shadows-often leads to reconstruction errors. Existing 3DGS-Bsed anti-noise reconstruction methods either fail to separate noise effectively or require scene-specific fine-tuning of hyperparameters, making them difficult to apply in practice. This paper re-examines the problem of anti-noise reconstruction from the perspective of epistemic uncertainty, proposing a novel framework, OCSplats. By combining key technologies such as hybrid noise assessment and observation-based cognitive correction, the accuracy of noise classification in areas with cognitive differences has been significantly improved. Moreover, to address the issue of varying noise proportions in different scenarios, we have designed a label noise classification pipeline based on dynamic anchor points. This pipeline enables OCSplats to be applied simultaneously to scenarios with vastly different noise proportions without adjusting parameters. Extensive experiments demonstrate that OCSplats always achieve leading reconstruction performance and precise label noise classification in scenes of different complexity levels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04844v2">Embracing Dynamics: Dynamics-aware 4D Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-08-02
      | 💬 This paper has been accepted by IROS 2025
    </div>
    <details class="paper-abstract">
      Simultaneous localization and mapping (SLAM) technology has recently achieved photorealistic mapping capabilities thanks to the real-time, high-fidelity rendering enabled by 3D Gaussian Splatting (3DGS). However, due to the static representation of scenes, current 3DGS-based SLAM encounters issues with pose drift and failure to reconstruct accurate maps in dynamic environments. To address this problem, we present D4DGS-SLAM, the first SLAM method based on 4DGS map representation for dynamic environments. By incorporating the temporal dimension into scene representation, D4DGS-SLAM enables high-quality reconstruction of dynamic scenes. Utilizing the dynamics-aware InfoModule, we can obtain the dynamics, visibility, and reliability of scene points, and filter out unstable dynamic points for tracking accordingly. When optimizing Gaussian points, we apply different isotropic regularization terms to Gaussians with varying dynamic characteristics. Experimental results on real-world dynamic scene datasets demonstrate that our method outperforms state-of-the-art approaches in both camera pose tracking and map quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04887v2">Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has demonstrated notable success in large-scale scene reconstruction, but challenges persist due to high training memory consumption and storage overhead. Hybrid representations that integrate implicit and explicit features offer a way to mitigate these limitations. However, when applied in parallelized block-wise training, two critical issues arise since reconstruction accuracy deteriorates due to reduced data diversity when training each block independently, and parallel training restricts the number of divided blocks to the available number of GPUs. To address these issues, we propose Momentum-GS, a novel approach that leverages momentum-based self-distillation to promote consistency and accuracy across the blocks while decoupling the number of blocks from the physical GPU count. Our method maintains a teacher Gaussian decoder updated with momentum, ensuring a stable reference during training. This teacher provides each block with global guidance in a self-distillation manner, promoting spatial consistency in reconstruction. To further ensure consistency across the blocks, we incorporate block weighting, dynamically adjusting each block's weight according to its reconstruction accuracy. Extensive experiments on large-scale scenes show that our method consistently outperforms existing techniques, achieving a 12.8% improvement in LPIPS over CityGaussian with much fewer divided blocks and establishing a new state of the art. Project page: https://jixuan-fan.github.io/Momentum-GS_Page/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01171v1">No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views</a></div>
    <div class="paper-meta">
      📅 2025-08-02
      | 💬 Project Page: https://ranrhuang.github.io/spfsplat/
    </div>
    <details class="paper-abstract">
      We introduce SPFSplat, an efficient framework for 3D Gaussian splatting from sparse multi-view images, requiring no ground-truth poses during training or inference. It employs a shared feature extraction backbone, enabling simultaneous prediction of 3D Gaussian primitives and camera poses in a canonical space from unposed inputs within a single feed-forward step. Alongside the rendering loss based on estimated novel-view poses, a reprojection loss is integrated to enforce the learning of pixel-aligned Gaussian primitives for enhanced geometric constraints. This pose-free training paradigm and efficient one-step feed-forward design make SPFSplat well-suited for practical applications. Remarkably, despite the absence of pose supervision, SPFSplat achieves state-of-the-art performance in novel view synthesis even under significant viewpoint changes and limited image overlap. It also surpasses recent methods trained with geometry priors in relative pose estimation. Code and trained models are available on our project page: https://ranrhuang.github.io/spfsplat/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01150v1">OpenGS-Fusion: Open-Vocabulary Dense Mapping with Hybrid 3D Gaussian Splatting for Refined Object-Level Understanding</a></div>
    <div class="paper-meta">
      📅 2025-08-02
      | 💬 IROS2025
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D scene understanding have made significant strides in enabling interaction with scenes using open-vocabulary queries, particularly for VR/AR and robotic applications. Nevertheless, existing methods are hindered by rigid offline pipelines and the inability to provide precise 3D object-level understanding given open-ended queries. In this paper, we present OpenGS-Fusion, an innovative open-vocabulary dense mapping framework that improves semantic modeling and refines object-level understanding. OpenGS-Fusion combines 3D Gaussian representation with a Truncated Signed Distance Field to facilitate lossless fusion of semantic features on-the-fly. Furthermore, we introduce a novel multimodal language-guided approach named MLLM-Assisted Adaptive Thresholding, which refines the segmentation of 3D objects by adaptively adjusting similarity thresholds, achieving an improvement 17\% in 3D mIoU compared to the fixed threshold strategy. Extensive experiments demonstrate that our method outperforms existing methods in 3D object understanding and scene reconstruction quality, as well as showcasing its effectiveness in language-guided scene interaction. The code is available at https://young-bit.github.io/opengs-fusion.github.io/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12811v2">AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Although 3D Gaussian Splatting (3DGS) has revolutionized 3D reconstruction, it still faces challenges such as aliasing, projection artifacts, and view inconsistencies, primarily due to the simplification of treating splats as 2D entities. We argue that incorporating full 3D evaluation of Gaussians throughout the 3DGS pipeline can effectively address these issues while preserving rasterization efficiency. Specifically, we introduce an adaptive 3D smoothing filter to mitigate aliasing and present a stable view-space bounding method that eliminates popping artifacts when Gaussians extend beyond the view frustum. Furthermore, we promote tile-based culling to 3D with screen-space planes, accelerating rendering and reducing sorting costs for hierarchical rasterization. Our method achieves state-of-the-art quality on in-distribution evaluation sets and significantly outperforms other approaches for out-of-distribution views. Our qualitative evaluations further demonstrate the effective removal of aliasing, distortions, and popping artifacts, ensuring real-time, artifact-free rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00354v1">Omni-Scan: Creating Visually-Accurate Digital Twin Object Models Using a Bimanual Robot with Handover and Gaussian Splat Merging</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      3D Gaussian Splats (3DGSs) are 3D object models derived from multi-view images. Such "digital twins" are useful for simulations, virtual reality, marketing, robot policy fine-tuning, and part inspection. 3D object scanning usually requires multi-camera arrays, precise laser scanners, or robot wrist-mounted cameras, which have restricted workspaces. We propose Omni-Scan, a pipeline for producing high-quality 3D Gaussian Splat models using a bi-manual robot that grasps an object with one gripper and rotates the object with respect to a stationary camera. The object is then re-grasped by a second gripper to expose surfaces that were occluded by the first gripper. We present the Omni-Scan robot pipeline using DepthAny-thing, Segment Anything, as well as RAFT optical flow models to identify and isolate objects held by a robot gripper while removing the gripper and the background. We then modify the 3DGS training pipeline to support concatenated datasets with gripper occlusion, producing an omni-directional (360 degree view) model of the object. We apply Omni-Scan to part defect inspection, finding that it can identify visual or geometric defects in 12 different industrial and household objects with an average accuracy of 83%. Interactive videos of Omni-Scan 3DGS models can be found at https://berkeleyautomation.github.io/omni-scan/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12781v2">Long-LRM: Long-sequence Large Reconstruction Model for Wide-coverage Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      We propose Long-LRM, a feed-forward 3D Gaussian reconstruction model for instant, high-resolution, 360{\deg} wide-coverage, scene-level reconstruction. Specifically, it takes in 32 input images at a resolution of 960x540 and produces the Gaussian reconstruction in just 1 second on a single A100 GPU. To handle the long sequence of 250K tokens brought by the large input size, Long-LRM features a mixture of the recent Mamba2 blocks and the classical transformer blocks, enhanced by a light-weight token merging module and Gaussian pruning steps that balance between quality and efficiency. We evaluate Long-LRM on the large-scale DL3DV benchmark and Tanks&Temples, demonstrating reconstruction quality comparable to the optimization-based methods while achieving an 800x speedup w.r.t. the optimization-based approaches and an input size at least 60x larger than the previous feed-forward approaches. We conduct extensive ablation studies on our model design choices for both rendering quality and computation efficiency. We also explore Long-LRM's compatibility with other Gaussian variants such as 2D GS, which enhances Long-LRM's ability in geometry reconstruction. Project page: https://arthurhero.github.io/projects/llrm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17812v2">FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 ICCV 2025 Camera-Ready Version. Project Page: https://weijielyu.github.io/FaceLift
    </div>
    <details class="paper-abstract">
      We present FaceLift, a novel feed-forward approach for generalizable high-quality 360-degree 3D head reconstruction from a single image. Our pipeline first employs a multi-view latent diffusion model to generate consistent side and back views from a single facial input, which then feeds into a transformer-based reconstructor that produces a comprehensive 3D Gaussian splats representation. Previous methods for monocular 3D face reconstruction often lack full view coverage or view consistency due to insufficient multi-view supervision. We address this by creating a high-quality synthetic head dataset that enables consistent supervision across viewpoints. To bridge the domain gap between synthetic training data and real-world images, we propose a simple yet effective technique that ensures the view generation process maintains fidelity to the input by learning to reconstruct the input image alongside the view generation. Despite being trained exclusively on synthetic data, our method demonstrates remarkable generalization to real-world images. Through extensive qualitative and quantitative evaluations, we show that FaceLift outperforms state-of-the-art 3D face reconstruction methods on identity preservation, detail recovery, and rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00259v1">PointGauss: Point Cloud-Guided Multi-Object Segmentation for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 22 pages, 9 figures
    </div>
    <details class="paper-abstract">
      We introduce PointGauss, a novel point cloud-guided framework for real-time multi-object segmentation in Gaussian Splatting representations. Unlike existing methods that suffer from prolonged initialization and limited multi-view consistency, our approach achieves efficient 3D segmentation by directly parsing Gaussian primitives through a point cloud segmentation-driven pipeline. The key innovation lies in two aspects: (1) a point cloud-based Gaussian primitive decoder that generates 3D instance masks within 1 minute, and (2) a GPU-accelerated 2D mask rendering system that ensures multi-view consistency. Extensive experiments demonstrate significant improvements over previous state-of-the-art methods, achieving performance gains of 1.89 to 31.78% in multi-view mIoU, while maintaining superior computational efficiency. To address the limitations of current benchmarks (single-object focus, inconsistent 3D evaluation, small scale, and partial coverage), we present DesktopObjects-360, a novel comprehensive dataset for 3D segmentation in radiance fields, featuring: (1) complex multi-object scenes, (2) globally consistent 2D annotations, (3) large-scale training data (over 27 thousand 2D masks), (4) full 360{\deg} coverage, and (5) 3D evaluation masks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01109v2">CountingFruit: Language-Guided 3D Fruit Counting with Semantic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Accurate 3D fruit counting in orchards is challenging due to heavy occlusion, semantic ambiguity between fruits and surrounding structures, and the high computational cost of volumetric reconstruction. Existing pipelines often rely on multi-view 2D segmentation and dense volumetric sampling, which lead to accumulated fusion errors and slow inference. We introduce FruitLangGS, a language-guided 3D fruit counting framework that reconstructs orchard-scale scenes using an adaptive-density Gaussian Splatting pipeline with radius-aware pruning and tile-based rasterization, enabling scalable 3D representation. During inference, compressed CLIP-aligned semantic vectors embedded in each Gaussian are filtered via a dual-threshold cosine similarity mechanism, retrieving Gaussians relevant to target prompts while suppressing common distractors (e.g., foliage), without requiring retraining or image-space masks. The selected Gaussians are then sampled into dense point clouds and clustered geometrically to estimate fruit instances, remaining robust under severe occlusion and viewpoint variation. Experiments on nine different orchard-scale datasets demonstrate that FruitLangGS consistently outperforms existing pipelines in instance counting recall, avoiding multi-view segmentation fusion errors and achieving up to 99.2\% recall on Fuji-SfM orchard dataset. Ablation studies further confirm that language-conditioned semantic embedding and dual-threshold prompt filtering are essential for suppressing distractors and improving counting accuracy under heavy occlusion. Beyond fruit counting, the same framework enables prompt-driven 3D semantic retrieval without retraining, highlighting the potential of language-guided 3D perception for scalable agricultural scene understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06271v2">SplatTalk: 3D VQA with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 Accepted at ICCV 2025
    </div>
    <details class="paper-abstract">
      Language-guided 3D scene understanding is important for advancing applications in robotics, AR/VR, and human-computer interaction, enabling models to comprehend and interact with 3D environments through natural language. While 2D vision-language models (VLMs) have achieved remarkable success in 2D VQA tasks, progress in the 3D domain has been significantly slower due to the complexity of 3D data and the high cost of manual annotations. In this work, we introduce SplatTalk, a novel method that uses a generalizable 3D Gaussian Splatting (3DGS) framework to produce 3D tokens suitable for direct input into a pretrained LLM, enabling effective zero-shot 3D visual question answering (3D VQA) for scenes with only posed images. During experiments on multiple benchmarks, our approach outperforms both 3D models trained specifically for the task and previous 2D-LMM-based models utilizing only images (our setting), while achieving competitive performance with state-of-the-art 3D LMMs that additionally utilize 3D inputs. Project website: https://splat-talk.github.io/
    </details>
</div>
