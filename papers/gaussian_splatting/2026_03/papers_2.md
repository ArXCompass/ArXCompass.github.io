# gaussian splatting - 2026_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.13558v3">GoDe: Gaussians on Demand for Progressive Level of Detail and Scalable Compression</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Recent progress in compressing explicit radiance field representations, particularly 3D Gaussian Splatting, has substantially reduced memory consumption while improving real-time rendering performance. However, existing approaches remain inherently single-rate: each compression level requires a separately optimized model, yielding a set of fixed operating points rather than a truly scalable representation. This limits deployment in scenarios where memory, bandwidth, or computational budgets vary across devices or over time. We argue that scalability should be an intrinsic property of the representation. We show that trained explicit radiance models exhibit a structured distribution of information, which can be revealed using standard optimization signals available during training. In particular, aggregated gradient sensitivity provides a simple, model-agnostic criterion to organize primitives from coarse structure to finer refinements. Building on this, we introduce GoDe (Gaussians on Demand), a general framework for scalable compression and progressive level-of-detail control, instantiated for 3D Gaussian Splatting. Starting from a single trained model, GoDe reorganizes Gaussian primitives into a fixed progressive hierarchy supporting multiple rate-distortion operating points without retraining or per-level fine-tuning. A single quantization-aware fine-tuning stage ensures consistent behavior across all levels under low-precision storage. Extensive experiments on standard benchmarks and multiple 3D Gaussian Splatting backbones show that GoDe achieves rate-distortion performance comparable to state-of-the-art single-rate methods, while enabling truly scalable compression and adaptive rendering within a unified representation. Project page: https://gaussians-on-demand.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05738v2">FeatureSLAM: Feature-enriched 3D gaussian splatting SLAM in real time</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      We present a real-time tracking SLAM system that unifies efficient camera tracking with photorealistic feature-enriched mapping using 3D Gaussian Splatting (3DGS). Our main contribution is integrating dense feature rasterization into the novel-view synthesis, aligned with a visual foundation model. This yields strong semantics, going beyond basic RGB-D input, aiding both tracking and mapping accuracy. Unlike previous semantic SLAM approaches (which embed pre-defined class labels) FeatureSLAM enables entirely new downstream tasks via free-viewpoint, open-set segmentation. Across standard benchmarks, our method achieves real-time tracking, on par with state-of-the-art systems while improving tracking stability and map fidelity without prohibitive compute. Quantitatively, we obtain 9\% lower pose error and 8\% higher mapping accuracy compared to recent fixed-set SLAM baselines. Our results confirm that real-time feature-embedded SLAM, is not only valuable for enabling new downstream applications. It also improves the performance of the underlying tracking and mapping subsystems, providing semantic and language masking results that are on-par with offline 3DGS models, alongside state-of-the-art tracking, depth and RGB rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18218v1">Semantic Segmentation and Depth Estimation for Real-Time Lunar Surface Mapping Using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      Navigation and mapping on the lunar surface require robust perception under challenging conditions, including poorly textured environments, high-contrast lighting, and limited computational resources. This paper presents a real-time mapping framework that integrates dense perception models with a 3D Gaussian Splatting (3DGS) representation. We first benchmark several models on synthetic datasets generated with the LuPNT simulator, selecting a stereo dense depth estimation model based on Gated Recurrent Units for its balance of speed and accuracy in depth estimation, and a convolutional neural network for its superior performance in detecting semantic segments. Using ground truth poses to decouple the local scene understanding from the global state estimation, our pipeline reconstructs a 120-meter traverse with a geometric height accuracy of approximately 3 cm, outperforming a traditional point cloud baseline without LiDAR. The resulting 3DGS map enables novel view synthesis and serves as a foundation for a full SLAM system, where its capacity for joint map and pose optimization would offer significant advantages. Our results demonstrate that combining semantic segmentation and dense depth estimation with learned map representations is an effective approach for creating detailed, large-scale maps to support future lunar surface missions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17975v1">AHOY! Animatable Humans under Occlusion from YouTube Videos with Gaussian Splatting and Video Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 Our project page is available at https://miraymen.github.io/ahoy/
    </div>
    <details class="paper-abstract">
      We present AHOY, a method for reconstructing complete, animatable 3D Gaussian avatars from in-the-wild monocular video despite heavy occlusion. Existing methods assume unoccluded input-a fully visible subject, often in a canonical pose-excluding the vast majority of real-world footage where people are routinely occluded by furniture, objects, or other people. Reconstructing from such footage poses fundamental challenges: large body regions may never be observed, and multi-view supervision per pose is unavailable. We address these challenges with four contributions: (i) a hallucination-as-supervision pipeline that uses identity-finetuned diffusion models to generate dense supervision for previously unobserved body regions; (ii) a two-stage canonical-to-pose-dependent architecture that bootstraps from sparse observations to full pose-dependent Gaussian maps; (iii) a map-pose/LBS-pose decoupling that absorbs multi-view inconsistencies from the generated data; (iv) a head/body split supervision strategy that preserves facial identity. We evaluate on YouTube videos and on multi-view capture data with significant occlusion and demonstrate state-of-the-art reconstruction quality. We also demonstrate that the resulting avatars are robust enough to be animated with novel poses and composited into 3DGS scenes captured using cell-phone video. Our project page is available at https://miraymen.github.io/ahoy/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17779v1">CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 Accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      Single-view 3D human reconstruction has garnered significant attention in recent years. Despite numerous advancements, prior research has concentrated on reconstructing 3D models from clear, close-up images of individual subjects, often yielding subpar results in the more prevalent multi-person scenarios. Reconstructing 3D human crowd models is a highly intricate task, laden with challenges such as: 1) extensive occlusions, 2) low clarity, and 3) numerous and various appearances. To address this task, we propose CrowdGaussian, a unified framework that directly reconstructs multi-person 3D Gaussian Splatting (3DGS) representations from single-image inputs. To handle occlusions, we devise a self-supervised adaptation pipeline that enables the pretrained large human model to reconstruct complete 3D humans with plausible geometry and appearance from heavily occluded inputs. Furthermore, we introduce Self-Calibrated Learning (SCL). This training strategy enables single-step diffusion models to adaptively refine coarse renderings to optimal quality by blending identity-preserving samples with clean/corrupted image pairs. The outputs can be distilled back to enhance the quality of multi-person 3DGS representations. Extensive experiments demonstrate that CrowdGaussian generates photorealistic, geometrically coherent reconstructions of multi-person scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17605v1">ReLaGS: Relational Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 Accepted at CVPR 2026
    </div>
    <details class="paper-abstract">
      Achieving unified 3D perception and reasoning across tasks such as segmentation, retrieval, and relation understanding remains challenging, as existing methods are either object-centric or rely on costly training for inter-object reasoning. We present a novel framework that constructs a hierarchical language-distilled Gaussian scene and its 3D semantic scene graph without scene-specific training. A Gaussian pruning mechanism refines scene geometry, while a robust multi-view language alignment strategy aggregates noisy 2D features into accurate 3D object embeddings. On top of this hierarchy, we build an open-vocabulary 3D scene graph with Vision Language derived annotations and Graph Neural Network-based relational reasoning. Our approach enables efficient and scalable open-vocabulary 3D reasoning by jointly modeling hierarchical semantics and inter/intra-object relationships, validated across tasks including open-vocabulary segmentation, scene graph generation, and relation-guided retrieval. Project page: https://dfki-av.github.io/ReLaGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17519v1">UniSem: Generalizable Semantic 3D Reconstruction from Sparse Unposed Images</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      Semantic-aware 3D reconstruction from sparse, unposed images remains challenging for feed-forward 3D Gaussian Splatting (3DGS). Existing methods often predict an over-complete set of Gaussian primitives under sparse-view supervision, leading to unstable geometry and inferior depth quality. Meanwhile, they rely solely on 2D segmenter features for semantic lifting, which provides weak 3D-level and limited generalizable supervision, resulting in incomplete 3D semantics in novel scenes. To address these issues, we propose UniSem, a unified framework that jointly improves depth accuracy and semantic generalization via two key components. First, Error-aware Gaussian Dropout (EGD) performs error-guided capacity control by suppressing redundancy-prone Gaussians using rendering error cues, producing meaningful, geometrically stable Gaussian representations for improved depth estimation. Second, we introduce a Mix-training Curriculum (MTC) that progressively blends 2D segmenter-lifted semantics with the model's own emergent 3D semantic priors, implemented with object-level prototype alignment to enhance semantic coherence and completeness. Extensive experiments on ScanNet and Replica show that UniSem achieves superior performance in depth prediction and open-vocabulary 3D segmentation across varying numbers of input views. Notably, with 16-view inputs, UniSem reduces depth Rel by 15.2% and improves open-vocabulary segmentation mAcc by 3.7% over strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.26219v3">Parameterizing Dataset Distillation via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 19 pages; Code is available on https://github.com/j-cyoung/GSDatasetDistillation
    </div>
    <details class="paper-abstract">
      Dataset distillation aims to compress training data while preserving training-aware knowledge, alleviating the reliance on large-scale datasets in modern model training. Dataset parameterization provides a more efficient storage structure for dataset distillation, reducing redundancy and accommodating richer information. However, existing methods either rely on complex auxiliary modules or fail to balance representational capacity and efficiency. In this paper, we propose GSDD, a simple, novel, and effective dataset parameterization technique for Dataset Distillation based on Gaussian Splatting. We adapt CUDA-based splatting operators for parallel training in batch, enabling high-quality rendering with minimal computational and memory overhead. Gaussian primitives can effectively capture meaningful training features, allowing a sparse yet expressive representation of individual images. Leveraging both high representational capacity and efficiency, GSDD substantially increases the diversity of distilled datasets under a given storage budget, thereby improving distillation performance. Beyond achieving competitive results on multiple standard benchmarks, GSDD also delivers significant performance gains on large-scale datasets such as ImageNet-1K and on video distillation tasks. In addition, we conduct comprehensive benchmarks to evaluate the computational efficiency, memory footprint, and cross-GPU architectural stability of GSDD. Code is available on https://github.com/j-cyoung/GSDatasetDistillation
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17227v1">Adaptive Anchor Policies for Efficient 4D Gaussian Streaming</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction with Gaussian Splatting has enabled efficient streaming for real-time rendering and free-viewpoint video. However, most pipelines rely on fixed anchor selection such as Farthest Point Sampling (FPS), typically using 8,192 anchors regardless of scene complexity, which over-allocates computation under strict budgets. We propose Efficient Gaussian Streaming (EGS), a plug-in, budget-aware anchor sampler that replaces FPS with a reinforcement-learned policy while keeping the Gaussian streaming reconstruction backbone unchanged. The policy jointly selects an anchor budget and a subset of informative anchors under discrete constraints, balancing reconstruction quality and runtime using spatial features of the Gaussian representation. We evaluate EGS in two settings: fast rendering, which prioritizes runtime efficiency, and high-quality refinement, which enables additional optimization. Experiments on dynamic multi-view datasets show consistent improvements in the quality--efficiency trade-off over FPS sampling. On unseen data, in fast rendering at 256 anchors ($32\times$ fewer than 8,192), EGS improves PSNR by $+0.52$--$0.61$\,dB while running $1.29$--$1.35\times$ faster than IGS@8192 (N3DV and MeetingRoom). In high-quality refinement, EGS remains competitive with the full-anchor baseline at substantially lower anchor budgets. \emph{Code and pretrained checkpoints will be released upon acceptance.} \keywords{4D Gaussian Splatting \and 4D Gaussian Streaming \and Reinforcement Learning}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17131v1">SMAL-pets: SMAL Based Avatars of Pets from Single Image</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Creating high-fidelity, animatable 3D dog avatars remains a formidable challenge in computer vision. Unlike human digital doubles, animal reconstruction faces a critical shortage of large-scale, annotated datasets for specialized applications. Furthermore, the immense morphological diversity across species, breeds, and crosses, which varies significantly in size, proportions, and features, complicates the generalization of existing models. Current reconstruction methods often struggle to capture realistic fur textures. Additionally, ensuring these avatars are fully editable and capable of performing complex, naturalistic movements typically necessitates labor-intensive manual mesh manipulation and expert rigging. This paper introduces SMAL-pets, a comprehensive framework that generates high-quality, editable animal avatars from a single input image. Our approach bridges the gap between reconstruction and generative modeling by leveraging a hybrid architecture. Our method integrates 3D Gaussian Splatting with the SMAL parametric model to provide a representation that is both visually high-fidelity and anatomically grounded. We introduce a multimodal editing suite that enables users to refine the avatar's appearance and execute complex animations through direct textual prompts. By allowing users to control both the aesthetic and behavioral aspects of the model via natural language, SMAL-pets provides a flexible, robust tool for animation and virtual reality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11298v2">InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      High dynamic range (HDR) novel view synthesis (NVS) aims to reconstruct HDR scenes from multi-exposure low dynamic range (LDR) images. Existing HDR pipelines heavily rely on known camera poses, well-initialized dense point clouds, and time-consuming per-scene optimization. Current feed-forward alternatives overlook the HDR problem by assuming exposure-invariant appearance. To bridge this gap, we propose InstantHDR, a feed-forward network that reconstructs 3D HDR scenes from uncalibrated multi-exposure LDR collections in a single forward pass. Specifically, we design a geometry-guided appearance modeling for multi-exposure fusion, and a meta-network for generalizable scene-specific tone mapping. Due to the lack of HDR scene data, we build a pre-training dataset, called HDR-Pretrain, for generalizable feed-forward HDR models, featuring 168 Blender-rendered scenes, diverse lighting types, and multiple camera response functions. Comprehensive experiments show that our InstantHDR delivers comparable synthesis performance to the state-of-the-art optimization-based HDR methods while enjoying $\sim700\times$ and $\sim20\times$ reconstruction speed improvement with our single-forward and post-optimization settings. All code, models, and datasets will be released after the review process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16844v1">M^3: Dense Matching Meets Multi-View Foundation Models for Monocular Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Project page: https://city-super.github.io/M3/
    </div>
    <details class="paper-abstract">
      Streaming reconstruction from uncalibrated monocular video remains challenging, as it requires both high-precision pose estimation and computationally efficient online refinement in dynamic environments. While coupling 3D foundation models with SLAM frameworks is a promising paradigm, a critical bottleneck persists: most multi-view foundation models estimate poses in a feed-forward manner, yielding pixel-level correspondences that lack the requisite precision for rigorous geometric optimization. To address this, we present M^3, which augments the Multi-view foundation model with a dedicated Matching head to facilitate fine-grained dense correspondences and integrates it into a robust Monocular Gaussian Splatting SLAM. M^3 further enhances tracking stability by incorporating dynamic area suppression and cross-inference intrinsic alignment. Extensive experiments on diverse indoor and outdoor benchmarks demonstrate state-of-the-art accuracy in both pose estimation and scene reconstruction. Notably, M^3 reduces ATE RMSE by 64.3% compared to VGGT-SLAM 2.0 and outperforms ARTDECO by 2.11 dB in PSNR on the ScanNet++ dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16538v1">Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 17 pages, 11 figures, CVPR 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as a powerful scene representation and is increasingly used for visual localization and pose refinement. However, despite its high-quality differentiable rendering, the robustness of 3DGS-based pose refinement remains highly sensitive to both the initial camera pose and the reconstructed geometry. In this work, we take a closer look at these limitations and identify two major sources of uncertainty: (i) pose prior uncertainty, which often arises from regression or retrieval models that output a single deterministic estimate, and (ii) geometric uncertainty, caused by imperfections in the 3DGS reconstruction that propagate errors into PnP solvers. Such uncertainties can distort reprojection geometry and destabilize optimization, even when the rendered appearance still looks plausible. To address these uncertainties, we introduce a relocalization framework that combines Monte Carlo pose sampling with Fisher Information-based PnP optimization. Our method explicitly accounts for both pose and geometric uncertainty and requires no retraining or additional supervision. Across diverse indoor and outdoor benchmarks, our approach consistently improves localization accuracy and significantly increases stability under pose and depth noise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.05296v3">Let it Snow! Animating 3D Gaussian Scenes with Dynamic Weather Effects via Physics-Guided Score Distillation</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Accepted to CVPR 2026. Project webpage: https://galfiebelman.github.io/let-it-snow/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has recently enabled fast and photorealistic reconstruction of static 3D scenes. However, dynamic editing of such scenes remains a significant challenge. We introduce a novel framework, Physics-Guided Score Distillation, to address a fundamental conflict: physics simulation provides a strong motion prior that is insufficient for photorealism , while video-based Score Distillation Sampling (SDS) alone cannot generate coherent motion for complex, multi-particle scenarios. We resolve this through a unified optimization framework where physics simulation guides Score Distillation to jointly refine the motion prior for photorealism while simultaneously optimizing appearance. Specifically, we learn a neural dynamics model that predicts particle motion and appearance, optimized end-to-end via a combined loss integrating Video-SDS for photorealism with our physics-guidance prior. This allows for photorealistic refinements while ensuring the dynamics remain plausible. Our framework enables scene-wide dynamic weather effects, including snowfall, rainfall, fog, and sandstorms, with physically plausible motion. Experiments demonstrate our physics-guided approach significantly outperforms baselines, with ablations confirming this joint refinement is essential for generating coherent, high-fidelity dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.13911v4">PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      Despite advances in physics-based 3D motion synthesis, current methods face key limitations: reliance on pre-reconstructed 3D Gaussian Splatting (3DGS) built from dense multi-view images with time-consuming per-scene optimization; physics integration via either inflexible, hand-specified attributes or unstable, optimization-heavy guidance from video models using Score Distillation Sampling (SDS); and naive concatenation of prebuilt 3DGS with physics modules, which ignores physical information embedded in appearance and yields suboptimal performance. To address these issues, we propose PhysGM, a feed-forward framework that jointly predicts 3D Gaussian representation and physical properties from a single image, enabling immediate simulation and high-fidelity 4D rendering. Unlike slow appearance-agnostic optimization methods, we first pre-train a physics-aware reconstruction model that directly infers both Gaussian and physical parameters. We further refine the model with Direct Preference Optimization (DPO), aligning simulations with the physically plausible reference videos and avoiding the high-cost SDS optimization. To address the absence of a supporting dataset for this task, we propose PhysAssets, a dataset of 50K+ 3D assets annotated with physical properties and corresponding reference videos. Experiments show that PhysGM produces high-fidelity 4D simulations from a single image in one minute, achieving a significant speedup over prior work while delivering realistic renderings. Our project page is at:https://hihixiaolv.github.io/PhysGM.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16211v1">Leveling3D: Leveling Up 3D Reconstruction with Feed-Forward 3D Gaussian Splatting and Geometry-Aware Generation</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 26 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Feed-forward 3D reconstruction has revolutionized 3D vision, providing a powerful baseline for downstream tasks such as novel-view synthesis with 3D Gaussian Splatting. Previous works explore fixing the corrupted rendering results with a diffusion model. However, they lack geometric concern and fail at filling the missing area on the extrapolated view. In this work, we introduce Leveling3D, a novel pipeline that integrates feed-forward 3D reconstruction with geometrical-consistent generation to enable holistic simultaneous reconstruction and generation. We propose a geometry-aware leveling adapter, a lightweight technique that aligns internal knowledge in the diffusion model with the geometry prior from the feed-forward model. The leveling adapter enables generation on the artifact area of the extrapolated novel views caused by underconstrained regions of the 3D representation. Specifically, to learn a more diverse distributed generation, we introduce the palette filtering strategy for training, and a test-time masking refinement to prevent messy boundaries along the fixing regions. More importantly, the enhanced extrapolated novel views from Leveling3D could be used as the inputs for feed-forward 3DGS, leveling up the 3D reconstruction. We achieve SOTA performance on public datasets, including tasks such as novel-view synthesis and depth estimation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09881v2">LTGS: Long-Term Gaussian Scene Chronology From Sparse View Updates</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Accepted to CVPR 2026 Findings
    </div>
    <details class="paper-abstract">
      Recent advances in novel-view synthesis can create the photo-realistic visualization of real-world environments from conventional camera captures. However, the everyday environment experiences frequent scene changes, which require dense observations, both spatially and temporally, that an ordinary setup cannot cover. We propose long-term Gaussian scene chronology from sparse-view updates, coined LTGS, an efficient scene representation that can embrace everyday changes from highly under-constrained casual captures. Given an incomplete and unstructured 3D Gaussian Splatting (3DGS) representation obtained from an initial set of input images, we robustly model the long-term chronology of the scene despite abrupt movements and subtle environmental variations. We construct objects as template Gaussians, which serve as structural, reusable priors for shared object tracks. Then, the object templates undergo a further refinement pipeline that modulates the priors to adapt to temporally varying environments given few-shot observations. Once trained, our framework is generalizable across multiple time steps through simple transformations, significantly enhancing the scalability for a temporal evolution of 3D environments. As existing datasets do not explicitly represent the long-term real-world changes with a sparse capture setup, we collect real-world datasets to evaluate the practicality of our pipeline. Experiments demonstrate that our framework achieves superior reconstruction quality compared to other baselines while enabling fast and light-weight updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10267v2">Long-LRM++: Preserving Fine Details in Feed-Forward Wide-Coverage Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Recent advances in generalizable Gaussian splatting (GS) have enabled feed-forward reconstruction of scenes from tens of input views. Long-LRM notably scales this paradigm to 32 input images at $950\times540$ resolution, achieving 360° scene-level reconstruction in a single forward pass. However, directly predicting millions of Gaussian parameters at once remains highly error-sensitive: small inaccuracies in positions or other attributes lead to noticeable blurring, particularly in fine structures such as text. In parallel, implicit representation methods such as LVSM and LaCT have demonstrated significantly higher rendering fidelity by compressing scene information into model weights rather than explicit Gaussians, and decoding RGB frames using the full transformer or TTT backbone. However, this computationally intensive decompression process for every rendered frame makes real-time rendering infeasible. These observations raise key questions: Is the deep, sequential "decompression" process necessary? Can we retain the benefits of implicit representations while enabling real-time performance? We address these questions with Long-LRM++, a model that adopts a semi-explicit scene representation combined with a lightweight decoder. Long-LRM++ matches the rendering quality of LaCT on DL3DV while achieving real-time 14 FPS rendering on an A100 GPU, overcoming the speed limitations of prior implicit methods. Our design also scales to 64 input views at the $950\times540$ resolution, demonstrating strong generalization to increased input lengths. Additionally, Long-LRM++ delivers superior novel-view depth prediction on ScanNetv2 compared to direct depth rendering from Gaussians. Extensive ablation studies validate the effectiveness of each component in the proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16103v1">NanoGS: Training-Free Gaussian Splat Simplification</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      3D Gaussian Splat (3DGS) enables high-fidelity, real-time novel view synthesis by representing scenes with large sets of anisotropic primitives, but often requires millions of Splats, incurring significant storage and transmission costs. Most existing compression methods rely on GPU-intensive post-training optimization with calibrated images, limiting practical deployment. We introduce NanoGS, a training-free and lightweight framework for Gaussian Splat simplification. Instead of relying on image-based rendering supervision, NanoGS formulates simplification as local pairwise merging over a sparse spatial graph. The method approximates a pair of Gaussians with a single primitive using mass preserved moment matching and evaluates merge quality through a principled merge cost between the original mixture and its approximation. By restricting merge candidates to local neighborhoods and selecting compatible pairs efficiently, NanoGS produces compact Gaussian representations while preserving scene structure and appearance. NanoGS operates directly on existing Gaussian Splat models, runs efficiently on CPU, and preserves the standard 3DGS parameterization, enabling seamless integration with existing rendering pipelines. Experiments demonstrate that NanoGS substantially reduces primitive count while maintaining high rendering fidelity, providing an efficient and practical solution for Gaussian Splat simplification. Our project website is available at https://saliteta.github.io/NanoGS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.22070v4">FreeGaussian: Annotation-free Control of Articulated Objects via 3D Gaussian Splats with Flow Derivatives</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Reconstructing controllable Gaussian splats for articulated objects from monocular video is especially challenging due to its inherently insufficient constraints. Existing methods address this by relying on dense masks and manually defined control signals, limiting their real-world applications. In this paper, we propose an annotation-free method, FreeGaussian, which mathematically disentangles camera egomotion and articulated movements via flow derivatives. By establishing a connection between 2D flows and 3D Gaussian dynamic flow, our method enables optimization and continuity of dynamic Gaussian motions from flow priors without any control signals. Furthermore, we introduce a 3D spherical vector controlling scheme, which represents the state as a 3D Gaussian trajectory, thereby eliminating the need for complex 1D control signal calculations and simplifying controllable Gaussian modeling. Extensive experiments on articulated objects demonstrate the state-of-the-art visual performance and precise, part-aware controllability of our method. Code is available at: https://github.com/Tavish9/freegaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15811v1">Feed-forward Gaussian Registration for Head Avatar Creation and Editing</a></div>
    <div class="paper-meta">
      📅 2026-03-16
      | 💬 Website: https://malteprinzler.github.io/projects/match ; Video: https://youtu.be/Z3xoXQ648sE
    </div>
    <details class="paper-abstract">
      We present MATCH (Multi-view Avatars from Topologically Corresponding Heads), a multi-view Gaussian registration method for high-quality head avatar creation and editing. State-of-the-art multi-view head avatar methods require time-consuming head tracking followed by expensive avatar optimization, often resulting in a total creation time of more than one day. MATCH, in contrast, directly predicts Gaussian splat textures in correspondence from calibrated multi-view images in just 0.5 seconds per frame, without requiring data preprocessing. The learned intra-subject correspondence across frames enables fast creation of personalized head avatars, while correspondence across subjects supports applications such as expression transfer, optimization-free tracking, semantic editing, and identity interpolation. We establish these correspondences end-to-end using a transformer-based model that predicts Gaussian splat textures in the fixed UV layout of a template mesh. To achieve this, we introduce a novel registration-guided attention block, where each UV-map token attends exclusively to image tokens depicting its corresponding mesh region. This design improves efficiency and performance compared to dense cross-view attention. MATCH outperforms existing methods in novel-view synthesis, geometry registration, and head avatar generation, while making avatar creation 10 times faster than the closest competing baseline. The code and model weights are available on the project website.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.17531v2">Laplace-Beltrami Operator for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-16
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      With the rising popularity of 3D Gaussian splatting and the expanse of applications from rendering to 3D reconstruction, there comes also a need for geometry processing applications directly on this new representation. While considering the centers of Gaussians as a point cloud or meshing them is an option that allows to apply existing algorithms, this might ignore information present in the data or be unnecessarily expensive. Additionally, Gaussian splatting tends to contain a large number of outliers which do not affect the rendering quality but need to be handled correctly in order not to produce noisy results in geometry processing applications. In this work, we propose a formulation to compute the Laplace-Beltrami operator, a widely used tool in geometry processing, directly on Gaussian splatting using the Mahalanobis distance. While conceptually similar to a point cloud Laplacian, our experiments show superior accuracy on the point clouds encoded in the Gaussian splatting centers and, additionally, the operator can be used to evaluate the quality of the output during optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.13766v2">LHM++: An Efficient Large Human Reconstruction Model for Pose-free Images to 3D</a></div>
    <div class="paper-meta">
      📅 2026-03-16
      | 💬 HomePage: https://lingtengqiu.github.io/LHM++/ Online Demo: https://huggingface.co/spaces/Lingteng/LHMPP
    </div>
    <details class="paper-abstract">
      Reconstructing animatable 3D humans from casually captured images of articulated subjects without camera or pose information is highly practical but remains challenging due to view misalignment, occlusions, and the absence of structural priors. In this work, we present LHM++, an efficient large-scale human reconstruction model that generates high-quality, animatable 3D avatars within seconds from one or multiple pose-free images. At its core is an Encoder-Decoder Point-Image Transformer architecture that progressively encodes and decodes 3D geometric point features to improve efficiency, while fusing hierarchical 3D point features with image features through multimodal attention. The fused features are decoded into 3D Gaussian splats to recover detailed geometry and appearance. To further enhance visual fidelity, we introduce a lightweight 3D-aware neural animation renderer that refines the rendering quality of reconstructed avatars in real time. Extensive experiments show that our method produces high-fidelity, animatable 3D humans without requiring camera or pose annotations. Our code and project page are available at https://lingtengqiu.github.io/LHM++/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15368v1">IRIS: Intersection-aware Ray-based Implicit Editable Scenes</a></div>
    <div class="paper-meta">
      📅 2026-03-16
    </div>
    <details class="paper-abstract">
      Neural Radiance Fields achieve high-fidelity scene representation but suffer from costly training and rendering, while 3D Gaussian splatting offers real-time performance with strong empirical results. Recently, solutions that harness the best of both worlds by using Gaussians as proxies to guide neural field evaluations, still suffer from significant computational inefficiencies. They typically rely on stochastic volumetric sampling to aggregate features, which severely limits rendering performance. To address this issue, a novel framework named IRIS (Intersection-aware Ray-based Implicit Editable Scenes) is introduced as a method designed for efficient and interactive scene editing. To overcome the limitations of standard ray marching, an analytical sampling strategy is employed that precisely identifies interaction points between rays and scene primitives, effectively eliminating empty space processing. Furthermore, to address the computational bottleneck of spatial neighbor lookups, a continuous feature aggregation mechanism is introduced that operates directly along the ray. By interpolating latent attributes from sorted intersections, costly 3D searches are bypassed, ensuring geometric consistency, enabling high-fidelity, real-time rendering, and flexible shape editing. Code can be found at https://github.com/gwilczynski95/iris.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15186v1">NavGSim: High-Fidelity Gaussian Splatting Simulator for Large-Scale Navigation</a></div>
    <div class="paper-meta">
      📅 2026-03-16
    </div>
    <details class="paper-abstract">
      Simulating realistic environments for robots is widely recognized as a critical challenge in robot learning, particularly in terms of rendering and physical simulation. This challenge becomes even more pronounced in navigation tasks, where trajectories often extend across multiple rooms or entire floors. In this work, we present NavGSim, a Gaussian Splatting-based simulator designed to generate high-fidelity, large-scale navigation environments. Built upon a hierarchical 3D Gaussian Splatting framework, NavGSim enables photorealistic rendering in expansive scenes spanning hundreds of square meters. To simulate navigation collisions, we introduce a Gaussian Splatting-based slice technique that directly extracts navigable areas from reconstructed Gaussians. Additionally, for ease of use, we provide comprehensive NavGSim APIs supporting multi-GPU development, including tools for custom scene reconstruction, robot configuration, policy training, and evaluation. To evaluate NavGSim's effectiveness, we train a Vision-Language-Action (VLA) model using trajectories collected from NavGSim and assess its performance in both simulated and real-world environments. Our results demonstrate that NavGSim significantly enhances the VLA model's scene understanding, enabling the policy to handle diverse navigation queries effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14965v1">GeoNVS: Geometry Grounded Video Diffusion for Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-03-16
      | 💬 The code will be available at https://sites.google.com/view/minjun-kang/geonvs-arxiv26
    </div>
    <details class="paper-abstract">
      Novel view synthesis requires strong 3D geometric consistency and the ability to generate visually coherent images across diverse viewpoints. While recent camera-controlled video diffusion models show promising results, they often suffer from geometric distortions and limited camera controllability. To overcome these challenges, we introduce GeoNVS, a geometry-grounded novel-view synthesizer that enhances both geometric fidelity and camera controllability through explicit 3D geometric guidance. Our key innovation is the Gaussian Splat Feature Adapter (GS-Adapter), which lifts input-view diffusion features into 3D Gaussian representations, renders geometry-constrained novel-view features, and adaptively fuses them with diffusion features to correct geometrically inconsistent representations. Unlike prior methods that inject geometry at the input level, GS-Adapter operates in feature space, avoiding view-dependent color noise that degrades structural consistency. Its plug-and-play design enables zero-shot compatibility with diverse feed-forward geometry models without additional training, and can be adapted to other video diffusion backbones. Experiments across 9 scenes and 18 settings demonstrate state-of-the-art performance, achieving 11.3% and 14.9% improvements over SEVA and CameraCtrl, with up to 2x reduction in translation error and 7x in Chamfer Distance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16943v1">KGS-GCN: Enhancing Sparse Skeleton Sensing via Kinematics-Driven Gaussian Splatting and Probabilistic Topology for Action Recognition</a></div>
    <div class="paper-meta">
      📅 2026-03-16
    </div>
    <details class="paper-abstract">
      Skeleton-based action recognition is widely utilized in sensor systems including human-computer interaction and intelligent surveillance. Nevertheless, current sensor devices typically generate sparse skeleton data as discrete coordinates, which inevitably discards fine-grained spatiotemporal details during highly dynamic movements. Moreover, the rigid constraints of predefined physical sensor topologies hinder the modeling of latent long-range dependencies. To overcome these limitations, we propose KGS-GCN, a graph convolutional network that integrates kinematics-driven Gaussian splatting with probabilistic topology. Our framework explicitly addresses the challenges of sensor data sparsity and topological rigidity by transforming discrete joints into continuous generative representations. Firstly, a kinematics-driven Gaussian splatting module is designed to dynamically construct anisotropic covariance matrices using instantaneous joint velocity vectors. This module enhances visual representation by rendering sparse skeleton sequences into multi-view continuous heatmaps rich in spatiotemporal semantics. Secondly, to transcend the limitations of fixed physical connections, a probabilistic topology construction method is proposed. This approach generates an adaptive prior adjacency matrix by quantifying statistical correlations via the Bhattacharyya distance between joint Gaussian distributions. Ultimately, the GCN backbone is adaptively modulated by the rendered visual features via a visual context gating mechanism. Empirical results demonstrate that KGS-GCN significantly enhances the modeling of complex spatiotemporal dynamics. By addressing the inherent limitations of sparse inputs, our framework offers a robust solution for processing low-fidelity sensor data. This approach establishes a practical pathway for improving perceptual reliability in real-world sensing applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.20791v3">MetaGS: A Meta-Learned Gaussian-Phong Model for Out-of-Distribution 3D Scene Relighting</a></div>
    <div class="paper-meta">
      📅 2026-03-16
      | 💬 Accepted by NeurIPS 2025 (Spotlight). Code: https://github.com/raynehe/MetaGS
    </div>
    <details class="paper-abstract">
      Out-of-distribution (OOD) 3D relighting requires novel view synthesis under unseen lighting conditions that differ significantly from the observed images. Existing relighting methods, which assume consistent light source distributions between training and testing, often degrade in OOD scenarios. We introduce MetaGS to tackle this challenge from two perspectives. First, we propose a meta-learning approach to train 3D Gaussian splatting, which explicitly promotes learning generalizable Gaussian geometries and appearance attributes across diverse lighting conditions, even with biased training data. Second, we embed fundamental physical priors from the Blinn-Phong reflection model into Gaussian splatting, which enhances the decoupling of shading components and leads to more accurate 3D scene reconstruction. Results on both synthetic and real-world datasets demonstrate the effectiveness of MetaGS in challenging OOD relighting tasks, supporting efficient point-light relighting and generalizing well to unseen environment lighting maps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14763v1">LiDAR-EVS: Enhance Extrapolated View Synthesis for 3D Gaussian Splatting with Pseudo-LiDAR Supervision</a></div>
    <div class="paper-meta">
      📅 2026-03-16
      | 💬 22 pages, 8 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for real-time LiDAR and camera synthesis in autonomous driving simulation. However, simulating LiDAR with 3DGS remains challenging for extrapolated views beyond the training trajectory, as existing methods are typically trained on single-traversal sensor scans, suffer from severe overfitting and poor generalization to novel ego-vehicle paths. To enable reliable simulation of LiDAR along unseen driving trajectories without external multi-pass data, we present LiDAR-EVS, a lightweight framework for robust extrapolated-view LiDAR simulation in autonomous driving. Designed to be plug-and-play, LiDAR-EVS readily extends to diverse LiDAR sensors and neural rendering baselines with minimal modification. Our framework comprises two key components: (1) pseudo extrapolated-view point cloud supervision with multi-frame LiDAR fusion, view transformation, occlusion curling, and intensity adjustment; (2) spatially-constrained dropout regularization that promotes robustness to diverse trajectory variations encountered in real-world driving. Extensive experiments demonstrate that LiDAR-EVS achieves SOTA performance on extrapolated-view LiDAR synthesis across three datasets, making it a promising tool for data-driven simulation, closed-loop evaluation, and synthetic data generation in autonomous driving systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14684v1">E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-16
      | 💬 10 pages, 6 figures, accepted to CVPR 2026
    </div>
    <details class="paper-abstract">
      The emergence of neural radiance fields (NeRF) and 3D Gaussian splatting (3DGS) has advanced novel view synthesis (NVS). These methods, however, require high-quality RGB inputs and accurate corresponding poses, limiting robustness under real-world conditions such as fast camera motion or adverse lighting. Event cameras, which capture brightness changes at each pixel with high temporal resolution and wide dynamic range, enable precise sensing of dynamic scenes and offer a promising solution. However, existing event-based NVS methods either assume known poses or rely on depth estimation models that are bounded by their initial observations, failing to generalize as the camera traverses previously unseen regions. We present E2EGS, a pose-free framework operating solely on event streams. Our key insight is that edge information provides rich structural cues essential for accurate trajectory estimation and high-quality NVS. To extract edges from noisy event streams, we exploit the distinct spatio-temporal characteristics of edges and non-edge regions. The event camera's movement induces consistent events along edges, while non-edge regions produce sparse noise. We leverage this through a patch-based temporal coherence analysis that measures local variance to extract edges while robustly suppressing noise. The extracted edges guide structure-aware Gaussian initialization and enable edge-weighted losses throughout initialization, tracking, and bundle adjustment. Extensive experiments on both synthetic and real datasets demonstrate that E2EGS achieves superior reconstruction quality and trajectory accuracy, establishing a fully pose-free paradigm for event-based 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.00831v3">UPGS: Unified Pose-aware Gaussian Splatting for Dynamic Scene Deblurring</a></div>
    <div class="paper-meta">
      📅 2026-03-15
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic 3D scenes from monocular video has broad applications in AR/VR, robotics, and autonomous navigation, but often fails due to severe motion blur caused by camera and object motion. Existing methods commonly follow a two-step pipeline, where camera poses are first estimated and then 3D Gaussians are optimized. Since blurring artifacts usually undermine pose estimation, pose errors could be accumulated to produce inferior reconstruction results. To address this issue, we introduce a unified optimization framework by incorporating camera poses as learnable parameters complementary to 3DGS attributes for end-to-end optimization. Specifically, we recast camera and object motion as per-primitive SE(3) affine transformations on 3D Gaussians and formulate a unified optimization objective. For stable optimization, we introduce a three-stage training schedule that optimizes camera poses and Gaussians alternatively. Particularly, 3D Gaussians are first trained with poses being fixed, and then poses are optimized with 3D Gaussians being untouched. Finally, all learnable parameters are optimized together. Extensive experiments on the Stereo Blur dataset and challenging real-world sequences demonstrate that our method achieves significant gains in reconstruction quality and pose estimation accuracy over prior dynamic deblurring methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14316v1">Direct Object-Level Reconstruction via Probabilistic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-15
    </div>
    <details class="paper-abstract">
      Object-level 3D reconstruction play important roles across domains such as cultural heritage digitization, industrial manufacturing, and virtual reality. However, existing Gaussian Splatting-based approaches generally rely on full-scene reconstruction, in which substantial redundant background information is introduced, leading to increased computational and storage overhead. To address this limitation, we propose an efficient single-object 3D reconstruction method based on 2D Gaussian Splatting. By directly integrating foreground-background probability cues into Gaussian primitives and dynamically pruning low-probability Gaussians during training, the proposed method fundamentally focuses on an object of interest and improves the memory and computational efficiency. Our pipeline leverages probability masks generated by YOLO and SAM to supervise probabilistic Gaussian attributes, replacing binary masks with continuous probability values to mitigate boundary ambiguity. Additionally, we propose a dual-stage filtering strategy for training's startup to suppress background Gaussians. And, during training, rendered probability masks are conversely employed to refine supervision and enhance boundary consistency across views. Experiments conducted on the MIP-360, T&T, and NVOS datasets demonstrate that our method exhibits strong self-correction capability in the presence of mask errors and achieves reconstruction quality comparable to standard 3DGS approaches, while requiring only approximately 1/10 of their Gaussian amount. These results validate the efficiency and robustness of our method for single-object reconstruction and highlight its potential for applications requiring both high fidelity and computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14309v1">In-Field 3D Wheat Head Instance Segmentation From TLS Point Clouds Using Deep Learning Without Manual Labels</a></div>
    <div class="paper-meta">
      📅 2026-03-15
      | 💬 to be published in ISPRS Annals of Photogrammetry and Remote Sensing at XXV ISPRS Congress, Toronto, Canada, July 2026, 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      3D instance segmentation for laser scanning (LiDAR) point clouds remains a challenge in many remote sensing-related domains. Successful solutions typically rely on supervised deep learning and manual annotations, and consequently focus on objects that can be well delineated through visual inspection and manual labeling of point clouds. However, for tasks with more complex and cluttered scenes, such as in-field plant phenotyping in agriculture, such approaches are often infeasible. In this study, we tackle the task of in-field wheat head instance segmentation directly from terrestrial laser scanning (TLS) point clouds. To address the problem and circumvent the need for manual annotations, we propose a novel two-stage pipeline. To obtain the initial 3D instance proposals, the first stage uses 3D-to-2D multi-view projections, the Grounded SAM pipeline for zero-shot 2D object-centric segmentation, and multi-view label fusion. The second stage uses these initial proposals as noisy pseudo-labels to train a supervised 3D panoptic-style segmentation neural network. Our results demonstrate the feasibility of the proposed approach and show performance improvementsrelative to Wheat3DGS, a recent alternative solution for in-field wheat head instance segmentation without manual 3D annotations based on multi-view RGB images and 3D Gaussian Splatting, showcasing TLS as a competitive sensing alternative. Moreover, the results show that both stages of the proposed pipeline can deliver usable 3D instance segmentation without manual annotations, indicating promising, low-effort transferability to other comparable TLS-based point cloud segmentation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14301v1">4D Synchronized Fields: Motion-Language Gaussian Splatting for Temporal Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2026-03-15
      | 💬 34 pages, 3 figures, 7 tables. Includes supplementary material. Preprint
    </div>
    <details class="paper-abstract">
      Current 4D representations decouple geometry, motion, and semantics: reconstruction methods discard interpretable motion structure; language-grounded methods attach semantics after motion is learned, blind to how objects move; and motion-aware methods encode dynamics as opaque per-point residuals without object-level organization. We propose 4D Synchronized Fields, a 4D Gaussian representation that learns object-factored motion in-loop during reconstruction and synchronizes language to the resulting kinematics through a per-object conditioned field. Each Gaussian trajectory is decomposed into shared object motion plus an implicit residual, and a kinematic-conditioned ridge map predicts temporal semantic variation, yielding a single representation in which reconstruction, motion, and semantics are structurally coupled and enabling open-vocabulary temporal queries that retrieve both objects and moments. On HyperNeRF, 4D Synchronized Fields achieves 28.52 dB mean PSNR, the highest among all language-grounded and motion-aware baselines, within 1.5 dB of reconstruction-only methods. On targeted temporal-state retrieval, the kinematic-conditioned field attains 0.884 mean accuracy, 0.815 mean vIoU, and 0.733 mean tIoU, surpassing 4D LangSplat (0.620, 0.433, and 0.439 respectively) and LangSplat (0.415, 0.304, and 0.262). Ablation confirms that kinematic conditioning is the primary driver, accounting for +0.45 tIoU over a static-embedding-only baseline. 4D Synchronized Fields is the only method that jointly exposes interpretable motion primitives and temporally grounded language fields from a single trained representation. Code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16119v2">RadarGaussianDet3D: Gaussian Representation-based Real-time 3D Object Detection with 4D Automotive Radars</a></div>
    <div class="paper-meta">
      📅 2026-03-15
      | 💬 Accepted by IEEE Robotics and Automation Letters
    </div>
    <details class="paper-abstract">
      4D automotive radars have gained increasing attention for autonomous driving due to their low cost, robustness, and inherent velocity measurement capability. However, existing 4D radar-based 3D detectors rely heavily on pillar encoders for BEV feature extraction, where each point contributes to only a single BEV grid, resulting in sparse feature maps and degraded representation quality. In addition, they also optimize bounding box attributes independently, leading to sub-optimal detection accuracy. Moreover, their inference speed, while sufficient for high-end GPUs, may fail to meet the real-time requirement on vehicle-mounted embedded devices. To overcome these limitations, an efficient and effective Gaussian-based 3D detector, namely RadarGaussianDet3D is introduced, leveraging Gaussian primitives and distributions as intermediate representations for radar points and bounding boxes. In RadarGaussianDet3D, a novel Point Gaussian Encoder (PGE) is designed to transform each point into a Gaussian primitive after feature aggregation and employs the 3D Gaussian Splatting (3DGS) technique for BEV rasterization, yielding denser feature maps. PGE exhibits exceptionally low latency, owing to the optimized algorithm for point feature aggregation and fast rendering of 3DGS. In addition, a new Box Gaussian Loss (BGL) is proposed, which converts bounding boxes into 3D Gaussian distributions and measures their distance to enable more comprehensive and consistent optimization. Extensive experiments on TJ4DRadSet and View-of-Delft demonstrate that RadarGaussianDet3D achieves high detection accuracy while delivering substantially faster inference, highlighting its potential for real-time deployment in autonomous driving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14232v1">S2GS: Streaming Semantic Gaussian Splatting for Online Scene Understanding and Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-15
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Existing offline feed-forward methods for joint scene understanding and reconstruction on long image streams often repeatedly perform global computation over an ever-growing set of past observations, causing runtime and GPU memory to increase rapidly with sequence length and limiting scalability. We propose Streaming Semantic Gaussian Splatting (S2GS), a strictly causal, incremental 3D Gaussian semantic field framework: it does not leverage future frames and continuously updates scene geometry, appearance, and instance-level semantics without reprocessing historical frames, enabling scalable online joint reconstruction and understanding. S2GS adopts a geometry-semantic decoupled dual-backbone design: the geometry branch performs causal modeling to drive incremental Gaussian updates, while the semantic branch leverages a 2D foundation vision model and a query-driven decoder to predict segmentation masks and identity embeddings, further stabilized by query-level contrastive alignment and lightweight online association with an instance memory. Experiments show that S2GS matches or outperforms strong offline baselines on joint reconstruction-and-understanding benchmarks, while significantly improving long-horizon scalability: it processes 1,000+ frames with much slower growth in runtime and GPU memory, whereas offline global-processing baselines typically run out of memory at around 80 frames under the same setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.15208v4">GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-15
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Transferring 2D textures onto complex 3D scenes plays a vital role in enhancing the efficiency and controllability of 3D multimedia content creation. However, existing 3D style transfer methods primarily focus on transferring abstract artistic styles to 3D scenes. These methods often overlook the geometric information of the scene, which makes it challenging to achieve high-quality 3D texture transfer results. In this paper, we present GT2-GS, a geometry-aware texture transfer framework for gaussian splatting. First, we propose a geometry-aware texture transfer loss that enables view-consistent texture transfer by leveraging prior view-dependent feature information and texture features augmented with additional geometric parameters. Moreover, an adaptive fine-grained control module is proposed to address the degradation of scene information caused by low-granularity texture features. Finally, a geometry preservation branch is introduced. This branch refines the geometric parameters using additionally bound Gaussian color priors, thereby decoupling the optimization objectives of appearance and geometry. Extensive experiments demonstrate the effectiveness and controllability of our method. Through geometric awareness, our approach achieves texture transfer results that better align with human visual perception. Our homepage is available at https://vpx-ecnu.github.io/GT2-GS-website.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.22218v2">ABC-GS: Alignment-Based Controllable Style Transfer for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-15
      | 💬 ICME 2025 Oral
    </div>
    <details class="paper-abstract">
      3D scene stylization approaches based on Neural Radiance Fields (NeRF) achieve promising results by optimizing with Nearest Neighbor Feature Matching (NNFM) loss. However, NNFM loss does not consider global style information. In addition, the implicit representation of NeRF limits their fine-grained control over the resulting scenes. In this paper, we introduce ABC-GS, a novel framework based on 3D Gaussian Splatting to achieve high-quality 3D style transfer. To this end, a controllable matching stage is designed to achieve precise alignment between scene content and style features through segmentation masks. Moreover, a style transfer loss function based on feature alignment is proposed to ensure that the outcomes of style transfer accurately reflect the global style of the reference image. Furthermore, the original geometric information of the scene is preserved with the depth loss and Gaussian regularization terms. Extensive experiments show that our ABC-GS provides controllability of style transfer and achieves stylization results that are more faithfully aligned with the global style of the chosen artistic reference. Our homepage is available at https://vpx-ecnu.github.io/ABC-GS-website.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12898v2">Towards High-Fidelity Gaussian Splatting with Queried-Convolution Neural Networks</a></div>
    <div class="paper-meta">
      📅 2026-03-15
      | 💬 38 pages, 8 figures, Project Page: https://abhi1kumar.github.io/qonvolution/
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has revolutionized the field of Novel View Synthesis (NVS) with faster training and real-time rendering. However, its reconstruction fidelity still trails behind the powerful radiance models such as Zip-NeRF. Motivated by our theoretical result that both queries (such as coordinates) and neighborhood are important to learn high-fidelity signals, this paper proposes Queried-Convolutions (Qonvolutions), a simple yet powerful modification using the neighborhood properties of convolution. Qonvolutions convolve a low-fidelity signal with queries to output residual and achieve high-fidelity reconstruction. We empirically demonstrate that combining Gaussian splatting with Qonvolution neural networks (QNNs) results in state-of-the-art NVS on real-world scenes, even outperforming Zip-NeRF on image fidelity. QNNs also enhance performance of 1D regression, 2D regression and 2D super-resolution tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14001v1">PhyGaP: Physically-Grounded Gaussians with Polarization Cues</a></div>
    <div class="paper-meta">
      📅 2026-03-14
      | 💬 The paper is accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated great success in modeling reflective 3D objects and their interaction with the environment via deferred rendering (DR). However, existing methods often struggle with correctly reconstructing physical attributes such as albedo and reflectance, and therefore they do not support high-fidelity relighting. Observing that this limitation stems from the lack of shape and material information in RGB images, we present PhyGaP, a physically-grounded 3DGS method that leverages polarization cues to facilitate precise reflection decomposition and visually consistent relighting of reconstructed objects. Specifically, we design a polarimetric deferred rendering (PolarDR) process to model polarization by reflection, and a self-occlusion-aware environment map building technique (GridMap) to resolve indirect lighting of non-convex objects. We validate on multiple synthetic and real-world scenes, including those featuring only partial polarization cues, that PhyGaP not only excels in reconstructing the appearance and surface normal of reflective 3D objects (~2 dB in PSNR and 45.7% in Cosine Distance better than existing RGB-based methods on average), but also achieves state-of-the-art inverse rendering and relighting capability. Our code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.13910v1">Scene Generation at Absolute Scale: Utilizing Semantic and Geometric Guidance From Text for Accurate and Interpretable 3D Indoor Scene Generation</a></div>
    <div class="paper-meta">
      📅 2026-03-14
    </div>
    <details class="paper-abstract">
      We present GuidedSceneGen, a text-to-3D generation framework that produces metrically accurate, globally consistent, and semantically interpretable indoor scenes. Unlike prior text-driven methods that often suffer from geometric drift or scale ambiguity, our approach maintains an absolute world coordinate frame throughout the entire generation process. Starting from a textual scene description, we predict a global 3D layout encoding both semantic and geometric structure, which serves as a guiding proxy for downstream stages. A semantics- and depth-conditioned panoramic diffusion model then synthesizes 360° imagery aligned with the global layout, substantially improving spatial coherence. To explore unobserved regions, we employ a video diffusion model guided by optimized camera trajectories that balances coverage and collision avoidance, achieving up to 10x faster sampling compared to exhaustive path exploration. The generated views are fused using 3D Gaussian Splatting, yielding a consistent and fully navigable 3D scene in absolute scale. GuidedSceneGen enables accurate transfer of object poses and semantic labels from layout to reconstruction, and supports progressive scene expansion without re-alignment. Quantitative results and a user study demonstrate greater 3D consistency and layout plausibility compared to recent panoramic text-to-3D baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.13783v1">RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-14
      | 💬 Accepted to CVPR2026
    </div>
    <details class="paper-abstract">
      Temporal retiming, the ability to reconstruct and render dynamic scenes at arbitrary timestamps, is crucial for applications such as slow-motion playback, temporal editing, and post-production. However, most existing 4D Gaussian Splatting (4DGS) methods overfit at discrete frame indices but struggle to represent continuous-time frames, leading to ghosting artifacts when interpolating between timestamps. We identify this limitation as a form of temporal aliasing and propose RetimeGS, a simple yet effective 4DGS representation that explicitly defines the temporal behavior of the 3D Gaussian and mitigates temporal aliasing. To achieve smooth and consistent interpolation, we incorporate optical flow-guided initialization and supervision, triple-rendering supervision, and other targeted strategies. Together, these components enable ghost-free, temporally coherent rendering even under large motions. Experiments on datasets featuring fast motion, non-rigid deformation, and severe occlusions demonstrate that RetimeGS achieves superior quality and coherence over state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07664v2">Ref-DGS: Reflective Dual Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-13
      | 💬 Project page: https://straybirdflower.github.io/Ref-DGS/
    </div>
    <details class="paper-abstract">
      Reflective appearance, especially strong and typically near-field specular reflections, poses a fundamental challenge for accurate surface reconstruction and novel view synthesis. Existing Gaussian splatting methods either fail to model near-field specular reflections or rely on explicit ray tracing at substantial computational cost. We present Ref-DGS, a reflective dual Gaussian splatting framework that addresses this trade-off by decoupling surface reconstruction from specular reflection within an efficient rasterization-based pipeline. Ref-DGS introduces a dual Gaussian scene representation consisting of geometry Gaussians and complementary local reflection Gaussians that capture near-field specular interactions without explicit ray tracing, along with a global environment reflection field for modeling far-field specular reflections. To predict specular radiance, we further propose a lightweight, physically-aware adaptive mixing shader that fuses global and local reflection features. Experiments demonstrate that Ref-DGS achieves state-of-the-art performance on reflective scenes while training substantially faster than ray-based Gaussian methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.16253v2">ExCellGen: Fast, Controllable, Photorealistic 3D Scene Generation from a Single Real-World Exemplar</a></div>
    <div class="paper-meta">
      📅 2026-03-13
    </div>
    <details class="paper-abstract">
      Photorealistic 3D scene generation is challenging due to the scarcity of large-scale, high-quality real-world 3D datasets and complex workflows requiring specialized expertise for manual modeling. These constraints often result in slow iteration cycles, where each modification demands substantial effort, ultimately stifling creativity. We propose a fast, exemplar-driven framework for generating 3D scenes from a single casual input, such as handheld video or drone footage. Our method first leverages 3D Gaussian Splatting (3DGS) to robustly reconstruct input scenes with a high-quality 3D appearance model. We then train a per-scene Generative Cellular Automaton (GCA) to produce a sparse volume of featurized voxels, effectively amortizing scene generation while enabling controllability. A subsequent patch-based remapping step composites the complete scene from the exemplar's initial 3D Gaussian splats, successfully recovering the appearance statistics of the input scene. The entire pipeline can be trained in less than 10 minutes for each exemplar and generates scenes in 0.5-2 seconds. Our method enables interactive creation with full user control, and we showcase complex 3D generation results from real-world exemplars within a self-contained interactive GUI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11638v3">Variation-aware Flexible 3D Gaussian Editing</a></div>
    <div class="paper-meta">
      📅 2026-03-13
    </div>
    <details class="paper-abstract">
      Indirect editing methods for 3D Gaussian Splatting (3DGS) have recently witnessed significant advancements. These approaches operate by first applying edits in the rendered 2D space and subsequently projecting the modifications back into 3D. However, this paradigm inevitably introduces cross-view inconsistencies and constrains both the flexibility and efficiency of the editing process. To address these challenges, we present VF-Editor, which enables native editing of Gaussian primitives by predicting attribute variations in a feedforward manner. To accurately and efficiently estimate these variations, we design a novel variation predictor distilled from 2D editing knowledge. The predictor encodes the input to generate a variation field and employs two learnable, parallel decoding functions to iteratively infer attribute changes for each 3D Gaussian. Thanks to its unified design, VF-Editor can seamlessly distill editing knowledge from diverse 2D editors and strategies into a single predictor, allowing for flexible and effective knowledge transfer into the 3D domain. Extensive experiments on both public and private datasets reveal the inherent limitations of indirect editing pipelines and validate the effectiveness and flexibility of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07540v3">Enhancing Novel View Synthesis via Geometry Grounded Set Diffusion</a></div>
    <div class="paper-meta">
      📅 2026-03-13
      | 💬 Paper and supplementary materials
    </div>
    <details class="paper-abstract">
      We present SetDiff, a geometry-grounded multi-view diffusion framework that enhances novel-view renderings produced by 3D Gaussian Splatting. Our method integrates explicit 3D priors, pixel-aligned coordinate maps and pose-aware Plucker ray embeddings, into a set-based diffusion model capable of jointly processing variable numbers of reference and target views. This formulation enables robust occlusion handling, reduces hallucinations under low-signal conditions, and improves photometric fidelity in visual content restoration. A unified set mixer performs global token-level attention across all input views, supporting scalable multi-camera enhancement while maintaining computational efficiency through latent-space supervision and selective decoding. Extensive experiments on EUVS, Para-Lane, nuScenes, and DL3DV demonstrate significant gains in perceptual fidelity, structural similarity, and robustness under severe extrapolation. SetDiff establishes a state-of-the-art diffusion-based solution for realistic and reliable novel-view synthesis in autonomous driving scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12823v3">TreeDGS: Aerial Gaussian Splatting for Distant DBH Measurement</a></div>
    <div class="paper-meta">
      📅 2026-03-13
    </div>
    <details class="paper-abstract">
      Aerial remote sensing efficiently surveys large areas, but accurate direct object-level measurement remains difficult in complex natural scenes. Advancements in 3D computer vision, particularly radiance field representations such as NeRF and 3D Gaussian splatting, can improve reconstruction fidelity from posed imagery. Nevertheless, direct aerial measurement of important attributes like tree diameter at breast height (DBH) remains challenging. Trunks in aerial forest scans are distant and sparsely observed in image views; at typical operating altitudes, stems may span only a few pixels. With these constraints, conventional reconstruction methods have inaccurate breast-height trunk geometry. TreeDGS is an aerial image reconstruction method that uses 3D Gaussian splatting as a continuous scene representation for trunk measurement. After SfM--MVS initialization and Gaussian optimization, we extract a dense point set from the Gaussian field using RaDe-GS's depth-aware cumulative-opacity integration and associate each sample with a multi-view opacity reliability score. Then, we isolate trunk points and estimate DBH using opacity-weighted solid-circle fitting. Evaluated on 10 plots with field-measured DBH, TreeDGS reaches 4.79 cm RMSE (about 2.6 pixels at this GSD) and outperforms a LiDAR baseline (7.66 cm RMSE). This shows that TreeDGS can enable accurate, low-cost aerial DBH measurement .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12796v1">Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-13
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) deliver high-quality rendering, yet the Gaussian representation exposes a new attack surface, the resource-targeting attack. This attack poisons training images, excessively inducing Gaussian growth to cause resource exhaustion. Although efficiency-oriented methods such as smoothing, thresholding, and pruning have been explored, these spatial-domain strategies operate on visible structures but overlook how stealthy perturbations distort the underlying spectral behaviors of training data. As a result, poisoned inputs introduce abnormal high-frequency amplifications that mislead 3DGS into interpreting noisy patterns as detailed structures, ultimately causing unstable Gaussian overgrowth and degraded scene fidelity. To address this, we propose \textbf{Spectral Defense} in Gaussian and image fields. We first design a 3D frequency filter to selectively prune Gaussians exhibiting abnormally high frequencies. Since natural scenes also contain legitimate high-frequency structures, directly suppressing high frequencies is insufficient, and we further develop a 2D spectral regularization on renderings, distinguishing naturally isotropic frequencies while penalizing anisotropic angular energy to constrain noisy patterns. Experiments show that our defense builds robust, accurate, and secure 3DGS, suppressing overgrowth by up to $5.92\times$, reducing memory by up to $3.66\times$, and improving speed by up to $4.34\times$ under attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12647v1">LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-13
      | 💬 8 pages, 7 figures, conference
    </div>
    <details class="paper-abstract">
      Recent 3D Gaussian Splatting (3DGS) methods have demonstrated the feasibility of self-driving scene reconstruction and novel view synthesis. However, most existing methods either rely solely on cameras or use LiDAR only for Gaussian initialization or depth supervision, while the rich scene information contained in point clouds, such as reflectance, and the complementarity between LiDAR and RGB have not been fully exploited, leading to degradation in challenging self-driving scenes, such as those with high ego-motion and complex lighting. To address these issues, we propose a robust and efficient LiDAR-reflectance-guided Salient Gaussian Splatting method (LR-SGS) for self-driving scenes, which introduces a structure-aware Salient Gaussian representation, initialized from geometric and reflectance feature points extracted from LiDAR and refined through a salient transform and improved density control to capture edge and planar structures. Furthermore, we calibrate LiDAR intensity into reflectance and attach it to each Gaussian as a lighting-invariant material channel, jointly aligned with RGB to enforce boundary consistency. Extensive experiments on the Waymo Open Dataset demonstrate that LR-SGS achieves superior reconstruction performance with fewer Gaussians and shorter training time. In particular, on Complex Lighting scenes, our method surpasses OmniRe by 1.18 dB PSNR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.01647v2">3DGS-DET: Empower 3D Gaussian Splatting with Boundary Guidance and Box-Focused Sampling for Indoor 3D Object Detection</a></div>
    <div class="paper-meta">
      📅 2026-03-13
      | 💬 The code and models will be made publicly available upon acceptance at: \href{https://github.com/yangcaoai/3DGS-DET}{https://github.com/yangcaoai/3DGS-DET}
    </div>
    <details class="paper-abstract">
      Neural Radiance Fields (NeRF) have been adapted for indoor 3D Object Detection (3DOD), offering a promising approach to indoor 3DOD via view-synthesis representation. But its implicit nature limits representational capacity. Recently, 3D Gaussian Splatting (3DGS) has emerged as an explicit 3D representation that addresses the limitation. This work introduces 3DGS into indoor 3DOD for the first time, identifying two main challenges: (i) Ambiguous spatial distribution of Gaussian blobs -- 3DGS primarily relies on 2D pixel-level supervision, resulting in unclear 3D spatial distribution of Gaussian blobs and poor differentiation between objects and background, which hinders indoor 3DOD; (ii) Excessive background blobs -- 2D images typically include numerous background pixels, leading to densely reconstructed 3DGS with many noisy Gaussian blobs representing the background, negatively affecting detection. To tackle (i), we leverage the fact that 3DGS reconstruction is derived from 2D images, and propose an elegant solution by incorporating 2D Boundary Guidance to significantly enhance the spatial distribution of Gaussian blobs, resulting in clearer differentiation between objects and their background (please see fig:teaser). To address (ii), we propose a Box-Focused Sampling strategy using 2D boxes to generate object probability distribution in 3D space, allowing effective probabilistic sampling in 3D to retain more object blobs and reduce noisy background blobs. Benefiting from these innovations, 3DGS-DET significantly outperforms the state-of-the-art NeRF-based method, NeRF-Det++, achieving improvements of +6.0 on mAP@0.25 and +7.8 on mAP@0.5 for the ScanNet, and the +14.9 on mAP@0.25 for the ARKITScenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06684v2">EMGauss: Continuous Slice-to-3D Reconstruction via Dynamic Gaussian Modeling in Volume Electron Microscopy</a></div>
    <div class="paper-meta">
      📅 2026-03-13
      | 💬 Accepted by CVPR 2026. Project page: https://raynehe.github.io/EMGauss/
    </div>
    <details class="paper-abstract">
      Volume electron microscopy (vEM) enables nanoscale 3D imaging of biological structures but remains constrained by acquisition trade-offs, leading to anisotropic volumes with limited axial resolution. Existing deep learning methods seek to restore isotropy by leveraging lateral priors, yet their assumptions break down for morphologically anisotropic structures. We present EMGauss, a general framework for 3D reconstruction from planar scanned 2D slices with applications in vEM, which circumvents the inherent limitations of isotropy-based approaches. Our key innovation is to reframe slice-to-3D reconstruction as a 3D dynamic scene rendering problem based on Gaussian splatting, where the progression of axial slices is modeled as the temporal evolution of 2D Gaussian point clouds. To enhance fidelity in data-sparse regimes, we incorporate a Teacher-Student bootstrapping mechanism that uses high-confidence predictions on unobserved slices as pseudo-supervisory signals. Compared with diffusion- and GAN-based reconstruction methods, EMGauss substantially improves interpolation quality, enables continuous slice synthesis, and eliminates the need for large-scale pretraining. Beyond vEM, it potentially provides a generalizable slice-to-3D solution across diverse imaging domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02293v2">VIGS-SLAM: Visual Inertial Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2026-03-13
      | 💬 Project page: https://vigs-slam.github.io
    </div>
    <details class="paper-abstract">
      We present VIGS-SLAM, a visual-inertial 3D Gaussian Splatting SLAM system that achieves robust real-time tracking and high-fidelity reconstruction. Although recent 3DGS-based SLAM methods achieve dense and photorealistic mapping, their purely visual design degrades under challenging conditions such as motion blur, low texture, and exposure variations. Our method tightly couples visual and inertial cues within a unified optimization framework, jointly optimizing camera poses, depths, and IMU states. It features robust IMU initialization, time-varying bias modeling, and loop closure with consistent Gaussian updates. Experiments on five challenging datasets demonstrate our superiority over state-of-the-art methods. Project page: https://vigs-slam.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08575v3">ReSplat: Learning Recurrent Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Project page: https://haofeixu.github.io/resplat/ Code: https://github.com/cvg/resplat
    </div>
    <details class="paper-abstract">
      While existing feed-forward Gaussian splatting models offer computational efficiency and can generalize to sparse view settings, their performance is fundamentally constrained by relying on a single forward pass for inference. We propose ReSplat, a feed-forward recurrent Gaussian splatting model that iteratively refines 3D Gaussians without explicitly computing gradients. Our key insight is that the Gaussian splatting rendering error serves as a rich feedback signal, guiding the recurrent network to learn effective Gaussian updates. This feedback signal naturally adapts to unseen data distributions at test time, enabling robust generalization across datasets, view counts, and image resolutions. To initialize the recurrent process, we introduce a compact reconstruction model that operates in a $16 \times$ subsampled space, producing $16 \times$ fewer Gaussians than previous per-pixel Gaussian models. This substantially reduces computational overhead and allows for efficient Gaussian updates. Extensive experiments across varying number of input views (2, 8, 16, 32), resolutions ($256 \times 256$ to $540 \times 960$), and datasets (DL3DV, RealEstate10K, and ACID) demonstrate that our method achieves state-of-the-art performance while significantly reducing the number of Gaussians and improving the rendering speed. Our project page is at https://haofeixu.github.io/resplat/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.19297v2">VolSplat: Rethinking Feed-Forward 3D Gaussian Splatting with Voxel-Aligned Prediction</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Project Page: https://lhmd.top/volsplat, Code: https://github.com/ziplab/VolSplat
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) has emerged as a highly effective solution for novel view synthesis. Existing methods predominantly rely on a \emph{pixel-aligned} Gaussian prediction paradigm, where each 2D pixel is mapped to a 3D Gaussian. We rethink this widely adopted formulation and identify several inherent limitations: it renders the reconstructed 3D models heavily dependent on the number of input views, leads to view-biased density distributions, and introduces alignment errors, particularly when source views contain occlusions or low texture. To address these challenges, we introduce VolSplat, a new multi-view feed-forward paradigm that replaces pixel alignment with voxel-aligned Gaussians. By directly predicting Gaussians from a predicted 3D voxel grid, it overcomes pixel alignment's reliance on error-prone 2D feature matching, ensuring robust multi-view consistency. Furthermore, it enables adaptive control over density based on 3D scene complexity, yielding more faithful Gaussians, improved geometric consistency, and enhanced novel-view rendering quality. Experiments on widely used benchmarks demonstrate that VolSplat achieves state-of-the-art performance, while producing more plausible and view-consistent results. The video results, code and trained models are available on our project page: https://lhmd.top/volsplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11969v1">AstroSplat: Physics-Based Gaussian Splatting for Rendering and Reconstruction of Small Celestial Bodies</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 10 pages, 6 figures, conference
    </div>
    <details class="paper-abstract">
      Image-based surface reconstruction and characterization are crucial for missions to small celestial bodies (e.g., asteroids), as it informs mission planning, navigation, and scientific analysis. Recent advances in Gaussian splatting enable high-fidelity neural scene representations but typically rely on a spherical harmonic intensity parameterization that is strictly appearance-based and does not explicitly model material properties or light-surface interactions. We introduce AstroSplat, a physics-based Gaussian splatting framework that integrates planetary reflectance models to improve the autonomous reconstruction and photometric characterization of small-body surfaces from in-situ imagery. The proposed framework is validated on real imagery taken by NASA's Dawn mission, where we demonstrate superior rendering performance and surface reconstruction accuracy compared to the typical spherical harmonic parameterization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.13639v5">4D Radar-Inertial Odometry based on Gaussian Modeling and Multi-Hypothesis Scan Matching</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Our code and results can be publicly accessed at: https://github.com/robotics-upo/gaussian-rio-cpp Accepted for publication in IEEE Robotics and Automation Letters
    </div>
    <details class="paper-abstract">
      4D millimeter-wave (mmWave) radars are sensors that provide robustness against adverse weather conditions (rain, snow, fog, etc.), and as such they are increasingly used for odometry and SLAM (Simultaneous Location and Mapping). However, the noisy and sparse nature of the returned scan data proves to be a challenging obstacle for existing registration algorithms, especially those originally intended for more accurate sensors such as LiDAR. Following the success of 3D Gaussian Splatting for vision, in this paper we propose a summarized representation for radar scenes based on global simultaneous optimization of 3D Gaussians as opposed to voxel-based approaches, and leveraging its inherent Probability Density Function (PDF) for registration. Moreover, we propose optimizing multiple registration hypotheses for better protection against local optima of the PDF. We evaluate our modeling and registration system against state of the art techniques, finding that our system provides richer models and more accurate registration results. Finally, we evaluate the effectiveness of our system in a real Radar-Inertial Odometry task. Experiments using publicly available 4D radar datasets show that our Gaussian approach is comparable to existing registration algorithms, outperforming them in several sequences. Copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09632v2">X-GS: An Extensible Open Framework for Perceiving and Thinking via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, subsequently extending into numerous spatial AI applications. However, most existing 3DGS methods operate in isolation, focusing on specific domains such as pose-free 3DGS, online SLAM, and semantic enrichment. In this paper, we introduce X-GS, an extensible open framework consisting of two major components: the X-GS-Perceiver, which unifies a broad range of 3DGS techniques to enable real-time online SLAM and distill semantic features; and the X-GS-Thinker, which interfaces with downstream multimodal models. In our implementation of the Perceiver, we integrate various 3DGS methods through three novel mechanisms: an online Vector Quantization (VQ) module, a GPU-accelerated grid-sampling scheme, and a highly parallelized pipeline design. The Thinker accommodates vision-language models and utilizes the resulting 3D semantic Gaussians, enabling downstream applications such as object detection, caption generation, and potentially embodied tasks. Experimental results on real-world datasets demonstrate the efficiency and newly unlocked multimodal capabilities of the X-GS framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24053v3">3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Published at ICLR 2026. Code is available at: https://github.com/boschresearch/3dgeer
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves an appealing balance between rendering quality and efficiency, but relies on approximating 3D Gaussians as 2D projections--an assumption that degrades accuracy, especially under generic large field-of-view (FoV) cameras. Despite recent extensions, no prior work has simultaneously achieved both projective exactness and real-time efficiency for general cameras. We introduce 3DGEER, a geometrically exact and efficient Gaussian rendering framework. From first principles, we derive a closed-form expression for integrating Gaussian density along a ray, enabling precise forward rendering and differentiable optimization under arbitrary camera models. To retain efficiency, we propose the Particle Bounding Frustum (PBF), which provides tight ray-Gaussian association without BVH traversal, and the Bipolar Equiangular Projection (BEAP), which unifies FoV representations, accelerates association, and improves reconstruction quality. Experiments on both pinhole and fisheye datasets show that 3DGEER outperforms prior methods across all metrics, runs 5x faster than existing projective exact ray-based baselines, and generalizes to wider FoVs unseen during training--establishing a new state of the art in real-time radiance field rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11543v1">Mango-GS: Enhancing Spatio-Temporal Consistency in Dynamic Scenes Reconstruction using Multi-Frame Node-Guided 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic 3D scenes with photorealistic detail and strong temporal coherence remains a significant challenge. Existing Gaussian splatting approaches for dynamic scene modeling often rely on per-frame optimization, which can overfit to instantaneous states instead of capturing underlying motion dynamics. To address this, we present Mango-GS, a multi-frame, node-guided framework for high-fidelity 4D reconstruction. Mango-GS leverages a temporal Transformer to model motion dependencies within a short window of frames, producing temporally consistent deformations. For efficiency, temporal modeling is confined to a sparse set of control nodes. Each node is represented by a decoupled canonical position and a latent code, providing a stable semantic anchor for motion propagation and preventing correspondence drift under large motion. Our framework is trained end-to-end, enhanced by an input masking strategy and two multi-frame losses to improve robustness. Extensive experiments demonstrate that Mango-GS achieves state-of-the-art reconstruction quality and real-time rendering speed, enabling high-fidelity reconstruction and interactive rendering of dynamic scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11531v1">Mobile-GS: Real-time Gaussian Splatting for Mobile Devices</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Project Page: https://xiaobiaodu.github.io/mobile-gs-project/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful representation for high-quality rendering across a wide range of applications.However, its high computational demands and large storage costs pose significant challenges for deployment on mobile devices. In this work, we propose a mobile-tailored real-time Gaussian Splatting method, dubbed Mobile-GS, enabling efficient inference of Gaussian Splatting on edge devices. Specifically, we first identify alpha blending as the primary computational bottleneck, since it relies on the time-consuming Gaussian depth sorting process. To solve this issue, we propose a depth-aware order-independent rendering scheme that eliminates the need for sorting, thereby substantially accelerating rendering. Although this order-independent rendering improves rendering speed, it may introduce transparency artifacts in regions with overlapping geometry due to the scarcity of rendering order. To address this problem, we propose a neural view-dependent enhancement strategy, enabling more accurate modeling of view-dependent effects conditioned on viewing direction, 3D Gaussian geometry, and appearance attributes. In this way, Mobile-GS can achieve both high-quality and real-time rendering. Furthermore, to facilitate deployment on memory-constrained mobile platforms, we also introduce first-order spherical harmonics distillation, a neural vector quantization technique, and a contribution-based pruning strategy to reduce the number of Gaussian primitives and compress the 3D Gaussian representation with the assistance of neural networks. Extensive experiments demonstrate that our proposed Mobile-GS achieves real-time rendering and compact model size while preserving high visual quality, making it well-suited for mobile applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10446v2">SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Generating natural and linguistically accurate sign language avatars remains a formidable challenge. Current Sign Language Production (SLP) frameworks face a stark trade-off: direct text-to-pose models suffer from regression-to-the-mean effects, while dictionary-retrieval methods produce robotic, disjointed transitions. To resolve this, we propose a novel training paradigm that leverages sparse keyframes to capture the true underlying kinematic distribution of human signing. By predicting dense motion from these discrete anchors, our approach mitigates regression-to-the-mean while ensuring fluid articulation. To realize this paradigm at scale, we first introduce FAST, an ultra-efficient sign segmentation model that automatically mines precise temporal boundaries. We then present SignSparK, a large-scale Conditional Flow Matching (CFM) framework that utilizes these extracted anchors to synthesize 3D signing sequences in SMPL-X and MANO spaces. This keyframe-driven formulation also uniquely unlocks Keyframe-to-Pose (KF2P) generation, making precise spatiotemporal editing of signing sequences possible. Furthermore, our adopted reconstruction-based CFM objective also enables high-fidelity synthesis in fewer than ten sampling steps; this allows SignSparK to scale across four distinct sign languages, establishing the largest multilingual SLP framework to date. Finally, by integrating 3D Gaussian Splatting for photorealistic rendering, we demonstrate through extensive evaluation that SignSparK establishes a new state-of-the-art across diverse SLP tasks and multilingual benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11298v1">InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      High dynamic range (HDR) novel view synthesis (NVS) aims to reconstruct HDR scenes from multi-exposure low dynamic range (LDR) images. Existing HDR pipelines heavily rely on known camera poses, well-initialized dense point clouds, and time-consuming per-scene optimization. Current feed-forward alternatives overlook the HDR problem by assuming exposure-invariant appearance. To bridge this gap, we propose InstantHDR, a feed-forward network that reconstructs 3D HDR scenes from uncalibrated multi-exposure LDR collections in a single forward pass. Specifically, we design a geometry-guided appearance modeling for multi-exposure fusion, and a meta-network for generalizable scene-specific tone mapping. Due to the lack of HDR scene data, we build a pre-training dataset, called HDR-Pretrain, for generalizable feed-forward HDR models, featuring 168 Blender-rendered scenes, diverse lighting types, and multiple camera response functions. Comprehensive experiments show that our InstantHDR delivers comparable synthesis performance to the state-of-the-art optimization-based HDR methods while enjoying $\sim700\times$ and $\sim20\times$ reconstruction speed improvement with our single-forward and post-optimization settings. All code, models, and datasets will be released after the review process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08958v3">Grow with the Flow: 4D Reconstruction of Growing Plants with Gaussian Flow Fields</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Project page: https://weihanluo.ca/growflow/
    </div>
    <details class="paper-abstract">
      Modeling the time-varying 3D appearance of plants during growth poses unique challenges: unlike most dynamic scenes, plants continuously generate new geometry as they expand, branch, and differentiate. Existing dynamic scene representations are ill-suited to this setting: deformation fields provide insufficient constraints to yield physically plausible scene dynamics, and 4D Gaussian splatting represents the same physical structures with different Gaussian primitives at different times, breaking temporal consistency. We introduce GrowFlow, a dynamic representation that couples 3D Gaussian primitives with a neural ordinary differential equation to model plant growth as a continuous flow field over geometric parameters (position, scale, and orientation). Our representation enables consistent appearance rendering and models nonlinear, continuous-time growth dynamics with full temporal correspondences for every primitive. To initialize a sufficient set of Gaussian primitives, we first reconstruct the mature plant and then learn a reverse-growth process, effectively simulating the plant's developmental history in reverse. GrowFlow achieves superior image quality and geometric coherence compared to prior methods on a new, multi-view timelapse dataset of plant growth, and provides the first temporally coherent representation for appearance modeling of growing 3D structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13310v2">InstantSfM: Towards GPU-Native SfM for the Deep Learning Era</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Structure-from-Motion (SfM) is a fundamental technique for recovering camera poses and scene structure from multi-view imagery, serving as a critical upstream component for applications ranging from 3D reconstruction to modern neural scene representations such as 3D Gaussian Splatting. However, most mature SfM systems remain CPU-centric and built upon traditional optimization toolchains, creating a growing mismatch with modern GPU-based, learning-driven pipelines and limiting scalability in large-scale scenes. While recent advances in GPU-accelerated bundle adjustment (BA) have demonstrated the potential of parallel sparse optimization, extending these techniques to build a complete global SfM system remains challenging due to unresolved issues in metric scale recovery and numerical robustness. In this paper, we implement a fully GPU-based and PyTorch-compatible global SfM system, named InstantSfM, to integrate seamlessly with modern learning pipelines. InstantSfM embeds metric depth priors directly into both global positioning and BA through a depth-constrained Jacobian structure, thereby resolving scale ambiguity within the optimization framework. To ensure numerical stability, we employ explicit filtering of under-constrained variables for the Jacobian matrix in an optimized GPU-friendly manner. Extensive experiments on diverse datasets demonstrate that InstantSfM achieves state-of-the-art efficiency while maintaining reconstruction accuracy comparable to both established classical pipelines and recent learning-based methods, showing up to ${\sim40\times}$ speedup over COLMAP on large-scale scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10893v1">S2D: Sparse to Dense Lifting for 3D Reconstruction with Minimal Inputs</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Explicit 3D representations have already become an essential medium for 3D simulation and understanding. However, the most commonly used point cloud and 3D Gaussian Splatting (3DGS) each suffer from non-photorealistic rendering and significant degradation under sparse inputs. In this paper, we introduce Sparse to Dense lifting (S2D), a novel pipeline that bridges the two representations and achieves high-quality 3DGS reconstruction with minimal inputs. Specifically, the S2D lifting is two-fold. We first present an efficient one-step diffusion model that lifts sparse point cloud for high-fidelity image artifact fixing. Meanwhile, to reconstruct 3D consistent scenes, we also design a corresponding reconstruction strategy with random sample drop and weighted gradient for robust model fitting from sparse input views to dense novel views. Extensive experiments show that S2D achieves the best consistency in generating novel view guidance and first-tier sparse view reconstruction quality under different input sparsity. By reconstructing stable scenes with the least possible captures among existing methods, S2D enables minimal input requirements for 3DGS applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.14373v3">SEGA: Drivable 3D Gaussian Head Avatar from a Single Image</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Creating photorealistic 3D head avatars from limited input has become increasingly important for applications in virtual reality, telepresence, and digital entertainment. While recent advances like neural rendering and 3D Gaussian splatting have enabled high-quality digital human avatar creation and animation, most methods rely on multiple images or multi-view inputs, limiting their practicality for real-world use. In this paper, we propose SEGA, a novel approach for Single-imagE-based 3D drivable Gaussian head Avatar creation that combines generalized prior models with a new hierarchical UV-space Gaussian Splatting framework. SEGA seamlessly combines priors derived from large-scale 2D datasets with 3D priors learned from multi-view, multi-expression, and multi-ID data, achieving robust generalization to unseen identities while ensuring 3D consistency across novel viewpoints and expressions. We further present a hierarchical UV-space Gaussian Splatting framework that leverages FLAME-based structural priors and employs a dual-branch architecture to disentangle dynamic and static facial components effectively. The dynamic branch encodes expression-driven fine details, while the static branch focuses on expression-invariant regions, enabling efficient parameter inference and precomputation. This design maximizes the utility of limited 3D data and achieves real-time performance for animation and rendering. Additionally, SEGA performs person-specific fine-tuning to further enhance the fidelity and realism of the generated avatars. Experiments show our method outperforms state-of-the-art approaches in generalization ability, identity preservation, and expression realism, advancing one-shot avatar creation for practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22046v4">PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Project page: https://city-super.github.io/PLANING/
    </div>
    <details class="paper-abstract">
      Streaming reconstruction from monocular image sequences remains challenging, as existing methods typically favor either high-quality rendering or accurate geometry, but rarely both. We present PLANING, an efficient on-the-fly reconstruction framework built on a hybrid representation that loosely couples explicit geometric primitives with neural Gaussians, enabling geometry and appearance to be modeled in a decoupled manner. This decoupling supports an online initialization and optimization strategy that separates geometry and appearance updates, yielding stable streaming reconstruction with substantially reduced structural redundancy. PLANING improves dense mesh Chamfer-L2 by 18.52% over PGSR, surpasses ARTDECO by 1.31 dB PSNR, and reconstructs ScanNetV2 scenes in under 100 seconds, over 5x faster than 2D Gaussian Splatting, while matching the quality of offline per-scene optimization. Beyond reconstruction quality, the structural clarity and computational efficiency of PLANING make it well suited for a broad range of downstream applications, such as enabling large-scale scene modeling and simulation-ready environments for embodied AI. Project page: https://city-super.github.io/PLANING/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10801v1">PolGS++: Physically-Guided Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 arXiv admin note: substantial text overlap with arXiv:2509.19726
    </div>
    <details class="paper-abstract">
      Accurate reconstruction of reflective surfaces remains a fundamental challenge in computer vision, with broad applications in real-time virtual reality and digital content creation. Although 3D Gaussian Splatting (3DGS) enables efficient novel-view rendering with explicit representations, its performance on reflective surfaces still lags behind implicit neural methods, especially in recovering fine geometry and surface normals. To address this gap, we propose PolGS++, a physically-guided polarimetric Gaussian Splatting framework for fast reflective surface reconstruction. Specifically, we integrate a polarized BRDF (pBRDF) model into 3DGS to explicitly decouple diffuse and specular components, providing physically grounded reflectance modeling and stronger geometric cues for reflective surface recovery. Furthermore, we introduce a depth-guided visibility mask acquisition mechanism that enables angle-of-polarization (AoP)-based tangent-space consistency constraints in Gaussian Splatting without costly ray-tracing intersections. This physically guided design improves reconstruction quality and efficiency, requiring only about 10 minutes of training. Extensive experiments on both synthetic and real-world datasets validate the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10638v1">Splat2Real: Novel-view Scaling for Physical AI with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Physical AI faces viewpoint shift between training and deployment, and novel-view robustness is essential for monocular RGB-to-3D perception. We cast Real2Render2Real monocular depth pretraining as imitation-learning-style supervision from a digital twin oracle: a student depth network imitates expert metric depth/visibility rendered from a scene mesh, while 3DGS supplies scalable novel-view observations. We present Splat2Real, centered on novel-view scaling: performance depends more on which views are added than on raw view count. We introduce CN-Coverage, a coverage+novelty curriculum that greedily selects views by geometry gain and an extrapolation penalty, plus a quality-aware guardrail fallback for low-reliability teachers. Across 20 TUM RGB-D sequences with step-matched budgets (N=0 to 2000 additional rendered views, with N unique <= 500 and resampling for larger budgets), naive scaling is unstable; CN-Coverage mitigates worst-case regressions relative to Robot/Coverage policies, and GOL-Gated CN-Coverage provides the strongest medium-high-budget stability with the lowest high-novelty tail error. Downstream control-proxy results versus N provides embodied-relevance evidence by shifting safety/progress trade-offs under viewpoint shift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10551v1">P-GSVC: Layered Progressive 2D Gaussian Splatting for Scalable Image and Video</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 MMSys 2026; Project Website: see https://longanwang-cs.github.io/PGSVC-webpage/
    </div>
    <details class="paper-abstract">
      Gaussian splatting has emerged as a competitive explicit representation for image and video reconstruction. In this work, we present P-GSVC, the first layered progressive 2D Gaussian splatting framework that provides a unified solution for scalable Gaussian representation in both images and videos. P-GSVC organizes 2D Gaussian splats into a base layer and successive enhancement layers, enabling coarse-to-fine reconstructions. To effectively optimize this layered representation, we propose a joint training strategy that simultaneously updates Gaussians across layers, aligning their optimization trajectories to ensure inter-layer compatibility and a stable progressive reconstruction. P-GSVC supports scalability in terms of both quality and resolution. Our experiments show that the joint training strategy can gain up to 1.9 dB improvement in PSNR for video and 2.6 dB improvement in PSNR for image when compared to methods that perform sequential layer-wise training. Project page: https://longanwang-cs.github.io/PGSVC-webpage/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16410v3">REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 CVPR 2026 Accepted
    </div>
    <details class="paper-abstract">
      Bridging the gap between complex human instructions and precise 3D object grounding remains a significant challenge in vision and robotics. Existing 3D segmentation methods often struggle to interpret ambiguous, reasoning-based instructions, while 2D vision-language models that excel at such reasoning lack intrinsic 3D spatial understanding. In this paper, we introduce REALM, an innovative MLLM-agent framework that enables open-world reasoning-based segmentation without requiring extensive 3D-specific post-training. We perform segmentation directly on 3D Gaussian Splatting representations, capitalizing on their ability to render photorealistic novel views that are highly suitable for MLLM comprehension. As directly feeding one or more rendered views to the MLLM can lead to high sensitivity to viewpoint selection, we propose a novel Global-to-Local Spatial Grounding strategy. Specifically, multiple global views are first fed into the MLLM agent in parallel for coarse-level localization, aggregating responses to robustly identify the target object. Then, several close-up novel views of the object are synthesized to perform fine-grained local segmentation, yielding accurate and consistent 3D masks. Extensive experiments show that REALM achieves remarkable performance in interpreting both explicit and implicit instructions across LERF, 3D-OVS, and our newly introduced REALM3D benchmarks. Furthermore, our agent framework seamlessly supports a range of 3D interaction tasks, including object removal, replacement, and style transfer, demonstrating its practical utility and versatility. Project page: https://ChangyueShi.github.io/REALM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09968v1">ReCoSplat: Autoregressive Feed-Forward Gaussian Splatting Using Render-and-Compare</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Online novel view synthesis remains challenging, requiring robust scene reconstruction from sequential, often unposed, observations. We present ReCoSplat, an autoregressive feed-forward Gaussian Splatting model supporting posed or unposed inputs, with or without camera intrinsics. While assembling local Gaussians using camera poses scales better than canonical-space prediction, it creates a dilemma during training: using ground-truth poses ensures stability but causes a distribution mismatch when predicted poses are used at inference. To address this, we introduce a Render-and-Compare (ReCo) module. ReCo renders the current reconstruction from the predicted viewpoint and compares it with the incoming observation, providing a stable conditioning signal that compensates for pose errors. To support long sequences, we propose a hybrid KV cache compression strategy combining early-layer truncation with chunk-level selective retention, reducing the KV cache size by over 90% for 100+ frames. ReCoSplat achieves state-of-the-art performance across different input settings on both in- and out-of-distribution benchmarks. Code and pretrained models will be released. Our project page is at https://freemancheng.com/ReCoSplat .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07789v2">SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      2D Gaussian Splatting has emerged as a novel image representation technique that can support efficient rendering on low-end devices. However, scaling to high-resolution images requires optimizing and storing millions of unstructured Gaussian primitives independently, leading to slow convergence and redundant parameters. To address this, we propose Structured Gaussian Image (SGI), a compact and efficient framework for representing high-resolution images. SGI decomposes a complex image into multi-scale local spaces defined by a set of seeds. Each seed corresponds to a spatially coherent region and, together with lightweight multi-layer perceptrons (MLPs), generates structured implicit 2D neural Gaussians. This seed-based formulation imposes structural regularity on otherwise unstructured Gaussian primitives, which facilitates entropy-based compression at the seed level to reduce the total storage. However, optimizing seed parameters directly on high-resolution images is a challenging and non-trivial task. Therefore, we designed a multi-scale fitting strategy that refines the seed representation in a coarse-to-fine manner, substantially accelerating convergence. Quantitative and qualitative evaluations demonstrate that SGI achieves up to 7.5x compression over prior non-quantized 2D Gaussian methods and 1.6x over quantized ones, while also delivering 1.6x and 6.5x faster optimization, respectively, without degrading, and often improving, image fidelity. Code is available at https://github.com/zx-pan/SGI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09718v1">GSStream: 3D Gaussian Splatting based Volumetric Scene Streaming System</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Recently, the 3D Gaussian splatting (3DGS) technique for real-time radiance field rendering has revolutionized the field of volumetric scene representation, providing users with an immersive experience. But in return, it also poses a large amount of data volume, which is extremely bandwidth-intensive. Cutting-edge researchers have tried to introduce different approaches and construct multiple variants for 3DGS to obtain a more compact scene representation, but it is still challenging for real-time distribution. In this paper, we propose GSStream, a novel volumetric scene streaming system to support 3DGS data format. Specifically, GSStream integrates a collaborative viewport prediction module to better predict users' future behaviors by learning collaborative priors and historical priors from multiple users and users' viewport sequences and a deep reinforcement learning (DRL)-based bitrate adaptation module to tackle the state and action space variability challenge of the bitrate adaptation problem, achieving efficient volumetric scene delivery. Besides, we first build a user viewport trajectory dataset for volumetric scenes to support the training and streaming simulation. Extensive experiments prove that our proposed GSStream system outperforms existing representative volumetric scene streaming systems in visual quality and network usage. Demo video: https://youtu.be/3WEe8PN8yvA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09703v1">ProGS: Towards Progressive Coding for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      With the emergence of 3D Gaussian Splatting (3DGS), numerous pioneering efforts have been made to address the effective compression issue of massive 3DGS data. 3DGS offers an efficient and scalable representation of 3D scenes by utilizing learnable 3D Gaussians, but the large size of the generated data has posed significant challenges for storage and transmission. Existing methods, however, have been limited by their inability to support progressive coding, a crucial feature in streaming applications with varying bandwidth. To tackle this limitation, this paper introduce a novel approach that organizes 3DGS data into an octree structure, enabling efficient progressive coding. The proposed ProGS is a streaming-friendly codec that facilitates progressive coding for 3D Gaussian splatting, and significantly improves both compression efficiency and visual fidelity. The proposed method incorporates mutual information enhancement mechanisms to mitigate structural redundancy, leveraging the relevance between nodes in the octree hierarchy. By adapting the octree structure and dynamically adjusting the anchor nodes, ProGS ensures scalable data compression without compromising the rendering quality. ProGS achieves a remarkable 45X reduction in file storage compared to the original 3DGS format, while simultaneously improving visual performance by over 10%. This demonstrates that ProGS can provide a robust solution for real-time applications with varying network conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09673v1">VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Accepted to CVPR 2026
    </div>
    <details class="paper-abstract">
      Simultaneous Localization and Mapping (SLAM) with 3D Gaussian Splatting (3DGS) enables fast, differentiable rendering and high-fidelity reconstruction across diverse real-world scenes. However, existing 3DGS-SLAM approaches handle measurement reliability implicitly, making pose estimation and global alignment susceptible to drift in low-texture regions, transparent surfaces, or areas with complex reflectance properties. To this end, we introduce VarSplat, an uncertainty-aware 3DGS-SLAM system that explicitly learns per-splat appearance variance. By using the law of total variance with alpha compositing, we then render differentiable per-pixel uncertainty map via efficient, single-pass rasterization. This map guides tracking, submap registration, and loop detection toward focusing on reliable regions and contributes to more stable optimization. Experimental results on Replica (synthetic) and TUM-RGBD, ScanNet, and ScanNet++ (real-world) show that VarSplat improves robustness and achieves competitive or superior tracking, mapping, and novel view synthesis rendering compared to existing studies for dense RGB-D SLAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09668v1">DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Accepted by ICLR 2026. Project page: https://zju3dv.github.io/DiffWind/
    </div>
    <details class="paper-abstract">
      Modeling wind-driven object dynamics from video observations is highly challenging due to the invisibility and spatio-temporal variability of wind, as well as the complex deformations of objects. We present DiffWind, a physics-informed differentiable framework that unifies wind-object interaction modeling, video-based reconstruction, and forward simulation. Specifically, we represent wind as a grid-based physical field and objects as particle systems derived from 3D Gaussian Splatting, with their interaction modeled by the Material Point Method (MPM). To recover wind-driven object dynamics, we introduce a reconstruction framework that jointly optimizes the spatio-temporal wind force field and object motion through differentiable rendering and simulation. To ensure physical validity, we incorporate the Lattice Boltzmann Method (LBM) as a physics-informed constraint, enforcing compliance with fluid dynamics laws. Beyond reconstruction, our method naturally supports forward simulation under novel wind conditions and enables new applications such as wind retargeting. We further introduce WD-Objects, a dataset of synthetic and real-world wind-driven scenes. Extensive experiments demonstrate that our method significantly outperforms prior dynamic scene modeling approaches in both reconstruction accuracy and simulation fidelity, opening a new avenue for video-based wind-object interaction modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.04859v3">CoRe-GS: Coarse-to-Refined Gaussian Splatting with Semantic Object Focus</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Fast and efficient 3D reconstruction is essential for time-critical robotic applications such as tele-guidance and disaster response, where operators must rapidly analyze specific points of interest (POIs). Existing semantic Gaussian Splatting (GS) approaches optimize the entire scene uniformly, incurring substantial computational cost even when only a small subset of the scene is operationally relevant. We propose CoRe-GS, a coarse-to-refine GS framework that enables task-driven POI-focused optimization. Our method first produces a segmentation-ready GS representation using a lightweight late-stage semantic refinement. Subsequently, only Gaussians associated with the selected POI are further optimized, reducing unnecessary background computation. To mitigate segmentation-induced outliers (floaters) during selective refinement, we introduce a color-based filtering mechanism that removes inconsistent Gaussians without requiring mask rasterization. We evaluate robustness multiple datasets. On LERF-Mask, our segmentation-ready representation achieves competitive mIoU using tremendously fewer optimization steps. Across synthetic and real-world datasets (NeRDS360, SCRREAM, Tanks and Temples), CoRe-GS drastically reduces training time compared to full semantic GS while improving POI reconstruction quality and mitigating floaters. These results demonstrate that task-aware selective refinement enables faster and higher-quality scene reconstruction tailored to robotic operational needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.18380v2">ARSGaussian: 3D Gaussian Splatting with LiDAR for Aerial Remote Sensing Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 This is the author's version of a work that was accepted for publication in [ISPRS]. Changes resulting from the publishing process... may not be reflected in this document
    </div>
    <details class="paper-abstract">
      Novel View Synthesis (NVS) can reconstruct scenes from multi-view images and synthesize novel images from new viewpoints, which provides technical support for tasks such as target recognition and environmental perception. Aerial remote sensing can conveniently capture a wealth of multi-view images with just a few flights. However, the challenges brought by large distances and sparse viewing angles during collection can cause the model to easily produce floaters and overgrowth issues due to geometric estimation errors. This results in low visual quality and a lack of precise geometric estimation capabilities. Therefore, this study presents ARSGaussian, an innovative novel view synthesis (NVS) method for aerial remote sensing. The method incorporates LiDAR point cloud as constraints into the 3D Gaussian Splatting approach, adaptively guiding the Gaussians to grow and split along geometric benchmarks, thereby addressing the overgrowth and floaters issues. Additionally, considering the geometric distortions arising from data acquisition, coordinate transformations with distortion parameters are integrated to replace the simple pinhole camera model parameters to achieve pixel-level alignment between LiDAR point cloud and multi-view optical images, facilitating the accurate fusion of heterogeneous data and achieving the high-precision geo-alignment. Moreover, depth, normal and scale consistency losses are introduced into the regularization process to guide Gaussians toward real depth and plane representations, significantly improving geometric estimation accuracy. To address the current lack of dense airborne hybrid datasets, we have established and released AIR-LONGYAN, an open-source dataset containing a dense LiDAR point cloud (8 pts/m) and multi-view optical images captured by airborne scanners and cameras in diverse scenes....
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09291v1">DenoiseSplat: Feed-Forward Gaussian Splatting for Noisy 3D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      3D scene reconstruction and novel-view synthesis are fundamental for VR, robotics, and content creation. However, most NeRF and 3D Gaussian Splatting pipelines assume clean inputs and degrade under real noise and artifacts. We therefore propose DenoiseSplat, a feed-forward 3D Gaussian splatting method for noisy multi-view images. We build a large-scale, scene-consistent noisy--clean benchmark on RE10K by injecting Gaussian, Poisson, speckle, and salt-and-pepper noise with controlled intensities. With a lightweight MVSplat-style feed-forward backbone, we train end-to-end using only clean 2D renderings as supervision and no 3D ground truth. On noisy RE10K, DenoiseSplat outperforms vanilla MVSplat and a strong two-stage baseline (IDF + MVSplat) in PSNR/SSIM and LPIPS across noise types and levels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09285v1">Learning Convex Decomposition via Feature Fields</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 14 pages, 12 figures
    </div>
    <details class="paper-abstract">
      This work proposes a new formulation to the long-standing problem of convex decomposition through learning feature fields, enabling the first feed-forward model for open-world convex decomposition. Our method produces high-quality decompositions of 3D shapes into a union of convex bodies, which are essential to accelerate collision detection in physical simulation, amongst many other applications. The key insight is to adopt a feature learning approach and learn a continuous feature field that can later be clustered to yield a good convex decomposition via our self-supervised, purely-geometric objective derived from the classical definition of convexity. Our formulation can be used for single shape optimization, but more importantly, feature prediction unlocks scalable, self-supervised learning on large datasets resulting in the first learned open-world model for convex decomposition. Experiments show that our decompositions are higher-quality than alternatives and generalize across open-world objects as well as across representations to meshes, CAD models, and even Gaussian splats. https://research.nvidia.com/labs/sil/projects/learning-convex-decomp/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09277v1">Speeding Up the Learning of 3D Gaussians with Much Shorter Gaussian Lists</a></div>
    <div class="paper-meta">
      📅 2026-03-10
      | 💬 Accepted to CVPR 2026. Project page: https://github.com/MachinePerceptionLab/ShorterSplatting
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has become a vital tool for learning a radiance field from multiple posed images. Although 3DGS shows great advantages over NeRF in terms of rendering quality and efficiency, it remains a research challenge to further improve the efficiency of learning 3D Gaussians. To overcome this challenge, we propose novel training strategies and losses to shorten each Gaussian list used to render a pixel, which speeds up the splatting by involving fewer Gaussians along a ray. Specifically, we shrink the size of each Gaussian by resetting their scales regularly, encouraging smaller Gaussians to cover fewer nearby pixels, which shortens the Gaussian lists of pixels. Additionally, we introduce an entropy constraint on the alpha blending procedure to sharpen the weight distribution of Gaussians along each ray, which drives dominant weights larger while making minor weights smaller. As a result, each Gaussian becomes more focused on the pixels where it is dominant, which reduces its impact on nearby pixels, leading to even shorter Gaussian lists. Eventually, we integrate our method into a rendering resolution scheduler which further improves efficiency through progressive resolution increase. We evaluate our method by comparing it with state-of-the-art methods on widely used benchmarks. Our results show significant advantages over others in efficiency without sacrificing rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08997v1">SkipGS: Post-Densification Backward Skipping for Efficient 3DGS Training</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves real-time novel-view synthesis by optimizing millions of anisotropic Gaussians, yet its training remains expensive, with the backward pass dominating runtime in the post-densification refinement phase. We observe substantial update redundancy in this phase: many sampled views have near-plateaued losses and provide diminishing gradient benefits, but standard training still runs full backpropagation. We propose SkipGS with a novel view-adaptive backward gating mechanism for efficient post-densification training. SkipGS always performs the forward pass to update per-view loss statistics, and selectively skips backward passes when the sampled view's loss is consistent with its recent per-view baseline, while enforcing a minimum backward budget for stable optimization. On Mip-NeRF 360, compared to 3DGS, SkipGS reduces end-to-end training time by 23.1%, driven by a 42.0% reduction in post-densification time, with comparable reconstruction quality. Because it only changes when to backpropagate -- without modifying the renderer, representation, or loss -- SkipGS is plug-and-play and compatible with other complementary efficiency strategies for additive speedups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08983v1">SurgCalib: Gaussian Splatting-Based Hand-Eye Calibration for Robot-Assisted Minimally Invasive Surgery</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 9 pages, 7 figures
    </div>
    <details class="paper-abstract">
      We present a Gaussian Splatting-based framework for hand-eye calibration of the da Vinci surgical robot. In a vision-guided robotic system, accurate estimation of the rigid transformation between the robot base and the camera frame is essential for reliable closed-loop control. For cable-driven surgical robots, this task faces unique challenges. The encoders of surgical instruments often produce inaccurate proprioceptive measurements due to cable stretch and backlash. Conventional hand-eye calibration approaches typically rely on known fiducial patterns and solve the AX = XB formulation. While effective, introducing additional markers into the operating room (OR) environment can violate sterility protocols and disrupt surgical workflows. In this study, we propose SurgCalib, an automatic, markerless framework that has the potential to be used in the OR. SurgCalib first initializes the pose of the surgical instrument using raw kinematic measurements and subsequently refines this pose through a two-phase optimization procedure under the RCM constraint within a Gaussian Splatting-based differentiable rendering pipeline. We evaluate the proposed method on the public dVRK benchmark, SurgPose. The results demonstrate average 2D tool-tip reprojection errors of 12.24 px (2.06 mm) and 11.33 px (1.9 mm), and 3D tool-tip Euclidean distance errors of 5.98 mm and 4.75 mm, for the left and right instruments, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08809v1">Where, What, Why: Toward Explainable 3D-GS Watermarking</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      As 3D Gaussian Splatting becomes the de facto representation for interactive 3D assets, robust yet imperceptible watermarking is critical. We present a representation-native framework that separates where to write from how to preserve quality. A Trio-Experts module operates directly on Gaussian primitives to derive priors for carrier selection, while a Safety and Budget Aware Gate (SBAG) allocates Gaussians to watermark carriers, optimized for bit resilience under perturbation and bitrate budgets, and to visual compensators that are insulated from watermark loss. To maintain fidelity, we introduce a channel-wise group mask that controls gradient propagation for carriers and compensators, thereby limiting Gaussian parameter updates, repairing local artifacts, and preserving high-frequency details without increasing runtime. Our design yields view-consistent watermark persistence and strong robustness against common image distortions such as compression and noise, while achieving a favorable robustness-quality trade-off compared with prior methods. In addition, decoupled finetuning provides per-Gaussian attributions that reveal where the message is carried and why those carriers are selected, enabling auditable explainability. Compared with state-of-the-art methods, our approach achieves a PSNR improvement of +0.83 dB and a bit-accuracy gain of +1.24%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08661v1">ImprovedGS+: A High-Performance C++/CUDA Re-Implementation Strategy for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 6 pages, 1 figure. Technical Report. This work introduces ImprovedGS+, a library-free C++/CUDA implementation for 3D Gaussian Splatting within the LichtFeld-Studio framework. Source code available at https://github.com/jordizv/ImprovedGS-Plus
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3DGS) have shifted the focus toward balancing reconstruction fidelity with computational efficiency. In this work, we propose ImprovedGS+, a high-performance, low-level reinvention of the ImprovedGS strategy, implemented natively within the LichtFeld-Studio framework. By transitioning from high-level Python logic to hardware-optimized C++/CUDA kernels, we achieve a significant reduction in host-device synchronization and training latency. Our implementation introduces a Long-Axis-Split (LAS) CUDA kernel, custom Laplacian-based importance kernels with Non-Maximum Suppression (NMS) for edge scores, and an adaptive Exponential Scale Scheduler. Experimental results on the Mip-NeRF360 dataset demonstrate that ImprovedGS+ establishes a new Pareto-optimal front for scene reconstruction. Our 1M-budget variant outperforms the state-of-the-art MCMC baseline by achieving a 26.8% reduction in training time (saving 17 minutes per session) and utilizing 13.3% fewer Gaussians while maintaining superior visual quality. Furthermore, our full variant demonstrates a 1.28 dB PSNR increase over the ADC baseline with a 38.4% reduction in parametric complexity. These results validate ImprovedGS+ as a scalable, high-speed solution that upholds the core pillars of Speed, Quality, and Usability within the LichtFeld-Studio ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.23273v2">LIVE-GS: Online LiDAR-Inertial-Visual State Estimation and Globally Consistent Mapping with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) enabled photorealistic mapping, its integration into SLAM has largely followed traditional camera-centric pipelines. As a result, they inherit well-known weaknesses such as high computational load, failure in texture-poor or illumination-varying environments, and limited operational range, particularly for RGB-D setups. On the other hand, LiDAR emerges as a robust alternative, but its integration with 3DGS introduces new challenges, such as the need for tighter global alignment for photorealistic quality and prolonged optimization times caused by sparse data. To address these challenges, we propose LIVE-GS, an online LiDAR-Inertial Visual SLAM framework that tightly couples 3D Gaussian Splatting with LiDAR-based surfels to ensure high-precision map consistency through global geometric optimization. Particularly, to handle sparse data, our system employs a depth-invariant Gaussian initialization strategy for efficient representation and a bounded sigmoid constraint to prevent uncontrolled Gaussian growth. Experiments on public and our datasets demonstrate competitive performance in rendering quality and map-building efficiency compared with representative 3DGS SLAM baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08503v1">Spherical-GOF: Geometry-Aware Panoramic Gaussian Opacity Fields for 3D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 The source code and dataset will be released at https://github.com/1170632760/Spherical-GOF
    </div>
    <details class="paper-abstract">
      Omnidirectional images are increasingly used in robotics and vision due to their wide field of view. However, extending 3D Gaussian Splatting (3DGS) to panoramic camera models remains challenging, as existing formulations are designed for perspective projections and naive adaptations often introduce distortion and geometric inconsistencies. We present Spherical-GOF, an omnidirectional Gaussian rendering framework built upon Gaussian Opacity Fields (GOF). Unlike projection-based rasterization, Spherical-GOF performs GOF ray sampling directly on the unit sphere in spherical ray space, enabling consistent ray-Gaussian interactions for panoramic rendering. To make the spherical ray casting efficient and robust, we derive a conservative spherical bounding rule for fast ray-Gaussian culling and introduce a spherical filtering scheme that adapts Gaussian footprints to distortion-varying panoramic pixel sampling. Extensive experiments on standard panoramic benchmarks (OmniBlender and OmniPhotos) demonstrate competitive photometric quality and substantially improved geometric consistency. Compared with the strongest baseline, Spherical-GOF reduces depth reprojection error by 57% and improves cycle inlier ratio by 21%. Qualitative results show cleaner depth and more coherent normal maps, with strong robustness to global panorama rotations. We further validate generalization on OmniRob, a real-world robotic omnidirectional dataset introduced in this work, featuring UAV and quadruped platforms. The source code and the OmniRob dataset will be released at https://github.com/1170632760/Spherical-GOF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08499v1">Improving Continual Learning for Gaussian Splatting based Environments Reconstruction on Commercial Off-the-Shelf Edge Devices</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) is increasingly relevant for edge robotics, where compact and incrementally updatable 3D scene models are needed for SLAM, navigation, and inspection under tight memory and latency budgets. Variational Bayesian Gaussian Splatting (VBGS) enables replay-free continual updates for the 3DGS algorithm by maintaining a probabilistic scene model, but its high-precision computations and large intermediate tensors make on-device training impractical. We present a precision-adaptive optimization framework that enables VBGS training on resource-constrained hardware without altering its variational formulation. We (i) profile VBGS to identify memory/latency hotspots, (ii) fuse memory-dominant kernels to reduce materialized intermediate tensors, and (iii) automatically assign operation-level precisions via a mixed-precision search with bounded relative error. Across the Blender, Habitat, and Replica datasets, our optimised pipeline reduces peak memory from 9.44 GB to 1.11 GB and training time from ~234 min to ~61 min on an A5000 GPU, while preserving (and in some cases improving) reconstruction quality of the state-of-the-art VBGS baseline. We also enable for the first time NVS training on a commercial embedded platform, the Jetson Orin Nano, reducing per-frame latency by 19x compared to 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24758v5">ExGS: Extreme 3D Gaussian Compression with Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      Neural scene representations, such as 3D Gaussian Splatting (3DGS), have enabled high-quality neural rendering; however, their large storage and transmission costs hinder deployment in resource-constrained environments. Existing compression methods either rely on costly optimization, which is slow and scene-specific, or adopt training-free pruning and quantization, which degrade rendering quality under high compression ratios. In contrast, recent data-driven approaches provide a promising direction to overcome this trade-off, enabling efficient compression while preserving high rendering quality. We introduce ExGS, a novel feed-forward framework that unifies Universal Gaussian Compression (UGC) with GaussPainter for Extreme 3DGS compression. UGC performs re-optimization-free pruning to aggressively reduce Gaussian primitives while retaining only essential information, whereas GaussPainter leverages powerful diffusion priors with mask-guided refinement to restore high-quality renderings from heavily pruned Gaussian scenes. Unlike conventional inpainting, GaussPainter not only fills in missing regions but also enhances visible pixels, yielding substantial improvements in degraded renderings. To ensure practicality, it adopts a lightweight VAE and a one-step diffusion design, enabling real-time restoration. Our framework can even achieve over 100X compression (reducing a typical 354.77 MB model to about 3.31 MB) while preserving fidelity and significantly improving image quality under challenging conditions. These results highlight the central role of diffusion priors in bridging the gap between extreme compression and high-quality neural rendering. Our code repository will be released at: https://github.com/chenttt2001/ExGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08313v1">HDR-NSFF: High Dynamic Range Neural Scene Flow Fields</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 ICLR 2026. Project page: https://shin-dong-yeon.github.io/HDR-NSFF/
    </div>
    <details class="paper-abstract">
      Radiance of real-world scenes typically spans a much wider dynamic range than what standard cameras can capture. While conventional HDR methods merge alternating-exposure frames, these approaches are inherently constrained to 2D pixel-level alignment, often leading to ghosting artifacts and temporal inconsistency in dynamic scenes. To address these limitations, we present HDR-NSFF, a paradigm shift from 2D-based merging to 4D spatio-temporal modeling. Our framework reconstructs dynamic HDR radiance fields from alternating-exposure monocular videos by representing the scene as a continuous function of space and time, and is compatible with both neural radiance field and 4D Gaussian Splatting (4DGS) based dynamic representations. This unified end-to-end pipeline explicitly models HDR radiance, 3D scene flow, geometry, and tone-mapping, ensuring physical plausibility and global coherence. We further enhance robustness by (i) extending semantic-based optical flow with DINO features to achieve exposure-invariant motion estimation, and (ii) incorporating a generative prior as a regularizer to compensate for limited observation in monocular captures and saturation-induced information loss. To evaluate HDR space-time view synthesis, we present the first real-world HDR-GoPro dataset specifically designed for dynamic HDR scenes. Experiments demonstrate that HDR-NSFF recovers fine radiance details and coherent dynamics even under challenging exposure variations, thereby achieving state-of-the-art performance in novel space-time view synthesis. Project page: https://shin-dong-yeon.github.io/HDR-NSFF/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14191v3">MCGS-SLAM: A Multi-Camera SLAM Framework Using Gaussian Splatting for High-Fidelity Mapping</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026
    </div>
    <details class="paper-abstract">
      Recent progress in dense SLAM has primarily targeted monocular setups, often at the expense of robustness and geometric coverage. We present MCGS-SLAM, the first purely RGB-based multi-camera SLAM system built on 3D Gaussian Splatting (3DGS). Unlike prior methods relying on sparse maps or inertial data, MCGS-SLAM fuses dense RGB inputs from multiple viewpoints into a unified, continuously optimized Gaussian map. A multi-camera bundle adjustment (MCBA) jointly refines poses and depths via dense photometric and geometric residuals, while a scale consistency module enforces metric alignment across views using low-rank priors. The system supports RGB input and maintains real-time performance at large scale. Experiments on synthetic and real-world datasets show that MCGS-SLAM consistently yields accurate trajectories and photorealistic reconstructions, usually outperforming monocular baselines. Notably, the wide field of view from multi-camera input enables reconstruction of side-view regions that monocular setups miss, critical for safe autonomous operation. These results highlight the promise of multi-camera Gaussian Splatting SLAM for high-fidelity mapping in robotics and autonomous driving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08254v1">DynamicVGGT: Learning Dynamic Point Maps for 4D Scene Reconstruction in Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction in autonomous driving remains a fundamental challenge due to significant temporal variations, moving objects, and complex scene dynamics. Existing feed-forward 3D models have demonstrated strong performance in static reconstruction but still struggle to capture dynamic motion. To address these limitations, we propose DynamicVGGT, a unified feed-forward framework that extends VGGT from static 3D perception to dynamic 4D reconstruction. Our goal is to model point motion within feed-forward 3D models in a dynamic and temporally coherent manner. To this end, we jointly predict the current and future point maps within a shared reference coordinate system, allowing the model to implicitly learn dynamic point representations through temporal correspondence. To efficiently capture temporal dependencies, we introduce a Motion-aware Temporal Attention (MTA) module that learns motion continuity. Furthermore, we design a Dynamic 3D Gaussian Splatting Head that explicitly models point motion by predicting Gaussian velocities using learnable motion tokens under scene flow supervision. It refines dynamic geometry through continuous 3D Gaussian optimization. Extensive experiments on autonomous driving datasets demonstrate that DynamicVGGT significantly outperforms existing methods in reconstruction accuracy, achieving robust feed-forward 4D dynamic scene reconstruction under complex driving scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.13911v3">PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      Despite advances in physics-based 3D motion synthesis, current methods face key limitations: reliance on pre-reconstructed 3D Gaussian Splatting (3DGS) built from dense multi-view images with time-consuming per-scene optimization; physics integration via either inflexible, hand-specified attributes or unstable, optimization-heavy guidance from video models using Score Distillation Sampling (SDS); and naive concatenation of prebuilt 3DGS with physics modules, which ignores physical information embedded in appearance and yields suboptimal performance. To address these issues, we propose PhysGM, a feed-forward framework that jointly predicts 3D Gaussian representation and physical properties from a single image, enabling immediate simulation and high-fidelity 4D rendering. Unlike slow appearance-agnostic optimization methods, we first pre-train a physics-aware reconstruction model that directly infers both Gaussian and physical parameters. We further refine the model with Direct Preference Optimization (DPO), aligning simulations with the physically plausible reference videos and avoiding the high-cost SDS optimization. To address the absence of a supporting dataset for this task, we propose PhysAssets, a dataset of 50K+ 3D assets annotated with physical properties and corresponding reference videos. Experiments show that PhysGM produces high-fidelity 4D simulations from a single image in one minute, achieving a significant speedup over prior work while delivering realistic renderings.Our project page is at:https://hihixiaolv.github.io/PhysGM.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.17635v3">LangSurf: Language-Embedded Surface Gaussians for 3D Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 \url{https://langsurf.github.io}
    </div>
    <details class="paper-abstract">
      Applying Gaussian Splatting to perception tasks for 3D scene understanding is becoming increasingly popular. Most existing works primarily focus on rendering 2D feature maps from novel viewpoints, which leads to an imprecise 3D language field with outlier languages, ultimately failing to align objects in 3D space. By utilizing masked images for feature extraction, these approaches also lack essential contextual information, leading to inaccurate feature representation. To this end, we propose a Language-Embedded Surface Field (LangSurf), which accurately aligns the 3D language fields with the surface of objects, facilitating precise 2D and 3D segmentation with text query, widely expanding the downstream tasks such as removal and editing. The core of LangSurf is a joint training strategy that flattens the language Gaussian on the object surfaces using geometry supervision and contrastive losses to assign accurate language features to the Gaussians of objects. In addition, we also introduce the Hierarchical-Context Awareness Module to extract features at the image level for contextual information then perform hierarchical mask pooling using masks segmented by SAM to obtain fine-grained language features in different hierarchies. Extensive experiments on open-vocabulary 2D and 3D semantic segmentation demonstrate that LangSurf outperforms the previous state-of-the-art method LangSplat by a large margin. As shown in Fig. 1, our method is capable of segmenting objects in 3D space, thus boosting the effectiveness of our approach in instance recognition, removal, and editing, which is also supported by comprehensive experiments. https://langsurf.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07664v1">Ref-DGS: Reflective Dual Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-08
      | 💬 Project page: https://straybirdflower.github.io/Ref-DGS/
    </div>
    <details class="paper-abstract">
      Reflective appearance, especially strong and typically near-field specular reflections, poses a fundamental challenge for accurate surface reconstruction and novel view synthesis. Existing Gaussian splatting methods either fail to model near-field specular reflections or rely on explicit ray tracing at substantial computational cost. We present Ref-DGS, a reflective dual Gaussian splatting framework that addresses this trade-off by decoupling surface reconstruction from specular reflection within an efficient rasterization-based pipeline. Ref-DGS introduces a dual Gaussian scene representation consisting of geometry Gaussians and complementary local reflection Gaussians that capture near-field specular interactions without explicit ray tracing, along with a global environment reflection field for modeling far-field specular reflections. To predict specular radiance, we further propose a lightweight, physically-aware adaptive mixing shader that fuses global and local reflection features. Experiments demonstrate that Ref-DGS achieves state-of-the-art performance on reflective scenes while training substantially faster than ray-based Gaussian methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07660v1">Holi-Spatial: Evolving Video Streams into Holistic 3D Spatial Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-03-08
      | 💬 project page: https://visionary-laboratory.github.io/holi-spatial/
    </div>
    <details class="paper-abstract">
      The pursuit of spatial intelligence fundamentally relies on access to large-scale, fine-grained 3D data. However, existing approaches predominantly construct spatial understanding benchmarks by generating question-answer (QA) pairs from a limited number of manually annotated datasets, rather than systematically annotating new large-scale 3D scenes from raw web data. As a result, their scalability is severely constrained, and model performance is further hindered by domain gaps inherent in these narrowly curated datasets. In this work, we propose Holi-Spatial, the first fully automated, large-scale, spatially-aware multimodal dataset, constructed from raw video inputs without human intervention, using the proposed data curation pipeline. Holi-Spatial supports multi-level spatial supervision, ranging from geometrically accurate 3D Gaussian Splatting (3DGS) reconstructions with rendered depth maps to object-level and relational semantic annotations, together with corresponding spatial Question-Answer (QA) pairs. Following a principled and systematic pipeline, we further construct Holi-Spatial-4M, the first large-scale, high-quality 3D semantic dataset, containing 12K optimized 3DGS scenes, 1.3M 2D masks, 320K 3D bounding boxes, 320K instance captions, 1.2M 3D grounding instances, and 1.2M spatial QA pairs spanning diverse geometric, relational, and semantic reasoning tasks. Holi-Spatial demonstrates exceptional performance in data curation quality, significantly outperforming existing feed-forward and per-scene optimized methods on datasets such as ScanNet, ScanNet++, and DL3DV. Furthermore, fine-tuning Vision-Language Models (VLMs) on spatial reasoning tasks using this dataset has also led to substantial improvements in model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07604v1">EmbedTalk: Triplane-Free Talking Head Synthesis using Embedding-Driven Gaussian Deformation</a></div>
    <div class="paper-meta">
      📅 2026-03-08
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Real-time talking head synthesis increasingly relies on deformable 3D Gaussian Splatting (3DGS) due to its low latency. Tri-planes are the standard choice for encoding Gaussians prior to deformation, since they provide a continuous domain with explicit spatial relationships. However, tri-plane representations are limited by grid resolution and approximation errors introduced by projecting 3D volumetric fields onto 2D subspaces. Recent work has shown the superiority of learnt embeddings for driving temporal deformations in 4D scene reconstruction. We introduce $\textbf{EmbedTalk}$, which shows how such embeddings can be leveraged for modelling speech deformations in talking head synthesis. Through comprehensive experiments, we show that EmbedTalk outperforms existing 3DGS-based methods in rendering quality, lip synchronisation, and motion consistency, while remaining competitive with state-of-the-art generative models. Moreover, replacing the tri-plane encoding with learnt embeddings enables significantly more compact models that achieve over 60 FPS on a mobile GPU (RTX 2060 6 GB). Our code will be placed in the public domain on acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07587v1">3DGS-HPC: Distractor-free 3D Gaussian Splatting with Hybrid Patch-wise Classification</a></div>
    <div class="paper-meta">
      📅 2026-03-08
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated remarkable performance in novel view synthesis and 3D scene reconstruction, yet its quality often degrades in real-world environments due to transient distractors, such as moving objects and varying shadows. Existing methods commonly rely on semantic cues extracted from pre-trained vision models to identify and suppress these distractors, but such semantics are misaligned with the binary distinction between static and transient regions and remain fragile under the appearance perturbations introduced during 3DGS optimization. We propose 3DGS-HPC, a framework that circumvents these limitations by combining two complementary principles: a patch-wise classification strategy that leverages local spatial consistency for robust region-level decisions, and a hybrid classification metric that adaptively integrates photometric and perceptual cues for more reliable separation. Extensive experiments demonstrate the superiority and robustness of our method in mitigating distractors to improve 3DGS-based novel view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07552v1">ReconDrive: Fast Feed-Forward 4D Gaussian Splatting for Autonomous Driving Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-08
    </div>
    <details class="paper-abstract">
      High-fidelity visual reconstruction and novel-view synthesis are essential for realistic closed-loop evaluation in autonomous driving. While 4D Gaussian Splatting (4DGS) offers a promising balance of accuracy and efficiency, existing per-scene optimization methods require costly iterative refinement, rendering them unscalable for extensive urban environments. Conversely, current feed-forward approaches often suffer from degraded photometric quality. To address these limitations, we propose ReconDrive, a feed-forward framework that leverages and extends the 3D foundation model VGGT for rapid, high-fidelity 4DGS generation. Our architecture introduces two core adaptations to tailor the foundation model to dynamic driving scenes: (1) Hybrid Gaussian Prediction Heads, which decouple the regression of spatial coordinates and appearance attributes to overcome the photometric deficiencies inherent in generalized foundation features; and (2) a Static-Dynamic 4D Composition strategy that explicitly captures temporal motion via velocity modeling to represent complex dynamic environments. Benchmarked on nuScenes, ReconDrive significantly outperforms existing feed-forward baselines in reconstruction, novel-view synthesis, and 3D perception. It achieves performance competitive with per-scene optimization while being orders of magnitude faster, providing a scalable and practical solution for realistic driving simulation.
    </details>
</div>
