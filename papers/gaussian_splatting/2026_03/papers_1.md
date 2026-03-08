# gaussian splatting - 2026_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05152v1">SSR-GS: Separating Specular Reflection in Gaussian Splatting for Glossy Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Project page: https://gsflyer.github.io/SSR-GS/
    </div>
    <details class="paper-abstract">
      In recent years, 3D Gaussian splatting (3DGS) has achieved remarkable progress in novel view synthesis. However, accurately reconstructing glossy surfaces under complex illumination remains challenging, particularly in scenes with strong specular reflections and multi-surface interreflections. To address this issue, we propose SSR-GS, a specular reflection modeling framework for glossy surface reconstruction. Specifically, we introduce a prefiltered Mip-Cubemap to model direct specular reflections efficiently, and propose an IndiASG module to capture indirect specular reflections. Furthermore, we design Visual Geometry Priors (VGP) that couple a reflection-aware visual prior via a reflection score (RS) to downweight the photometric loss contribution of reflection-dominated regions, with geometry priors derived from VGGT, including progressively decayed depth supervision and transformed normal constraints. Extensive experiments on both synthetic and real-world datasets demonstrate that SSR-GS achieves state-of-the-art performance in glossy surface reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05108v1">GaussTwin: Unified Simulation and Correction with Gaussian Splatting for Robotic Digital Twins</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 8 pages, 4 figures, 3 tables, ICRA 2026
    </div>
    <details class="paper-abstract">
      Digital twins promise to enhance robotic manipulation by maintaining a consistent link between real-world perception and simulation. However, most existing systems struggle with the lack of a unified model, complex dynamic interactions, and the real-to-sim gap, which limits downstream applications such as model predictive control. Thus, we propose GaussTwin, a real-time digital twin that combines position-based dynamics with discrete Cosserat rod formulations for physically grounded simulation, and Gaussian splatting for efficient rendering and visual correction. By anchoring Gaussians to physical primitives and enforcing coherent SE(3) updates driven by photometric error and segmentation masks, GaussTwin achieves stable prediction-correction while preserving physical fidelity. Through experiments in both simulation and on a Franka Research 3 platform, we show that GaussTwin consistently improves tracking accuracy and robustness compared to shape-matching and rigid-only baselines, while also enabling downstream tasks such as push-based planning. These results highlight GaussTwin as a step toward unified, physically meaningful digital twins that can support closed-loop robotic interaction and learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.24096v2">DiffusionHarmonizer: Bridging Neural Reconstruction and Photorealistic Simulation with Online Diffusion Enhancer</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 For more details and updates, please visit our project website: https://research.nvidia.com/labs/sil/projects/diffusion-harmonizer
    </div>
    <details class="paper-abstract">
      Simulation is essential to the development and evaluation of autonomous robots such as self-driving vehicles. Neural reconstruction is emerging as a promising solution as it enables simulating a wide variety of scenarios from real-world data alone in an automated and scalable way. However, while methods such as NeRF and 3D Gaussian Splatting can produce visually compelling results, they often exhibit artifacts particularly when rendering novel views, and fail to realistically integrate inserted dynamic objects, especially when they were captured from different scenes. To overcome these limitations, we introduce DiffusionHarmonizer, an online generative enhancement framework that transforms renderings from such imperfect scenes into temporally consistent outputs while improving their realism. At its core is a single-step temporally-conditioned enhancer that is converted from a pretrained multi-step image diffusion model, capable of running in online simulators on a single GPU. The key to training it effectively is a custom data curation pipeline that constructs synthetic-real pairs emphasizing appearance harmonization, artifact correction, and lighting realism. The result is a scalable system that significantly elevates simulation fidelity in both research and production environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13454v2">Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 ICLR 2026 (Oral), Project page: https://gohyojun15.github.io/VIST3A/
    </div>
    <details class="paper-abstract">
      The rapid progress of large, pretrained models for both visual content generation and 3D reconstruction opens up new possibilities for text-to-3D generation. Intuitively, one could obtain a formidable 3D scene generator if one were able to combine the power of a modern latent text-to-video model as "generator" with the geometric abilities of a recent (feedforward) 3D reconstruction system as "decoder". We introduce VIST3A, a general framework that does just that, addressing two main challenges. First, the two components must be joined in a way that preserves the rich knowledge encoded in their weights. We revisit model stitching, i.e., we identify the layer in the 3D decoder that best matches the latent representation produced by the text-to-video generator and stitch the two parts together. That operation requires only a small dataset and no labels. Second, the text-to-video generator must be aligned with the stitched 3D decoder, to ensure that the generated latents are decodable into consistent, perceptually convincing 3D scene geometry. To that end, we adapt direct reward finetuning, a popular technique for human preference alignment. We evaluate the proposed VIST3A approach with different video generators and 3D reconstruction models. All tested pairings markedly improve over prior text-to-3D models that output Gaussian splats. Moreover, by choosing a suitable 3D base model, VIST3A also enables high-quality text-to-pointmap generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19854v3">STAvatar: Soft Binding and Temporal Density Control for Monocular 3D Head Avatars Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Accepted to CVPR 2026. Project page: https://jiankuozhao.github.io/STAvatar/
    </div>
    <details class="paper-abstract">
      Reconstructing high-fidelity and animatable 3D head avatars from monocular videos remains a challenging yet essential task. Existing methods based on 3D Gaussian Splatting typically bind Gaussians to mesh triangles and model deformations solely via Linear Blend Skinning, which results in rigid motion and limited expressiveness. Moreover, they lack specialized strategies to handle frequently occluded regions (e.g., mouth interiors, eyelids). To address these limitations, we propose STAvatar, which consists of two key components: (1) a UV-Adaptive Soft Binding framework that leverages both image-based and geometric priors to learn per-Gaussian feature offsets within the UV space. This UV representation supports dynamic resampling, ensuring full compatibility with Adaptive Density Control (ADC) and enhanced adaptability to shape and textural variations. (2) a Temporal ADC strategy, which first clusters structurally similar frames to facilitate more targeted computation of the densification criterion. It further introduces a novel fused perceptual error as clone criterion to jointly capture geometric and textural discrepancies, encouraging densification in regions requiring finer details. Extensive experiments on four benchmark datasets demonstrate that STAvatar achieves state-of-the-art reconstruction performance, especially in capturing fine-grained details and reconstructing frequently occluded regions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04847v1">GloSplat: Joint Pose-Appearance Optimization for Faster and More Accurate 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      Feature extraction, matching, structure from motion (SfM), and novel view synthesis (NVS) have traditionally been treated as separate problems with independent optimization objectives. We present GloSplat, a framework that performs \emph{joint pose-appearance optimization} during 3D Gaussian Splatting training. Unlike prior joint optimization methods (BARF, NeRF--, 3RGS) that rely purely on photometric gradients for pose refinement, GloSplat preserves \emph{explicit SfM feature tracks} as first-class entities throughout training: track 3D points are maintained as separate optimizable parameters from Gaussian primitives, providing persistent geometric anchors via a reprojection loss that operates alongside photometric supervision. This architectural choice prevents early-stage pose drift while enabling fine-grained refinement -- a capability absent in photometric-only approaches. We introduce two pipeline variants: (1) \textbf{GloSplat-F}, a COLMAP-free variant using retrieval-based pair selection for efficient reconstruction, and (2) \textbf{GloSplat-A}, an exhaustive matching variant for maximum quality. Both employ global SfM initialization followed by joint photometric-geometric optimization during 3DGS training. Experiments demonstrate that GloSplat-F achieves state-of-the-art among COLMAP-free methods while GloSplat-A surpasses all COLMAP-based baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.24290v2">UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 ICLR 2026, Project page: https://ufo-4d.github.io/
    </div>
    <details class="paper-abstract">
      Dense 4D reconstruction from unposed images remains a critical challenge, with current methods relying on slow test-time optimization or fragmented, task-specific feedforward models. We introduce UFO-4D, a unified feedforward framework to reconstruct a dense, explicit 4D representation from just a pair of unposed images. UFO-4D directly estimates dynamic 3D Gaussian Splats, enabling the joint and consistent estimation of 3D geometry, 3D motion, and camera pose in a feedforward manner. Our core insight is that differentiably rendering multiple signals from a single Dynamic 3D Gaussian representation offers major training advantages. This approach enables a self-supervised image synthesis loss while tightly coupling appearance, depth, and motion. Since all modalities share the same geometric primitives, supervising one inherently regularizes and improves the others. This synergy overcomes data scarcity, allowing UFO-4D to outperform prior work by up to 3 times in joint geometry, motion, and camera pose estimation. Our representation also enables high-fidelity 4D interpolation across novel views and time. Please visit our project page for visual results: https://ufo-4d.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04770v1">DSA-SRGS: Super-Resolution Gaussian Splatting for Dynamic Sparse-View DSA Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 11 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Digital subtraction angiography (DSA) is a key imaging technique for the auxiliary diagnosis and treatment of cerebrovascular diseases. Recent advancements in gaussian splatting and dynamic neural representations have enabled robust 3D vessel reconstruction from sparse dynamic inputs. However, these methods are fundamentally constrained by the resolution of input projections, where performing naive upsampling to enhance rendering resolution inevitably results in severe blurring and aliasing artifacts. Such lack of super-resolution capability prevents the reconstructed 4D models from recovering fine-grained vascular details and intricate branching structures, which restricts their application in precision diagnosis and treatment. To solve this problem, this paper proposes DSA-SRGS, the first super-resolution gaussian splatting framework for dynamic sparse-view DSA reconstruction. Specifically, we introduce a Multi-Fidelity Texture Learning Module that integrates high-quality priors from a fine-tuned DSA-specific super-resolution model, into the 4D reconstruction optimization. To mitigate potential hallucination artifacts from pseudo-labels, this module employs a Confidence-Aware Strategy to adaptively weight supervision signals between the original low-resolution projections and the generated high-resolution pseudo-labels. Furthermore, we develop Radiative Sub-Pixel Densification, an adaptive strategy that leverages gradient accumulation from high-resolution sub-pixel sampling to refine the 4D radiative gaussian kernels. Extensive experiments on two clinical DSA datasets demonstrate that DSA-SRGS significantly outperforms state-of-the-art methods in both quantitative metrics and qualitative visual fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18140v2">Observer-Actor: Active Vision Imitation Learning with Sparse-View Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 Accepted at ICRA 2026. Project Webpage: https://obact.github.io
    </div>
    <details class="paper-abstract">
      We propose Observer Actor (ObAct), a novel framework for active vision imitation learning in which the observer moves to optimal visual observations for the actor. We study ObAct on a dual-arm robotic system equipped with wrist-mounted cameras. At test time, ObAct dynamically assigns observer and actor roles: the observer arm constructs a 3D Gaussian Splatting (3DGS) representation from three images, virtually explores this to find an optimal camera pose, then moves to this pose; the actor arm then executes a policy using the observer's observations. This formulation enhances the clarity and visibility of both the object and the gripper in the policy's observations. As a result, we enable the training of ambidextrous policies on observations that remain closer to the occlusion-free training distribution, leading to more robust policies. We study this formulation with two existing imitation learning methods -- trajectory transfer and behavior cloning -- and experiments show that ObAct significantly outperforms static-camera setups: trajectory transfer improves by 145% without occlusion and 233% with occlusion, while behavior cloning improves by 75% and 143%, respectively. Videos are available at https://obact.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02887v2">Generalized non-exponential Gaussian splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 13 pages, 6 figures, 4 tables
    </div>
    <details class="paper-abstract">
      In this work we generalize 3D Gaussian splatting (3DGS) to a wider family of physically-based alpha-blending operators. 3DGS has become the standard de-facto for radiance field rendering and reconstruction, given its flexibility and efficiency. At its core, it is based on alpha-blending sorted semitransparent primitives, which in the limit converges to the classic radiative transfer function with exponential transmittance. Inspired by recent research on non-exponential radiative transfer, we generalize the image formation model of 3DGS to non-exponential regimes. Based on this generalization, we use a quadratic transmittance to define sub-linear, linear, and super-linear versions of 3DGS, which exhibit faster-than-exponential decay. We demonstrate that these new non-exponential variants achieve similar quality than the original 3DGS but significantly reduce the number of overdraws, which result on speed-ups of up to $4\times$ in complex real-world captures, on a ray-tracing-based renderer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02134v2">OnlineX: Unified Online 3D Reconstruction and Understanding with Active-to-Stable State Evolution</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 Accepted by CVPR Finding 2026 (Project page: https://xiac20.github.io/OnlineX/)
    </div>
    <details class="paper-abstract">
      Recent advances in generalizable 3D Gaussian Splatting (3DGS) have enabled rapid 3D scene reconstruction within seconds, eliminating the need for per-scene optimization. However, existing methods primarily follow an offline reconstruction paradigm, lacking the capacity for continuous reconstruction, which limits their applicability to online scenarios such as robotics and VR/AR. In this paper, we introduce OnlineX, a feed-forward framework that reconstructs both 3D visual appearance and language fields in an online manner using only streaming images. A key challenge in online formulation is the cumulative drift issue, which is rooted in the fundamental conflict between two opposing roles of the memory state: an active role that constantly refreshes to capture high-frequency local geometry, and a stable role that conservatively accumulates and preserves the long-term global structure. To address this, we introduce a decoupled active-to-stable state evolution paradigm. Our framework decouples the memory state into a dedicated active state and a persistent stable state, and then cohesively fuses the information from the former into the latter to achieve both fidelity and stability. Moreover, we jointly model visual appearance and language fields and incorporate an implicit Gaussian fusion module to enhance reconstruction quality. Experiments on mainstream datasets demonstrate that our method consistently outperforms prior work in novel view synthesis and semantic understanding, showcasing robust performance across input sequences of varying lengths with real-time inference speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02986v1">VIRGi: View-dependent Instant Recoloring of 3D Gaussians Splats</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 IEEE Transactions on Pattern Analysis and Machine Intelligence. 2026 Feb 24
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently transformed the fields of novel view synthesis and 3D reconstruction due to its ability to accurately model complex 3D scenes and its unprecedented rendering performance. However, a significant challenge persists: the absence of an efficient and photorealistic method for editing the appearance of the scene's content. In this paper we introduce VIRGi, a novel approach for rapidly editing the color of scenes modeled by 3DGS while preserving view-dependent effects such as specular highlights. Key to our method are a novel architecture that separates color into diffuse and view-dependent components, and a multi-view training strategy that integrates image patches from multiple viewpoints. Improving over the conventional single-view batch training, our 3DGS representation provides more accurate reconstruction and serves as a solid representation for the recoloring task. For 3DGS recoloring, we then introduce a rapid scheme requiring only one manually edited image of the scene from the end-user. By fine-tuning the weights of a single MLP, alongside a module for single-shot segmentation of the editable area, the color edits are seamlessly propagated to the entire scene in just two seconds, facilitating real-time interaction and providing control over the strength of the view-dependent effects. An exhaustive validation on diverse datasets demonstrates significant quantitative and qualitative advancements over competitors based on Neural Radiance Fields representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02893v1">Intrinsic Geometry-Appearance Consistency Optimization for Sparse-View Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      3D human reconstruction from a single image is a challenging problem and has been exclusively studied in the literature. Recently, some methods have resorted to diffusion models for guidance, optimizing a 3D representation via Score Distillation Sampling(SDS) or generating a back-view image for facilitating reconstruction. However, these methods tend to produce unsatisfactory artifacts (\textit{e.g.} flattened human structure or over-smoothing results caused by inconsistent priors from multiple views) and struggle with real-world generalization in the wild. In this work, we present \emph{MVD-HuGaS}, enabling free-view 3D human rendering from a single image via a multi-view human diffusion model. We first generate multi-view images from the single reference image with an enhanced multi-view diffusion model, which is well fine-tuned on high-quality 3D human datasets to incorporate 3D geometry priors and human structure priors. To infer accurate camera poses from the sparse generated multi-view images for reconstruction, an alignment module is introduced to facilitate joint optimization of 3D Gaussians and camera poses. Furthermore, we propose a depth-based Facial Distortion Mitigation module to refine the generated facial regions, thereby improving the overall fidelity of the reconstruction. Finally, leveraging the refined multi-view images, along with their accurate camera poses, MVD-HuGaS optimizes the 3D Gaussians of the target human for high-fidelity free-view renderings. Extensive experiments on Thuman2.0 and 2K2K datasets show that the proposed MVD-HuGaS achieves state-of-the-art performance on single-view 3D human rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02866v1">Multimodal-Prior-Guided Importance Sampling for Hierarchical Gaussian Splatting in Sparse-View Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      We present multimodal-prior-guided importance sampling as the central mechanism for hierarchical 3D Gaussian Splatting (3DGS) in sparse-view novel view synthesis. Our sampler fuses complementary cues { -- } photometric rendering residuals, semantic priors, and geometric priors { -- } to produce a robust, local recoverability estimate that directly drives where to inject fine Gaussians. Built around this sampling core, our framework comprises (1) a coarse-to-fine Gaussian representation that encodes global shape with a stable coarse layer and selectively adds fine primitives where the multimodal metric indicates recoverable detail; and (2) a geometric-aware sampling and retention policy that concentrates refinement on geometrically critical and complex regions while protecting newly added primitives in underconstrained areas from premature pruning. By prioritizing regions supported by consistent multimodal evidence rather than raw residuals alone, our method alleviates overfitting texture-induced errors and suppresses noise from pose/appearance inconsistencies. Experiments on diverse sparse-view benchmarks demonstrate state-of-the-art reconstructions, with up to +0.3 dB PSNR on DTU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02801v1">R3GW: Relightable 3D Gaussians for Outdoor Scenes in the Wild</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 Accepted at VISAPP 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has established itself as a leading technique for 3D reconstruction and novel view synthesis of static scenes, achieving outstanding rendering quality and fast training. However, the method does not explicitly model the scene illumination, making it unsuitable for relighting tasks. Furthermore, 3DGS struggles to reconstruct scenes captured in the wild by unconstrained photo collections featuring changing lighting conditions. In this paper, we present R3GW, a novel method that learns a relightable 3DGS representation of an outdoor scene captured in the wild. Our approach separates the scene into a relightable foreground and a non-reflective background (the sky), using two distinct sets of Gaussians. R3GW models view-dependent lighting effects in the foreground reflections by combining Physically Based Rendering with the 3DGS scene representation in a varying illumination setting. We evaluate our method quantitatively and qualitatively on the NeRF-OSR dataset, offering state-of-the-art performance and enhanced support for physically-based relighting of unconstrained scenes. Our method synthesizes photorealistic novel views under arbitrary illumination conditions. Additionally, our representation of the sky mitigates depth reconstruction artifacts, improving rendering quality at the sky-foreground boundary
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01099v2">HeroGS: Hierarchical Guidance for Robust 3D Gaussian Splatting under Sparse Views</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as a promising approach in novel view synthesis, combining photorealistic rendering with real-time efficiency. However, its success heavily relies on dense camera coverage; under sparse-view conditions, insufficient supervision leads to irregular Gaussian distributions, characterized by globally sparse coverage, blurred background, and distorted high-frequency areas. To address this, we propose HeroGS, Hierarchical Guidance for Robust 3D Gaussian Splatting, a unified framework that establishes hierarchical guidance across the image, feature, and parameter levels. At the image level, sparse supervision is converted into pseudo-dense guidance, globally regularizing the Gaussian distributions and forming a consistent foundation for subsequent optimization. Building upon this, Feature-Adaptive Densification and Pruning (FADP) at the feature level leverages low-level features to refine high-frequency details and adaptively densifies Gaussians in background regions. The optimized distributions then support Co-Pruned Geometry Consistency (CPG) at parameter level, which guides geometric consistency through parameter freezing and co-pruning, effectively removing inconsistent splats. The hierarchical guidance strategy effectively constrains and optimizes the overall Gaussian distributions, thereby enhancing both structural fidelity and rendering quality. Extensive experiments demonstrate that HeroGS achieves high-fidelity reconstructions and consistently surpasses state-of-the-art baselines under sparse-view conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24421v4">Proxy-GS: Unified Occlusion Priors for Training and Inference in Structured 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 Project page: https://visionary-laboratory.github.io/Proxy-GS
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as an efficient approach for achieving photorealistic rendering. Recent MLP-based variants further improve visual fidelity but introduce substantial decoding overhead during rendering. To alleviate computation cost, several pruning strategies and level-of-detail (LOD) techniques have been introduced, aiming to effectively reduce the number of Gaussian primitives in large-scale scenes. However, our analysis reveals that significant redundancy still remains due to the lack of occlusion awareness. In this work, we propose Proxy-GS, a novel pipeline that exploits a proxy to introduce Gaussian occlusion awareness from any view. At the core of our approach is a fast proxy system capable of producing precise occlusion depth maps at a resolution of 1000x1000 under 1ms. This proxy serves two roles: first, it guides the culling of anchors and Gaussians to accelerate rendering speed. Second, it guides the densification towards surfaces during training, avoiding inconsistencies in occluded regions, and improving the rendering quality. In heavily occluded scenarios, such as the MatrixCity Streets dataset, Proxy-GS not only equips MLP-based Gaussian splatting with stronger rendering capability but also achieves faster rendering speed. Specifically, it achieves more than 2.5x speedup over Octree-GS, and consistently delivers substantially higher rendering quality. Code will be public upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02548v1">SemGS: Feed-Forward Semantic 3D Gaussian Splatting from Sparse Views for Generalizable Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 ICRA 2026
    </div>
    <details class="paper-abstract">
      Semantic understanding of 3D scenes is essential for robots to operate effectively and safely in complex environments. Existing methods for semantic scene reconstruction and semantic-aware novel view synthesis often rely on dense multi-view inputs and require scene-specific optimization, limiting their practicality and scalability in real-world applications. To address these challenges, we propose SemGS, a feed-forward framework for reconstructing generalizable semantic fields from sparse image inputs. SemGS uses a dual-branch architecture to extract color and semantic features, where the two branches share shallow CNN layers, allowing semantic reasoning to leverage textural and structural cues in color appearance. We also incorporate a camera-aware attention mechanism into the feature extractor to explicitly model geometric relationships between camera viewpoints. The extracted features are decoded into dual-Gaussians that share geometric consistency while preserving branch-specific attributes, and further rasterized to synthesize semantic maps under novel viewpoints. Additionally, we introduce a regional smoothness loss to enhance semantic coherence. Experiments show that SemGS achieves state-of-the-art performance on benchmark datasets, while providing rapid inference and strong generalization capabilities across diverse synthetic and real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15151v2">Zero-shot CT Super-Resolution using Diffusion-based 2D Projection Priors and Signed 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Computed tomography (CT) is important in clinical diagnosis, but acquiring high-resolution (HR) CT is constrained by radiation exposure risks. While deep learning-based super-resolution (SR) methods have shown promise for reconstructing HR CT from low-resolution (LR) inputs, supervised approaches require paired datasets that are often unavailable. Zero-shot methods address this limitation by operating on single LR inputs; however, they frequently fail to recover fine structural details due to limited LR information within individual volumes. To overcome these limitations, we propose a novel zero-shot 3D CT SR framework that integrates diffusion-based upsampled 2D projection priors into the 3D reconstruction process. Specifically, our framework consists of two stages: (1) LR CT projection SR, training a diffusion model on abundant X-ray data to upsample LR projections, thereby enhancing the scarce information inherent in the LR inputs. (2) 3D CT volume reconstruction, using 3D Gaussian splatting with our novel Negative Alpha Blending (NAB-GS), which models positive and negative Gaussian densities to learn signed residuals between diffusion-generated HR and upsampled LR projections. Our framework demonstrates superior quantitative and qualitative performance on two public datasets, and expert evaluations present the framework's clinical potential at 4x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08862v2">StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted by ICLR 2026, Project page: https://streamsplat3d.github.io/
    </div>
    <details class="paper-abstract">
      Real-time reconstruction of dynamic 3D scenes from uncalibrated video streams demands robust online methods that recover scene dynamics from sparse observations under strict latency and memory constraints. Yet most dynamic reconstruction methods rely on hours of per-scene optimization under full-sequence access, limiting practical deployment. In this work, we introduce StreamSplat, a fully feed-forward framework that instantly transforms uncalibrated video streams of arbitrary length into dynamic 3D Gaussian Splatting (3DGS) representations in an online manner. It is achieved via three key technical innovations: 1) a probabilistic sampling mechanism that robustly predicts 3D Gaussians from uncalibrated inputs; 2) a bidirectional deformation field that yields reliable associations across frames and mitigates long-term error accumulation; 3) an adaptive Gaussian fusion operation that propagates persistent Gaussians while handling emerging and vanishing ones. Extensive experiments on standard dynamic and static benchmarks demonstrate that StreamSplat achieves state-of-the-art reconstruction quality and dynamic scene modeling. Uniquely, our method supports the online reconstruction of arbitrarily long video streams with a 1200x speedup over optimization-based methods. Our code and models are available at https://streamsplat3d.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20160v2">tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted by CVPR 2026. Project Page: https://cwchenwang.github.io/tttLRM
    </div>
    <details class="paper-abstract">
      We propose tttLRM, a novel large 3D reconstruction model that leverages a Test-Time Training (TTT) layer to enable long-context, autoregressive 3D reconstruction with linear computational complexity, further scaling the model's capability. Our framework efficiently compresses multiple image observations into the fast weights of the TTT layer, forming an implicit 3D representation in the latent space that can be decoded into various explicit formats, such as Gaussian Splats (GS) for downstream applications. The online learning variant of our model supports progressive 3D reconstruction and refinement from streaming observations. We demonstrate that pretraining on novel view synthesis tasks effectively transfers to explicit 3D modeling, resulting in improved reconstruction quality and faster convergence. Extensive experiments show that our method achieves superior performance in feedforward 3D Gaussian reconstruction compared to state-of-the-art approaches on both objects and scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02129v1">LiftAvatar: Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 19 pages, 11 figures
    </div>
    <details class="paper-abstract">
      We present LiftAvatar, a new paradigm that completes sparse monocular observations in kinematic space (e.g., facial expressions and head pose) and uses the completed signals to drive high-fidelity avatar animation. LiftAvatar is a fine-grained, expression-controllable large-scale video diffusion Transformer that synthesizes high-quality, temporally coherent expression sequences conditioned on single or multiple reference images. The key idea is to lift incomplete input data into a richer kinematic representation, thereby strengthening both reconstruction and animation in downstream 3D avatar pipelines. To this end, we introduce (i) a multi-granularity expression control scheme that combines shading maps with expression coefficients for precise and stable driving, and (ii) a multi-reference conditioning mechanism that aggregates complementary cues from multiple frames, enabling strong 3D consistency and controllability. As a plug-and-play enhancer, LiftAvatar directly addresses the limited expressiveness and reconstruction artifacts of 3D Gaussian Splatting-based avatars caused by sparse kinematic cues in everyday monocular videos. By expanding incomplete observations into diverse pose-expression variations, LiftAvatar also enables effective prior distillation from large-scale video generative models into 3D pipelines, leading to substantial gains. Extensive experiments show that LiftAvatar consistently boosts animation quality and quantitative metrics of state-of-the-art 3D avatar methods, especially under extreme, unseen expressions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01603v1">Sparse View Distractor-Free Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables efficient training and fast novel view synthesis in static environments. To address challenges posed by transient objects, distractor-free 3DGS methods have emerged and shown promising results when dense image captures are available. However, their performance degrades significantly under sparse input conditions. This limitation primarily stems from the reliance on the color residual heuristics to guide the training, which becomes unreliable with limited observations. In this work, we propose a framework to enhance distractor-free 3DGS under sparse-view conditions by incorporating rich prior information. Specifically, we first adopt the geometry foundation model VGGT to estimate camera parameters and generate a dense set of initial 3D points. Then, we harness the attention maps from VGGT for efficient and accurate semantic entity matching. Additionally, we utilize Vision-Language Models (VLMs) to further identify and preserve the large static regions in the scene. We also demonstrate how these priors can be seamlessly integrated into existing distractor-free 3DGS methods. Extensive experiments confirm the effectiveness and robustness of our approach in mitigating transient distractors for sparse-view 3DGS training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01491v1">Radiometrically Consistent Gaussian Surfels for Inverse Rendering</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 9 pages, 6 figures, ICLR 2026 Oral paper
    </div>
    <details class="paper-abstract">
      Inverse rendering with Gaussian Splatting has advanced rapidly, but accurately disentangling material properties from complex global illumination effects, particularly indirect illumination, remains a major challenge. Existing methods often query indirect radiance from Gaussian primitives pre-trained for novel-view synthesis. However, these pre-trained Gaussian primitives are supervised only towards limited training viewpoints, thus lack supervision for modeling indirect radiances from unobserved views. To address this issue, we introduce radiometric consistency, a novel physically-based constraint that provides supervision towards unobserved views by minimizing the residual between each Gaussian primitive's learned radiance and its physically-based rendered counterpart. Minimizing the residual for unobserved views establishes a self-correcting feedback loop that provides supervision from both physically-based rendering and novel-view synthesis, enabling accurate modeling of inter-reflection. We then propose Radiometrically Consistent Gaussian Surfels (RadioGS), an inverse rendering framework built upon our principle by efficiently integrating radiometric consistency by utilizing Gaussian surfels and 2D Gaussian ray tracing. We further propose a finetuning-based relighting strategy that adapts Gaussian surfel radiances to new illuminations within minutes, achieving low rendering cost (<10ms). Extensive experiments on existing inverse rendering benchmarks show that RadioGS outperforms existing Gaussian-based methods in inverse rendering, while retaining the computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01844v3">CloDS: Visual-Only Unsupervised Cloth Dynamics Learning in Unknown Conditions</a></div>
    <div class="paper-meta">
      📅 2026-03-01
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Deep learning has demonstrated remarkable capabilities in simulating complex dynamic systems. However, existing methods require known physical properties as supervision or inputs, limiting their applicability under unknown conditions. To explore this challenge, we introduce Cloth Dynamics Grounding (CDG), a novel scenario for unsupervised learning of cloth dynamics from multi-view visual observations. We further propose Cloth Dynamics Splatting (CloDS), an unsupervised dynamic learning framework designed for CDG. CloDS adopts a three-stage pipeline that first performs video-to-geometry grounding and then trains a dynamics model on the grounded meshes. To cope with large non-linear deformations and severe self-occlusions during grounding, we introduce a dual-position opacity modulation that supports bidirectional mapping between 2D observations and 3D geometry via mesh-based Gaussian splatting in video-to-geometry grounding stage. It jointly considers the absolute and relative position of Gaussian components. Comprehensive experimental evaluations demonstrate that CloDS effectively learns cloth dynamics from visual data while maintaining strong generalization capabilities for unseen configurations. Our code is available at https://github.com/whynot-zyl/CloDS. Visualization results are available at https://github.com/whynot-zyl/CloDS_video}.%\footnote{As in this example.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01158v1">FLICKER: A Fine-Grained Contribution-Aware Accelerator for Real-Time 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-01
      | 💬 Accepted at DATE 2026 (Design, Automation and Test in Europe Conference)
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has emerged as a mainstream rendering technique due to its photorealistic quality and low latency. However, processing massive numbers of non-contributing Gaussian points introduces significant computational overhead on resource-limited edge platforms, limiting its deployment in next-generation AR/VR devices. Contribution-based prior skipping alleviates this inefficiency, yet the resulting contribution-testing workload becomes prohibitive for edge execution. In this paper, we present FLICKER, a contribution-aware 3DGS accelerator based on hardware-software co-design. The proposed framework integrates adaptive leader pixels, pixel-rectangle grouping, hierarchical Gaussian testing, and a mixed-precision architecture to enable near pixel-level, contribution-driven rendering with minimal overhead. Experimental results demonstrate up to $1.5\times$ speedup, $2.6\times$ improvement in energy efficiency, and $14%$ area reduction compared with a state-of-the-art accelerator. Compared with a representative edge GPU, FLICKER achieves a $19.8\times$ speedup and $26.7\times$ higher energy efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01151v1">D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping</a></div>
    <div class="paper-meta">
      📅 2026-03-01
      | 💬 ICLR 2026 Poster
    </div>
    <details class="paper-abstract">
      Simulation provides a cost-effective and flexible platform for data generation and policy learning to develop robotic systems. However, bridging the gap between simulation and real-world dynamics remains a significant challenge, especially in physical parameter identification. In this work, we introduce a real-to-sim-to-real engine that leverages the Gaussian Splat representations to build a differentiable engine, enabling object mass identification from real-world visual observations and robot control signals, while enabling grasping policy learning simultaneously. Through optimizing the mass of the manipulated object, our method automatically builds high-fidelity and physically plausible digital twins. Additionally, we propose a novel approach to train force-aware grasping policies from limited data by transferring feasible human demonstrations into simulated robot demonstrations. Through comprehensive experiments, we demonstrate that our engine achieves accurate and robust performance in mass identification across various object geometries and mass values. Those optimized mass values facilitate force-aware policy learning, achieving superior and high performance in object grasping, effectively reducing the sim-to-real gap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.18041v7">Openfly: A comprehensive platform for aerial vision-language navigation</a></div>
    <div class="paper-meta">
      📅 2026-03-01
      | 💬 accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation (VLN) aims to guide agents by leveraging language instructions and visual cues, playing a pivotal role in embodied AI. Indoor VLN has been extensively studied, whereas outdoor aerial VLN remains underexplored. The potential reason is that outdoor aerial view encompasses vast areas, making data collection more challenging, which results in a lack of benchmarks. To address this problem, we propose OpenFly, a platform comprising various rendering engines, a versatile toolchain, and a large-scale benchmark for aerial VLN. Firstly, we integrate diverse rendering engines and advanced techniques for environment simulation, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of our environments. Secondly, we develop a highly automated toolchain for aerial VLN data collection, streamlining point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Thirdly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering diverse heights and lengths across 18 scenes. Moreover, we propose OpenFly-Agent, a keyframe-aware VLN model emphasizing key observations during flight. For benchmarking, extensive experiments and analyses are conducted, evaluating several recent VLN methods and showcasing the superiority of our OpenFly platform and agent. The toolchain, dataset, and codes will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18489v2">Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2026-03-01
      | 💬 This paper is accepted by ICLR 2026. Project page is available at https://liujf1226.github.io/Mono4DGS-HDR/
    </div>
    <details class="paper-abstract">
      We introduce Mono4DGS-HDR, the first system for reconstructing renderable 4D high dynamic range (HDR) scenes from unposed monocular low dynamic range (LDR) videos captured with alternating exposures. To tackle such a challenging problem, we present a unified framework with two-stage optimization approach based on Gaussian Splatting. The first stage learns a video HDR Gaussian representation in orthographic camera coordinate space, eliminating the need for camera poses and enabling robust initial HDR video reconstruction. The second stage transforms video Gaussians into world space and jointly refines the world Gaussians with camera poses. Furthermore, we propose a temporal luminance regularization strategy to enhance the temporal consistency of the HDR appearance. Since our task has not been studied before, we construct a new evaluation benchmark using publicly available datasets for HDR video reconstruction. Extensive experiments demonstrate that Mono4DGS-HDR significantly outperforms alternative solutions adapted from state-of-the-art methods in both rendering quality and speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.19754v2">FastAvatar: Towards Unified and Fast 3D Avatar Reconstruction with Large Gaussian Reconstruction Transformers</a></div>
    <div class="paper-meta">
      📅 2026-03-01
    </div>
    <details class="paper-abstract">
      Despite significant progress in 3D avatar reconstruction, it still faces challenges such as high time complexity, sensitivity to data quality, and low data utilization. We propose FastAvatar, a feedforward 3D avatar framework capable of flexibly leveraging diverse daily recordings (e.g., a single image, multi-view observations, or monocular video) to reconstruct a high-quality 3D Gaussian Splatting (3DGS) model within seconds, using only a single unified model. The core of FastAvatar is a Large Gaussian Reconstruction Transformer (LGRT) featuring three key designs: First, a 3DGS transformer aggregating multi-frame cues while injecting initial 3D prompt to predict the corresponding registered canonical 3DGS representations; Second, multi-granular guidance encoding (camera pose, expression coefficient, head pose) mitigating animation-induced misalignment for variable-length inputs; Third, incremental Gaussian aggregation via landmark tracking and sliced fusion losses. Integrating these features, FastAvatar enables incremental reconstruction, i.e., improving quality with more observations without wasting input data as in previous works. This yields a quality-speed-tunable paradigm for highly usable 3D avatar modeling. Extensive experiments show that FastAvatar has a higher quality and highly competitive speed compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00952v1">Decoupling Motion and Geometry in 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-01
    </div>
    <details class="paper-abstract">
      High-fidelity reconstruction of dynamic scenes is an important yet challenging problem. While recent 4D Gaussian Splatting (4DGS) has demonstrated the ability to model temporal dynamics, it couples Gaussian motion and geometric attributes within a single covariance formulation, which limits its expressiveness for complex motions and often leads to visual artifacts. To address this, we propose VeGaS, a novel velocity-based 4D Gaussian Splatting framework that decouples Gaussian motion and geometry. Specifically, we introduce a Galilean shearing matrix that explicitly incorporates time-varying velocity to flexibly model complex non-linear motions, while strictly isolating the effects of Gaussian motion from the geometry-related conditional Gaussian covariance. Furthermore, a Geometric Deformation Network is introduced to refine Gaussian shapes and orientations using spatio-temporal context and velocity cues, enhancing temporal geometric modeling. Extensive experiments on public datasets demonstrate that VeGaS achieves state-of-the-art performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21333v2">HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles</a></div>
    <div class="paper-meta">
      📅 2026-03-01
      | 💬 Accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      Controllable driving scene generation is critical for realistic and scalable autonomous driving simulation, yet existing approaches struggle to jointly achieve photorealism and precise control. We introduce HorizonForge, a unified framework that reconstructs scenes as editable Gaussian Splats and Meshes, enabling fine-grained 3D manipulation and language-driven vehicle insertion. Edits are rendered through a noise-aware video diffusion process that enforces spatial and temporal consistency, producing diverse scene variations in a single feed-forward pass without per-trajectory optimization. To standardize evaluation, we further propose HorizonSuite, a comprehensive benchmark spanning ego- and agent-level editing tasks such as trajectory modifications and object manipulation. Extensive experiments show that Gaussian-Mesh representation delivers substantially higher fidelity than alternative 3D representations, and that temporal priors from video diffusion are essential for coherent synthesis. Combining these findings, HorizonForge establishes a simple yet powerful paradigm for photorealistic, controllable driving simulation, achieving an 83.4% user-preference gain and a 25.19% FID improvement over the second best state-of-the-art method. Project page: https://horizonforge.github.io/ .
    </details>
</div>
