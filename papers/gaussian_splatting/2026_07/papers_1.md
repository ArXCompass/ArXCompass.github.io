# gaussian splatting - 2026_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.00214v1">Manifold-GS: Certified Hybrid Assets via Varifold-Conservative Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-31
      | 💬 9 pages, 2 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) gives high-quality novel-view synthesis, but its adaptive radiance primitives are not directly usable as structured assets: opacity is not an additive area measure, refinement can change the induced geometry, and watertight mesh extraction can hallucinate collision surfaces in unobserved regions. We introduce Manifold-GS, a certified hybrid asset layer for Gaussian scenes. The method separates appearance opacity from geometric quadrature mass, represents surface-like Gaussians as a discrete unoriented varifold, and exports only confidence-certified open surface patches while retaining uncertified content as residual splats. It provides refinement-conservative mass transport, local realizability diagnostics, source-preserving patch bindings, and conservative collision candidates. On three DTU scenes, a frozen asset benchmark shows zero patch-defined edit leakage, texture round-trip PSNR of 30.1/35.3/33.7 dB, and lower collision floater area than official 2DGS meshes on all scenes, with large gaps on two scenes. The result is a precision-coverage tradeoff rather than a universal reconstruction claim. External-region annotations, phantom-collision probes, and 5k-face simplification further support the certified asset interpretation, while RGB-only experiments show that local realizability is not sufficient for sparse-view surface identifiability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.29633v1">OASIS: Occlusion-aware Single-image Hand Avatar Reconstruction via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-31
      | 💬 Accepted to ACM Multimedia 2026. Project page: https://mova-hand.github.io/MOVA/. Code repository: https://github.com/ivyyy77/OASIS
    </div>
    <details class="paper-abstract">
      Single-image 3D hand avatar reconstruction is fundamentally ill-posed and particularly challenging due to limited visual evidence under severe self-occlusion and the complex pose-dependent deformation of highly articulated hands. Existing methods predominantly rely on implicit NeRF-style representations, whose volumetric fitting is computationally expensive and often struggles to preserve fine-grained hand details. In this work, we present OASIS, a tailored 3D Gaussian Splatting framework for single-image hand avatar reconstruction. To faithfully encode sparse image-specific appearance cues in single-view reconstruction, we construct geometry-aligned visual evidence tokens by explicitly aligning input image observations with 3D hand geometry and context-adaptively tokenizing the resulting visual evidence. Since severe self-occlusion makes the reliability of image evidence inherently visibility-dependent, we introduce a visibility-conditioned point-image attention to reliably transfer visual evidence to geometric tokens, yielding occlusion-aware Gaussian features for faithful and robust reconstruction. To further capture non-rigid deformation of articulated hands, we introduce a Feature-on-Mesh representation to enable Gaussian deformation to be guided by local surface stretching. Under this framework, we adopt a one-shot adaptation scheme that learns a shared hand prior from multi-identity training data and then fits it to a target image for target-specific reconstruction. Extensive experiments show that OASIS outperforms existing baselines in both visual fidelity and efficiency across challenging poses and in-the-wild scenarios, and further demonstrates strong versatility in downstream applications such as text-to-avatar generation and texture editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14990v2">JADE-GS: Joint Allocation of Deblurring Evidence for Event-Assisted 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-31
    </div>
    <details class="paper-abstract">
      Neural radiance fields and 3D Gaussian Splatting assume that each training image is a sharp and geometrically consistent observation of the scene. Motion blur violates this assumption because a single exposure integrates a continuous range of camera poses. Exposure integration also removes the temporal information needed to recover the corresponding sharp observation. Event cameras preserve this information at microsecond resolution and therefore provide a natural complement to conventional images. Existing event-assisted reconstruction methods predominantly obtain image supervision through analytical inversion of the Event Double Integral. Learned restoration from frames and events offers a second prior. Although weaker when used alone, it fails in different regions and provides complementary evidence. We present JADE-GS, which formulates the combination of these priors as spatial evidence allocation. A lightweight Spatial Prior Router predicts a pixelwise allocation using only the blurry frame and event stream, then fuses the two fixed restorations into an additional supervision target. The router is trained without a sharp reference using consistency with the scene under reconstruction and the measured exposure, and is removed after optimization. Experiments show that JADE-GS achieves leading perceptual quality on both benchmarks, attains the best fidelity on the real benchmark, and remains competitive on the synthetic one. It requires substantially lower training overhead than diffusion-based alternatives and preserves native 3DGS rendering with no generative decoding at inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.29284v1">FillGS: Filling Observation Gaps in 4D Gaussian Splatting via Viewpoint-Time Selection and Generative Refinement</a></div>
    <div class="paper-meta">
      📅 2026-07-31
      | 💬 Accepted at ECCV2026
    </div>
    <details class="paper-abstract">
      4D Gaussian Splatting (4DGS) can render dynamic scenes photorealistically. However, with limited viewpoint coverage, some spatiotemporal regions remain sparsely observed, leading to artifacts, particularly in scenes with large motion. Existing approaches leveraging generative models rely on heuristic virtual-viewpoint selection before refining rendered views. As a result, they cannot actively explore such sparsely observed regions. To address this issue, we propose a pipeline that actively selects spatiotemporal virtual viewpoints to improve 4DGS reconstruction. Our method selects virtual viewpoints for generative enhancement based on the rendering sensitivity and motion-aware observation density of 4D Gaussians, prioritizing views that alleviate observation sparsity. In the refined images, we filter out regions that conflict with captured observations or are likely to contain generative artifacts and then fine-tune 4DGS using only the reliable regions. We evaluate our method on multi-view video benchmarks using new train/test splits designed to induce observation gaps. Results show consistent improvements over prior viewpoint selection strategies and fine-tuning methods in both qualitative and quantitative evaluations, while reducing artifacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13682v2">Posterior Variance Is a Constraint Map, Not an Error Map: Closed-Form Uncertainty for Radiative Gaussian Splatting in Sparse-View CT</a></div>
    <div class="paper-meta">
      📅 2026-07-31
      | 💬 v2: substantially revised and condensed; 31 pages total, 9 figures
    </div>
    <details class="paper-abstract">
      Radiative Gaussian splatting reconstructs sparse-view CT fast and accurately, and recent work attaches per-Gaussian posteriors to yield per-voxel uncertainty maps. We ask what such a map actually measures: posterior variance is a data-constraint map, not an error map -- its alarms are trustworthy, its all-clears are not. Exploiting the strict linearity of X-ray rendering in the per-Gaussian densities, we derive a clamp-aware closed form that the unchanged rasterizer evaluates exactly in one forward pass, in volume and projection space: the infinite-sample limit of the sampling estimator of concurrent work, at ~8x lower cost. On the official 15-scene benchmark this uncertainty ranks true error on 14 of 15 scenes. Restricted to the object interior -- the tissue a clinician reads -- the ranking collapses (median Spearman 0.11, 0/15 pass), identically for a deep ensemble and for a strictly positive log-normal posterior: three constructions, two estimator families, no survivors. The mechanism is structural: about 90% of in-object error is bias that reproduces across retrainings, invisible to model disagreement; 73-81% of the full-volume correlation is carried by object/surround contrast; and an exactly solvable control puts the observed in-object ranking 4-5x below what a perfectly calibrated posterior with the same sigma-spread would score. The error scale, by contrast, is an engineering problem, and we solve it: reparameterizing the posterior contracts the cross-scene temperature spread from 19.3x to 2.6x, one scene-agnostic temperature transfers to unseen scenes (10/15 leave-one-scene-out), and the repaired scale tracks photon count at the Poisson-predicted -1/2 power. We distill evaluation practice that would have caught the illusion -- masked calibration, seed-wise bias decomposition, an exact-posterior reference -- and release all protocols, seeds and per-run evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01207v2">TRACE: High-Fidelity 3D Scene Editing via Tangible Reconstruction and Geometry-Aligned Contextual Video Masking</a></div>
    <div class="paper-meta">
      📅 2026-07-31
      | 💬 9 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Existing 3D Gaussian Splatting (3DGS) editing methods primarily focus on appearance modification and often struggle to support flexible geometry editing while preserving structural integrity and scene-consistent appearance. To address this limitation, we present TRACE, a mesh-guided 3DGS editing framework that automatically aligns explicit 3D geometry with Gaussian scenes and decouples Geometric Anchoring from Appearance Harmonization. First, Multi-view 3D-Anchor Synthesis, trained on our MV-TRACE dataset for scene-coherent object addition and modification, generates geometrically aligned editing anchors, while Tangible Geometry Alignment (TGA) performs coarse-to-fine mesh-scene registration. Then, Contextual Video Masking (CVM) integrates projected 3D anchors into an autoregressive video diffusion pipeline, harmonizing their appearance with the surrounding scene while maintaining multi-view consistency. We evaluate TRACE on eight held-out scenes across six editing categories. TRACE completes each edit in approximately 10 minutes on a single NVIDIA RTX Pro 6000 GPU. Extensive experiments demonstrate consistent improvements over existing methods in editing versatility, structural integrity, semantic alignment, multi-view consistency, and visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.00177v2">FieryGS: In-the-Wild Fire Synthesis with Physics-Integrated Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-31
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      We consider the problem of synthesizing photorealistic, physically plausible combustion effects in in-the-wild 3D scenes. Traditional CFD and graphics pipelines can produce realistic fire effects but rely on handcrafted geometry, expert-tuned parameters, and labor-intensive workflows, limiting their scalability to the real world. Recent scene modeling advances like 3D Gaussian Splatting (3DGS) enable high-fidelity real-world scene reconstruction, yet lack physical grounding for combustion. To bridge this gap, we propose FieryGS, a physically-based framework that integrates physically-accurate and user-controllable combustion simulation and rendering within the 3DGS pipeline, enabling realistic fire synthesis for real scenes. Our approach tightly couples three key modules: (1) multimodal large-language-model-based physical material reasoning, (2) efficient volumetric combustion simulation, and (3) a unified renderer for fire and 3DGS. By unifying reconstruction, physical reasoning, simulation, and rendering, FieryGS removes manual tuning and automatically generates realistic, controllable fire dynamics consistent with scene geometry and materials. Our framework supports complex combustion phenomena -- including flame propagation, smoke dispersion, and surface carbonization -- with precise user control over fire intensity, airflow, ignition location and other combustion parameters. Evaluated on diverse indoor and outdoor scenes, FieryGS outperforms all comparative baselines in visual realism, physical fidelity, and controllability. Project page can be found at https://pku-vcl-geometry.github.io/FieryGS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.28164v1">S-Avatar: Diffusion-Guided Gaussian Head Avatars from a Single Image</a></div>
    <div class="paper-meta">
      📅 2026-07-30
      | 💬 15 pages, 12 figures
    </div>
    <details class="paper-abstract">
      We propose S-Avatar, a novel method for generating photorealistic 3D head avatars from a single image using a diffusion-guided 3D model generation module and strategies for animating 3D Gaussian Splatting (3DGS). While single-image head avatar reconstruction is crucial for lifelike Virtual Reality (VR) applications, existing approaches often struggle to preserve 3D consistency under unseen viewpoints. S-Avatar addresses this limitation through a three-stage pipeline. First, a high-resolution 3DGS is synthesized directly from a single image using a diffusion-based Gaussian splat generation module. Next, the parametric head model FLAME is aligned with the generated 3DGS by optimizing its parameters and spatial transformations. Finally, to adapt the 3DGS to FLAME variations, we construct a binding template that encodes the spatial relationship between the initial splats and FLAME. The dynamic 3D head avatar can then be rendered in real time by deforming the 3DGS with the binding template. By combining diffusion-guided canonical 3DGS generation with FLAME-based control, our method achieves efficient and accurate reconstruction with enhanced 3D consistency. Evaluations on public datasets demonstrate that S-Avatar outperforms state-of-the-art methods in novel-view and expression generation, achieving superior realism and consistency. Consequently, our approach represents a significant advance in accessible avatar creation, applicable to a wide range of VR/AR applications. The project page is available at https://github.com/hailsong/savatar.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.28049v1">TSOG: A Format For Temporally And Spatially Ordered Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-07-30
      | 💬 2026 IEEE International Conference on Image Processing (ICIP). IEEE, 2026
    </div>
    <details class="paper-abstract">
      We propose Temporally and Spatially Ordered Gaussians (TSOG), a format for efficient representation of 4D Gaussian Splatting (4DGS) content. TSOG extends the Spatially Ordered Gaussians (SOG) framework to the temporal domain by introducing a timeline attribute and temporal parameterization of geometry and appearance attributes. Similar to SOG, TSOG is a lossy format that assigns each Gaussian a unique index and encodes attribute values as index-aligned image data. TSOG is model-agnostic, extensible, and compatible with both discrete and continuous 4DGS representations. Evaluation using a PLYs sequence and FreeTimeGS as baselines, serving as simplistic and state-of-the-art 4DGS representations respectively, shows file size reductions exceeding 90%, with PSNR differences ranging between -0.42 and +0.85 dB. These results demonstrate substantial file size savings with minimal quality degradation, enabling efficient representation, storage, and delivery of dynamic scenes for next-generation 4D content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.03659v7">DehazeGS: Seeing Through Fog with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-30
      | 💬 9 pages,5 figures. Accepted by AAAI2026. visualizations are available at https://jz-y-cn.github.io/DehazeGS/
    </div>
    <details class="paper-abstract">
      Current novel view synthesis methods are typically designed for high-quality and clean input images. However, in foggy scenes, scattering and attenuation can significantly degrade the quality of rendering. Although NeRF-based dehazing approaches have been developed, their reliance on deep fully connected neural networks and per-ray sampling strategies leads to high computational costs. Furthermore, NeRF's implicit representation limits its ability to recover fine-grained details from hazy scenes. To overcome these limitations, we propose learning an explicit Gaussian representation to explain the formation mechanism of foggy images through a physically forward rendering process. Our method, DehazeGS, reconstructs and renders fog-free scenes using only multi-view foggy images as input. Specifically, based on the atmospheric scattering model, we simulate the formation of fog by establishing the transmission function directly onto Gaussian primitives via depth-to-transmission mapping. During training, we jointly learn the atmospheric light and scattering coefficients while optimizing the Gaussian representation of foggy scenes. At inference time, we remove the effects of scattering and attenuation in Gaussian distributions and directly render the scene to obtain dehazed views. Experiments on both real-world and synthetic foggy datasets demonstrate that DehazeGS achieves state-of-the-art performance. visualizations are available at https://jz-y-cn.github.io/DehazeGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.28032v1">Split and Drive: Dual-Axis Disentanglement for Real-Time Gaussian Head Avatars</a></div>
    <div class="paper-meta">
      📅 2026-07-30
    </div>
    <details class="paper-abstract">
      Creating photorealistic animatable head avatars from a single image remains a fundamental challenge in digital human synthesis. While recent 3D Gaussian Splatting methods have achieved promising results, they rely on external tracking pipelines whose latency is excluded from inference measurements. Furthermore, they adopt unified representations that entangle geometrically distinct facial regions, limiting both expressiveness and rendering fidelity. We propose SpiD (Split and Drive), a single-image Gaussian head avatar framework built on two disentanglement axes. The compute axis internalizes per-frame driving, eliminating external tracking dependency at inference. The feature axis decomposes the avatar into three specialized Gaussian branches, each modeling a geometrically distinct facial domain. Extensive experiments demonstrate consistently strong performance against state-of-the-art methods while achieving the fastest inference speed among all compared methods on a single GPU with the complete driving pipeline included.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.27634v1">4DHumanDiff: Direct Text-to-4DGS Generation for Consistent 360-Degree Dynamic Humans</a></div>
    <div class="paper-meta">
      📅 2026-07-30
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Generating high-quality 360-degree dynamic human assets from text prompts is challenging. Existing methods usually synthesize monocular or multi-view videos first and then fit a 4D representation, which is expensive and often causes incomplete geometry or view-inconsistent renderings. We present 4DHumanDiff, a diffusion framework that directly generates dynamic humans represented by 4D Gaussian Splatting (4DGS) from text prompts. By modeling the structured 4D representation space end-to-end, 4DHumanDiff avoids video pre-generation and per-scene reconstruction, making it better suited for view-consistent and temporally coherent asset generation. The model uses a 3D U-Net backbone with temporal attention for motion-aware generation. We further construct a large-scale text-to-4DGS dataset with 60,000 high-quality pairs, and introduce 2D regularization and training-free 4D interpolation to improve rendering quality and motion smoothness. Experiments show that 4DHumanDiff generates consistent 360-degree dynamic humans within one minute, achieves better temporal and multi-view consistency, and reduces inference time by more than 10x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.28834v1">FocusGS: Spatial Delta Layers for Local Repair and Deterministic Editing of Trained 3D Gaussian Assets</a></div>
    <div class="paper-meta">
      📅 2026-07-30
      | 💬 8 pages, 6 figures, 5 tables. Ancillary demonstration video included
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is evolving from one-time reconstruction into deliverable, inspectable, and maintainable visual assets. Existing workflows focus on global reconstruction, training-time density control, or open-ended generative editing, leaving trained assets without precise local maintenance. We propose FocusGS, which unifies local repair and deterministic editing as composite spatial deltas. Repair is the purely additive special case: its base-manipulation term is empty, and it adds only local Gaussian bases; deterministic editing uses erase-insert factorization (EIF) to combine old-carrier erasure with new-content insertion. FocusGS addresses spatial gradient starvation: local repair raises target-region PSNR by 7.91 dB over 93 evaluation views. Across all 83 deterministic editing trials, the target ROI improves, with a trial-averaged mean edited ROI PSNR of 21.97 dB and a mean gain of +11.05 dB; across five public editing cases, FocusGS-EIF reaches 33.17 dB Target-mask PSNR and 0.994 Target-delta Correlation, while both text-driven baselines fail to complete the prescribed updates. FocusGS provides a lightweight, verifiable 3DGS maintenance operator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24353v2">Open-Vocabulary BEV Segmentation with 3D-Aware Geometric Constraints</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 This paper has been accepted to ECCV 2026
    </div>
    <details class="paper-abstract">
      Bird's-eye view (BEV) perception fuses multi-camera images into a unified top-down representation for autonomous driving. Despite recent progress, state-of-the-art methods remain confined to closed-set scenarios, making them vulnerable to unpredictable real-world environments. In this work, we introduce open-vocabulary BEV segmentation (OVBS), which leverages vision-language models (VLMs) to recognize categories beyond the training set while maintaining precise BEV perception and real-time efficiency. A key challenge in OVBS lies in the 3D geometric inconsistency inherent in the ill-posed lifting of 2D VLM semantics into BEV. To address this, we propose OVBEVSeg, a geometry-aware OVBS framework that enhances efficient Gaussian splatting (GS)-based unprojection by leveraging robust 3D geometric constraints across three progressive stages: (1) 2D-to-BEV pseudo-labeling via reliable 3D projection for OV generalization; (2) joint 2D-BEV per-scene optimization with BEV structural constraints for 3D geometric consistency; and (3) 3D geometric distillation for online efficiency. On the nuScenes dataset, OVBEVSeg achieves state-of-the-art performance, outperforming closed-set methods by 15.3 mIoU on unseen categories. Remarkably, even with no novel-class ground-truth labels, it remains competitive with self- and semi-supervised baselines trained with up to 40% of ground-truth annotations. Furthermore, it achieves 2.5x faster inference with only 0.22x the memory consumption of projection-based methods. Project page: https://hchoi256.github.io/projects/ovbevseg/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26889v1">StructureGS: Structure-aware Gaussian Splatting for Articulated Object Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 accepted at ECCV 2026
    </div>
    <details class="paper-abstract">
      Reconstructing articulated objects with multiple movable parts is essential for understanding object structure and enabling physical interaction. However, this reconstruction task poses significant challenges due to the entanglement of geometry, appearance, and motion parameters during optimization. Existing methods rely primarily on photometric supervision, which commonly fails to disentangle these interdependent components, resulting in poor part decomposition with blurred boundaries and geometric artifacts. To address this limitation, we introduce StructureGS, a reconstruction framework for articulated objects that integrates structure-aware guidance into 3D Gaussian Splatting. Our approach leverages oriented bounding boxes of object parts to enforce two key structural properties: spatial coherence, which constrains each part's geometry to remain compact and spatially coherent within its designated region, and structural connectivity, which enforces physically plausible contact relationships between adjacent parts. These properties are realized through structure-aware losses that inject explicit structural constraints into the optimization process. Extensive experiments demonstrate that our method achieves state-of-the-art performance in articulated object reconstruction, producing high-quality results with well-defined part geometries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26595v1">SpatialQ: Understanding 3D Gaussian Splatting Scene Quality via Visual-based MLLM</a></div>
    <div class="paper-meta">
      📅 2026-07-29
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as an effective representation for novel view synthesis and 3D scene reconstruction, creating an increasing demand for reliable quality assessment. Unlike conventional image quality assessment (IQA), the quality of a 3DGS scene depends not only on the perceptual fidelity of rendered views, but also on scene-level factors such as spatial structure and cross-view consistency. Existing IQA methods are limited by their reliance on 2D perceptual cues, whereas general multimodal large language models (MLLMs) are not designed for stable quality regression and may produce unreliable judgments. To address these limitations, a multimodal quality assessment framework is developed for 3DGS scene understanding. First, a 3D-aware quality representation learning framework is introduced by augmenting a VGGT-based encoder with a dedicated quality head. Multi-view images are encoded into view-specific features and aggregated to capture cross-view consistency, while geometric cues are incorporated through joint modeling of depth and point-cloud-related structural information, enabling the learning of structure-aware quality representations beyond appearance-driven features. Second, a grounded multimodal reasoning mechanism is constructed by jointly feeding original images, depth maps, point cloud renderings, and camera parameters into a Qwen-based MLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26578v1">3DGBGS: 3D Granular Ball Gaussian Splatting for Compact Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-07-29
    </div>
    <details class="paper-abstract">
      Three-dimensional Gaussian Splatting (3DGS) enables high-quality real-time novel-view synthesis through explicit Gaussian primitives and differentiable rasterization. 3DGS and Granular Ball Computing (GBC), proposed in 2019, share a natural compatibility in adaptive representation. The efficiency of 3DGS partly stems from a coarse-to-fine and on-demand refinement process that draws on the generation principle of GBC. This connection motivates us to further introduce adaptive granular ball organization into anchor-based 3DGS. Existing anchor-based methods typically construct anchors from sparse SfM point clouds through fixed voxelization, which cannot adequately adapt to spatially non-uniform point distributions and leads to a trade-off among anchor count, model compactness, and rendering quality. To address this issue, we propose 3DGBGS (3D Granular Ball Gaussian Splatting), a compact anchor-based framework for novel-view synthesis. 3DGBGS adaptively partitions SfM point clouds into 3D granular balls, using larger balls to compactly represent smooth and redundant regions and smaller balls to preserve complex geometry and local details. Based on this representation, Granular Ball Anchor Initialization (GBAI) uses granular ball centers to initialize compact anchor positions, while the Granular Ball Scale Prior (GBSP) exploits granular ball radii to provide local scale priors for Gaussian generation. Experiments on four benchmarks show that 3DGBGS reduces initial and final anchors by 37.1% and 10.0%, respectively, and model storage by 9.8% on average, while maintaining comparable rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26525v1">AtlasLC: Fast Codec-Ready Compression of Object-Centric 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 Accepted to IEEE ISMAR 2026 (TVCG)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables photorealistic novel-view synthesis with real-time rendering, but deploying compressed object-centric 3DGS in XR requires more than image-space rate-distortion. In practical XR asset pipelines, reusable objects are repeatedly packaged, transmitted, decoded, and instantiated, making asset-preparation cost, codec compatibility, decoding latency, and preservation of depth and silhouette cues first-class concerns. Existing 3DGS compression methods are largely developed for scene-scale captures and often rely on heavy layout generation or aggressive global pruning, assumptions that transfer poorly to semantically concentrated foreground objects. We present AtlasLC, a source-free, training-free compression pipeline for object-centric 3DGS that operates directly on released Gaussian assets, without original images, camera poses, or per-asset optimization. AtlasLC couples local-competition pruning with deterministic atlas packing to remove the mapping/remapping bottleneck while preserving object-wide foreground support; a lightweight single-pass sort-based conditional transport is used as a shared coordinate backbone for these stages. Across the evaluated assets, AtlasLC reduces atlas-preparation time by up to a factor of 25 and end-to-end compression time by up to a factor of 5, while offering a favorable deployment-aware balance of payload, decode latency, runtime FPS, and 3D geometry relative to the evaluated compressed baselines. Relative to similarly compact structured baselines, it uses about 6 to 8 percent fewer bits while maintaining comparable perceptual and geometric quality. These results show that object-centric 3DGS compression should be optimized for a deployment-aware operating point enabling scalable XR asset libraries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.13287v4">InnerGS: Internal Scenes Reconstruction and Segmentation via Factorized 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently gained popularity for efficient scene rendering by representing scenes as explicit sets of anisotropic 3D Gaussians. However, most existing work focuses primarily on modeling external surfaces. In this work, we target the reconstruction of internal scenes, which is crucial for applications that require a deep understanding of an object's interior. By directly modeling a continuous volumetric density through the inner 3D Gaussian distribution, our model effectively reconstructs smooth and detailed internal structures from sparse sliced data. Beyond high-fidelity reconstruction, we further demonstrate the framework's potential for downstream tasks such as segmentation. By integrating language features, we extend our approach to enable text-guided segmentation of medical scenes via natural language queries. Our approach eliminates the need for camera poses, is plug-and-play, and is inherently compatible with any data modalities. We provide cuda implementation at: https://github.com/Shuxin-Liang/InnerGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.18314v2">Diff2DGS: Reliable Reconstruction of Occluded Surgical Scenes via 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 This work has been accpeted by the IEEE Robotics and Automation Letters(RA-L)
    </div>
    <details class="paper-abstract">
      Real-time reconstruction of deformable surgical scenes is vital for advancing robotic surgery, improving intraoperative guidance, and enabling automation. Recent methods achieve dense reconstructions from da Vinci robotic surgery videos, with Gaussian Splatting (GS) offering real-time performance via graphics acceleration. However, reconstruction quality in occluded regions remains limited, and depth accuracy has not been fully assessed, as benchmarks like EndoNeRF and StereoMIS lack 3D ground truth. We propose Diff2DGS, a two-stage framework for reliable 3D reconstruction of occluded surgical scenes. First, a diffusion-based video module with temporal priors inpaints tissue occluded by instruments with high spatiotemporal consistency. Second, we adapt 2D Gaussian Splatting (2DGS) with a Learnable Deformation Model (LDM) to capture dynamic tissue deformation and anatomical geometry, and introduce adaptive depth weight to improve geometric fidelity. We further extend evaluation beyond image-quality metrics by performing quantitative depth analysis on the SCARED dataset. Diff2DGS outperforms state-of-the-art methods in both appearance and geometry, reaching 38.02 dB PSNR on EndoNeRF and 34.40 dB on StereoMIS. Our experiments also show that optimizing image quality alone does not necessarily ensure accurate 3D geometry.The code is available at https://diff2dgs.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25569v1">CORF-GS: Real-Time Wireless Radiance Field Reconstruction via Coupled Optical-RF Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS)-based wireless radiance field (WRF) reconstruction provide an efficient solution for wireless channel modeling. However, existing WRF reconstruction methods rely on pre-collected observations and offline optimization, and thus struggle to provide real-time channel knowledge. To bridge this gap, we propose CORF-GS, a real-time WRF reconstruction framework that processes sequential optical and radio frequency (RF) keyframes. Specifically, CORF-GS constructs a unified Gaussian representation for optical and RF with shared geometry and modality-specific appearance, allowing high-resolution optical images to provide structural priors for WRF reconstruction. When a new keyframe arrives, CORF-GS first employs optical-guided Gaussian sampling to densify the WRF in under-represented regions. Since light and radio waves may respond differently to the same object surfaces due to wavelength mismatch, relying solely on optical guidance may neglect RF-informative areas. Therefore, CORF-GS performs coupled optical-RF optimization to jointly refine the shared Gaussians. Compared with the existing two-stage training pipelines, this prevents WRF from passively adapting to a frozen optical geometry and encourages the shared Gaussians to adapt to both optical structures and RF power distributions. Simulations show that CORF-GS achieves state-of-the-art RF spectrum synthesis quality and reduces the reconstruction time by $6.4\times$ compared with existing WRF methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25362v1">PanoLess: Environment Reconstruction from Partial Reflective Views</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 ECCV 2026. Main paper with supplementary material
    </div>
    <details class="paper-abstract">
      Reflections from shiny objects and glass facades naturally extend the field of view of a camera, capturing the surrounding environment without the need to pan the camera or acquire a full panorama. We propose PanoLess, a Gaussian-splat-based framework that reconstructs the surrounding environment as a distant illumination map from images captured on only one side of a reflective surface. PanoLess leverages surface-aligned 2D Gaussian splats with deferred shading to recover accurate per-pixel normals and reflection cues, which are fused into a neural cubemap representation of the environment. In addition, PanoLess produces a visibility map that explicitly denotes which regions of the environment are supported by the partial reflective observations. Unlike existing inverse-rendering and reflection-aware Gaussian-splatting approaches, which typically require full 360-degree coverage and struggle under incomplete views, PanoLess enables consistent, physically grounded illumination estimation from partial-view input. We show that PanoLess achieves high-fidelity and geometrically consistent environment reconstruction, outperforming reflection-aware baselines on a new custom synthetic benchmark and publicly available datasets, and demonstrating generalization to real-world reflective captures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15355v5">DAV-GSWT: Diffusion-Active-View Sampling for Data-Efficient Gaussian Splatting Wang Tiles</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 16 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The emergence of 3D Gaussian Splatting has fundamentally redefined the capabilities of photorealistic neural rendering by enabling high-throughput synthesis of complex environments. While procedural methods like Wang Tiles have recently been integrated to facilitate the generation of expansive landscapes, these systems typically remain constrained by a reliance on densely sampled exemplar reconstructions. We present DAV-GSWT, a data-efficient framework that leverages diffusion priors and active view sampling to synthesize high-fidelity Gaussian Splatting Wang Tiles from minimal input observations. By integrating a hierarchical uncertainty quantification mechanism with generative diffusion models, our approach autonomously identifies the most informative viewpoints while hallucinating missing structural details to ensure seamless tile transitions. Experimental results indicate that our system significantly reduces the required data volume while maintaining the visual integrity and interactive performance necessary for large-scale virtual environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.20417v2">ATSplat: Compact Feed-forward 3D Gaussian Splatting with Adaptive Token Expansion</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 Project page is at: https://join16.github.io/page-atsplat
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves high-quality novel-view synthesis by optimizing freely placed primitives in 3D and adaptively densifying them in under-reconstructed regions. However, this scene-adaptive capacity allocation is largely lost in existing feed-forward 3DGS methods, which commonly regress Gaussians at input pixels and lift them along camera rays. Such pixel-aligned formulations make the number and placement of primitives depend on image resolution and input viewpoints rather than scene complexity, resulting in dense and often redundant Gaussian sets. We present ATSplat, a feed-forward 3DGS framework that restores the adaptive allocation capability of 3DGS optimization through Adaptive 3D Tokens. ATSplat first lifts coarse patch-level depth and camera cues into sparse 3D anchor tokens, forming a compact scaffold of the scene. Each token is then regressed into local Gaussians with learnable 3D offsets, decoupling primitive placement from input image grids. An Adaptive Token Expansion module predicts a token-level uncertainty score, supervised by rendering error maps, and selectively expands high-uncertainty tokens through learnable expansion layers. This sparse-to-adaptive formulation enables ATSplat to concentrate primitives in challenging regions while maintaining a compact representation. Experiments on two representative datasets, RealEstate10K and DL3DV, show that ATSplat achieves state-of-the-art rendering quality while reducing the number of Gaussians by more than $5.7\times$ compared with dense feed-forward 3DGS methods. From 12 input images at $512 \times 960$ resolution, ATSplat completes reconstruction in less than a second using a single commercial GPU, and renders high-quality novel views at 1136 FPS ($512 \times 960$) with only 311K Gaussians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25219v1">SONG: A Photorealistic 3D Gaussian Simulation Platform for Benchmarking Social Navigation</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      Social navigation has progressed from simplified 2D environments toward a more general vision-based setting, in which a robot needs to achieve socially compliant behavior purely from onboard visual observations. Yet supporting simulation platforms have not kept pace: existing options either lack visual observations, lack moving human avatars, or fall short of real-world fidelity in appearance and pedestrian behavior, offering limited support for advancing vision-based social navigation. We introduce SONG, a SOcial Navigation platform powered by 3D Gaussian splatting (3DGS). It leverages 3DGS for both scene and avatar representations, drives pedestrians using semantically grounded trajectories generated by a large language model, and synthesizes their full-body motion with a trajectory-conditioned generator to produce continuous, natural movement. On top of the platform, we curate SONG-Bench, a set of evaluation episodes stratified by difficulty, and propose a multi-dimensional metric suite covering effectiveness, safety, and social compliance. A systematic evaluation of representative navigation baselines reveals three findings: (a) vision-based social navigation is far from solved; (b) a critical safety deficit precedes social etiquette; (c) real-world data matters more than model scale. Crucially, we demonstrate that fine-tuning on our curated data effectively improves the success rate in real-world environments. We hope our platform provides a faithful and rigorous testbed for the next generation of vision-based social navigation research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.12001v5">3D Gaussian Splatting against Moving Objects for High-Fidelity Street Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-07-27
      | 💬 arXiv admin note: This paper has been withdrawn by arXiv due to unverifiable authorship and affiliation
    </div>
    <details class="paper-abstract">
      The accurate reconstruction of dynamic street scenes is critical for applications in autonomous driving, augmented reality, and virtual reality. Traditional methods relying on dense point clouds and triangular meshes struggle with moving objects, occlusions, and real-time processing constraints, limiting their effectiveness in complex urban environments. While multi-view stereo and neural radiance fields have advanced 3D reconstruction, they face challenges in computational efficiency and handling scene dynamics. This paper proposes a novel 3D Gaussian point distribution method for dynamic street scene reconstruction. Our approach introduces an adaptive transparency mechanism that eliminates moving objects while preserving high-fidelity static scene details. Additionally, iterative refinement of Gaussian point distribution enhances geometric accuracy and texture representation. We integrate directional encoding with spatial position optimization to optimize storage and rendering efficiency, reducing redundancy while maintaining scene integrity. Experimental results demonstrate that our method achieves high reconstruction quality, improved rendering performance, and adaptability in large-scale dynamic environments. These contributions establish a robust framework for real-time, high-precision 3D reconstruction, advancing the practicality of dynamic scene modeling across multiple applications. The source code for this work is available to the public at https://github.com/okic-ca/3dgs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10760v2">MAGS-SLAM: Monocular Multi-Agent Gaussian Splatting SLAM for Geometrically and Photometrically Consistent Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Collaborative photorealistic 3D reconstruction from multiple agents enables rapid large-scale scene capture for virtual production and cooperative multi-robot exploration. While recent 3D Gaussian Splatting (3DGS) SLAM algorithms can generate high-fidelity real-time mapping, most of the existing multi-agent Gaussian SLAM methods still rely on RGB-D sensors to obtain metric depth and simplify cross-agent alignment, limiting their deployment on low-cost or power-constrained robotic platforms, especially given the wider availability of RGB cameras. To address this challenge, we propose MAGS-SLAM, the first RGB-only multi-agent 3DGS SLAM framework for collaborative scene reconstruction. Each agent independently builds local monocular Gaussian submaps and transmits compact submap summaries rather than raw observations or dense maps. To facilitate robust collaboration in the presence of monocular scale ambiguity, our framework integrates compact submap communication, geometry- and appearance-aware loop verification, and occupancy-aware Gaussian fusion, enabling coherent global reconstruction without active depth sensors. We further introduce ReplicaMultiagent Plus, a benchmark containing larger robot teams for evaluating collaborative Gaussian SLAM. Extensive experiments on synthetic and real-world datasets show that MAGS-SLAM achieves tracking accuracy and rendering quality competitive with or superior to those of state-of-the-art RGB-D collaborative Gaussian SLAM methods using RGB images alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24403v1">GenSplatCodec: Feed-Forward Gaussian Splatting Compression via One-Step Diffusion</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) enables scalable scene reconstruction without per-scene optimization, yet produces dense Gaussians that are costly to store and transmit. Existing feed-forward Gaussian compression methods formulate decoding as deterministic representation recovery, which becomes inadequate at low bitrates when high-frequency textures and view-dependent appearance are discarded. Although generative models offer a promising alternative, using them as standalone post-processing decouples generation from the transmitted scene structure, thereby compromising cross-view consistency. To address these limitations, we propose GenSplatCodec, a unified feed-forward Gaussian codec that reformulates low-bitrate Gaussian compression as geometry-guided generative decoding. We present a detail-aware feed-forward Gaussian coding scheme within a dual-stream formulation, where the resulting compact Gaussian structural stream is complemented by a lightweight reference appearance stream. We further introduce a geometry-guided one-step generative decoding approach that jointly exploits decoded structural and appearance cues through hierarchical geometry control to reconstruct high-fidelity and view-consistent novel views. Finally, we develop a three-stage optimization strategy that stabilizes the learning of the unified codec and adapts the generative decoder to codec-derived structural and appearance cues. Extensive experiments across multiple datasets demonstrate that GenSplatCodec consistently achieves superior rate-distortion (RD) performance over existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.23861v1">Head Avatars with Dynamic Explicit Hair</a></div>
    <div class="paper-meta">
      📅 2026-07-26
      | 💬 Project page: https://dynhair.is.tue.mpg.de/
    </div>
    <details class="paper-abstract">
      We present DynHair, a novel method for tracking and modeling dynamic hair for human head avatars. From video input, we reconstruct a dynamic head avatar with an explicit strand-based hair representation using structured 3D Gaussian Splatting. In contrast to the face region of human head avatars, which can be modeled with 3D Gaussians that are attached or generated with respect to some expressive 3D head model, hair is particularly challenging as it exhibits dynamic motion effects. Therefore, we present a novel method that models the dynamic deformations of the hair strands using a temporal network that is conditioned on angular velocity and acceleration of the head, as well as relative gravity. Specifically, an LSTM encodes the motion history and modulates per-point strand features via FiLM conditioning which further used by MLP to produce physically plausible displacements to canonical hairstyle. We jointly optimize this motion and appearance representation of the hair, with a 3DGS-based representation of the face-region, via differentiable Gaussian splatting with photometric, geometric, and physics-based supervision. As a result of our method, we retrieve hair tracking of the training video data and an animatable head avatar with controllable hair dynamics. In our experiments, we demonstrate state-of-the-art performance in terms of hair dynamics, temporal consistency, and generalization across subjects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.14736v3">HandSCS: Structural Coordinate Space for Animatable Hand Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-26
      | 💬 Accepted to ECCV 2026
    </div>
    <details class="paper-abstract">
      Photorealistic and animatable hand avatars are essential for applications such as AR/VR, gaming, and telepresence. Recent 3D Gaussian Splatting (3DGS) based avatar methods enable real-time rendering of articulated humans, but modeling hands remains challenging due to their compact structure, frequent self-occlusions, and complex finger interactions. Existing approaches primarily rely on pose-driven transformations while representing Gaussians in Euclidean space, lacking an explicit structural association with the underlying skeleton, which makes preserving fine-grained hand structures under complex articulations difficult. In this work, we introduce the Structural Coordinate Space (SCS), a skeleton-relative representation that assigns each Gaussian primitive an explicit structural coordinate with respect to the articulated hand skeleton. SCS is constructed using a hybrid static-virtual bone basis together with a distance-rotation structural descriptor that encodes the geometric relationship between Gaussians and bones. Based on SCS, we enforce both intra-pose and cross-pose structural consistency by combining per-Gaussian residual embeddings for local appearance modeling with structural correspondence across poses. Experiments demonstrate that our approach significantly improves structural consistency and preserves fine geometric details under challenging hand articulations compared with existing 3DGS-based avatar methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.13335v4">Deblur-Avatar: Animatable Avatars from Motion-Blurred Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2026-07-26
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for modeling high-fidelity, animatable 3D human avatars from motion-blurred monocular video inputs. Motion blur is prevalent in real-world dynamic video capture, especially due to human movements in 3D human avatar modeling. Existing methods either (1) assume sharp image inputs, failing to address the detail loss introduced by motion blur, or (2) mainly consider blur by camera movements, neglecting the human motion blur which is more common in animatable avatars. Our proposed approach integrates a human movement-based motion blur model into 3D Gaussian Splatting (3DGS). By explicitly modeling human motion trajectories during exposure time, we jointly optimize the trajectories and 3D Gaussians to reconstruct sharp, high-quality human avatars. We employ a pose-dependent fusion mechanism to distinguish moving body regions, optimizing both blurred and sharp areas effectively. Extensive experiments on synthetic and real-world datasets demonstrate that our method significantly outperforms existing methods in rendering quality and quantitative metrics, producing sharp avatar reconstructions and enabling real-time rendering under challenging motion blur conditions. Code and models are available at https://github.com/xianrui-luo/deblur_avatar.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.16910v2">LAGS: Low-Altitude Gaussian Splatting with Groupwise Heterogeneous Graph Learning</a></div>
    <div class="paper-meta">
      📅 2026-07-25
      | 💬 6 pages, 12 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Low-altitude Gaussian splatting (LAGS) facilitates 3D scene reconstruction by aggregating aerial images from distributed drones. However, as LAGS prioritizes maximizing reconstruction quality over communication throughput, existing low-altitude resource allocation schemes become inefficient. This inefficiency stems from their failure to account for image diversity introduced by varying viewpoints. To fill this gap, we propose a groupwise heterogeneous graph neural network (GW-HGNN) for LAGS resource allocation. GW-HGNN explicitly models the non-uniform contribution of different image groups to the reconstruction process, thus automatically balancing data fidelity and transmission cost. The key insight of GW-HGNN is to transform LAGS losses and communication constraints into graph learning costs for dual-level message passing. Experiments on real-world LAGS datasets demonstrate that GW-HGNN significantly outperforms state-of-the-art benchmarks across key rendering metrics, including PSNR, SSIM, and LPIPS. Furthermore, GW-HGNN reduces computational latency by approximately 100x compared to the widely-used MOSEK solver, achieving millisecond-level inference suitable for real-time deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.23189v1">Fashion-3DLR: A Controllable 3D Garment Generation Using Pairwise Fashion Elements for Intelligent Design</a></div>
    <div class="paper-meta">
      📅 2026-07-25
    </div>
    <details class="paper-abstract">
      AI-generated content (AIGC) has made significant progress, with 2D generative models becoming ready-to-use tools for the digital fashion industry. However, 3D garment generation remains in its nascent stage, where in the realm of fashion, the semantic information of diverse design elements exhibits intricate coupling relationships in 3D representations, posing substantial challenges for generating diverse 3D garments. In this work, to handle the above problem, We introduce Fashion-3DLR, a novel 3D garment generation framework that utilizes diverse design elements to create high-quality, versatile 3D garment assets. Specifically, to bridge the semantic gaps between different fashion elements, we propose a Garment Feature Fusion Diffusion Transformer (GFF-DiT) module to integrate 2D fashion design elements, e.g., sketch and texture, into latent space. Within the latent space, we then employ a rectified flow transformer to generate geometry latents, which can be decoded into various 3D garment representations, including 3D Gaussians and meshes. Furthermore, we integrate Fashion-3DLR into downstream tasks, achieving the 3D Gaussian Splatting (3DGS)-driven cloth physical simulation and mesh-based virtual try-on. Experimental results indicate that Fashion-3DLR surpass the previous state-of-the-art methods, which verify that the proposed work can generate well-structured, non-watertight garments capable of physical simulation and virtual try-on, underscoring its potential as a versatile 3D garment design tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.28828v2">Ground4D: Consistency-Aware 4D Reconstruction from Monocular Video</a></div>
    <div class="paper-meta">
      📅 2026-07-25
    </div>
    <details class="paper-abstract">
      Learning a 4D scene representation from a single monocular video that supports dynamic novel-view synthesis while maintaining faithful geometry over time remains challenging. Dynamic Gaussian Splatting achieves strong rendering performance through photometric optimization, yet does not explicitly enforce multi-view geometric consistency. In contrast, 3D foundation models recover coherent scene geometry and camera motion, but their point-based outputs are not designed for photorealistic rendering. We propose Ground4D, a geometry-grounded framework built on two stages. First, we perform geometry initialization via 3D foundation models, leveraging VGGT in a training-free manner to reconstruct multi-view-consistent 3D geometry and camera poses from monocular video. The recovered geometry provides a structured and reliable initialization for dynamic Gaussian representations. Second, we conduct geometry-consistency-aware refinement via dynamic Gaussian Splatting, optimizing the representation through differentiable rendering while maintaining multi-view geometric consistency across both observed and synthesized viewpoints. Furthermore, Ground4D inherently models the continuous 4D dynamics of the scene, naturally supporting rendering at arbitrary timestamps. By integrating foundation-level geometric priors into dynamic Gaussian optimization, Ground4D achieves stronger reconstruction fidelity and rendering performance, underscoring the role of geometry-grounded constraints in robust 4D scene modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.22997v1">Real2Sim2Real for Vision-Language-Action Manipulation: An AMD ROCm-Based Pipeline</a></div>
    <div class="paper-meta">
      📅 2026-07-25
    </div>
    <details class="paper-abstract">
      Physical AI -- the integration of large vision-language-action (VLA) models with embodied agents that act in the real world -- has emerged as the next major frontier for AI, echoed by industry leaders such as Jensen Huang (``the next big thing is Physical AI, AI with a body,'' GTC Paris, June 2025) and Dr. Lisa Su (`we're entering the world of Physical AI ... this is where AI enters the real world,' CES 2026). This paper presents an end-to-end, fully AMD-accelerated technology stack for embodied manipulation, spanning data-center training silicon, Radeon PRO simulation/rendering GPUs, and Ryzen AI edge compute, unified by the open ROCm software stack. We demonstrate that training and deploying VLA-based manipulation policies does not require a CUDA-locked ecosystem. Four progressive demonstrations are presented: (1) a Sim-to-Real manipulation pipeline trained with SmolVLA and deployed on a physical Franka arm; (2) a semantic, language-grounded object-selection task (`one-of-three'); (3) a Real2Sim synthetic-data generation pipeline that fuses 3D Gaussian Splatting (3DGS) reconstructions of real scenes with the Genesis physics engine; and (4) large-scale reinforcement learning for quadruped and humanoid locomotion benchmarked across multiple hardware platforms. All pipelines run natively on ROCm + PyTorch on RDNA4 (Radeon AI PRO R9700) and RDNA3.5 (Radeon PRO W7900) hardware and are reproducible on the free Radeon Cloud Platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.22956v1">3D Gaussian Splatting for Scientific Particle Data Compression and Rendering</a></div>
    <div class="paper-meta">
      📅 2026-07-24
      | 💬 13 pages, 13 figures. Accepted at SC26 (The International Conference for High Performance Computing, Networking, Storage and Analysis)
    </div>
    <details class="paper-abstract">
      Large-scale particle simulations produce hundreds of millions of particles, straining storage, transfer, and interactive visualization. Existing lossy compressors such as SZ3 operate in data space and provide no guarantees on downstream visualization fidelity. We propose ParticleGS, a visualization-aware framework based on 3D Gaussian Splatting (3DGS) that learns a compact representation directly optimized for rendered image quality, combining (1) a multi-stage, multi-orbit training pipeline, (2) VizMapper, a lightweight network that adapts a single trained model to user-specified visualization parameters at inference time, and (3) spatial block training with KD-tree decomposition and global fine-tuning. On a 281-million-particle HACC cosmological simulation, our 8-block model reaches 30.03 dB PSNR at 65x compression, outperforming SZ3 by 5-8 dB at comparable ratios, and generalizes without tuning to additional HACC regions and a dark-matter-only FIRE-2 simulation. It renders at 662 FPS on a single GPU, over 2,300x faster than ParaView on the full particle data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.22890v1">Meshless Domain Randomization via Explicit Parameter Perturbation of 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-24
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Domain Randomization (DR) is a standard technique for closing the Sim-to-Real gap, yet traditional DR pipelines rely on classical computer graphics rendering driven by polygon meshes. For complex organic subjects, such as insect specimens, extracting and rendering textured meshes is challenging. To address this issue, we propose a meshless DR framework that operates on the parameter space of 3D Gaussian Splatting (3DGS). Our method employs two independent perturbation pipelines to synthesize randomized training datasets. First, a Photometric DR pipeline alters the baked illumination and color balance by modulating the Spherical Harmonics (SH) coefficients. Second, a Procedural DR pipeline isolates the subject's geometric shape by replacing its original textures with 3D spatial noise. Finally, these perturbed radiance fields are composited over stochastically varied backgrounds using a rasterization engine. Our parameter manipulation provides a meshless alternative for generating robust datasets for complex geometries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.22780v1">Inter-Reflective Gaussian Splatting for Robust and Efficient Inverse Rendering</a></div>
    <div class="paper-meta">
      📅 2026-07-24
    </div>
    <details class="paper-abstract">
      Faithful inverse rendering requires visibility and indirect radiance to explain secondary illumination and inter-reflection, yet rasterization-oriented Gaussian representations do not naturally support the secondary-ray queries needed to recover them. We present IRGS++ (Inter-Reflective Gaussian Splatting), a unified robust and efficient Gaussian inverse rendering framework. During transport-aware optimization, IRGS++ employs differentiable 2D Gaussian ray tracing on surface-oriented Gaussian primitives to query visibility and indirect radiance on the fly and evaluate the full rendering equation for inter-reflective transport. This physical core makes Gaussian inverse rendering physically grounded beyond rasterized appearance modeling. To make this backbone useful beyond low-gloss dielectric scenes, the framework incorporates metallic-aware material modeling and robust reflective initialization for glossy, specular, and metallic materials. To make it practical, multiple importance sampling and denoising stabilize finite-sample rendering, while mesh-based secondary-attribute queries reduce the cost of relighting under novel illumination. Quantitative evaluations on low-gloss and glossy benchmarks show improved decomposition and relighting quality together with favorable quality--speed trade-offs under the reported configurations, while real-world studies illustrate plausible relighting under novel illumination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.22147v1">Visual Relocalization from Sparse Views in Aliased and Low-Texture Environments via Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-07-24
      | 💬 Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026)
    </div>
    <details class="paper-abstract">
      Visual localization becomes extremely challenging in planetary-like terrains characterized by low texture, perceptual aliasing, harsh illumination, and sparse, weakly overlapping viewpoints induced by forward rover motion and unconstrained driving directions. Under these conditions, state-of-the-art image-to-image and image-to-map matching pipelines suffer significant performance degradation. In this work, we propose a visual relocalization method that departs from classical correspondence-based pipelines by directly estimating camera poses against a differentiable map representation built with 3D Gaussian Splatting (3DGS). Our key contribution is a geometry-aware training strategy that combines photometric and geometric losses, where the geometric supervision is provided for the first time by combining multi-view stereo (MVS) and LiDAR depths. We show that this joint optimization produces a 3DGS model that better fits the underlying scene geometry, leading to improved photometric and geometric consistency and more robust, accurate single-image 6-DoF pose estimation. Extensive experiments on data acquired in planetary-analog environments validate the effectiveness of our approach, showing substantial gains in relocalization accuracy under challenging conditions. Code is available at https://github.com/DLR-RM/multimodal-gsplat-relocalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.21448v2">GrainGS: Gradient-Decoupled Gaussian Splatting for Efficient Dynamic Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-07-24
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction with 3D Gaussian Splatting requires a balance between fine-grained motion modeling, structural stability, and compact representation. Existing per-primitive methods provide flexible local deformation but often suffer from redundant primitive growth, while anchor-based methods improve spatial regularity at the cost of suppressing locally varying motion. To address these issues, we present GrainGS, a dynamic Gaussian framework that combines a hierarchical anchor scaffold with per-Gaussian deformation. A static warm-up stage first establishes a time-invariant canonical representation from observations across all timestamps. During joint training, a stop-gradient operation blocks the deformation-mediated gradient pathway to the canonical positions while preserving their direct refinement through the reconstruction objective. Each Gaussian then predicts independent temporal offsets for position, rotation, and scale, enabling detailed local motion within a structurally constrained scaffold. A canonical-residual appearance decomposition further models frame-dependent photometric changes without forcing them into geometric deformation. Experiments on synthetic monocular and real-world multiview benchmarks show that GrainGS achieves high reconstruction quality, real-time novel view synthesis, and compact storage. Under the synthetic benchmark setting, it reaches an average peak signal-to-noise ratio of 36.98 decibels, renders at 435.6 frames per second, and requires 4.67 megabytes of storage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10259v2">Real-Time Human Reconstruction and Animation using Feed-Forward Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-23
    </div>
    <details class="paper-abstract">
      We present HumanGS, a generalizable feed-forward Gaussian splatting framework for human 3D reconstruction and real-time animation from sparse multi-view RGB images and their associated SMPL-X poses. Unlike prior methods that rely on depth supervision, fixed input views, UV maps, repeated feed-forward inference for each target pose or view, or computationally expensive vertex-to-image cross-attention, HumanGS employs a simple transformer architecture that explicitly associates SMPL-X vertices with multi-scale image features through geometric back-projection. This eliminates the need for large pre-trained human representation models while naturally aggregating complementary information from multiple views. The aggregated vertex features are mapped by a lightweight MLP decoder to a canonical set of 3D Gaussian primitives aligned with SMPL-X vertices. One Gaussian is regularized to remain close to the SMPL-X surface, providing a strong geometric prior and stable correspondence to the parametric body model, while a small set of unconstrained Gaussians per vertex captures geometric details beyond the body surface, such as clothing and hair. The resulting canonical representation is animated efficiently using linear blend skinning and Gaussian rasterization without further network inference. Trained entirely from scratch on only 10K frames from THuman2.1, HumanGS achieves reconstruction quality comparable to or better than state-of-the-art methods while reducing reconstruction time by over 15 times compared to recent transformer-based approaches, enabling real-time animation and interactive applications. Code and pre-trained models are available at https://github.com/Devdoot57/HumanGS .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.21448v1">GrainGS: Gradient-Decoupled Gaussian Splatting for Efficient Dynamic Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-07-23
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction with 3D Gaussian Splatting requires a balance between fine-grained motion modeling, structural stability, and compact representation. Existing per-primitive methods provide flexible local deformation but often suffer from redundant primitive growth, while anchor-based methods improve spatial regularity at the cost of suppressing locally varying motion. To address these issues, we present GrainGS, a dynamic Gaussian framework that combines a hierarchical anchor scaffold with per-Gaussian deformation. A static warm-up stage first establishes a time-invariant canonical representation from observations across all timestamps. During joint training, a stop-gradient operation blocks the deformation-mediated gradient pathway to the canonical positions while preserving their direct refinement through the reconstruction objective. Each Gaussian then predicts independent temporal offsets for position, rotation, and scale, enabling detailed local motion within a structurally constrained scaffold. A canonical-residual appearance decomposition further models frame-dependent photometric changes without forcing them into geometric deformation. Experiments on synthetic monocular and real-world multiview benchmarks show that GrainGS achieves high reconstruction quality, real-time novel view synthesis, and compact storage. Under the synthetic benchmark setting, it reaches an average peak signal-to-noise ratio of 36.98 decibels, renders at 435.6 frames per second, and requires 4.67 megabytes of storage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.21416v1">GLAM-SLAM: Real-time Gaussian Large-scale Mapping via Flow Densification and Spatial Decomposition</a></div>
    <div class="paper-meta">
      📅 2026-07-23
      | 💬 Accepted to IROS 2026. Project page: https://glamslam.github.io/ Code: https://github.com/pmermigkas/GLAM-SLAM/
    </div>
    <details class="paper-abstract">
      Existing Gaussian-splatting-based monocular Simultaneous Localization and Mapping (SLAM) systems are either tailored to short sequences, are not real-time, or suffer from prohibitive GPU memory requirements, limiting their applicability in realistic, long-horizon scenarios. To address this, we present GLAM-SLAM, a real-time, decoupled Gaussian-splatting SLAM system designed for large-scale outdoor scenes. We ensure lightweight tracking using a robust, feature-based SLAM frontend, while for mapping, we adopt a structured, sparse anchor grid representation that ensures scalable operation and maintains scene coherence across long-term sequences. To satisfy the dense initialization requirements of 3D Gaussian Splatting (3DGS), we introduce a geometry-based flow-densification anchoring strategy using epipolar constraints. Furthermore, by treating mapping as a multi-scene problem, we propose a scene-partitioning strategy that introduces a strong spatial inductive bias via MLP initializations to generate localized Gaussians. We evaluate our system on the challenging, long-sequence KITTI Odometry, Oxford RobotCar, and M'alaga datasets. Extensive ablations and comparisons demonstrate a 15% improvement in reconstruction quality over the second-best performer, while maintaining real-time performance and the ability to scale to longer sequences. Code is publicly available for the benefit of the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.21099v1">Construction and Dynamic Update of Channel Gain Maps via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-23
    </div>
    <details class="paper-abstract">
      Channel knowledge maps (CKMs) have emerged as a promising technique for providing scene-specific and location-dependent propagation knowledge to enable environment-aware wireless network design. This paper investigates the construction and dynamic updating of a particular type of CKM, namely grid-based channel gain maps (CGMs), for large-scale networks using three-dimensional Gaussian splatting (3DGS). First, we formulate a grid-based channel gain model, where each map entry is defined as the locally averaged channel gain over a receiver grid, thereby suppressing phase-sensitive small-scale fluctuations. The resulting channel gain is decomposed into distance-dependent attenuation, path transmittance, and effective scattering contributions. Based on this decomposition, we develop a physics-informed Gaussian-splatting-based channel gain (GS-CG) model, which represents the propagation environment as a set of Gaussian primitives. The proposed model maps Gaussian geometry, opacity, and directional features to propagation-related factors and renders grid-level channel gains through a differentiable process. To accommodate real-time environmental changes, we further propose an incremental learning mechanism that updates a static reference GS-CG representation into a dynamic CGM. Specifically, the reference Gaussian primitives are frozen, while a compact set of tunable Gaussians is introduced to capture newly induced local channel-gain variations from sparse measurements.Numerical results demonstrate that the proposed GS-CG methods accurately reconstruct grid-based CGMs, efficiently adapt to dynamic environmental changes, and achieve a favorable accuracy-complexity tradeoff for fast CGM refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.03819v3">CGGS: Consistency-Augmented Geometric Gaussian Splatting for Ego-Centric 3D Scene Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-23
      | 💬 Update format
    </div>
    <details class="paper-abstract">
      Challenges remain in ego-centric 3D scene generation due to limited view overlap and the dominant influence of individual perspectives on scene interpretation. These factors hinder the creation of viewpoint-consistent and semantically aligned visual content, as well as the construction of accurate geometric structures. In this paper, we propose CGGS, a text-to-3D framework aiming to enhance 3D-content-awareness and address geometric distortions in ego-centric scene generation. Firstly, the Ego-centric Generator is proposed by fine-tuning a Multi-View Latent Diffusion Model with consistency-augmented loss to generate consistent, high-fidelity 2D content aligned with textual descriptions. Then, Layout Decorator leverages optical flow and point-track correspondence to estimate depth, therefore producing dense point clouds as coarse layouts from the ego-centric 2D priors. Building on this initialization, Geometric Refiner is proposed to enhance 3D Gaussian reconstruction via an entropy-based Mutual Information Depth Loss (MID) combined with a hierarchical optimization scheme for improving visual quality and geometric structure. Comprehensive experiments demonstrate that CGGS outperforms previous methods in generating coherent and accurate text-driven 3D scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06846v4">DynFOA: Generating First-Order Ambisonics with Conditional Diffusion for Dynamic and Acoustically Complex 360-Degree Videos</a></div>
    <div class="paper-meta">
      📅 2026-07-23
      | 💬 Revised version with updated content.This submission is the intended replacement of arXiv:2602.06846
    </div>
    <details class="paper-abstract">
      Spatial audio is crucial for immersive 360-degree video experiences, yet most 360-degree videos lack it due to the difficulty of capturing spatial audio during recording. Automatically generating spatial audio such as first-order ambisonics (FOA) from video therefore remains an important but challenging problem. In complex scenes, sound perception depends not only on sound source locations but also on scene geometry, materials, and dynamic interactions with the environment. However, existing approaches only rely on visual cues and fail to model dynamic sources and acoustic effects such as occlusion, reflections, and reverberation. To address these challenges, we propose DynFOA, a generative framework that synthesizes FOA from 360-degree videos by integrating dynamic scene reconstruction with conditional diffusion modeling. DynFOA analyzes the input video to detect and localize dynamic sound sources, estimate depth and semantics, and reconstruct scene geometry and materials using 3D Gaussian Splatting (3DGS). The reconstructed scene representation provides physically grounded features that capture acoustic interactions between sources, environment, and listener viewpoint. Conditioned on these features, a diffusion model generates spatial audio consistent with the scene dynamics and acoustic context. We introduce M2G-360, a dataset of 600 real-world clips divided into MoveSources, Multi-Source, and Geometry subsets for evaluating robustness under diverse conditions. Experiments show that DynFOA consistently outperforms existing methods in spatial accuracy, acoustic fidelity, distribution matching, and perceived immersive experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.20813v1">SubSplat: High-Resolution Pixel-aligned 3DGS via Sub-pixel Gaussian Reparameterization</a></div>
    <div class="paper-meta">
      📅 2026-07-23
    </div>
    <details class="paper-abstract">
      Pixel-aligned Gaussian splatting enables efficient and generalizable novel-view synthesis. However, high-resolution rendering faces a critical trade-off where increasing input resolution improves detail at the expense of quadratically rising network computational cost. Conversely, maintaining low-resolution inputs stabilizes this cost but results in insufficient Gaussian density and artifacts. To address this, we propose SubSplat, which introduces Sub-pixel Gaussian Reparameterizer(SPGR) to subdivide primary Gaussians into fine-grained primitives, restoring structural density directly from low-resolution features. We further enhance the reparameterization quality through feature aggregation, which effectively captures high-frequency details across multiple views. Experiments on RealEstate10K and ACID demonstrate that SubSplat achieves high-fidelity rendering with superior efficiency. Our results validate that the proposed framework successfully resolves the trade-off between reparameterization fidelity and network computational cost inherent in pixel-aligned Gaussian Splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.20789v1">3D-GIMP: When 3D Gaussian Inpainting Meets PatchMatch</a></div>
    <div class="paper-meta">
      📅 2026-07-22
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Recent advances in 3D scene editing have leveraged iterative diffusion models to update input views. However, this process is computationally expensive and struggles to produce sharp details. Meanwhile, ``hallucination drift'' frequently introduces multi-view inconsistencies, leading to structural artifacts when rendering novel viewpoints. To address this problem, we present 3D-GIMP (3D Gaussian Inpainting Meets Patch Matching), a novel hybrid paradigm designed for high-fidelity object removal in 3D Gaussian Splatting. Instead of diffusing every view, 3D-GIMP performs a single generative inpainting on a key reference view, which serves as an appearance prior. We then introduce a 3D-aware PatchMatch algorithm to propagate these reference textures across all remaining views via correspondence matching, effectively bypassing the stochastic nature of frame-by-frame diffusion. By prioritizing reconstructive consistency over iterative generation, 3D-GIMP maintains high-frequency details across arbitrary resolutions while ensuring a mathematically consistent 3D reconstruction. Our experiments demonstrate that 3D-GIMP not only achieves competitive inpainting quality as previous methods using diffusion in multiple views, but also outperforms these methods in rendering speed and view consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.20628v1">RealVDeblur: One-Step Diffusion for Generalizable Real-World Video Deblurring</a></div>
    <div class="paper-meta">
      📅 2026-07-22
      | 💬 Project page with code: https://rbjin.github.io/RealVDeblur/
    </div>
    <details class="paper-abstract">
      Real-world video deblurring remains challenging due to diverse motion patterns, complex degradations, and the scarcity of realistic training data, yet robust restoration is critical for downstream pipelines such as mobile imaging and 3D reconstruction. This work presents \textbf{RealVDeblur}, an efficient generative framework designed to improve in-the-wild robustness under diverse real capture conditions. First, a large-scale, physically grounded blur synthesis pipeline is constructed from scene-level 3D Gaussian Splatting (3DGS) assets and high-frame-rate videos, providing realistic training data covering both camera-induced and object-motion blur. Second, a video diffusion prior is leveraged for restoration; to better accommodate frame-dependent blur variations, temporal compression in the VAE is disabled and a frame-wise encoding scheme is adopted. For practical deployment on long videos, multi-step diffusion sampling is distilled into an efficient one-step generator, and a training-free Temporal Window Mask stabilizes inference beyond the training horizon with constant memory usage. Extensive experiments on diverse real-world benchmarks demonstrate strong perceptual quality, semantic fidelity, and temporal consistency on unseen videos, as well as improved robustness in downstream 3D reconstruction under severe motion blur. Project page: https://rbjin.github.io/RealVDeblur
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.20417v1">ATSplat: Compact Feed-forward 3D Gaussian Splatting with Adaptive Token Expansion</a></div>
    <div class="paper-meta">
      📅 2026-07-22
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves high-quality novel-view synthesis by optimizing freely placed primitives in 3D and adaptively densifying them in under-reconstructed regions. However, this scene-adaptive capacity allocation is largely lost in existing feed-forward 3DGS methods, which commonly regress Gaussians at input pixels and lift them along camera rays. Such pixel-aligned formulations make the number and placement of primitives depend on image resolution and input viewpoints rather than scene complexity, resulting in dense and often redundant Gaussian sets. We present ATSplat, a feed-forward 3DGS framework that restores the adaptive allocation capability of 3DGS optimization through Adaptive 3D Tokens. ATSplat first lifts coarse patch-level depth and camera cues into sparse 3D anchor tokens, forming a compact scaffold of the scene. Each token is then regressed into local Gaussians with learnable 3D offsets, decoupling primitive placement from input image grids. An Adaptive Token Expansion module predicts a token-level uncertainty score, supervised by rendering error maps, and selectively expands high-uncertainty tokens through learnable expansion layers. This sparse-to-adaptive formulation enables ATSplat to concentrate primitives in challenging regions while maintaining a compact representation. Experiments on two representative datasets, RealEstate10K and DL3DV, show that ATSplat achieves state-of-the-art rendering quality while reducing the number of Gaussians by more than $5.7\times$ compared with dense feed-forward 3DGS methods. From 12 input images at $512 \times 960$ resolution, ATSplat completes reconstruction in less than a second using a single commercial GPU, and renders high-quality novel views at 1136 FPS ($512 \times 960$) with only 311K Gaussians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.20325v1">MR-Compare: A Mixed-Reality Framework for Spatially Grounded Visual Comparison of 3D Gaussian Splatting and Mesh Reconstructions with the Physical Environment</a></div>
    <div class="paper-meta">
      📅 2026-07-22
      | 💬 27 pages, 12 figures. Includes the full appendix
    </div>
    <details class="paper-abstract">
      We introduce MR-Compare, a mixed reality framework for spatially grounded visual comparison between 3D Gaussian splatting and mesh reconstructions with live video see-through (VST). Implemented on a PC-tethered Meta Quest~3, it combines a two-stage registration pipeline with a 3D Slider for cross-media comparison. We evaluated five representative desktop and mobile reconstruction workflows through a real-world benchmark with an exploratory user study ($n=30$) in two static indoor rooms. MR-Compare achieved centimetre-level translation error across all workflows. The two desktop 3DGS workflows showed the strongest overall pattern, with 3DGS-MCMC yielding the lowest registration error and strongest VST-referenced visual consistency. Room-session measures indicated high perceived usability and low workload. We further propose an anisotropy filter, a zero-shot module that leverages Gaussian anisotropies to improve 3DGS registration in MR-Compare. A controlled Replica threshold sweep shows that moderate pruning can improve robustness and reduce residual errors. These results establish system-level feasibility in the tested setting rather than task-level effectiveness or standalone deployment. The project is available at https://github.com/changruizhu96/MR-Compare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.06210v2">VG3S: Visual Geometry Grounded Gaussian Splatting for Semantic Occupancy Prediction</a></div>
    <div class="paper-meta">
      📅 2026-07-22
      | 💬 Accepted by IROS 2026
    </div>
    <details class="paper-abstract">
      3D semantic occupancy prediction has become a crucial perception task for comprehensive scene understanding in autonomous driving. While recent advances have explored 3D Gaussian splatting for occupancy modeling to substantially reduce computational overhead, the generation of high-quality 3D Gaussians relies heavily on accurate geometric cues, which are often insufficient in purely vision-centric paradigms. To bridge this gap, we advocate for injecting the strong geometric grounding capability from Vision Foundation Models (VFMs) into occupancy prediction. In this regard, we introduce Visual Geometry Grounded Gaussian Splatting (VG3S), a novel framework that empowers Gaussian-based occupancy prediction with cross-view 3D geometric grounding. Specifically, to fully exploit the rich 3D geometric priors from a frozen VFM, we propose a plug-and-play hierarchical geometric feature adapter, which can effectively transform generic VFM tokens via feature aggregation, task-specific alignment, and multi-scale restructuring. Extensive experiments on the nuScenes occupancy benchmark demonstrate that VG3S achieves remarkable improvements of 12.6% in IoU and 7.5% in mIoU over the baseline. Furthermore, we show that VG3S generalizes seamlessly across diverse VFMs, consistently enhancing occupancy prediction accuracy and firmly underscoring the immense value of integrating priors derived from powerful, pre-trained geometry-grounded VFMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.19777v1">Look Before You Edit: Attention-Guided Camera Placement and Multi-View Alignment for 3D Gaussian Splatting Editing</a></div>
    <div class="paper-meta">
      📅 2026-07-22
    </div>
    <details class="paper-abstract">
      Text-driven 3D scene editing with 3D Gaussian Splatting (3DGS) typically applies a 2D diffusion editor to views rendered from fixed training cameras, limiting both the spatial coverage of edits and the user's freedom to target specific objects in complex scenes. We present LB-Edit, a framework that addresses two coupled problems: where to place editing cameras for localized edits, and how to make per-view edits agree with one another so that the 3D scene remains consistent after fine-tuning. First, Attention-Guided Editing Camera Placement (ACP) probes the diffusion model's self- and cross-attention at multiple candidate camera distances to find where attention is well-contained in the region of interest, then places a compact, geometrically diverse editing camera set at that attention-optimal distance. Second, Multi-view Attention Alignment (MAA) steers the editor toward the same edit across views along two axes: it aligns appearance by sharing self-attention features via token-level correspondence, and aligns spatial location by lifting cross-attention maps onto the 3D Gaussians as a shared 3D attention field, suppressing both appearance and spatial drift. Experiments on multi-object and single-object scenes show that our method achieves the highest user preference in instruction fidelity, multi-view consistency, and editing locality, using as few as 5 editing views and reducing latency by up to 7x over existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22069v3">TWINGS: Thin Plate Splines Warp-aligned Initialization for Sparse-View Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-22
      | 💬 Accepted at CVPR 2026, Project page: https://sandokim.github.io/twings/
    </div>
    <details class="paper-abstract">
      Novel view synthesis from sparse-view inputs poses a significant challenge in 3D computer vision, particularly for achieving high-quality scene reconstructions with limited viewpoints. We introduce TWINGS, a framework that enhances 3D Gaussian Splatting (3DGS) by directly addressing point sparsity. We employ Thin Plate Splines (TPS), a smooth non-rigid deformation model that minimizes bending energy to estimate a globally coherent warp from control-point correspondences, to align backprojected points from estimated depth with triangulated 3D control points, yielding calibrated backprojected points. By sampling these calibrated points near the control points, TWINGS provides a fast and geometrically accurate initialization for 3DGS, ultimately improving structural detail preservation and color fidelity in reconstructed scenes. Extensive experiments on DTU, LLFF, and Mip-NeRF360 demonstrate that TWINGS consistently outperforms existing methods, delivering detailed and accurate reconstructions under sparse-view scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15355v4">DAV-GSWT: Diffusion-Active-View Sampling for Data-Efficient Gaussian Splatting Wang Tiles</a></div>
    <div class="paper-meta">
      📅 2026-07-22
      | 💬 16 pages, 7 figures. In the previous version, Juntendo University was erroneously listed as the affiliation; we must clarify that this paper has absolutely no relation to Juntendo University. Therefore, we have replaced this affiliation in the new version
    </div>
    <details class="paper-abstract">
      The emergence of 3D Gaussian Splatting has fundamentally redefined the capabilities of photorealistic neural rendering by enabling high-throughput synthesis of complex environments. While procedural methods like Wang Tiles have recently been integrated to facilitate the generation of expansive landscapes, these systems typically remain constrained by a reliance on densely sampled exemplar reconstructions. We present DAV-GSWT, a data-efficient framework that leverages diffusion priors and active view sampling to synthesize high-fidelity Gaussian Splatting Wang Tiles from minimal input observations. By integrating a hierarchical uncertainty quantification mechanism with generative diffusion models, our approach autonomously identifies the most informative viewpoints while hallucinating missing structural details to ensure seamless tile transitions. Experimental results indicate that our system significantly reduces the required data volume while maintaining the visual integrity and interactive performance necessary for large-scale virtual environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.22676v6">TranSplat: Instant Object Relighting in Gaussian Splatting via Spherical Harmonic Radiance Transfer</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 ICCP 2026
    </div>
    <details class="paper-abstract">
      We present TranSplat, a method for instant, accurate object relighting within the Gaussian Splatting (GS) framework. Rather than relying on costly inverse rendering routines, we propose a BRDF-free radiance transfer strategy that analytically modulates the spherical harmonic (SH) appearance coefficients of an object's 2D Gaussian surfels using per-normal irradiance ratios derived from source and target environment maps. To handle view-dependent and glossy appearances without explicit material estimation, we introduce a specularity-aware dual-path SH transfer strategy that adapts higher-order SH bands in the reflection domain. Additionally, we propose a lightweight SH-domain self-shadowing module to ensure physically realistic occlusion without explicit mesh raycasting. Operating as a post-processing step, TranSplat requires no additional GS retraining for a pair of source and target scenes. Evaluations on synthetic and real-world objects demonstrate state-of-the-art accuracy, outperforming recent inverse-rendering and diffusion-based GS relighting methods across most conditions, all while completing relighting operations in under one second. Although bounded by radially symmetric BRDF approximations and the low-pass nature of the SH basis, TranSplat produces perceptually realistic renderings even for glossy, complex materials, establishing a valuable, lightweight path forward for GS relighting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13310v3">InstantSfM: Towards GPU-Native SfM for the Deep Learning Era</a></div>
    <div class="paper-meta">
      📅 2026-07-21
    </div>
    <details class="paper-abstract">
      Structure-from-Motion (SfM) is a fundamental technique for recovering camera poses and scene structure from multi-view imagery, serving as a critical upstream component for applications ranging from 3D reconstruction to modern neural scene representations such as 3D Gaussian Splatting. However, most mature SfM systems remain CPU-centric and built upon traditional optimization toolchains, creating a growing mismatch with modern GPU-based, learning-driven pipelines and limiting scalability in large-scale scenes. While recent advances in GPU-accelerated bundle adjustment (BA) have demonstrated the potential of parallel sparse optimization, extending these techniques to build a complete global SfM system remains challenging due to unresolved issues in metric scale recovery and numerical robustness. In this paper, we implement a fully GPU-based and PyTorch-compatible global SfM system, named InstantSfM, to integrate seamlessly with modern learning pipelines. InstantSfM embeds metric depth priors directly into both global positioning and BA through a depth-constrained Jacobian structure, thereby resolving scale ambiguity within the optimization framework. To ensure numerical stability, we employ explicit filtering of under-constrained variables for the Jacobian matrix in an optimized GPU-friendly manner. Extensive experiments on diverse datasets demonstrate that InstantSfM achieves state-of-the-art efficiency while maintaining reconstruction accuracy comparable to both established classical pipelines and recent learning-based methods, showing up to ${\sim40\times}$ speedup over COLMAP on large-scale scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.19100v1">FlexiAvatar: Unified 3D Gaussian Human Avatars Under Arbitrary Body Visibility</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 Accepted in ECCV 2026
    </div>
    <details class="paper-abstract">
      Reconstructing animatable 3D human avatars from monocular video is a fundamental problem in computer vision with broad applications in AR/VR and digital content creation. Existing approaches typically couple parametric body models with neural rendering or 3D Gaussian splatting and optimize all body regions jointly from short videos, which often degrades fidelity in the visible areas. To overcome this limitation, we introduce FlexiAvatar, a unified framework that explicitly optimizes only the visible body regions, effectively eliminating artifacts arising from unobserved limbs. Our method integrates occlusion-robust SMPL-X tracking with part-specific residual refinement to capture high-frequency geometric and appearance details. To complete entirely unseen regions (e.g., back views), we leverage a diffusion-based approach to generate texture consistent with the observed appearance. Experiments on full-body (NeuMan, ZJU-MoCap, WildAvatar), upper/half-body (talk-show clips), and head-only (INSTA) inputs show that FlexiAvatar delivers consistently higher reconstruction quality, outperforming state-of-the-art methods by an average PSNR improvement of approximately 3% across datasets. Finally, by restricting optimization to observed regions, our method reduces the effective number of Gaussians that must be optimized and rendered, leading to reduced runtime and memory overhead in partial-visibility scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18801v1">ZeroSplat: Generalized Referring Segmentation in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 Accepted to ECCV 2026
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3DGS) have enabled language-guided scene understanding. However, existing Referring 3D Gaussian Splatting (R3DGS) methods are fundamentally restricted to single-target queries. To reflect the ambiguity of real-world instructions, we introduce the Generalized Referring 3D Gaussian Splatting Segmentation (GR3DGS) task, which requires dynamically segmenting an arbitrary number of targets (0, 1, or $N$). To facilitate comprehensive evaluation of this new task, we construct two new benchmarks: GR-LERF and GR-ScanNet. Crucially, existing R3DGS paradigms exhibit fundamental technical bottlenecks that severely limit their performance on the GR3DGS task: they lack intrinsic 3D point-level understanding by operating merely on 2D rendered pixels, and they incur prohibitive computational overhead by requiring per-scene optimization to embed heavy semantic features. To dismantle these bottlenecks, we propose ZeroSplat, a novel training-free and zero-feature framework. ZeroSplat lifts 2D Vision-Language Model (VLM) priors into 3D space through robust multi-view geometric constraints. This strategy enables intrinsic point-level understanding without incurring any additional feature storage. Extensive experiments demonstrate that ZeroSplat significantly outperforms state-of-the-art methods across generalized and single-target scenarios while maintaining exceptional efficiency. Project Page: https://inkmind-ai.github.io/ZeroSplat
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03254v2">OF$^3$GS: On-the-Fly Feed-Forward 3D Gaussian Splatting from Unposed Images</a></div>
    <div class="paper-meta">
      📅 2026-07-21
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) enables efficient and high-fidelity novel view synthesis (NVS) from offline image sequences. However, achieving on-the-fly NVS from unposed images remains challenging: the system must reconstruct renderable 3D Gaussians as images arrive, without access to future observations. Although online feed-forward geometry methods have been developed for causal depth and point-cloud recovery, directly adapting them to NVS often leads to severe rendering artifacts because Gaussian-based rendering demands stricter multi-view consistency in primitive scale and pose-geometry alignment. Even minor deviations can accumulate under causal inference and visibly degrade rendering quality. To this end, we propose OF$^3$GS, a feed-forward framework for efficient and high-quality on-the-fly NVS from sparse-view unposed images under causal constraints. We introduce two mechanisms for causal geometric stability: a Decoupled Intrinsic Recovery Head that mitigates cumulative camera-intrinsic bias and scene-scale jitter, and Dynamic Point Refinement Offsets that relax rigid unprojection to compensate for coupled pose-depth drift. Extensive experiments show that OF$^3$GS outperforms online baselines and approaches offline feed-forward 3DGS methods under comparable sparse-input settings. It also remains memory-feasible with denser inputs. Homepage: https://richardchen225.github.io/of3gs/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.22717v1">TOM-GS: Editable Video Representation via Temporal Opacity Modulation of Static 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-07-21
    </div>
    <details class="paper-abstract">
      While Implicit Neural Representations (INRs) and dynamic 3D Gaussian Splatting (3DGS) achieve impressive results in video processing, they often fall short of producing representations that are easily editable. Recent methods address this by introducing complex spatial deformations or folded distributions, which constrain optimization and reduce flexibility for downstream editing. In this paper, we introduce TOM-GS, an editable video representation that forgoes complex deformations in favor of regular 3D Gaussians equipped with a continuous temporal opacity formulation. By assigning a learnable temporal mean and scale to the opacity of each Gaussian, our model enables static 3D spatial components to fade smoothly in and out of the scene. Grounded by robust, off-the-shelf pose estimation, our approach maintains a static spatial geometry that naturally supports a wide range of manual and physics-based edits. TOM-GS outperforms prior editable video representations in visual fidelity, while its reliance on standard 3D Gaussians ensures seamless compatibility with established 3D editing tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18539v1">AniGS: Bridging Rendering and Diffusion Prior for 3D Scene Animation</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 Preprint. Project page: https://yccyenchicheng.github.io/AniGS/
    </div>
    <details class="paper-abstract">
      Novel view rendering of large and complex reconstructed scenes is becoming increasingly photorealistic. However, most reconstructions remain static and lack the ambient motion that makes environments immersive. We present AniGS, a method for scene-level animation of 3D Gaussian Splatting (3DGS) reconstructions that adds subtle, distributed dynamics, e.g., vegetation motion, while preserving rigid structures. Unlike existing 3D animation techniques which are limited to object-centric subjects or small regions, AniGS is designed for large, cluttered, navigable scenes. AniGS represents the scene with a canonical 3DGS and models motion using a time-conditioned deformation field. To animate the entire scene, we leverage a pretrained video diffusion model and introduce an iterative dataset--model update strategy that progressively expands viewpoint coverage and repeatedly updates camera-fixed training videos using a render-and-refine scheme. To prevent artifacts from unintended motion in static areas, we further introduce a composed video-to-video refinement scheme that restricts motion to desired regions. Experiments on five real-world, large-scale outdoor scenes demonstrate that AniGS produces natural ambient dynamics and high-quality novel view videos, enabling more immersive viewing experiences of reconstructed environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18466v1">ECoNGS: Efficient Compressive Neural Gaussian Splats for Volume Visualization</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 To be published in Proceedings of IEEE VIS 2026, IEEE Transactions on Visualization and Computer Graphics
    </div>
    <details class="paper-abstract">
      Recent advances in differentiable Gaussian splatting have highlighted the potential of primitive-based approaches as alternative scene representations for interactive, high-quality, volume visualization (VolVis) of large datasets. However, the explicit nature of current primitive-based methods, combined with isolated optimization for each VolVis scene, results in redundant, non-compact representations. We present ECoNGS, an efficient compressive neural Gaussian splatting framework for VolVis scene representation. ECoNGS employs lightweight neural networks to dynamically predict implicit, editable Gaussian splats from explicit anchor points, effectively combining model compactness and parameter efficiency of implicit representations with high-performance rendering of explicit primitives. We explore a joint learning strategy that clusters geometrically similar scenes and shares parameters across them, significantly reducing overall training time and model size while maintaining reconstruction fidelity. To achieve a more compact scene representation, we further compress the explicit anchor attributes using a neural entropy model that estimates their probability distributions, enabling compact storage via entropy coding. We systematically investigate Gaussian initialization strategies and propose a simple yet effective scheme tailored for VolVis scenes, improving reconstruction accuracy and accelerating convergence. We evaluate ECoNGS qualitatively and quantitatively across various univariate and multivariate VolVis scenes, highlighting its superior performance over prior methods in training time, reconstruction quality, and model size. In particular, compared with the prior method iVR-GS, ECoNGS improves reconstruction quality by up to 2.2 dB in PSNR while reducing the model size by up to 6.1x and the training time by up to 5.9x. The code is available at https://github.com/TouKaienn/ECoNGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18067v1">QIRF Quantum-Inspired Non-Orthogonal Function-Space Compression for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves high-quality real-time rendering by representing a scene with a large collection of anisotropic Gaussian primitives. However, complex scenes often require millions of Gaussians, resulting in substantial storage and rendering costs. Existing compression methods mainly reduce redundancy through primitive-wise pruning, attribute quantization, clustering, or neural coding, while redundancy caused by strongly overlapping and non-orthogonal Gaussian basis functions remains largely unexplored. We present QIRF, a quantum-inspired non-orthogonal function-space compression method for 3D Gaussian Splatting. QIRF models neighboring Gaussian primitives as a local non-orthogonal basis and formulates primitive reduction as a subspace-aware selection problem. Specifically, an analytic Gaussian overlap matrix and a radiance-response density matrix are constructed to characterize functional redundancy and rendering relevance. Generalized eigendecomposition is then used to identify the dominant local subspace and select representative Gaussian primitives. An RRDM-based response model and detail-aware safeguarding further preserve visually important high-frequency structures under aggressive pruning. Experiments on 13 scenes from Mip-NeRF 360, Tanks and Temples, and Deep Blending show that QIRF reduces the Gaussian count and raw PLY storage by 71.7 percent on average, corresponding to approximately 3.54 times compression, while maintaining reconstruction quality comparable to 3DGS and achieving a marginal average PSNR improvement of 0.10 dB. QIRF also improves the average rendering speed over 3DGS by 34.3 percent. These results suggest that non-orthogonal function-space redundancy is an important yet underexplored source of representational redundancy in explicit Gaussian radiance fields.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17965v1">Exploration Matters for Escaping the Blur Trap in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 Project page: https://chengbo-wang.github.io/ExploreGS/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) employs Gaussian primitives for explicit scene representation, facilitating real-time, high-fidelity reconstruction and novel view synthesis of complex scenes. However, the explicit modeling inherent in 3DGS introduces a gradient bias during optimization, rendering its non-convex optimization process highly susceptible to convergence toward local suboptimal solutions. This constitutes a fundamental limitation in 3DGS optimization, which we term the Blur Trap. To address this limitation, we integrate simple explicit exploration into the 3DGS optimization framework. First, through rigorous mathematical analysis of the 3DGS optimization formulation, we identify the underlying optimization bias responsible for the Blur Trap and categorize it into two distinct subtypes: the Far-Side Blur Trap and the Near-Side Blur Trap. Subsequently, we propose two highly straightforward exploration strategies (Random Seeding and Random Splitting) to mitigate the far-side and near-side blur traps, respectively. Experimental validation demonstrates that the incorporation of these exploration operators effectively and complementarily overcome the Blur Trap, achieving high-quality rendering performance across multiple datasets. Project page: https://chengbo-wang.github.io/ExploreGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17916v1">Packet-Loss Robust 3D Gaussian Compression via Atomic Packaging and GNN-based Error Concealment</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 21 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) and recent compression schemes such as HAC++ enable high-fidelity real-time neural rendering, but their bitstreams are fragile under packet loss during network streaming. Existing compression methods often separate correlated anchor attributes into independent streams, so losing one packet can create attribute-inconsistent broken anchors and severe rendering artifacts. We propose a packet-loss robust 3DGS transmission and error concealment framework. On the encoder side, anchor-level atomic packaging jointly encapsulates all attributes of each anchor, converting corrupted-attribute failures into clean missing-anchor erasures. Stratified random grouping further disperses packet losses across the spatial domain to avoid large contiguous voids. On the decoder side, we formulate recovery as prior-aware attribute inpainting. A Context-Aware Residual Interpolation (CARI) branch uses hash-grid prior predictions and neighboring residuals to build a robust baseline, while a lightweight two-layer graph neural network with cross-attention over hash-grid priors refines high-frequency attribute residuals. Attribute-wise confidence control falls back to interpolation when learned predictions are unreliable. Experiments under 20 percent random packet loss on BungeeNeRF, Mip-NeRF 360, and Tanks and Temples show that the proposed method substantially improves over no-concealment transmission and limits average PSNR degradation to about 3 dB relative to the lossless HAC++ reference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17896v1">Locality-Aware Density Control for Efficient Gaussian-based Image Representation</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 Accepted by ACMMM 2026
    </div>
    <details class="paper-abstract">
      2D Gaussian Splatting is an attractive direction for image representation due to its explicit formulation, fast rasterization, and favorable decoding efficiency. The representation quality of this paradigm depends on the proper allocation of Gaussian capacity to the demanding regions. However, existing methods fail to allocate Gaussian capacity efficiently during optimization: under-reconstructed content is often refined in a fragmented pixel-wise manner, while neighboring optimized Gaussians with similar attributes are redundantly retained. This inefficiency motivates the need for a density control framework that jointly addresses insufficient allocation in under-reconstructed regions and redundant allocation in over-reconstructed regions. Our key insight is that this framework should exploit two complementary forms of locality: the local continuity of reconstruction errors in image space for improved Gaussian allocation, and the local similarity of neighboring Gaussians in Gaussian space for redundant elimination. Based on this insight, we propose Locality-Aware Density Control (LocoADC), a plug-and-play framework that improves Gaussian capacity utilization through Region-wise Gaussian Densification (RGD) and Similarity-Driven Gaussian Merging (SDGM) strategies, together with a local color consistency constraint for more reliable merging. Extensive experiments on diverse datasets show that LocoADC consistently improves multiple baselines by enabling more effective local Gaussian allocation, including a 2.93 dB PSNR gain over GI on the CLIC dataset under the same 30k Gaussian budget. Code is available at: \textit{https://github.com/ChenJiaCong-1005/LocoADC}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17842v1">CaT-GS: Efficient 3DGS Rendering for Large Scale Scenes via Inter-frame Caching and Tile Scheduling</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      Recent breakthroughs in 3D Gaussian Splatting (3DGS) have advanced neural rendering with high fidelity and speed. However, its performance degrades significantly in large-scale scenes due to the computational burden of tile-based rasterization. Existing optimization efforts either require costly scene re-training or focus on narrow aspects of the pipeline, overlooking critical inefficiencies in real-world deployments. Through a comprehensive analysis, we identify three primary sources of redundancy and low GPU utilization: redundant inter-frame pre-processing, viewpoint-based occlusion redundancy, and severe tile-level load imbalance. To address these issues, we propose CaT-GS, a novel and efficient 3DGS rendering pipeline. CaT-GS introduces a speculative multi-frame preprocessing method to eliminate redundant computations across consecutive frames, and an inter-frame caching mechanism to eliminate viewpoint redundant rendering stages. Furthermore, it refactors rasterization tasks with a dedicated kernel to mitigate tile load imbalance, significantly boosting GPU utilization. Extensive experiments demonstrate that CaT-GS achieves a speedup of up to 10 times over the original 3DGS and up to 70% over previous state-of-the-art methods, establishing a new benchmark for high-fidelity, real-time rendering of large-scale scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17803v1">FF-ProCams: Feed-Forward Gaussian Splatting for Projector-Camera System</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Projector-camera (ProCams) systems achieve active scene perception and controllable appearance manipulation via structured illumination, serving as a core infrastructure for spatial augmented reality, projection mapping, and surface reflectance acquisition. Existing inverse-rendering methods for ProCams deliver high-fidelity results but rely on time-consuming per-scene optimization, while mainstream feed-forward 3D reconstruction models produce baked appearance that cannot adapt to spatially varying projector illumination. To resolve this accuracy-efficiency trade-off, we propose FF-ProCams, a Feed-Forward 3D Gaussian inverse-rendering framework for ProCams. A hybrid Mamba2-Transformer encoder aggregates cross-view geometric and photometric cues from sparse multi-view observations, and lightweight heads predict a relightable Gaussian representation in a single forward pass. We further design a projector-aware differentiable renderer to synthesize camera observations under arbitrary active illumination and ProCams poses. To enable feed-forward training, we construct a large-scale synthetic ProCams dataset covering diverse object geometries and surface materials. Experiments show FF-ProCams achieves high-fidelity projector-aware rendering, generalizes to unseen patterns, and supports novel projector-camera poses. Using only 8 input views, it outperforms optimization-based baselines with 297 views while reducing test-time reconstruction to 0.13 seconds (a three-to-five-order-of-magnitude speedup). The code and data are available at https://github.com/CPREgroup/FF-ProCams/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17773v1">FillGauss: Fine-Grained Filling-Aware Impact Sound Generation for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Synthesizing physically plausible impact sounds from visual observations remains a great challenge in multi-modal AI. Existing 3D-aware audio generation methods primarily model the surface geometry of hollow rigid bodies. However, they fundamentally overlook internal filling states, a critical physical factor that drastically modulates acoustic resonance and damping. To address this issue, we have defined a new task called Fine-Grained Filling-Aware Impact Sound Generation. As a foundational step, we first introduce the fine-grained fill-aware dataset (FillImpact), a pioneering multi-modal collection comprising over 5,000 rigorous acoustic recordings from 88 diverse real-world objects. It captures impact interactions with varying internal contents (i.e., water, rice), a continuous range of fill levels, and distinct striker materials. Furthermore, comprehensive acoustic analysis confirms that the collected data closely aligns with established physical laws governing acoustic resonance and damping, indicating its suitability for physically grounded modeling. Building on this dataset, we propose a novel generative framework (FillGauss) that integrates 3D Gaussian Splatting (3DGS) with internal state conditioning for sound generation. By fusing 3DGS geometric features, precise 3D spatial strike coordinates, and fine-grained textual physical conditions within a latent diffusion architecture, FillGauss enables position-aware, striker-aware, and filling-aware audio generation. Extensive experiments demonstrate that our approach could generate high-fidelity impact sounds that adhere to underlying physical principles, establishing a new state-of-the-art for physically grounded cross-modal audio generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.08982v3">CityLoc: 6DoF Pose Distributional Localization for Text Descriptions in Large-Scale Scenes with Gaussian Representation</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Localizing textual descriptions within large-scale 3D scenes presents inherent ambiguities, such as identifying all traffic lights in a city. Addressing this, we introduce a method to generate distributions of camera poses conditioned on textual descriptions, facilitating robust reasoning for broadly defined concepts. Our approach employs a diffusion-based architecture to refine noisy 6DoF camera poses towards plausible locations, with conditional signals derived from pre-trained text encoders. Integration with the pretrained Vision-Language Model, CLIP, establishes a strong linkage between text descriptions and pose distributions. Enhancement of localization accuracy is achieved by rendering candidate poses using 3D Gaussian splatting, which corrects misaligned samples through visual reasoning. We validate our method's superiority by comparing it against standard distribution estimation methods across five large-scale datasets, demonstrating consistent outperformance. Code, datasets and more information will be publicly available at our project page.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.09260v2">AnythingReality: Robust Online Gaussian Splatting SLAM for Open-Vocabulary VR Scene Exploration</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      We present a novel integrated architecture for robust online 3D Gaussian splatting, real-time VR exploration, and speech-driven Vision-Language-Model interaction. Unlike methods assuming clean depth or external poses, our system combines ORB-SLAM3-based pose estimation with online Gaussian reconstruction for noisy real-world data. A VR pipeline enables immersive exploration of incremental reconstructions; a semantic module transcribes voice commands, generates scene descriptions, and records points of interest. Against state-of-the-art online Gaussian splatting methods, we improve image quality on our dataset (+14.5% PSNR, +8.6% SSIM, -14.3% LPIPS) and TUM-RGBD (+11.7% PSNR, +7.8% SSIM, -21.6% LPIPS), with comparable or superior frame rates via quality-speed configurations. We achieve an 88% VLM object-recognition rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12972v2">SplatSearch: Instance Image Goal Navigation for Mobile Robots using 3D Gaussian Splatting and Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 Project Page: https://splat-search.github.io/
    </div>
    <details class="paper-abstract">
      The Instance Image Goal Navigation (IIN) problem requires mobile robots deployed in unknown environments to search for specific objects or people of interest using only a single reference goal image of the target. This problem can be especially challenging when: 1) the reference image is captured from an arbitrary viewpoint, and 2) the robot must operate with sparse-view scene reconstructions. In this paper, we address the IIN problem, by introducing SplatSearch, a novel architecture that leverages sparse-view 3D Gaussian Splatting (3DGS) reconstructions. SplatSearch renders multiple viewpoints around candidate objects using a sparse online 3DGS map and uses a multi-view diffusion model to complete missing regions of the rendered images, enabling robust feature matching against the goal image. A novel frontier exploration policy is introduced which uses visual context from the synthesized viewpoints with semantic context from the goal image to evaluate frontier locations, allowing the robot to prioritize frontiers that are semantically and visually relevant to the goal image. Extensive experiments in photorealistic home and real-world environments validate the higher performance of SplatSearch against current state-of-the-art methods in terms of Success Rate and Success Path Length. An ablation study confirms the design choices of SplatSearch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.16926v1">Splat-based 3D Scene Reconstruction with Extreme Motion-blur</a></div>
    <div class="paper-meta">
      📅 2026-07-18
    </div>
    <details class="paper-abstract">
      We propose a splat-based 3D scene reconstruction method from RGB-D input that effectively handles extreme motion blur, a frequent challenge in low-light environments. Under dim illumination, RGB frames often suffer from severe motion blur due to extended exposure times, causing traditional camera pose estimation methods, such as COLMAP, to fail. This results in inaccurate camera pose and blurry color input, compromising the quality of 3D reconstructions. Although recent 3D reconstruction techniques like Neural Radiance Fields and Gaussian Splatting have demonstrated impressive results, they rely on accurate camera trajectory estimation, which becomes challenging under fast motion or poor lighting conditions. Furthermore, rapid camera movement and the limited field of view of depth sensors reduce point cloud overlap, limiting the effectiveness of pose estimation with the ICP algorithm. To address these issues, we introduce a method that combines camera pose estimation and image deblurring using a Gaussian Splatting framework, leveraging both 3D Gaussian splats and depth inputs for enhanced scene representation. Our method first aligns consecutive RGB-D frames through optical flow and ICP, then refines camera poses and 3D geometry by adjusting Gaussian positions for optimal depth alignment. To handle motion blur, we model camera movement during exposure and deblur images by comparing the input with a series of sharp, rendered frames. Experiments on a new RGB-D dataset with extreme motion blur show that our method outperforms existing approaches, enabling high-quality reconstructions even in challenging conditions. This approach has broad implications for 3D mapping applications in robotics, autonomous navigation, and augmented reality. Both code and dataset are publicly available on https://github.com/KAIST-VCLAB/gs-extreme-motion-blur.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.16838v1">TopoGS: Planar Reconstruction via Topology-aware 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-18
      | 💬 European Conference on Computer Vision (Proceedings of ECCV 2026); Project page: https://vcc.tech/research/2026/TopoGS
    </div>
    <details class="paper-abstract">
      Extracting structured, parametric 3D representations from raw images remains a fundamental challenge in computer vision and graphics. While recent advancements in the 3D Gaussian Splatting (3DGS) pipeline integrate planar primitives to yield compact and editable geometry, these approaches typically treat planes as isolated, discrete sets. This lack of topological connectivity hinders robust geometric reasoning, leading to fragmented reconstructions and misaligned boundaries that fall short of the precision for rigorous spatial analysis and professional design workflows. To address this, we introduce TopoGS, the first 3DGS framework to explicitly integrate both planar and topological constraints for coherent 3D reconstruction. Specifically, we extract global 2D topological relationships from multi-view image segmentations and anchor Gaussian primitives to these structural elements. This formulation enables the joint optimization of plane parameters, rendering fidelity, and topological adjacency. By enforcing strict multi-view consistency alongside these topological constraints, our method significantly mitigates geometric misalignments and produces connected, structured 3D models. Extensive evaluations on the ScanNet++ dataset demonstrate that TopoGS achieves state-of-the-art performance, providing a highly robust solution for generating accurate, topologically sound, and visually faithful scene representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20714v3">The Role of Initialization in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-18
      | 💬 Sources are available at https://github.com/deivse/ivd_splat . Changes in this version: extended experiments, adjusted direction
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become the method of choice for photo-realistic novel view synthesis (NVS), due to its efficiency and compelling visual quality. 3DGS represents the scene through a set of 3D Gaussians, parameterized by their position, spatial extent, and view-dependent color. Starting from an initial point cloud, 3DGS refines the Gaussians' parameters as to reconstruct a set of training images as accurately as possible. Typically, a sparse Structure-from-Motion point cloud is used as initialization. In order to obtain a full scene representation, 3DGS methods thus rely on a densification stage. In this paper, we systematically study how initialization affects 3DGS NVS performance and geometric quality, using several densification strategies. We show that dense initialization does not lead to consistent visual improvements when paired with strong densification. Despite that, it can help in generalization to off-trajectory views and significantly improves geometric consistency of the scene representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.16624v1">SPARE-GS: Structural Parsimony and Resource Efficiency for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-18
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves high-fidelity novel view synthesis in real-time; however its training efficiency and representation compactness are hindered by excessive primitive proliferation. To address this challenge, we formulate the structural evolution of 3DGS as a global budget-constrained optimization problem and derive an optimality condition, which requires the marginal utility of structural resources to be balanced across spatial regions under a finite primitive budget. Based on this formulation, we propose SPARE-GS, a general plug-and-play framework that dynamically aligns the distribution of 3D Gaussian primitives with regional representational demand. SPARE-GS estimates capacity-normalized regional demand, assigns adaptive target quotas, and uses regional budget deviations to coordinate densification, pruning and adaptive termination toward a more balanced structural allocation. Extensive experiments across standard, accelerated, and structure-enhanced 3DGS pipelines demonstrate that SPARE-GS reduces the Gaussian count and training time by an average of 30.38% and 23.81%, respectively, while improving the average PSNR. Moreover, the resulting compact representations reduce downstream processing time and improve the rate-distortion performance of diverse compression and pruning methods, demonstrating the broad applicability of global structural budget regulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02402v2">SoMA: A Real-to-Sim Neural Simulator for Robotic Soft-body Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-07-18
      | 💬 Project page: https://city-super.github.io/SoMA/
    </div>
    <details class="paper-abstract">
      Simulating deformable objects under rich interactions remains a fundamental challenge for real-to-sim robot manipulation, with dynamics jointly driven by environmental effects and robot actions. Existing simulators rely on predefined physics or data-driven dynamics without robot-conditioned control, limiting accuracy, stability, and generalization. This paper presents SoMA, a 3D Gaussian Splat simulator for soft-body manipulation. SoMA couples deformable dynamics, environmental forces, and robot joint actions in a unified latent neural space for end-to-end real-to-sim simulation. Modeling interactions over learned Gaussian splats enables controllable, stable long-horizon manipulation and generalization beyond observed trajectories without predefined physical models. SoMA improves resimulation accuracy and generalization on real-world robot manipulation by 20%, enabling stable simulation of complex tasks such as long-horizon cloth folding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26985v2">Vis4GS: A Visual Analytic Tool for 3D Gaussian Splatting Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-07-17
      | 💬 Accepted to IEEE VIS 2026 Short Papers
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) supports fast training and real-time rendering, but its optimization process remains difficult to interpret. Existing viewers mainly expose the final reconstructed scene and offer limited support for explaining how Gaussian properties contribute to visible artifacts or evolve during training. We present Vis4GS, a multi-view visual analytics tool for primitive-level diagnosis of 3DGS reconstruction artifacts. Built on the original 3DGS viewer and training framework, Vis4GS links rendered artifacts to Gaussian properties, View Coverage, training progress, and Gaussian genealogy through four linked views: an interactive Gaussian analysis view, a property timeline view, a Gaussian densification tree view, and a log and control panel. The system supports Gaussian selection, blur and needle-like artifact scoring, View Coverage analysis, and multiscale genealogy exploration of clone, split, prune, and clone-split events. By connecting scene-level artifacts with primitive-level evidence and optimization history, Vis4GS enables a structured workflow for diagnosing reconstruction failures beyond final-image inspection and global metrics. A user study also shows that Vis4GS provides stronger support for usability and artifact understanding than the original 3DGS viewer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.16362v1">OmniStyle-INR: Universal and Multimodal Style Transfer for INRs</a></div>
    <div class="paper-meta">
      📅 2026-07-17
    </div>
    <details class="paper-abstract">
      Style transfer remains a fundamental and highly important task across various data modalities, enabling creative manipulation conditioned by both reference images and textual descriptions. Recently, methods utilizing Gaussian Splatting have emerged as a unified representation for 2D images, video, 3D scenes, and 4D dynamics. However, representing videos and 2D images with Gaussian Splatting is structurally sub-optimal for dense continuous domains. The number of required Gaussians often approaches the total number of pixels, raising questions about the actual utility of such a representation for these specific modalities. In contrast, Implicit Neural Representations have established themselves as a much more popular and natural choice across all these data domains. Implicit Neural Representations naturally provide significant advantages, including data compression, inherent capabilities for super resolution, and seamless integration with deep generative models. To this end, we introduce OmniStyle-INR, a novel framework that leverages network-based continuous representations as a truly universal domain. Our approach successfully performs high-quality style transfer across all visual modalities, guided seamlessly by both text prompts and visual exemplars.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.15806v1">HybridSim: A Physics-Learning Hybrid Digital Twin for mmWave Human Sensing</a></div>
    <div class="paper-meta">
      📅 2026-07-17
      | 💬 Accepted to ECCV 2026. Project Page: https://weitao-xiong.github.io/HybridSim/
    </div>
    <details class="paper-abstract">
      High-fidelity simulation of mmWave radar signals for dynamic human motion is valuable for developing radar-based human sensing models; yet collecting accurately labeled measurements for a specific deployment site remains expensive. We present HybridSim, a physics-learning hybrid simulator that synthesizes mmWave radar signals from dynamic human meshes under a fixed indoor room configuration, explicitly decoupling propagation into two components. To parameterize the human subject, we use a tri-plane representation to extract human features and a Graph Convolutional Network to stabilize optimization and mitigate gradient instability. The direct signal path is modeled via an inverse-rendering formulation with a microfacet BRDF to capture primary surface reflections. In parallel, the indirect path is approximated by combining 3D Gaussian Splatting with a virtual-receiver geometry to fit and reproduce site-specific multipath interference patterns, achieving substantially lower computational cost than explicit full ray tracing. Experiments in a fixed-room setting show improved agreement with a physically based reference and consistent gains on downstream radar-based human sensing tasks when using HybridSim for site-specific data augmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07168v2">NoDrift3R: Raymap-Guided Coupling for Drift-Robust Unposed Feed-Forward 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-07-17
      | 💬 Project page: see https://xiangyu1sun.github.io/NoDrift3R-project-page/
    </div>
    <details class="paper-abstract">
      Pose-Free Feed-forward 3D Gaussian Splatting (3DGS) has recently emerged as a powerful paradigm for fast scene reconstruction. However, its performance degrades significantly in long image sequences due to cumulative camera pose estimation drift, which propagates errors into geometric modeling and severely limits rendering fidelity. In this work, we revisit the long-sequence bottleneck and identify pose drift as the primary factor restricting reconstruction quality. Furthermore, while SfM-based pseudo ground-truth poses introduce sensor noise, purely rendering-based supervision often leads to optimization instability and local minima due to the entangled optimization of geometry and pose. To address the challenges, we propose a synergistic pose-free framework that explicitly couples geometry and appearance via a Raymap-Guided Coupling Module (RGC). Concretely, we anchor Gaussian centers to raymap-induced geometry and jointly optimize RGB reconstruction, raymap consistency, and camera regularization under a unified objective, yielding a bidirectional feedback loop: stronger geometry improves rendering, and appearance supervision in turn refines geometry and pose. To further stabilize learning across wide temporal ranges, we introduce a Dual-Frequency Viewpoint Scheduling strategy that combines easy-to-hard interval expansion with replay of short-interval pairs. Extensive experiments across in-domain and cross-domain datasets show consistent gains in both rendering and pose estimation, with notably improved robustness on long sequences. Ablation studies validate our central insight: explicitly designed geometry-appearance synergy is the key to scalable and drift-robust pose-free feed-forward 3D reconstruction. Project page: https://xiangyu1sun.github.io/NoDrift3R-project-page/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13154v2">Worlds in One Demo: A Synthetic Data Engine for Learning Open-World Mobile Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-07-17
      | 💬 Project website: https://wanda.lecar-lab.org/
    </div>
    <details class="paper-abstract">
      Learning open-world mobile manipulation policies requires vast data to achieve spatial generalization, long-horizon robustness, and scene generalization. Current prevailing data collection paradigms, teleoperation and UMI, demand prohibitive human effort and cost at scale. To scale beyond the limits of manual data collection, we seek to maximize the value of each human demonstration by scalable data generation. To this end, we introduce WANDA: learning open-World mobile mANipulation from one demonstration via a synthetic DAta engine. WANDA first reconstructs background Gaussian splats and robot-object interaction trajectories from source RGBD observations, as a world substrate for later planning and rendering. It then rearranges contact-rich robot-object interaction segments into extensive spatial configurations, utilizing whole-body motion planning to chain them into new trajectories. To enhance long-horizon robustness, it applies Corrective State Expansion to increase the robot and object state diversity at different stages of mobile manipulation. To unlock cross-environment generalization, trajectories are synthesized on diverse generated 3D worlds from everyday photos. Furthermore, we synthesize photo-realistic observations by compositing rendered robot and object meshes with Gaussian splatting backgrounds. We evaluate our approach on extensive simulation and real-world tasks in various scenes. Experiments show that policies trained with WANDA achieve long-horizon robustness, broad spatial generalization and cross-environment generalization from one real demonstration. Moreover, WANDA naturally supports cross-embodiment data generation, validated by zero-shot deployment on another mobile manipulator with a distinct morphology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.15542v1">ImprovedVBGS: Real-time Continual Variational Bayes Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-17
      | 💬 5 pages, 4 figure. Technical Report. This work introduces ImprovedVBGS, accelerated continual learning for 3D Gaussian Splatting based Reconstruction. Code available at [https://github.com/damanimc/ImprovedVBGS](https://github.com/damanimc/ImprovedVBGS)
    </div>
    <details class="paper-abstract">
      On-the-fly reconstruction is a key requirement for many applications in robotics and autonomous navigation. Variational Bayes Gaussian Splatting (VBGS) enables continual learning without replay buffers using Coordinate Ascent Variational Inference (CAVI), but its per-frame iterations over all observed points make it too slow for real-time use with strict memory and latency requirements. We present ImprovedVBGS, an accelerated framework for on-the-fly continual reconstruction. This is achieved primarily through (i) spatially truncated variational inference, and (ii) improved reassignment that uses forwarding, truncation and eliminates wasteful dynamic recompilation. On the NeRF synthetic dataset, we reduce mean per-frame latency from ~84.0 s to ~0.050 s on an RTX 3070 Ti, a 1680x speed-up while maintaining reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.15536v1">E3DGS: Unified Geometric-Photometric Equivariance for 3D Gaussian Splatting via Color-as-Geometry Embedding</a></div>
    <div class="paper-meta">
      📅 2026-07-17
      | 💬 31 pages, 7 figures, 5 tables
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) captures scenes by coupling explicit geometry (position, covariance) with view-dependent photometry (Spherical Harmonics). However, building $\mathrm{SE}(3)$-equivariant architectures on these primitives presents a fundamental representation bottleneck. Color has been treated as a signal rather than a geometric entity, making it nontrivial to unify symmetry across geometry and appearance as the camera frame changes. While translations are handled by relative coordinates, rotations act heterogeneously across attributes: $μ\mapsto Rμ$, $Σ\mapsto RΣR^\top$, and $f_\ell\mapsto D^\ell(R)f_\ell$. This mismatch complicates strict equivariance, leading existing methods to either discard or flatten SH coefficients, thereby breaking symmetry. We propose a unified solution rooted in representation theory: for SH degrees $\ell\le2$, photometry is algebraically isomorphic to a rank-2 geometric tensor. We prove that the Wigner-$D$ action on these SH coefficients can be exactly reformulated as the conjugation action on $3\times3$ matrices. Leveraging this, we introduce the Unified Matrix Embedding, a lifting that maps all Gaussian attributes into a unified carrier space, $\mathfrak{gl}(3)$. Building on the "Color-as-Geometry" formulation, we present E3DGS, a rigid-body ($\mathrm{SE}(3)$) equivariant architecture that processes 3D Gaussians without Clebsch-Gordan tensor products. Evaluations on object vision and action-conditioned Gaussian world modeling demonstrate that our unified approach yields strong robustness under camera-frame changes and improved data efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.19474v2">DINO-SLAM: DINO-informed RGB-D SLAM for Neural Implicit and Explicit Representations</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      This paper presents DINO-SLAM, a DINO-informed design strategy to enhance implicit (Neural Radiance Field -- NeRF) and explicit representations (Gaussian Splatting -- GS) in SLAM systems through the more comprehensive semantics understanding enabled by DINO. This latter alone, however, lacks proper 3D geometry understanding, allowing only for marginal improvements. Therefore, we rely on a Scene Geometry Encoder (SGE) to enrich DINO features into geometry-aware DINO features (geoDINO), to better understand those geometric relationships that vanilla DINO features fail to capture. Building upon it, we propose two foundational paradigms for NeRF and GS SLAM systems integrating geoDINO features. Compared to state-of-the-art methods, our DINO-informed pipelines achieve superior performance on the Replica, ScanNet, and TUM datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14997v1">AeroAct: Action-Centered World-Action Models for Language-Conditioned Quadrotor Flight</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      Language-conditioned quadrotor flight requires a policy to ground semantic goals, anticipate the visual consequences of ego-motion, and output control references that remain smooth and dynamically executable under rapidly changing first-person views. Existing aerial vision-language navigation and vision-language-action methods commonly use discrete actions, high-level waypoints, or instantaneous velocity commands, which provide limited supervision about how flight actions change future observations. We present AeroAct, an action-centered world-action model (WAM) for quadrotor navigation. To the best of our knowledge, AeroAct is the first WAM instantiated and demonstrated for real-world aerial flight. The model adapts a pretrained video diffusion Transformer to predict local trajectory-action chunks from egocentric visual history, proprioception, and language. Future first-person frames are used during training as dense consequence supervision, while deployment directly decodes actions without generating future video. To obtain aligned visual, state, language, and dynamically feasible action data, we build a DiffAero-based pipeline with complementary Isaac Lab and 3D Gaussian splatting renderers. We further introduce a low-cost handheld collection device that couples camera observations with motion estimates to recreate flight-like egocentric trajectories, and a self-guidance procedure that improves temporal consistency across overlapping trajectory chunks. Closed-loop simulation and real-world experiments show that temporal visual context improves target tracking and object-search performance, and that WAM-based policies can be executed on a physical quadrotor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14990v1">JADE-GS: Joint Alternating Deblurring Guided by Events in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      When a camera moves fast during exposure, blur destroys the intra-exposure motion a 3D model needs to recover the sharp scene, while event cameras capture exactly this signal at microsecond resolution. Turning them into reliable 3D supervision faces two obstacles. First, the two restoration priors fail in opposite ways: physics-based event-integration priors preserve edges but accumulate drift; learned networks recover texture but distort boundaries. Second, existing pipelines run in one direction only, so raw event noise or the biases of fixed 2D pseudo-labels pass uncorrected into the geometry. JADE-GS addresses both: a pixel-adaptive routing gate fuses the complementary priors, and the resulting 2D restorer is coupled to a 3D Gaussian Splatting student in a bidirectional loop, where detached, multi-view-consistent renders and a physics-based reblurring constraint regularize the restorer, turning a fixed preprocessor into a geometry-aware predictor. Across synthetic and real benchmarks, JADE-GS attains the best perceptual quality, leading LPIPS and CLIP-IQA on both benchmarks with competitive PSNR and SSIM, and trainsin about one hour under 5 GB on a single consumer GPU while preserving real-time rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09243v2">A$^2$TG: Adaptive Anisotropic Textured Gaussians for Efficient 3D Scene Representation</a></div>
    <div class="paper-meta">
      📅 2026-07-16
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has emerged as a powerful representation for high-quality, real-time 3D scene rendering. While recent works extend Gaussians with learnable textures to enrich visual appearance, existing approaches allocate a fixed square texture per primitive, leading to inefficient memory usage and limited adaptability to scene variability. In this paper, we introduce adaptive anisotropic textured Gaussians (A$^2$TG), a novel representation that generalizes textured Gaussians by equipping each primitive with an anisotropic texture. Our method employs a gradient-guided adaptive rule to jointly determine texture resolution and aspect ratio, enabling non-uniform, detail-aware allocation that aligns with the anisotropic nature of Gaussian splats. This design significantly improves texture efficiency, reducing memory consumption while enhancing image quality. Experiments on multiple benchmark datasets demonstrate that A TG consistently outperforms fixed-texture Gaussian Splatting methods, achieving comparable rendering fidelity with substantially lower memory requirements. Project page: http://github.com/Rickyeeeeee/A2TG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04043v2">AnyStyle: Single-Pass Multimodal Stylization for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      The growing demand for rapid and scalable 3D asset creation has driven interest in feed-forward 3D reconstruction methods, with 3D Gaussian Splatting (3DGS) emerging as an effective scene representation. While recent approaches have demonstrated pose-free reconstruction from unposed image collections, integrating stylization or appearance control into such pipelines remains underexplored. Existing attempts largely rely on image-based conditioning, which limits both controllability and flexibility. In this work, we introduce AnyStyle, a feed-forward 3D reconstruction and stylization framework that enables pose-free, zero-shot stylization through multimodal conditioning. Our method supports both textual and visual style inputs, allowing users to control the scene appearance using natural language descriptions or reference images. We propose a modular stylization architecture that requires only minimal architectural modifications and can be integrated into existing feed-forward 3D reconstruction backbones. Experiments demonstrate that AnyStyle improves style controllability over prior feed-forward stylization methods while preserving high-quality geometric reconstruction. A user study further confirms that AnyStyle achieves superior stylization quality compared to an existing state-of-the-art approach. Repository: https://github.com/joaxkal/AnyStyle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14513v1">Compression of 3D Gaussian Splatting Data Using GPU-friendly Graphics Texture Coding</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      Techniques for modeling 3D scenes from image collections, such as 3D Gaussian Splatting (3DGS), are capable of generating high-quality novel views by leveraging graphics primitives with view-dependent appearance. In 3DGS, spherical harmonic (SH) are employed to model view-dependent color, resulting in a large number of SH coefficients per primitive and large memory requirements. While compression approaches have been proposed to mitigate this problem, they do not exploit the capabilities of modern Graphics Processing Units (GPUs) for parallel decoding and rendering. In this paper, we propose a method for compressing SH color coefficients using texture compression schemes specifically designed for efficient parallel GPU decoding and supported by dedicated hardware acceleration. It is shown that those methods can compress color coefficients more effectively than 2D textures by exploiting the fact that primitives can be locally grouped and reordered according to color. Furthermore, we introduce a bit-rate control strategy that preserves random access, enabling large-scale parallelization without compromising rendering performance. Experimental results using BC1 and BC7 texture compression formats show that GPU-based decompression can be achieved with negligible or imperceptible degradation in the visual quality of rendered 3DGS scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14481v1">Immediate 3D Gaussian Splat Reconstruction of Unordered Input with Global Consistency</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become the method of choice for reconstructing and real-time rendering of captured scenes. To capture a scene with good visual quality, continuous image sequences are usually combined with out-of-order shots for better scene coverage. Structure from motion can reconstruct such captures, but only after they are all available and often with high computational cost. Incremental reconstruction methods -- often derived from SLAM solutions -- provide immediate feedback, but cannot handle the out-of-order capture we require. We provide the first immediate feedback solution for such radiance field capture that provides global consistency. We first introduce a method for fast matching in out-of-order sequences, by repurposing visual place recognition models and a covisibility graph, and provide an efficient way to find highly connected keyframes, improving quality even for ordered sequences. We show how these steps -- together with GPU optimization and careful Gaussian primitive placement -- provide fast local reconstruction, in our challenging radiance field reconstruction case. We then introduce a novel cluster-based method, again using the covisibility graph, to provide efficient loop closure that does not require sequential input. Finally, to handle large scenes in our context, we introduce a progressive hierarchy that allows our method to scale to large environments, without compromising efficiency. Our results show we provide immediate feedback 3DGS reconstruction with good visual quality in several datasets, with up to thousands of input images.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14470v1">G$^2$SR: Geometric Methods for Fast and Memory-Efficient Gaussian-based Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-07-16
      | 💬 8 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Few-view surface reconstruction recovers the visible surfaces of a scene from a few posed RGB images, providing the 3D models that robots need to explore and interact online. On mobile platforms, the reconstruction must be fast and geometrically accurate while keeping a small memory footprint to ensure safe and efficient operation. 3D Gaussian Splatting (3DGS) offers a high-fidelity scene representation, but building it from a few views is ill-posed, as many distinct surfaces reproduce the same images, making traditional photometric methods prone to "floater" artifacts. End-to-end methods resolve the ambiguity by regressing splats with large, usually Transformer-based, networks that require heavy compute and memory while generalizing poorly to new scenes. We propose G2SR, which exploits a well-posed core of the task: given cross-view 2D splat correspondences, 3D splats follow analytically from multi-view geometry. G2SR employs a lightweight neural frontend to detect and track 2D Gaussian splats on the image plane and an analytic backend to triangulate each into a metric-scale 3D splat. On ScanNet, Replica, and DTU, G2SR matches or exceeds the geometric accuracy of state-of-the-art end-to-end methods while running at 69-89 reconstructions per second within 203 MB of GPU memory (5-107x less) for 2- and 3-view inputs at 384 x 512 resolution, offering a practical path to online Gaussian-based surface reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01578v2">MVFusion-GS: Motion-Variance Guided Temporal Attention for High-Quality Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables real-time novel view synthesis for static scenes. Extending it to dynamic scenes via deformation fields has recently attracted significant attention, particularly for dynamic scene reconstructionband distractor-free. However, existing deformation networks lack explicit motion awareness: they neither capture long-term motion intensity nor exploit short-term temporal coherence, leading to inaccurate foreground deformation and pseudo-static residuals in the background. We present MVFusion-GS, a method that enhances deformation networks with two complementary motion-aware mechanisms. The Motion-Variance Guided Refinement aggregates per-Gaussian deformation statistics across time to estimate motion variance and uses it to guide dynamic-static separation during deformation prediction. The MotionFormer Temporal Attention module applies Transformer self-attention over neighboring timesteps to model local motion dependencies and improve temporal consistency. Extensive experiments on both dynamic scene reconstruction and distractor-free reconstruction benchmarks demonstrate state-of-the-art performance, showing that explicit motion awareness improves both foreground motion modeling and static background reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.18242v2">GSVisLoc: Generalizable Visual Localization for Gaussian Splatting Scene Representations</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 Accepted to ICCV 2025 Workshops (CALIPOSE). Project page: https://gsvisloc.github.io/
    </div>
    <details class="paper-abstract">
      We introduce GSVisLoc, a visual localization method designed for 3D Gaussian Splatting (3DGS) scene representations. Given a 3DGS model of a scene and a query image, our goal is to estimate the camera's position and orientation. We accomplish this by robustly matching scene features to image features. Scene features are produced by downsampling and encoding the 3D Gaussians while image features are obtained by encoding image patches. Our algorithm proceeds in three steps, starting with coarse matching, then fine matching, and finally by applying pose refinement for an accurate final estimate. Importantly, our method leverages the explicit 3DGS scene representation for visual localization without requiring modifications, retraining, or additional reference images. We evaluate GSVisLoc on both indoor and outdoor scenes, demonstrating competitive localization performance on standard benchmarks while outperforming existing 3DGS-based baselines. Moreover, our approach generalizes effectively to novel scenes without additional training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17386v2">TerraTransfer: Learning End-to-End Driving Policies Without Expert Demonstrations</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      End-to-end autonomous driving has achieved state-of-the-art performance on benchmarks and real-world deployments. Its standard training recipe, however, is expensive across all stages: collecting and labeling millions of driving frames is costly, and closed-loop RL on images is bottlenecked by the per-step cost of photorealistic rendering plus a forward pass through a large vision backbone. Self-play in vectorized simulators changes the economics: millions of rollout steps per second, and a state distribution naturally rich in collisions, near-misses, and recoveries that no driving log contains. Our approach exploits this asymmetry by decoupling learning to drive from learning to see. We pretrain a single policy by self-play, then align its latent space with a pretrained vision backbone, through the action KL divergence and a batch-relational low-rank structural loss. The action target comes from the self-play policy, so alignment never supervises against a logged trajectory: a paired dataset of (image, scene-state) frames suffices, with no need for the curated expert demonstrations that imitation pretraining is built on. On photorealistic 3D Gaussian splatting closed-loop scenarios, the resulting end-to-end policy matches or exceeds prior end-to-end methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14203v1">Instant NuRec: Feed-Forward 3D Gaussian Reconstruction for Driving Scene Simulation</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 Project Page: https://research.nvidia.com/labs/sil/projects/instant-nurec/
    </div>
    <details class="paper-abstract">
      3D simulation platforms are critical for autonomous driving because they enable end-to-end policy evaluation, thereby reducing development costs and improving safety. In recent years, neural simulation has become predominant, with methods such as NuRec playing a central role; however, these methods remain relatively slow and typically require per-scene tuning. In this work, we present Instant NuRec, a feed-forward neural reconstruction model that turns a short multi-view driving log into a fully simulatable 3D Gaussian Splatting (3DGS) world in a single forward pass. The model accepts multi-view input from a calibrated camera rig and emits a layered output consisting of static and dynamic 3DGS layers, a sky cubemap, and per-camera ISP corrections, while providing native support for non-pinhole camera models via 3DGUT. It reconstructs a 10-20-second multi-camera scene in roughly 1.5 seconds and achieves a PSNR on the Waymo Open Dataset that is 2.01 dB above the strongest evaluated baseline. Instant NuRec is deeply integrated into NuRec and is compatible with AlpaSim for closed-loop simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13808v1">Bake It Till You Make It: Ultrafast Spatial Texture-Atlas Splatting</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 12 pages, 6 figures. Project page with videos and interactive demos: https://nilkel.github.io/bitymi/
    </div>
    <details class="paper-abstract">
      Recent extensions of 3D Gaussian Splatting (3DGS) capture fine color details using hash-grid-based appearance parameterization but incur high computational cost during fragment rendering. We introduce a decoupled radiance representation that models low-frequency geometry and view dependent appearance features with 2D surfels while representing high-frequency textures via a view-independent spatial hash grid that is baked into a compact texture atlas. By including sparsity-enhancing optimizations that penalize semi-transparency and per-primitive falloff, our method aggressively prunes insignificant surfels and achieves significantly faster and sparser reconstructions than prior work. Exploiting geometric sparsity and efficient GPU texture mapping, our approach achieves up to a fivefold speedup over 3DGS while preserving state-of-the-art visual fidelity, enabling real-time 4K rendering at 60 FPS on consumer hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13682v1">Calibrated Closed-Form Uncertainty for Radiative Gaussian Splatting in Sparse-View CT</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 19 pages, 6 figures. Equal contribution: Chulin Zhao and Yiran Xu.(Co-first authors) Corresponding author: Yiran Xu (2617300@dundee.ac.uk)
    </div>
    <details class="paper-abstract">
      Radiative Gaussian splatting has made sparse-view CT reconstruction fast, but existing methods output point estimates with no notion of where the reconstruction can be trusted. We exploit a property of transmissive X-ray imaging that RGB splatting cannot claim -- projection and voxelization are strictly linear in the per-Gaussian densities -- to equip radiative Gaussians with a variational density posterior whose predictive variance propagates in closed form, exactly, in a single forward pass, in both volume space ($σ^2(x)=\sum_i g_i(x)^2 s_i^2$) and projection space ($\mathrm{Var}[I_p]=\sum_i w_{i,p}^2 s_i^2$). We present the first systematic calibration study for Gaussian-splatting CT (Spearman / AUSE / ECE with temperature scaling), showing that the resulting per-voxel uncertainty ranks true reconstruction error on 14 of 15 scenes of the official benchmark across three view budgets -- 9 of 15 additionally meeting our magnitude-calibration target after a single temperature -- while the perturbation-ensemble heuristic of concurrent work, transplanted to voxel space under the same protocol on our development scenes, does not (rank correlation as low as $-0.08$). We then dissect why uncalibrated acquisition scores can nevertheless select acceptable views, identifying three regimes -- flat (isotropic, balanced), pathological (degenerate coverage), and anisotropic -- and showing, in controlled single-scene testbeds, that principled uncertainty earns a measurable premium only in the last, motivating a coverage-gated, maturity-scheduled acquisition policy; the same calibrated posterior further points toward a dose-adaptive stopping rule, whose experimental validation we leave to future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21400v4">You Only Gaussian Once: Controllable 3D Gaussian Splatting for Ultra-Densely Sampled Scenes</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 15 pages, 5 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has revolutionized neural rendering, yet existing methods remain predominantly research prototypes ill-suited for production-level deployment. We identify a critical "Industry-Academia Gap" hindering real-world application: unpredictable resource consumption from heuristic Gaussian growth, the "sparsity shield" of current benchmarks that rewards hallucination over physical fidelity, and severe multi-sensor data pollution. To bridge this gap, we propose YOGO (You Only Gaussian Once), a system-level framework that reformulates the stochastic growth process into a deterministic, budget-aware equilibrium. YOGO integrates a novel budget controller for hardware-constrained resource allocation and an availability-registration protocol for robust multi-sensor fusion. To push the boundaries of reconstruction fidelity, we introduce Immersion v1.0, the first ultra-dense indoor dataset specifically designed to break the "sparsity shield." By providing saturated viewpoint coverage, Immersion v1.0 forces algorithms to focus on extreme physical fidelity rather than viewpoint interpolation, and enables the community to focus on the upper limits of high-fidelity reconstruction. Extensive experiments demonstrate that YOGO achieves state-of-the-art visual quality while maintaining a strictly deterministic profile, establishing a new standard for production-grade 3DGS. To facilitate reproducibility, part scenes of Immersion v1.0 dataset and source code of YOGO has been publicly released. The project link is https://jjrcn.github.io/yogo-project-home/.
    </details>
</div>
