# gaussian splatting - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01847v1">ESGaussianFace: Emotional and Stylized Audio-Driven Facial Animation via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 13 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Most current audio-driven facial animation research primarily focuses on generating videos with neutral emotions. While some studies have addressed the generation of facial videos driven by emotional audio, efficiently generating high-quality talking head videos that integrate both emotional expressions and style features remains a significant challenge. In this paper, we propose ESGaussianFace, an innovative framework for emotional and stylized audio-driven facial animation. Our approach leverages 3D Gaussian Splatting to reconstruct 3D scenes and render videos, ensuring efficient generation of 3D consistent results. We propose an emotion-audio-guided spatial attention method that effectively integrates emotion features with audio content features. Through emotion-guided attention, the model is able to reconstruct facial details across different emotional states more accurately. To achieve emotional and stylized deformations of the 3D Gaussian points through emotion and style features, we introduce two 3D Gaussian deformation predictors. Futhermore, we propose a multi-stage training strategy, enabling the step-by-step learning of the character's lip movements, emotional variations, and style features. Our generated results exhibit high efficiency, high quality, and 3D consistency. Extensive experimental results demonstrate that our method outperforms existing state-of-the-art techniques in terms of lip movement accuracy, expression variation, and style feature expressiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01660v1">Animated 3DGS Avatars in Diverse Scenes with Consistent Lighting and Shadows</a></div>
    <div class="paper-meta">
      📅 2026-01-04
      | 💬 Our project page is available at https://miraymen.github.io/dgsm
    </div>
    <details class="paper-abstract">
      We present a method for consistent lighting and shadows when animated 3D Gaussian Splatting (3DGS) avatars interact with 3DGS scenes or with dynamic objects inserted into otherwise static scenes. Our key contribution is Deep Gaussian Shadow Maps (DGSM), a modern analogue of the classical shadow mapping algorithm tailored to the volumetric 3DGS representation. Building on the classic deep shadow mapping idea, we show that 3DGS admits closed form light accumulation along light rays, enabling volumetric shadow computation without meshing. For each estimated light, we tabulate transmittance over concentric radial shells and store them in octahedral atlases, which modern GPUs can sample in real time per query to attenuate affected scene Gaussians and thus cast and receive shadows consistently. To relight moving avatars, we approximate the local environment illumination with HDRI probes represented in a spherical harmonic (SH) basis and apply a fast per Gaussian radiance transfer, avoiding explicit BRDF estimation or offline optimization. We demonstrate environment consistent lighting for avatars from AvatarX and ActorsHQ, composited into ScanNet++, DL3DV, and SuperSplat scenes, and show interactions with inserted objects. Across single and multi avatar settings, DGSM and SH relighting operate fully in the volumetric 3DGS representation, yielding coherent shadows and relighting while avoiding meshing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09827v2">AHA! Animating Human Avatars in Diverse Scenes with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-04
      | 💬 Project page available at: https://miraymen.github.io/aha/
    </div>
    <details class="paper-abstract">
      We present a novel framework for animating humans in 3D scenes using 3D Gaussian Splatting (3DGS), a neural scene representation that has recently achieved state-of-the-art photorealistic results for novel-view synthesis but remains under-explored for human-scene animation and interaction. Unlike existing animation pipelines that use meshes or point clouds as the underlying 3D representation, our approach introduces the use of 3DGS as the 3D representation for animating humans in scenes. By representing humans and scenes as Gaussians, our approach allows geometry-consistent free-viewpoint rendering of humans interacting with 3D scenes. Our key insight is that rendering can be decoupled from motion synthesis, and each sub-problem can be addressed independently without the need for paired human-scene data. Central to our method is a Gaussian-aligned motion module that synthesizes motion without explicit scene geometry, using opacity-based cues and projected Gaussian structures to guide human placement and pose alignment. To ensure natural interactions, we further propose a human-scene Gaussian refinement optimization that enforces realistic contact and navigation. We evaluate our approach on scenes from Scannet++ and the SuperSplat library, and on avatars reconstructed from sparse and dense multi-view human capture. Finally, we demonstrate that our framework enables novel applications such as geometry-consistent free-viewpoint rendering of edited monocular RGB videos with newly animated humans, showcasing the unique advantages of 3DGS for monocular video-based human animation. To assess the full quality of our results, we encourage readers to view the supplementary material available at https://miraymen.github.io/aha/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15169v3">MeSS: City Mesh-Guided Outdoor Scene Generation with Cross-View Consistent Diffusion</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      Mesh models have become increasingly accessible for numerous cities; however, the lack of realistic textures restricts their application in virtual urban navigation and autonomous driving. To address this, this paper proposes MeSS (Meshbased Scene Synthesis) for generating high-quality, styleconsistent outdoor scenes with city mesh models serving as the geometric prior. While image and video diffusion models can leverage spatial layouts (such as depth maps or HD maps) as control conditions to generate street-level perspective views, they are not directly applicable to 3D scene generation. Video diffusion models excel at synthesizing consistent view sequences that depict scenes but often struggle to adhere to predefined camera paths or align accurately with rendered control videos. In contrast, image diffusion models, though unable to guarantee cross-view visual consistency, can produce more geometry-aligned results when combined with ControlNet. Building on this insight, our approach enhances image diffusion models by improving cross-view consistency. The pipeline comprises three key stages: first, we generate geometrically consistent sparse views using Cascaded Outpainting ControlNets; second, we propagate denser intermediate views via a component dubbed AGInpaint; and third, we globally eliminate visual inconsistencies (e.g., varying exposure) using the GCAlign module. Concurrently with generation, a 3D Gaussian Splatting (3DGS) scene is reconstructed by initializing Gaussian balls on the mesh surface. Our method outperforms existing approaches in both geometric alignment and generation quality. Once synthesized, the scene can be rendered in diverse styles through relighting and style transfer techniques. project page: https://albertchen98.github.io/mess/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00939v1">ShadowGS: Shadow-Aware 3D Gaussian Splatting for Satellite Imagery</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a novel paradigm for 3D reconstruction from satellite imagery. However, in multi-temporal satellite images, prevalent shadows exhibit significant inconsistencies due to varying illumination conditions. To address this, we propose ShadowGS, a novel framework based on 3DGS. It leverages a physics-based rendering equation from remote sensing, combined with an efficient ray marching technique, to precisely model geometrically consistent shadows while maintaining efficient rendering. Additionally, it effectively disentangles different illumination components and apparent attributes in the scene. Furthermore, we introduce a shadow consistency constraint that significantly enhances the geometric accuracy of 3D reconstruction. We also incorporate a novel shadow map prior to improve performance with sparse-view inputs. Extensive experiments demonstrate that ShadowGS outperforms current state-of-the-art methods in shadow decoupling accuracy, 3D reconstruction precision, and novel view synthesis quality, with only a few minutes of training. ShadowGS exhibits robust performance across various settings, including RGB, pansharpened, and sparse-view satellite inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01386v1">ParkGaussian: Surround-view 3D Gaussian Splatting for Autonomous Parking</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      Parking is a critical task for autonomous driving systems (ADS), with unique challenges in crowded parking slots and GPS-denied environments. However, existing works focus on 2D parking slot perception, mapping, and localization, 3D reconstruction remains underexplored, which is crucial for capturing complex spatial geometry in parking scenarios. Naively improving the visual quality of reconstructed parking scenes does not directly benefit autonomous parking, as the key entry point for parking is the slots perception module. To address these limitations, we curate the first benchmark named ParkRecon3D, specifically designed for parking scene reconstruction. It includes sensor data from four surround-view fisheye cameras with calibrated extrinsics and dense parking slot annotations. We then propose ParkGaussian, the first framework that integrates 3D Gaussian Splatting (3DGS) for parking scene reconstruction. To further improve the alignment between reconstruction and downstream parking slot detection, we introduce a slot-aware reconstruction strategy that leverages existing parking perception methods to enhance the synthesis quality of slot regions. Experiments on ParkRecon3D demonstrate that ParkGaussian achieves state-of-the-art reconstruction quality and better preserves perception consistency for downstream tasks. The code and dataset will be released at: https://github.com/wm-research/ParkGaussian
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.22324v2">AH-GS: Augmented 3D Gaussian Splatting for High-Frequency Detail Representation</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 need to revsie
    </div>
    <details class="paper-abstract">
      The 3D Gaussian Splatting (3D-GS) is a novel method for scene representation and view synthesis. Although Scaffold-GS achieves higher quality real-time rendering compared to the original 3D-GS, its fine-grained rendering of the scene is extremely dependent on adequate viewing angles. The spectral bias of neural network learning results in Scaffold-GS's poor ability to perceive and learn high-frequency information in the scene. In this work, we propose enhancing the manifold complexity of input features and using network-based feature map loss to improve the image reconstruction quality of 3D-GS models. We introduce AH-GS, which enables 3D Gaussians in structurally complex regions to obtain higher-frequency encodings, allowing the model to more effectively learn the high-frequency information of the scene. Additionally, we incorporate high-frequency reinforce loss to further enhance the model's ability to capture detailed frequency information. Our result demonstrates that our model significantly improves rendering fidelity, and in specific scenarios (e.g., MipNeRf360-garden), our method exceeds the rendering quality of Scaffold-GS in just 15K iterations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.12905v2">Matrix-free Second-order Optimization of Gaussian Splats with Residual Sampling</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is widely used for novel view synthesis due to its high rendering quality and fast inference time. However, 3DGS predominantly relies on first-order optimizers such as Adam, which leads to long training times. To address this limitation, we propose a novel second-order optimization strategy based on Levenberg-Marquardt (LM) and Conjugate Gradient (CG), which we specifically tailor towards Gaussian Splatting. Our key insight is that the Jacobian in 3DGS exhibits significant sparsity since each Gaussian affects only a limited number of pixels. We exploit this sparsity by proposing a matrix-free and GPU-parallelized LM optimization. To further improve its efficiency, we propose sampling strategies for both the camera views and loss function and, consequently, the normal equation, significantly reducing the computational complexity. In addition, we increase the convergence rate of the second-order approximation by introducing an effective heuristic to determine the learning rate that avoids the expensive computation cost of line search methods. As a result, our method achieves a $3\times$ speedup over standard LM and outperforms Adam by $~6\times$ when the Gaussian count is low while remaining competitive for moderate counts. Project Page: https://vcai.mpi-inf.mpg.de/projects/LM-IS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.14921v2">Stereo-GS: Multi-View Stereo Vision Model for Generalizable 3D Gaussian Splatting Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 ACM Multimedia 2025
    </div>
    <details class="paper-abstract">
      Generalizable 3D Gaussian Splatting reconstruction showcases advanced Image-to-3D content creation but requires substantial computational resources and large datasets, posing challenges to training models from scratch. Current methods usually entangle the prediction of 3D Gaussian geometry and appearance, which rely heavily on data-driven priors and result in slow regression speeds. To address this, we propose \method, a disentangled framework for efficient 3D Gaussian prediction. Our method extracts features from local image pairs using a stereo vision backbone and fuses them via global attention blocks. Dedicated point and Gaussian prediction heads generate multi-view point-maps for geometry and Gaussian features for appearance, combined as GS-maps to represent the 3DGS object. A refinement network enhances these GS-maps for high-quality reconstruction. Unlike existing methods that depend on camera parameters, our approach achieves pose-free 3D reconstruction, improving robustness and practicality. By reducing resource demands while maintaining high-quality outputs, \method provides an efficient, scalable solution for real-world 3D content generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00285v1">SV-GS: Sparse View 4D Reconstruction with Skeleton-Driven Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      Reconstructing a dynamic target moving over a large area is challenging. Standard approaches for dynamic object reconstruction require dense coverage in both the viewing space and the temporal dimension, typically relying on multi-view videos captured at each time step. However, such setups are only possible in constrained environments. In real-world scenarios, observations are often sparse over time and captured sparsely from diverse viewpoints (e.g., from security cameras), making dynamic reconstruction highly ill-posed. We present SV-GS, a framework that simultaneously estimates a deformation model and the object's motion over time under sparse observations. To initialize SV-GS, we leverage a rough skeleton graph and an initial static reconstruction as inputs to guide motion estimation. (Later, we show that this input requirement can be relaxed.) Our method optimizes a skeleton-driven deformation field composed of a coarse skeleton joint pose estimator and a module for fine-grained deformations. By making only the joint pose estimator time-dependent, our model enables smooth motion interpolation while preserving learned geometric details. Experiments on synthetic datasets show that our method outperforms existing approaches under sparse observations by up to 34% in PSNR, and achieves comparable performance to dense monocular video methods on real-world datasets despite using significantly fewer frames. Moreover, we demonstrate that the input initial static reconstruction can be replaced by a diffusion-based generative prior, making our method more practical for real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07743v2">UltraGS: Real-Time Physically-Decoupled Gaussian Splatting for Ultrasound Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Ultrasound imaging is a cornerstone of non-invasive clinical diagnostics, yet its limited field of view poses challenges for novel view synthesis. We present UltraGS, a real-time framework that adapts Gaussian Splatting to sensorless ultrasound imaging by integrating explicit radiance fields with lightweight, physics-inspired acoustic modeling. UltraGS employs depth-aware Gaussian primitives with learnable fields of view to improve geometric consistency under unconstrained probe motion, and introduces PD Rendering, a differentiable acoustic operator that combines low-order spherical harmonics with first-order wave effects for efficient intensity synthesis. We further present a clinical ultrasound dataset acquired under real-world scanning protocols. Extensive evaluations across three datasets demonstrate that UltraGS establishes a new performance-efficiency frontier, achieving state-of-the-art results in PSNR (up to 29.55) and SSIM (up to 0.89) while achieving real-time synthesis at 64.69 fps on a single GPU. The code and dataset are open-sourced at: https://github.com/Bean-Young/UltraGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.09977v3">A Survey on 3D Gaussian Splatting Applications: Segmentation, Editing, and Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 GitHub Repo: https://github.com/heshuting555/Awesome-3DGS-Applications
    </div>
    <details class="paper-abstract">
      In the context of novel view synthesis, 3D Gaussian Splatting (3DGS) has recently emerged as an efficient and competitive counterpart to Neural Radiance Field (NeRF), enabling high-fidelity photorealistic rendering in real time. Beyond novel view synthesis, the explicit and compact nature of 3DGS enables a wide range of downstream applications that require geometric and semantic understanding. This survey provides a comprehensive overview of recent progress in 3DGS applications. It first introduces 2D foundation models that support semantic understanding and control in 3DGS applications, followed by a review of NeRF-based methods that inform their 3DGS counterparts. We then categorize 3DGS applications into three foundational tasks: segmentation, editing, and generation, alongside additional functional applications built upon or tightly coupled with these foundational capabilities. For each, we summarize representative methods, supervision strategies, and learning paradigms, highlighting shared design principles and emerging trends. Commonly used datasets and evaluation protocols are also summarized, along with comparative analyses of recent methods across public benchmarks. To support ongoing research and development, a continually updated repository of papers, code, and resources is maintained at https://github.com/heshuting555/Awesome-3DGS-Applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00913v1">Clean-GS: Semantic Mask-Guided Pruning for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting produces high-quality scene reconstructions but generates hundreds of thousands of spurious Gaussians (floaters) scattered throughout the environment. These artifacts obscure objects of interest and inflate model sizes, hindering deployment in bandwidth-constrained applications. We present Clean-GS, a method for removing background clutter and floaters from 3DGS reconstructions using sparse semantic masks. Our approach combines whitelist-based spatial filtering with color-guided validation and outlier removal to achieve 60-80\% model compression while preserving object quality. Unlike existing 3DGS pruning methods that rely on global importance metrics, Clean-GS uses semantic information from as few as 3 segmentation masks (1\% of views) to identify and remove Gaussians not belonging to the target object. Our multi-stage approach consisting of (1) whitelist filtering via projection to masked regions, (2) depth-buffered color validation, and (3) neighbor-based outlier removal isolates monuments and objects from complex outdoor scenes. Experiments on Tanks and Temples show that Clean-GS reduces file sizes from 125MB to 47MB while maintaining rendering quality, making 3DGS models practical for web deployment and AR/VR applications. Our code is available at https://github.com/smlab-niser/clean-gs
    </details>
</div>
