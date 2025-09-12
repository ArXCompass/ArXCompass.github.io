# gaussian splatting - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06179v3">ForestSplats: Deformable transient field for Gaussian Splatting in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3D-GS) has emerged, showing real-time rendering speeds and high-quality results in static scenes. Although 3D-GS shows effectiveness in static scenes, their performance significantly degrades in real-world environments due to transient objects, lighting variations, and diverse levels of occlusion. To tackle this, existing methods estimate occluders or transient elements by leveraging pre-trained models or integrating additional transient field pipelines. However, these methods still suffer from two defects: 1) Using semantic features from the Vision Foundation model (VFM) causes additional computational costs. 2) The transient field requires significant memory to handle transient elements with per-view Gaussians and struggles to define clear boundaries for occluders, solely relying on photometric errors. To address these problems, we propose ForestSplats, a novel approach that leverages the deformable transient field and a superpixel-aware mask to efficiently represent transient elements in the 2D scene across unconstrained image collections and effectively decompose static scenes from transient distractors without VFM. We designed the transient field to be deformable, capturing per-view transient elements. Furthermore, we introduce a superpixel-aware mask that clearly defines the boundaries of occluders by considering photometric errors and superpixels. Additionally, we propose uncertainty-aware densification to avoid generating Gaussians within the boundaries of occluders during densification. Through extensive experiments across several benchmark datasets, we demonstrate that ForestSplats outperforms existing methods without VFM and shows significant memory efficiency in representing transient elements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06685v3">VIM-GS: Visual-Inertial Monocular Gaussian Splatting via Object-level Guidance in Large Scenes</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 Withdrawn due to an error in the author list & incomplete experimental results
    </div>
    <details class="paper-abstract">
      VIM-GS is a Gaussian Splatting (GS) framework using monocular images for novel-view synthesis (NVS) in large scenes. GS typically requires accurate depth to initiate Gaussian ellipsoids using RGB-D/stereo cameras. Their limited depth sensing range makes it difficult for GS to work in large scenes. Monocular images, however, lack depth to guide the learning and lead to inferior NVS results. Although large foundation models (LFMs) for monocular depth estimation are available, they suffer from cross-frame inconsistency, inaccuracy for distant scenes, and ambiguity in deceptive texture cues. This paper aims to generate dense, accurate depth images from monocular RGB inputs for high-definite GS rendering. The key idea is to leverage the accurate but sparse depth from visual-inertial Structure-from-Motion (SfM) to refine the dense but coarse depth from LFMs. To bridge the sparse input and dense output, we propose an object-segmented depth propagation algorithm that renders the depth of pixels of structured objects. Then we develop a dynamic depth refinement module to handle the crippled SfM depth of dynamic objects and refine the coarse LFM depth. Experiments using public and customized datasets demonstrate the superior rendering quality of VIM-GS in large scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10546v2">The Oxford Spires Dataset: Benchmarking Large-Scale LiDAR-Visual Localisation, Reconstruction and Radiance Field Methods</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 Accepted by IJRR. Website: https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/
    </div>
    <details class="paper-abstract">
      This paper introduces a large-scale multi-modal dataset captured in and around well-known landmarks in Oxford using a custom-built multi-sensor perception unit as well as a millimetre-accurate map from a Terrestrial LiDAR Scanner (TLS). The perception unit includes three synchronised global shutter colour cameras, an automotive 3D LiDAR scanner, and an inertial sensor - all precisely calibrated. We also establish benchmarks for tasks involving localisation, reconstruction, and novel-view synthesis, which enable the evaluation of Simultaneous Localisation and Mapping (SLAM) methods, Structure-from-Motion (SfM) and Multi-view Stereo (MVS) methods as well as radiance field methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting. To evaluate 3D reconstruction the TLS 3D models are used as ground truth. Localisation ground truth is computed by registering the mobile LiDAR scans to the TLS 3D models. Radiance field methods are evaluated not only with poses sampled from the input trajectory, but also from viewpoints that are from trajectories which are distant from the training poses. Our evaluation demonstrates a key limitation of state-of-the-art radiance field methods: we show that they tend to overfit to the training poses/images and do not generalise well to out-of-sequence poses. They also underperform in 3D reconstruction compared to MVS systems using the same visual inputs. Our dataset and benchmarks are intended to facilitate better integration of radiance field methods and SLAM systems. The raw and processed data, along with software for parsing and evaluation, can be accessed at https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07809v1">SplatFill: 3D Scene Inpainting via Depth-Guided Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has enabled the creation of highly realistic 3D scene representations from sets of multi-view images. However, inpainting missing regions, whether due to occlusion or scene editing, remains a challenging task, often leading to blurry details, artifacts, and inconsistent geometry. In this work, we introduce SplatFill, a novel depth-guided approach for 3DGS scene inpainting that achieves state-of-the-art perceptual quality and improved efficiency. Our method combines two key ideas: (1) joint depth-based and object-based supervision to ensure inpainted Gaussians are accurately placed in 3D space and aligned with surrounding geometry, and (2) we propose a consistency-aware refinement scheme that selectively identifies and corrects inconsistent regions without disrupting the rest of the scene. Evaluations on the SPIn-NeRF dataset demonstrate that SplatFill not only surpasses existing NeRF-based and 3DGS-based inpainting methods in visual fidelity but also reduces training time by 24.5%. Qualitative results show our method delivers sharper details, fewer artifacts, and greater coherence across challenging viewpoints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07774v1">HairGS: Hair Strand Reconstruction based on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 This is the arXiv preprint of the paper "Hair Strand Reconstruction based on 3D Gaussian Splatting" published at BMVC 2025. Project website: https://yimin-pan.github.io/hair-gs/
    </div>
    <details class="paper-abstract">
      Human hair reconstruction is a challenging problem in computer vision, with growing importance for applications in virtual reality and digital human modeling. Recent advances in 3D Gaussians Splatting (3DGS) provide efficient and explicit scene representations that naturally align with the structure of hair strands. In this work, we extend the 3DGS framework to enable strand-level hair geometry reconstruction from multi-view images. Our multi-stage pipeline first reconstructs detailed hair geometry using a differentiable Gaussian rasterizer, then merges individual Gaussian segments into coherent strands through a novel merging scheme, and finally refines and grows the strands under photometric supervision. While existing methods typically evaluate reconstruction quality at the geometric level, they often neglect the connectivity and topology of hair strands. To address this, we propose a new evaluation metric that serves as a proxy for assessing topological accuracy in strand reconstruction. Extensive experiments on both synthetic and real-world datasets demonstrate that our method robustly handles a wide range of hairstyles and achieves efficient reconstruction, typically completing within one hour. The project page can be found at: https://yimin-pan.github.io/hair-gs/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05752v2">PINGS: Gaussian Splatting Meets Distance Fields within a Point-Based Implicit Neural Map</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 15 pages, 8 figures, presented at RSS 2025
    </div>
    <details class="paper-abstract">
      Robots benefit from high-fidelity reconstructions of their environment, which should be geometrically accurate and photorealistic to support downstream tasks. While this can be achieved by building distance fields from range sensors and radiance fields from cameras, realising scalable incremental mapping of both fields consistently and at the same time with high quality is challenging. In this paper, we propose a novel map representation that unifies a continuous signed distance field and a Gaussian splatting radiance field within an elastic and compact point-based implicit neural map. By enforcing geometric consistency between these fields, we achieve mutual improvements by exploiting both modalities. We present a novel LiDAR-visual SLAM system called PINGS using the proposed map representation and evaluate it on several challenging large-scale datasets. Experimental results demonstrate that PINGS can incrementally build globally consistent distance and radiance fields encoded with a compact set of neural points. Compared to state-of-the-art methods, PINGS achieves superior photometric and geometric rendering at novel views by constraining the radiance field with the distance field. Furthermore, by utilizing dense photometric cues and multi-view consistency from the radiance field, PINGS produces more accurate distance fields, leading to improved odometry estimation and mesh reconstruction. We also provide an open-source implementation of PING at: https://github.com/PRBonn/PINGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07493v1">DiGS: Accurate and Complete Surface Reconstruction from 3D Gaussians via Direct SDF Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as a powerful paradigm for photorealistic view synthesis, representing scenes with spatially distributed Gaussian primitives. While highly effective for rendering, achieving accurate and complete surface reconstruction remains challenging due to the unstructured nature of the representation and the absence of explicit geometric supervision. In this work, we propose DiGS, a unified framework that embeds Signed Distance Field (SDF) learning directly into the 3DGS pipeline, thereby enforcing strong and interpretable surface priors. By associating each Gaussian with a learnable SDF value, DiGS explicitly aligns primitives with underlying geometry and improves cross-view consistency. To further ensure dense and coherent coverage, we design a geometry-guided grid growth strategy that adaptively distributes Gaussians along geometry-consistent regions under a multi-scale hierarchy. Extensive experiments on standard benchmarks, including DTU, Mip-NeRF 360, and Tanks& Temples, demonstrate that DiGS consistently improves reconstruction accuracy and completeness while retaining high rendering fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07435v1">DreamLifting: A Plug-in Module Lifting MV Diffusion Models for 3D Asset Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 14 pages, 7 figures, project page: https://zx-yin.github.io/dreamlifting/
    </div>
    <details class="paper-abstract">
      The labor- and experience-intensive creation of 3D assets with physically based rendering (PBR) materials demands an autonomous 3D asset creation pipeline. However, most existing 3D generation methods focus on geometry modeling, either baking textures into simple vertex colors or leaving texture synthesis to post-processing with image diffusion models. To achieve end-to-end PBR-ready 3D asset generation, we present Lightweight Gaussian Asset Adapter (LGAA), a novel framework that unifies the modeling of geometry and PBR materials by exploiting multi-view (MV) diffusion priors from a novel perspective. The LGAA features a modular design with three components. Specifically, the LGAA Wrapper reuses and adapts network layers from MV diffusion models, which encapsulate knowledge acquired from billions of images, enabling better convergence in a data-efficient manner. To incorporate multiple diffusion priors for geometry and PBR synthesis, the LGAA Switcher aligns multiple LGAA Wrapper layers encapsulating different knowledge. Then, a tamed variational autoencoder (VAE), termed LGAA Decoder, is designed to predict 2D Gaussian Splatting (2DGS) with PBR channels. Finally, we introduce a dedicated post-processing procedure to effectively extract high-quality, relightable mesh assets from the resulting 2DGS. Extensive quantitative and qualitative experiments demonstrate the superior performance of LGAA with both text-and image-conditioned MV diffusion models. Additionally, the modular design enables flexible incorporation of multiple diffusion priors, and the knowledge-preserving scheme leads to efficient convergence trained on merely 69k multi-view instances. Our code, pre-trained weights, and the dataset used will be publicly available via our project page: https://zx-yin.github.io/dreamlifting/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06685v2">VIM-GS: Visual-Inertial Monocular Gaussian Splatting via Object-level Guidance in Large Scenes</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Withdrawn due to an error in the author list & incomplete experimental results
    </div>
    <details class="paper-abstract">
      VIM-GS is a Gaussian Splatting (GS) framework using monocular images for novel-view synthesis (NVS) in large scenes. GS typically requires accurate depth to initiate Gaussian ellipsoids using RGB-D/stereo cameras. Their limited depth sensing range makes it difficult for GS to work in large scenes. Monocular images, however, lack depth to guide the learning and lead to inferior NVS results. Although large foundation models (LFMs) for monocular depth estimation are available, they suffer from cross-frame inconsistency, inaccuracy for distant scenes, and ambiguity in deceptive texture cues. This paper aims to generate dense, accurate depth images from monocular RGB inputs for high-definite GS rendering. The key idea is to leverage the accurate but sparse depth from visual-inertial Structure-from-Motion (SfM) to refine the dense but coarse depth from LFMs. To bridge the sparse input and dense output, we propose an object-segmented depth propagation algorithm that renders the depth of pixels of structured objects. Then we develop a dynamic depth refinement module to handle the crippled SfM depth of dynamic objects and refine the coarse LFM depth. Experiments using public and customized datasets demonstrate the superior rendering quality of VIM-GS in large scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06685v1">VIM-GS: Visual-Inertial Monocular Gaussian Splatting via Object-level Guidance in Large Scenes</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      VIM-GS is a Gaussian Splatting (GS) framework using monocular images for novel-view synthesis (NVS) in large scenes. GS typically requires accurate depth to initiate Gaussian ellipsoids using RGB-D/stereo cameras. Their limited depth sensing range makes it difficult for GS to work in large scenes. Monocular images, however, lack depth to guide the learning and lead to inferior NVS results. Although large foundation models (LFMs) for monocular depth estimation are available, they suffer from cross-frame inconsistency, inaccuracy for distant scenes, and ambiguity in deceptive texture cues. This paper aims to generate dense, accurate depth images from monocular RGB inputs for high-definite GS rendering. The key idea is to leverage the accurate but sparse depth from visual-inertial Structure-from-Motion (SfM) to refine the dense but coarse depth from LFMs. To bridge the sparse input and dense output, we propose an object-segmented depth propagation algorithm that renders the depth of pixels of structured objects. Then we develop a dynamic depth refinement module to handle the crippled SfM depth of dynamic objects and refine the coarse LFM depth. Experiments using public and customized datasets demonstrate the superior rendering quality of VIM-GS in large scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06433v1">Real-time Photorealistic Mapping for Situational Awareness in Robot Teleoperation</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Achieving efficient remote teleoperation is particularly challenging in unknown environments, as the teleoperator must rapidly build an understanding of the site's layout. Online 3D mapping is a proven strategy to tackle this challenge, as it enables the teleoperator to progressively explore the site from multiple perspectives. However, traditional online map-based teleoperation systems struggle to generate visually accurate 3D maps in real-time due to the high computational cost involved, leading to poor teleoperation performances. In this work, we propose a solution to improve teleoperation efficiency in unknown environments. Our approach proposes a novel, modular and efficient GPU-based integration between recent advancement in gaussian splatting SLAM and existing online map-based teleoperation systems. We compare the proposed solution against state-of-the-art teleoperation systems and validate its performances through real-world experiments using an aerial vehicle. The results show significant improvements in decision-making speed and more accurate interaction with the environment, leading to greater teleoperation efficiency. In doing so, our system enhances remote teleoperation by seamlessly integrating photorealistic mapping generation with real-time performances, enabling effective teleoperation in unfamiliar environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06400v1">3DOF+Quantization: 3DGS quantization for large scenes with limited Degrees of Freedom</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a major breakthrough in 3D scene reconstruction. With a number of views of a given object or scene, the algorithm trains a model composed of 3D gaussians, which enables the production of novel views from arbitrary points of view. This freedom of movement is referred to as 6DoF for 6 degrees of freedom: a view is produced for any position (3 degrees), orientation of camera (3 other degrees). On large scenes, though, the input views are acquired from a limited zone in space, and the reconstruction is valuable for novel views from the same zone, even if the scene itself is almost unlimited in size. We refer to this particular case as 3DoF+, meaning that the 3 degrees of freedom of camera position are limited to small offsets around the central position. Considering the problem of coordinate quantization, the impact of position error on the projection error in pixels is studied. It is shown that the projection error is proportional to the squared inverse distance of the point being projected. Consequently, a new quantization scheme based on spherical coordinates is proposed. Rate-distortion performance of the proposed method are illustrated on the well-known Garden scene.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11854v2">ComplicitSplat: Downstream Models are Vulnerable to Blackbox Attacks by 3D Gaussian Splat Camouflages</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 7 pages, 6 figures
    </div>
    <details class="paper-abstract">
      As 3D Gaussian Splatting (3DGS) gains rapid adoption in safety-critical tasks for efficient novel-view synthesis from static images, how might an adversary tamper images to cause harm? We introduce ComplicitSplat, the first attack that exploits standard 3DGS shading methods to create viewpoint-specific camouflage - colors and textures that change with viewing angle - to embed adversarial content in scene objects that are visible only from specific viewpoints and without requiring access to model architecture or weights. Our extensive experiments show that ComplicitSplat generalizes to successfully attack a variety of popular detector - both single-stage, multi-stage, and transformer-based models on both real-world capture of physical objects and synthetic scenes. To our knowledge, this is the first black-box attack on downstream object detectors using 3DGS, exposing a novel safety risk for applications like autonomous navigation and other mission-critical robotic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07021v1">MEGS$^{2}$: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 14 pages, 4 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a dominant novel-view synthesis technique, but its high memory consumption severely limits its applicability on edge devices. A growing number of 3DGS compression methods have been proposed to make 3DGS more efficient, yet most only focus on storage compression and fail to address the critical bottleneck of rendering memory. To address this problem, we introduce MEGS$^{2}$, a novel memory-efficient framework that tackles this challenge by jointly optimizing two key factors: the total primitive number and the parameters per primitive, achieving unprecedented memory compression. Specifically, we replace the memory-intensive spherical harmonics with lightweight arbitrarily-oriented spherical Gaussian lobes as our color representations. More importantly, we propose a unified soft pruning framework that models primitive-number and lobe-number pruning as a single constrained optimization problem. Experiments show that MEGS$^{2}$ achieves a 50% static VRAM reduction and a 40% rendering VRAM reduction compared to existing methods, while maintaining comparable rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10906v2">ShapeSplat: A Large-scale Dataset of Gaussian Splats and Their Self-Supervised Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-09-06
      | 💬 Accepted as 3DV'25 Oral, project page: https://unique1i.github.io/ShapeSplat_webpage/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become the de facto method of 3D representation in many vision tasks. This calls for the 3D understanding directly in this representation space. To facilitate the research in this direction, we first build ShapeSplat, a large-scale dataset of 3DGS using the commonly used ShapeNet, ModelNet and Objaverse datasets. Our dataset ShapeSplat consists of 206K objects spanning over 87 unique categories, whose labels are in accordance with the respective datasets. The creation of this dataset utilized the compute equivalent of 3.8 GPU years on a TITAN XP GPU. We utilize our dataset for unsupervised pretraining and supervised finetuning for classification and segmentation tasks. To this end, we introduce Gaussian-MAE, which highlights the unique benefits of representation learning from Gaussian parameters. Through exhaustive experiments, we provide several valuable insights. In particular, we show that (1) the distribution of the optimized GS centroids significantly differs from the uniformly sampled point cloud (used for initialization) counterpart; (2) this change in distribution results in degradation in classification but improvement in segmentation tasks when using only the centroids; (3) to leverage additional Gaussian parameters, we propose Gaussian feature grouping in a normalized feature space, along with splats pooling layer, offering a tailored solution to effectively group and embed similar Gaussians, which leads to notable improvement in finetuning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05216v1">Toward Distributed 3D Gaussian Splatting for High-Resolution Isosurface Visualization</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      We present a multi-GPU extension of the 3D Gaussian Splatting (3D-GS) pipeline for scientific visualization. Building on previous work that demonstrated high-fidelity isosurface reconstruction using Gaussian primitives, we incorporate a multi-GPU training backend adapted from Grendel-GS to enable scalable processing of large datasets. By distributing optimization across GPUs, our method improves training throughput and supports high-resolution reconstructions that exceed single-GPU capacity. In our experiments, the system achieves a 5.6X speedup on the Kingsnake dataset (4M Gaussians) using four GPUs compared to a single-GPU baseline, and successfully trains the Miranda dataset (18M Gaussians) that is an infeasible task on a single A100 GPU. This work lays the groundwork for integrating 3D-GS into HPC-based scientific workflows, enabling real-time post hoc and in situ visualization of complex simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05075v1">GeoSplat: A Deep Dive into Geometry-Constrained Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      A few recent works explored incorporating geometric priors to regularize the optimization of Gaussian splatting, further improving its performance. However, those early studies mainly focused on the use of low-order geometric priors (e.g., normal vector), and they are also unreliably estimated by noise-sensitive methods, like local principal component analysis. To address their limitations, we first present GeoSplat, a general geometry-constrained optimization framework that exploits both first-order and second-order geometric quantities to improve the entire training pipeline of Gaussian splatting, including Gaussian initialization, gradient update, and densification. As an example, we initialize the scales of 3D Gaussian primitives in terms of principal curvatures, leading to a better coverage of the object surface than random initialization. Secondly, based on certain geometric structures (e.g., local manifold), we introduce efficient and noise-robust estimation methods that provide dynamic geometric priors for our framework. We conduct extensive experiments on multiple datasets for novel view synthesis, showing that our framework: GeoSplat, significantly improves the performance of Gaussian splatting and outperforms previous baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04859v1">CoRe-GS: Coarse-to-Refined Gaussian Splatting with Semantic Object Focus</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Mobile reconstruction for autonomous aerial robotics holds strong potential for critical applications such as tele-guidance and disaster response. These tasks demand both accurate 3D reconstruction and fast scene processing. Instead of reconstructing the entire scene in detail, it is often more efficient to focus on specific objects, i.e., points of interest (PoIs). Mobile robots equipped with advanced sensing can usually detect these early during data acquisition or preliminary analysis, reducing the need for full-scene optimization. Gaussian Splatting (GS) has recently shown promise in delivering high-quality novel view synthesis and 3D representation by an incremental learning process. Extending GS with scene editing, semantics adds useful per-splat features to isolate objects effectively. Semantic 3D Gaussian editing can already be achieved before the full training cycle is completed, reducing the overall training time. Moreover, the semantically relevant area, the PoI, is usually already known during capturing. To balance high-quality reconstruction with reduced training time, we propose CoRe-GS. We first generate a coarse segmentation-ready scene with semantic GS and then refine it for the semantic object using our novel color-based effective filtering for effective object isolation. This is speeding up the training process to be about a quarter less than a full training cycle for semantic GS. We evaluate our approach on two datasets, SCRREAM (real-world, outdoor) and NeRDS 360 (synthetic, indoor), showing reduced runtime and higher novel-view-synthesis quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17422v3">Multimodal LLM Guided Exploration and Active Mapping using Fisher Information</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 ICCV 2025
    </div>
    <details class="paper-abstract">
      We present an active mapping system that plans for both long-horizon exploration goals and short-term actions using a 3D Gaussian Splatting (3DGS) representation. Existing methods either do not take advantage of recent developments in multimodal Large Language Models (LLM) or do not consider challenges in localization uncertainty, which is critical in embodied agents. We propose employing multimodal LLMs for long-horizon planning in conjunction with detailed motion planning using our information-based objective. By leveraging high-quality view synthesis from our 3DGS representation, our method employs a multimodal LLM as a zero-shot planner for long-horizon exploration goals from the semantic perspective. We also introduce an uncertainty-aware path proposal and selection algorithm that balances the dual objectives of maximizing the information gain for the environment while minimizing the cost of localization errors. Experiments conducted on the Gibson and Habitat-Matterport 3D datasets demonstrate state-of-the-art results of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14014v2">Online 3D Gaussian Splatting Modeling with Novel View Selection</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      This study addresses the challenge of generating online 3D Gaussian Splatting (3DGS) models from RGB-only frames. Previous studies have employed dense SLAM techniques to estimate 3D scenes from keyframes for 3DGS model construction. However, these methods are limited by their reliance solely on keyframes, which are insufficient to capture an entire scene, resulting in incomplete reconstructions. Moreover, building a generalizable model requires incorporating frames from diverse viewpoints to achieve broader scene coverage. However, online processing restricts the use of many frames or extensive training iterations. Therefore, we propose a novel method for high-quality 3DGS modeling that improves model completeness through adaptive view selection. By analyzing reconstruction quality online, our approach selects optimal non-keyframes for additional training. By integrating both keyframes and selected non-keyframes, the method refines incomplete regions from diverse viewpoints, significantly enhancing completeness. We also present a framework that incorporates an online multi-view stereo approach, ensuring consistency in 3D information throughout the 3DGS modeling process. Experimental results demonstrate that our method outperforms state-of-the-art methods, delivering exceptional performance in complex outdoor scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05515v1">Visibility-Aware Language Aggregation for Open-Vocabulary Segmentation in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Recently, distilling open-vocabulary language features from 2D images into 3D Gaussians has attracted significant attention. Although existing methods achieve impressive language-based interactions of 3D scenes, we observe two fundamental issues: background Gaussians contributing negligibly to a rendered pixel get the same feature as the dominant foreground ones, and multi-view inconsistencies due to view-specific noise in language embeddings. We introduce Visibility-Aware Language Aggregation (VALA), a lightweight yet effective method that computes marginal contributions for each ray and applies a visibility-aware gate to retain only visible Gaussians. Moreover, we propose a streaming weighted geometric median in cosine space to merge noisy multi-view features. Our method yields a robust, view-consistent language feature embedding in a fast and memory-efficient manner. VALA improves open-vocabulary localization and segmentation across reference datasets, consistently surpassing existing works.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04379v1">SSGaussian: Semantic-Aware and Structure-Preserving 3D Style Transfer</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Recent advancements in neural representations, such as Neural Radiance Fields and 3D Gaussian Splatting, have increased interest in applying style transfer to 3D scenes. While existing methods can transfer style patterns onto 3D-consistent neural representations, they struggle to effectively extract and transfer high-level style semantics from the reference style image. Additionally, the stylized results often lack structural clarity and separation, making it difficult to distinguish between different instances or objects within the 3D scene. To address these limitations, we propose a novel 3D style transfer pipeline that effectively integrates prior knowledge from pretrained 2D diffusion models. Our pipeline consists of two key stages: First, we leverage diffusion priors to generate stylized renderings of key viewpoints. Then, we transfer the stylized key views onto the 3D representation. This process incorporates two innovative designs. The first is cross-view style alignment, which inserts cross-view attention into the last upsampling block of the UNet, allowing feature interactions across multiple key views. This ensures that the diffusion model generates stylized key views that maintain both style fidelity and instance-level consistency. The second is instance-level style transfer, which effectively leverages instance-level consistency across stylized key views and transfers it onto the 3D representation. This results in a more structured, visually coherent, and artistically enriched stylization. Extensive qualitative and quantitative experiments demonstrate that our 3D style transfer pipeline significantly outperforms state-of-the-art methods across a wide range of scenes, from forward-facing to challenging 360-degree environments. Visit our project page https://jm-xu.github.io/SSGaussian for immersive visualization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06897v3">ActiveGAMER: Active GAussian Mapping through Efficient Rendering</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 Accepted to CVPR2025. Project page: https://oppo-us-research.github.io/ActiveGAMER-website/. Code: https://github.com/oppo-us-research/ActiveGAMER
    </div>
    <details class="paper-abstract">
      We introduce ActiveGAMER, an active mapping system that utilizes 3D Gaussian Splatting (3DGS) to achieve high-quality, real-time scene mapping and exploration. Unlike traditional NeRF-based methods, which are computationally demanding and restrict active mapping performance, our approach leverages the efficient rendering capabilities of 3DGS, allowing effective and efficient exploration in complex environments. The core of our system is a rendering-based information gain module that dynamically identifies the most informative viewpoints for next-best-view planning, enhancing both geometric and photometric reconstruction accuracy. ActiveGAMER also integrates a carefully balanced framework, combining coarse-to-fine exploration, post-refinement, and a global-local keyframe selection strategy to maximize reconstruction completeness and fidelity. Our system autonomously explores and reconstructs environments with state-of-the-art geometric and photometric accuracy and completeness, significantly surpassing existing approaches in both aspects. Extensive evaluations on benchmark datasets such as Replica and MP3D highlight ActiveGAMER's effectiveness in active mapping tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06269v3">BayesSDF: Surface-Based Laplacian Uncertainty Estimation for 3D Geometry with Neural Signed Distance Fields</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 ICCV 2025 Workshops (11 Pages, 6 Figures, 2 Tables)
    </div>
    <details class="paper-abstract">
      Accurate surface estimation is critical for downstream tasks in scientific simulation, and quantifying uncertainty in implicit neural 3D representations still remains a substantial challenge due to computational inefficiencies, scalability issues, and geometric inconsistencies. However, current neural implicit surface models do not offer a principled way to quantify uncertainty, limiting their reliability in real-world applications. Inspired by recent probabilistic rendering approaches, we introduce BayesSDF, a novel probabilistic framework for uncertainty estimation in neural implicit 3D representations. Unlike radiance-based models such as Neural Radiance Fields (NeRF) or 3D Gaussian Splatting, Signed Distance Functions (SDFs) provide continuous, differentiable surface representations, making them especially well-suited for uncertainty-aware modeling. BayesSDF applies a Laplace approximation over SDF weights and derives Hessian-based metrics to estimate local geometric instability. We empirically demonstrate that these uncertainty estimates correlate strongly with surface reconstruction error across both synthetic and real-world benchmarks. By enabling surface-aware uncertainty quantification, BayesSDF lays the groundwork for more robust, interpretable, and actionable 3D perception systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00831v2">UPGS: Unified Pose-aware Gaussian Splatting for Dynamic Scene Deblurring</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic 3D scenes from monocular video has broad applications in AR/VR, robotics, and autonomous navigation, but often fails due to severe motion blur caused by camera and object motion. Existing methods commonly follow a two-step pipeline, where camera poses are first estimated and then 3D Gaussians are optimized. Since blurring artifacts usually undermine pose estimation, pose errors could be accumulated to produce inferior reconstruction results. To address this issue, we introduce a unified optimization framework by incorporating camera poses as learnable parameters complementary to 3DGS attributes for end-to-end optimization. Specifically, we recast camera and object motion as per-primitive SE(3) affine transformations on 3D Gaussians and formulate a unified optimization objective. For stable optimization, we introduce a three-stage training schedule that optimizes camera poses and Gaussians alternatively. Particularly, 3D Gaussians are first trained with poses being fixed, and then poses are optimized with 3D Gaussians being untouched. Finally, all learnable parameters are optimized together. Extensive experiments on the Stereo Blur dataset and challenging real-world sequences demonstrate that our method achieves significant gains in reconstruction quality and pose estimation accuracy over prior dynamic deblurring methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08624v2">Communication Efficient Robotic Mixed Reality with Gaussian Splatting Cross-Layer Optimization</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 14 pages, 18 figures, to appear in IEEE Transactions on Cognitive Communications and Networking
    </div>
    <details class="paper-abstract">
      Realizing low-cost communication in robotic mixed reality (RoboMR) systems presents a challenge, due to the necessity of uploading high-resolution images through wireless channels. This paper proposes Gaussian splatting (GS) RoboMR (GSMR), which enables the simulator to opportunistically render a photo-realistic view from the robot's pose by calling ``memory'' from a GS model, thus reducing the need for excessive image uploads. However, the GS model may involve discrepancies compared to the actual environments. To this end, a GS cross-layer optimization (GSCLO) framework is further proposed, which jointly optimizes content switching (i.e., deciding whether to upload image or not) and power allocation (i.e., adjusting to content profiles) across different frames by minimizing a newly derived GSMR loss function. The GSCLO problem is addressed by an accelerated penalty optimization (APO) algorithm that reduces computational complexity by over $10$x compared to traditional branch-and-bound and search algorithms. Moreover, variants of GSCLO are presented to achieve robust, low-power, and multi-robot GSMR. Extensive experiments demonstrate that the proposed GSMR paradigm and GSCLO method achieve significant improvements over existing benchmarks on both wheeled and legged robots in terms of diverse metrics in various scenarios. For the first time, it is found that RoboMR can be achieved with ultra-low communication costs, and mixture of data is useful for enhancing GS performance in dynamic scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00911v2">GS-TG: 3D Gaussian Splatting Accelerator with Tile Grouping for Reducing Redundant Sorting while Preserving Rasterization Efficiency</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 DAC 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D-GS) has emerged as a promising alternative to neural radiance fields (NeRF) as it offers high speed as well as high image quality in novel view synthesis. Despite these advancements, 3D-GS still struggles to meet the frames per second (FPS) demands of real-time applications. In this paper, we introduce GS-TG, a tile-grouping-based accelerator that enhances 3D-GS rendering speed by reducing redundant sorting operations and preserving rasterization efficiency. GS-TG addresses a critical trade-off issue in 3D-GS rendering: increasing the tile size effectively reduces redundant sorting operations, but it concurrently increases unnecessary rasterization computations. So, during sorting of the proposed approach, GS-TG groups small tiles (for making large tiles) to share sorting operations across tiles within each group, significantly reducing redundant computations. During rasterization, a bitmask assigned to each Gaussian identifies relevant small tiles, to enable efficient sharing of sorting results. Consequently, GS-TG enables sorting to be performed as if a large tile size is used by grouping tiles during the sorting stage, while allowing rasterization to proceed with the original small tiles by using bitmasks in the rasterization stage. GS-TG is a lossless method requiring no retraining or fine-tuning and it can be seamlessly integrated with previous 3D-GS optimization techniques. Experimental results show that GS-TG achieves an average speed-up of 1.54 times over state-of-the-art 3D-GS accelerators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03775v1">ContraGS: Codebook-Condensed and Trainable Gaussian Splatting for Fast, Memory-Efficient Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a state-of-art technique to model real-world scenes with high quality and real-time rendering. Typically, a higher quality representation can be achieved by using a large number of 3D Gaussians. However, using large 3D Gaussian counts significantly increases the GPU device memory for storing model parameters. A large model thus requires powerful GPUs with high memory capacities for training and has slower training/rendering latencies due to the inefficiencies of memory access and data movement. In this work, we introduce ContraGS, a method to enable training directly on compressed 3DGS representations without reducing the Gaussian Counts, and thus with a little loss in model quality. ContraGS leverages codebooks to compactly store a set of Gaussian parameter vectors throughout the training process, thereby significantly reducing memory consumption. While codebooks have been demonstrated to be highly effective at compressing fully trained 3DGS models, directly training using codebook representations is an unsolved challenge. ContraGS solves the problem of learning non-differentiable parameters in codebook-compressed representations by posing parameter estimation as a Bayesian inference problem. To this end, ContraGS provides a framework that effectively uses MCMC sampling to sample over a posterior distribution of these compressed representations. With ContraGS, we demonstrate that ContraGS significantly reduces the peak memory during training (on average 3.49X) and accelerated training and rendering (1.36X and 1.88X on average, respectively), while retraining close to state-of-art quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05740v2">Micro-splatting: Multistage Isotropy-informed Covariance Regularization Optimization for High-Fidelity 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 This work has been submitted to journal for potential publication
    </div>
    <details class="paper-abstract">
      High-fidelity 3D Gaussian Splatting methods excel at capturing fine textures but often overlook model compactness, resulting in massive splat counts, bloated memory, long training, and complex post-processing. We present Micro-Splatting: Two-Stage Adaptive Growth and Refinement, a unified, in-training pipeline that preserves visual detail while drastically reducing model complexity without any post-processing or auxiliary neural modules. In Stage I (Growth), we introduce a trace-based covariance regularization to maintain near-isotropic Gaussians, mitigating low-pass filtering in high-frequency regions and improving spherical-harmonic color fitting. We then apply gradient-guided adaptive densification that subdivides splats only in visually complex regions, leaving smooth areas sparse. In Stage II (Refinement), we prune low-impact splats using a simple opacity-scale importance score and merge redundant neighbors via lightweight spatial and feature thresholds, producing a lean yet detail-rich model. On four object-centric benchmarks, Micro-Splatting reduces splat count and model size by up to 60% and shortens training by 20%, while matching or surpassing state-of-the-art PSNR, SSIM, and LPIPS in real-time rendering. These results demonstrate that Micro-Splatting delivers both compactness and high fidelity in a single, efficient, end-to-end framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05254v3">CF3: Compact and Fast 3D Feature Fields</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 ICCV 2025, Project Page: https://jjoonii.github.io/cf3-website/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has begun incorporating rich information from 2D foundation models. However, most approaches rely on a bottom-up optimization process that treats raw 2D features as ground truth, incurring increased computational costs. We propose a top-down pipeline for constructing compact and fast 3D Gaussian feature fields, namely, CF3. We first perform a fast weighted fusion of multi-view 2D features with pre-trained Gaussians. This approach enables training a per-Gaussian autoencoder directly on the lifted features, instead of training autoencoders in the 2D domain. As a result, the autoencoder better aligns with the feature distribution. More importantly, we introduce an adaptive sparsification method that optimizes the Gaussian attributes of the feature field while pruning and merging the redundant Gaussians, constructing an efficient representation with preserved geometric details. Our approach achieves a competitive 3D feature field using as little as 5% of the Gaussians compared to Feature-3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18677v2">Reconstructing Tornadoes in 3D with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Accurately reconstructing the 3D structure of tornadoes is critically important for understanding and preparing for this highly destructive weather phenomenon. While modern 3D scene reconstruction techniques, such as 3D Gaussian splatting (3DGS), could provide a valuable tool for reconstructing the 3D structure of tornados, at present we are critically lacking a controlled tornado dataset with which to develop and validate these tools. In this work we capture and release a novel multiview dataset of a small lab-based tornado. We demonstrate one can effectively reconstruct and visualize the 3D structure of this tornado using 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02232v1">Efficient Geometry Compression and Communication for 3D Gaussian Splatting Point Clouds</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 8 pages,5 figures
    </div>
    <details class="paper-abstract">
      Storage and transmission challenges in dynamic 3D scene representation based on the i3DV platform, With increasing scene complexity, the explosive growth of 3D Gaussian data volume causes excessive storage space occupancy. To address this issue, we propose adopting the AVS PCRM reference software for efficient compression of Gaussian point cloud geometry data. The strategy deeply integrates the advanced encoding capabilities of AVS PCRM into the i3DV platform, forming technical complementarity with the original rate-distortion optimization mechanism based on binary hash tables. On one hand, the hash table efficiently caches inter-frame Gaussian point transformation relationships, which allows for high-fidelity transmission within a 40 Mbps bandwidth constraint. On the other hand, AVS PCRM performs precise compression on geometry data. Experimental results demonstrate that the joint framework maintains the advantages of fast rendering and high-quality synthesis in 3D Gaussian technology while achieving significant 10\%-25\% bitrate savings on universal test sets. It provides a superior rate-distortion tradeoff solution for the storage, transmission, and interaction of 3D volumetric video.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.05220v3">StylizedGS: Controllable Stylization for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 [TPAMI 2025] Project Page: https://kristen-z.github.io/stylizedgs/
    </div>
    <details class="paper-abstract">
      As XR technology continues to advance rapidly, 3D generation and editing are increasingly crucial. Among these, stylization plays a key role in enhancing the appearance of 3D models. By utilizing stylization, users can achieve consistent artistic effects in 3D editing using a single reference style image, making it a user-friendly editing method. However, recent NeRF-based 3D stylization methods encounter efficiency issues that impact the user experience, and their implicit nature limits their ability to accurately transfer geometric pattern styles. Additionally, the ability for artists to apply flexible control over stylized scenes is considered highly desirable to foster an environment conducive to creative exploration. To address the above issues, we introduce StylizedGS, an efficient 3D neural style transfer framework with adaptable control over perceptual factors based on 3D Gaussian Splatting representation. We propose a filter-based refinement to eliminate floaters that affect the stylization effects in the scene reconstruction process. The nearest neighbor-based style loss is introduced to achieve stylization by fine-tuning the geometry and color parameters of 3DGS, while a depth preservation loss with other regularizations is proposed to prevent the tampering of geometry content. Moreover, facilitated by specially designed losses, StylizedGS enables users to control color, stylized scale, and regions during the stylization to possess customization capabilities. Our method achieves high-quality stylization results characterized by faithful brushstrokes and geometric consistency with flexible controls. Extensive experiments across various scenes and styles demonstrate the effectiveness and efficiency of our method concerning both stylization quality and inference speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02141v1">GRMM: Real-Time High-Fidelity Gaussian Morphable Head Model with Learned Residuals</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 Project page: https://mohitm1994.github.io/GRMM/
    </div>
    <details class="paper-abstract">
      3D Morphable Models (3DMMs) enable controllable facial geometry and expression editing for reconstruction, animation, and AR/VR, but traditional PCA-based mesh models are limited in resolution, detail, and photorealism. Neural volumetric methods improve realism but remain too slow for interactive use. Recent Gaussian Splatting (3DGS) based facial models achieve fast, high-quality rendering but still depend solely on a mesh-based 3DMM prior for expression control, limiting their ability to capture fine-grained geometry, expressions, and full-head coverage. We introduce GRMM, the first full-head Gaussian 3D morphable model that augments a base 3DMM with residual geometry and appearance components, additive refinements that recover high-frequency details such as wrinkles, fine skin texture, and hairline variations. GRMM provides disentangled control through low-dimensional, interpretable parameters (e.g., identity shape, facial expressions) while separately modelling residuals that capture subject- and expression-specific detail beyond the base model's capacity. Coarse decoders produce vertex-level mesh deformations, fine decoders represent per-Gaussian appearance, and a lightweight CNN refines rasterised images for enhanced realism, all while maintaining 75 FPS real-time rendering. To learn consistent, high-fidelity residuals, we present EXPRESS-50, the first dataset with 60 aligned expressions across 50 identities, enabling robust disentanglement of identity and expression in Gaussian-based 3DMMs. Across monocular 3D face reconstruction, novel-view synthesis, and expression transfer, GRMM surpasses state-of-the-art methods in fidelity and expression accuracy while delivering interactive real-time performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01964v1">2D Gaussian Splatting with Semantic Alignment for Image Inpainting</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS), a recent technique for converting discrete points into continuous spatial representations, has shown promising results in 3D scene modeling and 2D image super-resolution. In this paper, we explore its untapped potential for image inpainting, which demands both locally coherent pixel synthesis and globally consistent semantic restoration. We propose the first image inpainting framework based on 2D Gaussian Splatting, which encodes incomplete images into a continuous field of 2D Gaussian splat coefficients and reconstructs the final image via a differentiable rasterization process. The continuous rendering paradigm of GS inherently promotes pixel-level coherence in the inpainted results. To improve efficiency and scalability, we introduce a patch-wise rasterization strategy that reduces memory overhead and accelerates inference. For global semantic consistency, we incorporate features from a pretrained DINO model. We observe that DINO's global features are naturally robust to small missing regions and can be effectively adapted to guide semantic alignment in large-mask scenarios, ensuring that the inpainted content remains contextually consistent with the surrounding scene. Extensive experiments on standard benchmarks demonstrate that our method achieves competitive performance in both quantitative metrics and perceptual quality, establishing a new direction for applying Gaussian Splatting to 2D image processing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09573v2">FreeSplatter: Pose-free Gaussian Splatting for Sparse-view 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Project page: https://bluestyle97.github.io/projects/freesplatter/
    </div>
    <details class="paper-abstract">
      Sparse-view reconstruction models typically require precise camera poses, yet obtaining these parameters from sparse-view images remains challenging. We introduce FreeSplatter, a scalable feed-forward framework that generates high-quality 3D Gaussians from uncalibrated sparse-view images while estimating camera parameters within seconds. Our approach employs a streamlined transformer architecture where self-attention blocks facilitate information exchange among multi-view image tokens, decoding them into pixel-aligned 3D Gaussian primitives within a unified reference frame. This representation enables both high-fidelity 3D modeling and efficient camera parameter estimation using off-the-shelf solvers. We develop two specialized variants--for object-centric and scene-level reconstruction--trained on comprehensive datasets. Remarkably, FreeSplatter outperforms several pose-dependent Large Reconstruction Models (LRMs) by a notable margin while achieving comparable or even better pose estimation accuracy compared to state-of-the-art pose-free reconstruction approach MASt3R in challenging benchmarks. Beyond technical benchmarks, FreeSplatter streamlines text/image-to-3D content creation pipelines, eliminating the complexity of camera pose management while delivering exceptional visual fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10462v2">BloomScene: Lightweight Structured 3D Gaussian Splatting for Crossmodal Scene Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Accepted by AAAI 2025. Code: https://github.com/SparklingH/BloomScene
    </div>
    <details class="paper-abstract">
      With the widespread use of virtual reality applications, 3D scene generation has become a new challenging research frontier. 3D scenes have highly complex structures and need to ensure that the output is dense, coherent, and contains all necessary structures. Many current 3D scene generation methods rely on pre-trained text-to-image diffusion models and monocular depth estimators. However, the generated scenes occupy large amounts of storage space and often lack effective regularisation methods, leading to geometric distortions. To this end, we propose BloomScene, a lightweight structured 3D Gaussian splatting for crossmodal scene generation, which creates diverse and high-quality 3D scenes from text or image inputs. Specifically, a crossmodal progressive scene generation framework is proposed to generate coherent scenes utilizing incremental point cloud reconstruction and 3D Gaussian splatting. Additionally, we propose a hierarchical depth prior-based regularization mechanism that utilizes multi-level constraints on depth accuracy and smoothness to enhance the realism and continuity of the generated scenes. Ultimately, we propose a structured context-guided compression mechanism that exploits structured hash grids to model the context of unorganized anchor attributes, which significantly eliminates structural redundancy and reduces storage overhead. Comprehensive experiments across multiple scenes demonstrate the significant potential and advantages of our framework compared with several baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01681v1">GaussianGAN: Real-Time Photorealistic controllable Human Avatars</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 IEEE conference series on Automatic Face and Gesture Recognition 2025
    </div>
    <details class="paper-abstract">
      Photorealistic and controllable human avatars have gained popularity in the research community thanks to rapid advances in neural rendering, providing fast and realistic synthesis tools. However, a limitation of current solutions is the presence of noticeable blurring. To solve this problem, we propose GaussianGAN, an animatable avatar approach developed for photorealistic rendering of people in real-time. We introduce a novel Gaussian splatting densification strategy to build Gaussian points from the surface of cylindrical structures around estimated skeletal limbs. Given the camera calibration, we render an accurate semantic segmentation with our novel view segmentation module. Finally, a UNet generator uses the rendered Gaussian splatting features and the segmentation maps to create photorealistic digital avatars. Our method runs in real-time with a rendering speed of 79 FPS. It outperforms previous methods regarding visual perception and quality, achieving a state-of-the-art results in terms of a pixel fidelity of 32.94db on the ZJU Mocap dataset and 33.39db on the Thuman4 dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01469v1">Im2Haircut: Single-view Strand-based Hair Reconstruction for Human Avatars</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 For more results please refer to the project page https://im2haircut.is.tue.mpg.de
    </div>
    <details class="paper-abstract">
      We present a novel approach for 3D hair reconstruction from single photographs based on a global hair prior combined with local optimization. Capturing strand-based hair geometry from single photographs is challenging due to the variety and geometric complexity of hairstyles and the lack of ground truth training data. Classical reconstruction methods like multi-view stereo only reconstruct the visible hair strands, missing the inner structure of hairstyles and hampering realistic hair simulation. To address this, existing methods leverage hairstyle priors trained on synthetic data. Such data, however, is limited in both quantity and quality since it requires manual work from skilled artists to model the 3D hairstyles and create near-photorealistic renderings. To address this, we propose a novel approach that uses both, real and synthetic data to learn an effective hairstyle prior. Specifically, we train a transformer-based prior model on synthetic data to obtain knowledge of the internal hairstyle geometry and introduce real data in the learning process to model the outer structure. This training scheme is able to model the visible hair strands depicted in an input image, while preserving the general 3D structure of hairstyles. We exploit this prior to create a Gaussian-splatting-based reconstruction method that creates hairstyles from one or more images. Qualitative and quantitative comparisons with existing reconstruction pipelines demonstrate the effectiveness and superior performance of our method for capturing detailed hair orientation, overall silhouette, and backside consistency. For additional results and code, please refer to https://im2haircut.is.tue.mpg.de.
    </details>
</div>
