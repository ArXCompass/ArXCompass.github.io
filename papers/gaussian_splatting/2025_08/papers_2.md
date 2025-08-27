# gaussian splatting - 2025_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07557v1">Splat4D: Diffusion-Enhanced 4D Gaussian Splatting for Temporally and Spatially Consistent Content Creation</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Generating high-quality 4D content from monocular videos for applications such as digital humans and AR/VR poses challenges in ensuring temporal and spatial consistency, preserving intricate details, and incorporating user guidance effectively. To overcome these challenges, we introduce Splat4D, a novel framework enabling high-fidelity 4D content generation from a monocular video. Splat4D achieves superior performance while maintaining faithful spatial-temporal coherence by leveraging multi-view rendering, inconsistency identification, a video diffusion model, and an asymmetric U-Net for refinement. Through extensive evaluations on public benchmarks, Splat4D consistently demonstrates state-of-the-art performance across various metrics, underscoring the efficacy of our approach. Additionally, the versatility of Splat4D is validated in various applications such as text/image conditioned 4D generation, 4D human generation, and text-guided content editing, producing coherent outcomes following user instructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03310v2">3D Gaussian Splatting Data Compression with Mixture of Priors</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) data compression is crucial for enabling efficient storage and transmission in 3D scene modeling. However, its development remains limited due to inadequate entropy models and suboptimal quantization strategies for both lossless and lossy compression scenarios, where existing methods have yet to 1) fully leverage hyperprior information to construct robust conditional entropy models, and 2) apply fine-grained, element-wise quantization strategies for improved compression granularity. In this work, we propose a novel Mixture of Priors (MoP) strategy to simultaneously address these two challenges. Specifically, inspired by the Mixture-of-Experts (MoE) paradigm, our MoP approach processes hyperprior information through multiple lightweight MLPs to generate diverse prior features, which are subsequently integrated into the MoP feature via a gating mechanism. To enhance lossless compression, the resulting MoP feature is utilized as a hyperprior to improve conditional entropy modeling. Meanwhile, for lossy compression, we employ the MoP feature as guidance information in an element-wise quantization procedure, leveraging a prior-guided Coarse-to-Fine Quantization (C2FQ) strategy with a predefined quantization step value. Specifically, we expand the quantization step value into a matrix and adaptively refine it from coarse to fine granularity, guided by the MoP feature, thereby obtaining a quantization step matrix that facilitates element-wise quantization. Extensive experiments demonstrate that our proposed 3DGS data compression framework achieves state-of-the-art performance across multiple benchmarks, including Mip-NeRF360, BungeeNeRF, DeepBlending, and Tank&Temples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07483v1">Novel View Synthesis with Gaussian Splatting: Impact on Photogrammetry Model Accuracy and Resolution</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      In this paper, I present a comprehensive study comparing Photogrammetry and Gaussian Splatting techniques for 3D model reconstruction and view synthesis. I created a dataset of images from a real-world scene and constructed 3D models using both methods. To evaluate the performance, I compared the models using structural similarity index (SSIM), peak signal-to-noise ratio (PSNR), learned perceptual image patch similarity (LPIPS), and lp/mm resolution based on the USAF resolution chart. A significant contribution of this work is the development of a modified Gaussian Splatting repository, which I forked and enhanced to enable rendering images from novel camera poses generated in the Blender environment. This innovation allows for the synthesis of high-quality novel views, showcasing the flexibility and potential of Gaussian Splatting. My investigation extends to an augmented dataset that includes both original ground images and novel views synthesized via Gaussian Splatting. This augmented dataset was employed to generate a new photogrammetry model, which was then compared against the original photogrammetry model created using only the original images. The results demonstrate the efficacy of using Gaussian Splatting to generate novel high-quality views and its potential to improve photogrammetry-based 3D reconstructions. The comparative analysis highlights the strengths and limitations of both approaches, providing valuable information for applications in extended reality (XR), photogrammetry, and autonomous vehicle simulations. Code is available at https://github.com/pranavc2255/gaussian-splatting-novel-view-render.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2210.00379v7">NeRF: Neural Radiance Field in 3D Vision: A Comprehensive Review (Updated Post-Gaussian Splatting)</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 Updated Post-Gaussian Splatting
    </div>
    <details class="paper-abstract">
      In March 2020, Neural Radiance Field (NeRF) revolutionized Computer Vision, allowing for implicit, neural network-based scene representation and novel view synthesis. NeRF models have found diverse applications in robotics, urban mapping, autonomous navigation, virtual reality/augmented reality, and more. In August 2023, Gaussian Splatting, a direct competitor to the NeRF-based framework, was proposed, gaining tremendous momentum and overtaking NeRF-based research in terms of interest as the dominant framework for novel view synthesis. We present a comprehensive survey of NeRF papers from the past five years (2020-2025). These include papers from the pre-Gaussian Splatting era, where NeRF dominated the field for novel view synthesis and 3D implicit and hybrid representation neural field learning. We also include works from the post-Gaussian Splatting era where NeRF and implicit/hybrid neural fields found more niche applications. Our survey is organized into architecture and application-based taxonomies in the pre-Gaussian Splatting era, as well as a categorization of active research areas for NeRF, neural field, and implicit/hybrid neural representation methods. We provide an introduction to the theory of NeRF and its training via differentiable volume rendering. We also present a benchmark comparison of the performance and speed of classical NeRF, implicit and hybrid neural representation, and neural field models, and an overview of key datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07409v1">CharacterShot: Controllable and Consistent 4D Character Animation</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 13 pages, 10 figures. Code at https://github.com/Jeoyal/CharacterShot
    </div>
    <details class="paper-abstract">
      In this paper, we propose \textbf{CharacterShot}, a controllable and consistent 4D character animation framework that enables any individual designer to create dynamic 3D characters (i.e., 4D character animation) from a single reference character image and a 2D pose sequence. We begin by pretraining a powerful 2D character animation model based on a cutting-edge DiT-based image-to-video model, which allows for any 2D pose sequnce as controllable signal. We then lift the animation model from 2D to 3D through introducing dual-attention module together with camera prior to generate multi-view videos with spatial-temporal and spatial-view consistency. Finally, we employ a novel neighbor-constrained 4D gaussian splatting optimization on these multi-view videos, resulting in continuous and stable 4D character representations. Moreover, to improve character-centric performance, we construct a large-scale dataset Character4D, containing 13,115 unique characters with diverse appearances and motions, rendered from multiple viewpoints. Extensive experiments on our newly constructed benchmark, CharacterBench, demonstrate that our approach outperforms current state-of-the-art methods. Code, models, and datasets will be publicly available at https://github.com/Jeoyal/CharacterShot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.24009v2">Learning 3D-Gaussian Simulators from RGB Videos</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Realistic simulation is critical for applications ranging from robotics to animation. Learned simulators have emerged as a possibility to capture real world physics directly from video data, but very often require privileged information such as depth information, particle tracks and hand-engineered features to maintain spatial and temporal consistency. These strong inductive biases or ground truth 3D information help in domains where data is sparse but limit scalability and generalization in data rich regimes. To overcome the key limitations, we propose 3DGSim, a learned 3D simulator that directly learns physical interactions from multi-view RGB videos. 3DGSim unifies 3D scene reconstruction, particle dynamics prediction and video synthesis into an end-to-end trained framework. It adopts MVSplat to learn a latent particle-based representation of 3D scenes, a Point Transformer for particle dynamics, a Temporal Merging module for consistent temporal aggregation and Gaussian Splatting to produce novel view renderings. By jointly training inverse rendering and dynamics forecasting, 3DGSim embeds the physical properties into point-wise latent features. This enables the model to capture diverse physical behaviors, from rigid to elastic, cloth-like dynamics, and boundary conditions (e.g. fixed cloth corner), along with realistic lighting effects that also generalize to unseen multibody interactions and novel scene edits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07372v1">DIP-GS: Deep Image Prior For Gaussian Splatting Sparse View Recovery</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a leading 3D scene reconstruction method, obtaining high-quality reconstruction with real-time rendering runtime performance. The main idea behind 3DGS is to represent the scene as a collection of 3D gaussians, while learning their parameters to fit the given views of the scene. While achieving superior performance in the presence of many views, 3DGS struggles with sparse view reconstruction, where the input views are sparse and do not fully cover the scene and have low overlaps. In this paper, we propose DIP-GS, a Deep Image Prior (DIP) 3DGS representation. By using the DIP prior, which utilizes internal structure and patterns, with coarse-to-fine manner, DIP-based 3DGS can operate in scenarios where vanilla 3DGS fails, such as sparse view recovery. Note that our approach does not use any pre-trained models such as generative models and depth estimation, but rather relies only on the input frames. Among such methods, DIP-GS obtains state-of-the-art (SOTA) competitive results on various sparse-view reconstruction tasks, demonstrating its capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07355v1">GS4Buildings: Prior-Guided Gaussian Splatting for 3D Building Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 Accepted for presentation at ISPRS 3D GeoInfo & Smart Data, Smart Cities 2025, Kashiwa, Japan. To appear in the ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences
    </div>
    <details class="paper-abstract">
      Recent advances in Gaussian Splatting (GS) have demonstrated its effectiveness in photo-realistic rendering and 3D reconstruction. Among these, 2D Gaussian Splatting (2DGS) is particularly suitable for surface reconstruction due to its flattened Gaussian representation and integrated normal regularization. However, its performance often degrades in large-scale and complex urban scenes with frequent occlusions, leading to incomplete building reconstructions. We propose GS4Buildings, a novel prior-guided Gaussian Splatting method leveraging the ubiquity of semantic 3D building models for robust and scalable building surface reconstruction. Instead of relying on traditional Structure-from-Motion (SfM) pipelines, GS4Buildings initializes Gaussians directly from low-level Level of Detail (LoD)2 semantic 3D building models. Moreover, we generate prior depth and normal maps from the planar building geometry and incorporate them into the optimization process, providing strong geometric guidance for surface consistency and structural accuracy. We also introduce an optional building-focused mode that limits reconstruction to building regions, achieving a 71.8% reduction in Gaussian primitives and enabling a more efficient and compact representation. Experiments on urban datasets demonstrate that GS4Buildings improves reconstruction completeness by 20.5% and geometric accuracy by 32.8%. These results highlight the potential of semantic building model integration to advance GS-based reconstruction toward real-world urban applications such as smart cities and digital twins. Our project is available: https://github.com/zqlin0521/GS4Buildings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05591v2">QuickSplat: Fast 3D Surface Reconstruction via Learned Gaussian Initialization</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 ICCV 2025. Project page: https://liu115.github.io/quicksplat, Video: https://youtu.be/2IA_gnFvFG8
    </div>
    <details class="paper-abstract">
      Surface reconstruction is fundamental to computer vision and graphics, enabling applications in 3D modeling, mixed reality, robotics, and more. Existing approaches based on volumetric rendering obtain promising results, but optimize on a per-scene basis, resulting in a slow optimization that can struggle to model under-observed or textureless regions. We introduce QuickSplat, which learns data-driven priors to generate dense initializations for 2D gaussian splatting optimization of large-scale indoor scenes. This provides a strong starting point for the reconstruction, which accelerates the convergence of the optimization and improves the geometry of flat wall structures. We further learn to jointly estimate the densification and update of the scene parameters during each iteration; our proposed densifier network predicts new Gaussians based on the rendering gradients of existing ones, removing the needs of heuristics for densification. Extensive experiments on large-scale indoor scene reconstruction demonstrate the superiority of our data-driven optimization. Concretely, we accelerate runtime by 8x, while decreasing depth errors by up to 48% in comparison to state of the art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07263v1">Fading the Digital Ink: A Universal Black-Box Attack Framework for 3DGS Watermarking Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      With the rise of 3D Gaussian Splatting (3DGS), a variety of digital watermarking techniques, embedding either 1D bitstreams or 2D images, are used for copyright protection. However, the robustness of these watermarking techniques against potential attacks remains underexplored. This paper introduces the first universal black-box attack framework, the Group-based Multi-objective Evolutionary Attack (GMEA), designed to challenge these watermarking systems. We formulate the attack as a large-scale multi-objective optimization problem, balancing watermark removal with visual quality. In a black-box setting, we introduce an indirect objective function that blinds the watermark detector by minimizing the standard deviation of features extracted by a convolutional network, thus rendering the feature maps uninformative. To manage the vast search space of 3DGS models, we employ a group-based optimization strategy to partition the model into multiple, independent sub-optimization problems. Experiments demonstrate that our framework effectively removes both 1D and 2D watermarks from mainstream 3DGS watermarking methods while maintaining high visual fidelity. This work reveals critical vulnerabilities in existing 3DGS copyright protection schemes and calls for the development of more robust watermarking systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07182v1">3D Gaussian Representations with Motion Trajectory Field for Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      This paper addresses the challenge of novel-view synthesis and motion reconstruction of dynamic scenes from monocular video, which is critical for many robotic applications. Although Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have demonstrated remarkable success in rendering static scenes, extending them to reconstruct dynamic scenes remains challenging. In this work, we introduce a novel approach that combines 3DGS with a motion trajectory field, enabling precise handling of complex object motions and achieving physically plausible motion trajectories. By decoupling dynamic objects from static background, our method compactly optimizes the motion trajectory field. The approach incorporates time-invariant motion coefficients and shared motion trajectory bases to capture intricate motion patterns while minimizing optimization complexity. Extensive experiments demonstrate that our approach achieves state-of-the-art results in both novel-view synthesis and motion trajectory recovery from monocular video, advancing the capabilities of dynamic scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07118v1">DexFruit: Dexterous Manipulation and Gaussian Splatting Inspection of Fruit</a></div>
    <div class="paper-meta">
      📅 2025-08-09
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      DexFruit is a robotic manipulation framework that enables gentle, autonomous handling of fragile fruit and precise evaluation of damage. Many fruits are fragile and prone to bruising, thus requiring humans to manually harvest them with care. In this work, we demonstrate by using optical tactile sensing, autonomous manipulation of fruit with minimal damage can be achieved. We show that our tactile informed diffusion policies outperform baselines in both reduced bruising and pick-and-place success rate across three fruits: strawberries, tomatoes, and blackberries. In addition, we introduce FruitSplat, a novel technique to represent and quantify visual damage in high-resolution 3D representation via 3D Gaussian Splatting (3DGS). Existing metrics for measuring damage lack quantitative rigor or require expensive equipment. With FruitSplat, we distill a 2D strawberry mask as well as a 2D bruise segmentation mask into the 3DGS representation. Furthermore, this representation is modular and general, compatible with any relevant 2D model. Overall, we demonstrate a 92% grasping policy success rate, up to a 20% reduction in visual bruising, and up to an 31% improvement in grasp success rate on challenging fruit compared to our baselines across our three tested fruits. We rigorously evaluate this result with over 630 trials. Please checkout our website at https://dex-fruit.github.io .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07038v1">3DGS-VBench: A Comprehensive Video Quality Evaluation Benchmark for 3DGS Compression</a></div>
    <div class="paper-meta">
      📅 2025-08-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables real-time novel view synthesis with high visual fidelity, but its substantial storage requirements hinder practical deployment, prompting state-of-the-art (SOTA) 3DGS methods to incorporate compression modules. However, these 3DGS generative compression techniques introduce unique distortions lacking systematic quality assessment research. To this end, we establish 3DGS-VBench, a large-scale Video Quality Assessment (VQA) Dataset and Benchmark with 660 compressed 3DGS models and video sequences generated from 11 scenes across 6 SOTA 3DGS compression algorithms with systematically designed parameter levels. With annotations from 50 participants, we obtained MOS scores with outlier removal and validated dataset reliability. We benchmark 6 3DGS compression algorithms on storage efficiency and visual quality, and evaluate 15 quality assessment metrics across multiple paradigms. Our work enables specialized VQA model training for 3DGS, serving as a catalyst for compression and quality assessment research. The dataset is available at https://github.com/YukeXing/3DGS-VBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10256v2">ROODI: Reconstructing Occluded Objects with Denoising Inpainters</a></div>
    <div class="paper-meta">
      📅 2025-08-09
      | 💬 Project page: https://yeonjin-chang.github.io/ROODI/
    </div>
    <details class="paper-abstract">
      While the quality of novel-view images has improved dramatically with 3D Gaussian Splatting, extracting specific objects from scenes remains challenging. Isolating individual 3D Gaussian primitives for each object and handling occlusions in scenes remains far from being solved. We propose a novel object extraction method based on two key principles: (1) object-centric reconstruction through removal of irrelevant primitives; and (2) leveraging generative inpainting to compensate for missing observations caused by occlusions. For pruning, we propose to remove irrelevant Gaussians by looking into how close they are to its K-nearest neighbors and removing those that are statistical outliers. Importantly, these distances must take into account the actual spatial extent they cover -- we thus propose to use Wasserstein distances. For inpainting, we employ an off-the-shelf diffusion-based inpainter combined with occlusion reasoning, utilizing the 3D representation of the entire scene. Our findings highlight the crucial synergy between proper pruning and inpainting, both of which significantly enhance extraction performance. We evaluate our method on a standard real-world dataset and introduce a synthetic dataset for quantitative analysis. Our approach outperforms the state-of-the-art, demonstrating its effectiveness in object extraction from complex scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07003v1">EGS-SLAM: RGB-D Gaussian Splatting SLAM with Events</a></div>
    <div class="paper-meta">
      📅 2025-08-09
      | 💬 Accepted by IEEE RAL
    </div>
    <details class="paper-abstract">
      Gaussian Splatting SLAM (GS-SLAM) offers a notable improvement over traditional SLAM methods, enabling photorealistic 3D reconstruction that conventional approaches often struggle to achieve. However, existing GS-SLAM systems perform poorly under persistent and severe motion blur commonly encountered in real-world scenarios, leading to significantly degraded tracking accuracy and compromised 3D reconstruction quality. To address this limitation, we propose EGS-SLAM, a novel GS-SLAM framework that fuses event data with RGB-D inputs to simultaneously reduce motion blur in images and compensate for the sparse and discrete nature of event streams, enabling robust tracking and high-fidelity 3D Gaussian Splatting reconstruction. Specifically, our system explicitly models the camera's continuous trajectory during exposure, supporting event- and blur-aware tracking and mapping on a unified 3D Gaussian Splatting scene. Furthermore, we introduce a learnable camera response function to align the dynamic ranges of events and images, along with a no-event loss to suppress ringing artifacts during reconstruction. We validate our approach on a new dataset comprising synthetic and real-world sequences with significant motion blur. Extensive experimental results demonstrate that EGS-SLAM consistently outperforms existing GS-SLAM systems in both trajectory accuracy and photorealistic 3D Gaussian Splatting reconstruction. The source code will be available at https://github.com/Chensiyu00/EGS-SLAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06968v1">Evaluating Fisheye-Compatible 3D Gaussian Splatting Methods on Real Images Beyond 180 Degree Field of View</a></div>
    <div class="paper-meta">
      📅 2025-08-09
    </div>
    <details class="paper-abstract">
      We present the first evaluation of fisheye-based 3D Gaussian Splatting methods, Fisheye-GS and 3DGUT, on real images with fields of view exceeding 180 degree. Our study covers both indoor and outdoor scenes captured with 200 degree fisheye cameras and analyzes how each method handles extreme distortion in real world settings. We evaluate performance under varying fields of view (200 degree, 160 degree, and 120 degree) to study the tradeoff between peripheral distortion and spatial coverage. Fisheye-GS benefits from field of view (FoV) reduction, particularly at 160 degree, while 3DGUT remains stable across all settings and maintains high perceptual quality at the full 200 degree view. To address the limitations of SfM-based initialization, which often fails under strong distortion, we also propose a depth-based strategy using UniK3D predictions from only 2-3 fisheye images per scene. Although UniK3D is not trained on real fisheye data, it produces dense point clouds that enable reconstruction quality on par with SfM, even in difficult scenes with fog, glare, or sky. Our results highlight the practical viability of fisheye-based 3DGS methods for wide-angle 3D reconstruction from sparse and distortion-heavy image inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14642v2">3DGS-IEval-15K: A Large-scale Image Quality Evaluation Database for 3D Gaussian-Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a promising approach for novel view synthesis, offering real-time rendering with high visual fidelity. However, its substantial storage requirements present significant challenges for practical applications. While recent state-of-the-art (SOTA) 3DGS methods increasingly incorporate dedicated compression modules, there is a lack of a comprehensive framework to evaluate their perceptual impact. Therefore we present 3DGS-IEval-15K, the first large-scale image quality assessment (IQA) dataset specifically designed for compressed 3DGS representations. Our dataset encompasses 15,200 images rendered from 10 real-world scenes through 6 representative 3DGS algorithms at 20 strategically selected viewpoints, with different compression levels leading to various distortion effects. Through controlled subjective experiments, we collect human perception data from 60 viewers. We validate dataset quality through scene diversity and MOS distribution analysis, and establish a comprehensive benchmark with 30 representative IQA metrics covering diverse types. As the largest-scale 3DGS quality assessment dataset to date, our work provides a foundation for developing 3DGS specialized IQA metrics, and offers essential data for investigating view-dependent quality distribution patterns unique to 3DGS. The database is publicly available at https://github.com/YukeXing/3DGS-IEval-15K.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06318v1">Mixture of Experts Guided by Gaussian Splatters Matters: A new Approach to Weakly-Supervised Video Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Video Anomaly Detection (VAD) is a challenging task due to the variability of anomalous events and the limited availability of labeled data. Under the Weakly-Supervised VAD (WSVAD) paradigm, only video-level labels are provided during training, while predictions are made at the frame level. Although state-of-the-art models perform well on simple anomalies (e.g., explosions), they struggle with complex real-world events (e.g., shoplifting). This difficulty stems from two key issues: (1) the inability of current models to address the diversity of anomaly types, as they process all categories with a shared model, overlooking category-specific features; and (2) the weak supervision signal, which lacks precise temporal information, limiting the ability to capture nuanced anomalous patterns blended with normal events. To address these challenges, we propose Gaussian Splatting-guided Mixture of Experts (GS-MoE), a novel framework that employs a set of expert models, each specialized in capturing specific anomaly types. These experts are guided by a temporal Gaussian splatting loss, enabling the model to leverage temporal consistency and enhance weak supervision. The Gaussian splatting approach encourages a more precise and comprehensive representation of anomalies by focusing on temporal segments most likely to contain abnormal events. The predictions from these specialized experts are integrated through a mixture-of-experts mechanism to model complex relationships across diverse anomaly patterns. Our approach achieves state-of-the-art performance, with a 91.58% AUC on the UCF-Crime dataset, and demonstrates superior results on XD-Violence and MSAD datasets. By leveraging category-specific expertise and temporal guidance, GS-MoE sets a new benchmark for VAD under weak supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08279v2">MBA-SLAM: Motion Blur Aware Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 Accepted to TPAMI; Deblur Gaussian Splatting SLAM
    </div>
    <details class="paper-abstract">
      Emerging 3D scene representations, such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have demonstrated their effectiveness in Simultaneous Localization and Mapping (SLAM) for photo-realistic rendering, particularly when using high-quality video sequences as input. However, existing methods struggle with motion-blurred frames, which are common in real-world scenarios like low-light or long-exposure conditions. This often results in a significant reduction in both camera localization accuracy and map reconstruction quality. To address this challenge, we propose a dense visual deblur SLAM pipeline (i.e. MBA-SLAM) to handle severe motion-blurred inputs and enhance image deblurring. Our approach integrates an efficient motion blur-aware tracker with either neural radiance fields or Gaussian Splatting based mapper. By accurately modeling the physical image formation process of motion-blurred images, our method simultaneously learns 3D scene representation and estimates the cameras' local trajectory during exposure time, enabling proactive compensation for motion blur caused by camera movement. In our experiments, we demonstrate that MBA-SLAM surpasses previous state-of-the-art methods in both camera localization and map reconstruction, showcasing superior performance across a range of datasets, including synthetic and real datasets featuring sharp images as well as those affected by motion blur, highlighting the versatility and robustness of our approach. Code is available at https://github.com/WU-CVGL/MBA-SLAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06169v1">UW-3DGS: Underwater 3D Reconstruction with Physics-Aware Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Underwater 3D scene reconstruction faces severe challenges from light absorption, scattering, and turbidity, which degrade geometry and color fidelity in traditional methods like Neural Radiance Fields (NeRF). While NeRF extensions such as SeaThru-NeRF incorporate physics-based models, their MLP reliance limits efficiency and spatial resolution in hazy environments. We introduce UW-3DGS, a novel framework adapting 3D Gaussian Splatting (3DGS) for robust underwater reconstruction. Key innovations include: (1) a plug-and-play learnable underwater image formation module using voxel-based regression for spatially varying attenuation and backscatter; and (2) a Physics-Aware Uncertainty Pruning (PAUP) branch that adaptively removes noisy floating Gaussians via uncertainty scoring, ensuring artifact-free geometry. The pipeline operates in training and rendering stages. During training, noisy Gaussians are optimized end-to-end with underwater parameters, guided by PAUP pruning and scattering modeling. In rendering, refined Gaussians produce clean Unattenuated Radiance Images (URIs) free from media effects, while learned physics enable realistic Underwater Images (UWIs) with accurate light transport. Experiments on SeaThru-NeRF and UWBundle datasets show superior performance, achieving PSNR of 27.604, SSIM of 0.868, and LPIPS of 0.104 on SeaThru-NeRF, with ~65% reduction in floating artifacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06136v1">Roll Your Eyes: Gaze Redirection via Explicit 3D Eyeball Rotation</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 9 pages, 5 figures, ACM Multimeida 2025 accepted
    </div>
    <details class="paper-abstract">
      We propose a novel 3D gaze redirection framework that leverages an explicit 3D eyeball structure. Existing gaze redirection methods are typically based on neural radiance fields, which employ implicit neural representations via volume rendering. Unlike these NeRF-based approaches, where the rotation and translation of 3D representations are not explicitly modeled, we introduce a dedicated 3D eyeball structure to represent the eyeballs with 3D Gaussian Splatting (3DGS). Our method generates photorealistic images that faithfully reproduce the desired gaze direction by explicitly rotating and translating the 3D eyeball structure. In addition, we propose an adaptive deformation module that enables the replication of subtle muscle movements around the eyes. Through experiments conducted on the ETH-XGaze dataset, we demonstrate that our framework is capable of generating diverse novel gaze images, achieving superior image quality and gaze estimation accuracy compared to previous state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06014v1">ExploreGS: Explorable 3D Scene Reconstruction with Virtual Camera Samplings and Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 10 pages, 6 Figures, ICCV 2025
    </div>
    <details class="paper-abstract">
      Recent advances in novel view synthesis (NVS) have enabled real-time rendering with 3D Gaussian Splatting (3DGS). However, existing methods struggle with artifacts and missing regions when rendering from viewpoints that deviate from the training trajectory, limiting seamless scene exploration. To address this, we propose a 3DGS-based pipeline that generates additional training views to enhance reconstruction. We introduce an information-gain-driven virtual camera placement strategy to maximize scene coverage, followed by video diffusion priors to refine rendered results. Fine-tuning 3D Gaussians with these enhanced views significantly improves reconstruction quality. To evaluate our method, we present Wild-Explore, a benchmark designed for challenging scene exploration. Experiments demonstrate that our approach outperforms existing 3DGS-based methods, enabling high-quality, artifact-free rendering from arbitrary viewpoints. https://exploregs.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05254v2">CF3: Compact and Fast 3D Feature Fields</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 ICCV 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has begun incorporating rich information from 2D foundation models. However, most approaches rely on a bottom-up optimization process that treats raw 2D features as ground truth, incurring increased computational costs. We propose a top-down pipeline for constructing compact and fast 3D Gaussian feature fields, namely, CF3. We first perform a fast weighted fusion of multi-view 2D features with pre-trained Gaussians. This approach enables training a per-Gaussian autoencoder directly on the lifted features, instead of training autoencoders in the 2D domain. As a result, the autoencoder better aligns with the feature distribution. More importantly, we introduce an adaptive sparsification method that optimizes the Gaussian attributes of the feature field while pruning and merging the redundant Gaussians, constructing an efficient representation with preserved geometric details. Our approach achieves a competitive 3D feature field using as little as 5% of the Gaussians compared to Feature-3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05950v1">A 3DGS-Diffusion Self-Supervised Framework for Normal Estimation from a Single Image</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      The lack of spatial dimensional information remains a challenge in normal estimation from a single image. Recent diffusion-based methods have demonstrated significant potential in 2D-to-3D implicit mapping, they rely on data-driven statistical priors and miss the explicit modeling of light-surface interaction, leading to multi-view normal direction conflicts. Moreover, the discrete sampling mechanism of diffusion models causes gradient discontinuity in differentiable rendering reconstruction modules, preventing 3D geometric errors from being backpropagated to the normal generation network, thereby forcing existing methods to depend on dense normal annotations. This paper proposes SINGAD, a novel Self-supervised framework from a single Image for Normal estimation via 3D GAussian splatting guided Diffusion. By integrating physics-driven light-interaction modeling and a differentiable rendering-based reprojection strategy, our framework directly converts 3D geometric errors into normal optimization signals, solving the challenges of multi-view geometric inconsistency and data dependency. Specifically, the framework constructs a light-interaction-driven 3DGS reparameterization model to generate multi-scale geometric features consistent with light transport principles, ensuring multi-view normal consistency. A cross-domain feature fusion module is designed within a conditional diffusion model, embedding geometric priors to constrain normal generation while maintaining accurate geometric error propagation. Furthermore, a differentiable 3D reprojection loss strategy is introduced for self-supervised optimization that minimizes geometric error between the reconstructed and input image, eliminating dependence on annotated normal datasets. Quantitative evaluations on the Google Scanned Objects dataset demonstrate that our method outperforms state-of-the-art approaches across multiple metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05631v1">GAP: Gaussianize Any Point Clouds with Text Guidance</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 ICCV 2025. Project page: https://weiqi-zhang.github.io/GAP
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated its advantages in achieving fast and high-quality rendering. As point clouds serve as a widely-used and easily accessible form of 3D representation, bridging the gap between point clouds and Gaussians becomes increasingly important. Recent studies have explored how to convert the colored points into Gaussians, but directly generating Gaussians from colorless 3D point clouds remains an unsolved challenge. In this paper, we propose GAP, a novel approach that gaussianizes raw point clouds into high-fidelity 3D Gaussians with text guidance. Our key idea is to design a multi-view optimization framework that leverages a depth-aware image diffusion model to synthesize consistent appearances across different viewpoints. To ensure geometric accuracy, we introduce a surface-anchoring mechanism that effectively constrains Gaussians to lie on the surfaces of 3D shapes during optimization. Furthermore, GAP incorporates a diffuse-based inpainting strategy that specifically targets at completing hard-to-observe regions. We evaluate GAP on the Point-to-Gaussian generation task across varying complexity levels, from synthetic point clouds to challenging real-world scans, and even large-scale scenes. Project Page: https://weiqi-zhang.github.io/GAP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01225v2">Reality Fusion: Robust Real-time Immersive Mobile Robot Teleoperation with Volumetric Visual Data Fusion</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Accepted at IROS 2024
    </div>
    <details class="paper-abstract">
      We introduce Reality Fusion, a novel robot teleoperation system that localizes, streams, projects, and merges a typical onboard depth sensor with a photorealistic, high resolution, high framerate, and wide field of view (FoV) rendering of the complex remote environment represented as 3D Gaussian splats (3DGS). Our framework enables robust egocentric and exocentric robot teleoperation in immersive VR, with the 3DGS effectively extending spatial information of a depth sensor with limited FoV and balancing the trade-off between data streaming costs and data visual quality. We evaluated our framework through a user study with 24 participants, which revealed that Reality Fusion leads to significantly better user performance, situation awareness, and user preferences. To support further research and development, we provide an open-source implementation with an easy-to-replicate custom-made telepresence robot, a high-performance virtual reality 3DGS renderer, and an immersive robot control package. (Source code: https://github.com/uhhhci/RealityFusion)
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04326v2">Radiance Fields in XR: A Survey on How Radiance Fields are Envisioned and Addressed for XR Research</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 This work is a pre-print version of a paper that has been accepted to the IEEE TVCG journal for future publication
    </div>
    <details class="paper-abstract">
      The development of radiance fields (RF), such as 3D Gaussian Splatting (3DGS) and Neural Radiance Fields (NeRF), has revolutionized interactive photorealistic view synthesis and presents enormous opportunities for XR research and applications. However, despite the exponential growth of RF research, RF-related contributions to the XR community remain sparse. To better understand this research gap, we performed a systematic survey of current RF literature to analyze (i) how RF is envisioned for XR applications, (ii) how they have already been implemented, and (iii) the remaining research gaps. We collected 365 RF contributions related to XR from computer vision, computer graphics, robotics, multimedia, human-computer interaction, and XR communities, seeking to answer the above research questions. Among the 365 papers, we performed an analysis of 66 papers that already addressed a detailed aspect of RF research for XR. With this survey, we extended and positioned XR-specific RF research topics in the broader RF research field and provide a helpful resource for the XR community to navigate within the rapid development of RF research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05343v1">3DGabSplat: 3D Gabor Splatting for Frequency-adaptive Radiance Field Rendering</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Accepted by ACM MM'25
    </div>
    <details class="paper-abstract">
      Recent prominence in 3D Gaussian Splatting (3DGS) has enabled real-time rendering while maintaining high-fidelity novel view synthesis. However, 3DGS resorts to the Gaussian function that is low-pass by nature and is restricted in representing high-frequency details in 3D scenes. Moreover, it causes redundant primitives with degraded training and rendering efficiency and excessive memory overhead. To overcome these limitations, we propose 3D Gabor Splatting (3DGabSplat) that leverages a novel 3D Gabor-based primitive with multiple directional 3D frequency responses for radiance field representation supervised by multi-view images. The proposed 3D Gabor-based primitive forms a filter bank incorporating multiple 3D Gabor kernels at different frequencies to enhance flexibility and efficiency in capturing fine 3D details. Furthermore, to achieve novel view rendering, an efficient CUDA-based rasterizer is developed to project the multiple directional 3D frequency components characterized by 3D Gabor-based primitives onto the 2D image plane, and a frequency-adaptive mechanism is presented for adaptive joint optimization of primitives. 3DGabSplat is scalable to be a plug-and-play kernel for seamless integration into existing 3DGS paradigms to enhance both efficiency and quality of novel view synthesis. Extensive experiments demonstrate that 3DGabSplat outperforms 3DGS and its variants using alternative primitives, and achieves state-of-the-art rendering quality across both real-world and synthetic scenes. Remarkably, we achieve up to 1.35 dB PSNR gain over 3DGS with simultaneously reduced number of primitives and memory consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01109v3">CountingFruit: Language-Guided 3D Fruit Counting with Semantic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Accurate 3D fruit counting in orchards is challenging due to heavy occlusion, semantic ambiguity between fruits and surrounding structures, and the high computational cost of volumetric reconstruction. Existing pipelines often rely on multi-view 2D segmentation and dense volumetric sampling, which lead to accumulated fusion errors and slow inference. We introduce FruitLangGS, a language-guided 3D fruit counting framework that reconstructs orchard-scale scenes using an adaptive-density Gaussian Splatting pipeline with radius-aware pruning and tile-based rasterization, enabling scalable 3D representation. During inference, compressed CLIP-aligned semantic vectors embedded in each Gaussian are filtered via a dual-threshold cosine similarity mechanism, retrieving Gaussians relevant to target prompts while suppressing common distractors (e.g., foliage), without requiring retraining or image-space masks. The selected Gaussians are then sampled into dense point clouds and clustered geometrically to estimate fruit instances, remaining robust under severe occlusion and viewpoint variation. Experiments on nine different orchard-scale datasets demonstrate that FruitLangGS consistently outperforms existing pipelines in instance counting recall, avoiding multi-view segmentation fusion errors and achieving up to 99.7% recall on Pfuji-Size_Orch2018 orchard dataset. Ablation studies further confirm that language-conditioned semantic embedding and dual-threshold prompt filtering are essential for suppressing distractors and improving counting accuracy under heavy occlusion. Beyond fruit counting, the same framework enables prompt-driven 3D semantic retrieval without retraining, highlighting the potential of language-guided 3D perception for scalable agricultural scene understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05254v1">CF3: Compact and Fast 3D Feature Fields</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 ICCV 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has begun incorporating rich information from 2D foundation models. However, most approaches rely on a bottom-up optimization process that treats raw 2D features as ground truth, incurring increased computational costs. We propose a top-down pipeline for constructing compact and fast 3D Gaussian feature fields, namely, CF3. We first perform a fast weighted fusion of multi-view 2D features with pre-trained Gaussians. This approach enables training a per-Gaussian autoencoder directly on the lifted features, instead of training autoencoders in the 2D domain. As a result, the autoencoder better aligns with the feature distribution. More importantly, we introduce an adaptive sparsification method that optimizes the Gaussian attributes of the feature field while pruning and merging the redundant Gaussians, constructing an efficient representation with preserved geometric details. Our approach achieves a competitive 3D feature field using as little as 5% of the Gaussians compared to Feature-3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05187v1">Refining Gaussian Splatting: A Volumetric Densification Approach</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Achieving high-quality novel view synthesis in 3D Gaussian Splatting (3DGS) often depends on effective point primitive management. The underlying Adaptive Density Control (ADC) process addresses this issue by automating densification and pruning. Yet, the vanilla 3DGS densification strategy shows key shortcomings. To address this issue, in this paper we introduce a novel density control method, which exploits the volumes of inertia associated to each Gaussian function to guide the refinement process. Furthermore, we study the effect of both traditional Structure from Motion (SfM) and Deep Image Matching (DIM) methods for point cloud initialization. Extensive experimental evaluations on the Mip-NeRF 360 dataset demonstrate that our approach surpasses 3DGS in reconstruction quality, delivering encouraging performance across diverse scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05064v1">A Study of the Framework and Real-World Applications of Language Embedding for 3D Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has rapidly emerged as a transformative technique for real-time 3D scene representation, offering a highly efficient and expressive alternative to Neural Radiance Fields (NeRF). Its ability to render complex scenes with high fidelity has enabled progress across domains such as scene reconstruction, robotics, and interactive content creation. More recently, the integration of Large Language Models (LLMs) and language embeddings into Gaussian Splatting pipelines has opened new possibilities for text-conditioned generation, editing, and semantic scene understanding. Despite these advances, a comprehensive overview of this emerging intersection has been lacking. This survey presents a structured review of current research efforts that combine language guidance with 3D Gaussian Splatting, detailing theoretical foundations, integration strategies, and real-world use cases. We highlight key limitations such as computational bottlenecks, generalizability, and the scarcity of semantically annotated 3D Gaussian data and outline open challenges and future directions for advancing language-guided 3D scene understanding using Gaussian Splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04968v1">UGOD: Uncertainty-Guided Differentiable Opacity and Soft Dropout for Enhanced Sparse-View 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 11 pages, 5 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a competitive approach for novel view synthesis (NVS) due to its advanced rendering efficiency through 3D Gaussian projection and blending. However, Gaussians are treated equally weighted for rendering in most 3DGS methods, making them prone to overfitting, which is particularly the case in sparse-view scenarios. To address this, we investigate how adaptive weighting of Gaussians affects rendering quality, which is characterised by learned uncertainties proposed. This learned uncertainty serves two key purposes: first, it guides the differentiable update of Gaussian opacity while preserving the 3DGS pipeline integrity; second, the uncertainty undergoes soft differentiable dropout regularisation, which strategically transforms the original uncertainty into continuous drop probabilities that govern the final Gaussian projection and blending process for rendering. Extensive experimental results over widely adopted datasets demonstrate that our method outperforms rivals in sparse-view 3D synthesis, achieving higher quality reconstruction with fewer Gaussians in most datasets compared to existing sparse-view approaches, e.g., compared to DropGaussian, our method achieves 3.27\% PSNR improvements on the MipNeRF 360 dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04966v1">Laplacian Analysis Meets Dynamics Modelling: Gaussian Splatting for 4D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) excels in static scene modeling, its extension to dynamic scenes introduces significant challenges. Existing dynamic 3DGS methods suffer from either over-smoothing due to low-rank decomposition or feature collision from high-dimensional grid sampling. This is because of the inherent spectral conflicts between preserving motion details and maintaining deformation consistency at different frequency. To address these challenges, we propose a novel dynamic 3DGS framework with hybrid explicit-implicit functions. Our approach contains three key innovations: a spectral-aware Laplacian encoding architecture which merges Hash encoding and Laplacian-based module for flexible frequency motion control, an enhanced Gaussian dynamics attribute that compensates for photometric distortions caused by geometric deformation, and an adaptive Gaussian split strategy guided by KDTree-based primitive control to efficiently query and optimize dynamic areas. Through extensive experiments, our method demonstrates state-of-the-art performance in reconstructing complex dynamic scenes, achieving better reconstruction fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04965v1">Perceive-Sample-Compress: Towards Real-Time 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated remarkable capabilities in real-time and photorealistic novel view synthesis. However, traditional 3DGS representations often struggle with large-scale scene management and efficient storage, particularly when dealing with complex environments or limited computational resources. To address these limitations, we introduce a novel perceive-sample-compress framework for 3D Gaussian Splatting. Specifically, we propose a scene perception compensation algorithm that intelligently refines Gaussian parameters at each level. This algorithm intelligently prioritizes visual importance for higher fidelity rendering in critical areas, while optimizing resource usage and improving overall visible quality. Furthermore, we propose a pyramid sampling representation to manage Gaussian primitives across hierarchical levels. Finally, to facilitate efficient storage of proposed hierarchical pyramid representations, we develop a Generalized Gaussian Mixed model compression algorithm to achieve significant compression ratios without sacrificing visual fidelity. The extensive experiments demonstrate that our method significantly improves memory efficiency and high visual quality while maintaining real-time rendering speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05813v1">Optimization-Free Style Transfer for 3D Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      The task of style transfer for 3D Gaussian splats has been explored in many previous works, but these require reconstructing or fine-tuning the splat while incorporating style information or optimizing a feature extraction network on the splat representation. We propose a reconstruction- and optimization-free approach to stylizing 3D Gaussian splats. This is done by generating a graph structure across the implicit surface of the splat representation. A feed-forward, surface-based stylization method is then used and interpolated back to the individual splats in the scene. This allows for any style image and 3D Gaussian splat to be used without any additional training or optimization. This also allows for fast stylization of splats, achieving speeds under 2 minutes even on consumer-grade hardware. We demonstrate the quality results this approach achieves and compare to other 3D Gaussian splat style transfer methods. Code is publicly available at https://github.com/davidmhart/FastSplatStyler.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02408v2">GR-Gaussian: Graph-Based Radiative Gaussian Splatting for Sparse-View CT Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 10
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a promising approach for CT reconstruction. However, existing methods rely on the average gradient magnitude of points within the view, often leading to severe needle-like artifacts under sparse-view conditions. To address this challenge, we propose GR-Gaussian, a graph-based 3D Gaussian Splatting framework that suppresses needle-like artifacts and improves reconstruction accuracy under sparse-view conditions. Our framework introduces two key innovations: (1) a Denoised Point Cloud Initialization Strategy that reduces initialization errors and accelerates convergence; and (2) a Pixel-Graph-Aware Gradient Strategy that refines gradient computation using graph-based density differences, improving splitting accuracy and density representation. Experiments on X-3D and real-world datasets validate the effectiveness of GR-Gaussian, achieving PSNR improvements of 0.67 dB and 0.92 dB, and SSIM gains of 0.011 and 0.021. These results highlight the applicability of GR-Gaussian for accurate CT reconstruction under challenging sparse-view conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03643v2">Uni3R: Unified 3D Reconstruction and Semantic Understanding via Generalizable Gaussian Splatting from Unposed Multi-View Images</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 The code is available at https://github.com/HorizonRobotics/Uni3R
    </div>
    <details class="paper-abstract">
      Reconstructing and semantically interpreting 3D scenes from sparse 2D views remains a fundamental challenge in computer vision. Conventional methods often decouple semantic understanding from reconstruction or necessitate costly per-scene optimization, thereby restricting their scalability and generalizability. In this paper, we introduce Uni3R, a novel feed-forward framework that jointly reconstructs a unified 3D scene representation enriched with open-vocabulary semantics, directly from unposed multi-view images. Our approach leverages a Cross-View Transformer to robustly integrate information across arbitrary multi-view inputs, which then regresses a set of 3D Gaussian primitives endowed with semantic feature fields. This unified representation facilitates high-fidelity novel view synthesis, open-vocabulary 3D semantic segmentation, and depth prediction, all within a single, feed-forward pass. Extensive experiments demonstrate that Uni3R establishes a new state-of-the-art across multiple benchmarks, including 25.07 PSNR on RE10K and 55.84 mIoU on ScanNet. Our work signifies a novel paradigm towards generalizable, unified 3D scene reconstruction and understanding. The code is available at https://github.com/HorizonRobotics/Uni3R.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22342v2">UFV-Splatter: Pose-Free Feed-Forward 3D Gaussian Splatting Adapted to Unfavorable Views</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Project page: https://yfujimura.github.io/UFV-Splatter_page/
    </div>
    <details class="paper-abstract">
      This paper presents a pose-free, feed-forward 3D Gaussian Splatting (3DGS) framework designed to handle unfavorable input views. A common rendering setup for training feed-forward approaches places a 3D object at the world origin and renders it from cameras pointed toward the origin -- i.e., from favorable views, limiting the applicability of these models to real-world scenarios involving varying and unknown camera poses. To overcome this limitation, we introduce a novel adaptation framework that enables pretrained pose-free feed-forward 3DGS models to handle unfavorable views. We leverage priors learned from favorable images by feeding recentered images into a pretrained model augmented with low-rank adaptation (LoRA) layers. We further propose a Gaussian adapter module to enhance the geometric consistency of the Gaussians derived from the recentered inputs, along with a Gaussian alignment method to render accurate target views for training. Additionally, we introduce a new training strategy that utilizes an off-the-shelf dataset composed solely of favorable images. Experimental results on both synthetic images from the Google Scanned Objects dataset and real images from the OmniObject3D dataset validate the effectiveness of our method in handling unfavorable input views.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04326v1">Radiance Fields in XR: A Survey on How Radiance Fields are Envisioned and Addressed for XR Research</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 This work has been submitted to the IEEE TVCG journal for possible publication
    </div>
    <details class="paper-abstract">
      The development of radiance fields (RF), such as 3D Gaussian Splatting (3DGS) and Neural Radiance Fields (NeRF), has revolutionized interactive photorealistic view synthesis and presents enormous opportunities for XR research and applications. However, despite the exponential growth of RF research, RF-related contributions to the XR community remain sparse. To better understand this research gap, we performed a systematic survey of current RF literature to analyze (i) how RF is envisioned for XR applications, (ii) how they have already been implemented, and (iii) the remaining research gaps. We collected 365 RF contributions related to XR from computer vision, computer graphics, robotics, multimedia, human-computer interaction, and XR communities, seeking to answer the above research questions. Among the 365 papers, we performed an analysis of 66 papers that already addressed a detailed aspect of RF research for XR. With this survey, we extended and positioned XR-specific RF research topics in the broader RF research field and provide a helpful resource for the XR community to navigate within the rapid development of RF research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04297v1">MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 This work is accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      We present Multi-Baseline Gaussian Splatting (MuRF), a generalized feed-forward approach for novel view synthesis that effectively handles diverse baseline settings, including sparse input views with both small and large baselines. Specifically, we integrate features from Multi-View Stereo (MVS) and Monocular Depth Estimation (MDE) to enhance feature representations for generalizable reconstruction. Next, We propose a projection-and-sampling mechanism for deep depth fusion, which constructs a fine probability volume to guide the regression of the feature map. Furthermore, We introduce a reference-view loss to improve geometry and optimization efficiency. We leverage 3D Gaussian representations to accelerate training and inference time while enhancing rendering quality. MuRF achieves state-of-the-art performance across multiple baseline settings and diverse scenarios ranging from simple objects (DTU) to complex indoor and outdoor scenes (RealEstate10K). We also demonstrate promising zero-shot performance on the LLFF and Mip-NeRF 360 datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04224v1">SplitGaussian: Reconstructing Dynamic Scenes via Visual Geometry Decomposition</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic 3D scenes from monocular video remains fundamentally challenging due to the need to jointly infer motion, structure, and appearance from limited observations. Existing dynamic scene reconstruction methods based on Gaussian Splatting often entangle static and dynamic elements in a shared representation, leading to motion leakage, geometric distortions, and temporal flickering. We identify that the root cause lies in the coupled modeling of geometry and appearance across time, which hampers both stability and interpretability. To address this, we propose \textbf{SplitGaussian}, a novel framework that explicitly decomposes scene representations into static and dynamic components. By decoupling motion modeling from background geometry and allowing only the dynamic branch to deform over time, our method prevents motion artifacts in static regions while supporting view- and time-dependent appearance refinement. This disentangled design not only enhances temporal consistency and reconstruction fidelity but also accelerates convergence. Extensive experiments demonstrate that SplitGaussian outperforms prior state-of-the-art methods in rendering quality, geometric stability, and motion separation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04099v1">DET-GS: Depth- and Edge-Aware Regularization for High-Fidelity 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) represents a significant advancement in the field of efficient and high-fidelity novel view synthesis. Despite recent progress, achieving accurate geometric reconstruction under sparse-view conditions remains a fundamental challenge. Existing methods often rely on non-local depth regularization, which fails to capture fine-grained structures and is highly sensitive to depth estimation noise. Furthermore, traditional smoothing methods neglect semantic boundaries and indiscriminately degrade essential edges and textures, consequently limiting the overall quality of reconstruction. In this work, we propose DET-GS, a unified depth and edge-aware regularization framework for 3D Gaussian Splatting. DET-GS introduces a hierarchical geometric depth supervision framework that adaptively enforces multi-level geometric consistency, significantly enhancing structural fidelity and robustness against depth estimation noise. To preserve scene boundaries, we design an edge-aware depth regularization guided by semantic masks derived from Canny edge detection. Furthermore, we introduce an RGB-guided edge-preserving Total Variation loss that selectively smooths homogeneous regions while rigorously retaining high-frequency details and textures. Extensive experiments demonstrate that DET-GS achieves substantial improvements in both geometric accuracy and visual fidelity, outperforming state-of-the-art (SOTA) methods on sparse-view novel view synthesis benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04090v1">Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Accepted to ICCV 2025
    </div>
    <details class="paper-abstract">
      We propose 3D Super Resolution (3DSR), a novel 3D Gaussian-splatting-based super-resolution framework that leverages off-the-shelf diffusion-based 2D super-resolution models. 3DSR encourages 3D consistency across views via the use of an explicit 3D Gaussian-splatting-based scene representation. This makes the proposed 3DSR different from prior work, such as image upsampling or the use of video super-resolution, which either don't consider 3D consistency or aim to incorporate 3D consistency implicitly. Notably, our method enhances visual quality without additional fine-tuning, ensuring spatial coherence within the reconstructed scene. We evaluate 3DSR on MipNeRF360 and LLFF data, demonstrating that it produces high-resolution results that are visually compelling, while maintaining structural consistency in 3D reconstructions. Code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04078v1">RLGS: Reinforcement Learning-Based Adaptive Hyperparameter Tuning for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 14 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Hyperparameter tuning in 3D Gaussian Splatting (3DGS) is a labor-intensive and expert-driven process, often resulting in inconsistent reconstructions and suboptimal results. We propose RLGS, a plug-and-play reinforcement learning framework for adaptive hyperparameter tuning in 3DGS through lightweight policy modules, dynamically adjusting critical hyperparameters such as learning rates and densification thresholds. The framework is model-agnostic and seamlessly integrates into existing 3DGS pipelines without architectural modifications. We demonstrate its generalization ability across multiple state-of-the-art 3DGS variants, including Taming-3DGS and 3DGS-MCMC, and validate its robustness across diverse datasets. RLGS consistently enhances rendering quality. For example, it improves Taming-3DGS by 0.7dB PSNR on the Tanks and Temple (TNT) dataset, under a fixed Gaussian budget, and continues to yield gains even when baseline performance saturates. Our results suggest that RLGS provides an effective and general solution for automating hyperparameter tuning in 3DGS training, bridging a gap in applying reinforcement learning to 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04929v1">CryoGS: Gaussian Splatting for Cryo-EM Homogeneous Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      As a critical modality for structural biology, cryogenic electron microscopy (cryo-EM) facilitates the determination of macromolecular structures at near-atomic resolution. The core computational task in single-particle cryo-EM is to reconstruct the 3D electrostatic potential of a molecule from a large collection of noisy 2D projections acquired at unknown orientations. Gaussian mixture models (GMMs) provide a continuous, compact, and physically interpretable representation for molecular density and have recently gained interest in cryo-EM reconstruction. However, existing methods rely on external consensus maps or atomic models for initialization, limiting their use in self-contained pipelines. Addressing this issue, we introduce cryoGS, a GMM-based method that integrates Gaussian splatting with the physics of cryo-EM image formation. In particular, we develop an orthogonal projection-aware Gaussian splatting, with adaptations such as a normalization term and FFT-aligned coordinate system tailored for cryo-EM imaging. All these innovations enable stable and efficient homogeneous reconstruction directly from raw cryo-EM particle images using random initialization. Experimental results on real datasets validate the effectiveness and robustness of cryoGS over representative baselines. The code will be released upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10809v3">GaSLight: Gaussian Splats for Spatially-Varying Lighting in HDR</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      We present GaSLight, a method that generates spatially-varying lighting from regular images. Our method proposes using HDR Gaussian Splats as light source representation, marking the first time regular images can serve as light sources in a 3D renderer. Our two-stage process first enhances the dynamic range of images plausibly and accurately by leveraging the priors embedded in diffusion models. Next, we employ Gaussian Splats to model 3D lighting, achieving spatially variant lighting. Our approach yields state-of-the-art results on HDR estimations and their applications in illuminating virtual objects and scenes. To facilitate the benchmarking of images as light sources, we introduce a novel dataset of calibrated and unsaturated HDR to evaluate images as light sources. We assess our method using a combination of this novel dataset and an existing dataset from the literature. Project page: https://lvsn.github.io/gaslight/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11668v1">Neural Gaussian Radio Fields for Channel Estimation</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 This paper has been submitted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Accurate channel state information (CSI) remains the most critical bottleneck in modern wireless networks, with pilot overhead consuming up to 11-21% of transmission bandwidth, increasing latency by 20-40% in massive MIMO systems, and reducing potential spectral efficiency by over 53%. Traditional estimation techniques fundamentally fail under mobility, with feedback delays as small as 4 ms causing 50% throughput degradation at even modest speeds (30 km/h). We present neural Gaussian radio fields (nGRF), a novel framework that leverages explicit 3D Gaussian primitives to synthesize complex channel matrices accurately and efficiently. Unlike NeRF-based approaches that rely on slow implicit representations or existing Gaussian splatting methods that use non-physical 2D projections, nGRF performs direct 3D electromagnetic field aggregation, with each Gaussian acting as a localized radio modulator. nGRF demonstrates superior performance across diverse environments: in indoor scenarios, it achieves a 10.9$\times$ higher prediction SNR than state of the art methods while reducing inference latency from 242 ms to just 1.1 ms (a 220$\times$ speedup). For large-scale outdoor environments, where existing approaches fail to function, nGRF achieves an SNR of 26.2 dB. Moreover, nGRF requires only 0.011 measurements per cubic foot compared to 0.2-178.1 for existing methods, thereby reducing data collection burden by 18$\times$. Training time is similarly reduced from hours to minutes (a 180$\times$ reduction), enabling rapid adaptation to dynamic environments. The code and datasets are available at: https://github.com/anonym-auth/n-grf
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03643v1">Uni3R: Unified 3D Reconstruction and Semantic Understanding via Generalizable Gaussian Splatting from Unposed Multi-View Images</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 The code is available at https://github.com/HorizonRobotics/Uni3R
    </div>
    <details class="paper-abstract">
      Reconstructing and semantically interpreting 3D scenes from sparse 2D views remains a fundamental challenge in computer vision. Conventional methods often decouple semantic understanding from reconstruction or necessitate costly per-scene optimization, thereby restricting their scalability and generalizability. In this paper, we introduce Uni3R, a novel feed-forward framework that jointly reconstructs a unified 3D scene representation enriched with open-vocabulary semantics, directly from unposed multi-view images. Our approach leverages a Cross-View Transformer to robustly integrate information across arbitrary multi-view inputs, which then regresses a set of 3D Gaussian primitives endowed with semantic feature fields. This unified representation facilitates high-fidelity novel view synthesis, open-vocabulary 3D semantic segmentation, and depth prediction, all within a single, feed-forward pass. Extensive experiments demonstrate that Uni3R establishes a new state-of-the-art across multiple benchmarks, including 25.07 PSNR on RE10K and 55.84 mIoU on ScanNet. Our work signifies a novel paradigm towards generalizable, unified 3D scene reconstruction and understanding. The code is available at https://github.com/HorizonRobotics/Uni3R.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02493v2">Low-Frequency First: Eliminating Floating Artifacts in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Project Website: https://jcwang-gh.github.io/EFA-GS
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a powerful and computationally efficient representation for 3D reconstruction. Despite its strengths, 3DGS often produces floating artifacts, which are erroneous structures detached from the actual geometry and significantly degrade visual fidelity. The underlying mechanisms causing these artifacts, particularly in low-quality initialization scenarios, have not been fully explored. In this paper, we investigate the origins of floating artifacts from a frequency-domain perspective and identify under-optimized Gaussians as the primary source. Based on our analysis, we propose \textit{Eliminating-Floating-Artifacts} Gaussian Splatting (EFA-GS), which selectively expands under-optimized Gaussians to prioritize accurate low-frequency learning. Additionally, we introduce complementary depth-based and scale-based strategies to dynamically refine Gaussian expansion, effectively mitigating detail erosion. Extensive experiments on both synthetic and real-world datasets demonstrate that EFA-GS substantially reduces floating artifacts while preserving high-frequency details, achieving an improvement of 1.68 dB in PSNR over baseline method on our RWLQ dataset. Furthermore, we validate the effectiveness of our approach in downstream 3D editing tasks. We provide our implementation in https://jcwang-gh.github.io/EFA-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17044v2">4D Scaffold Gaussian Splatting with Dynamic-Aware Anchor Growing for Efficient and High-Fidelity Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Modeling dynamic scenes through 4D Gaussians offers high visual fidelity and fast rendering speeds, but comes with significant storage overhead. Recent approaches mitigate this cost by aggressively reducing the number of Gaussians. However, this inevitably removes Gaussians essential for high-quality rendering, leading to severe degradation in dynamic regions. In this paper, we introduce a novel 4D anchor-based framework that tackles the storage cost in different perspective. Rather than reducing the number of Gaussians, our method retains a sufficient quantity to accurately model dynamic contents, while compressing them into compact, grid-aligned 4D anchor features. Each anchor is processed by an MLP to spawn a set of neural 4D Gaussians, which represent a local spatiotemporal region. We design these neural 4D Gaussians to capture temporal changes with minimal parameters, making them well-suited for the MLP-based spawning. Moreover, we introduce a dynamic-aware anchor growing strategy to effectively assign additional anchors to under-reconstructed dynamic regions. Our method adjusts the accumulated gradients with Gaussians' temporal coverage, significantly improving reconstruction quality in dynamic regions. Experimental results highlight that our method achieves state-of-the-art visual quality in dynamic regions, outperforming all baselines by a large margin with practical storage costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14537v2">Personalize Your Gaussian: Consistent 3D Scene Personalization from a Single Image</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Personalizing 3D scenes from a single reference image enables intuitive user-guided editing, which requires achieving both multi-view consistency across perspectives and referential consistency with the input image. However, these goals are particularly challenging due to the viewpoint bias caused by the limited perspective provided in a single image. Lacking the mechanisms to effectively expand reference information beyond the original view, existing methods of image-conditioned 3DGS personalization often suffer from this viewpoint bias and struggle to produce consistent results. Therefore, in this paper, we present Consistent Personalization for 3D Gaussian Splatting (CP-GS), a framework that progressively propagates the single-view reference appearance to novel perspectives. In particular, CP-GS integrates pre-trained image-to-3D generation and iterative LoRA fine-tuning to extract and extend the reference appearance, and finally produces faithful multi-view guidance images and the personalized 3DGS outputs through a view-consistent generation process guided by geometric cues. Extensive experiments on real-world scenes show that our CP-GS effectively mitigates the viewpoint bias, achieving high-quality personalization that significantly outperforms existing methods. The code will be released at https://github.com/Yuxuan-W/CP-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03227v1">Trace3D: Consistent Segmentation Lifting via Gaussian Instance Tracing</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      We address the challenge of lifting 2D visual segmentation to 3D in Gaussian Splatting. Existing methods often suffer from inconsistent 2D masks across viewpoints and produce noisy segmentation boundaries as they neglect these semantic cues to refine the learned Gaussians. To overcome this, we introduce Gaussian Instance Tracing (GIT), which augments the standard Gaussian representation with an instance weight matrix across input views. Leveraging the inherent consistency of Gaussians in 3D, we use this matrix to identify and correct 2D segmentation inconsistencies. Furthermore, since each Gaussian ideally corresponds to a single object, we propose a GIT-guided adaptive density control mechanism to split and prune ambiguous Gaussians during training, resulting in sharper and more coherent 2D and 3D segmentation boundaries. Experimental results show that our method extracts clean 3D assets and consistently improves 3D segmentation in both online (e.g., self-prompting) and offline (e.g., contrastive lifting) settings, enabling applications such as hierarchical segmentation, object extraction, and scene editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03180v1">Duplex-GS: Proxy-Guided Weighted Blending for Real-Time Order-Independent Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated remarkable rendering fidelity and efficiency. However, these methods still rely on computationally expensive sequential alpha-blending operations, resulting in significant overhead, particularly on resource-constrained platforms. In this paper, we propose Duplex-GS, a dual-hierarchy framework that integrates proxy Gaussian representations with order-independent rendering techniques to achieve photorealistic results while sustaining real-time performance. To mitigate the overhead caused by view-adaptive radix sort, we introduce cell proxies for local Gaussians management and propose cell search rasterization for further acceleration. By seamlessly combining our framework with Order-Independent Transparency (OIT), we develop a physically inspired weighted sum rendering technique that simultaneously eliminates "popping" and "transparency" artifacts, yielding substantial improvements in both accuracy and efficiency. Extensive experiments on a variety of real-world datasets demonstrate the robustness of our method across diverse scenarios, including multi-scale training views and large-scale environments. Our results validate the advantages of the OIT rendering paradigm in Gaussian Splatting, achieving high-quality rendering with an impressive 1.5 to 4 speedup over existing OIT based Gaussian Splatting approaches and 52.2% to 86.9% reduction of the radix sort overhead without quality degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03118v1">H3R: Hybrid Multi-view Correspondence for Generalizable 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 ICCV 2025
    </div>
    <details class="paper-abstract">
      Despite recent advances in feed-forward 3D Gaussian Splatting, generalizable 3D reconstruction remains challenging, particularly in multi-view correspondence modeling. Existing approaches face a fundamental trade-off: explicit methods achieve geometric precision but struggle with ambiguous regions, while implicit methods provide robustness but suffer from slow convergence. We present H3R, a hybrid framework that addresses this limitation by integrating volumetric latent fusion with attention-based feature aggregation. Our framework consists of two complementary components: an efficient latent volume that enforces geometric consistency through epipolar constraints, and a camera-aware Transformer that leverages Pl\"ucker coordinates for adaptive correspondence refinement. By integrating both paradigms, our approach enhances generalization while converging 2$\times$ faster than existing methods. Furthermore, we show that spatial-aligned foundation models (e.g., SD-VAE) substantially outperform semantic-aligned models (e.g., DINOv2), resolving the mismatch between semantic representations and spatial reconstruction requirements. Our method supports variable-number and high-resolution input views while demonstrating robust cross-dataset generalization. Extensive experiments show that our method achieves state-of-the-art performance across multiple benchmarks, with significant PSNR improvements of 0.59 dB, 1.06 dB, and 0.22 dB on the RealEstate10K, ACID, and DTU datasets, respectively. Code is available at https://github.com/JiaHeng-DLUT/H3R.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03077v1">RobustGS: Unified Boosting of Feedforward 3D Gaussian Splatting under Low-Quality Conditions</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Feedforward 3D Gaussian Splatting (3DGS) overcomes the limitations of optimization-based 3DGS by enabling fast and high-quality reconstruction without the need for per-scene optimization. However, existing feedforward approaches typically assume that input multi-view images are clean and high-quality. In real-world scenarios, images are often captured under challenging conditions such as noise, low light, or rain, resulting in inaccurate geometry and degraded 3D reconstruction. To address these challenges, we propose a general and efficient multi-view feature enhancement module, RobustGS, which substantially improves the robustness of feedforward 3DGS methods under various adverse imaging conditions, enabling high-quality 3D reconstruction. The RobustGS module can be seamlessly integrated into existing pretrained pipelines in a plug-and-play manner to enhance reconstruction robustness. Specifically, we introduce a novel component, Generalized Degradation Learner, designed to extract generic representations and distributions of multiple degradations from multi-view inputs, thereby enhancing degradation-awareness and improving the overall quality of 3D reconstruction. In addition, we propose a novel semantic-aware state-space model. It first leverages the extracted degradation representations to enhance corrupted inputs in the feature space. Then, it employs a semantic-aware strategy to aggregate semantically similar information across different views, enabling the extraction of fine-grained cross-view correspondences and further improving the quality of 3D representations. Extensive experiments demonstrate that our approach, when integrated into existing methods in a plug-and-play manner, consistently achieves state-of-the-art reconstruction quality across various types of degradations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05280v3">Unifying Appearance Codes and Bilateral Grids for Driving Scene Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Project page: https://bigcileng.github.io/bilateral-driving ; Code: https://github.com/BigCiLeng/bilateral-driving
    </div>
    <details class="paper-abstract">
      Neural rendering techniques, including NeRF and Gaussian Splatting (GS), rely on photometric consistency to produce high-quality reconstructions. However, in real-world scenarios, it is challenging to guarantee perfect photometric consistency in acquired images. Appearance codes have been widely used to address this issue, but their modeling capability is limited, as a single code is applied to the entire image. Recently, the bilateral grid was introduced to perform pixel-wise color mapping, but it is difficult to optimize and constrain effectively. In this paper, we propose a novel multi-scale bilateral grid that unifies appearance codes and bilateral grids. We demonstrate that this approach significantly improves geometric accuracy in dynamic, decoupled autonomous driving scene reconstruction, outperforming both appearance codes and bilateral grids. This is crucial for autonomous driving, where accurate geometry is important for obstacle avoidance and control. Our method shows strong results across four datasets: Waymo, NuScenes, Argoverse, and PandaSet. We further demonstrate that the improvement in geometry is driven by the multi-scale bilateral grid, which effectively reduces floaters caused by photometric inconsistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12137v3">AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      Modeling and rendering dynamic urban driving scenes is crucial for self-driving simulation. Current high-quality methods typically rely on costly manual object tracklet annotations, while self-supervised approaches fail to capture dynamic object motions accurately and decompose scenes properly, resulting in rendering artifacts. We introduce AD-GS, a novel self-supervised framework for high-quality free-viewpoint rendering of driving scenes from a single log. At its core is a novel learnable motion model that integrates locality-aware B-spline curves with global-aware trigonometric functions, enabling flexible yet precise dynamic object modeling. Rather than requiring comprehensive semantic labeling, AD-GS automatically segments scenes into objects and background with the simplified pseudo 2D segmentation, representing objects using dynamic Gaussians and bidirectional temporal visibility masks. Further, our model incorporates visibility reasoning and physically rigid regularization to enhance robustness. Extensive evaluations demonstrate that our annotation-free model significantly outperforms current state-of-the-art annotation-free methods and is competitive with annotation-dependent approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03017v1">SA-3DGS: A Self-Adaptive Compression Method for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 9 pages, 7 figures. Under review at AAAI 2026
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting have enhanced efficient and high-quality novel view synthesis. However, representing scenes requires a large number of Gaussian points, leading to high storage demands and limiting practical deployment. The latest methods facilitate the compression of Gaussian models but struggle to identify truly insignificant Gaussian points in the scene, leading to a decline in subsequent Gaussian pruning, compression quality, and rendering performance. To address this issue, we propose SA-3DGS, a method that significantly reduces storage costs while maintaining rendering quality. SA-3DGS learns an importance score to automatically identify the least significant Gaussians in scene reconstruction, thereby enabling effective pruning and redundancy reduction. Next, the importance-aware clustering module compresses Gaussians attributes more accurately into the codebook, improving the codebook's expressive capability while reducing model size. Finally, the codebook repair module leverages contextual scene information to repair the codebook, thereby recovering the original Gaussian point attributes and mitigating the degradation in rendering quality caused by information loss. Experimental results on several benchmark datasets show that our method achieves up to 66x compression while maintaining or even improving rendering quality. The proposed Gaussian pruning approach is not only adaptable to but also improves other pruning-based methods (e.g., LightGaussian), showcasing excellent performance and strong generalization ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19830v2">Taking Language Embedded 3D Gaussian Splatting into the Wild</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Visit our project page at https://yuzewang1998.github.io/takinglangsplatw/
    </div>
    <details class="paper-abstract">
      Recent advances in leveraging large-scale Internet photo collections for 3D reconstruction have enabled immersive virtual exploration of landmarks and historic sites worldwide. However, little attention has been given to the immersive understanding of architectural styles and structural knowledge, which remains largely confined to browsing static text-image pairs. Therefore, can we draw inspiration from 3D in-the-wild reconstruction techniques and use unconstrained photo collections to create an immersive approach for understanding the 3D structure of architectural components? To this end, we extend language embedded 3D Gaussian splatting (3DGS) and propose a novel framework for open-vocabulary scene understanding from unconstrained photo collections. Specifically, we first render multiple appearance images from the same viewpoint as the unconstrained image with the reconstructed radiance field, then extract multi-appearance CLIP features and two types of language feature uncertainty maps-transient and appearance uncertainty-derived from the multi-appearance features to guide the subsequent optimization process. Next, we propose a transient uncertainty-aware autoencoder, a multi-appearance language field 3DGS representation, and a post-ensemble strategy to effectively compress, learn, and fuse language features from multiple appearances. Finally, to quantitatively evaluate our method, we introduce PT-OVS, a new benchmark dataset for assessing open-vocabulary segmentation performance on unconstrained photo collections. Experimental results show that our method outperforms existing methods, delivering accurate open-vocabulary segmentation and enabling applications such as interactive roaming with open-vocabulary queries, architectural style pattern recognition, and 3D scene editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20720v2">4D Gaussian Splatting: Modeling Dynamic Scenes with Native 4D Primitives</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Journal extension of ICLR 2024. arXiv admin note: text overlap with arXiv:2310.10642
    </div>
    <details class="paper-abstract">
      Dynamic 3D scene representation and novel view synthesis are crucial for enabling immersive experiences required by AR/VR and metaverse applications. It is a challenging task due to the complexity of unconstrained real-world scenes and their temporal dynamics. In this paper, we reformulate the reconstruction of a time-varying 3D scene as approximating its underlying spatiotemporal 4D volume by optimizing a collection of native 4D primitives, i.e., 4D Gaussians, with explicit geometry and appearance modeling. Equipped with a tailored rendering pipeline, our representation can be end-to-end optimized using only photometric supervision while free viewpoint viewing at interactive frame rate, making it suitable for representing real world scene with complex dynamic. This approach has been the first solution to achieve real-time rendering of high-resolution, photorealistic novel views for complex dynamic scenes. To facilitate real-world applications, we derive several compact variants that effectively reduce the memory footprint to address its storage bottleneck. Extensive experiments validate the superiority of 4DGS in terms of visual quality and efficiency across a range of dynamic scene-related tasks (e.g., novel view synthesis, 4D generation, scene understanding) and scenarios (e.g., single object, indoor scenes, driving environments, synthetic and real data).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02660v1">PMGS: Reconstruction of Projectile Motion across Large Spatiotemporal Spans via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Modeling complex rigid motion across large spatiotemporal spans remains an unresolved challenge in dynamic reconstruction. Existing paradigms are mainly confined to short-term, small-scale deformation and offer limited consideration for physical consistency. This study proposes PMGS, focusing on reconstructing Projectile Motion via 3D Gaussian Splatting. The workflow comprises two stages: 1) Target Modeling: achieving object-centralized reconstruction through dynamic scene decomposition and an improved point density control; 2) Motion Recovery: restoring full motion sequences by learning per-frame SE(3) poses. We introduce an acceleration consistency constraint to bridge Newtonian mechanics and pose estimation, and design a dynamic simulated annealing strategy that adaptively schedules learning rates based on motion states. Futhermore, we devise a Kalman fusion scheme to optimize error accumulation from multi-source observations to mitigate disturbances. Experiments show PMGS's superior performance in reconstructing high-speed nonlinear rigid motion compared to mainstream dynamic methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02493v1">Low-Frequency First: Eliminating Floating Artifacts in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a powerful and computationally efficient representation for 3D reconstruction. Despite its strengths, 3DGS often produces floating artifacts, which are erroneous structures detached from the actual geometry and significantly degrade visual fidelity. The underlying mechanisms causing these artifacts, particularly in low-quality initialization scenarios, have not been fully explored. In this paper, we investigate the origins of floating artifacts from a frequency-domain perspective and identify under-optimized Gaussians as the primary source. Based on our analysis, we propose \textit{Eliminating-Floating-Artifacts} Gaussian Splatting (EFA-GS), which selectively expands under-optimized Gaussians to prioritize accurate low-frequency learning. Additionally, we introduce complementary depth-based and scale-based strategies to dynamically refine Gaussian expansion, effectively mitigating detail erosion. Extensive experiments on both synthetic and real-world datasets demonstrate that EFA-GS substantially reduces floating artifacts while preserving high-frequency details, achieving an improvement of 1.68 dB in PSNR over baseline method on our RWLQ dataset. Furthermore, we validate the effectiveness of our approach in downstream 3D editing tasks. Our implementation will be released on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02443v1">Uncertainty Estimation for Novel Views in Gaussian Splatting from Primitive-Based Representations of Error and Visibility</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      In this work, we present a novel method for uncertainty estimation (UE) in Gaussian Splatting. UE is crucial for using Gaussian Splatting in critical applications such as robotics and medicine. Previous methods typically estimate the variance of Gaussian primitives and use the rendering process to obtain pixel-wise uncertainties. Our method establishes primitive representations of error and visibility of trainings views, which carries meaningful uncertainty information. This representation is obtained by projection of training error and visibility onto the primitives. Uncertainties of novel views are obtained by rendering the primitive representations of uncertainty for those novel views, yielding uncertainty feature maps. To aggregate these uncertainty feature maps of novel views, we perform a pixel-wise regression on holdout data. In our experiments, we analyze the different components of our method, investigating various combinations of uncertainty feature maps and regression models. Furthermore, we considered the effect of separating splatting into foreground and background. Our UEs show high correlations to true errors, outperforming state-of-the-art methods, especially on foreground objects. The trained regression models show generalization capabilities to new scenes, allowing uncertainty estimation without the need for holdout data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02408v1">GR-Gaussian: Graph-Based Radiative Gaussian Splatting for Sparse-View CT Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 10
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a promising approach for CT reconstruction. However, existing methods rely on the average gradient magnitude of points within the view, often leading to severe needle-like artifacts under sparse-view conditions. To address this challenge, we propose GR-Gaussian, a graph-based 3D Gaussian Splatting framework that suppresses needle-like artifacts and improves reconstruction accuracy under sparse-view conditions. Our framework introduces two key innovations: (1) a Denoised Point Cloud Initialization Strategy that reduces initialization errors and accelerates convergence; and (2) a Pixel-Graph-Aware Gradient Strategy that refines gradient computation using graph-based density differences, improving splitting accuracy and density representation. Experiments on X-3D and real-world datasets validate the effectiveness of GR-Gaussian, achieving PSNR improvements of 0.67 dB and 0.92 dB, and SSIM gains of 0.011 and 0.021. These results highlight the applicability of GR-Gaussian for accurate CT reconstruction under challenging sparse-view conditions.
    </details>
</div>
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
