# gaussian splatting - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24986v1">PhysTalk: Language-driven Real-time Physics in 3D Gaussian Scenes</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      Realistic visual simulations are omnipresent, yet their creation requires computing time, rendering, and expert animation knowledge. Open-vocabulary visual effects generation from text inputs emerges as a promising solution that can unlock immense creative potential. However, current pipelines lack both physical realism and effective language interfaces, requiring slow offline optimization. In contrast, PhysTalk takes a 3D Gaussian Splatting (3DGS) scene as input and translates arbitrary user prompts into real time, physics based, interactive 4D animations. A large language model (LLM) generates executable code that directly modifies 3DGS parameters through lightweight proxies and particle dynamics. Notably, PhysTalk is the first framework to couple 3DGS directly with a physics simulator without relying on time consuming mesh extraction. While remaining open vocabulary, this design enables interactive 3D Gaussian animation via collision aware, physics based manipulation of arbitrary, multi material objects. Finally, PhysTalk is train-free and computationally lightweight: this makes 4D animation broadly accessible and shifts these workflows from a "render and wait" paradigm toward an interactive dialogue with a modern, physics-informed pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24763v1">UniC-Lift: Unified 3D Instance Segmentation via Contrastive Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-31
      | 💬 Accepted to AAAI 2026. Project Page: https://unic-lift.github.io/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) and Neural Radiance Fields (NeRF) have advanced novel-view synthesis. Recent methods extend multi-view 2D segmentation to 3D, enabling instance/semantic segmentation for better scene understanding. A key challenge is the inconsistency of 2D instance labels across views, leading to poor 3D predictions. Existing methods use a two-stage approach in which some rely on contrastive learning with hyperparameter-sensitive clustering, while others preprocess labels for consistency. We propose a unified framework that merges these steps, reducing training time and improving performance by introducing a learnable feature embedding for segmentation in Gaussian primitives. This embedding is then efficiently decoded into instance labels through a novel "Embedding-to-Label" process, effectively integrating the optimization. While this unified framework offers substantial benefits, we observed artifacts at the object boundaries. To address the object boundary issues, we propose hard-mining samples along these boundaries. However, directly applying hard mining to the feature embeddings proved unstable. Therefore, we apply a linear layer to the rasterized feature embeddings before calculating the triplet loss, which stabilizes training and significantly improves performance. Our method outperforms baselines qualitatively and quantitatively on the ScanNet, Replica3D, and Messy-Rooms datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24742v1">Splatwizard: A Benchmark Toolkit for 3D Gaussian Splatting Compression</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      The recent advent of 3D Gaussian Splatting (3DGS) has marked a significant breakthrough in real-time novel view synthesis. However, the rapid proliferation of 3DGS-based algorithms has created a pressing need for standardized and comprehensive evaluation tools, especially for compression task. Existing benchmarks often lack the specific metrics necessary to holistically assess the unique characteristics of different methods, such as rendering speed, rate distortion trade-offs memory efficiency, and geometric accuracy. To address this gap, we introduce Splatwizard, a unified benchmark toolkit designed specifically for benchmarking 3DGS compression models. Splatwizard provides an easy-to-use framework to implement new 3DGS compression model and utilize state-of-the-art techniques proposed by previous work. Besides, an integrated pipeline that automates the calculation of key performance indicators, including image-based quality metrics, chamfer distance of reconstruct mesh, rendering frame rates, and computational resource consumption is included in the framework as well. Code is available at https://github.com/splatwizard/splatwizard
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02261v3">SplatSSC: Decoupled Depth-Guided Gaussian Splatting for Semantic Scene Completion</a></div>
    <div class="paper-meta">
      📅 2025-12-31
      | 💬 Accepted for oral presentation in The 40th Annual AAAI Conference on Artificial Intelligence (AAAI 2026)
    </div>
    <details class="paper-abstract">
      Monocular 3D Semantic Scene Completion (SSC) is a challenging yet promising task that aims to infer dense geometric and semantic descriptions of a scene from a single image. While recent object-centric paradigms significantly improve efficiency by leveraging flexible 3D Gaussian primitives, they still rely heavily on a large number of randomly initialized primitives, which inevitably leads to 1) inefficient primitive initialization and 2) outlier primitives that introduce erroneous artifacts. In this paper, we propose SplatSSC, a novel framework that resolves these limitations with a depth-guided initialization strategy and a principled Gaussian aggregator. Instead of random initialization, SplatSSC utilizes a dedicated depth branch composed of a Group-wise Multi-scale Fusion (GMF) module, which integrates multi-scale image and depth features to generate a sparse yet representative set of initial Gaussian primitives. To mitigate noise from outlier primitives, we develop the Decoupled Gaussian Aggregator (DGA), which enhances robustness by decomposing geometric and semantic predictions during the Gaussian-to-voxel splatting process. Complemented with a specialized Probability Scale Loss, our method achieves state-of-the-art performance on the Occ-ScanNet dataset, outperforming prior approaches by over 6.3% in IoU and 4.1% in mIoU, while reducing both latency and memory cost by more than 9.3%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19108v2">GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-30
      | 💬 Accepted to AAAI 2026. Code URL:https://github.com/Sweethyh/GaussianImage_plus.git
    </div>
    <details class="paper-abstract">
      Implicit neural representations (INRs) have achieved remarkable success in image representation and compression, but they require substantial training time and memory. Meanwhile, recent 2D Gaussian Splatting (GS) methods (\textit{e.g.}, GaussianImage) offer promising alternatives through efficient primitive-based rendering. However, these methods require excessive Gaussian primitives to maintain high visual fidelity. To exploit the potential of GS-based approaches, we present GaussianImage++, which utilizes limited Gaussian primitives to achieve impressive representation and compression performance. Firstly, we introduce a distortion-driven densification mechanism. It progressively allocates Gaussian primitives according to signal intensity. Secondly, we employ context-aware Gaussian filters for each primitive, which assist in the densification to optimize Gaussian primitives based on varying image content. Thirdly, we integrate attribute-separated learnable scalar quantizers and quantization-aware training, enabling efficient compression of primitive attributes. Experimental results demonstrate the effectiveness of our method. In particular, GaussianImage++ outperforms GaussianImage and INRs-based COIN in representation and compression performance while maintaining real-time decoding and low memory usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23054v2">Differentiable Physics-Driven Human Representation for Millimeter-Wave Based Pose Estimation</a></div>
    <div class="paper-meta">
      📅 2025-12-30
    </div>
    <details class="paper-abstract">
      While millimeter-wave (mmWave) presents advantages for Human Pose Estimation (HPE) through its non-intrusive sensing capabilities, current mmWave-based HPE methods face limitations in two predominant input paradigms: Heatmap and Point Cloud (PC). Heatmap represents dense multi-dimensional features derived from mmWave, but is significantly affected by multipath propagation and hardware modulation noise. PC, a set of 3D points, is obtained by applying the Constant False Alarm Rate algorithm to the Heatmap, which suppresses noise but results in sparse human-related features. To address these limitations, we study the feasibility of providing an alternative input paradigm: Differentiable Physics-driven Human Representation (DIPR), which represents humans as an ensemble of Gaussian distributions with kinematic and electromagnetic parameters. Inspired by Gaussian Splatting, DIPR leverages human kinematic priors and mmWave propagation physics to enhance human features while mitigating non-human noise through two strategies: 1) We incorporate prior kinematic knowledge to initialize DIPR based on the Heatmap and establish multi-faceted optimization objectives, ensuring biomechanical validity and enhancing motion features. 2) We simulate complete mmWave processing pipelines, re-render a new Heatmap from DIPR, and compare it with the original Heatmap, avoiding spurious noise generation due to kinematic constraints overfitting. Experimental results on three datasets with four methods demonstrate that existing mmWave-based HPE methods can easily integrate DIPR and achieve superior performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24018v1">Structure-Guided Allocation of 2D Gaussians for Image Representation and Compression</a></div>
    <div class="paper-meta">
      📅 2025-12-30
    </div>
    <details class="paper-abstract">
      Recent advances in 2D Gaussian Splatting (2DGS) have demonstrated its potential as a compact image representation with millisecond-level decoding. However, existing 2DGS-based pipelines allocate representation capacity and parameter precision largely oblivious to image structure, limiting their rate-distortion (RD) efficiency at low bitrates. To address this, we propose a structure-guided allocation principle for 2DGS, which explicitly couples image structure with both representation capacity and quantization precision, while preserving native decoding speed. First, we introduce a structure-guided initialization that assigns 2D Gaussians according to spatial structural priors inherent in natural images, yielding a localized and semantically meaningful distribution. Second, during quantization-aware fine-tuning, we propose adaptive bitwidth quantization of covariance parameters, which grants higher precision to small-scale Gaussians in complex regions and lower precision elsewhere, enabling RD-aware optimization, thereby reducing redundancy without degrading edge quality. Third, we impose a geometry-consistent regularization that aligns Gaussian orientations with local gradient directions to better preserve structural details. Extensive experiments demonstrate that our approach substantially improves both the representational power and the RD performance of 2DGS while maintaining over 1000 FPS decoding. Compared with the baseline GSImage, we reduce BD-rate by 43.44% on Kodak and 29.91% on DIV2K.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22489v2">Tracking by Predicting 3-D Gaussians Over Time</a></div>
    <div class="paper-meta">
      📅 2025-12-30
    </div>
    <details class="paper-abstract">
      We propose Video Gaussian Masked Autoencoders (Video-GMAE), a self-supervised approach for representation learning that encodes a sequence of images into a set of Gaussian splats moving over time. Representing a video as a set of Gaussians enforces a reasonable inductive bias: that 2-D videos are often consistent projections of a dynamic 3-D scene. We find that tracking emerges when pretraining a network with this architecture. Mapping the trajectory of the learnt Gaussians onto the image plane gives zero-shot tracking performance comparable to state-of-the-art. With small-scale finetuning, our models achieve 34.6% improvement on Kinetics, and 13.1% on Kubric datasets, surpassing existing self-supervised video approaches. The project page and code are publicly available at https://videogmae.org/ and https://github.com/tekotan/video-gmae.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23998v1">Improved 3D Gaussian Splatting of Unknown Spacecraft Structure Using Space Environment Illumination Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-12-30
      | 💬 Presented at 2025 IEEE International Conference on Space Robotics (iSpaRo)
    </div>
    <details class="paper-abstract">
      This work presents a novel pipeline to recover the 3D structure of an unknown target spacecraft from a sequence of images captured during Rendezvous and Proximity Operations (RPO) in space. The target's geometry and appearance are represented as a 3D Gaussian Splatting (3DGS) model. However, learning 3DGS requires static scenes, an assumption in contrast to dynamic lighting conditions encountered in spaceborne imagery. The trained 3DGS model can also be used for camera pose estimation through photometric optimization. Therefore, in addition to recovering a geometrically accurate 3DGS model, the photometric accuracy of the rendered images is imperative to downstream pose estimation tasks during the RPO process. This work proposes to incorporate the prior knowledge of the Sun's position, estimated and maintained by the servicer spacecraft, into the training pipeline for improved photometric quality of 3DGS rasterization. Experimental studies demonstrate the effectiveness of the proposed solution, as 3DGS models trained on a sequence of images learn to adapt to rapidly changing illumination conditions in space and reflect global shadowing and self-occlusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.05859v4">D-FCGS: Feedforward Compression of Dynamic Gaussian Splatting for Free-Viewpoint Videos</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 code:https://github.com/Mr-Zwkid/D-FCGS
    </div>
    <details class="paper-abstract">
      Free-Viewpoint Video (FVV) enables immersive 3D experiences, but efficient compression of dynamic 3D representation remains a major challenge. Existing dynamic 3D Gaussian Splatting methods couple reconstruction with optimization-dependent compression and customized motion formats, limiting generalization and standardization. To address this, we propose D-FCGS, a novel Feedforward Compression framework for Dynamic Gaussian Splatting. Key innovations include: (1) a standardized Group-of-Frames (GoF) structure with I-P coding, leveraging sparse control points to extract inter-frame motion tensors; (2) a dual prior-aware entropy model that fuses hyperprior and spatial-temporal priors for accurate rate estimation; (3) a control-point-guided motion compensation mechanism and refinement network to enhance view-consistent fidelity. Trained on Gaussian frames derived from multi-view videos, D-FCGS generalizes across diverse scenes in a zero-shot fashion. Experiments show that it matches the rate-distortion performance of optimization-based methods, achieving over 17 times compression compared to the baseline while preserving visual quality across viewpoints. This work advances feedforward compression of dynamic 3DGS, facilitating scalable FVV transmission and storage for immersive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23255v1">Contour Information Aware 2D Gaussian Splatting for Image Representation</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Image representation is a fundamental task in computer vision. Recently, Gaussian Splatting has emerged as an efficient representation framework, and its extension to 2D image representation enables lightweight, yet expressive modeling of visual content. While recent 2D Gaussian Splatting (2DGS) approaches provide compact storage and real-time decoding, they often produce blurry or indistinct boundaries when the number of Gaussians is small due to the lack of contour awareness. In this work, we propose a Contour Information-Aware 2D Gaussian Splatting framework that incorporates object segmentation priors into Gaussian-based image representation. By constraining each Gaussian to a specific segmentation region during rasterization, our method prevents cross-boundary blending and preserves edge structures under high compression. We also introduce a warm-up scheme to stabilize training and improve convergence. Experiments on synthetic color charts and the DAVIS dataset demonstrate that our approach achieves higher reconstruction quality around object edges compared to existing 2DGS methods. The improvement is particularly evident in scenarios with very few Gaussians, while our method still maintains fast rendering and low memory usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23176v1">GVSynergy-Det: Synergistic Gaussian-Voxel Representations for Multi-View 3D Object Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 11 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Image-based 3D object detection aims to identify and localize objects in 3D space using only RGB images, eliminating the need for expensive depth sensors required by point cloud-based methods. Existing image-based approaches face two critical challenges: methods achieving high accuracy typically require dense 3D supervision, while those operating without such supervision struggle to extract accurate geometry from images alone. In this paper, we present GVSynergy-Det, a novel framework that enhances 3D detection through synergistic Gaussian-Voxel representation learning. Our key insight is that continuous Gaussian and discrete voxel representations capture complementary geometric information: Gaussians excel at modeling fine-grained surface details while voxels provide structured spatial context. We introduce a dual-representation architecture that: 1) adapts generalizable Gaussian Splatting to extract complementary geometric features for detection tasks, and 2) develops a cross-representation enhancement mechanism that enriches voxel features with geometric details from Gaussian fields. Unlike previous methods that either rely on time-consuming per-scene optimization or utilize Gaussian representations solely for depth regularization, our synergistic strategy directly leverages features from both representations through learnable integration, enabling more accurate object localization. Extensive experiments demonstrate that GVSynergy-Det achieves state-of-the-art results on challenging indoor benchmarks, significantly outperforming existing methods on both ScanNetV2 and ARKitScenes datasets, all without requiring any depth or dense 3D geometry supervision (e.g., point clouds or TSDF).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23054v1">Differentiable Physics-Driven Human Representation for Millimeter-Wave Based Pose Estimation</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      While millimeter-wave (mmWave) presents advantages for Human Pose Estimation (HPE) through its non-intrusive sensing capabilities, current mmWave-based HPE methods face limitations in two predominant input paradigms: Heatmap and Point Cloud (PC). Heatmap represents dense multi-dimensional features derived from mmWave, but is significantly affected by multipath propagation and hardware modulation noise. PC, a set of 3D points, is obtained by applying the Constant False Alarm Rate algorithm to the Heatmap, which suppresses noise but results in sparse human-related features. To address these limitations, we study the feasibility of providing an alternative input paradigm: Differentiable Physics-driven Human Representation (DIPR), which represents humans as an ensemble of Gaussian distributions with kinematic and electromagnetic parameters. Inspired by Gaussian Splatting, DIPR leverages human kinematic priors and mmWave propagation physics to enhance human features while mitigating non-human noise through two strategies: 1) We incorporate prior kinematic knowledge to initialize DIPR based on the Heatmap and establish multi-faceted optimization objectives, ensuring biomechanical validity and enhancing motion features. 2) We simulate complete mmWave processing pipelines, re-render a new Heatmap from DIPR, and compare it with the original Heatmap, avoiding spurious noise generation due to kinematic constraints overfitting. Experimental results on three datasets with four methods demonstrate that existing mmWave-based HPE methods can easily integrate DIPR and achieve superior performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22882v1">Hash Grid Feature Pruning</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      Hash grids are widely used to learn an implicit neural field for Gaussian splatting, serving either as part of the entropy model or for inter-frame prediction. However, due to the irregular and non-uniform distribution of Gaussian splats in 3D space, numerous sparse regions exist, rendering many features in the hash grid invalid. This leads to redundant storage and transmission overhead. In this work, we propose a hash grid feature pruning method that identifies and prunes invalid features based on the coordinates of the input Gaussian splats, so that only the valid features are encoded. This approach reduces the storage size of the hash grid without compromising model performance, leading to improved rate-distortion performance. Following the Common Test Conditions (CTC) defined by the standardization committee, our method achieves an average bitrate reduction of 8% compared to the baseline approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22771v1">Next Best View Selections for Semantic and Dynamic 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      Understanding semantics and dynamics has been crucial for embodied agents in various tasks. Both tasks have much more data redundancy than the static scene understanding task. We formulate the view selection problem as an active learning problem, where the goal is to prioritize frames that provide the greatest information gain for model training. To this end, we propose an active learning algorithm with Fisher Information that quantifies the informativeness of candidate views with respect to both semantic Gaussian parameters and deformation networks. This formulation allows our method to jointly handle semantic reasoning and dynamic scene modeling, providing a principled alternative to heuristic or random strategies. We evaluate our method on large-scale static images and dynamic video datasets by selecting informative frames from multi-camera setups. Experimental results demonstrate that our approach consistently improves rendering quality and semantic segmentation performance, outperforming baseline methods based on random selection and uncertainty-based heuristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22706v1">SCPainter: A Unified Framework for Realistic 3D Asset Insertion and Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-12-27
    </div>
    <details class="paper-abstract">
      3D Asset insertion and novel view synthesis (NVS) are key components for autonomous driving simulation, enhancing the diversity of training data. With better training data that is diverse and covers a wide range of situations, including long-tailed driving scenarios, autonomous driving models can become more robust and safer. This motivates a unified simulation framework that can jointly handle realistic integration of inserted 3D assets and NVS. Recent 3D asset reconstruction methods enable reconstruction of dynamic actors from video, supporting their re-insertion into simulated driving scenes. While the overall structure and appearance can be accurate, it still struggles to capture the realism of 3D assets through lighting or shadows, particularly when inserted into scenes. In parallel, recent advances in NVS methods have demonstrated promising results in synthesizing viewpoints beyond the originally recorded trajectories. However, existing approaches largely treat asset insertion and NVS capabilities in isolation. To allow for interaction with the rest of the scene and to enable more diverse creation of new scenarios for training, realistic 3D asset insertion should be combined with NVS. To address this, we present SCPainter (Street Car Painter), a unified framework which integrates 3D Gaussian Splat (GS) car asset representations and 3D scene point clouds with diffusion-based generation to jointly enable realistic 3D asset insertion and NVS. The 3D GS assets and 3D scene point clouds are projected together into novel views, and these projections are used to condition a diffusion model to generate high quality images. Evaluation on the Waymo Open Dataset demonstrate the capability of our framework to enable 3D asset insertion and NVS, facilitating the creation of diverse and realistic driving data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10369v2">Breaking the Vicious Cycle: Coherent 3D Gaussian Splatting from Sparse and Motion-Blurred Views</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 20 pages, 14 figures. Manuscript v2: add the view selection of training in the appendix
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a state-of-the-art method for novel view synthesis. However, its performance heavily relies on dense, high-quality input imagery, an assumption that is often violated in real-world applications, where data is typically sparse and motion-blurred. These two issues create a vicious cycle: sparse views ignore the multi-view constraints necessary to resolve motion blur, while motion blur erases high-frequency details crucial for aligning the limited views. Thus, reconstruction often fails catastrophically, with fragmented views and a low-frequency bias. To break this cycle, we introduce CoherentGS, a novel framework for high-fidelity 3D reconstruction from sparse and blurry images. Our key insight is to address these compound degradations using a dual-prior strategy. Specifically, we combine two pre-trained generative models: a specialized deblurring network for restoring sharp details and providing photometric guidance, and a diffusion model that offers geometric priors to fill in unobserved regions of the scene. This dual-prior strategy is supported by several key techniques, including a consistency-guided camera exploration module that adaptively guides the generative process, and a depth regularization loss that ensures geometric plausibility. We evaluate CoherentGS through both quantitative and qualitative experiments on synthetic and real-world scenes, using as few as 3, 6, and 9 input views. Our results demonstrate that CoherentGS significantly outperforms existing methods, setting a new state-of-the-art for this challenging task. The code and video demos are available at https://potatobigroom.github.io/CoherentGS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17719v2">RaindropGS: A Benchmark for 3D Gaussian Splatting under Raindrop Conditions</a></div>
    <div class="paper-meta">
      📅 2025-12-27
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) under raindrop conditions suffers from severe occlusions and optical distortions caused by raindrop contamination on the camera lens, substantially degrading reconstruction quality. Existing benchmarks typically evaluate 3DGS using synthetic raindrop images with known camera poses (constrained images), assuming ideal conditions. However, in real-world scenarios, raindrops often interfere with accurate camera pose estimation and point cloud initialization. Moreover, a significant domain gap between synthetic and real raindrops further impairs generalization. To tackle these issues, we introduce RaindropGS, a comprehensive benchmark designed to evaluate the full 3DGS pipeline-from unconstrained, raindrop-corrupted images to clear 3DGS reconstructions. Specifically, the whole benchmark pipeline consists of three parts: data preparation, data processing, and raindrop-aware 3DGS evaluation, including types of raindrop interference, camera pose estimation and point cloud initialization, single image rain removal comparison, and 3D Gaussian training comparison. First, we collect a real-world raindrop reconstruction dataset, in which each scene contains three aligned image sets: raindrop-focused, background-focused, and rain-free ground truth, enabling a comprehensive evaluation of reconstruction quality under different focus conditions. Through comprehensive experiments and analyses, we reveal critical insights into the performance limitations of existing 3DGS methods on unconstrained raindrop images and the varying impact of different pipeline components: the impact of camera focus position on 3DGS reconstruction performance, and the interference caused by inaccurate pose and point cloud initialization on reconstruction. These insights establish clear directions for developing more robust 3DGS methods under raindrop conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22489v1">Tracking by Predicting 3-D Gaussians Over Time</a></div>
    <div class="paper-meta">
      📅 2025-12-27
    </div>
    <details class="paper-abstract">
      We propose Video Gaussian Masked Autoencoders (Video-GMAE), a self-supervised approach for representation learning that encodes a sequence of images into a set of Gaussian splats moving over time. Representing a video as a set of Gaussians enforces a reasonable inductive bias: that 2-D videos are often consistent projections of a dynamic 3-D scene. We find that tracking emerges when pretraining a network with this architecture. Mapping the trajectory of the learnt Gaussians onto the image plane gives zero-shot tracking performance comparable to state-of-the-art. With small-scale finetuning, our models achieve 34.6% improvement on Kinetics, and 13.1% on Kubric datasets, surpassing existing self-supervised video approaches. The project page and code are publicly available at https://videogmae.org/ and https://github.com/tekotan/video-gmae.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.12759v3">A-TDOM: Active TDOM via On-the-Fly 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      True Digital Orthophoto Map (TDOM), a 2D objective representation of the Earth's surface, is an essential geospatial product widely used in urban management, city planning, land surveying, and related applications. However, traditional TDOM generation typically relies on a complex offline photogrammetric pipeline, leading to substantial latency and making it unsuitable for time-critical or real-time scenarios. Moreover, the quality of TDOM may deteriorate due to inaccurate camera poses, imperfect Digital Surface Model (DSM), and incorrect occlusions detection. To address these challenges, this work introduces A-TDOM, a near real-time TDOM generation method built upon On-the-Fly 3DGS (3D Gaussian Splatting) optimization. As each incoming image arrives, its pose and sparse point cloud are computed via On-the-Fly SfM. Newly observed regions are then incrementally reconstructed as additional 3D Gaussians are inserted using a Delaunay triangulated Gaussian sampling and integration and are further optimized via adaptive training iterations and learning rate, especially in previously unseen or coarsely modeled areas. With orthogonal splatting integrated into the rendering pipeline, A-TDOM can actively produce updated TDOM outputs immediately after each 3DGS update. Code is now available at https://github.com/xywjohn/A-TDOM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20129v2">Dreamcrafter: Immersive Editing of 3D Radiance Fields Through Flexible, Generative Inputs and Outputs</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 CHI 2025, Project page: https://dream-crafter.github.io/
    </div>
    <details class="paper-abstract">
      Authoring 3D scenes is a central task for spatial computing applications. Competing visions for lowering existing barriers are (1) focus on immersive, direct manipulation of 3D content or (2) leverage AI techniques that capture real scenes (3D Radiance Fields such as, NeRFs, 3D Gaussian Splatting) and modify them at a higher level of abstraction, at the cost of high latency. We unify the complementary strengths of these approaches and investigate how to integrate generative AI advances into real-time, immersive 3D Radiance Field editing. We introduce Dreamcrafter, a VR-based 3D scene editing system that: (1) provides a modular architecture to integrate generative AI algorithms; (2) combines different levels of control for creating objects, including natural language and direct manipulation; and (3) introduces proxy representations that support interaction during high-latency operations. We contribute empirical findings on control preferences and discuss how generative AI interfaces beyond text input enhance creativity in scene editing and world building.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20943v1">AirGS: Real-Time 4D Gaussian Streaming for Free-Viewpoint Video Experiences</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 This paper is accepted by IEEE International Conference on Computer Communications (INFOCOM), 2026
    </div>
    <details class="paper-abstract">
      Free-viewpoint video (FVV) enables immersive viewing experiences by allowing users to view scenes from arbitrary perspectives. As a prominent reconstruction technique for FVV generation, 4D Gaussian Splatting (4DGS) models dynamic scenes with time-varying 3D Gaussian ellipsoids and achieves high-quality rendering via fast rasterization. However, existing 4DGS approaches suffer from quality degradation over long sequences and impose substantial bandwidth and storage overhead, limiting their applicability in real-time and wide-scale deployments. Therefore, we present AirGS, a streaming-optimized 4DGS framework that rearchitects the training and delivery pipeline to enable high-quality, low-latency FVV experiences. AirGS converts Gaussian video streams into multi-channel 2D formats and intelligently identifies keyframes to enhance frame reconstruction quality. It further combines temporal coherence with inflation loss to reduce training time and representation size. To support communication-efficient transmission, AirGS models 4DGS delivery as an integer linear programming problem and design a lightweight pruning level selection algorithm to adaptively prune the Gaussian updates to be transmitted, balancing reconstruction quality and bandwidth consumption. Extensive experiments demonstrate that AirGS reduces quality deviation in PSNR by more than 20% when scene changes, maintains frame-level PSNR consistently above 30, accelerates training by 6 times, reduces per-frame transmission size by nearly 50% compared to the SOTA 4DGS approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20927v1">Quantile Rendering: Efficiently Embedding High-dimensional Feature on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 Will be updated
    </div>
    <details class="paper-abstract">
      Recent advancements in computer vision have successfully extended Open-vocabulary segmentation (OVS) to the 3D domain by leveraging 3D Gaussian Splatting (3D-GS). Despite this progress, efficiently rendering the high-dimensional features required for open-vocabulary queries poses a significant challenge. Existing methods employ codebooks or feature compression, causing information loss, thereby degrading segmentation quality. To address this limitation, we introduce Quantile Rendering (Q-Render), a novel rendering strategy for 3D Gaussians that efficiently handles high-dimensional features while maintaining high fidelity. Unlike conventional volume rendering, which densely samples all 3D Gaussians intersecting each ray, Q-Render sparsely samples only those with dominant influence along the ray. By integrating Q-Render into a generalizable 3D neural network, we also propose Gaussian Splatting Network (GS-Net), which predicts Gaussian features in a generalizable manner. Extensive experiments on ScanNet and LeRF demonstrate that our framework outperforms state-of-the-art methods, while enabling real-time rendering with an approximate ~43.7x speedup on 512-D feature maps. Code will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14537v3">Personalize Your Gaussian: Consistent 3D Scene Personalization from a Single Image</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Personalizing 3D scenes from a single reference image enables intuitive user-guided editing, which requires achieving both multi-view consistency across perspectives and referential consistency with the input image. However, these goals are particularly challenging due to the viewpoint bias caused by the limited perspective provided in a single image. Lacking the mechanisms to effectively expand reference information beyond the original view, existing methods of image-conditioned 3DGS personalization often suffer from this viewpoint bias and struggle to produce consistent results. Therefore, in this paper, we present Consistent Personalization for 3D Gaussian Splatting (CP-GS), a framework that progressively propagates the single-view reference appearance to novel perspectives. In particular, CP-GS integrates pre-trained image-to-3D generation and iterative LoRA fine-tuning to extract and extend the reference appearance, and finally produces faithful multi-view guidance images and the personalized 3DGS outputs through a view-consistent generation process guided by geometric cues. Extensive experiments on real-world scenes show that our CP-GS effectively mitigates the viewpoint bias, achieving high-quality personalization that significantly outperforms existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20495v1">Nebula: Enable City-Scale 3D Gaussian Splatting in Virtual Reality via Collaborative Rendering and Accelerated Stereo Rasterization</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has drawn significant attention in the architectural community recently. However, current architectural designs often overlook the 3DGS scalability, making them fragile for extremely large-scale 3DGS. Meanwhile, the VR bandwidth requirement makes it impossible to deliver high-fidelity and smooth VR content from the cloud. We present Nebula, a coherent acceleration framework for large-scale 3DGS collaborative rendering. Instead of streaming videos, Nebula streams intermediate results after the LoD search, reducing 1925% data communication between the cloud and the client. To further enhance the motion-to-photon experience, we introduce a temporal-aware LoD search in the cloud that tames the irregular memory access and reduces redundant data access by exploiting temporal coherence across frames. On the client side, we propose a novel stereo rasterization that enables two eyes to share most computations during the stereo rendering with bit-accurate quality. With minimal hardware augmentations, Nebula achieves 2.7$\times$ motion-to-photon speedup and reduces 1925% bandwidth over lossy video streaming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20377v1">SmartSplat: Feature-Smart Gaussians for Scalable Compression of Ultra-High-Resolution Images</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Recent advances in generative AI have accelerated the production of ultra-high-resolution visual content, posing significant challenges for efficient compression and real-time decoding on end-user devices. Inspired by 3D Gaussian Splatting, recent 2D Gaussian image models improve representation efficiency, yet existing methods struggle to balance compression ratio and reconstruction fidelity in ultra-high-resolution scenarios. To address this issue, we propose SmartSplat, a highly adaptive and feature-aware GS-based image compression framework that supports arbitrary image resolutions and compression ratios. SmartSplat leverages image-aware features such as gradients and color variances, introducing a Gradient-Color Guided Variational Sampling strategy together with an Exclusion-based Uniform Sampling scheme to improve the non-overlapping coverage of Gaussian primitives in pixel space. In addition, we propose a Scale-Adaptive Gaussian Color Sampling method to enhance color initialization across scales. Through joint optimization of spatial layout, scale, and color initialization, SmartSplat efficiently captures both local structures and global textures using a limited number of Gaussians, achieving high reconstruction quality under strong compression. Extensive experiments on DIV8K and a newly constructed 16K dataset demonstrate that SmartSplat consistently outperforms state-of-the-art methods at comparable compression ratios and exceeds their compression limits, showing strong scalability and practical applicability. The code is publicly available at https://github.com/lif314/SmartSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20148v1">Enhancing annotations for 5D apple pose estimation through 3D Gaussian Splatting (3DGS)</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 33 pages, excluding appendices. 17 figures
    </div>
    <details class="paper-abstract">
      Automating tasks in orchards is challenging because of the large amount of variation in the environment and occlusions. One of the challenges is apple pose estimation, where key points, such as the calyx, are often occluded. Recently developed pose estimation methods no longer rely on these key points, but still require them for annotations, making annotating challenging and time-consuming. Due to the abovementioned occlusions, there can be conflicting and missing annotations of the same fruit between different images. Novel 3D reconstruction methods can be used to simplify annotating and enlarge datasets. We propose a novel pipeline consisting of 3D Gaussian Splatting to reconstruct an orchard scene, simplified annotations, automated projection of the annotations to images, and the training and evaluation of a pose estimation method. Using our pipeline, 105 manual annotations were required to obtain 28,191 training labels, a reduction of 99.6%. Experimental results indicated that training with labels of fruits that are $\leq95\%$ occluded resulted in the best performance, with a neutral F1 score of 0.927 on the original images and 0.970 on the rendered images. Adjusting the size of the training dataset had small effects on the model performance in terms of F1 score and pose estimation accuracy. It was found that the least occluded fruits had the best position estimation, which worsened as the fruits became more occluded. It was also found that the tested pose estimation method was unable to correctly learn the orientation estimation of apples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20129v1">Dreamcrafter: Immersive Editing of 3D Radiance Fields Through Flexible, Generative Inputs and Outputs</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 CHI 2025, Project page: https://dream-crafter.github.io/
    </div>
    <details class="paper-abstract">
      Authoring 3D scenes is a central task for spatial computing applications. Competing visions for lowering existing barriers are (1) focus on immersive, direct manipulation of 3D content or (2) leverage AI techniques that capture real scenes (3D Radiance Fields such as, NeRFs, 3D Gaussian Splatting) and modify them at a higher level of abstraction, at the cost of high latency. We unify the complementary strengths of these approaches and investigate how to integrate generative AI advances into real-time, immersive 3D Radiance Field editing. We introduce Dreamcrafter, a VR-based 3D scene editing system that: (1) provides a modular architecture to integrate generative AI algorithms; (2) combines different levels of control for creating objects, including natural language and direct manipulation; and (3) introduces proxy representations that support interaction during high-latency operations. We contribute empirical findings on control preferences and discuss how generative AI interfaces beyond text input enhance creativity in scene editing and world building.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.19073v2">WaveletGaussian: Wavelet-domain Diffusion for Sparse-view 3D Gaussian Object Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a powerful representation for image-based object reconstruction, yet its performance drops sharply in sparse-view settings. Prior works address this limitation by employing diffusion models to repair corrupted renders, subsequently using them as pseudo ground truths for later optimization. While effective, such approaches incur heavy computation from the diffusion fine-tuning and repair steps. We present WaveletGaussian, a framework for more efficient sparse-view 3D Gaussian object reconstruction. Our key idea is to shift diffusion into the wavelet domain: diffusion is applied only to the low-resolution LL subband, while high-frequency subbands are refined with a lightweight network. We further propose an efficient online random masking strategy to curate training pairs for diffusion fine-tuning, replacing the commonly used, but inefficient, leave-one-out strategy. Experiments across two benchmark datasets, Mip-NeRF 360 and OmniObject3D, show WaveletGaussian achieves competitive rendering quality while substantially reducing training time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.22615v2">GaussianVision: Vision-Language Alignment from Compressed Image Representations using 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Modern vision language pipelines are driven by RGB vision encoders trained on massive image text corpora. While these pipelines have enabled impressive zero-shot capabilities and strong transfer across tasks, they still inherit two structural inefficiencies from the pixel domain: (i) transmitting dense RGB images from edge devices to the cloud is energy-intensive and costly, and (ii) patch-based tokenization explodes sequence length, stressing attention budgets and context limits. We explore 2D Gaussian Splatting (2DGS) as an alternative visual substrate for alignment: a compact, spatially adaptive representation that parameterizes images by a set of colored anisotropic Gaussians. We develop a scalable 2DGS pipeline with structured initialization, luminance-aware pruning, and batched CUDA kernels, achieving over 90x faster fitting and about 97% GPU utilization compared to prior implementations. We further adapt contrastive language-image pre-training (CLIP) to 2DGS by reusing a frozen RGB-based transformer backbone with a lightweight splat-aware input stem and a perceiver resampler, training only 9.7% to 13.8% of the total parameters. On a 12.8M dataset from DataComp, GS encoders yield competitive zero-shot performance on 38 datasets from the CLIP benchmark while compressing inputs 3x to 23.5x relative to pixels. Our results establish 2DGS as a viable multimodal substrate, pinpoint architectural bottlenecks, and open a path toward representations that are both semantically powerful and transmission-efficient for edge-cloud learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.17329v3">SmokeSeer: 3D Gaussian Splatting for Smoke Removal and Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 Project website: https://imaging.cs.cmu.edu/smokeseer
    </div>
    <details class="paper-abstract">
      Smoke in real-world scenes can severely degrade image quality and hamper visibility. Recent image restoration methods either rely on data-driven priors that are susceptible to hallucinations, or are limited to static low-density smoke. We introduce SmokeSeer, a method for simultaneous 3D scene reconstruction and smoke removal from multi-view video sequences. Our method uses thermal and RGB images, leveraging the reduced scattering in thermal images to see through smoke. We build upon 3D Gaussian splatting to fuse information from the two image modalities, and decompose the scene into smoke and non-smoke components. Unlike prior work, SmokeSeer handles a broad range of smoke densities and adapts to temporally varying smoke. We validate our method on synthetic data and a new real-world smoke dataset with RGB and thermal images. We provide an open-source implementation and data on the project website.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.02913v2">Pointmap-Conditioned Diffusion for Consistent Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 WACV 2026. Project page: https://ntaquan0125.github.io/pointmap-conditioned-diffusion
    </div>
    <details class="paper-abstract">
      Synthesizing extrapolated views remains a difficult task, especially in urban driving scenes, where the only reliable sources of data are limited RGB captures and sparse LiDAR points. To address this problem, we present PointmapDiff, a framework for novel view synthesis that utilizes pre-trained 2D diffusion models. Our method leverages point maps (i.e., rasterized 3D scene coordinates) as a conditioning signal, capturing geometric and photometric priors from the reference images to guide the image generation process. With the proposed reference attention layers and ControlNet for point map features, PointmapDiff can generate accurate and consistent results across varying viewpoints while respecting geometric fidelity. Experiments on real-life driving data demonstrate that our method achieves high-quality generation with flexibility over point map conditioning signals (e.g., dense depth map or even sparse LiDAR points) and can be used to distill to 3D representations such as 3D Gaussian Splatting for improving view extrapolation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19678v1">WorldWarp: Propagating 3D Geometry with Asynchronous Video Diffusion</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 Project page: https://hyokong.github.io/worldwarp-page/
    </div>
    <details class="paper-abstract">
      Generating long-range, geometrically consistent video presents a fundamental dilemma: while consistency demands strict adherence to 3D geometry in pixel space, state-of-the-art generative models operate most effectively in a camera-conditioned latent space. This disconnect causes current methods to struggle with occluded areas and complex camera trajectories. To bridge this gap, we propose WorldWarp, a framework that couples a 3D structural anchor with a 2D generative refiner. To establish geometric grounding, WorldWarp maintains an online 3D geometric cache built via Gaussian Splatting (3DGS). By explicitly warping historical content into novel views, this cache acts as a structural scaffold, ensuring each new frame respects prior geometry. However, static warping inevitably leaves holes and artifacts due to occlusions. We address this using a Spatio-Temporal Diffusion (ST-Diff) model designed for a "fill-and-revise" objective. Our key innovation is a spatio-temporal varying noise schedule: blank regions receive full noise to trigger generation, while warped regions receive partial noise to enable refinement. By dynamically updating the 3D cache at every step, WorldWarp maintains consistency across video chunks. Consequently, it achieves state-of-the-art fidelity by ensuring that 3D logic guides structure while diffusion logic perfects texture. Project page: \href{https://hyokong.github.io/worldwarp-page/}{https://hyokong.github.io/worldwarp-page/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19648v1">4D Gaussian Splatting as a Learned Dynamical System</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      We reinterpret 4D Gaussian Splatting as a continuous-time dynamical system, where scene motion arises from integrating a learned neural dynamical field rather than applying per-frame deformations. This formulation, which we call EvoGS, treats the Gaussian representation as an evolving physical system whose state evolves continuously under a learned motion law. This unlocks capabilities absent in deformation-based approaches:(1) sample-efficient learning from sparse temporal supervision by modeling the underlying motion law; (2) temporal extrapolation enabling forward and backward prediction beyond observed time ranges; and (3) compositional dynamics that allow localized dynamics injection for controllable scene synthesis. Experiments on dynamic scene benchmarks show that EvoGS achieves better motion coherence and temporal consistency compared to deformation-field baselines while maintaining real-time rendering
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.14579v2">GSRender: Deduplicated Occupancy Prediction via Weakly Supervised 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Weakly-supervised 3D occupancy perception is crucial for vision-based autonomous driving in outdoor environments. Previous methods based on NeRF often face a challenge in balancing the number of samples used. Too many samples can decrease efficiency, while too few can compromise accuracy, leading to variations in the mean Intersection over Union (mIoU) by 5-10 points. Furthermore, even with surrounding-view image inputs, only a single image is rendered from each viewpoint at any given moment. This limitation leads to duplicated predictions, which significantly impacts the practicality of the approach. However, this issue has largely been overlooked in existing research. To address this, we propose GSRender, which uses 3D Gaussian Splatting for weakly-supervised occupancy estimation, simplifying the sampling process. Additionally, we introduce the Ray Compensation module, which reduces duplicated predictions by compensating for features from adjacent frames. Finally, we redesign the dynamic loss to remove the influence of dynamic objects from adjacent frames. Extensive experiments show that our approach achieves SOTA results in RayIoU (+6.0), while also narrowing the gap with 3D- supervised methods. This work lays a solid foundation for weakly-supervised occupancy perception. The code is available at https://github.com/Jasper-sudo-Sun/GSRender.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.08438v3">A Survey of 3D Reconstruction with Event Cameras</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 This survey has been accepted for publication in the Computational Visual Media Journal
    </div>
    <details class="paper-abstract">
      Event cameras are rapidly emerging as powerful vision sensors for 3D reconstruction, uniquely capable of asynchronously capturing per-pixel brightness changes. Compared to traditional frame-based cameras, event cameras produce sparse yet temporally dense data streams, enabling robust and accurate 3D reconstruction even under challenging conditions such as high-speed motion, low illumination, and extreme dynamic range scenarios. These capabilities offer substantial promise for transformative applications across various fields, including autonomous driving, robotics, aerial navigation, and immersive virtual reality. In this survey, we present the first comprehensive review exclusively dedicated to event-based 3D reconstruction. Existing approaches are systematically categorised based on input modality into stereo, monocular, and multimodal systems, and further classified according to reconstruction methodologies, including geometry-based techniques, deep learning approaches, and neural rendering techniques such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). Within each category, methods are chronologically organised to highlight the evolution of key concepts and advancements. Furthermore, we provide a detailed summary of publicly available datasets specifically suited to event-based reconstruction tasks. Finally, we discuss significant open challenges in dataset availability, standardised evaluation, effective representation, and dynamic scene reconstruction, outlining insightful directions for future research. This survey aims to serve as an essential reference and provides a clear and motivating roadmap toward advancing the state of the art in event-driven 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19108v1">GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 Accepted to AAAI 2026.Code URL:https://github.com/Sweethyh/GaussianImage_plus.git
    </div>
    <details class="paper-abstract">
      Implicit neural representations (INRs) have achieved remarkable success in image representation and compression, but they require substantial training time and memory. Meanwhile, recent 2D Gaussian Splatting (GS) methods (\textit{e.g.}, GaussianImage) offer promising alternatives through efficient primitive-based rendering. However, these methods require excessive Gaussian primitives to maintain high visual fidelity. To exploit the potential of GS-based approaches, we present GaussianImage++, which utilizes limited Gaussian primitives to achieve impressive representation and compression performance. Firstly, we introduce a distortion-driven densification mechanism. It progressively allocates Gaussian primitives according to signal intensity. Secondly, we employ context-aware Gaussian filters for each primitive, which assist in the densification to optimize Gaussian primitives based on varying image content. Thirdly, we integrate attribute-separated learnable scalar quantizers and quantization-aware training, enabling efficient compression of primitive attributes. Experimental results demonstrate the effectiveness of our method. In particular, GaussianImage++ outperforms GaussianImage and INRs-based COIN in representation and compression performance while maintaining real-time decoding and low memory usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17817v2">Chorus: Multi-Teacher Pretraining for Holistic 3D Gaussian Scene Encoding</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      While 3DGS has emerged as a high-fidelity scene representation, encoding rich, general-purpose features directly from its primitives remains under-explored. We address this gap by introducing Chorus, a multi-teacher pretraining framework that learns a holistic feed-forward 3D Gaussian Splatting (3DGS) scene encoder by distilling complementary signals from 2D foundation models. Chorus employs a shared 3D encoder and teacher-specific projectors to learn from language-aligned, generalist, and object-aware teachers, encouraging a shared embedding space that captures signals from high-level semantics to fine-grained structure. We evaluate Chorus on a wide range of tasks: open-vocabulary semantic and instance segmentation, linear and decoder probing, as well as data-efficient supervision. Besides 3DGS, we also test Chorus on several benchmarks that only support point clouds by pretraining a variant using only Gaussians' centers, colors, estimated normals as inputs. Interestingly, this encoder shows strong transfer and outperforms the point clouds baseline while using 39.9 times fewer training scenes. Finally, we propose a render-and-distill adaptation that facilitates out-of-domain finetuning. Our code and model will be released upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.14736v2">HandSCS: Structural Coordinate Space for Animatable Hand Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-21
    </div>
    <details class="paper-abstract">
      Creating animatable hand avatars from multi-view images requires modeling complex articulations and maintaining structural consistency across poses in real time. We present HandSCS, a structure-guided 3D Gaussian Splatting framework for high-fidelity hand animation. Unlike existing approaches that condition all Gaussians on the same global pose parameters, which are inadequate for highly articulated hands, HandSCS equips each Gaussian with explicit structural guidance from both intra-pose and inter-pose perspectives. To establish intra-pose structural guidance, we introduce a Structural Coordinate Space (SCS), which bridges the gap between sparse bones and dense Gaussians through hybrid static-dynamic coordinate basis and angular-radial descriptors. To improve cross-pose coherence, we further introduce an Inter-pose Consistency Loss that promotes consistent Gaussian attributes under similar articulations. Together, these components achieve high-fidelity results with consistent fine details, even in challenging high-deformation and self-contact regions. Experiments on the InterHand2.6M dataset demonstrate that HandSCS achieves state-of-the-art performance in hand avatar animation, confirming the effectiveness of explicit structural modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.14501v5">Advances in Feed-Forward 3D Reconstruction and View Synthesis: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 A project page associated with this survey is available at https://fnzhan.com/projects/Feed-Forward-3D
    </div>
    <details class="paper-abstract">
      3D reconstruction and view synthesis are foundational problems in computer vision, graphics, and immersive technologies such as augmented reality (AR), virtual reality (VR), and digital twins. Traditional methods rely on computationally intensive iterative optimization in a complex chain, limiting their applicability in real-world scenarios. Recent advances in feed-forward approaches, driven by deep learning, have revolutionized this field by enabling fast and generalizable 3D reconstruction and view synthesis. This survey offers a comprehensive review of feed-forward techniques for 3D reconstruction and view synthesis, with a taxonomy according to the underlying representation architectures including point cloud, 3D Gaussian Splatting (3DGS), Neural Radiance Fields (NeRF), etc. We examine key tasks such as pose-free reconstruction, dynamic 3D reconstruction, and 3D-aware image and video synthesis, highlighting their applications in digital humans, SLAM, robotics, and beyond. In addition, we review commonly used datasets with detailed statistics, along with evaluation protocols for various downstream tasks. We conclude by discussing open research challenges and promising directions for future work, emphasizing the potential of feed-forward approaches to advance the state of the art in 3D vision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18692v1">EcoSplat: Efficiency-controllable Feed-forward 3D Gaussian Splatting from Multi-view Images</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 The first two authors contributed equally to this work (equal contribution). The last two authors advised equally to this work. Please visit our project page at https://kaist-viclab.github.io/ecosplat-site/
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) enables efficient one-pass scene reconstruction, providing 3D representations for novel view synthesis without per-scene optimization. However, existing methods typically predict pixel-aligned primitives per-view, producing an excessive number of primitives in dense-view settings and offering no explicit control over the number of predicted Gaussians. To address this, we propose EcoSplat, the first efficiency-controllable feed-forward 3DGS framework that adaptively predicts the 3D representation for any given target primitive count at inference time. EcoSplat adopts a two-stage optimization process. The first stage is Pixel-aligned Gaussian Training (PGT) where our model learns initial primitive prediction. The second stage is Importance-aware Gaussian Finetuning (IGF) stage where our model learns rank primitives and adaptively adjust their parameters based on the target primitive count. Extensive experiments across multiple dense-view settings show that EcoSplat is robust and outperforms state-of-the-art methods under strict primitive-count constraints, making it well-suited for flexible downstream rendering tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18640v1">Geometric-Photometric Event-based 3D Gaussian Ray Tracing</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 15 pages, 10 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Event cameras offer a high temporal resolution over traditional frame-based cameras, which makes them suitable for motion and structure estimation. However, it has been unclear how event-based 3D Gaussian Splatting (3DGS) approaches could leverage fine-grained temporal information of sparse events. This work proposes a framework to address the trade-off between accuracy and temporal resolution in event-based 3DGS. Our key idea is to decouple the rendering into two branches: event-by-event geometry (depth) rendering and snapshot-based radiance (intensity) rendering, by using ray-tracing and the image of warped events. The extensive evaluation shows that our method achieves state-of-the-art performance on the real-world datasets and competitive performance on the synthetic dataset. Also, the proposed method works without prior information (e.g., pretrained image reconstruction models) or COLMAP-based initialization, is more flexible in the event selection number, and achieves sharp reconstruction on scene edges with fast training time. We hope that this work deepens our understanding of the sparse nature of events for 3D reconstruction. The code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18619v1">ChronoDreamer: Action-Conditioned World Model as an Online Simulator for Robotic Planning</a></div>
    <div class="paper-meta">
      📅 2025-12-21
    </div>
    <details class="paper-abstract">
      We present ChronoDreamer, an action-conditioned world model for contact-rich robotic manipulation. Given a history of egocentric RGB frames, contact maps, actions, and joint states, ChronoDreamer predicts future video frames, contact distributions, and joint angles via a spatial-temporal transformer trained with MaskGIT-style masked prediction. Contact is encoded as depth-weighted Gaussian splat images that render 3D forces into a camera-aligned format suitable for vision backbones. At inference, predicted rollouts are evaluated by a vision-language model that reasons about collision likelihood, enabling rejection sampling of unsafe actions before execution. We train and evaluate on DreamerBench, a simulation dataset generated with Project Chrono that provides synchronized RGB, contact splat, proprioception, and physics annotations across rigid and deformable object scenarios. Qualitative results demonstrate that the model preserves spatial coherence during non-contact motion and generates plausible contact predictions, while the LLM-based judge distinguishes collision from non-collision trajectories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.04262v2">Bridging Geometry-Coherent Text-to-3D Generation with Multi-View Diffusion Priors and Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-20
      | 💬 Accepted by Neural Networks. The final published version is available at https://doi.org/10.1016/j.neunet.2025.108511
    </div>
    <details class="paper-abstract">
      Score Distillation Sampling (SDS) leverages pretrained 2D diffusion models to advance text-to-3D generation but neglects multi-view correlations, being prone to geometric inconsistencies and multi-face artifacts in the generated 3D content. In this work, we propose Coupled Score Distillation (CSD), a framework that couples multi-view joint distribution priors to ensure geometrically consistent 3D generation while enabling the stable and direct optimization of 3D Gaussian Splatting. Specifically, by reformulating the optimization as a multi-view joint optimization problem, we derive an effective optimization rule that effectively couples multi-view priors to guide optimization across different viewpoints while preserving the diversity of generated 3D assets. Additionally, we propose a framework that directly optimizes 3D Gaussian Splatting (3D-GS) with random initialization to generate geometrically consistent 3D content. We further employ a deformable tetrahedral grid, initialized from 3D-GS and refined through CSD, to produce high-quality, refined meshes. Quantitative and qualitative experimental results demonstrate the efficiency and competitive quality of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18314v1">MatSpray: Fusing 2D Material World Knowledge on 3D Geometry</a></div>
    <div class="paper-meta">
      📅 2025-12-20
      | 💬 Project page: https://matspray.jdihlmann.com/
    </div>
    <details class="paper-abstract">
      Manual modeling of material parameters and 3D geometry is a time consuming yet essential task in the gaming and film industries. While recent advances in 3D reconstruction have enabled accurate approximations of scene geometry and appearance, these methods often fall short in relighting scenarios due to the lack of precise, spatially varying material parameters. At the same time, diffusion models operating on 2D images have shown strong performance in predicting physically based rendering (PBR) properties such as albedo, roughness, and metallicity. However, transferring these 2D material maps onto reconstructed 3D geometry remains a significant challenge. We propose a framework for fusing 2D material data into 3D geometry using a combination of novel learning-based and projection-based approaches. We begin by reconstructing scene geometry via Gaussian Splatting. From the input images, a diffusion model generates 2D maps for albedo, roughness, and metallic parameters. Any existing diffusion model that can convert images or videos to PBR materials can be applied. The predictions are further integrated into the 3D representation either by optimizing an image-based loss or by directly projecting the material parameters onto the Gaussians using Gaussian ray tracing. To enhance fine-scale accuracy and multi-view consistency, we further introduce a light-weight neural refinement step (Neural Merger), which takes ray-traced material features as input and produces detailed adjustments. Our results demonstrate that the proposed methods outperform existing techniques in both quantitative metrics and perceived visual realism. This enables more accurate, relightable, and photorealistic renderings from reconstructed scenes, significantly improving the realism and efficiency of asset creation workflows in content production pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.01171v2">No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views</a></div>
    <div class="paper-meta">
      📅 2025-12-20
      | 💬 Project Page: https://ranrhuang.github.io/spfsplat/
    </div>
    <details class="paper-abstract">
      We introduce SPFSplat, an efficient framework for 3D Gaussian splatting from sparse multi-view images, requiring no ground-truth poses during training or inference. It employs a shared feature extraction backbone, enabling simultaneous prediction of 3D Gaussian primitives and camera poses in a canonical space from unposed inputs within a single feed-forward step. Alongside the rendering loss based on estimated novel-view poses, a reprojection loss is integrated to enforce the learning of pixel-aligned Gaussian primitives for enhanced geometric constraints. This pose-free training paradigm and efficient one-step feed-forward design make SPFSplat well-suited for practical applications. Remarkably, despite the absence of pose supervision, SPFSplat achieves state-of-the-art performance in novel view synthesis even under significant viewpoint changes and limited image overlap. It also surpasses recent methods trained with geometry priors in relative pose estimation. Code and trained models are available on our project page: https://ranrhuang.github.io/spfsplat/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17817v1">Chorus: Multi-Teacher Pretraining for Holistic 3D Gaussian Scene Encoding</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      While 3DGS has emerged as a high-fidelity scene representation, encoding rich, general-purpose features directly from its primitives remains under-explored. We address this gap by introducing Chorus, a multi-teacher pretraining framework that learns a holistic feed-forward 3D Gaussian Splatting (3DGS) scene encoder by distilling complementary signals from 2D foundation models. Chorus employs a shared 3D encoder and teacher-specific projectors to learn from language-aligned, generalist, and object-aware teachers, encouraging a shared embedding space that captures signals from high-level semantics to fine-grained structure. We evaluate Chorus on a wide range of tasks: open-vocabulary semantic and instance segmentation, linear and decoder probing, as well as data-efficient supervision. Besides 3DGS, we also test Chorus on several benchmarks that only support point clouds by pretraining a variant using only Gaussians' centers, colors, estimated normals as inputs. Interestingly, this encoder shows strong transfer and outperforms the point clouds baseline while using 39.9 times fewer training scenes. Finally, we propose a render-and-distill adaptation that facilitates out-of-domain finetuning. Our code and model will be released upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17547v1">G3Splat: Geometrically Consistent Generalizable Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Project page: https://m80hz.github.io/g3splat/
    </div>
    <details class="paper-abstract">
      3D Gaussians have recently emerged as an effective scene representation for real-time splatting and accurate novel-view synthesis, motivating several works to adapt multi-view structure prediction networks to regress per-pixel 3D Gaussians from images. However, most prior work extends these networks to predict additional Gaussian parameters -- orientation, scale, opacity, and appearance -- while relying almost exclusively on view-synthesis supervision. We show that a view-synthesis loss alone is insufficient to recover geometrically meaningful splats in this setting. We analyze and address the ambiguities of learning 3D Gaussian splats under self-supervision for pose-free generalizable splatting, and introduce G3Splat, which enforces geometric priors to obtain geometrically consistent 3D scene representations. Trained on RE10K, our approach achieves state-of-the-art performance in (i) geometrically consistent reconstruction, (ii) relative pose estimation, and (iii) novel-view synthesis. We further demonstrate strong zero-shot generalization on ScanNet, substantially outperforming prior work in both geometry recovery and relative pose estimation. Code and pretrained models are released on our project page (https://m80hz.github.io/g3splat/).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17541v1">FLEG: Feed-Forward Language Embedded Gaussian Splatting from Any Views</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Project page: https://fangzhou2000.github.io/projects/fleg
    </div>
    <details class="paper-abstract">
      We present FLEG, a feed-forward network that reconstructs language-embedded 3D Gaussians from any views. Previous straightforward solutions combine feed-forward reconstruction with Gaussian heads but suffer from fixed input views and insufficient 3D training data. In contrast, we propose a 3D-annotation-free training framework for 2D-to-3D lifting from arbitrary uncalibrated and unposed multi-view images. Since the framework does not require 3D annotations, we can leverage large-scale video data with easily obtained 2D instance information to enrich semantic embedding. We also propose an instance-guided contrastive learning to align 2D semantics with the 3D representations. In addition, to mitigate the high memory and computational cost of dense views, we further propose a geometry-semantic hierarchical sparsification strategy. Our FLEG efficiently reconstructs language-embedded 3D Gaussian representation in a feed-forward manner from arbitrary sparse or dense views, jointly producing accurate geometry, high-fidelity appearance, and language-aligned semantics. Extensive experiments show that it outperforms existing methods on various related tasks. Project page: https://fangzhou2000.github.io/projects/fleg.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17528v1">Voxel-GS: Quantized Scaffold Gaussian Splatting Compression with Run-Length Coding</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Accepted by DCC 2026
    </div>
    <details class="paper-abstract">
      Substantial Gaussian splatting format point clouds require effective compression. In this paper, we propose Voxel-GS, a simple yet highly effective framework that departs from the complex neural entropy models of prior work, instead achieving competitive performance using only a lightweight rate proxy and run-length coding. Specifically, we employ a differentiable quantization to discretize the Gaussian attributes of Scaffold-GS. Subsequently, a Laplacian-based rate proxy is devised to impose an entropy constraint, guiding the generation of high-fidelity and compact reconstructions. Finally, this integer-type Gaussian point cloud is compressed losslessly using Octree and run-length coding. Experiments validate that the proposed rate proxy accurately estimates the bitrate of run-length coding, enabling Voxel-GS to eliminate redundancy and optimize for a more compact representation. Consequently, our method achieves a remarkable compression ratio with significantly faster coding speeds than prior art. The code is available at https://github.com/zb12138/VoxelGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15258v2">VLA-AN: An Efficient and Onboard Vision-Language-Action Framework for Aerial Navigation in Complex Environments</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      This paper proposes VLA-AN, an efficient and onboard Vision-Language-Action (VLA) framework dedicated to autonomous drone navigation in complex environments. VLA-AN addresses four major limitations of existing large aerial navigation models: the data domain gap, insufficient temporal navigation with reasoning, safety issues with generative action policies, and onboard deployment constraints. First, we construct a high-fidelity dataset utilizing 3D Gaussian Splatting (3D-GS) to effectively bridge the domain gap. Second, we introduce a progressive three-stage training framework that sequentially reinforces scene comprehension, core flight skills, and complex navigation capabilities. Third, we design a lightweight, real-time action module coupled with geometric safety correction. This module ensures fast, collision-free, and stable command generation, mitigating the safety risks inherent in stochastic generative policies. Finally, through deep optimization of the onboard deployment pipeline, VLA-AN achieves a robust real-time 8.3x improvement in inference throughput on resource-constrained UAVs. Extensive experiments demonstrate that VLA-AN significantly improves spatial grounding, scene reasoning, and long-horizon navigation, achieving a maximum single-task success rate of 98.1%, and providing an efficient, practical solution for realizing full-chain closed-loop autonomy in lightweight aerial robots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17349v1">Flying in Clutter on Monocular RGB by Learning in 3D Radiance Fields with Domain Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 8 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Modern autonomous navigation systems predominantly rely on lidar and depth cameras. However, a fundamental question remains: Can flying robots navigate in clutter using solely monocular RGB images? Given the prohibitive costs of real-world data collection, learning policies in simulation offers a promising path. Yet, deploying such policies directly in the physical world is hindered by the significant sim-to-real perception gap. Thus, we propose a framework that couples the photorealism of 3D Gaussian Splatting (3DGS) environments with Adversarial Domain Adaptation. By training in high-fidelity simulation while explicitly minimizing feature discrepancy, our method ensures the policy relies on domain-invariant cues. Experimental results demonstrate that our policy achieves robust zero-shot transfer to the physical world, enabling safe and agile flight in unstructured environments with varying illumination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.13911v2">PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Despite advances in physics-based 3D motion synthesis, current methods face key limitations: reliance on pre-reconstructed 3D Gaussian Splatting (3DGS) built from dense multi-view images with time-consuming per-scene optimization; physics integration via either inflexible, hand-specified attributes or unstable, optimization-heavy guidance from video models using Score Distillation Sampling (SDS); and naive concatenation of prebuilt 3DGS with physics modules, which ignores physical information embedded in appearance and yields suboptimal performance. To address these issues, we propose PhysGM, a feed-forward framework that jointly predicts 3D Gaussian representation and physical properties from a single image, enabling immediate simulation and high-fidelity 4D rendering. Unlike slow appearance-agnostic optimization methods, we first pre-train a physics-aware reconstruction model that directly infers both Gaussian and physical parameters. We further refine the model with Direct Preference Optimization (DPO), aligning simulations with the physically plausible reference videos and avoiding the high-cost SDS optimization. To address the absence of a supporting dataset for this task, we propose PhysAssets, a dataset of 50K+ 3D assets annotated with physical properties and corresponding reference videos. Experiments show that PhysGM produces high-fidelity 4D simulations from a single image in one minute, achieving a significant speedup over prior work while delivering realistic renderings.Our project page is at:https://hihixiaolv.github.io/PhysGM.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.15355v3">UniGaussian: Driving Scene Reconstruction from Multiple Camera Models via Unified Gaussian Representations</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 3DV 2026
    </div>
    <details class="paper-abstract">
      Urban scene reconstruction is crucial for real-world autonomous driving simulators. Although existing methods have achieved photorealistic reconstruction, they mostly focus on pinhole cameras and neglect fisheye cameras. In fact, how to effectively simulate fisheye cameras in driving scene remains an unsolved problem. In this work, we propose UniGaussian, a novel approach that learns a unified 3D Gaussian representation from multiple camera models for urban scene reconstruction in autonomous driving. Our contributions are two-fold. First, we propose a new differentiable rendering method that distorts 3D Gaussians using a series of affine transformations tailored to fisheye camera models. This addresses the compatibility issue of 3D Gaussian splatting with fisheye cameras, which is hindered by light ray distortion caused by lenses or mirrors. Besides, our method maintains real-time rendering while ensuring differentiability. Second, built on the differentiable rendering method, we design a new framework that learns a unified Gaussian representation from multiple camera models. By applying affine transformations to adapt different camera models and regularizing the shared Gaussians with supervision from different modalities, our framework learns a unified 3D Gaussian representation with input data from multiple sources and achieves holistic driving scene understanding. As a result, our approach models multiple sensors (pinhole and fisheye cameras) and modalities (depth, semantic, normal and LiDAR point clouds). Our experiments show that our method achieves superior rendering quality and fast rendering speed for driving scene simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16893v1">Instant Expressive Gaussian Head Avatar via 3D-Aware Expression Distillation</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 Project website is https://research.nvidia.com/labs/amri/projects/instant4d
    </div>
    <details class="paper-abstract">
      Portrait animation has witnessed tremendous quality improvements thanks to recent advances in video diffusion models. However, these 2D methods often compromise 3D consistency and speed, limiting their applicability in real-world scenarios, such as digital twins or telepresence. In contrast, 3D-aware facial animation feedforward methods -- built upon explicit 3D representations, such as neural radiance fields or Gaussian splatting -- ensure 3D consistency and achieve faster inference speed, but come with inferior expression details. In this paper, we aim to combine their strengths by distilling knowledge from a 2D diffusion-based method into a feed-forward encoder, which instantly converts an in-the-wild single image into a 3D-consistent, fast yet expressive animatable representation. Our animation representation is decoupled from the face's 3D representation and learns motion implicitly from data, eliminating the dependency on pre-defined parametric models that often constrain animation capabilities. Unlike previous computationally intensive global fusion mechanisms (e.g., multiple attention layers) for fusing 3D structural and animation information, our design employs an efficient lightweight local fusion strategy to achieve high animation expressivity. As a result, our method runs at 107.31 FPS for animation and pose control while achieving comparable animation quality to the state-of-the-art, surpassing alternative designs that trade speed for quality or vice versa. Project website is https://research.nvidia.com/labs/amri/projects/instant4d
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18600v2">NeAR: Coupled Neural Asset-Renderer Stack</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 20 pages, 19 figures. The project page: https://near-project.github.io/
    </div>
    <details class="paper-abstract">
      Neural asset authoring and neural rendering have traditionally evolved as disjoint paradigms: one generates digital assets for fixed graphics pipelines, while the other maps conventional assets to images. However, treating them as independent entities limits the potential for end-to-end optimization in fidelity and consistency. In this paper, we bridge this gap with NeAR, a Coupled Neural Asset--Renderer Stack. We argue that co-designing the asset representation and the renderer creates a robust "contract" for superior generation. On the asset side, we introduce the Lighting-Homogenized SLAT (LH-SLAT). Leveraging a rectified-flow model, NeAR lifts casually lit single images into a canonical, illumination-invariant latent space, effectively suppressing baked-in shadows and highlights. On the renderer side, we design a lighting-aware neural decoder tailored to interpret these homogenized latents. Conditioned on HDR environment maps and camera views, it synthesizes relightable 3D Gaussian splats in real-time without per-object optimization. We validate NeAR on four tasks: (1) G-buffer-based forward rendering, (2) random-lit reconstruction, (3) unknown-lit relighting, and (4) novel-view relighting. Extensive experiments demonstrate that our coupled stack outperforms state-of-the-art baselines in both quantitative metrics and perceptual quality. We hope this coupled asset-renderer perspective inspires future graphics stacks that view neural assets and renderers as co-designed components instead of independent entities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16706v1">SDFoam: Signed-Distance Foam for explicit surface reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Neural radiance fields (NeRF) have driven impressive progress in view synthesis by using ray-traced volumetric rendering. Splatting-based methods such as 3D Gaussian Splatting (3DGS) provide faster rendering by rasterizing 3D primitives. RadiantFoam (RF) brought ray tracing back, achieving throughput comparable to Gaussian Splatting by organizing radiance with an explicit Voronoi Diagram (VD). Yet, all the mentioned methods still struggle with precise mesh reconstruction. We address this gap by jointly learning an explicit VD with an implicit Signed Distance Field (SDF). The scene is optimized via ray tracing and regularized by an Eikonal objective. The SDF introduces metric-consistent isosurfaces, which, in turn, bias near-surface Voronoi cell faces to align with the zero level set. The resulting model produces crisper, view-consistent surfaces with fewer floaters and improved topology, while preserving photometric quality and maintaining training speed on par with RadiantFoam. Across diverse scenes, our hybrid implicit-explicit formulation, which we name SDFoam, substantially improves mesh reconstruction accuracy (Chamfer distance) with comparable appearance (PSNR, SSIM), without sacrificing efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16397v1">Using Gaussian Splats to Create High-Fidelity Facial Geometry and Texture</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 Submitted to CVPR 2026. 21 pages, 22 figures
    </div>
    <details class="paper-abstract">
      We leverage increasingly popular three-dimensional neural representations in order to construct a unified and consistent explanation of a collection of uncalibrated images of the human face. Our approach utilizes Gaussian Splatting, since it is more explicit and thus more amenable to constraints than NeRFs. We leverage segmentation annotations to align the semantic regions of the face, facilitating the reconstruction of a neutral pose from only 11 images (as opposed to requiring a long video). We soft constrain the Gaussians to an underlying triangulated surface in order to provide a more structured Gaussian Splat reconstruction, which in turn informs subsequent perturbations to increase the accuracy of the underlying triangulated surface. The resulting triangulated surface can then be used in a standard graphics pipeline. In addition, and perhaps most impactful, we show how accurate geometry enables the Gaussian Splats to be transformed into texture space where they can be treated as a view-dependent neural texture. This allows one to use high visual fidelity Gaussian Splatting on any asset in a scene without the need to modify any other asset or any other aspect (geometry, lighting, renderer, etc.) of the graphics pipeline. We utilize a relightable Gaussian model to disentangle texture from lighting in order to obtain a delit high-resolution albedo texture that is also readily usable in a standard graphics pipeline. The flexibility of our system allows for training with disparate images, even with incompatible lighting, facilitating robust regularization. Finally, we demonstrate the efficacy of our approach by illustrating its use in a text-driven asset creation pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.05859v3">D-FCGS: Feedforward Compression of Dynamic Gaussian Splatting for Free-Viewpoint Videos</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 changes of some major results
    </div>
    <details class="paper-abstract">
      Free-Viewpoint Video (FVV) enables immersive 3D experiences, but efficient compression of dynamic 3D representation remains a major challenge. Existing dynamic 3D Gaussian Splatting methods couple reconstruction with optimization-dependent compression and customized motion formats, limiting generalization and standardization. To address this, we propose D-FCGS, a novel Feedforward Compression framework for Dynamic Gaussian Splatting. Key innovations include: (1) a standardized Group-of-Frames (GoF) structure with I-P coding, leveraging sparse control points to extract inter-frame motion tensors; (2) a dual prior-aware entropy model that fuses hyperprior and spatial-temporal priors for accurate rate estimation; (3) a control-point-guided motion compensation mechanism and refinement network to enhance view-consistent fidelity. Trained on Gaussian frames derived from multi-view videos, D-FCGS generalizes across diverse scenes in a zero-shot fashion. Experiments show that it matches the rate-distortion performance of optimization-based methods, achieving over 40 times compression compared to the baseline while preserving visual quality across viewpoints. This work advances feedforward compression of dynamic 3DGS, facilitating scalable FVV transmission and storage for immersive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15711v1">Gaussian Pixel Codec Avatars: A Hybrid Representation for Efficient Rendering</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Tech report
    </div>
    <details class="paper-abstract">
      We present Gaussian Pixel Codec Avatars (GPiCA), photorealistic head avatars that can be generated from multi-view images and efficiently rendered on mobile devices. GPiCA utilizes a unique hybrid representation that combines a triangle mesh and anisotropic 3D Gaussians. This combination maximizes memory and rendering efficiency while maintaining a photorealistic appearance. The triangle mesh is highly efficient in representing surface areas like facial skin, while the 3D Gaussians effectively handle non-surface areas such as hair and beard. To this end, we develop a unified differentiable rendering pipeline that treats the mesh as a semi-transparent layer within the volumetric rendering paradigm of 3D Gaussian Splatting. We train neural networks to decode a facial expression code into three components: a 3D face mesh, an RGBA texture, and a set of 3D Gaussians. These components are rendered simultaneously in a unified rendering engine. The networks are trained using multi-view image supervision. Our results demonstrate that GPiCA achieves the realism of purely Gaussian-based avatars while matching the rendering performance of mesh-based avatars.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15508v1">Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) models enable real-time scene generation but are hindered by suboptimal pixel-aligned primitive placement, which relies on a dense, rigid grid and limits both quality and efficiency. We introduce a new feed-forward architecture that detects 3D Gaussian primitives at a sub-pixel level, replacing the pixel grid with an adaptive, "Off The Grid" distribution. Inspired by keypoint detection, our multi-resolution decoder learns to distribute primitives across image patches. This module is trained end-to-end with a 3D reconstruction backbone using self-supervised learning. Our resulting pose-free model generates photorealistic scenes in seconds, achieving state-of-the-art novel view synthesis for feed-forward models. It outperforms competitors while using far fewer primitives, demonstrating a more accurate and efficient allocation that captures fine details and reduces artifacts. Moreover, we observe that by learning to render 3D Gaussians, our 3D reconstruction backbone improves camera pose estimation, suggesting opportunities to train these foundational models without labels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15258v1">VLA-AN: An Efficient and Onboard Vision-Language-Action Framework for Aerial Navigation in Complex Environments</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      This paper proposes VLA-AN, an efficient and onboard Vision-Language-Action (VLA) framework dedicated to autonomous drone navigation in complex environments. VLA-AN addresses four major limitations of existing large aerial navigation models: the data domain gap, insufficient temporal navigation with reasoning, safety issues with generative action policies, and onboard deployment constraints. First, we construct a high-fidelity dataset utilizing 3D Gaussian Splatting (3D-GS) to effectively bridge the domain gap. Second, we introduce a progressive three-stage training framework that sequentially reinforces scene comprehension, core flight skills, and complex navigation capabilities. Third, we design a lightweight, real-time action module coupled with geometric safety correction. This module ensures fast, collision-free, and stable command generation, mitigating the safety risks inherent in stochastic generative policies. Finally, through deep optimization of the onboard deployment pipeline, VLA-AN achieves a robust real-time 8.3x improvement in inference throughput on resource-constrained UAVs. Extensive experiments demonstrate that VLA-AN significantly improves spatial grounding, scene reasoning, and long-horizon navigation, achieving a maximum single-task success rate of 98.1%, and providing an efficient, practical solution for realizing full-chain closed-loop autonomy in lightweight aerial robots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.15208v3">GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Transferring 2D textures onto complex 3D scenes plays a vital role in enhancing the efficiency and controllability of 3D multimedia content creation. However, existing 3D style transfer methods primarily focus on transferring abstract artistic styles to 3D scenes. These methods often overlook the geometric information of the scene, which makes it challenging to achieve high-quality 3D texture transfer results. In this paper, we present GT2-GS, a geometry-aware texture transfer framework for gaussian splatting. First, we propose a geometry-aware texture transfer loss that enables view-consistent texture transfer by leveraging prior view-dependent feature information and texture features augmented with additional geometric parameters. Moreover, an adaptive fine-grained control module is proposed to address the degradation of scene information caused by low-granularity texture features. Finally, a geometry preservation branch is introduced. This branch refines the geometric parameters using additionally bound Gaussian color priors, thereby decoupling the optimization objectives of appearance and geometry. Extensive experiments demonstrate the effectiveness and controllability of our method. Through geometric awareness, our approach achieves texture transfer results that better align with human visual perception. Our homepage is available at https://vpx-ecnu.github.io/GT2-GS-website.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15048v1">MVGSR: Multi-View Consistent 3D Gaussian Super-Resolution via Epipolar Guidance</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 9 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Scenes reconstructed by 3D Gaussian Splatting (3DGS) trained on low-resolution (LR) images are unsuitable for high-resolution (HR) rendering. Consequently, a 3DGS super-resolution (SR) method is needed to bridge LR inputs and HR rendering. Early 3DGS SR methods rely on single-image SR networks, which lack cross-view consistency and fail to fuse complementary information across views. More recent video-based SR approaches attempt to address this limitation but require strictly sequential frames, limiting their applicability to unstructured multi-view datasets. In this work, we introduce Multi-View Consistent 3D Gaussian Splatting Super-Resolution (MVGSR), a framework that focuses on integrating multi-view information for 3DGS rendering with high-frequency details and enhanced consistency. We first propose an Auxiliary View Selection Method based on camera poses, making our method adaptable for arbitrarily organized multi-view datasets without the need of temporal continuity or data reordering. Furthermore, we introduce, for the first time, an epipolar-constrained multi-view attention mechanism into 3DGS SR, which serves as the core of our proposed multi-view SR network. This design enables the model to selectively aggregate consistent information from auxiliary views, enhancing the geometric consistency and detail fidelity of 3DGS representations. Extensive experiments demonstrate that our method achieves state-of-the-art performance on both object-centric and scene-level 3DGS SR benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.09342v2">GASPACHO: Gaussian Splatting for Controllable Humans and Objects</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Project Page: https://miraymen.github.io/gaspacho/
    </div>
    <details class="paper-abstract">
      We present GASPACHO, a method for generating photorealistic, controllable renderings of human-object interactions from multi-view RGB video. Unlike prior work that reconstructs only the human and treats objects as background, GASPACHO simultaneously recovers animatable templates for both the human and the interacting object as distinct sets of Gaussians, thereby allowing for controllable renderings of novel human object interactions in different poses from novel-camera viewpoints. We introduce a novel formulation that learns object Gaussians on an underlying 2D surface manifold rather than in 3D volume, yielding sharper, fine-grained object details for dynamic object reconstruction. We further propose a contact constraint in Gaussian space that regularizes human-object relations and enables natural, physically plausible animation. Across three benchmarks - BEHAVE, NeuralDome, and DNA-Rendering - GASPACHO achieves high-quality reconstructions under heavy occlusion and supports controllable synthesis of novel human-object interactions. We also demonstrate that our method allows for composition of humans and objects in 3D scenes and for the first time showcase that neural rendering can be used for the controllable generation of photoreal humans interacting with dynamic objects in diverse scenes. Our results are available at: https://miraymen.github.io/gaspacho/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07345v2">Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Accepted by AAAI 2026, Code is available at: https://github.com/kimslong/AAAI26-TDAttn
    </div>
    <details class="paper-abstract">
      Versatile 3D tasks (e.g., generation or editing) that distill from Text-to-Image (T2I) diffusion models have attracted significant research interest for not relying on extensive 3D training data. However, T2I models exhibit limitations resulting from prior view bias, which produces conflicting appearances between different views of an object. This bias causes subject-words to preferentially activate prior view features during cross-attention (CA) computation, regardless of the target view condition. To overcome this limitation, we conduct a comprehensive mathematical analysis to reveal the root cause of the prior view bias in T2I models. Moreover, we find different UNet layers show different effects of prior view in CA. Therefore, we propose a novel framework, TD-Attn, which addresses multi-view inconsistency via two key components: (1) the 3D-Aware Attention Guidance Module (3D-AAG) constructs a view-consistent 3D attention Gaussian for subject-words to enforce spatial consistency across attention-focused regions, thereby compensating for the limited spatial information in 2D individual view CA maps; (2) the Hierarchical Attention Modulation Module (HAM) utilizes a Semantic Guidance Tree (SGT) to direct the Semantic Response Profiler (SRP) in localizing and modulating CA layers that are highly responsive to view conditions, where the enhanced CA maps further support the construction of more consistent 3D attention Gaussians. Notably, HAM facilitates semantic-specific interventions, enabling controllable and precise 3D editing. Extensive experiments firmly establish that TD-Attn has the potential to serve as a universal plugin, significantly enhancing multi-view consistency across 3D tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14406v1">Broadening View Synthesis of Dynamic Scenes from Constrained Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      In dynamic Neural Radiance Fields (NeRF) systems, state-of-the-art novel view synthesis methods often fail under significant viewpoint deviations, producing unstable and unrealistic renderings. To address this, we introduce Expanded Dynamic NeRF (ExpanDyNeRF), a monocular NeRF framework that leverages Gaussian splatting priors and a pseudo-ground-truth generation strategy to enable realistic synthesis under large-angle rotations. ExpanDyNeRF optimizes density and color features to improve scene reconstruction from challenging perspectives. We also present the Synthetic Dynamic Multiview (SynDM) dataset, the first synthetic multiview dataset for dynamic scenes with explicit side-view supervision-created using a custom GTA V-based rendering pipeline. Quantitative and qualitative results on SynDM and real-world datasets demonstrate that ExpanDyNeRF significantly outperforms existing dynamic NeRF methods in rendering fidelity under extreme viewpoint shifts. Further details are provided in the supplementary materials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14352v1">HGS: Hybrid Gaussian Splatting with Static-Dynamic Decomposition for Compact Dynamic View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 11 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Dynamic novel view synthesis (NVS) is essential for creating immersive experiences. Existing approaches have advanced dynamic NVS by introducing 3D Gaussian Splatting (3DGS) with implicit deformation fields or indiscriminately assigned time-varying parameters, surpassing NeRF-based methods. However, due to excessive model complexity and parameter redundancy, they incur large model sizes and slow rendering speeds, making them inefficient for real-time applications, particularly on resource-constrained devices. To obtain a more efficient model with fewer redundant parameters, in this paper, we propose Hybrid Gaussian Splatting (HGS), a compact and efficient framework explicitly designed to disentangle static and dynamic regions of a scene within a unified representation. The core innovation of HGS lies in our Static-Dynamic Decomposition (SDD) strategy, which leverages Radial Basis Function (RBF) modeling for Gaussian primitives. Specifically, for dynamic regions, we employ time-dependent RBFs to effectively capture temporal variations and handle abrupt scene changes, while for static regions, we reduce redundancy by sharing temporally invariant parameters. Additionally, we introduce a two-stage training strategy tailored for explicit models to enhance temporal coherence at static-dynamic boundaries. Experimental results demonstrate that our method reduces model size by up to 98% and achieves real-time rendering at up to 125 FPS at 4K resolution on a single RTX 3090 GPU. It further sustains 160 FPS at 1352 * 1014 on an RTX 3050 and has been integrated into the VR system. Moreover, HGS achieves comparable rendering quality to state-of-the-art methods while providing significantly improved visual fidelity for high-frequency details and abrupt scene changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.07733v2">RTR-GS: 3D Gaussian Splatting for Inverse Rendering with Radiance Transfer and Reflection</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated impressive capabilities in novel view synthesis. However, rendering reflective objects remains a significant challenge, particularly in inverse rendering and relighting. We introduce RTR-GS, a novel inverse rendering framework capable of robustly rendering objects with arbitrary reflectance properties, decomposing BRDF and lighting, and delivering credible relighting results. Given a collection of multi-view images, our method effectively recovers geometric structure through a hybrid rendering model that combines forward rendering for radiance transfer with deferred rendering for reflections. This approach successfully separates high-frequency and low-frequency appearances, mitigating floating artifacts caused by spherical harmonic overfitting when handling high-frequency details. We further refine BRDF and lighting decomposition using an additional physically-based deferred rendering branch. Experimental results show that our method enhances novel view synthesis, normal estimation, decomposition, and relighting while maintaining efficient training inference process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.15208v2">GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Transferring 2D textures to 3D modalities is of great significance for improving the efficiency of multimedia content creation. Existing approaches have rarely focused on transferring image textures onto 3D representations. 3D style transfer methods are capable of transferring abstract artistic styles to 3D scenes. However, these methods often overlook the geometric information of the scene, which makes it challenging to achieve high-quality 3D texture transfer results. In this paper, we present GT^2-GS, a geometry-aware texture transfer framework for gaussian splitting. From the perspective of matching texture features with geometric information in rendered views, we identify the issue of insufficient texture features and propose a geometry-aware texture augmentation module to expand the texture feature set. Moreover, a geometry-consistent texture loss is proposed to optimize texture features into the scene representation. This loss function incorporates both camera pose and 3D geometric information of the scene, enabling controllable texture-oriented appearance editing. Finally, a geometry preservation strategy is introduced. By alternating between the texture transfer and geometry correction stages over multiple iterations, this strategy achieves a balance between learning texture features and preserving geometric integrity. Extensive experiments demonstrate the effectiveness and controllability of our method. Through geometric awareness, our approach achieves texture transfer results that better align with human visual perception. Our homepage is available at https://vpx-ecnu.github.io/GT2-GS-website.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14200v1">Beyond a Single Light: A Large-Scale Aerial Dataset for Urban Scene Reconstruction Under Varying Illumination</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Recent advances in Neural Radiance Fields and 3D Gaussian Splatting have demonstrated strong potential for large-scale UAV-based 3D reconstruction tasks by fitting the appearance of images. However, real-world large-scale captures are often based on multi-temporal data capture, where illumination inconsistencies across different times of day can significantly lead to color artifacts, geometric inaccuracies, and inconsistent appearance. Due to the lack of UAV datasets that systematically capture the same areas under varying illumination conditions, this challenge remains largely underexplored. To fill this gap, we introduceSkyLume, a large-scale, real-world UAV dataset specifically designed for studying illumination robust 3D reconstruction in urban scene modeling: (1) We collect data from 10 urban regions data comprising more than 100k high resolution UAV images (four oblique views and nadir), where each region is captured at three periods of the day to systematically isolate illumination changes. (2) To support precise evaluation of geometry and appearance, we provide per-scene LiDAR scans and accurate 3D ground-truth for assessing depth, surface normals, and reconstruction quality under varying illumination. (3) For the inverse rendering task, we introduce the Temporal Consistency Coefficient (TCC), a metric that measuress cross-time albedo stability and directly evaluates the robustness of the disentanglement of light and material. We aim for this resource to serve as a foundation that advances research and real-world evaluation in large-scale inverse rendering, geometry reconstruction, and novel view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14180v1">Spherical Voronoi: Directional Appearance as a Differentiable Partition of the Sphere</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Radiance field methods (e.g. 3D Gaussian Splatting) have emerged as a powerful paradigm for novel view synthesis, yet their appearance modeling often relies on Spherical Harmonics (SH), which impose fundamental limitations. SH struggle with high-frequency signals, exhibit Gibbs ringing artifacts, and fail to capture specular reflections - a key component of realistic rendering. Although alternatives like spherical Gaussians offer improvements, they add significant optimization complexity. We propose Spherical Voronoi (SV) as a unified framework for appearance representation in 3D Gaussian Splatting. SV partitions the directional domain into learnable regions with smooth boundaries, providing an intuitive and stable parameterization for view-dependent effects. For diffuse appearance, SV achieves competitive results while keeping optimization simpler than existing alternatives. For reflections - where SH fail - we leverage SV as learnable reflection probes, taking reflected directions as input following principles from classical graphics. This formulation attains state-of-the-art results on synthetic and real-world datasets, demonstrating that SV offers a principled, efficient, and general solution for appearance modeling in explicit 3D representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14087v1">GaussianPlant: Structure-aligned Gaussian Splatting for 3D Reconstruction of Plants</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Submitted to IEEE TPAMI, under review
    </div>
    <details class="paper-abstract">
      We present a method for jointly recovering the appearance and internal structure of botanical plants from multi-view images based on 3D Gaussian Splatting (3DGS). While 3DGS exhibits robust reconstruction of scene appearance for novel-view synthesis, it lacks structural representations underlying those appearances (e.g., branching patterns of plants), which limits its applicability to tasks such as plant phenotyping. To achieve both high-fidelity appearance and structural reconstruction, we introduce GaussianPlant, a hierarchical 3DGS representation, which disentangles structure and appearance. Specifically, we employ structure primitives (StPs) to explicitly represent branch and leaf geometry, and appearance primitives (ApPs) to the plants' appearance using 3D Gaussians. StPs represent a simplified structure of the plant, i.e., modeling branches as cylinders and leaves as disks. To accurately distinguish the branches and leaves, StP's attributes (i.e., branches or leaves) are optimized in a self-organized manner. ApPs are bound to each StP to represent the appearance of branches or leaves as in conventional 3DGS. StPs and ApPs are jointly optimized using a re-rendering loss on the input multi-view images, as well as the gradient flow from ApP to StP using the binding correspondence information. We conduct experiments to qualitatively evaluate the reconstruction accuracy of both appearance and structure, as well as real-world experiments to qualitatively validate the practical performance. Experiments show that the GaussianPlant achieves both high-fidelity appearance reconstruction via ApPs and accurate structural reconstruction via StPs, enabling the extraction of branch structure and leaf instances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14039v1">ASAP-Textured Gaussians: Enhancing Textured Gaussians with Adaptive Sampling and Anisotropic Parameterization</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Recent advances have equipped 3D Gaussian Splatting with texture parameterizations to capture spatially varying attributes, improving the performance of both appearance modeling and downstream tasks. However, the added texture parameters introduce significant memory efficiency challenges. Rather than proposing new texture formulations, we take a step back to examine the characteristics of existing textured Gaussian methods and identify two key limitations in common: (1) Textures are typically defined in canonical space, leading to inefficient sampling that wastes textures' capacity on low-contribution regions; and (2) texture parameterization is uniformly assigned across all Gaussians, regardless of their visual complexity, resulting in over-parameterization. In this work, we address these issues through two simple yet effective strategies: adaptive sampling based on the Gaussian density distribution and error-driven anisotropic parameterization that allocates texture resources according to rendering error. Our proposed ASAP Textured Gaussians, short for Adaptive Sampling and Anisotropic Parameterization, significantly improve the quality efficiency tradeoff, achieving high-fidelity rendering with far fewer texture parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09397v2">OUGS: Active View Selection via Object-aware Uncertainty Estimation in 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Conditionally accepted to Eurographics 2026 (five reviewers)
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have achieved state-of-the-art results for novel view synthesis. However, efficiently capturing high-fidelity reconstructions of specific objects within complex scenes remains a significant challenge. A key limitation of existing active reconstruction methods is their reliance on scene-level uncertainty metrics, which are often biased by irrelevant background clutter and lead to inefficient view selection for object-centric tasks. We present OUGS, a novel framework that addresses this challenge with a more principled, physically-grounded uncertainty formulation for 3DGS. Our core innovation is to derive uncertainty directly from the explicit physical parameters of the 3D Gaussian primitives (e.g., position, scale, rotation). By propagating the covariance of these parameters through the rendering Jacobian, we establish a highly interpretable uncertainty model. This foundation allows us to then seamlessly integrate semantic segmentation masks to produce a targeted, object-aware uncertainty score that effectively disentangles the object from its environment. This allows for a more effective active view selection strategy that prioritizes views critical to improving object fidelity. Experimental evaluations on public datasets demonstrate that our approach significantly improves the efficiency of the 3DGS reconstruction process and achieves higher quality for targeted objects compared to existing state-of-the-art methods, while also serving as a robust uncertainty estimator for the global scene.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13411v1">Computer vision training dataset generation for robotic environments using Gaussian splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Code available at: https://patrykni.github.io/UnitySplat2Data/
    </div>
    <details class="paper-abstract">
      This paper introduces a novel pipeline for generating large-scale, highly realistic, and automatically labeled datasets for computer vision tasks in robotic environments. Our approach addresses the critical challenges of the domain gap between synthetic and real-world imagery and the time-consuming bottleneck of manual annotation. We leverage 3D Gaussian Splatting (3DGS) to create photorealistic representations of the operational environment and objects. These assets are then used in a game engine where physics simulations create natural arrangements. A novel, two-pass rendering technique combines the realism of splats with a shadow map generated from proxy meshes. This map is then algorithmically composited with the image to add both physically plausible shadows and subtle highlights, significantly enhancing realism. Pixel-perfect segmentation masks are generated automatically and formatted for direct use with object detection models like YOLO. Our experiments show that a hybrid training strategy, combining a small set of real images with a large volume of our synthetic data, yields the best detection and segmentation performance, confirming this as an optimal strategy for efficiently achieving robust and accurate models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08334v2">HybridSplat: Fast Reflection-baked Gaussian Tracing using Hybrid Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Project URL: https://aetheryne.github.io/HybridSplat/
    </div>
    <details class="paper-abstract">
      Rendering complex reflection of real-world scenes using 3D Gaussian splatting has been a quite promising solution for photorealistic novel view synthesis, but still faces bottlenecks especially in rendering speed and memory storage. This paper proposes a new Hybrid Splatting(HybridSplat) mechanism for Gaussian primitives. Our key idea is a new reflection-baked Gaussian tracing, which bakes the view-dependent reflection within each Gaussian primitive while rendering the reflection using tile-based Gaussian splatting. Then we integrate the reflective Gaussian primitives with base Gaussian primitives using a unified hybrid splatting framework for high-fidelity scene reconstruction. Moreover, we further introduce a pipeline-level acceleration for the hybrid splatting, and reflection-sensitive Gaussian pruning to reduce the model size, thus achieving much faster rendering speed and lower memory storage while preserving the reflection rendering quality. By extensive evaluation, our HybridSplat accelerates about 7x rendering speed across complex reflective scenes from Ref-NeRF, NeRF-Casting with 4x fewer Gaussian primitives than similar ray-tracing based Gaussian splatting baselines, serving as a new state-of-the-art method especially for complex reflective scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.21307v2">Towards Physically Executable 3D Gaussian for Embodied Navigation</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Project Page: https://sage-3d.github.io/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS), a 3D representation method with photorealistic real-time rendering capabilities, is regarded as an effective tool for narrowing the sim-to-real gap. However, it lacks fine-grained semantics and physical executability for Visual-Language Navigation (VLN). To address this, we propose SAGE-3D (Semantically and Physically Aligned Gaussian Environments for 3D Navigation), a new paradigm that upgrades 3DGS into an executable, semantically and physically aligned environment. It comprises two components: (1) Object-Centric Semantic Grounding, which adds object-level fine-grained annotations to 3DGS; and (2) Physics-Aware Execution Jointing, which embeds collision objects into 3DGS and constructs rich physical interfaces. We release InteriorGS, containing 1K object-annotated 3DGS indoor scene data, and introduce SAGE-Bench, the first 3DGS-based VLN benchmark with 2M VLN data. Experiments show that 3DGS scene data is more difficult to converge, while exhibiting strong generalizability, improving baseline performance by 31% on the VLN-CE Unseen task. Our data and code are available at: https://sage-3d.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13007v1">Light Field Based 6DoF Tracking of Previously Unobserved Objects</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Object tracking is an important step in robotics and reautonomous driving pipelines, which has to generalize to previously unseen and complex objects. Existing high-performing methods often rely on pre-captured object views to build explicit reference models, which restricts them to a fixed set of known objects. However, such reference models can struggle with visually complex appearance, reducing the quality of tracking. In this work, we introduce an object tracking method based on light field images that does not depend on a pre-trained model, while being robust to complex visual behavior, such as reflections. We extract semantic and geometric features from light field inputs using vision foundation models and convert them into view-dependent Gaussian splats. These splats serve as a unified object representation, supporting differentiable rendering and pose optimization. We further introduce a light field object tracking dataset containing challenging reflective objects with precise ground truth poses. Experiments demonstrate that our method is competitive with state-of-the-art model-based trackers in these difficult cases, paving the way toward universal object tracking in robotic systems. Code/data available at https://github.com/nagonch/LiFT-6DoF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12898v1">Qonvolution: Towards Learning High-Frequency Signals with Queried Convolution</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 28 pages, 8 figures, Project Page: https://abhi1kumar.github.io/qonvolution/
    </div>
    <details class="paper-abstract">
      Accurately learning high-frequency signals is a challenge in computer vision and graphics, as neural networks often struggle with these signals due to spectral bias or optimization difficulties. While current techniques like Fourier encodings have made great strides in improving performance, there remains scope for improvement when presented with high-frequency information. This paper introduces Queried-Convolutions (Qonvolutions), a simple yet powerful modification using the neighborhood properties of convolution. Qonvolution convolves a low-frequency signal with queries (such as coordinates) to enhance the learning of intricate high-frequency signals. We empirically demonstrate that Qonvolutions enhance performance across a variety of high-frequency learning tasks crucial to both the computer vision and graphics communities, including 1D regression, 2D super-resolution, 2D image regression, and novel view synthesis (NVS). In particular, by combining Gaussian splatting with Qonvolutions for NVS, we showcase state-of-the-art performance on real-world complex scenes, even outperforming powerful radiance field models on image quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13796v1">Nexels: Neurally-Textured Surfels for Real-Time Novel View Synthesis with Sparse Geometries</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Webpage at https://lessvrong.com/cs/nexels
    </div>
    <details class="paper-abstract">
      Though Gaussian splatting has achieved impressive results in novel view synthesis, it requires millions of primitives to model highly textured scenes, even when the geometry of the scene is simple. We propose a representation that goes beyond point-based rendering and decouples geometry and appearance in order to achieve a compact representation. We use surfels for geometry and a combination of a global neural field and per-primitive colours for appearance. The neural field textures a fixed number of primitives for each pixel, ensuring that the added compute is low. Our representation matches the perceptual quality of 3D Gaussian splatting while using $9.7\times$ fewer primitives and $5.5\times$ less memory on outdoor scenes and using $31\times$ fewer primitives and $3.7\times$ less memory on indoor scenes. Our representation also renders twice as fast as existing textured primitives while improving upon their visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12774v1">Fast 2DGS: Efficient Image Representation with Deep Gaussian Prior</a></div>
    <div class="paper-meta">
      📅 2025-12-14
    </div>
    <details class="paper-abstract">
      As generative models become increasingly capable of producing high-fidelity visual content, the demand for efficient, interpretable, and editable image representations has grown substantially. Recent advances in 2D Gaussian Splatting (2DGS) have emerged as a promising solution, offering explicit control, high interpretability, and real-time rendering capabilities (>1000 FPS). However, high-quality 2DGS typically requires post-optimization. Existing methods adopt random or heuristics (e.g., gradient maps), which are often insensitive to image complexity and lead to slow convergence (>10s). More recent approaches introduce learnable networks to predict initial Gaussian configurations, but at the cost of increased computational and architectural complexity. To bridge this gap, we present Fast-2DGS, a lightweight framework for efficient Gaussian image representation. Specifically, we introduce Deep Gaussian Prior, implemented as a conditional network to capture the spatial distribution of Gaussian primitives under different complexities. In addition, we propose an attribute regression network to predict dense Gaussian properties. Experiments demonstrate that this disentangled architecture achieves high-quality reconstruction in a single forward pass, followed by minimal fine-tuning. More importantly, our approach significantly reduces computational cost without compromising visual quality, bringing 2DGS closer to industry-ready deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02648v2">PoreTrack3D: A Benchmark for Dynamic 3D Gaussian Splatting in Pore-Scale Facial Trajectory Tracking</a></div>
    <div class="paper-meta">
      📅 2025-12-14
    </div>
    <details class="paper-abstract">
      We introduce PoreTrack3D, the first benchmark for dynamic 3D Gaussian splatting in pore-scale, non-rigid 3D facial trajectory tracking. It contains over 440,000 facial trajectories in total, among which more than 52,000 are longer than 10 frames, including 68 manually reviewed trajectories that span the entire 150 frames. To the best of our knowledge, PoreTrack3D is the first benchmark dataset to capture both traditional facial landmarks and pore-scale keypoints trajectory, advancing the study of fine-grained facial expressions through the analysis of subtle skin-surface motion. We systematically evaluate state-of-the-art dynamic 3D Gaussian splatting methods on PoreTrack3D, establishing the first performance baseline in this domain. Overall, the pipeline developed for this benchmark dataset's creation establishes a new framework for high-fidelity facial motion capture and dynamic 3D reconstruction. Our dataset are publicly available at: https://github.com/JHXion9/PoreTrack3D
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.16809v2">GeoTexDensifier: Geometry-Texture-Aware Densification for High-Quality Photorealistic 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-13
      | 💬 14 pages, 9 figures, 2 table
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently attracted wide attentions in various areas such as 3D navigation, Virtual Reality (VR) and 3D simulation, due to its photorealistic and efficient rendering performance. High-quality reconstrution of 3DGS relies on sufficient splats and a reasonable distribution of these splats to fit real geometric surface and texture details, which turns out to be a challenging problem. We present GeoTexDensifier, a novel geometry-texture-aware densification strategy to reconstruct high-quality Gaussian splats which better comply with the geometric structure and texture richness of the scene. Specifically, our GeoTexDensifier framework carries out an auxiliary texture-aware densification method to produce a denser distribution of splats in fully textured areas, while keeping sparsity in low-texture regions to maintain the quality of Gaussian point cloud. Meanwhile, a geometry-aware splitting strategy takes depth and normal priors to guide the splitting sampling and filter out the noisy splats whose initial positions are far from the actual geometric surfaces they aim to fit, under a Validation of Depth Ratio Change checking. With the help of relative monocular depth prior, such geometry-aware validation can effectively reduce the influence of scattered Gaussians to the final rendering quality, especially in regions with weak textures or without sufficient training views. The texture-aware densification and geometry-aware splitting strategies are fully combined to obtain a set of high-quality Gaussian splats. We experiment our GeoTexDensifier framework on various datasets and compare our Novel View Synthesis results to other state-of-the-art 3DGS approaches, with detailed quantitative and qualitative evaluations to demonstrate the effectiveness of our method in producing more photorealistic 3DGS models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.02421v2">TimeWalker: Personalized Neural Space for Lifelong Head Avatars</a></div>
    <div class="paper-meta">
      📅 2025-12-13
      | 💬 Project Page: https://timewalker2025.github.io/timewalker.github.io/
    </div>
    <details class="paper-abstract">
      We present TimeWalker, a novel framework that models realistic, full-scale 3D head avatars of a person on lifelong scale. Unlike current human head avatar pipelines that capture identity at the momentary level(e.g., instant photography or short videos), TimeWalker constructs a person's comprehensive identity from unstructured data collection over his/her various life stages, offering a paradigm to achieve full reconstruction and animation of that person at different moments of life. At the heart of TimeWalker's success is a novel neural parametric model that learns personalized representation with the disentanglement of shape, expression, and appearance across ages. Central to our methodology are the concepts of two aspects: (1) We track back to the principle of modeling a person's identity in an additive combination of average head representation in the canonical space, and moment-specific head attribute representations driven from a set of neural head basis. To learn the set of head basis that could represent the comprehensive head variations in a compact manner, we propose a Dynamic Neural Basis-Blending Module (Dynamo). It dynamically adjusts the number and blend weights of neural head bases, according to both shared and specific traits of the target person over ages. (2) Dynamic 2D Gaussian Splatting (DNA-2DGS), an extension of Gaussian splatting representation, to model head motion deformations like facial expressions without losing the realism of rendering and reconstruction. DNA-2DGS includes a set of controllable 2D oriented planar Gaussian disks that utilize the priors from parametric model, and move/rotate with the change of expression. Through extensive experimental evaluations, we show TimeWalker's ability to reconstruct and animate avatars across decoupled dimensions with realistic rendering effects, demonstrating a way to achieve personalized 'time traveling' in a breeze.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11800v1">Moment-Based 3D Gaussian Splatting: Resolving Volumetric Occlusion with Order-Independent Transmittance</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      The recent success of 3D Gaussian Splatting (3DGS) has reshaped novel view synthesis by enabling fast optimization and real-time rendering of high-quality radiance fields. However, it relies on simplified, order-dependent alpha blending and coarse approximations of the density integral within the rasterizer, thereby limiting its ability to render complex, overlapping semi-transparent objects. In this paper, we extend rasterization-based rendering of 3D Gaussian representations with a novel method for high-fidelity transmittance computation, entirely avoiding the need for ray tracing or per-pixel sample sorting. Building on prior work in moment-based order-independent transparency, our key idea is to characterize the density distribution along each camera ray with a compact and continuous representation based on statistical moments. To this end, we analytically derive and compute a set of per-pixel moments from all contributing 3D Gaussians. From these moments, a continuous transmittance function is reconstructed for each ray, which is then independently sampled within each Gaussian. As a result, our method bridges the gap between rasterization and physical accuracy by modeling light attenuation in complex translucent media, significantly improving overall reconstruction and rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08710v3">SceneSplat++: A Large Dataset and Comprehensive Benchmark for Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 15 pages, codes, data and benchmark are released
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) serves as a highly performant and efficient encoding of scene geometry, appearance, and semantics. Moreover, grounding language in 3D scenes has proven to be an effective strategy for 3D scene understanding. Current Language Gaussian Splatting line of work fall into three main groups: (i) per-scene optimization-based, (ii) per-scene optimization-free, and (iii) generalizable approach. However, most of them are evaluated only on rendered 2D views of a handful of scenes and viewpoints close to the training views, limiting ability and insight into holistic 3D understanding. To address this gap, we propose the first large-scale benchmark that systematically assesses these three groups of methods directly in 3D space, evaluating on 1060 scenes across three indoor datasets and one outdoor dataset. Benchmark results demonstrate a clear advantage of the generalizable paradigm, particularly in relaxing the scene-specific limitation, enabling fast feed-forward inference on novel scenes, and achieving superior segmentation performance. We further introduce GaussianWorld-49K a carefully curated 3DGS dataset comprising around 49K diverse indoor and outdoor scenes obtained from multiple sources, with which we demonstrate the generalizable approach could harness strong data priors. Our codes, benchmark, and datasets are released at https://scenesplatpp.gaussianworld.ai/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02069v3">Spec-Gloss Surfels and Normal-Diffuse Priors for Relightable Glossy Objects</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Accurate reconstruction and relighting of glossy objects remains a longstanding challenge, as object shape, material properties, and illumination are inherently difficult to disentangle. Existing neural rendering approaches often rely on simplified BRDF models or parameterizations that couple diffuse and specular components, which restrict faithful material recovery and limit relighting fidelity. We propose a relightable framework that integrates a microfacet BRDF with the specular-glossiness parameterization into 2D Gaussian Splatting with deferred shading. This formulation enables more physically consistent material decomposition, while diffusion-based priors for surface normals and diffuse color guide early-stage optimization and mitigate ambiguity. A coarse-to-fine environment map optimization accelerates convergence, and negative-only environment map clipping preserves high-dynamic-range specular reflections. Extensive experiments on complex, glossy scenes demonstrate that our method achieves high-quality geometry and material reconstruction, delivering substantially more realistic and consistent relighting under novel illumination compared to existing Gaussian splatting methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11356v1">Prior-Enhanced Gaussian Splatting for Dynamic Scene Reconstruction from Casual Video</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      We introduce a fully automatic pipeline for dynamic scene reconstruction from casually captured monocular RGB videos. Rather than designing a new scene representation, we enhance the priors that drive Dynamic Gaussian Splatting. Video segmentation combined with epipolar-error maps yields object-level masks that closely follow thin structures; these masks (i) guide an object-depth loss that sharpens the consistent video depth, and (ii) support skeleton-based sampling plus mask-guided re-identification to produce reliable, comprehensive 2-D tracks. Two additional objectives embed the refined priors in the reconstruction stage: a virtual-view depth loss removes floaters, and a scaffold-projection loss ties motion nodes to the tracks, preserving fine geometry and coherent motion. The resulting system surpasses previous monocular dynamic scene reconstruction methods and delivers visibly superior renderings
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11186v1">Lightweight 3D Gaussian Splatting Compression via Video Codec</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 Accepted by DCC2026 Oral
    </div>
    <details class="paper-abstract">
      Current video-based GS compression methods rely on using Parallel Linear Assignment Sorting (PLAS) to convert 3D GS into smooth 2D maps, which are computationally expensive and time-consuming, limiting the application of GS on lightweight devices. In this paper, we propose a Lightweight 3D Gaussian Splatting (GS) Compression method based on Video codec (LGSCV). First, a two-stage Morton scan is proposed to generate blockwise 2D maps that are friendly for canonical video codecs in which the coding units (CU) are square blocks. A 3D Morton scan is used to permute GS primitives, followed by a 2D Morton scan to map the ordered GS primitives to 2D maps in a blockwise style. However, although the blockwise 2D maps report close performance to the PLAS map in high-bitrate regions, they show a quality collapse at medium-to-low bitrates. Therefore, a principal component analysis (PCA) is used to reduce the dimensionality of spherical harmonics (SH), and a MiniPLAS, which is flexible and fast, is designed to permute the primitives within certain block sizes. Incorporating SH PCA and MiniPLAS leads to a significant gain in rate-distortion (RD) performance, especially at medium and low bitrates. MiniPLAS can also guide the setting of the codec CU size configuration and significantly reduce encoding time. Experimental results on the MPEG dataset demonstrate that the proposed LGSCV achieves over 20% RD gain compared with state-of-the-art methods, while reducing 2D map generation time to approximately 1 second and cutting encoding time by 50%. The code is available at https://github.com/Qi-Yangsjtu/LGSCV .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10939v1">GaussianHeadTalk: Wobble-Free 3D Talking Heads with Audio Driven Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 IEEE/CVF Winter Conference on Applications of Computer Vision 2026
    </div>
    <details class="paper-abstract">
      Speech-driven talking heads have recently emerged and enable interactive avatars. However, real-world applications are limited, as current methods achieve high visual fidelity but slow or fast yet temporally unstable. Diffusion methods provide realistic image generation, yet struggle with oneshot settings. Gaussian Splatting approaches are real-time, yet inaccuracies in facial tracking, or inconsistent Gaussian mappings, lead to unstable outputs and video artifacts that are detrimental to realistic use cases. We address this problem by mapping Gaussian Splatting using 3D Morphable Models to generate person-specific avatars. We introduce transformer-based prediction of model parameters, directly from audio, to drive temporal consistency. From monocular video and independent audio speech inputs, our method enables generation of real-time talking head videos where we report competitive quantitative and qualitative performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10572v1">DeMapGS: Simultaneous Mesh Deformation and Surface Attribute Mapping via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Project page see https://shuyizhou495.github.io/DeMapGS-project-page/
    </div>
    <details class="paper-abstract">
      We propose DeMapGS, a structured Gaussian Splatting framework that jointly optimizes deformable surfaces and surface-attached 2D Gaussian splats. By anchoring splats to a deformable template mesh, our method overcomes topological inconsistencies and enhances editing flexibility, addressing limitations of prior Gaussian Splatting methods that treat points independently. The unified representation in our method supports extraction of high-fidelity diffuse, normal, and displacement maps, enabling the reconstructed mesh to inherit the photorealistic rendering quality of Gaussian Splatting. To support robust optimization, we introduce a gradient diffusion strategy that propagates supervision across the surface, along with an alternating 2D/3D rendering scheme to handle concave regions. Experiments demonstrate that DeMapGS achieves state-of-the-art mesh reconstruction quality and enables downstream applications for Gaussian splats such as editing and cross-object manipulation through a shared parametric surface.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10424v1">Neural Hamiltonian Deformation Fields for Dynamic Scene Rendering</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Accepted by ACM SIGGRAPH Asia 2025, project page: https://qin-jingyun.github.io/NeHaD
    </div>
    <details class="paper-abstract">
      Representing and rendering dynamic scenes with complex motions remains challenging in computer vision and graphics. Recent dynamic view synthesis methods achieve high-quality rendering but often produce physically implausible motions. We introduce NeHaD, a neural deformation field for dynamic Gaussian Splatting governed by Hamiltonian mechanics. Our key observation is that existing methods using MLPs to predict deformation fields introduce inevitable biases, resulting in unnatural dynamics. By incorporating physics priors, we achieve robust and realistic dynamic scene rendering. Hamiltonian mechanics provides an ideal framework for modeling Gaussian deformation fields due to their shared phase-space structure, where primitives evolve along energy-conserving trajectories. We employ Hamiltonian neural networks to implicitly learn underlying physical laws governing deformation. Meanwhile, we introduce Boltzmann equilibrium decomposition, an energy-aware mechanism that adaptively separates static and dynamic Gaussians based on their spatial-temporal energy states for flexible rendering. To handle real-world dissipation, we employ second-order symplectic integration and local rigidity regularization as physics-informed constraints for robust dynamics modeling. Additionally, we extend NeHaD to adaptive streaming through scale-aware mipmapping and progressive optimization. Extensive experiments demonstrate that NeHaD achieves physically plausible results with a rendering quality-efficiency trade-off. To our knowledge, this is the first exploration leveraging Hamiltonian mechanics for neural Gaussian deformation, enabling physically realistic dynamic scene rendering with streaming capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.07752v2">DEGS: Deformable Event-based 3D Gaussian Splatting from RGB and Event Stream</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Accepted by IEEE TVCG
    </div>
    <details class="paper-abstract">
      Reconstructing Dynamic 3D Gaussian Splatting (3DGS) from low-framerate RGB videos is challenging. This is because large inter-frame motions will increase the uncertainty of the solution space. For example, one pixel in the first frame might have more choices to reach the corresponding pixel in the second frame. Event cameras can asynchronously capture rapid visual changes and are robust to motion blur, but they do not provide color information. Intuitively, the event stream can provide deterministic constraints for the inter-frame large motion by the event trajectories. Hence, combining low-temporal-resolution images with high-framerate event streams can address this challenge. However, it is challenging to jointly optimize Dynamic 3DGS using both RGB and event modalities due to the significant discrepancy between these two data modalities. This paper introduces a novel framework that jointly optimizes dynamic 3DGS from the two modalities. The key idea is to adopt event motion priors to guide the optimization of the deformation fields. First, we extract the motion priors encoded in event streams by using the proposed LoCM unsupervised fine-tuning framework to adapt an event flow estimator to a certain unseen scene. Then, we present the geometry-aware data association method to build the event-Gaussian motion correspondence, which is the primary foundation of the pipeline, accompanied by two useful strategies, namely motion decomposition and inter-frame pseudo-label. Extensive experiments show that our method outperforms existing image and event-based approaches across synthetic and real scenes and prove that our method can effectively optimize dynamic 3DGS with the help of event data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10369v1">Breaking the Vicious Cycle: Coherent 3D Gaussian Splatting from Sparse and Motion-Blurred Views</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 20 pages, 14 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a state-of-the-art method for novel view synthesis. However, its performance heavily relies on dense, high-quality input imagery, an assumption that is often violated in real-world applications, where data is typically sparse and motion-blurred. These two issues create a vicious cycle: sparse views ignore the multi-view constraints necessary to resolve motion blur, while motion blur erases high-frequency details crucial for aligning the limited views. Thus, reconstruction often fails catastrophically, with fragmented views and a low-frequency bias. To break this cycle, we introduce CoherentGS, a novel framework for high-fidelity 3D reconstruction from sparse and blurry images. Our key insight is to address these compound degradations using a dual-prior strategy. Specifically, we combine two pre-trained generative models: a specialized deblurring network for restoring sharp details and providing photometric guidance, and a diffusion model that offers geometric priors to fill in unobserved regions of the scene. This dual-prior strategy is supported by several key techniques, including a consistency-guided camera exploration module that adaptively guides the generative process, and a depth regularization loss that ensures geometric plausibility. We evaluate CoherentGS through both quantitative and qualitative experiments on synthetic and real-world scenes, using as few as 3, 6, and 9 input views. Our results demonstrate that CoherentGS significantly outperforms existing methods, setting a new state-of-the-art for this challenging task. The code and video demos are available at https://potatobigroom.github.io/CoherentGS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17951v4">SplatCo: Structure-View Collaborative Gaussian Splatting for Detail-Preserving Rendering of Large-Scale Unbounded Scenes</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      We present SplatCo, a structure-view collaborative Gaussian splatting framework for high-fidelity rendering of complex outdoor scenes. SplatCo builds upon three novel components: 1) a cross-structure collaboration module that combines global tri-plane representations, which capture coarse scene layouts, with local context grid features representing fine details. This fusion is achieved through a hierarchical compensation mechanism, ensuring both global spatial awareness and local detail preservation; 2) a cross-view pruning mechanism that removes overfitted or inaccurate Gaussians based on structural consistency, thereby improving storage efficiency and preventing rendering artifacts; 3) a structure view co-learning module that aggregates structural gradients with view gradients,thereby steering the optimization of Gaussian geometric and appearance attributes more robustly. By combining these key components, SplatCo effectively achieves high-fidelity rendering for large-scale scenes. Code and project page are available at https://splatco-tech.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10293v1">Physically Aware 360$^\circ$ View Generation from a Single Image using Disentangled Scene Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      We introduce Disentangled360, an innovative 3D-aware technology that integrates the advantages of direction disentangled volume rendering with single-image 360° unique view synthesis for applications in medical imaging and natural scene reconstruction. In contrast to current techniques that either oversimplify anisotropic light behavior or lack generalizability across various contexts, our framework distinctly differentiates between isotropic and anisotropic contributions inside a Gaussian Splatting backbone. We implement a dual-branch conditioning framework, one optimized for CT intensity driven scattering in volumetric data and the other for real-world RGB scenes through normalized camera embeddings. To address scale ambiguity and maintain structural realism, we present a hybrid pose agnostic anchoring method that adaptively samples scene depth and material transitions, functioning as stable pivots during scene distillation. Our design integrates preoperative radiography simulation and consumer-grade 360° rendering into a singular inference pipeline, facilitating rapid, photorealistic view synthesis with inherent directionality. Evaluations on the Mip-NeRF 360, RealEstate10K, and DeepDRR datasets indicate superior SSIM and LPIPS performance, while runtime assessments confirm its viability for interactive applications. Disentangled360 facilitates mixed-reality medical supervision, robotic perception, and immersive content creation, eliminating the necessity for scene-specific finetuning or expensive photon simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09335v2">Relightable and Dynamic Gaussian Avatar Reconstruction from Monocular Video</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 8 pages, 9 figures, published in ACM MM 2025
    </div>
    <details class="paper-abstract">
      Modeling relightable and animatable human avatars from monocular video is a long-standing and challenging task. Recently, Neural Radiance Field (NeRF) and 3D Gaussian Splatting (3DGS) methods have been employed to reconstruct the avatars. However, they often produce unsatisfactory photo-realistic results because of insufficient geometrical details related to body motion, such as clothing wrinkles. In this paper, we propose a 3DGS-based human avatar modeling framework, termed as Relightable and Dynamic Gaussian Avatar (RnD-Avatar), that presents accurate pose-variant deformation for high-fidelity geometrical details. To achieve this, we introduce dynamic skinning weights that define the human avatar's articulation based on pose while also learning additional deformations induced by body motion. We also introduce a novel regularization to capture fine geometric details under sparse visual cues. Furthermore, we present a new multi-view dataset with varied lighting conditions to evaluate relight. Our framework enables realistic rendering of novel poses and views while supporting photo-realistic lighting effects under arbitrary lighting conditions. Our method achieves state-of-the-art performance in novel view synthesis, novel pose rendering, and relighting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10267v1">Long-LRM++: Preserving Fine Details in Feed-Forward Wide-Coverage Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Recent advances in generalizable Gaussian splatting (GS) have enabled feed-forward reconstruction of scenes from tens of input views. Long-LRM notably scales this paradigm to 32 input images at $950\times540$ resolution, achieving 360° scene-level reconstruction in a single forward pass. However, directly predicting millions of Gaussian parameters at once remains highly error-sensitive: small inaccuracies in positions or other attributes lead to noticeable blurring, particularly in fine structures such as text. In parallel, implicit representation methods such as LVSM and LaCT have demonstrated significantly higher rendering fidelity by compressing scene information into model weights rather than explicit Gaussians, and decoding RGB frames using the full transformer or TTT backbone. However, this computationally intensive decompression process for every rendered frame makes real-time rendering infeasible. These observations raise key questions: Is the deep, sequential "decompression" process necessary? Can we retain the benefits of implicit representations while enabling real-time performance? We address these questions with Long-LRM++, a model that adopts a semi-explicit scene representation combined with a lightweight decoder. Long-LRM++ matches the rendering quality of LaCT on DL3DV while achieving real-time 14 FPS rendering on an A100 GPU, overcoming the speed limitations of prior implicit methods. Our design also scales to 64 input views at the $950\times540$ resolution, demonstrating strong generalization to increased input lengths. Additionally, Long-LRM++ delivers superior novel-view depth prediction on ScanNetv2 compared to direct depth rendering from Gaussians. Extensive ablation studies validate the effectiveness of each component in the proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12370v2">Changes in Real Time: Online Scene Change Detection with Multi-View Fusion</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Online Scene Change Detection (SCD) is an extremely challenging problem that requires an agent to detect relevant changes on the fly while observing the scene from unconstrained viewpoints. Existing online SCD methods are significantly less accurate than offline approaches. We present the first online SCD approach that is pose-agnostic, label-free, and ensures multi-view consistency, while operating at over 10 FPS and achieving new state-of-the-art performance, surpassing even the best offline approaches. Our method introduces a new self-supervised fusion loss to infer scene changes from multiple cues and observations, PnP-based fast pose estimation against the reference scene, and a fast change-guided update strategy for the 3D Gaussian Splatting scene representation. Extensive experiments on complex real-world datasets demonstrate that our approach outperforms both online and offline baselines.
    </details>
</div>
