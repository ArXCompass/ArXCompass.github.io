# gaussian splatting - 2026_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.23100v1">Splat-CBF: Safe Next-Best-View Control in 3D Gaussian-Splat Maps</a></div>
    <div class="paper-meta">
      📅 2026-09-19
    </div>
    <details class="paper-abstract">
      Where to look and how to move? A robot navigating an unmapped environment must do both at once, and the two goals pull against each other. The regions most worth observing are the ones the map knows least about, and those are exactly where the robot cannot trust its collision margins. We resolve this tension by introducing Splat-CBF, an active perception control barrier function that steers the camera toward the next best view while collision avoidance is enforced as a hard constraint. Safety is enforced by a risk-aware control barrier function that turns the Average Value-at-Risk of the Gaussian field into a single smooth hard constraint. Perception is enforced by a second barrier that rewards camera orientations with high expected Fisher information gain near the robot's planned path. The two meet in a quadratic program where safety is hard and perception is soft, with a slack penalty that adapts to how often perception has already been relaxed and how close the robot is to an uncertain region. We verify the method in indoor simulations, a Isaac Kinova manipulator and in experiments on an Ackermann-drive robot. Our results assert that robot navigates faster, gathers more information, and runs faster online than safety-only and perception-only baselines, giving up informative motion only when safety requires it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.23049v1">VDGS: Visibility-Driven Large-Scale 3D Gaussian Splatting for Aerial Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-09-19
      | 💬 8 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large-scale scene reconstruction is a critical foundational technology in robotic autonomous systems such as 3D mapping and autonomous driving. In recent years, 3D Gaussian Splatting (3DGS) has demonstrated remarkable advantages in both visual quality and computational efficiency, making it a promising representation for large-scale scene reconstruction. However, it still faces challenges in large-scale scenes, including excessive memory consumption and uneven viewpoint coverage caused by UAV acquisition, limiting its real-world applications. To address this, we propose VDGS, a novel 3DGS framework that incorporates camera distribution into scene modeling. VDGS introduces visibility-driven statistics for scene anchors to quantify supervision strength. These statistics are further leveraged for scene partitioning and for gradient compensation in under-optimized regions, thereby promoting balanced optimization across different regions. Extensive experiments on multiple large-scale aerial scene datasets demonstrate that, under imbalanced viewpoint distributions, VDGS consistently outperforms existing methods, while maintaining competitive performance in scenarios with more uniform view distributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.23005v1">Compressing 3D Gaussian Splatting via Cross-Representation Priors</a></div>
    <div class="paper-meta">
      📅 2026-09-19
      | 💬 14 pages, 8 figures. Accepted for publication in IEEE Transactions on Image Processing
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables high-quality novel view synthesis but incurs high storage and transmission costs due to dense Gaussian primitives. Recent anchor-based compression reduces per-primitive redundancy, yet redundancy across anchors remains largely unexploited. We propose CRP-GS (Cross-Representation Priors for Gaussian Splatting), a rate-distortion optimized compression framework that leverages cross-representation priors to improve anchor-level entropy modeling. First, a Correspondence-Oriented Hierarchical Structure (COHS) organizes anchors by feature correspondence rather than spatial proximity, constructing root-leaf dependencies so that selected anchors can act as informative priors to conditionally encode others, yielding more accurate likelihood prediction and lower conditional entropy. Second, Shared Feature Aggregation (SFA) extracts globally shared features from a contextual hash grid and injects them into anchor representations, factoring out scene-consistent low-frequency information that would otherwise be redundantly embedded in individual anchors. Both modules are trained under a unified rate-distortion objective to balance bitrate reduction and rendering fidelity. Experiments across multiple benchmarks show that CRP-GS achieves a favorable overall rate-distortion trade-off, yielding around 30% average bitrate reduction compared to anchor-based baselines while maintaining comparable rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.22941v1">D3GS: Depth, DINO, and RGB Diffusion Co-Guided 3D Gaussian Splatting for Sparse-View Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-09-19
    </div>
    <details class="paper-abstract">
      Novel view synthesis from sparse inputs remains challenging for 3D Gaussian Splatting (3DGS) due to ambiguous geometry, cross-view inconsistency, and missing details in under-constrained regions, resulting in degraded reconstruction and unstable rendering. To tackle these issues, we propose D$^{3}$GS, a Depth-DINO-Diffusion guided sparse-view Gaussian reconstruction framework that jointly enhances geometry and appearance. D$^{3}$GS first recovers a high-resolution, metric depth map via diffusion-based completion and DPT (Dense Prediction Transformer) refinement, providing robust Gaussian initialization and geometric constraints. Then, a DINO-guided view-consistent learning is introduced to augment Gaussian attributes with structural features, improving multi-view consistency. Finally, a diffusion-based Gaussian refinement module injects generative priors into an iterative optimization strategy, enhancing high-frequency geometric and appearance details within the Gaussian representation. Experiments on DTU, LLFF, and Mip-NeRF 360 show that D$^{3}$GS achieves consistent and substantial improvements over strong baselines, with ablation studies validating the effectiveness and complementary roles of each component.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.22849v1">LINGO: Latent Initialization and Gradient Optimization for Sparse-view X-ray Novel View Synthesis and CT Reconstruction with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-19
    </div>
    <details class="paper-abstract">
      In novel view synthesis and Computed Tomography (CT) reconstruction with sparse-view X-ray imaging, insufficient angular coverage leads to structural ambiguity and accumulated noise. Integrating 3D Gaussian Splatting (3DGS) with X-ray absorption physics can achieve promising results, but it suffers from noisy initialization, positional insensitivity, and weak gradients in low-density regions. In this paper, we propose a unified Latent Initialization and Gradient Optimization (LINGO) framework to address these issues. LINGO combines latent mask-space initialization with dynamic gradient optimization to improve point cloud structural completeness while accelerating training. It constructs voxel-level 3D filters from X-ray masks to robustly suppress background noise and provide reliable geometric priors. By employing an adaptive voxel scaling strategy and dynamically scaling loss, LINGO can adjust spatial resolution and explicitly amplify gradients in low-density structures. To evaluate the quality of initialization, we introduce the Initialization Point Cloud Structural Deviation (IPSD) metric. Experiments on the X3D dataset indicate that for the novel view synthesis task, LINGO improves the Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) by an average of 0.72 and 0.0039, respectively, over baselines under identical sparse-view settings, achieving comparable reconstruction quality within 5k steps to state-of-the-art models typically trained with 30k iterations. For the CT reconstruction task, LINGO also demonstrates consistent improvements, with average PSNR and SSIM gains of 0.36 and 0.0134. These results highlight LINGO's effectiveness in both accelerating training and enhancing reconstruction quality across different sparse-view imaging scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.22809v1">Kinematic Interface for the Wild: Modular Bimanual Loco-Manipulation Capture from 360$^{\circ}$ Cameras Alone</a></div>
    <div class="paper-meta">
      📅 2026-09-19
      | 💬 10 pages, 9 figures
    </div>
    <details class="paper-abstract">
      A wrist-mounted camera for UMI-style data collection must do two jobs: record the manipulation and localize in the scene. Most handheld devices localize online from workspace-facing views crowded by hands and objects, or add dedicated tracking hardware. Room-scale bimanual capture therefore still tends to instrument the operator or the scene for accurate localization. We present KIWI (Kinematic Interface for the Wild), a capture kit whose only electronics are off-the-shelf cameras. Our core system splits the two jobs across the two lenses of a 360-degree camera. The rear lens faces the room and builds a shared metric map that registers both hands, and an optional head camera, in one frame without workspace co-visibility; the front lens records the manipulation, and offline IMU fusion bridges front-lens tracking loss. Through our quick-release plate, the camera module attaches to chopstick grippers, parallel-jaw grippers, hand-wrist mounts, or robot flanges. Across six bimanual recordings, combining the rear and front lenses failed to localize only 0.1% of query frames, whereas front-only bimanual feature alignment failed on 24.8% of frames and lost one recording entirely; against evaluation fiducials, localization error stayed within 4.5 mm. KIWI's recovered poses were sufficiently consistent for the four wrist streams alone to reconstruct the scene as a 3D Gaussian splat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28431v4">LG-HCC: Local Geometry-Aware Hierarchical Context Compression for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-19
      | 💬 10
    </div>
    <details class="paper-abstract">
      Although 3D Gaussian Splatting (3DGS) enables high-fidelity real-time rendering, its prohibitive storage overhead severely hinders practical deployment. Recent anchor-based 3DGS compression schemes reduce gaussian redundancy through some advanced context models. However, they overlook explicit geometric dependencies, leading to structural degradation and suboptimal ratedistortion performance. In this paper, we propose a Local Geometry-aware Hierarchical Context Compression framework for 3DGS(LG-HCC) that incorporates inter-anchor geometric correlations into anchor pruning and entropy coding for compact representation. Specifically, we introduce an Neighborhood-Aware Anchor Pruning (NAAP) strategy, which evaluates anchor importance via weighted neighborhood feature aggregation and then merges low-contribution anchors into salient neighbors, yielding a compact yet geometry-consistent anchor set. Moreover, we further develop a hierarchical entropy coding scheme, in which coarse-to-fine priors are exploited through a lightweight Geometry-Guided Convolution(GG-Conv) operator to enable spatially adaptive context modeling and rate-distortion optimization. Extensive experiments show that LG-HCC effectively alleviates structural preservation issues,achieving superior geometric integrity and rendering fidelity while reducing storage by up to 30.85x compared to the Scaffold-GS baseline on the Mip-NeRF360 dataset
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00195v1">GS-PQM: A Parameter-Domain Quality Metric for Compressed Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-18
    </div>
    <details class="paper-abstract">
      Recent advances in Gaussian Splatting (GS) compression have enabled substantial reductions in GS model size. Reliable objective quality assessment is therefore essential for comparing compression methods and guiding the development of more efficient GS codecs. Existing GS quality assessment typically relies on image and video quality metrics, requiring rendering of predefined viewpoints and making the quality estimate dependent on the selected views. This paper introduces GS-PQM, a novel full-reference quality metric for post-training GS compression that operates directly in the GS parameter domain. GS-PQM estimates perceptual quality from a set of parameter-domain distortion errors using a Support Vector Regression model. Experimental results show that GS-PQM outperforms 25 existing image, video, and point-cloud quality metrics in assessing compressed GS content, providing an accurate and computationally efficient alternative to rendering-based quality assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.26880v2">GScomp-QA: A Subjective Dataset for Quality Assessment of Compressed Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-18
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has emerged as an efficient representation for high-quality 3D reconstruction and novel view synthesis. However, its large model size poses challenges for storage and transmission. While several GS compression solutions have been proposed, their perceptual impact remains poorly understood due to the lack of dedicated evaluation datasets. To address this gap, this paper introduces GScomp-QA, a subjective quality assessment dataset for evaluating synthesis quality from compressed GS models. The dataset comprises 331 video stimuli from 13 real-world scenes, covering 9 state-of-the-art GS compression solutions. By using videos synthesized from uncompressed models as reference, GScomp-QA isolates compression-induced distortions from synthesis artifacts. A subjective study with 20 participants was conducted, providing reliable perceptual scores. Based on these data, GS compression solutions are evaluated through perceptual rate-distortion analysis. In addition, 18 objective quality metrics are evaluated, showing that they do not fully capture GS-specific distortions. GScomp-QA will be publicly available and provide a benchmark for evaluating GS compression solutions and supporting the development of quality metrics tailored to GS compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.21516v1">2D GauSS-MI: Efficient Active Scene Reconstruction with Balanced Visual and Geometric Quality</a></div>
    <div class="paper-meta">
      📅 2026-09-18
    </div>
    <details class="paper-abstract">
      Active reconstruction requires efficient active view selection to achieve high-quality reconstruction within limited onboard computational resources. Existing methods face challenges in adequately balancing visual and geometric quality with the computational efficiency required for real-time operation. In this work, we present an active reconstruction framework based on 2D Gaussian Splatting (2DGS). We develop an efficient online 2DGS mapping pipeline for incremental RGB-D observations and introduce a probabilistic reliability model that characterizes the view-dependent reconstruction quality of individual 2D Gaussian splats. Building on this model, we formulate 2D Gaussian Splatting Shannon Mutual Information (2D GauSS-MI), a mutual-information-based metric that exploits the explicit surface orientation of 2DGS to evaluate the expected information gain of candidate views. The proposed metric enables active view selection to account for both visual and geometric reconstruction quality. We evaluate the proposed system against three state-of-the-art baselines on eight Replica scenes. Experimental results demonstrate that our method achieves a favorable balance between visual and geometric reconstruction quality with substantially lower computational cost and competitive model storage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.21498v1">VoxelTTO: Voxel-Aligned Feed-Forward 3D Gaussian Splatting with Test-Time Optimization</a></div>
    <div class="paper-meta">
      📅 2026-09-18
    </div>
    <details class="paper-abstract">
      Recent feed-forward 3D Gaussian Splatting (3DGS) methods typically regress pixel-aligned Gaussian primitives, often causing excessive overlap and artifacts, while inaccuracies in predicted camera poses can lead to misalignment in novel-view synthesis (NVS). We present VoxelTTO, a feed-forward framework for reconstructing geometrically accurate 3DGS scenes from an arbitrary number of images and optional camera parameters. VoxelTTO aggregates dense image features into a global voxel representation and decodes Gaussians from voxel features, breaking the pixel-to-Gaussian correspondence. To exploit known camera parameters while keeping the pretrained visual foundation model (VFM) parameters frozen, we introduce test-time optimization (TTO) that adapts lightweight LoRA modules using pose supervision. We further replace vanilla 3DGS rasterization with stochastic solid volume rendering during training and inference, improving geometric fidelity. Training updates only the voxel-aligned Gaussian reconstruction modules, requiring 80 GPU hours. Experiments on Replica, Tanks and Temples, and DTU demonstrate improved RGB-D NVS and camera-pose estimation relative to prior methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.21347v1">Cube-Splat: High-Fidelity 360° Gaussian Splatting SLAM via Cubemap Factorization and Adjoint-Consistent Optimization</a></div>
    <div class="paper-meta">
      📅 2026-09-18
      | 💬 Accepted to ECCV 2026. Source code : https://github.com/guoxf304/CubeSplat
    </div>
    <details class="paper-abstract">
      Recent progress in 3D Gaussian Splatting (3DGS) has enabled dense visual SLAM with pinhole cameras, yet most pipelines are not designed for panoramic imagery. We present Cube-Splat, the first panoramic GS-SLAM framework that factorizes each 360° frame into a cubemap of four fixed-orientation virtual pinhole views sharing a single optical center. By designating the front face as the primary pose state, we accumulate gradients from all faces via an adjoint mapping, thereby enabling multi-face observations to coherently update a single state while strictly preserving cross-view geometric consistency. Concurrently, our mapping module densifies and optimizes anisotropic Gaussians using aggregated cubemap rays for high-fidelity, dense reconstruction. Furthermore, to rigorously evaluate panoramic SLAM under diverse and challenging conditions, we introduce SynPano, a highly scalable, photorealistic synthetic dataset featuring parameterized complex trajectories and multi-modal ground truth. Extensive evaluations on two public benchmarks (PALVIO and OmniBlender) and our SynPano dataset, collectively encompassing both indoor and outdoor scenes, demonstrate that Cube-Splat achieves state-of-the-art (SOTA) performance in tracking accuracy and reconstruction fidelity. Both the source code and the SynPano dataset are available at https://github.com/guoxf304/CubeSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.21226v1">AirSplan: Risk-Aware Motion Planning for Quadrotors in Cluttered 3D Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2026-09-18
      | 💬 To appear in the proceedings of IROS 2026
    </div>
    <details class="paper-abstract">
      Quadrotors are increasingly deployed in applications such as agriculture, infrastructure inspection, and maintenance. In each of these applications, the robot must navigate complex scene geometry while remaining strictly collision-free. Unlike in ground domains, even minor collisions for aerial vehicles can result in the loss of the robot. This safety requirement induces a pair of technical challenges. First, the environment must be represented with sufficient fidelity to encode complex structure, even when no ground-truth obstacle data is available. Second, a motion planner must leverage this representation to determine a collision-free path to the goal. This paper proposes a system that addresses these complementary challenges. The proposed method, AirSplan, adopts a normalized variant of 3D Gaussian Splatting that encodes high-fidelity scene geometry. It then applies a novel reachability-based motion planner that leverages the differential flatness of quadrotors to compute continuous-time collision constraints that tightly overapproximate the robot's occupancy. Experiments demonstrate that AirSplan successfully finds a path in 81.2% of challenging test cases, a significant improvement over the nearest baseline method's 51.2%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.21112v1">Demonstration Synthesis from a Single Scan via Gaussian Splatting for Visuomotor Policy Learning</a></div>
    <div class="paper-meta">
      📅 2026-09-17
    </div>
    <details class="paper-abstract">
      Training a visuomotor policy calls for abundant demonstrations that closely match the target environment, yet collecting them anew remains expensive. Existing demonstration synthesis methods reduce this cost but remain constrained by high manual effort, limited visual fidelity, or heavy reliance on physics simulators. This paper introduces GaussianFactory, a high-fidelity data engine that mass-produces demonstrations with a single video scan as its only human input and no physics engine in the generation loop. Specifically, GaussianFactory reconstructs the scene as an editable 3D Gaussian Splatting (3DGS) replica and samples from the object-combination tasks the scene affords. For each task, it plans grasps and trajectories purely kinematically on the geometric reconstruction, rendering photorealistic demonstrations that visually match the target environment. Physical dynamics enter the pipeline only where contact force interactions dictate the outcome---during grasp formation, via a learned contact model pretrained once on an interaction dataset. To evaluate the downstream utility of the synthesized demonstrations, we implement an end-to-end scan-to-deployment workflow in two setups: a simulated scene that stands in for the real world to enable reproducibility, and a real-world workspace with a physical UR10e robot. In each setup, a standard diffusion policy trained solely on the synthesized demonstrations achieves 95.1% and 84.2% success rates, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.20818v1">SplashSplat: Reconstructing Splashing Liquids from Real-World Multi-View Videos</a></div>
    <div class="paper-meta">
      📅 2026-09-17
      | 💬 18 pages (11 main + 7 supplementary), 14 figures, 12 tables. Project page: https://niko-creater.github.io/splashsplat-web/
    </div>
    <details class="paper-abstract">
      A splash lives for a fraction of a second: sheets tear into ligaments and droplets, appearance is view-dependent and nearly textureless, and little persists long enough to track. Reconstruction research has consequently focused on smoke, synthetic liquids, or gently deforming surfaces. To our knowledge, no synchronized multi-view dataset of splashing liquids exists. We therefore introduce a benchmark of 20 real scenes, from coherent streams to violent splashes, captured by seven synchronized, calibrated 4K cameras at 60 fps, with manually refined per-view liquid and container masks and fixed evaluation splits. We further present SplashSplat, built on a single principle: impose physical structure only where the observations can constrain it. Per-frame liquid SDFs fused from the masks provide the geometry, level-set transport between consecutive SDFs yields a coarse velocity field, and Lagrangian carriers advected along this flow, corrected against each new observation and reseeded where coverage is lost, decode local Gaussians for differentiable rendering. SplashSplat outperforms state-of-the-art dynamic Gaussian splatting methods on our real captures and on a synthetic benchmark, with physically more plausible motion and a lower training cost. The same representation supports temporal interpolation and style transfer without re-optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.20623v1">PhGS: Post-Hoc Pruning and Refinement of Single-View Feed-Forward 3D Gaussian Reconstructions</a></div>
    <div class="paper-meta">
      📅 2026-09-17
    </div>
    <details class="paper-abstract">
      Recent single-view feed-forward 3D Gaussian Splatting (3DGS) generation predicts a fixed number of Gaussians per camera ray, introducing severe spatial redundancy. Most existing compaction strategies target multi-view setups to exploit cross-view consistency and are incompatible with single-image models. Instead of retraining the base feed-forward network to directly output compact representations, our insight is to keep the base models frozen and apply post-hoc pruning and recurrent refinement to the generated Gaussians. Consequently, we propose a backbone-agnostic compaction pipeline for single-view feed-forward 3DGS that couples an importance-score-based pruning mechanism with a trainable, lightweight recurrent refinement module, which iteratively updates the surviving primitives to restore image quality. Our results demonstrate seamless integration with existing baselines while preserving novel-view rendering fidelity and achieving high memory reduction. Furthermore, our method supports flexible inference-time keep ratios for application needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.20589v1">RawSLAM: Online HDR Gaussian SLAM from Linear Radiance</a></div>
    <div class="paper-meta">
      📅 2026-09-17
    </div>
    <details class="paper-abstract">
      Current dense visual SLAM systems rely almost exclusively on 8-bit tonemapped Low Dynamic Range (LDR) inputs, limiting their robustness in extreme lighting where shadows and highlights trigger tracking drift and mapping collapse. Conversely, existing raw and High Dynamic Range (HDR) reconstruction pipelines operate strictly offline. They depend on Structure-from-Motion preprocessing and are not suited for large inter-frame motion. We present, to the best of our knowledge, the first online Gaussian SLAM framework that tracks and maps directly on single-exposure 16-bit linear HDR imagery. Our method rests on three core components: an architecture-agnostic HDR Gaussian Splatting module featuring an MLP-free logarithmic parameterization of Gaussian color features; a Reinhard range-compressed photometric objective; and structure-guided spatial gradient weighting. Combined, these components allow our approach to outperform a direct HDR adaptation of MonoGS in both trajectory and reconstruction accuracy, while rendering natively in linear scene radiance for post-rendering processing. The same formulation runs unchanged on standard 8-bit inputs, roughly halving the MonoGS baseline error. Furthermore, our HDR Gaussian module transfers seamlessly to SplaTAM, Gaussian SLAM, and DROID-W, eliminating all tracking failures these systems suffer on challenging illumination sequences. To enable this research, we introduce RawSLAM: a dataset of 10 real-world indoor sequences featuring 16-bit RAW imagery, aligned depth, IMU measurements, and external OptiTrack poses. Code and dataset will be made publicly available soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.20586v1">CoRef-GS: Cooperative Referring Gaussian Splatting for Multi-Agent Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2026-09-17
      | 💬 The established benchmark and source code will be publicly released at https://github.com/ruojiruoli17/CoRef-GS.git
    </div>
    <details class="paper-abstract">
      Referring scene understanding for embodied robots requires grounding object- and relation-centric language queries from a designated viewpoint. While a local semantic Gaussian map can support such grounding within one agent's observations, cooperative settings require this ability to remain effective after independently reconstructed maps are aligned and fused. In this setting, the referred target or its contextual landmark may come from another agent's observations, while spatial relations must still be interpreted from the querying robot's viewpoint. We formulate this problem as cooperative referring Gaussian grounding over fused maps, which requires geometric alignability, instance-level semantic comparability, and view-conditioned relation reasoning. Existing language-aware Gaussian methods mainly focus on single-map querying, whereas Gaussian registration methods optimize geometric or photometric alignment without preserving language-grounding-oriented semantic compatibility. We propose CoRef-GS, a cooperative referring Gaussian splatting framework. CoRef-GS constructs local open-vocabulary instance-aware Gaussian maps, then aligns partially overlapping maps with a cross-agent alignment module by geometric and semantic consistency, and grounds queries using a view-conditioned mask relation graph. We further introduce CoQuad-Ref, a dual-quadruped benchmark spanning both real-world and simulated indoor scenes. Experiments show that, on simulated scenes, CoRef-GS reduces the rotation error from 2.58° after coarse initialization to 0.15° after refinement, and improves real-world referring mIoU over ReferSplat from 52.6% to 68.8%. The established benchmark and source code will be publicly released at https://github.com/ruojiruoli17/CoRef-GS.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.20348v1">EliGSiR: Continual RGB-D Mapping with Gaussian Splatting under Bounded Compute</a></div>
    <div class="paper-meta">
      📅 2026-09-17
      | 💬 8 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Conventional 3D Gaussian Splatting assumes a closed set of observations and long optimization schedules. Continual RGB-D mapping in contrast poses the problem that new observations arrive online, while previously reconstructed regions must be preserved. We present EliGSiR (Evidence-guided Load-adaptive Incremental Gaussian Splatting with Image Replay), a continual Gaussian mapper that controls how the available optimization budget is used as the reconstruction evolves. Map-Guided View Scheduling filters redundant incoming views and reconsiders retained views according to the current state of the map. Load-Adaptive Fidelity adjusts supervision resolution to the current mapping load instead of following a fixed resolution schedule. Targeted Geometry Growth separates depth supervision from Gaussian creation and adds geometric capacity only where repeated RGB-D observations indicate missing or misplaced structure. Together, these mechanisms adapt which views are optimized, how much image detail is used, and where the representation grows while mapping remains active. We evaluate EliGSiR on Replica, TUM RGB-D, ScanNet++, and real RGB-D sensor sequences, considering both the final reconstruction and the map available throughout acquisition. On TUM RGB-D fr3/long_office_household, EliGSiR reaches 21.52 dB with the same ground-truth mapping poses used by the controlled baselines, compared with 19.42 dB for SplaTAM. In the tracked-pose comparison, EliGSiR with live ORB-SLAM3 poses reaches 23.02 dB in 155.5 s, compared with 20.10 dB in 230.9 s for CaRtGS using its native tracker. We further evaluate reconstruction throughout acquisition and show how EliGSiR adaptive view scheduling, supervision fidelity, and geometry growth improve the use of the available mapping budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06438v4">AGORA: Adversarial Generation Of Real-time Animatable 3D Gaussian Head Avatars</a></div>
    <div class="paper-meta">
      📅 2026-09-17
      | 💬 ECCV2026 (Interactive Social Avatars Workshop) accepted version
    </div>
    <details class="paper-abstract">
      The generation of high-fidelity, animatable 3D human avatars remains a core challenge in computer graphics and vision, with applications in VR, telepresence, and entertainment. Existing approaches based on implicit representations like NeRFs suffer from slow rendering and dynamic inconsistencies, while 3D Gaussian Splatting (3DGS) methods are typically limited to static head generation, lacking dynamic control. We bridge this gap by introducing AGORA, a novel framework that extends 3DGS within a generative adversarial network to produce animatable avatars. Our key contribution is a lightweight, FLAME-conditioned deformation branch that predicts per-Gaussian residuals, enabling identity-preserving, fine-grained expression control while allowing real-time inference. Identity is further preserved through spatial shape conditioning of the identity branch, and expression fidelity is enforced via a dual-discriminator training scheme leveraging synthetic renderings of the parametric mesh. AGORA generates avatars that are not only visually realistic but also precisely controllable. Quantitatively, we outperform state-of-the-art NeRF-based methods on expression accuracy while rendering at 250 FPS on a single GPU and, notably, at $\sim$9 FPS under CPU-only inference -- to our knowledge the first demonstration of CPU-only animatable 3DGS avatar synthesis. This work represents a significant step toward practical, high-performance digital humans. Project website: https://ramazan793.github.io/AGORA/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.19907v1">GS-PI: An Optimization-Decoupled Appearance Decomposition Approach for Generating PBR Gaussian Assets</a></div>
    <div class="paper-meta">
      📅 2026-09-17
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) excels at novel-view synthesis but encodes baked-in radiance, tightly entangling illumination with geometry and preventing seamless integration into physically based rendering (PBR) pipelines. Existing inverse-rendering methods attempt to disentangle materials via joint optimization, but often suffer from competing objectives that cause severe ambiguities and residual lighting artifacts. To overcome this, we present GS-PI, a novel optimization-decoupled framework that casts PBR material generation as a geometry-conditioned diffusion process on 3D point clouds. By operating directly in the 3D domain, our method inherently guarantees multi-view consistency, sidestepping the severe pixel correspondence issues that challenge 2D diffusion approaches. We introduce a multi-scale cross-view conditioning mechanism that integrates three complementary components: a global semantic prior, source-anchored photometric cues, and an absolute spatial learned view-direction conditioning signal. This design efficiently compresses complex multi-view evidence, mitigating cross-view projection misalignment and successfully preventing specular highlights from baking into intrinsic colors. By extracting a point cloud from a pre-trained Gaussian model, predicting PBR attributes via conditional diffusion, and distilling them back through differentiable rasterisation, we yield a fully relightable PBR-GS asset. GS-PI outperforms recent inverse-rendering baselines while replacing per-scene joint illumination/BRDF optimization with a learned diffusion pass followed by a short target-driven distillation, without requiring proxy meshes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.18783v3">ArtNVG: Content-Style Separated Artistic Neighboring-View Gaussian Stylization</a></div>
    <div class="paper-meta">
      📅 2026-09-17
      | 💬 Accepted at ICMR 2025 Oral
    </div>
    <details class="paper-abstract">
      As demand from the film and gaming industries for 3D scenes with target styles grows, the importance of advanced 3D stylization techniques increases. However, recent methods often struggle to maintain local consistency in color and texture throughout stylized scenes, which is essential for maintaining aesthetic coherence. To solve this problem, this paper introduces ArtNVG, an innovative 3D stylization framework that efficiently generates stylized 3D scenes by leveraging reference style images. Built on 3D Gaussian Splatting (3DGS), ArtNVG achieves rapid optimization and rendering while upholding high reconstruction quality. Our framework realizes high-quality 3D stylization by incorporating two pivotal techniques: Content-Style Separated Control and Attention-based Neighboring-View Alignment. Content-Style Separated Control uses the CSGO model and the Tile ControlNet to decouple the content and style control, reducing risks of information leakage. Concurrently, Attention-based Neighboring-View Alignment ensures consistency of local colors and textures across neighboring views, significantly improving visual quality. Extensive experiments validate that ArtNVG surpasses existing methods, delivering superior results in content preservation, style alignment, and local consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.19716v1">GAPrompt++: Multi-Granular Geometry-Aware Point Cloud Prompt for 3D Vision Model</a></div>
    <div class="paper-meta">
      📅 2026-09-17
      | 💬 Accepted by TPAMI 2026. Code at https://github.com/PKU-OV3-LAB/GAPromptPlus.git
    </div>
    <details class="paper-abstract">
      Pre-trained 3D vision models have substantially advanced point cloud analysis, yet adapting them to downstream tasks via full fine-tuning is computationally expensive and storage-intensive. Parameter-Efficient Fine-Tuning (PEFT) offers a promising alternative by reducing both adaptation cost and storage burden. However, existing prompting-based approaches ignore the intrinsic geometric structures of point clouds, thereby limiting their adaptation capability. This limitation stems from their inability to encode both fine-grained geometric cues and coarse-grained structural semantics, as well as failing to propagate such information effectively through the model hierarchy. To address these challenges, we propose GAPrompt++, a multi-granular geometry-aware prompting method that provides richer geometric guidance for efficient 3D task adaptation. Specifically, we introduce a Point Shift Prompter that extracts multi-granular geometric features across different scales, enabling instance-specific geometric adjustments during adaptation. Next, a Keypoint Prompter adaptively generates point-level prompts to highlight local geometric saliency and fine-grained structural details. Furthermore, a Prompt Propagation mechanism injects these multi-granular geometric cues throughout the feature extraction hierarchy, strengthening the ability to capture essential geometric characteristics. Extensive experiments show that GAPrompt++ achieves state-of-the-art performance among prompting-based PEFT methods and even surpasses full fine-tuning across diverse benchmarks, while requiring less than 2\% trainable parameters. In addition, to address the saturation of existing evaluation datasets, we construct two more challenging benchmarks derived from 3D Gaussian Splatting and Multi-View Stereo reconstruction, offering diverse and realistic point cloud scenarios to promote future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.19628v1">VGGT-GS SLAM: Uncalibrated Monocular Gaussian Splatting SLAM with Feed-Forward Priors</a></div>
    <div class="paper-meta">
      📅 2026-09-17
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      We present VGGT-GS SLAM, a monocular 3D Gaussian Splatting SLAM system designed for uncalibrated videos. Starting from feed-forward VGGT pose and depth priors, our system performs submap differentiable bundle adjustment that jointly refines camera poses and a 3D Gaussian map, while optimizing submap-shared intrinsics and radial--tangential distortion through analytic calibration Jacobians. To improve global consistency, we introduce Gaussian-native alignment (GNA) for camera-anchored scale refinement between sequential submaps and verification of loop-closure candidates. Extensive experiments on standard indoor benchmarks show consistent improvements in localization accuracy and strong rendering quality under uncalibrated settings, establishing a strong baseline for uncalibrated Gaussian SLAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.19533v1">SLAMSqueezeBench: Comparing SLAM Systems under Resource Constraints</a></div>
    <div class="paper-meta">
      📅 2026-09-17
      | 💬 8 pages, 2 figures, 7 tables. This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Simultaneous localization and mapping (SLAM) is one of the services running on an autonomous robot. It is typically run to assist other tasks such as planning, manipulation, etc. All these tasks are run on edge hardware and are subject to severe resource constraints. However, most SLAM systems are built and tested in isolation, and their performance is reported as if they are the only task running on a system. We observe that existing benchmarks lack a common mechanism for comparing SLAM systems under realistic resource constraints. To address this limitation, we have developed SLAMSqueezeBench, a framework that allows testing of SLAM systems under realistic workloads on edge hardware. It does so by imposing constraints on compute and memory resources available for the SLAM system during execution. It also simulates realistic camera frame acquisition with frame drops when a finite buffer is full. Using SLAMSqueezeBench, we compare nine SLAM systems spanning classical systems, learning-based systems, and approaches for Gaussian splatting. Our testing framework will be available for use by the community upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.19463v1">ParticleSplat: Self-supervised Object-centric Latent Particle Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-16
      | 💬 Project page: https://lyuxinghe.github.io/ParticleSplat-website/
    </div>
    <details class="paper-abstract">
      We present ParticleSplat, a self-supervised object-centric learning method that decomposes scenes into a set of latent ''particles'' representing semantic entities through feedforward 3D Gaussian Splatting. Building on the Deep Latent Particles (DLP) framework, which represents images as a set of particles with attributes such as position, scale, and visual appearance, we address a key limitation of DLP: its inherently 2D nature, which prevents explicit 3D spatial and geometric reasoning that are critical for downstream tasks such as robotic manipulation. Leveraging the structural similarity between latent particles and 3D Gaussian primitives, we introduce a 3D latent particle space trained with a novel view synthesis objective. Our model jointly encodes multiple views with camera poses into a shared 3D object-centric latent space, then transforms particles into particle-aligned 3D Gaussians whose composition reconstructs the full scene. On simulated and real-world datasets, we show that this formulation inherently learns object masks without supervision and supports controllable 3D scene editing, such as moving objects by modifying particles in the latent space. We further establish that the learned 3D representation improves downstream performance on robotic manipulation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.19421v1">RGS: Reflection-aware Gaussian Splatting via Learning Geometry Continuity for Reflective Objects</a></div>
    <div class="paper-meta">
      📅 2026-09-16
      | 💬 Project Page: https://xiaobiaodu.github.io/reflectivegs/ Published in ICRA2026
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has significantly improved the quality of novel view synthesis with explicit Gaussian representation. However, we observed that existing 3D Gaussian Splatting methods (3DGS) often suffer from surface collapse issues on reflective regions, and thus produce inferior geometry and low-quality specular. In this work, we propose a physically-based deferred rendering framework, named Reflection-aware Gaussian Splatting (RGS), that can accurately model specular regions and improve novel view synthesis performance. Specifically, we found that a powerful 3D foundation model can provide a strong 3D geometric prior to foster correct geometric modeling. Based on this, we propose a cross-view shape consistency regularization to regularize the geometry surface with the large model prior and cross-view constraints. In this manner, our RGS can produce smoother geometric surfaces on reflective regions while reducing geometric hollows. To further improve rendering results on reflective regions, we present a reflection-aware densification strategy that is designed to capture specular variations across various views. With this strategy, our RGS is able to render novel views of objects in higher quality. Extensive experiments demonstrate our method consistently renders high-quality reflective objects, achieving state-of-the-art performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.19330v1">SemSafe-3DGS: Semantic Risk-Aware Active Navigation in Uncertain 3D Gaussian Splatting Maps</a></div>
    <div class="paper-meta">
      📅 2026-09-16
      | 💬 Accepted to be presented at the SeMaNa 2026 Workshop at IROS 2026
    </div>
    <details class="paper-abstract">
      Autonomous robots operating in partially observed environments must navigate safely while acquiring observations that improve future planning. Existing safety formulations generally reason primarily about geometry. Consequently, geometrically similar scene elements may induce comparable control responses despite having different semantic consequences. We present a semantic risk aware safe-active perception framework for navigation in attributed 3D Gaussian maps. Semantic attributes modulate an Average Value-at-Risk collision clearance model through class dependent risk weights, allowing safety-critical Gaussian primitives to receive greater influence in the composite barrier. The resulting weighted clearances are aggregated into a control barrier function, while a trajectory-relevant active perception barrier promotes observations that reduce geometric map uncertainty along the robot's anticipated motion. Both objectives are integrated in a unified CBF-QP that enforces semantic risk-aware collision avoidance as a hard constraint while relaxing information acquisition when it conflicts with safety or task progress. Experiments demonstrate efficient safety constraint, improved navigation through active perception, semantic dependent trajectory adaptation, and real-robot execution under Ackermann dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.18898v1">NormLift: From Lifted Features To Semantic Reliability In 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-16
      | 💬 20 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Training-free weighted aggregation is widely used to lift 2D semantic features onto 3D Gaussians for open-vocabulary scene understanding, yet its theoretical role remains insufficiently understood. Existing analyses typically justify this operation from the rendering side, treating Gaussian features as linearly composable Euclidean variables for reconstructing 2D feature maps. However, this view does not match downstream 3D usage, where each Gaussian is often queried independently in a cosine-based embedding space. We revisit feature lifting from the 3D side and formulate per-Gaussian assignment as a cosine alignment problem on the CLIP unit sphere. Under this objective, the L2-normalized semantic back-projected feature emerges as the closed-form solution, providing a complementary interpretation of the standard lifting rule from the perspective of per-Gaussian semantic assignment. The same formulation further yields a norm decomposition into intra-view and inter-view consistency, suggesting that feature magnitude itself can serve as a semantic reliability signal. Calibrated by effective multi-view support, this reliability score guides a mode-voting refinement that preserves CLIP feature validity by avoiding linear averaging. Experiments on open-vocabulary 3D semantic segmentation show that NormLift is an efficient, training-free framework that achieves strong performance across evaluation protocols.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.18624v1">MoQSplat: Adaptive Progressive Streaming of 3D Gaussian Splatting via MoQ</a></div>
    <div class="paper-meta">
      📅 2026-09-16
      | 💬 7 pages. Accepted at IEEE MMSP 2026 (Istanbul, 22-24 September 2026). First three authors contributed equally. Code: https://github.com/emanuele-artioli/MoQSplat
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables photorealistic novel view synthesis, but transmitting gigabyte-scale scene data remains challenging for immersive applications. Traditional HTTP Adaptive Streaming over TCP introduces Head-of-Line (HOL) blocking and coarse segmenting ill-suited to fine-grained 3DGS delivery. We propose MoQSplat, which maps 3DGS content onto the Media over QUIC (MoQ) transport hierarchy. MoQSplat partitions scenes into spatial Tracks, clusters splats into semantically coherent Groups, and constructs progressive-quality Subgroups mapped to independent QUIC streams to eliminate connection-level HOL blocking. Using a stateless, subscriber-driven adaptation loop, clients dynamically request spatial regions and quality tiers based on six degrees of freedom (6-DoF) frustum visibility, distance, and foveal alignment. We evaluate the core components on a prototype implementation, showing that opacity-based pruning outperforms scale-based pruning for progressive delivery. The source code is available at https://github.com/emanuele-artioli/MoQSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.05952v2">Generalizable Neural Reconstruction of High-Fidelity Surfaces via Sparse Volumetric Representations</a></div>
    <div class="paper-meta">
      📅 2026-09-16
    </div>
    <details class="paper-abstract">
      Neural implicit representations have recently achieved impressive results in novel view synthesis and multi-view 3D reconstruction, yet both NeRF- and Gaussian Splatting-based methods require per-scene optimization, which makes them inefficient. Generalizable Neural Surface Reconstruction (GNSR) methods have been proposed to remove this need by learning feature representations directly predicted from input images. However, their typical reliance on dense feature volumes severely limits achievable resolution and fidelity due to prohibitive memory costs. We introduce Sparse Volumetric Reconstruction (SVRecon), a new GNSR framework that unlocks high-resolution, memory-efficient reconstruction through learned occupancy-driven sparsity, in a more effective way than earlier approaches to introducing sparsity in GNSRs. Our approach uses a nested two-stage architecture: (1) an occupancy prediction network that identifies surface-containing voxels, and (2) a high-resolution sparse volume rendering framework defined only within these occupied regions, together with specialized sparsified algorithms for ray sampling, feature aggregation, and querying. This design enables fine-grained surface reconstruction while avoiding the heavy memory footprint of dense grids. SVRecon operates at resolutions up to $512^3$ on standard 32GB hardware---substantially higher than prior generalizable methods---and delivers smoother and more precise reconstructions across diverse datasets, particularly in sparse-view settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31679v2">Practical High-Fidelity Novel-View Synthesis of Mounted Lepidoptera</a></div>
    <div class="paper-meta">
      📅 2026-09-16
    </div>
    <details class="paper-abstract">
      Mounted butterflies are among the most striking objects in natural history collections. However, their beauty is notoriously hard to digitize in 3D: they are small and fragile, with microscopic hairs and vein structures. Capturing them in sufficient detail, therefore, requires a macro lens, which has a very limited Depth of Field (DoF). Moreover, a camera body cannot be maneuvered beneath a pinned specimen to photograph its ventral surface. We introduce an end-to-end pipeline that resolves these challenges, turning such specimens into photorealistic 3D models viewable from every direction. It combines three ingredients: handheld focus stacking for all-in-focus macro capture without a tripod, a non-contact first-surface mirror system that exposes the ventral surface without touching the specimen, and a segmentation-free, mirror-aware 3D Gaussian Splatting extension. We validate the reconstructions and design decisions on nine diverse specimens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05598v2">SSA-3DGS: Unsupervised Removal of Screen-Space Artifacts for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-16
    </div>
    <details class="paper-abstract">
      Novel View Synthesis (NVS) methods, such as 3D Gaussian Splatting (3DGS), rely on the assumption of clean, multi-view consistent, posed input images. Real-world captures can violate this assumption due to \textbf{screen-space artifacts}---static occlusions fixed to the 2D image plane rather than to the 3D world. Common examples include physical sensor defects, environmental obstructions (such as rain or mud on the lens enclosure), capture obstructions (such as a thumb over the camera sensor or a dashboard visible in dashcam footage), and digital overlays (such as watermarks or UI elements). When present, they are erroneously baked into the 3D geometry as ``floaters'' or near-camera artifacts, degrading the quality of novel-view rendering. In this work, we propose \textit{SSA-3DGS}, an unsupervised framework that jointly optimizes a 3D scene and a learnable 2D overlay to recover a clean 3D scene and the corrupting artifacts. By exploiting geometric consensus across views, our method effectively disentangles static artifacts from the 3D scene geometry without supervision or manual input. Across diverse synthetic corruptions and a self-captured real-world dataset, SSA-3DGS improves reconstruction fidelity by up to ${\sim}8$~dB PSNR over 3DGS trained on the same corrupted inputs, while faithfully preserving the corrupting artifact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.18473v1">CADSplat: Sparse-View 3D Gaussian Splatting Aided by CAD Models for Robust, Photorealistic Digital-Twin Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-09-16
    </div>
    <details class="paper-abstract">
      We present CADSplat, a framework that reconstructs photorealistic, geometrically accurate digital twins from sparse ($<15$ views), wide-baseline posed images of an object by regularizing 3D Gaussian Splatting (3DGS) with an explicit CAD shape prior. Using such a prior requires finding a CAD model whose shape resembles the object depicted in the images and determining the pose of each camera relative to the object. We obtain both by matching segmented object silhouettes against silhouettes rendered from a CAD library and keeping the camera-to-object poses of the best-matching model. We then anchor 3D Gaussian primitives to the surface of the retrieved model and jointly optimize the 3DGS parameters, the camera-to-object registration, and a non-rigid deformation field to account for shape differences between the physical object and the CAD model. Across two real-world datasets, CADSplat outperforms unconstrained, few-shot, and mesh-texturing baselines and degrades gracefully to as few as 3 views. Our experiments show that most of the gain in rendering quality comes from how the splats are constrained---a fixed set of splats tied to a surface and moved by a single smooth deformation field---rather than from the CAD shape itself. The CAD model adds shape knowledge where views are scarcest, in the sparsest captures and on strongly self-occluded objects, and it places every camera in the object's own frame. This enables applications beyond novel-view synthesis, such as markerless augmented reality registration, per-image object pose estimation, physical simulations, and the transfer of part labels from the design to the reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.17810v1">Wind on Trees: Testing Physical Grounding in Dynamic 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-15
      | 💬 Workshop Version of ECCV 2026 3DWM Workshop
    </div>
    <details class="paper-abstract">
      Monocular reconstruction of wind-driven vegetation is severely underconstrained: motion along the viewing direction is largely unobservable, a moving canopy offers few reliable correspondences, and nearly the entire scene is dynamic, providing little static reference. Directly-learned deformation fields in 4D Gaussian Splatting therefore optimize photometric consistency rather than recover the motion that produced it. We replace that field with a physically parameterized deformation prior: one damped harmonic oscillator per rigid part, driven by the observed wind and integrated by differentiable RK4, supervised photometrically alone. To test whether such a prior is physically grounded rather than merely well fit, we build a controlled synthetic testbed of three procedurally generated trees spanning an order of magnitude in skeleton complexity, whose per-part natural frequency follows from its own geometry and whose damping ratio is a fixed constant, both held out of training. On it, we measure held-out views, temporal extrapolation, zero-shot transfer to unseen wind speeds, and recovery of the physical parameters themselves. The prior costs appearance fidelity on in-distribution views and extrapolates markedly better outside the training window and the training wind, while parameter recovery is far weaker than it first appears: frequency recovery survives an untrained null control on only the sparsest of the three trees, and damping is not recovered at all.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.17387v1">PanoGS-SLAM: Panoramic 3D Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2026-09-15
    </div>
    <details class="paper-abstract">
      Real-time dense SLAM is a core capability for robotics applications that require robust localization and high- quality mapping in dynamic or fast-changing environments. Recent 3D Gaussian Splatting (3DGS)-based SLAM methods have shown promising performance, but most are designed for narrow-FoV pinhole cameras, where limited angular coverage weakens pose observability and often leads to unstable photo- metric optimization under rapid motion and large viewpoint changes. We present PanoGS-SLAM, the first panoramic dense SLAM system built on 3D Gaussian Splatting. Our method per- forms differentiable rendering and pose optimization directly in the spherical domain, enabling omnidirectional photometric constraints for more stable tracking. To improve geometric consistency and robustness, we introduce (1) a sphere-consistent photometric loss that compensates for the area distortion of equirectangular projection, and (2) a depth-guided Gaussian initialization strategy that stabilizes incremental mapping in newly observed regions. Extensive experiments on both real and synthetic panoramic benchmarks (PALVIO and SynPano) show that PanoGS-SLAM consistently outperforms geometric and GS-based baselines in tracking accuracy and rendering quality, while achieving fast front-end convergence and real-time perfor- mance. In addition, controlled field-of-view experiments reveal a clear monotonic improvement in optimization conditioning and convergence stability as angular coverage increases, high- lighting the fundamental role of sensing geometry in shaping the optimization landscape of differentiable Gaussian-based SLAM. The source code will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16324v2">LaGSplat: Inferring Physics-Governed Interactive Simulation from Monocular Video Using Latent Lagrangian Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-15
      | 💬 25 pages, 11 figures, 4 tables. Project page with interactive demo: https://louenpottier.github.io/lagsplat.html
    </div>
    <details class="paper-abstract">
      We present LaGSplat (Latent Lagrangian Gaussian Splatting), a framework that infers interactive, physics-governed dynamics from one or a few monocular videos. At inference it lets a user push on the filmed object, rigid or deformable, with an external force that was never measured, annotated, or seen during training. This is possible because a low-dimensional latent state $\mathbf{q} \in \mathbb{R}^d$ plays two roles at once: it is the generalised coordinate of a learned dissipative Lagrangian and the conditioning variable of a Gaussian Splatting decoder. The inductive bias of this decoder, whose primitives are explicit points $μ_i(\mathbf{q})$ that move with the object, is what lets a force $f$ applied in the image pull back into a latent generalised force $J(\mathbf{q})^\top f$ and enter the equations of motion, which pixel-space (CNN) or neural-field (NeRF) decoders cannot do. We validate LaGSplat on test cases of increasing difficulty, from rigid to deformable and from autonomous to forced real systems, combining monocular video and sensor measurements. We further demonstrate interactive use: forces of arbitrary magnitude and direction can be applied to the reconstructed object at any time, its response rendered in real time, in 2D or 3D. Assuming a dissipative Euler-Lagrange equation over a few generalised coordinates trades generality for a bounded, plausible response to unseen forces, where an unconstrained predictor diverges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.17106v1">BRAVE-6D: Benchmark for Robotic Active Vision in 6DOF Pose Estimation</a></div>
    <div class="paper-meta">
      📅 2026-09-15
      | 💬 3 pages, 2 figures. Extended abstract presented at ICRA@40, Rotterdam, The Netherlands, September 2024
    </div>
    <details class="paper-abstract">
      Detecting and grasping small objects remains a significant challenge in robotics. Active vision, where the robot moves closer to the object, is an intuitive solution, yet comparing approaches on common ground is difficult since identical physical scene setups are required. Hence, we introduce BRAVE-6D, a benchmark designed to evaluate robotic active vision systems for object pose estimation, a crucial first step in grasping objects. BRAVE-6D leverages view synthesis based on Gaussian Splats (3DGS) to provide scenes and tools for benchmarking active vision systems. We show baseline solutions performing visual servoing within the scene and accurately estimating the poses of small objects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.17039v1">Bi-FlowGS: Bridging Generative View Completion and Gaussian Geometry through Bidirectional Flow Co-Refinement</a></div>
    <div class="paper-meta">
      📅 2026-09-15
    </div>
    <details class="paper-abstract">
      Sparse-view 3D scene reconstruction with 3D Gaussian Splatting (3DGS) is inherently underconstrained. Plausible renderings can also coexist with erroneous Gaussian geometry, as errors in positions or depths may be concealed by opacity, scale, and appearance; we term this failure mode Geometry Cheating. Existing regularization methods constrain geometry but remain limited to observed views, while video-diffusion-based methods complete unseen views yet mainly use them as RGB pseudo-supervision, underusing motion and temporal priors and lacking explicit geometry supervision. We present Bi-FlowGS, which uses optical flow to bridge generative view completion and Gaussian geometry regularization. Our plug-and-play Video-to-Geometry Flow Distillation (V2G) distills temporal correspondence priors from restored videos into Gaussian geometry to alleviate Geometry Cheating. Conversely, Geometry-to-Video Flow-Guided Restoration (G2V) uses the current 3DGS geometry to guide temporally consistent video restoration, providing more reliable generative supervision. Together, V2G and G2V form an implicit bidirectional co-refinement process, enabling restored videos and the optimized 3DGS scene to iteratively improve each other. Experiments demonstrate improved rendering quality and geometric consistency across wide-baseline and unbounded 360° benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21265v2">Unlocking Zero-shot Potential of Semi-dense Image Matching via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-15
      | 💬 8 pages, 7 figures. Accepted to the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026)
    </div>
    <details class="paper-abstract">
      Learning-based image matching critically depends on large-scale, diverse, and geometrically accurate training data. 3D Gaussian Splatting (3DGS) enables photorealistic novel-view synthesis and thus is attractive for data generation. However, its geometric inaccuracies and biased depth rendering currently prevent robust correspondence labeling. To address this, we introduce MatchGS, the first framework designed to systematically correct and leverage 3DGS for robust, zero-shot image matching. Our approach is twofold: (1) a geometrically-faithful data generation pipeline that refines 3DGS geometry to produce highly precise correspondence labels, enabling the synthesis of a vast and diverse range of viewpoints without compromising rendering fidelity; and (2) a 2D-3D representation alignment strategy that infuses 3DGS' explicit 3D knowledge into the 2D matcher, guiding 2D semi-dense matchers to learn viewpoint-invariant 3D representations. Our generated ground-truth correspondences reduce the epipolar error by up to 40 times compared to existing datasets, enable supervision under extreme viewpoint changes, and provide self-supervisory signals through Gaussian attributes. Consequently, state-of-the-art matchers trained solely on our data achieve significant zero-shot performance gains on public benchmarks, with improvements of up to 17.7%. Our work demonstrates that with proper geometric refinement, 3DGS can serve as a scalable, high-fidelity, and structurally-rich data source, paving the way for a new generation of robust zero-shot image matchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.16443v1">The Neverwhere Visual Parkour Benchmark Suite</a></div>
    <div class="paper-meta">
      📅 2026-09-14
      | 💬 9 pages, 14 figures. Accepted to IROS 2026. Project page: https://ziyc.github.io/neverwhere-bench/
    </div>
    <details class="paper-abstract">
      State-of-the-art visual locomotion controllers are increasingly capable at handling complex visual environments, making evaluating their real-world performance before deployment increasingly difficult. This work intends to narrow this train/evaluation gap by developing a collection of hyper-photo-realistic, closed-loop evaluation environments - The Neverwhere Benchmark Suite - comprised of over sixty 3D Gaussian Splatting reconstructions of urban indoor and outdoor scenes. Our goal is to encourage large-scale and reproducible robot evaluation by making it easier to create and integrate Gaussian splats-based reconstructions into simulated continuous testing setups. We also underscore the potential pitfalls of relying exclusively on 3D Gaussian-generated data for training, by providing policy checkpoints trained over multiple Neverwhere scenes and their performance when evaluated in novel scenes. Our analysis illustrates the necessity of sourcing diverse data to ensure performance. Code and data are available on the project page: https://ziyc.github.io/neverwhere-bench/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.16310v1">Racing in Volume with Flow Ensembles</a></div>
    <div class="paper-meta">
      📅 2026-09-14
      | 💬 project page at https://humansensinglab.github.io/monaco4d/
    </div>
    <details class="paper-abstract">
      Streaming 4D reconstruction has been demonstrated only indoors, on dense camera rigs surrounding subjects that move at human pace. Outdoor 4D reconstruction exists but relies either on cameras mounted on the moving vehicle itself, or on limited-coverage arrays observing quasi-static subjects offline. The case that actually matters for spectators is a fast-moving subject, watched from a sparse ring of allocentric cameras, streaming. No method targets this, and no benchmark exists to evaluate one. To this end, we introduce FastFlowGS, a streaming 4D Gaussian Splatting method for reconstructing fast-moving subjects from a small set of fixed external cameras, and Monaco4D, a photorealistic Unreal Engine 5 benchmark for high-speed outdoor reconstruction. FastFlowGS fuses sparse matches, semi-dense tracks, and dense optical flow by lifting each signal to 3D with geometric uncertainty and combining them through a Kalman-style temporal update. Monaco4D provides Formula 1 sequences under varied illumination from trackside, onboard, and drone viewpoints with dense ground truth. On CMU-Panoptic, FastFlowGS exceeds the strongest baseline by 12.6% VMAF at 35% greater efficiency. On Monaco4D, where existing streaming methods degrade severely, it improves dynamic-region PSNR by up to 18.6% with 28.3% lower per-frame optimization time. Dataset and additional details can be found at https://humansensinglab.github.io/monaco4d/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.16233v1">SceneBench: A Hierarchical Benchmark for Vision-Language Understanding of 3D Scenes</a></div>
    <div class="paper-meta">
      📅 2026-09-14
    </div>
    <details class="paper-abstract">
      Vision-language models excel at 2D image understanding but remain limited in 3D spatial reasoning. Progress is hindered by limitations in current benchmarks. First, 3D datasets often rely on point clouds that capture geometry but discard rich visual features like texture, text, and materials. Second, annotations treat objects in isolation while ignoring real-world hierarchical organization (scenes, rooms, functional areas, object groups). Third, evaluation tasks focus narrowly on basic recognition rather than multi-step spatial reasoning. In this context, we introduce SceneBench, a benchmark of 966 photorealistic 3D scenes reconstructed with Gaussian Splatting and densely annotated with hierarchical semantics spanning scenes, rooms, functional areas, object groups, and individual objects. These annotations are produced through a human-in-the-loop pipeline combining vision-language models with roughly 1,500 human-hours of iterative refinement and verification, producing over 183K annotated nodes with textual descriptions and 3D bounding boxes. Building on this representation, we define three evaluation tasks: Existence-Based Questions probing object attributes, Spatial Intelligence Questions covering counting, size comparison, distance, and directional relations, and Grounded Question-Reasoning-Answer (QRA) triplets requiring multi-step reasoning across semantic levels. Experiments with state-of-the-art vision-language models show that while models perform well on basic recognition tasks (e.g., up to 85% accuracy for detection), performance drops substantially on hierarchical and compositional reasoning (e.g., down to 60% for counting), revealing limitations not captured by existing benchmarks. SceneBench provides a realistic testbed for developing and evaluating models capable of fine-grained spatial reasoning in photorealistic 3D environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.14899v1">What Makes a 3D Scene Editable? A Factorized Benchmark of Fidelity, Locality, Consistency, and Preservation</a></div>
    <div class="paper-meta">
      📅 2026-09-14
    </div>
    <details class="paper-abstract">
      Neural 3D scene editing is often evaluated by semantic alignment alone, although a convincing result may alter unrelated content or become inconsistent across views. We introduce EditBench3D, a representation-agnostic benchmark that treats editing as controlled information replacement. It evaluates four complementary properties: instruction fidelity, spatial locality, cross-view consistency, and preservation of non-target content. The protocol combines visibility-aware 3D target supports, paired descriptions, held-out cameras, and five edit families covering appearance, material, geometry, and object-level changes. We evaluate eight representative NeRF, 3D Gaussian Splatting, hybrid, and proxy-based editors on 240 scene-edit pairs. The study shows that semantic fidelity is only weakly associated with the other editing properties, and that no single method is optimal across all dimensions. Explicit Gaussian editors offer a strong overall balance, whereas direct proxy manipulation provides the most conservative edits at the cost of open-ended fidelity. These findings support reporting editability as a multi-objective profile rather than a single semantic score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.14634v1">SCOUT-SLAM: Structurally-Coupled Dual Uncertainty-Aware 3DGS SLAM in the Wild</a></div>
    <div class="paper-meta">
      📅 2026-09-13
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting SLAM (3DGS-SLAM) has gained significant momentum in simultaneous localization and 3DGS scene reconstruction. In real-world scenarios with rapid camera motion and cluttered dynamic environments, existing methods rely on the stability of the underlying scene reconstruction to model uncertainty. This leads to a circular dependency between camera tracking accuracy and reconstruction quality: reconstruction instabilities degrade uncertainty modeling, which affects accurate camera tracking and static scene reconstruction. To address this, the paper proposes SCOUT-SLAM, a structurally-coupled dual-uncertainty framework in which both uncertainties are estimated from a shared base network. A low-rank adaptation of this network, trained on multi-view feature consistency, estimates a tracking uncertainty that does not depend solely on the reconstruction quality. A spatially-adaptive prior modulates the network's training objective so that reconstruction instability does not inflate uncertainty on static regions, keeping the shared representation intact for both branches. Evaluations on dynamic benchmarks (TUM RGB-D, Bonn Dynamic, Wild-SLAM MoCap) demonstrate that SCOUT-SLAM achieves state-of-the-art camera tracking accuracy and artifact-free static scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12384v2">Is Semantic SLAM Ready for Embedded Systems ? A Comparative Survey</a></div>
    <div class="paper-meta">
      📅 2026-09-13
      | 💬 Accepted for publication in Elsevier Robotics and Autonomous Systems (RAS)
    </div>
    <details class="paper-abstract">
      Semantic SLAM holds promise for robust robot navigation in complex environments, but its practicality on embedded systems remains uncertain. This paper presents a comparative survey of existing semantic visual SLAM systems from an embedded-deployment perspective. Unlike recent surveys centered on implicit representations, NeRFs, or 3D Gaussian Splatting as scene representations, our scope is the deployability of complete Semantic SLAM pipelines on resource-constrained robotics platforms, with particular emphasis on the NVIDIA Jetson AGX Orin. Three semantic-aware architectural approaches, namely Geometric SLAM, NeRF, and Gaussian Splatting-based SLAM, are evaluated with respect to accuracy, semantic reconstruction quality, memory footprint, power consumption, and throughput. Our Jetson measurements show that Semantic Geometric SLAM currently offers the most practical balance between accuracy and efficiency, whereas Gaussian Splatting-based systems require more than 16 GB of memory, draw more than 15 W, and remain far from real-time execution. The results also show that offline semantic preprocessing can mask the true cost of semantic integration. We therefore identify a research gap in adapting Semantic SLAM algorithms for embedded systems and propose directions for future work, including hardware-algorithm co-design and energy-efficient optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.14129v1">Deformable 2D Gaussian Splatting for Efficient 4K Video Compression</a></div>
    <div class="paper-meta">
      📅 2026-09-12
    </div>
    <details class="paper-abstract">
      Ultra-High-Definition (UHD) video presents significant challenges for efficient storage and real-time decoding. Learning-based methods, such as Neural Video Compression (NVC) and Implicit Neural Representations (INR), achieve competitive rate-distortion performance but suffer from high decoding latency and excessive memory usage. Meanwhile, Gaussian Splatting has recently attracted attention in the computer graphics community due to its ultra-fast rendering and high-fidelity visual quality. Despite these advantages, its application in video compression remains largely unexplored. To bridge this gap, we propose a real-time video compression framework that represents and compresses a Group of Pictures (GOP) using a coarse-to-fine multi-scale 2D Gaussian Splatting (2DGS) structure coupled with a lightweight deformation network. Experiments demonstrate that our method delivers rate-distortion performance in LPIPS that surpasses H.265 and other state-of-the-art learning-based video compression methods. Our work demonstrates the potential of Gaussian Splatting as a practical solution for efficient high-resolution video compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12626v2">Habitat-GS: A High-Fidelity Navigation Simulator with Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-12
      | 💬 Accepted to ECCV 2026. Project page: https://zju3dv.github.io/habitat-gs/
    </div>
    <details class="paper-abstract">
      Training embodied AI agents depends critically on the visual fidelity of simulation environments and the ability to model dynamic humans. Current simulators predominantly rely on mesh-based rasterization, for which photorealistic assets are costly to author at scale, and their support for dynamic human avatars is largely constrained to mesh representations, hindering agent generalization to human-populated real-world scenarios. We present Habitat-GS, a navigation-centric embodied AI simulator extended from Habitat-Sim that integrates 3D Gaussian Splatting scene rendering and drivable gaussian avatars while maintaining full compatibility with the Habitat ecosystem. Our system implements a 3DGS renderer for real-time photorealistic rendering and supports scalable 3DGS asset import from diverse sources. For dynamic human modeling, we introduce a gaussian avatar module that enables each avatar to simultaneously serve as a photorealistic visual entity and an effective navigation obstacle, allowing agents to learn human-aware behaviors in realistic settings. Experiments on point-goal navigation demonstrate that agents trained on 3DGS scenes achieve stronger cross-domain generalization. Evaluations on avatar-aware navigation further confirm that gaussian avatars enable effective human-aware navigation, while performance benchmarks validate the system's scalability. Code is available at https://github.com/zju3dv/habitat-gs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.13903v1">SkyAnchor: Updating Metric-scale Aerial 3D Gaussian Scenes from Unposed Ground-View Sequences</a></div>
    <div class="paper-meta">
      📅 2026-09-12
    </div>
    <details class="paper-abstract">
      We study how to update a pre-built aerial scene with a newly captured, unposed ground-view sequence. The aerial scene already contains a reliable metric Structure-from-Motion (SfM) reconstruction and a pre-trained 3D Gaussian Splatting (3DGS) model, whereas the ground-view sequence is collected later to add street-level appearance but has unknown camera poses and global scale. Registering this sequence to the aerial SfM reconstruction is challenging because single-image cross-view localization is brittle and long trajectories are prone to drift. To address these challenges, we present SkyAnchor, which treats the existing aerial scene as a fixed scaffold for ground-view registration and scene update instead of jointly reconstructing aerial and ground imagery from scratch. It first localizes short groups of consecutive ground frames against geometrically verified aerial support, producing sparse anchor poses. It then recovers the full ground trajectory with anchor-constrained submaps, fixing the front and rear anchor poses during incremental registration and bundle adjustment. Finally, it inserts filtered ground Gaussians while preserving the aerial view, followed by lightweight joint refinement. Experiments on seven real aerial--ground scenes show accurate metric ground trajectories and updated 3D Gaussian scenes with strong aerial- and ground-view rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13682v4">When Variance Is Not an Error Map: Calibrated Uncertainty for Radiative Gaussian Splatting in Sparse-View CT</a></div>
    <div class="paper-meta">
      📅 2026-09-12
    </div>
    <details class="paper-abstract">
      Does an uncertainty map identify where a reconstruction is wrong? In sparse-view computed tomography (CT), we find a sharp gap between whole-volume evaluation and error localization inside the object. We derive clamp-aware analytic moments for factorized Gaussian-density distributions, with a variance pass through existing rendering interfaces that is $7.9\times$ faster than a 16-sample estimator. On a 15-scene benchmark, median variance--error Spearman correlation falls from $0.846$ over the whole volume to $0.108$ in foreground. The pattern recurs across representations and acquisition settings. Two analyses help explain the discrepancy: region contrast dominates global covariance, while $72$--$96\%$ of in-object squared error is shared across independently trained members. Spread is unchanged by a common error, although shared error alone does not determine ranking. Correcting the offset between the deployed reconstruction and predictive mean improves foreground correlation by only $0.0014$. The diagnosis separates two remedies. A log-normal control improves scale transfer without restoring localization; a supervised error predictor raises foreground correlation to $0.427$ on the benchmark and $0.566$ on eight held-out human subjects under simulated acquisition. A verified retrospective re-execution on eight additional subjects retains a median of $0.537$, with the same frozen predictors. The central lesson is to validate uncertainty in the region, and against the error target, for which it will be used.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07197v2">SUCCESS-GS: Survey of Compactness and Compression for Efficient Static and Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-11
      | 💬 The first three authors contributed equally to this work. The last two authors are co-corresponding authors. Please visit our project page at https://cmlab-korea.github.io/Awesome-Efficient-GS/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful explicit representation enabling real-time, high-fidelity 3D reconstruction and novel view synthesis. However, its practical use is hindered by the massive memory and computational demands required to store and render millions of Gaussians. These challenges become even more severe in 4D dynamic scenes. To address these issues, the field of Efficient Gaussian Splatting has rapidly evolved, proposing methods that reduce redundancy while preserving reconstruction quality. This survey provides the first unified overview of efficient 3D and 4D Gaussian Splatting techniques. For both 3D and 4D settings, we systematically categorize existing methods into two major directions, Parameter Compression and Restructuring Compression, and comprehensively summarize the core ideas and methodological trends within each category. We further cover widely used datasets, evaluation metrics, and representative benchmark comparisons. Finally, we discuss current limitations and outline promising research directions toward scalable, compact, and real-time Gaussian Splatting for both static and dynamic 3D scene representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.12682v1">NOVA-GS: Noise-Aware View-Consistent Gaussian Splatting for Low-Light Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-09-11
      | 💬 Accepted to the 3D4S Workshop at CVPR 2026; selected for the Best Paper Award
    </div>
    <details class="paper-abstract">
      Reconstructing 3D scenes under real-world low-light conditions remains challenging due to severe sensor noise, low signal-to-noise ratios, and degraded photometric consistency, which destabilize geometry estimation and novel view synthesis. Existing approaches often rely on well-lit reference data for reliable Structure-from-Motion (SfM) initialization under degraded inputs or apply per-view enhancement methods that introduce cross-view inconsistencies. To address these limitations, we propose \textbf{NOVA-GS}, a unified noise-aware framework for low-light 3D Gaussian Splatting that subsumes enhancement, denoising, and geometry optimization within a single process. Our method leverages VGGT-based feed-forward estimation to obtain robust camera poses and geometry directly from degraded inputs, eliminating the need for SfM. Building on this initialization, NOVA-GS integrates three coupled components: a structure-aware enhancement module for exposure correction, a self-supervised denoising module with blind-spot masking for pseudo-supervision, and a consistency-driven Gaussian Splatting optimization enforcing cross-view geometric coherence. We further introduce a noise-guided spherical harmonic regularization to suppress view-dependent artifacts in noisy regions. Extensive experiments on diverse real-world low-light datasets demonstrate improved geometric fidelity, color consistency, and robustness without requiring paired supervision or well-lit references. https://shaurya2524.github.io/nova-gs/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.12395v1">Is Gaussian Splatting Becoming Neural Again? A Taxonomy and Controlled Study of Learned Parameterization</a></div>
    <div class="paper-meta">
      📅 2026-09-11
    </div>
    <details class="paper-abstract">
      Three-dimensional Gaussian Splatting (3DGS) combines explicit primitives with efficient rasterization, yet recent systems increasingly use neural networks to generate or share Gaussian parameters. We characterize this trend along five axes: attribute decoding, spatial sharing, view-conditioned decoding, topology generation, and amortized inference. An analysis of 19 representative methods shows that these choices address different limitations and cannot be reduced to a binary neural label. We also isolate three forms of neural parameterization in a controlled mip-NeRF 360 study. Sharing appearance and opacity improves reconstruction quality, while decoding geometric structure offers no further gain. The evidence favors selective neuralization: shared functions help when they capture reusable correlations without sacrificing the local geometric freedom of explicit splats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.12343v1">VS-Splat: Voxel-Selective feed-forward Gaussian Splatting for end-to-end 3D object reconstruction from sparse-views</a></div>
    <div class="paper-meta">
      📅 2026-09-11
      | 💬 Main paper: 10 pages, 4 figures. Appendix: 4 pages, 5 figures, Transaction of Multimedia
    </div>
    <details class="paper-abstract">
      Feed-forward Gaussian splatting models have demonstrated remarkable effectiveness in reconstructing three-dimensional (3D) objects from a few two-dimensional (2D) images, even if they are unseen. As existing methods typically predict Gaussian primitives uniformly across the 3D space, most primitives are placed in non-object regions. This may hinder the representation of fine object details. This paper proposes a Voxel-Selective Gaussian Splatting model (VS-Splat), a new end-to-endfeed-forward Gaussian splatting framework that predicts many primitives only within selected voxels that are likely to belong to an object, without 3D structural supervision. To achieve this, we propose a new learnable voxel selection approach that identifies object-centric voxels only with 2D rendering supervision. Our sparse-view rendering experiments with three benchmark datasets show that proposed VS-Splat outperforms several state-of-the-art methods. We further demonstrate its effectiveness as a backbone for an existing densification method and show that anoptional extension improves its robustness to inaccurate camera pose estimates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.20386v2">ATLAS Navigator: Active Task-driven LAnguage-embedded Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-10
    </div>
    <details class="paper-abstract">
      We address the challenge of task-oriented navigation in unstructured and unknown environments, where robots must incrementally build and reason on informative metric-semantic maps in real time. Since tasks may require clarification or re-specification, it is necessary for the information in the map to be rich enough to enable generalization across a wide range of tasks. To effectively execute tasks specified in natural language, we propose a hierarchical representation built on language-embedded Gaussian splatting that enables both sparse semantic planning that lends itself to online operation and dense geometric representation for collision-free navigation. We validate the effectiveness of our method through real-world robot experiments on a ground robot equipped with only vision-based sensing. The experiments are conducted in both cluttered indoor and kilometer-scale outdoor environments, covering more than 2 km and 47,000 square meters of area. We achieve a competitive ratio of about 59% relative to the shortest possible path, demonstrating efficient exploration and semantic navigation in challenging real-world environments. Experiment videos and more details can be found on our project page: https://ongdexter.github.io/atlasnav.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.11223v1">Tri-DehazeGS: Scene--Medium Decoupled Gaussian Splatting with Transmittance-Aware Optimization</a></div>
    <div class="paper-meta">
      📅 2026-09-10
    </div>
    <details class="paper-abstract">
      Recovering clean 3D scenes from hazy multi-view images is challenging because haze attenuates scene radiance and introduces atmospheric scattering. Recent scattering-aware Gaussian Splatting methods introduce physical haze models into reconstruction, but they often apply degradation in image space or bind medium-related variables to Gaussian primitives, which can entangle clean scene radiance with atmospheric effects. Moreover, low-transmittance regions provide weakened supervision for Gaussian optimization, causing distant or dense-haze areas to be under-reconstructed. We argue that clean reconstruction under haze requires both scene--medium disentanglement and transmittance-aware optimization rebalancing. To this end, we propose Tri-DehazeGS, a scene--medium decoupled Gaussian Splatting framework. It represents the clean scene with Gaussian primitives, models the participating medium using an independent view-shared tri-plane field, and composes hazy observations through a physical scattering model. We further introduce Medium-Decoupled Transmittance Gradient Compensation (MD-TGC), which compensates haze-suppressed gradients after medium freezing without altering forward rendering. Experiments on real and synthetic haze benchmarks show that Tri-DehazeGS improves clean novel-view reconstruction. Code is available at https://github.com/aptx46/Tri-DehazeGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.06017v2">FujinSplat: Seeing Through Smoke with RAW-Domain Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-10
      | 💬 20 pages, 11 figures, including supplementary material. v2: added code link; scientific content unchanged. Code: https://github.com/I2WM/FujinSplat
    </div>
    <details class="paper-abstract">
      The appearance of a smoky scene is shaped by two processes that a camera records together: the participating medium alters scene radiance in a view-dependent way, and the image signal processor (ISP) then remaps the result through a nonlinear tone and color transformation. Recovering a clean 3D scene requires separating both. Per-view sRGB dehazing acts only after the ISP has entangled them; standard 3D reconstruction ignores the medium and absorbs it into scene geometry and radiance. FujinSplat addresses the problem in the RAW domain, where the two processes remain separable. A per-scene Base ISP is fitted from the scene's hazy RAW captures to its own camera renderings and then frozen, providing a fixed photometric anchor that performs no dehazing. Analyzing expert corrections reveals a compact, low-dimensional correction space identifiable from RAW alone. FujinSplat therefore fits per-view action answers at the training poses and trains a single scene-agnostic controller to regress them from RAW; the corrected views supervise one static 3D Gaussian representation, jointly with a bounded per-view residual that reconciles cross-view photometric inconsistencies. On the RealX3D real-world smoke benchmark FujinSplat clearly outperforms the strongest comparable baseline, ahead of both physics-based reconstruction and restoration-then-3DGS pipelines. Code is available at https://github.com/I2WM/FujinSplat .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.11079v1">RIDE: Relocalization-Informed Depth Estimation with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-10
    </div>
    <details class="paper-abstract">
      Render--match--PnP relocalization establishes correspondences between query image pixels and 3D map points for camera pose recovery, but their potential to support dense depth estimation is often overlooked. To exploit this geometric information, we present RIDE, which estimates dense metric depth from a robot's RGB stream. Given a metrically scaled 3D Gaussian Splatting (3DGS) model, RIDE combines sparse metric depth observations derived from PnP-RANSAC inlier correspondences with the geometric prior of a pretrained video-depth model. To handle uneven and intermittent observations, it integrates global and local depth correction with temporal memory, supporting depth estimation through short observation gaps after metric scale initialization. Trained on public RGB-D videos, RIDE is evaluated on robot sequences without fine tuning. Experiments show improved depth accuracy and temporal consistency over scale-only calibration, demonstrating how localization geometry can support both pose recovery and dense robot perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11298v3">InstantHDR: Single-forward Gaussian Splatting Initialization for HDR 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-09-09
    </div>
    <details class="paper-abstract">
      High dynamic range (HDR) novel view synthesis (NVS) aims to reconstruct HDR scenes from multi-exposure low dynamic range (LDR) images. Existing HDR pipelines heavily rely on known camera poses, well-initialized dense point clouds, and time-consuming per-scene optimization. Current feed-forward alternatives overlook the HDR problem by assuming exposure-invariant appearance. To bridge this gap, we propose InstantHDR, a feed-forward network that initializes 3D HDR scenes from uncalibrated multi-exposure LDR collections in a fast single forward pass. Specifically, we design a geometry-guided appearance modeling for multi-exposure fusion, and a meta-network for generalizable scene-specific tone mapping. Due to the lack of HDR scene data, we build a pre-training dataset, called HDR-Pretrain, for generalizable feed-forward HDR models, featuring 168 Blender-rendered scenes, diverse lighting types, and multiple camera response functions. Comprehensive experiments show that our InstantHDR delivers a single-forward HDR initialization at approximately 700 times the speed of SoTA optimization-based methods, and reaches comparable quality in real settings after lightweight post-optimization while remaining approximately 20 times faster. All code, models, and datasets: https://github.com/Bugjudger/InstantHDR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.10376v1">Shape-guided Gaussian Splatting for Sparse-View X-ray 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-09-09
      | 💬 Accepted at: Off-Grid: 1st Workshop on Continuous Representations and Grid-Free Methods in Medical Imaging, MICCAI 2026
    </div>
    <details class="paper-abstract">
      Sparse-view X-ray 3D reconstruction is essential for reducing radiation exposure, but recovering a density field from a handful of X-ray projections is severely ill-posed. Recently, 3D Gaussian Splatting has achieved state-of-the-art performance in sparse-view reconstruction by representing the volume using explicit, optimized primitives, but it requires dozens of projected views. With fewer views, reconstruction quality degrades severely since the explicit primitives are optimized freely without any anatomical information. Anatomical structures, in contrast, share similar geometry and density across a population. Their variations are bounded within a limited range that statistical shape models can capture. This paper proposes a shape-guided Gaussian splatting framework for sparse-view X-ray 3D reconstructions. Our contribution lies in driving Gaussian positions toward anatomically valid configurations, alongside atlas-based density regularization. Our method ensures anatomically consistent reconstruction and improves PSNR by 2.83 dB over a state-of-the-art Gaussian splatting baseline with as few as 5 views. Code Available: https://github.com/polyshape-lab/ShapeGuidedGaussian
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.10307v1">View-Structured Conformal Prediction for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) renders novel views in real time, but an uncertainty heatmap does not certify that a rendered view meets a certain prediction coverage. We treat novel-view synthesis as structured regression and ask that, with probability at least $1-α$, RGB prediction boxes cover at least a $1-β$ fraction of pixels in a new view. We propose View-Structured Conformal Prediction (VSCP). It splits the pre-calibration scale into a spatial shape from the renderer and a transferable view-difficulty factor, which predicts the smallest view-wise multiplier that shape needs. A held-out quantile over views (View-CP) then gives finite-sample validity even when transferring to new scenes. The same factorization makes the analysis exact: a conformity score is the ratio of oracle to predicted view difficulty, and excess width separates into a test-side and a calibration-side term. Across 13 real scenes, pixel-pooled calibration reaches 89.9\% marginal pixel coverage but only 61.4\% view-event coverage at a 90\% target, while View-CP reaches 91.7--92.0\%. At matched coverage VSCP cuts width by 22.1\% against a constant scale, and matches a ten-model ensemble's 21.0\% reduction using only one model per scene and four rather than ten rasterization passes per query. VSCP also improves on the closest single-model baseline, the 3DGS-U field, by 4.7 points ($p=0.0225$). The view predictor transfers from bounded source families to all nine unbounded Mip-NeRF~360 scenes. There the full scale beats the constant scale with 20.7\% width saving on all nine scenes. It also keeps an 18.3\% saving under a different densification backbone and runs at 216--280 FPS on an RTX~4090.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.10095v1">LinearMask-GS: Stable-Mask Importance Pruning for Compact 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-09
      | 💬 Accepted to BMVC 2026. 17 pages main paper + 17 pages supplementary material, 3 figures, 4 tables in the main paper
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables real-time novel view synthesis but produces millions of primitives through adaptive densification, leading to significant storage overhead. Learned-mask pruning methods such as LP-3DGS address this by assigning each Gaussian a learnable mask to identify and prune redundant primitives. However, we identify a limitation of this paradigm: the steep slope of the Gumbel-Sigmoid activation drives mask values to the extremes within the short mask-training window, before the importance ranking has stabilized, producing a sharply bimodal distribution from which that ranking can no longer be reliably recovered. We propose LinearMask-GS, which replaces Gumbel-Sigmoid with a linear increment activation that keeps mask values in a mid-confidence regime throughout mask training, producing a stable, unimodal mask distribution whose ranking tracks importance. On Mip-NeRF 360, our method achieves 3.6x and 1.6x Gaussian reductions over 3DGS and LP-3DGS, respectively, while maintaining or improving rendering quality. For outdoor scenes, it yields a 1.6x reduction (from 2.18M to 1.36M) with notable gains in PSNR (+0.38 dB), SSIM (+0.025), and LPIPS (-0.029).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.09606v1">RouteBridge: Reliability-Routed Bidirectional Distillation Between Neural Radiance Fields and 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-09
    </div>
    <details class="paper-abstract">
      Neural radiance fields (NeRFs) and 3D Gaussian Splatting (3DGS) encode a scene with complementary inductive biases, but existing cross-representation distillation typically fixes one representation as teacher for the entire scene. A globally fixed teacher can propagate local reconstruction errors. We present RouteBridge, a bidirectional framework that selects the teaching direction for each ray. Its reliability estimator combines photometric residuals with representation-specific geometric evidence and routes supervision from NeRF to 3DGS, from 3DGS to NeRF, or abstains. A renderer-independent interface transfers color, opacity, and normalized depth without shared features or point correspondence. On mip-NeRF 360, the NeRF and 3DGS exports reach 28.56 and 28.77 dB, respectively. The 3DGS export improves over 3DGS by 1.56 dB and over NeRF-GS by 0.45 dB while reducing LPIPS to 0.207. On static three-view DTU, RouteBridge obtains 21.12 dB. Ablations show that both adaptive routing and geometric ray targets contribute to the improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.08730v1">CVT-GS: Learning to Simplify 3D Gaussian Splatting with Centroidal Voronoi Tessellation</a></div>
    <div class="paper-meta">
      📅 2026-09-08
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has emerged as a powerful representation for real-time novel view synthesis, rendering high-fidelity scenes often relies on a massive number of Gaussian primitives, incurring substantial storage and computational overhead. Existing simplification techniques are largely intrusive, requiring training-time pruning, architectural modifications, or computationally expensive per-scene fine-tuning. These drawbacks limit their deployment on off-the-shelf pretrained models. In this paper, we propose CVT-GS, a novel optimization-free post-hoc simplification framework that directly compresses trained 3DGS scenes without sacrificing visual fidelity. Our approach first constructs spatially coherent cells over Gaussian centers via a geometry-aware Centroidal Voronoi Tessellation (CVT). Subsequently, a lightweight neural cell merger predicts the geometry and appearance of a single, highly representative Gaussian primitive for each cell under differentiable rendering supervision. By formulating simplification as a rendering-aware many-to-one merging process rather than naive primitive pruning, CVT-GS outputs a standard 3DGS scene that is seamlessly compatible with existing renderers. Experiments on various datasets demonstrate the superiority of our method. Notably, when achieving a 100-fold reduction in Gaussian points, our method operates 12 times faster than state-of-the-art methods while improving the PSNR by 1.3 dB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.29400v2">Adaptive Densification for High-Fidelity and Efficient Sparse Gaussian Splatting in Arbitrary-Scale Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2026-09-08
    </div>
    <details class="paper-abstract">
      Arbitrary-Scale Super-Resolution (ASR) aims to reconstruct high-resolution images at any continuous magnification. While 2D Gaussian Splatting (GS) has recently shown great promise for ASR, current methods struggle to balance visual quality and computational cost. Approaches targeting high fidelity rely on powerful backbones and uniform, highly dense Gaussian grids, leading to prohibitive memory and inference costs. Conversely, methods prioritizing efficiency aggressively simplify their architectures, severely compromising visual quality. To bridge this gap, we observe that a core capability of GS remains largely underexplored in ASR: the potential for dynamic densification, i.e., the spatially adaptive allocation of Gaussians based on image content. Unlike standard scene fitting, where densification is guided by a known ground truth, applying this to ASR is highly non-trivial because the high-resolution target is exactly what the model must predict. To address this challenge, we propose QuADA-GS, an approach that retains a powerful representational backbone but autonomously predicts where to allocate Gaussians relying strictly on the low-resolution input. By adopting a sparse approach, QuADA-GS refines features and increases Gaussian density strictly where structural complexity demands it. Because this adaptive allocation produces a non-uniform hierarchical topology, we introduce a novel, highly efficient communication mechanism to process these sparse features, bypassing standard dense bottlenecks. Extensive experiments indicate that our approach successfully balances visual quality and computational requirements, providing an improved and competitive trade-off for ASR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.08449v1">GSComplete: Gaussian Splat Completion with 2D Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2026-09-08
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      Gaussian splats provide a fast, high-fidelity representation for 3D objects but are often constructed from incomplete input data in practice, leaving missing regions. Existing completion methods either do not preserve the original splats or require scarcely available 3D training data. We propose GSComplete, which combines 3D generation based on Score Distillation Sampling with a novel preservation loss that encourages the original splats to be preserved where they should be visible. This effectively completes the Gaussian splat object using only 2D diffusion priors while fully preserving existing splats and generating new splats only in missing regions, without occluding the input. To evaluate our approach, we introduce a new dataset of partial Gaussian splat objects and show that GSComplete achieves significantly more accurate preservation of the input than existing methods with comparable plausibility of the completed result. Our code and dataset will be made available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.29453v3">Resonant Brane Splatting for Arbitrary-Scale Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2026-09-08
    </div>
    <details class="paper-abstract">
      Arbitrary-Scale Super-Resolution (ASR) reconstructs images at continuous magnification factors. Recent methods accelerate inference by replacing computationally heavy implicit neural decoders with explicit 2D Gaussian Splatting (GS). However, since standard Gaussians are smooth low-pass primitives, modeling edges and fine textures requires multiple overlapping, well-aligned splats, which creates severe bottlenecks during rasterization. To address this, we introduce Resonant Brane Splatting (RBS), a feed-forward ASR framework. RBS replaces flat Gaussians with Branes: expressive primitives that emit spatially varying colors to natively model local contrast and complex textures within a single footprint. We achieve this by augmenting the standard Gaussian envelope with internal Gaussian-Hermite modes, assigning a distinct color coefficient to each. The zero-order mode recovers standard GS, while higher-order modes capture high frequencies. We predict Brane parameters directly from low-resolution features. Because Branes provide a mathematically richer formulation than simple Gaussians, far fewer primitives need to overlap to reconstruct a given target pixel. To exploit this, we introduce an efficient fully differentiable rasterizer with a precise culling strategy based on the classical quantum turning point. This allows us to safely skip negligible regions, drastically reducing the rendering overhead. Experiments on standard ASR benchmarks show that RBS improves reconstruction quality over implicit and GS baselines, while achieving superior speed-quality trade-off than prior GS methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.08332v1">EdMCGS: Event-Driven Markov Chain Gaussian Splatting for Extreme-Low-Frame-Rate Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-09-08
      | 💬 29 pages, 3 figures
    </div>
    <details class="paper-abstract">
      We present EdMCGS (Event-driven Markov chain Gaussian Splatting), an end-to-end method for reconstructing dynamic 3D scenes from extreme-low-frame-rate RGB together with an event stream, which can then be rendered at any intermediate timestamp. Methods relying solely on RGB images generate numerous artifacts due to the lack of evidence from between consecutive frames. To supply this missing evidence, we model the scene motion as an event-driven Markov chain, in which the sparse RGB frames anchor the state at their own timestamps while the events recorded within an interval drive the transition across it. Since the transition reads the events of the current interval, it remains active at inference and produces the in-between motion of the 3D Gaussians directly from the events rather than by interpolation, which sets our method apart from prior work that uses events only as training-time supervision. The state is carried by a compact set of control points, each driven by the events sampled in the neighborhood of its own image projection, and a temporal local isometry term keeps the propagated motion locally rigid. Experiments on synthetic and real-world scenes show that EdMCGS outperforms both RGB-based and event-based baselines, while rendering in real time with far fewer Gaussians than the strongest event-based baseline. We release our source code and a new dataset at https://github.com/joseclipse/EdMCGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.25909v2">Accelerated Spatio-Temporal Gaussian Rendering via Kinematic and Semantic Priors</a></div>
    <div class="paper-meta">
      📅 2026-09-07
      | 💬 ECCV 2026 7th Workshop on Open-World 3D Scene Understanding and Representations (OpenSUN3D). Project page: https://be2rlab.github.io/r5dgs/
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic 3D scenes from multi-view videos is a foundational task for robotics, AR/VR, and digital twins. While 3D Gaussian Splatting (3DGS) provides state-of-the art scene representation, its dynamic adaptations (4DGS) face significant challenges. Early MLP-based deformation models excel at interpolating within training frames but fail at temporal extrapolation --- important feature for forecasting in predictive systems. To address this, physically grounded approaches have been proposed to enable robust future forecasting, while, concurrently, semantic 4DGS methods have been developed for scene analysis and language-driven prompting. However, there is currently no framework that unites both physical consistency and semantic understanding. In this paper, we propose the first, to the best of our knowledge, unified 4DGS approach that is both physically grounded and semantically aware. Furthermore, to optimize computational efficiency, we introduce a novel rigid-body constraint mechanism. Instead of predicting the physical dynamics for every Gaussian particle, our model predicts the motion of a single anchor Gaussian per semantic object, updating the remaining Gaussians via relative transformations. This architectural design significantly accelerates novel view synthesis compared to an unconstrained baseline, while enabling object-accurate extrapolation and intuitive semantic interaction. Our project page: https://be2rlab.github.io/r5dgs/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.07734v1">TV-SGS: Gaussian Splatting with Geometric Information Propagation via Tensor Voting under sparse views</a></div>
    <div class="paper-meta">
      📅 2026-09-07
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has been effective in inferring scene representations that excel in novel view synthesis. Multiple splats cooperate seamlessly to synthesize the pixels of novel views and are jointly optimized even though they only affect each other indirectly, via pixels they project to in common. We present an approach that enables direct communication among splats to enhance the geometric structures they form in 3D. This is accomplished by Tensor Voting, which was originally designed to infer structures from noisy inputs and has been adapted here to provide supervision during test-time optimization, leading to more accurate scene geometry. We introduce a new class of 3D losses that do not rely on rendering and can be combined with essentially all losses previously reported in the literature. Our 3D losses are especially effective when the input views are sparse and geometric regularization is essential due to limited supervision from the images. Our method is easy to integrate with a diverse set of backbones, and our experiments on the DTU and Tanks-and-Temples datasets demonstrate that TV-SGS improves the geometry of the outputs compared to the backbone, while maintaining or improving rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.07557v1">Heat Kernel Textures: the Geodesic Gaussians That Do Not Splat</a></div>
    <div class="paper-meta">
      📅 2026-09-07
      | 💬 ECCV 2026 (Long Oral)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has recently revolutionised novel view synthesis as well as many other 3D vision methods and applications. Drawing inspiration from this representation, we now rethink textures to overcome the main issues of UV mapping while considerably lowering their memory footprint. Heat Kernel Textures (HKTex) eliminate UV unwrapping as well as their persistent issues of wasted UV space, seams, distortions, vertex-duplication, and varying resolution. Grounded in discrete Riemannian geometry and intrinsically defined on any manifold surface discretised as a triangular mesh, HKTex uses anisotropic heat kernels as geodesic equivalents to Gaussians. Like our kernels, also the optimisation of their position and the adaptive densification strategies were redefined to operate on the surface of the object to be textureised. Our novel representation is also fully integrated with a physically based renderer and can be optimised either from existing textures or multi-view images. Our project page and code are available at circle-group.github.io/research/HeatKernelTextures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05155v4">Aes3D: Aesthetic Assessment in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-07
    </div>
    <details class="paper-abstract">
      As 3D Gaussian Splatting (3DGS) gains attention in immersive media and digital content creation, assessing the aesthetics of 3D scenes becomes important in helping creators build more visually compelling 3D content. However, existing evaluation methods for 3D scenes primarily emphasize reconstruction fidelity and perceptual realism, largely overlooking higher-level aesthetic attributes such as composition, harmony, and visual appeal. This limitation comes from two key challenges: (1) the absence of general 3DGS datasets with aesthetic annotations, and (2) the intrinsic nature of 3DGS as a low-level primitive representation, which makes it difficult to capture high-level aesthetic features. To address these challenges, we propose Aes3D, the first systematic framework for assessing the aesthetics of 3D neural rendering scenes. Aes3D includes Aesthetic3D, the first dataset dedicated to 3D scene aesthetic assessment, built on our proposed annotation strategy for 3D scene aesthetics. In addition, we present Aes3DGSNet, a lightweight model that directly predicts scene-level aesthetic scores from 3DGS representations. Notably, our model operates solely on 3D Gaussian primitives, eliminating the need for rendering multi-view images and thus reducing computational cost and hardware requirements. Through aesthetics-supervised learning on multi-view 3DGS scene representations, Aes3DGSNet effectively captures high-level aesthetic cues and accurately regresses aesthetic scores. Experimental results demonstrate that our approach achieves strong performance while maintaining a lightweight design, establishing a new benchmark for 3D scene aesthetic assessment. Code and datasets will be made available in a future version.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.14373v4">SEGA: Drivable 3D Gaussian Head Avatar from a Single Image</a></div>
    <div class="paper-meta">
      📅 2026-09-07
    </div>
    <details class="paper-abstract">
      Creating photorealistic 3D head avatars from limited input has become increasingly important for applications in virtual reality, telepresence, and digital entertainment. While recent advances like neural rendering and 3D Gaussian splatting have enabled high-quality digital human avatar creation and animation, most methods rely on multiple images or multi-view inputs, limiting their practicality for real-world use. In this paper, we propose SEGA, a novel approach for Single-imagE-based 3D drivable Gaussian head Avatar creation that combines generalized prior models with a new hierarchical UV-space Gaussian Splatting framework. SEGA seamlessly combines priors derived from large-scale 2D datasets with 3D priors learned from multi-view, multi-expression, and multi-ID data, achieving robust generalization to unseen identities while ensuring 3D consistency across novel viewpoints and expressions. We further present a hierarchical UV-space Gaussian Splatting framework that leverages FLAME-based structural priors and employs a dual-branch architecture to disentangle dynamic and static facial components effectively. The dynamic branch encodes expression-driven fine details, while the static branch focuses on expression-invariant regions, enabling efficient parameter inference and precomputation. This design maximizes the utility of limited 3D data and achieves real-time performance for animation and rendering. Additionally, SEGA performs person-specific fine-tuning to further enhance the fidelity and realism of the generated avatars. Experiments show our method outperforms state-of-the-art approaches in generalization ability, identity preservation, and expression realism, advancing one-shot avatar creation for practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.07346v1">From Explicit References to Scene Manifolds: Distributional Fidelity and Realism for Radiance Field Quality Assessment</a></div>
    <div class="paper-meta">
      📅 2026-09-07
    </div>
    <details class="paper-abstract">
      Radiance field representations such as 3D Gaussian Splatting (3DGS) enable high-quality novel view synthesis but can introduce complex, view-dependent artifacts from reconstruction, rendering, and compression. Reliable perceptual quality assessment (QA) is thus essential for evaluating rendered views and guiding the design of perceptually faithful scene representations. Existing full-reference QA metrics require an aligned reference image, while recent cross-reference metrics relax this requirement by comparing a test view with non-aligned references. However, under wide-baseline radiance field settings, selecting a reliable nearby reference can be difficult, particularly when evaluating views along arbitrary trajectories and poses. We propose SCODA, a lightweight scene-conditioned objective QA method that shifts QA from explicit image-to-image comparison to scene-manifold modeling. High-quality observations of each scene are represented as a multivariate Gaussian distribution in deep feature space, producing a semantic fidelity score that measures deviation from the scene distribution. A weakly-supervised distortion-aware patch discriminator provides a complementary realism signal, and both cues are combined through an unsupervised bounded fusion strategy. Experiments on multiple benchmarks show strong agreement with human judgments and robust generalization across GS- and NeRF-generated views and trajectories. Code is publicly available at https://gitlab.com/saeedmp/scoda.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.07274v1">LightSplat: Real-Time High-Fidelity 3D Gaussian SLAM with Loop Closure</a></div>
    <div class="paper-meta">
      📅 2026-09-07
      | 💬 Accepted to 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
    </div>
    <details class="paper-abstract">
      SLAM systems based on 3D Gaussian Splatting (3DGS) have recently demonstrated promising reconstruction accuracy for dense 3D scene representations. However, current 3DGS systems struggle to meet the strict demands of real-world deployments due to severe limitations in operational performance and map adaptability. To this end, we propose LightSplat, a hybrid-representation RGB-D SLAM framework. It synergizes local sparse features for robust and fast tracking with a dual-thread backend that progressively constructs dense Gaussian submaps. Crucially, we enable online loop closure through feature-accelerated 3DGS registration, refining overall map consistency through pose graph optimization. Ultimately, LightSplat achieves the online reconstruction of high-fidelity Gaussian map. Extensive experiments on multiple datasets and real-world robotic platform demonstrate that our method achieves near state-of-the-art reconstruction quality and the capability to accommodate practical camera motions, maintaining an average framerate of 8 FPS. Overall, LightSplat provides an efficient and robust foundation for deploying high-fidelity 3DGS in real-world environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.07231v1">Generalizable 6D Pose Estimation of Textureless Objects with Planar-based Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-07
      | 💬 7 pages, 5 figures. Accepted by the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026)
    </div>
    <details class="paper-abstract">
      Estimating the 6D pose of textureless objects without prior CAD models remains a critical challenge due to the lack of appearance features. While recent generalizable approaches alleviate the dependence on object-specific models, their performance on low-texture objects is often limited by insufficient geometric constraints in the underlying representations. In this work, we propose PG-Pose, a geometry-aware framework combining Planar-based Gaussian Splatting (PGS) reconstruction and Geometry-driven pose optimization. In the offline representation extraction stage, three distinct representations of the object are extracted from multi-view reference RGB images with known poses. PG-Pose reconstructs a 3D Gaussian representation and renders high-fidelity depth maps to generate 3D point clouds through back projection. In the online pose inference stage, the initial pose of the input image is estimated by 2D-3D correspondence matching between the input image and the reconstructed 3D point clouds, followed by a PGS-Refiner for iterative pose optimization. Evaluations on the OnePose-LowTexture datasets, PG-Pose achieves an average accuracy of 94.2% ADD(S)@0.1d, with a 2.1% improvement average accuracy compared with the state-of-the-art (SOTA) GS-based approach. To further demonstrate the effectiveness of PG-Pose for industrial robots in grasping tasks, we deploy it on a dual-arm industrial robot and successfully realize the grasping task on an unseen object.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14191v4">MCGS-SLAM: A Multi-Camera SLAM Framework Using Gaussian Splatting for High-Fidelity Mapping</a></div>
    <div class="paper-meta">
      📅 2026-09-07
      | 💬 Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026. Code: https://github.com/mcgs-slam/mcgs-slam
    </div>
    <details class="paper-abstract">
      Recent progress in dense SLAM has primarily targeted monocular setups, often at the expense of robustness and geometric coverage. We present MCGS-SLAM, the first purely RGB-based multi-camera SLAM system built on 3D Gaussian Splatting (3DGS). Unlike prior methods relying on sparse maps or inertial data, MCGS-SLAM fuses dense RGB inputs from multiple viewpoints into a unified, continuously optimized Gaussian map. A multi-camera bundle adjustment (MCBA) jointly refines poses and depths via dense photometric and geometric residuals, while a scale consistency module enforces metric alignment across views using low-rank priors. The system supports RGB input and maintains real-time performance at large scale. Experiments on synthetic and real-world datasets show that MCGS-SLAM consistently yields accurate trajectories and photorealistic reconstructions, usually outperforming monocular baselines. Notably, the wide field of view from multi-camera input enables reconstruction of side-view regions that monocular setups miss, critical for safe autonomous operation. These results highlight the promise of multi-camera Gaussian Splatting SLAM for high-fidelity mapping in robotics and autonomous driving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.06874v1">MedGSSR: Generalizable Medical Image Super-Resolution 3D Reconstruction via Hierarchical Feed-forward Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-06
      | 💬 ECCV 2026
    </div>
    <details class="paper-abstract">
      High-resolution volumetric medical imaging is critical for clinical diagnosis, yet acquisition is often limited by scanner hardware, scan time, and for CT, radiation dose. Medical 3D Super-Resolution (Med3DSR) offers a computational alternative, but existing methods commonly rely on per-subject optimization, pretrained priors, or coordinate-based implicit representations, which compromise anatomical fidelity and limit efficiency. To address these limitations, we present MedGSSR, a fully end-to-end feed-forward framework that represents volumes as an explicit 3D Gaussian field for Med3DSR. Unlike coordinate-based implicit functions, our explicit 3D Gaussian representation naturally enhances signal continuity and local high-frequency fidelity. Specifically, MedGSSR explicitly decouples the reconstruction process into coarse-grained structural preservation and fine-grained textural refinement through the proposed Pyramid Anatomical Encoder and a Hierarchical Gaussian Projector. To support arbitrary-scale super-resolution, we introduce sub-voxel Gaussian decomposition and a Differentiable Gaussian Voxelizer that directly queries the continuous 3D intensity field, reducing discretization artifacts. Extensive experiments on MRI and CT benchmarks demonstrate that MedGSSR significantly outperforms state-of-the-art methods. Notably, our framework exhibits robust generalizability across unseen datasets without requiring per-subject optimization, enabling fast inference and high-fidelity volumetric super-resolution in practical clinical settings. Our project webpage, including code, is at https://william2ai.github.io/medgssr
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.13262v1">Filling the Unseen: Scene Extrapolation via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-06
      | 💬 Accepted to ACM Multimedia (ACM MM) 2026. 9 pages
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting achieves photorealistic reconstruction within training view distribution, yet it degrades on out-of-distribution novel views, exhibiting holes in unobserved regions and artifacts in observable areas. Recent works formulate this task as extrapolation and interpolation and try to address it with generative models, but remain limited in extrapolation scale and quality. They repeat a generate-reconstruct-shift cycle to progressively build a scene, which introduces accumulated errors with every step conditioning on previous outcomes. In this work, we propose a holistic framework for extrapolation and interpolation. We devise an independent camera view detection mechanism to enable parallel conflict-free extrapolation, circumventing the reliance on the aforementioned error-prone cycle. Building upon this, we design a hierarchical pipeline that extrapolates independent and dependent camera views separately. Additionally, previous methods overlook inconsistency between generated and original images, resulting in compromising well-reconstructed areas. We propose a plug-and-play Quality-Aware Mask (QA-Mask) module, enabling selective utilization of generated data. By calibrating learning weights with pixel-wise rendering quality, it prevents generation-induced degradations on well-constructed areas. Extensive experiments demonstrate the superior performance of our framework, with QA-Mask generalizing to multiple generative reconstruction models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10239v3">AdaptSplat: Adapting Vision Foundation Models for Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-06
    </div>
    <details class="paper-abstract">
      This work explores a simple yet powerful lightweight adapter design for feed-forward 3D Gaussian Splatting (3DGS). Existing methods typically apply complex, architecture-specific designs on top of the generic pipeline of image feature extraction $\rightarrow$ multi-view interaction $\rightarrow$ feature decoding. However, constrained by the scale bottleneck of 3D training data and the low-pass filtering effect of deep networks, these methods still fall short in cross-domain generalization and high-frequency geometric fidelity. To address these problems, we propose AdaptSplat, which demonstrates that without complex component engineering, introducing a single adapter of only 1.5M parameters into the generic architecture is sufficient to achieve superior performance. Specifically, we design a lightweight Frequency-Preserving Adapter (FPA) that extracts direction-aware high-frequency structural priors from the shallow features of a powerful vision foundation model backbone, and seamlessly integrates them into the generic pipeline via high-frequency positional encodings and adaptive residual modulation. This effectively compensates for the high-frequency attenuation caused by over-smoothing in deep features, improving the fitting accuracy of Gaussian primitives on complex surfaces and sharp boundaries. Extensive experiments demonstrate that AdaptSplat achieves state-of-the-art feed-forward reconstruction performance on multiple standard benchmarks, with stable generalization across domains. Code available at: https://github.com/xmw666/AdaptSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04313v2">Mind-to-Face: Neural-Driven Photorealistic Avatar Synthesis via EEG Decoding</a></div>
    <div class="paper-meta">
      📅 2026-09-05
      | 💬 16 pages, 11 figures, ECCV 2026 Spotlight
    </div>
    <details class="paper-abstract">
      Current expressive avatar systems rely heavily on visual cues, failing when faces are occluded or when emotions remain internal. We present Mind-to-Face, the first framework that decodes non-invasive electroencephalogram (EEG) signals directly into high-fidelity facial expressions. We build a dual-modality recording setup to obtain synchronized EEG and multi-view facial video during emotion-eliciting stimuli, enabling precise supervision for neural-to-visual learning. Our model uses a CNN-Transformer encoder to map EEG signals into dense 3D position maps, capable of sampling over 65k vertices, capturing fine-scale geometry and subtle emotional dynamics, and renders them through a modified 3D Gaussian Splatting pipeline for photorealistic, view-consistent results. Through extensive evaluation, we show that EEG alone can reliably predict dynamic, subject-specific facial expressions, including subtle emotional responses, demonstrating that neural signals contain far richer affective and geometric information than previously assumed. Mind-to-Face establishes a new paradigm for neural-driven avatars, enabling personalized, emotion-aware telepresence and cognitive interaction in immersive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.06157v1">From Splats to Silicon: Rethinking Computational Efficiency of 3DGS</a></div>
    <div class="paper-meta">
      📅 2026-09-05
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) represents scenes with explicit primitives and supports real-time novel-view synthesis, yet its system efficiency varies substantially across scenes, viewpoints, rendering paths, and platform constraints. Existing studies pursue efficiency through representation and algorithm design, GPU runtime optimization, and architectural support, but their reported gains correspond to different points along the rendering and update paths. Connecting these indicators to end-to-end system benefit requires tracing how each optimization changes Gaussian selection, screen-space work, data movement, and stage or frame time. We therefore use a workload-centric framework to connect representation and algorithm research, GPU runtimes, and hardware architectures and to identify recurring workload patterns. We complement literature analysis with reproduced measurements and controlled GPU profiling of selected implementations, relating workload counts to stage time and memory traffic. Together, these comparisons show that system gains depend on workload reductions reaching downstream execution, granularity matching each stage, and the cost of data transfers, synchronization, and cached results, gradients, and optimizer data. Building on these findings, we discuss more consistent evaluation under rendering-quality constraints and identify key directions for future system design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.05925v1">AVSplat: Dense-View Feed-Forward 3D Gaussian Splatting with Assist-View Preconditioning</a></div>
    <div class="paper-meta">
      📅 2026-09-05
    </div>
    <details class="paper-abstract">
      Pose-free feed-forward 3D Gaussian Splatting enables novel view synthesis from uncalibrated multi-view images. Although more views should improve performance, existing methods often degrade with dense-view inputs because global aggregation spreads attention over many tokens, and naive voxel fusion averages many Gaussians into overly smooth representations. We present AVSplat, a framework that turns additional views into reliable signals for both aggregation and representation. Before global attention, each view performs a single lightweight interaction with a small set of Assist Views chosen for relevance and diversity, and the cached features provide a focused scene context that stabilizes correspondence. For representation, we use adaptive temperature-aware voxel fusion that sharpens attribution under high occupancy, guided by occupancy and point confidence. Crucially, AVSplat restores positive view scaling where performance remains stable or improves as more input views are added, instead of degrading in the dense-view regime. Ablations show that Assist View Preconditioning is primarily responsible for preventing dense-view degradation, while Occupancy-guided Voxel Fusion contributes most of the single-point image-quality gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.05888v1">UniFusion: Sparse-View 4D Reconstruction via Unified Spatio-temporal Depth Alignment</a></div>
    <div class="paper-meta">
      📅 2026-09-05
      | 💬 Project page: https://yongzhelyu.github.io/UniFusion
    </div>
    <details class="paper-abstract">
      In this paper, we address the challenging problem of 4D reconstruction from sparse-view videos. This setup usually relies on monocular depth estimation to provide priors for the reconstruction model. A key challenge arises from limited cross-view overlap and temporal variation, making monocular depth predictions inconsistent across views and time. Existing methods align spatial and temporal dimensions in separate stages, requiring foreground segmentation masks while failing to leverage temporal cues for cross-view alignment. Contrary to these methods, we propose a unified spatial-temporal depth alignment framework that jointly resolves cross-view and cross-time inconsistencies without distinguishing foreground/background. Our method represents depth maps across views and time as a set of spatio-temporal neural fields. This representation not only yields fast convergence, but also captures spatio-temporal correlation among depth maps implicitly, without dependence on external segmentation/tracking models. We also propose a multi-view depth-order loss while leveraging the classic scale-and-shift-invariant loss to further improve the final depth quality. The aligned depths initialize and supervise Gaussian splatting models for 4D reconstruction. Experiments on Ego-Exo4D and EgoHuman demonstrate that our improved depth alignment substantially benefits dynamic Gaussian-splatting-based reconstruction methods for novel-time/view synthesis and geometry accuracy/consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.11431v3">Remove360: Benchmarking Residuals After Object Removal in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-04
    </div>
    <details class="paper-abstract">
      An object can disappear from a 3D scene, yet still be detectable. Even after visual removal, modern vision models may infer what was originally present. In this work, we introduce a novel benchmark and evaluation framework to quantify semantic residuals, the unintended cues left behind after object removal in 3D Gaussian Splatting. We conduct experiments across a diverse set of indoor and outdoor scenes, showing that current methods often preserve semantic information despite the absence of visual geometry. Notably, even when removal is followed by inpainting, residual cues frequently remain detectable by foundation models. We also present Remove360, a real-world dataset of pre- and post-removal RGB captures with object-level masks. Unlike prior datasets focused on isolated object instances, Remove360 contains complex, cluttered scenes that enable evaluation of object removal in full-scene settings. By leveraging the ground-truth post-removal images, we directly assess whether semantic presence is eliminated and whether downstream models can still infer what was removed. Our results reveal a consistent gap between geometric removal and semantic erasure, exposing critical limitations in existing 3D editing pipelines and highlighting the need for privacy-aware removal methods that eliminate recoverable cues, not only visible geometry. Dataset and evaluation code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.11854v3">ComplicitSplat: Downstream Models are Vulnerable to Blackbox Attacks by 3D Gaussian Splat Camouflages</a></div>
    <div class="paper-meta">
      📅 2026-09-04
      | 💬 14 pages, 6 figures. Accepted to BMVC '26 Main Conference
    </div>
    <details class="paper-abstract">
      As 3D Gaussian Splatting (3DGS) gains rapid adoption in safety-critical tasks for efficient novel-view synthesis from static images, how might an adversary tamper images to cause harm? We introduce ComplicitSplat, the first attack that exploits standard 3DGS shading methods to create viewpoint-specific camouflage - colors and textures that change with viewing angle - to embed adversarial content in scene objects that are visible only from specific viewpoints and without requiring access to model architecture or weights. Our extensive experiments show that ComplicitSplat generalizes to successfully attack a variety of popular detector - both single-stage, multi-stage, and transformer-based models on both real-world capture of physical objects and synthetic scenes. To our knowledge, this is the first black-box attack on downstream object detectors using 3DGS, exposing a novel safety risk for applications like autonomous navigation and other mission-critical robotic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.05583v1">An overview of 3D Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2026-09-04
      | 💬 Accepted for presentation as a tutorial at SIBGRAPI 2026. Tutorial page: https://usmarcv.github.io/Tutorial-3DVLMs/
    </div>
    <details class="paper-abstract">
      Vision-Language Models (VLMs) are reshaping computer vision by aligning visual and textual embeddings, allowing models to recognize visual concepts and reason about them using natural language. Traditional 3D deep-learning models, however, are typically trained for specific tasks, such as classification, segmentation, or detection, and do not naturally support cross-modal retrieval from their embedding spaces using text or images as queries. To address this issue, Contrastive Language-Image Pretraining (CLIP)-based methods align 3D embeddings with pretrained image and text representations, giving rise to 3D Vision-Language Models (3D VLMs) that support zero-shot classification, cross-modal retrieval, and open-vocabulary recognition of 3D shapes. This tutorial provides an overview of 3D VLMs, ranging from basic definitions of 3D representations and their encoding into embeddings to cross-modal contrastive alignment, modern multimodal frameworks, and 3D Vision-Large Language Models (3D VLLMs). We present the main definitions of contrastive learning for multimodal embedding alignment and highlight recent advances in language-guided 3D Gaussian splatting, 3D shape generation, and embodied AI for robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.05255v1">Compact Neural Appearance Models for Efficient Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-04
      | 💬 Project page: https://fhahlbohm.github.io/efficient-gaussian-appearance
    </div>
    <details class="paper-abstract">
      Explicit primitive-based radiance fields such as 3D Gaussian Splatting typically model view-dependent appearance using low-order spherical harmonics (SH). Although efficient to evaluate, SH coefficients dominate per-primitive storage and memory traffic, while their band-limited basis restricts angular detail. We present a thorough, end-to-end comparison of SH and recent spherical appearance models and introduce an implicit alternative that decodes compact per-primitive latent codes using a tiny shared MLP. We integrate all models into the same optimized pipeline, fusing their forward and backward passes into a differentiable CUDA rasterizer and provide a portable WebGL viewer for laptop and mobile GPUs. Our evaluation across reconstruction quality, memory use, and optimization and rendering performance shows that recent spherical models offer the strongest overall quality-efficiency trade-off. Our neural representation is the most compact model evaluated and, compared to third-degree SH, reduces the per-primitive appearance footprint from 192 to 28 bytes, accelerates optimization by 1.3$\times$, while improving reconstruction quality. We further analyze how appearance parametrization shapes optimization, identifying differences in recovered geometry and the tendency of expressive models to absorb non-static scene content. Together, our framework and analysis provide practical guidance for replacing SH beyond what image metrics alone can capture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.05127v1">GradRig: Differentiable Weights for Skinned Gaussian Splat Deformation</a></div>
    <div class="paper-meta">
      📅 2026-09-04
    </div>
    <details class="paper-abstract">
      Skinned deformation is a common framework to turn a 3D shape from its rest pose into a dynamic pose through the deformation of a coarser kinematic structure, called rig. When applied to a 3D mesh, this rig only needs to displace vertices to deform the polygons that connect them. However, when deforming 3D Gaussian Splats, which do not provide connectivity information, rigidly transforming points is not enough to prevent the creation of holes when stretching shapes. In this paper, we use the spatial gradient of skinning weights to provide a full mesh-free deformation pipeline for Gaussian Splats, that more accurately stretches splats while remaining fully compatible with real-time rendering capabilities, which we demonstrate in a WebGL viewer. We present how we evaluate these gradients when the user creates the rig structure and propose an optional adaptive resampling scheme to split up splats that still produce artifacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.02006v3">ASTRA: Asynchronous Spatio-Temporal Reconstruction via Trajectory Alignment</a></div>
    <div class="paper-meta">
      📅 2026-09-04
      | 💬 We wish to withdraw this preprint because the current statistical analysis of the experimental data is incomplete and requires re-verification. We plan to submit a revised and thoroughly checked version in the near future
    </div>
    <details class="paper-abstract">
      Dynamic 3D scene reconstruction has made significant progress with multi-camera systems, often relying on temporally aligned observations across views. However, in real-world scenarios, temporal asynchrony among capturing devices remains a common limitation, leading to severe motion blur and geometric artifacts. Existing asynchronous reconstruction methods typically estimate temporal offsets through photometric supervision, but appearance matching provides weak temporal cues under large offsets and complex motions. We attribute this limitation to two critical issues: texture-induced collapse, where low-textured regions provide nearly vanishing alignment signals, and deformation-induced entanglement, where temporal errors are absorbed into distorted geometry or motion rather than being explicitly corrected. To address these issues, we propose ASTRA (Asynchronous Spatio-Temporal Reconstruction via Trajectory Alignment), a framework that introduces 2D motion trajectories as explicit, texture-robust supervision for asynchronous dynamic reconstruction. Instead of synchronizing cameras solely through rendered color residuals, ASTRA jointly optimizes temporal offsets and dynamic 3D representations by aligning the projected motion of reconstructed 3D points with observed 2D trajectories, while using dynamic and certainty masking to suppress unreliable trajectory constraints. Extensive experiments on different dynamic Gaussian Splatting backbones show that ASTRA preserves high-frequency spatial details and sustains strong robustness even under severe asynchrony with up to 25-frame offsets, achieving approximately 1.4 dB PSNR improvement, reducing temporal-offset MAE by 54.0%, and nearly quadrupling the synchronization success rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.04602v1">NavArena: Automated Construction of Goal-Oriented Navigation Benchmarks from 3D Gaussian Splatting Reconstructions</a></div>
    <div class="paper-meta">
      📅 2026-09-04
    </div>
    <details class="paper-abstract">
      Fixed 3D Gaussian Splatting (3DGS) reconstructions provide realistic novel views but lack the traversability constraints, valid goals, and closed-loop protocols required for navigation evaluation. We introduce NavArena, an automated framework that transforms fixed 3DGS reconstructions into benchmarks for goal-oriented visual navigation. NavArena integrates a frozen 3DGS model for egocentric RGB-D rendering, an occupancy costmap derived from Gaussian density and height statistics for reachability and collision queries, and semantic goal candidates lifted from multi-view open-vocabulary masks. These components support the automatic generation and unified closed-loop evaluation of goal-oriented navigation episodes. Across more than 2{,}000 scenes, NavArena generates 22.2 million expert trajectories. Spatial and semantic evaluations assess the derived navigation representations, while policy rollouts demonstrate the diagnostic value of the unified evaluation protocol. NavArena enables scalable and reproducible navigation evaluation on large-scale 3DGS reconstructions, and all benchmark-generation tools, evaluation protocols, and derived assets will be released publicly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03931v1">Sparse auto-regressive modeling for scene generation from multi-view images</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Accepted at ECCVV 2026
    </div>
    <details class="paper-abstract">
      Generating complete 3D scenes from sparse, unconstrained views is a fundamental challenge in 3D vision which requires reasoning beyond observed content while remaining computationally tractable. Existing feed-forward reconstruction methods are inherently limited to content visible in the input images, while 3D generative modeling is hindered by the high computational cost of dense volumetric representations and the scarcity of large-scale 3D supervision. We introduce SPAR3S, a sparse voxel-aligned 3D latent generative model for conditional scene completion without requiring ground-truth 3D data for supervision. Our key insight is to formulate 3D scene generation in a structured, compact, voxel-aligned 3D latent space where only occupied voxels are represented. We learn this sparse latent space directly from multi-view images using photometric supervision via differentiable 3D Gaussian Splatting. Given a partial set of observed voxels encoded from sparse input views, scene completion reduces to predicting the missing latent tokens and their spatial support within the voxel grid. To this end, we train a masked autoregressive transformer that jointly models voxel occupancy and latent token values, enabling efficient and spatially consistent generation of unseen regions. We demonstrate the effectiveness of our method on synthetic indoor scenes, achieving higher novel-view quality than prior work. We further validate its generalization on RealEstate10k, highlighting its applicability to real-world data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03897v1">Reparametrizing 3D Gaussian Splatting for Real-Time Palette-based Color and Luminance Editing</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 11 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Professional color editing requires precise control over both color (hue and saturation) and lightness, ideally through separate, independent controls. We present a real-time interactive color editing framework for 3D Gaussian Splatting that supports palette-based recoloring, per-palette tone curves for color-aware luminance adjustment, and pixel-level color constraints. Rather than training a new representation from scratch, we reparameterize the spherical harmonics of a pretrained vanilla 3DGS to encode view-dependent palette weights. We simultaneously solve for weights and palette colors via a loss based on image-space sparsity. Luminance editing is realized as a per-pixel weight shift along the achromatic axis, which we show is equivalent to a per-pixel palette-aware luminance edit. This view-space formulation addresses a core limitation of prior primitive-space methods, where alpha-blending breaks per-Gaussian sparsity and causes edits to bleed into unintended regions. Our edits run in tens of milliseconds via an iteratively reweighted least squares and damped block-coordinate descent that couples tone curves and palette shifts under view-space sparsity. Our representation can be efficiently baked back into a vanilla 3DGS, preserving compatibility with standard viewers. We demonstrate sparser, more localized edits than prior palette-based 3DGS methods, while enabling independent luminance control per palette color and view-consistent pixel-level constraints, capabilities previously unavailable for 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.21304v3">F4Splat: Feed-Forward Predictive Densification for Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Project Page: $\href{https://mlvlab.github.io/F4Splat}{\text{this http URL}}$
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting methods enable single-pass reconstruction and real-time rendering. However, they typically adopt rigid pixel-to-Gaussian or voxel-to-Gaussian pipelines that uniformly allocate Gaussians, leading to redundant Gaussians across views. Moreover, they lack an effective mechanism to control the total number of Gaussians while maintaining reconstruction fidelity. To address these limitations, we present F4Splat, which performs Feed-Forward predictive densification for Feed-Forward 3D Gaussian Splatting, introducing a densification-score-guided allocation strategy that adaptively distributes Gaussians according to spatial complexity and multi-view overlap. Our model predicts per-region densification scores to estimate the required Gaussian density and allows explicit control over the final Gaussian budget without retraining. This spatially adaptive allocation reduces redundancy in simple regions and minimizes duplicate Gaussians across overlapping views, producing compact yet high-quality 3D representations. Extensive experiments demonstrate that our model achieves superior novel-view synthesis performance compared to prior uncalibrated feed-forward methods, while using significantly fewer Gaussians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03657v1">Rethinking 3D Noise: Learning 3D-Aware Video Priors via Optimization-Free Morphological Perturbations</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      3D scene representations like NeRF and 3D Gaussian Splatting (3DGS) suffer severe artifacts in sparse-view settings. Recent generative 3D artifact fixers attempt to address this, but rely on paired corrupted and clean renders requiring costly, per-scene reconstructions across varying view configurations. While 2D image augmentations act as instant regularizers, no explicit equivalents exist for 3D representations to preserve spatial consistency across views, an essential property for 3D-aware training. We propose 3D Morphological Perturbations as an optimization-free regularizer that preserves spatial consistency. Leveraging explicit 3DGS, we treat each Gaussian as a fundamental building block - analogous to a 2D pixel - and apply perturbations across its morphological parameter space via scale, rotation, and pruning. Our method eliminates per-scene 3DGS optimization loops from dataset curation while enabling models to learn stronger geometric priors than sparse-view baselines in diagnostic ablations conducted on a lightweight video diffusion sandbox. Scaled to a 14B-parameter video model via ControlNet, our approach maintains visual fidelity while reducing mean depth error by 12.5% over state-of-the-art image-to-image 3D artifact refiners, ultimately boosting downstream robotics policy success rates by up to 8.0% across 3 of 4 manipulation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03613v1">TileGS: Tile-Local Depth Binning for Gaussian Splatting Rasterization</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Real-time 3D Gaussian Splatting (3DGS) achieves high rendering quality, but standard rasterization still traverses a globally sorted tile stream that creates long per-tile ranges and heavy geometry-attribute traffic. We present TileGS, a tile-local reorganization of Gaussian splatting. TileGS turns each long tile range into a sequence of shorter depth-local ranges, rasterizes those ranges in front-to-back order, and applies selective repair where coarse ordering is insufficient to match baseline compositing. Across a 9-scene benchmark on desktop and laptop Ada GPUs, our default No-GW (No Geometry-Write) variant delivers a mean 1.44x raster-kernel speedup on RTX 4090 and mean end-to-end frame speedups of 1.069x on RTX 4090 and 1.094x on RTX 1000 Ada over gsplat--a widely used optimized open-source 3DGS implementation--while matching the gsplat output up to numerical noise (|Delta PSNR| < 0.001 dB, |Delta SSIM| < 0.001, |Delta LPIPS| < 0.001). Full-suite RTX 4090 Nsight Compute profiling reveals TileGS is faster despite lower SM throughput, lower active-warp occupancy, and higher DRAM traffic, while total SASS thread instructions fall by 1.26x. Source-attributed profiling confirms that geometry attributes dominate the remaining memory pressure (85.8% of total raster traffic and 88.6% of excess sectors). Together, these counters support the interpretation that TileGS improves raster performance by reducing effective raster traversal work, rather than by reducing byte volume, improving coalescing, increasing occupancy, or directly reducing measured warp divergence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.15677v2">Camera Splatting for Continuous View Optimization</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 12 pages, Computer Graphics Forum (Proc. Pacific Graphics 2026)
    </div>
    <details class="paper-abstract">
      We propose Camera Splatting, a novel view optimization framework for novel view synthesis. Each camera is modeled as a 3D Gaussian, referred to as a camera splat, and virtual cameras, termed point cameras, are placed at 3D points sampled near the surface to observe the distribution of camera splats. View optimization is achieved by continuously and differentiably refining the camera splats so that desirable target distributions are observed from the point cameras, in a manner similar to the original 3D Gaussian splatting. Compared to the Farthest View Sampling (FVS) approach, our optimized views demonstrate superior performance in capturing complex view-dependent phenomena, including intense metallic reflections and intricate textures such as text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03534v1">TruncGradGS: Improved 3D Gaussian Splatting via Truncated Gradient Updates</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Accepted at Pacific Graphics 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has become a de facto scene representation for novel view synthesis, yet robustly learning 3D Gaussian primitives from visual input remains challenging. Standard optimization relies on gradient-based updates, but a common issue is the gradient vanishing phenomenon: a pixel far from a Gaussian primitive often has diminishing gradient magnitudes to influence primitive attributes, resulting in suboptimal scene reconstruction. In this paper, we propose a method to address gradient vanishing with a piecewise truncated gradient formulation that improves the optimization stability and robustness to initializations. We show that our method consistently improves 3D Gaussian Splatting with random and COLMAP initializations while being generalizable across static and dynamic Gaussian Splatting. As a by-product, we also examine the limitations of current benchmarks for dynamic scenes, and introduce a novel dataset for benchmarking dynamic Gaussian Splatting using synthetic 3D scenes. We demonstrate the effectiveness of our method in both static and dynamic settings for the public benchmarks and our proposed dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07604v2">EmbedTalk: Talking Head Synthesis using Gaussian Embeddings</a></div>
    <div class="paper-meta">
      📅 2026-09-03
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Deformable 3D Gaussian Splatting (3DGS) has emerged as a popular method for real-time talking head synthesis, offering high-quality renderings at low latency. Tri-planes are a common choice for encoding Gaussians prior to deformation since they provide a compact and continuous representation. However, tri-plane encodings are limited by grid resolution and approximation errors introduced by projecting 3D volumetric fields onto 2D subspaces. Recent work has demonstrated the effectiveness of per-Gaussian embeddings for driving temporal deformations in 4D scene reconstruction. We introduce EmbedTalk, which leverages these embeddings to model speech-driven facial deformations for talking head synthesis. Comprehensive experiments show that EmbedTalk improves rendering quality, lip synchronisation, and motion consistency over previous 3DGS-based methods, while remaining competitive with state-of-the-art generative models. Replacing tri-plane features with embeddings also yields significantly more compact models that achieve 60+ FPS on a laptop GPU (RTX 2060 6 GB). Our code will be placed in the public domain on acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.03447v1">STARS-GS: Structure-Aware Regularized Gaussian Splatting for Large-Scale Aerial Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-09-03
    </div>
    <details class="paper-abstract">
      Large-scale 3D surface reconstruction from aerial imagery is fundamental to geospatial mapping and urban modeling. Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated considerable potential for this task. However, existing methods still face three major challenges in large and complex scenes: scene partitioning may split continuous scene elements across independently optimized sub-regions; geometric constraints mainly focus on the attributes of individual Gaussians while overlooking their local organization; and uniform regularization struggles to accommodate heterogeneous geometric structures. To address these issues, we propose STARS-GS, a structure-aware 3DGS framework for large-scale surface reconstruction. First, we introduce a structure-aware scene partitioning strategy that better preserves continuous scene structures during partitioning and reduces cross-region geometric inconsistencies and stitching artifacts through boundary refinement. Second, we develop neighborhood-aware Gaussian organization that extends geometric constraints from individual primitives to their neighborhood organization, encouraging Gaussians to better conform to local surface geometry. Third, we introduce adaptive surface regularization that adjusts the regularization strength according to local geometric characteristics, promoting geometric consistency in structured regions while preserving plausible variations in unstructured regions. Extensive experiments on large-scale aerial photogrammetry benchmarks demonstrate that STARS-GS consistently outperforms the evaluated Gaussian-based methods in surface reconstruction. It increases the average F1-score from 0.640 for the second-best method to 0.698, corresponding to a relative improvement of approximately 9.1\%, demonstrating effective improvements in geometric accuracy and surface completeness.
    </details>
</div>
