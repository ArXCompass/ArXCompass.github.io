# gaussian splatting - 2026_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22809v1">Sensor2Sensor: Cross-Embodiment Sensor Conversion for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      Robust training and validation of Autonomous Driving Systems (ADS) require massive, diverse datasets. Proprietary data collected by Autonomous Vehicle (AV) fleets, while high-fidelity, are limited in scale, diversity of sensor configurations, as well as geographic and long-tail-behavioral coverage. In contrast, in-the-wild data from sources like dashcams offers immense scale and diversity, capturing critical long-tail scenarios and novel environments. However, this unstructured, in-the-wild video data is incompatible with ADS expecting structured, multi-modal sensor inputs for validation and training. To bridge this data gap, we propose Sensor2Sensor, a novel generative modeling paradigm that translates in-the-wild monocular dashcam videos into a high-fidelity, multi-modal sensor suite (AV logs) comprising multi-view camera images and LiDAR point clouds. A core challenge is the lack of paired training data. We address this by converting real AV logs into dashcam-style videos via 4D Gaussian Splatting (4DGS) reconstruction and novel-view rendering. Sensor2Sensor then utilizes a diffusion architecture to perform the generative conversion. We perform comprehensive quantitative evaluations on the fidelity and realism of the generated sensor data. We demonstrate Sensor2Sensor's practical utility by converting challenging in-the-wild internet and dashcam footage into realistic, multi-modal data formats, further unlocking vast external data sources for AV development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.02784v2">HumanSplatHMR: Closing the Loop Between Human Mesh Recovery and Gaussian Splatting Avatar</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Project page: https://scottyehengz.github.io/HumanSplat/
    </div>
    <details class="paper-abstract">
      Accurately recovering human pose and appearance from video is an essential component of scene reconstruction, with applications to motion capture, motion prediction, virtual reality, and digital twinning. Despite significant interest in building realistic human avatars from video, this paper demonstrates that existing methods do not accurately recover the 3D geometry of humans. ViT-based approaches are not consistently reliable and can overfit to 2D views, while NeRF- and Gaussian Splatting-based avatars treat pose and appearance separately, limiting rendering generalization to new poses. To resolve these shortcomings, this paper proposes HumanSplatHMR, a joint optimization framework that refines 3D human poses while simultaneously learning a high-fidelity avatar for novel-view and novel-pose synthesis. Our key insight is to close the loop between geometric pose estimation and differentiable rendering. Unlike prior human avatar methods that rely on accurate human pose obtained through motion capture systems or offline refinement, which are impractical in in-the-wild scenarios, our approach uses only human mesh estimates from a state-of-the-art human pose estimator to better reflect real-world conditions. Therefore, instead of using the human pose only as a deformation prior, HumanSplatHMR backpropagates photometric, segmentation, and depth losses through a differentiable renderer to the pose parameters and global position. This coupling refines the global 3D pose over time, improving accuracy and alignment while producing better renderings from novel views. Experiments show consistent improvements over pose recovery baselines that omit image-level refinement and avatar baselines that decouple pose estimation from avatar reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22536v1">SpaceDG: Benchmarking Spatial Intelligence under Visual Degradation</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have made rapid progress in spatial intelligence, yet existing spatial reasoning benchmarks largely assume pristine visual inputs and overlook the degradations that commonly occur in real-world deployment, such as motion blur, low light, adverse weather, lens distortion, and compression artifacts. This raises a fundamental question: how robust is the spatial intelligence of current MLLMs when visual observations are imperfect? To answer this question, we introduce SpaceDG, the first large-scale dataset for degradation-aware spatial understanding. It is constructed with a physically grounded degradation synthesis engine that embeds degradation formation process into 3D Gaussian Splatting (3DGS) rendering, enabling realistic simulation of nine degradation types. The resulting dataset contains approximately 1M QA pairs from nearly 1,000 indoor scenes. We further introduce SpaceDG-Bench, an human-verified benchmark with 1,102 questions spanning 11 reasoning categories and 9 visual degradation types, yielding over 10K VQA instances. Evaluating 25 open- and closed-source MLLMs reveals that visual degradations consistently and substantially impair spatial reasoning, exposing a critical robustness gap. Finally, we show that finetuning on SpaceDG markedly improves degradation robustness and can even surpass human performance under degraded conditions without any performance drop on clean images, highlighting the promise of degradation-aware training for robust spatial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22342v1">4D-GSW: Kinematic-Aware Spatio-Temporal Consistent Watermarking for 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 9 pages main paper, 7 figures, 18 pages in total
    </div>
    <details class="paper-abstract">
      While 4D Gaussian Splatting (4DGS) has revolutionized high-fidelity dynamic reconstruction, safeguarding the intellectual property of these assets remains an open challenge. Conventional steganographic techniques often neglect the underlying kinematic manifolds, triggering non-physical artifacts such as severe temporal flickering and "FVD collapse". To address this, we propose \textbf{4D-GSW}, a kinematic-aware watermarking framework designed to embed robust copyright information while preserving high spatio-temporal consistency. Unlike prior 4D steganography that primarily focuses on opacity-guided invisibility, our approach explicitly addresses the physical coherence of motion trajectories. We introduce a \textbf{Spatio-Temporal Curvature (STC)} metric to identify "Dynamic Instants," adaptively gating watermark gradient injection to shield critical motion manifolds from non-physical perturbations. To ensure global coherence across complex deformations, we formulate a joint \textbf{HMM-MRF energy minimization} model that synchronizes watermark phases within both temporal trajectories and spatial neighborhoods. Furthermore, an \textbf{anisotropic gradient routing} mechanism ensures that watermark embedding remains strictly decoupled from photometric reconstruction fidelity. Extensive experiments have demonstrated the superior performance of our method in robustly hiding watermarks while resisting various attacks and maintaining high rendering quality and spatiotemporal consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22190v1">No Pose, No Problem in 4D: Feed-Forward Dynamic Gaussians from Unposed Multi-View Videos</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 https://bralani.github.io/nopo4d_html/
    </div>
    <details class="paper-abstract">
      Recent feed-forward 3D gaussian splatting methods have made dramatic progress on individual aspects of 3D scene reconstruction, but no existing method jointly addresses dynamic content, multi-view input, and unknown camera poses in a single feed-forward pass. Methods that handle dynamics either require accurate camera poses or accept only monocular input; pose-free multi-view methods address only static scenes; and per-scene optimization methods bridge some of these gaps but at minutes-to-hours cost per scene. We introduce NoPo4D, the first feed-forward system that addresses this empty quadrant. Building on a pretrained geometry backbone and recent 4D Gaussian frameworks, NoPo4D introduces a velocity decomposition that splits Gaussian motion into per-pixel image-plane shifts and depth changes, allowing direct supervision from pseudo ground-truth optical flow on the 2D component. This sidesteps both the differentiable rendering that couples prior posed methods to pose accuracy and the 3D motion ground truth that prior pose-free methods require. The system is rounded out by a bidirectional motion encoder for cross-view and cross-frame feature aggregation, and view-dependent opacity that mitigates cross-view and cross-timestep Gaussian misalignments. On four multi-view dynamic benchmarks, NoPo4D consistently outperforms prior feed-forward baselines, and with an optional post-optimization stage surpasses per-scene optimization methods, while running orders of magnitude faster.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22147v1">Flow-based Gaussian Splatting for Continuous-Scale Remote Sensing Image Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      High-resolution remote sensing images (RSIs) are crucial for Earth observation applications, yet acquiring them is often limited by sensor constraints and costs. In recent years, generative super-resolution (SR) methods, particularly diffusion models, have made significant progress. However, they typically require slow iterative inference with 40--1000 steps and exhibit limited flexibility in continuous-scale SR settings. To address these issues, we propose FlowGS, a generative reconstruction framework for arbitrary-scale SR of RSIs. FlowGS models the high-frequency detail representations between high- and low-resolution images and learns a continuous probability flow from noise to detail priors via flow matching (FM) constrained by shortcut consistency, thereby reducing generative complexity and improving inference efficiency. Additionally, we employ 2D Gaussian splatting to construct a continuous feature field, thereby enabling flexible reconstruction at arbitrary query locations. Experimental results show that FlowGS delivers competitive perceptual quality compared with existing methods in both continuous-scale and fixed-scale SR settings, with substantially improved inference efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22069v1">TWINGS: Thin Plate Splines Warp-aligned Initialization for Sparse-View Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Accepted to CVPR 2025, Project page: https://sandokim.github.io/twings/
    </div>
    <details class="paper-abstract">
      Novel view synthesis from sparse-view inputs poses a significant challenge in 3D computer vision, particularly for achieving high-quality scene reconstructions with limited viewpoints. We introduce TWINGS, a framework that enhances 3D Gaussian Splatting (3DGS) by directly addressing point sparsity. We employ Thin Plate Splines (TPS), a smooth non-rigid deformation model that minimizes bending energy to estimate a globally coherent warp from control-point correspondences, to align backprojected points from estimated depth with triangulated 3D control points, yielding calibrated backprojected points. By sampling these calibrated points near the control points, TWINGS provides a fast and geometrically accurate initialization for 3DGS, ultimately improving structural detail preservation and color fidelity in reconstructed scenes. Extensive experiments on DTU, LLFF, and Mip-NeRF360 demonstrate that TWINGS consistently outperforms existing methods, delivering detailed and accurate reconstructions under sparse-view scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07287v2">SplatWeaver: Learning to Allocate Gaussian Primitives for Generalizable Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Project Page: https://yecongwan.github.io/SplatWeaver/
    </div>
    <details class="paper-abstract">
      Generalizable novel view synthesis aims to render unseen views from uncalibrated input images without requiring per-scene optimization. Recent feed-forward approaches based on 3D Gaussian Splatting have achieved promising efficiency and rendering quality. However, most of them assign a fixed number of Gaussians to each pixel or voxel, ignoring the spatially varying complexity of real-world scenes. Such uniform allocation often wastes Gaussian primitives in smooth regions while providing insufficient capacity for fine structures, complex geometry, and high-frequency details. This motivates us to predict region-dependent primitive cardinalities rather than impose a fixed primitive budget everywhere, enabling a more expressive 3D scene representation. Therefore, we propose SplatWeaver, a generalizable novel view synthesis framework that is able to dynamically allocate Gaussian primitives over different regions in a feed-forward manner. Specifically, SplatWeaver introduces cardinality Gaussian experts and a pixel-level routing scheme, wherein each expert specializes in producing a specific number of primitives from 0 to M, and the routing scheme coordinates these experts to adaptively determine how many Gaussian primitives should be allocated to each spatial location. Moreover, SplatWeaver incorporates a high-frequency prior with attendant guidance module and routing regularization to stabilize expert selection and promote complexity-aware allocation. By leveraging high-frequency cues, the routing process is encouraged to assign more Gaussian primitives to fine structures and textured regions, while suppressing redundancy in smooth areas. Extensive experiments across diverse scenarios show that SplatWeaver consistently outperforms state-of-the-art methods, delivering more faithful novel-view renderings with fewer Gaussian primitives. Project Page: https://yecongwan.github.io/SplatWeaver/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22020v1">ForeSplat: Optimization-Aware Foresight for Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) models offer fast single-pass reconstruction,but scaling them to match per-scene optimization quality is fundamentally hindered by the scarcity of large-scale 3D annotations.A practical compromise is predict-then-refine,where post-prediction optimization compensates for the limited capacity of the feed-forward network.However,standard feed-forward 3DGS is trained solely for zero-step rendering error,ignoring whether its output constitutes a good initialization for the downstream optimizer.We present ForeSplat,an optimization-aware training framework that equips feed-forward 3DGS models to produce initializations explicitly designed for rapid,effective refinement.By offloading part of the scene-modeling burden to the optimizer,ForeSplat substantially reduces the capacity pressure on the feed-forward model,making high-quality reconstruction feasible even with compact networks.At its core is MetaGrad,a lightweight multi-anchor meta-gradient training rule that bypasses costly higher-order differentiation through the 3DGS optimizer.MetaGrad unrolls a short inner-loop refinement trajectory,samples anchor states,and back-propagates aggregated first-order gradients to the prediction head as a surrogate optimization-aware signal.This fine-tuning adds no inference cost and enables high-quality reconstruction within seconds after a few refinement steps.We instantiate ForeSplat on diverse backbones,including AnySplat,Pi3X,and a distilled variant tailored for edge deployment.Across all tested architectures,a ForeSplat-trained initialization converges in fewer refinement steps and reaches a higher peak reconstruction quality than its vanilla counterpart,even fully converged.The framework consistently bridges the gap between amortized prediction and per-scene optimization,establishing a practical path toward lightweight,high-fidelity 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01466v2">SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Accepted as a Spotlight paper at ICML 2026; camera-ready version
    </div>
    <details class="paper-abstract">
      Although multi-modal learning has advanced point cloud completion, the theoretical mechanisms remain unclear. Recent works attribute success to the connection between modalities, yet we identify that standard hard projection severs this connection: projecting a sparse point cloud onto the image plane yields an extremely sparse support, which hinders visual prior propagation, a failure mode we term Cross-Modal Entropy Collapse. To address this practical limitation, we propose SplAttN, which replaces hard projection with Differentiable Gaussian Splatting to produce a dense, continuous image-plane representation. By reformulating projection as continuous density estimation, SplAttN avoids collapsed sparse support, facilitates gradient flow, and improves cross-modal connection learnability. Extensive experiments show that SplAttN achieves state-of-the-art performance on PCN and ShapeNet-55/34. Crucially, we utilize the real-world KITTI benchmark as a stress test for multi-modal reliance. Counter-factual evaluation reveals that while baselines degenerate into unimodal template retrievers insensitive to visual removal, SplAttN maintains a robust dependency on visual cues, validating that our method establishes an effective cross-modal connection. Code is available at https://github.com/zay002/SplAttN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21935v1">Learning to Evolve: Multi-modal Interactive Fields for Robust Humanoid Navigation in Dynamic Environments</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Accepted by Robotics: Science and Systems 2026
    </div>
    <details class="paper-abstract">
      Safe manipulation-oriented navigation for humanoid robots requires scene memory that remains reliable under locomotion-induced perceptual distortion, environmental changes, and interaction-level geometric safety constraints. Existing semantic mapping and scene-graph systems are difficult to deploy directly in this setting because they often assume stable camera trajectories, static environments, or coarse object geometry. We introduce the Multi-modal Interactive Field (MIF), a humanoid-oriented system that integrates confidence-aware semantic 3D Gaussian Splatting, discrepancy-triggered spatial memory updates, and task-driven geometric reconstruction within a closed-loop perception-adaptation pipeline. MIF couples three fields: an uncertainty-aware 3DGS Appearance Field that suppresses gait-induced blur, a Spatial Field that maintains topological memory, and a Geometry Field that supports Interaction Pose Safety (IPS) before manipulation. A discrepancy detection score is introduced to separate locomotion-induced false-positive changes from persistent changes and updates only locally inconsistent regions. On a Unitree-G1 humanoid in a real dynamic office, MIF improves relocation success in non-static environments from 12% to 94% compared with static scene-graph memory, while reducing semantic memory footprint by 91.4% through feature distillation for practical online operation. Project page and code: https://ziya-jiang.github.io/MIF-homepage/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17855v2">Accelerating 3D Gaussian Splatting using Tensor Cores</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a leading technique for real-time neural rendering and 3D scene reconstruction, but its rendering cost remains too high for many latency-sensitive scenarios. In particular, the rasterization stage in 3DGS dominates end-to-end rendering time, during which the renderer repeatedly evaluates each Gaussian's contribution to each covered pixel, making this stage compute-bound. At the same time, modern GPUs provide high-throughput Tensor Cores for low-precision matrix operations, yet existing 3DGS systems execute rasterization entirely on CUDA cores and leave Tensor Cores idle. We find that 3DGS rendering can be executed in FP16 with negligible quality degradation, suggesting a promising opportunity for Tensor Core acceleration. However, exploiting Tensor Cores for 3DGS is non-trivial because rasterization does not naturally match their execution model. Existing 3DGS rasterization is expressed as irregular per-pixel scalar operations, whereas Tensor Cores require dense, regular, and reuse-rich matrix workloads. Moreover, conventional tile-by-tile execution fails to exploit Gaussian reuse across neighboring tiles, resulting in repeated data loading and thus high data movement overhead. To this end, we present TensorGS, a 3DGS acceleration framework using Tensor Cores. TensorGS tensorizes the dominant rasterization computation into Tensor-Core-compatible matrix operations and introduces cross-tile grouping to improve Gaussian reuse, amortize overhead, and increase Tensor Core utilization. Experimental results show that TensorGS improves end-to-end rendering performance by 1.65$\times$ while preserving image quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21112v1">RCGDet3D: Rethinking 4D Radar-Camera Fusion-based 3D Object Detection with Enhanced Radar Feature Encoding</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      4D automotive radar is indispensable for autonomous driving due to its low cost and robustness, yet its point cloud sparsity challenges 3D object detection. Existing 4D radar-camera fusion methods focus on complex fusion strategies, trading inference speed for marginal gains. This trade-off hinders real-time deployment due to heavy computation on dense feature maps. In contrast, feature extraction from sparse radar points is less time-consuming but remains under-explored. This work uncovers that simply enhancing radar feature extraction can achieve comparable or even higher performance than elaborate fusion modules, while maintaining real-time performance. Based on this finding, we propose RCGDet3D, which centers on radar feature encoding and simplifies multi-modal fusion. Its encoder inherits from the efficient Gaussian Splatting-based Point Gaussian Encoder (PGE) in RadarGaussianDet3D with two key improvements. First, the Ray-centric PGE (R-PGE) predicts Gaussian attributes in ray-aligned coordinate systems before unifying them to Bird's-Eye View (BEV) space, significantly improving geometric consistency and reducing learning difficulty by decoupling the coordinate transformation from representation learning. Second, a Semantic Injection (SI) module incorporates visual cues from images, producing more geometrically accurate and semantically enriched radar features. Experiments on View-of-Delft (VoD) and TJ4DRadSet show that RCGDet3D outperforms state-of-the-art methods in both accuracy and speed, setting a new benchmark for real-time deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.14978v2">E2GS: Event Enhanced Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 7pages, Accepted at ICIP 2024
    </div>
    <details class="paper-abstract">
      Event cameras, known for their high dynamic range, absence of motion blur, and low energy usage, have recently found a wide range of applications thanks to these attributes. In the past few years, the field of event-based 3D reconstruction saw remarkable progress, with the Neural Radiance Field (NeRF) based approach demonstrating photorealistic view synthesis results. However, the volume rendering paradigm of NeRF necessitates extensive training and rendering times. In this paper, we introduce Event Enhanced Gaussian Splatting (E2GS), a novel method that incorporates event data into Gaussian Splatting, which has recently made significant advances in the field of novel view synthesis. Our E2GS effectively utilizes both blurry images and event data, significantly improving image deblurring and producing high-quality novel view synthesis. Our comprehensive experiments on both synthetic and real-world datasets demonstrate our E2GS can generate visually appealing renderings while offering faster training and rendering speed (140 FPS). Our code is available at https://github.com/deguchihiroyuki/E2GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20872v1">CAdam: Context-Adaptive Moment Estimation for 3D Gaussian Densification in Generative Distillation</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted to SIGGRAPH 2026 Conference Papers. 12 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Adaptive densification is the engine of 3D Gaussian Splatting (3DGS). However, when transposed to the optimization-based Generative Distillation paradigm, this reconstruction-native mechanism reveals fundamental limitations, resulting in inefficient representations cluttered with redundant primitives. We diagnose this failure as a Densification Dilemma stemming from the stochastic nature of generative guidance: the standard magnitude-based accumulation indiscriminately aggregates transient noise alongside geometric signals, making it difficult to strike a balance between over-densification and under-fitting. To resolve this, we introduce Context-Adaptive Moment Estimation (CAdam), a novel framework that reinterprets densification as a statistically grounded signal verification problem. CAdam leverages the first moment of gradients to exploit the interference principle, where stochastic fluctuations cancel out via destructive interference while consistent geometric drifts accumulate via constructive interference, effectively disentangling the underlying signal from the generative noise floor. This is further augmented by a quantile-based context awareness and an intrinsic Signal-to-Noise Ratio (SNR) gating mechanism, which ensure robust adaptation across optimization stages and enable the soft termination of densification. Extensive experiments across diverse objectives (SDS, ISM, VFDS) and strong generative 3DGS backbones show that CAdam reduces Gaussian count by 85%-97% relative to standard densification while preserving overall comparable perceptual quality. These results highlight signal-aware density control as a practical way to improve memory efficiency in optimization-based generative distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20820v1">AIR: Amortized Image Reconstruction Framework for Self-Supervised Feed-Forward 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 preprint version
    </div>
    <details class="paper-abstract">
      2D Gaussian splatting provides an efficient explicit representation for image reconstruction, but existing methods still require costly per-image iterative optimization or rely on handcrafted priors for primitive allocation. We present AIR, a self-supervised feed-forward framework that amortizes iterative Gaussian fitting into a single network pass, eliminating per-image test-time optimization. AIR adopts a stage-wise residual architecture that progressively predicts additional Gaussian primitives from reconstruction residuals, together with an explicit Stage Control mechanism that activates new primitives only in under-reconstructed regions. A Predict--Optimize--Distill training strategy stabilizes multi-stage prediction by distilling short-horizon optimized Gaussian increments back into the predictor. The stabilized predictor is then jointly finetuned across stages and equipped with an image-adaptive quantizer for compact Gaussian storage. Experiments on Kodak and DIV2K show that AIR achieves better reconstruction quality than representative Gaussian-based baselines while reducing encoding time to 160--300\,ms. Code: https://github.com/whoiszzj/AIR.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.13482v2">Improving 3D Gaussian Splatting Compression by Scene-Adaptive Lattice Vector Quantization</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted by IEEE TIP. Code available at https://github.com/hxu160/SALVQ
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is rapidly gaining popularity for its photorealistic rendering quality and real-time performance, but it generates massive amounts of data. Hence compressing 3DGS data is necessary for the cost effectiveness of 3DGS models. Recently, several anchor-based neural compression methods have been proposed, achieving good 3DGS compression performance. However, they all rely on uniform scalar quantization (USQ) due to its simplicity. A tantalizing question is whether more sophisticated quantizers can improve the current 3DGS compression methods with very little extra overhead and minimal change to the system. The answer is yes by replacing USQ with lattice vector quantization (LVQ). To better capture scene-specific characteristics, we optimize the lattice basis for each scene, improving LVQ's adaptability and R-D efficiency. This scene-adaptive LVQ (SALVQ) strikes a balance between the R-D efficiency of vector quantization and the low complexity of USQ. SALVQ can be seamlessly integrated into existing 3DGS compression architectures, enhancing their R-D performance with minimal modifications and computational overhead. Moreover, by scaling the lattice basis vectors, SALVQ can dynamically adjust lattice density, enabling a single model to accommodate multiple bit rate targets. This flexibility eliminates the need to train separate models for different compression levels, significantly reducing training time and memory consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20566v1">Conflict-Aware Active Perception and Control in 3D Gaussian Splatting Fields via Control Barrier Functions</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Project website: https://sircesoc.github.io/Conflict_Aware_Active_Perception/
    </div>
    <details class="paper-abstract">
      Active perception in uncertain environments requires robots to navigate safely while acquiring informative observations to reduce map uncertainty. These objectives inherently conflict, as informative viewpoints often lie near uncertain regions with higher collision risk. To address this challenge, we develop a conflict-aware active perception and control framework for robotic systems operating in environments represented by 3D Gaussian Splatting (3DGS). Safety is enforced using a Control Barrier Function (CBF) derived from an Average Value-at-Risk AV@R collision-risk metric that accounts for geometric uncertainty and guarantees forward invariance of a safe set. To improve perception, we propose a risk-aware Expected Information Gain (EIG) formulation for selecting the next-best-view and introduce perception barrier functions that align the camera orientation with the local information-ascent direction. To obtain a tractable formulation for these conflicting safety and perception objectives, we propose a unified safety-critical, perception-aware quadratic program that enforces safety as a hard constraint while relaxing perception constraints through slack variables. Simulation results demonstrate that the proposed method improves both safety and information acquisition compared to existing 3DGS-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20150v1">TideGS: Scalable Training of Over One Billion 3D Gaussian Splatting Primitives via Out-of-Core Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Accepted to ICML 2026 as Spotlight. Website: https://sponge-lab.github.io/TideGS
    </div>
    <details class="paper-abstract">
      Training 3D Gaussian Splatting (3DGS) at billion-primitive scale is fundamentally memory-bound: each Gaussian primitive carries a large attribute vector, and the aggregate parameter table quickly exceeds GPU capacity, limiting prior systems to tens of millions of Gaussians on commodity single-GPU hardware. We observe that 3DGS training is inherently sparse and trajectory-conditioned: each iteration activates only the Gaussians visible from the current camera batch, so GPU memory can serve as a working-set cache rather than a persistent parameter store. Building on this insight, we introduce TideGS, an out-of-core training framework that manages parameters across an SSD-CPU-GPU hierarchy via three synergistic techniques: block-virtualized geometry for SSD-aligned spatial locality, a hierarchical asynchronous pipeline to overlap I/O with computation, and trajectory-adaptive differential streaming that transfers only incremental working-set deltas between iterations. Experiments show that TideGS enables training with over one billion Gaussians on a single 24 GB GPU while achieving the best reconstruction quality among evaluated single-GPU baselines on large-scale scenes, scaling beyond prior out-of-core baselines (e.g., approximately 100M Gaussians) and standard in-memory training (e.g., approximately 11M Gaussians).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20044v1">OP2GS: Object-Aware 3D Gaussian Splatting with Dual-Opacity Primitives</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) provides an explicit and efficient scene representation, but its primitives lack inherent object-level identity, hindering downstream tasks such as open-vocabulary scene understanding. Existing methods typically address this by either distilling high-dimensional feature embeddings into Gaussians or by lifting 2D mask labels into 3D via heuristic refinement. However, feature-based approaches incur heavy storage and decoding overhead, while lifting-based pipelines remain vulnerable to label contamination: Gaussians necessary for appearance reconstruction often receive incorrect object labels during 2D-to-3D projection. We propose OP2GS, an object-aware Gaussian representation that augments each primitive with an explicit instance identity and a dedicated instance opacity $σ^{*}$ for object-mask rendering. The original opacity $σ$ remains responsible for visual reconstruction, while $σ^{*}$ models whether a Gaussian should contribute to a particular object mask. This dual-opacity formulation decouples visual existence from instance occupancy: mislabeled Gaussians can remain available for image rendering while becoming transparent in the object-mask branch. To learn this representation, we introduce a random object loss that optimizes the 1D instance occupancy field using the standard transmittance-based visibility of 3DGS. Semantic descriptors are then attached at the object level through multi-view aggregation, eliminating per-Gaussian feature storage. Compared with feature-training approaches, OP2GS achieves competitive open-vocabulary performance while significantly reducing computational overhead. Compared with training-free pipelines, it leverages physically consistent occupancy learning to resolve visibility ambiguities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19949v1">Feed-Forward Gaussian Splatting from Sparse Aerial Views</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Reconstructing large-scale urban scenes from sparse aerial views is a crucial yet challenging task. Due to biased top-down and shallow-oblique camera poses, sparse aerial captures exhibit strong evidence imbalance: roofs and open regions are repeatedly observed, while facades, distant buildings, and occluded structures receive little multi-view support. Existing feed-forward 3D Gaussian Splatting methods directly regress a deterministic representation from sparse inputs, but this often leads to ghosting, melted facades, and stretched textures. Recent pseudo-view and video-based generative reconstruction methods use additional supervision or generative priors. However, they often lack a clear separation between observed geometry and prior-driven content, which can lead to plausible but inconsistent structures. We propose AnyCity, an observation-grounded generative reconstruction framework for sparse aerial urban scenes. AnyCity first predicts an observation-supported geometry latent to anchor reliable structures, and then uses scaffold-conditioned aerial completion tokens to predict a gated residual update for weakly constrained content before Gaussian decoding. During training, dense-to-sparse distillation transfers structural cues from dense-view reconstruction, while an aerial-adapted video diffusion prior provides fine-grained urban appearance cues through gated token conditioning. Observation-preserving objectives keep the refined representation consistent with input-supported geometry. At inference time, AnyCity reconstructs the final 3D Gaussian scene from sparse aerial views in a single feed-forward pass, achieving coherent urban novel-view synthesis with second-level inference. Experiments on synthetic, aerial-domain, UAV-textured, and real-world scenes show consistent improvements over feed-forward baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19656v1">Cross-View Splatter: Feed-Forward View Synthesis with Georeferenced Images</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Submitted to CVPR 2026. 8 figures, 3 tables. Project page: https://nianticspatial.github.io/cross-view-splatter/
    </div>
    <details class="paper-abstract">
      We present Cross-View Splatter, a feed-forward method that predicts pixel-aligned Gaussian splats for outdoor scenes captured at ground level AND by satellite. Faithful reconstructions require good camera coverage, but ground imagery is time-consuming and hard to capture at scale for large outdoor scenes. Fortunately, satellite imagery can provide a global geometric prior that is easy to access via public APIs. Cross-View Splatter fuses orthorectified satellite views with GPS-tagged ground photos to predict Gaussian splats in a unified 3D coordinate frame. By aligning ground and bird's-eye feature representations, our model improves scene coverage and novel-view synthesis, compared to ground imagery alone. We train on curated georeferenced datasets and paired satellite-terrain data, mined from open mapping services. We evaluate our method on a new benchmark for novel-view synthesis with georeferenced imagery allowing comparison to prior state-of-the-art methods. Our code and data preparation will be available at https://nianticspatial.github.io/cross-view-splatter/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19600v1">FlyMirage: A Fully Automated Generation Pipeline for Diverse and Scalable UAV Flight Data via Generative World Model</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      In the field of Vision-Language Navigation (VLN), aerial datasets remain limited in their ability to combine scale, diversity, and realism, often relying on either costly real-world scenes or visually limited simulations. To address these challenges, we introduce FlyMirage, a highly scalable and fully automated data generation pipeline for aerial VLN. Our approach leverages large language models (LLM) as an environment designer to promote scene diversity, paired with a generative world model that instantiates these designs into high-fidelity 3D Gaussian Splatting (3DGS) scenes. To substantially reduce human labor and ensure the feasibility of flight data, FlyMirage automates scene exploration and semantic information acquisition, and further integrates a dynamically feasible planner for uncrewed aerial vehicle (UAV) trajectory generation. Utilizing this toolchain, we generate a large-scale, diverse, and photorealistic aerial VLN dataset, with dynamically feasible flying trajectories, designed to support the development of next-generation embodied navigation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19304v1">MMGS: 10$\times$ Compressed 3DGS through Optimal Transport Aggregation based on Multi-view Ranking</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has revolutionized 3D reconstruction, it suffers from significant overhead due to massive redundant primitives. Existing compression methods typically rely on local sampling or fixed pruning thresholds, which often struggle to balance redundancy reduction with high-fidelity rendering. To address this, we propose a novel framework that formulates Gaussian optimization as a global geometric distribution matching problem. Specifically, our approach integrates three components: (1) we introduce a multi-view 3D Gaussian contribution ranking mechanism that filters primitives using geometric consistency instead of local heuristics; (2) we propose a global Optimal Transport (OT)-based aggregation algorithm that merges redundant primitives while preserving the underlying geometry; and (3) we design an OT-based densification operator that maintains the Gaussian's distributional properties for stable optimization. Our approach achieves state-of-the-art rendering quality with only \textbf{10$\%$} primitives and \textbf{10$\times$} accelerated training speeds compared to vanilla 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17916v2">PanoWorld: A Generative Spatial World Model for Consistent Whole-House Panorama Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 17
    </div>
    <details class="paper-abstract">
      Generating a consistent whole-house VR tour from a floorplan and style reference requires both photorealistic panoramas and cross-view spatial coherence. Pure 2D generators produce appealing single panoramas but re-imagine geometry and materials when the viewpoint changes, whereas monolithic 3D generation becomes expensive and loses fine texture at multi-room scale. We introduce PanoWorld, a generative spatial world model that treats whole-house synthesis as autoregressive generation of node-based 360-degree panoramas, matching the discrete navigation used by real VR tour products. PanoWorld uses a floorplan-derived 3D shell as a global geometric proxy and a dynamic 3D Gaussian Splatting cache as renderable spatial memory. A feed-forward panoramic LRM designed for metric-scale multi-room 360-degree inputs lifts generated panoramas into local 3DGS updates, while Room-aware Group Attention suppresses cross-room feature interference. A topology-aware progressive caching strategy fuses these local updates without repeatedly reconstructing the full history. By decoupling shell-based geometry guidance from cache-rendered visual memory, PanoWorld preserves high-frequency 2D synthesis quality while improving cross-node layout and material consistency. The project link is https://jjrcn.github.io/PanoWorld-project-home/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19065v1">A Geometric Algebra-Informed 3D Gaussian Splatting Framework for Wireless Scene Representation</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 Accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      In this paper, we introduce Geometric Algebra-Informed 3D Gaussian Splatting (GAI-GS), a framework for wireless modeling that couples 3D Gaussian splatting with a geometric algebra-based attention mechanism to explicitly model ray-object interactions in complex propagation environments. GAI-GS encodes joint spatial-electromagnetic (EM) relations into token representations, enabling scene-level aggregation within a unified, end-to-end neural architecture. This design grounds wireless ray propagation in electromagnetic principles, allowing token interactions to model key effects such as multipath, attenuation, and reflection/diffraction. Through extensive evaluations on multiple real-world indoor datasets, GAI-GS consistently surpasses current baselines across various wireless tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09668v2">DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 Accepted by ICLR 2026. Project page: https://zju3dv.github.io/DiffWind/
    </div>
    <details class="paper-abstract">
      Modeling wind-driven object dynamics from video observations is highly challenging due to the invisibility and spatio-temporal variability of wind, as well as the complex deformations of objects. We present DiffWind, a physics-informed differentiable framework that unifies wind-object interaction modeling, video-based reconstruction, and forward simulation. Specifically, we represent wind as a grid-based physical field and objects as particle systems derived from 3D Gaussian Splatting, with their interaction modeled by the Material Point Method (MPM). To recover wind-driven object dynamics, we introduce a reconstruction framework that jointly optimizes the spatio-temporal wind force field and object motion through differentiable rendering and simulation. To ensure physical validity, we incorporate the Lattice Boltzmann Method (LBM) as a physics-informed constraint, enforcing compliance with fluid dynamics laws. Beyond reconstruction, our method naturally supports forward simulation under novel wind conditions and enables new applications such as wind retargeting. We further introduce WD-Objects, a dataset of synthetic and real-world wind-driven scenes. Extensive experiments demonstrate that our method significantly outperforms prior dynamic scene modeling approaches in both reconstruction accuracy and simulation fidelity, opening a new avenue for video-based wind-object interaction modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18334v1">3D Skew Gaussian Splatting with Any Camera Trajectory Visualization Engine</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has revolutionized real-time photorealistic view synthesis, its fundamental reliance on symmetric Gaussian distributions introduces visual artifacts that hinder accurate spatial data exploration. Specifically, symmetric kernels struggle to capture shape and color discontinuities , which cause blurriness and primitive redundancy that mislead human perception during visual analysis. To address these visualization barriers, we introduce 3D Skew Gaussian Splatting (3DSGS), a novel framework that significantly enhances the structural fidelity and compactness of explicit scene representations. Our key insight lies in extending the standard primitive to a general Skew Gaussian counterpart. This generalized primitive inherits the highly efficient rasterization properties of standard Gaussians while gaining intrinsic asymmetric modeling capabilities. We couple this with an enhanced opacity representation to better handle complex transparency, alongside a depth-aware densification strategy that intelligently manages primitive allocation. Furthermore, to make these advancements actionable for real-world visual analytics, we re-derive the CUDA rasterization pipeline to universally support both symmetric and skew Gaussians, integrating it into a decoupled, free-camera interactive visualization engine. Extensive experiments demonstrate that 3DSGS achieves superior rendering quality and structural compactness, particularly in regions with intricate details, while maintaining the real-time frame rates necessary for fluid interactive exploration. Supplementary derivations and visual results are available at \textbf{\textit{https://3d-skew-gs.github.io/}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18263v1">RT-Splatting: Joint Reflection-Transmission Modeling with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 CVPR 2026 Highlight, Project Page: https://sjj118.github.io/RT-Splatting/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables real-time novel view synthesis with high visual quality. However, existing methods struggle with semi-transparent specular surfaces that exhibit both complex reflections and clear transmission, often producing blurry reflections or overly occluded transmission. To address this, we present RT-Splatting, a framework that disentangles each Gaussian's geometric occupancy from its optical opacity. This factorization yields a unified surface-volume scene representation with a single set of Gaussian primitives. Our hybrid renderer interprets this representation both as a surface to capture high-frequency reflections and as a volume to preserve clear transmission. To mitigate the ambiguity in jointly optimizing reflection and transmission, we introduce Specular-Aware Gradient Gating, which suppresses misleading gradients from highly specular regions into the transmission branch, effectively reducing distracting floaters. Experiments on challenging semi-transparent scenes show that RT-Splatting achieves state-of-the-art performance, delivering high-fidelity reflections and clear transmission with real-time rendering. Moreover, our factorization naturally enables flexible scene editing. The project page is available at https://sjj118.github.io/RT-Splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18252v1">GaussianZoom: Progressive Zoom-in Generative 3D Gaussian Splatting with Geometric and Semantic Guidance</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      We introduce GaussianZoom, a generative zoom-in 3D reconstruction system with an iterative progressive framework that combines geometry-consistent scene modeling and multi-scale semantic reasoning to enable high-fidelity extreme zoom-in rendering from low-resolution inputs. To achieve this, we develop a novel multi-view consistent super-resolution module with depth-based feature warping and VLM-driven detail synthesis, ensuring accurate multi-view correspondence while enriching fine-scale appearance beyond the observed resolution. To support zooming across large magnification ranges, we further introduce a new expandable continuous Level-of-Detail hierarchy that dynamically modulates Gaussian visibility for smooth, alias-free cross-scale rendering. Experiments on Mip-NeRF360 and Tanks\&Temples demonstrate that GaussianZoom achieves superior perceptual quality, multi-view consistency, and robustness under extreme magnification, establishing a strong baseline for generative zoom-in 3D scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20155v2">GSCompleter: A Distillation-Free Plugin for Metric-Aware 3D Gaussian Splatting Completion in Seconds</a></div>
    <div class="paper-meta">
      📅 2026-05-18
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has revolutionized high-fidelity neural rendering with its explicit representation and efficiency. However, reconstructing scenes from sparse viewpoints suffers from severe geometric voids and floaters due to limited coverage. Current scene completion methods typically rely on an iterative "Repair-then-Distill" paradigm, which is computationally intensive, prone to unstable optimization, and susceptible to overfitting. To address these limitations, we propose GSCompleter, a distillation-free plugin that shifts scene completion to a stable "Generate-then-Register" workflow. Specifically, GSCompleter synthesizes visually plausible 2D reference images and explicitly lifts them into 3D Gaussian primitives with a consistent metric scale via a robust Stereo-Anchor View Selection mechanism. These newly generated primitives are then seamlessly integrated into the global scene using a novel Ray-Constrained Registration strategy. By replacing unstable distillation with rapid geometric registration, GSCompleter exhibits superior 3DGS completion performance across three benchmarks, enhancing both quality and efficiency over various baselines and achieving new state-of-the-art (SOTA) results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10239v2">AdaptSplat: Adapting Vision Foundation Models for Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-18
    </div>
    <details class="paper-abstract">
      This work explores a simple yet powerful lightweight adapter design for feed-forward 3D Gaussian Splatting (3DGS). Existing methods typically apply complex, architecture-specific designs on top of the generic pipeline of image feature extraction $\rightarrow$ multi-view interaction $\rightarrow$ feature decoding. However, constrained by the scale bottleneck of 3D training data and the low-pass filtering effect of deep networks, these methods still fall short in cross-domain generalization and high-frequency geometric fidelity. To address these problems, we propose AdaptSplat, which demonstrates that without complex component engineering, introducing a single adapter of only 1.5M parameters into the generic architecture is sufficient to achieve superior performance. Specifically, we design a lightweight Frequency-Preserving Adapter (FPA) that extracts direction-aware high-frequency structural priors from the shallow features of a powerful vision foundation model backbone, and seamlessly integrates them into the generic pipeline via high-frequency positional encodings and adaptive residual modulation. This effectively compensates for the high-frequency attenuation caused by over-smoothing in deep features, improving the fitting accuracy of Gaussian primitives on complex surfaces and sharp boundaries. Extensive experiments demonstrate that AdaptSplat achieves state-of-the-art feed-forward reconstruction performance on multiple standard benchmarks, with stable generalization across domains. Code available at: https://github.com/xmw666/AdaptSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17777v1">Efficient Sparse-to-Dense Visual Localization via Compact Gaussian Scene Representation and Accelerated Dense Pose Estimation</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 IEEE/CAA JAS 2026
    </div>
    <details class="paper-abstract">
      This letter presents LiteLoc, a novel and efficient localizer built on 3D Gaussian Splatting (3DGS). The previous state-of-the-art (SoTA) sparse-to-dense localizer, STDLoc, has shown remarkable localization capability but suffers from severe storage redundancy and computational latency. By revisiting its design decisions, we derive two simple yet highly effective improvements that cumulatively make LiteLoc much more efficient in both memory and computation, while also being easier to train. One key observation is that the color field, inherited directly from Feature 3DGS, is functionally useless for localization. Yet, its reconstruction of high-frequency photometric details necessitates excessive Gaussian primitives, resulting in a tightly coupled color-feature representation with significant memory overhead and sub-optimal feature field optimization. To resolve this, we propose a color-free decoupled feature field that constructs a compact Gaussian scene representation by retaining only task-essential feature attributes, thereby eliminating approximately 94% of redundant storage with no loss of localization-relevant information. We further find that the primary computational bottleneck lies in the dense Perspective-n-Point (PnP) solver, where most matches contribute saturated geometric constraints with diminishing accuracy gains. Accordingly, we propose a condensing strategy that distills dense matches into a subset of 5% representative matches, enabling a nearly 19-fold speedup in robust estimation with negligible performance drop. Extensive experiments show that LiteLoc surpasses STDLoc in multiple scenes with considerable efficiency benefits, opening up exciting prospects for latency-sensitive visual localization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00952v2">Decoupling Motion and Geometry in 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-16
    </div>
    <details class="paper-abstract">
      High-fidelity reconstruction of dynamic scenes is an important yet challenging problem. While recent 4D Gaussian Splatting (4DGS) has demonstrated the ability to model temporal dynamics, it couples Gaussian motion and geometric attributes within a single covariance formulation, which limits its expressiveness for complex motions and often leads to visual artifacts. To address this, we propose VeGaS, a novel velocity-based 4D Gaussian Splatting framework that decouples Gaussian motion and geometry. Specifically, we introduce a Galilean shearing matrix that explicitly incorporates time-varying velocity to flexibly model complex non-linear motions, while strictly isolating the effects of Gaussian motion from the geometry-related conditional Gaussian covariance. Furthermore, a Geometric Deformation Network is introduced to refine Gaussian shapes and orientations using spatio-temporal context and velocity cues, enhancing temporal geometric modeling. Extensive experiments on public datasets demonstrate that VeGaS achieves state-of-the-art performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17011v1">Topo-GS: Continuous Volumetric Embedding of High-Dimensional Data via Topological Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-16
      | 💬 7 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Dimensionality reduction algorithms map high-dimensional data into visualizable 2D or 3D spaces, but traditionally rely on a discrete point-cloud paradigm. This discrete abstraction is susceptible to visual occlusion and artificial discontinuities, often failing to represent the continuous density of the underlying manifold. To address these limitations, we introduce Topo-GS, a framework that repurposes 3D Gaussian Splatting (3DGS) to cast multidimensional projection as a meshless volumetric reconstruction process. Instead of standard photometric losses, Topo-GS is driven by local geometric constraints. By solving orthogonal Procrustes targets, the optimization enforces an As-Rigid-As-Possible prior while explicitly aligning the spatial covariance of each Gaussian to the local tangent space. Recognizing that unrolling data of varying intrinsic dimensionalities requires distinct spatial treatments, we utilize a topology-aware strategy that tailors the loss formulation to preserve either continuous 1D trajectories or cohesive 2D surfaces. Quantitative and visual evaluations demonstrate that Topo-GS successfully transforms discrete scatter plots into continuous volumetric representations, where inherent projection distortions explicitly manifest as observable geometric variations, while preserving local topological fidelity comparable to discrete baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17002v1">A Single Atlas is All You Need: Decoder-Side Gaussian Splatting for Immersive Video</a></div>
    <div class="paper-meta">
      📅 2026-05-16
    </div>
    <details class="paper-abstract">
      Immersive video delivery is bottlenecked by pixel-rate constraints, making the transmission of high-resolution depth maps or explicit 3D volumetric data expensive. Decoder-Side Depth Estimation (DSDE) shifts depth computation to the client, but struggles with complex geometries, inter-view flickering, and non-Lambertian reflections. Conversely, 3D Gaussian Splatting (3DGS) offers state-of-the-art view synthesis, but transmitting splats (or their projected 2D maps) incurs prohibitive bandwidth costs and is poorly aligned with standard video codecs. We propose Decoder-Side Gaussian Splatting (DSGS), a framework that natively replaces the depth-estimation stage of DSDE with feed-forward 3DGS inference, optimizing volumetric scenes entirely on the decoder side from compressed textures and metadata. A central, counterintuitive finding is that lossy compression acts as an implicit low-pass filter stabilizing feed-forward splat prediction: compressed bitstreams exceed lossless quality while shrinking tenfold. Under extreme view sparsity (one 2D atlas comprising 4 input views), DSGS achieves a +5.79 dB BD-PSNR and +0.054 BD-SSIM gain over the DSDE anchor while reducing maximum inter-view Delta IV-PSNR from 17.2 dB to 6.4 dB, minimizing the domain shift between transmitted and virtual viewports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16925v1">P2GS: Physical Prior-guided Gaussian Splatting for Photometrically Consistent Urban Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-16
      | 💬 Accepted CVPR2026 main
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as a powerful explicit representation enabling fast, high-fidelity rendering, making it a promising foundation for closed-loop simulators and perception models in autonomous driving. However, conventional 3DGS implicitly assumes consistent exposure and tone mapping across views. Real driving data violates this assumption due to heterogeneous camera pipelines and dynamic outdoor illumination, baking exposure discrepancies and sensor noise into the radiance field and producing artifacts and inconsistent illumination especially in static backgrounds crucial for realistic simulation. These issues are amplified in autonomous driving, where sparse viewpoints, varying exposures, and outdoor lighting interact, while prior work mainly targets dynamic-object reconstruction and overlooks cross-view photometric consistency. To address this limitation, we introduce P2GS, a physically consistent Gaussian Splatting framework that jointly decomposes a view-invariant linear HDR radiance field, per-view exposure scales, and tone-mapping functions from only LDR images without HDR supervision. P2GS employs a unified optimization strategy grounded in the physical image-formation process, enforcing relative-exposure consistency and HDR-domain radiance regularization. This yields a radiance field robust to inter-camera illumination differences while preserving the real-time efficiency of standard 3DGS. Experiments across real and simulated driving environments show that P2GS matches or surpasses prior methods in LDR reconstruction while providing substantially improved photometric consistency, reliable exposure normalization, and physically coherent illumination across diverse scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.14009v2">GRaD-Nav++: Vision-Language Model Enabled Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-05-15
      | 💬 Published in: IEEE Robotics and Automation Letters ( Volume: 11, Issue: 2, February 2026)
    </div>
    <details class="paper-abstract">
      Autonomous drones capable of interpreting and executing high-level language instructions in unstructured environments remain a long-standing goal. Yet existing approaches are constrained by their dependence on hand-crafted skills, extensive parameter tuning, or computationally intensive models unsuitable for onboard use. We introduce GRaD-Nav++, a lightweight Vision-Language-Action (VLA) framework that runs fully onboard and follows natural-language commands in real time. Our policy is trained in a photorealistic 3D Gaussian Splatting (3DGS) simulator via Differentiable Reinforcement Learning (DiffRL), enabling efficient learning of low-level control from visual and linguistic inputs. At its core is a Mixture-of-Experts (MoE) action head, which adaptively routes computation to improve generalization while mitigating forgetting. In multi-task generalization experiments, GRaD-Nav++ achieves a success rate of 83% on trained tasks and 75% on unseen tasks in simulation. When deployed on real hardware, it attains 67% success on trained tasks and 50% on unseen ones. In multi-environment adaptation experiments, GRaD-Nav++ achieves an average success rate of 81% across diverse simulated environments and 67% across varied real-world settings. These results establish a new benchmark for fully onboard Vision-Language-Action (VLA) flight and demonstrate that compact, efficient models can enable reliable, language-guided navigation without relying on external infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06088v2">OpenGaFF: Open-Vocabulary Gaussian Feature Field with Codebook Attention</a></div>
    <div class="paper-meta">
      📅 2026-05-15
    </div>
    <details class="paper-abstract">
      Understanding open-vocabulary 3D scenes with Gaussian-based representations remains challenging due to fragmented and spatially inconsistent semantic predictions across multi-view observations. In this paper, we present OpenGaFF, a novel framework for open-vocabulary 3D scene understanding built upon 3D Gaussian Splatting. At the core of our method is a Gaussian Feature Field that models semantics as a continuous function of Gaussian geometry and appearance. By explicitly conditioning semantic predictions on geometric structure, this formulation strengthens the coupling between geometry and semantics, leading to improved spatial coherence across similar structures in 3D space. To further enforce object-level semantic consistency, we introduce a structured codebook that serves as a set of shared semantic primitives. Furthermore, a codebook-guided attention mechanism is proposed to retrieve language features via similarity matching between query embeddings and learned codebook entries, enabling robust open-vocabulary reasoning while reducing intra-object feature variance. Extensive experiments on standard 2D and 3D open-vocabulary benchmarks demonstrate that our method consistently outperforms prior approaches, achieving improved segmentation quality, stronger 3D semantic consistency and a semantically interpretable codebook that provides insight into the learned representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16582v1">ArtMesh: Part-Aware Articulated Mesh Fields with Motion-Consistent Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-05-15
    </div>
    <details class="paper-abstract">
      We present ArtMesh, a mesh-native method for reconstructing articulated objects explicitly as connected triangle meshes with per-part rigid motion from multi-view images in start and end states. Existing 3D Gaussian Splatting pipelines for articulated reconstruction inherit the unstructured point-based geometry of their splatting base, which provides no surface topology for reasoning about part boundaries or enforcing motion consistency along the object's connectivity. ArtMesh instead builds on a mesh-based differentiable rendering backbone, enabling part-aware dynamics to act directly on the structured topology. To make the topology compatible with articulation, we introduce part-aware restricted Delaunay remeshing, producing connected submeshes whose triangles do not cross semantic part boundaries. The dynamic mesh field then optimizes articulation using bidirectional Vertex-wise Motion Consistency on transported mesh vertices and Pixel-wise Motion Consistency on rendered RGB-D observations. We introduce Articulate-100, a new benchmark of 100 articulated objects spanning 16 PartNet-Mobility categories. On this benchmark, ArtMesh outperforms prior 3DGS-based pipelines in joint parameter estimation and part-level geometric reconstruction, with the largest gains on objects with many movable parts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16158v1">Smart target point control for Gaussian Splatting methods</a></div>
    <div class="paper-meta">
      📅 2026-05-15
    </div>
    <details class="paper-abstract">
      Standard Gaussian splatting methods rely on heuristic densification and pruning to adaptively allocate primitives during training, and the resulting Gaussian count strongly influences both reconstruction quality and runtime. This makes comparisons across methods fragile: improvements can stem from higher representational capacity rather than algorithmic design. A common and naive workaround for this is hard-stopping or budgeting densification/pruning once a target count is reached, which biases training because different methods hit the cap at different times, yielding non-uniform densify/prune exposure across views and uneven point distributions. We propose a target point control scheme that preserves the standard densification window and cadence, but adjusts only the existing densification and opacity-culling hyper-parameters to track a quadratic target count trajectory. This quota-governor reaches the desired count by 15k iterations without abrupt cutoffs, ensuring that all methods and views receive equal densification and pruning cycles, enabling fairer, capacity-matched evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16065v1">Robust Prior-Guided Segmentation for Editable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-15
      | 💬 Accepted at IEEE International Conference on Image Processing 2026, 6 pages
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D-GS) enables real-time 3D scene reconstruction but lacks robust segmentation for editing tasks such as object removal, extraction, and recoloring. Existing approaches that lift 2D segmentations to the 3D domain suffer from view inconsistencies and coarse masks. In this paper, we propose a novel framework that leverages the Segment Anything Model High Quality (SAM-HQ) to generate accurate 2D masks, addressing the limitations of the standard SAM in boundary fidelity and fine-structure preservation. To achieve robust 3D segmentation of any target object in a given scene, we introduce a prior-guided label reassignment method that assigns labels to 3D Gaussians by enforcing multiview consistency with learned priors. Our approach achieves state-of-the-art segmentation accuracy and enables interactive, real-time object editing while maintaining high visual fidelity. Qualitative results demonstrate superior boundary preservation and practical utility in Virtual Reality (VR) and robotics, advancing 3D scene editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16022v1">EndoGSim: Physics-Aware 4D Dynamic Endoscopic Scene Simulations via MLLM-Guided Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-15
      | 💬 Early Accepted by MICCAI 2026
    </div>
    <details class="paper-abstract">
      In robot-assisted minimally invasive surgery, high-fidelity dynamic endoscopic scene reconstruction and simulation are crucial to enhancing downstream tasks and advancing surgical outcomes. However, existing methods primarily focus on visual reconstruction, lacking physics-based descriptions of the scene required for realistic simulation. We propose a unified framework that achieves physics-aware reconstruction and physical simulation of endoscopic scenes through Multi-modal Large Language Models (MLLMs)-guided Gaussian Splatting. Our approach utilizes 4D Gaussian Splatting (4DGS) integrated with pre-trained segmentation and depth estimation to represent deformable tissues and tools. To achieve automatic inference of physical properties, we introduce an object-wise material field that initializes material parameters via MLLM and refines them through a differentiable Material Point Method (MPM) under joint supervision from rendered images and optical flow. Validated on both open-source and in-house datasets, our framework achieves superior simulation fidelity and physical accuracy compared to state-of-the-art methods, underscoring its potential to advance robot-assisted surgical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.28111v3">GSDrive: Reinforcing Driving Policies by Multi-mode Future Trajectory Probing with 3D Gaussian Splatting Environment</a></div>
    <div class="paper-meta">
      📅 2026-05-15
      | 💬 2nd version
    </div>
    <details class="paper-abstract">
      End-to-end (E2E) autonomous driving aims to directly map sensory observations to driving actions, but its real-world deployment is hindered by evolving data distributions and the high cost of continual annotation. While combining imitation learning (IL) and reinforcement learning (RL) is a common strategy for policy improvement, conventional RL training relies on delayed, event-based rewards, where policies learn only from catastrophic outcomes such as collisions, leading to premature convergence to suboptimal behaviors. To address these limitations, we propose GSDrive, a framework that uses a differentiable 3D Gaussian Splatting (3DGS) environment for future-aware trajectory probing and reward shaping in E2E driving. GSDrive first learns a multi-mode trajectory probe via IL and then uses RL to evaluate multiple candidate futures in the 3DGS environment, converting their simulated returns into dense shaping rewards for policy optimization. This yields a cyclic hybrid IL-RL training loop, where IL supplies structured future priors and RL provides interactive feedback for iterative refinement. Evaluated on the reconstructed nuScenes dataset, our method outperforms other simulation-based RL approaches in closed-loop experiments. Code is available at https://github.com/ZionGo6/GSDrive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15760v1">Learn2Splat: Extending the Horizon of Learned 3DGS Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-15
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) optimization is most commonly performed using standard optimizers (Adam, SGD). While stable across diverse scenes, standard optimizers are general-purpose and not tailored to the structure of the problem. In particular, they produce independent parameter updates that do not capture the structural and spatial relationships within a scene, leading to inefficient optimization and slow convergence. Recent works introduced learned optimizers that predict correlated updates informed by inter-parameter and inter-Gaussian dependencies. However, these methods are trained for a fixed number of optimization iterations and rely on manually scheduled learning rates to avoid degradation. In this paper, we introduce a learned optimizer for 3DGS that avoids degradation over extended optimization horizons without auxiliary mechanisms. To enable this, we propose a meta-learning scheme that extends the optimization horizon via a checkpoint buffer and an optimizer rollout strategy, combined with an architecture that encodes gradient scale information in its latent states. Results show improved early novel view synthesis quality while remaining stable over long horizons, with zero-shot generalization to unseen reconstruction settings. To support our findings, we introduce the first unified framework for training and evaluating both learned and conventional optimizers across sparse and dense view settings. Code and models will be released publicly. Our project page is available at https://naamapearl.github.io/learn2splat .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18866v1">FLUIDSPLAT: Reconstructing Physical Fields from Sparse Sensors via Gaussian Primitives</a></div>
    <div class="paper-meta">
      📅 2026-05-15
      | 💬 23 pages, 4 figures,preprint
    </div>
    <details class="paper-abstract">
      Reconstructing continuous flow fields from sparse surface-mounted sensors is central to aerodynamic design, flow control, and digital-twin instrumentation. Existing neural methods for this task typically encode sensor readings into implicit latent codes with little spatial interpretability and limited formal guidance on how representational capacity should scale with observation count. Inspired by 3D Gaussian Splatting, we introduce FLUIDSPLAT, a sensor-conditioned model that predicts K anisotropic Gaussian primitives forming a partition-of-unity scaffold, a spatially explicit and interpretable intermediate representation of the flow. For an idealized Gaussian primitive estimator, we prove an $O(K^{-s/d})$ approximation rate for fields with Sobolev smoothness $s$; incorporating $N$ noisy observations yields a squared-risk decomposition with bias $O(K^{-2s/d})$ and variance $O(σ^{2}K/N)$.Balancing the two yields $K^{*}\!\sim\!(N/σ^{2})^{d/(2s+d)}$: primitive count cannot grow freely under sparse sensing, revealing a variance bottleneck that motivates complementing the scaffold with a state-conditioned residual decoder. On a standard cylinder-flow benchmark, FLUIDSPLAT achieves the best mean error across all surface-sensor layouts; on AirfRANS with 8 surface-pressure sensors, it reduces error by 11-23% over the strongest baseline across three standard splits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15010v2">3D Skew-Normal Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-15
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a leading representation for real-time novel view synthesis and has been widely adopted in various downstream applications. The core strength of 3DGS lies in its efficient kernel-based scene representation, where Gaussian primitives provide favorable mathematical and computational properties. However, under a finite primitive budget, the symmetric shape of each primitive directly affects representation compactness, especially near asymmetric structures such as object boundaries and one-sided surfaces. Recent works have explored more complex kernel distributions; however, they either remain within the elliptical family or rely on hard truncation, which limits continuous shape control and introduces distributional discontinuities. In this paper, we propose Skew-Normal Splatting (SNS), which adopts the Azzalini Skew-Normal distribution as the fundamental primitive. By introducing a learnable and bounded skewness parameter, SNS can continuously interpolate between symmetric Gaussians and Half-Gaussian-like shapes, enabling flexible modeling of both sharp boundaries and interior regions. Moreover, SNS preserves analytical tractability under affine transformations and marginalization. This property allows seamless integration into existing Gaussian Splatting rasterization pipelines. Furthermore, to address the strong coupling between scale, rotation, and skewness parameters, we introduce a decoupled parameterization and a block-wise optimization strategy to enhance training stability and accuracy. Extensive experiments on standard novel-view synthesis benchmarks show that SNS consistently improves reconstruction quality over Gaussian and recent non-Gaussian kernels, with clearer benefits on sharp boundaries and thin or one-sided structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13073v2">HarmoGS: Robust 3D Gaussian Splatting in the Wild via Conflict-Aware Gradient Harmonization</a></div>
    <div class="paper-meta">
      📅 2026-05-15
    </div>
    <details class="paper-abstract">
      In-the-wild 3D Gaussian Splatting remains challenging due to transient distractors and illumination-induced cross-view appearance inconsistencies. Existing methods mainly rely on image-level masking to suppress unreliable supervision, but masking alone cannot fully eliminate residual occlusions or resolve illumination-induced inconsistencies, both of which can introduce conflicting cross-view gradients. These unresolved conflicts may destabilize Gaussian optimization and lead to visible reconstruction artifacts. We propose a conflict-aware 3DGS framework that addresses this problem from both image-space supervision and gradient-level optimization. Semantic Consistency-Guided Masking learns pixel-wise consistency scores to adaptively refine prior masks and suppress unreliable supervision before gradient formation. A dual-view Conflict-Aware Gradient Harmonization strategy further reconciles view-specific gradients by mutually rotating them into an orthogonal configuration, reducing negative directional interference across views. We also introduce conflict-aware densification and pruning to stabilize Gaussian growth and remove persistently conflicting primitives. Extensive experiments on standard in-the-wild benchmarks demonstrate that our method achieves state-of-the-art rendering quality under complex transient distractors and cross-view inconsistencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15398v1">3DEditSafe: Defending 3D Editing Pipelines from Unsafe Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-14
    </div>
    <details class="paper-abstract">
      Recent advances in 3D generative editing, particularly pipelines based on 3D Gaussian Splatting (3DGS), have achieved high-fidelity, multi-view-consistent scene manipulation from text prompts. However, we find that these pipelines also introduce new safety risks when unsafe prompts produce edits that are propagated and optimized across views. In this work, we study unsafe generation in 3D editing pipelines and show that such behavior can lead to coherent, undesirable Not-Safe-For-Work (NSFW) content in the final 3D representation. To address this, we propose 3DEditSafe, a safety-regularized 3D editing framework that constrains unsafe semantic propagation during optimization. 3DEditSafe combines generation-stage safety guidance with rendered-view 3D safety regularization, safe semantic projection, residue suppression, and mask-aware preservation to steer optimization away from unsafe editing directions. We evaluate our approach on EditSplat scenes using an object-compatible unsafe prompt benchmark and show that 2D safety guidance alone is not consistently sufficient to prevent unsafe 3D edits. 3DEditSafe reduces unsafe semantic alignment and view-level attack success rates, while revealing a safety-quality tradeoff in which stronger unsafe suppression can introduce artifacts or reduce unsafe-prompt fidelity. To our knowledge, this work is the first attempt to study and defend against unsafe generation in text-driven 3D editing pipelines, highlighting the need for safety mechanisms that operate directly on optimized 3D representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.14880v1">Denoising-GS: Gaussian Splatting with Spatial-aware Denoising</a></div>
    <div class="paper-meta">
      📅 2026-05-14
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have achieved remarkable success in high-fidelity Novel View Synthesis (NVS), yet the optimization process inevitably introduces noisy Gaussian primitives due to the sparse and incomplete initialization from Structure-from-Motion (SfM) point clouds. Most existing methods focus solely on adjusting the positions of primitives during optimization, while neglecting the underlying spatial structure. To this end, we introduce a new perspective by formulating the optimization of 3DGS as a primitive denoising process and propose Denoising-GS, a spatial-aware denoising framework for Gaussian primitives by taking both the positions and spatial structure into consideration. Specifically, we design an optimizer that preserves the spatial optimization flow of primitives, facilitating coherent and directed denoising rather than random perturbations. Building upon this, the Spatial Gradient-based Denoising strategy jointly considers the spatial supports of primitives to ensure gradient-consistent updates. Furthermore, the Uncertainty-based Denoising module estimates primitive-wise uncertainty to prune redundant or noisy primitives, while the Spatial Coherence Refinement strategy selectively splits primitives in sparse regions to maintain structural completeness. Experiments conducted on three benchmark datasets demonstrate that Denoising-GS consistently enhances NVS fidelity while maintaining representation compactness, achieving state-of-the-art performance across all benchmarks. Source code and models will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24316v2">Large-Scale Photogrammetric Documentation of St. John's Co-Cathedral: A Workflow for Cultural Heritage Preservation</a></div>
    <div class="paper-meta">
      📅 2026-05-14
    </div>
    <details class="paper-abstract">
      We present a comprehensive methodology for the large-scale photogrammetric documentation of St. John's Co-Cathedral in Valletta, Malta, a UNESCO World Heritage site renowned for its ornate Baroque architecture and Caravaggio masterpieces. Over seven nights of evening-only data collection, we captured 99,000 images using DSLR cameras, drone photography, and LIDAR scanning to create a highly detailed 3D reconstruction comprising 25-30 billion triangles. This paper documents our complete workflow for cultural heritage preservation, addressing the unique challenges of digitizing complex baroque architectural spaces with highly reflective metallic surfaces, dark materials, intricate tapestries, and restricted access. We detail our pipeline from multi-modal data acquisition through processing, including strategic image grading and AI-assisted denoising to address low-light grain, extensive LIDAR point cloud cleanup, hybrid photogrammetric reconstruction using RealityCapture, and mesh subdivision strategies for real-time visualization engines. Our methodology combines automated workflows with necessary manual intervention to handle the scale and complexity of the project, with particular attention to reflective surface challenges characteristic of baroque heritage sites. We also present preliminary experiments with Gaussian splatting as a complementary representation technique. The resulting digital archive serves multiple preservation purposes including disaster recovery documentation, conservation analysis, virtual tourism, and scholarly research. This work provides a detailed, replicable workflow for heritage professionals undertaking similar large-scale architectural documentation projects, addressing the practical challenges of applying photogrammetric methods in complex real-world heritage scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.14629v1">Efficient Dense Matching for Enhanced Gaussian Splatting Using AV1 Motion Vectors</a></div>
    <div class="paper-meta">
      📅 2026-05-14
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a prominent framework for real-time, photorealistic scene reconstruction, offering significant speed-ups over Neural Radiance Fields (NeRF). However, the fidelity of 3DGS representations remains heavily dependent on the quality of the initial point cloud. While standard Structure-from-Motion (SfM) pipelines using COLMAP provide adequate initialisation, they often suffer from high computational costs and sparsity in textureless regions, which degrades subsequent reconstruction accuracy and convergence speed. In this work, we introduce an AV1-based feature detection and matching pipeline that significantly reduces SfM processing overhead. By leveraging motion vectors inherent to the AV1 video codec, we bypass computationally expensive exhaustive matching while maintaining geometric robustness. Our pipeline produces substantially denser point clouds, with up to eight times as many points as classical SfM. We demonstrate that this enhanced initialisation directly improves 3DGS performance, yielding an 9-point increase in VMAF and a 63% average reduction in training time required to reach baseline quality. The project page: https://sigmedia.tv/AV1-3DGS.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02482v2">G-SHARP: Gaussian Surgical Hardware Accelerated Real-time Pipeline</a></div>
    <div class="paper-meta">
      📅 2026-05-14
    </div>
    <details class="paper-abstract">
      We propose G-SHARP, a commercially compatible, real-time surgical scene reconstruction framework designed for minimally invasive procedures that require fast and accurate 3D modeling of deformable tissue. While recent Gaussian splatting approaches have advanced real-time endoscopic reconstruction, existing implementations often depend on non-commercial derivatives, limiting deployability. G-SHARP overcomes these constraints by being the first surgical pipeline built natively on the GSplat (Apache-2.0) differentiable Gaussian rasterizer, enabling principled deformation modeling, robust occlusion handling, and high-fidelity reconstructions on the EndoNeRF pulling benchmark. Our results demonstrate state-of-the-art reconstruction quality with strong speed-accuracy trade-offs suitable for intra-operative use. Finally, we provide a Holoscan SDK application that deploys G-SHARP on NVIDIA IGX Orin and Thor edge hardware, enabling real-time surgical visualization in practical operating-room settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.14135v1">PanoPlane: Plane-Aware Panoramic Completion for Sparse-View Indoor 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-13
    </div>
    <details class="paper-abstract">
      We present PanoPlane, an approach for high-fidelity sparse-view indoor novel view synthesis that reconstructs closed room geometry via panoramic scene completion. Unlike perspective-based methods that generate training views from limited fields of view, PanoPlane leverages $360^{\circ}$ panoramic completion to condition the generative process on the full spatial layout. We propose Layout Anchored Attention Steering, a training-free mechanism that steers attention within the diffusion model's internal representation toward scene's detected planar surfaces at inference time. By directing each unobserved region's attention toward geometrically consistent observed content, our method replaces unconstrained hallucination with grounded surface extrapolation. The resulting panoramic completions provide supervision for 3D Gaussian Splatting, enabling accurate novel-view synthesis across unobserved regions from as few as three input views. Experiments on Replica, ScanNet++, and Matterport3D demonstrate state-of-the-art novel view synthesis quality across 3, 6, and 9 input views, achieving up to $+17.8\%$ improvement in PSNR over the current state-of-the-art baseline without any training or fine-tuning of the diffusion model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13794v1">BlitzGS: City-Scale Gaussian Splatting at Lightning Speed</a></div>
    <div class="paper-meta">
      📅 2026-05-13
    </div>
    <details class="paper-abstract">
      We present BlitzGS, a distributed 3DGS framework that reduces active Gaussian workload for fast city-scale reconstruction. BlitzGS manages this workload at three coupled levels. At the system level, the framework shards Gaussians across GPUs by index parity rather than spatial blocks. This approach mitigates the cross-block visibility redundancy inherent in spatial partitioning. Furthermore, it distributes each rendering step through a single cross-GPU exchange that routes projected Gaussians to their tile owners. At the model level, scheduled importance-scoring passes shrink the global Gaussian population. During these passes, the framework generates a per-Gaussian visibility weight to bias density-control updates toward contributing primitives and a per-view importance mask for the view-level renderer. At the view level, BlitzGS trims each camera's active set with a distance-based LOD gate to exclude excessively fine primitives for the current frustum and the importance-based culling mask to skip Gaussians with negligible cross-view contribution. On large-scale benchmarks, BlitzGS matches the rendering quality of recent large-scale baselines while delivering an order-of-magnitude speedup, training city-scale scenes in tens of minutes. Our code is available at https: //github.com/AkierRaee/BlitzGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13600v1">Sparse Code Uplifting for Efficient 3D Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-13
      | 💬 18 pages (9 pages main paper), 10 figures, preprint
    </div>
    <details class="paper-abstract">
      3D Language Gaussian Splatting (3DLGS) augments 3D Gaussian Splatting with language-aligned visual features for open-vocabulary 3D scene understanding. A core challenge is efficiently associating high-dimensional vision-language embeddings with millions of 3D Gaussians while preserving efficient feature rendering for text-based querying. Existing methods either store dense features directly on Gaussians, causing high storage costs and slow rendering, or learn compact representations through expensive per-scene optimization with repeated feature rasterization. No existing method simultaneously achieves fast 3D semantic reconstruction, efficient storage, and fast rendering. We propose SCOUP (Sparse COde UPlifting), which addresses all three by decoupling language representation learning from 3D Gaussian optimization. Rather than working directly in 3D, we learn sparse codebook-based representations entirely using features associated with 2D image regions, associating each region with a sparse set of codebook coefficients. We then uplift these coefficients to 3D Gaussians with our weighted sparse aggregation using Gaussian-to-pixel associations, where each Gaussian accumulates coefficients over codebook atoms across views. Top-$K$ filtering then extracts the most dominant multi-view coefficients per Gaussian, enabling efficient storage and fast rendering. Our method achieves up to $400\times$ training speedup while being $3\times$ more memory efficient during training compared to the state-of-the-art in rendering speed. Across multiple benchmarks, SCOUP matches or outperforms existing methods in open-vocabulary querying accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13591v1">Real2Sim: A Physics-driven and Editable Gaussian Splatting Framework for Autonomous Driving Scenes</a></div>
    <div class="paper-meta">
      📅 2026-05-13
    </div>
    <details class="paper-abstract">
      Reliable autonomous driving relies on large-scale, well-labeled data and robust models. However, manual data collection is resource-intensive, and traditional simulation suffers from a persistent reality gap. While recent generative frameworks and radiance-field methods improve visual fidelity, they still struggle with temporal and spatial consistency and cannot ensure physics-aware behavior, limiting their applicability to driving scenario generation. To address these challenges, we propose Real2Sim, an unified framework that combines 4D Gaussian Splatting (4DGS) with a differentiable Material Point Method (MPM) solver. Real2Sim explicitly reconstructs dynamic driving scenes as temporally continuous Gaussian primitives, supports instance-level editing, and simulates realistic object-object and object-environment interactions. This framework enables physics-aware, high-fidelity synthesis of diverse, editable scenarios, including challenging corner cases such as collisions and post-impact trajectories. Experiments on the Waymo Open Dataset validate Real2Sim's capabilities in rendering, reconstruction, editing, and physics simulation, demonstrating its potential as a scalable tool for data generation in downstream tasks such as perception, tracking, trajectory prediction, and end-to-end policy learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21238v3">3D-UIR: 3D Gaussian for Underwater 3D Scene Reconstruction via Physics Based Appearance-Medium Decoupling</a></div>
    <div class="paper-meta">
      📅 2026-05-13
      | 💬 Accepted to IEEE TIP 2026. Project webpage: https://bilityniu.github.io/3D-UIR
    </div>
    <details class="paper-abstract">
      Novel view synthesis for underwater scene reconstruction presents unique challenges due to complex light-media interactions. Optical scattering and absorption in water body bring inhomogeneous medium attenuation interference that disrupts conventional volume rendering assumptions of uniform propagation medium. While 3D Gaussian Splatting (3DGS) offers real-time rendering capabilities, it struggles with underwater inhomogeneous environments where scattering media introduces artifacts and inconsistent appearance. In this study, we propose a physics-based framework that disentangles object appearance from water medium effects through tailored Gaussian modeling. Our approach introduces appearance embeddings, which are explicit medium representations for backscatter and attenuation, enhancing scene consistency. In addition, we propose a depth-guided optimization strategy that leverages pseudo-depth maps as supervision with depth regularization and scale penalty terms to improve geometric fidelity. By integrating the proposed appearance and medium modeling components via an underwater imaging model, our approach achieves both high-quality novel view synthesis and physically accurate scene restoration. Experiments demonstrate our significant improvements in rendering quality and restoration accuracy over existing methods. The project page is available at https://bilityniu.github.io/3D-UIR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13465v1">Z-Order Transformer for Feed-Forward Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-13
      | 💬 Accept by CVPR 2026, Oral
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled significant progress in photorealistic novel view synthesis. However, traditional 3DGS relies on a slow, iterative optimization process, which limits its use in scenarios demanding real-time results. To overcome this bottleneck, recent feed-forward methods aim to predict Gaussian attributes directly from images, but they often struggle with the redundancy of Gaussian primitives and rendering quality. In this work, we introduce a transformer-based architecture specifically designed for feed-forward Gaussian Splatting. Our key insight is that spatial and semantic relationships among Gaussians can be effectively captured through a sparse attention mechanism, enabled by a Z-order strategy that organizes the unstructured Gaussian set into a spatially coherent sequence. Furthermore, we incorporate this Z-order strategy to adaptively suppress redundancy while preserving critical structural details. This allows the transformer to efficiently model context, compress Gaussian primitives, and predict Gaussian attributes in a single forward pass. Comprehensive experiments demonstrate that our method achieves fast and high-quality novel view synthesis with fewer Gaussian primitives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05876v3">3DSS: 3D Surface Splatting for Inverse Rendering</a></div>
    <div class="paper-meta">
      📅 2026-05-13
    </div>
    <details class="paper-abstract">
      We present 3D Surface Splatting (3DSS), the first differentiable surface splatting renderer for physically-based inverse rendering from multi-view images. Our central insight is that the surface separation problem at the heart of surface splatting admits a direct formulation in terms of the reconstruction kernels themselves. From this foundation we derive a coverage-based compositing model whose per-layer opacity arises directly from the accumulated Elliptical Weighted Average reconstruction weight, yielding anti-aliased silhouettes and informative visibility gradients at sparsely covered edges. Combined with forward microfacet shading under co-optimized HDR environment lighting and density-aware adaptive refinement, 3DSS jointly recovers shape, spatially-varying BRDF materials, and illumination. Because the optimized representation is a set of oriented surface samples, it bridges natively to mesh-based workflows via surface reconstruction from oriented point cloud methods. We evaluate 3DSS against mesh-based, implicit, and Gaussian-splatting baselines across geometry reconstruction, novel-view synthesis, and novel-illumination relighting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.09479v2">SkySplat: Generalizable 3D Gaussian Splatting from Multi-Temporal Sparse Satellite Images</a></div>
    <div class="paper-meta">
      📅 2026-05-13
      | 💬 AAAI 2026. Code is available at https://github.com/NanCheng2001/SkySplat-main
    </div>
    <details class="paper-abstract">
      Three-dimensional scene reconstruction from sparse-view satellite images is a long-standing and challenging task. While 3D Gaussian Splatting (3DGS) and its variants have recently attracted attention for its high efficiency, existing methods remain unsuitable for satellite images due to incompatibility with rational polynomial coefficient (RPC) models and limited generalization capability. Recent advances in generalizable 3DGS approaches show potential, but they perform poorly on multi-temporal sparse satellite images due to limited geometric constraints, transient objects, and radiometric inconsistencies. To address these limitations, we propose SkySplat, a novel self-supervised framework that integrates the RPC model into the generalizable 3DGS pipeline, enabling more effective use of sparse geometric cues for improved reconstruction. SkySplat relies only on RGB images and radiometric-robust relative height supervision, thereby eliminating the need for ground-truth height maps. Key components include a Cross-Self Consistency Module (CSCM), which mitigates transient object interference via consistency-based masking, and a multi-view consistency aggregation strategy that refines reconstruction results. Compared to per-scene optimization methods, SkySplat achieves an 86 times speedup over EOGS with higher accuracy. It also outperforms generalizable 3DGS baselines, reducing MAE from 13.18 m to 1.80 m on the DFC19 dataset significantly, and demonstrates strong cross-dataset generalization on the MVS3D benchmark. The is available at https://github.com/NanCheng2001/SkySplat-main
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.12072v2">PairDropGS: Paired Dropout-Induced Consistency Regularization for Sparse-View Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-13
      | 💬 11 pages,8 figures
    </div>
    <details class="paper-abstract">
      Dropout-based sparse-view 3D Gaussian Splatting (3DGS) methods alleviate overfitting by randomly suppressing Gaussian primitives during training. Existing methods mainly focus on designing increasingly sophisticated dropout strategies, while they overlook the resulting inconsistencies among different dropped Gaussian subsets. This oversight often leads to unstable reconstruction and suboptimal Gaussian representation learning.In this paper, we revisit dropout-based sparse-view 3DGS from a consistency regularization perspective and propose PairDropGS, a Paired Dropout-induced Consistency Regularization framework for sparse-view Gaussian splatting. Specifically, PairDropGS first constructs a pair of the dropped Gaussian subsets from a shared Gaussian field and designs a low-frequency consistency regularization to constrain their low-frequency rendered structures. This design encourages the shared Gaussian field to preserve stable scene layout and coarse geometry under different random dropouts, while avoiding excessive constraints on ambiguous high-frequency details. Moreover, we introduce a progressive consistency scheduling strategy to gradually strengthen the consistency regularization during training for stability and robustness of reconstruction. Extensive experiments on widely-used sparse-view benchmarks demonstrate that PairDropGS achieves superior training stability, significantly outperforms existing dropout-based 3DGS methods in reconstruction quality, while exhibiting the simplicity and plug-and-play nature for improving dropout-based optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13093v1">RoSplat: Robust Feed-Forward Pixel-wise Gaussian Splatting for Varying Input Views and High-Resolution Rendering</a></div>
    <div class="paper-meta">
      📅 2026-05-13
    </div>
    <details class="paper-abstract">
      Generalizable 3D Gaussian Splatting has recently emerged as an efficient approach for novel-view synthesis, enabling feed-forward synthesis from only a few input views. However, existing pixel-wise feed-forward methods suffer from over-bright renderings when the number of input views varies during inference, as well as insufficient supervision for accurate Gaussian scale estimation, which leads to hole artifacts, particularly in high-resolution renderings. To address these issues, we identify that the over-brightness is caused by the varying number of overlapping Gaussians and propose a simple alpha normalization strategy to maintain brightness consistency across different number of input views. In addition, we introduce an auxiliary 3D sampling-based regularizer to improve Gaussian scale estimation, thereby mitigating hole artifacts in high-resolution rendering. Experiments on benchmark datasets demonstrate that our method significantly improves baseline models under varying input-view and high-resolution rendering settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2403.11247v3">Compact 3D Gaussian Splatting For Dense Visual SLAM</a></div>
    <div class="paper-meta">
      📅 2026-05-13
      | 💬 Accepted by IJCV 2026
    </div>
    <details class="paper-abstract">
      Recent work has shown that 3D Gaussian-based SLAM enables high-quality reconstruction, accurate pose estimation, and real-time rendering of scenes. However, these approaches are built on a tremendous number of redundant 3D Gaussian ellipsoids, leading to high memory and storage costs, and slow training speed. To address the limitation, we propose a compact 3D Gaussian Splatting SLAM system that reduces the number and the parameter size of Gaussian ellipsoids. A sliding window-based masking strategy is first proposed to reduce the redundant ellipsoids. Then we observe that the covariance matrix (geometry) of most 3D Gaussian ellipsoids are extremely similar, which motivates a novel geometry codebook to compress 3D Gaussian geometric attributes, i.e., the parameters. Robust and accurate pose estimation is achieved by a global bundle adjustment method with reprojection loss. Extensive experiments demonstrate that our method achieves faster training and rendering speed while maintaining the state-of-the-art (SOTA) quality of the scene representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.12919v1">GuardMarkGS: Unified Ownership Tracing and Edit Deterrence for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-13
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is becoming a practical representation for novel view synthesis, but its growing adoption, together with rapid advances in instruction-driven 3DGS editing, also exposes a dual copyright risk: once a 3DGS-based asset is released, it can be used without permission and manipulated through 3D editing. Existing protection methods address only one side of this problem. Watermarking can trace ownership after unauthorized use, but it cannot prevent malicious editing. Adversarial edit-deterrence methods can disrupt editing, but they do not provide evidence of ownership. To the best of our knowledge, we present the first unified protection framework for 3DGS that jointly optimizes ownership tracing and unauthorized editing deterrence. Our framework combines a scene-wide watermarking objective over all Gaussians with an adversarial objective for edit deterrence. The adversarial branch combines latent-anchor separation, denoising-trajectory diversion, and cross-attention diversion to divert the editing trajectory, while an update-saliency-motivated Gaussian selection strategy assigns stronger adversarial updates to mask-selected Gaussians, improving the balance among watermark recovery, edit deterrence, and rendering fidelity. Experiments on scenes from Mip-NeRF 360 and Instruct-NeRF2NeRF demonstrate that the proposed framework achieves a favorable balance among bit accuracy, edit deterrence, and rendering quality. These results suggest that practical copyright protection of 3DGS-based assets can be more effectively addressed by integrating ownership tracing and unauthorized editing deterrence into a single optimization framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04506v2">Ilov3Splat: Instance-Level Open-Vocabulary 3D Scene Understanding in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-13
      | 💬 The International Conference on Pattern Recognition (ICPR) 2026
    </div>
    <details class="paper-abstract">
      We introduce Ilov3Splat, a novel framework for instance-level open-vocabulary 3D scene understanding built on 3D Gaussian Splatting (3D-GS). Most prior work depends on 2D rendering-based matching or point-level semantic association, which undermines cross-view consistency, lacks coherent instance-level reasoning, and limits precision in downstream 3D tasks. To address these limitations, our method jointly optimizes scene geometry and semantic representations by augmenting Gaussian splats with view-consistent feature fields. Specifically, we leverage multi-resolution hash embedding to efficiently encode language-aligned CLIP features, enabling dense and coherent language grounding in 3D space. We further train an instance feature field using contrastive loss over SAM masks, supporting fine-grained object distinction across views. At inference time, CLIP-encoded queries are matched against the learned features, followed by two-stage 3D clustering to retrieve relevant Gaussian groups. This enables our framework to identify arbitrary objects in 3D scenes based on natural language descriptions, without requiring category supervision or manual annotations. Experiments on standard benchmarks demonstrate that Ilov3Splat outperforms prior open-vocabulary 3D-GS methods in both object selection and instance segmentation, offering a flexible and accurate solution for language-driven 3D scene understanding. Project page: https://csiro-robotics.github.io/Ilov3Splat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.12494v1">Revisiting Photometric Ambiguity for Accurate Gaussian-Splatting Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-12
      | 💬 Accepted at ICML 2026. Project page: https://fictionarry.github.io/AmbiSuR-Proj/
    </div>
    <details class="paper-abstract">
      Surface reconstruction with differentiable rendering has achieved impressive performance in recent years, yet the pervasive photometric ambiguities have strictly bottlenecked existing approaches. This paper presents AmbiSuR, a framework that explores an intrinsic solution upon Gaussian Splatting for the photometric ambiguity-robust surface 3D reconstruction with high performance. Starting by revisiting the foundation, our investigation uncovers two built-in primitive-wise ambiguities in representation, while revealing an intrinsic potential for ambiguity self-indication in Gaussian Splatting. Stemming from these, a photometric disambiguation is first introduced, constraining ill-posed geometry solution for definite surface formation. Then, we propose an ambiguity indication module that unleashes the self-indication potential to identify and further guide correcting underconstrained reconstructions. Extensive experiments demonstrate our superior surface reconstructions compared to existing methods across various challenging scenarios, excelling in broad compatibility. Project: https://fictionarry.github.io/AmbiSuR-Proj/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.12437v1">3D Gaussian Splatting for Efficient Retrospective Dynamic Scene Novel View Synthesis with a Standardized Benchmark</a></div>
    <div class="paper-meta">
      📅 2026-05-12
      | 💬 Accepted for publication at CVPR 2026; 4D World Models Workshop. Draft info: 14 pages, 4 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Retrospective novel view synthesis (NVS) of dynamic scenes is fundamental to applications such as sports. Recent dynamic 3D Gaussian Splatting (3DGS) approaches introduce temporally coupled formulations to enforce motion coherence across time. In this paper, we argue that, in a synchronized multi-view (MV) setting typical of sports, the dynamic scene at each time step is already strongly geometrically constrained. We posit that the availability of calibrated, synchronized viewpoints provides sufficient spatial consistency, and therefore, explicit temporal coupling, or complex multi-body constraints seems unnecessary for retrospective NVS. To this end, we propose an approach tailored for synchronized MV dynamic scene. By initializing the SfM-derived point cloud at the start time and propagating optimized Gaussians over time, we show that efficient retrospective NVS can be achieved without imposing a temporal deformation constraint. Complementing our methodological contribution, we introduce a Dynamic MV dataset framework built on Blender for reproducible NeRF and 3DGS research. The framework generates high-quality, synchronized camera rigs and exports training-ready datasets in standard formats, eliminating inconsistencies in coordinate conventions and data pipelines. Using the framework, we construct a dynamic benchmark suite and evaluate representative NeRF and 3DGS approaches under controlled conditions. Together, we show that, under a synchronized MV setup, efficient retrospective dynamic scene NVS can be achieved using 3DGS. At the same time, the dataset-generation framework enables reproducible and principled benchmarking of dynamic NVS methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.12399v1">GeoQuery: Geometry-Query Diffusion for Sparse-View Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-12
      | 💬 Accept to SIGGRAPH 2026 Conference Track
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a prominent paradigm for 3D reconstruction and novel view synthesis. However, it remains vulnerable to severe artifacts when trained under sparse-view constraints. While recent methods attempt to rectify artifacts in rendered views using image diffusion models, they typically rely on multi-view self-attention to retrieve information from reference images. We observe that this mechanism often fails when the rendered novel views output by 3DGS are heavily corrupted: damaged query features lead to erroneous cross-view retrieval, resulting in inconsistent rendering refinement. To address this, we propose GeoQuery, a geometry-guided diffusion framework that integrates generative priors with explicit geometric cues via a novel Geometry-guided Cross-view Attention (GCA) mechanism. First, by leveraging predicted depth maps and camera poses, we construct a geometry-induced correspondence field to sample reference features, forming a geometry-aligned proxy query that replaces the corrupted rendering features. Furthermore, we design a new cross-view feature aggregation pipeline, in which we restrict the cross-view attention to a local window around each proxy query to effectively retrieve useful features while suppressing spurious matches. GeoQuery can be seamlessly integrated into existing diffusion-based pipelines, enabling robust reconstruction even under extreme view sparsity. Extensive experiments on sparse-view novel view synthesis and rendering artifact removal demonstrate the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.12144v1">PoseCompass: Intelligent Synthetic Pose Selection for Visual Localization</a></div>
    <div class="paper-meta">
      📅 2026-05-12
    </div>
    <details class="paper-abstract">
      In visual localization, Absolute Pose Regression (APR) enables real-time 6-DoF camera pose inference from single images, yet critically depends on fine-tuning data quality and coverage. While recent methods leverage 3D Gaussian Splatting (3DGS) for novel view synthesis-based data augmentation, random sampling generates redundant views and noisy samples from poorly reconstructed regions. To mitigate this research gap, we propose PoseCompass, an intelligent pose selection pipeline for 3DGS-based APR. PoseCompass formulates synthetic pose selection and derives a value-based pose ranking mechanism to identify informative poses. The ranking integrates three dimensions: Localization Difficulty, favoring challenging regions; Coverage Novelty, exploring under-sampled areas; and Rendering Observability, filtering artifacts and noise. PoseCompass then generates trajectory-constrained candidates, selects the top-K ranked poses, and synthesizes views using 3DGS with lightweight diffusion-based alignment. Finally, the pose regressor is fine-tuned on mixed real and synthetic data. We evaluate PoseCompass on 7-Scenes, where it reduces adaptation time from 15.2 to 5.1 minutes, a 3x speedup, while cutting median pose errors by 53.8 percent and significantly outperforming random baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10360v2">DySurface: Consistent 4D Surface Reconstruction via Bridging Explicit Gaussians and Implicit Functions</a></div>
    <div class="paper-meta">
      📅 2026-05-12
    </div>
    <details class="paper-abstract">
      While novel view synthesis (NVS) for dynamic scenes has seen significant progress, reconstructing temporally consistent geometric surfaces remains a challenge. Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) offer powerful dynamic scene rendering capabilities; however, relying solely on photometric optimization often leads to geometric ambiguities. This results in discontinuous surfaces, severe artifacts, and broken surfaces over time. To address these limitations, we present DySurface, a novel framework that bridges the effectiveness of explicit Gaussians with the geometric fidelity of implicit Signed Distance Functions (SDFs) in dynamic scenes. Our approach tackles the structural discrepancy between the forward deformation of 3DGS ($canonical \rightarrow dynamic$) and the backward deformation required for volumetric SDF rendering ($dynamic \rightarrow canonical$). Specifically, we propose the VoxGS-DSDF branch that leverages deformed Gaussians to construct a dynamic sparse voxel grid, providing explicit geometric guidance to the implicit SDF field. This explicit anchoring effectively regularizes the volumetric rendering process, significantly improving surface reconstruction quality, with watertight boundaries and detailed representations. Quantitative and qualitative experiments demonstrate that DySurface significantly outperforms state-of-the-art baselines in geometric accuracy metrics while maintaining competitive rendering performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.11594v1">PointForward: Feedforward Driving Reconstruction through Point-Aligned Representations</a></div>
    <div class="paper-meta">
      📅 2026-05-12
    </div>
    <details class="paper-abstract">
      High-fidelity reconstruction of driving scenes is crucial for autonomous driving. While recent feedforward 3D Gaussian Splatting (3DGS) methods enable fast reconstruction, their per-pixel Gaussian prediction paradigm often suffers from multi-view inconsistency and layering artifacts. Moreover, existing methods often model dynamic instances via dense flow prediction, which lacks explicit cross-view correspondence and instance-level consistency. In this paper, we propose PointForward, a feedforward driving reconstruction framework through point-aligned representations. Unlike pixel-aligned methods, we initialize sparse 3D queries in world space and aggregate multi-view image information via spatial-temporal fusion onto these queries, enforcing explicit cross-view consistency in a single feedforward pass. To handle scene dynamics, we introduce scene graphs that explicitly organize moving instances during reconstruction. By leveraging 3D bounding boxes, our method enables instance-level motion propagation and temporally consistent dynamic representations. Extensive experiments demonstrate that PointForward achieves state-of-the-art performance on large-scale driving benchmarks. The code will be available upon the publication of the paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.11520v1">PointGS: Semantic-Consistent Unsupervised 3D Point Cloud Segmentation with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-12
      | 💬 Accepted by Computer Vision and Pattern Recognition (CVPR) 2026
    </div>
    <details class="paper-abstract">
      Unsupervised point cloud segmentation is critical for embodied artificial intelligence and autonomous driving, as it mitigates the prohibitive cost of dense point-level annotations required by fully supervised methods. While integrating 2D pre-trained models such as the Segment Anything Model (SAM) to supplement semantic information is a natural choice, this approach faces a fundamental mismatch between discrete 3D points and continuous 2D images. This mismatch leads to inevitable projection overlap and complex modality alignment, resulting in compromised semantic consistency across 2D-3D transfer. To address these limitations, this paper proposes PointGS, a simple yet effective pipeline for unsupervised 3D point cloud segmentation. PointGS leverages 3D Gaussian Splatting as a unified intermediate representation to bridge the discrete-continuous domain gap. Input sparse point clouds are first reconstructed into dense 3D Gaussian spaces via multi-view observations, filling spatial gaps and encoding occlusion relationships to eliminate projection-induced semantic conflation. Multi-view dense images are rendered from the Gaussian space, with 2D semantic masks extracted via SAM, and semantics are distilled to 3D Gaussian primitives through contrastive learning to ensure consistent semantic assignments across different views. The Gaussian space is aligned with the original point cloud via two-step registration, and point semantics are assigned through nearest-neighbor search on labeled Gaussians. Experiments demonstrate that PointGS outperforms state-of-the-art unsupervised methods, achieving +0.9% mIoU on ScanNet-V2 and +2.8% mIoU on S3DIS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.11489v1">3DGS$^3$: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering</a></div>
    <div class="paper-meta">
      📅 2026-05-12
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables high-quality real-time 3D rendering but faces challenges in efficiently scaling to ultra-dense scenes and high-resolution due to computational bottlenecks that limit its use in latency-sensitive applications. Instead of optimizing the splatting pipeline itself, we propose \textbf{3DGS$^3$}, a unified post-rendering framework that jointly performs super sampling and frame interpolation through differentiable processing of low-resolution outputs to achieve both high-resolution and high-frame-rate rendering. Our \textbf{Gradient\- \-Aware Super Sampling (GASS)} module leverages the continuous differentiability of 3DGS to extract image gradients that guide a GRU-based refinement network to enable high-fidelity super sampling. Furthermore, a \textbf{Lightweight Temporal Frame Interpolation (LTFI)} module based on a compact U-Net-like backbone fuses temporal and differentiable spatial cues from consecutive frames to synthesize temporally coherent intermediate frames. Experiments on public datasets demonstrate that 3DGS$^3$ achieves superior rendering efficiency and visual quality when compared with state-of-the-art methods and remains compatible with existing 3DGS acceleration techniques. The code will be publicly released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14199v2">Learnable Multi-level Discrete Wavelet Transforms for 3D Gaussian Splatting Frequency Modulation</a></div>
    <div class="paper-meta">
      📅 2026-05-12
      | 💬 Accepted to EUSIPCO 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful approach for novel view synthesis. However, the number of Gaussian primitives often grows substantially during training as finer scene details are reconstructed, leading to increased memory and storage costs. Recent coarse-to-fine strategies regulate Gaussian growth by modulating the frequency content of the ground-truth images. In particular, AutoOpti3DGS employs the learnable Discrete Wavelet Transform (DWT) to enable data-adaptive frequency modulation. Nevertheless, its modulation depth is limited by the 1-level DWT, and jointly optimizing wavelet regularization with 3D reconstruction introduces gradient competition that promotes excessive Gaussian densification. In this paper, we propose a multi-level DWT-based frequency modulation framework for 3DGS. By recursively decomposing the low-frequency subband, we construct a deeper curriculum that provides progressively coarser supervision during early training, consistently reducing Gaussian counts. Furthermore, we show that the modulation can be performed using only a single scaling parameter, rather than learning the full 2-tap high-pass filter. Experimental results on standard benchmarks demonstrate that our method further reduces Gaussian counts while maintaining competitive rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.11427v1">PD-4DGS:Progressive Decomposition of 4D Gaussian Splatting for Bandwidth-Adaptive Dynamic Scene Streaming</a></div>
    <div class="paper-meta">
      📅 2026-05-12
    </div>
    <details class="paper-abstract">
      4D Gaussian Splatting (4DGS) enables high-quality dynamic novel view synthesis, yet current models remain monolithic bitstreams that clients must download in full before any frame can be rendered, causing black-screen waits of tens to hundreds of seconds on mobile bandwidth and leaving 4DGS incompatible with modern adaptive-bitrate delivery. Progressive 3DGS compression alleviates this for static scenes, but it acts only on spatial anchors and cannot partition the temporal deformation networks that dominate dynamic-scene size. We present PD-4DGS, the first framework for progressive compression and on-demand transmission of 4DGS. Hierarchical Deformation Decomposition (HDD) externalises the coarse-to-fine motion hierarchy already latent in 4DGS into three independently transmittable layers -- a static scaffold, a global deformation, and a local refinement -- so that any prefix of the bitstream is already renderable, turning a single training run into a scalable, DASH/HLS-compatible bitstream. A Gaussian-entropy attribute rate-distortion loss together with a temporal mask consistency regulariser shrink the base layer while suppressing low-bitrate flicker; a capacity-weighted rollout schedule, gated online by a learnt activation rate rho, then prevents deformation-network under-training without any per-scene hyperparameter. On the Dycheck iPhone benchmark, PD-4DGS cuts the streamed bitstream by >60% at matched rendering fidelity and reduces first-frame latency from 73--930 s to ~1.7 s on a 2 Mbps link, uniquely enabling true on-demand progressive streaming for 4DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.11424v1">VidSplat: Gaussian Splatting Reconstruction with Geometry-Guided Video Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2026-05-12
      | 💬 Accepted by SIGGRAPH Conference 2026. Project Page: https://tangjm24.github.io/VidSplat
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has achieved remarkable progress in multi-view surface reconstruction, yet it exhibits notable degradation when only few views are available. Although recent efforts alleviate this issue by enhancing multi-view consistency to produce plausible surfaces, they struggle to infer unseen, occluded, or weakly constrained regions beyond the input coverage. To address this limitation, we present VidSplat, a training-free generative reconstruction framework that leverages powerful video diffusion priors to iteratively synthesize novel views that compensate for missing input coverage, and thereby recover complete 3D scenes from sparse inputs. Specifically, we tackle two key challenges that enable the effective integration of generation and reconstruction. First, for 3D consistent generation, we elaborate a training-free, stage-wise denoising strategy that adaptively guides the denoising direction toward the underlying geometry using the rendered RGB and mask images. Second, to enhance the reconstruction, we develop an iterative mechanism that samples camera trajectories, explores unobserved regions, synthesizes novel views, and supplements training through confidence weighted refinement. VidSplat performs robustly to sparse input and even a single image. Extensive experiments on widely used benchmarks demonstrate our superior performance in sparse-view scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.11266v1">PG-3DGS: Optimizing 3D Gaussian Splatting to Satisfy Physics Objectives</a></div>
    <div class="paper-meta">
      📅 2026-05-11
      | 💬 Submitted to Artificial Intelligence. 52 pages
    </div>
    <details class="paper-abstract">
      Recent advances in Gaussian Splatting have enabled fast, high-fidelity 3D scene generation, yet these methods remain purely visual and lack an understanding of how shapes behave in the physical world. We introduce Physics-Guided 3D Gaussian Splatting (PG-3DGS), a framework that couples differentiable physics simulation with 3D Gaussian representations to generate 3D structures satisfying physics functionalities. By allowing physical objectives to guide the shape optimization process alongside visual losses, our approach produces geometries that are not only photometrically accurate but also physically functional. The model learns to adjust shapes so that the generated objects exhibit physically meaningful behaviors, for example, teapots that can pour and airplanes that can generate lift, without sacrificing visual quality. Experiments on pouring and aerodynamic lift tasks show that PG-3DGS improves physical functionality while preserving visual quality. In addition to simulation gains, bench-top physical lift tests with 3D-printed aircraft (Cessna, B-2 Spirit, and paper plane) under identical airflow conditions show higher scale-measured lift for PG-3DGS, generated structures than an appearance-matching baseline in all three cases. Our unified framework connects appearance-based reconstruction with physics-based reasoning, enabling end-to-end generation of 3D structures that both look realistic and function correctly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.11144v1">Forecast-aware Gaussian Splatting for Predictive 3D Representation in Language-Guided Pick-and-Place Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-05-11
    </div>
    <details class="paper-abstract">
      We introduce Forecast-aware Gaussian Splatting (Forecast-GS), a predictive 3D representation framework for language-conditioned robotic manipulation. While recent manipulation systems have made progress by grounding language instructions into robot affordances, value maps, or relational keypoint constraints, they usually reason over the current scene and do not explicitly model the task-completed state. This limitation is critical when success depends on satisfying spatial and semantic goals under partial observations, where the robot must evaluate whether a candidate action leads to a feasible task-consistent outcome. We validate Forecast-GS on real-world pick-and-place manipulation tasks, including Cutter-to-Box, Apple-to-Bowl, and Sponge-to-Tray. For each task, we conduct 25 real-world trials under varied initial object configurations using the same robot platform and sensing setup. Forecast-GS with automatic candidate selection achieves success rates of 21/25, 23/25, and 16/25 on the three tasks, respectively, outperforming the ReKep baseline, which achieves 15/25, 19/25, and 10/25. A diagnostic human-assisted setting further improves success rates to 23/25, 24/25, and 19/25, suggesting that candidate generation is effective while automatic ranking remains imperfect. These results suggest that explicitly forecasting task-completed 3D states enables more reliable action evaluation, while the gap between automatic and human-assisted selection indicates that robust final-state ranking remains an important challenge for fully autonomous manipulation. Overall, Forecast-GS provides an interpretable bridge between language understanding, 3D perception, and robotic manipulation planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10760v1">MAGS-SLAM: Monocular Multi-Agent Gaussian Splatting SLAM for Geometrically and Photometrically Consistent Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-11
    </div>
    <details class="paper-abstract">
      Collaborative photorealistic 3D reconstruction from multiple agents enables rapid large-scale scene capture for virtual production and cooperative multi-robot exploration. While recent 3D Gaussian Splatting (3DGS) SLAM algorithms can generate high-fidelity real-time mapping, most of the existing multi-agent Gaussian SLAM methods still rely on RGB-D sensors to obtain metric depth and simplify cross-agent alignment, which limits the deployment on lightweight, low-cost, or power-constrained robotic platforms. To address this challenge, we propose MAGS-SLAM, the first RGB-only multi-agent 3DGS SLAM framework for collaborative scene reconstruction. Each agent independently builds local monocular Gaussian submaps and transmits compact submap summaries rather than raw observations or dense maps. To facilitate robust collaboration in the presence of monocular scale ambiguity, our framework integrates compact submap communication, geometry- and appearance-aware loop verification, and occupancy-aware Gaussian fusion, enabling coherent global reconstruction without active depth sensors. We further introduce ReplicaMultiagent Plus benchmark for evaluating collaborative Gaussian SLAM. Intensive experiments on synthetic and real-world datasets show that MAGS-SLAM achieves competitive tracking accuracy and comparable or superior rendering quality to state-of-the-art RGB-D collaborative Gaussian SLAM methods while relying only RGB images.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10715v1">UAV-Assisted Scan-to-Simulation for Landslides Using Physics-Informed Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-11
    </div>
    <details class="paper-abstract">
      Landslide monitoring and simulation play an important role in urban safety assessment and disaster prevention. Existing landslide simulation pipelines typically rely on digital elevation model and mesh-based representations, which are suitable for geometric analysis, but often lack visual realism. This limitation reduces their effectiveness in interactive applications, hazard communication, and public education. In this paper, we propose a UAV-based scan-to-simulation framework that bridges photorealistic scene capture and physics-based landslide simulation through 3DGS. Specifically, our pipeline includes four stages: (1) UAV-based acquisition of slope imagery, (2) reconstruction of a low-anisotropy 3DGS scene representation, (3) volumetric conversion of the target simulation region by filling the interior of the surface-based model, and (4) integration with the Material Point Method (MPM) for landslide simulation. We validate the proposed framework on a real landslide site in Hong Kong that experienced a severe landslide event. The results show that our method supports both realistic visual reconstruction and effective simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10705v1">TransmissiveGS: Residual-Guided Disentangled Gaussian Splatting for Transmissive Scene Reconstruction and Rendering</a></div>
    <div class="paper-meta">
      📅 2026-05-11
    </div>
    <details class="paper-abstract">
      Transmissive scenes are ubiquitous in daily life, yet reconstructing and rendering them remains highly challenging due to the inherent entanglement between near-field reflections from the surrounding environment on the transmissive surface, and the transmitted content of the scene behind it. This coupling gives rise to dual surface geometries and dual radiance components within each observation, posing ambiguities for standard methods. We present TransmissiveGS, a novel framework for disentangled reconstruction and rendering of transmissive scenes. Specifically, we model the scene with a dual-Gaussian representation and introduce a deferred shading function to jointly render the two Gaussian components. To separate reflection and transmission, we exploit the inherent multi-view inconsistency of reflections and leverage the residuals from reconstructing multi-view consistent content as cues for disentangled geometry and appearance modeling. We further propose a reflection light field that enables high-fidelity estimation of near-field reflections. During training, we introduce a high-frequency regularization to preserve fine details. We also contribute a new synthetic dataset for evaluating transmissive surface reconstruction. Experiments on both synthetic and real-world scenes demonstrate that TransmissiveGS consistently outperforms prior Gaussian Splatting-based methods in both reconstruction and rendering quality for transmissive scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11969v2">AstroSplat: Physics-Based Gaussian Splatting for Rendering and Reconstruction of Small Celestial Bodies</a></div>
    <div class="paper-meta">
      📅 2026-05-11
      | 💬 10 pages, 6 figures, conference
    </div>
    <details class="paper-abstract">
      Image-based surface reconstruction and characterization are crucial for missions to small celestial bodies (e.g., asteroids), as it informs mission planning, navigation, and scientific analysis. Recent advances in Gaussian splatting enable high-fidelity neural scene representations but typically rely on a spherical harmonic intensity parameterization that is strictly appearance-based and does not explicitly model material properties or light-surface interactions. We introduce AstroSplat, a physics-based Gaussian splatting framework that integrates planetary reflectance models to improve the autonomous reconstruction and photometric characterization of small-body surfaces from in-situ imagery. The proposed framework is validated on real imagery taken by NASA's Dawn mission, where we demonstrate superior rendering performance and surface reconstruction accuracy compared to the typical spherical harmonic parameterization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10485v1">VEGA: Visual Encoder Grounding Alignment for Spatially-Aware Vision-Language-Action Models</a></div>
    <div class="paper-meta">
      📅 2026-05-11
    </div>
    <details class="paper-abstract">
      Precise spatial reasoning is fundamental to robotic manipulation, yet the visual backbones of current vision-language-action (VLA) models are predominantly pretrained on 2D image data without explicit 3D geometric supervision, resulting in representations that lack accurate spatial awareness. Existing implicit spatial grounding methods partially address this by aligning VLA features with those of 3D-aware foundation models, but they rely on empirical layer search and perform alignment on LLM-level visual tokens where spatial structure has already been entangled with linguistic semantics, limiting both generalizability and geometric interpretability. We propose VEGA (Visual Encoder Grounding Alignment), a simple yet effective framework that directly aligns the output of the VLA's visual encoder with spatially-aware features from DINOv2-FiT3D, a DINOv2 model fine-tuned with multi-view consistent 3D Gaussian Splatting supervision. By performing alignment at the visual encoder output level, VEGA grounds spatial awareness before any linguistic entanglement occurs, offering a more interpretable and principled alignment target. The alignment is implemented via a lightweight projector trained with a cosine similarity loss alongside the standard action prediction objective, and is discarded at inference time, introducing no additional computational overhead. Extensive experiments on simulation benchmark and real-world manipulation tasks demonstrate that VEGA consistently outperforms existing implicit spatial grounding baselines, establishing a new state-of-the-art among implicit spatial grounding methods for VLA models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10307v1">PaMoSplat: Part-Aware Motion-Guided Gaussian Splatting for Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-11
      | 💬 Accepted by TCSVT. Project Url: https://pamosplat.github.io
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction represents a fundamental yet demanding challenge in computer vision and robotics. While recent progress in 3DGS-based methods has advanced dynamic scene modeling, obtaining high-fidelity rendering and accurate tracking in scenarios with substantial, intricate motions remains significantly challenging. To address these challenges, we propose PaMoSplat, a novel dynamic Gaussian splatting framework incorporating part awareness and motion priors. Our approach is grounded in two key observations: 1) Parts serve as primitives for scene deformation, and 2) Motion cues from optical flow can effectively guide part motion. Specifically, PaMoSplat initializes by lifting multi-view segmentation masks into 3D space via graph clustering, establishing coherent Gaussian parts. For subsequent timestamps, we leverage a differential evolutionary algorithm to estimate the rigid motion of these parts using multi-view optical flow cues, providing a robust warm-start for further optimization. Additionally, PaMoSplat introduces an adaptive iteration count mechanism, internal learnable rigidity, and flow-supervised rendering loss to accelerate and optimize the training process. Comprehensive evaluations across diverse scenes, including real-world environments, demonstrate that PaMoSplat delivers superior rendering quality, improved tracking precision, and faster convergence compared to existing methods. Furthermore, it enables multiple part-level downstream applications, such as 4D scene editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.00408v2">Beyond Heuristics: Learnable Density Control for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-11
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has demonstrated impressive real-time rendering performance, its efficacy remains constrained by a reliance on heuristic density control. Despite numerous refinements to these handcrafted rules, such methods inherently lack the flexibility to adapt to diverse scenes with complex geometries. In this paper, we propose a paradigm shift for density control from rigid heuristics to fully learnable policies. Specifically, we introduce \textbf{LeGS}, a framework that reformulates density control as a parameterized policy network optimized via Reinforcement Learning (RL). Central to our approach is the tailored effective reward function grounded in sensitivity analysis, which precisely quantifies the marginal contribution of individual Gaussians to reconstruction quality. To maintain computational tractability, we derive a closed-form solution that reduces the complexity of reward calculation from $O(N^2)$ to $O(N)$. Extensive experiments on the Mip-NeRF 360, Tanks \& Temples, and Deep Blending datasets demonstrate that \textbf{LeGS} significantly outperforms state-of-the-art methods, striking a superior balance between reconstruction quality and efficiency. The code will be released at https://github.com/AaronNZH/LeGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00560v2">4D Neural Voxel Splatting: Dynamic Scene Rendering with Voxelized Guassian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-11
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Although 3D Gaussian Splatting (3D-GS) achieves efficient rendering for novel view synthesis, extending it to dynamic scenes still results in substantial memory overhead from replicating Gaussians across frames. To address this challenge, we propose 4D Neural Voxel Splatting (4D-NVS), which combines voxel-based representations with neural Gaussian splatting for efficient dynamic scene modeling. Instead of generating separate Gaussian sets per timestamp, our method employs a compact set of neural voxels with learned deformation fields to model temporal dynamics. The design greatly reduces memory consumption and accelerates training while preserving high image quality. We further introduce a novel view refinement stage that selectively improves challenging viewpoints through targeted optimization, maintaining global efficiency while enhancing rendering quality for difficult viewing angles. Experiments demonstrate that our method outperforms state-of-the-art approaches with significant memory reduction and faster training, enabling real-time rendering with superior visual fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09956v1">SDTalk: Structured Facial Priors and Dual-Branch Motion Fields for Generalizable Gaussian Talking Head Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-05-11
      | 💬 5 pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      High-quality, real-time talking head synthesis remains a fundamental challenge in computer vision. Existing reconstruction- and rendering-based methods typically rely on identity-specific models, limiting cross-identity generalization. To address this issue, we propose SDTalk, a one-shot 3D Gaussian Splatting (3DGS)-based framework that generalizes to unseen identities without personalized training or fine-tuning. Our framework comprises two modules with a two-stage training strategy. In the first stage, we incorporate structured facial priors into the reconstruction module and separately predict 3DGS parameters for visible and occluded regions, enabling complete head reconstruction from a single image. In the second stage, we introduce a dual-branch motion field to model coarse and fine facial dynamics, improving detail fidelity and lip synchronization. Experiments demonstrate that SDTalk surpasses existing methods in both visual quality and inference efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07203v2">From Pixels to Primitives: Scene Change Detection in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-11
      | 💬 Project Page: https://chumsy0725.github.io/GS-DIFF/
    </div>
    <details class="paper-abstract">
      Scene change detection methods built on Gaussian splatting universally follow a render-then-compare paradigm: the pre-change scene is rendered into 2D and compared against post-change images via pixel or feature residuals. This change detection problem with Gaussian Splatting has been treated as a question about pixels; we treat it as a question about primitives. We provide direct evidence that native primitive attributes alone -- position, anisotropic covariance, and color -- carry sufficient signal for scene change detection. What makes primitive-space comparison hard is the under-constrained nature of Gaussian splatting representation: independent optimizations yield primitive solutions whose count, positions, shapes, and colors differ even where nothing has changed. We address this challenge with anisotropic models of geometric and photometric drift, complemented by a per-primitive observability term that reflects the extent to which each Gaussian is constrained by the camera geometry. Operating directly on primitives gives our method, GD-DIFF, two properties that distinguish it from render-then-compare methods. First, change maps are multi-view consistent by construction, where prior work had to learn this through an additional optimization objective. Second, geometric and appearance changes are scored separately, identifying not just where but what kind of change occurred, distinguishing structural changes (e.g., an added object) from surface-level ones (e.g., a color change) without supervision or external model dependencies. On real-world benchmarks, GS-DIFF surpasses the prior state-of-the-art approach by $\sim$17% in mean Intersection over Union.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09688v1">ConFixGS: Learning to Fix Feedforward 3D Gaussian Splatting with Confidence-Aware Diffusion Priors in Driving Scenes</a></div>
    <div class="paper-meta">
      📅 2026-05-10
      | 💬 28 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Feedforward 3D Gaussian Splatting (3DGS) often struggles in trajectory-based sparse-view driving scenes. Existing Gaussian repair methods mainly target optimization-based 3DGS, while diffusion-based repair is typically restricted to iterative refinement near observed viewpoints, leaving feedforward 3DGS repair underexplored. We propose ConFixGS, a plug-and-play method that learns to fix feedforward 3DGS with confidence-aware diffusion priors. Starting from a pretrained feedforward model, ConFixGS generates diffusion-enhanced local pseudo-targets and validates them through reprojection-based cross-checking against support views. The resulting dense confidence maps guide refinement, enhancing reliable details while suppressing hallucinated or inconsistent evidence. On Waymo, nuScenes, and KITTI, ConFixGS improves challenging novel view synthesis, with PSNR gains of up to 3.68 dB and FID reduced by nearly half. Our results highlight confidence-aware fusion of generative priors and support-view consistency as a key principle for robust feedforward 3D driving scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09662v1">BEA-GS: BEyond RAdiance Supervision in 3DGS for Precise Object Extraction</a></div>
    <div class="paper-meta">
      📅 2026-05-10
      | 💬 CVPR 2026 Highlight
    </div>
    <details class="paper-abstract">
      Most Gaussian Splatting techniques that provide a 3D semantic representation of the scene do not optimize the underlying 3D geometry, making object-level editing or asset extraction challenging. Recent methods, such as COBGS, Trace3D, ObjectGS, acknowledge this limitation and propose approaches that modify the scene's geometry to represent the underlying semantics. We advance this concept further by proposing a novel solution that provides near perfect boundaries in object extraction. We do so by introducing two new losses in the optimization that take care of: 1) a loss that modifies the geometry of visible Gaussians to respect semantic boundaries, and 2) a loss that adjusts the geometry of non-visible Gaussians that appear once the object is extracted. Our first loss propagates gradients directly through the rasterization, allowing for seamless integration within the optimization of the Gaussian parameters. The second loss also propagates gradients to Gaussian parameters but does so without passing through the rasterization, enabling modification of the scene's geometry even when little transmittance reaches a Gaussian (partial or non-visible). Exhaustive comparisons with 12 state of the art methods across 4 datasets, using six metrics, demonstrate that our approach produces overall the best boundary segmentation to date.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09362v1">FrameTwin: Curve-Anchored Gaussian Alignment from Sparse Views for Adaptive Wireframe 3D Printing</a></div>
    <div class="paper-meta">
      📅 2026-05-10
    </div>
    <details class="paper-abstract">
      We present FrameTwin, a curve-anchored Gaussian alignment framework that uses sparse-view images to close the control loop for adaptive wireframe 3D printing. Our key idea is to capture the deformation of thin wireframe structures from sparse-view images using Gaussian kernels anchored to parametric curves, yielding a compact and geometry-aware encoding that explicitly captures strut topology. Driven by a differentiable rendering pipeline, FrameTwin estimates a neural deformation field that aligns the partially printed target model with the deformed structure observed during fabrication, where the optimized curve-Gaussian representation serves as a digital twin of the evolving wireframe. Unlike general Gaussian-splatting approaches, our formulation constrains kernel placement along parametric curves, substantially reducing the ambiguity inherent in sparse-view observations of thin structures. The resultant deformation-field alignment enforces global consistency across all struts. By using the estimated deformation field to blend the distorted printed geometry with the remaining unprinted geometry, FrameTwin enables adaptive updates to future printing trajectories. We demonstrate that FrameTwin can robustly capture and compensate for deformation in wireframe models fabricated using a robotized 3D printing system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09299v1">LagrangianSplats: Divergence-Free Transport of Gaussian Primitives for Fluid Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-10
    </div>
    <details class="paper-abstract">
      Reconstructing 3D fluid velocity fields from sparse 2D video observations is a highly ill-posed inverse problem, demanding both transport consistency with observed motion and physical validity under fluid laws. Existing methods typically impose these constraints through soft penalties, often leading to compromised accuracy and convergence issues. We introduce a reconstruction framework that structurally enforces both constraints. Specifically, we parameterize the reconstructed velocity using a continuous Divergence-Free Kernel representation, driving the advection of a Lagrangian 3D Gaussian Splatting representation. This formulation intrinsically guarantees both flow incompressibility and long-range transport coherence by construction. To enable the efficient optimization of such a constrained system, we introduce a novel Sliding Window scheme that propagates gradients over meaningful temporal horizons while maintaining tractable training costs. Experiments on synthetic and real-world datasets demonstrate that our method outperforms state-of-the-art baselines in both transport consistency and physical accuracy, enabling applications such as high-quality re-simulation and flow analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09279v1">CAGS: Color-Adaptive Volumetric Video Streaming with Dynamic 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-10
      | 💬 SIGGRAPH 2026 Conference Paper. Code is available at https://github.com/yindaheng98/ColorAdaptiveGaussianSplatting
    </div>
    <details class="paper-abstract">
      Volumetric video (VV) streaming enables real-time, immersive access to remote 3D environments, powering telepresence, ecological monitoring, and robotic teleoperation. These applications turn VV streaming into a real-time interface to remote physical environments, imposing new system-level demands for photorealistic scene representation, low-latency interaction, and robust performance under heterogeneous networks. 3D Gaussian Splatting (3DGS) has been widely used for real-time photorealistic rendering, offering superior visual quality and rendering performance, but it faces challenges due to bandwidth consumption. Furthermore, as the foundation of adaptive VV streaming, existing Levels of Detail (LoD) methods based on density are not well-suited to Gaussian representations, leading to visible gaps and severe quality degradation. Recent studies have also explored attribute compression techniques to reduce bandwidth consumption. Our preliminary studies reveal that aggressive attribute compression primarily causes color distortion, which can be effectively corrected in the rendered image using a reference image. Motivated by these findings, we propose a novel Color-Adaptive scheme for adaptive VV streaming that uses vector quantization (VQ) to establish LoDs and correct color distortions with low-resolution reference images. We further present CAGS, an adaptive VV streaming system compatible with diverse Gaussian representations, which integrates the Color-Adaptive scheme by rendering reference images on the streaming server and performing color restoration on the client. Extensive experiments on our prototype system demonstrate that CAGS outperforms the existing adaptive streaming systems in PSNR by 5$\sim$20 dB under fluctuating bandwidth, operates significantly faster than existing scalable Gaussian compression methods, and generalizes across different Gaussian representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09024v1">Relightable Gaussian Splatting for Virtual Production Using Image-Based Illumination</a></div>
    <div class="paper-meta">
      📅 2026-05-09
    </div>
    <details class="paper-abstract">
      Virtual production (VP) use LED walls to provide both background imagery and image-based lighting. While this enables on-set compositing, it couples lighting to background and scene appearance, limiting flexibility for downstream editing. In addition, inverse rendering conventionally relies on physically-based rendering to estimates 3D geometry and lighting, using environment maps. However, these maps are typically low-resolution and assume far-field lighting. In VP, with near-field and high-resolution image-based lighting, this can lead to inaccuracies and introduce complexities when editing. Addressing this, we propose a VP-specific framework for 3D reconstruction and relighting using Gaussian Splatting. This uses the known background imagery to condition the relighting process. This avoids relying on environment maps and reduces compositing to a background-image editing task. To realize our framework, we introduce a process (and associated dataset) that captures real VP scenes under varying background content and illumination conditions. This data is used to decompose a 3D scene into fixed appearance and variable lighting components. The variable lighting process simulates light transport by parameterizing each primitive with a UV coordinate, intensity value and resolution modifier. Using mipmaps, these directly sample the background texture in image space - implicitly capturing reflections and refractions without physically-based rendering. Combined with the fixed appearance component, this allows us to render relit scenes using a Gaussian Splatting rasterizer. Compared to baselines, our approach achieves higher-quality 3D reconstruction and controllable relighting. The method is efficient (<3 GB RAM, <5 GB VRAM, <2 hours training, ~35 FPS) and supports rendering useful arbitrary output variables including depth, lighting intensity, lighting color, and unlit renders.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04549v2">Nix and Fix: Targeting 1000x Compression of 3D Gaussian Splatting with Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2026-05-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) revolutionized novel view rendering. Instead of inferring from dense spatial points, as implicit representations do, 3DGS uses sparse Gaussians. This enables real-time performance but increases space requirements, hindering rate-constrained applications. 3DGS compression emerged as a field aimed at alleviating this issue. While impressive progress has been made, at low rates, compression introduces artifacts that degrade visual quality significantly. We introduce NiFi, a method for extreme 3DGS compression through restoration via artifact-aware, diffusion-based one-step distillation. We show that our method achieves state-of-the-art perceptual quality at extremely low rates, down to 0.1 MB, and towards 1000x rate improvement over 3DGS at comparable perceptual performance. Code is available at: https://github.com/ceteke/nifi
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12592v2">ELoG-GS: Dual-Branch Gaussian Splatting with Luminance-Guided Enhancement for Extreme Low-light 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-09
      | 💬 Our method achieved a ranking of 9 out of 148 participants in Track 1 of the NTIRE 3DRR Challenge, as reported on the official competition website: https://www.codabench.org/competitions/13854/
    </div>
    <details class="paper-abstract">
      This paper presents our approach to the NTIRE 2026 3D Restoration and Reconstruction Challenge (Track 1), which focuses on reconstructing high-quality 3D representations from degraded multi-view inputs. The challenge involves recovering geometrically consistent and photorealistic 3D scenes in extreme low-light environments. To address this task, we propose Extreme Low-light Optimized Gaussian Splatting (ELoG-GS), a robust low-light 3D reconstruction pipeline that integrates learning-based point cloud initialization and luminance-guided color enhancement for stable and photorealistic Gaussian Splatting. Our method incorporates both geometry-aware initialization and photometric adaptation strategies to improve reconstruction fidelity under challenging conditions. Extensive experiments on the NTIRE Track 1 benchmark demonstrate that our approach significantly improves reconstruction quality over the baselines, achieving superior visual fidelity and geometric consistency. The proposed method provides a practical solution for robust 3D reconstruction in real-world degraded scenarios. In the final testing phase, our method achieved a PSNR of 18.6626 and an SSIM of 0.6855 on the official platform leaderboard. Code is available at https://github.com/lyh120/FSGS_EAPGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.13547v3">Turbo-GS: Accelerating 3D Gaussian Fitting for High-Quality Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2026-05-09
      | 💬 Accepted to CVPR 2026. Project page: https://ivl.cs.brown.edu/research/turbo-gs
    </div>
    <details class="paper-abstract">
      Novel-view synthesis plays a crucial role in computer vision with applications in 3D reconstruction, mixed reality, and robotics. Recent approaches, such as 3D Gaussian Splatting (3DGS), have emerged as state-of-the-art solutions, offering high-quality novel view synthesis in real time. However, training 3DGS models remains slow, particularly for high-resolution images, often requiring hours to fit a scene with 200 views. In this work, we aim to accelerate the fitting process by reducing computational overhead and improving learning efficiency. Specifically, we introduce a dilated rendering technique that renders only a subset of pixels instead of the full image, significantly reducing computational costs. To enhance learning efficiency, we develop a convergence-aware budget control mechanism that balances the addition of new Gaussians with the optimization of existing ones. Additionally, to improve densification efficiency and prevent gradient vanishing, we incorporate both positional and appearance errors to improve the effectiveness of densification. With these improvements, we achieve fast 4K-resolution fitting while maintaining, or even improving, novel view rendering quality. Extensive experiments demonstrate that our method achieves significantly faster optimization than existing approaches while preserving high rendering fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.08739v1">ReorgGS: Equivalent Distribution Reorganization for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-09
    </div>
    <details class="paper-abstract">
      A converged 3D Gaussian Splatting (3DGS) model may approximate the target scene while remaining poorly parameterized for further optimization. We identify this failure mode as \emph{parameterization degeneration}: high-opacity floaters attenuate gradients to true surfaces through alpha compositing, and redundant overlapping clusters create strongly coupled parameter blocks with nearly collinear Jacobian responses. These effects explain why continued optimization can plateau even when the model still contains removable artifacts. We propose ReorgGS, an equivalent distribution reorganization method for converged 3DGS models. ReorgGS treats the existing Gaussian set as an empirical probability field, resamples centers from it, estimates local anisotropic covariances with kNN, initializes low opacity, and continues optimization with the original 3DGS renderer and loss. Unlike opacity reset, which only rescales opacity on the old overlap graph, ReorgGS rebuilds centers, covariances, and visibility structure, thereby changing the graph itself. Our analysis shows that distributional equivalence is not optimization equivalence. The reorganized model preserves scene support while improving gradient accessibility under alpha compositing and reducing opacity-weighted overlap, thereby weakening local parameter coupling during subsequent optimization. Under the same additional optimization budget, ReorgGS improves fitting quality at a fixed Gaussian count, suppresses persistent floaters, and reduces rendering overhead from redundant overlap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.08713v1">REAP: Reinforcement-Learning End-to-End Autonomous Parking with Gaussian Splatting Simulator for Real2Sim2Real Transfer</a></div>
    <div class="paper-meta">
      📅 2026-05-09
    </div>
    <details class="paper-abstract">
      In recent years, autonomous parking has made significant advances, yet parking tasks still face challenges in extreme scenarios such as mechanical and dead-end parking slots, often resulting in failures. This is mainly due to traditional parking methods adopting a multistage approach, lacking the ability to optimize the parking problem as a whole. End-to-end methods enable joint optimization across perception and planning modules to eliminate the accumulation of errors, enhancing algorithm performance in extreme scenarios. Although several end-to-end parking methods use imitation or reinforcement learning, the former is limited by data cost and distribution coverage, while the latter suffers from inefficient exploration. To address these challenges, we propose a Reinforcement learning End-to-end Autonomous Parking method (REAP). REAP employs Soft Actor-Critic (SAC) within an asymmetric reinforcement learning framework to improve training efficiency and inference performance. To accelerate model convergence, we distill the capabilities of a rule-based planner into the end-to-end network through behavior cloning. We further introduce a soft predictive collision penalty mechanism to reduce collision rates by penalizing obstacle-approaching actions. To ensure that the trained reinforcement learning network can directly transfer to real-world scenarios, we have established a Real2Sim2Real simulator. In the Real2Sim step, we use 3D Gaussian Splatting (3DGS) to transform real-world scenes into digital scenes. In the Sim2Real step, we deploy the end-to-end model onto the vehicle to bridge the Sim2Real gap. Trained in the 3DGS simulator and deployed on physical vehicles, REAP successfully parks in various types of parking spaces, especially demonstrating the feasibility of end-to-end RL parking in extremely narrow mechanical slots.
    </details>
</div>
